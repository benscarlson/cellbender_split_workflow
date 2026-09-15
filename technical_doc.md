# Two-stage CellBender: implementation notes

Companion to `README.md`, which covers day-to-day use. This file covers how the
handover works, why the CPU stage needs the options it has, and what was tested.

Written against CellBender `0.3.2.post8+git.c5f5d9f4`. Line references are to
that revision.

## Files

| file | role |
|---|---|
| `templates/cellbender_gpu.sbatch` | stage 1, copied into the user's directory to edit |
| `templates/cellbender_cpu.sbatch` | stage 2, copied into the user's directory to edit |
| `submit.sh` | copies the templates out, and submits both jobs with a dependency |
| `cellbender_functions.sh` | `start_watcher` and `make_checkpoint_cpu_ready` |
| `cellbender_checkpoint_to_cpu.py` | the checkpoint conversion itself |

The two `.sbatch` files are meant to be edited and are deliberately plain: normal
`#SBATCH` directives, a few variables, and a `cellbender remove-background`
command you could copy out and run by hand. Everything with moving parts lives in
`cellbender_functions.sh`.

## Where things live

The workflow directory is read-only in practice: `submit.sh --init` copies the
two templates into whatever directory the user is standing in, and everything
after that — edited job scripts, data, `results/`, checkpoints, Slurm `.out`
files — stays in that working directory. Nothing is written back into the clone,
so it stays clean and `git pull` never conflicts.

Three things make that work:

- **`submit.sh` resolves its own location** with
  `readlink -f "$0"`, so it can be invoked by absolute path from anywhere, or put
  on `PATH`.
- **It submits from the current directory**, so Slurm sets each job's working
  directory to the user's folder. That is why the paths in the job scripts can be
  relative (`results/tiny.h5`) and need no editing for the common case.
- **It passes its own location to the jobs** as `CELLBENDER_SPLIT_DIR` via
  `sbatch --export`, and the job scripts source the helper file from there. This
  is the reason nothing the user edits contains a path back to the clone.

Because of that last point the job scripts cannot be submitted with a bare
`sbatch cellbender_cpu.sbatch` — `CELLBENDER_SPLIT_DIR` would be unset and the
`${VAR:?}` expansion stops the job with a message pointing at `submit.sh`. Use
`submit.sh --cpu-only` to resubmit stage 2 on its own.

### Why this is not a conda package

Installing the workflow into the CellBender conda environment was considered and
rejected. `submit.sh` does not need CellBender — it only calls `sbatch` — so
putting it in the environment's `bin/` would force a `ml miniconda; conda
activate` before a user could submit anything. The workflow is also independent
of CellBender's version, and a user with several CellBender environments would
have to install it into each one. Resolving the path at submission time costs
nothing and avoids all of that.

## How the two jobs are chained

`submit.sh` submits the GPU job, then submits the CPU job with
`--dependency=afterany:<gpu jobid>`.

`afterany` rather than `afterok` is deliberate. The GPU job ends by cancelling
itself, and `afterok` would refuse to release a dependant after a cancellation,
leaving the CPU job pending forever with `DependencyNeverSatisfied`.

The dependency also serves as the interlock: the CPU job cannot start reading the
checkpoint while the GPU job is still being torn down.

## `start_watcher <output.h5> <ckpt.tar.gz>`

Backgrounds a subshell that polls CellBender's log every `WATCHER_POLL_SECONDS`
(default 5) and cancels the job when inference is finished. It waits for two
conditions together:

- the log contains `cellbender:remove-background: Inference procedure complete.`
- the checkpoint tarball exists, is non-empty, and has no `.tmp` beside it

The ordering is guaranteed by CellBender itself. `train.py:224-236` saves a
checkpoint when `epoch == epochs`, and `make_tarball` (`checkpoint.py:319-330`)
writes `<name>.tmp` and then `os.replace`s it, so a partially written tarball is
never visible under the real name. The log line is emitted afterwards, at
`run.py:854`. Checking both is belt and braces rather than strictly necessary.

Two other things it does:

- It deletes the log before starting, so a marker left by a previous run cannot
  trigger it. (CellBender opens the log with `mode='w'` at `cli.py:218` and would
  truncate it anyway, but not until it gets that far.)
- It copies the log to `<output>.gpu.log` before cancelling, because the CPU job
  writes to the same filename and would otherwise destroy the training history.

If the log already says `Completed remove-background.` the watcher exits without
cancelling anything — CellBender got through the whole workflow on the GPU by
itself, which happens on small inputs. The CPU job still runs, resumes from the
checkpoint, and redoes the tail cheaply.

### Why it cancels the job rather than signalling CellBender

CellBender spawns a tree of subprocesses, so picking the right one to signal is
awkward. Cancelling the job lets Slurm tear the whole thing down.

Signalling would not have worked anyway. A background job started in a
non-interactive shell inherits `SIG_IGN` for `SIGINT`, and Python preserves that
disposition rather than installing its own handler, so `kill -INT` against the
CellBender process is silently ignored. This was observed directly during
development: the signal was sent, logged, and had no effect.

## `make_checkpoint_cpu_ready <gpu ckpt> <cpu ckpt>`

Runs `cellbender_checkpoint_to_cpu.py`, and `exit 1`s the job if it fails. The
guard matters: without a usable checkpoint CellBender does not fail, it retrains
from scratch, which looks like a working job until you read the log.

The script unpacks the tarball, loads the three pickled objects that carry device
state, flips two attributes on each, and repacks. It writes a **new** tarball
rather than editing in place, so the original stays resumable on a GPU.

Re-running is safe. If the destination already exists it is used as the input, so
a `posterior.h5` that a previous CPU attempt stored in it survives and does not
have to be recomputed; and if nothing needed changing, the tarball is left
untouched.

## Why the CPU stage needs `--force-use-checkpoint` and a converted checkpoint

Two separate problems, both of which fail quietly rather than loudly.

### 1. The workflow hash covers `--cuda`

Before looking for a checkpoint, CellBender hashes its own source plus the parsed
arguments, and only accepts a checkpoint whose hash matches. `run.py:64-87` lists
the arguments exempted from that hash. `use_cuda` is not among them, so a CPU run
can never match a checkpoint written by a `--cuda` run.

The failure is silent: `attempt_load_checkpoint` catches the resulting
`ValueError` (`checkpoint.py:314-316`), logs `No checkpoint loaded.`, and
CellBender starts training over from scratch on the CPU.

`--force-use-checkpoint` makes `load_from_checkpoint` accept the tarball
regardless of hash — it globs for `*_model.torch` instead of looking for the
current run's hash (`checkpoint.py:212-218`).

An alternative would have been to rename the files inside the tarball to the CPU
run's hash, which would remove the need for the flag. That means reproducing
`create_workflow_hashcode` and its exemption list outside CellBender, and keeping
them in sync. Not worth it.

### 2. The pickled model remembers its device

`save_checkpoint` pickles the model and dataloader objects whole, including their
`use_cuda` and `device` attributes. `load_from_checkpoint` maps the *tensors* to
whatever device the new run asks for (`checkpoint.py:196-198`), but leaves those
attributes alone, and nothing in `run_inference` resets them.

That matters because `Posterior` takes its device from the model rather than from
`args`:

```python
self.use_cuda = torch.cuda.is_available() if vi_model is None else vi_model.use_cuda
self.device = "cuda" if self.use_cuda else "cpu"        # posterior.py:215-216
```

So the posterior step would head for a GPU the CPU node does not have. The
conversion sets `use_cuda = False` and `device = "cpu"` on the model and on both
pickled dataloaders. The dataloaders do not matter while training is already
finished, but they do if a stage-1 job died mid-training and stage 2 has to
finish it.

### 3. Why the two scripts must agree on the CellBender options

On resume, `run_training` computes
`start_epoch = model.loss['train']['epoch'][-1] + 1` (`train.py:155-157`). If
`--epochs` matches between the two stages, that lands past `epochs`, the loop body
never runs, and CellBender goes straight to the posterior — which is exactly the
intended behaviour. If it does not match, stage 2 will train some more on the CPU.

### A caveat that comes with `--force-use-checkpoint`

It tells CellBender not to check that the checkpoint belongs to the input file.
Nothing will warn you if they do not match. The example scripts keep this safe by
naming both checkpoints after the output file, so samples cannot collide.

## Testing

### Does splitting the run change the results?

Starting from a single checkpoint and running stage 2 two ways, changing nothing
but the hardware:

| stage 2 run on | vs. an ordinary single-job GPU run |
|---|---|
| GPU, from the untouched original checkpoint | identical, every metric |
| CPU, from the converted checkpoint | counts removed differ by 0.1% |

In the CPU case, training is bit-identical (same ELBO to four decimals), cell
calls are identical (654 cells, 1346 empties, same convergence indicator), and
the difference is confined to the noise-count estimates — 341,843 counts removed
against 342,172, about 0.1%. That is ordinary CPU-versus-GPU floating point in
the posterior and estimation step, not an artefact of splitting the job.

The GPU row is the useful control: it isolates the hardware as the only variable,
and it confirms the original checkpoint really is left untouched and still
resumable on a GPU.

### Is the conversion lossless?

Checked directly by unpacking both tarballs and comparing:

- every tensor in the model's `state_dict` compares bitwise equal
- every file in the tarball the conversion does not touch is byte-identical
  (`_args.npy`, `_optim.pyro`, `_optim.torch`, `_params.pyro`, `_random.pyro`,
  `_random.cuda`)
- the only differences are the two attributes it is meant to change, and the
  `posterior.h5` that the CPU run adds afterwards

### Other paths exercised

- Full `submit.sh` run on `tiny_raw_feature_bc_matrix.h5ad`: watcher fired,
  GPU job `CANCELLED`, dependency released the CPU job, CPU job `COMPLETED`.
- CPU stage rerun with the converted checkpoint already present: conversion
  reported "already CPU-ready", and CellBender logged `Loaded pre-computed
  posterior from posterior.h5` instead of recomputing it.
- CPU stage rerun from the same checkpoint twice: byte-identical metrics, so the
  stage is deterministic.
- `source ... || exit 1` guard: verified it catches both a syntax error in the
  helper file (returns 2) and a missing helper file (returns 1). This was added
  after a real failure — an apostrophe inside `${1:?...}` broke the parse, the
  function was never defined, and the CPU job went on to retrain from scratch.

### Timings

`heart10k_raw_feature_bc_matrix.h5`, `--expected-cells 5000
--total-droplets-included 15000`, 150 epochs, on McCleary:

| | where | time | ended as |
|---|---|---|---|
| GPU job | `gpu`, RTX 5000, 4 cpu, 48G | 13m45s | `CANCELLED` |
| CPU job | `day`, 8 cpu, 96G, 13.7G peak RSS | 21m47s | `COMPLETED` |

Nearly 22 minutes of CPU-bound work came off the GPU. The checkpoints were 23 MB
each on the tiny dataset; they scale with the number of droplets analysed, since
the pickled dataloaders carry the trimmed count matrix.

## Known limits

- `--epochs 0` never logs `Inference procedure complete.` (`run.py:854` is on the
  training branch only), so the watcher will never fire. The run simply finishes
  on the GPU. Harmless, but there is no point using this workflow for it.
- If training trips CellBender's ELBO checks it restarts from scratch within
  stage 1 (`run.py:835-849`). That is intended — retries stay on the GPU and the
  handover waits for the attempt that succeeds.
- Between the watcher firing and Slurm killing the job, CellBender keeps working
  on the posterior for a few seconds. That work is discarded and stage 2 redoes
  it.
- `submit.sh` must stay in the same directory as `cellbender_functions.sh` and
  `templates/`, since it locates them relative to itself. Symlinking `submit.sh`
  onto a `PATH` directory is fine — `readlink -f` follows the symlink to the real
  file — but copying it out on its own is not.

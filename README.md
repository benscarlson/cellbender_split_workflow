# Running CellBender in two stages

CellBender starts by doing inference, which needs a GPU. Everything after that —
the posterior, the denoised counts, the plots and the report — does not, and on
a real dataset it takes longer than the inference does. Run the whole thing on a
GPU node and you hold the GPU for a long time without using it.

This splits the run in two. The first job does the inference on a GPU and stops
as soon as that part is finished. The second job picks up where it left off and
finishes on an ordinary CPU node.

## What to do

1. Open `cellbender_gpu.sbatch`. Set the paths at the top, set your conda
   environment name, and put your usual CellBender options at the bottom.

2. Open `cellbender_cpu.sbatch` and do the same. **`INPUT`, `OUTPUT`, `CKPT` and
   the CellBender options must match the GPU script.** The second job is
   continuing the first job's work, so it has to be the same run.

3. Change the `#SBATCH` lines in either script to suit your data. They are
   normal Slurm scripts.

4. Submit both:

   ```bash
   ./submit.sh
   ```

That is all. Slurm holds the second job until the first one finishes.

## The GPU job will say CANCELLED. That is normal.

The GPU job ends itself on purpose, the moment inference is done. `CANCELLED` is
what a successful first stage looks like here. As long as the CPU job then says
`COMPLETED`, your run worked.

Check on them with:

```bash
squeue -u $USER
```

## What you get

The usual CellBender outputs, in the same folder as `OUTPUT`:

```
sample1.h5                 denoised counts
sample1_filtered.h5        cells only
sample1_cell_barcodes.csv
sample1_metrics.csv
sample1.pdf
sample1_report.html
sample1_posterior.h5
sample1.log                the CPU job's log
sample1.gpu.log            the GPU job's log, with the training history
sample1_ckpt.tar.gz        checkpoints -- safe to delete when you are happy
sample1_ckpt_cpu.tar.gz      with the results
```

The two `cellbender_*_<number>.out` files are the Slurm logs, written wherever
you ran `./submit.sh` from.

One thing to be aware of: because the second half now runs on a CPU, the numbers
come out very slightly different from running everything on a GPU — around 0.1%
on the counts removed, with the same cells called. This is normal and is
explained in `technical_doc.md`.

## If something goes wrong

**The GPU job ran out of time.** Nothing is lost — CellBender saves its progress
every few minutes. Run `./submit.sh` again and it carries on from where it got
to.

**The CPU job failed.** The GPU work is safe. Fix whatever went wrong and
resubmit just that job:

```bash
sbatch cellbender_cpu.sbatch
```

**You want to start again from scratch.** Delete both checkpoint files
(`..._ckpt.tar.gz` and `..._ckpt_cpu.tar.gz`) first, or CellBender will carry on
from them instead of starting over.

**You are running several samples.** Give each one its own `OUTPUT`, `CKPT` and
`CKPT_CPU` paths. The example scripts name the checkpoints after the output file,
which keeps samples from overwriting each other's work.

## More detail

`technical_doc.md` covers how the handover works, why the CPU job needs the
extra options it has, and what was tested.

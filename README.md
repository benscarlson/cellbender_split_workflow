# Running CellBender in two stages

[CellBender](https://github.com/broadinstitute/CellBender) starts by doing inference, which is best run using a GPU. Everything after that uses only the CPU, and often takes longer than the inference stage. Thus running the entire analysis on a GPU node wastes a lot of GPU cycles and risks being terminated due to inefficient use.

This workflow splits the Cellbender analysis in two. The first job performs inference on a GPU and stops as soon as that part is finished. The second job picks up where it left off and finishes on an ordinary CPU node.

## Overview of the workflow

This section provides a summay of the workflow. See below for a full example. Also see the section to first verify the cellbender version.

The workflow has batch scripts for the gpu and cpu jobs, plus some bash and slurm functionality to run the workflow. A bash function watches the gpu job and cancels it once inference is complete. Slurm automatically starts the cpu job after the gpu job is complete. Running the workflow can be accomplished in three easy steps.

You work in your own directory — whatever folder holds your data.

1. **Copy the job scripts into your working directory.** From the folder with
   your data in it:

   ```bash
   /path/to/cellbender_split_workflow/submit.sh --init
   ```

   That puts `cellbender_gpu.sbatch`, `cellbender_cpu.sbatch` and an empty
   `results/` folder in the current directory.

2. **Edit the two job scripts.** In each one, set the paths at the top, set your
   conda environment name, and put your CellBender options on the `cellbender`
   command at the bottom. **`INPUT`, `OUTPUT`, `CKPT` and the CellBender options
   must match between the two scripts** — the second job is continuing the first
   job's work, so it has to be the same run. Change the `#SBATCH` lines to suit
   your data; they are normal Slurm scripts.

3. **Submit both jobs:**

   ```bash
   /path/to/cellbender_split_workflow/submit.sh
   ```

If you will be using this often, add the workflow directory to your `PATH` so
you can just type `submit.sh`:

```bash
echo 'export PATH="$PATH:/path/to/cellbender_split_workflow"' >> ~/.bashrc
```

## The GPU job will say CANCELLED. That is normal.

The GPU job ends itself on purpose, the moment inference is done. `CANCELLED` is
what a successful first stage looks like here. As long as the CPU job then says
`COMPLETED`, your run worked.

Check on them with:

```bash
squeue -u $USER
```

## Example: the CellBender quick-start dataset

This walks through the small demo dataset from the
[CellBender tutorial](https://cellbender.readthedocs.io/en/latest/tutorial/),
split across two jobs. It assumes you already have a working CellBender conda
environment called `cellbender`.

### 1. Get the workflow

```bash
cd ~/palmer_scratch

#Clone the repo with the split workflow
git clone https://github.com/ycrc/cellbender_split_workflow.git

#Make a working directory
mkdir -p ~/palmer_scratch/cellbender_demo
cd ~/palmer_scratch/cellbender_demo
```

Activate the environment

```bash
ml reset
ml miniconda
conda activate cellbender
```

Check your CellBender version

The whole workflow is built on CellBender's checkpoint file. In releases before
0.4.0, saving a checkpoint fails outright with `cannot pickle 'weakref' object`,
so there is nothing for the second job to pick up. Check your environment once,
before anything else:

```bash

~/palmer_scratch/cellbender_split_workflow/submit.sh --check
```

It should say `checkpointing: OK`
```
cellbender 0.4.0
  from /path/to/cellbender/remove_background/checkpoint.py
  checkpointing: OK
```

If it says `BROKEN`, you need a newer CellBender. Note that `cellbender
--version` on its own is not a reliable check — a development install can report
an old version string even when the code is current, which is why this looks at
the code instead.

(The same bug means `--checkpoint-mins` does nothing on those versions, so an
ordinary single-job CellBender run there has no crash protection either.)

### 2. Make the demo dataset

CellBender ships a script that downloads the 10x `heart10k` dataset and trims it
down to something small. Run the script from the `cellbender_demo` directory so that the results will be saved there.

```bash

python /path/to/CellBender/examples/remove_background/generate_tiny_10x_dataset.py
```

It downloads about 170 MB and writes `tiny_raw_feature_bc_matrix.h5ad`.

### 3. Initialize the workflow

This copies in the job script templates, submit script, and creates a folder for results.

```bash
~/palmer_scratch/cellbender_split_workflow/submit.sh --init
```

You should now have:

```
~/palmer_scratch/cellbender_demo/
    tiny_raw_feature_bc_matrix.h5ad
    heart10k_raw_feature_bc_matrix.h5
    cellbender_gpu.sbatch           <- edit this for a real run with your data
    cellbender_cpu.sbatch           <- edit this for a real run with your data
    results/
```

### 4. Check the job scripts

With your own data, you would edit cellbender_gpu.sbatch and cellbender_cpu.sbatch. For this demo, the scripts come set up to work out of the box, so there is nothing to change unless your conda environment is named something other than `cellbender`.

`cellbender_gpu.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=cellbender-gpu
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=5G
#SBATCH --time=00:30:00
#SBATCH --output=cellbender_gpu_%j.out

INPUT=tiny_raw_feature_bc_matrix.h5ad
OUTPUT=results/tiny.h5
CKPT=results/tiny_ckpt.tar.gz

ml reset
ml miniconda
conda activate cellbender

source "${CELLBENDER_SPLIT_DIR:?submit this job with submit.sh}/cellbender_functions.sh" || exit 1

start_watcher "$OUTPUT" "$CKPT"

cellbender remove-background \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --checkpoint "$CKPT" \
    --cuda \
    --expected-cells 500 \
    --total-droplets-included 2000
```

`cellbender_cpu.sbatch` 

This script contains the same paths and CellBender options, plus
`CKPT_CPU`. No GPU, so no `--gres` line and no `--cuda`:

```bash
#!/bin/bash
#SBATCH --job-name=cellbender-cpu
#SBATCH --partition=day
#SBATCH --cpus-per-task=2
#SBATCH --mem=5G
#SBATCH --time=00:30:00
#SBATCH --output=cellbender_cpu_%j.out

INPUT=tiny_raw_feature_bc_matrix.h5ad
OUTPUT=results/tiny.h5
CKPT=results/tiny_ckpt.tar.gz
CKPT_CPU=results/tiny_ckpt_cpu.tar.gz

ml reset
ml miniconda
conda activate cellbender

source "${CELLBENDER_SPLIT_DIR:?submit this job with submit.sh}/cellbender_functions.sh" || exit 1

make_checkpoint_cpu_ready "$CKPT" "$CKPT_CPU"

cellbender remove-background \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --checkpoint "$CKPT_CPU" \
    --force-use-checkpoint \
    --cpu-threads "$SLURM_CPUS_PER_TASK" \
    --expected-cells 500 \
    --total-droplets-included 2000
```

### 5. Submit

```bash
~/palmer_scratch/cellbender_split_workflow/submit.sh
```

```
GPU job: 10557826
CPU job: 10557827  (waits for 10557826 to end)
```

The GPU job runs 150 epochs in under a minute, then cancels itself. The CPU job
starts as soon as it does and takes another minute. When both are gone from
`squeue`, the results are in `results/` — and `sacct` will show the GPU job as
`CANCELLED` and the CPU job as `COMPLETED`, which is correct.

## What you get

The usual CellBender outputs, in the same folder as `OUTPUT`:

```
tiny.h5                 denoised counts
tiny_filtered.h5        cells only
tiny_cell_barcodes.csv
tiny_metrics.csv
tiny.pdf
tiny_report.html
tiny_posterior.h5
tiny.log                the CPU job's log
tiny.gpu.log            the GPU job's log, with the training history
tiny_ckpt.tar.gz        checkpoints -- safe to delete when you are happy
tiny_ckpt_cpu.tar.gz      with the results
```

The two `cellbender_*_<number>.out` files are the Slurm logs, written in the
directory you ran `submit.sh` from. You may also find a stray `posterior.h5`
there — CellBender writes that copy itself, and it is safe to delete.

## If something goes wrong

**Both jobs failed, and the log says `Could not save checkpoint`.** Your
CellBender is older than 0.4.0. See the section for checking your version.

**The GPU job ran out of time.** Nothing is lost — CellBender saves its progress
every few minutes. Run `submit.sh` again and it carries on from where it got to.

**The CPU job failed.** The GPU work is safe. Fix whatever went wrong and
resubmit just that job:

```bash
/path/to/cellbender_split_workflow/submit.sh --cpu-only
```

**You want to start again from scratch.** Delete both checkpoint files
(`..._ckpt.tar.gz` and `..._ckpt_cpu.tar.gz`) first, or CellBender will carry on
from them instead of starting over.

**You are running several samples.** Give each sample its own working directory.
That keeps its job scripts, checkpoints and results separate from every other
sample's.

## More detail

`technical_doc.md` covers how the handover works, why the CPU job needs the
extra options it has, and what was tested.

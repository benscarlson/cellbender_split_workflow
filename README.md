# Running CellBender in two stages

[CellBender](https://github.com/broadinstitute/CellBender) starts by doing inference, which is best run using a GPU. Everything after that uses only the CPU, and often takes longer than the inference stage. Thus running the entire analysis on a GPU node wastes a lot of GPU cycles and risks being terminated due to inefficient use.

This workflow splits the Cellbender analysis in two. The first job performs inference on a GPU and stops as soon as that part is finished. The second job picks up where it left off and finishes on an ordinary CPU node.

## Overview of the workflow

The workflow has batch scripts for the gpu and cpu jobs, plus some bash and slurm functionality to run the workflow. A bash function watches the gpu job and cancels it once inference is complete. Slurm automatically starts the cpu job after the cpu job is complete. Running the workflow can be accomplished in three easy steps.

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

## Example: the CellBender quick-start dataset

This walks through the small demo dataset from the
[CellBender tutorial](https://cellbender.readthedocs.io/en/latest/tutorial/),
split across two jobs. It assumes you already have a working CellBender conda
environment. The whole thing takes a couple of minutes.

### 1. Get the scripts

```bash
mkdir -p ~/palmer_scratch/cellbender_demo
cd ~/palmer_scratch/cellbender_demo
git clone https://github.com/ycrc/cellbender_split_workflow.git
```

### 2. Make the demo dataset

CellBender ships a script that downloads the 10x `heart10k` dataset and trims it
down to something small. Run it from the `examples/remove_background/` folder of
your CellBender checkout:

```bash
ml reset
ml miniconda
conda activate cellbender-main

cd /path/to/CellBender/examples/remove_background
python generate_tiny_10x_dataset.py
```

It downloads about 170 MB and writes `tiny_raw_feature_bc_matrix.h5ad`. Copy
that into the demo folder and make somewhere for the results:

```bash
cd ~/palmer_scratch/cellbender_demo
cp /path/to/CellBender/examples/remove_background/tiny_raw_feature_bc_matrix.h5ad .
mkdir results
```

You should now have:

```
~/palmer_scratch/cellbender_demo/
    tiny_raw_feature_bc_matrix.h5ad
    results/
    cellbender_split_workflow/      <- the scripts you cloned
```

### 3. Edit `cellbender_gpu.sbatch`

Set the paths, set your conda environment name, and put the tutorial's options
on the `cellbender` command. The demo is small enough for the short-queue
partitions, so `gpu_devel` and half an hour are plenty.

```bash
#!/bin/bash
#SBATCH --job-name=cellbender-gpu
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=5G
#SBATCH --time=00:30:00
#SBATCH --output=cellbender_gpu_%j.out

INPUT=$HOME/palmer_scratch/cellbender_demo/tiny_raw_feature_bc_matrix.h5ad
OUTPUT=$HOME/palmer_scratch/cellbender_demo/results/tiny.h5
CKPT=$HOME/palmer_scratch/cellbender_demo/results/tiny_ckpt.tar.gz

ml reset
ml miniconda
conda activate cellbender-main

# This demo dataset is so small that the watcher needs to look more often than
# usual to catch the handover. Leave this out on real data.
WATCHER_POLL_SECONDS=2
source "${SLURM_SUBMIT_DIR:-.}/cellbender_functions.sh" || exit 1

start_watcher "$OUTPUT" "$CKPT"

cellbender remove-background \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --checkpoint "$CKPT" \
    --cuda \
    --expected-cells 500 \
    --total-droplets-included 2000
```

### 4. Edit `cellbender_cpu.sbatch`

The same paths and the same CellBender options, plus `CKPT_CPU`. No GPU, so no
`--gres` line and no `--cuda`.

```bash
#!/bin/bash
#SBATCH --job-name=cellbender-cpu
#SBATCH --partition=devel
#SBATCH --cpus-per-task=2
#SBATCH --mem=5G
#SBATCH --time=00:30:00
#SBATCH --output=cellbender_cpu_%j.out

INPUT=$HOME/palmer_scratch/cellbender_demo/tiny_raw_feature_bc_matrix.h5ad
OUTPUT=$HOME/palmer_scratch/cellbender_demo/results/tiny.h5
CKPT=$HOME/palmer_scratch/cellbender_demo/results/tiny_ckpt.tar.gz
CKPT_CPU=$HOME/palmer_scratch/cellbender_demo/results/tiny_ckpt_cpu.tar.gz

ml reset
ml miniconda
conda activate cellbender-main

source "${SLURM_SUBMIT_DIR:-.}/cellbender_functions.sh" || exit 1

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
cd ~/palmer_scratch/cellbender_demo/cellbender_split_workflow
./submit.sh
```

```
GPU job: 10557826
CPU job: 10557827  (waits for 10557826 to end)
```

The GPU job runs 150 epochs in under a minute, then cancels itself. The CPU job
starts as soon as it does and takes another half minute. When both are gone from
`squeue`, the results are in `results/` — and `sacct` will show the GPU job as
`CANCELLED` and the CPU job as `COMPLETED`, which is correct.

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

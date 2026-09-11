#!/bin/bash
#SBATCH --job-name=cellbender-cpu
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=cellbender_cpu_%j.out

# Stage 2 of 2: everything after inference -- posterior, denoised counts,
# plots, metrics and the HTML report -- picking up from the checkpoint the GPU
# job wrote. No GPU needed.

# ---- edit these, and keep the first three the same as in cellbender_gpu.sbatch --
INPUT=/path/to/raw_feature_bc_matrix.h5
OUTPUT=/path/to/results/sample1.h5
CKPT=/path/to/results/sample1_ckpt.tar.gz
CKPT_CPU=/path/to/results/sample1_ckpt_cpu.tar.gz
# --------------------------------------------------------------------------------

ml reset
ml miniconda
conda activate cellbender-main

# Extra helper commands for this workflow. If you move this script somewhere
# else, put the full path to cellbender_functions.sh here.
source "${SLURM_SUBMIT_DIR:-.}/cellbender_functions.sh" || exit 1

# The GPU job's checkpoint remembers that it was made on a GPU, so make a copy
# of it that a CPU-only run can use. The original is left untouched.
make_checkpoint_cpu_ready "$CKPT" "$CKPT_CPU"

# The same options as cellbender_gpu.sbatch, with three differences:
#   - no --cuda
#   - --force-use-checkpoint, so it accepts the GPU job's checkpoint
#   - --cpu-threads, so CellBender uses the cores Slurm gave us and no more
# The options that appear in both scripts, like --expected-cells, have to match.
cellbender remove-background \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --checkpoint "$CKPT_CPU" \
    --force-use-checkpoint \
    --cpu-threads "$SLURM_CPUS_PER_TASK" \
    --expected-cells 5000 \
    --total-droplets-included 20000

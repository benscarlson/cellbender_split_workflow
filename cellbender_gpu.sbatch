#!/bin/bash
#SBATCH --job-name=cellbender-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=cellbender_gpu_%j.out

# Stage 1 of 2: run CellBender's inference step on a GPU.
#
# This job ends itself as soon as inference is done, and the CPU job does the
# rest. Seeing this job finish as CANCELLED is normal and means it worked.

# ---- edit these three paths, and keep them the same in cellbender_cpu.sbatch --
INPUT=/path/to/raw_feature_bc_matrix.h5
OUTPUT=/path/to/results/sample1.h5
CKPT=/path/to/results/sample1_ckpt.tar.gz
# ------------------------------------------------------------------------------

ml reset
ml miniconda
conda activate cellbender

# Extra helper commands for this workflow. If you move this script somewhere
# else, put the full path to cellbender_functions.sh here.
source "${SLURM_SUBMIT_DIR:-.}/cellbender_functions.sh" || exit 1

# Watch the log and end this job once inference is finished, so the GPU is not
# tied up during the CPU-bound part of the run.
start_watcher "$OUTPUT" "$CKPT"

cellbender remove-background \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --checkpoint "$CKPT" \
    --cuda \
    --expected-cells 5000 \
    --total-droplets-included 20000

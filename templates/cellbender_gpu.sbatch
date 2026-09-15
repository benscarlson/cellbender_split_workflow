#!/bin/bash
#SBATCH --job-name=cellbender-gpu
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=5G
#SBATCH --time=00:30:00
#SBATCH --output=cellbender_gpu_%j.out

# Stage 1 of 2: CellBender's inference step, on a GPU.
#
# This job ends itself as soon as inference is done, and the CPU job does the
# rest. Seeing this job finish as CANCELLED is normal and means it worked.
#
# The resources above are the smallest that work, and are enough for the
# quick-start demo dataset. For real data you will want --partition=gpu, a
# longer --time, and more --mem.

# ---- paths, relative to the directory you ran submit.sh from ----------------
INPUT=tiny_raw_feature_bc_matrix.h5ad
OUTPUT=results/tiny.h5
CKPT=results/tiny_ckpt.tar.gz
# -----------------------------------------------------------------------------

ml reset
ml miniconda
conda activate cellbender

# submit.sh tells the job where the workflow scripts live.
source "${CELLBENDER_SPLIT_DIR:?submit this job with submit.sh}/cellbender_functions.sh" || exit 1

# Watch the log and end this job once inference is finished, so the GPU is not
# tied up during the CPU-bound part of the run.
start_watcher "$OUTPUT" "$CKPT"

cellbender remove-background \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --checkpoint "$CKPT" \
    --cuda \
    --expected-cells 500 \
    --total-droplets-included 2000

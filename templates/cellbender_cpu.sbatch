#!/bin/bash
#SBATCH --job-name=cellbender-cpu
#SBATCH --partition=devel
#SBATCH --cpus-per-task=2
#SBATCH --mem=5G
#SBATCH --time=00:30:00
#SBATCH --output=cellbender_cpu_%j.out

# Stage 2 of 2: everything after inference -- posterior, denoised counts,
# plots, metrics and the HTML report -- picking up from the checkpoint the GPU
# job wrote. No GPU needed.
#
# The resources above are the smallest that work, and are enough for the
# quick-start demo dataset. For real data you will want --partition=day, a
# longer --time, and more --mem and --cpus-per-task.

# ---- paths, relative to the directory you ran submit.sh from ----------------
#      the first three must match cellbender_gpu.sbatch
INPUT=tiny_raw_feature_bc_matrix.h5ad
OUTPUT=results/tiny.h5
CKPT=results/tiny_ckpt.tar.gz
CKPT_CPU=results/tiny_ckpt_cpu.tar.gz
# -----------------------------------------------------------------------------

ml reset
ml miniconda
conda activate cellbender

# submit.sh tells the job where the workflow scripts live.
source "${CELLBENDER_SPLIT_DIR:?submit this job with submit.sh}/cellbender_functions.sh" || exit 1

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
    --expected-cells 500 \
    --total-droplets-included 2000

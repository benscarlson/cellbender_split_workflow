#!/bin/bash
# Submit both CellBender jobs. Slurm holds the CPU job until the GPU job ends.
#
#   ./submit.sh

set -euo pipefail
cd "$(dirname "$0")"

gpu_id=$(sbatch --parsable cellbender_gpu.sbatch)

# afterany, not afterok: the GPU job ends by cancelling itself once inference is
# done, and afterok would refuse to release the CPU job after that.
cpu_id=$(sbatch --parsable --dependency=afterany:"$gpu_id" cellbender_cpu.sbatch)

echo "GPU job: $gpu_id"
echo "CPU job: $cpu_id  (waits for $gpu_id to end)"
echo
echo "Watch them with:  squeue -u $USER"

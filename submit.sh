#!/bin/bash
# Submit the two CellBender jobs.
#
#   submit.sh --init       copy the job scripts into the current directory
#   submit.sh              submit both jobs
#   submit.sh --cpu-only   submit just the CPU job (to retry stage 2)
#
# Run this from your own working directory -- wherever your data is and where
# you want the results. It can be called by its full path from anywhere, and it
# never writes anything into the workflow directory itself.

set -euo pipefail

WORKFLOW_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
GPU_SCRIPT=cellbender_gpu.sbatch
CPU_SCRIPT=cellbender_cpu.sbatch

usage() {
    sed -n '2,10p' "$(readlink -f "$0")" | sed 's/^# \?//'
}

mode=both
case "${1:-}" in
    --init)     mode=init ;;
    --cpu-only) mode=cpu ;;
    -h|--help)  usage; exit 0 ;;
    "")         ;;
    *)          echo "error: unknown option '$1'" >&2; usage >&2; exit 2 ;;
esac

if [[ $mode == init ]]; then
    for script in "$GPU_SCRIPT" "$CPU_SCRIPT"; do
        if [[ -e $script ]]; then
            echo "error: $script is already here -- move it aside if you want a fresh copy" >&2
            exit 1
        fi
    done
    cp "$WORKFLOW_DIR/templates/$GPU_SCRIPT" "$WORKFLOW_DIR/templates/$CPU_SCRIPT" .
    mkdir -p results
    cat <<EOF
Copied into $(pwd):
  $GPU_SCRIPT
  $CPU_SCRIPT
  results/

Edit the two job scripts, then run:
  $0
EOF
    exit 0
fi

for script in "$GPU_SCRIPT" "$CPU_SCRIPT"; do
    if [[ ! -f $script ]]; then
        echo "error: $script not found in $(pwd)" >&2
        echo "       run '$0 --init' to copy the job scripts here first" >&2
        exit 1
    fi
done

# The job scripts use this to find cellbender_functions.sh, so that nothing the
# user edits has to contain a path to the workflow directory.
export_vars="ALL,CELLBENDER_SPLIT_DIR=$WORKFLOW_DIR"

if [[ $mode == cpu ]]; then
    cpu_id=$(sbatch --parsable --export="$export_vars" "$CPU_SCRIPT")
    echo "CPU job: $cpu_id"
    exit 0
fi

gpu_id=$(sbatch --parsable --export="$export_vars" "$GPU_SCRIPT")

# afterany, not afterok: the GPU job ends by cancelling itself once inference is
# done, and afterok would refuse to release the CPU job after that.
cpu_id=$(sbatch --parsable --export="$export_vars" --dependency=afterany:"$gpu_id" "$CPU_SCRIPT")

echo "GPU job: $gpu_id"
echo "CPU job: $cpu_id  (waits for $gpu_id to end)"
echo
echo "Watch them with:  squeue -u $USER"

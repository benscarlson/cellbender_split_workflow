# Helper functions for the two-stage CellBender workflow.
#
# Source this near the top of your Slurm scripts:
#
#     source "${SLURM_SUBMIT_DIR:-.}/cellbender_functions.sh"
#
# It gives you two commands:
#
#   start_watcher <output.h5> <ckpt.tar.gz>
#       For the GPU script. Ends the job as soon as inference is finished, so
#       the GPU is not held during the CPU-bound second half of the run.
#
#   make_checkpoint_cpu_ready <gpu ckpt.tar.gz> <cpu ckpt.tar.gz>
#       For the CPU script. Writes a copy of the GPU job's checkpoint that a
#       CPU-only run can pick up. Run it just before the cellbender command.

CELLBENDER_FUNCTIONS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# How often the watcher looks at the log file, in seconds.
: "${WATCHER_POLL_SECONDS:=15}"


# Watch CellBender's log in the background, and cancel this Slurm job once
# inference is done and the checkpoint has been written.
#
# The CPU job is already queued behind this one (submit.sh sets that up), so
# cancelling here is simply how the run moves on to the next stage.
start_watcher() {
    local output="${1:?start_watcher: pass the --output path}"
    local ckpt="${2:?start_watcher: pass the --checkpoint path}"
    local log="${output%.h5}.log"

    # CellBender starts its log fresh each run, but clear it here too so the
    # watcher can never react to a message left behind by an earlier run.
    rm -f "$log"

    (
        while true; do
            sleep "$WATCHER_POLL_SECONDS"
            [[ -s "$log" ]] || continue

            # CellBender got all the way to the end by itself. Nothing to do.
            if grep -qF 'Completed remove-background.' "$log"; then
                echo "watcher: CellBender finished the whole run here; leaving the job alone"
                exit 0
            fi

            # Inference is done once this line appears. CellBender saves the
            # final checkpoint just before printing it, so wait for the tarball
            # as well. The .tmp check means we never look at a half-written one.
            grep -qF 'Inference procedure complete.' "$log" || continue
            [[ -s "$ckpt" && ! -e "$ckpt.tmp" ]] || continue

            # The CPU job writes to the same log file, so keep a copy of this
            # one -- it has the training and ELBO history in it.
            cp -p "$log" "${output%.h5}.gpu.log"

            echo "watcher: inference finished, checkpoint written to $ckpt"
            echo "watcher: ending this job so the CPU job can take over"
            echo "watcher: (this job showing as CANCELLED is the normal, successful outcome)"
            scancel "$SLURM_JOB_ID"
            exit 0
        done
    ) &

    echo "watcher: started, checking $log every ${WATCHER_POLL_SECONDS}s"
}


# Write a copy of the GPU job's checkpoint that a CPU-only CellBender run can
# use. The original is left alone, so it can still be resumed on a GPU.
# See README.md for why this is needed at all.
make_checkpoint_cpu_ready() {
    local gpu_ckpt="${1:?make_checkpoint_cpu_ready: pass the GPU checkpoint}"
    local cpu_ckpt="${2:?make_checkpoint_cpu_ready: pass the checkpoint to write}"

    if ! python "$CELLBENDER_FUNCTIONS_DIR/cellbender_checkpoint_to_cpu.py" "$gpu_ckpt" "$cpu_ckpt"; then
        echo "ERROR: could not prepare the checkpoint for the CPU run -- stopping here." >&2
        echo "       Carrying on would look like it worked: CellBender would just" >&2
        echo "       quietly start training again from scratch." >&2
        exit 1
    fi
}

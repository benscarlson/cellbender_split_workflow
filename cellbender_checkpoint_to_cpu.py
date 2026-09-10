#!/usr/bin/env python
"""Write a CPU-ready copy of a CellBender checkpoint.

    cellbender_checkpoint_to_cpu.py SOURCE_CKPT DEST_CKPT

CellBender pickles its model and dataloader objects into the checkpoint tarball
whole, including the ``use_cuda`` / ``device`` attributes that were in force
when the checkpoint was written. Loading maps the *tensors* onto whatever device
the new run asks for, but leaves those attributes alone, and the posterior step
takes its device from the model rather than from the command line::

    self.use_cuda = torch.cuda.is_available() if vi_model is None else vi_model.use_cuda
    self.device = "cuda" if self.use_cuda else "cpu"        # posterior.py:215-216

So a checkpoint written by a ``--cuda`` run sends the posterior step looking for
a GPU that a CPU-only job does not have. This script flips those attributes and
writes the result to a new tarball, leaving the original untouched and still
usable for resuming on a GPU.

Re-running is safe and cheap: if DEST already exists it is used as the source,
so a posterior that a previous CPU run added to it is preserved, and if nothing
needs changing the tarball is left alone.
"""

import argparse
import glob
import os
import sys
import tempfile

import dill
import torch

from cellbender.remove_background.checkpoint import make_tarball, unpack_tarball

# Loaded and re-saved with pickle_module=dill, matching how CellBender itself
# reads these files back (checkpoint.py:233, 256, 262).
PATTERNS = ["*_model.torch", "*_train.loaderstate", "*_test.loaderstate"]


def move_file_to_cpu(path: str) -> bool:
    """Reset the cached device attributes on one pickled object. True if changed."""
    obj = torch.load(path, map_location="cpu", pickle_module=dill)
    if not getattr(obj, "use_cuda", False):
        return False
    obj.use_cuda = False
    obj.device = "cpu"
    torch.save(obj, path, pickle_module=dill)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="checkpoint written by the GPU run, e.g. sample1_ckpt.tar.gz")
    parser.add_argument("dest", help="CPU-ready checkpoint to write, e.g. sample1_ckpt_cpu.tar.gz")
    args = parser.parse_args()

    # Prefer an existing destination as the input, so anything a previous CPU
    # run added to it (notably posterior.h5) survives.
    source = args.dest if os.path.exists(args.dest) else args.source
    if not os.path.exists(source):
        print(f"ERROR: no checkpoint at {args.source}", file=sys.stderr)
        return 1
    print(f"reading checkpoint {source}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        if not unpack_tarball(tarball_name=source, directory=tmp_dir):
            print(f"ERROR: could not unpack {source}", file=sys.stderr)
            return 1

        changed = False
        for pattern in PATTERNS:
            for path in glob.glob(os.path.join(tmp_dir, pattern)):
                name = os.path.basename(path)
                if move_file_to_cpu(path):
                    print(f"  moved {name} onto the CPU")
                    changed = True
                else:
                    print(f"  {name} was already on the CPU")

        if not changed and source == args.dest:
            print(f"{args.dest} is already CPU-ready, leaving it as it is")
            return 0

        files = [os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir) if os.path.isfile(os.path.join(tmp_dir, f))]
        make_tarball(files=files, tarball_name=args.dest)

    size_mb = os.path.getsize(args.dest) / 1e6
    print(f"wrote {args.dest} ({size_mb:.0f} MB); {args.source} is unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3

from pathlib import Path

INPUT_DIR = Path("040-scan-pages")
OUTPUT_DIR = Path(Path(__file__).stem)


import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import psutil
from tqdm import tqdm

# import _shared.py from workdir or script_dir
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir))
sys.path.insert(0, os.getcwd())

from _shared import (
    load_config,
    remove_done_files,
)

config = load_config()


def process_file(file: Path):
    img = cv2.imread(str(file), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read image: {file}")

    out_file = OUTPUT_DIR / file.name

    if len(img.shape) == 2:
        # single-channel image
        # dont process this file
        # create hardlink
        out_file.hardlink_to(file)
        # this file was not processed
        return None

    # Convert only color images.
    if len(img.shape) == 3:
        if img.shape[2] == 3: # BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.shape[2] == 4: # BGRA
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    if not cv2.imwrite(str(out_file), img):
        raise RuntimeError(f"Failed to write image: {out_file}")

    # this file was processed
    return out_file


def main():
    files = sorted(INPUT_DIR.glob(f"*.{config.scan_format}"))

    if not files:
        print("No image files found in", INPUT_DIR)
        return

    # Only submit work for files that still need processing.
    files = remove_done_files(files, OUTPUT_DIR)

    if not files:
        print("nothing to do")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    num_workers = psutil.cpu_count(logical=False) or 1
    print(f"Using {num_workers} workers...")

    # By default, OpenCV uses multiple CPU threads.
    cv2.setNumThreads(1)

    if 0:
        # Debug: disable parallel processing.
        num_workers = 1

    tqdm_kwargs = dict(
        total=len(files),
        ncols=80,
        unit="page",
    )

    num_done = 0

    with (
        ProcessPoolExecutor(max_workers=num_workers) as executor,
        tqdm(**tqdm_kwargs) as pbar,
    ):
        futures = {
            executor.submit(process_file, fname): fname
            for fname in files
        }

        for future in as_completed(futures):
            fname = futures[future]
            try:
                out_file = future.result()
                if out_file:
                    num_done += 1
            except Exception as exc:
                print(f"Error processing {fname}: {exc}")
                raise
            finally:
                pbar.update(1)

    if num_done == 0:
        print("no files were converted")
        print("todo: remove output dir:")
        print(f"  rm -rf {str(OUTPUT_DIR)!r}")
        return

    print(f"converted {num_done} of {len(files)} files")
    print("todo: replace input files:")
    print(
        f"  mv -v {str(INPUT_DIR)!r} {(str(INPUT_DIR) + '.' + str(int(time.time())))!r} &&" +
        f" mv -v {str(OUTPUT_DIR)!r} {str(INPUT_DIR)!r}"
    )


if __name__ == "__main__":
    main()

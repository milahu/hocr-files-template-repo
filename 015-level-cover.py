#!/usr/bin/env python3

# NOTE this color-leveling step should run as late as possible
# because if rotation (and similar transforms)
# (especially in 065-remove-page-borders.py and 070-deskew.py)
# run after this color-leveling step
# then these transforms turn sharp edges into blurry edges

import os
import sys
import time
import shutil
import traceback
import importlib.util
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import psutil
import numpy as np
from tqdm import tqdm

from _shared import (
    load_config,
    latest_dst_exists,
    remove_done_files,
)


# --- Setup -------------------------------------------------------------------
# os.chdir(Path(__file__).resolve().parent)
src = Path("013-scan-cover-fixed-colors")
dst = Path(Path(__file__).stem)


# --- Settings ----------------------------------------------------------------
config = load_config()


# --- Helper: level adjustment (contrast stretch) -----------------------------
def apply_level(img: np.ndarray, low: float = 0.2, high: float = 0.9) -> np.ndarray:
    """
    Apply a linear color level adjustment using fractional thresholds.
    low/high are in [0,1], e.g., 0.2 = 20%, 0.9 = 90%.
    """
    if img.dtype == np.uint8:
        max_val = 255
    elif img.dtype == np.uint16:
        max_val = 65535
    else:
        raise ValueError(f"Unsupported image dtype: {img.dtype}")

    # Convert fractional thresholds to absolute values
    low_val = low * max_val
    high_val = high * max_val

    if high_val <= low_val:
        return img.copy()

    img_stretched = (img.astype(float) - low_val) * (max_val / (high_val - low_val))
    img_stretched = np.clip(img_stretched, 0, max_val)
    return img_stretched.astype(img.dtype)


# --- Worker ------------------------------------------------------------------
def process_image(image_path: Path) -> str:
    filename = image_path.name
    output_path = dst / filename

    # Load
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"error: failed to read {image_path}")
        sys.exit(1)

    # Level (contrast stretch)
    lowthresh, highthresh = config.cover_lowthresh, config.cover_highthresh
    img = apply_level(img, lowthresh, highthresh)

    # Save image
    cv2.imwrite(str(output_path), img)
    # print(f"writing {output_path}")


def try_process_image(*args):
    "ensure all exceptions are caught and serialized safely back to the main process"
    try:
        process_image(*args)
        return None
    except Exception as e:
        tb = traceback.format_exc()
        return (e, tb)


# --- Parallel execution ------------------------------------------------------
def main():

    if not config.do_level:
        print("not leveling")
        if dst.exists():
            print("keeping dst")
            return
        print("creating symlink from dst to src")
        dst.symlink_to(src)
        return

    dst.mkdir(parents=True, exist_ok=True)

    t1 = time.time()
    images = sorted(src.glob(f"*.{config.scan_format}"))
    if not images:
        print("No input files found.")
        exit(0)

    images = remove_done_files(images, dst)
    if not images:
        print("nothing to do")
        return

    if 1:
        # debug: print config
        print(f"leveling colors:")
        print(f"  levels for cover pages: low={config.cover_lowthresh} high={config.cover_highthresh}")

    num_workers = psutil.cpu_count(logical=False) or 1
    print(f"Using {num_workers} workers...")

    tqdm_kwargs = dict(
        total=len(images),
        ncols=80,
        unit="page",
    )

    with (
        ProcessPoolExecutor(max_workers=num_workers) as executor,
        tqdm(**tqdm_kwargs) as pbar,
    ):
        futures = {executor.submit(try_process_image, img): img for img in images}
        for future in as_completed(futures):
            err = future.result()
            if err:
                executor.shutdown(cancel_futures=True)
                e, tb = err
                print(f"\nException in worker:\n{tb}")
                raise e
            pbar.update(1)

    t2 = time.time()
    print(f"done {len(images)} pages in {int(t2 - t1)} seconds using {num_workers} workers")


if __name__ == "__main__":
    main()

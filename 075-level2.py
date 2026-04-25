#!/usr/bin/env python3

# apply color leveling again
# after 060-rotate-crop-level.py
# read from 070-deskew
# write to 075-level2.tmp.{time}
# move 070-deskew to 070-deskew.bak
# move 075-level2.tmp.{time} to 070-deskew

import os
import sys
import time
import traceback
import importlib.util
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import psutil
import numpy as np


# --- Setup -------------------------------------------------------------------
os.chdir(Path(__file__).resolve().parent)
src = Path("070-deskew")
bak = Path("070-deskew.bak")
dst = src # replace source
tmp = Path(Path(__file__).stem + f".tmp.{time.time()}")
tmp.mkdir(parents=True, exist_ok=True)

# --- Settings ----------------------------------------------------------------
config_path = Path("074-level2-config.py")

spec = importlib.util.spec_from_file_location("level2_config", config_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)


if not config.do_level:
    print("noop. nothing to do here. config.do_level is False")
    sys.exit()


if bak.exists():
    print(f"error: {bak} exists, so this script was already run")
    sys.exit(1)


def chained_levels(min_1, max_1, min_r, max_r):
    """
    Compute the second levels (min_2, max_2) such that:
    levels(min_2, max_2) ∘ levels(min_1, max_1)
    == levels(min_r, max_r)

    All values are expected in [0, 1].
    """
    scale_1 = max_1 - min_1
    if scale_1 == 0:
        raise ValueError("max_1 and min_1 must be different")

    min_2 = (min_r - min_1) / scale_1
    max_2 = (max_r - min_1) / scale_1

    return min_2, max_2


if config.do_level and not (config.lowthresh == config.previous_lowthresh and config.highthresh == config.previous_highthresh):
    # color leveling was already applied by 060-rotate-crop-level.py
    # (config.lowthresh, config.highthresh) specifies the resulting levels
    # so here we factor in the previous leveling operation
    if config.lowthresh < config.previous_lowthresh:
        print(f"error: config.lowthresh < config.previous_lowthresh")
        sys.exit(1)
    if config.highthresh > config.previous_highthresh:
        print(f"error: config.highthresh > config.previous_highthresh")
        sys.exit(1)
    min_2, max_2 = chained_levels(
        min_1=config.previous_lowthresh,
        max_1=config.previous_highthresh,
        min_r=config.lowthresh,
        max_r=config.highthresh,
    )
    print(f"previous: levels 1: min={config.previous_lowthresh} max={config.previous_highthresh}")
    print(f"now:      levels 2: min={min_2} max={max_2}")
    print(f"result:   levels r: min={config.lowthresh} max={config.highthresh}")
    config.lowthresh, config.highthresh = min_2, max_2


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
    page_number = int(filename.replace(".tiff", ""))
    output_path = tmp / filename
    if output_path.exists():
        print(f"keeping {output_path}")
        return

    if not config.do_level:
        # noop
        shutil.copy(image_path, output_path)
        return

    # Load
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"error: failed to read {image_path}")
        sys.exit(1)

    # Level (contrast stretch)
    if config.do_level:
        img = apply_level(img, config.lowthresh, config.highthresh)

    # Save as TIFF
    cv2.imwrite(str(output_path), img)
    print(f"writing {output_path}")


def try_process_image(*args):
    "ensure all exceptions are caught and serialized safely back to the main process"
    try:
        process_image(*args)
        return None
    except Exception as e:
        tb = traceback.format_exc()
        return (e, tb)


# --- Parallel execution ------------------------------------------------------
if __name__ == "__main__":
    t1 = time.time()
    images = sorted(src.glob(f"*.{config.scan_format}"))
    if not images:
        print("No input files found.")
        exit(0)

    num_workers = psutil.cpu_count(logical=False) or 1
    print(f"Using {num_workers} workers...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(try_process_image, img): img for img in images}
        for future in as_completed(futures):
            err = future.result()
            if err:
                executor.shutdown(cancel_futures=True)
                e, tb = err
                print(f"\nException in worker:\n{tb}")
                raise e

    print(f"> mv {str(src)!r} {str(bak)!r}")
    src.rename(bak)

    print(f"> mv {str(tmp)!r} {str(dst)!r}")
    tmp.rename(dst)

    t2 = time.time()
    print(f"done {len(images)} pages in {int(t2 - t1)} seconds using {num_workers} workers")

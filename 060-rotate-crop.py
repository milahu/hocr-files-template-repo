#!/usr/bin/env python3

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

from _shared import (
    load_config,
    get_page_num,
)


# --- Setup -------------------------------------------------------------------
os.chdir(Path(__file__).resolve().parent)
src = Path("046-compress")
dst = Path(Path(__file__).stem)
dst.mkdir(parents=True, exist_ok=True)


# --- Settings ----------------------------------------------------------------
config = load_config()


# --- Worker ------------------------------------------------------------------
def process_image(image_path: Path) -> str:
    filename = image_path.name
    page_number = int(filename.split(".")[0]) # "001.jpg" -> 1
    output_path = dst / filename
    if output_path.exists():
        print(f"keeping {output_path}")
        return

    crop_box = config.crop_odd_box if page_number % 2 == 1 else config.crop_even_box
    rotation = config.rotate_odd if page_number % 2 == 1 else config.rotate_even

    # Load
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"error: failed to read {image_path}")
        sys.exit(1)

    # Rotate
    if config.do_rotate:
        if rotation == 90:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 270:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rotation == 180:
            img = cv2.rotate(img, cv2.ROTATE_180)
        else:
            # arbitrary angle
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), rotation, 1.0)
            img = cv2.warpAffine(img, M, (w, h))

    # Crop
    if config.do_crop:
        x1, y1, x2, y2 = crop_box
        img = img[y1:y2, x1:x2]

    # Save image
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
    images = sorted(src.glob(f"*.{config.image_format}"))
    if not images:
        print("No input files found.")
        exit(0)

    if config.do_rotate == False and config.do_crop == False:
        print("no rotate, no crop -> hardlinking all files from src to dst")
        for f_src in images:
            f_dst = dst / f_src.name
            if f_dst.exists():
                continue
            os.link(f_src, f_dst)
        sys.exit()

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

    t2 = time.time()
    print(f"done {len(images)} pages in {int(t2 - t1)} seconds using {num_workers} workers")

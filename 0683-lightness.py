#!/usr/bin/env python3

import os
import time
import shutil
import traceback
import subprocess
import importlib.util
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import psutil
from PIL import Image, ImageStat
from tqdm import tqdm


# input directory
src = "065-remove-page-borders"
# src = "067-force-lightmode"
src = "0663-level"

# output file
dst = os.path.splitext(os.path.basename(__file__))[0] + ".txt"


def get_physical_cpu_count():
    try:
        # Attempt to get the number of physical cores using psutil
        return psutil.cpu_count(logical=False)
    except AttributeError:
        # If psutil is not available or does not support this function, use os.cpu_count()
        return os.cpu_count()

max_workers = get_physical_cpu_count() or 1

def compute_lightness(filepath):
    """Compute mean lightness (0–100) of image."""
    filename = os.path.basename(filepath)

    try:
        with Image.open(filepath) as img:
            gray = img.convert("L")
            stat = ImageStat.Stat(gray)
            lightness = stat.mean[0] / 255 * 100
    except Exception:
        lightness = -1.0

    return filename, lightness


def try_compute_lightness(filepath):
    """Compute lightness safely, returning (result, err)."""
    try:
        return compute_lightness(filepath), None
    except Exception as e:
        return None, e


def main():
    t1 = int(time.time())
    num_pages = 0

    # Collect all image files in src
    image_files = []
    for f in sorted(os.listdir(src)):
        # input file path
        in_path = os.path.join(src, f)
        if not os.path.isfile(in_path):
            continue
        image_files.append(in_path)

    # Compute lightness in parallel
    page_lightness = {}

    tqdm_kwargs = dict(
        total=len(image_files),
        ncols=80,
        unit="page",
    )

    with (
        ProcessPoolExecutor(max_workers=max_workers) as executor,
        tqdm(**tqdm_kwargs) as pbar
    ):
        futures = {executor.submit(try_compute_lightness, f): f for f in image_files}
        for future in as_completed(futures):
            res, err = future.result()
            if err is not None:
                executor.shutdown(cancel_futures=True)
                raise err  # propagate exception to main
            filename, lightness = res
            page_lightness[filename] = lightness
            # print(f"lightness: {lightness:10.6f} {filename}")
            pbar.update(1)

    with open(dst, "w", encoding="utf8") as fd:
        # sort by lightness descending
        for (filename, lightness) in sorted(page_lightness.items(), key=lambda x: -x[1]):
            fd.write(f"{lightness:010.6f} {filename}\n")

    print(f"done {dst}")

    # explain next steps
    print()
    print("next steps:")
    print()
    print("run ./06845-show-white-pages.py")
    print()
    print("find the lightness thresholds")
    print("- between white pages and non-white pages -> set config: fill_white_pages_white_lightness_threshold")
    print("- between non-white pages and text pages -> set config: deskew_white_lightness_threshold")
    print()
    print("non-white pages contain only small text elements like page numbers")
    print("which should be preserved by 0685-fill-white-pages.py")
    print()
    print("text pages also contain straight lines of text")
    print("which can be used to deskew the pages by their contents in 070-deskew.py")


if __name__ == "__main__":
    main()

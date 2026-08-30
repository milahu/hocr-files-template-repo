#!/usr/bin/env python3

import os
import sys
import time
import shutil
import traceback
import subprocess
import importlib.util
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import psutil
from PIL import Image, ImageStat

# load _shared.py from workdir
sys.path.insert(0, os.getcwd())

from _shared import (
    load_config,
    get_page_num,
    remove_done_files,
)

# Directories
# src = "065-remove-page-borders"
# src = "067-force-lightmode"
src = "0663-level"
dst = os.path.splitext(os.path.basename(__file__))[0]
os.makedirs(dst, exist_ok=True)

src = Path(src)
dst = Path(dst)

config = load_config()


def get_physical_cpu_count():
    try:
        # Attempt to get the number of physical cores using psutil
        return psutil.cpu_count(logical=False)
    except AttributeError:
        # If psutil is not available or does not support this function, use os.cpu_count()
        return os.cpu_count()

max_workers = get_physical_cpu_count() or 1

def compute_lightness(filepath):
    """Compute mean lightness (0–100) of image in filepath."""
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

    r'''
    # Collect all image files
    image_files = []
    for f in sorted(os.listdir(src)):
        in_path = os.path.join(src, f)
        if not os.path.isfile(in_path):
            continue
        image_files.append(in_path)
    '''

    if 0:
        # debug
        print(f"config.deskew_white_lightness_threshold={config.deskew_white_lightness_threshold}")
        print(f"config.deskew_black_lightness_threshold={config.deskew_black_lightness_threshold}")
        print(f"config.deskew_dark_lightness_threshold={config.deskew_dark_lightness_threshold}")

    # load lightness file
    lightness_txt_path = Path(config.deskew_lightness_file)
    page_lightness = {}
    image_files = []
    with lightness_txt_path.open() as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                lightness_str, filename = line.split(" ", 1)
                lightness = float(lightness_str)
                # TODO? dont store lightness values as percent values
                # or how important is the human-readability of the lightness.txt files
                lightness = lightness / 100 # convert from percent value
            except ValueError:
                print(f"Skipping malformed line {lineno}: {line}", file=sys.stderr)
                continue

            image_path = Path(src) / filename

            if not image_path.exists():
                print(f"Error: Missing file: {image_path}", file=sys.stderr)
                continue

            page_lightness[filename] = lightness
            image_files.append(image_path)

    # check if we have lightness values for all input files
    if not src.exists():
        sys.exit(f"error: input directory does not exist: {src}")
    files = sorted(src.iterdir())
    assert isinstance(files, list)
    if not files:
        sys.exit(f"nothing to do: input directory is empty: {src}")
    missing_lightness_files = []
    extra_files = []
    for f in files:
        page_num = get_page_num(f)
        if not 1 <= page_num <= config.num_pages:
            extra_files.append(f)
            # we dont need lightness of extra pages
            continue
        if f.name not in page_lightness:
            missing_lightness_files.append(f.name)
    if missing_lightness_files:
        print(f"error: missing lightness values for these input files: {missing_lightness_files}")

    # copy extra pages: book cover, etc
    if extra_files:
        print(f"copying {len(extra_files)} extra pages")
        for f in extra_files:
            f_dst = dst / f.name
            shutil.copy(f, f_dst)

    # Deskew non-empty pages
    for filepath in image_files:
        filename = os.path.basename(filepath)
        out_path = os.path.join(dst, filename)

        if os.path.exists(out_path):
            continue

        lightness = page_lightness[filename]

        if 0:
            # debug
            print(f"{filename}: lightness={lightness}")

        if lightness >= config.deskew_white_lightness_threshold:
            print(f"Skipping deskew on white page {filename}")
            shutil.copy2(filepath, out_path)
            continue
        if lightness <= config.deskew_black_lightness_threshold:
            print(f"Skipping deskew on black page {filename}")
            shutil.copy2(filepath, out_path)
            continue

        # TODO handle mixed pages
        # mixed = upper half white + lower half black (or similar)
        background_color = "FFFFFF"  # white
        if lightness < config.deskew_dark_lightness_threshold:
            background_color = "000000"  # black

        # Deskew command
        deskew_args = [
            "deskew",
            "-o", str(out_path),
            "-b", background_color,
            str(filepath)
        ]

        print("+", " ".join(deskew_args))
        subprocess.run(deskew_args, check=True)
        num_pages += 1

    t2 = int(time.time())
    print(f"Done {num_pages} pages in {t2 - t1} seconds")


if __name__ == "__main__":
    main()

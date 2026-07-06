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

# Directories
src = "065-remove-page-borders"
# src = "067-force-lightmode"
dst = os.path.splitext(os.path.basename(__file__))[0]
lightness_txt_path = Path("0683-lightness.txt")
os.makedirs(dst, exist_ok=True)


if 0:
    # set config here
    class DeskewConfig:
        # Threshold to consider a page "white" (mean lightness close to 100)
        WHITE_LIGHTNESS_THRESHOLD = 99.99

        # Threshold to consider a page "black" (mean lightness close to 0)
        # black page with little white text can have 0.49 to 0.84
        # black page with no text can have 0 to 0.43
        BLACK_LIGHTNESS_THRESHOLD = 0.45

        # Threshold to consider a page "dark" (black page with white text)
        # white page with lots of black text can have 80
        BLACK_LIGHTNESS_THRESHOLD_2 = 25
    config = DeskewConfig()
else:
    # load config from file
    config_path = Path("069-deskew-config.py")
    spec = importlib.util.spec_from_file_location("deskew_config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    config_path = Path("0684-fill-white-pages-config.py")
    spec = importlib.util.spec_from_file_location("fill_white_pages_config", config_path)
    fill_white_pages_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fill_white_pages_config)
    if hasattr(config, "config.WHITE_LIGHTNESS_THRESHOLD"):
        print(
            "warning: ignoring WHITE_LIGHTNESS_THRESHOLD from 069-deskew-config.py"
            " in favor of WHITE_LIGHTNESS_THRESHOLD from 0684-fill-white-pages-config.py"
        )
    config.WHITE_LIGHTNESS_THRESHOLD = fill_white_pages_config.WHITE_LIGHTNESS_THRESHOLD


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

    # load lightness file
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
            except ValueError:
                print(f"Skipping malformed line {lineno}: {line}", file=sys.stderr)
                continue

            image_path = Path(src) / filename

            if not image_path.exists():
                print(f"Error: Missing file: {image_path}", file=sys.stderr)
                continue

            page_lightness[filename] = lightness
            image_files.append(image_path)

    # Deskew non-empty pages
    for filepath in image_files:
        filename = os.path.basename(filepath)
        out_path = os.path.join(dst, filename)

        if os.path.exists(out_path):
            continue

        lightness = page_lightness[filename]
        if lightness >= config.WHITE_LIGHTNESS_THRESHOLD:
            print(f"Skipping deskew on white page {filename}")
            shutil.copy2(filepath, out_path)
            continue
        if lightness <= config.BLACK_LIGHTNESS_THRESHOLD:
            print(f"Skipping deskew on black page {filename}")
            shutil.copy2(filepath, out_path)
            continue

        # TODO handle black or mixed pages
        # mixed = upper half white + lower half black (or similar)
        background_color = "FFFFFF"  # white
        if lightness < config.BLACK_LIGHTNESS_THRESHOLD_2:
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

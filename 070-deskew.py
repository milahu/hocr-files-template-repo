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
lightness_txt_path = f"{dst}.lightness.txt"
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
    config_path = Path("068-deskew-config.py")
    spec = importlib.util.spec_from_file_location("deskew_config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)


def get_physical_cpu_count():
    try:
        # Attempt to get the number of physical cores using psutil
        return psutil.cpu_count(logical=False)
    except AttributeError:
        # If psutil is not available or does not support this function, use os.cpu_count()
        return os.cpu_count()

max_workers = get_physical_cpu_count() or 1

def compute_lightness(filepath):
    """Compute mean lightness (0–100) of a TIFF image."""
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

    # Collect all TIFF files that actually need processing
    tiff_files = []
    for f in sorted(os.listdir(src)):
        if not f.lower().endswith(".tiff"):
            continue
        in_path = os.path.join(src, f)
        if 1:
            # also process files that already exist in output
            tiff_files.append(in_path)
            continue
        out_path = os.path.join(dst, f)
        if os.path.exists(out_path):
            # Skip files that already exist in output
            continue
        tiff_files.append(in_path)

    # Compute lightness in parallel
    page_lightness = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(try_compute_lightness, f): f for f in tiff_files}
        for future in as_completed(futures):
            res, err = future.result()
            if err is not None:
                executor.shutdown(cancel_futures=True)
                raise err  # propagate exception to main
            filename, lightness = res
            page_lightness[filename] = lightness
            print(f"lightness: {lightness:10.6f} {filename}")

    with open(lightness_txt_path, "w", encoding="utf8") as fd:
        # sort by lightness descending
        for (filename, lightness) in sorted(page_lightness.items(), key=lambda x: -x[1]):
            fd.write(f"{lightness:010.6f} {filename}\n")

    # Deskew non-empty pages
    for filepath in tiff_files:
        filename = os.path.basename(filepath)
        # page_number = int(filename[:-5])  # strip ".tiff" suffix
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
            "-o", out_path,
            "-b", background_color,
            filepath
        ]

        print("+", " ".join(deskew_args))
        subprocess.run(deskew_args, check=True)
        num_pages += 1

    t2 = int(time.time())
    print(f"Done {num_pages} pages in {t2 - t1} seconds")


if __name__ == "__main__":
    main()

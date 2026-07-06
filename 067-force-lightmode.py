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
from PIL import Image, ImageStat, ImageOps

# Directories
src = "065-remove-page-borders"
dst = os.path.splitext(os.path.basename(__file__))[0]
lightness_txt_path = f"{dst}.lightness.txt"
os.makedirs(dst, exist_ok=True)


if 0:
    # set config here
    class DeskewConfig:
        # Threshold to consider a page "black" (mean lightness close to 0)
        # white page with lots of black text can have 80
        BLACK_LIGHTNESS_THRESHOLD = 25
    config = DeskewConfig()
else:
    # load config from file
    config_path = Path("066-force-lightmode-config.py")
    spec = importlib.util.spec_from_file_location("force_lightmode_config", config_path)
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

def process_page(filepath):
    filename = os.path.basename(filepath)

    # page_number = int(filename.split(".")[0])  # "001.jpg" -> 1
    out_path = os.path.join(dst, filename)

    # Compute mean lightness (0–100)
    try:
        with Image.open(filepath) as img:
            gray = img.convert("L")
            stat = ImageStat.Stat(gray)
            lightness = stat.mean[0] / 255 * 100
    except Exception:
        lightness = -1.0

    if lightness == -1.0 or lightness > config.BLACK_LIGHTNESS_THRESHOLD:
        shutil.copy2(filepath, out_path)
        return (filename, lightness, False)

    if os.path.exists(out_path):
        return (filename, lightness, True)

    with Image.open(filepath) as img:
        if img.mode not in ("L", "RGB"):
            img = img.convert("RGB")
        # Invert colors
        inverted = ImageOps.invert(img)
        inverted.save(out_path)

    return (filename, lightness, True)


def try_process_page(filepath):
    "return (result, exc)"
    try:
        return process_page(filepath), None
    except Exception as exc:
        return None, exc


def main():
    t1 = int(time.time())
    num_pages = 0

    # TODO use scan_format from 030-measure-page-size.txt
    scan_format = "jpg"

    # Collect all image files that actually need processing
    image_files = []
    for f in sorted(os.listdir(src)):
        if not f.lower().endswith(f".{scan_format}"):
            continue
        in_path = os.path.join(src, f)
        if 1:
            # also process files that already exist in output
            image_files.append(in_path)
            continue
        out_path = os.path.join(dst, f)
        if os.path.exists(out_path):
            # Skip files that already exist in output
            continue
        image_files.append(in_path)

    # process pages in parallel
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(try_process_page, f): f for f in image_files}
        for future in as_completed(futures):
            res, err = future.result()
            if err is not None:
                executor.shutdown(cancel_futures=True)
                raise err  # propagate exception to main
            # filename, lightness, done_invert = res
            results.append(res)
            num_pages += 1

    with open(lightness_txt_path, "w", encoding="utf8") as fd:
        # sort by lightness descending
        for (filename, lightness, done_invert) in sorted(results, key=lambda x: -x[1]):
            fd.write(f"{lightness:010.6f} {filename} {int(done_invert)}\n")

    t2 = int(time.time())
    print(f"Done {num_pages} pages in {t2 - t1} seconds")


if __name__ == "__main__":
    main()

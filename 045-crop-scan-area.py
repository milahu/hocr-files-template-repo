#!/usr/bin/env python3

# TODO rewrite 060-rotate-crop-level.sh in python
# and merge it with this script

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import re

import cv2
import numpy as np
import psutil
from PIL import Image

# -----------------------------
# Hard-coded paths
# -----------------------------
INPUT_DIR = r"040-scan-pages"
OUTPUT_DIR = r"045-crop-scan-area"

# -----------------------------------------------------------------------------
# Load configuration (replacement for "source")
# -----------------------------------------------------------------------------

config = {}

config_file = Path("030-measure-page-size.txt")
if config_file.exists():
    with config_file.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", line)
            if m:
                config[m.group(1)] = m.group(2)

scan_format = config.get("scan_format")

if not scan_format:
    sys.exit("error: scan_format not defined")

# -----------------------------------------------------------------------------

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


def remove_bottom_white_rectangle(pil_img):
    """
    Detect and remove bottom white rectangle (artifact) from a scanned image.
    Assumes the white rectangle spans the entire image width.
    """

    # Convert PIL image to OpenCV format (RGB → BGR)
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Image dimensions
    height, width = gray.shape

    # Determine where the bottom white area starts
    white_threshold = 250  # near pure white
    bottom_crop_y = height  # default (no crop)

    # Scan upward from the bottom to find the first non-white row
    for y in range(height - 1, -1, -1):
        row = gray[y, :]
        if np.mean(row < white_threshold) > 0.01:  # some non-white pixels
            bottom_crop_y = y + 1
            break

    # Crop only if a white rectangle was found
    if bottom_crop_y < height:
        return pil_img.crop((0, 0, width, bottom_crop_y))
    else:
        return pil_img


def process_file(filename):
    if not filename.lower().endswith(f".{scan_format}"):
        return

    input_path = os.path.join(INPUT_DIR, filename)
    output_path = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(output_path):
        print(f"keeping {output_path}")
        return

    try:
        with Image.open(input_path) as img:
            cleaned = remove_bottom_white_rectangle(img)
            # note: format is detected from the file extension
            cleaned.save(output_path)
            print(f"writing {output_path}")
    except Exception as exc:
        print(f"error processing {input_path}: {type(exc).__name__}: {exc}")


def process_directory():
    filenames = sorted(os.listdir(INPUT_DIR))

    suffix = f".{scan_format}"
    filenames = list(filter(lambda n: n.endswith(suffix), filenames))

    num_workers = psutil.cpu_count(logical=False) or 1

    print(f"Using {num_workers} workers...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(executor.map(process_file, filenames))


if __name__ == "__main__":
    process_directory()

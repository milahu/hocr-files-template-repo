#!/usr/bin/env python3

# TODO rewrite 060-rotate-crop-level.sh in python
# and merge it with this script

"""
AI prompt:

create a python script to remove bottom white rectangles from scanned images.
the bottom white rectangles are artifacts created by my scanner.
they are perfectly white rectangles (color #ffffff)
and above these rectangles, there is always a grey area.
the white rectangles have 100% width of the image.
the script should process an input directory with *.jpg images
and write output images to an output directory (same image format).
the input and output paths should be hard-coded in the script,
so the script takes no command-line arguments.
the script should be based on the PIL (pillow) image library
(and on the opencv and numpy libraries when necessary)
"""

import os
import cv2
import numpy as np
from PIL import Image

# -----------------------------
# Hard-coded paths
# -----------------------------
INPUT_DIR = r"040-scan-pages"
OUTPUT_DIR = r"045-crop-scan-area"

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
        cropped_img = pil_img.crop((0, 0, width, bottom_crop_y))
        return cropped_img
    else:
        return pil_img


def process_directory():
    for filename in sorted(os.listdir(INPUT_DIR)):
        # TODO use scan_format from 030-measure-page-size.txt
        scan_format = "jpg"
        if not filename.lower().endswith(f".{scan_format}"):
            continue
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(output_path):
            print(f"keeping {output_path}")
            continue

        try:
            with Image.open(input_path) as img:
                cleaned = remove_bottom_white_rectangle(img)
                # note: format is detected from the file extension
                cleaned.save(output_path)
                print(f"writing {output_path}")
        except Exception as exc:
            print(f"error processing {input_path}: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    process_directory()
    # print("Done.")

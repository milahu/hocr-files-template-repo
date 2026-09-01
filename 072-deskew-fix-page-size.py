#!/usr/bin/env python3

import sys
import time
import traceback
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import cv2
import psutil
from tqdm import tqdm

from _shared import (
    load_config,
    get_page_num,
    remove_done_files,
)


# --- Setup -------------------------------------------------------------------
src = Path("070-deskew")
dst = Path(Path(__file__).stem)


# --- Settings ----------------------------------------------------------------
config = load_config()


# --- Helper: millimeters to pixels -------------------------------------------
def px_of_mm(mm):
    return mm * config.scan_resolution / 25.4


# Target content page size in pixels.
# round() is needed because image dimensions must be integer pixels.
PAGE_WIDTH_PX = round(px_of_mm(config.page_width_mm))
PAGE_HEIGHT_PX = round(px_of_mm(config.page_height_mm))


# --- Helper: crop / restore page size ----------------------------------------
def crop_or_restore_page_size(img):
    """
    Make the image exactly PAGE_WIDTH_PX x PAGE_HEIGHT_PX.

    If the source image is larger than the target in both dimensions,
    symmetrically crop it around the center.

    If the source image is smaller than the target:
      - when config.deskew_fix_page_size_keep_small_pages is True,
        return the original image unchanged;
      - otherwise, place the image centered on a white canvas of the
        target size.

    If one dimension is smaller and the other is larger, the image is
    first cropped in the larger dimension and padded in the smaller
    dimension.
    """

    height, width = img.shape[:2]

    target_width = PAGE_WIDTH_PX
    target_height = PAGE_HEIGHT_PX

    # If the image is too small and the config says to keep it,
    # return it completely unchanged.
    if (
        config.deskew_fix_page_size_keep_small_pages
        and (width < target_width or height < target_height)
    ):
        return img

    # Determine the crop region in the source image.
    crop_width = min(width, target_width)
    crop_height = min(height, target_height)

    left = (width - crop_width) // 2
    top = (height - crop_height) // 2

    cropped = img[
        top:top + crop_height,
        left:left + crop_width,
    ]

    # If the cropped image already has the target dimensions, we're done.
    if crop_width == target_width and crop_height == target_height:
        return cropped

    # Create a white canvas with the same number of channels and dtype.
    if img.ndim == 2:
        canvas = np.full(
            (target_height, target_width),
            255,
            dtype=img.dtype,
        )
    else:
        channels = img.shape[2]

        # White in all channels.
        canvas = np.full(
            (target_height, target_width, channels),
            255,
            dtype=img.dtype,
        )

    # Center the cropped image on the canvas.
    x = (target_width - crop_width) // 2
    y = (target_height - crop_height) // 2

    canvas[
        y:y + crop_height,
        x:x + crop_width,
    ] = cropped

    return canvas


# --- Worker ------------------------------------------------------------------
def process_image(image_path: Path) -> str:
    filename = image_path.name
    output_path = dst / filename

    # Load
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"failed to read {image_path}")

    original_height, original_width = img.shape[:2]

    # Crop / restore page size
    img = crop_or_restore_page_size(img)

    # If small images are configured to be kept, they intentionally retain
    # their original dimensions.
    if not (
        config.deskew_fix_page_size_keep_small_pages
        and (
            original_width < PAGE_WIDTH_PX
            or original_height < PAGE_HEIGHT_PX
        )
    ):
        # Sanity check
        height, width = img.shape[:2]

        if width != PAGE_WIDTH_PX or height != PAGE_HEIGHT_PX:
            raise RuntimeError(
                f"Unexpected output size for {filename}: "
                f"{width}x{height}, "
                f"expected {PAGE_WIDTH_PX}x{PAGE_HEIGHT_PX}"
            )

    # Save image
    if not cv2.imwrite(str(output_path), img):
        raise RuntimeError(f"failed to write {output_path}")

    return filename


def try_process_image(*args):
    "ensure all exceptions are caught and serialized safely back to the main process"
    try:
        process_image(*args)
        return None
    except Exception as e:
        tb = traceback.format_exc()
        return (e, tb)


# --- Main --------------------------------------------------------------------
def main():

    print(
        f"target page size: "
        f"{config.page_width_mm} x {config.page_height_mm} mm "
        f"@ {config.scan_resolution} DPI "
        f"= {PAGE_WIDTH_PX} x {PAGE_HEIGHT_PX} px"
    )

    dst.mkdir(parents=True, exist_ok=True)

    t1 = time.time()

    images = sorted(src.glob(f"*.{config.scan_format}"))

    if not images:
        print("No input files found.")
        return

    images = remove_done_files(images, dst)

    if not images:
        print("nothing to do")
        return

    # Separate content pages from extra pages such as covers.
    content_files = []
    extra_files = []

    for f in images:
        page_num = get_page_num(f)

        if 1 <= page_num <= config.num_pages:
            content_files.append(f)
        else:
            extra_files.append(f)

    # Copy extra pages unchanged.
    if extra_files:
        print(f"copying {len(extra_files)} extra pages")

        for f in extra_files:
            f_dst = dst / f.name
            shutil.copy(f, f_dst)

    # Process only content pages.
    if not content_files:
        print("no content files")
        return

    images = content_files

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
        futures = {
            executor.submit(try_process_image, img): img
            for img in images
        }

        for future in as_completed(futures):
            err = future.result()

            if err:
                executor.shutdown(cancel_futures=True)

                e, tb = err
                print(f"\nException in worker:\n{tb}")
                raise e

            pbar.update(1)

    t2 = time.time()

    print(
        f"done {len(images)} pages in "
        f"{int(t2 - t1)} seconds using {num_workers} workers"
    )


if __name__ == "__main__":
    main()

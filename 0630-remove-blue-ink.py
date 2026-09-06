#!/usr/bin/env python3

# This is an optional cleanup step for books that have been disfigured by
# previous owners with handwritten annotations made with a blue ballpoint pen.
#
# Unlike graphite-pencil annotations, blue ballpoint ink cannot usually be
# removed safely with a rubber eraser without damaging the printed page.
# Therefore, when such annotations are present, they are removed in software
# as an optional post-processing step of the scanning pipeline.

import sys
import time
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import psutil
import numpy as np
from tqdm import tqdm

from _shared import (
    load_config,
    get_page_num,
    latest_dst_exists,
    remove_done_files,
)


# --- Setup -------------------------------------------------------------------

src = Path("060-rotate-crop")

dst = Path(Path(__file__).stem)
dst.mkdir(parents=True, exist_ok=True)


# --- Settings ----------------------------------------------------------------
config = load_config()


# These are deliberately conservative defaults.
#
# The input is expected to be a color scan. Blue ballpoint ink should have
# significantly more blue than red, while the original book content is black
# or gray and therefore has approximately equal RGB components.
#
# All thresholds can be overridden in 000-config.py if necessary.

BLUE_MIN_B = getattr(config, "remove_blue_min_b", 80)
BLUE_MIN_B_MINUS_R = getattr(config, "remove_blue_min_b_minus_r", 25)
BLUE_MIN_B_MINUS_G = getattr(config, "remove_blue_min_b_minus_g", 10)

# If a pixel is very dark, assume it is original black print rather than blue.
#
# This is important for the "black wins" rule:
#
#     blue pen + black printed line -> black printed line survives
#
# The threshold is intentionally fairly low because ballpoint ink can be dark.
BLACK_MAX_GRAY = getattr(config, "remove_blue_black_max_gray", 80)

# Blue pixels that are almost as dark as black print should be treated
# conservatively. This prevents accidentally deleting dark blue text.
DARK_BLUE_MAX_GRAY = getattr(config, "remove_blue_dark_max_gray", 110)

# Optional morphological cleanup. A ballpoint line can contain tiny gaps
# caused by scanner noise, so closing the blue mask makes detection more
# continuous.
BLUE_CLOSE_SIZE = getattr(config, "remove_blue_close_size", 3)

# How far to extend the blue mask.
BLUE_GROW_PIXELS = getattr(config, "remove_blue_grow_pixels", 2)

# Newly added pixels may be less blue than the core mask, because they are
# anti-aliased mixtures of blue ink, black ink and white paper.
#
# These thresholds are deliberately weaker than the core blue thresholds.
EDGE_MIN_B_MINUS_R = getattr(config, "remove_blue_edge_min_b_minus_r", 8)
EDGE_MIN_B_MINUS_G = getattr(config, "remove_blue_edge_min_b_minus_g", 3)

# Don't erase pixels that are too dark. This protects black printed lines.
EDGE_MIN_GRAY = getattr(config, "remove_blue_edge_min_gray", 55)


def get_bgr(img):
    """
    Return a BGR image.

    The normal pipeline input is a color scan. Grayscale input is also
    accepted, although no blue ink can be detected in a grayscale image.
    """

    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.ndim == 3 and img.shape[2] == 3:
        return img

    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    raise ValueError(f"Unsupported image shape: {img.shape}")


def detect_blue_ink(img):
    """
    Detect blue ballpoint ink.

    OpenCV stores color images as BGR.

    A blue pixel should have:
        B > R
        B > G

    We additionally require a minimum amount of blue dominance so that
    neutral gray/black text is not classified as blue.

    Returns:
        boolean mask, True where blue ink is detected.
    """

    bgr = get_bgr(img)

    b = bgr[:, :, 0].astype(np.int16)
    g = bgr[:, :, 1].astype(np.int16)
    r = bgr[:, :, 2].astype(np.int16)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    blue_mask = (
        (b >= BLUE_MIN_B)
        & ((b - r) >= BLUE_MIN_B_MINUS_R)
        & ((b - g) >= BLUE_MIN_B_MINUS_G)
    )

    # Preserve genuinely black pixels.
    #
    # This implements the important "black wins" rule. A dark pixel is
    # unlikely to be a blue-only annotation and should not be removed.
    blue_mask &= gray > BLACK_MAX_GRAY

    # Very dark blue ink is ambiguous. Be conservative and don't remove it.
    blue_mask &= gray > DARK_BLUE_MAX_GRAY

    # Close small gaps in pen strokes.
    if BLUE_CLOSE_SIZE > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (BLUE_CLOSE_SIZE, BLUE_CLOSE_SIZE),
        )
        blue_mask = cv2.morphologyEx(
            blue_mask.astype(np.uint8),
            cv2.MORPH_CLOSE,
            kernel,
        ).astype(bool)

    return blue_mask


# --- Blue ink detection ------------------------------------------------------

def detect_blue_ink(img):
    """
    Detect the strong/core part of blue handwritten ink.

    Returns a boolean mask.
    """

    bgr = get_bgr(img)

    b = bgr[:, :, 0].astype(np.int16)
    g = bgr[:, :, 1].astype(np.int16)
    r = bgr[:, :, 2].astype(np.int16)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    mask = (
        (b >= BLUE_MIN_B)
        & ((b - r) >= BLUE_MIN_B_MINUS_R)
        & ((b - g) >= BLUE_MIN_B_MINUS_G)
        & (gray > BLACK_MAX_GRAY)
    )

    return mask


def grow_blue_ink_mask(img, blue_mask):
    """
    Grow the confidently detected blue mask by a small number of pixels.

    Pixels added by the growth are accepted only when their color is still
    somewhat blue.

    This catches anti-aliased blue/black/white transition pixels without
    blindly deleting nearby black printed content.
    """

    if BLUE_GROW_PIXELS <= 0:
        return blue_mask

    bgr = get_bgr(img)

    b = bgr[:, :, 0].astype(np.int16)
    g = bgr[:, :, 1].astype(np.int16)
    r = bgr[:, :, 2].astype(np.int16)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Dilate the core blue mask.
    kernel_size = BLUE_GROW_PIXELS * 2 + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )

    expanded = cv2.dilate(
        blue_mask.astype(np.uint8),
        kernel,
    ).astype(bool)

    # Only inspect pixels that were added by dilation.
    added = expanded & ~blue_mask

    # A weaker blue signal is sufficient for edge pixels.
    #
    # Example:
    #
    #     pure blue       -> strong blue dominance
    #     anti-aliased    -> weak blue dominance
    #     black print     -> approximately neutral
    #
    edge_blue = (
        (b - r >= EDGE_MIN_B_MINUS_R)
        & (b - g >= EDGE_MIN_B_MINUS_G)
        & (gray >= EDGE_MIN_GRAY)
    )

    # Don't allow the growth operation to overwrite the core mask.
    #
    # The result is:
    #     core blue
    #       OR
    #     nearby weak-blue edge
    return blue_mask | (added & edge_blue)


def remove_blue_ink(img):
    """
    Remove blue handwritten annotations while preserving black book content.

    The algorithm distinguishes between:

        1. confidently blue pixels
        2. anti-aliased blue edge pixels
        3. neutral/dark black pixels

    The blue mask is grown by a small amount, but the added pixels must still
    show some evidence of blue.

    This is safer than simply dilating the mask because black printed text
    remains protected.
    """

    if img.ndim == 2:
        # Color information is required to distinguish blue from black.
        return img

    bgr = get_bgr(img).copy()

    # First find the confidently blue core.
    blue_mask = detect_blue_ink(bgr)

    if not np.any(blue_mask):
        return img

    # Then include the anti-aliased edge around the blue stroke.
    blue_mask = grow_blue_ink_mask(
        bgr,
        blue_mask,
    )

    # Replace detected blue + blue-edge pixels with white.
    bgr[blue_mask] = (255, 255, 255)

    # Preserve alpha if the source is BGRA.
    if img.shape[2] == 4:
        alpha = img[:, :, 3]

        result = cv2.cvtColor(
            bgr,
            cv2.COLOR_BGR2BGRA,
        )

        result[:, :, 3] = alpha

        return result

    return bgr


# --- Worker ------------------------------------------------------------------
def process_image(image_path: Path) -> str:
    filename = image_path.name
    page_number = int(filename.split(".")[0])  # "001.tiff" -> 1

    output_path = dst / filename

    if latest_dst_exists(image_path, output_path):
        return

    if output_path.exists():
        # replace outdated output_path with the latest version
        output_path.unlink()

    # Load
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    if img is None:
        print(f"error: failed to read {image_path}")
        sys.exit(1)

    # Remove blue handwritten annotations.
    img = remove_blue_ink(img)

    # Save image
    cv2.imwrite(
        str(output_path),
        img,
        config.cv2_imwrite_params,
    )


def try_process_image(*args):
    """
    Ensure all exceptions are caught and serialized safely back to the
    main process.
    """

    try:
        process_image(*args)
        return None

    except Exception as e:
        tb = traceback.format_exc()
        return (e, tb)


# --- Parallel execution ------------------------------------------------------
def main():
    t1 = time.time()

    images = sorted(src.glob(f"*.{config.scan_format}"))

    if not images:
        print("No input files found.")
        exit(0)

    images = remove_done_files(images, dst)

    if 0:
        # debug: process only some pages
        def filter_file(file):
            page_num = get_page_num(file)

            if page_num not in (1, 2, 3):
                return False

            return True

        images = list(filter(filter_file, images))

    if not images:
        print("nothing to do")
        return

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

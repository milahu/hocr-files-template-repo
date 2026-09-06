#!/usr/bin/env python3

# Convert color scans to grayscale after blue-ink removal.
#
# Scanning must be done in color so that blue handwritten annotations can
# be detected and removed. Once the blue ink has been removed, the remaining
# book content is bitonal (black on white), so the color channels are no
# longer useful. Converting to grayscale reduces each image from 3 channels
# to 1 channel and makes subsequent pipeline stages more efficient.

import sys
import time
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import psutil
from tqdm import tqdm

from _shared import (
    load_config,
    get_page_num,
    latest_dst_exists,
    remove_done_files,
)


# --- Setup -------------------------------------------------------------------

src = Path("0630-remove-blue-ink")

dst = Path(Path(__file__).stem)
dst.mkdir(parents=True, exist_ok=True)


# --- Settings ----------------------------------------------------------------
config = load_config()


# --- Worker ------------------------------------------------------------------
def process_image(image_path: Path) -> str:
    filename = image_path.name
    page_number = int(filename.split(".")[0])  # "001.jpg" -> 1

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

    # Convert color image to grayscale.
    #
    # cv2.imread() returns BGR, so COLOR_BGR2GRAY is the appropriate
    # conversion. If the input is already grayscale, leave it unchanged.
    if img.ndim == 3:
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

        else:
            raise ValueError(f"Unsupported image shape: {img.shape}")

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

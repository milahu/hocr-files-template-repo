#!/usr/bin/env python3

import time
from pathlib import Path

import cv2
import psutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from _shared import (
    load_config,
    get_page_num,
    latest_dst_exists,
    remove_done_files,
    get_image_viewer_argstr,
)

config = load_config()


INPUT_DIR = Path("060-rotate-crop")
OUTPUT_DIR = Path(Path(__file__).stem)


def get_erase_rectangle(file_path):
    """
    return rectangle to erase on page.
    Format: (x, y, width, height)
    """
    page_num = get_page_num(file_path)

    # filter pages
    if page_num % 2 == 0: return None # dont process even pages
    # if page_num % 2 == 1: return None # dont process odd pages
    if page_num > config.num_pages: return None # dont process appended pages

    # remove scanner artifacts on the left edge
    x = 5
    y = 1800
    width = 250
    if page_num == 161: width = 160 # special case: last page
    height = 1600

    return (x, y, width, height)


def process_file(file_path):

    out_path = OUTPUT_DIR / file_path.name

    erase_rect = get_erase_rectangle(file_path)

    if not erase_rect:
        # dont process this file
        # create hardlink
        out_path.hardlink_to(file_path)
        # this file was not processed
        return None

    # process this file

    img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read image: {file_path}")

    x, y, w, h = erase_rect

    # Clip rectangle to image bounds.
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(img.shape[1], x + w)
    y1 = min(img.shape[0], y + h)

    assert x1 > x0 and y1 > y0

    img[y0:y1, x0:x1] = 255

    if not cv2.imwrite(str(out_path), img):
        raise RuntimeError(f"Failed to write image: {out_path}")

    # this file was processed
    return out_path


def main():
    files = sorted(INPUT_DIR.glob(f"*.{config.scan_format}"))

    if not files:
        print("No image files found in", INPUT_DIR)
        return

    # Only submit work for files that still need processing.
    files = remove_done_files(files, OUTPUT_DIR)

    if not files:
        print("nothing to do")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    num_workers = psutil.cpu_count(logical=False) or 1

    # by default, OpenCV uses multiple CPU threads
    cv2.setNumThreads(1)

    if 0:
        # debug: disable parallel processing
        num_workers = 1

    tqdm_kwargs = dict(
        total=len(files),
        ncols=80,
        unit="page",
    )

    processed_output_files = []

    with (
        ProcessPoolExecutor(max_workers=num_workers) as executor,
        tqdm(**tqdm_kwargs) as pbar,
    ):
        futures = {
            executor.submit(process_file, fname): fname
            for fname in files
        }

        for future in as_completed(futures):
            fname = futures[future]
            try:
                out_path = future.result()
                if out_path:
                    processed_output_files.append(out_path)
            except Exception as exc:
                print(f"Error processing {fname}: {exc}")
                raise
            finally:
                pbar.update(1)

    print("view processed files:")
    print("  " + get_image_viewer_argstr(processed_output_files, config))

    print("todo: replace input files:")
    print(f"  mv -v {str(INPUT_DIR)!r} {(str(INPUT_DIR) + '.' + str(int(time.time())))!r} && mv -v {str(OUTPUT_DIR)!r} {str(INPUT_DIR)!r}")


if __name__ == "__main__":
    main()

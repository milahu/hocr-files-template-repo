#!/usr/bin/env python3

# NOTE this script will replace image files in-place

import argparse
import os
import secrets
import sys
from pathlib import Path
from tqdm import tqdm

from PIL import Image

from _shared import (
    load_config,
    get_page_num,
    parse_page_sequence,
    resolve_pdf_page_number,
    get_image_viewer_argstr,
)


src = "065-remove-page-borders"


def random_temp_path(path: Path) -> Path:
    """Return a random temporary filename in the same directory."""
    while True:
        tmp = path.with_name(
            f".{path.name}.{secrets.token_hex(8)}.tmp"
        )
        if not tmp.exists():
            return tmp


def rotate_180(image_path: Path):
    """Rotate an image by 180 degrees and replace it atomically."""
    tmp_path = random_temp_path(image_path)

    try:
        with Image.open(image_path) as img:
            rotated = img.rotate(180)
            rotated.save(tmp_path, format=img.format)

        # Atomic replacement (same filesystem)
        os.rename(tmp_path, image_path)

    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def main():

    parser = argparse.ArgumentParser(
        description="Rotate selected page images by 180 degrees."
    )

    parser.add_argument(
        "pages",
        nargs="+",
        help="page numbers and page ranges, e.g. 10,20-30 40",
    )

    args = parser.parse_args()

    config = load_config()

    image_dir = Path(src)

    if not image_dir.is_dir():
        sys.exit(f"Error: Image directory not found: {image_dir}")

    # --------------------------------------------------------------
    # Parse logical page sequence
    # --------------------------------------------------------------

    page_spec = ",".join(args.pages)

    print(
        f"page specification: "
        f"{page_spec!r}"
    )

    try:
        page_sequence = parse_page_sequence(
            page_spec,
            config.num_pages,
        )

    except ValueError as e:
        print(
            f"error: invalid page specification: "
            f"{e}",
            file=sys.stderr,
        )
        sys.exit(1)

    selected_pages = [
        page
        for page in page_sequence
        if page != 0
    ]

    # Avoid rotating the same page twice.
    selected_pages = list(dict.fromkeys(selected_pages))

    # count = 0
    done_paths = []

    tqdm_kwargs = dict(
        total=len(selected_pages),
        ncols=80,
        unit="page",
    )

    with (
        # ProcessPoolExecutor(max_workers=num_workers) as executor,
        tqdm(**tqdm_kwargs) as pbar,
    ):

        for page_num in selected_pages:

            # Find the image corresponding to this page.
            image_path = None

            for path in image_dir.iterdir():
                if not path.is_file():
                    continue

                try:
                    if get_page_num(path) == page_num:
                        image_path = path
                        break
                except (ValueError, IndexError):
                    continue

            if image_path is None:
                print(
                    f"Error: Missing image for page {page_num}",
                    file=sys.stderr,
                )
                continue

            # print(f"Rotating 180 degrees: {image_path}")
            rotate_180(image_path)
            done_paths.append(image_path)
            # count += 1
            pbar.update(1)

    # print(f"Done {count} images")
    print()
    print(f"view rotated pages:")
    print(f"  {get_image_viewer_argstr(done_paths, config)}")


if __name__ == "__main__":
    main()

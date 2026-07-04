#!/usr/bin/env python3

# NOTE this script will replace image files in-place

import argparse
import os
import secrets
import sys
from pathlib import Path
import importlib.util

from PIL import Image

config_path = Path("0684-fill-white-pages-config.py")


if 0:
    # set config here
    class FillWhitePagesConfig:
        # Threshold to consider a page "white" (mean lightness close to 100)
        WHITE_LIGHTNESS_THRESHOLD = 99.9
    config = FillWhitePagesConfig()
    image_dir = "065-remove-page-borders"
    lightness_file = "0683-lightness.txt"
else:
    # load config from config_path
    spec = importlib.util.spec_from_file_location("fill_white_pages_config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)


def random_temp_path(path: Path) -> Path:
    """Return a random temporary filename in the same directory."""
    while True:
        tmp = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
        if not tmp.exists():
            return tmp


def fill_white(image_path: Path):
    """Replace an image with a solid white version atomically."""
    tmp_path = random_temp_path(image_path)

    try:
        with Image.open(image_path) as img:
            white = Image.new(img.mode, img.size, "white")
            white.save(tmp_path, format=img.format)

        # Atomic replacement (same filesystem)
        os.rename(tmp_path, image_path)

    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def main():

    lightness_file = Path(config.lightness_file)

    image_dir = Path(config.image_dir)

    if not lightness_file.is_file():
        sys.exit(f"Error: Lightness file not found: {lightness_file}")

    if not image_dir.is_dir():
        sys.exit(f"Error: Image directory not found: {image_dir}")

    count = 0

    with lightness_file.open() as f:
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

            if lightness < config.WHITE_LIGHTNESS_THRESHOLD:
                # keep file
                continue

            # fill white

            image_path = image_dir / filename

            if not image_path.exists():
                print(f"Error: Missing file: {image_path}", file=sys.stderr)
                continue

            print(f"Filling white: {image_path} (lightness was {lightness:.6f})")
            fill_white(image_path)
            count += 1

    print(f"Done {count} images")


if __name__ == "__main__":
    main()

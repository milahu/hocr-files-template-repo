#!/usr/bin/env python3

import os
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import psutil
from PIL import Image

from _shared import (
    load_config,
    get_page_num,
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)

datetime_str = (
    datetime.now()
    .astimezone()
    .isoformat(timespec="seconds")
    .replace(":", "-")
)

# src = Path("040-scan-pages")
src = Path("045-crop-scan-area")
dst = Path(Path(__file__).stem)
# dst = src  # replace files in src

jpeg_quality = 90

num_workers = psutil.cpu_count(logical=False) or 1

config = load_config()

scan_format = config.scan_format
image_format = config.image_format
color_pages = config.color_pages

if not scan_format:
    sys.exit("error: scan_format not defined")

if not image_format:
    sys.exit("error: image_format not defined")

# -----------------------------------------------------------------------------

replace = src == dst

if replace:
    tmp = Path(f"{Path(__file__).stem}.tmp.{datetime_str}")
    bak = Path(f"070-deskew.bak.{datetime_str}")
else:
    dst.mkdir(parents=True, exist_ok=True)

if replace:
    tmp.mkdir(parents=True, exist_ok=True)
    bak.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------

if scan_format == image_format:
    print("no change in format -> hardlinking all files from src to dst")
    # for name in sorted(os.listdir(src)):
    for f_src in sorted(src.glob(f"*.{scan_format}")):
        f_dst = dst / f_src.name
        if f_dst.exists():
            continue
        os.link(f_src, f_dst)
    sys.exit()

# -----------------------------------------------------------------------------

TIFF_COMPRESSION = {
    1: "None",
    5: "LZW",
    6: "JPEG",
    7: "JPEG",
    8: "Deflate",
    32773: "PackBits",
}


def process_image(args):
    (
        f,
        replace,
        src,
        dst,
        tmp,
        bak,
        jpeg_quality,
        color_pages,
    ) = args

    f = Path(f)
    src = Path(src)
    dst = Path(dst)
    tmp = Path(tmp) if tmp else None
    bak = Path(bak) if bak else None

    base_dst = f.with_suffix(f".{image_format}").name

    if replace:
        f_tmp = tmp / base_dst
        f_dst = src / base_dst
        f_bak = bak / f.name
    else:
        f_dst = dst / base_dst
        f_tmp = f_dst

    size_a = f.stat().st_size

    with Image.open(f) as img:

        compression = TIFF_COMPRESSION.get(
            img.tag_v2.get(259),
            str(img.tag_v2.get(259, "unknown")),
        )

        page_num = int(f.stem)

        #
        # Preserve original disabled logic
        #
        if False:
            if (
                not color_pages
                or page_num not in color_pages
            ):
                img = img.convert("L")

        #
        # JPEG-compatible modes
        #
        if img.mode in ("RGBA", "RGBa"):
            img = img.convert("RGB")
        elif img.mode == "P":
            img = img.convert("RGB")
        elif img.mode == "LA":
            img = img.convert("L")
        elif img.mode == "1":
            img = img.convert("L")
        elif img.mode.startswith("I") or img.mode == "F":
            img = img.convert("L")

        img.save(
            f_tmp,
            # format="JPEG",
            quality=jpeg_quality,
            optimize=True,
        )

    size_b = f_tmp.stat().st_size

    message = None
    if size_b < size_a:
        percent = size_b * 100 // size_a
        message = (
            # my scanner returns uncompressed TIFF files
            # so compression is always None
            # f"compressing {f}: compression: {compression} -> JPEG. "
            f"file size: {size_a} -> {size_b} ({percent}%)"
        )

    done = False

    if replace:
        if size_b < size_a:
            os.link(f, f_bak)
            os.replace(f_tmp, f_dst)
            if f != f_dst:
                f.unlink()
            done = True
        else:
            f_tmp.unlink(missing_ok=True)
    else:
        done = True

    return done, message


# -----------------------------------------------------------------------------

files = [
    f
    for f in sorted(src.glob(f"*.{scan_format}"))
    # if re.fullmatch(rf"\d+\.{re.escape(scan_format)}", f.name)
    if f.name.endswith(f".{scan_format}")
]

def filter_file(f_src):
    base_dst = f_src.with_suffix(f".{image_format}").name
    f_dst = dst / base_dst
    # print(f"filter_file: f_src={f_src} f_dst={f_dst}")
    return not f_dst.exists()

files = list(filter(filter_file, files))

tasks = [
    (
        str(f),
        replace,
        str(src),
        str(dst),
        str(tmp) if replace else None,
        str(bak) if replace else None,
        jpeg_quality,
        color_pages,
    )
    for f in files
]

num_done = 0

with ProcessPoolExecutor(max_workers=num_workers) as executor:
    for done, message in executor.map(process_image, tasks):
        if message:
            print(message)
        if done:
            num_done += 1

# -----------------------------------------------------------------------------

if replace:
    try:
        tmp.rmdir()
    except OSError:
        pass

if replace and num_done == 0:
    try:
        bak.rmdir()
    except OSError:
        pass

print(f"done. compressed {num_done} images")

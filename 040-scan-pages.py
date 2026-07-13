#!/usr/bin/env python3

import os
import argparse
import importlib.util
import shlex
import subprocess
import sys
import time
from pathlib import Path

from _shared import (
    load_config,
    get_page_num,
)


def main():
    script_dir = Path(__file__).resolve().parent
    dst = Path(Path(__file__).stem)
    os.chdir(script_dir)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "scanner_name",
        help=(
            "your scanner name from 'scanimage -L'."
            " example: 'brother5:bus1;dev4'"
        )
    )
    parser.add_argument(
        "first_page",
        type=int,
        help=(
            "the first page number of this batch of pages."
            " most books start with page number 1."
        )
    )
    args = parser.parse_args()

    config = load_config()

    dst.mkdir(exist_ok=True)

    # allow appending some pages
    # without having to rename all files
    max_num_pages = int(max(
        config.num_pages + 100,
        config.num_pages * 1.2,
    ))

    page_num_width = len(str(max_num_pages))

    page_num_fmt = f"%0{page_num_width}d"

    scanimage_scan_format = config.scan_format
    if scanimage_scan_format == "jpg":
        # scanimage expects "jpeg"
        scanimage_scan_format = "jpeg"

    cmd = [
        "scanimage",
        f"--device-name={args.scanner_name}",
        f"--resolution={config.scan_resolution}",
        f"--format={scanimage_scan_format}",
        f"--mode={config.scan_mode}",
        f"--source={config.scan_source}",
        "--MultifeedDetection=yes",
        "--SkipBlankPage=no",
        "--AutoDocumentSize=no",
        "--AutoDeskew=no",
        "-x", str(config.margined_scan_x),
        "-y", str(config.margined_scan_y),
        f"--batch={dst / (page_num_fmt + '.' + config.scan_format)}",
        "--progress",
        "--batch-print",
        f"--batch-start={args.first_page}",
    ]

    args_file = dst / "scanimage-args.txt"
    if not args_file.exists():
        args_file.write_text(" ".join(shlex.quote(x) for x in cmd) + "\n")

    t1 = time.time()
    num_pages_1 = len(list(dst.glob(f"*.{config.scan_format}")))

    print("+", shlex.join(cmd))
    proc = subprocess.run(cmd)

    print("-" * 80)

    if proc.returncode != 0:
        print(f"scanimage failed with returncode {proc.returncode}")

    num_pages_2 = len(list(dst.glob(f"*.{config.scan_format}")))
    t2 = time.time()

    print(f"done {num_pages_2 - num_pages_1} pages in {int(t2 - t1)} seconds")


if __name__ == "__main__":
    main()

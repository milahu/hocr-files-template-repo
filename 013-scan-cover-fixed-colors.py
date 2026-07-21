#!/usr/bin/env python3

import os
from pathlib import Path
import subprocess
import sys
import shlex

from _shared import (
    load_config,
    get_page_num,
    latest_dst_exists,
    remove_done_files,
)


src = Path("010-scan-cover")
fix_colors_py = Path("012-fix-colors.py")
calibration_json = Path("012-fix-colors.calibration.json")


image_suffixes = (
    ".jpg",
    ".jpeg",
    ".tiff",
    ".png",
    ".avif",
)


def main():
    # Change to the directory containing this script
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent
    dst = Path(script_path.stem)

    os.chdir(script_dir)

    src_files = sorted(src.glob("*"))

    def filter_src_file(src_file):
        return src_file.suffix in image_suffixes

    src_files = list(filter(filter_src_file, src_files))

    src_files = remove_done_files(src_files, dst)

    if not src_files:
        print("nothing to do")
        return

    dst.mkdir(exist_ok=True)

    for src_file in src_files:

        dst_file = dst / src_file.name

        cmd = [
            sys.executable,
            str(fix_colors_py),
            "--apply-calibration",
            str(calibration_json),
            str(src_file),
            str(dst_file),
        ]

        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

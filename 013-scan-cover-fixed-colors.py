#!/usr/bin/env python3

import os
from pathlib import Path
import subprocess
import sys
import shlex


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
    dst.mkdir(exist_ok=True)

    # for src_file in sorted((script_dir / "010-scan-cover").glob("*.tiff")):
    for src_file in sorted(src.glob("*")):

        if src_file.suffix not in image_suffixes:
            continue

        dst_file = dst / src_file.name

        if dst_file.exists():
            print(f"keeping {dst_file}")
            continue

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

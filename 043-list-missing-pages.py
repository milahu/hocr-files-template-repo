#!/usr/bin/env python3

import argparse
from pathlib import Path

from _shared import (
    load_config,
    compress_paths,
)

src = Path("040-scan-pages")


def main():
    config = load_config()

    if not src.is_dir():
        parser.error(f"not a directory: {src}")

    # Keep this in sync with the scan-images script.
    max_num_pages = int(max(
        config.num_pages + 100,
        config.num_pages * 1.2,
    ))
    page_num_width = len(str(max_num_pages))
    page_num_fmt = f"%0{page_num_width}d"

    expected = [
        src / f"{page_num_fmt % page_num}.{config.scan_format}"
        for page_num in range(1, config.num_pages + 1)
    ]

    missing = [path for path in expected if not path.exists()]

    if not missing:
        print(
            f"all {config.num_pages} expected files are present "
            f"in {src}"
        )
        return

    print(
        f"missing {len(missing)} of {config.num_pages} expected files:"
    )
    print(compress_paths(missing))


if __name__ == "__main__":
    main()

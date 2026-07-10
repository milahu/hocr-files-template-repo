#!/usr/bin/env python3

import shutil
import sys
from pathlib import Path

from _shared import (
    load_config,
    get_page_num,
)


src = Path("040-scan-pages")


def shift_page_number(page_num, page_delta, config):
    page0a = f"{page_num:0{config.page_num_width}d}"
    new_page_num = page_num + page_delta
    page0b = f"{new_page_num:0{config.page_num_width}d}"

    a = src.joinpath(f"{page0a}.{config.scan_format}")
    b = src.joinpath(f"{page0b}.{config.scan_format}")

    if not a.exists():
        return

    if b.exists():
        print(f"FIXME collision: mv {str(a)!r} {str(b)!r}")
        sys.exit(1)

    print(f"renaming {str(a)!r} -> {str(b)!r}")
    shutil.move(a, b)


def main():
    if len(sys.argv) < 2:
        print("usage:")
        print(f"  {sys.argv[0]} -5")
        print(f"  {sys.argv[0]} +5")
        sys.exit(1)

    page_delta = int(sys.argv[1])

    print(f"page_delta: {page_delta!r}")

    config = load_config()

    max_page_num = 0
    for path in src.glob(f"*.{config.scan_format}"):
        page_num = get_page_num(path)
        if page_num > max_page_num:
            max_page_num = page_num

    if page_delta < 0:
        # decrease page numbers
        start, end, step = 1, max_page_num + 1, +1
    else:
        # increase page numbers: loop in reverse order
        start, end, step = max_page_num, 0, -1

    for page_num in range(start, end, step):
        shift_page_number(page_num, page_delta, config)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

"""
065-remove-page-borders.py

Detects left/right/bottom page edges and rectifies perspective while
preserving the top edge.
"""

import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from _shared import (
    load_config,
    get_page_num,
)

DEBUG = False

INPUT_DIR = Path("060-rotate-crop")
OUTPUT_DIR = Path("065-remove-page-borders")
DEBUG_DIR = Path("065-remove-page-borders-debug")

# TODO expose config
max_left = 100
max_right = 100
max_bottom = 50
# TODO what
strips = 40

OUTPUT_DIR.mkdir(exist_ok=True)

if DEBUG:
    DEBUG_DIR.mkdir(exist_ok=True)


def estimate_background(img):
    s = 20
    samples = np.concatenate(
        [
            img[:, -s:, :].reshape(-1, 3),
            img[:, :s, :].reshape(-1, 3),
            img[-s:, :, :].reshape(-1, 3),
        ],
        axis=0,
    )
    return np.median(samples, axis=0)


def diff_image(img, bg):
    return (
        np.abs(img.astype(np.int16) - bg.astype(np.int16))
        .sum(axis=2)
        .astype(np.float32)
    )


def edge_from_profile(profile):
    g = np.gradient(profile)
    return int(np.argmax(g))


def detect_left(diff):
    h, w = diff.shape
    pts = []
    for y0, y1 in zip(
        np.linspace(0, h, strips + 1, dtype=int)[:-1],
        np.linspace(0, h, strips + 1, dtype=int)[1:],
    ):
        prof = np.median(diff[y0:y1, :max_left], axis=0)
        x = edge_from_profile(cv2.GaussianBlur(prof.reshape(1, -1), (1, 1), 0).ravel())
        pts.append((x, (y0 + y1) / 2))
    return np.array(pts, np.float32)


def detect_right(diff):
    h, w = diff.shape
    pts = []
    for y0, y1 in zip(
        np.linspace(0, h, strips + 1, dtype=int)[:-1],
        np.linspace(0, h, strips + 1, dtype=int)[1:],
    ):
        prof = np.median(diff[y0:y1, w - max_right : w], axis=0)[::-1]
        x = edge_from_profile(prof)
        pts.append((w - x, (y0 + y1) / 2))
    return np.array(pts, np.float32)


def detect_bottom(diff):
    h, w = diff.shape
    pts = []
    for x0, x1 in zip(
        np.linspace(0, w, strips + 1, dtype=int)[:-1],
        np.linspace(0, w, strips + 1, dtype=int)[1:],
    ):
        prof = np.median(diff[h - max_bottom : h, x0:x1], axis=1)[::-1]
        y = edge_from_profile(prof)
        pts.append(((x0 + x1) / 2, h - y))
    return np.array(pts, np.float32)


def fit_line(points):
    vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    return float(vx), float(vy), float(x0), float(y0)


def x_at_y(line, y):
    vx, vy, x0, y0 = line
    return x0 + (y - y0) * vx / vy


def y_at_x(line, x):
    vx, vy, x0, y0 = line
    return y0 + (x - x0) * vy / vx


def process_image(path, page_num, config):
    scan_aspect = config.scan_aspect
    # TODO use page_num and scan_aspect to restore the page aspect ratio
    img = cv2.imread(str(path))
    if img is None:
        return
    h, w = img.shape[:2]
    bg = estimate_background(img)
    diff = diff_image(img, bg)
    lp = detect_left(diff)
    rp = detect_right(diff)
    bp = detect_bottom(diff)
    ll = fit_line(lp)
    rl = fit_line(rp)
    bl = fit_line(bp)
    lx = x_at_y(ll, 0)
    rx = x_at_y(rl, 0)
    byl = y_at_x(bl, lx)
    byr = y_at_x(bl, rx)
    if lx > max_left:
        raise RuntimeError(f"{path}: left edge outside range")
    if w - rx > max_right:
        raise RuntimeError(f"{path}: right edge outside range")
    if h - min(byl, byr) > max_bottom:
        raise RuntimeError(f"{path}: bottom edge outside range")
    src = np.float32([[lx, 0], [rx, 0], [rx, byr], [lx, byl]])
    dst = np.float32(
        [[0, 0], [rx - lx, 0], [rx - lx, max(byl, byr)], [0, max(byl, byr)]]
    )
    M = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(img, M, (int(dst[1, 0]), int(dst[2, 1])))
    output_path = OUTPUT_DIR / path.name
    print(f"writing {output_path}")
    cv2.imwrite(str(output_path), out)
    if DEBUG:
        dbg = img.copy()
        for p in lp:
            cv2.circle(dbg, (int(p[0]), int(p[1])), 2, (0, 0, 255), -1)
        for p in rp:
            cv2.circle(dbg, (int(p[0]), int(p[1])), 2, (0, 255, 0), -1)
        for p in bp:
            cv2.circle(dbg, (int(p[0]), int(p[1])), 2, (255, 0, 0), -1)
        cv2.polylines(dbg, [src.astype(int)], True, (0, 255, 255), 2)
        cv2.imwrite(str(DEBUG_DIR / (path.stem + "_debug.png")), dbg)
    return output_path


def get_page_num(path):
    # parse page number: "001.jpg" -> 1
    m = re.match(r"^(\d+)", path.name)
    # page_num = int(m.group(1)) if m else 0
    page_num = int(m.group(1)) # can throw
    return page_num


# -----------------------------------------------------------------------------
# Load configuration (replacement for "source")
# -----------------------------------------------------------------------------


def main():

    config = load_config()

    scan_x = config.scan_x
    scan_y = config.scan_y

    config.scan_aspect = scan_x / scan_y

    worker_args = []
    image_format = config.image_format
    for path in sorted(INPUT_DIR.glob(f"*.{image_format}")):
        output_path = OUTPUT_DIR / path.name
        if output_path.exists():
            continue
        page_num = get_page_num(path)
        args = (
            path,
            page_num,
            config,
        )
        worker_args.append(args)

    if not worker_args:
        print("nothing to do")
        return

    num_workers = psutil.cpu_count(logical=False) or 1
    print(f"Using {num_workers} workers...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(executor.map(process_image, worker_args))


if __name__ == "__main__":
    main()

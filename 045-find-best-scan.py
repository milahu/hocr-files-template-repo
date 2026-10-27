#!/usr/bin/env python3
"""
Choose the cleanest scan of each page from repeated book scans.

Inputs:
    040-scan-pages/010.tiff
    040-scan-pages.tmp.<timestamp>/010.tiff
    ...

Output:
    045-find-best-scan/010.tiff

This first version selects an entire source image; it does not combine pixels
from different scans. Therefore it does not need to geometrically align scans
to make its decision. Alignment is needed for future pixel-level fusion, not
for comparing per-image quality metrics.

Dependencies:
    pip install opencv-python numpy pillow tqdm psutil
"""

import argparse
import csv
import math
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import psutil
from PIL import Image
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
SRC = Path("040-scan-pages")
DST = Path("045-find-best-scan")
IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}

NUM_WORKERS = psutil.cpu_count(logical=False) or 1

# Score weights. All component scores are designed so lower is better.
WEIGHT_VERTICAL_LINES = 5.0
WEIGHT_SPECKLES = 2.0
WEIGHT_BLUR = 1.0
WEIGHT_CONTRAST = 0.25

# Ignore a narrow border for the quality metrics, where scanner edges and
# black page borders can otherwise dominate the score.
BORDER_RATIO = 0.015

# Downsample large images for scoring. The source image is copied unchanged.
MAX_SCORE_DIMENSION = 1800

# -----------------------------------------------------------------------------

def page_number(path):
    """Return the numeric page prefix, or None for non-page filenames."""
    match = re.match(r"^(\d+)\.", path.name)
    return int(match.group(1)) if match else None


def discover_sources(src):
    """Find the main scan directory and timestamped repeat-scan directories."""
    dirs = [src]
    dirs.extend(sorted(
        p for p in src.parent.glob(f"{src.name}.tmp.*")
        if p.is_dir()
    ))

    pages = {}
    for directory in dirs:
        if not directory.is_dir():
            continue
        for path in directory.iterdir():
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            number = page_number(path)
            if number is not None:
                pages.setdefault(number, []).append(path)

    for number in pages:
        pages[number] = sorted(pages[number], key=lambda p: (p.parent.name, p.name))
    return pages


def load_gray(path):
    """Read an image with OpenCV, supporting TIFF and grayscale/color inputs."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"cannot read image: {path}")

    if image.ndim == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            image = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)

    if image.dtype == np.uint16:
        gray = image.astype(np.float32) / 257.0
    elif image.dtype == np.uint8:
        gray = image.astype(np.float32)
    else:
        gray = cv2.normalize(image.astype(np.float32), None, 0, 255,
                             cv2.NORM_MINMAX)
    return np.clip(gray, 0, 255).astype(np.uint8)


def score_image(path):
    """
    Calculate explainable image-quality metrics; lower total score is better.

    vertical_lines:
        Mean absolute difference between a column and its nearby horizontal
        neighbors, averaged vertically. Persistent narrow streaks increase it.
    speckles:
        Count/strength of isolated dark or light pixels compared with a median
        filter. This is only a proxy: real punctuation can also be isolated.
    blur:
        Inverse Laplacian variance; sharper text generally scores better.
    contrast:
        Penalizes very low grayscale standard deviation slightly.
    """
    gray = load_gray(path)
    original_h, original_w = gray.shape

    scale = min(1.0, MAX_SCORE_DIMENSION / max(original_h, original_w))
    if scale < 1.0:
        gray = cv2.resize(
            gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )

    h, w = gray.shape
    bx = int(w * BORDER_RATIO)
    by = int(h * BORDER_RATIO)
    work = gray[by:max(by + 1, h - by), bx:max(bx + 1, w - bx)]
    if work.size == 0:
        work = gray

    # Detect persistent vertical streaks relative to nearby columns. Using
    # blurred pixels makes this less sensitive to individual letters/speckles.
    blurred = cv2.GaussianBlur(work, (3, 3), 0)
    f = blurred.astype(np.float32)
    left = np.roll(f, 2, axis=1)
    right = np.roll(f, -2, axis=1)
    vertical_residual = np.abs(f - (left + right) / 2.0)

    # Aggregate by column; a real streak tends to persist through many rows.
    column_strength = np.mean(vertical_residual, axis=0)
    # Remove the baseline contribution from normal text/paper texture.
    baseline = float(np.median(column_strength))
    vertical_lines = float(np.mean(np.maximum(column_strength - baseline, 0)))

    # Isolated-pixel residual from a 3x3 median filter. Count both dark and
    # light deviations, but ignore small differences due to compression/noise.
    median = cv2.medianBlur(work, 3).astype(np.int16)
    delta = np.abs(work.astype(np.int16) - median)
    speckles = float(np.mean(np.maximum(delta - 18, 0)))

    # High Laplacian variance indicates fine detail/sharpness. Convert it to
    # an inverse score so lower remains better.
    lap_var = float(cv2.Laplacian(work, cv2.CV_32F).var())
    blur = 1.0 / math.sqrt(max(lap_var, 1.0))

    contrast_std = float(work.std())
    contrast = 1.0 / max(contrast_std, 1.0)

    total = (
        WEIGHT_VERTICAL_LINES * vertical_lines
        + WEIGHT_SPECKLES * speckles
        + WEIGHT_BLUR * blur
        + WEIGHT_CONTRAST * contrast
    )

    return {
        "path": str(path),
        "score": float(total),
        "vertical_lines": vertical_lines,
        "speckles": speckles,
        "laplacian_variance": lap_var,
        "contrast_std": contrast_std,
        "width": original_w,
        "height": original_h,
    }


def file_contents_equal(path_a, path_b, chunk_size=1024 * 1024):
    """Compare two files byte-for-byte without loading them fully into memory."""
    path_a = Path(path_a)
    path_b = Path(path_b)

    if path_a.stat().st_size != path_b.stat().st_size:
        return False

    with path_a.open("rb") as a, path_b.open("rb") as b:
        while True:
            chunk_a = a.read(chunk_size)
            chunk_b = b.read(chunk_size)

            if chunk_a != chunk_b:
                return False

            if not chunk_a:
                return True


def process_page(task):
    number, candidates, dst_dir, force = task
    dst_dir = Path(dst_dir)
    dst = dst_dir / candidates[0].name

    if dst.exists() and not force:
        output_mtime = dst.stat().st_mtime_ns
        newest_input_mtime = max(
            p.stat().st_mtime_ns
            for p in candidates
        )

        if newest_input_mtime <= output_mtime:
            return (
                number,
                "skipped",
                str(dst),
                None,
                "inputs are not newer",
            )

    scored = []
    errors = []
    for candidate in candidates:
        try:
            scored.append(score_image(candidate))
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")

    if not scored:
        return number, "error", str(dst), None, "; ".join(errors)

    # Deterministic tie-break: stable source path order from discover_sources().
    winner = min(scored, key=lambda item: (item["score"], item["path"]))
    source = Path(winner["path"])

    dst_dir.mkdir(parents=True, exist_ok=True)

    if dst.exists() and file_contents_equal(source, dst):
        details = {
            "winner": str(source),
            "candidate_count": len(candidates),
            "score": winner["score"],
            "vertical_lines": winner["vertical_lines"],
            "speckles": winner["speckles"],
            "laplacian_variance": winner["laplacian_variance"],
            "contrast_std": winner["contrast_std"],
            "errors": " | ".join(errors),
        }
        return number, "unchanged", str(dst), details, None

    # Copy to a temporary file, then atomically replace the output.
    temp_dst = dst.with_name(dst.name + ".tmp")

    try:
        shutil.copy2(source, temp_dst)
        temp_dst.replace(dst)
    finally:
        temp_dst.unlink(missing_ok=True)

    details = {
        "winner": str(source),
        "candidate_count": len(candidates),
        "score": winner["score"],
        "vertical_lines": winner["vertical_lines"],
        "speckles": winner["speckles"],
        "laplacian_variance": winner["laplacian_variance"],
        "contrast_std": winner["contrast_std"],
        "errors": " | ".join(errors),
    }
    return number, "done", str(dst), details, None


def main():
    parser = argparse.ArgumentParser(
        description="Choose the cleanest scan from repeated scans of each page."
    )
    parser.add_argument(
        "src", nargs="?", type=Path, default=SRC,
        help=f"main scan directory (default: {SRC})",
    )
    parser.add_argument(
        "--dst", type=Path, default=DST,
        help=f"output directory (default: {DST})",
    )
    parser.add_argument(
        "--workers", type=int, default=NUM_WORKERS,
        help=f"parallel worker processes (default: {NUM_WORKERS})",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="overwrite existing output files",
    )
    args = parser.parse_args()

    src = args.src
    dst = args.dst
    pages = discover_sources(src)

    if not pages:
        sys.exit(f"no page images found in {src} or {src.name}.tmp.*")

    # Only create work for pages with at least one source candidate.
    tasks = [
        (number, candidates, str(dst), args.force)
        for number, candidates in sorted(pages.items())
    ]

    dst.mkdir(parents=True, exist_ok=True)
    print(f"found {len(tasks)} pages; output: {dst}")
    print("scoring candidates (lower score is better)")

    report_path = dst / "045-find-best-scan.csv"
    rows = []
    done = 0
    unchanged = 0
    skipped = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        with tqdm(total=len(tasks), ncols=80, unit="page") as pbar:
            for number, status, output, details, error in executor.map(
                process_page, tasks
            ):
                if status == "done":
                    done += 1
                    rows.append({"page": number, "status": status, **details})
                elif status == "unchanged":
                    unchanged += 1
                    rows.append({
                        "page": number,
                        "status": status,
                        **(details or {}),
                    })
                elif status == "skipped":
                    skipped += 1
                    rows.append({
                        "page": number, "status": status,
                        "output": output, "message": error,
                    })
                else:
                    failed += 1
                    rows.append({
                        "page": number, "status": status,
                        "output": output, "message": error,
                    })
                    print(f"\npage {number}: {error}")
                pbar.update(1)

    # Write a compact audit log. Candidate-level scores are printed to the CSV
    # in a separate row for each candidate, including the selected winner.
    with report_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "page", "status", "winner", "candidate_count", "score",
            "vertical_lines", "speckles", "laplacian_variance",
            "contrast_std", "errors", "output", "message",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"done: updated {done} pages, "
        f"unchanged {unchanged}, "
        f"skipped {skipped}, "
        f"failed {failed}; "
        f"report: {report_path}"
    )


if __name__ == "__main__":
    main()

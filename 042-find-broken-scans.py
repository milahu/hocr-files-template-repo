#!/usr/bin/env python3


"""
find vertical gray lines in scanned images
caused by dirt on the sensor of the document scanner
"""


import os
import re
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse
import shlex

import cv2
import numpy as np
import psutil
from tqdm import tqdm

from _shared import (
    load_config,
    get_page_num,
    get_image_viewer_argstr,
)

config = load_config()


INPUT_DIR = Path("040-scan-pages")

# Detection parameters
WHITE_THRESHOLD = 245
WHITE_ROW_RATIO = 0.98

MIN_LINE_HEIGHT_RATIO = 0.25
MAX_LINE_WIDTH = 4

DARK_LINE_THRESHOLD = 8
# LIGHT_LINE_THRESHOLD = 8

MIN_SCORE = 80

MAX_PAGE_GAP = 4

IMAGE_VIEWER = "feh"
# TODO move to config
# IMAGE_VIEWER = config.image_viewer


def find_bottom_white_rectangle(gray):
    """
    Find the scanner-added white rectangle at the bottom.
    Returns the first row after the rectangle.
    """

    h, w = gray.shape

    white_rows = []

    for y in range(h - 1, -1, -1):

        row = gray[y]

        ratio = np.mean(row > WHITE_THRESHOLD)

        if ratio > WHITE_ROW_RATIO:
            white_rows.append(y)
        else:
            break

    if len(white_rows) == 0:
        return h

    return min(white_rows)


def make_detection(
        x_start,
        x_end,
        height_ratio,
        y_start_ratio,
        y_end_ratio,
        strength,
        # row_mask,
        # height,
    ):
    return {
        # TODO round x values?
        "x_start": x_start,
        "x_end": x_end,
        "height_ratio": float(height_ratio),
        "y_start_ratio": float(y_start_ratio),
        "y_end_ratio": float(y_end_ratio),
        "strength": float(strength),
    }


def detect_vertical_lines(gray):

    h, w = gray.shape

    # Ignore bottom scanner white rectangle
    scan_end = find_bottom_white_rectangle(gray)

    work = gray[:scan_end, :]

    # ------------------------------------------------------------
    # Method 1:
    # Column deviation from local neighborhood
    # ------------------------------------------------------------

    # smooth horizontal noise but keep vertical structures
    blurred = cv2.GaussianBlur(work, (5, 5), 0)

    # Difference to horizontal neighbors
    left = np.roll(blurred, 2, axis=1)
    right = np.roll(blurred, -2, axis=1)

    background = (left.astype(np.float32) + right.astype(np.float32)) / 2

    deviation = blurred.astype(np.float32) - background

    # A scanner line can be darker on gray background
    # and lighter on white paper.
    score = np.maximum(deviation, -deviation)

    # Average vertically
    column_score = np.mean(score, axis=0)

    # ------------------------------------------------------------
    # Candidate columns
    # ------------------------------------------------------------

    candidates = column_score > DARK_LINE_THRESHOLD

    # Group neighboring columns
    groups = []
    x = 0
    while x < w:
        if candidates[x]:
            start = x
            while x < w and candidates[x]:
                x += 1
            end = x
            if end - start <= MAX_LINE_WIDTH:
                groups.append((start, end))
        x += 1

    # ------------------------------------------------------------
    # Validate vertical persistence
    # ------------------------------------------------------------

    lines = []
    for start, end in groups:
        region = score[:, start:end]
        strong_pixels = np.mean(region > DARK_LINE_THRESHOLD)

        # height_ratio = np.sum(np.max(region, axis=1) > DARK_LINE_THRESHOLD) / work.shape[0]
        row_mask = (np.max(region, axis=1) > DARK_LINE_THRESHOLD)
        height_ratio = np.mean(row_mask)
        ys = np.where(row_mask)[0]

        if len(ys):
            y_start_ratio = ys[0] / work.shape[0]
            y_end_ratio = ys[-1] / work.shape[0]
        else:
            y_start_ratio = 1
            y_end_ratio = 0

        if (
            height_ratio > MIN_LINE_HEIGHT_RATIO
            and strong_pixels > 0.05
        ):
            strength = column_score[start:end].max()
            detection = make_detection(
                start, # x_start
                end - 1, # x_end
                height_ratio,
                y_start_ratio,
                y_end_ratio,
                strength,
            )
            lines.append(detection)

    return lines


def extend_cluster_with_neighbors(
        run,
        all_detections,
        tolerance=3,
    ):

    first_page = min(run.pages)
    last_page = max(run.pages)

    target_x = run.median_x

    detection_candidates = [
        d for d in all_detections
        if abs(
            ((d["x_start"]+d["x_end"])/2)
            -
            target_x
        ) <= tolerance
    ]

    for detection in detection_candidates:

        page_num = detection["page_num"]

        # previous page:
        # artifact should be at bottom
        if page_num == first_page - 2:
            if (
                detection["y_start_ratio"] > 0.3
                and detection["y_end_ratio"] > 0.8
            ):
                run.add_detection(detection)

        # next page:
        # artifact should be at top/middle
        if page_num == last_page + 2:
            if (
                detection["y_start_ratio"] < 0.2
                and detection["y_end_ratio"] < 0.9
            ):
                run.add_detection(detection)


def detect_page(filename):
    img = cv2.imread(str(filename), cv2.IMREAD_COLOR)
    if img is None:
        print("Cannot read:", filename)
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return detect_vertical_lines(gray)


class ArtifactGroup:
    def __init__(self):
        self.strong_detections = []
        self.weak_detections = []

    def add_detection(self, detection):
        self.strong_detections.append(detection)

    def add_weak_detection(self, detection):
        self.weak_detections.append(detection)

    @property
    def detections(self):
        return self.strong_detections + self.weak_detections

    @property
    def pages(self):
        return [d["page_num"] for d in self.detections]

    @property
    def strong_pages(self):
        return [d["page_num"] for d in self.strong_detections]

    @property
    def filenames(self):
        return [d["filename"] for d in self.detections]

    @property
    def height_ratios(self):
        # TODO self.detections or self.strong_detections
        return [d["height_ratio"] for d in self.detections]

    @property
    def strengths(self):
        # TODO self.detections or self.strong_detections
        return [d["strength"] for d in self.detections]

    @property
    def median_x(self):
        return np.median([
            (d["x_start"] + d["x_end"]) / 2
            for d in self.strong_detections
        ])

    @property
    def confidence(self):
        # TODO self.pages or self.strong_pages
        return len(set(self.pages))

    @property
    def sorted_pages(self):
        return sorted(set(self.pages))


def add_to_groups(groups, detection, tolerance=3):
    x = (detection["x_start"] + detection["x_end"]) / 2
    for cluster in groups:
        if abs(x - cluster.median_x) <= tolerance:
            # expand existing cluster
            cluster.add_detection(detection)
            return
    # start new cluster
    # cluster = ArtifactCluster(x)
    group = ArtifactGroup()
    group.add_detection(detection)
    groups.append(group)


def consecutive_page_runs(pages):
    pages = sorted(set(pages))
    if not pages:
        return []
    runs = []
    current = [pages[0]]
    for p in pages[1:]:
        if p == current[-1] + 2:
            current.append(p)
        else:
            runs.append(current)
            current = [p]
    runs.append(current)
    return runs


def continuity_score(run):
    runs = consecutive_page_runs(run.pages)
    if not runs:
        return 0
    longest = max(
        len(r)
        for r in runs
    )
    return longest


def detect_known_artifact(gray, x, width=5):
    h, w = gray.shape

    x1 = max(0, int(round(x - width)))
    x2 = min(w, int(round(x + width + 1)))

    strip = gray[:, x1:x2]

    # compare against nearby columns
    left = gray[:, max(0, x1-10):x1]
    right = gray[:, x2:min(w, x2+10)]

    if left.size == 0 or right.size == 0:
        return None

    background = ( np.mean(left, axis=1) + np.mean(right, axis=1)) / 2

    line = np.mean(strip, axis=1)

    diff = np.abs(line.astype(np.float32) - background)

    # lower threshold for known artifact position
    mask = diff > 3

    # `mask = diff > 3`
    # this will also detect large document changes.
    # Since this detector is only used at a known artifact X position,
    # I would add a width consistency check:
    if x2 - x1 > 20:
        return None
    # and maybe require a minimum strength:
    if np.mean(diff[mask]) < 3:
        return None
    # Otherwise a dark image area crossing that X coordinate could create false positives.

    ys = np.where(mask)[0]

    if len(ys) == 0:
        return None

    detection = make_detection(
        x_start = x1,
        x_end = x2 - 1,
        height_ratio = np.mean(mask),
        y_start_ratio = ys[0] / h,
        y_end_ratio = ys[-1] / h,
        strength = np.mean(diff),
    )
    return detection


def find_weak_scanner_artifacts(
        runs,
        all_detections,
        filename_of_page
    ):
    """
    Try to recover missing pages for known artifacts.
    """
    existing = {
        d["page_num"]
        for d in all_detections
    }

    # no. use: run.add_weak_detection(detection)
    # weak_detections = []

    for run in runs:

        # only process strong artifacts
        if not is_scanner_artifact(run):
            continue

        x = run.median_x

        pages = sorted(
            set(run.pages)
        )

        if len(pages) < 3:
            continue

        first = min(pages)
        last = max(pages)

        # expected pages for odd/even sequence
        expected = range(
            first,
            last + 1,
            2
        )

        missing = [
            p
            for p in expected
            if p not in pages
        ]
        # print(f"find_weak_scanner_artifacts: missing={missing}")

        for page_num in missing:
            filename = filename_of_page.get(page_num)
            if filename is None:
                # page_num can be out of range
                # print(f"find_weak_scanner_artifacts: no filename for missing page_num={page_num}")
                continue
            img = cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)
            # if img is None:
            #     continue
            detection = detect_known_artifact(img, x)
            # print(f"find_weak_scanner_artifacts: detect_known_artifact -> detection={detection}")
            if detection is None:
                continue
            detection["page_num"] = page_num
            detection["filename"] = filename
            detection["recovered"] = True
            # add to run
            if 0:
                # no! this produces too many false positives
                # so in run, we keep only strong scanner artifacts
                run.add_detection(detection)
            else:
                run.add_weak_detection(detection)
            # weak_detections.append(detection)

    # return weak_detections


def process_one_page(filename):
    """
    Worker process function.
    Returns (page_num, detections)
    """
    page_num = get_page_num(filename)
    detections = detect_page(filename)
    for detection in detections:
        detection["page_num"] = page_num
        detection["filename"] = filename
        detection["recovered"] = False
    return page_num, detections


def process_pass(files, filename_of_page):
    groups = []
    all_detections = []

    # find groups = call detect_vertical_lines
    num_workers = psutil.cpu_count(
        logical=False
    ) or 1
    tqdm_kwargs = dict(
        total=len(files),
        ncols=80,
        unit="page",
    )
    with ProcessPoolExecutor(
        max_workers=num_workers
    ) as executor:
        results = executor.map(process_one_page, files)
        with tqdm(**tqdm_kwargs) as pbar:
            for page_num, detections in results:
                all_detections.extend(detections)
                for detection in detections:
                    add_to_groups(groups, detection)
                pbar.update(1)

    # filter groups to runs
    runs = find_scanner_artifacts(groups)

    if 1:
        # try to find weak scanner artifacts
        # which were not detected in the frist pass
        # because the sensitivity of the first pass was too low
        # FIXME this returns many false positives
        # but this is still helpful to find weak scanner artifacts
        # that affect only midtone-filled areas on pages
        # weak_detections = find_weak_scanner_artifacts(
        find_weak_scanner_artifacts(
            runs,
            all_detections,
            filename_of_page
        )
        # print(f"480: weak_detections={weak_detections}")
        # weak_pages = [
        #     detection["page_num"] for detection in weak_detections
        # ]
        # moved to print_runs

    return runs, all_detections


def get_neighbor_pages(detected_pages, radius=1):
    """
    Return missing/neighbor pages around detected pages.
    Assumes same-parity page numbering (step=2).
    """
    detected = sorted(set(detected_pages))
    if not detected:
        return []

    step = 2
    neighbors = set()

    # gaps inside detected range
    for a, b in zip(
        detected[:-1],
        detected[1:]
    ):
        p = a + step
        while p < b:
            neighbors.add(p)
            p += step

    # outside range
    first = detected[0]
    last = detected[-1]
    for i in range(1, radius + 1):
        neighbors.add(
            first - step * i
        )
        neighbors.add(
            last + step * i
        )

    # remove invalid pages
    neighbors = {
        p
        for p in neighbors
        if p > 0
    }

    # do not return pages already detected
    neighbors -= set(detected)

    return sorted(neighbors)


def print_runs(runs, filename_of_page):
    if not runs:
        print("no runs")
        return
    # strongest artifacts first
    runs.sort(
        key=lambda c: c.confidence,
        reverse=True
    )

    # print image viewer command to view all pages of this parity
    sample_page_num = get_page_num(runs[0].filenames[0])
    is_odd_pages = (sample_page_num % 2 == 1)
    filenames = []
    if is_odd_pages:
        for page, filename in filename_of_page.items():
            page_num = get_page_num(filename)
            if page_num % 2 == 1:
                filenames.append(filename)
    else:
        # even pages
        for page, filename in filename_of_page.items():
            page_num = get_page_num(filename)
            if page_num % 2 == 0:
                filenames.append(filename)
    filenames.sort()
    parity_name = "odd" if is_odd_pages else "even"
    print(f"  view all {parity_name} pages:")
    print(f"    {get_image_viewer_argstr(filenames, config)}")

    for cluster_idx, run in enumerate(runs):
        filenames = sorted(set(map(str, run.filenames)))
        data = {
            # "score": scanner_artifact_score(run),
            # "is_scanner_artifact": is_scanner_artifact(run),
            "x": float(run.median_x),
            "pages": sorted(set(run.pages)),
            # "filenames": filenames,
            "num_pages": len(set(run.pages)),
            # "height_ratio": {
            #     "min": min(run.height_ratios),
            #     "median": float(np.median(run.height_ratios)),
            #     "max": max(run.height_ratios),
            # },
            "height_ratio.min": min(run.height_ratios),
            "height_ratio.mean": float(np.mean(run.height_ratios)),
            "height_ratio.median": float(np.median(run.height_ratios)),
            "height_ratio.max": max(run.height_ratios),
            "strength.median": float(np.median(run.strengths)),
        }
        print(f"run {cluster_idx + 1}: {data}")

        if 1:
            # filenames = sorted(set(map(str, run.filenames)))
            filenames = sorted(set([
                str(d["filename"])
                for d in run.strong_detections
            ]))
            # print(f"580: filenames={filenames}")
            print("  view strong pages:")
            print(f"    {get_image_viewer_argstr(filenames, config)}")

        if 1:
            weak_pages = sorted({
                d["page_num"]
                for d in run.weak_detections
            })
            if weak_pages:
                # print(f"Recovered missing pages: {weak_pages}")
                # print(f"Found weak artifacts on pages: {weak_pages}")
                if 1:
                    weak_filenames = [
                        detection["filename"] for detection in run.weak_detections
                    ]
                    # print(f"620: weak_filenames={list(map(str, weak_filenames))}")
                    # print("  view recovered pages:")
                    # print("  view weak artifacts pages:")
                    # print(f"Found weak artifacts:")
                    print("  view weak pages:")
                    print(f"  {get_image_viewer_argstr(weak_filenames, config)}")

        if 1:
            pages = sorted(set(run.pages))
            neighbor_pages = get_neighbor_pages(pages, radius=1)
            if neighbor_pages:
                neighbor_filenames = []
                for page_num in neighbor_pages:
                    filename = filename_of_page.get(page_num)
                    if filename is None:
                        continue
                    neighbor_filenames.append(filename)
                # print(f"590: neighbor_filenames={neighbor_filenames}")
                if neighbor_filenames:
                    print("  view neighbor pages:")
                    print(f"    {get_image_viewer_argstr(neighbor_filenames, config)}")


def height_profile_score(run):
    pages = list(
        zip(
            run.pages,
            run.height_ratios
        )
    )
    pages.sort()
    heights = [
        h
        for _, h in pages
    ]
    if len(heights) < 3:
        return 0
    middle = heights[1:-1]
    if not middle:
        return 0
    # strong middle, weaker edges
    if (
        np.median(middle) > 0.9
        and heights[0] < 0.8
    ):
        return 20
    return 0


def is_scanner_artifact(run):
    pages = len(set(run.pages))
    height = np.median(run.height_ratios)
    longest_run = continuity_score(run)
    return (
        pages >= 3
        and longest_run >= 3
        and height >= 0.8
    )


def find_scanner_artifacts(groups):
    """
    filter groups to runs
    """
    runs = []
    for group in groups:
        runs.extend(split_runs_of_group(group))
    runs = list(filter(is_scanner_artifact, runs))
    return runs


def split_runs_of_group(group, max_page_gap=8):
    """
    Split one X-position group into continuous page runs.

    Strong detections remain strong.
    Weak detections remain weak.
    """

    detections = []

    for detection in group.strong_detections:
        detections.append((False, detection))

    for detection in group.weak_detections:
        detections.append((True, detection))

    if not detections:
        return []

    detections.sort(
        key=lambda t: t[1]["page_num"]
    )

    detection_runs = []
    current = [detections[0]]

    for is_weak, detection in detections[1:]:

        prev_page = current[-1][1]["page_num"]

        if detection["page_num"] <= prev_page + max_page_gap:
            current.append((is_weak, detection))
        else:
            detection_runs.append(current)
            current = [(is_weak, detection)]

    detection_runs.append(current)

    group_runs = []
    for detection_run in detection_runs:
        # artifact = ArtifactRun()
        artifact = ArtifactGroup()
        for is_weak, detection in detection_run:
            if is_weak:
                artifact.add_weak_detection(detection)
            else:
                artifact.add_detection(detection)
        group_runs.append(artifact)

    return group_runs


def compress_filenames(filenames):
    # compress file paths
    # a: feh 040-scan-pages/079.tiff 040-scan-pages/083.tiff
    # b: ( cd 040-scan-pages && feh 079.tiff 083.tiff; )
    files_dir = None
    parents = set()
    for filename in filenames:
        parents.add(Path(filename).parent)
    if len(parents) == 1:
        parent = parents.pop()
        if parent != Path("."):
            files_dir = parent
            filenames = list(map(lambda p: Path(Path(p).name), filenames))
    return files_dir, filenames


def parse_args():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--odd",
        action="store_true",
        help="process only odd pages",
    )
    group.add_argument(
        "--even",
        action="store_true",
        help="process only even pages",
    )
    # TODO implement: filter by page number
    r'''
    group.add_argument(
        "--pages",
        help="process only these pages. example: 1-10,21-30",
    )
    '''
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="image files to process",
    )
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    files = args.files

    # print(f"args.files={args.files}")

    if files:
        files = list(map(Path, files))
    else:
        files = sorted(INPUT_DIR.glob(f"*.{config.scan_format}"))

    filename_of_page = {
        get_page_num(f): f
        for f in files
    }

    # print(f"files={files}")

    odd_files = []
    even_files = []

    for f in files:
        page_num = get_page_num(f)
        # print(f"f={f}  page_num={page_num}")
        if page_num % 2 == 1:
            if not args.even:
                odd_files.append(f)
            # else: process only even pages
        else:
            if not args.odd:
                even_files.append(f)
            # else: process only odd pages

    if odd_files or even_files:
        num_workers = psutil.cpu_count(logical=False) or 1
        print(f"Using {num_workers} worker processes")
    else:
        print("no input files")
        return

    if odd_files:
        print()
        print(f"Processing odd pages")
        odd_groups, odd_detections = process_pass(odd_files, filename_of_page)
        # print("runs:")
        print_runs(odd_groups, filename_of_page)

    if even_files:
        print()
        print(f"Processing even pages")
        runs, detections = process_pass(even_files, filename_of_page)
        # print("runs:")
        print_runs(runs, filename_of_page)


if __name__ == "__main__":
    main()

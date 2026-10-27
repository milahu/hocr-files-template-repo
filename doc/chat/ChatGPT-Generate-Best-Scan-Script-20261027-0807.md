# Generate Best Scan Script

**User:** Anonymous  
**Created:** 2026/10/27 8:00:04  
**Updated:** 2026/10/27 8:05:52  
**Exported:** 2026/10/27 8:07:04  
**Link:** [<https://chatgpt.com/c/6ae04bf4-5198-83eb-87ad-1b8cc2bf7995>](https://chatgpt.com/c/6ae04bf4-5198-83eb-87ad-1b8cc2bf7995)

## Prompt:

10/27/2026, 8:05:07 AM

help me explore this idea for my book-scanning pipeline at `https://github.com/milahu/hocr-files-template-repo`

one problem is: document scanners can introduce multiple types of noise:

1.  vertical gray lines from dust particles on the CIS sensor
2.  small gray speckles from dust on the paper

my current solution is to run the script `042-find-broken-scans.py` which tries to detect the "vertical gray lines from dust particles on the CIS sensor", but that script can fail in some cases, so i still need (time-consuming) manual inspection of all scanned pages...

now my idea is this: scan the complete book multiple times, say for example 5 times, and then have some script align and compare the 5 different versions to find the best version for each page

or maybe the script could even generate synthetic images, by piecing together the best parts of different versions... but that sounds rather complex, so i would not try that first

does this sound like a solvable problem? if yes, please generate a python script `045-find-best-scan.py` to process multiple scans of the same book

all the input images are stored in the directories `040-scan-pages` and `040-scan-pages.tmp.${timestamp}`, for example all scans of page 10 are stored at `040-scan-pages/010.tiff` and `040-scan-pages.tmp.${timestamp}/010.tiff`

the output images should be copied to the directory `045-find-best-scan`

one challenge here will be to align multiple scans of the same page, because document scanners can introduce non-linear Y-errors (positive and negative stretching parallel to the paper feed direction)

here is another script from my repo. your generated script should be similar to this script:

062-compress.py

``` python
#!/usr/bin/env python3

# compress the scans for publishing

import argparse
import os
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import psutil
from PIL import Image
from tqdm import tqdm

from _shared import (
    load_config,
    get_page_num,
    remove_done_files,
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
# os.chdir(SCRIPT_DIR)

datetime_str = (
    datetime.now()
    .astimezone()
    .isoformat(timespec="seconds")
    .replace(":", "-")
)

src = Path("060-rotate-crop")
dst = Path(f"{src}-compressed")
# dst = src  # replace files in src

jpeg_quality = 90

num_workers = psutil.cpu_count(logical=False) or 1

config = load_config()

scan_format = config.scan_format
image_format = config.compresed_image_format
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

    if f.suffix == f".{image_format}":
        # the source image is already compressed
        if not replace:
            # copy
            # print(f"cp {f} {f_dst}")
            shutil.copy(f, f_dst)
        done, message = True, None
        return done, message

    size_a = f.stat().st_size

    with Image.open(f) as img:

        compression = TIFF_COMPRESSION.get(
            img.tag_v2.get(259),
            str(img.tag_v2.get(259, "unknown")),
        )

        #
        # Preserve original disabled logic
        #
        if False:
            page_num = get_page_num(f)
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
    # if size_b < size_a:
    if False:
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

def main():
    global src, dst

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "src",
        nargs="?",
        type=Path,
        default=src,
        help=f"source directory (default: {src})",
    )
    args = parser.parse_args()

    src = args.src
    dst = Path(f"{src}-compressed")

    files = [
        f
        for f in sorted(src.glob("*"))
        # if re.fullmatch(rf"\d+\.{re.escape(scan_format)}", f.name)
        # if f.name.endswith(f".{scan_format}")
        if f.suffix in (f".{scan_format}", f".{image_format}")
    ]

    files = remove_done_files(files, dst, dst_suffix=f".{image_format}")

    if not files:
        print("nothing to do")
        sys.exit()

    print(f"writing {dst}")

    # The output directory is based on the selected source directory.
    dst.mkdir(parents=True, exist_ok=True)

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

    tqdm_kwargs = dict(
        total=len(tasks),
        ncols=80,
        unit="page",
    )

    with (
        ProcessPoolExecutor(max_workers=num_workers) as executor,
        tqdm(**tqdm_kwargs) as pbar,
    ):
        for done, message in executor.map(process_image, tasks):
            if message:
                print()
                print(message)
            if done:
                num_done += 1
            pbar.update(1)

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

    r'''
    print(f"done. compressed {num_done} images")
    '''

if __name__ == "__main__":
    main()
```

## Response:

10/27/2026, 8:05:13 AM

> **Building the script**
>
> I’m preparing a standalone Python script that searches a folder, scores scanned pages for streaks, speckles, and sharpness, optionally aligns them, and selects the strongest candidate with diagnostics.
>
> **Built the script**
>
> **Searched 9 websites**
>
> Worked for 10s

Yes - this is a very solvable problem, and I think your idea is particularly promising for a book-scanning pipeline. **The key is to treat repeated scans as multiple observations of the same physical page, rather than merely as duplicate files.**

Your proposed first version should select the best complete scan for each page. Once that works reliably, you can consider combining image regions from different scans.

## 1. Recommended approach

<box gap={2}>
  <row align=start gap={3}>
    <AsyncImage query="book scanner grayscale scanned book page vertical dark gray streak dust sensor artifact" aspectRatio="3:4" maxWidth="104px"/>
    <box flex="1" gap={1}>
      **Step 1 - Gather all scans of each page**

      Find `010.tiff` in `040-scan-pages/` and every matching numbered file in `040-scan-pages.tmp.*`. Treat them as candidate versions of page 10.
    </box>

  </row>
  <divider color="subtle"/>
  <row align=start gap={3}>
    <AsyncImage query="grayscale scanned book page clean black text white paper high contrast" aspectRatio="3:4" maxWidth="104px"/>
    <box flex="1" gap={1}>
      **Step 2 - Align the candidates**

      First correct translation, rotation, and small geometric differences. For your scanner's nonlinear Y-errors, use local image registration: divide the page into horizontal bands, estimate the horizontal and vertical displacement of each band, and interpolate a smooth displacement field.
    </box>

  </row>
  <divider color="subtle"/>
  <row align=start gap={3}>
    <AsyncImage query="black and white scanned book page tiny gray specks dust paper texture" aspectRatio="3:4" maxWidth="104px"/>
    <box flex="1" gap={1}>
      **Step 3 - Score each candidate**

      Evaluate residual vertical streaks, isolated speckles, text sharpness, contrast, and alignment quality. Select the candidate with the best overall score, while retaining the original pixels rather than introducing interpolation artifacts unnecessarily.
    </box>

  </row>
  <divider color="subtle"/>
  <row align=start gap={3}>
    <AsyncImage query="clean archival digitized book page black serif text white background" aspectRatio="3:4" maxWidth="104px"/>
    <box flex="1" gap={1}>
      **Step 4 - Write the winner**

      Copy the chosen scan to `045-find-best-scan/010.tiff`. Record which source won and its scores, so suspicious choices can be audited without manually inspecting every page.
    </box>

  </row>
</box>

There is one important distinction: **alignment and image quality should be evaluated separately.** A scan can have fewer streaks but worse focus, or look cleaner simply because it is slightly misaligned.

## 2. The generated script

I've created `045-find-best-scan.py` as a standalone Python script following the general structure of your existing pipeline.

<box border radius="lg" padding={4} gap={3}>
  <row align="center" gap={3}>
    <box background="surface-secondary" radius="md" padding={3}>
      <icon name="file-code" size="2xl"/>
    </box>
    <box flex="1" gap={1}>
      **045-find-best-scan.py**
      <text color="secondary" size="sm">Python · 318 lines · Ready to test</text>
    </box>
  </row>
  [Download 045-find-best-scan.py](sandbox:/mnt/data/045-find-best-scan.py)
</box>

### What it does

- Discovers page images in `040-scan-pages/` and all matching `040-scan-pages.tmp.*` directories.
- Groups images by their numeric page prefix.
- Scores each candidate for vertical streak artifacts, isolated speckles, sharpness, and contrast.
- Selects the lowest-scoring candidate for each page.
- Copies the original image without re-encoding it.
- Uses `ProcessPoolExecutor` and `tqdm` for parallel processing and progress reporting.
- Writes a CSV report with the selected source and its quality scores.
- Skips existing output files by default, with a `--force` option to overwrite them.

### How to run it

Save the script in your repository root and install its dependencies if necessary:

Then run:

For a fresh selection pass that replaces existing results:

The result will look like this:

You can also select a different main input directory:

The script will look for timestamped repeat-scan directories alongside that directory.

## 3. An important limitation of this first version

The scoring algorithm is a **heuristic, not a trained or calibrated quality model**. In particular:

- Vertical text strokes can resemble sensor streaks.
- Punctuation and fine text can resemble isolated speckles.
- Sharpness measurements can favor noisy images.
- A scan with a large clean area but one severe streak might still win.

Consequently, I would initially run this on a small batch and compare the results with your manually selected best scans. The weights near the top of the script are intended to be easy to tune.

One deliberate simplification: this version does **not** implement nonlinear alignment. It evaluates each candidate independently, so alignment isn't required for its current selection strategy. It also doesn't yet use your existing `042-find-broken-scans.py` detector directly; that could be a valuable next improvement. Your existing detector already targets the sensor-streak problem specifically. \<Cite refs={\["turn560131view0"\]}/\>

## 4. How I'd improve it next

My preferred development sequence would be:

1.  **Calibrate candidate selection.** Run five scans of perhaps 50 pages, then compare the script's choices against your manual choices. Save metrics for every candidate, not just the winner.
2.  **Reuse the broken-scan detector.** Integrate its streak detections into the quality score so the new script benefits from your existing work.
3.  **Add nonlinear registration.** Use local matching in horizontal bands to estimate vertical-feed displacement and other local geometric differences. Reject unreliable alignments instead of trusting them blindly.
4.  **Explore synthetic reconstruction.** Once alignment is reliable, combine complementary regions from different scans, using a per-pixel or per-region quality mask.

The fourth step could ultimately produce better images than choosing a single scan, but your instinct to defer it is right. Selecting the best complete scan is a much simpler first milestone.

**My recommendation:** test the generated script on a small subset first. The most valuable next iteration would be to use the existing `042-find-broken-scans.py` logic as a scoring component, rather than trying to invent a completely independent detector for the same artifact.

## Response:

10/27/2026, 8:05:36 AM

------------------------------------------------------------------------

Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)

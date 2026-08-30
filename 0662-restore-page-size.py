#!/usr/bin/env python3

# restore the original book page size
#
# this runs
# - after 065-remove-page-borders.py
#
# destructive book scanning can remove part of the original inner page edge:
# - the book is unbinded with a guillotine cutter
# - the loose pages are scanned with a document scanner
#
# config.page_width_px and config.page_height_px define the original page aspect ratio.
#
# scanned pages can have slightly different heights, so first we normalize all
# pages to the average scanned page height:
# - crop equally from the top and bottom when a page is too tall
# - add white pixels equally to the top and bottom when a page is too short
#
# from this normalized height and the original page aspect ratio, we compute the
# normalized page width.
#
# the missing width is restored at the inner page edge:
# - odd-numbered pages:  add pixels on the left
# - even-numbered pages: add pixels on the right
#
# normally the missing area is filled with white.
#
# optionally, when config.reconstruct_inner_edge_pattern is True, try to
# reconstruct a simple horizontally repeating pattern near the inner edge.
#
# this is intended for simple borderless-printing cases such as:
# - solid color rectangles
# - horizontally periodic halftone / dither patterns
# - other approximately horizontally repeating image regions
#
# pattern recognition uses a blurred grayscale representation to tolerate small
# amounts of scan noise, but reconstruction uses the original-resolution pixels
# so the original dither / halftone structure is preserved.
#
# 065-remove-page-borders.py can leave a transparent triangle at the inner edge.
# When this happens:
# - detect the maximum extent of the transparent strip
# - crop the whole image to that maximum extent
# - thereby remove the transparent triangle and the corresponding crooked strip
#   from the remaining page
# - make the resulting inner edge vertical
# - composite any remaining transparency onto white and remove the alpha channel

import os
import sys
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import psutil
from PIL import Image, ImageFilter
from tqdm import tqdm

from _shared import (
    load_config,
    get_page_num,
    remove_done_files,
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

src = Path("065-remove-page-borders")
dst = Path(Path(__file__).stem)
# dst = src  # replace files in src

num_workers = psutil.cpu_count(logical=False) or 1

config = load_config()

scan_format = config.scan_format
# image_format = config.image_format
image_format = config.scan_format

page_width = config.page_width_px
page_height = config.page_height_px

reconstruct_inner_edge_pattern = getattr(
    config,
    "reconstruct_inner_edge_pattern",
    False,
)

if not scan_format:
    sys.exit("error: scan_format not defined")

if not image_format:
    sys.exit("error: image_format not defined")

if not page_width:
    sys.exit("error: page_width not defined")

if not page_height:
    sys.exit("error: page_height not defined")

if page_width <= 0:
    sys.exit("error: page_width must be greater than zero")

if page_height <= 0:
    sys.exit("error: page_height must be greater than zero")

# -----------------------------------------------------------------------------
# Pattern reconstruction parameters
# -----------------------------------------------------------------------------

# Maximum missing inner-edge width that we try to reconstruct.
#
# The actual missing width is usually determined by:
#
#     target_width - scanned_width
#
# so this is only a sanity limit for pattern analysis.
MAX_RECONSTRUCT_WIDTH = 1000

# Search for repeating periods between these values.
#
# A period of 1 is useful for perfectly uniform regions.
# Larger periods are needed for halftone / dither structures.
MIN_PATTERN_PERIOD = 1
MAX_PATTERN_PERIOD = 512

# Width of the area near the inner edge used to recognize the pattern.
#
# This should contain several repetitions of the pattern.
PATTERN_SEARCH_WIDTH = 2048

# Gaussian blur radius used only for pattern recognition.
#
# Reconstruction always uses original pixels.
PATTERN_BLUR_RADIUS = 1.0

# Maximum vertical downsampling factor used for pattern analysis.
#
# Averaging vertically reduces noise and converts the two-dimensional problem
# into a robust one-dimensional horizontal signal.
PATTERN_ANALYSIS_MAX_HEIGHT = 512

# Minimum normalized correlation required to accept a nontrivial repeating
# pattern.
PATTERN_MIN_CORRELATION = 0.80

# Maximum normalized boundary error after phase alignment.
#
# Smaller is stricter.
PATTERN_MAX_BOUNDARY_ERROR = 0.20

# -----------------------------------------------------------------------------

replace = src == dst

if not replace:
    dst.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Alpha handling
# -----------------------------------------------------------------------------

def has_alpha_channel(img):
    return (
        img.mode in ("RGBA", "LA", "RGBa", "La")
        or (
            img.mode == "P"
            and "transparency" in img.info
        )
    )


def get_alpha_channel(img):
    """
    Return the alpha channel as an L image.

    P images with transparency are converted to RGBA first.
    """

    if img.mode == "P" and "transparency" in img.info:
        img = img.convert("RGBA")

    if img.mode in ("RGBA", "RGBa"):
        return img.getchannel("A")

    if img.mode in ("LA", "La"):
        return img.getchannel("A")

    return None


def cleanup_transparent_inner_edge(img, inside_on_left):
    """
    Remove a transparent triangle / strip at the inner edge.

    Find the maximum horizontal extent of pixels that are not fully opaque.

    For an inner edge on the left:
        crop everything from x=0 through outmost_inside_x.

    For an inner edge on the right:
        crop everything from outmost_inside_x through the right edge.

    This follows the same principle as:

        outmost_inside_x

    in 065-remove-page-borders.py.

    The result has a perfectly vertical inner edge.
    """

    alpha = get_alpha_channel(img)

    if alpha is None:
        return img

    alpha_np = np.asarray(alpha)

    # Fully opaque is 255.
    #
    # Any value below 255 belongs to the transparent / partially transparent
    # strip. Using < 255 also catches half-transparent interpolation pixels.
    transparent = alpha_np < 255

    if not np.any(transparent):
        return img

    height, width = transparent.shape

    if inside_on_left:

        # For every row, find the rightmost transparent pixel.
        #
        # -1 means: no transparent pixel on that row.
        x_positions = np.where(
            transparent,
            np.arange(width)[None, :],
            -1,
        )

        outmost_inside_x = int(x_positions.max())

        # Same semantics as:
        #
        # crop_x = ceil(outmost_inside_x)
        # warped = warped[:, crop_x:, :]
        #
        crop_x = outmost_inside_x + 1

        if crop_x >= width:
            # Something is seriously wrong: the whole image is transparent.
            raise ValueError(
                "transparent inner-edge cleanup would remove the whole image"
            )

        return img.crop(
            (
                crop_x,
                0,
                width,
                height,
            )
        )

    # Inner edge is on the right.

    # For every row, find the leftmost transparent pixel.
    #
    # width means: no transparent pixel on that row.
    x_positions = np.where(
        transparent,
        np.arange(width)[None, :],
        width,
    )

    outmost_inside_x = int(x_positions.min())

    # Same semantics as:
    #
    # crop_x = floor(outmost_inside_x)
    # warped = warped[:, :crop_x, :]
    #
    crop_x = outmost_inside_x

    if crop_x <= 0:
        raise ValueError(
            "transparent inner-edge cleanup would remove the whole image"
        )

    return img.crop(
        (
            0,
            0,
            crop_x,
            height,
        )
    )


def remove_alpha_on_white(img):
    """
    Replace transparent pixels with white and remove the alpha channel.
    """

    if not has_alpha_channel(img):
        return img

    if img.mode == "P":
        img = img.convert("RGBA")

    if img.mode in ("RGBA", "RGBa"):
        if img.mode == "RGBa":
            img = img.convert("RGBA")

        background = Image.new("RGB", img.size, "white")
        background.paste(
            img,
            mask=img.getchannel("A"),
        )
        return background

    if img.mode in ("LA", "La"):
        if img.mode == "La":
            img = img.convert("LA")

        background = Image.new("L", img.size, 255)
        background.paste(
            img.getchannel("L"),
            mask=img.getchannel("A"),
        )
        return background

    return img.convert("RGB")


# -----------------------------------------------------------------------------
# Image helpers
# -----------------------------------------------------------------------------

def get_background_color(mode):

    if mode == "1":
        return 1

    if mode == "L":
        return 255

    if mode == "RGB":
        return "white"

    if mode == "CMYK":
        return (0, 0, 0, 0)

    return "white"


def make_canvas(mode, size):
    return Image.new(
        mode,
        size,
        get_background_color(mode),
    )


# -----------------------------------------------------------------------------
# Height normalization
# -----------------------------------------------------------------------------

def normalize_height(img, target_height):
    """
    Normalize image height to target_height.

    If the image is too tall, remove pixels from the top and bottom as evenly
    as possible.

    If the image is too short, add white pixels to the top and bottom as evenly
    as possible.
    """

    width, height = img.size

    if height == target_height:
        return img

    if height > target_height:

        pixels_to_remove = height - target_height

        crop_top = pixels_to_remove // 2
        crop_bottom = pixels_to_remove - crop_top

        return img.crop(
            (
                0,
                crop_top,
                width,
                height - crop_bottom,
            )
        )

    pixels_to_add = target_height - height

    pad_top = pixels_to_add // 2

    result = make_canvas(
        img.mode,
        (
            width,
            target_height,
        ),
    )

    result.paste(
        img,
        (
            0,
            pad_top,
        ),
    )

    return result


# -----------------------------------------------------------------------------
# Pattern analysis
# -----------------------------------------------------------------------------

def image_to_pattern_signal(
    img,
    inside_on_left,
    search_width,
):
    """
    Convert the area near the inner edge into a robust one-dimensional
    horizontal signal.

    Steps:
    - select a strip near the inner edge
    - convert to grayscale
    - blur slightly to reduce scan noise
    - average vertically

    Vertical averaging is especially useful for horizontally repeating patterns
    with white top/bottom margins: the signal still retains horizontal structure
    while random scan noise and individual dither-dot noise are reduced.
    """

    width, height = img.size

    search_width = min(
        search_width,
        width,
    )

    if search_width < 2:
        return None

    if inside_on_left:
        strip = img.crop(
            (
                0,
                0,
                search_width,
                height,
            )
        )
    else:
        strip = img.crop(
            (
                width - search_width,
                0,
                width,
                height,
            )
        )

    if strip.mode != "L":
        strip = strip.convert("L")

    if PATTERN_BLUR_RADIUS > 0:
        strip = strip.filter(
            ImageFilter.GaussianBlur(
                radius=PATTERN_BLUR_RADIUS
            )
        )

    strip_np = np.asarray(
        strip,
        dtype=np.float32,
    )

    # Limit analysis cost on very tall scans.
    if strip_np.shape[0] > PATTERN_ANALYSIS_MAX_HEIGHT:

        step = (
            strip_np.shape[0]
            / PATTERN_ANALYSIS_MAX_HEIGHT
        )

        rows = np.arange(
            PATTERN_ANALYSIS_MAX_HEIGHT
        )

        row_indices = np.minimum(
            (
                rows * step
                + step / 2
            ).astype(int),
            strip_np.shape[0] - 1,
        )

        strip_np = strip_np[row_indices, :]

    # Average vertically.
    signal = strip_np.mean(axis=0)

    # Reverse the signal for right inner edges so index 0 always means
    # "closest to the inner edge".
    if not inside_on_left:
        signal = signal[::-1]

    return signal


def normalized_correlation(a, b):
    """
    Pearson-style normalized correlation.
    """

    if len(a) != len(b) or len(a) == 0:
        return 0.0

    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)

    a = a - a.mean()
    b = b - b.mean()

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator == 0:
        # Perfectly uniform regions are a special case.
        #
        # They are handled separately by reconstruction.
        return 0.0

    return float(
        np.dot(a, b)
        / denominator
    )


def find_pattern_period(signal):
    """
    Find the strongest horizontal repetition period.

    Compare:

        signal[0:n-period]

    with:

        signal[period:n]

    using normalized correlation.

    This is a simple autocorrelation-based period detector.

    Return:

        (period, correlation)

    or:

        (None, correlation)

    when no sufficiently strong periodic structure is found.
    """

    n = len(signal)

    if n < 4:
        return None, 0.0

    signal_std = float(signal.std())

    # Uniform or nearly uniform signal:
    #
    # treat it as period 1. This supports color-filled rectangles.
    if signal_std < 1.0:
        return 1, 1.0

    max_period = min(
        MAX_PATTERN_PERIOD,
        n // 2,
    )

    min_period = min(
        MIN_PATTERN_PERIOD,
        max_period,
    )

    best_period = None
    best_correlation = -1.0

    for period in range(
        min_period,
        max_period + 1,
    ):

        correlation = normalized_correlation(
            signal[:-period],
            signal[period:],
        )

        if correlation > best_correlation:
            best_correlation = correlation
            best_period = period

    if best_correlation < PATTERN_MIN_CORRELATION:
        return None, best_correlation

    return best_period, best_correlation


# -----------------------------------------------------------------------------
# Pattern reconstruction
# -----------------------------------------------------------------------------

def pattern_boundary_error(
    source_np,
    period,
    inside_on_left,
):
    """
    Measure how well the periodic texture connects across one period boundary.

    This uses original-resolution pixels.

    For a left inner edge, compare:

        [0 : period]

    with:

        [period : 2*period]

    For a right inner edge, compare the corresponding region at the right side.

    The result is normalized by local image contrast.
    """

    height, width = source_np.shape[:2]

    if width < period * 2:
        return float("inf")

    if inside_on_left:
        a = source_np[:, 0:period]
        b = source_np[:, period:period * 2]
    else:
        a = source_np[:, width - period:width]
        b = source_np[
            :,
            width - period * 2:width - period,
        ]

    a = a.astype(np.float32)
    b = b.astype(np.float32)

    difference = np.mean(
        np.abs(a - b)
    )

    contrast = np.std(
        source_np.astype(np.float32)
    )

    if contrast < 1.0:
        return 0.0

    return float(
        difference
        / (contrast + 1e-6)
    )


def reconstruct_inner_edge(
    img,
    missing_width,
    inside_on_left,
):
    """
    Try to reconstruct missing pixels at the inner edge.

    Returns:

        reconstructed_img, reconstructed

    reconstructed is True only when a periodic pattern was detected with
    sufficient confidence.

    The reconstructed pixels are copied from the original-resolution image.
    No blur is applied to the final reconstructed pixels.

    For a detected period P, pixels are extended periodically:

        f(x) = f(x mod P)

    with the phase anchored at the existing inner edge.

    This means dither / halftone dots can continue through boundaries between
    repeated periods, rather than repeatedly pasting a rectangle with an
    arbitrary seam.
    """

    if missing_width <= 0:
        return img, False

    if missing_width > MAX_RECONSTRUCT_WIDTH:
        return img, False

    signal = image_to_pattern_signal(
        img,
        inside_on_left=inside_on_left,
        search_width=PATTERN_SEARCH_WIDTH,
    )

    if signal is None:
        return img, False

    period, correlation = find_pattern_period(
        signal
    )

    if period is None:
        return img, False

    img_np = np.asarray(img)

    if img_np.shape[1] < period * 2:
        return img, False

    boundary_error = pattern_boundary_error(
        img_np,
        period,
        inside_on_left,
    )

    if boundary_error > PATTERN_MAX_BOUNDARY_ERROR:
        return img, False

    width, height = img.size

    result = make_canvas(
        img.mode,
        (
            width + missing_width,
            height,
        ),
    )

    if inside_on_left:

        # Existing image begins at x=missing_width.
        result.paste(
            img,
            (
                missing_width,
                0,
            ),
        )

        # Extend the pattern backwards.
        #
        # The existing pixel at source x=0 corresponds to result
        # x=missing_width.
        #
        # For destination x:
        #
        # source_x = (x - missing_width) mod period
        #
        # This produces a continuous periodic extension through the boundary.
        y_indices = np.arange(height)[:, None]

        source_x = (
            np.arange(missing_width)
            - missing_width
        ) % period

        if img_np.ndim == 2:
            reconstructed_np = img_np[
                y_indices,
                source_x[None, :],
            ]
        else:
            reconstructed_np = img_np[
                y_indices,
                source_x[None, :],
                :,
            ]

        reconstructed = Image.fromarray(
            reconstructed_np.astype(
                img_np.dtype,
                copy=False,
            ),
            mode=img.mode,
        )

        result.paste(
            reconstructed,
            (
                0,
                0,
            ),
        )

    else:

        # Existing image remains at x=0.
        result.paste(
            img,
            (
                0,
                0,
            ),
        )

        # Extend the pattern forwards.
        #
        # Source x=width-period ... width-1 is treated as the final period.
        source_x = (
            np.arange(missing_width)
            % period
            + width
            - period
        )

        y_indices = np.arange(height)[:, None]

        if img_np.ndim == 2:
            reconstructed_np = img_np[
                y_indices,
                source_x[None, :],
            ]
        else:
            reconstructed_np = img_np[
                y_indices,
                source_x[None, :],
                :,
            ]

        reconstructed = Image.fromarray(
            reconstructed_np.astype(
                img_np.dtype,
                copy=False,
            ),
            mode=img.mode,
        )

        result.paste(
            reconstructed,
            (
                width,
                0,
            ),
        )

    return result, True


# -----------------------------------------------------------------------------
# Restore page width
# -----------------------------------------------------------------------------

def restore_width(
    img,
    target_width,
    page_num,
):
    """
    Restore the missing width at the inner page edge.

    Assumes normal left-to-right pagination:

    - odd page:  right-hand page -> inner edge on the left
    - even page: left-hand page  -> inner edge on the right
    """

    width, height = img.size

    if width == target_width:
        return img, False

    if width > target_width:
        raise ValueError(
            f"page {page_num}: image width {width} "
            f"is greater than restored page width {target_width}"
        )

    missing_width = target_width - width

    inside_on_left = (
        page_num % 2 == 1
    )

    if reconstruct_inner_edge_pattern:

        reconstructed_img, reconstructed = reconstruct_inner_edge(
            img,
            missing_width=missing_width,
            inside_on_left=inside_on_left,
        )

        if reconstructed:
            return reconstructed_img, True

    # Fallback: restore missing width with white pixels.

    result = make_canvas(
        img.mode,
        (
            target_width,
            height,
        ),
    )

    if inside_on_left:

        result.paste(
            img,
            (
                missing_width,
                0,
            ),
        )

    else:

        result.paste(
            img,
            (
                0,
                0,
            ),
        )

    return result, False


# -----------------------------------------------------------------------------
# Scanner aspect-ratio correction
# -----------------------------------------------------------------------------

def correct_scanner_aspect_ratio(img):
    """
    Correct scanner-induced aspect-ratio distortion.

    The document scanner's scan-axis dimension may be geometrically inaccurate.

    When config.do_rotate is True:
        The physical page was scanned rotated and subsequently rotated back.
        Therefore the scanner's Y-axis error appears as an error in image width.
        Height is assumed correct; width is corrected.

    When config.do_rotate is False:
        Width is assumed correct; height is corrected.

    The corrected image has the expected aspect ratio of the unbinded page:

        config.unbinded_page_width_mm / config.page_height_mm
    """

    width, height = img.size

    target_aspect = (
        config.unbinded_page_width_mm
        / config.page_height_mm
    )

    if config.do_rotate:

        # Height is correct.
        target_width = round(
            height * target_aspect
        )

        if target_width == width:
            return img

        return img.resize(
            (
                target_width,
                height,
            ),
            resample=Image.Resampling.LANCZOS,
        )

    # Width is correct.
    target_height = round(
        width / target_aspect
    )

    if target_height == height:
        return img

    return img.resize(
        (
            width,
            target_height,
        ),
        resample=Image.Resampling.LANCZOS,
    )


# -----------------------------------------------------------------------------
# Process one image
# -----------------------------------------------------------------------------

def process_image(args):

    (
        f,
        dst,
        target_height,
        target_width,
        reconstruct_inner_edge_pattern,
    ) = args

    f = Path(f)
    dst = Path(dst)

    page_num = get_page_num(f)

    f_dst = dst / f"{f.stem}.{image_format}"

    # Normal left-to-right book pagination.
    inside_on_left = (page_num % 2 == 1)

    with Image.open(f) as img:

        img.load()

        # ---------------------------------------------------------------------
        # Alpha handling
        # ---------------------------------------------------------------------

        if has_alpha_channel(img):

            # First remove the maximum extent of the transparent inner-edge
            # triangle. This makes the inner edge vertical.
            img = cleanup_transparent_inner_edge(
                img,
                inside_on_left=inside_on_left,
            )

            # Then replace any remaining transparency with white and remove the
            # alpha channel.
            img = remove_alpha_on_white(
                img,
            )

        # ---------------------------------------------------------------------
        # Correct scanner aspect-ratio distortion
        # ---------------------------------------------------------------------

        img = correct_scanner_aspect_ratio(img)

        # ---------------------------------------------------------------------
        # Normalize height
        # ---------------------------------------------------------------------

        img = normalize_height(
            img,
            target_height,
        )

        # ---------------------------------------------------------------------
        # Restore width
        # ---------------------------------------------------------------------

        img, reconstructed = restore_width(
            img,
            target_width,
            page_num,
        )

        # ---------------------------------------------------------------------
        # Save
        # ---------------------------------------------------------------------

        save_kwargs = {}

        if image_format.lower() in (
            "jpg",
            "jpeg",
        ):

            if img.mode == "RGBA":
                img = img.convert("RGB")
            elif img.mode == "LA":
                img = img.convert("L")
            elif img.mode == "P":
                img = img.convert("RGB")

            save_kwargs.update(
                quality=95,
                optimize=True,
            )

        img.save(
            f_dst,
            **save_kwargs,
        )

    return reconstructed


def main():
    # -----------------------------------------------------------------------------
    # Find source files
    # -----------------------------------------------------------------------------

    files = [
        f
        for f in sorted(src.glob("*"))
        if f.is_file()
        # and f.suffix.lower() in (
        #     f".{scan_format.lower()}",
        #     f".{image_format.lower()}",
        # )
        and f.suffix.lower() == f".{scan_format.lower()}"
    ]

    files = remove_done_files(
        files,
        dst,
        dst_suffix=f".{image_format}",
    )

    if not files:
        print("nothing to do")
        sys.exit()

    if reconstruct_inner_edge_pattern:
        print("inner-edge pattern reconstruction: enabled")
    else:
        print("inner-edge pattern reconstruction: disabled")

    # -----------------------------------------------------------------------------
    # Compute the average scanned page height
    # -----------------------------------------------------------------------------

    print("measuring page heights")

    page_heights = []

    content_files = []
    extra_files = []
    for f in files:
        page_num = get_page_num(f)
        if 1 <= page_num <= config.num_pages:
            # process content pages
            # restore the page size only for content pages
            content_files.append(f)
        else:
            # copy extra pages: book cover, etc
            extra_files.append(f)

    # copy extra pages: book cover, etc
    if extra_files:
        print(f"copying {len(extra_files)} extra pages")
        for f in extra_files:
            f_dst = dst / f.name
            shutil.copy(f, f_dst)
    extra_files = []

    if not content_files:
        print("no content files")
        return

    page_aspect = config.page_width_px / config.page_height_px

    process_content_files = []

    if 0:
        # debug: process only the first page
        content_files = content_files[:1]

    original_width_mm = config.page_width_mm
    original_height_mm = config.page_height_mm
    unbinded_width_mm = config.unbinded_page_width_mm

    original_aspect = original_width_mm / original_height_mm
    unbinded_aspect = unbinded_width_mm / original_height_mm

    if 0:
        # debug
        print()
        print("page aspect-ratio diagnostics")
        print(f"  original page: {original_width_mm:.2f} x {original_height_mm:.2f} mm")
        print(f"  unbinded page: {unbinded_width_mm:.2f} x {original_height_mm:.2f} mm")
        print(f"  target unbinded aspect ratio: {unbinded_aspect:.8f}")
        print(f"  scanner rotation correction: {config.do_rotate}")
        print()

    # TODO move this to a function: get_page_heights
    for f in tqdm(
        content_files,
        ncols=80,
        unit="page",
    ):
        with Image.open(f) as img:

            original_scan_width = img.width
            original_scan_height = img.height

            img_aspect = img.width / img.height

            # -------------------------------------------------------------
            # Correct scanner-induced aspect-ratio distortion.
            # -------------------------------------------------------------

            if config.do_rotate:
                # Height is the good dimension.
                corrected_width = round(img.height * unbinded_aspect)
                corrected_height = img.height
            else:
                # Width is the good dimension.
                corrected_width = img.width
                corrected_height = round(img.width / unbinded_aspect)

            corrected_aspect = corrected_width / corrected_height

            if 0:
                # debug
                print()
                print(f"{f.name}")
                print(f"  scanner image: {original_scan_width} x {original_scan_height} px")
                print(f"  scanner aspect ratio: {img_aspect:.8f}")
                print(f"  corrected image: {corrected_width} x {corrected_height} px")
                print(f"  corrected aspect ratio: {corrected_aspect:.8f}")
                if config.do_rotate:
                    print(
                        f"  width correction: {original_scan_width} -> {corrected_width} px "
                        f"({(corrected_width / original_scan_width - 1) * 100:+.3f}%)"
                    )
                else:
                    print(
                        f"  height correction: {original_scan_height} -> {corrected_height} px "
                        f"({(corrected_height / original_scan_height - 1) * 100:+.3f}%)"
                    )

            # -------------------------------------------------------------
            # sanity check
            # -------------------------------------------------------------

            if img_aspect > unbinded_aspect:
                print()
                print(f"WARNING: image is wider than expected unbinded page aspect ratio: {f.name}")
                extra_files.append(f)
                continue

            # IMPORTANT:
            # We want the height AFTER scanner aspect-ratio correction.
            page_heights.append(corrected_height)

            process_content_files.append(f)

    # copy extra pages: book cover, etc
    if extra_files:
        print(f"copying {len(extra_files)} extra pages")
        for f in extra_files:
            f_dst = dst / f.name
            shutil.copy(f, f_dst)
    extra_files = []

    if not process_content_files:
        print("no content files")
        return

    average_height = sum(page_heights) / len(page_heights)

    # Image dimensions must be integers. Round the arithmetic mean to the nearest
    # pixel so every output page has exactly the same height.
    target_height = round(average_height)

    # Restore the original aspect ratio:
    #
    #     target_width / target_height
    #         =
    #     config.page_width_px / config.page_height_px
    #
    # Round to the nearest pixel so every output page also has exactly the same
    # width.
    target_width = round(
        target_height
        * page_width
        / page_height
    )

    if target_width <= 0:
        sys.exit("error: computed target_width must be greater than zero")

    print(
        "page height: "
        f"average {average_height:.2f} px "
        f"-> normalized {target_height} px"
    )

    print(
        "page size: "
        f"{target_width} x {target_height} px"
    )

    # -----------------------------------------------------------------------------
    # Process pages
    # -----------------------------------------------------------------------------

    tasks = [
        (
            str(f),
            str(dst),
            target_height,
            target_width,
            reconstruct_inner_edge_pattern,
        )
        for f in process_content_files
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
        for done in executor.map(process_image, tasks):
            if done:
                num_done += 1
            pbar.update(1)

    # -----------------------------------------------------------------------------

    print(f"done. restored page size for {num_done} images")


if __name__ == "__main__":
    main()

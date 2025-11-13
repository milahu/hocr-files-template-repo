#!/usr/bin/env python3

"""
Scanner color calibration + correction tool

Modes:
1. Calibration:
   python color_fix.py bad_colors.png good_colors.png --calibrate scanner_calib.json [--debug]

2. Apply calibration:
   python color_fix.py new_scan.png output.png --apply-calibration scanner_calib.json
"""


# the low and high values were calibrated manually with GIMP
# these values work for my scanner, your mileage may vary
color_levels_low = 37 / 255
color_levels_high = 220 / 255


import os
import sys
import json
import argparse

import cv2
import numpy as np
from skimage import exposure


def debug_print_calib(calib_path):
    with open(calib_path) as f:
        calib = json.load(f)
    bad_mean = np.array(calib["bad_mean"], dtype=float)
    bad_std  = np.array(calib["bad_std"], dtype=float)
    good_mean = np.array(calib["good_mean"], dtype=float)
    good_std  = np.array(calib["good_std"], dtype=float)

    print("bad_mean (Lab):", bad_mean)
    print("bad_std  (Lab):", bad_std)
    print("good_mean(Lab):", good_mean)
    print("good_std (Lab):", good_std)

    # scale factors and mean diffs
    with np.errstate(divide='ignore', invalid='ignore'):
        scale = good_std / (bad_std + 1e-12)
    shift = good_mean - bad_mean
    print("scale factors (good_std / bad_std):", scale)
    print("mean shifts (good_mean - bad_mean):", shift)


# if 1:
#     debug_print_calib("012-fix-colors.calibration.json")
#     sys.exit()


# ---------------------------------------------------------
# Color calibration helpers
# ---------------------------------------------------------
def compute_lab_stats(img, mask=None):
    """Compute mean/std in Lab space with correct scaling."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    if mask is not None:
        lab = lab[mask == 1]
    mean = lab.mean(axis=0)
    std = lab.std(axis=0)
    return mean, std


def apply_color_calibration(image, calib_file):
    """Apply saved Lab calibration to a new image."""
    with open(calib_file) as f:
        calib = json.load(f)

    bad_mean = np.array(calib["bad_mean"], dtype=np.float32)
    bad_std = np.array(calib["bad_std"], dtype=np.float32)
    good_mean = np.array(calib["good_mean"], dtype=np.float32)
    good_std = np.array(calib["good_std"], dtype=np.float32)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    corrected = (lab - bad_mean) * (good_std / bad_std) + good_mean
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    return cv2.cvtColor(corrected, cv2.COLOR_LAB2BGR)


def apply_color_calibration_mean_only(image, calib_file):
    import json, numpy as np, cv2
    with open(calib_file) as f:
        calib = json.load(f)

    bad_mean = np.array(calib["bad_mean"], dtype=np.float32)
    good_mean = np.array(calib["good_mean"], dtype=np.float32)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

    # subtract bad mean, add good mean (translation only)
    lab_corrected = lab - bad_mean + good_mean

    # clip safely and convert back
    lab_corrected = np.clip(lab_corrected, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)


def apply_color_calibration_regularized(image, calib_file, min_scale=0.8, max_scale=1.25, eps=1e-6):
    import json, numpy as np, cv2
    with open(calib_file) as f:
        calib = json.load(f)

    bad_mean = np.array(calib["bad_mean"], dtype=np.float32)
    bad_std = np.array(calib["bad_std"], dtype=np.float32)
    good_mean = np.array(calib["good_mean"], dtype=np.float32)
    good_std = np.array(calib["good_std"], dtype=np.float32)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

    scale = good_std / (bad_std + eps)
    scale_clipped = np.clip(scale, min_scale, max_scale)

    lab_corr = (lab - bad_mean) * scale_clipped + good_mean
    lab_corr = np.clip(lab_corr, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab_corr, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------
# Debug visualization
# ---------------------------------------------------------
def draw_detected_region(image, corners):
    vis = image.copy()
    cv2.polylines(vis, [np.int32(corners)], True, (0, 255, 0), 3, cv2.LINE_AA)
    return vis


# ---------------------------------------------------------
# Calibration (find subimage + compute correction)
# ---------------------------------------------------------
def calibrate(bad_path, good_path, calib_path, debug=False):
    bad = cv2.imread(bad_path)
    good = cv2.imread(good_path)
    if bad is None or good is None:
        raise FileNotFoundError("Could not read input images.")

    gray_bad = cv2.cvtColor(bad, cv2.COLOR_BGR2GRAY)
    gray_good = cv2.cvtColor(good, cv2.COLOR_BGR2GRAY)

    # Feature detection + matching
    orb = cv2.ORB_create(8000)
    kp1, des1 = orb.detectAndCompute(gray_good, None)
    kp2, des2 = orb.detectAndCompute(gray_bad, None)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:400]

    # Compute homography
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    H, mask_h = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        raise RuntimeError("Homography could not be computed.")

    h, w = good.shape[:2]
    warped_good = cv2.warpPerspective(good, H, (bad.shape[1], bad.shape[0]))

    if 0:
        mask = (cv2.cvtColor(warped_good, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)

        # Compute Lab statistics in overlap region
        bad_mean, bad_std = compute_lab_stats(bad, mask)
        good_mean, good_std = compute_lab_stats(warped_good, mask)
    elif 1:
        # compute overlap mask
        mask = (cv2.cvtColor(warped_good, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)

        # --- NEW: shrink mask by 10% on each side (remove edge artifacts) ---
        h_mask, w_mask = mask.shape
        border = int(0.1 * min(h_mask, w_mask))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, border//5), max(3, border//5)))
        mask_eroded = cv2.erode(mask, kernel, iterations=1)

        # optional: exclude too dark/light pixels from stats
        lab_bad = cv2.cvtColor(bad, cv2.COLOR_BGR2LAB)
        L_channel = lab_bad[..., 0]
        valid_L = ((L_channel > 20) & (L_channel < 240)).astype(np.uint8)
        mask_final = cv2.bitwise_and(mask_eroded, mask_eroded, mask=valid_L)

        # visualize mask area
        if debug:
            mask_vis = cv2.addWeighted(bad, 0.7, cv2.cvtColor(mask_final*255, cv2.COLOR_GRAY2BGR), 0.3, 0)
            base = os.path.splitext(calib_path)[0]
            cv2.imwrite(base + "_5_mask_area.png", mask_vis)
            print(f"🧭 Mask visualization saved to {base}_5_mask_area.png")

        # use mask_final for statistics
        bad_mean, bad_std = compute_lab_stats(bad, mask_final)
        good_mean, good_std = compute_lab_stats(warped_good, mask_final)

    calib = {
        "bad_mean": bad_mean.tolist(),
        "bad_std": bad_std.tolist(),
        "good_mean": good_mean.tolist(),
        "good_std": good_std.tolist()
    }

    with open(calib_path, "w") as f:
        json.dump(calib, f, indent=2)
    print(f"✅ Calibration saved to {calib_path}")

    # Debug outputs
    if debug:
        base = os.path.splitext(calib_path)[0]

        # Draw where the subimage was found
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, H)
        detected = draw_detected_region(bad, dst)

        # Match visualization
        match_vis = cv2.drawMatches(good, kp1, bad, kp2, good_matches[:50], None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        cv2.imwrite(base + "_1_matches.png", match_vis)
        cv2.imwrite(base + "_2_detected_region.png", detected)
        cv2.imwrite(base + "_3_warped_good.png", warped_good)
        cv2.imwrite(base + "_4_mask.png", mask * 255)
        print(f"🔍 Debug images saved with prefix {base}_*")


def calibrate_histogram(bad_path, good_path, calib_path, debug=False):
    bad = cv2.imread(bad_path)
    good = cv2.imread(good_path)
    if bad is None or good is None:
        raise FileNotFoundError("Could not read images.")

    # Step 1: find approximate region (as before)
    orb = cv2.ORB_create(8000)
    gray_bad = cv2.cvtColor(bad, cv2.COLOR_BGR2GRAY)
    gray_good = cv2.cvtColor(good, cv2.COLOR_BGR2GRAY)
    kp1, des1 = orb.detectAndCompute(gray_good, None)
    kp2, des2 = orb.detectAndCompute(gray_bad, None)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:400]
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    h, w = good.shape[:2]
    warped_good = cv2.warpPerspective(good, H, (bad.shape[1], bad.shape[0]))
    mask = (cv2.cvtColor(warped_good, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)

    # Step 2: extract overlapping pixels only
    overlap_bad = cv2.bitwise_and(bad, bad, mask=mask)
    overlap_good = cv2.bitwise_and(warped_good, warped_good, mask=mask)

    # Step 3: downsample to reduce noise
    overlap_bad_small = cv2.resize(overlap_bad, (512, 512))
    overlap_good_small = cv2.resize(overlap_good, (512, 512))

    # Step 4: convert to RGB for skimage
    bad_rgb = cv2.cvtColor(overlap_bad_small, cv2.COLOR_BGR2RGB)
    good_rgb = cv2.cvtColor(overlap_good_small, cv2.COLOR_BGR2RGB)

    # Step 5: compute per-channel histogram mapping with skimage
    matched = exposure.match_histograms(bad_rgb, good_rgb, channel_axis=-1)

    # Compute LUTs (color mapping functions) per channel
    luts = []
    for ch in range(3):
        src_vals = bad_rgb[..., ch].flatten()
        dst_vals = matched[..., ch].flatten()
        lut = np.zeros(256, dtype=np.uint8)
        hist_src, bins = np.histogram(src_vals, 256, [0, 256])
        hist_dst, _ = np.histogram(dst_vals, 256, [0, 256])
        cdf_src = np.cumsum(hist_src) / np.sum(hist_src)
        cdf_dst = np.cumsum(hist_dst) / np.sum(hist_dst)
        for i in range(256):
            j = np.searchsorted(cdf_dst, cdf_src[i])
            lut[i] = np.clip(j, 0, 255)
        luts.append(lut.tolist())

    json.dump({"luts": luts}, open(calib_path, "w"), indent=2)
    print(f"✅ Histogram calibration saved to {calib_path}")

    if debug:
        cv2.imwrite(calib_path.replace(".json", "_debug_matched.png"),
                    cv2.cvtColor(matched, cv2.COLOR_RGB2BGR))


def apply_histogram_calibration(img, calib_path):
    with open(calib_path) as f:
        data = json.load(f)
    luts = [np.array(lut, dtype=np.uint8) for lut in data["luts"]]

    corrected = cv2.merge([cv2.LUT(img[..., c], luts[c]) for c in range(3)])
    return corrected


def calibrate_histogram(bad_path, good_path, calib_path, debug=False, alpha=0.7):
    """
    Calibrate color transfer between a 'bad color' scan and a 'good color' reference.

    Parameters
    ----------
    bad_path : str
        Path to the low-quality color (bad) image from the scanner.
    good_path : str
        Path to the high-quality or correctly colored reference image.
    calib_path : str
        Output path for JSON calibration file.
    debug : bool
        If True, saves debug visualizations.
    alpha : float
        Blend factor [0–1]; lower values reduce saturation (0.7 recommended).
    """
    bad = cv2.imread(bad_path)
    good = cv2.imread(good_path)
    if bad is None or good is None:
        raise FileNotFoundError("Could not read one or both images.")

    # --- Step 1: feature-based alignment (fuzzy match) ---
    orb = cv2.ORB_create(8000)
    gray_bad = cv2.cvtColor(bad, cv2.COLOR_BGR2GRAY)
    gray_good = cv2.cvtColor(good, cv2.COLOR_BGR2GRAY)
    kp1, des1 = orb.detectAndCompute(gray_good, None)
    kp2, des2 = orb.detectAndCompute(gray_bad, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:400]

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        raise RuntimeError("Failed to compute homography (fuzzy alignment).")

    h, w = good.shape[:2]
    warped_good = cv2.warpPerspective(good, H, (bad.shape[1], bad.shape[0]))

    mask = (cv2.cvtColor(warped_good, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)

    # Erode mask to remove edges (avoid warp artifacts)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    mask = cv2.erode(mask, kernel, iterations=1)

    # --- Step 2: extract overlap ---
    overlap_bad = cv2.bitwise_and(bad, bad, mask=mask)
    overlap_good = cv2.bitwise_and(warped_good, warped_good, mask=mask)

    # Downsample to make histogram matching stable
    overlap_bad_small = cv2.resize(overlap_bad, (512, 512))
    overlap_good_small = cv2.resize(overlap_good, (512, 512))

    bad_rgb = cv2.cvtColor(overlap_bad_small, cv2.COLOR_BGR2RGB)
    good_rgb = cv2.cvtColor(overlap_good_small, cv2.COLOR_BGR2RGB)

    # --- Step 3: color histogram matching (per-channel) ---
    matched = exposure.match_histograms(bad_rgb, good_rgb, channel_axis=-1)

    # Blend with original to control saturation
    matched_bgr = cv2.cvtColor(matched, cv2.COLOR_RGB2BGR)
    blended = cv2.addWeighted(matched_bgr, alpha, overlap_bad_small, 1 - alpha, 0)

    # --- Step 4: compute per-channel LUTs ---
    luts = []
    for c in range(3):
        src = overlap_bad_small[..., c].flatten()
        dst = blended[..., c].flatten()
        lut = np.zeros(256, dtype=np.uint8)
        hist_src, _ = np.histogram(src, bins=256, range=[0, 256])
        hist_dst, _ = np.histogram(dst, bins=256, range=[0, 256])
        cdf_src = np.cumsum(hist_src) / np.sum(hist_src)
        cdf_dst = np.cumsum(hist_dst) / np.sum(hist_dst)
        for i in range(256):
            j = np.searchsorted(cdf_dst, cdf_src[i])
            lut[i] = np.clip(j, 0, 255)
        luts.append(lut.tolist())

    with open(calib_path, "w") as f:
        json.dump({"luts": luts, "alpha": alpha}, f, indent=2)
    print(f"✅ Histogram calibration saved to {calib_path}")

    if debug:
        base = os.path.splitext(calib_path)[0]
        cv2.imwrite(base + "_1_overlap_bad.png", overlap_bad)
        cv2.imwrite(base + "_2_overlap_good.png", overlap_good)
        cv2.imwrite(base + "_3_matched.png", cv2.cvtColor(matched, cv2.COLOR_RGB2BGR))
        cv2.imwrite(base + "_4_blended.png", blended)
        cv2.imwrite(base + "_5_mask.png", mask * 255)
        print(f"🔍 Debug images saved with prefix {base}_*")


def apply_histogram_calibration(img, calib_path):
    """
    Apply histogram-based color calibration (from calibrate_histogram) to a new image.
    """
    with open(calib_path) as f:
        data = json.load(f)
    luts = [np.array(lut, dtype=np.uint8) for lut in data["luts"]]
    alpha = data.get("alpha", 0.7)

    # Apply LUTs per channel
    corrected = cv2.merge([cv2.LUT(img[..., c], luts[c]) for c in range(3)])

    # Optional mild blending with original to avoid harshness
    corrected = cv2.addWeighted(corrected, alpha, img, 1 - alpha, 0)

    return corrected


def match_luminance_lab(bad_bgr, good_bgr):
    bad_lab = cv2.cvtColor(bad_bgr, cv2.COLOR_BGR2LAB)
    good_lab = cv2.cvtColor(good_bgr, cv2.COLOR_BGR2LAB)

    L_bad, a_bad, b_bad = cv2.split(bad_lab)
    L_good, _, _ = cv2.split(good_lab)

    # Histogram match only L
    matched_L = exposure.match_histograms(L_bad, L_good)

    matched_L = np.clip(matched_L, 0, 255).astype(np.uint8)
    merged_lab = cv2.merge([matched_L, a_bad, b_bad])
    return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)


def stretch_contrast_lab(img_bgr, low=2, high=98):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    # Percentile-based contrast stretch
    p_low, p_high = np.percentile(L, (low, high))
    L_stretched = np.clip((L - p_low) * 255 / (p_high - p_low), 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([L_stretched, a, b]), cv2.COLOR_LAB2BGR)


# --------------------------
# Calibration function
# --------------------------
def calibrate_histogram_lab(
        bad_path, good_path, calib_path, debug=False,
        blend_alpha=0.85, contrast_percentiles=(2, 98)
    ):
    """
    Calibrate color correction using LAB histogram matching on L channel only.
    
    Parameters
    ----------
    bad_path : str
        Path to the scanner image with bad colors.
    good_path : str
        Path to the reference image with correct colors.
    calib_path : str
        Output JSON file path for storing LUT and settings.
    debug : bool
        Save debug images.
    blend_alpha : float
        Blend factor between matched and original image to avoid over-enhancement.
    contrast_percentiles : tuple
        Percentiles (low, high) for contrast stretching.
    """
    # --- Load images ---
    bad = cv2.imread(bad_path)
    good = cv2.imread(good_path)
    if bad is None or good is None:
        raise FileNotFoundError("Could not read input images.")

    # --- Step 1: Feature-based fuzzy alignment ---
    orb = cv2.ORB_create(8000)
    gray_bad = cv2.cvtColor(bad, cv2.COLOR_BGR2GRAY)
    gray_good = cv2.cvtColor(good, cv2.COLOR_BGR2GRAY)
    kp1, des1 = orb.detectAndCompute(gray_good, None)
    kp2, des2 = orb.detectAndCompute(gray_bad, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:400]

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        raise RuntimeError("Failed to compute homography.")

    h, w = good.shape[:2]
    warped_good = cv2.warpPerspective(good, H, (bad.shape[1], bad.shape[0]))
    mask = (cv2.cvtColor(warped_good, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)

    # Erode mask to remove edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    mask = cv2.erode(mask, kernel, iterations=1)

    # --- Step 2: extract overlapping region ---
    overlap_bad = cv2.bitwise_and(bad, bad, mask=mask)
    overlap_good = cv2.bitwise_and(warped_good, warped_good, mask=mask)

    # Downsample to reduce noise and speed
    overlap_bad_small = cv2.resize(overlap_bad, (512, 512))
    overlap_good_small = cv2.resize(overlap_good, (512, 512))

    # --- Step 3: Convert to LAB ---
    bad_lab = cv2.cvtColor(overlap_bad_small, cv2.COLOR_BGR2LAB)
    good_lab = cv2.cvtColor(overlap_good_small, cv2.COLOR_BGR2LAB)

    L_bad, a_bad, b_bad = cv2.split(bad_lab)
    L_good, _, _ = cv2.split(good_lab)

    # Histogram matching on L channel only
    L_matched = exposure.match_histograms(L_bad, L_good)

    # Contrast stretching based on percentiles
    low, high = np.percentile(L_matched, contrast_percentiles)
    L_stretched = np.clip((L_matched - low) * 255 / (high - low), 0, 255).astype(np.uint8)

    # Merge channels and convert back to BGR
    matched_lab = cv2.merge([L_stretched, a_bad, b_bad])
    matched_bgr = cv2.cvtColor(matched_lab, cv2.COLOR_LAB2BGR)

    # Blend with original to reduce harshness
    blended = cv2.addWeighted(matched_bgr, blend_alpha, overlap_bad_small, 1 - blend_alpha, 0)

    # --- Step 4: Create LUTs per channel for full image application ---
    luts = []
    for c in range(3):
        src = overlap_bad_small[..., c].flatten()
        dst = blended[..., c].flatten()
        lut = np.zeros(256, dtype=np.uint8)
        hist_src, _ = np.histogram(src, bins=256, range=[0, 256])
        hist_dst, _ = np.histogram(dst, bins=256, range=[0, 256])
        cdf_src = np.cumsum(hist_src) / np.sum(hist_src)
        cdf_dst = np.cumsum(hist_dst) / np.sum(hist_dst)
        for i in range(256):
            j = np.searchsorted(cdf_dst, cdf_src[i])
            lut[i] = np.clip(j, 0, 255)
        luts.append(lut.tolist())

    # Save LUTs + settings to JSON
    with open(calib_path, "w") as f:
        json.dump({"luts": luts, "blend_alpha": blend_alpha, "contrast_percentiles": contrast_percentiles}, f, indent=2)

    print(f"✅ LAB histogram calibration saved to {calib_path}")

    # Debug images
    if debug:
        base = os.path.splitext(calib_path)[0]
        cv2.imwrite(base + "_1_overlap_bad.png", overlap_bad)
        cv2.imwrite(base + "_2_overlap_good.png", overlap_good)
        cv2.imwrite(base + "_3_matched_bgr.png", matched_bgr)
        cv2.imwrite(base + "_4_blended.png", blended)
        cv2.imwrite(base + "_5_mask.png", mask * 255)
        print(f"🔍 Debug images saved with prefix {base}_*")


# --------------------------
# Application function
# --------------------------
def apply_histogram_calibration_lab(img, calib_path):
    """
    Apply previously computed LAB histogram calibration to a new image.
    """
    with open(calib_path) as f:
        data = json.load(f)

    luts = [np.array(lut, dtype=np.uint8) for lut in data["luts"]]
    blend_alpha = data.get("blend_alpha", 0.85)

    # Apply LUTs per channel
    corrected = cv2.merge([cv2.LUT(img[..., c], luts[c]) for c in range(3)])

    # Blend with original
    corrected = cv2.addWeighted(corrected, blend_alpha, img, 1 - blend_alpha, 0)

    return corrected


def auto_levels(img_bgr, low_pct=2, high_pct=98):
    """
    Automatically determine low/high thresholds for linear levels.
    Maps the low/high percentiles to 0..255.
    
    Parameters:
        img_bgr : np.ndarray
            Input BGR image.
        low_pct : float
            Percentile for black point (0..100)
        high_pct : float
            Percentile for white point (0..100)
    """
    # Convert to LAB and use L channel (perceptual lightness)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L = lab[..., 0].flatten()
    
    # Compute percentile thresholds
    low = np.percentile(L, low_pct)
    high = np.percentile(L, high_pct)
    
    # Apply linear mapping
    img = img_bgr.astype(np.float32)
    img = (img - low) * 255 / (high - low)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, low, high


def linear_levels(img_bgr, low=37, high=220):
    """
    Apply linear 'color levels' adjustment to image (like GIMP).
    Maps low..high to 0..255.
    """
    img = img_bgr.astype(np.float32)
    img = (img - low) * 255 / (high - low)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def linear_levels(img_bgr, low=0.145, high=0.863):
    """
    Apply linear 'color levels' adjustment to an image (like GIMP).

    Parameters
    ----------
    img_bgr : np.ndarray
        Input BGR image (uint8 or float).
    low : float
        Lower threshold in normalized range [0,1].
    high : float
        Upper threshold in normalized range [0,1].

    Returns
    -------
    np.ndarray
        Contrast-stretched BGR image (uint8, 0-255).
    """
    if not (0.0 <= low <= 1.0) or not (0.0 <= high <= 1.0):
        raise ValueError("low and high must be in the range [0,1]")
    if low >= high:
        raise ValueError("low must be less than high")

    img = img_bgr.astype(np.float32) / 255.0  # normalize to 0..1
    img = (img - low) / (high - low)          # linear stretch
    img = np.clip(img, 0, 1.0)
    img = (img * 255).astype(np.uint8)       # back to uint8
    return img


# ---------------------------------------------------------
# Apply mode: use existing calibration on new scans
# ---------------------------------------------------------
def apply_mode(scan_path, output_path, calib_path):
    scan = cv2.imread(scan_path)
    if scan is None:
        raise FileNotFoundError(f"Cannot read image: {scan_path}")

    # fixed = apply_color_calibration(scan, calib_path)
    # fixed = apply_color_calibration_mean_only(scan, calib_path)
    # fixed = apply_color_calibration_regularized(scan, calib_path)
    # fixed = apply_histogram_calibration(scan, calib_path)
    fixed = apply_histogram_calibration_lab(scan, calib_path)

    # no. this is wrong in most cases
    # fixed, low_val, high_val = auto_levels(fixed, low_pct=2, high_pct=98)
    # print(f"auto_levels: low={low_val:.1f}, high={high_val:.1f}")

    fixed = linear_levels(fixed, color_levels_low, color_levels_high)

    cv2.imwrite(output_path, fixed)
    print(f"✅ Corrected image saved to {output_path}")


# ---------------------------------------------------------
# CLI entry
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scanner color calibration + correction tool")

    parser.add_argument("input1", help="In calibration mode: bad_colors image. In apply mode: input image to correct.")
    parser.add_argument("input2", help="In calibration mode: good_colors image. In apply mode: output corrected image.")
    parser.add_argument("--calibrate", metavar="FILE", help="Create calibration JSON from input1=bad, input2=good.")
    parser.add_argument("--apply-calibration", metavar="FILE", help="Apply existing calibration JSON to input1 and save to input2.")
    parser.add_argument("--debug", action="store_true", help="Save intermediate debug images during calibration.")

    args = parser.parse_args()

    if args.calibrate:
        # calibrate(args.input1, args.input2, args.calibrate, args.debug)
        # calibrate_histogram(args.input1, args.input2, args.calibrate, args.debug)
        calibrate_histogram_lab(args.input1, args.input2, args.calibrate, args.debug)

    elif args.apply_calibration:
        apply_mode(args.input1, args.input2, args.apply_calibration)
    else:
        parser.error("You must specify either --calibrate or --apply-calibration.")

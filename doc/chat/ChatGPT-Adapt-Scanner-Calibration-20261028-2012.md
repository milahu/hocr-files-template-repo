# Adapt Scanner Calibration

**User:** Anonymous  
**Created:** 2026/10/28 18:54:17  
**Updated:** 2026/10/28 19:57:42  
**Exported:** 2026/10/28 20:12:34  
**Link:** [<https://chatgpt.com/c/6ae236c8-bdac-83eb-91be-fc9d1ca0b44c>](https://chatgpt.com/c/6ae236c8-bdac-83eb-91be-fc9d1ca0b44c)

## Prompt:

10/28/2026, 6:58:57 PM

help me adapt my scanner calibration python script, to make it work with my IT8 calibration target

this python script is part of my book-scanning pipeline `https://github.com/milahu/hocr-files-template-repo`

currently i have this calibration script:

012-fix-colors.py

``` python
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
```

but that calibration script produces too much errors (red tint), so now i bought a (rather expensive) IT8 calibration target so i can build a proper fix

my IT8 calibration target has the data number R190823, and the corresponding color reference data can be downloaded from `https://www.silverfast.com/it8calibration/en.html?get_reference=R190823&click=1`

here is the color reference data for my IT8 target:

``` arduino
IT8.7/2
ORIGINATOR "LaserSoft Imaging AG"
DESCRIPTOR "Kodak Professional Endura, L* a* b* (light D50, viewing angle 2)"
MANUFACTURER "LaserSoft Imaging AG - http://www.silverfast.com"
CREATED "September 5, 2019"
PROD_DATE "2019:09"
SERIAL "R190823 Batch Average Data"
MATERIAL "Kodak Professional Endura"
NUMBER_OF_FIELDS 7
BEGIN_DATA_FORMAT
SAMPLE_ID   XYZ_X   XYZ_Y   XYZ_Z          LAB_L   LAB_A   LAB_B
END_DATA_FORMAT
NUMBER_OF_SETS 288
BEGIN_DATA
A1           4.60    4.21    2.95          24.34    7.43    3.66
A2           5.32    4.07    2.46          23.91   18.24    6.79
A3           5.81    3.92    2.09          23.39   26.25    9.19
A4           8.27    4.96    1.48          26.62   36.76   21.09
A5          12.82   11.60    8.08          40.56   11.41    5.36
A6          15.37   11.95    7.43          41.14   24.82    8.85
A7          15.91   11.15    6.34          39.82   33.63   11.21
A8          13.96    8.07    3.63          34.13   46.47   15.85
A9          40.48   40.21   30.66          69.62    5.35    3.84
A10         41.57   39.58   29.59          69.17   10.62    4.75
A11         42.54   38.86   27.76          68.65   15.77    6.85
A12         39.40   33.14   23.18          64.27   25.03    7.42
A13         70.22   72.12   58.01          88.03    1.45    1.52
A14         70.24   72.11   58.03          88.02    1.52    1.50
A15         70.23   72.12   57.93          88.03    1.48    1.61
A16         70.34   72.23   58.01          88.08    1.49    1.62
A17         70.03   71.87   57.85          87.91    1.57    1.48
A18         70.22   72.11   57.91          88.02    1.49    1.62
A19         70.38   72.28   58.16          88.10    1.47    1.50
A20          7.57    5.50    1.99          28.10   23.98   18.28
A21          5.59    4.49    2.21          25.24   15.68   11.32
A22          3.87    3.55    2.39          22.12    6.79    4.34
B1           4.47    4.18    2.44          24.24    6.10    7.51
B2           5.10    4.18    1.79          24.25   14.21   13.62
B3           5.62    4.00    1.40          23.66   22.83   17.03
B4           7.17    4.23    0.88          24.41   36.09   25.61
B5          13.60   12.15    6.17          41.45   12.60   14.81
B6          15.84   12.50    4.49          42.00   23.78   24.22
B7          15.10   10.69    2.72          39.06   32.13   30.84
B8          13.42    7.42    0.99          32.75   49.00   38.26
B9          40.39   40.25   28.91          69.64    4.95    6.67
B10         41.30   39.58   25.80          69.17    9.77   11.10
B11         42.34   39.17   22.31          68.87   14.20   17.02
B12         38.63   33.39   14.29          64.47   21.73   27.26
B13         69.08   71.32   57.56          87.64    0.68    1.32
B14         68.34   69.31   56.98          86.65    3.31    0.22
B15         69.51   71.52   55.62          87.74    1.19    3.51
B16         67.33   69.28   54.83          86.64    1.17    2.44
B17         68.07   69.31   54.77          86.65    2.74    2.53
B18         68.63   70.99   55.32          87.48    0.40    3.38
B19         67.07   68.61   56.88          86.31    2.02   -0.27
B20         30.24   29.55   17.25          61.26    6.70   14.52
B21         27.61   26.44   14.74          58.45    8.66   15.73
B22         22.77   21.01   11.42          52.96   11.82   15.46
C1           5.75    5.78    3.08          28.83    2.09   10.45
C2           6.24    5.82    2.24          28.94    6.99   17.40
C3           6.59    5.66    1.63          28.55   12.35   22.76
C4           7.17    5.20    1.15          27.30   23.64   26.55
C5          24.41   23.56   10.55          55.64    7.49   22.76
C6          25.93   23.35    5.81          55.43   14.85   40.56
C7          23.47   19.58    2.69          51.35   21.87   52.23
C8          22.43   15.91    1.91          46.86   36.53   51.42
C9          46.71   47.21   31.36          74.32    3.37   10.85
C10         48.00   47.03   25.79          74.21    7.46   19.80
C11         48.03   46.12   20.58          73.62   10.05   28.64
C12         43.64   40.52    9.45          69.84   13.90   50.87
C13         59.62   64.72   55.31          84.34   -6.52   -2.03
C14         59.03   55.89   52.20          79.55   12.70   -6.94
C15         66.05   68.52   43.56          86.26   -0.05   14.69
C16         59.71   61.58   48.11          82.68    0.80    3.07
C17         60.04   58.81   44.62          81.18    8.07    4.63
C18         56.73   62.12   42.43          82.98   -7.66   10.42
C19         53.82   54.37   52.80          78.68    3.58   -9.10
C20         30.92   29.60   13.77          61.30    9.03   23.18
C21         27.48   26.07   10.42          58.10    9.65   27.42
C22         28.50   27.21    9.20          59.16    9.06   33.32
D1           5.46    5.74    3.10          28.75   -0.91   10.18
D2           5.60    5.84    2.24          29.02   -0.46   17.46
D3           5.61    5.78    1.75          28.86    0.44   21.98
D4           5.36    5.45    1.46          27.96    1.33   23.67
D5          26.76   27.85   13.33          59.75   -0.37   21.69
D6          28.78   29.35    7.79          61.09    1.87   41.84
D7          26.20   25.93    4.01          57.97    4.99   54.58
D8          25.74   25.21    3.50          57.28    6.10   56.61
D9          55.41   56.98   36.66          80.17    1.17   13.21
D10         55.54   57.45   29.47          80.43    0.36   24.37
D11         56.44   58.50   24.46          81.02    0.07   33.92
D12         50.34   51.39   10.48          76.91    2.12   59.67
D13         53.13   59.98   54.77          81.82  -11.74   -5.80
D14         52.73   47.21   48.81          74.32   19.55  -12.16
D15         63.22   65.80   34.65          84.89   -0.51   24.19
D16         45.20   47.12   37.76          74.27   -0.67    1.52
D17         52.69   49.49   35.47          75.75   13.31    7.26
D18         46.64   53.92   31.50          78.41  -14.48   17.70
D19         42.78   42.82   49.15          71.43    4.50  -17.52
D20         36.53   39.13   14.33          68.85   -3.92   34.71
D21         31.18   32.53   11.49          63.78   -0.68   33.89
D22         14.80   16.72    4.71          47.90   -7.77   33.19
E1           4.76    5.44    3.13          27.94   -6.01    8.54
E2           4.58    5.54    2.39          28.21   -9.50   14.79
E3           4.18    5.32    1.84          27.64  -12.45   18.90
E4           2.89    4.41    1.57          24.99  -21.31   17.27
E5          13.26   15.16    7.88          45.85   -8.56   15.24
E6          12.97   16.12    5.49          47.13  -15.96   27.80
E7          11.02   14.64    3.56          45.14  -20.84   35.22
E8           7.12   11.31    2.80          40.10  -31.99   31.98
E9          37.66   40.32   26.81          69.70   -3.91   10.27
E10         36.73   40.82   22.90          70.05   -8.45   17.92
E11         36.02   41.38   18.81          70.44  -12.49   26.88
E12         30.17   37.02    9.40          67.29  -19.58   46.65
E13         47.70   55.89   54.33          79.55  -16.42   -9.24
E14         46.34   38.83   45.16          68.63   26.86  -17.68
E15         60.58   63.01   26.91          83.45   -0.41   33.80
E16         34.24   35.88   28.40          66.43   -1.20    1.95
E17         44.88   39.94   26.71          69.43   19.27    9.97
E18         37.02   45.47   22.20          73.20  -21.07   24.68
E19         32.64   31.88   44.66          63.24    6.92  -26.35
E20         29.96   36.14   19.75          66.62  -17.48   18.29
E21         27.86   33.61   15.56          64.65  -17.08   24.35
E22          7.20   10.32    3.82          38.42  -24.03   22.01
F1           2.18    2.78    2.19          19.13   -9.98    0.96
F2           1.88    2.67    1.99          18.66  -14.82    1.99
F3           1.83    2.68    1.85          18.69  -16.10    3.45
F4           1.86    2.75    1.36          18.99  -16.79    9.50
F5           7.21    9.23    6.56          36.42  -15.31    4.39
F6           6.28    9.57    6.19          37.06  -27.44    7.12
F7           6.07   10.14    5.41          38.09  -34.24   12.64
F8           5.98   10.19    3.35          38.17  -35.57   24.66
F9          36.96   40.47   30.62          69.80   -6.62    4.21
F10         35.81   40.70   29.73          69.97  -11.14    5.91
F11         34.36   40.88   28.83          70.09  -16.60    7.58
F12         27.35   37.15   23.02          67.39  -30.93   13.09
F13         41.93   51.16   53.31          76.78  -21.07  -12.93
F14         39.66   30.59   40.83          62.16   34.95  -23.42
F15         58.03   60.09   20.86          81.89    0.22   42.32
F16         23.60   24.68   19.52          56.76   -0.87    1.77
F17         37.46   31.32   18.75          62.77   25.30   13.76
F18         28.59   37.43   15.13          67.60  -26.92   30.53
F19         23.63   22.52   39.90          54.58    8.68  -35.30
F20          4.11    5.09    3.48          26.99  -10.64    4.47
F21          3.35    4.24    2.74          24.45  -11.22    5.47
F22          3.07    3.77    3.27          22.89   -9.12   -1.14
G1           3.12    4.06    3.99          23.87  -12.44   -4.13
G2           2.81    4.13    4.46          24.08  -18.85   -6.51
G3           3.03    4.46    4.73          25.14  -19.51   -6.18
G4           3.57    5.30    5.59          27.57  -21.08   -6.41
G5           9.65   12.12   10.51          41.40  -15.29   -1.66
G6           8.42   12.30   11.82          41.69  -26.83   -5.18
G7           8.83   13.27   13.28          43.17  -29.68   -6.77
G8          10.38   15.52   15.69          46.33  -30.84   -7.55
G9          36.80   40.26   32.74          69.65   -6.50    0.72
G10         35.30   41.07   33.90          70.23  -13.99   -0.02
G11         33.41   40.92   34.47          70.12  -20.03   -1.03
G12         23.26   33.72   31.91          64.74  -36.72   -6.51
G13         36.50   46.44   52.18          73.83  -25.52  -16.78
G14         33.74   23.81   36.69          55.89   42.44  -28.70
G15         55.74   57.34   15.76          80.37    1.12   50.98
G16         15.98   16.72   13.41          47.90   -0.82    1.02
G17         31.74   24.83   13.02          56.90   30.98   17.64
G18         22.61   31.45   10.68          62.88  -31.68   34.83
G19         16.74   15.60   35.82          46.44    9.77  -43.78
G20          9.27   13.59   15.10          43.63  -28.02  -10.75
G21          7.54    4.56   10.57          25.45   35.17  -29.36
G22         14.04   15.01    2.73          45.65   -2.70   42.12
H1           3.28    4.02    4.77          23.74   -9.34   -8.81
H2           3.07    4.07    6.32          23.91  -13.57  -16.15
H3           3.36    4.34    7.59          24.77  -12.39  -20.00
H4           4.60    5.34   12.66          27.69   -7.00  -31.74
H5          10.10   11.81   13.19          40.91   -9.63  -10.41
H6           9.49   12.24   17.81          41.59  -17.42  -20.68
H7           9.32   12.15   21.21          41.45  -18.19  -28.11
H8          11.93   14.17   32.35          44.48  -11.54  -42.10
H9          37.26   40.53   35.82          69.85   -5.83   -3.42
H10         36.21   40.76   40.32          70.01   -9.98   -9.23
H11         35.15   40.70   44.18          69.96  -13.34  -14.19
H12         25.47   34.16   47.21          65.09  -28.72  -26.23
H13         31.40   41.78   50.85          70.72  -29.80  -20.68
H14         29.30   19.03   33.37          50.72   48.58  -32.85
H15         53.55   54.66   12.08          78.85    2.17   58.13
H16         10.34   10.98    8.77          39.55   -1.93    1.04
H17         26.26   18.94    8.02          50.62   36.96   22.93
H18         19.29   27.98    8.75          59.87  -34.62   36.15
H19         12.43   11.23   32.76          39.97   11.28  -50.50
H20          4.54    6.60    7.65          30.88  -21.53   -9.68
H21          4.03    2.72    5.54          18.88   23.23  -21.12
H22          7.13    7.66    1.82          33.26   -2.48   28.82
I1           4.95    5.37    7.35          27.76   -2.84  -13.84
I2           5.05    5.44   11.16          27.94   -2.29  -26.91
I3           5.01    5.41   13.90          27.88   -2.55  -34.80
I4           6.63    6.66   22.61          31.02    2.13  -48.83
I5          13.78   14.43   15.19          44.84   -0.83   -8.88
I6          13.89   14.70   19.98          45.22   -1.82  -19.10
I7          14.13   14.85   25.06          45.44   -1.23  -28.50
I8          12.15   13.35   34.33          43.28   -4.89  -47.10
I9          38.64   40.31   35.36          69.69   -0.73   -3.04
I10         38.90   40.39   39.21          69.75   -0.14   -8.23
I11         38.79   40.21   43.49          69.62    0.09  -13.93
I12         35.47   37.34   47.26          67.54   -1.79  -22.06
I13         28.43   38.91   49.66          68.68  -32.23  -22.85
I14         25.14   14.91   29.78          45.52   54.28  -36.33
I15         51.29   51.67    8.74          77.08    3.91   65.88
I16          6.31    6.80    5.56          31.34   -2.61    0.23
I17         21.42   14.06    4.66          44.32   42.79   27.28
I18         16.89   25.26    7.02          57.33  -36.29   38.47
I19          9.22    7.94   29.66          33.86   13.72  -56.23
I20          1.75    2.36    2.55          17.26  -12.07   -5.41
I21          1.65    1.40    2.09          11.98    8.22  -10.46
I22          2.33    2.50    1.11          17.90   -1.69   10.94
J1           2.78    2.83    4.04          19.35    0.91  -12.23
J2           3.01    2.80    6.06          19.19    5.75  -23.05
J3           3.22    2.81    8.38          19.29    8.93  -32.46
J4           4.00    3.05   15.05          20.24   16.91  -50.92
J5          11.47   11.49   13.05          40.39    2.86  -10.92
J6          12.02   11.47   16.57          40.36    6.82  -19.94
J7          13.01   11.77   21.41          40.84   11.44  -29.56
J8          12.99   10.84   30.71          39.31   17.93  -48.50
J9          39.17   40.05   35.15          69.50    1.78   -3.07
J10         39.88   40.21   39.49          69.62    3.49   -8.82
J11         40.38   39.82   42.88          69.34    6.22  -13.65
J12         38.74   36.98   46.02          67.26   10.06  -21.07
J13         26.60   37.03   48.74          67.30  -33.55  -24.18
J14         22.30   12.31   27.18          41.70   58.21  -38.63
J15         49.99   49.65    6.70          75.85    5.76   71.74
J16          3.35    3.65    3.15          22.48   -2.68   -0.95
J17         17.76   10.76    2.58          39.17   46.68   32.17
J18         14.93   22.97    5.64          55.04  -37.75   40.73
J19          7.40    5.84   27.09          29.00   18.50  -60.39
J20          6.88    4.06    0.86          23.86   35.52   25.04
J21          5.75    9.43    2.60          36.79  -32.18   27.85
J22          2.47    1.96    8.42          15.26   12.72  -39.56
K1           5.96    5.34    6.22          27.66    9.43   -9.22
K2           6.94    5.31    8.48          27.61   19.95  -18.48
K3           7.36    4.88   10.25          26.39   29.34  -26.71
K4          10.72    6.15   17.34          29.78   43.09  -39.97
K5          15.26   14.34   14.70          44.71    8.77   -7.86
K6          17.49   14.78   18.57          45.34   18.61  -15.90
K7          19.14   14.60   22.73          45.08   28.36  -24.81
K8          18.36   10.61   25.68          38.91   50.95  -40.85
K9          40.30   40.03   35.62          69.50    5.33   -3.75
K10         41.58   39.61   37.96          69.18   10.55   -7.51
K11         42.95   39.03   40.78          68.78   16.43  -11.96
K12         41.56   33.85   42.42          64.84   29.25  -20.83
K13         25.93   36.39   48.29          66.82  -34.23  -24.50
K14         20.68   10.86   25.53          39.34   60.73  -39.86
K15         48.94   48.02    5.53          74.84    7.32   75.36
K16          1.38    1.51    1.43          12.67   -2.32   -2.26
K17         15.09    8.59    1.45          35.19   48.82   36.27
K18         13.48   21.09    4.66          53.04  -38.10   42.29
K19          6.36    4.51   24.95          25.29   24.07  -63.04
K20          3.92    2.56    0.85          18.18   24.52   15.45
K21          3.20    5.04    1.74          26.85  -23.97   18.60
K22          1.66    1.50    4.63          12.62    5.78  -27.23
L1           4.57    4.11    3.59          24.03    8.40   -1.35
L2           5.19    3.87    3.88          23.24   19.64   -4.52
L3           5.38    3.67    4.10          22.55   24.80   -7.07
L4           7.60    4.70    4.95          25.85   33.99   -6.16
L5          13.06   11.60   10.19          40.58   12.93   -2.05
L6          15.61   11.87   11.46          41.00   26.82   -5.30
L7          16.20   10.81   11.89          39.25   37.76   -9.58
L8          15.55    8.64   11.91          35.28   51.13  -16.50
L9          40.56   39.84   32.23          69.36    6.72    0.98
L10         42.12   39.83   33.72          69.35   11.49   -1.26
L11         43.24   38.98   34.19          68.73   17.46   -3.00
L12         41.22   33.43   35.16          64.51   29.66  -11.69
L13         25.11   35.56   48.00          66.18  -34.92  -25.26
L14         20.11   10.41   24.98          38.57   61.35  -40.21
L15         48.39   47.26    5.16          74.36    7.87   76.40
L16          0.67    0.70    0.74           6.34   -0.34   -3.00
L17         13.82    7.69    1.04          33.33   49.07   38.53
L18         12.93   20.32    4.22          52.20  -38.04   43.35
L19          5.94    3.97   23.91          23.57   26.88  -64.12
L20          1.30    1.11    0.82           9.85    7.61    1.59
L21          1.32    1.76    1.03          14.15  -10.51    5.57
L22          1.01    1.05    1.90           9.44   -0.05  -13.01
GS0         71.83   73.82   59.49          88.84    1.37    1.42
GS1         69.38   71.31   56.81          87.64    1.35    2.09
GS2         62.88   64.80   51.23          84.38    0.92    2.45
GS3         56.44   58.49   46.07          81.01    0.11    2.58
GS4         46.84   48.80   39.04          75.33   -0.60    1.62
GS5         39.71   41.45   33.06          70.49   -0.81    1.68
GS6         34.54   36.19   29.02          66.67   -1.22    1.35
GS7         29.59   31.00   24.68          62.51   -1.15    1.62
GS8         24.84   26.02   20.83          58.06   -1.07    1.30
GS9         21.77   22.90   18.26          54.97   -1.46    1.40
GS10        18.12   19.14   15.40          50.85   -1.74    0.95
GS11        15.16   15.88   12.90          46.82   -0.92    0.56
GS12        13.24   13.97   11.43          44.19   -1.50    0.29
GS13        10.95   11.65    9.36          40.65   -2.10    0.86
GS14         8.77    9.33    7.60          36.60   -1.89    0.36
GS15         7.40    7.95    6.53          33.87   -2.45    0.09
GS16         5.89    6.35    5.28          30.27   -2.53   -0.22
GS17         4.72    5.08    4.26          26.96   -2.33   -0.41
GS18         3.44    3.77    3.26          22.88   -3.03   -1.07
GS19         2.62    2.91    2.57          19.67   -3.34   -1.39
GS20         1.82    2.02    1.84          15.58   -3.12   -1.81
GS21         1.31    1.44    1.41          12.23   -2.37   -2.87
GS22         0.92    0.99    1.01           8.94   -1.35   -3.09
GS23         0.69    0.73    0.76           6.55   -0.44   -3.08
END_DATA
```

when i scan the IT8 target with my flatbed scanner at 600dpi, then i get an image with a width of 3580 pixels and a height of 2400 pixels. the color squares are in the rectangle from X=300 to X=3270 and from Y=150 to Y=2130. this rectangle (with a width of 3270 - 300 = 2970 pixels) (with a height of 2130 - 150 = 1980 pixels) has a table with 24 columns (0 to 23) (2970 / 24 = 123.75 pixels width per column) and 16 rows (1980 / 16 = 123.75 pixels height per row).

the "color cells" have X positions from idx_x=1 to idx_x=22, and Y positions from idx_y=1=A to idx_y=12=L. the top-left color cell A1 has the index positions idx_x=1 and idx_y=1, and the bottom-right color cell L22 has the index positions idx_x=22 and idx_y=12.

the "grayscale cells" (GSxx) have the Y position idx_y=14 and idx_y=15 (same color for both cells) and the grayscale cell X positions range from idx_x=0 to idx_x=23. the leftmost grayscale cell (idx_x=0) is white, the rightmost grayscale cell (idx_x=23) is black.

ok, now i need a script which takes two inputs:

1.  a scanned image of the IT8 calibration target (the image has been rotated and cropped manually by the user, so it contains only the IT8 target at 600dpi)
2.  a text file with the reference data for the scanned IT8 target (data see above)

the script should calculate the differences between expected colors (from the reference data) and the actual colors (from the scanned image), and writes these differences to a JSON file, which can be used in the future to correct the colors of scanned images, scanned with the same scanner

## Response:

10/28/2026, 6:59:04 PM

> **Building a calibration script**
>
> I’m designing a standalone workflow that detects or samples the IT8 grid, fits a polynomial RGB-to-Lab correction against reference data, and applies the resulting transform to future scans. I’m keeping grid coordinates configurable because the crop dimensions aren’t specified.
>
> **Built a calibration script**
>
> **Searched 8 websites**
>
> Worked for 13s

The right approach is to replace your current image-to-image histogram matching with a **reference-based IT8 color calibration**. Your target gives you something your existing script doesn't have: known reference colors for individual patches, so you can measure your scanner's color errors and build a correction from actual measurements rather than trying to make one image's histogram resemble another's.

For your `012-fix-colors.py` pipeline, I'd make this a separate calibration mode that produces a JSON profile, then add a corresponding apply mode for all subsequent book scans.

## How the new calibration will work

<box gap={2}>
  <row align=start gap={3}>
    <box background="surface-secondary" radius="lg" size="44px" align=center justify=center>
      <icon name="scan" size="xl"/>
    </box>
    <box flex="1" gap={1}>
      **1. Read the scanned IT8 target**

      Use the manually rotated and cropped 600 DPI image. Sample the center of each of the 264 color patches and 24 grayscale patches, avoiding their borders.
    </box>

  </row>
  <divider color="subtle"/>
  <row align=start gap={3}>
    <box background="surface-secondary" radius="lg" size="44px" align=center justify=center>
      <icon name="file-text" size="xl"/>
    </box>
    <box flex="1" gap={1}>
      **2. Parse the reference file**

      Read the `SAMPLE_ID` and `LAB_L`, `LAB_A`, `LAB_B` values from your IT8 text file. Match patches by their IDs (`A1` through `L22`, and `GS0` through `GS23`), not by guessing their colors.
    </box>

  </row>
  <divider color="subtle"/>
  <row align=start gap={3}>
    <box background="surface-secondary" radius="lg" size="44px" align=center justify=center>
      <icon name="chart-scatter" size="xl"/>
    </box>
    <box flex="1" gap={1}>
      **3. Measure and fit the correction**

      Convert measured scan colors to a consistent color space, compare them against the reference values, and fit a regularized color transformation. Validate it against held-out patches to avoid overfitting.
    </box>

  </row>
  <divider color="subtle"/>
  <row align=start gap={3}>
    <box background="surface-secondary" radius="lg" size="44px" align=center justify=center>
      <icon name="file-json" size="xl"/>
    </box>
    <box flex="1" gap={1}>
      **4. Save and reuse the calibration**

      Write the model, patch measurements, and error statistics to JSON. The apply mode can load that profile and correct book scans from the same scanner.
    </box>

  </row>
</box>

One important detail: the reference data is specified in **CIE Lab under D50 illumination**, whereas an ordinary OpenCV `BGR → LAB` conversion uses OpenCV's own 8-bit Lab encoding. Those values are not numerically interchangeable. The implementation must use a consistent color-space conversion, and the final result will depend on your scanner's RGB color space and settings.

SilverFast's documentation describes the same core process: measure the scanner's RGB response for each patch, compare it with the target-specific reference data, and create a profile for subsequent scans. \<Cite refs={\["turn842101search13","turn842101search0"\]}/\>

I'll use a polynomial RGB-to-Lab model rather than independent per-channel corrections. This is important because the red tint you describe may be caused by interactions between channels, which simple histogram matching cannot model reliably.

One limitation to keep in mind: the profile is valid for the same scanner and scan settings. Keep exposure, color mode, bit depth, automatic corrections, and any scanner-side color management consistent between calibration and book scanning.

## 1. New script: `012-calibrate-it8.py`

This is a standalone implementation, so you can test the IT8 calibration without changing your existing pipeline yet.

It supports two operations:

- `--calibrate`: sample the IT8 chart, parse the reference data, fit a color transformation, and save a JSON profile.
- `--apply-calibration`: apply the saved transformation to another scan.

The default grid coordinates match the geometry you provided. If your manually cropped image has different dimensions or margins, you can override those coordinates on the command line.

<CodeBlock language="python" editable>
#!/usr/bin/env python3
"""
IT8 scanner calibration for book-scanning pipelines.

Calibration:  
python 012-calibrate-it8.py target.png R190823.txt  
--calibrate scanner-it8.json --debug

Apply:  
python 012-calibrate-it8.py book-scan.png corrected.png  
--apply-calibration scanner-it8.json

Dependencies:  
pip install numpy opencv-python scikit-image

The default grid rectangle is based on the supplied 3580x2400  
scan geometry. Override --grid-x0, --grid-x1, --grid-y0, --grid-y1  
if your cropped image uses different coordinates.

The model learns a polynomial mapping from scanner RGB to the  
reference CIE Lab values, then converts the predicted Lab values  
to RGB for output.  
"""

import argparse  
import json  
import os  
import re

import cv2  
import numpy as np  
from skimage.color import lab2rgb, deltaE_ciede2000

MODEL_VERSION = 1

# IT8 patch grid geometry, in pixels.

DEFAULT_GRID = {  
"x0": 300.0,  
"x1": 3270.0,  
"y0": 150.0,  
"y1": 2130.0,  
"columns": 24,  
"rows": 16,  
}

COLOR_ROWS = "ABCDEFGHIJKL"

def parse_reference_file(path):  
"""Read SAMPLE_ID and CIE Lab values from an IT8 reference file."""  
reference = {}

    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            parts = line.split()

            if len(parts) < 7:
                continue

            sample_id = parts[0]

            # Ignore headers and non-patch lines.
            if sample_id in {
                "IT8.7/2", "ORIGINATOR", "DESCRIPTOR", "MANUFACTURER",
                "CREATED", "PROD_DATE", "SERIAL", "MATERIAL",
                "NUMBER_OF_FIELDS", "BEGIN_DATA_FORMAT",
                "END_DATA_FORMAT", "NUMBER_OF_SETS", "BEGIN_DATA",
                "END_DATA",
            }:
                continue

            # IT8 row format:
            # ID XYZ_X XYZ_Y XYZ_Z LAB_L LAB_A LAB_B
            if not re.fullmatch(r"(?:[A-L](?:[1-9]|1[0-9]|2[0-2])|GS(?:[0-9]|1[0-9]|2[0-3]))", sample_id):
                continue

            try:
                lab = [float(v) for v in parts[4:7]]
            except ValueError:
                continue

            if len(lab) != 3:
                continue

            reference[sample_id] = lab

    if len(reference) < 250:
        raise ValueError(
            f"Only found {len(reference)} reference patches in {path}. "
            "Check that this is the complete IT8 reference data file."
        )

    return reference

def sample_patch(image, cx, cy, patch_width, patch_height, fraction=0.40):  
"""  
Return the median RGB value near the patch center.

    Sampling the central 40% of each cell helps avoid borders,
    neighboring colors, and small alignment errors.
    """
    h, w = image.shape[:2]

    rx = max(1, int(patch_width * fraction / 2))
    ry = max(1, int(patch_height * fraction / 2))

    x0 = max(0, int(round(cx)) - rx)
    x1 = min(w, int(round(cx)) + rx + 1)
    y0 = max(0, int(round(cy)) - ry)
    y1 = min(h, int(round(cy)) + ry + 1)

    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid patch region centered at ({cx}, {cy}).")

    pixels = image[y0:y1, x0:x1].reshape(-1, 3)

    # OpenCV loads BGR; store measurements in RGB order.
    rgb = pixels[:, ::-1].astype(np.float64)

    return np.median(rgb, axis=0)

def measure_patches(image, reference, grid, debug=False):  
"""Measure all reference patches at their expected grid positions."""  
h, w = image.shape\[:2\]

    x0, x1 = grid["x0"], grid["x1"]
    y0, y1 = grid["y0"], grid["y1"]
    columns, rows = grid["columns"], grid["rows"]

    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError(
            f"Grid rectangle ({x0}, {y0}) to ({x1}, {y1}) "
            f"does not fit image dimensions {w}x{h}. "
            "Adjust the --grid-* arguments."
        )

    cell_w = (x1 - x0) / columns
    cell_h = (y1 - y0) / rows

    measured = {}
    expected = {}
    debug_image = image.copy() if debug else None

    def measure(sample_id, col, row):
        cx = x0 + (col + 0.5) * cell_w
        cy = y0 + (row + 0.5) * cell_h

        rgb = sample_patch(image, cx, cy, cell_w, cell_h)

        measured[sample_id] = rgb
        expected[sample_id] = np.asarray(reference[sample_id])

        if debug:
            cv2.rectangle(
                debug_image,
                (int(cx - cell_w * 0.2), int(cy - cell_h * 0.2)),
                (int(cx + cell_w * 0.2), int(cy + cell_h * 0.2)),
                (0, 255, 0),
                2,
            )
            cv2.putText(
                debug_image,
                sample_id,
                (int(cx - 15), int(cy)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 0, 255),
                1,
            )

    # 264 color patches: A1-L22.
    for row, letter in enumerate(COLOR_ROWS, start=1):
        for col in range(1, 23):
            sample_id = f"{letter}{col}"
            if sample_id in reference:
                measure(sample_id, col, row)

    # 24 grayscale patches. Each reference value represents
    # two vertically adjacent cells, so average their measurements.
    for col in range(24):
        sample_id = f"GS{col}"
        if sample_id not in reference:
            continue

        rgb_values = []

        for row in (14, 15):
            cx = x0 + (col + 0.5) * cell_w
            cy = y0 + (row + 0.5) * cell_h
            rgb_values.append(
                sample_patch(image, cx, cy, cell_w, cell_h)
            )

        measured[sample_id] = np.mean(rgb_values, axis=0)
        expected[sample_id] = np.asarray(reference[sample_id])

    if debug:
        cv2.imwrite("it8-detected-patches.png", debug_image)

    if len(measured) < 250:
        raise ValueError(
            f"Measured only {len(measured)} patches. "
            "Check the grid geometry and reference data."
        )

    ids = sorted(measured)
    rgb = np.asarray([measured[i] for i in ids])
    lab = np.asarray([expected[i] for i in ids])

    return ids, rgb, lab

def polynomial_features(rgb):  
"""  
Build second-degree RGB features.

    Inputs are normalized to [0, 1]. The output contains:
    constant, R, G, B, R², G², B², RG, RB, GB.
    """
    rgb = np.asarray(rgb, dtype=np.float64) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    return np.stack(
        [
            np.ones_like(r),
            r, g, b,
            r * r, g * g, b * b,
            r * g, r * b, g * b,
        ],
        axis=-1,
    )

def fit_model(rgb, lab, regularization=0.01):  
"""Fit a regularized polynomial RGB-to-Lab transformation."""  
features = polynomial_features(rgb)

    # Ridge regularization, excluding the constant intercept.
    penalty = np.eye(features.shape[1]) * regularization
    penalty[0, 0] = 0.0

    lhs = features.T @ features + penalty
    rhs = features.T @ lab

    coefficients = np.linalg.solve(lhs, rhs)

    return coefficients

def predict_lab(rgb, coefficients):  
return polynomial_features(rgb) @ coefficients

def lab_to_output_rgb(lab):  
"""  
Convert reference Lab (D50, 2-degree observer) to sRGB.

    skimage handles the Lab-to-XYZ-to-sRGB conversion and gamut
    clipping. Results are 8-bit RGB values.
    """
    lab = np.asarray(lab, dtype=np.float64)
    rgb_float = lab2rgb(
        lab,
        illuminant="D50",
        observer="2",
        channel_axis=-1,
    )

    return np.clip(np.rint(rgb_float * 255.0), 0, 255).astype(np.uint8)

def calculate_metrics(predicted_lab, reference_lab):  
delta_e = deltaE_ciede2000(  
predicted_lab.reshape(-1, 1, 3),  
reference_lab.reshape(-1, 1, 3),  
).reshape(-1)

    return {
        "mean_delta_e_2000": float(np.mean(delta_e)),
        "median_delta_e_2000": float(np.median(delta_e)),
        "p95_delta_e_2000": float(np.percentile(delta_e, 95)),
        "max_delta_e_2000": float(np.max(delta_e)),
    }

def calibrate(image_path, reference_path, output_path, args):  
image = cv2.imread(image_path, cv2.IMREAD_COLOR)  
if image is None:  
raise FileNotFoundError(f"Cannot read target image: {image_path}")

    reference = parse_reference_file(reference_path)

    grid = {
        "x0": args.grid_x0,
        "x1": args.grid_x1,
        "y0": args.grid_y0,
        "y1": args.grid_y1,
        "columns": 24,
        "rows": 16,
    }

    ids, rgb, lab = measure_patches(
        image, reference, grid, debug=args.debug
    )

    # Keep a held-out subset for a basic generalization check.
    # Every fifth patch is held out; remaining patches train the model.
    indices = np.arange(len(ids))
    validation_mask = indices % 5 == 0
    training_mask = ~validation_mask

    coefficients = fit_model(
        rgb[training_mask],
        lab[training_mask],
        regularization=args.regularization,
    )

    predicted_train = predict_lab(rgb[training_mask], coefficients)
    predicted_validation = predict_lab(rgb[validation_mask], coefficients)

    metrics = {
        "training": calculate_metrics(
            predicted_train, lab[training_mask]
        ),
        "validation": calculate_metrics(
            predicted_validation, lab[validation_mask]
        ),
    }

    print(f"Measured patches: {len(ids)}")
    print("Training error:", json.dumps(metrics["training"], indent=2))
    print("Validation error:", json.dumps(metrics["validation"], indent=2))

    profile = {
        "model": "polynomial_rgb_to_lab_degree_2",
        "model_version": MODEL_VERSION,
        "reference_illuminant": "D50",
        "reference_observer": "2",
        "reference_serial": "R190823",
        "input_rgb_assumption": "scanner RGB values, 8-bit",
        "grid": grid,
        "regularization": args.regularization,
        "feature_order": [
            "1", "R", "G", "B", "R2", "G2", "B2",
            "RG", "RB", "GB",
        ],
        "coefficients": coefficients.tolist(),
        "patch_ids": ids,
        "measured_rgb": rgb.tolist(),
        "reference_lab": lab.tolist(),
        "metrics": metrics,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    print(f"Calibration profile saved to {output_path}")

    if args.debug:
        predicted_all = predict_lab(rgb, coefficients)
        corrected_rgb = lab_to_output_rgb(predicted_all)

        with open(
            os.path.splitext(output_path)[0] + "-patch-report.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                {
                    sample_id: {
                        "measured_rgb": measured_rgb.tolist(),
                        "reference_lab": reference_lab.tolist(),
                        "predicted_lab": predicted_lab.tolist(),
                        "delta_e_2000": float(delta_e),
                    }
                    for sample_id, measured_rgb, reference_lab,
                    predicted_lab, delta_e in zip(
                        ids, rgb, lab, predicted_all,
                        deltaE_ciede2000(
                            predicted_all.reshape(-1, 1, 3),
                            lab.reshape(-1, 1, 3),
                        ),
                    )
                },
                f,
                indent=2,
            )

        print("Debug patch overlay: it8-detected-patches.png")
        print("Debug patch report: " +
              os.path.splitext(output_path)[0] + "-patch-report.json")

def apply_calibration(image_path, output_path, profile_path):  
image = cv2.imread(image_path, cv2.IMREAD_COLOR)  
if image is None:  
raise FileNotFoundError(f"Cannot read input image: {image_path}")

    with open(profile_path, encoding="utf-8") as f:
        profile = json.load(f)

    if profile.get("model_version") != MODEL_VERSION:
        raise ValueError("Unsupported calibration profile version.")

    coefficients = np.asarray(profile["coefficients"], dtype=np.float64)

    # Convert OpenCV BGR to RGB, preserving the original image shape.
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    # Process in chunks to limit memory usage on large book scans.
    flat_rgb = rgb.reshape(-1, 3)
    corrected = np.empty((len(flat_rgb), 3), dtype=np.uint8)

    chunk_size = 250_000
    for start in range(0, len(flat_rgb), chunk_size):
        end = min(start + chunk_size, len(flat_rgb))

        predicted_lab = predict_lab(
            flat_rgb[start:end], coefficients
        )
        corrected[start:end] = lab_to_output_rgb(predicted_lab)

    corrected = corrected.reshape(h, w, 3)
    corrected_bgr = cv2.cvtColor(corrected, cv2.COLOR_RGB2BGR)

    if not cv2.imwrite(output_path, corrected_bgr):
        raise OSError(f"Could not write output image: {output_path}")

    print(f"Corrected image saved to {output_path}")

if **name** == "**main**":  
parser = argparse.ArgumentParser(  
description="IT8 scanner calibration and color correction"  
)

    parser.add_argument("input1", help="IT8 scan or image to correct")
    parser.add_argument(
        "input2",
        help="IT8 reference text file (calibration) or output image (apply)",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--calibrate", metavar="JSON")
    mode.add_argument("--apply-calibration", metavar="JSON")

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--regularization", type=float, default=0.01)

    parser.add_argument("--grid-x0", type=float, default=300)
    parser.add_argument("--grid-x1", type=float, default=3270)
    parser.add_argument("--grid-y0", type=float, default=150)
    parser.add_argument("--grid-y1", type=float, default=2130)

    args = parser.parse_args()

    if args.regularization < 0:
        parser.error("--regularization must be non-negative")

    if args.calibrate:
        calibrate(args.input1, args.input2, args.calibrate, args)
    else:
        apply_calibration(
            args.input1, args.input2, args.apply_calibration
        )

</CodeBlock>

## 2. Run the calibration

Save your reference data as `R190823.txt`, preserving the complete `BEGIN_DATA` section and its header.

First, install the dependencies if necessary:

Then generate the profile:

If the scan has the dimensions and coordinates you specified, the default grid should work. The script samples the central portion of each patch and writes `it8-detected-patches.png` so you can visually check the detected locations.

If your cropped image starts at the target's edges rather than retaining the original 3580 × 2400 coordinate system, adjust the grid arguments. For example, if the patch grid fills the image with no margins, use `--grid-x0 0 --grid-x1 WIDTH --grid-y0 0 --grid-y1 HEIGHT`, substituting the actual pixel dimensions.

## 3. Apply the profile to book scans

The calibration profile is reusable; you only need to scan and calibrate the target again if your scanner or scan settings change significantly.

**Important:** temporarily remove the old `linear_levels()` adjustment and histogram calibration when evaluating this new profile. Those operations would apply additional corrections after the IT8 correction and could reintroduce color errors.

## 4. Check the results before trusting the profile

The JSON includes mean, median, 95th-percentile, and maximum $`\Delta E_{00}`$ errors for the training and held-out patches. These provide a useful indication of how well the model reproduces the reference colors.

- If the validation error is much higher than the training error, the model may be overfitting.
- If the grayscale patches show a systematic color cast, investigate scanner settings, color management, and patch alignment.
- If the detected patch overlay is misplaced, fix the grid geometry before interpreting the errors.

One further technical caveat: the polynomial model assumes that the scanner's RGB values are stable and repeatable. For a production-grade archival workflow, a properly generated scanner ICC profile is preferable because it explicitly handles device color characterization and color-space conversion. SilverFast's IT8 workflow is designed to generate such profiles. \<Cite refs={\["turn842101search13","turn842101search0"\]}/\>

The script above is a practical starting point, not a guarantee of colorimetric accuracy. Its output should be evaluated against the measured patch errors and a few representative book scans before you integrate it into the main pipeline.

## Prompt:

10/28/2026, 7:44:09 PM

# Selected text:

## Selection 1

The script samples the central portion of each patch

## My request:

that sounds wrong, because when i zoom into these color patches, they can be quite noisy, so ideally, the script should take the average color of the (lets say) 25% center square of the color patch, so if a color patch has a width and height of about 124 pixels, then the script should use the center square from X=31 to X=93, and from Y=31 to Y=93, and use that center square to get the average color of the color patch. it should use only the center square to avoid accidentally getting pixels from neighbor color patches (scanner can always add small X/Y errors...)

## Response:

10/28/2026, 7:44:11 PM

I agree. The script should use the **mean RGB color of the central 50% × 50% square** of each patch, rather than sampling a few pixels or using a median. This averages out scanner noise while leaving a generous margin around each patch to avoid neighboring colors.

For a patch approximately 124 × 124 pixels:

<box gap={2} align="center">
  <box border radius="lg" padding={3} width="100%" maxWidth="360px" gap={2}>
    <box background="#8A8A8A" radius="md" padding={2} align="center" justify="center" aspectRatio="1/1" width="100%" maxWidth="280px">
      <box border={{size:2,color:"#FFFFFF"}} background="#A0A0A0" width="50%" aspectRatio="1/1" align="center" justify="center" radius="sm">
        <text color="#111111" weight="medium" textAlign="center" size="sm">Use this area</text>
        <text color="#111111" textAlign="center" size="xs">62 × 62 px</text>
      </box>
    </box>
    <text color="secondary" size="sm" textAlign="center">Ignore the outer 25% on every side.</text>
  </box>
</box>

The sampling coordinates relative to the patch's top-left corner should be:

- **X:** 31 through 92 (approximately; Python's end index is exclusive).
- **Y:** 31 through 92 (approximately; Python's end index is exclusive).
- **Measurement:** calculate the arithmetic mean of all pixel values independently for the red, green, and blue channels.

Replace the existing patch-sampling function with something like this:

This assumes `image` is loaded by OpenCV in BGR format and that the patch center and dimensions are already known. If the image has already been cropped or rotated, the grid coordinates must refer to that actual image.

**One important distinction:** averaging noisy pixels is useful, but it won't correct uneven illumination or systematic scanner color errors. The calibration model should handle those separately.

## Prompt:

10/28/2026, 7:54:47 PM

ok, i just noticed that you already had implemented a very similar function (sample_patch) in your first version of the script (0*12*-calibrate-it8.py), and the only differences are:

1.  sample_patch uses the central 40% of each cell, sample_patch_mean uses the central 50% of each cell
2.  sample_patch returns the median color, sample_patch_mean returns the mean color

which brings me to the question: which is better, median color or mean color? my naive guess would be that mean color should be correct... or which is better??

## Response:

10/28/2026, 7:54:49 PM

Your intuition is reasonable: **the mean is the better default for IT8 calibration, provided the sampled pixels are mostly valid pixels from the patch.** The median is more robust to outliers, so the best choice depends on what causes the noise in your scans.

## Mean vs. median

<box border radius="lg" gap={0}>
  <row padding={3} gap={3} align="start">
    <box background="surface-secondary" radius="md" padding={2} align="center" justify="center" size="44px">
      <icon name="chart-no-axes-combined" size="xl" />
    </box>
    <box flex="1" gap={1}>
      **Mean (arithmetic average)**

      - Uses every pixel and averages its RGB values.
      - Reduces random scanner noise efficiently.
      - Can be skewed by a few bad pixels, dust, scratches, or pixels from adjacent patches.
      - Estimates the average recorded color of the sampled area.
    </box>

  </row>
  <divider color="subtle" />
  <row padding={3} gap={3} align="start">
    <box background="surface-secondary" radius="md" padding={2} align="center" justify="center" size="44px">
      <icon name="chart-no-axes-column" size="xl" />
    </box>
    <box flex="1" gap={1}>
      **Median**

      - Less affected by isolated extreme pixels.
      - Useful when dust, defects, or occasional corrupted pixels are present.
      - Does not estimate the arithmetic average; it selects the middle value in each channel.
      - Can be preferable when the sampled area contains outliers.
    </box>

  </row>
</box>

## A simple example

Imagine a patch with 100 pixels in one color channel:

- 98 pixels have value 120.
- 2 pixels have value 255 because of dust or a scanning artifact.

The mean is **122.7**, while the median is **120**.

If all 100 pixels instead contain legitimate measurements fluctuating randomly around 120, the mean generally makes better use of the available data and estimates the underlying average more efficiently.

## What I'd use for your IT8 calibration

I'd recommend the following approach:

1.  **Sample the central 50% × 50% square** to keep away from patch boundaries and reduce the risk of neighboring colors contaminating the measurement.
2.  **Use the mean RGB value** of those pixels as the initial implementation.
3.  If the scans contain dust, scratches, or occasional bright/dark defects, consider a *trimmed mean*: discard a small percentage of the most extreme pixel values in each channel, then average the rest. This combines much of the mean's noise reduction with some robustness to outliers.

One subtle point: calculate the mean in the scanner image's recorded RGB values if that's what the calibration model expects. Don't convert individual pixels to Lab first and then assume that averaging their Lab values is equivalent; color-space conversion is nonlinear.

**My verdict:** start with the central 50% square and the ordinary mean. It's a sound baseline for a reasonably clean scan. If you later observe isolated defects or outliers, compare it with a trimmed mean using the same patches and evaluate which produces lower color errors on held-out reference patches.

## Prompt:

10/28/2026, 7:57:35 PM

# Selected text:

## Selection 1

If the scans contain dust, scratches, or occasional bright/dark defects

## My request:

no, the user should manually fix these defects, for example in GIMP with the "clone" tool, by cloning good regions of a patch to replace bad regions of a patch, so all patches are free from visual defects, so the mean color should be the correct color

## Response:

10/28/2026, 7:57:36 PM

Yes, that makes sense. If the user manually removes dust, scratches, and other defects in GIMP before calibration, **the arithmetic mean is the appropriate choice** for your workflow.

With defects removed and the sampling region restricted to the central 50% × 50% of each patch:

- **Mean RGB** averages all sampled pixels, reducing the influence of random scanner noise.
- **Median RGB** is more robust to outliers, but that robustness is less important when the patches have already been cleaned.
- **Central 50% sampling** reduces the chance of including pixels from adjacent patches if the scanner introduces small alignment errors.

So I'd use the **mean of the red, green, and blue channel values independently**, across all pixels in the central square. For a 124 × 124 pixel patch, that means averaging roughly 62 × 62 = 3,844 pixels per patch.

One small caveat: cloning should preserve the patch's actual color and texture as much as possible. If the cloned source region has a different tone, it can bias the mean even though the patch looks visually clean.

For your calibration script, I'd therefore settle on **central 50% square + arithmetic mean RGB** as the default sampling method.

------------------------------------------------------------------------

Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)

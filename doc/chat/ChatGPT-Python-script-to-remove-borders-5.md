# Python script debugging

**User:** Anonymous  
**Created:** 2026/7/12 11:38:36  
**Updated:** 2026/7/13 20:20:01  
**Exported:** 2026/7/13 20:26:40  
**Link:** [<https://chatgpt.com/c/6a536071-dc3c-83eb-b5de-6c9e223cd9a3>](https://chatgpt.com/c/6a536071-dc3c-83eb-b5de-6c9e223cd9a3)

## Prompt:

7/12/2026, 11:38:35 AM

help me fix this python script

the problem with this script is that it assumes that the binding edge is always straight, but this is wrong in most cases, because when i unbind books with a guillotine cutter, then the cut at the binding edge is not perfectly orthogonal to the other edges (top, bottom, outside)...

so the script should ignore the angle of the binding edge

and the script should not try to detect the top edge of the page, because my scanner removes the top edge

so the script should only use the bottom edge and the outside edge (opposite of the binding edge) to detect the page edges, and from there, try to rotate the page, so the bottom edge and the outside edge become orthogonal ("deskew")

also, to "restore the binding edge of the page", the script should not simply fill the missing width with the average color near the binding edge, but the script should try to preserve the vertical pattern near the binding edge, so the script should apply its "repair logic" line by line: get the average color of this line near the binding edge and then fill the missing width with this color

``` py
#!/usr/bin/env python3
"""
extract scanned page from gray background

restore the binding edge of the page
by filling the missing width
with the average color near the binding edge
"""

INPUT_DIR = "060-rotate-crop"
OUTPUT_DIR = "065-remove-page-borders"

# === Tuning parameters ===
DEBUG = True
DEBUG = False
BORDER_SIZE = 100  # pixels

RANSAC_ITER = 400
RANSAC_INLIER_DIST = 6.0
# RANSAC_MIN_INLIERS = 30
RANSAC_MIN_INLIERS = 20

THRESH_HIGH_PERCENTILE = 99
THRESH_MIN = 200

import os
import re
import math
import random
import numpy as np
import cv2

from _shared import (
    load_config,
    get_page_num,
)

config = load_config()

scan_x = config.scan_x
scan_y = config.scan_y

config.scan_aspect = scan_x / scan_y

ASPECT = config.scan_aspect

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def save_dbg(img, path):
    ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, img)

def percentile_threshold(gray):
    high_p = np.percentile(gray, THRESH_HIGH_PERCENTILE)
    thr = max(THRESH_MIN, int(high_p * 0.95))
    _, mask = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
    return mask, thr, int(high_p)

def detect_vertical_streaks(mask, approx_width=3, length_thresh_ratio=0.15):
    h, w = mask.shape
    kx = approx_width
    ky = max(15, int(h * 0.02))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
    long_vertical = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(long_vertical, connectivity=8)
    streak_mask = np.zeros_like(mask)
    length_thresh = max(10, int(h * length_thresh_ratio))
    for i in range(1, num_labels):
        x, y, ww, hh, area = stats[i]
        if hh >= length_thresh and ww <= max(5, int(w * 0.01)):
            streak_mask[labels == i] = 255
    return streak_mask

def keep_largest_component(mask):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    out = np.zeros_like(mask)
    out[labels == best] = 255
    return out

def contour_to_pts(contour):
    return contour.reshape(-1,2)

def fit_line_ransac(pts, iterations=RANSAC_ITER, inlier_dist=RANSAC_INLIER_DIST, min_inliers=RANSAC_MIN_INLIERS):
    if len(pts) < 2:
        raise ValueError("Not enough points")
    best_inliers = None
    best_model = None
    n = len(pts)
    ptsf = pts.astype(np.float32)
    for _ in range(iterations):
        i1, i2 = random.sample(range(n), 2)
        p1 = ptsf[i1]; p2 = ptsf[i2]
        vx = float(p2[0] - p1[0]); vy = float(p2[1] - p1[1])
        if vx == 0 and vy == 0:
            continue
        dists = np.abs(vy*(ptsf[:,0]-p1[0]) - vx*(ptsf[:,1]-p1[1])) / (math.hypot(vx, vy) + 1e-12)
        inliers = dists <= inlier_dist
        cnt = int(inliers.sum())
        if cnt >= min_inliers and (best_inliers is None or cnt > int(best_inliers.sum())):
            best_inliers = inliers.copy()
            best_model = (vx, vy, float(p1[0]), float(p1[1]))
    if best_model is None:
        vx, vy, x0, y0 = cv2.fitLine(ptsf, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        inlier_mask = np.ones(len(pts), dtype=bool)
        return float(vx), float(vy), float(x0), float(y0), inlier_mask
    inlier_pts = ptsf[best_inliers]
    vx, vy, x0, y0 = cv2.fitLine(inlier_pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    inlier_mask = best_inliers
    return float(vx), float(vy), float(x0), float(y0), inlier_mask

def intersect_lines(l1, l2):
    vx1, vy1, x1, y1 = l1
    vx2, vy2, x2, y2 = l2
    A = np.array([[vx1, -vx2], [vy1, -vy2]], dtype=np.float32)
    b = np.array([x2 - x1, y2 - y1], dtype=np.float32)
    det = np.linalg.det(A)
    if abs(det) < 1e-8:
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    t1, t2 = np.linalg.solve(A, b)
    xi = x1 + t1 * vx1
    yi = y1 + t1 * vy1
    return float(xi), float(yi)

def build_affine_without_shear(pt_v_top, pt_v_bot, expected_w, expected_h, bad_on_left, img, dbgdir=None):
    """
    Build a source triangle that enforces orthogonal page axes (no shear) using:
      - pt_v_top: intersection of top line and good vertical (top-right when good vertical is right)
      - pt_v_bot: intersection of bottom line and good vertical (bottom-right when good vertical is right)
    Returns (M_aff, src_corners_dict, dst_corners_dict, reason)
    - M_aff: 2x3 affine matrix or None on failure
    - src_corners_dict: dict of TL, TR, BL, BR (np.float32 2-vectors)
    - dst_corners_dict: dict of TL, TR, BL, BR in destination coordinates
    - reason: diagnostic string
    """
    # convert to vectors
    pt_v_top = np.array(pt_v_top, dtype=np.float32)
    pt_v_bot = np.array(pt_v_bot, dtype=np.float32)

    # compute direction along top edge: vector from bottom intersection to top intersection projected horizontally
    # but we don't have explicit top_line direction vector here; we will approximate using (pt_v_top - pt_v_bot) rotated?
    # Better: caller should supply top_line direction (vx,vy). If you don't have it, approximate from nearby contour points.
    # For compatibility with your code, expect you have top_vx, top_vy; if not, fall back to vector between two top contour points.
    # Here we'll assume top_vx, top_vy are available in outer scope; otherwise compute from pt_v_top->pt_v_bot displacement projected perpendicular:
    # To keep this helper self-contained, we compute top_dir as average of (pt_v_top_to_some_right) - but simplest robust approach:
    # compute approximate top direction by taking small offset vector along top by sampling image gradient: fallback to (1,0).

    # --- Here caller should provide top_unit; we'll compute a safe top_unit using neighbor pixels if available ---
    # We'll attempt to derive a top_unit by sampling a short step along the image: use vector from pt_v_top to pt_v_top projected to image center
    cx, cy = img.shape[1] / 2.0, img.shape[0] / 2.0
    # prefer vector pointing rightwards by projecting displacement to center
    approx = pt_v_top - np.array([cx, cy], dtype=np.float32)
    if np.linalg.norm(approx) < 1e-6:
        approx = np.array([1.0, 0.0], dtype=np.float32)
    ux, uy = approx / (np.linalg.norm(approx) + 1e-12)

    # Force ux positive (pointing to the right) — we want u to be rightward
    if ux < 0:
        ux, uy = -ux, -uy

    # make perpendicular downwards: p = (-uy, ux)
    px, py = -uy, ux

    # normalize again (just in case)
    n_u = math.hypot(ux, uy)
    n_p = math.hypot(px, py)
    if n_u < 1e-6 or n_p < 1e-6:
        return None, None, None, "degenerate_axes"
    ux, uy = ux / n_u, uy / n_u
    px, py = px / n_p, py / n_p

    # Now construct corners. We know pt_v_top/pt_v_bot correspond to the known vertical edge:
    # If bad_on_left is True, good vertical is the RIGHT edge => pt_v_top == TR, pt_v_bot == BR.
    # If bad_on_left is False, good vertical is LEFT edge => pt_v_top == TL, pt_v_bot == BL.
    if bad_on_left:
        src_TR = pt_v_top
        src_BR = pt_v_bot
        # compute TL and BL by moving left along top unit by expected_w
        src_TL = src_TR - np.array([ux, uy], dtype=np.float32) * float(expected_w)
        src_BL = src_BR - np.array([ux, uy], dtype=np.float32) * float(expected_w)
    else:
        src_TL = pt_v_top
        src_BL = pt_v_bot
        # compute TR and BR by moving right along top unit by expected_w
        src_TR = src_TL + np.array([ux, uy], dtype=np.float32) * float(expected_w)
        src_BR = src_BL + np.array([ux, uy], dtype=np.float32) * float(expected_w)

    # Optional: enforce vertical height by projecting (TL->BL) onto perp and rescale to expected_h
    # Compute current vertical vector and its projection onto perp (px,py)
    cur_v = src_BL - src_TL
    proj = cur_v.dot(np.array([px, py], dtype=np.float32))
    # if projection is tiny, fall back but otherwise adjust BL to ensure exact page height
    if abs(proj) > 1e-6:
        # adjust scale to exact expected_h
        scale = float(expected_h) / proj
        # recompute BL, BR as TL + perp*expected_h and TR + perp*expected_h
        src_BL = src_TL + np.array([px, py], dtype=np.float32) * float(expected_h)
        src_BR = src_TR + np.array([px, py], dtype=np.float32) * float(expected_h)
    else:
        # if proj nearly zero, we cannot rely on vertical measurement; still use perpendicular step
        src_BL = src_TL + np.array([px, py], dtype=np.float32) * float(expected_h)
        src_BR = src_TR + np.array([px, py], dtype=np.float32) * float(expected_h)

    # Build src/dst dicts
    src_corners = {
        "TL": src_TL.astype(np.float32),
        "TR": src_TR.astype(np.float32),
        "BL": src_BL.astype(np.float32),
        "BR": src_BR.astype(np.float32)
    }
    dst_corners = {
        "TL": np.array([0.0, 0.0], dtype=np.float32),
        "TR": np.array([expected_w - 1.0, 0.0], dtype=np.float32),
        "BL": np.array([0.0, expected_h - 1.0], dtype=np.float32),
        "BR": np.array([expected_w - 1.0, expected_h - 1.0], dtype=np.float32)
    }

    # choose triangle for affine depending on orientation (map TL, BL, TR -> dst TL, BL, TR)
    src_tri = np.vstack([src_corners["TL"], src_corners["BL"], src_corners["TR"]]).astype(np.float32)
    dst_tri = np.vstack([dst_corners["TL"], dst_corners["BL"], dst_corners["TR"]]).astype(np.float32)

    # validate triangle
    area = abs(0.5 * (src_tri[0,0]*(src_tri[1,1]-src_tri[2,1]) + src_tri[1,0]*(src_tri[2,1]-src_tri[0,1]) + src_tri[2,0]*(src_tri[0,1]-src_tri[1,1])))
    if area < 1.0 or not np.isfinite(src_tri).all():
        return None, src_corners, dst_corners, f"invalid_src_tri_area={area:.3f}"

    try:
        M = cv2.getAffineTransform(src_tri, dst_tri)
    except cv2.error as e:
        return None, src_corners, dst_corners, f"getAffineTransform_failed:{e}"

    # debug overlay: draw corners & axes if dbgdir provided
    if dbgdir is not None:
        vis = img.copy()
        def draw_pt(pt, color=(0,0,255), tag=""):
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 6, color, -1)
            if tag:
                cv2.putText(vis, tag, (int(pt[0]+6), int(pt[1]-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        draw_pt(src_corners["TL"], (0,255,0), "TL")
        draw_pt(src_corners["TR"], (0,128,255), "TR")
        draw_pt(src_corners["BL"], (255,0,0), "BL")
        draw_pt(src_corners["BR"], (255,255,0), "BR")
        # draw axis arrows from TL
        origin = src_corners["TL"]
        arrow_u = origin + np.array([ux, uy], dtype=np.float32) * 80.0
        arrow_p = origin + np.array([px, py], dtype=np.float32) * 80.0
        cv2.arrowedLine(vis, (int(origin[0]), int(origin[1])), (int(arrow_u[0]), int(arrow_u[1])), (255,0,255), 2)
        cv2.arrowedLine(vis, (int(origin[0]), int(origin[1])), (int(arrow_p[0]), int(arrow_p[1])), (0,255,255), 2)
        cv2.imwrite(os.path.join(dbgdir, "debug_axes_and_corners.png"), vis)

    return M, src_corners, dst_corners, "ok"

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

def process_image(in_path, out_path):
    """
    Robust page extraction that handles missing top (or bottom) edges.
    - preserves original colors (warps the original image)
    - builds orthogonal axes (no shear)
    - if top is missing, uses bottom + good vertical to infer top by moving up expected_h
    - fills outside the filled page polygon with pure white
    - extensive debug output in OUTPUT_DIR/debug/<page_num>/
    """
    # ---------- small helpers ----------
    def ensure_dir(p):
        os.makedirs(p, exist_ok=True)

    def save_dbg(img, path):
        ensure_dir(os.path.dirname(path))
        cv2.imwrite(path, img)

    def percentile_threshold(gray):
        high_p = np.percentile(gray, THRESH_HIGH_PERCENTILE)
        thr = max(THRESH_MIN, int(high_p * 0.95))
        _, mask = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
        return mask, thr, int(high_p)

    def keep_largest_component(mask):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels <= 1:
            return mask
        areas = stats[1:, cv2.CC_STAT_AREA]
        best = 1 + int(np.argmax(areas))
        out = np.zeros_like(mask)
        out[labels == best] = 255
        return out

    def detect_vertical_streaks(mask, approx_width=3, length_thresh_ratio=0.15):
        h, w = mask.shape
        kx = approx_width
        ky = max(15, int(h * 0.02))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
        long_vertical = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(long_vertical, connectivity=8)
        streak_mask = np.zeros_like(mask)
        length_thresh = max(10, int(h * length_thresh_ratio))
        for i in range(1, num_labels):
            x, y, ww, hh, area = stats[i]
            if hh >= length_thresh and ww <= max(5, int(w * 0.01)):
                streak_mask[labels == i] = 255
        return streak_mask

    def contour_to_pts(c):
        return c.reshape(-1, 2)

    def fit_line_ransac(pts, iterations=RANSAC_ITER, inlier_dist=RANSAC_INLIER_DIST, min_inliers=RANSAC_MIN_INLIERS):
        if len(pts) < 2:
            raise ValueError("Not enough points for line fit")
        best_inliers = None
        best_cnt = 0
        best_model = None
        ptsf = pts.astype(np.float32)
        n = len(ptsf)
        for _ in range(iterations):
            i1, i2 = random.sample(range(n), 2)
            p1 = ptsf[i1]; p2 = ptsf[i2]
            vx = float(p2[0] - p1[0]); vy = float(p2[1] - p1[1])
            if abs(vx) < 1e-6 and abs(vy) < 1e-6:
                continue
            dists = np.abs(vy*(ptsf[:,0]-p1[0]) - vx*(ptsf[:,1]-p1[1])) / (math.hypot(vx, vy) + 1e-12)
            inliers = dists <= inlier_dist
            cnt = int(inliers.sum())
            if cnt >= min_inliers and cnt > best_cnt:
                best_cnt = cnt
                best_inliers = inliers.copy()
                best_model = (vx, vy, float(p1[0]), float(p1[1]))
        if best_model is None:
            vx, vy, x0, y0 = cv2.fitLine(ptsf, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
            inlier_mask = np.ones(len(ptsf), dtype=bool)
            return float(vx), float(vy), float(x0), float(y0), inlier_mask
        inlier_pts = ptsf[best_inliers]
        vx, vy, x0, y0 = cv2.fitLine(inlier_pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        return float(vx), float(vy), float(x0), float(y0), best_inliers

    def intersect_lines(l1, l2):
        vx1, vy1, x1, y1 = l1
        vx2, vy2, x2, y2 = l2
        A = np.array([[vx1, -vx2], [vy1, -vy2]], dtype=np.float32)
        b = np.array([x2 - x1, y2 - y1], dtype=np.float32)
        det = np.linalg.det(A)
        if abs(det) < 1e-8:
            return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        t1, t2 = np.linalg.solve(A, b)
        xi = x1 + t1 * vx1
        yi = y1 + t1 * vy1
        return float(xi), float(yi)

    # ---------- start ----------
    fname = os.path.basename(in_path)
    m = re.match(r"^(\d+)", fname)
    page_num = int(m.group(1)) if m else 0
    bad_on_left = (page_num % 2 == 1)

    img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print("Failed to read", in_path); return
    input_is_grayscale = (len(img.shape) == 2)
    if len(img.shape) == 2:
        # OpenCV expects 3-channel images for many operations
        # later convert back to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    H_img, W_img = img.shape[:2]

    expected_w = int(round(ASPECT * H_img))
    expected_h = H_img

    dbgdir = os.path.join(OUTPUT_DIR, "debug", f"{page_num:03d}")
    if DEBUG: ensure_dir(dbgdir)

    # --- threshold for geometry only ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask_init, thr, hp = percentile_threshold(gray)
    if cv2.mean(gray)[0] < 127:
        mask_init = cv2.bitwise_not(mask_init)
    if DEBUG:
        save_dbg(gray, os.path.join(dbgdir, "01_gray.png"))
        save_dbg(mask_init, os.path.join(dbgdir, f"02_thresh_thr{thr}_hp{hp}.png"))

    # without a9dc2b24e6b0d1d49b6fc232223d6431ba3442a5 bad: fix perspective transform for broken ADF scanners
    # Invert if necessary, so the page is always "white" for thresholding
    # We'll use Otsu's method to detect high-contrast area
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Determine whether the page is darker or lighter than background
    mean_val = cv2.mean(gray, mask=None)[0]
    if mean_val < 127:  # dark page: invert mask
        mask = cv2.bitwise_not(mask)

    # Morphology to remove small gaps
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"Warning: no contours found in {in_path}")
        return
    page_contour = max(contours, key=cv2.contourArea)

    # Approximate contour to quadrilateral
    epsilon = 0.02 * cv2.arcLength(page_contour, True)
    approx = cv2.approxPolyDP(page_contour, epsilon, True)
    if len(approx) != 4:
        approx = cv2.convexHull(page_contour)
        if len(approx) < 4:
            print(f"Warning: not enough points for perspective in {in_path}")
            return
        # pick 4 extreme points
        pts = np.array([
            approx[approx[:,0,0].argmin()][0],  # leftmost
            approx[approx[:,0,1].argmin()][0],  # topmost
            approx[approx[:,0,0].argmax()][0],  # rightmost
            approx[approx[:,0,1].argmax()][0]   # bottommost
        ])
    else:
        pts = approx.reshape(4,2)

    rect = order_points(pts)

    # Perspective transform
    widthA = np.linalg.norm(rect[2] - rect[3])
    widthB = np.linalg.norm(rect[1] - rect[0])
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(rect[1] - rect[2])
    heightB = np.linalg.norm(rect[0] - rect[3])
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    # # Add internal white border
    # h, w = warped.shape[:2]
    # canvas = np.ones_like(warped) * 255
    # b = BORDER_SIZE
    # canvas[b:h-b, b:w-b] = warped[b:h-b, b:w-b]

    h, w = warped.shape[:2]
    b = BORDER_SIZE

    # Prepare a canvas with the same content as warped
    canvas = warped.copy()

    # Function to compute average color along a strip
    def avg_color_strip(img, axis, start, end, strip_width=1):
        if axis == 'top':
            strip = img[start:end, :, :]
        elif axis == 'bottom':
            strip = img[h-end:h-start, :, :]
        elif axis == 'left':
            strip = img[:, start:end, :]
        elif axis == 'right':
            strip = img[:, w-end:w-start, :]
        else:
            raise ValueError("Invalid axis")
        return np.mean(strip, axis=(0,1)).astype(np.uint8)

    # Fill borders with local average color
    canvas[0:b, :, :] = avg_color_strip(canvas, 'top', 50, 100)       # top border
    canvas[h-b:h, :, :] = avg_color_strip(canvas, 'bottom', 50, 100)  # bottom border
    canvas[:, 0:b, :] = avg_color_strip(canvas, 'left', 50, 100)      # left border
    canvas[:, w-b:w, :] = avg_color_strip(canvas, 'right', 50, 100)   # right border

    if input_is_grayscale:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

    ensure_dir(os.path.dirname(out_path))
    cv2.imwrite(out_path, canvas)
    print(f"writing {out_path}")

def main():
    ensure_dir(OUTPUT_DIR)
    # TODO use image_format from 030-measure-page-size.txt
    image_format = "jpg"
    files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(f".{image_format}")])
    if not files:
        print("No image files found in", INPUT_DIR)
        return
    for fname in files:
        m = re.match(r"^(\d+)", fname)
        # page_num = int(m.group(1)) if m else 0
        page_num = int(m.group(1)) # can throw
        # if page_num != 14: continue # debug
        # if not page_num in [340, 345]: continue # debug
        # if page_num < 300: continue # debug
        # if page_num != 320: continue # debug
        in_path = os.path.join(INPUT_DIR, fname)
        out_path = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(out_path):
            continue
        try:
            process_image(in_path, out_path)
        except Exception as exc:
            print(f"Error processing {fname}: {exc}")
            raise

if __name__ == "__main__":
    main()
```

## Response:

7/12/2026, 11:38:35 AM

I think the current script is solving the wrong geometry.

It still assumes "find four page edges, fit a quadrilateral, rectify it". That works for intact pages, but **it is fundamentally the wrong model for guillotine-cut books**.

Your pages have these properties:

- ✅ outside edge is straight
- ✅ bottom edge is straight
- ❌ binding edge is often cut at an arbitrary angle
- ❌ top edge is frequently missing because the scanner cropped it away

So you actually have **only two reliable edges**.

## I'd completely change the geometry

Instead of estimating a quadrilateral:

1.  detect outside edge
2.  detect bottom edge
3.  ignore every other edge
4.  rotate the image so those two lines become perpendicular
5.  compute the missing binding edge from the expected page width

This is a much more stable problem.

------------------------------------------------------------------------

## Step 1: detect only two edges

Don't use `approxPolyDP()` at all.

Instead:

- extract the page contour
- classify contour points

For example:

    outside candidates:
        rightmost 10% of contour points
        (or leftmost on verso pages)

    bottom candidates:
        bottommost 10%

Fit each with RANSAC.

    outside line

    |

    |

    |

    +------------------ bottom line

Ignore every other contour point.

------------------------------------------------------------------------

## Step 2: compute rotation from those two lines

Suppose

    outside angle = 89.2°

    bottom angle = -0.6°

Instead of trusting one edge, compute the rotation that makes them orthogonal.

For example

    bottom should become 0°

    outside should become 90°

The optimal rotation is

    rotation =
    (
    -bottom_angle
    +
    (90 - outside_angle)
    )
    /
    2

or more generally minimize

    (bottom_rot)^2 +
    ((outside_rot)-90)^2

This averages both measurements and is much more robust.

------------------------------------------------------------------------

## Step 3: rotate only

After that

    rotated = warpAffine(...)

Now

- bottom is horizontal
- outside is vertical

No perspective transform.

No projective warp.

No inferred top edge.

No inferred binding edge.

------------------------------------------------------------------------

## Step 4: crop from those two edges

Now the page position is known.

Example (recto):

    binding = outside_x - expected_width

    outside = detected

    bottom = detected

    top = bottom - expected_height

You don't need to detect the top.

You compute it.

Likewise

    binding_x =
    outside_x
    -
    expected_width

The binding edge becomes purely geometric.

------------------------------------------------------------------------

# This is much better than perspective correction

Your current script assumes

     _________
    |        /
    |       /
    |______/

is perspective.

It isn't.

It's

     _________
    |        |
    |        |
    |_______/

where only the binding side was cut diagonally.

Trying to "unskew" that diagonal inevitably distorts the page.

Ignoring it completely avoids introducing distortion.

------------------------------------------------------------------------

# Don't use getPerspectiveTransform()

I would remove this entire block:

    approxPolyDP()

    order_points()

    getPerspectiveTransform()

    warpPerspective()

None of it is appropriate for this problem.

A simple affine rotation is sufficient.

------------------------------------------------------------------------

# Restoring the binding edge

I also agree with your second point.

Currently the code computes

    average of entire left strip

    AAAAAAAAAAAA
    AAAAAAAAAAAA
    AAAAAAAAAAAA
    AAAAAAAAAAAA

which destroys gradients.

Instead do it row-by-row.

For every row

    BBBBBBBB
    CCCCCCCC
    DDDDDDDD
    EEEEEEEE

compute

    mean(row[x:x+N])

and fill only that row.

Pseudo-code:

``` python
sample_width = 20

for y in range(height):

    sample = img[y, bind_x:bind_x+sample_width]

    color = sample.mean(axis=0)

    img[y, :bind_x] = color
```

or for the opposite side

``` python
for y in range(height):

    sample = img[y, bind_x-sample_width:bind_x]

    color = sample.mean(axis=0)

    img[y, bind_x:] = color
```

This preserves

- paper yellowing
- scanner illumination gradients
- shadows
- foxing
- stains
- page discoloration

because each scanline gets its own color.

------------------------------------------------------------------------

# An even better repair

Instead of using the **average** of each row, I'd copy the first few pixels and extend them.

For each row:

    xxxxxxxx|abcdefghijk
            ^
       binding edge

Take

    abcdefgh

and compute either:

- the median color, or
- a Gaussian-weighted average favoring pixels closest to the binding edge.

That preserves the local paper tone even better than averaging over a wide strip.

## Overall recommendation

I would simplify the pipeline to:

1.  Threshold to isolate the page.
2.  Find the largest contour.
3.  Fit **only the outside edge** and **bottom edge** with RANSAC.
4.  Compute a single affine rotation that makes those edges orthogonal.
5.  Rotate the image.
6.  Compute the page rectangle from the known page dimensions using the outside and bottom edges only.
7.  Crop to that rectangle.
8.  Restore the missing binding margin by filling it **line-by-line** using colors sampled immediately adjacent to the binding edge.

This approach matches the physical reality of guillotine-cut books much better than trying to infer a four-corner quadrilateral from incomplete and intentionally irregular page edges.

## Prompt:

7/12/2026, 12:11:55 PM

i just had a genius idea...

i can feed my book pages to my scanner in a rotated position, so the binding edge becomes the "top edge" in the raw scan, so then i have 3 valid edges, so from there, the "deskew" operation becomes more stable

this works for most books, because most books have a page height under 215mm, so only in rare cases, i need the "deskew by 2 edges" logic, and in most cases, i can use a "deskew by 3 edges" logic...

can you verify that my old script already has this "deskew by 3 edges" logic? it should ignore the top edge, and only use the 3 other pages to detect the actual page edges...

but when i rotate the raw scans, then the scanner's top edge becomes the inner edge (binding edge) (left side on odd pages, right side on even pages)

in my config i have

``` py
# --- Rotation settings ---
do_rotate = True
# book binding side = scanner top side
rotate_odd, rotate_even = 270, 90
```

so the script should check for the condition  
`config.do_rotate and config.rotate_odd == 270 and config.rotate_even == 90`

## Response:

7/12/2026, 12:11:55 PM

Yes, and this is actually a clever workaround.

By rotating the pages before scanning, you're changing **which physical edge the scanner destroys**.

Originally:

    scanner

    +--------------------+   <- scanner crops this edge (BAD)
    |                    |
    |                    |
    |                    |
    |                    |
    +--------------------+

After rotating 270°/90°:

    scanner

    +--------------------+
    |                    |
    |                    |
    |                    |
    |                    |
    +--------------------+

    ^
    this cropped edge is now the book binding

So after rotating the image back to its normal orientation, you have:

- ✔ outside edge
- ✔ top edge
- ✔ bottom edge
- ✘ binding edge (cropped)

That's exactly the edge you wanted to ignore anyway.

------------------------------------------------------------------------

## Does your current script already do this?

Unfortunately, **no.**

The script does **not** have explicit "ignore one edge" logic.

Instead it does this:

``` python
contours = cv2.findContours(...)
page_contour = max(contours, key=cv2.contourArea)

approx = cv2.approxPolyDP(page_contour, ...)
```

Then

``` python
order_points(...)
```

Then

``` python
cv2.getPerspectiveTransform(...)
```

That assumes

> I have a quadrilateral.

It never says

> ignore the binding edge.

------------------------------------------------------------------------

It only falls back to

``` python
convexHull(...)
```

when `approxPolyDP` doesn't produce four points.

That's still trying to infer four corners.

------------------------------------------------------------------------

## There is some older code that *looks* like it wanted to ignore one edge...

Near the top of the file you have

``` python
detect_vertical_streaks()

fit_line_ransac()

intersect_lines()

build_affine_without_shear(...)
```

Notice the comment inside `build_affine_without_shear()`:

> If bad_on_left is True, good vertical is the RIGHT edge

That is exactly the design you're describing.

It assumes

- one vertical edge is unreliable
- the opposite vertical edge is reliable

which is perfect for a missing binding edge.

------------------------------------------------------------------------

Even more telling is

``` python
pt_v_top
pt_v_bot
```

Those are

    top  x outside edge

    bottom x outside edge

There is **no intersection with the binding edge**.

That function was clearly intended to work from

- top
- bottom
- outside

only.

------------------------------------------------------------------------

## But it isn't actually used anymore

Later in `process_image()` the whole pipeline changes to

``` python
approxPolyDP()

↓

order_points()

↓

getPerspectiveTransform()
```

so all that careful RANSAC code is effectively bypassed.

------------------------------------------------------------------------

## Your rotated scanning makes the problem much easier

When

``` python
config.do_rotate
and rotate_odd == 270
and rotate_even == 90
```

then after rotation back to reading orientation the missing edge is always

    odd page

    +-----------+
    |           |
    |           |
    |           |
    /////////////  <- missing binding

    even page

    +-----------+
    |           |
    |           |
    |           |
    \\\\\\\\\\\\\  <- missing binding

So the reliable edges are always

- top
- bottom
- outside

which is the ideal case.

------------------------------------------------------------------------

## I'd explicitly add a special mode

Rather than trying to detect this implicitly, I'd write something like

``` python
use_three_edge_mode = (
    config.do_rotate
    and config.rotate_odd == 270
    and config.rotate_even == 90
)
```

Then:

``` python
if use_three_edge_mode:
    detect_top()
    detect_bottom()
    detect_outside()
    ignore_binding()
else:
    detect_bottom()
    detect_outside()
```

That keeps the logic clear and makes it obvious why the algorithm changes.

## This is probably the best long-term solution

I actually think this is better than investing a lot of effort into making the two-edge algorithm perfect.

With the rotated scans, you'll have three reliable edges for the overwhelming majority of books:

- the scanner no longer destroys a useful edge,
- the binding edge is the only missing one (which you already know is unreliable after guillotine cutting),
- and the geometry becomes much more constrained and stable.

The only time you'd need the two-edge fallback is for unusually tall books that can't be rotated to fit on the scanner. That makes the simpler three-edge algorithm your primary path and the two-edge version a rare fallback, which is a much cleaner design.

## Prompt:

7/12/2026, 1:19:48 PM

> With the rotated scans, you'll have three reliable edges for the overwhelming majority of books:

yes. help me implement this solution

## Response:

7/12/2026, 1:19:49 PM

Good. I would implement this as a **new primary geometry path** and keep your current two-edge idea as fallback.

The important change is: **do not detect a quadrilateral anymore**. Detect three independent lines:

- top edge
- bottom edge
- outside edge

Then construct the page rectangle from those three lines.

The binding edge is never detected.

------------------------------------------------------------------------

## 1. Add a mode switch

Near your config:

``` python
USE_THREE_EDGE_DESKEW = (
    config.do_rotate
    and config.rotate_odd == 270
    and config.rotate_even == 90
)
```

------------------------------------------------------------------------

## 2. Add a helper to classify contour points

After you get:

``` python
page_contour = max(contours, key=cv2.contourArea)
```

replace the quadrilateral logic.

Add:

``` python
def split_edge_candidates(contour, bad_on_left):
    pts = contour.reshape(-1, 2)

    xs = pts[:,0]
    ys = pts[:,1]

    w = xs.max() - xs.min()
    h = ys.max() - ys.min()

    margin_x = w * 0.15
    margin_y = h * 0.15

    # outside edge
    if bad_on_left:
        outside = pts[xs > xs.max() - margin_x]
    else:
        outside = pts[xs < xs.min() + margin_x]

    # top edge
    top = pts[ys < ys.min() + margin_y]

    # bottom edge
    bottom = pts[ys > ys.max() - margin_y]

    return top, bottom, outside
```

This deliberately ignores the binding side.

------------------------------------------------------------------------

## 3. Fit the three lines

Add:

``` python
top_pts, bottom_pts, outside_pts = split_edge_candidates(
    page_contour,
    bad_on_left
)

top_line = fit_line_ransac(top_pts)[:4]
bottom_line = fit_line_ransac(bottom_pts)[:4]
outside_line = fit_line_ransac(outside_pts)[:4]
```

Now you have:

    top_line
    ------------
                |
                |
                |
    bottom_line

and the outside vertical.

------------------------------------------------------------------------

## 4. Deskew using all three edges

Add:

``` python
def line_angle(line):
    vx, vy, _, _ = line
    return math.atan2(vy, vx)

def normalize_angle_deg(a):
    while a < -90:
        a += 180
    while a > 90:
        a -= 180
    return a
```

Then:

``` python
top_angle = math.degrees(line_angle(top_line))
bottom_angle = math.degrees(line_angle(bottom_line))
outside_angle = math.degrees(line_angle(outside_line))
```

The desired angles are:

    top      = 0°
    bottom   = 0°
    outside  = 90°

Compute correction:

``` python
rotation_error = (
    -top_angle
    -bottom_angle
    +(90 - outside_angle)
) / 3.0
```

Then:

``` python
Mrot = cv2.getRotationMatrix2D(
    (W_img/2, H_img/2),
    rotation_error,
    1.0
)

rotated = cv2.warpAffine(
    img,
    Mrot,
    (W_img, H_img),
    borderValue=(255,255,255)
)
```

Now the page is deskewed.

------------------------------------------------------------------------

## 5. Re-detect only the outside edge after rotation

This is important.

Do not transform old coordinates.

After rotation:

``` python
gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)

mask = ...
contours,_ = cv2.findContours(...)
page_contour=max(contours,key=cv2.contourArea)
```

Then:

``` python
_,_,outside_pts = split_edge_candidates(
    page_contour,
    bad_on_left
)

outside_line = fit_line_ransac(outside_pts)[:4]
```

Now the outside edge is accurate.

------------------------------------------------------------------------

## 6. Create the final crop rectangle

Because the page size is known:

``` python
expected_h = H_img
expected_w = int(round(ASPECT * expected_h))
```

Find the outside edge position.

Example:

``` python
vx,vy,x0,y0 = outside_line
```

Since the image is deskewed, this line should be almost vertical.

For odd pages:

``` python
if bad_on_left:
    x_out = x0
    x0_page = x_out
    x1_page = x_out + expected_w
else:
    x_out = x0
    x1_page = x_out
    x0_page = x_out - expected_w
```

Then crop:

``` python
crop = rotated[
    0:expected_h,
    int(x0_page):int(x1_page)
]
```

No perspective transform.

No four corners.

------------------------------------------------------------------------

## 7. Replace binding repair

After crop:

``` python
def repair_binding(img, bad_on_left, width=50):

    h,w = img.shape[:2]

    if bad_on_left:
        sample_range = range(width)
        fill_range = range(width)
        sample_x = width
    else:
        sample_range = range(w-width,w)
        fill_range = range(w-width,w)
        sample_x = w-width

    for y in range(h):

        if bad_on_left:
            sample = img[y, width:width+20]
            color = np.mean(sample,axis=0)
            img[y,:width] = color

        else:
            sample = img[y,w-width-20:w-width]
            color = np.mean(sample,axis=0)
            img[y,w-width:] = color

    return img
```

This preserves vertical variation.

------------------------------------------------------------------------

## One more important thing

Your current `expected_w` calculation:

``` python
expected_w = int(round(ASPECT * H_img))
```

assumes the rotated scan still has the same height.

That is only true **after you rotate the scanner image back**.

So I would move the page-size calculation **after** rotation.

------------------------------------------------------------------------

## Suggested migration strategy

Don't delete your old code yet.

Structure it:

``` python
if USE_THREE_EDGE_DESKEW:
    process_three_edge_mode(...)
else:
    process_two_edge_mode(...)
```

Then you can compare output folders.

I would expect the three-edge mode to outperform the current script by a lot because it matches the physical scanning setup instead of fighting the missing binding edge.

## Prompt:

7/12/2026, 2:08:07 PM

ok, i have tried to apply your fixes, and the resulting script is this

``` py
#!/usr/bin/env python3
"""
extract scanned page from gray background

restore the binding edge of the page
by filling the missing width
with the average color near the binding edge
"""

INPUT_DIR = "060-rotate-crop"
OUTPUT_DIR = "065-remove-page-borders"

# === Tuning parameters ===
DEBUG = True
DEBUG = False
BORDER_SIZE = 100  # pixels

RANSAC_ITER = 400
RANSAC_INLIER_DIST = 6.0
# RANSAC_MIN_INLIERS = 30
RANSAC_MIN_INLIERS = 20

THRESH_HIGH_PERCENTILE = 99
THRESH_MIN = 200

import os
import re
import math
import random
import numpy as np
import cv2

from _shared import (
    load_config,
    get_page_num,
)

config = load_config()

scan_x = config.scan_x
scan_y = config.scan_y

config.scan_aspect = scan_x / scan_y

ASPECT = config.scan_aspect

# book binding side = scanner top side
USE_THREE_EDGE_DESKEW = (
    config.do_rotate
    and config.rotate_odd == 270
    and config.rotate_even == 90
)

print(f"USE_THREE_EDGE_DESKEW: {USE_THREE_EDGE_DESKEW}")

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def save_dbg(img, path):
    ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, img)

def percentile_threshold(gray):
    high_p = np.percentile(gray, THRESH_HIGH_PERCENTILE)
    thr = max(THRESH_MIN, int(high_p * 0.95))
    _, mask = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
    return mask, thr, int(high_p)

def detect_vertical_streaks(mask, approx_width=3, length_thresh_ratio=0.15):
    h, w = mask.shape
    kx = approx_width
    ky = max(15, int(h * 0.02))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
    long_vertical = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(long_vertical, connectivity=8)
    streak_mask = np.zeros_like(mask)
    length_thresh = max(10, int(h * length_thresh_ratio))
    for i in range(1, num_labels):
        x, y, ww, hh, area = stats[i]
        if hh >= length_thresh and ww <= max(5, int(w * 0.01)):
            streak_mask[labels == i] = 255
    return streak_mask

def keep_largest_component(mask):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    out = np.zeros_like(mask)
    out[labels == best] = 255
    return out

def contour_to_pts(contour):
    return contour.reshape(-1,2)

def fit_line_ransac(pts, iterations=RANSAC_ITER, inlier_dist=RANSAC_INLIER_DIST, min_inliers=RANSAC_MIN_INLIERS):
    if len(pts) < 2:
        raise ValueError("Not enough points")
    best_inliers = None
    best_model = None
    n = len(pts)
    ptsf = pts.astype(np.float32)
    for _ in range(iterations):
        i1, i2 = random.sample(range(n), 2)
        p1 = ptsf[i1]; p2 = ptsf[i2]
        vx = float(p2[0] - p1[0]); vy = float(p2[1] - p1[1])
        if vx == 0 and vy == 0:
            continue
        dists = np.abs(vy*(ptsf[:,0]-p1[0]) - vx*(ptsf[:,1]-p1[1])) / (math.hypot(vx, vy) + 1e-12)
        inliers = dists <= inlier_dist
        cnt = int(inliers.sum())
        if cnt >= min_inliers and (best_inliers is None or cnt > int(best_inliers.sum())):
            best_inliers = inliers.copy()
            best_model = (vx, vy, float(p1[0]), float(p1[1]))
    if best_model is None:
        vx, vy, x0, y0 = cv2.fitLine(ptsf, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        inlier_mask = np.ones(len(pts), dtype=bool)
        return float(vx), float(vy), float(x0), float(y0), inlier_mask
    inlier_pts = ptsf[best_inliers]
    vx, vy, x0, y0 = cv2.fitLine(inlier_pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    inlier_mask = best_inliers
    return float(vx), float(vy), float(x0), float(y0), inlier_mask

def intersect_lines(l1, l2):
    vx1, vy1, x1, y1 = l1
    vx2, vy2, x2, y2 = l2
    A = np.array([[vx1, -vx2], [vy1, -vy2]], dtype=np.float32)
    b = np.array([x2 - x1, y2 - y1], dtype=np.float32)
    det = np.linalg.det(A)
    if abs(det) < 1e-8:
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    t1, t2 = np.linalg.solve(A, b)
    xi = x1 + t1 * vx1
    yi = y1 + t1 * vy1
    return float(xi), float(yi)

def build_affine_without_shear(pt_v_top, pt_v_bot, expected_w, expected_h, bad_on_left, img, dbgdir=None):
    """
    Build a source triangle that enforces orthogonal page axes (no shear) using:
      - pt_v_top: intersection of top line and good vertical (top-right when good vertical is right)
      - pt_v_bot: intersection of bottom line and good vertical (bottom-right when good vertical is right)
    Returns (M_aff, src_corners_dict, dst_corners_dict, reason)
    - M_aff: 2x3 affine matrix or None on failure
    - src_corners_dict: dict of TL, TR, BL, BR (np.float32 2-vectors)
    - dst_corners_dict: dict of TL, TR, BL, BR in destination coordinates
    - reason: diagnostic string
    """
    # convert to vectors
    pt_v_top = np.array(pt_v_top, dtype=np.float32)
    pt_v_bot = np.array(pt_v_bot, dtype=np.float32)

    # compute direction along top edge: vector from bottom intersection to top intersection projected horizontally
    # but we don't have explicit top_line direction vector here; we will approximate using (pt_v_top - pt_v_bot) rotated?
    # Better: caller should supply top_line direction (vx,vy). If you don't have it, approximate from nearby contour points.
    # For compatibility with your code, expect you have top_vx, top_vy; if not, fall back to vector between two top contour points.
    # Here we'll assume top_vx, top_vy are available in outer scope; otherwise compute from pt_v_top->pt_v_bot displacement projected perpendicular:
    # To keep this helper self-contained, we compute top_dir as average of (pt_v_top_to_some_right) - but simplest robust approach:
    # compute approximate top direction by taking small offset vector along top by sampling image gradient: fallback to (1,0).

    # --- Here caller should provide top_unit; we'll compute a safe top_unit using neighbor pixels if available ---
    # We'll attempt to derive a top_unit by sampling a short step along the image: use vector from pt_v_top to pt_v_top projected to image center
    cx, cy = img.shape[1] / 2.0, img.shape[0] / 2.0
    # prefer vector pointing rightwards by projecting displacement to center
    approx = pt_v_top - np.array([cx, cy], dtype=np.float32)
    if np.linalg.norm(approx) < 1e-6:
        approx = np.array([1.0, 0.0], dtype=np.float32)
    ux, uy = approx / (np.linalg.norm(approx) + 1e-12)

    # Force ux positive (pointing to the right) — we want u to be rightward
    if ux < 0:
        ux, uy = -ux, -uy

    # make perpendicular downwards: p = (-uy, ux)
    px, py = -uy, ux

    # normalize again (just in case)
    n_u = math.hypot(ux, uy)
    n_p = math.hypot(px, py)
    if n_u < 1e-6 or n_p < 1e-6:
        return None, None, None, "degenerate_axes"
    ux, uy = ux / n_u, uy / n_u
    px, py = px / n_p, py / n_p

    # Now construct corners. We know pt_v_top/pt_v_bot correspond to the known vertical edge:
    # If bad_on_left is True, good vertical is the RIGHT edge => pt_v_top == TR, pt_v_bot == BR.
    # If bad_on_left is False, good vertical is LEFT edge => pt_v_top == TL, pt_v_bot == BL.
    if bad_on_left:
        src_TR = pt_v_top
        src_BR = pt_v_bot
        # compute TL and BL by moving left along top unit by expected_w
        src_TL = src_TR - np.array([ux, uy], dtype=np.float32) * float(expected_w)
        src_BL = src_BR - np.array([ux, uy], dtype=np.float32) * float(expected_w)
    else:
        src_TL = pt_v_top
        src_BL = pt_v_bot
        # compute TR and BR by moving right along top unit by expected_w
        src_TR = src_TL + np.array([ux, uy], dtype=np.float32) * float(expected_w)
        src_BR = src_BL + np.array([ux, uy], dtype=np.float32) * float(expected_w)

    # Optional: enforce vertical height by projecting (TL->BL) onto perp and rescale to expected_h
    # Compute current vertical vector and its projection onto perp (px,py)
    cur_v = src_BL - src_TL
    proj = cur_v.dot(np.array([px, py], dtype=np.float32))
    # if projection is tiny, fall back but otherwise adjust BL to ensure exact page height
    if abs(proj) > 1e-6:
        # adjust scale to exact expected_h
        scale = float(expected_h) / proj
        # recompute BL, BR as TL + perp*expected_h and TR + perp*expected_h
        src_BL = src_TL + np.array([px, py], dtype=np.float32) * float(expected_h)
        src_BR = src_TR + np.array([px, py], dtype=np.float32) * float(expected_h)
    else:
        # if proj nearly zero, we cannot rely on vertical measurement; still use perpendicular step
        src_BL = src_TL + np.array([px, py], dtype=np.float32) * float(expected_h)
        src_BR = src_TR + np.array([px, py], dtype=np.float32) * float(expected_h)

    # Build src/dst dicts
    src_corners = {
        "TL": src_TL.astype(np.float32),
        "TR": src_TR.astype(np.float32),
        "BL": src_BL.astype(np.float32),
        "BR": src_BR.astype(np.float32)
    }
    dst_corners = {
        "TL": np.array([0.0, 0.0], dtype=np.float32),
        "TR": np.array([expected_w - 1.0, 0.0], dtype=np.float32),
        "BL": np.array([0.0, expected_h - 1.0], dtype=np.float32),
        "BR": np.array([expected_w - 1.0, expected_h - 1.0], dtype=np.float32)
    }

    # choose triangle for affine depending on orientation (map TL, BL, TR -> dst TL, BL, TR)
    src_tri = np.vstack([src_corners["TL"], src_corners["BL"], src_corners["TR"]]).astype(np.float32)
    dst_tri = np.vstack([dst_corners["TL"], dst_corners["BL"], dst_corners["TR"]]).astype(np.float32)

    # validate triangle
    area = abs(0.5 * (src_tri[0,0]*(src_tri[1,1]-src_tri[2,1]) + src_tri[1,0]*(src_tri[2,1]-src_tri[0,1]) + src_tri[2,0]*(src_tri[0,1]-src_tri[1,1])))
    if area < 1.0 or not np.isfinite(src_tri).all():
        return None, src_corners, dst_corners, f"invalid_src_tri_area={area:.3f}"

    try:
        M = cv2.getAffineTransform(src_tri, dst_tri)
    except cv2.error as e:
        return None, src_corners, dst_corners, f"getAffineTransform_failed:{e}"

    # debug overlay: draw corners & axes if dbgdir provided
    if dbgdir is not None:
        vis = img.copy()
        def draw_pt(pt, color=(0,0,255), tag=""):
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 6, color, -1)
            if tag:
                cv2.putText(vis, tag, (int(pt[0]+6), int(pt[1]-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        draw_pt(src_corners["TL"], (0,255,0), "TL")
        draw_pt(src_corners["TR"], (0,128,255), "TR")
        draw_pt(src_corners["BL"], (255,0,0), "BL")
        draw_pt(src_corners["BR"], (255,255,0), "BR")
        # draw axis arrows from TL
        origin = src_corners["TL"]
        arrow_u = origin + np.array([ux, uy], dtype=np.float32) * 80.0
        arrow_p = origin + np.array([px, py], dtype=np.float32) * 80.0
        cv2.arrowedLine(vis, (int(origin[0]), int(origin[1])), (int(arrow_u[0]), int(arrow_u[1])), (255,0,255), 2)
        cv2.arrowedLine(vis, (int(origin[0]), int(origin[1])), (int(arrow_p[0]), int(arrow_p[1])), (0,255,255), 2)
        cv2.imwrite(os.path.join(dbgdir, "debug_axes_and_corners.png"), vis)

    return M, src_corners, dst_corners, "ok"

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

def split_edge_candidates(contour, bad_on_left):
    pts = contour.reshape(-1, 2)

    xs = pts[:,0]
    ys = pts[:,1]

    w = xs.max() - xs.min()
    h = ys.max() - ys.min()

    margin_x = w * 0.15
    margin_y = h * 0.15

    # outside edge
    if bad_on_left:
        outside = pts[xs > xs.max() - margin_x]
    else:
        outside = pts[xs < xs.min() + margin_x]

    # top edge
    top = pts[ys < ys.min() + margin_y]

    # bottom edge
    bottom = pts[ys > ys.max() - margin_y]

    return top, bottom, outside

def line_angle(line):
    vx, vy, _, _ = line
    return math.atan2(vy, vx)

def normalize_angle_deg(a):
    while a < -90:
        a += 180
    while a > 90:
        a -= 180
    return a

def get_gray_mask_contours(img, dbgdir):
    # --- threshold for geometry only ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask_init, thr, hp = percentile_threshold(gray)
    if cv2.mean(gray)[0] < 127:
        mask_init = cv2.bitwise_not(mask_init)
    if DEBUG:
        save_dbg(gray, os.path.join(dbgdir, "01_gray.png"))
        save_dbg(mask_init, os.path.join(dbgdir, f"02_thresh_thr{thr}_hp{hp}.png"))

    # without a9dc2b24e6b0d1d49b6fc232223d6431ba3442a5 bad: fix perspective transform for broken ADF scanners
    # Invert if necessary, so the page is always "white" for thresholding
    # We'll use Otsu's method to detect high-contrast area
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Determine whether the page is darker or lighter than background
    mean_val = cv2.mean(gray, mask=None)[0]
    if mean_val < 127:  # dark page: invert mask
        mask = cv2.bitwise_not(mask)

    # Morphology to remove small gaps
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return gray, mask, contours

def repair_binding(img, bad_on_left, width=50):

    h,w = img.shape[:2]

    if bad_on_left:
        sample_range = range(width)
        fill_range = range(width)
        sample_x = width
    else:
        sample_range = range(w-width,w)
        fill_range = range(w-width,w)
        sample_x = w-width

    for y in range(h):

        if bad_on_left:
            sample = img[y, width:width+20]
            color = np.mean(sample,axis=0)
            img[y,:width] = color

        else:
            sample = img[y,w-width-20:w-width]
            color = np.mean(sample,axis=0)
            img[y,w-width:] = color

    return img

def process_image(in_path, out_path):
    """
    Robust page extraction that handles missing top (or bottom) edges.
    - preserves original colors (warps the original image)
    - builds orthogonal axes (no shear)
    - if top is missing, uses bottom + good vertical to infer top by moving up expected_h
    - fills outside the filled page polygon with pure white
    - extensive debug output in OUTPUT_DIR/debug/<page_num>/
    """
    # ---------- small helpers ----------
    def ensure_dir(p):
        os.makedirs(p, exist_ok=True)

    def save_dbg(img, path):
        ensure_dir(os.path.dirname(path))
        cv2.imwrite(path, img)

    def percentile_threshold(gray):
        high_p = np.percentile(gray, THRESH_HIGH_PERCENTILE)
        thr = max(THRESH_MIN, int(high_p * 0.95))
        _, mask = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
        return mask, thr, int(high_p)

    def keep_largest_component(mask):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels <= 1:
            return mask
        areas = stats[1:, cv2.CC_STAT_AREA]
        best = 1 + int(np.argmax(areas))
        out = np.zeros_like(mask)
        out[labels == best] = 255
        return out

    def detect_vertical_streaks(mask, approx_width=3, length_thresh_ratio=0.15):
        h, w = mask.shape
        kx = approx_width
        ky = max(15, int(h * 0.02))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
        long_vertical = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(long_vertical, connectivity=8)
        streak_mask = np.zeros_like(mask)
        length_thresh = max(10, int(h * length_thresh_ratio))
        for i in range(1, num_labels):
            x, y, ww, hh, area = stats[i]
            if hh >= length_thresh and ww <= max(5, int(w * 0.01)):
                streak_mask[labels == i] = 255
        return streak_mask

    def contour_to_pts(c):
        return c.reshape(-1, 2)

    def fit_line_ransac(pts, iterations=RANSAC_ITER, inlier_dist=RANSAC_INLIER_DIST, min_inliers=RANSAC_MIN_INLIERS):
        if len(pts) < 2:
            raise ValueError("Not enough points for line fit")
        best_inliers = None
        best_cnt = 0
        best_model = None
        ptsf = pts.astype(np.float32)
        n = len(ptsf)
        for _ in range(iterations):
            i1, i2 = random.sample(range(n), 2)
            p1 = ptsf[i1]; p2 = ptsf[i2]
            vx = float(p2[0] - p1[0]); vy = float(p2[1] - p1[1])
            if abs(vx) < 1e-6 and abs(vy) < 1e-6:
                continue
            dists = np.abs(vy*(ptsf[:,0]-p1[0]) - vx*(ptsf[:,1]-p1[1])) / (math.hypot(vx, vy) + 1e-12)
            inliers = dists <= inlier_dist
            cnt = int(inliers.sum())
            if cnt >= min_inliers and cnt > best_cnt:
                best_cnt = cnt
                best_inliers = inliers.copy()
                best_model = (vx, vy, float(p1[0]), float(p1[1]))
        if best_model is None:
            vx, vy, x0, y0 = cv2.fitLine(ptsf, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
            inlier_mask = np.ones(len(ptsf), dtype=bool)
            return float(vx), float(vy), float(x0), float(y0), inlier_mask
        inlier_pts = ptsf[best_inliers]
        vx, vy, x0, y0 = cv2.fitLine(inlier_pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        return float(vx), float(vy), float(x0), float(y0), best_inliers

    def intersect_lines(l1, l2):
        vx1, vy1, x1, y1 = l1
        vx2, vy2, x2, y2 = l2
        A = np.array([[vx1, -vx2], [vy1, -vy2]], dtype=np.float32)
        b = np.array([x2 - x1, y2 - y1], dtype=np.float32)
        det = np.linalg.det(A)
        if abs(det) < 1e-8:
            return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        t1, t2 = np.linalg.solve(A, b)
        xi = x1 + t1 * vx1
        yi = y1 + t1 * vy1
        return float(xi), float(yi)

    # ---------- start ----------
    fname = os.path.basename(in_path)
    m = re.match(r"^(\d+)", fname)
    page_num = int(m.group(1)) if m else 0
    bad_on_left = (page_num % 2 == 1)

    img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print("Failed to read", in_path); return
    input_is_grayscale = (len(img.shape) == 2)
    if len(img.shape) == 2:
        # OpenCV expects 3-channel images for many operations
        # later convert back to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    H_img, W_img = img.shape[:2]

    # FIXME wrong H_img?
    expected_w = int(round(ASPECT * H_img))
    expected_h = H_img

    dbgdir = os.path.join(OUTPUT_DIR, "debug", f"{page_num:03d}")
    if DEBUG: ensure_dir(dbgdir)

    gray, mask, contours = get_gray_mask_contours(img, dbgdir)

    if not contours:
        print(f"Warning: no contours found in {in_path}")
        return

    page_contour = max(contours, key=cv2.contourArea)

    if USE_THREE_EDGE_DESKEW:

        top_pts, bottom_pts, outside_pts = split_edge_candidates(
            page_contour,
            bad_on_left
        )

        top_line = fit_line_ransac(top_pts)[:4]
        bottom_line = fit_line_ransac(bottom_pts)[:4]
        outside_line = fit_line_ransac(outside_pts)[:4]

        top_angle = math.degrees(line_angle(top_line))
        bottom_angle = math.degrees(line_angle(bottom_line))
        outside_angle = math.degrees(line_angle(outside_line))

        rotation_error = (
            -top_angle
            -bottom_angle
            +(90 - outside_angle)
        ) / 3.0

        Mrot = cv2.getRotationMatrix2D(
            (W_img/2, H_img/2),
            rotation_error,
            1.0
        )

        rotated = cv2.warpAffine(
            img,
            Mrot,
            (W_img, H_img),
            borderValue=(255,255,255)
        )

        # img = rotated # ?

        # gray, mask, contours = get_gray_mask_contours(img, dbgdir)
        gray, mask, contours = get_gray_mask_contours(rotated, dbgdir)

        if not contours:
            print(f"Warning: no contours found in {in_path}")
            return

        page_contour = max(contours, key=cv2.contourArea)

        _, _, outside_pts = split_edge_candidates(
            page_contour,
            bad_on_left
        )

        outside_line = fit_line_ransac(outside_pts)[:4]

        # FIXME H_img is wrong
        expected_h = H_img
        expected_w = int(round(ASPECT * expected_h))

        vx, vy, x0, y0 = outside_line

        if bad_on_left:
            x_out = x0
            x0_page = x_out
            x1_page = x_out + expected_w
        else:
            x_out = x0
            x1_page = x_out
            x0_page = x_out - expected_w

        crop = rotated[
            0:expected_h,
            int(x0_page):int(x1_page)
        ]

        # img = repair_binding(img, bad_on_left, width=50)
        rotated = repair_binding(rotated, bad_on_left, width=50)

        warped = rotated

    else:
        # USE_THREE_EDGE_DESKEW == False
        # FIXME use only two page edges: outside, bottom
        # Approximate contour to quadrilateral
        epsilon = 0.02 * cv2.arcLength(page_contour, True)
        approx = cv2.approxPolyDP(page_contour, epsilon, True)
        if len(approx) != 4:
            approx = cv2.convexHull(page_contour)
            if len(approx) < 4:
                print(f"Warning: not enough points for perspective in {in_path}")
                return
            # pick 4 extreme points
            pts = np.array([
                approx[approx[:,0,0].argmin()][0],  # leftmost
                approx[approx[:,0,1].argmin()][0],  # topmost
                approx[approx[:,0,0].argmax()][0],  # rightmost
                approx[approx[:,0,1].argmax()][0]   # bottommost
            ])
        else:
            pts = approx.reshape(4,2)

        rect = order_points(pts)

        # Perspective transform
        widthA = np.linalg.norm(rect[2] - rect[3])
        widthB = np.linalg.norm(rect[1] - rect[0])
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.linalg.norm(rect[1] - rect[2])
        heightB = np.linalg.norm(rect[0] - rect[3])
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    # # Add internal white border
    # h, w = warped.shape[:2]
    # canvas = np.ones_like(warped) * 255
    # b = BORDER_SIZE
    # canvas[b:h-b, b:w-b] = warped[b:h-b, b:w-b]

    h, w = warped.shape[:2]
    b = BORDER_SIZE

    # Prepare a canvas with the same content as warped
    canvas = warped.copy()

    # Function to compute average color along a strip
    def avg_color_strip(img, axis, start, end, strip_width=1):
        if axis == 'top':
            strip = img[start:end, :, :]
        elif axis == 'bottom':
            strip = img[h-end:h-start, :, :]
        elif axis == 'left':
            strip = img[:, start:end, :]
        elif axis == 'right':
            strip = img[:, w-end:w-start, :]
        else:
            raise ValueError("Invalid axis")
        return np.mean(strip, axis=(0,1)).astype(np.uint8)

    # Fill borders with local average color
    canvas[0:b, :, :] = avg_color_strip(canvas, 'top', 50, 100)       # top border
    canvas[h-b:h, :, :] = avg_color_strip(canvas, 'bottom', 50, 100)  # bottom border
    canvas[:, 0:b, :] = avg_color_strip(canvas, 'left', 50, 100)      # left border
    canvas[:, w-b:w, :] = avg_color_strip(canvas, 'right', 50, 100)   # right border

    if input_is_grayscale:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

    ensure_dir(os.path.dirname(out_path))
    cv2.imwrite(out_path, canvas)
    print(f"writing {out_path}")

def main():
    ensure_dir(OUTPUT_DIR)
    # TODO use image_format from 030-measure-page-size.txt
    image_format = "jpg"
    files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(f".{image_format}")])
    if not files:
        print("No image files found in", INPUT_DIR)
        return
    for fname in files:
        m = re.match(r"^(\d+)", fname)
        # page_num = int(m.group(1)) if m else 0
        page_num = int(m.group(1)) # can throw
        # if page_num != 14: continue # debug
        # if not page_num in [340, 345]: continue # debug
        # if page_num < 300: continue # debug
        # if page_num != 320: continue # debug
        in_path = os.path.join(INPUT_DIR, fname)
        out_path = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(out_path):
            continue
        try:
            process_image(in_path, out_path)
        except Exception as exc:
            print(f"Error processing {fname}: {exc}")
            raise

if __name__ == "__main__":
    main()
```

but the script fails on both odd and even pages

all input images have a mid-gray (50% gray) margin on 3 sides: top bottom outside

on odd pages, the output images have a light-gray margin on the top and bottom side

on even pages, all pages are rotated by about 45 degrees (CCW), which is totally wrong, because the rotation error is somewhere between -5 and +5 degrees

ideas?

## Response:

7/12/2026, 2:08:08 PM

Yes. I see several problems. The good news is: the overall approach is right, but the implementation has a few geometry mistakes that explain exactly the symptoms you see.

Let's go through them.

------------------------------------------------------------------------

# Problem 1: `line_angle()` is not normalized

This is the biggest cause of the **45° rotation on even pages**.

Your function:

``` python
def line_angle(line):
    vx, vy, _, _ = line
    return math.atan2(vy, vx)
```

returns the direction of the RANSAC vector.

But a line has **no direction**.

Example:

    outside edge:

       |
       |
       |

    vector A:
    (0, +1)

    angle = +90°

    same line, vector B:

    (0, -1)

    angle = -90°

For odd pages you get lucky.

For even pages your outside line probably flips direction, producing something like:

    outside_angle = -89°

Then:

``` python
90 - outside_angle
```

becomes:

    90 - (-89)

    =179°

and your rotation becomes approximately:

    179 / 3 = 60°

which explains the huge rotation.

------------------------------------------------------------------------

## Fix

Replace:

``` python
def line_angle(line):
    vx, vy, _, _ = line
    return math.degrees(math.atan2(vy, vx))
```

with:

``` python
def line_angle(line):
    vx, vy, _, _ = line

    angle = math.degrees(math.atan2(vy, vx))

    # lines have no direction
    while angle <= -90:
        angle += 180

    while angle > 90:
        angle -= 180

    return angle
```

Now:

    +89°
    -91°

both become:

    89°

------------------------------------------------------------------------

# Problem 2: your rotation formula is wrong

This:

``` python
rotation_error = (
    -top_angle
    -bottom_angle
    +(90 - outside_angle)
) / 3.0
```

is not mathematically correct.

The outside edge is perpendicular to the horizontal edges.

You don't want:

    top correction
    +
    bottom correction
    +
    outside correction

because the outside correction is already the same rotation expressed differently.

Use:

``` python
rotation_error = (
    -top_angle
    -bottom_angle
) / 2
```

and use the outside edge only as validation.

Then:

``` python
outside_after = outside_angle + rotation_error

if abs(abs(outside_after)-90) > 2:
    print("warning: outside edge disagreement")
```

The horizontal edges are your primary deskew signal.

------------------------------------------------------------------------

# Problem 3: your crop is calculated but never used

You have:

``` python
crop = rotated[
    0:expected_h,
    int(x0_page):int(x1_page)
]
```

but later:

``` python
warped = rotated
```

So you throw away the crop.

It should be:

``` python
warped = crop
```

This explains:

> odd pages have light-gray margin top and bottom

because you are still saving the whole rotated scanner image, including scanner background.

------------------------------------------------------------------------

# Problem 4: your repair is applied before cropping

Currently:

``` python
rotated = repair_binding(rotated, bad_on_left, width=50)

warped = rotated
```

The repair should happen **after** cropping.

Correct:

``` python
crop = rotated[
    0:expected_h,
    int(x0_page):int(x1_page)
]

crop = repair_binding(
    crop,
    bad_on_left,
    width=50
)

warped = crop
```

Otherwise you are repairing the scanner margin, not the missing binding edge.

------------------------------------------------------------------------

# Problem 5: `expected_w` is probably wrong after rotation

You wrote:

``` python
expected_w = int(round(ASPECT * H_img))
```

but your rotated image dimensions are not guaranteed to have the same orientation.

Add:

``` python
Hr, Wr = rotated.shape[:2]
```

then:

``` python
expected_h = Hr
expected_w = int(round(ASPECT * expected_h))
```

------------------------------------------------------------------------

# Problem 6: your edge detector assumes outside is left/right

This part:

``` python
if bad_on_left:
    outside = pts[xs > xs.max() - margin_x]
else:
    outside = pts[xs < xs.min() + margin_x]
```

is only correct if the image is already in reading orientation.

You said:

    scanner top side = binding
    rotate_odd=270
    rotate_even=90

After your preprocessing step (`060-rotate-crop`) has already rotated the images, this is probably OK.

But verify.

Add:

``` python
print(
    page_num,
    "bad_on_left",
    bad_on_left,
    "bbox",
    xs.min(),
    xs.max()
)
```

and inspect.

------------------------------------------------------------------------

# Minimal patch

I would change only these pieces first.

### Replace:

``` python
def line_angle(line):
    vx, vy, _, _ = line
    return math.atan2(vy, vx)
```

with:

``` python
def line_angle(line):
    vx, vy, _, _ = line
    a = math.degrees(math.atan2(vy, vx))

    while a <= -90:
        a += 180

    while a > 90:
        a -= 180

    return a
```

------------------------------------------------------------------------

### Replace:

``` python
rotation_error = (
    -top_angle
    -bottom_angle
    +(90 - outside_angle)
) / 3.0
```

with:

``` python
rotation_error = (
    -top_angle
    -bottom_angle
) / 2.0
```

------------------------------------------------------------------------

### Replace:

``` python
crop = rotated[
    0:expected_h,
    int(x0_page):int(x1_page)
]

rotated = repair_binding(rotated, bad_on_left, width=50)

warped = rotated
```

with:

``` python
crop = rotated[
    0:expected_h,
    int(x0_page):int(x1_page)
]

crop = repair_binding(
    crop,
    bad_on_left,
    width=50
)

warped = crop
```

------------------------------------------------------------------------

I expect these three fixes alone will remove both observed failures:

- the ~45° CCW even-page rotation → fixed by angle normalization
- the gray scanner margins → fixed by actually using `crop`

After that, the next thing I would inspect is whether the crop is shifted by a few pixels, but the core deskew logic should start working.

## Prompt:

7/12/2026, 2:31:29 PM

only your rotation_error fix made it somewhat "better"...  
but now all pages have a slightly wrong rotation

with your line_angle fix, all pages end up with a rotation error around 45 degrees

with your repair_binding fix, all pages end up with a really small width (maybe 5% of the expected width)

## Response:

7/12/2026, 2:31:29 PM

Good, these symptoms are actually very informative. They tell us that my previous diagnosis was only partially correct. The **rotation_error formula was indeed wrong**, but the deeper issue is that we are feeding it unreliable angles.

Let's debug the geometry instead of patching symptoms.

------------------------------------------------------------------------

## 1. The 45° rotation after my `line_angle()` change

That means your `outside_angle` is not the problem.

The problem is this:

``` python
top_angle = math.degrees(line_angle(top_line))
bottom_angle = math.degrees(line_angle(bottom_line))
```

For page edges, `cv2.fitLine()` can return vectors pointing in arbitrary directions.

Even after normalizing to `[-90,90]`, the **horizontal edges can become vertical-ish**.

Example:

Actual top edge:

    -----------------

RANSAC returns:

    <-----------------

or:

    ----------------->

Both are okay.

But for a nearly horizontal line, you can get:

    vx = 0.999
    vy = 0.02

or:

    vx = -0.999
    vy = -0.02

Those are equivalent.

The correct normalization is not `[-90,90]`.

For your purpose you need:

- top/bottom edges → force them into `[-45,+45]`
- outside edge → force it near `90°`

Try this instead:

``` python
def horizontal_line_angle(line):
    vx, vy, _, _ = line

    a = math.degrees(math.atan2(vy, vx))

    while a < -45:
        a += 180

    while a > 45:
        a -= 180

    return a

def vertical_line_angle(line):
    vx, vy, _, _ = line

    a = math.degrees(math.atan2(vy, vx))

    while a < 45:
        a += 180

    while a > 135:
        a -= 180

    return a
```

Then:

``` python
top_angle = horizontal_line_angle(top_line)
bottom_angle = horizontal_line_angle(bottom_line)
outside_angle = vertical_line_angle(outside_line)
```

Now print:

``` python
print(
    page_num,
    "top",
    top_angle,
    "bottom",
    bottom_angle,
    "outside",
    outside_angle
)
```

Before applying rotation.

I expect you will see the issue immediately.

------------------------------------------------------------------------

## 2. The crop becoming 5% width

This is caused by my repair change, but the root cause is different.

The problem is here:

``` python
if bad_on_left:
    x_out = x0
    x0_page = x_out
    x1_page = x_out + expected_w
```

You assume:

``` python
x0
```

is the actual x-coordinate of the vertical edge.

It isn't.

For a line represented as:

    vx, vy, x0, y0

`x0,y0` is only **one point on the line**.

For a vertical line, it might be near the middle:

          |
          |
      x0 ->*
          |
          |

You cannot use:

``` python
x_out = x0
```

as the edge position.

You need to intersect the outside line with the bottom line.

You already have:

``` python
intersect_lines()
```

Use it.

After rotation:

``` python
outside_top = intersect_lines(
    outside_line,
    top_line
)

outside_bottom = intersect_lines(
    outside_line,
    bottom_line
)
```

Then:

``` python
x_out = (
    outside_top[0]
    +
    outside_bottom[0]
) / 2
```

Now you have the real outside edge x coordinate.

------------------------------------------------------------------------

## 3. The correct crop code

Replace:

``` python
vx, vy, x0, y0 = outside_line

if bad_on_left:
    x_out = x0
    x0_page = x_out
    x1_page = x_out + expected_w
else:
    x_out = x0
    x1_page = x_out
    x0_page = x_out - expected_w
```

with:

``` python
outside_top = intersect_lines(
    outside_line,
    top_line
)

outside_bottom = intersect_lines(
    outside_line,
    bottom_line
)

x_out = (
    outside_top[0]
    +
    outside_bottom[0]
) / 2

if bad_on_left:
    x0_page = int(round(x_out))
    x1_page = int(round(x_out + expected_w))

else:
    x0_page = int(round(x_out - expected_w))
    x1_page = int(round(x_out))
```

------------------------------------------------------------------------

## 4. One more issue: don't crop before repair

Your original intuition was right:

> restore binding edge after deskew

The correct order is:

    raw scan

     ↓

    deskew rotation

     ↓

    crop page

     ↓

    repair missing binding edge

     ↓

    save

not:

    deskew

     ↓

    repair scanner image

     ↓

    crop

------------------------------------------------------------------------

## What I would do next

Add this debug print right after fitting the lines:

``` python
print(
    f"{page_num}: "
    f"top={top_angle:.2f} "
    f"bottom={bottom_angle:.2f} "
    f"outside={outside_angle:.2f} "
    f"rotation={rotation_error:.2f}"
)
```

Run on:

- one odd page
- one even page

The expected output should look like:

    101: top=1.2 bottom=-0.8 outside=89.4 rotation=-0.2
    102: top=-2.0 bottom=0.5 outside=91.1 rotation=0.75

If you see values like:

    top=44
    bottom=-46
    outside=90

then the edge classification is wrong, not the deskew.

At this point I would not touch the repair code again until those three line angles are sane. The geometry has to be correct first.

## Prompt:

7/12/2026, 2:39:50 PM

the cropping is still totally wrong (output width about 5% of expected width, everything is black)

the rotation angles are still wrong, almost like it was rotated in the wrong direction

debug prints

    1 top -0.41112419570351727 bottom -0.38916038416581633 outside 89.76906014397322
    2 top 0.3824096634132325 bottom 0.38732149216507755 outside 90.26309637155515

## Response:

7/12/2026, 2:39:50 PM

These debug values are actually **excellent news**:

    1 top -0.411 bottom -0.389 outside 89.769
    2 top  0.382 bottom  0.387 outside 90.263

Your **edge detection is working**. The three detected edges are already almost perfectly aligned.

That means the problem is **not RANSAC, not line angles, not edge classification**.

The remaining problems are downstream.

------------------------------------------------------------------------

## Problem 1: your rotation direction is backwards

OpenCV's `getRotationMatrix2D()` uses a positive angle for counter-clockwise rotation.

Your pages are already:

    top    ≈ 0°
    bottom ≈ 0°

So your correction should be tiny.

For page 1:

    top=-0.411
    bottom=-0.389

average:

    -0.400°

The correction is:

``` python
rotation_error = -average
```

therefore:

``` python
rotation_error = +0.400°
```

For page 2:

    average = +0.384°

therefore:

``` python
rotation_error = -0.384°
```

So replace:

``` python
rotation_error = (
    -top_angle
    -bottom_angle
) / 2.0
```

with:

``` python
rotation_error = -(top_angle + bottom_angle) / 2.0
```

The parentheses matter.

------------------------------------------------------------------------

## Problem 2: you are using the wrong image size after rotation

This is the reason your crop becomes tiny/black.

You have:

``` python
Mrot = cv2.getRotationMatrix2D(
    (W_img/2, H_img/2),
    rotation_error,
    1.0
)

rotated = cv2.warpAffine(
    img,
    Mrot,
    (W_img, H_img),
)
```

The output image keeps the **original scanner dimensions**.

But after rotation, the page content moves slightly.

You then do:

``` python
x0_page
x1_page
```

and crop.

If those coordinates are outside the image, OpenCV returns a tiny black strip.

------------------------------------------------------------------------

Add this immediately after rotation:

``` python
Hr, Wr = rotated.shape[:2]

print(
    "rotated size",
    Wr,
    Hr
)
```

and before crop:

``` python
print(
    "crop",
    x0_page,
    x1_page,
    "expected",
    expected_w
)
```

I strongly suspect you will see something like:

    crop -200 120
    expected 2500

------------------------------------------------------------------------

## Problem 3: you lost `top_line` and `bottom_line` after re-detection

This is a subtle bug.

After rotation you do:

``` python
gray, mask, contours = get_gray_mask_contours(rotated, dbgdir)

page_contour = max(contours, key=cv2.contourArea)

_, _, outside_pts = split_edge_candidates(
    page_contour,
    bad_on_left
)

outside_line = fit_line_ransac(outside_pts)[:4]
```

You only recompute:

    outside_line

but later you need:

``` python
outside_line + top_line
```

for:

``` python
intersect_lines()
```

After rotation, recompute all three:

``` python
top_pts, bottom_pts, outside_pts = split_edge_candidates(
    page_contour,
    bad_on_left
)

top_line = fit_line_ransac(top_pts)[:4]
bottom_line = fit_line_ransac(bottom_pts)[:4]
outside_line = fit_line_ransac(outside_pts)[:4]
```

------------------------------------------------------------------------

## Problem 4: your outside coordinate calculation

This:

``` python
outside_top = intersect_lines(
    outside_line,
    top_line
)

outside_bottom = intersect_lines(
    outside_line,
    bottom_line
)
```

is correct.

But before cropping, add:

``` python
print(
    "outside top",
    outside_top,
    "outside bottom",
    outside_bottom
)
```

Expected:

something like:

    outside top (3300, 20)
    outside bottom (3300, 4900)

If you see:

    outside top (50, 2500)
    outside bottom (50, 2500)

then your outside line is wrong.

------------------------------------------------------------------------

# One more correction: don't use x0/x1 blindly

After computing:

``` python
x_out = (
    outside_top[0]
    +
    outside_bottom[0]
) / 2
```

clamp:

``` python
x_out = int(round(x_out))

if bad_on_left:
    x0_page = x_out
    x1_page = x_out + expected_w
else:
    x0_page = x_out - expected_w
    x1_page = x_out
```

then:

``` python
x0_page = max(0, x0_page)
x1_page = min(rotated.shape[1], x1_page)
```

for debugging.

------------------------------------------------------------------------

## Current state summary

Your pipeline status:

| Part                | Status                                     |
|---------------------|--------------------------------------------|
| Detect top edge     | ✅                                         |
| Detect bottom edge  | ✅                                         |
| Detect outside edge | ✅                                         |
| Determine page side | probably ✅                                |
| Deskew angle        | small sign error                           |
| Crop position       | broken because outside coordinate handling |
| Repair              | not reached yet                            |

The important discovery is that your hardest part is solved. The remaining bugs are just coordinate bookkeeping. I would add the four debug prints above (`rotated size`, `crop`, `outside top`, `outside bottom`) and the cause should become obvious.

## Prompt:

7/12/2026, 2:48:43 PM

yes, flipping the sign of rotation_error did work

with all your fixes, the output images still have a lightgray strip on the top side...

debug prints

    1 top -0.41112419570351727 bottom -0.38916038416581633 outside 89.76783071904023
    line 580: rotated size 3478 5078
    line 640: outside top (3454.716552734375, 69.978271484375) outside bottom (3440.2587890625, 5030.9912109375)
    line 650: crop 3447 3478 expected 6836
    line 660: rotated size 3478 5078
    writing 065-remove-page-borders/001.jpg

    2 top 0.3824096634132325 bottom 0.38732149216507755 outside 90.26436680474167
    line 580: rotated size 3470 5078
    line 640: outside top (31.2772216796875, 77.547119140625) outside bottom (40.665740966796875, 4485.56884765625)
    line 650: crop 0 36 expected 6836
    line 660: rotated size 3470 5078
    writing 065-remove-page-borders/002.jpg

## Response:

7/12/2026, 2:48:43 PM

These debug prints reveal the real problem, and it's a fundamental one:

    rotated size      3478 × 5078
    expected width    6836

Your expected page width is **almost twice the image width**.

Everything that follows is doomed:

    page width = 6836
    image width = 3478

so

    crop = 3447 ... 3447+6836

gets clipped to

    3447 ... 3478

which is exactly the ~30 pixel strip you're seeing.

So the crop code is actually behaving correctly. The bug is earlier.

------------------------------------------------------------------------

# Why is expected_w = 6836?

This line:

``` python
expected_w = int(round(ASPECT * expected_h))
```

with

    expected_h = 5078

gives

    expected_w = 6836

Therefore

    ASPECT = 6836 / 5078 ≈ 1.346

A page is wider than it is tall?!

That's impossible for a normal book.

------------------------------------------------------------------------

## I think your aspect ratio is backwards.

Earlier you have

``` python
scan_x = config.scan_x
scan_y = config.scan_y

config.scan_aspect = scan_x / scan_y
```

Suppose your page is

    148 × 210 mm

Then

    scan_x = 148
    scan_y = 210

giving

    ASPECT = 148/210 = 0.705

which would produce

    5078 * 0.705 = 3579 px

which is almost exactly your scanner width of 3478 px.

Instead you're getting

    6836 / 5078 = 1.346

which is

    210 / 156

approximately the inverse.

------------------------------------------------------------------------

# I would print these immediately

``` python
print(scan_x, scan_y)
print(ASPECT)
```

I strongly suspect you'll see something like

    210
    156

    ASPECT = 1.346

------------------------------------------------------------------------

# The light gray strip at the top

That is a different issue.

Remember:

Your scanner removes the binding edge.

After rotation,

the binding edge becomes the **top** edge.

Therefore a gray strip on the top is expected unless you crop it away.

Your crop currently is

``` python
crop = rotated[
    0:expected_h,
    ...
]
```

Notice:

    0

is always the scanner edge.

You're never moving the top boundary down to the detected top edge.

------------------------------------------------------------------------

You should also detect

``` python
top_line
```

after rotation and compute

``` python
y_top
```

exactly like you computed

    x_out

Then

``` python
crop = rotated[
    y_top:y_top+expected_h,
    x0_page:x1_page
]
```

instead of

``` python
0:expected_h
```

------------------------------------------------------------------------

# I think the crop should actually be

After the second contour detection:

    top_line
    bottom_line
    outside_line

Compute

    outside_top = intersect(top,outside)
    outside_bottom = intersect(bottom,outside)

Now

    x_out = average x

    y_top = outside_top.y

    y_bottom = outside_bottom.y

Notice you already have **both** vertical coordinates.

Use them.

------------------------------------------------------------------------

# Biggest issue remaining

The most important thing to fix first is **ASPECT**.

Everything else depends on it.

Because right now your code is trying to crop a page that is literally wider than the scanner bed.

That's why every crop collapses to a tiny strip.

Once `expected_w` becomes ~3500 instead of ~6800, I think 90% of the remaining problems will disappear.

## Prompt:

7/12/2026, 2:51:58 PM

> ## I think your aspect ratio is backwards.

yes, the problem is that config.scan_x and config.scan_y are the raw scan sizes, but my 060-rotate-crop.py script applies config.do_rotate, config.rotate_odd, config.rotate_even... so when rotate_odd=270 then scan_x becomes scan_y and vice versa....

## Response:

7/12/2026, 2:52:00 PM

Exactly. That explains everything.

My earlier conclusion that the aspect ratio was "backwards" was too simplistic. The ratio itself isn't wrong—it's being interpreted in the wrong coordinate system.

Your pipeline is:

``` text
scanner
    ↓
raw scan
    ↓
060-rotate-crop.py   <-- rotates 90°/270°
    ↓
065-remove-page-borders.py
```

So by the time `065-remove-page-borders.py` runs, the image dimensions have already changed, but you're still using the **raw scanner dimensions** from the config.

For example, suppose the scanner captures:

``` text
raw scanner:
scan_x = 215 mm
scan_y = 297 mm
```

After a 90° rotation, the image is effectively:

``` text
image:
width  = 297 mm
height = 215 mm
```

but you're still computing

``` python
ASPECT = scan_x / scan_y
```

instead of

``` python
ASPECT = scan_y / scan_x
```

for rotated scans.

------------------------------------------------------------------------

## I would compute the effective page geometry once

Instead of modifying `scan_x`/`scan_y` throughout the code, compute the effective dimensions immediately after loading the config:

``` python
scan_x = config.scan_x
scan_y = config.scan_y

if (
    config.do_rotate
    and config.rotate_odd == 270
    and config.rotate_even == 90
):
    scan_x, scan_y = scan_y, scan_x

ASPECT = scan_x / scan_y
```

or even clearer:

``` python
effective_scan_x = config.scan_x
effective_scan_y = config.scan_y

if USE_THREE_EDGE_DESKEW:
    effective_scan_x, effective_scan_y = (
        effective_scan_y,
        effective_scan_x,
    )

ASPECT = effective_scan_x / effective_scan_y
```

Now the rest of the code doesn't need to know whether the images were rotated earlier.

------------------------------------------------------------------------

## Even better: avoid `ASPECT` entirely

Since you're already using physical dimensions, I'd avoid carrying around a ratio if possible.

Instead:

``` python
page_width_px = (
    effective_scan_x
    * rotated.shape[0]
    / effective_scan_y
)
```

or

``` python
expected_w = round(
    rotated.shape[0]
    * effective_scan_x
    / effective_scan_y
)
```

That makes it obvious that you're scaling from the page height in pixels using the physical page aspect ratio, and it avoids having to remember whether `ASPECT` was computed before or after a rotation.

I think this will fix the "5% width" crop immediately, because you'll end up with an expected width close to the rotated image width instead of nearly double it.

## Prompt:

7/12/2026, 3:05:14 PM

> compute
>
> Python
>
> Run
>
>     y_top
>
> exactly like you computed
>
>     x_out

how can i compute y_top

## Response:

7/12/2026, 3:05:14 PM

You compute it exactly the same way as `x_out`, except you use the **intersection of the top edge and the outside edge**.

After re-detecting the lines on the **rotated** image:

``` python
top_pts, bottom_pts, outside_pts = split_edge_candidates(
    page_contour,
    bad_on_left
)

top_line = fit_line_ransac(top_pts)[:4]
bottom_line = fit_line_ransac(bottom_pts)[:4]
outside_line = fit_line_ransac(outside_pts)[:4]
```

compute the intersections:

``` python
outside_top = intersect_lines(outside_line, top_line)
outside_bottom = intersect_lines(outside_line, bottom_line)
```

These are approximately:

``` text
outside_top    = (x_out, y_top)
outside_bottom = (x_out, y_bottom)
```

So simply:

``` python
x_out = (outside_top[0] + outside_bottom[0]) / 2.0

y_top = outside_top[1]
y_bottom = outside_bottom[1]
```

or, if you want to reduce noise slightly,

``` python
y_top = round(outside_top[1])
y_bottom = round(outside_bottom[1])

actual_height = y_bottom - y_top
```

Then your crop becomes

``` python
crop = rotated[
    y_top:y_top + expected_h,
    x0_page:x1_page
]
```

instead of

``` python
crop = rotated[
    0:expected_h,
    x0_page:x1_page
]
```

------------------------------------------------------------------------

### Even better

Since you detect **both** top and bottom, I actually wouldn't use `expected_h` at all.

Use the measured page height:

``` python
y_top = int(round(outside_top[1]))
y_bottom = int(round(outside_bottom[1]))

crop = rotated[
    y_top:y_bottom,
    x0_page:x1_page
]
```

This has two advantages:

- it removes the gray scanner strip at the top automatically,
- it tolerates scanners that crop a few pixels from the bottom.

The measured height should be very close to `expected_h` anyway (within a few pixels). If you later want to enforce an exact output size, you can simply resize the cropped image to your target dimensions after the crop, rather than forcing the crop coordinates to match the expected height exactly. I think that's the more robust approach for scanner-derived images.

## Prompt:

7/12/2026, 3:13:16 PM

> After rotation,
>
> the binding edge becomes the **top** edge.

wrong.  
in the raw scan, the binding edge is the top edge.  
After rotation, the binding edge becomes the inside edge:  
left edge of odd pages, right edge of even pages.

i have tried to apply your fixes, but nothing helps....

debug prints

    1 top -0.41112419570351727 bottom -0.38916038416581633 outside 89.76783071904023
    line 580: rotated size 3478 5078
    line 640: outside top (3454.768310546875, 69.978271484375) outside bottom (3440.1533203125, 5030.9912109375)
    line 650: crop 3447 3478 expected 3772
    line 660: rotated size 3478 5078
    writing 065-remove-page-borders/001.jpg

    2 top 0.3700868223351246 bottom 0.38732149216507755 outside 90.26436680474167
    line 580: rotated size 3470 5078
    line 640: outside top (31.7396240234375, 77.890380859375) outside bottom (41.12670135498047, 5037.9697265625)
    line 650: crop 0 36 expected 3772
    line 660: rotated size 3470 5078
    writing 065-remove-page-borders/002.jpg

should i add more debug prints to debug the "lightgray strip at the top" issue?

## Response:

7/12/2026, 3:13:16 PM

Yes. I think we've reached the point where adding more debug prints is much more valuable than guessing.

One thing I noticed from your log is actually more important than the light gray strip.

For page 1:

``` text
outside top    (3454.8,   70)
outside bottom (3440.2, 5031)
```

The outside edge moves **15 pixels horizontally** over the page height:

``` text
3454.8 -> 3440.2
```

That means the page is **still tilted after your deskew**.

So before worrying about cropping, I'd verify whether the deskew is actually correct.

## I would print these

### 1. Rotation before/after

Before rotation:

``` python
print(
    f"before: "
    f"top={top_angle:.3f} "
    f"bottom={bottom_angle:.3f} "
    f"outside={outside_angle:.3f}"
)
```

After rotating and re-fitting:

``` python
print(
    f"after: "
    f"top={top_angle2:.3f} "
    f"bottom={bottom_angle2:.3f} "
    f"outside={outside_angle2:.3f}"
)
```

Ideally you should see something like

    before
    top      -0.41
    bottom   -0.39
    outside  89.77

    after
    top       0.00
    bottom    0.00
    outside  90.00

If you don't, the deskew itself is still wrong.

------------------------------------------------------------------------

### 2. Print the detected crop rectangle

``` python
print(
    x0_page,
    x1_page,
    y_top,
    y_bottom
)
```

and

``` python
print(
    expected_w,
    expected_h
)
```

------------------------------------------------------------------------

### 3. Draw the detected lines

This is the most useful one.

Take the rotated image and draw

- top line
- bottom line
- outside line

in different colors.

If the outside line is 15 pixels away from vertical, you'll see it immediately.

------------------------------------------------------------------------

## I also suspect one thing

Your current code computes

``` python
outside_top = intersect_lines(outside_line, top_line)
outside_bottom = intersect_lines(outside_line, bottom_line)
```

but then crops using

``` python
x_out = x0
```

or some variant of that.

I would **not** use `x0` from `cv2.fitLine()`.

`x0,y0` is just **one arbitrary point on the fitted line**. It is **not** the page edge.

Instead, always use

``` python
outside_top
outside_bottom
```

Those are the actual edge intersections.

------------------------------------------------------------------------

## About the light-gray strip

That strip strongly suggests that **the top crop coordinate is a little too high**.

I'd add one more debug image:

- draw the crop rectangle on the rotated image before cropping.

Something like:

``` python
vis = rotated.copy()

cv2.rectangle(
    vis,
    (int(x0_page), int(y_top)),
    (int(x1_page), int(y_bottom)),
    (0,0,255),
    3,
)

cv2.imwrite("crop_debug.jpg", vis)
```

That image will tell us in a few seconds whether:

- the page detection is wrong,
- the crop rectangle is wrong,
- or the repair step is overwriting part of the page.

I think that debug image will be much more informative than additional numeric prints.

## Prompt:

7/12/2026, 3:26:41 PM

> cv2.rectangle( vis, (int(x0_page), int(y_top)), (int(x1_page), int(y_bottom)), (0,0,255), 3, )

that red rectangle is always on the outter side of the page, outside of the page. on odd pages, the left edge of the rectangle is on the right edge of the page, and the right edge of the rectangle is on the right edge of the image. the top and bottom edges of the rectangle have correct Y positions (the top and bottom edges of the page)

## Response:

7/12/2026, 3:26:41 PM

That is a **very useful observation**. It narrows the bug down a lot.

The fact that:

- ✅ `y_top` is correct
- ✅ `y_bottom` is correct
- ❌ the rectangle is entirely **outside** the page horizontally

means **your outside edge detection is working**, but **you're constructing the crop rectangle on the wrong side of the outside edge**.

Looking at your code:

``` python
if bad_on_left:
    x_out = x0
    x0_page = x_out
    x1_page = x_out + expected_w
else:
    x_out = x0
    x1_page = x_out
    x0_page = x_out - expected_w
```

This logic assumes:

- odd page (`bad_on_left=True`): outside edge is the **left** side
- even page: outside edge is the **right** side

But that's the opposite of reality.

After your 060 rotation:

- **odd pages**
  - binding = left
  - **outside = right**
- **even pages**
  - binding = right
  - **outside = left**

So if `split_edge_candidates()` is correct, then

``` python
if bad_on_left:
    outside = pts[xs > xs.max() - margin_x]
else:
    outside = pts[xs < xs.min() + margin_x]
```

means

- odd pages → detected outside edge is the **right edge**
- even pages → detected outside edge is the **left edge**

Therefore the crop should be

``` python
if bad_on_left:
    # outside edge is RIGHT
    x1_page = x_out
    x0_page = x_out - expected_w
else:
    # outside edge is LEFT
    x0_page = x_out
    x1_page = x_out + expected_w
```

Notice that's exactly the opposite of your current code.

------------------------------------------------------------------------

## One more thing

I also would **not** use

``` python
x_out = x0
```

because `x0` is just the arbitrary anchor point returned by `cv2.fitLine()`.

You already computed

``` python
outside_top
outside_bottom
```

Use those instead:

``` python
x_out = (outside_top[0] + outside_bottom[0]) * 0.5
```

That is the actual detected page edge.

------------------------------------------------------------------------

The fact that your debug rectangle is perfectly aligned vertically but completely outside the page horizontally is almost a textbook symptom of "correct edge detected, wrong side chosen." I would try reversing the crop logic above first. I think there's a good chance that alone will move the rectangle onto the page.

## Prompt:

7/12/2026, 3:48:39 PM

> Therefore the crop should be

ok, now the red crop rectangle in the debug output is correct, but it looks like the cropping is never applied, because the output image has the exact same size as the debug image, and the page contents have the exact same positions in both images..... only the grey margins were overpainted by the "fill white near edges" logic....

here is my full script, maybe you see the bug

``` py
#!/usr/bin/env python3
"""
extract scanned page from gray background

restore the binding edge of the page
by filling the missing width
with the average color near the binding edge
"""

INPUT_DIR = "060-rotate-crop"
OUTPUT_DIR = "065-remove-page-borders"

# === Tuning parameters ===
DEBUG = True
DEBUG = False
BORDER_SIZE = 100  # pixels

RANSAC_ITER = 400
RANSAC_INLIER_DIST = 6.0
# RANSAC_MIN_INLIERS = 30
RANSAC_MIN_INLIERS = 20

THRESH_HIGH_PERCENTILE = 99
THRESH_MIN = 200

import os
import re
import math
import random
import numpy as np
import cv2

from _shared import (
    load_config,
    get_page_num,
)

config = load_config()

# no, this is wrong if (config.do_rotate == True)
# scan_x = config.scan_x
# scan_y = config.scan_y
# config.scan_aspect = scan_x / scan_y
# ASPECT = config.scan_aspect

# TODO move to load_config
# book binding side = scanner top side
config.use_three_edge_deskew = (
    config.do_rotate
    and config.rotate_odd == 270
    and config.rotate_even == 90
)

print(f"config.use_three_edge_deskew: {config.use_three_edge_deskew}")

# TODO move to load_config
def get_rotated_scan_x_y(config):
    # scanned image size after 060-rotate-crop.py
    # NOTE dont apply crop
    x = config.scan_x
    y = config.scan_y

    if config.use_three_edge_deskew:
        # rotate by 90 or 270 degrees
        x, y = y, x

    return (x, y)

(
    config.rotated_scan_x,
    config.rotated_scan_y
) = get_rotated_scan_x_y(config)

config.rotated_scan_aspect = config.rotated_scan_x / config.rotated_scan_y

ASPECT = config.rotated_scan_aspect

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def save_dbg(img, path):
    ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, img)

def percentile_threshold(gray):
    high_p = np.percentile(gray, THRESH_HIGH_PERCENTILE)
    thr = max(THRESH_MIN, int(high_p * 0.95))
    _, mask = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
    return mask, thr, int(high_p)

def detect_vertical_streaks(mask, approx_width=3, length_thresh_ratio=0.15):
    h, w = mask.shape
    kx = approx_width
    ky = max(15, int(h * 0.02))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
    long_vertical = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(long_vertical, connectivity=8)
    streak_mask = np.zeros_like(mask)
    length_thresh = max(10, int(h * length_thresh_ratio))
    for i in range(1, num_labels):
        x, y, ww, hh, area = stats[i]
        if hh >= length_thresh and ww <= max(5, int(w * 0.01)):
            streak_mask[labels == i] = 255
    return streak_mask

def keep_largest_component(mask):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    out = np.zeros_like(mask)
    out[labels == best] = 255
    return out

def contour_to_pts(contour):
    return contour.reshape(-1,2)

def fit_line_ransac(pts, iterations=RANSAC_ITER, inlier_dist=RANSAC_INLIER_DIST, min_inliers=RANSAC_MIN_INLIERS):
    if len(pts) < 2:
        raise ValueError("Not enough points")
    best_inliers = None
    best_model = None
    n = len(pts)
    ptsf = pts.astype(np.float32)
    for _ in range(iterations):
        i1, i2 = random.sample(range(n), 2)
        p1 = ptsf[i1]; p2 = ptsf[i2]
        vx = float(p2[0] - p1[0]); vy = float(p2[1] - p1[1])
        if vx == 0 and vy == 0:
            continue
        dists = np.abs(vy*(ptsf[:,0]-p1[0]) - vx*(ptsf[:,1]-p1[1])) / (math.hypot(vx, vy) + 1e-12)
        inliers = dists <= inlier_dist
        cnt = int(inliers.sum())
        if cnt >= min_inliers and (best_inliers is None or cnt > int(best_inliers.sum())):
            best_inliers = inliers.copy()
            best_model = (vx, vy, float(p1[0]), float(p1[1]))
    if best_model is None:
        vx, vy, x0, y0 = cv2.fitLine(ptsf, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        inlier_mask = np.ones(len(pts), dtype=bool)
        return float(vx), float(vy), float(x0), float(y0), inlier_mask
    inlier_pts = ptsf[best_inliers]
    vx, vy, x0, y0 = cv2.fitLine(inlier_pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    inlier_mask = best_inliers
    return float(vx), float(vy), float(x0), float(y0), inlier_mask

def intersect_lines(l1, l2):
    vx1, vy1, x1, y1 = l1
    vx2, vy2, x2, y2 = l2
    A = np.array([[vx1, -vx2], [vy1, -vy2]], dtype=np.float32)
    b = np.array([x2 - x1, y2 - y1], dtype=np.float32)
    det = np.linalg.det(A)
    if abs(det) < 1e-8:
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    t1, t2 = np.linalg.solve(A, b)
    xi = x1 + t1 * vx1
    yi = y1 + t1 * vy1
    return float(xi), float(yi)

def build_affine_without_shear(pt_v_top, pt_v_bot, expected_w, expected_h, bad_on_left, img, dbgdir=None):
    """
    Build a source triangle that enforces orthogonal page axes (no shear) using:
      - pt_v_top: intersection of top line and good vertical (top-right when good vertical is right)
      - pt_v_bot: intersection of bottom line and good vertical (bottom-right when good vertical is right)
    Returns (M_aff, src_corners_dict, dst_corners_dict, reason)
    - M_aff: 2x3 affine matrix or None on failure
    - src_corners_dict: dict of TL, TR, BL, BR (np.float32 2-vectors)
    - dst_corners_dict: dict of TL, TR, BL, BR in destination coordinates
    - reason: diagnostic string
    """
    # convert to vectors
    pt_v_top = np.array(pt_v_top, dtype=np.float32)
    pt_v_bot = np.array(pt_v_bot, dtype=np.float32)

    # compute direction along top edge: vector from bottom intersection to top intersection projected horizontally
    # but we don't have explicit top_line direction vector here; we will approximate using (pt_v_top - pt_v_bot) rotated?
    # Better: caller should supply top_line direction (vx,vy). If you don't have it, approximate from nearby contour points.
    # For compatibility with your code, expect you have top_vx, top_vy; if not, fall back to vector between two top contour points.
    # Here we'll assume top_vx, top_vy are available in outer scope; otherwise compute from pt_v_top->pt_v_bot displacement projected perpendicular:
    # To keep this helper self-contained, we compute top_dir as average of (pt_v_top_to_some_right) - but simplest robust approach:
    # compute approximate top direction by taking small offset vector along top by sampling image gradient: fallback to (1,0).

    # --- Here caller should provide top_unit; we'll compute a safe top_unit using neighbor pixels if available ---
    # We'll attempt to derive a top_unit by sampling a short step along the image: use vector from pt_v_top to pt_v_top projected to image center
    cx, cy = img.shape[1] / 2.0, img.shape[0] / 2.0
    # prefer vector pointing rightwards by projecting displacement to center
    approx = pt_v_top - np.array([cx, cy], dtype=np.float32)
    if np.linalg.norm(approx) < 1e-6:
        approx = np.array([1.0, 0.0], dtype=np.float32)
    ux, uy = approx / (np.linalg.norm(approx) + 1e-12)

    # Force ux positive (pointing to the right) — we want u to be rightward
    if ux < 0:
        ux, uy = -ux, -uy

    # make perpendicular downwards: p = (-uy, ux)
    px, py = -uy, ux

    # normalize again (just in case)
    n_u = math.hypot(ux, uy)
    n_p = math.hypot(px, py)
    if n_u < 1e-6 or n_p < 1e-6:
        return None, None, None, "degenerate_axes"
    ux, uy = ux / n_u, uy / n_u
    px, py = px / n_p, py / n_p

    # Now construct corners. We know pt_v_top/pt_v_bot correspond to the known vertical edge:
    # If bad_on_left is True, good vertical is the RIGHT edge => pt_v_top == TR, pt_v_bot == BR.
    # If bad_on_left is False, good vertical is LEFT edge => pt_v_top == TL, pt_v_bot == BL.
    if bad_on_left:
        src_TR = pt_v_top
        src_BR = pt_v_bot
        # compute TL and BL by moving left along top unit by expected_w
        src_TL = src_TR - np.array([ux, uy], dtype=np.float32) * float(expected_w)
        src_BL = src_BR - np.array([ux, uy], dtype=np.float32) * float(expected_w)
    else:
        src_TL = pt_v_top
        src_BL = pt_v_bot
        # compute TR and BR by moving right along top unit by expected_w
        src_TR = src_TL + np.array([ux, uy], dtype=np.float32) * float(expected_w)
        src_BR = src_BL + np.array([ux, uy], dtype=np.float32) * float(expected_w)

    # Optional: enforce vertical height by projecting (TL->BL) onto perp and rescale to expected_h
    # Compute current vertical vector and its projection onto perp (px,py)
    cur_v = src_BL - src_TL
    proj = cur_v.dot(np.array([px, py], dtype=np.float32))
    # if projection is tiny, fall back but otherwise adjust BL to ensure exact page height
    if abs(proj) > 1e-6:
        # adjust scale to exact expected_h
        scale = float(expected_h) / proj
        # recompute BL, BR as TL + perp*expected_h and TR + perp*expected_h
        src_BL = src_TL + np.array([px, py], dtype=np.float32) * float(expected_h)
        src_BR = src_TR + np.array([px, py], dtype=np.float32) * float(expected_h)
    else:
        # if proj nearly zero, we cannot rely on vertical measurement; still use perpendicular step
        src_BL = src_TL + np.array([px, py], dtype=np.float32) * float(expected_h)
        src_BR = src_TR + np.array([px, py], dtype=np.float32) * float(expected_h)

    # Build src/dst dicts
    src_corners = {
        "TL": src_TL.astype(np.float32),
        "TR": src_TR.astype(np.float32),
        "BL": src_BL.astype(np.float32),
        "BR": src_BR.astype(np.float32)
    }
    dst_corners = {
        "TL": np.array([0.0, 0.0], dtype=np.float32),
        "TR": np.array([expected_w - 1.0, 0.0], dtype=np.float32),
        "BL": np.array([0.0, expected_h - 1.0], dtype=np.float32),
        "BR": np.array([expected_w - 1.0, expected_h - 1.0], dtype=np.float32)
    }

    # choose triangle for affine depending on orientation (map TL, BL, TR -> dst TL, BL, TR)
    src_tri = np.vstack([src_corners["TL"], src_corners["BL"], src_corners["TR"]]).astype(np.float32)
    dst_tri = np.vstack([dst_corners["TL"], dst_corners["BL"], dst_corners["TR"]]).astype(np.float32)

    # validate triangle
    area = abs(0.5 * (src_tri[0,0]*(src_tri[1,1]-src_tri[2,1]) + src_tri[1,0]*(src_tri[2,1]-src_tri[0,1]) + src_tri[2,0]*(src_tri[0,1]-src_tri[1,1])))
    if area < 1.0 or not np.isfinite(src_tri).all():
        return None, src_corners, dst_corners, f"invalid_src_tri_area={area:.3f}"

    try:
        M = cv2.getAffineTransform(src_tri, dst_tri)
    except cv2.error as e:
        return None, src_corners, dst_corners, f"getAffineTransform_failed:{e}"

    # debug overlay: draw corners & axes if dbgdir provided
    if dbgdir is not None:
        vis = img.copy()
        def draw_pt(pt, color=(0,0,255), tag=""):
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 6, color, -1)
            if tag:
                cv2.putText(vis, tag, (int(pt[0]+6), int(pt[1]-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        draw_pt(src_corners["TL"], (0,255,0), "TL")
        draw_pt(src_corners["TR"], (0,128,255), "TR")
        draw_pt(src_corners["BL"], (255,0,0), "BL")
        draw_pt(src_corners["BR"], (255,255,0), "BR")
        # draw axis arrows from TL
        origin = src_corners["TL"]
        arrow_u = origin + np.array([ux, uy], dtype=np.float32) * 80.0
        arrow_p = origin + np.array([px, py], dtype=np.float32) * 80.0
        cv2.arrowedLine(vis, (int(origin[0]), int(origin[1])), (int(arrow_u[0]), int(arrow_u[1])), (255,0,255), 2)
        cv2.arrowedLine(vis, (int(origin[0]), int(origin[1])), (int(arrow_p[0]), int(arrow_p[1])), (0,255,255), 2)
        cv2.imwrite(os.path.join(dbgdir, "debug_axes_and_corners.png"), vis)

    return M, src_corners, dst_corners, "ok"

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

def split_edge_candidates(contour, bad_on_left):
    pts = contour.reshape(-1, 2)

    xs = pts[:,0]
    ys = pts[:,1]

    w = xs.max() - xs.min()
    h = ys.max() - ys.min()

    margin_x = w * 0.15
    margin_y = h * 0.15

    # outside edge
    if bad_on_left:
        outside = pts[xs > xs.max() - margin_x]
    else:
        outside = pts[xs < xs.min() + margin_x]

    # top edge
    top = pts[ys < ys.min() + margin_y]

    # bottom edge
    bottom = pts[ys > ys.max() - margin_y]

    return top, bottom, outside

def line_angle(line):
    vx, vy, _, _ = line
    return math.atan2(vy, vx)

def horizontal_line_angle(line):
    vx, vy, _, _ = line

    a = math.degrees(math.atan2(vy, vx))

    while a < -45:
        a += 180

    while a > 45:
        a -= 180

    return a

def vertical_line_angle(line):
    vx, vy, _, _ = line

    a = math.degrees(math.atan2(vy, vx))

    while a < 45:
        a += 180

    while a > 135:
        a -= 180

    return a

def normalize_angle_deg(a):
    while a < -90:
        a += 180
    while a > 90:
        a -= 180
    return a

def get_gray_mask_contours(img, dbgdir):
    # --- threshold for geometry only ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask_init, thr, hp = percentile_threshold(gray)
    if cv2.mean(gray)[0] < 127:
        mask_init = cv2.bitwise_not(mask_init)
    if DEBUG:
        save_dbg(gray, os.path.join(dbgdir, "01_gray.png"))
        save_dbg(mask_init, os.path.join(dbgdir, f"02_thresh_thr{thr}_hp{hp}.png"))

    # without a9dc2b24e6b0d1d49b6fc232223d6431ba3442a5 bad: fix perspective transform for broken ADF scanners
    # Invert if necessary, so the page is always "white" for thresholding
    # We'll use Otsu's method to detect high-contrast area
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Determine whether the page is darker or lighter than background
    mean_val = cv2.mean(gray, mask=None)[0]
    if mean_val < 127:  # dark page: invert mask
        mask = cv2.bitwise_not(mask)

    # Morphology to remove small gaps
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return gray, mask, contours

def repair_binding(img, bad_on_left, width=50):

    h,w = img.shape[:2]

    if bad_on_left:
        sample_range = range(width)
        fill_range = range(width)
        sample_x = width
    else:
        sample_range = range(w-width,w)
        fill_range = range(w-width,w)
        sample_x = w-width

    for y in range(h):

        if bad_on_left:
            sample = img[y, width:width+20]
            color = np.mean(sample,axis=0)
            img[y,:width] = color

        else:
            sample = img[y,w-width-20:w-width]
            color = np.mean(sample,axis=0)
            img[y,w-width:] = color

    return img

def process_image(in_path, out_path):
    """
    Robust page extraction that handles missing top (or bottom) edges.
    - preserves original colors (warps the original image)
    - builds orthogonal axes (no shear)
    - if top is missing, uses bottom + good vertical to infer top by moving up expected_h
    - fills outside the filled page polygon with pure white
    - extensive debug output in OUTPUT_DIR/debug/<page_num>/
    """
    # ---------- small helpers ----------
    def ensure_dir(p):
        os.makedirs(p, exist_ok=True)

    def save_dbg(img, path):
        ensure_dir(os.path.dirname(path))
        cv2.imwrite(path, img)

    def percentile_threshold(gray):
        high_p = np.percentile(gray, THRESH_HIGH_PERCENTILE)
        thr = max(THRESH_MIN, int(high_p * 0.95))
        _, mask = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
        return mask, thr, int(high_p)

    def keep_largest_component(mask):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels <= 1:
            return mask
        areas = stats[1:, cv2.CC_STAT_AREA]
        best = 1 + int(np.argmax(areas))
        out = np.zeros_like(mask)
        out[labels == best] = 255
        return out

    def detect_vertical_streaks(mask, approx_width=3, length_thresh_ratio=0.15):
        h, w = mask.shape
        kx = approx_width
        ky = max(15, int(h * 0.02))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
        long_vertical = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(long_vertical, connectivity=8)
        streak_mask = np.zeros_like(mask)
        length_thresh = max(10, int(h * length_thresh_ratio))
        for i in range(1, num_labels):
            x, y, ww, hh, area = stats[i]
            if hh >= length_thresh and ww <= max(5, int(w * 0.01)):
                streak_mask[labels == i] = 255
        return streak_mask

    def contour_to_pts(c):
        return c.reshape(-1, 2)

    def fit_line_ransac(pts, iterations=RANSAC_ITER, inlier_dist=RANSAC_INLIER_DIST, min_inliers=RANSAC_MIN_INLIERS):
        if len(pts) < 2:
            raise ValueError("Not enough points for line fit")
        best_inliers = None
        best_cnt = 0
        best_model = None
        ptsf = pts.astype(np.float32)
        n = len(ptsf)
        for _ in range(iterations):
            i1, i2 = random.sample(range(n), 2)
            p1 = ptsf[i1]; p2 = ptsf[i2]
            vx = float(p2[0] - p1[0]); vy = float(p2[1] - p1[1])
            if abs(vx) < 1e-6 and abs(vy) < 1e-6:
                continue
            dists = np.abs(vy*(ptsf[:,0]-p1[0]) - vx*(ptsf[:,1]-p1[1])) / (math.hypot(vx, vy) + 1e-12)
            inliers = dists <= inlier_dist
            cnt = int(inliers.sum())
            if cnt >= min_inliers and cnt > best_cnt:
                best_cnt = cnt
                best_inliers = inliers.copy()
                best_model = (vx, vy, float(p1[0]), float(p1[1]))
        if best_model is None:
            vx, vy, x0, y0 = cv2.fitLine(ptsf, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
            inlier_mask = np.ones(len(ptsf), dtype=bool)
            return float(vx), float(vy), float(x0), float(y0), inlier_mask
        inlier_pts = ptsf[best_inliers]
        vx, vy, x0, y0 = cv2.fitLine(inlier_pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        return float(vx), float(vy), float(x0), float(y0), best_inliers

    def intersect_lines(l1, l2):
        vx1, vy1, x1, y1 = l1
        vx2, vy2, x2, y2 = l2
        A = np.array([[vx1, -vx2], [vy1, -vy2]], dtype=np.float32)
        b = np.array([x2 - x1, y2 - y1], dtype=np.float32)
        det = np.linalg.det(A)
        if abs(det) < 1e-8:
            return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        t1, t2 = np.linalg.solve(A, b)
        xi = x1 + t1 * vx1
        yi = y1 + t1 * vy1
        return float(xi), float(yi)

    # ---------- start ----------
    fname = os.path.basename(in_path)
    m = re.match(r"^(\d+)", fname)
    page_num = int(m.group(1)) if m else 0
    bad_on_left = (page_num % 2 == 1)

    img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print("Failed to read", in_path); return
    input_is_grayscale = (len(img.shape) == 2)
    if len(img.shape) == 2:
        # OpenCV expects 3-channel images for many operations
        # later convert back to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    H_img, W_img = img.shape[:2]

    # FIXME wrong H_img?
    expected_w = int(round(ASPECT * H_img))
    expected_h = H_img

    dbgdir = os.path.join(OUTPUT_DIR, "debug", f"{page_num:03d}")
    if DEBUG: ensure_dir(dbgdir)

    gray, mask, contours = get_gray_mask_contours(img, dbgdir)

    if not contours:
        print(f"Warning: no contours found in {in_path}")
        return

    page_contour = max(contours, key=cv2.contourArea)

    if config.use_three_edge_deskew:

        top_pts, bottom_pts, outside_pts = split_edge_candidates(
            page_contour,
            bad_on_left
        )

        top_line = fit_line_ransac(top_pts)[:4]
        bottom_line = fit_line_ransac(bottom_pts)[:4]
        outside_line = fit_line_ransac(outside_pts)[:4]

        # old
        # top_angle = math.degrees(line_angle(top_line))
        # bottom_angle = math.degrees(line_angle(bottom_line))
        # outside_angle = math.degrees(line_angle(outside_line))

        # new
        top_angle = horizontal_line_angle(top_line)
        bottom_angle = horizontal_line_angle(bottom_line)
        outside_angle = vertical_line_angle(outside_line)

        # start debug prints
        print()
        print(f"line 570: page_num={page_num}")

        rotation_error = -1 * (
            -top_angle
            -bottom_angle
        ) / 2.0

        Mrot = cv2.getRotationMatrix2D(
            (W_img/2, H_img/2),
            rotation_error,
            1.0
        )

        print(
            f"line 575: before rotation: "
            f"top_angle={top_angle:.3f} "
            f"bottom_angle={bottom_angle:.3f} "
            f"outside_angle={outside_angle:.3f}"
        )

        rotated = cv2.warpAffine(
            img,
            Mrot,
            (W_img, H_img),
            borderValue=(255,255,255)
        )

        Hr, Wr = rotated.shape[:2]
        print("line 580: rotated size", Wr, Hr)

        # img = rotated # ?

        # re-detect lines in the rotated image

        # gray, mask, contours = get_gray_mask_contours(img, dbgdir)
        gray, mask, contours = get_gray_mask_contours(rotated, dbgdir)

        top_angle2 = horizontal_line_angle(top_line)
        bottom_angle2 = horizontal_line_angle(bottom_line)
        outside_angle2 = vertical_line_angle(outside_line)

        print(
            f"line 620: after rotation and re-fitting: "
            f"top_angle2={top_angle2:.3f} "
            f"bottom_angle2={bottom_angle2:.3f} "
            f"outside_angle2={outside_angle2:.3f}"
        )

        if not contours:
            print(f"line 630: Warning: no contours found in {in_path}")
            return

        page_contour = max(contours, key=cv2.contourArea)

        top_pts, bottom_pts, outside_pts = split_edge_candidates(
            page_contour,
            bad_on_left
        )

        top_line = fit_line_ransac(top_pts)[:4]
        bottom_line = fit_line_ransac(bottom_pts)[:4]
        outside_line = fit_line_ransac(outside_pts)[:4]

        # FIXME H_img is wrong
        expected_h = H_img
        expected_w = int(round(ASPECT * expected_h))

        vx, vy, x0, y0 = outside_line

        if 0:
            old
            if bad_on_left:
                x_out = x0
                x0_page = x_out
                x1_page = x_out + expected_w
            else:
                x_out = x0
                x1_page = x_out
                x0_page = x_out - expected_w
        else:
            # new
            outside_top = intersect_lines(
                outside_line,
                top_line
            )

            outside_bottom = intersect_lines(
                outside_line,
                bottom_line
            )

            x_out = (
                outside_top[0]
                +
                outside_bottom[0]
            ) / 2

            x_out = int(round(x_out))

            # if bad_on_left:
            #     x0_page = x_out
            #     x1_page = x_out + expected_w
            # else:
            #     x0_page = x_out - expected_w
            #     x1_page = x_out

            if bad_on_left:
                # outside edge is RIGHT
                x1_page = x_out
                x0_page = x_out - expected_w
            else:
                # outside edge is LEFT
                x0_page = x_out
                x1_page = x_out + expected_w

            # clamp
            x0_page = max(0, x0_page)
            x1_page = min(rotated.shape[1], x1_page)

            print(
                "line 640:",
                f"outside_top={outside_top}",
                f"outside_bottom={outside_bottom}",
            )

        y_top = int(round(outside_top[1]))
        y_bottom = int(round(outside_bottom[1]))
        crop = rotated[
            y_top:y_top+expected_h,
            x0_page:x1_page
        ]

        print(
            "line 650: crop",
            f"x0_page={x0_page}",
            f"x1_page={x1_page}",
            f"y_top={y_top}",
            f"y_bottom={y_bottom}",
            f"expected_w={expected_w}",
            f"expected_h={expected_h}",
        )

        # crop = rotated[
        #     0:expected_h,
        #     int(x0_page):int(x1_page)
        # ]

        if 1:
            # debug
            vis = rotated.copy()
            cv2.rectangle(
                vis,
                (int(x0_page), int(y_top)),
                (int(x1_page), int(y_bottom)),
                (0,0,255),
                3,
            )
            name = f"{page_num:03d}.crop_debug.jpg"
            print(f"writing {name}")
            cv2.imwrite(name, vis)

        # y_top = round(outside_top[1])
        # y_bottom = round(outside_bottom[1])
        # actual_height = y_bottom - y_top
        # crop = rotated[
        #     y_top:y_top + expected_h,
        #     x0_page:x1_page
        # ]

        # y_top = int(round(outside_top[1]))
        # y_bottom = int(round(outside_bottom[1]))
        # crop = rotated[
        #     y_top:y_bottom,
        #     x0_page:x1_page
        # ]

        # img = repair_binding(img, bad_on_left, width=50)
        if 1:
            rotated = repair_binding(rotated, bad_on_left, width=50)

            Hr, Wr = rotated.shape[:2]
            print("line 660: rotated size", Wr, Hr)

            warped = rotated
        else:
            crop = repair_binding(crop, bad_on_left, width=50)
            warped = crop

    else:
        # config.use_three_edge_deskew == False
        # FIXME use only two page edges: outside, bottom
        # Approximate contour to quadrilateral
        epsilon = 0.02 * cv2.arcLength(page_contour, True)
        approx = cv2.approxPolyDP(page_contour, epsilon, True)
        if len(approx) != 4:
            approx = cv2.convexHull(page_contour)
            if len(approx) < 4:
                print(f"Warning: not enough points for perspective in {in_path}")
                return
            # pick 4 extreme points
            pts = np.array([
                approx[approx[:,0,0].argmin()][0],  # leftmost
                approx[approx[:,0,1].argmin()][0],  # topmost
                approx[approx[:,0,0].argmax()][0],  # rightmost
                approx[approx[:,0,1].argmax()][0]   # bottommost
            ])
        else:
            pts = approx.reshape(4,2)

        rect = order_points(pts)

        # Perspective transform
        widthA = np.linalg.norm(rect[2] - rect[3])
        widthB = np.linalg.norm(rect[1] - rect[0])
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.linalg.norm(rect[1] - rect[2])
        heightB = np.linalg.norm(rect[0] - rect[3])
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    # # Add internal white border
    # h, w = warped.shape[:2]
    # canvas = np.ones_like(warped) * 255
    # b = BORDER_SIZE
    # canvas[b:h-b, b:w-b] = warped[b:h-b, b:w-b]

    h, w = warped.shape[:2]
    b = BORDER_SIZE

    # Prepare a canvas with the same content as warped
    canvas = warped.copy()

    # Function to compute average color along a strip
    def avg_color_strip(img, axis, start, end, strip_width=1):
        if axis == 'top':
            strip = img[start:end, :, :]
        elif axis == 'bottom':
            strip = img[h-end:h-start, :, :]
        elif axis == 'left':
            strip = img[:, start:end, :]
        elif axis == 'right':
            strip = img[:, w-end:w-start, :]
        else:
            raise ValueError("Invalid axis")
        return np.mean(strip, axis=(0,1)).astype(np.uint8)

    # Fill borders with local average color
    canvas[0:b, :, :] = avg_color_strip(canvas, 'top', 50, 100)       # top border
    canvas[h-b:h, :, :] = avg_color_strip(canvas, 'bottom', 50, 100)  # bottom border
    canvas[:, 0:b, :] = avg_color_strip(canvas, 'left', 50, 100)      # left border
    canvas[:, w-b:w, :] = avg_color_strip(canvas, 'right', 50, 100)   # right border

    if input_is_grayscale:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

    ensure_dir(os.path.dirname(out_path))
    cv2.imwrite(out_path, canvas)
    print(f"writing {out_path}")

def main():
    ensure_dir(OUTPUT_DIR)
    # TODO use image_format from 030-measure-page-size.txt
    image_format = "jpg"
    files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(f".{image_format}")])
    if not files:
        print("No image files found in", INPUT_DIR)
        return
    for fname in files:
        m = re.match(r"^(\d+)", fname)
        # page_num = int(m.group(1)) if m else 0
        page_num = int(m.group(1)) # can throw
        # if page_num != 14: continue # debug
        # if not page_num in [340, 345]: continue # debug
        # if page_num < 300: continue # debug
        # if page_num != 320: continue # debug
        in_path = os.path.join(INPUT_DIR, fname)
        out_path = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(out_path):
            continue
        try:
            process_image(in_path, out_path)
        except Exception as exc:
            print(f"Error processing {fname}: {exc}")
            raise

if __name__ == "__main__":
    main()
```

## Response:

7/12/2026, 3:48:39 PM

Yes, there is a very clear bug. The crop rectangle is correct now, but you never use it.

This part:

``` py
crop = rotated[
    y_top:y_top+expected_h,
    x0_page:x1_page
]

print(
    "line 650: crop",
    ...
)
```

creates the correct crop.

But immediately afterwards:

``` py
if 1:
    # debug
    vis = rotated.copy()
    cv2.rectangle(...)
    cv2.imwrite(name, vis)

if 1:
    rotated = repair_binding(rotated, bad_on_left, width=50)

    Hr, Wr = rotated.shape[:2]
    print("line 660: rotated size", Wr, Hr)

    warped = rotated
```

You throw away `crop` and set:

``` py
warped = rotated
```

so the output is always the full rotated image. The only thing you see changing is the later border-fill code painting the gray margins.

Change this:

``` py
if 1:
    rotated = repair_binding(rotated, bad_on_left, width=50)

    Hr, Wr = rotated.shape[:2]
    print("line 660: rotated size", Wr, Hr)

    warped = rotated
```

to:

``` py
if 1:
    crop = repair_binding(crop, bad_on_left, width=50)

    Hr, Wr = crop.shape[:2]
    print("line 660: crop size", Wr, Hr)

    warped = crop
```

------------------------------------------------------------------------

There is another issue you will probably hit next:

Your `repair_binding()` assumes the missing binding area is inside the image:

``` py
if bad_on_left:
    sample = img[y, width:width+20]
    img[y,:width] = color
```

That is only correct if the crop already has the correct page width and the missing area is at the left edge.

Since your crop rectangle is now correct, this is probably what you want. But check that `crop.shape[1] == expected_w`.

Add:

``` py
print(
    "crop actual:",
    crop.shape[1],
    "expected:",
    expected_w
)
```

right after the crop.

I expect you will see something like:

    crop actual: 3772 expected: 3772

instead of the old:

    rotated size 3478 5078

because the previous output was still the uncropped rotated scan.

------------------------------------------------------------------------

One more bug: your debug rectangle is misleading because it is drawn on `rotated`:

``` py
vis = rotated.copy()
```

After the fix, make it:

``` py
vis = rotated.copy()
cv2.rectangle(...)
```

as you have it now, but remember that this is only a visualization of the crop. The actual image to save is `warped`.

After this change, if there is still a light gray strip, then it is a real geometry/fill problem. Right now the crop pipeline is simply bypassed.

## Prompt:

7/12/2026, 3:56:51 PM

ok, fixed

debug prints:

    line 570: page_num=1
    line 575: before rotation: top_angle=-0.411 bottom_angle=-0.389 outside_angle=89.769
    line 580: rotated size 3478 5078
    line 620: after rotation and re-fitting: top_angle2=-0.411 bottom_angle2=-0.389 outside_angle2=89.769
    line 640: outside_top=(3454.768310546875, 69.978271484375) outside_bottom=(3440.152587890625, 5031.193359375)
    line 650: crop x0_page=0 x1_page=3447 y_top=70 y_bottom=5031 expected_w=3772 expected_h=5078
    line 660: crop size 3447 5008
    line 760: after crop: crop actual: crop.shape[1]=3447 expected_w=3772
    writing 065-remove-page-borders/001.jpg

    line 570: page_num=2
    line 575: before rotation: top_angle=0.370 bottom_angle=0.387 outside_angle=90.264
    line 580: rotated size 3470 5078
    line 620: after rotation and re-fitting: top_angle2=0.370 bottom_angle2=0.387 outside_angle2=90.264
    line 640: outside_top=(31.709163665771484, 77.89013671875) outside_bottom=(41.22395706176758, 5037.9697265625)
    line 650: crop x0_page=36 x1_page=3470 y_top=78 y_bottom=5038 expected_w=3772 expected_h=5078
    line 660: crop size 3434 5000
    line 760: after crop: crop actual: crop.shape[1]=3434 expected_w=3772
    writing 065-remove-page-borders/002.jpg

... these are looking good, no?

next problem:  
on some pages, the script fails to detect the page edges, but instead, it detects the edges of a dark rectangle on a white page...  
what helps here is that the scanner always adds a midgray (50% gray) background behind the scanned pages, so the actual page edge is always surrounded by a gray margin

## Response:

7/12/2026, 3:56:51 PM

Yes, these numbers are **much better**. The pipeline is finally using the crop.

However, there is one thing to notice:

    expected_w=3772
    crop.shape[1]=3447

and

    expected_w=3772
    crop.shape[1]=3434

So the crop is still being clipped by the image boundary. The reason is here:

``` py
x0_page = max(0, x0_page)
x1_page = min(rotated.shape[1], x1_page)
```

For page 1:

    x0_page=0
    x1_page=3447
    rotated width=3478

The calculated page extends ~300 px beyond the left side. The clamp removes that missing part.

This is actually consistent with your original problem: the binding edge is missing and needs reconstruction.

So the next step is **not** to make the crop wider — it cannot be wider because the pixels do not exist. The correct sequence is:

1.  Calculate the full expected page rectangle.
2.  Crop the existing part.
3.  Create a new `expected_w × expected_h` canvas.
4.  Paste the existing crop at the correct offset.
5.  Fill the missing binding side using `repair_binding()`.

Right now you do step 2 only.

------------------------------------------------------------------------

For example, instead of:

``` py
x0_page = max(0, x0_page)
x1_page = min(rotated.shape[1], x1_page)

crop = rotated[
    y_top:y_top+expected_h,
    x0_page:x1_page
]
```

do something like:

``` py
full_crop = np.ones(
    (expected_h, expected_w, 3),
    dtype=rotated.dtype
) * 255

src_x0 = max(0, x0_page)
src_x1 = min(rotated.shape[1], x1_page)

src_y0 = max(0, y_top)
src_y1 = min(rotated.shape[0], y_top + expected_h)

dst_x0 = src_x0 - x0_page
dst_y0 = src_y0 - y_top

full_crop[
    dst_y0:dst_y0 + (src_y1-src_y0),
    dst_x0:dst_x0 + (src_x1-src_x0)
] = rotated[
    src_y0:src_y1,
    src_x0:src_x1
]

crop = full_crop
```

Now:

    crop.shape[1] == expected_w

always.

Then your existing:

``` py
crop = repair_binding(crop, bad_on_left, width=50)
```

can fill the missing strip.

------------------------------------------------------------------------

For your **dark rectangle detection problem**: yes, the gray margin is the key information. Right now your segmentation uses:

``` py
cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

That has no knowledge of "page vs printed content". A dark rectangle inside a white page can easily become the largest connected component.

A better strategy for your scanner:

### 1. Estimate the background color

The scanner background should be around 128.

Sample the outer border:

``` py
border = np.concatenate([
    gray[:100,:].ravel(),
    gray[-100:,:].ravel(),
    gray[:,:100].ravel(),
    gray[:,-100:].ravel(),
])

bg = np.median(border)
print("background:", bg)
```

You should see something like:

    background: 125

### 2. Detect distance from background

Instead of thresholding bright pixels:

``` py
diff = cv2.absdiff(
    gray,
    np.uint8(bg)
)

mask = diff > 20
```

Now:

- gray scanner background → 0
- white paper → 1
- dark rectangle → maybe 1 too, but...

### 3. Use the fact that the page touches the outer boundary of the gray region

The page contour should have a large rectangular shape surrounded by gray. Printed rectangles inside the page will not.

After getting contours:

``` py
contours, _ = cv2.findContours(
    mask.astype(np.uint8),
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)
```

filter by:

- area
- aspect ratio close to scan aspect
- distance from image edges

Something like:

``` py
for c in contours:
    x,y,w,h = cv2.boundingRect(c)

    aspect = w/h

    if area < 0.5 * gray.size:
        continue

    if abs(aspect - ASPECT) > 0.2:
        continue
```

The page should win over internal dark objects.

------------------------------------------------------------------------

One more thing: because your scanner background is reliable, I would actually stop using Otsu here. Otsu is solving a generic segmentation problem, but you have a much stronger prior:

> "Find the largest rectangle that separates from a mid-gray background."

That is much easier and should make the whole pipeline more stable.

## Prompt:

7/12/2026, 4:10:03 PM

> full_crop\[

no, this is wrong... expected_h and expected_w are wrong

``` py
    img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)

    H_img, W_img = img.shape[:2]

    # FIXME wrong H_img?
    expected_w = int(round(ASPECT * H_img))
    expected_h = H_img
```

`H_img, W_img` are simply the sizes of the original input image...

but the actual expected sizes are more complex. if the scan was rotated, then we have 3 page edges: top bottom outside. so expected_h is the distance between the top and bottom page edge. and we know the aspect ratio `ASPECT = config.rotated_scan_x / config.rotated_scan_y` so with expected_h and ASPECT we can calculate expected_w

## Response:

7/12/2026, 4:10:03 PM

You are right — my previous assumption was wrong. I treated `H_img` as if it represented the page height after rotation/cropping, but it is only the **canvas height of the rotated scan**. It has nothing to do with the actual page dimensions.

Your geometry is actually:

- The raw input image size (`W_img`, `H_img`) is just the scanner canvas.
- After `060-rotate-crop.py`, the page may have been rotated.
- In the three-edge case you have:
  - top edge
  - bottom edge
  - outside edge (the scanner-visible outer edge)
- The page height must come from the detected page edges, not from the image height.

So this:

``` py
expected_w = int(round(ASPECT * H_img))
expected_h = H_img
```

is indeed wrong.

The correct order should be:

1.  Detect the three page edges.
2.  Compute page height from top/bottom edge intersections.
3.  Compute page width from aspect ratio.

For example:

``` py
outside_top = intersect_lines(outside_line, top_line)
outside_bottom = intersect_lines(outside_line, bottom_line)

expected_h = int(round(
    math.dist(outside_top, outside_bottom)
))

expected_w = int(round(
    expected_h * ASPECT
))
```

But one subtle point: because the page edges are slightly rotated, `math.dist()` between the two outside intersections is the height **along the outside edge**, not necessarily the perpendicular page height.

A better version is:

``` py
page_height = math.hypot(
    outside_bottom[0] - outside_top[0],
    outside_bottom[1] - outside_top[1]
)

expected_h = int(round(page_height))
expected_w = int(round(expected_h * ASPECT))
```

For your debug values:

Page 1:

    outside_top=(3454.8, 69.98)
    outside_bottom=(3440.15, 5031.19)

Height:

    sqrt((-14.6)^2 + 4961.2^2) ≈ 4961

So expected height should be around 4961, not 5078.

Then:

    expected_w = 4961 * ASPECT

which should be close to the actual page width.

------------------------------------------------------------------------

This also explains the current crop result:

    expected_w=3772
    crop width=3447

The script is trying to create a page wider than the actual detected page because it used the scanner canvas height. The clamp then cuts it off.

So the missing binding reconstruction logic was being asked to compensate for a wrong page model.

The correct flow should be:

    detect edges
        |
        v
    calculate real page rectangle
        |
        v
    calculate expected_h / expected_w
        |
        v
    create full page canvas
        |
        v
    paste existing pixels
        |
        v
    repair missing binding edge

Your observation about `config.rotated_scan_x / config.rotated_scan_y` is exactly the right source for the aspect ratio; the missing piece was deriving the page height from the detected geometry rather than from `H_img`.

## Prompt:

7/12/2026, 4:34:28 PM

i have tried to apply your fix for expected_h and expected_w, and now the output height is correct, but the output width is wrong.... almost as if the image was expanded 2 times on the binding edge

maybe there is something wrong with x0_page or x1_page?

here is my full script

``` py
#!/usr/bin/env python3
"""
extract scanned page from gray background

restore the binding edge of the page
by filling the missing width
with the average color near the binding edge
"""

INPUT_DIR = "060-rotate-crop"
OUTPUT_DIR = "065-remove-page-borders"

# === Tuning parameters ===
DEBUG = True
DEBUG = False
BORDER_SIZE = 100  # pixels

RANSAC_ITER = 400
RANSAC_INLIER_DIST = 6.0
# RANSAC_MIN_INLIERS = 30
RANSAC_MIN_INLIERS = 20

THRESH_HIGH_PERCENTILE = 99
THRESH_MIN = 200

import os
import re
import math
import random
import numpy as np
import cv2

from _shared import (
    load_config,
    get_page_num,
)

config = load_config()

# no, this is wrong if (config.do_rotate == True)
# scan_x = config.scan_x
# scan_y = config.scan_y
# config.scan_aspect = scan_x / scan_y
# ASPECT = config.scan_aspect

# TODO move to load_config
# book binding side = scanner top side
config.use_three_edge_deskew = (
    config.do_rotate
    and config.rotate_odd == 270
    and config.rotate_even == 90
)

print(f"config.use_three_edge_deskew: {config.use_three_edge_deskew}")

# TODO move to load_config
def get_rotated_scan_x_y(config):
    # scanned image size after 060-rotate-crop.py
    # NOTE dont apply crop
    x = config.scan_x
    y = config.scan_y

    if config.use_three_edge_deskew:
        # rotate by 90 or 270 degrees
        x, y = y, x

    return (x, y)

(
    config.rotated_scan_x,
    config.rotated_scan_y
) = get_rotated_scan_x_y(config)

config.rotated_scan_aspect = config.rotated_scan_x / config.rotated_scan_y

ASPECT = config.rotated_scan_aspect

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def save_dbg(img, path):
    ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, img)

def percentile_threshold(gray):
    high_p = np.percentile(gray, THRESH_HIGH_PERCENTILE)
    thr = max(THRESH_MIN, int(high_p * 0.95))
    _, mask = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
    return mask, thr, int(high_p)

def detect_vertical_streaks(mask, approx_width=3, length_thresh_ratio=0.15):
    h, w = mask.shape
    kx = approx_width
    ky = max(15, int(h * 0.02))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
    long_vertical = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(long_vertical, connectivity=8)
    streak_mask = np.zeros_like(mask)
    length_thresh = max(10, int(h * length_thresh_ratio))
    for i in range(1, num_labels):
        x, y, ww, hh, area = stats[i]
        if hh >= length_thresh and ww <= max(5, int(w * 0.01)):
            streak_mask[labels == i] = 255
    return streak_mask

def keep_largest_component(mask):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    out = np.zeros_like(mask)
    out[labels == best] = 255
    return out

def contour_to_pts(contour):
    return contour.reshape(-1,2)

def fit_line_ransac(pts, iterations=RANSAC_ITER, inlier_dist=RANSAC_INLIER_DIST, min_inliers=RANSAC_MIN_INLIERS):
    if len(pts) < 2:
        raise ValueError("Not enough points")
    best_inliers = None
    best_model = None
    n = len(pts)
    ptsf = pts.astype(np.float32)
    for _ in range(iterations):
        i1, i2 = random.sample(range(n), 2)
        p1 = ptsf[i1]; p2 = ptsf[i2]
        vx = float(p2[0] - p1[0]); vy = float(p2[1] - p1[1])
        if vx == 0 and vy == 0:
            continue
        dists = np.abs(vy*(ptsf[:,0]-p1[0]) - vx*(ptsf[:,1]-p1[1])) / (math.hypot(vx, vy) + 1e-12)
        inliers = dists <= inlier_dist
        cnt = int(inliers.sum())
        if cnt >= min_inliers and (best_inliers is None or cnt > int(best_inliers.sum())):
            best_inliers = inliers.copy()
            best_model = (vx, vy, float(p1[0]), float(p1[1]))
    if best_model is None:
        vx, vy, x0, y0 = cv2.fitLine(ptsf, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        inlier_mask = np.ones(len(pts), dtype=bool)
        return float(vx), float(vy), float(x0), float(y0), inlier_mask
    inlier_pts = ptsf[best_inliers]
    vx, vy, x0, y0 = cv2.fitLine(inlier_pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    inlier_mask = best_inliers
    return float(vx), float(vy), float(x0), float(y0), inlier_mask

def intersect_lines(l1, l2):
    vx1, vy1, x1, y1 = l1
    vx2, vy2, x2, y2 = l2
    A = np.array([[vx1, -vx2], [vy1, -vy2]], dtype=np.float32)
    b = np.array([x2 - x1, y2 - y1], dtype=np.float32)
    det = np.linalg.det(A)
    if abs(det) < 1e-8:
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    t1, t2 = np.linalg.solve(A, b)
    xi = x1 + t1 * vx1
    yi = y1 + t1 * vy1
    return float(xi), float(yi)

def build_affine_without_shear(pt_v_top, pt_v_bot, expected_w, expected_h, bad_on_left, img, dbgdir=None):
    """
    Build a source triangle that enforces orthogonal page axes (no shear) using:
      - pt_v_top: intersection of top line and good vertical (top-right when good vertical is right)
      - pt_v_bot: intersection of bottom line and good vertical (bottom-right when good vertical is right)
    Returns (M_aff, src_corners_dict, dst_corners_dict, reason)
    - M_aff: 2x3 affine matrix or None on failure
    - src_corners_dict: dict of TL, TR, BL, BR (np.float32 2-vectors)
    - dst_corners_dict: dict of TL, TR, BL, BR in destination coordinates
    - reason: diagnostic string
    """
    # convert to vectors
    pt_v_top = np.array(pt_v_top, dtype=np.float32)
    pt_v_bot = np.array(pt_v_bot, dtype=np.float32)

    # compute direction along top edge: vector from bottom intersection to top intersection projected horizontally
    # but we don't have explicit top_line direction vector here; we will approximate using (pt_v_top - pt_v_bot) rotated?
    # Better: caller should supply top_line direction (vx,vy). If you don't have it, approximate from nearby contour points.
    # For compatibility with your code, expect you have top_vx, top_vy; if not, fall back to vector between two top contour points.
    # Here we'll assume top_vx, top_vy are available in outer scope; otherwise compute from pt_v_top->pt_v_bot displacement projected perpendicular:
    # To keep this helper self-contained, we compute top_dir as average of (pt_v_top_to_some_right) - but simplest robust approach:
    # compute approximate top direction by taking small offset vector along top by sampling image gradient: fallback to (1,0).

    # --- Here caller should provide top_unit; we'll compute a safe top_unit using neighbor pixels if available ---
    # We'll attempt to derive a top_unit by sampling a short step along the image: use vector from pt_v_top to pt_v_top projected to image center
    cx, cy = img.shape[1] / 2.0, img.shape[0] / 2.0
    # prefer vector pointing rightwards by projecting displacement to center
    approx = pt_v_top - np.array([cx, cy], dtype=np.float32)
    if np.linalg.norm(approx) < 1e-6:
        approx = np.array([1.0, 0.0], dtype=np.float32)
    ux, uy = approx / (np.linalg.norm(approx) + 1e-12)

    # Force ux positive (pointing to the right) — we want u to be rightward
    if ux < 0:
        ux, uy = -ux, -uy

    # make perpendicular downwards: p = (-uy, ux)
    px, py = -uy, ux

    # normalize again (just in case)
    n_u = math.hypot(ux, uy)
    n_p = math.hypot(px, py)
    if n_u < 1e-6 or n_p < 1e-6:
        return None, None, None, "degenerate_axes"
    ux, uy = ux / n_u, uy / n_u
    px, py = px / n_p, py / n_p

    # Now construct corners. We know pt_v_top/pt_v_bot correspond to the known vertical edge:
    # If bad_on_left is True, good vertical is the RIGHT edge => pt_v_top == TR, pt_v_bot == BR.
    # If bad_on_left is False, good vertical is LEFT edge => pt_v_top == TL, pt_v_bot == BL.
    if bad_on_left:
        src_TR = pt_v_top
        src_BR = pt_v_bot
        # compute TL and BL by moving left along top unit by expected_w
        src_TL = src_TR - np.array([ux, uy], dtype=np.float32) * float(expected_w)
        src_BL = src_BR - np.array([ux, uy], dtype=np.float32) * float(expected_w)
    else:
        src_TL = pt_v_top
        src_BL = pt_v_bot
        # compute TR and BR by moving right along top unit by expected_w
        src_TR = src_TL + np.array([ux, uy], dtype=np.float32) * float(expected_w)
        src_BR = src_BL + np.array([ux, uy], dtype=np.float32) * float(expected_w)

    # Optional: enforce vertical height by projecting (TL->BL) onto perp and rescale to expected_h
    # Compute current vertical vector and its projection onto perp (px,py)
    cur_v = src_BL - src_TL
    proj = cur_v.dot(np.array([px, py], dtype=np.float32))
    # if projection is tiny, fall back but otherwise adjust BL to ensure exact page height
    if abs(proj) > 1e-6:
        # adjust scale to exact expected_h
        scale = float(expected_h) / proj
        # recompute BL, BR as TL + perp*expected_h and TR + perp*expected_h
        src_BL = src_TL + np.array([px, py], dtype=np.float32) * float(expected_h)
        src_BR = src_TR + np.array([px, py], dtype=np.float32) * float(expected_h)
    else:
        # if proj nearly zero, we cannot rely on vertical measurement; still use perpendicular step
        src_BL = src_TL + np.array([px, py], dtype=np.float32) * float(expected_h)
        src_BR = src_TR + np.array([px, py], dtype=np.float32) * float(expected_h)

    # Build src/dst dicts
    src_corners = {
        "TL": src_TL.astype(np.float32),
        "TR": src_TR.astype(np.float32),
        "BL": src_BL.astype(np.float32),
        "BR": src_BR.astype(np.float32)
    }
    dst_corners = {
        "TL": np.array([0.0, 0.0], dtype=np.float32),
        "TR": np.array([expected_w - 1.0, 0.0], dtype=np.float32),
        "BL": np.array([0.0, expected_h - 1.0], dtype=np.float32),
        "BR": np.array([expected_w - 1.0, expected_h - 1.0], dtype=np.float32)
    }

    # choose triangle for affine depending on orientation (map TL, BL, TR -> dst TL, BL, TR)
    src_tri = np.vstack([src_corners["TL"], src_corners["BL"], src_corners["TR"]]).astype(np.float32)
    dst_tri = np.vstack([dst_corners["TL"], dst_corners["BL"], dst_corners["TR"]]).astype(np.float32)

    # validate triangle
    area = abs(0.5 * (src_tri[0,0]*(src_tri[1,1]-src_tri[2,1]) + src_tri[1,0]*(src_tri[2,1]-src_tri[0,1]) + src_tri[2,0]*(src_tri[0,1]-src_tri[1,1])))
    if area < 1.0 or not np.isfinite(src_tri).all():
        return None, src_corners, dst_corners, f"invalid_src_tri_area={area:.3f}"

    try:
        M = cv2.getAffineTransform(src_tri, dst_tri)
    except cv2.error as e:
        return None, src_corners, dst_corners, f"getAffineTransform_failed:{e}"

    # debug overlay: draw corners & axes if dbgdir provided
    if dbgdir is not None:
        vis = img.copy()
        def draw_pt(pt, color=(0,0,255), tag=""):
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 6, color, -1)
            if tag:
                cv2.putText(vis, tag, (int(pt[0]+6), int(pt[1]-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        draw_pt(src_corners["TL"], (0,255,0), "TL")
        draw_pt(src_corners["TR"], (0,128,255), "TR")
        draw_pt(src_corners["BL"], (255,0,0), "BL")
        draw_pt(src_corners["BR"], (255,255,0), "BR")
        # draw axis arrows from TL
        origin = src_corners["TL"]
        arrow_u = origin + np.array([ux, uy], dtype=np.float32) * 80.0
        arrow_p = origin + np.array([px, py], dtype=np.float32) * 80.0
        cv2.arrowedLine(vis, (int(origin[0]), int(origin[1])), (int(arrow_u[0]), int(arrow_u[1])), (255,0,255), 2)
        cv2.arrowedLine(vis, (int(origin[0]), int(origin[1])), (int(arrow_p[0]), int(arrow_p[1])), (0,255,255), 2)
        cv2.imwrite(os.path.join(dbgdir, "debug_axes_and_corners.png"), vis)

    return M, src_corners, dst_corners, "ok"

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

def split_edge_candidates(contour, bad_on_left):
    pts = contour.reshape(-1, 2)

    xs = pts[:,0]
    ys = pts[:,1]

    w = xs.max() - xs.min()
    h = ys.max() - ys.min()

    margin_x = w * 0.15
    margin_y = h * 0.15

    # outside edge
    if bad_on_left:
        outside = pts[xs > xs.max() - margin_x]
    else:
        outside = pts[xs < xs.min() + margin_x]

    # top edge
    top = pts[ys < ys.min() + margin_y]

    # bottom edge
    bottom = pts[ys > ys.max() - margin_y]

    return top, bottom, outside

def line_angle(line):
    vx, vy, _, _ = line
    return math.atan2(vy, vx)

def horizontal_line_angle(line):
    vx, vy, _, _ = line

    a = math.degrees(math.atan2(vy, vx))

    while a < -45:
        a += 180

    while a > 45:
        a -= 180

    return a

def vertical_line_angle(line):
    vx, vy, _, _ = line

    a = math.degrees(math.atan2(vy, vx))

    while a < 45:
        a += 180

    while a > 135:
        a -= 180

    return a

def normalize_angle_deg(a):
    while a < -90:
        a += 180
    while a > 90:
        a -= 180
    return a

def get_gray_mask_contours(img, dbgdir):
    # --- threshold for geometry only ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask_init, thr, hp = percentile_threshold(gray)
    if cv2.mean(gray)[0] < 127:
        mask_init = cv2.bitwise_not(mask_init)
    if DEBUG:
        save_dbg(gray, os.path.join(dbgdir, "01_gray.png"))
        save_dbg(mask_init, os.path.join(dbgdir, f"02_thresh_thr{thr}_hp{hp}.png"))

    # without a9dc2b24e6b0d1d49b6fc232223d6431ba3442a5 bad: fix perspective transform for broken ADF scanners
    # Invert if necessary, so the page is always "white" for thresholding
    # We'll use Otsu's method to detect high-contrast area
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Determine whether the page is darker or lighter than background
    mean_val = cv2.mean(gray, mask=None)[0]
    if mean_val < 127:  # dark page: invert mask
        mask = cv2.bitwise_not(mask)

    # Morphology to remove small gaps
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return gray, mask, contours

def repair_binding(img, bad_on_left, width=50):

    h,w = img.shape[:2]

    if bad_on_left:
        sample_range = range(width)
        fill_range = range(width)
        sample_x = width
    else:
        sample_range = range(w-width,w)
        fill_range = range(w-width,w)
        sample_x = w-width

    for y in range(h):

        if bad_on_left:
            sample = img[y, width:width+20]
            color = np.mean(sample,axis=0)
            img[y,:width] = color

        else:
            sample = img[y,w-width-20:w-width]
            color = np.mean(sample,axis=0)
            img[y,w-width:] = color

    return img

def process_image(in_path, out_path):
    """
    Robust page extraction that handles missing top (or bottom) edges.
    - preserves original colors (warps the original image)
    - builds orthogonal axes (no shear)
    - if top is missing, uses bottom + good vertical to infer top by moving up expected_h
    - fills outside the filled page polygon with pure white
    - extensive debug output in OUTPUT_DIR/debug/<page_num>/
    """
    # ---------- small helpers ----------
    def ensure_dir(p):
        os.makedirs(p, exist_ok=True)

    def save_dbg(img, path):
        ensure_dir(os.path.dirname(path))
        cv2.imwrite(path, img)

    def percentile_threshold(gray):
        high_p = np.percentile(gray, THRESH_HIGH_PERCENTILE)
        thr = max(THRESH_MIN, int(high_p * 0.95))
        _, mask = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
        return mask, thr, int(high_p)

    def keep_largest_component(mask):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels <= 1:
            return mask
        areas = stats[1:, cv2.CC_STAT_AREA]
        best = 1 + int(np.argmax(areas))
        out = np.zeros_like(mask)
        out[labels == best] = 255
        return out

    def detect_vertical_streaks(mask, approx_width=3, length_thresh_ratio=0.15):
        h, w = mask.shape
        kx = approx_width
        ky = max(15, int(h * 0.02))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
        long_vertical = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(long_vertical, connectivity=8)
        streak_mask = np.zeros_like(mask)
        length_thresh = max(10, int(h * length_thresh_ratio))
        for i in range(1, num_labels):
            x, y, ww, hh, area = stats[i]
            if hh >= length_thresh and ww <= max(5, int(w * 0.01)):
                streak_mask[labels == i] = 255
        return streak_mask

    def contour_to_pts(c):
        return c.reshape(-1, 2)

    def fit_line_ransac(pts, iterations=RANSAC_ITER, inlier_dist=RANSAC_INLIER_DIST, min_inliers=RANSAC_MIN_INLIERS):
        if len(pts) < 2:
            raise ValueError("Not enough points for line fit")
        best_inliers = None
        best_cnt = 0
        best_model = None
        ptsf = pts.astype(np.float32)
        n = len(ptsf)
        for _ in range(iterations):
            i1, i2 = random.sample(range(n), 2)
            p1 = ptsf[i1]; p2 = ptsf[i2]
            vx = float(p2[0] - p1[0]); vy = float(p2[1] - p1[1])
            if abs(vx) < 1e-6 and abs(vy) < 1e-6:
                continue
            dists = np.abs(vy*(ptsf[:,0]-p1[0]) - vx*(ptsf[:,1]-p1[1])) / (math.hypot(vx, vy) + 1e-12)
            inliers = dists <= inlier_dist
            cnt = int(inliers.sum())
            if cnt >= min_inliers and cnt > best_cnt:
                best_cnt = cnt
                best_inliers = inliers.copy()
                best_model = (vx, vy, float(p1[0]), float(p1[1]))
        if best_model is None:
            vx, vy, x0, y0 = cv2.fitLine(ptsf, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
            inlier_mask = np.ones(len(ptsf), dtype=bool)
            return float(vx), float(vy), float(x0), float(y0), inlier_mask
        inlier_pts = ptsf[best_inliers]
        vx, vy, x0, y0 = cv2.fitLine(inlier_pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        return float(vx), float(vy), float(x0), float(y0), best_inliers

    def intersect_lines(l1, l2):
        vx1, vy1, x1, y1 = l1
        vx2, vy2, x2, y2 = l2
        A = np.array([[vx1, -vx2], [vy1, -vy2]], dtype=np.float32)
        b = np.array([x2 - x1, y2 - y1], dtype=np.float32)
        det = np.linalg.det(A)
        if abs(det) < 1e-8:
            return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        t1, t2 = np.linalg.solve(A, b)
        xi = x1 + t1 * vx1
        yi = y1 + t1 * vy1
        return float(xi), float(yi)

    # ---------- start ----------
    fname = os.path.basename(in_path)
    m = re.match(r"^(\d+)", fname)
    page_num = int(m.group(1)) if m else 0
    bad_on_left = (page_num % 2 == 1)

    img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print("Failed to read", in_path); return
    input_is_grayscale = (len(img.shape) == 2)
    if len(img.shape) == 2:
        # OpenCV expects 3-channel images for many operations
        # later convert back to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    H_img, W_img = img.shape[:2]

    # fixed
    # # FIXME wrong H_img?
    # expected_w = int(round(ASPECT * H_img))
    # expected_h = H_img

    dbgdir = os.path.join(OUTPUT_DIR, "debug", f"{page_num:03d}")
    if DEBUG: ensure_dir(dbgdir)

    gray, mask, contours = get_gray_mask_contours(img, dbgdir)

    if not contours:
        print(f"Warning: no contours found in {in_path}")
        return

    page_contour = max(contours, key=cv2.contourArea)

    if config.use_three_edge_deskew:

        top_pts, bottom_pts, outside_pts = split_edge_candidates(
            page_contour,
            bad_on_left
        )

        top_line = fit_line_ransac(top_pts)[:4]
        bottom_line = fit_line_ransac(bottom_pts)[:4]
        outside_line = fit_line_ransac(outside_pts)[:4]

        # old
        # top_angle = math.degrees(line_angle(top_line))
        # bottom_angle = math.degrees(line_angle(bottom_line))
        # outside_angle = math.degrees(line_angle(outside_line))

        # new
        top_angle = horizontal_line_angle(top_line)
        bottom_angle = horizontal_line_angle(bottom_line)
        outside_angle = vertical_line_angle(outside_line)

        # start debug prints
        print()
        print(f"line 570: page_num={page_num}")

        rotation_error = -1 * (
            -top_angle
            -bottom_angle
        ) / 2.0

        Mrot = cv2.getRotationMatrix2D(
            (W_img/2, H_img/2),
            rotation_error,
            1.0
        )

        print(
            f"line 575: before rotation: "
            f"top_angle={top_angle:.3f} "
            f"bottom_angle={bottom_angle:.3f} "
            f"outside_angle={outside_angle:.3f}"
        )

        rotated = cv2.warpAffine(
            img,
            Mrot,
            (W_img, H_img),
            borderValue=(255,255,255)
        )

        Hr, Wr = rotated.shape[:2]
        print("line 580: rotated size", Wr, Hr)

        # img = rotated # ?

        # re-detect lines in the rotated image

        # gray, mask, contours = get_gray_mask_contours(img, dbgdir)
        gray, mask, contours = get_gray_mask_contours(rotated, dbgdir)

        top_angle2 = horizontal_line_angle(top_line)
        bottom_angle2 = horizontal_line_angle(bottom_line)
        outside_angle2 = vertical_line_angle(outside_line)

        print(
            f"line 620: after rotation and re-fitting: "
            f"top_angle2={top_angle2:.3f} "
            f"bottom_angle2={bottom_angle2:.3f} "
            f"outside_angle2={outside_angle2:.3f}"
        )

        if not contours:
            print(f"line 630: Warning: no contours found in {in_path}")
            return

        page_contour = max(contours, key=cv2.contourArea)

        top_pts, bottom_pts, outside_pts = split_edge_candidates(
            page_contour,
            bad_on_left
        )

        top_line = fit_line_ransac(top_pts)[:4]
        bottom_line = fit_line_ransac(bottom_pts)[:4]
        outside_line = fit_line_ransac(outside_pts)[:4]

        # fixed
        # # FIXME H_img is wrong
        # expected_h = H_img
        # expected_w = int(round(ASPECT * expected_h))

        vx, vy, x0, y0 = outside_line

        if 0:
            old
            if bad_on_left:
                x_out = x0
                x0_page = x_out
                x1_page = x_out + expected_w
            else:
                x_out = x0
                x1_page = x_out
                x0_page = x_out - expected_w
        else:
            # new
            outside_top = intersect_lines(
                outside_line,
                top_line
            )

            outside_bottom = intersect_lines(
                outside_line,
                bottom_line
            )

            if 0:
                # approximated page height
                page_height = math.dist(
                    outside_top,
                    outside_bottom
                )
            else:
                # perpendicular page height
                page_height = math.hypot(
                    outside_bottom[0] - outside_top[0],
                    outside_bottom[1] - outside_top[1]
                )

            expected_h = int(round(page_height))

            # ASPECT = x / y
            # x = y * ASPECT
            # expected_w = int(round(expected_h * ASPECT))
            expected_w = int(round(page_height * ASPECT))

            x_out = (
                outside_top[0]
                +
                outside_bottom[0]
            ) / 2

            x_out = int(round(x_out))

            # if bad_on_left:
            #     x0_page = x_out
            #     x1_page = x_out + expected_w
            # else:
            #     x0_page = x_out - expected_w
            #     x1_page = x_out

            if bad_on_left:
                # outside edge is RIGHT
                x1_page = x_out
                x0_page = x_out - expected_w
            else:
                # outside edge is LEFT
                x0_page = x_out
                x1_page = x_out + expected_w

            # clamp
            # x0_page = max(0, x0_page)
            # x1_page = min(rotated.shape[1], x1_page)

            y_top = int(round(outside_top[1]))
            y_bottom = int(round(outside_bottom[1]))

            src_x0 = max(0, x0_page)
            src_x1 = min(rotated.shape[1], x1_page)

            src_y0 = max(0, y_top)
            src_y1 = min(rotated.shape[0], y_top + expected_h)

            dst_x0 = src_x0 - x0_page
            dst_y0 = src_y0 - y_top

            print(
                "line 640:",
                f"outside_top={outside_top}",
                f"outside_bottom={outside_bottom}",
            )

        # crop = rotated[
        #     y_top:y_top+expected_h,
        #     x0_page:x1_page
        # ]

        full_crop = np.ones(
            (expected_h, expected_w, 3),
            dtype=rotated.dtype
        ) * 255

        full_crop[
            dst_y0:dst_y0 + (src_y1-src_y0),
            dst_x0:dst_x0 + (src_x1-src_x0)
        ] = rotated[
            src_y0:src_y1,
            src_x0:src_x1
        ]

        # FIXME fill empty area near binding edge
        # currently it is all white
        # but it should copy the vertical pattern near the binding edge

        crop = full_crop

        print(
            "line 650: crop",
            f"x0_page={x0_page}",
            f"x1_page={x1_page}",
            f"y_top={y_top}",
            f"y_bottom={y_bottom}",
            f"src_x0={src_x0}",
            f"src_x1={src_x1}",
            f"src_y0={src_y0}",
            f"src_y1={src_y1}",
            f"dst_x0={dst_x0}",
            f"dst_y0={dst_y0}",
            f"expected_w={expected_w}",
            f"expected_h={expected_h}",
        )

        # crop = rotated[
        #     0:expected_h,
        #     int(x0_page):int(x1_page)
        # ]

        if 1:
            # debug
            vis = rotated.copy()
            cv2.rectangle(
                vis,
                (int(x0_page), int(y_top)),
                (int(x1_page), int(y_bottom)),
                (0,0,255),
                3,
            )
            name = f"{page_num:03d}.crop_debug_rotated_line730.jpg"
            print(f"writing {name}")
            cv2.imwrite(name, vis)

        # y_top = round(outside_top[1])
        # y_bottom = round(outside_bottom[1])
        # actual_height = y_bottom - y_top
        # crop = rotated[
        #     y_top:y_top + expected_h,
        #     x0_page:x1_page
        # ]

        # y_top = int(round(outside_top[1]))
        # y_bottom = int(round(outside_bottom[1]))
        # crop = rotated[
        #     y_top:y_bottom,
        #     x0_page:x1_page
        # ]

        # img = repair_binding(img, bad_on_left, width=50)
        if 1:
            # rotated = repair_binding(rotated, bad_on_left, width=50)
            # Hr, Wr = rotated.shape[:2]
            # print("line 660: rotated size", Wr, Hr)
            # warped = rotated
            crop = repair_binding(crop, bad_on_left, width=50)
            Hr, Wr = crop.shape[:2]
            print("line 660: crop size", Wr, Hr)
            warped = crop
        else:
            crop = repair_binding(crop, bad_on_left, width=50)
            warped = crop

        print(
            "line 760: after crop: crop actual:",
            f"crop.shape[1]={crop.shape[1]}",
            f"expected_w={expected_w}",
        )

    else:
        # config.use_three_edge_deskew == False
        # FIXME use only two page edges: outside, bottom
        # Approximate contour to quadrilateral
        epsilon = 0.02 * cv2.arcLength(page_contour, True)
        approx = cv2.approxPolyDP(page_contour, epsilon, True)
        if len(approx) != 4:
            approx = cv2.convexHull(page_contour)
            if len(approx) < 4:
                print(f"Warning: not enough points for perspective in {in_path}")
                return
            # pick 4 extreme points
            pts = np.array([
                approx[approx[:,0,0].argmin()][0],  # leftmost
                approx[approx[:,0,1].argmin()][0],  # topmost
                approx[approx[:,0,0].argmax()][0],  # rightmost
                approx[approx[:,0,1].argmax()][0]   # bottommost
            ])
        else:
            pts = approx.reshape(4,2)

        rect = order_points(pts)

        # Perspective transform
        widthA = np.linalg.norm(rect[2] - rect[3])
        widthB = np.linalg.norm(rect[1] - rect[0])
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.linalg.norm(rect[1] - rect[2])
        heightB = np.linalg.norm(rect[0] - rect[3])
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    # # Add internal white border
    # h, w = warped.shape[:2]
    # canvas = np.ones_like(warped) * 255
    # b = BORDER_SIZE
    # canvas[b:h-b, b:w-b] = warped[b:h-b, b:w-b]

    h, w = warped.shape[:2]
    b = BORDER_SIZE

    # Prepare a canvas with the same content as warped
    canvas = warped.copy()

    # Function to compute average color along a strip
    def avg_color_strip(img, axis, start, end, strip_width=1):
        if axis == 'top':
            strip = img[start:end, :, :]
        elif axis == 'bottom':
            strip = img[h-end:h-start, :, :]
        elif axis == 'left':
            strip = img[:, start:end, :]
        elif axis == 'right':
            strip = img[:, w-end:w-start, :]
        else:
            raise ValueError("Invalid axis")
        return np.mean(strip, axis=(0,1)).astype(np.uint8)

    # FIXME preserve patterns near edges
    if 0:
        # Fill borders with local average color
        canvas[0:b, :, :] = avg_color_strip(canvas, 'top', 50, 100)       # top border
        canvas[h-b:h, :, :] = avg_color_strip(canvas, 'bottom', 50, 100)  # bottom border
        canvas[:, 0:b, :] = avg_color_strip(canvas, 'left', 50, 100)      # left border
        canvas[:, w-b:w, :] = avg_color_strip(canvas, 'right', 50, 100)   # right border

    if input_is_grayscale:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

    ensure_dir(os.path.dirname(out_path))
    cv2.imwrite(out_path, canvas)
    print(f"writing {out_path}")

def main():
    ensure_dir(OUTPUT_DIR)
    # TODO use image_format from 030-measure-page-size.txt
    image_format = "jpg"
    files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(f".{image_format}")])
    if not files:
        print("No image files found in", INPUT_DIR)
        return
    for fname in files:
        m = re.match(r"^(\d+)", fname)
        # page_num = int(m.group(1)) if m else 0
        page_num = int(m.group(1)) # can throw
        # if page_num != 14: continue # debug
        # if not page_num in [340, 345]: continue # debug
        # if page_num < 300: continue # debug
        # if page_num != 320: continue # debug
        in_path = os.path.join(INPUT_DIR, fname)
        out_path = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(out_path):
            continue
        try:
            process_image(in_path, out_path)
        except Exception as exc:
            print(f"Error processing {fname}: {exc}")
            raise

if __name__ == "__main__":
    main()
```

debug prints

    line 570: page_num=1
    line 575: before rotation: top_angle=-0.411 bottom_angle=-0.389 outside_angle=89.769
    line 580: rotated size 3478 5078
    line 620: after rotation and re-fitting: top_angle2=-0.411 bottom_angle2=-0.389 outside_angle2=89.769
    line 640: outside_top=(3454.768310546875, 69.978271484375) outside_bottom=(3440.1533203125, 5030.9912109375)
    line 650: crop x0_page=-238 x1_page=3447 y_top=70 y_bottom=5031 src_x0=0 src_x1=3447 src_y0=70 src_y1=5031 dst_x0=238 dst_y0=0 expected_w=3685 expected_h=4961
    writing 001.crop_debug_rotated_line730.jpg
    line 660: crop size 3685 4961
    line 760: after crop: crop actual: crop.shape[1]=3685 expected_w=3685
    writing 065-remove-page-borders/001.jpg

    line 570: page_num=2
    line 575: before rotation: top_angle=0.382 bottom_angle=0.387 outside_angle=90.263
    line 580: rotated size 3470 5078
    line 620: after rotation and re-fitting: top_angle2=0.382 bottom_angle2=0.387 outside_angle2=90.263
    line 640: outside_top=(31.2772216796875, 77.547119140625) outside_bottom=(40.665740966796875, 4485.56884765625)
    line 650: crop x0_page=36 x1_page=3311 y_top=78 y_bottom=4486 src_x0=36 src_x1=3311 src_y0=78 src_y1=4486 dst_x0=0 dst_y0=0 expected_w=3275 expected_h=4408
    writing 002.crop_debug_rotated_line730.jpg
    line 660: crop size 3275 4408
    line 760: after crop: crop actual: crop.shape[1]=3275 expected_w=3275
    writing 065-remove-page-borders/002.jpg

## Response:

7/12/2026, 4:34:28 PM

The new debug is actually very useful. The `expected_w` calculation is now internally consistent:

    page 1:
    expected_h = 4961
    expected_w = 3685
    3685 / 4961 = 0.743

    page 2:
    expected_h = 4408
    expected_w = 3275
    3275 / 4408 = 0.743

So the aspect calculation is not the problem anymore.

The suspicious part is this:

    page 1:
    outside_top=(3454.8, 70.0)
    outside_bottom=(3440.2, 5031.0)

    x0_page=-238
    x1_page=3447

    src_x0=0
    src_x1=3447

    dst_x0=238

This means:

- the detected outside edge is at the **right**
- the desired page rectangle starts 238 pixels outside the scan
- you create a 3685 pixel wide canvas
- only 3447 pixels are copied from the image
- the first 238 pixels are artificial fill

So the output width is mathematically correct, but the coordinate system is wrong for the missing binding edge.

The problem is here:

``` python
x_out = (
    outside_top[0]
    +
    outside_bottom[0]
) / 2
```

You reduce the outside edge to a single x coordinate. But your page edge is not vertical:

    outside_top.x    = 3454.8
    outside_bottom.x = 3440.2

The edge leans by ~15 pixels over the page height.

More importantly, after you do that:

``` python
x0_page = x_out - expected_w
x1_page = x_out
```

you are inventing a rectangular page from the **center** of the outside edge. That shifts the whole page by half the slant error and makes the binding-side reconstruction look stretched.

------------------------------------------------------------------------

I would change the logic to keep the actual detected edge.

Replace:

``` python
x_out = (
    outside_top[0]
    +
    outside_bottom[0]
) / 2

x_out = int(round(x_out))

if bad_on_left:
    x1_page = x_out
    x0_page = x_out - expected_w
else:
    x0_page = x_out
    x1_page = x_out + expected_w
```

with something like:

``` python
if bad_on_left:
    # outside edge is the RIGHT edge
    x1_top = outside_top[0]
    x1_bottom = outside_bottom[0]

    x0_top = x1_top - expected_w
    x0_bottom = x1_bottom - expected_w

else:
    # outside edge is the LEFT edge
    x0_top = outside_top[0]
    x0_bottom = outside_bottom[0]

    x1_top = x0_top + expected_w
    x1_bottom = x0_bottom + expected_w
```

Then do not crop as a rectangle. Use the quadrilateral:

``` python
src = np.float32([
    [x0_top,    outside_top[1]],
    [x1_top,    outside_top[1]],
    [x1_bottom, outside_bottom[1]],
    [x0_bottom, outside_bottom[1]],
])

dst = np.float32([
    [0,0],
    [expected_w-1,0],
    [expected_w-1,expected_h-1],
    [0,expected_h-1],
])

M = cv2.getPerspectiveTransform(src, dst)

crop = cv2.warpPerspective(
    rotated,
    M,
    (expected_w, expected_h),
    borderValue=(255,255,255)
)
```

This keeps:

- the real outside edge
- the correct page aspect ratio
- the missing binding width
- the small skew of the scanner

and avoids the "expanded binding edge" artifact.

Also, your debug rectangle currently hides this problem because it draws:

``` python
cv2.rectangle(
    vis,
    (int(x0_page), int(y_top)),
    (int(x1_page), int(y_bottom)),
)
```

but your actual page is a parallelogram. I would draw the four corners instead:

``` python
pts = np.array([
    [x0_top, outside_top[1]],
    [x1_top, outside_top[1]],
    [x1_bottom, outside_bottom[1]],
    [x0_bottom, outside_bottom[1]],
], np.int32)

cv2.polylines(vis, [pts], True, (0,0,255), 3)
```

I think this is the remaining geometric issue. The expected size calculation is now correct; the rectangle reconstruction from `x_out` is what is causing the binding side distortion.

## Prompt:

7/12/2026, 5:10:29 PM

ok, the cropping code seems to be correct, but i still get too much width at the binding edge...

i have checked the aspect of the output image, the aspect is correct

one problem seems to be that my scanner distorts the height of scanned images! i have measured the aspect ratio of a rectangle on a scanned page, and the scanned version has an aspect ratio of 0.8 while the original version has an aspect ratio of 0.825

all book pages have the same height, and our edge detector should reliably detect the top and bottom edges, so we have an exact scanned page height, and we know the exact original page height from config.rotated_scan_y and we know the scan resolution from config.scan_resolution (usually this is 600 dpi).... from these values, can we calculate a stretch factor to fix the scanned page height?

## Response:

7/12/2026, 5:10:29 PM

Yes. In fact, this is probably the right way to fix it. What you are seeing is not a crop-width problem anymore; it is a **non-uniform scanner scaling problem**.

Your observation:

- original rectangle aspect: `0.825`
- scanned rectangle aspect: `0.800`

means the scanner has compressed the **height direction** (or equivalently stretched width relative to height).

Since:

- the page height in the scan is measurable from the detected top/bottom edges
- the physical page height is fixed
- DPI is known

you can derive a vertical correction factor.

### Derivation

Let:

- `H_scan_px` = detected scanned page height in pixels
- `H_real_mm` = physical page height
- `DPI` = scan resolution

The expected pixel height at that DPI is:

``` math
H_{expected} = H_{real\_mm} \times \frac{DPI}{25.4}
```

Then the vertical scale correction is:

``` math
S_y = \frac{H_{expected}}{H_{scan\_px}}
```

If:

    S_y > 1

the scanner compressed the height and you need to stretch vertically.

If:

    S_y < 1

the scanner stretched the height and you need to compress.

------------------------------------------------------------------------

### But you already have the best measurement

You mentioned:

> we know the exact original page height from config.rotated_scan_y

If `config.rotated_scan_y` is already the intended pixel height at the scan resolution, then you don't even need physical units:

``` python
stretch_y = config.rotated_scan_y / detected_page_height
```

Example:

    detected_page_height = 4961
    config.rotated_scan_y = 5120

    stretch_y = 5120 / 4961
              = 1.032

Meaning:

"stretch the scanned page vertically by 3.2%"

------------------------------------------------------------------------

### Where to apply it

Do **not** modify `expected_w`.

Your geometry is currently:

    detected page
          |
          v
    crop to expected_w x expected_h
          |
          v
    vertical correction

So after your crop:

``` python
crop = cv2.resize(
    crop,
    (
        crop.shape[1],
        config.rotated_scan_y
    ),
    interpolation=cv2.INTER_CUBIC
)
```

or more generally:

``` python
new_h = int(round(crop.shape[0] * stretch_y))

crop = cv2.resize(
    crop,
    (crop.shape[1], new_h),
    interpolation=cv2.INTER_CUBIC
)
```

------------------------------------------------------------------------

### However, there is one subtle point

Your current `expected_w` comes from:

``` python
expected_w = expected_h * ASPECT
```

where `expected_h` is the **distorted scanner height**.

That means your width is also being computed from the distorted measurement.

Instead:

1.  Detect scanned height:

``` python
detected_height = distance(top_edge, bottom_edge)
```

2.  Correct height:

``` python
corrected_height = config.rotated_scan_y
```

3.  Calculate width from the corrected height:

``` python
corrected_width = int(round(corrected_height * ASPECT))
```

4.  Warp directly to:

<!-- -->

    corrected_width x corrected_height

That will remove the binding-edge "too wide" appearance.

------------------------------------------------------------------------

### I would add a diagnostic print

Before warping:

``` python
print(
    "scanner scale:",
    "detected_h=", detected_height,
    "target_h=", config.rotated_scan_y,
    "stretch_y=", config.rotated_scan_y / detected_height
)
```

For your example I would expect something around:

    scanner scale:
    detected_h=4961
    target_h=5120
    stretch_y=1.032

The nice thing is that this correction should be extremely stable across the whole book because your scanner distortion is mechanical, not page-dependent. You can probably measure it once and even store it in config.

## Prompt:

7/12/2026, 5:14:13 PM

> If `config.rotated_scan_y` is already the intended pixel height at the scan resolution, then you don't even need physical units:

no, config.rotated_scan_y is a mm value. scan_x and scan_y are passed to scanimage like `scanimage -x $scan_x -y $scan_y`

## Response:

7/12/2026, 5:14:13 PM

Ah, yes — that changes the calculation. I incorrectly treated `config.rotated_scan_y` as a pixel dimension. Since `scan_x` / `scan_y` are the physical scan area passed to `scanimage` (mm), you need to go through the DPI conversion.

Your situation is actually better defined:

- The scanner was commanded to scan a physical rectangle:
  - width = `config.rotated_scan_x` mm
  - height = `config.rotated_scan_y` mm
- The scanner output has a pixel height that corresponds to that physical height **but is geometrically distorted**.
- Your edge detector gives you the **actual scanned pixel height** of the page:
  - `detected_height_px = distance(outside_top, outside_bottom)`
- You know the intended pixel height from the physical size and DPI.

The intended pixel dimensions are:

``` math
pixels = mm \times \frac{dpi}{25.4}
```

so:

``` python
expected_scan_h_px = config.rotated_scan_y * config.scan_resolution / 25.4
expected_scan_w_px = config.rotated_scan_x * config.scan_resolution / 25.4
```

Then the vertical correction factor is:

``` python
stretch_y = expected_scan_h_px / detected_height_px
```

For your example:

- measured scanned rectangle aspect: `0.8`
- original aspect: `0.825`

That means the scanner made the page too tall (or too narrow depending on which ratio you measured). The correction should be close to:

    0.825 / 0.8 = 1.03125

so about a **3.1% vertical stretch**.

The important part: apply this correction **before calculating the final page width**, otherwise your aspect correction is compensating for the scanner distortion.

Your current flow is:

    detect edges
        |
        v
    measure page height
        |
        v
    expected_w = height * ASPECT
        |
        v
    crop

The problem is that `height` is the distorted scanner height.

Instead:

    detect edges
        |
        v
    measure scanned page height
        |
        v
    stretch image vertically to correct physical scale
        |
        v
    detect/calculate page width

or equivalently, if you don't want another resampling step:

``` python
detected_h = math.dist(outside_top, outside_bottom)

target_h = config.rotated_scan_y * config.scan_resolution / 25.4

scale_y = target_h / detected_h
```

Then:

``` python
expected_h = round(detected_h * scale_y)
expected_w = round(expected_h * ASPECT)
```

and the affine crop should use the corrected coordinate system.

However, there is a subtle issue in your current code: you are filling a rectangular crop:

``` python
full_crop = np.ones(
    (expected_h, expected_w, 3),
    dtype=rotated.dtype
)
```

This assumes that the page height distortion is already fixed. It is not. The binding-edge expansion you see is probably because the missing binding edge is being reconstructed in the wrong coordinate system.

I would add one explicit vertical normalization step immediately after rotation:

``` python
detected_h = math.dist(outside_top, outside_bottom)

target_h = (
    config.rotated_scan_y
    * config.scan_resolution
    / 25.4
)

scale_y = target_h / detected_h

rotated = cv2.resize(
    rotated,
    None,
    fx=1.0,
    fy=scale_y,
    interpolation=cv2.INTER_CUBIC
)
```

Then rerun the edge detection on `rotated`.

After this:

- top/bottom edge distance should match the physical scan size
- `ASPECT` becomes valid again
- `expected_w = expected_h * ASPECT` should no longer create a stretched binding side

One more thing: your debug values show page 1:

    outside_top=(3454.8,69.9)
    outside_bottom=(3440.1,5031.0)

    height = 4961 px

but your scan output is:

    5078 px high

so the detected page is already ~2.3% shorter than the full scan. That is very close to the distortion magnitude you measured. The missing 117 px is not just border loss — it is likely the geometric scaling problem.

The physical-DPI correction is the right next step.

## Prompt:

7/12/2026, 6:53:16 PM

hmm. i tried some more things, but it seems like this problem is not solvable for individual files, so for now i want to disable the "expand the binding edge to expected_w" code in my 065-remove-page-borders.py script, and later run a separate script to get the average height of all scans and then use that average height to normalize all page heights (all pages of a book have the same height, the scanner crops only the binding side) and then from that normalized page height try to expand the binding edge to expected_w....

now, how can i disable the "expand the binding edge to expected_w" code in my script?

``` py
#!/usr/bin/env python3
"""
extract scanned page from gray background

restore the binding edge of the page
by filling the missing width
with the average color near the binding edge
"""

INPUT_DIR = "060-rotate-crop"
OUTPUT_DIR = "065-remove-page-borders"

# === Tuning parameters ===
DEBUG = True
DEBUG = False
BORDER_SIZE = 100  # pixels

RANSAC_ITER = 400
RANSAC_INLIER_DIST = 6.0
# RANSAC_MIN_INLIERS = 30
RANSAC_MIN_INLIERS = 20

THRESH_HIGH_PERCENTILE = 99
THRESH_MIN = 200

import os
import re
import math
import random
import numpy as np
import cv2

from _shared import (
    load_config,
    get_page_num,
)

config = load_config()

# no, this is wrong if (config.do_rotate == True)
# scan_x = config.scan_x
# scan_y = config.scan_y
# config.scan_aspect = scan_x / scan_y
# ASPECT = config.scan_aspect

ASPECT = config.rotated_scan_aspect

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def save_dbg(img, path):
    ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, img)

def percentile_threshold(gray):
    high_p = np.percentile(gray, THRESH_HIGH_PERCENTILE)
    thr = max(THRESH_MIN, int(high_p * 0.95))
    _, mask = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
    return mask, thr, int(high_p)

def detect_vertical_streaks(mask, approx_width=3, length_thresh_ratio=0.15):
    h, w = mask.shape
    kx = approx_width
    ky = max(15, int(h * 0.02))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
    long_vertical = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(long_vertical, connectivity=8)
    streak_mask = np.zeros_like(mask)
    length_thresh = max(10, int(h * length_thresh_ratio))
    for i in range(1, num_labels):
        x, y, ww, hh, area = stats[i]
        if hh >= length_thresh and ww <= max(5, int(w * 0.01)):
            streak_mask[labels == i] = 255
    return streak_mask

def keep_largest_component(mask):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    out = np.zeros_like(mask)
    out[labels == best] = 255
    return out

def contour_to_pts(contour):
    return contour.reshape(-1,2)

def fit_line_ransac(pts, iterations=RANSAC_ITER, inlier_dist=RANSAC_INLIER_DIST, min_inliers=RANSAC_MIN_INLIERS):
    if len(pts) < 2:
        raise ValueError("Not enough points")
    best_inliers = None
    best_model = None
    n = len(pts)
    ptsf = pts.astype(np.float32)
    for _ in range(iterations):
        i1, i2 = random.sample(range(n), 2)
        p1 = ptsf[i1]; p2 = ptsf[i2]
        vx = float(p2[0] - p1[0]); vy = float(p2[1] - p1[1])
        if vx == 0 and vy == 0:
            continue
        dists = np.abs(vy*(ptsf[:,0]-p1[0]) - vx*(ptsf[:,1]-p1[1])) / (math.hypot(vx, vy) + 1e-12)
        inliers = dists <= inlier_dist
        cnt = int(inliers.sum())
        if cnt >= min_inliers and (best_inliers is None or cnt > int(best_inliers.sum())):
            best_inliers = inliers.copy()
            best_model = (vx, vy, float(p1[0]), float(p1[1]))
    if best_model is None:
        vx, vy, x0, y0 = cv2.fitLine(ptsf, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        inlier_mask = np.ones(len(pts), dtype=bool)
        return float(vx), float(vy), float(x0), float(y0), inlier_mask
    inlier_pts = ptsf[best_inliers]
    vx, vy, x0, y0 = cv2.fitLine(inlier_pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    inlier_mask = best_inliers
    return float(vx), float(vy), float(x0), float(y0), inlier_mask

def intersect_lines(l1, l2):
    vx1, vy1, x1, y1 = l1
    vx2, vy2, x2, y2 = l2
    A = np.array([[vx1, -vx2], [vy1, -vy2]], dtype=np.float32)
    b = np.array([x2 - x1, y2 - y1], dtype=np.float32)
    det = np.linalg.det(A)
    if abs(det) < 1e-8:
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    t1, t2 = np.linalg.solve(A, b)
    xi = x1 + t1 * vx1
    yi = y1 + t1 * vy1
    return float(xi), float(yi)

def build_affine_without_shear(pt_v_top, pt_v_bot, expected_w, expected_h, bad_on_left, img, dbgdir=None):
    """
    Build a source triangle that enforces orthogonal page axes (no shear) using:
      - pt_v_top: intersection of top line and good vertical (top-right when good vertical is right)
      - pt_v_bot: intersection of bottom line and good vertical (bottom-right when good vertical is right)
    Returns (M_aff, src_corners_dict, dst_corners_dict, reason)
    - M_aff: 2x3 affine matrix or None on failure
    - src_corners_dict: dict of TL, TR, BL, BR (np.float32 2-vectors)
    - dst_corners_dict: dict of TL, TR, BL, BR in destination coordinates
    - reason: diagnostic string
    """
    # convert to vectors
    pt_v_top = np.array(pt_v_top, dtype=np.float32)
    pt_v_bot = np.array(pt_v_bot, dtype=np.float32)

    # compute direction along top edge: vector from bottom intersection to top intersection projected horizontally
    # but we don't have explicit top_line direction vector here; we will approximate using (pt_v_top - pt_v_bot) rotated?
    # Better: caller should supply top_line direction (vx,vy). If you don't have it, approximate from nearby contour points.
    # For compatibility with your code, expect you have top_vx, top_vy; if not, fall back to vector between two top contour points.
    # Here we'll assume top_vx, top_vy are available in outer scope; otherwise compute from pt_v_top->pt_v_bot displacement projected perpendicular:
    # To keep this helper self-contained, we compute top_dir as average of (pt_v_top_to_some_right) - but simplest robust approach:
    # compute approximate top direction by taking small offset vector along top by sampling image gradient: fallback to (1,0).

    # --- Here caller should provide top_unit; we'll compute a safe top_unit using neighbor pixels if available ---
    # We'll attempt to derive a top_unit by sampling a short step along the image: use vector from pt_v_top to pt_v_top projected to image center
    cx, cy = img.shape[1] / 2.0, img.shape[0] / 2.0
    # prefer vector pointing rightwards by projecting displacement to center
    approx = pt_v_top - np.array([cx, cy], dtype=np.float32)
    if np.linalg.norm(approx) < 1e-6:
        approx = np.array([1.0, 0.0], dtype=np.float32)
    ux, uy = approx / (np.linalg.norm(approx) + 1e-12)

    # Force ux positive (pointing to the right) — we want u to be rightward
    if ux < 0:
        ux, uy = -ux, -uy

    # make perpendicular downwards: p = (-uy, ux)
    px, py = -uy, ux

    # normalize again (just in case)
    n_u = math.hypot(ux, uy)
    n_p = math.hypot(px, py)
    if n_u < 1e-6 or n_p < 1e-6:
        return None, None, None, "degenerate_axes"
    ux, uy = ux / n_u, uy / n_u
    px, py = px / n_p, py / n_p

    # Now construct corners. We know pt_v_top/pt_v_bot correspond to the known vertical edge:
    # If bad_on_left is True, good vertical is the RIGHT edge => pt_v_top == TR, pt_v_bot == BR.
    # If bad_on_left is False, good vertical is LEFT edge => pt_v_top == TL, pt_v_bot == BL.
    if bad_on_left:
        src_TR = pt_v_top
        src_BR = pt_v_bot
        # compute TL and BL by moving left along top unit by expected_w
        src_TL = src_TR - np.array([ux, uy], dtype=np.float32) * float(expected_w)
        src_BL = src_BR - np.array([ux, uy], dtype=np.float32) * float(expected_w)
    else:
        src_TL = pt_v_top
        src_BL = pt_v_bot
        # compute TR and BR by moving right along top unit by expected_w
        src_TR = src_TL + np.array([ux, uy], dtype=np.float32) * float(expected_w)
        src_BR = src_BL + np.array([ux, uy], dtype=np.float32) * float(expected_w)

    # Optional: enforce vertical height by projecting (TL->BL) onto perp and rescale to expected_h
    # Compute current vertical vector and its projection onto perp (px,py)
    cur_v = src_BL - src_TL
    proj = cur_v.dot(np.array([px, py], dtype=np.float32))
    # if projection is tiny, fall back but otherwise adjust BL to ensure exact page height
    if abs(proj) > 1e-6:
        # adjust scale to exact expected_h
        scale = float(expected_h) / proj
        # recompute BL, BR as TL + perp*expected_h and TR + perp*expected_h
        src_BL = src_TL + np.array([px, py], dtype=np.float32) * float(expected_h)
        src_BR = src_TR + np.array([px, py], dtype=np.float32) * float(expected_h)
    else:
        # if proj nearly zero, we cannot rely on vertical measurement; still use perpendicular step
        src_BL = src_TL + np.array([px, py], dtype=np.float32) * float(expected_h)
        src_BR = src_TR + np.array([px, py], dtype=np.float32) * float(expected_h)

    # Build src/dst dicts
    src_corners = {
        "TL": src_TL.astype(np.float32),
        "TR": src_TR.astype(np.float32),
        "BL": src_BL.astype(np.float32),
        "BR": src_BR.astype(np.float32)
    }
    dst_corners = {
        "TL": np.array([0.0, 0.0], dtype=np.float32),
        "TR": np.array([expected_w - 1.0, 0.0], dtype=np.float32),
        "BL": np.array([0.0, expected_h - 1.0], dtype=np.float32),
        "BR": np.array([expected_w - 1.0, expected_h - 1.0], dtype=np.float32)
    }

    # choose triangle for affine depending on orientation (map TL, BL, TR -> dst TL, BL, TR)
    src_tri = np.vstack([src_corners["TL"], src_corners["BL"], src_corners["TR"]]).astype(np.float32)
    dst_tri = np.vstack([dst_corners["TL"], dst_corners["BL"], dst_corners["TR"]]).astype(np.float32)

    # validate triangle
    area = abs(0.5 * (src_tri[0,0]*(src_tri[1,1]-src_tri[2,1]) + src_tri[1,0]*(src_tri[2,1]-src_tri[0,1]) + src_tri[2,0]*(src_tri[0,1]-src_tri[1,1])))
    if area < 1.0 or not np.isfinite(src_tri).all():
        return None, src_corners, dst_corners, f"invalid_src_tri_area={area:.3f}"

    try:
        M = cv2.getAffineTransform(src_tri, dst_tri)
    except cv2.error as e:
        return None, src_corners, dst_corners, f"getAffineTransform_failed:{e}"

    # debug overlay: draw corners & axes if dbgdir provided
    if dbgdir is not None:
        vis = img.copy()
        def draw_pt(pt, color=(0,0,255), tag=""):
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 6, color, -1)
            if tag:
                cv2.putText(vis, tag, (int(pt[0]+6), int(pt[1]-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        draw_pt(src_corners["TL"], (0,255,0), "TL")
        draw_pt(src_corners["TR"], (0,128,255), "TR")
        draw_pt(src_corners["BL"], (255,0,0), "BL")
        draw_pt(src_corners["BR"], (255,255,0), "BR")
        # draw axis arrows from TL
        origin = src_corners["TL"]
        arrow_u = origin + np.array([ux, uy], dtype=np.float32) * 80.0
        arrow_p = origin + np.array([px, py], dtype=np.float32) * 80.0
        cv2.arrowedLine(vis, (int(origin[0]), int(origin[1])), (int(arrow_u[0]), int(arrow_u[1])), (255,0,255), 2)
        cv2.arrowedLine(vis, (int(origin[0]), int(origin[1])), (int(arrow_p[0]), int(arrow_p[1])), (0,255,255), 2)
        cv2.imwrite(os.path.join(dbgdir, "debug_axes_and_corners.png"), vis)

    return M, src_corners, dst_corners, "ok"

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

def split_edge_candidates(contour, bad_on_left):
    pts = contour.reshape(-1, 2)

    xs = pts[:,0]
    ys = pts[:,1]

    w = xs.max() - xs.min()
    h = ys.max() - ys.min()

    margin_x = w * 0.15
    margin_y = h * 0.15

    # outside edge
    if bad_on_left:
        outside = pts[xs > xs.max() - margin_x]
    else:
        outside = pts[xs < xs.min() + margin_x]

    # top edge
    top = pts[ys < ys.min() + margin_y]

    # bottom edge
    bottom = pts[ys > ys.max() - margin_y]

    return top, bottom, outside

def line_angle(line):
    vx, vy, _, _ = line
    return math.atan2(vy, vx)

def horizontal_line_angle(line):
    vx, vy, _, _ = line

    a = math.degrees(math.atan2(vy, vx))

    while a < -45:
        a += 180

    while a > 45:
        a -= 180

    return a

def vertical_line_angle(line):
    vx, vy, _, _ = line

    a = math.degrees(math.atan2(vy, vx))

    while a < 45:
        a += 180

    while a > 135:
        a -= 180

    return a

def normalize_angle_deg(a):
    while a < -90:
        a += 180
    while a > 90:
        a -= 180
    return a

def get_gray_mask_contours(img, dbgdir):
    # --- threshold for geometry only ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask_init, thr, hp = percentile_threshold(gray)
    if cv2.mean(gray)[0] < 127:
        mask_init = cv2.bitwise_not(mask_init)
    if DEBUG:
        save_dbg(gray, os.path.join(dbgdir, "01_gray.png"))
        save_dbg(mask_init, os.path.join(dbgdir, f"02_thresh_thr{thr}_hp{hp}.png"))

    # without a9dc2b24e6b0d1d49b6fc232223d6431ba3442a5 bad: fix perspective transform for broken ADF scanners
    # Invert if necessary, so the page is always "white" for thresholding
    # We'll use Otsu's method to detect high-contrast area
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Determine whether the page is darker or lighter than background
    mean_val = cv2.mean(gray, mask=None)[0]
    if mean_val < 127:  # dark page: invert mask
        mask = cv2.bitwise_not(mask)

    # Morphology to remove small gaps
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return gray, mask, contours

def repair_binding(img, bad_on_left, width=50):

    h,w = img.shape[:2]

    if bad_on_left:
        sample_range = range(width)
        fill_range = range(width)
        sample_x = width
    else:
        sample_range = range(w-width,w)
        fill_range = range(w-width,w)
        sample_x = w-width

    for y in range(h):

        if bad_on_left:
            sample = img[y, width:width+20]
            color = np.mean(sample,axis=0)
            img[y,:width] = color

        else:
            sample = img[y,w-width-20:w-width]
            color = np.mean(sample,axis=0)
            img[y,w-width:] = color

    return img

def px_of_mm(mm, dpi):
    return mm * dpi / 25.4

def process_image(in_path, out_path):
    """
    Robust page extraction that handles missing top (or bottom) edges.
    - preserves original colors (warps the original image)
    - builds orthogonal axes (no shear)
    - if top is missing, uses bottom + good vertical to infer top by moving up expected_h
    - fills outside the filled page polygon with pure white
    - extensive debug output in OUTPUT_DIR/debug/<page_num>/
    """
    # ---------- small helpers ----------
    def ensure_dir(p):
        os.makedirs(p, exist_ok=True)

    def save_dbg(img, path):
        ensure_dir(os.path.dirname(path))
        cv2.imwrite(path, img)

    def percentile_threshold(gray):
        high_p = np.percentile(gray, THRESH_HIGH_PERCENTILE)
        thr = max(THRESH_MIN, int(high_p * 0.95))
        _, mask = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
        return mask, thr, int(high_p)

    def keep_largest_component(mask):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels <= 1:
            return mask
        areas = stats[1:, cv2.CC_STAT_AREA]
        best = 1 + int(np.argmax(areas))
        out = np.zeros_like(mask)
        out[labels == best] = 255
        return out

    def detect_vertical_streaks(mask, approx_width=3, length_thresh_ratio=0.15):
        h, w = mask.shape
        kx = approx_width
        ky = max(15, int(h * 0.02))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
        long_vertical = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(long_vertical, connectivity=8)
        streak_mask = np.zeros_like(mask)
        length_thresh = max(10, int(h * length_thresh_ratio))
        for i in range(1, num_labels):
            x, y, ww, hh, area = stats[i]
            if hh >= length_thresh and ww <= max(5, int(w * 0.01)):
                streak_mask[labels == i] = 255
        return streak_mask

    def contour_to_pts(c):
        return c.reshape(-1, 2)

    def fit_line_ransac(pts, iterations=RANSAC_ITER, inlier_dist=RANSAC_INLIER_DIST, min_inliers=RANSAC_MIN_INLIERS):
        if len(pts) < 2:
            raise ValueError("Not enough points for line fit")
        best_inliers = None
        best_cnt = 0
        best_model = None
        ptsf = pts.astype(np.float32)
        n = len(ptsf)
        for _ in range(iterations):
            i1, i2 = random.sample(range(n), 2)
            p1 = ptsf[i1]; p2 = ptsf[i2]
            vx = float(p2[0] - p1[0]); vy = float(p2[1] - p1[1])
            if abs(vx) < 1e-6 and abs(vy) < 1e-6:
                continue
            dists = np.abs(vy*(ptsf[:,0]-p1[0]) - vx*(ptsf[:,1]-p1[1])) / (math.hypot(vx, vy) + 1e-12)
            inliers = dists <= inlier_dist
            cnt = int(inliers.sum())
            if cnt >= min_inliers and cnt > best_cnt:
                best_cnt = cnt
                best_inliers = inliers.copy()
                best_model = (vx, vy, float(p1[0]), float(p1[1]))
        if best_model is None:
            vx, vy, x0, y0 = cv2.fitLine(ptsf, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
            inlier_mask = np.ones(len(ptsf), dtype=bool)
            return float(vx), float(vy), float(x0), float(y0), inlier_mask
        inlier_pts = ptsf[best_inliers]
        vx, vy, x0, y0 = cv2.fitLine(inlier_pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        return float(vx), float(vy), float(x0), float(y0), best_inliers

    def intersect_lines(l1, l2):
        vx1, vy1, x1, y1 = l1
        vx2, vy2, x2, y2 = l2
        A = np.array([[vx1, -vx2], [vy1, -vy2]], dtype=np.float32)
        b = np.array([x2 - x1, y2 - y1], dtype=np.float32)
        det = np.linalg.det(A)
        if abs(det) < 1e-8:
            return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        t1, t2 = np.linalg.solve(A, b)
        xi = x1 + t1 * vx1
        yi = y1 + t1 * vy1
        return float(xi), float(yi)

    # ---------- start ----------
    fname = os.path.basename(in_path)
    m = re.match(r"^(\d+)", fname)
    page_num = int(m.group(1)) if m else 0
    bad_on_left = (page_num % 2 == 1)

    img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print("Failed to read", in_path); return
    input_is_grayscale = (len(img.shape) == 2)
    if len(img.shape) == 2:
        # OpenCV expects 3-channel images for many operations
        # later convert back to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    H_img, W_img = img.shape[:2]

    # fixed
    # # FIXME wrong H_img?
    # expected_w = int(round(ASPECT * H_img))
    # expected_h = H_img

    dbgdir = os.path.join(OUTPUT_DIR, "debug", f"{page_num:03d}")
    if DEBUG: ensure_dir(dbgdir)

    gray, mask, contours = get_gray_mask_contours(img, dbgdir)

    if not contours:
        print(f"Warning: no contours found in {in_path}")
        return

    page_contour = max(contours, key=cv2.contourArea)

    if config.use_three_edge_deskew:

        top_pts, bottom_pts, outside_pts = split_edge_candidates(
            page_contour,
            bad_on_left
        )

        top_line = fit_line_ransac(top_pts)[:4]
        bottom_line = fit_line_ransac(bottom_pts)[:4]
        outside_line = fit_line_ransac(outside_pts)[:4]

        # old
        # top_angle = math.degrees(line_angle(top_line))
        # bottom_angle = math.degrees(line_angle(bottom_line))
        # outside_angle = math.degrees(line_angle(outside_line))

        # new
        top_angle = horizontal_line_angle(top_line)
        bottom_angle = horizontal_line_angle(bottom_line)
        outside_angle = vertical_line_angle(outside_line)

        # start debug prints
        print()
        print(f"line 570: page_num={page_num}")

        rotation_error = -1 * (
            -top_angle
            -bottom_angle
        ) / 2.0

        Mrot = cv2.getRotationMatrix2D(
            (W_img/2, H_img/2),
            rotation_error,
            1.0
        )

        print(
            f"line 575: before rotation: "
            f"top_angle={top_angle:.3f} "
            f"bottom_angle={bottom_angle:.3f} "
            f"outside_angle={outside_angle:.3f}"
        )

        rotated = cv2.warpAffine(
            img,
            Mrot,
            (W_img, H_img),
            borderValue=(255,255,255)
        )

        Hr, Wr = rotated.shape[:2]
        print("line 580: rotated size", Wr, Hr)

        # img = rotated # ?

# re-detect lines in the rotated image

        # gray, mask, contours = get_gray_mask_contours(img, dbgdir)
        gray, mask, contours = get_gray_mask_contours(rotated, dbgdir)

        if 1:
            top_angle2 = horizontal_line_angle(top_line)
            bottom_angle2 = horizontal_line_angle(bottom_line)
            outside_angle2 = vertical_line_angle(outside_line)
            print(
                f"line 620: after rotation and re-fitting: "
                f"top_angle2={top_angle2:.3f} "
                f"bottom_angle2={bottom_angle2:.3f} "
                f"outside_angle2={outside_angle2:.3f}"
            )

        if not contours:
            print(f"line 630: Warning: no contours found in {in_path}")
            return

        page_contour = max(contours, key=cv2.contourArea)

top_pts, bottom_pts, outside_pts = split_edge_candidates(
            page_contour,
            bad_on_left
        )

        top_line = fit_line_ransac(top_pts)[:4]
        bottom_line = fit_line_ransac(bottom_pts)[:4]
        outside_line = fit_line_ransac(outside_pts)[:4]

        # fixed
        # # FIXME H_img is wrong
        # expected_h = H_img
        # expected_w = int(round(ASPECT * expected_h))

        vx, vy, x0, y0 = outside_line

        outside_top = intersect_lines(
            outside_line,
            top_line
        )

        outside_bottom = intersect_lines(
            outside_line,
            bottom_line
        )

        if 0:
            # approximated page height
            page_height = math.dist(
                outside_top,
                outside_bottom
            )
        else:
            # perpendicular page height
            page_height = math.hypot(
                outside_bottom[0] - outside_top[0],
                outside_bottom[1] - outside_top[1]
            )

        # no. this fails to reconstruct the page height...
        # TODO try to solve this with the average height of multiple pages
        # assuming all pages must have the same height
        # also allowing the user to specify a scale_y factor
        if 0:
            # fix scan height
            # document scanners can distort scans in the Y direction
            if 0:
                # use rotated_scan_y and page_height
                rotated_scan_y_px = px_of_mm(config.rotated_scan_y, config.scan_resolution)
                target_h = rotated_scan_y_px
                scale_y = target_h / page_height
            elif 1:
                # use rotated_margined_scan_y and H_img
                rotated_margined_scan_y_px = px_of_mm(config.rotated_margined_scan_y, config.scan_resolution)
                target_h = rotated_margined_scan_y_px
                scale_y = target_h / H_img
            else:
                # no, this fails because the scanner removes one edge
                # so actual_aspect is always wrong...
                #
                # use rotated_margined_scan_y and H_img
                # expected: what we ordered from the scanner
                expected_aspect = (
                    config.rotated_margined_scan_x /
                    config.rotated_margined_scan_y
                )
                expected_height = config.rotated_margined_scan_y
                # actual: what the scanner gave us
                actual_aspect = W_img / H_img
                actual_height = H_img
                # we assume the scanner always returns correct X coordinates
                # and all errors are only in Y coordinates
                # expected_aspect / actual_aspect = actual_height / expected_height
                actual_height_2 = expected_aspect / actual_aspect / expected_height
                expected_aspect_factor = expected_aspect / actual_aspect
                # the scale of actual_height relative to actual_height_2
                actual_height_scale = actual_height / actual_height_2
                print(f"expected_aspect={expected_aspect} actual_aspect={actual_aspect} expected_aspect_factor={expected_aspect_factor}")
                print(f"expected_height={expected_height} actual_height={actual_height} actual_height_2={actual_height_2} actual_height_scale={actual_height_scale}")
                target_h = actual_height_2
                scale_y = target_h / H_img

            # debug: manually set the scale_y factor
            # scale_y = 1 / 1.03 # shrink the page height by 3%

            scale_y_tolerance = 0.001 # 0.1%

            if scale_y < (1 - scale_y_tolerance) or (1 + scale_y_tolerance) < scale_y:
                print(f"line 680: scale_y={scale_y} target_h={target_h} page_height={page_height} -> scaling height")
                rotated = cv2.resize(
                    rotated,
                    None,
                    fx=1.0,
                    fy=scale_y,
                    interpolation=cv2.INTER_CUBIC
                )
                page_height = page_height * scale_y

                # re-detect lines in the scaled image

                # gray, mask, contours = get_gray_mask_contours(img, dbgdir)
                gray, mask, contours = get_gray_mask_contours(rotated, dbgdir)

                if 1:
                    top_angle2 = horizontal_line_angle(top_line)
                    bottom_angle2 = horizontal_line_angle(bottom_line)
                    outside_angle2 = vertical_line_angle(outside_line)
                    print(
                        f"line 620: after rotation and re-fitting: "
                        f"top_angle2={top_angle2:.3f} "
                        f"bottom_angle2={bottom_angle2:.3f} "
                        f"outside_angle2={outside_angle2:.3f}"
                    )

                if not contours:
                    print(f"line 630: Warning: no contours found in {in_path}")
                    return

                page_contour = max(contours, key=cv2.contourArea)

                top_pts, bottom_pts, outside_pts = split_edge_candidates(
                    page_contour,
                    bad_on_left
                )

                top_line = fit_line_ransac(top_pts)[:4]
                bottom_line = fit_line_ransac(bottom_pts)[:4]
                outside_line = fit_line_ransac(outside_pts)[:4]

                # fixed
                # # FIXME H_img is wrong
                # expected_h = H_img
                # expected_w = int(round(ASPECT * expected_h))

                vx, vy, x0, y0 = outside_line

                outside_top = intersect_lines(
                    outside_line,
                    top_line
                )

                outside_bottom = intersect_lines(
                    outside_line,
                    bottom_line
                )

            else:
                print(f"line 680: scale_y={scale_y} target_h={target_h} page_height={page_height} -> not scaling height")

        expected_h = int(round(page_height))

        # ASPECT = x / y
        # x = y * ASPECT
        # expected_w = int(round(expected_h * ASPECT))
        expected_w = int(round(page_height * ASPECT))

        if 0:
            # crop as a rectangle
            x_out = (
                outside_top[0]
                +
                outside_bottom[0]
            ) / 2

            x_out = int(round(x_out))

            # if bad_on_left:
            #     x0_page = x_out
            #     x1_page = x_out + expected_w
            # else:
            #     x0_page = x_out - expected_w
            #     x1_page = x_out

            if bad_on_left:
                # outside edge is RIGHT
                x1_page = x_out
                x0_page = x_out - expected_w
            else:
                # outside edge is LEFT
                x0_page = x_out
                x1_page = x_out + expected_w

            # clamp
            # x0_page = max(0, x0_page)
            # x1_page = min(rotated.shape[1], x1_page)

            y_top = int(round(outside_top[1]))
            y_bottom = int(round(outside_bottom[1]))

            src_x0 = max(0, x0_page)
            src_x1 = min(rotated.shape[1], x1_page)

            src_y0 = max(0, y_top)
            src_y1 = min(rotated.shape[0], y_top + expected_h)

            dst_x0 = src_x0 - x0_page
            dst_y0 = src_y0 - y_top

            print(
                "line 640:",
                f"outside_top={outside_top}",
                f"outside_bottom={outside_bottom}",
            )

            # crop = rotated[
            #     y_top:y_top+expected_h,
            #     x0_page:x1_page
            # ]

            full_crop = np.ones(
                (expected_h, expected_w, 3),
                dtype=rotated.dtype
            ) * 255

            full_crop[
                dst_y0:dst_y0 + (src_y1-src_y0),
                dst_x0:dst_x0 + (src_x1-src_x0)
            ] = rotated[
                src_y0:src_y1,
                src_x0:src_x1
            ]

            # FIXME fill empty area near binding edge
            # currently it is all white
            # but it should copy the vertical pattern near the binding edge

            crop = full_crop

            print(
                "line 650: crop",
                f"W_img={W_img}",
                f"H_img={H_img}",
                f"x0_page={x0_page}",
                f"x1_page={x1_page}",
                f"y_top={y_top}",
                f"y_bottom={y_bottom}",
                f"src_x0={src_x0}",
                f"src_x1={src_x1}",
                f"src_y0={src_y0}",
                f"src_y1={src_y1}",
                f"dst_x0={dst_x0}",
                f"dst_y0={dst_y0}",
                f"expected_w={expected_w}",
                f"expected_h={expected_h}",
            )

            if 1:
                # debug
                vis = rotated.copy()
                cv2.rectangle(
                    vis,
                    (int(x0_page), int(y_top)),
                    (int(x1_page), int(y_bottom)),
                    (0,0,255),
                    3,
                )
                name = f"{page_num:03d}.crop_debug_rotated_line730.jpg"
                print(f"writing {name}")
                cv2.imwrite(name, vis)

        else:
            # crop as a quadrilateral
            # not better than "crop as a rectangle"?
            # expand the binding edge to expected_w
            if bad_on_left:
                # outside edge is the RIGHT edge
                x1_top = outside_top[0]
                x1_bottom = outside_bottom[0]
                x0_top = x1_top - expected_w
                x0_bottom = x1_bottom - expected_w
            else:
                # outside edge is the LEFT edge
                x0_top = outside_top[0]
                x0_bottom = outside_bottom[0]
                x1_top = x0_top + expected_w
                x1_bottom = x0_bottom + expected_w

            src = np.float32([
                [x0_top, outside_top[1]],
                [x1_top, outside_top[1]],
                [x1_bottom, outside_bottom[1]],
                [x0_bottom, outside_bottom[1]],
            ])

            dst = np.float32([
                [0,0],
                [expected_w-1,0],
                [expected_w-1,expected_h-1],
                [0,expected_h-1],
            ])

            M = cv2.getPerspectiveTransform(src, dst)

            crop = cv2.warpPerspective(
                rotated,
                M,
                (expected_w, expected_h),
                borderValue=(255,255,255)
            )

            print(
                "line 650: crop",
                f"W_img={W_img}",
                f"H_img={H_img}",
                f"outside_top={outside_top}",
                f"outside_bottom={outside_bottom}",
                f"x1_top={x1_top}",
                f"x1_bottom={x1_bottom}",
                f"x0_top={x0_top}",
                f"x0_bottom={x0_bottom}",
                f"expected_w={expected_w}",
                f"expected_h={expected_h}",
            )

            if 1:
                # debug
                vis = rotated.copy()
                pts = np.array([
                    [x0_top, outside_top[1]],
                    [x1_top, outside_top[1]],
                    [x1_bottom, outside_bottom[1]],
                    [x0_bottom, outside_bottom[1]],
                ], np.int32)
                cv2.polylines(vis, [pts], True, (0,0,255), 3)
                name = f"{page_num:03d}.crop_debug_rotated_line730.jpg"
                print(f"writing {name}")
                cv2.imwrite(name, vis)

        # crop = rotated[
        #     0:expected_h,
        #     int(x0_page):int(x1_page)
        # ]

        # y_top = round(outside_top[1])
        # y_bottom = round(outside_bottom[1])
        # actual_height = y_bottom - y_top
        # crop = rotated[
        #     y_top:y_top + expected_h,
        #     x0_page:x1_page
        # ]

        # y_top = int(round(outside_top[1]))
        # y_bottom = int(round(outside_bottom[1]))
        # crop = rotated[
        #     y_top:y_bottom,
        #     x0_page:x1_page
        # ]

        if 0:
            # img = repair_binding(img, bad_on_left, width=50)
            if 1:
                # rotated = repair_binding(rotated, bad_on_left, width=50)
                # Hr, Wr = rotated.shape[:2]
                # print("line 660: rotated size", Wr, Hr)
                # warped = rotated
                crop = repair_binding(crop, bad_on_left, width=50)
                Hr, Wr = crop.shape[:2]
                print("line 660: crop size", Wr, Hr)
            else:
                crop = repair_binding(crop, bad_on_left, width=50)

            print(
                "line 760: after repair_binding: crop actual:",
                f"crop.shape[1]={crop.shape[1]}",
                f"expected_w={expected_w}",
            )

        warped = crop

    else:
        # config.use_three_edge_deskew == False
        # FIXME use only two page edges: outside, bottom
        # Approximate contour to quadrilateral
        epsilon = 0.02 * cv2.arcLength(page_contour, True)
        approx = cv2.approxPolyDP(page_contour, epsilon, True)
        if len(approx) != 4:
            approx = cv2.convexHull(page_contour)
            if len(approx) < 4:
                print(f"Warning: not enough points for perspective in {in_path}")
                return
            # pick 4 extreme points
            pts = np.array([
                approx[approx[:,0,0].argmin()][0],  # leftmost
                approx[approx[:,0,1].argmin()][0],  # topmost
                approx[approx[:,0,0].argmax()][0],  # rightmost
                approx[approx[:,0,1].argmax()][0]   # bottommost
            ])
        else:
            pts = approx.reshape(4,2)

        rect = order_points(pts)

        # Perspective transform
        widthA = np.linalg.norm(rect[2] - rect[3])
        widthB = np.linalg.norm(rect[1] - rect[0])
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.linalg.norm(rect[1] - rect[2])
        heightB = np.linalg.norm(rect[0] - rect[3])
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    # # Add internal white border
    # h, w = warped.shape[:2]
    # canvas = np.ones_like(warped) * 255
    # b = BORDER_SIZE
    # canvas[b:h-b, b:w-b] = warped[b:h-b, b:w-b]

    h, w = warped.shape[:2]
    b = BORDER_SIZE

    # Prepare a canvas with the same content as warped
    canvas = warped.copy()

    # Function to compute average color along a strip
    def avg_color_strip(img, axis, start, end, strip_width=1):
        if axis == 'top':
            strip = img[start:end, :, :]
        elif axis == 'bottom':
            strip = img[h-end:h-start, :, :]
        elif axis == 'left':
            strip = img[:, start:end, :]
        elif axis == 'right':
            strip = img[:, w-end:w-start, :]
        else:
            raise ValueError("Invalid axis")
        return np.mean(strip, axis=(0,1)).astype(np.uint8)

    # FIXME preserve patterns near edges
    if 0:
        # Fill borders with local average color
        canvas[0:b, :, :] = avg_color_strip(canvas, 'top', 50, 100)       # top border
        canvas[h-b:h, :, :] = avg_color_strip(canvas, 'bottom', 50, 100)  # bottom border
        canvas[:, 0:b, :] = avg_color_strip(canvas, 'left', 50, 100)      # left border
        canvas[:, w-b:w, :] = avg_color_strip(canvas, 'right', 50, 100)   # right border

    if input_is_grayscale:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

    ensure_dir(os.path.dirname(out_path))
    cv2.imwrite(out_path, canvas)
    print(f"writing {out_path}")

def main():
    ensure_dir(OUTPUT_DIR)
    # TODO use image_format from 030-measure-page-size.txt
    image_format = "jpg"
    files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(f".{image_format}")])
    if not files:
        print("No image files found in", INPUT_DIR)
        return
    for fname in files:
        if fname != "012.jpg": continue # debug
        m = re.match(r"^(\d+)", fname)
        # page_num = int(m.group(1)) if m else 0
        page_num = int(m.group(1)) # can throw
        # if page_num != 14: continue # debug
        # if not page_num in [340, 345]: continue # debug
        # if page_num < 300: continue # debug
        # if page_num != 320: continue # debug
        in_path = os.path.join(INPUT_DIR, fname)
        out_path = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(out_path):
            continue
        try:
            process_image(in_path, out_path)
        except Exception as exc:
            print(f"Error processing {fname}: {exc}")
            raise

if __name__ == "__main__":
    main()
```

## Response:

7/12/2026, 6:53:16 PM

Yes. The simplest way is to disable only the **binding-edge expansion** while keeping the rest of the quadrilateral crop logic.

Right now the expansion happens here:

``` python
# expand the binding edge to expected_w
if bad_on_left:
    # outside edge is the RIGHT edge
    x1_top = outside_top[0]
    x1_bottom = outside_bottom[0]
    x0_top = x1_top - expected_w
    x0_bottom = x1_bottom - expected_w
else:
    # outside edge is the LEFT edge
    x0_top = outside_top[0]
    x0_bottom = outside_bottom[0]
    x1_top = x0_top + expected_w
    x1_bottom = x0_bottom + expected_w
```

This forces the crop width to `expected_w`, even if the detected page is narrower because the binding edge was cropped away.

For your current testing, replace that block with a version that uses the **actual detected page width**:

``` python
# temporarily disabled:
# do not expand missing binding edge to expected_w
# use only the detected page edges

if bad_on_left:
    # outside edge is RIGHT edge
    x1_top = outside_top[0]
    x1_bottom = outside_bottom[0]

    # use the actual detected left edge
    x0_top = np.min(page_contour[:,0,0])
    x0_bottom = x0_top

else:
    # outside edge is LEFT edge
    x0_top = outside_top[0]
    x0_bottom = outside_bottom[0]

    # use the actual detected right edge
    x1_top = np.max(page_contour[:,0,0])
    x1_bottom = x1_top
```

However, there is a subtle problem: `page_contour` after thresholding may include the background or may not have a reliable missing-edge position. A cleaner temporary solution is to use the detected quadrilateral width from the two horizontal edge intersections:

``` python
# temporarily disable binding expansion:
# keep detected width

if bad_on_left:
    # outside edge is right
    x1_top = outside_top[0]
    x1_bottom = outside_bottom[0]

    # find leftmost detected page boundary
    x0_top = np.min(top_pts[:,0])
    x0_bottom = np.min(bottom_pts[:,0])

else:
    # outside edge is left
    x0_top = outside_top[0]
    x0_bottom = outside_bottom[0]

    # find rightmost detected page boundary
    x1_top = np.max(top_pts[:,0])
    x1_bottom = np.max(bottom_pts[:,0])
```

Then leave everything below unchanged:

``` python
src = np.float32([
    [x0_top, outside_top[1]],
    [x1_top, outside_top[1]],
    [x1_bottom, outside_bottom[1]],
    [x0_bottom, outside_bottom[1]],
])

dst = np.float32([
    [0,0],
    [expected_w-1,0],
    [expected_w-1,expected_h-1],
    [0,expected_h-1],
])
```

One more thing: because `dst` still uses `expected_w`, this will **stretch the narrower crop horizontally** to the configured width. If you want to completely disable width normalization too, change:

``` python
expected_w = int(round(page_height * ASPECT))
```

to:

``` python
expected_w = int(round(
    math.dist((x0_top, outside_top[1]), (x1_top, outside_top[1]))
))
```

or after defining `src`:

``` python
expected_w = int(round(
    (abs(x1_top-x0_top) + abs(x1_bottom-x0_bottom)) / 2
))
```

Then your output pages will preserve the currently detected width.

For your stated plan (later normalize height across the book, then reconstruct the missing binding edge), I would temporarily do **both**:

1.  Keep the measured `page_height` → no Y normalization.
2.  Disable the `expected_w` expansion.
3.  Output variable-width pages.
4.  Run a separate script over all pages to find the median/average true page height.
5.  Normalize all pages to that height.
6.  Re-run a binding reconstruction step using the corrected height.

That separates the two unknowns (scanner Y distortion and missing binding width), which is exactly the right direction.

## Prompt:

7/12/2026, 7:21:48 PM

good news: i have made a mistake in my "fix scan height" code: i forgot that all my raw scans are rotated by 90 or 270 degrees, so the scanner's Y errors become our X errors, so we need scale_x to fix the page width...

can you fix my code around line 655?

lets give this one more try

``` py
#!/usr/bin/env python3
"""
extract scanned page from gray background

restore the binding edge of the page
by filling the missing width
with the average color near the binding edge
"""

INPUT_DIR = "060-rotate-crop"
OUTPUT_DIR = "065-remove-page-borders"

# === Tuning parameters ===
DEBUG = True
DEBUG = False
BORDER_SIZE = 100  # pixels

RANSAC_ITER = 400
RANSAC_INLIER_DIST = 6.0
# RANSAC_MIN_INLIERS = 30
RANSAC_MIN_INLIERS = 20

THRESH_HIGH_PERCENTILE = 99
THRESH_MIN = 200

import os
import re
import math
import random
import numpy as np
import cv2

from _shared import (
    load_config,
    get_page_num,
)

config = load_config()

# no, this is wrong if (config.do_rotate == True)
# scan_x = config.scan_x
# scan_y = config.scan_y
# config.scan_aspect = scan_x / scan_y
# ASPECT = config.scan_aspect

ASPECT = config.rotated_scan_aspect

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def save_dbg(img, path):
    ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, img)

def percentile_threshold(gray):
    high_p = np.percentile(gray, THRESH_HIGH_PERCENTILE)
    thr = max(THRESH_MIN, int(high_p * 0.95))
    _, mask = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
    return mask, thr, int(high_p)

def detect_vertical_streaks(mask, approx_width=3, length_thresh_ratio=0.15):
    h, w = mask.shape
    kx = approx_width
    ky = max(15, int(h * 0.02))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
    long_vertical = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(long_vertical, connectivity=8)
    streak_mask = np.zeros_like(mask)
    length_thresh = max(10, int(h * length_thresh_ratio))
    for i in range(1, num_labels):
        x, y, ww, hh, area = stats[i]
        if hh >= length_thresh and ww <= max(5, int(w * 0.01)):
            streak_mask[labels == i] = 255
    return streak_mask

def keep_largest_component(mask):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    out = np.zeros_like(mask)
    out[labels == best] = 255
    return out

def contour_to_pts(contour):
    return contour.reshape(-1,2)

def fit_line_ransac(pts, iterations=RANSAC_ITER, inlier_dist=RANSAC_INLIER_DIST, min_inliers=RANSAC_MIN_INLIERS):
    if len(pts) < 2:
        raise ValueError("Not enough points")
    best_inliers = None
    best_model = None
    n = len(pts)
    ptsf = pts.astype(np.float32)
    for _ in range(iterations):
        i1, i2 = random.sample(range(n), 2)
        p1 = ptsf[i1]; p2 = ptsf[i2]
        vx = float(p2[0] - p1[0]); vy = float(p2[1] - p1[1])
        if vx == 0 and vy == 0:
            continue
        dists = np.abs(vy*(ptsf[:,0]-p1[0]) - vx*(ptsf[:,1]-p1[1])) / (math.hypot(vx, vy) + 1e-12)
        inliers = dists <= inlier_dist
        cnt = int(inliers.sum())
        if cnt >= min_inliers and (best_inliers is None or cnt > int(best_inliers.sum())):
            best_inliers = inliers.copy()
            best_model = (vx, vy, float(p1[0]), float(p1[1]))
    if best_model is None:
        vx, vy, x0, y0 = cv2.fitLine(ptsf, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        inlier_mask = np.ones(len(pts), dtype=bool)
        return float(vx), float(vy), float(x0), float(y0), inlier_mask
    inlier_pts = ptsf[best_inliers]
    vx, vy, x0, y0 = cv2.fitLine(inlier_pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    inlier_mask = best_inliers
    return float(vx), float(vy), float(x0), float(y0), inlier_mask

def intersect_lines(l1, l2):
    vx1, vy1, x1, y1 = l1
    vx2, vy2, x2, y2 = l2
    A = np.array([[vx1, -vx2], [vy1, -vy2]], dtype=np.float32)
    b = np.array([x2 - x1, y2 - y1], dtype=np.float32)
    det = np.linalg.det(A)
    if abs(det) < 1e-8:
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    t1, t2 = np.linalg.solve(A, b)
    xi = x1 + t1 * vx1
    yi = y1 + t1 * vy1
    return float(xi), float(yi)

def build_affine_without_shear(pt_v_top, pt_v_bot, expected_w, expected_h, bad_on_left, img, dbgdir=None):
    """
    Build a source triangle that enforces orthogonal page axes (no shear) using:
      - pt_v_top: intersection of top line and good vertical (top-right when good vertical is right)
      - pt_v_bot: intersection of bottom line and good vertical (bottom-right when good vertical is right)
    Returns (M_aff, src_corners_dict, dst_corners_dict, reason)
    - M_aff: 2x3 affine matrix or None on failure
    - src_corners_dict: dict of TL, TR, BL, BR (np.float32 2-vectors)
    - dst_corners_dict: dict of TL, TR, BL, BR in destination coordinates
    - reason: diagnostic string
    """
    # convert to vectors
    pt_v_top = np.array(pt_v_top, dtype=np.float32)
    pt_v_bot = np.array(pt_v_bot, dtype=np.float32)

    # compute direction along top edge: vector from bottom intersection to top intersection projected horizontally
    # but we don't have explicit top_line direction vector here; we will approximate using (pt_v_top - pt_v_bot) rotated?
    # Better: caller should supply top_line direction (vx,vy). If you don't have it, approximate from nearby contour points.
    # For compatibility with your code, expect you have top_vx, top_vy; if not, fall back to vector between two top contour points.
    # Here we'll assume top_vx, top_vy are available in outer scope; otherwise compute from pt_v_top->pt_v_bot displacement projected perpendicular:
    # To keep this helper self-contained, we compute top_dir as average of (pt_v_top_to_some_right) - but simplest robust approach:
    # compute approximate top direction by taking small offset vector along top by sampling image gradient: fallback to (1,0).

    # --- Here caller should provide top_unit; we'll compute a safe top_unit using neighbor pixels if available ---
    # We'll attempt to derive a top_unit by sampling a short step along the image: use vector from pt_v_top to pt_v_top projected to image center
    cx, cy = img.shape[1] / 2.0, img.shape[0] / 2.0
    # prefer vector pointing rightwards by projecting displacement to center
    approx = pt_v_top - np.array([cx, cy], dtype=np.float32)
    if np.linalg.norm(approx) < 1e-6:
        approx = np.array([1.0, 0.0], dtype=np.float32)
    ux, uy = approx / (np.linalg.norm(approx) + 1e-12)

    # Force ux positive (pointing to the right) — we want u to be rightward
    if ux < 0:
        ux, uy = -ux, -uy

    # make perpendicular downwards: p = (-uy, ux)
    px, py = -uy, ux

    # normalize again (just in case)
    n_u = math.hypot(ux, uy)
    n_p = math.hypot(px, py)
    if n_u < 1e-6 or n_p < 1e-6:
        return None, None, None, "degenerate_axes"
    ux, uy = ux / n_u, uy / n_u
    px, py = px / n_p, py / n_p

    # Now construct corners. We know pt_v_top/pt_v_bot correspond to the known vertical edge:
    # If bad_on_left is True, good vertical is the RIGHT edge => pt_v_top == TR, pt_v_bot == BR.
    # If bad_on_left is False, good vertical is LEFT edge => pt_v_top == TL, pt_v_bot == BL.
    if bad_on_left:
        src_TR = pt_v_top
        src_BR = pt_v_bot
        # compute TL and BL by moving left along top unit by expected_w
        src_TL = src_TR - np.array([ux, uy], dtype=np.float32) * float(expected_w)
        src_BL = src_BR - np.array([ux, uy], dtype=np.float32) * float(expected_w)
    else:
        src_TL = pt_v_top
        src_BL = pt_v_bot
        # compute TR and BR by moving right along top unit by expected_w
        src_TR = src_TL + np.array([ux, uy], dtype=np.float32) * float(expected_w)
        src_BR = src_BL + np.array([ux, uy], dtype=np.float32) * float(expected_w)

    # Optional: enforce vertical height by projecting (TL->BL) onto perp and rescale to expected_h
    # Compute current vertical vector and its projection onto perp (px,py)
    cur_v = src_BL - src_TL
    proj = cur_v.dot(np.array([px, py], dtype=np.float32))
    # if projection is tiny, fall back but otherwise adjust BL to ensure exact page height
    if abs(proj) > 1e-6:
        # adjust scale to exact expected_h
        scale = float(expected_h) / proj
        # recompute BL, BR as TL + perp*expected_h and TR + perp*expected_h
        src_BL = src_TL + np.array([px, py], dtype=np.float32) * float(expected_h)
        src_BR = src_TR + np.array([px, py], dtype=np.float32) * float(expected_h)
    else:
        # if proj nearly zero, we cannot rely on vertical measurement; still use perpendicular step
        src_BL = src_TL + np.array([px, py], dtype=np.float32) * float(expected_h)
        src_BR = src_TR + np.array([px, py], dtype=np.float32) * float(expected_h)

    # Build src/dst dicts
    src_corners = {
        "TL": src_TL.astype(np.float32),
        "TR": src_TR.astype(np.float32),
        "BL": src_BL.astype(np.float32),
        "BR": src_BR.astype(np.float32)
    }
    dst_corners = {
        "TL": np.array([0.0, 0.0], dtype=np.float32),
        "TR": np.array([expected_w - 1.0, 0.0], dtype=np.float32),
        "BL": np.array([0.0, expected_h - 1.0], dtype=np.float32),
        "BR": np.array([expected_w - 1.0, expected_h - 1.0], dtype=np.float32)
    }

    # choose triangle for affine depending on orientation (map TL, BL, TR -> dst TL, BL, TR)
    src_tri = np.vstack([src_corners["TL"], src_corners["BL"], src_corners["TR"]]).astype(np.float32)
    dst_tri = np.vstack([dst_corners["TL"], dst_corners["BL"], dst_corners["TR"]]).astype(np.float32)

    # validate triangle
    area = abs(0.5 * (src_tri[0,0]*(src_tri[1,1]-src_tri[2,1]) + src_tri[1,0]*(src_tri[2,1]-src_tri[0,1]) + src_tri[2,0]*(src_tri[0,1]-src_tri[1,1])))
    if area < 1.0 or not np.isfinite(src_tri).all():
        return None, src_corners, dst_corners, f"invalid_src_tri_area={area:.3f}"

    try:
        M = cv2.getAffineTransform(src_tri, dst_tri)
    except cv2.error as e:
        return None, src_corners, dst_corners, f"getAffineTransform_failed:{e}"

    # debug overlay: draw corners & axes if dbgdir provided
    if dbgdir is not None:
        vis = img.copy()
        def draw_pt(pt, color=(0,0,255), tag=""):
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 6, color, -1)
            if tag:
                cv2.putText(vis, tag, (int(pt[0]+6), int(pt[1]-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        draw_pt(src_corners["TL"], (0,255,0), "TL")
        draw_pt(src_corners["TR"], (0,128,255), "TR")
        draw_pt(src_corners["BL"], (255,0,0), "BL")
        draw_pt(src_corners["BR"], (255,255,0), "BR")
        # draw axis arrows from TL
        origin = src_corners["TL"]
        arrow_u = origin + np.array([ux, uy], dtype=np.float32) * 80.0
        arrow_p = origin + np.array([px, py], dtype=np.float32) * 80.0
        cv2.arrowedLine(vis, (int(origin[0]), int(origin[1])), (int(arrow_u[0]), int(arrow_u[1])), (255,0,255), 2)
        cv2.arrowedLine(vis, (int(origin[0]), int(origin[1])), (int(arrow_p[0]), int(arrow_p[1])), (0,255,255), 2)
        cv2.imwrite(os.path.join(dbgdir, "debug_axes_and_corners.png"), vis)

    return M, src_corners, dst_corners, "ok"

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

def split_edge_candidates(contour, bad_on_left):
    pts = contour.reshape(-1, 2)

    xs = pts[:,0]
    ys = pts[:,1]

    w = xs.max() - xs.min()
    h = ys.max() - ys.min()

    margin_x = w * 0.15
    margin_y = h * 0.15

    # outside edge
    if bad_on_left:
        outside = pts[xs > xs.max() - margin_x]
    else:
        outside = pts[xs < xs.min() + margin_x]

    # top edge
    top = pts[ys < ys.min() + margin_y]

    # bottom edge
    bottom = pts[ys > ys.max() - margin_y]

    return top, bottom, outside

def line_angle(line):
    vx, vy, _, _ = line
    return math.atan2(vy, vx)

def horizontal_line_angle(line):
    vx, vy, _, _ = line

    a = math.degrees(math.atan2(vy, vx))

    while a < -45:
        a += 180

    while a > 45:
        a -= 180

    return a

def vertical_line_angle(line):
    vx, vy, _, _ = line

    a = math.degrees(math.atan2(vy, vx))

    while a < 45:
        a += 180

    while a > 135:
        a -= 180

    return a

def normalize_angle_deg(a):
    while a < -90:
        a += 180
    while a > 90:
        a -= 180
    return a

def get_gray_mask_contours(img, dbgdir):
    # --- threshold for geometry only ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask_init, thr, hp = percentile_threshold(gray)
    if cv2.mean(gray)[0] < 127:
        mask_init = cv2.bitwise_not(mask_init)
    if DEBUG:
        save_dbg(gray, os.path.join(dbgdir, "01_gray.png"))
        save_dbg(mask_init, os.path.join(dbgdir, f"02_thresh_thr{thr}_hp{hp}.png"))

    # without a9dc2b24e6b0d1d49b6fc232223d6431ba3442a5 bad: fix perspective transform for broken ADF scanners
    # Invert if necessary, so the page is always "white" for thresholding
    # We'll use Otsu's method to detect high-contrast area
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Determine whether the page is darker or lighter than background
    mean_val = cv2.mean(gray, mask=None)[0]
    if mean_val < 127:  # dark page: invert mask
        mask = cv2.bitwise_not(mask)

    # Morphology to remove small gaps
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return gray, mask, contours

def repair_binding(img, bad_on_left, width=50):

    h,w = img.shape[:2]

    if bad_on_left:
        sample_range = range(width)
        fill_range = range(width)
        sample_x = width
    else:
        sample_range = range(w-width,w)
        fill_range = range(w-width,w)
        sample_x = w-width

    for y in range(h):

        if bad_on_left:
            sample = img[y, width:width+20]
            color = np.mean(sample,axis=0)
            img[y,:width] = color

        else:
            sample = img[y,w-width-20:w-width]
            color = np.mean(sample,axis=0)
            img[y,w-width:] = color

    return img

def px_of_mm(mm, dpi):
    return mm * dpi / 25.4

def process_image(in_path, out_path):
    """
    Robust page extraction that handles missing top (or bottom) edges.
    - preserves original colors (warps the original image)
    - builds orthogonal axes (no shear)
    - if top is missing, uses bottom + good vertical to infer top by moving up expected_h
    - fills outside the filled page polygon with pure white
    - extensive debug output in OUTPUT_DIR/debug/<page_num>/
    """
    # ---------- small helpers ----------
    def ensure_dir(p):
        os.makedirs(p, exist_ok=True)

    def save_dbg(img, path):
        ensure_dir(os.path.dirname(path))
        cv2.imwrite(path, img)

    def percentile_threshold(gray):
        high_p = np.percentile(gray, THRESH_HIGH_PERCENTILE)
        thr = max(THRESH_MIN, int(high_p * 0.95))
        _, mask = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
        return mask, thr, int(high_p)

    def keep_largest_component(mask):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels <= 1:
            return mask
        areas = stats[1:, cv2.CC_STAT_AREA]
        best = 1 + int(np.argmax(areas))
        out = np.zeros_like(mask)
        out[labels == best] = 255
        return out

    def detect_vertical_streaks(mask, approx_width=3, length_thresh_ratio=0.15):
        h, w = mask.shape
        kx = approx_width
        ky = max(15, int(h * 0.02))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
        long_vertical = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(long_vertical, connectivity=8)
        streak_mask = np.zeros_like(mask)
        length_thresh = max(10, int(h * length_thresh_ratio))
        for i in range(1, num_labels):
            x, y, ww, hh, area = stats[i]
            if hh >= length_thresh and ww <= max(5, int(w * 0.01)):
                streak_mask[labels == i] = 255
        return streak_mask

    def contour_to_pts(c):
        return c.reshape(-1, 2)

    def fit_line_ransac(pts, iterations=RANSAC_ITER, inlier_dist=RANSAC_INLIER_DIST, min_inliers=RANSAC_MIN_INLIERS):
        if len(pts) < 2:
            raise ValueError("Not enough points for line fit")
        best_inliers = None
        best_cnt = 0
        best_model = None
        ptsf = pts.astype(np.float32)
        n = len(ptsf)
        for _ in range(iterations):
            i1, i2 = random.sample(range(n), 2)
            p1 = ptsf[i1]; p2 = ptsf[i2]
            vx = float(p2[0] - p1[0]); vy = float(p2[1] - p1[1])
            if abs(vx) < 1e-6 and abs(vy) < 1e-6:
                continue
            dists = np.abs(vy*(ptsf[:,0]-p1[0]) - vx*(ptsf[:,1]-p1[1])) / (math.hypot(vx, vy) + 1e-12)
            inliers = dists <= inlier_dist
            cnt = int(inliers.sum())
            if cnt >= min_inliers and cnt > best_cnt:
                best_cnt = cnt
                best_inliers = inliers.copy()
                best_model = (vx, vy, float(p1[0]), float(p1[1]))
        if best_model is None:
            vx, vy, x0, y0 = cv2.fitLine(ptsf, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
            inlier_mask = np.ones(len(ptsf), dtype=bool)
            return float(vx), float(vy), float(x0), float(y0), inlier_mask
        inlier_pts = ptsf[best_inliers]
        vx, vy, x0, y0 = cv2.fitLine(inlier_pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        return float(vx), float(vy), float(x0), float(y0), best_inliers

    def intersect_lines(l1, l2):
        vx1, vy1, x1, y1 = l1
        vx2, vy2, x2, y2 = l2
        A = np.array([[vx1, -vx2], [vy1, -vy2]], dtype=np.float32)
        b = np.array([x2 - x1, y2 - y1], dtype=np.float32)
        det = np.linalg.det(A)
        if abs(det) < 1e-8:
            return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        t1, t2 = np.linalg.solve(A, b)
        xi = x1 + t1 * vx1
        yi = y1 + t1 * vy1
        return float(xi), float(yi)

    # ---------- start ----------
    fname = os.path.basename(in_path)
    m = re.match(r"^(\d+)", fname)
    page_num = int(m.group(1)) if m else 0
    bad_on_left = (page_num % 2 == 1)

    img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print("Failed to read", in_path); return
    input_is_grayscale = (len(img.shape) == 2)
    if len(img.shape) == 2:
        # OpenCV expects 3-channel images for many operations
        # later convert back to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    H_img, W_img = img.shape[:2]

    # fixed
    # # FIXME wrong H_img?
    # expected_w = int(round(ASPECT * H_img))
    # expected_h = H_img

    dbgdir = os.path.join(OUTPUT_DIR, "debug", f"{page_num:03d}")
    if DEBUG: ensure_dir(dbgdir)

    gray, mask, contours = get_gray_mask_contours(img, dbgdir)

    if not contours:
        print(f"Warning: no contours found in {in_path}")
        return

    page_contour = max(contours, key=cv2.contourArea)

    if config.use_three_edge_deskew:

        top_pts, bottom_pts, outside_pts = split_edge_candidates(
            page_contour,
            bad_on_left
        )

        top_line = fit_line_ransac(top_pts)[:4]
        bottom_line = fit_line_ransac(bottom_pts)[:4]
        outside_line = fit_line_ransac(outside_pts)[:4]

        # old
        # top_angle = math.degrees(line_angle(top_line))
        # bottom_angle = math.degrees(line_angle(bottom_line))
        # outside_angle = math.degrees(line_angle(outside_line))

        # new
        top_angle = horizontal_line_angle(top_line)
        bottom_angle = horizontal_line_angle(bottom_line)
        outside_angle = vertical_line_angle(outside_line)

        if DEBUG:
            # start debug prints
            print()
            print(f"line 570: page_num={page_num}")

        rotation_error = -1 * (
            -top_angle
            -bottom_angle
        ) / 2.0

        Mrot = cv2.getRotationMatrix2D(
            (W_img/2, H_img/2),
            rotation_error,
            1.0
        )

        if DEBUG:
            print(
                f"line 575: before rotation: "
                f"top_angle={top_angle:.3f} "
                f"bottom_angle={bottom_angle:.3f} "
                f"outside_angle={outside_angle:.3f}"
            )

        rotated = cv2.warpAffine(
            img,
            Mrot,
            (W_img, H_img),
            borderValue=(255,255,255)
        )

        Hr, Wr = rotated.shape[:2]

        if DEBUG:
            print("line 580: rotated size", Wr, Hr)

        # img = rotated # ?

# re-detect lines in the rotated image

        # gray, mask, contours = get_gray_mask_contours(img, dbgdir)
        gray, mask, contours = get_gray_mask_contours(rotated, dbgdir)

        if 1:
            top_angle2 = horizontal_line_angle(top_line)
            bottom_angle2 = horizontal_line_angle(bottom_line)
            outside_angle2 = vertical_line_angle(outside_line)

            if DEBUG:
                print(
                    f"line 620: after rotation and re-fitting: "
                    f"top_angle2={top_angle2:.3f} "
                    f"bottom_angle2={bottom_angle2:.3f} "
                    f"outside_angle2={outside_angle2:.3f}"
                )

        if not contours:
            print(f"line 630: Warning: no contours found in {in_path}")
            return

        page_contour = max(contours, key=cv2.contourArea)

top_pts, bottom_pts, outside_pts = split_edge_candidates(
            page_contour,
            bad_on_left
        )

        top_line = fit_line_ransac(top_pts)[:4]
        bottom_line = fit_line_ransac(bottom_pts)[:4]
        outside_line = fit_line_ransac(outside_pts)[:4]

        # fixed
        # # FIXME H_img is wrong
        # expected_h = H_img
        # expected_w = int(round(ASPECT * expected_h))

        vx, vy, x0, y0 = outside_line

        outside_top = intersect_lines(
            outside_line,
            top_line
        )

        outside_bottom = intersect_lines(
            outside_line,
            bottom_line
        )

        if 0:
            # approximated page height
            page_height = math.dist(
                outside_top,
                outside_bottom
            )
        else:
            # perpendicular page height
            page_height = math.hypot(
                outside_bottom[0] - outside_top[0],
                outside_bottom[1] - outside_top[1]
            )

        # no. this fails to reconstruct the page height...
        # TODO try to solve this with the average height of multiple pages
        # assuming all pages must have the same height
        # also allowing the user to specify a scale_y factor
        if 0:
            # fix scan height
            # document scanners can distort scans in the Y direction
            # FIXME honor config.do_rotate
            # if (config.do_rotate == True)
            # then all pages are rotated by 90 or 270 degrees
            # so the scanner's Y errors become our X errors
            # so we need scale_x to fix the page width
            if 0:
                # use rotated_scan_y and page_height
                rotated_scan_y_px = px_of_mm(config.rotated_scan_y, config.scan_resolution)
                target_h = rotated_scan_y_px
                scale_y = target_h / page_height
            elif 1:
                # use rotated_margined_scan_y and H_img
                rotated_margined_scan_y_px = px_of_mm(config.rotated_margined_scan_y, config.scan_resolution)
                target_h = rotated_margined_scan_y_px
                scale_y = target_h / H_img
            else:
                # no, this fails because the scanner removes one edge
                # so actual_aspect is always wrong...
                #
                # use rotated_margined_scan_y and H_img
                # expected: what we ordered from the scanner
                expected_aspect = (
                    config.rotated_margined_scan_x /
                    config.rotated_margined_scan_y
                )
                expected_height = config.rotated_margined_scan_y
                # actual: what the scanner gave us
                actual_aspect = W_img / H_img
                actual_height = H_img
                # we assume the scanner always returns correct X coordinates
                # and all errors are only in Y coordinates
                # expected_aspect / actual_aspect = actual_height / expected_height
                actual_height_2 = expected_aspect / actual_aspect / expected_height
                expected_aspect_factor = expected_aspect / actual_aspect
                # the scale of actual_height relative to actual_height_2
                actual_height_scale = actual_height / actual_height_2
                if DEBUG:
                    print(f"expected_aspect={expected_aspect} actual_aspect={actual_aspect} expected_aspect_factor={expected_aspect_factor}")
                    print(f"expected_height={expected_height} actual_height={actual_height} actual_height_2={actual_height_2} actual_height_scale={actual_height_scale}")
                target_h = actual_height_2
                scale_y = target_h / H_img

            # debug: manually set the scale_y factor
            # scale_y = 1 / 1.03 # shrink the page height by 3%

            scale_y_tolerance = 0.001 # 0.1%

            if scale_y < (1 - scale_y_tolerance) or (1 + scale_y_tolerance) < scale_y:
                if DEBUG:
                    print(f"line 680: scale_y={scale_y} target_h={target_h} page_height={page_height} -> scaling height")
                rotated = cv2.resize(
                    rotated,
                    None,
                    fx=1.0,
                    fy=scale_y,
                    interpolation=cv2.INTER_CUBIC
                )
                page_height = page_height * scale_y

                # re-detect lines in the scaled image

                # gray, mask, contours = get_gray_mask_contours(img, dbgdir)
                gray, mask, contours = get_gray_mask_contours(rotated, dbgdir)

                if 1:
                    top_angle2 = horizontal_line_angle(top_line)
                    bottom_angle2 = horizontal_line_angle(bottom_line)
                    outside_angle2 = vertical_line_angle(outside_line)
                    if DEBUG:
                        print(
                            f"line 620: after rotation and re-fitting: "
                            f"top_angle2={top_angle2:.3f} "
                            f"bottom_angle2={bottom_angle2:.3f} "
                            f"outside_angle2={outside_angle2:.3f}"
                        )

                if not contours:
                    print(f"line 630: Warning: no contours found in {in_path}")
                    return

                page_contour = max(contours, key=cv2.contourArea)

                top_pts, bottom_pts, outside_pts = split_edge_candidates(
                    page_contour,
                    bad_on_left
                )

                top_line = fit_line_ransac(top_pts)[:4]
                bottom_line = fit_line_ransac(bottom_pts)[:4]
                outside_line = fit_line_ransac(outside_pts)[:4]

                # fixed
                # # FIXME H_img is wrong
                # expected_h = H_img
                # expected_w = int(round(ASPECT * expected_h))

                vx, vy, x0, y0 = outside_line

                outside_top = intersect_lines(
                    outside_line,
                    top_line
                )

                outside_bottom = intersect_lines(
                    outside_line,
                    bottom_line
                )

            else:
                if DEBUG:
                    print(f"line 680: scale_y={scale_y} target_h={target_h} page_height={page_height} -> not scaling height")

        expected_h = int(round(page_height))

        # ASPECT = x / y
        # x = y * ASPECT

        # expand the binding edge to expected_w
        # expected_w = int(round(expected_h * ASPECT))
        expected_w = int(round(page_height * ASPECT))

        if 0:
            # crop as a rectangle
            x_out = (
                outside_top[0]
                +
                outside_bottom[0]
            ) / 2

            x_out = int(round(x_out))

            # if bad_on_left:
            #     x0_page = x_out
            #     x1_page = x_out + expected_w
            # else:
            #     x0_page = x_out - expected_w
            #     x1_page = x_out

            if bad_on_left:
                # outside edge is RIGHT
                x1_page = x_out
                x0_page = x_out - expected_w
            else:
                # outside edge is LEFT
                x0_page = x_out
                x1_page = x_out + expected_w

            # clamp
            # x0_page = max(0, x0_page)
            # x1_page = min(rotated.shape[1], x1_page)

            y_top = int(round(outside_top[1]))
            y_bottom = int(round(outside_bottom[1]))

            src_x0 = max(0, x0_page)
            src_x1 = min(rotated.shape[1], x1_page)

            src_y0 = max(0, y_top)
            src_y1 = min(rotated.shape[0], y_top + expected_h)

            dst_x0 = src_x0 - x0_page
            dst_y0 = src_y0 - y_top

            if DEBUG:
                print(
                    "line 640:",
                    f"outside_top={outside_top}",
                    f"outside_bottom={outside_bottom}",
                )

            # crop = rotated[
            #     y_top:y_top+expected_h,
            #     x0_page:x1_page
            # ]

            full_crop = np.ones(
                (expected_h, expected_w, 3),
                dtype=rotated.dtype
            ) * 255

            full_crop[
                dst_y0:dst_y0 + (src_y1-src_y0),
                dst_x0:dst_x0 + (src_x1-src_x0)
            ] = rotated[
                src_y0:src_y1,
                src_x0:src_x1
            ]

            # FIXME fill empty area near binding edge
            # currently it is all white
            # but it should copy the vertical pattern near the binding edge

            crop = full_crop

            if DEBUG:
                print(
                    "line 650: crop",
                    f"W_img={W_img}",
                    f"H_img={H_img}",
                    f"x0_page={x0_page}",
                    f"x1_page={x1_page}",
                    f"y_top={y_top}",
                    f"y_bottom={y_bottom}",
                    f"src_x0={src_x0}",
                    f"src_x1={src_x1}",
                    f"src_y0={src_y0}",
                    f"src_y1={src_y1}",
                    f"dst_x0={dst_x0}",
                    f"dst_y0={dst_y0}",
                    f"expected_w={expected_w}",
                    f"expected_h={expected_h}",
                )

            if DEBUG:
                vis = rotated.copy()
                cv2.rectangle(
                    vis,
                    (int(x0_page), int(y_top)),
                    (int(x1_page), int(y_bottom)),
                    (0,0,255),
                    3,
                )
                name = f"{page_num:03d}.crop_debug_rotated_line730.jpg"
                print(f"writing {name}")
                cv2.imwrite(name, vis)

        else:
            # crop as a quadrilateral
            # not better than "crop as a rectangle"?
            if 0:
                # expand the binding edge to expected_w
                if bad_on_left:
                    # outside edge is the RIGHT edge
                    x1_top = outside_top[0]
                    x1_bottom = outside_bottom[0]
                    x0_top = x1_top - expected_w
                    x0_bottom = x1_bottom - expected_w
                else:
                    # outside edge is the LEFT edge
                    x0_top = outside_top[0]
                    x0_bottom = outside_bottom[0]
                    x1_top = x0_top + expected_w
                    x1_bottom = x0_bottom + expected_w
            else:
                # dont expand the binding edge to expected_w
                # use only the detected page edges
                # if bad_on_left:
                #     # outside edge is RIGHT edge
                #     x1_top = outside_top[0]
                #     x1_bottom = outside_bottom[0]
                #     # use the actual detected left edge
                #     x0_top = np.min(page_contour[:,0,0])
                #     x0_bottom = x0_top
                # else:
                #     # outside edge is LEFT edge
                #     x0_top = outside_top[0]
                #     x0_bottom = outside_bottom[0]
                #     # use the actual detected right edge
                #     x1_top = np.max(page_contour[:,0,0])
                #     x1_bottom = x1_top
                # problem: page_contour after thresholding may include the background
                # or may not have a reliable missing-edge position.
                # A cleaner temporary solution is to use the detected quadrilateral width
                # from the two horizontal edge intersections
                if bad_on_left:
                    # outside edge is right
                    x1_top = outside_top[0]
                    x1_bottom = outside_bottom[0]
                    # find leftmost detected page boundary
                    x0_top = np.min(top_pts[:,0])
                    x0_bottom = np.min(bottom_pts[:,0])
                else:
                    # outside edge is left
                    x0_top = outside_top[0]
                    x0_bottom = outside_bottom[0]
                    # find rightmost detected page boundary
                    x1_top = np.max(top_pts[:,0])
                    x1_bottom = np.max(bottom_pts[:,0])

                # dont expand the binding edge to expected_w
                expected_w = int(round(
                    math.dist((x0_top, outside_top[1]), (x1_top, outside_top[1]))
                ))

src = np.float32([
                [x0_top, outside_top[1]],
                [x1_top, outside_top[1]],
                [x1_bottom, outside_bottom[1]],
                [x0_bottom, outside_bottom[1]],
            ])

            dst = np.float32([
                [0,0],
                [expected_w-1,0],
                [expected_w-1,expected_h-1],
                [0,expected_h-1],
            ])

            M = cv2.getPerspectiveTransform(src, dst)

            crop = cv2.warpPerspective(
                rotated,
                M,
                (expected_w, expected_h),
                borderValue=(255,255,255)
            )

            if DEBUG:
                print(
                    "line 650: crop",
                    f"W_img={W_img}",
                    f"H_img={H_img}",
                    f"outside_top={outside_top}",
                    f"outside_bottom={outside_bottom}",
                    f"x1_top={x1_top}",
                    f"x1_bottom={x1_bottom}",
                    f"x0_top={x0_top}",
                    f"x0_bottom={x0_bottom}",
                    f"expected_w={expected_w}",
                    f"expected_h={expected_h}",
                )

            if DEBUG:
                vis = rotated.copy()
                pts = np.array([
                    [x0_top, outside_top[1]],
                    [x1_top, outside_top[1]],
                    [x1_bottom, outside_bottom[1]],
                    [x0_bottom, outside_bottom[1]],
                ], np.int32)
                cv2.polylines(vis, [pts], True, (0,0,255), 3)
                name = f"{page_num:03d}.crop_debug_rotated_line730.jpg"
                print(f"writing {name}")
                cv2.imwrite(name, vis)

        # crop = rotated[
        #     0:expected_h,
        #     int(x0_page):int(x1_page)
        # ]

        # y_top = round(outside_top[1])
        # y_bottom = round(outside_bottom[1])
        # actual_height = y_bottom - y_top
        # crop = rotated[
        #     y_top:y_top + expected_h,
        #     x0_page:x1_page
        # ]

        # y_top = int(round(outside_top[1]))
        # y_bottom = int(round(outside_bottom[1]))
        # crop = rotated[
        #     y_top:y_bottom,
        #     x0_page:x1_page
        # ]

        if 0:
            # img = repair_binding(img, bad_on_left, width=50)
            if 1:
                # rotated = repair_binding(rotated, bad_on_left, width=50)
                # Hr, Wr = rotated.shape[:2]
                # print("line 660: rotated size", Wr, Hr)
                # warped = rotated
                crop = repair_binding(crop, bad_on_left, width=50)
                Hr, Wr = crop.shape[:2]
                if DEBUG:
                    print("line 660: crop size", Wr, Hr)
            else:
                crop = repair_binding(crop, bad_on_left, width=50)

            if DEBUG:
                print(
                    "line 760: after repair_binding: crop actual:",
                    f"crop.shape[1]={crop.shape[1]}",
                    f"expected_w={expected_w}",
                )

        warped = crop

    else:
        # config.use_three_edge_deskew == False
        # FIXME use only two page edges: outside, bottom
        # Approximate contour to quadrilateral
        epsilon = 0.02 * cv2.arcLength(page_contour, True)
        approx = cv2.approxPolyDP(page_contour, epsilon, True)
        if len(approx) != 4:
            approx = cv2.convexHull(page_contour)
            if len(approx) < 4:
                print(f"Warning: not enough points for perspective in {in_path}")
                return
            # pick 4 extreme points
            pts = np.array([
                approx[approx[:,0,0].argmin()][0],  # leftmost
                approx[approx[:,0,1].argmin()][0],  # topmost
                approx[approx[:,0,0].argmax()][0],  # rightmost
                approx[approx[:,0,1].argmax()][0]   # bottommost
            ])
        else:
            pts = approx.reshape(4,2)

        rect = order_points(pts)

        # Perspective transform
        widthA = np.linalg.norm(rect[2] - rect[3])
        widthB = np.linalg.norm(rect[1] - rect[0])
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.linalg.norm(rect[1] - rect[2])
        heightB = np.linalg.norm(rect[0] - rect[3])
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    # # Add internal white border
    # h, w = warped.shape[:2]
    # canvas = np.ones_like(warped) * 255
    # b = BORDER_SIZE
    # canvas[b:h-b, b:w-b] = warped[b:h-b, b:w-b]

    h, w = warped.shape[:2]
    b = BORDER_SIZE

    # Prepare a canvas with the same content as warped
    canvas = warped.copy()

    # Function to compute average color along a strip
    def avg_color_strip(img, axis, start, end, strip_width=1):
        if axis == 'top':
            strip = img[start:end, :, :]
        elif axis == 'bottom':
            strip = img[h-end:h-start, :, :]
        elif axis == 'left':
            strip = img[:, start:end, :]
        elif axis == 'right':
            strip = img[:, w-end:w-start, :]
        else:
            raise ValueError("Invalid axis")
        return np.mean(strip, axis=(0,1)).astype(np.uint8)

    # FIXME preserve patterns near edges
    if 0:
        # Fill borders with local average color
        canvas[0:b, :, :] = avg_color_strip(canvas, 'top', 50, 100)       # top border
        canvas[h-b:h, :, :] = avg_color_strip(canvas, 'bottom', 50, 100)  # bottom border
        canvas[:, 0:b, :] = avg_color_strip(canvas, 'left', 50, 100)      # left border
        canvas[:, w-b:w, :] = avg_color_strip(canvas, 'right', 50, 100)   # right border

    if input_is_grayscale:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

    ensure_dir(os.path.dirname(out_path))
    cv2.imwrite(out_path, canvas)
    print(f"writing {out_path}")

def main():
    ensure_dir(OUTPUT_DIR)
    # TODO use image_format from 030-measure-page-size.txt
    image_format = "jpg"
    files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(f".{image_format}")])
    if not files:
        print("No image files found in", INPUT_DIR)
        return
    for fname in files:
        m = re.match(r"^(\d+)", fname)
        # page_num = int(m.group(1)) if m else 0
        page_num = int(m.group(1)) # can throw
        # if page_num != 14: continue # debug
        # if not page_num in [340, 345]: continue # debug
        # if page_num < 300: continue # debug
        # if page_num != 320: continue # debug
        in_path = os.path.join(INPUT_DIR, fname)
        out_path = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(out_path):
            continue
        try:
            process_image(in_path, out_path)
        except Exception as exc:
            print(f"Error processing {fname}: {exc}")
            raise

if __name__ == "__main__":
    main()
```

## Response:

7/12/2026, 7:21:48 PM

Yes — your correction is right. The old reasoning around `scale_y` is backwards **because your coordinate system after rotation is not the scanner coordinate system**.

Your scanner errors happen in **scanner Y**. Since the raw scans are already rotated 90°/270°:

- scanner **Y** → image **X**
- scanner **X** → image **Y**

So the dimension that needs correction is **width**, not height.

The block around line ~655 should become a **scale_x correction**, and the later `expected_w = page_height * ASPECT` logic should use the corrected width.

### Replace your whole `if 0:` "fix scan height" block with this

``` python
        # fix scan width
        #
        # IMPORTANT:
        # Raw scans are rotated by 90/270 degrees.
        #
        # Therefore:
        #   scanner Y errors -> image X errors
        #
        # We must correct width (scale_x), not height (scale_y).

        if 1:
            # The detected outside edge gives us the real page height
            # in image coordinates.  After rotation this corresponds to
            # scanner X, which is assumed correct.
            #
            # The missing/damaged dimension is the page width.

            rotated_margined_scan_x_px = px_of_mm(
                config.rotated_margined_scan_x,
                config.scan_resolution
            )

            target_w = rotated_margined_scan_x_px

            actual_w = math.dist(
                outside_top,
                outside_bottom
            )

            # This is actually the width in scanner coordinates because
            # the document is rotated.
            scale_x = target_w / actual_w

            scale_x_tolerance = 0.001  # 0.1%

            if (
                scale_x < (1 - scale_x_tolerance)
                or
                scale_x > (1 + scale_x_tolerance)
            ):
                if DEBUG:
                    print(
                        "line 680:",
                        f"scale_x={scale_x}",
                        f"target_w={target_w}",
                        f"actual_w={actual_w}",
                        "-> scaling width"
                    )

                rotated = cv2.resize(
                    rotated,
                    None,
                    fx=scale_x,
                    fy=1.0,
                    interpolation=cv2.INTER_CUBIC
                )

                # Update image size
                Hr, Wr = rotated.shape[:2]

                if DEBUG:
                    print(
                        "line 690: after scale_x",
                        f"Wr={Wr}",
                        f"Hr={Hr}"
                    )

                # Re-detect page after scaling
                gray, mask, contours = get_gray_mask_contours(
                    rotated,
                    dbgdir
                )

                if not contours:
                    print(
                        f"line 700: Warning: no contours found after scaling {in_path}"
                    )
                    return

                page_contour = max(
                    contours,
                    key=cv2.contourArea
                )

                top_pts, bottom_pts, outside_pts = split_edge_candidates(
                    page_contour,
                    bad_on_left
                )

                top_line = fit_line_ransac(top_pts)[:4]
                bottom_line = fit_line_ransac(bottom_pts)[:4]
                outside_line = fit_line_ransac(outside_pts)[:4]

                outside_top = intersect_lines(
                    outside_line,
                    top_line
                )

                outside_bottom = intersect_lines(
                    outside_line,
                    bottom_line
                )

            else:
                if DEBUG:
                    print(
                        "line 680:",
                        f"scale_x={scale_x}",
                        "-> not scaling width"
                    )
```

Then leave:

``` python
expected_h = int(round(page_height))
expected_w = int(round(page_height * ASPECT))
```

as-is.

------------------------------------------------------------------------

### Why this fixes the earlier failure

Your previous code:

``` python
scale_y = target_h / H_img
```

was assuming:

    scanner Y == image Y

but your actual pipeline is:

    scanner image
         |
         | rotate 90/270
         v

    image X = scanner Y
    image Y = scanner X

Therefore the distortion you were trying to remove was sitting in `W_img`, not `H_img`.

------------------------------------------------------------------------

One more subtle bug: after the `cv2.resize(... fx=scale_x)` you correctly need to **recompute all intersections** (as above). Scaling the image invalidates:

``` python
outside_top
outside_bottom
top_line
bottom_line
outside_line
```

because their coordinates are in the old image space.

This should get you much closer. The remaining thing I would watch is whether `rotated_margined_scan_x` includes the missing binding-edge area or only the visible page width — that determines whether `target_w` should be based on the scanner format or on the detected page aspect.

## Prompt:

7/13/2026, 7:34:04 AM

ok, now i have multiple edge cases where the script fails to detect one or more page edges

remember: my scanner adds a midgray (50% gray) background behind each page, so the actual page edges must always be surrounded by gray (except the scanner's top edge, which when rotated becomes the binding edge)

## Response:

7/13/2026, 7:34:04 AM

Yes, and that gray background is a very strong prior. It changes the problem from "find a page" to "find the white→gray transition."

The important geometric fact is:

- **outside edge** → white page next to gray background
- **top edge** → white page next to gray background
- **bottom edge** → white page next to gray background
- **binding edge** → may disappear into the scanner edge after rotation (no gray there)

So, after rotation:

    gray
    ───────────────
    │             │
    │   PAGE      │ gray
    │             │
    ───────────────
    binding edge (scanner edge)

There should only ever be **one edge that is allowed to be missing**: the binding edge.

------------------------------------------------------------------------

## I'd stop fitting contours

Right now your pipeline is approximately

    threshold
        ↓
    largest contour
        ↓
    split into top/bottom/outside
        ↓
    RANSAC

This assumes the contour is already correct.

But the contour is exactly what breaks when

- the page has a dark illustration
- the page touches the scanner edge
- the gray background is noisy
- the page corner is torn off
- shadows exist

Instead, exploit the gray background directly.

------------------------------------------------------------------------

## Detect edges independently

Treat each edge as an independent detection problem.

For example, for the top edge:

For every column

    gray
    gray
    gray
    gray
    white page
    white page
    white page

Find the first

    gray -> white

transition.

This gives one point.

Repeat for every x.

You now have hundreds of candidate points.

Run RANSAC.

Exactly the same for the bottom edge.

------------------------------------------------------------------------

For the outside edge:

For every row

    gray gray gray page page page

find the transition

    gray -> white

Again you get one point per row.

Run RANSAC.

No contours required.

------------------------------------------------------------------------

## Why this is much more robust

Suppose this happens:

    ##############
    ######image###
    ##############

inside the page.

Contours become horrible.

Column scanning doesn't care:

    gray
    gray
    gray
    white
    black
    black
    white

The first gray→page transition is still correct.

------------------------------------------------------------------------

## Missing corners become irrelevant

Suppose the top-left corner is folded.

    xxxxxxxxxxxx
    xxxxx#######
    gray page###
    gray page###

Only a few columns fail.

RANSAC ignores them.

------------------------------------------------------------------------

## Missing entire edge

Suppose the bottom edge is hidden.

Then the detector simply returns

    15 points

instead of

    2000 points

You immediately know

> bottom edge confidence = low

instead of forcing RANSAC onto garbage.

------------------------------------------------------------------------

## The gray background also lets you score confidence

For every detected edge pixel, compute

    left side average
    right side average

(or above/below for horizontal edges).

A real page edge should satisfy approximately

    gray ≈ 128

    page ≈ 220–255

If both sides are white

    245
    248

that's not an edge.

If both sides are gray

    128
    132

that's not an edge either.

You can reject those samples before RANSAC.

------------------------------------------------------------------------

## I'd redesign the detector like this

    rotate

    ↓

    grayscale

    ↓

    detect page→gray transitions
        top
        bottom
        outside
        binding(optional)

    ↓

    RANSAC each independently

    ↓

    confidence score for each edge

    ↓

    if confidence high:
        use line

    if confidence low:
        infer from
            aspect ratio
            opposite edge
            page height

That removes the dependence on `findContours()` almost entirely.

Given your scanner always provides a uniform mid-gray surround, this transition-based approach is likely to be substantially more reliable than contour extraction, because it uses the strongest invariant in your data: every real page edge (except the scanner-edge/binding side) is defined by a gray-to-page intensity transition.

## Prompt:

7/13/2026, 7:37:46 AM

> I'd redesign the detector like this

ok, do it!

## Response:

7/13/2026, 7:37:46 AM

I think this is the right direction. I'd actually simplify it even further and get rid of contours almost entirely.

## Step 1: Segment page vs. gray background

Since your scanner background is always around 50% gray, don't use a high-percentile threshold.

Instead:

``` python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Otsu is usually sufficient here
_, page_mask = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# Page should be white
if np.mean(gray[page_mask == 255]) < np.mean(gray[page_mask == 0]):
    page_mask = cv2.bitwise_not(page_mask)

page_mask = cv2.morphologyEx(
    page_mask,
    cv2.MORPH_CLOSE,
    np.ones((5,5), np.uint8)
)
```

------------------------------------------------------------------------

# Step 2: Scan for transitions

Instead of contours, build three point clouds.

## Top edge

``` python
def detect_top_points(mask):
    h, w = mask.shape
    pts = []

    for x in range(w):

        col = mask[:, x]

        ys = np.where(col == 255)[0]

        if len(ys) == 0:
            continue

        y = ys[0]

        pts.append((x, y))

    return np.asarray(pts, np.float32)
```

------------------------------------------------------------------------

## Bottom edge

``` python
def detect_bottom_points(mask):
    h, w = mask.shape
    pts = []

    for x in range(w):

        col = mask[:, x]

        ys = np.where(col == 255)[0]

        if len(ys) == 0:
            continue

        y = ys[-1]

        pts.append((x, y))

    return np.asarray(pts, np.float32)
```

------------------------------------------------------------------------

## Left edge

``` python
def detect_left_points(mask):
    h, w = mask.shape
    pts = []

    for y in range(h):

        row = mask[y]

        xs = np.where(row == 255)[0]

        if len(xs) == 0:
            continue

        pts.append((xs[0], y))

    return np.asarray(pts, np.float32)
```

------------------------------------------------------------------------

## Right edge

``` python
def detect_right_points(mask):
    h, w = mask.shape
    pts = []

    for y in range(h):

        row = mask[y]

        xs = np.where(row == 255)[0]

        if len(xs) == 0:
            continue

        pts.append((xs[-1], y))

    return np.asarray(pts, np.float32)
```

Then simply choose

``` python
if bad_on_left:
    outside_pts = detect_right_points(mask)
else:
    outside_pts = detect_left_points(mask)

top_pts = detect_top_points(mask)
bottom_pts = detect_bottom_points(mask)
```

Your existing `fit_line_ransac()` can remain exactly as it is.

------------------------------------------------------------------------

# Step 3: Add transition verification

This is the important improvement.

The first white pixel isn't always a page edge. It might be noise.

Verify that every candidate actually crosses gray → page.

For example:

``` python
def is_vertical_transition(gray, x, y):

    if x < 3 or x >= gray.shape[1]-3:
        return False

    left = np.mean(gray[y, x-3:x])
    right = np.mean(gray[y, x:x+3])

    return abs(left-right) > 40
```

For horizontal edges:

``` python
top = np.mean(gray[y-3:y, x])
bottom = np.mean(gray[y:y+3, x])
```

Reject points that don't have sufficient contrast.

------------------------------------------------------------------------

# Step 4: Reject isolated outliers

Even after transition checking you'll occasionally get

    ..........
    .....x....
    ..........

Use a median filter before RANSAC.

Example:

``` python
ys = pts[:,1]

median = np.median(ys)

pts = pts[
    np.abs(ys - median) < 40
]
```

For vertical edges do the same with x.

This removes almost all garbage.

------------------------------------------------------------------------

# Step 5: Feed directly into your existing code

Nothing below this changes.

``` python
top_line = fit_line_ransac(top_pts)[:4]
bottom_line = fit_line_ransac(bottom_pts)[:4]
outside_line = fit_line_ransac(outside_pts)[:4]
```

Everything after that (rotation, intersection, perspective transform, binding repair) stays the same.

------------------------------------------------------------------------

## One refinement I'd recommend

Instead of maintaining four nearly identical functions (`detect_top_points`, `detect_bottom_points`, etc.), write a single generic scanner:

``` python
def detect_edge_points(mask, direction):
    ...
```

where `direction` is `"top"`, `"bottom"`, `"left"`, or `"right"`.

That reduces duplication and makes it easier to add common logic like transition verification, noise rejection, and confidence scoring in one place. Since your pipeline is already quite large, consolidating the edge detection into one reusable function will make future tuning much easier.

## Prompt:

7/13/2026, 7:51:06 AM

ok, now my script is this.... can you fill in the blanks around line 620 (Step 3: Transition verification)

``` py
#!/usr/bin/env python3
"""
extract scanned page from gray background

restore the binding edge of the page
by filling the missing width
with the average color near the binding edge
"""

INPUT_DIR = "060-rotate-crop"
OUTPUT_DIR = "065-remove-page-borders"

# === Tuning parameters ===
DEBUG = True
# DEBUG = False
BORDER_SIZE = 100  # pixels

RANSAC_ITER = 400
RANSAC_INLIER_DIST = 6.0
# RANSAC_MIN_INLIERS = 30
RANSAC_MIN_INLIERS = 20

THRESH_HIGH_PERCENTILE = 99
THRESH_MIN = 200

import os
import re
import math
import random
import numpy as np
import cv2

from _shared import (
    load_config,
    get_page_num,
)

config = load_config()

# no, this is wrong if (config.do_rotate == True)
# scan_x = config.scan_x
# scan_y = config.scan_y
# config.scan_aspect = scan_x / scan_y
# ASPECT = config.scan_aspect

ASPECT = config.rotated_scan_aspect

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def save_dbg(img, path):
    ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, img)

def percentile_threshold(gray):
    high_p = np.percentile(gray, THRESH_HIGH_PERCENTILE)
    thr = max(THRESH_MIN, int(high_p * 0.95))
    _, mask = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
    return mask, thr, int(high_p)

def detect_vertical_streaks(mask, approx_width=3, length_thresh_ratio=0.15):
    h, w = mask.shape
    kx = approx_width
    ky = max(15, int(h * 0.02))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
    long_vertical = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(long_vertical, connectivity=8)
    streak_mask = np.zeros_like(mask)
    length_thresh = max(10, int(h * length_thresh_ratio))
    for i in range(1, num_labels):
        x, y, ww, hh, area = stats[i]
        if hh >= length_thresh and ww <= max(5, int(w * 0.01)):
            streak_mask[labels == i] = 255
    return streak_mask

def keep_largest_component(mask):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    out = np.zeros_like(mask)
    out[labels == best] = 255
    return out

def contour_to_pts(contour):
    return contour.reshape(-1,2)

def fit_line_ransac(pts, iterations=RANSAC_ITER, inlier_dist=RANSAC_INLIER_DIST, min_inliers=RANSAC_MIN_INLIERS):
    if len(pts) < 2:
        raise ValueError("Not enough points")
    best_inliers = None
    best_model = None
    n = len(pts)
    ptsf = pts.astype(np.float32)
    for _ in range(iterations):
        i1, i2 = random.sample(range(n), 2)
        p1 = ptsf[i1]; p2 = ptsf[i2]
        vx = float(p2[0] - p1[0]); vy = float(p2[1] - p1[1])
        if vx == 0 and vy == 0:
            continue
        dists = np.abs(vy*(ptsf[:,0]-p1[0]) - vx*(ptsf[:,1]-p1[1])) / (math.hypot(vx, vy) + 1e-12)
        inliers = dists <= inlier_dist
        cnt = int(inliers.sum())
        if cnt >= min_inliers and (best_inliers is None or cnt > int(best_inliers.sum())):
            best_inliers = inliers.copy()
            best_model = (vx, vy, float(p1[0]), float(p1[1]))
    if best_model is None:
        vx, vy, x0, y0 = cv2.fitLine(ptsf, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        inlier_mask = np.ones(len(pts), dtype=bool)
        return float(vx), float(vy), float(x0), float(y0), inlier_mask
    inlier_pts = ptsf[best_inliers]
    vx, vy, x0, y0 = cv2.fitLine(inlier_pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    inlier_mask = best_inliers
    return float(vx), float(vy), float(x0), float(y0), inlier_mask

def intersect_lines(l1, l2):
    vx1, vy1, x1, y1 = l1
    vx2, vy2, x2, y2 = l2
    A = np.array([[vx1, -vx2], [vy1, -vy2]], dtype=np.float32)
    b = np.array([x2 - x1, y2 - y1], dtype=np.float32)
    det = np.linalg.det(A)
    if abs(det) < 1e-8:
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    t1, t2 = np.linalg.solve(A, b)
    xi = x1 + t1 * vx1
    yi = y1 + t1 * vy1
    return float(xi), float(yi)

def build_affine_without_shear(pt_v_top, pt_v_bot, expected_w, expected_h, bad_on_left, img, dbgdir=None):
    """
    Build a source triangle that enforces orthogonal page axes (no shear) using:
      - pt_v_top: intersection of top line and good vertical (top-right when good vertical is right)
      - pt_v_bot: intersection of bottom line and good vertical (bottom-right when good vertical is right)
    Returns (M_aff, src_corners_dict, dst_corners_dict, reason)
    - M_aff: 2x3 affine matrix or None on failure
    - src_corners_dict: dict of TL, TR, BL, BR (np.float32 2-vectors)
    - dst_corners_dict: dict of TL, TR, BL, BR in destination coordinates
    - reason: diagnostic string
    """
    # convert to vectors
    pt_v_top = np.array(pt_v_top, dtype=np.float32)
    pt_v_bot = np.array(pt_v_bot, dtype=np.float32)

    # compute direction along top edge: vector from bottom intersection to top intersection projected horizontally
    # but we don't have explicit top_line direction vector here; we will approximate using (pt_v_top - pt_v_bot) rotated?
    # Better: caller should supply top_line direction (vx,vy). If you don't have it, approximate from nearby contour points.
    # For compatibility with your code, expect you have top_vx, top_vy; if not, fall back to vector between two top contour points.
    # Here we'll assume top_vx, top_vy are available in outer scope; otherwise compute from pt_v_top->pt_v_bot displacement projected perpendicular:
    # To keep this helper self-contained, we compute top_dir as average of (pt_v_top_to_some_right) - but simplest robust approach:
    # compute approximate top direction by taking small offset vector along top by sampling image gradient: fallback to (1,0).

    # --- Here caller should provide top_unit; we'll compute a safe top_unit using neighbor pixels if available ---
    # We'll attempt to derive a top_unit by sampling a short step along the image: use vector from pt_v_top to pt_v_top projected to image center
    cx, cy = img.shape[1] / 2.0, img.shape[0] / 2.0
    # prefer vector pointing rightwards by projecting displacement to center
    approx = pt_v_top - np.array([cx, cy], dtype=np.float32)
    if np.linalg.norm(approx) < 1e-6:
        approx = np.array([1.0, 0.0], dtype=np.float32)
    ux, uy = approx / (np.linalg.norm(approx) + 1e-12)

    # Force ux positive (pointing to the right) — we want u to be rightward
    if ux < 0:
        ux, uy = -ux, -uy

    # make perpendicular downwards: p = (-uy, ux)
    px, py = -uy, ux

    # normalize again (just in case)
    n_u = math.hypot(ux, uy)
    n_p = math.hypot(px, py)
    if n_u < 1e-6 or n_p < 1e-6:
        return None, None, None, "degenerate_axes"
    ux, uy = ux / n_u, uy / n_u
    px, py = px / n_p, py / n_p

    # Now construct corners. We know pt_v_top/pt_v_bot correspond to the known vertical edge:
    # If bad_on_left is True, good vertical is the RIGHT edge => pt_v_top == TR, pt_v_bot == BR.
    # If bad_on_left is False, good vertical is LEFT edge => pt_v_top == TL, pt_v_bot == BL.
    if bad_on_left:
        src_TR = pt_v_top
        src_BR = pt_v_bot
        # compute TL and BL by moving left along top unit by expected_w
        src_TL = src_TR - np.array([ux, uy], dtype=np.float32) * float(expected_w)
        src_BL = src_BR - np.array([ux, uy], dtype=np.float32) * float(expected_w)
    else:
        src_TL = pt_v_top
        src_BL = pt_v_bot
        # compute TR and BR by moving right along top unit by expected_w
        src_TR = src_TL + np.array([ux, uy], dtype=np.float32) * float(expected_w)
        src_BR = src_BL + np.array([ux, uy], dtype=np.float32) * float(expected_w)

    # Optional: enforce vertical height by projecting (TL->BL) onto perp and rescale to expected_h
    # Compute current vertical vector and its projection onto perp (px,py)
    cur_v = src_BL - src_TL
    proj = cur_v.dot(np.array([px, py], dtype=np.float32))
    # if projection is tiny, fall back but otherwise adjust BL to ensure exact page height
    if abs(proj) > 1e-6:
        # adjust scale to exact expected_h
        scale = float(expected_h) / proj
        # recompute BL, BR as TL + perp*expected_h and TR + perp*expected_h
        src_BL = src_TL + np.array([px, py], dtype=np.float32) * float(expected_h)
        src_BR = src_TR + np.array([px, py], dtype=np.float32) * float(expected_h)
    else:
        # if proj nearly zero, we cannot rely on vertical measurement; still use perpendicular step
        src_BL = src_TL + np.array([px, py], dtype=np.float32) * float(expected_h)
        src_BR = src_TR + np.array([px, py], dtype=np.float32) * float(expected_h)

    # Build src/dst dicts
    src_corners = {
        "TL": src_TL.astype(np.float32),
        "TR": src_TR.astype(np.float32),
        "BL": src_BL.astype(np.float32),
        "BR": src_BR.astype(np.float32)
    }
    dst_corners = {
        "TL": np.array([0.0, 0.0], dtype=np.float32),
        "TR": np.array([expected_w - 1.0, 0.0], dtype=np.float32),
        "BL": np.array([0.0, expected_h - 1.0], dtype=np.float32),
        "BR": np.array([expected_w - 1.0, expected_h - 1.0], dtype=np.float32)
    }

    # choose triangle for affine depending on orientation (map TL, BL, TR -> dst TL, BL, TR)
    src_tri = np.vstack([src_corners["TL"], src_corners["BL"], src_corners["TR"]]).astype(np.float32)
    dst_tri = np.vstack([dst_corners["TL"], dst_corners["BL"], dst_corners["TR"]]).astype(np.float32)

    # validate triangle
    area = abs(0.5 * (src_tri[0,0]*(src_tri[1,1]-src_tri[2,1]) + src_tri[1,0]*(src_tri[2,1]-src_tri[0,1]) + src_tri[2,0]*(src_tri[0,1]-src_tri[1,1])))
    if area < 1.0 or not np.isfinite(src_tri).all():
        return None, src_corners, dst_corners, f"invalid_src_tri_area={area:.3f}"

    try:
        M = cv2.getAffineTransform(src_tri, dst_tri)
    except cv2.error as e:
        return None, src_corners, dst_corners, f"getAffineTransform_failed:{e}"

    # debug overlay: draw corners & axes if dbgdir provided
    if dbgdir is not None:
        vis = img.copy()
        def draw_pt(pt, color=(0,0,255), tag=""):
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 6, color, -1)
            if tag:
                cv2.putText(vis, tag, (int(pt[0]+6), int(pt[1]-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        draw_pt(src_corners["TL"], (0,255,0), "TL")
        draw_pt(src_corners["TR"], (0,128,255), "TR")
        draw_pt(src_corners["BL"], (255,0,0), "BL")
        draw_pt(src_corners["BR"], (255,255,0), "BR")
        # draw axis arrows from TL
        origin = src_corners["TL"]
        arrow_u = origin + np.array([ux, uy], dtype=np.float32) * 80.0
        arrow_p = origin + np.array([px, py], dtype=np.float32) * 80.0
        cv2.arrowedLine(vis, (int(origin[0]), int(origin[1])), (int(arrow_u[0]), int(arrow_u[1])), (255,0,255), 2)
        cv2.arrowedLine(vis, (int(origin[0]), int(origin[1])), (int(arrow_p[0]), int(arrow_p[1])), (0,255,255), 2)
        cv2.imwrite(os.path.join(dbgdir, "debug_axes_and_corners.png"), vis)

    return M, src_corners, dst_corners, "ok"

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

def split_edge_candidates(contour, bad_on_left):
    pts = contour.reshape(-1, 2)

    xs = pts[:,0]
    ys = pts[:,1]

    w = xs.max() - xs.min()
    h = ys.max() - ys.min()

    margin_x = w * 0.15
    margin_y = h * 0.15

    # outside edge
    if bad_on_left:
        outside = pts[xs > xs.max() - margin_x]
    else:
        outside = pts[xs < xs.min() + margin_x]

    # top edge
    top = pts[ys < ys.min() + margin_y]

    # bottom edge
    bottom = pts[ys > ys.max() - margin_y]

    return top, bottom, outside

def line_angle(line):
    vx, vy, _, _ = line
    return math.atan2(vy, vx)

def horizontal_line_angle(line):
    vx, vy, _, _ = line

    a = math.degrees(math.atan2(vy, vx))

    while a < -45:
        a += 180

    while a > 45:
        a -= 180

    return a

def vertical_line_angle(line):
    vx, vy, _, _ = line

    a = math.degrees(math.atan2(vy, vx))

    while a < 45:
        a += 180

    while a > 135:
        a -= 180

    return a

def normalize_angle_deg(a):
    while a < -90:
        a += 180
    while a > 90:
        a -= 180
    return a

def get_gray_mask_contours(img, dbgdir):
    # --- threshold for geometry only ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask_init, thr, hp = percentile_threshold(gray)
    if cv2.mean(gray)[0] < 127:
        mask_init = cv2.bitwise_not(mask_init)
    if DEBUG:
        save_dbg(gray, os.path.join(dbgdir, "01_gray.png"))
        save_dbg(mask_init, os.path.join(dbgdir, f"02_thresh_thr{thr}_hp{hp}.png"))

    # without a9dc2b24e6b0d1d49b6fc232223d6431ba3442a5 bad: fix perspective transform for broken ADF scanners
    # Invert if necessary, so the page is always "white" for thresholding
    # We'll use Otsu's method to detect high-contrast area
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Determine whether the page is darker or lighter than background
    mean_val = cv2.mean(gray, mask=None)[0]
    if mean_val < 127:  # dark page: invert mask
        mask = cv2.bitwise_not(mask)

    # Morphology to remove small gaps
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return gray, mask, contours

def repair_binding(img, bad_on_left, width=50):

    h,w = img.shape[:2]

    if bad_on_left:
        sample_range = range(width)
        fill_range = range(width)
        sample_x = width
    else:
        sample_range = range(w-width,w)
        fill_range = range(w-width,w)
        sample_x = w-width

    for y in range(h):

        if bad_on_left:
            sample = img[y, width:width+20]
            color = np.mean(sample,axis=0)
            img[y,:width] = color

        else:
            sample = img[y,w-width-20:w-width]
            color = np.mean(sample,axis=0)
            img[y,w-width:] = color

    return img

def px_of_mm(mm, dpi):
    return mm * dpi / 25.4

def detect_top_points(mask):
    h, w = mask.shape
    pts = []
    for x in range(w):
        col = mask[:, x]
        ys = np.where(col == 255)[0]
        if len(ys) == 0:
            continue
        y = ys[0]
        pts.append((x, y))
    return np.asarray(pts, np.float32)

def detect_bottom_points(mask):
    h, w = mask.shape
    pts = []
    for x in range(w):
        col = mask[:, x]
        ys = np.where(col == 255)[0]
        if len(ys) == 0:
            continue
        y = ys[-1]
        pts.append((x, y))
    return np.asarray(pts, np.float32)

def detect_left_points(mask):
    h, w = mask.shape
    pts = []
    for y in range(h):
        row = mask[y]
        xs = np.where(row == 255)[0]
        if len(xs) == 0:
            continue
        pts.append((xs[0], y))
    return np.asarray(pts, np.float32)

def detect_right_points(mask):
    h, w = mask.shape
    pts = []
    for y in range(h):
        row = mask[y]
        xs = np.where(row == 255)[0]
        if len(xs) == 0:
            continue
        pts.append((xs[-1], y))
    return np.asarray(pts, np.float32)

def detect_edge_points(mask, direction):
    pass
    # TODO refactor detect_top_points etc

def is_vertical_transition(gray, x, y):

    if x < 3 or x >= gray.shape[1]-3:
        return False

    left = np.mean(gray[y, x-3:x])
    right = np.mean(gray[y, x:x+3])

    return abs(left-right) > 40

def process_image(in_path, out_path):
    """
    Robust page extraction that handles missing top (or bottom) edges.
    - preserves original colors (warps the original image)
    - builds orthogonal axes (no shear)
    - if top is missing, uses bottom + good vertical to infer top by moving up expected_h
    - fills outside the filled page polygon with pure white
    - extensive debug output in OUTPUT_DIR/debug/<page_num>/
    """
    # ---------- small helpers ----------
    def ensure_dir(p):
        os.makedirs(p, exist_ok=True)

    def save_dbg(img, path):
        ensure_dir(os.path.dirname(path))
        cv2.imwrite(path, img)

    def percentile_threshold(gray):
        high_p = np.percentile(gray, THRESH_HIGH_PERCENTILE)
        thr = max(THRESH_MIN, int(high_p * 0.95))
        _, mask = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
        return mask, thr, int(high_p)

    def keep_largest_component(mask):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels <= 1:
            return mask
        areas = stats[1:, cv2.CC_STAT_AREA]
        best = 1 + int(np.argmax(areas))
        out = np.zeros_like(mask)
        out[labels == best] = 255
        return out

    def detect_vertical_streaks(mask, approx_width=3, length_thresh_ratio=0.15):
        h, w = mask.shape
        kx = approx_width
        ky = max(15, int(h * 0.02))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
        long_vertical = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(long_vertical, connectivity=8)
        streak_mask = np.zeros_like(mask)
        length_thresh = max(10, int(h * length_thresh_ratio))
        for i in range(1, num_labels):
            x, y, ww, hh, area = stats[i]
            if hh >= length_thresh and ww <= max(5, int(w * 0.01)):
                streak_mask[labels == i] = 255
        return streak_mask

    def contour_to_pts(c):
        return c.reshape(-1, 2)

    def fit_line_ransac(pts, iterations=RANSAC_ITER, inlier_dist=RANSAC_INLIER_DIST, min_inliers=RANSAC_MIN_INLIERS):
        if len(pts) < 2:
            raise ValueError("Not enough points for line fit")
        best_inliers = None
        best_cnt = 0
        best_model = None
        ptsf = pts.astype(np.float32)
        n = len(ptsf)
        for _ in range(iterations):
            i1, i2 = random.sample(range(n), 2)
            p1 = ptsf[i1]; p2 = ptsf[i2]
            vx = float(p2[0] - p1[0]); vy = float(p2[1] - p1[1])
            if abs(vx) < 1e-6 and abs(vy) < 1e-6:
                continue
            dists = np.abs(vy*(ptsf[:,0]-p1[0]) - vx*(ptsf[:,1]-p1[1])) / (math.hypot(vx, vy) + 1e-12)
            inliers = dists <= inlier_dist
            cnt = int(inliers.sum())
            if cnt >= min_inliers and cnt > best_cnt:
                best_cnt = cnt
                best_inliers = inliers.copy()
                best_model = (vx, vy, float(p1[0]), float(p1[1]))
        if best_model is None:
            vx, vy, x0, y0 = cv2.fitLine(ptsf, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
            inlier_mask = np.ones(len(ptsf), dtype=bool)
            return float(vx), float(vy), float(x0), float(y0), inlier_mask
        inlier_pts = ptsf[best_inliers]
        vx, vy, x0, y0 = cv2.fitLine(inlier_pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        return float(vx), float(vy), float(x0), float(y0), best_inliers

    def intersect_lines(l1, l2):
        vx1, vy1, x1, y1 = l1
        vx2, vy2, x2, y2 = l2
        A = np.array([[vx1, -vx2], [vy1, -vy2]], dtype=np.float32)
        b = np.array([x2 - x1, y2 - y1], dtype=np.float32)
        det = np.linalg.det(A)
        if abs(det) < 1e-8:
            return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        t1, t2 = np.linalg.solve(A, b)
        xi = x1 + t1 * vx1
        yi = y1 + t1 * vy1
        return float(xi), float(yi)

    # ---------- start ----------
    fname = os.path.basename(in_path)
    m = re.match(r"^(\d+)", fname)
    page_num = int(m.group(1)) if m else 0
    bad_on_left = (page_num % 2 == 1)

    img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print("Failed to read", in_path); return
    input_is_grayscale = (len(img.shape) == 2)
    if len(img.shape) == 2:
        # OpenCV expects 3-channel images for many operations
        # later convert back to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    H_img, W_img = img.shape[:2]

    # fixed
    # # FIXME wrong H_img?
    # expected_w = int(round(ASPECT * H_img))
    # expected_h = H_img

    dbgdir = os.path.join(OUTPUT_DIR, "debug", f"{page_num:03d}")
    if DEBUG: ensure_dir(dbgdir)

# Step 1: Segment page vs. gray background

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Otsu is usually sufficient here
    _, page_mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Page should be white
    if np.mean(gray[page_mask == 255]) < np.mean(gray[page_mask == 0]):
        page_mask = cv2.bitwise_not(page_mask)

    page_mask = cv2.morphologyEx(
        page_mask,
        cv2.MORPH_CLOSE,
        np.ones((5,5), np.uint8)
    )

# Step 2: Scan for transitions

    if bad_on_left:
        outside_pts = detect_right_points(mask)
    else:
        outside_pts = detect_left_points(mask)

    top_pts = detect_top_points(mask)
    bottom_pts = detect_bottom_points(mask)

# Step 3: Transition verification
    # The first white pixel isn't always a page edge. It might be noise.
    # Verify that every candidate actually crosses gray → page.

    # horizontal edges
    top = np.mean(gray[y-3:y, x])
    bottom = np.mean(gray[y:y+3, x])
    # TODO Reject points that don't have sufficient contrast.
    # TODO use is_vertical_transition

    # TODO vertical edges

# Step 4: Reject isolated outliers
    # Use a median filter before RANSAC.

    ys = pts[:,1]

    median = np.median(ys)

    pts = pts[
        np.abs(ys - median) < 40
    ]

top_line = fit_line_ransac(top_pts)[:4]
    bottom_line = fit_line_ransac(bottom_pts)[:4]
    outside_line = fit_line_ransac(outside_pts)[:4]

# # TODO remove

    # gray, mask, contours = get_gray_mask_contours(img, dbgdir)

    # if not contours:
    #     print(f"Warning: no contours found in {in_path}")
    #     return

    # page_contour = max(contours, key=cv2.contourArea)

    if config.use_three_edge_deskew:

        # top_pts, bottom_pts, outside_pts = split_edge_candidates(
        #     page_contour,
        #     bad_on_left
        # )

        # top_line = fit_line_ransac(top_pts)[:4]
        # bottom_line = fit_line_ransac(bottom_pts)[:4]
        # outside_line = fit_line_ransac(outside_pts)[:4]

        # old
        # top_angle = math.degrees(line_angle(top_line))
        # bottom_angle = math.degrees(line_angle(bottom_line))
        # outside_angle = math.degrees(line_angle(outside_line))

        # new
        top_angle = horizontal_line_angle(top_line)
        bottom_angle = horizontal_line_angle(bottom_line)
        outside_angle = vertical_line_angle(outside_line)

        if DEBUG:
            # start debug prints
            print()
            print(f"line 570: page_num={page_num}")

        rotation_error = -1 * (
            -top_angle
            -bottom_angle
        ) / 2.0

        Mrot = cv2.getRotationMatrix2D(
            (W_img/2, H_img/2),
            rotation_error,
            1.0
        )

        if DEBUG:
            print(
                f"line 575: before rotation: "
                f"top_angle={top_angle:.3f} "
                f"bottom_angle={bottom_angle:.3f} "
                f"outside_angle={outside_angle:.3f}"
            )

        rotated = cv2.warpAffine(
            img,
            Mrot,
            (W_img, H_img),
            borderValue=(255,255,255)
        )

        Hr, Wr = rotated.shape[:2]

        if DEBUG:
            print("line 580: rotated size", Wr, Hr)

        # img = rotated # ?

# re-detect lines in the rotated image

        # gray, mask, contours = get_gray_mask_contours(img, dbgdir)
        gray, mask, contours = get_gray_mask_contours(rotated, dbgdir)

        if 1:
            top_angle2 = horizontal_line_angle(top_line)
            bottom_angle2 = horizontal_line_angle(bottom_line)
            outside_angle2 = vertical_line_angle(outside_line)

            if DEBUG:
                print(
                    f"line 620: after rotation and re-fitting: "
                    f"top_angle2={top_angle2:.3f} "
                    f"bottom_angle2={bottom_angle2:.3f} "
                    f"outside_angle2={outside_angle2:.3f}"
                )

        if not contours:
            print(f"line 630: Warning: no contours found in {in_path}")
            return

        page_contour = max(contours, key=cv2.contourArea)

top_pts, bottom_pts, outside_pts = split_edge_candidates(
            page_contour,
            bad_on_left
        )

        top_line = fit_line_ransac(top_pts)[:4]
        bottom_line = fit_line_ransac(bottom_pts)[:4]
        outside_line = fit_line_ransac(outside_pts)[:4]

        # fixed
        # # FIXME H_img is wrong
        # expected_h = H_img
        # expected_w = int(round(ASPECT * expected_h))

        vx, vy, x0, y0 = outside_line

        outside_top = intersect_lines(
            outside_line,
            top_line
        )

        outside_bottom = intersect_lines(
            outside_line,
            bottom_line
        )

        if 0:
            # approximated page height
            page_height = math.dist(
                outside_top,
                outside_bottom
            )
        else:
            # perpendicular page height
            page_height = math.hypot(
                outside_bottom[0] - outside_top[0],
                outside_bottom[1] - outside_top[1]
            )

        # no. this fails to reconstruct the page height...
        # TODO try to solve this with the average height of multiple pages
        # assuming all pages must have the same height
        # also allowing the user to specify a scale_y factor
        if 0:
            if config.do_rotate == False:
                # fix scan height
                # document scanners can distort scans in the Y direction
                # FIXME honor config.do_rotate
                # if (config.do_rotate == True)
                # then all pages are rotated by 90 or 270 degrees
                # so the scanner's Y errors become our X errors
                # so we need scale_x to fix the page width
                if 0:
                    # use rotated_scan_y and page_height
                    rotated_scan_y_px = px_of_mm(config.rotated_scan_y, config.scan_resolution)
                    target_h = rotated_scan_y_px
                    scale_y = target_h / page_height
                elif 1:
                    # use rotated_margined_scan_y and H_img
                    rotated_margined_scan_y_px = px_of_mm(config.rotated_margined_scan_y, config.scan_resolution)
                    target_h = rotated_margined_scan_y_px
                    scale_y = target_h / H_img
                else:
                    # no, this fails because the scanner removes one edge
                    # so actual_aspect is always wrong...
                    #
                    # use rotated_margined_scan_y and H_img
                    # expected: what we ordered from the scanner
                    expected_aspect = (
                        config.rotated_margined_scan_x /
                        config.rotated_margined_scan_y
                    )
                    expected_height = config.rotated_margined_scan_y
                    # actual: what the scanner gave us
                    actual_aspect = W_img / H_img
                    actual_height = H_img
                    # we assume the scanner always returns correct X coordinates
                    # and all errors are only in Y coordinates
                    # expected_aspect / actual_aspect = actual_height / expected_height
                    actual_height_2 = expected_aspect / actual_aspect / expected_height
                    expected_aspect_factor = expected_aspect / actual_aspect
                    # the scale of actual_height relative to actual_height_2
                    actual_height_scale = actual_height / actual_height_2
                    if DEBUG:
                        print(f"expected_aspect={expected_aspect} actual_aspect={actual_aspect} expected_aspect_factor={expected_aspect_factor}")
                        print(f"expected_height={expected_height} actual_height={actual_height} actual_height_2={actual_height_2} actual_height_scale={actual_height_scale}")
                    target_h = actual_height_2
                    scale_y = target_h / H_img

                # debug: manually set the scale_y factor
                # scale_y = 1 / 1.03 # shrink the page height by 3%

                scale_y_tolerance = 0.001 # 0.1%

                if scale_y < (1 - scale_y_tolerance) or (1 + scale_y_tolerance) < scale_y:
                    if DEBUG:
                        print(f"line 680: scale_y={scale_y} target_h={target_h} page_height={page_height} -> scaling height")
                    rotated = cv2.resize(
                        rotated,
                        None,
                        fx=1.0,
                        fy=scale_y,
                        interpolation=cv2.INTER_CUBIC
                    )
                    page_height = page_height * scale_y

                    # re-detect lines in the scaled image

                    # gray, mask, contours = get_gray_mask_contours(img, dbgdir)
                    gray, mask, contours = get_gray_mask_contours(rotated, dbgdir)

                    if 1:
                        top_angle2 = horizontal_line_angle(top_line)
                        bottom_angle2 = horizontal_line_angle(bottom_line)
                        outside_angle2 = vertical_line_angle(outside_line)
                        if DEBUG:
                            print(
                                f"line 620: after rotation and re-fitting: "
                                f"top_angle2={top_angle2:.3f} "
                                f"bottom_angle2={bottom_angle2:.3f} "
                                f"outside_angle2={outside_angle2:.3f}"
                            )

                    if not contours:
                        print(f"line 630: Warning: no contours found in {in_path}")
                        return

                    page_contour = max(contours, key=cv2.contourArea)

                    top_pts, bottom_pts, outside_pts = split_edge_candidates(
                        page_contour,
                        bad_on_left
                    )

                    top_line = fit_line_ransac(top_pts)[:4]
                    bottom_line = fit_line_ransac(bottom_pts)[:4]
                    outside_line = fit_line_ransac(outside_pts)[:4]

                    # fixed
                    # # FIXME H_img is wrong
                    # expected_h = H_img
                    # expected_w = int(round(ASPECT * expected_h))

                    vx, vy, x0, y0 = outside_line

                    outside_top = intersect_lines(
                        outside_line,
                        top_line
                    )

                    outside_bottom = intersect_lines(
                        outside_line,
                        bottom_line
                    )

                else:
                    if DEBUG:
                        print(f"line 680: scale_y={scale_y} target_h={target_h} page_height={page_height} -> not scaling height")

            else:
                # config.do_rotate == True
                # fix scan width
                # Raw scans are rotated by 90/270 degrees.
                # Therefore: scanner Y errors -> image X errors
                # We must correct width (scale_x), not height (scale_y).
                if 1:
                    # The detected outside edge gives us the real page height
                    # in image coordinates.  After rotation this corresponds to
                    # scanner X, which is assumed correct.
                    #
                    # The missing/damaged dimension is the page width.

                    rotated_margined_scan_x_px = px_of_mm(
                        config.rotated_margined_scan_x,
                        config.scan_resolution
                    )

                    target_w = rotated_margined_scan_x_px

                    actual_w = math.dist(
                        outside_top,
                        outside_bottom
                    )

                    # This is actually the width in scanner coordinates because
                    # the document is rotated.
                    scale_x = target_w / actual_w

                    scale_x_tolerance = 0.001  # 0.1%

                    if (
                        scale_x < (1 - scale_x_tolerance)
                        or
                        scale_x > (1 + scale_x_tolerance)
                    ):
                        if DEBUG:
                            print(
                                "line 680:",
                                f"scale_x={scale_x}",
                                f"target_w={target_w}",
                                f"actual_w={actual_w}",
                                "-> scaling width"
                            )

                        rotated = cv2.resize(
                            rotated,
                            None,
                            fx=scale_x,
                            fy=1.0,
                            interpolation=cv2.INTER_CUBIC
                        )

                        # Update image size
                        Hr, Wr = rotated.shape[:2]

                        if DEBUG:
                            print(
                                "line 690: after scale_x",
                                f"Wr={Wr}",
                                f"Hr={Hr}"
                            )

                        # Re-detect page after scaling
                        gray, mask, contours = get_gray_mask_contours(
                            rotated,
                            dbgdir
                        )

                        if not contours:
                            print(
                                f"line 700: Warning: no contours found after scaling {in_path}"
                            )
                            return

                        page_contour = max(
                            contours,
                            key=cv2.contourArea
                        )

                        top_pts, bottom_pts, outside_pts = split_edge_candidates(
                            page_contour,
                            bad_on_left
                        )

                        top_line = fit_line_ransac(top_pts)[:4]
                        bottom_line = fit_line_ransac(bottom_pts)[:4]
                        outside_line = fit_line_ransac(outside_pts)[:4]

                        outside_top = intersect_lines(
                            outside_line,
                            top_line
                        )

                        outside_bottom = intersect_lines(
                            outside_line,
                            bottom_line
                        )

                    else:
                        if DEBUG:
                            print(
                                "line 680:",
                                f"scale_x={scale_x}",
                                "-> not scaling width"
                            )

expected_h = int(round(page_height))

        # ASPECT = x / y
        # x = y * ASPECT

        # expand the binding edge to expected_w
        # expected_w = int(round(expected_h * ASPECT))
        expected_w = int(round(page_height * ASPECT))

        if 0:
            # crop as a rectangle
            x_out = (
                outside_top[0]
                +
                outside_bottom[0]
            ) / 2

            x_out = int(round(x_out))

            # if bad_on_left:
            #     x0_page = x_out
            #     x1_page = x_out + expected_w
            # else:
            #     x0_page = x_out - expected_w
            #     x1_page = x_out

            if bad_on_left:
                # outside edge is RIGHT
                x1_page = x_out
                x0_page = x_out - expected_w
            else:
                # outside edge is LEFT
                x0_page = x_out
                x1_page = x_out + expected_w

            # clamp
            # x0_page = max(0, x0_page)
            # x1_page = min(rotated.shape[1], x1_page)

            y_top = int(round(outside_top[1]))
            y_bottom = int(round(outside_bottom[1]))

            src_x0 = max(0, x0_page)
            src_x1 = min(rotated.shape[1], x1_page)

            src_y0 = max(0, y_top)
            src_y1 = min(rotated.shape[0], y_top + expected_h)

            dst_x0 = src_x0 - x0_page
            dst_y0 = src_y0 - y_top

            if DEBUG:
                print(
                    "line 640:",
                    f"outside_top={outside_top}",
                    f"outside_bottom={outside_bottom}",
                )

            # crop = rotated[
            #     y_top:y_top+expected_h,
            #     x0_page:x1_page
            # ]

            full_crop = np.ones(
                (expected_h, expected_w, 3),
                dtype=rotated.dtype
            ) * 255

            full_crop[
                dst_y0:dst_y0 + (src_y1-src_y0),
                dst_x0:dst_x0 + (src_x1-src_x0)
            ] = rotated[
                src_y0:src_y1,
                src_x0:src_x1
            ]

            # FIXME fill empty area near binding edge
            # currently it is all white
            # but it should copy the vertical pattern near the binding edge

            crop = full_crop

            if DEBUG:
                print(
                    "line 650: crop",
                    f"W_img={W_img}",
                    f"H_img={H_img}",
                    f"x0_page={x0_page}",
                    f"x1_page={x1_page}",
                    f"y_top={y_top}",
                    f"y_bottom={y_bottom}",
                    f"src_x0={src_x0}",
                    f"src_x1={src_x1}",
                    f"src_y0={src_y0}",
                    f"src_y1={src_y1}",
                    f"dst_x0={dst_x0}",
                    f"dst_y0={dst_y0}",
                    f"expected_w={expected_w}",
                    f"expected_h={expected_h}",
                )

            if DEBUG:
                vis = rotated.copy()
                cv2.rectangle(
                    vis,
                    (int(x0_page), int(y_top)),
                    (int(x1_page), int(y_bottom)),
                    (0,0,255),
                    3,
                )
                name = f"{page_num:03d}.crop_debug_rotated_line730.jpg"
                print(f"writing {name}")
                cv2.imwrite(name, vis)

        else:
            # crop as a quadrilateral
            # not better than "crop as a rectangle"?
            if 0:
                # expand the binding edge to expected_w
                if bad_on_left:
                    # outside edge is the RIGHT edge
                    x1_top = outside_top[0]
                    x1_bottom = outside_bottom[0]
                    x0_top = x1_top - expected_w
                    x0_bottom = x1_bottom - expected_w
                else:
                    # outside edge is the LEFT edge
                    x0_top = outside_top[0]
                    x0_bottom = outside_bottom[0]
                    x1_top = x0_top + expected_w
                    x1_bottom = x0_bottom + expected_w
            else:
                # dont expand the binding edge to expected_w
                # use only the detected page edges
                # if bad_on_left:
                #     # outside edge is RIGHT edge
                #     x1_top = outside_top[0]
                #     x1_bottom = outside_bottom[0]
                #     # use the actual detected left edge
                #     x0_top = np.min(page_contour[:,0,0])
                #     x0_bottom = x0_top
                # else:
                #     # outside edge is LEFT edge
                #     x0_top = outside_top[0]
                #     x0_bottom = outside_bottom[0]
                #     # use the actual detected right edge
                #     x1_top = np.max(page_contour[:,0,0])
                #     x1_bottom = x1_top
                # problem: page_contour after thresholding may include the background
                # or may not have a reliable missing-edge position.
                # A cleaner temporary solution is to use the detected quadrilateral width
                # from the two horizontal edge intersections
                if bad_on_left:
                    # outside edge is right
                    x1_top = outside_top[0]
                    x1_bottom = outside_bottom[0]
                    # find leftmost detected page boundary
                    x0_top = np.min(top_pts[:,0])
                    x0_bottom = np.min(bottom_pts[:,0])
                else:
                    # outside edge is left
                    x0_top = outside_top[0]
                    x0_bottom = outside_bottom[0]
                    # find rightmost detected page boundary
                    x1_top = np.max(top_pts[:,0])
                    x1_bottom = np.max(bottom_pts[:,0])

                # dont expand the binding edge to expected_w
                expected_w = int(round(
                    math.dist((x0_top, outside_top[1]), (x1_top, outside_top[1]))
                ))

src = np.float32([
                [x0_top, outside_top[1]],
                [x1_top, outside_top[1]],
                [x1_bottom, outside_bottom[1]],
                [x0_bottom, outside_bottom[1]],
            ])

            dst = np.float32([
                [0,0],
                [expected_w-1,0],
                [expected_w-1,expected_h-1],
                [0,expected_h-1],
            ])

            M = cv2.getPerspectiveTransform(src, dst)

            crop = cv2.warpPerspective(
                rotated,
                M,
                (expected_w, expected_h),
                borderValue=(255,255,255)
            )

            if DEBUG:
                print(
                    "line 650: crop",
                    f"W_img={W_img}",
                    f"H_img={H_img}",
                    f"outside_top={outside_top}",
                    f"outside_bottom={outside_bottom}",
                    f"x1_top={x1_top}",
                    f"x1_bottom={x1_bottom}",
                    f"x0_top={x0_top}",
                    f"x0_bottom={x0_bottom}",
                    f"expected_w={expected_w}",
                    f"expected_h={expected_h}",
                )

            if DEBUG:
                vis = rotated.copy()
                pts = np.array([
                    [x0_top, outside_top[1]],
                    [x1_top, outside_top[1]],
                    [x1_bottom, outside_bottom[1]],
                    [x0_bottom, outside_bottom[1]],
                ], np.int32)
                cv2.polylines(vis, [pts], True, (0,0,255), 3)
                name = f"{page_num:03d}.crop_debug_rotated_line730.jpg"
                print(f"writing {name}")
                cv2.imwrite(name, vis)

        # crop = rotated[
        #     0:expected_h,
        #     int(x0_page):int(x1_page)
        # ]

        # y_top = round(outside_top[1])
        # y_bottom = round(outside_bottom[1])
        # actual_height = y_bottom - y_top
        # crop = rotated[
        #     y_top:y_top + expected_h,
        #     x0_page:x1_page
        # ]

        # y_top = int(round(outside_top[1]))
        # y_bottom = int(round(outside_bottom[1]))
        # crop = rotated[
        #     y_top:y_bottom,
        #     x0_page:x1_page
        # ]

        if 0:
            # img = repair_binding(img, bad_on_left, width=50)
            if 1:
                # rotated = repair_binding(rotated, bad_on_left, width=50)
                # Hr, Wr = rotated.shape[:2]
                # print("line 660: rotated size", Wr, Hr)
                # warped = rotated
                crop = repair_binding(crop, bad_on_left, width=50)
                Hr, Wr = crop.shape[:2]
                if DEBUG:
                    print("line 660: crop size", Wr, Hr)
            else:
                crop = repair_binding(crop, bad_on_left, width=50)

            if DEBUG:
                print(
                    "line 760: after repair_binding: crop actual:",
                    f"crop.shape[1]={crop.shape[1]}",
                    f"expected_w={expected_w}",
                )

        warped = crop

    else:
        # config.use_three_edge_deskew == False

        raise NotImplementedError("sorry, your book is too tall for your scanner...")

        # FIXME use only two page edges: outside, bottom

        # Approximate contour to quadrilateral
        epsilon = 0.02 * cv2.arcLength(page_contour, True)
        approx = cv2.approxPolyDP(page_contour, epsilon, True)
        if len(approx) != 4:
            approx = cv2.convexHull(page_contour)
            if len(approx) < 4:
                print(f"Warning: not enough points for perspective in {in_path}")
                return
            # pick 4 extreme points
            pts = np.array([
                approx[approx[:,0,0].argmin()][0],  # leftmost
                approx[approx[:,0,1].argmin()][0],  # topmost
                approx[approx[:,0,0].argmax()][0],  # rightmost
                approx[approx[:,0,1].argmax()][0]   # bottommost
            ])
        else:
            pts = approx.reshape(4,2)

        rect = order_points(pts)

        # Perspective transform
        widthA = np.linalg.norm(rect[2] - rect[3])
        widthB = np.linalg.norm(rect[1] - rect[0])
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.linalg.norm(rect[1] - rect[2])
        heightB = np.linalg.norm(rect[0] - rect[3])
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    # # Add internal white border
    # h, w = warped.shape[:2]
    # canvas = np.ones_like(warped) * 255
    # b = BORDER_SIZE
    # canvas[b:h-b, b:w-b] = warped[b:h-b, b:w-b]

    h, w = warped.shape[:2]
    b = BORDER_SIZE

    # Prepare a canvas with the same content as warped
    canvas = warped.copy()

    # Function to compute average color along a strip
    def avg_color_strip(img, axis, start, end, strip_width=1):
        if axis == 'top':
            strip = img[start:end, :, :]
        elif axis == 'bottom':
            strip = img[h-end:h-start, :, :]
        elif axis == 'left':
            strip = img[:, start:end, :]
        elif axis == 'right':
            strip = img[:, w-end:w-start, :]
        else:
            raise ValueError("Invalid axis")
        return np.mean(strip, axis=(0,1)).astype(np.uint8)

    # FIXME preserve patterns near edges
    if 0:
        # Fill borders with local average color
        canvas[0:b, :, :] = avg_color_strip(canvas, 'top', 50, 100)       # top border
        canvas[h-b:h, :, :] = avg_color_strip(canvas, 'bottom', 50, 100)  # bottom border
        canvas[:, 0:b, :] = avg_color_strip(canvas, 'left', 50, 100)      # left border
        canvas[:, w-b:w, :] = avg_color_strip(canvas, 'right', 50, 100)   # right border

    if input_is_grayscale:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

    ensure_dir(os.path.dirname(out_path))
    cv2.imwrite(out_path, canvas)
    print(f"writing {out_path}")

def main():
    ensure_dir(OUTPUT_DIR)
    # TODO use image_format from 030-measure-page-size.txt
    image_format = "jpg"
    files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(f".{image_format}")])
    if not files:
        print("No image files found in", INPUT_DIR)
        return
    for fname in files:
        m = re.match(r"^(\d+)", fname)
        # page_num = int(m.group(1)) if m else 0
        page_num = int(m.group(1)) # can throw
        # if page_num != 14: continue # debug
        if not page_num in (1, 2, 12, 13): continue # debug
        # if not page_num in [340, 345]: continue # debug
        # if page_num < 300: continue # debug
        # if page_num != 320: continue # debug
        in_path = os.path.join(INPUT_DIR, fname)
        out_path = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(out_path):
            continue
        try:
            process_image(in_path, out_path)
        except Exception as exc:
            print(f"Error processing {fname}: {exc}")
            raise

if __name__ == "__main__":
    main()
```

## Response:

7/13/2026, 7:51:06 AM

I think this is the right place to add the gray→page verification. Since your scanner **guarantees a mid-gray background**, you don't actually need to look for "bright" pixels—you want to verify that each detected point is sitting on a **strong gray→page transition**.

I'd also refactor the four `detect_*_points()` functions into a single detector later, because the verification logic becomes almost identical.

I'd replace Step 3 with something like this:

``` python
    # Step 3: Transition verification
    #
    # The first white pixel isn't always the page.
    # It might be dust, glare, or text sticking out.
    #
    # Keep only points that actually separate gray background
    # from the white page.

    def verify_horizontal(points, top_edge):
        good = []

        H, W = gray.shape

        for x, y in points.astype(int):

            if y < 3 or y >= H-3:
                continue

            if top_edge:
                outside = np.mean(gray[y-3:y, x])
                inside  = np.mean(gray[y:y+3, x])
            else:
                outside = np.mean(gray[y:y+3, x])
                inside  = np.mean(gray[y-3:y, x])

            # must have strong contrast
            if abs(float(inside) - float(outside)) < 40:
                continue

            # outside should be scanner gray
            if not (60 < outside < 200):
                continue

            # inside should be brighter
            if inside <= outside:
                continue

            good.append((x, y))

        return np.asarray(good, np.float32)

def verify_vertical(points, right_edge):
        good = []

        H, W = gray.shape

        for x, y in points.astype(int):

            if x < 3 or x >= W-3:
                continue

            if right_edge:
                outside = np.mean(gray[y, x:x+3])
                inside  = np.mean(gray[y, x-3:x])
            else:
                outside = np.mean(gray[y, x-3:x])
                inside  = np.mean(gray[y, x:x+3])

            if abs(float(inside) - float(outside)) < 40:
                continue

            if not (60 < outside < 200):
                continue

            if inside <= outside:
                continue

            good.append((x, y))

        return np.asarray(good, np.float32)

top_pts = verify_horizontal(
        top_pts,
        top_edge=True
    )

    bottom_pts = verify_horizontal(
        bottom_pts,
        top_edge=False
    )

    outside_pts = verify_vertical(
        outside_pts,
        right_edge=bad_on_left
    )
```

------------------------------------------------------------------------

### I'd make one small improvement

Instead of

``` python
if abs(float(inside) - float(outside)) < 40:
```

I'd use your existing helper:

``` python
if top_edge:
    if not is_horizontal_transition(gray, x, y):
        continue
else:
    ...
```

and similarly

``` python
if not is_vertical_transition(gray, x, y):
    continue
```

so the threshold only exists in one place.

------------------------------------------------------------------------

### Even more robust (I think this will solve most remaining failures)

Since your background is **always gray**, don't merely require a contrast. Require the **outside** to actually look like scanner background.

For example:

``` python
BACKGROUND_MIN = 80
BACKGROUND_MAX = 180
```

Then

``` python
if not (BACKGROUND_MIN <= outside <= BACKGROUND_MAX):
    continue
```

This rejects:

- text
- black illustrations
- shadows
- dirt
- torn paper
- holes in the page

because none of those are 50% gray.

That scanner invariant ("outside must be gray") is much stronger than what most page detection algorithms have available, and I'd lean on it as much as possible. It will make the RANSAC stage much cleaner because it receives almost exclusively true edge points.

## Prompt:

7/13/2026, 2:06:26 PM

problem: pts is undefined. i have top_pts, bottom_pts, outside_pts. which is it? all three?

``` py

# ---------- start ----------
fname = os.path.basename(in_path)
m = re.match(r"^(\d+)", fname)
page_num = int(m.group(1)) if m else 0
bad_on_left = (page_num % 2 == 1)

img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
if img is None:
    print("Failed to read", in_path); return
input_is_grayscale = (len(img.shape) == 2)
if len(img.shape) == 2:
    # OpenCV expects 3-channel images for many operations
    # later convert back to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
H_img, W_img = img.shape[:2]

# fixed
# # FIXME wrong H_img?
# expected_w = int(round(ASPECT * H_img))
# expected_h = H_img

dbgdir = os.path.join(OUTPUT_DIR, "debug", f"{page_num:03d}")
if DEBUG: ensure_dir(dbgdir)

# Step 1: Segment page vs. gray background

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Otsu is usually sufficient here
_, page_mask = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# Page should be white
if np.mean(gray[page_mask == 255]) < np.mean(gray[page_mask == 0]):
    page_mask = cv2.bitwise_not(page_mask)

page_mask = cv2.morphologyEx(
    page_mask,
    cv2.MORPH_CLOSE,
    np.ones((5,5), np.uint8)
)

# TODO verify
# TODO rename
mask = page_mask

# Step 2: Scan for transitions

if bad_on_left:
    outside_pts = detect_right_points(mask)
else:
    outside_pts = detect_left_points(mask)

top_pts = detect_top_points(mask)
bottom_pts = detect_bottom_points(mask)

# Step 3: Transition verification
#
# The first white pixel isn't always the page.
# It might be dust, glare, or text sticking out.
#
# Keep only points that actually separate gray background
# from the white page.

top_pts = verify_horizontal(
    gray,
    top_pts,
    top_edge=True
)

bottom_pts = verify_horizontal(
    gray,
    bottom_pts,
    top_edge=False
)

outside_pts = verify_vertical(
    gray,
    outside_pts,
    right_edge=bad_on_left
)

# Step 4: Reject isolated outliers
# Use a median filter before RANSAC.

# FIXME UnboundLocalError: cannot access local variable 'pts' where it is not associated with a value

ys = pts[:,1]

median = np.median(ys)

pts = pts[
    np.abs(ys - median) < 40
]

top_line = fit_line_ransac(top_pts)[:4]
bottom_line = fit_line_ransac(bottom_pts)[:4]
outside_line = fit_line_ransac(outside_pts)[:4]
```

## Response:

7/13/2026, 2:06:27 PM

Yes. The median filter should be applied **independently to each edge**, because they have different coordinate systems.

For the horizontal edges, filter on **y**.

For the vertical edge, filter on **x**.

I'd make a helper:

``` python
def reject_outliers_horizontal(pts, tolerance=40):
    if len(pts) == 0:
        return pts

    ys = pts[:,1]
    median = np.median(ys)

    return pts[np.abs(ys - median) < tolerance]

def reject_outliers_vertical(pts, tolerance=40):
    if len(pts) == 0:
        return pts

    xs = pts[:,0]
    median = np.median(xs)

    return pts[np.abs(xs - median) < tolerance]
```

Then use it like this:

``` python
top_pts = reject_outliers_horizontal(top_pts)
bottom_pts = reject_outliers_horizontal(bottom_pts)
outside_pts = reject_outliers_vertical(outside_pts)
```

I actually wouldn't stop there, though.

Since you're fitting lines with RANSAC immediately afterwards, a **median** isn't ideal if a large chunk of the page edge is missing (which is exactly one of your failure cases). A better approach is a **running median** or **median filter along the scan direction**, which removes isolated spikes while preserving gradual skew.

For example, for the top edge:

    x=0      y=103
    x=1      y=104
    x=2      y=103
    x=3      y=220   <-- dust
    x=4      y=104
    x=5      y=103

a 5- or 7-pixel median filter on the `y` values would replace the `220` with `104`, after which RANSAC has a much cleaner dataset.

Given the way your scanner works (continuous page edges with occasional bad detections), I'd favor:

1.  Detect first white pixel.
2.  Verify gray → page transition.
3.  Apply a 1D median filter to the edge coordinates.
4.  Feed the cleaned points to RANSAC.

That combination should be noticeably more robust than a global median threshold.

## Prompt:

7/13/2026, 2:21:25 PM

ok, this works better than before, but it still fails on a page with a solid orange rectangle on the page, where the rectangle edge is detected as the page edge...

if the script would scan from outside to inside, it would see: gray - white - orange... so it should not only search for edges with high contrast, but it should also:

1.  detect the contrast between outside gray and inside non-gray (not always white... pages can also be black, or mixed... but in most cases different than the solid midgray background)

2.  rank the found lines, so the page rectangle is the biggest rectangle in the image

## Response:

7/13/2026, 2:21:26 PM

I think you've arrived at the key insight.

Right now your algorithm is:

> "Find the first transition."

What you actually want is:

> "Find all plausible page-edge transitions, then choose the rectangle that best matches a page."

Those are very different problems.

## 1. Detect every gray→non-gray transition

Instead of returning a single point per scanline, return **every** transition.

For example, scanning from the left:

    gray gray gray white white orange orange white white
                    ^      ^          ^
                    A      B          C

where

- A = page edge (correct)
- B = illustration edge (wrong)
- C = illustration edge (wrong)

Store all of them.

Likewise from the top:

    gray
    gray
    gray
    white
    white
    orange
    orange
    white

Again, keep every transition.

------------------------------------------------------------------------

## 2. Fit multiple candidate lines

Instead of one RANSAC:

    candidate lines:
    x = 1203
    x = 1050
    x =  820

Likewise for top:

    y = 52
    y = 310

RANSAC can even be run repeatedly:

- fit best line
- remove its inliers
- fit next best
- ...

This is a common technique called **iterative RANSAC**.

------------------------------------------------------------------------

## 3. Build every possible page rectangle

For example

    top:
        48
        310

    outside:
        1210
        1035

gives

    48 × 1210
    48 × 1035
    310 × 1210
    310 × 1035

Four candidate rectangles.

------------------------------------------------------------------------

## 4. Score them

This is where your scanner gives you a huge advantage.

A real page has

- outside is gray
- inside is not gray
- large area
- correct aspect ratio
- correct physical size
- nearly rectangular

So your score can be something like

    score =
        10 * transition_strength
      + 20 * rectangle_area
      + 30 * aspect_ratio_match
      + 20 * outside_gray_score
      + 20 * inside_non_gray_score

The orange rectangle loses because

- its area is much smaller
- its aspect ratio is wrong
- the outside isn't gray on all four sides

------------------------------------------------------------------------

## Even simpler

Because your page is always surrounded by gray, you don't actually need "the strongest edge."

You want

> **the first gray→non-gray region that continues for a long distance.**

Imagine scanning one row:

    gray gray gray white white orange orange white white

Current algorithm

    first transition

Better algorithm

    find every transition
    measure how long the new region lasts

The page continues for thousands of pixels.

The orange rectangle lasts maybe 200 pixels.

That's a massive difference.

------------------------------------------------------------------------

## I think there's an even better formulation

I wouldn't think in terms of edges at all anymore.

I'd think in terms of **connected regions**.

Your scanner background is a single connected gray component.

The page is the largest connected non-gray component.

Once you have that component, finding its top, bottom, and outside edges becomes much easier.

In other words:

    gray background
            ↓
    connected-component labeling
            ↓
    largest non-gray component
            ↓
    extract its boundary
            ↓
    fit 3 lines

rather than

    scanline
        ↓
    transition
        ↓
    hope it's the page

Given your scanner's invariant ("background is always solid mid-gray"), I'd build the algorithm around **regions first, edges second**. It's a much stronger constraint than almost all generic document-scanning algorithms get to assume, and it naturally rejects things like a large orange rectangle inside the page because that rectangle is part of the same connected page region, not a separate page boundary.

## Prompt:

7/13/2026, 2:28:19 PM

ok. we can also exploit the fact that i know the approximate size of the gray background from config.scan_margin (in millimeters)

my config parser is the function load_config in \_shared.py

``` py
def load_config(config_path=config_path):
    spec = importlib.util.spec_from_file_location("config", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {config_path}")

    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    # allow appending some pages
    # without having to rename all files
    config.max_num_pages = int(max(
        config.num_pages + 100,
        config.num_pages * 1.2,
    ))

    config.page_num_width = len(str(config.max_num_pages))

    # book binding side = scanner top side
    # TODO handle more cases?
    config.use_three_edge_deskew = (
        config.do_rotate
        and config.rotate_odd == 270
        and config.rotate_even == 90
    )

    config.scan_x = min(config.scan_x, config.max_scan_x)
    config.scan_y = min(config.scan_y, config.max_scan_y)

    def get_rotated_x_y(config, x, y):
        if config.use_three_edge_deskew:
            # rotate by 90 or 270 degrees
            x, y = y, x
        return x, y

    # scanned image size after 060-rotate-crop.py
    # NOTE dont apply crop
    (
        config.rotated_scan_x,
        config.rotated_scan_y
    ) = get_rotated_x_y(config, config.scan_x, config.scan_y)

    config.rotated_scan_aspect = config.rotated_scan_x / config.rotated_scan_y

    config.margined_scan_x = min(
        config.scan_x + config.scan_margin,
        config.max_scan_x
    )
    config.margined_scan_y = min(
        config.scan_y + config.scan_margin,
        config.max_scan_y
    )

    (
        config.rotated_margined_scan_x,
        config.rotated_margined_scan_y
    ) = get_rotated_x_y(config, config.margined_scan_x, config.margined_scan_y)

    # TODO validate config
    # if orientation_is_portrait and (scan_x < scan_y) and do_rotate:

    return config
```

so we can use the size differences  
config.rotated_margined_scan_x - config.rotated_scan_x  
config.rotated_margined_scan_y - config.rotated_scan_y  
to get the approximate margin sizes

i would upscale the margin sizes with some safety factor like 1.5 or 2  
and then i would search for the page edges only in the area of the margins

## Response:

7/13/2026, 2:28:19 PM

I think that's a much better direction than making the edge detector increasingly sophisticated.

You're adding **prior knowledge** instead of more heuristics.

### Why this works

Your scanner geometry is known:

    +-----------------------------------------+
    |               gray margin               |
    |   +-------------------------------+     |
    |   |                               |     |
    |   |            page               |     |
    |   |                               |     |
    |   +-------------------------------+     |
    |                                         |
    +-----------------------------------------+

The page edge **must** lie inside the scanner margin.

It **cannot** suddenly appear 30% into the page.

So the orange rectangle immediately becomes impossible, because it's far outside the search band.

------------------------------------------------------------------------

## Compute the expected margins

You already have

``` python
margin_x_mm = (
    config.rotated_margined_scan_x -
    config.rotated_scan_x
)

margin_y_mm = (
    config.rotated_margined_scan_y -
    config.rotated_scan_y
)
```

Convert them to pixels:

``` python
margin_x_px = px_of_mm(
    margin_x_mm,
    config.scan_resolution
)

margin_y_px = px_of_mm(
    margin_y_mm,
    config.scan_resolution
)
```

Then enlarge them:

``` python
SEARCH_MARGIN_FACTOR = 2.0

margin_x_px *= SEARCH_MARGIN_FACTOR
margin_y_px *= SEARCH_MARGIN_FACTOR
```

I'd even round upward:

``` python
margin_x_px = int(math.ceil(margin_x_px))
margin_y_px = int(math.ceil(margin_y_px))
```

------------------------------------------------------------------------

## Restrict the scan

Instead of scanning the entire image:

Top edge:

``` python
for y in range(H):
```

scan only

``` python
for y in range(margin_y_px):
```

Bottom:

``` python
for y in range(H-margin_y_px, H):
```

Left:

``` python
for x in range(margin_x_px):
```

Right:

``` python
for x in range(W-margin_x_px, W):
```

Now the detector literally **cannot** lock onto an illustration that's 300 pixels inside the page.

------------------------------------------------------------------------

## It gets even better

Your detector currently finds

> first white pixel

Instead, within the margin band, find

> the strongest gray→non-gray transition.

For example, in the top band:

    gray
    gray
    gray
    white  ← page edge
    white
    orange
    orange

Only the first transition lies inside the allowed search window.

------------------------------------------------------------------------

## I would actually change the API

Instead of

``` python
detect_top_points(mask)
```

I'd make it

``` python
detect_top_points(
    gray,
    search_height=margin_y_px
)
```

Similarly

``` python
detect_bottom_points(
    gray,
    search_height=margin_y_px
)

detect_left_points(
    gray,
    search_width=margin_x_px
)

detect_right_points(
    gray,
    search_width=margin_x_px
)
```

Then those functions can directly examine the grayscale image, look for gray→non-gray transitions, verify the transition, and never even consider pixels outside the expected margin. You no longer need to threshold the whole image first just to get candidate points.

## One thing I'd tweak

Rather than a factor of **2×**, I'd define the search band as:

    expected margin
    + maximum expected skew
    + a small safety margin

For example:

``` python
SEARCH_MARGIN = expected_margin + 100
```

or

``` python
SEARCH_MARGIN = expected_margin * 1.5 + 50
```

The reason is that scanner margins are usually fairly consistent. If the nominal margin is 12 mm, searching 24 mm into the page gives an illustration twice as much room to confuse the detector. A narrower band based on the known margin plus a fixed allowance for rotation, cropping, and measurement error usually provides a stronger geometric constraint while still being robust.

## Prompt:

7/13/2026, 2:36:57 PM

now would be a good time to refactor all the `detect_*_points` functions into one function, where the direction is specified with a global integer enum

## Response:

7/13/2026, 2:36:57 PM

I think this is exactly the right time. Once you start restricting searches to the expected margins, all four `detect_*_points()` functions become identical except for:

- scan direction
- search window
- returned coordinate

I'd define an enum like this:

``` py
EDGE_TOP = 0
EDGE_BOTTOM = 1
EDGE_LEFT = 2
EDGE_RIGHT = 3
```

or, a little cleaner,

``` py
from enum import IntEnum

class Edge(IntEnum):
    TOP = 0
    BOTTOM = 1
    LEFT = 2
    RIGHT = 3
```

Then one function handles everything:

``` py
def detect_edge_points(mask, edge, margin):
    h, w = mask.shape
    pts = []

    if edge == EDGE_TOP:
        y_max = min(margin, h)
        for x in range(w):
            ys = np.where(mask[:y_max, x] == 255)[0]
            if len(ys):
                pts.append((x, ys[0]))

    elif edge == EDGE_BOTTOM:
        y_min = max(0, h - margin)
        for x in range(w):
            ys = np.where(mask[y_min:, x] == 255)[0]
            if len(ys):
                pts.append((x, y_min + ys[-1]))

    elif edge == EDGE_LEFT:
        x_max = min(margin, w)
        for y in range(h):
            xs = np.where(mask[y, :x_max] == 255)[0]
            if len(xs):
                pts.append((xs[0], y))

    elif edge == EDGE_RIGHT:
        x_min = max(0, w - margin)
        for y in range(h):
            xs = np.where(mask[y, x_min:] == 255)[0]
            if len(xs):
                pts.append((x_min + xs[-1], y))

    return np.asarray(pts, np.float32)
```

Then the call site becomes

``` py
margin_x = int(round(
    px_of_mm(
        config.rotated_margined_scan_x - config.rotated_scan_x,
        config.scan_resolution,
    ) * 1.5
))

margin_y = int(round(
    px_of_mm(
        config.rotated_margined_scan_y - config.rotated_scan_y,
        config.scan_resolution,
    ) * 1.5
))
```

``` py
top_pts = detect_edge_points(
    mask,
    EDGE_TOP,
    margin_y,
)

bottom_pts = detect_edge_points(
    mask,
    EDGE_BOTTOM,
    margin_y,
)

if bad_on_left:
    outside_pts = detect_edge_points(
        mask,
        EDGE_RIGHT,
        margin_x,
    )
else:
    outside_pts = detect_edge_points(
        mask,
        EDGE_LEFT,
        margin_x,
    )
```

## I would go one step further

Since you're now adding transition verification, I'd combine them into a single routine.

Instead of

``` py
detect_edge_points(...)
verify_horizontal(...)
verify_vertical(...)
```

make

``` py
detect_edge_points(
    gray,
    mask,
    edge,
    margin,
)
```

Internally it would

1.  search from the expected margin,
2.  find the first white pixel,
3.  verify the gray→page transition,
4.  reject weak transitions,
5.  return only verified points.

Then your entire detection stage shrinks to

``` py
top_pts = detect_edge_points(
    gray,
    mask,
    EDGE_TOP,
    margin_y,
)

bottom_pts = detect_edge_points(
    gray,
    mask,
    EDGE_BOTTOM,
    margin_y,
)

outside_pts = detect_edge_points(
    gray,
    mask,
    EDGE_RIGHT if bad_on_left else EDGE_LEFT,
    margin_x,
)
```

At that point there is exactly one piece of code responsible for "finding page edge samples," which makes it much easier to keep improving (for example by later returning the first *strongest* transition instead of merely the first valid one). I think that's the direction I'd take for the rest of this pipeline.

## Prompt:

7/13/2026, 3:45:58 PM

line 849:  
FIXME merge old code with new code

``` py
#!/usr/bin/env python3
"""
extract scanned page from gray background

restore the binding edge of the page
by filling the missing width
with the average color near the binding edge
"""

INPUT_DIR = "060-rotate-crop"
OUTPUT_DIR = "065-remove-page-borders"

# === Tuning parameters ===
DEBUG = True
# DEBUG = False
BORDER_SIZE = 100  # pixels

RANSAC_ITER = 400
RANSAC_INLIER_DIST = 6.0
# RANSAC_MIN_INLIERS = 30
RANSAC_MIN_INLIERS = 20

THRESH_HIGH_PERCENTILE = 99
THRESH_MIN = 200

import os
import re
import math
import random
import numpy as np
import cv2

from _shared import (
    load_config,
    get_page_num,
)

config = load_config()

# no, this is wrong if (config.do_rotate == True)
# scan_x = config.scan_x
# scan_y = config.scan_y
# config.scan_aspect = scan_x / scan_y
# ASPECT = config.scan_aspect

ASPECT = config.rotated_scan_aspect

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def save_dbg(img, path):
    ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, img)

def percentile_threshold(gray):
    high_p = np.percentile(gray, THRESH_HIGH_PERCENTILE)
    thr = max(THRESH_MIN, int(high_p * 0.95))
    _, mask = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
    return mask, thr, int(high_p)

def detect_vertical_streaks(mask, approx_width=3, length_thresh_ratio=0.15):
    h, w = mask.shape
    kx = approx_width
    ky = max(15, int(h * 0.02))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
    long_vertical = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(long_vertical, connectivity=8)
    streak_mask = np.zeros_like(mask)
    length_thresh = max(10, int(h * length_thresh_ratio))
    for i in range(1, num_labels):
        x, y, ww, hh, area = stats[i]
        if hh >= length_thresh and ww <= max(5, int(w * 0.01)):
            streak_mask[labels == i] = 255
    return streak_mask

def keep_largest_component(mask):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    out = np.zeros_like(mask)
    out[labels == best] = 255
    return out

def contour_to_pts(contour):
    return contour.reshape(-1,2)

def fit_line_ransac(pts, iterations=RANSAC_ITER, inlier_dist=RANSAC_INLIER_DIST, min_inliers=RANSAC_MIN_INLIERS):
    if len(pts) < 2:
        raise ValueError("Not enough points")
    best_inliers = None
    best_model = None
    n = len(pts)
    ptsf = pts.astype(np.float32)
    for _ in range(iterations):
        i1, i2 = random.sample(range(n), 2)
        p1 = ptsf[i1]; p2 = ptsf[i2]
        vx = float(p2[0] - p1[0]); vy = float(p2[1] - p1[1])
        if vx == 0 and vy == 0:
            continue
        dists = np.abs(vy*(ptsf[:,0]-p1[0]) - vx*(ptsf[:,1]-p1[1])) / (math.hypot(vx, vy) + 1e-12)
        inliers = dists <= inlier_dist
        cnt = int(inliers.sum())
        if cnt >= min_inliers and (best_inliers is None or cnt > int(best_inliers.sum())):
            best_inliers = inliers.copy()
            best_model = (vx, vy, float(p1[0]), float(p1[1]))
    if best_model is None:
        vx, vy, x0, y0 = cv2.fitLine(ptsf, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        inlier_mask = np.ones(len(pts), dtype=bool)
        return float(vx), float(vy), float(x0), float(y0), inlier_mask
    inlier_pts = ptsf[best_inliers]
    vx, vy, x0, y0 = cv2.fitLine(inlier_pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    inlier_mask = best_inliers
    return float(vx), float(vy), float(x0), float(y0), inlier_mask

def intersect_lines(l1, l2):
    vx1, vy1, x1, y1 = l1
    vx2, vy2, x2, y2 = l2
    A = np.array([[vx1, -vx2], [vy1, -vy2]], dtype=np.float32)
    b = np.array([x2 - x1, y2 - y1], dtype=np.float32)
    det = np.linalg.det(A)
    if abs(det) < 1e-8:
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    t1, t2 = np.linalg.solve(A, b)
    xi = x1 + t1 * vx1
    yi = y1 + t1 * vy1
    return float(xi), float(yi)

def build_affine_without_shear(pt_v_top, pt_v_bot, expected_w, expected_h, bad_on_left, img, dbgdir=None):
    """
    Build a source triangle that enforces orthogonal page axes (no shear) using:
      - pt_v_top: intersection of top line and good vertical (top-right when good vertical is right)
      - pt_v_bot: intersection of bottom line and good vertical (bottom-right when good vertical is right)
    Returns (M_aff, src_corners_dict, dst_corners_dict, reason)
    - M_aff: 2x3 affine matrix or None on failure
    - src_corners_dict: dict of TL, TR, BL, BR (np.float32 2-vectors)
    - dst_corners_dict: dict of TL, TR, BL, BR in destination coordinates
    - reason: diagnostic string
    """
    # convert to vectors
    pt_v_top = np.array(pt_v_top, dtype=np.float32)
    pt_v_bot = np.array(pt_v_bot, dtype=np.float32)

    # compute direction along top edge: vector from bottom intersection to top intersection projected horizontally
    # but we don't have explicit top_line direction vector here; we will approximate using (pt_v_top - pt_v_bot) rotated?
    # Better: caller should supply top_line direction (vx,vy). If you don't have it, approximate from nearby contour points.
    # For compatibility with your code, expect you have top_vx, top_vy; if not, fall back to vector between two top contour points.
    # Here we'll assume top_vx, top_vy are available in outer scope; otherwise compute from pt_v_top->pt_v_bot displacement projected perpendicular:
    # To keep this helper self-contained, we compute top_dir as average of (pt_v_top_to_some_right) - but simplest robust approach:
    # compute approximate top direction by taking small offset vector along top by sampling image gradient: fallback to (1,0).

    # --- Here caller should provide top_unit; we'll compute a safe top_unit using neighbor pixels if available ---
    # We'll attempt to derive a top_unit by sampling a short step along the image: use vector from pt_v_top to pt_v_top projected to image center
    cx, cy = img.shape[1] / 2.0, img.shape[0] / 2.0
    # prefer vector pointing rightwards by projecting displacement to center
    approx = pt_v_top - np.array([cx, cy], dtype=np.float32)
    if np.linalg.norm(approx) < 1e-6:
        approx = np.array([1.0, 0.0], dtype=np.float32)
    ux, uy = approx / (np.linalg.norm(approx) + 1e-12)

    # Force ux positive (pointing to the right) — we want u to be rightward
    if ux < 0:
        ux, uy = -ux, -uy

    # make perpendicular downwards: p = (-uy, ux)
    px, py = -uy, ux

    # normalize again (just in case)
    n_u = math.hypot(ux, uy)
    n_p = math.hypot(px, py)
    if n_u < 1e-6 or n_p < 1e-6:
        return None, None, None, "degenerate_axes"
    ux, uy = ux / n_u, uy / n_u
    px, py = px / n_p, py / n_p

    # Now construct corners. We know pt_v_top/pt_v_bot correspond to the known vertical edge:
    # If bad_on_left is True, good vertical is the RIGHT edge => pt_v_top == TR, pt_v_bot == BR.
    # If bad_on_left is False, good vertical is LEFT edge => pt_v_top == TL, pt_v_bot == BL.
    if bad_on_left:
        src_TR = pt_v_top
        src_BR = pt_v_bot
        # compute TL and BL by moving left along top unit by expected_w
        src_TL = src_TR - np.array([ux, uy], dtype=np.float32) * float(expected_w)
        src_BL = src_BR - np.array([ux, uy], dtype=np.float32) * float(expected_w)
    else:
        src_TL = pt_v_top
        src_BL = pt_v_bot
        # compute TR and BR by moving right along top unit by expected_w
        src_TR = src_TL + np.array([ux, uy], dtype=np.float32) * float(expected_w)
        src_BR = src_BL + np.array([ux, uy], dtype=np.float32) * float(expected_w)

    # Optional: enforce vertical height by projecting (TL->BL) onto perp and rescale to expected_h
    # Compute current vertical vector and its projection onto perp (px,py)
    cur_v = src_BL - src_TL
    proj = cur_v.dot(np.array([px, py], dtype=np.float32))
    # if projection is tiny, fall back but otherwise adjust BL to ensure exact page height
    if abs(proj) > 1e-6:
        # adjust scale to exact expected_h
        scale = float(expected_h) / proj
        # recompute BL, BR as TL + perp*expected_h and TR + perp*expected_h
        src_BL = src_TL + np.array([px, py], dtype=np.float32) * float(expected_h)
        src_BR = src_TR + np.array([px, py], dtype=np.float32) * float(expected_h)
    else:
        # if proj nearly zero, we cannot rely on vertical measurement; still use perpendicular step
        src_BL = src_TL + np.array([px, py], dtype=np.float32) * float(expected_h)
        src_BR = src_TR + np.array([px, py], dtype=np.float32) * float(expected_h)

    # Build src/dst dicts
    src_corners = {
        "TL": src_TL.astype(np.float32),
        "TR": src_TR.astype(np.float32),
        "BL": src_BL.astype(np.float32),
        "BR": src_BR.astype(np.float32)
    }
    dst_corners = {
        "TL": np.array([0.0, 0.0], dtype=np.float32),
        "TR": np.array([expected_w - 1.0, 0.0], dtype=np.float32),
        "BL": np.array([0.0, expected_h - 1.0], dtype=np.float32),
        "BR": np.array([expected_w - 1.0, expected_h - 1.0], dtype=np.float32)
    }

    # choose triangle for affine depending on orientation (map TL, BL, TR -> dst TL, BL, TR)
    src_tri = np.vstack([src_corners["TL"], src_corners["BL"], src_corners["TR"]]).astype(np.float32)
    dst_tri = np.vstack([dst_corners["TL"], dst_corners["BL"], dst_corners["TR"]]).astype(np.float32)

    # validate triangle
    area = abs(0.5 * (src_tri[0,0]*(src_tri[1,1]-src_tri[2,1]) + src_tri[1,0]*(src_tri[2,1]-src_tri[0,1]) + src_tri[2,0]*(src_tri[0,1]-src_tri[1,1])))
    if area < 1.0 or not np.isfinite(src_tri).all():
        return None, src_corners, dst_corners, f"invalid_src_tri_area={area:.3f}"

    try:
        M = cv2.getAffineTransform(src_tri, dst_tri)
    except cv2.error as e:
        return None, src_corners, dst_corners, f"getAffineTransform_failed:{e}"

    # debug overlay: draw corners & axes if dbgdir provided
    if dbgdir is not None:
        vis = img.copy()
        def draw_pt(pt, color=(0,0,255), tag=""):
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 6, color, -1)
            if tag:
                cv2.putText(vis, tag, (int(pt[0]+6), int(pt[1]-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        draw_pt(src_corners["TL"], (0,255,0), "TL")
        draw_pt(src_corners["TR"], (0,128,255), "TR")
        draw_pt(src_corners["BL"], (255,0,0), "BL")
        draw_pt(src_corners["BR"], (255,255,0), "BR")
        # draw axis arrows from TL
        origin = src_corners["TL"]
        arrow_u = origin + np.array([ux, uy], dtype=np.float32) * 80.0
        arrow_p = origin + np.array([px, py], dtype=np.float32) * 80.0
        cv2.arrowedLine(vis, (int(origin[0]), int(origin[1])), (int(arrow_u[0]), int(arrow_u[1])), (255,0,255), 2)
        cv2.arrowedLine(vis, (int(origin[0]), int(origin[1])), (int(arrow_p[0]), int(arrow_p[1])), (0,255,255), 2)
        cv2.imwrite(os.path.join(dbgdir, "debug_axes_and_corners.png"), vis)

    return M, src_corners, dst_corners, "ok"

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

def split_edge_candidates(contour, bad_on_left):
    pts = contour.reshape(-1, 2)

    xs = pts[:,0]
    ys = pts[:,1]

    w = xs.max() - xs.min()
    h = ys.max() - ys.min()

    margin_x = w * 0.15
    margin_y = h * 0.15

    # outside edge
    if bad_on_left:
        outside = pts[xs > xs.max() - margin_x]
    else:
        outside = pts[xs < xs.min() + margin_x]

    # top edge
    top = pts[ys < ys.min() + margin_y]

    # bottom edge
    bottom = pts[ys > ys.max() - margin_y]

    return top, bottom, outside

def line_angle(line):
    vx, vy, _, _ = line
    return math.atan2(vy, vx)

def horizontal_line_angle(line):
    vx, vy, _, _ = line

    a = math.degrees(math.atan2(vy, vx))

    while a < -45:
        a += 180

    while a > 45:
        a -= 180

    return a

def vertical_line_angle(line):
    vx, vy, _, _ = line

    a = math.degrees(math.atan2(vy, vx))

    while a < 45:
        a += 180

    while a > 135:
        a -= 180

    return a

def normalize_angle_deg(a):
    while a < -90:
        a += 180
    while a > 90:
        a -= 180
    return a

def get_gray_mask_contours(img, dbgdir):
    # --- threshold for geometry only ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask_init, thr, hp = percentile_threshold(gray)
    if cv2.mean(gray)[0] < 127:
        mask_init = cv2.bitwise_not(mask_init)
    if DEBUG:
        save_dbg(gray, os.path.join(dbgdir, "01_gray.png"))
        save_dbg(mask_init, os.path.join(dbgdir, f"02_thresh_thr{thr}_hp{hp}.png"))

    # without a9dc2b24e6b0d1d49b6fc232223d6431ba3442a5 bad: fix perspective transform for broken ADF scanners
    # Invert if necessary, so the page is always "white" for thresholding
    # We'll use Otsu's method to detect high-contrast area
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Determine whether the page is darker or lighter than background
    mean_val = cv2.mean(gray, mask=None)[0]
    if mean_val < 127:  # dark page: invert mask
        mask = cv2.bitwise_not(mask)

    # Morphology to remove small gaps
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return gray, mask, contours

def repair_binding(img, bad_on_left, width=50):

    h,w = img.shape[:2]

    if bad_on_left:
        sample_range = range(width)
        fill_range = range(width)
        sample_x = width
    else:
        sample_range = range(w-width,w)
        fill_range = range(w-width,w)
        sample_x = w-width

    for y in range(h):

        if bad_on_left:
            sample = img[y, width:width+20]
            color = np.mean(sample,axis=0)
            img[y,:width] = color

        else:
            sample = img[y,w-width-20:w-width]
            color = np.mean(sample,axis=0)
            img[y,w-width:] = color

    return img

def px_of_mm(mm, dpi):
    return mm * dpi / 25.4

EDGE_TOP = 0
EDGE_BOTTOM = 1
EDGE_LEFT = 2
EDGE_RIGHT = 3

def detect_edge_points(gray, mask, edge, margin):
    h, w = mask.shape
    pts = []

    if edge == EDGE_TOP:
        y_max = min(margin, h)
        for x in range(w):
            # TODO why 255
            ys = np.where(mask[:y_max, x] == 255)[0]
            if len(ys):
                pts.append((x, ys[0]))

        # Transition verification
        #
        # The first white pixel isn't always the page.
        # It might be dust, glare, or text sticking out.
        #
        # Keep only points that actually separate gray background
        # from the white page.
        pts = verify_horizontal(
            gray,
            # pts,
            np.asarray(pts, np.float32),
            top_edge=True
        )

    elif edge == EDGE_BOTTOM:
        y_min = max(0, h - margin)
        for x in range(w):
            ys = np.where(mask[y_min:, x] == 255)[0]
            if len(ys):
                pts.append((x, y_min + ys[-1]))

        # Transition verification
        pts = verify_horizontal(
            gray,
            # pts,
            np.asarray(pts, np.float32),
            top_edge=False
        )

    elif edge == EDGE_LEFT:
        x_max = min(margin, w)
        for y in range(h):
            xs = np.where(mask[y, :x_max] == 255)[0]
            if len(xs):
                pts.append((xs[0], y))

        # Transition verification
        pts = verify_vertical(
            gray,
            # pts,
            np.asarray(pts, np.float32),
            right_edge=False
        )

    elif edge == EDGE_RIGHT:
        x_min = max(0, w - margin)
        for y in range(h):
            xs = np.where(mask[y, x_min:] == 255)[0]
            if len(xs):
                pts.append((x_min + xs[-1], y))

        # Transition verification
        pts = verify_vertical(
            gray,
            # pts,
            np.asarray(pts, np.float32),
            right_edge=True
        )

    return np.asarray(pts, np.float32)

# def is_vertical_transition(gray, x, y):

#     if x < 3 or x >= gray.shape[1]-3:
#         return False

#     left = np.mean(gray[y, x-3:x])
#     right = np.mean(gray[y, x:x+3])

#     return abs(left-right) > 40

def verify_horizontal(gray, points, top_edge):
    good = []

    H, W = gray.shape

    for x, y in points.astype(int):

        if y < 3 or y >= H-3:
            continue

        if top_edge:
            outside = np.mean(gray[y-3:y, x])
            inside  = np.mean(gray[y:y+3, x])
        else:
            outside = np.mean(gray[y:y+3, x])
            inside  = np.mean(gray[y-3:y, x])

        r'''
        # must have strong contrast
        if abs(float(inside) - float(outside)) < 40:
            continue
        '''
        r'''
        # must have strong contrast
        if top_edge:
            if not is_horizontal_transition(gray, x, y):
                continue
        else:
            if not is_vertical_transition(gray, x, y):
                continue
        '''
        # background is always gray
        # Require the outside to actually look like scanner background
        BACKGROUND_MIN = 80
        BACKGROUND_MAX = 180
        if not (BACKGROUND_MIN <= outside <= BACKGROUND_MAX):
            continue

        # outside should be scanner gray
        if not (60 < outside < 200):
            continue

        # inside should be brighter
        if inside <= outside:
            continue

        good.append((x, y))

    return np.asarray(good, np.float32)

def verify_vertical(gray, points, right_edge):
    good = []

    H, W = gray.shape

    for x, y in points.astype(int):

        if x < 3 or x >= W-3:
            continue

        if right_edge:
            outside = np.mean(gray[y, x:x+3])
            inside  = np.mean(gray[y, x-3:x])
        else:
            outside = np.mean(gray[y, x-3:x])
            inside  = np.mean(gray[y, x:x+3])

        if abs(float(inside) - float(outside)) < 40:
            continue

        if not (60 < outside < 200):
            continue

        if inside <= outside:
            continue

        good.append((x, y))

    return np.asarray(good, np.float32)

# TODO refactor x/y

def reject_outliers_horizontal(pts, tolerance=40):
    if len(pts) == 0:
        return pts

    ys = pts[:,1]
    median = np.median(ys)

    return pts[np.abs(ys - median) < tolerance]

def reject_outliers_vertical(pts, tolerance=40):
    if len(pts) == 0:
        return pts

    xs = pts[:,0]
    median = np.median(xs)

    return pts[np.abs(xs - median) < tolerance]

def process_image(in_path, out_path):
    """
    Robust page extraction that handles missing top (or bottom) edges.
    - preserves original colors (warps the original image)
    - builds orthogonal axes (no shear)
    - if top is missing, uses bottom + good vertical to infer top by moving up expected_h
    - fills outside the filled page polygon with pure white
    - extensive debug output in OUTPUT_DIR/debug/<page_num>/
    """
    # ---------- small helpers ----------
    def ensure_dir(p):
        os.makedirs(p, exist_ok=True)

    def save_dbg(img, path):
        ensure_dir(os.path.dirname(path))
        cv2.imwrite(path, img)

    def percentile_threshold(gray):
        high_p = np.percentile(gray, THRESH_HIGH_PERCENTILE)
        thr = max(THRESH_MIN, int(high_p * 0.95))
        _, mask = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
        return mask, thr, int(high_p)

    def keep_largest_component(mask):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels <= 1:
            return mask
        areas = stats[1:, cv2.CC_STAT_AREA]
        best = 1 + int(np.argmax(areas))
        out = np.zeros_like(mask)
        out[labels == best] = 255
        return out

    def detect_vertical_streaks(mask, approx_width=3, length_thresh_ratio=0.15):
        h, w = mask.shape
        kx = approx_width
        ky = max(15, int(h * 0.02))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
        long_vertical = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(long_vertical, connectivity=8)
        streak_mask = np.zeros_like(mask)
        length_thresh = max(10, int(h * length_thresh_ratio))
        for i in range(1, num_labels):
            x, y, ww, hh, area = stats[i]
            if hh >= length_thresh and ww <= max(5, int(w * 0.01)):
                streak_mask[labels == i] = 255
        return streak_mask

    def contour_to_pts(c):
        return c.reshape(-1, 2)

    def fit_line_ransac(pts, iterations=RANSAC_ITER, inlier_dist=RANSAC_INLIER_DIST, min_inliers=RANSAC_MIN_INLIERS):
        if len(pts) < 2:
            raise ValueError("Not enough points for line fit")
        best_inliers = None
        best_cnt = 0
        best_model = None
        ptsf = pts.astype(np.float32)
        n = len(ptsf)
        for _ in range(iterations):
            i1, i2 = random.sample(range(n), 2)
            p1 = ptsf[i1]; p2 = ptsf[i2]
            vx = float(p2[0] - p1[0]); vy = float(p2[1] - p1[1])
            if abs(vx) < 1e-6 and abs(vy) < 1e-6:
                continue
            dists = np.abs(vy*(ptsf[:,0]-p1[0]) - vx*(ptsf[:,1]-p1[1])) / (math.hypot(vx, vy) + 1e-12)
            inliers = dists <= inlier_dist
            cnt = int(inliers.sum())
            if cnt >= min_inliers and cnt > best_cnt:
                best_cnt = cnt
                best_inliers = inliers.copy()
                best_model = (vx, vy, float(p1[0]), float(p1[1]))
        if best_model is None:
            vx, vy, x0, y0 = cv2.fitLine(ptsf, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
            inlier_mask = np.ones(len(ptsf), dtype=bool)
            return float(vx), float(vy), float(x0), float(y0), inlier_mask
        inlier_pts = ptsf[best_inliers]
        vx, vy, x0, y0 = cv2.fitLine(inlier_pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        return float(vx), float(vy), float(x0), float(y0), best_inliers

    def intersect_lines(l1, l2):
        vx1, vy1, x1, y1 = l1
        vx2, vy2, x2, y2 = l2
        A = np.array([[vx1, -vx2], [vy1, -vy2]], dtype=np.float32)
        b = np.array([x2 - x1, y2 - y1], dtype=np.float32)
        det = np.linalg.det(A)
        if abs(det) < 1e-8:
            return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        t1, t2 = np.linalg.solve(A, b)
        xi = x1 + t1 * vx1
        yi = y1 + t1 * vy1
        return float(xi), float(yi)

    # ---------- start ----------
    fname = os.path.basename(in_path)
    m = re.match(r"^(\d+)", fname)
    page_num = int(m.group(1)) if m else 0
    bad_on_left = (page_num % 2 == 1)

    img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print("Failed to read", in_path); return
    input_is_grayscale = (len(img.shape) == 2)
    if len(img.shape) == 2:
        # OpenCV expects 3-channel images for many operations
        # later convert back to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    H_img, W_img = img.shape[:2]

    # fixed
    # # FIXME wrong H_img?
    # expected_w = int(round(ASPECT * H_img))
    # expected_h = H_img

    dbgdir = os.path.join(OUTPUT_DIR, "debug", f"{page_num:03d}")
    if DEBUG: ensure_dir(dbgdir)

# Compute the expected ranges of margins
    # NOTE the scanner removes the "scan top" margin
    # so in the X direction, we have only one margin
    # so here we divide by 4, not by 2
    margin_range_x_mm = (
        config.rotated_margined_scan_x -
        config.rotated_scan_x
    ) / 4.0
    margin_range_y_mm = (
        config.rotated_margined_scan_y -
        config.rotated_scan_y
    ) / 2.0

    if DEBUG:
        print(f"config.rotated_margined_scan: ({config.rotated_margined_scan_x}, {config.rotated_margined_scan_y}) mm")
        print(f"config.rotated_scan: ({config.rotated_scan_x}, {config.rotated_scan_y}) mm")
        print(f"margin_range: ({margin_range_x_mm}, {margin_range_y_mm}) mm")

    # Convert them to pixels
    margin_range_x_px = px_of_mm(
        margin_range_x_mm,
        config.scan_resolution
    )
    margin_range_y_px = px_of_mm(
        margin_range_y_mm,
        config.scan_resolution
    )

    # TODO rename, move to config

    SEARCH_MARGIN_FACTOR = 2.0
    SEARCH_MARGIN_ADD_MM = 5

    SEARCH_MARGIN_FACTOR = 1.2
    SEARCH_MARGIN_ADD_MM = 2

    # debug: dont increase
    # SEARCH_MARGIN_FACTOR = 1; SEARCH_MARGIN_ADD_MM = 0

    # Then enlarge them
    margin_range_x_px *= SEARCH_MARGIN_FACTOR
    margin_range_y_px *= SEARCH_MARGIN_FACTOR
    search_margin_add_px = px_of_mm(
        SEARCH_MARGIN_ADD_MM,
        config.scan_resolution
    )
    margin_range_x_px += search_margin_add_px
    margin_range_y_px += search_margin_add_px

    # we need integers for array indices
    margin_range_x_px = int(math.ceil(margin_range_x_px))
    margin_range_y_px = int(math.ceil(margin_range_y_px))

# Step 1: Segment page vs. gray background

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Otsu is usually sufficient here
    _, page_mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Page should be white
    if np.mean(gray[page_mask == 255]) < np.mean(gray[page_mask == 0]):
        page_mask = cv2.bitwise_not(page_mask)

    page_mask = cv2.morphologyEx(
        page_mask,
        cv2.MORPH_CLOSE,
        np.ones((5,5), np.uint8)
    )

# TODO rename
    mask = page_mask

# Step 2: Scan for transitions
    # Step 3: Transition verification

    if bad_on_left:
        outside_pts = detect_edge_points(
            gray,
            mask,
            EDGE_RIGHT,
            margin_range_x_px,
        )
    else:
        outside_pts = detect_edge_points(
            gray,
            mask,
            EDGE_LEFT,
            margin_range_x_px,
        )

    top_pts = detect_edge_points(
        gray,
        mask,
        EDGE_TOP,
        margin_range_y_px,
    )

    bottom_pts = detect_edge_points(
        gray,
        mask,
        EDGE_BOTTOM,
        margin_range_y_px,
    )

# Step 4: Reject isolated outliers
    # Use a median filter before RANSAC.

    top_pts = reject_outliers_horizontal(top_pts)
    bottom_pts = reject_outliers_horizontal(bottom_pts)
    outside_pts = reject_outliers_vertical(outside_pts)

# Step 5: RANSAC

    top_line = fit_line_ransac(top_pts)[:4]
    bottom_line = fit_line_ransac(bottom_pts)[:4]
    outside_line = fit_line_ransac(outside_pts)[:4]

# FIXME merge old code with new code

# # TODO remove

    # gray, mask, contours = get_gray_mask_contours(img, dbgdir)

    # if not contours:
    #     print(f"Warning: no contours found in {in_path}")
    #     return

    # page_contour = max(contours, key=cv2.contourArea)

    if config.use_three_edge_deskew:

        # top_pts, bottom_pts, outside_pts = split_edge_candidates(
        #     page_contour,
        #     bad_on_left
        # )

        # top_line = fit_line_ransac(top_pts)[:4]
        # bottom_line = fit_line_ransac(bottom_pts)[:4]
        # outside_line = fit_line_ransac(outside_pts)[:4]

        # old
        # top_angle = math.degrees(line_angle(top_line))
        # bottom_angle = math.degrees(line_angle(bottom_line))
        # outside_angle = math.degrees(line_angle(outside_line))

        # new
        top_angle = horizontal_line_angle(top_line)
        bottom_angle = horizontal_line_angle(bottom_line)
        outside_angle = vertical_line_angle(outside_line)

        if DEBUG:
            # start debug prints
            print()
            print(f"line 570: page_num={page_num}")

        rotation_error = -1 * (
            -top_angle
            -bottom_angle
        ) / 2.0

        Mrot = cv2.getRotationMatrix2D(
            (W_img/2, H_img/2),
            rotation_error,
            1.0
        )

        if DEBUG:
            print(
                f"line 575: before rotation: "
                f"top_angle={top_angle:.3f} "
                f"bottom_angle={bottom_angle:.3f} "
                f"outside_angle={outside_angle:.3f}"
            )

        rotated = cv2.warpAffine(
            img,
            Mrot,
            (W_img, H_img),
            borderValue=(255,255,255)
        )

        Hr, Wr = rotated.shape[:2]

        if DEBUG:
            print("line 580: rotated size", Wr, Hr)

        # img = rotated # ?

# re-detect lines in the rotated image

        # gray, mask, contours = get_gray_mask_contours(img, dbgdir)
        gray, mask, contours = get_gray_mask_contours(rotated, dbgdir)

        if 1:
            top_angle2 = horizontal_line_angle(top_line)
            bottom_angle2 = horizontal_line_angle(bottom_line)
            outside_angle2 = vertical_line_angle(outside_line)

            if DEBUG:
                print(
                    f"line 620: after rotation and re-fitting: "
                    f"top_angle2={top_angle2:.3f} "
                    f"bottom_angle2={bottom_angle2:.3f} "
                    f"outside_angle2={outside_angle2:.3f}"
                )

        if not contours:
            print(f"line 630: Warning: no contours found in {in_path}")
            return

        page_contour = max(contours, key=cv2.contourArea)

top_pts, bottom_pts, outside_pts = split_edge_candidates(
            page_contour,
            bad_on_left
        )

        top_line = fit_line_ransac(top_pts)[:4]
        bottom_line = fit_line_ransac(bottom_pts)[:4]
        outside_line = fit_line_ransac(outside_pts)[:4]

        # fixed
        # # FIXME H_img is wrong
        # expected_h = H_img
        # expected_w = int(round(ASPECT * expected_h))

        vx, vy, x0, y0 = outside_line

        outside_top = intersect_lines(
            outside_line,
            top_line
        )

        outside_bottom = intersect_lines(
            outside_line,
            bottom_line
        )

        if 0:
            # approximated page height
            page_height = math.dist(
                outside_top,
                outside_bottom
            )
        else:
            # perpendicular page height
            page_height = math.hypot(
                outside_bottom[0] - outside_top[0],
                outside_bottom[1] - outside_top[1]
            )

        # no. this fails to reconstruct the page height...
        # TODO try to solve this with the average height of multiple pages
        # assuming all pages must have the same height
        # also allowing the user to specify a scale_y factor
        if 0:
            if config.do_rotate == False:
                # fix scan height
                # document scanners can distort scans in the Y direction
                # FIXME honor config.do_rotate
                # if (config.do_rotate == True)
                # then all pages are rotated by 90 or 270 degrees
                # so the scanner's Y errors become our X errors
                # so we need scale_x to fix the page width
                if 0:
                    # use rotated_scan_y and page_height
                    rotated_scan_y_px = px_of_mm(config.rotated_scan_y, config.scan_resolution)
                    target_h = rotated_scan_y_px
                    scale_y = target_h / page_height
                elif 1:
                    # use rotated_margined_scan_y and H_img
                    rotated_margined_scan_y_px = px_of_mm(config.rotated_margined_scan_y, config.scan_resolution)
                    target_h = rotated_margined_scan_y_px
                    scale_y = target_h / H_img
                else:
                    # no, this fails because the scanner removes one edge
                    # so actual_aspect is always wrong...
                    #
                    # use rotated_margined_scan_y and H_img
                    # expected: what we ordered from the scanner
                    expected_aspect = (
                        config.rotated_margined_scan_x /
                        config.rotated_margined_scan_y
                    )
                    expected_height = config.rotated_margined_scan_y
                    # actual: what the scanner gave us
                    actual_aspect = W_img / H_img
                    actual_height = H_img
                    # we assume the scanner always returns correct X coordinates
                    # and all errors are only in Y coordinates
                    # expected_aspect / actual_aspect = actual_height / expected_height
                    actual_height_2 = expected_aspect / actual_aspect / expected_height
                    expected_aspect_factor = expected_aspect / actual_aspect
                    # the scale of actual_height relative to actual_height_2
                    actual_height_scale = actual_height / actual_height_2
                    if DEBUG:
                        print(f"expected_aspect={expected_aspect} actual_aspect={actual_aspect} expected_aspect_factor={expected_aspect_factor}")
                        print(f"expected_height={expected_height} actual_height={actual_height} actual_height_2={actual_height_2} actual_height_scale={actual_height_scale}")
                    target_h = actual_height_2
                    scale_y = target_h / H_img

                # debug: manually set the scale_y factor
                # scale_y = 1 / 1.03 # shrink the page height by 3%

                scale_y_tolerance = 0.001 # 0.1%

                if scale_y < (1 - scale_y_tolerance) or (1 + scale_y_tolerance) < scale_y:
                    if DEBUG:
                        print(f"line 680: scale_y={scale_y} target_h={target_h} page_height={page_height} -> scaling height")
                    rotated = cv2.resize(
                        rotated,
                        None,
                        fx=1.0,
                        fy=scale_y,
                        interpolation=cv2.INTER_CUBIC
                    )
                    page_height = page_height * scale_y

                    # re-detect lines in the scaled image

                    # gray, mask, contours = get_gray_mask_contours(img, dbgdir)
                    gray, mask, contours = get_gray_mask_contours(rotated, dbgdir)

                    if 1:
                        top_angle2 = horizontal_line_angle(top_line)
                        bottom_angle2 = horizontal_line_angle(bottom_line)
                        outside_angle2 = vertical_line_angle(outside_line)
                        if DEBUG:
                            print(
                                f"line 620: after rotation and re-fitting: "
                                f"top_angle2={top_angle2:.3f} "
                                f"bottom_angle2={bottom_angle2:.3f} "
                                f"outside_angle2={outside_angle2:.3f}"
                            )

                    if not contours:
                        print(f"line 630: Warning: no contours found in {in_path}")
                        return

                    page_contour = max(contours, key=cv2.contourArea)

                    top_pts, bottom_pts, outside_pts = split_edge_candidates(
                        page_contour,
                        bad_on_left
                    )

                    top_line = fit_line_ransac(top_pts)[:4]
                    bottom_line = fit_line_ransac(bottom_pts)[:4]
                    outside_line = fit_line_ransac(outside_pts)[:4]

                    # fixed
                    # # FIXME H_img is wrong
                    # expected_h = H_img
                    # expected_w = int(round(ASPECT * expected_h))

                    vx, vy, x0, y0 = outside_line

                    outside_top = intersect_lines(
                        outside_line,
                        top_line
                    )

                    outside_bottom = intersect_lines(
                        outside_line,
                        bottom_line
                    )

                else:
                    if DEBUG:
                        print(f"line 680: scale_y={scale_y} target_h={target_h} page_height={page_height} -> not scaling height")

            else:
                # config.do_rotate == True
                # fix scan width
                # Raw scans are rotated by 90/270 degrees.
                # Therefore: scanner Y errors -> image X errors
                # We must correct width (scale_x), not height (scale_y).
                if 1:
                    # The detected outside edge gives us the real page height
                    # in image coordinates.  After rotation this corresponds to
                    # scanner X, which is assumed correct.
                    #
                    # The missing/damaged dimension is the page width.

                    rotated_margined_scan_x_px = px_of_mm(
                        config.rotated_margined_scan_x,
                        config.scan_resolution
                    )

                    target_w = rotated_margined_scan_x_px

                    actual_w = math.dist(
                        outside_top,
                        outside_bottom
                    )

                    # This is actually the width in scanner coordinates because
                    # the document is rotated.
                    scale_x = target_w / actual_w

                    scale_x_tolerance = 0.001  # 0.1%

                    if (
                        scale_x < (1 - scale_x_tolerance)
                        or
                        scale_x > (1 + scale_x_tolerance)
                    ):
                        if DEBUG:
                            print(
                                "line 680:",
                                f"scale_x={scale_x}",
                                f"target_w={target_w}",
                                f"actual_w={actual_w}",
                                "-> scaling width"
                            )

                        rotated = cv2.resize(
                            rotated,
                            None,
                            fx=scale_x,
                            fy=1.0,
                            interpolation=cv2.INTER_CUBIC
                        )

                        # Update image size
                        Hr, Wr = rotated.shape[:2]

                        if DEBUG:
                            print(
                                "line 690: after scale_x",
                                f"Wr={Wr}",
                                f"Hr={Hr}"
                            )

                        # Re-detect page after scaling
                        gray, mask, contours = get_gray_mask_contours(
                            rotated,
                            dbgdir
                        )

                        if not contours:
                            print(
                                f"line 700: Warning: no contours found after scaling {in_path}"
                            )
                            return

                        page_contour = max(
                            contours,
                            key=cv2.contourArea
                        )

                        top_pts, bottom_pts, outside_pts = split_edge_candidates(
                            page_contour,
                            bad_on_left
                        )

                        top_line = fit_line_ransac(top_pts)[:4]
                        bottom_line = fit_line_ransac(bottom_pts)[:4]
                        outside_line = fit_line_ransac(outside_pts)[:4]

                        outside_top = intersect_lines(
                            outside_line,
                            top_line
                        )

                        outside_bottom = intersect_lines(
                            outside_line,
                            bottom_line
                        )

                    else:
                        if DEBUG:
                            print(
                                "line 680:",
                                f"scale_x={scale_x}",
                                "-> not scaling width"
                            )

expected_h = int(round(page_height))

        # ASPECT = x / y
        # x = y * ASPECT

        # expand the binding edge to expected_w
        # expected_w = int(round(expected_h * ASPECT))
        expected_w = int(round(page_height * ASPECT))

        if 0:
            # crop as a rectangle
            x_out = (
                outside_top[0]
                +
                outside_bottom[0]
            ) / 2

            x_out = int(round(x_out))

            # if bad_on_left:
            #     x0_page = x_out
            #     x1_page = x_out + expected_w
            # else:
            #     x0_page = x_out - expected_w
            #     x1_page = x_out

            if bad_on_left:
                # outside edge is RIGHT
                x1_page = x_out
                x0_page = x_out - expected_w
            else:
                # outside edge is LEFT
                x0_page = x_out
                x1_page = x_out + expected_w

            # clamp
            # x0_page = max(0, x0_page)
            # x1_page = min(rotated.shape[1], x1_page)

            y_top = int(round(outside_top[1]))
            y_bottom = int(round(outside_bottom[1]))

            src_x0 = max(0, x0_page)
            src_x1 = min(rotated.shape[1], x1_page)

            src_y0 = max(0, y_top)
            src_y1 = min(rotated.shape[0], y_top + expected_h)

            dst_x0 = src_x0 - x0_page
            dst_y0 = src_y0 - y_top

            if DEBUG:
                print(
                    "line 640:",
                    f"outside_top={outside_top}",
                    f"outside_bottom={outside_bottom}",
                )

            # crop = rotated[
            #     y_top:y_top+expected_h,
            #     x0_page:x1_page
            # ]

            full_crop = np.ones(
                (expected_h, expected_w, 3),
                dtype=rotated.dtype
            ) * 255

            full_crop[
                dst_y0:dst_y0 + (src_y1-src_y0),
                dst_x0:dst_x0 + (src_x1-src_x0)
            ] = rotated[
                src_y0:src_y1,
                src_x0:src_x1
            ]

            # FIXME fill empty area near binding edge
            # currently it is all white
            # but it should copy the vertical pattern near the binding edge

            crop = full_crop

            if DEBUG:
                print(
                    "line 650: crop",
                    f"W_img={W_img}",
                    f"H_img={H_img}",
                    f"x0_page={x0_page}",
                    f"x1_page={x1_page}",
                    f"y_top={y_top}",
                    f"y_bottom={y_bottom}",
                    f"src_x0={src_x0}",
                    f"src_x1={src_x1}",
                    f"src_y0={src_y0}",
                    f"src_y1={src_y1}",
                    f"dst_x0={dst_x0}",
                    f"dst_y0={dst_y0}",
                    f"expected_w={expected_w}",
                    f"expected_h={expected_h}",
                )

            if DEBUG:
                vis = rotated.copy()
                cv2.rectangle(
                    vis,
                    (int(x0_page), int(y_top)),
                    (int(x1_page), int(y_bottom)),
                    (0,0,255),
                    3,
                )
                name = f"{page_num:03d}.crop_debug_rotated_line730.jpg"
                print(f"writing {name}")
                cv2.imwrite(name, vis)

        else:
            # crop as a quadrilateral
            # not better than "crop as a rectangle"?
            if 0:
                # expand the binding edge to expected_w
                if bad_on_left:
                    # outside edge is the RIGHT edge
                    x1_top = outside_top[0]
                    x1_bottom = outside_bottom[0]
                    x0_top = x1_top - expected_w
                    x0_bottom = x1_bottom - expected_w
                else:
                    # outside edge is the LEFT edge
                    x0_top = outside_top[0]
                    x0_bottom = outside_bottom[0]
                    x1_top = x0_top + expected_w
                    x1_bottom = x0_bottom + expected_w
            else:
                # dont expand the binding edge to expected_w
                # use only the detected page edges
                # if bad_on_left:
                #     # outside edge is RIGHT edge
                #     x1_top = outside_top[0]
                #     x1_bottom = outside_bottom[0]
                #     # use the actual detected left edge
                #     x0_top = np.min(page_contour[:,0,0])
                #     x0_bottom = x0_top
                # else:
                #     # outside edge is LEFT edge
                #     x0_top = outside_top[0]
                #     x0_bottom = outside_bottom[0]
                #     # use the actual detected right edge
                #     x1_top = np.max(page_contour[:,0,0])
                #     x1_bottom = x1_top
                # problem: page_contour after thresholding may include the background
                # or may not have a reliable missing-edge position.
                # A cleaner temporary solution is to use the detected quadrilateral width
                # from the two horizontal edge intersections
                if bad_on_left:
                    # outside edge is right
                    x1_top = outside_top[0]
                    x1_bottom = outside_bottom[0]
                    # find leftmost detected page boundary
                    x0_top = np.min(top_pts[:,0])
                    x0_bottom = np.min(bottom_pts[:,0])
                else:
                    # outside edge is left
                    x0_top = outside_top[0]
                    x0_bottom = outside_bottom[0]
                    # find rightmost detected page boundary
                    x1_top = np.max(top_pts[:,0])
                    x1_bottom = np.max(bottom_pts[:,0])

                # dont expand the binding edge to expected_w
                expected_w = int(round(
                    math.dist((x0_top, outside_top[1]), (x1_top, outside_top[1]))
                ))

src = np.float32([
                [x0_top, outside_top[1]],
                [x1_top, outside_top[1]],
                [x1_bottom, outside_bottom[1]],
                [x0_bottom, outside_bottom[1]],
            ])

            dst = np.float32([
                [0,0],
                [expected_w-1,0],
                [expected_w-1,expected_h-1],
                [0,expected_h-1],
            ])

            M = cv2.getPerspectiveTransform(src, dst)

            crop = cv2.warpPerspective(
                rotated,
                M,
                (expected_w, expected_h),
                borderValue=(255,255,255)
            )

            if DEBUG:
                print(
                    "line 650: crop",
                    f"W_img={W_img}",
                    f"H_img={H_img}",
                    f"outside_top={outside_top}",
                    f"outside_bottom={outside_bottom}",
                    f"x1_top={x1_top}",
                    f"x1_bottom={x1_bottom}",
                    f"x0_top={x0_top}",
                    f"x0_bottom={x0_bottom}",
                    f"expected_w={expected_w}",
                    f"expected_h={expected_h}",
                )

            if DEBUG:
                vis = rotated.copy()
                # margin range: green
                if bad_on_left:
                    # outside edge is right
                    # no line on the left
                    pts = np.array([
                        [0, margin_range_y_px], # top left
                        [W_img - margin_range_x_px, margin_range_y_px], # top right
                        [W_img - margin_range_x_px, H_img - margin_range_y_px], # bottom right
                        [0, H_img - margin_range_y_px], # bottom left
                    ], np.int32)
                else:
                    # outside edge is left
                    # no line on the right
                    pts = np.array([
                        [W_img, margin_range_y_px], # top right
                        [margin_range_x_px, margin_range_y_px], # top left
                        [margin_range_x_px, H_img - margin_range_y_px], # bottom left
                        [W_img, H_img - margin_range_y_px], # bottom right
                    ], np.int32)
                cv2.polylines(vis, [pts], False, (0,255,0), 3) # (0,255,0) == green?
                # page margin: red
                pts = np.array([
                    [x0_top, outside_top[1]],
                    [x1_top, outside_top[1]],
                    [x1_bottom, outside_bottom[1]],
                    [x0_bottom, outside_bottom[1]],
                ], np.int32)
                cv2.polylines(vis, [pts], True, (0,0,255), 3)
                name = f"{page_num:03d}.crop_debug_rotated_line730.jpg"
                print(f"writing {name}")
                cv2.imwrite(name, vis)

        # crop = rotated[
        #     0:expected_h,
        #     int(x0_page):int(x1_page)
        # ]

        # y_top = round(outside_top[1])
        # y_bottom = round(outside_bottom[1])
        # actual_height = y_bottom - y_top
        # crop = rotated[
        #     y_top:y_top + expected_h,
        #     x0_page:x1_page
        # ]

        # y_top = int(round(outside_top[1]))
        # y_bottom = int(round(outside_bottom[1]))
        # crop = rotated[
        #     y_top:y_bottom,
        #     x0_page:x1_page
        # ]

        if 0:
            # img = repair_binding(img, bad_on_left, width=50)
            if 1:
                # rotated = repair_binding(rotated, bad_on_left, width=50)
                # Hr, Wr = rotated.shape[:2]
                # print("line 660: rotated size", Wr, Hr)
                # warped = rotated
                crop = repair_binding(crop, bad_on_left, width=50)
                Hr, Wr = crop.shape[:2]
                if DEBUG:
                    print("line 660: crop size", Wr, Hr)
            else:
                crop = repair_binding(crop, bad_on_left, width=50)

            if DEBUG:
                print(
                    "line 760: after repair_binding: crop actual:",
                    f"crop.shape[1]={crop.shape[1]}",
                    f"expected_w={expected_w}",
                )

        warped = crop

    else:
        # config.use_three_edge_deskew == False

        raise NotImplementedError("sorry, your book is too tall for your scanner...")

        # FIXME use only two page edges: outside, bottom

        # Approximate contour to quadrilateral
        epsilon = 0.02 * cv2.arcLength(page_contour, True)
        approx = cv2.approxPolyDP(page_contour, epsilon, True)
        if len(approx) != 4:
            approx = cv2.convexHull(page_contour)
            if len(approx) < 4:
                print(f"Warning: not enough points for perspective in {in_path}")
                return
            # pick 4 extreme points
            pts = np.array([
                approx[approx[:,0,0].argmin()][0],  # leftmost
                approx[approx[:,0,1].argmin()][0],  # topmost
                approx[approx[:,0,0].argmax()][0],  # rightmost
                approx[approx[:,0,1].argmax()][0]   # bottommost
            ])
        else:
            pts = approx.reshape(4,2)

        rect = order_points(pts)

        # Perspective transform
        widthA = np.linalg.norm(rect[2] - rect[3])
        widthB = np.linalg.norm(rect[1] - rect[0])
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.linalg.norm(rect[1] - rect[2])
        heightB = np.linalg.norm(rect[0] - rect[3])
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    # # Add internal white border
    # h, w = warped.shape[:2]
    # canvas = np.ones_like(warped) * 255
    # b = BORDER_SIZE
    # canvas[b:h-b, b:w-b] = warped[b:h-b, b:w-b]

    h, w = warped.shape[:2]
    b = BORDER_SIZE

    # Prepare a canvas with the same content as warped
    canvas = warped.copy()

    # Function to compute average color along a strip
    def avg_color_strip(img, axis, start, end, strip_width=1):
        if axis == 'top':
            strip = img[start:end, :, :]
        elif axis == 'bottom':
            strip = img[h-end:h-start, :, :]
        elif axis == 'left':
            strip = img[:, start:end, :]
        elif axis == 'right':
            strip = img[:, w-end:w-start, :]
        else:
            raise ValueError("Invalid axis")
        return np.mean(strip, axis=(0,1)).astype(np.uint8)

    # FIXME preserve patterns near edges
    if 0:
        # Fill borders with local average color
        canvas[0:b, :, :] = avg_color_strip(canvas, 'top', 50, 100)       # top border
        canvas[h-b:h, :, :] = avg_color_strip(canvas, 'bottom', 50, 100)  # bottom border
        canvas[:, 0:b, :] = avg_color_strip(canvas, 'left', 50, 100)      # left border
        canvas[:, w-b:w, :] = avg_color_strip(canvas, 'right', 50, 100)   # right border

    if input_is_grayscale:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

    ensure_dir(os.path.dirname(out_path))
    cv2.imwrite(out_path, canvas)
    print(f"writing {out_path}")

def main():
    ensure_dir(OUTPUT_DIR)
    # TODO use image_format from 030-measure-page-size.txt
    image_format = "jpg"
    files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(f".{image_format}")])
    if not files:
        print("No image files found in", INPUT_DIR)
        return
    for fname in files:
        m = re.match(r"^(\d+)", fname)
        # page_num = int(m.group(1)) if m else 0
        page_num = int(m.group(1)) # can throw
        # if page_num != 14: continue # debug
        if not page_num in (1, 2, 12, 13): continue # debug
        # if not page_num in [340, 345]: continue # debug
        # if page_num < 300: continue # debug
        # if page_num != 320: continue # debug
        in_path = os.path.join(INPUT_DIR, fname)
        out_path = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(out_path):
            continue
        try:
            process_image(in_path, out_path)
        except Exception as exc:
            print(f"Error processing {fname}: {exc}")
            raise

if __name__ == "__main__":
    main()
```

## Response:

7/13/2026, 3:54:44 PM

I think the direction you are heading is correct, but I would change the architecture slightly before adding more heuristics.

Right now the failure mode is:

    gray background
         |
         v
    +------------+
    | white page |
    |            |
    |  orange    |
    |  rectangle |
    |            |
    +------------+

Your threshold mask sees:

    gray -> white -> orange

and because orange may be darker than white, the "last white pixel" logic finds the orange rectangle boundary instead of the real page boundary.

The key insight: **the page edge is not a color transition. It is a transition from "scanner background" to "anything belonging to the page".**

So the detector should not ask:

> where does white start?

It should ask:

> where does the image stop looking like the scanner background?

That suggests these changes:

------------------------------------------------------------------------

## 1. Replace the binary mask edge scan

This:

``` py
ys = np.where(mask[:y_max, x] == 255)[0]
if len(ys):
    pts.append((x, ys[0]))
```

is the root problem.

Instead scan the grayscale directly.

For top edge:

``` text
for each x:

outside region:
    y = 0 ... margin

find first y where:
    average(gray[y:y+n]) != average(gray[y-n:y])

and outside looks like scanner background
```

The test should be something like:

``` py
abs(inside_mean - outside_mean) > contrast_threshold
```

but **without requiring inside to be brighter**.

So remove:

``` py
if inside <= outside:
    continue
```

because that assumes white pages.

You want:

``` py
if abs(float(inside)-float(outside)) < 30:
    continue
```

This handles:

    gray -> white
    gray -> black
    gray -> orange
    gray -> blue

------------------------------------------------------------------------

## 2. The outside gray constraint is good

Keep this:

``` py
if not (60 < outside < 200):
    continue
```

Actually I would make it adaptive.

Instead of:

``` py
60 < outside < 200
```

calculate the scanner background from the corners:

``` py
bg_samples = np.concatenate([
    gray[:50,:].ravel(),
    gray[-50:,:].ravel(),
    gray[:,:50].ravel(),
    gray[:,-50:].ravel(),
])

bg_mean = np.median(bg_samples)
bg_std = np.std(bg_samples)
```

Then:

``` py
if abs(outside - bg_mean) > max(30, bg_std*3):
    continue
```

because your scanner background is probably not always exactly 128.

------------------------------------------------------------------------

## 3. Your margin idea is excellent

This is probably the biggest improvement.

You already know:

``` py
rotated_margined_scan_x
rotated_scan_x
```

so you know approximately:

    scanner canvas:

    +----------------------+
    | margin               |
    |   +--------------+   |
    |   |              |   |
    |   |    page      |   |
    |   |              |   |
    |   +--------------+   |
    |                      |
    +----------------------+

Therefore do not search the whole image.

Your current:

``` py
margin_range_x_px *= SEARCH_MARGIN_FACTOR
```

is good.

I would rename:

    margin_range_x_px

to:

    edge_search_width_px

because it is not the margin anymore.

------------------------------------------------------------------------

## 4. Rank candidates instead of taking the first

This is the other major improvement.

Instead of:

``` py
pts.append((x, y))
```

return candidates:

``` py
[
 {
   "point": (x,y),
   "contrast": 55,
   "background_error": 3,
 },
 ...
]
```

Then score them:

    score =
        contrast
        - background_error * 2

Keep the highest scoring line.

------------------------------------------------------------------------

## 5. Biggest rectangle ranking

I would not do this from contours anymore.

Contours are unreliable because:

- orange boxes
- photos
- text blocks
- illustrations

all create rectangles.

Instead use your three fitted lines:

    top_line
    bottom_line
    outside_line

and estimate the fourth edge.

Candidate page rectangle:

    area = width * height

Then reject impossible sizes:

``` py
expected_area = (
    config.rotated_scan_x *
    config.rotated_scan_y
)

if area < expected_area * 0.7:
    reject

if area > expected_area * 1.3:
    reject
```

This is much stronger than contour area.

Your config already gives you a prior.

------------------------------------------------------------------------

## 6. Refactor detect\_\*\_points

Yes. Do this.

Your enum idea is good:

``` py
EDGE_TOP = 0
EDGE_BOTTOM = 1
EDGE_LEFT = 2
EDGE_RIGHT = 3
```

Keep that.

I would change the function signature:

``` py
def detect_edge_points(
    gray,
    edge,
    search_width,
):
```

Remove `mask`.

The mask is the wrong abstraction now.

Something like:

``` py
def detect_edge_points(gray, edge, search_width):

    if edge == EDGE_TOP:
        ...
    elif edge == EDGE_BOTTOM:
        ...
```

Inside use a helper:

``` py
def transition_score(outside, inside):
    contrast = abs(float(inside)-float(outside))

    if contrast < 30:
        return 0

    return contrast
```

------------------------------------------------------------------------

## 7. The resulting pipeline becomes cleaner

I would change the flow to:

    load image

    calculate expected edge search zones
            |
            v

    detect_edge_points()
            |
            v

    verify candidates
            |
            v

    reject outliers
            |
            v

    RANSAC lines
            |
            v

    calculate rectangle
            |
            v

    compare rectangle size with config
            |
            v

    warp

No threshold mask needed for geometry.

Keep thresholding only for debug visualization.

------------------------------------------------------------------------

One more thing: your current code has a hidden bug from the refactor:

``` py
top_pts = detect_edge_points(...)
bottom_pts = detect_edge_points(...)
outside_pts = detect_edge_points(...)
```

then:

``` py
top_line = fit_line_ransac(top_pts)
```

will crash if verification leaves too few points.

I would make `fit_line_ransac()` return:

``` py
None
```

if:

``` py
len(pts) < RANSAC_MIN_INLIERS
```

and then let the rectangle ranking choose between candidates.

The next logical step would be to replace `verify_horizontal()` and `verify_vertical()` with one generic:

``` py
verify_transition(gray, points, direction)
```

because they are now doing almost the same thing. That will make the new detector much easier to maintain.

## Prompt:

7/13/2026, 4:12:41 PM

> Remove `mask`.

but then i can no longer call

``` py
    pts = verify_transition(
        gray,
        # pts,
        np.asarray(pts, np.float32),
        edge,
    )
```

from inside of detect_edge_points

or should i move the call to verify_transition back, outside of detect_edge_points?

relevant parts of my script:

maybe i should remove most of the process_image function...?

at least some parts after  
`# FIXME merge old code with new code`

``` py
EDGE_TOP = 0
EDGE_BOTTOM = 1
EDGE_LEFT = 2
EDGE_RIGHT = 3

def detect_edge_points(gray, edge, search_size):
    h, w = mask.shape
    pts = []

    if edge == EDGE_TOP:
        y_max = min(search_size, h)
        for x in range(w):
            # TODO why 255
            ys = np.where(mask[:y_max, x] == 255)[0]
            if len(ys):
                pts.append((x, ys[0]))

    elif edge == EDGE_BOTTOM:
        y_min = max(0, h - search_size)
        for x in range(w):
            ys = np.where(mask[y_min:, x] == 255)[0]
            if len(ys):
                pts.append((x, y_min + ys[-1]))

    elif edge == EDGE_LEFT:
        x_max = min(search_size, w)
        for y in range(h):
            xs = np.where(mask[y, :x_max] == 255)[0]
            if len(xs):
                pts.append((xs[0], y))

    # elif edge == EDGE_RIGHT:
    else:
        x_min = max(0, w - search_size)
        for y in range(h):
            xs = np.where(mask[y, x_min:] == 255)[0]
            if len(xs):
                pts.append((x_min + xs[-1], y))

    # Transition verification
    #
    # The first white pixel isn't always the page.
    # It might be dust, glare, or text sticking out.
    #
    # Keep only points that actually separate gray background
    # from the white page.
    pts = verify_transition(
        gray,
        # pts,
        np.asarray(pts, np.float32),
        edge,
    )

    return np.asarray(pts, np.float32)

def verify_transition(gray, points, edge):
    # TODO actually refactor verify_horizontal and verify_vertical
    if edge == EDGE_TOP:
        return verify_horizontal(gray, points, top_edge=True)

    elif edge == EDGE_BOTTOM:
        return verify_horizontal(gray, points, top_edge=False)

    elif edge == EDGE_LEFT:
        return verify_vertical(gray, points, right_edge=False)

    # elif edge == EDGE_RIGHT:
    else:
        return verify_vertical(gray, points, right_edge=True)

def verify_horizontal(gray, points, top_edge):
    good = []

    H, W = gray.shape

    for x, y in points.astype(int):

        if y < 3 or y >= H-3:
            continue

        if top_edge:
            outside = np.mean(gray[y-3:y, x])
            inside  = np.mean(gray[y:y+3, x])
        else:
            outside = np.mean(gray[y:y+3, x])
            inside  = np.mean(gray[y-3:y, x])

        r'''
        # must have strong contrast
        if abs(float(inside) - float(outside)) < 40:
            continue
        '''
        r'''
        # must have strong contrast
        if top_edge:
            if not is_horizontal_transition(gray, x, y):
                continue
        else:
            if not is_vertical_transition(gray, x, y):
                continue
        '''
        # background is always gray
        # Require the outside to actually look like scanner background
        BACKGROUND_MIN = 80
        BACKGROUND_MAX = 180
        if not (BACKGROUND_MIN <= outside <= BACKGROUND_MAX):
            continue

        # outside should be scanner gray
        if not (60 < outside < 200):
            continue

        # inside should be brighter
        if inside <= outside:
            continue

        good.append((x, y))

    return np.asarray(good, np.float32)

def verify_vertical(gray, points, right_edge):
    good = []

    H, W = gray.shape

    for x, y in points.astype(int):

        if x < 3 or x >= W-3:
            continue

        if right_edge:
            outside = np.mean(gray[y, x:x+3])
            inside  = np.mean(gray[y, x-3:x])
        else:
            outside = np.mean(gray[y, x-3:x])
            inside  = np.mean(gray[y, x:x+3])

        if abs(float(inside) - float(outside)) < 40:
            continue

        if not (60 < outside < 200):
            continue

        if inside <= outside:
            continue

        good.append((x, y))

    return np.asarray(good, np.float32)

# TODO refactor x/y

def reject_outliers_horizontal(pts, tolerance=40):
    if len(pts) == 0:
        return pts

    ys = pts[:,1]
    median = np.median(ys)

    return pts[np.abs(ys - median) < tolerance]

def reject_outliers_vertical(pts, tolerance=40):
    if len(pts) == 0:
        return pts

    xs = pts[:,0]
    median = np.median(xs)

    return pts[np.abs(xs - median) < tolerance]

def process_image(in_path, out_path):

                rotated_scan_y_px = px_of_mm(config.rotated_scan_y, config.scan_resolution)
                    target_h = rotated_scan_y_px
                    scale_y = target_h / page_height
                elif 1:
                    # use rotated_margined_scan_y and H_img
                    rotated_margined_scan_y_px = px_of_mm(config.rotated_margined_scan_y, config.scan_resolution)
                    target_h = rotated_margined_scan_y_px
                    scale_y = target_h / H_img
                else:
                    # no, this fails because the scanner removes one edge
                    # so actual_aspect is always wrong...
                    #
                    # use rotated_margined_scan_y and H_img
                    # expected: what we ordered from the scanner
                    expected_aspect = (
                        config.rotated_margined_scan_x /
                        config.rotated_margined_scan_y
                    )
                    expected_height = config.rotated_margined_scan_y
                    # actual: what the scanner gave us
                    actual_aspect = W_img / H_img
                    actual_height = H_img
                    # we assume the scanner always returns correct X coordinates
                    # and all errors are only in Y coordinates
                    # expected_aspect / actual_aspect = actual_height / expected_height
                    actual_height_2 = expected_aspect / actual_aspect / expected_height
                    expected_aspect_factor = expected_aspect / actual_aspect
                    # the scale of actual_height relative to actual_height_2
                    actual_height_scale = actual_height / actual_height_2
                    if DEBUG:
                        print(f"expected_aspect={expected_aspect} actual_aspect={actual_aspect} expected_aspect_factor={expected_aspect_factor}")
                        print(f"expected_height={expected_height} actual_height={actual_height} actual_height_2={actual_height_2} actual_height_scale={actual_height_scale}")
                    target_h = actual_height_2
                    scale_y = target_h / H_img

                # debug: manually set the scale_y factor
                # scale_y = 1 / 1.03 # shrink the page height by 3%

                scale_y_tolerance = 0.001 # 0.1%

                if scale_y < (1 - scale_y_tolerance) or (1 + scale_y_tolerance) < scale_y:
                    if DEBUG:
                        print(f"line 680: scale_y={scale_y} target_h={target_h} page_height={page_height} -> scaling height")
                    rotated = cv2.resize(
                        rotated,
                        None,
                        fx=1.0,
                        fy=scale_y,
                        interpolation=cv2.INTER_CUBIC
                    )
                    page_height = page_height * scale_y

                    # re-detect lines in the scaled image

                    # gray, mask, contours = get_gray_mask_contours(img, dbgdir)
                    gray, mask, contours = get_gray_mask_contours(rotated, dbgdir)

                    if 1:
                        top_angle2 = horizontal_line_angle(top_line)
                        bottom_angle2 = horizontal_line_angle(bottom_line)
                        outside_angle2 = vertical_line_angle(outside_line)
                        if DEBUG:
                            print(
                                f"line 620: after rotation and re-fitting: "
                                f"top_angle2={top_angle2:.3f} "
                                f"bottom_angle2={bottom_angle2:.3f} "
                                f"outside_angle2={outside_angle2:.3f}"
                            )

                    if not contours:
                        print(f"line 630: Warning: no contours found in {in_path}")
                        return

                    page_contour = max(contours, key=cv2.contourArea)

                    top_pts, bottom_pts, outside_pts = split_edge_candidates(
                        page_contour,
                        bad_on_left
                    )

                    top_line = fit_line_ransac(top_pts)[:4]
                    bottom_line = fit_line_ransac(bottom_pts)[:4]
                    outside_line = fit_line_ransac(outside_pts)[:4]

                    # fixed
                    # # FIXME H_img is wrong
                    # expected_h = H_img
                    # expected_w = int(round(ASPECT * expected_h))

                    vx, vy, x0, y0 = outside_line

                    outside_top = intersect_lines(
                        outside_line,
                        top_line
                    )

                    outside_bottom = intersect_lines(
                        outside_line,
                        bottom_line
                    )

                else:
                    if DEBUG:
                        print(f"line 680: scale_y={scale_y} target_h={target_h} page_height={page_height} -> not scaling height")

            else:
                # config.do_rotate == True
                # fix scan width
                # Raw scans are rotated by 90/270 degrees.
                # Therefore: scanner Y errors -> image X errors
                # We must correct width (scale_x), not height (scale_y).
                if 1:
                    # The detected outside edge gives us the real page height
                    # in image coordinates.  After rotation this corresponds to
                    # scanner X, which is assumed correct.
                    #
                    # The missing/damaged dimension is the page width.

                    rotated_margined_scan_x_px = px_of_mm(
                        config.rotated_margined_scan_x,
                        config.scan_resolution
                    )

                    target_w = rotated_margined_scan_x_px

                    actual_w = math.dist(
                        outside_top,
                        outside_bottom
                    )

                    # This is actually the width in scanner coordinates because
                    # the document is rotated.
                    scale_x = target_w / actual_w

                    scale_x_tolerance = 0.001  # 0.1%

                    if (
                        scale_x < (1 - scale_x_tolerance)
                        or
                        scale_x > (1 + scale_x_tolerance)
                    ):
                        if DEBUG:
                            print(
                                "line 680:",
                                f"scale_x={scale_x}",
                                f"target_w={target_w}",
                                f"actual_w={actual_w}",
                                "-> scaling width"
                            )

                        rotated = cv2.resize(
                            rotated,
                            None,
                            fx=scale_x,
                            fy=1.0,
                            interpolation=cv2.INTER_CUBIC
                        )

                        # Update image size
                        Hr, Wr = rotated.shape[:2]

                        if DEBUG:
                            print(
                                "line 690: after scale_x",
                                f"Wr={Wr}",
                                f"Hr={Hr}"
                            )

                        # Re-detect page after scaling
                        gray, mask, contours = get_gray_mask_contours(
                            rotated,
                            dbgdir
                        )

                        if not contours:
                            print(
                                f"line 700: Warning: no contours found after scaling {in_path}"
                            )
                            return

                        page_contour = max(
                            contours,
                            key=cv2.contourArea
                        )

                        top_pts, bottom_pts, outside_pts = split_edge_candidates(
                            page_contour,
                            bad_on_left
                        )

                        top_line = fit_line_ransac(top_pts)[:4]
                        bottom_line = fit_line_ransac(bottom_pts)[:4]
                        outside_line = fit_line_ransac(outside_pts)[:4]

                        outside_top = intersect_lines(
                            outside_line,
                            top_line
                        )

                        outside_bottom = intersect_lines(
                            outside_line,
                            bottom_line
                        )

                    else:
                        if DEBUG:
                            print(
                                "line 680:",
                                f"scale_x={scale_x}",
                                "-> not scaling width"
                            )

expected_h = int(round(page_height))

        # ASPECT = x / y
        # x = y * ASPECT

        # expand the binding edge to expected_w
        # expected_w = int(round(expected_h * ASPECT))
        expected_w = int(round(page_height * ASPECT))

        if 0:
            # crop as a rectangle
            x_out = (
                outside_top[0]
                +
                outside_bottom[0]
            ) / 2

            x_out = int(round(x_out))

            # if bad_on_left:
            #     x0_page = x_out
            #     x1_page = x_out + expected_w
            # else:
            #     x0_page = x_out - expected_w
            #     x1_page = x_out

            if bad_on_left:
                # outside edge is RIGHT
                x1_page = x_out
                x0_page = x_out - expected_w
            else:
                # outside edge is LEFT
                x0_page = x_out
                x1_page = x_out + expected_w

            # clamp
            # x0_page = max(0, x0_page)
            # x1_page = min(rotated.shape[1], x1_page)

            y_top = int(round(outside_top[1]))
            y_bottom = int(round(outside_bottom[1]))

            src_x0 = max(0, x0_page)
            src_x1 = min(rotated.shape[1], x1_page)

            src_y0 = max(0, y_top)
            src_y1 = min(rotated.shape[0], y_top + expected_h)

            dst_x0 = src_x0 - x0_page
            dst_y0 = src_y0 - y_top

            if DEBUG:
                print(
                    "line 640:",
                    f"outside_top={outside_top}",
                    f"outside_bottom={outside_bottom}",
                )

            # crop = rotated[
            #     y_top:y_top+expected_h,
            #     x0_page:x1_page
            # ]

            full_crop = np.ones(
                (expected_h, expected_w, 3),
                dtype=rotated.dtype
            ) * 255

            full_crop[
                dst_y0:dst_y0 + (src_y1-src_y0),
                dst_x0:dst_x0 + (src_x1-src_x0)
            ] = rotated[
                src_y0:src_y1,
                src_x0:src_x1
            ]

            # FIXME fill empty area near binding edge
            # currently it is all white
            # but it should copy the vertical pattern near the binding edge

            crop = full_crop

            if DEBUG:
                print(
                    "line 650: crop",
                    f"W_img={W_img}",
                    f"H_img={H_img}",
                    f"x0_page={x0_page}",
                    f"x1_page={x1_page}",
                    f"y_top={y_top}",
                    f"y_bottom={y_bottom}",
                    f"src_x0={src_x0}",
                    f"src_x1={src_x1}",
                    f"src_y0={src_y0}",
                    f"src_y1={src_y1}",
                    f"dst_x0={dst_x0}",
                    f"dst_y0={dst_y0}",
                    f"expected_w={expected_w}",
                    f"expected_h={expected_h}",
                )

            if DEBUG:
                vis = rotated.copy()
                cv2.rectangle(
                    vis,
                    (int(x0_page), int(y_top)),
                    (int(x1_page), int(y_bottom)),
                    (0,0,255),
                    3,
                )
                name = f"{page_num:03d}.crop_debug_rotated_line730.jpg"
                print(f"writing {name}")
                cv2.imwrite(name, vis)

        else:
            # crop as a quadrilateral
            # not better than "crop as a rectangle"?
            if 0:
                # expand the binding edge to expected_w
                if bad_on_left:
                    # outside edge is the RIGHT edge
                    x1_top = outside_top[0]
                    x1_bottom = outside_bottom[0]
                    x0_top = x1_top - expected_w
                    x0_bottom = x1_bottom - expected_w
                else:
                    # outside edge is the LEFT edge
                    x0_top = outside_top[0]
                    x0_bottom = outside_bottom[0]
                    x1_top = x0_top + expected_w
                    x1_bottom = x0_bottom + expected_w
            else:
                # dont expand the binding edge to expected_w
                # use only the detected page edges
                # if bad_on_left:
                #     # outside edge is RIGHT edge
                #     x1_top = outside_top[0]
                #     x1_bottom = outside_bottom[0]
                #     # use the actual detected left edge
                #     x0_top = np.min(page_contour[:,0,0])
                #     x0_bottom = x0_top
                # else:
                #     # outside edge is LEFT edge
                #     x0_top = outside_top[0]
                #     x0_bottom = outside_bottom[0]
                #     # use the actual detected right edge
                #     x1_top = np.max(page_contour[:,0,0])
                #     x1_bottom = x1_top
                # problem: page_contour after thresholding may include the background
                # or may not have a reliable missing-edge position.
                # A cleaner temporary solution is to use the detected quadrilateral width
                # from the two horizontal edge intersections
                if bad_on_left:
                    # outside edge is right
                    x1_top = outside_top[0]
                    x1_bottom = outside_bottom[0]
                    # find leftmost detected page boundary
                    x0_top = np.min(top_pts[:,0])
                    x0_bottom = np.min(bottom_pts[:,0])
                else:
                    # outside edge is left
                    x0_top = outside_top[0]
                    x0_bottom = outside_bottom[0]
                    # find rightmost detected page boundary
                    x1_top = np.max(top_pts[:,0])
                    x1_bottom = np.max(bottom_pts[:,0])

                # dont expand the binding edge to expected_w
                expected_w = int(round(
                    math.dist((x0_top, outside_top[1]), (x1_top, outside_top[1]))
                ))

src = np.float32([
                [x0_top, outside_top[1]],
                [x1_top, outside_top[1]],
                [x1_bottom, outside_bottom[1]],
                [x0_bottom, outside_bottom[1]],
            ])

            dst = np.float32([
                [0,0],
                [expected_w-1,0],
                [expected_w-1,expected_h-1],
                [0,expected_h-1],
            ])

            M = cv2.getPerspectiveTransform(src, dst)

            crop = cv2.warpPerspective(
                rotated,
                M,
                (expected_w, expected_h),
                borderValue=(255,255,255)
            )

            if DEBUG:
                print(
                    "line 650: crop",
                    f"W_img={W_img}",
                    f"H_img={H_img}",
                    f"outside_top={outside_top}",
                    f"outside_bottom={outside_bottom}",
                    f"x1_top={x1_top}",
                    f"x1_bottom={x1_bottom}",
                    f"x0_top={x0_top}",
                    f"x0_bottom={x0_bottom}",
                    f"expected_w={expected_w}",
                    f"expected_h={expected_h}",
                )

            if DEBUG:
                vis = rotated.copy()
                # margin range: green
                if bad_on_left:
                    # outside edge is right
                    # no line on the left
                    pts = np.array([
                        [0, edge_search_height_px], # top left
                        [W_img - edge_search_width_px, edge_search_height_px], # top right
                        [W_img - edge_search_width_px, H_img - edge_search_height_px], # bottom right
                        [0, H_img - edge_search_height_px], # bottom left
                    ], np.int32)
                else:
                    # outside edge is left
                    # no line on the right
                    pts = np.array([
                        [W_img, edge_search_height_px], # top right
                        [edge_search_width_px, edge_search_height_px], # top left
                        [edge_search_width_px, H_img - edge_search_height_px], # bottom left
                        [W_img, H_img - edge_search_height_px], # bottom right
                    ], np.int32)
                cv2.polylines(vis, [pts], False, (0,255,0), 3) # (0,255,0) == green?
                # page margin: red
                pts = np.array([
                    [x0_top, outside_top[1]],
                    [x1_top, outside_top[1]],
                    [x1_bottom, outside_bottom[1]],
                    [x0_bottom, outside_bottom[1]],
                ], np.int32)
                cv2.polylines(vis, [pts], True, (0,0,255), 3)
                name = f"{page_num:03d}.crop_debug_rotated_line730.jpg"
                print(f"writing {name}")
                cv2.imwrite(name, vis)

        # crop = rotated[
        #     0:expected_h,
        #     int(x0_page):int(x1_page)
        # ]

        # y_top = round(outside_top[1])
        # y_bottom = round(outside_bottom[1])
        # actual_height = y_bottom - y_top
        # crop = rotated[
        #     y_top:y_top + expected_h,
        #     x0_page:x1_page
        # ]

        # y_top = int(round(outside_top[1]))
        # y_bottom = int(round(outside_bottom[1]))
        # crop = rotated[
        #     y_top:y_bottom,
        #     x0_page:x1_page
        # ]

        if 0:
            # img = repair_binding(img, bad_on_left, width=50)
            if 1:
                # rotated = repair_binding(rotated, bad_on_left, width=50)
                # Hr, Wr = rotated.shape[:2]
                # print("line 660: rotated size", Wr, Hr)
                # warped = rotated
                crop = repair_binding(crop, bad_on_left, width=50)
                Hr, Wr = crop.shape[:2]
                if DEBUG:
                    print("line 660: crop size", Wr, Hr)
            else:
                crop = repair_binding(crop, bad_on_left, width=50)

            if DEBUG:
                print(
                    "line 760: after repair_binding: crop actual:",
                    f"crop.shape[1]={crop.shape[1]}",
                    f"expected_w={expected_w}",
                )

        warped = crop

    else:
        # config.use_three_edge_deskew == False

        raise NotImplementedError("sorry, your book is too tall for your scanner...")

        # FIXME use only two page edges: outside, bottom

        # Approximate contour to quadrilateral
        epsilon = 0.02 * cv2.arcLength(page_contour, True)
        approx = cv2.approxPolyDP(page_contour, epsilon, True)
        if len(approx) != 4:
            approx = cv2.convexHull(page_contour)
            if len(approx) < 4:
                print(f"Warning: not enough points for perspective in {in_path}")
                return
            # pick 4 extreme points
            pts = np.array([
                approx[approx[:,0,0].argmin()][0],  # leftmost
                approx[approx[:,0,1].argmin()][0],  # topmost
                approx[approx[:,0,0].argmax()][0],  # rightmost
                approx[approx[:,0,1].argmax()][0]   # bottommost
            ])
        else:
            pts = approx.reshape(4,2)

        rect = order_points(pts)

        # Perspective transform
        widthA = np.linalg.norm(rect[2] - rect[3])
        widthB = np.linalg.norm(rect[1] - rect[0])
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.linalg.norm(rect[1] - rect[2])
        heightB = np.linalg.norm(rect[0] - rect[3])
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    # # Add internal white border
    # h, w = warped.shape[:2]
    # canvas = np.ones_like(warped) * 255
    # b = BORDER_SIZE
    # canvas[b:h-b, b:w-b] = warped[b:h-b, b:w-b]

    h, w = warped.shape[:2]
    b = BORDER_SIZE

    # Prepare a canvas with the same content as warped
    canvas = warped.copy()

    # Function to compute average color along a strip
    def avg_color_strip(img, axis, start, end, strip_width=1):
        if axis == 'top':
            strip = img[start:end, :, :]
        elif axis == 'bottom':
            strip = img[h-end:h-start, :, :]
        elif axis == 'left':
            strip = img[:, start:end, :]
        elif axis == 'right':
            strip = img[:, w-end:w-start, :]
        else:
            raise ValueError("Invalid axis")
        return np.mean(strip, axis=(0,1)).astype(np.uint8)

    # FIXME preserve patterns near edges
    if 0:
        # Fill borders with local average color
        canvas[0:b, :, :] = avg_color_strip(canvas, 'top', 50, 100)       # top border
        canvas[h-b:h, :, :] = avg_color_strip(canvas, 'bottom', 50, 100)  # bottom border
        canvas[:, 0:b, :] = avg_color_strip(canvas, 'left', 50, 100)      # left border
        canvas[:, w-b:w, :] = avg_color_strip(canvas, 'right', 50, 100)   # right border

    if input_is_grayscale:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

    ensure_dir(os.path.dirname(out_path))
    cv2.imwrite(out_path, canvas)
    print(f"writing {out_path}")
```

## Response:

7/13/2026, 4:12:41 PM

Yes: the current direction is good, but I would **not move `verify_transition()` outside `detect_edge_points()`**. The abstraction you are moving toward is actually cleaner if `detect_edge_points()` owns the whole pipeline:

    search region
        ↓
    candidate transition pixels
        ↓
    verify transition
        ↓
    return usable edge points

The problem is not `verify_transition()` being inside. The problem is that `detect_edge_points()` still depends on the global `mask`.

This line:

``` py
h, w = mask.shape
```

is the real smell. Pass `mask` as an argument.

So:

``` py
def detect_edge_points(gray, mask, edge, search_size):
```

not:

``` py
def detect_edge_points(gray, edge, search_size):
```

Then this is perfectly fine:

``` py
pts = verify_transition(
    gray,
    np.asarray(pts, np.float32),
    edge,
)
```

because `detect_edge_points()` has produced candidate points and now validates them.

------------------------------------------------------------------------

I would make the refactor in three steps.

## 1. Keep `verify_transition()` as the public interface

Your direction enum is good:

``` py
EDGE_TOP = 0
EDGE_BOTTOM = 1
EDGE_LEFT = 2
EDGE_RIGHT = 3
```

Keep it.

Then:

``` py
def verify_transition(gray, points, edge):
    if len(points) == 0:
        return np.empty((0,2), dtype=np.float32)

    if edge == EDGE_TOP:
        return verify_horizontal(gray, points, top_edge=True)

    if edge == EDGE_BOTTOM:
        return verify_horizontal(gray, points, top_edge=False)

    if edge == EDGE_LEFT:
        return verify_vertical(gray, points, right_edge=False)

    if edge == EDGE_RIGHT:
        return verify_vertical(gray, points, right_edge=True)

    raise ValueError(edge)
```

------------------------------------------------------------------------

## 2. Refactor `detect_edge_points()` only

Something like:

``` py
def detect_edge_points(gray, mask, edge, search_size):

    h, w = mask.shape
    pts = []

    if edge == EDGE_TOP:

        y_max = min(search_size, h)

        for x in range(w):
            ys = np.where(mask[:y_max, x] == 255)[0]

            if len(ys):
                pts.append((x, ys[0]))

elif edge == EDGE_BOTTOM:

        y_min = max(0, h - search_size)

        for x in range(w):
            ys = np.where(mask[y_min:, x] == 255)[0]

            if len(ys):
                pts.append((x, y_min + ys[-1]))

elif edge == EDGE_LEFT:

        x_max = min(search_size, w)

        for y in range(h):
            xs = np.where(mask[y, :x_max] == 255)[0]

            if len(xs):
                pts.append((xs[0], y))

elif edge == EDGE_RIGHT:

        x_min = max(0, w - search_size)

        for y in range(h):
            xs = np.where(mask[y, x_min:] == 255)[0]

            if len(xs):
                pts.append((x_min + xs[-1], y))

pts = np.asarray(pts, np.float32)

    return verify_transition(
        gray,
        pts,
        edge,
    )
```

This is now self-contained.

------------------------------------------------------------------------

## 3. Remove the old `process_image()` duplicate code

Yes.

The biggest problem in your file is not the edge detector. It is that `process_image()` contains copies of:

- `fit_line_ransac`
- `intersect_lines`
- `detect_vertical_streaks`
- `keep_largest_component`
- `percentile_threshold`

Those inner versions shadow the global versions.

That makes debugging painful because you think you are editing one function but execution uses another.

Delete the inner copies.

`process_image()` should eventually become only orchestration:

``` py
def process_image(in_path, out_path):

    load image

    calculate search margins

    build mask

    top_pts = detect_edge_points(
        gray,
        mask,
        EDGE_TOP,
        edge_search_height_px
    )

    bottom_pts = detect_edge_points(
        gray,
        mask,
        EDGE_BOTTOM,
        edge_search_height_px
    )

    outside_pts = detect_edge_points(
        gray,
        mask,
        EDGE_LEFT if not bad_on_left else EDGE_RIGHT,
        edge_search_width_px
    )

    reject outliers

    fit lines

    rotate

    rescale

    perspective transform

    save
```

That is probably 50% shorter.

------------------------------------------------------------------------

## One more thing: your `verify_horizontal()` is currently biased

This part:

``` py
if inside <= outside:
    continue
```

assumes:

    gray background -> brighter page

which fails for black pages.

Since you already mentioned:

> pages can also be black, or mixed

this should become:

``` py
if abs(float(inside) - float(outside)) < 40:
    continue
```

and remove the brightness direction requirement.

The scanner background constraint is the useful part:

``` py
if not (60 < outside < 200):
    continue
```

That gives you:

    gray background -> anything sufficiently different

which handles:

    gray -> white page
    gray -> black page
    gray -> colored page
    gray -> photo

------------------------------------------------------------------------

So the immediate cleanup order I would do:

1.  ✅ Pass `mask` into `detect_edge_points()`
2.  ✅ Keep `verify_transition()` inside `detect_edge_points()`
3.  ✅ Remove duplicated functions inside `process_image()`
4.  ✅ Remove brightness assumption from transition verification
5.  Later: replace `page_mask` selection with "largest rectangle scoring"

The "largest rectangle" idea will fit naturally after this because `detect_edge_points()` gives you candidate edges, and then you can score combinations of `(top,bottom,left/right)` lines instead of trusting the first threshold result.

## Prompt:

7/13/2026, 7:11:16 PM

my script still fails in the case where a solid orange rectangle is mis-detected as the page edge... the script seems to ignore the search margins (edge_search_width_px, edge_search_height_px)

i have added the search margins as green lines to the debug image `f"{page_num:03d}.crop_debug_rotated_line730.jpg"` (`# margin range: green`) and the red lines (detected page edges) are obviously inside of the green lines, but this should not be possible...

here is my current process_image function

``` py
def process_image(in_path, out_path):
    """
    Robust page extraction that handles missing top (or bottom) edges.
    - preserves original colors (warps the original image)
    - builds orthogonal axes (no shear)
    - if top is missing, uses bottom + good vertical to infer top by moving up expected_h
    - fills outside the filled page polygon with pure white
    - extensive debug output in OUTPUT_DIR/debug/<page_num>/
    """
    # ---------- start ----------

    # 1. load image

    fname = os.path.basename(in_path)
    m = re.match(r"^(\d+)", fname)
    page_num = int(m.group(1)) if m else 0
    bad_on_left = (page_num % 2 == 1)

    img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print("Failed to read", in_path); return
    input_is_grayscale = (len(img.shape) == 2)
    if len(img.shape) == 2:
        # OpenCV expects 3-channel images for many operations
        # later convert back to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    H_img, W_img = img.shape[:2]

    # fixed
    # # FIXME wrong H_img?
    # expected_w = int(round(ASPECT * H_img))
    # expected_h = H_img

    dbgdir = os.path.join(OUTPUT_DIR, "debug", f"{page_num:03d}")
    if DEBUG: ensure_dir(dbgdir)

# 2. calculate search margins

    # Compute the expected ranges of margins
    # NOTE the scanner removes the "scan top" margin
    # so in the X direction, we have only one margin
    # so here we divide by 4, not by 2
    edge_search_width_mm = (
        config.rotated_margined_scan_x -
        config.rotated_scan_x
    ) / 4.0
    edge_search_height_mm = (
        config.rotated_margined_scan_y -
        config.rotated_scan_y
    ) / 2.0

    if DEBUG:
        print(f"config.rotated_margined_scan: ({config.rotated_margined_scan_x}, {config.rotated_margined_scan_y}) mm")
        print(f"config.rotated_scan: ({config.rotated_scan_x}, {config.rotated_scan_y}) mm")
        print(f"margin_range: ({edge_search_width_mm}, {edge_search_height_mm}) mm")

    # Convert them to pixels
    edge_search_width_px = px_of_mm(
        edge_search_width_mm,
        config.scan_resolution
    )
    edge_search_height_px = px_of_mm(
        edge_search_height_mm,
        config.scan_resolution
    )

    # TODO rename, move to config

    SEARCH_MARGIN_FACTOR = 2.0
    SEARCH_MARGIN_ADD_MM = 5

    SEARCH_MARGIN_FACTOR = 1.2
    SEARCH_MARGIN_ADD_MM = 2

    # debug: dont increase
    # SEARCH_MARGIN_FACTOR = 1; SEARCH_MARGIN_ADD_MM = 0

    # Then enlarge them
    edge_search_width_px *= SEARCH_MARGIN_FACTOR
    edge_search_height_px *= SEARCH_MARGIN_FACTOR
    search_margin_add_px = px_of_mm(
        SEARCH_MARGIN_ADD_MM,
        config.scan_resolution
    )
    edge_search_width_px += search_margin_add_px
    edge_search_height_px += search_margin_add_px

    # we need integers for array indices
    edge_search_width_px = int(math.ceil(edge_search_width_px))
    edge_search_height_px = int(math.ceil(edge_search_height_px))

# 3. build mask

    # Step 1: Segment page vs. gray background
    # -> gray, mask

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Otsu is usually sufficient here
    _, page_mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Page should be white
    if np.mean(gray[page_mask == 255]) < np.mean(gray[page_mask == 0]):
        page_mask = cv2.bitwise_not(page_mask)

    page_mask = cv2.morphologyEx(
        page_mask,
        cv2.MORPH_CLOSE,
        np.ones((5,5), np.uint8)
    )

# TODO rename
    mask = page_mask

# Step 2: Scan for transitions
    # Step 3: Transition verification

    if bad_on_left:
        outside_pts = detect_edge_points(
            gray,
            mask,
            EDGE_RIGHT,
            edge_search_width_px,
        )
    else:
        outside_pts = detect_edge_points(
            gray,
            mask,
            EDGE_LEFT,
            edge_search_width_px,
        )

    top_pts = detect_edge_points(
        gray,
        mask,
        EDGE_TOP,
        edge_search_height_px,
    )

    bottom_pts = detect_edge_points(
        gray,
        mask,
        EDGE_BOTTOM,
        edge_search_height_px,
    )

# 4. reject outliers

    # Step 4: Reject isolated outliers
    # Use a median filter before RANSAC.

    top_pts = reject_outliers_horizontal(top_pts)
    bottom_pts = reject_outliers_horizontal(bottom_pts)
    outside_pts = reject_outliers_vertical(outside_pts)

# 5. fit lines

    # Step 5: RANSAC

    top_line = fit_line_ransac(top_pts)[:4]
    bottom_line = fit_line_ransac(bottom_pts)[:4]
    outside_line = fit_line_ransac(outside_pts)[:4]

# FIXME merge old code with new code

# # TODO remove

    # gray, mask, contours = get_gray_mask_contours(img, dbgdir)

    # if not contours:
    #     print(f"Warning: no contours found in {in_path}")
    #     return

    # page_contour = max(contours, key=cv2.contourArea)

    if config.use_three_edge_deskew:

        # top_pts, bottom_pts, outside_pts = split_edge_candidates(
        #     page_contour,
        #     bad_on_left
        # )

        # top_line = fit_line_ransac(top_pts)[:4]
        # bottom_line = fit_line_ransac(bottom_pts)[:4]
        # outside_line = fit_line_ransac(outside_pts)[:4]

# 6. rotate

        # old
        # top_angle = math.degrees(line_angle(top_line))
        # bottom_angle = math.degrees(line_angle(bottom_line))
        # outside_angle = math.degrees(line_angle(outside_line))

        # new
        top_angle = horizontal_line_angle(top_line)
        bottom_angle = horizontal_line_angle(bottom_line)
        outside_angle = vertical_line_angle(outside_line)

        if DEBUG:
            # start debug prints
            print()
            print(f"line 570: page_num={page_num}")

        rotation_error = -1 * (
            -top_angle
            -bottom_angle
        ) / 2.0

        Mrot = cv2.getRotationMatrix2D(
            (W_img/2, H_img/2),
            rotation_error,
            1.0
        )

        if DEBUG:
            print(
                f"line 575: before rotation: "
                f"top_angle={top_angle:.3f} "
                f"bottom_angle={bottom_angle:.3f} "
                f"outside_angle={outside_angle:.3f}"
            )

        rotated = cv2.warpAffine(
            img,
            Mrot,
            (W_img, H_img),
            borderValue=(255,255,255)
        )

        Hr, Wr = rotated.shape[:2]

        if DEBUG:
            print("line 580: rotated size", Wr, Hr)

        # img = rotated # ?

# re-detect lines in the rotated image

        # gray, mask, contours = get_gray_mask_contours(img, dbgdir)
        gray, mask, contours = get_gray_mask_contours(rotated, dbgdir)

        if 1:
            top_angle2 = horizontal_line_angle(top_line)
            bottom_angle2 = horizontal_line_angle(bottom_line)
            outside_angle2 = vertical_line_angle(outside_line)

            if DEBUG:
                print(
                    f"line 620: after rotation and re-fitting: "
                    f"top_angle2={top_angle2:.3f} "
                    f"bottom_angle2={bottom_angle2:.3f} "
                    f"outside_angle2={outside_angle2:.3f}"
                )

        if not contours:
            print(f"line 630: Warning: no contours found in {in_path}")
            return

        page_contour = max(contours, key=cv2.contourArea)

# 7. rescale

        top_pts, bottom_pts, outside_pts = split_edge_candidates(
            page_contour,
            bad_on_left
        )

        top_line = fit_line_ransac(top_pts)[:4]
        bottom_line = fit_line_ransac(bottom_pts)[:4]
        outside_line = fit_line_ransac(outside_pts)[:4]

        # fixed
        # # FIXME H_img is wrong
        # expected_h = H_img
        # expected_w = int(round(ASPECT * expected_h))

        vx, vy, x0, y0 = outside_line

        outside_top = intersect_lines(
            outside_line,
            top_line
        )

        outside_bottom = intersect_lines(
            outside_line,
            bottom_line
        )

        if 0:
            # approximated page height
            page_height = math.dist(
                outside_top,
                outside_bottom
            )
        else:
            # perpendicular page height
            page_height = math.hypot(
                outside_bottom[0] - outside_top[0],
                outside_bottom[1] - outside_top[1]
            )

        # no. this fails to reconstruct the page height...
        # TODO try to solve this with the average height of multiple pages
        # assuming all pages must have the same height
        # also allowing the user to specify a scale_y factor
        if 0:
            if config.do_rotate == False:
                # fix scan height
                # ...
                pass

        expected_h = int(round(page_height))

        # ASPECT = x / y
        # x = y * ASPECT

        # expand the binding edge to expected_w
        # expected_w = int(round(expected_h * ASPECT))
        expected_w = int(round(page_height * ASPECT))

# 8. perspective transform

        if 0:
            # crop as a rectangle
            x_out = (
                outside_top[0]
                +
                outside_bottom[0]
            ) / 2

            x_out = int(round(x_out))

            # if bad_on_left:
            #     x0_page = x_out
            #     x1_page = x_out + expected_w
            # else:
            #     x0_page = x_out - expected_w
            #     x1_page = x_out

            if bad_on_left:
                # outside edge is RIGHT
                x1_page = x_out
                x0_page = x_out - expected_w
            else:
                # outside edge is LEFT
                x0_page = x_out
                x1_page = x_out + expected_w

            # clamp
            # x0_page = max(0, x0_page)
            # x1_page = min(rotated.shape[1], x1_page)

            y_top = int(round(outside_top[1]))
            y_bottom = int(round(outside_bottom[1]))

            src_x0 = max(0, x0_page)
            src_x1 = min(rotated.shape[1], x1_page)

            src_y0 = max(0, y_top)
            src_y1 = min(rotated.shape[0], y_top + expected_h)

            dst_x0 = src_x0 - x0_page
            dst_y0 = src_y0 - y_top

            if DEBUG:
                print(
                    "line 640:",
                    f"outside_top={outside_top}",
                    f"outside_bottom={outside_bottom}",
                )

            # crop = rotated[
            #     y_top:y_top+expected_h,
            #     x0_page:x1_page
            # ]

            full_crop = np.ones(
                (expected_h, expected_w, 3),
                dtype=rotated.dtype
            ) * 255

            full_crop[
                dst_y0:dst_y0 + (src_y1-src_y0),
                dst_x0:dst_x0 + (src_x1-src_x0)
            ] = rotated[
                src_y0:src_y1,
                src_x0:src_x1
            ]

            # FIXME fill empty area near binding edge
            # currently it is all white
            # but it should copy the vertical pattern near the binding edge

            crop = full_crop

            if DEBUG:
                print(
                    "line 650: crop",
                    f"W_img={W_img}",
                    f"H_img={H_img}",
                    f"x0_page={x0_page}",
                    f"x1_page={x1_page}",
                    f"y_top={y_top}",
                    f"y_bottom={y_bottom}",
                    f"src_x0={src_x0}",
                    f"src_x1={src_x1}",
                    f"src_y0={src_y0}",
                    f"src_y1={src_y1}",
                    f"dst_x0={dst_x0}",
                    f"dst_y0={dst_y0}",
                    f"expected_w={expected_w}",
                    f"expected_h={expected_h}",
                )

            if DEBUG:
                vis = rotated.copy()
                cv2.rectangle(
                    vis,
                    (int(x0_page), int(y_top)),
                    (int(x1_page), int(y_bottom)),
                    (0,0,255),
                    3,
                )
                name = f"{page_num:03d}.crop_debug_rotated_line730.jpg"
                print(f"writing {name}")
                cv2.imwrite(name, vis)

        else:
            # crop as a quadrilateral
            # not better than "crop as a rectangle"?
            if 0:
                # expand the binding edge to expected_w
                if bad_on_left:
                    # outside edge is the RIGHT edge
                    x1_top = outside_top[0]
                    x1_bottom = outside_bottom[0]
                    x0_top = x1_top - expected_w
                    x0_bottom = x1_bottom - expected_w
                else:
                    # outside edge is the LEFT edge
                    x0_top = outside_top[0]
                    x0_bottom = outside_bottom[0]
                    x1_top = x0_top + expected_w
                    x1_bottom = x0_bottom + expected_w
            else:
                # dont expand the binding edge to expected_w
                # use only the detected page edges
                # if bad_on_left:
                #     # outside edge is RIGHT edge
                #     x1_top = outside_top[0]
                #     x1_bottom = outside_bottom[0]
                #     # use the actual detected left edge
                #     x0_top = np.min(page_contour[:,0,0])
                #     x0_bottom = x0_top
                # else:
                #     # outside edge is LEFT edge
                #     x0_top = outside_top[0]
                #     x0_bottom = outside_bottom[0]
                #     # use the actual detected right edge
                #     x1_top = np.max(page_contour[:,0,0])
                #     x1_bottom = x1_top
                # problem: page_contour after thresholding may include the background
                # or may not have a reliable missing-edge position.
                # A cleaner temporary solution is to use the detected quadrilateral width
                # from the two horizontal edge intersections
                if bad_on_left:
                    # outside edge is right
                    x1_top = outside_top[0]
                    x1_bottom = outside_bottom[0]
                    # find leftmost detected page boundary
                    x0_top = np.min(top_pts[:,0])
                    x0_bottom = np.min(bottom_pts[:,0])
                else:
                    # outside edge is left
                    x0_top = outside_top[0]
                    x0_bottom = outside_bottom[0]
                    # find rightmost detected page boundary
                    x1_top = np.max(top_pts[:,0])
                    x1_bottom = np.max(bottom_pts[:,0])

                # dont expand the binding edge to expected_w
                expected_w = int(round(
                    math.dist((x0_top, outside_top[1]), (x1_top, outside_top[1]))
                ))

src = np.float32([
                [x0_top, outside_top[1]],
                [x1_top, outside_top[1]],
                [x1_bottom, outside_bottom[1]],
                [x0_bottom, outside_bottom[1]],
            ])

            dst = np.float32([
                [0,0],
                [expected_w-1,0],
                [expected_w-1,expected_h-1],
                [0,expected_h-1],
            ])

            M = cv2.getPerspectiveTransform(src, dst)

            crop = cv2.warpPerspective(
                rotated,
                M,
                (expected_w, expected_h),
                borderValue=(255,255,255)
            )

            if DEBUG:
                print(
                    "line 650: crop",
                    f"W_img={W_img}",
                    f"H_img={H_img}",
                    f"outside_top={outside_top}",
                    f"outside_bottom={outside_bottom}",
                    f"x1_top={x1_top}",
                    f"x1_bottom={x1_bottom}",
                    f"x0_top={x0_top}",
                    f"x0_bottom={x0_bottom}",
                    f"expected_w={expected_w}",
                    f"expected_h={expected_h}",
                )

            if DEBUG:
                vis = rotated.copy()
                # margin range: green
                if bad_on_left:
                    # outside edge is right
                    # no line on the left
                    pts = np.array([
                        [0, edge_search_height_px], # top left
                        [W_img - edge_search_width_px, edge_search_height_px], # top right
                        [W_img - edge_search_width_px, H_img - edge_search_height_px], # bottom right
                        [0, H_img - edge_search_height_px], # bottom left
                    ], np.int32)
                else:
                    # outside edge is left
                    # no line on the right
                    pts = np.array([
                        [W_img, edge_search_height_px], # top right
                        [edge_search_width_px, edge_search_height_px], # top left
                        [edge_search_width_px, H_img - edge_search_height_px], # bottom left
                        [W_img, H_img - edge_search_height_px], # bottom right
                    ], np.int32)
                cv2.polylines(vis, [pts], False, (0,255,0), 3) # (0,255,0) == green?
                # page margin: red
                pts = np.array([
                    [x0_top, outside_top[1]],
                    [x1_top, outside_top[1]],
                    [x1_bottom, outside_bottom[1]],
                    [x0_bottom, outside_bottom[1]],
                ], np.int32)
                cv2.polylines(vis, [pts], True, (0,0,255), 3)
                name = f"{page_num:03d}.crop_debug_rotated_line730.jpg"
                print(f"writing {name}")
                cv2.imwrite(name, vis)

        # crop = rotated[
        #     0:expected_h,
        #     int(x0_page):int(x1_page)
        # ]

        # y_top = round(outside_top[1])
        # y_bottom = round(outside_bottom[1])
        # actual_height = y_bottom - y_top
        # crop = rotated[
        #     y_top:y_top + expected_h,
        #     x0_page:x1_page
        # ]

        # y_top = int(round(outside_top[1]))
        # y_bottom = int(round(outside_bottom[1]))
        # crop = rotated[
        #     y_top:y_bottom,
        #     x0_page:x1_page
        # ]

        if 0:
            # img = repair_binding(img, bad_on_left, width=50)
            if 1:
                # rotated = repair_binding(rotated, bad_on_left, width=50)
                # Hr, Wr = rotated.shape[:2]
                # print("line 660: rotated size", Wr, Hr)
                # warped = rotated
                crop = repair_binding(crop, bad_on_left, width=50)
                Hr, Wr = crop.shape[:2]
                if DEBUG:
                    print("line 660: crop size", Wr, Hr)
            else:
                crop = repair_binding(crop, bad_on_left, width=50)

            if DEBUG:
                print(
                    "line 760: after repair_binding: crop actual:",
                    f"crop.shape[1]={crop.shape[1]}",
                    f"expected_w={expected_w}",
                )

        warped = crop

    else:
        # config.use_three_edge_deskew == False

        raise NotImplementedError("sorry, your book is too tall for your scanner...")

        # FIXME use only two page edges: outside, bottom

        # Approximate contour to quadrilateral
        epsilon = 0.02 * cv2.arcLength(page_contour, True)
        approx = cv2.approxPolyDP(page_contour, epsilon, True)
        if len(approx) != 4:
            approx = cv2.convexHull(page_contour)
            if len(approx) < 4:
                print(f"Warning: not enough points for perspective in {in_path}")
                return
            # pick 4 extreme points
            pts = np.array([
                approx[approx[:,0,0].argmin()][0],  # leftmost
                approx[approx[:,0,1].argmin()][0],  # topmost
                approx[approx[:,0,0].argmax()][0],  # rightmost
                approx[approx[:,0,1].argmax()][0]   # bottommost
            ])
        else:
            pts = approx.reshape(4,2)

        rect = order_points(pts)

        # Perspective transform
        widthA = np.linalg.norm(rect[2] - rect[3])
        widthB = np.linalg.norm(rect[1] - rect[0])
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.linalg.norm(rect[1] - rect[2])
        heightB = np.linalg.norm(rect[0] - rect[3])
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    # # Add internal white border
    # h, w = warped.shape[:2]
    # canvas = np.ones_like(warped) * 255
    # b = BORDER_SIZE
    # canvas[b:h-b, b:w-b] = warped[b:h-b, b:w-b]

    h, w = warped.shape[:2]
    b = BORDER_SIZE

    # Prepare a canvas with the same content as warped
    canvas = warped.copy()

    # Function to compute average color along a strip
    def avg_color_strip(img, axis, start, end, strip_width=1):
        if axis == 'top':
            strip = img[start:end, :, :]
        elif axis == 'bottom':
            strip = img[h-end:h-start, :, :]
        elif axis == 'left':
            strip = img[:, start:end, :]
        elif axis == 'right':
            strip = img[:, w-end:w-start, :]
        else:
            raise ValueError("Invalid axis")
        return np.mean(strip, axis=(0,1)).astype(np.uint8)

    # FIXME preserve patterns near edges
    if 0:
        # Fill borders with local average color
        canvas[0:b, :, :] = avg_color_strip(canvas, 'top', 50, 100)       # top border
        canvas[h-b:h, :, :] = avg_color_strip(canvas, 'bottom', 50, 100)  # bottom border
        canvas[:, 0:b, :] = avg_color_strip(canvas, 'left', 50, 100)      # left border
        canvas[:, w-b:w, :] = avg_color_strip(canvas, 'right', 50, 100)   # right border

# 9. save

    if input_is_grayscale:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

    ensure_dir(os.path.dirname(out_path))
    cv2.imwrite(out_path, canvas)
    print(f"writing {out_path}")
```

## Response:

7/13/2026, 7:11:16 PM

From what you've shown, I don't think the search margins are actually being ignored by `detect_edge_points`.

The more likely problem is that **you stop using the result of `detect_edge_points()` halfway through the pipeline.**

Your pipeline currently looks like this:

    mask
        ↓
    detect_edge_points()      ← search margins applied here ✓
        ↓
    top_pts, bottom_pts, outside_pts
        ↓
    RANSAC
        ↓
    rotate
        ↓
    get_gray_mask_contours(rotated)
        ↓
    page_contour
        ↓
    split_edge_candidates(page_contour)   ← old algorithm
        ↓
    top_pts, bottom_pts, outside_pts

So after rotation you completely throw away the carefully detected edge points and go back to using the contour.

That means:

- your initial search-margin detection is correct,
- but the final crop is computed from `page_contour`,
- and `page_contour` happily contains the orange rectangle.

------------------------------------------------------------------------

## I think this is exactly where it goes wrong

After rotation you have

``` py
gray, mask, contours = get_gray_mask_contours(rotated, dbgdir)
...
page_contour = max(contours, key=cv2.contourArea)

top_pts, bottom_pts, outside_pts = split_edge_candidates(
    page_contour,
    bad_on_left
)
```

`split_edge_candidates()` has **no knowledge whatsoever** of

    edge_search_width_px
    edge_search_height_px

It simply splits the contour into three edges.

So any large blob connected to the page contour (orange cover, shadow, etc.) can become part of an edge.

------------------------------------------------------------------------

## Another clue

You said

> the red lines are obviously inside of the green lines

But how are you drawing the red lines?

You're drawing

``` py
x0_top
x1_top
...
```

Those come from

``` py
top_pts, bottom_pts, outside_pts = split_edge_candidates(page_contour)
```

not from

``` py
detect_edge_points(...)
```

So the debug image is visualizing the **old contour algorithm**, not the new search-margin algorithm.

That explains why the red polygon violates the green search region.

------------------------------------------------------------------------

## I would remove almost all of the old code

I think this is what your pipeline should become:

    gray
    ↓
    mask
    ↓
    detect_edge_points()      <-- only detector
    ↓
    verify_transition()
    ↓
    reject_outliers()
    ↓
    RANSAC
    ↓
    rotate image
    ↓
    repeat detect_edge_points() on rotated image
    ↓
    repeat verify
    ↓
    repeat RANSAC
    ↓
    crop

Notice what's missing:

    get_gray_mask_contours()
    split_edge_candidates()
    page_contour

Those belong to the previous algorithm.

------------------------------------------------------------------------

## After rotation

Instead of

``` py
gray, mask, contours = get_gray_mask_contours(rotated, dbgdir)

page_contour = max(contours, key=cv2.contourArea)

top_pts, bottom_pts, outside_pts = split_edge_candidates(
    page_contour,
    bad_on_left
)
```

I would literally do

``` py
gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)

_, mask = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

if bad_on_left:
    outside_pts = detect_edge_points(
        gray,
        mask,
        EDGE_RIGHT,
        edge_search_width_px,
    )
else:
    outside_pts = detect_edge_points(
        gray,
        mask,
        EDGE_LEFT,
        edge_search_width_px,
    )

top_pts = detect_edge_points(
    gray,
    mask,
    EDGE_TOP,
    edge_search_height_px,
)

bottom_pts = detect_edge_points(
    gray,
    mask,
    EDGE_BOTTOM,
    edge_search_height_px,
)

top_pts = reject_outliers_horizontal(top_pts)
bottom_pts = reject_outliers_horizontal(bottom_pts)
outside_pts = reject_outliers_vertical(outside_pts)

top_line = fit_line_ransac(top_pts)[:4]
bottom_line = fit_line_ransac(bottom_pts)[:4]
outside_line = fit_line_ransac(outside_pts)[:4]
```

Then the search margins are enforced **both before and after rotation**.

------------------------------------------------------------------------

## One more thing I noticed

There is a suspicious bug in your debug drawing.

You draw

``` py
pts = np.array([
    [W_img, edge_search_height_px],
    [edge_search_width_px, edge_search_height_px],
    [edge_search_width_px, H_img - edge_search_height_px],
    [W_img, H_img - edge_search_height_px],
])
```

But this uses

    W_img
    H_img

which are the **original** image dimensions.

Your crop is performed on

``` py
rotated
```

whose size is

    Wr
    Hr

You even compute

``` py
Hr, Wr = rotated.shape[:2]
```

earlier.

If rotation or later scaling changes the size, your green search rectangle is no longer in the correct place.

So at minimum, the debug visualization should use

``` py
Wr
Hr
```

instead of

``` py
W_img
H_img
```

Otherwise the green guide itself can be misleading.

Overall, though, the biggest issue isn't the search margins—it's that the new edge detector is only used for the initial fit, while the final crop still falls back to the old contour-based `split_edge_candidates()` logic. As long as that code remains in the pipeline, objects like a solid orange rectangle can still influence the detected page edges regardless of the initial search-margin filtering.

## Prompt:

7/13/2026, 8:07:32 PM

good news:  
now edge_search is honored

bad news:  
now the script fails to detect the binding edge...  
expected:  
on all pages, the binding edge should have X=0 on odd pages and X=X_max on even pages  
actual:  
the bottom point of each binding edge is correct, but the top point of each binding edge is somewhere in the middle of the page, so the red line in the debug image looks almost like a triangle...

## Response:

7/13/2026, 8:07:33 PM

That's actually a useful failure mode—it suggests the detector is mostly working, but one endpoint of the fitted line is being pulled by bad points.

The key observation is:

- **bottom of the binding edge is correct**
- **top of the binding edge is wrong**
- **RANSAC still fits a line through them**

This means one of these is happening:

1.  The **top half of `outside_pts` contains many bad points**, so RANSAC thinks they are the dominant line.
2.  The **top half has very few points**, so the fit is determined mostly by the bottom half.
3.  Your **transition verification rejects the correct top-edge points** but accepts the bottom ones.

## First thing I'd do

Before fitting the line, dump the detected points:

``` py
vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

for x, y in outside_pts.astype(int):
    cv2.circle(vis, (x, y), 2, (0, 0, 255), -1)

cv2.imwrite("outside_pts.png", vis)
```

This image will tell you immediately whether the problem is

- bad point detection, or
- bad line fitting.

------------------------------------------------------------------------

## My suspicion

I suspect it's **not** `fit_line_ransac()`.

Suppose your detected points look like this:

    top

    page edge
    |
    |
    |
    |
    |
    |

    xxxxxxxxxxxxxxxx    <-- orange rectangle
    xxxxxxxxxxxxxxxx

    |
    |
    |
    |

    page edge

    bottom

Then the upper half of `outside_pts` will be shifted inward.

RANSAC happily fits

    \
     \
      \
       |
       |
       |

which is exactly what you're describing.

------------------------------------------------------------------------

## Another possibility

Your `verify_vertical()` only checks intensity:

``` py
outside = ...
inside = ...

if inside <= outside:
    continue
```

That is a *local* test.

An orange rectangle can easily satisfy

    gray background
    ↓

    orange rectangle
    ↓

    white page

or

    gray background
    ↓

    orange

depending on the grayscale values.

So it doesn't verify that you're seeing the **actual page boundary**, only that brightness changes.

------------------------------------------------------------------------

## I'd also print how many points survive

For example:

``` py
print(
    len(outside_pts),
    outside_pts[:,1].min(),
    outside_pts[:,1].max(),
)
```

or even better

``` py
ys = outside_pts[:,1]

print(
    np.percentile(ys, [0,10,20,30,40,50,60,70,80,90,100])
)
```

If all surviving points are in the lower half, you've found the culprit.

------------------------------------------------------------------------

## One thing I noticed

You previously had

``` py
xs = np.where(mask[y, x_min:] == 255)[0]
```

for each row.

That means **one point per row**.

If verification removes the point for a row,

that entire row disappears.

If the first 300 rows disappear, then RANSAC only sees

    ................
    ................
    ................
    xxxxxxxxxxxxxxxx
    xxxxxxxxxxxxxxxx
    xxxxxxxxxxxxxxxx

and extrapolates upward.

------------------------------------------------------------------------

## I would debug in this order

1.  Save an image of **every `outside_pts`**.
2.  Save another image of the **RANSAC inliers**.
3.  Compare the two.

My guess is you'll discover that the incorrect top endpoint is already present in `outside_pts`, meaning the problem is in `detect_edge_points()` or `verify_vertical()`, not in the line fitting.

One question that would help narrow it down: **Is the fitted line wrong because its *slope* is wrong, or is the slope correct but the line is simply shifted horizontally at the top?** Those two cases point to different bugs.

## Prompt:

7/13/2026, 8:15:56 PM

> Before fitting the line, dump the detected points:

no, the problem is not outside_pts, the problem is "inside_pts" = the binding edge, which should simply have X=0 on odd pages and X=X_max on even pages. currently only the bottom point is correct, but the top point of the detected binding edge is somewhere at X=X_max/2 ...

## Response:

7/13/2026, 8:15:58 PM

Ah, that changes the diagnosis.

If the **outside edge is correct**, but the **binding edge** (the inferred missing edge) is wrong, then the problem is **after** line detection.

The binding edge is not detected directly. You construct it here:

``` py
if bad_on_left:
    # outside edge is right
    x1_top = outside_top[0]
    x1_bottom = outside_bottom[0]

    # binding edge
    x0_top = np.min(top_pts[:,0])
    x0_bottom = np.min(bottom_pts[:,0])
else:
    # outside edge is left
    x0_top = outside_top[0]
    x0_bottom = outside_bottom[0]

    # binding edge
    x1_top = np.max(top_pts[:,0])
    x1_bottom = np.max(bottom_pts[:,0])
```

Notice that **`x0_top` and `x0_bottom` are computed completely independently**.

If the top edge detection doesn't extend all the way to the binding (which is common because of the gutter shadow or missing page corner), then

- `np.min(top_pts[:,0])` may be halfway across the page,
- while `np.min(bottom_pts[:,0])` is correctly near the binding.

The result is exactly the "triangle" you describe.

## Why this happens

Your `top_pts` are found by scanning downward from the top.

Near the binding, the first white pixel may never be found because of:

- gutter shadow,
- fingers,
- page curl,
- dark print,
- missing corner.

So those columns simply contribute no point.

Consequently,

``` py
np.min(top_pts[:,0])
```

is **the leftmost column that happened to survive**, not the true binding position.

The bottom edge doesn't have that problem, so

``` py
np.min(bottom_pts[:,0])
```

is much closer to the true binding.

## I wouldn't use `min(top_pts[:,0])` at all

The binding edge is supposed to be vertical.

Instead of taking two unrelated extrema, estimate a **single x coordinate**.

For example:

``` py
binding_x = np.min(
    np.concatenate([
        top_pts[:,0],
        bottom_pts[:,0],
    ])
)
```

or better, use a robust statistic:

``` py
binding_x = np.percentile(
    np.concatenate([
        top_pts[:,0],
        bottom_pts[:,0],
    ]),
    5,
)
```

Then

``` py
x0_top = binding_x
x0_bottom = binding_x
```

and the binding edge is vertical.

## Even better

Since this is a scanned book, you already know a strong prior:

- odd pages → binding is the **left image border**
- even pages → binding is the **right image border**

If the scanner consistently crops at the binding, then there is little value in estimating it from `top_pts` and `bottom_pts` at all.

You can simply use

``` py
if bad_on_left:
    x0_top = 0
    x0_bottom = 0
else:
    x1_top = Wr - 1
    x1_bottom = Wr - 1
```

(or `W_img - 1` if you're still using the original image dimensions at that point).

That is actually consistent with what you wrote:

> on all pages, the binding edge should have X=0 on odd pages and X=X_max on even pages.

If that's an invariant of your scanning setup, then it's much more reliable than trying to infer the missing edge from incomplete `top_pts`/`bottom_pts`. It also removes an entire source of noise from the perspective transform.

------------------------------------------------------------------------

Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)

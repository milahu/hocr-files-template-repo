#!/usr/bin/env python3
"""
Robust removal of crooked binding edge — preserves original colors and character holes.
Key fix: create a FILLED page mask (no inner holes) from the page contour and use that
for filling after warping. The original image pixels are preserved and warped directly.
"""

CONFIG_FILE = "030-measure-page-size.txt"
INPUT_DIR = "060-rotate-crop-level"
OUTPUT_DIR = "065-remove-page-borders"

# === Tuning parameters ===
DEBUG = True
DEBUG = False
BORDER_SIZE = 0

# Number of pixels to fill with the page background color inside the bad page edge
# to remove faint grey line artifacts
BAD_EDGE_CLEANUP_FILL_PX = 5

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

def read_page_size_from_config(config_path):
    """
    Parse scan_x and scan_y from a bash-style config file.
    Only integer assignments are considered; ignores comments and dynamic expressions.
    Returns (W_mm, H_mm)
    """
    W_mm = H_mm = None
    with open(config_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # match lines like: scan_x=120
            m = re.match(r'^scan_x\s*=\s*(\d+)', line)
            if m:
                W_mm = int(m.group(1))
            m = re.match(r'^scan_y\s*=\s*(\d+)', line)
            if m:
                H_mm = int(m.group(1))
            if W_mm is not None and H_mm is not None:
                break
    if W_mm is None or H_mm is None:
        raise ValueError(f"scan_x or scan_y not found in {config_path}")
    return W_mm, H_mm

# W_mm = 120
# H_mm = 190
# ASPECT = W_mm / H_mm
W_mm, H_mm = read_page_size_from_config(CONFIG_FILE)
ASPECT = W_mm / H_mm

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
    if len(img.shape) == 2:
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

    # --- remove vertical streaks but preserve holes ---
    streaks = detect_vertical_streaks(mask_init)
    mask_nostreak = mask_init.copy()
    mask_nostreak[streaks == 255] = 0
    page_mask_with_holes = keep_largest_component(mask_nostreak)
    if DEBUG:
        save_dbg(streaks, os.path.join(dbgdir, "03_streaks.png"))
        save_dbg(mask_nostreak, os.path.join(dbgdir, "04_mask_nostreak.png"))
        save_dbg(page_mask_with_holes, os.path.join(dbgdir, "05_page_mask_with_holes.png"))

    # --- contour and filled mask (no holes) ---
    contours, _ = cv2.findContours(page_mask_with_holes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        print(f"No contours for {in_path}"); return
    page_contour = max(contours, key=lambda c: cv2.contourArea(c))
    contour_pts = contour_to_pts(page_contour)

    filled_page_mask = np.zeros_like(page_mask_with_holes)
    cv2.drawContours(filled_page_mask, [page_contour], -1, 255, thickness=-1)
    if DEBUG:
        save_dbg(filled_page_mask, os.path.join(dbgdir, "06_filled_page_mask.png"))
        overlay = img.copy()
        cv2.drawContours(overlay, [page_contour], -1, (0,255,0), 2)
        save_dbg(overlay, os.path.join(dbgdir, "07_contour_overlay.png"))

    # --- candidate sets for RANSAC ---
    ys = contour_pts[:,1]; xs = contour_pts[:,0]
    top_cand = contour_pts[ys <= np.percentile(ys, 20)]
    bottom_cand = contour_pts[ys >= np.percentile(ys, 80)]
    left_cand = contour_pts[xs <= np.percentile(xs, 20)]
    right_cand = contour_pts[xs >= np.percentile(xs, 80)]
    if len(top_cand) < 20: top_cand = contour_pts[np.argsort(ys)[:max(20, len(contour_pts)//8)]]
    if len(bottom_cand) < 20: bottom_cand = contour_pts[np.argsort(ys)[-max(20, len(contour_pts)//8):]]
    if len(left_cand) < 12: left_cand = contour_pts[np.argsort(xs)[:max(12, len(contour_pts)//10)]]
    if len(right_cand) < 12: right_cand = contour_pts[np.argsort(xs)[-max(12, len(contour_pts)//10):]]

    # --- fit lines robustly ---
    top_vx, top_vy, top_x0, top_y0, top_inliers = fit_line_ransac(top_cand)
    bottom_vx, bottom_vy, bottom_x0, bottom_y0, bottom_inliers = fit_line_ransac(bottom_cand)
    if bad_on_left:
        vert_vx, vert_vy, vert_x0, vert_y0, vert_inliers = fit_line_ransac(right_cand)
    else:
        vert_vx, vert_vy, vert_x0, vert_y0, vert_inliers = fit_line_ransac(left_cand)

    # counts for diagnostics
    top_in_count = int(np.count_nonzero(top_inliers)) if isinstance(top_inliers, (list, np.ndarray)) else len(top_cand)
    bottom_in_count = int(np.count_nonzero(bottom_inliers)) if isinstance(bottom_inliers, (list, np.ndarray)) else len(bottom_cand)
    vert_in_count = int(np.count_nonzero(vert_inliers)) if isinstance(vert_inliers, (list, np.ndarray)) else len(right_cand if bad_on_left else left_cand)

    if DEBUG:
        viz = img.copy()
        cv2.drawContours(viz, [page_contour], -1, (0,255,0), 2)
        def draw_line(vx,vy,x0,y0,color,label):
            norm = math.hypot(vx,vy)
            if norm < 1e-6: return
            ux, uy = vx/norm, vy/norm
            L = max(W_img, H_img) * 2
            p1 = (int(x0 - ux*L), int(y0 - uy*L))
            p2 = (int(x0 + ux*L), int(y0 + uy*L))
            cv2.line(viz, p1, p2, color, 2)
            cv2.putText(viz, f"{label}:{(top_in_count if label=='top' else bottom_in_count if label=='bottom' else vert_in_count)}",
                        (int(x0), int(y0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        draw_line(top_vx, top_vy, top_x0, top_y0, (255,0,0), "top")
        draw_line(bottom_vx, bottom_vy, bottom_x0, bottom_y0, (0,0,255), "bottom")
        draw_line(vert_vx, vert_vy, vert_x0, vert_y0, (0,255,255), "good_vert")
        save_dbg(viz, os.path.join(dbgdir, "08_fitted_lines_overlay.png"))

    # --- compute intersections on good vertical ---
    top_line = (top_vx, top_vy, top_x0, top_y0)
    bottom_line = (bottom_vx, bottom_vy, bottom_x0, bottom_y0)
    vert_line = (vert_vx, vert_vy, vert_x0, vert_y0)
    # attempt to compute pt_v_top and pt_v_bot where possible
    pt_v_top = None; pt_v_bot = None
    try:
        pt_v_top = np.array(intersect_lines(top_line, vert_line), dtype=np.float32)
    except Exception:
        pt_v_top = None
    try:
        pt_v_bot = np.array(intersect_lines(bottom_line, vert_line), dtype=np.float32)
    except Exception:
        pt_v_bot = None

    # --- Decide which edges are reliable ---
    top_ok = (top_in_count >= max(6, RANSAC_MIN_INLIERS//2)) and (pt_v_top is not None)
    bottom_ok = (bottom_in_count >= max(6, RANSAC_MIN_INLIERS//2)) and (pt_v_bot is not None)
    vert_ok = (vert_in_count >= max(6, RANSAC_MIN_INLIERS//2))

    if DEBUG:
        with open(os.path.join(dbgdir, "diagnostics.txt"), "a") as fh:
            fh.write(f"top_inliers={top_in_count}, bottom_inliers={bottom_in_count}, vert_inliers={vert_in_count}\n")
            fh.write(f"top_ok={top_ok}, bottom_ok={bottom_ok}, vert_ok={vert_ok}\n")

    # --- Build orthogonal axes and corners ---
    # Prefer to use top line if available; if top missing, use bottom + vertical to infer top by moving up expected_h
    used_anchor = None
    if vert_ok and top_ok:
        # standard: use top intersection and vertical to define right/left anchors
        used_anchor = "top"
        anchor_top = pt_v_top
        anchor_bot = pt_v_bot if bottom_ok else (pt_v_top + np.array([0.0, expected_h], dtype=np.float32))  # fallback
    elif vert_ok and bottom_ok:
        # top is missing: anchor on bottom intersection and move up
        used_anchor = "bottom"
        anchor_bot = pt_v_bot
        # vertical unit pointing downwards
        vvx, vvy = vert_vx, vert_vy
        vnorm = math.hypot(vvx, vvy) + 1e-12
        v_unit = np.array([vvx / vnorm, vvy / vnorm], dtype=np.float32)
        # ensure pointing downward: dot with (0,1) > 0
        if v_unit[1] < 0:
            v_unit = -v_unit
        # compute top anchor by moving up expected_h along vertical
        anchor_top = anchor_bot - v_unit * float(expected_h)
    else:
        # not enough reliable edges; fall back to convex hull approx (let the perspective fallback handle)
        used_anchor = "none"

    # if we have an anchor and vertical available, construct orthonormal axes u (right) and p (down)
    src_TL = src_TR = src_BL = src_BR = None
    if used_anchor in ("top", "bottom") and vert_ok:
        # vertical direction unit
        vvx, vvy = vert_vx, vert_vy
        vnorm = math.hypot(vvx, vvy) + 1e-12
        v_unit = np.array([vvx / vnorm, vvy / vnorm], dtype=np.float32)
        # make sure vertical points downwards
        if v_unit[1] < 0:
            v_unit = -v_unit
        # horizontal unit is perpendicular to vertical
        u_unit = np.array([-v_unit[1], v_unit[0]], dtype=np.float32)  # rightwards candidate
        # ensure u_unit points right (positive x)
        if u_unit[0] < 0:
            u_unit = -u_unit

        # anchor_top and anchor_bot are set above (top_ok or computed)
        # anchored points lie on the *good* vertical edge: if bad_on_left==True then this vertical is RIGHT edge else LEFT edge
        if bad_on_left:
            # vertical is RIGHT edge: anchor_top is TR, anchor_bot is BR
            src_TR = anchor_top
            src_BR = anchor_bot
            src_TL = src_TR - u_unit * float(expected_w)
            src_BL = src_BR - u_unit * float(expected_w)
            # ensure exact vertical height (recompute using p vector)
            # p vector (down) is v_unit, so TL->BL = v_unit*expected_h
            src_BL = src_TL + v_unit * float(expected_h)
            src_BR = src_TR + v_unit * float(expected_h)
        else:
            # vertical is LEFT edge: anchor_top is TL, anchor_bot is BL
            src_TL = anchor_top
            src_BL = anchor_bot
            src_TR = src_TL + u_unit * float(expected_w)
            src_BR = src_TR + v_unit * float(expected_h)
            src_BL = src_TL + v_unit * float(expected_h)
    else:
        # we don't have enough to build shear-free rectangle; fallback to earlier methods
        src_tri = None

    # --- If we built corners, make source/destination triangles and warp affine ---
    warped = None; warped_mask = None
    if src_TL is not None and src_TR is not None and src_BL is not None:
        src_tri = np.vstack([src_TL, src_BL, src_TR]).astype(np.float32)  # TL, BL, TR
        dst_tri = np.vstack([
            np.array([0.0, 0.0], dtype=np.float32),
            np.array([0.0, float(expected_h - 1)], dtype=np.float32),
            np.array([float(expected_w - 1), 0.0], dtype=np.float32)
        ]).astype(np.float32)

        # validate source triangle
        def tri_area(tri):
            return abs(0.5 * (tri[0,0]*(tri[1,1]-tri[2,1]) + tri[1,0]*(tri[2,1]-tri[0,1]) + tri[2,0]*(tri[0,1]-tri[1,1])))
        area = tri_area(src_tri)
        if DEBUG:
            with open(os.path.join(dbgdir, "diagnostics.txt"), "a") as fh:
                fh.write(f"src_tri_area={area:.3f}\n")
        if area > 1.0 and np.isfinite(src_tri).all():
            try:
                M_aff = cv2.getAffineTransform(src_tri, dst_tri)
                warped = cv2.warpAffine(img, M_aff, (expected_w, expected_h), flags=cv2.INTER_LINEAR, borderValue=(255,255,255))
                warped_mask = cv2.warpAffine(filled_page_mask, M_aff, (expected_w, expected_h), flags=cv2.INTER_NEAREST, borderValue=0)
                if DEBUG:
                    # draw axes and corners overlay
                    vis = img.copy()
                    def draw_pt(pt, color, tag):
                        cv2.circle(vis, (int(pt[0]), int(pt[1])), 6, color, -1)
                        cv2.putText(vis, tag, (int(pt[0]+6), int(pt[1]-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    draw_pt(src_TL, (0,255,0), "TL")
                    draw_pt(src_TR, (0,128,255), "TR")
                    draw_pt(src_BL, (255,0,0), "BL")
                    draw_pt(src_BR, (255,255,0), "BR")
                    save_dbg(vis, os.path.join(dbgdir, "debug_axes_and_corners.png"))
                    save_dbg(warped, os.path.join(dbgdir, "09_warped_affine.png"))
                    save_dbg(warped_mask, os.path.join(dbgdir, "10_warped_mask_affine.png"))
            except Exception as e:
                if DEBUG:
                    with open(os.path.join(dbgdir, "diagnostics.txt"), "a") as fh:
                        fh.write(f"affine warp exception: {e}\n")
                warped = None; warped_mask = None
        else:
            if DEBUG:
                with open(os.path.join(dbgdir, "diagnostics.txt"), "a") as fh:
                    fh.write(f"src_tri invalid area={area}\n")
            warped = None; warped_mask = None

    # --- If affine failed or wasn't built, do perspective fallback as before ---
    def warped_ok(wimg, wmask):
        if wimg is None or wmask is None:
            return False, "none"
        nonwhite = np.count_nonzero(np.any(wimg != 255, axis=2))
        nonwhite_ratio = nonwhite / float(expected_w * expected_h)
        mask_count = int(np.count_nonzero(wmask))
        mask_ratio = mask_count / float(expected_w * expected_h)
        if DEBUG:
            with open(os.path.join(dbgdir, "diagnostics.txt"), "a") as fh:
                fh.write(f"nonwhite_ratio={nonwhite_ratio:.4f}, mask_ratio={mask_ratio:.4f}\n")
        if nonwhite_ratio < 0.02 or mask_ratio < 0.005:
            return False, f"low_content({nonwhite_ratio:.3f}, mask={mask_ratio:.3f})"
        return True, "ok"

    ok, why = warped_ok(warped, warped_mask)
    if not ok:
        # perspective fallback using approx polygon (same as earlier code)
        hull = cv2.convexHull(page_contour)
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
        if len(approx) >= 4:
            pts = approx.reshape(-1,2).astype(np.float32)
            tl = pts[np.argmin(pts[:,0] + pts[:,1])]
            br = pts[np.argmax(pts[:,0] + pts[:,1])]
            tr = pts[np.argmin(pts[:,1] - pts[:,0])]
            bl = pts[np.argmax(pts[:,1] - pts[:,0])]
            src_quad = np.vstack([tl, tr, br, bl]).astype(np.float32)
            dst_quad = np.array([[0,0], [expected_w-1,0], [expected_w-1,expected_h-1], [0,expected_h-1]], dtype=np.float32)
            try:
                M_p = cv2.getPerspectiveTransform(src_quad, dst_quad)
                warped_p = cv2.warpPerspective(img, M_p, (expected_w, expected_h), flags=cv2.INTER_LINEAR, borderValue=(255,255,255))
                warped_mask_p = cv2.warpPerspective(filled_page_mask, M_p, (expected_w, expected_h), flags=cv2.INTER_NEAREST, borderValue=0)
                ok2, why2 = warped_ok(warped_p, warped_mask_p)
                if ok2:
                    warped = warped_p; warped_mask = warped_mask_p
                    if DEBUG:
                        save_dbg(warped, os.path.join(dbgdir, "09_warped_perspective.png"))
                        save_dbg(warped_mask, os.path.join(dbgdir, "10_warped_mask_persp.png"))
                else:
                    if DEBUG:
                        with open(os.path.join(dbgdir, "diagnostics.txt"), "a") as fh:
                            fh.write(f"perspective fallback bad: {why2}\n")
            except Exception as e:
                if DEBUG:
                    with open(os.path.join(dbgdir, "diagnostics.txt"), "a") as fh:
                        fh.write(f"perspective fallback error: {e}\n")

    # final fallback: bounding box crop+resize
    ok_final, why_final = warped_ok(warped, warped_mask)
    if not ok_final:
        if DEBUG:
            with open(os.path.join(dbgdir, "diagnostics.txt"), "a") as fh:
                fh.write("Falling back to bbox crop+resize\n")
        x,y,ww,hh = cv2.boundingRect(page_contour)
        crop = img[max(0,y-2):min(H_img,y+hh+2), max(0,x-2):min(W_img,x+ww+2)]
        if crop.size == 0:
            print(f"[{page_num}] bbox crop empty; skipping.")
            return
        warped = cv2.resize(crop, (expected_w, expected_h), interpolation=cv2.INTER_LINEAR)
        warped_mask = cv2.resize(filled_page_mask[y:y+hh, x:x+ww], (expected_w, expected_h), interpolation=cv2.INTER_NEAREST)
        if DEBUG:
            save_dbg(warped, os.path.join(dbgdir, "09_warped_bbox_resized.png"))
            save_dbg(warped_mask, os.path.join(dbgdir, "10_warped_mask_bbox.png"))

    # shrink the page mask by a few pixels to also cover the old edge artifact
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BAD_EDGE_CLEANUP_FILL_PX, BAD_EDGE_CLEANUP_FILL_PX))
    eroded_mask = cv2.erode(warped_mask, kernel, iterations=1)

    if DEBUG:
        diff = cv2.subtract(warped_mask, eroded_mask)
        save_dbg(diff, os.path.join(dbgdir, "10b_eroded_diff.png"))

    # --- finalize: fill outside filled mask with white (preserve letter holes) ---
    canvas = warped.copy()
    empty = (eroded_mask == 0)
    if np.any(empty):
        canvas[empty] = (255, 255, 255) # fill white

    ensure_dir(os.path.dirname(out_path))
    cv2.imwrite(out_path, canvas)
    print(f"writing {out_path}")

    if DEBUG:
        save_dbg(canvas, os.path.join(dbgdir, "11_final_output.png"))
        overlay = warped.copy()
        overlay[warped_mask==0] = (0,0,255)
        save_dbg(overlay, os.path.join(dbgdir, "12_missing_overlay.png"))

def main():
    ensure_dir(OUTPUT_DIR)
    files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".tif", ".tiff"))])
    if not files:
        print("No TIFF files found in", INPUT_DIR)
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

if __name__ == "__main__":
    main()

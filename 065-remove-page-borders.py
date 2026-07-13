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



    # TODO verify
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

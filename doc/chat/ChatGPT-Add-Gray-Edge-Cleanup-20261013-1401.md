# Add Gray Edge Cleanup

**User:** Anonymous  
**Created:** 2026/10/11 9:18:48  
**Updated:** 2026/10/13 14:01:13  
**Exported:** 2026/10/13 14:01:30  
**Link:** [<https://chatgpt.com/c/6acb3858-0868-83ed-a551-86b3bdf99d53>](https://chatgpt.com/c/6acb3858-0868-83ed-a551-86b3bdf99d53)

## Prompt:

10/11/2026, 9:28:33 AM

help me to add a feature to my python script

my python script is this:  
<https://github.com/milahu/hocr-files-template-repo/blob/main/065-remove-page-borders.py>

the 065 script is part of a book-scanning pipeline

the feature i want to add:  
the script should cleanup the scanned book page edges, to remove midgray lines parallel to the detected book edges. these midgray lines appear, because my document scanner can distort the page edges, so a straight physical edge can become a curved scanned edge, but the "detect page edge" logic in the script (`outside_pts = detect_edge_points(...)` etc) is only looking for straight lines, which are used to rotate and crop the scanned image, to extract the scanned book page. but the cropping can leave traces of the midgray scanner background (slightly curved lines parallel to the cropped image edges).  
so now i want to cleanup these midgray lines parallel to the cropped image edges, and replace them with white, assuming that the page background is white (black text and graphics on white background).  
so i want to detect this situation:  
a thin midgray strip (with a slightly curved shape on the inner side of the page), followed by a white background (further on the inside of the page). if such a midgray strip is detected, the script should remove it (by filling that area with white color) (that area plus maybe 10 pixels of margin, this should be a parameter for fine-tuning)

here is my current 065 script:

    #!/usr/bin/env python3
    """
    extract scanned page from gray background

    restore the binding edge of the page
    by filling the missing width
    with the average color near the binding edge
    """

    from pathlib import Path

    INPUT_DIR = Path("060-rotate-crop")
    OUTPUT_DIR = Path("065-remove-page-borders")

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

    # cleanup the inside edge
    #
    # the "dirty" inside edge is created
    # by rotating and warping the three other edges
    # so in many cases, the inside edge is crooked (not vertical)
    #
    # if we remove the dirty inside edge
    # then we remove a small rectangle from the inside edge
    # to get a straight inside edge (vertical)
    remove_inside_transparent_strip = True
    #
    # FIXME this produces black triangles on the inside edges
    # if we keep the dirty inside edge
    # then there is a small transparent rectangle inside of the inside edge
    # remove_inside_transparent_strip = False

    # 8c. remove small gray artifacts along page edges

    WHITE_BORDER_WIDTH = 10
    WHITE_TEST_INNER = 10
    WHITE_TEST_OUTER = 20
    WHITE_FRACTION_THRESHOLD = 0.99
    PRACTICALLY_WHITE_THRESHOLD = 250  # for uint8 images

    import os
    import re
    import math
    import time
    import random
    import shutil
    from concurrent.futures import ProcessPoolExecutor, as_completed

    import numpy as np
    import cv2
    import PIL.Image
    import psutil
    from tqdm import tqdm

    from _shared import (
        load_config,
        get_page_num,
        latest_dst_exists,
        remove_done_files,
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
        ensure_dir(path.parent)
        save_image(path, img)

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
            raise ValueError("Not enough points for line fit")
        best_inliers = None
        best_cnt = 0
        best_model = None
        ptsf = pts.astype(np.float32)
        n = len(ptsf)
        for iteration_idx in range(iterations):
            # if DEBUG: print(f"fit_line_ransac: iteration_idx={iteration_idx}")
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
                if DEBUG: print(f"fit_line_ransac: iteration_idx={iteration_idx}: updating best_model to {best_model}")
        if best_model is None:
            if DEBUG: print(f"fit_line_ransac: not found best_model")
            vx, vy, x0, y0 = cv2.fitLine(ptsf, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
            inlier_mask = np.ones(len(ptsf), dtype=bool)
            return float(vx), float(vy), float(x0), float(y0), inlier_mask
        if DEBUG: print(f"fit_line_ransac: found best_model")
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

    def intersect_line_with_vertical_boundary(line, x):
        """
        Intersect a parametric line

            (x0, y0) + t * (vx, vy)

        with the vertical line

            x = constant

        Returns (x, y).
        """
        vx, vy, x0, y0 = map(float, line)
        if abs(vx) < 1e-12:
            raise ValueError("Line is parallel to vertical boundary")
        t = (x - x0) / vx
        y = y0 + t * vy
        return np.array([x, y], dtype=np.float32)

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
            save_image(dbgdir.joinpath("debug_axes_and_corners.png"), vis)

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
            save_dbg(gray, dbgdir.joinpath("01_gray.png"))
            save_dbg(mask_init, dbgdir.joinpath(f"02_thresh_thr{thr}_hp{hp}.png"))

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

    def detect_edge_points(gray, mask, edge, search_size):
        h, w = mask.shape
        pts = []

        if edge == EDGE_TOP:
            region = mask[:min(search_size, h), :] == 255
            valid = region.any(axis=0)
            xs = np.flatnonzero(valid)
            ys = region.argmax(axis=0)[valid]
            pts = np.column_stack((xs, ys))

        elif edge == EDGE_BOTTOM:
            y_min = max(0, h - search_size)
            region = mask[y_min:, :] == 255
            valid = region.any(axis=0)
            xs = np.flatnonzero(valid)
            ys = region.shape[0] - 1 - np.argmax(
                region[::-1, :], axis=0
            )[valid]
            ys += y_min
            pts = np.column_stack((xs, ys))

        elif edge == EDGE_LEFT:
            region = mask[:, :min(search_size, w)] == 255
            valid = region.any(axis=1)
            ys = np.flatnonzero(valid)
            xs = region.argmax(axis=1)[valid]
            pts = np.column_stack((xs, ys))

        else:  # EDGE_RIGHT
            x_min = max(0, w - search_size)
            region = mask[:, x_min:] == 255
            valid = region.any(axis=1)
            ys = np.flatnonzero(valid)
            xs = region.shape[1] - 1 - np.argmax(
                region[:, ::-1], axis=1
            )[valid]
            xs += x_min
            pts = np.column_stack((xs, ys))

        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
        return verify_transition(gray, pts, edge)

    def verify_transition(gray, points, edge):
        if len(points) == 0:
            return np.empty((0,2), dtype=np.float32)

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

            # The outside should look like the solid midgray scanner background.
            if not (80 <= outside <= 180):
                continue

            # The page may be lighter OR darker than the background.
            if abs(float(inside) - float(outside)) < 40:
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

            # The outside should look like the solid midgray scanner background.
            if not (80 <= outside <= 180):
                continue

            # The page may be lighter OR darker than the background.
            if abs(float(inside) - float(outside)) < 40:
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

    def fit_line_ransac_or_none(
            pts,
            iterations=RANSAC_ITER,
            inlier_dist=RANSAC_INLIER_DIST,
            min_inliers=RANSAC_MIN_INLIERS,
        ):
        """
        Fit a line if enough edge points were detected.

        Unlike fit_line_ransac(), this returns None when the edge is
        genuinely missing. A missing edge is expected for scans where
        the scanner has cropped one page edge.
        """
        pts = np.asarray(pts, dtype=np.float32)

        if len(pts) < 2:
            return None

        try:
            return fit_line_ransac(
                pts,
                iterations=iterations,
                inlier_dist=inlier_dist,
                min_inliers=min_inliers,
            )[:4]
        except (ValueError, cv2.error):
            return None

    def horizontal_line_from_image_boundary(
            reference_line,
            boundary_y,
        ):
        """
        Create a horizontal-page edge at the image boundary.

        The missing edge is represented by a line parallel to the detected
        horizontal page edge, but passing through y=boundary_y.

        This is mainly useful for diagnostics / geometry. For the actual
        perspective transform we keep the missing edge at the image boundary.
        """
        vx, vy, x0, y0 = map(float, reference_line)

        # Keep the same direction as the detected edge.
        # We only need a point on the boundary plus the same direction.
        return (
            vx,
            vy,
            0.0,
            float(boundary_y),
        )

    def make_boundary_line(boundary_y, reference_line):
        """
        Return a line parallel to reference_line passing through the
        horizontal image boundary y=boundary_y.

        This is used when one of top/bottom is missing.
        """
        vx, vy, _, _ = map(float, reference_line)

        # A horizontal edge should have vx != 0.
        # Choose x=0 as the point on the line.
        if abs(vx) < 1e-12:
            raise ValueError("Cannot construct horizontal boundary line")

        return (
            vx,
            vy,
            0.0,
            float(boundary_y),
        )

    def line_y_at_x(line, x):
        """
        Return y coordinate of a line at x.
        """
        vx, vy, x0, y0 = map(float, line)

        if abs(vx) < 1e-12:
            raise ValueError("Vertical line has no unique y(x)")

        return y0 + ((x - x0) / vx) * vy

    def line_angle_deg(line):
        vx, vy, _, _ = line
        return math.degrees(math.atan2(vy, vx))

    def intersect_line_with_horizontal_boundary(line, y):
        """
        Intersect a parametric line

            (x0, y0) + t * (vx, vy)

        with the horizontal line

            y = constant

        Returns (x, y).
        """
        vx, vy, x0, y0 = map(float, line)

        if abs(vy) < 1e-12:
            raise ValueError("Line is parallel to horizontal boundary")

        t = (y - y0) / vy
        x = x0 + t * vx

        return np.array([x, y], dtype=np.float32)

    def transform_points_affine(points, M):
        """
        Transform Nx2 points using a 2x3 OpenCV affine matrix.
        """
        points = np.asarray(points, dtype=np.float32)
        ones = np.ones(
            (points.shape[0], 1),
            dtype=np.float32,
        )
        points_h = np.hstack([
            points,
            ones,
        ])
        transformed = points_h @ M.T
        return transformed

    def save_image(path, image):
        """
        Save image correctly, including alpha-channel TIFFs,
        using lossless TIFF compression.
        """
        global config
        pil_image_suffix = ("." + config.pil_image_save_kwargs["format"].lower())
        assert Path(path).suffix == pil_image_suffix, f"bad path: {path!r}"
        path = str(path)
        if image.ndim == 2:
            # Grayscale
            pil_image = PIL.Image.fromarray(image)
        elif image.shape[2] == 3:
            # OpenCV BGR -> RGB
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = PIL.Image.fromarray(rgb)
        elif image.shape[2] == 4:
            # OpenCV BGRA -> RGBA
            rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            pil_image = PIL.Image.fromarray(rgba)
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        pil_image.save(path, **config.pil_image_save_kwargs)

    def edge_is_white(
        image,
        edge,
        inner_depth=WHITE_TEST_INNER,
        outer_depth=WHITE_TEST_OUTER,
        white_threshold=PRACTICALLY_WHITE_THRESHOLD,
        white_fraction_threshold=WHITE_FRACTION_THRESHOLD,
    ):
        """
        Return True if the region 10..20 pixels inside an edge is
        practically white, indicating that page content probably
        does not reach the page edge.

        The test is based on the fraction of practically-white pixels,
        rather than average brightness, so a small amount of dark content
        is not hidden by averaging.
        """

        h, w = image.shape[:2]

        # Convert to grayscale for the lightness test.
        if image.ndim == 2:
            gray = image
        elif image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        # note: we exclude corners with `inner_depth:-inner_depth`

        # Make sure the requested strip fits.
        if edge == "top":
            if outer_depth > h:
                return False
            # strip = gray[inner_depth:outer_depth, :]
            strip = gray[inner_depth:outer_depth, inner_depth:-inner_depth]

        elif edge == "bottom":
            if outer_depth > h:
                return False
            # strip = gray[h - outer_depth:h - inner_depth, :]
            strip = gray[h - outer_depth:h - inner_depth, inner_depth:-inner_depth]

        elif edge == "left":
            if outer_depth > w:
                return False
            # strip = gray[:, inner_depth:outer_depth]
            strip = gray[inner_depth:-inner_depth, inner_depth:outer_depth]

        elif edge == "right":
            if outer_depth > w:
                return False
            # strip = gray[:, w - outer_depth:w - inner_depth]
            strip = gray[inner_depth:-inner_depth, w - outer_depth:w - inner_depth]

        else:
            raise ValueError(f"Invalid edge: {edge}")

        if strip.size == 0:
            return False

        practically_white = strip >= white_threshold
        white_fraction = np.mean(practically_white)

        if DEBUG:
            print(f"white_fraction: {white_fraction}")

        return white_fraction >= white_fraction_threshold

    def paint_white_edge_border(
        image,
        edge,
        border_width=WHITE_BORDER_WIDTH,
    ):
        """
        Paint a white border inside one edge of the image.
        """

        if edge == "top":
            image[:border_width, :, :] = 255

        elif edge == "bottom":
            image[-border_width:, :, :] = 255

        elif edge == "left":
            image[:, :border_width, :] = 255

        elif edge == "right":
            image[:, -border_width:, :] = 255

        else:
            raise ValueError(f"Invalid edge: {edge}")

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

        if DEBUG: print() # end line of tqdm progress bar

        # 1. load image

        fname = in_path.name
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

        # dbgdir = OUTPUT_DIR.joinpath("debug", f"{page_num:03d}")
        # if DEBUG: ensure_dir(dbgdir)

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

        # NOTE we ignore inside_pts because
        # - guillotine cutter can produce non-orthogonal inside edge of the book
        # - document scanner does remove the top edge of the scan
        # if config.do_rotate is True then
        # "inside edge of the book" == "top edge of the scan"
        #
        # this also means:
        # the widths of the top and bottom edges
        # must have no influence on the deskew step
        # (deskew = Rotation + Perspective transform)

        # Expected horizontal margin between the page and scan boundary.
        expected_horizontal_margin_mm = (
            config.margined_scan_width_mm
            - config.unbinded_page_width_mm
        ) / 2.0
        expected_horizontal_margin_px = int(round(
            expected_horizontal_margin_mm
            * W_img
            / config.margined_scan_width_mm
        ))
        # Search up to 2x the expected margin from the relevant image side.
        vertical_edge_search_size = min(
            W_img,
            max(1, 2 * expected_horizontal_margin_px),
        )

        # if bad_on_left:
        #     outside_pts = detect_edge_points(
        #         gray,
        #         mask,
        #         EDGE_RIGHT,
        #         edge_search_width_px,
        #     )
        # else:
        #     outside_pts = detect_edge_points(
        #         gray,
        #         mask,
        #         EDGE_LEFT,
        #         edge_search_width_px,
        #     )
        outside_edge = EDGE_RIGHT if bad_on_left else EDGE_LEFT

        outside_pts = detect_edge_points(
            gray,
            mask,
            outside_edge,
            vertical_edge_search_size,
        )

        # inside_pts = None
        # if (
        #     not config.use_three_edge_deskew
        #     and config.scan_top_edge == "top"
        # ):
        #     inside_edge = EDGE_LEFT if bad_on_left else EDGE_RIGHT
        #     if DEBUG: print(f"{in_path}: getting inside_pts with detect_edge_points: edge_search_width_px={edge_search_width_px}")
        #     inside_pts = detect_edge_points(
        #         gray,
        #         mask,
        #         inside_edge,
        #         edge_search_width_px,
        #     )
        #     if DEBUG: print(f"{in_path}: len(inside_pts) = {len(inside_pts)}")
        # else:
        #     if DEBUG: print(f"{in_path}: not getting inside_pts")

        inside_pts = None
        if (
            not config.use_three_edge_deskew
            and config.scan_top_edge == "top"
        ):
            inside_edge = EDGE_LEFT if bad_on_left else EDGE_RIGHT
            if DEBUG:
                print(
                    f"{in_path}: expected inside margin="
                    f"{expected_horizontal_margin_px}px, "
                    f"search width={vertical_edge_search_size}px"
                )
            inside_pts = detect_edge_points(
                gray,
                mask,
                inside_edge,
                vertical_edge_search_size,
            )
            if DEBUG: print(f"{in_path}: len(inside_pts) = {len(inside_pts)}")
        else:
            if DEBUG: print(f"{in_path}: not getting inside_pts")

    # The normal detector can fail when the physical page edge is
        # visually weak, but the page mask still gives us its boundary.
        if len(outside_pts) < RANSAC_MIN_INLIERS:
            if DEBUG: print("detect_edge_points failed to detect enough outside_pts -> using fallback")
            ys, xs = np.nonzero(mask > 0)

            if outside_edge == EDGE_LEFT:
                x_edge = xs.min() if len(xs) else None
                if x_edge is not None:
                    fallback_pts = np.column_stack([
                        np.full(len(ys), x_edge),
                        ys,
                    ])
                else:
                    fallback_pts = np.empty((0, 2))
            else:
                x_edge = xs.max() if len(xs) else None
                if x_edge is not None:
                    fallback_pts = np.column_stack([
                        np.full(len(ys), x_edge),
                        ys,
                    ])
                else:
                    fallback_pts = np.empty((0, 2))
            if len(fallback_pts) >= RANSAC_MIN_INLIERS:
                outside_pts = fallback_pts

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

        inside_line = None

        if (
            not config.use_three_edge_deskew
            and config.scan_top_edge == "top"
        ):
            if DEBUG: print(f"{in_path}: getting inside_line with fit_line_ransac_or_none")
            if DEBUG: print(f"{in_path}: len(inside_pts) before reject_outliers_vertical: {len(inside_pts)}")
            inside_pts = reject_outliers_vertical(inside_pts)
            if DEBUG: print(f"{in_path}: len(inside_pts) after reject_outliers_vertical: {len(inside_pts)}")
            inside_line = fit_line_ransac_or_none(inside_pts)
        else:
            if DEBUG: print(f"{in_path}: not getting inside_line")

        # 5. fit lines
        #
        # Normally we have three visible page edges:
        #
        #     top
        #     bottom
        #     outside
        #
        # However, when the page is almost as large as the scanner's maximum
        # scan width, the scanner can crop one of top/bottom completely.
        #
        # In that case a missing horizontal edge is NOT an error.
        # We can still deskew using:
        #
        #     outside + top
        #
        # or:
        #
        #     outside + bottom
        #
        # The missing edge is then represented by the corresponding image
        # boundary later in the perspective transform.

        if DEBUG: print(f"fit_line_ransac_or_none(top_pts) # len(top_pts) = {len(top_pts)}")
        if DEBUG: t1 = time.time()
        top_line = fit_line_ransac_or_none(top_pts)
        if DEBUG: print(f"fit_line_ransac_or_none: done in {time.time() - t1:.3f} seconds")

        if DEBUG: print(f"fit_line_ransac_or_none(bottom_pts) # len(bottom_pts) = {len(bottom_pts)}")
        if DEBUG: t1 = time.time()
        bottom_line = fit_line_ransac_or_none(bottom_pts)
        if DEBUG: print(f"fit_line_ransac_or_none: done in {time.time() - t1:.3f} seconds")

        # FIXME this takes too long because we have too many outside_pts
        # len(inside_pts) = 5277
        # len(bottom_pts) = 3481
        # len(outside_pts) = 17742474
        if DEBUG: print(f"fit_line_ransac_or_none(outside_pts) # len(outside_pts) = {len(outside_pts)}")
        if DEBUG: t1 = time.time()
        outside_line = fit_line_ransac_or_none(outside_pts)
        if DEBUG: print(f"fit_line_ransac_or_none: done in {time.time() - t1:.3f} seconds")

        if 0:
            # make RANSAC less strict
            if DEBUG: print("fit_line_ransac_or_none(outside_pts) # less strict")
            outside_line = fit_line_ransac_or_none(
                outside_pts,
                iterations=RANSAC_ITER,
                inlier_dist=RANSAC_INLIER_DIST * 2,
                min_inliers=10,
            )

        if outside_line is None:
            raise ValueError(
                f"Could not detect outside page edge "
                f"(page {page_num}); "
                f"detected {len(outside_pts)} candidate points"
            )

        if top_line is None and bottom_line is None:
            raise ValueError(
                f"Could not detect either top or bottom page edge "
                f"(page {page_num}); need at least one horizontal edge"
            )

        if DEBUG:
            print(
                f"page {page_num}: detected edges:",
                f"top={'yes' if top_line is not None else 'NO'}",
                f"bottom={'yes' if bottom_line is not None else 'NO'}",
                f"outside={'yes' if outside_line is not None else 'NO'}",
                f"inside={'yes' if inside_line is not None else 'NO'}",
            )

        missing_top = top_line is None
        missing_bottom = bottom_line is None

    if 0:
            # debug
            # float(vx), float(vy), float(x0), float(y0)
            print(f"top_line={top_line}")
            print(f"bottom_line={bottom_line}")
            print(f"outside_line={outside_line}")

    # page margin
        # Intersections with the detected horizontal edges.
        #
        # If one edge is missing, we use the corresponding image boundary
        # as the page edge. This is intentional: the scanner has already
        # cropped that edge, so extrapolating the page would manufacture
        # pixels that do not exist.

        if top_line is not None:
            if DEBUG: print("outside_top = intersect_lines(...)")
            outside_top = intersect_lines(
                outside_line,
                top_line,
            )
        else:
            if DEBUG: print("outside_top = intersect_line_with_horizontal_boundary(...)")
            outside_top = intersect_line_with_horizontal_boundary(
                outside_line,
                0.0,
            )

        if bottom_line is not None:
            if DEBUG: print("outside_bottom = intersect_lines(...)")
            outside_bottom = intersect_lines(
                outside_line,
                bottom_line,
            )
        else:
            if DEBUG: print("outside_bottom = intersect_line_with_horizontal_boundary(...)")
            outside_bottom = intersect_line_with_horizontal_boundary(
                outside_line,
                float(H_img - 1),
            )

    # also calculate inside_top and inside_bottom
        # inside_line is simply the inside edge of the source image
        # y1 = 0; y2 = y_max
        # on odd pages: x1 = x2 = 0 # inside is left
        # on even pages: x1 = x2 = x_max # inside is right
        if bad_on_left:
            # Odd page:
            # binding/inside edge is the LEFT image boundary
            inside_x = 0.0
        else:
            # Even page:
            # binding/inside edge is the RIGHT image boundary
            inside_x = float(W_img - 1)

        if top_line is not None:
            if DEBUG: print("inside_top = intersect_line_with_vertical_boundary(...)")
            inside_top = intersect_line_with_vertical_boundary(
                top_line,
                inside_x,
            )
        else:
            # Missing top page edge: preserve the top image boundary.
            inside_top = np.array(
                [inside_x, 0.0],
                dtype=np.float32,
            )

        if bottom_line is not None:
            if DEBUG: print("inside_bottom = intersect_line_with_vertical_boundary(...)")
            inside_bottom = intersect_line_with_vertical_boundary(
                bottom_line,
                inside_x,
            )
        else:
            # Missing bottom page edge: preserve the bottom image boundary.
            inside_bottom = np.array(
                [inside_x, float(H_img - 1)],
                dtype=np.float32,
            )

    if DEBUG:
            vis = img.copy()
            # margin range
            if bad_on_left:
                # outside edge is right
                # no line on the left
                if 0:
                    # Wr, Hr
                    pts = np.array([
                        [0, edge_search_height_px], # top left
                        [Wr - edge_search_width_px, edge_search_height_px], # top right
                        [Wr - edge_search_width_px, Hr - edge_search_height_px], # bottom right
                        [0, Hr - edge_search_height_px], # bottom left
                    ], np.int32)
                else:
                    # W_img, H_img
                    pts = np.array([
                        [0, edge_search_height_px], # top left
                        [W_img - edge_search_width_px, edge_search_height_px], # top right
                        [W_img - edge_search_width_px, H_img - edge_search_height_px], # bottom right
                        [0, H_img - edge_search_height_px], # bottom left
                    ], np.int32)
            else:
                # outside edge is left
                # no line on the right
                if 0:
                    # Wr, Hr
                    pts = np.array([
                        [Wr, edge_search_height_px], # top right
                        [edge_search_width_px, edge_search_height_px], # top left
                        [edge_search_width_px, Hr - edge_search_height_px], # bottom left
                        [Wr, Hr - edge_search_height_px], # bottom right
                    ], np.int32)
                else:
                    # W_img, H_img
                    pts = np.array([
                        [W_img, edge_search_height_px], # top right
                        [edge_search_width_px, edge_search_height_px], # top left
                        [edge_search_width_px, H_img - edge_search_height_px], # bottom left
                        [W_img, H_img - edge_search_height_px], # bottom right
                    ], np.int32)
            cv2.polylines(vis, [pts], False, (0,255,0), 3) # green

            # Margin range
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
                if bad_on_left:
                    # outside edge is right
                    x1_top = outside_top[0]
                    x1_bottom = outside_bottom[0]
                    # binding edge
                    # find leftmost detected page boundary
                    # x0_top = np.min(top_pts[:,0])
                    # x0_bottom = np.min(bottom_pts[:,0])
                    x0_top = 0
                    x0_bottom = 0
                else:
                    # outside edge is left
                    x0_top = outside_top[0]
                    x0_bottom = outside_bottom[0]
                    # binding edge
                    # find rightmost detected page boundary
                    # x1_top = np.max(top_pts[:,0])
                    # x1_bottom = np.max(bottom_pts[:,0])
                    if 0:
                        # Wr, Hr
                        x1_top = Wr - 1
                        x1_bottom = Wr - 1
                    else:
                        # W_img, H_img
                        x1_top = W_img - 1
                        x1_bottom = W_img - 1

            # Detected page quadrilateral
            if 0:
                pts = np.array([
                    # top left: bad: y=outside_top[1] is wrong
                    # this would have to be min(top_line) or so
                    [x0_top, outside_top[1]], # top left
                    [x1_top, outside_top[1]], # top right: good
                    [x1_bottom, outside_bottom[1]], # bottom right: good
                    [x0_bottom, outside_bottom[1]], # bottom left: bad: y=outside_bottom[1] is wrong
                ], np.int32)
            elif 0:
                pts = np.array([
                    [x0_top, outside_top[1]],
                    [x1_top, outside_top[1]],
                    [x1_bottom, outside_bottom[1]],
                    [x0_bottom, outside_bottom[1]],
                ], np.int32)
            elif 1:
                pts = np.round(np.array([
                    inside_top,
                    outside_top,
                    outside_bottom,
                    inside_bottom,
                ], dtype=np.float32)).astype(np.int32)
            cv2.polylines(vis, [pts], True, (0,0,255), 3) # red

            name = OUTPUT_DIR / f"{page_num:03d}.line-0900-fit-lines-ransac.tiff"
            # print(f"writing {name}")
            save_image(name, vis)

        src_corners = np.float32([
            inside_top,
            outside_top,
            outside_bottom,
            inside_bottom,
        ])

    # FIXME merge old code with new code

    # # TODO remove

        # gray, mask, contours = get_gray_mask_contours(img, dbgdir)

        # if not contours:
        #     print(f"Warning: no contours found in {in_path}")
        #     return

        # page_contour = max(contours, key=cv2.contourArea)

        if config.use_three_edge_deskew:

            if DEBUG: print("config.use_three_edge_deskew == True")

            # top_pts, bottom_pts, outside_pts = split_edge_candidates(
            #     page_contour,
            #     bad_on_left
            # )

            # top_line = fit_line_ransac(top_pts)[:4]
            # bottom_line = fit_line_ransac(bottom_pts)[:4]
            # outside_line = fit_line_ransac(outside_pts)[:4]

    # 6. rotate
            # deskew part 1

            # old
            # top_angle = math.degrees(line_angle(top_line))
            # bottom_angle = math.degrees(line_angle(bottom_line))
            # outside_angle = math.degrees(line_angle(outside_line))

            if DEBUG:
                # start debug prints
                print()
                print(f"line 570: page_num={page_num}")

            top_angle = (
                horizontal_line_angle(top_line)
                if top_line is not None
                else None
            )

            bottom_angle = (
                horizontal_line_angle(bottom_line)
                if bottom_line is not None
                else None
            )

            outside_angle = vertical_line_angle(outside_line)

            rotation_errors = [
                outside_angle - 90.0,
            ]

            if top_angle is not None:
                rotation_errors.append(top_angle)

            if bottom_angle is not None:
                rotation_errors.append(bottom_angle)

            rotation_error = float(np.mean(rotation_errors))

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

            if remove_inside_transparent_strip:
                # remove the extra inside edge
                # rotate with gray background
                rotated = cv2.warpAffine(
                    img,
                    Mrot,
                    (W_img, H_img),
                    # borderValue=(255,255,255) # white
                    borderValue=(128,128,128) # gray
                )
            else:
                # keep the extra inside edge
                # rotate with transparent background
                # add alpha channel
                img_bgra = cv2.cvtColor(
                    img,
                    cv2.COLOR_BGR2BGRA,
                )
                # init alpha channel
                # img_bgra[:, :, 3] = 255
                rotated_bgra = cv2.warpAffine(
                    img_bgra,
                    Mrot,
                    (W_img, H_img),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0, 0),
                )
                rotated = rotated_bgra

            Hr, Wr = rotated.shape[:2]

    # also rotate the detected edges

            rotated_corners = transform_points_affine(
                src_corners,
                Mrot,
            )

            inside_top = rotated_corners[0]
            outside_top = rotated_corners[1]
            outside_bottom = rotated_corners[2]
            inside_bottom = rotated_corners[3]

            # remember the inside edge for later cropping
            actual_inside_top = rotated_corners[0].copy()
            actual_inside_bottom = rotated_corners[3].copy()

            # ignore the inside edge for perspective transform
            # Ignore the unknown inside-edge angle for perspective correction.
            # Use the image boundary as the inside edge.
            if bad_on_left:
                # outside edge is the RIGHT edge
                # inside edge is the LEFT edge
                inside_top[0] = 0
                inside_bottom[0] = 0
            else:
                # outside edge is the LEFT edge
                # inside edge is the RIGHT edge
                inside_top[0] = Wr - 1
                inside_bottom[0] = Wr - 1

            # don't use the missing edge as though it were detected after rotation
            # (inside_top, outside_top, outside_bottom, inside_bottom)
            # contain image-boundary points when an edge is missing.
            # but the perspective transform should distinguish the two cases
            if missing_top:
                # The top edge was cropped by the scanner.
                #
                # Preserve the rotated image's top boundary instead of trying
                # to invent the missing page edge.
                inside_top = np.array(
                    [inside_top[0], 0.0],
                    dtype=np.float32,
                )
                outside_top = np.array(
                    [outside_top[0], 0.0],
                    dtype=np.float32,
                )
            if missing_bottom:
                # The bottom edge was cropped by the scanner.
                #
                # Preserve the rotated image's bottom boundary.
                inside_bottom = np.array(
                    [inside_bottom[0], float(Hr - 1)],
                    dtype=np.float32,
                )
                outside_bottom = np.array(
                    [outside_bottom[0], float(Hr - 1)],
                    dtype=np.float32,
                )

            top_width = np.linalg.norm(
                outside_top - inside_top
            )
            bottom_width = np.linalg.norm(
                outside_bottom - inside_bottom
            )
            left_height = np.linalg.norm(
                inside_bottom - inside_top
            )
            right_height = np.linalg.norm(
                outside_bottom - outside_top
            )

            # expected size after rotation
            expected_w = int(round(
                (top_width + bottom_width) / 2.0
            ))
            expected_h = int(round(
                (left_height + right_height) / 2.0
            ))
            if 0:
                # Or, if you want the height to be determined specifically by the outside edge
                # The average is usually more stable, though.
                expected_h = int(round(
                    right_height
                ))

            if DEBUG:
                print("line 580: rotated size", Wr, Hr)

            # img = rotated # ?

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

            # # no. this fails to reconstruct the page height...
            # # TODO try to solve this with the average height of multiple pages
            # # assuming all pages must have the same height
            # # also allowing the user to specify a scale_y factor
            # if 0:
            #     if config.do_rotate == False:
            #         # fix scan height
            #         # ...
            #         pass

            expected_h = int(round(page_height))

            # # ASPECT = x / y
            # # x = y * ASPECT

            # # expand the binding edge to expected_w
            # # expected_w = int(round(expected_h * ASPECT))
            # expected_w = int(round(page_height * ASPECT))

            if DEBUG:
                vis = rotated.copy()
                pts_rotated = np.round(
                    rotated_corners
                ).astype(np.int32)
                cv2.polylines(
                    vis,
                    [pts_rotated],
                    True,
                    (0, 0, 255), # red
                    3,
                )
                name = OUTPUT_DIR / (f"{page_num:03d}.line-1200-rotated-corners.tiff")
                save_image(str(name), vis)

    # 7b. cleanup the inside edge
            # use outmost_inside_x to remove the half-transparent inside edge
            # if remove_inside_transparent_strip:
            if 0:
                if bad_on_left:
                    # odd page
                    # Inside edge is on the LEFT.
                    # Remove everything from x=0 through outmost_inside_x.
                    crop_x = int(math.ceil(outmost_inside_x))
                    rotated = rotated[:, crop_x:, :]

                else:
                    # even page
                    # Inside edge is on the RIGHT.
                    # Remove everything from outmost_inside_x through the right edge.
                    crop_x = int(math.floor(outmost_inside_x))
                    rotated = rotated[:, :crop_x, :]

    if DEBUG:
                print(
                    f"page {page_num}: geometry mode = "
                    f"{'3-edge' if not (missing_top or missing_bottom) else '2-edge'}"
                )

                if missing_top:
                    print(
                        f"page {page_num}: top edge missing; "
                        f"preserving image top boundary"
                    )

                if missing_bottom:
                    print(
                        f"page {page_num}: bottom edge missing; "
                        f"preserving image bottom boundary"
                    )

    # 8. perspective transform
            # deskew part 2

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
                    # binding edge
                    # find leftmost detected page boundary
                    # x0_top = np.min(top_pts[:,0])
                    # x0_bottom = np.min(bottom_pts[:,0])
                    x0_top = 0
                    x0_bottom = 0
                else:
                    # outside edge is left
                    x0_top = outside_top[0]
                    x0_bottom = outside_bottom[0]
                    # binding edge
                    # find rightmost detected page boundary
                    # x1_top = np.max(top_pts[:,0])
                    # x1_bottom = np.max(bottom_pts[:,0])
                    x1_top = Wr - 1
                    x1_bottom = Wr - 1

                # dont expand the binding edge to expected_w
                expected_w = int(round(
                    math.dist((x0_top, outside_top[1]), (x1_top, outside_top[1]))
                ))

            if bad_on_left:
                # inside = LEFT
                # outside = RIGHT
                src = np.float32([
                    inside_top,       # TL
                    outside_top,      # TR
                    outside_bottom,   # BR
                    inside_bottom,    # BL
                ])
            else:
                # outside = LEFT
                # inside = RIGHT
                src = np.float32([
                    outside_top,      # TL
                    inside_top,       # TR
                    inside_bottom,    # BR
                    outside_bottom,   # BL
                ])

            dst = np.float32([
                [0, 0],
                [expected_w - 1, 0],
                [expected_w - 1, expected_h - 1],
                [0, expected_h - 1],
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

            actual_inside_warped = cv2.perspectiveTransform(
                np.array([[
                    actual_inside_top,
                    actual_inside_bottom,
                ]], dtype=np.float32),
                M,
            )[0]

            actual_inside_top_warped = actual_inside_warped[0]
            actual_inside_bottom_warped = actual_inside_warped[1]

            inside_top_x = actual_inside_top_warped[0]
            inside_bottom_x = actual_inside_bottom_warped[0]
            if bad_on_left:
                # odd page
                # inside edge is the LEFT edge
                outmost_inside_x = max(inside_top_x, inside_bottom_x)
            else:
                # even page
                # inside edge is the RIGHT edge
                outmost_inside_x = min(inside_top_x, inside_bottom_x)

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

    # 8b. cleanup the inside edge
            # use outmost_inside_x to remove the half-transparent inside edge
            if remove_inside_transparent_strip:
                if bad_on_left:
                    # Inside edge is on the LEFT.
                    # Remove everything from x=0 through outmost_inside_x.
                    crop_x = int(math.ceil(outmost_inside_x))
                    warped = warped[:, crop_x:, :]

                else:
                    # Inside edge is on the RIGHT.
                    # Remove everything from outmost_inside_x through the right edge.
                    crop_x = int(math.floor(outmost_inside_x))
                    warped = warped[:, :crop_x, :]

    # 8c. remove small gray artifacts along page edges
            #
            # The perspective transform can leave thin light-gray lines
            # along the page edges because the detected page boundary
            # is not perfectly straight.
            #
            # However, do NOT paint over page content in borderless printing.
            #
            # For each edge:
            #   - inspect a strip 20..10 pixels inside the page
            #   - if >= 99% of pixels are practically white,
            #     assume there is no content near that edge
            #   - paint a 10-pixel white border inside that edge
            #
            # The decision is made independently for each edge.

            # Apply independently to all four edges.
            for edge in ("top", "bottom", "left", "right"):

                if not edge_is_white(warped, edge):
                    # Content reaches the page edge.
                    # Do not destroy it.
                    if DEBUG:
                        print(f"white edge cleanup: {edge}: borderless -> KEEP CONTENT")
                else:
                    # The 10..20 px interior strip is practically white,
                    # so it is safe to remove the gray artifact at the edge.
                    paint_white_edge_border(warped, edge, WHITE_BORDER_WIDTH)

                    if DEBUG:
                        print(f"white edge cleanup: {edge}: white margin -> PAINT {WHITE_BORDER_WIDTH}px")

    else:
            # config.use_three_edge_deskew == False
            #
            # two-edge deskew
            #
            # The book is too tall for `scan_top_edge = "inside"`,
            # so we have to use `scan_top_edge = "top"`,
            # so the top page edge is missing.
            # Use only the outside edge and bottom edge.
            #
            # `scan_top_edge == "top"` means:
            # the image top is the missing page edge.

            if DEBUG: print("config.use_three_edge_deskew == False")

            if config.scan_top_edge != "top":
                raise NotImplementedError("two-edge deskew currently supports only scan_top_edge='top'")

            if outside_line is None:
                raise RuntimeError(f"not found outside edge of {in_path}: {len(outside_pts)} candidate points")

            if bottom_line is None:
                raise RuntimeError(f"not found bottom edge of {in_path}: {len(bottom_pts)} candidate points")

            if inside_line is None:
                raise RuntimeError(f"not found inside edge of {in_path}: {len(inside_pts)} candidate points")

            # Use the two detected edges to estimate rotation.
            outside_angle = vertical_line_angle(outside_line)
            bottom_angle = horizontal_line_angle(bottom_line)

            rotation_error = float(
                np.mean([
                    outside_angle - 90.0,
                    bottom_angle,
                ])
            )

            Mrot = cv2.getRotationMatrix2D(
                (W_img / 2.0, H_img / 2.0),
                rotation_error,
                1.0,
            )

            rotated = cv2.warpAffine(
                img,
                Mrot,
                (W_img, H_img),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255),
            )

            # # Transform the two useful corners.
            # outside_bottom = intersect_lines(outside_line, bottom_line)

            # p = np.array(
            #     [outside_bottom[0], outside_bottom[1], 1.0],
            #     dtype=np.float64,
            # )
            # outside_bottom_rot = Mrot @ p
            # outside_bottom_rot = outside_bottom_rot[:2]

            # # # The missing top edge is the top of the rotated scan.
            # # inside_x = 0.0 if bad_on_left else float(W_img - 1)

            # # FIXME patch the crop geometry

            # inside_top = np.array([inside_x, 0.0], dtype=np.float64)
            # outside_top = np.array(
            #     [outside_bottom_rot[0], 0.0],
            #     dtype=np.float64,
            # )
            # inside_bottom = np.array(
            #     [inside_x, outside_bottom_rot[1]],
            #     dtype=np.float64,
            # )

            # # Page dimensions.
            # expected_w = int(round(
            #     abs(outside_bottom_rot[0] - inside_x)
            # ))
            # expected_h = int(round(
            #     abs(outside_bottom_rot[1])
            # ))

            # if expected_w <= 0 or expected_h <= 0:
            #     raise RuntimeError(
            #         f"invalid two-edge page size: {expected_w}x{expected_h}"
            #     )

            # src_corners = np.array([
            #     inside_top,
            #     outside_top,
            #     outside_bottom_rot,
            #     inside_bottom,
            # ], dtype=np.float32)

            # if not bad_on_left:
            #     src_corners = np.array([
            #         outside_top,
            #         inside_top,
            #         inside_bottom,
            #         outside_bottom_rot,
            #     ], dtype=np.float32)

            # dst_corners = np.array([
            #     [0, 0],
            #     [expected_w - 1, 0],
            #     [expected_w - 1, expected_h - 1],
            #     [0, expected_h - 1],
            # ], dtype=np.float32)

            # M = cv2.getPerspectiveTransform(src_corners, dst_corners)

            # warped = cv2.warpPerspective(
            #     rotated,
            #     M,
            #     (expected_w, expected_h),
            #     flags=cv2.INTER_LINEAR,
            #     borderMode=cv2.BORDER_CONSTANT,
            #     borderValue=(255, 255, 255),
            # )

            # Transform the detected lines into the rotated image.
            # OpenCV's affine transform maps points from the original image
            # into the rotated image.

            # def transform_point(point, M):
            #     p = np.array([point[0], point[1], 1.0], dtype=np.float64)
            #     return M @ p

            # def transform_line(line, M):
            #     # Represent the line as a homogeneous vector.
            #     # This assumes intersect_lines() uses lines represented by
            #     # two endpoints: ((x1, y1), (x2, y2)).
            #     p1, p2 = line
            #     p1 = transform_point(p1, M)
            #     p2 = transform_point(p2, M)

            #     p1 = np.asarray(p1, dtype=np.float64)
            #     p2 = np.asarray(p2, dtype=np.float64)

            #     return (p1, p2)

            # # Transform representative points on each detected line.
            # outside_p1, outside_p2 = transform_line(outside_line, Mrot)
            # inside_p1, inside_p2 = transform_line(inside_line, Mrot)
            # bottom_p1, bottom_p2 = transform_line(bottom_line, Mrot)

            # def line_coefficients(p1, p2):
            #     # Return a*x + b*y + c = 0.
            #     a = p1[1] - p2[1]
            #     b = p2[0] - p1[0]
            #     c = p1[0] * p2[1] - p2[0] * p1[1]
            #     return np.array([a, b, c], dtype=np.float64)

            # outside_rot = line_coefficients(outside_p1, outside_p2)
            # inside_rot = line_coefficients(inside_p1, inside_p2)
            # bottom_rot = line_coefficients(bottom_p1, bottom_p2)

            def transform_point(point, M):
                p = np.array(
                    [point[0], point[1], 1.0],
                    dtype=np.float64,
                )
                return M @ p

    def transform_line(line, M):
                vx, vy, x0, y0 = map(float, line)

                # Two points on the original fitted line.
                p1 = np.array([x0 - vx, y0 - vy], dtype=np.float64)
                p2 = np.array([x0 + vx, y0 + vy], dtype=np.float64)

                # Transform both points into rotated-image coordinates.
                p1_rot = transform_point(p1, M)
                p2_rot = transform_point(p2, M)

                return line_coefficients(p1_rot, p2_rot)

    def line_coefficients(p1, p2):
                # Return homogeneous line coefficients a*x + b*y + c = 0.
                a = p1[1] - p2[1]
                b = p2[0] - p1[0]
                c = p1[0] * p2[1] - p2[0] * p1[1]
                return np.array([a, b, c], dtype=np.float64)

    outside_rot = transform_line(outside_line, Mrot)
            inside_rot = transform_line(inside_line, Mrot)
            bottom_rot = transform_line(bottom_line, Mrot)

            def intersect_coefficients(line1, line2):
                a1, b1, c1 = line1
                a2, b2, c2 = line2

                det = a1 * b2 - a2 * b1
                if abs(det) < 1e-10:
                    raise RuntimeError("Cannot intersect nearly parallel crop edges")

                x = (b1 * c2 - b2 * c1) / det
                y = (c1 * a2 - c2 * a1) / det
                return np.array([x, y], dtype=np.float64)

            # The missing top edge is the top of the rotated scan.
            top_boundary = np.array([0.0, 1.0, 0.0], dtype=np.float64)

            inside_top = intersect_coefficients(inside_rot, top_boundary)
            outside_top = intersect_coefficients(outside_rot, top_boundary)

            inside_bottom = intersect_coefficients(inside_rot, bottom_rot)
            outside_bottom = intersect_coefficients(outside_rot, bottom_rot)

            # Order the corners consistently around the page:
            # top-left, top-right, bottom-right, bottom-left.
            if bad_on_left:
                corners = [
                    inside_top,
                    outside_top,
                    outside_bottom,
                    inside_bottom,
                ]
            else:
                corners = [
                    outside_top,
                    inside_top,
                    inside_bottom,
                    outside_bottom,
                ]

            src_corners = np.asarray(corners, dtype=np.float32)

            # Crop dimensions from the distances between the corners.
            top_w = np.linalg.norm(src_corners[1] - src_corners[0])
            bottom_w = np.linalg.norm(src_corners[2] - src_corners[3])
            left_h = np.linalg.norm(src_corners[3] - src_corners[0])
            right_h = np.linalg.norm(src_corners[2] - src_corners[1])

            expected_w = int(round(max(top_w, bottom_w)))
            expected_h = int(round(max(left_h, right_h)))

            if expected_w <= 0 or expected_h <= 0:
                raise RuntimeError(
                    f"invalid two-edge page size: {expected_w}x{expected_h}"
                )

            dst_corners = np.array([
                [0, 0],
                [expected_w - 1, 0],
                [expected_w - 1, expected_h - 1],
                [0, expected_h - 1],
            ], dtype=np.float32)

            M = cv2.getPerspectiveTransform(src_corners, dst_corners)

            warped = cv2.warpPerspective(
                rotated,
                M,
                (expected_w, expected_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255),
            )

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

        ensure_dir(out_path.parent)
        # cv2.imwrite(out_path, canvas, config.cv2_imwrite_params)
        save_image(out_path, canvas)
        # print(f"writing {out_path}")

    def _process_file(in_path):
        # page_num = get_page_num(in_path)

        out_path = OUTPUT_DIR / in_path.name

        process_image(in_path, out_path)

    def main():
        ensure_dir(OUTPUT_DIR)

        files = sorted(INPUT_DIR.glob(f"*.{config.scan_format}"))

        if not files:
            print("No image files found in", INPUT_DIR)
            return

        # Only submit work for files that still need processing.
        files = remove_done_files(files, OUTPUT_DIR)

        if 0:
            # debug: process only some pages
            def filter_file(file):
                page_num = get_page_num(file)
                if not page_num in (1, 2, 3):
                    return False
                return True
            files = list(filter(filter_file, files))

        if not files:
            print("nothing to do")
            return

        dst = OUTPUT_DIR
        content_files = []
        extra_files = []
        for f in files:
            page_num = get_page_num(f)
            if 1 <= page_num <= config.num_pages:
                # process content pages
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
        # process only content files
        files = content_files

        num_workers = psutil.cpu_count(logical=False) or 1

        # by default, OpenCV uses multiple CPU threads
        cv2.setNumThreads(1)

        if DEBUG:
            # debug: disable parallel processing
            print("num_workers = 1")
            num_workers = 1

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_process_file, fname): fname
                for fname in files
            }

            tqdm_kwargs = dict(
                total=len(files),
                ncols=80,
                unit="page",
            )

            with tqdm(**tqdm_kwargs) as pbar:
                for future in as_completed(futures):
                    fname = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        print() # end line of tqdm progress bar
                        print(f"Error processing {fname}: {type(exc).__name__}: {exc}")
                        # FIXME this fails to kill workers
                        # kill all other workers to raise the exception immediately
                        # executor.kill_workers()
                        executor.terminate_workers()
                        raise
                        break # unreachable?
                    finally:
                        pbar.update(1)

    if __name__ == "__main__":
        main()

## Response:

10/11/2026, 9:28:43 AM

> **Planning edge detection**
>
> I’m designing a function that supports grayscale, BGR, and BGRA images. I’m scanning inward from each edge for a narrow mid-gray strip followed by a sustained white region, which should distinguish crop borders from ordinary content.
>
> **Planned edge detection**
>
> Worked for 10s

I’d implement this as a separate cleanup step in `065-remove-page-borders.py`, immediately after the perspective transform and before the final image is saved.

The important distinction is that your existing `edge_is_white()` / `paint_white_edge_border()` logic paints a fixed-width border when the interior is almost entirely white. Your new feature needs to be more selective: **detect the actual gray strip, follow its potentially curved inner boundary, and whiten only that strip plus a configurable margin.**

I would keep the existing straight-line edge detection and perspective correction unchanged. The new cleanup can operate on the resulting `warped` image, where the page edges should already be approximately horizontal and vertical.

## Proposed detection strategy

For each of the four page edges:

1.  Sample pixel intensity along scanlines perpendicular to that edge.
2.  Look for a sequence of midgray pixels followed by white pixels further inside the page.
3.  Require a sufficiently white interior region before accepting a candidate, so ordinary dark text and graphics are not mistaken for scanner background.
4.  Estimate the gray strip's inner boundary independently for each scanline. This allows the detected boundary to curve.
5.  Fill the strip and an adjustable number of pixels on its inner side with white, without painting a fixed-width border across the entire page.

For example, the intensity profile going inward from the page edge might look like this:

<box border radius="lg" padding={3} gap={3}>
  <row align="center" gap={1}>
    <box background="#999999" radius="sm" padding={2} flex={1} align="center">
      <text color="#FFFFFF" weight="medium" size="sm">Gray strip</text>
    </box>
    <box background="#FFFFFF" border={{size:1,color:"#BBBBBB"}} radius="sm" padding={2} flex={1} align="center">
      <text color="#222222" weight="medium" size="sm">White page</text>
    </box>
    <box background="#222222" radius="sm" padding={2} flex={1} align="center">
      <text color="#FFFFFF" weight="medium" size="sm">Text / graphics</text>
    </box>
  </row>
  <text color="secondary" size="sm">Schematic: the gray strip is removed only where a sufficiently white region follows it. The boundary may vary from one scanline to the next.</text>
</box>

One safety consideration: white page margins and black text can also occur next to gray artifacts. The algorithm should therefore use the gray-to-white transition as well as the interior-white test, and it should leave the image unchanged when it cannot confidently identify a strip.

## 1. Add tuning parameters

Add these alongside your existing `WHITE_BORDER_WIDTH`, `WHITE_TEST_INNER`, etc.

<CodeBlock language="python" editable>
# 8d. remove curved midgray scanner artifacts

CLEAN_GRAY_EDGE_ARTIFACTS = True

GRAY_STRIP_MIN = 70  
GRAY_STRIP_MAX = 205

GRAY_STRIP_MAX_WIDTH = 25  
GRAY_STRIP_SEARCH_DEPTH = 80

GRAY_STRIP_WHITE_THRESHOLD = 245  
GRAY_STRIP_WHITE_RUN = 8

# Extra pixels to whiten on the inside of a detected gray strip.

GRAY_STRIP_CLEANUP_MARGIN = 10

# Require this many pixels along the edge to have a detected strip.

GRAY_STRIP_MIN_EDGE_FRACTION = 0.05  
</CodeBlock>

These are starting values to tune against your scans. The gray thresholds assume 8-bit grayscale images. The search depth and maximum strip width are deliberately limited to avoid accidentally removing content far inside the page.

## 2. Add the cleanup function

Place this function after `paint_white_edge_border()`.

The function supports grayscale, BGR, and BGRA images. It scans inward from each edge, finds a narrow midgray run followed by a sufficiently long white run, and whitens that run plus the configured inner margin.

<CodeBlock language="python" editable>
def remove_gray_edge_artifacts(
    image,
    edges=("top", "bottom", "left", "right"),
    gray_min=GRAY_STRIP_MIN,
    gray_max=GRAY_STRIP_MAX,
    max_strip_width=GRAY_STRIP_MAX_WIDTH,
    search_depth=GRAY_STRIP_SEARCH_DEPTH,
    white_threshold=GRAY_STRIP_WHITE_THRESHOLD,
    white_run=GRAY_STRIP_WHITE_RUN,
    cleanup_margin=GRAY_STRIP_CLEANUP_MARGIN,
    min_edge_fraction=GRAY_STRIP_MIN_EDGE_FRACTION,
):
    """
    Remove narrow midgray strips followed by white page background.

    Each scanline is analyzed independently, so the detected boundary
    may curve along the edge.

    Only gray runs followed by a sufficiently long white run qualify.
    The detected gray run and cleanup_margin pixels on its inner side
    are painted white.

    The image is modified in place and returned.
    """

    if image.ndim == 2:
        gray = image
    elif image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    h, w = gray.shape

    # Boolean mask of pixels to whiten.
    cleanup_mask = np.zeros((h, w), dtype=bool)

    def find_gray_run(profile):
        """
        Find a narrow gray run followed by a white run.

        profile[0] is nearest the image edge. The profile proceeds
        inward into the page.
        """
        limit = min(len(profile), search_depth)

        for start in range(limit):
            # Skip non-gray pixels.
            if not (gray_min <= profile[start] <= gray_max):
                continue

            end = start + 1

            while (
                end < limit
                and gray_min <= profile[end] <= gray_max
            ):
                end += 1

            strip_width = end - start

            if strip_width > max_strip_width:
                # This is probably not a thin scanner artifact.
                continue

            if end + white_run > limit:
                continue

            # The strip must be followed by consecutive white pixels.
            following = profile[end:end + white_run]

            if not np.all(following >= white_threshold):
                continue

            # Do not accept a gray patch surrounded by other gray
            # pixels on its outer side, when that outer side exists.
            # A gray strip may legitimately start at the image edge.
            if start > 0:
                preceding = profile[start - 1]
                if gray_min <= preceding <= gray_max:
                    continue

            return start, end

        return None

    # Analyze every scanline perpendicular to the requested edge.
    for edge in edges:
        edge_hits = 0
        edge_length = h if edge in ("left", "right") else w

        for i in range(edge_length):
            if edge == "left":
                profile = gray[i, :]
            elif edge == "right":
                profile = gray[i, ::-1]
            elif edge == "top":
                profile = gray[:, i]
            elif edge == "bottom":
                profile = gray[::-1, i]
            else:
                raise ValueError(f"Invalid edge: {edge}")

            result = find_gray_run(profile)

            if result is None:
                continue

            start, end = result

            # Include the gray strip and an inner cleanup margin.
            clean_end = min(
                end + cleanup_margin,
                search_depth,
                len(profile),
            )

            if edge == "left":
                cleanup_mask[i, start:clean_end] = True
            elif edge == "right":
                cleanup_mask[i, w-clean_end:w-start] = True
            elif edge == "top":
                cleanup_mask[start:clean_end, i] = True
            else:  # bottom
                cleanup_mask[h-clean_end:h-start, i] = True

            edge_hits += 1

        # Avoid applying an edge cleanup based on a few isolated
        # scanlines. If the feature is not sufficiently widespread,
        # discard its detections for this edge.
        if edge_hits < max(1, int(edge_length * min_edge_fraction)):
            if DEBUG:
                print(
                    f"gray edge cleanup: {edge}: "
                    f"only {edge_hits}/{edge_length} hits; skipped"
                )

            if edge == "left":
                cleanup_mask[:, :search_depth] = False
            elif edge == "right":
                cleanup_mask[:, max(0, w-search_depth):] = False
            elif edge == "top":
                cleanup_mask[:search_depth, :] = False
            else:
                cleanup_mask[max(0, h-search_depth):, :] = False

            continue

        if DEBUG:
            print(
                f"gray edge cleanup: {edge}: "
                f"{edge_hits}/{edge_length} scanlines detected"
            )

    # Paint the detected regions white. Preserve the alpha channel
    # of BGRA images.
    if image.ndim == 2:
        image[cleanup_mask] = 255
    else:
        image[cleanup_mask, :3] = 255

    return image

</CodeBlock>

### A small implementation detail

This first version uses a per-scanline search, rather than assuming the gray line is straight. That is the key property you need for curved edges.

The minimum edge fraction is a safeguard against a few coincidental gray pixels being interpreted as an entire edge artifact. You can reduce it if the strip is present along only a small part of the page.

## 3. Call it in `process_image()`

I recommend inserting the call **after your existing 8c cleanup loop**, just before the end of the `if config.use_three_edge_deskew:` branch.

This keeps the current deskew, perspective transform, and existing white-border cleanup unchanged.

**Important:** your current 8c cleanup can paint a full 10-pixel border when the interior is sufficiently white. If that already removes the gray strip, the new step won't have anything left to detect. If you want the curved-strip detector to be solely responsible for this problem, consider disabling the existing 8c border painting for the affected edges, rather than running both indiscriminately.

## 4. Fine-tuning

| Parameter | What to change |
|----|----|
| `GRAY_STRIP_MIN` / `MAX` | Widen the range if the scanner gray is lighter or darker than expected. |
| `GRAY_STRIP_MAX_WIDTH` | Increase if the gray strip is wider than 25 pixels. |
| `GRAY_STRIP_SEARCH_DEPTH` | Increase if the curved line can be farther inside the cropped page. |
| `GRAY_STRIP_WHITE_THRESHOLD` | Lower slightly if the paper is off-white. |
| `GRAY_STRIP_WHITE_RUN` | Increase to demand a longer stretch of white after the strip. |
| `GRAY_STRIP_CLEANUP_MARGIN` | Set to 10 initially; increase to whiten more pixels on the inside of the detected line. |
| `GRAY_STRIP_MIN_EDGE_FRACTION` | Lower if the artifact is visible along only a short section of an edge. |

One caveat with this initial implementation: the minimum-edge-fraction fallback clears detections within the edge's search band, which can also clear overlapping detections belonging to another edge near a corner. If corner behavior matters for your scans, I would refine that by keeping separate cleanup masks for each edge.

## 5. How I would validate it

Start with a small set of pages that includes:

- A clearly curved gray strip along one edge.
- A page with no gray artifacts.
- A borderless page where text or graphics reach the edge.
- A page with off-white paper or gray illustrations near the margin.

I would also add a debug image that overlays the detected cleanup mask in a bright color before applying it. That makes it easy to confirm that the detector follows the curved strip instead of erasing page content.

The main thing to validate is the **false-positive rate**: it is better to leave an uncertain gray line untouched than to whiten actual text or graphics.

------------------------------------------------------------------------

Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)

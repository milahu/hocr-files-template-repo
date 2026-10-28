#!/usr/bin/env python3

"""
IT8 scanner calibration for book-scanning pipelines.

Calibration:
    python 012-calibrate-it8.py target.png R190823.txt \
        --calibrate scanner-it8.json --debug

Apply:
    python 012-calibrate-it8.py book-scan.png corrected.png \
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


def sample_patch_median(image, center_x, center_y, patch_width, patch_height, fraction=0.50):
    """
    Return the median RGB value near the patch center.

    Sampling the central 50% of each cell helps avoid borders,
    neighboring colors, and small alignment errors.
    """
    h, w = image.shape[:2]

    radius_x = max(1, int(patch_width * fraction / 2))
    radius_y = max(1, int(patch_height * fraction / 2))

    x0 = max(0, int(round(center_x)) - radius_x)
    x1 = min(w, int(round(center_x)) + radius_x + 1)
    y0 = max(0, int(round(center_y)) - radius_y)
    y1 = min(h, int(round(center_y)) + radius_y + 1)

    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid patch region centered at ({center_x}, {center_y}).")

    pixels = image[y0:y1, x0:x1].reshape(-1, 3)

    # OpenCV loads BGR; store measurements in RGB order.
    rgb = pixels[:, ::-1].astype(np.float64)

    return np.median(rgb, axis=0)


def sample_patch_mean(image, center_x, center_y, patch_width, patch_height):
    """
    Average RGB values from the central 50% x 50% of a patch.

    center_x, center_y: center of the patch in image coordinates.
    patch_width, patch_height: estimated full patch dimensions.
    """
    half_w = patch_width * 0.25
    half_h = patch_height * 0.25

    x0 = int(round(center_x - half_w))
    x1 = int(round(center_x + half_w))
    y0 = int(round(center_y - half_h))
    y1 = int(round(center_y + half_h))

    # Clip coordinates to the image boundaries.
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(image.shape[1], x1)
    y1 = min(image.shape[0], y1)

    if x0 >= x1 or y0 >= y1:
        raise ValueError("Patch sampling region is empty or outside the image.")

    # OpenCV loads images in BGR order.
    pixels = image[y0:y1, x0:x1].astype("float64")
    mean_bgr = pixels.mean(axis=(0, 1))

    # Return RGB order.
    return mean_bgr[::-1]


# NOTE the median function would be better to ignore outliers
# but here we expect the user to manually cleanup the scanned image
# for example in GIMP with the "clone" tool
# by cloning a good region of a color patch to replace bad regions of the same color patch
# so all color patches are free from visual defects
# (scratches, dust, ...)
# so the mean function returns the correct color of the color patch
sample_patch = sample_patch_mean


def measure_patches(image, reference, grid, debug=False):
    """Measure all reference patches at their expected grid positions."""
    h, w = image.shape[:2]

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
        center_x = x0 + (col + 0.5) * cell_w
        center_y = y0 + (row + 0.5) * cell_h

        rgb = sample_patch(image, center_x, center_y, cell_w, cell_h)

        measured[sample_id] = rgb
        expected[sample_id] = np.asarray(reference[sample_id])

        if debug:
            cv2.rectangle(
                debug_image,
                (int(center_x - cell_w * 0.2), int(center_y - cell_h * 0.2)),
                (int(center_x + cell_w * 0.2), int(center_y + cell_h * 0.2)),
                (0, 255, 0),
                2,
            )
            cv2.putText(
                debug_image,
                sample_id,
                (int(center_x - 15), int(center_y)),
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
            center_x = x0 + (col + 0.5) * cell_w
            center_y = y0 + (row + 0.5) * cell_h
            rgb_values.append(
                sample_patch(image, center_x, center_y, cell_w, cell_h)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="IT8 scanner calibration and color correction"
    )

    parser.add_argument("it8scan_or_inputimage", help="IT8 scan or image to correct")
    parser.add_argument(
        "it8referencedata_or_outputimage",
        help=(
            "IT8 reference text file (calibration) or output image (apply)."
            " Download the reference text file from https://www.silverfast.com/it8calibration/en.html"
        ),
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
        calibrate(args.it8scan_or_inputimage, args.it8referencedata_or_outputimage, args.calibrate, args)
    else:
        apply_calibration(
            args.it8scan_or_inputimage, args.it8referencedata_or_outputimage, args.apply_calibration
        )

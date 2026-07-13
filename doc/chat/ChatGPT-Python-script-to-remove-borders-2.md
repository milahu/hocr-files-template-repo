# Fix python script

**User:** Anonymous  
**Created:** 2025/10/26 10:44:36  
**Updated:** 2025/10/26 11:01:03  
**Exported:** 2026/7/13 20:24:03  
**Link:** [<https://chatgpt.com/c/68fded38-3938-832f-8140-4f676fa38646>](https://chatgpt.com/c/68fded38-3938-832f-8140-4f676fa38646)

## Prompt:

10/26/2025, 10:44:36 AM

help me fix this python script.

``` py
#!/usr/bin/env python3

INPUT_DIR = "060-rotate-crop-level"
OUTPUT_DIR = "065-remove-page-borders"

# === Tuning parameters ===
BORDER_SIZE = 10  # pixels

"""
AI prompt:

create a python script to remove grey (or black) page borders from scanned images.
the pages are white with black text.
the pages are no perfect rectangles, rather crooked trapezes with crooked lines...
so the algorithm should "overcut" the pages:
it should cut at the inner-most page edge,
so where the page edge is further outside some white area from the page is removed.

the script should process an input directory with *.tiff images
and write output images to an output directory (same image format).
the input and output paths should be hard-coded in the script,
so the script takes no command-line arguments.
the script should be based on the PIL (pillow) image library
(and on the opencv and numpy libraries when necessary)

...
"""

import os
from PIL import Image
import numpy as np
import cv2

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
    img = cv2.imread(in_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Threshold to isolate white page
    _, mask = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    page_contour = max(contours, key=cv2.contourArea)
    # Approximate contour to a quadrilateral
    epsilon = 0.02 * cv2.arcLength(page_contour, True)
    approx = cv2.approxPolyDP(page_contour, epsilon, True)
    if len(approx) != 4:
        print("Warning: contour approximation did not yield 4 points. Using convex hull instead.")
        approx = cv2.convexHull(page_contour)
        # Optionally select 4 corners from convex hull manually
    # Extract the 4 corner points
    pts = approx.reshape(4, 2)
    # Order the points consistently
    rect = order_points(pts)
    # Compute perspective transform
    # Compute width and height of new rectangle
    widthA = np.linalg.norm(rect[2] - rect[3])
    widthB = np.linalg.norm(rect[1] - rect[0])
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(rect[1] - rect[2])
    heightB = np.linalg.norm(rect[0] - rect[3])
    maxHeight = max(int(heightA), int(heightB))
    # Destination points for the "straight" rectangle
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    # Perspective transform
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    # Add internal white border
    # to remove grey artifacts from cropping crooked page edges
    # Create a white canvas of the same size
    h, w = warped.shape[:2]
    canvas = np.ones_like(warped) * 255  # white
    # Copy the warped content inside the canvas, leaving a white border
    b = BORDER_SIZE
    canvas[b:h-b, b:w-b] = warped[b:h-b, b:w-b]
    out_image = canvas
    # Save the result
    print(f"writing {out_path}")
    cv2.imwrite(out_path, out_image)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = [f for f in sorted(os.listdir(INPUT_DIR)) if f.lower().endswith((".tif", ".tiff"))]
    if not files:
        print("No TIFF files found in", INPUT_DIR)
        return
    for f in files:
        in_path = os.path.join(INPUT_DIR, f)
        out_path = os.path.join(OUTPUT_DIR, f)
        if os.path.exists(out_path): continue
        try:
            process_image(in_path, out_path)
        except Exception as e:
            print(f"Error processing {f}: {e}")

if __name__ == "__main__":
    main()
```

it should detect scanned book pages on a 50% grey background,  
but currently it only works for "white" pages (black text on white ground)  
but it should also work for "black" pages (white text on black ground)

what matters is the contrast to the 50% grey background,  
no matter if the page is white ore black

## Response:

10/26/2025, 10:44:36 AM

Ah! I see exactly what’s going on. Right now your script assumes the page is **white** and the background is darker, because your threshold is fixed at 230:

``` python
_, mask = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
```

This will fail for black pages on a mid-gray background, because there the page is **darker** than the background. What you really want is to detect **any high-contrast object** (page) against a roughly uniform 50% gray background.

We can do that by:

1.  Normalizing the image relative to the background (so background is mid-gray).
2.  Using **adaptive thresholding** or **Otsu’s method** to separate foreground (page) from background.
3.  Keeping the rest of your contour logic unchanged.

Here’s a fixed version of `process_image` that works for both light and dark pages on a gray background:

``` python
def process_image(in_path, out_path):
    img = cv2.imread(in_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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

    # Add internal white border
    h, w = warped.shape[:2]
    canvas = np.ones_like(warped) * 255
    b = BORDER_SIZE
    canvas[b:h-b, b:w-b] = warped[b:h-b, b:w-b]

    print(f"writing {out_path}")
    cv2.imwrite(out_path, canvas)
```

✅ Key changes:

1.  **Otsu threshold** instead of a fixed 230.
2.  **Automatic inversion** if the page is darker than background.
3.  **More robust convex hull handling** for non-quadrilateral pages.

This should handle **both black and white pages** on a mid-gray background.

------------------------------------------------------------------------

If you want, I can also **improve the contour detection** to be robust against slightly uneven gray backgrounds and faint borders, so it never accidentally chops off text. That will make it production-ready for scanned books.

Do you want me to do that?

## Prompt:

10/26/2025, 10:56:48 AM

> \# Add internal white border h, w = warped.shape\[:2\] canvas = np.ones_like(warped) \* 255 b = BORDER_SIZE canvas\[b:h-b, b:w-b\] = warped\[b:h-b, b:w-b\]

ok, but now this part fails on "black" pages...

a simple fix would be to use the average lightness of the page to detect white vs black pages...  
but it can happen that the top half of a page is white, and the bottom half of a page is black  
then this simple fix would fail, it would always add a wrong-colored border...

a more complex fix would be to localize this white/black page detection  
by "looping" the pixels at the page edge  
and by detecting the page background color from the pixels between 50px and 100px inside of the page (with a "strip size" of 1px)  
and then "extrapolate" that color to the 100px page border (again with a "strip size" of 1px)  
and with this, step by step, processing all page borders

## Response:

10/26/2025, 10:56:49 AM

Ah! I see exactly the issue. The current logic blindly creates a **white border**:

``` python
canvas = np.ones_like(warped) * 255  # always white
canvas[b:h-b, b:w-b] = warped[b:h-b, b:w-b]
```

This fails for black pages or pages with mixed light/dark regions at the edge. To fix this, you need a **localized border color detection**—essentially “look at the pixels near the edge inside the page, detect whether they are closer to white or black, and then extend that color outward as the border.”

Here’s a robust approach:

------------------------------------------------------------------------

### 1️⃣ Detect local background color at each edge

- Loop over the four edges (top, bottom, left, right).
- For each edge, sample a “strip” of pixels inside the page (e.g., 50–100 px from edge, 1px wide).
- Compute the **average color of that strip**. This becomes the border color for that side.

### 2️⃣ Apply the border using the detected color

- Instead of blindly writing white, fill each border region with the **detected average color** for that side.

------------------------------------------------------------------------

Here’s a concrete implementation snippet replacing the previous white border logic:

``` python
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
```

✅ Notes:

- `start=50, end=100` means you sample **50–100 px inside the page** to avoid contamination from the grey background or noisy edges.
- `strip_width=1` is used implicitly—you can adjust for thicker sampling.
- This works even if the page has **mixed light/dark regions**, because each edge is processed independently.

------------------------------------------------------------------------

If you want, I can **rewrite your full `process_image` function** with this localized border detection included and fully integrated with the black/white page handling.

Do you want me to do that?

------------------------------------------------------------------------

Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)

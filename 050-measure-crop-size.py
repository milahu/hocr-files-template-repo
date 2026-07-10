"""
Config for 060-rotate-crop-level.py
"""

# TODO use image_format from 030-measure-page-size.txt
image_format = "jpg"

# --- Rotation settings ---
do_rotate = False
rotate_odd = 90
rotate_even = 270

# --- Crop settings ---
do_crop = False
crop_size = (1580, 2480)
crop_x = 168
# x1, y1, x2, y2 = crop_box
crop_odd_box = (crop_x, 0, crop_x + crop_size[0], crop_size[1])
crop_even_box = (0, 0, crop_size[0], crop_size[1])

# --- Level / brightness normalization ---
# leveling is useful to remove noise
# from black and white areas in text
# but too much leveling causes too much loss in contrast
# in darkgray and lightgray areas in grayscale graphics
# lowthresh=0.3 is necessary to remove dither from black areas
# lowthresh and highthresh should be symmetrical (lowthresh + highthresh == 1)
# so contours are preserved
# todo: use different leveling values for text and graphics
# for graphics, we want lowthresh=0.05
# to preserve darkgray and lightgray areas in grayscale graphics
do_level = True
# lowthresh, highthresh = 0.05, 0.95
# lowthresh, highthresh = 0.2, 0.8
lowthresh, highthresh = 0.3, 0.7

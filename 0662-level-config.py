"""
Config for 0663-level.py
"""

# TODO use image_format from 030-measure-page-size.txt
image_format = "jpg"

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

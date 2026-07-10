"""
Config for 060-rotate-crop.py
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

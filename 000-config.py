# TODO set config values

num_pages = 592

color_pages = []

# TODO rename all mm sizes to scan_x_mm etc

max_scan_x, max_scan_y = 215.88, 355.567 # maximum

# NOTE this should be the full page size in millimeters
# full size = before the book binding was removed
# this size is used
# 1. in 040-scan-pages.py
# 2. in 065-remove-page-borders.py to restore the original page size
# scan_x, scan_y = 210, 297 # DIN A4
scan_x, scan_y = 147, 231
scan_x, scan_y = 156, 210
scan_x, scan_y = 210, 156 # rotate 90

# scan_x, scan_y = max_scan_x, max_scan_y # maximum

# this is True for most book pages
# in reading order, the page width is smaller than the page height
# orientation_is_portrait = False
orientation_is_portrait = True

assert scan_x <= max_scan_x, f"scan_x is out of range: {scan_x} > {max_scan_x}"
assert scan_y <= max_scan_y, f"scan_y is out of range: {scan_y} > {max_scan_y}"

# add margin for 065-remove-page-borders.py
scan_margin = 10

# no! do this only in 040-scan-pages.py
# scan_x = scan_x + scan_margin
# scan_y = scan_y + scan_margin



# 300 dpi -> 12 MB
# 600 dpi -> 50 MB
# 1200 dpi -> 200 MB
# scan_resolution = 300
scan_resolution = 600
# scan_resolution = 1200

# uncompressed image format
# no. JPEG quality is too low (below 95%) and not configurable
# scan_format = "jpg"
# no. PNG compression is too slow
# scan_format = "png"
# PNM and TIFF formats are uncompressed = fast
# scan_format = "pnm"
scan_format = "tiff"

import cv2

cv2_imwrite_params = [
    # disable compression in TIFF container (lossless)
    # done 480 pages in 150 seconds using 6 workers
    # 24825889728     060-rotate-crop # 100%
    # cv2.IMWRITE_TIFF_COMPRESSION, cv2.IMWRITE_TIFF_COMPRESSION_NONE,

    # default: use LZW compression in TIFF container (lossless)
    # done 480 pages in 111 seconds using 6 workers
    # done 480 pages in 140 seconds using 6 workers
    # 6948263236      060-rotate-crop # 28%
    # cv2.IMWRITE_TIFF_COMPRESSION, cv2.IMWRITE_TIFF_COMPRESSION_LZW,

    # default: use Deflate compression in TIFF container (lossless)
    # done 480 pages in 120 seconds using 6 workers
    # 5928015210      060-rotate-crop # 24%
    cv2.IMWRITE_TIFF_COMPRESSION, cv2.IMWRITE_TIFF_COMPRESSION_ADOBE_DEFLATE,

    # default: use ZSTD compression in TIFF container (lossless)
    # done 480 pages in 130 seconds using 6 workers
    # 5985643952      060-rotate-crop # 24%
    # cv2.IMWRITE_TIFF_COMPRESSION, cv2.IMWRITE_TIFF_COMPRESSION_ZSTD,
]



# compressed image format
# for 062-compress.py
image_format = "jpg"



# these values depend on the scanner model
# see also:
# scanimage --help --device-name="your_device_name"

# scan_mode = "24bit Color[Fast]"
scan_mode = "True Gray"
scan_mode = "24bit Color[Fast]"

# "center aligned" is not working: scanimage failed with returncode -11
# scan_source = "Automatic Document Feeder(center aligned,Duplex)"
scan_source = "Automatic Document Feeder(left aligned,Duplex)"



# Config for 060-rotate-crop.py

# --- Rotation settings ---
do_rotate = True
# rotate_odd, rotate_even = 90, 270
# book binding side = scanner top side
rotate_odd, rotate_even = 270, 90



# --- Crop settings ---
do_crop = False
crop_size = (1580, 2480)
crop_x = 168
# x1, y1, x2, y2 = crop_box
crop_odd_box = (crop_x, 0, crop_x + crop_size[0], crop_size[1])
crop_even_box = (0, 0, crop_size[0], crop_size[1])



# Config for 0663-level.py

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



# https://github.com/derf/feh
image_viewer = "feh"

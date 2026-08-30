# TODO set config values

num_pages = 592

color_pages = []

# pages with images (grayscale or color)
# TODO use the OCR result to separate text and image regions
# default: infer image_pages from color_pages
# image_pages = []

# scanner limits
max_scan_width_mm = 215.88
max_scan_height_mm = 355.567

# physical page size (before unbinding)
page_width_mm = 124
page_height_mm = 190

# physical page width after unbinding
# unbinding removes a small strip (about 5 mm width) from the inside edge
unbinded_page_width_mm = 121

# which physical page edge is fed first into the document scanner?
# that edge becomes the scan top edge
# possible value: "inside" or "top" or "bottom"
# the value "outside" makes no sense
# because the scanner removes a small part of the scan top edge
# so that edge can no longer be used to detect the page rotation
# so if possible, the user should try to feed the physical inside edge first into the scanner
# Rotation and scanner-area dimensions are derived in _shared.load_config().
# scan_top_edge = "inside"
# scan_top_edge = "top"
# scan_top_edge = "bottom"
scan_top_edge = "inside"

# Less clearance than this makes the outside edge unreliable for detection.
outside_edge_detection_min_margin_mm = 2.0

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

pil_image_save_kwargs = dict(
    format="TIFF",
    compression="tiff_adobe_deflate",
)



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



# --- Crop settings ---
do_crop = False
crop_size = (1580, 2480)
crop_x = 168
# x1, y1, x2, y2 = crop_box
crop_odd_box = (crop_x, 0, crop_x + crop_size[0], crop_size[1])
crop_even_box = (0, 0, crop_size[0], crop_size[1])



# Config for 0663-level.py

# TODO use different thresholds for text and images
# a too low text_lowthresh produces too much noise in black areas
# a too high images_lowthresh causes excessive tonal clipping
# use the OCR result to separate text and image regions
text_lowthresh = 0.3
images_lowthresh = 0.05

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
# lowthresh = 0.05
# lowthresh = 0.2
lowthresh = 0.3

# leveling is too destructive on images and colors
# so we keep the darkgray text on lightgray background
# do_level = False



# https://github.com/derf/feh
image_viewer = "feh"



# config for 0685-fill-white-pages.py

fill_white_pages_lightness_file = "0683-lightness.txt"

# 099.999270 325.tiff # white
# 099.792250 001.tiff # non-white
fill_white_pages_white_lightness_threshold = 99.99 / 100 # 99.99%



# config for 070-deskew.py

# Threshold to consider a page "white" (mean lightness close to 1)
# deskew_white_lightness_threshold = 99.99 / 100 # 99.99%
deskew_white_lightness_threshold = 99.95 / 100 # 99.95%
# deskew_white_lightness_threshold = 99.90 / 100 # 99.90%

# Threshold to consider a page "black" (mean lightness close to 0)
# black page with little white text can have 0.49 to 0.84
# black page with no text can have 0 to 0.68
deskew_black_lightness_threshold = 0.05 / 100 # 0.05%

# Threshold to consider a page "dark" (black page with white text)
# white page with lots of black text can have 0.80
deskew_dark_lightness_threshold = 25 / 100 # 25%



# config for 090-ocr.py

ocr_lang = "deu+eng" # german + english

# TODO? remove tessdata_dir in favor of tessdata_cache_dir
tessdata_dir = "tessdata_best"

tessdata_cache_dir = "$HOME/.cache/tessdata_best"

# file URL format: f"{base_url}/{lang}.traineddata"
# example file URL: f"{base_url}/eng.traineddata"
tessdata_base_url = "https://github.com/tesseract-ocr/tessdata_best/raw/main"

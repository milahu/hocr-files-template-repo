# TODO set config values

num_pages = 592

color_pages = []

# NOTE this should be the full page size in millimeters
# full size = before the book binding was removed
# this size is used
# 1. in 040-scan-pages.py
# 2. in 065-remove-page-borders.py to restore the original page size
# scan_x, scan_y = 210, 297 # DIN A4
# scan_x, scan_y = 215.88, 355.567 # maximum
scan_x, scan_y = 147, 231

# add margin for 065-remove-page-borders.py
scan_margin = 10
scan_x = scan_x + scan_margin
scan_y = scan_y + scan_margin

# 300 dpi -> 12 MB
# 600 dpi -> 50 MB
# 1200 dpi -> 200 MB
# scan_resolution = 300
scan_resolution = 600
# scan_resolution = 1200

# no. JPEG quality is too low (below 95%) and not configurable
# scan_format = "jpg"
# no. PNG compression is too slow
# scan_format = "png"
# PNM and TIFF formats are uncompressed = fast
# scan_format = "pnm"
scan_format = "tiff"

# for 046-compress.py etc
image_format = "jpg"



# these values depend on the scanner model
# see also:
# scanimage --help --device-name="your_device_name"

# scan_mode = "24bit Color[Fast]"
scan_mode = "True Gray"

scan_source = "Automatic Document Feeder(left aligned,Duplex)"

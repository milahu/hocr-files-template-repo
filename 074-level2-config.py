"""
Config for 075-level2.py
"""

scan_format = "tiff"

# --- Level / brightness normalization ---
# leveling is useful to remove noise
# from black and white areas in text
# but too much leveling causes too much loss in contrast
# in darkgray and lightgray areas in grayscale graphics

do_level = True

# this color leveling was already applied in 060-rotate-crop-level.py
# using an old config in 050-measure-crop-size.py
previous_lowthresh = 0.05 # about 15/255 in GIMP
previous_highthresh = 0.95 # about 240/255 in GIMP

# NOTE these are the final result levels
# after running 060-rotate-crop-level.py and 075-level.py
# so if the levels here are the same as in 050-measure-crop-size.py
# then 075-level.py is a noop

# lowthresh = 0.05 # about 15/255 in GIMP
# highthresh = 0.95 # about 240/255 in GIMP

# lowthresh = 0.2 # too dark for the book cover -> use 0.1
# lowthresh = 0.1 # about 25/255 in GIMP

# leveling for text-only books
lowthresh = 0.2 # about 50/255 in GIMP
highthresh = 0.9 # about 230/255 in GIMP



# leveling has been done. dont repeat
do_level = False

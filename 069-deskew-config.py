# config for 070-deskew.py

# Threshold to consider a page "white" (mean lightness close to 100)
WHITE_LIGHTNESS_THRESHOLD = 99.99

# Threshold to consider a page "black" (mean lightness close to 0)
# black page with little white text can have 0.49 to 0.84
# black page with no text can have 0 to 0.68
BLACK_LIGHTNESS_THRESHOLD = 1.0

# Threshold to consider a page "dark" (black page with white text)
# white page with lots of black text can have 80
BLACK_LIGHTNESS_THRESHOLD_2 = 25

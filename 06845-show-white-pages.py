#!/usr/bin/env python3

import subprocess
from pathlib import Path

from _shared import (
    load_config,
)

config = load_config()

# Input text file
txt_file = Path("0683-lightness.txt")

# Directory containing the images
image_dir = Path("065-remove-page-borders")

# Read filenames from the second column
image_files = []

with txt_file.open("r", encoding="utf-8") as f:
    for line in f:
        parts = line.split()
        if len(parts) >= 2:
            filename = parts[1]
            image_files.append(image_dir / filename)

# Open all images
subprocess.run([config.image_viewer, *map(str, image_files)])

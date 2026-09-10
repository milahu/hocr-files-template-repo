#!/bin/sh

# TODO set config values
cover_src=072-deskew-fix-page-size/999.tiff

magick "$cover_src" -scale 50% -quality 50% cover.avif

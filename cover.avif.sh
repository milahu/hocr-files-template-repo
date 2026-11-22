#!/bin/sh

# TODO set config values
cover_src=075-level/999.tiff

magick "$cover_src" -scale 50% -quality 50% cover.avif

#!/bin/sh

# TODO set config values
cover_src=070-deskew/273.tiff

magick "$cover_src" -scale 50% -quality 30% cover.avif

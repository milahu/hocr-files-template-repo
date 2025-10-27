#!/bin/sh

# TODO set config values
cover_src=070-deskew/225.png

magick "$cover_src" -scale 50% -quality 5% cover.avif

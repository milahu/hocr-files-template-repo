#!/usr/bin/env bash

cd "$(dirname "$0")"
datetime=$(date -Is | tr : -)
tmp=$(basename "$0" .sh).tmp.$datetime
src=070-deskew
dst=070-deskew # replace files
bak=070-deskew.bak.$datetime

# scan_format=tiff
source 030-measure-page-size.txt

function check_command() {
  if ! command -v "$1" &>/dev/null; then
    echo "error: missing command: $1"
    exit 1
  fi
}

check_command tiffcp
check_command identify

mkdir -p "$tmp"
mkdir -p "$bak"

num_done=0

for f in "$src"/*.$scan_format; do

    base=$(basename "$f")
    f_tmp="$tmp/$base"
    f_bak="$bak/$base"

    # Count number of images (IFDs) in TIFF
    count=$(identify "$f" | wc -l)

    if [ "$count" -gt 1 ]; then
        echo "fixing $f"

        # quasi-lossless in quality, but maybe different compression
        false &&
        magick "$f[0]" \
            -strip \
            -compress JPEG \
            -quality 95 \
            "$f_tmp"

        # lossless in quality, but maybe different compression
        false &&
        magick "$f[0]" -strip -compress LZW "$f_tmp"

        # lossless
        tiffcp -1 "$f" "$f_tmp"

        # atomic backup and replace
        cp --link "$f" "$f_bak"
        mv "$f_tmp" "$f"

        num_done=$((num_done + 1))

    # else
    #     echo "keeping $f"

    fi

done

rmdir "$tmp"

if [ $num_done = 0 ]; then
  rmdir "$bak" 2>/dev/null
fi

echo "done. fixed $num_done images"

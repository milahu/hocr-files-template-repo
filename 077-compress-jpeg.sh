#!/usr/bin/env bash

# set -x # debug

cd "$(dirname "$0")"
datetime=$(date -Is | tr : -)
src=070-deskew
dst=$(basename "$0" .sh)
# dst="$src" # replace files in src

if [ "$src" = "$dst" ]; then
  tmp=$(basename "$0" .sh).tmp.$datetime
  bak=070-deskew.bak.$datetime
  replace=true
else
  replace=false
  mkdir -p "$dst"
fi

jpeg_quality=90

# scan_format=tiff
# color_pages=123,124,125
source 030-measure-page-size.txt

function check_command() {
  if ! command -v "$1" &>/dev/null; then
    echo "error: missing command: $1"
    exit 1
  fi
}

check_command tiffinfo
check_command identify

if $replace; then
  mkdir -p "$tmp"

  echo "creating $bak"
  mkdir -p "$bak"
fi

num_done=0

set -e

for f in "$src"/*.$scan_format; do

    base=$(basename "$f")
    if ! grep -q -E "^[0-9]+\.$scan_format$" <<<"$base"; then continue; fi

    base_dst=$base
    base_dst=${base_dst%.*} # remove file extension
    base_dst=${base_dst}.jpg # add file extension

    # echo "f: ${f@Q}" # debug

    if $replace; then
      f_tmp="$tmp/$base_dst"
      f_dst="$src/$base_dst"
      f_bak="$bak/$base"
    else
      f_dst="$dst/$base_dst"
      f_tmp="$f_dst"
    fi

    compression=$(tiffinfo "$f" | grep Compression | sed -E 's/^.*?: //')
    # echo "compression: ${compression@Q}" # debug

    # compression should be "LZW"

    # if [ "$compression" != "JPEG" ]; then
    if true; then

        size_a=$(stat -c%s "$f")

        page_num=$base
        page_num=${page_num%%.*} # remove file extensions
        page_num=$(expr "$page_num" + 0) # remove leading zeros

        colorspace_args=
        if false; then # preserve colorspace
        if [ -z "$color_pages" ] || ! grep -q -w "$page_num" <<<"$color_pages"; then
            colorspace_args="-colorspace Gray"
        fi
        fi

        magick "$f" \
            $colorspace_args \
            -compress JPEG \
            -quality "$jpeg_quality" \
            "$f_tmp"

        size_b=$(stat -c%s "$f_tmp")

        if ((size_b < size_a)); then
            size_b_percent=$(expr "$size_b" "*" 100 / "$size_a")
            echo "compressing $f: compression: ${compression@Q} -> JPEG. file size: $size_a -> $size_b (${size_b_percent}%)"
        fi
        # else: compression made it worse

        if $replace && ((size_b < size_a)); then
            # backup
            cp --link "$f" "$f_bak"

            if [ "$f" = "$f_dst" ]; then
                # atomic replace
                mv "$f_tmp" "$f_dst"
            else
                # replace
                mv "$f_tmp" "$f_dst"
                rm "$f"
            fi

            num_done=$((num_done + 1))
        # else: $replace == false -> "$f_tmp" == "$f_dst"
        fi

        if ! $replace; then
            num_done=$((num_done + 1))
        fi

    # else
    #     echo "keeping $f"

    fi

done

if $replace; then
    rmdir "$tmp"
fi

if $replace && [ $num_done = 0 ]; then
  rmdir "$bak" 2>/dev/null
fi

echo "done. compressed $num_done images"

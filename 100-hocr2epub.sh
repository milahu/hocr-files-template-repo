#!/usr/bin/env bash

set -eu

# dst=$(basename "$0" .sh).epub
dst=.

doc_title="$(head -n1 readme.md | sed 's/^#\s*//')"

if false; then
  scan_resolution=600
else
  source 030-measure-page-size.txt
fi

if [ "$dst" != "." ] && [ -e "$dst" ]; then
  echo "error: output exists: $dst"
  exit 1
fi

# downscale to 300 dpi
scale=$(python -c "print(300 / $scan_resolution)")

args=(
  hocr-to-epub-fxl
  --output "$dst"
)
if [ "$dst" = "." ]; then
  args+=(
    --output-unpacked
  )
fi
args+=(
  --scale "$scale"
  --image-format avif
  --text-format html
  --doc-title "$doc_title"
)

printf '>'
for a in "${args[@]}" "$@"; do printf ' %q' "$a"; done
echo ' *-ocr/*.hocr'

"${args[@]}" "$@" *-ocr/*.hocr

if [ "$dst" = "." ]; then
  echo "done ./index.html"
  exit
fi

echo "done $dst"

rm -rf $dst.unzip
mkdir $dst.unzip
cd $dst.unzip
unzip -q ../$dst
cd ..

echo "done $dst.unzip/index.html"

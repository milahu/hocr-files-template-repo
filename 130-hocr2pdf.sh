#!/usr/bin/env bash

# FIXME archive-pdf-tools does not support HTML in words
# example: Word<sup>123</sup>
# see also https://github.com/internetarchive/archive-hocr-tools/pull/23

cd "$(dirname "$0")"
src_hocr=090-ocr
src_image=070-deskew
dst=$(basename "$0" .sh)

if false; then
  scan_format=jpg
else
  source 030-measure-page-size.txt
fi

mkdir $dst

for hocr in 090-ocr/*.hocr; do

  page=${hocr%.hocr}
  page=${page##*/}

  hocr=$src_hocr/$page.hocr
  image=$src_image/$page.$scan_format
  pdf=$dst/$page.pdf

  [ -e $pdf ] && continue

  echo $page
  recode_pdf --hocr-file $hocr --dpi 300 --out-pdf $pdf --bw-pdf --from-imagestack $image

done

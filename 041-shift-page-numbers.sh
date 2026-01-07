#!/usr/bin/env bash

src=040-scan-pages

page_delta="$1"

if ! echo "$page_delta" | grep -qE '^[+-]?[0-9]+$'; then
  echo "bad page_delta: ${page_delta@Q}"
  echo "example use:"
  echo "  $0 -5"
  echo "  $0 +5"
  exit 1
fi

if echo "$page_delta" | grep -qE '^[0-9]+$'; then
  page_delta="+$page_delta"
fi

echo "page_delta: ${page_delta@Q}"

# set: num_pages scan_format
source 030-measure-page-size.txt

page_num_fmt="%0${#num_pages}d"

page_delta_sign="${page_delta:0:1}"
page_delta_abs="${page_delta:1}"
echo "page_delta_sign: ${page_delta_sign@Q}"
echo "page_delta_abs: ${page_delta_abs@Q}"

function shift_page_number() {
  local page="$1"
  local page0a=$(printf "$page_num_fmt" "$page")
  local page0b=$(printf "$page_num_fmt" $((page $page_delta_sign $page_delta_abs)))
  local a="$src"/$page0a.$scan_format
  local b="$src"/$page0b.$scan_format
  if [ -e "$b" ]; then
    echo "FIXME collision: mv ${a@Q} ${b@Q}"
    exit 1
  fi
  mv -v "$a" "$b"
}

if [ "$page_delta_sign" = "-" ]; then
  # decrease page numbers
  for ((page=1; page<=$num_pages; page++)); do
    shift_page_number $page
  done
else
  # increase page numbers: loop in reverse order
  for ((page=$num_pages; page>=1; page--)); do
    shift_page_number $page
  done
fi

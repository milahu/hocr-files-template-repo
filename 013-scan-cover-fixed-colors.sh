#!/usr/bin/env bash

# set -x

cd "$(dirname "$0")"
dst=$(basename "$0" .sh)

mkdir -p $dst

for f in 010-scan-cover/*.tiff; do

  n="$(basename "$f")"
  f2="$dst/$n"

  if [ -e "$f2" ]; then
    echo "keeping $f2"
    continue
  fi

  # echo "writing $f2"
  ./012-fix-colors.py --apply-calibration 012-fix-colors.calibration.json "$f" "$f2"

done

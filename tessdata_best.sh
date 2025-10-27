#!/bin/sh

# download tessdata files from https://github.com/tesseract-ocr/tessdata_best

if [ $# = 0 ]; then
  echo "error: no arguments"
  echo "example use: ./tessdata_best.sh eng deu rus"
  exit 1
fi

dst=tessdata_best

# cache downloaded files in ~/.cache/tessdata_best
cache_dir="$HOME/.cache/tessdata_best"

cd "$(dirname "$0")"

mkdir -p "$dst"
mkdir -p "$cache_dir"

urls=()
for lang in "$@"; do
  [ -e "$dst/$lang.traineddata" ] && continue
  [ -e "$cache_dir/$lang.traineddata" ] && continue
  urls+=(https://github.com/tesseract-ocr/tessdata_best/raw/main/"$lang".traineddata)
done

if [ ${#urls[@]} != 0 ]; then
  # write cache
  pushd "$cache_dir"
  wget --no-clobber "${urls[@]}"
  popd
fi

for lang in "$@"; do
  [ -e "$dst/$lang.traineddata" ] && continue
  if [ -e "$cache_dir/$lang.traineddata" ]; then
    # read cache
    ln -sr "$cache_dir/$lang.traineddata" "$dst/$lang.traineddata"
  else
    echo "error: missing cache file: $cache_dir/$lang.traineddata"
  fi
done

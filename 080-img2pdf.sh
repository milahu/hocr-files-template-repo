#!/usr/bin/env bash

# src=077-compress-jpeg; src_ext=jpg

src=OEBPS/images; src_ext=avif

dst=$(basename "$0" .sh).pdf
tmp=$(basename "$0" .sh)-tmp

if [ -e $dst ]; then
  echo "error: output exists: $dst"
  exit 1
fi

# NOTE this can fail with
# raise AlphaChannelError("This function must not be called on images with alpha")
# https://gitlab.mister-muffin.de/josch/img2pdf/src/commit/bb188a3eaf7d956b82f7f9a18bbda774301c586f/src/img2pdf.py#L1408
# img2pdf --output $dst --pagesize A4 --border 0cm $src/*.tiff

mkdir -p $tmp

for i in $src/*.$src_ext; do
  o=$tmp/${i##*/}
  [ -e "$o" ] && continue
  if ! is_opaque=$(identify -format '%[opaque]' "$i"); then
    echo "error: failed to detect alpha channel in $i: $is_opaque"
    exit 1
  else
    case "$is_opaque" in
      True)
        # no alpha channel -> symlink
        echo "no alpha channel in $i"
        ln -sr "$i" "$o"
        continue
        ;;
      False)
        # alpha channel -> remove alpha channel
        echo "removing alpha channel from $i"
        magick "$i" -alpha off "$o"
        continue
        ;;
      *)
        echo "error: failed to detect alpha channel in $i: $is_opaque"
        exit 1
        ;;
    esac
  fi
done

echo "calling img2pdf. this will take some time..."
echo "+ img2pdf --output $dst --pagesize A4 --border 0cm $tmp/*.$src_ext"
img2pdf --output $dst --pagesize A4 --border 0cm $tmp/*.$src_ext

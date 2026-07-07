#!/bin/sh

cat 0683-lightness.txt |
cut -c12- |
sed 's|^|065-remove-page-borders/|' |
xargs feh

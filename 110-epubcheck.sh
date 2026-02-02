#!/usr/bin/env bash

# check unpacked epub files

set -eu

export LANG=C

# TODO? grep --line-buffered

# epubcheck is really slow...

time epubcheck -mode exp . 2>&1 | grep -v '^ERROR(RSC-032)'

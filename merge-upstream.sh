#!/usr/bin/env bash

set -eu
# set -x # debug

upstream_url='https://github.com/milahu/hocr-files-template-repo'

origin_url=$(
    git remote get-url origin 2>/dev/null ||
    git remote get-url github.com 2>/dev/null ||
    true
)

if [ "$origin_url" = "$upstream_url" ]; then
    echo "error: this is not a fork of $upstream_url"
    exit 1
fi

if ! git remote show upstream &>/dev/null; then
    git remote add upstream "$upstream_url"
fi

git fetch upstream main

git merge upstream/main -m "merge the main branch of $upstream_url"

# TODO fix merge conflicts

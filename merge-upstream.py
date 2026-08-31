#!/usr/bin/env python3

import argparse
import subprocess
import sys

UPSTREAM_URL = "https://github.com/milahu/hocr-files-template-repo"


def run_git(*args, check=True, capture_output=False):
    """Run a git command."""
    return subprocess.run(
        ["git", *args],
        check=check,
        capture_output=capture_output,
        text=True,
    )


def get_origin_url():
    """Get the configured origin URL, or an empty string if unavailable."""
    for remote in ("origin", "github.com"):
        result = run_git(
            "remote",
            "get-url",
            remote,
            check=False,
            capture_output=True,
        )
        if result.returncode == 0:
            return result.stdout.strip()

    return ""


def main():
    parser = argparse.ArgumentParser(
        description="Merge the upstream main branch into the current repository."
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Allow running even if origin points to the upstream repository.",
    )
    args = parser.parse_args()

    origin_url = get_origin_url()

    r'''
    if origin_url == UPSTREAM_URL and not args.force:
        print(
            f"error: this is not a fork of {UPSTREAM_URL}",
            file=sys.stderr,
        )
        sys.exit(1)
    '''

    # Add the upstream remote if it doesn't already exist.
    result = run_git(
        "remote",
        "show",
        "upstream",
        check=False,
        capture_output=True,
    )

    if result.returncode != 0:
        run_git("remote", "add", "upstream", UPSTREAM_URL)

    # Fetch upstream main.
    run_git("fetch", "upstream", "main")

    # Merge upstream/main.
    run_git(
        "merge",
        "upstream/main",
        "-m",
        f"merge the main branch of {UPSTREAM_URL}",
    )


if __name__ == "__main__":
    main()

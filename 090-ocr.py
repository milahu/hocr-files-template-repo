#!/usr/bin/env python3

import os
import time
import json
import subprocess
from pathlib import Path
from multiprocessing import Pool

import psutil
import requests


# source directory
src = "077-compress-jpeg"

# default config
scan_resolution = 300
scan_format = "jpg"
ocr_lang = "deu+eng"


# ----------------------------
# config loading (bash "source")
# ----------------------------
def load_kv_file(path):
    cfg = {}
    p = Path(path)
    if not p.exists():
        return cfg

    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip()
    return cfg


# ----------------------------
# worker function
# ----------------------------
def run_tesseract(task):
    inp, out_hocr, scan_resolution, ocr_lang, tessdata_dir = task

    inp = Path(inp)
    out_hocr = Path(out_hocr)

    if out_hocr.exists():
        return f"{inp.name}: keeping {out_hocr}"

    cmd = [
        "tesseract",
        str(inp),
        "-",
        "-c", "tessedit_create_hocr=1",
        "--dpi", str(scan_resolution),
        "-l", ocr_lang,
        "--tessdata-dir", tessdata_dir,
    ]

    env = dict(os.environ)

    # https://github.com/tesseract-ocr/tesseract/issues/1600
    # Tesseract 4 also uses up to four CPU threads while processing a page
    # Using a single thread eliminates the computation overhead of multithreading
    # and is also the best solution for processing lots of images
    # by running one Tesseract process per CPU core.
    # To disable multithreading, use OMP_THREAD_LIMIT=1.
    env["OMP_THREAD_LIMIT"] = "1"

    with open(out_hocr, "wb") as f:
        proc = subprocess.run(
            cmd,
            env=env,
            stdout=f,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

    result = ""

    if proc.stderr:
        for line in proc.stderr.splitlines():
            # "stderr" as in: "tesseract printed this to stderr"
            result += f"{inp.name}: stderr: {line}\n"

    result += f"{inp.name}: ok"
    result = result.rstrip() # remove trailing "\n"
    return result


def download_tessdata_best(langs, dst="tessdata_best"):
    """
    Download Tesseract tessdata_best models with caching (Python version of tessdata_best.sh)
    """

    if not langs:
        raise ValueError("no arguments (example: tessdata_best('eng', 'deu', 'rus'))")

    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)

    dst = Path(dst)
    cache_dir = Path.home() / ".cache" / "tessdata_best"

    dst.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    base_url = "https://github.com/tesseract-ocr/tessdata_best/raw/main"

    to_download = []

    # ----------------------------
    # decide what needs downloading
    # ----------------------------
    for lang in langs:
        target = dst / f"{lang}.traineddata"
        cache = cache_dir / f"{lang}.traineddata"

        if target.exists():
            continue
        if cache.exists():
            continue

        url = f"{base_url}/{lang}.traineddata"
        to_download.append((url, cache))

    # ----------------------------
    # download missing files
    # ----------------------------
    for url, cache_path in to_download:
        print(f"downloading {url}")
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()

        with open(cache_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    # ----------------------------
    # link into dst
    # ----------------------------
    for lang in langs:
        target = dst / f"{lang}.traineddata"
        cache = cache_dir / f"{lang}.traineddata"

        if target.exists():
            continue

        if cache.exists():
            try:
                # relative symlink like ln -sr
                rel = os.path.relpath(cache, start=dst)
                os.symlink(rel, target)
                print(f"linked {target} -> {cache}")
            except FileExistsError:
                pass
        else:
            print(f"error: missing cache file: {cache}")


def main():

    global src
    global scan_resolution
    global scan_format
    global ocr_lang

    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)

    dst = Path(Path(__file__).stem)

    dst.mkdir(exist_ok=True)

    # optional config files (like "source ...")
    scan_cfg = load_kv_file("030-measure-page-size.txt")
    print(f"scan_cfg: {json.dumps(scan_cfg, indent=2)}")
    ocr_cfg = load_kv_file("080-ocr-config.txt")
    print(f"ocr_cfg: {json.dumps(ocr_cfg, indent=2)}")

    scan_resolution = int(scan_cfg.get("scan_resolution", scan_resolution))
    # scan_format = scan_cfg.get("scan_format", scan_format)
    ocr_lang = ocr_cfg.get("ocr_lang", ocr_lang)

    tessdata_dir = "tessdata_best"

    download_tessdata_best(ocr_lang.split("+"), tessdata_dir)

    # tesseract writes HOCR files with the image paths relative to the workdir
    os.chdir(dst)

    tessdata_dir = "../" + tessdata_dir
    dst = Path("..") / dst

    # ----------------------------
    # CPU-aware multiprocessing
    # ----------------------------
    num_workers = psutil.cpu_count(logical=False) or 1

    print(f"num_workers: {num_workers}")

    src_dir = Path("..") / src
    inputs = sorted(src_dir.glob(f"*.{scan_format}"))

    tasks = [
        (str(inp), dst / (inp.stem + ".hocr"), scan_resolution, ocr_lang, tessdata_dir)
        for inp in inputs
    ]

    t1 = time.time()
    num_pages = 0

    with Pool(processes=num_workers) as pool:
        for result in pool.imap_unordered(run_tesseract, tasks):
            print(result)
            num_pages += 1

    t2 = time.time()

    print(f"done {num_pages} pages in {int(t2 - t1)} seconds")


if __name__ == "__main__":
    main()

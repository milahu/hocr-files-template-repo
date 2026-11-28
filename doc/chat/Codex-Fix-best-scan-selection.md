# Fix best scan selection

> fix the python script 045-find-best-scans.py
>
> for example, it fails on these pages:
>
> 163,171: x=.8
>
> 16,18,20,22,24,26,30,32,34,38,40,42: x=.95 \
> where x is the approximated relative x-position of the expected vertical gray scanner-artifact line
>
> the output image files are at 045-find-best-scans/016.tiff, 045-find-best-scans/018.tiff, etc
>
> ```
> $ ls 045-find-best-scans/*.tiff | wc -l
> 244
> ```
>
> the input image files  are at 040-scan-pages\*/\*.tiff
>
> ```
> $ ls 040-scan-pages*/*.tiff | wc -l
> 976
> ```
>
> for example, for page 171, the script should not select 040-scan-pages.tmp.1795622484/171.tiff, but rather all other versions, because all input images seem to contain scanner-artifacts, but 040-scan-pages.tmp.1795622484/171.tiff really is the worst version, because there, the artifact is really dark and big (average color is about #a5a5a5 = midgray, 7 pixels wide), whereas in the other versions, the artifact is light and small (average color is about #f6f6f6 = lightgray = almost white, 1 pixel wide). all the better versions have artifacts at position x=0.15
>
> ```
> $ md5sum 040-scan-pages*/171.tiff 045-find-best-scans/171.tiff
> f5b4e79e3d8fa3a3b6cf2426cbcda6b0  040-scan-pages/171.tiff
> 2712107c0c380323f94f04169a07205e  040-scan-pages.tmp.1795620804/171.tiff
> 6c03cd02b49eef115dd9d26cc1c69b74  040-scan-pages.tmp.1795621073/171.tiff
> 80f1e96111479f52f5c67ddec9bb5f67  040-scan-pages.tmp.1795622484/171.tiff
> 80f1e96111479f52f5c67ddec9bb5f67  045-find-best-scans/171.tiff
> ```

<details><summary>4 previous messages</summary>

> I’ll inspect the scan ranking and check the revised scores against the pages you listed.
>
> <details><summary>Edited a file, ran 11 commands</summary>
>
> - Ran `git status --short; rg --files -g 'AGENTS.md' -g '*045*'; ls -d 04*; cat 045-find-best-scans.py`
> - Ran `ls -ld 040-scan-pages* 045-find-best-scans; realpath 045-find-best-scans; ls -l 040-scan-pages/171.tiff; python -c 'import cv2,numpy,psutil,PIL,tqdm'; for f in AGENTS.md ../AGENTS.md .agents/AGENTS.md; do if test -f "$f"; then cat "$f"; fi; done; rg -n 'median|column|scanline' 042-find-broken-scans.py | head -35`
> - Ran `sed -n '55,200p;330,380p' 042-find-broken-scans.py; ls -a | rg 'venv|nix'; which nix nix-shell uv; ls /tmp | head -50; python -c 'import numpy,PIL,psutil,tqdm'`
> - Ran `ls -d /nix/store/*python*{opencv,numpy,pillow,psutil,tqdm}*; cat shell.nix | rg -n 'opencv|python|psutil|tqdm'; python --version; du -shL 045-find-best-scans`
> - Ran `cat > /tmp/hocr-score-env <<'EOF'
> export PYTHONDONTWRITEBYTECODE=1
> export PYTHONPATH=$(printf '%s\n' /nix/store/{w17cm*,dqlkb*,2fwiim*,10xrz*,0bksf*}/lib/python3.14/site-packages | paste -sd:)
> EOF
> source /tmp/hocr-score-env
> python - <<'PY'
> import cv2, runpy, numpy as np
> m=runpy.run_path('045-find-best-scans.py')
> pages=m['discover_sources'](m['SRC'])
> for n in [16,18,20,22,24,26,30,32,34,38,40,42,163,171]:
>   print(n)
>   for p in pages[n]:
>     g=m['load_gray'](p)
>     h,w=g.shape
>     d=max(2,round(w*.004))
>     bx=max(d, int(.015*w))
>     a=g[int(.015*h):int(.985*h):4].astype(np.int16)
>     c=a[:,bx:w-bx]
>     delta=np.minimum(a[:,bx-d:w-bx-d],a[:,bx+d:w-bx+d])-c
>     strength=np.maximum(np.percentile(delta,35,axis=0)-2,0)
>     peak=np.argmax(strength)
>     print(p.parent.name[-10:],round(m['score_image'](p)['score'],3),round(strength.sum()),strength[peak],round((peak+bx)/w,3), g.shape)
> PY`
> - Ran `ls -R /nix/store/w17cm* | head -35; ls /nix/store/10d89*/lib; ls /nix/store/dqlkb*/lib; cat /tmp/hocr-score-env`
> - Ran `ls -d /nix/store/*/lib/python3.14/site-packages/cv2 | head; python -c 'import sys; print(sys.version)'`
> - Ran `cat /nix/store/w17cm*/nix-support/propagated-build-inputs; ls -d /nix/store/*opencv* | grep -vE 'drv$|python'`
> - Ran `ls -d /nix/store/*-opencv-*/lib/python3.14/site-packages/cv2; ls -l 045-find-best-scans/045-find-best-scan.csv; df -h /tmp`
> - Ran `sed -i 's/w17cm\*/xgmb0*/' /tmp/hocr-score-env
> cat >/tmp/hocr-probe.py <<'PY'
> import runpy, numpy as np
> m=runpy.run_path('045-find-best-scans.py')
> pages=m['discover_sources'](m['SRC'])
> for n in [16,18,20,22,24,26,30,32,34,38,40,42,163,171]:
>   print(n)
>   for p in pages[n]:
>     g=m['load_gray'](p)
>     h,w=g.shape
>     d=max(2,round(w*.004))
>     bx=max(d, int(.015*w))
>     a=g[int(.015*h):int(.985*h):4].astype(np.int16)
>     c=a[:,bx:w-bx]
>     delta=np.minimum(a[:,bx-d:w-bx-d],a[:,bx+d:w-bx+d])-c
>     strength=np.maximum(np.percentile(delta,35,axis=0)-2,0)
>     peak=np.argmax(strength)
>     print(p.parent.name[-10:],round(m['score_image'](p)['score'],3),round(strength.sum()),strength[peak],round((peak+bx)/w,3), g.shape)
> PY
> source /tmp/hocr-score-env
> python /tmp/hocr-probe.py`
> - Ran `source /tmp/hocr-score-env
> sed -n '1,4p' 045-find-best-scans/045-find-best-scan.csv
> OPENBLAS_NUM_THREADS=1 python -u /tmp/hocr-probe.py`
> - Wrote `./045-find-best-scans.py` (+31 -19)
>
> </details>
>
> The old score downsampled the scans and averaged column differences across the page, which diluted narrow streaks. The revised score measures persistent dark columns at their original width. I’ll check which TIFFs it selects for all 14 reported pages.
>
> <details><summary>Edited a file, ran 7 commands</summary>
>
> - Ran `source /tmp/hocr-score-env
> test_dir=$(mktemp -d /tmp/hocr-best-scans.XXXXXX)
> printf '%s\n' "$test_dir" > /tmp/hocr-best-scans-test-dir
> for dir in 040-scan-pages 040-scan-pages.tmp.*; do
>   mkdir "$test_dir/$dir"
>   for page in 016 018 020 022 024 026 030 032 034 038 040 042 163 171; do
>     ln -s "$(realpath "$dir/$page.tiff")" "$test_dir/$dir"
>   done
> done
> mkdir "$test_dir/output"
> touch -r 045-find-best-scans/171.tiff "$test_dir/output/171.tiff"
> OPENBLAS_NUM_THREADS=1 python 045-find-best-scans.py "$test_dir/040-scan-pages" --dst "$test_dir/output" --workers 4
> cat "$test_dir/output/045-find-best-scan.csv"
> OPENBLAS_NUM_THREADS=1 python 045-find-best-scans.py "$test_dir/040-scan-pages" --dst "$test_dir/output" --workers 4`
> - Ran `source /tmp/hocr-score-env
> test_dir=$(cat /tmp/hocr-best-scans-test-dir)
> cat > "$test_dir/run.py" <<'PY'
> import multiprocessing
> import runpy
> import sys
>
> sys.argv.pop(0)
> if __name__ == '__main__':
>     multiprocessing.set_start_method('spawn')
> runpy.run_path(sys.argv[0], run_name=__name__)
> PY
> OPENBLAS_NUM_THREADS=1 python "$test_dir/run.py" "$PWD/045-find-best-scans.py" "$test_dir/040-scan-pages" --dst "$test_dir/output" --workers 4
> cut -d, -f1-6 "$test_dir/output/045-find-best-scan.csv"
> OPENBLAS_NUM_THREADS=1 python "$test_dir/run.py" "$PWD/045-find-best-scans.py" "$test_dir/040-scan-pages" --dst "$test_dir/output" --workers 4`
> - Ran `git diff --check; git diff --stat; git diff -- 045-find-best-scans.py`
> - Ran `awk -F, '$1 ~ /^(16|18|20|22|24|26|30|32|34|38|40|42|163|171)$/{print $1,$3}' 045-find-best-scans/045-find-best-scan.csv
> test_dir=$(cat /tmp/hocr-best-scans-test-dir)
> md5sum "$test_dir/output/171.tiff" 040-scan-pages.tmp.1795621073/171.tiff
> git diff --check -- 045-find-best-scans.py`
> - Ran `source /tmp/hocr-score-env
> cat >/tmp/hocr-probe-local.py <<'PY'
> import runpy,numpy as np
> m=runpy.run_path('045-find-best-scans.py')
> pages=m['discover_sources'](m['SRC'])
> for n in (30,163,171):
>  for p in pages[n]:
>     g=m['load_gray'](p)
>     h,w=g.shape; d=13; a=g[79:-79:5].astype(np.int16)
>     c=a[:,50:-50]; l=a[:,50-d:-50-d]; r=a[:,50+d:-50+d]
>     print(n,p.parent.name)
>     for polarity,delta in [('dark',np.minimum(l,r)-c),('light',c-np.maximum(l,r))]:
>         for q in (35,50,65,75):
>             s=np.maximum(np.percentile(delta,q,axis=0)-2,0)
>             lo,hi=(.94,.98) if n==30 else (.75,.82)
>             local=s[int(lo*w)-50:int(hi*w)-50]
>             print(polarity,q,round(float(np.sum(s*s)/w),3),local.max(),round(np.argmax(local)/w+lo,4),end=' | ')
>         print()
> PY
> OPENBLAS_NUM_THREADS=1 python -u /tmp/hocr-probe-local.py`
> - Wrote `./045-find-best-scans.py` (+4 -4)
> - Ran `source /tmp/hocr-score-env
> test_dir=$(cat /tmp/hocr-best-scans-test-dir)
> OPENBLAS_NUM_THREADS=1 python "$test_dir/run.py" "$PWD/045-find-best-scans.py" "$test_dir/040-scan-pages" --dst "$test_dir/output" --workers 4
> cut -d, -f1-6 "$test_dir/output/045-find-best-scan.csv"`
> - Ran `source /tmp/hocr-score-env
> OPENBLAS_NUM_THREADS=1 python - <<'PY'
> import runpy
> import numpy as np
> m = runpy.run_path('045-find-best-scans.py')
> score = m['vertical_line_score']
> clean = np.full((1000, 3000), 255, dtype=np.uint8)
> thin = clean.copy()
> thin[:, 450] = 246
> thick = clean.copy()
> thick[100:550, 2400:2407] = 165
> edge = clean.copy()
> edge[:, 2860:] = 165
> short = clean.copy()
> short[150:250, 2400:2407] = 165
> assert score(clean) == score(edge) == score(short) == 0
> assert 0 < score(thin) < score(thick)
> for n in (30,163,171):
>     results = [m['score_image'](p) for p in m['discover_sources'](m['SRC'])[n]]
>     print(n, [(r['path'], round(r['score'], 3)) for r in sorted(results, key=lambda r:r['score'])])
>     assert max(results, key=lambda r:r['score'])['path'].startswith('040-scan-pages.tmp.1795622484/')
> PY
> git diff --check -- 045-find-best-scans.py`
>
> </details>
</details>

Fixed [045-find-best-scans.py](./045-find-best-scans.py). It now scores vertical streaks at the original column resolution and penalizes darker, wider streaks even when they fade along the page.

Verified all 14 reported pages against their four source scans. Each selected a different TIFF from the existing output. For page 171, all three alternatives outranked `040-scan-pages.tmp.1795622484/171.tiff`, and the script selected `040-scan-pages.tmp.1795621073/171.tiff`.

Verification used a temporary output directory. The next normal run will rescore the existing outputs automatically.

# Remove shadows from scanned top edge

> fix the python script 065-remove-page-borders.py
>
> currently it fails to remove dark shadows around the scanned top edge, probably because these shadows are darker than the midgray scanner background. the dark shadow can go from midgray to pure black (#000000).
>
> example input image: 060-rotate-crop/007.tiff
>
> example output image: 065-remove-page-borders/007.tiff (on the left side of the top edge, the 065 script correctly removes the midgray scanner background, but on the right side of the top edge, the 065 script fails to remove the black shadow, which has a height of up to 10 pixels)
>
> the 065 script receives the rotated version of the scanned image. in the case of `config.scan_top_edge = "top"` the scanned image was not rotated, so here, the scanned top edge is the input image top edge.
>
> currently i am not sure whether we should generalize this "remove dark shadows" cleanup to all page edges, or whether we should limit this special cleanup to the scanned top edge. for future changes, it would be nice to have a config to select the physical scanned page edges for this "remove dark shadows" cleanup. for now, it should be enabled only for the scanned top edge. (for example, if `config.scan_top_edge = "inside"`, then the scanned top edge becomes the left edge on odd pages and the right edge on even pages).
>
> in the future, this will become painful with "borderless printed" pages... but i dont want to solve that problem now.

<details><summary>6 previous messages</summary>

> I’ll inspect how the script detects borders and maps the scanned top edge after rotation, then test the fix on page 007.
>
> <details><summary>Explored a search, ran 6 commands</summary>
>
> - Ran `git status --short; rg --files --hidden -g 'AGENTS.md' -g '*065*' -g '*config*' -g '007.tiff'; ls -la`
>
> - Ran `rg -n '^(def |class )|scan_top_edge|scanner_background' 065-remove-page-borders.py _shared.py 000-config.py
> ls -l {060-rotate-crop,065-remove-page-borders}/007.tiff
> for d in . .. ../.. ../../..; do if test -f "$d/AGENTS.md"; then cat "$d/AGENTS.md"; fi; done`
>
> <details><summary>Explored a search</summary>
>
> - Searched for `gray_low|gray_high|gray_min|gray_max|gray_mask|outside_mask` in `065-remove-page-borders.py`
>
> </details>
>
> - Ran `sed -n '1915,2148p;110,174p;3608,3645p' 065-remove-page-borders.py
> python -c 'import cv2,numpy,psutil,tqdm'`
>
> - Ran `command -v nix-shell magick convert identify uv
> ls /usr/bin/python*; /usr/bin/python3 -c 'import cv2,numpy,psutil,tqdm'
> rg -n 'dpi|scan_top_edge' 000-config.py
> sed -n '3260,3360p' 065-remove-page-borders.py
> ls /nix/store | rg 'opencv|python3-3' | head -25
> identify {060-rotate-crop,065-remove-page-borders}/007.tiff`
>
> - Ran `cat shell.nix
> ls -d /nix/store/*-python3*-opencv* | rg -v '\.drv' | head -30
> python - <<'PY'
> import subprocess
> for path,w in [('060-rotate-crop/007.tiff',3354),('065-remove-page-borders/007.tiff',3107)]:
>  p=subprocess.check_output(['magick',path,'-crop',f'{w}x40+0+0','-depth','8','rgba:-'])
>  print(path)
>  for x in range(100,w,250):
>   print(x,[p[(y*w+x)*4] if p[(y*w+x)*4+3]==255 else 'A' for y in range(24)])
> PY`
>
> - Ran `sed -n '45,65p' 000-config.py; sed -n '1,45p;110,145p' _shared.py; rg -n 'rotate|EDGE_TOP|EDGE_LEFT' 060-rotate-crop.py 065-remove-page-borders.py | head -40
> ls -d /nix/store/*python3.14-{numpy,pillow,psutil,tqdm}-* | rg -v '\.drv' | head -20
> python --version`
>
> </details>
>
> The active cleanup skips the scanned top edge. Page 007 also has profiles that get darker before they reach white paper, so I’ll use a bounded scan from that edge that allows those dips.
>
> <details><summary>Edited 2 files, explored a file, a search, ran 9 commands</summary>
>
> - Wrote `./065-remove-page-borders.py` (+42 -5)
>
> - Wrote `./000-config.py` (+3 -0)
>
> - Ran `cat > /tmp/hocr-065-python <<'EOF'
> #!/usr/bin/env bash
> for package in /nix/store/{10d89jlmvr2i0495iff5d5m8qvz1kmc0,dqlkbzknp8j79084x37a5ac6qfbyqjqs,2fwiimdk8289pp3fd0k413dyj05q85ll,10xrzbbi4sj4va3jny7rcpji6mp0pp28,0bksfalic5kr2payqrh0y6b9rm60sj39}-python3.14-*; do
>     export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$package/lib/python3.14/site-packages"
> done
> exec python -B "$@"
> EOF
> bash /tmp/hocr-065-python - <<'PY'
> from pathlib import Path
> import runpy
> import tempfile
> import cv2
> import numpy as np
> m = runpy.run_path('065-remove-page-borders.py')
> config = m['config']
> cv2.setNumThreads(1)
> out = Path(tempfile.mkdtemp(prefix='hocr-065-'))
> src = Path('060-rotate-crop/007.tiff')
> for enabled in (False, True):
>  config.clean_scan_top_shadow = enabled
>  m['random'].seed(42)
>  m['process_image'](src, out / f'{enabled}.tiff')
> a,b = (cv2.imread(out / f'{enabled}.tiff', cv2.IMREAD_UNCHANGED) for enabled in (False, True))
> print(out, a.shape,b.shape)
> changed = np.any(a!=b,axis=-1)
> print('changes by depth', changed.sum(axis=1)[:16], 'total',changed.sum())
> for label,img in [('before',a),('after',b)]:
>  print(label, 'opaque dark (<128) in top 12 right half:', ((img[:12,1550:,0]<128) & (img[:12,1550:,3]==255)).sum())
> PY`
>
> - Ran `ls /nix/store/10d89*-python*/lib; ls /nix/store/w17cm*-python*/lib/python3.14/site-packages
> which python; find /nix/store/10d89*-python* -name 'cv2*' | head; ls -d /nix/store/*python3-3.14.7/bin/python3 | head`
>
> - Ran `find /nix/store -mindepth 1 -maxdepth 1 -iname '*opencv*' ! -name '*.drv' | while read -r p; do find "$p/lib" -iname 'cv2*' 2>/dev/null; done
> cat /nix/store/10d89*-python*/lib/python3.14/site-packages/*/RECORD | head -10`
>
> - Ran `sed -i 's/10d89jlmvr2i0495iff5d5m8qvz1kmc0,//; s@exec python@export PYTHONPATH="/nix/store/gr9mxapvqlj50arjwbahi3bq9xs9bgkw-opencv4-python-4.14.0/lib/python3.14/site-packages:$PYTHONPATH"\nexec python@' /tmp/hocr-065-python
> cat > /tmp/hocr-065-check <<'PY'
> from pathlib import Path
> import runpy
> import tempfile
> import cv2
> import numpy as np
> m = runpy.run_path('065-remove-page-borders.py')
> config = m['config']
> cv2.setNumThreads(1)
> out = Path(tempfile.mkdtemp(prefix='hocr-065-'))
> src = Path('060-rotate-crop/007.tiff')
> for enabled in (False, True):
>  config.clean_scan_top_shadow = enabled
>  m['random'].seed(42)
>  m['process_image'](src, out / f'{enabled}.tiff')
> a,b = (cv2.imread(out / f'{enabled}.tiff', cv2.IMREAD_UNCHANGED) for enabled in (False, True))
> print(out, a.shape,b.shape)
> changed = np.any(a!=b,axis=-1)
> print('changes by depth', changed.sum(axis=1)[:16], 'total',changed.sum())
> for label,img in [('before',a),('after',b)]:
>  print(label, 'opaque dark (<128) in top 12 right half:', ((img[:12,1550:,0]<128) & (img[:12,1550:,3]==255)).sum())
> PY
> bash /tmp/hocr-065-python - < /tmp/hocr-065-check`
>
> - Ran `sed -i 's/gr9mxapvqlj50arjwbahi3bq9xs9bgkw-opencv4-python-4.14.0/xgmb0vcwsy39pvj8fc9d44gghvbbjwyp-opencv-4.13.0/' /tmp/hocr-065-python
> bash /tmp/hocr-065-python - < /tmp/hocr-065-check`
>
> - Wrote `./065-remove-page-borders.py` (+2 -1)
>
> - Ran `bash /tmp/hocr-065-python - <<'PY'
> import runpy
> import numpy as np
> m = runpy.run_path('065-remove-page-borders.py')
> cleanup = m['remove_background_outside_page']
> identity = np.eye(3, dtype=np.float32)
> for setting, odd, k in [('top', True, 0), ('bottom', True, 2), ('inside', True, 1), ('inside', False, 3)]:
>     page = np.full((24, 30, 4), 255, np.uint8)
>     page[:10, 4:10, :3] = np.array([128, 30, 0, 0, 0, 15, 40, 95, 175, 235])[:, None, None]
>     page[:13, 14:17, :3] = 0
>     page[:2, 20, :3] = (0, 0, 200)
>     page[-4:, 4:10, :3] = 0
>     expected = page.copy()
>     expected[:10, 4:10] = 0
>     rotated = np.rot90(page, k).copy()
>     actual = cleanup(rotated, {}, identity[:2], identity, odd, setting, transparent=True)
>     np.testing.assert_array_equal(actual, np.rot90(expected, k))
> print('verified top, bottom, and inside on odd/even pages, with deep and colored pixels preserved')
> PY
> git diff --check`
>
> - Ran `bash /tmp/hocr-065-python - <<'PY'
> import cv2,numpy as np
> a=cv2.imread('/tmp/hocr-065-s1yglwde/True.tiff',-1)
> ys,xs=np.where((a[:12,1550:,0]<128)&(a[:12,1550:,3]==255))
> xs+=1550
> print(list(zip(xs.tolist(),ys.tolist())))
> for x in np.unique(xs)[::4]:
>  print(x,a[:22,x,0],a[:22,x,3])
> PY`
>
> - Wrote `./065-remove-page-borders.py` (+4 -1)
>
> - Ran `bash /tmp/hocr-065-python - < /tmp/hocr-065-check`
>
> - Ran `git diff --check
> git diff -- 065-remove-page-borders.py
> git diff -- 000-config.py | tail -35`
>
> <details><summary>Explored a file, a search</summary>
>
> - Read `./_shared.py`
> - Searched for `clean_scan_top_shadow|def mask_scan_top_shadow` in `000-config.py`
>
> </details>
>
> </details>
>
> The first comparison removed nearly all of the dark pixels from the right half of page 007’s top strip. I’ve adjusted the white-paper threshold for a few uneven spots and am checking the result. Synthetic checks passed for both odd and even `inside` pages.
>
> <details><summary>Ran a command</summary>
>
> - Ran `bash /tmp/hocr-065-python - <<'PY'
> from pathlib import Path
> import cv2
> import numpy as np
> out = Path('/tmp/hocr-065-hgunb17a')
> before = cv2.imread(out / 'False.tiff', -1)
> after = cv2.imread(out / 'True.tiff', -1)
> assert np.array_equal(before[12:], after[12:])
> y,x = np.where((after[:12,1550:,0]<128) & (after[:12,1550:,3]==255))
> print('remaining dark pixels at x=', np.unique(x + 1550))
> (out / 'True.tiff').rename(out / '007.tiff')
> PY
> git diff --check`
>
> </details>
</details>

Fixed [065-remove-page-borders.py](065-remove-page-borders.py) to remove shadows that dip to pure black along the scanned top edge. With `scan_top_edge = "inside"`, cleanup follows the left edge on odd pages and the right edge on even pages. Set `clean_scan_top_shadow = False` in [000-config.py](000-config.py) to disable it.

On [reprocessed page 007](/tmp/hocr-065-hgunb17a/007.tiff), dark opaque pixels in the right half of the top 12 rows dropped from 5,570 to 16, all in the outermost three columns. Pixels below that strip were identical. The existing output TIFF was left in place.

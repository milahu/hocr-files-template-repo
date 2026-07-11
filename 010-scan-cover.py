#!/usr/bin/env python3

import os
import subprocess
import sys
import time
import shlex
from pathlib import Path


# JPEG: compressed, lossy
# scan_format = "jpg"

# TIFF: uncompressed, lossless
scan_format = "tiff"

scan_resolution = 600

scan_mode = "Color"


def main():
    if len(sys.argv) != 2:
        print(f"Error: Not enough arguments")
        print(f"Usage: {sys.argv[0]} <scanner_name>")
        print(f'Example: {sys.argv[0]} "genesys:libusb:003:013"')
        print('Hint: To get your scanner name, run: scanimage -L')
        sys.exit(1)

    scanner_name = sys.argv[1]

    script_path = Path(__file__).resolve()
    script_dir = script_path.parent
    dst = Path(script_path.stem)

    os.chdir(script_dir)

    dst.mkdir(parents=True, exist_ok=True)

    # here we assume:
    # - each scan takes longer than one second
    # - only one scanner is writing to dst
    output_file = dst / f"{int(time.time())}.{scan_format}"

    scanimage_scan_format = scan_format
    if scanimage_scan_format == "jpg":
        # scanimage expects "jpeg"
        scanimage_scan_format = "jpeg"

    cmd = [
        "scanimage",
        f"--device-name={scanner_name}",
        f"--resolution={scan_resolution}",
        f"--format={scanimage_scan_format}",
        f"--mode={scan_mode}",
        f"--output-file={output_file}",
        "--progress",
    ]

    print("+", shlex.join(cmd))

    if True:
        # simple but correct
        # wait until scanimage returns
        # the output image file is finalized
        # on the very end of the scanimage process
        # so we just... have... to... wait...
        subprocess.run(cmd, check=True)

    else:
        # fancy but wrong
        # wait for "Progress: 100.0%\r" in proc.stdout
        # and print the output file path as soon as possible
        # as soon as the file has been completely written to disk
        # then exit the script
        # but keep scanimage running in the background
        # so it can reset the scanner for the next scan
        # use setsid to run scanimage in a separate process group
        # so when we exit this script
        # scanimage keeps running in the background
        # TODO why does this fail?
        # if we stop reading the stdout/stderr of scanimage
        # then scanimage produces broken output image files
        # http://www.sane-project.org/mailing-lists.html

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,  # binary mode to preserve "\r"
            bufsize=0,  # unbuffered pipe object
            start_new_session=True,  # setsid
        )

        assert proc.stdout is not None

        # wait for the "progress 100%" output
        # progress_100 = b"\rProgress: 100.0%"
        progress_100 = b'Progress: 100.0%\r'

        # buf = bytearray()

        scan_done = False

        last_out_size = 0

        while True:
            # chunk = b'Progress: 1.0%\r'
            # ...
            # chunk = b'Progress: 100.0%\r'

            chunk = proc.stdout.read(4096)
            if not chunk:
                break

            out_size = os.path.getsize(output_file)

            if 1:
                sys.stdout.buffer.write(b"size: " + str(out_size).encode("ascii") + b"  ")
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
            else:
                print(f"chunk={chunk!r} out_size={out_size}")

            last_out_size = out_size

            if scan_done:
                continue

            # buf.extend(chunk)

            # if progress_100 in buf:
            if chunk == progress_100:
                # progress 100%
                scan_done = True
                # the scan is complete
                # but the output image is still broken at this point
                print()
                # print(output_file)
                # no. this produces broken image files
                if False:
                    # Stop reading; scanimage continues in the background.
                    proc.stdout.close()
                break

            r'''
            # keep only enough history to match across chunk boundaries
            if len(buf) > len(progress_100):
                del buf[:-len(progress_100)]
            '''

        # no. scanimage must run until the very end to finalize the image file
        # wait for complete output file
        while True:
            out_size = os.path.getsize(output_file)
            try:
                stdout, _stderr = proc.communicate(timeout=1)
                if not proc.returncode is None:
                    if stdout != b'Progress: 100.0%\r':
                        print(f"size: {out_size}  done  stdout={stdout!r}")
                    else:
                        print(f"size: {out_size}  done")
                    break
                else:
                    # this is never reached?
                    if out_size != last_out_size:
                        print(f"size: {out_size}  finishing  stdout={stdout!r}")
            except subprocess.TimeoutExpired:
                if out_size != last_out_size:
                    print(f"size: {out_size}  finishing")
            last_out_size = out_size

    print(f"TODO manually rotate and crop {output_file}")
    print("hint:")
    print(f"  gimp {output_file}")


if __name__ == "__main__":
    main()

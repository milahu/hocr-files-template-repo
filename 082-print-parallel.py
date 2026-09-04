#!/usr/bin/env python3

"""
print a PDF document with multiple printers in parallel
"""

# TODO add incremental printing:
# render pages in subchunks of 2 pages
# (or subchunks of 4 pages with nup2)
# and send each chunk separately to the printer
# so the printer can start printing as soon as possible
# currently we render all pages of a chunk
# and then send the complete chunk to the printer
# which takes lots of CPU time before the first page is printed
# and later, when all pages are rendered, CPU usage drops to zero
# so ideally, we want to distribute the CPU load over the whole print process

import argparse
import hashlib
import os
import subprocess
import sys
import time
import re
import tempfile
import shlex
import json

import pymupdf


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

# Delay between starting print jobs.
T1 = 0.1

# Cache directory suffix.
DEFAULT_CACHE_SUFFIX = ".print-cache"

default_printers_json = "~/.config/printers.json"


# ----------------------------------------------------------------------
# Page filter
# ----------------------------------------------------------------------

class PageFilter:
    # condition types
    LE = 0
    GE = 1
    RANGE = 2
    EQ = 3

    # modes
    MODE_CUSTOM = 0
    MODE_NONE = 1
    MODE_ALL = 2

    def __init__(self, spec: str):
        spec = spec.strip()

        self.mode = self.MODE_CUSTOM
        self.conditions = []

        if not spec or spec == "none":
            self.mode = self.MODE_NONE
            return

        if spec == "all":
            self.mode = self.MODE_ALL
            return

        for part in spec.split(","):
            part = part.strip()

            if not part:
                continue

            if "-" in part:

                if part.startswith("-") and part != "-":
                    end = int(part[1:])
                    self.conditions.append(
                        (self.LE, end)
                    )

                elif part.endswith("-"):
                    start = int(part[:-1])
                    self.conditions.append(
                        (self.GE, start)
                    )

                else:
                    start, end = part.split("-", 1)

                    start = int(start)
                    end = int(end)

                    if start > end:
                        raise ValueError(
                            f"Invalid range: {part}"
                        )

                    self.conditions.append(
                        (self.RANGE, start, end)
                    )

            else:
                value = int(part)

                self.conditions.append(
                    (self.EQ, value)
                )

    def __call__(self, page: int) -> bool:
        if self.mode == self.MODE_NONE:
            return False

        if self.mode == self.MODE_ALL:
            return True

        for cond in self.conditions:

            kind = cond[0]

            if kind == self.LE:

                if page <= cond[1]:
                    return True

            elif kind == self.GE:

                if page >= cond[1]:
                    return True

            elif kind == self.RANGE:

                if cond[1] <= page <= cond[2]:
                    return True

            else:  # self.EQ

                if page == cond[1]:
                    return True

        return False


def make_filter_page(spec: str):
    return PageFilter(spec)


def parse_page_sequence(
    spec,
    page_count,
):
    """
    Parse a logical page specification into an ordered sequence.

    Page zero is a special logical page and may occur multiple times.

    Examples:

        1-10
            -> [1, 2, ..., 10]

        0-10
            -> [0, 1, 2, ..., 10]

        0,1-5,0,15-20
            -> [0, 1, 2, 3, 4, 5, 0, 15, ..., 20]

        100-
            -> [100, 101, ..., page_count]

        -100
            -> [1, 2, ..., 100]

    Positive and negative page numbers refer to original PDF pages.
    Zero refers to the synthetic logical page zero.

    Returns:

        list[int]

    where:
        0     = logical zero page
        > 0   = one-based original PDF page number
    """

    spec = spec.strip()

    if not spec or spec == "none":
        return []

    if spec == "all":
        return list(
            range(
                1,
                page_count + 1,
            )
        )

    pages = []

    for part in spec.split(","):

        part = part.strip()

        if not part:
            continue

        # ----------------------------------------------------------
        # Single page
        # ----------------------------------------------------------

        if "-" not in part:

            page = int(part)

            if page == 0:
                pages.append(0)
                continue

            # Validate the original PDF page number.
            resolve_pdf_page_number(
                page,
                page_count,
            )

            # Convert negative page numbers to positive
            # one-based page numbers so the resulting sequence
            # is unambiguous.
            if page < 0:
                page = (
                    page_count
                    + page
                    + 1
                )

            pages.append(page)

            continue

        # ----------------------------------------------------------
        # Range
        # ----------------------------------------------------------

        if part == "-":
            raise ValueError(
                "invalid page range: '-'"
            )

        if part.startswith("-"):

            # Examples:
            #
            #   -10
            #
            # means pages 1-10.
            #
            # A negative range such as -10--1 is intentionally
            # not supported by the existing syntax.

            end = int(
                part[1:]
            )

            if end < 0:
                raise ValueError(
                    f"invalid page range: {part}"
                )

            if end == 0:
                pages.append(0)
                continue

            end = min(
                end,
                page_count,
            )

            pages.extend(
                range(
                    1,
                    end + 1,
                )
            )

            continue

        if part.endswith("-"):

            start = int(
                part[:-1]
            )

            if start < 0:
                raise ValueError(
                    f"invalid page range: {part}"
                )

            if start == 0:
                start = 0

            if start > page_count:
                raise ValueError(
                    f"page range {part} "
                    f"is outside the PDF page range"
                )

            pages.extend(
                range(
                    start,
                    page_count + 1,
                )
            )

            continue

        start, end = part.split(
            "-",
            1,
        )

        start = int(start)
        end = int(end)

        if start > end:
            raise ValueError(
                f"Invalid range: {part}"
            )

        if start < 0 or end < 0:
            raise ValueError(
                f"invalid page range: {part}"
            )

        if end > page_count:
            raise ValueError(
                f"page range {part} "
                f"is outside the PDF page range"
            )

        pages.extend(
            range(
                start,
                end + 1,
            )
        )

    return pages







# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Split a PDF into page ranges and print each range "
            "on a separate printer."
        )
    )

    parser.add_argument(
        "pdf",
        help="input PDF file",
    )

    parser.add_argument(
        "--nup2",
        action="store_true",
        help=(
            "print 2 PDF pages per physical A4 page, "
            "with odd pages on the right and even pages "
            "on the left; uses two-sided short-edge printing"
        ),
    )

    parser.add_argument(
        "--printers-json", # args.printers_json
        default=default_printers_json,
        help=(
            'path to printers.json file.'
            f' default: {default_printers_json}.'
            ' example content:'
            ' ["Printer1", "Printer2"]'
        ),
    )

    zero_group = parser.add_mutually_exclusive_group()

    zero_group.add_argument(
        "--zero-page",
        type=int,
        metavar="PAGE",
        help=(
            "use this page from the input PDF as page zero; "
            "negative values count from the end "
            "(e.g. -1 is the last page)"
        ),
    )

    zero_group.add_argument(
        "--zero-image",
        metavar="IMAGE",
        help=(
            "use this image as logical page zero (e.g. zero.avif)"
        ),
    )

    parser.add_argument(
        "--pages",
        default=None,
        metavar="PAGES_SPEC",
        help=(
            "select logical pages to print; page zero is a special "
            "blank/zero page that may be inserted anywhere; examples: "
            "'all', 'none', '1-10', '5,10,20', "
            "'0-10', '0,1-5,0,15-20', "
            "'100-', '-100', '1-10,50,100-'"
        ),
    )

    # TODO also cache temporary PDF files
    # "if False:"
    parser.add_argument(
        "--cache",
        action="store_true",
        help=(
            "cache the printer raster output to avoid repeating "
            "expensive PDF-to-raster conversion"
        ),
    )

    parser.add_argument(
        "--cache-dir",
        metavar="DIR",
        help=(
            "directory for cached PPD and raster files "
            "(default: <PDF>.print-cache)"
        ),
    )

    args = parser.parse_args()

    if args.pages is None:

        if args.nup2:
            args.pages = "0-"
        else:
            args.pages = "1-"

    return args


# ----------------------------------------------------------------------
# Printing options
# ----------------------------------------------------------------------

def get_lp_options(nup2):
    """
    Return CUPS options for the selected print mode.

    The PDF has already been transformed to 2-up by
    create_nup2_pdf() when nup2=True, so CUPS must NOT
    apply number-up=2 again.
    """

    options = [
        "-o", "PageSize=A4",
        "-o", "PrintQuality=600dpi",
        "-o", "scaling=100",
        "-o", "page-border=none",
    ]

    if nup2:
        # The PDF is already 2-up.
        #
        # Each PDF page is already a full A4 page containing
        # the desired content. Therefore CUPS must print
        # one PDF page per physical A4 side.
        options += [
            "-o", "sides=two-sided-short-edge",
        ]

    else:
        options += [
            "-o", "sides=two-sided-long-edge",
        ]

    return options


# ----------------------------------------------------------------------
# Hashing
# ----------------------------------------------------------------------

def sha256_file(path, chunk_size=1024 * 1024):
    """
    Calculate SHA-256 checksum of a file.
    """

    digest = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            data = f.read(chunk_size)

            if not data:
                break

            digest.update(data)

    return digest.hexdigest()


# ----------------------------------------------------------------------
# Page handling
# ----------------------------------------------------------------------

def resolve_pdf_page_number(
    page_number,
    page_count,
):
    """
    Convert a user-facing page number to a zero-based PyMuPDF index.

    Positive numbers are one-based:

        1   -> first page
        2   -> second page
        123 -> page 123

    Negative numbers count backwards:

        -1 -> last page
        -2 -> second-to-last page

    Zero is a logical/synthetic page and must be handled by the
    caller. It is not an original PDF page.
    """

    if page_number == 0:
        raise ValueError(
            "page zero is a logical page and has no original PDF index"
        )

    if page_number > 0:
        page_index = page_number - 1

    else:
        page_index = (
            page_count + page_number
        )

    if not 0 <= page_index < page_count:
        raise ValueError(
            f"--zero-page {page_number} "
            f"is outside the PDF page range"
        )

    return page_index


def normalize_pdf_for_printing(
        input_pdf,
        output_pdf,
    ):
    """
    Normalize PDF pages for printing.

    Landscape pages are rotated into portrait orientation.

    This handles both:
        - pages with an explicit PDF rotation
        - pages whose actual page geometry is landscape

    The page contents are preserved using show_pdf_page().
    """

    source = pymupdf.open(input_pdf)
    output = pymupdf.open()

    did_change = False

    for index, source_page in enumerate(source):

        rect = source_page.rect

        # A page is considered landscape when its effective
        # displayed width is greater than its height.
        landscape = (
            rect.width > rect.height
        )

        if not landscape:
            # Normal portrait page: copy unchanged.
            output.insert_pdf(
                source,
                from_page=index,
                to_page=index,
            )
            continue

        did_change = True

        print(
            f"normalizing landscape page "
            f"{index + 1}: "
            f"{rect.width:.2f} x "
            f"{rect.height:.2f}"
        )

        # ----------------------------------------------------------
        # Create a portrait page with width/height exchanged.
        # ----------------------------------------------------------

        portrait_width = rect.height
        portrait_height = rect.width

        destination_page = output.new_page(
            width=portrait_width,
            height=portrait_height,
        )

        destination_rect = pymupdf.Rect(
            0,
            0,
            portrait_width,
            portrait_height,
        )

        # show_pdf_page() can rotate the source page while
        # placing it into the destination rectangle.
        destination_page.show_pdf_page(
            destination_rect,
            source,
            index,
            rotate=90,
            keep_proportion=True,
        )

    if not did_change:
        output.close()
        source.close()
        return None

    output.save(output_pdf)

    output.close()
    source.close()

    return output_pdf


def create_filtered_pdf(
    input_pdf,
    page_sequence,
    zero_page_index=None,
    zero_image=None,
):
    """
    Create a logical PDF according to an ordered page sequence.

    page_sequence contains:

        0     -> synthetic logical zero page
        > 0   -> one-based original PDF page number

    The zero page may occur multiple times and at arbitrary positions.

    If zero_page_index is not None, that original PDF page is used
    for every logical page zero.

    If zero_image is not None, the image is used for every logical
    page zero.

    If neither is provided, logical page zero is a blank page whose
    size matches the first original PDF page.

    Returns:

        (output_pdf, selected_original_page_count)
    """

    source = pymupdf.open(
        input_pdf
    )

    if len(source) == 0:

        source.close()

        raise ValueError(
            "input PDF has no pages"
        )

    if zero_page_index is not None:

        if not 0 <= zero_page_index < len(source):

            source.close()

            raise ValueError(
                "zero page index is outside "
                "the PDF page range"
            )

    # --------------------------------------------------------------
    # Count actual original PDF pages selected.
    #
    # Logical zero pages do not count as selected original pages.
    # --------------------------------------------------------------

    selected_original_pages = [
        page
        for page in page_sequence
        if page != 0
    ]

    print()
    print(
        f"selected "
        f"{len(selected_original_pages)} "
        f"of {len(source)} original pages"
    )

    if selected_original_pages:

        print(
            f"first selected page: "
            f"{selected_original_pages[0]}"
        )

        print(
            f"last selected page: "
            f"{selected_original_pages[-1]}"
        )

    else:

        print(
            "no original PDF pages selected"
        )

    zero_count = sum(
        1
        for page in page_sequence
        if page == 0
    )

    if zero_count:

        print(
            f"logical zero pages inserted: "
            f"{zero_count}"
        )

    # --------------------------------------------------------------
    # Normalize PDF rotation for printing
    # --------------------------------------------------------------

    normalized_pdf = f"{input_pdf}.normalized.pdf"

    normalized_pdf = normalize_pdf_for_printing(input_pdf, normalized_pdf)

    if normalized_pdf:
        print(f"done normalized_pdf: {normalized_pdf}")
        input_pdf = normalized_pdf
        source = pymupdf.open(input_pdf)

    # --------------------------------------------------------------
    # Create logical output PDF.
    # --------------------------------------------------------------

    output = pymupdf.open()

    for logical_page_number in page_sequence:

        # ----------------------------------------------------------
        # Synthetic logical page zero
        # ----------------------------------------------------------

        if logical_page_number == 0:

            if zero_page_index is not None:

                output.insert_pdf(
                    source,
                    from_page=zero_page_index,
                    to_page=zero_page_index,
                )

            else:

                if len(source) == 0:

                    raise ValueError(
                        "cannot create zero page "
                        "without a source page size"
                    )

                first_page = source[0]

                zero = output.new_page(
                    width=first_page.rect.width,
                    height=first_page.rect.height,
                )

                if zero_image is not None:

                    zero.insert_image(
                        zero.rect,
                        filename=zero_image,
                        keep_proportion=True,
                    )

            continue

        # ----------------------------------------------------------
        # Original PDF page
        # ----------------------------------------------------------

        page_index = (
            logical_page_number - 1
        )

        if not 0 <= page_index < len(source):

            source.close()
            output.close()

            raise ValueError(
                f"page {logical_page_number} "
                f"is outside the PDF page range"
            )

        output.insert_pdf(
            source,
            from_page=page_index,
            to_page=page_index,
        )

    output_pdf = (
        f"{input_pdf}.filtered.pdf"
    )

    print(
        f"writing {output_pdf}"
    )

    output.save(
        output_pdf
    )

    output.close()
    source.close()

    return (
        output_pdf,
        len(selected_original_pages),
    )


# ----------------------------------------------------------------------
# N-up PDF creation
# ----------------------------------------------------------------------

def create_nup2_pdf(
    input_pdf,
    output_pdf,
):
    """
    Create a real 2-up A4 PDF.

    Two input PDF pages are placed side-by-side on one A4 landscape
    physical page.

    Input page 1 -> left
    Input page 2 -> right

    If the input has an odd number of pages, the final physical page
    contains only one logical page.

    The resulting PDF has one PDF page per physical A4 sheet.

    IMPORTANT:

    This function performs the actual 2-up transformation.

    It does NOT rely on CUPS:
        -o number-up=2

    Therefore the resulting PDF can safely be split into chunks
    before printing.
    """

    source = pymupdf.open(
        input_pdf
    )

    if len(source) == 0:
        source.close()

        raise ValueError(
            "no pages available for nup2 printing"
        )

    output = pymupdf.open()

    # A4 in points, landscape.
    #
    # Portrait A4:
    #
    #   595.28 x 841.89
    #
    # Landscape A4:
    #
    #   841.89 x 595.28
    #
    # Each logical page is scaled to fit one half.
    a4_width = 841.89
    a4_height = 595.28

    half_width = (
        a4_width / 2
    )

    print(
        f"creating real 2-up PDF: "
        f"{input_pdf}"
    )

    print(
        f"input logical pages: "
        f"{len(source)}"
    )

    physical_page_count = (
        len(source) + 1
    ) // 2

    print(
        f"output physical A4 pages: "
        f"{physical_page_count}"
    )

    for index in range(
        physical_page_count
    ):

        left_index = (
            index * 2
        )

        right_index = (
            left_index + 1
        )

        # Create one physical A4 landscape page.
        physical_page = (
            output.new_page(
                width=a4_width,
                height=a4_height,
            )
        )

        # ----------------------------------------------------------
        # Left logical page
        # ----------------------------------------------------------

        left_rect = pymupdf.Rect(
            0,
            0,
            half_width,
            a4_height,
        )

        physical_page.show_pdf_page(
            left_rect,
            source,
            left_index,
            keep_proportion=True,
        )

        # ----------------------------------------------------------
        # Right logical page
        # ----------------------------------------------------------

        if (
            right_index
            < len(source)
        ):

            right_rect = pymupdf.Rect(
                half_width,
                0,
                a4_width,
                a4_height,
            )

            physical_page.show_pdf_page(
                right_rect,
                source,
                right_index,
                keep_proportion=True,
            )

    # if os.path.exists(output_pdf):
    if False:
        print(
            f"keeping existing "
            f"{output_pdf}"
        )

    else:

        print(
            f"writing "
            f"{output_pdf}"
        )

        output.save(
            output_pdf
        )

    output.close()
    source.close()

    return output_pdf


# ----------------------------------------------------------------------
# PPD handling
# ----------------------------------------------------------------------

def get_ppd_path(
    printer_name,
):
    return (
        f"/etc/cups/ppd/"
        f"{printer_name}.ppd"
    )


def copy_ppd_with_sudo(
    source_ppd,
    destination_ppd,
):
    """
    Copy a PPD using sudo cat.
    """

    print(
        f"PPD is not readable directly: "
        f"{source_ppd}"
    )

    print(
        f"copying PPD with sudo to: "
        f"{destination_ppd}"
    )

    temp_ppd = (
        f"{destination_ppd}.tmp"
    )

    try:

        if os.path.exists(temp_ppd):
            os.remove(temp_ppd)

        with open(
            temp_ppd,
            "wb",
        ) as output:

            subprocess.run(
                [
                    "sudo",
                    "cat",
                    source_ppd,
                ],
                stdout=output,
                check=True,
            )

        os.replace(
            temp_ppd,
            destination_ppd,
        )

    finally:

        if os.path.exists(temp_ppd):
            os.remove(temp_ppd)


def get_readable_ppd(
    printer_name,
    cache_dir,
):
    """
    Return:

        (ppd_path, ppd_hash)
    """

    source_ppd = get_ppd_path(
        printer_name
    )

    try:

        with open(
            source_ppd,
            "rb",
        ) as f:

            f.read(1)

        ppd_path = (
            source_ppd
        )

    except OSError:

        cached_ppd = os.path.join(
            cache_dir,
            f"{printer_name}.ppd",
        )

        os.makedirs(
            cache_dir,
            exist_ok=True,
        )

        if not os.path.exists(cached_ppd):

            copy_ppd_with_sudo(
                source_ppd,
                cached_ppd,
            )

        ppd_path = (
            cached_ppd
        )

    ppd_hash = sha256_file(
        ppd_path
    )

    return (
        ppd_path,
        ppd_hash,
    )


# ----------------------------------------------------------------------
# Raster cache
# ----------------------------------------------------------------------

def get_raster_cache_path(
    cache_dir,
    pdf_hash,
    ppd_hash,
):
    return os.path.join(
        cache_dir,
        (
            f"{pdf_hash}."
            f"{ppd_hash}."
            f"raster.pwg"
        ),
    )


def create_raster_cache(
    pdf_path,
    ppd_path,
    raster_path,
    lp_options,
):
    """
    Convert PDF to PWG Raster using cupsfilter.
    """

    print()
    print(
        f"writing raster cache: "
        f"{raster_path}"
    )

    temp_raster = f"{raster_path}.pwg"

    if os.path.exists(temp_raster):
        os.remove(temp_raster)

    command = [
        "cupsfilter",
        "-p",
        ppd_path,
        "-m",
        "image/pwg-raster",
        pdf_path,
        *lp_options,
    ]

    print(
        "cupsfilter command:"
    )

    print(
        " ".join(command)
    )

    try:

        with open(
            temp_raster,
            "wb",
        ) as output:

            subprocess.run(
                command,
                stdout=output,
                check=True,
            )

        if os.path.getsize(
            temp_raster
        ) == 0:

            raise RuntimeError(
                "cupsfilter generated "
                "an empty raster file"
            )

        os.replace(
            temp_raster,
            raster_path,
        )

    finally:

        if os.path.exists(temp_raster):
            os.remove(temp_raster)


def get_raster_cache(
    pdf_path,
    ppd_path,
    ppd_hash,
    cache_dir,
    lp_options,
):
    """
    Return cached PWG Raster.

    Cache key:

        SHA256(PDF)
        +
        SHA256(PPD)
    """

    pdf_hash = sha256_file(
        pdf_path
    )

    print(
        f"PDF SHA-256: "
        f"{pdf_hash}"
    )

    raster_path = (
        get_raster_cache_path(
            cache_dir,
            pdf_hash,
            ppd_hash,
        )
    )

    if (
        os.path.exists(raster_path)
        and os.path.getsize(raster_path) == 0
    ):

        print(
            f"clearing empty raster cache: "
            f"{raster_path}"
        )

        os.remove(
            raster_path
        )

    if os.path.exists(raster_path):

        print(
            f"using cached raster: "
            f"{raster_path}"
        )

    else:

        create_raster_cache(
            pdf_path,
            ppd_path,
            raster_path,
            lp_options,
        )

    return raster_path


# ----------------------------------------------------------------------
# IPP printing
# ----------------------------------------------------------------------

def print_raster_ipp(
    printer_ipp_url,
    raster_path,
    copies=1,
):
    """
    Submit a cached PWG Raster file to a printer using IPP.

    ipptool runs detached in the background:

    - setsid creates a new session
    - stdin is connected to /dev/null
    - stdout is connected to /dev/null
    - stderr is connected to /dev/null
    - ipptool continues running after this Python script exits

    The temporary .ipptool file is deleted by the background
    shell after ipptool has finished.
    """

    ipp_print_job = f"""\
{{
  OPERATION Print-Job

  GROUP operation-attributes-tag
    ATTR charset attributes-charset utf-8
    ATTR language attributes-natural-language en
    ATTR uri printer-uri $uri
    ATTR name requesting-user-name "user"
    ATTR name job-name "Broadcast Job"

  GROUP job-attributes-tag
    ATTR integer copies {copies}
    ATTR keyword sides two-sided-short-edge
    ATTR keyword print-scaling none
    ATTR keyword media iso_a4_210x297mm
    ATTR mimeMediaType document-format image/pwg-raster

  FILE {raster_path}
}}
"""

    # Keep the temporary file after this function returns.
    #
    # It must remain available because ipptool is going to read
    # it asynchronously in the background.
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".ipptool",
        delete=False,
    ) as temp_file:

        ipp_job_file = temp_file.name

        temp_file.write(
            ipp_print_job
        )

    print(
        f"starting detached ipptool "
        f"for {printer_ipp_url}"
    )

    # Use a shell so the temporary .ipptool file can be removed
    # after ipptool finishes.
    #
    # The important part is:
    #
    #   setsid ... </dev/null >/dev/null 2>/dev/null
    #
    # This detaches ipptool from the Python process.
    command = (
        "ipptool "
        "-tv "
        f"{shlex.quote(printer_ipp_url)} "
        f"{shlex.quote(ipp_job_file)} "
        "; "
        "rm -f "
        f"{shlex.quote(ipp_job_file)}"
    )

    subprocess.Popen(
        [
            "setsid",
            "sh",
            "-c",
            command,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=False,
    )


def get_cups_printer_uris():
    """
    Return a dictionary:

        {
            "CUPS_PRINTER_NAME": "DEVICE_URI",
            ...
        }

    Example device URI:

        dnssd://Brother%20HL-L5100DN%20series%20%5Bb42200c3c310%5D._ipp._tcp.local/?uuid=e3248000-80ce-11db-8000-b42200c3c310
    """

    result = subprocess.run(
        [
            "lpstat",
            "-v",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    printer_uris = {}

    for line in result.stdout.splitlines():

        # Expected:
        #
        # device for Brother_HL-L5100DN_21:
        # dnssd://Brother%20HL-L5100DN...

        if not line.startswith(
            "device for "
        ):
            continue

        prefix = "device for "

        try:
            rest = line[
                len(prefix):
            ]

            printer_name, printer_uri = (
                rest.split(
                    ":",
                    1,
                )
            )

            printer_name = (
                printer_name.strip()
            )

            printer_uri = (
                printer_uri.strip()
            )

        except ValueError:

            print(
                f"warning: could not parse "
                f"lpstat output: {line!r}"
            )

            continue

        printer_uris[
            printer_name
        ] = printer_uri

    return printer_uris


def extract_mac_from_printer_uri(
    printer_uri,
):
    """
    Extract the printer MAC address from
    a CUPS dnssd URI.

    Example:

        dnssd://...uuid=e3248000-80ce-11db-8000-b42200c3c310

    returns:

        b42200c3c310
    """

    match = re.search(
        r"uuid=[^&]*-([0-9a-fA-F]{12})",
        printer_uri,
    )

    if match is None:

        return None

    return (
        match.group(1).lower()
    )


def get_avahi_ipp_printers():
    """
    Discover IPP printers using Avahi.

    Returns a dictionary indexed by the printer MAC address:

        {
            "b42200c3c310": {
                "ip": "192.168.178.26",
                "port": 631,
                "ipp_uri":
                    "ipp://192.168.178.26:631/ipp/print",
            },
        }

    The MAC address is normalized to lowercase
    without separators.
    """

    result = subprocess.run(
        [
            "avahi-browse",
            "-rt",
            "_ipp._tcp",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    printers = {}

    # Current Avahi service record.
    current_mac = None
    current_ip = None
    current_port = None

    def finish_record():
        """
        Store the current Avahi service record.
        """

        if (
            current_mac is None
            or current_ip is None
            or current_port is None
        ):
            return

        ipp_uri = (
            f"ipp://"
            f"{current_ip}:"
            f"{current_port}"
            f"/ipp/print"
        )

        printers[
            current_mac
        ] = {
            "ip": current_ip,
            "port": current_port,
            "ipp_uri": ipp_uri,
        }

        print(
            "found avahi printer: "
            f"MAC={current_mac} "
            f"IP={current_ip} "
            f"port={current_port} "
            f"IPP={ipp_uri}"
        )

    for line in result.stdout.splitlines():

        # print(f"get_avahi_ipp_printers: line={line!r}")

        # A line beginning with "=" starts
        # a new Avahi service record.
        #
        # Example:
        #
        # =   eth0 IPv4 Brother HL-L5100DN series
        #     [b42200c3c310] _ipp._tcp local

        if line.startswith("="):

            # Finish the previous service record
            # before starting the next one.
            finish_record()

            current_mac = None
            current_ip = None
            current_port = None

            match = re.search(
                r"\[([0-9a-fA-F]{12})\]",
                line,
            )

            if match:

                current_mac = (
                    match.group(1).lower()
                )

                print(
                    "  found MAC in service line: "
                    f"{current_mac}"
                )

            continue

        stripped = line.strip()

        # Example:
        #
        # hostname = [BRNB42200C3C310.local]

        if stripped.startswith(
            "hostname ="
        ):

            continue

        # Example:
        #
        # address = [192.168.178.26]

        if stripped.startswith(
            "address ="
        ):

            value = (
                stripped[
                    len("address ="):
                ]
                .strip()
            )

            current_ip = (
                value.strip(
                    "[]"
                )
            )

            print(
                "  found IP: "
                f"{current_ip}"
            )

            continue

        # Example:
        #
        # port = [631]

        if stripped.startswith(
            "port ="
        ):

            value = (
                stripped[
                    len("port ="):
                ]
                .strip()
            )

            value = value.strip(
                "[]"
            )

            try:

                current_port = int(
                    value
                )

            except ValueError:

                print(
                    "warning: invalid "
                    f"Avahi port: {value!r}"
                )

                current_port = None

            print(
                "  found port: "
                f"{current_port}"
            )

            continue

        # Example:
        #
        # txt = ["mopria-certified=1.2"
        #        ...
        #        "UUID=e3248000-80ce-11db-8000-b42200c3c310"
        #        ...]

        if stripped.startswith(
            "txt ="
        ):

            txt = (
                stripped[
                    len("txt ="):
                ]
            )

            uuid_match = re.search(
                r'UUID=[^"]*-([0-9a-fA-F]{12})',
                txt,
                re.IGNORECASE,
            )

            if uuid_match:

                uuid_mac = (
                    uuid_match.group(1).lower()
                )

                print(
                    "  found MAC in UUID: "
                    f"{uuid_mac}"
                )

                # Prefer the MAC from the UUID.
                #
                # This matches the behavior of the
                # Bash implementation and is useful
                # if the MAC parsed from the service
                # name is absent or incorrect.
                current_mac = uuid_mac

            # The TXT record is the final part
            # of the Avahi service record.
            finish_record()

            current_mac = None
            current_ip = None
            current_port = None

            continue

    # Handle a final record in case the output
    # does not end with a TXT line.
    finish_record()

    return printers


def resolve_printer_ipp_uri(
    printer_name,
    avahi_printers,
):
    """
    Resolve a CUPS printer name to a direct IPP URI.

    The CUPS printer may have a dnssd:// device URI,
    but ipptool needs an ipp:// URI.

    Example result:

        ipp://192.168.178.100:631/ipp/print
    """

    cups_printers = (
        get_cups_printer_uris()
    )

    if printer_name not in cups_printers:

        raise RuntimeError(
            f"printer {printer_name!r} "
            f"was not found in lpstat -v"
        )

    printer_uri = (
        cups_printers[
            printer_name
        ]
    )

    print(
        f"CUPS device URI for "
        f"{printer_name}: "
        f"{printer_uri}"
    )

    # If CUPS already provides a direct IPP URI,
    # no Avahi lookup is necessary.
    if printer_uri.startswith(
        "ipp://"
    ):

        return printer_uri

    if printer_uri.startswith(
        "ipps://"
    ):

        return printer_uri

    printer_mac = (
        extract_mac_from_printer_uri(
            printer_uri
        )
    )

    if printer_mac is None:

        raise RuntimeError(
            f"could not extract printer MAC "
            f"address from CUPS URI: "
            f"{printer_uri}"
        )

    print(
        f"printer MAC from CUPS URI: "
        f"{printer_mac}"
    )

    if printer_mac not in avahi_printers:

        raise RuntimeError(
            f"printer {printer_name!r} "
            f"with MAC {printer_mac} "
            f"was not found by avahi-browse"
        )

    printer_info = (
        avahi_printers[
            printer_mac
        ]
    )

    ipp_uri = (
        printer_info[
            "ipp_uri"
        ]
    )

    print(
        f"resolved IPP URI for "
        f"{printer_name}: "
        f"{ipp_uri}"
    )

    return ipp_uri


def make_duplex_chunks(
    page_count,
    printer_count,
):
    """
    Split an already-prepared PDF into balanced duplex chunks.

    Two consecutive PDF pages form one physical duplex sheet.

    Therefore:

        PDF pages 1-2  -> physical sheet 1
        PDF pages 3-4  -> physical sheet 2
        PDF pages 5-6  -> physical sheet 3
        ...

    The returned chunks contain an even number of PDF pages
    whenever possible, so each printer receives complete duplex
    sheets.

    The final chunk may contain an odd number of PDF pages,
    in which case the printer produces a blank reverse side
    for the final physical sheet.

    Examples:

        11 PDF pages, 8 printers:

            printer 1: pages 1-4
            printer 2: pages 5-8
            printer 3: pages 9-11

        10 PDF pages, 10 printers:

            printer 1: pages 1-2
            printer 2: pages 3-4
            printer 3: pages 5-6
            printer 4: pages 7-8
            printer 5: pages 9-10

    Pages are one-based and inclusive.
    """

    if page_count <= 0:
        return []

    if printer_count <= 0:
        raise ValueError(
            "printer_count must be greater than zero"
        )

    # Two PDF pages form one physical duplex sheet.
    sheet_count = (
        page_count + 1
    ) // 2

    # We cannot use more printers than physical sheets.
    printer_count_used = min(
        printer_count,
        sheet_count,
    )

    # Distribute physical sheets as evenly as possible.
    base_sheets = (
        sheet_count
        // printer_count_used
    )

    extra_sheets = (
        sheet_count
        % printer_count_used
    )

    chunks = []

    current_page = 1

    for printer_index in range(
        printer_count_used
    ):

        sheets_for_printer = (
            base_sheets
        )

        if (
            printer_index
            < extra_sheets
        ):
            sheets_for_printer += 1

        pages_for_printer = (
            sheets_for_printer
            * 2
        )

        end_page = min(
            current_page
            + pages_for_printer
            - 1,
            page_count,
        )

        chunks.append(
            (
                current_page,
                end_page,
            )
        )

        current_page = (
            end_page + 1
        )

    return chunks


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():

    args = parse_args()

    input_pdf = args.pdf

    if not input_pdf.lower().endswith(
        ".pdf"
    ):
        print(
            "error: input file is not a PDF"
        )
        sys.exit(1)

    if not os.path.isfile(
        input_pdf
    ):
        print(
            f"error: file not found: "
            f"{input_pdf}"
        )
        sys.exit(1)

    if args.zero_image is not None:
        if not os.path.isfile(args.zero_image):
            print(f"error: zero image not found: {args.zero_image}")
            sys.exit(1)

    with open(os.path.expanduser(args.printers_json)) as f:
        printers = json.load(f)

    printer_count = len(printers)

    if printer_count == 0:
        print("error: no printers configured")
        sys.exit(1)



    # --------------------------------------------------------------
    # Read original PDF
    # --------------------------------------------------------------

    original_pdf = pymupdf.open(
        input_pdf
    )

    original_page_count = len(
        original_pdf
    )

    original_pdf.close()

    if original_page_count == 0:
        print(
            "error: PDF has no pages"
        )
        sys.exit(1)



    # --------------------------------------------------------------
    # Create page filter
    # Parse logical page sequence
    # --------------------------------------------------------------

    print(
        f"page filter: "
        f"{args.pages!r}"
    )

    try:
        page_sequence = parse_page_sequence(
            args.pages,
            original_page_count,
        )

    except ValueError as e:
        print(
            f"error: invalid --pages specification: "
            f"{e}"
        )
        sys.exit(1)



    zero_page_index = None

    if args.zero_page is not None:

        try:

            zero_page_index = (
                resolve_pdf_page_number(
                    args.zero_page,
                    original_page_count,
                )
            )

        except ValueError as e:

            print(
                f"error: {e}"
            )

            sys.exit(1)

        print(
            f"zero page source: "
            f"original PDF page "
            f"{args.zero_page} "
            f"(zero-based index "
            f"{zero_page_index})"
        )



    # --------------------------------------------------------------
    # Create logical filtered PDF
    # --------------------------------------------------------------

    (
        filtered_pdf,
        selected_page_count,
    ) = create_filtered_pdf(
        input_pdf,
        page_sequence,
        zero_page_index=zero_page_index,
        zero_image=args.zero_image,
    )

    if selected_page_count == 0:

        # A page sequence containing only logical zero pages is
        # still valid, so only reject an entirely empty sequence.
        if not page_sequence:

            print(
                "error: --pages selected no pages"
            )

            sys.exit(1)



    # --------------------------------------------------------------
    # Prepare final printable PDF
    # --------------------------------------------------------------

    if args.nup2:

        print(
            "mode: real 2-up PDF, "
            "duplex short-edge"
        )

        final_pdf = (
            f"{input_pdf}.nup2.pdf"
        )

        create_nup2_pdf(
            filtered_pdf,
            final_pdf,
        )

    else:

        print(
            "mode: 1-up, "
            "duplex long-edge"
        )

        final_pdf = (
            filtered_pdf
        )

    pdf_path = final_pdf



    # --------------------------------------------------------------
    # Determine final printable page count
    # --------------------------------------------------------------


    final_pdf_doc = pymupdf.open(
        final_pdf
    )

    final_page_count = len(
        final_pdf_doc
    )

    final_pdf_doc.close()

    print()
    print(
        f"final printable PDF: "
        f"{final_pdf}"
    )

    print(
        f"final printable pages: "
        f"{final_page_count}"
    )



    # --------------------------------------------------------------
    # Printing options
    # --------------------------------------------------------------

    lp_options = get_lp_options(
        args.nup2
    )

    print()
    print(f"input PDF: {input_pdf}")
    print(f"original pages: {original_page_count}")
    print(f"selected pages: {selected_page_count}")
    print(f"prepared PDF: {pdf_path}")
    print(f"pages being split: {final_page_count}")
    print(f"printers: {printer_count}")

    if args.cache:
        print("cache: enabled")
    else:
        print("cache: disabled")

    # --------------------------------------------------------------
    # Cache setup
    # --------------------------------------------------------------

    if args.cache:

        if args.cache_dir:

            cache_dir = (
                args.cache_dir
            )

        else:

            cache_dir = (
                f"{input_pdf}"
                f"{DEFAULT_CACHE_SUFFIX}"
            )

        os.makedirs(
            cache_dir,
            exist_ok=True,
        )

        print(
            f"cache directory: "
            f"{cache_dir}"
        )

        printer_ppd_cache = {}

        for printer in printers:

            (
                ppd_path,
                ppd_hash,
            ) = get_readable_ppd(
                printer,
                cache_dir,
            )

            printer_ppd_cache[
                printer
            ] = (
                ppd_path,
                ppd_hash,
            )



    # --------------------------------------------------------------
    # Split final printable PDF into chunks
    # --------------------------------------------------------------

    chunks = make_duplex_chunks(
        final_page_count,
        printer_count,
    )

    print()
    print(
        f"printers available: "
        f"{printer_count}"
    )

    print(
        f"printers used: "
        f"{len(chunks)}"
    )

    if args.cache:

        avahi_printers = (
            get_avahi_ipp_printers()
        )

    for printer, (
        start_page,
        end_page,
    ) in zip(
        printers,
        chunks,
    ):

        chunk_page_count = (
            end_page
            - start_page
            + 1
        )

        print()
        print(
            f"Printer {printer}: "
            f"pages {start_page}-"
            f"{end_page} "
            f"({chunk_page_count} pages)"
        )

        # Every chunk except the final chunk must
        # have an even number of pages.
        if (
            end_page < final_page_count
            and chunk_page_count % 2
            != 0
        ):

            raise RuntimeError(
                f"Internal error: "
                f"chunk {start_page}-"
                f"{end_page} has an odd "
                f"number of pages"
            )

        # ----------------------------------------------------------
        # Create chunk PDF
        # ----------------------------------------------------------

        chunk_pdf = (
            f"{final_pdf}"
            f".pages{start_page}-"
            f"{end_page}.pdf"
        )


        # if os.path.exists(chunk_pdf):
        if False:
            print(
                f"keeping existing "
                f"{chunk_pdf}"
            )

        else:

            print(
                f"writing {chunk_pdf}"
            )

            source = pymupdf.open(
                final_pdf
            )

            chunk = pymupdf.open()

            chunk.insert_pdf(
                source,
                from_page=(
                    start_page - 1
                ),
                to_page=(
                    end_page - 1
                ),
            )

            chunk.save(
                chunk_pdf
            )

            chunk.close()
            source.close()

        # ----------------------------------------------------------
        # Print with or without raster cache
        # ----------------------------------------------------------

        if args.cache:

            (
                ppd_path,
                ppd_hash,
            ) = printer_ppd_cache[
                printer
            ]

            raster_path = (
                get_raster_cache(
                    chunk_pdf,
                    ppd_path,
                    ppd_hash,
                    cache_dir,
                    lp_options,
                )
            )

            printer_ipp_url = (
                resolve_printer_ipp_uri(
                    printer,
                    avahi_printers,
                )
            )

            print_raster_ipp(
                printer_ipp_url,
                raster_path,
            )

        else:

            command = [
                "lp",
                *lp_options,
                "-d",
                printer,
                chunk_pdf,
            ]

            print()
            print(
                "lp command:"
            )

            print(
                shlex.join(command)
            )

            subprocess.run(
                command,
                check=True,
            )

        time.sleep(T1)


if __name__ == "__main__":

    try:

        main()

    except ValueError as e:

        print(
            f"error: {e}"
        )

        sys.exit(1)

    except subprocess.CalledProcessError as e:

        print(
            f"error: command failed "
            f"with exit code {e.returncode}"
        )

        sys.exit(
            e.returncode
        )

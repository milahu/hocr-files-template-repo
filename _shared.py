from pathlib import Path
import importlib.util
import re
from typing import Iterable
from collections import defaultdict
from types import SimpleNamespace
import types

config_path = Path("000-config.py")

new_config_path = Path("000-config.py")

# old config paths
old_config_path_030_txt = Path("030-measure-page-size.txt")
old_config_path_050_py = Path("050-measure-crop-size.py")
old_config_path_050_txt = Path("050-measure-crop-size.txt")

debug_load_config = False

def load_config(config_path=config_path, base_config=None):

    if debug_load_config:
        print(f"loading config: {config_path}")

    if config_path == new_config_path and not config_path.exists():
        # load old config files
        config = None
        if old_config_path_030_txt.exists():
            config = load_bash_config(old_config_path_030_txt, base_config=config)
        if old_config_path_050_py.exists():
            config = load_config(old_config_path_050_py, base_config=config)
        elif old_config_path_050_txt.exists():
            config = load_bash_config(old_config_path_050_txt, base_config=config)
        if config:
            return config
        # else: no old config was loaded

    spec = importlib.util.spec_from_file_location("config", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {config_path}")

    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    # convert module to namespace
    # to make it pickle-able for multiprocessing
    # fix: TypeError: cannot pickle 'module' object
    config = SimpleNamespace(
        **{
            name: value
            for name, value in vars(config).items()
            if not name.startswith("__")
            and not isinstance(value, types.ModuleType)
        }
    )

    if base_config:
        # merge configs
        for key in dir(config):
            if key[0] == "_": continue
            val = getattr(config, key)
            if debug_load_config:
                if hasattr(base_config, key) and getattr(base_config, key) != val:
                    print(f"replacing config: {key}={val!r}")
                else:
                    print(f"merging config: {key}={val!r}")
            setattr(base_config, key, val)
        config = base_config

    # allow appending some pages
    # without having to rename all files
    config.max_num_pages = int(max(
        config.num_pages + 100,
        config.num_pages * 1.2,
    ))

    config.page_num_width = len(str(config.max_num_pages))

    def hasattrs(obj, attrs):
        "does the object have all attributes?"
        for attr in attrs:
            if not hasattr(obj, attr):
                return False
        return True

    if not hasattr(config, "scan_top_edge"):
        if hasattrs(config, ("do_rotate", "rotate_odd", "rotate_even")):
            # infer scan_top_edge for backward-compatibility with old configs
            if config.do_rotate == True:
                if (config.rotate_odd, config.rotate_even) == (270, 90):
                    config.scan_top_edge = "inside"
                elif (config.rotate_odd, config.rotate_even) == (180, 180):
                    config.scan_top_edge = "bottom"
            elif config.do_rotate == False:
                config.scan_top_edge = "top"

    assert config.scan_top_edge in ("inside", "top", "bottom"), f"config.scan_top_edge={config.scan_top_edge!r}"

    if not hasattrs(config, ("page_width_mm", "page_height_mm")):
        if hasattrs(config, ("scan_x", "scan_y")):
            # infer (page_width_mm, page_height_mm) for backward-compatibility with old configs
            if config.scan_top_edge == "inside":
                # rotate by 90 degrees
                config.page_width_mm = config.scan_y
                config.page_height_mm = config.scan_x
            else:
                # rotate by 0 or 180 degrees
                config.page_width_mm = config.scan_x
                config.page_height_mm = config.scan_y

    if not hasattr(config, "unbinded_page_width_mm"):
        # assume that a width of 5 mm was removed by unbinding the book
        config.unbinded_page_width_mm = config.page_width_mm - 5

    # Scanner geometry:
    # X is parallel to the scan top edge.
    # Y is the feed direction.
    # Keep these derived values separate from physical page size.
    if config.scan_top_edge == "inside":
        # common case: the physical inside edge is fed first into the scanner
        config.scan_width_mm = config.page_height_mm
        config.scan_height_mm = config.unbinded_page_width_mm
        config.do_rotate = True
        config.rotate_odd, config.rotate_even = 270, 90
    else:
        # rare case: the physical top or bottom edge is fed first into the scanner
        config.scan_width_mm = config.unbinded_page_width_mm
        config.scan_height_mm = config.page_height_mm
        if config.scan_top_edge == "top":
            # rare case: the physical top edge is fed first into the scanner
            config.do_rotate = False
            config.rotate_odd = config.rotate_even = 0
        elif config.scan_top_edge == "bottom":
            # rare case: the physical bottom edge is fed first into the scanner
            config.do_rotate = True
            config.rotate_odd = config.rotate_even = 180

    # TODO verify: 040-scan-pages.py should try to add scan_margin
    # to the scan left + right + bottom edges
    # as permitted by config.max_scan_width_mm and config.max_scan_height_mm
    # yes?
    # config.margined_scan_width_mm
    # config.margined_scan_height_mm

    if not hasattrs(config, ("max_scan_width_mm", "max_scan_height_mm")):
        # infer scanner limits from my DIN A4 document scanner: Brother ADS-2400N
        config.max_scan_width_mm = 215.88
        config.max_scan_height_mm = 355.567

    if config.scan_width_mm > config.max_scan_width_mm:
        raise ValueError(
            f"scan_top_edge requires a scan width of {config.scan_width_mm} mm,"
            f" but the scanner supports only {config.max_scan_width_mm} mm"
        )
    if config.scan_height_mm > config.max_scan_height_mm:
        raise ValueError(
            f"scan_top_edge requires a scan height of {config.scan_height_mm} mm,"
            f" but the scanner supports only {config.max_scan_height_mm} mm"
        )

    if not hasattr(config, "outside_edge_detection_min_margin_mm"):
        config.outside_edge_detection_min_margin_mm = 2.0

    # TODO? remove in favor of config.edge_deskew_mode
    config.use_three_edge_deskew = config.scan_top_edge == "inside"
    # "outside edge is reliable" means:
    # the physical outside edge can be detected reliably
    # because the scanned image contains enough of the gray scanner background
    # so there is enough visible contrast between the physical outside edge and the scanner background
    config.outside_edge_is_reliable = (
        # TODO verify:
        # does "config.use_three_edge_deskew == True" always mean
        # that we can reliably detect the physical outside edge?
        config.use_three_edge_deskew
        or (
            config.max_scan_width_mm - config.unbinded_page_width_mm
            >= config.outside_edge_detection_min_margin_mm
        )
    )
    # config for 065-remove-page-borders.py
    # NOTE this "edge deskew" is different from the "content deskew" in 070-deskew.py
    if config.use_three_edge_deskew:
        config.edge_deskew_mode = "three_edges"
    elif config.outside_edge_is_reliable:
        config.edge_deskew_mode = "two_edges"
    elif config.scan_top_edge == "top":
        config.edge_deskew_mode = "bottom_only"
    else:
        config.edge_deskew_mode = "top_only"

    # FIXME
    # scanimage: rounded value of br-x from 216 to 215.88
    # scanimage: rounded value of br-y from 166 to 165.985
    # no, this depends on the scanner model = sane backend
    r'''
    def snap_mm(name, mm, max_mm, dpi):
        mm_bak = mm
        if max_mm:
            mm = min(mm, max_mm)
        mm_per_inch = 25.4
        px = mm * dpi / mm_per_inch
        # px = int(px) # wrong
        px = round(px) # wrong?
        mm = px * mm_per_inch / dpi
        if mm != mm_bak:
            print(f"snap_mm {name}: {mm_bak} -> {mm}")
        return mm
    config.scan_x = snap_mm("scan_x", config.scan_x, config.max_scan_x, config.scan_resolution)
    config.scan_y = snap_mm("scan_y", config.scan_y, config.max_scan_y, config.scan_resolution)
    '''

    def get_rotated_x_y(config, x, y):
        if config.do_rotate and config.rotate_odd in (90, 270):
            # rotate by 90 or 270 degrees
            x, y = y, x
        return x, y

    # scanned image size after 060-rotate-crop.py
    # NOTE dont apply crop
    (
        config.rotated_scan_x,
        config.rotated_scan_y
    ) = get_rotated_x_y(config, config.scan_width_mm, config.scan_height_mm)

    def px_of_mm(mm, dpi):
        return mm * dpi / 25.4

    dpi = config.scan_resolution

    (
        config.page_width_px,
        config.page_height_px
    ) = (
        px_of_mm(config.page_width_mm, dpi),
        px_of_mm(config.page_height_mm, dpi)
    )

    config.rotated_scan_aspect = config.rotated_scan_x / config.rotated_scan_y

    if not hasattr(config, "scan_margin"):
        config.scan_margin = 10

    config.margined_scan_width_mm = min(
        config.scan_width_mm + config.scan_margin,
        config.max_scan_width_mm
    )
    config.margined_scan_height_mm = min(
        config.scan_height_mm + config.scan_margin,
        config.max_scan_height_mm
    )

    # TODO remove?
    (
        config.rotated_margined_scan_x,
        config.rotated_margined_scan_y
    ) = get_rotated_x_y(
        config,
        config.margined_scan_width_mm,
        config.margined_scan_height_mm,
    )

    # thresholds for color leveling should be symmetrical:
    # lowthresh + highthresh == 1
    # so it is enough to specify only the low threshold
    if hasattr(config, "lowthresh") and not hasattr(config, "highthresh"):
        config.highthresh = 1 - config.lowthresh
    if not hasattr(config, "text_lowthresh"):
        config.text_lowthresh = config.lowthresh
    if hasattr(config, "text_lowthresh") and not hasattr(config, "text_highthresh"):
        config.text_highthresh = 1 - config.text_lowthresh
    if not hasattr(config, "images_lowthresh"):
        config.images_lowthresh = config.lowthresh
    if hasattr(config, "images_lowthresh") and not hasattr(config, "images_highthresh"):
        config.images_highthresh = 1 - config.images_lowthresh

    if not hasattr(config, "color_pages"):
        config.color_pages = []

    if not hasattr(config, "image_pages"):
        # infer image_pages from color_pages
        config.image_pages = config.color_pages

    if not hasattr(config, "deskew_fix_page_size_keep_small_pages"):
        config.deskew_fix_page_size_keep_small_pages = False

    # TODO validate config
    # if orientation_is_portrait and (scan_x < scan_y) and do_rotate:

    if getattr(config, "deskew_white_lightness_threshold", None) != None:
        assert 0 <= config.deskew_white_lightness_threshold <= 1

    if getattr(config, "deskew_black_lightness_threshold", None) != None:
        assert 0 <= config.deskew_black_lightness_threshold <= 1

    if getattr(config, "deskew_dark_lightness_threshold", None) != None:
        assert 0 <= config.deskew_dark_lightness_threshold <= 1

    return config


def get_page_num(path):
    # parse page number: "001.jpg" -> 1
    if not isinstance(path, Path):
        path = Path(path)
    m = re.match(r"^(\d+)", path.name)
    # page_num = int(m.group(1)) if m else 0
    page_num = int(m.group(1)) # can throw
    return page_num


def compress_paths(paths: Iterable[str | Path]) -> str:
    """
    Compress a list of file paths into Bash brace expansion syntax.

    Example:
        >>> compress_paths([
        ...     "common-dir/001.tiff",
        ...     "common-dir/003.tiff",
        ...     "common-dir/005.tiff",
        ... ])
        'common-dir/{001,003,005}.tiff'

        >>> compress_paths([
        ...     Path("foo/a.txt"),
        ...     Path("foo/b.txt"),
        ... ])
        'foo/{a,b}.txt'
    """
    paths = [Path(p) for p in paths]

    if not paths:
        return ""

    groups = defaultdict(list)

    for path in paths:
        stem = path.stem
        suffix = "".join(path.suffixes)
        parent = path.parent

        groups[(parent, suffix)].append(stem)

    results = []

    for (parent, suffix), stems in groups.items():
        stems = sorted(stems)

        # Find longest common prefix/suffix of stems
        prefix = stems[0]
        for s in stems[1:]:
            i = 0
            while i < min(len(prefix), len(s)) and prefix[i] == s[i]:
                i += 1
            prefix = prefix[:i]

        suffix_part = stems[0]
        for s in stems[1:]:
            i = 0
            while (
                i < min(len(suffix_part), len(s))
                and suffix_part[-(i + 1)] == s[-(i + 1)]
            ):
                i += 1
            suffix_part = suffix_part[len(suffix_part) - i :] if i else ""

        middles = []
        valid = True
        for s in stems:
            if not (s.startswith(prefix) and s.endswith(suffix_part)):
                valid = False
                break
            middle = s[len(prefix) :]
            if suffix_part:
                middle = middle[: -len(suffix_part)]
            middles.append(middle)

        if valid and len(stems) > 1 and all(m for m in middles):
            filename = f"{prefix}{{{','.join(middles)}}}{suffix_part}{suffix}"
        elif len(stems) > 1:
            filename = f"{{{','.join(stems)}}}{suffix}"
        else:
            filename = stems[0] + suffix

        if parent == Path("."):
            results.append(filename)
        else:
            results.append(str(parent / filename))

    return " ".join(sorted(results))


def get_image_viewer_argstr(filenames, config):
    return f"{config.image_viewer} {compress_paths(filenames)}"


def latest_dst_exists(f_src, f_dst):
    """
    check if the latest version of dst exists
    """
    if not isinstance(f_src, Path): f_src = Path(f_src)
    if not isinstance(f_dst, Path): f_dst = Path(f_dst)
    if not f_dst.exists():
        # f_dst does not exist at all
        return False
    # f_dst exists
    if f_dst.stat().st_mtime < f_src.stat().st_mtime:
        # f_src was modified after f_dst
        # so f_dst is not the latest version
        return False
    # f_src was modified before f_dst
    # so f_dst is the latest version
    return True


def remove_done_files(files, dst, dst_suffix=None):
    """
    filter input files by existing output files

    if the corresponding output file
    exists and is not older than the input file
    then remove the input file

    if the corresponding output file
    does not exist or is older than the input file
    then keep the input file
    """
    if not isinstance(dst, Path): dst = Path(dst)
    files2 = []
    for f_src in files:
        if not isinstance(f_src, Path): f_src = Path(f_src)
        f_dst = dst / f_src.name
        if dst_suffix:
            f_dst = f_dst.with_suffix(dst_suffix)
        # if f_dst.exists():
        if latest_dst_exists(f_src, f_dst):
            continue
        files2.append(f_src)
    return files2


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


def filename_of_page(page_num, config, extension=None) -> str:
    """
    return the zero-padded filename of this page

    example: 1 -> "001.tiff"
    """
    if extension is None:
        extension = f".{config.scan_format}"
    return f"{page_num:0{config.page_num_width}d}{extension}"



# parse old bash configs
# 030-measure-page-size.txt
# 050-measure-crop-size.txt

import ast
import operator
import re
from types import SimpleNamespace

def load_bash_config(config_path, base_config=None):

    if debug_load_config:
        print(f"loading bash config: {config_path}")

    _ARITHMETIC = re.compile(r"^\$\(\((.*?)\)\)$")

    _OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
    }


    def eval_arithmetic(expr, config):
        tree = ast.parse(expr, mode="eval")

        def evaluate(node):
            if isinstance(node, ast.Expression):
                return evaluate(node.body)

            if isinstance(node, ast.Constant) and isinstance(node.value, int):
                return node.value

            if isinstance(node, ast.Name):
                try:
                    return getattr(config, node.id)
                except AttributeError:
                    raise ValueError(f"Unknown variable: {node.id}")

            if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
                return _OPERATORS[type(node.op)](
                    evaluate(node.left),
                    evaluate(node.right),
                )

            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
                return -evaluate(node.operand)

            raise ValueError(f"Unsupported expression: {ast.dump(node)}")

        return evaluate(tree)


    def parse_value(value, config):
        value = value.strip()

        # Quoted string
        if len(value) >= 2 and value[0] == value[-1] == '"':
            return value[1:-1]

        if len(value) >= 2 and value[0] == value[-1] == "'":
            return value[1:-1]

        # Bash arithmetic expression: $((...))
        match = _ARITHMETIC.fullmatch(value)
        if match:
            return eval_arithmetic(match.group(1), config)

        # Integer
        if re.fullmatch(r"-?\d+", value):
            return int(value)

        # Plain string
        return value

    config = SimpleNamespace()

    with open(config_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)=(.*)", line)
            if not match:
                raise ValueError(
                    f"{config_path}:{lineno}: invalid line: {line!r}"
                )

            name, value = match.groups()
            setattr(config, name, parse_value(value, config))

    if debug_load_config:
        for key in dir(config):
            if key[0] == "_": continue
            val = getattr(config, key)
            print(f"{old_config_path_030_txt}: {key}={val!r}")

    if base_config:
        # merge configs
        for key in dir(config):
            if key[0] == "_": continue
            val = getattr(config, key)
            if debug_load_config:
                if hasattr(base_config, key) and getattr(base_config, key) != val:
                    print(f"replacing config: {key}={val!r}")
                else:
                    print(f"merging config: {key}={val!r}")
            setattr(base_config, key, val)
        config = base_config

    return config

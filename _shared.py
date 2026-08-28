from pathlib import Path
import importlib.util
import re
from typing import Iterable
from collections import defaultdict
from types import SimpleNamespace
import types

config_path = Path("000-config.py")

def load_config(config_path=config_path):
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

    # allow appending some pages
    # without having to rename all files
    config.max_num_pages = int(max(
        config.num_pages + 100,
        config.num_pages * 1.2,
    ))

    config.page_num_width = len(str(config.max_num_pages))

    assert config.scan_top_edge in ("inside", "top", "bottom"), f"config.scan_top_edge={config.scan_top_edge!r}"

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

    if not hasattr(config, "image_pages"):
        # infer image_pages from color_pages
        config.image_pages = config.color_pages

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

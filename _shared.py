from pathlib import Path
import importlib.util
import re

config_path = Path("000-config.py")

def load_config(config_path=config_path):
    spec = importlib.util.spec_from_file_location("config", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {config_path}")

    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    # allow appending some pages
    # without having to rename all files
    config.max_num_pages = int(max(
        config.num_pages + 100,
        config.num_pages * 1.2,
    ))

    config.page_num_width = len(str(config.max_num_pages))

    # book binding side = scanner top side
    # TODO handle more cases?
    config.use_three_edge_deskew = (
        config.do_rotate
        and config.rotate_odd == 270
        and config.rotate_even == 90
    )

    config.scan_x = min(config.scan_x, config.max_scan_x)
    config.scan_y = min(config.scan_y, config.max_scan_y)

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
        if config.use_three_edge_deskew:
            # rotate by 90 or 270 degrees
            x, y = y, x
        return x, y

    # scanned image size after 060-rotate-crop.py
    # NOTE dont apply crop
    (
        config.rotated_scan_x,
        config.rotated_scan_y
    ) = get_rotated_x_y(config, config.scan_x, config.scan_y)

    config.rotated_scan_aspect = config.rotated_scan_x / config.rotated_scan_y

    config.margined_scan_x = min(
        config.scan_x + config.scan_margin,
        config.max_scan_x
    )
    config.margined_scan_y = min(
        config.scan_y + config.scan_margin,
        config.max_scan_y
    )

    (
        config.rotated_margined_scan_x,
        config.rotated_margined_scan_y
    ) = get_rotated_x_y(config, config.margined_scan_x, config.margined_scan_y)

    # TODO validate config
    # if orientation_is_portrait and (scan_x < scan_y) and do_rotate:

    return config


def get_page_num(path):
    # parse page number: "001.jpg" -> 1
    m = re.match(r"^(\d+)", path.name)
    # page_num = int(m.group(1)) if m else 0
    page_num = int(m.group(1)) # can throw
    return page_num

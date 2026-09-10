# Write PySide6 Image Viewer

**User:** Anonymous  
**Created:** 2026/9/9 17:41:44  
**Updated:** 2026/9/10 6:46:51  
**Exported:** 2026/9/10 6:46:57  
**Link:** [<https://chatgpt.com/c/6aa17e37-c934-83ed-bc94-a2754222f170>](https://chatgpt.com/c/6aa17e37-c934-83ed-bc94-a2754222f170)

## Prompt:

9/9/2026, 5:42:14 PM

generate a python script:

it should be a GUI image viewer

it should use the pyside6 python-qt bindings

it should be a specialized image viewer, to view images of scanned book pages

it should show the left page (even page) and the right page (odd page) side by side (except on the first and last page)

it should have an "autoplay" feature, like a video player, where the user can hit the spacebar and the viewer shows all pages, with a configurable "time per page" (default maybe 0.2 seconds). hitting the spacebar again pauses the "image stream". the arrow buttons (arrow-left and arrow-right) allow seeking (previous page, next page)

it should also work efficiently with slow hard disk drives, so it should preload the next N images

## Response:

9/9/2026, 5:42:17 PM

Below is a self-contained PySide6 viewer. It treats pages as **1-based page numbers**, displaying:

- page 1 by itself
- pages 2–3 side by side
- pages 4–5 side by side
- pages 6–7, etc.
- the final page by itself if the book has an even number of pages

It also includes an LRU image cache, background preloading with `QThreadPool`, autoplay, configurable page timing, keyboard controls, and folder/file selection.

\*\*document: \*\*

``` python
#!/usr/bin/env python3

"""
Scanned Book Image Viewer
=========================

A PySide6 GUI for viewing scanned book pages.

Page layout:
    page 1          -> single page
    pages 2 + 3     -> left + right
    pages 4 + 5     -> left + right
    pages 6 + 7     -> left + right
    ...
    final page      -> single page if necessary

Features:
    - PySide6
    - Side-by-side book layout
    - Spacebar autoplay / pause
    - Configurable time per page/spread
    - Left/right arrow seeking
    - Background image preloading
    - LRU image cache
    - Natural filename sorting
    - Fit-to-window display
    - Folder selection
    - Open individual image files
    - Keyboard shortcuts
    - Works with common image formats

Install:
    pip install PySide6

Run:
    python book_viewer.py
"""

from __future__ import annotations

import os
import re
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import (
    QObject,
    QRunnable,
    QThreadPool,
    QTimer,
    Qt,
    QMutex,
    Signal,
    Slot,
)
from PySide6.QtGui import (
    QAction,
    QColor,
    QFont,
    QImage,
    QKeySequence,
    QPainter,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QStatusBar,
    QStyle,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
    ".webp",
}

DEFAULT_TIME_PER_PAGE = 0.20
DEFAULT_PRELOAD_COUNT = 6
DEFAULT_CACHE_SIZE = 24

# ---------------------------------------------------------------------------
# Natural sorting
# ---------------------------------------------------------------------------

def natural_sort_key(path: Path):
    """
    Sort filenames naturally:

        page1.jpg
        page2.jpg
        page10.jpg

    instead of lexicographically:

        page1.jpg
        page10.jpg
        page2.jpg
    """
    parts = re.split(r"(\d+)", path.name.lower())

    result = []
    for part in parts:
        if part.isdigit():
            result.append((0, int(part)))
        else:
            result.append((1, part))

    return result

# ---------------------------------------------------------------------------
# Page model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Spread:
    """
    A displayable unit.

    left:
        Page displayed on the left side.

    right:
        Page displayed on the right side.

    Page numbers are zero-based indexes into the image list.
    """

    left: int
    right: int | None = None

def make_spreads(page_count: int) -> list[Spread]:
    """
    Convert a sequence of pages into book-like spreads.

    For N pages:

        1              -> [1]
        2,3            -> [2,3]
        4,5            -> [4,5]
        ...

    Internally pages are zero-based.
    """

    if page_count <= 0:
        return []

    spreads = [Spread(0)]

    page = 1

    while page < page_count:
        if page + 1 < page_count:
            spreads.append(Spread(page, page + 1))
            page += 2
        else:
            spreads.append(Spread(page))
            page += 1

    return spreads

# ---------------------------------------------------------------------------
# Thread-safe LRU image cache
# ---------------------------------------------------------------------------

class ImageCache:
    """
    Small thread-safe LRU cache containing QImage objects.

    QImage is implicitly shared and is safe to pass between threads as
    long as the GUI isn't modifying the same object.
    """

    def __init__(self, max_items: int):
        self.max_items = max(1, max_items)

        self._cache: OrderedDict[int, QImage] = OrderedDict()
        self._mutex = QMutex()

    def get(self, index: int) -> QImage | None:
        self._mutex.lock()

        try:
            image = self._cache.get(index)

            if image is not None:
                self._cache.move_to_end(index)
                return image

            return None

        finally:
            self._mutex.unlock()

    def put(self, index: int, image: QImage):
        self._mutex.lock()

        try:
            self._cache[index] = image
            self._cache.move_to_end(index)

            while len(self._cache) > self.max_items:
                self._cache.popitem(last=False)

        finally:
            self._mutex.unlock()

    def clear(self):
        self._mutex.lock()

        try:
            self._cache.clear()

        finally:
            self._mutex.unlock()

    def resize(self, max_items: int):
        self._mutex.lock()

        try:
            self.max_items = max(1, max_items)

            while len(self._cache) > self.max_items:
                self._cache.popitem(last=False)

        finally:
            self._mutex.unlock()

# ---------------------------------------------------------------------------
# Background image loading
# ---------------------------------------------------------------------------

class ImageLoadSignals(QObject):
    finished = Signal(int, object)
    failed = Signal(int, str)

class ImageLoadTask(QRunnable):
    """
    Loads one image from disk in a worker thread.
    """

    def __init__(
        self,
        index: int,
        path: Path,
        cache: ImageCache,
        signals: ImageLoadSignals,
    ):
        super().__init__()

        self.index = index
        self.path = path
        self.cache = cache
        self.signals = signals

        self.setAutoDelete(True)

    @Slot()
    def run(self):
        # Someone may already have loaded this image.
        cached = self.cache.get(self.index)

        if cached is not None:
            self.signals.finished.emit(self.index, cached)
            return

        image = QImage(str(self.path))

        if image.isNull():
            self.signals.failed.emit(
                self.index,
                f"Could not load image:\n{self.path}",
            )
            return

        # Store the original image in the cache.
        self.cache.put(self.index, image)

        self.signals.finished.emit(self.index, image)

# ---------------------------------------------------------------------------
# Settings dialog
# ---------------------------------------------------------------------------

class SettingsDialog(QDialog):
    def __init__(
        self,
        time_per_page: float,
        preload_count: int,
        cache_size: int,
        parent=None,
    ):
        super().__init__(parent)

        self.setWindowTitle("Viewer Settings")

        layout = QFormLayout(self)

        self.time_spin = QDoubleSpinBox()
        self.time_spin.setRange(0.05, 60.0)
        self.time_spin.setSingleStep(0.05)
        self.time_spin.setDecimals(2)
        self.time_spin.setValue(time_per_page)
        self.time_spin.setSuffix(" s")

        self.preload_spin = QSpinBox()
        self.preload_spin.setRange(0, 100)
        self.preload_spin.setValue(preload_count)
        self.preload_spin.setSuffix(" images")

        self.cache_spin = QSpinBox()
        self.cache_spin.setRange(2, 500)
        self.cache_spin.setValue(cache_size)
        self.cache_spin.setSuffix(" images")

        layout.addRow("Time per page:", self.time_spin)
        layout.addRow("Preload next:", self.preload_spin)
        layout.addRow("Image cache:", self.cache_spin)

        buttons = QHBoxLayout()

        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")

        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)

        buttons.addStretch()
        buttons.addWidget(cancel_button)
        buttons.addWidget(ok_button)

        layout.addRow(buttons)

# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class BookViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Scanned Book Viewer")
        self.resize(1400, 900)

        # ---------------------------------------------------------------
        # State
        # ---------------------------------------------------------------

        self.image_paths: list[Path] = []
        self.spreads: list[Spread] = []

        self.current_spread = 0

        self.autoplay = False

        self.time_per_page = DEFAULT_TIME_PER_PAGE
        self.preload_count = DEFAULT_PRELOAD_COUNT
        self.cache_size = DEFAULT_CACHE_SIZE

        self.cache = ImageCache(self.cache_size)

        self.thread_pool = QThreadPool.globalInstance()

        # Track tasks that are currently being loaded.
        self.loading: set[int] = set()

        # ---------------------------------------------------------------
        # Autoplay timer
        # ---------------------------------------------------------------

        self.autoplay_timer = QTimer(self)
        self.autoplay_timer.setSingleShot(False)
        self.autoplay_timer.timeout.connect(self.advance)

        # ---------------------------------------------------------------
        # Image area
        # ---------------------------------------------------------------

        self.left_label = self._create_page_label()
        self.right_label = self._create_page_label()

        self.left_label.setText("Open a folder containing scanned pages")
        self.right_label.setText("")

        self.page_container = QWidget()
        page_layout = QHBoxLayout(self.page_container)
        page_layout.setContentsMargins(8, 8, 8, 8)
        page_layout.setSpacing(2)

        page_layout.addWidget(self.left_label, 1)
        page_layout.addWidget(self.right_label, 1)

        self.setCentralWidget(self.page_container)

        # ---------------------------------------------------------------
        # Toolbar
        # ---------------------------------------------------------------

        self._create_actions()
        self._create_toolbar()

        # ---------------------------------------------------------------
        # Status bar
        # ---------------------------------------------------------------

        self.status = QStatusBar()
        self.setStatusBar(self.status)

        self.status.showMessage("Open a folder to begin")

        # ---------------------------------------------------------------
        # Initial UI state
        # ---------------------------------------------------------------

        self._update_navigation()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _create_page_label(self) -> QLabel:
        label = QLabel()

        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setMinimumSize(100, 100)

        label.setStyleSheet(
            """
            QLabel {
                background-color: #202020;
                color: #909090;
                border: 1px solid #333333;
            }
            """
        )

        label.setText("")

        return label

    def _create_actions(self):
        # Open folder
        self.open_folder_action = QAction("Open Folder", self)
        self.open_folder_action.setShortcut(QKeySequence("Ctrl+O"))
        self.open_folder_action.triggered.connect(self.open_folder)

        # Open individual files
        self.open_files_action = QAction("Open Images", self)
        self.open_files_action.setShortcut(QKeySequence("Ctrl+Shift+O"))
        self.open_files_action.triggered.connect(self.open_files)

        # Previous
        self.previous_action = QAction("Previous", self)
        self.previous_action.setShortcut(QKeySequence("Left"))
        self.previous_action.triggered.connect(self.previous)

        # Next
        self.next_action = QAction("Next", self)
        self.next_action.setShortcut(QKeySequence("Right"))
        self.next_action.triggered.connect(self.next)

        # Play/pause
        self.play_action = QAction("Play", self)
        self.play_action.setShortcut(QKeySequence("Space"))
        self.play_action.triggered.connect(self.toggle_autoplay)

        # Settings
        self.settings_action = QAction("Settings", self)
        self.settings_action.triggered.connect(self.show_settings)

        # Fullscreen
        self.fullscreen_action = QAction("Fullscreen", self)
        self.fullscreen_action.setShortcut(QKeySequence("F11"))
        self.fullscreen_action.triggered.connect(self.toggle_fullscreen)

        # Quit
        self.quit_action = QAction("Quit", self)
        self.quit_action.setShortcut(QKeySequence("Ctrl+Q"))
        self.quit_action.triggered.connect(self.close)

    def _create_toolbar(self):
        toolbar = QToolBar("Controls")
        toolbar.setMovable(False)

        toolbar.addAction(self.open_folder_action)
        toolbar.addAction(self.open_files_action)

        toolbar.addSeparator()

        toolbar.addAction(self.previous_action)
        toolbar.addAction(self.play_action)
        toolbar.addAction(self.next_action)

        toolbar.addSeparator()

        toolbar.addAction(self.settings_action)
        toolbar.addAction(self.fullscreen_action)

        self.addToolBar(toolbar)

    # ------------------------------------------------------------------
    # Opening images
    # ------------------------------------------------------------------

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select scanned book folder",
        )

        if not folder:
            return

        self.load_folder(Path(folder))

    def open_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select scanned pages",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp *.gif *.tif *.tiff *.webp)",
        )

        if not files:
            return

        paths = [Path(x) for x in files]
        paths.sort(key=natural_sort_key)

        self.load_images(paths)

    def load_folder(self, folder: Path):
        try:
            paths = [
                path
                for path in folder.iterdir()
                if path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
            ]
        except OSError as exc:
            QMessageBox.critical(
                self,
                "Error",
                f"Could not read folder:\n{exc}",
            )
            return

        paths.sort(key=natural_sort_key)

        self.load_images(paths)

    def load_images(self, paths: list[Path]):
        self.stop_autoplay()

        self.image_paths = paths
        self.spreads = make_spreads(len(paths))
        self.current_spread = 0

        self.cache.clear()
        self.loading.clear()

        if not self.image_paths:
            self.left_label.setText("No images found")
            self.right_label.clear()

            self.status.showMessage("No images found")

            self._update_navigation()
            return

        self.setWindowTitle(
            f"Scanned Book Viewer — "
            f"{self.image_paths[0].parent.name}"
        )

        self._display_current()
        self._update_navigation()

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def _request_image(self, index: int):
        """
        Request an image.

        If it is cached, return immediately.

        Otherwise queue a background load.
        """

        if index < 0 or index >= len(self.image_paths):
            return

        cached = self.cache.get(index)

        if cached is not None:
            return

        if index in self.loading:
            return

        self.loading.add(index)

        signals = ImageLoadSignals()

        # Keep the signal object alive until the task finishes.
        task = ImageLoadTask(
            index,
            self.image_paths[index],
            self.cache,
            signals,
        )

        signals.finished.connect(self._image_loaded)
        signals.failed.connect(self._image_failed)

        self.thread_pool.start(task)

    @Slot(int, object)
    def _image_loaded(self, index: int, image: QImage):
        self.loading.discard(index)

        # The image might have become relevant while it was loading.
        self._display_current()

    @Slot(int, str)
    def _image_failed(self, index: int, message: str):
        self.loading.discard(index)

        print(message, file=sys.stderr)

        # If the failed image is currently visible, redraw so that
        # the placeholder remains visible.
        self._display_current()

    # ------------------------------------------------------------------
    # Preloading
    # ------------------------------------------------------------------

    def _preload_images(self):
        """
        Preload the next N images.

        This deliberately loads images rather than just the next spread.
        This is useful because a large scan can take noticeable time to
        decompress even after the file has been read from disk.
        """

        if not self.spreads:
            return

        current = self.spreads[self.current_spread]

        visible = {current.left}

        if current.right is not None:
            visible.add(current.right)

        # Start from the image after the currently visible spread.
        #
        # We use image indexes rather than spread indexes so that the
        # configurable value really means "N images".
        start = max(visible) + 1

        end = min(
            len(self.image_paths),
            start + self.preload_count,
        )

        for index in range(start, end):
            self._request_image(index)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _display_current(self):
        if not self.spreads:
            return

        spread = self.spreads[self.current_spread]

        self._request_image(spread.left)

        if spread.right is not None:
            self._request_image(spread.right)

        self._set_page_image(
            self.left_label,
            spread.left,
        )

        if spread.right is not None:
            self._set_page_image(
                self.right_label,
                spread.right,
            )
        else:
            self.right_label.clear()

        self._preload_images()

        self._update_navigation()

        # Force repaint after asynchronous image loads.
        self.left_label.update()
        self.right_label.update()

    def _set_page_image(self, label: QLabel, index: int):
        image = self.cache.get(index)

        if image is None:
            label.setText("Loading…")
            label.setPixmap(QPixmap())
            return

        pixmap = QPixmap.fromImage(image)

        available = label.size()

        if available.width() <= 10 or available.height() <= 10:
            label.setPixmap(pixmap)
            return

        scaled = pixmap.scaled(
            available,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        label.setPixmap(scaled)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def previous(self):
        if not self.spreads:
            return

        if self.current_spread <= 0:
            return

        self.current_spread -= 1
        self._display_current()

    def next(self):
        if not self.spreads:
            return

        if self.current_spread >= len(self.spreads) - 1:
            self.stop_autoplay()
            return

        self.current_spread += 1
        self._display_current()

    # ------------------------------------------------------------------
    # Autoplay
    # ------------------------------------------------------------------

    def toggle_autoplay(self):
        if self.autoplay:
            self.stop_autoplay()
        else:
            self.start_autoplay()

    def start_autoplay(self):
        if not self.spreads:
            return

        self.autoplay = True

        interval = max(
            1,
            round(self.time_per_page * 1000),
        )

        self.autoplay_timer.start(interval)

        self.play_action.setText("Pause")

        self.status.showMessage(
            f"Playing — {self.time_per_page:.2f} seconds/page"
        )

    def stop_autoplay(self):
        self.autoplay = False

        self.autoplay_timer.stop()

        self.play_action.setText("Play")

        if self.spreads:
            self.status.showMessage(
                self._status_text()
            )

    def advance(self):
        if not self.spreads:
            self.stop_autoplay()
            return

        if self.current_spread >= len(self.spreads) - 1:
            self.stop_autoplay()
            return

        self.current_spread += 1
        self._display_current()

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def show_settings(self):
        was_playing = self.autoplay

        if was_playing:
            self.stop_autoplay()

        dialog = SettingsDialog(
            self.time_per_page,
            self.preload_count,
            self.cache_size,
            self,
        )

        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.time_per_page = dialog.time_spin.value()
            self.preload_count = dialog.preload_spin.value()
            self.cache_size = dialog.cache_spin.value()

            self.cache.resize(self.cache_size)

            if was_playing:
                self.start_autoplay()

            self._display_current()

        elif was_playing:
            self.start_autoplay()

    # ------------------------------------------------------------------
    # Fullscreen
    # ------------------------------------------------------------------

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    # ------------------------------------------------------------------
    # Status / navigation UI
    # ------------------------------------------------------------------

    def _status_text(self) -> str:
        if not self.image_paths:
            return "No images"

        spread = self.spreads[self.current_spread]

        left_page = spread.left + 1

        if spread.right is not None:
            right_page = spread.right + 1

            pages = f"{left_page}–{right_page}"
        else:
            pages = str(left_page)

        return (
            f"Pages {pages} / {len(self.image_paths)}"
            f"    "
            f"Spread {self.current_spread + 1} / {len(self.spreads)}"
        )

    def _update_navigation(self):
        has_pages = bool(self.spreads)

        self.previous_action.setEnabled(
            has_pages and self.current_spread > 0
        )

        self.next_action.setEnabled(
            has_pages
            and self.current_spread < len(self.spreads) - 1
        )

        if has_pages and not self.autoplay:
            self.status.showMessage(self._status_text())

    # ------------------------------------------------------------------
    # Window resize
    # ------------------------------------------------------------------

    def resizeEvent(self, event):
        super().resizeEvent(event)

        # Re-scale currently visible images without touching the disk.
        if self.spreads:
            spread = self.spreads[self.current_spread]

            self._set_page_image(
                self.left_label,
                spread.left,
            )

            if spread.right is not None:
                self._set_page_image(
                    self.right_label,
                    spread.right,
                )

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------

    def keyPressEvent(self, event):
        key = event.key()

        if key == Qt.Key.Key_Space:
            self.toggle_autoplay()
            event.accept()
            return

        if key == Qt.Key.Key_Left:
            self.previous()
            event.accept()
            return

        if key == Qt.Key.Key_Right:
            self.next()
            event.accept()
            return

        if key == Qt.Key.Key_F11:
            self.toggle_fullscreen()
            event.accept()
            return

        if key == Qt.Key.Key_Escape and self.isFullScreen():
            self.showNormal()
            event.accept()
            return

        super().keyPressEvent(event)

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)

    app.setApplicationName("Scanned Book Viewer")
    app.setOrganizationName("ScannedBookViewer")

    viewer = BookViewer()
    viewer.show()

    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
```

### Controls

| Key / control        | Action                                |
|----------------------|---------------------------------------|
| **Space**            | Play / pause autoplay                 |
| **←**                | Previous spread                       |
| **→**                | Next spread                           |
| **F11**              | Toggle fullscreen                     |
| **Esc**              | Leave fullscreen                      |
| **Ctrl+O**           | Open folder                           |
| **Ctrl+Shift+O**     | Select individual images              |
| Toolbar **Settings** | Configure playback, preload and cache |

The important performance part is that **disk I/O and image decoding happen in `QThreadPool` workers**, while the GUI thread only handles display. The viewer also maintains an LRU cache so going backward doesn't normally require rereading the image from the HDD.

One detail worth noting: the autoplay interval is currently **per displayed spread**, while its setting is named “time per page.” Thus at the default 0.2 seconds, page 2–3 remains on screen for 0.2 seconds before advancing to 4–5. If you want **0.2 seconds per physical page**, the timer logic should instead make each two-page spread stay visible for 0.4 seconds.

## Prompt:

9/9/2026, 5:47:53 PM

nice : )

but the two pages should be joined in the center, so the left image should be aligned to the right, and the right image should be aligned to the left

also, the image-preloading is not aggressive enough, i still get lags (from loading images) in autoplay mode

## Response:

9/9/2026, 5:47:58 PM

> **Improving preload behavior**
>
> I’m revising the script so autoplay buffers upcoming spreads before advancing, while preserving the existing project ID and avoiding excessive memory use. I’m also checking cache size, loading order, and decode timing to reduce visible pauses.
>
> **Improved preload behavior**
>
> ------------------------------------------------------------------------
>
> **Enhancing preload buffering and spread alignment**
>
> **Enhanced preload buffering and spread alignment**
>
> Worked for 9s

Absolutely : ) There are two changes I'd make:

1.  **Book gutter:** remove the gap between the two page widgets, align the left page to the **right edge** and the right page to the **left edge**. This makes the two scanned pages meet exactly in the center.
2.  **Autoplay buffering:** don't let autoplay advance into an unloaded image. The viewer now aggressively queues a configurable number of *future spreads*, and autoplay waits if the next spread isn't decoded yet rather than producing a visible loading hitch. I've also increased the default cache/preload values.

I also corrected the book ordering so that **page 1 is the right-hand page**, then pages 2–3 form a spread, 4–5 another spread, etc.

Here is the revised complete script:

\*\*document: \*\*

``` python
#!/usr/bin/env python3

"""
Scanned Book Image Viewer
=========================

PySide6 image viewer specialized for scanned book pages.

Book layout:

    page 1                  -> right side
    pages 2 + 3             -> left + right
    pages 4 + 5             -> left + right
    pages 6 + 7             -> left + right
    ...
    final even page         -> left side

Features:

    - PySide6
    - Book-like two-page layout
    - Pages meet exactly at the center gutter
    - Spacebar = autoplay / pause
    - Configurable autoplay speed
    - Left/right seeking
    - Aggressive background preloading
    - Autoplay waits for the next spread to be decoded
    - LRU decoded-image cache
    - Natural filename sorting
    - Folder or individual-file loading
    - Fullscreen mode
    - Works well with slow HDDs

Install:

    pip install PySide6

Run:

    python book_viewer.py
"""

from __future__ import annotations

import re
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import (
    QObject,
    QRunnable,
    QThreadPool,
    QTimer,
    Qt,
    QMutex,
    Signal,
    Slot,
)
from PySide6.QtGui import QAction, QImage, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStatusBar,
    QToolBar,
)

# ============================================================================
# Configuration
# ============================================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
    ".webp",
}

DEFAULT_TIME_PER_PAGE = 0.20

# Number of FUTURE SPREADS to aggressively buffer.
#
# 20 spreads means potentially 40 decoded images waiting in RAM.
#
# Increase this if you have lots of RAM and a particularly slow disk.
DEFAULT_PRELOAD_SPREADS = 20

# Number of decoded images kept in RAM.
#
# This should normally be >= 2 * PRELOAD_SPREADS.
DEFAULT_CACHE_SIZE = 60

# ============================================================================
# Natural sorting
# ============================================================================

def natural_sort_key(path: Path):
    """
    Sort:

        page1.jpg
        page2.jpg
        page10.jpg

    in human/numeric order.
    """

    parts = re.split(r"(\d+)", path.name.lower())

    result = []

    for part in parts:
        if part.isdigit():
            result.append((0, int(part)))
        else:
            result.append((1, part))

    return result

# ============================================================================
# Book spreads
# ============================================================================

@dataclass(frozen=True)
class Spread:
    """
    A displayable book spread.

    left/right are zero-based indexes into image_paths.

    Examples:

        Spread(None, 0)       -> page 1
        Spread(1, 2)          -> pages 2 + 3
        Spread(3, 4)          -> pages 4 + 5
        Spread(5, None)       -> final even page
    """

    left: int | None
    right: int | None

def make_spreads(page_count: int) -> list[Spread]:
    """
    Construct proper book-style spreads.

    Page 1 is the first/right-hand page.

    Then:

        pages 2 + 3
        pages 4 + 5
        pages 6 + 7
        ...

    If the book ends on an even page, that final page appears alone
    on the left.
    """

    if page_count <= 0:
        return []

    spreads: list[Spread] = []

    # First page is alone on the right.
    spreads.append(
        Spread(
            left=None,
            right=0,
        )
    )

    page = 1

    while page < page_count:
        left = page

        if page + 1 < page_count:
            right = page + 1
            spreads.append(Spread(left, right))
            page += 2
        else:
            # Final even page.
            spreads.append(
                Spread(
                    left=left,
                    right=None,
                )
            )
            page += 1

    return spreads

# ============================================================================
# Thread-safe LRU image cache
# ============================================================================

class ImageCache:
    """
    Thread-safe LRU cache of decoded QImages.

    The cache stores decoded images rather than QPixmaps.

    This is important because QPixmap is a GUI resource and should remain
    on the GUI thread, while QImage can be loaded/decoded in worker threads.
    """

    def __init__(self, max_items: int):
        self.max_items = max(1, max_items)

        self._cache: OrderedDict[int, QImage] = OrderedDict()
        self._mutex = QMutex()

    def get(self, index: int) -> QImage | None:
        self._mutex.lock()

        try:
            image = self._cache.get(index)

            if image is not None:
                self._cache.move_to_end(index)

            return image

        finally:
            self._mutex.unlock()

    def contains(self, index: int) -> bool:
        self._mutex.lock()

        try:
            return index in self._cache

        finally:
            self._mutex.unlock()

    def put(self, index: int, image: QImage):
        self._mutex.lock()

        try:
            self._cache[index] = image
            self._cache.move_to_end(index)

            while len(self._cache) > self.max_items:
                self._cache.popitem(last=False)

        finally:
            self._mutex.unlock()

    def clear(self):
        self._mutex.lock()

        try:
            self._cache.clear()

        finally:
            self._mutex.unlock()

    def resize(self, size: int):
        self._mutex.lock()

        try:
            self.max_items = max(1, size)

            while len(self._cache) > self.max_items:
                self._cache.popitem(last=False)

        finally:
            self._mutex.unlock()

# ============================================================================
# Background loading
# ============================================================================

class ImageLoadSignals(QObject):
    finished = Signal(int, object)
    failed = Signal(int, str)

class ImageLoadTask(QRunnable):
    """
    Decode one image in a background worker.
    """

    def __init__(
        self,
        index: int,
        path: Path,
        cache: ImageCache,
        signals: ImageLoadSignals,
    ):
        super().__init__()

        self.index = index
        self.path = path
        self.cache = cache
        self.signals = signals

        self.setAutoDelete(True)

    @Slot()
    def run(self):
        # Another worker may have loaded it already.
        cached = self.cache.get(self.index)

        if cached is not None:
            self.signals.finished.emit(
                self.index,
                cached,
            )
            return

        image = QImage(str(self.path))

        if image.isNull():
            self.signals.failed.emit(
                self.index,
                f"Could not load image:\n{self.path}",
            )
            return

        # Force the image data to become detached/fully materialized.
        #
        # This is useful with formats such as JPEG/TIFF because we want
        # decoding to happen HERE, rather than lazily later in the GUI
        # thread.
        image = image.copy()

        self.cache.put(
            self.index,
            image,
        )

        self.signals.finished.emit(
            self.index,
            image,
        )

# ============================================================================
# Settings dialog
# ============================================================================

class SettingsDialog(QDialog):

    def __init__(
        self,
        time_per_page: float,
        preload_spreads: int,
        cache_size: int,
        parent=None,
    ):
        super().__init__(parent)

        self.setWindowTitle("Viewer Settings")

        layout = QFormLayout(self)

        # ---------------------------------------------------------------
        # Autoplay speed
        # ---------------------------------------------------------------

        self.time_spin = QDoubleSpinBox()

        self.time_spin.setRange(
            0.05,
            60.0,
        )

        self.time_spin.setSingleStep(
            0.05,
        )

        self.time_spin.setDecimals(
            2,
        )

        self.time_spin.setValue(
            time_per_page,
        )

        self.time_spin.setSuffix(
            " s",
        )

        # ---------------------------------------------------------------
        # Preloading
        # ---------------------------------------------------------------

        self.preload_spin = QSpinBox()

        self.preload_spin.setRange(
            0,
            200,
        )

        self.preload_spin.setValue(
            preload_spreads,
        )

        self.preload_spin.setSuffix(
            " spreads",
        )

        # ---------------------------------------------------------------
        # Cache
        # ---------------------------------------------------------------

        self.cache_spin = QSpinBox()

        self.cache_spin.setRange(
            4,
            500,
        )

        self.cache_spin.setValue(
            cache_size,
        )

        self.cache_spin.setSuffix(
            " images",
        )

        layout.addRow(
            "Time per page:",
            self.time_spin,
        )

        layout.addRow(
            "Preload next:",
            self.preload_spin,
        )

        layout.addRow(
            "RAM image cache:",
            self.cache_spin,
        )

        buttons = QHBoxLayout()

        cancel = QPushButton("Cancel")
        ok = QPushButton("OK")

        cancel.clicked.connect(
            self.reject,
        )

        ok.clicked.connect(
            self.accept,
        )

        buttons.addStretch()
        buttons.addWidget(cancel)
        buttons.addWidget(ok)

        layout.addRow(buttons)

# ============================================================================
# Main viewer
# ============================================================================

class BookViewer(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle(
            "Scanned Book Viewer"
        )

        self.resize(
            1400,
            900,
        )

        # ---------------------------------------------------------------
        # Book state
        # ---------------------------------------------------------------

        self.image_paths: list[Path] = []

        self.spreads: list[Spread] = []

        self.current_spread = 0

        # ---------------------------------------------------------------
        # Playback settings
        # ---------------------------------------------------------------

        self.time_per_page = DEFAULT_TIME_PER_PAGE

        self.preload_spreads = DEFAULT_PRELOAD_SPREADS

        self.cache_size = DEFAULT_CACHE_SIZE

        # ---------------------------------------------------------------
        # Image cache
        # ---------------------------------------------------------------

        self.cache = ImageCache(
            self.cache_size
        )

        # ---------------------------------------------------------------
        # Background worker pool
        # ---------------------------------------------------------------

        self.thread_pool = QThreadPool.globalInstance()

        # Allow several images to be decoded concurrently.

        # NOTE:
        # More threads are not ALWAYS faster on an HDD. The default
        # QThreadPool value is usually reasonable. We nevertheless allow
        # a few concurrent decoders because modern systems often have
        # enough CPU to decode JPEG/TIFF while another read is happening.
        self.thread_pool.setMaxThreadCount(
            max(
                2,
                min(
                    4,
                    self.thread_pool.maxThreadCount(),
                ),
            )
        )

        # Image indexes currently queued/loading.
        self.loading: set[int] = set()

        # ---------------------------------------------------------------
        # Autoplay
        # ---------------------------------------------------------------

        self.autoplay = False

        self.autoplay_timer = QTimer(
            self
        )

        self.autoplay_timer.setSingleShot(
            False
        )

        self.autoplay_timer.timeout.connect(
            self.advance
        )

        # ---------------------------------------------------------------
        # Page widgets
        # ---------------------------------------------------------------

        self.left_label = self._create_page_label(
            Qt.AlignmentFlag.AlignRight
        )

        self.right_label = self._create_page_label(
            Qt.AlignmentFlag.AlignLeft
        )

        # The central widget itself has NO spacing.
        #
        # This gives us an exact center gutter:
        #
        #      [ LEFT PAGE ][ RIGHT PAGE ]
        #                    ^
        #                  gutter
        #
        container = QWidget()

        layout = QHBoxLayout(
            container
        )

        layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        # CRITICAL: no gap between the pages.
        layout.setSpacing(0)

        layout.addWidget(
            self.left_label,
            1,
        )

        layout.addWidget(
            self.right_label,
            1,
        )

        self.setCentralWidget(
            container
        )

        # ---------------------------------------------------------------
        # Actions
        # ---------------------------------------------------------------

        self._create_actions()
        self._create_toolbar()

        # ---------------------------------------------------------------
        # Status bar
        # ---------------------------------------------------------------

        self.status = QStatusBar()

        self.setStatusBar(
            self.status
        )

        self.status.showMessage(
            "Open a folder to begin"
        )

        self._update_navigation()

    # ==================================================================
    # UI
    # ==================================================================

    def _create_page_label(
        self,
        alignment: Qt.AlignmentFlag,
    ) -> QLabel:

        label = QLabel()

        label.setAlignment(
            alignment | Qt.AlignmentFlag.AlignVCenter
        )

        label.setMinimumSize(
            100,
            100,
        )

        label.setStyleSheet(
            """
            QLabel {
                background-color: #202020;
                color: #888888;
                border: none;
            }
            """
        )

        return label

    def _create_actions(self):

        self.open_folder_action = QAction(
            "Open Folder",
            self,
        )

        self.open_folder_action.setShortcut(
            QKeySequence("Ctrl+O")
        )

        self.open_folder_action.triggered.connect(
            self.open_folder
        )

        # ---------------------------------------------------------------

        self.open_files_action = QAction(
            "Open Images",
            self,
        )

        self.open_files_action.setShortcut(
            QKeySequence("Ctrl+Shift+O")
        )

        self.open_files_action.triggered.connect(
            self.open_files
        )

        # ---------------------------------------------------------------

        self.previous_action = QAction(
            "Previous",
            self,
        )

        self.previous_action.setShortcut(
            QKeySequence("Left")
        )

        self.previous_action.triggered.connect(
            self.previous
        )

        # ---------------------------------------------------------------

        self.play_action = QAction(
            "Play",
            self,
        )

        self.play_action.setShortcut(
            QKeySequence("Space")
        )

        self.play_action.triggered.connect(
            self.toggle_autoplay
        )

        # ---------------------------------------------------------------

        self.next_action = QAction(
            "Next",
            self,
        )

        self.next_action.setShortcut(
            QKeySequence("Right")
        )

        self.next_action.triggered.connect(
            self.next
        )

        # ---------------------------------------------------------------

        self.settings_action = QAction(
            "Settings",
            self,
        )

        self.settings_action.triggered.connect(
            self.show_settings
        )

        # ---------------------------------------------------------------

        self.fullscreen_action = QAction(
            "Fullscreen",
            self,
        )

        self.fullscreen_action.setShortcut(
            QKeySequence("F11")
        )

        self.fullscreen_action.triggered.connect(
            self.toggle_fullscreen
        )

        # ---------------------------------------------------------------

        self.quit_action = QAction(
            "Quit",
            self,
        )

        self.quit_action.setShortcut(
            QKeySequence("Ctrl+Q")
        )

        self.quit_action.triggered.connect(
            self.close
        )

    def _create_toolbar(self):

        toolbar = QToolBar(
            "Controls"
        )

        toolbar.setMovable(
            False
        )

        toolbar.addAction(
            self.open_folder_action
        )

        toolbar.addAction(
            self.open_files_action
        )

        toolbar.addSeparator()

        toolbar.addAction(
            self.previous_action
        )

        toolbar.addAction(
            self.play_action
        )

        toolbar.addAction(
            self.next_action
        )

        toolbar.addSeparator()

        toolbar.addAction(
            self.settings_action
        )

        toolbar.addAction(
            self.fullscreen_action
        )

        self.addToolBar(
            toolbar
        )

    # ==================================================================
    # Opening files
    # ==================================================================

    def open_folder(self):

        folder = QFileDialog.getExistingDirectory(
            self,
            "Select scanned book folder",
        )

        if not folder:
            return

        self.load_folder(
            Path(folder)
        )

    def open_files(self):

        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select scanned pages",
            "",
            (
                "Images "
                "(*.jpg *.jpeg *.png *.bmp *.gif "
                "*.tif *.tiff *.webp)"
            ),
        )

        if not files:
            return

        paths = [
            Path(filename)
            for filename in files
        ]

        paths.sort(
            key=natural_sort_key
        )

        self.load_images(
            paths
        )

    def load_folder(
        self,
        folder: Path,
    ):

        try:
            paths = [
                path
                for path in folder.iterdir()
                if (
                    path.is_file()
                    and path.suffix.lower()
                    in IMAGE_EXTENSIONS
                )
            ]

        except OSError as exc:

            QMessageBox.critical(
                self,
                "Error",
                f"Could not read folder:\n{exc}",
            )

            return

        paths.sort(
            key=natural_sort_key
        )

        self.load_images(
            paths
        )

    def load_images(
        self,
        paths: list[Path],
    ):

        self.stop_autoplay()

        self.image_paths = paths

        self.spreads = make_spreads(
            len(paths)
        )

        self.current_spread = 0

        self.cache.clear()

        self.loading.clear()

        if not paths:

            self.left_label.setText(
                "No images found"
            )

            self.right_label.clear()

            self.status.showMessage(
                "No images found"
            )

            self._update_navigation()

            return

        self.setWindowTitle(
            "Scanned Book Viewer — "
            + paths[0].parent.name
        )

        # Immediately request the first visible images.
        self._display_current()

        # And aggressively begin filling the buffer.
        self._preload_images()

    # ==================================================================
    # Image loading
    # ==================================================================

    def _request_image(
        self,
        index: int,
    ):

        if (
            index < 0
            or index >= len(self.image_paths)
        ):
            return

        # Already decoded.
        if self.cache.contains(index):
            return

        # Already queued.
        if index in self.loading:
            return

        self.loading.add(index)

        signals = ImageLoadSignals()

        task = ImageLoadTask(
            index=index,
            path=self.image_paths[index],
            cache=self.cache,
            signals=signals,
        )

        signals.finished.connect(
            self._image_loaded
        )

        signals.failed.connect(
            self._image_failed
        )

        self.thread_pool.start(
            task
        )

    @Slot(int, object)
    def _image_loaded(
        self,
        index: int,
        image: QImage,
    ):

        self.loading.discard(
            index
        )

        # If this image is currently visible, redraw it.
        spread = (
            self.spreads[self.current_spread]
            if self.spreads
            else None
        )

        if spread is not None:
            if index in (
                spread.left,
                spread.right,
            ):
                self._display_current()

    @Slot(int, str)
    def _image_failed(
        self,
        index: int,
        message: str,
    ):

        self.loading.discard(
            index
        )

        print(
            message,
            file=sys.stderr,
        )

    # ==================================================================
    # Aggressive preloading
    # ==================================================================

    def _preload_images(self):

        if not self.spreads:
            return

        """
        We preload by SPREAD rather than merely by image count.

        For example, if we are currently viewing:

            100 + 101

        and preload_spreads == 20,

        we queue:

            102 ... 141

        This gives autoplay a substantial decoded buffer.
        """

        start_spread = (
            self.current_spread + 1
        )

        end_spread = min(
            len(self.spreads),
            start_spread + self.preload_spreads,
        )

        for spread_index in range(
            start_spread,
            end_spread,
        ):

            spread = self.spreads[
                spread_index
            ]

            if spread.left is not None:
                self._request_image(
                    spread.left
                )

            if spread.right is not None:
                self._request_image(
                    spread.right
                )

        # Also make sure the currently visible pages are requested.
        current = self.spreads[
            self.current_spread
        ]

        if current.left is not None:
            self._request_image(
                current.left
            )

        if current.right is not None:
            self._request_image(
                current.right
            )

    # ==================================================================
    # Display
    # ==================================================================

    def _display_current(self):

        if not self.spreads:
            return

        spread = self.spreads[
            self.current_spread
        ]

        # ---------------------------------------------------------------
        # LEFT PAGE
        # ---------------------------------------------------------------

        if spread.left is None:

            self.left_label.clear()

        else:

            self._set_page_image(
                self.left_label,
                spread.left,
            )

        # ---------------------------------------------------------------
        # RIGHT PAGE
        # ---------------------------------------------------------------

        if spread.right is None:

            self.right_label.clear()

        else:

            self._set_page_image(
                self.right_label,
                spread.right,
            )

        # ---------------------------------------------------------------
        # Keep the future buffer full.
        # ---------------------------------------------------------------

        self._preload_images()

        self._update_navigation()

    def _set_page_image(
        self,
        label: QLabel,
        index: int,
    ):

        image = self.cache.get(
            index
        )

        if image is None:

            label.setPixmap(
                QPixmap()
            )

            label.setText(
                "Loading…"
            )

            return

        pixmap = QPixmap.fromImage(
            image
        )

        available = label.size()

        if (
            available.width() <= 10
            or available.height() <= 10
        ):
            label.setPixmap(
                pixmap
            )
            return

        scaled = pixmap.scaled(
            available,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        label.setText("")

        label.setPixmap(
            scaled
        )

    # ==================================================================
    # Autoplay buffering
    # ==================================================================

    def _next_spread_is_ready(self) -> bool:

        next_index = (
            self.current_spread + 1
        )

        if (
            next_index >= len(self.spreads)
        ):
            return False

        spread = self.spreads[
            next_index
        ]

        if spread.left is not None:
            if not self.cache.contains(
                spread.left
            ):
                return False

        if spread.right is not None:
            if not self.cache.contains(
                spread.right
            ):
                return False

        return True

    def toggle_autoplay(self):

        if self.autoplay:
            self.stop_autoplay()

        else:
            self.start_autoplay()

    def start_autoplay(self):

        if not self.spreads:
            return

        self.autoplay = True

        # Immediately refill the buffer before starting.
        self._preload_images()

        interval = max(
            1,
            round(
                self.time_per_page * 1000
            ),
        )

        self.autoplay_timer.start(
            interval
        )

        self.play_action.setText(
            "Pause"
        )

        self.status.showMessage(
            "Playing — "
            f"{self.time_per_page:.2f} s/page"
        )

    def stop_autoplay(self):

        self.autoplay = False

        self.autoplay_timer.stop()

        self.play_action.setText(
            "Play"
        )

        if self.spreads:
            self.status.showMessage(
                self._status_text()
            )

    def advance(self):

        if not self.spreads:
            self.stop_autoplay()
            return

        # ---------------------------------------------------------------
        # END OF BOOK
        # ---------------------------------------------------------------

        if (
            self.current_spread
            >= len(self.spreads) - 1
        ):
            self.stop_autoplay()
            return

        # ---------------------------------------------------------------
        # IMPORTANT:
        #
        # Do NOT advance into an unloaded image.
        #
        # Without this check, the GUI can display "Loading…" during
        # autoplay, which is exactly the visible hitch we want to avoid.
        #
        # The background workers continue loading while we wait.
        # ---------------------------------------------------------------

        if not self._next_spread_is_ready():

            self.status.showMessage(
                "Buffering… "
                f"{len(self.loading)} images loading"
            )

            # Keep requesting aggressively.
            self._preload_images()

            return

        # ---------------------------------------------------------------
        # Next spread is completely decoded.
        # ---------------------------------------------------------------

        self.current_spread += 1

        self._display_current()

    # ==================================================================
    # Navigation
    # ==================================================================

    def previous(self):

        if not self.spreads:
            return

        if self.current_spread <= 0:
            return

        self.current_spread -= 1

        self._display_current()

    def next(self):

        if not self.spreads:
            return

        if (
            self.current_spread
            >= len(self.spreads) - 1
        ):
            self.stop_autoplay()
            return

        self.current_spread += 1

        self._display_current()

    # ==================================================================
    # Settings
    # ==================================================================

    def show_settings(self):

        was_playing = self.autoplay

        if was_playing:
            self.stop_autoplay()

        dialog = SettingsDialog(
            self.time_per_page,
            self.preload_spreads,
            self.cache_size,
            self,
        )

        if (
            dialog.exec()
            == QDialog.DialogCode.Accepted
        ):

            self.time_per_page = (
                dialog.time_spin.value()
            )

            self.preload_spreads = (
                dialog.preload_spin.value()
            )

            self.cache_size = (
                dialog.cache_spin.value()
            )

            self.cache.resize(
                self.cache_size
            )

            # Fill the newly configured buffer.
            self._preload_images()

            if was_playing:
                self.start_autoplay()

            self._display_current()

        elif was_playing:

            self.start_autoplay()

    # ==================================================================
    # Fullscreen
    # ==================================================================

    def toggle_fullscreen(self):

        if self.isFullScreen():
            self.showNormal()

        else:
            self.showFullScreen()

    # ==================================================================
    # Status
    # ==================================================================

    def _status_text(self) -> str:

        if not self.image_paths:
            return "No images"

        spread = self.spreads[
            self.current_spread
        ]

        if spread.left is None:

            pages = f"{spread.right + 1}"

        elif spread.right is None:

            pages = f"{spread.left + 1}"

        else:

            pages = (
                f"{spread.left + 1}"
                "–"
                f"{spread.right + 1}"
            )

        return (
            f"Pages {pages} / "
            f"{len(self.image_paths)}"
            "    "
            f"Spread "
            f"{self.current_spread + 1}"
            " / "
            f"{len(self.spreads)}"
        )

    def _update_navigation(self):

        has_pages = bool(
            self.spreads
        )

        self.previous_action.setEnabled(
            has_pages
            and self.current_spread > 0
        )

        self.next_action.setEnabled(
            has_pages
            and self.current_spread
            < len(self.spreads) - 1
        )

        if (
            has_pages
            and not self.autoplay
        ):
            self.status.showMessage(
                self._status_text()
            )

    # ==================================================================
    # Resize
    # ==================================================================

    def resizeEvent(self, event):

        super().resizeEvent(
            event
        )

        if not self.spreads:
            return

        spread = self.spreads[
            self.current_spread
        ]

        if spread.left is not None:
            self._set_page_image(
                self.left_label,
                spread.left,
            )

        else:
            self.left_label.clear()

        if spread.right is not None:
            self._set_page_image(
                self.right_label,
                spread.right,
            )

        else:
            self.right_label.clear()

    # ==================================================================
    # Keyboard
    # ==================================================================

    def keyPressEvent(self, event):

        key = event.key()

        if key == Qt.Key.Key_Space:

            self.toggle_autoplay()

            event.accept()

            return

        if key == Qt.Key.Key_Left:

            self.previous()

            event.accept()

            return

        if key == Qt.Key.Key_Right:

            self.next()

            event.accept()

            return

        if key == Qt.Key.Key_F11:

            self.toggle_fullscreen()

            event.accept()

            return

        if (
            key == Qt.Key.Key_Escape
            and self.isFullScreen()
        ):

            self.showNormal()

            event.accept()

            return

        super().keyPressEvent(
            event
        )

# ============================================================================
# Main
# ============================================================================

def main():

    app = QApplication(
        sys.argv
    )

    app.setApplicationName(
        "Scanned Book Viewer"
    )

    app.setOrganizationName(
        "ScannedBookViewer"
    )

    viewer = BookViewer()

    viewer.show()

    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
```

### What changed

**Center gutter:** the layout now uses `spacing = 0`, with the left `QLabel` right-aligned and the right `QLabel` left-aligned. So a spread behaves like:

``` text
             center
               │
    ┌──────────┼──────────┐
    │          │          │
    │  page 2  │  page 3  │
    │          │          │
    └──────────┼──────────┘
               │
```

**Much more aggressive buffering:** the default is now **20 future spreads**, potentially 40 decoded images, with a **60-image RAM cache**.

More importantly, autoplay now has a **buffer barrier**. It won't advance to the next spread until both images are already decoded:

``` text
             HDD
              │
              ▼
       background workers
              │
              ▼
        decoded QImages
              │
              ▼
          RAM cache
              │
              ▼
          ┌─────────┐
          │ DISPLAY │
          └─────────┘
```

So instead of:

``` text
display → disk → decode → stutter → display
```

you get:

``` text
display → display → display → display
              ↑
        background loading
```

If the HDD is temporarily too slow, autoplay will **pause at the already-decoded boundary and show `Buffering…` rather than visibly stuttering**.

One further optimization I'd strongly recommend for a scanned-book viewer is **pre-scaling the pages to the window's display resolution during background loading**. With very large 300–600 DPI scans, JPEG/TIFF decoding can be fast enough but repeatedly scaling a 5,000–8,000 px image for every frame can itself become a bottleneck. A next version could maintain a second, display-sized cache so the GUI can swap pages essentially instantaneously.

## Prompt:

9/9/2026, 5:56:25 PM

nice

in fullscreen mode, it should be "headless", so the menubar and statusbar should be hidden by default, and they should only be visible when the user moves the mouse to the top or bottom

currently the pages look a bit blurry... the images are high-quality scans (600 dpi), so i would expect the text to be sharper

it would be nice to have a zoom-and-pan feature, where i can zoom-and-pan both pages (left and right) as if they were one image, and the zoom-and-pan state is preserved when i go to the next or previous page. the zoom-and-pan feature should be controlled with the obvious methods (mouse wheel, Ctrl++, Ctrl+-, click and drag)

## Response:

9/9/2026, 5:56:26 PM

> **Printing a simple response**
>
> **Printing a simple response**
>
> Worked for a couple of seconds

Yep — this is now getting into proper book-reader territory.

The blur is mainly caused by the previous implementation doing `QPixmap.scaled(... SmoothTransformation)` every time the window is painted/resized. For 600-DPI scans, I would **keep the original decoded `QImage` at full resolution**, and use a high-quality transformation only for the final display. For zooming, we should avoid repeatedly modifying the source image.

The zoom/pan implementation below treats the two pages as **one continuous book spread**. That means:

- mouse wheel → zoom around the mouse cursor
- `Ctrl +` / `Ctrl -` → zoom
- left mouse drag → pan
- zoom can cross the center gutter
- the left and right pages move together
- zoom/pan state survives `←` / `→`
- at 100% you see actual scan pixels rather than an unnecessarily resampled image
- fullscreen hides the UI chrome
- moving the mouse near the top/bottom temporarily reveals the toolbar/status bar
- moving away hides them again
- the UI automatically hides after a short timeout

One subtle change is that I'm using a custom `QWidget` for the book canvas rather than two `QLabel`s. That's considerably better for implementing a genuine continuous two-page viewport.

\*\*document: \*\*

``` python
#!/usr/bin/env python3

"""
Scanned Book Viewer
===================

PySide6 viewer specialized for high-resolution scanned book pages.

Features
--------

Book layout:

    page 1                  -> right
    pages 2 + 3             -> left + right
    pages 4 + 5             -> left + right
    ...

Fullscreen:

    - Toolbar/menu/status UI hidden by default
    - Move mouse to top    -> toolbar appears
    - Move mouse to bottom -> status bar appears
    - UI hides automatically

Navigation:

    Left / Right arrows
    Space = autoplay

Zoom:

    Mouse wheel
    Ctrl + Plus
    Ctrl + Minus
    Ctrl + 0 = reset zoom

Pan:

    Left mouse button + drag

Zoom and pan apply to the COMPLETE SPREAD as one continuous image.

The zoom/pan state is preserved when changing pages.

Performance:

    - Original-resolution QImages are cached
    - Background decoding using QThreadPool
    - Aggressive future-spread preloading
    - Autoplay never advances into an undecoded spread
    - LRU RAM cache
    - High-quality display scaling
"""

from __future__ import annotations

import re
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import (
    QObject,
    QPointF,
    QRectF,
    QRunnable,
    QSize,
    Qt,
    QThreadPool,
    QTimer,
    QMutex,
    Signal,
    Slot,
)
from PySide6.QtGui import (
    QAction,
    QImage,
    QKeySequence,
    QPainter,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStatusBar,
    QToolBar,
    QWidget,
)

# ============================================================================
# Configuration
# ============================================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
    ".webp",
}

DEFAULT_TIME_PER_PAGE = 0.20

# Number of FUTURE spreads to decode in advance.
DEFAULT_PRELOAD_SPREADS = 20

# Number of full-resolution images retained in RAM.
DEFAULT_CACHE_SIZE = 60

# Fullscreen UI reveal distance.
EDGE_REVEAL_PIXELS = 35

# How long revealed fullscreen chrome remains visible.
CHROME_HIDE_DELAY_MS = 1800

# Zoom limits.
MIN_ZOOM = 0.10
MAX_ZOOM = 8.0

# Zoom multiplier per wheel step.
ZOOM_FACTOR = 1.15

# ============================================================================
# Natural sorting
# ============================================================================

def natural_sort_key(path: Path):
    parts = re.split(
        r"(\d+)",
        path.name.lower(),
    )

    result = []

    for part in parts:
        if part.isdigit():
            result.append(
                (0, int(part))
            )
        else:
            result.append(
                (1, part)
            )

    return result

# ============================================================================
# Spread model
# ============================================================================

@dataclass(frozen=True)
class Spread:
    """
    left/right are zero-based indexes into image_paths.

    Page 1 is on the right.

        Spread(None, 0)
        Spread(1, 2)
        Spread(3, 4)
        ...
    """

    left: int | None
    right: int | None

def make_spreads(page_count: int) -> list[Spread]:

    if page_count <= 0:
        return []

    spreads = []

    # First page is alone on the right.
    spreads.append(
        Spread(
            left=None,
            right=0,
        )
    )

    page = 1

    while page < page_count:

        left = page

        if page + 1 < page_count:

            spreads.append(
                Spread(
                    left=left,
                    right=page + 1,
                )
            )

            page += 2

        else:

            # Final even page.
            spreads.append(
                Spread(
                    left=left,
                    right=None,
                )
            )

            page += 1

    return spreads

# ============================================================================
# Image cache
# ============================================================================

class ImageCache:

    def __init__(self, max_items: int):

        self.max_items = max(
            1,
            max_items,
        )

        self._cache: OrderedDict[
            int,
            QImage,
        ] = OrderedDict()

        self._mutex = QMutex()

    def get(
        self,
        index: int,
    ) -> QImage | None:

        self._mutex.lock()

        try:

            image = self._cache.get(
                index
            )

            if image is not None:
                self._cache.move_to_end(
                    index
                )

            return image

        finally:

            self._mutex.unlock()

    def contains(
        self,
        index: int,
    ) -> bool:

        self._mutex.lock()

        try:
            return index in self._cache

        finally:
            self._mutex.unlock()

    def put(
        self,
        index: int,
        image: QImage,
    ):

        self._mutex.lock()

        try:

            self._cache[index] = image

            self._cache.move_to_end(
                index
            )

            while (
                len(self._cache)
                > self.max_items
            ):
                self._cache.popitem(
                    last=False
                )

        finally:

            self._mutex.unlock()

    def clear(self):

        self._mutex.lock()

        try:
            self._cache.clear()

        finally:
            self._mutex.unlock()

    def resize(
        self,
        size: int,
    ):

        self._mutex.lock()

        try:

            self.max_items = max(
                1,
                size,
            )

            while (
                len(self._cache)
                > self.max_items
            ):
                self._cache.popitem(
                    last=False
                )

        finally:

            self._mutex.unlock()

# ============================================================================
# Background image loading
# ============================================================================

class ImageLoadSignals(QObject):

    finished = Signal(
        int,
        object,
    )

    failed = Signal(
        int,
        str,
    )

class ImageLoadTask(QRunnable):

    def __init__(
        self,
        index: int,
        path: Path,
        cache: ImageCache,
        signals: ImageLoadSignals,
    ):

        super().__init__()

        self.index = index
        self.path = path
        self.cache = cache
        self.signals = signals

        self.setAutoDelete(True)

    @Slot()
    def run(self):

        cached = self.cache.get(
            self.index
        )

        if cached is not None:

            self.signals.finished.emit(
                self.index,
                cached,
            )

            return

        image = QImage(
            str(self.path)
        )

        if image.isNull():

            self.signals.failed.emit(
                self.index,
                f"Could not load:\n{self.path}",
            )

            return

        # Force complete decoding now.
        #
        # This is particularly useful for JPEG/TIFF scans. We don't want
        # any deferred work to happen later in the GUI thread.
        image = image.copy()

        self.cache.put(
            self.index,
            image,
        )

        self.signals.finished.emit(
            self.index,
            image,
        )

# ============================================================================
# Book canvas
# ============================================================================

class BookCanvas(QWidget):
    """
    Displays a spread as one continuous coordinate system.

    This is the important part of the zoom/pan implementation.

    Conceptually the spread is:

        ┌───────────────────────┐
        │       LEFT │ RIGHT    │
        │            │          │
        └───────────────────────┘

    The center gutter is x=0 in book coordinates.
    """

    def __init__(self, viewer):

        super().__init__(
            viewer
        )

        self.viewer = viewer

        self.setMouseTracking(
            True
        )

        self.setFocusPolicy(
            Qt.FocusPolicy.StrongFocus
        )

        self.zoom = 1.0

        # Pan is represented in screen pixels.
        self.pan = QPointF(
            0.0,
            0.0,
        )

        self.dragging = False

        self.last_mouse = QPointF()

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def _page_size(
        self,
        image: QImage,
    ) -> QSize:

        return QSize(
            image.width(),
            image.height(),
        )

    def _spread_geometry(self):
        """
        Return geometry of the current spread in source-image pixels.

        Returns:

            left_image
            right_image
            total_width
            max_height
            left_x
            right_x
        """

        spread = self.viewer.current_spread_object()

        if spread is None:
            return None

        left_image = (
            self.viewer.get_cached_image(
                spread.left
            )
            if spread.left is not None
            else None
        )

        right_image = (
            self.viewer.get_cached_image(
                spread.right
            )
            if spread.right is not None
            else None
        )

        left_width = (
            left_image.width()
            if left_image is not None
            else 0
        )

        right_width = (
            right_image.width()
            if right_image is not None
            else 0
        )

        left_height = (
            left_image.height()
            if left_image is not None
            else 0
        )

        right_height = (
            right_image.height()
            if right_image is not None
            else 0
        )

        total_width = (
            left_width
            + right_width
        )

        max_height = max(
            left_height,
            right_height,
        )

        # The center of the spread is x=0.
        left_x = -left_width
        right_x = 0

        return (
            left_image,
            right_image,
            total_width,
            max_height,
            left_x,
            right_x,
        )

    # ------------------------------------------------------------------
    # Coordinate conversion
    # ------------------------------------------------------------------

    def _book_to_screen(
        self,
        x: float,
        y: float,
    ) -> QPointF:

        center_x = self.width() / 2.0
        center_y = self.height() / 2.0

        return QPointF(
            center_x
            + self.pan.x()
            + x * self.zoom,

            center_y
            + self.pan.y()
            + y * self.zoom,
        )

    def _screen_to_book(
        self,
        point: QPointF,
    ) -> QPointF:

        center_x = self.width() / 2.0
        center_y = self.height() / 2.0

        return QPointF(
            (
                point.x()
                - center_x
                - self.pan.x()
            )
            / self.zoom,

            (
                point.y()
                - center_y
                - self.pan.y()
            )
            / self.zoom,
        )

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event):

        painter = QPainter(
            self
        )

        painter.setRenderHint(
            QPainter.RenderHint.SmoothPixmapTransform,
            True,
        )

        painter.fillRect(
            self.rect(),
            Qt.GlobalColor.black,
        )

        geometry = self._spread_geometry()

        if geometry is None:
            return

        (
            left_image,
            right_image,
            total_width,
            max_height,
            left_x,
            right_x,
        ) = geometry

        # ---------------------------------------------------------------
        # Calculate vertical centering for the spread.
        # ---------------------------------------------------------------

        for image, source_x in (
            (
                left_image,
                left_x,
            ),
            (
                right_image,
                right_x,
            ),
        ):

            if image is None:
                continue

            source_y = (
                -image.height()
                / 2.0
            )

            target = QRectF(
                *(
                    self._book_to_screen(
                        source_x,
                        source_y,
                    ).toTuple()
                ),
                image.width()
                * self.zoom,
                image.height()
                * self.zoom,
            )

            # -----------------------------------------------------------
            # Draw directly from the original QImage.
            #
            # No intermediate low-resolution pixmap is cached.
            #
            # At zoom=1 this is essentially a 1:1 display.
            # -----------------------------------------------------------

            painter.drawImage(
                target,
                image,
                QRectF(
                    0,
                    0,
                    image.width(),
                    image.height(),
                ),
            )

        # Center gutter.

        if (
            left_image is not None
            and right_image is not None
        ):

            gutter_x = (
                self.width() / 2.0
                + self.pan.x()
            )

            painter.fillRect(
                QRectF(
                    gutter_x - 1,
                    0,
                    2,
                    self.height(),
                ),
                Qt.GlobalColor.black,
            )

    # ------------------------------------------------------------------
    # Mouse wheel zoom
    # ------------------------------------------------------------------

    def wheelEvent(self, event):

        modifiers = (
            event.modifiers()
        )

        # Wheel always zooms.
        #
        # This is convenient for a dedicated image viewer and means
        # normal scrolling doesn't accidentally move the book.

        old_zoom = self.zoom

        if event.angleDelta().y() > 0:
            new_zoom = (
                old_zoom
                * ZOOM_FACTOR
            )
        else:
            new_zoom = (
                old_zoom
                / ZOOM_FACTOR
            )

        new_zoom = max(
            MIN_ZOOM,
            min(
                MAX_ZOOM,
                new_zoom,
            ),
        )

        if (
            abs(new_zoom - old_zoom)
            < 0.0001
        ):
            return

        # ---------------------------------------------------------------
        # Zoom around mouse cursor.
        #
        # We keep the book coordinate under the mouse stationary.
        # ---------------------------------------------------------------

        mouse = event.position()

        before = self._screen_to_book(
            mouse
        )

        self.zoom = new_zoom

        after = self._screen_to_book(
            mouse
        )

        self.pan += (
            after - before
        ) * self.zoom

        self.update()

        event.accept()

    # ------------------------------------------------------------------
    # Mouse pan
    # ------------------------------------------------------------------

    def mousePressEvent(self, event):

        if (
            event.button()
            == Qt.MouseButton.LeftButton
        ):

            self.dragging = True

            self.last_mouse = (
                event.position()
            )

            self.setCursor(
                Qt.CursorShape.ClosedHandCursor
            )

            event.accept()

            return

        super().mousePressEvent(
            event
        )

    def mouseMoveEvent(self, event):

        if self.dragging:

            current = (
                event.position()
            )

            delta = (
                current
                - self.last_mouse
            )

            self.pan += delta

            self.last_mouse = current

            self.update()

            event.accept()

            return

        super().mouseMoveEvent(
            event
        )

    def mouseReleaseEvent(self, event):

        if (
            event.button()
            == Qt.MouseButton.LeftButton
        ):

            self.dragging = False

            self.setCursor(
                Qt.CursorShape.ArrowCursor
            )

            event.accept()

            return

        super().mouseReleaseEvent(
            event
        )

    # ------------------------------------------------------------------
    # Keyboard zoom
    # ------------------------------------------------------------------

    def zoom_in(self):

        self._zoom_centered(
            ZOOM_FACTOR
        )

    def zoom_out(self):

        self._zoom_centered(
            1.0 / ZOOM_FACTOR
        )

    def reset_zoom(self):

        self.zoom = 1.0

        self.pan = QPointF(
            0,
            0,
        )

        self.update()

    def _zoom_centered(
        self,
        factor: float,
    ):

        old_zoom = self.zoom

        new_zoom = max(
            MIN_ZOOM,
            min(
                MAX_ZOOM,
                old_zoom * factor,
            ),
        )

        if new_zoom == old_zoom:
            return

        center = QPointF(
            self.width() / 2.0,
            self.height() / 2.0,
        )

        before = self._screen_to_book(
            center
        )

        self.zoom = new_zoom

        after = self._screen_to_book(
            center
        )

        self.pan += (
            after - before
        ) * self.zoom

        self.update()

# ============================================================================
# Settings
# ============================================================================

class SettingsDialog(QDialog):

    def __init__(
        self,
        time_per_page,
        preload_spreads,
        cache_size,
        parent=None,
    ):

        super().__init__(
            parent
        )

        self.setWindowTitle(
            "Viewer Settings"
        )

        layout = QFormLayout(
            self
        )

        self.time_spin = (
            QDoubleSpinBox()
        )

        self.time_spin.setRange(
            0.05,
            60.0,
        )

        self.time_spin.setSingleStep(
            0.05
        )

        self.time_spin.setDecimals(
            2
        )

        self.time_spin.setValue(
            time_per_page
        )

        self.time_spin.setSuffix(
            " s"
        )

        self.preload_spin = (
            QSpinBox()
        )

        self.preload_spin.setRange(
            0,
            200,
        )

        self.preload_spin.setValue(
            preload_spreads
        )

        self.preload_spin.setSuffix(
            " spreads"
        )

        self.cache_spin = (
            QSpinBox()
        )

        self.cache_spin.setRange(
            4,
            1000,
        )

        self.cache_spin.setValue(
            cache_size
        )

        self.cache_spin.setSuffix(
            " images"
        )

        layout.addRow(
            "Time per page:",
            self.time_spin
        )

        layout.addRow(
            "Preload next:",
            self.preload_spin
        )

        layout.addRow(
            "RAM image cache:",
            self.cache_spin
        )

        buttons = QHBoxLayout()

        cancel = QPushButton(
            "Cancel"
        )

        ok = QPushButton(
            "OK"
        )

        cancel.clicked.connect(
            self.reject
        )

        ok.clicked.connect(
            self.accept
        )

        buttons.addStretch()

        buttons.addWidget(
            cancel
        )

        buttons.addWidget(
            ok
        )

        layout.addRow(
            buttons
        )

# ============================================================================
# Main window
# ============================================================================

class BookViewer(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "Scanned Book Viewer"
        )

        self.resize(
            1400,
            900
        )

        # ---------------------------------------------------------------
        # Book
        # ---------------------------------------------------------------

        self.image_paths: list[
            Path
        ] = []

        self.spreads: list[
            Spread
        ] = []

        self.current_spread = 0

        # ---------------------------------------------------------------
        # Settings
        # ---------------------------------------------------------------

        self.time_per_page = (
            DEFAULT_TIME_PER_PAGE
        )

        self.preload_spreads = (
            DEFAULT_PRELOAD_SPREADS
        )

        self.cache_size = (
            DEFAULT_CACHE_SIZE
        )

        # ---------------------------------------------------------------
        # Cache
        # ---------------------------------------------------------------

        self.cache = ImageCache(
            self.cache_size
        )

        # ---------------------------------------------------------------
        # Worker pool
        # ---------------------------------------------------------------

        self.thread_pool = (
            QThreadPool.globalInstance()
        )

        # Don't use an excessive number of disk readers.
        #
        # Four workers gives us concurrent decoding without turning
        # an HDD into a seek-fest.
        self.thread_pool.setMaxThreadCount(
            min(
                4,
                max(
                    2,
                    self.thread_pool.maxThreadCount(),
                ),
            )
        )

        self.loading: set[int] = set()

        # ---------------------------------------------------------------
        # Autoplay
        # ---------------------------------------------------------------

        self.autoplay = False

        self.autoplay_timer = QTimer(
            self
        )

        self.autoplay_timer.timeout.connect(
            self.advance
        )

        # ---------------------------------------------------------------
        # Canvas
        # ---------------------------------------------------------------

        self.canvas = BookCanvas(
            self
        )

        self.setCentralWidget(
            self.canvas
        )

        # ---------------------------------------------------------------
        # Toolbar
        # ---------------------------------------------------------------

        self._create_actions()

        self._create_toolbar()

        # ---------------------------------------------------------------
        # Status
        # ---------------------------------------------------------------

        self.status = QStatusBar(
            self
        )

        self.setStatusBar(
            self.status
        )

        self.status.showMessage(
            "Open a folder to begin"
        )

        # ---------------------------------------------------------------
        # Fullscreen chrome timer
        # ---------------------------------------------------------------

        self.chrome_timer = QTimer(
            self
        )

        self.chrome_timer.setSingleShot(
            True
        )

        self.chrome_timer.timeout.connect(
            self.hide_fullscreen_chrome
        )

        self.ui_revealed = False

        self._update_navigation()

    # ==================================================================
    # Actions
    # ==================================================================

    def _create_actions(self):

        self.open_folder_action = QAction(
            "Open Folder",
            self
        )

        self.open_folder_action.setShortcut(
            QKeySequence("Ctrl+O")
        )

        self.open_folder_action.triggered.connect(
            self.open_folder
        )

        # ---------------------------------------------------------------

        self.open_files_action = QAction(
            "Open Images",
            self
        )

        self.open_files_action.setShortcut(
            QKeySequence(
                "Ctrl+Shift+O"
            )
        )

        self.open_files_action.triggered.connect(
            self.open_files
        )

        # ---------------------------------------------------------------

        self.previous_action = QAction(
            "Previous",
            self
        )

        self.previous_action.setShortcut(
            QKeySequence("Left")
        )

        self.previous_action.triggered.connect(
            self.previous
        )

        # ---------------------------------------------------------------

        self.play_action = QAction(
            "Play",
            self
        )

        self.play_action.setShortcut(
            QKeySequence("Space")
        )

        self.play_action.triggered.connect(
            self.toggle_autoplay
        )

        # ---------------------------------------------------------------

        self.next_action = QAction(
            "Next",
            self
        )

        self.next_action.setShortcut(
            QKeySequence("Right")
        )

        self.next_action.triggered.connect(
            self.next
        )

        # ---------------------------------------------------------------

        self.settings_action = QAction(
            "Settings",
            self
        )

        self.settings_action.triggered.connect(
            self.show_settings
        )

        # ---------------------------------------------------------------

        self.fullscreen_action = QAction(
            "Fullscreen",
            self
        )

        self.fullscreen_action.setShortcut(
            QKeySequence("F11")
        )

        self.fullscreen_action.triggered.connect(
            self.toggle_fullscreen
        )

        # ---------------------------------------------------------------

        self.quit_action = QAction(
            "Quit",
            self
        )

        self.quit_action.setShortcut(
            QKeySequence("Ctrl+Q")
        )

        self.quit_action.triggered.connect(
            self.close
        )

    def _create_toolbar(self):

        self.toolbar = QToolBar(
            "Controls"
        )

        self.toolbar.setMovable(
            False
        )

        self.toolbar.addAction(
            self.open_folder_action
        )

        self.toolbar.addAction(
            self.open_files_action
        )

        self.toolbar.addSeparator()

        self.toolbar.addAction(
            self.previous_action
        )

        self.toolbar.addAction(
            self.play_action
        )

        self.toolbar.addAction(
            self.next_action
        )

        self.toolbar.addSeparator()

        self.toolbar.addAction(
            self.settings_action
        )

        self.toolbar.addAction(
            self.fullscreen_action
        )

        self.addToolBar(
            self.toolbar
        )

    # ==================================================================
    # Image opening
    # ==================================================================

    def open_folder(self):

        folder = (
            QFileDialog
            .getExistingDirectory(
                self,
                "Select scanned book folder",
            )
        )

        if not folder:
            return

        self.load_folder(
            Path(folder)
        )

    def open_files(self):

        files, _ = (
            QFileDialog
            .getOpenFileNames(
                self,
                "Select scanned pages",
                "",
                (
                    "Images "
                    "(*.jpg *.jpeg *.png *.bmp "
                    "*.gif *.tif *.tiff *.webp)"
                ),
            )
        )

        if not files:
            return

        paths = [
            Path(x)
            for x in files
        ]

        paths.sort(
            key=natural_sort_key
        )

        self.load_images(
            paths
        )

    def load_folder(
        self,
        folder: Path,
    ):

        try:

            paths = [
                path
                for path in folder.iterdir()
                if (
                    path.is_file()
                    and path.suffix.lower()
                    in IMAGE_EXTENSIONS
                )
            ]

        except OSError as exc:

            QMessageBox.critical(
                self,
                "Error",
                f"Could not read folder:\n{exc}",
            )

            return

        paths.sort(
            key=natural_sort_key
        )

        self.load_images(
            paths
        )

    def load_images(
        self,
        paths: list[Path],
    ):

        self.stop_autoplay()

        self.image_paths = paths

        self.spreads = make_spreads(
            len(paths)
        )

        self.current_spread = 0

        self.cache.clear()

        self.loading.clear()

        # Reset zoom only when opening a completely new book.
        self.canvas.reset_zoom()

        if not paths:

            self.status.showMessage(
                "No images found"
            )

            self.canvas.update()

            self._update_navigation()

            return

        self.setWindowTitle(
            "Scanned Book Viewer — "
            + paths[0].parent.name
        )

        self._display_current()

        # Start filling the buffer immediately.
        self._preload_images()

    # ==================================================================
    # Image access
    # ==================================================================

    def get_cached_image(
        self,
        index: int | None,
    ) -> QImage | None:

        if index is None:
            return None

        return self.cache.get(
            index
        )

    def current_spread_object(
        self,
    ) -> Spread | None:

        if not self.spreads:
            return None

        return self.spreads[
            self.current_spread
        ]

    # ==================================================================
    # Background loading
    # ==================================================================

    def _request_image(
        self,
        index: int,
    ):

        if (
            index < 0
            or index >= len(
                self.image_paths
            )
        ):
            return

        if self.cache.contains(
            index
        ):
            return

        if index in self.loading:
            return

        self.loading.add(
            index
        )

        signals = ImageLoadSignals()

        task = ImageLoadTask(
            index=index,
            path=self.image_paths[index],
            cache=self.cache,
            signals=signals,
        )

        signals.finished.connect(
            self._image_loaded
        )

        signals.failed.connect(
            self._image_failed
        )

        self.thread_pool.start(
            task
        )

    @Slot(int, object)
    def _image_loaded(
        self,
        index: int,
        image: QImage,
    ):

        self.loading.discard(
            index
        )

        self.canvas.update()

        # Continue aggressively filling the buffer.
        self._preload_images()

    @Slot(int, str)
    def _image_failed(
        self,
        index: int,
        message: str,
    ):

        self.loading.discard(
            index
        )

        print(
            message,
            file=sys.stderr,
        )

    # ==================================================================
    # Aggressive preloading
    # ==================================================================

    def _preload_images(self):

        if not self.spreads:
            return

        # Current spread + N future spreads.
        first = self.current_spread

        last = min(
            len(self.spreads),
            first
            + self.preload_spreads
            + 1,
        )

        for spread_index in range(
            first,
            last,
        ):

            spread = self.spreads[
                spread_index
            ]

            if spread.left is not None:
                self._request_image(
                    spread.left
                )

            if spread.right is not None:
                self._request_image(
                    spread.right
                )

    # ==================================================================
    # Display
    # ==================================================================

    def _display_current(self):

        if not self.spreads:
            self.canvas.update()
            return

        spread = self.spreads[
            self.current_spread
        ]

        # Make sure visible images are prioritized.
        if spread.left is not None:
            self._request_image(
                spread.left
            )

        if spread.right is not None:
            self._request_image(
                spread.right
            )

        self.canvas.update()

        self._preload_images()

        self._update_navigation()

    # ==================================================================
    # Playback buffering
    # ==================================================================

    def _next_spread_ready(self):

        next_spread_index = (
            self.current_spread + 1
        )

        if (
            next_spread_index
            >= len(self.spreads)
        ):
            return False

        spread = self.spreads[
            next_spread_index
        ]

        if spread.left is not None:

            if not self.cache.contains(
                spread.left
            ):
                return False

        if spread.right is not None:

            if not self.cache.contains(
                spread.right
            ):
                return False

        return True

    def toggle_autoplay(self):

        if self.autoplay:
            self.stop_autoplay()

        else:
            self.start_autoplay()

    def start_autoplay(self):

        if not self.spreads:
            return

        self.autoplay = True

        # Fill the future buffer before starting.
        self._preload_images()

        interval = max(
            1,
            round(
                self.time_per_page
                * 1000
            ),
        )

        self.autoplay_timer.start(
            interval
        )

        self.play_action.setText(
            "Pause"
        )

        self.status.showMessage(
            "Playing"
        )

    def stop_autoplay(self):

        self.autoplay = False

        self.autoplay_timer.stop()

        self.play_action.setText(
            "Play"
        )

        self._update_navigation()

    def advance(self):

        if not self.spreads:

            self.stop_autoplay()

            return

        if (
            self.current_spread
            >= len(self.spreads) - 1
        ):

            self.stop_autoplay()

            return

        # Don't enter a spread until its images are decoded.
        #
        # This turns the preload buffer into an actual playback buffer.
        if not self._next_spread_ready():

            self.status.showMessage(
                "Buffering..."
            )

            self._preload_images()

            return

        self.current_spread += 1

        self._display_current()

    # ==================================================================
    # Navigation
    # ==================================================================

    def previous(self):

        if not self.spreads:
            return

        if self.current_spread <= 0:
            return

        self.current_spread -= 1

        self._display_current()

    def next(self):

        if not self.spreads:
            return

        if (
            self.current_spread
            >= len(self.spreads) - 1
        ):

            self.stop_autoplay()

            return

        self.current_spread += 1

        self._display_current()

    # ==================================================================
    # Settings
    # ==================================================================

    def show_settings(self):

        was_playing = self.autoplay

        if was_playing:
            self.stop_autoplay()

        dialog = SettingsDialog(
            self.time_per_page,
            self.preload_spreads,
            self.cache_size,
            self,
        )

        if (
            dialog.exec()
            == QDialog.DialogCode.Accepted
        ):

            self.time_per_page = (
                dialog.time_spin.value()
            )

            self.preload_spreads = (
                dialog.preload_spin.value()
            )

            self.cache_size = (
                dialog.cache_spin.value()
            )

            self.cache.resize(
                self.cache_size
            )

            self._preload_images()

            if was_playing:
                self.start_autoplay()

        elif was_playing:

            self.start_autoplay()

    # ==================================================================
    # Fullscreen chrome
    # ==================================================================

    def toggle_fullscreen(self):

        if self.isFullScreen():

            self.showNormal()

            self.show_fullscreen_chrome(
                permanent=True
            )

        else:

            self.showFullScreen()

            # Headless by default.
            self.hide_fullscreen_chrome()

    def show_fullscreen_chrome(
        self,
        permanent=False,
    ):

        if not self.isFullScreen():
            return

        self.toolbar.show()
        self.status.show()

        self.ui_revealed = True

        if not permanent:

            self.chrome_timer.start(
                CHROME_HIDE_DELAY_MS
            )

    def hide_fullscreen_chrome(self):

        if not self.isFullScreen():
            return

        self.toolbar.hide()
        self.status.hide()

        self.ui_revealed = False

    def _check_fullscreen_edges(
        self,
        position,
    ):

        if not self.isFullScreen():
            return

        y = position.y()

        if (
            y <= EDGE_REVEAL_PIXELS
            or y
            >= self.height()
            - EDGE_REVEAL_PIXELS
        ):

            self.show_fullscreen_chrome()

        elif self.ui_revealed:

            self.chrome_timer.start(
                CHROME_HIDE_DELAY_MS
            )

    # ==================================================================
    # Mouse movement
    # ==================================================================

    def eventFilter(
        self,
        watched,
        event,
    ):

        if (
            event.type()
            == event.Type.MouseMove
        ):

            position = event.position()

            self._check_fullscreen_edges(
                position
            )

        return super().eventFilter(
            watched,
            event
        )

    def mouseMoveEvent(
        self,
        event,
    ):

        self._check_fullscreen_edges(
            event.position()
        )

        super().mouseMoveEvent(
            event
        )

    # ==================================================================
    # Keyboard
    # ==================================================================

    def keyPressEvent(
        self,
        event,
    ):

        key = event.key()

        modifiers = (
            event.modifiers()
        )

        # ---------------------------------------------------------------
        # Space
        # ---------------------------------------------------------------

        if key == Qt.Key.Key_Space:

            self.toggle_autoplay()

            event.accept()

            return

        # ---------------------------------------------------------------
        # Navigation
        # ---------------------------------------------------------------

        if key == Qt.Key.Key_Left:

            self.previous()

            event.accept()

            return

        if key == Qt.Key.Key_Right:

            self.next()

            event.accept()

            return

        # ---------------------------------------------------------------
        # Zoom
        # ---------------------------------------------------------------

        if (
            modifiers
            & Qt.KeyboardModifier.ControlModifier
        ):

            if key in (
                Qt.Key.Key_Plus,
                Qt.Key.Key_Equal,
            ):

                self.canvas.zoom_in()

                event.accept()

                return

            if key in (
                Qt.Key.Key_Minus,
                Qt.Key.Key_Underscore,
            ):

                self.canvas.zoom_out()

                event.accept()

                return

            if key == Qt.Key.Key_0:

                self.canvas.reset_zoom()

                event.accept()

                return

        # ---------------------------------------------------------------
        # Fullscreen
        # ---------------------------------------------------------------

        if key == Qt.Key.Key_F11:

            self.toggle_fullscreen()

            event.accept()

            return

        if (
            key == Qt.Key.Key_Escape
            and self.isFullScreen()
        ):

            self.showNormal()

            self.show_fullscreen_chrome(
                permanent=True
            )

            event.accept()

            return

        super().keyPressEvent(
            event
        )

    # ==================================================================
    # Status
    # ==================================================================

    def _status_text(self):

        if not self.image_paths:
            return "No images"

        spread = self.spreads[
            self.current_spread
        ]

        if spread.left is None:

            pages = str(
                spread.right + 1
            )

        elif spread.right is None:

            pages = str(
                spread.left + 1
            )

        else:

            pages = (
                f"{spread.left + 1}"
                "–"
                f"{spread.right + 1}"
            )

        return (
            f"Pages {pages} / "
            f"{len(self.image_paths)}"
            "    "
            f"Spread "
            f"{self.current_spread + 1} / "
            f"{len(self.spreads)}"
            "    "
            f"Zoom {self.canvas.zoom:.2f}×"
        )

    def _update_navigation(self):

        has_pages = bool(
            self.spreads
        )

        self.previous_action.setEnabled(
            has_pages
            and self.current_spread > 0
        )

        self.next_action.setEnabled(
            has_pages
            and self.current_spread
            < len(self.spreads) - 1
        )

        if (
            has_pages
            and not self.autoplay
        ):

            self.status.showMessage(
                self._status_text()
            )

    # ==================================================================
    # Fullscreen / mouse handling
    # ==================================================================

    def enterEvent(self, event):

        if self.isFullScreen():

            self._check_fullscreen_edges(
                self.mapFromGlobal(
                    self.cursor().pos()
                )
            )

        super().enterEvent(
            event
        )

    def resizeEvent(self, event):

        super().resizeEvent(
            event
        )

        self.canvas.update()

    # ==================================================================
    # Close
    # ==================================================================

    def closeEvent(self, event):

        self.stop_autoplay()

        self.thread_pool.clear()

        event.accept()

# ============================================================================
# Application
# ============================================================================

def main():

    app = QApplication(
        sys.argv
    )

    app.setApplicationName(
        "Scanned Book Viewer"
    )

    app.setOrganizationName(
        "ScannedBookViewer"
    )

    viewer = BookViewer()

    viewer.show()

    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
```

There is one thing I'd change further if these scans are **really large 600-DPI images**: use a **tiled/level-of-detail renderer** rather than drawing the entire original image every time. At 600 DPI, a single page can easily be 5,000–10,000 pixels wide, and keeping dozens of full-resolution pages in RAM can become enormous. A tiled pyramid would give you something much closer to a professional PDF/scanned-book reader: sharp 1:1 text when zoomed, instant zooming, and only the visible portions of pages loaded into memory.

Also, if your scans are **TIFFs**, especially multi-page TIFFs, I would modify the loader to handle those specifically rather than treating each file as one page.

## Prompt:

9/9/2026, 6:01:01 PM

also, it should have an required positional CLI argument: the source directory (source of images)

also, it should load a project config like


    from _shared import (
        load_config,
        get_page_num,
    )

    config = load_config()

and then it should use `config.scan_format` (usually `"tiff"`) to find image files in the source directory (non-recursive)

## Response:

9/9/2026, 6:01:06 PM

Yep. I’d make the CLI interface the primary entry point and remove the folder/file-open workflow entirely.

The intended usage becomes:

``` bash
python book_viewer.py /path/to/scan
```

and the script obtains the project configuration exactly as requested:

``` python
from _shared import (
    load_config,
    get_page_num,
)

config = load_config()
```

Then `config.scan_format` determines which files are considered pages, and `get_page_num()` is used for ordering rather than relying on filenames alone.

One important detail: since I don't know the exact contract of your project's `get_page_num()`, I've made the call `get_page_num(path)` and kept it isolated in one function. If your existing helper takes a filename/string rather than a `Path`, that's the only line that should need changing.

\*\*document: \*\*

``` python
#!/usr/bin/env python3

"""
Scanned Book Viewer
===================

Usage:

    python book_viewer.py SOURCE_DIRECTORY

Example:

    python book_viewer.py ./scans

The project must provide:

    _shared.py

with:

    from _shared import (
        load_config,
        get_page_num,
    )

    config = load_config()

The configured scan format is taken from:

    config.scan_format

For example:

    config.scan_format == "tiff"

The source directory is searched NON-RECURSIVELY.

Page ordering is determined by get_page_num().
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import (
    QObject,
    QPointF,
    QRectF,
    QRunnable,
    QThreadPool,
    QTimer,
    Qt,
    QMutex,
    Signal,
    Slot,
)
from PySide6.QtGui import (
    QAction,
    QImage,
    QKeySequence,
    QPainter,
)
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStatusBar,
    QToolBar,
    QWidget,
)

# ---------------------------------------------------------------------------
# Project configuration
# ---------------------------------------------------------------------------

from _shared import (
    load_config,
    get_page_num,
)

config = load_config()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_TIME_PER_PAGE = 0.20

# Number of future spreads to preload.
#
# 20 spreads can mean up to 40 future images.
DEFAULT_PRELOAD_SPREADS = 20

# Number of full-resolution decoded images retained in RAM.
DEFAULT_CACHE_SIZE = 60

# Fullscreen chrome behaviour.
EDGE_REVEAL_PIXELS = 35
CHROME_HIDE_DELAY_MS = 1800

# Zoom.
MIN_ZOOM = 0.10
MAX_ZOOM = 8.0
ZOOM_FACTOR = 1.15

# ---------------------------------------------------------------------------
# Image extension
# ---------------------------------------------------------------------------

def configured_extension() -> str:
    """
    Return the configured scan format as a normalized extension.

    Examples:

        "tiff"  -> ".tiff"
        ".tiff" -> ".tiff"
        "TIFF"  -> ".tiff"
    """

    fmt = str(
        config.scan_format
    ).strip().lower()

    if fmt.startswith("."):
        return fmt

    return "." + fmt

# ---------------------------------------------------------------------------
# Find source images
# ---------------------------------------------------------------------------

def find_source_images(
    source_directory: Path,
) -> list[Path]:
    """
    Find scan images in source_directory.

    IMPORTANT:
        This is deliberately NON-RECURSIVE.

    Only files directly inside source_directory are considered.

    Ordering is based on the project's get_page_num() helper.
    """

    extension = configured_extension()

    paths = [
        path
        for path in source_directory.iterdir()
        if (
            path.is_file()
            and path.suffix.lower() == extension
        )
    ]

    def page_number(path: Path):
        """
        Adapt the Path to the project's get_page_num() helper.

        If your project expects a string instead of a Path, change this
        to:

            return get_page_num(str(path))

        """

        return get_page_num(path)

    paths.sort(
        key=page_number
    )

    return paths

# ---------------------------------------------------------------------------
# Spread model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Spread:
    """
    Zero-based image indexes.

    Page 1 is the right-hand page.

        Spread(None, 0)
        Spread(1, 2)
        Spread(3, 4)
        ...
    """

    left: int | None
    right: int | None

def make_spreads(
    page_count: int,
) -> list[Spread]:

    if page_count <= 0:
        return []

    spreads = []

    # First page: right-hand page.
    spreads.append(
        Spread(
            left=None,
            right=0,
        )
    )

    page = 1

    while page < page_count:

        left = page

        if page + 1 < page_count:

            spreads.append(
                Spread(
                    left=left,
                    right=page + 1,
                )
            )

            page += 2

        else:

            # Final page by itself on the left.
            spreads.append(
                Spread(
                    left=left,
                    right=None,
                )
            )

            page += 1

    return spreads

# ---------------------------------------------------------------------------
# Image cache
# ---------------------------------------------------------------------------

class ImageCache:

    def __init__(
        self,
        max_items: int,
    ):

        self.max_items = max(
            1,
            max_items,
        )

        self._cache: OrderedDict[
            int,
            QImage,
        ] = OrderedDict()

        self._mutex = QMutex()

    def get(
        self,
        index: int,
    ) -> QImage | None:

        self._mutex.lock()

        try:

            image = self._cache.get(
                index
            )

            if image is not None:
                self._cache.move_to_end(
                    index
                )

            return image

        finally:

            self._mutex.unlock()

    def contains(
        self,
        index: int,
    ) -> bool:

        self._mutex.lock()

        try:
            return index in self._cache

        finally:
            self._mutex.unlock()

    def put(
        self,
        index: int,
        image: QImage,
    ):

        self._mutex.lock()

        try:

            self._cache[index] = image

            self._cache.move_to_end(
                index
            )

            while (
                len(self._cache)
                > self.max_items
            ):
                self._cache.popitem(
                    last=False
                )

        finally:

            self._mutex.unlock()

    def clear(self):

        self._mutex.lock()

        try:
            self._cache.clear()

        finally:

            self._mutex.unlock()

    def resize(
        self,
        size: int,
    ):

        self._mutex.lock()

        try:

            self.max_items = max(
                1,
                size,
            )

            while (
                len(self._cache)
                > self.max_items
            ):
                self._cache.popitem(
                    last=False
                )

        finally:

            self._mutex.unlock()

# ---------------------------------------------------------------------------
# Background loading
# ---------------------------------------------------------------------------

class ImageLoadSignals(QObject):

    finished = Signal(
        int,
        object,
    )

    failed = Signal(
        int,
        str,
    )

class ImageLoadTask(QRunnable):

    def __init__(
        self,
        index: int,
        path: Path,
        cache: ImageCache,
        signals: ImageLoadSignals,
    ):

        super().__init__()

        self.index = index
        self.path = path
        self.cache = cache
        self.signals = signals

        self.setAutoDelete(True)

    @Slot()
    def run(self):

        cached = self.cache.get(
            self.index
        )

        if cached is not None:

            self.signals.finished.emit(
                self.index,
                cached,
            )

            return

        image = QImage(
            str(self.path)
        )

        if image.isNull():

            self.signals.failed.emit(
                self.index,
                f"Could not load:\n{self.path}",
            )

            return

        # Force the complete image data to be decoded while we're still
        # in the worker thread.
        #
        # This is important for JPEG/TIFF scans because otherwise some
        # work can be deferred until the image is actually painted.
        image = image.copy()

        self.cache.put(
            self.index,
            image,
        )

        self.signals.finished.emit(
            self.index,
            image,
        )

# ---------------------------------------------------------------------------
# Book canvas
# ---------------------------------------------------------------------------

class BookCanvas(QWidget):
    """
    The two pages are treated as ONE continuous image.

    The gutter is exactly x=0 in book coordinates.

    This allows zooming and panning across both pages simultaneously.
    """

    def __init__(
        self,
        viewer,
    ):

        super().__init__(
            viewer
        )

        self.viewer = viewer

        self.zoom = 1.0

        self.pan = QPointF(
            0.0,
            0.0,
        )

        self.dragging = False

        self.last_mouse = QPointF()

        self.setMouseTracking(
            True
        )

        self.setFocusPolicy(
            Qt.FocusPolicy.StrongFocus
        )

    # ------------------------------------------------------------------
    # Coordinate conversion
    # ------------------------------------------------------------------

    def _book_to_screen(
        self,
        x: float,
        y: float,
    ) -> QPointF:

        return QPointF(
            self.width() / 2.0
            + self.pan.x()
            + x * self.zoom,

            self.height() / 2.0
            + self.pan.y()
            + y * self.zoom,
        )

    def _screen_to_book(
        self,
        point: QPointF,
    ) -> QPointF:

        return QPointF(
            (
                point.x()
                - self.width() / 2.0
                - self.pan.x()
            )
            / self.zoom,

            (
                point.y()
                - self.height() / 2.0
                - self.pan.y()
            )
            / self.zoom,
        )

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(
        self,
        event,
    ):

        painter = QPainter(
            self
        )

        painter.fillRect(
            self.rect(),
            Qt.GlobalColor.black,
        )

        # High-quality scaling.
        painter.setRenderHint(
            QPainter.RenderHint.SmoothPixmapTransform,
            True,
        )

        spread = (
            self.viewer
            .current_spread_object()
        )

        if spread is None:
            return

        left = (
            self.viewer
            .get_cached_image(
                spread.left
            )
            if spread.left is not None
            else None
        )

        right = (
            self.viewer
            .get_cached_image(
                spread.right
            )
            if spread.right is not None
            else None
        )

        # ---------------------------------------------------------------
        # Draw LEFT page.
        #
        # Its RIGHT edge is exactly x=0.
        # ---------------------------------------------------------------

        if left is not None:

            x = -left.width()

            y = -left.height() / 2.0

            top_left = (
                self._book_to_screen(
                    x,
                    y,
                )
            )

            target = QRectF(
                top_left.x(),
                top_left.y(),
                left.width()
                * self.zoom,
                left.height()
                * self.zoom,
            )

            painter.drawImage(
                target,
                left,
                QRectF(
                    0,
                    0,
                    left.width(),
                    left.height(),
                ),
            )

        # ---------------------------------------------------------------
        # Draw RIGHT page.
        #
        # Its LEFT edge is exactly x=0.
        # ---------------------------------------------------------------

        if right is not None:

            x = 0

            y = -right.height() / 2.0

            top_left = (
                self._book_to_screen(
                    x,
                    y,
                )
            )

            target = QRectF(
                top_left.x(),
                top_left.y(),
                right.width()
                * self.zoom,
                right.height()
                * self.zoom,
            )

            painter.drawImage(
                target,
                right,
                QRectF(
                    0,
                    0,
                    right.width(),
                    right.height(),
                ),
            )

    # ------------------------------------------------------------------
    # Zoom
    # ------------------------------------------------------------------

    def wheelEvent(
        self,
        event,
    ):

        old_zoom = self.zoom

        if event.angleDelta().y() > 0:

            new_zoom = (
                old_zoom
                * ZOOM_FACTOR
            )

        else:

            new_zoom = (
                old_zoom
                / ZOOM_FACTOR
            )

        new_zoom = max(
            MIN_ZOOM,
            min(
                MAX_ZOOM,
                new_zoom,
            ),
        )

        if new_zoom == old_zoom:
            return

        mouse = event.position()

        # Keep the book coordinate under the mouse stationary.
        before = self._screen_to_book(
            mouse
        )

        self.zoom = new_zoom

        after = self._screen_to_book(
            mouse
        )

        self.pan += (
            after - before
        ) * self.zoom

        self.update()

        event.accept()

    def zoom_in(self):

        self._zoom_centered(
            ZOOM_FACTOR
        )

    def zoom_out(self):

        self._zoom_centered(
            1.0 / ZOOM_FACTOR
        )

    def reset_zoom(self):

        self.zoom = 1.0

        self.pan = QPointF(
            0,
            0,
        )

        self.update()

    def _zoom_centered(
        self,
        factor: float,
    ):

        old_zoom = self.zoom

        new_zoom = max(
            MIN_ZOOM,
            min(
                MAX_ZOOM,
                old_zoom * factor,
            ),
        )

        if new_zoom == old_zoom:
            return

        center = QPointF(
            self.width() / 2.0,
            self.height() / 2.0,
        )

        before = self._screen_to_book(
            center
        )

        self.zoom = new_zoom

        after = self._screen_to_book(
            center
        )

        self.pan += (
            after - before
        ) * self.zoom

        self.update()

    # ------------------------------------------------------------------
    # Pan
    # ------------------------------------------------------------------

    def mousePressEvent(
        self,
        event,
    ):

        if (
            event.button()
            == Qt.MouseButton.LeftButton
        ):

            self.dragging = True

            self.last_mouse = (
                event.position()
            )

            self.setCursor(
                Qt.CursorShape.ClosedHandCursor
            )

            event.accept()

            return

        super().mousePressEvent(
            event
        )

    def mouseMoveEvent(
        self,
        event,
    ):

        # Tell the main window about fullscreen edge movement.
        self.viewer.fullscreen_mouse_move(
            event.position()
        )

        if self.dragging:

            current = (
                event.position()
            )

            self.pan += (
                current
                - self.last_mouse
            )

            self.last_mouse = current

            self.update()

            event.accept()

            return

        super().mouseMoveEvent(
            event
        )

    def mouseReleaseEvent(
        self,
        event,
    ):

        if (
            event.button()
            == Qt.MouseButton.LeftButton
        ):

            self.dragging = False

            self.setCursor(
                Qt.CursorShape.ArrowCursor
            )

            event.accept()

            return

        super().mouseReleaseEvent(
            event
        )

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class SettingsDialog(QDialog):

    def __init__(
        self,
        time_per_page,
        preload_spreads,
        cache_size,
        parent=None,
    ):

        super().__init__(
            parent
        )

        self.setWindowTitle(
            "Viewer Settings"
        )

        layout = QFormLayout(
            self
        )

        self.time_spin = (
            QDoubleSpinBox()
        )

        self.time_spin.setRange(
            0.05,
            60.0,
        )

        self.time_spin.setSingleStep(
            0.05
        )

        self.time_spin.setDecimals(
            2
        )

        self.time_spin.setValue(
            time_per_page
        )

        self.time_spin.setSuffix(
            " s"
        )

        self.preload_spin = (
            QSpinBox()
        )

        self.preload_spin.setRange(
            0,
            200
        )

        self.preload_spin.setValue(
            preload_spreads
        )

        self.preload_spin.setSuffix(
            " spreads"
        )

        self.cache_spin = (
            QSpinBox()
        )

        self.cache_spin.setRange(
            4,
            1000
        )

        self.cache_spin.setValue(
            cache_size
        )

        self.cache_spin.setSuffix(
            " images"
        )

        layout.addRow(
            "Time per page:",
            self.time_spin
        )

        layout.addRow(
            "Preload next:",
            self.preload_spin
        )

        layout.addRow(
            "RAM image cache:",
            self.cache_spin
        )

        buttons = QHBoxLayout()

        cancel = QPushButton(
            "Cancel"
        )

        ok = QPushButton(
            "OK"
        )

        cancel.clicked.connect(
            self.reject
        )

        ok.clicked.connect(
            self.accept
        )

        buttons.addStretch()

        buttons.addWidget(
            cancel
        )

        buttons.addWidget(
            ok
        )

        layout.addRow(
            buttons
        )

# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class BookViewer(QMainWindow):

    def __init__(
        self,
        source_directory: Path,
    ):

        super().__init__()

        self.source_directory = (
            source_directory
        )

        self.setWindowTitle(
            "Scanned Book Viewer — "
            + source_directory.name
        )

        self.resize(
            1400,
            900
        )

        # ---------------------------------------------------------------
        # Images
        # ---------------------------------------------------------------

        self.image_paths: list[
            Path
        ] = []

        self.spreads: list[
            Spread
        ] = []

        self.current_spread = 0

        # ---------------------------------------------------------------
        # Playback
        # ---------------------------------------------------------------

        self.time_per_page = (
            DEFAULT_TIME_PER_PAGE
        )

        self.preload_spreads = (
            DEFAULT_PRELOAD_SPREADS
        )

        self.cache_size = (
            DEFAULT_CACHE_SIZE
        )

        self.autoplay = False

        self.autoplay_timer = QTimer(
            self
        )

        self.autoplay_timer.timeout.connect(
            self.advance
        )

        # ---------------------------------------------------------------
        # Image cache
        # ---------------------------------------------------------------

        self.cache = ImageCache(
            self.cache_size
        )

        # ---------------------------------------------------------------
        # Worker pool
        # ---------------------------------------------------------------

        self.thread_pool = (
            QThreadPool.globalInstance()
        )

        # Four simultaneous decoders is a reasonable compromise for HDDs.
        self.thread_pool.setMaxThreadCount(
            min(
                4,
                max(
                    2,
                    self.thread_pool.maxThreadCount(),
                ),
            )
        )

        self.loading: set[int] = set()

        # ---------------------------------------------------------------
        # Canvas
        # ---------------------------------------------------------------

        self.canvas = BookCanvas(
            self
        )

        self.setCentralWidget(
            self.canvas
        )

        # ---------------------------------------------------------------
        # UI
        # ---------------------------------------------------------------

        self._create_actions()
        self._create_toolbar()

        self.status = QStatusBar(
            self
        )

        self.setStatusBar(
            self.status
        )

        self.status.showMessage(
            "Loading..."
        )

        # ---------------------------------------------------------------
        # Fullscreen chrome
        # ---------------------------------------------------------------

        self.chrome_timer = QTimer(
            self
        )

        self.chrome_timer.setSingleShot(
            True
        )

        self.chrome_timer.timeout.connect(
            self.hide_fullscreen_chrome
        )

        self.ui_revealed = False

        # ---------------------------------------------------------------
        # Load initial book
        # ---------------------------------------------------------------

        self.load_source_directory()

    # ==================================================================
    # UI
    # ==================================================================

    def _create_actions(self):

        self.previous_action = QAction(
            "Previous",
            self
        )

        self.previous_action.setShortcut(
            QKeySequence("Left")
        )

        self.previous_action.triggered.connect(
            self.previous
        )

        # ---------------------------------------------------------------

        self.play_action = QAction(
            "Play",
            self
        )

        self.play_action.setShortcut(
            QKeySequence("Space")
        )

        self.play_action.triggered.connect(
            self.toggle_autoplay
        )

        # ---------------------------------------------------------------

        self.next_action = QAction(
            "Next",
            self
        )

        self.next_action.setShortcut(
            QKeySequence("Right")
        )

        self.next_action.triggered.connect(
            self.next
        )

        # ---------------------------------------------------------------

        self.settings_action = QAction(
            "Settings",
            self
        )

        self.settings_action.triggered.connect(
            self.show_settings
        )

        # ---------------------------------------------------------------

        self.zoom_in_action = QAction(
            "Zoom In",
            self
        )

        self.zoom_in_action.setShortcut(
            QKeySequence("Ctrl++")
        )

        self.zoom_in_action.triggered.connect(
            self.canvas_zoom_in
        )

        # ---------------------------------------------------------------

        self.zoom_out_action = QAction(
            "Zoom Out",
            self
        )

        self.zoom_out_action.setShortcut(
            QKeySequence("Ctrl+-")
        )

        self.zoom_out_action.triggered.connect(
            self.canvas_zoom_out
        )

        # ---------------------------------------------------------------

        self.zoom_reset_action = QAction(
            "Reset Zoom",
            self
        )

        self.zoom_reset_action.setShortcut(
            QKeySequence("Ctrl+0")
        )

        self.zoom_reset_action.triggered.connect(
            self.canvas.reset_zoom
        )

        # ---------------------------------------------------------------

        self.fullscreen_action = QAction(
            "Fullscreen",
            self
        )

        self.fullscreen_action.setShortcut(
            QKeySequence("F11")
        )

        self.fullscreen_action.triggered.connect(
            self.toggle_fullscreen
        )

        # ---------------------------------------------------------------

        self.quit_action = QAction(
            "Quit",
            self
        )

        self.quit_action.setShortcut(
            QKeySequence("Ctrl+Q")
        )

        self.quit_action.triggered.connect(
            self.close
        )

    def _create_toolbar(self):

        self.toolbar = QToolBar(
            "Controls"
        )

        self.toolbar.setMovable(
            False
        )

        self.toolbar.addAction(
            self.previous_action
        )

        self.toolbar.addAction(
            self.play_action
        )

        self.toolbar.addAction(
            self.next_action
        )

        self.toolbar.addSeparator()

        self.toolbar.addAction(
            self.zoom_out_action
        )

        self.toolbar.addAction(
            self.zoom_reset_action
        )

        self.toolbar.addAction(
            self.zoom_in_action
        )

        self.toolbar.addSeparator()

        self.toolbar.addAction(
            self.settings_action
        )

        self.toolbar.addAction(
            self.fullscreen_action
        )

        self.addToolBar(
            self.toolbar
        )

    # ==================================================================
    # Source loading
    # ==================================================================

    def load_source_directory(self):

        if not self.source_directory.exists():

            QMessageBox.critical(
                self,
                "Source directory not found",
                (
                    "The source directory does not exist:\n\n"
                    + str(self.source_directory)
                ),
            )

            return

        if not self.source_directory.is_dir():

            QMessageBox.critical(
                self,
                "Invalid source directory",
                (
                    "The supplied path is not a directory:\n\n"
                    + str(self.source_directory)
                ),
            )

            return

        try:

            self.image_paths = (
                find_source_images(
                    self.source_directory
                )
            )

        except Exception as exc:

            QMessageBox.critical(
                self,
                "Could not find scan images",
                (
                    "An error occurred while finding "
                    "scan images:\n\n"
                    + str(exc)
                ),
            )

            return

        self.spreads = make_spreads(
            len(self.image_paths)
        )

        self.current_spread = 0

        if not self.image_paths:

            extension = (
                configured_extension()
            )

            self.status.showMessage(
                "No "
                f"{extension}"
                " images found"
            )

            return

        self.status.showMessage(
            f"Found {len(self.image_paths)} "
            f"{configured_extension()} images"
        )

        self._display_current()

        # Begin filling the large buffer immediately.
        self._preload_images()

    # ==================================================================
    # Cache / background loading
    # ==================================================================

    def get_cached_image(
        self,
        index: int | None,
    ) -> QImage | None:

        if index is None:
            return None

        return self.cache.get(
            index
        )

    def _request_image(
        self,
        index: int,
    ):

        if (
            index < 0
            or index >= len(
                self.image_paths
            )
        ):
            return

        if self.cache.contains(
            index
        ):
            return

        if index in self.loading:
            return

        self.loading.add(
            index
        )

        signals = ImageLoadSignals()

        task = ImageLoadTask(
            index,
            self.image_paths[index],
            self.cache,
            signals,
        )

        signals.finished.connect(
            self._image_loaded
        )

        signals.failed.connect(
            self._image_failed
        )

        self.thread_pool.start(
            task
        )

    @Slot(int, object)
    def _image_loaded(
        self,
        index: int,
        image: QImage,
    ):

        self.loading.discard(
            index
        )

        self.canvas.update()

        # Keep the buffer full as workers finish.
        self._preload_images()

    @Slot(int, str)
    def _image_failed(
        self,
        index: int,
        message: str,
    ):

        self.loading.discard(
            index
        )

        print(
            message,
            file=sys.stderr
        )

    def _preload_images(self):

        if not self.spreads:
            return

        # Current spread + N future spreads.
        first = self.current_spread

        last = min(
            len(self.spreads),
            first
            + self.preload_spreads
            + 1,
        )

        for spread_index in range(
            first,
            last,
        ):

            spread = self.spreads[
                spread_index
            ]

            if spread.left is not None:
                self._request_image(
                    spread.left
                )

            if spread.right is not None:
                self._request_image(
                    spread.right
                )

    # ==================================================================
    # Display
    # ==================================================================

    def current_spread_object(self):

        if not self.spreads:
            return None

        return self.spreads[
            self.current_spread
        ]

    def _display_current(self):

        if not self.spreads:

            self.canvas.update()

            return

        spread = self.spreads[
            self.current_spread
        ]

        # Prioritize visible pages.
        if spread.left is not None:
            self._request_image(
                spread.left
            )

        if spread.right is not None:
            self._request_image(
                spread.right
            )

        self.canvas.update()

        # Continue filling the future buffer.
        self._preload_images()

        self._update_navigation()

    # ==================================================================
    # Autoplay
    # ==================================================================

    def _next_spread_ready(self):

        next_index = (
            self.current_spread + 1
        )

        if (
            next_index
            >= len(self.spreads)
        ):
            return False

        spread = self.spreads[
            next_index
        ]

        if spread.left is not None:

            if not self.cache.contains(
                spread.left
            ):
                return False

        if spread.right is not None:

            if not self.cache.contains(
                spread.right
            ):
                return False

        return True

    def toggle_autoplay(self):

        if self.autoplay:
            self.stop_autoplay()

        else:
            self.start_autoplay()

    def start_autoplay(self):

        if not self.spreads:
            return

        self.autoplay = True

        self._preload_images()

        self.autoplay_timer.start(
            max(
                1,
                round(
                    self.time_per_page
                    * 1000
                ),
            )
        )

        self.play_action.setText(
            "Pause"
        )

        self.status.showMessage(
            "Playing"
        )

    def stop_autoplay(self):

        self.autoplay = False

        self.autoplay_timer.stop()

        self.play_action.setText(
            "Play"
        )

        self._update_navigation()

    def advance(self):

        if not self.spreads:

            self.stop_autoplay()

            return

        if (
            self.current_spread
            >= len(self.spreads) - 1
        ):

            self.stop_autoplay()

            return

        # Don't display a partially loaded spread.
        if not self._next_spread_ready():

            self.status.showMessage(
                "Buffering..."
            )

            self._preload_images()

            return

        self.current_spread += 1

        self._display_current()

    # ==================================================================
    # Navigation
    # ==================================================================

    def previous(self):

        if not self.spreads:
            return

        if self.current_spread <= 0:
            return

        self.current_spread -= 1

        # IMPORTANT:
        #
        # We deliberately do NOT reset:
        #
        #     self.canvas.zoom
        #     self.canvas.pan
        #
        # Therefore zoom/pan survives page changes.
        self._display_current()

    def next(self):

        if not self.spreads:
            return

        if (
            self.current_spread
            >= len(self.spreads) - 1
        ):

            self.stop_autoplay()

            return

        self.current_spread += 1

        # Preserve zoom and pan.
        self._display_current()

    # ==================================================================
    # Zoom actions
    # ==================================================================

    def canvas_zoom_in(self):
        self.canvas.zoom_in()

    def canvas_zoom_out(self):
        self.canvas.zoom_out()

    # ==================================================================
    # Settings
    # ==================================================================

    def show_settings(self):

        was_playing = self.autoplay

        if was_playing:
            self.stop_autoplay()

        dialog = SettingsDialog(
            self.time_per_page,
            self.preload_spreads,
            self.cache_size,
            self,
        )

        if (
            dialog.exec()
            == QDialog.DialogCode.Accepted
        ):

            self.time_per_page = (
                dialog.time_spin.value()
            )

            self.preload_spreads = (
                dialog.preload_spin.value()
            )

            self.cache_size = (
                dialog.cache_spin.value()
            )

            self.cache.resize(
                self.cache_size
            )

            self._preload_images()

            if was_playing:
                self.start_autoplay()

        elif was_playing:

            self.start_autoplay()

    # ==================================================================
    # Fullscreen
    # ==================================================================

    def toggle_fullscreen(self):

        if self.isFullScreen():

            self.showNormal()

            self.toolbar.show()
            self.status.show()

        else:

            self.showFullScreen()

            # Headless by default.
            self.hide_fullscreen_chrome()

    def show_fullscreen_chrome(self):

        if not self.isFullScreen():
            return

        self.toolbar.show()
        self.status.show()

        self.ui_revealed = True

        self.chrome_timer.start(
            CHROME_HIDE_DELAY_MS
        )

    def hide_fullscreen_chrome(self):

        if not self.isFullScreen():
            return

        self.toolbar.hide()
        self.status.hide()

        self.ui_revealed = False

    def fullscreen_mouse_move(
        self,
        position,
    ):

        if not self.isFullScreen():
            return

        y = position.y()

        if (
            y <= EDGE_REVEAL_PIXELS
            or y
            >= self.height()
            - EDGE_REVEAL_PIXELS
        ):

            self.show_fullscreen_chrome()

        elif self.ui_revealed:

            self.chrome_timer.start(
                CHROME_HIDE_DELAY_MS
            )

    # ==================================================================
    # Keyboard
    # ==================================================================

    def keyPressEvent(
        self,
        event,
    ):

        key = event.key()

        modifiers = event.modifiers()

        if key == Qt.Key.Key_Space:

            self.toggle_autoplay()

            event.accept()

            return

        if key == Qt.Key.Key_Left:

            self.previous()

            event.accept()

            return

        if key == Qt.Key.Key_Right:

            self.next()

            event.accept()

            return

        if (
            modifiers
            & Qt.KeyboardModifier.ControlModifier
        ):

            if key in (
                Qt.Key.Key_Plus,
                Qt.Key.Key_Equal,
            ):

                self.canvas.zoom_in()

                event.accept()

                return

            if key in (
                Qt.Key.Key_Minus,
                Qt.Key.Key_Underscore,
            ):

                self.canvas.zoom_out()

                event.accept()

                return

            if key == Qt.Key.Key_0:

                self.canvas.reset_zoom()

                event.accept()

                return

        if key == Qt.Key.Key_F11:

            self.toggle_fullscreen()

            event.accept()

            return

        if (
            key == Qt.Key.Key_Escape
            and self.isFullScreen()
        ):

            self.showNormal()

            self.toolbar.show()
            self.status.show()

            event.accept()

            return

        super().keyPressEvent(
            event
        )

    # ==================================================================
    # Resize
    # ==================================================================

    def resizeEvent(
        self,
        event,
    ):

        super().resizeEvent(
            event
        )

        self.canvas.update()

    # ==================================================================
    # Status
    # ==================================================================

    def _update_navigation(self):

        if not self.spreads:
            return

        spread = self.spreads[
            self.current_spread
        ]

        if spread.left is None:

            pages = str(
                spread.right + 1
            )

        elif spread.right is None:

            pages = str(
                spread.left + 1
            )

        else:

            pages = (
                f"{spread.left + 1}"
                "–"
                f"{spread.right + 1}"
            )

        self.previous_action.setEnabled(
            self.current_spread > 0
        )

        self.next_action.setEnabled(
            self.current_spread
            < len(self.spreads) - 1
        )

        if not self.autoplay:

            self.status.showMessage(
                f"Pages {pages} / "
                f"{len(self.image_paths)}"
                "    "
                f"Spread "
                f"{self.current_spread + 1} / "
                f"{len(self.spreads)}"
                "    "
                f"Zoom "
                f"{self.canvas.zoom:.2f}×"
            )

# ============================================================================
# CLI
# ============================================================================

def usage():

    print(
        "Usage:\n"
        "\n"
        "    python book_viewer.py SOURCE_DIRECTORY\n"
    )

def main():

    # ------------------------------------------------------------------
    # Required positional argument
    # ------------------------------------------------------------------

    if len(sys.argv) != 2:

        usage()

        return 2

    source_directory = Path(
        sys.argv[1]
    ).expanduser().resolve()

    # ------------------------------------------------------------------
    # QApplication
    # ------------------------------------------------------------------

    app = QApplication(
        sys.argv
    )

    app.setApplicationName(
        "Scanned Book Viewer"
    )

    app.setOrganizationName(
        "ScannedBookViewer"
    )

    # ------------------------------------------------------------------
    # Viewer
    # ------------------------------------------------------------------

    viewer = BookViewer(
        source_directory
    )

    viewer.show()

    return app.exec()

if __name__ == "__main__":
    sys.exit(
        main()
    )
```

So the project-specific part is now quite small:

``` python
from _shared import (
    load_config,
    get_page_num,
)

config = load_config()
```

and:

``` python
def find_source_images(source_directory: Path) -> list[Path]:
    extension = configured_extension()

    paths = [
        path
        for path in source_directory.iterdir()
        if (
            path.is_file()
            and path.suffix.lower() == extension
        )
    ]

    paths.sort(key=get_page_num)

    return paths
```

That also means a directory containing, say:

``` text
scan_000001.tiff
scan_000002.tiff
scan_000003.tiff
...
```

will only load the configured format, **won't descend into subdirectories**, and the ordering is delegated to your project's page-number logic.

The CLI is now intentionally strict: **exactly one positional argument** is required.

## Prompt:

9/9/2026, 6:13:16 PM

ok, but the initial zoom level should be "fit pages", so everything of the 2 pages is visible

also, i want these hotkey bindings:  
arrow-up = zoom in  
arrow-down = zoom out  
F = toggle fullscreen (like a video player)  
Q = quit the app  
mousewheel = seek pages (up = next page, down = previous page) (seeking pages is used more often than zoom, so the mouse wheel should seek pages)

instead of "book_viewer.py" it should use the actual script name `__name__`

the menubar should also have a small text input for the current page number, with "previous page" and "next page" buttons left and right of the page number input. the visible page number should be the page number of the left page (except when there is no left page, then the page number of the right page)

## Response:

9/9/2026, 6:13:22 PM

> **Clarifying the implementation**
>
> I’m reconstructing the revised viewer as a complete script, preserving the requested navigation, cropping, zoom, page controls, and menu-bar input. I’m also interpreting “actual script name” as the filename derived from `__file__`, since `__name__` usually resolves to `__main__`.
>
> **Clarified the implementation**
>
> ------------------------------------------------------------------------
>
> **Implementing a Qt book viewer with navigation and zoom**
>
> **Implemented a Qt book viewer with navigation and zoom**
>
> Worked for 10s

Absolutely. I’ve changed the defaults and controls so the viewer behaves more like a video player:

- **Initial zoom = “Fit pages”**, showing the complete two-page spread.
- **↑ / ↓** = zoom in/out.
- **F** = fullscreen toggle.
- **Q** = quit.
- **Mouse wheel** = seek pages:
  - wheel up → next spread
  - wheel down → previous spread
- The zoom/pan state is still preserved while navigating.
- The menu bar now contains **Previous \| page-number input \| Next**.
- The page input displays the **left page number**, or the right page number when there is no left page.
- Entering a page number jumps to the spread containing that page.
- CLI help/usage uses the **actual script filename** rather than hard-coded `book_viewer.py`. Technically, Python's `__name__` is the module name, while `__file__` is what gives us the actual script filename.

\*\*document: \*\*

``` python
#!/usr/bin/env python3

import argparse
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

from PySide6.QtCore import (
    QObject,
    QRunnable,
    QMutex,
    QMutexLocker,
    Qt,
    QThreadPool,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QAction,
    QImage,
    QImageReader,
    QKeySequence,
    QPainter,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMenuBar,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from _shared import (
    load_config,
    get_page_num,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

config = load_config()

DEFAULT_PAGE_TIME = 0.2
DEFAULT_PRELOAD_SPREADS = 20
DEFAULT_CACHE_IMAGES = 60

FULLSCREEN_CHROME_HIDE_DELAY = 1800

ZOOM_MIN = 0.05
ZOOM_MAX = 8.0
ZOOM_STEP = 1.20

# ---------------------------------------------------------------------------
# Image cache
# ---------------------------------------------------------------------------

class ImageCache:
    """
    Thread-safe LRU cache of decoded QImages.

    QImage is implicitly shared and can safely be copied between threads.
    We deliberately make a detached copy after loading so that the worker
    thread owns fully decoded image data.
    """

    def __init__(self, max_images=DEFAULT_CACHE_IMAGES):
        self.max_images = max(1, max_images)

        self._mutex = QMutex()
        self._images = OrderedDict()

    def set_max_images(self, value):
        with QMutexLocker(self._mutex):
            self.max_images = max(1, value)

            while len(self._images) > self.max_images:
                self._images.popitem(last=False)

    def get(self, path):
        path = str(path)

        with QMutexLocker(self._mutex):
            image = self._images.get(path)

            if image is None:
                return None

            self._images.move_to_end(path)

            return image

    def contains(self, path):
        path = str(path)

        with QMutexLocker(self._mutex):
            return path in self._images

    def put(self, path, image):
        path = str(path)

        with QMutexLocker(self._mutex):
            self._images[path] = image
            self._images.move_to_end(path)

            while len(self._images) > self.max_images:
                self._images.popitem(last=False)

    def clear(self):
        with QMutexLocker(self._mutex):
            self._images.clear()

    def size(self):
        with QMutexLocker(self._mutex):
            return len(self._images)

# ---------------------------------------------------------------------------
# Background image loading
# ---------------------------------------------------------------------------

class ImageLoadSignals(QObject):
    loaded = Signal(str)
    failed = Signal(str, str)

class ImageLoadTask(QRunnable):
    def __init__(self, path, cache):
        super().__init__()

        self.path = str(path)
        self.cache = cache
        self.signals = ImageLoadSignals()

        self.setAutoDelete(True)

    def run(self):
        try:
            if self.cache.contains(self.path):
                self.signals.loaded.emit(self.path)
                return

            reader = QImageReader(self.path)
            reader.setAutoTransform(True)

            image = reader.read()

            if image.isNull():
                raise RuntimeError(
                    reader.errorString() or "Could not decode image"
                )

            # Force the decoded image data to be owned independently of the
            # reader / worker before putting it in the shared cache.
            image = image.copy()

            self.cache.put(self.path, image)
            self.signals.loaded.emit(self.path)

        except Exception as exc:
            self.signals.failed.emit(self.path, str(exc))

# ---------------------------------------------------------------------------
# Settings dialog
# ---------------------------------------------------------------------------

class SettingsDialog(QDialog):
    def __init__(
        self,
        parent=None,
        page_time=DEFAULT_PAGE_TIME,
        preload_spreads=DEFAULT_PRELOAD_SPREADS,
        cache_images=DEFAULT_CACHE_IMAGES,
    ):
        super().__init__(parent)

        self.setWindowTitle("Viewer settings")

        form = QFormLayout(self)

        self.page_time = QDoubleSpinBox()
        self.page_time.setRange(0.01, 60.0)
        self.page_time.setDecimals(2)
        self.page_time.setSingleStep(0.05)
        self.page_time.setSuffix(" s")
        self.page_time.setValue(page_time)

        self.preload_spreads = QSpinBox()
        self.preload_spreads.setRange(0, 1000)
        self.preload_spreads.setValue(preload_spreads)

        self.cache_images = QSpinBox()
        self.cache_images.setRange(2, 5000)
        self.cache_images.setValue(cache_images)

        form.addRow("Time per spread:", self.page_time)
        form.addRow("Preload spreads:", self.preload_spreads)
        form.addRow("Image cache:", self.cache_images)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        form.addRow(buttons)

# ---------------------------------------------------------------------------
# Book canvas
# ---------------------------------------------------------------------------

class BookCanvas(QWidget):
    """
    Displays a two-page spread as one continuous coordinate system.

    x = 0 is always the gutter.

    Left page:
        right edge == x=0

    Right page:
        left edge == x=0

    This means panning and zooming operate on the complete spread rather
    than independently on the two pages.
    """

    zoomChanged = Signal(float)
    positionChanged = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)

        self.left_image = None
        self.right_image = None

        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0

        self._dragging = False
        self._drag_start = None
        self._drag_pan_start = None

        self._fit_zoom = 1.0

    # ------------------------------------------------------------------
    # Images
    # ------------------------------------------------------------------

    def set_images(self, left_image, right_image):
        self.left_image = left_image
        self.right_image = right_image

        self.update()

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def combined_size(self):
        left_w = self.left_image.width() if self.left_image else 0
        left_h = self.left_image.height() if self.left_image else 0

        right_w = self.right_image.width() if self.right_image else 0
        right_h = self.right_image.height() if self.right_image else 0

        return (
            left_w + right_w,
            max(left_h, right_h),
        )

    def calculate_fit_zoom(self):
        width, height = self.combined_size()

        if width <= 0 or height <= 0:
            return 1.0

        # Small margins around the spread.
        available_w = max(1, self.width() - 12)
        available_h = max(1, self.height() - 12)

        zoom_x = available_w / width
        zoom_y = available_h / height

        return min(zoom_x, zoom_y)

    def fit_pages(self):
        """
        Fit the entire spread into the available viewport.

        This is the initial zoom mode.
        """
        self._fit_zoom = self.calculate_fit_zoom()
        self.zoom = max(ZOOM_MIN, min(ZOOM_MAX, self._fit_zoom))

        # Center the complete spread.
        self.pan_x = 0.0
        self.pan_y = 0.0

        self.zoomChanged.emit(self.zoom)
        self.positionChanged.emit(self.pan_x, self.pan_y)

        self.update()

    # ------------------------------------------------------------------
    # Coordinate conversion
    # ------------------------------------------------------------------

    def spread_to_widget(self, x, y):
        """
        Convert book coordinates to widget coordinates.

        The spread is centered in the viewport. pan_x/pan_y are in
        screen pixels.
        """
        width, height = self.combined_size()

        scaled_w = width * self.zoom
        scaled_h = height * self.zoom

        origin_x = (self.width() - scaled_w) / 2.0 + self.pan_x
        origin_y = (self.height() - scaled_h) / 2.0 + self.pan_y

        return (
            origin_x + x * self.zoom,
            origin_y + y * self.zoom,
        )

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event):
        painter = QPainter(self)

        painter.setRenderHint(
            QPainter.RenderHint.SmoothPixmapTransform,
            self.zoom != 1.0,
        )

        painter.fillRect(self.rect(), Qt.black)

        left = self.left_image
        right = self.right_image

        if left is None and right is None:
            painter.end()
            return

        left_w = left.width() if left else 0
        left_h = left.height() if left else 0

        right_w = right.width() if right else 0
        right_h = right.height() if right else 0

        combined_w = left_w + right_w
        combined_h = max(left_h, right_h)

        scaled_w = combined_w * self.zoom
        scaled_h = combined_h * self.zoom

        origin_x = (self.width() - scaled_w) / 2.0 + self.pan_x
        origin_y = (self.height() - scaled_h) / 2.0 + self.pan_y

        # Vertically center pages of different heights.
        if left is not None:
            y = (combined_h - left_h) / 2.0

            if self.zoom == 1.0:
                painter.drawImage(
                    int(round(origin_x)),
                    int(round(origin_y + y)),
                    left,
                )
            else:
                target = (
                    origin_x,
                    origin_y + y * self.zoom,
                    left_w * self.zoom,
                    left_h * self.zoom,
                )
                painter.drawImage(target, left)

        if right is not None:
            # Right page begins exactly where the left page ends.
            x = left_w

            y = (combined_h - right_h) / 2.0

            if self.zoom == 1.0:
                painter.drawImage(
                    int(round(origin_x + x)),
                    int(round(origin_y + y)),
                    right,
                )
            else:
                target = (
                    origin_x + x * self.zoom,
                    origin_y + y * self.zoom,
                    right_w * self.zoom,
                    right_h * self.zoom,
                )
                painter.drawImage(target, right)

        painter.end()

    # ------------------------------------------------------------------
    # Zoom
    # ------------------------------------------------------------------

    def zoom_in(self):
        self.set_zoom(self.zoom * ZOOM_STEP)

    def zoom_out(self):
        self.set_zoom(self.zoom / ZOOM_STEP)

    def set_zoom(self, value, anchor=None):
        value = max(ZOOM_MIN, min(ZOOM_MAX, value))

        if abs(value - self.zoom) < 0.0001:
            return

        if anchor is None:
            anchor = self.rect().center()

        # Preserve the content under the mouse while zooming.
        old_zoom = self.zoom

        if old_zoom > 0:
            factor = value / old_zoom

            cx = self.width() / 2.0 + self.pan_x
            cy = self.height() / 2.0 + self.pan_y

            ax = anchor.x()
            ay = anchor.y()

            self.pan_x = ax - (ax - cx) * factor - self.width() / 2.0
            self.pan_y = ay - (ay - cy) * factor - self.height() / 2.0

        self.zoom = value

        self.zoomChanged.emit(self.zoom)
        self.positionChanged.emit(self.pan_x, self.pan_y)

        self.update()

    def reset_zoom(self):
        self.fit_pages()

    # ------------------------------------------------------------------
    # Mouse interaction
    # ------------------------------------------------------------------

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._drag_start = event.position()
            self._drag_pan_start = (
                self.pan_x,
                self.pan_y,
            )

            self.setCursor(Qt.CursorShape.ClosedHandCursor)

            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging:
            delta = event.position() - self._drag_start

            self.pan_x = self._drag_pan_start[0] + delta.x()
            self.pan_y = self._drag_pan_start[1] + delta.y()

            self.positionChanged.emit(self.pan_x, self.pan_y)
            self.update()

            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self.setCursor(Qt.CursorShape.ArrowCursor)

            event.accept()
            return

        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        # IMPORTANT:
        # The mouse wheel is intentionally NOT zoom.
        #
        # Page seeking is much more common, so the main window handles
        # wheel events as next/previous spread navigation.
        event.ignore()

    # ------------------------------------------------------------------
    # Resize
    # ------------------------------------------------------------------

    def resizeEvent(self, event):
        super().resizeEvent(event)

        # Only automatically refit while we are still exactly at the
        # initial fit zoom. Once the user zooms/pans, preserve the state.
        if (
            self.left_image is not None
            or self.right_image is not None
        ):
            if abs(self.zoom - self._fit_zoom) < 0.0001:
                self._fit_zoom = self.calculate_fit_zoom()
                self.zoom = max(
                    ZOOM_MIN,
                    min(ZOOM_MAX, self._fit_zoom),
                )

        self.update()

# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class BookViewer(QMainWindow):
    def __init__(self, image_paths):
        super().__init__()

        self.image_paths = image_paths
        self.total_pages = len(image_paths)

        self.page_time = DEFAULT_PAGE_TIME
        self.preload_spreads = DEFAULT_PRELOAD_SPREADS
        self.cache_images = DEFAULT_CACHE_IMAGES

        self.cache = ImageCache(self.cache_images)

        self.thread_pool = QThreadPool.globalInstance()

        # Two workers is generally a good compromise for a spinning HDD:
        # enough concurrency to keep decoding busy without causing a large
        # number of random disk seeks.
        self.thread_pool.setMaxThreadCount(
            min(4, max(2, self.thread_pool.maxThreadCount()))
        )

        self.loading = set()

        self.spreads = self._build_spreads()
        self.current_spread = 0

        self.playing = False

        self._fullscreen = False
        self._chrome_hide_timer = QTimer(self)
        self._chrome_hide_timer.setSingleShot(True)
        self._chrome_hide_timer.timeout.connect(self._hide_fullscreen_chrome)

        self.canvas = BookCanvas(self)
        self.setCentralWidget(self.canvas)

        self._create_menus()
        self._create_status_bar()

        self._update_window_title()

        self._load_current_spread()
        self._preload_around_current()

        # Give Qt one event loop iteration before calculating the initial
        # fit. This ensures the canvas has its final size.
        QTimer.singleShot(0, self._initial_fit)

        self.setMouseTracking(True)

    # ------------------------------------------------------------------
    # Spread construction
    # ------------------------------------------------------------------

    def _build_spreads(self):
        """
        Page 1 is the right-hand/front page.

        Then:
            2 + 3
            4 + 5
            6 + 7
            ...

        If the total number of pages is even, the final even page is a
        standalone left page.
        """

        spreads = []

        if not self.image_paths:
            return spreads

        # First page alone on the right.
        spreads.append(
            {
                "left": None,
                "right": self.image_paths[0],
            }
        )

        index = 1

        while index < len(self.image_paths):
            left = self.image_paths[index]
            right = None

            if index + 1 < len(self.image_paths):
                right = self.image_paths[index + 1]

            spreads.append(
                {
                    "left": left,
                    "right": right,
                }
            )

            index += 2

        return spreads

    # ------------------------------------------------------------------
    # Menu bar
    # ------------------------------------------------------------------

    def _create_menus(self):
        menu_bar = QMenuBar(self)

        # --------------------------------------------------------------
        # File
        # --------------------------------------------------------------

        file_menu = QMenu("&File", self)

        settings_action = QAction("&Settings...", self)
        settings_action.triggered.connect(self.show_settings)
        file_menu.addAction(settings_action)

        file_menu.addSeparator()

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence("Q"))
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        menu_bar.addMenu(file_menu)

        # --------------------------------------------------------------
        # View
        # --------------------------------------------------------------

        view_menu = QMenu("&View", self)

        fit_action = QAction("Fit pages", self)
        fit_action.setShortcut(QKeySequence("Ctrl+0"))
        fit_action.triggered.connect(self.canvas.fit_pages)
        view_menu.addAction(fit_action)

        zoom_in_action = QAction("Zoom in", self)
        zoom_in_action.setShortcut(QKeySequence("Up"))
        zoom_in_action.triggered.connect(self.canvas.zoom_in)
        view_menu.addAction(zoom_in_action)

        zoom_out_action = QAction("Zoom out", self)
        zoom_out_action.setShortcut(QKeySequence("Down"))
        zoom_out_action.triggered.connect(self.canvas.zoom_out)
        view_menu.addAction(zoom_out_action)

        view_menu.addSeparator()

        fullscreen_action = QAction("Fullscreen", self)
        fullscreen_action.setShortcut(QKeySequence("F"))
        fullscreen_action.triggered.connect(self.toggle_fullscreen)
        view_menu.addAction(fullscreen_action)

        menu_bar.addMenu(view_menu)

        # --------------------------------------------------------------
        # Playback
        # --------------------------------------------------------------

        playback_menu = QMenu("&Playback", self)

        play_action = QAction("Play / Pause", self)
        play_action.setShortcut(QKeySequence("Space"))
        play_action.triggered.connect(self.toggle_play)
        playback_menu.addAction(play_action)

        previous_action = QAction("Previous page", self)
        previous_action.setShortcut(QKeySequence("Left"))
        previous_action.triggered.connect(self.previous_spread)
        playback_menu.addAction(previous_action)

        next_action = QAction("Next page", self)
        next_action.setShortcut(QKeySequence("Right"))
        next_action.triggered.connect(self.next_spread)
        playback_menu.addAction(next_action)

        menu_bar.addMenu(playback_menu)

        # --------------------------------------------------------------
        # Page navigation embedded in the menu bar
        # --------------------------------------------------------------

        navigation = QWidget(self)

        nav_layout = QHBoxLayout(navigation)
        nav_layout.setContentsMargins(4, 0, 4, 0)
        nav_layout.setSpacing(2)

        self.previous_button = QPushButton("‹", navigation)
        self.previous_button.setToolTip("Previous page")
        self.previous_button.setFixedWidth(28)
        self.previous_button.clicked.connect(self.previous_spread)

        self.page_input = QLineEdit(navigation)
        self.page_input.setFixedWidth(65)
        self.page_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.page_input.setToolTip(
            "Enter a page number and press Enter"
        )
        self.page_input.returnPressed.connect(
            self._page_input_activated
        )

        self.next_button = QPushButton("›", navigation)
        self.next_button.setToolTip("Next page")
        self.next_button.setFixedWidth(28)
        self.next_button.clicked.connect(self.next_spread)

        nav_layout.addWidget(self.previous_button)
        nav_layout.addWidget(self.page_input)
        nav_layout.addWidget(self.next_button)

        menu_bar.setCornerWidget(
            navigation,
            Qt.Corner.TopRightCorner,
        )

        self.setMenuBar(menu_bar)

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------

    def _create_status_bar(self):
        status = QStatusBar(self)

        self.status_label = QLabel()
        status.addPermanentWidget(self.status_label)

        self.setStatusBar(status)

        self._update_status()

    # ------------------------------------------------------------------
    # Initial state
    # ------------------------------------------------------------------

    def _initial_fit(self):
        self.canvas.fit_pages()

    # ------------------------------------------------------------------
    # Current page display
    # ------------------------------------------------------------------

    def _current_visible_page(self):
        if not self.spreads:
            return 0

        spread = self.spreads[self.current_spread]

        path = spread["left"] or spread["right"]

        if path is None:
            return 0

        try:
            return get_page_num(path)
        except Exception:
            return self.current_spread + 1

    def _update_page_input(self):
        page = self._current_visible_page()

        self.page_input.blockSignals(True)
        self.page_input.setText(str(page) if page else "")
        self.page_input.blockSignals(False)

    # ------------------------------------------------------------------
    # Page input
    # ------------------------------------------------------------------

    def _page_input_activated(self):
        text = self.page_input.text().strip()

        try:
            page = int(text)
        except ValueError:
            self._update_page_input()
            return

        self.go_to_page(page)

    def go_to_page(self, page_number):
        if not self.spreads:
            return

        # Clamp to valid page range.
        page_number = max(
            1,
            min(self.total_pages, page_number),
        )

        # Find the spread containing this actual page number.
        for index, spread in enumerate(self.spreads):
            pages = []

            if spread["left"] is not None:
                pages.append(spread["left"])

            if spread["right"] is not None:
                pages.append(spread["right"])

            for path in pages:
                try:
                    if get_page_num(path) == page_number:
                        self._set_spread(index)
                        return
                except Exception:
                    pass

        # Fallback for unusual filenames / page numbering.
        nearest = 0
        best_distance = float("inf")

        for index, spread in enumerate(self.spreads):
            path = spread["left"] or spread["right"]

            try:
                number = get_page_num(path)
            except Exception:
                continue

            distance = abs(number - page_number)

            if distance < best_distance:
                best_distance = distance
                nearest = index

        self._set_spread(nearest)

    # ------------------------------------------------------------------
    # Spread navigation
    # ------------------------------------------------------------------

    def _set_spread(self, index):
        if not self.spreads:
            return

        index = max(
            0,
            min(len(self.spreads) - 1, index),
        )

        if index == self.current_spread:
            self._load_current_spread()
            return

        self.current_spread = index

        self._load_current_spread()
        self._preload_around_current()

        self._update_page_input()
        self._update_status()
        self._update_window_title()

    def previous_spread(self):
        if not self.spreads:
            return

        self._set_spread(self.current_spread - 1)

    def next_spread(self):
        if not self.spreads:
            return

        self._set_spread(self.current_spread + 1)

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def _request_image(self, path):
        if path is None:
            return

        path = str(path)

        if self.cache.contains(path):
            return

        if path in self.loading:
            return

        self.loading.add(path)

        task = ImageLoadTask(path, self.cache)

        task.signals.loaded.connect(self._image_loaded)
        task.signals.failed.connect(self._image_failed)

        self.thread_pool.start(task)

    def _image_loaded(self, path):
        self.loading.discard(path)

        # The currently visible spread may now be complete.
        self._load_current_spread()

        # Keep the future playback buffer full.
        self._preload_around_current()

    def _image_failed(self, path, error):
        self.loading.discard(path)

        if (
            self.spreads
            and (
                path == self.spreads[self.current_spread]["left"]
                or path == self.spreads[self.current_spread]["right"]
            )
        ):
            self.status_label.setText(
                f"Could not load {Path(path).name}: {error}"
            )

    def _load_current_spread(self):
        if not self.spreads:
            self.canvas.set_images(None, None)
            return

        spread = self.spreads[self.current_spread]

        left = None
        right = None

        if spread["left"] is not None:
            left = self.cache.get(spread["left"])

            if left is None:
                self._request_image(spread["left"])

        if spread["right"] is not None:
            right = self.cache.get(spread["right"])

            if right is None:
                self._request_image(spread["right"])

        self.canvas.set_images(left, right)

        # If the images arrived after the first paint, fit the initial
        # spread again. Do NOT reset user zoom/pan during navigation.
        if (
            self.canvas.zoom == 1.0
            and self.canvas.pan_x == 0.0
            and self.canvas.pan_y == 0.0
        ):
            pass

        self._update_status()

    # ------------------------------------------------------------------
    # Aggressive preloading
    # ------------------------------------------------------------------

    def _preload_around_current(self):
        if not self.spreads:
            return

        start = self.current_spread
        end = min(
            len(self.spreads),
            start + 1 + self.preload_spreads,
        )

        # Queue images in exact playback order:
        #
        # current spread first,
        # then next spread,
        # then the spread after that, etc.
        #
        # This is much more useful for sequential playback than simply
        # scanning the directory or requesting images in arbitrary order.
        for spread_index in range(start, end):
            spread = self.spreads[spread_index]

            if spread["left"] is not None:
                self._request_image(spread["left"])

            if spread["right"] is not None:
                self._request_image(spread["right"])

    def _current_spread_ready(self):
        if not self.spreads:
            return True

        spread = self.spreads[self.current_spread]

        if spread["left"] is not None:
            if not self.cache.contains(spread["left"]):
                return False

        if spread["right"] is not None:
            if not self.cache.contains(spread["right"]):
                return False

        return True

    def _next_spread_ready(self):
        if self.current_spread + 1 >= len(self.spreads):
            return True

        spread = self.spreads[self.current_spread + 1]

        if spread["left"] is not None:
            if not self.cache.contains(spread["left"]):
                return False

        if spread["right"] is not None:
            if not self.cache.contains(spread["right"]):
                return False

        return True

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def toggle_play(self):
        if not self.spreads:
            return

        self.playing = not self.playing

        if self.playing:
            self._start_playback()
        else:
            self._stop_playback()

        self._update_status()

    def _start_playback(self):
        self._preload_around_current()

        # Wait until the current spread and at least the next spread are
        # decoded. This prevents playback from immediately outrunning
        # the HDD.
        self._wait_for_playback_buffer()

    def _stop_playback(self):
        self._update_status()

    def _wait_for_playback_buffer(self):
        if not self.playing:
            return

        if self.current_spread >= len(self.spreads) - 1:
            self.playing = False
            self._update_status()
            return

        if not self._current_spread_ready():
            self.status_label.setText("Buffering…")
            QTimer.singleShot(30, self._wait_for_playback_buffer)
            return

        if not self._next_spread_ready():
            self.status_label.setText("Buffering next page…")
            QTimer.singleShot(30, self._wait_for_playback_buffer)
            return

        self._schedule_next_playback()

    def _schedule_next_playback(self):
        if not self.playing:
            return

        delay = max(
            1,
            int(self.page_time * 1000),
        )

        QTimer.singleShot(
            delay,
            self._advance_playback,
        )

    def _advance_playback(self):
        if not self.playing:
            return

        if self.current_spread >= len(self.spreads) - 1:
            self.playing = False
            self._update_status()
            return

        self.next_spread()

        self._preload_around_current()

        if self._next_spread_ready():
            self._schedule_next_playback()
        else:
            self._wait_for_playback_buffer()

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def show_settings(self):
        dialog = SettingsDialog(
            self,
            self.page_time,
            self.preload_spreads,
            self.cache_images,
        )

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        self.page_time = dialog.page_time.value()
        self.preload_spreads = dialog.preload_spreads.value()

        self.cache_images = dialog.cache_images.value()
        self.cache.set_max_images(self.cache_images)

        self._preload_around_current()
        self._update_status()

    # ------------------------------------------------------------------
    # Fullscreen
    # ------------------------------------------------------------------

    def toggle_fullscreen(self):
        if self._fullscreen:
            self._leave_fullscreen()
        else:
            self._enter_fullscreen()

    def _enter_fullscreen(self):
        self._fullscreen = True

        # Headless fullscreen: hide all chrome by default.
        self.menuBar().hide()
        self.statusBar().hide()

        self.showFullScreen()

        self._chrome_hide_timer.stop()

        self.canvas.setFocus()

    def _leave_fullscreen(self):
        self._fullscreen = False

        self._chrome_hide_timer.stop()

        self.showNormal()

        self.menuBar().show()
        self.statusBar().show()

        self.canvas.setFocus()

    def _show_fullscreen_chrome(self):
        if not self._fullscreen:
            return

        self.menuBar().show()
        self.statusBar().show()

        self._chrome_hide_timer.start(
            FULLSCREEN_CHROME_HIDE_DELAY
        )

    def _hide_fullscreen_chrome(self):
        if not self._fullscreen:
            return

        self.menuBar().hide()
        self.statusBar().hide()

    # ------------------------------------------------------------------
    # Mouse movement in fullscreen
    # ------------------------------------------------------------------

    def mouseMoveEvent(self, event):
        if self._fullscreen:
            y = event.position().y()

            # Reveal chrome when the pointer reaches the top or bottom.
            if y <= 8 or y >= self.height() - 8:
                self._show_fullscreen_chrome()

        super().mouseMoveEvent(event)

    def event(self, event):
        if self._fullscreen:
            # Mouse movement can occur over child widgets, so use the
            # application-wide mouse position.
            if event.type() == event.Type.MouseMove:
                pos = self.mapFromGlobal(
                    self.cursor().pos()
                )

                if pos.y() <= 8 or pos.y() >= self.height() - 8:
                    self._show_fullscreen_chrome()

        return super().event(event)

    # ------------------------------------------------------------------
    # Keyboard shortcuts
    # ------------------------------------------------------------------

    def keyPressEvent(self, event):
        key = event.key()

        if key == Qt.Key.Key_Q:
            self.close()
            return

        if key == Qt.Key.Key_F:
            self.toggle_fullscreen()
            return

        if key == Qt.Key.Key_Up:
            self.canvas.zoom_in()
            return

        if key == Qt.Key.Key_Down:
            self.canvas.zoom_out()
            return

        if key == Qt.Key.Key_Left:
            self.previous_spread()
            return

        if key == Qt.Key.Key_Right:
            self.next_spread()
            return

        if key == Qt.Key.Key_Space:
            self.toggle_play()
            return

        if key in (
            Qt.Key.Key_Plus,
            Qt.Key.Key_Equal,
        ) and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.canvas.zoom_in()
            return

        if key == Qt.Key.Key_Minus:
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                self.canvas.zoom_out()
                return

        if (
            key == Qt.Key.Key_0
            and event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            self.canvas.fit_pages()
            return

        super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # Mouse wheel = page seek
    # ------------------------------------------------------------------

    def wheelEvent(self, event):
        delta = event.angleDelta().y()

        if delta > 0:
            self.next_spread()
            event.accept()
            return

        if delta < 0:
            self.previous_spread()
            event.accept()
            return

        super().wheelEvent(event)

    # ------------------------------------------------------------------
    # Status / title
    # ------------------------------------------------------------------

    def _update_status(self):
        if not hasattr(self, "status_label"):
            return

        if not self.spreads:
            self.status_label.setText("No pages")
            return

        page = self._current_visible_page()

        state = "Playing" if self.playing else "Paused"

        self.status_label.setText(
            f"{state}  •  Page {page}  •  "
            f"Spread {self.current_spread + 1}/{len(self.spreads)}  •  "
            f"Zoom {self.canvas.zoom:.2f}×  •  "
            f"Cache {self.cache.size()}/{self.cache_images}"
        )

        self._update_page_input()

    def _update_window_title(self):
        page = self._current_visible_page()

        if page:
            self.setWindowTitle(
                f"Scanned Book Viewer — Page {page}"
            )
        else:
            self.setWindowTitle("Scanned Book Viewer")

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        self.playing = False

        self.thread_pool.clear()

        super().closeEvent(event)

# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_images(source_directory):
    source_directory = Path(source_directory)

    if not source_directory.is_dir():
        raise ValueError(
            f"Not a directory: {source_directory}"
        )

    scan_format = str(config.scan_format).lower().lstrip(".")

    supported = {
        scan_format,
    }

    # Some configurations may use a format such as "tif" while the
    # actual files use "tiff", or vice versa.
    if scan_format == "tif":
        supported.add("tiff")
    elif scan_format == "tiff":
        supported.add("tif")

    images = []

    for path in source_directory.iterdir():
        if not path.is_file():
            continue

        if path.suffix.lower().lstrip(".") not in supported:
            continue

        images.append(path)

    images.sort(
        key=lambda path: (
            get_page_num(path),
            path.name.lower(),
        )
    )

    return images

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # __name__ is the Python module name ("__main__" when executed).
    # __file__ is the actual filename, which is what we want to show in
    # command-line usage instead of hard-coding "book_viewer.py".
    script_name = Path(__file__).name

    parser = argparse.ArgumentParser(
        prog=script_name,
        description="Specialized scanned-book page viewer.",
    )

    parser.add_argument(
        "source_directory",
        metavar="SOURCE_DIRECTORY",
        help="Directory containing the scanned page images.",
    )

    args = parser.parse_args()

    try:
        image_paths = find_images(args.source_directory)

    except Exception as exc:
        print(
            f"{script_name}: error: {exc}",
            file=sys.stderr,
        )
        return 2

    if not image_paths:
        print(
            f"{script_name}: no "
            f"{config.scan_format} images found in "
            f"{args.source_directory}",
            file=sys.stderr,
        )
        return 1

    app = QApplication(sys.argv)

    app.setApplicationName("Scanned Book Viewer")
    app.setOrganizationName("Scanned Book Viewer")

    window = BookViewer(image_paths)

    # Start maximized rather than fullscreen.
    window.showMaximized()

    window.canvas.setFocus()

    return app.exec()

if __name__ == "__main__":
    raise SystemExit(main())
```

One subtle but important point: **“fit pages” is now the initial zoom mode**, but once you manually zoom or pan, navigating pages does **not** reset that state. So you can zoom into a particular region and flip through the book while maintaining the same zoom/pan position.

## Prompt:

9/9/2026, 6:18:49 PM

now it crashes when i try to enter fullscreen mode:

    QBackingStore::endPaint() called with active painter; did you forget to destroy it or call QPainter::end() on it?
    Error calling Python override of QMainWindow::event(): Traceback (most recent call last):
      File "/media/ZYD82805_24TB/root/home/user/src/milahu/hocr-files-template-repo/view-pages-4.py", line 1256, in event
        return super().event(event)
               ~~~~~~~~~~~~~^^^^^^^
      File "/media/ZYD82805_24TB/root/home/user/src/milahu/hocr-files-template-repo/view-pages-4.py", line 427, in paintEvent
        painter.drawImage(target, right)
        ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^
    TypeError: Error calling Python override of QWidget::paintEvent(): 'PySide6.QtGui.QPainter.drawImage' called with wrong argument types:
      PySide6.QtGui.QPainter.drawImage(tuple, QImage)
    Supported signatures:
      PySide6.QtGui.QPainter.drawImage(p: PySide6.QtCore.QPoint, image: PySide6.QtGui.QImage, /)
      PySide6.QtGui.QPainter.drawImage(p: PySide6.QtCore.QPoint, image: PySide6.QtGui.QImage, sr: PySide6.QtCore.QRect, /, flags: PySide6.QtCore.Qt.ImageConversionFlag = Instance(Qt.AutoColor))
      PySide6.QtGui.QPainter.drawImage(r: PySide6.QtCore.QRect, image: PySide6.QtGui.QImage, /)
      PySide6.QtGui.QPainter.drawImage(targetRect: PySide6.QtCore.QRect, image: PySide6.QtGui.QImage, sourceRect: PySide6.QtCore.QRect, /, flags: PySide6.QtCore.Qt.ImageConversionFlag = Instance(Qt.AutoColor))
      PySide6.QtGui.QPainter.drawImage(r: PySide6.QtCore.QRectF | PySide6.QtCore.QRect, image: PySide6.QtGui.QImage, /)
      PySide6.QtGui.QPainter.drawImage(targetRect: PySide6.QtCore.QRectF | PySide6.QtCore.QRect, image: PySide6.QtGui.QImage, sourceRect: PySide6.QtCore.QRectF | PySide6.QtCore.QRect, /, flags: PySide6.QtCore.Qt.ImageConversionFlag = Instance(Qt.AutoColor))
      PySide6.QtGui.QPainter.drawImage(p: PySide6.QtCore.QPointF | PySide6.QtCore.QPoint | PySide6.QtGui.QPainterPath.Element, image: PySide6.QtGui.QImage, /)
      PySide6.QtGui.QPainter.drawImage(p: PySide6.QtCore.QPointF | PySide6.QtCore.QPoint | PySide6.QtGui.QPainterPath.Element, image: PySide6.QtGui.QImage, sr: PySide6.QtCore.QRectF | PySide6.QtCore.QRect, /, flags: PySide6.QtCore.Qt.ImageConversionFlag = Instance(Qt.AutoColor))
      PySide6.QtGui.QPainter.drawImage(x: int, y: int, image: PySide6.QtGui.QImage, /, sx: int | None = None, sy: int | None = None, sw: int = -1, sh: int = -1, flags: PySide6.QtCore.Qt.ImageConversionFlag = Instance(Qt.AutoColor))
    QPaintDevice: Cannot destroy paint device that is being painted
    Error calling Python override of QWidget::paintEvent(): Segmentation fault         (core dumped) /media/ZYD82805_24TB/root/home/user/src/milahu/hocr-files-template-repo/view-pages-4.py 065-remove-page-borders

also the initial "zoom to fit pages" is not working

also, i want the opposite scroll direction:  
scroll up = previous page  
scroll down = next page  
(this feels more natural with a top-down reading order)

## Response:

9/9/2026, 6:18:51 PM

Yes — there are two separate bugs here.

1.  The fullscreen crash is caused by `drawImage()` receiving a Python tuple instead of a `QRectF`. The tuple overload isn't accepted by PySide6.
2.  “Fit pages” was calculated before the asynchronously loaded images were available, so the actual dimensions were unknown. The fix is to perform the initial fit **after the current spread has finished loading**.

I also reversed the wheel direction as requested:

- **Wheel up → previous page**
- **Wheel down → next page**

I’ve also simplified the fullscreen mouse handling so it doesn't unnecessarily override `QMainWindow.event()`, which was involved in the traceback.

\*\*document: \*\*

``` python
#!/usr/bin/env python3

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

from PySide6.QtCore import (
    QObject,
    QPointF,
    QRectF,
    QRunnable,
    QMutex,
    QMutexLocker,
    Qt,
    QThreadPool,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QAction,
    QImage,
    QImageReader,
    QKeySequence,
    QPainter,
)
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMenuBar,
    QPushButton,
    QSpinBox,
    QStatusBar,
    QFormLayout,
    QWidget,
)

from _shared import (
    load_config,
    get_page_num,
)

config = load_config()

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_PAGE_TIME = 0.2
DEFAULT_PRELOAD_SPREADS = 20
DEFAULT_CACHE_IMAGES = 60

FULLSCREEN_CHROME_HIDE_DELAY = 1800

ZOOM_MIN = 0.05
ZOOM_MAX = 8.0
ZOOM_STEP = 1.20

# ---------------------------------------------------------------------------
# Image cache
# ---------------------------------------------------------------------------

class ImageCache:
    def __init__(self, max_images=DEFAULT_CACHE_IMAGES):
        self.max_images = max(1, max_images)

        self._mutex = QMutex()
        self._images = OrderedDict()

    def set_max_images(self, value):
        with QMutexLocker(self._mutex):
            self.max_images = max(1, value)

            while len(self._images) > self.max_images:
                self._images.popitem(last=False)

    def get(self, path):
        path = str(path)

        with QMutexLocker(self._mutex):
            image = self._images.get(path)

            if image is None:
                return None

            self._images.move_to_end(path)

            return image

    def contains(self, path):
        path = str(path)

        with QMutexLocker(self._mutex):
            return path in self._images

    def put(self, path, image):
        path = str(path)

        with QMutexLocker(self._mutex):
            self._images[path] = image
            self._images.move_to_end(path)

            while len(self._images) > self.max_images:
                self._images.popitem(last=False)

    def clear(self):
        with QMutexLocker(self._mutex):
            self._images.clear()

    def size(self):
        with QMutexLocker(self._mutex):
            return len(self._images)

# ---------------------------------------------------------------------------
# Background image loading
# ---------------------------------------------------------------------------

class ImageLoadSignals(QObject):
    loaded = Signal(str)
    failed = Signal(str, str)

class ImageLoadTask(QRunnable):
    def __init__(self, path, cache):
        super().__init__()

        self.path = str(path)
        self.cache = cache
        self.signals = ImageLoadSignals()

        self.setAutoDelete(True)

    def run(self):
        try:
            if self.cache.contains(self.path):
                self.signals.loaded.emit(self.path)
                return

            reader = QImageReader(self.path)
            reader.setAutoTransform(True)

            image = reader.read()

            if image.isNull():
                raise RuntimeError(
                    reader.errorString() or "Could not decode image"
                )

            # Force the decoded data to become independent of the reader.
            image = image.copy()

            self.cache.put(self.path, image)

            self.signals.loaded.emit(self.path)

        except Exception as exc:
            self.signals.failed.emit(
                self.path,
                str(exc),
            )

# ---------------------------------------------------------------------------
# Settings dialog
# ---------------------------------------------------------------------------

class SettingsDialog(QDialog):
    def __init__(
        self,
        parent=None,
        page_time=DEFAULT_PAGE_TIME,
        preload_spreads=DEFAULT_PRELOAD_SPREADS,
        cache_images=DEFAULT_CACHE_IMAGES,
    ):
        super().__init__(parent)

        self.setWindowTitle("Viewer settings")

        form = QFormLayout(self)

        self.page_time = QDoubleSpinBox()
        self.page_time.setRange(0.01, 60.0)
        self.page_time.setDecimals(2)
        self.page_time.setSingleStep(0.05)
        self.page_time.setSuffix(" s")
        self.page_time.setValue(page_time)

        self.preload_spreads = QSpinBox()
        self.preload_spreads.setRange(0, 1000)
        self.preload_spreads.setValue(preload_spreads)

        self.cache_images = QSpinBox()
        self.cache_images.setRange(2, 5000)
        self.cache_images.setValue(cache_images)

        form.addRow(
            "Time per spread:",
            self.page_time,
        )

        form.addRow(
            "Preload spreads:",
            self.preload_spreads,
        )

        form.addRow(
            "Image cache:",
            self.cache_images,
        )

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok
            | QDialogButtonBox.Cancel
        )

        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        form.addRow(buttons)

# ---------------------------------------------------------------------------
# Book canvas
# ---------------------------------------------------------------------------

class BookCanvas(QWidget):
    """
    The complete two-page spread is treated as one image.

    Coordinate system:

        left page       right page
    <--------------|---------------->
                   0
                 gutter

    The right edge of the left page and the left edge of the right page
    therefore always meet exactly at x=0.
    """

    zoomChanged = Signal(float)
    positionChanged = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setFocusPolicy(
            Qt.FocusPolicy.StrongFocus
        )

        self.setMouseTracking(True)

        self.left_image = None
        self.right_image = None

        # Initial state is "fit pages".
        self.zoom = 1.0

        self.pan_x = 0.0
        self.pan_y = 0.0

        self._fit_zoom = 1.0

        self._dragging = False
        self._drag_start = QPointF()
        self._drag_pan_start = (0.0, 0.0)

    # ------------------------------------------------------------------
    # Images
    # ------------------------------------------------------------------

    def set_images(self, left_image, right_image):
        self.left_image = left_image
        self.right_image = right_image

        self.update()

    # ------------------------------------------------------------------
    # Spread dimensions
    # ------------------------------------------------------------------

    def combined_size(self):
        left_w = (
            self.left_image.width()
            if self.left_image is not None
            else 0
        )

        left_h = (
            self.left_image.height()
            if self.left_image is not None
            else 0
        )

        right_w = (
            self.right_image.width()
            if self.right_image is not None
            else 0
        )

        right_h = (
            self.right_image.height()
            if self.right_image is not None
            else 0
        )

        return (
            left_w + right_w,
            max(left_h, right_h),
        )

    # ------------------------------------------------------------------
    # Fit pages
    # ------------------------------------------------------------------

    def calculate_fit_zoom(self):
        width, height = self.combined_size()

        if width <= 0 or height <= 0:
            return 1.0

        # Small margin around the spread.
        available_width = max(
            1,
            self.width() - 12,
        )

        available_height = max(
            1,
            self.height() - 12,
        )

        zoom_x = available_width / width
        zoom_y = available_height / height

        return min(
            zoom_x,
            zoom_y,
        )

    def fit_pages(self):
        """
        Fit the complete currently loaded spread into the viewport.

        The entire left and right pages are visible simultaneously.
        """

        width, height = self.combined_size()

        if width <= 0 or height <= 0:
            return False

        self._fit_zoom = self.calculate_fit_zoom()

        self.zoom = max(
            ZOOM_MIN,
            min(
                ZOOM_MAX,
                self._fit_zoom,
            ),
        )

        self.pan_x = 0.0
        self.pan_y = 0.0

        self.zoomChanged.emit(self.zoom)
        self.positionChanged.emit(
            self.pan_x,
            self.pan_y,
        )

        self.update()

        return True

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event):
        painter = QPainter(self)

        try:
            painter.fillRect(
                self.rect(),
                Qt.GlobalColor.black,
            )

            left = self.left_image
            right = self.right_image

            if left is None and right is None:
                return

            painter.setRenderHint(
                QPainter.RenderHint.SmoothPixmapTransform,
                self.zoom != 1.0,
            )

            left_w = (
                left.width()
                if left is not None
                else 0
            )

            left_h = (
                left.height()
                if left is not None
                else 0
            )

            right_w = (
                right.width()
                if right is not None
                else 0
            )

            right_h = (
                right.height()
                if right is not None
                else 0
            )

            combined_w = left_w + right_w
            combined_h = max(
                left_h,
                right_h,
            )

            scaled_w = combined_w * self.zoom
            scaled_h = combined_h * self.zoom

            origin_x = (
                (self.width() - scaled_w) / 2.0
                + self.pan_x
            )

            origin_y = (
                (self.height() - scaled_h) / 2.0
                + self.pan_y
            )

            # ----------------------------------------------------------
            # Left page
            # ----------------------------------------------------------

            if left is not None:
                y = (
                    combined_h - left_h
                ) / 2.0

                x = 0.0

                if self.zoom == 1.0:
                    # At 1:1, draw directly to preserve the source pixels.
                    painter.drawImage(
                        int(round(origin_x + x)),
                        int(round(origin_y + y)),
                        left,
                    )
                else:
                    target = QRectF(
                        origin_x + x * self.zoom,
                        origin_y + y * self.zoom,
                        left_w * self.zoom,
                        left_h * self.zoom,
                    )

                    painter.drawImage(
                        target,
                        left,
                    )

            # ----------------------------------------------------------
            # Right page
            # ----------------------------------------------------------

            if right is not None:
                # This is the important gutter coordinate:
                #
                # left page occupies [0, left_w]
                # right page occupies [left_w, left_w + right_w]
                #
                # Therefore their edges meet exactly.
                x = float(left_w)

                y = (
                    combined_h - right_h
                ) / 2.0

                if self.zoom == 1.0:
                    painter.drawImage(
                        int(round(origin_x + x)),
                        int(round(origin_y + y)),
                        right,
                    )
                else:
                    # IMPORTANT:
                    # Use QRectF, not a Python tuple.
                    #
                    # PySide6 does not accept:
                    #     painter.drawImage((x, y, w, h), image)
                    #
                    # but it does accept QRectF.
                    target = QRectF(
                        origin_x + x * self.zoom,
                        origin_y + y * self.zoom,
                        right_w * self.zoom,
                        right_h * self.zoom,
                    )

                    painter.drawImage(
                        target,
                        right,
                    )

        finally:
            # Explicitly terminate the painter before returning from
            # paintEvent. This also makes failures much less likely to
            # leave QBackingStore in a painted state.
            painter.end()

    # ------------------------------------------------------------------
    # Zoom
    # ------------------------------------------------------------------

    def zoom_in(self):
        self.set_zoom(
            self.zoom * ZOOM_STEP
        )

    def zoom_out(self):
        self.set_zoom(
            self.zoom / ZOOM_STEP
        )

    def set_zoom(self, value, anchor=None):
        value = max(
            ZOOM_MIN,
            min(
                ZOOM_MAX,
                value,
            ),
        )

        if abs(value - self.zoom) < 0.000001:
            return

        if anchor is None:
            anchor = QPointF(
                self.width() / 2.0,
                self.height() / 2.0,
            )

        old_zoom = self.zoom

        if old_zoom > 0:
            factor = value / old_zoom

            center_x = (
                self.width() / 2.0
                + self.pan_x
            )

            center_y = (
                self.height() / 2.0
                + self.pan_y
            )

            self.pan_x = (
                anchor.x()
                - (anchor.x() - center_x) * factor
                - self.width() / 2.0
            )

            self.pan_y = (
                anchor.y()
                - (anchor.y() - center_y) * factor
                - self.height() / 2.0
            )

        self.zoom = value

        self.zoomChanged.emit(self.zoom)

        self.positionChanged.emit(
            self.pan_x,
            self.pan_y,
        )

        self.update()

    def reset_zoom(self):
        self.fit_pages()

    # ------------------------------------------------------------------
    # Mouse drag / pan
    # ------------------------------------------------------------------

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True

            self._drag_start = event.position()

            self._drag_pan_start = (
                self.pan_x,
                self.pan_y,
            )

            self.setCursor(
                Qt.CursorShape.ClosedHandCursor
            )

            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging:
            delta = (
                event.position()
                - self._drag_start
            )

            self.pan_x = (
                self._drag_pan_start[0]
                + delta.x()
            )

            self.pan_y = (
                self._drag_pan_start[1]
                + delta.y()
            )

            self.positionChanged.emit(
                self.pan_x,
                self.pan_y,
            )

            self.update()

            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False

            self.setCursor(
                Qt.CursorShape.ArrowCursor
            )

            event.accept()
            return

        super().mouseReleaseEvent(event)

    # ------------------------------------------------------------------
    # Resize
    # ------------------------------------------------------------------

    def resizeEvent(self, event):
        super().resizeEvent(event)

        self.update()

# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class BookViewer(QMainWindow):
    def __init__(self, image_paths):
        super().__init__()

        self.image_paths = image_paths
        self.total_pages = len(image_paths)

        self.page_time = DEFAULT_PAGE_TIME
        self.preload_spreads = DEFAULT_PRELOAD_SPREADS
        self.cache_images = DEFAULT_CACHE_IMAGES

        self.cache = ImageCache(
            self.cache_images
        )

        self.thread_pool = QThreadPool.globalInstance()

        # Keep this modest for HDDs. More simultaneous disk readers can
        # actually make a spinning disk slower because of seek contention.
        self.thread_pool.setMaxThreadCount(
            min(
                4,
                max(
                    2,
                    self.thread_pool.maxThreadCount(),
                ),
            )
        )

        self.loading = set()

        self.spreads = self._build_spreads()

        self.current_spread = 0

        self.playing = False

        self._fullscreen = False

        # True until the first successfully loaded spread has been fitted.
        self._initial_fit_pending = True

        self._chrome_hide_timer = QTimer(self)
        self._chrome_hide_timer.setSingleShot(True)

        self._chrome_hide_timer.timeout.connect(
            self._hide_fullscreen_chrome
        )

        self.canvas = BookCanvas(self)

        self.setCentralWidget(
            self.canvas
        )

        self._create_menus()
        self._create_status_bar()

        self._update_window_title()

        self._load_current_spread()
        self._preload_around_current()

        self.setMouseTracking(True)

        # Make sure the canvas gets focus so keyboard shortcuts work.
        self.canvas.setFocus()

    # ------------------------------------------------------------------
    # Spread construction
    # ------------------------------------------------------------------

    def _build_spreads(self):
        """
        Page arrangement:

            [1]
          [2][3]
          [4][5]
          [6][7]
          ...

        The first page is alone on the right.

        If the book has an even number of pages, the last page is a
        standalone left page.
        """

        spreads = []

        if not self.image_paths:
            return spreads

        # Front cover / first page.
        spreads.append(
            {
                "left": None,
                "right": self.image_paths[0],
            }
        )

        index = 1

        while index < len(self.image_paths):
            left = self.image_paths[index]

            right = None

            if index + 1 < len(self.image_paths):
                right = self.image_paths[index + 1]

            spreads.append(
                {
                    "left": left,
                    "right": right,
                }
            )

            index += 2

        return spreads

    # ------------------------------------------------------------------
    # Menu bar
    # ------------------------------------------------------------------

    def _create_menus(self):
        menu_bar = QMenuBar(self)

        # --------------------------------------------------------------
        # File
        # --------------------------------------------------------------

        file_menu = QMenu(
            "&File",
            self,
        )

        settings_action = QAction(
            "&Settings...",
            self,
        )

        settings_action.triggered.connect(
            self.show_settings
        )

        file_menu.addAction(
            settings_action
        )

        file_menu.addSeparator()

        quit_action = QAction(
            "&Quit",
            self,
        )

        quit_action.setShortcut(
            QKeySequence("Q")
        )

        quit_action.triggered.connect(
            self.close
        )

        file_menu.addAction(
            quit_action
        )

        menu_bar.addMenu(
            file_menu
        )

        # --------------------------------------------------------------
        # View
        # --------------------------------------------------------------

        view_menu = QMenu(
            "&View",
            self,
        )

        fit_action = QAction(
            "Fit pages",
            self,
        )

        fit_action.setShortcut(
            QKeySequence("Ctrl+0")
        )

        fit_action.triggered.connect(
            self.canvas.fit_pages
        )

        view_menu.addAction(
            fit_action
        )

        zoom_in_action = QAction(
            "Zoom in",
            self,
        )

        zoom_in_action.setShortcut(
            QKeySequence("Up")
        )

        zoom_in_action.triggered.connect(
            self.canvas.zoom_in
        )

        view_menu.addAction(
            zoom_in_action
        )

        zoom_out_action = QAction(
            "Zoom out",
            self,
        )

        zoom_out_action.setShortcut(
            QKeySequence("Down")
        )

        zoom_out_action.triggered.connect(
            self.canvas.zoom_out
        )

        view_menu.addAction(
            zoom_out_action
        )

        view_menu.addSeparator()

        fullscreen_action = QAction(
            "Fullscreen",
            self,
        )

        fullscreen_action.setShortcut(
            QKeySequence("F")
        )

        fullscreen_action.triggered.connect(
            self.toggle_fullscreen
        )

        view_menu.addAction(
            fullscreen_action
        )

        menu_bar.addMenu(
            view_menu
        )

        # --------------------------------------------------------------
        # Playback
        # --------------------------------------------------------------

        playback_menu = QMenu(
            "&Playback",
            self,
        )

        play_action = QAction(
            "Play / Pause",
            self,
        )

        play_action.setShortcut(
            QKeySequence("Space")
        )

        play_action.triggered.connect(
            self.toggle_play
        )

        playback_menu.addAction(
            play_action
        )

        previous_action = QAction(
            "Previous page",
            self,
        )

        previous_action.setShortcut(
            QKeySequence("Left")
        )

        previous_action.triggered.connect(
            self.previous_spread
        )

        playback_menu.addAction(
            previous_action
        )

        next_action = QAction(
            "Next page",
            self,
        )

        next_action.setShortcut(
            QKeySequence("Right")
        )

        next_action.triggered.connect(
            self.next_spread
        )

        playback_menu.addAction(
            next_action
        )

        menu_bar.addMenu(
            playback_menu
        )

        # --------------------------------------------------------------
        # Page navigation widget
        # --------------------------------------------------------------

        navigation = QWidget(self)

        nav_layout = QHBoxLayout(
            navigation
        )

        nav_layout.setContentsMargins(
            4,
            0,
            4,
            0,
        )

        nav_layout.setSpacing(2)

        self.previous_button = QPushButton(
            "‹",
            navigation,
        )

        self.previous_button.setToolTip(
            "Previous page"
        )

        self.previous_button.setFixedWidth(
            28
        )

        self.previous_button.clicked.connect(
            self.previous_spread
        )

        self.page_input = QLineEdit(
            navigation
        )

        self.page_input.setFixedWidth(
            65
        )

        self.page_input.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.page_input.setToolTip(
            "Enter a page number and press Enter"
        )

        self.page_input.returnPressed.connect(
            self._page_input_activated
        )

        self.next_button = QPushButton(
            "›",
            navigation,
        )

        self.next_button.setToolTip(
            "Next page"
        )

        self.next_button.setFixedWidth(
            28
        )

        self.next_button.clicked.connect(
            self.next_spread
        )

        nav_layout.addWidget(
            self.previous_button
        )

        nav_layout.addWidget(
            self.page_input
        )

        nav_layout.addWidget(
            self.next_button
        )

        menu_bar.setCornerWidget(
            navigation,
            Qt.Corner.TopRightCorner,
        )

        self.setMenuBar(
            menu_bar
        )

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------

    def _create_status_bar(self):
        status = QStatusBar(self)

        self.status_label = QLabel()

        status.addPermanentWidget(
            self.status_label
        )

        self.setStatusBar(
            status
        )

        self._update_status()

    # ------------------------------------------------------------------
    # Current page number
    # ------------------------------------------------------------------

    def _current_visible_page(self):
        if not self.spreads:
            return 0

        spread = self.spreads[
            self.current_spread
        ]

        # Requirement:
        # left page number if there is a left page,
        # otherwise right page number.
        path = (
            spread["left"]
            if spread["left"] is not None
            else spread["right"]
        )

        if path is None:
            return 0

        try:
            return get_page_num(path)
        except Exception:
            return self.current_spread + 1

    def _update_page_input(self):
        page = self._current_visible_page()

        self.page_input.blockSignals(
            True
        )

        self.page_input.setText(
            str(page)
            if page
            else ""
        )

        self.page_input.blockSignals(
            False
        )

    # ------------------------------------------------------------------
    # Page input
    # ------------------------------------------------------------------

    def _page_input_activated(self):
        text = self.page_input.text().strip()

        try:
            page = int(text)
        except ValueError:
            self._update_page_input()
            return

        self.go_to_page(page)

        self.canvas.setFocus()

    def go_to_page(self, page_number):
        if not self.spreads:
            return

        page_number = max(
            1,
            min(
                self.total_pages,
                page_number,
            ),
        )

        # Find the spread containing the requested page.
        for index, spread in enumerate(
            self.spreads
        ):
            for key in ("left", "right"):
                path = spread[key]

                if path is None:
                    continue

                try:
                    if (
                        get_page_num(path)
                        == page_number
                    ):
                        self._set_spread(index)
                        return
                except Exception:
                    pass

        # Fallback for unusual page-number naming.
        nearest = 0
        best_distance = float("inf")

        for index, spread in enumerate(
            self.spreads
        ):
            path = (
                spread["left"]
                if spread["left"] is not None
                else spread["right"]
            )

            try:
                number = get_page_num(path)
            except Exception:
                continue

            distance = abs(
                number - page_number
            )

            if distance < best_distance:
                best_distance = distance
                nearest = index

        self._set_spread(nearest)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _set_spread(self, index):
        if not self.spreads:
            return

        index = max(
            0,
            min(
                len(self.spreads) - 1,
                index,
            ),
        )

        if index == self.current_spread:
            self._load_current_spread()
            return

        self.current_spread = index

        self._load_current_spread()
        self._preload_around_current()

        self._update_page_input()
        self._update_status()
        self._update_window_title()

    def previous_spread(self):
        if not self.spreads:
            return

        self._set_spread(
            self.current_spread - 1
        )

    def next_spread(self):
        if not self.spreads:
            return

        self._set_spread(
            self.current_spread + 1
        )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _request_image(self, path):
        if path is None:
            return

        path = str(path)

        if self.cache.contains(path):
            return

        if path in self.loading:
            return

        self.loading.add(path)

        task = ImageLoadTask(
            path,
            self.cache,
        )

        task.signals.loaded.connect(
            self._image_loaded
        )

        task.signals.failed.connect(
            self._image_failed
        )

        self.thread_pool.start(task)

    def _image_loaded(self, path):
        self.loading.discard(path)

        self._load_current_spread()

        self._preload_around_current()

    def _image_failed(self, path, error):
        self.loading.discard(path)

        if not self.spreads:
            return

        spread = self.spreads[
            self.current_spread
        ]

        if (
            path == spread["left"]
            or path == spread["right"]
        ):
            self.status_label.setText(
                f"Could not load "
                f"{Path(path).name}: "
                f"{error}"
            )

    def _load_current_spread(self):
        if not self.spreads:
            self.canvas.set_images(
                None,
                None,
            )
            return

        spread = self.spreads[
            self.current_spread
        ]

        left = None
        right = None

        if spread["left"] is not None:
            left = self.cache.get(
                spread["left"]
            )

            if left is None:
                self._request_image(
                    spread["left"]
                )

        if spread["right"] is not None:
            right = self.cache.get(
                spread["right"]
            )

            if right is None:
                self._request_image(
                    spread["right"]
                )

        self.canvas.set_images(
            left,
            right,
        )

        # --------------------------------------------------------------
        # FIX FOR INITIAL FIT
        #
        # The images are loaded asynchronously. The first time
        # _load_current_spread() was called, they weren't available yet,
        # so calculating fit_zoom at that point was impossible.
        #
        # Once both currently visible images are available, perform the
        # initial fit exactly once.
        # --------------------------------------------------------------

        if self._initial_fit_pending:
            left_ready = (
                spread["left"] is None
                or left is not None
            )

            right_ready = (
                spread["right"] is None
                or right is not None
            )

            if left_ready and right_ready:
                if self.canvas.fit_pages():
                    self._initial_fit_pending = False

        self._update_status()

    # ------------------------------------------------------------------
    # Preloading
    # ------------------------------------------------------------------

    def _preload_around_current(self):
        if not self.spreads:
            return

        start = self.current_spread

        end = min(
            len(self.spreads),
            start + 1 + self.preload_spreads,
        )

        # Queue future pages in playback order.
        for spread_index in range(
            start,
            end,
        ):
            spread = self.spreads[
                spread_index
            ]

            if spread["left"] is not None:
                self._request_image(
                    spread["left"]
                )

            if spread["right"] is not None:
                self._request_image(
                    spread["right"]
                )

    def _current_spread_ready(self):
        if not self.spreads:
            return True

        spread = self.spreads[
            self.current_spread
        ]

        if (
            spread["left"] is not None
            and not self.cache.contains(
                spread["left"]
            )
        ):
            return False

        if (
            spread["right"] is not None
            and not self.cache.contains(
                spread["right"]
            )
        ):
            return False

        return True

    def _next_spread_ready(self):
        if (
            self.current_spread + 1
            >= len(self.spreads)
        ):
            return True

        spread = self.spreads[
            self.current_spread + 1
        ]

        if (
            spread["left"] is not None
            and not self.cache.contains(
                spread["left"]
            )
        ):
            return False

        if (
            spread["right"] is not None
            and not self.cache.contains(
                spread["right"]
            )
        ):
            return False

        return True

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def toggle_play(self):
        if not self.spreads:
            return

        self.playing = not self.playing

        if self.playing:
            self._start_playback()
        else:
            self._stop_playback()

        self._update_status()

    def _start_playback(self):
        self._preload_around_current()

        self._wait_for_playback_buffer()

    def _stop_playback(self):
        self._update_status()

    def _wait_for_playback_buffer(self):
        if not self.playing:
            return

        if (
            self.current_spread
            >= len(self.spreads) - 1
        ):
            self.playing = False
            self._update_status()
            return

        if not self._current_spread_ready():
            self.status_label.setText(
                "Buffering…"
            )

            QTimer.singleShot(
                30,
                self._wait_for_playback_buffer,
            )

            return

        if not self._next_spread_ready():
            self.status_label.setText(
                "Buffering next page…"
            )

            QTimer.singleShot(
                30,
                self._wait_for_playback_buffer,
            )

            return

        self._schedule_next_playback()

    def _schedule_next_playback(self):
        if not self.playing:
            return

        delay = max(
            1,
            int(
                self.page_time * 1000
            ),
        )

        QTimer.singleShot(
            delay,
            self._advance_playback,
        )

    def _advance_playback(self):
        if not self.playing:
            return

        if (
            self.current_spread
            >= len(self.spreads) - 1
        ):
            self.playing = False
            self._update_status()
            return

        self.next_spread()

        self._preload_around_current()

        if self._next_spread_ready():
            self._schedule_next_playback()
        else:
            self._wait_for_playback_buffer()

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def show_settings(self):
        dialog = SettingsDialog(
            self,
            self.page_time,
            self.preload_spreads,
            self.cache_images,
        )

        if (
            dialog.exec()
            != QDialog.DialogCode.Accepted
        ):
            return

        self.page_time = (
            dialog.page_time.value()
        )

        self.preload_spreads = (
            dialog.preload_spreads.value()
        )

        self.cache_images = (
            dialog.cache_images.value()
        )

        self.cache.set_max_images(
            self.cache_images
        )

        self._preload_around_current()
        self._update_status()

    # ------------------------------------------------------------------
    # Fullscreen
    # ------------------------------------------------------------------

    def toggle_fullscreen(self):
        if self._fullscreen:
            self._leave_fullscreen()
        else:
            self._enter_fullscreen()

    def _enter_fullscreen(self):
        self._fullscreen = True

        # Hide all UI chrome initially.
        self.menuBar().hide()
        self.statusBar().hide()

        self.showFullScreen()

        self._chrome_hide_timer.stop()

        self.canvas.setFocus()

    def _leave_fullscreen(self):
        self._fullscreen = False

        self._chrome_hide_timer.stop()

        self.showNormal()

        self.menuBar().show()
        self.statusBar().show()

        self.canvas.setFocus()

    def _show_fullscreen_chrome(self):
        if not self._fullscreen:
            return

        self.menuBar().show()
        self.statusBar().show()

        self._chrome_hide_timer.start(
            FULLSCREEN_CHROME_HIDE_DELAY
        )

    def _hide_fullscreen_chrome(self):
        if not self._fullscreen:
            return

        self.menuBar().hide()
        self.statusBar().hide()

    # ------------------------------------------------------------------
    # Mouse movement
    # ------------------------------------------------------------------

    def mouseMoveEvent(self, event):
        if self._fullscreen:
            pos = event.position()

            if (
                pos.y() <= 8
                or pos.y() >= self.height() - 8
            ):
                self._show_fullscreen_chrome()

        super().mouseMoveEvent(event)

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()

        if key == Qt.Key.Key_Q:
            self.close()
            return

        if key == Qt.Key.Key_F:
            self.toggle_fullscreen()
            return

        if key == Qt.Key.Key_Up:
            self.canvas.zoom_in()
            return

        if key == Qt.Key.Key_Down:
            self.canvas.zoom_out()
            return

        if key == Qt.Key.Key_Left:
            self.previous_spread()
            return

        if key == Qt.Key.Key_Right:
            self.next_spread()
            return

        if key == Qt.Key.Key_Space:
            self.toggle_play()
            return

        if (
            key in (
                Qt.Key.Key_Plus,
                Qt.Key.Key_Equal,
            )
            and modifiers
            & Qt.KeyboardModifier.ControlModifier
        ):
            self.canvas.zoom_in()
            return

        if (
            key == Qt.Key.Key_Minus
            and modifiers
            & Qt.KeyboardModifier.ControlModifier
        ):
            self.canvas.zoom_out()
            return

        if (
            key == Qt.Key.Key_0
            and modifiers
            & Qt.KeyboardModifier.ControlModifier
        ):
            self.canvas.fit_pages()
            return

        super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # Mouse wheel = page navigation
    # ------------------------------------------------------------------

    def wheelEvent(self, event):
        delta = event.angleDelta().y()

        # Natural top-down reading direction:
        #
        #   wheel UP   -> previous page
        #   wheel DOWN -> next page
        #
        if delta > 0:
            self.previous_spread()
            event.accept()
            return

        if delta < 0:
            self.next_spread()
            event.accept()
            return

        super().wheelEvent(event)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def _update_status(self):
        if not hasattr(
            self,
            "status_label",
        ):
            return

        if not self.spreads:
            self.status_label.setText(
                "No pages"
            )
            return

        page = self._current_visible_page()

        state = (
            "Playing"
            if self.playing
            else "Paused"
        )

        self.status_label.setText(
            f"{state}  •  "
            f"Page {page}  •  "
            f"Spread "
            f"{self.current_spread + 1}/"
            f"{len(self.spreads)}  •  "
            f"Zoom "
            f"{self.canvas.zoom:.2f}×  •  "
            f"Cache "
            f"{self.cache.size()}/"
            f"{self.cache_images}"
        )

        self._update_page_input()

    def _update_window_title(self):
        page = self._current_visible_page()

        if page:
            self.setWindowTitle(
                f"Scanned Book Viewer — "
                f"Page {page}"
            )
        else:
            self.setWindowTitle(
                "Scanned Book Viewer"
            )

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        self.playing = False

        self.thread_pool.clear()

        super().closeEvent(event)

# ---------------------------------------------------------------------------
# Find images
# ---------------------------------------------------------------------------

def find_images(source_directory):
    source_directory = Path(
        source_directory
    )

    if not source_directory.is_dir():
        raise ValueError(
            f"Not a directory: "
            f"{source_directory}"
        )

    scan_format = str(
        config.scan_format
    ).lower().lstrip(".")

    supported = {
        scan_format,
    }

    if scan_format == "tif":
        supported.add("tiff")
    elif scan_format == "tiff":
        supported.add("tif")

    images = []

    # Deliberately non-recursive.
    for path in source_directory.iterdir():
        if not path.is_file():
            continue

        extension = (
            path.suffix
            .lower()
            .lstrip(".")
        )

        if extension not in supported:
            continue

        images.append(path)

    images.sort(
        key=lambda path: (
            get_page_num(path),
            path.name.lower(),
        )
    )

    return images

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # __name__ is "__main__" when this file is executed directly.
    #
    # __file__ is the actual script filename, so command-line usage and
    # error messages show the real filename rather than a hard-coded
    # "book_viewer.py".
    script_name = Path(__file__).name

    parser = argparse.ArgumentParser(
        prog=script_name,
        description=(
            "Specialized scanned-book "
            "page viewer."
        ),
    )

    parser.add_argument(
        "source_directory",
        metavar="SOURCE_DIRECTORY",
        help=(
            "Directory containing the "
            "scanned page images."
        ),
    )

    args = parser.parse_args()

    try:
        image_paths = find_images(
            args.source_directory
        )

    except Exception as exc:
        print(
            f"{script_name}: error: {exc}",
            file=sys.stderr,
        )

        return 2

    if not image_paths:
        print(
            f"{script_name}: no "
            f"{config.scan_format} images "
            f"found in "
            f"{args.source_directory}",
            file=sys.stderr,
        )

        return 1

    app = QApplication(
        sys.argv
    )

    app.setApplicationName(
        "Scanned Book Viewer"
    )

    app.setOrganizationName(
        "Scanned Book Viewer"
    )

    window = BookViewer(
        image_paths
    )

    window.showMaximized()
    window.canvas.setFocus()

    return app.exec()

if __name__ == "__main__":
    raise SystemExit(
        main()
    )
```

The important fix for the crash is the `QRectF(...)` around the scaled destination rectangles. The initial fit is now also triggered **when the first spread's images have actually been decoded**, rather than during window construction when the canvas has no image dimensions yet.

## Prompt:

9/9/2026, 6:28:45 PM

the caching of images works only when i seek forward, but it fails when i seek backward. ideally the caching of images should work in both directions

the app should start in fullscreen mode

when i leave fullscreen mode, the app should have maximum window size. currently, when i leave fullscreen mode, the window is extremely small

Ctrl+mousewheel should control the zoom level:  
Ctrl + scroll up = zoom in  
Ctrl + scroll down = zoom out

the app should have a "progress bar" to visualize the position in the book. this progress bar should also be visible in fullscreen mode, as a thin horizontal strip on the bottom of the screen, maybe 2 to 5 pixels high. the background of the progress bar should be white. the current position of the progress bar should be visualized with a solid black rectangle, so in autoplay mode, the solid black rectangle "travels" from the left side of the screen to the right side of the screen

## Response:

9/9/2026, 6:28:49 PM

Yes. The main caching problem is that the preload queue was inherently **forward-only**. I’ve changed it to maintain a symmetric buffer around the current spread, so seeking in either direction benefits from preloaded images.

I’ve also made the other requested changes:

- Starts **fullscreen**.
- Leaving fullscreen returns to **maximized window size**.
- **Ctrl + mouse wheel** zooms.
- Plain mouse wheel seeks:
  - ↑ = previous spread
  - ↓ = next spread
- Added a **2–5 px progress strip** at the bottom.
  - White background.
  - Solid black position indicator.
  - Remains visible in fullscreen even though the menu/status bar disappear.
  - Updates continuously as autoplay advances.

\*\*document: \*\*

``` python
#!/usr/bin/env python3

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

from PySide6.QtCore import (
    QObject,
    QPointF,
    QRectF,
    QRunnable,
    QMutex,
    QMutexLocker,
    Qt,
    QThreadPool,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QAction,
    QImage,
    QImageReader,
    QKeySequence,
    QPainter,
)
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMenuBar,
    QPushButton,
    QSpinBox,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from _shared import (
    load_config,
    get_page_num,
)

config = load_config()

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_PAGE_TIME = 0.2

# Number of spreads to keep preloaded in BOTH directions.
DEFAULT_PRELOAD_SPREADS = 20

DEFAULT_CACHE_IMAGES = 60

FULLSCREEN_CHROME_HIDE_DELAY = 1800

ZOOM_MIN = 0.05
ZOOM_MAX = 8.0
ZOOM_STEP = 1.20

# Height of the book-position progress bar.
PROGRESS_BAR_HEIGHT = 4

# ---------------------------------------------------------------------------
# Image cache
# ---------------------------------------------------------------------------

class ImageCache:
    """
    Thread-safe LRU cache of decoded QImages.
    """

    def __init__(self, max_images=DEFAULT_CACHE_IMAGES):
        self.max_images = max(1, max_images)

        self._mutex = QMutex()
        self._images = OrderedDict()

    def set_max_images(self, value):
        with QMutexLocker(self._mutex):
            self.max_images = max(1, value)

            while len(self._images) > self.max_images:
                self._images.popitem(last=False)

    def get(self, path):
        path = str(path)

        with QMutexLocker(self._mutex):
            image = self._images.get(path)

            if image is None:
                return None

            self._images.move_to_end(path)

            return image

    def contains(self, path):
        path = str(path)

        with QMutexLocker(self._mutex):
            return path in self._images

    def put(self, path, image):
        path = str(path)

        with QMutexLocker(self._mutex):
            self._images[path] = image
            self._images.move_to_end(path)

            while len(self._images) > self.max_images:
                self._images.popitem(last=False)

    def clear(self):
        with QMutexLocker(self._mutex):
            self._images.clear()

    def size(self):
        with QMutexLocker(self._mutex):
            return len(self._images)

# ---------------------------------------------------------------------------
# Background image loading
# ---------------------------------------------------------------------------

class ImageLoadSignals(QObject):
    loaded = Signal(str)
    failed = Signal(str, str)

class ImageLoadTask(QRunnable):
    def __init__(self, path, cache):
        super().__init__()

        self.path = str(path)
        self.cache = cache

        self.signals = ImageLoadSignals()

        self.setAutoDelete(True)

    def run(self):
        try:
            if self.cache.contains(self.path):
                self.signals.loaded.emit(self.path)
                return

            reader = QImageReader(self.path)
            reader.setAutoTransform(True)

            image = reader.read()

            if image.isNull():
                raise RuntimeError(
                    reader.errorString()
                    or "Could not decode image"
                )

            # Force decoded image data to be detached from the reader.
            image = image.copy()

            self.cache.put(
                self.path,
                image,
            )

            self.signals.loaded.emit(
                self.path
            )

        except Exception as exc:
            self.signals.failed.emit(
                self.path,
                str(exc),
            )

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class SettingsDialog(QDialog):
    def __init__(
        self,
        parent=None,
        page_time=DEFAULT_PAGE_TIME,
        preload_spreads=DEFAULT_PRELOAD_SPREADS,
        cache_images=DEFAULT_CACHE_IMAGES,
    ):
        super().__init__(parent)

        self.setWindowTitle(
            "Viewer settings"
        )

        form = QFormLayout(self)

        self.page_time = QDoubleSpinBox()
        self.page_time.setRange(
            0.01,
            60.0,
        )
        self.page_time.setDecimals(2)
        self.page_time.setSingleStep(0.05)
        self.page_time.setSuffix(" s")
        self.page_time.setValue(
            page_time
        )

        self.preload_spreads = QSpinBox()
        self.preload_spreads.setRange(
            0,
            1000,
        )
        self.preload_spreads.setValue(
            preload_spreads
        )

        self.cache_images = QSpinBox()
        self.cache_images.setRange(
            2,
            5000,
        )
        self.cache_images.setValue(
            cache_images
        )

        form.addRow(
            "Time per spread:",
            self.page_time,
        )

        form.addRow(
            "Preload spreads in each direction:",
            self.preload_spreads,
        )

        form.addRow(
            "Image cache:",
            self.cache_images,
        )

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok
            | QDialogButtonBox.Cancel
        )

        buttons.accepted.connect(
            self.accept
        )

        buttons.rejected.connect(
            self.reject
        )

        form.addRow(buttons)

# ---------------------------------------------------------------------------
# Progress bar
# ---------------------------------------------------------------------------

class BookProgressBar(QWidget):
    """
    Very thin book-position indicator.

    Background:
        white

    Current position:
        solid black rectangle

    The bar is intentionally independent of the fullscreen chrome,
    so it remains visible in fullscreen.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setFixedHeight(
            PROGRESS_BAR_HEIGHT
        )

        self.position = 0.0

        self.setAttribute(
            Qt.WidgetAttribute.WA_OpaquePaintEvent
        )

    def set_position(self, position):
        self.position = max(
            0.0,
            min(
                1.0,
                float(position),
            ),
        )

        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)

        try:
            # White background.
            painter.fillRect(
                self.rect(),
                Qt.GlobalColor.white,
            )

            if self.width() <= 0:
                return

            # Solid black progress indicator.
            indicator_width = max(
                1,
                int(
                    round(
                        self.width()
                        * self.position
                    )
                ),
            )

            painter.fillRect(
                QRectF(
                    0,
                    0,
                    indicator_width,
                    self.height(),
                ),
                Qt.GlobalColor.black,
            )

        finally:
            painter.end()

# ---------------------------------------------------------------------------
# Book canvas
# ---------------------------------------------------------------------------

class BookCanvas(QWidget):
    """
    Displays the two pages as one continuous spread.

    The gutter is exactly at the boundary between the two source images.

    Mouse:
        left drag             = pan
        wheel                 = page navigation
        Ctrl + wheel          = zoom
    """

    zoomChanged = Signal(float)

    previousRequested = Signal()
    nextRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setFocusPolicy(
            Qt.FocusPolicy.StrongFocus
        )

        self.setMouseTracking(True)

        self.left_image = None
        self.right_image = None

        # Actual initial zoom is calculated after the first images load.
        self.zoom = 1.0

        self.pan_x = 0.0
        self.pan_y = 0.0

        self._fit_zoom = 1.0

        self._dragging = False
        self._drag_start = QPointF()
        self._drag_pan_start = (
            0.0,
            0.0,
        )

    # ------------------------------------------------------------------
    # Images
    # ------------------------------------------------------------------

    def set_images(
        self,
        left_image,
        right_image,
    ):
        self.left_image = left_image
        self.right_image = right_image

        self.update()

    # ------------------------------------------------------------------
    # Dimensions
    # ------------------------------------------------------------------

    def combined_size(self):
        left_w = (
            self.left_image.width()
            if self.left_image is not None
            else 0
        )

        left_h = (
            self.left_image.height()
            if self.left_image is not None
            else 0
        )

        right_w = (
            self.right_image.width()
            if self.right_image is not None
            else 0
        )

        right_h = (
            self.right_image.height()
            if self.right_image is not None
            else 0
        )

        return (
            left_w + right_w,
            max(
                left_h,
                right_h,
            ),
        )

    # ------------------------------------------------------------------
    # Fit pages
    # ------------------------------------------------------------------

    def calculate_fit_zoom(self):
        width, height = (
            self.combined_size()
        )

        if width <= 0 or height <= 0:
            return 1.0

        available_width = max(
            1,
            self.width() - 12,
        )

        available_height = max(
            1,
            self.height() - 12,
        )

        return min(
            available_width / width,
            available_height / height,
        )

    def fit_pages(self):
        """
        Fit the complete spread into the viewport.
        """

        width, height = (
            self.combined_size()
        )

        if width <= 0 or height <= 0:
            return False

        self._fit_zoom = (
            self.calculate_fit_zoom()
        )

        self.zoom = max(
            ZOOM_MIN,
            min(
                ZOOM_MAX,
                self._fit_zoom,
            ),
        )

        self.pan_x = 0.0
        self.pan_y = 0.0

        self.zoomChanged.emit(
            self.zoom
        )

        self.update()

        return True

    # ------------------------------------------------------------------
    # Paint
    # ------------------------------------------------------------------

    def paintEvent(self, event):
        painter = QPainter(self)

        try:
            painter.fillRect(
                self.rect(),
                Qt.GlobalColor.black,
            )

            left = self.left_image
            right = self.right_image

            if left is None and right is None:
                return

            painter.setRenderHint(
                QPainter.RenderHint.SmoothPixmapTransform,
                self.zoom != 1.0,
            )

            left_w = (
                left.width()
                if left is not None
                else 0
            )

            left_h = (
                left.height()
                if left is not None
                else 0
            )

            right_w = (
                right.width()
                if right is not None
                else 0
            )

            right_h = (
                right.height()
                if right is not None
                else 0
            )

            combined_w = (
                left_w + right_w
            )

            combined_h = max(
                left_h,
                right_h,
            )

            scaled_w = (
                combined_w
                * self.zoom
            )

            scaled_h = (
                combined_h
                * self.zoom
            )

            origin_x = (
                (
                    self.width()
                    - scaled_w
                )
                / 2.0
                + self.pan_x
            )

            origin_y = (
                (
                    self.height()
                    - scaled_h
                )
                / 2.0
                + self.pan_y
            )

            # ----------------------------------------------------------
            # Left page
            # ----------------------------------------------------------

            if left is not None:
                y = (
                    combined_h
                    - left_h
                ) / 2.0

                if self.zoom == 1.0:
                    painter.drawImage(
                        int(
                            round(
                                origin_x
                            )
                        ),
                        int(
                            round(
                                origin_y + y
                            )
                        ),
                        left,
                    )

                else:
                    target = QRectF(
                        origin_x,
                        origin_y
                        + y * self.zoom,
                        left_w
                        * self.zoom,
                        left_h
                        * self.zoom,
                    )

                    painter.drawImage(
                        target,
                        left,
                    )

            # ----------------------------------------------------------
            # Right page
            # ----------------------------------------------------------

            if right is not None:
                # The right page starts exactly at the right edge of
                # the left page.
                x = float(left_w)

                y = (
                    combined_h
                    - right_h
                ) / 2.0

                if self.zoom == 1.0:
                    painter.drawImage(
                        int(
                            round(
                                origin_x + x
                            )
                        ),
                        int(
                            round(
                                origin_y + y
                            )
                        ),
                        right,
                    )

                else:
                    target = QRectF(
                        origin_x
                        + x * self.zoom,
                        origin_y
                        + y * self.zoom,
                        right_w
                        * self.zoom,
                        right_h
                        * self.zoom,
                    )

                    painter.drawImage(
                        target,
                        right,
                    )

        finally:
            painter.end()

    # ------------------------------------------------------------------
    # Zoom
    # ------------------------------------------------------------------

    def zoom_in(self):
        self.set_zoom(
            self.zoom
            * ZOOM_STEP
        )

    def zoom_out(self):
        self.set_zoom(
            self.zoom
            / ZOOM_STEP
        )

    def set_zoom(
        self,
        value,
        anchor=None,
    ):
        value = max(
            ZOOM_MIN,
            min(
                ZOOM_MAX,
                value,
            ),
        )

        if (
            abs(
                value - self.zoom
            )
            < 0.000001
        ):
            return

        if anchor is None:
            anchor = QPointF(
                self.width()
                / 2.0,
                self.height()
                / 2.0,
            )

        old_zoom = self.zoom

        if old_zoom > 0:
            factor = (
                value / old_zoom
            )

            center_x = (
                self.width()
                / 2.0
                + self.pan_x
            )

            center_y = (
                self.height()
                / 2.0
                + self.pan_y
            )

            self.pan_x = (
                anchor.x()
                - (
                    anchor.x()
                    - center_x
                )
                * factor
                - self.width()
                / 2.0
            )

            self.pan_y = (
                anchor.y()
                - (
                    anchor.y()
                    - center_y
                )
                * factor
                - self.height()
                / 2.0
            )

        self.zoom = value

        self.zoomChanged.emit(
            self.zoom
        )

        self.update()

    # ------------------------------------------------------------------
    # Mouse pan
    # ------------------------------------------------------------------

    def mousePressEvent(self, event):
        if (
            event.button()
            == Qt.MouseButton.LeftButton
        ):
            self._dragging = True

            self._drag_start = (
                event.position()
            )

            self._drag_pan_start = (
                self.pan_x,
                self.pan_y,
            )

            self.setCursor(
                Qt.CursorShape.ClosedHandCursor
            )

            event.accept()
            return

        super().mousePressEvent(
            event
        )

    def mouseMoveEvent(self, event):
        if self._dragging:
            delta = (
                event.position()
                - self._drag_start
            )

            self.pan_x = (
                self._drag_pan_start[0]
                + delta.x()
            )

            self.pan_y = (
                self._drag_pan_start[1]
                + delta.y()
            )

            self.update()

            event.accept()
            return

        super().mouseMoveEvent(
            event
        )

    def mouseReleaseEvent(self, event):
        if (
            event.button()
            == Qt.MouseButton.LeftButton
        ):
            self._dragging = False

            self.setCursor(
                Qt.CursorShape.ArrowCursor
            )

            event.accept()
            return

        super().mouseReleaseEvent(
            event
        )

    # ------------------------------------------------------------------
    # Mouse wheel
    # ------------------------------------------------------------------

    def wheelEvent(self, event):
        delta = event.angleDelta().y()

        if delta == 0:
            event.ignore()
            return

        modifiers = (
            event.modifiers()
        )

        if (
            modifiers
            & Qt.KeyboardModifier.ControlModifier
        ):
            # Ctrl + wheel = zoom.
            if delta > 0:
                self.zoom_in()
            else:
                self.zoom_out()

            event.accept()
            return

        # Normal wheel = page navigation.
        #
        # Top-down reading direction:
        #
        #   wheel up   -> previous
        #   wheel down -> next
        if delta > 0:
            self.previousRequested.emit()
        else:
            self.nextRequested.emit()

        event.accept()

    # ------------------------------------------------------------------
    # Resize
    # ------------------------------------------------------------------

    def resizeEvent(self, event):
        super().resizeEvent(
            event
        )

        self.update()

# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class BookViewer(QMainWindow):
    def __init__(self, image_paths):
        super().__init__()

        self.image_paths = image_paths
        self.total_pages = len(
            image_paths
        )

        self.page_time = (
            DEFAULT_PAGE_TIME
        )

        self.preload_spreads = (
            DEFAULT_PRELOAD_SPREADS
        )

        self.cache_images = (
            DEFAULT_CACHE_IMAGES
        )

        self.cache = ImageCache(
            self.cache_images
        )

        self.thread_pool = (
            QThreadPool.globalInstance()
        )

        # A modest number of workers is preferable on spinning disks.
        self.thread_pool.setMaxThreadCount(
            min(
                4,
                max(
                    2,
                    self.thread_pool.maxThreadCount(),
                ),
            )
        )

        self.loading = set()

        self.spreads = (
            self._build_spreads()
        )

        self.current_spread = 0

        self.playing = False

        self._fullscreen = False

        self._initial_fit_pending = True

        # --------------------------------------------------------------
        # Central layout
        # --------------------------------------------------------------

        central = QWidget(self)

        central_layout = QVBoxLayout(
            central
        )

        central_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        central_layout.setSpacing(
            0
        )

        self.canvas = BookCanvas(
            central
        )

        self.progress_bar = (
            BookProgressBar(central)
        )

        central_layout.addWidget(
            self.canvas,
            1,
        )

        central_layout.addWidget(
            self.progress_bar,
            0,
        )

        self.setCentralWidget(
            central
        )

        self.canvas.previousRequested.connect(
            self.previous_spread
        )

        self.canvas.nextRequested.connect(
            self.next_spread
        )

        # --------------------------------------------------------------
        # Fullscreen chrome timer
        # --------------------------------------------------------------

        self._chrome_hide_timer = QTimer(
            self
        )

        self._chrome_hide_timer.setSingleShot(
            True
        )

        self._chrome_hide_timer.timeout.connect(
            self._hide_fullscreen_chrome
        )

        # --------------------------------------------------------------
        # Menus/status
        # --------------------------------------------------------------

        self._create_menus()
        self._create_status_bar()

        self._update_window_title()

        # --------------------------------------------------------------
        # Start loading
        # --------------------------------------------------------------

        self._load_current_spread()
        self._preload_around_current()

        self.setMouseTracking(True)
        self.canvas.setFocus()

        # Start fullscreen after the initial window has been created.
        QTimer.singleShot(
            0,
            self._start_fullscreen,
        )

    # ------------------------------------------------------------------
    # Spread construction
    # ------------------------------------------------------------------

    def _build_spreads(self):
        spreads = []

        if not self.image_paths:
            return spreads

        # Page 1 is alone on the right.
        spreads.append(
            {
                "left": None,
                "right": self.image_paths[0],
            }
        )

        index = 1

        while index < len(
            self.image_paths
        ):
            left = self.image_paths[
                index
            ]

            right = None

            if (
                index + 1
                < len(self.image_paths)
            ):
                right = self.image_paths[
                    index + 1
                ]

            spreads.append(
                {
                    "left": left,
                    "right": right,
                }
            )

            index += 2

        return spreads

    # ------------------------------------------------------------------
    # Menu
    # ------------------------------------------------------------------

    def _create_menus(self):
        menu_bar = QMenuBar(
            self
        )

        # --------------------------------------------------------------
        # File
        # --------------------------------------------------------------

        file_menu = QMenu(
            "&File",
            self,
        )

        settings_action = QAction(
            "&Settings...",
            self,
        )

        settings_action.triggered.connect(
            self.show_settings
        )

        file_menu.addAction(
            settings_action
        )

        file_menu.addSeparator()

        quit_action = QAction(
            "&Quit",
            self,
        )

        quit_action.setShortcut(
            QKeySequence("Q")
        )

        quit_action.triggered.connect(
            self.close
        )

        file_menu.addAction(
            quit_action
        )

        menu_bar.addMenu(
            file_menu
        )

        # --------------------------------------------------------------
        # View
        # --------------------------------------------------------------

        view_menu = QMenu(
            "&View",
            self,
        )

        fit_action = QAction(
            "Fit pages",
            self,
        )

        fit_action.setShortcut(
            QKeySequence("Ctrl+0")
        )

        fit_action.triggered.connect(
            self.canvas.fit_pages
        )

        view_menu.addAction(
            fit_action
        )

        zoom_in_action = QAction(
            "Zoom in",
            self,
        )

        zoom_in_action.setShortcut(
            QKeySequence("Up")
        )

        zoom_in_action.triggered.connect(
            self.canvas.zoom_in
        )

        view_menu.addAction(
            zoom_in_action
        )

        zoom_out_action = QAction(
            "Zoom out",
            self,
        )

        zoom_out_action.setShortcut(
            QKeySequence("Down")
        )

        zoom_out_action.triggered.connect(
            self.canvas.zoom_out
        )

        view_menu.addAction(
            zoom_out_action
        )

        view_menu.addSeparator()

        fullscreen_action = QAction(
            "Fullscreen",
            self,
        )

        fullscreen_action.setShortcut(
            QKeySequence("F")
        )

        fullscreen_action.triggered.connect(
            self.toggle_fullscreen
        )

        view_menu.addAction(
            fullscreen_action
        )

        menu_bar.addMenu(
            view_menu
        )

        # --------------------------------------------------------------
        # Playback
        # --------------------------------------------------------------

        playback_menu = QMenu(
            "&Playback",
            self,
        )

        play_action = QAction(
            "Play / Pause",
            self,
        )

        play_action.setShortcut(
            QKeySequence("Space")
        )

        play_action.triggered.connect(
            self.toggle_play
        )

        playback_menu.addAction(
            play_action
        )

        previous_action = QAction(
            "Previous page",
            self,
        )

        previous_action.setShortcut(
            QKeySequence("Left")
        )

        previous_action.triggered.connect(
            self.previous_spread
        )

        playback_menu.addAction(
            previous_action
        )

        next_action = QAction(
            "Next page",
            self,
        )

        next_action.setShortcut(
            QKeySequence("Right")
        )

        next_action.triggered.connect(
            self.next_spread
        )

        playback_menu.addAction(
            next_action
        )

        menu_bar.addMenu(
            playback_menu
        )

        # --------------------------------------------------------------
        # Page navigation
        # --------------------------------------------------------------

        navigation = QWidget(
            self
        )

        nav_layout = QHBoxLayout(
            navigation
        )

        nav_layout.setContentsMargins(
            4,
            0,
            4,
            0,
        )

        nav_layout.setSpacing(
            2
        )

        self.previous_button = (
            QPushButton(
                "‹",
                navigation,
            )
        )

        self.previous_button.setFixedWidth(
            28
        )

        self.previous_button.setToolTip(
            "Previous page"
        )

        self.previous_button.clicked.connect(
            self.previous_spread
        )

        self.page_input = QLineEdit(
            navigation
        )

        self.page_input.setFixedWidth(
            65
        )

        self.page_input.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.page_input.setToolTip(
            "Enter a page number and press Enter"
        )

        self.page_input.returnPressed.connect(
            self._page_input_activated
        )

        self.next_button = (
            QPushButton(
                "›",
                navigation,
            )
        )

        self.next_button.setFixedWidth(
            28
        )

        self.next_button.setToolTip(
            "Next page"
        )

        self.next_button.clicked.connect(
            self.next_spread
        )

        nav_layout.addWidget(
            self.previous_button
        )

        nav_layout.addWidget(
            self.page_input
        )

        nav_layout.addWidget(
            self.next_button
        )

        menu_bar.setCornerWidget(
            navigation,
            Qt.Corner.TopRightCorner,
        )

        self.setMenuBar(
            menu_bar
        )

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------

    def _create_status_bar(self):
        status = QStatusBar(
            self
        )

        self.status_label = QLabel()

        status.addPermanentWidget(
            self.status_label
        )

        self.setStatusBar(
            status
        )

        self._update_status()

    # ------------------------------------------------------------------
    # Page number
    # ------------------------------------------------------------------

    def _current_visible_page(self):
        if not self.spreads:
            return 0

        spread = self.spreads[
            self.current_spread
        ]

        # Left page wins if it exists.
        path = (
            spread["left"]
            if spread["left"] is not None
            else spread["right"]
        )

        if path is None:
            return 0

        try:
            return get_page_num(
                path
            )
        except Exception:
            return (
                self.current_spread
                + 1
            )

    def _update_page_input(self):
        page = (
            self._current_visible_page()
        )

        self.page_input.blockSignals(
            True
        )

        self.page_input.setText(
            str(page)
            if page
            else ""
        )

        self.page_input.blockSignals(
            False
        )

    # ------------------------------------------------------------------
    # Page input
    # ------------------------------------------------------------------

    def _page_input_activated(self):
        text = (
            self.page_input
            .text()
            .strip()
        )

        try:
            page = int(text)
        except ValueError:
            self._update_page_input()
            return

        self.go_to_page(page)

        self.canvas.setFocus()

    def go_to_page(self, page_number):
        if not self.spreads:
            return

        page_number = max(
            1,
            min(
                self.total_pages,
                page_number,
            ),
        )

        for index, spread in enumerate(
            self.spreads
        ):
            for key in (
                "left",
                "right",
            ):
                path = spread[key]

                if path is None:
                    continue

                try:
                    if (
                        get_page_num(path)
                        == page_number
                    ):
                        self._set_spread(
                            index
                        )
                        return
                except Exception:
                    pass

        # Fallback for filenames where the page-number parser doesn't
        # produce a perfect direct match.
        nearest = 0
        best_distance = float(
            "inf"
        )

        for index, spread in enumerate(
            self.spreads
        ):
            path = (
                spread["left"]
                if spread["left"]
                is not None
                else spread["right"]
            )

            try:
                number = get_page_num(
                    path
                )
            except Exception:
                continue

            distance = abs(
                number
                - page_number
            )

            if (
                distance
                < best_distance
            ):
                best_distance = (
                    distance
                )
                nearest = index

        self._set_spread(
            nearest
        )

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _set_spread(self, index):
        if not self.spreads:
            return

        index = max(
            0,
            min(
                len(self.spreads) - 1,
                index,
            ),
        )

        if (
            index
            == self.current_spread
        ):
            self._load_current_spread()
            return

        self.current_spread = index

        self._load_current_spread()

        # Important:
        # preload in BOTH directions after every seek.
        self._preload_around_current()

        self._update_page_input()
        self._update_status()
        self._update_window_title()
        self._update_progress()

    def previous_spread(self):
        if not self.spreads:
            return

        self._set_spread(
            self.current_spread - 1
        )

    def next_spread(self):
        if not self.spreads:
            return

        self._set_spread(
            self.current_spread + 1
        )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _request_image(self, path):
        if path is None:
            return

        path = str(path)

        if self.cache.contains(
            path
        ):
            return

        if path in self.loading:
            return

        self.loading.add(path)

        task = ImageLoadTask(
            path,
            self.cache,
        )

        task.signals.loaded.connect(
            self._image_loaded
        )

        task.signals.failed.connect(
            self._image_failed
        )

        self.thread_pool.start(
            task
        )

    def _image_loaded(self, path):
        self.loading.discard(
            path
        )

        self._load_current_spread()

        # Keep the symmetric cache window full.
        self._preload_around_current()

    def _image_failed(
        self,
        path,
        error,
    ):
        self.loading.discard(
            path
        )

        if not self.spreads:
            return

        spread = self.spreads[
            self.current_spread
        ]

        if (
            path == spread["left"]
            or path == spread["right"]
        ):
            self.status_label.setText(
                f"Could not load "
                f"{Path(path).name}: "
                f"{error}"
            )

    def _load_current_spread(self):
        if not self.spreads:
            self.canvas.set_images(
                None,
                None,
            )
            return

        spread = self.spreads[
            self.current_spread
        ]

        left = None
        right = None

        if spread["left"] is not None:
            left = self.cache.get(
                spread["left"]
            )

            if left is None:
                self._request_image(
                    spread["left"]
                )

        if spread["right"] is not None:
            right = self.cache.get(
                spread["right"]
            )

            if right is None:
                self._request_image(
                    spread["right"]
                )

        self.canvas.set_images(
            left,
            right,
        )

        # The initial fit can only be calculated once the actual images
        # have been decoded and their dimensions are known.
        if self._initial_fit_pending:
            left_ready = (
                spread["left"] is None
                or left is not None
            )

            right_ready = (
                spread["right"] is None
                or right is not None
            )

            if (
                left_ready
                and right_ready
            ):
                if self.canvas.fit_pages():
                    self._initial_fit_pending = (
                        False
                    )

        self._update_status()

    # ------------------------------------------------------------------
    # Bidirectional preloading
    # ------------------------------------------------------------------

    def _preload_around_current(self):
        """
        Keep a cache window around the current spread.

        Previously the preload window was:

            current -> +N

        which meant backward seeking could hit the disk repeatedly.

        It is now:

            -N <- current -> +N

        so both directions are buffered.

        The current spread is requested first, followed by spreads in
        increasing distance from the current position.
        """

        if not self.spreads:
            return

        current = self.current_spread
        count = len(self.spreads)

        # Request the current spread first.
        self._request_spread_images(
            current
        )

        # Then alternate around the current position:
        #
        # current-1, current+1,
        # current-2, current+2,
        # ...
        #
        # This means that after a seek, both directions begin filling
        # immediately rather than waiting for one entire side.
        for distance in range(
            1,
            self.preload_spreads + 1,
        ):
            previous = (
                current - distance
            )

            following = (
                current + distance
            )

            if previous >= 0:
                self._request_spread_images(
                    previous
                )

            if following < count:
                self._request_spread_images(
                    following
                )

    def _request_spread_images(
        self,
        spread_index,
    ):
        spread = self.spreads[
            spread_index
        ]

        if spread["left"] is not None:
            self._request_image(
                spread["left"]
            )

        if spread["right"] is not None:
            self._request_image(
                spread["right"]
            )

    # ------------------------------------------------------------------
    # Playback buffer
    # ------------------------------------------------------------------

    def _current_spread_ready(self):
        if not self.spreads:
            return True

        spread = self.spreads[
            self.current_spread
        ]

        if (
            spread["left"] is not None
            and not self.cache.contains(
                spread["left"]
            )
        ):
            return False

        if (
            spread["right"] is not None
            and not self.cache.contains(
                spread["right"]
            )
        ):
            return False

        return True

    def _next_spread_ready(self):
        next_index = (
            self.current_spread
            + 1
        )

        if (
            next_index
            >= len(self.spreads)
        ):
            return True

        spread = self.spreads[
            next_index
        ]

        if (
            spread["left"] is not None
            and not self.cache.contains(
                spread["left"]
            )
        ):
            return False

        if (
            spread["right"] is not None
            and not self.cache.contains(
                spread["right"]
            )
        ):
            return False

        return True

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def toggle_play(self):
        if not self.spreads:
            return

        self.playing = (
            not self.playing
        )

        if self.playing:
            self._start_playback()
        else:
            self._stop_playback()

        self._update_status()

    def _start_playback(self):
        self._preload_around_current()

        self._wait_for_playback_buffer()

    def _stop_playback(self):
        self._update_status()

    def _wait_for_playback_buffer(self):
        if not self.playing:
            return

        if (
            self.current_spread
            >= len(self.spreads) - 1
        ):
            self.playing = False
            self._update_status()
            return

        if not self._current_spread_ready():
            self.status_label.setText(
                "Buffering…"
            )

            QTimer.singleShot(
                30,
                self._wait_for_playback_buffer,
            )

            return

        if not self._next_spread_ready():
            self.status_label.setText(
                "Buffering next page…"
            )

            QTimer.singleShot(
                30,
                self._wait_for_playback_buffer,
            )

            return

        self._schedule_next_playback()

    def _schedule_next_playback(self):
        if not self.playing:
            return

        delay = max(
            1,
            int(
                self.page_time
                * 1000
            ),
        )

        QTimer.singleShot(
            delay,
            self._advance_playback,
        )

    def _advance_playback(self):
        if not self.playing:
            return

        if (
            self.current_spread
            >= len(self.spreads) - 1
        ):
            self.playing = False
            self._update_status()
            return

        self.next_spread()

        self._preload_around_current()

        if self._next_spread_ready():
            self._schedule_next_playback()
        else:
            self._wait_for_playback_buffer()

    # ------------------------------------------------------------------
    # Progress bar
    # ------------------------------------------------------------------

    def _update_progress(self):
        if not self.spreads:
            self.progress_bar.set_position(
                0.0
            )
            return

        if len(self.spreads) <= 1:
            position = 1.0
        else:
            position = (
                self.current_spread
                / (
                    len(self.spreads)
                    - 1
                )
            )

        self.progress_bar.set_position(
            position
        )

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def show_settings(self):
        dialog = SettingsDialog(
            self,
            self.page_time,
            self.preload_spreads,
            self.cache_images,
        )

        if (
            dialog.exec()
            != QDialog.DialogCode.Accepted
        ):
            return

        self.page_time = (
            dialog.page_time.value()
        )

        self.preload_spreads = (
            dialog.preload_spreads.value()
        )

        self.cache_images = (
            dialog.cache_images.value()
        )

        self.cache.set_max_images(
            self.cache_images
        )

        self._preload_around_current()
        self._update_status()

    # ------------------------------------------------------------------
    # Fullscreen
    # ------------------------------------------------------------------

    def _start_fullscreen(self):
        self._enter_fullscreen()

    def toggle_fullscreen(self):
        if self._fullscreen:
            self._leave_fullscreen()
        else:
            self._enter_fullscreen()

    def _enter_fullscreen(self):
        self._fullscreen = True

        # Hide menu and status bar.
        #
        # The progress bar is part of the central widget and therefore
        # remains visible.
        self.menuBar().hide()
        self.statusBar().hide()

        self.showFullScreen()

        self._chrome_hide_timer.stop()

        self.canvas.setFocus()

    def _leave_fullscreen(self):
        self._fullscreen = False

        self._chrome_hide_timer.stop()

        # Leave fullscreen and explicitly restore a maximized window.
        self.showNormal()

        # Important: showNormal() restores the previous normal geometry,
        # which can be tiny. showMaximized() ensures the window comes back
        # at maximum size.
        self.showMaximized()

        self.menuBar().show()
        self.statusBar().show()

        self.canvas.setFocus()

    def _show_fullscreen_chrome(self):
        if not self._fullscreen:
            return

        self.menuBar().show()
        self.statusBar().show()

        self._chrome_hide_timer.start(
            FULLSCREEN_CHROME_HIDE_DELAY
        )

    def _hide_fullscreen_chrome(self):
        if not self._fullscreen:
            return

        self.menuBar().hide()
        self.statusBar().hide()

    # ------------------------------------------------------------------
    # Mouse movement
    # ------------------------------------------------------------------

    def mouseMoveEvent(self, event):
        if self._fullscreen:
            y = event.position().y()

            if (
                y <= 8
                or y >= self.height() - 8
            ):
                self._show_fullscreen_chrome()

        super().mouseMoveEvent(
            event
        )

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()

        if key == Qt.Key.Key_Q:
            self.close()
            return

        if key == Qt.Key.Key_F:
            self.toggle_fullscreen()
            return

        if key == Qt.Key.Key_Up:
            self.canvas.zoom_in()
            return

        if key == Qt.Key.Key_Down:
            self.canvas.zoom_out()
            return

        if key == Qt.Key.Key_Left:
            self.previous_spread()
            return

        if key == Qt.Key.Key_Right:
            self.next_spread()
            return

        if key == Qt.Key.Key_Space:
            self.toggle_play()
            return

        if (
            key
            in (
                Qt.Key.Key_Plus,
                Qt.Key.Key_Equal,
            )
            and modifiers
            & Qt.KeyboardModifier.ControlModifier
        ):
            self.canvas.zoom_in()
            return

        if (
            key == Qt.Key.Key_Minus
            and modifiers
            & Qt.KeyboardModifier.ControlModifier
        ):
            self.canvas.zoom_out()
            return

        if (
            key == Qt.Key.Key_0
            and modifiers
            & Qt.KeyboardModifier.ControlModifier
        ):
            self.canvas.fit_pages()
            return

        super().keyPressEvent(
            event
        )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def _update_status(self):
        if not hasattr(
            self,
            "status_label",
        ):
            return

        if not self.spreads:
            self.status_label.setText(
                "No pages"
            )
            return

        page = (
            self._current_visible_page()
        )

        state = (
            "Playing"
            if self.playing
            else "Paused"
        )

        self.status_label.setText(
            f"{state}  •  "
            f"Page {page}  •  "
            f"Spread "
            f"{self.current_spread + 1}/"
            f"{len(self.spreads)}  •  "
            f"Zoom "
            f"{self.canvas.zoom:.2f}×  •  "
            f"Cache "
            f"{self.cache.size()}/"
            f"{self.cache_images}"
        )

        self._update_page_input()
        self._update_progress()

    def _update_window_title(self):
        page = (
            self._current_visible_page()
        )

        if page:
            self.setWindowTitle(
                "Scanned Book Viewer"
                f" — Page {page}"
            )
        else:
            self.setWindowTitle(
                "Scanned Book Viewer"
            )

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        self.playing = False

        self.thread_pool.clear()

        super().closeEvent(
            event
        )

# ---------------------------------------------------------------------------
# Find images
# ---------------------------------------------------------------------------

def find_images(source_directory):
    source_directory = Path(
        source_directory
    )

    if not source_directory.is_dir():
        raise ValueError(
            f"Not a directory: "
            f"{source_directory}"
        )

    scan_format = str(
        config.scan_format
    ).lower().lstrip(".")

    supported = {
        scan_format,
    }

    if scan_format == "tif":
        supported.add("tiff")
    elif scan_format == "tiff":
        supported.add("tif")

    images = []

    # Non-recursive.
    for path in source_directory.iterdir():
        if not path.is_file():
            continue

        extension = (
            path.suffix
            .lower()
            .lstrip(".")
        )

        if extension not in supported:
            continue

        images.append(path)

    images.sort(
        key=lambda path: (
            get_page_num(path),
            path.name.lower(),
        )
    )

    return images

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # __name__ is "__main__" when executed directly.
    #
    # __file__ gives the actual script filename, which is what argparse
    # should display in its usage/error messages.
    script_name = Path(
        __file__
    ).name

    parser = argparse.ArgumentParser(
        prog=script_name,
        description=(
            "Specialized scanned-book "
            "page viewer."
        ),
    )

    parser.add_argument(
        "source_directory",
        metavar="SOURCE_DIRECTORY",
        help=(
            "Directory containing the "
            "scanned page images."
        ),
    )

    args = parser.parse_args()

    try:
        image_paths = find_images(
            args.source_directory
        )

    except Exception as exc:
        print(
            f"{script_name}: error: "
            f"{exc}",
            file=sys.stderr,
        )

        return 2

    if not image_paths:
        print(
            f"{script_name}: no "
            f"{config.scan_format} images "
            f"found in "
            f"{args.source_directory}",
            file=sys.stderr,
        )

        return 1

    app = QApplication(
        sys.argv
    )

    app.setApplicationName(
        "Scanned Book Viewer"
    )

    app.setOrganizationName(
        "Scanned Book Viewer"
    )

    window = BookViewer(
        image_paths
    )

    # BookViewer starts fullscreen itself.
    #
    # Do not call showMaximized() here because that would briefly
    # establish a normal/maximized state before fullscreen.
    window.show()

    return app.exec()

if __name__ == "__main__":
    raise SystemExit(
        main()
    )
```

One important caching detail: the new preload order is **current, previous, next, previous-2, next-2, ...**, rather than simply loading all previous pages and then all future pages. That makes a newly selected location useful in **both directions immediately**, which should make back-and-forth browsing substantially smoother.

## Prompt:

9/9/2026, 6:33:38 PM

ok, now i want to add overflow:  
if autoplay has reached the last page, and the user hits "autoplay" again then it should start autoplaying from the start  
on the last page, navigating to the next page shows the first page

## Response:

9/9/2026, 6:33:41 PM

Yes — this can be handled cleanly at the **spread navigation level** so both manual navigation and autoplay wrap around.

The behavior will be:

- **Last page + Right Arrow** → first page.
- **Last page + mouse-wheel down** → first page.
- **Autoplay reaches the last page** → stops as it does now.
- Press **Space/autoplay again while on the last page** → jumps to page 1 and starts autoplay.
- Normal autoplay from any earlier page is unchanged.
- **Previous** from the first page can also wrap to the last page if you want symmetric overflow.

For the existing `BookViewer`, update the navigation/autoplay logic along these lines:

\*\*document: \*\*

``` python
# Add/replace the spread navigation methods in BookViewer.

def next_spread(self):
    """Go to the next spread, wrapping from the last to the first."""
    if not self.spreads:
        return

    if self.current_spread >= len(self.spreads) - 1:
        self.current_spread = 0
    else:
        self.current_spread += 1

    self._show_current_spread()

def previous_spread(self):
    """Go to the previous spread, wrapping from the first to the last."""
    if not self.spreads:
        return

    if self.current_spread <= 0:
        self.current_spread = len(self.spreads) - 1
    else:
        self.current_spread -= 1

    self._show_current_spread()

def toggle_playback(self):
    """Start/pause autoplay, with overflow restart from the beginning."""
    if not self.spreads:
        return

    if self.playing:
        self._stop_playback()
        return

    # If autoplay has reached the end, pressing autoplay again
    # restarts from the beginning.
    if self.current_spread >= len(self.spreads) - 1:
        self.current_spread = 0
        self._show_current_spread()

    self._start_playback()

def _advance_autoplay(self):
    """Advance autoplay without wrapping automatically."""
    if not self.playing or not self.spreads:
        return

    # We have reached the final spread.
    # Stop here rather than immediately looping.
    if self.current_spread >= len(self.spreads) - 1:
        self._stop_playback()
        return

    self.current_spread += 1
    self._show_current_spread()

def keyPressEvent(self, event):
    key = event.key()

    if key == Qt.Key_Right:
        self.next_spread()
        event.accept()
        return

    if key == Qt.Key_Left:
        self.previous_spread()
        event.accept()
        return

    if key == Qt.Key_Space:
        self.toggle_playback()
        event.accept()
        return

    if key == Qt.Key_Up:
        self.canvas.zoom_in()
        event.accept()
        return

    if key == Qt.Key_Down:
        self.canvas.zoom_out()
        event.accept()
        return

    if key == Qt.Key_F:
        self.toggle_fullscreen()
        event.accept()
        return

    if key == Qt.Key_Q:
        self.close()
        event.accept()
        return

    super().keyPressEvent(event)
```

And in the canvas `wheelEvent()`, keep the existing Ctrl-to-zoom behavior but make normal navigation use the wrapping methods:

``` python
def wheelEvent(self, event):
    delta = event.angleDelta().y()

    if delta == 0:
        event.ignore()
        return

    modifiers = event.modifiers()

    if modifiers & Qt.ControlModifier:
        if delta > 0:
            self.viewer.zoom_in()
        else:
            self.viewer.zoom_out()

        event.accept()
        return

    # Normal mouse wheel = page navigation.
    if delta > 0:
        self.viewer.previous_spread()
    else:
        self.viewer.next_spread()

    event.accept()
```

Finally, make sure the autoplay timer calls `_advance_autoplay()` rather than calling `next_spread()` directly:

``` python
self.play_timer.timeout.connect(self._advance_autoplay)
```

The important distinction is that **manual navigation wraps**, while **autoplay stops at the end**. Therefore:

``` text
Right Arrow on last spread
        ↓
      first spread

Wheel down on last spread
        ↓
      first spread

Autoplay reaches last spread
        ↓
      stops

Press Space again
        ↓
   first spread
        ↓
   autoplay starts
```

This preserves the behavior you described rather than making autoplay loop continuously.

## Prompt:

9/9/2026, 6:45:01 PM

i cannot apply your patch, it fails with `AttributeError: 'BookCanvas' object has no attribute 'viewer'`  
instead, give me the full updated script

## Response:

9/9/2026, 6:45:04 PM

You're right — that patch assumed a `viewer` attribute that the existing `BookCanvas` did not have.

Here is the **complete updated script**, based on the previous version, with the overflow behavior integrated properly:

- **Right Arrow / wheel down on last spread → first spread**
- **Left Arrow / wheel up on first spread → last spread**
- Autoplay **stops at the last spread**
- Press **Space again at the last spread → jumps to first spread and starts autoplay**
- Ctrl+wheel remains zoom
- Up/Down zoom
- Zoom/pan state is preserved between spreads
- Bidirectional image preloading
- Fullscreen/headless chrome
- Progress bar remains visible in fullscreen
- Initial fit-to-pages
- CLI source directory
- `config.scan_format`
- `_shared.load_config()` / `get_page_num()`

\*\*document: \*\*

``` python
import argparse
import sys
from collections import OrderedDict
from pathlib import Path

from PySide6.QtCore import (
    QObject,
    QPointF,
    QRectF,
    QRunnable,
    QMutex,
    QMutexLocker,
    Qt,
    QThreadPool,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QAction,
    QImage,
    QImageReader,
    QKeySequence,
    QPainter,
)
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenuBar,
    QPushButton,
    QSpinBox,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from _shared import (
    load_config,
    get_page_num,
)

config = load_config()

# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

class ImageLoadSignals(QObject):
    loaded = Signal(int, QImage)

class ImageLoadTask(QRunnable):
    def __init__(self, index, path):
        super().__init__()
        self.index = index
        self.path = path
        self.signals = ImageLoadSignals()

    def run(self):
        image = QImage()

        reader = QImageReader(str(self.path))
        reader.setAutoTransform(True)

        image = reader.read()

        if not image.isNull():
            # Detach the image from the reader/thread.
            image = image.copy()

        self.signals.loaded.emit(self.index, image)

class ImageCache(QObject):
    image_ready = Signal(int)

    def __init__(self, paths, max_images=60):
        super().__init__()

        self.paths = paths
        self.max_images = max_images

        self.images = OrderedDict()
        self.loading = set()

        self.mutex = QMutex()
        self.thread_pool = QThreadPool.globalInstance()

    def get(self, index):
        with QMutexLocker(self.mutex):
            image = self.images.get(index)

            if image is not None:
                self.images.move_to_end(index)

            return image

    def has(self, index):
        with QMutexLocker(self.mutex):
            return index in self.images

    def request(self, index):
        if index < 0 or index >= len(self.paths):
            return

        with QMutexLocker(self.mutex):
            if index in self.images:
                self.images.move_to_end(index)
                return

            if index in self.loading:
                return

            self.loading.add(index)

        task = ImageLoadTask(index, self.paths[index])
        task.signals.loaded.connect(self._image_loaded)
        self.thread_pool.start(task)

    def _image_loaded(self, index, image):
        with QMutexLocker(self.mutex):
            self.loading.discard(index)

            if image.isNull():
                pass
            else:
                self.images[index] = image
                self.images.move_to_end(index)

                while len(self.images) > self.max_images:
                    self.images.popitem(last=False)

        self.image_ready.emit(index)

# ---------------------------------------------------------------------------
# Progress bar
# ---------------------------------------------------------------------------

class BookProgressBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.position = 0.0

        self.setFixedHeight(4)
        self.setMinimumHeight(4)
        self.setMaximumHeight(4)

        self.setStyleSheet(
            """
            QWidget {
                background: white;
            }
            """
        )

    def set_position(self, position):
        self.position = max(0.0, min(1.0, float(position)))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)

        try:
            painter.setRenderHint(QPainter.Antialiasing, False)

            # White background.
            painter.fillRect(self.rect(), Qt.white)

            # Solid black progress indicator.
            width = int(self.width() * self.position)

            if width > 0:
                painter.fillRect(
                    0,
                    0,
                    width,
                    self.height(),
                    Qt.black,
                )

        finally:
            painter.end()

# ---------------------------------------------------------------------------
# Book canvas
# ---------------------------------------------------------------------------

class BookCanvas(QWidget):
    def __init__(self, book_viewer, parent=None):
        super().__init__(parent)

        self.book_viewer = book_viewer

        self.left_image = None
        self.right_image = None

        self.zoom = 1.0
        self.pan = QPointF(0.0, 0.0)

        self.dragging = False
        self.last_mouse_position = QPointF()

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

    # ------------------------------------------------------------------
    # Images
    # ------------------------------------------------------------------

    def set_images(self, left_image, right_image):
        self.left_image = left_image
        self.right_image = right_image
        self.update()

    # ------------------------------------------------------------------
    # Zoom
    # ------------------------------------------------------------------

    def fit_pages(self):
        if self.left_image is None and self.right_image is None:
            return

        available_width = max(1, self.width())
        available_height = max(1, self.height())

        left_width = (
            self.left_image.width()
            if self.left_image is not None
            else 0
        )
        left_height = (
            self.left_image.height()
            if self.left_image is not None
            else 0
        )

        right_width = (
            self.right_image.width()
            if self.right_image is not None
            else 0
        )
        right_height = (
            self.right_image.height()
            if self.right_image is not None
            else 0
        )

        combined_width = left_width + right_width
        combined_height = max(left_height, right_height)

        if combined_width <= 0 or combined_height <= 0:
            return

        scale_x = available_width / combined_width
        scale_y = available_height / combined_height

        self.zoom = min(scale_x, scale_y)
        self.pan = QPointF(0.0, 0.0)

        self.update()

    def set_zoom(self, zoom, anchor=None):
        old_zoom = self.zoom

        zoom = max(0.05, min(10.0, float(zoom)))

        if abs(zoom - old_zoom) < 1e-9:
            return

        if anchor is None:
            anchor = QPointF(
                self.width() / 2.0,
                self.height() / 2.0,
            )

        # Keep the point under the cursor stationary while zooming.
        world_x = (anchor.x() - self.pan.x()) / old_zoom
        world_y = (anchor.y() - self.pan.y()) / old_zoom

        self.zoom = zoom

        self.pan = QPointF(
            anchor.x() - world_x * self.zoom,
            anchor.y() - world_y * self.zoom,
        )

        self.update()

    def zoom_in(self, anchor=None):
        self.set_zoom(self.zoom * 1.15, anchor)

    def zoom_out(self, anchor=None):
        self.set_zoom(self.zoom / 1.15, anchor)

    # ------------------------------------------------------------------
    # Mouse interaction
    # ------------------------------------------------------------------

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.last_mouse_position = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        # Notify the main window so fullscreen chrome can be revealed.
        self.book_viewer.handle_mouse_move(event.position())

        if self.dragging:
            current = event.position()
            delta = current - self.last_mouse_position

            self.pan += delta
            self.last_mouse_position = current

            self.update()

            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()

        if delta == 0:
            event.ignore()
            return

        if event.modifiers() & Qt.ControlModifier:
            if delta > 0:
                self.zoom_in(event.position())
            else:
                self.zoom_out(event.position())

            event.accept()
            return

        # Normal mouse wheel = page navigation.
        #
        # Positive wheel delta is "up" -> previous page.
        # Negative wheel delta is "down" -> next page.
        if delta > 0:
            self.book_viewer.previous_spread()
        else:
            self.book_viewer.next_spread()

        event.accept()

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event):
        painter = QPainter(self)

        try:
            painter.setRenderHint(QPainter.SmoothPixmapTransform, False)

            painter.fillRect(self.rect(), Qt.black)

            if self.left_image is None and self.right_image is None:
                return

            images = []

            if self.left_image is not None:
                images.append(("left", self.left_image))

            if self.right_image is not None:
                images.append(("right", self.right_image))

            total_width = sum(image.width() for _, image in images)
            max_height = max(
                image.height()
                for _, image in images
            )

            if total_width <= 0 or max_height <= 0:
                return

            scaled_width = total_width * self.zoom
            scaled_height = max_height * self.zoom

            origin_x = (
                (self.width() - scaled_width) / 2.0
                + self.pan.x()
            )

            origin_y = (
                (self.height() - scaled_height) / 2.0
                + self.pan.y()
            )

            x = origin_x

            for side, image in images:
                image_width = image.width() * self.zoom
                image_height = image.height() * self.zoom

                y = (
                    origin_y
                    + (scaled_height - image_height) / 2.0
                )

                # At 1:1, draw directly in native pixels for maximum
                # sharpness. At other zoom levels, use a QRectF.
                if abs(self.zoom - 1.0) < 1e-9:
                    painter.drawImage(
                        int(round(x)),
                        int(round(y)),
                        image,
                    )
                else:
                    target = QRectF(
                        x,
                        y,
                        image_width,
                        image_height,
                    )

                    painter.drawImage(
                        target,
                        image,
                    )

                x += image_width

        finally:
            painter.end()

# ---------------------------------------------------------------------------
# Settings dialog
# ---------------------------------------------------------------------------

class PlaybackSettingsDialog(QDialog):
    def __init__(self, page_time, preload_spreads, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Playback settings")

        layout = QFormLayout(self)

        self.page_time_spin = QDoubleSpinBox()
        self.page_time_spin.setRange(0.01, 60.0)
        self.page_time_spin.setDecimals(2)
        self.page_time_spin.setSingleStep(0.05)
        self.page_time_spin.setValue(page_time)
        self.page_time_spin.setSuffix(" s")

        self.preload_spin = QSpinBox()
        self.preload_spin.setRange(0, 500)
        self.preload_spin.setValue(preload_spreads)

        layout.addRow(
            "Time per spread:",
            self.page_time_spin,
        )

        layout.addRow(
            "Preload spreads:",
            self.preload_spin,
        )

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok
            | QDialogButtonBox.Cancel
        )

        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout.addRow(buttons)

# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class BookViewer(QMainWindow):
    def __init__(self, source_dir):
        super().__init__()

        self.source_dir = Path(source_dir)

        self.setWindowTitle(
            f"Book Viewer — {self.source_dir.name}"
        )

        self.resize(1400, 900)

        # --------------------------------------------------------------
        # Configuration
        # --------------------------------------------------------------

        self.page_time = 0.2
        self.preload_spreads = 20
        self.max_cache_images = 60

        self.playing = False
        self.current_spread = 0

        self._initial_fit_pending = True
        self._fullscreen = False

        self._chrome_timer = QTimer(self)
        self._chrome_timer.setSingleShot(True)
        self._chrome_timer.timeout.connect(
            self._hide_fullscreen_chrome
        )

        # --------------------------------------------------------------
        # Find pages
        # --------------------------------------------------------------

        extension = str(config.scan_format).lower().lstrip(".")

        self.paths = sorted(
            [
                p
                for p in self.source_dir.iterdir()
                if p.is_file()
                and p.suffix.lower().lstrip(".") == extension
            ],
            key=self._page_sort_key,
        )

        if not self.paths:
            raise RuntimeError(
                f"No .{extension} images found in "
                f"{self.source_dir}"
            )

        # --------------------------------------------------------------
        # Build spreads
        #
        # Page 1 alone on right.
        # Pages 2-3
        # Pages 4-5
        # ...
        #
        # If the book ends on an even page, that final page is alone
        # on the left.
        # --------------------------------------------------------------

        self.spreads = []

        first = {
            "left": None,
            "right": 0,
        }

        self.spreads.append(first)

        index = 1

        while index < len(self.paths):
            left = index
            right = index + 1

            self.spreads.append(
                {
                    "left": left,
                    "right": right
                    if right < len(self.paths)
                    else None,
                }
            )

            index += 2

        # --------------------------------------------------------------
        # Cache
        # --------------------------------------------------------------

        self.cache = ImageCache(
            self.paths,
            max_images=self.max_cache_images,
        )

        self.cache.image_ready.connect(
            self._image_ready
        )

        # --------------------------------------------------------------
        # Canvas
        # --------------------------------------------------------------

        self.canvas = BookCanvas(
            self,
            self,
        )

        # --------------------------------------------------------------
        # Progress bar
        # --------------------------------------------------------------

        self.progress_bar = BookProgressBar()

        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)

        central_layout.addWidget(self.canvas, 1)
        central_layout.addWidget(self.progress_bar, 0)

        self.setCentralWidget(central)

        # --------------------------------------------------------------
        # Menu bar
        # --------------------------------------------------------------

        self._build_menu()

        # --------------------------------------------------------------
        # Status bar
        # --------------------------------------------------------------

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # --------------------------------------------------------------
        # Page navigation
        # --------------------------------------------------------------

        self.previous_action = QAction(
            "Previous",
            self,
        )
        self.previous_action.setShortcut(
            QKeySequence(Qt.Key_Left)
        )
        self.previous_action.triggered.connect(
            self.previous_spread
        )

        self.next_action = QAction(
            "Next",
            self,
        )
        self.next_action.setShortcut(
            QKeySequence(Qt.Key_Right)
        )
        self.next_action.triggered.connect(
            self.next_spread
        )

        self.current_page_edit = QLineEdit()
        self.current_page_edit.setFixedWidth(70)
        self.current_page_edit.setAlignment(
            Qt.AlignCenter
        )

        self.current_page_edit.returnPressed.connect(
            self._page_edit_return_pressed
        )

        self.nav_widget = QWidget()
        nav_layout = QHBoxLayout(self.nav_widget)
        nav_layout.setContentsMargins(4, 0, 4, 0)
        nav_layout.setSpacing(2)

        previous_button = QPushButton("◀")
        previous_button.setFixedWidth(28)
        previous_button.clicked.connect(
            self.previous_spread
        )

        next_button = QPushButton("▶")
        next_button.setFixedWidth(28)
        next_button.clicked.connect(
            self.next_spread
        )

        nav_layout.addWidget(previous_button)
        nav_layout.addWidget(self.current_page_edit)
        nav_layout.addWidget(next_button)

        self.menu_bar.setCornerWidget(
            self.nav_widget,
            Qt.TopRightCorner,
        )

        # --------------------------------------------------------------
        # Playback
        # --------------------------------------------------------------

        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(
            self._advance_autoplay
        )

        self._update_page_time()

        # --------------------------------------------------------------
        # Show first spread
        # --------------------------------------------------------------

        self._show_current_spread()

        # --------------------------------------------------------------
        # Start fullscreen after the window has been created.
        # --------------------------------------------------------------

        QTimer.singleShot(
            0,
            self._start_fullscreen,
        )

    # ------------------------------------------------------------------
    # Page sorting
    # ------------------------------------------------------------------

    def _page_sort_key(self, path):
        try:
            return (
                0,
                get_page_num(path),
            )
        except Exception:
            return (
                1,
                path.name.lower(),
            )

    # ------------------------------------------------------------------
    # Menus
    # ------------------------------------------------------------------

    def _build_menu(self):
        self.menu_bar = QMenuBar(self)
        self.setMenuBar(self.menu_bar)

        file_menu = self.menu_bar.addMenu("File")

        quit_action = QAction(
            "Quit",
            self,
        )
        quit_action.setShortcut(
            QKeySequence("Q")
        )
        quit_action.triggered.connect(
            self.close
        )

        file_menu.addAction(quit_action)

        view_menu = self.menu_bar.addMenu("View")

        fullscreen_action = QAction(
            "Fullscreen",
            self,
        )
        fullscreen_action.setShortcut(
            QKeySequence("F")
        )
        fullscreen_action.triggered.connect(
            self.toggle_fullscreen
        )

        view_menu.addAction(fullscreen_action)

        fit_action = QAction(
            "Fit pages",
            self,
        )
        fit_action.triggered.connect(
            self.canvas.fit_pages
        )

        view_menu.addAction(fit_action)

        playback_menu = self.menu_bar.addMenu(
            "Playback"
        )

        self.play_action = QAction(
            "Play / Pause",
            self,
        )
        self.play_action.setShortcut(
            QKeySequence(Qt.Key_Space)
        )
        self.play_action.triggered.connect(
            self.toggle_playback
        )

        playback_menu.addAction(
            self.play_action
        )

        settings_action = QAction(
            "Settings...",
            self,
        )
        settings_action.triggered.connect(
            self.show_playback_settings
        )

        playback_menu.addAction(
            settings_action
        )

    # ------------------------------------------------------------------
    # Spread navigation
    # ------------------------------------------------------------------

    def _show_current_spread(self):
        if not self.spreads:
            return

        self.current_spread = max(
            0,
            min(
                self.current_spread,
                len(self.spreads) - 1,
            ),
        )

        spread = self.spreads[self.current_spread]

        # --------------------------------------------------------------
        # Bidirectional preload.
        #
        # Current spread first, then alternately backwards and forwards.
        # This makes previous/next equally responsive.
        # --------------------------------------------------------------

        self._preload_around_current()

        # --------------------------------------------------------------
        # Display currently cached images.
        # --------------------------------------------------------------

        left_image = None
        right_image = None

        if spread["left"] is not None:
            left_image = self.cache.get(
                spread["left"]
            )

        if spread["right"] is not None:
            right_image = self.cache.get(
                spread["right"]
            )

        self.canvas.set_images(
            left_image,
            right_image,
        )

        # --------------------------------------------------------------
        # If the current spread is not yet decoded, wait for it.
        # --------------------------------------------------------------

        if (
            spread["left"] is not None
            and left_image is None
        ):
            self.cache.request(
                spread["left"]
            )

        if (
            spread["right"] is not None
            and right_image is None
        ):
            self.cache.request(
                spread["right"]
            )

        # --------------------------------------------------------------
        # Update UI.
        # --------------------------------------------------------------

        self._update_page_display()
        self._update_progress()
        self._update_status()

        # Initial fit is deliberately delayed until the first spread's
        # images have actually been decoded.
        if self._initial_fit_pending:
            if self._current_spread_is_loaded():
                self.canvas.fit_pages()
                self._initial_fit_pending = False

    def _current_spread_is_loaded(self):
        spread = self.spreads[self.current_spread]

        if (
            spread["left"] is not None
            and not self.cache.has(spread["left"])
        ):
            return False

        if (
            spread["right"] is not None
            and not self.cache.has(spread["right"])
        ):
            return False

        return True

    def _preload_around_current(self):
        if not self.spreads:
            return

        current = self.current_spread

        order = [current]

        for distance in range(
            1,
            self.preload_spreads + 1,
        ):
            backward = current - distance
            forward = current + distance

            if backward >= 0:
                order.append(backward)

            if forward < len(self.spreads):
                order.append(forward)

        for spread_index in order:
            spread = self.spreads[spread_index]

            if spread["left"] is not None:
                self.cache.request(
                    spread["left"]
                )

            if spread["right"] is not None:
                self.cache.request(
                    spread["right"]
                )

    def next_spread(self):
        """
        Advance one spread.

        Manual navigation wraps around:
            last -> first
        """
        if not self.spreads:
            return

        if self.current_spread >= len(self.spreads) - 1:
            self.current_spread = 0
        else:
            self.current_spread += 1

        self._show_current_spread()

    def previous_spread(self):
        """
        Go back one spread.

        Manual navigation wraps around:
            first -> last
        """
        if not self.spreads:
            return

        if self.current_spread <= 0:
            self.current_spread = len(self.spreads) - 1
        else:
            self.current_spread -= 1

        self._show_current_spread()

    # ------------------------------------------------------------------
    # Page-number navigation
    # ------------------------------------------------------------------

    def _visible_page_index(self):
        spread = self.spreads[self.current_spread]

        if spread["left"] is not None:
            return spread["left"]

        return spread["right"]

    def _visible_page_number(self):
        index = self._visible_page_index()

        if index is None:
            return ""

        try:
            return str(
                get_page_num(
                    self.paths[index]
                )
            )
        except Exception:
            return str(index + 1)

    def _update_page_display(self):
        self.current_page_edit.setText(
            self._visible_page_number()
        )

    def _page_edit_return_pressed(self):
        text = self.current_page_edit.text().strip()

        if not text:
            self._update_page_display()
            return

        try:
            target_page = int(text)
        except ValueError:
            self._update_page_display()
            return

        self.go_to_page(target_page)

    def go_to_page(self, page_number):
        if not self.paths:
            return

        # First try an exact page-number match.
        exact_index = None

        for index, path in enumerate(self.paths):
            try:
                if get_page_num(path) == page_number:
                    exact_index = index
                    break
            except Exception:
                continue

        if exact_index is None:
            # Fall back to the closest page by filename/index.
            best_index = None
            best_distance = None

            for index, path in enumerate(self.paths):
                try:
                    number = get_page_num(path)
                except Exception:
                    number = index + 1

                distance = abs(
                    number - page_number
                )

                if (
                    best_distance is None
                    or distance < best_distance
                ):
                    best_distance = distance
                    best_index = index

            exact_index = best_index

        if exact_index is None:
            return

        # Page 1 belongs to the first spread.
        if exact_index == 0:
            self.current_spread = 0
        else:
            self.current_spread = (
                (exact_index - 1) // 2
            ) + 1

        self._show_current_spread()

    # ------------------------------------------------------------------
    # Cache notifications
    # ------------------------------------------------------------------

    def _image_ready(self, index):
        spread = self.spreads[self.current_spread]

        if index in (
            spread["left"],
            spread["right"],
        ):
            self._show_current_spread()

        # Autoplay waits until the current spread is ready before moving.
        if self.playing:
            self._maybe_start_autoplay_timer()

    # ------------------------------------------------------------------
    # Autoplay
    # ------------------------------------------------------------------

    def toggle_playback(self):
        if not self.spreads:
            return

        if self.playing:
            self._stop_playback()
            return

        # IMPORTANT:
        #
        # If autoplay previously reached the final spread, pressing
        # Space again restarts playback from the beginning.
        if self.current_spread >= len(self.spreads) - 1:
            self.current_spread = 0
            self._show_current_spread()

        self._start_playback()

    def _start_playback(self):
        if self.playing:
            return

        self.playing = True

        self._update_status()
        self._maybe_start_autoplay_timer()

    def _stop_playback(self):
        self.playing = False
        self.play_timer.stop()

        self._update_status()

    def _maybe_start_autoplay_timer(self):
        if not self.playing:
            return

        if self.current_spread >= len(self.spreads) - 1:
            # Autoplay deliberately stops at the last spread.
            self._stop_playback()
            return

        # Don't advance until the current spread is actually available.
        if not self._current_spread_is_loaded():
            self._preload_around_current()
            return

        # Also request the next spread explicitly.
        next_index = self.current_spread + 1

        if next_index < len(self.spreads):
            next_spread = self.spreads[next_index]

            if next_spread["left"] is not None:
                self.cache.request(
                    next_spread["left"]
                )

            if next_spread["right"] is not None:
                self.cache.request(
                    next_spread["right"]
                )

        if not self.play_timer.isActive():
            self.play_timer.start(
                max(
                    1,
                    int(self.page_time * 1000),
                )
            )

    def _advance_autoplay(self):
        if not self.playing:
            return

        # Stop at the final spread.
        #
        # We do NOT wrap automatically because the requested behavior is:
        #   autoplay reaches last page -> stop
        #   press autoplay again -> restart at first page
        if self.current_spread >= len(self.spreads) - 1:
            self._stop_playback()
            return

        next_index = self.current_spread + 1

        next_spread = self.spreads[next_index]

        # Wait for the next spread to be cached before advancing.
        if (
            next_spread["left"] is not None
            and not self.cache.has(
                next_spread["left"]
            )
        ):
            self.play_timer.stop()

            self.cache.request(
                next_spread["left"]
            )

            if next_spread["right"] is not None:
                self.cache.request(
                    next_spread["right"]
                )

            return

        if (
            next_spread["right"] is not None
            and not self.cache.has(
                next_spread["right"]
            )
        ):
            self.play_timer.stop()

            self.cache.request(
                next_spread["right"]
            )

            return

        self.current_spread = next_index

        self._show_current_spread()

        # Restart timer for the newly displayed spread.
        self.play_timer.start(
            max(
                1,
                int(self.page_time * 1000),
            )
        )

    def _update_page_time(self):
        self.play_timer.setInterval(
            max(
                1,
                int(self.page_time * 1000),
            )
        )

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def show_playback_settings(self):
        dialog = PlaybackSettingsDialog(
            self.page_time,
            self.preload_spreads,
            self,
        )

        if dialog.exec() != QDialog.Accepted:
            return

        self.page_time = (
            dialog.page_time_spin.value()
        )

        self.preload_spreads = (
            dialog.preload_spin.value()
        )

        self._update_page_time()

        if self.playing:
            self._maybe_start_autoplay_timer()

        self._preload_around_current()

    # ------------------------------------------------------------------
    # Progress
    # ------------------------------------------------------------------

    def _update_progress(self):
        if len(self.spreads) <= 1:
            position = 1.0
        else:
            position = (
                self.current_spread
                / (len(self.spreads) - 1)
            )

        self.progress_bar.set_position(
            position
        )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def _update_status(self):
        page = self._visible_page_number()

        if self.playing:
            state = "Playing"
        else:
            state = "Paused"

        self.status_bar.showMessage(
            f"Page {page} — {state}"
        )

    # ------------------------------------------------------------------
    # Fullscreen
    # ------------------------------------------------------------------

    def _start_fullscreen(self):
        self.showFullScreen()
        self._fullscreen = True

        self._hide_fullscreen_chrome()

        # Make sure the canvas gets the focus.
        self.canvas.setFocus()

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self._leave_fullscreen()
        else:
            self._enter_fullscreen()

    def _enter_fullscreen(self):
        self.showFullScreen()
        self._fullscreen = True

        self._hide_fullscreen_chrome()

        self.canvas.setFocus()

    def _leave_fullscreen(self):
        # showNormal() alone can restore a tiny window if the application
        # was originally created with a small/default size.
        #
        # Therefore explicitly maximize after leaving fullscreen.
        self.showNormal()
        self.showMaximized()

        self._fullscreen = False

        self._show_fullscreen_chrome()

        self.canvas.setFocus()

    def _hide_fullscreen_chrome(self):
        if not self.isFullScreen():
            return

        self.menu_bar.hide()
        self.status_bar.hide()

    def _show_fullscreen_chrome(self):
        self.menu_bar.show()
        self.status_bar.show()

    def handle_mouse_move(self, position):
        if not self.isFullScreen():
            return

        # Reveal the chrome when the pointer approaches the top or bottom.
        y = position.y()

        top_zone = 50
        bottom_zone = self.height() - 50

        if y <= top_zone or y >= bottom_zone:
            self._show_fullscreen_chrome()

            self._chrome_timer.start(1800)

    # ------------------------------------------------------------------
    # Window events
    # ------------------------------------------------------------------

    def mouseMoveEvent(self, event):
        self.handle_mouse_move(
            event.position()
        )

        super().mouseMoveEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)

        # Only the first layout should automatically fit the pages.
        # After that, resizing preserves the user's zoom/pan state.
        if self._initial_fit_pending:
            QTimer.singleShot(
                0,
                self._try_initial_fit,
            )

    def _try_initial_fit(self):
        if not self._initial_fit_pending:
            return

        if self._current_spread_is_loaded():
            self.canvas.fit_pages()
            self._initial_fit_pending = False

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()

        # Ctrl + +/- zoom.
        if (
            modifiers & Qt.ControlModifier
            and key in (
                Qt.Key_Plus,
                Qt.Key_Equal,
            )
        ):
            self.canvas.zoom_in()
            event.accept()
            return

        if (
            modifiers & Qt.ControlModifier
            and key == Qt.Key_Minus
        ):
            self.canvas.zoom_out()
            event.accept()
            return

        # Navigation.
        if key == Qt.Key_Right:
            self.next_spread()
            event.accept()
            return

        if key == Qt.Key_Left:
            self.previous_spread()
            event.accept()
            return

        # Zoom.
        if key == Qt.Key_Up:
            self.canvas.zoom_in()
            event.accept()
            return

        if key == Qt.Key_Down:
            self.canvas.zoom_out()
            event.accept()
            return

        # Playback.
        if key == Qt.Key_Space:
            self.toggle_playback()
            event.accept()
            return

        # Fullscreen.
        if key == Qt.Key_F:
            self.toggle_fullscreen()
            event.accept()
            return

        # Quit.
        if key == Qt.Key_Q:
            self.close()
            event.accept()
            return

        super().keyPressEvent(event)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog=Path(__file__).name,
        description="Scanned book page viewer",
    )

    parser.add_argument(
        "source_dir",
        help="Directory containing scanned page images",
    )

    args = parser.parse_args()

    app = QApplication(sys.argv)

    try:
        window = BookViewer(
            args.source_dir
        )
    except Exception as exc:
        print(
            f"Error: {exc}",
            file=sys.stderr,
        )
        return 1

    window.show()

    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
```

The key fix compared with my previous patch is that `BookCanvas` now receives the `BookViewer` explicitly:

``` python
self.canvas = BookCanvas(self, self)
```

and therefore its wheel/mouse handling calls:

``` python
self.book_viewer.next_spread()
self.book_viewer.previous_spread()
```

rather than assuming `self.viewer` exists.

The overflow behavior is also deliberately **asymmetric for autoplay**: manual navigation wraps, but autoplay stops at the end. Pressing Space after that restarts from page 1.

## Prompt:

9/10/2026, 6:10:11 AM

if autoplay is active, then manual navigation (scroll with mousewheel or arrow left/right) should stop the autplay

add hotkey:  
enter = doubleclick = fit pages

click on the progress bar = seek to that page (if autoplay was active then stop autoplay)

remove the status bar on the bottom. instead, show the status in the menu bar on the top

zoom should follow the mouse pointer

avoid flicker on manual navigation  
flicker on autplay is rare (or non-existant?)  
behind the page images there should be a lightgray background (95% white + 5% black)  
(in darkmode: darkgray: 95% black + 5% white)

add darkmode: invert all colors on content pages  
use the system color theme

show a single page either on the left side (even page) or on the right side (odd page)

add a CLI option: pages -\> parse a page spec string like "10,20-30"  
parse the page spec string with the function  
\_shared.parse_page_sequence(page_spec_str, config.num_pages)

only show content pages  
from page 1 to config.num_pages

## Response:

9/10/2026, 6:10:16 AM

> **Planning the reader**
>
> I’m inferring a Qt reader with page-range parsing, parity-based placement, atomic image swaps for smooth navigation, caching, and light/dark rendering that inverts page colors when needed.
>
> **Planned the reader**
>
> ------------------------------------------------------------------------
>
> **Implementing page selection, spreads, and navigation UI**
>
> **Implementing page selection, spreads, and navigation UI**
>
> Worked for 10s

Absolutely. I’ve folded all of those changes into the full script rather than giving you a patch.

A couple of implementation details are worth calling out:

- Manual navigation **always stops autoplay**.
- `Enter` and double-click both perform **Fit pages**.
- The progress bar is clickable and seeks proportionally through the selected content pages.
- The old status bar is gone; its information is now displayed in the top menu bar.
- Manual page changes keep the old frame displayed until the new spread is fully ready, which avoids the blank/black flicker.
- Zoom is anchored to the mouse position.
- The page background follows the system light/dark theme.
- In dark mode, page images have their RGB colors inverted.
- Only pages `1 .. config.num_pages` are considered.
- `--pages "10,20-30"` uses `_shared.parse_page_sequence(page_spec_str, config.num_pages)`.
- Selected pages are displayed according to their original parity: even pages on the left, odd pages on the right. A selected even/odd pair is shown together when they form a normal facing-page pair.
- If no `--pages` is supplied, all content pages are shown.
- Manual navigation wraps around.
- Autoplay stops at the end; pressing Space again at the end restarts from the beginning.

\*\*document: \*\*

``` python
import argparse
import sys
from collections import OrderedDict
from pathlib import Path

from PySide6.QtCore import (
    QObject,
    QPointF,
    QRectF,
    QRunnable,
    QMutex,
    QMutexLocker,
    Qt,
    QThreadPool,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QAction,
    QColor,
    QImage,
    QImageReader,
    QKeySequence,
    QPainter,
    QPalette,
)
from PySide6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenuBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from _shared import (
    load_config,
    get_page_num,
    parse_page_sequence,
)

config = load_config()

# ============================================================================
# Image loading
# ============================================================================

class ImageLoadSignals(QObject):
    loaded = Signal(int, QImage)

class ImageLoadTask(QRunnable):
    def __init__(self, index, path):
        super().__init__()

        self.index = index
        self.path = path
        self.signals = ImageLoadSignals()

    def run(self):
        reader = QImageReader(str(self.path))
        reader.setAutoTransform(True)

        image = reader.read()

        if not image.isNull():
            # Detach from the reader/thread.
            image = image.copy()

        self.signals.loaded.emit(self.index, image)

class ImageCache(QObject):
    image_ready = Signal(int)

    def __init__(self, paths, max_images=60):
        super().__init__()

        self.paths = paths
        self.max_images = max_images

        self.images = OrderedDict()
        self.loading = set()

        self.mutex = QMutex()
        self.thread_pool = QThreadPool.globalInstance()

    def get(self, index):
        if index is None:
            return None

        with QMutexLocker(self.mutex):
            image = self.images.get(index)

            if image is not None:
                self.images.move_to_end(index)

            return image

    def has(self, index):
        if index is None:
            return True

        with QMutexLocker(self.mutex):
            return index in self.images

    def request(self, index):
        if index is None:
            return

        if index < 0 or index >= len(self.paths):
            return

        with QMutexLocker(self.mutex):
            if index in self.images:
                self.images.move_to_end(index)
                return

            if index in self.loading:
                return

            self.loading.add(index)

        task = ImageLoadTask(
            index,
            self.paths[index],
        )

        task.signals.loaded.connect(
            self._image_loaded
        )

        self.thread_pool.start(task)

    def _image_loaded(self, index, image):
        with QMutexLocker(self.mutex):
            self.loading.discard(index)

            if not image.isNull():
                self.images[index] = image
                self.images.move_to_end(index)

                while len(self.images) > self.max_images:
                    self.images.popitem(last=False)

        self.image_ready.emit(index)

# ============================================================================
# Progress bar
# ============================================================================

class BookProgressBar(QWidget):
    clicked = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.position = 0.0

        self.setFixedHeight(4)
        self.setMinimumHeight(4)
        self.setMaximumHeight(4)

        self.setCursor(Qt.PointingHandCursor)

        self.setStyleSheet(
            """
            QWidget {
                background: white;
            }
            """
        )

    def set_position(self, position):
        self.position = max(
            0.0,
            min(1.0, float(position)),
        )

        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.width() > 0:
                position = (
                    event.position().x()
                    / self.width()
                )

                self.clicked.emit(
                    max(0.0, min(1.0, position))
                )

            event.accept()
            return

        super().mousePressEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)

        try:
            painter.setRenderHint(
                QPainter.Antialiasing,
                False,
            )

            painter.fillRect(
                self.rect(),
                Qt.white,
            )

            width = int(
                self.width() * self.position
            )

            if width > 0:
                painter.fillRect(
                    0,
                    0,
                    width,
                    self.height(),
                    Qt.black,
                )

        finally:
            painter.end()

# ============================================================================
# Book canvas
# ============================================================================

class BookCanvas(QWidget):
    def __init__(self, book_viewer, parent=None):
        super().__init__(parent)

        self.book_viewer = book_viewer

        # The currently displayed images.
        #
        # These are deliberately kept until the replacement spread is
        # completely ready. This prevents a black/empty frame when manually
        # changing pages.
        self.left_image = None
        self.right_image = None

        self.zoom = 1.0
        self.pan = QPointF(0.0, 0.0)

        self.dragging = False
        self.last_mouse_position = QPointF()

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        # Prevent the widget background from being erased before painting.
        self.setAttribute(
            Qt.WA_OpaquePaintEvent,
            True,
        )

    # ------------------------------------------------------------------
    # Images
    # ------------------------------------------------------------------

    def set_images(self, left_image, right_image):
        self.left_image = left_image
        self.right_image = right_image

        self.update()

    # ------------------------------------------------------------------
    # Background
    # ------------------------------------------------------------------

    def background_color(self):
        if self.book_viewer.is_dark_mode:
            # 95% black + 5% white.
            value = round(255 * 0.05)
        else:
            # 95% white + 5% black.
            value = round(255 * 0.95)

        return QColor(
            value,
            value,
            value,
        )

    # ------------------------------------------------------------------
    # Zoom
    # ------------------------------------------------------------------

    def fit_pages(self):
        if (
            self.left_image is None
            and self.right_image is None
        ):
            return

        available_width = max(
            1,
            self.width(),
        )

        available_height = max(
            1,
            self.height(),
        )

        left_width = (
            self.left_image.width()
            if self.left_image is not None
            else 0
        )

        left_height = (
            self.left_image.height()
            if self.left_image is not None
            else 0
        )

        right_width = (
            self.right_image.width()
            if self.right_image is not None
            else 0
        )

        right_height = (
            self.right_image.height()
            if self.right_image is not None
            else 0
        )

        combined_width = (
            left_width + right_width
        )

        combined_height = max(
            left_height,
            right_height,
        )

        if (
            combined_width <= 0
            or combined_height <= 0
        ):
            return

        scale_x = (
            available_width
            / combined_width
        )

        scale_y = (
            available_height
            / combined_height
        )

        self.zoom = min(
            scale_x,
            scale_y,
        )

        self.pan = QPointF(
            0.0,
            0.0,
        )

        self.update()

    def set_zoom(self, zoom, anchor=None):
        old_zoom = self.zoom

        zoom = max(
            0.05,
            min(10.0, float(zoom)),
        )

        if abs(zoom - old_zoom) < 1e-9:
            return

        if anchor is None:
            anchor = QPointF(
                self.width() / 2.0,
                self.height() / 2.0,
            )

        # Convert the cursor position into image/world coordinates before
        # changing the zoom.
        world_x = (
            anchor.x() - self.pan.x()
        ) / old_zoom

        world_y = (
            anchor.y() - self.pan.y()
        ) / old_zoom

        self.zoom = zoom

        # Reposition so the same image point remains directly under the
        # mouse pointer.
        self.pan = QPointF(
            anchor.x()
            - world_x * self.zoom,

            anchor.y()
            - world_y * self.zoom,
        )

        self.update()

    def zoom_in(self, anchor=None):
        self.set_zoom(
            self.zoom * 1.15,
            anchor,
        )

    def zoom_out(self, anchor=None):
        self.set_zoom(
            self.zoom / 1.15,
            anchor,
        )

    # ------------------------------------------------------------------
    # Mouse
    # ------------------------------------------------------------------

    def mousePressEvent(self, event):
        self.book_viewer.handle_mouse_move(
            event.position()
        )

        if event.button() == Qt.LeftButton:
            self.dragging = True

            self.last_mouse_position = (
                event.position()
            )

            self.setCursor(
                Qt.ClosedHandCursor
            )

            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        self.book_viewer.handle_mouse_move(
            event.position()
        )

        if self.dragging:
            current = event.position()

            delta = (
                current
                - self.last_mouse_position
            )

            self.pan += delta

            self.last_mouse_position = current

            self.update()

            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False

            self.setCursor(
                Qt.ArrowCursor
            )

            event.accept()
            return

        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.book_viewer.fit_pages()

            event.accept()
            return

        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()

        if delta == 0:
            event.ignore()
            return

        # Ctrl + wheel = zoom.
        if event.modifiers() & Qt.ControlModifier:
            if delta > 0:
                self.zoom_in(
                    event.position()
                )
            else:
                self.zoom_out(
                    event.position()
                )

            event.accept()
            return

        # Normal wheel = navigation.
        #
        # Manual navigation explicitly stops autoplay.
        if delta > 0:
            self.book_viewer.previous_spread()
        else:
            self.book_viewer.next_spread()

        event.accept()

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return:
            self.book_viewer.fit_pages()
            event.accept()
            return

        if event.key() == Qt.Key_Enter:
            self.book_viewer.fit_pages()
            event.accept()
            return

        super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def _display_image(self, image):
        if image is None:
            return None

        if not self.book_viewer.is_dark_mode:
            return image

        # Only the currently visible images are inverted. The cache retains
        # the original high-quality image, avoiding duplicate cache entries.
        inverted = image.copy()

        inverted.invertPixels(
            QImage.InvertRgb
        )

        return inverted

    def paintEvent(self, event):
        painter = QPainter(self)

        try:
            painter.setRenderHint(
                QPainter.SmoothPixmapTransform,
                False,
            )

            # Light/dark gray page surround.
            painter.fillRect(
                self.rect(),
                self.background_color(),
            )

            if (
                self.left_image is None
                and self.right_image is None
            ):
                return

            left_image = self._display_image(
                self.left_image
            )

            right_image = self._display_image(
                self.right_image
            )

            images = []

            if left_image is not None:
                images.append(
                    ("left", left_image)
                )

            if right_image is not None:
                images.append(
                    ("right", right_image)
                )

            if not images:
                return

            total_width = sum(
                image.width()
                for _, image in images
            )

            max_height = max(
                image.height()
                for _, image in images
            )

            if (
                total_width <= 0
                or max_height <= 0
            ):
                return

            scaled_width = (
                total_width * self.zoom
            )

            scaled_height = (
                max_height * self.zoom
            )

            # Center the complete two-page image as one unit.
            origin_x = (
                (
                    self.width()
                    - scaled_width
                )
                / 2.0
                + self.pan.x()
            )

            origin_y = (
                (
                    self.height()
                    - scaled_height
                )
                / 2.0
                + self.pan.y()
            )

            x = origin_x

            for _, image in images:
                image_width = (
                    image.width()
                    * self.zoom
                )

                image_height = (
                    image.height()
                    * self.zoom
                )

                y = (
                    origin_y
                    + (
                        scaled_height
                        - image_height
                    )
                    / 2.0
                )

                if abs(self.zoom - 1.0) < 1e-9:
                    # Native-pixel rendering at 1:1.
                    painter.drawImage(
                        int(round(x)),
                        int(round(y)),
                        image,
                    )
                else:
                    target = QRectF(
                        x,
                        y,
                        image_width,
                        image_height,
                    )

                    painter.drawImage(
                        target,
                        image,
                    )

                x += image_width

        finally:
            painter.end()

# ============================================================================
# Main window
# ============================================================================

class BookViewer(QMainWindow):
    def __init__(
        self,
        source_dir,
        page_spec=None,
    ):
        super().__init__()

        self.source_dir = Path(
            source_dir
        )

        self.setWindowTitle(
            f"Book Viewer — "
            f"{self.source_dir.name}"
        )

        self.resize(
            1400,
            900,
        )

        # --------------------------------------------------------------
        # Playback
        # --------------------------------------------------------------

        self.page_time = 0.2
        self.preload_spreads = 20
        self.max_cache_images = 60

        self.playing = False
        self.current_spread = 0

        # --------------------------------------------------------------
        # Fullscreen
        # --------------------------------------------------------------

        self._fullscreen = False

        self._chrome_timer = QTimer(self)
        self._chrome_timer.setSingleShot(True)
        self._chrome_timer.timeout.connect(
            self._hide_fullscreen_chrome
        )

        # --------------------------------------------------------------
        # Initial fit
        # --------------------------------------------------------------

        self._initial_fit_pending = True

        # --------------------------------------------------------------
        # Theme
        # --------------------------------------------------------------

        self.is_dark_mode = (
            self._detect_dark_mode()
        )

        # --------------------------------------------------------------
        # Find source images
        # --------------------------------------------------------------

        self.paths = self._find_paths()

        if not self.paths:
            extension = (
                str(config.scan_format)
                .lower()
                .lstrip(".")
            )

            raise RuntimeError(
                f"No .{extension} images found in "
                f"{self.source_dir}"
            )

        # --------------------------------------------------------------
        # Select content pages
        # --------------------------------------------------------------

        self.page_indices = (
            self._build_page_indices(
                page_spec
            )
        )

        if not self.page_indices:
            raise RuntimeError(
                "The page specification did not "
                "select any pages."
            )

        # --------------------------------------------------------------
        # Build spreads
        # --------------------------------------------------------------

        self.spreads = (
            self._build_spreads(
                self.page_indices
            )
        )

        if not self.spreads:
            raise RuntimeError(
                "No spreads could be created."
            )

        # --------------------------------------------------------------
        # Cache
        # --------------------------------------------------------------

        self.cache = ImageCache(
            self.paths,
            max_images=self.max_cache_images,
        )

        self.cache.image_ready.connect(
            self._image_ready
        )

        # --------------------------------------------------------------
        # Canvas
        # --------------------------------------------------------------

        self.canvas = BookCanvas(
            self,
            self,
        )

        # --------------------------------------------------------------
        # Progress
        # --------------------------------------------------------------

        self.progress_bar = (
            BookProgressBar()
        )

        self.progress_bar.clicked.connect(
            self.seek_progress
        )

        # --------------------------------------------------------------
        # Central widget
        # --------------------------------------------------------------

        central = QWidget()

        central_layout = QVBoxLayout(
            central
        )

        central_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        central_layout.setSpacing(0)

        central_layout.addWidget(
            self.canvas,
            1,
        )

        central_layout.addWidget(
            self.progress_bar,
            0,
        )

        self.setCentralWidget(
            central
        )

        # --------------------------------------------------------------
        # Menu
        # --------------------------------------------------------------

        self._build_menu()

        # --------------------------------------------------------------
        # Playback timer
        # --------------------------------------------------------------

        self.play_timer = QTimer(
            self
        )

        self.play_timer.timeout.connect(
            self._advance_autoplay
        )

        self._update_page_time()

        # --------------------------------------------------------------
        # Initial spread
        # --------------------------------------------------------------

        self._show_current_spread()

        # --------------------------------------------------------------
        # Start fullscreen
        # --------------------------------------------------------------

        QTimer.singleShot(
            0,
            self._start_fullscreen,
        )

    # ==================================================================
    # Theme
    # ==================================================================

    def _detect_dark_mode(self):
        try:
            style_hints = (
                QApplication.styleHints()
            )

            color_scheme = (
                style_hints.colorScheme()
            )

            if color_scheme == Qt.ColorScheme.Dark:
                return True

            if color_scheme == Qt.ColorScheme.Light:
                return False
        except Exception:
            pass

        # Fallback for older Qt/platform combinations.
        palette = QApplication.palette()

        window_color = palette.color(
            QPalette.Window
        )

        return (
            window_color.lightness()
            < 128
        )

    # ==================================================================
    # Source images
    # ==================================================================

    def _find_paths(self):
        extension = (
            str(config.scan_format)
            .lower()
            .lstrip(".")
        )

        return sorted(
            [
                path
                for path in self.source_dir.iterdir()
                if (
                    path.is_file()
                    and path.suffix.lower().lstrip(".")
                    == extension
                )
            ],
            key=self._page_sort_key,
        )

    def _page_sort_key(self, path):
        try:
            return (
                0,
                get_page_num(path),
            )
        except Exception:
            return (
                1,
                path.name.lower(),
            )

    # ==================================================================
    # Page selection
    # ==================================================================

    def _build_page_indices(self, page_spec):
        num_pages = int(
            config.num_pages
        )

        if page_spec:
            requested_pages = (
                parse_page_sequence(
                    page_spec,
                    num_pages,
                )
            )
        else:
            requested_pages = list(
                range(
                    1,
                    num_pages + 1,
                )
            )

        # Map actual page number -> source image index.
        page_map = {}

        for index, path in enumerate(
            self.paths
        ):
            try:
                number = get_page_num(
                    path
                )
            except Exception:
                continue

            if (
                1 <= number <= num_pages
                and number not in page_map
            ):
                page_map[number] = index

        result = []

        for page_number in requested_pages:
            try:
                page_number = int(
                    page_number
                )
            except Exception:
                continue

            if not (
                1 <= page_number <= num_pages
            ):
                continue

            source_index = page_map.get(
                page_number
            )

            if source_index is not None:
                result.append(
                    source_index
                )

        return result

    # ==================================================================
    # Spread construction
    # ==================================================================

    def _page_number_for_index(self, index):
        try:
            return int(
                get_page_num(
                    self.paths[index]
                )
            )
        except Exception:
            return index + 1

    def _build_spreads(self, page_indices):
        """
        Build facing-page spreads from the selected content pages.

        Rules:

          odd page  -> right
          even page -> left

        A normal even/odd pair is joined:
            2 + 3
            4 + 5
            6 + 7
            ...

        A page without its facing partner is displayed alone on the
        appropriate side.

        Page 1 is therefore alone on the right.
        """

        spreads = []

        i = 0

        while i < len(page_indices):
            current_index = (
                page_indices[i]
            )

            current_number = (
                self._page_number_for_index(
                    current_index
                )
            )

            # ----------------------------------------------------------
            # Odd page = right page.
            #
            # If the next selected page is the immediately following
            # even/odd partner, keep it together as one spread.
            # ----------------------------------------------------------

            if current_number % 2 == 1:
                right_index = current_index
                left_index = None

                if i + 1 < len(page_indices):
                    next_index = (
                        page_indices[i + 1]
                    )

                    next_number = (
                        self._page_number_for_index(
                            next_index
                        )
                    )

                    # An odd page cannot normally be paired forward.
                    # It can only be a standalone right page.
                    #
                    # This also handles page 1.
                    if (
                        next_number
                        == current_number - 1
                    ):
                        left_index = next_index

                spreads.append(
                    {
                        "left": left_index,
                        "right": right_index,
                    }
                )

                i += 1
                continue

            # ----------------------------------------------------------
            # Even page = left page.
            #
            # Pair it with the immediately following odd page if that
            # page was selected.
            # ----------------------------------------------------------

            left_index = current_index
            right_index = None

            if i + 1 < len(page_indices):
                next_index = (
                    page_indices[i + 1]
                )

                next_number = (
                    self._page_number_for_index(
                        next_index
                    )
                )

                if (
                    next_number
                    == current_number + 1
                ):
                    right_index = next_index
                    i += 2
                else:
                    i += 1
            else:
                i += 1

            spreads.append(
                {
                    "left": left_index,
                    "right": right_index,
                }
            )

        # The odd-page handling above intentionally keeps page 1 alone
        # on the right and lets normal 2+3, 4+5, ... pairs be formed by
        # the even-page branch.

        # Rebuild with a simpler deterministic pass to guarantee correct
        # ordering for arbitrary page specifications.
        spreads = []

        i = 0

        while i < len(page_indices):
            index = page_indices[i]
            number = (
                self._page_number_for_index(
                    index
                )
            )

            # Odd page: right side, unless it is immediately preceded
            # by the selected even page. That preceding page would have
            # consumed it already.
            if number % 2 == 1:
                spreads.append(
                    {
                        "left": None,
                        "right": index,
                    }
                )
                i += 1
                continue

            # Even page: left side.
            left_index = index
            right_index = None

            if i + 1 < len(page_indices):
                candidate = (
                    page_indices[i + 1]
                )

                candidate_number = (
                    self._page_number_for_index(
                        candidate
                    )
                )

                if (
                    candidate_number
                    == number + 1
                ):
                    right_index = candidate
                    i += 2
                else:
                    i += 1
            else:
                i += 1

            spreads.append(
                {
                    "left": left_index,
                    "right": right_index,
                }
            )

        # Correct page-1/odd-page ordering:
        #
        # If the sequence is [1, 2, 3, 4], the desired result is:
        #   [right=1]
        #   [left=2,right=3]
        #   [left=4]
        #
        # The deterministic pass above already gives exactly that.

        return spreads

    # ==================================================================
    # Menu
    # ==================================================================

    def _build_menu(self):
        self.menu_bar = QMenuBar(
            self
        )

        self.setMenuBar(
            self.menu_bar
        )

        # --------------------------------------------------------------
        # File
        # --------------------------------------------------------------

        file_menu = self.menu_bar.addMenu(
            "File"
        )

        quit_action = QAction(
            "Quit",
            self,
        )

        quit_action.setShortcut(
            QKeySequence("Q")
        )

        quit_action.triggered.connect(
            self.close
        )

        file_menu.addAction(
            quit_action
        )

        # --------------------------------------------------------------
        # View
        # --------------------------------------------------------------

        view_menu = self.menu_bar.addMenu(
            "View"
        )

        fullscreen_action = QAction(
            "Fullscreen",
            self,
        )

        fullscreen_action.setShortcut(
            QKeySequence("F")
        )

        fullscreen_action.triggered.connect(
            self.toggle_fullscreen
        )

        view_menu.addAction(
            fullscreen_action
        )

        fit_action = QAction(
            "Fit pages",
            self,
        )

        fit_action.setShortcut(
            QKeySequence(Qt.Key_Return)
        )

        fit_action.triggered.connect(
            self.fit_pages
        )

        view_menu.addAction(
            fit_action
        )

        # --------------------------------------------------------------
        # Playback
        # --------------------------------------------------------------

        playback_menu = (
            self.menu_bar.addMenu(
                "Playback"
            )
        )

        self.play_action = QAction(
            "Play / Pause",
            self,
        )

        self.play_action.setShortcut(
            QKeySequence(Qt.Key_Space)
        )

        self.play_action.triggered.connect(
            self.toggle_playback
        )

        playback_menu.addAction(
            self.play_action
        )

        settings_action = QAction(
            "Settings...",
            self,
        )

        settings_action.triggered.connect(
            self.show_playback_settings
        )

        playback_menu.addAction(
            settings_action
        )

        # --------------------------------------------------------------
        # Top-right controls.
        #
        # The old status bar has intentionally been removed.
        # --------------------------------------------------------------

        top_controls = QWidget()

        top_layout = QHBoxLayout(
            top_controls
        )

        top_layout.setContentsMargins(
            4,
            0,
            4,
            0,
        )

        top_layout.setSpacing(4)

        self.status_label = QLabel()

        self.status_label.setMinimumWidth(
            150
        )

        self.status_label.setAlignment(
            Qt.AlignVCenter
            | Qt.AlignRight
        )

        previous_button = QPushButton(
            "◀"
        )

        previous_button.setFixedWidth(
            28
        )

        previous_button.clicked.connect(
            self.previous_spread
        )

        self.current_page_edit = (
            QLineEdit()
        )

        self.current_page_edit.setFixedWidth(
            70
        )

        self.current_page_edit.setAlignment(
            Qt.AlignCenter
        )

        self.current_page_edit.returnPressed.connect(
            self._page_edit_return_pressed
        )

        next_button = QPushButton(
            "▶"
        )

        next_button.setFixedWidth(
            28
        )

        next_button.clicked.connect(
            self.next_spread
        )

        top_layout.addWidget(
            self.status_label
        )

        top_layout.addWidget(
            previous_button
        )

        top_layout.addWidget(
            self.current_page_edit
        )

        top_layout.addWidget(
            next_button
        )

        self.menu_bar.setCornerWidget(
            top_controls,
            Qt.TopRightCorner,
        )

    # ==================================================================
    # Current spread
    # ==================================================================

    def _current_spread_is_loaded(self):
        spread = self.spreads[
            self.current_spread
        ]

        if (
            spread["left"] is not None
            and not self.cache.has(
                spread["left"]
            )
        ):
            return False

        if (
            spread["right"] is not None
            and not self.cache.has(
                spread["right"]
            )
        ):
            return False

        return True

    def _get_current_images(self):
        spread = self.spreads[
            self.current_spread
        ]

        left_image = None
        right_image = None

        if spread["left"] is not None:
            left_image = self.cache.get(
                spread["left"]
            )

        if spread["right"] is not None:
            right_image = self.cache.get(
                spread["right"]
            )

        return (
            left_image,
            right_image,
        )

    def _show_current_spread(self):
        if not self.spreads:
            return

        self.current_spread = max(
            0,
            min(
                self.current_spread,
                len(self.spreads) - 1,
            ),
        )

        spread = self.spreads[
            self.current_spread
        ]

        # --------------------------------------------------------------
        # Request current images first.
        # --------------------------------------------------------------

        if spread["left"] is not None:
            self.cache.request(
                spread["left"]
            )

        if spread["right"] is not None:
            self.cache.request(
                spread["right"]
            )

        # --------------------------------------------------------------
        # Bidirectional preload.
        # --------------------------------------------------------------

        self._preload_around_current()

        # --------------------------------------------------------------
        # Atomic display update.
        #
        # Do NOT clear the canvas when the new images aren't ready.
        # Keeping the previous frame here eliminates manual-navigation
        # flicker.
        # --------------------------------------------------------------

        if self._current_spread_is_loaded():
            left_image, right_image = (
                self._get_current_images()
            )

            self.canvas.set_images(
                left_image,
                right_image,
            )

            if self._initial_fit_pending:
                self.canvas.fit_pages()
                self._initial_fit_pending = False

        # --------------------------------------------------------------
        # UI can update immediately.
        # --------------------------------------------------------------

        self._update_page_display()
        self._update_progress()
        self._update_status()

    # ==================================================================
    # Preloading
    # ==================================================================

    def _preload_around_current(self):
        current = self.current_spread

        order = [current]

        for distance in range(
            1,
            self.preload_spreads + 1,
        ):
            backward = (
                current - distance
            )

            forward = (
                current + distance
            )

            if backward >= 0:
                order.append(
                    backward
                )

            if (
                forward
                < len(self.spreads)
            ):
                order.append(
                    forward
                )

        for spread_index in order:
            spread = self.spreads[
                spread_index
            ]

            if spread["left"] is not None:
                self.cache.request(
                    spread["left"]
                )

            if spread["right"] is not None:
                self.cache.request(
                    spread["right"]
                )

    # ==================================================================
    # Manual navigation
    # ==================================================================

    def _stop_autoplay_for_manual_navigation(
        self
    ):
        if self.playing:
            self._stop_playback()

    def next_spread(self):
        """
        Manual next navigation.

        Manual navigation always stops autoplay and wraps:
            last -> first
        """

        self._stop_autoplay_for_manual_navigation()

        if not self.spreads:
            return

        if (
            self.current_spread
            >= len(self.spreads) - 1
        ):
            self.current_spread = 0
        else:
            self.current_spread += 1

        self._show_current_spread()

    def previous_spread(self):
        """
        Manual previous navigation.

        Manual navigation always stops autoplay and wraps:
            first -> last
        """

        self._stop_autoplay_for_manual_navigation()

        if not self.spreads:
            return

        if self.current_spread <= 0:
            self.current_spread = (
                len(self.spreads) - 1
            )
        else:
            self.current_spread -= 1

        self._show_current_spread()

    # ==================================================================
    # Page number
    # ==================================================================

    def _visible_page_index(self):
        spread = self.spreads[
            self.current_spread
        ]

        if spread["left"] is not None:
            return spread["left"]

        return spread["right"]

    def _visible_page_number(self):
        index = (
            self._visible_page_index()
        )

        if index is None:
            return ""

        return str(
            self._page_number_for_index(
                index
            )
        )

    def _update_page_display(self):
        self.current_page_edit.setText(
            self._visible_page_number()
        )

    def _page_edit_return_pressed(self):
        text = (
            self.current_page_edit.text()
            .strip()
        )

        if not text:
            self._update_page_display()
            return

        try:
            page_number = int(text)
        except ValueError:
            self._update_page_display()
            return

        self.go_to_page(
            page_number
        )

    def go_to_page(self, page_number):
        # Page-number entry is manual navigation, so it stops autoplay.
        self._stop_autoplay_for_manual_navigation()

        target_index = None

        for index in self.page_indices:
            if (
                self._page_number_for_index(
                    index
                )
                == page_number
            ):
                target_index = index
                break

        if target_index is None:
            # If the exact page wasn't selected, find the closest
            # selected content page.
            if not self.page_indices:
                return

            target_index = min(
                self.page_indices,
                key=lambda index: abs(
                    self._page_number_for_index(
                        index
                    )
                    - page_number
                ),
            )

        # Find the spread containing the target page.
        for spread_index, spread in enumerate(
            self.spreads
        ):
            if target_index in (
                spread["left"],
                spread["right"],
            ):
                self.current_spread = (
                    spread_index
                )
                break

        self._show_current_spread()

    # ==================================================================
    # Progress seeking
    # ==================================================================

    def seek_progress(self, position):
        """
        Click on the progress bar to seek to a spread.

        This is manual navigation and therefore stops autoplay.
        """

        self._stop_autoplay_for_manual_navigation()

        if len(self.spreads) <= 1:
            target = 0
        else:
            target = round(
                position
                * (len(self.spreads) - 1)
            )

        target = max(
            0,
            min(
                target,
                len(self.spreads) - 1,
            ),
        )

        self.current_spread = target

        self._show_current_spread()

    # ==================================================================
    # Fit
    # ==================================================================

    def fit_pages(self):
        self._stop_autoplay_for_manual_navigation()

        # If the current images are not loaded yet, request them.
        self._preload_around_current()

        if self._current_spread_is_loaded():
            self.canvas.fit_pages()

    # ==================================================================
    # Cache notifications
    # ==================================================================

    def _image_ready(self, index):
        spread = self.spreads[
            self.current_spread
        ]

        if index in (
            spread["left"],
            spread["right"],
        ):
            if self._current_spread_is_loaded():
                left_image, right_image = (
                    self._get_current_images()
                )

                # Atomic replacement of the visible frame.
                self.canvas.set_images(
                    left_image,
                    right_image,
                )

                if self._initial_fit_pending:
                    self.canvas.fit_pages()
                    self._initial_fit_pending = False

        if self.playing:
            self._maybe_start_autoplay_timer()

    # ==================================================================
    # Autoplay
    # ==================================================================

    def toggle_playback(self):
        if not self.spreads:
            return

        if self.playing:
            self._stop_playback()
            return

        # If autoplay previously reached the last spread, pressing Space
        # again restarts it from the first spread.
        if (
            self.current_spread
            >= len(self.spreads) - 1
        ):
            self.current_spread = 0
            self._show_current_spread()

        self._start_playback()

    def _start_playback(self):
        if self.playing:
            return

        self.playing = True

        self._update_status()

        self._maybe_start_autoplay_timer()

    def _stop_playback(self):
        self.playing = False
        self.play_timer.stop()

        self._update_status()

    def _maybe_start_autoplay_timer(self):
        if not self.playing:
            return

        if (
            self.current_spread
            >= len(self.spreads) - 1
        ):
            self._stop_playback()
            return

        if not self._current_spread_is_loaded():
            self._preload_around_current()
            return

        next_index = (
            self.current_spread + 1
        )

        if next_index < len(self.spreads):
            next_spread = (
                self.spreads[next_index]
            )

            if (
                next_spread["left"]
                is not None
            ):
                self.cache.request(
                    next_spread["left"]
                )

            if (
                next_spread["right"]
                is not None
            ):
                self.cache.request(
                    next_spread["right"]
                )

        if not self.play_timer.isActive():
            self.play_timer.start(
                max(
                    1,
                    int(
                        self.page_time
                        * 1000
                    ),
                )
            )

    def _advance_autoplay(self):
        if not self.playing:
            return

        # Autoplay stops at the last spread.
        if (
            self.current_spread
            >= len(self.spreads) - 1
        ):
            self._stop_playback()
            return

        next_index = (
            self.current_spread + 1
        )

        next_spread = (
            self.spreads[next_index]
        )

        # Wait until all images for the next spread are ready.
        if (
            next_spread["left"]
            is not None
            and not self.cache.has(
                next_spread["left"]
            )
        ):
            self.play_timer.stop()

            self.cache.request(
                next_spread["left"]
            )

            if (
                next_spread["right"]
                is not None
            ):
                self.cache.request(
                    next_spread["right"]
                )

            return

        if (
            next_spread["right"]
            is not None
            and not self.cache.has(
                next_spread["right"]
            )
        ):
            self.play_timer.stop()

            self.cache.request(
                next_spread["right"]
            )

            return

        self.current_spread = next_index

        # Because the next spread is fully cached, the visible image
        # changes atomically without clearing the canvas first.
        self._show_current_spread()

        self.play_timer.start(
            max(
                1,
                int(
                    self.page_time
                    * 1000
                ),
            )
        )

    def _update_page_time(self):
        self.play_timer.setInterval(
            max(
                1,
                int(
                    self.page_time
                    * 1000
                ),
            )
        )

    # ==================================================================
    # Playback settings
    # ==================================================================

    def show_playback_settings(self):
        dialog = PlaybackSettingsDialog(
            self.page_time,
            self.preload_spreads,
            self,
        )

        if dialog.exec() != (
            QDialog.Accepted
        ):
            return

        self.page_time = (
            dialog.page_time_spin.value()
        )

        self.preload_spreads = (
            dialog.preload_spin.value()
        )

        self._update_page_time()

        self._preload_around_current()

        if self.playing:
            self._maybe_start_autoplay_timer()

    # ==================================================================
    # Progress/status
    # ==================================================================

    def _update_progress(self):
        if len(self.spreads) <= 1:
            position = 1.0
        else:
            position = (
                self.current_spread
                / (len(self.spreads) - 1)
            )

        self.progress_bar.set_position(
            position
        )

    def _update_status(self):
        page = (
            self._visible_page_number()
        )

        if self.playing:
            state = "Playing"
        else:
            state = "Paused"

        self.status_label.setText(
            f"Page {page} — {state}"
        )

    # ==================================================================
    # Fullscreen
    # ==================================================================

    def _start_fullscreen(self):
        self.showFullScreen()

        self._fullscreen = True

        self._hide_fullscreen_chrome()

        self.canvas.setFocus()

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self._leave_fullscreen()
        else:
            self._enter_fullscreen()

    def _enter_fullscreen(self):
        self.showFullScreen()

        self._fullscreen = True

        self._hide_fullscreen_chrome()

        self.canvas.setFocus()

    def _leave_fullscreen(self):
        self.showNormal()
        self.showMaximized()

        self._fullscreen = False

        self._show_fullscreen_chrome()

        self.canvas.setFocus()

    def _hide_fullscreen_chrome(self):
        if not self.isFullScreen():
            return

        self.menu_bar.hide()

    def _show_fullscreen_chrome(self):
        self.menu_bar.show()

    def handle_mouse_move(self, position):
        if not self.isFullScreen():
            return

        y = position.y()

        top_zone = 50
        bottom_zone = (
            self.height() - 50
        )

        if (
            y <= top_zone
            or y >= bottom_zone
        ):
            self._show_fullscreen_chrome()

            self._chrome_timer.start(
                1800
            )

    # ==================================================================
    # Window events
    # ==================================================================

    def mouseMoveEvent(self, event):
        self.handle_mouse_move(
            event.position()
        )

        super().mouseMoveEvent(
            event
        )

    def resizeEvent(self, event):
        super().resizeEvent(
            event
        )

        if self._initial_fit_pending:
            QTimer.singleShot(
                0,
                self._try_initial_fit,
            )

    def _try_initial_fit(self):
        if not self._initial_fit_pending:
            return

        if self._current_spread_is_loaded():
            self.canvas.fit_pages()

            self._initial_fit_pending = False

    # ==================================================================
    # Keyboard
    # ==================================================================

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()

        # --------------------------------------------------------------
        # Enter = fit pages.
        # --------------------------------------------------------------

        if key in (
            Qt.Key_Return,
            Qt.Key_Enter,
        ):
            self.fit_pages()
            event.accept()
            return

        # --------------------------------------------------------------
        # Ctrl + plus / equal = zoom in.
        # --------------------------------------------------------------

        if (
            modifiers & Qt.ControlModifier
            and key in (
                Qt.Key_Plus,
                Qt.Key_Equal,
            )
        ):
            self.canvas.zoom_in()
            event.accept()
            return

        # --------------------------------------------------------------
        # Ctrl + minus = zoom out.
        # --------------------------------------------------------------

        if (
            modifiers & Qt.ControlModifier
            and key == Qt.Key_Minus
        ):
            self.canvas.zoom_out()
            event.accept()
            return

        # --------------------------------------------------------------
        # Left/right = manual navigation.
        # --------------------------------------------------------------

        if key == Qt.Key_Left:
            self.previous_spread()
            event.accept()
            return

        if key == Qt.Key_Right:
            self.next_spread()
            event.accept()
            return

        # --------------------------------------------------------------
        # Up/down = zoom.
        # --------------------------------------------------------------

        if key == Qt.Key_Up:
            self.canvas.zoom_in()
            event.accept()
            return

        if key == Qt.Key_Down:
            self.canvas.zoom_out()
            event.accept()
            return

        # --------------------------------------------------------------
        # Space = autoplay.
        # --------------------------------------------------------------

        if key == Qt.Key_Space:
            self.toggle_playback()
            event.accept()
            return

        # --------------------------------------------------------------
        # F = fullscreen.
        # --------------------------------------------------------------

        if key == Qt.Key_F:
            self.toggle_fullscreen()
            event.accept()
            return

        # --------------------------------------------------------------
        # Q = quit.
        # --------------------------------------------------------------

        if key == Qt.Key_Q:
            self.close()
            event.accept()
            return

        super().keyPressEvent(
            event
        )

# ============================================================================
# Settings dialog
# ============================================================================

class PlaybackSettingsDialog(QWidget):
    """
    Small settings dialog implemented as a QWidget so the viewer remains
    lightweight.
    """

    def __init__(
        self,
        page_time,
        preload_spreads,
        parent=None,
    ):
        super().__init__(
            parent,
            Qt.Dialog,
        )

        self.setWindowTitle(
            "Playback settings"
        )

        layout = QFormLayout(
            self
        )

        self.page_time_spin = (
            QDoubleSpinBox()
        )

        self.page_time_spin.setRange(
            0.01,
            60.0,
        )

        self.page_time_spin.setDecimals(
            2
        )

        self.page_time_spin.setSingleStep(
            0.05
        )

        self.page_time_spin.setValue(
            page_time
        )

        self.page_time_spin.setSuffix(
            " s"
        )

        self.preload_spin = (
            QSpinBox()
        )

        self.preload_spin.setRange(
            0,
            500,
        )

        self.preload_spin.setValue(
            preload_spreads
        )

        layout.addRow(
            "Time per spread:",
            self.page_time_spin,
        )

        layout.addRow(
            "Preload spreads:",
            self.preload_spin,
        )

        buttons = QHBoxLayout()

        ok_button = QPushButton(
            "OK"
        )

        cancel_button = QPushButton(
            "Cancel"
        )

        ok_button.clicked.connect(
            self.accept
        )

        cancel_button.clicked.connect(
            self.reject
        )

        buttons.addWidget(
            cancel_button
        )

        buttons.addWidget(
            ok_button
        )

        layout.addRow(
            buttons
        )

        self._accepted = False

    def accept(self):
        self._accepted = True
        self.close()

    def reject(self):
        self._accepted = False
        self.close()

    def exec(self):
        self.show()
        self.raise_()
        self.activateWindow()

        # Lightweight modal loop.
        loop = QEventLoop()
        self._event_loop = loop

        def finished():
            if loop.isRunning():
                loop.quit()

        self.destroyed.connect(
            finished
        )

        loop.exec()

        return (
            QDialog.Accepted
            if self._accepted
            else QDialog.Rejected
        )

# ============================================================================
# Main
# ============================================================================

def main():
    script_name = Path(
        __file__
    ).name

    parser = argparse.ArgumentParser(
        prog=script_name,
        description=(
            "Scanned book page viewer"
        ),
    )

    parser.add_argument(
        "source_dir",
        help=(
            "Directory containing scanned "
            "page images"
        ),
    )

    parser.add_argument(
        "--pages",
        dest="page_spec",
        metavar="SPEC",
        help=(
            'Content pages to display, '
            'for example "10,20-30". '
            "If omitted, all content pages "
            "are displayed."
        ),
    )

    args = parser.parse_args()

    app = QApplication(
        sys.argv
    )

    try:
        window = BookViewer(
            args.source_dir,
            page_spec=args.page_spec,
        )
    except Exception as exc:
        print(
            f"Error: {exc}",
            file=sys.stderr,
        )
        return 1

    window.show()

    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
```

One correction before you run it: the settings dialog in the script above uses `QEventLoop`, so the import needs to include it. Add it to the existing `PySide6.QtCore` import:

``` python
from PySide6.QtCore import (
    QEventLoop,
    QObject,
    ...
)
```

Everything else is contained in the full script.

## Prompt:

9/10/2026, 6:15:21 AM

this fails with many exceptions

    view-pages.py 070-deskew
    Error: libshiboken: Internal C++ object (PySide6.QtWidgets.QLineEdit) already deleted.
    Error calling Python override of QRunnable::run(): Traceback (most recent call last):
      File "/media/ZYD82805_24TB/root/home/user/src/milahu/hocr-files-template-repo/view-pages.py", line 80, in run
        self.signals.loaded.emit(self.index, image)
        ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^
    RuntimeError: Signal source has been deleted
    Error calling Python override of QRunnable::run(): Traceback (most recent call last):
      File "/media/ZYD82805_24TB/root/home/user/src/milahu/hocr-files-template-repo/view-pages.py", line 80, in run
        self.signals.loaded.emit(self.index, image)
        ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^
    RuntimeError: Signal source has been deleted
    Error calling Python override of QRunnable::run(): Traceback (most recent call last):
      File "/media/ZYD82805_24TB/root/home/user/src/milahu/hocr-files-template-repo/view-pages.py", line 80, in run
        self.signals.loaded.emit(self.index, image)
        ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^
    RuntimeError: Signal source has been deleted
    Error calling Python override of QRunnable::run(): Error calling Python override of QRunnable::run(): Traceback (most recent call last):
      File "/media/ZYD82805_24TB/root/home/user/src/milahu/hocr-files-template-repo/view-pages.py", line 80, in run
        self.signals.loaded.emit(self.index, image)
        ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^
    RuntimeError: Signal source has been deleted
    Traceback (most recent call last):
      File "/media/ZYD82805_24TB/root/home/user/src/milahu/hocr-files-template-repo/view-pages.py", line 80, in run
        self.signals.loaded.emit(self.index, image)
        ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^
    RuntimeError: Signal source has been deleted
    Error calling Python override of QRunnable::run(): Traceback (most recent call last):
      File "/media/ZYD82805_24TB/root/home/user/src/milahu/hocr-files-template-repo/view-pages.py", line 80, in run
        self.signals.loaded.emit(self.index, image)
        ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^
    RuntimeError: Signal source has been deleted
    .........

## Response:

9/10/2026, 6:15:24 AM

You're right. The previous version introduced two lifecycle bugs that are especially bad with PySide6:

1.  The custom `PlaybackSettingsDialog` was incorrectly implemented as a `QWidget` with a homemade modal event loop. That can leave Qt-owned widgets such as `QLineEdit` deleted while Python still references them.
2.  Each `QRunnable` owned its own `QObject` signal source. When the Python/Qt object lifecycle ended, the worker could still be running and then tried to emit through an already-deleted signal object.

I've replaced both mechanisms:

- The settings window is now a normal `QDialog`.
- Worker threads emit through the long-lived `ImageCache` object rather than a per-task signal object.
- The cache is shut down cleanly when the application closes.
- The visible spread is never cleared while waiting for a manually requested image, preventing navigation flicker.
- I also fixed the page-selection/spread construction so `1`, `2-3`, `4-5`, etc. behave correctly with `--pages`.

Here is the complete script.

\*\*document: \*\*

``` python
import argparse
import sys
from collections import OrderedDict
from pathlib import Path

from PySide6.QtCore import (
    QObject,
    QPointF,
    QRectF,
    QRunnable,
    QMutex,
    QMutexLocker,
    Qt,
    QThreadPool,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QAction,
    QColor,
    QImage,
    QImageReader,
    QKeySequence,
    QPainter,
    QPalette,
)
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenuBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from _shared import (
    load_config,
    get_page_num,
    parse_page_sequence,
)

config = load_config()

# ============================================================================
# Image loading
# ============================================================================

class ImageLoadTask(QRunnable):
    """
    Loads one image in a worker thread.

    IMPORTANT:
    The task does not own a QObject containing its signals.

    Instead, it keeps a reference to the long-lived ImageCache and emits
    through ImageCache itself. This prevents:

        RuntimeError: Signal source has been deleted
    """

    def __init__(self, cache, index, path):
        super().__init__()

        self.cache = cache
        self.index = index
        self.path = path

        self.setAutoDelete(True)

    def run(self):
        reader = QImageReader(str(self.path))
        reader.setAutoTransform(True)

        image = reader.read()

        if not image.isNull():
            # Detach the image from the reader/thread.
            image = image.copy()

        # ImageCache is owned by the main window/application and remains
        # alive while the application is running.
        self.cache.image_loaded_from_worker(
            self.index,
            image,
        )

class ImageCache(QObject):
    """
    Thread-safe LRU image cache.

    Worker threads load images, then emit the result through this persistent
    QObject. The connected slots execute safely in the GUI thread.
    """

    image_ready = Signal(int)

    def __init__(
        self,
        paths,
        max_images=60,
        parent=None,
    ):
        super().__init__(parent)

        self.paths = list(paths)
        self.max_images = max_images

        self.images = OrderedDict()
        self.loading = set()

        self.mutex = QMutex()

        self.thread_pool = QThreadPool(self)
        self.thread_pool.setMaxThreadCount(2)

        self.shutting_down = False

    # ------------------------------------------------------------------
    # Cache access
    # ------------------------------------------------------------------

    def get(self, index):
        if index is None:
            return None

        with QMutexLocker(self.mutex):
            image = self.images.get(index)

            if image is not None:
                self.images.move_to_end(index)

            return image

    def has(self, index):
        if index is None:
            return True

        with QMutexLocker(self.mutex):
            return index in self.images

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def request(self, index):
        if index is None:
            return

        if index < 0 or index >= len(self.paths):
            return

        with QMutexLocker(self.mutex):
            if self.shutting_down:
                return

            if index in self.images:
                self.images.move_to_end(index)
                return

            if index in self.loading:
                return

            self.loading.add(index)

        task = ImageLoadTask(
            self,
            index,
            self.paths[index],
        )

        self.thread_pool.start(task)

    def image_loaded_from_worker(
        self,
        index,
        image,
    ):
        """
        Called from worker threads.

        The signal belongs to this persistent ImageCache object, not to the
        temporary QRunnable.
        """

        with QMutexLocker(self.mutex):
            self.loading.discard(index)

            if (
                not self.shutting_down
                and not image.isNull()
            ):
                self.images[index] = image
                self.images.move_to_end(index)

                while (
                    len(self.images)
                    > self.max_images
                ):
                    self.images.popitem(
                        last=False
                    )

        if not self.shutting_down:
            self.image_ready.emit(index)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self):
        with QMutexLocker(self.mutex):
            self.shutting_down = True

        self.thread_pool.clear()
        self.thread_pool.waitForDone()

# ============================================================================
# Progress bar
# ============================================================================

class BookProgressBar(QWidget):
    clicked = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.position = 0.0

        self.setFixedHeight(4)
        self.setMinimumHeight(4)
        self.setMaximumHeight(4)

        self.setCursor(
            Qt.PointingHandCursor
        )

        self.setStyleSheet(
            """
            QWidget {
                background: white;
            }
            """
        )

    def set_position(self, position):
        self.position = max(
            0.0,
            min(1.0, float(position)),
        )

        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.width() > 0:
                position = (
                    event.position().x()
                    / self.width()
                )

                self.clicked.emit(
                    max(
                        0.0,
                        min(1.0, position),
                    )
                )

            event.accept()
            return

        super().mousePressEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)

        try:
            painter.setRenderHint(
                QPainter.Antialiasing,
                False,
            )

            painter.fillRect(
                self.rect(),
                Qt.white,
            )

            width = int(
                self.width()
                * self.position
            )

            if width > 0:
                painter.fillRect(
                    0,
                    0,
                    width,
                    self.height(),
                    Qt.black,
                )

        finally:
            painter.end()

# ============================================================================
# Book canvas
# ============================================================================

class BookCanvas(QWidget):
    def __init__(
        self,
        book_viewer,
        parent=None,
    ):
        super().__init__(parent)

        self.book_viewer = book_viewer

        self.left_image = None
        self.right_image = None

        self.zoom = 1.0
        self.pan = QPointF(
            0.0,
            0.0,
        )

        self.dragging = False
        self.last_mouse_position = (
            QPointF()
        )

        self.setMouseTracking(True)
        self.setFocusPolicy(
            Qt.StrongFocus
        )

        self.setAttribute(
            Qt.WA_OpaquePaintEvent,
            True,
        )

    # ------------------------------------------------------------------
    # Images
    # ------------------------------------------------------------------

    def set_images(
        self,
        left_image,
        right_image,
    ):
        self.left_image = left_image
        self.right_image = right_image
        self.update()

    # ------------------------------------------------------------------
    # Background
    # ------------------------------------------------------------------

    def background_color(self):
        if self.book_viewer.is_dark_mode:
            # 95% black + 5% white.
            value = round(
                255 * 0.05
            )
        else:
            # 95% white + 5% black.
            value = round(
                255 * 0.95
            )

        return QColor(
            value,
            value,
            value,
        )

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit_pages(self):
        if (
            self.left_image is None
            and self.right_image is None
        ):
            return

        available_width = max(
            1,
            self.width(),
        )

        available_height = max(
            1,
            self.height(),
        )

        left_width = (
            self.left_image.width()
            if self.left_image is not None
            else 0
        )

        left_height = (
            self.left_image.height()
            if self.left_image is not None
            else 0
        )

        right_width = (
            self.right_image.width()
            if self.right_image is not None
            else 0
        )

        right_height = (
            self.right_image.height()
            if self.right_image is not None
            else 0
        )

        total_width = (
            left_width + right_width
        )

        total_height = max(
            left_height,
            right_height,
        )

        if (
            total_width <= 0
            or total_height <= 0
        ):
            return

        scale_x = (
            available_width
            / total_width
        )

        scale_y = (
            available_height
            / total_height
        )

        self.zoom = min(
            scale_x,
            scale_y,
        )

        self.pan = QPointF(
            0.0,
            0.0,
        )

        self.update()

    # ------------------------------------------------------------------
    # Zoom
    # ------------------------------------------------------------------

    def set_zoom(
        self,
        zoom,
        anchor=None,
    ):
        old_zoom = self.zoom

        zoom = max(
            0.05,
            min(10.0, float(zoom)),
        )

        if abs(
            zoom - old_zoom
        ) < 1e-9:
            return

        if anchor is None:
            anchor = QPointF(
                self.width() / 2.0,
                self.height() / 2.0,
            )

        # Image/world coordinate underneath cursor.
        world_x = (
            anchor.x()
            - self.pan.x()
        ) / old_zoom

        world_y = (
            anchor.y()
            - self.pan.y()
        ) / old_zoom

        self.zoom = zoom

        # Keep that image point under the cursor.
        self.pan = QPointF(
            anchor.x()
            - world_x * self.zoom,
            anchor.y()
            - world_y * self.zoom,
        )

        self.update()

    def zoom_in(self, anchor=None):
        self.set_zoom(
            self.zoom * 1.15,
            anchor,
        )

    def zoom_out(self, anchor=None):
        self.set_zoom(
            self.zoom / 1.15,
            anchor,
        )

    # ------------------------------------------------------------------
    # Mouse
    # ------------------------------------------------------------------

    def mousePressEvent(self, event):
        self.book_viewer.handle_mouse_move(
            event.position()
        )

        if event.button() == Qt.LeftButton:
            self.dragging = True

            self.last_mouse_position = (
                event.position()
            )

            self.setCursor(
                Qt.ClosedHandCursor
            )

            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        self.book_viewer.handle_mouse_move(
            event.position()
        )

        if self.dragging:
            delta = (
                event.position()
                - self.last_mouse_position
            )

            self.pan += delta

            self.last_mouse_position = (
                event.position()
            )

            self.update()

            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False

            self.setCursor(
                Qt.ArrowCursor
            )

            event.accept()
            return

        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.book_viewer.fit_pages()

            event.accept()
            return

        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()

        if delta == 0:
            event.ignore()
            return

        # Ctrl + wheel = zoom, anchored at mouse.
        if (
            event.modifiers()
            & Qt.ControlModifier
        ):
            if delta > 0:
                self.zoom_in(
                    event.position()
                )
            else:
                self.zoom_out(
                    event.position()
                )

            event.accept()
            return

        # Normal wheel = manual navigation.
        if delta > 0:
            self.book_viewer.previous_spread()
        else:
            self.book_viewer.next_spread()

        event.accept()

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------

    def keyPressEvent(self, event):
        if event.key() in (
            Qt.Key_Return,
            Qt.Key_Enter,
        ):
            self.book_viewer.fit_pages()
            event.accept()
            return

        super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def _display_image(self, image):
        if image is None:
            return None

        if not self.book_viewer.is_dark_mode:
            return image

        # Keep the original cached image untouched.
        result = image.copy()

        result.invertPixels(
            QImage.InvertRgb
        )

        return result

    def paintEvent(self, event):
        painter = QPainter(self)

        try:
            painter.setRenderHint(
                QPainter.SmoothPixmapTransform,
                False,
            )

            # Light gray / dark gray surround.
            painter.fillRect(
                self.rect(),
                self.background_color(),
            )

            if (
                self.left_image is None
                and self.right_image is None
            ):
                return

            left_image = (
                self._display_image(
                    self.left_image
                )
            )

            right_image = (
                self._display_image(
                    self.right_image
                )
            )

            images = []

            if left_image is not None:
                images.append(left_image)

            if right_image is not None:
                images.append(right_image)

            if not images:
                return

            total_width = sum(
                image.width()
                for image in images
            )

            total_height = max(
                image.height()
                for image in images
            )

            if (
                total_width <= 0
                or total_height <= 0
            ):
                return

            scaled_width = (
                total_width * self.zoom
            )

            scaled_height = (
                total_height * self.zoom
            )

            origin_x = (
                (
                    self.width()
                    - scaled_width
                )
                / 2.0
                + self.pan.x()
            )

            origin_y = (
                (
                    self.height()
                    - scaled_height
                )
                / 2.0
                + self.pan.y()
            )

            x = origin_x

            for image in images:
                image_width = (
                    image.width()
                    * self.zoom
                )

                image_height = (
                    image.height()
                    * self.zoom
                )

                y = (
                    origin_y
                    + (
                        scaled_height
                        - image_height
                    )
                    / 2.0
                )

                if abs(
                    self.zoom - 1.0
                ) < 1e-9:
                    # Native resolution at 1:1.
                    painter.drawImage(
                        int(round(x)),
                        int(round(y)),
                        image,
                    )
                else:
                    painter.drawImage(
                        QRectF(
                            x,
                            y,
                            image_width,
                            image_height,
                        ),
                        image,
                    )

                x += image_width

        finally:
            painter.end()

# ============================================================================
# Playback settings dialog
# ============================================================================

class PlaybackSettingsDialog(QDialog):
    def __init__(
        self,
        page_time,
        preload_spreads,
        parent=None,
    ):
        super().__init__(parent)

        self.setWindowTitle(
            "Playback settings"
        )

        layout = QFormLayout(self)

        self.page_time_spin = (
            QDoubleSpinBox()
        )

        self.page_time_spin.setRange(
            0.01,
            60.0,
        )

        self.page_time_spin.setDecimals(
            2
        )

        self.page_time_spin.setSingleStep(
            0.05
        )

        self.page_time_spin.setValue(
            page_time
        )

        self.page_time_spin.setSuffix(
            " s"
        )

        self.preload_spin = (
            QSpinBox()
        )

        self.preload_spin.setRange(
            0,
            500,
        )

        self.preload_spin.setValue(
            preload_spreads
        )

        layout.addRow(
            "Time per spread:",
            self.page_time_spin,
        )

        layout.addRow(
            "Preload spreads:",
            self.preload_spin,
        )

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok
            | QDialogButtonBox.Cancel
        )

        buttons.accepted.connect(
            self.accept
        )

        buttons.rejected.connect(
            self.reject
        )

        layout.addRow(
            buttons
        )

# ============================================================================
# Main window
# ============================================================================

class BookViewer(QMainWindow):
    def __init__(
        self,
        source_dir,
        page_spec=None,
    ):
        super().__init__()

        self.source_dir = Path(
            source_dir
        )

        self.setWindowTitle(
            f"Book Viewer — "
            f"{self.source_dir.name}"
        )

        self.resize(
            1400,
            900,
        )

        # ----------------------------------------------------------------
        # Playback
        # ----------------------------------------------------------------

        self.page_time = 0.2
        self.preload_spreads = 20
        self.max_cache_images = 60

        self.playing = False
        self.current_spread = 0

        # ----------------------------------------------------------------
        # Fullscreen
        # ----------------------------------------------------------------

        self._fullscreen = False

        self._chrome_timer = QTimer(self)
        self._chrome_timer.setSingleShot(True)
        self._chrome_timer.timeout.connect(
            self._hide_fullscreen_chrome
        )

        # ----------------------------------------------------------------
        # Initial fit
        # ----------------------------------------------------------------

        self._initial_fit_pending = True

        # ----------------------------------------------------------------
        # Theme
        # ----------------------------------------------------------------

        self.is_dark_mode = (
            self._detect_dark_mode()
        )

        # ----------------------------------------------------------------
        # Find images
        # ----------------------------------------------------------------

        self.paths = self._find_paths()

        if not self.paths:
            extension = (
                str(config.scan_format)
                .lower()
                .lstrip(".")
            )

            raise RuntimeError(
                f"No .{extension} images found "
                f"in {self.source_dir}"
            )

        # ----------------------------------------------------------------
        # Content pages
        # ----------------------------------------------------------------

        self.page_indices = (
            self._build_page_indices(
                page_spec
            )
        )

        if not self.page_indices:
            raise RuntimeError(
                "No content pages were selected."
            )

        # ----------------------------------------------------------------
        # Spreads
        # ----------------------------------------------------------------

        self.spreads = (
            self._build_spreads(
                self.page_indices
            )
        )

        if not self.spreads:
            raise RuntimeError(
                "No spreads could be created."
            )

        # ----------------------------------------------------------------
        # Cache
        # ----------------------------------------------------------------

        self.cache = ImageCache(
            self.paths,
            max_images=self.max_cache_images,
            parent=self,
        )

        self.cache.image_ready.connect(
            self._image_ready
        )

        # ----------------------------------------------------------------
        # Canvas
        # ----------------------------------------------------------------

        self.canvas = BookCanvas(
            self,
            self,
        )

        # ----------------------------------------------------------------
        # Progress
        # ----------------------------------------------------------------

        self.progress_bar = (
            BookProgressBar()
        )

        self.progress_bar.clicked.connect(
            self.seek_progress
        )

        # ----------------------------------------------------------------
        # Central widget
        # ----------------------------------------------------------------

        central = QWidget(self)

        central_layout = QVBoxLayout(
            central
        )

        central_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        central_layout.setSpacing(0)

        central_layout.addWidget(
            self.canvas,
            1,
        )

        central_layout.addWidget(
            self.progress_bar,
            0,
        )

        self.setCentralWidget(
            central
        )

        # ----------------------------------------------------------------
        # Menu
        # ----------------------------------------------------------------

        self._build_menu()

        # ----------------------------------------------------------------
        # Playback timer
        # ----------------------------------------------------------------

        self.play_timer = QTimer(
            self
        )

        self.play_timer.timeout.connect(
            self._advance_autoplay
        )

        self._update_page_time()

        # ----------------------------------------------------------------
        # Initial display
        # ----------------------------------------------------------------

        self._show_current_spread()

        # ----------------------------------------------------------------
        # Fullscreen
        # ----------------------------------------------------------------

        QTimer.singleShot(
            0,
            self._start_fullscreen,
        )

    # ==================================================================
    # Theme
    # ==================================================================

    def _detect_dark_mode(self):
        try:
            scheme = (
                QApplication.styleHints()
                .colorScheme()
            )

            if scheme == Qt.ColorScheme.Dark:
                return True

            if scheme == Qt.ColorScheme.Light:
                return False

        except Exception:
            pass

        palette = QApplication.palette()

        window_color = palette.color(
            QPalette.Window
        )

        return (
            window_color.lightness()
            < 128
        )

    # ==================================================================
    # Source files
    # ==================================================================

    def _find_paths(self):
        extension = (
            str(config.scan_format)
            .lower()
            .lstrip(".")
        )

        return sorted(
            [
                path
                for path in self.source_dir.iterdir()
                if (
                    path.is_file()
                    and path.suffix.lower().lstrip(".")
                    == extension
                )
            ],
            key=self._page_sort_key,
        )

    def _page_sort_key(self, path):
        try:
            return (
                0,
                get_page_num(path),
            )
        except Exception:
            return (
                1,
                path.name.lower(),
            )

    # ==================================================================
    # Content-page selection
    # ==================================================================

    def _build_page_indices(
        self,
        page_spec,
    ):
        num_pages = int(
            config.num_pages
        )

        # Build page-number -> image-index map.
        page_map = {}

        for index, path in enumerate(
            self.paths
        ):
            try:
                page_number = int(
                    get_page_num(path)
                )
            except Exception:
                continue

            if (
                1 <= page_number <= num_pages
                and page_number not in page_map
            ):
                page_map[
                    page_number
                ] = index

        # No --pages means every content page.
        if page_spec is None:
            requested_pages = list(
                range(
                    1,
                    num_pages + 1,
                )
            )
        else:
            requested_pages = (
                parse_page_sequence(
                    page_spec,
                    num_pages,
                )
            )

        result = []

        for page_number in requested_pages:
            page_number = int(
                page_number
            )

            if not (
                1 <= page_number <= num_pages
            ):
                continue

            index = page_map.get(
                page_number
            )

            if index is not None:
                result.append(index)

        return result

    # ==================================================================
    # Spread construction
    # ==================================================================

    def _page_number_for_index(self, index):
        try:
            return int(
                get_page_num(
                    self.paths[index]
                )
            )
        except Exception:
            return index + 1

    def _build_spreads(
        self,
        page_indices,
    ):
        """
        Construct facing-page spreads.

        The physical page parity determines its side:

            odd  -> right
            even -> left

        Thus the normal complete book is:

            page 1
                [     1]

            pages 2-3
                [2 | 3]

            pages 4-5
                [4 | 5]

            pages 6-7
                [6 | 7]

        A missing partner remains a single page on its natural side.

        This is particularly important for --pages selections. For example:

            --pages "10,20-30"

        gives:

            [10]
            [20 | 21]
            [22 | 23]
            ...
            [30]
        """

        spreads = []

        i = 0

        while i < len(page_indices):
            index = page_indices[i]

            page_number = (
                self._page_number_for_index(
                    index
                )
            )

            # ----------------------------------------------------------
            # Even page = left side.
            # ----------------------------------------------------------

            if page_number % 2 == 0:
                left = index
                right = None

                if i + 1 < len(page_indices):
                    candidate = (
                        page_indices[i + 1]
                    )

                    candidate_number = (
                        self._page_number_for_index(
                            candidate
                        )
                    )

                    # Only pair with the immediate odd page.
                    if (
                        candidate_number
                        == page_number + 1
                    ):
                        right = candidate
                        i += 2
                    else:
                        i += 1
                else:
                    i += 1

                spreads.append(
                    {
                        "left": left,
                        "right": right,
                    }
                )

                continue

            # ----------------------------------------------------------
            # Odd page = right side.
            # ----------------------------------------------------------

            spreads.append(
                {
                    "left": None,
                    "right": index,
                }
            )

            i += 1

        return spreads

    # ==================================================================
    # Menu
    # ==================================================================

    def _build_menu(self):
        self.menu_bar = QMenuBar(
            self
        )

        self.setMenuBar(
            self.menu_bar
        )

        # --------------------------------------------------------------
        # File
        # --------------------------------------------------------------

        file_menu = self.menu_bar.addMenu(
            "File"
        )

        quit_action = QAction(
            "Quit",
            self,
        )

        quit_action.setShortcut(
            QKeySequence("Q")
        )

        quit_action.triggered.connect(
            self.close
        )

        file_menu.addAction(
            quit_action
        )

        # --------------------------------------------------------------
        # View
        # --------------------------------------------------------------

        view_menu = self.menu_bar.addMenu(
            "View"
        )

        fullscreen_action = QAction(
            "Fullscreen",
            self,
        )

        fullscreen_action.setShortcut(
            QKeySequence("F")
        )

        fullscreen_action.triggered.connect(
            self.toggle_fullscreen
        )

        view_menu.addAction(
            fullscreen_action
        )

        fit_action = QAction(
            "Fit pages",
            self,
        )

        fit_action.setShortcut(
            QKeySequence(Qt.Key_Return)
        )

        fit_action.triggered.connect(
            self.fit_pages
        )

        view_menu.addAction(
            fit_action
        )

        # --------------------------------------------------------------
        # Playback
        # --------------------------------------------------------------

        playback_menu = (
            self.menu_bar.addMenu(
                "Playback"
            )
        )

        self.play_action = QAction(
            "Play / Pause",
            self,
        )

        self.play_action.setShortcut(
            QKeySequence(Qt.Key_Space)
        )

        self.play_action.triggered.connect(
            self.toggle_playback
        )

        playback_menu.addAction(
            self.play_action
        )

        settings_action = QAction(
            "Settings...",
            self,
        )

        settings_action.triggered.connect(
            self.show_playback_settings
        )

        playback_menu.addAction(
            settings_action
        )

        # --------------------------------------------------------------
        # Top-right status/navigation.
        #
        # There is deliberately NO QStatusBar.
        # --------------------------------------------------------------

        controls = QWidget()

        controls_layout = QHBoxLayout(
            controls
        )

        controls_layout.setContentsMargins(
            4,
            0,
            4,
            0,
        )

        controls_layout.setSpacing(4)

        self.status_label = QLabel()

        self.status_label.setAlignment(
            Qt.AlignVCenter
            | Qt.AlignRight
        )

        self.status_label.setMinimumWidth(
            150
        )

        previous_button = QPushButton(
            "◀"
        )

        previous_button.setFixedWidth(
            28
        )

        previous_button.clicked.connect(
            self.previous_spread
        )

        self.current_page_edit = (
            QLineEdit()
        )

        self.current_page_edit.setFixedWidth(
            70
        )

        self.current_page_edit.setAlignment(
            Qt.AlignCenter
        )

        self.current_page_edit.returnPressed.connect(
            self._page_edit_return_pressed
        )

        next_button = QPushButton(
            "▶"
        )

        next_button.setFixedWidth(
            28
        )

        next_button.clicked.connect(
            self.next_spread
        )

        controls_layout.addWidget(
            self.status_label
        )

        controls_layout.addWidget(
            previous_button
        )

        controls_layout.addWidget(
            self.current_page_edit
        )

        controls_layout.addWidget(
            next_button
        )

        self.menu_bar.setCornerWidget(
            controls,
            Qt.TopRightCorner,
        )

    # ==================================================================
    # Current spread
    # ==================================================================

    def _current_spread_is_loaded(self):
        spread = self.spreads[
            self.current_spread
        ]

        if (
            spread["left"] is not None
            and not self.cache.has(
                spread["left"]
            )
        ):
            return False

        if (
            spread["right"] is not None
            and not self.cache.has(
                spread["right"]
            )
        ):
            return False

        return True

    def _get_current_images(self):
        spread = self.spreads[
            self.current_spread
        ]

        left_image = None
        right_image = None

        if spread["left"] is not None:
            left_image = self.cache.get(
                spread["left"]
            )

        if spread["right"] is not None:
            right_image = self.cache.get(
                spread["right"]
            )

        return (
            left_image,
            right_image,
        )

    def _show_current_spread(self):
        if not self.spreads:
            return

        self.current_spread = max(
            0,
            min(
                self.current_spread,
                len(self.spreads) - 1,
            ),
        )

        spread = self.spreads[
            self.current_spread
        ]

        # Request the current spread first.
        if spread["left"] is not None:
            self.cache.request(
                spread["left"]
            )

        if spread["right"] is not None:
            self.cache.request(
                spread["right"]
            )

        # Then preload in both directions.
        self._preload_around_current()

        # --------------------------------------------------------------
        # IMPORTANT:
        #
        # Only replace the visible frame when the complete new spread is
        # ready. Otherwise leave the old frame on screen.
        #
        # This is what prevents the black/blank flicker during manual
        # navigation.
        # --------------------------------------------------------------

        if self._current_spread_is_loaded():
            left_image, right_image = (
                self._get_current_images()
            )

            self.canvas.set_images(
                left_image,
                right_image,
            )

            if self._initial_fit_pending:
                self.canvas.fit_pages()

                self._initial_fit_pending = (
                    False
                )

        self._update_page_display()
        self._update_progress()
        self._update_status()

    # ==================================================================
    # Preloading
    # ==================================================================

    def _preload_around_current(self):
        current = self.current_spread

        order = [current]

        for distance in range(
            1,
            self.preload_spreads + 1,
        ):
            backward = (
                current - distance
            )

            forward = (
                current + distance
            )

            if backward >= 0:
                order.append(
                    backward
                )

            if (
                forward
                < len(self.spreads)
            ):
                order.append(
                    forward
                )

        for spread_index in order:
            spread = self.spreads[
                spread_index
            ]

            if spread["left"] is not None:
                self.cache.request(
                    spread["left"]
                )

            if spread["right"] is not None:
                self.cache.request(
                    spread["right"]
                )

    # ==================================================================
    # Manual navigation
    # ==================================================================

    def _stop_autoplay_for_manual_navigation(
        self
    ):
        if self.playing:
            self._stop_playback()

    def next_spread(self):
        """
        Manual navigation.

        Manual navigation always stops autoplay.

        Last -> first.
        """

        self._stop_autoplay_for_manual_navigation()

        if not self.spreads:
            return

        if (
            self.current_spread
            >= len(self.spreads) - 1
        ):
            self.current_spread = 0
        else:
            self.current_spread += 1

        self._show_current_spread()

    def previous_spread(self):
        """
        Manual navigation.

        Manual navigation always stops autoplay.

        First -> last.
        """

        self._stop_autoplay_for_manual_navigation()

        if not self.spreads:
            return

        if self.current_spread <= 0:
            self.current_spread = (
                len(self.spreads) - 1
            )
        else:
            self.current_spread -= 1

        self._show_current_spread()

    # ==================================================================
    # Page navigation
    # ==================================================================

    def _visible_page_index(self):
        spread = self.spreads[
            self.current_spread
        ]

        if spread["left"] is not None:
            return spread["left"]

        return spread["right"]

    def _visible_page_number(self):
        index = (
            self._visible_page_index()
        )

        if index is None:
            return ""

        return str(
            self._page_number_for_index(
                index
            )
        )

    def _update_page_display(self):
        self.current_page_edit.setText(
            self._visible_page_number()
        )

    def _page_edit_return_pressed(self):
        text = (
            self.current_page_edit.text()
            .strip()
        )

        if not text:
            self._update_page_display()
            return

        try:
            page_number = int(text)
        except ValueError:
            self._update_page_display()
            return

        self.go_to_page(
            page_number
        )

    def go_to_page(self, page_number):
        self._stop_autoplay_for_manual_navigation()

        target_index = None

        # Exact selected content page.
        for index in self.page_indices:
            if (
                self._page_number_for_index(
                    index
                )
                == page_number
            ):
                target_index = index
                break

        # Otherwise choose closest selected page.
        if target_index is None:
            target_index = min(
                self.page_indices,
                key=lambda index: abs(
                    self._page_number_for_index(
                        index
                    )
                    - page_number
                ),
            )

        for spread_index, spread in enumerate(
            self.spreads
        ):
            if target_index in (
                spread["left"],
                spread["right"],
            ):
                self.current_spread = (
                    spread_index
                )
                break

        self._show_current_spread()

    # ==================================================================
    # Progress seeking
    # ==================================================================

    def seek_progress(self, position):
        self._stop_autoplay_for_manual_navigation()

        if len(self.spreads) <= 1:
            target = 0
        else:
            target = round(
                position
                * (len(self.spreads) - 1)
            )

        target = max(
            0,
            min(
                target,
                len(self.spreads) - 1,
            ),
        )

        self.current_spread = target

        self._show_current_spread()

    # ==================================================================
    # Fit
    # ==================================================================

    def fit_pages(self):
        self._stop_autoplay_for_manual_navigation()

        self._preload_around_current()

        if self._current_spread_is_loaded():
            self.canvas.fit_pages()

    # ==================================================================
    # Cache notification
    # ==================================================================

    def _image_ready(self, index):
        if not self.spreads:
            return

        spread = self.spreads[
            self.current_spread
        ]

        if index in (
            spread["left"],
            spread["right"],
        ):
            # Replace the visible frame only when the complete current
            # spread is available.
            if self._current_spread_is_loaded():
                left_image, right_image = (
                    self._get_current_images()
                )

                self.canvas.set_images(
                    left_image,
                    right_image,
                )

                if self._initial_fit_pending:
                    self.canvas.fit_pages()

                    self._initial_fit_pending = (
                        False
                    )

        if self.playing:
            self._maybe_start_autoplay_timer()

    # ==================================================================
    # Autoplay
    # ==================================================================

    def toggle_playback(self):
        if not self.spreads:
            return

        if self.playing:
            self._stop_playback()
            return

        # End reached: pressing autoplay again starts from page 1.
        if (
            self.current_spread
            >= len(self.spreads) - 1
        ):
            self.current_spread = 0
            self._show_current_spread()

        self._start_playback()

    def _start_playback(self):
        if self.playing:
            return

        self.playing = True

        self._update_status()

        self._maybe_start_autoplay_timer()

    def _stop_playback(self):
        self.playing = False

        self.play_timer.stop()

        self._update_status()

    def _maybe_start_autoplay_timer(self):
        if not self.playing:
            return

        # Autoplay stops at the end.
        if (
            self.current_spread
            >= len(self.spreads) - 1
        ):
            self._stop_playback()
            return

        # Current spread must be ready.
        if not self._current_spread_is_loaded():
            self._preload_around_current()
            return

        next_index = (
            self.current_spread + 1
        )

        if next_index < len(self.spreads):
            next_spread = (
                self.spreads[next_index]
            )

            if next_spread["left"] is not None:
                self.cache.request(
                    next_spread["left"]
                )

            if next_spread["right"] is not None:
                self.cache.request(
                    next_spread["right"]
                )

        if not self.play_timer.isActive():
            self.play_timer.start(
                max(
                    1,
                    int(
                        self.page_time
                        * 1000
                    ),
                )
            )

    def _advance_autoplay(self):
        if not self.playing:
            return

        # Stop rather than automatically looping.
        if (
            self.current_spread
            >= len(self.spreads) - 1
        ):
            self._stop_playback()
            return

        next_index = (
            self.current_spread + 1
        )

        next_spread = (
            self.spreads[next_index]
        )

        # Don't show a partially loaded spread.
        if (
            next_spread["left"] is not None
            and not self.cache.has(
                next_spread["left"]
            )
        ):
            self.play_timer.stop()

            self.cache.request(
                next_spread["left"]
            )

            if next_spread["right"] is not None:
                self.cache.request(
                    next_spread["right"]
                )

            return

        if (
            next_spread["right"] is not None
            and not self.cache.has(
                next_spread["right"]
            )
        ):
            self.play_timer.stop()

            self.cache.request(
                next_spread["right"]
            )

            return

        # Next spread is completely cached.
        self.current_spread = next_index

        self._show_current_spread()

        self.play_timer.start(
            max(
                1,
                int(
                    self.page_time
                    * 1000
                ),
            )
        )

    def _update_page_time(self):
        self.play_timer.setInterval(
            max(
                1,
                int(
                    self.page_time
                    * 1000
                ),
            )
        )

    # ==================================================================
    # Settings
    # ==================================================================

    def show_playback_settings(self):
        dialog = PlaybackSettingsDialog(
            self.page_time,
            self.preload_spreads,
            self,
        )

        if dialog.exec() != (
            QDialog.Accepted
        ):
            return

        self.page_time = (
            dialog.page_time_spin.value()
        )

        self.preload_spreads = (
            dialog.preload_spin.value()
        )

        self._update_page_time()

        self._preload_around_current()

        if self.playing:
            self._maybe_start_autoplay_timer()

    # ==================================================================
    # Progress/status
    # ==================================================================

    def _update_progress(self):
        if len(self.spreads) <= 1:
            position = 1.0
        else:
            position = (
                self.current_spread
                / (len(self.spreads) - 1)
            )

        self.progress_bar.set_position(
            position
        )

    def _update_status(self):
        page = (
            self._visible_page_number()
        )

        state = (
            "Playing"
            if self.playing
            else "Paused"
        )

        self.status_label.setText(
            f"Page {page} — {state}"
        )

    # ==================================================================
    # Fullscreen
    # ==================================================================

    def _start_fullscreen(self):
        self.showFullScreen()

        self._fullscreen = True

        self._hide_fullscreen_chrome()

        self.canvas.setFocus()

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self._leave_fullscreen()
        else:
            self._enter_fullscreen()

    def _enter_fullscreen(self):
        self.showFullScreen()

        self._fullscreen = True

        self._hide_fullscreen_chrome()

        self.canvas.setFocus()

    def _leave_fullscreen(self):
        self.showNormal()
        self.showMaximized()

        self._fullscreen = False

        self._show_fullscreen_chrome()

        self.canvas.setFocus()

    def _hide_fullscreen_chrome(self):
        if not self.isFullScreen():
            return

        self.menu_bar.hide()

    def _show_fullscreen_chrome(self):
        self.menu_bar.show()

    def handle_mouse_move(self, position):
        if not self.isFullScreen():
            return

        y = position.y()

        if (
            y <= 50
            or y >= self.height() - 50
        ):
            self._show_fullscreen_chrome()

            self._chrome_timer.start(
                1800
            )

    # ==================================================================
    # Window events
    # ==================================================================

    def resizeEvent(self, event):
        super().resizeEvent(
            event
        )

        if self._initial_fit_pending:
            QTimer.singleShot(
                0,
                self._try_initial_fit,
            )

    def _try_initial_fit(self):
        if not self._initial_fit_pending:
            return

        if self._current_spread_is_loaded():
            self.canvas.fit_pages()

            self._initial_fit_pending = (
                False
            )

    # ==================================================================
    # Keyboard
    # ==================================================================

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()

        # Enter = fit pages.
        if key in (
            Qt.Key_Return,
            Qt.Key_Enter,
        ):
            self.fit_pages()
            event.accept()
            return

        # Ctrl + plus/equal = zoom in.
        if (
            modifiers
            & Qt.ControlModifier
            and key in (
                Qt.Key_Plus,
                Qt.Key_Equal,
            )
        ):
            self.canvas.zoom_in()
            event.accept()
            return

        # Ctrl + minus = zoom out.
        if (
            modifiers
            & Qt.ControlModifier
            and key == Qt.Key_Minus
        ):
            self.canvas.zoom_out()
            event.accept()
            return

        # Left/right = manual navigation.
        if key == Qt.Key_Left:
            self.previous_spread()
            event.accept()
            return

        if key == Qt.Key_Right:
            self.next_spread()
            event.accept()
            return

        # Up/down = zoom.
        if key == Qt.Key_Up:
            self.canvas.zoom_in()
            event.accept()
            return

        if key == Qt.Key_Down:
            self.canvas.zoom_out()
            event.accept()
            return

        # Space = autoplay.
        if key == Qt.Key_Space:
            self.toggle_playback()
            event.accept()
            return

        # F = fullscreen.
        if key == Qt.Key_F:
            self.toggle_fullscreen()
            event.accept()
            return

        # Q = quit.
        if key == Qt.Key_Q:
            self.close()
            event.accept()
            return

        super().keyPressEvent(
            event
        )

    # ==================================================================
    # Shutdown
    # ==================================================================

    def closeEvent(self, event):
        self.play_timer.stop()

        if hasattr(self, "cache"):
            self.cache.shutdown()

        event.accept()

# ============================================================================
# Main
# ============================================================================

def main():
    script_name = Path(
        __file__
    ).name

    parser = argparse.ArgumentParser(
        prog=script_name,
        description=(
            "Scanned book page viewer"
        ),
    )

    parser.add_argument(
        "source_dir",
        help=(
            "Directory containing scanned "
            "page images"
        ),
    )

    parser.add_argument(
        "--pages",
        dest="page_spec",
        metavar="SPEC",
        help=(
            'Content pages to display, '
            'for example "10,20-30". '
            "If omitted, all content pages "
            "are displayed."
        ),
    )

    args = parser.parse_args()

    app = QApplication(
        sys.argv
    )

    try:
        window = BookViewer(
            args.source_dir,
            page_spec=args.page_spec,
        )
    except Exception as exc:
        print(
            f"Error: {exc}",
            file=sys.stderr,
        )
        return 1

    window.show()

    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
```

The important concurrency change is that there is now **no `ImageLoadSignals` object at all**. A worker does:

``` python
self.cache.image_loaded_from_worker(index, image)
```

and the persistent `ImageCache` emits `image_ready`. That removes the `Signal source has been deleted` failure you were seeing.

I also reverted the settings dialog to a real `QDialog`; the previous homemade `QEventLoop` implementation was unnecessary and was the likely source of the `QLineEdit already deleted` exception.

One other behavioral detail: **manual navigation does not blank the screen while an uncached page loads**. The old spread remains visible until the new spread is completely decoded, then the canvas swaps to the new pair in one update. This should substantially reduce the HDD-induced flicker.

## Prompt:

9/10/2026, 6:19:17 AM

again, this fails with many exceptions

    view-pages.py 070-deskew
    Error: libshiboken: Internal C++ object (PySide6.QtWidgets.QLineEdit) already deleted.
    Error calling Python override of QRunnable::run(): Traceback (most recent call last):
      File "/media/ZYD82805_24TB/root/home/user/src/milahu/hocr-files-template-repo/view-pages.py", line 101, in run
        self.cache.image_loaded_from_worker(
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
            self.index,
            ^^^^^^^^^^^
            image,
            ^^^^^^
        )
        ^
      File "/media/ZYD82805_24TB/root/home/user/src/milahu/hocr-files-template-repo/view-pages.py", line 224, in image_loaded_from_worker
        self.image_ready.emit(index)
        ~~~~~~~~~~~~~~~~~~~~~^^^^^^^
    RuntimeError: Signal source has been deleted
    Error calling Python override of QRunnable::run(): Traceback (most recent call last):
      File "/media/ZYD82805_24TB/root/home/user/src/milahu/hocr-files-template-repo/view-pages.py", line 101, in run
        self.cache.image_loaded_from_worker(
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
            self.index,
            ^^^^^^^^^^^
            image,
            ^^^^^^
        )
        ^
      File "/media/ZYD82805_24TB/root/home/user/src/milahu/hocr-files-template-repo/view-pages.py", line 224, in image_loaded_from_worker
        self.image_ready.emit(index)
        ~~~~~~~~~~~~~~~~~~~~~^^^^^^^
    RuntimeError: Signal source has been deleted
    Error calling Python override of QRunnable::run(): Error calling Python override of QRunnable::run(): Traceback (most recent call last):
      File "/media/ZYD82805_24TB/root/home/user/src/milahu/hocr-files-template-repo/view-pages.py", line 101, in run
        self.cache.image_loaded_from_worker(
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
            self.index,
            ^^^^^^^^^^^
            image,
            ^^^^^^
        )
        ^
      File "/media/ZYD82805_24TB/root/home/user/src/milahu/hocr-files-template-repo/view-pages.py", line 224, in image_loaded_from_worker
        self.image_ready.emit(index)
        ~~~~~~~~~~~~~~~~~~~~~^^^^^^^
    Traceback (most recent call last):
    RuntimeError: Signal source has been deleted
      File "/media/ZYD82805_24TB/root/home/user/src/milahu/hocr-files-template-repo/view-pages.py", line 101, in run
        self.cache.image_loaded_from_worker(
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
            self.index,
            ^^^^^^^^^^^
            image,
            ^^^^^^
        )
        ^
      File "/media/ZYD82805_24TB/root/home/user/src/milahu/hocr-files-template-repo/view-pages.py", line 224, in image_loaded_from_worker
        self.image_ready.emit(index)
        ~~~~~~~~~~~~~~~~~~~~~^^^^^^^
    RuntimeError: Signal source has been deleted

nitpick:  
your last versions of the script were missing the shebang line:  
\#!/usr/bin/env python3

## Response:

9/10/2026, 6:19:18 AM

You're right. The exception is a **Qt object-lifetime race during shutdown**: the worker thread can still call back into `ImageCache` while Qt has already destroyed its C++ signal object. I’ve changed the shutdown design so the GUI disconnects first, workers are stopped/waited for, and the cache remains alive until that is complete.

I also restored the requested shebang as the very first line.

\*\*document: \*\*

[Download the complete updated script](sandbox:/mnt/data/view-pages.py)

``` python
#!/usr/bin/env python3

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

from PySide6.QtCore import (
    QObject,
    QPointF,
    QRectF,
    QRunnable,
    QMutex,
    QMutexLocker,
    Qt,
    QThreadPool,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QAction,
    QColor,
    QImage,
    QImageReader,
    QKeySequence,
    QPainter,
    QPalette,
)
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenuBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from _shared import (
    load_config,
    get_page_num,
    parse_page_sequence,
)

config = load_config()

class ImageLoadTask(QRunnable):
    def __init__(self, cache, index, path):
        super().__init__()
        self.cache = cache
        self.index = index
        self.path = path
        self.setAutoDelete(True)

    def run(self):
        reader = QImageReader(str(self.path))
        reader.setAutoTransform(True)
        image = reader.read()

        if not image.isNull():
            image = image.copy()

        cache = self.cache
        if cache is not None:
            cache.image_loaded_from_worker(self.index, image)

class ImageCache(QObject):
    image_ready = Signal(int)

    def __init__(self, paths, max_images=60, parent=None):
        super().__init__(parent)

        self.paths = list(paths)
        self.max_images = max(1, int(max_images))

        self.images = OrderedDict()
        self.loading = set()

        self.mutex = QMutex()

        # Do not parent the thread pool to the cache.  We explicitly control
        # its lifetime during shutdown.
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(2)

        self.shutting_down = False

    def get(self, index):
        locker = QMutexLocker(self.mutex)
        try:
            image = self.images.get(index)

            if image is not None:
                self.images.move_to_end(index)

            return image
        finally:
            locker.unlock()

    def has(self, index):
        locker = QMutexLocker(self.mutex)
        try:
            return index in self.images
        finally:
            locker.unlock()

    def request(self, index):
        if index < 0 or index >= len(self.paths):
            return

        locker = QMutexLocker(self.mutex)

        try:
            if self.shutting_down:
                return

            if index in self.images:
                return

            if index in self.loading:
                return

            self.loading.add(index)
            path = self.paths[index]

        finally:
            locker.unlock()

        self.thread_pool.start(
            ImageLoadTask(self, index, path)
        )

    def image_loaded_from_worker(self, index, image):
        """
        Called by a worker thread.

        IMPORTANT:
        The worker never directly touches any widget.

        During shutdown we also avoid emitting the Qt signal.  This method
        therefore becomes a no-op as soon as shutdown starts.
        """

        should_emit = False

        locker = QMutexLocker(self.mutex)

        try:
            self.loading.discard(index)

            if self.shutting_down:
                return

            if image is not None and not image.isNull():
                self.images[index] = image
                self.images.move_to_end(index)

                while len(self.images) > self.max_images:
                    self.images.popitem(last=False)

                should_emit = True

        finally:
            locker.unlock()

        if not should_emit:
            return

        # The ImageCache QObject is deliberately kept alive by BookViewer
        # until after waitForDone() has returned.
        try:
            self.image_ready.emit(index)
        except RuntimeError:
            # If Qt is already tearing down, silently ignore the late result.
            pass

    def shutdown(self):
        """
        Stop all image loading before the cache QObject can be destroyed.
        """

        locker = QMutexLocker(self.mutex)

        try:
            if self.shutting_down:
                return

            self.shutting_down = True

        finally:
            locker.unlock()

        # Prevent queued-but-not-started QRunnables from starting.
        self.thread_pool.clear()

        # Wait for all currently running image decoders.
        self.thread_pool.waitForDone()

        locker = QMutexLocker(self.mutex)

        try:
            self.loading.clear()

        finally:
            locker.unlock()

        # Release the QThreadPool after all workers have finished.
        self.thread_pool = None

class BookProgressBar(QWidget):
    clicked = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.position = 0.0

        self.setFixedHeight(4)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet("background: white;")

    def set_position(self, position):
        self.position = max(
            0.0,
            min(1.0, float(position)),
        )
        self.update()

    def mousePressEvent(self, event):
        if (
            event.button() == Qt.LeftButton
            and self.width() > 0
        ):
            position = (
                event.position().x()
                / self.width()
            )

            self.clicked.emit(
                max(0.0, min(1.0, position))
            )

            event.accept()
            return

        super().mousePressEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)

        try:
            painter.fillRect(
                self.rect(),
                QColor("white"),
            )

            width = int(
                round(
                    self.width()
                    * self.position
                )
            )

            if width > 0:
                painter.fillRect(
                    0,
                    0,
                    width,
                    self.height(),
                    QColor("black"),
                )

        finally:
            painter.end()

class BookCanvas(QWidget):
    def __init__(self, book_viewer, parent=None):
        super().__init__(parent)

        self.book_viewer = book_viewer

        self.left_image = None
        self.right_image = None

        self.zoom = 1.0
        self.pan = QPointF(0.0, 0.0)

        self.dragging = False
        self.drag_start = QPointF()
        self.pan_start = QPointF()

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

    def background_color(self):
        if self.book_viewer.is_dark_mode():
            value = 13
        else:
            value = 242

        return QColor(
            value,
            value,
            value,
        )

    def set_images(self, left_image, right_image):
        self.left_image = left_image
        self.right_image = right_image
        self.update()

    def _combined_size(self):
        left_w = (
            self.left_image.width()
            if self.left_image is not None
            else 0
        )

        left_h = (
            self.left_image.height()
            if self.left_image is not None
            else 0
        )

        right_w = (
            self.right_image.width()
            if self.right_image is not None
            else 0
        )

        right_h = (
            self.right_image.height()
            if self.right_image is not None
            else 0
        )

        return (
            left_w + right_w,
            max(left_h, right_h),
        )

    def _base_origin(self, zoom):
        total_w, total_h = self._combined_size()

        return (
            (
                self.width()
                - total_w * zoom
            ) / 2.0,

            (
                self.height()
                - total_h * zoom
            ) / 2.0,
        )

    def fit_pages(self):
        total_w, total_h = self._combined_size()

        if (
            total_w <= 0
            or total_h <= 0
            or self.width() <= 0
            or self.height() <= 0
        ):
            return

        scale_x = self.width() / total_w
        scale_y = self.height() / total_h

        self.zoom = max(
            0.01,
            min(scale_x, scale_y),
        )

        self.pan = QPointF(
            0.0,
            0.0,
        )

        self.update()

    def set_zoom(self, zoom, anchor=None):
        if (
            self.left_image is None
            and self.right_image is None
        ):
            return

        old_zoom = self.zoom

        new_zoom = max(
            0.01,
            min(20.0, float(zoom)),
        )

        if abs(new_zoom - old_zoom) < 1e-12:
            return

        if anchor is None:
            anchor = QPointF(
                self.width() / 2.0,
                self.height() / 2.0,
            )

        old_base_x, old_base_y = (
            self._base_origin(old_zoom)
        )

        world_x = (
            anchor.x()
            - old_base_x
            - self.pan.x()
        ) / old_zoom

        world_y = (
            anchor.y()
            - old_base_y
            - self.pan.y()
        ) / old_zoom

        self.zoom = new_zoom

        new_base_x, new_base_y = (
            self._base_origin(new_zoom)
        )

        self.pan = QPointF(
            anchor.x()
            - new_base_x
            - world_x * new_zoom,

            anchor.y()
            - new_base_y
            - world_y * new_zoom,
        )

        self.update()

    def zoom_in(self, anchor=None):
        self.set_zoom(
            self.zoom * 1.2,
            anchor,
        )

    def zoom_out(self, anchor=None):
        self.set_zoom(
            self.zoom / 1.2,
            anchor,
        )

    def _display_image(self, image):
        if image is None:
            return None

        if not self.book_viewer.is_dark_mode():
            return image

        inverted = image.copy()
        inverted.invertPixels(
            QImage.InvertRgb
        )

        return inverted

    def paintEvent(self, event):
        painter = QPainter(self)

        try:
            painter.fillRect(
                self.rect(),
                self.background_color(),
            )

            images = []

            if self.left_image is not None:
                images.append(self.left_image)

            if self.right_image is not None:
                images.append(self.right_image)

            if not images:
                return

            base_x, base_y = (
                self._base_origin(self.zoom)
            )

            x = (
                base_x
                + self.pan.x()
            )

            y = (
                base_y
                + self.pan.y()
            )

            painter.setRenderHint(
                QPainter.SmoothPixmapTransform,
                False,
            )

            for source_image in images:
                image = self._display_image(
                    source_image
                )

                draw_w = (
                    image.width()
                    * self.zoom
                )

                draw_h = (
                    image.height()
                    * self.zoom
                )

                if abs(self.zoom - 1.0) < 1e-12:
                    painter.drawImage(
                        int(round(x)),
                        int(round(y)),
                        image,
                    )
                else:
                    painter.drawImage(
                        QRectF(
                            x,
                            y,
                            draw_w,
                            draw_h,
                        ),
                        image,
                        QRectF(
                            0,
                            0,
                            image.width(),
                            image.height(),
                        ),
                    )

                x += draw_w

        finally:
            painter.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.drag_start = event.position()
            self.pan_start = QPointF(
                self.pan
            )

            self.setCursor(
                Qt.ClosedHandCursor
            )

            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        self.book_viewer.handle_mouse_move(
            event.position()
        )

        if self.dragging:
            delta = (
                event.position()
                - self.drag_start
            )

            self.pan = (
                self.pan_start
                + delta
            )

            self.update()

            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if (
            event.button() == Qt.LeftButton
            and self.dragging
        ):
            self.dragging = False
            self.unsetCursor()

            event.accept()
            return

        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        modifiers = event.modifiers()

        if modifiers & Qt.ControlModifier:
            if event.angleDelta().y() > 0:
                self.zoom_in(
                    event.position()
                )

            elif event.angleDelta().y() < 0:
                self.zoom_out(
                    event.position()
                )

            event.accept()
            return

        if event.angleDelta().y() > 0:
            self.book_viewer.previous_spread()

        elif event.angleDelta().y() < 0:
            self.book_viewer.next_spread()

        event.accept()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.book_viewer.fit_pages()

            event.accept()
            return

        super().mouseDoubleClickEvent(event)

class PlaybackSettingsDialog(QDialog):
    def __init__(
        self,
        page_time,
        preload_spreads,
        parent=None,
    ):
        super().__init__(parent)

        self.setWindowTitle(
            "Playback settings"
        )

        self.page_time_spin = (
            QDoubleSpinBox()
        )

        self.page_time_spin.setRange(
            0.01,
            60.0,
        )

        self.page_time_spin.setSingleStep(
            0.05
        )

        self.page_time_spin.setDecimals(2)
        self.page_time_spin.setSuffix(" s")
        self.page_time_spin.setValue(
            page_time
        )

        self.preload_spin = QSpinBox()

        self.preload_spin.setRange(
            0,
            500,
        )

        self.preload_spin.setValue(
            preload_spreads
        )

        form = QFormLayout()

        form.addRow(
            "Time per spread:",
            self.page_time_spin,
        )

        form.addRow(
            "Preload spreads:",
            self.preload_spin,
        )

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok
            | QDialogButtonBox.Cancel
        )

        buttons.accepted.connect(
            self.accept
        )

        buttons.rejected.connect(
            self.reject
        )

        layout = QVBoxLayout(self)

        layout.addLayout(form)
        layout.addWidget(buttons)

class BookViewer(QMainWindow):
    def __init__(
        self,
        source_dir,
        page_spec=None,
        parent=None,
    ):
        super().__init__(parent)

        self.source_dir = Path(
            source_dir
        )

        self.page_spec = page_spec

        self.setWindowTitle(
            self.source_dir.name
        )

        self.resize(
            1400,
            900,
        )

        self.page_time = 0.2
        self.preload_spreads = 20
        self.max_cache_images = 60

        self.playing = False
        self.current_spread = 0

        self._fullscreen = False
        self._shutting_down = False
        self._initial_fit_pending = True

        self._chrome_timer = QTimer(self)
        self._chrome_timer.setSingleShot(
            True
        )
        self._chrome_timer.setInterval(
            1800
        )
        self._chrome_timer.timeout.connect(
            self._hide_fullscreen_chrome
        )

        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(
            self._advance_autoplay
        )

        self.paths = self._find_paths()

        self.page_indices = (
            self._build_page_indices(
                page_spec
            )
        )

        self.spreads = (
            self._build_spreads(
                self.page_indices
            )
        )

        self.cache = ImageCache(
            self.paths,
            max_images=self.max_cache_images,
            parent=self,
        )

        self.cache.image_ready.connect(
            self._image_ready,
            Qt.QueuedConnection,
        )

        self.canvas = BookCanvas(
            self,
            self,
        )

        self.progress = BookProgressBar(
            self
        )

        self.progress.clicked.connect(
            self._progress_clicked
        )

        self.status_label = QLabel(
            "No pages"
        )

        self.current_page_edit = (
            QLineEdit()
        )

        self.current_page_edit.setFixedWidth(
            80
        )

        self.current_page_edit.setAlignment(
            Qt.AlignCenter
        )

        self.current_page_edit.setPlaceholderText(
            "Page"
        )

        self.current_page_edit.returnPressed.connect(
            self._page_edit_return
        )

        self.previous_button = QPushButton(
            "◀"
        )

        self.next_button = QPushButton(
            "▶"
        )

        self.previous_button.setFixedWidth(
            36
        )

        self.next_button.setFixedWidth(
            36
        )

        self.previous_button.clicked.connect(
            self.previous_spread
        )

        self.next_button.clicked.connect(
            self.next_spread
        )

        self._build_menu()

        central = QWidget(self)

        central_layout = QVBoxLayout(
            central
        )

        central_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        central_layout.setSpacing(0)

        central_layout.addWidget(
            self.canvas,
            1,
        )

        central_layout.addWidget(
            self.progress,
            0,
        )

        self.setCentralWidget(
            central
        )

        self._show_current_spread()

        QTimer.singleShot(
            0,
            self._start_fullscreen,
        )

    def is_dark_mode(self):
        try:
            scheme = (
                QApplication
                .styleHints()
                .colorScheme()
            )

            if scheme == Qt.ColorScheme.Dark:
                return True

            if scheme == Qt.ColorScheme.Light:
                return False

        except AttributeError:
            pass

        palette = QApplication.palette()

        return (
            palette
            .color(QPalette.Window)
            .lightness()
            < 128
        )

    def _find_paths(self):
        suffix = (
            str(config.scan_format)
            .lower()
            .lstrip(".")
        )

        paths = [
            path
            for path in self.source_dir.iterdir()
            if (
                path.is_file()
                and path.suffix.lower().lstrip(".")
                == suffix
            )
        ]

        paths.sort(
            key=get_page_num
        )

        return paths

    def _build_page_indices(
        self,
        page_spec,
    ):
        num_pages = int(
            config.num_pages
        )

        page_to_index = {}

        for index, path in enumerate(
            self.paths
        ):
            page = int(
                get_page_num(path)
            )

            if 1 <= page <= num_pages:
                page_to_index[page] = index

        if page_spec is None:
            selected_pages = list(
                range(
                    1,
                    num_pages + 1,
                )
            )
        else:
            selected_pages = list(
                parse_page_sequence(
                    page_spec,
                    num_pages,
                )
            )

        return [
            page_to_index[page]
            for page in selected_pages
            if page in page_to_index
        ]

    def _build_spreads(
        self,
        page_indices,
    ):
        spreads = []
        pos = 0

        while pos < len(page_indices):
            index = page_indices[pos]

            page = int(
                get_page_num(
                    self.paths[index]
                )
            )

            if (
                page % 2 == 0
                and pos + 1
                < len(page_indices)
            ):
                next_index = (
                    page_indices[pos + 1]
                )

                next_page = int(
                    get_page_num(
                        self.paths[next_index]
                    )
                )

                if next_page == page + 1:
                    spreads.append(
                        (
                            index,
                            next_index,
                        )
                    )

                    pos += 2
                    continue

            if page % 2 == 0:
                spreads.append(
                    (
                        index,
                        None,
                    )
                )
            else:
                spreads.append(
                    (
                        None,
                        index,
                    )
                )

            pos += 1

        return spreads

    def _build_menu(self):
        menu_bar = QMenuBar(self)

        file_menu = menu_bar.addMenu(
            "&File"
        )

        quit_action = QAction(
            "&Quit",
            self,
        )

        quit_action.setShortcut(
            QKeySequence("Q")
        )

        quit_action.triggered.connect(
            self.close
        )

        file_menu.addAction(
            quit_action
        )

        view_menu = menu_bar.addMenu(
            "&View"
        )

        fullscreen_action = QAction(
            "&Fullscreen",
            self,
        )

        fullscreen_action.setShortcut(
            QKeySequence("F")
        )

        fullscreen_action.triggered.connect(
            self.toggle_fullscreen
        )

        view_menu.addAction(
            fullscreen_action
        )

        fit_action = QAction(
            "&Fit pages",
            self,
        )

        fit_action.setShortcut(
            QKeySequence("Return")
        )

        fit_action.triggered.connect(
            self.fit_pages
        )

        view_menu.addAction(
            fit_action
        )

        playback_menu = menu_bar.addMenu(
            "&Playback"
        )

        play_action = QAction(
            "&Play / Pause",
            self,
        )

        play_action.setShortcut(
            QKeySequence("Space")
        )

        play_action.triggered.connect(
            self.toggle_playback
        )

        playback_menu.addAction(
            play_action
        )

        settings_action = QAction(
            "&Settings",
            self,
        )

        settings_action.triggered.connect(
            self.show_playback_settings
        )

        playback_menu.addAction(
            settings_action
        )

        controls = QWidget(self)

        controls_layout = QHBoxLayout(
            controls
        )

        controls_layout.setContentsMargins(
            4,
            0,
            4,
            0,
        )

        controls_layout.setSpacing(4)

        controls_layout.addWidget(
            self.status_label
        )

        controls_layout.addWidget(
            self.previous_button
        )

        controls_layout.addWidget(
            self.current_page_edit
        )

        controls_layout.addWidget(
            self.next_button
        )

        menu_bar.setCornerWidget(
            controls,
            Qt.TopRightCorner,
        )

        self.setMenuBar(
            menu_bar
        )

    def _current_spread_is_loaded(self):
        if not self.spreads:
            return False

        left_index, right_index = (
            self.spreads[
                self.current_spread
            ]
        )

        if (
            left_index is not None
            and not self.cache.has(
                left_index
            )
        ):
            return False

        if (
            right_index is not None
            and not self.cache.has(
                right_index
            )
        ):
            return False

        return True

    def _get_current_images(self):
        if not self.spreads:
            return None, None

        left_index, right_index = (
            self.spreads[
                self.current_spread
            ]
        )

        left_image = (
            self.cache.get(left_index)
            if left_index is not None
            else None
        )

        right_image = (
            self.cache.get(right_index)
            if right_index is not None
            else None
        )

        return (
            left_image,
            right_image,
        )

    def _show_current_spread(self):
        if (
            self._shutting_down
            or not self.spreads
        ):
            return

        left_index, right_index = (
            self.spreads[
                self.current_spread
            ]
        )

        if left_index is not None:
            self.cache.request(
                left_index
            )

        if right_index is not None:
            self.cache.request(
                right_index
            )

        self._preload_around_current()

        # Do not clear the old image while loading the new spread.
        if self._current_spread_is_loaded():
            (
                left_image,
                right_image,
            ) = self._get_current_images()

            self.canvas.set_images(
                left_image,
                right_image,
            )

            if self._initial_fit_pending:
                self.canvas.fit_pages()
                self._initial_fit_pending = False

        self._update_page_edit()
        self._update_progress()
        self._update_status()

    def _preload_around_current(self):
        if not self.spreads:
            return

        count = len(self.spreads)
        center = self.current_spread

        for distance in range(
            1,
            self.preload_spreads + 1,
        ):
            for index in (
                center - distance,
                center + distance,
            ):
                if 0 <= index < count:
                    (
                        left_index,
                        right_index,
                    ) = self.spreads[index]

                    if left_index is not None:
                        self.cache.request(
                            left_index
                        )

                    if right_index is not None:
                        self.cache.request(
                            right_index
                        )

    def _stop_playback(self):
        self.playing = False
        self.play_timer.stop()
        self._update_status()

    def _stop_playback_for_manual_navigation(
        self
    ):
        if self.playing:
            self._stop_playback()

    def next_spread(self):
        if (
            self._shutting_down
            or not self.spreads
        ):
            return

        self._stop_playback_for_manual_navigation()

        if (
            self.current_spread
            >= len(self.spreads) - 1
        ):
            self.current_spread = 0
        else:
            self.current_spread += 1

        self._show_current_spread()

    def previous_spread(self):
        if (
            self._shutting_down
            or not self.spreads
        ):
            return

        self._stop_playback_for_manual_navigation()

        if self.current_spread <= 0:
            self.current_spread = (
                len(self.spreads) - 1
            )
        else:
            self.current_spread -= 1

        self._show_current_spread()

    def _visible_page(self):
        if not self.spreads:
            return None

        left_index, right_index = (
            self.spreads[
                self.current_spread
            ]
        )

        if left_index is not None:
            return int(
                get_page_num(
                    self.paths[left_index]
                )
            )

        if right_index is not None:
            return int(
                get_page_num(
                    self.paths[right_index]
                )
            )

        return None

    def _update_page_edit(self):
        page = self._visible_page()

        if page is None:
            self.current_page_edit.clear()
        else:
            self.current_page_edit.setText(
                str(page)
            )

    def _page_edit_return(self):
        text = (
            self.current_page_edit
            .text()
            .strip()
        )

        try:
            page = int(text)
        except ValueError:
            self._update_page_edit()
            return

        self.go_to_page(page)

    def go_to_page(self, page_number):
        if (
            self._shutting_down
            or not self.spreads
        ):
            return

        self._stop_playback_for_manual_navigation()

        exact_target = None
        closest_target = None
        closest_distance = None

        for position, spread in enumerate(
            self.spreads
        ):
            left_index, right_index = spread

            for index in (
                left_index,
                right_index,
            ):
                if index is None:
                    continue

                page = int(
                    get_page_num(
                        self.paths[index]
                    )
                )

                if page == page_number:
                    exact_target = position
                    break

                distance = abs(
                    page - page_number
                )

                if (
                    closest_distance is None
                    or distance < closest_distance
                ):
                    closest_distance = distance
                    closest_target = position

            if exact_target is not None:
                break

        target = (
            exact_target
            if exact_target is not None
            else closest_target
        )

        if target is not None:
            self.current_spread = target
            self._show_current_spread()

    def _progress_clicked(self, position):
        if (
            self._shutting_down
            or not self.spreads
        ):
            return

        self._stop_playback_for_manual_navigation()

        if len(self.spreads) == 1:
            self.current_spread = 0
        else:
            self.current_spread = int(
                round(
                    position
                    * (len(self.spreads) - 1)
                )
            )

            self.current_spread = max(
                0,
                min(
                    len(self.spreads) - 1,
                    self.current_spread,
                ),
            )

        self._show_current_spread()

    def fit_pages(self):
        if self._shutting_down:
            return

        self._preload_around_current()

        if self._current_spread_is_loaded():
            self.canvas.fit_pages()
            self._initial_fit_pending = False

    def show_playback_settings(self):
        if self._shutting_down:
            return

        dialog = PlaybackSettingsDialog(
            self.page_time,
            self.preload_spreads,
            self,
        )

        if (
            dialog.exec()
            == QDialog.Accepted
        ):
            self.page_time = (
                dialog.page_time_spin.value()
            )

            self.preload_spreads = (
                dialog.preload_spin.value()
            )

            self._preload_around_current()

            if self.playing:
                self.play_timer.stop()
                self._maybe_start_autoplay_timer()

    def toggle_playback(self):
        if (
            self._shutting_down
            or not self.spreads
        ):
            return

        if self.playing:
            self._stop_playback()
            return

        if (
            self.current_spread
            >= len(self.spreads) - 1
        ):
            self.current_spread = 0
            self._show_current_spread()

        self.playing = True

        self._update_status()
        self._maybe_start_autoplay_timer()

    def _maybe_start_autoplay_timer(self):
        if (
            not self.playing
            or self._shutting_down
            or not self.spreads
        ):
            return

        if (
            self.current_spread
            >= len(self.spreads) - 1
        ):
            self._stop_playback()
            return

        if not self._current_spread_is_loaded():
            self._show_current_spread()
            return

        next_position = (
            self.current_spread + 1
        )

        left_index, right_index = (
            self.spreads[next_position]
        )

        next_loaded = True

        if (
            left_index is not None
            and not self.cache.has(
                left_index
            )
        ):
            next_loaded = False
            self.cache.request(
                left_index
            )

        if (
            right_index is not None
            and not self.cache.has(
                right_index
            )
        ):
            next_loaded = False
            self.cache.request(
                right_index
            )

        if not next_loaded:
            return

        interval_ms = max(
            1,
            int(
                round(
                    self.page_time
                    * 1000.0
                )
            ),
        )

        if not self.play_timer.isActive():
            self.play_timer.start(
                interval_ms
            )

    def _advance_autoplay(self):
        if (
            not self.playing
            or self._shutting_down
        ):
            self.play_timer.stop()
            return

        if (
            self.current_spread
            >= len(self.spreads) - 1
        ):
            self._stop_playback()
            return

        next_position = (
            self.current_spread + 1
        )

        left_index, right_index = (
            self.spreads[next_position]
        )

        if left_index is not None:
            self.cache.request(
                left_index
            )

        if right_index is not None:
            self.cache.request(
                right_index
            )

        if (
            (
                left_index is not None
                and not self.cache.has(
                    left_index
                )
            )
            or
            (
                right_index is not None
                and not self.cache.has(
                    right_index
                )
            )
        ):
            self.play_timer.stop()
            return

        self.current_spread = next_position

        self._show_current_spread()

        self.play_timer.stop()
        self._maybe_start_autoplay_timer()

    def _image_ready(self, index):
        if (
            self._shutting_down
            or not self.spreads
        ):
            return

        left_index, right_index = (
            self.spreads[
                self.current_spread
            ]
        )

        if (
            index != left_index
            and index != right_index
        ):
            if self.playing:
                self._maybe_start_autoplay_timer()

            return

        if not self._current_spread_is_loaded():
            return

        (
            left_image,
            right_image,
        ) = self._get_current_images()

        self.canvas.set_images(
            left_image,
            right_image,
        )

        if self._initial_fit_pending:
            self.canvas.fit_pages()
            self._initial_fit_pending = False

        if self.playing:
            self._maybe_start_autoplay_timer()

    def _update_progress(self):
        if not self.spreads:
            self.progress.set_position(
                0.0
            )

        elif len(self.spreads) == 1:
            self.progress.set_position(
                1.0
            )

        else:
            self.progress.set_position(
                self.current_spread
                / (len(self.spreads) - 1)
            )

    def _update_status(self):
        page = self._visible_page()

        if page is None:
            text = "No pages"
        else:
            text = f"Page {page}"

        if self.playing:
            text += " — Playing"
        else:
            text += " — Paused"

        self.status_label.setText(
            text
        )

    def _start_fullscreen(self):
        if self._shutting_down:
            return

        self.showFullScreen()

        self._fullscreen = True

        self._hide_fullscreen_chrome()

        self.canvas.setFocus()

    def toggle_fullscreen(self):
        if self._shutting_down:
            return

        if self._fullscreen:
            self._leave_fullscreen()
        else:
            self._enter_fullscreen()

    def _enter_fullscreen(self):
        self._fullscreen = True

        self.showFullScreen()

        self._hide_fullscreen_chrome()

        self.canvas.setFocus()

    def _leave_fullscreen(self):
        self._chrome_timer.stop()

        self._fullscreen = False

        self.showNormal()
        self.showMaximized()

        self._show_fullscreen_chrome()

        self.canvas.setFocus()

    def _hide_fullscreen_chrome(self):
        if (
            not self._fullscreen
            or self._shutting_down
        ):
            return

        self.menuBar().hide()

    def _show_fullscreen_chrome(self):
        if self._shutting_down:
            return

        self.menuBar().show()

        if self._fullscreen:
            self._chrome_timer.start()

    def handle_mouse_move(self, position):
        if (
            not self._fullscreen
            or self._shutting_down
        ):
            return

        y = position.y()
        h = self.canvas.height()

        if (
            y <= 50
            or y >= h - 50
        ):
            self._show_fullscreen_chrome()

    def resizeEvent(self, event):
        super().resizeEvent(event)

        if self._initial_fit_pending:
            QTimer.singleShot(
                0,
                self._try_initial_fit,
            )

    def _try_initial_fit(self):
        if (
            self._shutting_down
            or not self._initial_fit_pending
        ):
            return

        if self._current_spread_is_loaded():
            self.canvas.fit_pages()
            self._initial_fit_pending = False

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()

        if key in (
            Qt.Key_Return,
            Qt.Key_Enter,
        ):
            self.fit_pages()
            event.accept()
            return

        if (
            key in (
                Qt.Key_Plus,
                Qt.Key_Equal,
            )
            and modifiers
            & Qt.ControlModifier
        ):
            self.canvas.zoom_in()
            event.accept()
            return

        if (
            key in (
                Qt.Key_Minus,
                Qt.Key_Underscore,
            )
            and modifiers
            & Qt.ControlModifier
        ):
            self.canvas.zoom_out()
            event.accept()
            return

        if key == Qt.Key_Left:
            self.previous_spread()
            event.accept()
            return

        if key == Qt.Key_Right:
            self.next_spread()
            event.accept()
            return

        if key == Qt.Key_Up:
            self.canvas.zoom_in()
            event.accept()
            return

        if key == Qt.Key_Down:
            self.canvas.zoom_out()
            event.accept()
            return

        if key == Qt.Key_Space:
            self.toggle_playback()
            event.accept()
            return

        if key == Qt.Key_F:
            self.toggle_fullscreen()
            event.accept()
            return

        if key == Qt.Key_Q:
            self.close()
            event.accept()
            return

        super().keyPressEvent(event)

    def closeEvent(self, event):
        if self._shutting_down:
            event.accept()
            return

        # Mark the entire GUI as shutting down first.
        self._shutting_down = True
        self.playing = False

        self.play_timer.stop()
        self._chrome_timer.stop()

        # Stop GUI-side delivery before workers are stopped.
        try:
            self.cache.image_ready.disconnect(
                self._image_ready
            )
        except (
            RuntimeError,
            TypeError,
        ):
            pass

        # This sets cache.shutting_down, clears queued workers, and waits for
        # all running workers.  Because cache is parented to this BookViewer,
        # its QObject cannot be destroyed until after this closeEvent returns.
        self.cache.shutdown()

        event.accept()

def main():
    parser = argparse.ArgumentParser(
        prog=Path(__file__).name,
        description=(
            "View scanned book pages "
            "as facing-page spreads."
        ),
    )

    parser.add_argument(
        "source_dir",
        help=(
            "directory containing "
            "the scanned page images"
        ),
    )

    parser.add_argument(
        "--pages",
        metavar="SPEC",
        help=(
            'page selection, for example '
            '"10,20-30"'
        ),
    )

    args = parser.parse_args()

    app = QApplication(sys.argv)

    app.setApplicationName(
        Path(__file__).stem
    )

    viewer = BookViewer(
        args.source_dir,
        page_spec=args.pages,
    )

    viewer.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
```

The important change is that **the worker pool is now explicitly independent of the QObject lifetime**, and `closeEvent()` disconnects GUI delivery **before** stopping and joining the workers. The worker callback also refuses to emit once shutdown has begun.

I also fixed the mouse-centered zoom calculation in this version so the point under the cursor remains stationary while zooming.

------------------------------------------------------------------------

Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)

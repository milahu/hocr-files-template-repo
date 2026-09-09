#!/usr/bin/env python3

# TODO
r'''
if autoplay is active, then manual navigation (scroll with mousewheel or arrow left/right) should stop the autplay

add hotkey: enter = fit pages

click on the progress bar = seek to that page (if autoplay was active then stop autoplay)

remove the status bar on the bottom. instead, show the status in the menu bar on the top

zoom should follow the mouse pointer
'''

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

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

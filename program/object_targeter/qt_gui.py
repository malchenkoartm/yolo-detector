from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import os
import sys
import time
from typing import Deque

import cv2
import numpy as np

from interfaces import (
    DeviceInfo,
    GUIEvent,
    GUIEventType,
    IIOOperatorGUI,
)


def _bgr_to_qimage(frame_bgr: np.ndarray):
    # Local import to keep module importable without Qt in some contexts.
    from PyQt6.QtGui import QImage

    if frame_bgr is None:
        return QImage()
    if frame_bgr.ndim == 2:
        frame_bgr = np.stack([frame_bgr, frame_bgr, frame_bgr], axis=-1)
    elif frame_bgr.ndim == 3 and frame_bgr.shape[2] == 4:
        frame_bgr = frame_bgr[:, :, :3]
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame must be HxWx3 BGR-like")

    h, w = frame_bgr.shape[:2]
    # BGR -> RGB, then wrap memory (copy to detach from numpy lifetime).
    rgb = frame_bgr[:, :, ::-1].copy()
    bytes_per_line = 3 * w
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()


def _scan_v4l2_devices(max_index: int = 15) -> list[DeviceInfo]:
    devices: list[DeviceInfo] = []
    for i in range(max_index + 1):
        path = f"/dev/video{i}"
        if os.path.exists(path):
            devices.append(DeviceInfo(id=path, label=path))
    return devices


def _scan_opencv_cameras(max_index: int = 10) -> list[DeviceInfo]:
    if sys.platform.startswith("win"):
        # Avoid backend probing on Windows here: repeated OpenCV open() calls
        # can trigger native crashes on some driver stacks.
        return [
            DeviceInfo(id="0", label="Camera 0"),
            DeviceInfo(id="1", label="Camera 1"),
            DeviceInfo(id="2", label="Camera 2"),
        ]

    devices: list[DeviceInfo] = []
    misses = 0
    max_probe = max_index
    for i in range(max_probe + 1):
        backends = [cv2.CAP_ANY]
        found = False
        for be in backends:
            cap = cv2.VideoCapture(i, be)
            ok = cap.isOpened()
            if ok:
                ret, _ = cap.read()
                if ret:
                    devices.append(DeviceInfo(id=str(i), label=f"Camera {i}"))
                    found = True
                    cap.release()
                    break
            cap.release()
        if found:
            misses = 0
        else:
            misses += 1
            if devices and misses >= 1:
                break
    return devices


def _qt_multimedia_devices() -> tuple[list[DeviceInfo], list[DeviceInfo]]:
    """
    Best-effort device listing via QtMultimedia.
    On some Linux setups, ids may not map 1:1 to /dev/videoX; we still expose them.
    """
    try:
        from PyQt6.QtMultimedia import QMediaDevices
    except Exception:
        return [], []

    def to_id_str(qid) -> str:
        try:
            b = bytes(qid)
            return b.hex()
        except Exception:
            return str(qid)

    videos: list[DeviceInfo] = []
    audios: list[DeviceInfo] = []

    try:
        for d in QMediaDevices.videoInputs():
            videos.append(DeviceInfo(id=to_id_str(d.id()), label=d.description()))
    except Exception:
        pass

    try:
        for d in QMediaDevices.audioInputs():
            audios.append(DeviceInfo(id=to_id_str(d.id()), label=d.description()))
    except Exception:
        pass

    return videos, audios


def _sounddevice_inputs() -> list[DeviceInfo]:
    try:
        import sounddevice as sd
    except Exception:
        return []

    devices: list[DeviceInfo] = []
    try:
        default_in = sd.default.device[0] if sd.default.device else None
        hostapis = sd.query_hostapis()
        host_names = {i: str(h.get("name", "")) for i, h in enumerate(hostapis)}
        all_devs = sd.query_devices()
        items: list[tuple[int, str]] = []
        for idx, d in enumerate(all_devs):
            if int(d.get("max_input_channels", 0)) > 0:
                name = str(d.get("name", f"mic {idx}"))
                host = host_names.get(int(d.get("hostapi", -1)), "?")
                prefix = "* " if default_in == idx else ""
                label = f"{prefix}[{idx}] {name} ({host})"
                items.append((idx, label))
        def host_rank(label: str) -> int:
            l = label.lower()
            if "directsound" in l:
                return 0
            if "wasapi" in l:
                return 1
            if "wdm-ks" in l:
                return 2
            if "mme" in l:
                return 3
            return 4
        items.sort(key=lambda p: (host_rank(p[1]), p[0]))
        devices = [DeviceInfo(id=str(idx), label=label) for idx, label in items]
    except Exception:
        return []

    return devices


@dataclass(frozen=True)
class _Style:
    panel_height: int = 44
    font_px: int = 11
    status_font_px: int = 10


class PyQt6IOOperatorGUI(IIOOperatorGUI):
    def __init__(self, title: str = "PTZ Detector", style: _Style | None = None):
        self.__style = style or _Style()

        self.__app = None
        self.__win = None
        self.__video_label = None
        self.__overlay_lines: list[str] = []
        self.__status = ""

        self.__events: Deque[GUIEvent] = deque(maxlen=256)
        self.__last_fps_t = time.time()
        self.__fps = 0.0
        self.__frame_count = 0

        self.__selected_video_id: str | None = None
        self.__selected_audio_id: str | None = None

        self.__title = title
        self.__conf_slider = None
        self.__target_spin = None
        self.__mic_combo = None
        self.__zoom_badge = None
        self.__conf_badge = None
        self.__gain_badge = None
        self.__gate_badge = None
        self.__fs_badge = None
        self.__audio_gain = 1.0
        self.__audio_threshold = 0.0
        self.__audio_fs = 44100
        self.__class_apply_callback = None

    def set_class_apply_callback(self, callback):
        self.__class_apply_callback = callback

    def start(self):
        from PyQt6.QtCore import Qt, QTimer
        from PyQt6.QtGui import QFont, QAction, QShortcut, QKeySequence
        from PyQt6.QtWidgets import (
            QApplication,
            QComboBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QMenu,
            QMainWindow,
            QProgressBar,
            QPushButton,
            QSlider,
            QSpinBox,
            QVBoxLayout,
            QDialog,
            QFormLayout,
            QDoubleSpinBox,
            QDialogButtonBox,
            QComboBox as QComboBox2,
            QFrame,
            QWidget,
        )

        self.__app = QApplication.instance() or QApplication([])

        win = QMainWindow()
        win.setWindowTitle(self.__title)
        win.setMinimumSize(980, 620)
        win.setStyleSheet(
            """
            QMainWindow { background: #0f1115; }
            QLabel { color: #d7dae0; }
            QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
                background: #151822; color: #d7dae0; border: 1px solid #22263a;
                padding: 5px 8px; border-radius: 7px;
                selection-background-color: #3558b8;
            }
            QSlider::groove:horizontal { height: 4px; background: #22263a; border-radius: 2px; }
            QSlider::handle:horizontal { width: 12px; margin: -6px 0; border-radius: 6px; background: #4d7cff; }
            QPushButton {
                background: #151822; color: #d7dae0; border: 1px solid #22263a;
                padding: 5px 10px; border-radius: 7px;
            }
            QPushButton:pressed { background: #1c2030; }
            QPushButton:hover { border-color: #3654a2; }
            """
        )

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        def mk_label(text: str) -> QLabel:
            l = QLabel(text)
            f = QFont()
            f.setPixelSize(self.__style.font_px)
            l.setFont(f)
            l.setStyleSheet("color: #9aa3b2;")
            return l

        cam_label = mk_label("CAM")
        cam_combo = QComboBox()
        cam_combo.setMinimumWidth(360)
        cam_combo.setMaxVisibleItems(20)
        cam_combo.setStyleSheet("QComboBox QAbstractItemView { min-width: 580px; }")
        cam_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)

        mic_label = mk_label("MIC")
        mic_combo = QComboBox()
        mic_combo.setMinimumWidth(360)
        mic_combo.setMaxVisibleItems(20)
        mic_combo.setStyleSheet("QComboBox QAbstractItemView { min-width: 580px; }")
        mic_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        mic_meter = QProgressBar()
        mic_meter.setRange(0, 100)
        mic_meter.setValue(0)
        mic_meter.setFixedWidth(110)
        mic_meter.setTextVisible(False)
        mic_meter.setToolTip("Mic input level")
        mic_meter.setStyleSheet(
            "QProgressBar { background: rgba(21,24,34,0.45); border: 1px solid #22263a; border-radius: 6px; }"
            "QProgressBar::chunk { background: rgba(77,124,255,0.75); border-radius: 6px; }"
        )

        def mk_badge(text: str) -> QLabel:
            b = QLabel(text)
            b.setStyleSheet(
                "background: #10131b; border: 1px solid #22263a; color: #cbd2e2;"
                "padding: 4px 8px; border-radius: 8px;"
            )
            return b

        work_btn = QPushButton("STOP")
        zoom_in_btn = QPushButton("ZOOM +")
        zoom_out_btn = QPushButton("ZOOM -")

        zoom_badge = mk_badge("Zoom: x1.0")

        conf_label = mk_label("CONF")
        conf_slider = QSlider(Qt.Orientation.Horizontal)
        conf_slider.setMinimum(1)
        conf_slider.setMaximum(99)
        conf_slider.setValue(10)
        conf_slider.setFixedWidth(160)

        conf_badge = mk_badge("Conf: 10%")

        gain_label = mk_label("GAIN")
        gain_spin = QDoubleSpinBox()
        gain_spin.setDecimals(2)
        gain_spin.setMinimum(0.10)
        gain_spin.setMaximum(30.0)
        gain_spin.setSingleStep(0.25)
        gain_spin.setValue(1.00)
        gain_spin.setFixedWidth(90)

        gate_label = mk_label("GATE")
        gate_spin = QDoubleSpinBox()
        gate_spin.setDecimals(4)
        gate_spin.setMinimum(0.0000)
        gate_spin.setMaximum(0.2000)
        gate_spin.setSingleStep(0.0025)
        gate_spin.setValue(0.0000)
        gate_spin.setFixedWidth(95)

        fs_label = mk_label("FS")
        fs_combo = QComboBox2()
        fs_combo.addItem("8000", 8000)
        fs_combo.addItem("16000", 16000)
        fs_combo.addItem("44100", 44100)
        fs_combo.addItem("48000", 48000)
        fs_combo.setCurrentIndex(fs_combo.findData(44100))
        fs_combo.setFixedWidth(95)

        target_label = mk_label("TRACK ID")
        target_spin = QSpinBox()
        target_spin.setMinimum(-1)
        target_spin.setMaximum(999999)
        target_spin.setValue(-1)
        target_spin.setFixedWidth(110)
        target_spin.setToolTip("-1 = no target")
        classes_label = mk_label("CLASSES")
        classes_edit = QLineEdit()
        classes_edit.setPlaceholderText("person, car")
        classes_edit.setMinimumWidth(240)
        classes_place = QPushButton("PLACE")
        classes_add = QPushButton("ADD")
        classes_place.setAutoDefault(False)
        classes_add.setAutoDefault(False)
        classes_place.setDefault(False)
        classes_add.setDefault(False)

        status = QLabel("")
        f = QFont()
        f.setPixelSize(self.__style.status_font_px)
        status.setFont(f)
        status.setStyleSheet("color: #8b93a7;")
        status.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        toolbar = QFrame()
        toolbar.setStyleSheet("QFrame { border: 1px solid #151822; border-radius: 8px; background: #0f1115; }")
        toolbar_layout = QVBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(8, 8, 8, 8)
        toolbar_layout.setSpacing(9)

        row1 = QHBoxLayout()
        row1.setSpacing(8)
        row1.addWidget(cam_label)
        row1.addWidget(cam_combo)
        row1.addWidget(mic_label)
        row1.addWidget(mic_combo)
        row1.addWidget(mic_meter)
        row1.addWidget(gain_label)
        row1.addWidget(gain_spin)
        row1.addWidget(gate_label)
        row1.addWidget(gate_spin)
        row1.addWidget(fs_label)
        row1.addWidget(fs_combo)
        row1.addSpacing(8)
        row1.addWidget(status)

        row2 = QHBoxLayout()
        row2.setSpacing(8)
        row2.addWidget(work_btn)
        row2.addWidget(zoom_out_btn)
        row2.addWidget(zoom_in_btn)
        row2.addWidget(zoom_badge)
        row2.addSpacing(10)
        row2.addWidget(conf_label)
        row2.addWidget(conf_slider)
        row2.addWidget(conf_badge)
        row2.addWidget(target_label)
        row2.addWidget(target_spin)
        row2.addSpacing(10)
        row2.addWidget(classes_label)
        row2.addWidget(classes_edit)
        row2.addWidget(classes_place)
        row2.addWidget(classes_add)
        row2.addStretch(1)

        toolbar_layout.addLayout(row1)
        toolbar_layout.addLayout(row2)

        video = QLabel()
        video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        video.setStyleSheet("background: #07080b; border: 1px solid #151822; border-radius: 10px;")
        video.setMinimumHeight(520)

        root.addWidget(toolbar)
        root.addWidget(video, 1)

        win.setCentralWidget(central)

        self.__win = win
        self.__video_label = video
        self.__conf_slider = conf_slider
        self.__target_spin = target_spin
        self.__mic_combo = mic_combo
        self.__zoom_badge = zoom_badge
        self.__conf_badge = conf_badge
        self.__mic_meter = mic_meter
        self.__gain_spin = gain_spin
        self.__gate_spin = gate_spin
        self.__fs_combo = fs_combo
        self.__classes_edit = classes_edit
        self.__classes_buffer = ""

        def push(ev_type: GUIEventType, value=None):
            if ev_type == GUIEventType.CLASS_NAMES_CHANGED:
                print(f"[GUI] class event queued: {value}")
            self.__events.append(GUIEvent(ev_type, value))

        def refresh_devices():
            qt_videos, qt_audios = _qt_multimedia_devices()
            v4l = _scan_v4l2_devices()
            cv2_cams = _scan_opencv_cameras()
            sd_inputs = _sounddevice_inputs()

            # Prefer /dev/videoX listing for linux+ffmpeg path; fall back to Qt list.
            if sys.platform.startswith("linux"):
                video_items = v4l if v4l else qt_videos
            else:
                video_items = cv2_cams if cv2_cams else qt_videos
            # Prefer sounddevice list because AudioRecorder uses sounddevice.
            audio_items = sd_inputs if sd_inputs else qt_audios

            cam_combo.blockSignals(True)
            cam_combo.clear()
            for d in video_items:
                cam_combo.addItem(d.label, d.id)
                cam_combo.setItemData(cam_combo.count() - 1, d.label, Qt.ItemDataRole.ToolTipRole)
            cam_combo.blockSignals(False)

            mic_combo.blockSignals(True)
            mic_combo.clear()
            for d in audio_items:
                mic_combo.addItem(d.label, d.id)
                mic_combo.setItemData(mic_combo.count() - 1, d.label, Qt.ItemDataRole.ToolTipRole)
            mic_combo.blockSignals(False)

            if self.__selected_video_id is not None:
                idx = cam_combo.findData(self.__selected_video_id)
                if idx >= 0:
                    cam_combo.setCurrentIndex(idx)
            if self.__selected_audio_id is not None:
                idx = mic_combo.findData(self.__selected_audio_id)
                if idx >= 0:
                    mic_combo.setCurrentIndex(idx)

        refresh_devices()

        cam_combo.currentIndexChanged.connect(
            lambda _=None: (
                setattr(self, "_PyQt6IOOperatorGUI__selected_video_id", cam_combo.currentData()),
                push(GUIEventType.VIDEO_SOURCE_CHANGED, cam_combo.currentData()),
            )
        )
        mic_combo.currentIndexChanged.connect(
            lambda _=None: (
                setattr(self, "_PyQt6IOOperatorGUI__selected_audio_id", mic_combo.currentData()),
                push(GUIEventType.AUDIO_SOURCE_CHANGED, mic_combo.currentData()),
            )
        )
        def emit_audio_settings():
            push(
                GUIEventType.AUDIO_SETTINGS_CHANGED,
                {
                    "device_id": mic_combo.currentData(),
                    "gain": float(gain_spin.value()),
                    "threshold": float(gate_spin.value()),
                    "fs": int(fs_combo.currentData()),
                },
            )
        gain_spin.valueChanged.connect(lambda _=None: emit_audio_settings())
        gate_spin.valueChanged.connect(lambda _=None: emit_audio_settings())
        fs_combo.currentIndexChanged.connect(lambda _=None: emit_audio_settings())

        def place_classes():
            text = str(getattr(self, "_PyQt6IOOperatorGUI__classes_buffer", classes_edit.text())).strip()
            if not text:
                self.set_status("classes: empty input")
                return
            print(f"[GUI] PLACE clicked: {text}", flush=True)
            cb = getattr(self, "_PyQt6IOOperatorGUI__class_apply_callback", None)
            if callable(cb):
                cb("place", text)
            else:
                push(GUIEventType.CLASS_NAMES_CHANGED, {"mode": "place", "text": text})

        def add_classes():
            text = str(getattr(self, "_PyQt6IOOperatorGUI__classes_buffer", classes_edit.text())).strip()
            if not text:
                self.set_status("classes: empty input")
                return
            print(f"[GUI] ADD clicked: {text}", flush=True)
            cb = getattr(self, "_PyQt6IOOperatorGUI__class_apply_callback", None)
            if callable(cb):
                cb("add", text)
            else:
                push(GUIEventType.CLASS_NAMES_CHANGED, {"mode": "add", "text": text})

        classes_place.clicked.connect(place_classes)
        classes_add.clicked.connect(add_classes)
        classes_edit.returnPressed.connect(place_classes)

        # Fallback: auto-apply typed classes shortly after user stops typing.
        class_timer = QTimer(win)
        class_timer.setSingleShot(True)
        class_timer.setInterval(900)

        def remember_classes_text(text: str):
            self.__classes_buffer = (text or "").strip()
            if self.__classes_buffer:
                class_timer.start()

        def auto_apply_classes():
            text = str(getattr(self, "_PyQt6IOOperatorGUI__classes_buffer", "")).strip()
            if not text:
                return
            cb = getattr(self, "_PyQt6IOOperatorGUI__class_apply_callback", None)
            if callable(cb):
                print(f"[GUI] AUTO APPLY classes: {text}", flush=True)
                cb("place", text)
            else:
                push(GUIEventType.CLASS_NAMES_CHANGED, {"mode": "place", "text": text})

        classes_edit.textChanged.connect(remember_classes_text)
        class_timer.timeout.connect(auto_apply_classes)

        # Right-click mic combo: audio settings dialog
        mic_combo.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        def open_mic_settings(_pos=None):
            device_id = mic_combo.currentData()
            if device_id is None:
                return

            dlg = QDialog(win)
            dlg.setWindowTitle("Microphone settings")
            dlg.setModal(True)
            lay = QFormLayout(dlg)

            gain = QDoubleSpinBox(dlg)
            gain.setDecimals(2)
            gain.setMinimum(0.10)
            gain.setMaximum(30.0)
            gain.setSingleStep(0.25)
            gain.setValue(float(self.__audio_gain))
            gain.setToolTip("Software gain (can strongly amplify)")

            thr = QDoubleSpinBox(dlg)
            thr.setDecimals(4)
            thr.setMinimum(0.0000)
            thr.setMaximum(0.2000)
            thr.setSingleStep(0.0025)
            thr.setValue(float(self.__audio_threshold))
            thr.setToolTip("RMS gate threshold (0..~0.2)")

            fs = QComboBox2(dlg)
            fs.addItem("8000", 8000)
            fs.addItem("16000", 16000)
            fs.addItem("44100", 44100)
            fs.addItem("48000", 48000)
            current_fs_idx = fs.findData(int(self.__audio_fs))
            fs.setCurrentIndex(current_fs_idx if current_fs_idx >= 0 else 2)

            lay.addRow("Gain", gain)
            lay.addRow("Threshold", thr)
            lay.addRow("Sample rate", fs)

            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
            lay.addRow(buttons)

            def on_ok():
                # Update local badges immediately for responsiveness.
                self.sync_audio_settings(float(gain.value()), float(thr.value()), int(fs.currentData()))
                push(
                    GUIEventType.AUDIO_SETTINGS_CHANGED,
                    {
                        "device_id": device_id,
                        "gain": float(gain.value()),
                        "threshold": float(thr.value()),
                        "fs": int(fs.currentData()),
                    },
                )
                dlg.accept()

            buttons.accepted.connect(on_ok)
            buttons.rejected.connect(dlg.reject)
            dlg.exec()

        def mic_menu(_pos):
            menu = QMenu(win)
            act = QAction("Settings…", win)
            act.triggered.connect(open_mic_settings)
            menu.addAction(act)
            menu.exec(mic_combo.mapToGlobal(_pos))

        mic_combo.customContextMenuRequested.connect(mic_menu)
        work_btn.clicked.connect(lambda: push(GUIEventType.TOGGLE_WORK))
        zoom_in_btn.clicked.connect(lambda: push(GUIEventType.ZOOM_IN))
        zoom_out_btn.clicked.connect(lambda: push(GUIEventType.ZOOM_OUT))

        conf_slider.valueChanged.connect(lambda v: push(GUIEventType.CONF_CHANGED, float(v) / 100.0))

        def on_target(v: int):
            push(GUIEventType.TARGET_TRACK_CHANGED, None if v < 0 else int(v))

        target_spin.valueChanged.connect(on_target)

        # Keyboard shortcuts without overriding keyPressEvent.
        # Do not trigger shortcuts while typing in text-like controls.
        def can_handle_shortcuts() -> bool:
            fw = QApplication.focusWidget()
            return not isinstance(fw, (QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox))

        sc_exit_esc = QShortcut(QKeySequence(Qt.Key.Key_Escape), win)
        sc_exit_q = QShortcut(QKeySequence(Qt.Key.Key_Q), win)
        sc_toggle = QShortcut(QKeySequence(Qt.Key.Key_Space), win)
        sc_zoom_in_plus = QShortcut(QKeySequence(Qt.Key.Key_Plus), win)
        sc_zoom_in_eq = QShortcut(QKeySequence(Qt.Key.Key_Equal), win)
        sc_zoom_out = QShortcut(QKeySequence(Qt.Key.Key_Minus), win)
        for sc in [sc_exit_esc, sc_exit_q, sc_toggle, sc_zoom_in_plus, sc_zoom_in_eq, sc_zoom_out]:
            sc.setContext(Qt.ShortcutContext.WindowShortcut)
        sc_exit_esc.activated.connect(lambda: push(GUIEventType.EXIT) if can_handle_shortcuts() else None)
        sc_exit_q.activated.connect(lambda: push(GUIEventType.EXIT) if can_handle_shortcuts() else None)
        sc_toggle.activated.connect(lambda: push(GUIEventType.TOGGLE_WORK) if can_handle_shortcuts() else None)
        sc_zoom_in_plus.activated.connect(lambda: push(GUIEventType.ZOOM_IN) if can_handle_shortcuts() else None)
        sc_zoom_in_eq.activated.connect(lambda: push(GUIEventType.ZOOM_IN) if can_handle_shortcuts() else None)
        sc_zoom_out.activated.connect(lambda: push(GUIEventType.ZOOM_OUT) if can_handle_shortcuts() else None)

        def on_close(_):
            push(GUIEventType.EXIT)

        win.destroyed.connect(on_close)

        self.__status_label = status

        # Periodically sync status label; UI updates stay very quiet.
        t = QTimer()
        t.setInterval(120)
        t.timeout.connect(lambda: status.setText(self.__status))
        t.start()
        self.__status_timer = t

        win.showMaximized()

    def stop(self):
        if self.__win is not None:
            self.__win.close()
        self.__win = None
        self.__video_label = None

    def render_frame(self, frame: np.ndarray):
        if self.__video_label is None:
            return

        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QPainter, QPixmap, QColor, QFont

        try:
            img = _bgr_to_qimage(frame)
        except Exception:
            # Last-resort fallback to avoid blank UI on unusual camera formats.
            f = np.asarray(frame)
            if f.ndim == 2:
                f = np.stack([f, f, f], axis=-1)
            elif f.ndim == 3 and f.shape[2] > 3:
                f = f[:, :, :3]
            img = _bgr_to_qimage(f)
        if img.isNull():
            return

        # Update a small FPS estimate for status.
        self.__frame_count += 1
        now = time.time()
        if now - self.__last_fps_t >= 0.7:
            self.__fps = self.__frame_count / (now - self.__last_fps_t)
            self.__frame_count = 0
            self.__last_fps_t = now

        pix = QPixmap.fromImage(img)

        # Draw overlay lines (small, semi-transparent).
        if self.__overlay_lines:
            p = QPainter(pix)
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            pad = 10
            line_h = 16
            box_w = 0
            for ln in self.__overlay_lines[:6]:
                box_w = max(box_w, 8 * len(ln))
            box_h = pad * 2 + line_h * min(6, len(self.__overlay_lines))
            p.fillRect(10, 10, min(pix.width() - 20, box_w + pad * 2), box_h, QColor(0, 0, 0, 120))
            f = QFont()
            f.setPixelSize(12)
            p.setFont(f)
            p.setPen(QColor(220, 225, 235))
            y = 10 + pad + 12
            for ln in self.__overlay_lines[:6]:
                p.drawText(10 + pad, y, ln)
                y += line_h
            p.end()

        # Scale to label keeping aspect ratio.
        target_w = max(1, self.__video_label.width())
        target_h = max(1, self.__video_label.height())
        scaled = pix.scaled(target_w, target_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.__video_label.setPixmap(scaled)

    def set_overlay_text(self, lines: list[str]):
        self.__overlay_lines = [ln for ln in lines if ln][:12]

    def set_status(self, text: str):
        # Keep it short: status is meant to be “quiet”.
        t = (text or "").strip()
        if self.__fps > 0:
            t = f"{t}   {self.__fps:.0f} FPS" if t else f"{self.__fps:.0f} FPS"
        self.__status = t

    def list_video_devices(self) -> list[DeviceInfo]:
        if sys.platform.startswith("linux"):
            v4l = _scan_v4l2_devices()
            if v4l:
                return v4l
        else:
            cv2_cams = _scan_opencv_cameras()
            if cv2_cams:
                return cv2_cams
        qt_videos, _ = _qt_multimedia_devices()
        return qt_videos

    def list_audio_devices(self) -> list[DeviceInfo]:
        sd_inputs = _sounddevice_inputs()
        if sd_inputs:
            return sd_inputs
        _, qt_audios = _qt_multimedia_devices()
        return qt_audios

    def set_video_device(self, device_id: str):
        self.__selected_video_id = device_id
        self.__events.append(GUIEvent(GUIEventType.VIDEO_SOURCE_CHANGED, device_id))

    def set_audio_device(self, device_id: str):
        self.__selected_audio_id = device_id
        self.__events.append(GUIEvent(GUIEventType.AUDIO_SOURCE_CHANGED, device_id))

    def poll_event(self) -> GUIEvent | None:
        return self.__events.popleft() if self.__events else None

    def sync_conf(self, conf: float):
        s = self.__conf_slider
        if s is None:
            return
        v = int(round(float(conf) * 100))
        v = max(s.minimum(), min(s.maximum(), v))
        if s.value() != v:
            s.blockSignals(True)
            s.setValue(v)
            s.blockSignals(False)
        b = self.__conf_badge
        if b is not None:
            b.setText(f"Conf: {v}%")

    def sync_target(self, target_track: int | None):
        sp = self.__target_spin
        if sp is None:
            return
        v = -1 if target_track is None else int(target_track)
        if sp.value() != v:
            sp.blockSignals(True)
            sp.setValue(v)
            sp.blockSignals(False)

    def sync_zoom(self, zoom_level: float):
        b = self.__zoom_badge
        if b is None:
            return
        try:
            b.setText(f"Zoom: x{float(zoom_level):.1f}")
        except Exception:
            pass

    def sync_audio_settings(self, gain: float, threshold: float, fs: int):
        self.__audio_gain = float(gain)
        self.__audio_threshold = float(threshold)
        self.__audio_fs = int(fs)
        gs = getattr(self, "_PyQt6IOOperatorGUI__gain_spin", None)
        if gs is not None:
            try:
                gv = float(gain)
                if abs(gs.value() - gv) > 1e-6:
                    gs.blockSignals(True)
                    gs.setValue(gv)
                    gs.blockSignals(False)
            except Exception:
                pass
        ts = getattr(self, "_PyQt6IOOperatorGUI__gate_spin", None)
        if ts is not None:
            try:
                tv = float(threshold)
                if abs(ts.value() - tv) > 1e-6:
                    ts.blockSignals(True)
                    ts.setValue(tv)
                    ts.blockSignals(False)
            except Exception:
                pass
        fc = getattr(self, "_PyQt6IOOperatorGUI__fs_combo", None)
        if fc is not None:
            try:
                idx = fc.findData(int(fs))
                if idx >= 0 and fc.currentIndex() != idx:
                    fc.blockSignals(True)
                    fc.setCurrentIndex(idx)
                    fc.blockSignals(False)
            except Exception:
                pass

    def sync_classes(self, names: list[str]):
        ce = getattr(self, "_PyQt6IOOperatorGUI__classes_edit", None)
        if ce is None:
            return
        # Do not overwrite while user is typing.
        if ce.hasFocus():
            return
        txt = ", ".join([n for n in names if n])
        if ce.text().strip() != txt.strip():
            ce.blockSignals(True)
            ce.setText(txt)
            ce.blockSignals(False)
        self.__classes_buffer = txt

    def sync_mic_level(self, level: float):
        meter = getattr(self, "_PyQt6IOOperatorGUI__mic_meter", None)
        if meter is None:
            return
        try:
            v = max(0, min(100, int(float(level) * 100)))
            meter.setValue(v)
        except Exception:
            pass


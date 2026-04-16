from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import os
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
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame must be HxWx3 BGR")

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
    devices: list[DeviceInfo] = []
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i)
        ok = cap.isOpened()
        if ok:
            ret, _ = cap.read()
            if ret:
                devices.append(DeviceInfo(id=str(i), label=f"Camera {i}"))
        cap.release()
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
        all_devs = sd.query_devices()
        for idx, d in enumerate(all_devs):
            if int(d.get("max_input_channels", 0)) > 0:
                name = str(d.get("name", f"mic {idx}"))
                prefix = "* " if default_in == idx else ""
                devices.append(DeviceInfo(id=str(idx), label=f"{prefix}[{idx}] {name}"))
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

    def start(self):
        from PyQt6.QtCore import Qt, QTimer
        from PyQt6.QtGui import QFont, QAction
        from PyQt6.QtWidgets import (
            QApplication,
            QComboBox,
            QToolBar,
            QLabel,
            QMenu,
            QMainWindow,
            QPushButton,
            QSlider,
            QSpinBox,
            QVBoxLayout,
            QDialog,
            QFormLayout,
            QDoubleSpinBox,
            QDialogButtonBox,
            QComboBox as QComboBox2,
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
            QComboBox, QSpinBox {
                background: #151822; color: #d7dae0; border: 1px solid #22263a;
                padding: 4px 8px; border-radius: 6px;
            }
            QSlider::groove:horizontal { height: 4px; background: #22263a; border-radius: 2px; }
            QSlider::handle:horizontal { width: 12px; margin: -6px 0; border-radius: 6px; background: #4d7cff; }
            QPushButton {
                background: #151822; color: #d7dae0; border: 1px solid #22263a;
                padding: 5px 10px; border-radius: 6px;
            }
            QPushButton:pressed { background: #1c2030; }
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

        gain_badge = mk_badge("Gain: x1.0")
        gate_badge = mk_badge("Gate: off")
        fs_badge = mk_badge("FS: 44100")

        target_label = mk_label("TRACK ID")
        target_spin = QSpinBox()
        target_spin.setMinimum(-1)
        target_spin.setMaximum(999999)
        target_spin.setValue(-1)
        target_spin.setFixedWidth(110)
        target_spin.setToolTip("-1 = no target")

        status = QLabel("")
        f = QFont()
        f.setPixelSize(self.__style.status_font_px)
        status.setFont(f)
        status.setStyleSheet("color: #8b93a7;")
        status.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        toolbar = QToolBar("Controls")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setStyleSheet("QToolBar { border: 1px solid #151822; spacing: 8px; padding: 6px; }")
        toolbar.addWidget(cam_label)
        toolbar.addWidget(cam_combo)
        toolbar.addWidget(mic_label)
        toolbar.addWidget(mic_combo)
        toolbar.addWidget(gain_badge)
        toolbar.addWidget(gate_badge)
        toolbar.addWidget(fs_badge)
        toolbar.addSeparator()
        toolbar.addWidget(work_btn)
        toolbar.addWidget(zoom_out_btn)
        toolbar.addWidget(zoom_in_btn)
        toolbar.addWidget(zoom_badge)
        toolbar.addSeparator()
        toolbar.addWidget(conf_label)
        toolbar.addWidget(conf_slider)
        toolbar.addWidget(conf_badge)
        toolbar.addWidget(target_label)
        toolbar.addWidget(target_spin)
        toolbar.addSeparator()
        toolbar.addWidget(status)

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
        self.__gain_badge = gain_badge
        self.__gate_badge = gate_badge
        self.__fs_badge = fs_badge

        def push(ev_type: GUIEventType, value=None):
            self.__events.append(GUIEvent(ev_type, value))

        def refresh_devices():
            qt_videos, qt_audios = _qt_multimedia_devices()
            v4l = _scan_v4l2_devices()
            cv2_cams = _scan_opencv_cameras()
            sd_inputs = _sounddevice_inputs()

            # Prefer /dev/videoX listing for linux+ffmpeg path; fall back to Qt list.
            if os.name == "posix":
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

        # Light keyboard handling without a “game UI” vibe.
        def keyPressEvent(ev):
            k = ev.key()
            if k in (Qt.Key.Key_Escape, Qt.Key.Key_Q):
                push(GUIEventType.EXIT)
            elif k == Qt.Key.Key_Space:
                push(GUIEventType.TOGGLE_WORK)
            elif k in (Qt.Key.Key_Equal, Qt.Key.Key_Plus):
                push(GUIEventType.ZOOM_IN)
            elif k == Qt.Key.Key_Minus:
                push(GUIEventType.ZOOM_OUT)
            else:
                push(GUIEventType.UNKNOWN, int(k))

        win.keyPressEvent = keyPressEvent  # type: ignore

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

        win.show()

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

        img = _bgr_to_qimage(frame)
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
        if os.name == "posix":
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
        gb = self.__gain_badge
        if gb is not None:
            try:
                gb.setText(f"Gain: x{float(gain):.2g}")
            except Exception:
                pass

        tb = self.__gate_badge
        if tb is not None:
            try:
                thr = float(threshold)
                tb.setText("Gate: off" if thr <= 0 else f"Gate: {thr:.4f}")
            except Exception:
                pass

        fb = self.__fs_badge
        if fb is not None:
            try:
                fb.setText(f"FS: {int(fs)}")
            except Exception:
                pass


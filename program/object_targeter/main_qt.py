import threading
import time
import logging

import cv2
import numpy as np

from vosk import SetLogLevel

from config import WorkConfigManager
from serialwriter import SerialWriter
from zoom import ZoomController
from preprocessor import Preprocessor
from audio import AudioRecorder
from analyze import VideoAnalyzer
from logger import Logger
from smooth import SmoothingFilter
from commands import CommandParser
from angle import AngleCalculator

from interfaces import GUIEventType
from ioops_headless import IOOperatorHeadless
from qt_gui import PyQt6IOOperatorGUI


class OverlayLite:
    """
    Lighter on-screen text; keeps focus on video.
    Uses OpenCV drawing because model results are already OpenCV-friendly.
    """

    _FONT = cv2.FONT_HERSHEY_SIMPLEX

    def draw(self, frame: np.ndarray, results, config, zoom_level: float, colors_fn=None) -> np.ndarray:
        if frame is None:
            return frame

        h = frame.shape[0]

        if not results or results[0].boxes is None:
            cv2.putText(frame, "loading…", (12, h - 14), self._FONT, 0.7, (120, 120, 255), 2)
            return frame

        boxes = results[0].boxes
        ids = boxes.id
        has_track = ids is not None
        fh, fw, _ = frame.shape
        oh, ow = results[0].orig_shape
        scale_y, scale_x = fh / oh, fw / ow

        model_names = getattr(results[0], "names", {}) if results else {}
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].to(int).tolist()
            conf = float(boxes.conf[i].item())
            cls = int(boxes.cls[i].item())

            color = colors_fn(cls) if colors_fn else (70, 160, 255)
            thick = 5 if has_track and ids[i] == config.target_track else 2

            x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)

            if isinstance(model_names, dict):
                cls_name = str(model_names.get(cls, cls))
            elif isinstance(model_names, list) and cls < len(model_names):
                cls_name = str(model_names[cls])
            else:
                cls_name = str(cls)
            label = f"{cls_name} {conf*100:.0f}%"
            if has_track:
                label += f" #{int(ids[i])}"

            cv2.putText(frame, label, (x1, max(16, y1 - 8)), self._FONT, 0.65, color, 2)

        return frame


class PlatformQt:
    def __init__(self, size: tuple[int, int], fov: tuple[int, int], init_conf: float = 0.1,
                 videodev: str = "/dev/video2", fps: int = 30):
        SetLogLevel(-1)
        self.__size = size
        self.__init_conf = init_conf
        self.__logger = Logger(level=logging.INFO)

        self.__config_manager = WorkConfigManager(init_conf=self.__init_conf, logger=self.__logger)
        self.__calculator = AngleCalculator(fov, size)
        self.__writer = SerialWriter(calculator=self.__calculator, logger=self.__logger, size=self.__size)
        self.__zoom = ZoomController(writer=self.__writer, logger=self.__logger, min_zoom=1.0, max_zoom=5.0, step=0.5, size=self.__size)
        self.__preprocessor = Preprocessor(use_clahe=False, use_bilateral=False)
        self.__parser = CommandParser()

        self.__io = IOOperatorHeadless(videodev, fps, self.__size, self.__zoom, logger=self.__logger)
        self.__smoother = SmoothingFilter(window=2)
        self.__analyzer = VideoAnalyzer(
            io=self.__io,
            config_manager=self.__config_manager,
            zoom=self.__zoom,
            logger=self.__logger,
            serial_writer=self.__writer,
            smoother=self.__smoother,
            preprocessor=self.__preprocessor,
            imsize=self.__size,
            init_conf_score=self.__init_conf,
        )
        self.__io.analyzer = self.__analyzer

        # Audio recorder is restartable to switch device.
        self.__recorder = AudioRecorder(
            config_manager=self.__config_manager,
            zoom=self.__zoom,
            parser=self.__parser,
            logger=self.__logger,
        )
        self.__audio_settings = {"gain": 1.0, "threshold": 0.0, "fs": 44100}

        self.__overlay = OverlayLite()
        self.__gui = PyQt6IOOperatorGUI()

        self.__thread_classes = [self.__io, self.__analyzer, self.__recorder, self.__writer]
        self.__threads = [threading.Thread(target=c.start) for c in self.__thread_classes]

    def __restart_audio(self, device):
        try:
            self.__recorder.stop()
        except Exception:
            pass
        self.__recorder = AudioRecorder(
            config_manager=self.__config_manager,
            zoom=self.__zoom,
            parser=self.__parser,
            logger=self.__logger,
            device=device,
        )
        # Re-apply current mixer settings after restart.
        try:
            self.__recorder.set_gain(float(self.__audio_settings["gain"]))
            self.__recorder.set_gate_threshold(float(self.__audio_settings["threshold"]))
            self.__recorder.set_fs(int(self.__audio_settings["fs"]))
        except Exception:
            pass
        t = threading.Thread(target=self.__recorder.start)
        t.start()
        self.__threads.append(t)

    def run(self):
        # Start workers first.
        for t in self.__threads:
            t.start()

        # Start GUI on main thread.
        self.__gui.start()

        from PyQt6.QtCore import QTimer
        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is None:
            raise RuntimeError("QApplication is not available")

        def tick():
            # Apply GUI events.
            while True:
                ev = self.__gui.poll_event()
                if ev is None:
                    break
                if ev.type == GUIEventType.EXIT:
                    self.__config_manager.stop()
                elif ev.type == GUIEventType.TOGGLE_WORK:
                    self.__config_manager.stop()
                elif ev.type == GUIEventType.ZOOM_IN:
                    self.__zoom.zoom_in()
                elif ev.type == GUIEventType.ZOOM_OUT:
                    self.__zoom.zoom_out()
                elif ev.type == GUIEventType.CONF_CHANGED and isinstance(ev.value, float):
                    self.__config_manager.conf = float(ev.value)
                elif ev.type == GUIEventType.TARGET_TRACK_CHANGED:
                    self.__config_manager.target_track = ev.value if isinstance(ev.value, int) else None
                elif ev.type == GUIEventType.VIDEO_SOURCE_CHANGED and isinstance(ev.value, str):
                    # In the linux+ffmpeg pipeline we expect /dev/videoX here.
                    self.__io.set_videodev(ev.value)
                elif ev.type == GUIEventType.AUDIO_SOURCE_CHANGED:
                    if isinstance(ev.value, str) and ev.value.isdigit():
                        self.__restart_audio(int(ev.value))
                    elif isinstance(ev.value, int):
                        self.__restart_audio(ev.value)
                elif ev.type == GUIEventType.AUDIO_SETTINGS_CHANGED and isinstance(ev.value, dict):
                    try:
                        if "gain" in ev.value:
                            self.__audio_settings["gain"] = float(ev.value["gain"])
                        if "threshold" in ev.value:
                            self.__audio_settings["threshold"] = float(ev.value["threshold"])
                        if "fs" in ev.value:
                            self.__audio_settings["fs"] = int(ev.value["fs"])

                        if "gain" in ev.value:
                            self.__recorder.set_gain(float(ev.value["gain"]))
                        if "threshold" in ev.value:
                            self.__recorder.set_gate_threshold(float(ev.value["threshold"]))
                        if "fs" in ev.value:
                            self.__recorder.set_fs(int(ev.value["fs"]))
                    except Exception as e:
                        self.__logger.warning(f"[Audio] settings apply failed: {e}")

            if not self.__config_manager.to_work:
                self.__gui.set_status("stopping…")
                self.__gui.stop()
                app.quit()
                return

            raw = self.__io.get_latest_raw()
            if raw is None:
                self.__gui.set_status("waiting for frames…")
                return

            results = self.__analyzer.get_results()
            cfg = self.__config_manager.config
            zoom_level = self.__zoom.zoom / 10.0

            frame = raw.copy()
            try:
                frame = self.__overlay.draw(frame, results, cfg, zoom_level, colors_fn=self.__config_manager.colors)
            except Exception as e:
                self.__gui.set_status(f"overlay skipped: {e}")

            # Small overlay text (Qt supports Cyrillic; keep it here).
            lines = []
            if cfg.target_track is not None:
                lines.append(f"track #{cfg.target_track}")
            # Show selected classes in full to avoid hidden filters.
            keys = [k for k in cfg.names.keys() if k]
            if keys:
                lines.append("classes: " + ", ".join(keys[:8]) + ("…" if len(keys) > 8 else ""))
            self.__gui.set_overlay_text(lines)

            # Sync GUI controls in case voice changed them.
            self.__gui.sync_conf(cfg.conf)
            self.__gui.sync_target(cfg.target_track)
            self.__gui.sync_zoom(zoom_level)
            self.__gui.sync_audio_settings(
                float(self.__audio_settings["gain"]),
                float(self.__audio_settings["threshold"]),
                int(self.__audio_settings["fs"]),
            )

            self.__gui.set_status(f"cam {self.__io.videodev}")
            self.__gui.render_frame(frame)

        timer = QTimer()
        timer.setInterval(15)  # ~60Hz UI, actual rendering limited by camera
        timer.timeout.connect(tick)
        timer.start()

        try:
            app.exec()
        finally:
            try:
                self.__writer.stop()
            except Exception:
                pass
            try:
                self.__config_manager.stop()
            except Exception:
                pass
            for c in [self.__io, self.__analyzer, self.__recorder, self.__writer]:
                try:
                    c.stop()
                except Exception:
                    pass
            for t in self.__threads:
                try:
                    t.join(timeout=3)
                except Exception:
                    pass


if __name__ == "__main__":
    PlatformQt((1920, 1080), (58, 33)).run()


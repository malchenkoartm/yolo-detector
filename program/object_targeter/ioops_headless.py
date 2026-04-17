import subprocess
import threading
import time
import numpy as np

from interfaces import IIOOperator, IZoomController, ILogger, IAnalyzer
from threadmanager import ThreadManager


class IOOperatorHeadless(ThreadManager, IIOOperator):
    """
    Frame grabber based on ffmpeg+v4l2 that only stores latest frame.
    No OpenCV windows, no GUI dependencies.
    """

    def __init__(
        self,
        videodev: str | int,
        fps: int,
        size: tuple[int, int],
        zoom: IZoomController,
        logger: ILogger | None,
        input_format: str = "mjpeg",
    ):
        super().__init__()

        self.__analyzer: IAnalyzer | None = None
        self.__videodev = videodev
        self.__fps = fps
        self.__size = size
        self.__zoom = zoom
        self.__logger = logger
        self.__input_format = input_format

        w, h = self.__size
        self.__chunk = w * h * 3

        self.__latest_raw: np.ndarray | None = None
        self.__raw_lock = threading.Lock()

        self.__proc: subprocess.Popen | None = None
        self.__cv_cap = None

    @property
    def analyzer(self) -> IAnalyzer | None:
        return self.__analyzer

    @analyzer.setter
    def analyzer(self, analyzer: IAnalyzer):
        self.__analyzer = analyzer

    def get_latest_raw(self) -> np.ndarray | None:
        with self.__raw_lock:
            return self.__latest_raw.copy() if self.__latest_raw is not None else None

    @property
    def videodev(self) -> str | int:
        return self.__videodev

    def set_videodev(self, videodev: str | int):
        """
        Switch v4l2 device. Safe to call while running:
        we restart ffmpeg and continue producing frames.
        """
        if videodev == self.__videodev:
            return
        if self.__logger:
            self.__logger.info(f"[IO] video source → {videodev}")
        self.__videodev = videodev
        self.__restart_proc()

    def __use_ffmpeg_backend(self) -> bool:
        return isinstance(self.__videodev, str) and self.__videodev.startswith("/dev/video")

    def __build_cmd(self) -> list[str]:
        w, h = self.__size
        return [
            "ffmpeg",
            "-f",
            "v4l2",
            "-framerate",
            str(self.__fps),
            "-video_size",
            f"{w}x{h}",
            "-input_format",
            self.__input_format,
            "-fflags",
            "+nobuffer+discardcorrupt",
            "-avioflags",
            "direct",
            "-flags",
            "+low_delay",
            "-thread_queue_size",
            "1",
            "-i",
            self.__videodev,
            "-probesize",
            "32",
            "-analyzeduration",
            "0",
            "-pix_fmt",
            "bgr24",
            "-f",
            "rawvideo",
            "-",
        ]

    def __start_proc(self):
        if self.__use_ffmpeg_backend():
            cmd = self.__build_cmd()
            self.__proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
            )
            self.__cv_cap = None
            return

        import cv2

        dev = self.__videodev
        if isinstance(dev, str) and dev.isdigit():
            dev = int(dev)
        if isinstance(dev, int):
            backends = []
            if hasattr(cv2, "CAP_DSHOW"):
                backends.append(cv2.CAP_DSHOW)
            if hasattr(cv2, "CAP_MSMF"):
                backends.append(cv2.CAP_MSMF)
            backends.append(cv2.CAP_ANY)
            self.__cv_cap = None
            for be in backends:
                cap = None
                try:
                    cap = cv2.VideoCapture(dev, be)
                    if cap.isOpened():
                        ok, _ = cap.read()
                        if ok:
                            self.__cv_cap = cap
                            break
                except Exception as e:
                    if self.__logger:
                        self.__logger.warning(f"[IO] backend init failed ({be}): {e}")
                finally:
                    if cap is not None and self.__cv_cap is not cap:
                        cap.release()
        else:
            try:
                self.__cv_cap = cv2.VideoCapture(dev)
            except Exception as e:
                self.__cv_cap = None
                if self.__logger:
                    self.__logger.warning(f"[IO] camera open failed ({dev}): {e}")
        if self.__cv_cap is None or not self.__cv_cap.isOpened():
            if self.__cv_cap is not None:
                self.__cv_cap.release()
            self.__cv_cap = None
            raise RuntimeError(f"Cannot open camera: {self.__videodev}")
        # On Windows, forcing unsupported camera mode can produce corrupted frames.
        try:
            self.__cv_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass
        self.__cv_cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.__size[0]))
        self.__cv_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.__size[1]))
        self.__cv_cap.set(cv2.CAP_PROP_FPS, float(self.__fps))
        self.__proc = None

    def __stop_proc(self):
        cap = self.__cv_cap
        self.__cv_cap = None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

        proc = self.__proc
        self.__proc = None
        if not proc:
            return

        try:
            if proc.stdout:
                proc.stdout.close()
        except Exception:
            pass

        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=2)
            except Exception:
                pass

    def __restart_proc(self):
        self.__stop_proc()
        if not self._stop_event.is_set():
            self.__start_proc()

    def stop(self):
        super().stop()
        self.__stop_proc()

    def start(self):
        w, h = self.__size
        total_frames = 0
        record_start_time = time.time()

        while not self._stop_event.is_set():
            if self.__proc is None and self.__cv_cap is None:
                try:
                    self.__start_proc()
                except Exception as e:
                    if self.__logger:
                        self.__logger.error(f"[IO] camera start failed: {e}")
                    time.sleep(0.7)
                    continue

            cap = self.__cv_cap
            if cap is not None:
                ok, frame = cap.read()
                if not ok or frame is None:
                    if self.__logger:
                        self.__logger.warning("[IO] short read from camera; restarting")
                    try:
                        self.__restart_proc()
                    except Exception as e:
                        if self.__logger:
                            self.__logger.error(f"[IO] camera restart failed: {e}")
                        time.sleep(0.2)
                    continue
                # Normalize camera output to BGR for downstream pipeline.
                try:
                    if frame.ndim == 2:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    elif frame.ndim == 3 and frame.shape[2] == 4:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    elif frame.ndim == 3 and frame.shape[2] == 1:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                except Exception:
                    pass
                if frame.shape[:2] != (h, w):
                    frame = np.ascontiguousarray(frame)
                    try:
                        # Keep camera native aspect and only scale for model input.
                        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA if frame.shape[1] > w else cv2.INTER_LINEAR)
                    except Exception:
                        pass
            else:
                proc = self.__proc
                if not proc or not proc.stdout:
                    time.sleep(0.05)
                    continue
                raw = proc.stdout.read(self.__chunk)
                if len(raw) < self.__chunk:
                    if self.__logger:
                        self.__logger.warning("[IO] short read from ffmpeg; restarting")
                    self.__restart_proc()
                    time.sleep(0.05)
                    continue
                frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3).copy()
            frame = self.__zoom.apply(frame)

            with self.__raw_lock:
                self.__latest_raw = frame

            total_frames += 1

        if self.__logger:
            elapsed = time.time() - record_start_time
            if elapsed > 0:
                self.__logger.info(f"IO FPS average: {total_frames / elapsed:.1f}")


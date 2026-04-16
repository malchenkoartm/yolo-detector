
import cv2
import torch
import time
import subprocess
import numpy as np
import tkinter as tk

from interfaces import IIOOperator, IZoomController, ILogger, IWorkConfigManager, WorkConfig, IAnalyzer 

import threading
from ultralytics.engine.results import Results
from threadmanager import ThreadManager

class Overlay:
    __TEXT_THICKNESS = 2
    __NORMAL_THICKNESS = 2
    __TARGET_THICKNESS = 5
    __FONT = cv2.FONT_HERSHEY_COMPLEX
    __FONT_SCALE = 1
    __SMALL_FONT_SCALE = 0.7

    def draw(self, frame: np.ndarray, results: list[Results], config: WorkConfig, 
             zoom_level: float, colors_fn=None):
        
        h = frame.shape[0]
    
        cv2.putText(frame, f"Зум: x{zoom_level:.1f}", (10, 25),
                    self.__FONT, self.__SMALL_FONT_SCALE, (0, 255, 255), self.__TEXT_THICKNESS)
        cv2.putText(frame, f"Уверенность: {config.conf*100:.0f}%", (10, 50),
                    self.__FONT, self.__SMALL_FONT_SCALE, (0, 255, 255), self.__TEXT_THICKNESS)
        if config.target_track is not None:
            cv2.putText(frame, f"{config.target_track} отслеживается", (10, 75),
                        self.__FONT, self.__SMALL_FONT_SCALE, (0, 255, 255), self.__TEXT_THICKNESS)
            
        if not results or results[0].boxes is None:
            cv2.putText(frame, "Загрузка классов", (10, h - 10),
                self.__FONT, self.__SMALL_FONT_SCALE,
                (0, 0, 255), self.__TEXT_THICKNESS)
            return frame
        
        boxes = results[0].boxes
        ids = boxes.id
        has_track = ids is not None
        fh, fw, _ = frame.shape
        oh, ow = results[0].orig_shape
        
        scale_y, scale_x = fh/oh, fw/ow
        keys = list(config.names.keys())

        for i, name in enumerate(keys):
            cv2.putText(frame, name, (10, h - 10 - i * 15),
                        self.__FONT, self.__SMALL_FONT_SCALE,
                        (255, 0, 128), self.__TEXT_THICKNESS)

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].to(torch.int).tolist()
            conf = boxes.conf[i].item()
            cls = int(boxes.cls[i].item())

            color = colors_fn(cls) if colors_fn else (0, 255, 0)
            thickness = self.__TARGET_THICKNESS if has_track and ids[i] == config.target_track \
                else self.__NORMAL_THICKNESS
                
            x1, y1, x2, y2 = int(x1*scale_x), int(y1 * scale_y), int(x2* scale_x), int(y2*scale_y)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            label = f"{keys[cls]}, ув = {conf*100:.0f}%" + (f' id = {ids[i]}' if has_track else '')
            cv2.putText(frame, label, (x1, y1 - 8),
                        self.__FONT, self.__FONT_SCALE, color, self.__TEXT_THICKNESS)

        return frame


class IOOperator(ThreadManager, IIOOperator):

    def __init__(self, videodev: str, fps: int, size: tuple[int, int], zoom: IZoomController,
                 config_manager: IWorkConfigManager, logger: ILogger | None):
        super().__init__()
        
        self.__analyzer = None
        
        self.__size = size
        w, h = self.__size
        self.__zoom = zoom
        self.__config_manager = config_manager
        self.__logger = logger

        self.__chunk = w * h * 3

        self.__latest_raw: np.ndarray | None = None
        self.__raw_lock = threading.Lock()

        self.__overlay = Overlay()

        root = tk.Tk()
        self.__screen_w = root.winfo_screenwidth()
        self.__screen_h = root.winfo_screenheight()
        root.destroy()

        cmd = [
            "ffmpeg", "-f", "v4l2", "-framerate", str(fps),
            "-video_size", f"{w}x{h}",
            "-input_format", "mjpeg",
            "-fflags", "+nobuffer+discardcorrupt",
            "-avioflags", "direct",
            "-flags", "+low_delay",
            "-thread_queue_size", "1",
            "-i", videodev,
            "-probesize", "32",
            "-analyzeduration", "0",
            "-pix_fmt", "bgr24", "-f", "rawvideo", "-"
        ]
        self.__proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )

        if self.__logger:
            self.__logger.info(f"Resolution: {w}x{h}, "
                               f"Screen: {self.__screen_w}x{self.__screen_h}")

    def resize_to_window(self, frame: np.ndarray, win_name: str, min_delta: int = 12) -> np.ndarray:
        rect = cv2.getWindowImageRect(win_name)
        if not rect or rect[2] <= 0 or rect[3] <= 0:
            return frame
        
        w, h = int(rect[2]), int(rect[3])
        if not hasattr(self, '_win_sz'): self._win_sz = (0, 0)
        if abs(w - self._win_sz[0]) < min_delta and abs(h - self._win_sz[1]) < min_delta:
            w, h = self._win_sz
        else:
            self._win_sz = (w, h)
            
        fh, fw = frame.shape[:2]
        scale = min(w / fw, h / fh)
        nw, nh = int(fw * scale), int(fh * scale)
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        return cv2.resize(frame, (nw, nh), interpolation=interp)
    
    @property        
    def analyzer(self):
        return self.__analyzer
    
    @analyzer.setter        
    def analyzer(self, analyzer: IAnalyzer):
        self.__analyzer = analyzer

    def start(self):
        analyzer = self.__analyzer
        if not analyzer:
            raise ValueError("First should be set analyzer")
        
        total_frames = 0
        record_start_time = time.time()
        w, h = self.__size
        winname = "Tracking"

        cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(winname, w, h)  # начальный размер


        while not self._stop_event.is_set():
            raw = self.__proc.stdout.read(self.__chunk)
            if len(raw) < self.__chunk:
                if self.__logger:
                    self.__logger.error("Readed less than chunk data")
                break

            frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3).copy()
            frame = self.__zoom.apply(frame)

            with self.__raw_lock:
                self.__latest_raw = frame

            model_results = analyzer.get_results()
            config = self.__config_manager.config
            if not config.to_work:
                self.__stop()
                break
            
            try:

                display_frame = self.__overlay.draw(
                    frame, model_results, config,
                    self.__zoom.zoom / 10,
                    colors_fn=self.__config_manager.colors
                )
            except Exception as e:
                if self.__logger:
                    self.__logger.warning(f"Skipping drawing frame: {e}")
                display_frame = frame

#            display_frame = self.resize_to_window(display_frame, winname)
            cv2.imshow(winname, display_frame)
            cv2.waitKey(1)

            total_frames += 1

        self.__proc.terminate()
        self.__proc.wait()
        if self.__logger:
            elapsed = time.time() - record_start_time
            self.__logger.info(
                f"IO FPS average: {total_frames / elapsed:.1f}"
            )
        cv2.destroyAllWindows()

    def get_latest_raw(self) -> np.ndarray | None:
        with self.__raw_lock:
            return self.__latest_raw.copy() if self.__latest_raw is not None else None

    def __stop(self):
        
        self.stop()

        if self.__proc.stdout:
            if self.__logger:
                self.__logger.info("Closing ffmpeg stdout")
            self.__proc.stdout.close()

        try:
            if self.__logger:
                self.__logger.info("Terminating ffmpeg process")
            self.__proc.terminate()
            self.__proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            if self.__logger:
                self.__logger.warning("Can't terminate ffmpeg — killing it")
            self.__proc.kill()
            self.__proc.wait()
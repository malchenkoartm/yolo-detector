import cv2
from ultralytics import YOLOWorld
import time
import numpy as np
import threading

from interfaces import IVideoAnalyzer, IWorkConfigManager, WorkConfig, ISerialWriter
from interfaces import IZoomController, ISmoothingFilter, ILogger, IIOOperator, IPreprocessor
from selector import ObjectSelector

from threadmanager import ThreadManager

class VideoAnalyzer(ThreadManager, IVideoAnalyzer):
    def __init__(self,
                 io: IIOOperator,
                 config_manager: IWorkConfigManager,
                 zoom: IZoomController,
                 serial_writer: ISerialWriter,
                 smoother: ISmoothingFilter,
                 preprocessor: IPreprocessor | None,
                 logger: ILogger | None = None,
                 model_name: str ="yolov8m-worldv2.pt",
                 imsize: tuple[int, int] = (1920, 1080),
                 model_imsize: tuple[int,int] | None = None,
                 init_conf_score: float=0.1):
        super().__init__()
        
        self.__config_manager = config_manager
        self.__serial_writer = serial_writer
        self.__smoother = smoother
        self.__zoom = zoom
        self.__preprocessor = preprocessor
        self.__logger = logger
        if logger:
            logger.info(f"io image size: {imsize}, model image size: {model_imsize}")
        self.__imsize = imsize
        self.__model_imsize = model_imsize

        self.__results_lock = threading.Lock()
        self.__model_results = None

        self.__model = YOLOWorld(model_name)
        self.__model.to('cuda')

        self.__conf_score = init_conf_score

        self.__io = io
        
    def get_results(self):
        with self.__results_lock:
            return self.__model_results

    def __set_classes(self, config: WorkConfig):
        if config.names_updated:
            with self.__results_lock:
                self.__model_results = None
            start_time = time.time()
            self.__model.model.clip_model = None
            self.__model.set_classes(config.names.values())
            if self.__logger:
                self.__logger.info(
                    f"Classes changed: {list(config.names.keys())} "
                    f"in {time.time() - start_time:.2f}s"
                )
    
    @staticmethod
    def resize_frame(frame: np.ndarray, model_imsize: tuple[int,int] | None, imsize: tuple[int,int]) -> np.ndarray:
        if model_imsize and model_imsize != imsize:
            frame = cv2.resize(frame, dsize = model_imsize, interpolation=cv2.INTER_AREA)
        return frame
    
    @staticmethod
    def resize_coords(coords: tuple[int, int] | None, model_imsize: tuple[int,int] | None, imsize: tuple[int,int]) -> tuple[int, int] | None:
        if not coords or not model_imsize or model_imsize == imsize:
            return coords
        return tuple(int(c * o / s) for (c, s, o) in zip(coords, model_imsize, imsize))

    def start(self):
        total_frames = 0
        record_start_time = time.time()
        model_imsize = self.__model_imsize
        imsize = self.__imsize
        
        track_kwargs = {
            "persist": True,
            "verbose": False,
            "conf": self.__conf_score
        }
        if model_imsize:
            track_kwargs["imgsz"] = list(model_imsize)

        while not self._stop_event.is_set():
            config = self.__config_manager.update_names()
            
            if not self.__config_manager.to_work:
                self.stop()
                break
            
            track_kwargs['conf'] = config.conf
            
            self.__set_classes(config)
            
            frame = self.__io.get_latest_raw()
            if frame is None:
                if self.__logger:
                    self.__logger.trace("Frame is None")
                continue
            
            frame = self.resize_frame(frame, model_imsize, imsize)
            
            preprocessed = frame
            if self.__preprocessor:
                preprocessed = self.__preprocessor.apply(frame)

            model_results = self.__model.track(
                preprocessed, **track_kwargs
            )

            with self.__results_lock:
                self.__model_results = model_results

            coords = ObjectSelector.select(results=model_results, target=config.target_track)
            
            if coords:
                coords = self.resize_coords(coords, model_imsize, imsize)
                zoomed_coords = self.__zoom.to_original_coords(coords[0], coords[1])
                coords = self.__smoother.update(zoomed_coords)

                if self.__logger:
                    self.__logger.trace(f"Video sends coords: {coords}, zoom: {zoomed_coords}")
                self.__serial_writer.coords = coords

            total_frames += 1

        self.__serial_writer.stop()
        
        if self.__logger:
            elapsed = time.time() - record_start_time
            self.__logger.info(
                f"Analyzer FPS average: {total_frames / elapsed:.1f}"
            )
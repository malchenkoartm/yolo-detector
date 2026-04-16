import threading
from vosk import SetLogLevel
from config import WorkConfigManager
from serialwriter import SerialWriter
from zoom import ZoomController
from preprocessor import Preprocessor
from audio import AudioRecorder
from analyze import VideoAnalyzer
from ioops import IOOperator
from logger import Logger
from smooth import SmoothingFilter
from commands import CommandParser
from angle import AngleCalculator

import logging

class Platform:
    def __init__(self, size: tuple[int, int], fov:tuple[int, int], init_conf: float = 0.1):
        SetLogLevel(-1)
        self.__size = size
        self.__init_conf = init_conf
        self.__logger = Logger(level=logging.DEBUG)
        self.__config_manager = WorkConfigManager(init_conf=self.__init_conf, logger = self.__logger)
        self.__calculator = AngleCalculator(fov, size)
        self.__writer = SerialWriter(calculator=self.__calculator, logger = self.__logger, size=self.__size)
        self.__zoom = ZoomController(writer= self.__writer, logger = self.__logger, min_zoom=1.0, max_zoom=5.0, step=0.5, size = self.__size)
        self.__preprocessor = Preprocessor(use_clahe=False, use_bilateral=False)
        self.__parser = CommandParser()
        self.__recorder = AudioRecorder(config_manager=self.__config_manager, zoom=self.__zoom, parser=self.__parser, logger=self.__logger)
        
        self.__io = IOOperator("/dev/video2", 30, self.__size, self.__zoom, self.__config_manager, logger=self.__logger)
        self.__smoother = SmoothingFilter(window=2)

        self.__analyzer = VideoAnalyzer(io = self.__io, config_manager=self.__config_manager, 
                                        zoom=self.__zoom, logger= self.__logger,
                                        serial_writer=self.__writer, smoother= self.__smoother,
                                        preprocessor=self.__preprocessor,
                                        imsize=self.__size, init_conf_score=self.__init_conf)
        self.__io.analyzer = self.__analyzer
        
        self.__thread_classes = [self.__io, self.__analyzer, self.__recorder, self.__writer]
        
        self.__threads = [
            threading.Thread(target=c.start) for c in self.__thread_classes
        ]

    def run(self):
        for t in self.__threads:
            t.start()
            
        try:
            for t in self.__threads:
                t.join()
        except Exception as e:
            if self.__logger:
                self.__logger.info(f"\nОстановка: {e}")
        finally:
            self.__writer.stop()
            self.__config_manager.stop()
            for t in self.__threads:
                t.join(timeout=3)
            if self.__logger:
                self.__logger.info("Завершено.")

if __name__ == '__main__':
    Platform((1920, 1080), (58, 33)).run()
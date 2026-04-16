from interfaces import IWorkConfigManager, WorkConfig, ILogger

import threading
from ultralytics.utils.plotting import Colors
import argostranslate.translate as translate
from model_names import en_model_names
import time

class WorkConfigManager(IWorkConfigManager):

    def __init__(self, init_conf: float, logger: ILogger | None = None):
        self.__translater = translate.get_translation_from_codes(from_code="ru", to_code="en")
        self.__colors = Colors()
        self.__names = {"": ""}
        self.__conf = init_conf
        self.__to_work = True
        self.__lock = threading.Lock()
        self.__logger = logger
        self.__target_track = None

        self.__updated: bool = True

    def place(self, ru_text):
        if ru_text is not None:
            start_time = time.time()
            translates = self.__translater.hypotheses(ru_text)
            common = [x.value for x in translates if x.value in en_model_names]
            en_text = common[0] if common else translates[0].value
            if self.__logger:
                self.__logger.info(f"translated: {en_text}, time = {(time.time() - start_time):.2f}")
            with self.__lock:
                self.__names = {ru_text: en_text}
                self.__updated = True

    def add(self, ru_text):
        if ru_text is not None:
            start_time = time.time()
            translates = self.__translater.hypotheses(ru_text)
            common = [x.value for x in translates if x.value in en_model_names]
            en_text = common[0] if common else translates[0].value
            if self.__logger:
                self.__logger.info(f"translated: {en_text}, time = {(time.time() - start_time):.2f}")
            with self.__lock:
                self.__names[ru_text] = en_text
                self.__updated = True

    @property
    def config(self):
        with self.__lock:
            return WorkConfig(self.__names.copy(), self.__updated, self.__conf, self.__target_track, self.__to_work)
        
    def update_names(self):
        with self.__lock:
            updated = self.__updated
            self.__updated = False
            return WorkConfig(self.__names.copy(), updated, self.__conf, self.__target_track, self.__to_work)
    
    @property
    def target_track(self):
        with self.__lock:
            return self.__target_track
    
    @target_track.setter
    def target_track(self, target_track: int | None):
        with self.__lock:
            self.__target_track = target_track
    
    @property
    def conf(self):
        with self.__lock:
            return self.__conf
    
    @conf.setter
    def conf(self, conf: float):
        with self.__lock:
            self.__conf = conf
        
    @property
    def colors(self):
        return self.__colors
    
    @property
    def to_work(self):
        with self.__lock:
            return self.__to_work

    def stop(self):
        with self.__lock:
            self.__to_work = False
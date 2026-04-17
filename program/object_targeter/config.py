from interfaces import IWorkConfigManager, WorkConfig, ILogger

import threading
from ultralytics.utils.plotting import Colors
import argostranslate.translate as translate
from model_names import en_model_names, ru_model_names
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
        self.__en_names_set = {str(v).strip().lower() for v in en_model_names.values()}
        self.__ru_to_en = {str(ru).strip().lower(): str(en_model_names[idx]).strip().lower()
                           for idx, ru in ru_model_names.items() if idx in en_model_names}

    def __normalize_to_en(self, text: str) -> str:
        t = str(text or "").strip().lower()
        if not t:
            return t
        if t in self.__en_names_set:
            return t
        if t in self.__ru_to_en:
            return self.__ru_to_en[t]
        # Fallback: ru->en MT
        translates = self.__translater.hypotheses(t)
        guess = str(translates[0].value).strip().lower() if translates else t
        return guess if guess else t

    def place(self, ru_text):
        if ru_text is not None:
            start_time = time.time()
            en_text = self.__normalize_to_en(ru_text)
            if self.__logger:
                self.__logger.info(f"translated: {en_text}, time = {(time.time() - start_time):.2f}")
            with self.__lock:
                self.__names = {ru_text: en_text}
                self.__updated = True

    def add(self, ru_text):
        if ru_text is not None:
            start_time = time.time()
            en_text = self.__normalize_to_en(ru_text)
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
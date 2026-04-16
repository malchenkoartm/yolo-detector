from interfaces import IThreadManager

from abc import ABC, abstractmethod
import threading

class ThreadManager(IThreadManager, ABC):
    def __init__(self):
        super().__init__()
        self._stop_event = threading.Event()
        
    def stop(self):
        self._stop_event.set()
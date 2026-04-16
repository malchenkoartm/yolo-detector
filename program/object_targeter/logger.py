import logging
import os
from datetime import datetime

from interfaces import ILogger

class Logger(ILogger):
    def __init__(self, log_dir: str = "logs", level = logging.INFO ):
        os.makedirs(log_dir, exist_ok=True)
        filename = os.path.join(log_dir,
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(filename),
                logging.StreamHandler()
            ]
        )
        self.__log = logging.getLogger("App")
        
    def trace(self, msg: str):
        self.__log.debug(msg)

    def info(self, msg: str):
        self.__log.info(msg)

    def warning(self, msg: str):
        self.__log.warning(msg)

    def error(self, msg: str):
        self.__log.error(msg)
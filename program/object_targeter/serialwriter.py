import threading
import time
import serial
from utils import get_distance, is_in_ellipse
from logger import ILogger
from threadmanager import ThreadManager
from interfaces import ISerialWriter, IAngleCalculator

class SerialWriter(ThreadManager, ISerialWriter):
    def __init__(self, calculator: IAngleCalculator, dev_name = '/dev/ttyUSB0',  baudrate = 9600,
                 logger: ILogger | None = None,
                 size = (1920, 1080), notsend_zone_factor = 0.3):
        super().__init__()
        
        self.__calculator = calculator
        
        self.__ser = serial.Serial(dev_name, baudrate)
        self.__size = size
        self.__screen_center = (size[0] // 2, size[1] // 2)
        self.__notsend_zone = (self.__size[0] * notsend_zone_factor, self.__size[1] * notsend_zone_factor)
        self.__notsend_zone_factor = notsend_zone_factor
        
        self.__logger = logger
        
        self.__lock = threading.Lock()
        self.__coords = None
    
    @property 
    def not_send_zone(self):
        return self.__notsend_zone
    
    @not_send_zone.setter
    def not_send_zone(self, zone):
        with self.__lock:
            self.__notsend_zone = zone
            
    def update_notsend_zone_by_size(self, size):
        with self.__lock:
            self.__notsend_zone = (size[0] * self.__notsend_zone_factor, size[1] * self.__notsend_zone_factor)
        
    @property
    def coords(self):
        with self.__lock:
            return self.__coords
        
    @coords.setter
    def coords(self, coords):
        with self.__lock:
            self.__coords = coords
    
    def start(self):
        if self.__logger:
            self.__logger.info('SerialWriter Loop started')
                    
        while not self._stop_event.is_set():
            data_to_send = None
            
            with self.__lock:
                current_val = self.__coords
                self.__coords = None
                
            if current_val is not None:
                a = self.__notsend_zone[0] / 2
                b = self.__notsend_zone[1] / 2
                if not is_in_ellipse(self.__screen_center, current_val, a, b):
                    angles = self.__calculator.calculate(self.__size[0] - current_val[0], current_val[1])
                    data_to_send = f'{int(angles[0])}:{int(angles[1])}\n'
                else:
                    data_to_send = None
                    
            
            if data_to_send:
                try:
                    if self.__logger:
                        self.__logger.trace(f'Angles to send: {data_to_send}')
                    self.__ser.reset_input_buffer()
                    self.__ser.write(data_to_send.encode('utf-8'))
                    self.__ser.flush()
                    time.sleep(0.025) 
                except Exception as e:
                    if self.__logger:
                        self.__logger.error(f"Serial Write Error: {e}")
                    break
            
            time.sleep(0.005)
            
        self.__ser.close()
        if self.__logger:
            self.__logger.info("SerialWriter stopped")
        
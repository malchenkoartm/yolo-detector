import threading
import cv2
from interfaces import ISerialWriter, ILogger, IZoomController

class ZoomController(IZoomController):
    def __init__(self, writer: ISerialWriter, size, logger: ILogger | None = None, min_zoom=1.0, max_zoom=5.0, step=0.5):
        self.__zoom = 10
        self.__init_size = size
        self.__cropped_size = size
        self.__min = int(10*min_zoom)
        self.__max = int(10*max_zoom)
        self.__step = int(step*10)
        self.__writer = writer
        self.__lock = threading.Lock()
        
        self.__logger = logger
        
    def __update_cropped_size(self):
        if self.__zoom == 10:
            self.__cropped_size = self.__init_size
            self.__writer.update_notsend_zone_by_size(self.__cropped_size)
            return
        float_zoom = self.__zoom / 10
        w, h = self.__init_size
        self.__cropped_size = (int(w / float_zoom), int(h / float_zoom))
        self.__writer.update_notsend_zone_by_size(self.__cropped_size)

    def zoom_in(self):
        with self.__lock:
            self.__zoom = min(self.__zoom + self.__step, self.__max)
            if self.__logger:
                self.__logger.info(f"[Zoom] приблизить → x{(self.__zoom/10):.1f}")
            self.__update_cropped_size()

    def zoom_out(self):
        with self.__lock:
            self.__zoom = max(self.__zoom - self.__step, self.__min)
            if self.__logger:
                self.__logger.info(f"[Zoom] удалить → x{(self.__zoom/10):.1f}")
            self.__update_cropped_size()

    @property
    def zoom(self):
        with self.__lock:
            return self.__zoom

    def get_state(self):
        with self.__lock:
            return self.__zoom, self.__cropped_size

    def apply(self, frame):
        _, cropped_size = self.get_state()
        w, h = self.__init_size
        
        new_w, new_h = cropped_size

        y1 = (h - new_h) // 2
        x1 = (w - new_w) // 2
        cropped = frame[y1:y1 + new_h, x1:x1 + new_w]

        return cv2.resize(cropped, (w, h))
    
    def to_original_coords(self, cx: int, cy: int) -> tuple[int, int]:
        zoom, cropped_size = self.get_state()
        if zoom == 10:
            return cx, cy
        
        float_zoom = zoom / 10
        
        orig_w, orig_h = self.__init_size
        crop_w, crop_h = cropped_size
        
        offset_x = (orig_w - crop_w) // 2
        offset_y = (orig_h - crop_h) // 2

        orig_cx = int(cx / float_zoom) + offset_x
        orig_cy = int(cy / float_zoom) + offset_y
        return orig_cx, orig_cy
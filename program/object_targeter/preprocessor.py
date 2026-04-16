import cv2
from interfaces import IPreprocessor

class Preprocessor(IPreprocessor):
    def __init__(self,
                 use_clahe: bool = True,
                 use_bilateral: bool = True,
                 clahe_clip: float = 2.0,
                 clahe_grid: tuple = (8, 8),
                 bilateral_d: int = 5,
                 bilateral_sigma: int = 75):
        self.__use_clahe = use_clahe
        self.__use_bilateral = use_bilateral
        self.__bilateral_d = bilateral_d
        self.__bilateral_sigma = bilateral_sigma
        self.__clahe = cv2.createCLAHE(clipLimit=clahe_clip,
                                        tileGridSize=clahe_grid)
        
    def apply(self, frame):
        if self.__use_bilateral:
            frame = cv2.bilateralFilter(frame, self.__bilateral_d,
                                        self.__bilateral_sigma,
                                        self.__bilateral_sigma)
        if self.__use_clahe:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = self.__clahe.apply(l)
            frame = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
            
        return frame
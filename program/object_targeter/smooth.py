from collections import deque
from interfaces import ISmoothingFilter

class SmoothingFilter(ISmoothingFilter):
    def __init__(self, window: int = 4):
        self.__xs = deque(maxlen=window)
        self.__ys = deque(maxlen=window)

    def update(self, coords: tuple[int, int] | None) -> tuple[int, int] | None:
        if not coords:
            return coords
        self.__xs.append(coords[0])
        self.__ys.append(coords[1])
        return int(sum(self.__xs) / len(self.__xs)), \
               int(sum(self.__ys) / len(self.__ys))

    def reset(self):
        self.__xs.clear()
        self.__ys.clear()
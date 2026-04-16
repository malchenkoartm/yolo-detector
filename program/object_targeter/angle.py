from interfaces import IAngleCalculator

class AngleCalculator(IAngleCalculator):
    def __init__(self, fov: tuple[int, int], resolution: tuple[int, int]):
        self.__fov = fov
        self.__res = resolution
        
    @staticmethod
    def calculate_axis(coord: float, view: int, size: int) -> float:
        bias = coord - size/2
        return bias*view/size
        
    def calculate(self, x, y) -> tuple[float,float]:
        return self.calculate_axis(x,self.__fov[0], self.__res[0]),\
            self.calculate_axis(y, self.__fov[1], self.__res[1])
    
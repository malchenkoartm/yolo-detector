#utils.py
def get_distance(first: tuple[int, int], second: tuple[int, int]) -> bool:
    return ((second[0] - first[0])**2 + ((second[1] - first[1])**2))**0.5

# x, y
def is_in_ellipse(c: tuple[int, int], p: tuple[int, int], a: float, b: float) -> bool:
    """ Is point in ellips

    Args:
        c (tuple[int, int]): center of ellipse (x,y)
        p (tuple[int, int]): point (x, y)
        a (int): big semi-axis (x)
        b (int): little semi-axis (y)
    """
    dx = p[0] - c[0]
    dy = p[1] - c[1]
    return (dx*dx)/(a*a) + (dy*dy)/(b*b) <= 1
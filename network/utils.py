
def find_largest_factors(c: int):
    """
    Returns the two largest factors a and b of an integer c, such that a * b = c.
    """
    for a in range(int(c**0.5), 0, -1):
        if c % a == 0:
            b = c // a
            return b, a
    return 1, c

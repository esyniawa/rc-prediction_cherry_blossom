import numpy as np
import os


def safe_save(save_name: str, array: np.ndarray) -> None:
    """
    If a folder is specified and does not yet exist, it will be created automatically.
    :param save_name: full path + data name
    :param array: array to save
    :return:
    """
    # create folder if not exists
    folder, data_name = os.path.split(save_name)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)

    if data_name[-3:] == 'npy':
        np.save(save_name, array)
    else:
        np.save(save_name + '.npy', array)


def find_largest_factors(c: int):
    """
    Returns the two largest factors a and b of an integer c, such that a * b = c.
    """
    for a in range(int(c**0.5), 0, -1):
        if c % a == 0:
            b = c // a
            return b, a
    return 1, c

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


def load_lasso_weights(
        save_path: str,
        n_runs: int):

    ws = []
    for run in range(n_runs):
        folder = f'{save_path}/run_{run}/lasso_weights.npy'
        w = np.load(folder)
        ws.append(w)

    ws = np.array(ws)
    return np.mean(ws, axis=0), ws

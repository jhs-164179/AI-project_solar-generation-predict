import os
import random
import numpy as np
import tensorflow as tf


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)


def minmax(arr, reverse=False, x_max=None, x_min=None, axis=None):
    if reverse:
        arr = arr * (x_max - x_min) + x_min
        return arr
    else:
        if x_min==None:
            x_min = np.min(arr, axis=axis)
        if x_max==None:
            x_max = np.max(arr, axis=axis)
        arr = (arr - x_min) / (x_max - x_min)
        return arr, x_min, x_max    
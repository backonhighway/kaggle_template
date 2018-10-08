import numpy as np
import math


def inv_y(a): return np.exp(a)


def exp_rmspe(y_pred, y_true):
    y_true = inv_y(y_true)
    pct_var = (y_true - inv_y(y_pred)) / y_true
    return math.sqrt((pct_var**2).mean())
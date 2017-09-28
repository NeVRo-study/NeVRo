# coding=utf-8
"""
Some meta functions

Author: Simon Hofmann | <[surname].[lastname][at]protonmail.com> | 2017
"""

import datetime
from functools import wraps
import numpy as np


# Timer
def function_timed(funct):
    """
    This allows to define new function with the timer-wrapper
    Write:
        @function_timed
        def foo():
            print("Any Function")
    And try:
        foo()
    http://stackoverflow.com/questions/2245161/how-to-measure-execution-time-of-functions-automatically-in-python
    """

    @wraps(funct)
    def wrapper(*args, **kwds):
        start_timer = datetime.datetime.now()

        output = funct(*args, **kwds)  # == function()

        duration = datetime.datetime.now() - start_timer

        print("Processing time of {}: {} [h:m:s:ms]".format(funct.__name__, duration))

        return output

    return wrapper

# @function_timed
# def foo():
#     print("Any Function")
#
# foo()


def normalization(array, lower_bound, upper_bound):
    """
    Normalizes Input Array
    :param upper_bound: Upper Bound b
    :param lower_bound: Lower Bound a
    :param array: To be transformed array
    :return: normalized array
    """

    assert lower_bound < upper_bound, "lower_bound must be < upper_bound"

    a, b = lower_bound, upper_bound

    normed_array = (b-a) * ((array - np.nanmin(array)) / (np.nanmax(array) - np.nanmin(array))) + a

    return normed_array


def z_score(array):
    """
    Create z-score
    :return: z-score array
    """
    sub_mean = np.nanmean(array)
    sub_std = np.nanstd(array)
    z_array = (array - sub_mean) / sub_std

    return z_array


def smooth(array_to_smooth, w_size, sliding_mode="ontop"):
    len_array = len(array_to_smooth)
    smoothed_array = np.zeros((len_array,))  # init. output array

    if sliding_mode == "hanging":  # causes shift of peaks, but more realistic
        # attach NaN in the beginning for correct sliding window calculions
        edge_nan = np.repeat(np.nan, w_size - 1)
        array_to_smooth = np.concatenate((edge_nan, array_to_smooth), axis=0)

        for i in range(len_array):
            smoothed_array[i] = np.nanmean(array_to_smooth[i: i + w_size])

    if sliding_mode == "ontop":
        # self.w_size  need to be odd number 1, 3, 5, ...
        if w_size % 2 == 0:
            w_size -= 1
            print("Smoothing window size need to be odd, adjusted(i.e.: -1) to:", w_size)

        # attach NaN in the beginning and end for correct sliding window calculions
        edge_nan_start = np.repeat(np.nan, int(w_size / 2))
        edge_nan_end = edge_nan_start
        array_to_smooth = np.concatenate((edge_nan_start, array_to_smooth), axis=0)
        array_to_smooth = np.concatenate((array_to_smooth, edge_nan_end), axis=0)

        for i in range(len_array):
            smoothed_array[i] = np.nanmean(array_to_smooth[i: i + w_size])

    return smoothed_array


def create_s_fold_idx(s_folds, list_prev_indices=[]):

    if not list_prev_indices:  # list_prev_indices == []
        s_fold_idx = np.random.randint(0, s_folds)
    else:
        choose_from = list(set(range(s_folds)) - set(list_prev_indices))
        s_fold_idx = np.random.choice(choose_from, 1)[0]

    list_prev_indices.append(s_fold_idx)

    return s_fold_idx, list_prev_indices

# s_idx, list_indices = create_s_fold_idx(s_folds=10)
# for _ in range(9):
#     s_idx, list_indices = create_s_fold_idx(s_folds=10, list_prev_indices=list_indices)
# len(list_indices)

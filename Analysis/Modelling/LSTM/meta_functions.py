# coding=utf-8
"""
Some meta functions

Author: Simon Hofmann | <[surname].[lastname][at]pm.me> | 2017, 2019 (Update)
"""

import datetime
from functools import wraps
import numpy as np
from scipy.signal import hilbert
import subprocess
import platform
import os
import psutil


def setwd(new_dir):

    # Remove '/' if new_dir == 'folder/' OR '/folder'
    new_dir = new_dir[:-1] if new_dir[-1] == "/" else new_dir
    new_dir = new_dir[1:] if new_dir[0] == "/" else new_dir

    cprint("Current working dir:\t{}".format(os.getcwd()), "b")

    found = False if new_dir != os.getcwd().split("/")[-1] else True

    # First look down the tree
    if not found:
        for path, j, files in os.walk('.'):
            # print(path, j, files)
            if new_dir in path:
                os.chdir(path)
                found = True
                break

        # Else look up the tree
        if not found:
            if new_dir in os.getcwd():

                path = os.getcwd().split("/")

                while new_dir != path[-1]:
                    path.pop()

                path = "/".join(path)
                os.chdir(path)
                found = True

        if found:
            cprint("New working dir:\t\t{}\n".format(os.getcwd()), "y")
        else:
            cprint("Given folder not found. Working dir remains:\t{}\n".format(os.getcwd()), "r")

    # else:
    #     print("Already in correct working dir")


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

        # duration = chop_microseconds(delta=duration)  # do not show microseconds(ms)

        print("Processing time of {}: {} [h:m:s:ms]".format(funct.__name__, duration))

        return output

    return wrapper

# @function_timed
# def foo():
#     print("Any Function")
#
# foo()


def chop_microseconds(delta):
    return delta - datetime.timedelta(microseconds=delta.microseconds)


def average_time(list_of_timestamps, in_timedelta=True):
    """
    Method to average time of a list of time-stamps. Necessary for Python 2.
    In Python3 simply: np.mean([datetime.timedelta(0, 20), ... , datetime.timedelta(0, 32)])

    :param list_of_timestamps: list of time-stamps
    :param in_timedelta: whether to return in datetime.timedelta-format.
    :return: average time
    """
    mean_time = sum(list_of_timestamps, datetime.timedelta()).total_seconds() / len(list_of_timestamps)

    if in_timedelta:
        mean_time = datetime.timedelta(seconds=mean_time)

    return mean_time


def s(subject):
    """
    :param subject: take subject ID in form of integer
    :return: string form, e.g. S36 or S02
    """
    return 'S' + str(subject).zfill(2)


def normalization(array, lower_bound, upper_bound):
    """
    Min-Max-Scaling: Normalizes Input Array to lower and upper bound
    :param upper_bound: Upper Bound b
    :param lower_bound: Lower Bound a
    :param array: To be transformed array
    :return: normalized array
    """

    assert lower_bound < upper_bound, "lower_bound must be < upper_bound"

    a, b = lower_bound, upper_bound

    normed_array = (b-a) * ((array - np.nanmin(array)) / (np.nanmax(array) - np.nanmin(array))) + a

    return normed_array


def denormalize(array, denorm_minmax, norm_minmax):
    """
    :param array: array to be de-normalized
    :param denorm_minmax: tuple of (min, max) of de-normalized (target) vector
    :param norm_minmax: tuple of (min, max) of normalized vector
    :return: de-normalized value
    """
    array = np.array(array)

    dnmin, dnmax = denorm_minmax
    nmin, nmax = norm_minmax

    assert nmin < nmax, "norm_minmax must be tuple (min, max), where min < max"
    assert dnmin < dnmax, "denorm_minmax must be tuple (min, max), where min < max"

    denormed_array = (array - nmin)/(nmax - nmin) * (dnmax-dnmin) + dnmin

    return denormed_array


def z_score(array, inf=False):
    """
    Create z-score
    :return: z-score array
    """
    if inf:
        array[np.where(np.abs(array) == np.inf)] = np.nan
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


def downsampling(array_to_ds, target_hertz=1, given_hertz=250):
    ds_ratio = given_hertz/target_hertz
    assert float(ds_ratio).is_integer(), \
        "Ratio between given frequency and target frequency must be an integer."

    output_shape = None
    if float(len(array_to_ds)/ds_ratio).is_integer():
        output_shape = int(len(array_to_ds) / ds_ratio)
    else:
        ValueError("Input-array must be pruned. Cannot be split in equally sized pieces.")

    output_array = np.zeros(shape=output_shape)

    idx = int(ds_ratio)
    for i in range(output_shape):
        output_array[i] = np.nanmean(array_to_ds[idx-int(ds_ratio):idx])
        idx += int(ds_ratio)

    return output_array


def calc_hilbert_z_power(array):
    """
    square(abs(complex number)) = power = squarred length of complex number, see: Cohen (2014, p.160-2)
    Do on narrow-band data (alpha-filtered)
    """
    analytical_signal = hilbert(array)
    amplitude_envelope = np.abs(analytical_signal)
    power = np.square(amplitude_envelope)
    # z-score of power contains power information and its variance, while centred around zero
    hilbert_z_power = z_score(array=power)  # z-score
    # could be smoothed to small degree, e.g., smooth(hilbert_z_power, 10)...
    return hilbert_z_power


def create_s_fold_idx(s_folds, list_prev_indices=None):

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


def calc_binary_class_accuracy(prediction_matrix):
    """
    Calculate accuracy of binary classification from given prediction_matrix.
    Postive values are considered as prediction of high arousal, and negative values as of low arousal.
    :param prediction_matrix: Contains predictions and ground truth
    :return: list of accuracy of each fold
    """
    n_folds = int(prediction_matrix.shape[0]/2)
    val_class_acc_list = np.zeros(shape=n_folds)
    for fo in range(n_folds):
        # 1: correct, -1: incorrect
        cor_incor = np.sign(prediction_matrix[fo*2, :]*prediction_matrix[fo*2+1, :])
        # delete nan's
        cor_incor = np.delete(arr=cor_incor, obj=np.where(np.isnan(cor_incor)))
        if len(cor_incor) > 0:
            fo_accur = sum(cor_incor == 1)/len(cor_incor)
        else:
            fo_accur = np.nan
        val_class_acc_list[fo] = fo_accur
    return val_class_acc_list


def sort_mat_by_mat(mat, mat_idx):
    """
    mat         mat_idx         sorted mat
    [[1,2,3],   [[1,2,0],  ==>  [[2,3,1],
     [4,5,6]]    [2,0,1]]  ==>   [6,4,5]]

    :param mat: matrix to be sorted by rows of mat_idx
    :param mat_idx: matrix with corresponding indices
    :return: sorted matrix
    """
    mat_idx = mat_idx.astype(int)
    assert mat.shape == mat_idx.shape, "Matrices must be the same shape"
    n_rows = mat.shape[0]

    sorted_mat = np.zeros(shape=mat.shape)

    for row in range(n_rows):

        sorted_mat[row, :] = inverse_indexing(arr=mat[row, :], idx=mat_idx[row, :])
        # sorted_mat[row, :] = mat[row, :][mat_idx[row, :]]

    return sorted_mat


def inverse_indexing(arr, idx):
    """
    Inverse indexing of array (e.g., [16.,2.,4.]) to its origin (e.g., [2.,4.,16.])
    given the former index-vector (here: [2,0,1]).
    :param arr: altered array
    :param idx: former indexing vector
    :return: recovered array
    """
    inv_arr = np.repeat(np.nan, len(arr))
    for i, ix in enumerate(idx):
        inv_arr[ix] = arr[i]
    return inv_arr


def interpolate_nan(arr_with_nan, verbose=False):
    """
    Return array with linearly interpolated values.
    :param arr_with_nan: array with missing values
    :param verbose: True: print number of interpolated values
    :return: updated array
    """
    missing_idx = np.where(np.isnan(arr_with_nan))[0]

    if len(missing_idx) == 0:
        print("There are no nan-values in the given vector.")

    else:
        for midx in missing_idx:
            # if the first value is missing take average of the next 5sec
            if midx == 0:
                arr_with_nan[midx] = np.nanmean(arr_with_nan[midx+1: midx+1+5])
            # Else Take the mean of the two adjacent values
            else:  # midx > 0
                if np.isnan(arr_with_nan[midx]):  # Check if still missing (see linspace filling below)
                    if not np.isnan(arr_with_nan[midx+1]):  # we coming from below
                        arr_with_nan[midx] = np.mean([arr_with_nan[midx-1], arr_with_nan[midx+1]])
                    else:  # next value is also missing
                        count = 0
                        while True:
                            if np.isnan(arr_with_nan[midx+1+count]):
                                count += 1
                            else:
                                break

                        fillvec = np.linspace(start=arr_with_nan[midx-1], stop=arr_with_nan[midx+1+count],
                                              num=3 + count)[1:-1]

                        assert len(fillvec) == 1 + count, "Implementation error at interpolation"

                        arr_with_nan[midx: midx+count+1] = fillvec
        if verbose:
            print("Interpolated {} values.".format(len(missing_idx)))

    updated_array = arr_with_nan

    return updated_array


def cln(factor=1):
    """Clean the console"""
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n" * factor)


def true_false_request(func):

    @wraps(func)
    def wrapper(*args, **kwds):
        func(*args, **kwds)  # should be only a print command

        tof = input("(T)rue or (F)alse: ").lower()
        assert tof in ["true", "false", "t", "f"], "Must be 'T', 't' or 'T/true', or 'F', 'f', 'F/false'"
        output = True if tof in "true" else False

        return output

    return wrapper


def open_folder(path):
    """Open specific folder in Finder. Can also be a file"""
    if platform.system() == "Windows":  # for Windows
        os.startfile(path)
    elif platform.system() == 'Darwin':  # ≈ sys.platform = 'darwin' | for Mac
        subprocess.Popen(["open", path])
    else:  # for Linux
        subprocess.Popen(["xdg-open", path])


def check_computer():
    print("Current computer is: \t{}{}{}".format(Bcolors.OKBLUE, platform.node(), Bcolors.ENDC))
    return platform.node()


def check_executor(return_bash_bool=False):
    # Check whether scipt is run via bash
    ppid = os.getppid()
    bash = psutil.Process(ppid).name() == "bash"
    print("Current script{} executed via: {}{}{}".format(
        ' {}{}{}'.format(Bcolors.WARNING, platform.sys.argv[0], Bcolors.ENDC) if bash else "",
        Bcolors.OKBLUE, psutil.Process(ppid).name(), Bcolors.ENDC))

    if return_bash_bool:
        return bash


def check_mpi_gpu(name_it=False):
    """
    This is a MPI CBS specific function, to check whether the script is run on GPU server at the
    institute
    :return True if current computer is MPI CBS' GPU server else False
    """

    path_data = "../../../Data/"
    cinfo = path_data + 'compute_server'  # contains information about gpu compute server

    if os.path.isfile(cinfo):
        cinfo = np.genfromtxt(path_data + 'compute_server', str)
        gpu_server_name = cinfo[cinfo[:, 0] == "server:", 1][0]

        if name_it:
            print("GPU server name is: \t{}{}{}".format(Bcolors.OKBLUE, gpu_server_name, Bcolors.ENDC))

        if check_computer() == gpu_server_name:
            return True
        else:
            return False

    else:
        cprint("No information about MPI CBS' GPU server found.", "r")
        return False


def path2_mpi_gpu_hd(disk):
    assert disk in [1, 2], "disk must be 1 or 2"

    path_data = "../../../Data/"

    # cinfo contains information about gpu compute server
    cinfo = np.genfromtxt(path_data + 'compute_server', str)
    gpu_server_hd = cinfo[cinfo[:, 0] == "disk{}:".format(disk), 1][0]  # 1:=/...2/ and 2:=/...3/

    if gpu_server_hd not in os.getcwd():
        path_2_mpi_gpu_hd = "../../../../../../../../{}/ResearchProjects/NeVRo/Data/".format(
            gpu_server_hd)
    else:
        path_2_mpi_gpu_hd = path_data

    return path_2_mpi_gpu_hd


def set_path2data():

    path_data = "../../../Data/"
    # <<<<<< MPI-specific
    if check_mpi_gpu():
        path_data = path2_mpi_gpu_hd(disk=1)  # disk=1 contains the NeVRo data
    # MPI-specific >>>>>>>>

    cprint("Data dir: \t\t{}".format(path_data), "y")

    return path_data


class Bcolors:
    """
    Colours print-commands in Console
    Usage:
    print(Bcolors.HEADER + "Warning: No active frommets remain. Continue?" + Bcolors.ENDC)
    print(Bcolors.OKBLUE + "Warning: No active frommets remain. Continue?" + Bcolors.ENDC)

    For more:

        CSELECTED = '\33[7m'

        CBLACK  = '\33[30m'
        CRED    = '\33[31m'
        CGREEN  = '\33[32m'
        CYELLOW = '\33[33m'
        CBLUE   = '\33[34m'
        CVIOLET = '\33[35m'
        CBEIGE  = '\33[36m'
        CWHITE  = '\33[37m'

        CBLACKBG  = '\33[40m'
        CREDBG    = '\33[41m'
        CGREENBG  = '\33[42m'
        CYELLOWBG = '\33[43m'
        CBLUEBG   = '\33[44m'
        CVIOLETBG = '\33[45m'
        CBEIGEBG  = '\33[46m'
        CWHITEBG  = '\33[47m'

        CGREY    = '\33[90m'
        CBEIGE2  = '\33[96m'
        CWHITE2  = '\33[97m'

        CGREYBG    = '\33[100m'
        CREDBG2    = '\33[101m'
        CGREENBG2  = '\33[102m'

        CYELLOWBG2 = '\33[103m'
        CBLUEBG2   = '\33[104m'
        CVIOLETBG2 = '\33[105m'
        CBEIGEBG2  = '\33[106m'
        CWHITEBG2  = '\33[107m'

    # For preview type:
    for i in [1, 4, 7] + list(range(30, 38)) + list(range(40, 48)) + list(range(90, 98)) + list(
            range(100, 108)):  # range(107+1)
        print(i, '\33[{}m'.format(i) + "ABC & abc" + '\33[0m')
    """

    HEADERPINK = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    UNDERLINE = '\033[4m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'  # this is necessary in the end to reset to default print

    DICT = {'p': HEADERPINK, 'b': OKBLUE, 'g': OKGREEN, 'y': WARNING, 'r': FAIL,
            'ul': UNDERLINE, 'bo': BOLD}

# print(Bcolors.HEADER + "Header: No active frommets remain. Continue?" + Bcolors.ENDC)
# print(Bcolors.OKBLUE + "Ok Blue: No active frommets remain. Continue?" + Bcolors.ENDC)
# print(Bcolors.OKGREEN + "Ok Green: No active frommets remain. Continue?" + Bcolors.ENDC)
# print(Bcolors.WARNING + "Warning: No active frommets remain. Continue?" + Bcolors.ENDC)
# print(Bcolors.FAIL + "Fail: No active frommets remain. Continue?" + Bcolors.ENDC)
# print(Bcolors.ENDC + "Endc: No active frommets remain. Continue?" + Bcolors.ENDC)
# print(Bcolors.BOLD + "Bold: No active frommets remain. Continue?" + Bcolors.ENDC)
# print(Bcolors.UNDERLINE + "Underline: No active frommets remain. Continue?" + Bcolors.ENDC)
# print(Bcolors.UNDERLINE + Bcolors.BOLD + Bcolors.WARNING +
#       "Underline: No active frommets remain. Continue?" + Bcolors.ENDC)


def cprint(string, col=None, fm=None):
    """Format given string"""
    if col:
        col = col[0].lower()
        assert col in ['p', 'b', 'g', 'y', 'r'], \
            "col must be 'p'(ink), 'b'(lue), 'g'(reen), 'y'(ellow), 'r'(ed)"
        col = Bcolors.DICT[col]

    if fm:
        fm = fm[0:2].lower()
        assert fm in ['ul', 'bo'], "fm must be 'ul'(:underline), 'bo'(:bold)"
        fm = Bcolors.DICT[fm]

    # print given string with formatting
    print("{}{}".format(col if col else "",
                        fm if fm else "") + string + Bcolors.ENDC)


def cinput(string, col=None):

    if col:
        col = col[0].lower()
        assert col in ['p', 'b', 'g', 'y', 'r'], \
            "col must be 'p'(ink), 'b'(lue), 'g'(reen), 'y'(ellow), 'r'(ed)"
        col = Bcolors.DICT[col]

    # input(given string) with formatting
    return input("{}".format(col if col else "") + string + Bcolors.ENDC)


@true_false_request
def ask_true_false(question):
    """
    Ask user for input for given True-or-False question
    :param question: str
    :return: answer
    """
    cprint(question, "b")

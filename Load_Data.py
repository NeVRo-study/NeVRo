# coding=utf-8
"""
Load Data for LSTM Model. Only use Space-Break-Andes (SBA) Versions, which are pruned to 148, 30 & 92 sec, respectively.
    • Load Data
    • Create Batches

......................................................................................................................
Model 1) Feed to best SSD-extracted alpha components (250Hz), i.e. 2 channels, into LSTM to predict ratings (1Hz)


Comp1: [x1_0, ..., x1_250], [x1_251, ..., x1_500], ... ]  ==> LSTM \
                                                                    ==>> Rating [y0, y1, ... ]
Comp2: [x2_0, ..., x2_250], [x2_251, ..., x2_500], ... ]  ==> LSTM /

......................................................................................................................
MODEL 2: Feed slightly pre-processed channels data, i.e. 30 channels, into LSTM to predict ratings

Channel01: [x01_0, ..., x01_250], [x01_251, ..., x01_500], ... ]  ==> LSTM \
Channel02: [x02_0, ..., x02_250], [x02_251, ..., x02_500], ... ]  ==> LSTM  \
...                                                                          ==>> Rating [y0, y1, ... ]
...                                                                         /
Channel30: [x30_0, ..., x30_250], [x30_251, ..., x30_500], ... ]  ==> LSTM /
......................................................................................................................

Author: Simon Hofmann | <[surname].[lastname][at]protonmail.com> | 2017
"""

import os.path
import copy
from scipy.signal import hilbert
from Meta_Functions import *
import pandas as pd

# from Meta_Functions import *
# import tensorflow as tf
# from tensorflow.contrib.learn.python.learn.datasets import base
# import mne

#  load EEG data of from *.set (EEGlab format) with: mne.io.read_raw_eeglab() and mne.read_epochs_eeglab()

# Change to Folder which contains files
wdic = "../../Data/EEG_SSD/"
wdic_Comp = "../../Data/EEG_SSD/Components/"
wdic_SBA_Comp = "../../Data/EEG_SSD/SBA/Components/"
# wdic_Rating = "../../Data/ratings/preprocessed/z_scored_alltog/"  # SBA-pruned data (148:space, 30:break, 92:ande)
wdic_Rating = "../../Data/ratings/preprocessed/not_z_scored/"  # transform later to [-1, +1]

wdic_Data = "../../Data/"
wdic_cropECG = "../../Data/Data EEG export/NeVRo_ECG_Cropped/"
wdic_x_corr = "../../Results/x_corr/"

# initialize variables
subjects = [36]  # [36, 37]
# subjects = range(1, 45+1)

# roller_coasters = np.array(['Space_NoMov', 'Space_Mov', 'Ande_Mov', 'Ande_NoMov'])
roller_coasters = np.array(['Space_NoMov', "Break_NoMov", 'Ande_NoMov']) if True \
    else np.array(['Space_NoMov', 'Ande_NoMov'])


def update_coaster_lengths(empty_t_array, sba=True):
    """
    Creates an array of the lengths of the different phases (e.g., roller caosters)
    :param empty_t_array: array to be filled
    :param sba: asks whether pruned SBA-files should be used to calculate times.
    :return: array with lengths of phases
    """
    sfreq = 500.
    if not sba:
        for sub in subjects:
            for n, coast in enumerate(roller_coasters):
                time_ecg_fname = wdic_cropECG + "NVR_S{}_{}.txt".format(str(sub).zfill(2), coast)
                if os.path.isfile(time_ecg_fname):
                    ecg_file = np.loadtxt(time_ecg_fname)
                    len_time_ecg = len(ecg_file) / sfreq
                    empty_t_array[n] = len_time_ecg if len_time_ecg > empty_t_array[n] else empty_t_array[n]
    else:
        empty_t_array = np.array([148., 30., 92])

    full_t_array = empty_t_array  # make-up
    # print("Length of each roller coaster:\n", full_t_array)
    return full_t_array

# t_roller_coasters = np.zeros((len(roller_coasters)))  # init
# t_roller_coasters = update_coaster_lengths(empty_t_array=t_roller_coasters, sba=True)
# t_roller_coasters = np.array([148, 30, 92])   # SBA


# Load file
def load_ssd_files(samp_freq=250., sba=True):
    """
    Loads all channel SSD files of each subject in subjects
    :return: SSD files [channels, time_steps, sub_df] in form of dictionary
    """
    count = 0
    # Create SSD-dictionary
    # dic = {k: v for k, v in [("bla", [])]}
    # dic.update({newkey: newkey_value})
    ssd_dic_keys = ["channels"] + [str(i) for i in subjects]
    ssd_dic = dict.fromkeys(ssd_dic_keys, {})  # creats dict from given sequence and given value (could be empty =None)
    ssd_dic_keys.pop(0)  # remove channels-key again
    # create sub_dic for each roller coaster (time, df)
    coast_dic = {}
    coast_dic.update((key, {"t_steps": [], "df": []}) for key in roller_coasters)
    # each subejct has a sub-dic for each roller coaster
    ssd_dic.update((key, copy.deepcopy(coast_dic)) for key in ssd_dic_keys)  # deep.copy(!)
    # # Test
    # print("ssd_dic\n", ssd_dic)
    # ssd_dic["36"]['Ande_NoMov']["df"] = 999
    # print("ssd_dic\n", ssd_dic)  # ssd_dic["36"] is ssd_dic["37"]

    for subject in subjects:
        for num, coaster in enumerate(roller_coasters):
            count += 1
            file_name = wdic + "S{}_{}_SSD.txt".format(str(subject).zfill(2), coaster)  # adapt file_name

            if os.path.isfile(file_name):
                if count == 1:  # do only once
                    # x = x.split("\t")  # '\t' divides values
                    channels = np.genfromtxt(file_name, delimiter="\t", dtype="str")[1:, 0]  # 32 channels; shape=(32,)
                    ssd_dic["channels"] = channels
                sub_df = np.genfromtxt(file_name, delimiter="\t")[0:, 1:-1]  # first row:= time; in last col only NAs
                # Sampling Frequency = 250
                time_steps = sub_df[0, :]  # shape=(24276,) for S36_Ande_NoMov
                sub_df = sub_df[1:, :]  # shape=(32, 24276) for S36_Ande_NoMov
                # Fill in SSD Dictionary: ssd_dic
                ssd_dic[str(subject)][coaster]["t_steps"] = copy.copy(time_steps)
                ssd_dic[str(subject)][coaster]["df"] = copy.copy(sub_df)

                # Check times:
                # samp_freq = 250.

                t_roller_coasters = update_coaster_lengths(empty_t_array=np.zeros((len(roller_coasters))), sba=sba)
                if not (np.round(t_roller_coasters[num], 1) == np.round(time_steps[-1] / 1000., 1)
                        and np.round(time_steps[-1] / 1000., 1) == np.round(len(time_steps) / samp_freq, 1)):
                    print("Should be all approx. the same:\nt_roller_coasters[num]: {} \ntime_steps[-1]/1000: {}"
                          "\nlen(time_steps)/sfreq(250): {}".format(t_roller_coasters[num],
                                                                    time_steps[-1] / 1000.,
                                                                    len(time_steps) / samp_freq))

    return ssd_dic

# SSD_dic = load_ssd_files(samp_freq=250.)
# print(SSD_dic[str(subjects[0])][roller_coasters[0]]["df"].shape)  # roller_coasters[0] == 'Space_NoMov'
# print(SSD_dic[str(subjects[0])][roller_coasters[0]]["df"].shape)  # str(subjects[0]) == '36'
# print(SSD_dic[str(subjects[0])][roller_coasters[0]]["t_steps"].shape)
# print(SSD_dic["channels"].shape)


def load_ssd_component(samp_freq=250., sba=True):
    """
    :param samp_freq: sampling frequency of SSD-components
    :param sba: if True (=Default), take SBA-z-scored components
    Loads SSD components (files) of each subject in subjects
    :return: SSD component files [sub_df] in form of dictionary
    """

    # Create SSD Component Dictionary
    ssd_comp_dic_keys = [str(i) for i in subjects]  # == list(map(str, subjects))  # these keys also work int
    ssd_comp_dic = {}
    ssd_comp_dic.update((key, dict.fromkeys(roller_coasters, [])) for key in ssd_comp_dic_keys)

    condition_keys = [element.split("_")[1] for element in roller_coasters]  # "NoMov", "Mov"
    condition_keys = list(set(condition_keys))

    t_roller_coasters = update_coaster_lengths(empty_t_array=np.zeros((len(roller_coasters))), sba=sba)

    if sba:
        for key in ssd_comp_dic_keys:
            ssd_comp_dic[key].update({"SBA": dict.fromkeys(condition_keys, [])})

    for subject in subjects:
        for num, coaster in enumerate(roller_coasters):
            folder = coaster.split("_")[1]  # either "Mov" or "NoMov" (only needed for sba case)
            if sba:
                file_name = wdic_SBA_Comp + folder + "/NVR_S{}_SBA_{}_SSD_Components_SBA_CNT.txt".format(
                    str(subject).zfill(2), folder)
                # "NVR_S36_SBA_NoMov_SSD_Components_SBA_CNT.txt"
            else:
                file_name = wdic_Comp + "S{}_{}_Components.txt".format(str(subject).zfill(2), coaster)

            if os.path.isfile(file_name):
                # x = x.split("\t")  # '\t' divides values
                # N ≈ 20 components sorted from 1-N according to the signal-to-noise-ration (SNR), i.e. "good to bad"
                # components = np.genfromtxt(file_name, delimiter=";", dtype="str")[0].split("\t")[:-1]  # last=' '
                # for idx, comp in enumerate(components):
                #     components[idx] = "comp_" + comp
                # ssd_comp_dic["components"] = components  # 32 components; shape=(32,)
                # n_comp = len(components)

                n_comp = len(np.genfromtxt(file_name, delimiter=";", dtype="str")[0].split("\t")[:-1])  # last=' '

                # columns = components, rows value per timestep
                pre_sub_df = np.genfromtxt(file_name, delimiter=";", dtype="str")[1:]  # first row = comp.names
                sub_df = np.zeros((pre_sub_df.shape[0], n_comp))
                for row in range(pre_sub_df.shape[0]):
                    values_at_t = pre_sub_df[row].split("\t")
                    sub_df[row] = list(map(float, values_at_t))  # == [float(i) for i in values_at_t]
                # del pre_sub_df  # save WM

                # sub_df.shape=(38267, n_comp=22) for S36_Space_NoMov | Samp. Freq. = 250 | 38267/250 = 153.068sec

                # Fill in SSD Dictionary: ssd_comp_dic
                if not sba:
                    ssd_comp_dic[str(subject)][coaster] = copy.copy(sub_df)

                else:  # if sba
                    # Split SBA file in Space-Break-Andes
                    if "Space" in coaster:
                        sba_idx_start = 0
                        sba_idx_end = int(t_roller_coasters[0] * samp_freq)
                    elif "Break" in coaster:
                        sba_idx_start = int(t_roller_coasters[0] * samp_freq)
                        sba_idx_end = int(sum(t_roller_coasters[0:2]) * samp_freq)
                    else:  # "Ande" in coaster
                        sba_idx_start = int(sum(t_roller_coasters[0:2]) * samp_freq)
                        sba_idx_end = int(sum(t_roller_coasters) * samp_freq)
                        # Test lengths
                        if sba_idx_end < sub_df.shape[0]:
                            print("sub_df of S{} has not expected size: Diff={} data points.".format(
                                str(subject).zfill(2), sub_df.shape[0] - sba_idx_end))
                            assert (sub_df.shape[0] - sba_idx_end) <= 3, "Difference too big, check files"

                    ssd_comp_dic[str(subject)][coaster] = copy.copy(sub_df[sba_idx_start:sba_idx_end, :])

                    # process whole SBA only once per condition (NoMov, Mov)
                    if "Break" in coaster:
                        assert folder in condition_keys, "Wrong key"
                        ssd_comp_dic[str(subject)]["SBA"][folder] = copy.copy(sub_df)

                # Check times:
                # samp_freq = 250.
                if not sba:
                    if not (np.round(t_roller_coasters[num], 1) == np.round(sub_df.shape[0] / samp_freq, 1)):
                        print("Should be all approx. the same:\nTime of {}: {}"
                              "\nLength of sub_df/samp_freq({}): {}".format(roller_coasters[num],
                                                                            t_roller_coasters[num],
                                                                            int(samp_freq),
                                                                            sub_df.shape[0] / samp_freq))
                else:  # if sba:
                    if not (np.round(sum(t_roller_coasters), 1) == np.round(sub_df.shape[0] / samp_freq, 1)):
                        print("Should be all approx. the same:\nTime of SBA: {}"
                              "\nLength of sub_df/samp_freq({}): {}".format(sum(t_roller_coasters),
                                                                            int(samp_freq),
                                                                            sub_df.shape[0] / samp_freq))

    return ssd_comp_dic

# SSD_Comp_dic = load_ssd_component(samp_freq=250., sba=True)
# for num, coaster in enumerate(roller_coasters):
#     print(SSD_Comp_dic[str(subjects[0])][roller_coasters[num]].shape,
#           "=",
#           SSD_Comp_dic[str(subjects[0])][roller_coasters[num]].shape[0]/250.,
#           "sec\t|",
#           SSD_Comp_dic[str(subjects[0])][roller_coasters[num]].shape[1],
#           "components\t|",
#           roller_coasters[num])


def best_component(subject, best=True):
    """
    Choose the best SSD-alpha-component, w.r.t. x-corr with ratings, from list for given subject.
    :param subject: subject number
    :param best: if false, take the worst
    :return: best component number
    """

    # x_corr_table = np.genfromtxt(wdic_x_corr + "CC_AllSubj_Alpha_Ratings_smooth.csv",
    #                              delimiter=",", skip_header=1, dtype=float)

    # Load Table
    x_corr_table = pd.read_csv(wdic_x_corr + "CC_AllSubj_Alpha_Ratings_smooth.csv", index_col=0)  # first col is idx
    # drop non-used columns
    x_corr_table = x_corr_table.drop(x_corr_table.columns[0:3], axis=1)

    if best:
        # x_corr_table.columns = ["components"]  # renamce columns
        component = x_corr_table.loc["S{}".format(str(subject).zfill(2))].values[0]

    else:
        component = best_comp = x_corr_table.loc["S{}".format(str(subject).zfill(2))].values[0]

        # choose another component that the best
        while component == best_comp:
            component = np.random.randint(low=1, high=5+1)

    # TODO choose the worst
    print("The best correlating SSD component of Subject {} is Component Number {}".format(subject, component))

    return component


# @function_timed  # after executing following function this returns runtime
def load_rating_files(samp_freq=1., sba=True):
    """
    Loads (z-scored) Ratings files of each subject in subjects (ignore other files, due to fluctuating samp.freq.).
    :param sba: if TRUE (default), process SBA files
    :param samp_freq: sampling frequency, either 1Hz [default], oder 50Hz
    :return:  Rating Dict
    """

    # Check whether input correct
    # assert (samp_freq == 1 or samp_freq == 50), "samp_freq must be either 1 or 50 Hz"
    if not (samp_freq == 1. or samp_freq == 50.):
        raise ValueError("samp_freq must be either 1 or 50 Hz")
    # Load Table of Conditions
    table_of_condition = np.genfromtxt(wdic_Data + "Table_of_Conditions.csv", delimiter=";")
    table_of_condition = table_of_condition[1:, ]  # remove first column (sub-nr, condition, gender (1=f, 2=m))
    # Create Rating-dictionary
    rating_dic_keys = list(map(str, subjects))  # == [str(i) for i in subjects]
    rating_dic = {}
    # each subejct has a sub-dic for each roller coaster
    rating_dic.update((key, dict.fromkeys(roller_coasters, [])) for key in rating_dic_keys)

    condition_keys = [element.split("_")[1] for element in roller_coasters]  # "NoMov", "Mov"
    condition_keys = list(set(condition_keys))

    t_roller_coasters = update_coaster_lengths(empty_t_array=np.zeros((len(roller_coasters))), sba=sba)

    # For each subject fill condtition in
    for key in rating_dic_keys:
        key_cond = int(str(table_of_condition[np.where(table_of_condition[:, 0] == int(key)), 1])[3])
        rating_dic[key].update({"condition": key_cond})
        if sba:
            rating_dic[key].update({"SBA": dict.fromkeys(condition_keys, [])})

    for subject in subjects:
        for num, coaster in enumerate(roller_coasters):
            # adapt file_name accordingly (run <=> condition,
            if "Space" in coaster:
                coast = "Space".lower()
            elif "Ande" in coaster:
                coast = "andes"
            else:  # Break
                coast = "break"

            runs = "1" if "NoMov" in coaster and rating_dic[str(subject)]["condition"] == 2 else "2"
            coast_folder = "nomove" if "NoMov" in coaster else "move"

            rating_filename = wdic_Rating + "{}/{}/{}Hz/NVR_S{}_run_{}_{}_rat_z.txt".format(coast,
                                                                                            coast_folder,
                                                                                            str(int(samp_freq)),
                                                                                            str(subject).zfill(2),
                                                                                            runs,
                                                                                            coast)

            if os.path.isfile(rating_filename):
                # Load according file
                rating_file = np.genfromtxt(rating_filename, delimiter=',')[:, 1]  # only load column with ratings
                # Fill in rating dictionary
                rating_dic[str(subject)][coaster] = copy.copy(rating_file)

                # Check times:
                if not (np.round(t_roller_coasters[num], 1) == np.round(len(rating_file) / samp_freq, 1)):
                    print("Should be all approx. the same:\nTime of {}: {}\nLength of Rating / s_freq({}): {}".format(
                        roller_coasters[num], t_roller_coasters[num], samp_freq, len(rating_file) / samp_freq))

            # just process SBA once per condition (NoMov, Mov):
            if "Break" in coaster:
                cond_key = coaster.split("_")[1]  # either "Mov" or "NoMov" (only needed for sba case)
                assert cond_key in condition_keys, "Wrong Key"

                sba_rating_filename = wdic_Rating + "alltog/{}/{}Hz/NVR_S{}_run_{}_alltog_rat_z.txt".format(
                    coast_folder,
                    str(int(samp_freq)),
                    str(subject).zfill(2),
                    runs)

                if os.path.isfile(sba_rating_filename):
                    # Load according file
                    sba_rating_file = np.genfromtxt(sba_rating_filename, delimiter=',')[:, 1]
                    # Fill in rating dictionary
                    rating_dic[str(subject)]["SBA"][cond_key] = copy.copy(sba_rating_file)

                    # Check times:
                    if not (np.round(sum(t_roller_coasters), 1) == np.round(len(sba_rating_file) / samp_freq, 1)):
                        print("Should be all approx. the same:\nTime of SBA: {}"
                              "\nLength of Rating/s_freq({}Hz): {}".format(sum(t_roller_coasters),
                                                                           int(samp_freq),
                                                                           len(sba_rating_file) / samp_freq))

    return rating_dic


class DataSet(object):
    """
    Utility class (http://wiki.c2.com/?UtilityClasses) to handle dataset structure
    """

    # s_fold_idx_list = []  # this needs to be changed across DataSet-instances
    # s_fold_idx = []  # this needs to be changed across DataSet-instances

    def __init__(self, name, eeg, ratings, subject, condition, eeg_samp_freq=250., rating_samp_freq=1.):
        """
        Builds dataset with EEG data and Ratings
        :param eeg: eeg data, SBA format (space-break-ande) (so far only NoMov)
        :param ratings: rating data, SBA format (space-break-ande)
        :param subject: Subject Nummer
        :param condition: Subject condition
        :param eeg_samp_freq: sampling frequency of EEG data (default = 250Hz)
        :param rating_samp_freq: sampling frequency of Rating data (default = 1Hz)
        """

        assert eeg.shape[0] == ratings.shape[0], "eeg.shape: {}, ratings.shape: {}".format(eeg.shape, ratings.shape)

        self.name = name
        self.eeg_samp_freq = eeg_samp_freq
        self.rating_samp_freq = rating_samp_freq
        self._num_time_slices = eeg.shape[0]
        self.remaining_slices = np.arange(self.num_time_slices)  # for randomized drawing of new_batch
        self._eeg = eeg  # input
        self._ratings = ratings  # target
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self.current_batch = []
        self.subject = subject
        self.condition = condition

    # @classmethod
    # def update_s_fold_idx(cls):
    #     """This updates the class variable s_fold_idx_list and s_fold_idx"""
    #     pass

    @property
    def eeg(self):
        return self._eeg

    @property
    def ratings(self):
        return self._ratings

    @property
    def num_time_slices(self):
        return self._num_time_slices

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def reset_remaining_slices(self):
        return np.arange(self.num_time_slices)

    def new_epoch(self):
        self._epochs_completed += 1
        print("\nStarting new epoch ({} completed) in {} dateset\n".format(self.epochs_completed, self.name))

        # Testing
        # print("Index in previous epoch:", self._index_in_epoch)
        # print("Still available slices in epoch: {}/{}\n".format(len(self.remaining_slices),
        #                                                         self._num_time_slices))

    def next_batch(self, batch_size=1, randomize=False):
        """
        Return the next 'batch_size' examples from this data set
        For MODEL 1: the batch size = 1, i.e. input 1sec=250 data points, gives 1 output (rating), aka. many-to-one
        :param batch_size: Batch size
        :param randomize: Whether to randomize the order in data
        :return: Next batch
        """

        # if batch_size > 1:
        #     raise ValueError("A batch_size of > 1 is not recommanded at this point")

        if len(self.remaining_slices) >= batch_size:

            # Select slize according to number of batches
            if randomize:

                selection_array_idx = np.random.choice(a=range(len(self.remaining_slices)),
                                                       size=batch_size,
                                                       replace=False)
                # = np.random.randint(low=0, high=len(self.remaining_slices), size=batch_size)  # with replacement

            else:  # if not random
                selection_array_idx = range(batch_size)

            selection_array = self.remaining_slices[selection_array_idx]

            # Remove drawn batch from _still_available_slices:
            self.remaining_slices = np.delete(arr=self.remaining_slices, obj=selection_array_idx)

            # Update index
            self._index_in_epoch += batch_size

            # Check whether slices left for next batch
            if len(self.remaining_slices) == 0:
                # if no slice left, start new epoch
                self.new_epoch()  # self._epochs_completed += 1
                self._index_in_epoch = 0
                self.remaining_slices = self.reset_remaining_slices()

                # print("\nNo slices left take new slices")  # test

        else:  # there are still slices left but not for the whole batch
            # First copy remaining slices to selection_array
            selection_array = self.remaining_slices

            # Now reset, i.e. new epoch
            self.new_epoch()  # self._epochs_completed += 1
            remaining_slices_to_draw = batch_size - len(self.remaining_slices)
            self._index_in_epoch = remaining_slices_to_draw  # index in new epoch
            self.remaining_slices = np.arange(self._num_time_slices)  # reset

            # Draw remaining slices from new epoch
            if randomize:
                additional_array_idx = np.random.choice(a=range(len(self.remaining_slices)),
                                                        size=remaining_slices_to_draw,
                                                        replace=False)
                # np.random.randint(low=0, high=len(self.remaining_slices), size=remaining_slices_to_draw)

            else:  # if not random
                additional_array_idx = range(remaining_slices_to_draw)

            additional_array = self.remaining_slices[additional_array_idx]

            # Remove from _still_available_slices:
            self.remaining_slices = np.delete(arr=self.remaining_slices, obj=additional_array_idx)

            # Append to selection_array
            selection_array = np.append(arr=selection_array, values=additional_array)

            # print("\nTake the rest slices and fill with new slices")  # test

        # Update current_batch
        self.current_batch = selection_array

        return self._eeg[selection_array, :, :], self._ratings[selection_array]


def concatenate(array1, array2):
    return np.concatenate((array1, array2), axis=0)


def splitter(array_to_split, n_splits):
    n_prune = 0
    while True:
        try:
            array_to_split = np.array(np.split(array_to_split, n_splits, axis=0))
        except ValueError:
            # print(err)
            array_to_split = array_to_split[0:-1]  # prune until split is possible
            n_prune += 1

        else:
            if n_prune > 0:
                print("Input array was pruned by", n_prune)
            break

    return array_to_split


def read_data_sets(subject, component, s_fold_idx, s_fold=10, cond="NoMov", sba=True, hilbert_power=True,
                   s_freq_eeg=250.):
    """
    Returns the s-fold-prepared dataset.
    S-Fold Validation, S=5: [ |-Train-|-Train-|-Train-|-Valid-|-Train-|] Dataset
    Args:
        subject: Subject Nr., subject dataset for training
        component: which component to feed
        s_fold_idx: index which of the folds is taken as validation set
        s_fold: s-value of S-Fold Validation [default=10]
        cond: Either "NoMov"(=default) or "Mov"
        sba: Whether to use SBA-data
        hilbert_power: hilbert-transform SSD-components, then extract z-scored power
        s_freq_eeg: Sampling Frequency of EEG
    Returns:
        Train, Validation Datasets
    """

    assert s_fold_idx < s_fold, "s_fold_idx (={}) must be in the range of the number of folds (={})".format(s_fold_idx,
                                                                                                            s_fold)
    assert cond in ["NoMov", "Mov"], "cond must be either 'NoMov' or 'Mov'"

    if not type(component) is list:
        assert component in range(5+1), "Component must be in range [1,2,3,4,5]"

    eeg_data = load_ssd_component()
    rating_data = load_rating_files()
    condition = rating_data[str(subject)]["condition"]
    t_roller_coasters = update_coaster_lengths(empty_t_array=np.zeros((len(roller_coasters))), sba=sba)

    # Subsample the validation set from the train set
    # 0) Take Space-Break-Ande (SBA) files from dictionaries

    # eeg_concat = concatenate(array1=eeg_data[str(subject)]["Space_NoMov"][:, 0:2],
    #                          array2=eeg_data[str(subject)]["Ande_NoMov"][:, 0:2])
    # rating_concat = concatenate(array1=rating_data[str(subject)]["Space_NoMov"],
    #                             array2=rating_data[str(subject)]["Ande_NoMov"])

    # Best component selected based on highest xcorr with ratings
    # Comparison with worst xcorr
    # eeg_sba = eeg_data[str(subject)]["SBA"][cond][:, 0:2]  # = first 2 components
    if not type(component) is list:
        eeg_sba = eeg_data[str(subject)]["SBA"][cond][:, component-1]  # does not work with list
        eeg_sba = np.reshape(eeg_sba, newshape=(eeg_sba.shape[0], 1))
    else:
        eeg_sba = eeg_data[str(subject)]["SBA"][cond][:, [comp-1 for comp in component]]

    rating_sba = rating_data[str(subject)]["SBA"][cond]

    # Check whether EEG data too long
    if sba:
        len_test = eeg_sba.shape[0] / s_freq_eeg - rating_sba.shape[0]
        if len_test > 0.0:
            to_cut = int(len_test * s_freq_eeg)
            print("EEG data of S{} trimmed by {} data points".format(str(subject).zfill(2), to_cut))
            to_cut /= 3  # 3 phases of SBA
            to_delete = np.cumsum(t_roller_coasters)  # [ 148.,  178.,  270.]
            to_delete *= s_freq_eeg
            assert to_cut == 1, "Code needs to be adapted for other diverging lengths of eeg_sba"
            # np.delete(arr=eeg_sba, obj=to_delete, axis=0).shape
            eeg_sba = np.delete(arr=eeg_sba, obj=to_delete, axis=0)

    # Normalize rating_sba to [-1;1] due to tanh-output of LSTMnet
    rating_sba = normalization(array=rating_sba, lower_bound=-1, upper_bound=1)

    if hilbert_power:
        # print("I load Hilbert transformed data (z-power)")

        def calc_hilbert_z_power(array):
            """square(abs(complex number)) = power = squarred length of complex number, see: Cohen (2014, p.160-2)"""
            analytical_signal = hilbert(array)
            amplitude_envelope = np.abs(analytical_signal)
            power = np.square(amplitude_envelope)
            # z-score of power contains power information and its variance, while centred around zero
            hilbert_z_power = z_score(array=power)  # z-score
            # could be smoothed to small degree, e.g., smooth(hilbert_z_power, 10)...
            return hilbert_z_power

        if type(component) is list:
            for comp in range(eeg_sba.shape[1]):
                eeg_sba[:, comp] = calc_hilbert_z_power(array=eeg_sba[:, comp])
        else:
            eeg_sba = calc_hilbert_z_power(array=eeg_sba)

    # 1) Split data in S(=s_fold) sets
    # np.split(np.array([1,2,3,4,5,6]), 3) >> [array([1, 2]), array([3, 4]), array([5, 6])]

    # split EEG data w.r.t. to total sba-length
    eeg_sba_split = splitter(eeg_sba, n_splits=int(sum(t_roller_coasters)))  # [sec, data-points, components)

    # eeg_concat_split[0][0:] first to  250th value in time
    # eeg_concat_split[1][0:] 250...500th value in time
    rating_split = splitter(array_to_split=rating_sba, n_splits=s_fold)
    eeg_split = splitter(array_to_split=eeg_sba_split, n_splits=s_fold)

    # eeg_concat_split.shape    # (n_chunks[in 1sec], n_samples_per_chunk [250Hz], channels)
    # eeg_split.shape           # (s_fold, n_chunks_per_fold, n_samples_per_chunk, channels)
    # rating_split.shape        # (s_fold, n_samples_per_fold,)

    # Assign variables accordingly:
    validation_eeg = eeg_split[s_fold_idx]
    validation_ratings = rating_split[s_fold_idx]
    train_eeg = np.delete(arr=eeg_split, obj=s_fold_idx, axis=0)  # removes the val-set from the data set (np.delete)
    train_ratings = np.delete(arr=rating_split, obj=s_fold_idx, axis=0)
    # Merge the training sets again (concatenate for 2D & vstack for >=3D)
    # Cautious: Assumption that performance is partly independent of correct order (also done already with SBA)
    train_eeg = np.vstack(train_eeg)  # == np.concatenate(train_eeg, axis=0)
    train_ratings = np.concatenate(train_ratings, axis=0)

    # Create datasets
    train = DataSet(name="Training", eeg=train_eeg, ratings=train_ratings, subject=subject, condition=condition)
    validation = DataSet(name="Validation", eeg=validation_eeg, ratings=validation_ratings, subject=subject,
                         condition=condition)
    # Test set
    # test = DataSet(eeg=test_eeg, ratings=test_ratings, subject=subject, condition=condition)
    test = None

    # return base.Datasets(train=train, validation=validation, test=test), s_fold_idx
    return {"train": train, "validation": validation, "test": test}


def get_nevro_data(subject, component, s_fold_idx=None, s_fold=10, cond="NoMov", sba=True, hilbert_power=True,
                   s_freq_eeg=250.):
    """
      Prepares NeVRo dataset.
      Args:
        subject: Which subject data to train
        component: Which component to feed
        s_fold_idx: index which of the folds is taken as validation set
        s_fold: s-value of S-Fold Validation [default=10]
        cond: Either "NoMov"(=default) or "Mov"
        sba: Whether to use SBA-data
        hilbert_power: hilbert-transform SSD-components, then extract z-scored power
        s_freq_eeg: Sampling Frequency of EEG
      Returns:
        Train, Validation Datasets (+Test set)
      """
    if s_fold_idx is None:
        s_fold_idx = np.random.randint(low=0, high=s_fold)
        print("s_fold_idx randomly chosen:", s_fold_idx)

    return read_data_sets(subject=subject, component=component, s_fold_idx=s_fold_idx, s_fold=s_fold,
                          cond=cond, sba=sba, hilbert_power=hilbert_power, s_freq_eeg=s_freq_eeg)


# Testing
# nevro_data = get_nevro_data(subject=36, component=5, s_fold_idx=9, s_fold=10, cond="NoMov", sba=True)

# for _ in range(27):
#     x = nevro_data["validation"].next_batch(batch_size=4, randomize=True)
#     print("Current Batch:", nevro_data["validation"].current_batch)
#     print("n_remaining slices: {}/{}".format(len(nevro_data["validation"].remaining_slices),
#                                              nevro_data["validation"]._num_time_slices))
#     print("index in current epoch:", nevro_data["validation"]._index_in_epoch)
#     print("epochs copmleted:", nevro_data["validation"]._epochs_completed)
#     print("")

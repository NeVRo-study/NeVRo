"""
Load Data for LSTM Model
    • Load Data
    • Create Batches

Author: Simon Hofmann | <[surname].[lastname][at]protonmail.com> | 2017
"""

import numpy as np
import os.path
import copy
# from Meta_Functions import *

# Change to Folder which contains files
wdic = "../../Data/EEG_SSD/"
wdic_Comp = "../../Data/EEG_SSD/Components/"
wdic_Rating = "../../Data/Ratings/"
wdic_Data = "../../Data/"
wdic_cropECG = "../../Data/Data EEG export/NeVRo_ECG_Cropped/"

# initialize variables
subjects = [36, 37]
n_sub = len(subjects)
# roller_coasters = np.array(['Space_NoMov', 'Space_Mov', 'Ande_Mov', 'Ande_NoMov'])
roller_coasters = np.array(['Space_NoMov', 'Ande_NoMov'])  # for testing
t_roller_coasters = np.zeros((len(roller_coasters)))  # init


def update_coaster_lengths(empty_t_array):
    sfreq = 500
    for sub in subjects:
        for n, coast in enumerate(roller_coasters):
            time_ecg_fname = wdic_cropECG + "NVR_S{}_{}.txt".format(str(sub).zfill(2), coast)
            if os.path.isfile(time_ecg_fname):
                ecg_file = np.loadtxt(time_ecg_fname)
                len_time_ecg = len(ecg_file)/sfreq
                empty_t_array[n] = len_time_ecg if len_time_ecg > empty_t_array[n] \
                    else empty_t_array[n]

    full_t_array = empty_t_array  # make-up
    print("Length of each roller coaster:\n", full_t_array)
    return full_t_array

t_roller_coasters = update_coaster_lengths(empty_t_array=t_roller_coasters)


# Load file
def load_ssd_files():
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
                if not (np.round(t_roller_coasters[num], 1) == np.round(time_steps[-1]/1000, 1)
                        and np.round(time_steps[-1]/1000, 1) == np.round(len(time_steps)/250, 1)):
                    print("Should be all approx. the same:\nt_roller_coasters[num]: {} \ntime_steps[-1]/1000: {}"
                          "\nlen(time_steps)/sfreq(250): {}".format(t_roller_coasters[num],
                                                                    time_steps[-1]/1000,
                                                                    len(time_steps)/250))

    return ssd_dic

SSD_dic = load_ssd_files()
# print(SSD_dic[str(subjects[0])][roller_coasters[0]]["df"].shape)  # roller_coasters[0] == 'Space_NoMov'
# print(SSD_dic[str(subjects[0])][roller_coasters[0]]["df"].shape)  # str(subjects[0]) == '36'
# print(SSD_dic[str(subjects[0])][roller_coasters[0]]["t_steps"].shape)
# print(SSD_dic["channels"].shape)


def load_ssd_component():
    """
    Loads SSD components (files) of each subject in subjects
    :return: SSD component files [sub_df] in form of dictionary
    """

    # Create SSD Component Dictionary
    ssd_comp_dic_keys = [str(i) for i in subjects]  # == list(map(str, subjects))  # these keys also work int
    ssd_comp_dic = {}
    ssd_comp_dic.update((key, dict.fromkeys(roller_coasters, [])) for key in ssd_comp_dic_keys)
    # ssd_comp_dic.update({"components": []})  # this varies for each subject & roller coaster
    for subject in subjects:
        for num, coaster in enumerate(roller_coasters):
            file_name = wdic_Comp + "S{}_{}_Components.txt".format(str(subject).zfill(2), coaster)  # adapt file_name

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
                pre_sub_df = np.genfromtxt(file_name, delimiter=";", dtype="str")[1:]  # leave first row = comp. names
                sub_df = np.zeros((pre_sub_df.shape[0], n_comp))
                for row in range(pre_sub_df.shape[0]):
                    values_at_t = pre_sub_df[row].split("\t")
                    sub_df[row] = list(map(float, values_at_t))  # == [float(i) for i in values_at_t]
                # del pre_sub_df  # save WM

                # sub_df.shape=(38267, n_comp=22) for S36_Space_NoMov | Samp. Freq. = 250 | 38267/250 = 153.068sec
                # Fill in SSD Dictionary: ssd_comp_dic
                ssd_comp_dic[str(subject)][coaster] = copy.copy(sub_df)

                # Check times:
                samp_freq = 250
                if not (np.round(t_roller_coasters[num], 1) == np.round(sub_df.shape[0]/samp_freq, 1)):
                    print("Should be all approx. the same:\nTime of {}: {}\nLength of sub_df/samp_freq(250): {}".format(
                        roller_coasters[num], t_roller_coasters[num], sub_df.shape[0] / samp_freq))

    return ssd_comp_dic

SSD_Comp_dic = load_ssd_component()
# print(SSD_Comp_dic[str(subjects[0])][roller_coasters[0]].shape,
#       "=",
#       SSD_Comp_dic[str(subjects[0])][roller_coasters[0]].shape[0]/250,
#       "sec\t|",
#       SSD_Comp_dic[str(subjects[0])][roller_coasters[0]].shape[1],
#       "components")
# print(SSD_Comp_dic[str(subjects[0])][roller_coasters[1]].shape,
#       "=",
#       SSD_Comp_dic[str(subjects[0])][roller_coasters[1]].shape[0]/250,
#       "sec\t|",
#       SSD_Comp_dic[str(subjects[0])][roller_coasters[1]].shape[1],
#       "components")


# @function_timed  # after executing following function this returns runtime
def load_rating_files(samp_freq=1):
    """
    Loads (z-scored) Ratings files of each subject in subjects (ignore other files, since fluctuating samp.freq.).
    :param samp_freq: sampling frequency, either 1Hz [default], oder 50Hz
    :return:  Rating Dict
    """

    # Check whether input correct
    # assert (samp_freq == 1 or samp_freq == 50), "samp_freq must be either 1 or 50 Hz"
    if not (samp_freq == 1 or samp_freq == 50):
        raise ValueError("samp_freq must be either 1 or 50 Hz")
    # Load Table of Conditions
    table_of_condition = np.genfromtxt(wdic_Data + "Table_of_Conditions.csv", delimiter=";")
    table_of_condition = table_of_condition[1:, ]  # remove first column (sub-nr, condition, gender (1=f, 2=m))
    # Create Rating-dictionary
    rating_dic_keys = list(map(str, subjects))  # == [str(i) for i in subjects]
    rating_dic = {}
    # each subejct has a sub-dic for each roller coaster
    rating_dic.update((key, dict.fromkeys(roller_coasters, [])) for key in rating_dic_keys)
    # For each subject fill condtition in
    for key in rating_dic_keys:
        key_cond = int(str(table_of_condition[np.where(table_of_condition[:, 0] == int(key)), 1])[3])
        rating_dic[key].update({"condition": key_cond})

    for subject in subjects:
        for num, coaster in enumerate(roller_coasters):
            # adapt file_name accordingly (run <=> condition,
            coast = "Space".lower() if "Space" in coaster else "andes"
            runs = "1" if "NoMov" in coaster and rating_dic[str(subject)]["condition"] == 2 else "2"

            rating_filename = wdic_Rating + "{}/{}_z/{}Hz/NVR_S{}_run_{}_{}_rat_z.txt".format(coast,
                                                                                              coast,
                                                                                              str(samp_freq),
                                                                                              str(subject).zfill(2),
                                                                                              runs,
                                                                                              coast)

            if os.path.isfile(rating_filename):
                # Load according file
                rating_file = np.genfromtxt(rating_filename, delimiter=',')[:, 1]  # only load column with ratings
                # Fill in rating dictionary
                rating_dic[str(subject)][coaster] = copy.copy(rating_file)

                # Check times:
                if not (np.round(t_roller_coasters[num], 1) == np.round(len(rating_file)/samp_freq, 1)):
                    print("Should be all approx. the same:\nTime of {}: {}\nLength of Rating / s_freq({}): {}".format(
                        roller_coasters[num], t_roller_coasters[num], samp_freq, len(rating_file)/samp_freq))

    return rating_dic

Rating_dic = load_rating_files(samp_freq=1)  # samp_freq=1 or samp_freq=50


# TODO create Batches

# TODO S-FOLD: train-test-set split

class DataSet(object):
    """
    Utility class (http://wiki.c2.com/?UtilityClasses) to handle dataset structure
    """
    def __init__(self, eeg, ratings):
        """
        Builds dataset with EEG data and Ratings
        :param eeg: EEG data
        :param ratings: Rating Data
        """
        # TODO this assert needs to be adapted according to the sampling freq: s-freq(eeg) >= s-freq(ratings)
        assert eeg.shape[0] == ratings[0], "eeg.shape: {}, ratings.shape: {}".format(eeg.shape, ratings.shape)
        self._num_time_slices = eeg.shape[0]
        self._eeg = eeg  # input
        self._ratings = ratings  # target
        self._epochs_completed = 0
        self._index_in_epoch = 0
        # TODO include subject-option

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

    def next_batch(self, batch_size):
        """
        Return the next 'batch_size' examples from this data set
        :param batch_size: Batch size
        :return: Next batch
        """
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_time_slices:
            self._epochs_completed += 1

            perm = np.arrange(self._num_time_slices)
            np.random.shuffle(perm)  # TODO check whether makes sense for LSTM
            self._eeg = self._eeg[perm]
            self._ratings = self._ratings[perm]

            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_time_slices

        end = self._index_in_epoch

        return self._eeg[start:end], self._ratings[start:end]

        # TODO continue here:

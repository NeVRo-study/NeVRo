"""
Load Data for LSTM Model
    • Load Data
    • Create Batches

Author: Simon Hofmann | <[surname].[lastname][at]protonmail.com> | 2017
"""

import numpy as np
import os.path
import copy

# Change to Folder which contains files
wdic = "../../Data/EEG_SSD/"
wdic_Comp = "../../Data/EEG_SSD/Components/"
wdic_Rating = "../../Data/Ratings/"
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


# TODO below...
def load_ssd_component():
    """
    Loads 2 best SSD components (files) of each subject in subjects
    :return: SSD component files [components, time_steps, sub_df] in form of dictionary
    """
    count = 0
    # Create SSD Component Dictionary
    # dic = {k: v for k, v in [("comp1", [])]}
    # dic.update({newkey: newkey_value})
    ssd_comp_dic_keys = ["components"] + [str(i) for i in subjects]
    ssd_comp_dic = dict.fromkeys(ssd_comp_dic_keys, {})  # creats dict from given sequence and given value (could be empty =None)
    ssd_comp_dic_keys.pop(0)  # remove channels-key again
    # create sub_dic for each roller coaster (time, df)
    coast_dic = {}
    coast_dic.update((key, {"t_steps": [], "df": []}) for key in roller_coasters)
    # each subejct has a sub-dic for each roller coaster
    ssd_comp_dic.update((key, copy.deepcopy(coast_dic)) for key in ssd_comp_dic_keys)  # deep.copy(!)
    # # Test
    # print("ssd_comp_dic\n", ssd_comp_dic)
    # ssd_comp_dic["36"]['Ande_NoMov']["df"] = 999
    # print("ssd_comp_dic\n", ssd_comp_dic)  # ssd_comp_dic["36"] is ssd_comp_dic["37"]

    for subject in subjects:
        for num, coaster in enumerate(roller_coasters):
            count += 1
            file_name = wdic_Comp + "S{}_{}_Components.txt".format(str(subject).zfill(2), coaster)  # adapt file_name

            if os.path.isfile(file_name):
                if count == 1:  # do only once
                    # x = x.split("\t")  # '\t' divides values
                    components = np.genfromtxt(file_name, delimiter="\t", dtype="str")[1:, 0]
                    ssd_comp_dic["channels"] = components  # 32 components; shape=(32,)
                sub_df = np.genfromtxt(file_name, delimiter="\t")[0:, 1:-1]  # first row:= time; in last col only NAs
                # Sampling Frequency = 250
                time_steps = sub_df[0, :]  # shape=(24276,) for S36_Ande_NoMov
                sub_df = sub_df[1:, :]  # shape=(32, 24276) for S36_Ande_NoMov
                # Fill in SSD Dictionary: ssd_comp_dic
                ssd_comp_dic[str(subject)][coaster]["t_steps"] = copy.copy(time_steps)
                ssd_comp_dic[str(subject)][coaster]["df"] = copy.copy(sub_df)

                # Check times:
                if not (np.round(t_roller_coasters[num], 1) == np.round(time_steps[-1] / 1000, 1)
                        and np.round(time_steps[-1] / 1000, 1) == np.round(len(time_steps) / 250, 1)):
                    print("Should be all approx. the same:\nt_roller_coasters[num]: {} \ntime_steps[-1]/1000: {}"
                          "\nlen(time_steps)/sfreq(250): {}".format(t_roller_coasters[num],
                                                                    time_steps[-1] / 1000,
                                                                    len(time_steps) / 250))

    return ssd_comp_dic

SSD_Comp_dic = load_ssd_component()
# print(SSD_Comp_dic[str(subjects[0])][roller_coasters[0]]["df"].shape)  # roller_coasters[0] == 'Space_NoMov'
# print(SSD_Comp_dic[str(subjects[0])][roller_coasters[0]]["df"].shape)  # str(subjects[0]) == '36'
# print(SSD_Comp_dic[str(subjects[0])][roller_coasters[0]]["t_steps"].shape)
# print(SSD_Comp_dic["channels"].shape)


def load_rating_files():
    """
    Loads Ratings files of each subject in subjects
    :return: Rating Dict
    """
    pass
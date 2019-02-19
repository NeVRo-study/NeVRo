# coding=utf-8
"""
Load Data for LSTM Model. Only use Space-Break-Andes (SBA) versions, which are pruned to 148, 30 & 92 sec,
respectively.

    • Load Data
    • Create Batches

..........................................................................................................
MODEL 1: Feed best SSD-extracted alpha components (250Hz) into LSTM to predict ratings (1Hz)


Comp1: [x1_0, ..., x1_250], [x1_251, ..., x1_500], ... ]  ==> LSTM \
 ...                                                                ==>> Rating [y0, y1, ... ]
CompN: [x2_0, ..., x2_250], [x2_251, ..., x2_500], ... ]  ==> LSTM /

..........................................................................................................
MODEL 2: Feed slightly pre-processed channels data, i.e. 30 channels, into LSTM to predict ratings

Channel01: [x01_0, ..., x01_250], [x01_251, ..., x01_500], ... ]  ==> LSTM \
Channel02: [x02_0, ..., x02_250], [x02_251, ..., x02_500], ... ]  ==> LSTM  \
...                                                                          ==>> Rating [y0, y1, ... ]
...                                                                         /
Channel30: [x30_0, ..., x30_250], [x30_251, ..., x30_500], ... ]  ==> LSTM /
..........................................................................................................

Author: Simon Hofmann | <[surname].[lastname][at]protonmail.com> | 2019
"""

# import sys
# sys.path.insert(0, './LSTM Model')  # or set the folder as source root
import os.path
import copy
from Meta_Functions import *
import pandas as pd

# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

#  load EEG data of from *.set (EEGlab format) with: mne.io.read_raw_eeglab() and mne.read_epochs_eeglab()

# # # Define data folder roots

path_data = "../../Data/"

path_ssd = path_data + "EEG/10_SSD/"
path_spoc = path_data + "EEG/11_SPOC/"

# # SBA-pruned data (148:space, 30:break, 92:ande)

# # Rating data
# path_rating = path_data + "ratings/preprocessed/z_scored_alltog/"
path_rating = path_data + "ratings/preprocessed/not_z_scored/"  # min-max scale later to [-1, +1]
path_rating_bins = path_data + "ratings/preprocessed/classbin_ratings/"
# # ECG data
path_ecg_crop = path_data + "Data EEG export/NeVRo_ECG_Cropped/"
path_ecg_sba = path_data + "ECG/SBA/z_scored_alltog/"
path_results_xcorr = "../../Results/x_corr/"

# # # Initialize variables
# subjects = [36]  # [36, 37]
# subjects = range(1, 45+1)
# dropouts = [1,12,32,33,38,40,45]

# roller_coasters = np.array(['Space_NoMov', 'Space_Mov', 'Ande_Mov', 'Ande_NoMov'])
roller_coasters = np.array(['Space_NoMov', "Break_NoMov", 'Ande_NoMov']) if True \
    else np.array(['Space_NoMov', 'Ande_NoMov'])

# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>


def update_coaster_lengths(subjects, empty_t_array, sba=True):
    """
    Creates an array of the lengths of the different phases (e.g., roller coasters)
    :param subjects: list of subjects or single subject
    :param empty_t_array: array to be filled
    :param sba: asks whether pruned SBA-files should be used to calculate times.
    :return: array with lengths of phases
    """
    sfreq = 500.

    if not sba:

        if not isinstance(subjects, list):
            subjects = list(subjects)

        for sub in subjects:
            for n, coast in enumerate(roller_coasters):
                time_ecg_fname = path_ecg_crop + "NVR_S{}_{}.txt".format(str(sub).zfill(2), coast)
                if os.path.isfile(time_ecg_fname):
                    ecg_file = np.loadtxt(time_ecg_fname)
                    len_time_ecg = len(ecg_file) / sfreq
                    empty_t_array[n] = len_time_ecg if len_time_ecg > empty_t_array[n] \
                        else empty_t_array[n]

    else:
        empty_t_array = np.array([148., 30., 92])

    full_t_array = empty_t_array  # make-up
    # print("Length of each roller coaster:\n", full_t_array)
    return full_t_array

# t_roller_coasters = np.zeros((len(roller_coasters)))  # init
# t_roller_coasters = update_coaster_lengths(subjects=[36, 44], empty_t_array=t_roller_coasters, sba=True)
# t_roller_coasters = np.array([148, 30, 92])   # SBA


def get_filename(subject, filetype, band_pass, cond="nomov", sba=True):
    """
    Receive the filename for specific setup.
    :param subject: subject number
    :param filetype: Either 'SSD' or 'SPOC'
    :param band_pass: Data either band-pass filtered or not. 'SPOC' always band-pass filtered
    :param cond: 'nomov' [Default] or 'mov'
    :param sba: SBA data
    :return: filename
    """

    assert cond in ["nomov", "mov"], "cond must be either 'nomov' or 'mov'"

    assert filetype.upper() in ["SSD", "SPOC"], "filetype must be either 'SSD' or 'SPOC'"

    if sba:
        if filetype.upper() == "SSD":
            if not band_pass:
                file_name = path_ssd_sba_comp_broadband + cond + \
                            "/NVR_S{}_SBA_{}_SSD_Components_SBA_CNT.txt".format(
                                str(subject).zfill(2), cond)
            else:
                file_name = path_ssd_sba_comp + cond + "/NVR_S{}_SBA_{}_SSD_Components_SBA_CNT.txt".format(
                    str(subject).zfill(2), cond)

        else:  # filetype.upper() == "SPOC":
            if band_pass:
                file_name = path_spoc_sba_comp + cond + "/NVR_S{}_SBA_{}_PREP_SPOC_Components.txt".format(
                    str(subject).zfill(2), cond)
            else:
                raise ValueError("There are no non-band_passed SPOC files here (yet)")

    else:
        raise FileExistsError("Non-SBA data not yet implemented.")

    return file_name


# Load file
def load_ssd_files(subjects, samp_freq=250., sba=True):
    """
    Loads all channel SSD files of each subject in subjects
    :return: SSD files [channels, time_steps, sub_df] in form of dictionary
    """

    if not isinstance(subjects, list):
        subjects = list(subjects)

    count = 0
    # Create SSD-dictionary
    # dic = {k: v for k, v in [("bla", [])]}
    # dic.update({newkey: newkey_value})
    ssd_dic_keys = ["channels"] + [str(i) for i in subjects]
    ssd_dic = dict.fromkeys(ssd_dic_keys, {})  # creats dict from sequence and given value (can be None)
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
                    channels = np.genfromtxt(file_name, delimiter="\t", dtype="str")[1:, 0]
                    ssd_dic["channels"] = channels  # 32 channels; shape=(32,)
                sub_df = np.genfromtxt(file_name, delimiter="\t")[0:, 1:-1]
                # first row:= time; in last col only NAs

                # Sampling Frequency = 250
                time_steps = sub_df[0, :]  # shape=(24276,) for S36_Ande_NoMov
                sub_df = sub_df[1:, :]  # shape=(32, 24276) for S36_Ande_NoMov

                # Fill in SSD Dictionary: ssd_dic
                ssd_dic[str(subject)][coaster]["t_steps"] = copy.copy(time_steps)
                ssd_dic[str(subject)][coaster]["df"] = copy.copy(sub_df)

                # Check times:
                # samp_freq = 250.

                t_roller_coasters = update_coaster_lengths(subjects=subject,
                                                           empty_t_array=np.zeros((len(roller_coasters))),
                                                           sba=sba)
                if not (np.round(t_roller_coasters[num], 1) == np.round(time_steps[-1] / 1000., 1)
                        and np.round(time_steps[-1] / 1000., 1) == np.round(len(time_steps) / samp_freq, 1)):
                    print("Should be all approx. the same:"
                          "\nt_roller_coasters[num]: {} "
                          "\ntime_steps[-1]/1000: {}"
                          "\nlen(time_steps)/sfreq(250): {}".format(t_roller_coasters[num],
                                                                    time_steps[-1] / 1000.,
                                                                    len(time_steps) / samp_freq))

    return ssd_dic

# SSD_dic = load_ssd_files(subjects=[36,44], samp_freq=250.)
# print(SSD_dic[str(subjects[0])][roller_coasters[0]]["df"].shape)  # roller_coasters[0] == 'Space_NoMov'
# print(SSD_dic[str(subjects[0])][roller_coasters[0]]["df"].shape)  # str(subjects[0]) == '36'
# print(SSD_dic[str(subjects[0])][roller_coasters[0]]["t_steps"].shape)
# print(SSD_dic["channels"].shape)


def load_ssd_component(subjects, samp_freq=250., sba=True, band_pass=True):
    """
    :param band_pass: Whether SSD components are band-passed filter for alpha
    :param subjects: list of subjects or single subject
    :param samp_freq: sampling frequency of SSD-components
    :param sba: if True (=Default), take SBA-z-scored components
    Loads SSD components (files) of each subject in subjects
    :return: SSD component files [sub_df] in form of dictionary
    """

    if not isinstance(subjects, list):
        subjects = [subjects]

    # Create SSD Component Dictionary
    ssd_comp_dic_keys = [str(i) for i in subjects]
    # == list(map(str, subjects))  # these keys also work int
    ssd_comp_dic = {}
    ssd_comp_dic.update((key, dict.fromkeys(roller_coasters, [])) for key in ssd_comp_dic_keys)

    condition_keys = [element.split("_")[1] for element in roller_coasters]  # "NoMov", "Mov"
    condition_keys = list(set(condition_keys))

    t_roller_coasters = update_coaster_lengths(subjects=subjects,
                                               empty_t_array=np.zeros((len(roller_coasters))),
                                               sba=sba)

    if sba:
        for key in ssd_comp_dic_keys:
            ssd_comp_dic[key].update({"SBA": dict.fromkeys(condition_keys, [])})

    for subject in subjects:

        for num, coaster in enumerate(roller_coasters):

            folder = coaster.split("_")[1]  # either "Mov" or "NoMov" (only needed for sba case)

            if sba:

                if band_pass:
                    file_name = path_ssd_sba_comp + \
                                folder + \
                                "/NVR_S{}_SBA_{}_SSD_Components_SBA_CNT.txt".format(str(subject).zfill(2),
                                                                                    folder)
                else:
                    file_name = path_ssd_sba_comp_broadband + \
                                folder + \
                                "/NVR_S{}_SBA_{}_SSD_Components_SBA_CNT.txt".format(str(subject).zfill(2),
                                                                                    folder)
                    # "NVR_S36_SBA_NoMov_SSD_Components_SBA_CNT.txt"

            else:
                file_name = path_ssd_comp + "S{}_{}_Components.txt".format(str(subject).zfill(2), coaster)

            if os.path.isfile(file_name):
                # x = x.split("\t")  # '\t' divides values
                # N ≈ 20 components sorted from 1-N according to the signal-to-noise-ration (SNR)
                # cpts = np.genfromtxt(file_name, delimiter=";", dtype="str")[0].split("\t")[:-1]
                # # last=' '
                # for idx, comp in enumerate(cpts):
                #     cpts[idx] = "comp_" + comp
                # ssd_comp_dic["cpts"] = cpts  # 32 cpts; shape=(32,)

                n_comp = len(np.genfromtxt(file_name, delimiter=";", dtype="str")[0].split("\t")[:-1])
                # last=' '

                # columns = components, rows value per timestep
                pre_sub_df = np.genfromtxt(file_name, delimiter=";", dtype="str")[1:]
                # first row = comp.names
                sub_df = np.zeros((pre_sub_df.shape[0], n_comp))
                for row in range(pre_sub_df.shape[0]):
                    values_at_t = pre_sub_df[row].split("\t")
                    sub_df[row] = list(map(float, values_at_t))  # == [float(i) for i in values_at_t]

                # sub_df.shape=(38267, n_comp=22)
                # for S36_Space_NoMov | Samp. Freq. = 250 | 38267/250 = 153.068sec

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
                    if not (np.round(t_roller_coasters[num], 1) == np.round(sub_df.shape[0] / samp_freq,
                                                                            1)):
                        print("Should be all approx. the same:\nTime of {}: {}"
                              "\nLength of sub_df/samp_freq({}): {}".format(roller_coasters[num],
                                                                            t_roller_coasters[num],
                                                                            int(samp_freq),
                                                                            sub_df.shape[0] / samp_freq))
                else:  # if sba:
                    if not (np.round(sum(t_roller_coasters), 1) == np.round(sub_df.shape[0] / samp_freq,
                                                                            1)):
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


def load_spoc_component(subjects, samp_freq=250., sba=True, band_pass=True):
    """
    :param band_pass: Whether SPOC components are band-passed filter for alpha
    :param subjects: list of subjects or single subject
    :param samp_freq: sampling frequency of SPOC-components
    :param sba: if True (=Default), take SBA-z-scored components
    Loads SSD components (files) of each subject in subjects
    :return: SSD component files [sub_df] in form of dictionary
    """

    if not isinstance(subjects, list):
        subjects = [subjects]

    # Create SSD Component Dictionary
    spoc_comp_dic_keys = [str(i) for i in subjects]
    # == list(map(str, subjects))  # these keys also work int
    spoc_comp_dic = {}
    spoc_comp_dic.update((key, dict.fromkeys(roller_coasters, [])) for key in spoc_comp_dic_keys)

    condition_keys = [element.split("_")[1] for element in roller_coasters]  # "NoMov", "Mov"
    condition_keys = list(set(condition_keys))

    t_roller_coasters = update_coaster_lengths(subjects=subjects,
                                               empty_t_array=np.zeros((len(roller_coasters))),
                                               sba=sba)

    if sba:
        for key in spoc_comp_dic_keys:
            spoc_comp_dic[key].update({"SBA": dict.fromkeys(condition_keys, [])})

    for subject in subjects:
        for num, coaster in enumerate(roller_coasters):
            folder = coaster.split("_")[1]  # either "Mov" or "NoMov" (only needed for sba case)
            if sba:
                if band_pass:
                    file_name = path_spoc_sba_comp + \
                                folder + \
                                "/NVR_S{}_SBA_{}_PREP_SPOC_Components.txt".format(str(subject).zfill(2),
                                                                                  folder)
                else:
                    raise ValueError("There are no non-band_passed SPOC files here (yet)")

            else:
                file_name = path_ssd_comp + "S{}_{}_Components.txt".format(str(subject).zfill(2), coaster)

            if os.path.isfile(file_name):

                n_comp = len(np.genfromtxt(file_name, delimiter=";", dtype="str")[0].split("\t")[:-1])
                # last=' '

                # columns = components, rows value per timestep
                pre_sub_df = np.genfromtxt(file_name, delimiter=";", dtype="str")[1:]
                # first row = comp.names
                sub_df = np.zeros((pre_sub_df.shape[0], n_comp))
                for row in range(pre_sub_df.shape[0]):
                    values_at_t = pre_sub_df[row].split("\t")
                    sub_df[row] = list(map(float, values_at_t))  # == [float(i) for i in values_at_t]
                # del pre_sub_df  # save WM

                # sub_df.shape=(38267, n_comp=22)
                # for S36_Space_NoMov | Samp. Freq. = 250 | 38267/250 = 153.068sec

                # Fill in SSD Dictionary: spoc_comp_dic
                if not sba:
                    spoc_comp_dic[str(subject)][coaster] = copy.copy(sub_df)

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

                    spoc_comp_dic[str(subject)][coaster] = copy.copy(sub_df[sba_idx_start:sba_idx_end, :])

                    # process whole SBA only once per condition (NoMov, Mov)
                    if "Break" in coaster:
                        assert folder in condition_keys, "Wrong key"
                        spoc_comp_dic[str(subject)]["SBA"][folder] = copy.copy(sub_df)

                # Check times:
                # samp_freq = 250.
                if not sba:
                    if not (np.round(t_roller_coasters[num], 1) == np.round(sub_df.shape[0] / samp_freq,
                                                                            1)):
                        print("Should be all approx. the same:\nTime of {}: {}"
                              "\nLength of sub_df/samp_freq({}): {}".format(roller_coasters[num],
                                                                            t_roller_coasters[num],
                                                                            int(samp_freq),
                                                                            sub_df.shape[0] / samp_freq))
                else:  # if sba:
                    if not (np.round(sum(t_roller_coasters), 1) == np.round(sub_df.shape[0] / samp_freq,
                                                                            1)):
                        print("Should be all approx. the same:\nTime of SBA: {}"
                              "\nLength of sub_df/samp_freq({}): {}".format(sum(t_roller_coasters),
                                                                            int(samp_freq),
                                                                            sub_df.shape[0] / samp_freq))

    return spoc_comp_dic
# SPOC_Comp_dic = load_spoc_component(subjects=44)


def best_component(subject, best=True):
    """
    Choose the best SSD-alpha-component, w.r.t. x-corr with ratings, from list for given subject.
    :param subject: subject number
    :param best: if false, take random
    :return: best component number
    """

    # x_corr_table = np.genfromtxt(path_results_xcorr + "CC_AllSubj_Alpha_Ratings_smooth.csv",
    #                              delimiter=",", skip_header=1, dtype=float)

    # Load Table
    x_corr_table = pd.read_csv(path_results_xcorr + "CC_AllSubj_Alpha_Ratings_smooth.csv", index_col=0)
    # first col is idx
    # drop non-used columns
    x_corr_table = x_corr_table.drop(x_corr_table.columns[0:3], axis=1)

    if best:
        # x_corr_table.columns = ["components"]  # renamce columns
        component = x_corr_table.loc["S{}".format(str(subject).zfill(2))].values[0]

    else:
        component = best_comp = x_corr_table.loc["S{}".format(str(subject).zfill(2))].values[0]
        # choose first for now

        # choose another component that is not the best
        while component == best_comp:
            component = np.random.randint(low=1, high=5+1)

    print("The best correlating SSD component of Subject {} is Component Number {}".format(subject,
                                                                                           component))

    return component


# @function_timed  # after executing following function this returns runtime
def load_rating_files(subjects, samp_freq=1., sba=True, bins=False):
    """
    Loads (z-scored) Ratings files of each subject in subjects.
    (ignore other files, due to fluctuating samp.freq.)
    :param bins: whether to load ratings in forms of bins (low, medium, high arousal)
    :param subjects: list of subjects or single subject
    :param sba: if TRUE (default), process SBA files
    :param samp_freq: sampling frequency, either 1Hz [default], oder 50Hz
    :return:  Rating Dict
    """

    if not isinstance(subjects, list):
        subjects = [subjects]

    # Check whether input correct
    # assert (samp_freq == 1 or samp_freq == 50), "samp_freq must be either 1 or 50 Hz"
    if not (samp_freq == 1. or samp_freq == 50.):
        raise ValueError("samp_freq must be either 1 or 50 Hz")

    # Load Table of Conditions
    table_of_condition = np.genfromtxt(path_data + "Table_of_Conditions.csv", delimiter=";")
    table_of_condition = table_of_condition[1:, ]
    # remove first column (sub-nr, condition, gender (1=f, 2=m))

    # Create Rating-dictionary
    rating_dic_keys = list(map(str, subjects))  # == [str(i) for i in subjects]
    rating_dic = {}
    # each subejct has a sub-dic for each roller coaster
    rating_dic.update((key, dict.fromkeys(roller_coasters, [])) for key in rating_dic_keys)

    condition_keys = [element.split("_")[1] for element in roller_coasters]  # "NoMov", "Mov"
    condition_keys = list(set(condition_keys))

    t_roller_coasters = update_coaster_lengths(subjects=subjects,
                                               empty_t_array=np.zeros((len(roller_coasters))),
                                               sba=sba)

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

            rating_filename = path_rating + \
                              "{}/{}/{}Hz/NVR_S{}_run_{}_{}_rat_z.txt".format(coast,
                                                                              coast_folder,
                                                                              str(int(samp_freq)),
                                                                              str(subject).zfill(2),
                                                                              runs,
                                                                              coast)

            if os.path.isfile(rating_filename):
                # Load according file
                rating_file = np.genfromtxt(rating_filename, delimiter=',')[:, 1]
                # only load column with ratings
                # Fill in rating dictionary
                rating_dic[str(subject)][coaster] = copy.copy(rating_file)

                # Check times:
                if not (np.round(t_roller_coasters[num], 1) == np.round(len(rating_file) / samp_freq, 1)):
                    print("Should be all approx. the same:"
                          "\nTime of {}: {}"
                          "\nLength of Rating / s_freq({}): {}".format(roller_coasters[num],
                                                                       t_roller_coasters[num],
                                                                       samp_freq,
                                                                       len(rating_file) / samp_freq))

            # just process SBA once per condition (NoMov, Mov):
            if "Break" in coaster:
                cond_key = coaster.split("_")[1]  # either "Mov" or "NoMov" (only needed for sba case)
                assert cond_key in condition_keys, "Wrong Key"

                if bins:
                    sba_rating_filename = path_rating_bins + "{}/NVR_S{}_run_{}_alltog_epochs.txt".format(
                        coast_folder, str(subject).zfill(2), runs)
                else:
                    sba_rating_filename = path_rating + \
                                          "alltog/{}/{}Hz/NVR_S{}_run_{}_alltog_rat_z.txt".format(
                                              coast_folder, str(int(samp_freq)), str(subject).zfill(2),
                                              runs)

                if os.path.isfile(sba_rating_filename):
                    # Load according file
                    sba_rating_file = np.genfromtxt(sba_rating_filename, delimiter=',')[:, 1]
                    # in case of bin-files delete 1.entry (header)
                    if np.isnan(sba_rating_file[0]) and len(sba_rating_file) == 271:
                        sba_rating_file = np.delete(sba_rating_file, 0) - 2
                        # substract 2 to adapt range to [-1,1]
                        # -1: low, 0: mid, 1: high arousal
                        # print("Rating bins and count:", np.unique(sba_rating_file, return_counts=True))

                    # Fill in rating dictionary
                    rating_dic[str(subject)]["SBA"][cond_key] = copy.copy(sba_rating_file)

                    # Check times:
                    if not (np.round(sum(t_roller_coasters), 1) == np.round(
                            len(sba_rating_file) / samp_freq, 1)):
                        print("Should be all approx. the same:"
                              "\nTime of SBA: {}"
                              "\nLength of Rating/s_freq({}Hz): {}".format(
                            sum(t_roller_coasters), int(samp_freq), len(sba_rating_file) / samp_freq))

    return rating_dic


def load_ecg_files(subjects, sba=True, interpolate=True):
    """
    Load 1Hz heart rate data (bmp -> z-scored).
    :param subjects: list of subjects
    :param sba: Loading them in single phases or concatented into Space-Break-Andes (SBA)
    :param interpolate: whether to interpolate missing values
    :return: ECG data
    """

    if not isinstance(subjects, list):
        subjects = [subjects]

    # Create ECG Component Dictionary
    ecg_dic_keys = [str(i) for i in subjects]  # == list(map(str, subjects))  # these keys also work int
    ecg_dic = {}
    ecg_dic.update((key, dict.fromkeys(roller_coasters, [])) for key in ecg_dic_keys)

    condition_keys = [element.split("_")[1] for element in roller_coasters]  # "NoMov", "Mov"
    condition_keys = list(set(condition_keys))

    t_roller_coasters = update_coaster_lengths(subjects=subjects,
                                               empty_t_array=np.zeros((len(roller_coasters))),
                                               sba=sba)

    if sba:
        for key in ecg_dic_keys:
            ecg_dic[key].update({"SBA": dict.fromkeys(condition_keys, [])})

    for subject in subjects:
        for num, coaster in enumerate(roller_coasters):
            folder = coaster.split("_")[1]  # either "Mov" or "NoMov" (only needed for sba case)
            if sba:
                file_name = path_ecg_sba + "{}/NVR_S{}_SBA_{}.txt".format(folder, str(subject).zfill(2),
                                                                          folder)
                # "NVR_S02_SBA_NoMov.txt"
            else:
                raise ValueError("There are no ECG files here (yet) | sba=False case not implemented")

            if os.path.isfile(file_name):

                # columns = components, rows value per timestep
                hr_vector = np.genfromtxt(file_name, delimiter=";", dtype="float")
                # first row = comp.names
                # hr_vector.shape=(270,) sec

                # Interpolate missing values
                if interpolate:
                    hr_vector = interpolate_nan(arr_with_nan=hr_vector)

                # Fill in ECG Dictionary: ecg_dic
                if not sba:
                    ecg_dic[str(subject)][coaster] = copy.copy(hr_vector)

                else:  # if sba
                    # Split SBA file in Space-Break-Andes
                    if "Space" in coaster:
                        sba_idx_start = 0
                        sba_idx_end = int(t_roller_coasters[0])
                    elif "Break" in coaster:
                        sba_idx_start = int(t_roller_coasters[0])
                        sba_idx_end = int(sum(t_roller_coasters[0:2]))
                    else:  # "Ande" in coaster
                        sba_idx_start = int(sum(t_roller_coasters[0:2]))
                        sba_idx_end = int(sum(t_roller_coasters))
                        # Test lengths
                        if sba_idx_end < hr_vector.shape[0]:
                            print("hr_vector of S{} has not expected size: Diff={} data points.".format(
                                str(subject).zfill(2), hr_vector.shape[0] - sba_idx_end))
                            assert (hr_vector.shape[0] - sba_idx_end) <= 3, \
                                "Difference too big, check files"

                    ecg_dic[str(subject)][coaster] = copy.copy(hr_vector[sba_idx_start:sba_idx_end])

                    # process whole SBA only once per condition (NoMov, Mov)
                    if "Break" in coaster:
                        assert folder in condition_keys, "Wrong key"
                        ecg_dic[str(subject)]["SBA"][folder] = copy.copy(hr_vector)

                # Check times:
                if not sba:
                    if not (np.round(t_roller_coasters[num], 1) == hr_vector.shape[0]):
                        print("Should be all approx. the same:\nTime of {}: {}"
                              "\nLength of hr_vector: {}".format(roller_coasters[num],
                                                                 t_roller_coasters[num],
                                                                 hr_vector.shape[0]))
                else:  # if sba:
                    if not (np.round(sum(t_roller_coasters), 1) == hr_vector.shape[0]):
                        print("Should be all approx. the same:\nTime of SBA: {}"
                              "\nLength of hr_vector: {}".format(sum(t_roller_coasters),
                                                                 hr_vector.shape[0]))

    return ecg_dic


def zero_line_prediction(subject):
    """
    Returns the accuracy if the all predictions would be zero
    :param subject: Subject ID
    :return: accuracy of zero-line prediction
    """
    rating = load_rating_files(subjects=subject)[str(subject)]["SBA"]["NoMov"]
    rating = normalization(array=rating, lower_bound=-1, upper_bound=1)
    zero_line = np.zeros(shape=rating.shape)
    max_diff = 1.0 - (np.abs(rating) * -1.0)  # chances depending on rating-level
    correct = 1.0 - np.abs((zero_line - rating)) / max_diff
    zero_line_accuracy = np.nanmean(correct)

    return zero_line_accuracy


class DataSet(object):
    """
    Utility class (http://wiki.c2.com/?UtilityClasses) to handle dataset structure
    """

    # s_fold_idx_list = []  # this needs to be changed across DataSet-instances
    # s_fold_idx = []  # this needs to be changed across DataSet-instances

    def __init__(self, name, eeg, ratings, subject, condition, task, eeg_samp_freq=250.,
                 rating_samp_freq=1.):
        """
        Builds dataset with EEG data and Ratings
        :param eeg: eeg data, SBA format (space-break-ande) (so far only NoMov)
        :param ratings: rating data, SBA format (space-break-ande)
        :param subject: Subject Nummer
        :param condition: Subject condition
        :param task: whether data is for binary 'classifaction' (low, high) or 'regression'
        :param eeg_samp_freq: sampling frequency of EEG data (default = 250Hz)
        :param rating_samp_freq: sampling frequency of Rating data (default = 1Hz)
        """

        assert eeg.shape[0] == ratings.shape[0], "eeg.shape: {}, ratings.shape: {}".format(eeg.shape,
                                                                                           ratings.shape)

        self.name = name
        self.task = task
        self.eeg_samp_freq = eeg_samp_freq
        self.rating_samp_freq = rating_samp_freq
        # self.remaining_slices = np.arange(self.num_time_slices)  # for randomized drawing of new_batch
        self._eeg = eeg  # input
        self._ratings = ratings  # target
        self._num_time_slices = eeg.shape[0]
        self.remaining_slices = []
        self.reset_remaining_slices()  # creates self.remaining_slices
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
        # return np.arange(self.num_time_slices)
        if self.task == "regression":
            self.remaining_slices = np.arange(self.num_time_slices)
        else:  # == "classification"
            self.remaining_slices = np.delete(arr=np.arange(self.num_time_slices),
                                              obj=np.where(self._ratings == 0))

    def new_epoch(self):
        self._epochs_completed += 1

    def next_batch(self, batch_size=1, randomize=True, successive=1, successive_mode=1):
        """
        Return the next 'batch_size' examples from this data set
        For MODEL 1: the batch size = 1, i.e. input 1sec=250 data points, gives 1 output (rating),
        aka. many-to-one
        :param successive: how many of the random batches shall remain in successive order
        :param successive_mode: Mode 1) batches start always at the same spot (-): perfect line-up (+)
                                Mode 2) batches can start at random spots, no perfect line-up guaranteed
        :param batch_size: Batch size
        :param randomize: Whether to randomize the order in data
        :return: Next batch
        """

        # if batch_size > 1:
        #     raise ValueError("A batch_size of > 1 is not recommanded at this point")

        assert batch_size % successive == 0, "batch_size must be a multiple of successive"
        assert successive_mode in [1, 2], "successive_mode must be either 1 or 2"

        if len(self.remaining_slices) >= batch_size:

            # Select slize according to number of batches
            if randomize:
                if successive == 1:
                    selection_array_idx = np.random.choice(a=range(len(self.remaining_slices)),
                                                           size=batch_size, replace=False)
                else:

                    selection_array_idx = np.array([])

                    if successive_mode == 1:
                        batch_counter = batch_size

                        while batch_counter > 0:

                            while True:
                                sel_arr_idx = np.random.choice(a=range(len(self.remaining_slices)),
                                                               size=1, replace=False)

                                # Search around index if remaining_slides are not equally distributed
                                while len(self.remaining_slices[:sel_arr_idx[0]]) % successive > 0:
                                    if sel_arr_idx < len(self.remaining_slices)-1:
                                        if sel_arr_idx > 0:
                                            sel_arr_idx -= np.random.choice(a=[-1, 1], size=1)
                                        else:
                                            sel_arr_idx += 1
                                    else:
                                        sel_arr_idx -= 1

                                if sel_arr_idx not in selection_array_idx:
                                    break

                            # Check whether right number of remaining batches
                            assert len(self.remaining_slices[sel_arr_idx[0]:]) % successive == 0, \
                                "successive must devide length of dataset equally"

                            # Attach successive batches
                            selection_array_idx = np.append(selection_array_idx, sel_arr_idx).astype(int)
                            for _ in range(successive-1):
                                selection_array_idx = np.append(selection_array_idx,
                                                                selection_array_idx[-1]+1)

                            batch_counter -= successive

                    else:  # successive_mode == 2:

                        while True:
                            sel_arr_idx = np.random.choice(a=range(len(self.remaining_slices)),
                                                           size=int(batch_size/successive), replace=False)

                            # Check whether distances big enough
                            counter = 0
                            for element in sel_arr_idx:
                                diff = np.abs(element - sel_arr_idx)
                                diff = np.delete(diff, np.where(diff == 0))
                                if not np.all(diff >= successive):
                                    counter += 1
                            if counter == 0 and np.max(sel_arr_idx)+successive-1 < len(
                                    self.remaining_slices):
                                break

                        for element in sel_arr_idx:
                            # Attach successive batches
                            selection_array_idx = np.append(selection_array_idx, element).astype(int)
                            for _ in range(successive - 1):
                                selection_array_idx = np.append(selection_array_idx,
                                                                selection_array_idx[-1] + 1)

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
                # self.remaining_slices = self.reset_remaining_slices()
                self.reset_remaining_slices()

                # print("\nNo slices left take new slices")  # test

        else:
            # First copy remaining slices to selection_array
            selection_array = self.remaining_slices

            # Now reset, i.e. new epoch
            self.new_epoch()  # self._epochs_completed += 1
            remaining_slices_to_draw = batch_size - len(self.remaining_slices)
            self._index_in_epoch = remaining_slices_to_draw  # index in new epoch
            # self.remaining_slices = self.reset_remaining_slices()  # reset
            self.reset_remaining_slices()  # reset

            # Draw remaining slices from new epoch
            if randomize:

                if successive == 1:

                    additional_array_idx = np.random.choice(a=range(len(self.remaining_slices)),
                                                            size=remaining_slices_to_draw, replace=False)

                else:
                    additional_array_idx = np.array([])

                    if successive_mode == 1:
                        batch_counter = remaining_slices_to_draw

                        while batch_counter > 0:

                            sel_arr_idx = np.random.choice(a=range(len(self.remaining_slices)),
                                                           size=1, replace=False)

                            # Search around index if remaining_slides are not equally distributed
                            while len(self.remaining_slices[:sel_arr_idx[0]]) % successive > 0:
                                if sel_arr_idx < len(self.remaining_slices)-1:
                                    if sel_arr_idx > 0:
                                        sel_arr_idx -= np.random.choice(a=[-1, 1], size=1)
                                    else:
                                        sel_arr_idx += 1
                                else:
                                    sel_arr_idx -= 1

                            # Check whether right number of remaining batches
                            assert len(self.remaining_slices[sel_arr_idx[0]:]) % successive == 0, \
                                "successive must devide length of dataset equally"

                            # Attach successive batches
                            additional_array_idx = np.append(
                                additional_array_idx, sel_arr_idx).astype(int)
                            for _ in range(successive-1):
                                additional_array_idx = np.append(additional_array_idx,
                                                                 additional_array_idx[-1]+1)

                            batch_counter -= successive

                    else:  # successive_mode == 2:
                        while True:
                            sel_arr_idx = np.random.choice(a=range(len(self.remaining_slices)),
                                                           size=int(remaining_slices_to_draw/successive),
                                                           replace=False)

                            # Check whether distances big enough
                            counter = 0
                            for element in sel_arr_idx:
                                diff = np.abs(element - sel_arr_idx)
                                diff = np.delete(diff, np.where(diff == 0))
                                if not np.all(diff >= successive):
                                    counter += 1
                            if counter == 0 and np.max(sel_arr_idx)+successive-1 < len(
                                    self.remaining_slices):
                                break

                        for element in sel_arr_idx:
                            # Attach successive batches
                            additional_array_idx = np.append(additional_array_idx, element).astype(int)
                            for _ in range(successive - 1):
                                additional_array_idx = np.append(additional_array_idx,
                                                                 additional_array_idx[-1] + 1)

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

        if len(np.unique(selection_array)) != batch_size:  # TESTING WHETHER UNUSUAL BATCH PROPERTIES
            print("selection_array (len={}, unique_len={})".format(len(selection_array),
                                                                   len(np.unique(selection_array))),
                  selection_array)

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


def read_data_sets(subject, component, hr_component, s_fold_idx, s_fold=10, cond="NoMov", sba=True,
                   s_freq_eeg=250.,
                   filetype="SSD", band_pass=True, hilbert_power=True, task="regression", shuffle=False,
                   testmode=False):
    """
    Returns the s-fold-prepared dataset.
    S-Fold Validation, S=5: [ |-Train-|-Train-|-Train-|-Valid-|-Train-|] Dataset
    Args:
        subject: Subject Nr., subject dataset for training
        component: which component to feed (int, list(int))
        hr_component: Whether to attach hear rate component to neural components (True/False)
        s_fold_idx: index which of the folds is taken as validation set
        s_fold: s-value of S-Fold Validation [default=10]
        cond: Either "NoMov"(=default) or "Mov"
        sba: Whether to use SBA-data
        s_freq_eeg: Sampling Frequency of EEG
        filetype: Whether 'SSD' or 'SPOC'
        band_pass: Whether SSD components are band-passed filter for alpha
        hilbert_power: hilbert-transform SSD-components, then extract z-scored power
        task: either regression or (binary)-classification
        shuffle: shuffle data (primarily for classification task to have balance low/high arousal in all folds/valsets)
        testmode: Whether to load data for testmode
    Returns:
        Train, Validation Datasets
    """

    # Check inputs
    assert s_fold_idx < s_fold, \
        "s_fold_idx (={}) must be in the range of the number of folds (={})".format(s_fold_idx, s_fold)
    assert cond in ["NoMov", "Mov"], "cond must be either 'NoMov' or 'Mov'"

    assert filetype.upper() in ["SSD", "SPOC"], "filetype must be either 'SSD' or 'SPOC'"
    task = task.lower()
    assert task in ["regression", "classification"], "task must be 'regression' or 'classification'"

    # If int, transform to list
    if not type(component) is list:
        component = [component]

    for comp_idx, comp in enumerate(component):
        if filetype.upper() == "SSD":
            if band_pass:
                assert comp in range(1, 5+1) or comp in range(91, 95+1), \
                    "Components must be in range [1,2,3,4,5]"
            else:  # not band_pass
                pass
        else:  # == 'SPOC'
            assert comp in range(1, 9 + 1) or comp in range(91, 99 + 1), \
                "Components must be in correct range"
        # Check demand for noise component
        if comp in range(91, 95+1):
            component[comp_idx] -= 90   # decode
            noise_comp = True  # switch on noise-mode of given component
        else:
            try:
                if noise_comp:
                    raise ValueError("If one, then all given components must be noise components")
            except NameError:
                noise_comp = False

    if filetype.upper() == "SSD":
        eeg_data = load_ssd_component(subjects=subject, band_pass=band_pass)
    else:  # == 'SPOC'
        eeg_data = load_spoc_component(subjects=subject, band_pass=band_pass)

    rating_data = load_rating_files(subjects=subject, bins=True if task == "classification" else False)

    if hr_component:
        ecg_data = load_ecg_files(subjects=subject, sba=sba)  # interpolation as default

    # TODO Load for more than one subject
    condition = rating_data[str(subject)]["condition"]
    t_roller_coasters = update_coaster_lengths(subjects=subject,
                                               empty_t_array=np.zeros((len(roller_coasters))),
                                               sba=sba)

    # Subsample the validation set from the train set
    # 0) Take Space-Break-Ande (SBA) files from dictionaries

    # eeg_concat = concatenate(array1=eeg_data[str(subject)]["Space_NoMov"][:, 0:2],
    #                          array2=eeg_data[str(subject)]["Ande_NoMov"][:, 0:2])
    # rating_concat = concatenate(array1=rating_data[str(subject)]["Space_NoMov"],
    #                             array2=rating_data[str(subject)]["Ande_NoMov"])

    # Best component selected based on highest xcorr with ratings
    # Comparison with worst xcorr
    # eeg_sba = eeg_data[str(subject)]["SBA"][cond][:, 0:2]  # = first 2 components

    eeg_sba = eeg_data[str(subject)]["SBA"][cond][:, [comp-1 for comp in component]]

    # If Noise-Mode, shuffle component
    if noise_comp:
        np.random.shuffle(eeg_sba)

    rating_sba = rating_data[str(subject)]["SBA"][cond]

    if hr_component:
        ecg_sba = ecg_data[str(subject)]["SBA"][cond]

    # Check whether EEG data too long
    if sba:

        len_test = eeg_sba.shape[0] / s_freq_eeg - rating_sba.shape[0]
        if len_test > 0.0:
            to_delete = np.cumsum(t_roller_coasters)  # [ 148.,  178.,  270.]
            # 3 intersections where values can be removed (instead of cutting only at the end/start)
            to_delete *= s_freq_eeg
            to_cut = int(round(len_test * s_freq_eeg))
            print("EEG data of S{} trimmed by {} data points".format(str(subject).zfill(2), to_cut))
            del_counter = 2
            while len_test > 0.0:
                if del_counter == -1:
                    del_counter = 2
                eeg_sba = np.delete(arr=eeg_sba, obj=to_delete[del_counter], axis=0)
                # starts to delete in the end
                del_counter -= 1
                len_test = eeg_sba.shape[0] / s_freq_eeg - rating_sba.shape[0]
        elif len_test < 0.0:
            raise OverflowError("Eeg_sba file is too short. Implement interpolation function.")

    # Normalize rating_sba to [-1;1] due to tanh-output of NeVRoNet
    rating_sba = normalization(array=rating_sba, lower_bound=-1, upper_bound=1)

    # IF, then attach HR to neural components
    if hr_component:
        ecg_sba_streched = np.reshape(np.repeat(a=ecg_sba, repeats=int(s_freq_eeg)), newshape=[-1, 1])
        # HR is in 1Hz
        eeg_sba = np.concatenate((eeg_sba, ecg_sba_streched), 1)

    if hilbert_power:
        # print("I load Hilbert transformed data (z-power)")
        # Perform Hilbert Transformation
        for comp in range(eeg_sba.shape[1]):
            eeg_sba[:, comp] = calc_hilbert_z_power(array=eeg_sba[:, comp])

    # If Testset, overwrite eeg_sba data with artifical data (for model testing)
    if testmode:
        # # 1) negative sin(ratings), then stretch
        # eeg_sba = np.reshape(a=np.repeat(a=-np.sin(rating_sba), repeats=250, axis=0),
        #                      newshape=eeg_sba.shape)
        # # 2) negative sin(ratings)**3, then stretch
        # eeg_sba = np.reshape(a=np.repeat(a=-np.sin(rating_sba**3), repeats=250, axis=0),
        #                      newshape=eeg_sba.shape)
        # # 3) test with global slope
        # slope = np.linspace(0, 0.2, len(rating_sba)) + np.random.normal(loc=0., scale=0.01,
        #                                                                 size=len(rating_sba))
        slope = np.linspace(0, 0.3, len(rating_sba)) + np.random.normal(loc=0., scale=0.02,
                                                                        size=len(rating_sba))
        # eeg_sba = np.reshape(a=np.repeat(a=-np.sin(rating_sba+slope), repeats=250, axis=0),
        #                      newshape=eeg_sba.shape)
        # 4) with-in-1-second slope (local slope) 5) stronger slope
        eeg_sba = np.reshape(a=np.repeat(a=-rating_sba, repeats=250, axis=0), newshape=eeg_sba.shape)
        eeg_sba += np.reshape(a=np.repeat(a=slope, repeats=250, axis=0), newshape=eeg_sba.shape)
        # Get it out of [-1,1]-range
        eeg_sba *= 3
        # 6) inverse [1,2,3] -> [3,2,1]
        # eeg_sba = eeg_sba[::-1]

        # add some random noise ε (scale=0.05 is relatively large)
        eeg_sba += np.random.normal(loc=0., scale=0.05, size=eeg_sba.shape[0]).reshape(eeg_sba.shape)
        print("In test mode: Input data: (-3*ratings + strong_slope) + noise ε")

    # 1) Split data in S(=s_fold) sets
    # np.split(np.array([1,2,3,4,5,6]), 3) >> [array([1, 2]), array([3, 4]), array([5, 6])]

    # split EEG data w.r.t. to total sba-length
    eeg_sba_split = splitter(eeg_sba, n_splits=int(sum(t_roller_coasters)))
    # [sec, data-points, components)

    # If semi-balanced low-high-arousal values for valid. set is required (in binary classi.) then do:
    shuf_idx = np.arange(len(rating_sba))  # here still no change of data rating_sba[shuf_idx]==rating_sba
    if shuffle:
        np.random.shuffle(shuf_idx)
        if task == "regression":
            print("Shuffling data for regression task leads to more difficult interpretation of "
                  "results/plots and makes successive batches redundant (if applied).")

    # eeg_concat_split[0][0:] first to  250th value in time
    # eeg_concat_split[1][0:] 250...500th value in time
    rating_split = splitter(array_to_split=rating_sba[shuf_idx], n_splits=s_fold)
    eeg_split = splitter(array_to_split=eeg_sba_split[shuf_idx], n_splits=s_fold)

    # eeg_concat_split.shape    # (n_chunks[in 1sec], n_samples_per_chunk [250Hz], channels)
    # eeg_split.shape           # (s_fold, n_chunks_per_fold, n_samples_per_chunk, channels)
    # rating_split.shape        # (s_fold, n_samples_per_fold,)

    # Assign variables accordingly:
    validation_eeg = eeg_split[s_fold_idx]
    validation_ratings = rating_split[s_fold_idx]
    train_eeg = np.delete(arr=eeg_split, obj=s_fold_idx, axis=0)
    # removes the val-set from the data set (np.delete)
    train_ratings = np.delete(arr=rating_split, obj=s_fold_idx, axis=0)
    # Merge the training sets again (concatenate for 2D & vstack for >=3D)
    # Cautious:Assumption that performance is partly independent of correct order ( done already with SBA)
    train_eeg = np.vstack(train_eeg)  # == np.concatenate(train_eeg, axis=0)
    train_ratings = np.concatenate(train_ratings, axis=0)

    # Create datasets
    train = DataSet(name="Training", eeg=train_eeg, ratings=train_ratings, subject=subject,
                    condition=condition, task=task)
    validation = DataSet(name="Validation", eeg=validation_eeg, ratings=validation_ratings,
                         subject=subject, condition=condition, task=task)
    # Test set
    # test = DataSet(eeg=test_eeg, ratings=test_ratings, subject=subject, condition=condition, task=task)
    test = None

    # return base.Datasets(train=train, validation=validation, test=test), s_fold_idx
    return {"train": train, "validation": validation, "test": test, "order": shuf_idx}


def get_nevro_data(subject, component, hr_component, s_fold_idx=None, s_fold=10, cond="NoMov", sba=True,
                   s_freq_eeg=250., filetype="SSD", band_pass=True, hilbert_power=True, task="regression",
                   shuffle=False, testmode=False):
    """
      Prepares NeVRo dataset.
      Args:
        subject: Which subject data to train
        component: Which component to feed, can be list
        hr_component: Whether to attach hear rate component to neural components (True/False)
        s_fold_idx: index which of the folds is taken as validation set
        s_fold: s-value of S-Fold Validation [default=10]
        cond: Either "NoMov"(=default) or "Mov"
        sba: Whether to use SBA-data
        s_freq_eeg: Sampling Frequency of EEG
        filetype: Whether 'SSD' or 'SPOC'
        band_pass: Whether SSD components are band-passed filter for alpha
        hilbert_power: hilbert-transform SSD-components, then extract z-scored power
        task: either regression or (binary)-classification
        shuffle: shuffle data (primarily for classif. task to have balance low/high arousal in all folds)
        testmode: Whether to load data for testmode
      Returns:
        Train, Validation Datasets (+Test set)
      """
    if s_fold_idx is None:
        s_fold_idx = np.random.randint(low=0, high=s_fold)
        print("s_fold_idx randomly chosen:", s_fold_idx)

    return read_data_sets(subject=subject, component=component, hr_component=hr_component,
                          s_fold_idx=s_fold_idx, s_fold=s_fold,
                          cond=cond, sba=sba, s_freq_eeg=s_freq_eeg,
                          filetype=filetype, band_pass=band_pass, hilbert_power=hilbert_power,
                          task=task, shuffle=shuffle,
                          testmode=testmode)


# # Testing
# nevro_data = get_nevro_data(subject=36, component=5, s_fold_idx=9, s_fold=10, cond="NoMov",
#                             hilbert_power=True)
# nevro_data = get_nevro_data(subject=44, component=4, s_fold_idx=9, s_fold=10, cond="NoMov",
#                             hilbert_power=True)

# for _ in range(27):
#     x = nevro_data["validation"].next_batch(batch_size=4, randomize=True)
#     print("Current Batch:", nevro_data["validation"].current_batch)
#     print("n_remaining slices: {}/{}".format(len(nevro_data["validation"].remaining_slices),
#                                              nevro_data["validation"]._num_time_slices))
#     print("index in current epoch:", nevro_data["validation"]._index_in_epoch)
#     print("epochs copmleted:", nevro_data["validation"]._epochs_completed)
#     print("")

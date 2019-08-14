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

Author: Simon Hofmann | <[surname].[lastname][at]pm.me> | 2017, 2019 (Update)
"""

# import sys
# sys.path.insert(0, './LSTM Model')  # or set the folder as source root
import copy
from meta_functions import *
import pandas as pd

# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

#  load EEG data of from *.set (EEGlab format) with: mne.io.read_raw_eeglab() and mne.read_epochs_eeglab()

fresh_prep = True  # with refreshed preprocessed data is to be used

n_sub = 45

# # # Define root abd data folders

setwd("/Analysis/Modelling/LSTM/")

path_data = set_path2data()

path_ssd = path_data + "EEG/07_SSD/"
path_spoc = path_data + "EEG/08.1_SPOC/"  # TODO not there yet
# # SBA-pruned data (148:space, 30:break, 92:andes)

# # Rating data
# path_rating = path_data + "ratings/preprocessed/z_scored_alltog/"
path_rating = path_data + "ratings/"
path_rating_cont = path_rating + "continuous/not_z_scored/"  # min-max scale later to [-1, +1]
path_rating_bins = path_rating + "class_bins/"
# # ECG data
path_ecg_crop = path_data + "Data EEG export/NeVRo_ECG_Cropped/"
path_ecg_sba = path_data + "ECG/SBA/z_scored_alltog/"

path_results_xcorr = "../../../Results/x_corr/"

# # # Initialize variables
# subjects = [36]  # [36, 37]
# subjects = range(1, 45+1)
# dropouts = [1,12,32,33,38,40,45]

# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>


def t_roller_coasters(sba=True):
    """
    :return: array with lengths of phases for SBA or SA
    """
    return np.array([148., 30., 92]) if sba else np.array([148., 92])


def roller_coasters(cond, sba=True):
    """
    :return: list of VR roller coaster names:
                if SBA:
                    ['Space_Mov', 'Break_Mov', 'Ande_Mov']
                else SA:
                    ['Space_Mov', 'Ande_Mov']
    """
    sfx = "_Mov" if cond.lower() == "mov" else "_NoMov"  # suffix

    rl = np.array(['Space' + sfx, 'Break' + sfx, 'Ande' + sfx])

    return rl if sba else rl[[0, 2]]


def check_condition(cond):
    assert cond.lower() in ["nomov", "mov"], "cond must be either 'nomov' or 'mov'"
    return cond.lower()


def get_filename(subject, filetype, band_pass, cond="nomov", sba=True, check_existence=False):
    """
    Receive the filename for specific setup.
    :param subject: subject number
    :param filetype: Either 'SSD' or 'SPOC'
    :param band_pass: Data either band-pass filtered (True) or not. 'SPOC' always band-pass filtered
    :param cond: 'nomov' [Default] or 'mov'
    :param sba: SBA data
    :param check_existence: True: raise Error if file does not exist
    :return: filename
    """

    cond = check_condition(cond=cond)

    assert filetype.upper() in ["SSD", "SPOC"], "filetype must be either 'SSD' or 'SPOC'"

    if not sba:
        raise NotImplementedError("For SA data not implemented yet.")

    get_path = path_ssd if filetype.upper() == "SSD" else path_spoc

    if filetype.upper() == "SPOC":
        band_pass = True  # 'SPOC' always band-pass filtered

    band_pass_str = "narrowband" if band_pass else "broadband"

    get_path += cond.lower() + "/"  # os.path.join("a", "b") -> "a/b"
    get_path += "SBA/" if sba else "SA/"
    get_path += band_pass_str + "/"

    # e.g. "NVR_S05nomov_PREP_SBA_eventsaro_rejcomp_SSD_broadband.csv"
    file_name = get_path + "NVR_{}{}_PREP_{}_eventsaro_rejcomp_{}_{}.csv".format(s(subject),
                                                                                 cond,
                                                                                 "SBA" if sba else "SA",
                                                                                 filetype.upper(),
                                                                                 band_pass_str)
    # Check existence of file
    if check_existence:
        if not os.path.exists(file_name):
            raise FileExistsError(file_name, "does not exisit")

    return file_name


# @function_timed  # after executing following function this returns runtime
def get_num_components(subject, condition, filetype, selected=True):
    """
    Get number of components for subject in given condition. Information should have been saved in
    corresponding table. If not: Create this table for all subjects.

    :param subject: subject number
    :param condition: mov (1), or nomov(2)
    :param filetype: 'SSD' or 'SPOC'
    :param selected: True: selected components
    :return: number of components or nan (if no information)
    """

    condition = check_condition(cond=condition)
    filetype = filetype.upper()
    assert filetype in ["SSD", "SPOC"], "filetype must be either 'SSD' or 'SPOC'"

    # Path to table of number of components
    path_base = path_ssd if filetype == "SSD" else path_spoc
    fname_tab_ncomp = path_base + f"{condition}/{filetype}_selected_components_{condition}.csv"

    # If table does not exist, create it
    if not os.path.isfile(fname_tab_ncomp):
        raise FileNotFoundError(f"Component information in {fname_tab_ncomp} not found.\n"
                                "Execute selection process first with SelectSSDcomponents().")

    else:  # Load table if exsits already
        tab_ncomp = pd.read_csv(fname_tab_ncomp, sep=";")
        # tab_ncomp = np.genfromtxt(fname_tab_ncomp, delimiter=";", dtype=str)

    # Get number of components in condition from table
    ncomp = tab_ncomp[tab_ncomp["# ID"] == subject]["n_sel_comps" if selected else "n_all_comps"].item()
    ncomp = int(ncomp) if ncomp != "nan" else np.nan

    return ncomp

# get_num_components(subject=24, condition="mov", filetype="SSD")


def get_list_components(subject, condition, filetype, selected=True, lstype="list"):
    """
    Get list of components for subject in given condition. Information should have been saved in
    corresponding table. If not: Create this table for all subjects.

    :param subject: subject number
    :param condition: mov (1), or nomov(2)
    :param filetype: 'SSD' or 'SPOC'
    :param selected: True: selected components
    :param lstype: 'list': [1,2,3,...] or 'string' '1,2,3,...'
    :return: list of components or nan (if no information)
    """

    condition = check_condition(cond=condition)
    filetype = filetype.upper()
    assert filetype in ["SSD", "SPOC"], "filetype must be either 'SSD' or 'SPOC'"

    # Path to table of number of components
    path_base = path_ssd if filetype == "SSD" else path_spoc
    fname_tab_ncomp = path_base + f"{condition}/{filetype}_selected_components_{condition}.csv"

    # If table does not exist, create it
    if not os.path.isfile(fname_tab_ncomp):
        raise FileNotFoundError(f"Component information in {fname_tab_ncomp} not found.\n"
                                "Execute selection process first with SelectSSDcomponents().")

    else:  # Load table if exsits already
        tab_ncomp = pd.read_csv(fname_tab_ncomp, sep=";")
        # tab_ncomp = np.genfromtxt(fname_tab_ncomp, delimiter=";", dtype=str)

    # Get list of components in condition from table
    if selected:
        comp_ls = tab_ncomp[tab_ncomp["# ID"] == subject]["selected_comps"].item()
        if pd.isna(comp_ls):
            comp_ls = np.nan
        elif lstype == "list":
            comp_ls = [int(i) for i in comp_ls.split(",")]
    else:
        n_all_comps = tab_ncomp[tab_ncomp["# ID"] == subject]["n_all_comps"].item()
        n_all_comps = np.nan if pd.isna(n_all_comps) else int(n_all_comps)
        if not np.isnan(n_all_comps):
            comp_ls = list(range(1, n_all_comps+1))
            if lstype == "string":
                comp_ls = ",".join(str(i) for i in comp_ls)   # '1,2,3,...'
        else:
            comp_ls = np.nan

    return comp_ls


def load_component(subjects, condition, f_type, band_pass, samp_freq=250., sba=True):
    """
    Load components files (SSD, SPOC) and prepare them in dictionary

    :param subjects: list of subjects or single subject
    :param condition: 'nomov' or 'mov' condition
    :param f_type: either "SSD" or "SPOC"
    :param band_pass: Whether components are band-passed filter around alpha (SPOC normally is)
    :param samp_freq: sampling frequency of components
    :param sba: if True (=Default), take SBA-z-scored components
    Loads components of each subject in subjects.
    Note: Components are centred around zero
    :return: component files [sub_df] in form of dictionary
    """

    if not isinstance(subjects, list):
        subjects = [subjects]
    f_type = f_type.upper()
    assert f_type in ["SSD", "SPOC"], "f_type must be 'SSD' or 'SPOC'"

    # Create Component Dictionary
    comp_dic_keys = [str(i) for i in subjects]
    # == list(map(str, subjects))  # these keys also work int
    comp_dic = {}
    comp_dic.update((key, dict.fromkeys(roller_coasters(condition, sba), [])) for key in comp_dic_keys)

    condition_keys = [condition]  # nomov", "mov"

    for key in comp_dic_keys:
        comp_dic[key].update({"SBA" if sba else "SA": dict.fromkeys(condition_keys, [])})

    for subject in subjects:

        file_name = get_filename(subject=subject, filetype=f_type, band_pass=band_pass,
                                 cond=condition, sba=sba, check_existence=True)
        # "NVR_S36_1_SSD_nonfilt_cmp.csv"

        if os.path.isfile(file_name):

            # N components sorted
            #   SSD:  from 1-N according to the signal-to-noise-ration (SNR)
            #   SPOC: from 1-N according to comodulation between component and target

            # rows = components, columns value per timestep
            # first column: Nr. of component, last column is empty
            sub_df = np.genfromtxt(file_name, delimiter=",")[:, 1:-1].transpose()
            # sub_df.shape=(67503, 6=N-comps)  # for S36 with samp_freq=250: 67503/250. = 270.012 sec

            # Save whole SBA/SA under condition (nomov, mov)
            comp_dic[str(subject)]["SBA" if sba else "SA"][condition] = copy.copy(sub_df)

            for coaster in roller_coasters(condition, sba):

                # Fill in Dictionary: comp_dic
                # Split SBA file in Space-Break-Andes
                if "Space" in coaster:
                    sba_idx_start = 0
                    sba_idx_end = int(t_roller_coasters(sba)[0] * samp_freq)
                elif "Break" in coaster:
                    sba_idx_start = int(t_roller_coasters(sba)[0] * samp_freq)
                    sba_idx_end = int(sum(t_roller_coasters(sba)[0:2]) * samp_freq)
                else:  # "Ande" in coaster
                    sba_idx_start = int(sum(t_roller_coasters(sba)[0:2]) * samp_freq)
                    sba_idx_end = int(sum(t_roller_coasters(sba)) * samp_freq)
                    # Test lengths
                    if sba_idx_end < sub_df.shape[0]:
                        print("sub_df of S{} has not expected size: Difference is {} data points.".format(
                            str(subject).zfill(2), sub_df.shape[0] - sba_idx_end))
                        assert (sub_df.shape[0] - sba_idx_end) <= 3, "Difference too big, check files"

                comp_dic[str(subject)][coaster] = copy.copy(sub_df[sba_idx_start:sba_idx_end, :])

                # Check times:
                # samp_freq = 250.
                if not (np.round(sum(t_roller_coasters(sba)), 1) == np.round(sub_df.shape[0] / samp_freq,
                                                                             1)):
                    print(f"Should be all approx. the same:\nTime of SBA: {sum(t_roller_coasters(sba))}\n"
                          f"Length of sub_df/samp_freq({int(samp_freq)}): {sub_df.shape[0] / samp_freq}")

    return comp_dic


def best_or_random_component(subject, condition, f_type, best, sba=True):
    """
    Choose the SSD-alpha-component, either randomly or the best
    :param subject: subject number
    :param condition: nomov or mov
    :param f_type: 'SSD' or 'SPOC'
    :param best: False: take random. If True: Choose w.r.t. x-corr with ratings from list for given subj.
    :param sba: True (SBA) or False (SA)
    :return: component number
    """

    f_type = f_type.upper()

    # TODO table need to be updated
    if fresh_prep and best and f_type == "SSD":
        raise ImportError("Xcorr table needs to be updated")
    if condition == "mov" and f_type == "SSD":
        raise ImportError("Xcorr table has no entries for mov-condition")

    # Load Table
    x_corr_table = pd.read_csv(path_results_xcorr + "CC_AllSubj_Alpha_Ratings_smooth.csv",
                               index_col=0) if f_type == "SSD" else None
    # Drop non-used columns
    if x_corr_table is not None:
        x_corr_table = x_corr_table.drop(x_corr_table.columns[0:3], axis=1)  # first col is idx
    # Find component (choose best for now)
    component = best_comp = x_corr_table.loc["{}".format(s(subject))].values[0] if f_type == "SSD" else 1

    if best:
        print("Best correlating {} component of S{} is Component number: {}".format(f_type,
                                                                                    s(subject),
                                                                                    component))
    else:
        # If not best: choose another component that is not the best
        file_name = get_filename(subject=subject, filetype=f_type,
                                 band_pass=True,  # for False the same
                                 cond=condition, sba=sba, check_existence=True)

        n_comp = len(np.genfromtxt(file_name, delimiter=",")[:, 0])  # each row is one component

        while component == best_comp:
            component = np.random.randint(low=1, high=n_comp + 1)  # choose random component != best

        print("A random {} component of {} is chosen (which is not the best one): Component {}".format(
            f_type, s(subject), component))

    return component


def get_group_affiliation(subject):
    """
    Subjects were either in Group '12' or Group '21', where '1' stands for the movement condition and '2'
    for the non-movement condition.

    :param subject: subject number
    :returns: group affiliation
    """

    subject = int(subject)

    # Load Table of Conditions
    table_of_condition = np.genfromtxt(path_data + "Table_of_Conditions.csv", delimiter=";",
                                       skip_header=True, dtype=float)
    # remove first column (sub-nr, condition-order, gender (1=f, 2=m))

    # Get group affiliation of subject
    group = int(table_of_condition[np.where(table_of_condition[:, 0] == subject), 1].item())

    return group


def get_run(subject, condition, group=None):
    """Subjects were either in Group '12' or Group '21', where '1' stands for the movement condition and
    '2' for the non-movement condition. Consequently, subjects in condition '1' of Group '21' are in their
    their second run, i.e. run '2', and vice versa.
    :param subject: subject number
    :param condition: either 'mov' or 'nomov'
    :param group: can be given as (int) 12 or 21
    :returns: run (as string)
    """

    subject = int(subject)

    if group is None:
        group = get_group_affiliation(subject=subject)
    else:
        assert group in [12, 21], "Group must be either 12 or 21 (int), not {}".format(group)

    # Get run of subject in given condition
    if (condition == "mov" and group == 12) or (condition == "nomov" and group == 21):
        run = str(1)
    else:  # (condition == "nomov" and group == 12) or (condition == "mov" and group == 21):
        run = str(2)

    return run


# @function_timed
def load_rating_files(subjects, condition, sba=True, bins=False, samp_freq=1.):
    """
    Loads rating files of each subject in subjects (and take z-score later)
    (ignore other files, due to fluctuating samp.freq.)
    :param subjects: list of subjects or single subject
    :param condition: nomov or mov
    :param sba: if TRUE (default), process SBA files
    :param bins: whether to load ratings in forms of bins (low, medium, high arousal)
    :param samp_freq: sampling frequency, either 1Hz [default], oder 50Hz (no fully implemented for 50Hz)
    :return:  Rating Dict
    """

    if not isinstance(subjects, list):
        subjects = [subjects]

    # Check whether input correct
    # assert (samp_freq == 1 or samp_freq == 50), "samp_freq must be either 1 or 50 Hz"
    if not (samp_freq == 1. or samp_freq == 50.):
        raise ValueError("samp_freq must be either 1 or 50 Hz")

    if samp_freq == 50.:
        cprint("Implementation Warning: No loading of 50Hz Ratings possible yet.", "y")

    # Create Rating-dictionary
    rating_dic_keys = list(map(str, subjects))  # == [str(i) for i in subjects]
    rating_dic = {}
    # each subejct has a sub-dic for each roller coaster
    rating_dic.update((key, dict.fromkeys(roller_coasters(condition, sba),
                                          [])) for key in rating_dic_keys)

    condition_keys = [condition]  # "nomov", "mov"

    # For each subject fill condtition in
    for key in rating_dic_keys:  # key:= subject ID
        rating_dic[key].update({"condition_order": get_group_affiliation(key)})
        # 12: mov-nomov | 21: nomov-mov

        rating_dic[key].update({"SBA" if sba else "SA": dict.fromkeys(condition_keys, [])})

    for subject in subjects:
        for num, coaster in enumerate(roller_coasters(condition, sba=sba)):

            # Adapt file_name accordingly (run <=> condition,
            if "Space" in coaster:
                coast = "space"
            elif "Ande" in coaster:
                coast = "andes"
            else:  # Break
                coast = "break"

            runs = get_run(subject=subject, condition=condition,
                           group=rating_dic[str(subject)]["condition_order"])

            coast_folder = "nomov" if "NoMov" in coaster else "mov"

            rating_filename = path_rating_bins if bins else path_rating_cont

            if not bins:
                rating_filename = rating_filename + \
                                  f"{coast_folder}/{coast}/NVR_{s(subject)}_run_{runs}_{coast}_rat_z.txt"

                if os.path.isfile(rating_filename):

                    # # Load according file
                    rating_file = np.genfromtxt(rating_filename, delimiter=',')[:, 1]
                    # only load column with ratings

                    # Fill in rating dictionary
                    rating_dic[str(subject)][coaster] = copy.copy(rating_file)

                    # Check times:
                    if not (np.round(t_roller_coasters(sba)[num],
                                     1) == np.round(len(rating_file) / samp_freq, 1)):
                        print("Should be all approx. the same:"
                              "\nTime of {}: {}"
                              "\nLength of Rating / s_freq({}): {}".format(roller_coasters(condition,
                                                                                           sba)[num],
                                                                           t_roller_coasters(sba)[num],
                                                                           samp_freq,
                                                                           len(rating_file) / samp_freq))
            # else:
            #     print("So far, no need for rating bins for different epochs (Space, Break, Andes).")

        if not sba:  # == if SA
            raise NotImplementedError("SA case not implemented yet.")

        if bins:
            rat_fname = path_rating_bins + "{}/{}/NVR_S{}_run_{}_alltog_epochs.txt" \
                .format(condition, "SBA" if sba else "SA", str(subject).zfill(2), runs)

        else:  # continuous
            rat_fname = path_rating_cont + \
                            "{}/{}/NVR_S{}_run_{}_alltog_rat_z.txt".format(
                                condition, "SBA" if sba else "SA",
                                str(subject).zfill(2), runs)

        if not os.path.isfile(rat_fname):
            cprint("No rating file found for S{}:\t{}".format(str(subject).zfill(2), rat_fname), col="r")

        else:
            # Load according file
            curr_rating_file = np.genfromtxt(rat_fname, delimiter=',',
                                             skip_header=True if bins else False)[:, 1]
            # in case of bin-files delete 1.entry (header)

            if bins:
                # substract 2 to adapt range to [-1,1]
                curr_rating_file -= 2
                # -1: low, 0: mid, 1: high arousal

                # print("Rating bins and count:", np.unique(curr_rating_file,
                #                                           return_counts=True))

            # Fill in rating dictionary
            rating_dic[str(subject)]["SBA" if sba else "SA"][condition] = copy.copy(
                curr_rating_file)

            # Check times:
            if not (np.round(sum(t_roller_coasters(sba)), 1) == np.round(
                    len(curr_rating_file) / samp_freq, 1)):
                print("Should be all approx. the same:"
                      "\nTime of {}: {}"
                      "\nLength of Rating/s_freq({}Hz): {}".format("SBA" if sba else "SA",
                                                                   sum(t_roller_coasters(sba)),
                                                                   int(samp_freq),
                                                                   len(curr_rating_file) / samp_freq))

    return rating_dic


def load_ecg_files(subjects, condition, sba=True, interpolate=True):
    """
    Load 1Hz heart rate data (bmp -> z-scored).
    :param subjects: list of subjects
    :param condition: nomov or mov
    :param sba: Loading them in single phases or concatented into Space-Break-Andes (SBA) or SA
    :param interpolate: whether to interpolate missing values
    :return: ECG data
    """

    if not isinstance(subjects, list):
        subjects = [subjects]

    # Create ECG Component Dictionary
    ecg_dic_keys = [str(i) for i in subjects]  # == list(map(str, subjects))  # these keys also work int
    ecg_dic = {}
    ecg_dic.update((key, dict.fromkeys(roller_coasters(condition, sba), [])) for key in ecg_dic_keys)

    condition_keys = [condition]  # "NoMov", "Mov"

    for key in ecg_dic_keys:
        ecg_dic[key].update({"SBA" if sba else "SA": dict.fromkeys(condition_keys, [])})

    for subject in subjects:
        for num, coaster in enumerate(roller_coasters(condition, sba)):

            folder = coaster.split("_")[1]  # either "Mov" or "NoMov" (only needed for sba case)

            if sba:
                file_name = path_ecg_sba + f"{folder}/NVR_{s(subject)}_SBA_{folder}.txt"

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

                # if sba
                # Split SBA file in Space-Break-Andes
                if "Space" in coaster:
                    sba_idx_start = 0
                    sba_idx_end = int(t_roller_coasters(sba)[0])
                elif "Break" in coaster:
                    sba_idx_start = int(t_roller_coasters(sba)[0])
                    sba_idx_end = int(sum(t_roller_coasters(sba)[0:2]))
                else:  # "Ande" in coaster
                    sba_idx_start = int(sum(t_roller_coasters(sba)[0:2]))
                    sba_idx_end = int(sum(t_roller_coasters(sba)))
                    # Test lengths
                    if sba_idx_end < hr_vector.shape[0]:
                        print("hr_vector of S{} has not expected size: Diff={} data points.".format(
                            str(subject).zfill(2), hr_vector.shape[0] - sba_idx_end))
                        assert (hr_vector.shape[0] - sba_idx_end) <= 3, \
                            "Difference too big, check files"

                ecg_dic[str(subject)][coaster] = copy.copy(hr_vector[sba_idx_start:sba_idx_end])

                # process whole S(B)A only once per condition (NoMov, Mov)
                if num == 0:
                    ecg_dic[str(subject)]["SBA" if sba else "SA"][condition] = copy.copy(hr_vector)

                # Check times:
                if not sba:
                    if not (np.round(t_roller_coasters(sba)[num], 1) == hr_vector.shape[0]):
                        print("Should be all approx. the same:\nTime of {}: {}"
                              "\nLength of hr_vector: {}".format(roller_coasters(condition, sba=sba)[num],
                                                                 t_roller_coasters(sba)[num],
                                                                 hr_vector.shape[0]))
                else:  # if sba:
                    if not (np.round(sum(t_roller_coasters(sba)), 1) == hr_vector.shape[0]):
                        print("Should be all approx. the same:\nTime of SBA: {}"
                              "\nLength of hr_vector: {}".format(sum(t_roller_coasters(sba)),
                                                                 hr_vector.shape[0]))

    return ecg_dic


def mean_line_prediction(subject, condition, sba=True):
    """
    Returns the accuracy if all prediction steps would output overall mean
    :param subject: Subject ID
    :param condition: nomov or mov
    :param sba: SBA or SA (False)
    :return: accuracy of mean-line prediction
    """
    rating = load_rating_files(subjects=subject,
                               condition=condition,
                               sba=sba)[str(subject)]["SBA" if sba else "SA"][condition]

    rating = normalization(array=rating, lower_bound=-1, upper_bound=1)
    mean_line = np.mean(rating)
    max_diff = 1.0 - (np.abs(rating) * -1.0)  # chances depending on rating-level
    correct = 1.0 - np.abs((mean_line - rating)) / max_diff
    mean_line_accuracy = np.nanmean(correct)

    return mean_line_accuracy


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

    def next_batch(self, batch_size=1, successive=1, successive_mode=1, randomize=True):
        """
        Return the next 'batch_size' examples from this data set

        :param successive: How many of the random batches shall remain in successive order. That is, the
                           time-slices (1-sec each) that are kept in succession. Representing subjective
                           experience, this could be 2-3 sec in order to capture responses to the
                           stimulus environment.
        :param successive_mode: Mode 1) batches start always at the same spot (-): perfect line-up (+)
                                Mode 2) batches can start at random spots, no perfect line-up guaranteed
        :param batch_size: Batch size
        :param randomize: Randomize order in data. To avoid learning of only the autocorrelation in data
                          the Default is set to True
        :return: Next batch
        """

        # if batch_size > 1:
        #     raise ValueError("A batch_size of > 1 is not recommended at this point")

        assert batch_size % successive == 0, "batch_size must be a multiple of successive"
        assert successive_mode in [1, 2], "successive_mode must be either 1 or 2"

        if len(self.remaining_slices) >= batch_size:

            # Select slice according to number of batches
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
                                    if sel_arr_idx < len(self.remaining_slices) - 1:
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
                                "successive must divide length of dataset equally"

                            # Attach successive batches
                            selection_array_idx = np.append(selection_array_idx, sel_arr_idx).astype(int)
                            for _ in range(successive - 1):
                                selection_array_idx = np.append(selection_array_idx,
                                                                selection_array_idx[-1] + 1)

                            batch_counter -= successive

                    else:  # successive_mode == 2:

                        while True:
                            sel_arr_idx = np.random.choice(a=range(len(self.remaining_slices)),
                                                           size=int(batch_size / successive),
                                                           replace=False)

                            # Check whether distances big enough
                            counter = 0
                            for element in sel_arr_idx:
                                diff = np.abs(element - sel_arr_idx)
                                diff = np.delete(diff, np.where(diff == 0))
                                if not np.all(diff >= successive):
                                    counter += 1
                            if counter == 0 and np.max(sel_arr_idx) + successive - 1 < len(
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
                                if sel_arr_idx < len(self.remaining_slices) - 1:
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
                            for _ in range(successive - 1):
                                additional_array_idx = np.append(additional_array_idx,
                                                                 additional_array_idx[-1] + 1)

                            batch_counter -= successive

                    else:  # successive_mode == 2:
                        while True:
                            sel_arr_idx = np.random.choice(a=range(len(self.remaining_slices)),
                                                           size=int(
                                                               remaining_slices_to_draw / successive),
                                                           replace=False)

                            # Check whether distances big enough
                            counter = 0
                            for element in sel_arr_idx:
                                diff = np.abs(element - sel_arr_idx)
                                diff = np.delete(diff, np.where(diff == 0))
                                if not np.all(diff >= successive):
                                    counter += 1
                            if counter == 0 and np.max(sel_arr_idx) + successive - 1 < len(
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


def get_nevro_data(subject, task, cond, component, hr_component, filetype, hilbert_power, band_pass,
                   equal_comp_matrix=None,
                   s_fold_idx=None, s_fold=10,
                   sba=True, s_freq_eeg=250.,
                   shuffle=False, testmode=False):
    """
    Prepares NeVRo dataset and returns the s-fold-prepared dataset.
    S-Fold Validation, S=5: [ |-Train-|-Train-|-Train-|-Valid-|-Train-|] Dataset
    Args:
        subject: Subject Nr., subject dataset for training
        task: either regression or (binary)-classification
        cond: Either "NoMov"(=default) or "Mov"
        component: which component to feed (int, list(int))
        hr_component: Whether to attach hear rate component to neural components (True/False)
        filetype: Whether 'SSD' or 'SPOC'
        hilbert_power: hilbert-transform SSD-components, then extract z-scored power
        band_pass: Whether SSD components are band-passed filter for alpha
        equal_comp_matrix: None: return matrix with given components;
                           m(int): column-span m of matrix, fill columns with zeros where no component
        s_fold_idx: index which of the folds is taken as validation set
        s_fold: s-value of S-Fold Validation [default=10]
        sba: Whether to use SBA-data
        s_freq_eeg: Sampling Frequency of EEG
        shuffle: shuffle data (for classific. task to have balance low/high arousal in all folds/valsets)
        testmode: Whether to load data for testmode
    Returns:
        Train, Validation Datasets
    """
    # Adjust inputs
    cond = check_condition(cond=cond)
    task = task.lower()
    filetype = filetype.upper()
    if not isinstance(component, list):  # If int, transform to list
        component = [component]

    # Check inputs
    if s_fold_idx:
        assert s_fold_idx < s_fold, \
            "s_fold_idx (={}) must be in the range of the number of folds (={})".format(s_fold_idx,
                                                                                        s_fold)
    assert filetype in ["SSD", "SPOC"], "filetype must be either 'SSD' or 'SPOC'"
    assert task in ["regression", "classification"], "task must be 'regression' or 'classification'"
    if equal_comp_matrix == "None":
        equal_comp_matrix = None
    if equal_comp_matrix:
        assert isinstance(equal_comp_matrix, int), "equal_comp_matrix must be None or integer."
        assert equal_comp_matrix >= (len(component) + (1 if hr_component else 0)), \
            "List of given component(s) is too long (>equal_comp_matrix)"

    if s_fold_idx is None:
        s_fold_idx = np.random.randint(low=0, high=s_fold)
        print("s_fold_idx randomly chosen:", s_fold_idx)

    noise_comp = False  # init
    max_comp = get_num_components(subject=subject, condition=cond, filetype=filetype, selected=False)

    for comp_idx, comp in enumerate(component):

        assert comp in range(1, max_comp+1) or comp in range(91, 90 + max_comp+1), \
            f"Components must be in range (1, {max_comp})."

        # Check demand for noise component
        if comp in range(91, 90 + max_comp+1):
            component[comp_idx] -= 90  # decode
            noise_comp = True  # switch on noise-mode of given component

        else:
            try:
                if noise_comp:
                    raise ValueError("If one, then all given components must be noise components")
            except NameError:
                noise_comp = False

    # Load and prepare EEG components
    eeg_data = load_component(subjects=subject, condition=cond, f_type=filetype,
                              band_pass=band_pass, sba=sba)

    # Load and prepare Rating targets
    rating_data = load_rating_files(subjects=subject, condition=cond, sba=sba,
                                    bins=True if task == "classification" else False)

    # Load and prepare heart-rate data
    ecg_data = None  # init
    if hr_component:
        ecg_data = load_ecg_files(subjects=subject, condition=cond, sba=sba)  # interpolation as default

    # TODO Load for more than one subject

    # 0) If SBA: Take Space-Break-Ande files from dictionaries
    # 1) and choose specific components
    # eeg_cnt = eeg_data[str(subject)]["SBA"][cond][:, 0:2]  # = first 2 components; or best, or other

    eeg_cnt = eeg_data[str(subject)]["SBA" if sba else "SA"][cond][:, [comp-1 for comp in component]]

    # If Noise-Mode, shuffle component
    if noise_comp:
        np.random.shuffle(eeg_cnt)

    # Load ratings
    rating_cnt = rating_data[str(subject)]["SBA"][cond]
    # Normalize rating_cnt to [-1;1] due to tanh-output of NeVRoNet
    rating_cnt = normalization(array=rating_cnt, lower_bound=-1, upper_bound=1)

    # Load heart rate data (ECG) if applicable:
    ecg_cnt = ecg_data[str(subject)]["SBA"][cond] if hr_component else None

    # Check whether EEG data too long. If yes: prune it
    if sba:

        len_test = eeg_cnt.shape[0] / s_freq_eeg - rating_cnt.shape[0]

        if len_test > 0.0:

            # 3 intersections where values can be removed (instead of cutting only at the end/start)
            to_delete = np.cumsum(t_roller_coasters(sba))  # [ 148.,  178.,  270.]
            to_delete *= s_freq_eeg
            to_cut = int(round(len_test * s_freq_eeg))

            print(f"EEG data of {s(subject)} gets trimmed by {to_cut} data points")

            del_counter = 2
            while len_test > 0.0:
                if del_counter == -1:
                    del_counter = 2
                # starts to delete in the end (Andes), then after Break, then after first part (Space)
                eeg_cnt = np.delete(arr=eeg_cnt, obj=to_delete[del_counter], axis=0)
                del_counter -= 1

                len_test = eeg_cnt.shape[0] / s_freq_eeg - rating_cnt.shape[0]

        elif len_test < 0.0:
            raise OverflowError("eeg_cnt file is too short. Implement interpolation function.")

    else:
        raise NotImplementedError("'SA' case not implemented yet.")

    if hilbert_power:
        # print("I load Hilbert transformed data (z-power)")
        # Perform Hilbert Transformation
        for comp in range(eeg_cnt.shape[1]):
            eeg_cnt[:, comp] = calc_hilbert_z_power(array=eeg_cnt[:, comp])

    # IF, then attach HR to neural components
    if hr_component:
        # HR is in 1Hz: so stretch to length of eeg (250Hz)
        ecg_cnt_streched = np.reshape(np.repeat(a=ecg_cnt, repeats=int(s_freq_eeg)), newshape=[-1, 1])
        # Then attach to (neural) input matrix
        eeg_cnt = np.concatenate((eeg_cnt, ecg_cnt_streched), 1)

    # If the model input should always be equal in size, i.e. a matrix with the same shape (len(eeg), m):
    if equal_comp_matrix:
        m = equal_comp_matrix  # m:= column-dimension of input matrix (for readability)
        if m > eeg_cnt.shape[1]:
            # Attach m - (number of column of eeg_cnt) null-vector(s) to input matrix:
            eeg_cnt = np.concatenate((eeg_cnt, np.zeros(shape=(eeg_cnt.shape[0], m-eeg_cnt.shape[1]))), 1)

    # If Testset, overwrite eeg_cnt data with artifical data (for model testing)
    if testmode:
        # # 1) negative sin(ratings), then stretch
        # eeg_cnt = np.reshape(a=np.repeat(a=-np.sin(rating_cnt), repeats=250, axis=0),
        #                      newshape=eeg_cnt.shape)
        # # 2) negative sin(ratings)**3, then stretch
        # eeg_cnt = np.reshape(a=np.repeat(a=-np.sin(rating_cnt**3), repeats=250, axis=0),
        #                      newshape=eeg_cnt.shape)
        # # 3) test with global slope
        # slope = np.linspace(0, 0.2, len(rating_cnt)) + np.random.normal(loc=0., scale=0.01,
        #                                                                 size=len(rating_cnt))
        slope = np.linspace(0, 0.3, len(rating_cnt)) + np.random.normal(loc=0., scale=0.02,
                                                                        size=len(rating_cnt))
        # eeg_cnt = np.reshape(a=np.repeat(a=-np.sin(rating_cnt+slope), repeats=250, axis=0),
        #                      newshape=eeg_cnt.shape)
        # 4) with-in-1-second slope (local slope) 5) stronger slope
        eeg_cnt = np.reshape(a=np.repeat(a=-rating_cnt, repeats=250, axis=0), newshape=eeg_cnt.shape)
        eeg_cnt += np.reshape(a=np.repeat(a=slope, repeats=250, axis=0), newshape=eeg_cnt.shape)
        # Get it out of [-1,1]-range
        eeg_cnt *= 3
        # 6) inverse [1,2,3] -> [3,2,1]
        # eeg_cnt = eeg_cnt[::-1]

        # add some random noise ε (scale=0.05 is relatively large)
        eeg_cnt += np.random.normal(loc=0., scale=0.05, size=eeg_cnt.shape[0]).reshape(eeg_cnt.shape)
        print("In test mode: Input data: (-3*ratings + strong_slope) + noise ε")

    # 1) Split data in S(=s_fold) sets
    # np.split(np.array([1,2,3,4,5,6]), 3) >> [array([1, 2]), array([3, 4]), array([5, 6])]

    # split EEG data w.r.t. to total sba-length
    eeg_cnt_split = splitter(eeg_cnt, n_splits=int(sum(t_roller_coasters(sba))))
    # [sec, data-points per sec, components)

    # If semi-balanced low-high-arousal values for validation set is required (in binary classi.)
    # then do shuffle:
    idx = np.arange(len(rating_cnt))  # init
    if shuffle:
        np.random.shuffle(idx)
        if task == "regression":
            cprint("Note: Shuffling data for regression task leads to more difficult interpretation of "
                   "results/plots and makes successive batches redundant (if applied).", col='y')

    # eeg_concat_split[0][0:] first to  250th value in time
    # eeg_concat_split[1][0:] 250...500th value in time
    eeg_split = splitter(array_to_split=eeg_cnt_split[idx], n_splits=s_fold)
    rating_split = splitter(array_to_split=rating_cnt[idx], n_splits=s_fold)

    # eeg_concat_split.shape    # (n_chunks[in 1sec], n_samples_per_chunk [250Hz], channels)
    # eeg_split.shape           # (s_fold, n_chunks_per_fold, n_samples_per_chunk, channels)
    # rating_split.shape        # (s_fold, n_samples_per_fold,)

    # # Assign variables accordingly:
    # Validation set:
    validation_eeg = eeg_split[s_fold_idx]
    validation_ratings = rating_split[s_fold_idx]

    # Training  set:
    train_eeg = np.delete(arr=eeg_split, obj=s_fold_idx, axis=0)
    # removes the val-set from the data set (np.delete)
    train_ratings = np.delete(arr=rating_split, obj=s_fold_idx, axis=0)
    # Merge the training sets again (concatenate for 2D & vstack for >=3D)
    # Cautious: Assumption that performance is partly independent of correct order (done already with SBA)
    train_eeg = np.vstack(train_eeg)  # == np.concatenate(train_eeg, axis=0)
    train_ratings = np.concatenate(train_ratings, axis=0)

    # Create datasets
    train = DataSet(name="Training", eeg=train_eeg, ratings=train_ratings, subject=subject,
                    condition=cond, task=task)
    validation = DataSet(name="Validation", eeg=validation_eeg, ratings=validation_ratings,
                         subject=subject, condition=cond, task=task)
    # # Test set
    # test = DataSet(eeg=test_eeg, ratings=test_ratings, subject=subject, condition=condition, task=task)
    test = None  # TODO implement

    # return base.Datasets(train=train, validation=validation, test=test), s_fold_idx
    return {"train": train, "validation": validation, "test": test, "order": idx}


# # Testing
# nevro_data = get_nevro_data(subject=36, task="regression", cond="NoMov",
#                             component=5, hr_component=True,
#                             filetype="SSD", hilbert_power=True, band_pass=True,
#                             s_fold_idx=9, s_fold=10, sba=True)
# print("Subject:", nevro_data["train"].subject,
#       "\nEEG shape:", nevro_data["train"].eeg.shape,
#       "\nRating shape:", nevro_data["train"].ratings.shape,
#       "\nCondition:", nevro_data["train"].condition)
#
# nevro_data = get_nevro_data(subject=44, task="classification", cond="NoMov",
#                             component=4, hr_component=False,
#                             filetype="SSD", hilbert_power=False, band_pass=False,
#                             s_fold_idx=9, s_fold=10, sba=True)
# print("Subject:", nevro_data["validation"].subject,
#       "\nEEG shape:", nevro_data["validation"].eeg.shape,
#       "\nRating shape:", nevro_data["validation"].ratings.shape,
#       "\nCondition:", nevro_data["validation"].condition)
#
# # Test equal_comp_matrix
# nevro_data = get_nevro_data(subject=44, task="classification", cond="NoMov",
#                             component=[1, 2, 4], hr_component=True, equal_comp_matrix=None,
#                             filetype="SSD", hilbert_power=False, band_pass=False,
#                             s_fold_idx=9, s_fold=10, sba=True)
# print("Subject:", nevro_data["validation"].subject,
#       "\nEEG shape:", nevro_data["validation"].eeg.shape,
#       "\nRating shape:", nevro_data["validation"].ratings.shape,
#       "\nCondition:", nevro_data["validation"].condition)
#
# nevro_data = get_nevro_data(subject=44, task="classification", cond="NoMov",
#                             component=[1, 2, 4], hr_component=True, equal_comp_matrix=6,
#                             filetype="SSD", hilbert_power=False, band_pass=False,
#                             s_fold_idx=9, s_fold=10, sba=True)
# print("Subject:", nevro_data["validation"].subject,
#       "\nEEG shape:", nevro_data["validation"].eeg.shape,  # (27, 250, 6=columns)
#       "\nRating shape:", nevro_data["validation"].ratings.shape,
#       "\nCondition:", nevro_data["validation"].condition)

# for _ in range(27):
#     x = nevro_data["validation"].next_batch(batch_size=4, randomize=True)
#     print("Current Batch:", nevro_data["validation"].current_batch)
#     print("n_remaining slices: {}/{}".format(len(nevro_data["validation"].remaining_slices),
#                                              nevro_data["validation"]._num_time_slices))
#     print("index in current epoch:", nevro_data["validation"]._index_in_epoch)
#     print("epochs copmleted:", nevro_data["validation"]._epochs_completed)
#     print("")

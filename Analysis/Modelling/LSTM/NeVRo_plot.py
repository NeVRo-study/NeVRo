# coding=utf-8
"""
Plot ECG (RR and HR), EEG, and Rating Trajectories
    • indicate different phases
    • plot RR intervals
    • plot HR (heart beats per minute, 1/RR)
    • plot EEG (SSD, SPoC)
    • plot subjective Ratings
    • save plots

Author: Simon M. Hofmann | <[surname].[lastname][at]pm.me> | 2017, 2020 (Update)
"""

#%% Import

from utils import *
from load_data import load_component, t_roller_coasters
import os.path
import copy
import matplotlib.pyplot as plt


#%% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >

class NeVRoPlot:
    def __init__(self, n_sub=45, dropouts=None, subject_selection=None, smooth_w_size=3, trimmed=True):
        """
        :param n_sub: number of all subjects
        :param dropouts: dropouts=[1, 12, 32, 33, 35, 38, 40, 41, 42, 43, 45]  # conservative
        33, 38, 41 are ok if only for NoMov; 40 also, if correct markers
        :param subject_selection: hand-selected subset of subjects
        :param smooth_w_size: window-size for smoothing
        :param trimmed: SBA-trimmed times
        """

        # Change to folder which contains files
        # (asserts being executed from NeVRo/Analysis/Modelling/LSTM/)
        self.wdic = "../../../Data/"
        self.wdic_plots = "../../../Results/Plots/"
        self.wdic_cropRR = f"{self.wdic}ECG/TR_cropped/"
        self.wdic_SBA = f"{self.wdic}ECG/SBA/"
        self.wdic_SA = f"{self.wdic}ECG/SA/"
        self.wdic_Rating = f"{self.wdic}ratings/continuous/z_scored/"
        self.wdic_Rating_not_z = f"{self.wdic}ratings/continuous/not_z_scored/"
        self.wdic_cropTR_trim = f"{self.wdic}ECG/TR_cropped/trimmed/"

        # Set variables
        self.n_sub = n_sub  # number of subjects
        self.subjects = np.arange(1, n_sub+1)  # array with all subjects
        # Define array of dropouts (subject ID)
        self.dropouts = [1, 12, 32, 35, 40, 42, 45] if not dropouts else dropouts
        # Select specific subjects of interest
        self.subject_selection = np.array(subject_selection) if subject_selection else np.array([])
        self.n_sub, self.subjects = self.subjects_request(subjects_array=self.subjects,
                                                          dropouts_array=self.dropouts,
                                                          selected_array=self.subject_selection)

        self.phases = ["Resting_Open", "Resting_Close",
                       "Space_Mov",          "Break_Mov",          "Ande_Mov",
                       "Rating_Space_Mov",   "Rating_Break_Mov",   "Rating_Ande_Mov",
                       "Space_NoMov",        "Break_NoMov",        "Ande_NoMov",
                       "Rating_Space_NoMov", "Rating_Break_NoMov", "Rating_Ande_NoMov"]

        # Trimmed time
        self.trimmed = trimmed
        self.trim_time = 5.
        self.trimmed_time_space = 153. - self.trim_time
        self.trimmed_time_break = 30.
        self.trimmed_time_ande = 97. - self.trim_time
        self.trimmed_resting = 300.

        # Get phase-times (=max length of phases of all subjects)
        self.phase_lengths = np.zeros((len(self.phases),))
        self._update_phase_lengths()
        self.phase_lengths_int = np.array([int(self.phase_lengths[i])+1 for i in range(len(
            self.phase_lengths))])
        if self.trimmed:
            self.phase_lengths_int[2:] -= 1

        # Will contain z-scored HR for each phase, i.e. z-scored within one phase.
        self.each_phases_z = {"Resting_Open": [], "Resting_Close": [],
                              "Space_Mov": [],          "Break_Mov": [],          "Ande_Mov": [],
                              "Rating_Space_Mov": [],   "Rating_Break_Mov": [],   "Rating_Ande_Mov": [],
                              "Space_NoMov": [],        "Break_NoMov": [],        "Ande_NoMov": [],
                              "Rating_Space_NoMov": [], "Rating_Break_NoMov": [], "Rating_Ande_NoMov": []}

        # Will contain each phase separately, and various concatenated versions (all-phases):
        # 'all_phases', 'all_phases_z'*, 'all_phases_z_smooth'* | *z-scored across all concatenated phases
        self.all_phases = copy.copy(self.each_phases_z)
        self._concat_all_phases()  # will update self.all_phases
        self._create_z_scores()  # will update self.all_phases

        self.w_size = smooth_w_size  # default from McCall et al. (2015)
        self._create_smoothing(mode=0)  # modes:=["ontop"[0], "hanging"[1]]

        self.ratings_dic = {}
        self._update_ratings_dic()
        self.roller_coaster = np.array(['Space_NoMov', 'Space_Mov', 'Ande_Mov', 'Ande_NoMov'])

        # "Space", "Break" and "Ande" (SBA) concatenated. Raw and z-scored
        self.SBA_ecg = self._create_sba()
        self.SBA_ecg_split = self._create_sba_split()
        self._split_sba()  # Updates self.SBA_ecg_split dictionary with split z-scored sba data
        self.SBA_ratings = copy.deepcopy(self.SBA_ecg)
        self._update_sba_ratings_dic()

        # "Space" and "Ande" (SA) concatenated. Raw and z-scored
        self.SA = copy.deepcopy(self.SBA_ecg)
        self.SA_ratings = copy.deepcopy(self.SBA_ratings)
        self._update_sa_dics()

    @staticmethod
    def close(n=1):
        for _ in range(n):
            plt.close()

    def save_plot(self, filename, fm="pdf"):
        if ".eps" in filename or fm == "eps":
            print("For '*.eps'-export use fig./ax.set_rasterized(True). Alternatievly use '*.pdf'")

        plt.savefig(self.wdic_plots + (filename + f".{fm}" if ("." not in filename) else filename))
        plt.close()

    def smooth(self, array_to_smooth, sliding_mode="ontop"):

        len_array = len(array_to_smooth)
        smoothed_array = np.zeros((len_array,))  # init. output array

        if sliding_mode == "hanging":  # causes shift of peaks, but more realistic
            # attach NaN in the beginning for correct sliding window calculions
            edge_nan = np.repeat(np.nan, self.w_size-1)
            array_to_smooth = np.concatenate((edge_nan, array_to_smooth), axis=0)

            for i in range(len_array):
                smoothed_array[i] = np.nanmean(array_to_smooth[i: i+self.w_size])

        if sliding_mode == "ontop":
            # self.w_size  need to be odd number 1, 3, 5, ...
            if self.w_size % 2 == 0:
                self.w_size -= 1
                print("Smoothing window size need to be odd, adjusted(i.e.: -1) to:", self.w_size)

            # attach NaN in the beginning and end for correct sliding window calculions
            edge_nan_start = np.repeat(np.nan, int(self.w_size/2))
            edge_nan_end = edge_nan_start
            array_to_smooth = np.concatenate((edge_nan_start, array_to_smooth), axis=0)
            array_to_smooth = np.concatenate((array_to_smooth, edge_nan_end), axis=0)

            for i in range(len_array):
                smoothed_array[i] = np.nanmean(array_to_smooth[i: i+self.w_size])

        return smoothed_array

    @staticmethod
    def subjects_request(subjects_array, dropouts_array, selected_array):
        print("Do you want to include all subjects or only subjects with complete datasets for plotting")
        print("")
        print("     1:= all Subjects")
        print("     2:= subset of subjects (without dropouts)")
        print("     3:= subset of selected subjects")
        print("")
        set_subset = input("Press: 1, 2 or 3")

        if int(set_subset):
            set_subset = int(set_subset)

        if set_subset < 1 or set_subset > 3:
            raise ValueError("Choice must be 1, 2 or 3")

        if set_subset == 2:  # Create array of selected subjects, i.e. kick dropouts out
            subjects_array = np.setdiff1d(subjects_array, dropouts_array)  # elegant, 1
            print("Create subset of all subjects without dropouts")
        elif set_subset == 3:
            subjects_array = selected_array
            print("Create subset of selected subjects")

        num_sub = len(subjects_array)

        return num_sub, subjects_array

    @staticmethod
    def plot_request():
        print("Do you want to plot z-scored HR per phase over all subjects?")
        print("")
        print("     11:= yes and save")
        print("     12:= yes but not save")
        print("     2:= no")
        print("")
        plot_answer = input("Type: 11, 12 or 2")

        if int(plot_answer):
            plot_answer = int(plot_answer)

        if plot_answer not in [11, 12, 2]:
            raise ValueError("Choice must be 11, 12 or 2")

        save_answer = True if plot_answer == 11 else False
        plot_answer = True if plot_answer >= 11 else False

        return plot_answer, save_answer

    def _update_phase_lengths(self):

        if self.trimmed:
            t_vec = np.array([self.trimmed_time_space, self.trimmed_time_break, self.trimmed_time_ande])
            self.phase_lengths[:] = np.append(np.tile(self.trimmed_resting, 2), np.tile(t_vec, 4))

        else:
            for sub in self.subjects:
                for num, phase in enumerate(self.phases):
                    tr_file_name = self.wdic_cropRR + "NVR_S{}_{}_T_R.txt".format(str(sub).zfill(2),
                                                                                  phase)
                    if os.path.isfile(tr_file_name):
                        tr_file = np.loadtxt(tr_file_name)
                        self.phase_lengths[num] = tr_file[-1] if tr_file[-1] > self.phase_lengths[num] \
                            else self.phase_lengths[num]

        print("Length of each phase:\n", self.phase_lengths)

    def subject_condition(self, subject_id):
        table_of_condition = np.genfromtxt(self.wdic + "Table_of_Conditions.csv", delimiter=";")
        table_of_condition = table_of_condition[1:, ]
        # remove first column (sub-nr, condition, gender (1=f, 2=m))

        sub_cond = int(table_of_condition[np.where(table_of_condition[:, 0] == subject_id), 1])

        return sub_cond

    def _update_ratings_dic(self):
        """Load z-rating files and write them in ratings_dic"""
        if self.trimmed:
            len_space = int(self.trimmed_time_space)
            len_break = int(self.trimmed_time_break)
            len_ande = int(self.trimmed_time_ande)
        else:
            len_space = self.all_phases["Space_Mov"].shape[1]  # same length for NoMov
            len_break = self.all_phases["Break_Mov"].shape[1]
            len_ande = self.all_phases["Ande_Mov"].shape[1]

        space_init = np.reshape(np.repeat(np.nan, self.n_sub * len_space), newshape=(self.n_sub,
                                                                                     len_space))
        break_init = np.reshape(np.repeat(np.nan, self.n_sub * len_break), newshape=(self.n_sub,
                                                                                     len_break))
        ande_init = np.reshape(np.repeat(np.nan, self.n_sub * len_ande), newshape=(self.n_sub, len_ande))

        self.ratings_dic = {"Space_Mov": copy.copy(space_init),
                            "Break_Mov": copy.copy(break_init),
                            "Ande_Mov": copy.copy(ande_init),
                            "Space_NoMov": copy.copy(space_init),
                            "Break_NoMov": copy.copy(break_init),
                            "Ande_NoMov": copy.copy(ande_init)}

        r_coasters = ["space", "break", "andes"]
        condition = ["mov", "nomov"]

        for sub_idx, sub in enumerate(self.subjects):
            for num, coaster in enumerate(r_coasters):
                for cond in condition:
                    try:
                        sub_cond = self.subject_condition(subject_id=sub)  # cond of sub
                    except Exception:
                        sub_cond = "NaN"  # in case of S12
                    r = 1 if (sub_cond == 12 and cond == "mov") or \
                             (sub_cond == 21 and cond == "nomov") else 2  # run
                    rating_filename = self.wdic_Rating + "{}/{}/NVR_S{}_run_{}_{}_rat_z.txt".format(
                        cond,
                        coaster,
                        str(sub).zfill(2),
                        str(r),
                        coaster)

                    if os.path.isfile(rating_filename):
                        rating_file = np.genfromtxt(rating_filename, delimiter=',')[:, 1]
                        # only load col with ratings

                        # Fill in right slot of ratings_dic
                        ratings_key = ''
                        if coaster == 'space':
                            key_add = "Space_"
                        elif coaster == "break":
                            key_add = "Break_"
                        else:  # coaster == "andes"
                            key_add = "Ande_"
                        ratings_key += key_add
                        mov = "Mov" if cond == 'move' else "NoMov"
                        ratings_key += mov
                        # print("ratings_key:", ratings_key)
                        if self.ratings_dic[ratings_key][sub_idx, :].shape[0] != len(rating_file):
                            print(f"For Subject {sub} in {ratings_key}:")
                            print("self.ratings_dic[ratings_key][sub_idx, :].shape[0]={} \n"
                                  "len(rating_file)={}".format(
                                self.ratings_dic[ratings_key][sub_idx, :].shape[0], len(rating_file)))
                            raise ValueError(
                                f"Rating_file: '{rating_filename}' has not same length as HR_file!")
                        # print("Update Subject{}:".format(str(sub)), ratings_key)
                        self.ratings_dic[ratings_key][sub_idx, :] = copy.copy(rating_file)

                    else:
                        print(rating_filename, " does not exist")

    def _update_sba_ratings_dic(self):

        for tfs in self.SBA_ratings.keys():  # ["SBA", "zSBA"]
            conditions = [key for key in self.SBA_ratings[tfs].keys()]
            for cond in conditions:
                for sub_idx, sub in enumerate(self.subjects):

                    try:
                        sub_cond = self.subject_condition(subject_id=sub)  # cond of sub
                    except Exception:
                        sub_cond = "NaN"  # in case of S12

                    r = 1 if (sub_cond == 12 and cond.lower() == "mov") or \
                             (sub_cond == 21 and cond.lower() == "nomov") else 2

                    file_name = (self.wdic_Rating if tfs == "zSBA" else
                                 self.wdic_Rating_not_z) + f"/{cond.lower()}/SBA/NVR_{s(sub)}_run_{r}_" \
                                                           f"alltog_rat_z.txt"

                    if os.path.isfile(file_name):
                        rating_file = np.genfromtxt(file_name, delimiter=',')[:, 1]
                        # only load col with ratings
                    else:
                        rating_file = np.repeat(np.nan, self.SBA_ratings[tfs][cond].shape[1])

                    self.SBA_ratings[tfs][cond][sub_idx] = rating_file

    def _update_sa_ratings_dic(self):
        """
        This is just an intermediate step before pruning the loaded data to the SA-format
        in self._update_sa_dics()
        """

        for key in self.SA_ratings.keys():
            for cond in self.SA_ratings[key].keys():
                for sub_idx, sub in enumerate(self.subjects):

                    try:
                        sub_cond = self.subject_condition(subject_id=sub)  # cond of sub
                    except ValueError:  # or NameError
                        sub_cond = "NaN"  # in case of S12

                    r = 1 if (sub_cond == 12 and cond.lower() == "mov") or \
                             (sub_cond == 21 and cond.lower() == "nomov") else 2

                    file_name = self.wdic_Rating + f"/{cond.lower()}/SA/NVR_{s(sub)}_run_{r}_" \
                                                   f"alltog_rat_z.txt"

                    if os.path.isfile(file_name):
                        rating_file = np.genfromtxt(file_name, delimiter=',')[:, 1]
                        # only load col with ratings

                    else:
                        rating_file = np.repeat(np.nan, self.SA_ratings["zSBA"][cond].shape[1])

                    self.SA_ratings["zSBA"][cond][sub_idx, :] = rating_file

    def _update_sa_dics(self):
        """
        Update SA dict and SA_ratings. That is, deleting the break from the copied SBA dicts
        """

        # Delete-Index
        del_start = int(self.trimmed_time_space)  # beginning of break after 148sec
        del_end = int(self.trimmed_time_space + self.trimmed_time_break)  # end of break 148+30sec

        # Hear-Rate
        for key in self.SA.keys():
            for cond_key in self.SA[key].keys():
                # self.SA[key][cond_key].shape  # = (Nsub, 270)
                # Update SA
                self.SA[key][cond_key] = np.delete(arr=self.SA[key][cond_key],
                                                   obj=range(del_start, del_end), axis=1)

                assert self.SA[key][cond_key].shape[1] == self.trimmed_time_space + \
                       self.trimmed_time_ande, f"SA has wrong length={self.SA[key][cond_key].shape[1]}"

        for key in self.SA.keys():
            # Update Key Name from SBA => SA
            newkey = "zSA" if "z" in key else "SA"
            self.SA[newkey] = self.SA.pop(key)

        # Ratings
        self._update_sa_ratings_dic()  # loads non-zscored files in "zSA"

        for key in self.SA_ratings.keys():
            for cond_key in self.SA_ratings[key].keys():

                # Update SA_ratings
                self.SA_ratings[key][cond_key] = np.delete(arr=self.SA_ratings[key][cond_key],
                                                           obj=range(del_start, del_end), axis=1)

                assert self.SA_ratings[key][cond_key].shape[1] == self.trimmed_time_space + \
                       self.trimmed_time_ande, \
                    "SA_ratings has wrong length={}".format(self.SA_ratings[key][cond_key].shape[1])

        # Update Key Name from SBA => SA
        for key in self.SA_ratings.keys():
            newkey = "zSA" if "z" in key else "SA"
            self.SA_ratings[newkey] = self.SA_ratings.pop(key)

        # Now take z-scores in self.SA_ratings["zSA"]
        for cond_key in self.SA_ratings["zSA"].keys():
            for sub_idx in range(len(self.subjects)):
                sub_mean = np.nanmean(self.SA_ratings["zSA"][cond_key][sub_idx, :])
                sub_std = np.nanstd(self.SA_ratings["zSA"][cond_key][sub_idx, :])
                # z-scored
                self.SA_ratings["zSA"][cond_key][sub_idx, :] = \
                    copy.copy((self.SA_ratings["zSA"][cond_key][sub_idx, :])-sub_mean)/sub_std

        # Update the z-scores in self.SA["zSA"]
        for cond_key in self.SA["SA"].keys():
            for sub_idx, _ in enumerate(self.subjects):
                sub_mean = np.nanmean(self.SA["SA"][cond_key][sub_idx, :])
                sub_std = np.nanstd(self.SA["SA"][cond_key][sub_idx, :])
                # z-scored
                self.SA["zSA"][cond_key][sub_idx, :] = copy.copy(
                    (self.SA["SA"][cond_key][sub_idx, :]) - sub_mean)/sub_std

    def _concat_all_phases(self):
        """Concatenate all phases per subject"""

        all_length = sum(self.phase_lengths_int)
        all_phase = np.reshape(np.repeat(np.nan, self.n_sub * all_length), newshape=(self.n_sub,
                                                                                     all_length))
        # Update dic
        self.all_phases.update({"all_phases": copy.copy(all_phase)})
        self.all_phases.update({"all_phases_z": copy.copy(all_phase)})

        plotten, save_plots = self.plot_request()  # True or False

        for num, phase in enumerate(self.phases):

            # Update dict: all_phase_z
            phase_table = np.reshape(np.repeat(np.nan, self.phase_lengths_int[num] * self.n_sub),
                                     newshape=(self.n_sub, -1))
            self.each_phases_z[phase] = copy.copy(phase_table)
            self.all_phases[phase] = copy.copy(phase_table)

            if plotten:
                xlim = self.phase_lengths[num]
                fig_phase = plt.figure("z-HR in {}".format(phase), figsize=(14, 8))
                # fig_phase.legend(labels=phase, handles=...)

            for sub_idx, sub in enumerate(self.subjects):

                wdic = self.wdic_cropTR_trim if self.trimmed else self.wdic_cropRR
                tr_file_name = wdic + "NVR_S{}_{}_T_R.txt".format(str(sub).zfill(2), phase)
                if os.path.isfile(tr_file_name):

                    tr_file = np.loadtxt(tr_file_name)
                    # rr_file = np.array([tr_file[i]-tr_file[i-1] for i in range(1, len(tr_file))])
                    # hr_file = 60/rr_file  # HR (B/min)

                    # Convert from T_R to HR file, down-sample HR file to 1Hz
                    hr_file_ds = self._tr_to_hr(tr_array=tr_file)

                    # calculate z_score within phase (!)
                    z_hr_file_ds = self._z_score_hr(hr_array=hr_file_ds)

                    # Fill in each_phases_z:
                    for idx, item in enumerate(z_hr_file_ds):
                        self.each_phases_z[phase][sub_idx, idx] = item

                    for idx2, item2 in enumerate(hr_file_ds):
                        self.all_phases[phase][sub_idx, idx2] = item2

                    # plot
                    if plotten:
                        plt.plot(z_hr_file_ds)  # z_score

                if plotten:
                    mean_z = np.nanmean(self.each_phases_z[phase], axis=0)  # z_score-mean
                    plt.plot(mean_z, color="black", linewidth=2)  # plot z_score-mean
                    if sub_idx == self.n_sub-1:
                        y_min_max = 10
                        plt.vlines(0, ymin=-y_min_max, ymax=y_min_max, linestyles="--", alpha=0.6)
                        plt.vlines(xlim, ymin=-y_min_max, ymax=y_min_max, linestyles="--", alpha=0.6)

                        # Include events for roller coasters:
                        _phase, u = ("space",
                                     "18") if "Space" in phase else ("andes",
                                                                     "12") if "Andes" in phase else (None,
                                                                                                     None)

                        if _phase is not None:

                            events = np.genfromtxt(self.wdic + f"{_phase}_events.csv", delimiter=",",
                                                   dtype=f"|U{u}")
                            # U18 (or U12), 18 for max-length of str (n characters) of col_names
                            events = events[:, 1:]  # drop start=0

                            # Events need to be shifted, if trimmed
                            subtractor = self.trim_time / 2 if self.trimmed else 0

                            for idxe, event in enumerate(events[0, :]):
                                t_event = float(events[1, idxe]) - subtractor  # timepoint of event
                                shift = 0 if idxe % 2 == 0 else 1
                                plt.vlines(x=t_event, ymin=-(y_min_max-shift), ymax=mean_z[int(t_event)],
                                           linestyles="dotted", alpha=0.3)
                                plt.text(x=t_event, y=-(y_min_max-shift), s=event.replace("_", " "))

                    fig_phase.suptitle(f"z-scored HR of all subjects in phase: {phase}")

            if plotten:
                if save_plots:
                    self.save_plot(f"All_S_z_HR_in_{phase}")

        # close(n=len(phases))

        # Concatenate all phases to one array per subject
        for sub in range(self.n_sub):
            idx = 0
            old_idx = 0

            for num, phase in enumerate(self.phases):
                # phase_lengths_int[num] == all_phases[phase][sub].shape[0]
                idx += self.all_phases[phase][sub].shape[0]
                self.all_phases["all_phases"][sub][old_idx: idx] = self.all_phases[phase][sub]
                if num < 13:
                    old_idx = idx

    def _create_z_scores(self):
        # Create z-scores
        for sub in range(self.n_sub):
            # print("subject:", subjects[sub])
            sub_mean = np.nanmean(self.all_phases["all_phases"][sub])
            sub_std = np.nanstd(self.all_phases["all_phases"][sub])
            self.all_phases["all_phases_z"][sub] = (copy.copy(
                self.all_phases["all_phases"][sub]) - sub_mean) / sub_std

    def _create_smoothing(self, mode=0):
        # smoothing the HR: try 3data-point sliding-window [1,2,3] average and write over last value
        # Update dic
        self.all_phases.update({"all_phases_smooth": copy.copy(self.all_phases["all_phases"])})
        self.all_phases.update({"all_phases_z_smooth": copy.copy(self.all_phases["all_phases_z"])})
        # all_phases.keys()  # 'all_phases', 'all_phases_z'

        s_modes = ["ontop", "hanging"]
        s_mode = s_modes[mode]  # can be changed
        # w_size = 3  # can be changed, w_size € [3, 5, 11, 21]
        for i in range(self.n_sub):
            self.all_phases["all_phases_smooth"][i, :] = self.smooth(
                array_to_smooth=self.all_phases["all_phases"][i, :], sliding_mode=s_mode)
            self.all_phases["all_phases_z_smooth"][i, :] = self.smooth(
                array_to_smooth=self.all_phases["all_phases_z"][i, :], sliding_mode=s_mode)

    @staticmethod
    def _tr_to_hr(tr_array, exp_time=None):
        """
        Convert T_R array to HR_file (1Hz)
        :param tr_array: Array of T_R values
        :return: hr_array
        """
        if exp_time is None:
            exp_time = int(tr_array[-1]) + 1
        else:
            exp_time = int(exp_time)

        # down-sample HR file to 1Hz
        hr_array = np.zeros(shape=(exp_time,))
        for i in range(len(hr_array)):
            # bender = len(hr_file_ds)/len(HR_file)
            if len(tr_array[tr_array < i]) > 1:
                rr_int_i = (tr_array[tr_array < i][-1] - tr_array[tr_array < i][-2])
                hr_i = 60 / rr_int_i
                hr_array[i] = hr_i

        hr_array[np.where(hr_array == 0)] = np.nan  # replace zeros with NaN

        return hr_array

    @staticmethod
    def _z_score_hr(hr_array):
        """
        Calculate z_score of given HR array
        :param hr_array: HR array
        :return: z-scored HR array
        """

        z_hr_array = hr_array - np.nanmean(hr_array)
        z_hr_array /= np.nanstd(hr_array)

        return z_hr_array

    def _create_sba(self):
        """
        Create and save following data: concatenated and then z-scored data of
        "Space", "Break" and "Ande" (SBA)
        For this data-set we used trimmed versions of the coasters.
        At this point, ignore ECG during Ratings.
        """
        sba = {"SBA": {"Mov": [], "NoMov": []},
               "zSBA": {"Mov": [], "NoMov": []}}

        # Check with ECG_crop_RR.py
        all_trims = [self.trimmed_time_space, self.trimmed_time_break, self.trimmed_time_ande]
        total_trim_len = int(sum(all_trims))

        # Prepare Keys
        sba_keys = [key for key in self.each_phases_z.keys()]
        sba_keys_mov = []
        sba_keys_no_mov = []
        for key in sba_keys:
            if "NoMov" in key and "Rating" not in key:
                sba_keys_no_mov.append(key)
            elif "_Mov" in key and "Rating" not in key:
                sba_keys_mov.append(key)

        sba_order = ["S", "B", "A"]  # SBA
        sba_keys_mov_ordered = np.repeat("____________", len(sba_order))
        sba_keys_no_mov_ordered = np.repeat("____________", len(sba_order))
        for odx, ords in enumerate(sba_order):
            for i in range(len(sba_order)):
                if ords in sba_keys_mov[i]:
                    sba_keys_mov_ordered[odx] = sba_keys_mov[i]
                if ords in sba_keys_no_mov[i]:
                    sba_keys_no_mov_ordered[odx] = sba_keys_no_mov[i]
        sba_keys_mov, sba_keys_no_mov = sba_keys_mov_ordered, sba_keys_no_mov_ordered
        # sba_keys_mov = np.flip(np.roll(sba_keys_mov, 2), 0)  # should be right order: Space, Break, Ande
        # sba_keys_no_mov = np.roll(sba_keys_no_mov, 1)
        for i, sba_ord in enumerate(sba_order):
            if sba_ord not in sba_keys_mov[i] or sba_ord not in sba_keys_no_mov[i]:
                raise ValueError("Order not right")

        # Prepare dataframes
        sba_mov_all = np.reshape(np.repeat(np.nan, self.n_sub*total_trim_len), newshape=(self.n_sub,
                                                                                         total_trim_len))
        sba_no_mov_all = copy.copy(sba_mov_all)
        zsba_mov_all = copy.copy(sba_mov_all)
        zsba_no_mov_all = copy.copy(sba_mov_all)

        # Create files for NoMov and Mov, for each normal and z-scored SBA
        if len(os.listdir(self.wdic_cropTR_trim)) > 0:
            for sub_idx, sub in enumerate(self.subjects):

                # For Movement condition
                mov_sba = np.array([])
                for kdx, mov_key in enumerate(sba_keys_mov):
                    # e.g., "NVR_S01_Space_Mov_T_R.txt"
                    mov_file_name = self.wdic_cropTR_trim + "NVR_S{}_{}_T_R.txt".format(str(sub).zfill(2),
                                                                                        mov_key)
                    if os.path.isfile(mov_file_name):
                        mov_file = np.loadtxt(mov_file_name)
                        # Create HR out of TR files
                        mov_hr_file_ds = self._tr_to_hr(tr_array=mov_file, exp_time=all_trims[kdx])
                        # Check length:
                        if not len(mov_hr_file_ds) == all_trims[kdx]:
                            print("Length Problem with S{} in {}".format(sub, mov_key))
                            print("len(no_mov_hr_file_ds)={}, all_trims[kdx]={}".format(
                                len(mov_hr_file_ds), all_trims[kdx]))
                            # raise ValueError("mov_hr_file_ds and actual trim length differ")
                        # Concatenate
                        mov_sba = np.append(mov_sba, mov_hr_file_ds)

                # Check length
                if not mov_sba.shape[0] == total_trim_len:
                    print(f"For Sub{sub} in mov-cond does the concatenated file diverge from expected "
                          f"length")
                    print(f"mov_sba.shape[0]={mov_sba.shape[0]}, total_trim_len={total_trim_len}")
                else:
                    # Create z-score
                    z_mov_sba = self._z_score_hr(hr_array=mov_sba)
                    # Save in sba_mov_all
                    sba_mov_all[sub_idx, :] = mov_sba
                    zsba_mov_all[sub_idx, :] = z_mov_sba

                # For No-Movement condition
                no_mov_sba = np.array([])
                for kdx, no_mov_key in enumerate(sba_keys_no_mov):
                    no_mov_file_name = self.wdic_cropTR_trim + f"NVR_S{str(sub).zfill(2)}_{no_mov_key}_" \
                                                               f"T_R.txt"

                    if os.path.isfile(no_mov_file_name):
                        no_mov_file = np.loadtxt(no_mov_file_name)
                        # Create HR out of TR files
                        no_mov_hr_file_ds = self._tr_to_hr(tr_array=no_mov_file, exp_time=all_trims[kdx])
                        # Check length:
                        if not len(no_mov_hr_file_ds) == all_trims[kdx]:
                            print(f"Length Problem with S{sub} in {no_mov_key}")
                            print(f"len(no_mov_hr_file_ds)={len(no_mov_hr_file_ds)}, "
                                  f"all_trims[kdx]={all_trims[kdx]}")
                            # raise ValueError("no_mov_hr_file_ds and actual trim length differ")

                        # Concatenate
                        no_mov_sba = np.append(no_mov_sba, no_mov_hr_file_ds)

                # Check length
                if not no_mov_sba.shape[0] == total_trim_len:
                    print("For Sub{} in nomov-cond does the concatenated file diverge "
                          "from expected length".format(sub))
                    print(f"no_mov_sba.shape[0]={no_mov_sba.shape[0]}, total_trim_len={total_trim_len}")
                else:
                    # Create z-score
                    z_no_mov_sba = self._z_score_hr(hr_array=no_mov_sba)

                    # Save in sba_no_mov_all
                    sba_no_mov_all[sub_idx, :] = no_mov_sba
                    zsba_no_mov_all[sub_idx, :] = z_no_mov_sba

            # Save in SBA Dictionary
            sba["SBA"]["Mov"] = sba_mov_all
            sba["zSBA"]["Mov"] = zsba_mov_all
            sba["SBA"]["NoMov"] = sba_no_mov_all
            sba["zSBA"]["NoMov"] = zsba_no_mov_all

        return sba

    def _create_sba_split(self):
        sba_split_dic = copy.deepcopy(self.SBA_ecg)
        sba_split_dic.pop("SBA", None)  # removes "SBA" key from dic
        conditions = [key for key in self.SBA_ecg["zSBA"].keys()]  # ['NoMov', 'Mov']

        space_df = np.reshape(np.repeat(np.nan, self.n_sub * int(self.trimmed_time_space)),
                              newshape=(self.n_sub, int(self.trimmed_time_space)))

        break_df = np.reshape(np.repeat(np.nan, self.n_sub * int(self.trimmed_time_break)),
                              newshape=(self.n_sub, int(self.trimmed_time_break)))

        ande_df = np.reshape(np.repeat(np.nan, self.n_sub * int(self.trimmed_time_ande)),
                             newshape=(self.n_sub, int(self.trimmed_time_ande)))

        for cond in conditions:
            sba_split_dic["zSBA"][cond] = {}
            sba_split_dic["zSBA"][cond].update({"space": copy.copy(space_df),
                                                "break": copy.copy(break_df),
                                                "ande": copy.copy(ande_df)})

        return sba_split_dic

    def _split_sba(self):
        """
        Separates the zSBA trials in single phases: Space, Break, Andes
        And updates the self.SBA_ecg dictionary
        """

        conditions = [key for key in self.SBA_ecg["zSBA"].keys()]  # ['NoMov', 'Mov']
        phases = ["space", "break", "ande"]

        for cond in conditions:
            # self.SBA_ecg["zSBA"][cond]  # shape(nSub, 270=concat_time SBA)
            for sub_idx, _ in enumerate(self.subjects):
                start, end = 0, int(self.trimmed_time_space)
                space = self.SBA_ecg["zSBA"][cond][sub_idx][start: end]  # len=148
                start, end = end,  end + int(self.trimmed_time_break)
                breaks = self.SBA_ecg["zSBA"][cond][sub_idx][start: end]  # len=30
                start, end = end, end + int(self.trimmed_time_ande)
                ande = self.SBA_ecg["zSBA"][cond][sub_idx][start: end]  # len=92

                phase_data = [space, breaks, ande]

                for num, phase in enumerate(phases):
                    self.SBA_ecg_split["zSBA"][cond][phase][sub_idx, :] = phase_data[num]

    def save_sba(self):
        sba_keys = [key for key in self.SBA_ecg.keys()]  # ['zSBA', 'SBA']
        cond_keys = [key for key in self.SBA_ecg[sba_keys[0]].keys()]  # ['NoMov', 'Mov']

        for key in sba_keys:
            for sub_idx, sub in enumerate(self.subjects):
                for cond in cond_keys:
                    # extract file to save
                    file_to_save = self.SBA_ecg[key][cond][sub_idx, :]  # takes file per subject
                    # Define file_name for specific folder
                    subfolder = "z_scored_alltog/" if "z" in key else "not_z_scored/"
                    file_name = self.wdic_SBA + subfolder + cond + "/NVR_S{}_SBA_{}.txt".format(
                        str(sub).zfill(2), cond)
                    # Save the file
                    with open(file_name, "w") as file:
                        for item in file_to_save:
                            file.write("{}\n".format(item))

    def save_sa(self):

        for key in self.SA.keys():  # ['zSA', 'SA']
            for sub_idx, sub in enumerate(self.subjects):
                for cond in self.SA[key].keys():  # ['NoMov', 'Mov']
                    # extract file to save
                    file_to_save = self.SA[key][cond][sub_idx, :]  # takes file per subject
                    # Define file_name for specific folder
                    subfolder = "z_scored_alltog/" if "z" in key else "not_z_scored/"
                    file_name = self.wdic_SA + subfolder + cond + "/NVR_S{}_SA_{}.txt".format(
                        str(sub).zfill(2), cond)
                    # Save the file
                    with open(file_name, "w") as file:
                        for item in file_to_save:
                            file.write("{}\n".format(item))

    def save_sba_split(self):
        condition = [key for key in self.SBA_ecg_split["zSBA"].keys()]  # ['NoMov', 'Mov']
        phases = [key for key in self.SBA_ecg_split["zSBA"][condition[0]].keys()]
        # ['space', 'ande', 'break']

        for cond in condition:
            for phase in phases:
                for sub_idx, sub in enumerate(self.subjects):
                    w_dict = self.wdic_SBA + "z_scored_alltog/{}/{}/".format(cond, phase)
                    file_name = w_dict + "NVR_S{}_SBA_{}_{}.txt".format(str(sub).zfill(2), phase, cond)

                    with open(file_name, "w") as file:
                        for item in self.SBA_ecg_split["zSBA"][cond][phase][sub_idx, :]:
                            file.write("{}\n".format(item))

    def plot_sba_ratings(self, condition="NoMov", save_plot=False, class_bins=True):
        """Plot ratings for each subject over all phases (SBA)"""
        # self.SBA_ratings["zSBA"]["NoMov"].shape

        fig_rat = plt.figure(f"All Subjects | Rating | SBA | {condition}", figsize=(14, 8))
        for sub in range(self.n_sub):
            plt.plot(self.SBA_ratings["zSBA"][condition][sub], alpha=.5)

        mean_rating = np.nanmean(self.SBA_ratings["zSBA"][condition], axis=0)
        plt.plot(mean_rating, alpha=.8, c="black", lw=3, label="mean rating")

        # For comparability between conditions, find min/max across them
        rat_min = np.nanmin(np.append(self.SBA_ratings["zSBA"]["NoMov"],
                                      self.SBA_ratings["zSBA"]["Mov"]))
        rat_max = np.nanmax(np.append(self.SBA_ratings["zSBA"]["NoMov"],
                                      self.SBA_ratings["zSBA"]["Mov"]))

        ylims = np.max(np.abs((rat_min, rat_max)))

        for coaster in ["Space", "Andes"]:
            # Include events for roller coasters:
            if "Space" in coaster:
                events = np.genfromtxt(self.wdic + "space_events.csv", delimiter=",", dtype="|U18")
                # U18, 18 for max-length of str (n characters) of col_names
            else:  # elif "Ande" in coaster:
                events = np.genfromtxt(self.wdic + "ande_events.csv", delimiter=",", dtype="|U12")
                # U12, 12 for max-length of str (n characters) of col_names

            events = events[:, 1:]  # drop start=0

            subtractor = self.trim_time / 2 if self.trimmed else 0
            # Events need to be shifted, if trimmed

            if "Andes" in coaster:
                subtractor -= self.trimmed_time_space + self.trimmed_time_break

            shift_counter = 0
            for idxe, event in enumerate(events[0, :]):
                shift_counter += 1
                if shift_counter > 4:  # reset
                    shift_counter = 1
                shift = 1 if shift_counter > 2 else 0  # switch between 1 and zero

                t_event = float(events[1, idxe]) - subtractor  # timepoint of event
                up_down = -1 if idxe % 2 == 0 else 1

                if up_down > 0:
                    y_max_value = np.min(mean_rating[int(t_event)])
                else:  # up_down < 0:
                    y_max_value = np.max(mean_rating[int(t_event)])

                plt.vlines(x=t_event, ymin=(ylims - shift) * up_down, ymax=y_max_value,
                           linestyles="dotted",
                           alpha=1)
                plt.text(x=t_event+1, y=(ylims - shift) * up_down, s=event.replace("_", " "), size=8)

        plt.legend(loc="lower center")

        line = 0
        phase_names = [x.split("_")[0] for x in self.phases[-6:-3]]
        phase_names[-1] = phase_names[-1] + "s"
        plt.vlines(line, ymin=int(rat_min), ymax=rat_max, linestyles="--", alpha=0.5, lw=.8)
        # int(min) smaller
        for num, lines in enumerate(self.phase_lengths_int[-3:]):
            line += lines
            plt.vlines(line, ymin=int(rat_min), ymax=rat_max, linestyles="--", alpha=0.5, lw=.8)
            plt.text(x=line - lines, y=rat_max, s=phase_names[num], size=10)
        # fig_rat.suptitle("Ratings of all Subjects ({} condition, SBA)".format(condition))
        fig_rat.suptitle(
            f"Emotional arousal ratings ({'no-' if 'No' in condition else ''}movement condition)")
        plt.ylim(-4, 4)
        plt.xlabel("Time (seconds)")
        plt.ylabel("Emotional arousal rating (z-scored)")

        fig_rat.axes[0].set_xticks(np.arange(0, 270 + 1, 30))

        fig_rat.tight_layout(pad=0.6)

        if save_plot:
            self.save_plot(filename=f"All_Subjects_|_Rating_|_SBA_|_{condition}")

        # # Classification bins
        if class_bins:
            fig_rat_bins = plt.figure(f"All Subjects | Rating 2-Class Bins | SBA | {condition}",
                                      figsize=(14, 8))
            plt.plot(mean_rating, alpha=.8, c="black", lw=3, label="mean rating")

            tertile = int(len(mean_rating) / 3)
            lower_tert_bound = np.sort(mean_rating)[tertile]  # np.percentile(a=real_rating, q=33.33)
            upper_tert_bound = np.sort(mean_rating)[tertile * 2]  # np.percentile(a=real_rating, q=66.66)

            lw = 0.5
            plt.hlines(y=lower_tert_bound, xmin=0, xmax=len(mean_rating), linestyle="dashed",
                       colors="darkgrey", lw=lw, alpha=.8)
            plt.hlines(y=upper_tert_bound, xmin=0, xmax=len(mean_rating), linestyle="dashed",
                       colors="darkgrey", lw=lw, alpha=.8)

            plt.fill_between(x=np.arange(0, len(mean_rating), 1), y1=upper_tert_bound, y2=mean_rating,
                             where=mean_rating > upper_tert_bound,
                             color="orangered",
                             alpha='0.4')

            plt.fill_between(x=np.arange(0, len(mean_rating), 1), y1=lower_tert_bound, y2=mean_rating,
                             where=mean_rating < lower_tert_bound,
                             color="blue",
                             alpha='0.4')

            for coaster in ["Space", "Ande"]:
                # Include events for roller coasters:
                if "Space" in coaster:
                    events = np.genfromtxt(self.wdic + "space_events.csv", delimiter=",", dtype="|U18")
                    # U18, 18 for max-length of str (n characters) of col_names
                else:  # elif "Ande" in coaster:
                    events = np.genfromtxt(self.wdic + "ande_events.csv", delimiter=",", dtype="|U12")
                    # U12, 12 for max-length of str (n characters) of col_names

                events = events[:, 1:]  # drop start=0

                subtractor = self.trim_time / 2 if self.trimmed else 0
                # Events need to be shifted, if trimmed

                if "Ande" in coaster:
                    subtractor -= self.trimmed_time_space + self.trimmed_time_break

                shift_counter = 0
                for idxe, event in enumerate(events[0, :]):
                    shift_counter += 1
                    if shift_counter > 4:  # reset
                        shift_counter = 1
                    shift = 1 if shift_counter > 2 else 0  # switch between 1 and zero

                    t_event = float(events[1, idxe]) - subtractor  # timepoint of event
                    up_down = -1 if idxe % 2 == 0 else 1

                    if up_down > 0:
                        y_max_value = np.min(mean_rating[int(t_event)])
                    else:  # up_down < 0:
                        y_max_value = np.max(mean_rating[int(t_event)])

                    plt.vlines(x=t_event, ymin=(ylims - shift) * up_down, ymax=y_max_value,
                               linestyles="dotted", alpha=1)
                    plt.text(x=t_event+1, y=(ylims - shift) * up_down, s=event.replace("_", " "), size=8)

            plt.legend(loc="lower center")

            line = 0
            phase_names = [x.split("_")[0] for x in self.phases[-6:-3]]
            phase_names[-1] = phase_names[-1] + "s"
            plt.vlines(line, ymin=int(rat_min), ymax=rat_max, linestyles="--", alpha=0.5, lw=.8)
            # int(min) smaller
            for num, lines in enumerate(self.phase_lengths_int[-3:]):
                line += lines
                plt.vlines(line, ymin=int(rat_min), ymax=rat_max, linestyles="--", alpha=0.5, lw=.8)
                plt.text(x=line - lines, y=rat_max, s=phase_names[num], size=10)

            fig_rat_bins.suptitle(
                f"Emotional arousal ratings ({'no-' if 'No' in condition else ''}movement condition)")
            plt.ylim(-4, 4)
            plt.xlabel("Time (seconds)")
            plt.ylabel("Emotional arousal rating (z-scored)")

            fig_rat_bins.tight_layout(pad=0.6)

            if save_plot:
                self.save_plot(filename=f"All_Subjects_|_Rating_2-Class_Bins_|_SBA_|_{condition}")

    def plot_sba_hr(self, condition="NoMov", save_plot=False):
        """Plot heart rate vectors for each subject over all phases (SBA)"""
        # self.SBA_ecg["zSBA"]["NoMov"].shape

        fig_hr = plt.figure("All Subjects | Heart Rate | SBA", figsize=(14, 8))
        for sub in range(self.n_sub):
            plt.plot(interpolate_nan(arr_with_nan=self.SBA_ecg["zSBA"][condition][sub]), alpha=.5)
            # Note: this automat. overwrites self.SBA_ecg["zSBA"][condition][sub] with interpolated data
            # Normalization range(-1,1), Note: does not overwrite automatically
            # plt.plot(normalization(interpolate_nan(arr_with_nan=self.SBA_ecg["zSBA"][condition][sub]),
            #                        -1, 1), alpha=.5)
            # self.SBA_ecg["zSBA"][condition][sub] = normalization(self.SBA_ecg["zSBA"][condition][sub], -1, 1)

        mean_hr = np.nanmean(self.SBA_ecg["zSBA"][condition], axis=0)  # of interpolated data
        plt.plot(mean_hr, alpha=.8, c="black", lw=3, label="mean hr")

        hr_min = np.nanmin(self.SBA_ecg["zSBA"][condition])
        hr_max = np.nanmax(self.SBA_ecg["zSBA"][condition])
        ylims = np.max(np.abs((hr_min, hr_max)))

        for coaster in ["Space", "Andes"]:
            # Include events for roller coasters:
            if "Space" in coaster:
                events = np.genfromtxt(self.wdic + "space_events.csv", delimiter=",", dtype="|U18")
                # U18, 18 for max-length of str (n characters) of col_names
            else:  # elif "Andes" in coaster:
                events = np.genfromtxt(self.wdic + "ande_events.csv", delimiter=",", dtype="|U12")
                # U12, 12 for max-length of str (n characters) of col_names

            events = events[:, 1:]  # drop start=0

            subtractor = self.trim_time / 2 if self.trimmed else 0  # Events need to be shifted, if trim
            if "Andes" in coaster:
                subtractor -= self.trimmed_time_space + self.trimmed_time_break

            shift_counter = 0
            for idxe, event in enumerate(events[0, :]):
                shift_counter += 1
                if shift_counter > 4:  # reset
                    shift_counter = 1
                shift = 1 if shift_counter > 2 else 0  # switch between 1 and zero

                t_event = float(events[1, idxe]) - subtractor  # timepoint of event
                up_down = -1 if idxe % 2 == 0 else 1

                if up_down > 0:
                    y_max_value = np.min(mean_hr[int(t_event)])
                else:  # up_down < 0:
                    y_max_value = np.max(mean_hr[int(t_event)])

                plt.vlines(x=t_event, ymin=(ylims - shift) * up_down, ymax=y_max_value,
                           linestyles="dotted", alpha=1)
                plt.text(x=t_event, y=(ylims - shift) * up_down, s=event.replace("_", " "), size=6)

        plt.legend(loc="lower center")

        line = 0
        phase_names = [x.split("_")[0] for x in self.phases[-6:-3]]
        phase_names[-1] = phase_names[-1] + "s"
        plt.vlines(line, ymin=int(hr_min), ymax=hr_max, linestyles="--", alpha=0.5, lw=.8)
        # int(min) smaller

        for num, lines in enumerate(self.phase_lengths_int[-3:]):
            line += lines
            plt.vlines(line, ymin=int(hr_min), ymax=hr_max, linestyles="--", alpha=0.5, lw=.8)
            plt.text(x=line - lines, y=hr_max, s=phase_names[num], size=10)
        fig_hr.suptitle(f"HR of all Subjects ({condition} condition, SBA)")

        fig_hr.tight_layout(pad=0.6)

        if save_plot:
            self.save_plot(filename="All_Subjects_|_HR_|_SBA.png")

    def plot_sba_eeg(self, condition="NoMov", filetype="SSD", hilbert_power=True, band_pass=False,
                     sba=True,
                     save_plot=False):
        """Plot heart rate vectors for each subject over all phases (SBA)"""
        # self.SBA_ecg["zSBA"]["NoMov"].shape

        fs = 250  # sampling frequency

        # roller_coasters = self.roller_coaster

        eeg_data = load_component(subjects=list(self.subjects),
                                  condition=condition.lower(),
                                  f_type=filetype.upper(),
                                  band_pass=band_pass,
                                  sba=sba)

        len_runs = int(np.cumsum(t_roller_coasters(sba))[-1])
        for sub in self.subjects:
            if len(eeg_data[str(sub)]["SBA"][condition]) > 0:

                len_test = eeg_data[str(sub)]["SBA"][condition].shape[0] / fs - len_runs
                if len_test > 0.0:
                    to_delete = np.cumsum(t_roller_coasters(sba))  # [ 148.,  178.,  270.]
                    # 3 intersections where values can be removed (instead of cutting only at end/start)
                    to_delete *= fs
                    to_cut = int(round(len_test * fs))
                    print(f"EEG data of {s(sub)} trimmed by {to_cut} data points")
                    del_counter = 2
                    while len_test > 0.0:
                        if del_counter == -1:
                            del_counter = 2
                        # starts to delete in the end
                            eeg_data[str(sub)]["SBA"][condition] = np.delete(
                                arr=eeg_data[str(sub)]["SBA"][condition], obj=to_delete[del_counter],
                                axis=0)
                        del_counter -= 1
                        len_test = eeg_data[str(sub)]["SBA"][condition].shape[0] / fs - len_runs

                elif len_test < 0.0:
                    raise OverflowError("Eeg_sba file is too short. Implement interpolation function.")

        if hilbert_power:
            # print("I load Hilbert transformed data (z-power)")
            # Perform Hilbert Transformation only on first component
            for sub in self.subjects:
                # for comp in range(eeg_data[str(sub)]["SBA"][condition].shape[1]):
                if len(eeg_data[str(sub)]["SBA"][condition]) > 0:
                    for comp in [0]:
                        eeg_data[str(sub)]["SBA"][condition][:, comp] = \
                            calc_hilbert_z_power(array=eeg_data[str(sub)]["SBA"][condition][:, comp])

        fig_eeg = plt.figure(f"All Subjects | EEG {filetype} | SBA", figsize=(14, 8))

        comp = 0  # first component
        complete_count = 0
        mean_eeg = []
        for sub in self.subjects:
            if len(eeg_data[str(sub)]["SBA"][condition]) > 0:
                complete_count += 1
                plt.plot(eeg_data[str(sub)]["SBA"][condition][:, comp], alpha=.5)

                if complete_count == 1:
                    mean_eeg = eeg_data[str(sub)]["SBA"][condition][:, comp]
                    eeg_min = np.nanmin(eeg_data[str(sub)]["SBA"][condition][:, comp])
                    eeg_max = np.nanmax(eeg_data[str(sub)]["SBA"][condition][:, comp])
                else:
                    mean_eeg += eeg_data[str(sub)]["SBA"][condition][:, comp]
                    if np.nanmin(eeg_data[str(sub)]["SBA"][condition][:, comp]) < eeg_min:
                        eeg_min = np.nanmin(eeg_data[str(sub)]["SBA"][condition][:, comp])
                    if np.nanmax(eeg_data[str(sub)]["SBA"][condition][:, comp]) > eeg_max:
                        eeg_max = np.nanmax(eeg_data[str(sub)]["SBA"][condition][:, comp])

        mean_eeg /= complete_count
        plt.plot(mean_eeg, alpha=.8, c="black", lw=1, label="mean {}".format(filetype))

        mean_rating = np.nanmean(self.SBA_ratings["zSBA"][condition], axis=0)
        mean_rating = np.repeat(mean_rating, fs)
        mean_rating = normalization(mean_rating, lower_bound=eeg_min, upper_bound=eeg_max)
        mean_rating -= np.nanmean(mean_rating)

        plt.plot(mean_rating, alpha=.8, c="darkgrey", lw=1, ls="-.", label="mean rating")

        ylims = np.max(np.abs((eeg_min, eeg_max)))

        for coaster in ["Space", "Andes"]:
            # Include events for roller coasters:
            if "Space" in coaster:
                events = np.genfromtxt(self.wdic + "space_events.csv", delimiter=",", dtype="|U18")
                # U18, 18 for max-length of str (n characters) of col_names
            else:  # elif "Andes" in coaster:
                events = np.genfromtxt(self.wdic + "ande_events.csv", delimiter=",", dtype="|U12")
                # U12, 12 for max-length of str (n characters) of col_names

            events = events[:, 1:]  # drop start=0

            subtractor = self.trim_time / 2 if self.trimmed else 0
            # Events need to be shifted, if trimmed

            if "Andes" in coaster:
                subtractor -= self.trimmed_time_space + self.trimmed_time_break

            shift_counter = 0
            for idxe, event in enumerate(events[0, :]):
                shift_counter += 1
                if shift_counter > 4:  # reset
                    shift_counter = 1
                shift = 1 if shift_counter > 2 else 0  # switch between 1 and zero

                t_event = float(events[1, idxe]) - subtractor  # timepoint of event
                up_down = -1 if idxe % 2 == 0 else 1

                if up_down > 0:
                    y_max_value = np.min(mean_eeg[int(t_event)*fs])
                else:  # up_down < 0:
                    y_max_value = np.max(mean_eeg[int(t_event)*fs])

                plt.vlines(x=t_event*fs, ymin=(ylims - shift) * up_down, ymax=y_max_value,
                           linestyles="dotted", alpha=1)
                plt.text(x=t_event*fs, y=(ylims - shift) * up_down, s=event.replace("_", " "), size=6)

        plt.legend(loc="lower center")

        line = 0
        phase_names = [x.split("_")[0] for x in self.phases[-6:-3]]
        phase_names[-1] = phase_names[-1] + "s"
        plt.vlines(line, ymin=int(eeg_min), ymax=eeg_max, linestyles="--", alpha=0.5, lw=.8)
        # int(min) smaller

        for num, lines in enumerate(self.phase_lengths_int[-3:]):
            line += lines*fs
            plt.vlines(line, ymin=int(eeg_min), ymax=eeg_max, linestyles="--", alpha=0.5, lw=.8)
            plt.text(x=line - lines*fs, y=eeg_max, s=phase_names[num], size=10)
        fig_eeg.suptitle(f"First {filetype} component of all Subjects ({condition} condition, SBA)")

        fig_eeg.tight_layout(pad=0.6)
        fig_eeg.show()

        if save_plot:
            self.save_plot(filename=f"All_Subjects_|_EEG_{filetype}_|_SBA.png")

    def plot_hr(self, save_plot=False):
        """Plot HR for each subject over all phases"""
        for sub in range(self.n_sub):
            fig_sub = plt.figure("S{} | all_phases".format(str(self.subjects[sub]).zfill(2)),
                                 figsize=(14, 8))
            plt.plot(self.all_phases["all_phases"][sub])
            line = 0
            plt.vlines(line, ymin=40, ymax=150, linestyles="--", alpha=0.6)
            for num, lines in enumerate(self.phase_lengths_int):
                line += lines
                plt.vlines(line, ymin=40, ymax=150, linestyles="--", alpha=0.6)
                plt.text(x=line - lines, y=150, s=self.phases[num], size=4)
            sub_cond = str(self.subject_condition(self.subjects[sub]))[0]
            fig_sub.suptitle("HR of S{} (cond {}) over all phases".format(str(
                self.subjects[sub]).zfill(2), sub_cond))

            # Save plot
            if save_plot:
                self.save_plot(f"S{str(self.subjects[sub]).zfill(2)}_HR_all_phases")
        # close(n=n_sub)

    def plot_hr_z(self, save_plot=False):
        """Plot same as above but with z-scores of each subject (z-score across all phases)"""
        # Plot z-scored HR for each subject over all phases
        for sub in range(self.n_sub):
            fig_sub_z = plt.figure("S{} | all_phases_z".format(str(self.subjects[sub]).zfill(2)),
                                   figsize=(14, 8))
            plt.plot(self.all_phases["all_phases_z"][sub])
            line = 0
            plt.vlines(line, ymin=-10, ymax=10, linestyles="--", alpha=0.6)
            for num, lines in enumerate(self.phase_lengths_int):
                line += lines
                plt.vlines(line, ymin=-10, ymax=10, linestyles="--", alpha=0.6)
                plt.text(x=line-lines, y=10, s=self.phases[num], size=4)

            sub_cond = str(self.subject_condition(self.subjects[sub]))[0]
            fig_sub_z.suptitle("Z-scored HR of S{} (cond {}) over all phases".format(
                str(self.subjects[sub]).zfill(2), sub_cond))

            # Save Plot
            if save_plot:
                self.save_plot(f"S{str(self.subjects[sub]).zfill(2)}_HR_all_phases_z")

            # close(n=n_sub)

    def plot_hr_z_all(self, save_plot=False):
        # Plot z-scored HR for each subject over all phases (in single plot)
        fig_sub_z_single = plt.figure("all_phases_z", figsize=(14, 8))
        # fig_sub_z_single.legend(handles="upper center", labels="blabal")
        for sub in range(self.n_sub):
            plt.plot(self.all_phases["all_phases_z"][sub])
        plt.plot(np.nanmean(self.all_phases["all_phases_z"], axis=0), color="black", linewidth=2)
        # mean z-score HR

        line = 0
        plt.vlines(line, ymin=-10, ymax=10, linestyles="--", alpha=0.6)
        for num, lines in enumerate(self.phase_lengths_int):
            line += lines
            plt.vlines(line, ymin=-10, ymax=10, linestyles="--", alpha=0.6)
            plt.text(x=line-lines, y=10, s=self.phases[num], size=4)
        fig_sub_z_single.suptitle("Z-scored HR for each subject over all phases + mean")

        # Save Plot
        if save_plot:
            self.save_plot("All_S_HR_all_phases_z")
        # close()

    def plot_smoothed(self, save_plot=False):
        """Plot smoothed data"""
        # Plot HR for each subject over all phases
        for sub in range(self.n_sub):
            fig_sub_smooth = plt.figure(f"S{str(self.subjects[sub]).zfill(2)} | all_phases_smoothed",
                                        figsize=(14, 8))
            plt.plot(self.all_phases["all_phases"][sub], alpha=0.3)  # can be taken out
            plt.plot(self.all_phases["all_phases_smooth"][sub])
            line = 0
            plt.vlines(line, ymin=40, ymax=150, linestyles="--", alpha=0.6)
            for num, lines in enumerate(self.phase_lengths_int):
                line += lines
                plt.vlines(line, ymin=40, ymax=150, linestyles="--", alpha=0.6)
                plt.text(x=line-lines, y=150, s=self.phases[num], size=4)

            fig_sub_smooth.suptitle("smoothed({}dpts.) HR of S{} "
                                    "over all phases".format(self.w_size,
                                                             str(self.subjects[sub]).zfill(2)))

            # Save Plot
            if save_plot:
                self.save_plot(f"S{str(self.subjects[sub]).zfill(2)}_all_phases_smoothed")
        # close(n=n_sub)

    def plot_hr_z_smoothed(self, save_plot=False):
        """Plot smoothed z-scored HR for each subject over all phases"""
        for sub in range(self.n_sub):
            fig_sub_z_smooth = plt.figure("S{} | all_phases_z".format(str(self.subjects[sub]).zfill(2)),
                                          figsize=(14, 8))
            plt.plot(self.all_phases["all_phases_z"][sub], alpha=0.3)  # can be taken out
            plt.plot(self.all_phases["all_phases_z_smooth"][sub])
            line = 0
            plt.vlines(line, ymin=-10, ymax=10, linestyles="--", alpha=0.6)
            for num, lines in enumerate(self.phase_lengths_int):
                line += lines
                plt.vlines(line, ymin=-10, ymax=10, linestyles="--", alpha=0.6)
                plt.text(x=line-lines, y=10, s=self.phases[num], size=4)

            sub_cond = str(self.subject_condition(self.subjects[sub]))[0]
            fig_sub_z_smooth.suptitle("smoothed({}dpts.) z-scored HR of S{} (cond {}) "
                                      "over all phases".format(self.w_size,
                                                               str(self.subjects[sub]).zfill(2),
                                                               sub_cond))

            if save_plot:
                self.save_plot(f"S{str(self.subjects[sub]).zfill(2)}_all_phases_z_smoothed")
        # close(n=n_sub)

    def plot_hr_z_smoothed_all(self, save_plot=False):
        """Plot z-scored HR for each subject over all phases (in single plot)"""
        fig_sub_z_single_smooth = plt.figure("all_phases_z_smoothed", figsize=(14, 8))
        # fig_sub_z_single.legend(handles="upper center", labels="blabal")
        for sub in range(self.n_sub):
            plt.plot(self.all_phases["all_phases_z_smooth"][sub])
        plt.plot(np.nanmean(self.all_phases["all_phases_z_smooth"], axis=0),
                 color="black",
                 linewidth=2)  # mean z-score HR
        plt.plot(np.nanmean(self.all_phases["all_phases_z"], axis=0),
                 color="black",
                 linewidth=2,
                 alpha=0.3)  # can be taken out
        line = 0
        plt.vlines(line, ymin=-10, ymax=10, linestyles="--", alpha=0.6)
        for num, lines in enumerate(self.phase_lengths_int):
            line += lines
            plt.vlines(line, ymin=-10, ymax=10, linestyles="--", alpha=0.6)
            plt.text(x=line-lines, y=10, s=self.phases[num], size=4)
        fig_sub_z_single_smooth.suptitle(
            f"smoothed({self.w_size}dpts.) z-scored HR for each subject over all phases + mean")

        if save_plot:
            self.save_plot("All_S_all_phases_z_smoothed")

        # close()

        # plt.plot(self.all_phases["all_phases"][0, :])
        # plt.plot(self.all_phases["all_phases_z"][0, :])

        # plt.plot(self.all_phases["all_phases_z"][0, :])
        # plt.plot(self.all_phases["all_phases_z_smooth"][0, :])

    def cross_cor(self, save_plot=False, sba=True, maxlag=10):
        """
        Cross-cor z-ratings with corresponding z-HR of each phase
        :param maxlag: Max Lag of x-corr [default=10, see e.g., Kettunen et al, 2000; Mauss et al, 2005]
        :param save_plot: Whether to save plots
        :param sba: if True, takes the z-scored values of SBA, otherwise z-scored for each trial
        """

        for sub_idx, sub in enumerate(self.subjects):

            plt.figure("S{} | z-HR and z-Rating | xcorr | 1Hz".format(str(sub).zfill(2)), figsize=(8, 10))
            subplot_nr = 420
            ylims = 0

            if sba:  # z-scored over "space-break-ande" (SBA)
                # z-score heart rate as was done for ratings (concatenate two coasters+break then z-score)
                conditions = [key for key in self.SBA_ecg["zSBA"].keys()]  # ['NoMov', 'Mov']
                for cond in conditions:
                    if np.abs(int(np.nanmin(self.SBA_ecg["zSBA"][cond])-1)) > ylims:
                        ylims = np.abs(int(np.nanmin(self.SBA_ecg["zSBA"][cond])) - 1)
                    if np.int(np.nanmax(self.SBA_ecg["zSBA"][cond])) + 1 > ylims:
                        ylims = np.int(np.nanmax(self.SBA_ecg["zSBA"][cond])) + 1

            else:  # z-scored for single phases (Caution: so far Ratings are SBA-z-scored)
                # Found min/max values over all roller coasters of zHR
                for coaster in self.roller_coaster:
                    if np.abs(int(np.nanmin(self.each_phases_z[coaster])) - 1) > ylims:
                        ylims = np.abs(int(np.nanmin(self.each_phases_z[coaster])) - 1)
                    elif np.int(np.nanmax(self.each_phases_z[coaster])) + 1 > ylims:
                        ylims = np.int(np.nanmax(self.each_phases_z[coaster])) + 1

            for coaster in self.roller_coaster:  # ['Space_NoMov', 'Space_Mov', 'Ande_Mov', 'Ande_NoMov']
                if not sba:
                    if self.each_phases_z[coaster][sub_idx].shape != \
                            self.ratings_dic[coaster][sub_idx].shape:
                        print("Data length not the same | {} | S{}.".format(coaster, str(sub).zfill(2)))

                if sba:
                    # Create keys for self.SBA_ecg_split["zSBA][cond][phase]
                    phase, cond = coaster.split("_")  # e.g., ['Space', 'NoMov']
                    phase = phase.lower()  # e.g., "Space" ==> "space"

                var1 = copy.copy(self.ratings_dic[coaster][sub_idx])
                var2 = copy.copy(self.SBA_ecg_split["zSBA"][cond][phase][sub_idx, :]) if sba \
                    else copy.copy(self.each_phases_z[coaster][sub_idx])
                var1[np.isnan(var2)] = 0.  # x-corr does not work with nan, so set nan = zero
                var2[np.isnan(var2)] = 0.

                # Plot
                subplot_nr += 1
                plt.subplot(subplot_nr)

                plt.plot(var1, label="z-Ratings")  # Ratings
                plt.plot(var2, label="z-HR")  # HR
                plt.ylim(-ylims, ylims)

                # Include events for roller coasters:
                if "Space" in coaster:
                    events = np.genfromtxt(self.wdic + "space_events.csv", delimiter=",", dtype="|U18")
                    # U18, 18 for max-length of str (n characters) of col_names
                else:  # elif "Ande" in coaster:
                    events = np.genfromtxt(self.wdic + "ande_events.csv", delimiter=",", dtype="|U12")
                    # U12, 12 for max-length of str (n characters) of col_names

                events = events[:, 1:]  # drop start=0

                subtractor = self.trim_time/2 if self.trimmed else 0
                # Events need to be shifted, if trimmed

                shift_counter = 0
                for idxe, event in enumerate(events[0, :]):
                    shift_counter += 1
                    if shift_counter > 4:  # reset
                        shift_counter = 1
                    shift = 1 if shift_counter > 2 else 0  # switch between 1 and zero

                    t_event = float(events[1, idxe]) - subtractor  # timepoint of event
                    up_down = -1 if idxe % 2 == 0 else 1

                    if up_down > 0:
                        y_max_value = np.min((var1[int(t_event)], var2[int(t_event)]))
                    else:  # up_down < 0:
                        y_max_value = np.max((var1[int(t_event)], var2[int(t_event)]))

                    plt.vlines(x=t_event, ymin=(ylims-shift)*up_down, ymax=y_max_value,
                               linestyles="dotted", alpha=0.3)
                    plt.text(x=t_event, y=(ylims-shift)*up_down, s=event.replace("_", " "), size=6)

                plt.legend()

                # plt.figure("S{} in {} | HR, Rating".format(str(sub).zfill(2), coaster))
                # plt.plot(var1, var2, "o")
                # plt.xlabel("z_Ratings")
                # plt.ylabel("z_HR")
                # plt.legend()

                subplot_nr += 1
                plt.subplot(subplot_nr)
                plt.xcorr(var1, var2, maxlags=maxlag)
                plt.ylim(-0.9, 0.9)
                plt.text(x=0-maxlag/1.5,
                         y=0.6,
                         s=f"S{str(sub).zfill(2)} in {coaster} | "
                           f"cond {str(self.subject_condition(sub))[0]} | xcorr")

                # plt.figure("S{} in {} | np.correlate".format(str(sub).zfill(2), coaster))
                # plt.plot(np.correlate(var1, var2, mode=2))  # mode = "full"

            plt.tight_layout()

            if save_plot:
                self.save_plot(f"S{str(sub).zfill(2)}_z_Ratings_HR_x_corr")

        # np.correlate(var1, var2, "full")
        # from scipy.stats.stats import pearsonr
        # print(pearsonr(var1, var1))

    def cross_cor_sba(self, save_plot=False, maxlag=20):
        """
        Plots the cross-correlations of SBA
        :param save_plot: Save plot yes or no
        :param maxlag: Maximal Lag of cross-correlation
        """

        for sub_idx, sub in enumerate(self.subjects):

            plt.figure("S{} | SBA | z-HR and z-Rating | xcorr | 1Hz ".format(str(sub).zfill(2)),
                       figsize=(12, 8))
            subplot_nr = 220
            ylims = 0

            # z-scored over "space-break-ande" (SBA)
            # Finding the max ylims
            conditions = [key for key in self.SBA_ecg["zSBA"].keys()]  # ['NoMov', 'Mov']
            for cond in conditions:
                if np.abs(int(np.nanmin(self.SBA_ecg["zSBA"][cond]) - 1)) > ylims:
                    ylims = np.abs(int(np.nanmin(self.SBA_ecg["zSBA"][cond])) - 1)
                if np.int(np.nanmax(self.SBA_ecg["zSBA"][cond])) + 1 > ylims:
                    ylims = np.int(np.nanmax(self.SBA_ecg["zSBA"][cond])) + 1

            # Create keys for self.SBA_ecg_split["zSBA][cond][phase]

            for cond in conditions:
                # self.SBA_ecg["zSBA"][cond].shape

                var1 = copy.copy(self.SBA_ratings["zSBA"][cond][sub_idx, :])
                var2 = copy.copy(self.SBA_ecg["zSBA"][cond][sub_idx, :])
                var1[np.isnan(var2)] = 0.  # x-corr does not work with nan, so set nan = zero
                var2[np.isnan(var2)] = 0.

                # Plot
                subplot_nr += 1
                plt.subplot(subplot_nr)

                plt.plot(var1, label="z-Ratings")  # Ratings
                plt.plot(var2, label="z-HR")  # HR
                plt.ylim(-ylims, ylims)

                # Include events for roller coasters:
                events_space = np.genfromtxt(self.wdic + "space_events.csv", delimiter=",", dtype="|U18")
                events_ande = np.genfromtxt(self.wdic + "ande_events.csv", delimiter=",", dtype="|U12")

                events_space = events_space[:, 1:]  # drop start=0
                events_ande = events_ande[:, 1:]

                subtractor = self.trim_time/2 if self.trimmed else 0
                # Events need to be shifted, if trimmed

                # Plot events.
                shift_counter = 0
                # First for Space
                for idxe, event in enumerate(events_space[0, :]):
                    shift_counter += 1
                    if shift_counter > 4:  # reset
                        shift_counter = 1
                    shift = 1 if shift_counter > 2 else 0  # switch between 1 and zero

                    t_event = float(events_space[1, idxe]) - subtractor  # time point of event
                    up_down = -1 if idxe % 2 == 0 else 1

                    if up_down > 0:
                        y_max_value = np.min((var1[int(t_event)], var2[int(t_event)]))
                    else:  # up_down < 0:
                        y_max_value = np.max((var1[int(t_event)], var2[int(t_event)]))

                    plt.vlines(x=t_event, ymin=(ylims-shift)*up_down, ymax=y_max_value, linestyles="dotted",
                               alpha=0.3)
                    plt.text(x=t_event, y=(ylims-shift)*up_down, s=event.replace("_", " "), size=6)

                # Now for Ande
                for idxe, event in enumerate(events_ande[0, :]):
                    shift_counter += 1
                    if shift_counter > 4:  # reset
                        shift_counter = 1
                    shift = 1 if shift_counter > 2 else 0  # switch between 1 and zero

                    t_event = float(events_ande[1, idxe]) - \
                              subtractor+self.trimmed_time_break+self.trimmed_time_space
                    up_down = -1 if idxe % 2 == 0 else 1

                    if up_down > 0:
                        y_max_value = np.min((var1[int(t_event)], var2[int(t_event)]))
                    else:  # up_down < 0:
                        y_max_value = np.max((var1[int(t_event)], var2[int(t_event)]))

                    plt.vlines(x=t_event, ymin=(ylims - shift) * up_down, ymax=y_max_value,
                               linestyles="dotted", alpha=0.3)
                    plt.text(x=t_event, y=(ylims - shift) * up_down, s=event.replace("_", " "), size=6)

                # Plot boarders between Coasters and break

                plt.vlines(x=self.trimmed_time_space, ymax=ylims, ymin=-ylims, linestyles="--", alpha=0.1)
                plt.vlines(x=self.trimmed_time_space+self.trimmed_time_break,
                           ymax=ylims, ymin=-ylims, linestyles="--", alpha=0.1)

                plt.legend()

                subplot_nr += 1
                plt.subplot(subplot_nr)
                plt.xcorr(var1, var2, maxlags=maxlag)
                plt.ylim(-0.9, 0.9)
                plt.text(x=0-maxlag/1.5,
                         y=0.6,
                         s=f"S{str(sub).zfill(2)} in {cond} | cond {str(self.subject_condition(sub))[0]} "
                           f"| xcorr")

            plt.tight_layout()

            if save_plot:
                self.save_plot(f"S{str(sub).zfill(2)}_z_Ratings_HR_x_corr")


#%% Main >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

if __name__ == "__main__":

    # Dropouts recording time + ICA prep + SSD selected (< 4)
    # dropouts = [1, 12, 32, 33, 35, 38, 41, 45]  # recording time

    # Dropouts after SSD selection
    mov_dropouts = [1, 3, 5, 7, 9, 10, 11, 12, 13, 15, 16, 17, 19, 20, 23, 24, 26, 27, 30,
                    32, 33, 38, 40, 41, 43, 45]  # mov
    nomov_dropouts = [1, 7, 9, 10, 12, 16, 19, 20, 23, 24, 27, 31, 32, 33, 38, 40, 41, 43, 45]  # nomov

    for idx, drops in enumerate([mov_dropouts, nomov_dropouts]):

        ec = NeVRoPlot(n_sub=45,
                       dropouts=drops,
                       subject_selection=[6, 11, 14, 17, 20, 27, 31, 34, 36],
                       smooth_w_size=21)

        # # Some descriptive stats on non-z-scored ratings:
        if idx == 0:
            mean_rat_across_subs_mov = np.nanmean(ec.SBA_ratings["SBA"][["Mov", "NoMov"][idx]], axis=0)
        else:
            mean_rat_across_subs_nomov = np.nanmean(ec.SBA_ratings["SBA"][["Mov", "NoMov"][idx]], axis=0)

        mean_rat_across_subs = np.nanmean(ec.SBA_ratings["SBA"][["Mov", "NoMov"][idx]], axis=0)
        print(f"\nRating in {['Mov', 'NoMov'][idx]}-Condition:\n"
              f"Mean:\t\t {np.nanmean(mean_rat_across_subs):.2f}\n"
              f"STD:\t\t {np.nanstd(mean_rat_across_subs):.2f}\n"
              f"MIN-MAX:\t {np.nanmin(mean_rat_across_subs):.2f} – {np.nanmax(mean_rat_across_subs):.2f}")

    # # Explore difference of ratings between conditions
    from scipy.stats import ttest_rel
    pt = ttest_rel(a=mean_rat_across_subs_mov, b=mean_rat_across_subs_nomov, nan_policy="omit")
    dgf = (len(mean_rat_across_subs_mov) + len(mean_rat_across_subs_nomov))/2 - 1  # degrees of freedom

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
    fig.suptitle(f"Difference of mean-ratings (mov - nomov),  "
                 f"t({int(dgf)})={pt.statistic:.3f}, p={pt.pvalue:.2e}")

    # Difference at each time point
    ax1.plot(mean_rat_across_subs_mov - mean_rat_across_subs_nomov, lw=2, color="red",
             label="mean rating (mov-nomov)")
    ax1.hlines(y=0, xmin=0, xmax=len(mean_rat_across_subs_mov), alpha =.5, linestyles="-.")
    ax1.legend()
    [ax1.spines[corner].set_visible(False) for corner in ax1.spines.keys()]

    # Histograms
    ax2.hist(mean_rat_across_subs_mov, bins=100, color="orange")  # , label="mov")
    ax2.hist(mean_rat_across_subs_nomov, bins=100, color="lightblue", alpha=.8)  # , label="nomov")
    ax2.vlines(x=np.nanmean(mean_rat_across_subs_mov), ymin=0, ymax=16, color="darkorange", lw=2,
               linestyles="--", label="mean(mov)")
    ax2.vlines(x=np.nanmean(mean_rat_across_subs_nomov), ymin=0, ymax=16, color="cornflowerblue", lw=2,
               linestyles="--", label="mean(nomov)")
    ax2.legend()
    # ax2.axis('off')
    ax2.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')
    [ax2.spines[corner].set_visible(False) for corner in ax2.spines.keys()]  # plt.box(on=False)
    fig.tight_layout()

# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<  END

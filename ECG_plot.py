"""
Plot ECG (RR and HR) Trajectories
    • indicate different phases
    • plot RR intervals
    • plot HR (heart beats per minute, 1/RR)
    • save plots

Author: Simon Hofmann | <[surname].[lastname][at]protonmail.com> | 2017
"""

import numpy as np
import os.path
import matplotlib.pyplot as plt
import copy


class ECGplot:

    def __init__(self, n_sub=45, dropouts=[1, 12, 32, 33, 35, 38, 41, 42, 45], subject_selection=[], smooth_w_size=3):

        # Change to folder which contains files
        self.wdic = "../../Data/"
        self.wdic_plots = "../../Data/Plots/"
        self.wdic_cropRR = "../../Data/ECG/TR_cropped/"
        self.wdic_Rating = "../../Data/Ratings/"
        # Set variables
        self.n_sub = n_sub  # number of all Subjects
        self.subjects = np.arange(1, n_sub+1)  # Array of all subjects
        # Define array of dropouts (subject ID)
        self.dropouts = dropouts
        # Select specific subjects of interest
        self.subject_selection = np.array(subject_selection)
        self.n_sub, self.subjects = self.subjects_request(subjects_array=self.subjects,
                                                          dropouts_array=self.dropouts,
                                                          selected_array=self.subject_selection)

        self.phases = ["Resting_Open", "Resting_Close",
                       "Space_Mov", "Break_Mov", "Ande_Mov",
                       "Rating_Space_Mov", "Rating_Ande_Mov",
                       "Space_NoMov", "Break_NoMov", "Ande_NoMov",
                       "Rating_Space_NoMov", "Rating_Ande_NoMov"]

        # Get phase-times (=max length of phases of all subjects)
        self.phase_lengths = np.zeros((len(self.phases),))
        self._update_phase_lengths()
        self.phase_lengths_int = np.array([int(self.phase_lengths[i])+1 for i in range(len(self.phase_lengths))])
        # For each phase plot RR/HR trajectories per subject and mean across subjects (RR/HR) bold

        self.each_phases_z = {"Resting_Open": [],
                              "Resting_Close": [],
                              "Space_Mov": [],
                              "Break_Mov": [],
                              "Ande_Mov": [],
                              "Rating_Space_Mov": [],
                              "Rating_Ande_Mov": [],
                              "Space_NoMov": [],
                              "Break_NoMov": [],
                              "Ande_NoMov": [],
                              "Rating_Space_NoMov": [],
                              "Rating_Ande_NoMov": []}

        self.all_phases = copy.copy(self.each_phases_z)
        self._concat_all_phases()
        self._create_z_scores()

        self.w_size = smooth_w_size  # McCall has a window of 3 (see: SCRIPT_paperAnalyses_HR.m)
        self._create_smoothing()

        self.ratings_dic = {}
        self._update_ratings_dic()
        self.roller_coaster = np.array(['Space_NoMov', 'Space_Mov', 'Ande_Mov', 'Ande_NoMov'])

    def close(self, n=1):
        for _ in range(n):
            plt.close()

    def save_plot(self, filename):
        plt.savefig(self.wdic_plots+filename)
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

    def subjects_request(self, subjects_array, dropouts_array, selected_array):
        print("Do you want to include all subjects or only the subjects with complete datasets for plotting")
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
            # subjects_array = np.array(list(set(subjects_array) ^ set(dropouts_array)))  # elegant, 2.1
            # subjects_array = np.array(list(set(subjects_array).symmetric_difference(set(dropouts_array))))  #2.2
            # subjects_array = np.array(list(set(subjects_array) - set(dropouts_array)))  # elegant, 3.1
            # subjects_array = np.array(list(set(subjects_array).difference(set(dropouts_array))))  # 3.2
            # for drop in dropouts_array:  # not so elegant
            #     subjects_array = np.delete(subjects_array, np.where(subjects_array == drop))
            print("Create subset of all subjects without dropouts")
        elif set_subset == 3:
            subjects_array = selected_array
            print("Create subset of selected subjects")

        num_sub = len(subjects_array)

        return num_sub, subjects_array

    def plot_request(self):
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
        for sub in self.subjects:
            for num, phase in enumerate(self.phases):
                TR_file_name = self.wdic_cropRR + "NVR_S{}_{}_T_R.txt".format(str(sub).zfill(2), phase)
                if os.path.isfile(TR_file_name):
                    TR_file = np.loadtxt(TR_file_name)
                    self.phase_lengths[num] = TR_file[-1] if TR_file[-1] > self.phase_lengths[num] \
                        else self.phase_lengths[num]

        print("Length of each phase:\n", self.phase_lengths)

    def _update_ratings_dic(self):
        """Load z-rating files and write them in ratings_dic"""
        len_space = self.all_phases["Space_Mov"].shape[1]  # same length for NoMov
        len_ande = self.all_phases["Ande_Mov"].shape[1]  # same length for NoMov
        space_init = np.reshape(np.repeat(np.nan, self.n_sub * len_space), newshape=(self.n_sub, len_space))
        ande_init = np.reshape(np.repeat(np.nan, self.n_sub * len_ande), newshape=(self.n_sub, len_ande))

        self.ratings_dic = {"Space_Mov": copy.copy(space_init),
                            "Ande_Mov": copy.copy(ande_init),
                            "Space_NoMov": copy.copy(space_init),
                            "Ande_NoMov": copy.copy(ande_init)}

        r_coasters = ["space", "andes"]
        runs = [1, 2]

        table_of_condition = np.genfromtxt(self.wdic + "Table_of_Conditions.csv", delimiter=";")
        table_of_condition = table_of_condition[1:, ]  # remove first column (sub-nr, condition, gender (1=f, 2=m))

        for sub_idx, sub in enumerate(self.subjects):
            for num, coaster in enumerate(r_coasters):
                for r in runs:
                    rating_filename = self.wdic_Rating + "/{}/{}_z/1Hz/NVR_S{}_run_{}_{}_rat_z.txt".format(coaster,
                                                                                                           coaster,
                                                                                                           str(sub).zfill(2),
                                                                                                           str(r),
                                                                                                           coaster)
                    if os.path.isfile(rating_filename):
                        rating_file = np.genfromtxt(rating_filename, delimiter=',')[:, 1]  # only load col with ratings

                        # Fill in right slot of ratings_dic
                        ratings_key = ''
                        ratings_key += "Space_" if coaster == "space" else "Ande_"
                        sub_cond = int(table_of_condition[np.where(table_of_condition[:, 0] == sub), 1])  # cond of sub
                        mov = "Mov" if (sub_cond == 12 and r == 1) or (sub_cond == 21 and r == 2) else "NoMov"
                        ratings_key += mov
                        # print("ratings_key:", ratings_key)
                        if self.ratings_dic[ratings_key][sub_idx, :].shape[0] != len(rating_file):
                            raise ValueError("Rating_file: '{}' has not same length as HR_file!".format(rating_filename))
                        # print("Update Subject{}:".format(str(sub)), ratings_key)
                        self.ratings_dic[ratings_key][sub_idx, :] = copy.copy(rating_file)

                    else:
                        print(rating_filename, " does not exist")

    def _concat_all_phases(self):
        """Concatenate all phases per subject"""
        all_length = 0
        for phase in self.phase_lengths:
            all_length += (int(phase)+1)
        all_phase = np.reshape(np.repeat(np.nan, self.n_sub * all_length), newshape=(self.n_sub, all_length))
        # Update dic
        self.all_phases.update({"all_phases": copy.copy(all_phase)})
        self.all_phases.update({"all_phases_z": copy.copy(all_phase)})

        plotten, save_plots = self.plot_request()  # True or False

        for num, phase in enumerate(self.phases):

            # Update dict: all_phase_z
            phase_table = np.reshape(np.repeat(np.nan, (int(self.phase_lengths[num])+1) * self.n_sub),
                                     newshape=(self.n_sub, -1))
            self.each_phases_z[phase] = copy.copy(phase_table)
            self.all_phases[phase] = copy.copy(phase_table)

            if plotten:
                xlim = self.phase_lengths[num]
                fig_phase = plt.figure("z-HR in {}".format(phase), figsize=(14, 8))
                # fig_phase.legend(labels=phase, handles=...)

            for sub_idx, sub in enumerate(self.subjects):
                TR_file_name = self.wdic_cropRR + "NVR_S{}_{}_T_R.txt".format(str(sub).zfill(2), phase)
                if os.path.isfile(TR_file_name):
                    TR_file = np.loadtxt(TR_file_name)
                    RR_file = np.array([TR_file[i]-TR_file[i-1] for i in range(1, len(TR_file))])
                    HR_file = 60/RR_file  # HR (B/min)

                    # downsample HR file to 1Hz
                    HR_file_ds = np.zeros(shape=(int(TR_file[-1]) + 1,))
                    for i in range(len(HR_file_ds)):
                        # bender = len(HR_file_ds)/len(HR_file)
                        if len(TR_file[TR_file < i]) > 1:
                            RR_int_i = (TR_file[TR_file < i][-1] - TR_file[TR_file < i][-2])
                            HR_i = 60/RR_int_i
                            HR_file_ds[i] = HR_i

                    # calculate z_score
                    HR_file_ds[np.where(HR_file_ds == 0)] = np.nan  # replace zeros with NaN
                    z_HR_file_ds = HR_file_ds - np.nanmean(HR_file_ds)
                    z_HR_file_ds /= np.nanstd(HR_file_ds)
                    # Fill in each_phases_z:
                    for idx, item in enumerate(z_HR_file_ds):
                        self.each_phases_z[phase][sub_idx, idx] = item

                    for idx2, item2 in enumerate(HR_file_ds):
                        self.all_phases[phase][sub_idx, idx2] = item2

                    # plot
                    if plotten:
                        plt.plot(z_HR_file_ds)  # z_score
                    # print("z_HR_file_ds.shape", z_HR_file_ds.shape)
                    # plt.plot(HR_file)
                if plotten:
                    mean_z = np.nanmean(self.each_phases_z[phase], axis=0)  # z_score-mean
                    plt.plot(mean_z, color="black", linewidth=2)  # plot z_score-mean
                    if sub_idx == self.n_sub-1:
                        y_min_max = 10
                        plt.vlines(0, ymin=-y_min_max, ymax=y_min_max, linestyles="--", alpha=0.6)
                        plt.vlines(xlim, ymin=-y_min_max, ymax=y_min_max, linestyles="--", alpha=0.6)

                        # Include events for roller coasters:
                        if "Space" in phase:
                            events = np.genfromtxt(self.wdic + "space_events.csv", delimiter=",", dtype="|U18")
                            # U18, 18 for max-length of str (n characters) of col_names
                            events = events[:, 1:]  # drop start=0

                            for idxe, event in enumerate(events[0, :]):
                                t_event = int(events[1, idxe])  # timepoint of event
                                shift = 0 if idxe % 2 == 0 else 1
                                plt.vlines(x=t_event, ymin=-(y_min_max-shift), ymax=mean_z[t_event],
                                           linestyles="dotted", alpha=0.3)
                                plt.text(x=t_event, y=-(y_min_max-shift), s=event)

                        elif "Ande" in phase:

                            events = np.genfromtxt(self.wdic + "ande_events.csv", delimiter=",", dtype="|U12")
                            # U12, 12 for max-length of str (n characters) of col_names
                            events = events[:, 1:]  # drop start=0

                            for idxe, event in enumerate(events[0, :]):
                                t_event = int(events[1, idxe])  # timepoint of event
                                shift = 0 if idxe % 2 == 0 else 1
                                plt.vlines(x=t_event, ymin=-(y_min_max-shift), ymax=mean_z[t_event],
                                           linestyles="dotted", alpha=0.3)
                                plt.text(x=t_event, y=-(y_min_max-shift), s=event)

                    fig_phase.suptitle("z-scored HR of all subjects in phase: {}".format(phase))

            if plotten:
                if save_plots:
                    self.save_plot("All_S_z_HR_in_{}".format(phase))

        # close(n=len(phases))

        # Concatenate all phases to one array per subject
        for sub in range(self.n_sub):
            idx = 0
            old_idx = 0

            for num, phase in enumerate(self.phases):
                # phase_lengths_int[num] == all_phases[phase][sub].shape[0]
                idx += self.all_phases[phase][sub].shape[0]
                self.all_phases["all_phases"][sub][old_idx: idx] = self.all_phases[phase][sub]
                if num < 11:
                    old_idx = idx

    def _create_z_scores(self):
        # Create z-scores
        for sub in range(self.n_sub):
            # print("subject:", subjects[sub])
            sub_mean = np.nanmean(self.all_phases["all_phases"][sub])
            sub_std = np.nanstd(self.all_phases["all_phases"][sub])
            self.all_phases["all_phases_z"][sub] = (copy.copy(self.all_phases["all_phases"][sub]) - sub_mean) / sub_std

    def _create_smoothing(self):
        # smoothing the HR: try 3data-point sliding-window [1,2,3] average and write over last value
        # Update dic
        self.all_phases.update({"all_phases_smooth": copy.copy(self.all_phases["all_phases"])})
        self.all_phases.update({"all_phases_z_smooth": copy.copy(self.all_phases["all_phases_z"])})
        # all_phases.keys()  # 'all_phases', 'all_phases_z'

        s_modes = ["ontop", "hanging"]
        s_mode = s_modes[0]  # can be changed
        # w_size = 3  # can be changed, w_size € [3, 5, 11, 21]
        for i in range(self.n_sub):
            self.all_phases["all_phases_smooth"][i, :] = self.smooth(array_to_smooth=self.all_phases["all_phases"][i, :],
                                                                   sliding_mode=s_mode)
            self.all_phases["all_phases_z_smooth"][i, :] = self.smooth(
                array_to_smooth=self.all_phases["all_phases_z"][i, :],
                sliding_mode=s_mode)

    def plot_HR(self, save_plot=False):
        """Plot HR for each subject over all phases"""
        for sub in range(self.n_sub):
            fig_sub = plt.figure("S{} | all_phases".format(str(self.subjects[sub]).zfill(2)), figsize=(14, 8))
            plt.plot(self.all_phases["all_phases"][sub])
            line = 0
            plt.vlines(line, ymin=40, ymax=150, linestyles="--", alpha=0.6)
            for num, lines in enumerate(self.phase_lengths_int):
                line += lines
                plt.vlines(line, ymin=40, ymax=150, linestyles="--", alpha=0.6)
                plt.text(x=line - lines, y=150, s=self.phases[num], size=4)

            fig_sub.suptitle("HR of S{} over all phases".format(str(self.subjects[sub]).zfill(2)))

            # Save plot
            if save_plot:
                self.save_plot("S{}_HR_all_phases".format(str(self.subjects[sub]).zfill(2)))
        # close(n=n_sub)

    def plot_HR_z(self, save_plot=False):
        """Plot same as above but with z-scores of each subject (z-score across all phases)"""
        # Plot z-scored HR for each subject over all phases
        for sub in range(self.n_sub):
            fig_sub_z = plt.figure("S{} | all_phases_z".format(str(self.subjects[sub]).zfill(2)), figsize=(14, 8))
            plt.plot(self.all_phases["all_phases_z"][sub])
            line = 0
            plt.vlines(line, ymin=-10, ymax=10, linestyles="--", alpha=0.6)
            for num, lines in enumerate(self.phase_lengths_int):
                line += lines
                plt.vlines(line, ymin=-10, ymax=10, linestyles="--", alpha=0.6)
                plt.text(x=line-lines, y=10, s=self.phases[num], size=4)

            fig_sub_z.suptitle("Z-scored HR of S{} over all phases".format(str(self.subjects[sub]).zfill(2)))

            # Save Plot
            if save_plot:
                self.save_plot("S{}_HR_all_phases_z".format(str(self.subjects[sub]).zfill(2)))

            # close(n=n_sub)

    def plot_HR_z_all(self, save_plot=False):
        # Plot z-scored HR for each subject over all phases (in single plot)
        fig_sub_z_single = plt.figure("all_phases_z", figsize=(14, 8))
        # fig_sub_z_single.legend(handles="upper center", labels="blabal")
        for sub in range(self.n_sub):
            plt.plot(self.all_phases["all_phases_z"][sub])
        plt.plot(np.nanmean(self.all_phases["all_phases_z"], axis=0), color="black", linewidth=2)  # mean z-score HR
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
            fig_sub_smooth = plt.figure("S{} | all_phases_smoothed".format(str(self.subjects[sub]).zfill(2)), figsize=(14, 8))
            plt.plot(self.all_phases["all_phases"][sub], alpha=0.3)  # can be taken out
            plt.plot(self.all_phases["all_phases_smooth"][sub])
            line = 0
            plt.vlines(line, ymin=40, ymax=150, linestyles="--", alpha=0.6)
            for num, lines in enumerate(self.phase_lengths_int):
                line += lines
                plt.vlines(line, ymin=40, ymax=150, linestyles="--", alpha=0.6)
                plt.text(x=line-lines, y=150, s=self.phases[num], size=4)

            fig_sub_smooth.suptitle("smoothed({}dpts.) HR of S{} over all phases".format(self.w_size,
                                                                                       str(self.subjects[sub]).zfill(2)))

            # Save Plot
            if save_plot:
                self.save_plot("S{}_all_phases_smoothed".format(str(self.subjects[sub]).zfill(2)))
        # close(n=n_sub)

    def plot_HR_z_smoothed(self, save_plot=False):
        """Plot smoothed z-scored HR for each subject over all phases"""
        for sub in range(self.n_sub):
            fig_sub_z_smooth = plt.figure("S{} | all_phases_z".format(str(self.subjects[sub]).zfill(2)), figsize=(14, 8))
            plt.plot(self.all_phases["all_phases_z"][sub], alpha=0.3)  # can be taken out
            plt.plot(self.all_phases["all_phases_z_smooth"][sub])
            line = 0
            plt.vlines(line, ymin=-10, ymax=10, linestyles="--", alpha=0.6)
            for num, lines in enumerate(self.phase_lengths_int):
                line += lines
                plt.vlines(line, ymin=-10, ymax=10, linestyles="--", alpha=0.6)
                plt.text(x=line-lines, y=10, s=self.phases[num], size=4)

            fig_sub_z_smooth.suptitle("smoothed({}dpts.) z-scored HR of S{} over all phases".format(self.w_size,
                                                                                                  str(self.subjects[sub]).zfill(2)))

            if save_plot:
                self.save_plot("S{}_all_phases_z_smoothed".format(str(self.subjects[sub]).zfill(2)))
        # close(n=n_sub)

    def plot_HR_z_smoothed_all(self, save_plot=False):
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
        fig_sub_z_single_smooth.suptitle("smoothed({}dpts.) z-scored HR for each subject over all phases + mean".format(self.w_size))

        if save_plot:
            self.save_plot("All_S_all_phases_z_smoothed")

        # close()

        # plt.plot(self.all_phases["all_phases"][0, :])
        # plt.plot(self.all_phases["all_phases_z"][0, :])

        # plt.plot(self.all_phases["all_phases_z"][0, :])
        # plt.plot(self.all_phases["all_phases_z_smooth"][0, :])

    def cross_cor(self, save_plot=False):
        """Cross-cor z-ratings with corresponding z-HR of each phase"""

        for sub_idx, sub in enumerate(self.subjects):

            plt.figure("S{} | z-HR and z-Rating | xcorr | 1Hz".format(str(sub).zfill(2)), figsize=(8, 10))
            subplot_nr = 420
            ylims = 0
            # Found min/max values over all roller coasters of zHR
            for coaster in self.roller_coaster:
                if np.abs(int(np.nanmin(self.each_phases_z[coaster])) - 1) > ylims:
                    ylims = np.abs(int(np.nanmin(self.each_phases_z[coaster])) - 1)
                elif np.int(np.nanmax(self.each_phases_z[coaster])) + 1 > ylims:
                    ylims = np.int(np.nanmax(self.each_phases_z[coaster])) + 1

            for coaster in self.roller_coaster:
                if self.each_phases_z[coaster][sub_idx].shape != self.ratings_dic[coaster][sub_idx].shape:
                    print("Data length not the same | {} | S{}.".format(coaster, str(sub).zfill(2)))

                var1 = copy.copy(self.ratings_dic[coaster][sub_idx])
                var2 = copy.copy(self.each_phases_z[coaster][sub_idx])
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
                shift_counter = 0
                for idxe, event in enumerate(events[0, :]):
                    shift_counter += 1
                    if shift_counter > 4:  # reset
                        shift_counter = 1
                    shift = 1 if shift_counter > 2 else 0  # switch between 1 and zero

                    t_event = int(events[1, idxe])  # timepoint of event
                    up_down = -1 if idxe % 2 == 0 else 1

                    if up_down > 0:
                        y_max_value = np.min((var1[t_event], var2[t_event]))
                    else:  # up_down < 0:
                        y_max_value = np.max((var1[t_event], var2[t_event]))

                    plt.vlines(x=t_event, ymin=(ylims-shift)*up_down, ymax=y_max_value, linestyles="dotted",
                               alpha=0.3)
                    plt.text(x=t_event, y=(ylims-shift)*up_down, s=event, size=6)

                plt.legend()

                # plt.figure("S{} in {} | HR, Rating".format(str(sub).zfill(2), coaster))
                # plt.plot(var1, var2, "o")
                # plt.xlabel("z_Ratings")
                # plt.ylabel("z_HR")
                # plt.legend()

                subplot_nr += 1
                plt.subplot(subplot_nr)
                maxlag = 20
                plt.xcorr(var1, var2, maxlags=maxlag)
                plt.ylim(-0.9, 0.9)
                plt.text(x=0-maxlag/2, y=0.6, s="S{} in {} | xcorr".format(str(sub).zfill(2), coaster))

                # plt.figure("S{} in {} | np.correlate".format(str(sub).zfill(2), coaster))
                # plt.plot(np.correlate(var1, var2, mode=2))  # mode = "full"

            plt.tight_layout()

            if save_plot:
                self.save_plot("S{}_z_Ratings_HR_x_corr".format(str(sub).zfill(2)))

        # np.correlate(var1, var2, "full")
        # from scipy.stats.stats import pearsonr
        # print(pearsonr(var1, var1))

    # TODO run cross-correlation with ratings
    # 2.1) RE-z-score heart rate as was done for ratings (concatenate two coasters then z-score)
    # 2.2) cross-cor rz-ratings with corresponding Re-z-HR of each phase


ec = ECGplot(n_sub=45,
             dropouts=[1, 12, 32, 33, 35, 38, 41, 42, 45],
             subject_selection=[6, 11, 14, 17, 20, 27, 31, 34, 36],
             smooth_w_size=21)

# ec.plot_HR(save_plot=False)
# ec.plot_HR(save_plot=True)

# ec.plot_HR_z(save_plot=False)
# ec.plot_HR_z(save_plot=True)

# ec.plot_HR_z_all(save_plot=False)
# ec.plot_HR_z_all(save_plot=True)

# ec.plot_HR_z_smoothed(save_plot=False)
# ec.plot_HR_z_smoothed(save_plot=True)

# ec.plot_HR_z_smoothed_all(save_plot=False)
# ec.plot_HR_z_smoothed_all(save_plot=True)

# ec.cross_cor(save_plot=False)
# ec.cross_cor(save_plot=True)

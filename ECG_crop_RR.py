"""
Takes the concatenated ECG (TR) Kubios file of a participant and (re-)crops them in single phases.
    • read/load
    • crop
    • export (*.txt)

Author: Simon Hofmann | <[surname].[lastname][at]protonmail.com> | 2017
"""

import numpy as np
import scipy.io as sio
import os.path
import matplotlib.pyplot as plt

# TODO Cut phases with disrupted signal. Look at NeVRo_Data_HPB_ECG (DRIVE)

# Change to folder which contains files
wdic = "../../Data/ECG/concatenated/"
wdic_cropTR = "../../Data/ECG/TR_cropped/"
wdic_cropTR_trim = "../../Data/ECG/TR_cropped/trimmed/"

# Set variables
nSub = 45  # Number of Subjects
phases = ["Resting_Open", "Resting_Close",
          "Space_Mov", "Break_Mov", "Ande_Mov",
          "Rating_Space_Mov", "Rating_Ande_Mov",
          "Space_NoMov", "Break_NoMov", "Ande_NoMov",
          "Rating_Space_NoMov", "Rating_Ande_NoMov"]
s_freq = 500.  # sampling frequency

# Get crop-timing from N_datapoint-matrix
time_table = np.genfromtxt(wdic + "entries_per_phase_{}.csv".format(str(nSub)), delimiter=";")
# if time_table.ndim == 1: np.reshape(time_table, newshape=(nSub, -1))

missing = []  # List for missing files
log_file = ["Check these entries\n"]  # log_file for suspicious entries

# Matrix of all T_R-entries  per phase per Subject.
n_entries_df = np.zeros(shape=(nSub, len(phases)+3), dtype=np.int)
m_HR_df = np.zeros(shape=(nSub, len(phases)+2), dtype=np.int)
n_entries_df[:, 0] = range(1, nSub+1)  # first column: Subject-Numbers
m_HR_df[:, 0] = range(1, nSub+1)

# for subject in [2]:  # # in case to convert single subjects
for subject in range(1, nSub+1):

    print("Crop Subject_{}".format(subject))
    phase_count = 0
    # t_cut = 0  # time to cut

    # Filename
    file_name = "Kubios/NVR_S{}_hrv.mat".format(str(subject).zfill(2))

    if os.path.isfile(wdic + file_name):
        concat_hrv_mat = sio.loadmat(wdic + file_name)
        # type(concat_hrv_mat)  # Dict
        # concat_hrv_mat['Res'][0][0][3][0][0] + [0]Param, [1]Data, [2]Statistics, [3]Frequency, [4]NonLinear
        T_R = concat_hrv_mat['Res'][0][0][3][0][0][1][0][0][1]  # Data + [0][0] + [0]Rsgn, [1]T_R, [2]RR ...
        T_R = T_R.reshape(-1)  # convert format in array shape(n, )
        RR = concat_hrv_mat['Res'][0][0][3][0][0][1][0][0][2]  # Data/RR
        RR = RR.reshape(-1)  # convert format in array shape(n, )

        # Go through and find too close values:
        kick_out = []
        for num, time in enumerate(T_R):
            if (num + 1) < len(T_R):
                if np.around(time, 3) == np.around(T_R[num + 1], 3):
                    kick_out.append(num)
        # Kick out double entries
        if len(kick_out) > 0:
            T_R = np.delete(T_R, kick_out)
            RR = np.delete(RR, kick_out)
            fill_word = ["entry", "was", "its"] if len(kick_out) == 1 else ["entries", "were", "their"]
            log_file.append("Subject_{} \n".format(subject))
            log_file.append("__T_R & RR: {} {}, which {} too close to {} neighbour was kicked out – at index{} \n"
                            .format(len(kick_out), fill_word[0], fill_word[1], fill_word[2], kick_out))

        # HR_s = 1/RR  # heart rate (HR): heart beats per second
        HR_m = 60 / RR  # HR: heart beats per minute

        # Get crop-timing from N_datapoint-matrix
        if int(time_table[subject-1, 0]) == subject:

            # Check whether last entry of T_R fits to whole num of time-points
            if T_R[-1] > time_table.sum(axis=1)[subject-1]/s_freq:  # or sum(time_table[subject-1])
                t_T_R = np.around(T_R[-1], 2)
                t_time_table = np.around(time_table.sum(axis=1)[subject-1]/s_freq, 2)
                log_file.append("Subject_{} \n".format(subject))
                log_file.append("(!)__Last entry T_R={}sec, but calculated duration={}sec \n"
                                .format(t_T_R, t_time_table))
                too_long = np.around(t_T_R - t_time_table, 2)
                log_file.append("(!)__T_R is {}sec too long \n".format(too_long))

            # Check whether length of T_R reasonable:
            tt_sec = time_table.sum(axis=1)[subject-1]/s_freq  # Total time in sec
            # len(T_R)/tt_sec  # average HR (beats/sec) through out the whole experiment
            m_HR_min = np.around(((float(len(T_R)) / tt_sec) * 60), 1)  # a HR (b/m)

            if m_HR_min < 60 or m_HR_min > 110:
                log_file.append("Subject_{} average HR (b/min): {}\n".format(subject, m_HR_min))
                # log_file.append("Subject_{} average HR (b/min): {}\n".format(subject, np.around(np.mean(HR_m), 2)))

            # set t_crop_count to zero for each subject
            t_crop_count = 0
            # write number of overall R-Peaks in table
            n_entries_df[subject-1, len(phases) + 2] = len(T_R)
            # write number of overall mean heart rate in table
            m_HR_df[subject-1, len(phases) + 1] = np.around(np.mean(HR_m), 2)

            # vector_01: 2) fill 1s in 0/1 vector, when there is a R-peak
            vector_01 = np.zeros((int(time_table.sum(axis=1)[subject-1])))
            # len(vector_01)  # == N of all datapoints

            test_count_01 = 0
            for t_point in T_R:
                test_count_01 += 1
                ind_01 = int(np.around(t_point * s_freq))
                if ind_01 < len(vector_01):
                    x_test_01 = vector_01[ind_01]
                    vector_01[ind_01] = 1
                    y_test_01 = vector_01[ind_01]
                    if x_test_01 == y_test_01:
                        log_file.append("Subject_{} \n".format(subject))
                        log_file.append("__T_R[{}:{}]: {}, {}\n".format(test_count_01 - 2, test_count_01,
                                                                         T_R[test_count_01-2], T_R[test_count_01-1]))
                else:
                    log_file.append("Subject_{} \n".format(subject))
                    log_file.append("____Try to set 1 in vector_01 (len={}) beyond bounds (at idx={}), "
                                    "T_R is {} sec too long\n".format(len(vector_01),
                                                                     ind_01,
                                                                     (ind_01 - len(vector_01)) / s_freq))

            # Going in different phases
            for phase in phases:
                phase_count += 1
                # log_file.append("Subject_{} | phase: {}\n".format(subject, phase))

                # 1) Cut by second
                # 2) Fill a vector of original file length and indicate with 1 [0, 0, 0, 1, 0, 0,...]
                file_name_crop = "NVR_S{}_{}_T_R.txt".format(str(subject).zfill(2), phase)
                file_name_crop_v01 = "NVR_S{}_{}_T_R_v01.txt".format(str(subject).zfill(2), phase)
                file_name_crop_trim = "NVR_S{}_{}_T_R.txt".format(str(subject).zfill(2), phase)
                file_name_crop_v01_trim = "NVR_S{}_{}_T_R_v01.txt".format(str(subject).zfill(2), phase)

                t_cut_pts = 0  # time to cut
                for cut_sum in range(phase_count):
                    if (time_table[subject-1, cut_sum+1]) > 0:
                        t_cut_pts += (time_table[subject-1, cut_sum+1])
                        no_entries = False
                    else:
                        no_entries = True

                if not no_entries:
                    t_cut_pts_old = (t_cut_pts - (time_table[subject-1, cut_sum+1]))
                    t_cut_old = t_cut_pts_old/s_freq
                    t_cut = t_cut_pts/s_freq
                    # t_cut += (time_table[subject-1, phase_count]+1)/s_freq  # 1) Time/samp.-freq, time point to cut.
                    # vector_01  # 2) init. 0/1 vector

                    # Find position, where to cut

                    old_t_crop_count = t_crop_count
                    # print("old_t_crop_count:", old_t_crop_count)
                    while T_R[old_t_crop_count:t_crop_count+1][-1] < t_cut:
                        if len(T_R) > t_crop_count+1:
                            t_crop_count += 1
                        else:
                            break

                    # Test
                    # if phase_count < 12 or len(TR) > t_crop_count+1:  # For last phase not applicable
                    if len(T_R) > t_crop_count+1:  # For last phase not applicable
                        if not (T_R[old_t_crop_count:t_crop_count][-1] < t_cut <= T_R[old_t_crop_count:t_crop_count+1][-1]):
                            print("S_{} | Phase: {}, Wrong cut at T_R".format(subject, phase))
                            print("Last value: {}sec".format(T_R[old_t_crop_count:t_crop_count][-1]))
                            print("Time to cut: {}sec".format(t_cut))
                            print("Next value: {}sec".format(T_R[old_t_crop_count:t_crop_count+1][-1]))

                    # Cut according part
                    if len(T_R) > t_crop_count + 1:
                        if phase_count == 1:
                            crop_phase = T_R[old_t_crop_count:t_crop_count]
                        else:
                            crop_phase = T_R[old_t_crop_count:t_crop_count] - t_cut_old

                    else:  # for last phase
                        crop_phase = T_R[old_t_crop_count:t_crop_count + 1] - t_cut_old

                    crop_phase = crop_phase.reshape(-1, )  # unpack from brackets

                    # Cut vector_01 in parts
                    vec_01_phase = vector_01[int(t_cut_pts_old):int(t_cut_pts)]
                    # len(vec_01_phase) == time_table[subject-1, phase_count]  # must be True

                    # 1) Save cropped phase
                    export_file = open(wdic_cropTR + file_name_crop, "w")
                    if len(crop_phase) > 0:
                        for item in crop_phase:
                            export_file.write("{}\n".format(item))
                        export_file.close()

                    # Export vector_01
                    export_file_v01 = open(wdic_cropTR + file_name_crop_v01, "w")
                    for zero_or_one in vec_01_phase:
                        export_file_v01.write("{}\n".format(int(zero_or_one)))
                    export_file_v01.close()

                    # Fill number of ECG-entries in Matrix
                    n_entries_df[subject-1, phase_count] = int(len(crop_phase))
                    # Fill HR (b/m) per phase in Matrix
                    if not len(crop_phase) == sum(vec_01_phase):
                        log_file.append("Subject_{} | {} : difference in number of R-Peaks in "
                                        "vec01 file vs. time_stamp file\n".format(subject, phase))
                        log_file.append("__time_stamp file: {} R_Peaks\n".format(len(crop_phase)))
                        log_file.append("__vec01 file: {} R_Peaks\n".format(int(sum(vec_01_phase))))

                    m_HR_df[subject-1, phase_count] = (len(crop_phase)*60)/(time_table[subject-1, phase_count]/s_freq)

                else:  # if no entries for that phase
                    log_file.append("Subject_{} | {} : no entries\n".format(subject, phase))

        else:
            print("Subject is not found in timetable! \n (Instead S_{})".format(time_table[subject - 1, 0]))

    else:  # if file is not there
            print("Not in folder:", wdic + file_name)
            missing.append(subject)

# write Missing data in file
missing_file = open(wdic_cropTR + "missing_TR.txt", "w")
for item in missing:
    missing_file.write("{}\n".format(item))
missing_file.close()

log_file_save = open(wdic_cropTR + "log_file_TR.txt", "w")
for log_entry in log_file:
    log_file_save.write("{}".format(log_entry))
log_file_save.close()

# Save n_entries_df
n_entries_df[:, len(phases)+1] = n_entries_df[:, 1:len(phases)+1].sum(axis=1)
np.savetxt(wdic_cropTR + "entries_per_phase_TR_{}.csv".format(str(nSub).zfill(2)),
           n_entries_df, fmt='%i', delimiter=";")
# Save HR_df
np.savetxt(wdic_cropTR + "HR_per_phase_{}.csv".format(str(nSub).zfill(2)),
           m_HR_df, fmt='%i', delimiter=";")


# Plot m_HR distribution
# a = np.sort(m_HR_df[:, -1])
a = m_HR_df[:, -1]
a = np.delete(a, np.where(a == 0)[0])  # cut out zero values
plt.hist(a)


# Create trimmed data files
trim_time = 5.
trimmed_time_space = 153. - trim_time
trimmed_break = 30.
trimmed_time_ande = 97. - trim_time
trimmed_resting = 300.

# first for Space and Ande
for file in os.listdir(wdic_cropTR):
    if "Space" in file or "Ande" in file:
        trimmed_time = trimmed_time_space if "Space" in file else trimmed_time_ande  # if "Ande" in file

        to_trim = np.genfromtxt(wdic_cropTR + file, delimiter="\n")
        if "v01" not in file:
            # Trim 2.5sec in the beginning:
            trim_start = np.where(to_trim < trim_time/2)[0][-1] + 1
            try:
                # this should be only for S01 a problem
                trim_end = np.where(to_trim > trimmed_time+(trim_time/2))[0][0]
            except:
                print("trim_end adapted for:", file)
                trim_end = len(to_trim) - 1

            trimmed = to_trim[trim_start:trim_end]
            trimmed -= trim_time/2

            # print(file[4:7] + " T_R[last] - T_R[first) =", trimmed[-1]-trimmed[0])

        else:
            # Trim 2.5sec*samp.freq in the beginning:
            trim_start = int(s_freq * trim_time/2)
            trim_end = int(s_freq*(trimmed_time + trim_time/2))
            trimmed = to_trim[trim_start:trim_end]
            # print(file[4:7] + " v01_length/sampling freq=", len(trimmed)/s_freq)

        # Save trimmed data
        with open(wdic_cropTR_trim + file, "w") as export_file:
            if len(trimmed) > 0:
                for trimtem in trimmed:
                    exp_item = trimtem if "v01" not in file else int(trimtem)
                    export_file.write("{}\n".format(exp_item))
                    # export_file.write("{}\n".format(int(trimtem)))  # in case of v01

# now for breaks
for file in os.listdir(wdic_cropTR):
    if "Break" in file:
        trimmed_time = trimmed_break

        to_trim = np.genfromtxt(wdic_cropTR + file, delimiter="\n")
        if "v01" not in file:
            trim_start = 0
            try:
                # this should be only for S01 a problem
                trim_end = np.where(to_trim > trimmed_time)[0][0]
            except:
                # print(file)
                trim_end = len(to_trim) - 1

            trimmed = to_trim[trim_start:trim_end]

            # print(file[4:7] + " T_R[last]", trimmed[-1])

        else:
            trim_start = 0
            trim_end = int(s_freq * trimmed_time)
            trimmed = to_trim[trim_start:trim_end]
            # print(file[4:7] + " v01_length/sampling freq=", len(trimmed)/s_freq)

        # Save trimmed data
        with open(wdic_cropTR_trim + file, "w") as export_file:
            if len(trimmed) > 0:
                for trimtem in trimmed:
                    exp_item = trimtem if "v01" not in file else int(trimtem)
                    export_file.write("{}\n".format(exp_item))
                    # export_file.write("{}\n".format(int(trimtem)))  # in case of v01

# and for resting_state
for file in os.listdir(wdic_cropTR):
    if "Resting" in file:
        trimmed_time = trimmed_resting

        to_trim = np.genfromtxt(wdic_cropTR + file, delimiter="\n")
        if "v01" not in file:
            trim_start = 0
            try:
                # this should be only for S01 a problem
                trim_end = np.where(to_trim > trimmed_time)[0][0]
            except:
                # print(file)
                trim_end = len(to_trim) - 1

            trimmed = to_trim[trim_start:trim_end]

            # print(file[4:7] + " T_R[last]", trimmed[-1])

        else:
            trim_start = 0
            trim_end = int(s_freq * trimmed_time)
            trimmed = to_trim[trim_start:trim_end]
            # print(file[4:7] + " v01_length/sampling freq=", len(trimmed)/s_freq)

        # Save trimmed data
        with open(wdic_cropTR_trim + file, "w") as export_file:
            if len(trimmed) > 0:
                for trimtem in trimmed:
                    exp_item = trimtem if "v01" not in file else int(trimtem)
                    export_file.write("{}\n".format(exp_item))
                    # export_file.write("{}\n".format(int(trimtem)))  # in case of v01

# # Calculate Marker, take mean of 2 random subjects
# # Date: 2017.07.18 (S_44)
# Mk2 = 0
# Mk3 = 12590
# Mk4 = 19141
# Mk5 = 41692
# Mk6 = 48243
# Mk7 = 55793
# Mk8 = 62344
# Mk9 = 89896
# Mk10 = 96446
# marker_list = np.array([0, Mk3, Mk4, Mk5, Mk6, Mk7, Mk8, Mk9, Mk10])
# # marker_list/500
#
# # Date: 2017.06.20 (S_10)
# Mk3 = 12552
# Mk4 = 19103
# Mk5 = 41654
# Mk6 = 48205
# Mk7 = 55755
# Mk8 = 62306
# Mk9 = 89857
# Mk10 = 96408
# marker_list2 = np.array([0, Mk3, Mk4, Mk5, Mk6, Mk7, Mk8, Mk9, Mk10])
#
# marker_list/500
# marker_list2/500
#
# marker_list_mean = (marker_list + marker_list2)/(2*500)


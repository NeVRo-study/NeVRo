"""
Takes the single cropped ECG (mV) files of a participant and concatenates them in one single vector.
    • read/load
    • concatenate
    • export (*.txt)

Author: Simon Hofmann | <[surname].[lastname][at]protonmail.com> | 2017
"""

import numpy as np
import os.path
# import matplotlib.pyplot as plt


# Function to change array of strings into array of floats
def to_float(array):
    array_floats = []
    for j in range(len(array)):
        cut_array = array[j][2:-1]
        # print(j, ":", cut_array)
        float_array = float(cut_array)
        array_floats.append(float_array)
    return array_floats

# Change to Folder which contains files
wdic = "../../Data/ECG/raw_cropped/"

# Set variables
nSub = 45  # Number of Subjects
phases = ["Resting_Open", "Resting_Close",
          "Space_Mov", "Break_Mov", "Ande_Mov",
          "Rating_Space_Mov", "Rating_Ande_Mov",
          "Space_NoMov", "Break_NoMov", "Ande_NoMov",
          "Rating_Space_NoMov", "Rating_Ande_NoMov"]

# Load files (e.g., NVR_S02_Rating_Ande_Mov.txt), concatenate and export them
missing = []  # List for missing files

n_entries_df = np.zeros(shape=(nSub, len(phases)+1), dtype=np.int)  # Matrix of all ECG-entries per phase per Subj.
n_entries_df[:, 0] = range(1, nSub+1)  # first column: Subject-Numbers

# for subject in [32]:  # in case to convert single subjects
# for subject in range(33, nSub+1):  # (subject_01 is dropout)
for subject in range(1, nSub+1):  # (subject_01 is dropout)
    concat = []  # list of concatenated files

    print("Concatenate Subject_{}".format(subject))

    phase_count = 0
    for phase in phases:
        phase_count += 1

        file_name = "NVR_S{}_{}.txt".format(str(subject).zfill(2), phase)  # adapt file_name
        print("Current File:", file_name)

        if os.path.isfile(wdic + file_name):
            subject_file = np.loadtxt(wdic + file_name, dtype="str", delimiter=" ")  # load file
            # print(subject_file[0:10])
            subject_file_float = to_float(subject_file)  # change to float
            # print("len(subject_file_float):", len(subject_file_float))

            for item in subject_file_float:
                concat.append(item)  # concatenate list

            # Fill number of ECG-entries in Matrix
            n_entries_df[subject-1, phase_count] = int(len(subject_file_float))


        else:
            print("Not in folder:", wdic + file_name)
            missing.append((subject, phase))

    # print("Entries in concatenated List of S_{}: {}".format(subject, len(concat)))
    # export concatenated list
    # np.savetxt(fname=wdic+"concatenated/S{}_ECG_concat.txt".format(str(subject).zfill(2)), X=concat)
    if len(concat) > 0:
        export_file = open(wdic+"concatenated/S{}_ECG_concat.txt".format(str(subject).zfill(2)), "w")
        for item in concat:
            export_file.write("{}\n".format(item))
        export_file.close()

# write Missing data in file
missing_file = open(wdic + "concatenated/missing_concat.txt", "w")
for item in missing:
    missing_file.write("{}\n".format(item))
missing_file.close()

# Save n_entries_df
np.savetxt(wdic + "concatenated/entries_per_phase_{}.csv".format(nSub), n_entries_df, fmt='%i', delimiter=";")

# plt.plot(x_floats)

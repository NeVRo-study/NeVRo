"""
Plot predictions made by the LSTM model

Author: Simon Hofmann | <[surname].[lastname][at]protonmail.com> | 2017
"""

import numpy as np
import os.path
import matplotlib.pyplot as plt


subjects = [36]
wdic = "./LSTM/"

for subject in subjects:
    wdic_sub = wdic + "S{}/".format(str(subject).zfill(2))

    # Find correct files (csv-tables)
    for file in os.listdir(wdic_sub):
        if ".csv" in file:
            if "val_" in file:
                val_filename = file
            else:
                file_name = file

    # Load data
    pred_matrix = np.loadtxt(wdic_sub + file_name, delimiter=",")
    val_pred_matrix  = np.loadtxt(wdic_sub + val_filename, delimiter=",")

    s_fold = int(len(pred_matrix[:, 0])/2)

    # Plot
    fig = plt.figure("{}-Folds | S{} | 1Hz".format(s_fold, str(subject).zfill(2)), figsize=(12, s_fold*3))
    subplot_nr = int(str(s_fold) + "10")

    for fold in range(s_fold):

        # Vars to plot
        pred = pred_matrix[fold*2, :]
        rating = pred_matrix[fold*2 + 1, :]
        val_pred = val_pred_matrix[fold*2, :]
        val_rating = val_pred_matrix[fold*2 + 1, :]

        # open frame
        subplot_nr += 1
        plt.subplot(subplot_nr)
        plt.plot(pred, label="prediction")
        plt.plot(rating, label="rating")
        plt.plot(val_pred, label="val_prediction")
        plt.plot(val_rating, label="val_rating")
        plt.xlim(0, len(pred))
        plt.ylim(-1, 1)
        plt.legend()



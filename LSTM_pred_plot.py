"""
Plot predictions made by the LSTM model

Author: Simon Hofmann | <[surname].[lastname][at]protonmail.com> | 2017
"""

import numpy as np
import os.path
import matplotlib.pyplot as plt


subjects = [36]
wdic = "./LSTM/"
wdic_lists = wdic + "logs/debug/"

for subject in subjects:
    wdic_sub = wdic + "S{}/".format(str(subject).zfill(2))
    wdic_lists_sub = wdic_lists + "S{}/".format(str(subject).zfill(2))

    # Find correct files (csv-tables)
    for file in os.listdir(wdic_sub):
        if ".csv" in file:
            if "val_" in file:
                val_filename = file
            else:
                file_name = file

        elif ".txt" in file:
            acc_filename = file

    # Load data
    pred_matrix = np.loadtxt(wdic_sub + file_name, delimiter=",")
    val_pred_matrix = np.loadtxt(wdic_sub + val_filename, delimiter=",")

    # Import accuracies
    acc_date = np.loadtxt(wdic_sub + acc_filename, dtype=str, delimiter=",")
    for info in acc_date:

        if "S-Fold(Round):" in info:
            s_rounds = np.array(list(map(int, info.split(": [")[1][0:-1].split(" "))))

        elif "Validation-Acc:" in info:
            val_acc = info.split(": ")[1].split("  ")
            v = []
            for i, item in enumerate(val_acc):
                if i == 0:  # first
                    v.append(float(item[1:]))
                elif i == (len(val_acc)-1):  # last one
                    v.append(float(item[0:-1]))
                else:
                    v.append(float(item))
            val_acc = v
            del v

        elif "mean(Accuracy):" in info:
            mean_acc = np.round(a=float(info.split(": ")[1]), decimals=3)

    # Number of Folds
    s_fold = int(len(pred_matrix[:, 0])/2)

    # # Plot predictions
    # open frame
    figsize = (12, s_fold * (3 if s_fold < 4 else 1))
    fig = plt.figure("{}-Folds | S{} | mean(val_acc)={} | 1Hz".format(s_fold, str(subject).zfill(2), mean_acc),
                     figsize=figsize)

    # Prepare subplot division
    def subplot_div(n_s_fold):
        if n_s_fold < 10:
            sub_rows_f, sub_col_f, sub_n_f = n_s_fold, 1, 0
        else:
            sub_rows_f, sub_col_f, sub_n_f = int(n_s_fold/2), 2, 0

        return sub_rows_f, sub_col_f, sub_n_f

    sub_rows, sub_col, sub_n = subplot_div(n_s_fold=s_fold)

    # For each fold create plot
    for fold in range(s_fold):

        # Vars to plot
        pred = pred_matrix[fold*2, :]
        rating = pred_matrix[fold*2 + 1, :]
        val_pred = val_pred_matrix[fold*2, :]
        val_rating = val_pred_matrix[fold*2 + 1, :]

        # add subplot
        sub_n += 1
        fig.add_subplot(sub_rows, sub_col, sub_n)

        # plot
        # plt.plot(pred, label="prediction", marker='o', markersize=3)  # , style='r-'
        # plt.plot(rating, ls="dotted", label="rating", marker='o', mfc='none', markersize=3)
        # plt.plot(val_pred, label="val_prediction", marker='o', markersize=3)
        # plt.plot(val_rating, ls="dotted", label="val_rating", marker='o', mfc='none', markersize=3)
        plt.plot(pred, label="prediction")  # , style='r-'
        plt.plot(rating, ls="dotted", label="rating")
        plt.plot(val_pred, label="val_prediction")
        plt.plot(val_rating, ls="dotted", label="val_rating")
        plt.title(s="{}-Fold | val_acc={}".format(fold+1,
                                                  np.round(val_acc[int(np.where(np.array(s_rounds) == fold)[0])], 3)))

        # adjust size, add legend
        plt.xlim(0, len(pred))
        plt.ylim(-1.2, 2)
        plt.legend(bbox_to_anchor=(0., 0.92, 1., .102), loc=1, ncol=4, mode="expand", borderaxespad=0.)
        plt.tight_layout(pad=2)

    # # Plot accuracy-trajectories

    fig2 = plt.figure("{}-Folds Accuracies | S{} | mean(val_acc)={} | 1Hz ".format(s_fold,
                                                                                   str(subject).zfill(2),
                                                                                   mean_acc), figsize=figsize)

    # Prepare subplot division
    sub_rows, sub_col, sub_n = subplot_div(n_s_fold=s_fold)

    for fold in range(s_fold):

        # Load Data
        val_acc_fold = np.loadtxt(wdic_lists_sub + "{}/val_acc_list.txt".format(fold), delimiter=",")
        train_acc_fold = np.loadtxt(wdic_lists_sub + "{}/train_acc_list.txt".format(fold), delimiter=",")

        # add subplot
        sub_n += 1
        fig2.add_subplot(sub_rows, sub_col, sub_n)

        # plot
        plt.plot(train_acc_fold, label="train_acc")
        plt.plot(val_acc_fold, label="val_acc")

        plt.title(s="{}-Fold | val_acc={}".format(fold + 1,
                                                  np.round(val_acc[int(np.where(np.array(s_rounds) == fold)[0])], 3)))

        # adjust size, add legend
        plt.xlim(0, len(train_acc_fold))
        plt.ylim(0.0, 1.5)
        plt.legend(bbox_to_anchor=(0., 0.92, 1., .102), loc=1, ncol=4, mode="expand", borderaxespad=0.)
        plt.tight_layout(pad=2)

    # # Plot loss-trajectories

    fig3 = plt.figure("{}-Folds Loss | S{} | mean(val_acc)={} | 1Hz ".format(s_fold, str(subject).zfill(2), mean_acc),
                      figsize=figsize)

    # Prepare subplot division
    sub_rows, sub_col, sub_n = subplot_div(n_s_fold=s_fold)

    for fold in range(s_fold):
        # Load Data
        val_loss_fold = np.loadtxt(wdic_lists_sub + "{}/val_loss_list.txt".format(fold), delimiter=",")
        train_loss_fold = np.loadtxt(wdic_lists_sub + "{}/train_loss_list.txt".format(fold), delimiter=",")

        # add subplot
        sub_n += 1
        fig3.add_subplot(sub_rows, sub_col, sub_n)

        # plot
        plt.plot(train_loss_fold, label="train_loss")
        plt.plot(val_loss_fold, label="val_loss")

        plt.title(s="{}-Fold | val_loss={}".format(fold + 1,
                                                   np.round(val_acc[int(np.where(np.array(s_rounds) == fold)[0])], 3)))

        # adjust size, add legend
        plt.xlim(0, len(train_loss_fold))
        plt.ylim(-0.1, 2.5)
        plt.legend(bbox_to_anchor=(0., 0.92, 1., .102), loc=1, ncol=4, mode="expand", borderaxespad=0.)
        plt.tight_layout(pad=2)



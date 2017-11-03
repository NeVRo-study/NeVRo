# coding=utf-8
"""
Plot predictions made by the LSTM model

Author: Simon Hofmann | <[surname].[lastname][at]protonmail.com> | 2017
"""

# TODO average val acc and loss across Folds

from Meta_Functions import *
from tensorflow import gfile
import string
if platform.system() != 'Darwin':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
else:
    import matplotlib.pyplot as plt


# Save plot
@true_false_request
def save_request():
    print("Do you want to save the plots")

try:
    plots = sys.argv[1]
    try:
        int(plots)
        plots = save_request()
        script_external_exe = False
    except ValueError:
        plots = True if "True" in plots else False
        script_external_exe = True
except IndexError:
    plots = save_request()
    script_external_exe = True


if script_external_exe:
    subject = sys.argv[2]
else:
    subject = input("Enter subject number (int): ")
    assert float(subject).is_integer(), "subject number must be integer"
subject = int(subject)

if script_external_exe:
    try:
        path_specificity = sys.argv[3]
    except IndexError:
        path_specificity = ""  # this way path given via terminal, e.g., python3 LSTM_pred_plot.py False False lstm-150
else:
    path_specificity = input("Provide specific subfolder (if any), in form 'subfolder/': ")

assert path_specificity == "" or path_specificity[-1] == "/", "path_specificity must be either empty or end with '/'"

wdic = "./LSTM"
wdic_plot = "../../Results/Plots/LSTM/"
wdic_lists = wdic + "/logs"
lw = 0.5  # linewidth

wdic_sub = wdic + "/S{}/{}".format(str(subject).zfill(2), path_specificity)
wdic_lists_sub = wdic_lists + "/S{}/{}".format(str(subject).zfill(2), path_specificity)

# Find correct files (csv-tables)
for file in os.listdir(wdic_sub):
    if ".csv" in file:
        if "val_" in file:
            val_filename = file
        else:
            file_name = file

    elif ".txt" in file:
        acc_filename = file

# Intermediate step: check whether filenames alreay exist in already_plotted_dic
if plots:
    already_plotted_dic = wdic + "/S{}/already_plotted/".format(str(subject).zfill(2))
    if not gfile.Exists(already_plotted_dic):
        gfile.MakeDirs(already_plotted_dic)

    # add subfix if filename already exists
    abc = ''
    abc_counter = 0
    new_file_name = acc_filename  # could be also 'file_name' or 'val_filename'
    while os.path.exists(already_plotted_dic + new_file_name):
        new_file_name = new_file_name.split(abc + "_S")[0] + string.ascii_lowercase[abc_counter] \
                        + "_S" + new_file_name.split("_S")[1]
        abc = string.ascii_lowercase[abc_counter]
        abc_counter += 1

# Load data
pred_matrix = np.loadtxt(wdic_sub + file_name, delimiter=",")
val_pred_matrix = np.loadtxt(wdic_sub + val_filename, delimiter=",")

# Import accuracies
acc_date = np.loadtxt(wdic_sub + acc_filename, dtype=str, delimiter=";")  # Check (before ",")
for info in acc_date:

    if "S-Fold(Round):" in info:
        s_rounds = np.array(list(map(int, info.split(": [")[1][0:-1].split(" "))))

    elif "Validation-Acc:" in info:
        val_acc = info.split(": ")[1].split(", ")  # Check (before "  ")
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

    elif "Hilbert_z-Power:" in info:
        hilb = info.split(": ")[1]
        hilb = True if "True" in hilb else False

    elif "repetition_set:" in info:
        reps = float(info.split(": ")[1])

    elif "batch_random:" in info:
        rnd_batch = info.split(": ")[1]
        rnd_batch = True if rnd_batch in "True" else False

    elif "batch_size:" in info:
        batch_size = int(info.split(": ")[1])

# Number of Folds
s_fold = int(len(pred_matrix[:, 0])/2)


# Subplot division
def subplot_div(n_s_fold):
    if n_s_fold < 10:
        sub_rows_f, sub_col_f, sub_n_f = n_s_fold, 1, 0
    else:
        sub_rows_f, sub_col_f, sub_n_f = int(n_s_fold / 2), 2, 0

    return sub_rows_f, sub_col_f, sub_n_f


# # Plot predictions
# open frame
figsize = (12, s_fold * (3 if s_fold < 4 else 1))
fig = plt.figure("{}-Folds | S{} | mean(val_acc)={} | 1Hz".format(s_fold, str(subject).zfill(2), mean_acc),
                 figsize=figsize)

# Prepare subplot division
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
    plt.plot(pred, label="prediction", linewidth=lw)  # , style='r-'
    plt.plot(rating, ls="dotted", label="rating")
    plt.plot(val_pred, label="val_prediction", linewidth=lw)
    plt.plot(val_rating, ls="dotted", label="val_rating")
    plt.title(s="{}-Fold | val_acc={}".format(fold+1,
                                              np.round(val_acc[int(np.where(np.array(s_rounds) == fold)[0])], 3)))

    # adjust size, add legend
    plt.xlim(0, len(pred))
    plt.ylim(-1.2, 2)
    plt.legend(bbox_to_anchor=(0., 0.90, 1., .102), loc=1, ncol=4, mode="expand", borderaxespad=0.)
    plt.tight_layout(pad=2)

fig.show()

if plots:
    plot_filename = "{}{}_|{}{}*{}({})_|_{}-Folds_|_S{}_|_mean(val_acc)_{:.2f}_|_{}.png".format(
        file_name[0:10], abc, "_Hilbert_" if hilb else "_", int(reps),
        "rnd-batch" if rnd_batch else "subsequent-batch", batch_size, s_fold, str(subject).zfill(2), mean_acc,
        path_specificity[:-1])

    fig.savefig(wdic_plot + plot_filename)

# # Plot accuracy-trajectories

fig2 = plt.figure("{}-Folds Accuracies | S{} | mean(val_acc)={} | 1Hz ".format(s_fold,
                                                                               str(subject).zfill(2),
                                                                               mean_acc), figsize=figsize)

# Prepare subplot division
sub_rows, sub_col, sub_n = subplot_div(n_s_fold=s_fold)

for fold in range(s_fold):

    # Load Data
    train_acc_fold = np.loadtxt(wdic_lists_sub + "{}/train_acc_list.txt".format(fold), delimiter=",")
    val_acc_fold = np.loadtxt(wdic_lists_sub + "{}/val_acc_list.txt".format(fold), delimiter=",")
    val_acc_training_fold = [eval(line.split("\n")[0])
                             for line in open(wdic_lists_sub + "{}/val_acc_training_list.txt".format(fold))]

    # Attach also last val_acc to list
    last_acc = (np.nanmean(val_acc_fold), len(train_acc_fold)-1)
    val_acc_training_fold.append(last_acc)

    where, vacc = zip(*val_acc_training_fold)

    # add subplot
    sub_n += 1
    fig2.add_subplot(sub_rows, sub_col, sub_n)

    plt.plot(train_acc_fold, label="train_acc", linewidth=lw/2)
    plt.plot(vacc, where, label="val_acc", linewidth=lw)

    plt.title(s="{}-Fold | val_acc={}".format(fold + 1,
                                              np.round(val_acc[int(np.where(np.array(s_rounds) == fold)[0])], 3)))

    # adjust size, add legend
    plt.xlim(0, len(train_acc_fold))
    plt.ylim(0.0, 1.5)
    plt.legend(bbox_to_anchor=(0., 0.90, 1., .102), loc=1, ncol=4, mode="expand", borderaxespad=0.)
    plt.tight_layout(pad=2)

fig2.show()

# Plot
if plots:
    plot_filename = "{}{}_|{}{}*{}({})_|_{}-Folds_|_Accuracies_|_S{}_|_mean(val_acc)_{:.2f}_|_{}.png".format(
        file_name[0:10], abc, "_Hilbert_" if hilb else "_", int(reps),
        "rnd-batch" if rnd_batch else "subsequent-batch", batch_size, s_fold, str(subject).zfill(2), mean_acc,
        path_specificity[:-1])

    fig2.savefig(wdic_plot + plot_filename)

# # Plot loss-trajectories

fig3 = plt.figure("{}-Folds Loss | S{} | mean(val_acc)={} | 1Hz ".format(s_fold, str(subject).zfill(2), mean_acc),
                  figsize=figsize)

# Prepare subplot division
sub_rows, sub_col, sub_n = subplot_div(n_s_fold=s_fold)

for fold in range(s_fold):
    # Load Data
    val_loss_fold = np.loadtxt(wdic_lists_sub + "{}/val_loss_list.txt".format(fold), delimiter=",")
    train_loss_fold = np.loadtxt(wdic_lists_sub + "{}/train_loss_list.txt".format(fold), delimiter=",")
    val_loss_training_fold = [eval(line.split("\n")[0])
                              for line in open(wdic_lists_sub + "{}/val_loss_training_list.txt".format(fold))]

    # Attach also last val_loss to list
    last_loss = (np.nanmean(val_loss_fold), len(train_loss_fold) - 1)
    val_loss_training_fold.append(last_loss)

    where_loss, vloss = zip(*val_loss_training_fold)

    # add subplot
    sub_n += 1
    fig3.add_subplot(sub_rows, sub_col, sub_n)

    # plot
    plt.plot(train_loss_fold, label="train_loss", linewidth=lw/2)
    plt.plot(vloss, where_loss, label="val_loss", linewidth=lw)

    plt.title(s="{}-Fold | val_loss={}".format(fold + 1,
                                               np.round(val_acc[int(np.where(np.array(s_rounds) == fold)[0])], 3)))

    # adjust size, add legend
    plt.xlim(0, len(train_loss_fold))
    plt.ylim(-0.05, 1.8)
    plt.legend(bbox_to_anchor=(0., 0.90, 1., .102), loc=1, ncol=4, mode="expand", borderaxespad=0.)
    plt.tight_layout(pad=2)

fig3.show()

# Plot
if plots:
    plot_filename = "{}{}_|{}{}*{}({})_|_{}-Folds_|_Loss_|_S{}_|_mean(val_acc)_{:.2f}_|_{}.png".format(
        file_name[0:10], abc, "_Hilbert_" if hilb else "_", int(reps),
        "rnd-batch" if rnd_batch else "subsequent-batch", batch_size, s_fold, str(subject).zfill(2), mean_acc,
        path_specificity[:-1])

    fig3.savefig(wdic_plot + plot_filename)

# # Plot i) average training prediction and ii) concatenated val_prediction

fig4 = plt.figure("{}-Folds mean(train)_&_concat(val)_| S{} | mean(val_acc)={} | 1Hz ".format(
    s_fold, str(subject).zfill(2), mean_acc),
    figsize=(10, 6))

# delete ratings out of pred_matrix first and then average across rows
average_train_pred = np.nanmean(a=np.delete(arr=pred_matrix, obj=np.arange(1, 2*s_fold, 2), axis=0), axis=0)
concat_val_pred = np.nanmean(a=np.delete(arr=val_pred_matrix, obj=np.arange(1, 2*s_fold, 2), axis=0), axis=0)
whole_rating = np.nanmean(a=np.delete(arr=pred_matrix, obj=np.arange(0, 2*s_fold-1, 2), axis=0), axis=0)

# Plot average train prediction
fig4.add_subplot(2, 1, 1)
plt.plot(average_train_pred, label="mean_train_prediction", linewidth=lw)  # , style='r-'
plt.plot(whole_rating, ls="dotted", label="rating")
plt.title(s="Average train prediction | {}-Folds".format(s_fold))
# adjust size, add legend
plt.xlim(0, len(whole_rating))
plt.ylim(-1.2, 2)
plt.legend(bbox_to_anchor=(0., 0.90, 1., .102), loc=1, ncol=4, mode="expand", borderaxespad=0.)
plt.tight_layout(pad=2)

# Plot average train prediction
fig4.add_subplot(2, 1, 2)
plt.plot(concat_val_pred, label="concat_val_prediction", linewidth=lw, c="xkcd:coral")
plt.plot(whole_rating, ls="dotted", label="rating")
plt.title(s="Concatenated val_prediction | {}-Folds | mean_val_acc={}".format(s_fold, np.round(mean_acc, 3)))
# adjust size, add legend
plt.xlim(0, len(whole_rating))
plt.ylim(-1.2, 2)
plt.legend(bbox_to_anchor=(0., 0.90, 1., .102), loc=1, ncol=4, mode="expand", borderaxespad=0.)
plt.tight_layout(pad=2)

fig4.show()

# Plot
if plots:
    plot_filename = "{}{}_|{}{}*{}({})_|_{}-Folds_|_all_val_|_S{}_|_mean(val_acc)_{:.2f}_|_{}.png".format(
        file_name[0:10], abc, "_Hilbert_" if hilb else "_", int(reps),
        "rnd-batch" if rnd_batch else "subsequent-batch", batch_size, s_fold, str(subject).zfill(2), mean_acc,
        path_specificity[:-1])

    fig4.savefig(wdic_plot + plot_filename)


@true_false_request
def close_plots():
    print("Do you want to close plots?")


if not script_external_exe:  # Check whether script is opened from intern(python) or extern(terminal)
    if close_plots():
        for _ in range(4):
            plt.close()

# When saved then move *.csv & *.txt files into folder "Already Plotted"
if plots:
    for file in os.listdir(wdic_sub):
        if file != 'already_plotted':
            new_file_name = file.split("_S")[0] + abc + "_S" + file.split("_S")[1]

            while True:
                try:
                    gfile.Rename(oldname=wdic_sub+file, newname=already_plotted_dic+new_file_name, overwrite=False)
                    break
                except Exception:
                    new_file_name = new_file_name.split(abc + "_S")[0] + string.ascii_lowercase[abc_counter] \
                                    + "_S" + new_file_name.split("_S")[1]
                    abc = string.ascii_lowercase[abc_counter]
                    abc_counter += 1

    # open folder
    open_folder(wdic_plot)

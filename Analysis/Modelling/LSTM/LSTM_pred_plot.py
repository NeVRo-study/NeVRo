# coding=utf-8
"""
Plot predictions made by the LSTM model

Author: Simon Hofmann | <[surname].[lastname][at]pm.me> | 2017, 2019 (Update)
"""

import sys
# sys.path.insert(0, './LSTM Model')  # or set the folder as source root
from load_data import *
from tensorflow import gfile
import string
import shutil
import ast

if platform.system() != 'Darwin':
    import matplotlib

    matplotlib.use('Agg')
    # print(matplotlib.rcParams['backend'])
    import matplotlib.pyplot as plt
else:
    import matplotlib.pyplot as plt

# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

# Add mean input data
plt_input_data = False  # default value: False TODO revisit


# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

# Save plot
@true_false_request
def save_request():
    print("Do you want to save the plots?")


@true_false_request
def logfolder_delete_request():
    print("Do you want to delete the log folders after plotting (+saving)?")


def plot_all_there_is(dellog):
    """
    Search for processed files which weren't plotted yet.
    :param dellog: True or False: Delete log and checkpoint files after plotting
    """

    for folder in os.listdir("./processed/"):
        if "mov" in folder:
            for cond_folder in os.listdir(f"./processed/{folder}/"):
                if "S" == cond_folder[0]:
                    sub = int(cond_folder.split('S')[1])
                    print("subject:", sub)
                    for subfolder in os.listdir(f"./processed/{folder}/{cond_folder}"):
                        if subfolder != 'already_plotted':
                            subfol = subfolder
                            print("subfolder:", subfol)
                            if len(os.listdir(f"./processed/{folder}/{cond_folder}/{subfol}/")) >= 3:
                                subprocess.Popen(["python3", "LSTM_pred_plot.py", 'True', str(sub),
                                                  subfol + "/", str(dellog)])


# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

# # Check from where the script is executed, and adapt arguments accordingly:
try:
    plots = sys.argv[1]
    try:
        assert plots in ["--mode=client"]
        plots = save_request()
        script_external_exe = False
    except AssertionError:
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
        path_specificity = ""
        # this way path given via terminal, e.g., python3 LSTM_pred_plot.py False False lstm-150
else:
    path_specificity = input("Provide specific subfolder (if any), in form 'subfolder/': ")

assert path_specificity == "" or path_specificity[-1] == "/", \
    "path_specificity must be either empty or end with '/'"

# Ask whether to delete log (large files) and checkpoint folders after plotting and saving plots:
# if platform.system() != 'Darwin':
if plots:
    if script_external_exe:
        try:
            delete_log_folder = sys.argv[4]
            delete_log_folder = True if "True" in delete_log_folder else False
        except IndexError:
            delete_log_folder = True  # rather False to save logs if not clear
    else:
        delete_log_folder = logfolder_delete_request()
        # delete_log_folder = True  # switch on over night
        # delete_log_folder = False

else:  # If plots are not saved, do not delete log folders
    delete_log_folder = False

# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

# Get condition
cond = path_specificity.split("_")[1]  # 'BiCl_nomov_RndHPS_lstm-20-....'
assert cond in ["mov", "nomov"], "condition must be indicated 'mov', 'nomov'!"

# Set paths
wdic = f"./processed/{cond}"
wdic_plot = "../../../Results/Plots/LSTM/"
wdic_lists = wdic + f"/logs/{cond}"
wdic_checkpoint = wdic + "/checkpoints"
lw = 0.5  # linewidth

wdic_sub = wdic + f"/{s(subject)}/{path_specificity}"  # wdic + f"/nomov/{s(subject)}/already_plotted"
wdic_lists_sub = wdic_lists + f"/{s(subject)}/{path_specificity}"
wdic_checkpoint_sub = wdic_checkpoint + f"/{s(subject)}/{path_specificity}"

# Find correct files (csv-tables)
shuff_filename = "None"
file_name = ''  # init
acc_filename = ''  # init
val_filename = ''  # init
for file in os.listdir(wdic_sub):
    if ".csv" in file:
        if "val_" in file:
            val_filename = file
        else:
            file_name = file

    elif ".txt" in file:
        acc_filename = file

    elif ".npy" in file:
        shuff_filename = file

# Intermediate step: check whether filenames already exist in already_plotted_dic
abc = ''  # init
if plots:
    already_plotted_dic = wdic + f"/{s(subject)}/already_plotted/"
    if not gfile.Exists(already_plotted_dic):
        gfile.MakeDirs(already_plotted_dic)

    # add subfix if filename already exists
    abc_counter = 0
    new_file_name = acc_filename  # could be also 'file_name' or 'val_filename'
    while os.path.exists(already_plotted_dic + new_file_name):
        new_file_name = new_file_name.split(abc + "_S")[0] + string.ascii_lowercase[abc_counter] \
                        + "_S" + new_file_name.split("_S")[1]
        abc = string.ascii_lowercase[abc_counter]
        abc_counter += 1

# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

# Load data
pred_matrix = np.loadtxt(wdic_sub + file_name, delimiter=",")
val_pred_matrix = np.loadtxt(wdic_sub + val_filename, delimiter=",")

# Look for shuffle matrix
if os.path.exists(wdic_sub + shuff_filename):
    shuffle_order_matrix = np.load(wdic_sub + shuff_filename)

    # shuffle_order_fold5 = np.load(wdic_lists + "/S22/test_task-class_shuffle-T/5/5_shuffle_order.npy")
    dub_shuffle_order_matrix = np.repeat(a=shuffle_order_matrix, repeats=2, axis=0)

    # Correct order of matrices according to shuffle order of each fold (saved in shuffle_order_matrix)
    pred_matrix = sort_mat_by_mat(mat=pred_matrix, mat_idx=dub_shuffle_order_matrix)

    val_pred_matrix = sort_mat_by_mat(mat=val_pred_matrix, mat_idx=dub_shuffle_order_matrix)
    del dub_shuffle_order_matrix

# Number of Folds
s_fold = int(len(pred_matrix[:, 0]) / 2)

# Import accuracies
acc_date = np.loadtxt(wdic_sub + acc_filename, dtype=str, delimiter=";")  # Check (before ",")

# First initialize variables
task = ''  # init
cond = None  # init
sba = ''  # init
mean_acc = None  # init
val_acc = None  # init
components = None  # init
hr_comp = None  # init
file_type = None  # init
hilb = None  # init
bpass = None  # init
s_rounds = None  # init
reps = None  # init
rnd_batch = None  # init
batch_size = None  # init

for info in acc_date:

    if "Condition:" in info:
        cond = info.split(": ")[1]

    elif "SBA:" in info:
        sba = info.split(": ")[1]
        sba = "SBA" if "True" in sba else "SA"

    elif "Task:" in info:
        task = info.split(": ")[1]
        wdic_plot += f"{cond}/{task}/"
        if not gfile.Exists(wdic_plot):
            gfile.MakeDirs(wdic_plot)  # "../../../Results/Plots/LSTM/nomov/regression/"

    elif "Shuffle_data:" in info:
        shuffle = True if info.split(": ")[1] == "True" else False
        if shuffle:
            assert os.path.exists(wdic_sub + shuff_filename), "shuffle_order_matrix is missing"

    elif "datatype:" in info:
        file_type = info.split(": ")[1]

    elif "band_pass:" in info:
        bpass = True if "True" in info.split(": ")[1] else False

    elif "Hilbert_z-Power:" in info:
        hilb = info.split(": ")[1]
        hilb = True if "True" in hilb else False

    elif "repetition_set:" in info:
        reps = float(info.split(": ")[1])

    elif "batch_size:" in info:
        batch_size = int(info.split(": ")[1])

    elif "batch_random:" in info:
        rnd_batch = info.split(": ")[1]
        rnd_batch = True if rnd_batch in "True" else False

    elif "component" in info:
        components = info.split(": ")[1].split("(")[1].split(")")[0]
        if components[0] == "[":
            components = components[1:-1]

        components = [int(comp) for comp in components.split(",")]

        hr_comp = True if "HRcomp" in info else False

    elif "S-Fold(Round):" in info:
        s_rounds = np.array(list(map(int, info.split(": [")[1][0:-1].split(" "))))

    elif "Validation-Acc:" in info:
        val_acc = info.split(": ")[1].split(", ")  # Check (before "  ")
        v = []
        for i, item in enumerate(val_acc):
            if i == 0:  # first
                v.append(float(item[1:]))
            elif i == (len(val_acc) - 1):  # last one
                v.append(float(item[0:-1]))
            else:
                v.append(float(item))
        val_acc = v
        del v

    elif "mean(Accuracy):" in info:
        mean_acc = np.round(a=float(info.split(": ")[1]), decimals=3)

    elif "mean_line_acc:" in info:
        mean_line_acc = float(info.split(": ")[1])

    elif "Validation-Class-Acc:" in info:
        val_class_acc = info.split(": ")[1].split(", ")  # Check (before "  ")
        v = []
        for i, item in enumerate(val_class_acc):
            if i == 0:  # first
                v.append(float(item[1:]))
            elif i == (len(val_class_acc) - 1):  # last one
                v.append(float(item[0:-1]))
            else:
                v.append(float(item))
        val_class_acc = v
        del v

    # elif "mean(Classification_Accuracy):" in info:
    #     mean_class_acc = np.round(a=float(info.split(": ")[1]), decimals=3)

# Load full rating vector
whole_rating = load_rating_files(subjects=subject, condition=cond,
                                 bins=False if task == "regression" else True)[str(subject)][sba][cond]

real_rating = None  # init
only_entries_rating = None  # init
lower_tert_bound = None  # init
upper_tert_bound = None  # init
if task == "classification":
    whole_rating[np.where(whole_rating == 0)] = np.nan
    real_rating = normalization(array=load_rating_files(subjects=subject,
                                                        condition=cond,
                                                        sba=sba,
                                                        bins=False)[str(subject)][sba][cond],
                                lower_bound=-1, upper_bound=1)  # range [-1, 1]

    tertile = int(len(real_rating) / 3)
    lower_tert_bound = np.sort(real_rating)[tertile]  # np.percentile(a=real_rating, q=33.33)
    upper_tert_bound = np.sort(real_rating)[tertile * 2]  # np.percentile(a=real_rating, q=66.66)

    no_entries = np.isnan(np.nanmean(a=np.vstack((pred_matrix, val_pred_matrix)), axis=0))
    only_entries_rating = copy.copy(real_rating)
    only_entries_rating[np.where(no_entries)] = np.nan

    # Exchange mean_acc, if:
    mean_acc = np.round(np.mean(calc_binary_class_accuracy(prediction_matrix=val_pred_matrix)), 3)
    # == np.round(np.nanmean(val_class_acc), 3)
else:
    whole_rating = normalization(array=whole_rating, lower_bound=-1, upper_bound=1)  # range [-1, 1]
    # whole_rating = np.nanmean(a=np.delete(arr=pred_matrix, obj=np.arange(0, 2*s_fold-1, 2), axis=0),
    #                           axis=0)

# Load Neural/HR Data
mpsec_eeg_data = []  # init
if plt_input_data:
    data = get_nevro_data(subject=subject, task=task, cond=cond,
                          component=components, hr_component=hr_comp,
                          filetype=file_type, hilbert_power=hilb, band_pass=bpass,
                          s_fold_idx=s_fold - 1, s_fold=s_fold)

    eeg_data = np.concatenate((data["train"].eeg, data["validation"].eeg))
    assert eeg_data.shape[0] == pred_matrix.shape[1], "Shapes differ!"
    mpsec_eeg_data = []
    for data_per_sec in eeg_data:
        # Take mean across components to get a rough estimate about the magnitude of the input composition
        mpsec_eeg_data.append(np.nanmean(data_per_sec))

    mpsec_eeg_data = normalization(array=mpsec_eeg_data, lower_bound=1.1, upper_bound=2.)


# Subplot division
def subplot_div(n_s_fold):
    if n_s_fold < 10:
        sub_rows_f, sub_col_f, sub_n_f = n_s_fold, 1, 0
    else:
        sub_rows_f, sub_col_f, sub_n_f = int(n_s_fold / 2), 2, 0

    return sub_rows_f, sub_col_f, sub_n_f


# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

# # Plot predictions 1)
# open frame
figsize = (12, s_fold * (3 if s_fold < 4 else 1))
fig = plt.figure(f"{s_fold}-Folds | {s(subject)} | {cond} | {task} | mean(val_acc)={mean_acc} | 1Hz",
                 figsize=figsize)

# Prepare subplot division
sub_rows, sub_col, sub_n = subplot_div(n_s_fold=s_fold)

# For each fold create plot
for fold in range(s_fold):

    # Vars to plot
    pred = pred_matrix[fold * 2, :]
    rating = pred_matrix[fold * 2 + 1, :]
    val_pred = val_pred_matrix[fold * 2, :]
    val_rating = val_pred_matrix[fold * 2 + 1, :]

    # add subplot
    sub_n += 1
    fig.add_subplot(sub_rows, sub_col, sub_n)

    # plot
    # plt.plot(pred, label="prediction", marker='o', markersize=3)  # , style='r-'
    # plt.plot(rating, ls="dotted", label="rating", marker='o', mfc='none', markersize=3)
    # plt.plot(val_pred, label="val_prediction", marker='o', markersize=3)
    # plt.plot(val_rating, ls="dotted", label="val_rating", marker='o', mfc='none', markersize=3)
    if task == "regression":
        # Predictions
        plt.plot(pred, color="steelblue", linewidth=lw, label="train-pred")
        plt.plot(val_pred, color="aquamarine", linewidth=lw, label="val-pred")
        # Ratings
        plt.plot(whole_rating, ls="dotted", color="black", label="rating")
        # Area between
        plt.fill_between(x=np.arange(0, pred_matrix.shape[1], 1), y1=whole_rating, y2=pred,
                         color="steelblue",
                         alpha='0.2')
        plt.fill_between(x=np.arange(0, pred_matrix.shape[1], 1), y1=whole_rating, y2=val_pred,
                         color="aquamarine",
                         alpha='0.2')
        # Input Data
        if plt_input_data:
            plt.plot(0, ls="dotted", color="cadetblue", alpha=1, lw=lw, label="mean input data")
            # only for the legend

        # add means and boarders
        plt.hlines(y=0, xmin=0, xmax=pred_matrix.shape[1], colors="darkgrey", lw=lw, alpha=.8)  # midline
        plt.hlines(y=np.nanmean(pred), xmin=0, xmax=pred_matrix.shape[1], linestyle="dashed",
                   colors="steelblue", lw=lw, alpha=.8)  # mean prediction
        plt.hlines(y=np.nanmean(val_pred), xmin=0, xmax=pred_matrix.shape[1], linestyle="dashed",
                   colors="aquamarine", lw=lw, alpha=.8)  # mean validation prediction
        plt.hlines(y=np.nanmean(whole_rating), xmin=0, xmax=pred_matrix.shape[1], linestyle="dotted",
                   colors="black", lw=lw, alpha=.8)  # mean ratings

        if plt_input_data and fold == 1:
            # bottomline
            plt.hlines(y=1.05, xmin=0, xmax=pred_matrix.shape[1], colors="darkgrey", lw=lw, alpha=.8)
            plt.hlines(y=2.05, xmin=0, xmax=pred_matrix.shape[1], colors="darkgrey", lw=lw, alpha=.8)
            plt.plot(mpsec_eeg_data, ls="dotted", color="cadetblue", alpha=1, lw=lw,
                     label="mean input data")

        fold_acc = np.round(val_acc[int(np.where(np.array(s_rounds) == fold)[0])], 3)
    else:  # == "classification"
        # Predictions
        plt.plot(pred, marker="o", markerfacecolor="None", ms=2, color="steelblue", linewidth=lw,
                 label="train-pred")
        plt.plot(val_pred, marker="o", markerfacecolor="None", ms=2, color="aquamarine", linewidth=lw,
                 label="val-pred")
        # Ratings
        plt.plot(real_rating, color="darkgrey", alpha=.5, lw=lw)
        plt.plot(only_entries_rating, color="teal", alpha=.3, label="rating")

        whole_rating_shift = copy.copy(whole_rating)
        for idx, ele in enumerate(whole_rating):
            if ele == 1.:
                whole_rating_shift[idx] = ele + .1
            elif ele == -1.:
                whole_rating_shift[idx] = ele - .1

        plt.plot(whole_rating_shift, ls="None", marker="s", markerfacecolor="None", ms=2, color="black",
                 label="arousal: low=-1 | high=1")
        # midline and tertile borders
        plt.hlines(y=0, xmin=0, xmax=pred_matrix.shape[1], colors="darkgrey", lw=lw, alpha=.8)
        plt.hlines(y=lower_tert_bound, xmin=0, xmax=pred_matrix.shape[1], linestyle="dashed",
                   colors="darkgrey", lw=lw, alpha=.8)
        plt.hlines(y=upper_tert_bound, xmin=0, xmax=pred_matrix.shape[1], linestyle="dashed",
                   colors="darkgrey", lw=lw, alpha=.8)
        # Correct classified
        corr_class_train = np.sign(pred * rating)
        corr_class_val = np.sign(val_pred * val_rating)

        plt.fill_between(x=np.arange(0, corr_class_train.shape[0], 1), y1=pred, y2=real_rating,
                         where=corr_class_train == 1,
                         color="green",
                         alpha='0.2')

        plt.fill_between(x=np.arange(0, corr_class_train.shape[0], 1), y1=pred, y2=real_rating,
                         where=corr_class_train == -1,
                         color="red",
                         alpha='0.2')

        plt.fill_between(x=np.arange(0, corr_class_val.shape[0], 1), y1=val_pred, y2=real_rating,
                         where=corr_class_val == 1,
                         color="lime",
                         alpha='0.2')

        plt.fill_between(x=np.arange(0, corr_class_val.shape[0], 1), y1=val_pred, y2=real_rating,
                         where=corr_class_val == -1,
                         color="orangered",
                         alpha='0.2')

        for i in range(corr_class_train.shape[0]):
            corr_col_train = "green" if corr_class_train[i] == 1 else "red"
            corr_col_val = "lime" if corr_class_val[i] == 1 else "orangered"
            # wr = whole_rating[i]
            wr = whole_rating_shift[i]

            # plt.vlines(i, ymin=wr, ymax=pred[i], colors=corr_col_train, alpha=.5, lw=lw/1.5)
            # plt.vlines(i, ymin=wr, ymax=real_rating[i], colors=corr_col_train, alpha=.5, lw=lw/1.5)
            if not np.isnan(pred[i]):
                # set a point at the end of line
                plt.plot(i, np.sign(wr) * (np.abs(wr) + .01) if corr_class_train[i] == 1 else wr,
                         marker="o", color=corr_col_train, alpha=.5, ms=1)
            # plt.vlines(i, ymin=wr, ymax=val_pred[i], colors=corr_col_val, alpha=.5, lw=lw/1.5)
            # plt.vlines(i, ymin=wr, ymax=real_rating[i], colors=corr_col_val, alpha=.5, lw=lw/1.5)
            if not np.isnan(val_pred[i]):
                plt.plot(i, np.sign(wr) * (np.abs(wr) + .01) if corr_class_val[i] == 1 else wr,
                         marker="o", color=corr_col_val, alpha=.5, ms=1)
        corr_class_val = np.delete(corr_class_val, np.where(np.isnan(corr_class_val)))
        if len(corr_class_val) > 0:
            fold_acc = np.round(sum(corr_class_val[corr_class_val == 1]) / len(corr_class_val), 3)
            # ==val_class_acc[fold]
        else:
            fold_acc = np.nan

    plot_acc = np.round(val_acc[int(np.where(np.array(s_rounds) == fold)[0])], 3)
    plt.title(label=f"{fold+1}-Fold | val-acc={fold_acc}")  # fold_acc
    if fold == 0:
        plt.legend(bbox_to_anchor=(0., 0.90, 1., .102), loc=1, ncol=4, mode="expand", borderaxespad=0.)

    # adjust size, add legend
    plt.xlim(0, len(pred))
    plt.ylim(-1.15, 2.1)

plt.xlabel("time(s)")
plt.tight_layout(pad=2)
if matplotlib.rcParams['backend'] != 'agg':
    fig.show()

if plots:

    plot_filename = f"{file_name[0:10]}{abc}_|{'_Hilbert_' if hilb else '_'}{int(reps)}*" \
        f"{'rnd-batch' if rnd_batch else 'subsequent-batch'}({batch_size})_|_{s_fold}-Folds_|" \
        f"_{task}_|_{s(subject)}_|_{cond}_|_mean(val_acc)_{mean_acc:.2f}_|_{path_specificity[:-1]}.png"

    fig.savefig(wdic_plot + plot_filename)

# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

# # Plot accuracy-trajectories 2)
fig2 = plt.figure(f"{s_fold}-Folds Accuracies | {s(subject)} | {cond} | {task} | mean(val_acc)={mean_acc} | 1Hz",
                  figsize=figsize)

# Prepare subplot division
sub_rows, sub_col, sub_n = subplot_div(n_s_fold=s_fold)

for fold in range(s_fold):

    # Load Data
    train_acc_fold = np.loadtxt(wdic_lists_sub + f"{fold}/train_acc_list.txt", delimiter=",")
    val_acc_fold = np.loadtxt(wdic_lists_sub + f"{fold}/val_acc_list.txt", delimiter=",")
    val_acc_training_fold = [ast.literal_eval(line.split("\n")[0]) for line in open(
        wdic_lists_sub + f"{fold}/val_acc_training_list.txt")]

    # Attach also last val_acc to list
    last_acc = (np.nanmean(val_acc_fold), len(train_acc_fold) - 1)
    val_acc_training_fold.append(last_acc)

    vacc, where = zip(*val_acc_training_fold)

    # Save average for later plot
    if fold == 0:
        x_fold_mean_vacc = np.array(vacc)
        x_fold_mean_tacc = train_acc_fold
    else:
        x_fold_mean_vacc += np.array(vacc)
        x_fold_mean_tacc += train_acc_fold
        if fold == s_fold - 1:  # when last fold added, divide by s_fold
            x_fold_mean_vacc /= s_fold
            x_fold_mean_tacc /= s_fold

    # add subplot
    sub_n += 1
    fig2.add_subplot(sub_rows, sub_col, sub_n)

    plt.plot(train_acc_fold, color="steelblue", linewidth=lw / 2, alpha=0.6, label="training set")
    plt.plot(where, vacc, color="aquamarine", linewidth=2 * lw, alpha=0.9, label="validation set")
    plt.hlines(y=mean_line_acc, xmin=0, xmax=train_acc_fold.shape[0], colors="red", linestyles="dashed",
               lw=2 * lw, label="meanline accuracy")

    plt.title(label=f"{fold+1}-Fold | val-acc={val_acc[int(np.where(np.array(s_rounds)==fold)[0])]:.3f}")

    # adjust size, add legend
    plt.xlim(0, len(train_acc_fold))
    plt.ylim(0.5, 1.18)  # min: approx. mean_line_acc-0.2
    if fold == 0:
        plt.legend(bbox_to_anchor=(0., 0.90, 1., .102), loc=1, ncol=4, mode="expand", borderaxespad=0.)
        plt.ylabel("accuracy")

plt.xlabel("Training iterations")
plt.tight_layout(pad=2)
if matplotlib.rcParams['backend'] != 'agg':
    fig2.show()

# Plot
if task == "regression":
    if plots:

        plot_filename = f"{file_name[0:10]}{abc}_|{'_Hilbert_' if hilb else '_'}{int(reps)}*" \
            f"{'rnd-batch' if rnd_batch else 'subsequent-batch'}({batch_size})_|_{s_fold}-Folds_|" \
            f"_Accuracies_|_{ s(subject)}_|_{cond}_|_mean(val_acc)_{mean_acc:.2f}_|_{path_specificity[:-1]}.png"

        fig2.savefig(wdic_plot + plot_filename)

else:  # task == "classification"
    plt.close()

# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

# # Plot loss-trajectories

fig3 = plt.figure(f"{s_fold}-Folds Loss | {s(subject)} | {cond} | {task} | mean(val_acc)={mean_acc} | 1Hz",
                  figsize=figsize)

# Prepare subplot division
sub_rows, sub_col, sub_n = subplot_div(n_s_fold=s_fold)

for fold in range(s_fold):
    # Load Data
    val_loss_fold = np.loadtxt(wdic_lists_sub + f"{fold}/val_loss_list.txt", delimiter=",")
    train_loss_fold = np.loadtxt(wdic_lists_sub + f"{fold}/train_loss_list.txt", delimiter=",")
    val_loss_training_fold = [ast.literal_eval(line.split("\n")[0]) for line in
                              open(wdic_lists_sub + f"{fold}/val_loss_training_list.txt")]

    # Attach also last val_loss to list
    last_loss = (np.nanmean(val_loss_fold), len(train_loss_fold) - 1)
    val_loss_training_fold.append(last_loss)

    vloss, where_loss = zip(*val_loss_training_fold)

    # Save average for later plot
    if fold == 0:
        x_fold_mean_vloss = np.array(vloss)
        x_fold_mean_tloss = train_loss_fold
    else:
        x_fold_mean_vloss += np.array(vloss)
        x_fold_mean_tloss += train_loss_fold
        if fold == s_fold - 1:  # when last fold added, divide by s_fold
            x_fold_mean_vloss /= s_fold
            x_fold_mean_tloss /= s_fold

    # add subplot
    sub_n += 1
    fig3.add_subplot(sub_rows, sub_col, sub_n)

    # plot
    plt.plot(train_loss_fold, color="steelblue", linewidth=lw / 2, alpha=0.6, label="training loss")
    plt.plot(where_loss, vloss, color="aquamarine", linewidth=2 * lw, alpha=0.9, label="validation loss")

    plt.title(label=f"{fold+1}-Fold | val-acc={val_acc[int(np.where(np.array(s_rounds)==fold)[0])]:.3f}")

    # adjust size, add legend
    plt.xlim(0, len(train_loss_fold))
    plt.ylim(-0.05, 1.55)
    if fold == 0:
        plt.legend(bbox_to_anchor=(0., 0.90, 1., .102), loc=1, ncol=4, mode="expand", borderaxespad=0.)
        plt.ylabel("Loss")

plt.xlabel("Training iterations")
plt.tight_layout(pad=2)
if matplotlib.rcParams['backend'] != 'agg':
    fig3.show()

# Plot
if task == "regression":
    if plots:

        plot_filename = f"{file_name[0:10]}{abc}_|{'_Hilbert_' if hilb else '_'}{int(reps)}*" \
            f"{'rnd-batch' if rnd_batch else 'subsequent-batch'}({batch_size})_|_{s_fold}-Folds_|" \
            f"_Loss_|_{s(subject)}_|_{cond}_|_mean(val_acc)_{mean_acc:.2f}_|_{path_specificity[:-1]}.png"

        fig3.savefig(wdic_plot + plot_filename)
else:  # task == "classification"
    plt.close()

# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

# # Plot i) average training prediction and ii) concatenated val_prediction

fig4 = plt.figure(f"{s_fold}-Folds mean(train)_&_concat(val)_| {s(subject)} | {cond} | {task} | "
                  f"mean(val_acc)={mean_acc} | 1Hz ", figsize=(10, 12))

# delete ratings out of pred_matrix first and then average across rows
average_train_pred = np.nanmean(a=np.delete(arr=pred_matrix, obj=np.arange(1, 2 * s_fold, 2), axis=0),
                                axis=0)
concat_val_pred = np.nanmean(a=np.delete(arr=val_pred_matrix, obj=np.arange(1, 2 * s_fold, 2), axis=0),
                             axis=0)
# whole_rating = np.nanmean(a=np.delete(arr=pred_matrix, obj=np.arange(0, 2*s_fold-1, 2), axis=0), axis=0)

# Plot average train prediction
fig4.add_subplot(4, 1, 1)
if task == "regression":
    # Predictions
    plt.plot(average_train_pred, color="steelblue", linewidth=2 * lw, label="mean train-prediction")
    # , style='r-'
    # Ratings
    plt.plot(whole_rating, ls="dotted", color="black", label="rating")
    plt.fill_between(x=np.arange(0, pred_matrix.shape[1], 1), y1=whole_rating, y2=average_train_pred,
                     color="steelblue", alpha='0.2')
    plt.title(label=f"Average train prediction | {s_fold}-Folds")

    plt.hlines(y=0, xmin=0, xmax=pred_matrix.shape[1], colors="darkgrey", lw=lw, alpha=.8)  # midline
    plt.hlines(y=np.nanmean(average_train_pred), xmin=0, xmax=pred_matrix.shape[1], linestyle="dashed",
               colors="steelblue", lw=lw, alpha=.8)  # mean mean train prediction
    plt.hlines(y=np.nanmean(whole_rating), xmin=0, xmax=pred_matrix.shape[1], linestyle="dotted",
               colors="black", lw=lw, alpha=.8)  # mean ratings
else:  # if task == "classification":
    # Predictions
    plt.plot(average_train_pred, color="steelblue", marker="o", markerfacecolor="None", ms=2,
             linewidth=2 * lw, label="mean_train_prediction")
    # Ratings
    plt.plot(real_rating, color="darkgrey", alpha=.5)
    plt.plot(only_entries_rating, color="teal", alpha=.3, label="ground-truth rating")
    plt.plot(whole_rating_shift, ls="None", marker="s", markerfacecolor="None", ms=3, color="black",
             label="arousal classes: low=-1 | high=1")
    # midline and tertile borders
    plt.hlines(y=0, xmin=0, xmax=pred_matrix.shape[1], colors="darkgrey", lw=lw, alpha=.8)
    plt.hlines(y=lower_tert_bound, xmin=0, xmax=pred_matrix.shape[1], linestyle="dashed",
               colors="darkgrey", lw=lw, alpha=.8)
    plt.hlines(y=upper_tert_bound, xmin=0, xmax=pred_matrix.shape[1], linestyle="dashed",
               colors="darkgrey", lw=lw, alpha=.8)
    # Correct classified
    correct_class_train = np.sign(average_train_pred * whole_rating)

    plt.fill_between(x=np.arange(0, correct_class_train.shape[0], 1), y1=average_train_pred,
                     y2=real_rating, where=correct_class_train == 1, color="green", alpha='0.2')

    plt.fill_between(x=np.arange(0, correct_class_train.shape[0], 1), y1=average_train_pred,
                     y2=real_rating, where=correct_class_train == -1, color="red", alpha='0.2')

    for i in range(correct_class_train.shape[0]):
        corr_col = "green" if correct_class_train[i] == 1 else "red"
        # wr = whole_rating[i]
        wr = whole_rating_shift[i]
        # plt.vlines(i, ymin=wr, ymax=average_train_pred[i], colors=corr_col, alpha=.5, lw=lw/1.5)
        if not np.isnan(average_train_pred[i]):
            # set a point at the end of line
            plt.plot(i, np.sign(wr) * (np.abs(wr) + .01) if correct_class_train[i] == 1 else wr,
                     marker="o", color=corr_col, alpha=.5, ms=2)

    correct_class_train = np.delete(correct_class_train, np.where(np.isnan(correct_class_train)))
    mean_train_acc = sum(correct_class_train[correct_class_train == 1]) / len(correct_class_train)
    plt.xlabel("time(s)")
    plt.title(label=f'Average train prediction | {s_fold}-Folds| {task} | '
                    f'mean training accuracy={mean_train_acc:.3f}')

# adjust size, add legend
plt.xlim(0, len(whole_rating))
plt.ylim(-1.2, 1.6)
plt.legend(bbox_to_anchor=(0., 0.90, 1., .102), loc=1, ncol=4, mode="expand", borderaxespad=0.)
plt.tight_layout(pad=2)

# Plot average train prediction
fig4.add_subplot(4, 1, 2)
if task == "regression":
    plt.plot(concat_val_pred, c="aquamarine", linewidth=2 * lw, label="concatenated val-prediction")
    plt.plot(whole_rating, ls="dotted", color="black", label="rating")
    plt.fill_between(x=np.arange(0, pred_matrix.shape[1], 1), y1=whole_rating, y2=concat_val_pred,
                     color="aquamarine", alpha='0.2')

    plt.hlines(y=0, xmin=0, xmax=pred_matrix.shape[1], colors="darkgrey", lw=lw, alpha=.8)  # midline
    plt.hlines(y=np.nanmean(concat_val_pred), xmin=0, xmax=pred_matrix.shape[1], linestyle="dashed",
               colors="aquamarine", lw=lw, alpha=.8)  # mean concat validation prediction
    plt.hlines(y=np.nanmean(whole_rating), xmin=0, xmax=pred_matrix.shape[1], linestyle="dotted",
               colors="black", lw=lw, alpha=.8)  # mean ratings

else:  # task == "classification":

    plt.plot(concat_val_pred, marker="o", markerfacecolor="None", ms=2, c="aquamarine", linewidth=2 * lw,
             label="concatenated val-prediction")
    # Ratings
    plt.plot(real_rating, color="darkgrey", alpha=.5)
    plt.plot(only_entries_rating, color="teal", alpha=.3, label="ground-truth rating")
    plt.plot(whole_rating_shift, ls="None", marker="s", markerfacecolor="None", ms=3, color="black",
             label="arousal classes: low=-1 | high=1")
    # midline and tertile borders
    plt.hlines(y=0, xmin=0, xmax=pred_matrix.shape[1], colors="darkgrey", lw=lw, alpha=.8)
    plt.hlines(y=lower_tert_bound, xmin=0, xmax=pred_matrix.shape[1], linestyle="dashed",
               colors="darkgrey", lw=lw, alpha=.8)
    plt.hlines(y=upper_tert_bound, xmin=0, xmax=pred_matrix.shape[1], linestyle="dashed",
               colors="darkgrey", lw=lw, alpha=.8)
    # Correct classified
    correct_class = np.sign(concat_val_pred * whole_rating)

    plt.fill_between(x=np.arange(0, correct_class.shape[0], 1), y1=concat_val_pred, y2=real_rating,
                     where=correct_class == 1,
                     color="lime",
                     alpha='0.2')

    plt.fill_between(x=np.arange(0, correct_class.shape[0], 1), y1=concat_val_pred, y2=real_rating,
                     where=correct_class == -1,
                     color="orangered",
                     alpha='0.2')

    for i in range(correct_class.shape[0]):
        corr_col = "lime" if correct_class[i] == 1 else "orangered"
        # wr = whole_rating[i]
        wr = whole_rating_shift[i]
        # plt.vlines(i, ymin=wr, ymax=concat_val_pred[i], colors=corr_col, alpha=.5, lw=lw/1.5)
        if not np.isnan(concat_val_pred[i]):
            # set a point at the end of line
            plt.plot(i, np.sign(wr) * (np.abs(wr) + .01) if correct_class[i] == 1 else wr,
                     marker="o", color=corr_col, alpha=.5, ms=2)

        # plt.vlines(i, ymin=concat_val_pred[i], ymax=wr, colors=corr_col, alpha=.5, lw=lw)
        # print(wr, concat_val_pred[i])

plt.xlabel("time(s)")
plt.title(label=f"Concatenated val-prediction | {s_fold}-Folds | {task} | "
                f"mean validation accuracy={mean_acc:.3f}")
# adjust size, add legend
plt.xlim(0, len(whole_rating))
plt.ylim(-1.2, 1.6)
plt.legend(bbox_to_anchor=(0., 0.90, 1., .102), loc=1, ncol=4, mode="expand", borderaxespad=0.)
plt.tight_layout(pad=2)

# # Plot i) average training & validation accuracy and ii) loss across folds

# Plot average training & validation accuracy
fig4.add_subplot(4, 1, 3)

plt.plot(x_fold_mean_tacc, color="steelblue", linewidth=lw / 2, alpha=0.6, label="mean training accuracy")
plt.plot(where, x_fold_mean_vacc, color="aquamarine", linewidth=2 * lw, alpha=0.9,
         label="mean validation accuracy")
if task == "regression":
    plt.hlines(y=mean_line_acc, xmin=0, xmax=train_acc_fold.shape[0], colors="red", linestyles="dashed",
               lw=2 * lw, label="meanline accuracy")
else:  # == "classification"
    plt.hlines(y=1.0, xmin=0, xmax=train_acc_fold.shape[0], colors="darkgrey", linestyles="dashed", lw=lw)
    plt.hlines(y=0.5, xmin=0, xmax=train_acc_fold.shape[0], colors="red", linestyles="dashed", lw=lw)
plt.xlabel("Training iterations")
plt.ylabel("Accuracy")
plt.title(label="Across all S-folds | mean training & validation accuracy")

# adjust size, add legend
plt.xlim(0, len(train_acc_fold))
plt.ylim(0.5, 1.2)
plt.legend(bbox_to_anchor=(0., 0.90, 1., .102), loc=1, ncol=4, mode="expand", borderaxespad=0.)
plt.tight_layout(pad=2)

# Plot average training & validation loss
fig4.add_subplot(4, 1, 4)

plt.plot(x_fold_mean_tloss, color="steelblue", linewidth=lw / 2, alpha=0.6,
         label="mean training loss")
plt.plot(where_loss, x_fold_mean_vloss, color="aquamarine", linewidth=2 * lw, alpha=0.9,
         label="mean validation loss")
plt.hlines(y=0.0, xmin=0, xmax=x_fold_mean_tloss.shape[0], colors="darkgrey", linestyles="dashed", lw=lw)
plt.xlabel("Training iterations")
plt.ylabel("Loss")
plt.title(label="Across all S-folds | mean training & validation loss")

# adjust size, add legend
plt.xlim(0, len(train_loss_fold))
plt.ylim(-0.05, 1.8)
plt.legend(bbox_to_anchor=(0., 0.90, 1., .102), loc=1, ncol=4, mode="expand", borderaxespad=0.)
plt.tight_layout(pad=2)

if matplotlib.rcParams['backend'] != 'agg':
    fig4.show()

# Plot
if plots:
    plot_filename = f"{file_name[0:10]}{abc}_|{'_Hilbert_' if hilb else '_'}" \
        f"{int(reps)}*{'rnd-batch' if rnd_batch else 'subsequent-batch'}({batch_size})_|_{s_fold}-Folds" \
        f"_|_{task}_|_all_train_val_|_{s(subject)}_|_{cond}_|_mean(val_acc)_{mean_acc:.2f}_|_" \
        f"{path_specificity[:-1]}.png"

    fig4.savefig(wdic_plot + plot_filename)


# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

@true_false_request
def close_plots():
    print("Do you want to close plots?")


if not script_external_exe:  # Check whether script is opened from intern(python) or extern(terminal)
    if close_plots():
        for _ in range(4):
            plt.close()
else:
    for _ in range(4):
        plt.close()

# When saved then move *.csv & *.txt files into folder "Already Plotted"
if plots:
    for file in os.listdir(wdic_sub):
        if file.split(".")[-1] in ['txt', 'csv', 'npy']:
            new_file_name = file.split("_S")[0] + abc + "_S" + file.split("_S")[1]

            while True:
                try:
                    gfile.Rename(oldname=wdic_sub + file, newname=already_plotted_dic + new_file_name,
                                 overwrite=False)
                    break
                except Exception:
                    new_file_name = new_file_name.split(abc + "_S")[0] + \
                                    string.ascii_lowercase[abc_counter] + "_S" + \
                                    new_file_name.split("_S")[1]
                    abc = string.ascii_lowercase[abc_counter]
                    abc_counter += 1

    # Delete corresponding folder (if empty)
    if len(os.listdir(wdic_sub)) == 0:
        gfile.DeleteRecursively(wdic_sub)

    # open folder
    if not script_external_exe:
        open_folder(wdic_plot)

    # delete log + checkpoint folders and subfolders
    if delete_log_folder:
        shutil.rmtree(wdic_lists_sub)
        shutil.rmtree(wdic_checkpoint_sub)
        # Delete also /logs/S.. folder (parent) if empty
        sub_log_dir = wdic_lists + f"/{s(subject)}/"
        sub_ckpt_dir = wdic_checkpoint_sub.split("/")
        sub_ckpt_dir.pop(-2)
        sub_ckpt_dir = "/".join(sub_ckpt_dir)
        if len(os.listdir(sub_log_dir)) == 0:
            shutil.rmtree(sub_log_dir)
        if len(os.listdir(sub_ckpt_dir)) == 0:
            shutil.rmtree(sub_ckpt_dir)

    else:  # In case files are saved on MPI GPU server, delete manually:
        cprint("Delete log and checkpoint files manually.", "y")

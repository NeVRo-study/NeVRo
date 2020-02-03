# coding=utf-8
"""
Plot predictions made by the different models of the study

Author: Simon Hofmann | <[surname].[lastname][at]pm.me> | 2020
"""

from load_data import *
import sys
from tensorflow import gfile
import string
import shutil
import ast
import matplotlib
import matplotlib.pyplot as plt


# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

lw = 0.5  # linewidth
plt_input_data = False
plots = False

# Models
model = "LSTM"  # "CSP"

if model == "CSP":
    NotImplementedError("Plotting for CSP is not implemented yet")

# Subjects
subject = 17

# Condition
cond = "nomov"

# Set paths

wdic_results = f"../../../Results/"
wdic_plot = wdic_results + f"Plots/{model}/"

# For "LSTM"
wdic = "./processed"
wdic_sub = wdic + f"/{cond}/{s(subject)}/already_plotted/"

wdic_val_pred = wdic_results + f"Tables/{model}/{cond}/"

path_specificity = "BiCl_nomov_RndHPS_lstm-20-15_fc-0_lr-1e-3_wreg-l2-0.36_actfunc-relu_ftype-SSD_hilb-F_bpass-T_comp-1_hrcomp-F_fixncol-10"

# Find correct files (csv-tables)
shuff_filename = "None"
file_name = ''  # init
acc_filename = ''  # init
val_filename = ''  # init
for file in os.listdir(wdic_sub):
    if path_specificity in file:
        if ".csv" in file:
            if "val_" in file:
                val_filename = file
            else:
                file_name = file

        elif ".txt" in file:
            acc_filename = file

        elif ".npy" in file:
            shuff_filename = file

# Load data
pred_matrix = np.loadtxt(wdic_sub + file_name, delimiter=",")
val_pred_matrix = np.loadtxt(wdic_sub + val_filename, delimiter=",")

# Look for shuffle matrix
if os.path.exists(wdic_sub + shuff_filename):
    shuffle_order_matrix = np.load(wdic_sub + shuff_filename)

    # shuffle_order_fold5 = np.load(wdic_lists + "/S22/test_task-class_shuffle-T/5/5_shuffle_order.npy")
    dub_shuffle_order_matrix = np.repeat(a=shuffle_order_matrix, repeats=2, axis=0)

    # Correct oder of matrices according to shuffle order of each fold (saved in shuffle_order_matrix)
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


whole_rating_shift = copy.copy(whole_rating)
for idx, ele in enumerate(whole_rating):
    if ele == 1.:
        whole_rating_shift[idx] = ele + .1
    elif ele == -1.:
        whole_rating_shift[idx] = ele - .1


# Get concatenated validation predictions for given subject
concat_val_pred_lstm = pd.read_csv(wdic_val_pred + f"predictionTableProbabilities{model}_{cond}.csv",
                                   header=None, index_col=0).loc[f"NVR_{s(subject)}"].to_numpy()

# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

# # # Plotting
fig = plt.figure(f"{s_fold}-Folds concat(val) | {s(subject)} | {cond} | {task} | mean(val_acc)={mean_acc} | 1Hz ",
                 figsize=(10, 12))

fig.add_subplot(3, 1, 1)

# Plot val predictions
plt.plot(concat_val_pred_lstm, marker="o", markerfacecolor="None", ms=2, c="aquamarine", linewidth=2 * lw,
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
correct_class = np.sign(concat_val_pred_lstm * whole_rating)

plt.fill_between(x=np.arange(0, correct_class.shape[0], 1), y1=concat_val_pred_lstm, y2=real_rating,
                 where=correct_class == 1,
                 color="lime",
                 alpha='0.2')

plt.fill_between(x=np.arange(0, correct_class.shape[0], 1), y1=concat_val_pred_lstm, y2=real_rating,
                 where=correct_class == -1,
                 color="orangered",
                 alpha='0.2')

for i in range(correct_class.shape[0]):
    corr_col = "lime" if correct_class[i] == 1 else "orangered"
    # wr = whole_rating[i]
    wr = whole_rating_shift[i]
    # plt.vlines(i, ymin=wr, ymax=concat_val_pred[i], colors=corr_col, alpha=.5, lw=lw/1.5)
    if not np.isnan(concat_val_pred_lstm[i]):
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


# TODO CSP data

# For "CSP"
model = "CSP"
wdic_csp = f"../../../Results/CSP/{cond}/"
wdic_val_pred = wdic_results + f"Tables/{model}/{cond}/"

concat_val_pred = pd.read_csv(wdic_val_pred + f"predictionTableProbabilities{model}_{cond}.csv",
                              header=None, index_col=0).loc[f"NVR_{s(subject)}"].to_numpy()  # (270,)


# Plot average val prediction of CSP
fig.add_subplot(3, 1, 2)
plt.plot(concat_val_pred, marker="o", markerfacecolor="None", ms=2, c="aquamarine", linewidth=2 * lw,
         label="CSP concatenated val-prediction ")
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


# TODO SPoC: fig.add_subplot(3, 1, 3)

if matplotlib.rcParams['backend'] != 'agg':
    fig.show()

# Plot
if plots:
    plot_filename = f"{file_name[0:10]}_|{'_Hilbert_' if hilb else '_'}" \
        f"{int(reps)}*{'rnd-batch' if rnd_batch else 'subsequent-batch'}({batch_size})_|_{s_fold}-Folds" \
        f"_|_{task}_|_all_train_val_|_{s(subject)}_|_{cond}_|_mean(val_acc)_{mean_acc:.2f}_|_" \
        f"{path_specificity[:-1]}.png"

    # fig.savefig(wdic_plot + plot_filename)


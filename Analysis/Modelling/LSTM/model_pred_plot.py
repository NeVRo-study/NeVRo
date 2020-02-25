# coding=utf-8
"""
Plot predictions made by the different models of the study

Author: Simon Hofmann | <[surname].[lastname][at]pm.me> | 2020
"""

from load_data import *
from tensorflow import gfile
import matplotlib
import matplotlib.pyplot as plt


# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

plots = False

# Subjects
subject = 34

# Condition
cond = "nomov"

# Set paths

# Models
models = ["LSTM", "CSP", "SPoC"]
for midx, model in enumerate(models):

    wdic_results = f"../../../Results/"
    wdic_plot = wdic_results + f"Plots/{model}/"
    wdic_val_pred = wdic_results + f"Tables/{model}/{cond}/"

    if model == "LSTM":
        wdic = "./processed"
        wdic_sub = wdic + f"/{cond}/{s(subject)}/already_plotted/"
        # TODO path_specificity needs to be extracted from ... AllSub_Random_Search_Table_(no)mov_BiCl:
        path_specificity = "BiCl_nomov_RndHPS_lstm-50-40_fc-10_lr-5e-4_wreg-l2-0.00_actfunc-relu_" \
                           "ftype-SSD_hilb-F_bpass-T_comp-1-2-3-4_hrcomp-F_fixncol-10_shuf-T_balcv-T"

        # Find correct files (csv-tables)
        file_name, val_filename, acc_filename, shuff_filename = filnames_processed_models(
            wdic_of_subject=wdic_sub, path_specificity=path_specificity)

        # Load data
        pred_matrix = np.loadtxt(wdic_sub + file_name, delimiter=",")
        val_pred_matrix = np.loadtxt(wdic_sub + val_filename, delimiter=",")

        # Look for shuffle matrix
        if os.path.exists(wdic_sub + shuff_filename):
            shuffle_order_matrix = np.load(wdic_sub + shuff_filename)

            dub_shuffle_order_matrix = np.repeat(a=shuffle_order_matrix, repeats=2, axis=0)

            # Correct order of mat according to shuf order of each fold (saved in shuffle_order_matrix)
            pred_matrix = sort_mat_by_mat(mat=pred_matrix, mat_idx=dub_shuffle_order_matrix)

            val_pred_matrix = sort_mat_by_mat(mat=val_pred_matrix, mat_idx=dub_shuffle_order_matrix)
            del dub_shuffle_order_matrix

        # Number of Folds
        s_fold = int(len(pred_matrix[:, 0]) / 2)

        # Import accuracies
        acc_date = np.loadtxt(wdic_sub + acc_filename, dtype=str, delimiter=";")  # Check (before ",")

        # First initialize variables
        task = ''  # init
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
        whole_rating = load_rating_files(
            subjects=subject, condition=cond,
            bins=False if task == "regression" else True)[str(subject)]["SBA"][cond]

        real_rating = None  # init
        only_entries_rating = None  # init
        lower_tert_bound = None  # init
        upper_tert_bound = None  # init
        if task == "classification":
            whole_rating[np.where(whole_rating == 0)] = np.nan
            real_rating = normalization(array=load_rating_files(subjects=subject,
                                                                condition=cond,
                                                                sba="SBA",
                                                                bins=False)[str(subject)]["SBA"][cond],
                                        lower_bound=-1, upper_bound=1)  # range [-1, 1]

            tertile = int(len(real_rating) / 3)
            lower_tert_bound = np.sort(real_rating)[tertile]  # np.percentile(a=real_rating, q=33.33)
            upper_tert_bound = np.sort(real_rating)[tertile * 2]  # np.percentile(a=real_rating, q=66.66)

            no_entries = np.isnan(np.nanmean(a=np.vstack((pred_matrix, val_pred_matrix)), axis=0))
            only_entries_rating = real_rating.copy()
            only_entries_rating[np.where(no_entries)] = np.nan

            # Exchange mean_acc, if:
            mean_acc = np.round(np.mean(calc_binary_class_accuracy(prediction_matrix=val_pred_matrix)), 3)
            # == np.round(np.nanmean(val_class_acc), 3)
        else:
            whole_rating = normalization(array=whole_rating, lower_bound=-1, upper_bound=1)  # [-1, 1]

        whole_rating_shift = copy.copy(whole_rating)
        for idx, ele in enumerate(whole_rating):
            if ele == 1.:
                whole_rating_shift[idx] = ele + .1
            elif ele == -1.:
                whole_rating_shift[idx] = ele - .1

    # Get concatenated validation predictions for given subject and model
    model_prediction = pd.read_csv(wdic_val_pred + f"predictionTableProbabilities{model}_{cond}.csv",
                                   header=None, index_col=0).loc[f"NVR_{s(subject)}"].to_numpy()

    # Normalize for plotting
    if model == "SPoC":
        model_prediction = z_score(model_prediction)  # before only >= 0, now zero-centred
    model_prediction /= np.nanmax(np.abs(model_prediction))  # keep in range [-1,1]

# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

    # # # Plotting
    if midx == 0:
        fig = plt.figure(f"Predictions of all models | {s(subject)} | {cond} | {task} | 1Hz ",
                         figsize=(10, 12))

    ax = fig.add_subplot(3, 1, midx+1)

    # Plot val predictions
    lw = 0.5  # line-width
    plt.plot(model_prediction, marker="o", markerfacecolor="None", ms=2, c="aquamarine",
             linewidth=2 * lw, label="concatenated val-prediction")

    # Ratings
    plt.plot(real_rating, color="darkgrey", alpha=.5)
    if model != "SPoC":
        plt.plot(only_entries_rating, color="teal", alpha=.3, label="ground-truth rating")
        plt.plot(whole_rating_shift, ls="None", marker="s", markerfacecolor="None", ms=3, color="black",
                 label="arousal classes: low=-1 | high=1")

    # midline and tertile borders
    plt.hlines(y=0, xmin=0, xmax=pred_matrix.shape[1], colors="darkgrey", lw=lw, alpha=.8)
    if model != "SPoC":
        plt.hlines(y=lower_tert_bound, xmin=0, xmax=pred_matrix.shape[1], linestyle="dashed",
                   colors="darkgrey", lw=lw, alpha=.8)
        plt.hlines(y=upper_tert_bound, xmin=0, xmax=pred_matrix.shape[1], linestyle="dashed",
                   colors="darkgrey", lw=lw, alpha=.8)

    # Correct classified
    if model != "SPoC":
        correct_class = np.sign(model_prediction * whole_rating)
        mean_acc2 = correct_class[correct_class == 1].sum() / np.sum(~np.isnan(correct_class))
        print(f"Calculated {model} accuracy: {mean_acc2:.2f}")
        if (np.round(mean_acc2, 3) != mean_acc) and model == "LSTM":
            # for LSTM: there is the balanced_cv=False (in load_data.py), which leads to possible
            # differences of the calculation of mean_acc. In that case, mean_acc2 is to be ignored.
            pass
        else:
            mean_acc = mean_acc2

        plt.fill_between(x=np.arange(0, correct_class.shape[0], 1), y1=model_prediction, y2=real_rating,
                         where=correct_class == 1,
                         color="lime",
                         alpha='0.2')

        plt.fill_between(x=np.arange(0, correct_class.shape[0], 1), y1=model_prediction, y2=real_rating,
                         where=correct_class == -1,
                         color="orangered",
                         alpha='0.2')

        for i in range(correct_class.shape[0]):
            corr_col = "lime" if correct_class[i] == 1 else "orangered"
            # wr = whole_rating[i]
            wr = whole_rating_shift[i]
            # plt.vlines(i, ymin=wr, ymax=model_prediction[i], colors=corr_col, alpha=.5, lw=lw/1.5)
            if not np.isnan(model_prediction[i]):
                # set a point at the end of line
                plt.plot(i, np.sign(wr) * (np.abs(wr) + .01) if correct_class[i] == 1 else wr,
                         marker="o", color=corr_col, alpha=.5, ms=2)

    else:
        plt.fill_between(x=np.arange(0, model_prediction.shape[0], 1), y1=model_prediction, y2=real_rating,
                         # where=correct_class == -1,
                         color="gray",
                         alpha='0.2')

    # Adapt y-axis ticks
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels([-1, 0, 1])

    # Set x-label
    if midx == len(models)-1:
        plt.xlabel("time(s)")

    # Set title
    if model != "SPoC":  # LSTM, CSP:
        plt.title(label=f"{model} | concatenated predictions on validation sets | "
                        f"mean {task}-accuracy={mean_acc:.3f}")
    else:  # for SPoC:
        plt.title(label=f"{model} | {'W*X'} | max-correlation | "
                        f"r={-.527:.3f}")  # TODO S34
                        # f"r={mean_acc:.3f}")  # TODO read max.corr

    # adjust size, add legend
    plt.xlim(0, len(whole_rating))
    plt.ylim(-1.2, 1.6 if midx == 0 else 1.2)
    if midx == 0:
        plt.legend(bbox_to_anchor=(0., 0.90, 1., .102), loc=1, ncol=4, mode="expand", borderaxespad=0.)
    plt.tight_layout(pad=2)

    if matplotlib.rcParams['backend'] != 'agg':
        fig.show()

    # Plot
    if plots:
        plot_filename = f"Prediction across models |_{s(subject)}_|_{cond}.png"
        fig.savefig(wdic_results + plot_filename)
        plt.close()

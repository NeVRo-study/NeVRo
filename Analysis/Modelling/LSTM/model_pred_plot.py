# coding=utf-8
"""
Plot predictions made by the different models of the study

Author: Simon Hofmann | <[surname].[lastname][at]pm.me> | 2020
"""

from load_data import *
# from tensorflow import gfile
import matplotlib
import matplotlib.pyplot as plt


# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

plots = True

task = "classification"

# Set Paths
wdic_results = f"../../../Results/"

for cond in ["nomov", "mov"]:  # Conditions

    # Read results per method/model:
    p2results = f"../../../Results/Tables/results_across_methods_{cond}.csv"
    result_tab = pd.read_csv(p2results, sep=",", index_col="Subject").dropna()  # drop NaNs

    # Subjects
    subjects = [int(sub.lstrip("NVR_S")) for sub in result_tab.index.to_list()]

    for subject in subjects:

        # Models
        models = ["LSTM", "CSP", "SPoC"]
        for midx, model in enumerate(models):

            wdic_plot = wdic_results + f"Plots/{model}/"
            wdic_val_pred = wdic_results + f"Tables/{model}/{cond}/"

            if model == "LSTM":
                wdic = "./processed"
                wdic_sub = wdic + f"/{cond}/{s(subject)}/already_plotted/"
                # Get path specs from table of model performance per subject
                wdic_pathspecs = wdic+f"/Random_Search_Tables/{cond}/1_narrow_search/{task}/per_subject/"
                perform_tab = pd.read_csv(wdic_pathspecs+f"AllSub_Narrow_Search_Table_{cond}_BiCl.csv",
                                          sep=";")
                path_specificity = perform_tab.loc[perform_tab["subject"] ==
                                                   str(subject).zfill(2)]["path_specificities"].item()
                mean_acc1 = perform_tab.loc[perform_tab["subject"] ==
                                            str(subject).zfill(2)]["mean_class_val_acc"].item()

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

                    # Correct order of mat wrt. to shuf order of each fold (saved in shuffle_order_matrix)
                    pred_matrix = sort_mat_by_mat(mat=pred_matrix, mat_idx=dub_shuffle_order_matrix)

                    val_pred_matrix = sort_mat_by_mat(mat=val_pred_matrix,
                                                      mat_idx=dub_shuffle_order_matrix)
                    del dub_shuffle_order_matrix

                # Number of Folds
                s_fold = int(len(pred_matrix[:, 0]) / 2)

                # Load full rating vector
                whole_rating = load_rating_files(subjects=subject,
                                                 condition=cond,
                                                 bins=True)[str(subject)]["SBA"][cond]

                real_rating = None  # init
                only_entries_rating = None  # init
                lower_tert_bound = None  # init
                upper_tert_bound = None  # init

                whole_rating[np.where(whole_rating == 0)] = np.nan
                real_rating = normalization(array=load_rating_files(
                    subjects=subject,
                    condition=cond,
                    sba="SBA",
                    bins=False)[str(subject)]["SBA"][cond], lower_bound=-1, upper_bound=1)  # range -1, 1

                tertile = int(len(real_rating) / 3)
                lower_tert_bound = np.sort(real_rating)[tertile]  # np.percentile(a=real_rating, q=33.33)
                upper_tert_bound = np.sort(real_rating)[tertile * 2]  # ... q=66.66)

                no_entries = np.isnan(np.nanmean(a=np.vstack((pred_matrix, val_pred_matrix)), axis=0))
                only_entries_rating = real_rating.copy()
                only_entries_rating[np.where(no_entries)] = np.nan

                whole_rating_shift = copy.copy(whole_rating)
                for idx, ele in enumerate(whole_rating):
                    if ele == 1.:
                        whole_rating_shift[idx] = ele + .1
                    elif ele == -1.:
                        whole_rating_shift[idx] = ele - .1

            # Get concatenated validation predictions for given subject and model
            model_prediction = pd.read_csv(
                wdic_val_pred + f"predictionTableProbabilities{model}_{cond}.csv",
                header=None, index_col=0).loc[f"NVR_{s(subject)}"].to_numpy()

            # Normalize for plotting
            if model == "SPoC":
                model_prediction = z_score(model_prediction)  # before only >= 0, now zero-centred
            model_prediction /= np.nanmax(np.abs(model_prediction))  # keep in range [-1,1]

        # < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >

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
                plt.plot(whole_rating_shift, ls="None", marker="s", markerfacecolor="None", ms=3,
                         color="black", label="arousal classes: low=-1 | high=1")

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
                # mean_acc2 = correct_class[correct_class == 1].sum() / np.sum(~np.isnan(correct_class))
                mean_acc = float(result_tab.loc[f"NVR_{s(subject)}"][model])
                print(f"Calculated {model} accuracy: {mean_acc:.4f}")

                plt.fill_between(x=np.arange(0, correct_class.shape[0], 1), y1=model_prediction,
                                 y2=real_rating,
                                 where=correct_class == 1,
                                 color="lime",
                                 alpha='0.2')

                plt.fill_between(x=np.arange(0, correct_class.shape[0], 1), y1=model_prediction,
                                 y2=real_rating,
                                 where=correct_class == -1,
                                 color="orangered",
                                 alpha='0.2')

                for i in range(correct_class.shape[0]):
                    corr_col = "lime" if correct_class[i] == 1 else "orangered"
                    # wr = whole_rating[i]
                    wr = whole_rating_shift[i]
                    # plt.vlines(i, ymin=wr, ymax=model_prediction[i], colors=corr_col, alpha=.5,
                    #            lw=lw/1.5)
                    if not np.isnan(model_prediction[i]):
                        # set a point at the end of line
                        plt.plot(i, np.sign(wr) * (np.abs(wr) + .01) if correct_class[i] == 1 else wr,
                                 marker="o", color=corr_col, alpha=.5, ms=2)

            else:
                pass
                # plt.fill_between(x=np.arange(0, model_prediction.shape[0], 1),
                #                  y1=model_prediction,
                #                  y2=real_rating,
                #                  # where=correct_class == -1,
                #                  color="gray",
                #                  alpha='0.2')

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
                r = float(result_tab.loc[f"NVR_{s(subject)}"][model.upper()+"_CORR"])
                plt.title(label=f"{model} | {'W*X'} | max-correlation | r={r:.3f}")

            # adjust size, add legend
            plt.xlim(0, len(whole_rating))
            plt.ylim(-1.2, 1.6 if midx == 0 else 1.2)
            if midx == 0:
                plt.legend(bbox_to_anchor=(0., 0.90, 1., .102), loc=1, ncol=4, mode="expand",
                           borderaxespad=0.)
            plt.tight_layout(pad=2)

        # # Show & save plot
        if matplotlib.rcParams['backend'] != 'agg':
            fig.show()

        # Plot
        if plots:
            plot_filename = f"Prediction across models |_{s(subject)}_|_{cond}.png"
            fig.savefig(wdic_results + f"Plots/Across_Models/{cond}/" + plot_filename)
            plt.close()

# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<  END

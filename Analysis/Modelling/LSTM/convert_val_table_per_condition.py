# coding=utf-8
"""
Convert validation results into a table per condition

Author: Simon Hofmann | <[surname].[lastname][at]pm.me> | 2020
"""

from load_data import *
import pandas as pd

# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

# # Convert validation results into a table per condition and save

# # Set paths
wdic = f"./processed/"
wdic_result = "../../../Results/Tables/LSTM/"
if not os.path.exists(wdic_result):
    os.makedirs(wdic_result, exist_ok=True)

# Subjects
n_subs = 45

# # Do in each condition
for cond in ["mov", "nomov"]:

    # # Init tables
    # val_pred_table_per_cond_probs[np.where(val_pred_table_per_cond_probs == 0.)] = np.nan
    val_pred_table_per_cond_probs = np.zeros(shape=(n_subs, 270))  # write probabilities
    val_pred_table_per_cond = copy.copy(val_pred_table_per_cond_probs)  # take sign
    target_table_per_cond = copy.copy(val_pred_table_per_cond_probs)  # take sign
    ls_subs = []

    # # Set paths
    wdic_rs_tables = wdic + f"Random_Search_Tables/{cond}/1_narrow_search/classification/per_subject/"
    p2_path_specificity = wdic_rs_tables + f"AllSub_Narrow_Search_Table_{cond}_BiCl.csv"

    # Get specific paths from best HPsets
    path_specificity_tab = np.loadtxt(p2_path_specificity, delimiter=";", dtype=str)

    # Iterate through subjects
    for isub in range(1, n_subs+1):

        wdic_sub = wdic + f"{cond}/{s(isub)}/already_plotted/"

        if not os.path.exists(wdic_sub):
            continue

        # Get best HPset specificities
        sidx = np.where(path_specificity_tab[:, 0] == str(isub).zfill(2))[0]

        if len(sidx) == 0:  # if there is no entry in path_specificity_tab
            continue

        ls_subs.append(isub)
        path_specificity = path_specificity_tab[sidx, 1][0].rstrip("/")

        # # Find correct files (csv-tables)
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

        # # Load validation results
        val_pred_matrix = np.loadtxt(wdic_sub + val_filename, delimiter=",")
        # row fold*2: pred; row fold*2+1: ground truth

        # # Correctly order val_pred_matrix in each fold via shuffle matrix
        shuffle_order_matrix = np.load(wdic_sub + shuff_filename)
        dub_shuffle_order_matrix = np.repeat(a=shuffle_order_matrix, repeats=2, axis=0)

        val_pred_matrix = sort_mat_by_mat(mat=val_pred_matrix, mat_idx=dub_shuffle_order_matrix)  # (20, 270)
        val_pred_matrix = val_pred_matrix[np.arange(0, 20, 2), :]  # without ground truth, shape (10, 270)

        # Fill pred vector with predictions (in binary form)
        concat_val_pred = np.nanmean(a=val_pred_matrix, axis=0)
        # could be also majority vote for steps with > 1 entries

        # Validation Predictions
        val_pred_table_per_cond_probs[isub-1, :] = concat_val_pred
        val_pred_table_per_cond[isub-1, :] = np.sign(concat_val_pred)
        # Targets
        target_table_per_cond[isub-1, :] = load_rating_files(subjects=isub, condition=cond,
                                                             bins=True)[str(isub)]["SBA"][cond]

    # Save table
    sub_idx = np.array(["NVR_"+s(sub) for sub in ls_subs])

    pd.DataFrame(val_pred_table_per_cond[np.array(ls_subs)-1, :], index=sub_idx).to_csv(
        path_or_buf=wdic_result + f"predictionTableLSTM_{cond}.csv", sep=",", header=False, na_rep="NaN")

    pd.DataFrame(val_pred_table_per_cond_probs[np.array(ls_subs)-1, :], index=sub_idx).to_csv(
        path_or_buf=wdic_result + f"predictionTableProbabilitiesLSTM_{cond}.csv", sep=",", header=False,
        na_rep="NaN")

    pd.DataFrame(target_table_per_cond[np.array(ls_subs)-1, :], index=sub_idx).to_csv(
        path_or_buf=wdic_result + f"targetTableLSTM_{cond}.csv", sep=",", header=False, na_rep="NaN")

# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<  END

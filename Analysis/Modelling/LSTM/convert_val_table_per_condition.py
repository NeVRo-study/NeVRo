# coding=utf-8
"""
Convert validation results into a table per condition

Author: Simon Hofmann | <[surname].[lastname][at]pm.me> | 2020
"""

import sys
# sys.path.insert(0, './LSTM Model')  # or set the folder as source root
from load_data import *
import string
import shutil
import ast

# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

# Set paths
wdic = f"./processed/"
wdic_result = "../../../Results/Tables/"

for cond in ["mov", "nomov"]:

    wdic_rs_tables = wdic + f"Random_Search_Tables/{cond}/1_narrow_search/classification/per_subject/"
    p2_path_specificity = wdic_rs_tables + f"AllSub_Narrow_Search_Table_{cond}_BiCl.csv"

    path_specificity_tab = np.loadtxt(p2_path_specificity, delimiter=";", dtype=str)

    for isub in range(1, 46):

        wdic_sub = wdic + f"{cond}/{s(isub)}/already_plotted"

        if not os.path.exists(wdic_sub):
            continue

        path_specificity = path_specificity_tab[np.where(path_specificity_tab[:, 0] == str(isub)), 1][0][0].rstrip("/")

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

        # TODO continue here
        pred_matrix = np.loadtxt(wdic_sub + file_name, delimiter=",")
        val_pred_matrix = np.loadtxt(wdic_sub + val_filename, delimiter=",")

# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<  END

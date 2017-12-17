"""
After we run the gridsearch on S36, we find now the best hyperparamter from the processed files.
"""

import os
import subprocess
import numpy as np
import copy

subjects = [2, 36]

for sub in subjects:

    wdic = "./LSTM/S{}/already_plotted/".format(str(sub).zfill(2))
    # wdic = "./LSTM/S02/already_plotted/"
    wdic_plot = "../../Results/Plots/LSTM/Hyperparameter_Search/"

    acc_name_list = []
    acc_list = []

    for file_name in os.listdir(wdic):
        if "txt" in file_name:
            with open(wdic+file_name) as f:
                for line_terminated in f:
                    line = line_terminated.rstrip("\n")
                    if "mean(Accuracy)" in line:
                        accuracy = line.split(" ")[1]
                        # print(accuracy)

                        # Fill in lists
                        acc_list.append(float(accuracy))
                        acc_name_list.append(file_name)

    acc_list_sorted, acc_name_list_sorted = map(list, zip(*sorted(zip(acc_list, acc_name_list))[::-1]))

    print("\n")
    for i, j in zip(acc_list_sorted[:5], acc_name_list_sorted[:5]):
        print(i, "\t\t\t", j)

    for file_n in acc_name_list_sorted[:10]:
        for plot_file in os.listdir(wdic_plot):
            try:
                identifier = file_n.split("folds_")[1].split(".")[0]
            except IndexError:
                identifier = file_n.split("_S{}".format(sub))[0]

            if identifier in plot_file and "_all_train_val_" in plot_file:
                current_plot_file = wdic_plot+plot_file
                subprocess.Popen(["open", current_plot_file])  # 'open' only works for Mac


# Merge Random Search Tables from server and local computer:
def sort_table(task, table=None):
    save_externally = False
    if table is None:
        task_fix = "_BiCl_merged.csv" if task.lower() == "classification" else "_Reg_merged.csv"
        file_found = False
        for file in os.listdir("./LSTM/"):
            if "Random_Search_Table" in file and task_fix in file:
                table_filename = "./LSTM/" + file
                file_found = True
                save_externally = True

        if file_found:
            table = np.genfromtxt(table_filename, delimiter=";", dtype=str)
        else:
            raise FileNotFoundError("No table given and no table found in './LSTM/'")

    acc_col = -1 if task == "classification" else -3
    sorted_table = copy.copy(table)
    sorted_table[1:, :] = sorted_table[np.argsort(sorted_table[1:, acc_col]) + 1]
    if "nan" in sorted_table[:, acc_col]:
        nan_start = np.where(sorted_table[:, acc_col] == "nan")[0][0]
    else:
        nan_start = sorted_table.shape[0]
    sorted_table[1:nan_start, ] = sorted_table[np.argsort(sorted_table[1:nan_start, acc_col]) + 1, ][::-1]

    if save_externally:
        np.savetxt(fname="." + table_filename.split(".")[1] + "_sorted.csv", X=sorted_table, delimiter=";", fmt="%s")
    else:
        return sorted_table


def merge_randsearch_tables(task, sort=True):
    """
    Merges random search table from server with the one from local computer and saves them
    :param task: either 'classification' or 'regression'
    :param sort: sort table according to accuracy
    """

    # Check function input
    assert task.lower() in ["classification", "regression"], "task must be either 'classification' or 'regression'"

    # Find tables
    wd_table = "./LSTM/"
    for file in os.listdir(wd_table):
        if "Random_Search_Table" in file and "_server.csv" not in file:
            local_table_filename = wd_table + file
        elif "Random_Search_Table" in file and "_server.csv" in file:
            server_table_filename = wd_table + file

    try:
        # merge tables
        if local_table_filename.split(".")[1] in server_table_filename:
            rs_table = np.genfromtxt(local_table_filename, delimiter=";", dtype=str)
            rs_table_server = np.genfromtxt(server_table_filename, delimiter=";", dtype=str)
            for row in range(rs_table.shape[0]):
                if rs_table[row, -3] == "nan":
                    if rs_table_server[row, -3] != "nan":
                        rs_table[row, -3:] = rs_table_server[row, -3:]

            # Sort table
            if sort:
                rs_table = sort_table(table=rs_table, task=task)

            # Save file
            export_filename = "." + local_table_filename.split(".")[1] + "_merged.csv"
            if sort:
                export_filename = "." + export_filename.split(".")[1] + "_sorted.csv"
            np.savetxt(fname=export_filename, X=rs_table, delimiter=";", fmt="%s")

    except NameError:
        raise FileExistsError("There is no set of tables to merge.")


# merge_randsearch_tables(task="classification", sort=True)
# sort_table(task="classification")

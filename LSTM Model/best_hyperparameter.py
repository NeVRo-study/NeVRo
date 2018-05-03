# coding=utf-8
"""
After we run grid or random search on subjects, we find now the best hyperparameters from the processed files.

Author: Simon Hofmann | <[surname].[lastname][at]protonmail.com> | 2017
"""

import os
import subprocess
import numpy as np
import copy


def open_best_params(subjects, rand_search, task, n=5):
    """
    Print filenames and open plots of best hyperparameters per subjects
    :param subjects: list of subjects
    :param rand_search: From Random Search: True or False
    :param task: "classification" or "regression"
    :param n: number of settings to print and plot
    :return:
    """

    assert task in ["classification", "regression"], "Task must be either 'classification' or 'regression'."

    if not isinstance(subjects, list):
        subjects = [subjects]

    for sub in subjects:

        wdic = "./LSTM/S{}/already_plotted/".format(str(sub).zfill(2))
        # wdic = "./LSTM/S02/already_plotted/"
        if rand_search:
            wdic_plot = "../../Results/Plots/LSTM/Hyperparameter_Search_C_RndSearch/"
        else:
            wdic_plot = "../../Results/Plots/LSTM/Hyperparameter_Search/"

        acc_name_list = []
        acc_list = []

        for file_name in os.listdir(wdic):
            if "txt" in file_name:
                with open(wdic+file_name) as f:
                    # classification = False  # init
                    for line_terminated in f:
                        line = line_terminated.rstrip("\n")

                        # if "Task" in line:
                        #     tsk = line.split(" ")[1]
                        #     assert tsk in ["classification", "regression"], "Task is not given."
                        #     classification = True if tsk == "classification" else False

                        if task == "classification":  # and if classification
                            if "mean(Classification_Accuracy)" in line:
                                accuracy = line.split(" ")[1]
                                # Fill in lists
                                acc_list.append(float(accuracy))
                                acc_name_list.append(file_name)

                        elif task == "regression":
                            if "mean(Accuracy)" in line:
                                accuracy = line.split(" ")[1]
                                # print(accuracy)
                                # Fill in lists
                                acc_list.append(float(accuracy))
                                acc_name_list.append(file_name)

        acc_list_sorted, acc_name_list_sorted = map(list, zip(*sorted(zip(acc_list, acc_name_list))[::-1]))

        print("\n")
        for i, j in zip(acc_list_sorted[:n], acc_name_list_sorted[:n]):
            print(i, "\t\t\t", j)

        for file_n in acc_name_list_sorted[:n]:
            for plot_file in os.listdir(wdic_plot):
                try:
                    identifier = file_n.split("folds_")[1].split(".")[0]
                except IndexError:
                    identifier = file_n.split("_S{}".format(sub))[0]

                if identifier in plot_file and "_all_train_val_" in plot_file:
                    current_plot_file = wdic_plot+plot_file
                    subprocess.Popen(["open", current_plot_file])  # 'open' only works for Mac

# open_best_params(subjects=[2, 36], rand_search=True, task="classification", n=5)
# open_best_params(subjects=[2, 36], rand_search=False, task="regression", n=5)


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

    # acc_col = -1 if task == "classification" else -3
    acc_col = -1
    if task == "regression":
        # Sort according to difference of mean validation accuracy and zeroline accuracy
        table[0, -1] = "meanval-zeroline_acc"
        for idx in range(1, table.shape[0]):
            table[idx, -1] = "{:.5}".format(str(float(table[idx, -3]) - float(table[idx, -2])))  # acc - zeroline_acc

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
# sort_table(task="classification")


def merge_randsearch_tables(task, sort=True):
    """
    Merges random search table from server with the one from local computer and saves them
    :param task: either 'classification' or 'regression'
    :param sort: sort table according to accuracy
    """

    # Check function input
    assert task.lower() in ["classification", "regression"], "task must be either 'classification' or 'regression'"

    tfix = "_BiCl" if task.lower() == "classification" else "_Reg"

    # Find tables
    wd_table = "./LSTM/"
    for file in os.listdir(wd_table):
        if "Random_Search_Table" in file and tfix in file and "_server.csv" not in file and ".csv" in file:
            local_table_filename = wd_table + file
        elif "Random_Search_Table" in file and tfix in file and "_server.csv" in file:
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
# merge_randsearch_tables(task="regression", sort=True)


# # Save random search tables per subject, extract n best settings per subject
def rnd_search_table_per_subject(table_name):

    wd_tables = "./LSTM/Random_Search_Tables/"

    assert ".csv" in table_name, "Must be a csv file and filename ending with '.csv'"
    assert os.path.exists(wd_tables+table_name), "File does not exist:\t{}".format(table_name)

    rs_table = np.genfromtxt(wd_tables+table_name, delimiter=";", dtype=str)
    subjects = np.unique([int(sub) for sub in rs_table[1:, 1]])  # subjects
    header = np.reshape(rs_table[0, :], newshape=(1, rs_table.shape[1]))

    for sub in subjects:
        sub_rs_table = rs_table[np.where(rs_table[:, 1] == str(sub))[0], :]  # extract sub-table
        sub_rs_table = np.concatenate((header, sub_rs_table), axis=0)  # attach header

        # Save
        export_filename = "S{}_Ran".format(str(sub).zfill(2)) + table_name.split("Ran")[1].split("_m")[0] + ".csv"
        np.savetxt(fname=wd_tables+export_filename, X=sub_rs_table, delimiter=";", fmt="%s")
# rnd_search_table_per_subject(table_name='Random_Search_Final_Table_BiCl_merged_sorted.csv')
# rnd_search_table_per_subject(table_name='Random_Search_Final_Table_Reg_merged_sorted.csv')
# rnd_search_table_per_subject(table_name='Random_Search_Table_Reg_merged_sorted.csv')
# rnd_search_table_per_subject(table_name='Random_Search_Final_Table_BiCl_SPOC_merged_sorted.csv')
# rnd_search_table_per_subject(table_name='Random_Search_Final_Table_Reg_SPOC_merged_sorted.csv')


def table_per_hp_setting(table_name):

    wd_tables = "./LSTM/Random_Search_Tables/"

    assert ".csv" in table_name, "Must be a csv file and filename ending with '.csv'"
    assert os.path.exists(wd_tables+table_name), "File does not exist:\t{}".format(table_name)

    rs_table = np.genfromtxt(wd_tables+table_name, delimiter=";", dtype=str)
    # subjects = np.unique([int(sub) for sub in rs_table[1:, 1]])  # subjects
    settings = np.unique([str(setting) for setting in rs_table[1:, 23]])  # settings
    header = np.reshape(rs_table[0, :], newshape=(1, rs_table.shape[1]))

    for setx, setting in enumerate(settings):
        set_rs_table = rs_table[np.where(rs_table[:, 23] == str(setting))[0], :]  # extract sub-table
        set_rs_table = np.concatenate((header, set_rs_table), axis=0)  # attach header

        # Save
        export_filename = "Set{}_Ran".format(str(setx+1).zfill(2)) + table_name.split("Ran")[1].split("_m")[0] + ".csv"
        np.savetxt(fname=wd_tables+export_filename, X=set_rs_table, delimiter=";", fmt="%s")
# table_per_hp_setting(table_name='Random_Search_Final_Table_BiCl_merged_sorted.csv')
# table_per_hp_setting(table_name='Random_Search_Final_Table_Reg_merged_sorted.csv')
# table_per_hp_setting(table_name='Random_Search_Final_Table_BiCl_SPOC_merged_sorted.csv')
# table_per_hp_setting(table_name='Random_Search_Final_Table_Reg_SPOC_merged_sorted.csv')


# first apply rnd_search_table_per_subject() then:
def table_of_best_hp_over_all_subjects(n, task):
    """
    Takes from each subject n best hyper parameter settings and writes in new table
    :param task: 'classification' or 'regression'
    :param n: number of hyper parameter settings to be saved in new table
    """
    assert task.lower() in ["classification", "regression"], "task must be either 'classification' or 'regression'"

    wd_tables = "./LSTM/Random_Search_Tables/"

    tsk = "BiCl" if task.lower() == "classification" else "Reg"
    # count subjects:
    cntr = 0  # == number of subjects
    for file_name in os.listdir(wd_tables):
        if file_name[0] == "S" and tsk in file_name:
            # Load random search table:
            rs_table = np.genfromtxt(wd_tables + file_name, delimiter=";", dtype=str)
            if cntr == 0:
                # init new table
                header = np.reshape(rs_table[0, :], newshape=(1, rs_table.shape[1]))
                bhp_table = copy.copy(header)
            cntr += 1

            bhp_table = np.concatenate((bhp_table, rs_table[1:n+1, :]))
            exp_filename = file_name

    if tsk == "BiCl":
        mc = np.mean([float(x) for x in bhp_table[1:, -1]])
        print("Average Accuracy:", mc)

    else:  # tsk == "Reg"
        mean_val_acc = np.array([float(x) for x in bhp_table[1:, -3]])
        zeroline_acc = np.array([float(x) for x in bhp_table[1:, -2]])
        mc = np.mean(mean_val_acc - zeroline_acc)
        print("Average Above-Zero-Accuracy:", mc)

    # Delete redundant entries
    bhp_table_unique = np.unique(bhp_table[1:, 2:-3], axis=0)
    bhp_table_unique = np.concatenate((header[:, 2:-3], bhp_table_unique))
    print("{} entries repeat each other. ".format(bhp_table.shape[0]-bhp_table_unique.shape[0]))

    # Save
    export_filename = "Best_{}_HPsets_over_{}_Subjects_mean_acc_{:.3f}_Ran".format(
        n, cntr, mc) + exp_filename.split("_Ran")[1]
    export_filename_unique = "unique_" + export_filename
    np.savetxt(fname=wd_tables + export_filename, X=bhp_table, delimiter=";", fmt="%s")
    np.savetxt(fname=wd_tables + export_filename_unique, X=bhp_table_unique, delimiter=";", fmt="%s")
# table_of_best_hp_over_all_subjects(n=2, task="classification")
# table_of_best_hp_over_all_subjects(n=2, task="regression")


def model_performance(over, task, input_type):
    """
    Calculate the overall performance over subjects or over hyperparameter sets
    :param task: which task: either 'classification' or 'regression'
    :param over: type=str, either 'subjects' or 'hp-sets'
    :param input_type: either "SSD" for SSD data, or "SPOC" for SPoC data
    :return: performance table
    """

    assert task.lower() in ['classification', 'regression'], "task must be either 'regression' or 'classification'"
    assert input_type.lower() in ['ssd', 'spoc'], "input_type must be either 'SSD' or 'SPOC'"
    assert over.lower() in ['subjects', 'hp-sets'], "over must be either 'subjects' or 'hp-sets'"

    wd_tables = "./LSTM/Random_Search_Tables/Random_Search_Final_Table{}{}/".format(
        "_Reg" if task == "regression" else "",
        "_SPOC" if input_type.lower() == "spoc" else "")
    wd_tables += "per_subject/" if over == "subjects" else "per_hp_set/"

    count_entries = 0

    if over == "subjects":

        head_idx = [1, 23, -1] if task == "classification" else [1, 23, 24, 25, 26]

        for file_name in os.listdir(wd_tables):
            if ".csv" in file_name:
                count_entries += 1
                rs_table = np.genfromtxt(wd_tables + file_name, delimiter=";", dtype=str)
                if count_entries == 1:
                    fin_table = np.reshape(rs_table[0, head_idx], newshape=(1, len(head_idx)))  # header

                perform = [rs_table[1, -1]] if task == "classification" else list(rs_table[1, -3:])  # float()
                setting = rs_table[1, 23]
                sub = file_name.split("_Random")[0].split("S")[1]  # subject number (str)

                fin_table = np.concatenate((fin_table, np.reshape([sub, setting] + perform, (1, len(fin_table[0])))))

        # performances = [float(x) if x != "nan" else np.nan for x in fin_table[1:, 2:]]
        # mean_perform = np.round(np.nanmean(performances), 3)
        performances = np.array(fin_table[1:, -1 if task == "classification" else -3:], dtype=float)
        mean_perform = np.round(np.nanmean(performances, axis=0), 3)
        fin_table = np.concatenate((fin_table, np.reshape(np.array(["all",
                                                                    "average_performance",
                                                                    ] + list(mean_perform.astype(str))),
                                                          newshape=(1, len(fin_table[0])))))

        # Save
        export_filename = "AllSub_Ran" + file_name.split("Ran")[1]
        np.savetxt(fname=wd_tables + export_filename, X=fin_table, delimiter=";", fmt="%s")

    else:  # over == "hp-sets"

        head_idx = [23, -1]  # here no differentiation between the tasks

        for file_name in os.listdir(wd_tables):
            if "Set" in file_name and ".csv" in file_name:
                count_entries += 1
                rs_table = np.genfromtxt(wd_tables + file_name, delimiter=";", dtype=str)
                if count_entries == 1:
                    fin_table = np.reshape(np.concatenate((["subjects"], rs_table[0, head_idx], ["SD"])),
                                           newshape=(1, len(head_idx)+2))  # header

                # performances = [float(x) if x != "nan" else np.nan for x in rs_table[1:, -1]]
                # mean_perform = np.round(np.nanmean(performances), 3)
                performances = np.array(rs_table[1:, -1], dtype=float)
                mean_perform = np.round(np.nanmean(performances, axis=0), 3)
                std_perform = np.round(np.nanstd(performances, axis=0), 3)
                setting = rs_table[1, 23]  # for all the same

                fin_table = np.concatenate((fin_table, np.reshape(np.array(["all", setting, mean_perform, std_perform]),
                                                                  newshape=(1, len(head_idx)+2))))

        # Sort w.r.t. mean_perform
        acc_col = 2  # mean_acc column,  fin_table[:, acc_col]
        sorted_fin_table = copy.copy(fin_table)
        sorted_fin_table[1:, :] = sorted_fin_table[np.argsort(sorted_fin_table[1:, acc_col]) + 1]
        if "nan" in sorted_fin_table[:, acc_col]:
            nan_start = np.where(sorted_fin_table[:, acc_col] == "nan")[0][0]
        else:
            nan_start = sorted_fin_table.shape[0]
        sorted_fin_table[1:nan_start, ] = sorted_fin_table[np.argsort(sorted_fin_table[1:nan_start, acc_col])+1, ][::-1]

        # Save
        export_filename = "AllHPsets_Ran" + file_name.split("Ran")[1]
        np.savetxt(fname=wd_tables + export_filename, X=sorted_fin_table, delimiter=";", fmt="%s")
# model_performance(over="subjects", task="classification", input_type="SSD")
# model_performance(over="hp-sets", task="classification", input_type="SSD")
# model_performance(over="subjects", task="regression", input_type="SSD")
# model_performance(over="hp-sets", task="regression", input_type="SSD")
# model_performance(over="subjects", task="classification", input_type="SPOC")
# model_performance(over="hp-sets", task="classification", input_type="SPOC")
# model_performance(over="subjects", task="regression", input_type="SPOC")
# model_performance(over="hp-sets", task="regression", input_type="SPOC")

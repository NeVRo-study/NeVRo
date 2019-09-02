# coding=utf-8
"""
After we run grid or random search on subjects,
we find now the best hyperparameters from the processed files.

Author: Simon Hofmann | <[surname].[lastname][at]pm.me> | 2017, 2019 (Update)
"""

import copy
from meta_functions import *

setwd("/Analysis/Modelling/LSTM/")


def open_best_params(subjects, task, n=5):
    """
    Print filenames and open plots of best hyperparameters per subjects
    :param subjects: list of subjects
    :param task: "classification" or "regression"
    :param n: number of settings to print and plot
    :return:
    """

    assert task in ["classification", "regression"], \
        "Task must be either 'classification' or 'regression'."

    if not isinstance(subjects, list):
        subjects = [subjects]

    for sub in subjects:
        wdic = f"./processed/{s(sub)}/already_plotted/"
        wdic_plot = "../../../Results/Plots/LSTM/" + task + "/"

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

                        if task == "classification":
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

        acc_list_sorted, acc_name_list_sorted = map(list, zip(*sorted(zip(acc_list,
                                                                          acc_name_list))[::-1]))

        print("\n")
        for i, j in zip(acc_list_sorted[:n], acc_name_list_sorted[:n]):
            print(i, "\t\t\t", j)

        for file_n in acc_name_list_sorted[:n]:
            try:
                identifier = file_n.split("folds_")[1][:-4]
            except IndexError:
                identifier = file_n.split(f"_S{sub}")[0]

            for plot_file in os.listdir(wdic_plot):
                if identifier in plot_file and "_all_train_val_" in plot_file:
                    current_plot_file = wdic_plot+plot_file

                    if os.sys.platform == 'darwin':  # for Mac
                        subprocess.Popen(["open", current_plot_file])
                    elif 'linux' in os.sys.platform:
                        subprocess.Popen(["display", current_plot_file])  # feh
                    else:
                        subprocess.Popen(["start", current_plot_file])

# open_best_params(subjects=[2, 36], task="classification", n=5)
# open_best_params(subjects=[14, 25], task="classification", n=5)


# Merge Random Search Tables from server and local computer:
def sort_table(task, table=None):
    save_externally = False
    if table is None:
        tfix = "_BiCl" if task.lower() == "classification" else "_Reg"
        task_merg_fix = f"{tfix}_merged.csv"

        file_found = False
        for file in os.listdir("./processed/"):
            if "Random_Search_Table" in file and task_merg_fix in file:
                table_filename = "./processed/" + file
                file_found = True
                save_externally = True
                break
            elif "Random_Search_Table" + tfix in file and task_merg_fix not in file:
                if ask_true_false(f"Does this file: '{file}' contain all processed results?", col="b"):
                    table_filename = "./processed/" + file
                    file_found = True
                    save_externally = True
                    break

        if file_found:
            table = np.genfromtxt(table_filename, delimiter=";", dtype=str)
        else:
            raise FileNotFoundError("No table given and no table found in './processed/'")

    # acc_col = -1 if task == "classification" else -3
    acc_col = -1
    if task == "regression":
        # Sort according to difference of mean validation accuracy and meanline accuracy
        table[0, -1] = "meanval-meanline_acc"
        for idx in range(1, table.shape[0]):
            table[idx, -1] = "{:.5}".format(
                str(float(table[idx, -3]) - float(table[idx, -2])))  # acc - meanline_acc

    sorted_table = copy.copy(table)
    sorted_table[1:, :] = sorted_table[np.argsort(sorted_table[1:, acc_col]) + 1]
    if "nan" in sorted_table[:, acc_col]:
        nan_start = np.where(sorted_table[:, acc_col] == "nan")[0][0]
    else:
        nan_start = sorted_table.shape[0]
    sorted_table[1:nan_start, ] = sorted_table[np.argsort(sorted_table[1:nan_start, acc_col]) + 1, ][::-1]

    if save_externally:
        np.savetxt(fname="." + table_filename.split(".")[1] + "_sorted.csv", X=sorted_table,
                   delimiter=";", fmt="%s")
    else:
        return sorted_table
# sort_table(task="classification")


def merge_randsearch_tables(task, sort=True):
    """
    Merges random search table from server with the one from local/other computer and saves them
    :param task: either 'classification' or 'regression'
    :param sort: sort table according to accuracy
    """

    # Check function input
    assert task.lower() in ["classification", "regression"], \
        "task must be either 'classification' or 'regression'"

    tfix = "_BiCl" if task.lower() == "classification" else "_Reg"
    merge = None  # init

    # Find tables
    wd_table = "./processed/"
    list_of_tables = []
    for file in os.listdir(wd_table):
        if "Random_Search_Table" + tfix in file:
            list_of_tables.append(file)

    if len(list_of_tables) <= 1:
        cprint("Not enough tables to merge.", 'r')
        if len(list_of_tables) == 1:
            sort = ask_true_false(f"Do you want to sort this one table '{list_of_tables[0]}'", 'b')
            if sort:
                prime_tablename = wd_table + list_of_tables[0]
                rs_table = np.genfromtxt(prime_tablename, delimiter=";", dtype=str)
            else:
                return
        else:  # == 0: no table(s)
            return

    else:  # len(list_of_tables) > 1:
        list_of_tables.sort()

        cprint("Following tables were found:", 'b')
        for tab in list_of_tables:
            print("\t", tab)

        if ask_true_false("Do you want to merge these tables:", col='b'):
            # Merge tables
            merge = True
            prime_tablename = wd_table + list_of_tables[0]
            rs_table = np.genfromtxt(prime_tablename, delimiter=";", dtype=str)

            # Load other tables
            tctn = 1
            tab_dict = {}  # init to save other tables
            for table_name in list_of_tables[1:]:
                tctn += 1
                pull_table = np.genfromtxt(wd_table+table_name, delimiter=";", dtype=str)
                # Test whether tables are of same structure
                if not np.all(pull_table[:, :-3] == rs_table[:, :-3]):
                    raise ValueError("Except of results, given tables must be same in order to merge them.")
                tab_dict.update({f"table_{tctn}": pull_table})

            # Write results from pull tables into primary table (rs_table)
            for row in range(rs_table.shape[0]):
                if rs_table[row, -3] == "nan":  # if there is no result/value in primary table...
                    val_found = False
                    for tkey in tab_dict:
                        if tab_dict[tkey][row, -3] != "nan":
                            if not val_found:
                                rs_table[row, -3:] = tab_dict[tkey][row, -3]  # ... fill information from another table
                                val_found = True
                            elif rs_table[row, -3:] != tab_dict[tkey][row, -3]:
                                cprint("A different result was found in yet another table.", 'y')
                                print("Previous result is:", rs_table[row, -3:])
                                print("Current pull-table result is:", tab_dict[tkey][row, -3])
                                if ask_true_false("Do you want to stop merging in order to inspect tables", 'y'):
                                    raise StopIteration("Merging was manually stopped.")
                                elif ask_true_false("Do you want to overwrite previous value", 'b'):
                                    rs_table[row, -3:] = tab_dict[tkey][row, -3]

    # Sort table
    if sort:
        rs_table = sort_table(table=rs_table, task=task)

    # Save file
    export_filename = prime_tablename.split(".csv")[0] + ("_merged" if merge else "") + ("_sorted" if sort else "")
    np.savetxt(fname=export_filename+".csv", X=rs_table, delimiter=";", fmt="%s")
# merge_randsearch_tables(task="classification", sort=True)
# merge_randsearch_tables(task="regression", sort=True)


# # Save random search tables per subject, extract n best settings per subject
def rnd_search_table_per_subject(table_name, condition):
    """This function splits given random search table into single sub-tables per subject"""

    assert "mov" in condition, "condition must be either 'mov' or 'nomov'"
    cond = "nomov" if "no" in condition.lower() else "mov"
    task = "classification" if "BiCl" in table_name else "regression"

    wd_tables = f"./processed/Random_Search_Tables/{cond}/"

    assert ".csv" in table_name, "Must be a csv file and filename ending with '.csv'"
    assert os.path.exists(wd_tables+table_name), f"File does not exist:\t{table_name}"

    if "merged" not in table_name:
        if not ask_true_false(f"Does the table '{table_name}' contain all (merged) information and "
                              f"do you want you to split it?", col='b'):
            cprint("Given table won't be split. Exit function.", 'r')
            return

    rs_table = np.genfromtxt(wd_tables+table_name, delimiter=";", dtype=str)
    subjects = np.unique([int(sub) for sub in rs_table[1:, 1]])  # subjects
    header = np.reshape(rs_table[0, :], newshape=(1, rs_table.shape[1]))

    if "sorted" not in table_name:
        if ask_true_false(f"Table must be sorted before splitting. Do you want to continue?", col='b'):
            rs_table = sort_table(task=task, table=rs_table)
            table_name = table_name.split(".csv")[0] + "_sorted.csv"
            np.savetxt(fname=wd_tables+table_name, X=rs_table, delimiter=";", fmt="%s")
            print(f"Sorted version of table is saved under: '{wd_tables+table_name}'")
        else:
            cprint("Given table won't be split. Exit function.", 'r')
            return

    for sub in subjects:
        sub_rs_table = rs_table[np.where(rs_table[:, 1] == str(sub))[0], :]  # extract sub-table
        sub_rs_table = np.concatenate((header, sub_rs_table), axis=0)  # attach header

        # Save
        export_filename = f"{s(sub)}_Ran"
        export_filename += table_name.split("Ran")[1].split("_m" if "merged" in table_name else "_s")[0] + ".csv"
        np.savetxt(fname=wd_tables+export_filename, X=sub_rs_table, delimiter=";", fmt="%s")
# rnd_search_table_per_subject(table_name='Random_Search_Final_Table_BiCl_merged_sorted.csv', condition="nomov")
# rnd_search_table_per_subject(table_name='Random_Search_Final_Table_Reg_merged_sorted.csv', condition="nomov")
# rnd_search_table_per_subject(table_name='Random_Search_Table_Reg_merged_sorted.csv', condition="nomov")
# rnd_search_table_per_subject(table_name='Random_Search_Final_Table_BiCl_SPOC_merged_sorted.csv', condition="nomov")
# rnd_search_table_per_subject(table_name='Random_Search_Final_Table_Reg_SPOC_merged_sorted.csv', condition="nomov")


def table_per_hp_setting(table_name, condition, fixed_comps=False):
    """This function splits given random search table into single subtables per set of hyper-parameters
    :param table_name: name of parent table
    :param condition: "mov" or "nomov"
    :param fixed_comps: True: selection of components is identical for each subject per hyper-parameter set;
                        False: individual comps per subject, selected under same rules ('random' OR 'one-up') and same N
    """
    assert "mov" in condition, "condition must be either 'mov' or 'nomov'"
    cond = "nomov" if "no" in condition.lower() else "mov"
    task = "classification" if "BiCl" in table_name else "regression"

    wd_tables = f"./processed/Random_Search_Tables/{cond}/"

    assert ".csv" in table_name, "Must be a csv file and filename ending with '.csv'"
    assert os.path.exists(wd_tables+table_name), f"File does not exist:\t{table_name}"
    # Prepare folder to save files
    sav_dir = wd_tables + task + "/per_hp_set/"
    if not os.path.exists(sav_dir):
        os.makedirs(sav_dir)

    # Load table
    rs_table = np.genfromtxt(wd_tables+table_name, delimiter=";", dtype=str)
    # subjects = np.unique([int(sub) for sub in rs_table[1:, 1]])  # subjects
    hp_set_col = np.where(rs_table[0, :] == 'path_specificities')[0][0]
    hp_settings = np.unique([str(setting) for setting in rs_table[1:, hp_set_col]])  # settings

    header = np.reshape(rs_table[0, :], newshape=(1, rs_table.shape[1]))

    if fixed_comps:
        for setx, setting in enumerate(hp_settings):

            set_rs_table = rs_table[np.where(rs_table[:, hp_set_col] == setting)[0], :]  # extract sub-table
            set_rs_table = np.concatenate((header, set_rs_table), axis=0)  # attach header

            # Save
            sp = '_merged' if "_merged" in table_name else '_sort' if '_sorted' in table_name else ".csv"  # splitter
            export_filename = f"Set{str(setx+1).zfill(2)}_Ran{table_name.split('Ran')[1].split(sp)[0]}.csv"
            np.savetxt(fname=sav_dir+export_filename, X=set_rs_table, delimiter=";", fmt="%s")

    else:
        # Remove individual component information (per subject) from list of unique hyper-parameter settings
        hp_settings_without_comps = np.unique(
            [setting.split("_comp")[0]+"_hrcomp"+setting.split("_hrcomp")[1] for setting in hp_settings])

        for setx, setting in enumerate(hp_settings_without_comps):
            sel_rows = []
            for ridx, full_setting in enumerate(rs_table[:, hp_set_col]):
                if setting.split("_hrc")[0] in full_setting and setting.split("_hrc")[1] in full_setting:
                    sel_rows.append(ridx)
            set_rs_table = rs_table[sel_rows, :]  # extract sub-table
            set_rs_table = np.concatenate((header, set_rs_table), axis=0)  # attach header

            # Save
            sp = '_merged' if "_merged" in table_name else '_sort' if '_sorted' in table_name else ".csv"  # splitter
            export_filename = f"Set{str(setx + 1).zfill(2)}_Ran{table_name.split('Ran')[1].split(sp)[0]}.csv"
            np.savetxt(fname=sav_dir + export_filename, X=set_rs_table, delimiter=";", fmt="%s")
# table_per_hp_setting(table_name='Random_Search_Final_Table_BiCl_merged_sorted.csv', condition="nomov")
# table_per_hp_setting(table_name='Random_Search_Final_Table_Reg_merged_sorted.csv', condition="nomov")
# table_per_hp_setting(table_name='Random_Search_Final_Table_BiCl_SPOC_merged_sorted.csv', condition="nomov")
# table_per_hp_setting(table_name='Random_Search_Final_Table_Reg_SPOC_merged_sorted.csv', condition="nomov")


# TODO continue here
# first apply rnd_search_table_per_subject() then:
def table_of_best_hp_over_all_subjects(n, task):
    """
    Takes from each subject n best hyper parameter settings and writes in new table
    :param task: 'classification' or 'regression'
    :param n: number of hyper parameter settings to be saved in new table
    """
    assert task.lower() in ["classification", "regression"], "task must be either 'classification' or 'regression'"

    wd_tables = "./processed/Random_Search_Tables/"

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
        meanline_acc = np.array([float(x) for x in bhp_table[1:, -2]])
        mc = np.mean(mean_val_acc - meanline_acc)
        print("Average Above-Meanline-Accuracy:", mc)

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

    assert task.lower() in ['classification', 'regression'], \
        "task must be either 'regression' or 'classification'"
    assert input_type.lower() in ['ssd', 'spoc'], "input_type must be either 'SSD' or 'SPOC'"
    assert over.lower() in ['subjects', 'hp-sets'], "over must be either 'subjects' or 'hp-sets'"

    wd_tables = "./processed/Random_Search_Tables/Random_Search_Final_Table{}{}/".format(
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

                perform = [rs_table[1, -1]] if task == "classification" else list(rs_table[1, -3:])
                setting = rs_table[1, 23]
                sub = file_name.split("_Random")[0].split("S")[1]  # subject number (str)

                fin_table = np.concatenate((fin_table, np.reshape([sub, setting] + perform,
                                                                  (1, len(fin_table[0])))))

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

                fin_table = np.concatenate((fin_table, np.reshape(np.array(["all", setting, mean_perform,
                                                                            std_perform]),
                                                                  newshape=(1, len(head_idx)+2))))

        # Sort w.r.t. mean_perform
        acc_col = 2  # mean_acc column,  fin_table[:, acc_col]
        sorted_fin_table = copy.copy(fin_table)
        sorted_fin_table[1:, :] = sorted_fin_table[np.argsort(sorted_fin_table[1:, acc_col]) + 1]
        if "nan" in sorted_fin_table[:, acc_col]:
            nan_start = np.where(sorted_fin_table[:, acc_col] == "nan")[0][0]
        else:
            nan_start = sorted_fin_table.shape[0]
        sorted_fin_table[1:nan_start, ] = sorted_fin_table[np.argsort(sorted_fin_table[1:nan_start,
                                                                      acc_col])+1, ][::-1]

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

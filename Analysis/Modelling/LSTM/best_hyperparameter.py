# coding=utf-8
"""
After we run grid or random search on subjects,
we find now the best hyperparameters from the processed files.

Author: Simon Hofmann | <[surname].[lastname][at]pm.me> | 2017, 2019 (Update)
"""

import copy
import ast
from meta_functions import *

# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

setwd("/Analysis/Modelling/LSTM/")


# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

def open_best_params(subjects, task, condition, n=5):
    """
    Print filenames and open plots of best hyperparameters per subjects
    :param subjects: list of subjects
    :param task: "classification" or "regression"
    :param condition: "mov" or "nomov"
    :param n: number of settings to print and plot
    :return:
    """

    task = task.lower()
    assert task in ["classification", "regression"], \
        "Task must be either 'classification' or 'regression'."
    condition = condition.lower()
    assert "mov" in condition, "condition must be either 'mov' or 'nomov'"
    cond = "nomov" if "no" in condition else "mov"

    if not isinstance(subjects, list):
        subjects = [subjects]

    for sub in subjects:
        wdic = f"./processed/{cond}/{s(sub)}/already_plotted/"
        wdic_plot = f"../../../Results/Plots/LSTM/{cond}/{task}/"

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
            identifier = file_n.split("folds_")[1][:-4]
            # try:
            #     identifier = file_n.split("folds_")[1][:-4]
            # except IndexError:
            #     identifier = file_n.split(f"_S{sub}")[0]

            for plot_file in os.listdir(wdic_plot):
                if identifier in plot_file and "_all_train_val_" in plot_file:
                    current_plot_file = wdic_plot+plot_file

                    if os.sys.platform == 'darwin':  # for Mac
                        subprocess.Popen(["open", current_plot_file])
                    elif 'linux' in os.sys.platform:
                        subprocess.Popen(["display", current_plot_file])  # feh
                    else:
                        subprocess.Popen(["start", current_plot_file])
# open_best_params(subjects=[9, 31], task="classification", condition="nomov", n=3)


# Merge Random Search Tables from server and local computer:
def sort_table(task, table=None, table_path=None):
    """
    Sort given table, given by python object or via external file.
    If table comes via an external (csv)-file, the sorted version will be saved next to it.
    :param task: 'classification' OR 'regression'
    :param table: numpy 2D-array
    :param table_path: if no table: then load table from provided path.
                       Assumes table to be in subfolder of '/processed/Random_Search_Tables/'
    :return: sorted table (if via 'table' arg) otherwise just saved externally.
    """
    save_externally = False
    if table is None:

        assert table_path, "path to table file (.csv) must be given. " \
                           "This assumes that table lies in subfolder of '/processed/Random_Search_Tables/'"

        table_path = "./processed/Random_Search_Tables/" + table_path

        if os.path.isfile(table_path):
            save_externally = True
            table = np.genfromtxt(table_path, delimiter=";", dtype=str)
        else:
            raise FileNotFoundError(f"No table given and no table found in './processed/Random_Search_Tables/'")

    # Sort table
    acc_col = -1  # acc_col = -1 if task == "classification" else -3
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
        np.savetxt(fname="." + table_path.split(".")[1] + "_sorted.csv", X=sorted_table, delimiter=";", fmt="%s")
    else:
        return sorted_table
# sort_table(task="classification", table_path="nomov/0_broad_search/classification/Random_Search_Table_BiCl.csv")


def merge_randsearch_tables(task, condition, search, sort=True):
    """
    Merges random search table from server with the one from local/other computer and saves them
    :param task: either 'classification' or 'regression'
    :param condition: either 'mov' or 'nomov'
    :param search: either 'broad' or 'narrow'
    :param sort: sort table according to accuracy
    """

    # Check function input
    task = task.lower()
    assert task in ["classification", "regression"], "task must be either 'classification' or 'regression'"
    condition = condition.lower()
    assert "mov" in condition, "condition must be either 'mov' or 'nomov'"
    cond = "nomov" if "no" in condition else "mov"
    search = search.lower()
    assert search in ['broad', 'narrow'], "search must be either 'broad' or 'narrow'"

    tfix = "_BiCl" if task == "classification" else "_Reg"
    merge = None  # init

    # Find tables
    wd_table = f"./processed/Random_Search_Tables/{cond}/{0 if search == 'broad' else 1}_{search}_search/{task}/"
    if not os.path.exists(wd_table):
        cprint(f"Assumes table to lie in '{wd_table}'.\nThis path wasn't found!", 'r')

    list_of_tables = []
    for file in os.listdir(wd_table):
        if f"{'Random' if search == 'broad' else search.title()}_Search_Table_{cond}{tfix}" in file:
            list_of_tables.append(file)

    if len(list_of_tables) <= 1:
        cprint("Not enough tables to merge.", 'r')
        if len(list_of_tables) == 1:
            if "sorted" in list_of_tables[0]:
                sort = False
            else:
                sort = ask_true_false(f"Do you want to sort this one table '{wd_table+list_of_tables[0]}'?", 'b')
            if sort:
                prime_tablename = wd_table + list_of_tables[0]
                rs_table = np.genfromtxt(prime_tablename, delimiter=";", dtype=str)
            else:
                return  # None
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

    return export_filename.split(f"{task}/")[1] + ".csv"  # e.g. Random_Search_Table_Reg__merged_sorted.csv
# merge_randsearch_tables(task="classification", condition="nomov", search="broad", sort=True)
# merge_randsearch_tables(task="regression", condition="nomov", search="broad", sort=True)


# # Save random search tables per subject, extract n best settings per subject
def table_per_subject(table_name, condition, search):
    """
    This function splits given random search table into single sub-tables per subject.
    :param table_name: full table name + .csv, must lie in wd_tables (see below)
    :param condition: "mov" OR "nomov"
    :param search: 'broad' OR 'narrow'
    """

    condition = condition.lower()
    assert "mov" in condition, "condition must be either 'mov' or 'nomov'"
    cond = "nomov" if "no" in condition else "mov"
    search = search.lower()
    assert search in ['broad', 'narrow'], "search must be either 'broad' or 'narrow'"

    task = "classification" if "BiCl" in table_name else "regression"

    wd_tables = f"./processed/Random_Search_Tables/{cond}/{0 if search == 'broad' else 1}_{search}_search/{task}/"

    assert ".csv" in table_name, "Must be a csv file and filename ending with '.csv'"
    assert os.path.exists(wd_tables+table_name), f"File does not exist:\t{wd_tables+table_name}"

    # Prepare folder to save files
    sav_dir = wd_tables + "per_subject/"
    if not os.path.exists(sav_dir):
        os.makedirs(sav_dir)

    if "merged" not in table_name:
        if not ask_true_false(f"Does the table '{table_name}' contain all (merged) information and "
                              f"do you want you to split it?", col='b'):
            cprint("Given table won't be split. Exit function.", 'r')
            return

    rs_table = np.genfromtxt(wd_tables+table_name, delimiter=";", dtype=str)
    assert np.all(rs_table[1:, np.where(rs_table[0, :] == "cond")[0][0]] == cond), \
        f"Table contains other conditions than '{cond}'"

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
        tpfix = "Ran" if search == "broad" else "Nar"  # broad search: 'Ran' OR narrow search: Nar
        export_filename = f"{s(sub)}_{tpfix}"
        export_filename += table_name.split(tpfix)[1].split("_m" if "merged" in table_name else "_s")[0] + ".csv"
        np.savetxt(fname=sav_dir+export_filename, X=sub_rs_table, delimiter=";", fmt="%s")
# table_per_subject(table_name='Random_Search_Table_Reg_merged_sorted.csv', condition="nomov")
# table_per_subject(table_name='Random_Search_Final_Table_BiCl_merged_sorted.csv', condition="nomov")
# table_per_subject(table_name='Random_Search_Table_BiCl.csv', condition="nomov", search="broad")
# table_per_subject(table_name='Random_Search_Table_Reg.csv', condition="nomov", search="broad")


def table_per_hp_setting(table_name, condition, search, fixed_comps=False):
    """
    This function splits given random search table into single subtables per set of hyper-parameters
    :param table_name: name of parent table
    :param condition: "mov" or "nomov"
    :param search: 'broad' OR 'narrow'
    :param fixed_comps: True: selection of components is identical for each subject per hyper-parameter set;
                        False: individual comps per subject, selected under same rules ('random' OR 'one-up') and same N
    """
    condition = condition.lower()
    assert "mov" in condition, "condition must be either 'mov' or 'nomov'"
    cond = "nomov" if "no" in condition else "mov"
    search = search.lower()
    assert search in ['broad', 'narrow'], "search must be either 'broad' or 'narrow'"

    task = "classification" if "BiCl" in table_name else "regression"

    wd_tables = f"./processed/Random_Search_Tables/{cond}/{0 if search == 'broad' else 1}_{search}_search/{task}/"

    assert ".csv" in table_name, "Must be a csv file and filename ending with '.csv'"
    assert os.path.exists(wd_tables+table_name), f"File does not exist:\t{wd_tables+table_name}"
    # Prepare folder to save files
    sav_dir = wd_tables + "per_hp_set/"
    if not os.path.exists(sav_dir):
        os.makedirs(sav_dir)

    # Load table
    rs_table = np.genfromtxt(wd_tables+table_name, delimiter=";", dtype=str)

    assert np.all(rs_table[1:, np.where(rs_table[0, :] == "cond")[0][0]] == cond), \
        f"Table contains other conditions than '{cond}'"

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
            [setting.split("_comp")[0]+"_hrcomp" + setting.split("_hrcomp")[1] for setting in hp_settings])

        for setx, setting in enumerate(hp_settings_without_comps):
            sel_rows = []
            for ridx, full_setting in enumerate(rs_table[:, hp_set_col]):
                if setting.split("_hrc")[0] in full_setting and setting.split("_hrc")[1] in full_setting:
                    sel_rows.append(ridx)
            set_rs_table = rs_table[sel_rows, :]  # extract sub-table
            set_rs_table = np.concatenate((header, set_rs_table), axis=0)  # attach header

            # Save
            tpfix = "Ran" if search == "broad" else "Nar"  # broad search: 'Ran' OR narrow search: Nar
            sp = '_merged' if "_merged" in table_name else '_sort' if '_sorted' in table_name else ".csv"  # splitter
            export_filename = f"Set{str(setx+1).zfill(2)}_{tpfix}{table_name.split(tpfix)[1].split(sp)[0]}.csv"
            np.savetxt(fname=sav_dir + export_filename, X=set_rs_table, delimiter=";", fmt="%s")
# table_per_hp_setting(table_name='Random_Search_Final_Table_BiCl_merged_sorted.csv', condition="nomov")
# table_per_hp_setting(table_name='Random_Search_Table_BiCl.csv', condition="nomov", search="broad", fixed_comps=False)
# table_per_hp_setting(table_name='Random_Search_Table_Reg.csv', condition="nomov", search="broad", fixed_comps=False)


# first apply table_per_subject() then:
def table_of_best_hp_over_all_subjects(n, task, condition, search, fixed_comps=False):
    """
    Takes from each subject n best hyper parameter settings and writes in new table
    :param n: number of hyper parameter settings to be saved in new table
    :param task: 'classification' or 'regression'
    :param condition: 'mov' OR 'nomov'
    :param search: 'broad' OR 'narrow'
    :param fixed_comps: True: selection of components is identical for each subject per hyper-parameter set;
                        False: individual comps per subject, selected under same rules ('random' OR 'one-up') and same N
    """
    # Check given arguments
    task = task.lower()
    assert task in ["classification", "regression"], "task must be either 'classification' or 'regression'"
    tfix = "BiCl" if task == "classification" else "Reg"

    condition = condition.lower()
    assert "mov" in condition, "condition must be either 'mov' or 'nomov'"
    cond = "nomov" if "no" in condition else "mov"

    # Set path
    wd_sub_tables = f"./processed/Random_Search_Tables/{cond}/{0 if search == 'broad' else 1}_{search}_search/" \
                    f"{task}/per_subject/"

    # count subjects:
    cntr = 0  # == number of subjects
    for file_name in os.listdir(wd_sub_tables):
        if file_name[0] == "S" and tfix in file_name:
            # Load random search table:
            rs_table = np.genfromtxt(wd_sub_tables+file_name, delimiter=";", dtype=str)
            if cntr == 0:
                # init new table
                header = np.reshape(rs_table[0, :], newshape=(1, rs_table.shape[1]))
                bhp_table = copy.copy(header)
                exp_filename = file_name  # save file_name for later
            cntr += 1

            # Create one long table
            bhp_table = np.concatenate((bhp_table, rs_table[1:n+1, :]))

    if task == "classification":
        mc = np.mean([float(x) for x in bhp_table[1:, -1]])
        print("Average Accuracy:", mc)

    else:  # task == "regression"
        mean_val_acc = np.array([float(x) for x in bhp_table[1:, -3]])
        meanline_acc = np.array([float(x) for x in bhp_table[1:, -2]])
        mc = np.mean(mean_val_acc - meanline_acc)
        print(f"Average Above-Meanline-Accuracy: {mc:.3f}")

    # Delete redundant entries, i.e. 2 or more subjects share n-m best hyperparameter sets, where m in [0,n]
    if fixed_comps:
        bhp_table_unique = np.unique(bhp_table[:, 2:-3], axis=0)  # [cond ... path_specificities]
        print("Number of unique hyperparameter sets:", bhp_table_unique.shape[0])

    else:
        bhp_table_unique = copy.copy(bhp_table[:, 2:])  # init

        idx_path = np.where(bhp_table_unique[0, :] == "path_specificities")[0][0]
        idx_comp = np.where(bhp_table_unique[0, :] == "component")[0][0]
        bhp_table_unique = bhp_table_unique[:, :idx_path+1]

        # Create index for table without component and path_specificities
        idx_wo_comp_path = np.arange(bhp_table_unique.shape[1])
        idx_wo_comp_path = np.delete(idx_wo_comp_path, [idx_comp, idx_path])

        # Create table of unique entries ignoring the component information
        bhp_table_unique_wo_comps = copy.copy(bhp_table_unique)
        bhp_table_unique_wo_comps[1:,  [idx_comp, idx_path]] = None
        bhp_table_unique_wo_comps = np.unique(bhp_table_unique_wo_comps, axis=0)
        # bhp_table_unique_wo_comps = np.unique(bhp_table_unique[:, idx_wo_comp_path], axis=0)  # alternative

        # Following is the table of unique entries including the component information
        bhp_table_unique = np.unique(bhp_table_unique, axis=0)

        # Fill the max number of components in table (for both columns: 'component', 'path_specificities')
        for ridx, row in enumerate(bhp_table_unique_wo_comps[1:, idx_wo_comp_path]):
            n_max_comp = 0  # init
            row_path = False  # init
            for row_full in bhp_table_unique[1:, :]:
                if np.all(row == row_full[idx_wo_comp_path]):  # check if rows match (without component-information)
                    # Find max N components
                    cmps = ast.literal_eval(row_full[idx_comp])
                    n_cmps = 1 if isinstance(cmps, int) else len(cmps)
                    n_max_comp = n_cmps if n_cmps > n_max_comp else n_max_comp

                    # Extract 'path_specificities'
                    row_path = row_full[idx_path] if not row_path else row_path   # Do only once per row

            # Write max N in table per row
            bhp_table_unique_wo_comps[ridx+1, idx_comp] = f"n_max={n_max_comp}"

            # Write modified path in table (component information exchanged with max N comp) per row
            row_path = row_path.split("_comp")[0] + f"_comp-nmax-{n_max_comp}_hrc" + row_path.split("_hrc")[1]
            bhp_table_unique_wo_comps[ridx+1, idx_path] = row_path

        # Contains now unique HP-sets and summary over components.
        bhp_table_unique = bhp_table_unique_wo_comps
        # (Note: paths need to be modified when read out of this table)

    print(f"Number of unique hyperparameter sets: {bhp_table_unique.shape[0]} (out of {bhp_table.shape[0]})")

    # Save
    tpfix = "Ran" if search == "broad" else "Nar"  # broad search: 'Ran' OR narrow search: Nar
    export_filename = f"Best_{n}_HPsets_over_{cntr}_Subjects_mean_acc_{mc:.3f}_{tpfix}" + exp_filename.split(tpfix)[1]
    export_filename_unique = "unique_" + export_filename
    np.savetxt(fname=wd_sub_tables+export_filename, X=bhp_table, delimiter=";", fmt="%s")
    np.savetxt(fname=wd_sub_tables+export_filename_unique, X=bhp_table_unique, delimiter=";", fmt="%s")
# table_of_best_hp_over_all_subjects(n=2, task="classification", condition="nomov", search="broad")


# TODO continue here
def model_performance(over, task, condition, search, input_type):
    """
    Calculate the overall performance over subjects or over hyperparameter sets
    :param over: type=str, either 'subjects' or 'hpsets'
    :param task: which task: either 'classification' or 'regression'
    :param condition: "mov" OR "nomov"
    :param search: 'broad' OR 'narrow'
    :param input_type: either "SSD" OR "SPOC" for corresponding data type
    :return: performance table
    """

    over = over.lower()
    assert over in ['subjects', 'hpsets'], "over must be either 'subjects' or 'hpsets'"
    task = task.lower()
    assert task in ['classification', 'regression'], "task must be either 'regression' or 'classification'"
    condition = condition.lower()
    assert "mov" in condition, "condition must be either 'mov' or 'nomov'"
    cond = "nomov" if "no" in condition else "mov"
    search = search.lower()
    assert search in ['broad', 'narrow'], "search must be either 'broad' or 'narrow'"
    assert input_type.upper() in ['SSD', 'SPOC'], "input_type must be either 'SSD' or 'SPOC'"

    wd_tables = f"./processed/Random_Search_Tables/{cond}/{0 if search == 'broad' else 1}_{search}_search/{task}/"

    wd_tables += "per_subject/" if over == "subjects" else "per_hp_set/"
    tpfix = "Ran" if search == "broad" else "Nar"  # broad search: 'Ran' OR narrow search: Nar

    count_entries = 0

    if over == "subjects":

        for file_name in os.listdir(wd_tables):
            if file_name[0] == "S" and ".csv" in file_name:
                count_entries += 1
                rs_table = np.genfromtxt(wd_tables + file_name, delimiter=";", dtype=str)
                if count_entries == 1:

                    # Reduce final table to subject:path:performance
                    idx_sub = np.where(rs_table[0, :] == "subject")[0][0]  # 1
                    idx_path = np.where(rs_table[0, :] == "path_specificities")[0][0]  # 26
                    head_idx = [idx_sub, idx_path]
                    head_idx += [-1] if task == "classification" else [-3, -2, -1]  # [1, 23,( 24, 25,) 26] : accuracies

                    fin_table = np.reshape(rs_table[0, head_idx], newshape=(1, len(head_idx)))  # header

                # Extract best performance per subject, which is in the first row (tables are sorted)
                perform = [rs_table[1, -1]] if task == "classification" else list(rs_table[1, -3:])
                hp_setting = rs_table[1, idx_path]
                # subject number (str):
                sub = file_name.split(
                    f"_{'Random' if search == 'broad' else search.title()}")[0].split("S")[1]

                fin_table = np.concatenate((fin_table, np.reshape([sub, hp_setting] + perform,
                                                                  (1, len(fin_table[0])))))

        # performances = [float(x) if x != "nan" else np.nan for x in fin_table[1:, 2:]]
        # mean_perform = np.round(np.nanmean(performances), 3)
        performances = np.array(fin_table[1:, -1 if task == "classification" else -3:], dtype=float)
        mean_perform = np.round(np.nanmean(performances, axis=0), 3)
        fin_table = np.concatenate((fin_table,
                                    np.reshape(np.array(["all",
                                                         "average_performance"] + list(mean_perform.astype(str))),
                                               newshape=(1, len(fin_table[0])))))

        # Save
        export_filename = f"AllSub_{tpfix}" + file_name.split(tpfix)[1]
        np.savetxt(fname=wd_tables + export_filename, X=fin_table, delimiter=";", fmt="%s")

    else:  # over == "hpsets"

        for file_name in os.listdir(wd_tables):
            if "Set" in file_name and ".csv" in file_name:
                count_entries += 1
                rs_table = np.genfromtxt(wd_tables + file_name, delimiter=";", dtype=str)
                if count_entries == 1:
                    # Reduce final table to subject:path:performance
                    idx_path = np.where(rs_table[0, :] == "path_specificities")[0][0]  # 26
                    head_idx = [idx_path, -1]  # here no differentiation between the tasks
                    fin_table = np.reshape(np.concatenate((["subjects"], rs_table[0, head_idx], ["SD"])),  # add SD col
                                           newshape=(1, len(head_idx)+2))  # header

                # performances = [float(x) if x != "nan" else np.nan for x in rs_table[1:, -1]]
                # mean_perform = np.round(np.nanmean(performances), 3)
                performances = np.array(rs_table[1:, -1], dtype=float)
                mean_perform = np.round(np.nanmean(performances, axis=0), 3)
                std_perform = np.round(np.nanstd(performances, axis=0), 3)
                hp_setting = rs_table[1, idx_path]  # for all the same

                fin_table = np.concatenate((fin_table, np.reshape(np.array(["all", hp_setting, mean_perform,
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
        export_filename = f"AllHPsets_{tpfix}" + file_name.split(tpfix)[1]
        np.savetxt(fname=wd_tables + export_filename, X=sorted_fin_table, delimiter=";", fmt="%s")
# model_performance(over="subjects", task="classification", input_type="SSD")
# model_performance(over="hpsets", task="regression", input_type="SPOC")
# model_performance(over="subjects", task="classification", condition="nomov", search="broad", input_type="SSD")
# model_performance(over="hpsets", task="classification", condition="nomov", search="broad", input_type="SSD")
# model_performance(over="subjects", task="regression", condition="nomov", search="broad", input_type="SSD")
# model_performance(over="hpsets", task="regression", condition="nomov", search="broad", input_type="SSD")


# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

if __name__ == "__main__":

    # # First run broad random search then narrow search, i.e. run on set of best hyperparameters
    searches = [["broad"], ["narrow"], ["broad", "narrow"]]
    conditions = [["mov"], ["nomov"], ["mov", "nomov"]]
    tasks = [["classification"], ["regression"], ["classification", "regression"]]
    src = cinput("\nExtract information of [0] broad search, [1] narrow search, or [2] both?\nType 0, 1 or 2:", 'y')
    scr = int(src)
    cnd = cinput("\nApplied on condition [0] 'mov', [1] 'nomov', or [2] both?\nType 0, 1 or 2:", 'y')
    cnd = int(cnd)
    tsk = cinput("\nApplied on task [0] 'classification', [1] 'regression', or [2] both?\nType 0, 1 or 2:", 'y')
    tsk = int(tsk)
    fxcmp = ask_true_false("\nAre the components in one hyperparameter set fixed?", 'y')
    d_type = "SSD" if ask_true_false("\nIs 'SSD' the data type? False automatically assumes 'SPoC'", 'y') else 'SPOC'

    for searchi in searches[scr]:
        for taski in tasks[tsk]:
            for condi in conditions[cnd]:

                cprint(f"\nExtract results from {searchi} random search in {condi} condition ({taski}) ...", 'b')

                # For both conditions: 'mov' and 'nomov' AND for both tasks: 'classification' and 'regression' do:
                # 1) Merge tables (if necessary)
                tab_name = merge_randsearch_tables(task=taski, condition=condi, search=searchi, sort=True)

                # 2) Create table per subjects and per hp-setting
                table_per_subject(table_name=tab_name, condition=condi, search=searchi)
                table_per_hp_setting(table_name=tab_name, condition=condi, search=searchi, fixed_comps=fxcmp)

                # 3) Then create table of best HPs over all subjects
                table_of_best_hp_over_all_subjects(n=2, task=taski, condition=condi, search=searchi, fixed_comps=fxcmp)

                # 4) Calculate model performance
                # (Has to be done after "narrow"-search. Can be done on 'broad'-random search)
                model_performance(over="subjects", task=taski, condition=condi, search=searchi, input_type=d_type)
                model_performance(over="hpsets", task=taski, condition=condi, search=searchi, input_type=d_type)

end()
# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<  END

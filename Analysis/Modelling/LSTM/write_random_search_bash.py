# coding=utf-8
"""
Write random search bashfiles.
1) Do broad search: run over small subset of 10 random subjects
2) apply best working hyperparameter sets, i.e. 2 best sets per subject of broad search, on all subjects

Author: Simon Hofmann | <[surname].[lastname][at]pm.me> | 2017, 2019 (Update)
"""

from load_data import *
import sys
import fileinput

# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>
# Set paths to bashfile dir
p2_bash = './bashfiles/'

# If no bashfile dir: create it
if not os.path.exists(p2_bash):
    os.mkdir(p2_bash)


# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

def draw_components(n, subject, condition, filetype, select_mode="one_up", dtype=None):
    """

    :param n: number of components
    :param subject: subject number
    :param condition: 'mov' OR 'nomov'
    :param filetype: 'SSD' OR 'SPOC'
    :param select_mode: 'one-up' OR 'random_set'
    :param dtype: None: return np.array; 'str': as string
    :return: list of components per subject
    """

    assert select_mode in ["one_up", "random_set"], "component_mode must be either 'one_up' or 'random_set'"
    filetype = filetype.upper()
    assert filetype in ['SSD', 'SPOC'], "filetype must be either 'SSD' or 'SPOC'"
    cond = condition.lower()
    assert cond in ['mov', 'nomov'], "condition must be either 'mov' or 'nomov'"

    # Number of components to select
    sub_max_n = get_num_components(subject, cond, filetype)
    n_comp = n if n <= sub_max_n else sub_max_n

    # Get component list for subject
    sub_ls_pos_comps = get_list_components(subject, cond, filetype)

    # Choose components
    if select_mode == "one_up":
        sub_components = sub_ls_pos_comps[0:n]  # works also for n_comp < n

    else:  # select_mode == 'random_set'
        sub_components = np.sort(np.random.choice(a=sub_ls_pos_comps, size=n_comp, replace=False))

    # Adapt dytpe of output
    if isinstance(dtype, str):
        sub_components = ','.join([str(i) for i in sub_components])

    return sub_components


def write_search_bash_files(subs, filetype, condition,
                            task_request=None, component_mode=2, eqcompmat=None,
                            seed=False, repet_scalar=30, s_fold=10, balanced_cv=True,
                            batch_size=9, successive_mode=1, rand_batch=True, plot=True,
                            successive_default=3, del_log_folders=True, summaries=False,
                            n_combinations=None, n_subbash=4):
    """
    :param subs: subject list
    :param filetype: 'SSD' or 'SPOC'
    :param condition: 'mov' or 'nomov'
    :param task_request: '(r)egression' or '(c)lassification'
    :param component_mode: mode 1: only 'one-up'; mode 2: one-up or random choice of components
    :param eqcompmat: True: fixed number of columns in input (filled with zero-vectors if not enough data)
    :param seed: regarding randomization of folds, batches etc.
    :param repet_scalar: how many times it runs through whole set (can be also fraction)
    :param s_fold: number (s) of folds
    :param balanced_cv: non-overlapping, equal class-sized data splits at each fold
    :param batch_size: Batch size
    :param successive_mode: 1 or 2 (see load_data.py in next_batch())
    :param rand_batch: Should remain True
    :param plot: Whether to plot results after processing
    :param successive_default: How many of random batches shall remain in successive order. That is, the
    time-slices (1-sec each) that are kept in succession. Representing subjective experience, this could
    be 2-3 sec in order to capture responses to the stimulus environment.
    :param del_log_folders: Whether to delete log files and checkpoints after processing and plotting
    :param summaries: Whether verbose summaries (get deleted if del_log_folders==True)
    :param n_combinations: Number of random combinations in hyperparameter search
    :param n_subbash: Define number of sub-bashfiles (for distributed processing)
    :return:
    """

    # Adjust and test input variables
    if not isinstance(subs, int):
        subs = list(subs)
    else:
        subs = [subs]

    filetype = filetype.upper()
    assert filetype in ['SSD', 'SPOC'], "filetype must be either 'SSD' or 'SPOC'"
    cond = condition.lower()
    assert cond in ['mov', 'nomov'], "condition must be either 'mov' or 'nomov'"
    assert component_mode in [1, 2], "component_mode must be either 1 or 2 (int)"
    assert isinstance(n_subbash, int), "n_subbash must be integer"

    if del_log_folders and summaries:
        cprint("Note: Verbose summaries are redundant since they get deleted after processing.", "y")

    # Request variables if not given yet
    if n_combinations is None:
        n_combinations = int(cinput(
            "How many random combinations of hyperparameters to test (given value will be multiplied "
            "with n_subjects)): ", "b"))

    _tasks = ["regression", "classification"]
    if task_request is None:
        task_request = cinput(
            "For which task is the random search bash? ['r' for'regression', 'c' for 'classification']: ", 'b')
    assert task_request.lower() in _tasks[0] or task_request.lower() in _tasks[1], \
        "Task input must be eitehr 'r' or 'c'"
    task = _tasks[0] if task_request.lower() in _tasks[0] else _tasks[1]
    tfix = "BiCl" if task == "classification" else "Reg"

    shuffle = True if task == "classification" else False
    balanced_cv = False if task == "regression" else balanced_cv
    successive = 1 if task == "classification" else successive_default

    if eqcompmat is None:
        eqcompmat = ask_true_false(question="Shall the model input matrix always be the same in size?")
    eqcompmat = 10 if eqcompmat else 0
    # eqcompmat = int(cinput("What should be the number (int) of columns (i.e. components)?",
    #                        "b"))

    # Create bashfile if not there already:
    bash_file_name = p2_bash + f"bashfile_randomsearch_{cond}_{tfix}.sh"
    sub_bash_file_names = []
    subbash_suffix = ["_local.sh"] + [f"_{subba}.sh" for subba in range(1, n_subbash)]

    for sub_bash in subbash_suffix:
        sub_bash_file_name = "." + bash_file_name.split(".")[1] + sub_bash
        sub_bash_file_names.append(sub_bash_file_name)

    if not os.path.exists(bash_file_name):
        with open(bash_file_name, "w") as bashfile:  # 'a' for append
            bashfile.write("#!/usr/bin/env bash\n\n" + f"# Random Search Bashfile: {task}")
        for subash_fname in sub_bash_file_names:
            with open(subash_fname, "w") as bashfile:  # 'a' for append
                bashfile.write("#!/usr/bin/env bash\n\n"+"# Random Search Bashfile_{}: {}".format(
                    subash_fname.split("_")[-1].split(".")[0], task))

    # # Randomly Draw
    combi_count = 0
    for _ in range(n_combinations):

        # lstm_size
        n_lstm_layers = np.random.choice([1, 2])  # either 1 or 2 layers
        layer_size = [10, 15, 20, 25, 30, 40, 50, 65, 80, 100]  # possible layer sizes

        if n_lstm_layers == 1:
            lstm_size = np.random.choice(layer_size)
        else:  # n_lstm_layers == 2
            lstm_l1 = np.random.choice(layer_size)
            while True:  # size of second layer should be smaller or equal to size of first layer
                lstm_l2 = np.random.choice(layer_size)
                if lstm_l2 <= lstm_l1:
                    break
            lstm_size = f"{lstm_l1},{lstm_l2}"

        # fc_n_hidden
        n_fc_layers = np.random.choice(range(n_lstm_layers))  # n_fc_layers <= n_lstm_layers
        # note: if n_fc_layers == len(fc_n_hidden) == 0, there is 1 fc-lay attached to lstm,
        # so 1 n_fc_layers == 2 fc layers

        if n_fc_layers == 0:
            fc_n_hidden = 0
        else:
            while True:
                fc_n_hidden = np.random.choice(layer_size)
                if n_lstm_layers == 1:
                    if fc_n_hidden <= lstm_size:
                        break
                else:  # n_lstm_layers == 2
                    if fc_n_hidden <= int(lstm_size.split(",")[1]):
                        break

        # learning_rate
        # learning_rate = np.random.choice(a=['1e-1', '1e-2', '1e-3', '5e-4'])
        learning_rate = np.random.choice(a=['1e-2', '1e-3', '5e-4'])

        # weight_reg
        weight_reg = np.random.choice(a=['l1', 'l2'])

        # weight_reg_strength
        weight_reg_strength = np.random.choice(a=[1e-5, 0.18, 0.36, 0.72, 1.44])  # .00001 as no regul.

        # activation_fct
        activation_fct = np.random.choice(a=['elu', 'relu'])

        # hilbert_power
        hilbert_power = False
        # hilbert_power = np.random.choice(a=[True, False])  # had no impact on model performance

        # band_pass
        if filetype == "SPOC":
            band_pass = True  # there is no non-band-pass SPOC data (yet).
        else:  # filetype == "SSD"
            band_pass = True  # for comparison to CSP
            # band_pass = np.random.choice(a=[True, False])

        # hrcomp
        hrcomp = False
        # hrcomp = np.random.choice(a=[True, False])

        # component
        if component_mode == 1:
            component_modes = "one_up"
        else:  # component_mode == 2
            component_modes = np.random.choice(a=["random_set", "one_up"])

        # From here on it is subject-dependent
        ncomp = [get_num_components(sub, cond, filetype) for sub in subs]  # TODO SPOC not there yet
        max_n_comp = max(ncomp)

        # Randomly choice number of feed-components
        while True:
            choose_n_comp = np.random.randint(1, max_n_comp+1)  # x
            if choose_n_comp <= 10:  # don't feed more than 10 components
                break

        # Prepare to write line in bash file per subject
        for sidx, sub in enumerate(subs):

            sub_component = draw_components(n=choose_n_comp, subject=sub, condition=cond, filetype=filetype,
                                            select_mode=component_modes, dtype='str')

            # path_specificities
            # TODO: Could indicate in comp- which selection criterion applied (update then also best_hyperparameter.py)
            path_specificities = f"{tfix}_{cond}_RndHPS_" \
                f"lstm-{'-'.join(str(lstm_size).split(','))}_" \
                f"fc-{'-'.join(str(fc_n_hidden).split(','))}_" \
                f"lr-{learning_rate}_wreg-{weight_reg}-{weight_reg_strength:.2f}_" \
                f"actfunc-{activation_fct}_ftype-{filetype}_hilb-{'T' if hilbert_power else 'F'}_" \
                f"bpass-{'T' if band_pass else 'F'}_comp-{'-'.join(str(sub_component).split(','))}_" \
                f"hrcomp-{'T' if hrcomp else 'F'}_fixncol-{eqcompmat}"

            if task == "classification":
                path_specificities += f"_shuf-{'T' if shuffle else 'F'}_balcv-{'T' if balanced_cv else 'F'}/"
            else:
                path_specificities += "/"

                # Write line for bashfile
            bash_line = f"python3 NeVRo.py --subject {sub} --condition {cond} --seed {seed} " \
                f"--task {task} --shuffle {shuffle} --balanced_cv {balanced_cv} " \
                f"--repet_scalar {repet_scalar} --s_fold {s_fold} " \
                f"--batch_size {batch_size} --successive {successive} " \
                f"--successive_mode {successive_mode} --rand_batch {rand_batch} --plot {plot} " \
                f"--dellog {del_log_folders} --lstm_size {lstm_size} --fc_n_hidden {fc_n_hidden} " \
                f"--learning_rate {learning_rate} --weight_reg {weight_reg} " \
                f"--weight_reg_strength {weight_reg_strength} --activation_fct {activation_fct} " \
                f"--filetype {filetype} --hilbert_power {hilbert_power} --band_pass {band_pass} " \
                f"--component {sub_component} --hrcomp {hrcomp} --eqcompmat {eqcompmat} " \
                f"--summaries {summaries} --path_specificities {path_specificities}"

            # Write in bashfile
            if not os.path.exists("./processed/"):
                os.mkdir("./processed/")

            with open(bash_file_name, "a") as bashfile:  # 'a' for append
                bashfile.write("\n"+bash_line)

            # and in subbashfile
            sub_bash_file_name = sub_bash_file_names[combi_count]

            with open(sub_bash_file_name, "a") as sub_bashfile:  # 'a' for append
                sub_bashfile.write("\n"+bash_line)

            # Fill in Random_Search_Table.csv
            table_name = f"./processed/Random_Search_Table_{cond}_{tfix}.csv"

            if not os.path.exists(table_name):
                rs_table = np.array(['round', 'subject', 'cond', 'seed', 'task',
                                     'shuffle', 'repet_scalar', 's_fold', 'balanced_cv', 'batch_size',
                                     'successive', 'successive_mode', 'rand_batch', 'plot',
                                     'lstm_size', 'fc_n_hidden', 'learning_rate',
                                     'weight_reg', 'weight_reg_strength',
                                     'activation_fct', 'filetype', 'hilbert_power', 'band_pass',
                                     'component', 'hrcomp', 'fixncol', 'summaries',
                                     'path_specificities',
                                     'mean_val_acc', 'meanline_acc', 'mean_class_val_acc'],
                                    dtype='<U150')

                rs_table = np.reshape(rs_table, newshape=(-1, rs_table.shape[0]))
                # Could write del_log_folders in table

            else:
                rs_table = np.genfromtxt(table_name, delimiter=";", dtype=str)

            rnd = int(rs_table[-1, 0]) + 1 if rs_table[-1, 0].isnumeric() else 0

            exp_data = [rnd, sub, cond, seed, task,
                        shuffle, repet_scalar, s_fold, balanced_cv, batch_size,
                        successive, successive_mode, rand_batch, plot,
                        lstm_size, fc_n_hidden, learning_rate,
                        weight_reg, weight_reg_strength,
                        activation_fct, filetype, hilbert_power, band_pass,
                        sub_component, hrcomp, eqcompmat, summaries,
                        path_specificities]

            fill_vec = np.repeat(a="nan", repeats=rs_table.shape[1])
            fill_vec = fill_vec.reshape((-1, len(fill_vec)))
            rs_table = np.concatenate((rs_table, fill_vec), axis=0).astype("<U150")
            rs_table[-1, 0:len(exp_data)] = exp_data

            np.savetxt(fname=table_name, X=rs_table, delimiter=";", fmt="%s")

            # Set Counter
            combi_count = combi_count+1 if combi_count < n_subbash-1 else 0

    print("\nBashfiles and table completed.")


def write_bash_from_table(subs, table_path, condition, task=None, n_subbash=4, del_log_folders=True):
    """
    Write executable bash scripts for list of subjects (usually full cohort). Corresponding hyperparameters will be
    extracted from given table (path). This is (usually) done for a 'narrow' search over best hyperparameter settings
    that are received from a broad random search on a subset of subjects.
    :param subs: list of subjects
    :param table_path: path to table. Asserts table to be in subfolder './processed/Random_Search_Tables/...'
    :param condition: 'mov' OR 'nomov'
    :param task: 'classification' OR 'regression'; If None: try to extract from path
    :param n_subbash: Define number of sub-bashfiles (for distributed processing)
    :param del_log_folders: Whether to delete log files and checkpoints after processing and plotting
    """

    # Check given arguments
    tfixes = ["BiCl", "Reg"]

    if task:
        task = task.lower()
        assert task in ["classification", "regression"], "task must be either 'classification' or 'regression'"
        tfix = tfixes[0] if task == "classification" else tfixes[1]
    else:
        try:
            tfix = tfixes[np.where([fix in table_path for fix in tfixes])[0][0]]
            task = "classification" if tfix == tfixes[0] else "regression"
        except IndexError:
            raise TypeError("'task' argument is necessary to proceed. Could not be extracted from path.")

    condition = condition.lower()
    assert "mov" in condition, "condition must be either 'mov' or 'nomov'"
    cond = "nomov" if "no" in condition else "mov"

    assert isinstance(n_subbash, int), "n_subbash must be integer"
    subbash_suffix = ["_local.sh"] + [f"_{subba}.sh" for subba in range(1, n_subbash)]

    # Set full path to table (asserts that table is in '.../0_broad_search/{task}/per_subject/
    # [write full path implementation if necessary]
    wd_tables = f"./processed/Random_Search_Tables/{cond}/0_broad_search/{task}/per_subject/"
    full_table_path = wd_tables + table_path
    # table_path = wd_tables \
    #              + "unique_Best_2_HPsets_over_10_Subjects_mean_acc_0.660_Random_Search_Table_BiCl.csv"

    if not isinstance(subs, list) and not isinstance(subs, np.ndarray):
        subs = [subs]

    num_sub = len(subs)

    assert os.path.exists(full_table_path), \
        f"Given table '{table_path}' wasn't found! Asserts table to be in '{wd_tables}'"
    hp_table = np.genfromtxt(full_table_path, delimiter=";", dtype=str)

    # Create new HP-table
    n_combis = hp_table[1:].shape[0]
    rounds = np.arange(n_combis*num_sub)
    rounds = np.reshape(rounds, newshape=(len(rounds), 1))
    subs = np.tile(subs, n_combis)
    subs = np.reshape(subs, newshape=(len(subs), 1))
    lside_table = np.concatenate((rounds, subs), 1)
    lside_header = np.reshape(np.array(['round', 'subject'], dtype='<U150'), newshape=(1, 2))
    lside_table = np.concatenate((lside_header, lside_table))
    rside_header = np.reshape(np.array(['mean_val_acc', 'meanline_acc', 'mean_class_val_acc'],
                                       dtype='<U150'), (1, 3))
    rside_table = np.reshape(np.repeat(np.repeat(a="nan", repeats=n_combis*num_sub), 3), newshape=(-1, 3))
    rside_table = np.concatenate((rside_header, rside_table))
    mid_table = np.repeat(hp_table[1:, :], num_sub, axis=0)
    mid_header = np.reshape(hp_table[0, :], newshape=(1, -1))
    mid_table = np.concatenate((mid_header, mid_table))

    new_hp_table = np.concatenate((np.concatenate((lside_table, mid_table), 1), rside_table), 1)

    # Check whether (fixed) components are given or must be generated/drawn
    idx_sub = 1  # subject
    idx_comp = np.where(new_hp_table[0, :] == "component")[0][0]
    idx_path = np.where(new_hp_table[0, :] == "path_specificities")[0][0]

    fix_comp = False if np.any(["n_max" in row for row in new_hp_table[1:, idx_comp]]) else True

    if not fix_comp:
        # Generate component list for each subject
        filetype = np.unique(new_hp_table[1:, np.where(new_hp_table[0, :] == "filetype")[0][0]])
        assert len(filetype) == 1, f"filetype in table must be the same. Found '{filetype}'"
        filetype = filetype[0]

        for ridx, row in enumerate(new_hp_table[1:, :]):
            # Get max N components per hyperparameter set (i.e., row)
            n_max = int(row[idx_comp].split("=")[1])
            # Draw components for subjects individually
            sub_comps = draw_components(n=n_max, subject=int(row[idx_sub]), condition=cond, filetype=filetype,
                                        select_mode="one_up", dtype="str")

            # Overwrite in table: 'component' AND 'path_specificities'
            new_hp_table[ridx+1, idx_comp] = sub_comps  # 'component'

            pspecs = row[idx_path]  # 'path_specificities'
            pspecs = pspecs.split("_comp")[0] + f"_comp-{'-'.join(sub_comps.split(','))}_hrc" + pspecs.split("_hrc")[1]
            new_hp_table[ridx + 1, idx_path] = pspecs

    # Save new HP-table
    new_table_name = f"./processed/Narrow_Search_Table_{cond}_{tfix}.csv"
    np.savetxt(fname=new_table_name, X=new_hp_table, delimiter=";", fmt="%s")

    # Create bashfile if not there already:
    bash_filename = p2_bash + f"bashfile_narrowsearch_{cond}_{tfix}.sh"
    if not os.path.exists(bash_filename):
        with open(bash_filename, "w") as bash_file:  # 'a' for append
            bash_file.write("#!/usr/bin/env bash\n\n" + "# Narrow Search Bashfile:")

        for subbash in subbash_suffix:
            subbash_filename = bash_filename.split(".sh")[0] + subbash
            with open(subbash_filename, "w") as bash_file:  # 'a' for append
                bash_file.write(
                    "#!/usr/bin/env bash\n\n" + f"# Narrow Search Bashfile{subbash.split('.')[0]}: {task}")

    # Write according bashfiles
    combi_count = 0
    for line in new_hp_table[1:, 1:-3]:

        subject, cond, seed, task, shuffle, \
            repet_scalar, s_fold, balanced_cv, batch_size, \
            successive, successive_mode, rand_batch, \
            plot, \
            lstm_size, fc_n_hidden, learning_rate, \
            weight_reg, weight_reg_strength,\
            activation_fct, \
            filetype, hilbert_power, band_pass, \
            component, hrcomp, eqcompmat, summaries, \
            path_specificities = line

        # Write line for bashfile (Important: [Space] after each entry)
        bash_line = f"python3 NeVRo.py --subject {subject} --condition {cond} --seed {seed} " \
                    f"--task {task} --shuffle {shuffle} --balanced_cv {balanced_cv} " \
                    f"--repet_scalar {repet_scalar} --s_fold {s_fold} " \
                    f"--batch_size {batch_size} --successive {successive} " \
                    f"--successive_mode {successive_mode} --rand_batch {rand_batch} --plot {plot} " \
                    f"--dellog {del_log_folders} --lstm_size {lstm_size} --fc_n_hidden {fc_n_hidden} " \
                    f"--learning_rate {learning_rate} --weight_reg {weight_reg} " \
                    f"--weight_reg_strength {weight_reg_strength} --activation_fct {activation_fct} " \
                    f"--filetype {filetype} --hilbert_power {hilbert_power} --band_pass {band_pass} " \
                    f"--component {component} --hrcomp {hrcomp} --eqcompmat {eqcompmat} " \
                    f"--summaries {summaries} --path_specificities {path_specificities}"

        # Write in bashfile
        with open(bash_filename, "a") as bashfile:  # 'a' for append
            bashfile.write("\n" + bash_line)

        # and in subbashfile
        subbash = subbash_suffix[combi_count]
        sub_bash_file_name = bash_filename.split(".sh")[0] + subbash
        with open(sub_bash_file_name, "a") as subbashfile:  # 'a' for append
            subbashfile.write("\n" + bash_line)

        # Set Counter
        combi_count = combi_count + 1 if combi_count < n_subbash-1 else 0

    print("\nBashfiles and table completed.")

# write_bash_from_table(subs=subjects,
#                       table_path='unique_Best_2_HPsets_over_10_Subjects_mean_acc_0.660_Random_Search_Table_BiCl.csv')
# write_bash_from_table(subs=subjects,
#                       table_path='unique_Best_2_HPsets_over_10_Subjects_mean_acc_0.717_Random_Search_Table_BiCl.csv',
#                       condition="nomov", task="classification")


def update_bashfiles(table_name=None, subject=None, path_specs=None, all_runs=False, unhash=False):
    """Comment out all lines in bashfile(s) which have been run."""

    if os.path.exists(table_name):
        rs_table = np.genfromtxt(table_name, delimiter=";", dtype=str)
        idx_sub = np.where(rs_table[0, :] == 'subject')[0][0]  # find column "subject"
        idx_pspec = np.where(rs_table[0, :] == 'path_specificities')[0][0]  # column "path_specificities"

        if not unhash:

            if not all_runs:
                subject = int(subject)
                # Find entry which needs to be updated
                idx_sub = np.where(rs_table[:, idx_sub] == str(subject))[0]  # rows with respective subject
                idx_pspec = np.where(rs_table[:, idx_pspec] == path_specs)[0]  # find respective path_specs
                idx_check = list(set(idx_sub).intersection(idx_pspec))  # find index of case at hand

                # If there is an entry, the run was successful, update bashfile(s)
                if not np.all(rs_table[idx_check, -3:] == 'nan'):

                    # Run through all bashfiles and comment out those lines that were successfully executed
                    for bfile in os.listdir("./bashfiles/"):
                        if bfile.split(".")[-1] == 'sh':  # check whether bash file
                            for line in fileinput.input("./bashfiles/" + bfile, inplace=True):
                                if path_specs in line and str(subject) == line.split("subject ")[1].split(" --")[0] \
                                        and "#" not in line:
                                    # Note: This doesn't print in console, but overwrites in file
                                    sys.stdout.write(f'# {line}')
                                    # sys.stdout.write() instead of print() avoids new lines
                                else:
                                    sys.stdout.write(line)

            else:  # check for all runs

                # Run through table and find successfully executed entries
                for idx, rs_line in enumerate(rs_table[1:, -3:]):
                    if not np.all(rs_line == 'nan'):
                        subject = rs_table[1+idx, idx_sub]
                        path_specs = rs_table[1+idx, idx_pspec]

                        for bfile in os.listdir("./bashfiles/"):
                            if bfile.split(".")[-1] == 'sh':  # check whether bash file
                                for line in fileinput.input("./bashfiles/" + bfile, inplace=True):
                                    if path_specs in line and str(subject) == line.split("subject ")[1].split(" --")[0] \
                                            and "#" not in line:
                                        sys.stdout.write(f'# {line}')
                                    else:
                                        sys.stdout.write(line)

        else:

            # If unhash: Remove hash from lines which weren't processed yet according to table
            for idx, rs_line in enumerate(rs_table[1:, -3:]):
                if np.all(rs_line == 'nan'):
                    subject = rs_table[1 + idx, idx_sub]
                    path_specs = rs_table[1 + idx, idx_pspec]

                    for bfile in os.listdir("./bashfiles/"):
                        if bfile.split(".")[-1] == 'sh':  # check whether bash file
                            for line in fileinput.input("./bashfiles/" + bfile, inplace=True):
                                if path_specs in line and str(subject) == line.split("subject ")[1].split(" --")[0] \
                                        and "#" in line:
                                    sys.stdout.write(f'{line.split("# ")[-1]}')
                                else:
                                    sys.stdout.write(line)

    else:
        cprint(f"There is no corresponding table: '{table_name}'", 'r')


# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

# # # Adapt here for which subjects bash files should be written and in which condition

if __name__ == "__main__":
    wb = ask_true_false(question="Do you want to write new bashfiles?")
    if wb:
        # Check for which datatype: SSD or SPoC
        datatype = cinput("For which datatype: 'SSD' or 'SPOC'", 'b').upper()

        # Define condition for random search
        # condi = "nomov"  # "mov"  # manually
        if "condi" not in locals():
            condi = input("Write bash files for 'M'ov or 'N'oMov condition: ").lower()
            condi = "nomov" if "n" in condi else "mov"
        cprint(f"Condition is set to: {condi}", 'y')

        tasks = ["regression", "classification"]

        # # Set up subjects
        n_sub = 45  # number of all tested subjects
        all_subjects = np.linspace(start=1, stop=n_sub, num=n_sub, dtype=int)  # np.arange(1, n_sub+1)

        # # Define Dropouts
        if datatype == "SSD":
            # Find subjects with valid components in the selection table (so far only for SSD)
            # Load SSD selection table
            ssd_comp_sel_tab = pd.read_csv(
                f"../../../Data/EEG/07_SSD/{condi}/SSD_selected_components_{condi}.csv", sep=";")

            # Remove Subjects without any selected component
            dropouts = np.array(ssd_comp_sel_tab[pd.isna(ssd_comp_sel_tab['n_sel_comps'])]["# ID"])
            dropouts = np.append(dropouts,
                                 np.array(ssd_comp_sel_tab[ssd_comp_sel_tab['n_sel_comps'] == 0]["# ID"]))
            # # Dropout Criterion can be set to threshold, e.g. here min 4 selected comps:
            dropouts = np.append(dropouts,
                                 np.array(ssd_comp_sel_tab[ssd_comp_sel_tab['n_sel_comps'] < 4]["# ID"]))

        elif datatype == "SPOC":
            raise NotImplementedError("Dropouts are not defined yet via SPOC component selection.")

            # # Alternatively define dropouts here:
            # dropouts = np.array([1, 12, 32, 33, 38, 40, 45])  # due to acquisition problems

            # These are additional dropouts for the nomov-condition (add_drop). Criterium: More than 1/3
            # of all epochs (1 epoch = 1 sec of EEG data) in the EEG-channel data were above 100 μV. This
            # would lead to a calculation of unreliable ICA-weights, dominated by the noisy segments.
            # add_drop = [10, 16, 19, 23, 41, 43]  # no mov
            # TODO add_drop = [...]  # mov

            # dropouts = np.append(dropouts, add_drop)

        else:
            raise ValueError(f"Given datatype '{datatype}' does not exist.")

        subjects = np.setdiff1d(all_subjects, dropouts)  # These are all subjects without dropouts

        # # For pure testing define subjects here:
        # subjects = [21, 37]  # set manually
        testing = False  # set to True for testing

        # # Without already computed subjects
        # already_proc = [2, 36]  #  manually set already processed subjects
        # subjects = np.setdiff1d(subjects, already_proc)

        # # Step 1) Broad random search on subset of 10 subjects
        brs = ask_true_false(question="Do you want to run a broad random search on random subset?")
        if brs:
            subsubjects = np.random.choice(a=subjects, size=10, replace=False)

            for _task in tasks:  # Binary classification & Regression

                # Broad random search
                write_search_bash_files(subs=subsubjects, filetype=datatype, condition=condi, balanced_cv=True,
                                        task_request=_task, component_mode=1, eqcompmat=True, n_combinations=20,
                                        seed=True, repet_scalar=20, n_subbash=8)
        else:
            if testing:
                write_search_bash_files(subs=subjects, filetype=datatype, condition=condi,
                                        task_request=None, component_mode=1, eqcompmat=None,
                                        seed=False, repet_scalar=5, s_fold=10, balanced_cv=True,
                                        batch_size=9, successive_mode=1, rand_batch=True, plot=True,
                                        successive_default=3, del_log_folders=True, summaries=False,
                                        n_combinations=None, n_subbash=4)

            else:
                # Step 2) Run model over pre-selected hyperparameter sets on whole dataset
                cprint("\nCreate table and bashfiles for narrow search on pre-selected hp-sets on whole dataset.", "b")

                # Narrow search
                for _task in tasks:
                    file_found = False
                    p2tables = f"./processed/Random_Search_Tables/{condi}/0_broad_search/{_task}/per_subject/"
                    if os.path.exists(p2tables):
                        for tab_name in os.listdir(p2tables):
                            if "unique" in tab_name:
                                write_bash_from_table(subs=subjects, table_path=tab_name, condition=condi, task=_task,
                                                      n_subbash=8, del_log_folders=True)
                                file_found = True

                    if not file_found:
                        cprint(f"For {_task} task no corresponding 'unique...'-table found in '{p2tables}'.", 'y')

        end()
# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<  END

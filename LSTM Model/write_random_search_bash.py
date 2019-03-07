# coding=utf-8
"""
Write a random search bash file.

Author: Simon Hofmann | <[surname].[lastname][at]protonmail.com> | 2017, 2019 (Update)
"""

from load_data import *

# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>
# Set paths to bashfile dir
p2_bash = './bashfiles/'

# If no bashfile dir: create it
if not os.path.exists(p2_bash):
    os.mkdir(p2_bash)
# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

# # Adapt here for which subjects bashiles should be written

n_sub = 45
# all_subjects = np.linspace(start=1, stop=n_sub, num=n_sub, dtype=int)  # np.arange(1, n_sub+1)
# dropouts = np.array([1, 12, 32, 33, 38, 40, 45])

subjects = [2, 36]  # Test
# already_proc = [2, 36]  # already processed subjects

# # These are all subjects without dropouts
# subjects = np.setdiff1d(all_subjects, dropouts)

# # Without already computed subjects
# subjects = np.setdiff1d(subjects, already_proc)

# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>


# TODO adapt
def write_search_bash_files(subs, filetype, condition):

    # # Following need to be set manually (Default)
    seed = True  # TODO revisit
    repet_scalar = 30
    s_fold = 10
    sba = True
    batch_size = 9
    successive_mode = 1
    rand_batch = True
    plot = True
    # How many of random batches shall remain in successive order. That is, the time-slices (1-sec each)
    # that are kept in succession. Representing subjective experience, this could be 2-3 sec in order to
    # capture responses to the stimulus environment.
    successive_default = 3
    del_log_folders = True

    # Adjust input variable
    if not type(subs) is list:
        subs = [subs]
    # strsubs = [s(strsub) for strsub in subs]  # create form such as: ['S02', 'S36']

    filetype = filetype.upper()  # filetype (alternatively: np.random.choice(a=['SSD', 'SPOC']))
    assert filetype in ['SSD', 'SPOC'], "filetype must be either 'SSD' or 'SPOC'"
    cond = condition.lower()
    assert cond in ['mov', 'nomov'], "condition must be either 'mov' or 'nomov'"

    # Request
    n_combinations = int(cinput(
        "How many combinations (multiple of 4) to test (given value will be multpied with n_subjects)): ",
        "b"))
    assert n_combinations % 4 == 0, "Number of combinations must be a multiple of 4"
    # TODO why multiple of 4: ? due to 4 bashscripts? ["_local.sh", "_1.sh", "_2.sh", "_3.sh"] ?

    tasks = ["regression", "classification"]
    task_request = cinput(
        "For which task is the random search bash? ['r' for'regression', 'c' for 'classification']: ",
        "b")
    assert task_request.lower() in tasks[0] or task_request.lower() in tasks[1], \
        "Input must be eitehr 'r' or 'c'"
    task = tasks[0] if task_request.lower() in tasks[0] else tasks[1]
    shuffle = True if task == "classification" else False
    successive = 1 if task == "classification" else successive_default

    eqcompmat = ask_true_false(question="Shall the model input matrix always be the same in size?")
    if eqcompmat:
        eqcompmat = int(cinput("What should be the number (int) of columns (i.e. components)?", "b"))
    else:
        eqcompmat = None

    # Create bashfile if not there already:
    bash_file_name = p2_bash + "bashfile_randomsearch_{}.sh".format('BiCl' if "c" in task else "Reg")
    sub_bash_file_names = []
    for sub_bash in ["_local.sh", "_1.sh", "_2.sh", "_3.sh"]:
        sub_bash_file_name = "." + bash_file_name.split(".")[1] + sub_bash
        sub_bash_file_names.append(sub_bash_file_name)

    if not os.path.exists(bash_file_name):
        with open(bash_file_name, "w") as bashfile:  # 'a' for append
            bashfile.write("#!/usr/bin/env bash\n\n" + "# Random Search Bashfile: {}".format(task))
        for subash_fname in sub_bash_file_names:
            with open(subash_fname, "w") as bashfile:  # 'a' for append
                bashfile.write("#!/usr/bin/env bash\n\n"+"# Random Search Bashfile_{}: {}".format(
                    subash_fname.split("_")[-1].split(".")[0], task))

    # # Randomly Draw
    combi_count = 0
    for combi in range(n_combinations):

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
            lstm_size = "{},{}".format(lstm_l1, lstm_l2)

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
        weight_reg_strength = np.random.choice(a=[0.00001, 0.18, 0.36, 0.72, 1.44])  # 0. == no regulariz.

        # activation_fct
        activation_fct = np.random.choice(a=['elu', 'relu'])

        # hilbert_power
        hilbert_power = np.random.choice(a=[True, False])

        # band_pass
        if filetype == "SPOC":
            band_pass = True  # there is no non-band-pass SPOC data (yet).
        else:  # filetype == "SSD"
            band_pass = np.random.choice(a=[True, False])

        # hrcomp
        hrcomp = np.random.choice(a=[True, False])

        # component
        # component_modes = np.random.choice(a=["best", "random_set", "one_up"])
        component_modes = np.random.choice(a=["random_set", "one_up"])  # TODO new Xcorr table required

        # From here on it is subject-dependent
        ncomp = [get_num_components(sub, cond, filetype, sba) for sub in subs]  # TODO SPOC not there yet
        max_n_comp = max(ncomp)  # init
        sub_dep = False  # init

        if component_modes == "best":
            component = "best"

        else:  # component_modes == 'random_set' or == 'one_up'
            # Randomly choice number of feed-components

            while True:
                choose_n_comp = np.random.randint(1, max_n_comp+1)  # x
                if choose_n_comp <= 10:  # don't feed more than 10 components
                    break

            if component_modes == "one_up":
                # Choose from component 1 to n_choose (see: SPOC(comp_order) & SSD(alpha-hypotheses)):
                component = np.arange(start=1, stop=choose_n_comp + 1)  # range [1, n_choose]

            else:  # component_modes == "random_set"
                # Choose x random components, where x == choose_n_comp
                component = np.sort(np.random.choice(a=range(1, max_n_comp + 1), size=choose_n_comp,
                                                     replace=False))

            # Does this needs to be adapted per subject?
            if not all([choose_n_comp <= maxcomp for maxcomp in ncomp]):
                sub_dep = True

        # eqcompmat
        if eqcompmat is not None:
            eqcompmat = max_n_comp if max_n_comp > eqcompmat else eqcompmat

        # Prepare to write line in bash file per subject
        for sidx, sub in enumerate(subs):

            sub_component = component

            if component_modes != "best":
                # Shorten component list for subject, if necessary
                if sub_dep:
                    while len(sub_component) > ncomp[sidx]:
                        # Delete highest component number
                        # Works for component_modes: "random_set" AND "one_up"
                        sub_component = np.delete(sub_component,
                                                  np.where(sub_component == max(sub_component)))

                sub_component = ','.join([str(i) for i in sub_component])

            # path_specificities
            path_specificities = "{}RndHPS_lstm-{}_fc-{}_lr-{}_wreg-{}-{:.2f}_actfunc-{}_ftype-{}_" \
                                 "hilb-{}_bpass-{}_comp-{}_" \
                                 "hrcomp-{}_ncol-{}/".format('BiCl_' if "c" in task else "Reg_",
                                                             "-".join(str(lstm_size).split(",")),
                                                             "-".join(str(fc_n_hidden).split(",")),
                                                             learning_rate,
                                                             weight_reg, weight_reg_strength,
                                                             activation_fct,
                                                             filetype, "T" if hilbert_power else "F",
                                                             "T" if band_pass else "F",
                                                             "-".join(str(sub_component).split(",")),
                                                             "T" if hrcomp else "F", eqcompmat)

            # TODO integrate one line in the end where the file moves itself to bashfile_done folder
            # TODO Hash those lines which are processed.
            # Write line for bashfile
            bash_line = "python3 NeVRo.py " \
                        "--subject {} --condition {} --seed {} --task {} --shuffle {} " \
                        "--repet_scalar {} --s_fold {} --batch_size {} " \
                        "--successive {} --successive_mode {} --rand_batch {} " \
                        "--plot {} --dellog {} " \
                        "--lstm_size {} --fc_n_hidden {} --learning_rate {} " \
                        "--weight_reg {} --weight_reg_strength {} " \
                        "--activation_fct {} " \
                        "--filetype {} --hilbert_power {} --band_pass {} " \
                        "--component {} --hrcomp {} " \
                        "--path_specificities {}".format(sub, cond, seed, task, shuffle,
                                                         repet_scalar, s_fold, batch_size,
                                                         successive, successive_mode, rand_batch,
                                                         plot, del_log_folders,
                                                         lstm_size, fc_n_hidden, learning_rate,
                                                         weight_reg, weight_reg_strength,
                                                         activation_fct,
                                                         filetype, hilbert_power, band_pass,
                                                         sub_component, hrcomp,
                                                         path_specificities)

            # Write in bashfile
            if not os.path.exists("./LSTM/"):
                os.mkdir("./LSTM/")

            with open(bash_file_name, "a") as bashfile:  # 'a' for append
                bashfile.write("\n"+bash_line)

            # and in subbashfile
            sub_bash_file_name = sub_bash_file_names[combi_count]

            with open(sub_bash_file_name, "a") as sub_bashfile:  # 'a' for append
                sub_bashfile.write("\n"+bash_line)

            # Fill in Random_Search_Table.csv
            table_name = "./LSTM/Random_Search_Table_{}.csv".format('BiCl' if "c" in task else "Reg")

            if not os.path.exists(table_name):
                rs_table = np.array(['round', 'subject', 'cond', 'seed', 'task',
                                     'shuffle', 'repet_scalar', 's_fold', 'batch_size',
                                     'successive', 'successive_mode', 'rand_batch', 'plot',
                                     'lstm_size', 'fc_n_hidden', 'learning_rate',
                                     'weight_reg', 'weight_reg_strength',
                                     'activation_fct', 'filetype', 'hilbert_power', 'band_pass',
                                     'component', 'hrcomp',
                                     'path_specificities',
                                     'mean_val_acc', 'zeroline_acc', 'mean_class_val_acc'],
                                    dtype='<U113')
                # Could write del_log_folders in table
            else:
                rs_table = np.genfromtxt(table_name, delimiter=";", dtype=str)

            rs_table = np.reshape(rs_table, newshape=(-1, 28))

            rnd = int(rs_table[-1, 0]) + 1 if rs_table[-1, 0].isnumeric() else 0

            exp_data = [rnd, sub, cond, seed, task,
                        shuffle, repet_scalar, s_fold, batch_size,
                        successive, successive_mode, rand_batch, plot,
                        lstm_size, fc_n_hidden, learning_rate,
                        weight_reg, weight_reg_strength,
                        activation_fct, filetype, hilbert_power, band_pass,
                        sub_component, hrcomp,
                        path_specificities]

            fill_vec = np.repeat(a="nan", repeats=rs_table.shape[1])
            fill_vec = fill_vec.reshape((-1, len(fill_vec)))
            rs_table = np.concatenate((rs_table, fill_vec), axis=0).astype("<U120")
            rs_table[-1, 0:len(exp_data)] = exp_data

            np.savetxt(fname=table_name, X=rs_table, delimiter=";", fmt="%s")

            # Set Counter
            combi_count = combi_count+1 if combi_count < 3 else 0

    print("\nBashfiles and table completed.")

# write_search_bash_files(subs=subjects, filetype="SSD", condition="nomov")


# TODO continue here
def write_bash_from_table(subs, table_path):

    # # Following need to be set manually (Default)
    del_log_folders = True

    wd_tables = "./LSTM/Random Search Tables/"
    table_path = wd_tables + table_path
    # table_path = wd_tables \
    #              + "unique_Best_2_HPsets_over_10_Subjects_mean_acc_0.660_Random_Search_Table_BiCl.csv"

    if not isinstance(subs, list) and not isinstance(subs, np.ndarray):
        subs = [subs]

    num_sub = len(subs)

    assert os.path.exists(table_path), "Given table path does not exist"
    hp_table = np.genfromtxt(table_path, delimiter=";", dtype=str)

    # Create new HP-table
    n_combis = hp_table[1:].shape[0]
    rounds = np.arange(n_combis*num_sub)
    rounds = np.reshape(rounds, newshape=(len(rounds), 1))
    subs = np.tile(subs, n_combis)
    subs = np.reshape(subs, newshape=(len(subs), 1))
    lside_table = np.concatenate((rounds, subs), 1)
    lside_header = np.reshape(np.array(['round', 'subject'], dtype='<U113'), newshape=(1, 2))
    lside_table = np.concatenate((lside_header, lside_table))
    rside_header = np.reshape(np.array(['mean_val_acc', 'zeroline_acc', 'mean_class_val_acc'],
                                       dtype='<U113'), (1, 3))
    rside_table = np.reshape(np.repeat(np.repeat(a="nan", repeats=n_combis*num_sub), 3), newshape=(-1, 3))
    rside_table = np.concatenate((rside_header, rside_table))
    mid_table = np.repeat(hp_table[1:, :], num_sub, axis=0)
    mid_header = np.reshape(hp_table[0, :], newshape=(1, -1))
    mid_table = np.concatenate((mid_header, mid_table))

    new_hp_table = np.concatenate((np.concatenate((lside_table, mid_table), 1), rside_table), 1)

    # Save new HP-table
    new_table_name = "./LSTM/" + "Ran" + table_path.split("_Ran")[-1]
    np.savetxt(fname=new_table_name, X=new_hp_table, delimiter=";", fmt="%s")

    # Create bashfile if not there already:
    bash_filename = "bashfile_specific_search_{}.sh".format(table_path.split("_")[-1].split(".")[0])
    if not os.path.exists(bash_filename):
        with open(bash_filename, "w") as bash_file:  # 'a' for append
            bash_file.write("#!/usr/bin/env bash\n\n" + "# Specific Search Bashfile:")

        for subbash in ["_local.sh", "_1.sh", "_2.sh", "_3.sh"]:
            subbash_filename = bash_filename.split(".")[0] + subbash
            with open(subbash_filename, "w") as bash_file:  # 'a' for append
                bash_file.write(
                    "#!/usr/bin/env bash\n\n" + "# Specific Search Bashfile{}:".format(
                        subbash.split(".")[0]))

    # Write according bashfiles
    combi_count = 0
    for line in new_hp_table[1:, 1:-3]:

        subject, cond, seed, task, shuffle, \
            repet_scalar, s_fold, batch_size, \
            successive, successive_mode, rand_batch, \
            plot, \
            lstm_size, fc_n_hidden, learning_rate, \
            weight_reg, weight_reg_strength,\
            activation_fct, \
            filetype, hilbert_power, band_pass, \
            component, hrcomp, \
            path_specificities = line

        # Write line for bashfile (Important: [Space] after each entry)

        bash_line = "python3 NeVRo.py " \
                    "--subject {} --condition {} --seed {} --task {} --shuffle {} " \
                    "--repet_scalar {} --s_fold {} --batch_size {} " \
                    "--successive {} --successive_mode {} --rand_batch {} " \
                    "--plot {} --dellog {} " \
                    "--lstm_size {} --fc_n_hidden {} --learning_rate {} " \
                    "--weight_reg {} --weight_reg_strength {} " \
                    "--activation_fct {} " \
                    "--filetype {} --hilbert_power {} --band_pass {} " \
                    "--component {} --hrcomp {} " \
                    "--path_specificities {}".format(subject, cond, seed, task, shuffle,
                                                     repet_scalar, s_fold, batch_size,
                                                     successive, successive_mode, rand_batch,
                                                     plot, del_log_folders,
                                                     lstm_size, fc_n_hidden, learning_rate,
                                                     weight_reg, weight_reg_strength,
                                                     activation_fct,
                                                     filetype, hilbert_power, band_pass,
                                                     component, hrcomp,
                                                     path_specificities)

        # Write in bashfile
        with open(bash_filename, "a") as bashfile:  # 'a' for append
            bashfile.write("\n" + bash_line)

        # and in subbashfile
        subbash = ["_local.sh", "_1.sh", "_2.sh", "_3.sh"][combi_count]
        sub_bash_file_name = bash_filename.split(".")[0] + subbash
        with open(sub_bash_file_name, "a") as subbashfile:  # 'a' for append
            subbashfile.write("\n" + bash_line)

        # Set Counter
        combi_count = combi_count + 1 if combi_count < 3 else 0

    print("\nBashfiles and Table completed.")

# write_bash_from_table(
#     subs=subjects,
#     table_path='unique_Best_2_HPsets_over_10_Subjects_mean_acc_0.660_Random_Search_Table_BiCl.csv')
# write_bash_from_table(
#     subs=subjects,
#     table_path='unique_Best_2_HPsets_over_10_Subjects_mean_acc_0.046_Random_Search_Table_Reg.csv')

# coding=utf-8
"""
Main script (light)
• Light-weight version of NeVRo.py.
• Compatible with tensorflow >= 2.

Note: This in not fully tested yet [Dec 6, 2021]. The published model training was implemented in NeVRo.py

Author: Simon M. Hofmann | <[surname].[lastname][at]pm.me> | 2021
"""

# %% Import
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # don't use GPU

import sys
import argparse
import ast
import fileinput
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf  # for version >= 2.
import tensorflow.keras as keras
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
# import matplotlib.pyplot as plt

from utils import setwd, cprint, loop_timer, str2bool, s, end

# %% Set global vars & paths >> o <<<< o >>>> o <<<< o >>>> o <<<< o >>>> o <<<< o >>>> o <<<< o >>>> o <<
create_bash: bool = False  # toggle to create bash-files or run models

setwd("Modelling/LSTM")
p2nevro = os.getcwd()[:os.getcwd().find("NeVRo")+5]

wreg = {"l1": keras.regularizers.l1,
        "l2": keras.regularizers.l2}

# Hyperparameters
hps = ["subject", "condition", "sba", "task", "s_fold", "balanced_cv", "subblock_cv", "repet_scalar",
       "batch_size", "permutation", "n_components", "learning_rate", "weight_reg",
       "weight_reg_strength", "activation_fct", "lstm_size", "fc_n_hidden", "softmax"]  # keep order


# %% Functions >> o <<<< o >>>> o <<<< o >>>> o <<<< o >>>> o <<<< o >>>> o <<<< o >>>> o <<<< o >>>> o <<

def bash_line_from(source: pd.Series = None):
    """
    Create bashline from FLAGS or row of prediction table.

    :param source: None: uses variables in FLAGS OR row of prediction table
    :return: line for bashline
    """

    source = vars(FLAGS) if source is None else source

    bash_str = "python3 NeVRolight.py"
    for hp in hps:
        bash_str += f" --{hp} {source[hp]}"
    bash_str += "\n"

    return bash_str


def update_bashfiles(source: pd.Series = None):

    bash_str = bash_line_from(source=source)

    p2bash = os.path.join(os.getcwd(), "bashfiles")
    for bfn in os.listdir(p2bash):
        if bfn.endswith("_NeVRolight.sh"):
            for line in fileinput.input(files=os.path.join(p2bash, bfn), inplace=True):
                if bash_str in line and "#" not in line:
                    sys.stdout.write(f"# {line}")
                else:
                    sys.stdout.write(line)


def create_training_bash_file(condition: str, sba: bool, subblock_cv: bool, permutation: bool = False,
                              softmax: bool = False, task: str = "classification", n_hps: int = 2):

    # Preps args
    condition = condition.lower()
    task = task.lower()

    # Prep paths
    p2hps = os.path.join(os.getcwd(), "processed", "Random_Search_Tables", condition, "1_narrow_search",
                         task, "per_subject")
    p2bash = os.path.join(os.getcwd(), "bashfiles",
                          f"{str(datetime.now()).split(' ')[0]}_bashfile_"
                          f"{'subblock_cv'if subblock_cv else 'vanilla_cv'}_{condition}_"
                          f"{'BiCl' if 'classi' in task else 'Reg'}_NeVRolight.sh")

    # Prep bashfile
    with open(p2bash, "w") as f:
        f.write("#!/usr/bin/env bash\n\n")
        f.write(f"# Bashfile: {p2bash.split('/')[-1]}\n")

    for sub_hps_fn in os.listdir(p2hps):
        if not sub_hps_fn.startswith("S"):
            continue
        # Load hyperparameter (HP) table of each subject and take the n best HP sets
        sub_hps = pd.read_csv(os.path.join(p2hps, sub_hps_fn), sep=";")[:n_hps]
        sub_hps.rename(columns={"cond": "condition"}, inplace=True)

        # Fill HPs in bash string
        for i in range(n_hps):
            bash_str = "\npython3 NeVRolight.py"
            for hp in hps:
                if hp in ["sba", "subblock_cv", "permutation", "softmax"]:
                    bash_str += f" --{hp} {locals()[hp]}"

                elif hp == "n_components":
                    # Convert list of components into number of components
                    n_comps = ast.literal_eval(sub_hps.iloc[i]['component'])
                    n_comps = 1 if isinstance(n_comps, int) else len(n_comps)
                    bash_str += f" --{hp} {n_comps}"
                else:
                    bash_str += f" --{hp} {sub_hps.iloc[i][hp]}"

            # bash_str = f"python3 NeVRolight.py --subject 36 --condition nomov --sba True " \
            #            f"--task classification --s_fold 10 --balanced_cv True --subblock_cv False " \
            #            f"--repet_scalar 20 --batch_size 9 --permutation False --n_components 3 " \
            #            f"--learning_rate 0.001 --weight_reg l2 --weight_reg_strength 0.18 " \
            #            f"--activation_fct relu --lstm_size 20,15 --fc_n_hidden 10 --softmax False"

            with open(p2bash, "a") as f:
                f.write(bash_str)


def one_hot(y):
    """Create one-hot encoding of given label-data."""
    _one_hot = np.zeros(shape=(len(y), len(np.unique(y))))  # init
    for i, u in enumerate(np.unique(y)):
        _one_hot[:, i] = (y == u) * 1
    return _one_hot


def load_data(subject: int, n_comps: int, condition: str, sba: bool, task: str = "classification"):
    """
    Load subject data (EEG, ratings) in given movement condition.
    :param subject: subject number
    :param n_comps: number of EEG components
    :param condition: movement condition 'nomov' OR 'mov'
    :param sba: True: using data with break
    :param task: Decoding task "classification" OR "regression"
    :return: EEG, rating data
    """

    cl: bool = "class" in task.lower()

    p2eeg = os.path.join(p2nevro, f"Data/EEG/07_SSD/{condition}/{'SBA' if sba else 'SA'}/narrowband/")
    p2rating = os.path.join(
        p2nevro, f"Data/ratings/{'class_bins' if cl else 'continuous/z_scored'}/{condition}/"
                 f"{'SBA' if sba else 'SA'}/")
    eeg_fn = [fn for fn in os.listdir(p2eeg) if s(subject) in fn][0]
    rating_fn = [fn for fn in os.listdir(p2rating) if s(subject) in fn][0]

    # # Load EEG data of subject
    sampl_f: int = 250
    _eeg = np.genfromtxt(os.path.join(p2eeg, eeg_fn), dtype=np.float32, delimiter=",")
    # take first n components, cut off last n samples to balance number of samples per timestep
    surplus = _eeg.shape[1] % 270
    if surplus > 0:
        _eeg = _eeg[:n_comps, :-surplus]
    _eeg = np.array(np.array_split(_eeg, sampl_f, axis=1)).reshape((270, sampl_f, n_comps))

    # Load rating data of subject
    _rating = np.genfromtxt(os.path.join(p2rating, rating_fn), dtype=np.float32,
                            delimiter=",", skip_header=True if cl else False)[:, 1]  # shape: (270,)

    # Prep for binary classification
    if cl:
        _rating -= 2  # transform bins to: -1, 0, 1
        _eeg = _eeg[np.where(_rating != 0)[0]]
        _rating = _rating[np.where(_rating != 0)[0]]
        if FLAGS.softmax:
            _rating = one_hot(y=_rating)

    return _eeg, _rating


def balance_training_set(xtrain: np.ndarray, ytrain: np.ndarray):
    """
    Use SMOTE to balance training dataset with respect to classes.
    :param xtrain: model input data
    :param ytrain: ground truth class labels
    :return: balanced training set
    """

    sm = SMOTE(random_state=42)
    org_shape = xtrain.shape
    flat_shape = (xtrain.shape[0], xtrain.shape[1] * xtrain.shape[2])
    x_res, y_res = sm.fit_resample(xtrain.reshape(flat_shape), ytrain[:, 1] if FLAGS.softmax else ytrain)

    xtrain = x_res.reshape((-1, org_shape[1], org_shape[2]))
    ytrain = one_hot(y_res) if FLAGS.softmax else y_res

    return xtrain, ytrain


def tain_val_index_cv(s_fold_idx: int, n_sampl: int, s_fold: int = 10, shuffle_within: bool = True,
                      kind: str = "vanilla", balance_y: bool = False, y: np.ndarray = None):
    """
    Produce indices for stratified rolling S-fold cross-validation.

    kind: 'vanilla':
        balanced CV split

    kind: 'subblock':
        Full length:
        [0, 1, 2, 3, ..., N-2, N-1, N]
        Split in 3 main chunks:
        [[0, 1, 2, 3, ..., N/3], [N/3 + 1, N/3 + 2, ..., 2*N/3], [2*N/3 + 1, 2*N/3 + 2, ..., N]]
        Draw from each each m-samples for validation set:
        [0, 1, 2, ..., m, N/3 + 1, N/3 + 2, ..., N/3 + m+1, 2*N/3 + 1, 2*N/3 + 2, ..., 2*N/3 + m+1]]

    Rest is for the training set
    """

    kind = kind.lower()
    assert kind in ["vanilla", "subblock"], "kind must be 'vanilla' OR 'subblock'!"

    if kind == "subblock":
        if balance_y:
            print("'balance_y=True': For subblock CV balancing of classes must be done outside of "
                  "function via SMOTE!")

        _idx_val = [int(s_fold_idx * n_sampl / (3 * s_fold) + i_ + k_ * n_sampl / 3) for k_ in range(3)
                    for i_ in range(int(n_sampl / (3 * s_fold)))]
        _idx_train = list(set(range(n_sampl)) - set(_idx_val))

    else:  # kind == "vanilla":
        np.random.seed(42)
        idx_data = np.arange(n_sampl)
        np.random.shuffle(idx_data)

        if balance_y:
            assert y is not None, "y data must be given, if split shall be balanced!"
            _y = y.copy()
            _y = _y[idx_data]  # re-order y-data

            # Draw from each class an equal amount of samples
            _idx_val = None
            for u in np.unique(_y, axis=0):
                if FLAGS.softmax:
                    class_arr = np.array(np.array_split(idx_data[np.all(_y == u, 1)], s_fold))
                else:
                    class_arr = np.array(np.array_split(idx_data[_y == u], s_fold))

                _idx_val = class_arr[s_fold_idx] if _idx_val is None else np.append(_idx_val,
                                                                                    class_arr[s_fold_idx])
            _idx_train = np.array([e for e in idx_data if e not in _idx_val])
        else:
            idx_data = np.array(np.array_split(idx_data, s_fold))
            _idx_val = idx_data[s_fold_idx]
            _idx_train = np.delete(idx_data, s_fold_idx, axis=0).flatten()

    if shuffle_within:
        np.random.seed(42)  # that order remains same after recalling the function
        # np.random.shuffle(_idx_val)
        np.random.shuffle(_idx_train)

    return _idx_train, _idx_val


def create_model(n_comps: int, s_idx: int, lstm: tuple = (10, 10), fc: (tuple, int) = (10, 1),
                 activation: str = "relu", summary: bool = True):
    """
    Create (keras) LSTM model.

    :param n_comps: number of fed components
    :param s_idx: index of S-fold cross-validation
    :param lstm: size per LSTM LSTM layer
    :param fc: size per fully-connected layer
    :param activation: activation function between layers
    :param summary: print model summary
    :return: model
    """

    sampl_f: int = 250
    n_class: int = 2

    # For 2 class logit output for softmax loss
    fc = (fc,) if not isinstance(fc, tuple) else fc
    if fc[-1] != (n_class if FLAGS.softmax else 1):
        fc += ((n_class if FLAGS.softmax else 1),)

    # Build model
    inputs = keras.layers.Input(shape=(sampl_f, n_comps), name="Input", dtype=np.float32)

    hidden_inputs = None
    for li, lstm_size in enumerate(lstm):
        hidden_inputs = keras.layers.LSTM(
            units=lstm_size,
            return_sequences=True if (li+1) < len(lstm) else False,
            # False: only return the last hidden state output
            return_state=True, unroll=True, activation=activation, dtype=tf.float32, time_major=False)(
            inputs=inputs if hidden_inputs is None else hidden_inputs[0])

    hidden_inputs = hidden_inputs[0]
    # hidden_inputs = keras.layers.Flatten()(hidden_inputs[0])
    # hidden_inputs[0]: lstm output; hidden_inputs[1]: state_h; hidden_inputs[2]: state_c

    for fi, fc_size in enumerate(fc):
        hidden_inputs = keras.layers.Dense(
            units=fc_size, activation=activation if (fi + 1) < len(fc) else None,
            kernel_regularizer=wreg[FLAGS.weight_reg](FLAGS.weight_reg_strength),
            kernel_initializer="glorot_normal")(inputs=hidden_inputs)  # glorot_normal == xavier

    #
    if FLAGS.softmax:
        logits = keras.layers.Activation("linear")(hidden_inputs)
        # or use Activation("softmax")
    else:
        logits = keras.layers.Activation("tanh")(hidden_inputs)

    # Model build & summary
    _model = keras.Model(inputs=inputs, outputs=logits, name=f"NeVRoNet_fold-{s_idx}")
    if summary:
        print(_model.summary())

    if FLAGS.softmax:
        loss = keras.losses.BinaryCrossentropy(from_logits=True)
        metrics = [keras.metrics.CategoricalAccuracy()] if not FLAGS.subblock_cv else []
        # keras.metrics.AUC() causes issues
    else:
        loss = keras.losses.MeanSquaredError()
        metrics = []

    # Define loss and metrics and compile
    _model.compile(optimizer=keras.optimizers.Adam(learning_rate=FLAGS.learning_rate), loss=loss,
                   metrics=metrics)

    return _model


def print_flags():
    """Prints all entries in FLAGS variable."""
    for key, value in vars(FLAGS).items():
        cprint(key + ' : ' + str(value), "b")


def mdir():
    """Create path to model directory."""
    # t = str(datetime.now()).split(" ")[0]  # e.g. 2021-07-27
    fld = f"{'BiCl' if 'class' in FLAGS.task.lower() else 'Reg'}_{FLAGS.condition}_" \
          f"lstm-{FLAGS.lstm_size.replace(',', '-')}_fc-{FLAGS.fc_n_hidden.replace(',', '-')}_" \
          f"lr-{FLAGS.learning_rate:g}_wreg-{FLAGS.weight_reg}-{FLAGS.weight_reg_strength:.2f}_" \
          f"actfunc-{FLAGS.activation_fct}_comp-{FLAGS.n_components}_" \
          f"balcv-{'T' if FLAGS.balanced_cv else 'F'}_subcv-{'T' if FLAGS.subblock_cv else 'F'}"
    return os.path.join(os.getcwd(), "processed", FLAGS.condition, s(FLAGS.subject), fld)


def main():

    # Load performance table
    p2perform_tab = os.path.join(os.getcwd(), "processed", f"performance_table_{FLAGS.condition}.csv")
    if os.path.isfile(p2perform_tab):
        perform_tab = pd.read_csv(filepath_or_buffer=p2perform_tab, sep=";")
    else:
        perform_tab = pd.DataFrame(data=None,
                                   columns=[key for key, _ in vars(FLAGS).items()] + ["acc", "auc"])

    # Check whether already computed
    if any([bash_line_from(row) == bash_line_from(None) for _, row in perform_tab.iterrows()]):
        cprint("Model for given hyperparameters and subject was trained already!", col='y')
        # Update bashfiles
        for r, row in perform_tab.iterrows():
            update_bashfiles(source=row)
        return None

    # Load data
    eeg, rating = load_data(subject=FLAGS.subject, n_comps=FLAGS.n_components, condition=FLAGS.condition,
                            sba=FLAGS.sba, task=FLAGS.task)

    # Prep data
    accuracies = []
    aucs = []
    start_time = datetime.now()
    for sidx in range(FLAGS.s_fold):
        idx_train, idx_val = tain_val_index_cv(s_fold_idx=sidx, n_sampl=eeg.shape[0], s_fold=FLAGS.s_fold,
                                               kind="subblock" if FLAGS.subblock_cv else 'vanilla',
                                               balance_y=FLAGS.balanced_cv, y=rating)

        x_train = eeg[idx_train]
        y_train = rating[idx_train]
        x_val = eeg[idx_val]
        y_val = rating[idx_val]

        # To account for unbalanced classes
        if FLAGS.subblock_cv and FLAGS.balanced_cv:
            x_train, y_train = balance_training_set(xtrain=x_train, ytrain=y_train)

        # Create or load trained model
        model_name = f"NeVRo_model_{s(FLAGS.subject)}_{FLAGS.condition}_" \
            f"{'subblock' if FLAGS.subblock_cv else 'vanilla'}CV{'_soft' if FLAGS.softmax else''}" \
            f"_fold-{sidx}"
        p2model = os.path.join(mdir(), "models", f"{model_name}.h5")

        if os.path.isfile(p2model):
            model = keras.models.load_model(filepath=p2model)
        else:
            model = create_model(n_comps=FLAGS.n_components, s_idx=sidx,
                                 lstm=ast.literal_eval(FLAGS.lstm_size),
                                 fc=ast.literal_eval(FLAGS.fc_n_hidden), summary=sidx == 0)

            # # Fit model
            model.fit(x=x_train, y=y_train,
                      batch_size=FLAGS.batch_size, epochs=FLAGS.repet_scalar,
                      validation_data=(x_val, y_val),
                      validation_freq=max(FLAGS.repet_scalar//10, 1),
                      use_multiprocessing=True, workers=os.cpu_count()//2,
                      callbacks=[keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                                                   patience=7, verbose=1, min_delta=1e-4,
                                                                   mode='min'),
                                 keras.callbacks.EarlyStopping(monitor='val_loss',
                                                               patience=FLAGS.repet_scalar//5,
                                                               verbose=0, mode='min'),
                                 keras.callbacks.ModelCheckpoint(filepath=p2model, save_best_only=True,
                                                                 monitor='val_loss', mode='min')])

        # Evaluate
        predictions = model.predict(x=x_val)

        auc = roc_auc_score(y_true=y_val, y_score=predictions)  # AUC
        if FLAGS.softmax:
            accuracy = np.sum(predictions.argmax(1) == y_val.argmax(1)) / len(predictions)  # accuracy
        else:
            accuracy = np.sum(np.sign(predictions.ravel()) == y_val) / len(predictions)

        print(f"{model_name} performance:\n\t• Acc = {accuracy:.3f}\n\t• AUC = {auc:.3f}")

        aucs.append(auc)
        accuracies.append(accuracy)

        # Save prediction - y_true pairs for each fold
        traindf = pd.DataFrame(data=None, columns=["train_y", "train_pred"])
        traindf["train_y"] = y_train[:, 1].astype(int) if FLAGS.softmax else y_train.astype(int)
        traindf["train_pred"] = model.predict(x=x_train).argmax(1) if FLAGS.softmax else \
            np.sign(model.predict(x=x_train)).astype(int)
        valdf = pd.DataFrame(data=None, columns=["val_y", "val_pred"])
        valdf["val_y"] = y_val[:, 1].astype(int) if FLAGS.softmax else y_val.astype(int)
        valdf["val_pred"] = predictions.argmax(1) if FLAGS.softmax else np.sign(predictions).astype(int)
        os.makedirs(os.path.join(mdir(), "predictions"), exist_ok=True)
        traindf.to_csv(p2model.replace('models', 'predictions').replace('.h5', '_train.csv'))
        valdf.to_csv(p2model.replace('models', 'predictions').replace('.h5', '_val.csv'))

        # Time-it
        loop_timer(start_time=start_time, loop_length=FLAGS.s_fold, loop_idx=sidx,
                   loop_name=f"{model_name.split('_fold')[0]}", add_daytime=True)

    cprint(f"\nAverage performance across CV folds:\n\t"
           f"• Acc = {np.mean(accuracies):.3f}\n\t"
           f"• AUC = {np.mean(aucs):.3f}\n", col='b', fm='bo')

    # Save in table
    perform_tab = perform_tab.append(dict(zip(perform_tab.columns,
                                              list(vars(FLAGS).values()) + [np.mean(accuracies),
                                                                            np.mean(aucs)])),
                                     ignore_index=True, verify_integrity=True)

    perform_tab = perform_tab[~perform_tab.duplicated(keep="first")]
    perform_tab.to_csv(p2perform_tab, sep=";", index=False)

    # Update bashfiles
    update_bashfiles()


# %% Main >> o <<<< o >>>> o <<<< o >>>> o <<<< o >>>> o <<<< o >>>> o <<<< o >>>> o <<<< o >>>> o <<<< o

if __name__ == "__main__":

    # Parser setup << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<
    parser = argparse.ArgumentParser()

    parser.add_argument('--subject', type=int, default=36, help='Which subject data to process')
    parser.add_argument('--condition', type=str, default="nomov",
                        help="Which condition: 'nomov' (no movement) or 'mov'")
    parser.add_argument('--sba', type=str2bool, default=True, help="True for SBA; False for SA")
    parser.add_argument('--task', type=str, default="classification",
                        help="Either 'classification' or 'regression'")
    parser.add_argument('--s_fold', type=int, default=10,
                        help='Number of folds in S-Fold-Cross Validation')
    parser.add_argument('--balanced_cv', type=str2bool, default=True,
                        help='Balanced CV. False: at each iteration/fold data gets shuffled '
                             '(semi-balanced, this can lead to overlapping samples in validation set)')
    parser.add_argument('--subblock_cv', type=str2bool, default=False,
                        help='Stratified rolling 10-fold CV. True: CV is stratified by sub-blocks. Each '
                             'fold consists of equal parts from each sub-bock. ')
    parser.add_argument('--repet_scalar', type=int, default=20,
                        help='Number of times it should run through set. repet_scalar*(270 - 270/s_fold)')
    parser.add_argument('--batch_size', type=int, default=9, help='Batch size to run trainer.')
    parser.add_argument('--permutation', type=str2bool, default=False,
                        help="Compute block-permutation (n=1000).")
    parser.add_argument('--n_components', type=int, default=3,
                        help="How many components are to be fed to model")
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_reg', type=str, default="l2",
                        help='Regularizer type for weights of fully-connected layers [none, l1, l2].')
    parser.add_argument('--weight_reg_strength', type=float, default=0.18,
                        help='Regularizer strength for weights of fully-connected layers.')
    parser.add_argument('--activation_fct', type=str, default="relu",
                        help='Type of activation function from LSTM to fully-connected layers: elu, relu')
    parser.add_argument('--lstm_size', type=str, default="20,15",
                        help='Comma separated list of size of hidden states in each LSTM layer')
    parser.add_argument('--fc_n_hidden', type=str, default="10",
                        help="Comma separated list of N of hidden units in each FC layer")
    parser.add_argument('--softmax', type=str2bool, default=False,
                        help="Use softmax for classification, else tanh.")

    FLAGS, unparsed = parser.parse_known_args()
    assert list(vars(FLAGS).keys()) == hps, "FLAGS must match hps (also in order)!"

    # Run main << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>
    if create_bash:
        cprint("Writing bashfiles ...", col='b')

        for cond in ["nomov", "mov"]:
            for sbcv in [True, False]:
                # TODO train SA case, too
                create_training_bash_file(condition=cond, sba=True, subblock_cv=sbcv, permutation=False,
                                          softmax=False, task="classification", n_hps=2)

    else:
        print_flags()
        main()
    end()

    # TODO 1000 permutation 10 blocks, shuffle blocks (of ratings) and feed through same pipeline

# << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< END

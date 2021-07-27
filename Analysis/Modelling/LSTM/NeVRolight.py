# coding=utf-8
"""
Main script (light)
• Light-weight version of NeVRo.py.
• Compatible with tensorflow >= 2.

Author: Simon M. Hofmann | <[surname].[lastname][at]pm.me> | 2021
"""

# % Import
import ast
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.metrics import roc_auc_score
# import matplotlib.pyplot as plt

try:
    from imblearn.over_sampling import SMOTE   # scikit package
except ModuleNotFoundError:
    print("imblearn must be installed with scikit-learn, but currently not working together with tf via "
          "conda!")

from utils import *

# % Set global vars & paths >> o <<<< o >>>> o <<<< o >>>> o <<<< o >>>> o <<<< o >>>> o <<<< o >>>> o <<<
setwd("Modelling/LSTM")
p2nevro = os.getcwd()[:os.getcwd().find("NeVRo")+5]

wreg = {"l1": keras.regularizers.l1,
        "l2": keras.regularizers.l2}


# % Functions >> o <<<< o >>>> o <<<< o >>>> o <<<< o >>>> o <<<< o >>>> o <<<< o >>>> o <<<< o >>>> o <<<

def one_hot(y):
    _one_hot = np.zeros(shape=(len(y), len(np.unique(y))))  # init
    for i, u in enumerate(np.unique(y)):
        _one_hot[:, i] = (y == u) * 1
    return _one_hot


def load_data(subject: int, n_comps: int, condition: str, sba: bool, task: str = "classification"):

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


def balance_training_set(xtrain, ytrain):

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


def create_model(n_comps: int, s_idx: int, lstm=(10, 10), fc=(10, 1), activation="relu"):

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

    # TODO change back to tanh
    if FLAGS.softmax:
        logits = keras.layers.Activation("softmax" if FLAGS.subblock_cv else "linear")(hidden_inputs)
        # TODO for AUC the output must be > 0 & class prediction is argmax([a,b]), results remain the same
    else:
        logits = keras.layers.Activation("tanh")(hidden_inputs)

    # Model build & summary
    _model = keras.Model(inputs=inputs, outputs=logits, name=f"NeVRoNet_fold-{s_idx}")
    print(_model.summary())

    if FLAGS.softmax:
        loss = keras.losses.BinaryCrossentropy(from_logits=not FLAGS.subblock_cv)
        metrics = [keras.metrics.CategoricalAccuracy()] if not FLAGS.subblock_cv else []
    else:
        loss = keras.losses.MeanSquaredError()
        metrics = []

    # Define loss and metrics and compile
    _model.compile(optimizer=keras.optimizers.Adam(learning_rate=FLAGS.learning_rate), loss=loss,
                   metrics=metrics)

    # keras.metrics.AUC(from_logits=not FLAGS.subblock_cv)
    # got an unexpected keyword argument 'from_logits' ?

    return _model


def print_flags():
    """Prints all entries in FLAGS variable."""
    for key, value in vars(FLAGS).items():
        cprint(key + ' : ' + str(value), "b")


def mdir():
    """Path to model dir."""
    # t = str(datetime.now()).split(" ")[0]  # e.g. 2021-07-27
    fld = f"{'BiCl' if 'class' in FLAGS.task.lower() else 'Reg'}_{FLAGS.condition}_" \
          f"lstm-{FLAGS.lstm_size.replace(',', '-')}_fc-{FLAGS.fc_n_hidden.replace(',', '-')}_" \
          f"lr-{FLAGS.learning_rate:g}_wreg-{FLAGS.weight_reg}-{FLAGS.weight_reg_strength:.2f}_" \
          f"actfunc-{FLAGS.activation_fct}_comp-{FLAGS.n_components}_" \
          f"balcv-{'T' if FLAGS.balanced_cv else 'F'}_subcv-{'T' if FLAGS.subblock_cv else 'F'}"
    return os.path.join(os.getcwd(), "processed", FLAGS.condition, s(FLAGS.subject), fld)


def main():

    # Load data
    eeg, rating = load_data(subject=FLAGS.subject, n_comps=FLAGS.n_components, condition=FLAGS.condition,
                            sba=FLAGS.sba, task=FLAGS.task)

    # Prep data
    performs = []
    start_time = datetime.now()
    for sidx in range(FLAGS.s_fold):
        idx_train, idx_val = tain_val_index_cv(s_fold_idx=sidx, n_sampl=eeg.shape[0], s_fold=FLAGS.s_fold,
                                               kind="subblock" if FLAGS.subblock_cv else 'vanilla',
                                               balance_y=not FLAGS.subblock_cv, y=rating)

        x_train = eeg[idx_train]
        y_train = rating[idx_train]
        x_val = eeg[idx_val]
        y_val = rating[idx_val]

        # To account for unbalanced classes
        if FLAGS.subblock_cv:
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
                                 fc=ast.literal_eval(FLAGS.fc_n_hidden))

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

        if FLAGS.subblock_cv:
            perform = roc_auc_score(y_true=y_val, y_score=predictions)  # AUC
        else:
            if FLAGS.softmax:
                perform = np.sum(predictions.argmax(1) == y_val.argmax(1)) / len(predictions)  # accuracy
            else:
                perform = np.sum(np.sign(predictions.ravel()) == y_val) / len(predictions)
        print(f"Fold {sidx}: Model {'AUC' if FLAGS.subblock_cv else 'accuracy'} = {perform:.3f}")

        performs.append(perform)

        # Save prediction - y_true pairs for each fold
        traindf = pd.DataFrame(data=None, columns=["train_y", "train_pred"])
        traindf["train_y"] = y_train[:, 1].astype(int) if FLAGS.softmax else y_train.astype(int)
        traindf["train_pred"] = model.predict(x=x_train).argmax(1)
        valdf = pd.DataFrame(data=None, columns=["val_y", "val_pred"])
        valdf["val_y"] = y_val[:, 1].astype(int) if FLAGS.softmax else y_val.astype(int)
        valdf["val_pred"] = predictions.argmax(1)
        os.makedirs(os.path.join(mdir(), "predictions"), exist_ok=True)
        traindf.to_csv(p2model.replace('models', 'predictions').replace('.h5', '_train.csv'))
        valdf.to_csv(p2model.replace('models', 'predictions').replace('.h5', '_val.csv'))

        # Time-it
        loop_timer(start_time=start_time, loop_length=FLAGS.s_fold, loop_idx=sidx,
                   loop_name=f"{model_name.split('_fold')[0]}", add_daytime=True)

    cprint(f"\nAverage performance across CV folds: {'AUC' if FLAGS.subblock_cv else 'accuracy'} = "
           f"{np.mean(performs):.3f}", col='b', fm='bo')


# % Main >> o <<<< o >>>> o <<<< o >>>> o <<<< o >>>> o <<<< o >>>> o <<<< o >>>> o <<<< o >>>> o <<<< o >

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--sba', type=str2bool, default=True,
                        help="True for SBA; False for SA")
    parser.add_argument('--task', type=str, default="classification",
                        help="Either 'classification' or 'regression'")
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--repet_scalar', type=int, default=20,
                        help='Number of times it should run through set. repet_scalar*(270 - 270/s_fold)')
    parser.add_argument('--batch_size', type=int, default=9,
                        help='Batch size to run trainer.')
    parser.add_argument('--weight_reg', type=str, default="l2",
                        help='Regularizer type for weights of fully-connected layers [none, l1, l2].')
    parser.add_argument('--weight_reg_strength', type=float, default=0.18,
                        help='Regularizer strength for weights of fully-connected layers.')
    parser.add_argument('--activation_fct', type=str, default="relu",
                        help='Type of activation function from LSTM to fully-connected layers: elu, relu')
    parser.add_argument('--subject', type=int, default=36,
                        help='Which subject data to process')
    parser.add_argument('--condition', type=str, default="nomov",
                        help="Which condition: 'nomov' (no movement) or 'mov'")
    parser.add_argument('--lstm_size', type=str, default="20,15",
                        help='Comma separated list of size of hidden states in each LSTM layer')
    parser.add_argument('--balanced_cv', type=str2bool, default=True,
                        help='Balanced CV. False: at each iteration/fold data gets shuffled '
                             '(semi-balanced, this can lead to overlapping samples in validation set)')
    parser.add_argument('--subblock_cv', type=str2bool, default=False,
                        help='Stratified rolling 10-fold CV. True: CV is stratified by sub-blocks. Each '
                             'fold consists of equal parts from each sub-bock. ')
    parser.add_argument('--s_fold', type=int, default=10,
                        help='Number of folds in S-Fold-Cross Validation')
    parser.add_argument('--fc_n_hidden', type=str, default="10",
                        help="Comma separated list of N of hidden units in each FC layer")
    parser.add_argument('--n_components', type=int, default=3,
                        help="How many components are to be fed to model")
    parser.add_argument('--permutation', type=str2bool, default=False,
                        help="Compute block-permutation (n=1000).")
    parser.add_argument('--softmax', type=str2bool, default=False,
                        help="Use softmax for classification, else tanh.")

    FLAGS, unparsed = parser.parse_known_args()
    print_flags()

    main()
    end()

    # TODO 1000 permutation 10 blocks, shuffle blocks (of ratings) and feed through same pipeline

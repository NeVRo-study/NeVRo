# coding=utf-8
"""
Main script
    - run model
    - script can be directly called (use/adapt DEFAULT values)
    - Analogue to NeVRo.py

Author: Simon Hofmann | <[surname].[lastname][at]protonmail.com> | 2017, 2019 (Update)
"""

# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from load_data import *

import tensorflow as tf  # implemented with TensorFlow 1.13.1
import time
import copy
import subprocess

from NeVRoNet import NeVRoNet

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Makes TensorFlow less verbose, comment out for debugging

# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

# Define Default Values for FLAGS.xx
TASK_DEFAULT = 'regression'  # prediction via 'regression' (continious) or 'classification' (Low-High)
LEARNING_RATE_DEFAULT = 1e-3  # 1e-4
BATCH_SIZE_DEFAULT = 9  # or bigger, batch_size must be a multiple of 'successive batches'
SUCCESSIVE_BATCHES_DEFAULT = 1  # (time-)length per sample is hyperparameter in form of successive batches
SUCCESSIVE_MODE_DEFAULT = 1  # either 1 or 2
S_FOLD_DEFAULT = 10
REPETITION_SCALAR_DEFAULT = 2  # scaler for how many times it runs through set (can be also fraction)
OPTIMIZER_DEFAULT = 'ADAM'
WEIGHT_REGULARIZER_DEFAULT = 'l2'
WEIGHT_REGULARIZER_STRENGTH_DEFAULT = 0.18
ACTIVATION_FCT_DEFAULT = 'elu'
LOSS_DEFAULT = "normal"  # is not used yet
LSTM_SIZE_DEFAULT = '100'  # N of hidden units per LSTM layer, e.g., '10,5' would create second lstm_layer
FC_NUM_HIDDEN_UNITS = None  # if len(n_hidden_units)>0, create len(n_hidden_units) layers
FILE_TYPE_DEFAULT = "SSD"  # Either 'SSD' or 'SPOC'
COMPONENT_DEFAULT = "1,2,3"
# 'best', 'noise', 'random', 'all' or list of 1 or more comps (1-5), e.g. '1,3,5' or '4'

SUBJECT_DEFAULT = 36
CONDITION_DEFAULT = "nomov"

# Default values below (True, False) only needed for DEBUG script
SBA_DEFAULT = True  # if False: SA
SHUFFLE_DEFAULT = False  # True for 'classification'
RANDOM_BATCH_DEFAULT = True
HILBERT_POWER_INPUT_DEFAULT = True
BAND_PASS_INPUT_DEFAULT = True
SUMMARIES_DEFAULT = False  # or False
PLOT_DEFAULT = True
DEL_LOG_DEFAULT = False
SEED = True  # seed on/off for random
HR_COMPONENT_DEFAULT = True  # Whether to attach the heart-rate (HR) vector to neural components
EQCOMPMAT_DEFAULT = None  # If input matrix should be equal in size (int) across all tested conditions
TESTMODEL = False
# '--is_train', default=True,

PATH_SPECIFICITIES_DEFAULT = "debug/"  # dont change
SUB_DIR_DEFAULT = "./LSTM/{1}/{0}".format(PATH_SPECIFICITIES_DEFAULT, s(SUBJECT_DEFAULT))
LOG_DIR_DEFAULT = "./LSTM/logs/{1}/{0}".format(PATH_SPECIFICITIES_DEFAULT, s(SUBJECT_DEFAULT))
CHECKPOINT_DIR_DEFAULT = "./LSTM/checkpoints/{1}/{0}".format(PATH_SPECIFICITIES_DEFAULT,
                                                             s(SUBJECT_DEFAULT))

# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

WEIGHT_REGULARIZER_DICT = {'none': lambda x: None,  # No regularization
                           # L1 regularization
                           'l1': tf.keras.regularizers.l1,  # tf.contrib.layers.l1_regularizer,
                           # L2 regularization
                           'l2': tf.keras.regularizers.l2}   # tf.contrib.layers.l2_regularizer

ACTIVATION_FCT_DICT = {'elu': tf.nn.elu,
                       'relu': tf.nn.relu}

# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>


def initialize_folders():
    """
    Initializes all folders in FLAGS variable.
    """
    # This way, it does not create subfolder
    # if not os.path.exists(SUB_DIR_DEFAULT):
    #     os.mkdir(SUB_DIR_DEFAULT)

    if not tf.gfile.Exists(SUB_DIR_DEFAULT):
        tf.gfile.MakeDirs(SUB_DIR_DEFAULT)

    if not tf.gfile.Exists(LOG_DIR_DEFAULT):
        tf.gfile.MakeDirs(LOG_DIR_DEFAULT)

    if not tf.gfile.Exists(CHECKPOINT_DIR_DEFAULT):
        tf.gfile.MakeDirs(CHECKPOINT_DIR_DEFAULT)


def train_step(loss):
    """
    Defines the ops to conduct an optimization step. Optional: Implement Learning
    rate scheduler or pick optimizer here.

    Args:
        loss: scalar float Tensor, full loss = mean squared error + reg_loss

    Returns:
        train_op: Ops for optimization.
    """

    train_op = None
    # could do: Learning rate scheduler
    if OPTIMIZER_DEFAULT == 'ADAM':
        train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE_DEFAULT).minimize(loss)

    return train_op


def fill_pred_matrix(pred, y, current_mat, s_idx, current_batch, sfold, train=True):
    """
    Updates prediction matrix with current prediction and ground truth (rating)
    :param pred: prediction
    :param y: rating
    :param current_mat: prediction matrix (S-Folds x 270)
    :param s_idx: indicates which fold is validation set
    :param current_batch: current_batch in the training or val set
    :param sfold: Number of folds
    :param train: whether training set or validation set
    :return: updated prediction matrix
    """

    # Need to be integer
    current_batch = [int(s_tep) for s_tep in current_batch]

    # Reshape prediction and target
    pred = pred.reshape(pred.shape[0])
    y = y.reshape(y.shape[0])

    # Set Index
    pred_idx = int(s_idx * 2)
    rat_idx = int(pred_idx + 1)

    fulllength = len(current_mat[0, :])  # 270

    # S-Fold, for S=10:
    #    FOLD 0          FOLD 1          FOLD 2                FOLD 9
    # [0, ..., 26] | [27, ..., 53] | [54, ..., 80] | ... | [235, ..., 269]
    fold_length = int(fulllength/sfold)  # len([0, ..., 26]) = 27

    # S-Fold-Index, e.g. s_idx=1. That is, FOLD 1 is validation set
    verge = s_idx * fold_length  # verge = 27

    if train:
        # if step=27 => fill_pos=54 (in FOLD 2)
        fill_pos = [step if step < verge else step + fold_length for step in current_batch]

        # fill_pos = step if step < verge else step + fold_length
        # if not np.isnan(current_mat[pred_idx, fill_pos]):
        #     print("Overwrites position in pred_matrix")

    else:  # case of validation (use: batch_size=1, but works also with >1)
        fill_pos = [step + int(verge) for step in current_batch]
        # Check whether position is already filled
        for pos in fill_pos:
            if not np.isnan(current_mat[pred_idx, pos]):
                print("Overwrites position ({},{}) in val_pred_matrix".format(pred_idx, pos))

    # Update Matrix
    current_mat[pred_idx, fill_pos] = pred  # inference/prediction
    current_mat[rat_idx, fill_pos] = y      # ratings
    updated_mat = current_mat

    # Prints only for testing
    # print("{}: Current batch is: {}".format("Training" if train else "Validation", current_batch))
    # print("{}: Current verge is: {}".format("Training" if train else "Validation", verge))
    # print("{}: Current fold_length is: {}".format("Training" if train else "Validation", fold_length))
    # print("{}: Current position is: {}".format("Training" if train else "Validation", fill_pos))

    return updated_mat


# # Parameters
# runs max_steps-times through set
max_steps = REPETITION_SCALAR_DEFAULT*(270 - 270/S_FOLD_DEFAULT)/BATCH_SIZE_DEFAULT
assert float(max_steps).is_integer(), "max steps must be integer"
cprint("max_steps: {}".format(max_steps), "r")  # TESTING
eval_freq = 5
cprint("eval_freq: {}".format(eval_freq), "r")  # TESTING
checkpoint_freq = int(max_steps)
cprint("checkpoint_freq: {}".format(checkpoint_freq), "r")  # TESTING
# print_freq = int(max_steps/8)  # every 30th step for test or int(max_steps/10). Too low:uses much memory
print_freq = 6  # TESTING
cprint("print_freq: {}".format(print_freq), "r")  # TESTING
assert BATCH_SIZE_DEFAULT % SUCCESSIVE_BATCHES_DEFAULT == 0, \
    "BATCH_SIZE must be a multiple of SUCCESSIVE_BATCHES"
assert TASK_DEFAULT.upper() in ['REGRESSION', 'CLASSIFICATION'], "Prediction task is undefined."

# Set the random seeds on True for reproducibility.
if SEED:
    tf.set_random_seed(42)
    np.random.seed(42)
    print("Seed is turned on")

# Get number of units in each hidden layer specified in the string such as 100,100
# For LSTM layers
if LSTM_SIZE_DEFAULT:
    lstm_hidden_states = LSTM_SIZE_DEFAULT.split(",")
    lstm_hidden_states = [int(hidden_states_) for hidden_states_ in lstm_hidden_states]
else:
    lstm_hidden_states = []

# For fully connected layers
if FC_NUM_HIDDEN_UNITS and len(FC_NUM_HIDDEN_UNITS) > 0 and FC_NUM_HIDDEN_UNITS not in "0":
    n_hidden_units = FC_NUM_HIDDEN_UNITS.split(",")
    n_hidden_units = [int(hidden_unites_) for hidden_unites_ in n_hidden_units]
    n_hidden_units.append(1)  # output layer == 1 rating-prediction

else:
    n_hidden_units = [1]

# Init Folders
initialize_folders()

# Start TensorFlow Interactive Session
# sess = tf.InteractiveSession()

# s_fold_idx depending on s_fold and previous index
s_fold_idx_list = np.arange(S_FOLD_DEFAULT)
np.random.shuffle(s_fold_idx_list)
# rnd = 0

# Create graph_dic
graph_dict = dict.fromkeys(s_fold_idx_list)
for key in graph_dict.keys():
    graph_dict[key] = tf.Graph()

# Create to save the performance for each validation set
# all_acc_val = tf.Variable(tf.zeros(shape=S_FOLD_DEFAULT, dtype=tf.float32, name="all_valid_accuracies"))
all_acc_val = np.zeros(S_FOLD_DEFAULT)

# Choose component
input_component = None  # initialize

# First find best component
if COMPONENT_DEFAULT in ['best', 'noise']:
    best_comp = best_or_random_component(subject=SUBJECT_DEFAULT, condition=CONDITION_DEFAULT,
                                         f_type=FILE_TYPE_DEFAULT,
                                         best=True,
                                         sba=SBA_DEFAULT)

if not COMPONENT_DEFAULT.split(",")[0].isnumeric():

    assert COMPONENT_DEFAULT in ["best", "noise", "random", "all"], \
        "Component must be either 'best', 'noise', 'random', or 'all'"

    if COMPONENT_DEFAULT == "best":
        input_component = best_comp
    elif COMPONENT_DEFAULT == "noise":
        input_component = 90 + best_comp  # coding for noise component
    elif COMPONENT_DEFAULT == "random":
        input_component = best_or_random_component(subject=SUBJECT_DEFAULT,
                                                   condition=CONDITION_DEFAULT,
                                                   f_type=FILE_TYPE_DEFAULT,
                                                   best=False,  # Finds random component != best
                                                   sba=SBA_DEFAULT)
    elif COMPONENT_DEFAULT == "all":
        n_comp = get_num_components(subject=SUBJECT_DEFAULT, condition=CONDITION_DEFAULT,
                                    filetype=FILE_TYPE_DEFAULT, sba=SBA_DEFAULT)

        input_component = list(range(1, n_comp + 1))

else:  # given components are in form of list
    assert np.all([comp.isnumeric() for comp in COMPONENT_DEFAULT.split(",")]), \
        "All given components must be numeric"
    input_component = [int(comp) for comp in COMPONENT_DEFAULT.split(",")]

print("LSTM model get trained on input_component(s):", input_component)

# Load first data-set for preparation
nevro_data = get_nevro_data(subject=SUBJECT_DEFAULT,
                            task=TASK_DEFAULT,
                            cond=CONDITION_DEFAULT,
                            component=input_component,
                            hr_component=HR_COMPONENT_DEFAULT,
                            filetype=FILE_TYPE_DEFAULT,
                            hilbert_power=HILBERT_POWER_INPUT_DEFAULT,
                            band_pass=BAND_PASS_INPUT_DEFAULT,
                            equal_comp_matrix=EQCOMPMAT_DEFAULT,
                            s_fold_idx=s_fold_idx_list[0],
                            s_fold=S_FOLD_DEFAULT,
                            sba=SBA_DEFAULT,
                            shuffle=SHUFFLE_DEFAULT,
                            testmode=TESTMODEL)

mean_line_acc = mean_line_prediction(subject=SUBJECT_DEFAULT, condition=CONDITION_DEFAULT,
                                     sba=SBA_DEFAULT)

# Define graph using class NeVRoNet and its methods:
ddims = list(nevro_data["train"].eeg.shape[1:])  # [250, n_comp]
full_length = len(nevro_data["validation"].ratings) * S_FOLD_DEFAULT

# Prediction Matrix for each fold
pred_matrix = np.zeros(shape=(S_FOLD_DEFAULT*2, full_length), dtype=np.float32)
pred_matrix[np.where(pred_matrix == 0)] = np.nan  # set to NaN values
# We create a separate matrix for the validation (which can be merged with the first pred_matrix later)
val_pred_matrix = copy.copy(pred_matrix)

# In case data was shuffled save corresponding order per fold in matrix and save later
shuffle_order_matrix = np.zeros(shape=(S_FOLD_DEFAULT, pred_matrix.shape[1]))
shuffle_order_matrix[s_fold_idx_list[0], :] = nevro_data["order"]

# Set Variables for timer
timer_fold_list = []  # list of duration(time) of each fold
duration_fold = []  # duration of 1 fold
# rest_duration = 0

# Run through S-Fold-Cross-Validation (take the mean-performance across all validation sets)
for rnd, s_fold_idx in enumerate(s_fold_idx_list):
    cprint("Train now on Fold-Nr.{} (fold {}/{}) | {} | {}".format(s_fold_idx, rnd + 1,
                                                                   len(s_fold_idx_list),
                                                                   PATH_SPECIFICITIES_DEFAULT[:-1],
                                                                   s(SUBJECT_DEFAULT)),
           "b")
    start_timer_fold = datetime.datetime.now().replace(microsecond=0)

    with tf.Session(graph=graph_dict[s_fold_idx]) as sess:
        # This is a way to re-initialise the model completely
        # Alternative just reset the weights after one round (maybe with tf.reset_default_graph())
        with tf.variable_scope(name_or_scope="Fold_Nr{}/{}".format(str(rnd).zfill(len(str(
                S_FOLD_DEFAULT))), len(s_fold_idx_list))):

            # Load Data:
            if rnd > 0:

                # Show time passed per fold and estimation of rest time
                cprint("Duration of previous fold {} [h:m:s] | {} | {}".format(
                    duration_fold, PATH_SPECIFICITIES_DEFAULT[:-1], s(SUBJECT_DEFAULT)), "y")
                timer_fold_list.append(duration_fold)
                # average over previous folds (np.mean(timer_fold_list) not possible in python2)
                rest_duration_fold = average_time(timer_fold_list,
                                                  in_timedelta=True) * (S_FOLD_DEFAULT - rnd)
                rest_duration_fold = chop_microseconds(delta=rest_duration_fold)
                cprint("Estimated time to train rest {} fold(s): {} [h:m:s] | {} | {}\n".format(
                    S_FOLD_DEFAULT - rnd, rest_duration_fold, PATH_SPECIFICITIES_DEFAULT[:-1],
                    s(SUBJECT_DEFAULT)), "y")

                nevro_data = get_nevro_data(subject=SUBJECT_DEFAULT,
                                            component=input_component,
                                            hr_component=HR_COMPONENT_DEFAULT,
                                            s_fold_idx=s_fold_idx_list[rnd],
                                            s_fold=S_FOLD_DEFAULT,
                                            cond=CONDITION_DEFAULT,
                                            sba=SBA_DEFAULT,
                                            filetype=FILE_TYPE_DEFAULT,
                                            band_pass=BAND_PASS_INPUT_DEFAULT,
                                            equal_comp_matrix=EQCOMPMAT_DEFAULT,
                                            hilbert_power=HILBERT_POWER_INPUT_DEFAULT,
                                            task=TASK_DEFAULT,
                                            shuffle=SHUFFLE_DEFAULT,
                                            testmode=TESTMODEL)

                # Save order in shuffle_order_matrix: shuffle=False:(1,2,...,270);
                # shuffle=True:(56,4,...,173)
                shuffle_order_matrix[s_fold_idx, :] = nevro_data["order"]

            cprint("I am here before placeholder", "r")  # TESTING

            with tf.name_scope("input"):
                # shape = [None] + ddims includes num_steps = 250
                #  Tensorflow requires input as a tensor (a Tensorflow variable) of the dimensions
                # [batch_size, sequence_length, input_dimension] (a 3d variable).
                x = tf.placeholder(dtype=tf.float32, shape=[None] + ddims, name="x-input")
                # None for Batch-Size
                # x = tf.placeholder(dtype=tf.float32, shape=[None, 250, 2], name="x-input")
                y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="y-input")

            # Model
            lstm_model = NeVRoNet(activation_function=ACTIVATION_FCT_DICT.get(ACTIVATION_FCT_DEFAULT),
                                  weight_regularizer=WEIGHT_REGULARIZER_DICT.get(
                                      WEIGHT_REGULARIZER_DEFAULT)(WEIGHT_REGULARIZER_STRENGTH_DEFAULT),
                                  lstm_size=lstm_hidden_states, fc_hidden_unites=n_hidden_units,
                                  n_steps=ddims[0], batch_size=BATCH_SIZE_DEFAULT,
                                  summaries=SUMMARIES_DEFAULT)  # n_step = 250

            cprint("I am here before infer", "r")  # TESTING

            # Prediction
            with tf.variable_scope(name_or_scope="Inference"):
                infer = lstm_model.inference(x=x)

            cprint("I am here before loss", "r")  # TESTING
            # Loss
            loss = lstm_model.loss(infer=infer, ratings=y)

            # Define train_step, savers, summarizers
            # Optimization
            optimization = train_step(loss=loss)

            # Performance
            accuracy = lstm_model.accuracy(infer=infer, ratings=y)

            # Writer
            merged = tf.summary.merge_all()

            train_writer = tf.summary.FileWriter(logdir=LOG_DIR_DEFAULT + str(s_fold_idx) + "/train",
                                                 graph=sess.graph)
            test_writer = tf.summary.FileWriter(logdir=LOG_DIR_DEFAULT + str(s_fold_idx) + "/val")
            # test_writer = tf.train.SummaryWriter(logdir=LOG_DIR_DEFAULT + "/test")

            # Saver
            # https://www.tensorflow.org/versions/r0.11/api_docs/python/state_ops.html#Saver
            saver = tf.train.Saver()  # might be under tf.initialize_all_variables().run()

            cprint("I am here before global int", "r")  # TESTING

            # Initialize your model within a tf.Session
            tf.global_variables_initializer().run()  # or without .run()

            cprint("I am here before def _feed_dict()", "r")  # TESTING

            def _feed_dict(training):
                """creates feed_dicts depending on training or no training"""
                # Train
                if training:
                    xs, ys = nevro_data["train"].next_batch(batch_size=BATCH_SIZE_DEFAULT,
                                                            randomize=RANDOM_BATCH_DEFAULT,
                                                            successive=SUCCESSIVE_BATCHES_DEFAULT,
                                                            successive_mode=SUCCESSIVE_MODE_DEFAULT)

                    # ys = np.reshape(ys, newshape=([BATCH_SIZE_DEFAULT] + list(ys.shape)))
                    ys = np.reshape(ys, newshape=([BATCH_SIZE_DEFAULT, 1]))

                else:
                    # Validation:
                    xs, ys = nevro_data["validation"].next_batch(batch_size=1,
                                                                 randomize=False,
                                                                 successive=1)
                    # ys = np.reshape(ys, newshape=([BATCH_SIZE_DEFAULT] + list(ys.shape)))
                    ys = np.reshape(ys, newshape=([1, 1]))

                return {x: xs, y: ys}

            # RUN
            cprint("I am here starting the steps", "r")  # TESTING

            # val_counter = 0  # Needed when validation should only be run in the end of training
            val_steps = int(270 / S_FOLD_DEFAULT)

            # Init Lists of Accuracy and Loss
            train_loss_list = []
            train_acc_list = []
            val_acc_list = []
            val_loss_list = []
            val_acc_training_list = []
            val_loss_training_list = []

            # Set Variables for timer
            timer_list = []  # # list of duration(time) of timer_freq steps
            # duration = []  # durations of timer_freq iterations
            start_timer = 0
            # end_timer = 0
            timer_freq = 100

            for step in range(int(max_steps)):

                cprint("\nStep {} in fold {}".format(step, rnd), "r")  # TESTING

                # Timer for every timer_freq steps
                if step == 0:
                    # Set Start Timer
                    start_timer = datetime.datetime.now().replace(microsecond=0)

                if step % 20 == 0:
                    cprint("Steps {}/{} in Fold Nr.{} ({}/{}) | {} | {}".format(
                        step, int(max_steps), s_fold_idx, rnd + 1, len(s_fold_idx_list),
                        PATH_SPECIFICITIES_DEFAULT[:-1], s(SUBJECT_DEFAULT)), "b")

                # Evaluate on training set every print_freq (=10) iterations
                if (step + 1) % print_freq == 0:
                    cprint("print_freq active in step {}".format(step), "r")  # TESTING

                    # TODO <<<<<< here is the problem
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    # TODO  here is the problem >>>>>
                    run_metadata = tf.RunMetadata()

                    cprint("I am about to train and with summary in step {}".format(step), "r")  # TESTING

                    summary, _, train_loss, train_acc, train_infer, train_y = sess.run([
                        merged, optimization, loss, accuracy, infer, y],
                        feed_dict=_feed_dict(training=True),
                        options=run_options,
                        run_metadata=run_metadata)

                    cprint("I am about to save summaries in step {}".format(step), "r")  # TESTING

                    train_writer.add_run_metadata(run_metadata, "step{}".format(str(step).zfill(4)))
                    train_writer.add_summary(summary=summary, global_step=step)

                    print("\nTrain-Loss: {:.3f} at step: {} | {} | {}".format(
                        np.round(train_loss, 3), step + 1, PATH_SPECIFICITIES_DEFAULT[:-1],
                        s(SUBJECT_DEFAULT)))
                    print("Train-Accuracy: {:.3f} at step: {} | {} | S{}\n".format(
                        np.round(train_acc, 3), step + 1, PATH_SPECIFICITIES_DEFAULT[:-1],
                        str(SUBJECT_DEFAULT).zfill(2)))

                    # Update Lists
                    train_acc_list.append(train_acc)
                    train_loss_list.append(train_loss)

                else:
                    # summary, _, train_loss, train_acc = sess.run([merged, optimization, loss, accuracy],
                    #                                              feed_dict=_feed_dict(training=True))
                    # train_writer.add_summary(summary=summary, global_step=step)
                    cprint("I am here about to train in step {}".format(step), "r")  # TESTING
                    _, train_loss, train_acc, train_infer, train_y = sess.run(
                        [optimization, loss, accuracy, infer, y], feed_dict=_feed_dict(training=True))

                    if step % 10 == 0:
                        print("\nTrain-Loss: {:.3f} at step: {} | {} | {}".format(
                            np.round(train_loss, 3), step + 1, PATH_SPECIFICITIES_DEFAULT[:-1],
                            s(SUBJECT_DEFAULT)))
                        print("Train-Accuracy: {:.3f} at step: {} | {} | {}\n".format(
                            np.round(train_acc, 3), step + 1, PATH_SPECIFICITIES_DEFAULT[:-1],
                            s(SUBJECT_DEFAULT)))

                    # Update Lists
                    train_acc_list.append(train_acc)
                    train_loss_list.append(train_loss)

                # Write train_infer & train_y in prediction matrix
                pred_matrix = fill_pred_matrix(pred=train_infer, y=train_y, current_mat=pred_matrix,
                                               sfold=S_FOLD_DEFAULT, s_idx=s_fold_idx,
                                               current_batch=nevro_data["train"].current_batch,
                                               train=True)

                # Evaluate on validation set every eval_freq (=1000) iterations
                if (step + 1) % eval_freq == 0:
                    # val_counter += 1  # count the number of val steps (implementation could improved)

                    # Check (average) val_performance during training
                    va_ls_acc = []
                    va_ls_loss = []

                    for val_step in range(val_steps):
                        val_train_loss, val_train_acc = sess.run([loss, accuracy],
                                                                 feed_dict=_feed_dict(training=False))

                        va_ls_acc.append(val_train_acc)
                        va_ls_loss.append(val_train_loss)

                    if step % 10 == 0:
                        print("Val-Loss: {:.3f} at step: {} | {} | {}".format(
                            np.round(np.mean(va_ls_loss), 3), step + 1, PATH_SPECIFICITIES_DEFAULT[:-1],
                            s(SUBJECT_DEFAULT)))
                        print("Val-Accuracy: {:.3f} at step: {} | {} | {}".format(
                            np.round(np.mean(va_ls_acc), 3), step + 1, PATH_SPECIFICITIES_DEFAULT[:-1],
                            s(SUBJECT_DEFAULT)))

                    # Update Lists
                    val_acc_training_list.append((np.mean(va_ls_acc), step))   # tuple: val_acc & step
                    val_loss_training_list.append((np.mean(va_ls_loss), step))

                if step == max_steps - 1:  # Validation in last round

                    # val_counter /= REPETITION_SCALAR_DEFAULT
                    # assert float(val_counter).is_integer(), \
                    #     "val_counter (={}) devided by repetition (={}) is not integer".format(
                    #         val_counter, REPETITION_SCALAR_DEFAULT)
                    # assert val_counter == val_steps, "Final val_counter must be 270/s_fold = val_steps"
                    # if it does not work use nevro_data["validation"].eeg.shape[0]

                    for val_step in range(val_steps):
                        summary, val_loss, val_acc, val_infer, val_y = sess.run(
                            [merged, loss, accuracy, infer, y], feed_dict=_feed_dict(training=False))
                        # test_loss, test_acc = sess.run([loss, accuracy], feed_dict=_feed_dict(False))
                        # print("now do: test_writer.add_summary(summary=summary, global_step=step)")
                        test_writer.add_summary(summary=summary, global_step=step)

                        if val_step % 5 == 0:
                            print("Validation-Loss: {:.3f} of Fold Nr.{} ({}/{}) | {} | {}".format(
                                np.round(val_loss, 3), s_fold_idx, rnd+1, len(s_fold_idx_list),
                                PATH_SPECIFICITIES_DEFAULT[:-1], s(SUBJECT_DEFAULT)))
                            print("Validation-Accuracy: {:.3f} of Fold Nr.{} ({}/{}) | {} | {}".format(
                                np.round(val_acc, 3), s_fold_idx, rnd+1, len(s_fold_idx_list),
                                PATH_SPECIFICITIES_DEFAULT[:-1], s(SUBJECT_DEFAULT)))

                        # Update Lists
                        val_acc_list.append(val_acc)
                        val_loss_list.append(val_loss)

                        # Write val_infer & val_y in val_pred_matrix
                        val_pred_matrix = fill_pred_matrix(pred=val_infer, y=val_y,
                                                           current_mat=val_pred_matrix,
                                                           sfold=S_FOLD_DEFAULT, s_idx=s_fold_idx,
                                                           current_batch=nevro_data[
                                                               "validation"].current_batch,
                                                           train=False)

                # Save the variables to disk every checkpoint_freq (=5000) iterations
                if (step + 1) % checkpoint_freq == 0:
                    save_path = saver.save(sess=sess,
                                           save_path=CHECKPOINT_DIR_DEFAULT+"/lstmnet_rnd{}.ckpt".format(
                                               str(rnd).zfill(2)),
                                           global_step=step)
                    print("Model saved in file: %s" % save_path)

                # End Timer
                if step % timer_freq == 0 and step > 0:

                    end_timer = datetime.datetime.now().replace(microsecond=0)

                    # cprint("I am here at end_timer: {}",format(end_timer), "r)  # TESTING

                    # Calculate Duration and Estimations
                    duration = end_timer - start_timer

                    # cprint("I am here at duration: {}".format(duration), "r")  # TESTING

                    timer_list.append(duration)  # mean(timer_list) = average time per xx steps

                    # For this fold
                    # Cannot take mean(daytime) in python2
                    # estim_t_per_step = np.mean(timer_list) / timer_freq  # only python3
                    mean_timer_list = average_time(list_of_timestamps=timer_list, in_timedelta=False)
                    estim_t_per_step = mean_timer_list / timer_freq
                    remaining_steps_in_fold = (max_steps - (step + 2))
                    rest_duration = remaining_steps_in_fold * estim_t_per_step

                    # cprint("I am here at rest_duration: {}".format(rest_duration), "r")  # TESTING

                    # For whole training
                    remaining_folds = len(s_fold_idx_list) - (rnd + 1)
                    if rnd == 0:
                        remaining_steps = max_steps*remaining_folds
                        rest_duration_all_folds = rest_duration + remaining_steps * estim_t_per_step
                    else:  # this is more accurate, but only possible after first round(rnd)/fold
                        rest_duration_all_folds = rest_duration + \
                                                  average_time(timer_fold_list,
                                                               in_timedelta=False)*remaining_folds
                    # convert back to: datetime.timedelta(seconds=27)
                    rest_duration = datetime.timedelta(seconds=rest_duration)
                    rest_duration_all_folds = datetime.timedelta(seconds=rest_duration_all_folds)
                    # Remove microseconds
                    rest_duration = chop_microseconds(delta=rest_duration)

                    # cprint("I am here at rest_duration: {}".format(rest_duration), "r")  # TESTING

                    rest_duration_all_folds = chop_microseconds(delta=rest_duration_all_folds)

                    cprint("Time passed to train {} steps: {} [h:m:s] | {} | {}".format(
                        timer_freq, duration,
                        PATH_SPECIFICITIES_DEFAULT[:-1],
                        s(SUBJECT_DEFAULT)), "y")
                    cprint("Estimated time to train the rest {} steps in current Fold-Nr.{}: "
                           "{} [h:m:s] | {} | {}".format(int(max_steps - (step + 1)), s_fold_idx,
                                                         rest_duration,
                                                         PATH_SPECIFICITIES_DEFAULT[:-1],
                                                         s(SUBJECT_DEFAULT)), "y")
                    cprint("Estimated time to train the rest steps and {} {}: {} [h:m:s] | {} | "
                           "{}".format(remaining_folds, "folds" if remaining_folds > 1 else "fold",
                                       rest_duration_all_folds, PATH_SPECIFICITIES_DEFAULT[:-1],
                                       s(SUBJECT_DEFAULT)), "y")

                    # Set Start Timer
                    start_timer = datetime.datetime.now().replace(microsecond=0)

            # Close Writers:
            train_writer.close()
            test_writer.close()

            # Save last val_acc in all_acc_val-vector
            # since we validate in the end of training average across all
            all_acc_val[rnd] = np.nanmean(val_acc_list)

            # Save loss_ & acc_lists externally per S-Fold
            loss_acc_lists = [train_loss_list, train_acc_list,
                              val_acc_list, val_loss_list,
                              val_acc_training_list, val_loss_training_list]
            loss_acc_lists_names = ["train_loss_list", "train_acc_list",
                                    "val_acc_list", "val_loss_list",
                                    "val_acc_training_list", "val_loss_training_list"]
            for list_idx, liste in enumerate(loss_acc_lists):
                with open(LOG_DIR_DEFAULT + str(s_fold_idx) + "/{}.txt".format(
                        loss_acc_lists_names[list_idx]), "w") as list_file:
                    for value in liste:
                        list_file.write(str(value) + "\n")

        # Fold End Timer
        end_timer_fold = datetime.datetime.now().replace(microsecond=0)
        duration_fold = end_timer_fold - start_timer_fold

    # In case data was shuffled save corresponding order externally
    if SHUFFLE_DEFAULT:
        np.save(file=LOG_DIR_DEFAULT + str(s_fold_idx) + "/{}_shuffle_order.npy".format(s_fold_idx),
                arr=nevro_data["order"])

# Final Accuracy & Time
timer_fold_list.append(duration_fold)
print("Time to train all folds (each {} steps): {} [h:m:s] | {} | {}".format(
    int(max_steps), np.sum(timer_fold_list), PATH_SPECIFICITIES_DEFAULT[:-1],
    s(SUBJECT_DEFAULT)))

# Create all_acc_val_binary
all_acc_val_binary = calc_binary_class_accuracy(prediction_matrix=val_pred_matrix)
# Calculate mean val accuracy across all folds
mean_val_acc = np.nanmean(all_acc_val if TASK_DEFAULT == "regression" else all_acc_val_binary)

print("Average accuracy across all {} validation set: {:.3f} | {} | {}".format(
    S_FOLD_DEFAULT, mean_val_acc, PATH_SPECIFICITIES_DEFAULT[:-1], s(SUBJECT_DEFAULT)))


with open(SUB_DIR_DEFAULT + "{}S{}_accuracy_across_{}_folds_{}.txt".format(
        time.strftime('%Y_%m_%d_'), SUBJECT_DEFAULT, S_FOLD_DEFAULT, PATH_SPECIFICITIES_DEFAULT[:-1]),
          "w") as file:
    file.write("Subject {}\nCondition: {}\nSBA: {}\nTask: {}\nShuffle_data: {}\ndatatype: {}"
               "\nband_pass: {}\nHilbert_z-Power: {}"
               "\ns-Fold: {}\nmax_step: {}\nrepetition_set: {}\nlearning_rate: {}"
               "\nbatch_size: {}\nbatch_random: {}"
               "\nsuccessive_batches: {}(mode {})\nweight_reg: {}({})\nact_fct: {}"
               "\nlstm_h_size: {}\nn_hidden_units: {}"
               "\ncomponent: {}({}){}"
               "\nfix_input_matrix_size: {}"
               "\npath_specificities: {}\n".format(SUBJECT_DEFAULT, CONDITION_DEFAULT, SBA_DEFAULT,
                                                   TASK_DEFAULT, SHUFFLE_DEFAULT, FILE_TYPE_DEFAULT,
                                                   BAND_PASS_INPUT_DEFAULT, HILBERT_POWER_INPUT_DEFAULT,
                                                   S_FOLD_DEFAULT, int(max_steps),
                                                   REPETITION_SCALAR_DEFAULT,
                                                   LEARNING_RATE_DEFAULT,
                                                   BATCH_SIZE_DEFAULT, RANDOM_BATCH_DEFAULT,
                                                   SUCCESSIVE_BATCHES_DEFAULT, SUCCESSIVE_MODE_DEFAULT,
                                                   WEIGHT_REGULARIZER_DEFAULT,
                                                   WEIGHT_REGULARIZER_STRENGTH_DEFAULT,
                                                   ACTIVATION_FCT_DEFAULT,
                                                   LSTM_SIZE_DEFAULT, str(n_hidden_units),
                                                   COMPONENT_DEFAULT, input_component,
                                                   " + HRcomp"if HR_COMPONENT_DEFAULT else "",
                                                   EQCOMPMAT_DEFAULT,
                                                   PATH_SPECIFICITIES_DEFAULT))

    # rounding for the export
    rnd_all_acc_val = ["{:.3f}".format(np.round(acc, 3)) for acc in all_acc_val]
    rnd_all_acc_val_binary = ["{:.3f}".format(np.round(acc, 3)) for acc in all_acc_val_binary]
    rnd_all_acc_val = [float(acc) for acc in rnd_all_acc_val]  # cleaning
    rnd_all_acc_val_binary = [float(acc) for acc in rnd_all_acc_val_binary]

    # preparing export
    lists_export = [s_fold_idx_list, rnd_all_acc_val, np.round(np.mean(all_acc_val), 3),
                    np.round(mean_line_acc, 3), np.sum(timer_fold_list)]
    label_export = ["S-Fold(Round): ", "Validation-Acc: ", "mean(Accuracy): ",
                    "mean_line_acc: ", "Train-Time: "]

    if TASK_DEFAULT == "classification":
        lists_export.insert(len(lists_export)-1, rnd_all_acc_val_binary)
        lists_export.insert(len(lists_export)-1, np.round(np.nanmean(rnd_all_acc_val_binary), 3))
        label_export.insert(len(label_export)-1, "Validation-Class-Acc: ")
        label_export.insert(len(label_export)-1, "mean(Classification_Accuracy): ")

    for i, item in enumerate(lists_export):
        file.write(label_export[i] + str(item) + "\n")


# Save Accuracies in Random_Search_Table.csv if applicable
table_name = "./LSTM/Random_Search_Table_{}.csv".format(
    "BiCl" if TASK_DEFAULT == "classification" else "Reg")
if os.path.exists(table_name):
    rs_table = np.genfromtxt(table_name, delimiter=";", dtype=str)

    if PATH_SPECIFICITIES_DEFAULT in rs_table:
        # Find corresponding indeces
        path_idx = np.where(rs_table == PATH_SPECIFICITIES_DEFAULT)
        # len(path_idx[0]) > 0 and len(path_idx[1]) > 0
        sub_idx = np.where(rs_table[:, np.where(rs_table == "subject")[1]] == str(SUBJECT_DEFAULT))[0]
        trial_row = list(set(path_idx[0]) & set(sub_idx))[0]
        mvacc_col = np.where(rs_table == "mean_val_acc")[1][0]
        zlacc_col = np.where(rs_table == "zeroline_acc")[1][0]
        mcvacc_col = np.where(rs_table == "mean_class_val_acc")[1][0]
        # Write in table
        rs_table[trial_row, [mvacc_col, zlacc_col, mcvacc_col]] = np.array(
            [np.round(np.mean(all_acc_val), 3), np.round(mean_line_acc, 3),
             np.round(np.nanmean(rnd_all_acc_val_binary), 3) if TASK_DEFAULT == "classification" else
             rs_table[trial_row, mcvacc_col]])

        # Save table
        np.savetxt(fname=table_name, X=rs_table, delimiter=";", fmt="%s")

    else:
        cprint("There is no entry for this trial in Random_Search_Table_{}.csv".format(
            "BiCl" if TASK_DEFAULT == "classification" else "Reg"), "r")


# Save Prediction Matrices in File
np.savetxt(SUB_DIR_DEFAULT + "{}S{}_pred_matrix_{}_folds_{}.csv".format(
    time.strftime('%Y_%m_%d_'), SUBJECT_DEFAULT, S_FOLD_DEFAULT, PATH_SPECIFICITIES_DEFAULT[:-1]),
           pred_matrix, delimiter=",")

np.savetxt(SUB_DIR_DEFAULT + "{}S{}_val_pred_matrix_{}_folds_{}.csv".format(
    time.strftime('%Y_%m_%d_'), SUBJECT_DEFAULT, S_FOLD_DEFAULT, PATH_SPECIFICITIES_DEFAULT[:-1]),
           val_pred_matrix, delimiter=",")

if SHUFFLE_DEFAULT:
    np.save(SUB_DIR_DEFAULT + "{}S{}_shuffle_order_matrix_{}_folds_{}.npy".format(
        time.strftime('%Y_%m_%d_'), SUBJECT_DEFAULT, S_FOLD_DEFAULT, PATH_SPECIFICITIES_DEFAULT[:-1]),
            shuffle_order_matrix)


if PLOT_DEFAULT:
    # ["python3", "LSTM_pred_plot.py", Save_plots='True', Path specificities]
    subprocess.Popen(["python3", "LSTM_pred_plot.py", 'True',
                      str(SUBJECT_DEFAULT), PATH_SPECIFICITIES_DEFAULT, str(DEL_LOG_DEFAULT)])

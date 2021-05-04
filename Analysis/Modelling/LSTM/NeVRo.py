# coding=utf-8
"""
Main script
    • run model
    • script should be called via bash files (see parser)

Author: Simon M. Hofmann | <[surname].[lastname][at]pm.me> | 2017, 2019, 2021 (Update)
"""

# %% Import

# Adaptations if code is run under Python2
from __future__ import absolute_import
from __future__ import division  # int/int can result in float now, 1/2 = 0.5 (in python2 1/2=0, 1/2.=0.5)
from __future__ import print_function  # : Use print as a function as in Python 3: print()

# import sys
# sys.path.insert(0, './LSTM Model')  # or set the folder as source root
from load_data import *

import numpy as np
import tensorflow as tf  # implemented with TensorFlow 1.13.1
import argparse
import time
import copy
import subprocess

from NeVRoNet import NeVRoNet
from write_random_search_bash import update_bashfiles

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Makes TensorFlow less verbose, comment out for debugging

# %% TO DO's >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

# TODO successively adding SSD components, adding more non-alpha related information (non-b-pass)
# TODO test trained model on different subject dataset.
# TODO Train model on various subjects

# %% Set Defaults >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<

TASK_DEFAULT = 'classification'  # prediction via 'regression' (continuous) or 'classification' (low-high)
LEARNING_RATE_DEFAULT = 1e-3  # 1e-4
BATCH_SIZE_DEFAULT = 9  # or bigger, batch_size must be a multiple of 'successive batches'
SUCCESSIVE_BATCHES_DEFAULT = 1  # (time-)length per sample is hyperparameter in form of successive batches
SUCCESSIVE_MODE_DEFAULT = 1  # either 1 or 2
S_FOLD_DEFAULT = 10
REPETITION_SCALAR_DEFAULT = 30  # scaler for how many times it runs through set (can be also fraction)
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

PATH_SPECIFICITIES_DEFAULT = ""  # or fill like this: "special_folder/"

# %% Model dicts >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<

WEIGHT_REGULARIZER_DICT = {'none': lambda x: None,  # No regularization
                           # L1 regularization
                           'l1': tf.keras.regularizers.l1,  # tf.contrib.layers.l1_regularizer,
                           # L2 regularization
                           'l2': tf.keras.regularizers.l2}  # tf.contrib.layers.l2_regularizer

ACTIVATION_FCT_DICT = {'elu': tf.nn.elu,
                       'relu': tf.nn.relu}


# %% Functions: Model training  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<


def train_step(loss):
    """
    Defines the ops to conduct an optimization step. Could do: Implement Learning rate scheduler.

    Args:
        loss: scalar float Tensor, full loss = mean squared error + reg_loss

    Returns:
        train_op: Ops for optimization.
    """

    train_op = None
    # could do: Learning rate scheduler
    if OPTIMIZER_DEFAULT == 'ADAM':
        train_op = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss)

    return train_op


def train_lstm():
    """
    Performs training and evaluation of LSTM model.
    """

    # # Parameters
    # runs max_steps-times through set
    num_samples = 270 if FLAGS.task == "regression" else 180
    max_steps = FLAGS.repet_scalar * (num_samples - num_samples / FLAGS.s_fold) / FLAGS.batch_size
    assert float(max_steps).is_integer(), "max steps must be integer"
    eval_freq = int(((num_samples - num_samples / FLAGS.s_fold) / FLAGS.batch_size) / 2)
    # approx. 2 times per epoch
    checkpoint_freq = int(max_steps / 2)  # int(max_steps)/2 for chechpoint after half the training
    print_freq = int(max_steps / 8)  # if too low, uses much memory
    assert FLAGS.batch_size % FLAGS.successive == 0, \
        "batch_size must be a multiple of successive (batches)."
    assert FLAGS.task.upper() in ['REGRESSION', 'CLASSIFICATION'], "Prediction task is undefined."

    # Set the random seeds on True for reproducibility.
    # Switch for seed
    if FLAGS.seed:
        tf.set_random_seed(42)
        np.random.seed(42)
        cprint("Seed is turned on", "b")

    # Get number of units in each hidden layer specified in the string such as 100,100
    if FLAGS.lstm_size:
        lstm_hidden_states = [int(lstm_l) for lstm_l in FLAGS.lstm_size.split(",")]
    else:
        lstm_hidden_states = []

    # For fully connected layers
    if FLAGS.fc_n_hidden and len(FLAGS.fc_n_hidden) > 0 and FLAGS.fc_n_hidden not in "0":
        n_hidden_units = [int(fc_l) for fc_l in FLAGS.fc_n_hidden.split(",")] + [1]
        # output layer==1 rating-prediction
    else:
        n_hidden_units = [1]

    # Start TensorFlow Interactive Session
    # sess = tf.InteractiveSession()

    # s_fold_idx depending on s_fold and previous index
    s_fold_idx_list = np.arange(FLAGS.s_fold)
    np.random.shuffle(s_fold_idx_list)

    # Create graph_dic
    graph_dict = dict.fromkeys(s_fold_idx_list)
    for key in graph_dict.keys():
        graph_dict[key] = tf.Graph()

    # Create to save the performance for each validation set
    all_acc_val = np.zeros(FLAGS.s_fold)

    # # Choose component: Given by list e.g., '1,3,5' or as label, e.g., 'best'
    input_component = None  # initialize

    # First find best component
    # choose_component(subject, condition, f_type, best, sba=True)
    # TODO remove deprecated 'best' - component test
    if FLAGS.component in ['best', 'noise']:
        best_comp = best_or_random_component(subject=FLAGS.subject, condition=FLAGS.condition,
                                             f_type=FLAGS.filetype.upper(),
                                             best=True,
                                             sba=FLAGS.sba)

    if not FLAGS.component.split(",")[0].isnumeric():

        assert FLAGS.component in ["best", "noise", "random", "all"], \
            "Component must be either 'best', 'noise', 'random', or 'all'"

        if FLAGS.component == "best":
            input_component = best_comp
        elif FLAGS.component == "noise":
            input_component = 90 + best_comp  # coding for noise component
        elif FLAGS.component == "random":
            input_component = best_or_random_component(subject=FLAGS.subject,
                                                       condition=FLAGS.condition,
                                                       f_type=FLAGS.filetype.upper(),
                                                       best=False,  # Finds random component != best
                                                       sba=FLAGS.sba)
        elif FLAGS.component == "all":
            n_comp = get_num_components(subject=FLAGS.subject, condition=FLAGS.condition,
                                        filetype=FLAGS.filetype)
            input_component = list(range(1, n_comp + 1))

    else:  # given components are in form of list
        assert np.all([comp.isnumeric() for comp in FLAGS.component.split(",")]), \
            "All given components must be numeric"
        input_component = [int(comp) for comp in
                           FLAGS.component.split(",")]  # list(ast.literal_eval(FLAGS.component))

    print("LSTM model get trained on input_component(s):", input_component)

    # Load first data-set for preparation
    nevro_data = get_nevro_data(subject=FLAGS.subject,
                                task=FLAGS.task,
                                cond=FLAGS.condition,
                                component=input_component,
                                hr_component=FLAGS.hrcomp,
                                filetype=FLAGS.filetype,
                                hilbert_power=FLAGS.hilbert_power,
                                band_pass=FLAGS.band_pass,
                                equal_comp_matrix=None if FLAGS.eqcompmat == 0 else FLAGS.eqcompmat,
                                s_fold_idx=s_fold_idx_list[0],
                                s_fold=FLAGS.s_fold,
                                sba=FLAGS.sba,
                                shuffle=FLAGS.shuffle,
                                shuffle_order=None,
                                balanced_cv=FLAGS.balanced_cv,
                                testmode=FLAGS.testmodel)

    mean_line_acc = mean_line_prediction(subject=FLAGS.subject, condition=FLAGS.condition, sba=FLAGS.sba)

    # Define graph using class NeVRoNet and its methods:
    ddims = list(nevro_data["train"].eeg.shape[1:])  # [250, n_comp]
    full_length = len(nevro_data["validation"].ratings) * FLAGS.s_fold  # 270

    # Prediction Matrix for each fold
    # 1 Fold predic [0.156, ..., 0.491]
    # 1 Fold rating [0.161, ..., 0.423]
    # ...   ...   ...   ...   ...   ...
    # S Fold predic [0.397, ..., -0.134]
    # S Fold rating [0.412, ..., -0.983]
    pred_matrix = np.zeros(shape=(FLAGS.s_fold * 2, full_length), dtype=np.float32)
    pred_matrix[np.where(pred_matrix == 0)] = np.nan  # set to NaN values
    # We create a separate matrix for the validation (which can be merged with the 1. pred_matrix later)
    val_pred_matrix = copy.copy(pred_matrix)

    # In case data was shuffled save corresponding order per fold in matrix and save later
    shuffle_order_matrix = np.zeros(shape=(FLAGS.s_fold, pred_matrix.shape[1]))
    shuffle_order_matrix[s_fold_idx_list[0], :] = nevro_data["order"]
    global_shuffle_order = nevro_data["order"] if FLAGS.balanced_cv else None

    # Set Variables for timer
    timer_fold_list = []  # list of duration(time) of each fold
    duration_fold = []  # duration of 1 fold
    # rest_duration = 0

    # Run through S-Fold-Cross-Validation (take the mean-performance across all validation sets)
    for rnd, s_fold_idx in enumerate(s_fold_idx_list):
        cprint(f"\nTrain now on Fold-Nr.{s_fold_idx} (fold {rnd + 1}/{len(s_fold_idx_list)}) | "
               f"{FLAGS.path_specificities[:-1]} | {s(FLAGS.subject)}", "b")
        start_timer_fold = datetime.now().replace(microsecond=0)

        # For each fold define new graph to finally compare the validation accuracies of each fold
        with tf.Session(graph=graph_dict[s_fold_idx]) as sess:  # (re-)initialise the model completely

            with tf.variable_scope(name_or_scope=f"Fold_Nr{str(rnd).zfill(len(str(FLAGS.s_fold)))}/"
            f"{len(s_fold_idx_list)}"):

                # Load Data:
                if rnd > 0:
                    # Show time passed per fold and estimation of rest time
                    cprint(f"{s(FLAGS.subject)} | {FLAGS.path_specificities[:-1]}", "y")
                    cprint(f"{datetime.now().replace(microsecond=0)}", "g")
                    cprint(f"Duration of previous fold {duration_fold} [h:m:s]", "y")
                    timer_fold_list.append(duration_fold)
                    # average over previous folds (np.mean(timer_fold_list) not possible in python2)
                    rest_duration_fold = average_time(timer_fold_list,
                                                      in_timedelta=True) * (FLAGS.s_fold - rnd)
                    rest_duration_fold = chop_microseconds(delta=rest_duration_fold)
                    cprint(f"Estimated time to train rest {FLAGS.s_fold - rnd} fold(s): "
                           f"{rest_duration_fold} [h:m:s]\n", "y")

                    nevro_data = get_nevro_data(subject=FLAGS.subject,
                                                component=input_component,
                                                hr_component=FLAGS.hrcomp,
                                                s_fold_idx=s_fold_idx,
                                                s_fold=FLAGS.s_fold,
                                                cond=FLAGS.condition,
                                                sba=FLAGS.sba,
                                                filetype=FLAGS.filetype,
                                                band_pass=FLAGS.band_pass,
                                                equal_comp_matrix=None if FLAGS.eqcompmat == 0
                                                else FLAGS.eqcompmat,
                                                hilbert_power=FLAGS.hilbert_power,
                                                task=FLAGS.task,
                                                shuffle=FLAGS.shuffle,
                                                shuffle_order=global_shuffle_order,
                                                # is None if not FLAGS.balanced_cv
                                                balanced_cv=FLAGS.balanced_cv,
                                                testmode=FLAGS.testmodel)

                    # Save order in shuffle_order_matrix: shuffle=False:(1,2,...,270);
                    # shuffle=True:(56,4,...,173)
                    shuffle_order_matrix[s_fold_idx, :] = nevro_data["order"]

                with tf.name_scope("input"):
                    # shape = [None] + ddims includes num_steps = 250
                    # Tensorflow requires input as a tensor (a Tensorflow variable) of the dimensions
                    # [batch_size, sequence_length, input_dimension] (a 3d variable).
                    x = tf.placeholder(dtype=tf.float32, shape=[None] + ddims, name="x-input")
                    # None for Batch-Size
                    y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="y-input")

                # Model
                lstm_model = NeVRoNet(activation_function=ACTIVATION_FCT_DICT.get(FLAGS.activation_fct),
                                      weight_regularizer=WEIGHT_REGULARIZER_DICT.get(FLAGS.weight_reg)(
                                          FLAGS.weight_reg_strength),
                                      lstm_size=lstm_hidden_states, fc_hidden_unites=n_hidden_units,
                                      n_steps=ddims[0], batch_size=FLAGS.batch_size,
                                      summaries=FLAGS.summaries)  # n_step = 250

                # Prediction
                with tf.variable_scope(name_or_scope="Inference"):
                    infer = lstm_model.inference(x=x)
                # Loss
                loss = lstm_model.loss(infer=infer, ratings=y)

                # Define train_step, savers, summarizers
                # Optimization
                optimization = train_step(loss=loss)

                # Performance
                accuracy = lstm_model.accuracy(infer=infer, ratings=y)

                # Writer
                merged = tf.summary.merge_all()

                # Define logdir
                logdir = f'./processed/logs/{FLAGS.condition}/{s(FLAGS.subject)}/' \
                    f'{FLAGS.path_specificities}'

                if not tf.gfile.Exists(logdir):
                    tf.gfile.MakeDirs(logdir)

                train_writer = tf.summary.FileWriter(logdir=logdir + str(s_fold_idx) + "/train",
                                                     graph=sess.graph)
                test_writer = tf.summary.FileWriter(logdir=logdir + str(s_fold_idx) + "/val")
                # test_writer = tf.train.SummaryWriter(logdir=logdir + "/test")

                # Saver
                # https://www.tensorflow.org/versions/r0.11/api_docs/python/state_ops.html#Saver
                saver = tf.train.Saver(max_to_keep=1)

                # Initialize your model within a tf.Session
                tf.global_variables_initializer().run()  # or without .run()

                def _feed_dict(training):
                    """creates feed_dicts depending on training or no training"""
                    # Train
                    if training:
                        xs, ys = nevro_data["train"].next_batch(batch_size=FLAGS.batch_size,
                                                                randomize=FLAGS.rand_batch,
                                                                successive=FLAGS.successive,
                                                                successive_mode=FLAGS.successive_mode)
                        ys = np.reshape(ys, newshape=([FLAGS.batch_size, 1]))

                    else:
                        # Validation:
                        xs, ys = nevro_data["validation"].next_batch(batch_size=1,
                                                                     randomize=False,
                                                                     successive=1)
                        ys = np.reshape(ys, newshape=([1, 1]))

                    return {x: xs, y: ys}

                # RUN
                # val_counter = 0  # Needed when validation should only be run in the end of training
                if FLAGS.task == "classification":
                    val_steps = len(nevro_data["validation"].remaining_slices)  # 18
                else:  # for regression
                    total_length = nevro_data["validation"].num_time_slices
                    val_steps = int(total_length / FLAGS.s_fold)  # 27

                # Init Lists of Accuracy and Loss
                train_loss_list = []
                train_acc_list = []
                val_acc_list = []
                val_loss_list = []
                val_acc_training_list = []
                val_loss_training_list = []

                # Set Variables for timer
                timer_list = []  # # list of duration(time) of 100 steps
                # duration = []  # durations of 100 iterations
                start_timer = 0
                # end_timer = 0
                timer_freq = 100

                for step in range(int(max_steps)):

                    # Timer for every timer_freq=100 steps
                    if step == 0:
                        # Set Start Timer
                        start_timer = datetime.now().replace(microsecond=0)

                    if step % (timer_freq / 2) == 0.:
                        cprint(f"Step {step}/{int(max_steps)} in Fold Nr.{s_fold_idx} "
                               f"({rnd + 1}/{len(s_fold_idx_list)}) | {FLAGS.path_specificities[:-1]} | "
                               f"{s(FLAGS.subject)}", "b")

                    # Evaluate on training set every print_freq (=10) iterations
                    if (step + 1) % print_freq == 0:
                        run_metadata = tf.RunMetadata()

                        summary, _, train_loss, train_acc, tain_infer, train_y = sess.run([
                            merged, optimization, loss, accuracy, infer, y],
                            feed_dict=_feed_dict(training=True),  # options=run_options,
                            run_metadata=run_metadata)

                        train_writer.add_run_metadata(run_metadata, f"step{str(step).zfill(4)}")
                        train_writer.add_summary(summary=summary, global_step=step)

                        print(f"\nTrain-Loss:\t{train_loss:.3f} at step: {step + 1}")
                        print(f"Train-Accuracy:\t{train_acc:.3f} at step: {step + 1}\n")

                        # Update Lists
                        train_acc_list.append(train_acc)
                        train_loss_list.append(train_loss)

                    else:
                        _, train_loss, train_acc, tain_infer, train_y = sess.run(
                            [optimization, loss, accuracy, infer, y], feed_dict=_feed_dict(training=True))

                        if step % 25 == 0:
                            print(f"\nTrain-Loss:\t{train_loss:.3f} at step: {step + 1}")
                            print(f"Train-Accuracy:\t{train_acc:.3f} at step: {step + 1}\n")

                        # Update Lists
                        train_acc_list.append(train_acc)
                        train_loss_list.append(train_loss)

                    # Write train_infer & train_y in prediction matrix
                    pred_matrix = fill_pred_matrix(pred=tain_infer, y=train_y, current_mat=pred_matrix,
                                                   sfold=FLAGS.s_fold, s_idx=s_fold_idx,
                                                   current_batch=nevro_data["train"].current_batch,
                                                   train=True)

                    # Evaluate on validation set every eval_freq iterations
                    if (step + 1) % eval_freq == 0:
                        # val_counter += 1  # count the number of val steps (implementation could improve)

                        # Check (average) val_performance during training
                        va_ls_acc = []
                        va_ls_loss = []

                        for val_step in range(val_steps):
                            val_train_loss, val_train_acc = sess.run([loss, accuracy],
                                                                     feed_dict=_feed_dict(training=False))

                            va_ls_acc.append(val_train_acc)
                            va_ls_loss.append(val_train_loss)

                        if step % 25 == 0:
                            print(f"Val-Loss:\t{np.mean(va_ls_loss):.3f} at step: {step + 1}")
                            print(f"Val-Accuracy:\t{np.mean(va_ls_acc):.3f} at step: {step + 1}")

                        # Update Lists
                        val_acc_training_list.append((np.mean(va_ls_acc), step))  # tuple: val_acc & step
                        val_loss_training_list.append((np.mean(va_ls_loss), step))

                    if step + 1 == max_steps:  # Validation in last round

                        # for val_step in range(int(val_counter)):
                        for val_step in range(int(val_steps)):
                            summary, val_loss, val_acc, val_infer, val_y = sess.run(
                                [merged, loss, accuracy, infer, y], feed_dict=_feed_dict(training=False))
                            # test_loss, test_acc = sess.run([loss, accuracy], feed_dict=feed_dict(False))
                            test_writer.add_summary(summary=summary, global_step=step)

                            if val_step % 5 == 0:
                                print(f"Validation-Loss:\t{val_loss:.3f} of Fold Nr.{s_fold_idx} "
                                      f"({rnd + 1}/{len(s_fold_idx_list)})")
                                print(f"Validation-Accuracy:\t{val_acc:.3f} of Fold Nr.{s_fold_idx} "
                                      f"({rnd + 1}/{len(s_fold_idx_list)})")

                            # Update Lists
                            val_acc_list.append(val_acc)
                            val_loss_list.append(val_loss)

                            # Write val_infer & val_y in val_pred_matrix
                            val_pred_matrix = fill_pred_matrix(pred=val_infer, y=val_y,
                                                               current_mat=val_pred_matrix,
                                                               sfold=FLAGS.s_fold, s_idx=s_fold_idx,
                                                               current_batch=nevro_data[
                                                                   "validation"].current_batch,
                                                               train=False)

                    # Save the variables to disk every checkpoint_freq (=5000) iterations
                    if (step + 1) % checkpoint_freq == 0 or (step + 1) == max_steps:

                        # Define checkpoint_dir
                        checkpoint_dir = f'./processed/checkpoints/{FLAGS.condition}/' \
                            f'{s(FLAGS.subject)}/{FLAGS.path_specificities}'

                        if not tf.gfile.Exists(checkpoint_dir):
                            tf.gfile.MakeDirs(checkpoint_dir)

                        save_path = saver.save(sess=sess,
                                               save_path=checkpoint_dir + f"lstmnet_rnd"
                                               f"{str(rnd).zfill(2)}.ckpt", global_step=step)
                        cprint(f"Model saved in file: {save_path}", "b")

                    # End Timer
                    if step % timer_freq == 0 and step > 0:
                        end_timer = datetime.now().replace(microsecond=0)

                        # Calculate Duration and Estimations
                        duration = end_timer - start_timer
                        timer_list.append(duration)  # mean(timer_list) = average time per 100steps

                        # For this fold
                        # Can not take mean(daytime) in python2
                        # estim_t_per_step = np.mean(timer_list) / timer_freq  # only python3
                        mean_timer_list = average_time(list_of_timestamps=timer_list, in_timedelta=False)
                        estim_t_per_step = mean_timer_list / timer_freq
                        remaining_steps_in_fold = (max_steps - (step + 2))
                        rest_duration = remaining_steps_in_fold * estim_t_per_step

                        # For whole training
                        remaining_folds = len(s_fold_idx_list) - (rnd + 1)
                        if rnd == 0:
                            remaining_steps = max_steps * remaining_folds
                            rest_duration_all_folds = rest_duration + remaining_steps * estim_t_per_step
                        else:  # this is more accurate, but only possible after first round(rnd)/fold
                            rest_duration_all_folds = rest_duration + \
                                                      average_time(timer_fold_list,
                                                                   in_timedelta=False) * remaining_folds
                        # convert back to: timedelta(seconds=27)
                        rest_duration = timedelta(seconds=rest_duration)
                        rest_duration_all_folds = timedelta(seconds=rest_duration_all_folds)
                        # Remove microseconds
                        rest_duration = chop_microseconds(delta=rest_duration)
                        rest_duration_all_folds = chop_microseconds(delta=rest_duration_all_folds)

                        cprint(f"{FLAGS.path_specificities[:-1]} | {s(FLAGS.subject)}", "y")
                        cprint(f"{datetime.now().replace(microsecond=0)}", "g")
                        cprint(f"Time passed to train {timer_freq} steps: {duration} [h:m:s]", "y")
                        cprint(f"Estimated time to train the rest {int(max_steps - (step + 1))} steps in "
                               f"current Fold-Nr.{s_fold_idx}: {rest_duration} [h:m:s]", "y")
                        cprint(f"Estimated time to train the rest steps and {remaining_folds} "
                               f"{'folds' if remaining_folds > 1 else 'fold'}: {rest_duration_all_folds} "
                               f"[h:m:s]", "y")

                        # Set Start Timer
                        start_timer = datetime.now().replace(microsecond=0)

                # Close Writers:
                train_writer.close()
                test_writer.close()

                # Save last val_acc in all_acc_val-vector
                # since we validate in the end of train average across all:
                all_acc_val[rnd] = np.nanmean(val_acc_list)

                # Save loss_ & acc_lists externally per S-Fold
                loss_acc_lists = [train_loss_list, train_acc_list,
                                  val_acc_list, val_loss_list,
                                  val_acc_training_list, val_loss_training_list]
                loss_acc_lists_names = ["train_loss_list", "train_acc_list",
                                        "val_acc_list", "val_loss_list",
                                        "val_acc_training_list", "val_loss_training_list"]
                for list_idx, liste in enumerate(loss_acc_lists):
                    with open(logdir + str(s_fold_idx) + f"/{loss_acc_lists_names[list_idx]}.txt",
                              "w") as list_file:
                        for value in liste:
                            list_file.write(str(value) + "\n")

            # Fold End Timer
            end_timer_fold = datetime.now().replace(microsecond=0)
            duration_fold = end_timer_fold - start_timer_fold

        # In case data was shuffled save corresponding order externally
        if FLAGS.shuffle:
            np.save(file=logdir + str(s_fold_idx) + f"/{s_fold_idx}_shuffle_order.npy",
                    arr=nevro_data["order"])

    # Final Accuracy & Time
    timer_fold_list.append(duration_fold)
    cprint(f"{FLAGS.path_specificities[:-1]} | {s(FLAGS.subject)}", "y")
    cprint(f"Time to train all folds (each {int(max_steps)} steps): {np.sum(timer_fold_list)} [h:m:s]",
           "y")

    # Create all_acc_val_binary
    all_acc_val_binary = calc_binary_class_accuracy(prediction_matrix=val_pred_matrix)
    # Calculate mean val accuracy across all folds
    mean_val_acc = np.nanmean(all_acc_val if FLAGS.task == "regression" else all_acc_val_binary)

    cprint(f"Average accuracy across all {FLAGS.s_fold} validation sets: {mean_val_acc:.3f}", "b")

    # Save training information in Textfile
    # Define sub_dir
    sub_dir = f"./processed/{FLAGS.condition}/{s(FLAGS.subject)}/{FLAGS.path_specificities}"
    if not tf.gfile.Exists(sub_dir):
        tf.gfile.MakeDirs(sub_dir)

    with open(sub_dir + f"{time.strftime('%Y_%m_%d_')}S{FLAGS.subject}_accuracy_across_{FLAGS.s_fold}"
    f"_folds_{FLAGS.path_specificities[:-1]}.txt", "w") as file:
        file.write(f"Subject {FLAGS.subject}\nCondition: {FLAGS.condition}\nSBA: {FLAGS.sba}"
                   f"\nTask: {FLAGS.task}\nShuffle_data: {FLAGS.shuffle}"
                   f"\nBalanced CV: {FLAGS.balanced_cv}\ndatatype: {FLAGS.filetype}"
                   f"\nband_pass: {FLAGS.band_pass}\nHilbert_z-Power: {FLAGS.hilbert_power}"
                   f"\ns-Fold: {FLAGS.s_fold}\nmax_step: {int(max_steps)}"
                   f"\nrepetition_set: {FLAGS.repet_scalar}\nlearning_rate: {FLAGS.learning_rate}"
                   f"\nbatch_size: {FLAGS.batch_size}\nbatch_random: {FLAGS.rand_batch}"
                   f"\nsuccessive_batches: {FLAGS.successive}(mode {FLAGS.successive_mode})"
                   f"\nweight_reg: {FLAGS.weight_reg}({FLAGS.weight_reg_strength})"
                   f"\nact_fct: {FLAGS.activation_fct}"
                   f"\nlstm_h_size: {FLAGS.lstm_size}\nn_hidden_units: {str(n_hidden_units)}"
                   f"\ncomponent: {FLAGS.component}({input_component})"
                   f"{' + HRcomp' if FLAGS.hrcomp else ''}"
                   f"\nfix_input_matrix_size: {FLAGS.eqcompmat}"
                   f"\npath_specificities: {FLAGS.path_specificities}\n")

        # rounding for the export
        rnd_all_acc_val = [f"{acc:.3f}" for acc in all_acc_val]
        rnd_all_acc_val_binary = [f"{acc:.3f}" for acc in all_acc_val_binary]
        rnd_all_acc_val = [float(acc) for acc in rnd_all_acc_val]  # cleaning
        rnd_all_acc_val_binary = [float(acc) for acc in rnd_all_acc_val_binary]

        # preparing export
        lists_export = [s_fold_idx_list, rnd_all_acc_val, np.round(np.mean(all_acc_val), 3),
                        np.round(mean_line_acc, 3), np.sum(timer_fold_list)]
        label_export = ["S-Fold(Round): ", "Validation-Acc: ", "mean(Accuracy): ",
                        "mean_line_acc: ", "Train-Time: "]

        if FLAGS.task == "classification":
            lists_export.insert(len(lists_export) - 1, rnd_all_acc_val_binary)
            lists_export.insert(len(lists_export) - 1, np.round(np.nanmean(rnd_all_acc_val_binary), 3))
            label_export.insert(len(label_export) - 1, "Validation-Class-Acc: ")
            label_export.insert(len(label_export) - 1, "mean(Classification_Accuracy): ")

        for i, item in enumerate(lists_export):
            file.write(label_export[i] + str(item) + "\n")

    # Save Accuracies in Random/Narrow_Search_Table.csv if applicable
    for search in ['Narrow', 'Random']:
        # Check for narrow or random/broad search.
        table_name = f"./processed/{search}_Search_Table_{FLAGS.condition}_" \
            f"{'BiCl' if FLAGS.task == 'classification' else 'Reg'}.csv"
        # Since condition is given, these two cases (search types) are mutually exclusive.
        if os.path.exists(table_name):
            break

    if os.path.exists(table_name):
        rs_table = np.genfromtxt(table_name, delimiter=";", dtype=str)  # load table

        if FLAGS.path_specificities in rs_table:

            # Find corresponding indeces
            path_idx = np.where(rs_table == FLAGS.path_specificities)
            # len(path_idx[0]) > 0 and len(path_idx[1]) > 0for
            sub_idx = np.where(rs_table[:, np.where(rs_table == "subject")[1]] == str(FLAGS.subject))[0]
            trial_row = list(set(path_idx[0]) & set(sub_idx))[0]
            mvacc_col = np.where(rs_table == "mean_val_acc")[1][0]
            zlacc_col = np.where(rs_table == "meanline_acc")[1][0]
            mcvacc_col = np.where(rs_table == "mean_class_val_acc")[1][0]

            # Write in table
            rs_table[trial_row, [mvacc_col, zlacc_col, mcvacc_col]] = np.array(
                [np.round(np.mean(all_acc_val), 3), np.round(mean_line_acc, 3),
                 np.round(np.nanmean(rnd_all_acc_val_binary), 3) if FLAGS.task == "classification" else
                 rs_table[trial_row, mcvacc_col]])

            # Save table
            np.savetxt(fname=table_name, X=rs_table, delimiter=";", fmt="%s")

        else:
            cprint(f"There is no entry for this trial in {table_name.split('/')[-1]}", 'r')

    else:
        cprint(f"There is no Random/Narrow_Search_Table", 'r')

    # Save Prediction Matrices in File
    np.savetxt(sub_dir + f"{time.strftime('%Y_%m_%d_')}S{FLAGS.subject}_pred_matrix_{FLAGS.s_fold}"
    f"_folds_{FLAGS.path_specificities[:-1]}.csv", pred_matrix, delimiter=",")

    np.savetxt(sub_dir + f"{time.strftime('%Y_%m_%d_')}S{FLAGS.subject}_val_pred_matrix_{FLAGS.s_fold}"
    f"_folds_{FLAGS.path_specificities[:-1]}.csv", val_pred_matrix, delimiter=",")

    if FLAGS.shuffle:
        np.save(sub_dir + f"{time.strftime('%Y_%m_%d_')}S{FLAGS.subject}_shuffle_order_matrix"
        f"_{FLAGS.s_fold}_folds_{FLAGS.path_specificities[:-1]}.npy",
                shuffle_order_matrix)

    # Update bash files
    update_bashfiles(table_name=table_name, subject=FLAGS.subject, path_specs=FLAGS.path_specificities,
                     all_runs=False)


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
    fold_length = int(fulllength / sfold)  # len([0, ..., 26]) = 27

    # S-Fold-Index, e.g. s_idx=1. That is, FOLD 1 is validation set
    verge = s_idx * fold_length  # verge = 27

    if train:
        # if step=27 => fill_pos=54 (in FOLD 2)
        fill_pos = [step if step < verge else step + fold_length for step in current_batch]

    else:  # case of validation (use: batch_size=1, but works also with >1)
        fill_pos = [step + int(verge) for step in current_batch]
        # Check whether position is already filled
        for pos in fill_pos:
            if not np.isnan(current_mat[pred_idx, pos]):
                print(f"Overwrites position ({pred_idx},{pos}) in val_pred_matrix")

    # Update Matrix
    current_mat[pred_idx, fill_pos] = pred  # inference/prediction
    current_mat[rat_idx, fill_pos] = y  # ratings
    updated_mat = current_mat

    return updated_mat


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        cprint(key + ' : ' + str(value), "b")


def main(_):
    print_flags()

    if FLAGS.is_train:
        # Run main training
        train_lstm()

    else:
        pass

    if FLAGS.plot:
        # ["python3", "LSTM_pred_plot.py", Save_plots='True', Path specificities]
        subprocess.Popen(["python3", "LSTM_pred_plot.py", 'True', str(FLAGS.subject),
                          FLAGS.path_specificities,
                          str(FLAGS.dellog)])


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# %% Main: Run >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--is_train', type=str2bool, default=True,
                        help='Training or feature extraction')
    parser.add_argument('--seed', type=str2bool, default=False,
                        help='Random seed(42) either off or on')
    parser.add_argument('--sba', type=str2bool, default=True,
                        help="True for SBA; False for SA")
    parser.add_argument('--task', type=str, default=TASK_DEFAULT,
                        help="Either 'classification' or 'regression'")
    # parser.add_argument('--shuffle', type=bool, default=False) # This does not work: 'type=bool' (!)
    parser.add_argument('--shuffle', type=str2bool, default=False,
                        help="shuffle data (to have balance low/high arousal in all valsets of "
                             "classification task)")
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--repet_scalar', type=int, default=REPETITION_SCALAR_DEFAULT,
                        help='Number of times it should run through set. repet_scalar*(270 - 270/s_fold)')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--successive', type=int, default=SUCCESSIVE_BATCHES_DEFAULT,
                        help='Number of successive batches.')
    parser.add_argument('--successive_mode', type=int, default=SUCCESSIVE_MODE_DEFAULT,
                        help='Mode of successive batching, 1 or 2.')
    parser.add_argument('--path_specificities', type=str, default=PATH_SPECIFICITIES_DEFAULT,
                        help='Specificities for the paths (depending on model-setups)')
    parser.add_argument('--weight_reg', type=str, default=WEIGHT_REGULARIZER_DEFAULT,
                        help='Regularizer type for weights of fully-connected layers [none, l1, l2].')
    parser.add_argument('--weight_reg_strength', type=float, default=WEIGHT_REGULARIZER_STRENGTH_DEFAULT,
                        help='Regularizer strength for weights of fully-connected layers.')
    parser.add_argument('--activation_fct', type=str, default=ACTIVATION_FCT_DEFAULT,
                        help='Type of activation function from LSTM to fully-connected layers: elu, relu')
    parser.add_argument('--loss', type=str, default=LOSS_DEFAULT,
                        help='Type of loss. For now only: "normal".')
    parser.add_argument('--subject', type=int, default=SUBJECT_DEFAULT,
                        help='Which subject data to process')
    parser.add_argument('--condition', type=str, default=CONDITION_DEFAULT,
                        help="Which condition: 'nomov' (no movement) or 'mov'")
    parser.add_argument('--lstm_size', type=str, default=LSTM_SIZE_DEFAULT,
                        help='Comma separated list of size of hidden states in each LSTM layer')
    parser.add_argument('--balanced_cv', type=str2bool, default=True,
                        help='Balanced CV. False: at each iteration/fold data gets shuffled '
                             '(semi-balanced, this can lead to overlapping samples in validation set)')
    parser.add_argument('--s_fold', type=int, default=S_FOLD_DEFAULT,
                        help='Number of folds in S-Fold-Cross Validation')
    parser.add_argument('--rand_batch', type=str, default=True,
                        help='Whether random batch (True), or cronologically drawn batches (False)')
    parser.add_argument('--hilbert_power', type=str2bool, default=True,
                        help='Whether input is z-scored power extraction of SSD components '
                             '(via Hilbert transform)')
    parser.add_argument('--filetype', type=str, default=FILE_TYPE_DEFAULT,
                        help="Either 'SSD' or 'SPOC'")
    parser.add_argument('--band_pass', type=str2bool, default=True,
                        help='Whether to load (alpha-)band-passed SSD components')
    parser.add_argument('--summaries', type=str2bool, default=False,
                        help='Whether to write verbose summaries of tf variables')
    parser.add_argument('--fc_n_hidden', type=str, default=FC_NUM_HIDDEN_UNITS,
                        help="Comma separated list of N of hidden units in each FC layer")
    parser.add_argument('--plot', type=str2bool, default=True,
                        help="Whether to plot results and save them.")
    parser.add_argument('--dellog', type=str2bool, default=True,  # TODO rather False
                        help="Whether to delete log folders after plotting.")
    parser.add_argument('--component', type=str, default=COMPONENT_DEFAULT,
                        help="Which component: 'best', 'noise', 'random', 'all', "
                             "or comma separated list, e.g., 1,3,5")
    parser.add_argument('--hrcomp', type=str2bool, default=False,
                        help="Whether to attach the heart rate (HR) vector to neural components")
    parser.add_argument('--eqcompmat', type=int, default=None,
                        help="Provide N (int) of columns of input matrix which should be equal in size "
                             "across all tested conditions")
    parser.add_argument('--testmodel', type=str2bool, default=False,
                        help="Whether to test the model's learning ability with "
                             "inverse+stretch+noise ratings as input")

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run()
    end()

# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<  END

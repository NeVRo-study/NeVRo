# coding=utf-8
"""
Main script
    • run model

Author: Simon Hofmann | <[surname].[lastname][at]protonmail.com> | 2017
"""

# Adaptations if code is run under Python2
from __future__ import absolute_import
from __future__ import division  # int/int can result in float now, e.g. 1/2 = 0.5 (in python2 1/2=0, 1/2.=0.5)
from __future__ import print_function  # : Use print as a function as in Python 3: print()

from Load_Data import *
import numpy as np
import tensorflow as tf
import argparse
import time
import copy

from LSTMnet import LSTMnet

# TODO (time-)length per sample is hyperparameter: try also lengths >1sec(=250datapoins)

# TODO Define Default Values dependencies
LEARNING_RATE_DEFAULT = 1e-4  # 1e-2
BATCH_SIZE_DEFAULT = 1  # or bigger
RANDOM_BATCH_DEFAULT = True
S_FOLD_DEFAULT = 10
REPETITION_SCALAR_DEFAULT = 2  # scaler for how many times it should run through set (can be also fraction)
MAX_STEPS_DEFAULT = REPETITION_SCALAR_DEFAULT*(270 - 270/S_FOLD_DEFAULT)  # now it runs scalar-times throug whole set
EVAL_FREQ_DEFAULT = S_FOLD_DEFAULT - 1  # == MAX_STEPS_DEFAULT / (270/S_FOLD_DEFAULT)
CHECKPOINT_FREQ_DEFAULT = MAX_STEPS_DEFAULT
PRINT_FREQ_DEFAULT = int(MAX_STEPS_DEFAULT/8)  # if too low, uses much memory
OPTIMIZER_DEFAULT = 'ADAM'
WEIGHT_REGULARIZER_DEFAULT = 'l2'
WEIGHT_REGULARIZER_STRENGTH_DEFAULT = 0.18
ACTIVATION_FCT_DEFAULT = 'elu'
MARGIN_DEFAULT = 0.2
LOSS_DEFAULT = "normal"
FEAT_STEP_DEFAULT = CHECKPOINT_FREQ_DEFAULT-1
LSTM_SIZE_DEFAULT = 10
HILBERT_POWER_INPUT_DEFAULT = True
SUMMARIES_DEFAULT = True

SUBJECT_DEFAULT = 36

"""Lastly, all the weights are re-initialized (using the same random number generator used to initialize them 
    originally) or reset in some fashion to undo the learning that was done before moving on to the next set of 
    validation, training, and testing sets.
The idea behind cross validation is that each iteration is like training the algorithm from scratch. 
    This is desirable since by averaging your validation score, you get a more robust value. 
    It protects against the possibility of a biased validation set."""
# https://stackoverflow.com/questions/41216976/how-is-cross-validation-implemented

DATA_DIR_DEFAULT = '../../Data/'
SUB_DIR_DEFAULT = "./LSTM/" + "S{}/".format(str(SUBJECT_DEFAULT).zfill(2))
LOG_DIR_DEFAULT = './LSTM/logs/' + "S{}/".format(str(SUBJECT_DEFAULT).zfill(2))
CHECKPOINT_DIR_DEFAULT = './LSTM/checkpoints/' + "S{}".format(str(SUBJECT_DEFAULT).zfill(2))

WEIGHT_REGULARIZER_DICT = {'none': lambda x: None,  # No regularization
                           # L1 regularization
                           'l1': tf.contrib.layers.l1_regularizer,
                           # L2 regularization
                           'l2': tf.contrib.layers.l2_regularizer}

ACTIVATION_FCT_DICT = {'elu': tf.nn.elu,
                       'relu': tf.nn.relu}


def train_step(loss):
    """
    Defines the ops to conduct an optimization step. Optional: Implement Learning
    rate scheduler or pick optimizer here.

    Args:
        loss: scalar float Tensor, full loss = cross_entropy + reg_loss

    Returns:
        train_op: Ops for optimization.
    """

    train_op = None
    # TODO: Learning rate scheduler
    if OPTIMIZER_DEFAULT == 'ADAM':
        train_op = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss)

    return train_op


def train_lstm():
    """
    Performs training and evaluation of LSTM model.

    First define graph using class LSTM and its methods. Then define
    necessary operations such as trainer (train_step in this case), savers
    and summarizers. Finally, initialize model within a tf.Session and do the training.
    ---------------------------
    How to evaluate the model:
    ---------------------------
    Evaluation on test set should be conducted over full batch, i.e. 10k images,
    while it is alright to do it over mini-batch for train set.
    ---------------------------------
    How often to evaluate the model:
    ---------------------------------
    - on training set every print_freq iterations
    - on test set every eval_freq iterations
    ------------------------
    Additional:
    ------------------------
    Also takes snapshots of the model state (i.e. graph, weights and etc.)
    every checkpoint_freq iterations. For this, use tf.train.Saver class.
    checkout:
    [https://www.tensorflow.org/versions/r0.11/how_tos/variables/index.html]
    """

    # Set the random seeds for reproducibility. DO NOT CHANGE.
    tf.set_random_seed(42)
    np.random.seed(42)

    # Start TensorFlow Interactive Session
    # sess = tf.InteractiveSession()
    # s_fold_idx depending on s_fold and previous index
    s_fold_idx_list = np.arange(FLAGS.s_fold)
    np.random.shuffle(s_fold_idx_list)
    # rnd = 0

    # Create graph_dic
    graph_dict = dict.fromkeys(s_fold_idx_list)
    for key in graph_dict.keys():
        graph_dict[key] = tf.Graph()

    # Create to save the performance for each validation set
    # all_acc_val = tf.Variable(tf.zeros(shape=S_FOLD_DEFAULT, dtype=tf.float32, name="all_valid_accuracies"))
    all_acc_val = np.zeros(FLAGS.s_fold)  # case of non-tensor list

    # Find best component
    # TODO adapt model
    best_comp = best_component(subject=FLAGS.subject)

    # Load first data-set
    nevro_data = get_nevro_data(subject=FLAGS.subject,
                                component=best_comp,
                                s_fold_idx=s_fold_idx_list[0],
                                s_fold=FLAGS.s_fold,
                                cond="NoMov",
                                sba=True,
                                hilbert_power=FLAGS.hilbert_power)

    # Define graph using class LSTMnet and its methods:
    ddims = list(nevro_data["train"].eeg.shape[1:])  # [250, 2]
    full_length = len(nevro_data["validation"].ratings) * FLAGS.s_fold

    # Prediction Matrix for each fold
    # 1 Fold predic [0.156, ..., 0.491]
    # 1 Fold rating [0.161, ..., 0.423]
    # ...   ...   ...   ...   ...   ...
    # S Fold predic [0.397, ..., -0.134]
    # S Fold rating [0.412, ..., -0.983]

    pred_matrix = np.zeros(shape=(FLAGS.s_fold*2, full_length), dtype=np.float32)
    pred_matrix[np.where(pred_matrix == 0)] = np.nan  # set to NaN values
    # We create a separate matrix for the validation (which can be merged with the first pred_matrix later)
    val_pred_matrix = copy.copy(pred_matrix)

    # Set Variables for timer
    timer_fold_list = []  # list of duration(time) of each fold
    duration_fold = []  # duration of 1 fold
    # rest_duration = 0

    # Run through S-Fold-Cross-Validation (take the mean-performance across all validation sets)
    for rnd, s_fold_idx in enumerate(s_fold_idx_list):
        print("Train now on Fold-Nr.{}, that is, fold {}/{}".format(s_fold_idx, rnd+1, len(s_fold_idx_list)))
        start_timer_fold = datetime.datetime.now().replace(microsecond=0)

        # For each fold we need to define new graph to compare the validation accuracies of each fold in the end
        with tf.Session(graph=graph_dict[s_fold_idx]) as sess:  # This is a way to re-initialise the model completely

            with tf.variable_scope(name_or_scope="Fold_Nr{}/{}".format(str(rnd).zfill(len(str(FLAGS.s_fold))),
                                                                       len(s_fold_idx_list))):

                # Load Data:
                if rnd > 0:
                    # Show time passed per fold and estimation of rest time
                    print("Duration of previous fold {} [h:m:s]".format(duration_fold))
                    timer_fold_list.append(duration_fold)
                    # average over previous folds (np.mean(timer_fold_list) not possible in python2)
                    rest_duration_fold = average_time(timer_fold_list, in_timedelta=True) * (FLAGS.s_fold - rnd)
                    rest_duration_fold = chop_microseconds(delta=rest_duration_fold)
                    print("Estimated time to train the rest {} fold(s): {} [h:m:s]".format(FLAGS.s_fold - rnd,
                                                                                           rest_duration_fold))

                    nevro_data = get_nevro_data(subject=FLAGS.subject,
                                                component=best_comp,
                                                s_fold_idx=s_fold_idx,
                                                s_fold=FLAGS.s_fold,
                                                cond="NoMov",
                                                sba=True,
                                                hilbert_power=FLAGS.hilbert_power)

                with tf.name_scope("input"):
                    # shape = [None] + ddims includes num_steps = 250
                    #  Tensorflow requires input as a tensor (a Tensorflow variable) of the dimensions
                    # [batch_size, sequence_length, input_dimension] (a 3d variable).
                    x = tf.placeholder(dtype=tf.float32, shape=[None] + ddims, name="x-input")  # None for Batch-Size
                    # x = tf.placeholder(dtype=tf.float32, shape=[None, 250, 2], name="x-input")
                    y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="y-input")

                # Model
                lstm_model = LSTMnet(lstm_size=FLAGS.lstm_size,
                                     activation_function=ACTIVATION_FCT_DICT.get(FLAGS.activation_fct),
                                     weight_regularizer=WEIGHT_REGULARIZER_DICT.get(FLAGS.weight_reg)(
                                         scale=FLAGS.weight_reg_strength),
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

                train_writer = tf.summary.FileWriter(logdir=FLAGS.log_dir + str(s_fold_idx) + "/train",
                                                     graph=sess.graph)
                test_writer = tf.summary.FileWriter(logdir=FLAGS.log_dir + str(s_fold_idx) + "/val")
                # test_writer = tf.train.SummaryWriter(logdir=FLAGS.log_dir + "/test")

                # Saver
                # https://www.tensorflow.org/versions/r0.11/api_docs/python/state_ops.html#Saver
                saver = tf.train.Saver()  # might be under tf.initialize_all_variables().run()

                # Initialize your model within a tf.Session
                tf.global_variables_initializer().run()  # or without .run()

                def _feed_dict(training):
                    """creates feed_dicts depending on training or no training"""
                    # Train
                    if training:
                        xs, ys = nevro_data["train"].next_batch(batch_size=FLAGS.batch_size,
                                                                randomize=FLAGS.rand_batch)
                        # print("that is the x-shape:", xs.shape)
                        # print("I am in _feed_dict(Trainining True)")
                        # keep_prob = 1.0-FLAGS.dropout_rate
                        ys = np.reshape(ys, newshape=([FLAGS.batch_size] + list(ys.shape)))

                    else:
                        # Validation:
                        xs, ys = nevro_data["validation"].next_batch(batch_size=FLAGS.batch_size,
                                                                     randomize=FLAGS.rand_batch)
                        ys = np.reshape(ys, newshape=([FLAGS.batch_size] + list(ys.shape)))

                    return {x: xs, y: ys}

                # RUN
                val_counter = 0  # Needed when validation should only be run in the end of training

                # Init Lists of Accuracy and Loss
                train_loss_list = []
                train_acc_list = []
                val_acc_list = []
                val_loss_list = []

                # Set Variables for timer
                timer_list = []  # # list of duration(time) of 100 steps
                # duration = []  # durations of 100 iterations
                start_timer = 0
                # end_timer = 0
                timer_freq = 100

                for step in range(int(FLAGS.max_steps)):

                    # Timer for every timer_freq=100 steps
                    if step == 0:
                        # Set Start Timer
                        start_timer = datetime.datetime.now().replace(microsecond=0)

                    if step % (timer_freq/2) == 0.:
                        print("Step {}/{} in Fold Nr.{} ({}/{})".format(step, FLAGS.max_steps, s_fold_idx,
                                                                        rnd+1, len(s_fold_idx_list)))

                    # Evaluate on training set every print_freq (=10) iterations
                    if (step + 1) % FLAGS.print_freq == 0:
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()

                        summary, _, train_loss, train_acc, tain_infer, train_y = sess.run([merged, optimization, loss,
                                                                                           accuracy, infer, y],
                                                                                          feed_dict=_feed_dict(
                                                                                              training=True),
                                                                                          options=run_options,
                                                                                          run_metadata=run_metadata)

                        train_writer.add_run_metadata(run_metadata, "step{}".format(str(step).zfill(4)))
                        train_writer.add_summary(summary=summary, global_step=step)

                        print("\nTrain-Loss: {:.3f} at step:{}".format(np.round(train_loss, 3), step + 1))
                        print("Train-Accuracy: {:.3f} at step:{}\n".format(np.round(train_acc, 3), step + 1))
                        # Update Lists
                        train_acc_list.append(train_acc)
                        train_loss_list.append(train_loss)

                    else:
                        # summary, _, train_loss, train_acc = sess.run([merged, optimization, loss, accuracy],
                        #                                              feed_dict=_feed_dict(training=True))
                        # train_writer.add_summary(summary=summary, global_step=step)

                        _, train_loss, train_acc, tain_infer, train_y = sess.run([optimization, loss, accuracy,
                                                                                  infer, y],
                                                                                 feed_dict=_feed_dict(training=True))

                        # Update Lists
                        train_acc_list.append(train_acc)
                        train_loss_list.append(train_loss)

                    # Write train_infer & train_y in prediction matrix
                    pred_matrix = fill_pred_matrix(pred=tain_infer, y=train_y, current_mat=pred_matrix,
                                                   sfold=FLAGS.s_fold, s_idx=s_fold_idx,
                                                   current_batch=nevro_data["train"].current_batch, train=True)

                    # Evaluate on validation set every eval_freq iterations
                    if (step + 1) % FLAGS.eval_freq == 0:
                        val_counter += 1  # count the number of validation steps (implementation could improved)
                        if step == FLAGS.max_steps - 1:  # Validation in last round
                            val_counter /= FLAGS.repet_scalar
                            assert float(val_counter).is_integer(), \
                                "val_counter (={}) devided by repetition (={}) is not integer".format(
                                    val_counter, FLAGS.repet_scalar)

                            for val_step in range(int(val_counter)):
                                summary, val_loss, val_acc, val_infer, val_y = sess.run([merged, loss, accuracy, infer,
                                                                                         y],
                                                                                        feed_dict=_feed_dict(
                                                                                            training=False))
                                # test_loss, test_acc = sess.run([loss, accuracy], feed_dict=_feed_dict(training=False))
                                # print("now do: test_writer.add_summary(summary=summary, global_step=step)")
                                test_writer.add_summary(summary=summary, global_step=step)
                                # print("Validation-Loss: {} at step:{}".format(np.round(val_loss, 3), step + 1))
                                if val_step % 5 == 0:
                                    print("Validation-Loss: {:.3f} of Fold Nr.{} ({}/{})".format(np.round(val_loss, 3),
                                                                                                 s_fold_idx, rnd + 1,
                                                                                                 len(s_fold_idx_list)))
                                    # print("Validation-Accuracy: {} at step:{}".format(np.round(val_acc, 3), step + 1))
                                    print("Validation-Accuracy: {:.3f} of "
                                          "Fold Nr.{} ({}/{})".format(np.round(val_acc, 3),
                                                                      s_fold_idx, rnd+1,
                                                                      len(s_fold_idx_list)))
                                # Update Lists
                                val_acc_list.append(val_acc)
                                val_loss_list.append(val_loss)

                                # Write val_infer & val_y in val_pred_matrix
                                val_pred_matrix = fill_pred_matrix(pred=val_infer, y=val_y, current_mat=val_pred_matrix,
                                                                   sfold=FLAGS.s_fold, s_idx=s_fold_idx,
                                                                   current_batch=nevro_data["validation"].current_batch,
                                                                   train=False)

                    # Save the variables to disk every checkpoint_freq (=5000) iterations
                    if (step + 1) % FLAGS.checkpoint_freq == 0:
                        save_path = saver.save(sess=sess, save_path=FLAGS.checkpoint_dir + "/lstmnet_rnd{}.ckpt".format(
                                                   str(rnd).zfill(2)), global_step=step)
                        print("Model saved in file: %s" % save_path)

                    # End Timer
                    if step % timer_freq == 0 and step > 0:
                        end_timer = datetime.datetime.now().replace(microsecond=0)

                        # Calculate Duration and Estimations
                        duration = end_timer - start_timer
                        timer_list.append(duration)  # mean(timer_list) = average time per 100steps
                        # For this fold
                        # Cannot take mean(daytime) in python2
                        # estim_t_per_step = np.mean(timer_list) / timer_freq  # only python3
                        mean_timer_list = average_time(list_of_timestamps=timer_list, in_timedelta=False)
                        estim_t_per_step = mean_timer_list/timer_freq
                        remaining_steps_in_fold = (FLAGS.max_steps - (step + 2))
                        rest_duration = remaining_steps_in_fold * estim_t_per_step
                        # For whole training
                        remaining_folds = len(s_fold_idx_list) - (rnd + 1)
                        if rnd == 0:
                            remaining_steps = FLAGS.max_steps * remaining_folds
                            rest_duration_all_folds = rest_duration + remaining_steps*estim_t_per_step
                        else:  # this is more accurate, but only possible after first round(rnd)/fold
                            rest_duration_all_folds = rest_duration + \
                                                      average_time(timer_fold_list, in_timedelta=False)*remaining_folds
                        # convert back to: datetime.timedelta(seconds=27)
                        rest_duration = datetime.timedelta(seconds=rest_duration)
                        rest_duration_all_folds = datetime.timedelta(seconds=rest_duration_all_folds)
                        # Remove microseconds
                        rest_duration = chop_microseconds(delta=rest_duration)
                        rest_duration_all_folds = chop_microseconds(delta=rest_duration_all_folds)

                        print("Time passed to train {} steps: {} [h:m:s]".format(timer_freq, duration))
                        print("Estimated time to train the rest {} steps in current Fold-Nr.{}: {} [h:m:s]".format(
                            int(FLAGS.max_steps - (step + 1)), s_fold_idx, rest_duration))
                        print("Estimated time to train the rest steps and {} {}: {} [h:m:s]".format(
                            remaining_folds, "folds" if remaining_folds > 1 else "fold", rest_duration_all_folds))

                        # Set Start Timer
                        start_timer = datetime.datetime.now().replace(microsecond=0)

                # Close Writers:
                train_writer.close()
                test_writer.close()

                # Save last val_acc in all_acc_val-vector
                all_acc_val[rnd] = np.nanmean(val_acc_list)  # since we validate in the end of train average across all
                # all_acc_val[rnd] = val_acc  # Save last val_acc in all_acc_val-vector
                # all_acc_val = all_acc_val[rnd].assign(val_acc)  # if all_acc_val is Tensor variable

                # Save loss_ & acc_lists externally per S-Fold
                loss_acc_lists = [train_loss_list, train_acc_list, val_acc_list, val_loss_list]
                loss_acc_lists_names = ["train_loss_list", "train_acc_list", "val_acc_list", "val_loss_list"]
                for list_idx, liste in enumerate(loss_acc_lists):
                    with open(FLAGS.log_dir + str(s_fold_idx) + "/{}.txt".format(loss_acc_lists_names[list_idx]), "w") \
                            as list_file:
                        for value in liste:
                            list_file.write(str(value) + "\n")

            # Fold End Timer
            end_timer_fold = datetime.datetime.now().replace(microsecond=0)
            duration_fold = end_timer_fold - start_timer_fold

    # Final Accuracy & Time
    timer_fold_list.append(duration_fold)
    print("Time to train all folds (each {} steps): {} [h:m:s]".format(FLAGS.max_steps, np.sum(timer_fold_list)))
    print("Average accuracy across all {} validation set: {}".format(FLAGS.s_fold, np.mean(all_acc_val)))

    # Save training information in Textfile
    with open(FLAGS.sub_dir + "{}S{}_accuracy_across_{}_folds.txt".format(time.strftime('%y_%m_%d_'),
                                                                          FLAGS.subject,
                                                                          FLAGS.s_fold), "w") as file:
        file.write("Subject {}\nHilbert_z-Power: {}\ns-Fold: {}\nmax_step: {}\nrepetition_set: {}\nlearning_rate: {}"
                   "\nbatch_size: {}\nbatch_random: {}\nweight_reg: {}({})\nact_fct: {}"
                   "\nlstm_h_size: {}\n".format(FLAGS.subject, FLAGS.hilbert_power, FLAGS.s_fold, int(FLAGS.max_steps),
                                                FLAGS.repet_scalar, FLAGS.learning_rate, FLAGS.batch_size,
                                                FLAGS.rand_batch, FLAGS.weight_reg, FLAGS.weight_reg_strength,
                                                FLAGS.activation_fct, FLAGS.lstm_size))
        for i, item in enumerate([s_fold_idx_list, all_acc_val, np.mean(all_acc_val)]):
            file.write(["S-Fold(Round): ", "Validation-Acc: ", "mean(Accuracy): "][i] + str(item)+"\n")

    # Save Prediction Matrices in File
    np.savetxt(FLAGS.sub_dir + "{}S{}_pred_matrix_{}_folds.csv".format(time.strftime('%y_%m_%d_'), FLAGS.subject,
                                                                       FLAGS.s_fold), pred_matrix, delimiter=",")

    np.savetxt(FLAGS.sub_dir + "{}S{}_val_pred_matrix_{}_folds.csv".format(time.strftime('%y_%m_%d_'), FLAGS.subject,
                                                                           FLAGS.s_fold),
               val_pred_matrix, delimiter=",")


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

    assert FLAGS.batch_size == 1, "Filling pred_matrix only works for batch_size=1 (currently)"

    step = int(current_batch)

    pred_idx = int(s_idx * 2)
    rat_idx = int(pred_idx + 1)

    full_length = len(current_mat[0, :])  # 270
    # S-Fold, for S=10:
    #    FOLD 0          FOLD 1          FOLD 2                FOLD 9
    # [0, ..., 26] | [27, ..., 53] | [54, ..., 80] | ... | [235, ..., 269]
    fold_length = int(full_length/sfold)  # len([0, ..., 26]) = 27
    # S-Fold-Index, e.g. s_idx=1. That is, FOLD 1 is validation set
    verge = s_idx * fold_length  # verge = 27

    if train:
        fill_pos = step if step < verge else step + fold_length  # if step=27 => fill_pos=54 (in FOLD 2)
        # if not np.isnan(current_mat[pred_idx, fill_pos]):
        #     print("Overwrites position in pred_matrix")

    else:  # case of validation
        fill_pos = int(verge) + step
        # Check whether position is already filled
        # if not np.isnan(current_mat[pred_idx, fill_pos]):
        #     print("Overwrites position in val_pred_matrix")

    current_mat[pred_idx, fill_pos] = pred  # inference/prediction
    current_mat[rat_idx, fill_pos] = y      # ratings
    updated_mat = current_mat

    return updated_mat


def initialize_folders():
    """
    Initializes all folders in FLAGS variable.
    """
    if not tf.gfile.Exists(FLAGS.sub_dir):
        tf.gfile.MakeDirs(FLAGS.sub_dir)

    if not tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.MakeDirs(FLAGS.log_dir)

    if not tf.gfile.Exists(FLAGS.data_dir):
        tf.gfile.MakeDirs(FLAGS.data_dir)

    if not tf.gfile.Exists(FLAGS.checkpoint_dir):
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main(_):
    print_flags()

    initialize_folders()

    print("FLAGS.is_train is boolean:", isinstance(FLAGS.is_train, bool))

    # if eval(FLAGS.is_train):
    if FLAGS.is_train:
        if FLAGS.train_model == 'lstm':
            train_lstm()
        else:
            raise ValueError("--train_model argument can be 'lstm'")
    else:
        pass
        # print("I run now feature_extraction()")
        # feature_extraction(layer=FLAGS.layer_feat_extr)


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--repet_scalar', type=int, default=REPETITION_SCALAR_DEFAULT,
                        help='Number of times it should run through set. repet_scalar*(270 - 270/s_fold)')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--print_freq', type=int, default=PRINT_FREQ_DEFAULT,
                        help='Frequency of printing and saving in log on the train set')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--checkpoint_freq', type=int, default=CHECKPOINT_FREQ_DEFAULT,
                        help='Frequency with which the model state is saved.')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    parser.add_argument('--log_dir', type=str, default=LOG_DIR_DEFAULT,
                        help='Summaries log directory')
    parser.add_argument('--sub_dir', type=str, default=SUB_DIR_DEFAULT,
                        help='Summaries per Subject')
    parser.add_argument('--checkpoint_dir', type=str, default=CHECKPOINT_DIR_DEFAULT,
                        help='Checkpoint directory')
    parser.add_argument('--is_train', type=str, default=True,
                        help='Training or feature extraction')
    parser.add_argument('--train_model', type=str, default='lstm',
                        help='Type of model. Possible option(s): lstm')
    parser.add_argument('--weight_reg', type=str, default=WEIGHT_REGULARIZER_DEFAULT,
                        help='Regularizer type for weights of fully-connected layers [none, l1, l2].')
    parser.add_argument('--weight_reg_strength', type=float, default=WEIGHT_REGULARIZER_STRENGTH_DEFAULT,
                        help='Regularizer strength for weights of fully-connected layers.')
    parser.add_argument('--activation_fct', type=str, default=ACTIVATION_FCT_DEFAULT,
                        help='Type of activation function from lstm to fully-connected layers [elu, relu].')
    parser.add_argument('--margin', type=float, default=MARGIN_DEFAULT,
                        help='Regularizer strength for weights of fully-connected layers.')
    parser.add_argument('--loss', type=str, default=LOSS_DEFAULT,
                        help='Type of loss. Either "normal" or "Hadsell".')
    parser.add_argument('--feat_ext_step', type=str, default=FEAT_STEP_DEFAULT,
                        help='feature_extraction will be applied on specific step of checkpoint data')
    parser.add_argument('--subject', type=int, default=SUBJECT_DEFAULT,
                        help='Which subject data to process')
    parser.add_argument('--lstm_size', type=int, default=LSTM_SIZE_DEFAULT,
                        help='size of hidden state in LSTM layer')
    parser.add_argument('--s_fold', type=int, default=S_FOLD_DEFAULT,
                        help='Number of folds in S-Fold-Cross Validation')
    parser.add_argument('--rand_batch', type=str, default=RANDOM_BATCH_DEFAULT,
                        help='Whether random batch (True), or cronologically drawn batches (False)')
    parser.add_argument('--hilbert_power', type=str, default=HILBERT_POWER_INPUT_DEFAULT,
                        help='Whether input is z-scored power extraction of SSD components (via Hilbert transform)')
    parser.add_argument('--summaries', type=str, default=SUMMARIES_DEFAULT,
                        help='Whether to write summaries of tf variables')

    # parser.add_argument('--layer_feat_extr', type=str, default="fc2",
    #                     help='Choose layer for feature extraction')

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run()

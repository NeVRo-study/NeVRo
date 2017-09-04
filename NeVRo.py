# coding=utf-8
"""
Main script
    • run model

Author: Simon Hofmann | <[surname].[lastname][at]protonmail.com> | 2017
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import Load_Data
import numpy as np
import tensorflow as tf
import argparse

from LSTMnet import LSTMnet


# TODO Define Default Values
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 1  # or bigger
MAX_STEPS_DEFAULT = 15000
EVAL_FREQ_DEFAULT = 1000
CHECKPOINT_FREQ_DEFAULT = 5000
PRINT_FREQ_DEFAULT = 10
OPTIMIZER_DEFAULT = 'ADAM'
WEIGHT_REGULARIZER_DEFAULT = 'l2'
WEIGHT_REGULARIZER_STRENGTH_DEFAULT = 0.0
MARGIN_DEFAULT = 0.2
LOSS_DEFAULT = "normal"
FEAT_EPOCH_DEFAULT = CHECKPOINT_FREQ_DEFAULT-1

SUBJECT_DEFAULT = 36
S_FOLD_DEFAULT = 10

DATA_DIR_DEFAULT = '../../Data/'
LOG_DIR_DEFAULT = '../../Data/LSTM/logs/'
CHECKPOINT_DIR_DEFAULT = '../../Data/LSTM/checkpoints'

WEIGHT_REGULARIZER_DICT = {'none': lambda x: None,  # No regularization
                           # L1 regularization
                           'l1': tf.contrib.layers.l1_regularizer,
                           # L2 regularization
                           'l2': tf.contrib.layers.l2_regularizer}


def train_step(loss):
    """
    Defines the ops to conduct an optimization step. Optional: Implement Learning
    rate scheduler or pick optimizer here.

    Args:
        loss: scalar float Tensor, full loss = cross_entropy + reg_loss

    Returns:
        train_op: Ops for optimization.
    """

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

    # Start TensorFlow Session
    sess = tf.InteractiveSession()

    # s_fold_idx depending on s_fold and previous index
    s_fold_idx_list = np.arange(FLAGS.s_fold)
    np.random.shuffle(s_fold_idx_list)

    # Run through S-Fold-Cross-Validation (take the mean-performance across all validation sets)
    for rnd, s_fold_idx in enumerate(s_fold_idx_list):
        # Load Data:
        nevro_data = Load_Data.get_nevro_data(subject=FLAGS.subject,
                                              s_fold_idx=s_fold_idx_list[rnd],
                                              s_fold=FLAGS.s_fold,
                                              cond="NoMov",
                                              sba=True)

        # Define graph using class LSTMnet and its methods:
        ddims = list(nevro_data["train"].eeg.shape[1:])  # [250, 2]
        # print("ddims", ddims)

        with tf.name_scope("input"):
            # shape = [None] + ddims includes num_steps = 250
            #  Tensorflow requires input as a tensor (a Tensorflow variable) of the dimensions
            # [batch_size, sequence_length, input_dimension] (a 3d variable).
            x = tf.placeholder(dtype=tf.float32, shape=[None] + ddims, name="x-input")  # None for Batch-Size
            # x = tf.placeholder(dtype=tf.float32, shape=[None, 250, 2], name="x-input")
            y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="y-input")

        # Model
        lstm_model = LSTMnet(weight_regularizer=WEIGHT_REGULARIZER_DICT.get(
            FLAGS.weight_reg)(scale=FLAGS.weight_reg_strength))

        # Prediction
        with tf.variable_scope(name_or_scope="Inference") as scope:
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

        train_writer = tf.summary.FileWriter(logdir=FLAGS.log_dir + "/train", graph=sess.graph)
        test_writer = tf.summary.FileWriter(logdir=FLAGS.log_dir + "/val")
        # test_writer = tf.train.SummaryWriter(logdir=FLAGS.log_dir + "/test")

        # Saver
        # https://www.tensorflow.org/versions/r0.11/api_docs/python/state_ops.html#Saver
        saver = tf.train.Saver()  # might be under tf.initialize_all_variables().run()

        # Initialize your model within a tf.Session
        tf.initialize_all_variables().run()  # or without .run()

        def _feed_dict(training):
            """creates feed_dicts depending on training or no training"""
            # Train
            if training:
                xs, ys = nevro_data["train"].next_batch(FLAGS.batch_size)  # BATCH_SIZE_DEFAULT
                # print("that is the x-shape:", xs.shape)
                # keep_prob = 1.0-FLAGS.dropout_rate
                # print("I am in _feed_dict(Trainining True)")

            else:
                # Validation:
                xs, ys = nevro_data["validation"].eeg, nevro_data["validation"].ratings

            return {x: xs, y: ys}

        # Run
        for epoch in range(FLAGS.max_steps):
            # Evaluate on training set every print_freq (=10) iterations
            if (epoch + 1) % FLAGS.print_freq == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

                summary, _, train_loss, train_acc = sess.run([merged, optimization, loss, accuracy],
                                                             feed_dict=_feed_dict(training=True),
                                                             options=run_options,
                                                             run_metadata=run_metadata)

                train_writer.add_run_metadata(run_metadata, "step{}".format(str(epoch).zfill(4)))
                train_writer.add_summary(summary=summary, global_step=epoch)

                print("Train-Loss: {} at epoch:{}".format(np.round(train_loss, 3), epoch + 1))
                print("Train-Accuracy: {} at epoch:{}".format(np.round(train_acc, 3), epoch + 1))

            else:
                # summary, _, train_loss, train_acc = sess.run([merged, optimization, loss, accuracy],
                #                                              feed_dict=_feed_dict(training=True))
                # train_writer.add_summary(summary=summary, global_step=epoch)

                _, train_loss, train_acc = sess.run([optimization, loss, accuracy], feed_dict=_feed_dict(training=True))

            # Evaluate on validation set every eval_freq (=1000) iterations
            if (epoch + 1) % FLAGS.eval_freq == 0:
                summary, val_loss, val_acc = sess.run([merged, loss, accuracy], feed_dict=_feed_dict(training=False))
                # test_loss, test_acc = sess.run([loss, accuracy], feed_dict=_feed_dict(training=False))
                # print("now do: test_writer.add_summary(summary=summary, global_step=epoch)")
                test_writer.add_summary(summary=summary, global_step=epoch)
                print("Validation-Loss: {} at epoch:{}".format(np.round(val_loss, 3), epoch + 1))
                print("Validation-Accuracy: {} at epoch:{}".format(np.round(val_acc, 3), epoch + 1))

            # Save the variables to disk every checkpoint_freq (=5000) iterations
            if (epoch + 1) % FLAGS.checkpoint_freq == 0:
                save_path = saver.save(sess=sess, save_path=FLAGS.checkpoint_dir + "/lstmnet.ckpt", global_step=epoch)
                print("Model saved in file: %s" % save_path)

        # Close Writers:
        train_writer.close()
        test_writer.close()


def initialize_folders():
    """
    Initializes all folders in FLAGS variable.
    """

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
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--print_freq', type=int, default=PRINT_FREQ_DEFAULT,
                        help='Frequency of evaluation on the train set')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--checkpoint_freq', type=int, default=CHECKPOINT_FREQ_DEFAULT,
                        help='Frequency with which the model state is saved.')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    parser.add_argument('--log_dir', type=str, default=LOG_DIR_DEFAULT,
                        help='Summaries log directory')
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
    parser.add_argument('--margin', type=float, default=MARGIN_DEFAULT,
                        help='Regularizer strength for weights of fully-connected layers.')
    parser.add_argument('--loss', type=str, default=LOSS_DEFAULT,
                        help='Type of loss. Either "normal" or "Hadsell".')
    parser.add_argument('--feat_ext_epoch', type=str, default=FEAT_EPOCH_DEFAULT,
                        help='feature_extraction will be applied on specific epoch of checkpoint data')
    parser.add_argument('--subject', type=int, default=SUBJECT_DEFAULT,
                        help='Which subject data to process')
    parser.add_argument('--s_fold', type=int, default=S_FOLD_DEFAULT,
                        help='Number of folds in S-Fold-Cross Validation')
    # parser.add_argument('--layer_feat_extr', type=str, default="fc2",
    #                     help='Choose layer for feature extraction')

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run()


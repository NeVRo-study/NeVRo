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

import Load_Data
import numpy as np
import tensorflow as tf
import argparse

from LSTMnet import LSTMnet


# TODO Define Default Values
LEARNING_RATE_DEFAULT = 1e-2  # 1e-4
BATCH_SIZE_DEFAULT = 1  # or bigger
MAX_STEPS_DEFAULT = 150
EVAL_FREQ_DEFAULT = MAX_STEPS_DEFAULT/15
CHECKPOINT_FREQ_DEFAULT = MAX_STEPS_DEFAULT/3
PRINT_FREQ_DEFAULT = 5
OPTIMIZER_DEFAULT = 'ADAM'
WEIGHT_REGULARIZER_DEFAULT = 'l2'
WEIGHT_REGULARIZER_STRENGTH_DEFAULT = 0.18
ACTIVATION_FCT_DEFAULT = 'elu'  # TODO HyperParameter, to be tuned
MARGIN_DEFAULT = 0.2
LOSS_DEFAULT = "normal"
FEAT_EPOCH_DEFAULT = CHECKPOINT_FREQ_DEFAULT-1
LSTM_SIZE_DEFAULT = 10  # TODO HyperParameter, to be tuned

SUBJECT_DEFAULT = 36
S_FOLD_DEFAULT = 10
"""Lastly, all the weights are re-initialized (using the same random number generator used to initialize them 
    originally) or reset in some fashion to undo the learning that was done before moving on to the next set of 
    validation, training, and testing sets.
The idea behind cross validation is that each iteration is like training the algorithm from scratch. 
    This is desirable since by averaging your validation score, you get a more robust value. 
    It protects against the possibility of a biased validation set."""
# https://stackoverflow.com/questions/41216976/how-is-cross-validation-implemented

DATA_DIR_DEFAULT = '../../Data/'
LOG_DIR_DEFAULT = './LSTM/logs/'
CHECKPOINT_DIR_DEFAULT = './LSTM/checkpoints/'

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
    all_acc_val = np.zeros(S_FOLD_DEFAULT)  # case of non-tensor list

    # Run through S-Fold-Cross-Validation (take the mean-performance across all validation sets)
    for rnd, s_fold_idx in enumerate(s_fold_idx_list):

        print("Train now on Fold-Nr.{}/{}".format(rnd+1, len(s_fold_idx_list)))

        with tf.Session(graph=graph_dict[s_fold_idx]) as sess:
            # This is a way to re-initialise the model completely
            # Alternative just reset the weights after one round (maybe with tf.reset_default_graph())
            with tf.variable_scope(name_or_scope="Round{}".format(str(rnd).zfill(len(str(FLAGS.s_fold))))):

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
                lstm_model = LSTMnet(lstm_size=FLAGS.lstm_size,
                                     activation_function=ACTIVATION_FCT_DICT.get(FLAGS.activation_fct),
                                     weight_regularizer=WEIGHT_REGULARIZER_DICT.get(FLAGS.weight_reg)(
                                         scale=FLAGS.weight_reg_strength),
                                     n_steps=ddims[0], batch_size=FLAGS.batch_size)  # n_step = 250

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

                train_writer = tf.summary.FileWriter(logdir=FLAGS.log_dir + "/train", graph=sess.graph)
                test_writer = tf.summary.FileWriter(logdir=FLAGS.log_dir + "/val")
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
                        xs, ys = nevro_data["train"].next_batch(FLAGS.batch_size)
                        # print("that is the x-shape:", xs.shape)
                        # print("I am in _feed_dict(Trainining True)")
                        # keep_prob = 1.0-FLAGS.dropout_rate
                        ys = np.reshape(ys, newshape=([FLAGS.batch_size] + list(ys.shape)))

                    else:
                        # Validation:
                        xs, ys = nevro_data["validation"].next_batch(FLAGS.batch_size)
                        ys = np.reshape(ys, newshape=([FLAGS.batch_size] + list(ys.shape)))

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

                        _, train_loss, train_acc = sess.run([optimization, loss, accuracy],
                                                            feed_dict=_feed_dict(training=True))

                    # Evaluate on validation set every eval_freq (=1000) iterations
                    if (epoch + 1) % FLAGS.eval_freq == 0:
                        summary, val_loss, val_acc = sess.run([merged, loss, accuracy],
                                                              feed_dict=_feed_dict(training=False))
                        # test_loss, test_acc = sess.run([loss, accuracy], feed_dict=_feed_dict(training=False))
                        # print("now do: test_writer.add_summary(summary=summary, global_step=epoch)")
                        test_writer.add_summary(summary=summary, global_step=epoch)
                        print("Validation-Loss: {} at epoch:{}".format(np.round(val_loss, 3), epoch + 1))
                        print("Validation-Accuracy: {} at epoch:{}".format(np.round(val_acc, 3), epoch + 1))

                    # Save the variables to disk every checkpoint_freq (=5000) iterations
                    if (epoch + 1) % FLAGS.checkpoint_freq == 0:
                        save_path = saver.save(sess=sess, save_path=FLAGS.checkpoint_dir + "/lstmnet_rnd{}.ckpt".format(
                                                   str(rnd).zfill(2)), global_step=epoch)
                        print("Model saved in file: %s" % save_path)

                # Close Writers:
                train_writer.close()
                test_writer.close()

                # Save last val_acc in all_acc_val-vector
                all_acc_val[rnd] = val_acc
                # all_acc_val = all_acc_val[rnd].assign(val_acc)  # if all_acc_val is Tensor variable

    print("Average accuracy across all {} validation set: {}".format(FLAGS.s_fold, np.mean(all_acc_val)))

    # Save information in Textfile
    with open("./LSTM/S{}_accuracy_across_{}_folds.txt".format(FLAGS.subject, FLAGS.s_fold), "w") as file:
        file.write("Subject {}\ns-Fold: {}\nmax_step: {}\nlearning_rate: {}\nweight_reg: {}({})\nact_fct: {}"
                   "\nlstm_h_size: {}\n".format(FLAGS.subject, FLAGS.s_fold, FLAGS.max_steps, FLAGS.learning_rate,
                                                FLAGS.weight_reg, FLAGS.weight_reg_strength, FLAGS.activation_fct,
                                                FLAGS.lstm_size))
        for i, item in enumerate([s_fold_idx_list, all_acc_val, np.mean(all_acc_val)]):
            file.write(["S-Fold(Round): ", "Validation-Acc: ", "mean(Accuracy): "][i] + str(item)+"\n")


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
    parser.add_argument('--activation_fct', type=str, default=ACTIVATION_FCT_DEFAULT,
                        help='Type of activation function from lstm to fully-connected layers [elu, relu].')
    parser.add_argument('--margin', type=float, default=MARGIN_DEFAULT,
                        help='Regularizer strength for weights of fully-connected layers.')
    parser.add_argument('--loss', type=str, default=LOSS_DEFAULT,
                        help='Type of loss. Either "normal" or "Hadsell".')
    parser.add_argument('--feat_ext_epoch', type=str, default=FEAT_EPOCH_DEFAULT,
                        help='feature_extraction will be applied on specific epoch of checkpoint data')
    parser.add_argument('--subject', type=int, default=SUBJECT_DEFAULT,
                        help='Which subject data to process')
    parser.add_argument('--lstm_size', type=int, default=LSTM_SIZE_DEFAULT,
                        help='size of hidden state in LSTM layer')
    parser.add_argument('--s_fold', type=int, default=S_FOLD_DEFAULT,
                        help='Number of folds in S-Fold-Cross Validation')
    # parser.add_argument('--layer_feat_extr', type=str, default="fc2",
    #                     help='Choose layer for feature extraction')

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run()


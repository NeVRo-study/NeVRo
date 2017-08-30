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


# TODO Define Default Values and Dirs
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 128
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

# TODO Dirs
DATA_DIR_DEFAULT = './LSTM/LSTM-batches-py'
LOG_DIR_DEFAULT = './logs/LSTM'
CHECKPOINT_DIR_DEFAULT = './checkpoints'

WEIGHT_REGULARIZER_DICT = {'none': lambda x: None,  # No regularization
                           # L1 regularization
                           'l1': tf.contrib.layers.l1_regularizer,
                           # L2 regularization
                           'l2': tf.contrib.layers.l2_regularizer}


def create_s_fold_idx(s_folds, list_prev_indices=[]):

    if not list_prev_indices:  # list_prev_indices == []
        s_fold_idx = np.random.randint(0, s_folds)
    else:
        choose_from = list(set(range(s_folds)) - set(list_prev_indices))
        s_fold_idx = np.random.choice(choose_from, 1)[0]

    list_prev_indices.append(s_fold_idx)

    return s_fold_idx, list_prev_indices


# s_idx, list_indices = create_s_fold_idx(s_folds=10)
# for _ in range(9):
#     s_idx, list_indices = create_s_fold_idx(s_folds=10, list_prev_indices=list_indices)
# len(list_indices)

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
    s_fold_idx_list = np.arange(0, FLAGS.s_fold)
    np.random.shuffle(s_fold_idx_list)

    # Run through S-Fold-Cross-Validation (take the mean-performance across all validation sets)
    for rnd, s_fold_idx in enumerate(s_fold_idx_list):
        # Load Data:
        nevro_data = Load_Data.get_nevro_data(subject=FLAGS.subject,
                                              s_fold_idx=s_fold_idx_list[rnd],
                                              s_fold=FLAGS.s_fold,
                                              cond="NoMov",
                                              sba=True)
        #TODO continue here...
        ddims = list(nevro_data.train.images.shape[1:])  # [32,32,3]


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
    parser.add_argument('--train_model', type=str, default='linear',
                        help='Type of model. Possible options: linear and siamese')
    parser.add_argument('--layer_feat_extr', type=str, default="fc2",
                        help='Choose layer for feature extraction')
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

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run()


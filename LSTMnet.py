"""
Build LSTM architecture
Author: Simon Hofmann | <[surname].[lastname][at]protonmail.com> | 2017
"""

import tensorflow as tf
# import numpy as np


class LSTMnet:
    """
    This class implements a LSTM neural network in TensorFlow.
    It incorporates a certain graph model to be trained and to be used
    in inference.
    """

    def __init__(self, n_classes, weight_regularizer=tf.contrib.layers.l2_regularizer(scale=0.18)):
        """
        Constructor for an LSTMnet object.
        Args:
            n_classes: int, number of classes of the classification problem. This number is required in order to specify
            the output dimensions of the LSTMnet.
            weight_regularizer: to be applied weight regularization
        """

        self.n_classes = n_classes
        self.fc2_post_activation = None
        self.fc1_post_activation = None
        self.post_flatten = None
        self.weight_regularizer = weight_regularizer

    def inference(self, x):
        """
        Performs inference given an input tensor. Here an input
        tensor undergoes a series of nonlinear operations as defined in this method.

        Using variable and name scopes in order to make your graph more intelligible
        for later references in TensorBoard.
        Define name scope for the whole model or for each operator group (e.g. fc+relu)
        individually to group them by name.

        Args:
          x: 3D float Tensor of size [batch_size, input_length, input_channels]

        Returns:
          infer: 2D float Tensor of size [batch_size, self.n_classes]. Returns
                 the infer outputs (before softmax transformation) of the
                 network. These infer(logits) can be used with loss and accuracy
                 to evaluate the model.
        """

        with tf.variable_scope('LSTMnet'):
            # TODO Build Model here
            infer = None
            pass

        return infer

    def accuracy(self, infer, ratings):
        pass

    def loss(self, infer, ratings):
        pass

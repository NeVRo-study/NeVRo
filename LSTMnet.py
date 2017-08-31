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
        # ,,,
        probabilities = []

        # infer = tf.matmul(output, softmax_w) + softmax_b
        return infer

    def _create_lstm_layer(self, x, layer_name, filter_depth):
        """
        Creates a LSTM Layer.
        https://www.tensorflow.org/tutorials/recurrent#lstm
        'Unrolled' version of the network contains a fixed number (num_steps) of LSTM inputs and outputs.

        lstm(num_units)
        The number of units (num_units) is a parameter in the LSTM, referring to the dimensionality of the hidden
        state and dimensionality of the output state (they must be equal)
        (see: https://www.quora.com/What-is-the-meaning-of-“The-number-of-units-in-the-LSTM-cell)
        => num_units = n_hidden = e.g., 128 << hidden layer num of features
        (see: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/
        3_NeuralNetworks/recurrent_network.py)

        'The definition of cell in this package differs from the definition used in the literature.
        In the literature, cell refers to an object with a single scalar output. The definition in this package refers
        to a horizontal array of such units.'

        :param x: Input to layer
        :param layer_name: Name of Layer
        :param filter_depth:
        :return: Layer Output
        """
        with tf.variable_scope(layer_name):
            # TODO continue here
            num_steps = x.shape[0]  # = samp.freq. = 250
            batch_size = 1
            lstm_size = n_hidden = x.shape[1]  # = 2 components
            # lstm_size = n_hidden = 128  # some other values...

            # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
            # if x hase shape [batch_size(1), samples-per-second(250), components(2))
            x = tf.unstack(value=x, num=num_steps, axis=1, name="unstack")  # does not work like that
            # Now: x is list of [250 x (1, 2)]

            # Define LSTM cell
            lstm = tf.contrib.rnn.BasicLSTMCell(num_units=lstm_size)  # 2

            # Initial state of the LSTM memory
            initial_state = state = tf.zeros([batch_size, lstm.state_size])

            loss = 0.0

            for i in range(num_steps):
                output, state = lstm(x[:, i], state)
                #  outputs, states = rnn.static_rnn(lstm, x, dtype=tf.float32)
                # Linear activation, using rnn inner loop last output
                # return tf.matmul(outputs[-1], weights['out']) + biases['out']

            final_state = state
            pass

            lstm_output = None

        return lstm_output

    def _create_fc_layer(self, x, layer_name, shape):
        """

        :param x: Input to layer
        :param layer_name: Name of Layer
        :param shape: Shape from input to output
        :return: Layer activation
        """
        with tf.variable_scope(layer_name):
            weights = tf.get_variable(name=layer_name + "/weights",
                                      shape=shape,
                                      initializer=tf.contrib.layers.xavier_initializer(),
                                      # Reg-scale based on practical2 experiment
                                      regularizer=self.weight_regularizer)

            self._var_summaries(weights, layer_name + "/weights")

            biases = tf.get_variable(name=layer_name + "/biases",
                                     shape=[shape[1]],
                                     initializer=tf.constant_initializer(0.0))

            self._var_summaries(biases, layer_name + "/biases")

            # activation:
            with tf.name_scope(layer_name + "/XW_Bias"):
                pre_activation = tf.matmul(x, weights) + biases
                tf.histogram_summary(layer_name + "/pre_activation", pre_activation)

        return pre_activation

    def accuracy(self, infer, ratings):
        pass

    def loss(self, infer, ratings):
        pass


# # Display tf.variables
# Check: https://stackoverflow.com/questions/33633370/how-to-print-the-value-of-a-tensor-object-in-tensorflow
# sess = tf.InteractiveSession()
# test_var = tf.constant([1., 2., 3.])
# test_var.eval()
# # Add print operation
# test_var = tf.Print(input_=test_var, data=[test_var], message="This is a tf. test variable")
# test_var.eval()
# # Add more stuff
# test_var2 = tf.add(x=test_var, y=test_var).eval()
# test_var2

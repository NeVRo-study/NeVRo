# coding=utf-8
"""
Build LSTM architecture

Author: Simon Hofmann | <[surname].[lastname][at]protonmail.com> | 2017
"""

import tensorflow as tf
# import numpy as np


class NeVRoNet:
    """
    This class implements a LSTM neural network in TensorFlow.
    It incorporates a certain graph model to be trained and to be used
    in inference.

    Potentially for layer visualization check out Beholder PlugIn
    https://www.youtube.com/watch?feature=youtu.be&v=06HjEr0OX5k&app=desktop
    """

    def __init__(self, activation_function=tf.nn.elu,
                 weight_regularizer=tf.contrib.layers.l2_regularizer(scale=0.18),
                 lstm_size=None, fc_hidden_unites=None, n_steps=250, batch_size=1, summaries=True):
        """
        Constructor for an NeVRoNet object.
        Args:
            # n_classes: in t, number of classes of the classification problem. This number is required in order to
            # specify
            the output dimensions of the NeVRoNet.
            weight_regularizer: to be applied weight regularization
        """

        self.lstm_post_activation = []
        self.fc_post_activation = []
        self.weight_regularizer = weight_regularizer
        self.activation_function = activation_function
        self.lstm_size = lstm_size  # = [n_hidden], list
        self.fc_hidden_units = fc_hidden_unites  # number of hidden units in fc layers (if n-layers > 1), list
        self.n_steps = n_steps  # sampling freq. = 250
        self.batch_size = batch_size
        self.final_state = None
        self.lstm_output = None
        self.summaries = summaries  # so far just for weights and biases of FC
        # self.n_classes = n_classes  # n_classes (include for, e.g., binary cases)

    def inference(self, x):
        """
        Performs inference given an input tensor. Here an input
        tensor undergoes a series of nonlinear operations as defined in this method.

        Using variable and name scopes in order to make your graph more intelligible
        for later references in TensorBoard.
        Define name scope for the whole model or for each operator group (e.g., fc+relu)
        individually to group them by name.

        Args:
          x: 3D float Tensor of size [batch_size, input_length, input_channels]

        Returns:
          infer: 2D float Tensor of size [batch_size, self.n_classes]. Returns
                 the infer outputs (as tanh transformation) of the
                 network. These infer can be used with loss and accuracy
                 to evaluate the model.
        """

        with tf.variable_scope('NeVRoNet'):

            lstm_input = x

            for layer in range(len(self.lstm_size)):

                # LSTM Layer
                last_layer = False if (layer + 1) < len(self.lstm_size) else True

                post_lstm = self._create_lstm_layer(x=lstm_input, layer_name="lstm{}".format(layer+1),
                                                    lstm_size=self.lstm_size[layer],
                                                    last_layer=last_layer)

                # For the featuer_extractor:
                self.lstm_post_activation.append(post_lstm)

                # In case there are more LSTM layer
                if len(self.lstm_size) > 1 and not last_layer:
                    lstm_input = tf.stack(post_lstm)
                    lstm_input = tf.transpose(lstm_input, [1, 0, 2])  # [batch_size, 250, 1]

            # Fully Connected Layer(s)
            # post_lstm = tf.Print(input_=post_lstm, data=[post_lstm.get_shape()], message="post_lstm")

            fc_layer_input = self.lstm_post_activation[-1]  # [batch_size, lstm_size[-1]]
            # This works due to lstm_output[-1], which takes only the last and ignores the last 249 outputs of lstm

            # Define first fc-layer shape
            assert isinstance(self.fc_hidden_units, list), "fc_hidden_units must be a list"

            fc_shape = [self.lstm_size[-1], self.fc_hidden_units[0]]

            # Build up layers
            for fc_layer in range(len(self.fc_hidden_units)):

                last_layer = False if fc_layer+1 < len(self.fc_hidden_units) else True

                if fc_layer > 0:
                    old_shape = fc_shape
                    # Update shape
                    fc_shape[0] = old_shape[1]
                    fc_shape[1] = self.fc_hidden_units[fc_layer]
                    fc_layer_input = self.fc_post_activation[-1]  # Feed last output in new fc-layer

                fc_activation = self._create_fc_layer(x=fc_layer_input,
                                                      layer_name="fc{}".format(fc_layer+1),
                                                      shape=fc_shape,  # shape=(lstm_size, 1-rating)
                                                      last_layer=last_layer)

                # shape = [post_lstm.get_shape()[1], post_lstm.get_shape()[0]]

                # For the feature_extractor:
                self.fc_post_activation.append(fc_activation)

            # Use tanh ([-1, 1]) for final prediction
            infer = tf.nn.tanh(x=self.fc_post_activation[-1], name="tanh_inference")

            # Write summary
            with tf.name_scope("inference"):
                tf.summary.histogram("hist", infer)
                tf.summary.scalar(name="scalar", tensor=infer[0][0])

        # probabilities = []

        return infer

    def _create_lstm_layer(self, x, layer_name, lstm_size, last_layer):
        """
        Creates a LSTM Layer.
        https://www.tensorflow.org/tutorials/recurrent#lstm
        'Unrolled' version of the network contains a fixed number (num_steps) of LSTM inputs and outputs.

        lstm(num_units)
        The number of units (num_units) is a parameter in the LSTM, referring to the dimensionality of the hidden
        state and dimensionality of the output state (they must be equal)
        (see: https://www.quora.com/What-is-the-meaning-of-“The-number-of-units-in-the-LSTM-cell)
        => num_units = n_hidden = e.g., 128 << hidden layer num of features, equals also to the number of outputs
        (see: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/
        3_NeuralNetworks/recurrent_network.py)

        'The definition of cell in this package differs from the definition used in the literature.
        In the literature, cell refers to an object with a single scalar output. The definition in this package refers
        to a horizontal array of such units.'

        :param x: Input to layer
        :param layer_name: Name of Layer
        :param lstm_size: Number of hidden units in cell (HyperParameter, to be tuned)
        :return: Layer Output
        """
        with tf.variable_scope(layer_name):
            # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
            # x.shape [batch_size, samples-per-second(250), components(1))
            x = tf.unstack(value=x, num=self.n_steps, axis=1, name="unstack")  # does not work like that
            # Now: x is list of [250 x (batch_size, 1)]

            # Define LSTM cell (Zaremba et al., 2015)
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_size)  # == tf.contrib.rnn.BasicLSTMCell(lstm_size)
            # lstm_cell.state_size

            # Initial state of the LSTM memory
            # (previous state is not taken over in next batch, regardless of zero-state implementation)
            # init_state = lstm_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)

            # Run LSTM cell
            self.lstm_output, self.final_state = tf.contrib.rnn.static_rnn(cell=lstm_cell,
                                                                           inputs=x,
                                                                           # initial_state=init_state,  # (optional)
                                                                           # sequence_length=num_steps,
                                                                           dtype=tf.float32,
                                                                           scope=None)
            # lstm_output: len(lstm_output) == len(x) == 250
            # state (final_state):
            # LSTMStateTuple(c=array([[ ...]], dtype=float32),  # C-State
            #                h=array([[ ...]], dtype=float32))  # H-State
            # state shape: [2, 1, lstm_size]

            #  rnn.static_rnn calculates basically this:
            # outputs = []
            # for input_ in x:
            #     output, state = lstm_cell(input_, state)
            #     outputs.append(output)
            # Check: https://www.tensorflow.org/versions/r1.1/api_docs/python/tf/contrib/rnn/static_rnn

            # # How to define output:
            # Different options: 1) last lstm output 2) average over all outputs
            # or 3) right-weighted average (last values have stronger impact)
            # here option 1)

            # Write summaries
            self._var_summaries(name=layer_name + "/lstm_outputs", var=self.lstm_output[-1])
            self._var_summaries(name=layer_name + "/final_state", var=self.final_state)

            # Push through activation function
            with tf.name_scope(layer_name + "_elu"):  # or relu
                # post_activation = tf.nn.relu(lstm_output, name="post_activation")
                # lstm_output.shape = (250, batch_size, lstm_size) | lstm_output[-1].shape = (batch_size, lstm_size)
                # Only do '[-1]' if last lstm-layer
                pre_activation = self.lstm_output[-1] if last_layer else self.lstm_output

                post_activation = self.activation_function(features=pre_activation, name="post_activation")

                # Write Summaries
                self._var_summaries(name=layer_name + "_elu" + "/post_activation", var=post_activation)
                tf.summary.histogram(layer_name + "/post_activation_hist", post_activation)

        return post_activation

    def _create_fc_layer(self, x, layer_name, shape, last_layer):
        """
        Create fully connected layer
        [Check out tf.layers.dense()]
        :param x: Input to layer
        :param layer_name: Name of Layer
        :param shape: Shape from input to output
        :return: Layer activation
        """

        with tf.variable_scope(layer_name):
            weights = tf.get_variable(name="weights",
                                      shape=shape,
                                      # recommend (e.g., see: cs231n_2017_lecture8.pdf)
                                      initializer=tf.contrib.layers.xavier_initializer(),
                                      regularizer=self.weight_regularizer)
            if self.summaries:
                self._var_summaries(name="weights", var=weights)

            biases = tf.get_variable(name="biases",
                                     shape=[shape[1]],
                                     initializer=tf.constant_initializer(0.0))

            if self.summaries:
                self._var_summaries(name="biases", var=biases)

            # activation y=XW+b:
            with tf.name_scope(layer_name + "/XW_Bias"):
                # Linear activation, using rnn inner loop last output
                pre_activation = tf.matmul(x, weights) + biases
                tf.summary.histogram(layer_name + "/pre_activation", pre_activation)

            if not last_layer:
                # Push through activation function, if not last layer
                with tf.name_scope(layer_name + "_elu"):  # or relu
                    post_activation = self.activation_function(features=pre_activation, name="post_activation")

                return post_activation

            else:  # in case its the last layer
                return pre_activation

    @staticmethod
    def _var_summaries(name, var):

        with tf.name_scope("summaries"):
            mean = tf.reduce_mean(var)
            tf.summary.scalar("mean/" + name, mean)
            with tf.name_scope("stddev"):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

            tf.summary.scalar("stddev/" + name, stddev)
            tf.summary.scalar("max/" + name, tf.reduce_max(var))
            tf.summary.scalar("min/" + name, tf.reduce_min(var))
            tf.summary.histogram(name, var)

    @staticmethod
    def accuracy(infer, ratings):
        """
        Calculate the prediction accuracy, i.e. the average correct predictions of the network.
        As in self.loss above, use tf.summary.scalar to save
        scalar summaries of accuracy for later use with the TensorBoard.

        Args:
          infer: 2D float Tensor of size [batch_size].
                       The predictions returned through infer.
          ratings: 2D int Tensor of size [batch_size]
                     continious range [-1,1]

        Returns:
          accuracy: scalar float Tensor, the accuracy of predictions,
                    i.e. the average correct predictions over the whole batch.
        """
        with tf.name_scope("accuracy"):
            with tf.name_scope("correct_prediction"):

                # 1 - abs(infer-rating)/max_diff, max_diff
                # max_diff = 2.0  # since ratings in [-1, 1]
                max_diff = tf.subtract(1.0, tf.multiply(tf.abs(ratings), -1.0))  # chances depending on rating-level
                correct = tf.subtract(1.0, tf.abs(tf.divide(tf.subtract(infer, ratings), max_diff)), name="correct")

            with tf.name_scope("accuracy"):
                # Return the number of true entries.
                accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

            tf.summary.scalar("accuracy", accuracy)

        return accuracy

    @staticmethod
    def loss(infer, ratings):
        """
        Calculates the mean squared error loss (MSE) from infer/predictions and
        the ground truth labels. The function will also add the regularization
        loss from network weights to the total loss that is return.
        Use tf.summary.scalar to save scalar summaries of mean squared error loss, regularization loss,
        and full loss (both summed) for use with TensorBoard.

        Args:
          infer: 2D float Tensor of size [batch_size].
          ratings: 2D int Tensor of size [batch_size].
                    Ground truth labels for each observation in batch.

        Returns:
          loss: scalar float Tensor, full loss = MSE + reg_loss
        """
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        # with tf.name_scope("cross_entropy"):
        with tf.name_scope("mean_squared_error"):

            # TODO adaptations as in accuracy depending on given rating?
            diff = tf.losses.mean_squared_error(labels=ratings, predictions=infer, scope="mean_squared_error")
            #                                   , loss_collection=reg_losses)

            with tf.name_scope("total"):
                mean_squared_error = tf.reduce_mean(diff, name='mean_squared_error_mean')
                loss = tf.add(mean_squared_error, tf.add_n(reg_losses), name="Full_Loss")
                # add_n==tf.reduce_sum(reg_losses)

            with tf.name_scope("summaries"):
                tf.summary.scalar("Full_loss", loss)
                tf.summary.scalar("Mean_Squared_Error_Loss", mean_squared_error)
                tf.summary.scalar("Reg_Losses", tf.reduce_sum(reg_losses))

        return loss

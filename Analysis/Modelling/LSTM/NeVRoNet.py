# coding=utf-8
"""
Build LSTM architecture

Author: Simon Hofmann | <[surname].[lastname][at]pm.me> | 2017, 2019 (Update)
"""

# %% Import

import tensorflow as tf  # implemented with TensorFlow 1.13.1


# %% Network >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>
class NeVRoNet:
    """
    This class implements a LSTM neural network in TensorFlow.
    It incorporates a certain graph model to be trained and to be used in inference.
    """

    def __init__(self, activation_function=tf.nn.elu,
                 weight_regularizer=tf.contrib.layers.l2_regularizer(scale=0.18),
                 lstm_size=None, fc_hidden_unites=None, n_steps=250, batch_size=1, summaries=True):
        """
        Constructor for an NeVRoNet object.
        Args:
            output dimensions of the NeVRoNet.
            weight_regularizer: to be applied weight regularization
        """

        self.lstm_post_activation = []
        self.fc_post_activation = []
        self.weight_regularizer = weight_regularizer
        self.activation_function = activation_function
        self.lstm_size = lstm_size  # = [n_hidden], list
        self.fc_hidden_units = fc_hidden_unites
        # number of hidden units in fc layers (if n-layers > 1), list
        self.n_steps = n_steps  # sampling freq. = 250
        self.batch_size = batch_size
        self.final_state = None
        self.lstm_output = None
        self.summaries = summaries  # so far just for weights and biases of FC

    def inference(self, x):
        """
        Performs inference given an input tensor. Here an input tensor undergoes a series of nonlinear
        operations as defined in this method.

        Args:
          x: 3D float Tensor of size [batch_size, input_length, input_channels]

        Returns:
          infer: 2D float Tensor of size [batch_size, -1]. Returns the infer outputs (as tanh
                 transformation) of the network. These infer can be used with loss and accuracy to
                 evaluate the model.
        """

        with tf.variable_scope('NeVRoNet'):

            lstm_input = x

            for layer in range(1, len(self.lstm_size)+1):

                # LSTM Layer
                last_layer = False if layer < len(self.lstm_size) else True

                post_lstm = self._create_lstm_layer(x=lstm_input, layer_name=f"lstm{layer}",
                                                    lstm_size=self.lstm_size[layer-1],
                                                    last_layer=last_layer)

                # For the featuer_extractor:
                self.lstm_post_activation.append(post_lstm)

                # In case there are more LSTM layer
                if not last_layer:
                    lstm_input = post_lstm

            # Fully Connected Layer(s)
            fc_layer_input = self.lstm_post_activation[-1]  # [batch_size, lstm_size[-1]]
            # This works due to lstm_output[-1]: takes only last and ignores the last 249 outputs of lstm

            # Define first fc-layer shape
            assert isinstance(self.fc_hidden_units, list), "fc_hidden_units must be a list"

            fc_shape = [self.lstm_size[-1], self.fc_hidden_units[0]]

            # Build up layers
            for fc_layer in range(1, len(self.fc_hidden_units)+1):

                last_layer = False if fc_layer < len(self.fc_hidden_units) else True

                if fc_layer > 1:
                    old_shape = fc_shape
                    # Update shape
                    fc_shape[0] = old_shape[1]
                    fc_shape[1] = self.fc_hidden_units[fc_layer-1]
                    fc_layer_input = self.fc_post_activation[-1]  # Feed last output in new fc-layer

                fc_activation = self._create_fc_layer(x=fc_layer_input,
                                                      layer_name=f"fc{fc_layer}",
                                                      shape=fc_shape,  # shape=(lstm_size, 1-rating)
                                                      last_layer=last_layer)

                # For the feature_extractor:
                self.fc_post_activation.append(fc_activation)

            # Use tanh ([-1, 1]) for final prediction
            infer = tf.nn.tanh(x=self.fc_post_activation[-1], name="tanh_inference")

            # Write summary
            # if self.summaries:
            with tf.name_scope("inference"):
                tf.summary.histogram("hist", infer)
                tf.summary.scalar(name="scalar", tensor=infer[0][0])

        return infer

    def _create_lstm_layer(self, x, layer_name, lstm_size, last_layer):
        """
        Creates a LSTM Layer.
        https://www.tensorflow.org/tutorials/recurrent#lstm
        'Unrolled' version of the network contains a fixed number (num_steps) of LSTM inputs and outputs.

        lstm(num_units)
        The number of units (num_units) is a parameter in the LSTM, referring to the dimensionality of the
        hidden state and dimensionality of the output state (they must be equal)
        (see: https://www.quora.com/What-is-the-meaning-of-“The-number-of-units-in-the-LSTM-cell)
        => num_units = n_hidden = e.g., 128 << hidden layer num of features, equals also to the number of
        outputs
        (see: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/
        3_NeuralNetworks/recurrent_network.py)

        'The definition of cell in this package differs from the definition used in the literature.
        In the literature, cell refers to an object with a single scalar output. The definition in this
        package refers to a horizontal array of such units.'

        :param x: Input to layer
        :param layer_name: Name of Layer
        :param lstm_size: Number of hidden units in cell (HyperParameter, to be tuned)
        :return: Layer Output
        """
        with tf.variable_scope(layer_name):

            # # Define LSTM cell (Zaremba et al., 2015)
            lstm_cell = tf.keras.layers.LSTMCell(units=lstm_size)  # units: int+, dimens. of output space

            # # Initial state of the LSTM memory
            # # (previous state is not taken over in next batch, regardless of zero-state implementation)

            # # Run LSTM cell
            lstm_layer = tf.keras.layers.RNN(cell=lstm_cell, dtype=tf.float32, unroll=True,
                                             return_sequences=True,
                                             # False: return only the last output in sequence
                                             return_state=True,
                                             time_major=False)

            self.lstm_output, state_h, state_c = lstm_layer(inputs=x)
            self.final_state = (state_h, state_c)

            # Write summaries
            self._var_summaries(name=layer_name + "/lstm_outputs",
                                var=self.lstm_output[:, -1, :] if last_layer else self.lstm_output)
            self._var_summaries(name=layer_name + "/final_state", var=self.final_state)

            # Push through activation function
            with tf.name_scope(layer_name + "_elu"):  # or relu
                # lstm_output.shape: (batch_size, 250, lstm_size) |
                # lstm_output[:, -1, :].shape: (batch_size, lstm_size)
                # Only do '[-1]' if last lstm-layer
                pre_activation = self.lstm_output[:, -1, :] if last_layer else self.lstm_output

                post_activation = self.activation_function(features=pre_activation,
                                                           name="post_activation")

                # Write Summaries
                self._var_summaries(name=layer_name + "_elu" + "/post_activation", var=post_activation)
                if self.summaries:
                    tf.summary.histogram(layer_name + "/post_activation_hist", post_activation)

        return post_activation

    def _create_fc_layer(self, x, layer_name, shape, last_layer):
        """
        Create fully connected layer
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

            biases = tf.get_variable(name="biases",
                                     shape=[shape[1]],
                                     initializer=tf.constant_initializer(0.0))

            self._var_summaries(name="weights", var=weights)
            self._var_summaries(name="biases", var=biases)

            # activation y=XW+b:
            with tf.name_scope(layer_name + "/XW_Bias"):
                # Linear activation, using rnn inner loop last output
                pre_activation = tf.matmul(x, weights) + biases
                if self.summaries:
                    tf.summary.histogram(layer_name + "/pre_activation", pre_activation)

            if not last_layer:
                # Push through activation function, if not last layer
                with tf.name_scope(layer_name + "_elu"):  # or relu
                    post_activation = self.activation_function(features=pre_activation,
                                                               name="post_activation")

                return post_activation

            else:  # in case its the last layer
                return pre_activation

    def _var_summaries(self, name, var):
        if self.summaries:
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

        Args:
          infer: 2D float Tensor of size [batch_size]. The predictions returned through infer.
          ratings: 2D int Tensor of size [batch_size], continuous range [-1,1]

        Returns:
          accuracy: scalar float Tensor, the accuracy of predictions,
                    i.e. the average correct predictions over the whole batch.
        """
        with tf.name_scope("accuracy"):
            with tf.name_scope("correct_prediction"):

                # 1 - abs(infer-rating)/max_diff, max_diff
                # max_diff = 2.0  # since ratings in [-1, 1]
                max_diff = tf.subtract(1.0, tf.multiply(tf.abs(ratings), -1.0))
                # chances depending on rating-level
                correct = tf.subtract(1.0, tf.abs(tf.divide(tf.subtract(infer, ratings), max_diff)),
                                      name="correct")

            with tf.name_scope("accuracy"):
                # Return the number of true entries.
                accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

            # if self.summaries:
            tf.summary.scalar("accuracy", accuracy)

        return accuracy

    @staticmethod
    def loss(infer, ratings):
        """
        Calculates the mean squared error loss (MSE) from infer/predictions and the ground truth labels.
        The function will also add the regularization loss from network weights to the total loss that is
        return.

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

            diff = tf.losses.mean_squared_error(labels=ratings, predictions=infer,
                                                scope="mean_squared_error")

            with tf.name_scope("total"):
                mean_squared_error = tf.reduce_mean(diff, name='mean_squared_error_mean')
                loss = tf.add(mean_squared_error, tf.add_n(reg_losses), name="Full_Loss")

            # if self.summaries:
            with tf.name_scope("summaries"):
                tf.summary.scalar("Full_loss", loss)
                tf.summary.scalar("Mean_Squared_Error_Loss", mean_squared_error)
                tf.summary.scalar("Reg_Losses", tf.reduce_sum(reg_losses))

        return loss

# < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<  END

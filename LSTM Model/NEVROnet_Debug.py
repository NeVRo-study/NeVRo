# coding=utf-8

# import sys
# sys.path.insert(0, './LSTM Model')  # or set the folder as source root
import tensorflow as tf  # implemented with TensorFlow 1.3.0
from Load_Data import *


class NeVRoNet:
    def __init__(self, activation_function=tf.nn.elu,
                 weight_regularizer=tf.contrib.layers.l2_regularizer(scale=0.18),
                 lstm_size=None, fc_hidden_unites=None, n_steps=250, batch_size=1, summaries=True):

        self.lstm_post_activation = []
        self.fc_post_activation = []
        self.weight_regularizer = weight_regularizer
        self.activation_function = activation_function
        self.lstm_size = lstm_size  # = [n_hidden], list
        self.fc_hidden_units = fc_hidden_unites  # number of hidden units in fc layers, list
        self.n_steps = n_steps  # samp.freq. = 250
        self.batch_size = batch_size
        self.final_state = None
        self.lstm_output = None
        self.summaries = summaries

    def inference(self, x):

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

            fc_layer_input = self.lstm_post_activation[-1]

            # Define first fc-layer shape
            assert isinstance(self.fc_hidden_units, list), "fc_hidden_units must be a list"

            fc_shape = [self.lstm_size[-1], self.fc_hidden_units[0]]

            # Build up layers
            for fc_layer in range(len(self.fc_hidden_units)):

                last_layer = False if (fc_layer + 1) < len(self.fc_hidden_units) else True

                if fc_layer > 0:
                    old_shape = fc_shape
                    # Update shape
                    fc_shape[0] = old_shape[1]
                    fc_shape[1] = self.fc_hidden_units[fc_layer]
                    fc_layer_input = self.fc_post_activation[-1]

                fc_activation = self._create_fc_layer(x=fc_layer_input,
                                                      layer_name="fc{}".format(fc_layer+1),
                                                      shape=fc_shape,  # shape=(lstm, 1-rating)
                                                      last_layer=last_layer)

                # For the featuer_extractor:
                self.fc_post_activation.append(fc_activation)

            # Use tanh ([-1, 1]) for final prediction
            infer = tf.nn.tanh(x=self.fc_post_activation[-1], name="tanh_inference")

        return infer

    def _create_lstm_layer(self, x, layer_name, lstm_size, last_layer):

        with tf.variable_scope(layer_name):

            # x.shape [batch_size, samples-per-second(250), components(1))
            x = tf.unstack(value=x, num=self.n_steps, axis=1, name="unstack")
            # Now: x is list of [250 x (batch_size, component(1))]

            # tf.Print(x, [x.get_shape()])
            # x = tf.cond(pred=tf.equal(x.get_shape()[1], self.n_steps),
            #             true_fn=lambda: tf.unstack(value=x, num=self.n_steps, axis=1, name="unstack"),
            #             false_fn=lambda: x)

            # Define LSTM cell
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_size)  # == tf.contrib.rnn.BasicLSTMCell(lstm_size)

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

            # Push through activation function
            with tf.name_scope(layer_name + "_elu"):  # or relu
                # lstm_output.shape = (250, batch_size, lstm_size) | lstm_output[-1].shape = (batch_size, lstm_size)
                pre_activation = self.lstm_output[-1] if last_layer else self.lstm_output

                post_activation = self.activation_function(features=pre_activation, name="post_activation")

        return post_activation

    def _create_fc_layer(self, x, layer_name, shape, last_layer):

        with tf.variable_scope(layer_name):
            weights = tf.get_variable(name="weights",
                                      shape=shape,
                                      initializer=tf.contrib.layers.xavier_initializer(),
                                      regularizer=self.weight_regularizer,
                                      dtype=tf.float32)

            biases = tf.get_variable(name="biases",
                                     shape=[shape[1]],
                                     initializer=tf.constant_initializer(0.0),
                                     dtype=tf.float32)

            # activation y=XW+b:
            with tf.name_scope(layer_name + "/XW_Bias"):
                # Linear activation, using rnn inner loop last output
                pre_activation = tf.matmul(x, weights) + biases

            if not last_layer:
                # Push through activation function, if not last layer
                with tf.name_scope(layer_name + "_elu"):  # or relu
                    post_activation = self.activation_function(features=pre_activation, name="post_activation")

                return post_activation

            else:  # in case its the last layer
                return pre_activation

    @staticmethod
    def accuracy(infer, ratings):

        with tf.name_scope("accuracy"):
            with tf.name_scope("correct_prediction"):
                # correct = tf.nn.in_top_k(predictions=logits, targetss=labels, k=1)  # should be: [1,0,0,1,0...]
                # correct = tf.equal(tf.argmax(input=infer, axis=1), tf.argmax(input=ratings, axis=1))

                # 1 - abs(infer-rating)/max_diff, max_diff=2 (since ratings/output in range [-1, 1])
                correct = tf.subtract(1.0, tf.abs(tf.divide(tf.subtract(infer, ratings), 2)), name="correct")

            with tf.name_scope("accuracy"):
                # Return the number of true entries.
                accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        return accuracy

    @staticmethod
    def loss(infer, ratings):

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        # with tf.name_scope("cross_entropy"):
        with tf.name_scope("mean_squared_error"):
            # sparse_softmax_cross_entropy_with_logits(), could also be, since we have an exclusive classification
            # diff = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='cross_entropy_diff')
            # diff = tf.nn.softmax_cross_entropy_with_logits(logits=infer, labels=ratings, name="cross_entropy_diff")
            # diff = tf.losses.mean_squared_error(labels=ratings, predictions=infer, scope="mean_squared_error",
            #                                     loss_collection=reg_losses)
            diff = tf.losses.mean_squared_error(labels=ratings, predictions=infer, scope="mean_squared_error")
            # print("Just produced the diff in the loss function")

            with tf.name_scope("total"):
                # cross_entropy = tf.reduce_mean(diff, name='cross_entropy_mean')
                mean_squared_error = tf.reduce_mean(diff, name='mean_squared_error_mean')
                loss = tf.add(x=mean_squared_error, y=tf.add_n(reg_losses), name="Full_Loss")
                # add_n==tf.reduce_sum(reg_losses)
        return loss


sess = tf.InteractiveSession()

batch_sizes = 3  # 1


def _feed_dict(training):
    """creates feed_dicts depending on training or no training"""
    # Train
    if training:
        xs, ys = nevro_data["train"].next_batch(batch_size=batch_sizes, randomize=True)
        ys = np.reshape(ys, newshape=([batch_sizes, 1]))

    else:
        # Validation:
        xs, ys = nevro_data["validation"].next_batch(batch_size=batch_sizes, randomize=True)
        ys = np.reshape(ys, newshape=([batch_sizes, 1]))

    return {x: xs, y: ys}


nevro_data = get_nevro_data(subject=36, component=best_component(36), s_fold_idx=9, s_fold=10,
                            cond="NoMov", sba=True, hilbert_power=True)

ddims = list(nevro_data["train"].eeg.shape[1:])  # [250, 2]

with tf.name_scope("input"):
    x = tf.placeholder(dtype=tf.float32, shape=[None] + ddims, name="x-input")  # None for batch-Size
    y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="y-input")

lstm_model = NeVRoNet(lstm_size=[10, 10], fc_hidden_unites=[1], n_steps=ddims[0], batch_size=batch_sizes,  # n_steps=250
                     activation_function=tf.nn.elu,
                     weight_regularizer=tf.contrib.layers.l2_regularizer(scale=0.18), summaries=False)

inference = lstm_model.inference(x=x)
losses = lstm_model.loss(infer=inference, ratings=y)
optimization = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(losses)
accuracies = lstm_model.accuracy(infer=inference, ratings=y)

tf.global_variables_initializer().run()  # or without .run()

for _ in range(2):
    _, prediction, train_loss, train_acc, final_state, lstm_output = sess.run([optimization, inference, losses,
                                                                               accuracies, lstm_model.final_state,
                                                                               lstm_model.lstm_output],
                                                                              feed_dict=_feed_dict(training=True))

    print("Train-Loss: {:.4f}".format(np.round(train_loss, 4)))
    print("Train-Accuracy: {:.4f}".format(np.round(train_acc, 4)))

    print("final_state: shape:[2,{}]".format(final_state[0].shape))  # 2:= c,h states >> (states, batch_size, lstm_size)
    print("final_state: c-state & h-state")
    print(final_state)
    print("lstm_output, length:", len(lstm_output))
    print(lstm_output)
    print("prediction")
    print(prediction)


# # Display tf.variables
# Check: https://stackoverflow.com/questions/33633370/how-to-print-the-value-of-a-tensor-object-in-tensorflow
# sess = tf.InteractiveSession()
# test_var = tf.constant([1., 2., 3.])
# test_var.eval()
# # Add print operation
# test_var = tf.Print(input_=test_var, data=[test_var, " with shape:", test_var.get_shape()],
#                     message="This is a tf. test variable: ")
# test_var.eval()
# # Add more stuff
# test_var2 = tf.add(x=test_var, y=test_var).eval()
# test_var2

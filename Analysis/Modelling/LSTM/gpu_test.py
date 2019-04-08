# coding=utf-8
"""
Test GPU implementation
"""

import tensorflow as tf
from datetime import datetime

print("GPU device is available:",
      tf.test.is_gpu_available())  # Returns True iff a gpu device of the requested kind is available.

compute_on = '/gpu:0' if tf.test.is_gpu_available() else '/cpu:0'

with tf.device(compute_on):  # '/cpu:0'
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))

if 'gpu' not in compute_on:
    print("Can only be run on CPU")


# Another test
# In case there is a GPU: test it against CPU
device_name = ["/gpu:0", "/cpu:0"] if tf.test.is_gpu_available() else ["/cpu:0"]

for device in device_name:
    for shape in [4500, 6000, 12000]:
        with tf.device(device):
            random_matrix = tf.random_uniform(shape=(shape, shape), minval=0, maxval=1)
            dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
            sum_operation = tf.reduce_sum(dot_operation)

        startTime = datetime.now()
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
                result = session.run(sum_operation)
                print(result)

        print("\n" * 3)
        print("Shape:", (shape, shape), "Device:", device)
        print("Time taken:", datetime.now() - startTime)
        print("\n" * 3)

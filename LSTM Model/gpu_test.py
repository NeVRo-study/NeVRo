# coding=utf-8
"""
Test GPU implementation
"""

import tensorflow as tf

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

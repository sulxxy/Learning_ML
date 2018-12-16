import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

## Data Preparation
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

def weight_init(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_init(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def MLP():
    # def calc_accuracy2():
    #     correct_pred = tf.equal(tf.argmax(y_estimate, 1), tf.argmax(y, 1))
    #     accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    #     return accuracy.eval({x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})

    in_units = 784
    h1_units = 300
    h2_units = 16
    h3_units = 17
    out_units = 10
    sess = tf.InteractiveSession()
    W1 = weight_init([in_units, h1_units])
    b1 = bias_init([h1_units])
    W2 = weight_init([h1_units, h2_units])
    b2 = bias_init([h2_units])
    W3 = weight_init([h2_units, h3_units])
    b3 = bias_init([h3_units])
    W4 = weight_init([h1_units, out_units])
    b4 = bias_init([out_units])

    x = tf.placeholder(tf.float32, [None, in_units])
    y = tf.placeholder(tf.float32, [None, 10])
    #     keep_prob = tf.placeholder(tf.float32)

    hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
    #     hidden1_drop = tf.nn.dropout(hidden1, keep_prob)

    hidden2 = tf.nn.relu(tf.matmul(hidden1, W2) + b2)
    #     hidden2_drop = tf.nn.dropout(hidden2, keep_prob)

    hidden3 = tf.nn.relu(tf.matmul(hidden2, W3) + b3)
    #     hidden3_drop = tf.nn.dropout(hidden3, keep_prob)

    y_estimate = tf.nn.softmax(tf.matmul(hidden1, W4) + b4)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_estimate), reduction_indices=[1]))
    train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

    tf.global_variables_initializer().run()
    for i in range(3000):
        x_batch, y_batch = mnist.train.next_batch(100)
        train_step.run({x: x_batch, y: y_batch})
        if i % 1000 == 0:
            #             print ("Step %d, accuracy: %g" % (i, calc_accuracy2()))
            correct_pred = tf.equal(tf.argmax(y_estimate, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            print accuracy.eval({x: mnist.test.images, y: mnist.test.labels})

    correct_pred = tf.equal(tf.argmax(y_estimate, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    print accuracy.eval({x: mnist.test.images, y: mnist.test.labels})
    #     print("final accuracy: %g" % (calc_accuracy2()))

    print ("finshed.")


MLP()





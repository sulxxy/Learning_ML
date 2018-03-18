import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# read data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# linear model
def lm():
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # training
    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entroy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entroy)
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(500)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # evaluation
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

def weight_init(shape):
    initial = tf.truncated_normal(shape, mean=0, stddev=0.01)
    return tf.Variable(initial)

def bias_init(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def lstm_cell(hidden_unit, layer_num):
    lstm_cells = tf.contrib.rnn.BasicLSTMCell(hidden_unit, forget_bias=0.0, state_is_tuple=True)
    lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cells]*layer_num, state_is_tuple=True)
    return lstm_cells


def RNN(x, mini_batch_size, hidden_unit, layer_num):
    # x_in = tf.matmul(x, W_in) + bias_in
    lstm_cells = lstm_cell(hidden_unit, layer_num)
    init_state = lstm_cells.zero_state(mini_batch_size, dtype=tf.float32)
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cells, x, initial_state=init_state)
    return outputs, final_state[-1][1]

def lstm():

    # params
    Epoch = 60
    mini_batch_size = 50
    train_set_num = 8000
    test_set_num = 2000
    iterations_num = Epoch * train_set_num/mini_batch_size
    time_step_size = 1
    input_unit = 30
    steps =input_unit/time_step_size
    hidden_unit = 200
    layer_num = 1

    # data preparation
    X = np.random.randint(0, 9, [10000, input_unit])
    Y = (np.sum(X, axis = 1) >= 100).astype(float)
    X_train = X[:8000, :]
    Y_train = Y[:8000]
    X_test = X[8000:, :]
    Y_test = Y[8000:].reshape(2000,1)
    print Y_test.shape

    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32, [None, input_unit])
    y = tf.placeholder(tf.float32, [None, 1])

    x_reshaped = tf.reshape(x, [-1, time_step_size, steps])

    output, final_state = RNN(x_reshaped, mini_batch_size, hidden_unit, layer_num)

    W = weight_init([hidden_unit, 1])
    bias = bias_init([1])

    y_estimate = tf.matmul(final_state, W) + bias

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_estimate, labels=y))
    train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_estimate, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    tf.global_variables_initializer().run()

    for i in range(iterations_num):
        b = np.random.choice(8000, 50)
        x_batch = np.take(X_train, b, axis=0)
        y_batch = np.take(Y_train, b).reshape(50,1)
        if i % 100 == 0:
            acc = accuracy.eval(feed_dict = {x: X_test, y: Y_test})
            print "step %d, acc %g" % (i, acc)
        train_step.run(feed_dict={x: x_batch, y: y_batch})

    print "acc: %g" % acc

lstm()




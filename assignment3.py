import tensorflow as tf
from preprocessing import csv_to_dataframe
from sklearn.model_selection import train_test_split
import numpy as np
from random import randint


def mlp(X, weights, biases):
    print(np.transpose(X))
    print(weights['hidden_layer_1'])
    layer_1 = tf.add(np.dot(X, weights['hidden_layer_1']), biases['bias_1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, weights['hidden_layer_2']), biases['bias_2'])
    layer_2 = tf.nn.sigmoid(layer_2)
    out_layer = tf.add(tf.matmul(layer_2, weights['out']), biases['out'])

    return out_layer

def input_train(train_x):
    return (
        # Shuffling with a buffer larger than the data set ensures
        # that the examples are well mixed.
        train_x.shuffle(1000).batch(128)
        # Repeat forever
        .repeat().make_one_shot_iterator().get_next())

def model():

    # data
    data = csv_to_dataframe('sine.csv')
    train_x, test_x, train_y, test_y = train_test_split(data['X'], data['Y'], test_size=0.20, random_state=20)

    # Parameters
    learning_rate = 0.1
    number_of_steps = 200


    # NN parameters
    n_hidden_1 = 3
    n_hidden_2 = 3
    n_input = 1
    n_output = 1

    # inputs

    X = tf.placeholder(tf.float32, [None, 1])
    Y = tf.placeholder(tf.float32, [None, 1])

    weights = {
        'hidden_layer_1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'hidden_layer_2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_output]))
    }

    biases = {
        'bias_1': tf.Variable(tf.random_normal([n_hidden_1])),
        'bias_2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_output]))
    }

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    model = tf.estimator.LinearRegressor(feature_columns=[train_x, train_y])

    # Optimizer
    # Start training
    # with tf.Session() as sess:
    #     for step in range(1, number_of_steps):
    #         random_step = randint(0, (train_y.size - 1))
    #         y_predicted = mlp([float(train_x[random_step])], weights, biases)
    #         cost = tf.square(train_y[random_step] - y_predicted, name="cost")
    #         training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    #         sess.run(training_step, feed_dict={X: train_x[random_step], Y: train_y[random_step]})

    # pred_y = sess.run(test_y, feed_dict={X: test_x})
    # mse = tf.reduce_mean(tf.square(pred_y - test_y))
    # print(mse)

    model.train(input_fn=input_train(tf.tensor(train_x)), steps=number_of_steps)
    # Evaluate how the model performs on data it has not yet seen.
    eval_result = model.evaluate(input_fn=input_test)




model()














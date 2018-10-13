import tensorflow as tf


def neural_net(x, n_input, n_output):

    # NN parameters
    n_hidden_1 = 3
    n_hidden_2 = 3

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

    layer_1 = tf.add(tf.matmul(x, weights['hidden_layer_1'], biases['bias_1']))
    layer_2 = tf.add(tf.matmul(layer_1, weights['hidden_layer_2'], biases['bias_2']))
    out_layer = tf.matmul(layer_2, weights['out'], biases['out'])

    return out_layer


def model():

    # Parameters
    learning_rate = 0.1
    epochs = 100
    batch_size = 15
    display_step = 100

    n_input = 1
    n_output = 1

    # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # inputs
    X = tf.placeholder("float", [None, n_input])
    Y = tf.placeholder("float", [None, n_output])

    apply_model = neural_net(X, n_input, n_output)


    











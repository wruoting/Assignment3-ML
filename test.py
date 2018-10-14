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

# Optimizer
# Start training
# with tf.Session() as sess:
#     for step in range(1, number_of_steps):
#         random_step = randint(0, (train_y.size - 1))
#         y_predicted = mlp([float(train_x[random_step])], weights, biases)
#         cost = tf.square(train_y[random_step] - y_predicted, name="cost")
#         training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
#         sess.run(training_step, feed_dict={X: train_x[random_step], Y: train_y[random_step]})
#
# pred_y = sess.run(test_y, feed_dict={X: test_x})
# mse = tf.reduce_mean(tf.square(pred_y - test_y))
# print(mse)

# filename_queue = tf.train.string_input_producer(['sine.csv'])
#
# reader = tf.TextLineReader()
# key, value = reader.read(filename_queue)
# record_defaults = [[1.0], [1.0]]
# X, Y = tf.decode_csv(
#     value, record_defaults=record_defaults)
# features = tf.stack([X, Y])
#


def mlp(X, weights, biases):
    print(np.transpose(X))
    print(weights['hidden_layer_1'])
    layer_1 = tf.add(np.dot(X, weights['hidden_layer_1']), biases['bias_1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, weights['hidden_layer_2']), biases['bias_2'])
    layer_2 = tf.nn.sigmoid(layer_2)
    out_layer = tf.add(tf.matmul(layer_2, weights['out']), biases['out'])

    return out_layer

from random import randint

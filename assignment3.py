from preprocessing import csv_to_dataframe
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def model():
    # data
    data = csv_to_dataframe('sine.csv')
    train_x, test_x, train_y, test_y = train_test_split(data['X'], data['Y'], test_size=0.80, random_state=20)

    # Parameters
    number_of_steps = 500

    input_test = np.arange(-1.0, 1.0, 0.05, dtype=float)
    feature_columns = [
        tf.feature_column.numeric_column(key="X")
    ]

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"X": np.array(train_x.astype(float))},
        y=np.array(train_y.astype(float)),
        num_epochs=200,
        shuffle=True)
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"X": np.array(test_x.astype(float))},
        y=np.array(test_y.astype(float)),
        num_epochs=1,
        shuffle=True)
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"X": input_test},
        shuffle=False)

    model = tf.estimator.DNNRegressor(feature_columns=feature_columns,
                                      hidden_units=[3, 3],
                                      optimizer=lambda: tf.train.AdamOptimizer(
                                        learning_rate=tf.train.exponential_decay(
                                        learning_rate=0.01,
                                        global_step=tf.train.get_global_step(),
                                        decay_steps=10000,
                                        decay_rate=1))
                                      )

    # train_x_tensor = tf.constant(train_x.astype(float))
    model.train(input_fn=train_input_fn, steps=number_of_steps)
    # Evaluate how the model performs on data it has not yet seen.
    eval_result = model.evaluate(input_fn=test_input_fn)
    print(eval_result)

    predict_results = model.predict(input_fn=predict_input_fn)
    y_out = []
    for i, prediction in enumerate(predict_results):
        y_out.append(prediction.get('predictions'))

    print(np.column_stack((input_test, y_out)))
    plt.plot(input_test, y_out)
    plt.plot(data['X'].convert_objects(convert_numeric=True), data['Y'].convert_objects(convert_numeric=True))
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])

    plt.show()



model()














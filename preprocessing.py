from sklearn import metrics, preprocessing
import numpy as np
import csv
import pandas as pd


# we assume data is distributed normally
def processing_data(data):
    # Normalization
    scaler = preprocessing.StandardScaler()
    data = data.astype(str).astype(int)
    X = data.get('X')
    Y = data.get('Y')

    # Preprocess data, normalization
    xdata = np.column_stack((X, Y))
    xdata = np.nan_to_num(xdata)

    xdata = scaler.fit_transform(xdata)
    state = xdata[0:1, :]

    return state, xdata


def csv_to_dataframe(file):
    relative_path = "./"
    full_relative_path = relative_path + str(file)
    columns = []
    data = []
    with open(full_relative_path, 'rt', encoding='utf8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for index, row in enumerate(csv_reader):
            if index == 0:
                columns = row
            else:
                data.append(row)

    df = pd.DataFrame(data, columns=columns)

    return df



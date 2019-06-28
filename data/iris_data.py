import pandas as pd
import numpy as np


def get_iris_data():
    """
    Function for fetching iris data from archive. The function obtains
    the data and puts it in the format needed for examples/perceptron

    Returns
    -------
    data: iterable
        Iris data in format for perceptron algorithm (iterable of pairs x,y)
    """

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

    df = pd.read_csv(url, header=None)

    x_vals = df.iloc[:, [0, 2]].values

    y_vals = df.iloc[:, 4].values
    y_vals = np.where(y_vals == 'Iris-setosa', -1, 1)

    data = []

    for i in range(len(x_vals)):
        data[i] = (x_vals, y_vals)

    return data

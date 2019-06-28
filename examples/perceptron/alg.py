# this file implements the raw perceptron algorithm for testing purposes
# ideally the output of this algorithm will match that of the circuit

import numpy as np


def perceptron(data, num_iterations, initial_w=0, initial_b=0):
    """
    Takes data (assumed to be an iterable of pairs (x,y)) and runs the
    perceptron algorithm for the number of iterations specified. We
    also assume that x is a numpy array-like object.

    The goal of the perceptron algorithm is to find (optimal) w,b such that
    y(dot(w,x) + b) > 0 is true for as many data points as possible

    Parameters
    ----------
    data: iteratable
        Data to be input into algorithm (assumed iterable pairs)
    num_iterations: int
        Number of iterations that algorithm will run for
    (optional) initial_w=0: int
        Initial value of w, parameter of perceptron algorithm
    (optional) initial_b=0: int
        Initial value of b, parameter of perceptron algorithm

    Returns
    -------
    w: float
        w value achieved after num_iterations of perceptron
    b: int
        b value achieved after num_iterations of perceptron
    """

    w = initial_w
    b = initial_b

    # need to make dimenions of w the same as x
    if initial_w == 0:
        first_x = data[0][0]
        initial_w = np.zeros(len(first_x))

    for i in range(num_iterations):

        np_x = np.array(data[i][0])
        y = data[i][1]

        # if point misclassified, update w and b, else do nothing
        if y * (np.dot(np_x, w) + b) <= 0:
            w += y * np_x
            b += y

    return (w, b)
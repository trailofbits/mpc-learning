# this file implements the raw perceptron algorithm for testing purposes
# ideally the output of this algorithm will match that of the circuit

import numpy as np
import time


def perceptron(data, num_iterations, modulus, initial_w=0, initial_b=0, fp_precision=16):
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
    modulus: int
        Value representing the modulus of field used
    (optional) initial_w=0: int
        Initial value of w, parameter of perceptron algorithm
    (optional) initial_b=0: int
        Initial value of b, parameter of perceptron algorithm
    (optional) fp_precision=16: int
        Fixed point number precision

    Returns
    -------
    w: float
        w value achieved after num_iterations of perceptron
    b: int
        b value achieved after num_iterations of perceptron
    """

    # need to make dimenions of w the same as x
    if initial_w == 0:
        first_x = data[0][0]
        initial_w = np.zeros(len(first_x))

    w = initial_w
    b = initial_b

    # use fixed point numbers
    # input data should be scaled up by 10^fp_precision
    # also scale down by 10^fp_precision after every mult
    scale = 10**fp_precision

    start_time = time.time()

    for i in range(num_iterations):
        
        np_x = np.array(data[i][0])
        y = data[i][1]

        # we use fixed point, so multiply by precision and round to integer
        for a in range(len(np_x)):
            np_x[a] = int( np_x[a] * scale)
        y = int( y * scale)

        # if point misclassified, update w and b, else do nothing
        xw_dot = np.dot(np_x,w) / scale
        if (y * (xw_dot + b)) / scale <= 0:
            w += (y * np_x) / scale
            b += y

        
        print("iteration: " + str(i))
        print(w)
        print(b)


    w = w / scale
    b = b / scale

    elapsed_time = time.time() - start_time
    print("elapsed time: " + str(elapsed_time))

    return (w, b)

if __name__ == "__main__":

    import data.iris_data as iris

    data = iris.get_iris_data()

    num_iter = len(data)

    print(perceptron(data,num_iter,2**128))
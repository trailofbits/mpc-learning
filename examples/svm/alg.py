import numpy as np
import matplotlib.pyplot as plt
import time
import math

MOD = MOD = 622288097498926496141095869268883999563096063592498055290461
MOD_BIT_SIZE = len(bin(MOD)[2:])

def svm(data, num_iterations, initial_w=0, initial_b=0, hyper_param=1, fp_precision=16):

    svm_data = alter_data(data)
    #svm_data = data

    # need to make dimenions of w the same as x
    if initial_w == 0:
        first_x = svm_data[0][0]
        initial_w = np.zeros(len(first_x))

    w = initial_w
    b = initial_b

    # use fixed point numbers
    # input data should be scaled up by 10^fp_precision
    # also scale down by 10^fp_precision after every mult
    scale = 10**fp_precision
    #scale = 1.0

    start_time = time.time()

    for i in range(num_iterations):

        learning_rate = int(round((1.0 / (1.0 + i))*10**7) / 10**7 * scale)
        #learning_rate = int(0.5 * scale)
        #learning_rate = int(1*scale)
        #learning_rate = int(scale / 3)

        np_x = np.array(svm_data[i][0])
        y = svm_data[i][1]

        # we use fixed point, so multiply by precision and round to integer
        for a in range(len(np_x)):
            np_x[a] = int( np_x[a] * scale)
        y = int( y * scale)

        # if y * (w dot x) <= 1:
        #   w <- (1 - learning rate) * w + (learning rate) * hyper_param * y * x
        # else:
        #   w <- (1 - learning rate) * w
        xw_dot = int(np.dot(np_x,w) / scale)
        #xw_dot = np.dot(np_x,w)
        if (y / scale * xw_dot) <= (1*scale):
            #print(w)
            w = ((scale - learning_rate) / scale * w) + ((learning_rate / scale * hyper_param * y) / scale * np_x)
            #print(w)
        else:
            w = (1*scale - learning_rate) / scale * w
        
        mod_bit_size = MOD_BIT_SIZE
        #trunc_val = 2**int((mod_bit_size - 1) / 3)
        #trunc_val = 2**20
        trunc_val = 10**7

        #w = np.array([int(round(el / 10**7)*10**7) for el in w])
        w = np.array([int(math.floor(el / trunc_val)*trunc_val) for el in w])
        #for a in range(len(w)):
        #    w[a] = int(w[a])
        #    print(w[a])
        #    print(int(w[a]))

        if (i < 150):
            print("iter " + str(i) + ": " + str(w))
        #print(w)

        #if 98 <  i < 102:
        #    print("ITERATION: " + str(i))
        #    print("data: " + str((np_x,y)))

    return_w = w[:-1]
    return_b = w[-1]
    #print(w)
    #print(return_w)
    elapsed_time = time.time() - start_time
    print("elapsed time: " + str(elapsed_time))
        
    return (return_w / scale,return_b / scale)
    #return w / scale

def alter_data(data):

    new_data = []

    for i in range(len(data)):
        old_x = data[i][0]
        old_y = data[i][1]

        new_x = np.append(old_x,1)

        new_data.append([np.array(new_x),old_y])

    #for line in new_data:
        #print(line)
    return new_data

def plot_data_line(data,line):

    x1_vals = []
    y1_vals = []
    x2_vals = []
    y2_vals = []

    for i in range(len(data)):
        y = data[i][1]
        x = data[i][0]

        if y == -1:
            x1_vals.append(x[0])
            y1_vals.append(x[1])
        else:
            x2_vals.append(x[0])
            y2_vals.append(x[1])

    plt.scatter(x1_vals,y1_vals,color='r')
    plt.scatter(x2_vals,y2_vals,color='b')
    plt.show()


if __name__ == "__main__":

    import data.iris_data as iris
    data = iris.get_iris_data()

    num_iter = len(data)
    print(svm(data,num_iter,fp_precision=10))
    #plot_data_line(data,"")
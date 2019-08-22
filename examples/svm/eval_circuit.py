from src.circuits.evaluator import BasicEvaluator
from src.circuits.evaluator import SecureEvaluator
from src.circuits.dealer import Dealer
from src.circuits.oracle import Oracle
from examples.svm import circuit as circ
import numpy as np
from threading import Thread
import copy
from examples.svm.alg import alter_data
import asyncio
import time



def secure_eval_circuit(data,num_iterations,modulus,initial_w=0,initial_b=0,fp_precision=16):
    """
    Function that evaluates the perceptron circuit using three SecureEvaluator
    objects. The current protocol also requires a Dealer and an Oracle.

    Parameters
    ----------
    data: iterable
        Data to be input into the perceptron algorithm (assumed iterable pairs)
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

    # alter data to work for svm
    data = alter_data(data)

    # account for fixed point precision
    scale = 10**fp_precision

    # initialize oracle
    oracle = Oracle(modulus,fp_precision=fp_precision)

    circ1 = copy.deepcopy(circ.circuit)
    circ2 = copy.deepcopy(circ.circuit)
    circ3 = copy.deepcopy(circ.circuit)

    # initialize evaluators
    evaluator1 = SecureEvaluator(circ1,circ.in_gates,circ.out_gates,1,oracle,modulus,fp_precision=fp_precision)
    evaluator2 = SecureEvaluator(circ2,circ.in_gates,circ.out_gates,2,oracle,modulus,fp_precision=fp_precision)
    evaluator3 = SecureEvaluator(circ3,circ.in_gates,circ.out_gates,3,oracle,modulus,fp_precision=fp_precision)

    #evaluator1 = SecureEvaluator(circ1,circ.in_gates,circ.out_gates,1,oracle,modulus)
    #evaluator2 = SecureEvaluator(circ2,circ.in_gates,circ.out_gates,2,oracle,modulus)
    #evaluator3 = SecureEvaluator(circ3,circ.in_gates,circ.out_gates,3,oracle,modulus)


    parties = [evaluator1,evaluator2,evaluator3]
    party_dict = {1: evaluator1, 2: evaluator2, 3: evaluator3}

    evaluator1.add_parties(party_dict)
    evaluator2.add_parties(party_dict)
    evaluator3.add_parties(party_dict)

    # initialize dealer
    dealer = Dealer(parties,modulus,fp_precision=fp_precision)

    start_time = time.time()

    # split x_data and y_data into 3 lists, one for each party
    # this simulates each party having private input data
    data_len = len(data)
    data1x = []
    data2x = []
    data3x = []
    data1y = []
    data2y = []
    data3y = []

    split = int(data_len/3)

    for i in range(split):
        data1x.append(data[i][0])
        data1y.append(data[i][1])
        data2x.append(data[split + i][0])
        data2y.append(data[split + i][1])
        data3x.append(data[2*split + i][0])
        data3y.append(data[2*split + i][1])

    # use dealer to create shares of all inputs
    dealer.distribute_shares(data1x)
    dealer.distribute_shares(data2x)
    dealer.distribute_shares(data3x)

    dealer.distribute_shares(data1y)
    dealer.distribute_shares(data2y)
    dealer.distribute_shares(data3y)

    # use dealer to create random values for interactive operations
    num_randomness = 10000 * num_iterations
    dealer.generate_randomness(num_randomness)
    dealer.generate_truncate_randomness(5*num_iterations)

    # need to make dimenions of w the same as x
    if initial_w == 0:
        first_x = data[0][0]
        initial_w = np.zeros(len(first_x))
    initial_w = [initial_w,[]]

    dealer.distribute_shares(initial_w)
    #dealer.distribute_shares(initial_b)

    results = {}

    # for each iteration of perceptron algorithm, have each SecureEvaluator
    # compute the circuit, each on their own thread, so they can interact
    res = {}
    for i in range(num_iterations):
    #for i in range(1):

        print("iteration: " + str(i))
        
        t1 = Thread(target=run_eval,args=(evaluator1,i,data_len,results,1,modulus,fp_precision,res))
        t2 = Thread(target=run_eval,args=(evaluator2,i,data_len,results,2,modulus,fp_precision,res))
        t3 = Thread(target=run_eval,args=(evaluator3,i,data_len,results,3,modulus,fp_precision,res))

        t1.start()
        t2.start()
        t3.start()

        t1.join()
        t2.join()
        t3.join()

    #for a in range(150):
    for a in range(5):
        print("iter " + str(a) + ": " + str(unshare(res[str(a)+"_1"][0],res[str(a)+"_2"][0])))

    # extract final outputs, scale them down
    (w,b) = get_w_b(results)
    #return (w / scale, b / scale)
    wout = []
    for el in w:
        wout.append(el / scale)
    bout = b / scale
    elapsed_time = time.time() - start_time
    print("elapsed time: " + str(elapsed_time))
    return (np.array(wout),bout)

def unshare(share1,share2):
    """
    Method for converting shares into their hidden value

    Parameters
    ----------
    share1: int or iterable
        Shares of value
    share2: int or iterable
        Shares of same value as share1

    Returns
    -------
    res:
        value hidden by share1 and share2
    """

    if type(share1) == list:
        res = []
        for i in range(len(share1)):
            res.append(share1[i].unshare(share2[i]))

    else:
        res = share1.unshare(share2)

    return res

def get_w_b(w_b_shares):
    """
    Method for computing (w,b) from their shares

    Parameters
    ----------
    w_b_shares: dictionary
        Dictionary of shares for values of (w,b)

    Returns
    -------
    w: float
        w value achieved after num_iterations of perceptron
    b: int
        b value achieved after num_iterations of perceptron
    """

    w1 = w_b_shares[1]['w']
    w2 = w_b_shares[2]['w']
    w3 = w_b_shares[3]['w']

    w = [w1[0].unshare(w2[0]), w1[1].unshare(w2[1])]
    b = w1[2].unshare(w2[2])

    return (w,b)


def run_eval(evaluator,iter_num,data_length,results_dict,party_index,mod,fp_precision=16,wd={}):
    """
    Method to be run by each SecureEvaluator within their Thread (this will be
    called with secure_eval_circuit).

    Parameters
    ----------
    evaluator: SecureEvaluator object
        SecureEvaluator that will compute an iteration of perceptron algorithm
    iter_num: int
        Iteration number of perceptron algorithm
    data_length: int
        Integer representing length of input data
    results_dict: dictionary
        Dictionary for each thread to insert ouput values
    party_index: int
        Integer representing evaluator party index
    (optional) fp_precision=16: int
        Fixed point number precision
    """

    scale = 10**fp_precision

    # input will map wire name to index in list of shares
    cur_input = {}
    cur_input["in0"] = iter_num
    cur_input["in1"] = data_length + iter_num

    # only load initial b and w
    #if iter_num == 0:
    #    cur_input["in2"] = -2
    #    cur_input["in3"] = -1

    cur_input["in2"] = -1

    evaluator.load_secure_inputs(cur_input)

    # need to load in constant values
    # -1: for computing <=1, subtract 1 and comp <= 0
    # gam1: need to have 1 - gamma for computing w
    # gamC: need gamma * C also for computing w
    neg1 = int(-1*scale)
    pre_gamma = 1.0 / (1.0 + iter_num)
    gamma = round(pre_gamma * 10**7)
    gamma = int(gamma * scale)
    gamma = int(gamma / 10**7)
    gam1 = int(1*scale - gamma)
    gamC = int(gamma * 1)

    load_in_constants = {}
    load_in_constants["in3"] = gam1
    load_in_constants["in4"] = gamC
    load_in_constants["in5"] = neg1

    evaluator.load_inputs(load_in_constants)

    evaluator.run()

    [w] = evaluator.get_outputs()
    evaluator.reset_circuit()

    wd[str(iter_num) + "_" + str(party_index)] = [w]

    #cur_in = {}
    #cur_in["in2"] = w
    #cur_in["in3"] = b
    #evaluator.load_inputs(cur_in)

    evaluator.receive_shares([w])

    results_dict[party_index] = {"w": w}

def mod_inverse(val, mod):
    g, x, y = egcd(val, mod)
    if g != 1:
        raise Exception('modular inverse does not exist')
    else:
        return x % mod

def egcd(a,b):
    if a == 0:
        return (b,0,1)
    else:
        g, y, x = egcd(b %a, a)
        return (g, x - (b //a) * y, y)


if __name__ == "__main__":

    #MOD = 10001112223334445556667778889991

    # 199 bits
    MOD = 622288097498926496141095869268883999563096063592498055290461

    #MOD = 24684249032065892333066123534168930441269525239006410135714283699648991959894332868446109170827166448301044689

    import data.iris_data as iris

    data = iris.get_iris_data()

    num_iter = len(data)
    #num_iter = 10

    print(secure_eval_circuit(data,num_iter,MOD,fp_precision=9))
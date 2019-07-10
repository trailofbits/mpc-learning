from src.circuits.evaluator import BasicEvaluator
from src.circuits.evaluator import SecureEvaluator
from src.circuits.dealer import Dealer
from src.circuits.oracle import Oracle
import circuit
import numpy as np
from threading import Thread

def eval_circuit(data,num_iterations,initial_w=0,initial_b=0,fp_precision=16):

    evaluator = BasicEvaluator(circuit.circuit,circuit.gate_order,fp_precision=fp_precision)

    # perceptron circuit in circuit.py uses the following names for wires
    # x = "input0"
    # y = "input1"
    # wi = "input2"
    # bi = "input3"
    #
    # wo = "output0"
    # bo = "output1"

    # need to make dimenions of w the same as x
    if initial_w == 0:
        first_x = data[0][0]
        initial_w = np.zeros(len(first_x))

    w = initial_w
    b = initial_b

    # used fixed point arithmetic, accurate up to fp_precision decimal places
    # need to scale x,y up by 10^fp_precision
    # after every mult, numbers will be scaled back down by 10^fp_precision
    scale = 10**fp_precision

    for i in range(num_iterations):

        if i < 0:
            VERBOSE = True
        else:
            VERBOSE = False

        (x,y) = data[i]
        x = x*scale
        y = y*scale
        cur_input = {}
        cur_input["input0"] = x
        cur_input["input1"] = y
        cur_input["input2"] = w
        cur_input["input3"] = b

        evaluator.load_inputs(cur_input)
        evaluator.run(verbose=VERBOSE)

        [w,b] = evaluator.get_outputs()


    return (w / scale,b / scale)

def secure_eval_circuit(data,num_iterations,modulus,initial_w=0,initial_b=0,fp_precision=16):

    scale = 10**fp_precision

    oracle = Oracle(modulus,fp_precision=fp_precision)

    evaluator1 = SecureEvaluator(circuit.circuit,circuit.gate_order,1,oracle)
    evaluator2 = SecureEvaluator(circuit.circuit,circuit.gate_order,2,oracle)
    evaluator3 = SecureEvaluator(circuit.circuit,circuit.gate_order,3,oracle)

    parties = [evaluator1,evaluator2,evaluator3]


    dealer = Dealer(parties,modulus,fp_precision=fp_precision)

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
        data3x.append(data[2*split + 1][0])
        data3y.append(data[2*split + 1][1])

    dealer.distribute_shares(data1x)
    dealer.distribute_shares(data2x)
    dealer.distribute_shares(data3x)

    dealer.distribute_shares(data1y)
    dealer.distribute_shares(data2y)
    dealer.distribute_shares(data3y)

    num_randomness = 12 * num_iterations
    dealer.generate_randomness(num_randomness)

    # need to make dimenions of w the same as x
    if initial_w == 0:
        first_x = data[0][0]
        initial_w = np.zeros(len(first_x))
    initial_w = [initial_w,[]]
    dealer.distribute_shares(initial_w)
    dealer.distribute_shares(initial_b)

    results = {}
    wire_dict = {}

    for i in range(num_iterations):


        if i < 0:
            VERBOSE = True
        else:
            VERBOSE = False
        
        t1 = Thread(target=run_eval,args=(evaluator1,i,data_len,results,1,fp_precision,VERBOSE,wire_dict))
        t2 = Thread(target=run_eval,args=(evaluator2,i,data_len,results,2,fp_precision,VERBOSE,wire_dict))
        t3 = Thread(target=run_eval,args=(evaluator3,i,data_len,results,3,fp_precision,VERBOSE,wire_dict))

        t1.start()
        t2.start()
        t3.start()

        t1.join()
        t2.join()
        t3.join()


    (w,b) = get_w_b(results)
    return (w / scale, b / scale)

def unshare(share1,share2):

    print(share1)
    print(share2)

    if type(share1) == tuple:
        (x,a) = share1
        (y,b) = share2
        return x - b
    else:
        res = []
        for i in range(len(share1)):
            (x,a) = share1[i]
            (y,b) = share2[i]
            res.append(x - b)
        return res

def get_w_b(w_b_shares):

    w1 = w_b_shares[1]['w']
    b1 = w_b_shares[1]['b']
    w2 = w_b_shares[2]['w']
    b2 = w_b_shares[2]['b']
    w3 = w_b_shares[3]['w']
    b3 = w_b_shares[3]['b']

    w = np.array([w1[0][0]-w2[0][1],w1[1][0]-w2[1][1]])
    b = b1[0] - b2[1]

    return (w,b)


def run_eval(evaluator,iter_num,data_length,results_dict,party_index,fp_precision=16,verbose=False,wd={}):

    scale = 10**fp_precision

    # input will map wire name to index in list of shares
    cur_input = {}
    cur_input["input0"] = iter_num
    cur_input["input1"] = data_length + iter_num

    # only load initial b and w
    if iter_num == 0:
        cur_input["input2"] = -2
        cur_input["input3"] = -1

    evaluator.load_secure_inputs(cur_input)
    evaluator.run(verbose=verbose)

    #[wo,bo] = evaluator.get_outputs()
    #[(w00,w01),(w10,w11)] = wo
    #(b0,b1) = bo
    
    #w = [(w00/scale,w01/scale),(w10/scale,w11/scale)]
    #b = (b0/scale,b1/scale)

    [w,b] = evaluator.get_outputs()

    if iter_num == 0:
        #print(str(party_index))
        #print(evaluator.get_wire_dict())
        wd[party_index] = dict(evaluator.get_wire_dict())

    cur_in = {}
    cur_in["input2"] = w
    cur_in["input3"] = b
    evaluator.load_inputs(cur_in)

    results_dict[party_index] = {"w": w, "b": b}



if __name__ == "__main__":

    import data.iris_data as iris

    data = iris.get_iris_data()

    num_iter = len(data)

    print(eval_circuit(data,num_iter))

    print(secure_eval_circuit(data,num_iter,10**32,fp_precision=16))
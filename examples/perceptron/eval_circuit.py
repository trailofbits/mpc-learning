from src.circuits.evaluator import BasicEvaluator
import circuit
import numpy as np

def eval_circuit(data,num_iterations,fp_precision=16):

    evaluator = BasicEvaluator(circuit.circuit,circuit.gate_order)

    # perceptron circuit in circuit.py uses the following names for wires
    # x = "input0"
    # y = "input1"
    # wi = "input2"
    # bi = "input3"
    #
    # wo = "output0"
    # bo = "output1"

    initial_w = np.zeros(2)
    initial_b = 0

    w = initial_w
    b = initial_b

    # used fixed point arithmetic, accurate up to fp_precision decimal places
    # need to scale x,y up by 10^fp_precision
    # after every mult, numbers will be scaled back down by 10^fp_precision
    scale = 10**fp_precision

    for i in range(num_iterations):
        (x,y) = data[i]
        x = x*scale
        y = y*scale
        cur_input = {}
        cur_input["input0"] = x
        cur_input["input1"] = y
        cur_input["input2"] = w
        cur_input["input3"] = b

        evaluator.load_inputs(cur_input)
        evaluator.run()

        [w,b] = evaluator.get_outputs()

    return (w,b)

if __name__ == "__main__":

    import data.iris_data as iris

    data = iris.get_iris_data()

    num_iter = len(data)

    print(eval_circuit(data,num_iter))
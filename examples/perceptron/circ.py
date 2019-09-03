from src.circuits.gate import Gate
from queue import Queue

# this file contains the circuit for one iteration of the perceptron algorithm
# the circuit is represented via a python dictionary
# each wire is given a unique ID, and each gate is given a unique ID and label
# they unique ID for the gate will serve as its key
# the IDs will be assigned as g1 for gate one, g2 ...
# the values of the dictionary correspond to the gate type and in/out wires

# here is a summary of the perceptron algorithm

# initialize w,b (for us, we take w,b to be random integer in {-1,1})
# for each data point (x,y):
#   if y * ( dotproduct(w,x) + b ) <= 0:
#       w = w + yx
#       b = b + y
#   else:
#       w = w
#       b = b

# this circuit will consist of the code block below the for loop
# there are four input values for each iteration: x, y, w, and b
# they are labeled as follows:
# x -> input0
# y -> input1
# w -> input2
# b -> input3
# there are two output values: w and b
# they are labeled as follows:
# w -> output0
# b -> output1
# the additional intermediate wires will be given the lable zi for all i

# we use the following gate labels: ADD, MULT, SMULT, COMP, DOT, and NOT
# ADD is the addition gate
# MULT is the multiplication gate
# SMULT is the scalar multiplication gate
# COMP is the comparison gate, which computes the boolean (input <= 0)
# DOT is the dot product gate, which computes the dot product of two inputs
# NOT is the not gate, which computes 1 - input (input 0 or 1 here)

x = "input0"
y = "input1"
wi = "input2"
bi = "input3"

wo = "output0"
bo = "output1"

circuit = {}

# specify input, intermediate, and output wires
circuit["input"] = [x,y,wi,bi]
circuit["output"] = [wo,bo]
wires = []
for i in range(12):
    wires.append("z"+str(i))
circuit["wires"] = wires

# specify order of gates to be evaluated
gate_order = []
for i in range(14):
    gate_order.append("g"+str(i))

Q = Queue()

x = Gate("in0","INPUT",[],Q)
y = Gate("in1","INPUT",[],Q)
wi = Gate("in2","INPUT",[],Q)
bi = Gate("in3","INPUT",[],Q)

# gate for dot product of x and w
g0 = Gate("g0","DOT",[x.get_id(),wi.get_id()],Q)

# gate for adding b to dot product of x and w
g1 = Gate("g1","ADD",[bi.get_id(),g0.get_id()],Q)

# gate for multiplying y with b + dot(x,w)
g2 = Gate("g2","MULT",[y.get_id(),g1.get_id()],Q)

# gate for computing y(b + dot(x,w)) <= 0
g3 = Gate("g3","COMP",[g2.get_id()],Q)

# gate for computing y*x for conditional assignment to w
g4 = Gate("g4","SMULT",[y.get_id(),x.get_id()],Q)

# gate for computing w + x*y for conditional assignment to w
g5 = Gate("g5","ADD",[wi.get_id(),g4.get_id()],Q)

# gate for computing b + y for conditional assignment to b
g6 = Gate("g6","ADD",[bi.get_id(),y.get_id()],Q)

# gate for computing not of if statement
g7 = Gate("g7","NOT",[g3.get_id()],Q)

# gate for computing if conditional assignment to w
g8 = Gate("g8","SMULT",[g3.get_id(),g5.get_id()],Q)

# gate for computing else conditional assignment to w
g9 = Gate("g9","SMULT",[g7.get_id(),wi.get_id()],Q)

# gate for computing output for w
g10 = Gate("g10","ADD",[g8.get_id(),g9.get_id()],Q)

# gate for computing if conditional assignment to b
g11 = Gate("g11","MULT",[g3.get_id(),g6.get_id()],Q)

# gate for computing else conditional assignment to b
g12 = Gate("g12","MULT",[bi.get_id(),g7.get_id()],Q)

# gate for computing output for b
g13 = Gate("g13","ADD",[g11.get_id(),g12.get_id()],Q)

# output values
wo = Gate("out0","OUTPUT",[g10.get_id()],Q)
bo = Gate("out1","OUTPUT",[g13.get_id()],Q)

circuit = {}

circuit[x.get_id()] = [g0,g4]
circuit[y.get_id()] = [g2,g4,g6]
circuit[wi.get_id()] = [g0,g5,g9]
circuit[bi.get_id()] = [g1,g6,g12]
circuit[g0.get_id()] = [g1]
circuit[g1.get_id()] = [g2]
circuit[g2.get_id()] = [g3]
circuit[g3.get_id()] = [g7,g8,g11]
circuit[g4.get_id()] = [g5]
circuit[g5.get_id()] = [g8]
circuit[g6.get_id()] = [g11]
circuit[g7.get_id()] = [g9,g12]
circuit[g8.get_id()] = [g10]
circuit[g9.get_id()] = [g10]
circuit[g10.get_id()] = [wo]
circuit[g11.get_id()] = [g13]
circuit[g12.get_id()] = [g13]
circuit[g13.get_id()] = [bo]

in_gates = [x,y,wi,bi]
out_gates = [wo,bo]
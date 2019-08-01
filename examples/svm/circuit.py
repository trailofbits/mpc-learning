from src.circuits.gate import Gate
from queue import Queue

# this file contains the circuit for one iteration of a support vector machine
# here we use sub gradient descent
# the circuit is represented via a python dictionary
# each gate is given a unique ID and label
# the unique ID for the gate will serve as its key
# the IDs will be assigned as g1 for gate one, g2 ...
# the values of the dictionary correspond to which gates the output of the key 
# will input into
# for example circuit["g1"] = [g2,g3] means that the output of gate "g1"
# will serve as input for both gates "g2" and "g3"

# here is a summary of the algorithm

# first, we augment the data such that we set all x values to be x' = [x,1]
# this will then solve for a w', where w' = [w,b]
# after augmenting the data, we do the following

# initialize w,b (for us, we take w,b to be random integer in {-1,1})
# for each data point (x',y):
#   if y * ( dotproduct(w',x') ) <= 1:
#       w' = (1 - gamma) * w' + gamma * C * y * x'
#   else:
#       w' = (1 - gamma) * w'
#
# here gamma is the "learning rate", and C is a hyper-parameter
# we chose the following values:
# gamma = 1 / (1 + [iteration_number])
# C = 1

# this circuit will consist of the code block below the for loop
# there are 5 input values for each iteration: x', y, w', (1 - gamma), gamma*C
# they are labeled as follows:
# x' -> input0
# y -> input1
# w' -> input2
# (1 - gamma) -> input3
# gamma * C -> input4

# there is one output value: w'
# labeled as follows:
# w' -> output0
# the additional intermediate wires will be given the lable gi for all i

# we use the following gate labels: ADD, MULT, SMULT, COMP, DOT, NOT, CMULT, CADD
# ADD is the addition gate
# MULT is the multiplication gate
# SMULT is the scalar multiplication gate
# COMP is the comparison gate, which computes the boolean (input <= 0)
# DOT is the dot product gate, which computes the dot product of two inputs
# NOT is the not gate, which computes 1 - input (input 0 or 1 here)
# CMULT is multiplying by a constant (i.e. multiply share of x by public c val)
# CADD is addition by constant (i.e. adding a public c val to a share of x)

x = Gate("in0","INPUT",[])
y = Gate("in1","INPUT",[])
wi = Gate("in2","INPUT",[])
gamma1 = Gate("in3","INPUT",[])
gammaC = Gate("in4","INPUT",[])
minus1 = Gate("in5","INPUT",[])

# gate for dot product of x' and w'
g0 = Gate("g0","DOT",[x.get_id(),wi.get_id()])

# gate for multiplying y with dot product of x' and w'
g1 = Gate("g1","MULT",[y.get_id(),g0.get_id()])

# gate for subtracting 1 from y*dot(x',w')
g2 = Gate("g2","CADD",[g1.get_id(),minus1.get_id()],const_input=minus1.get_id())

# gate for comparing y*dot(x',w') - 1 <= 0
g3 = Gate("g3","COMP",[g2.get_id()])

# gate for multipling y with gamma*C
g4 = Gate("g4","CMULT",[y.get_id(),gammaC.get_id()],const_input=gammaC.get_id())

# gate for multiplying y*gamma*C with x'
g5 = Gate("g5","SMULT",[g4.get_id(),x.get_id()])

# gate for multiplying w' with (1 - gamma)
g6 = Gate("g6","CMULT",[wi.get_id(),gamma1.get_id()],const_input=gamma1.get_id())

# gate for adding (1-gamma)w' with y*gamma*C*x'
g7 = Gate("g7","ADD",[g5.get_id(),g6.get_id()])

# gate for multiplying result of comparison with g7 output
g8 = Gate("g8","SMULT",[g3.get_id(),g7.get_id()])

# gate for computing NOT of comparison (for multiplexing)
g9 = Gate("g9","NOT",[g3.get_id()])

# gate for multiplying g9 (not of comparison) with (1-gamma)w'
g10 = Gate("g10","SMULT",[g9.get_id(),g6.get_id()])

# gate for adding two multiplexes
g11 = Gate("g11","ADD",[g8.get_id(),g10.get_id()])

# gate for rounding results
g12 = Gate("g12","ROUND",[g11.get_id()])

# gate for outputing w'
wo = Gate("out0","OUTPUT",[g12.get_id()])

circuit = {}

circuit[x.get_id()] = [g0,g5]
circuit[y.get_id()] = [g1,g4]
circuit[wi.get_id()] = [g0,g6]
circuit[gamma1.get_id()] = [g6]
circuit[gammaC.get_id()] = [g4]
circuit[minus1.get_id()] = [g2]
circuit[g0.get_id()] = [g1]
circuit[g1.get_id()] = [g2]
circuit[g2.get_id()] = [g3]
circuit[g3.get_id()] = [g8,g9]
circuit[g4.get_id()] = [g5]
circuit[g5.get_id()] = [g7]
circuit[g6.get_id()] = [g7,g10]
circuit[g7.get_id()] = [g8]
circuit[g8.get_id()] = [g11]
circuit[g9.get_id()] = [g10]
circuit[g10.get_id()] = [g11]
circuit[g11.get_id()] = [g12]
circuit[g12.get_id()] = [wo]

in_gates = [x,y,wi,gamma1,gammaC,minus1]
out_gates = [wo]
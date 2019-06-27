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

# we use the following gate labels: ADD, MULT, COMP, DOT, and NOT
# ADD is the addition gate
# MULT is the multiplication gate
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

# gate for dot product of x and w
circuit["g0"] = { "type": "DOT", "input": [x,wi], "output": ["z0"] }

# gate for adding b to dot product of x and w
circuit["g1"] = { "type": "ADD", "input": [bi,"z0"], "output": ["z1"] }

# gate for multiplying y with b + dot(x,w)
circuit["g2"] = { "type": "MULT", "input": [y,"z1"], "output": ["z2"] }

# gate for computing y(b + dot(x,w)) <= 0
circuit["g3"] = { "type": "COMP", "input": ["z2"], "output": ["z3"] }

# gate for computing x*y for conditional assignment to w
circuit["g4"] = { "type": "NOT", "input": [x,y], "output": ["z4"] }

# gate for computing w + x*y for conditional assignment to w
circuit["g5"] = { "type": "ADD", "input": [wi,"z4"], "output": ["z5"] }

# gate for computing b + y for conditional assignment to b
circuit["g6"] = { "type": "ADD", "input": [bi,y], "output": ["z6"] }

# gate for computing not of if statement
circuit["g7"] = { "type": "NOT", "input": ["z3"], "output": ["z7"] }

# gate for computing if conditional assignment to w
circuit["g8"] = { "type": "MULT", "input": ["z3","z5"], "output": ["z8"] }

# gate for computing else conditional assignment to w
circuit["g9"] = { "type": "MULT", "input": [wi,"z7"], "output": ["z9"] }

# gate for computing output for w
circuit["g10"] = { "type": "ADD", "input": ["z8","z9"], "output": [wo] }

# gate for computing if conditional assignment to b
circuit["g11"] = { "type": "MULT", "input": ["z3","z6"], "output": ["z10"] }

# gate for computing else conditional assignment to b
circuit["g12"] = { "type": "MULT", "input": [bi,"z7"], "output": ["z11"] }

# gate for computing output for b
circuit["g13"] = { "type": "ADD", "input": ["z10","z11"], "output": [bo] }

#print(circuit)
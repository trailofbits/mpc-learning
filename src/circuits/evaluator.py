import numpy as np

class Evaluator:
    def __init__(self, circuit, gate_order, fp_precision=16):
        self.circuit = circuit
        self.gate_order = gate_order
        self.scale = 10**fp_precision
        self.party_index = ""

        self.inputs = self.circuit["input"]
        self.outputs = self.circuit["output"]
        self.wire_dict = {}
        for inp in self.inputs:
            self.wire_dict[inp] = ""
        for outp in self.outputs:
            self.wire_dict[outp] = ""
        for intermediate in self.circuit["wires"]:
            self.wire_dict[intermediate] = ""
    
    def load_inputs(self, inputs):
        for wire_key in inputs:
            self.wire_dict[wire_key] = inputs[wire_key]

    def run(self,verbose=False):
        for gate in self.gate_order:
            #print("begin " + str(self.circuit[gate]["type"]) + " gate " + str(gate))
            self._eval_gate(gate,verbose=verbose)
            #("finished evaluating " + str(self.circuit[gate]["type"]) + " gate " + str(gate))
    
    def _eval_gate(self,gate,verbose=False):
            gate_type = self.circuit[gate]["type"]
            gate_input = self.circuit[gate]["input"]
            gate_output = self.circuit[gate]["output"]

            if gate_type == "ADD":
                self._add(gate_input,gate_output)
            elif gate_type == "MULT":
                self._mult(gate_input,gate_output)
            elif gate_type == "SMULT":
                self._smult(gate_input,gate_output)
            elif gate_type == "DOT":
                self._dot(gate_input,gate_output)
            elif gate_type == "NOT":
                self._not(gate_input,gate_output)
            elif gate_type == "COMP":
                self._comp(gate_input,gate_output)
            else:
                raise(Exception('{} is not a valid gate type'.format(gate_type)))

            if verbose:
                print("party index: " + str(self.party_index))
                print("gate type: " + gate_type)
                print("input wire labels: " + str(gate_input))
                for wire in gate_input:
                    print("wire " + wire + " value: " + str(self.wire_dict[wire]))
                [z] = gate_output
                print("output wire label: " + str(z))
                print("output wire value: " + str(self.wire_dict[z]))

    def _add(self, wire_in, wire_out):
        pass

    def _mult(self, wire_in, wire_out):
        pass

    def _smult(self, wire_in, wire_out):
        pass

    def _dot(self, wire_in, wire_out):
        pass

    def _not(self, wire_in, wire_out):
        pass

    def _comp(self, wire_in, wire_out):
        pass


class BasicEvaluator(Evaluator):    
    def _add(self, wire_in, wire_out):
        [x,y] = wire_in
        [z] = wire_out

        x_val = self.wire_dict[x]
        y_val = self.wire_dict[y]

        self.wire_dict[z] = x_val + y_val

    def _mult(self, wire_in, wire_out):
        [x,y] = wire_in
        [z] = wire_out

        x_val = self.wire_dict[x]
        y_val = self.wire_dict[y]

        self.wire_dict[z] = (x_val / self.scale) * y_val

    def _smult(self, wire_in, wire_out):
        [x,y] = wire_in
        [z] = wire_out

        x_val = self.wire_dict[x]
        y_val = np.array(self.wire_dict[y])

        self.wire_dict[z] = (x_val * y_val) / self.scale

    def _dot(self, wire_in, wire_out):
        [x,y] = wire_in
        [z] = wire_out

        x_val = np.array(self.wire_dict[x])
        y_val = np.array(self.wire_dict[y])
        
        self.wire_dict[z] = (np.dot(x_val,y_val)) / self.scale

    def _not(self, wire_in, wire_out):
        [x] = wire_in
        [z] = wire_out

        x_val = self.wire_dict[x]

        self.wire_dict[z] = self.scale - x_val

    def _comp(self, wire_in, wire_out):
        [x] = wire_in
        [z] = wire_out

        x_val = self.wire_dict[x]

        self.wire_dict[z] = int(x_val <= 0) * self.scale

    def get_outputs(self):
        outs = []
        for out in self.outputs:
            outs.append(self.wire_dict[out])
        return outs

    

class SecureEvaluator(Evaluator):

    def __init__(self,circuit,gate_order,party_index,oracle):
        Evaluator.__init__(self,circuit,gate_order)
        self.party_index = party_index
        self.oracle = oracle
        self.input_shares = []

    def add_parties(self,parties):
        self.parties = {}
        self.parties[self.party_index] = self
        for (party,party_index) in parties:
            if party_index in self.parties:
                raise Exception("Party number: {} already exists".format(party_index))
            else:
                self.parties

    def receive_randomness(self,random_values):
        self.randomness = random_values
        self.random_index = 0
        self.interaction = {}
        for i in range(len(self.randomness)):
            self.interaction[i] = "wait"            

    def receive_shares(self,shares):
        for share in shares:
            self.input_shares.append(share)

    def load_secure_inputs(self,inputs):
        for wire_key in inputs:
            inputs[wire_key] = self.input_shares[inputs[wire_key]]

        self.load_inputs(inputs)

    def _add(self, wire_in, wire_out):
        [x,y] = wire_in
        [z] = wire_out

        if type(self.wire_dict[x]) == list:
            z_vals = []

            for i in range(len(self.wire_dict[x])):
                (x_val,a_val) = self.wire_dict[x][i]
                (y_val,b_val) = self.wire_dict[y][i]
                z_vals.append((x_val + y_val,a_val + b_val))

            self.wire_dict[z] = z_vals

        else:

            (x_val,a_val) = self.wire_dict[x]
            (y_val,b_val) = self.wire_dict[y]

            self.wire_dict[z] = (x_val + y_val,a_val + b_val)

    def _mult(self, wire_in, wire_out):
        [x,y] = wire_in
        [z] = wire_out

        rindex = self.random_index
        cur_random_val = self.randomness[rindex]

        #(x_val,a_val) = self.wire_dict[x]
        #(y_val,b_val) = self.wire_dict[y]
        xa = self.wire_dict[x]
        yb = self.wire_dict[y]

        self.oracle.send_mult([xa,yb],self.party_index,rindex)

        out_val = self.oracle.receive_mult(self.party_index,rindex)
        while out_val == "wait":
            out_val = self.oracle.receive_mult(self.party_index,rindex)

        #r = a_val * b_val / self.scale
        #r += x_val * y_val / self.scale
        #r += cur_random_val
        #r = r / 3

        #if self.party_index == 1:
        #    self._send_share(r,2,self.random_index)
        #elif self.party_index == 2:
        #    self._send_share(r,3,self.random_index)
        #elif self.party_index == 3:
        #    self._send_share(r,1,self.random_index)

        # must wait until we receive share from party
        #while self.interaction[self.random_index] == "wait":
        #    pass

        #new_r = self.interaction[self.random_index]

        #print(new_r)

        #self.wire_dict[z] = (new_r - r, -2 * new_r - r)

        self.wire_dict[z] = out_val

        self.random_index += 1

    def _smult(self, wire_in, wire_out):
        [x,y] = wire_in
        [z] = wire_out

        #(x_val,a_val) = self.wire_dict[x]
        #(y_val,b_val) = self.wire_dict[y]

        #z_vals = []

        #for (y_val,b_val) in self.wire_dict[y]:
        #    z_vals.append((((x_val / self.scale) * y_val),((a_val / self.scale) * b_val)))

        #self.wire_dict[z] = z_vals

        rindex = self.random_index

        cur_random_val = self.randomness[rindex]

        xa = self.wire_dict[x]
        ybv = self.wire_dict[y]

        self.oracle.send_smult([xa,ybv],self.party_index,rindex)

        out_val = self.oracle.receive_smult(self.party_index,rindex)
        while out_val == "wait":
            out_val = self.oracle.receive_smult(self.party_index,rindex)
        
        self.wire_dict[z] = out_val

        self.random_index += 1

    def _dot(self, wire_in, wire_out):
        [x,y] = wire_in
        [z] = wire_out

        rindex = self.random_index

        cur_random_val = self.randomness[rindex]

        (x_val,a_val) = self.wire_dict[x]
        (y_val,b_val) = self.wire_dict[y]

        xa = self.wire_dict[x]
        yb = self.wire_dict[y]

        self.oracle.send_dot([xa,yb],self.party_index,rindex)

        out_val = self.oracle.receive_dot(self.party_index,rindex)
        while out_val == "wait":
            out_val = self.oracle.receive_dot(self.party_index,rindex)
        
        self.wire_dict[z] = out_val

        self.random_index += 1

    def _not(self, wire_in, wire_out):
        [x] = wire_in
        [z] = wire_out

        (x_val,a_val) = self.wire_dict[x]

        self.wire_dict[z] = (-x_val, -(a_val + 1*self.scale))

    def _comp(self, wire_in, wire_out):
        [x] = wire_in
        [z] = wire_out

        rindex = self.random_index
        cur_random_val = self.randomness[rindex]

        (x_val,a_val) = self.wire_dict[x]
        xa = self.wire_dict[x]

        self.oracle.send_comp([xa],self.party_index,rindex)

        out_val = self.oracle.receive_comp(self.party_index,rindex)
        while out_val == "wait":
            out_val = self.oracle.receive_comp(self.party_index,rindex)

        self.wire_dict[z] = out_val

        self.random_index += 1

    def _send_share(self, value, party_index, random_index):
        receiver = self.parties[party_index]
        receiver._receive_party_share(value,random_index)

    def _receive_party_share(self, share, random_index):
        self.interaction[random_index] = share

    def get_outputs(self):
        outs = []
        for out in self.outputs:
            outs.append(self.wire_dict[out])
        return outs

    def get_wire_dict(self):
        return self.wire_dict
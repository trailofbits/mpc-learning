import numpy as np
from src.circuits.share import Share

class Evaluator:
    """
    Generic evaluator class that serves as an interface for other evaluators.
    This interface is designed to be objects that take a function (represent
    as a circuit) and computes the ouput based on input wires.

    Methods
    -------
    __init__(self, circuit, gate_order, fp_precision=16)
        Evaluator object constructor
        Initializes the circuit and gate order and scale (10^fp_precision)
        Initializes dictionary for wire_values
        Initializes list of input and output wires

        Parameters
        ----------
        circuit: dictionary
            Circuit to be used to compute function, assumed to be in correct
            format- i.e. contains the following (key,value) pairs:
            ("input",<list of input wire names>)
            ("output",<list of output wire names>)
            ("wires",<list of intermediate wire names>)
            (<gate>,dict({"type": <gate-type>, "input": <input wire list>,
            "output": <output wire list>})) (for each gate in circuit)
        gate_order: iterable (ordered)
            Iterable representing the order in which circuit gates should be
            evaluated. Each gate in gate_order should be a key in the circuit
        (optional) fp_precision=16: int
            Fixed point number precision

    load_inputs(self, inputs)
        Initialzes inputs to be used for computing the circuit

        Parameters
        ----------
        inputs: dictionary
            Inputs to be used to compute circuit, assumed to be in correct
            format- i.e. contains following (key,value) pairs:
            (<input-wire>,<input-wire-value>) for each <input-wire> in
            circuit
    
    run(self, verbose=False)
        Computes the circuit using the loaded inputs

        Parameters
        ----------
        (optional) verbose: boolean
            Optional argument to display intermediate wire values for debugging
        
    _eval_gate(self, gate, verbose=False)
        Evaluate an individual gate of the circuit

        Parameters
        ----------
        gate: dictionary
            Gate to be computed, assumed to be in correct format- i.e. contains
            the following (key,value) pairs:
            ("type",<gate-type>) where <gate-type> can be "ADD", "MULT", 
            "SMULT", "DOT", "NOT", or "COMP"
            ("input",<input-wire-name-list>)
            ("output",<output-wire-name-list)
        (optional) verbose=False: boolean
            Optional argument to display intermediate wire values for debugging

    _add(self, wire_in, wire_out)
        Method specifying how to compute ADD gate
        
        Parameters
        ----------
        wire_in: iterable
            Iterable of input wire names
        wire_out: iterable
            Iterable of output wire names

    _mult(self, wire_in, wire_out)
        Method specifying how to compute MULT gate
        
        Parameters
        ----------
        wire_in: iterable
            Iterable of input wire names
        wire_out: iterable
            Iterable of output wire names

    _smult(self, wire_in, wire_out)
        Method specifying how to compute SMULT gate
        
        Parameters
        ----------
        wire_in: iterable
            Iterable of input wire names
        wire_out: iterable
            Iterable of output wire names

    _dot(self, wire_in, wire_out)
        Method specifying how to compute DOT gate
        
        Parameters
        ----------
        wire_in: iterable
            Iterable of input wire names
        wire_out: iterable
            Iterable of output wire names

    _not(self, wire_in, wire_out)
        Method specifying how to compute NOT gate
        
        Parameters
        ----------
        wire_in: iterable
            Iterable of input wire names
        wire_out: iterable
            Iterable of output wire names

    _comp(self, wire_in, wire_out)
        Method specifying how to compute COMP gate
        
        Parameters
        ----------
        wire_in: iterable
            Iterable of input wire names
        wire_out: iterable
            Iterable of output wire names

    """
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
            self._eval_gate(gate,verbose=verbose)
            
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
    """
    Inherits from Evaluator. BasicEvaluator computes the circuit in the
    straightforward fashion. The input values are potentially floating point
    numbers. These values are multiplied by 10^fp_precision (fixed point 
    precision) and then treated as integers. To account for the fixed point
    precision, we need to divide by 10^fp_precision after every multiplication.

    Methods
    -------
    _add(self, wire_in, wire_out)
        Method specifying how to compute ADD gate
        
        Parameters
        ----------
        wire_in: iterable
            Iterable of input wire names
        wire_out: iterable
            Iterable of output wire names

    _mult(self, wire_in, wire_out)
        Method specifying how to compute MULT gate
        
        Parameters
        ----------
        wire_in: iterable
            Iterable of input wire names
        wire_out: iterable
            Iterable of output wire names

    _smult(self, wire_in, wire_out)
        Method specifying how to compute SMULT gate
        
        Parameters
        ----------
        wire_in: iterable
            Iterable of input wire names
        wire_out: iterable
            Iterable of output wire names

    _dot(self, wire_in, wire_out)
        Method specifying how to compute DOT gate
        
        Parameters
        ----------
        wire_in: iterable
            Iterable of input wire names
        wire_out: iterable
            Iterable of output wire names

    _not(self, wire_in, wire_out)
        Method specifying how to compute NOT gate
        
        Parameters
        ----------
        wire_in: iterable
            Iterable of input wire names
        wire_out: iterable
            Iterable of output wire names

    _comp(self, wire_in, wire_out)
        Method specifying how to compute COMP gate
        
        Parameters
        ----------
        wire_in: iterable
            Iterable of input wire names
        wire_out: iterable
            Iterable of output wire names

    get_outputs(self)
        Getter for outputs of circuit

        Returns
        -------
        outs: list
            List of output values from the computation of the circuit
    """

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
    """
    Inherits from Evaluator. SecureEvaluator computes the circuit using
    an MPC protocol, where the wires are shares of the inputs. These values 
    are multiplied by 10^fp_precision (fixed point precision) and then treated
    as integers. To account for the fixed point precision, we need to divide 
    by 10^fp_precision after every multiplication.
    The current implementation relies on a Dealer to distribute shares of
    input values and an Oracle to compute operations that require interaction
    between parties (i.e. MULT, SMULT, DOT, and COMP)

    Methods
    -------
    __init__(self,circuit,gate_order,party_index,oracle,fp_precision=16)
        Constructor for SecureEvaluator object
        Overrides Evaluator constructor
        Initializes party_index and oracle

        Parameters
        ----------
        circuit: dictionary
            Circuit to be used to compute function, assumed to be in correct
            format- i.e. contains the following (key,value) pairs:
            ("input",<list of input wire names>)
            ("output",<list of output wire names>)
            ("wires",<list of intermediate wire names>)
            (<gate>,dict({"type": <gate-type>, "input": <input wire list>,
            "output": <output wire list>})) (for each gate in circuit)
        gate_order: iterable (ordered)
            Iterable representing the order in which circuit gates should be
            evaluated. Each gate in gate_order should be a key in the circuit
        party_index: int
            Integer representing index of evaluator object
            (must be 1, 2, or 3 for current implementation)
        oracle: oracle object
            Oracle that will perform computation for MULT, SMULT, DOT, and COMP
        (optional) fp_precision=16: int
            Fixed point number precision

    add_parties(self,parties)
        Method to initialize other Evaluator objects and their party index

        Parameters
        ----------
        parties: iterable of pairs
            Iterable containing pair of (Evaluator object, party index)

    receive_randomness(self,random_values)
        Method for getting random values to be used for interactive computation
        Current iteration receives random values from dealer
        
        Parameters
        ----------
        random_values: iterable
            Iterable containing random values to be used for interactive
            computation. According to current protocol, each of the three
            parties will contain a random value x_i such that:
            x_1 + x_2 + x_3 = 0

    receive_shares(self,shares)
        Method for getting shares of input values
        Current iteration receives shares from dealer

        Parameters
        ----------
        shares: iterable
            Iterable containing shares of input values. According to current
            protocol, each share is a pair of values.

    load_secure_inputs(self,inputs)
        Method for loading input shares into Evaluator object. This makes
        secure evaluation compatible with Evaluator interface.

        Parameters
        ----------
        inputs: dictionary
            Dictionary mapping input wires to their index in the list of
            input shares (self.shares). This index is used to obtain
            the desired share value and load it into input wire.
    
    _add(self, wire_in, wire_out)
        Method specifying how to compute ADD gate
        
        Parameters
        ----------
        wire_in: iterable
            Iterable of input wire names
        wire_out: iterable
            Iterable of output wire names

    _mult(self, wire_in, wire_out)
        Method specifying how to compute MULT gate
        
        Parameters
        ----------
        wire_in: iterable
            Iterable of input wire names
        wire_out: iterable
            Iterable of output wire names

    _smult(self, wire_in, wire_out)
        Method specifying how to compute SMULT gate
        
        Parameters
        ----------
        wire_in: iterable
            Iterable of input wire names
        wire_out: iterable
            Iterable of output wire names

    _dot(self, wire_in, wire_out)
        Method specifying how to compute DOT gate
        
        Parameters
        ----------
        wire_in: iterable
            Iterable of input wire names
        wire_out: iterable
            Iterable of output wire names

    _not(self, wire_in, wire_out)
        Method specifying how to compute NOT gate
        
        Parameters
        ----------
        wire_in: iterable
            Iterable of input wire names
        wire_out: iterable
            Iterable of output wire names

    _comp(self, wire_in, wire_out)
        Method specifying how to compute COMP gate
        
        Parameters
        ----------
        wire_in: iterable
            Iterable of input wire names
        wire_out: iterable
            Iterable of output wire names

    _send_share(self, value, party_index, random_index)
        Method for sending value to other Evaluator parties

        Parameters
        ----------
        value: int
            Value to be send to other party
        party_index: int
            party_index of Evaluator to receive value
        random_index: int
            Index representing which random interactive value to use

    _receive_party_share(self, share, random_index)
        Method for receiving value from another Evaluator party

        Parameters
        ----------
        share: int
            Value to be received from other party
        random_index:
            Index representing which random interactive value to use

    
    get_outputs(self)
        Getter for outputs of circuit

        Returns
        -------
        outs: list
            List of output values from the computation of the circuit

    get_wire_dict(self)
        Getter for dictionary of wire values

        Returns
        -------
        wire_dict: dictionary
            Dictionary of wire values (key: wire name, value: wire value)
    """

    def __init__(self,circuit,gate_order,party_index,oracle,fp_precision=16):
        Evaluator.__init__(self,circuit,gate_order,fp_precision=fp_precision)
        self.party_index = party_index
        self.oracle = oracle
        self.input_shares = []

    def add_parties(self,parties):
        self.parties = {}
        self.parties[self.party_index] = self
        for pindex in parties:
            if pindex == self.party_index:
                continue
            if pindex in self.parties:
                raise Exception("Party number: {} already exists".format(pindex))
            else:
                self.parties[pindex] = parties[pindex]

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
                z_vals.append(self.wire_dict[x][i] + self.wire_dict[y][i])

            self.wire_dict[z] = z_vals

        else:
            self.wire_dict[z] = self.wire_dict[x] + self.wire_dict[y]

    def _mult(self, wire_in, wire_out):
        [x,y] = wire_in
        [z] = wire_out

        rindex = self.random_index
        cur_random_val = self.randomness[rindex]

        x_val = self.wire_dict[x]
        y_val = self.wire_dict[y]

        r = x_val.pre_mult(y_val, cur_random_val)

        if self.party_index == 1:
            self._send_share(r,2,self.random_index)
        elif self.party_index == 2:
            self._send_share(r,3,self.random_index)
        elif self.party_index == 3:
            self._send_share(r,1,self.random_index)

        # must wait until we receive share from party
        while self.interaction[self.random_index] == "wait":
            pass

        new_r = self.interaction[self.random_index]
        self.wire_dict[z] = Share(new_r - r, -2 * new_r - r)
        self.random_index += 1

    def _smult(self, wire_in, wire_out):
        [x,y] = wire_in
        [z] = wire_out

        x_val = self.wire_dict[x]
        yvec = self.wire_dict[y]

        z_vals = []

        for i in range(len(yvec)):
            cur_random_val = self.randomness[self.random_index]

            y_val = yvec[i]

            r = x_val.pre_mult(y_val, cur_random_val)

            if self.party_index == 1:
                self._send_share(r,2,self.random_index)
            elif self.party_index == 2:
                self._send_share(r,3,self.random_index)
            elif self.party_index == 3:
                self._send_share(r,1,self.random_index)

            # must wait until we receive share from party
            while self.interaction[self.random_index] == "wait":
                pass

            new_r = self.interaction[self.random_index]
            z_vals.append(Share(new_r - r, -2 * new_r - r))

            self.random_index += 1

        self.wire_dict[z] = z_vals

    def _dot(self, wire_in, wire_out):
        [x,y] = wire_in
        [z] = wire_out

        xvec = self.wire_dict[x]
        yvec = self.wire_dict[y]

        z_val = Share(0,0)

        for i in range(len(xvec)):
            cur_random_val = self.randomness[self.random_index]

            x_val = xvec[i]
            y_val = yvec[i]

            r = x_val.pre_mult(y_val, cur_random_val)

            if self.party_index == 1:
                self._send_share(r,2,self.random_index)
            elif self.party_index == 2:
                self._send_share(r,3,self.random_index)
            elif self.party_index == 3:
                self._send_share(r,1,self.random_index)

            # must wait until we receive share from party
            while self.interaction[self.random_index] == "wait":
                pass

            new_r = self.interaction[self.random_index]
            z_val += Share(new_r - r, -2 * new_r - r)

            self.random_index += 1

        self.wire_dict[z] = z_val

    def _not(self, wire_in, wire_out):
        [x] = wire_in
        [z] = wire_out
        self.wire_dict[z] = self.wire_dict[x].not_op()

    def _comp(self, wire_in, wire_out):
        [x] = wire_in
        [z] = wire_out

        rindex = self.random_index
        cur_random_val = self.randomness[rindex]

        #(x_val,a_val) = self.wire_dict[x]
        xa = self.wire_dict[x]

        #self.oracle.send_comp([xa],self.party_index,rindex)
        self.oracle.send_op([xa],self.party_index,rindex,"COMP")

        #out_val = self.oracle.receive_comp(self.party_index,rindex)
        out_val = self.oracle.receive_op(self.party_index,rindex)
        while out_val == "wait":
            #out_val = self.oracle.receive_comp(self.party_index,rindex)
            out_val = self.oracle.receive_op(self.party_index,rindex)

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
       
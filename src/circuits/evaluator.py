import numpy as np
from src.circuits.share import Share
from queue import Queue
import time

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

    def __init__(self,circuit,input_gates,output_gates,party_index,oracle,fp_precision=16):
        #Evaluator.__init__(self,circuit,[],fp_precision=fp_precision)
        self.circuit = circuit
        self.scale = 10**fp_precision
        self.party_index = party_index
        self.oracle = oracle
        self.input_gates = {}
        for ing in input_gates:
            self.input_gates[ing.get_id()] = ing
        self.output_gates = output_gates
        self.input_shares = []
        self.q = Queue()
        #for gate in circuit:
        #    print("loading queue for gate: " + gate.get_id())
        #    gate.set_queue(self.q)
        self.outputs = {}
        for outg in self.output_gates:
            self.outputs[outg.get_id()] = ""

    def run(self, verbose=False):
        #if self.q.empty():
        #    raise(Exception('Queue empty, no inputs added'))

        i = 0
        
        gate = self.q.get()
        while gate != "FIN":
            self._eval_gate(gate)
            #print("gate: " + str(i) + " " + str(gate.get_id()))
            i += 1
            gate = self.q.get()
        
    def reset_circuit(self):
        self._clear_gates()

    def _clear_gates(self):
        # need to remove inputs from input gates
        for in_id in self.input_gates:
            self.input_gates[in_id].reset()
        
        # also reset all other gates
        for gid in self.circuit:
            for gate in self.circuit[gid]:
                gate.reset()

        self.outputs = {}
        for outg in self.output_gates:
            self.outputs[outg.get_id()] = ""

    def _eval_gate(self,gate,verbose=False):
            gate_type = gate.get_type()
            #gate_input = gate.inputs
            #gate_input = self.circuit[gate]["input"]
            #gate_output = self.circuit[gate]["output"]
            #gate_output = self.circuit[gate]

            if gate_type == "ADD":
                #self._add(gate_input,gate_output)
                self._add(gate)
            elif gate_type == "MULT":
                #self._mult(gate_input,gate_output)
                self._mult(gate)
            elif gate_type == "SMULT":
                #self._smult(gate_input,gate_output)
                self._smult(gate)
            elif gate_type == "DOT":
                #self._dot(gate_input,gate_output)
                self._dot(gate)
            elif gate_type == "NOT":
                #self._not(gate_input,gate_output)
                self._not(gate)
            elif gate_type == "COMP":
                #self._comp(gate_input,gate_output)
                self._comp(gate)
            elif gate_type == "CMULT":
                self._cmult(gate)
            elif gate_type == "CADD":
                self._cadd(gate)
            elif gate_type == "INPUT":
                #if gate.get_id() == "in2":
                #    self._reveal(gate)
                self._input(gate)
            elif gate_type == "OUTPUT":
                self._output(gate)
                if self._is_run_finished():
                    self.q.put("FIN")
            else:
                raise(Exception('{} is not a valid gate type'.format(gate_type)))
        

    def initialize_state(self, inputs):
        self.load_inputs(inputs)

    def load_inputs(self, inputs):
        #if 'in2' in inputs:
        #    print("pid: " + str(self.party_index) + " input: " + str(inputs['in2'][0].get_x()) + ", " + str(inputs['in2'][0].get_a()))

        for ing in inputs:
            #print("loading gate: " + str(ing))
            #if ing == "in2":
                #print("before: " + str(self.input_gates[ing].inputs))
                #print("inputs[ing]: " + str(self.input_shares[-2]))
            self.input_gates[ing].add_input("",inputs[ing])
            #if ing == "in2":
                #print("after: " + str(self.input_gates[ing].inputs))
            if self.input_gates[ing].is_ready():
                self.q.put(self.input_gates[ing])
    
    def load_secure_inputs(self,inputs):
        for ing_key in inputs:
            #print("key: " + str(ing_key) + " val: " + str(self.input_shares[inputs[ing_key]]))
            inputs[ing_key] = self.input_shares[inputs[ing_key]]

        self.load_inputs(inputs)

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

    def _add(self, gate):
        gid = gate.get_id()

        [x,y] = gate.get_inputs()

        gate_output = self.circuit[gid]

        if type(x) == list:
            z_vals = []

            for i in range(len(x)):
                z_vals.append(x[i] + y[i])

            for gout in gate_output:
                gout.add_input(gid, z_vals)
                if gout.is_ready():
                    self.q.put(gout)
            #self.wire_dict[z] = z_vals

        else:
            #self.wire_dict[z] = self.wire_dict[x] + self.wire_dict[y]
            for gout in gate_output:
                gout.add_input(gid, x + y)
                if gout.is_ready():
                    self.q.put(gout)

    def _mult(self, gate):
        gid = gate.get_id()

        [x,y] = gate.get_inputs()
        gate_output = self.circuit[gid]

        rindex = self.random_index
        cur_random_val = self.randomness[rindex]

        #x_val = self.wire_dict[x]
        #y_val = self.wire_dict[y]

        #r = x_val.pre_mult(y_val, cur_random_val)
        r = x.pre_mult(y, cur_random_val)

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

        for gout in gate_output:
            #self.wire_dict[z] = Share(new_r - r, -2 * new_r - r)
            gout.add_input(gid,Share(new_r - r, -2 * new_r - r))
            if gout.is_ready():
                self.q.put(gout)
        self.random_index += 1

    def _smult(self, gate):
        gid = gate.get_id()

        #x,y] = wire_in
        #[z] = wire_out

        #x_val = self.wire_dict[x]
        #yvec = self.wire_dict[y]

        [x_val, yvec] = gate.get_inputs()
        gate_output = self.circuit[gid]

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

        #self.wire_dict[z] = z_vals
        for gout in gate_output:
            gout.add_input(gid, z_vals)
            if gout.is_ready():
                self.q.put(gout)

    def _dot(self, gate):
        gid = gate.get_id()

        #[x,y] = wire_in
        #[z] = wire_out

        #xvec = self.wire_dict[x]
        #yvec = self.wire_dict[y]

        [xvec, yvec] = gate.get_inputs()
        gate_output = self.circuit[gid]

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

        #self.wire_dict[z] = z_val
        for gout in gate_output:
            gout.add_input(gid, z_val)
            if gout.is_ready():
                self.q.put(gout)

    def _not(self, gate):
        gid = gate.get_id()

        #[x] = wire_in
        #[z] = wire_out

        [x] = gate.get_inputs()
        gate_output = self.circuit[gid]

        #self.wire_dict[z] = self.wire_dict[x].not_op()
        for gout in gate_output:
            gout.add_input(gid,x.not_op())
            if gout.is_ready():
                self.q.put(gout)

    def _reveal(self, gate):
        gid = gate.get_id()

        rindex = self.random_index
        cur_random_val = self.randomness[rindex]

        num_inputs = len(gate.get_inputs())

        if num_inputs == 2:
            [x,y] = gate.get_inputs()
            self.oracle.send_op([x,y],self.party_index,rindex,"REVEAL")
        else:
            [x] = gate.get_inputs()
            self.oracle.send_op([x],self.party_index,rindex,"REVEAL")

        out_val = self.oracle.receive_op(self.party_index,rindex)
        while out_val == "wait":
            out_val = self.oracle.receive_op(self.party_index,rindex)

    def _comp(self, gate):
        gid = gate.get_id()

        #[x] = wire_in
        #[z] = wire_out

        [xa] = gate.get_inputs()
        gate_output = self.circuit[gid]

        rindex = self.random_index
        cur_random_val = self.randomness[rindex]

        #(x_val,a_val) = self.wire_dict[x]
        #xa = self.wire_dict[x]

        #self.oracle.send_comp([xa],self.party_index,rindex)
        self.oracle.send_op([xa],self.party_index,rindex,"COMP")

        #out_val = self.oracle.receive_comp(self.party_index,rindex)
        out_val = self.oracle.receive_op(self.party_index,rindex)
        while out_val == "wait":
            #out_val = self.oracle.receive_comp(self.party_index,rindex)
            out_val = self.oracle.receive_op(self.party_index,rindex)

        #self.wire_dict[z] = out_val
        for gout in gate_output:
            gout.add_input(gid, out_val)
            if gout.is_ready():
                self.q.put(gout)

        self.random_index += 1

    def _cmult(self, gate):
        gid = gate.get_id()
        [x] = gate.get_inputs()
        [const] = gate.get_const_inputs()
        gate_output = self.circuit[gid]

        if type(x) == list:
            out_val = []

            for i in range(len(x)):
                out_val.append(x[i].const_mult(const))
        
        else:
            out_val = x.const_mult(const)

        for gout in gate_output:
            gout.add_input(gid, out_val)
            if gout.is_ready():
                self.q.put(gout)

    def _cadd(self, gate):
        gid = gate.get_id()
        [x] = gate.get_inputs()
        [const] = gate.get_const_inputs()
        gate_output = self.circuit[gid]

        if type(x) == list:
            out_val = []
            for i in range(len(x)):
                out_val.append(x[i].const_add(const))
        else:
            out_val = x.const_add(const)

        for gout in gate_output:
            gout.add_input(gid, out_val)
            if gout.is_ready():
                self.q.put(gout)

    def _input(self, gate):
        gid = gate.get_id()
        [x] = gate.get_inputs()
        #print("INPUT VAL x: " + str(x))
        gate_output = self.circuit[gid]
        for gout in gate_output:
            gout.add_input(gid, x)
            #print("IS " + str(gout.get_id()) + " RDY?: " + str(gout.inputs))
            if gout.is_ready():
                self.q.put(gout)

    def _output(self, gate):
        gid = gate.get_id()
        [x] = gate.get_inputs()
        self.outputs[gid] = x

    def _is_run_finished(self):
        finished = True
        for out in self.outputs:
            if self.outputs[out] == "":
                finished = False
        return finished

    def _send_share(self, value, party_index, random_index):
        receiver = self.parties[party_index]
        receiver._receive_party_share(value,random_index)

    def _receive_party_share(self, share, random_index):
        self.interaction[random_index] = share

    def get_outputs(self):
        outs = []
        for out in self.outputs:
            #outs.append(self.wire_dict[out])
            outs.append(self.outputs[out])
        return outs

    def get_wire_dict(self):
        return self.wire_dict
       
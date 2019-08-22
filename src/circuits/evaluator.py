import numpy as np
from src.circuits.share import Share
from src.util.mod import mod_inverse
from queue import Queue
import time
import asyncio
from threading import Event
import math

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

    def __init__(self,circuit,input_gates,output_gates,party_index,oracle,mod,fp_precision=16):
        #Evaluator.__init__(self,circuit,[],fp_precision=fp_precision)
        self.circuit = circuit
        self.fpp = fp_precision
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
        
        self.mod = mod
        self.mod_bit_size = len(bin(self.mod)[2:])

        self.interaction_listener = None
        self.oracle_listener = None

        self.truncate_randomness = []
        self.trunc_index = 0

    def get_truncate_randomness(self, index, rand_type):
        return self.truncate_randomness[index][rand_type]

    def run(self, verbose=False):
        #if self.q.empty():
        #    raise(Exception('Queue empty, no inputs added'))

        i = 0
        
        gate = self.q.get()
        while gate != "FIN":
            self._eval_gate(gate, verbose=verbose)
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

            if verbose:
                print("gate type: " + str(gate_type))
                print("gate id: " + str(gate.get_id()))
                print("party index: " + str(self.party_index))
                ins = gate.get_inputs()
                ext = []
                for element in ins:
                    if type(element) == int:
                        ext.append(element)
                    elif type(element) == list:
                        el_list = []
                        for el in element:
                            x_val = el.get_x()
                            a_val = el.get_a()
                            el_list.append((x_val,a_val))
                        ext.append(el_list)
                    else:
                        x_val = element.get_x()
                        a_val = element.get_a()
                        ext.append((x_val,a_val))
                print("gate inputs: " + str(ext))

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
            elif gate_type == "ROUND":
                self._round(gate)
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

    def receive_truncate_randomness(self, trunc_share_dict):
        self.truncate_randomness.append(trunc_share_dict)

    def _truncate(self, value, k, m):
        if self.trunc_index < 10:
            print("TRUNC INDEX: " + str(self.trunc_index))
            IN_VAL = self._reveal(value)
            if self.party_index == 1:
                print(str(self.trunc_index) + " in val " + str(IN_VAL))
        a_prime = self._mod2m(value, k, m)
        if self.trunc_index < 10:
            APR_VAL = self._reveal(a_prime)
            if self.party_index == 1:
                print(str(self.trunc_index) + " a prime " + str(APR_VAL))
            
        self.trunc_index += 1
        d = value + a_prime.const_mult(-1,scaled=False)

        #d = d.const_mult(mod_inverse(2**m,self.mod),scaled=False)
        d = d.const_mult(mod_inverse(10**7,self.mod),scaled=False)

        #d = d.const_mult(2**m,scaled=False)
        d = d.const_mult(10**7,scaled=False)

        if self.trunc_index < 10:
            D_VAL = self._reveal(d)
            if self.party_index == 1:
                print(str(self.trunc_index) + " d_val " + str(D_VAL))
        return d

    def _mod2m(self, value, k, m):
        r2_r1_shares = self.get_truncate_randomness(self.trunc_index,"mod2m")
        #r2_r1_shares = self.truncate_randomness[self.trunc_index]["mod2m"]
        r2 = r2_r1_shares[0]
        r1 = r2_r1_shares[1]
        r1_bits = r2_r1_shares[2:]
        #m = len(r1_bits)
        if self.trunc_index < 10:
            R2_VAL = self._reveal(r2) % self.mod
            R1_VAL = self._reveal(r1) % self.mod
            if self.party_index == 1:
                print(str(self.trunc_index) + " r2: " + str(R2_VAL))
                print(str(self.trunc_index) + " r1: " + str(R1_VAL))
        #pre_c = value.const_add(2**(k-1))
        pre_c = value.const_add(self.mod)
        #if self.trunc_index < 10:
        #    prec1 = self._reveal(pre_c)
        #    if self.party_index == 1:
        #        print(str(self.trunc_index) + " prec1: " + str(prec1))
        
        #pre_c += r2.const_mult(2**m,scaled=False)
        pre_c += r2.const_mult(10**7,scaled=False)
        
        pre_c += r1
        c = self._reveal(pre_c)

        #c_prime = int(c % 2**m)
        c_prime = int(c % 10**7)

        if self.trunc_index < 10:
            if self.party_index == 1:
                print(str(self.trunc_index) + " c': " + str(c_prime))
        u = self._bit_lt(c_prime, r1_bits)
        
        if self.trunc_index < 10:
            U_VAL = self._reveal(u)
            if self.party_index == 1:
                print(str(self.trunc_index) + " u: " + str(U_VAL))
        
        a_prime = r1.const_mult(-1,scaled=False).const_add(c_prime)

        #a_prime += u.const_mult(2**m)
        a_prime += u.const_mult(10**7)

        return a_prime

    def _bit_lt(self, a, b_bits):
        d_vals = []
        a_bits = []
        for bit in bin(a)[2:]:
            a_bits.append(int(bit)*self.scale)
        a_bits = [0]*(len(b_bits) - len(a_bits)) + a_bits

        #if self.party_index == 1:
        #    print(str(self.trunc_index) + " a bits: " + str(a_bits))
        
        #B_BITS = []
        #for bv in b_bits:
        #    B_BITS.append(self._reveal(bv))
        
        #if self.party_index == 1:
        #    print(str(self.trunc_index) + " b bits: " + str(B_BITS))

        for i in range(len(a_bits)):
            d_val = b_bits[i].const_add(a_bits[i])
            d_val += b_bits[i].const_mult(-2*a_bits[i])
            d_vals.append(d_val.const_add(1*self.scale))

        #D_VALS = []
        #for dv in d_vals:
        #    D_VALS.append(self._reveal(dv))

        #if self.party_index == 1:
        #    print(str(self.trunc_index) + " d _vals: " + str(D_VALS))

        p_vals = self._premul(d_vals)
        p_vals.reverse()

        #P_VALS = []
        #for pv in p_vals:
        #    P_VALS.append(self._reveal(pv))

        #if self.party_index == 1:
        #    print(str(self.trunc_index) + " p_vals: " + str(P_VALS))

        s_vals = []
        for i in range(len(p_vals)-1):
            s_val = p_vals[i] + p_vals[i+1].const_mult(-1,scaled=False)
            s_vals.append(s_val)
        s_vals.append(p_vals[-1].const_add(-1,scaled=False))

        #S_VALS = []
        #for sv in s_vals:
        #    S_VALS.append(self._reveal(sv))

        #if self.party_index == 1:
        #    print(str(self.trunc_index) + " s_vals: " + str(S_VALS))

        a_bits.reverse()
        #print("real a_bits: " + str(a_bits))
        s = Share(0,0,mod=self.mod,fp_prec=self.fpp)
        slen = len(s_vals)
        for i in range(slen):
            s += s_vals[i].const_mult(self.scale - a_bits[i])

        #SV = self._reveal(s)

        #if self.party_index == 1:
        #    print(str(self.trunc_index) + " s val: " + str(SV))

        ret_val = self._mod2(s,len(b_bits))
        
        #RET = self._reveal(ret_val)
        #if self.party_index == 1:
        #    print(str(self.trunc_index) + " return value (of bitlt): " + str(RET))

        return ret_val

    def _mod2(self, value, k):
        value = value.switch_precision(0)
        bits = self.get_truncate_randomness(self.trunc_index,"mod2")
        for i,bit in enumerate(bits):
            bits[i] = bit.switch_precision(0)
        #c_pre = value.const_add(2**(k-1))
        c_pre = value
        c_pre += bits[0].const_mult(2) + bits[2]
        c = self._reveal(c_pre)
        c0 = int(bin(math.floor(c))[-1])
        a = bits[2].const_add(c0)
        a += bits[2].const_mult(-2*c0)
        return a.switch_precision(self.fpp)

    def _premul(self, a_vals):
        premul_rand = self.get_truncate_randomness(self.trunc_index,"premul")
        r_vals = premul_rand['r']
        s_vals = premul_rand['s']
        u_vals = []

        mod_scale = mod_inverse(self.scale,self.mod)

        for i in range(len(r_vals)):
            r_val = self._reveal(r_vals[i])
            s_val = self._reveal(s_vals[i])
            u_val = (r_val * s_val * mod_scale) % self.mod
            u_vals.append(u_val)

        u_inv_vals = []
        for i in range(len(u_vals)):
            u_val = u_vals[i] * mod_scale % self.mod
            u_inv_vals.append(mod_inverse(u_val,self.mod) * self.scale)

        v_vals = []
        for i in range(len(r_vals)-1):
            v_vals.append(self._multiply(r_vals[i+1],s_vals[i]))
        
        w_vals = []
        w_vals.append(r_vals[0])
        for i in range(len(v_vals)):
            w_val = v_vals[i].const_mult(u_inv_vals[i])
            w_vals.append(w_val)
        
        z_vals = []
        for i in range(len(s_vals)):
            z_val = s_vals[i].const_mult(u_inv_vals[i])
            z_vals.append(z_val)

        m_vals = []
        for i in range(len(w_vals)):
            m_val = self._reveal(w_vals[i]) * self._reveal(a_vals[i]) * mod_scale
            m_vals.append(m_val % self.mod)
             
        p_vals = []
        p_vals.append(a_vals[0])
        for i in range(1,len(z_vals)):
            m_prod = 1 * self.scale
            for j in range(i+1):
                m_prod *= m_vals[j] 
                m_prod *= mod_scale
            p_vals.append(z_vals[i].const_mult(m_prod))
        
        return p_vals

    def _reveal(self, value):

        other_party_value = self._interact(value)
        if self.party_index == 1:
            return value.unshare(other_party_value,indices=[1,3],neg_representation=False)
        elif self.party_index == 2:
            return value.unshare(other_party_value,indices=[2,1],neg_representation=False)
        elif self.party_index == 3:
            return value.unshare(other_party_value,indices=[3,2],neg_representation=False)

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

    def _interact_oracle(self):
        pass

    def _interact(self, r_val):
        # if self.interaction[rand_index] doesnt equal "wait"
        # this means that another party already send us a value
        # so we do not need an async event listener, so it will
        # be set to None
        if self.interaction[self.random_index] == "wait":
            self.interaction_listener = Event()
        else:
            self.interaction_listener = None

        # each party sends value to other party
        if self.party_index == 1:
            self._send_share(r_val,2,self.random_index)
        elif self.party_index == 2:
            self._send_share(r_val,3,self.random_index)
        elif self.party_index == 3:
            self._send_share(r_val,1,self.random_index)

        # if we don't have event listener, we don't have to wait
        # because we already received share from party
        if self.interaction_listener != None:
            self.interaction_listener.wait()
            new_r = self.interaction[self.random_index]
        else:
        # pull new value from self.interaction list
            new_r = self.interaction[self.random_index]
        self.random_index += 1
        self.interaction_listener = None
        return new_r

    def _multiply(self, share1, share2):

        rand_value = self.randomness[self.random_index]
        r = share1.pre_mult(share2, rand_value)
        new_r = self._interact(r)

        return Share(new_r - r, -2* new_r - r, mod=self.mod, fp_prec=self.fpp)

    def _mult(self, gate):
        gid = gate.get_id()

        [x,y] = gate.get_inputs()
        gate_output = self.circuit[gid]

        z_val = self._multiply(x,y)

        for gout in gate_output:
            gout.add_input(gid,z_val)
            if gout.is_ready():
                self.q.put(gout)

    def _smult(self, gate):
        gid = gate.get_id()

        [x_val, yvec] = gate.get_inputs()
        gate_output = self.circuit[gid]

        z_vals = []

        for i in range(len(yvec)):
            y_val = yvec[i]
            z_val = self._multiply(x_val,y_val)

            z_vals.append(z_val)

        for gout in gate_output:
            gout.add_input(gid, z_vals)
            if gout.is_ready():
                self.q.put(gout)

    def _dot(self, gate):
        gid = gate.get_id()

        [xvec, yvec] = gate.get_inputs()
        gate_output = self.circuit[gid]

        z_val = Share(0,0,mod=self.mod,fp_prec=self.fpp)

        for i in range(len(xvec)):
            x_val = xvec[i]
            y_val = yvec[i]

            z_val += self._multiply(x_val,y_val)

            self.random_index += 1

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

    def _round(self, gate):
        gid = gate.get_id()

        [xa] = gate.get_inputs()
        gate_output = self.circuit[gid]

        """
        rindex = self.random_index
        cur_random_val = self.randomness[rindex]

        #self.oracle.send_comp([xa],self.party_index,rindex)
        self.oracle.send_op([xa],self.party_index,rindex,"ROUND")

        #out_val = self.oracle.receive_comp(self.party_index,rindex)
        out_val = self.oracle.receive_op(self.party_index,rindex)
        while out_val == "wait":
            #out_val = self.oracle.receive_comp(self.party_index,rindex)
            out_val = self.oracle.receive_op(self.party_index,rindex)
        """
        #k = int((self.mod_bit_size - 1) / 3)
        #m = int((self.mod_bit_size - 1) / 3)
        k = 20
        m = 20

        #if self.party_index == 1:
        #    print("======")
        #for xa_val in xa:
        #    XA_VAL = self._reveal(xa_val)
        #    if self.party_index == 1:
        #        print(str(self.party_index) + " trunc input: " + str(XA_VAL))

        if type(xa) is list:
            out_val = []
            for share in xa:
                cur = self._truncate(share, k, m)
                #CUR_VAL = self._reveal(cur)
                #if self.party_index == 1:
                #    print(str(self.party_index) + " trunc out: " + str(CUR_VAL))
                out_val.append(cur)
                #out_val.append(self._truncate(share, k, m))
        else:
            out_val = self._truncate(xa, k, m)
            #OV = self._reveal(out_val)
            #if self.party_index == 1:
            #        print(str(self.party_index) + " trunc out: " + str(OV))
                
        #if self.party_index == 1:
            #print("=======")
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
        if self.interaction_listener != None:
            self.interaction_listener.set()
        self.interaction[random_index] = share

    def get_outputs(self):
        outs = []
        for out in self.outputs:
            #outs.append(self.wire_dict[out])
            outs.append(self.outputs[out])
        return outs

    def get_wire_dict(self):
        return self.wire_dict
       
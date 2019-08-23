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
        i = 0
        
        gate = self.q.get()
        while gate != "FIN":
            self._eval_gate(gate, verbose=verbose)
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
                self._add(gate)
            elif gate_type == "MULT":
                self._mult(gate)
            elif gate_type == "SMULT":
                self._smult(gate)
            elif gate_type == "DOT":
                self._dot(gate)
            elif gate_type == "NOT":
                self._not(gate)
            elif gate_type == "COMP":
                self._comp(gate)
            elif gate_type == "ROUND":
                self._round(gate)
            elif gate_type == "CMULT":
                self._cmult(gate)
            elif gate_type == "CADD":
                self._cadd(gate)
            elif gate_type == "INPUT":
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
        for ing in inputs:
            self.input_gates[ing].add_input("",inputs[ing])
            if self.input_gates[ing].is_ready():
                self.q.put(self.input_gates[ing])
    
    def load_secure_inputs(self,inputs):
        for ing_key in inputs:
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

    def _truncate(self, value, k, m, pow2_switch=False):

        if pow2_switch:
            m_val = 2**m
        else:
            m_val = m

        a_prime = self._mod2m(value, k, m, pow2_switch=pow2_switch)   

        self.trunc_index += 1
        d = value + a_prime.const_mult(-1,scaled=False)
        d = d.const_mult(mod_inverse(m_val,self.mod),scaled=False)

        return d

    def _mod2m(self, value, k, m, pow2_switch=False):

        if pow2_switch:
            m_val = 2**m
        else:
            m_val = m

        r2_r1_shares = self.get_truncate_randomness(self.trunc_index,"mod2m")
        r2 = r2_r1_shares[0]
        r1 = r2_r1_shares[1]
        r1_bits = r2_r1_shares[2:]
        
        pre_c = value.const_add(self.mod)
        pre_c += r2.const_mult(m_val,scaled=False)
        pre_c += r1
        c = self._reveal(pre_c)

        c_prime = int(c % m_val)

        u = self._bit_lt(c_prime, r1_bits)

        a_prime = r1.const_mult(-1,scaled=False).const_add(c_prime)
        a_prime += u.const_mult(m_val)

        return a_prime

    def _bit_lt(self, a, b_bits):
        d_vals = []
        a_bits = []
        for bit in bin(a)[2:]:
            a_bits.append(int(bit)*self.scale)
        a_bits = [0]*(len(b_bits) - len(a_bits)) + a_bits

        for i in range(len(a_bits)):
            d_val = b_bits[i].const_add(a_bits[i])
            d_val += b_bits[i].const_mult(-2*a_bits[i])
            d_vals.append(d_val.const_add(1*self.scale))

        p_vals = self._premul(d_vals)
        p_vals.reverse()

        s_vals = []
        for i in range(len(p_vals)-1):
            s_val = p_vals[i] + p_vals[i+1].const_mult(-1,scaled=False)
            s_vals.append(s_val)
        s_vals.append(p_vals[-1].const_add(-1,scaled=False))

        a_bits.reverse()

        s = Share(0,0,mod=self.mod,fp_prec=self.fpp)
        slen = len(s_vals)
        for i in range(slen):
            s += s_vals[i].const_mult(self.scale - a_bits[i])

        ret_val = self._mod2(s,len(b_bits))
        
        return ret_val

    def _mod2(self, value, k):
        value = value.switch_precision(0)
        bits = self.get_truncate_randomness(self.trunc_index,"mod2")
        for i,bit in enumerate(bits):
            bits[i] = bit.switch_precision(0)
            
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

        else:
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

        [x] = gate.get_inputs()
        gate_output = self.circuit[gid]

        for gout in gate_output:
            gout.add_input(gid,x.not_op())
            if gout.is_ready():
                self.q.put(gout)

    def _comp(self, gate):
        gid = gate.get_id()

        [xa] = gate.get_inputs()
        gate_output = self.circuit[gid]

        half_mod = self.mod / 2

        # need to do truncate in pieces in order to work
        # take square root and perform truncate twice
        sq_half_mod = math.floor(half_mod**(1/2))

        s_val1 = self._truncate(xa, sq_half_mod, sq_half_mod)
        s_val2 = self._truncate(s_val1, sq_half_mod, sq_half_mod)

        # need to invert truncation to get comparison value
        out_val = s_val2.const_mult(-1,scaled=False)

        # need to scale value back up to fixed-point precision
        out_val = out_val.const_mult(self.scale, scaled=False)

        for gout in gate_output:
            gout.add_input(gid, out_val)
            if gout.is_ready():
                self.q.put(gout)

        self.random_index += 1

    def _round(self, gate):
        gid = gate.get_id()

        [xa] = gate.get_inputs()
        gate_output = self.circuit[gid]

        k = 10**7
        m = 10**7

        if type(xa) is list:
            out_val = []
            for share in xa:
                cur = self._truncate(share, k, m)
                cur = cur.const_mult(m,scaled=False)
                
                out_val.append(cur)
        else:
            out_val = self._truncate(xa, k, m)
            out_val = out_val.const_mult(m,scaled=False)
            
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
        gate_output = self.circuit[gid]
        for gout in gate_output:
            gout.add_input(gid, x)
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
            outs.append(self.outputs[out])
        return outs

    def get_wire_dict(self):
        return self.wire_dict
       
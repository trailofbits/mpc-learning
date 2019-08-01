from random import randint
import numpy as np
from src.circuits.share import Share
import math

class Dealer():
    """
    Dealer class that is responsible for creating input shares for the
    SecureEvaluators.

    Methods
    -------
    __init__(self, parties, modulus, fp_precision=16)
        Dealer object constructor
        Initliaze parties, mod, modulus, and fp_precision

        Parameters
        ----------
        parties: iterable
            Iterable of Evaluators
        modulus: integer
            Modulus representing input domain
        (optional) fp_precision=16: int
            Fixed point number precision

    generate_randomness(self, number_of_values)
        Method that generates number_of_values random values for parties
        According to current protocol, each party receives a value x_i
        such that x_1 + x_2 + x_3 = 0

        Parameters
        ----------
        number_of_values: int
            Number of random shares to be generated

    distribute_shares(self, inputs, random=False, verbose=False)
        Method that generates shares of inputs and sends the shares
        to the three parties

        Parameters
        ----------
        inputs: iterable
            Iterable of input values to be converted to shares
        (optional) random=False: boolean
            Boolean indicating whether we are creating shares of inputs
            or shares for generate_randomness
        (optional) verbose=False: boolean
            Boolean to turn on verbose mode for debugging
        
    _make_shares(self, input_value)
        Method for generating shares from an input value (called within
        distribute_shares)

        Parameters
        ----------
        input_value: int or iterable
            Value to be made into shares and sent to parties
        
    _send_shares(self, shares, receiver)
        Method for sending shares to parties (called within distribute_shares)

        Parameters
        ----------
        shares: iterable
            Iterable of values/shares to be sent to parties
        receiver: Evaluator object
            Evaluator (party) that will receive shares

    _send_randomness(self, shares, receiver)
        Method for sending randomness to parties (called within 
        generate_randomness)

        Parameters
        ----------
        shares: iterable
            Iterable of values/shares to be sent to parties
        receiver: Evaluator object
            Evaluator (party) that will receive shares
            
    """
    def __init__(self,parties,modulus,fp_precision=16):
        self.parties = parties
        self.mod = modulus
        self.scale = 10**fp_precision
        #self.modulus = modulus / self.scale

    def generate_randomness(self,number_of_values):
        inputs = []
        for i in range(number_of_values):
            inputs.append(0)

        self.distribute_shares(inputs,random=True)

    def distribute_shares(self,inputs, random=False, verbose=False):
        # generate and send shares to each party
        # here we assume there are exactly three parties
        shares_for_1 = []
        shares_for_2 = []
        shares_for_3 = []
        
        if (type(inputs) == int) or (type(inputs) == float):
            (sh1,sh2,sh3) = self._make_shares(inputs, random=random)
            shares_for_1.append(sh1)
            shares_for_2.append(sh2)
            shares_for_3.append(sh3)

        else:
            for val in inputs:
                if verbose:
                    print("val: " + str(val))
                if val == []:
                    continue
                (sh1,sh2,sh3) = self._make_shares(val, random=random)
                shares_for_1.append(sh1)
                shares_for_2.append(sh2)
                shares_for_3.append(sh3)

        shares = [shares_for_1,shares_for_2,shares_for_3]
        
        for i,party in enumerate(self.parties):
            if random:
                self._send_randomness(shares[i],party)
            else:
                self._send_shares(shares[i],party)

    def _make_shares(self,input_value, random=False):
        it = type(input_value)
        if (it == int) or (it == np.int64) or (it == np.float64) or (it == float):
            
            val = int(input_value*self.scale) % self.mod
            
            # first generate three random values a, b, c s.t. a + b + c = 0
            #a = int(randint(0,self.mod-1) )
            #b = int(randint(0,self.mod-1) )
            #c = (- (a + b)) % self.mod

            a = int(randint(0,math.floor(self.mod / self.scale) - 1) * self.scale)
            b = int(randint(0,math.floor(self.mod / self.scale) - 1) * self.scale)
            c = (- (a + b)) % self.mod

            if random:
                share1 = a
                share2 = b
                share3 = c
            else:
                #share1 = (a,c-val)
                #share2 = (b,a-val)
                #share3 = (c,b-val)
                share1 = Share(a,c-val,mod=self.mod)
                share2 = Share(b,a-val,mod=self.mod)
                share3 = Share(c,b-val,mod=self.mod)

        else:
            share1 = []
            share2 = []
            share3 = []
            
            for val in input_value:
                mod_val = int(round(val*self.scale)) % self.mod
                
                # first generate three random values a, b, c s.t. a + b + c = 0
                #a = int(randint(0,self.mod-1))
                #b = int(randint(0,self.mod-1))
                #c = (- (a + b)) % self.mod

                a = int(randint(0,math.floor(self.mod / self.scale) - 1) * self.scale)
                b = int(randint(0,math.floor(self.mod / self.scale) - 1) * self.scale)
                c = (- (a + b)) % self.mod
                
                if random:
                    share1.append(a)
                    share2.append(b)
                    share3.append(c)
                else:
                    share1.append(Share(a,c-mod_val,mod=self.mod))
                    share2.append(Share(b,a-mod_val,mod=self.mod))
                    share3.append(Share(c,b-mod_val,mod=self.mod))
            
        return (share1,share2,share3)

    def _send_shares(self,shares,receiver):
        receiver.receive_shares(shares)

    def _send_randomness(self,shares,receiver):
        receiver.receive_randomness(shares)

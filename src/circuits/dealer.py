import random
import numpy as np

class Dealer():
    def __init__(self,parties,modulus,fp_precision=16):
        self.parties = parties
        self.mod = modulus
        self.scale = 10**fp_precision
        self.modulus = modulus / self.scale

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
            (sh1,sh2,sh3) = self._make_shares(inputs)
            shares_for_1.append(sh1)
            shares_for_2.append(sh2)
            shares_for_3.append(sh3)

        else:
            for val in inputs:
                if verbose:
                    print("val: " + str(val))
                if val == []:
                    continue
                (sh1,sh2,sh3) = self._make_shares(val)
                shares_for_1.append(sh1)
                shares_for_2.append(sh2)
                shares_for_3.append(sh3)

        shares = [shares_for_1,shares_for_2,shares_for_3]
        
        for i,party in enumerate(self.parties):
            if random:
                self._send_randomness(shares[i],party)
            else:
                self._send_shares(shares[i],party)

        
    
    def _make_shares(self,input_value):
        it = type(input_value)
        if (it == int) or (it == np.int64) or (it == np.float64) or (it == float):
            
            
            val = int(input_value*self.scale)
            
            # first generate three random values a, b, c s.t. a + b + c = 0
            a = int(random.randint(0,self.modulus-1) * self.scale)
            b = int(random.randint(0,self.modulus-1) * self.scale)
            c = - (a + b)

            share1 = (a,c-val)
            share2 = (b,a-val)
            share3 = (c,b-val)

            

        else:
            share1 = []
            share2 = []
            share3 = []

            

            for val in input_value:
                val = int(val*self.scale)
                
                # first generate three random values a, b, c s.t. a + b + c = 0
                a = int(random.randint(0,self.modulus-1) * self.scale)
                b = int(random.randint(0,self.modulus-1) * self.scale)
                c = - (a + b)
                
                share1.append((a,c-val))
                share2.append((b,a-val))
                share3.append((c,b-val))

        return (share1,share2,share3)

    def _send_shares(self,shares,receiver):
        receiver.receive_shares(shares)

    def _send_randomness(self,shares,receiver):
        receiver.receive_randomness(shares)

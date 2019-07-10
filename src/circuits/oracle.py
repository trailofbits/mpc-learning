import random
from src.circuits.dealer import Dealer

class Oracle(Dealer):
    def __init__(self,modulus,fp_precision=16):
        Dealer.__init__(self,[],modulus,fp_precision)
        self.shares = {}
        self.outputs = {}
    
    def add_parties(self,parties):
        self.parties = parties

    def send_mult(self,values,pindex,rindex):
        if rindex not in self.shares:
            self.shares[rindex] = {pindex: values}
        else:
            self.shares[rindex][pindex] = values

        self._mult(rindex)

    def _mult(self,rindex):
        shrs = self.shares[rindex]
        if (1 not in shrs) or (2 not in shrs) or (3 not in shrs):
            self.outputs[rindex] = "wait"
        else:
            x1 = shrs[1][0][0]
            y1 = shrs[1][1][0]
            a2 = shrs[2][0][1]
            b2 = shrs[2][1][1]

            x_val = x1 - a2
            y_val = y1 - b2

            #print("x_val: " + str(x_val))
            #print("y_val: " + str(y_val))

            z = (x_val / self.scale) * (y_val / self.scale)
            #print("mult z: " + str(z))
            [sh1,sh2,sh3] = self._make_shares(z)
            #print("sh1: " + str(sh1))
            #print("sh2: " + str(sh2))
            #print("sh3: " + str(sh3))
            #print("mult z shares: " + str([sh1,sh2,sh3]))
            self.outputs[rindex] = {1: sh1, 2: sh2, 3: sh3}

    
    def receive_mult(self,pindex,rindex):
        if rindex not in self.outputs:
            return "wait"
        if self.outputs[rindex] == "wait":
            return "wait"
        else:
            #print("about to return: " + str(self.outputs[rindex][pindex]))
            return self.outputs[rindex][pindex]

    def send_dot(self,values,pindex,rindex):
        if rindex not in self.shares:
            self.shares[rindex] = {pindex: values}
        else:
            self.shares[rindex][pindex] = values
        self._dot(rindex)

    def _dot(self,rindex):
        shrs = self.shares[rindex]
        if (1 not in shrs) or (2 not in shrs) or (3 not in shrs):
            self.outputs[rindex] = "wait"
        else:
            z = 0
            
            [xvec1,yvec1] = shrs[1]
            [xvec2,yvec2] = shrs[2]

            for i in range(len(xvec1)):
                (x1,a1) = xvec1[i]
                (y1,b1) = yvec1[i]
                (x2,a2) = xvec2[i]
                (y2,b2) = yvec2[i]

                x_val = x1 - a2
                y_val = y1 - b2
                z += (x_val / self.scale) * y_val

            [sh1,sh2,sh3] = self._make_shares(z / self.scale)
            self.outputs[rindex] = {1: sh1, 2: sh2, 3: sh3}

    def receive_dot(self,pindex,rindex):
        if rindex not in self.outputs:
            return "wait"
        if self.outputs[rindex] == "wait":
            return "wait"
        else:
            return self.outputs[rindex][pindex]

    def send_comp(self,values,pindex,rindex):
        if rindex not in self.shares:
            self.shares[rindex] = {pindex: values}
        else:
            self.shares[rindex][pindex] = values

        self._comp(rindex)

    def _comp(self,rindex):
        shrs = self.shares[rindex]
        if (1 not in shrs) or (2 not in shrs) or (3 not in shrs):
            self.outputs[rindex] = "wait"
        else:
            z = 0
 
            (x1,a1) = shrs[1][0]
            (x2,a2) = shrs[2][0]

            x_val = x1 - a2
            z = int(x_val <= 0)

            [sh1,sh2,sh3] = self._make_shares(z)
            self.outputs[rindex] = {1: sh1, 2: sh2, 3: sh3}

    def receive_comp(self,pindex,rindex):
        if rindex not in self.outputs:
            return "wait"
        if self.outputs[rindex] == "wait":
            return "wait"
        else:
            return self.outputs[rindex][pindex]

    def send_smult(self,values,pindex,rindex):
        if rindex not in self.shares:
            self.shares[rindex] = {pindex: values}
        else:
            self.shares[rindex][pindex] = values

        self._smult(rindex)

    def _smult(self,rindex):
        shrs = self.shares[rindex]
        if (1 not in shrs) or (2 not in shrs) or (3 not in shrs):
            self.outputs[rindex] = "wait"
        else:
            z = 0
 
            (x1,a1) = shrs[1][0]
            (x2,a2) = shrs[2][0]

            yvec1 = shrs[1][1]
            yvec2 = shrs[2][1]

            x_val = x1 - a2

            z = []

            for i in range(len(yvec1)):
                (y1,b1) = yvec1[i]
                (y2,b2) = yvec2[i]

                y_val = y1 - b2

                z.append((x_val / self.scale) * (y_val / self.scale))

            [sh1,sh2,sh3] = self._make_shares(z)
            self.outputs[rindex] = {1: sh1, 2: sh2, 3: sh3}

    def receive_smult(self,pindex,rindex):
        if rindex not in self.outputs:
            return "wait"
        if self.outputs[rindex] == "wait":
            return "wait"
        else:
            return self.outputs[rindex][pindex]
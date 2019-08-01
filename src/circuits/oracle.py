import random
from src.circuits.dealer import Dealer

class Oracle(Dealer):
    """
    Oracle class that inherits from the Dealer class (to create shares).
    The Oracle is responsible for receiving shares from the parties, performing
    an operation (either MULT, SMULT, DOT, or COMP), creating new shares for
    the results and sending them to the parties.

    Methods
    -------
    __init__(self, modulus, fp_precision=16)
        Constructor for Dealer object.
        Calls the Dealer constructor and initializes shares and outputs.

        Parameters
        ----------
        modulus: int
            Modulus representing input domain
        (optional) fp_precision=16: int
            Fixed point number precision

    send_op(self, values, pindex, rindex, op)
        Method for performing an operation for the parties

        Parameters
        ----------
        values: int or iterable
            Inputs to be used for performing operation
        pindex: int
            Index of party sending the value (must be 1, 2, or 3)
        rindex: int
            Index to keep track of which values to use
        op: string
            String indicating which operation to perform
            Must be "MULT", "DOT", "COMP", or "SMULT"

    receive_op(self, pindex, rindex)
        Method for distributing shares of result of operation to parties

        Parameters
        ----------
        pindex: int
            Index of party receiving the value
        rindex: int
            Index to keep track of which values to use

        Returns
        -------
        "wait"
            Value returned if all parties have not yet contributed shares
        self.outputs[rindex][pindex]
            Share of result of operation to be sent to party

    _mult(self, rindex)
        Method specifying how to compute multiplication

        Parameters
        ----------
        rindex: int
            Index to keep track of which values to use

    _dot(self, rindex)
        Method specifying how to compute dot product

        Parameters
        ----------
        rindex: int
            Index to keep track of which values to use
        
    _comp(self, rindex)
        Method specifying how to compute comparison (input <= 0)

        Parameters
        ----------
        rindex: int
            Index to keep track of which values to use

    _smult(self, rindex)
        Method specifying how to compute scalar multiplication

        Parameters
        ----------
        rindex: int
            Index to keep track of which values to use
                
    """
    def __init__(self,modulus,fp_precision=16):
        Dealer.__init__(self,[],modulus,fp_precision)
        self.shares = {}
        self.outputs = {}

    def send_op(self,values,pindex,rindex,op):
        if rindex not in self.shares:
            self.shares[rindex] = {pindex: values}
        else:
            self.shares[rindex][pindex] = values

        if op == "MULT":
            self._mult(rindex)
        elif op == "DOT":
            self._dot(rindex)
        elif op == "COMP":
            self._comp(rindex)
        elif op == "ROUND":
            self._round(rindex)
        elif op == "SMULT":
            self._smult(rindex)
        elif op == "REVEAL":
            self._reveal(rindex)

    def receive_op(self,pindex,rindex):
        if rindex not in self.outputs:
            return "wait"
        if self.outputs[rindex] == "wait":
            return "wait"
        else:
            return self.outputs[rindex][pindex]

    def _mult(self,rindex):
        shrs = self.shares[rindex]
        if (1 not in shrs) or (2 not in shrs) or (3 not in shrs):
            self.outputs[rindex] = "wait"
        else:
            #x1 = shrs[1][0][0]
            #y1 = shrs[1][1][0]
            #a2 = shrs[2][0][1]
            #b2 = shrs[2][1][1]

            #x_val = x1 - a2
            #y_val = y1 - b2

            [x_val1,y_val1] = shrs[1]
            [x_val2,y_val2] = shrs[2]

            x_val = x_val1.unshare(x_val2)
            y_val = y_val1.unshare(y_val2)

            z = (x_val / self.scale) * (y_val / self.scale)
            [sh1,sh2,sh3] = self._make_shares(z)
            self.outputs[rindex] = {1: sh1, 2: sh2, 3: sh3}

    def _dot(self,rindex):
        shrs = self.shares[rindex]
        if (1 not in shrs) or (2 not in shrs) or (3 not in shrs):
            self.outputs[rindex] = "wait"
        else:
            z = 0
            
            [xvec1,yvec1] = shrs[1]
            [xvec2,yvec2] = shrs[2]

            for i in range(len(xvec1)):
                #(x1,a1) = xvec1[i]
                #(y1,b1) = yvec1[i]
                #(x2,a2) = xvec2[i]
                #(y2,b2) = yvec2[i]

                #x_val = x1 - a2
                #y_val = y1 - b2

                x_val = xvec1[i].unshare(xvec2[i])
                y_val = yvec1[i].unshare(yvec2[i])

                z += (x_val / self.scale) * y_val

            [sh1,sh2,sh3] = self._make_shares(z / self.scale)
            self.outputs[rindex] = {1: sh1, 2: sh2, 3: sh3}

    def _comp(self,rindex):
        shrs = self.shares[rindex]
        if (1 not in shrs) or (2 not in shrs) or (3 not in shrs):
            self.outputs[rindex] = "wait"
        else:
            z = 0
 
            #(x1,a1) = shrs[1][0]
            #(x2,a2) = shrs[2][0]

            #x_val = x1 - a2

            [x_val1] = shrs[1]
            [x_val2] = shrs[2]

            x_val = x_val1.unshare(x_val2)

            z = int(x_val <= 0)

            [sh1,sh2,sh3] = self._make_shares(z)
            self.outputs[rindex] = {1: sh1, 2: sh2, 3: sh3}

    def _round(self,rindex):
        shrs = self.shares[rindex]
        if (1 not in shrs) or (2 not in shrs) or (3 not in shrs):
            self.outputs[rindex] = "wait"
        else:
            z = 0
 
            #(x1,a1) = shrs[1][0]
            #(x2,a2) = shrs[2][0]

            #x_val = x1 - a2

            [x_val1] = shrs[1]
            [x_val2] = shrs[2]

            if type(x_val1) == list:
                z = []
                for i in range(len(x_val1)):
                    cur = x_val1[i].unshare(x_val2[i])
                    z.append((round(cur / 10**7) * 10**7) / self.scale)
                print(z)
            else:
                x_val = x_val1.unshare(x_val2)
                print(x_val)
                z = (round(x_val / 10**7) * 10**7) / self.scale

            [sh1,sh2,sh3] = self._make_shares(z)
            
            self.outputs[rindex] = {1: sh1, 2: sh2, 3: sh3}

    def _smult(self,rindex):
        shrs = self.shares[rindex]
        if (1 not in shrs) or (2 not in shrs) or (3 not in shrs):
            self.outputs[rindex] = "wait"
        else:
            #z = 0
 
            #(x1,a1) = shrs[1][0]
            #(x2,a2) = shrs[2][0]

            #yvec1 = shrs[1][1]
            #yvec2 = shrs[2][1]

            #x_val = x1 - a2

            [x_val1, yvec1] = shrs[1]
            [x_val2, yvec2] = shrs[2]

            x_val = x_val1.unshare(x_val2)

            z = []

            for i in range(len(yvec1)):
                #(y1,b1) = yvec1[i]
                #(y2,b2) = yvec2[i]

                #y_val = y1 - b2

                y_val = yvec1[i].unshare(yvec2[i])

                z.append((x_val / self.scale) * (y_val / self.scale))

            [sh1,sh2,sh3] = self._make_shares(z)
            self.outputs[rindex] = {1: sh1, 2: sh2, 3: sh3}

    def _reveal(self, rindex):
        shrs = self.shares[rindex]
        if (1 not in shrs) or (2 not in shrs) or (3 not in shrs):
            self.outputs[rindex] = "wait"
        else:
            z = 0
 
            #(x1,a1) = shrs[1][0]
            #(x2,a2) = shrs[2][0]

            #x_val = x1 - a2

            [x_val1] = shrs[1]
            [x_val2] = shrs[2]

            if type(x_val1) == list:
                z_vals = []
                for i in range(len(x_val1)):
                    z_vals.append(x_val1[i].unshare(x_val2[i]))

                print("REVEALING z: " + str(z_vals))
            
            else:
                z = x_val1.unshare(x_val2)
                print("REVEALING z: " + str(z))
            self.outputs[rindex] = "done"

            
# I picked a large, 32-digit prime number to be used for default modulus: 
# 10001112223334445556667778889991
# 1/3 mod 10001112223334445556667778889991 is 3333704074444815185555926296664
# this will be needed for MPC protocol

MOD = 10001112223334445556667778889991
INVERSE_OF_3 = 3333704074444815185555926296664
MOD_SCALE = 1056332347261636068068068068067 # inverse of 10^16 mod MOD

class Share():
    def __init__(self, value1, value2, mod=MOD, inv_3=INVERSE_OF_3, fp_prec=16, mod_scale=MOD_SCALE):
        self.mod = mod
        self.inv_3 = inv_3
        self.fp = fp_prec
        self.x = value1 % self.mod
        self.a = value2 % self.mod
        self.scale = 10**self.fp
        self.mod_scale = mod_scale

    def __add__(self,other):
        new_x = (self.x + other.x) % self.mod
        new_a = (self.a + other.a) % self.mod
        return Share(new_x, new_a, mod=self.mod, inv_3=self.inv_3, fp_prec=self.fp)

    def not_op(self):
        new_x = (-self.x) % self.mod
        new_a = (-(self.a + 1*self.scale)) % self.mod
        return Share(new_x, new_a, mod=self.mod, inv_3=self.inv_3, fp_prec=self.fp)

    def pre_mult(self, other, random_val):
        r = ((self.a * self.mod_scale) * other.a)
        r -= ((self.x * self.mod_scale) * other.x)
        r += random_val
        r = r % self.mod
        return (r * self.inv_3) % self.mod

    def unshare(self, other, indices=[1,2]):
        ind = indices
        if (ind == [1,2]) or (ind == [2,3]) or (ind == [3,1]):
            res = (self.x - other.a) % self.mod
        else:
            res = (self.a - other.x) % self.mod
        if res > (self.mod / 2):
            res = res - self.mod
        return res
    
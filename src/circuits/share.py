from src.util.mod import mod_inverse

# I picked a large, 32-digit prime number to be used for default modulus: 
# 10001112223334445556667778889991
# 1/3 mod 10001112223334445556667778889991 is 3333704074444815185555926296664
# this will be needed for MPC protocol

#MOD = 10001112223334445556667778889991
#INVERSE_OF_3 = 3333704074444815185555926296664
#MOD_SCALE = 1056332347261636068068068068067 # inverse of 10^16 mod MOD

# I picked an even large, 60-digit prime for other use cases
MOD = 622288097498926496141095869268883999563096063592498055290461

#MOD = 24684249032065892333066123534168930441269525239006410135714283699648991959894332868446109170827166448301044689

class Share():
    def __init__(self, value1, value2, mod=MOD, inv_3=None, fp_prec=12, mod_scale=None):
        self.mod = mod
        if inv_3 != None:
            self.inv_3 = inv_3
        else:
            self.inv_3 = mod_inverse(3, self.mod)
        self.fp = fp_prec
        self.x = value1 % self.mod
        self.a = value2 % self.mod
        self.scale = 10**self.fp
        if mod_scale != None:
            self.mod_scale = mod_scale
        else:
            self.mod_scale = mod_inverse(self.scale, self.mod)

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

    def const_mult(self, const_value):
        new_x = (self.x * const_value * self.mod_scale) % self.mod
        new_a = (self.a * const_value * self.mod_scale) % self.mod
        return Share(new_x, new_a, mod=self.mod, inv_3=self.inv_3, fp_prec=self.fp)

    def const_add(self, const_value):
        new_x = self.x
        new_a = (self.a - const_value) % self.mod
        return Share(new_x, new_a, mod=self.mod, inv_3=self.inv_3, fp_prec=self.fp)

    def get_x(self):
        return self.x

    def get_a(self):
        return self.a
    
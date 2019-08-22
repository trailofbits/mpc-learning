import pytest
from random import randint
from mock import Mock
from _pytest.monkeypatch import MonkeyPatch
from src.circuits.share import Share
from src.util.mod import mod_inverse
import math

from src.circuits.evaluator import SecureEvaluator

class mock_mod2:
    def __init__(self,val1):
        self.val1 = val1
        self.cval = None

    def _reveal(self,value):
        self.cval = value.unshare(self.val1)
        return self.cval

    def get_cval(self):
        return self.cval

@pytest.fixture
def mock_secure_evaluator():
    return Mock(spec=SecureEvaluator)

@pytest.mark.parametrize(
    "value,k,mod,fpp,expected",
    [(1,2,19,0,1),(2,2,19,0,0),(3,5,61,1,10),(3,5,11003,2,100),(8,6,1009,1,0),(19,6,131,0,1)]
)
def test_mod2(value,k,mod,fpp,expected): 
    scale = 10**fpp
    value = value * scale

    b1 = randint(0,1) * scale
    b2 = randint(0,1) * scale
    b3 = b2

    # make shares of b1, b2, and b3
    b1a = randint(0,mod - 1)
    b1b = randint(0,mod - 1)
    b1c = (- (b1a+b1b)) % mod

    b2a = randint(0,mod - 1)
    b2b = randint(0,mod - 1)
    b2c = (- (b2a+b2b)) % mod

    b3a = randint(0,mod - 1)
    b3b = randint(0,mod - 1)
    b3c = (- (b3a+b3b)) % mod

    b1_share1 = Share(b1a,b1c - b1,mod=mod, fp_prec=fpp)
    b1_share2 = Share(b1b,b1a - b1,mod=mod, fp_prec=fpp)

    b2_share1 = Share(b2a,b2c - b2,mod=mod, fp_prec=fpp)
    b2_share2 = Share(b2b,b2a - b2,mod=mod, fp_prec=fpp)

    b3_share1 = Share(b3a,b3c - b3,mod=mod, fp_prec=fpp)
    b3_share2 = Share(b3b,b3a - b3,mod=mod, fp_prec=fpp)

    vala = randint(0,mod -1)
    valb = randint(0,mod -1)
    valc = (-(vala + valb)) % mod

    val_share1 = Share(vala,valc-value,mod=mod, fp_prec=fpp)
    val_share2 = Share(valb,vala-value,mod=mod, fp_prec=fpp)

    b1_share2 = b1_share2.switch_precision(0)
    b2_share2 = b2_share2.switch_precision(0)
    b3_share2 = b3_share2.switch_precision(0)

    val_share2 = val_share2.switch_precision(0)
    #val = val_share2.const_add(2**(k-1))
    val = val_share2
    val += b1_share2.const_mult(2) + b3_share2
    m = mock_mod2(val)

    rbits = [b1_share1,b2_share1,b3_share1]

    def rand_bits(obj,index,rand):
        return rbits

    monkeypatch = MonkeyPatch()
    evltr = SecureEvaluator(None,[],[],1,None,mod,fp_precision=fpp)
    monkeypatch.setattr('src.circuits.evaluator.SecureEvaluator.get_truncate_randomness',rand_bits)
    monkeypatch.setattr('src.circuits.evaluator.SecureEvaluator._reveal',m._reveal)
    evltr_out = evltr._mod2(val_share1,k)

    b3_share1 = b3_share1.switch_precision(0)

    c0 = int(bin(m.get_cval())[-1])
    party2_out = b3_share2.const_add(c0)
    party2_out += b3_share2.const_mult(-2*c0)
    party2_out = party2_out.switch_precision(fpp)


    assert (evltr_out.unshare(party2_out) % mod) == expected

class mock_premul:
    def __init__(self,mod,fpp):
        self.mod = mod
        self.fpp = fpp

    def add_rs_vals(self,r1_vals,r_vals,s1_vals,s_vals):
        self.r1_vals = r1_vals
        self.r_vals = r_vals
        self.s1_vals = s1_vals
        self.s_vals = s_vals

    def add_wa_vals(self,w1_vals,w_vals,a1_vals,a_vals):
        self.w1_vals = w1_vals
        self.w_vals = w_vals
        self.a1_vals = a1_vals
        self.a_vals = a_vals

    def add_v_vals_dict(self,r1_vals,s1_vals,r3_vals,s3_vals):
        self.v_vals_dict = {}
        for i in range(len(r1_vals) - 1):
            val1 = r1_vals[i+1].get_x() + r1_vals[i+1].get_a()
            val2 = s1_vals[i].get_x() + s1_vals[i].get_a()
            self.v_vals_dict[(val1,val2)] = r3_vals[i+1].pre_mult(s3_vals[i],0)

    def _reveal(self,value):
        for i in range(len(self.r1_vals)):
            if value == self.r1_vals[i]:
                return self.r_vals[i]
        for i in range(len(self.s1_vals)):
            if value == self.s1_vals[i]:
                return self.s_vals[i]
        for i in range(len(self.w1_vals)):
            if value == self.w1_vals[i]:
                return self.w_vals[i]
        for i in range(len(self.a1_vals)):
            if value == self.a1_vals[i]:
                return self.a_vals[i]
        else:
            print("value: " + str(value))
            print("couldn't find value !!!!!!!!!!!!!")

    def _multiply(self,value1,value2):
        r = value1.pre_mult(value2,0)
        val1 = value1.get_x() + value1.get_a()
        val2 = value2.get_x() + value2.get_a()
        if (val1,val2) in self.v_vals_dict:
            new_r = self.v_vals_dict[(val1,val2)]
        elif (val2,val1) in self.v_vals_dict:
            new_r = self.v_vals_dict[(val2,val1)]
        else:
            print("couldnt find tuple in v vals dict!!!!!")

        return Share(new_r - r, -2*new_r - r,mod=self.mod,fp_prec=self.fpp)

@pytest.mark.parametrize(
    "inputs,mod,fpp,expected",
    [([1,2],11003,0,[1,2]),([1,2,3],11003,0,[1,2,6]),
    ([2,2,2,2],11003,1,[2,4,8,16]),([3,5],11003,2,[3,15]),
    ([15,2,3],15485867,5,[15,30,90]),([6,1,7],15485867,4,[6,6,42])]
)
def test_premul(inputs,mod,fpp,expected):
    scale = 10**fpp
    mod_scale = mod_inverse(scale,mod)
    new_rand_mod = math.floor(mod / scale) - 1

    for i,inp in enumerate(inputs):
        inputs[i] = inp * scale
    a_vals = inputs

    m = mock_premul(mod,fpp)

    k = len(inputs)
    r_vals = []
    s_vals = []
    u_vals = []
    u_inv_vals = []
    a1_vals = []
    a2_vals = []
    r1_vals = []
    s1_vals = []
    r2_vals = []
    s2_vals = []
    r3_vals = []
    s3_vals = []
    for i in range(k):
        r_val = randint(1,new_rand_mod) * scale
        s_val = randint(1,new_rand_mod) * scale
        u_val = (r_val * s_val * mod_scale) % mod
        print("test vals:")
        print("r_" + str(i) + ": " + str(r_val))
        print("s_" + str(i) + ": " + str(s_val))
        print("u_" + str(i) + ": " + str(u_val))
        u_inv = mod_inverse(u_val * mod_scale,mod) * scale

        r_vals.append(r_val)
        s_vals.append(s_val)
        u_vals.append(u_val)
        u_inv_vals.append(u_inv)

        ra = randint(0,new_rand_mod) * scale
        rb = randint(0,new_rand_mod) * scale
        rc = (-(ra+rb)) % mod
        sa = randint(0,new_rand_mod) * scale
        sb = randint(0,new_rand_mod) * scale
        sc = (-(sa+sb)) % mod
        aa = randint(0,new_rand_mod) * scale
        ab = randint(0,new_rand_mod) * scale
        ac = (-(aa+ab)) % mod

        r1_vals.append(Share(ra, rc - r_val,mod=mod,fp_prec=fpp))
        s1_vals.append(Share(sa, sc - s_val,mod=mod,fp_prec=fpp))
        r2_vals.append(Share(rb, ra - r_val,mod=mod,fp_prec=fpp))
        s2_vals.append(Share(sb, sa - s_val,mod=mod,fp_prec=fpp))
        r3_vals.append(Share(rc, rb - r_val,mod=mod,fp_prec=fpp))
        s3_vals.append(Share(sc, sb - s_val,mod=mod,fp_prec=fpp))
        a1_vals.append(Share(aa, ac - a_vals[i],mod=mod,fp_prec=fpp))
        a2_vals.append(Share(ab, aa - a_vals[i],mod=mod,fp_prec=fpp))

    m.add_rs_vals(r1_vals,r_vals,s1_vals,s_vals)
    m.add_v_vals_dict(r1_vals,s1_vals,r3_vals,s3_vals)

    v_vals = []
    for i in range(len(r_vals) - 1):
        v_vals.append(r_vals[i+1]*s_vals[i]*mod_scale)

    v1_vals = []
    for i in range(len(r1_vals) - 1):
        r = r1_vals[i+1].pre_mult(s1_vals[i],0)
        new_r = r3_vals[i+1].pre_mult(s3_vals[i],0)
        v1_vals.append(Share(new_r - r, -2*new_r - r,mod=mod,fp_prec=fpp))

    v2_vals = []
    for i in range(len(r1_vals) - 1):
        r = r2_vals[i+1].pre_mult(s2_vals[i],0)
        new_r = r1_vals[i+1].pre_mult(s1_vals[i],0)
        v2_vals.append(Share(new_r - r, -2*new_r - r,mod=mod,fp_prec=fpp))

    print("raw v vals: ")
    for i in range(len(v_vals)):
        print(v_vals[i] % mod)
    print("cooked v vals: ")
    for i in range(len(v1_vals)):
        print(v1_vals[i].unshare(v2_vals[i]) % mod)

    w_vals = []
    w_vals.append(r_vals[0])
    for i in range(len(v_vals)):
        w_vals.append(v_vals[i]*u_inv_vals[i] * mod_scale % mod)

    w1_vals = []
    w1_vals.append(r1_vals[0])
    for i in range(len(v1_vals)):
        w1_vals.append(v1_vals[i].const_mult(u_inv_vals[i]))

    m.add_wa_vals(w1_vals,w_vals,a1_vals,a_vals)

    w2_vals = []
    w2_vals.append(r2_vals[0])
    for i in range(len(v2_vals)):
        w2_vals.append(v2_vals[i].const_mult(u_inv_vals[i]))

    print("raw w vals: ")
    for i in range(len(w_vals)):
        print(w_vals[i] % mod)
    print("cooked w vals: ")
    for i in range(len(w1_vals)):
        print(w1_vals[i].unshare(w2_vals[i]) % mod)

    z_vals = []
    for i in range(len(s_vals)):
        z_vals.append(s_vals[i] * u_inv_vals[i] * mod_scale % mod)

    z1_vals = []
    for i in range(len(s1_vals)):
        z1_vals.append(s1_vals[i].const_mult(u_inv_vals[i]))

    z2_vals = []
    for i in range(len(s2_vals)):
        z2_vals.append(s2_vals[i].const_mult(u_inv_vals[i]))

    print("raw z vals: ")
    for i in range(len(z_vals)):
        print(z_vals[i] % mod)
    print("cooked z vals: ")
    for i in range(len(z1_vals)):
        print(z1_vals[i].unshare(z2_vals[i]) % mod)
    

    m2_vals = []
    for i in range(len(w2_vals)):
        m2_vals.append(w_vals[i]*a_vals[i] * mod_scale % mod)
    
    print("test m vals: " + str(m2_vals))

    p_vals = []
    p_vals.append(a_vals[0])
    for i in range(1,len(z_vals)):
        m_prod = 1 * scale
        for j in range(i+1):
            m_prod *= m2_vals[j] * mod_scale
        p_vals.append(z_vals[i] * m_prod * mod_scale % mod)

    p2_vals = []
    p2_vals.append(a2_vals[0])
    for i in range(1,len(z2_vals)):
        m_prod = 1 * scale
        for j in range(i+1):
            m_prod *= m2_vals[j]
            m_prod *= mod_scale
        p2_vals.append(z2_vals[i].const_mult(m_prod))

    def rand_vals(obj,index,rand):
        return {'r': r1_vals, 's': s1_vals}
    
    monkeypatch = MonkeyPatch()
    evaluator = SecureEvaluator(None,[],[],1,None,mod,fp_precision=fpp)
    monkeypatch.setattr('src.circuits.evaluator.SecureEvaluator.get_truncate_randomness',rand_vals)
    monkeypatch.setattr('src.circuits.evaluator.SecureEvaluator._reveal',m._reveal)
    monkeypatch.setattr('src.circuits.evaluator.SecureEvaluator._multiply',m._multiply)

    eval_out = evaluator._premul(a1_vals)
    p1_vals = eval_out

    print("raw p vals: ")
    for i in range(len(p_vals)):
        print(p_vals[i] % mod)
    print("cooked p vals: ")
    for i in range(len(p1_vals)):
        print(p1_vals[i].unshare(p2_vals[i]) % mod)
    
    outs = []
    for i in range(len(eval_out)):
        out_val = (eval_out[i].unshare(p2_vals[i])) % mod
        outs.append(int(out_val / scale))

    assert outs == expected

class mock_bit_lt:
    def __init__(self,mod,fpp,a_bits):
        self.mod = mod
        self.fpp = fpp
        self.scale = 10**fpp
        self.mod_scale = mod_inverse(self.scale,self.mod)
        self.rand_mod = math.floor(mod / self.scale) - 1
        self.a_bits = a_bits

    def add_premul_vals(self,vals):
        self.premul_vals = vals

    def _premul(self,values):
        d_vals = []
        print("cooked d vals:")
        for i in range(len(values)):
            d_vals.append(values[i].unshare(self.premul_vals[i]))
            print(values[i].unshare(self.premul_vals[i]))
        
        raw_pm_vals = []
        for i in range(len(d_vals)):
            prod = 1*self.scale
            for j in range(i+1):
                prod *= d_vals[j] * self.mod_scale
                prod = prod % self.mod
            raw_pm_vals.append(prod)
        self.premul_raw = raw_pm_vals

        print("cooked p vals: ")
        for i in range(len(raw_pm_vals)):
            print(raw_pm_vals[i])

        shrs1 = []
        shrs2 = []
        shrs3 = []
        for i in range(len(raw_pm_vals)):
            a = randint(0,self.rand_mod) * self.scale
            b = randint(0,self.rand_mod) * self.scale
            c = (-(a+b)) % self.mod

            shrs1.append(Share(a,c - raw_pm_vals[i],mod=self.mod,fp_prec=self.fpp))
            shrs2.append(Share(b,a - raw_pm_vals[i],mod=self.mod,fp_prec=self.fpp))
            shrs3.append(Share(c,b - raw_pm_vals[i],mod=self.mod,fp_prec=self.fpp))

        shrs2.reverse()
        shrs3.reverse()
        
        self.pm_sh1 = shrs1
        self.pm_sh2 = shrs2
        self.pm_sh3 = shrs3

        self.premul_raw.reverse()

        return shrs1

    def _mod2(self,value,k):
        sh1 = self.pm_sh1
        s1_vals = []
        for i in range(len(sh1)-1):
            s1_vals.append(sh1[i] + sh1[i+1].const_mult(-1,scaled=False))
        s1_vals.append(sh1[-1].const_add(-1,scaled=False))

        sh2 = self.pm_sh2
        s2_vals = []
        for i in range(len(sh2)-1):
            s2_vals.append(sh2[i] + sh2[i+1].const_mult(-1,scaled=False))
        s2_vals.append(sh2[-1].const_add(-1,scaled=False))

        print("cooked s_vals: ")
        for i in range(len(s1_vals)):
            print(s1_vals[i].unshare(s2_vals[i]))
        
        #self.a_bits.reverse()
        s2len = len(s2_vals)
        s1 = Share(0,0,mod=self.mod,fp_prec=self.fpp)
        for i in range(len(s1_vals)):
            s1 += s1_vals[i].const_mult(self.scale - self.a_bits[i])

        s_val = Share(0,0,mod=self.mod,fp_prec=self.fpp)
        
        for i in range(s2len):
            s_val += s2_vals[i].const_mult(self.scale - self.a_bits[i])

        print("cooked a bits: " + str(self.a_bits))

        print("1st cooked s val: " + str(s1.unshare(s_val)))
        print("2nd cooked u val: " + bin(s1.unshare(s_val))[-1])
        
        print("cooked s val: " + str(value.unshare(s_val)))
        print("cooked u val: " + bin(value.unshare(s_val))[-1])

        self.u_val = int(bin(value.unshare(s_val) * self.mod_scale % self.mod)[-1]) * self.scale

        a = randint(0,self.rand_mod) * self.scale
        b = randint(0,self.rand_mod) * self.scale
        c = (-(a+b)) % self.mod

        self.u_sh1 = Share(a,c-self.u_val,mod=self.mod,fp_prec=self.fpp)
        self.u_sh2 = Share(b,a-self.u_val,mod=self.mod,fp_prec=self.fpp)
        self.u_sh3 = Share(c,b-self.u_val,mod=self.mod,fp_prec=self.fpp)

        return self.u_sh1

    def get_u_sh2(self):
        return self.u_sh2

@pytest.mark.parametrize(
    "val1,val2_bits,mod,fpp,expected",
    [(3,[0,1],11,0,0),(3,[1,1,1],11,0,1),(15,[1,0,0,1,0,0],11003,0,1),
    (50,[1,0,1,0,0],11003,0,0),(200,[1,0,0,1,0,1,1,0,0],11003,0,1),
    (6,[1,1,0],11003,0,0),(6,[1,1,1],11003,1,1),(19,[1,0,0,0,1],15485867,3,0),
    (19,[1,0,0,1,1],15485867,2,0)]
)
def test_bit_lt(val1,val2_bits,mod,fpp,expected):   
    scale = 10**fpp
    #val1 = val1 * scale % mod
    rand_mod = math.floor(mod / scale) - 1
    mod_scale = mod_inverse(scale,mod)

    val1_len = len(bin(val1)[2:])
    print("val1 len: " + str(val1_len))
    print("val2 len: " + str(len(val2_bits)))

    for i,bit in enumerate(val2_bits):
        val2_bits[i] = bit*scale

    print("test")
    
    if len(val2_bits) < val1_len:
        print("add bits")
        val2_bits = [0]*(val1_len - len(val2_bits)) + val2_bits

    val1_bits = []
    for bit in bin(val1)[2:]:
        val1_bits.append(int(bit)*scale)
    if len(val1_bits) > len(val2_bits):
        print("not enough bits for val 2!!!!!!")
    else:
        print("adding val1 bits")
        val1_bits = [0]*(len(val2_bits) - len(val1_bits)) + val1_bits
        print("val1: " + str(val1) + " val1_bits: " + str(val1_bits))
        print("val2_bits: " + str(val2_bits))

    if len(val1_bits) != len(val2_bits):
        print("LENGHT MISMATCH!!")

    raw_d = []
    print("raw d vals: ")
    for i in range(len(val1_bits)):
        d_val = val1_bits[i] + val2_bits[i] - 2*val1_bits[i]*val2_bits[i]*mod_scale + 1*scale
        print(d_val % mod)
        raw_d.append(d_val % mod)

    bit_shares1 = []
    bit_shares2 = []
    bit_shares3 = []
    for i in range(len(val2_bits)):
        a = randint(0,rand_mod) * scale
        b = randint(0,rand_mod) * scale
        c = (-(a+b)) % mod

        bit_shares1.append(Share(a,c - val2_bits[i],mod=mod,fp_prec=fpp))
        bit_shares2.append(Share(b,a - val2_bits[i],mod=mod,fp_prec=fpp))
        bit_shares3.append(Share(c,b - val2_bits[i],mod=mod,fp_prec=fpp))

    m = mock_bit_lt(mod,fpp,val1_bits)

    pm2_vals = []
    for i in range(len(val2_bits)):
        d_val = bit_shares2[i].const_add(val1_bits[i])
        d_val += bit_shares2[i].const_mult(-2*val1_bits[i])
        d_val = d_val.const_add(1,scaled=False)
        pm2_vals.append(d_val)

    m.add_premul_vals(pm2_vals)

    raw_p = []
    print("raw p vals: ")
    for i in range(len(raw_d)):
        prod = 1*scale
        for j in range(i+1):
            prod *= raw_d[j]
            prod *= mod_scale
            prod = prod % mod
        raw_p.append(prod)
        print(prod)

    raw_p.reverse()
    raw_s_vals = []
    print("raw s vals: ")
    for i in range(len(raw_p)-1):
        s_vals = raw_p[i] - raw_p[i+1]
        s_vals = s_vals % mod
        print(s_vals)
        raw_s_vals.append(s_vals)
    print(raw_p[-1] - 1*scale)
    raw_s_vals.append((raw_p[-1] - 1*scale) % mod)

    raw_s = 0
    print("raw intermediate s:")
    val1_bits.reverse()
    print("raw a bits: " + str(val1_bits))
    for i in range(len(raw_s_vals)):
        print(raw_s_vals[i] * (1*scale - val1_bits[i]) * mod_scale % mod)
        raw_s += raw_s_vals[i]*(1*scale - val1_bits[i]) * mod_scale
    raw_s = raw_s % mod
    print("raw s: ") 
    print(raw_s)

    print("raw u: " + bin(raw_s * mod_scale % mod)[-1])
    
    monkeypatch = MonkeyPatch()
    evaluator = SecureEvaluator(None,[],[],1,None,mod,fp_precision=fpp)
    monkeypatch.setattr('src.circuits.evaluator.SecureEvaluator._premul',m._premul)
    monkeypatch.setattr('src.circuits.evaluator.SecureEvaluator._mod2',m._mod2)

    out_val = evaluator._bit_lt(val1,bit_shares1)
    u_sh2 = m.get_u_sh2()

    assert ((out_val.unshare(u_sh2)) / scale) % mod == expected

class mock_mod2m:

    def __init__(self,mod,fpp):
        self.mod = mod
        self.fpp = fpp
        self.scale = 10**fpp
        self.mod_scale = mod_inverse(self.scale,self.mod)
        self.rand_mod = math.floor(mod / self.scale) - 1

    def add_reveal(self,reveal):
        self.reveal = reveal

    def add_raw_r1(self, r1):
        self.r1 = r1

    def _reveal(self,value):
        self.c = value.unshare(self.reveal) % self.mod
        return self.c

    def _bit_lt(self, a, b_bits):
        self.u = (a < self.r1) * self.scale
        
        ua = randint(0,self.rand_mod) * self.scale
        ub = randint(0,self.rand_mod) * self.scale
        uc = (-(ua+ub)) % self.mod

        self.u1 = Share(ua,uc-self.u,mod=self.mod,fp_prec=self.fpp)
        self.u2 = Share(ub,ua-self.u,mod=self.mod,fp_prec=self.fpp)
        self.u3 = Share(uc,ub-self.u,mod=self.mod,fp_prec=self.fpp)

        return self.u1

    def get_c(self):
        return self.c

    def get_u1(self):
        return self.u1

    def get_u2(self):
        return self.u2

@pytest.mark.parametrize(
    "value,k,fpp,mod_exp,mod,expected",
    [(14,2,0,5,11003,14),(20,2,0,4,11003,4),
    (104,3,1,6,11003,16),(3,14,3,8,15485867,184),
    (6,10,5,1,15458567,0),(2,11,1,3,11003,4)]
)
def test_mod2m(value,k,fpp,mod_exp,mod,expected):
    scale = 10**fpp
    mod_scale = mod_inverse(scale,mod)
    rand_mod = math.floor(mod / scale) - 1
    value = value * scale
    rand_val_mod = 2**mod_exp - 1

    vala = randint(0,rand_mod) * scale
    valb = randint(0,rand_mod) * scale
    valc = (-(vala+valb)) % mod

    val_1 = Share(vala,valc - value,mod=mod,fp_prec=fpp)
    val_2 = Share(valb,vala - value,mod=mod,fp_prec=fpp)
    val_3 = Share(valc,valb - value,mod=mod,fp_prec=fpp)

    r2 = randint(0,rand_val_mod)
    r1 = randint(0,rand_val_mod)
    r1_bits = []
    for bit in bin(r1)[2:]:
        r1_bits.append(int(bit)*scale)
    r1_bits = [0]*(mod_exp - len(r1_bits)) + r1_bits

    print("raw r2: " + str(r2))
    print("raw r1: " + str(r1))
    print("raw r1_bits: " + str(r1_bits))

    r2_r1_shares1 = []
    r2_r1_shares2 = []
    r2_r1_shares3 = []

    r2a = randint(0,rand_mod) * scale
    r2b = randint(0,rand_mod) * scale
    r2c = (-(r2a + r2b)) % mod
    r1a = randint(0,rand_mod) * scale
    r1b = randint(0,rand_mod) * scale
    r1c = (-(r1a + r1b)) % mod

    r2_share1 = Share(r2a,r2c-r2,mod=mod,fp_prec=fpp)
    r2_share2 = Share(r2b,r2a-r2,mod=mod,fp_prec=fpp)
    r2_share3 = Share(r2c,r2b-r2,mod=mod,fp_prec=fpp)
    
    r1_share1 = Share(r1a,r1c-r1,mod=mod,fp_prec=fpp)
    r1_share2 = Share(r1b,r1a-r1,mod=mod,fp_prec=fpp)
    r1_share3 = Share(r1c,r1b-r1,mod=mod,fp_prec=fpp)

    r2_r1_shares1.append(r2_share1)
    r2_r1_shares1.append(r1_share1)
    r2_r1_shares2.append(r2_share2)
    r2_r1_shares2.append(r1_share2)
    r2_r1_shares3.append(r2_share3)
    r2_r1_shares3.append(r1_share3)

    for i in range(len(r1_bits)):
        a = randint(0,rand_mod) * scale
        b = randint(0,rand_mod) * scale
        c = (-(a+b)) % mod

        shr1 = Share(a,c-r1_bits[i],mod=mod,fp_prec=fpp)
        shr2 = Share(b,a-r1_bits[i],mod=mod,fp_prec=fpp)
        shr3 = Share(c,b-r1_bits[i],mod=mod,fp_prec=fpp)

        r2_r1_shares1.append(shr1)
        r2_r1_shares2.append(shr2)
        r2_r1_shares3.append(shr3)
    
    def rand_vals(obj,index,rand):
        return r2_r1_shares1

    r2_1 = r2_r1_shares1[0]
    r1_1 = r2_r1_shares1[1]

    r2_2 = r2_r1_shares2[0]
    r1_2 = r2_r1_shares2[1]
    r1_bits_2 = r2_r1_shares2[2:]

    print("real r2: " + str(r2) + ", chk r2: " + str(r2_1.unshare(r2_2) % mod))
    print("real r1: " + str(r1) + ", chk r1: " + str(r1_1.unshare(r1_2) % mod))

    raw_c = value + (2**mod_exp)*r2 + r1
    raw_c_prime = raw_c % 2**mod_exp
    print("raw c: " + str(raw_c))
    print("raw c_prime: " + str(raw_c_prime))

    raw_u = int(raw_c_prime < r1)
    print("raw u: " + str(raw_u))

    raw_a_prime = raw_c_prime - r1 + (2**mod_exp)*raw_u
    print("raw a_prime: " + str(raw_a_prime))

    #pre2_c = val_2.const_add(2**(k-1),scaled=False)
    pre2_c = val_2
    pre2_c += r2_2.const_mult(2**mod_exp,scaled=False)
    pre2_c += r1_2

    pre1_c = val_1
    pre1_c += r2_1.const_mult(2**mod_exp,scaled=False)
    pre1_c += r1_1

    print("test c: " + str(pre1_c.unshare(pre2_c) % mod))

    m = mock_mod2m(mod,fpp)
    m.add_reveal(pre2_c)
    m.add_raw_r1(r1)

    monkeypatch = MonkeyPatch()
    evaluator = SecureEvaluator(None,[],[],1,None,mod,fp_precision=fpp)
    monkeypatch.setattr('src.circuits.evaluator.SecureEvaluator.get_truncate_randomness',rand_vals)
    monkeypatch.setattr('src.circuits.evaluator.SecureEvaluator._reveal',m._reveal)
    monkeypatch.setattr('src.circuits.evaluator.SecureEvaluator._bit_lt',m._bit_lt)

    real_a1_prime = evaluator._mod2m(val_1,k,mod_exp)

    c = m.get_c()
    print("cooked c: " + str(c))
    c_prime = int(c % 2**mod_exp)
    print("cooked c_prime: " + str(c_prime))
    u1 = m.get_u1()
    u2 = m.get_u2()
    #print("cooked u: " + str(u2))

    a1_prime = r1_1.const_mult(-1,scaled=False)
    a2_prime = r1_2.const_mult(-1,scaled=False)
    print("raw intermediate a_prime: " + str(-r1 % mod))
    print("cooked intermediate a_prime: " + str(a1_prime.unshare(a2_prime) % mod))
    a1_prime = a1_prime.const_add(c_prime)
    a2_prime = a2_prime.const_add(c_prime)
    print("raw intermediate a_prime: " + str((-r1 + raw_c_prime) % mod))
    print("cooked intermediate a_prime: " + str(a1_prime.unshare(a2_prime) % mod))
    
    print("raw u * 2**mod_exp: " + str(raw_u * (2**mod_exp)))
    print("cooked u * 2**mod_exp: " + str(u1.const_mult(2**mod_exp).unshare(u2.const_mult(2**mod_exp))))

    a1_prime += u1.const_mult(2**mod_exp)
    a2_prime += u2.const_mult(2**mod_exp)
    print("raw final a_prime: " + str((-r1 + raw_c_prime + raw_u*(2**mod_exp)) % mod))
    print("cooked final a_prime: " + str(a1_prime.unshare(a2_prime) % mod))

    assert (real_a1_prime.unshare(a2_prime) % mod) == expected    

class mock_truncate:

    def __init__(self,mod,fpp):
        self.mod = mod
        self.fpp = fpp
        self.scale = 10**fpp
        self.mod_scale = mod_inverse(self.scale,self.mod)
        self.rand_mod = math.floor(mod / self.scale) - 1

    def add_value(self, value):
        self.value = value

    def _mod2m(self, val, k, m):
        self.raw_mod2m = self.value % 2**m

        a = randint(0,self.rand_mod) * self.scale
        b = randint(0,self.rand_mod) * self.scale
        c = (-(a+b)) % self.mod

        self.shr1 = Share(a,c - self.raw_mod2m,mod=self.mod,fp_prec=self.fpp)
        self.shr2 = Share(b,a - self.raw_mod2m,mod=self.mod,fp_prec=self.fpp)
        self.shr3 = Share(c,b - self.raw_mod2m,mod=self.mod,fp_prec=self.fpp)

        return self.shr1

    def get_shr1(self):
        return self.shr1

    def get_shr2(self):
        return self.shr2

@pytest.mark.parametrize(
    "value,k,fpp,trunc_exp,mod,expected",
    [(3,0,0,1,43,1),(15,0,0,3,43,1),(15,0,0,2,43,3),
    (12,0,0,2,43,3),(401,0,0,2,11003,100),
    (7,0,3,6,11003,109),(12,0,2,5,11003,37)]
)
def test_truncate(value,k,fpp,trunc_exp,mod,expected):
    scale = 10**fpp
    mod_scale = mod_inverse(scale,mod)
    rand_mod = math.floor(mod / scale) - 1
    value = value * scale
    trunc = 2**trunc_exp

    raw_a_prime = value % trunc
    print("raw a_prime: " + str(raw_a_prime))

    raw_d = ((value - raw_a_prime) * mod_inverse(trunc,mod)) % mod
    print("raw d: " + str(raw_d))
    print("value: " + str(value))
    print("trunc: " + str(trunc))
    print("expected: " + str(expected))

    a = randint(0,rand_mod) * scale
    b = randint(0,rand_mod) * scale
    c = (-(a+b)) % mod

    val1 = Share(a,c-value,mod=mod,fp_prec=fpp)
    val2 = Share(b,a-value,mod=mod,fp_prec=fpp)
    val3 = Share(c,b-value,mod=mod,fp_prec=fpp)

    m = mock_truncate(mod,fpp)
    m.add_value(value)

    monkeypatch = MonkeyPatch()
    evaluator = SecureEvaluator(None,[],[],1,None,mod,fp_precision=fpp)
    monkeypatch.setattr('src.circuits.evaluator.SecureEvaluator._mod2m',m._mod2m)

    d1_val = evaluator._truncate(val1,k,trunc_exp)

    a2_prime = m.get_shr2()
    d2 = val2 + a2_prime.const_mult(-1,scaled=False)
    d2 = d2.const_mult(mod_inverse(trunc,mod),scaled=False)

    assert (d1_val.unshare(d2) % mod) == expected

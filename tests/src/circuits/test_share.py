import pytest
from random import randint

from src.circuits.share import Share


@pytest.mark.parametrize(
    "share,const,mod,expected",
    [(2,2,13,4),(1,2,11,3),(14,15,43,29),
    (21,6,23,4),(100,200,5,0),(14,-6,43,8),
    (51,-53,11003,11001)]
)
def test_const_add(share,const,mod,expected):
    a = randint(0,mod-1)
    b = randint(0,mod-1)
    c = (-(a+b)) % mod

    share1 = Share(a, c - share, mod=mod, fp_prec=0)
    share2 = Share(b, a - share, mod=mod, fp_prec=0)

    share1_add = share1.const_add(const)
    share2_add = share2.const_add(const)

    assert expected == (share1_add.unshare(share2_add) % mod)

@pytest.mark.parametrize(
    "share,const,mod,fpp,expected",
    [(2,2,11003,2,4),(1,2,11003,1,3),(14,15,43,0,29),
    (21,6,23,0,4),(100,200,5,0,0),(14,-6,43,0,8),
    (51,-53,11003,1,11001)]
)
def test_const_add_scale(share,const,mod,fpp,expected):
    scale = 10**fpp
    share = share * scale

    a = randint(0,mod-1)
    b = randint(0,mod-1)
    c = (-(a+b)) % mod

    share1 = Share(a, c - share, mod=mod, fp_prec=fpp)
    share2 = Share(b, a - share, mod=mod, fp_prec=fpp)

    share1_add = share1.const_add(const,scaled=False)
    share2_add = share2.const_add(const,scaled=False)

    assert expected * scale == (share1_add.unshare(share2_add) % mod)

@pytest.mark.parametrize(
    "share,const,mod,expected",
    [(2,2,13,4),(1,2,11,2),(3,3,17,9),(14,2,5,3)]
)
def test_const_mult(share,const,mod,expected):

    a = randint(0,mod-1)
    b = randint(0,mod-1)
    c = (-(a+b)) % mod

    share1 = Share(a, c - share, mod=mod, fp_prec=0)
    share2 = Share(b, a - share, mod=mod, fp_prec=0)

    share1_mult = share1.const_mult(const)
    share2_mult = share2.const_mult(const)

    assert expected == (share1_mult.unshare(share2_mult) % mod)

@pytest.mark.parametrize(
    "share,old_prec,new_prec,mod,expected",
    [(20,1,0,43,2),(300,2,1,1009,30),(1400,2,1,11003,140),(20,1,3,11003,2000)]
)
def test_switch_precision(share,old_prec,new_prec,mod,expected):
    a = randint(0,mod-1)
    b = randint(0,mod-1)
    c = (-(a+b)) % mod

    share1 = Share(a, c - share, mod=mod, fp_prec=old_prec)
    share2 = Share(b, a - share, mod=mod, fp_prec=old_prec)

    share1_new = share1.switch_precision(new_prec)
    share2_new = share2.switch_precision(new_prec)

    assert expected == (share1_new.unshare(share2_new) % mod)

@pytest.mark.parametrize(
    "share1,share2,mod,fpp,expected",
    [((1,1),(1,1),11,0,True),((1,2),(12,13),11,0,True),((1,5),(1,4),11,0,False)]
)
def test_eq(share1,share2,mod,fpp,expected):
    shr1 = Share(share1[0],share1[1],mod=mod,fp_prec=fpp)
    shr2 = Share(share2[0],share2[1],mod=mod,fp_prec=fpp)

    assert (shr1 == shr2) == expected
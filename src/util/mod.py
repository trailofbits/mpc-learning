def mod_inverse(val, mod):
    g, x, y = egcd(val, mod)
    if g != 1:
        raise Exception('modular inverse does not exist')
    else:
        return x % mod

def egcd(a,b):
    if a == 0:
        return (b,0,1)
    else:
        g, y, x = egcd(b %a, a)
        return (g, x - (b //a) * y, y)
import random

# Code taken from https://gist.github.com/Ayrx/5884790
# implements primality testing

def miller_rabin(n, k=40):

    # Implementation uses the Miller-Rabin Primality Test
    # The optimal number of rounds for this test is 40
    # See http://stackoverflow.com/questions/6325576/how-many-iterations-of-rabin-miller-should-i-use-for-cryptographic-safe-primes
    # for justification

    # If number is even, it's a composite number

    if n == 2 or n == 3:
        return True

    if n % 2 == 0:
        return False

    r, s = 0, n - 1
    while s % 2 == 0:
        r += 1
        s //= 2
    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = pow(a, s, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True

if __name__ == "__main__":

    test_vals = [2,3,4,5,6,7,9,43,11003,10233]

    for tv in test_vals:
        print("value " + str(tv) + " is prime: " + str(miller_rabin(tv,k=40)))
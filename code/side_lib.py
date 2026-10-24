import math
def hw(x): return bin(x & 0xFF).count('1')
def postmax(eta, sig):
    mus = sorted(set(hw(s % 256) for s in range(-eta, eta + 1)))
    m = 2 * eta + 1
    Phi = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))
    tot = 0.0
    for i, mu in enumerate(mus):
        hi = 1.0 if i == len(mus) - 1 else Phi(((mus[i + 1] + mu) / 2 - mu) / sig)
        lo = 0.0 if i == 0 else Phi(((mus[i - 1] + mu) / 2 - mu) / sig)
        tot += hi - lo
    return tot / m
def h_bits(eta, sig): return -math.log2(postmax(eta, sig))

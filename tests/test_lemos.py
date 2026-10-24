"""Smoke tests for the reference implementation: python3 tests/test_lemos.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))
from lemos import Ring, LeMoS, lambda0

n, q = 256, 8380417
R = Ring(n, q)
rng = np.random.default_rng(1)
a = rng.integers(0, q, n); b = rng.integers(-3, 4, n)
c = R.intt(R.ntt(a) * R.ntt(b) % q)
ref = [0] * n
for i in range(n):
    for j in range(n):
        v = int(a[i]) * int(b[j])
        if i + j < n: ref[i + j] += v
        else: ref[i + j - n] -= v
assert np.array_equal(c, np.array([x % q for x in ref])), 'NTT product differs from schoolbook negacyclic product'
print('negacyclic NTT matches schoolbook product')

sch = LeMoS(256, q, 1, 16, 1)
pk, sk = sch.keygen(rng)
assert sch.verify(pk, sk), 'honest token rejected'
assert not sch.verify(pk, np.roll(sk, 1, axis=1)), 'wrong token accepted'
print('honest token accepted, wrong token rejected')

assert abs(lambda0(256, 1, 8, 7681, 3) - 2445.3) < 1.0
print('lambda_0 for profile P2 = %.1f bits' % lambda0(256, 1, 8, 7681, 3))
print('all tests passed')

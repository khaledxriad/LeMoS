"""LeMoS reference implementation (Python/NumPy).

Ring R_q = Z_q[X]/(X^n+1) with q = 1 (mod 2n) so that a full negacyclic NTT exists.
Key pair:  sk = s in S_eta^l (uniform short vector),  pk = (seed, t = A s mod q),
A = ExpandA(seed) in R_q^{k x l}.
"""
import hashlib, math, os
import numpy as np

# ---------------------------------------------------------------- ring / NTT
def is_prime(p):
    if p < 2:
        return False
    i = 2
    while i * i <= p:
        if p % i == 0:
            return False
        i += 1
    return True


def prim_root_2n(q, n):
    """a primitive 2n-th root of unity modulo q (q prime, q = 1 mod 2n)."""
    assert (q - 1) % (2 * n) == 0
    for g in range(2, q):
        w = pow(g, (q - 1) // (2 * n), q)
        if pow(w, n, q) == q - 1:
            return w
    raise ValueError


class Ring:
    def __init__(self, n, q):
        self.n, self.q = n, q
        self.psi = prim_root_2n(q, n)
        self.psi_inv = pow(self.psi, q - 2, q)
        w = self.psi * self.psi % q
        self.n_inv = pow(n, q - 2, q)
        # twist vectors
        self.tw = np.array([pow(self.psi, i, q) for i in range(n)], dtype=np.int64)
        self.tw_inv = np.array([pow(self.psi_inv, i, q) * self.n_inv % q for i in range(n)], dtype=np.int64)
        # per-stage twiddles for iterative radix-2 DIT cyclic NTT
        self.stages = []
        self.stages_inv = []
        wi = pow(w, q - 2, q)
        m = 2
        while m <= n:
            half = m // 2
            wm = pow(w, n // m, q)
            wmi = pow(wi, n // m, q)
            self.stages.append(np.array([pow(wm, j, q) for j in range(half)], dtype=np.int64))
            self.stages_inv.append(np.array([pow(wmi, j, q) for j in range(half)], dtype=np.int64))
            m *= 2
        # bit reversal permutation
        bits = n.bit_length() - 1
        self.rev = np.array([int(format(i, '0%db' % bits)[::-1], 2) for i in range(n)])

    def _cyc(self, a, stages):
        q, n = self.q, self.n
        a = a[..., self.rev]
        lead = a.shape[:-1]
        m = 2
        for tw in stages:
            half = m // 2
            a = a.reshape(lead + (n // m, 2, half))
            u = a[..., 0, :]
            v = (a[..., 1, :] * tw) % q
            a = np.stack(((u + v) % q, (u - v) % q), axis=-2)
            m *= 2
        return a.reshape(lead + (n,))

    def ntt(self, a):
        return self._cyc((a % self.q) * self.tw % self.q, self.stages)

    def intt(self, a):
        return self._cyc(a, self.stages_inv) * self.tw_inv % self.q


# ---------------------------------------------------------------- sampling
def expand_A(ring, seed, k, l):
    """Deterministic uniform A (kept in the NTT domain) from a 32-byte seed using SHAKE-128."""
    n, q = ring.n, ring.q
    need = k * l * n
    nbytes = 3 * (need + need // 2 + 64)
    # q < 2^23 for the default prime; mask accordingly and reject
    bitlen = q.bit_length()
    nb = (bitlen + 7) // 8
    raw = np.frombuffer(hashlib.shake_128(seed).digest(nb * (need * 3)), dtype=np.uint8)
    raw = raw[: (len(raw) // nb) * nb].reshape(-1, nb).astype(np.int64)
    vals = np.zeros(raw.shape[0], dtype=np.int64)
    for i in range(nb):
        vals |= raw[:, i] << (8 * i)
    vals &= (1 << bitlen) - 1
    vals = vals[vals < q][:need]
    assert len(vals) == need
    return vals.reshape(k, l, n)


def sample_secret(rng, eta, l, n):
    return rng.integers(-eta, eta + 1, size=(l, n), dtype=np.int64)


# ---------------------------------------------------------------- scheme
class LeMoS:
    def __init__(self, n, q, k, l, eta, eta_v=None):
        self.n, self.q, self.k, self.l, self.eta = n, q, k, l, eta
        self.eta_v = eta if eta_v is None else eta_v
        self.ring = Ring(n, q)

    # Algorithm 1 -------------------------------------------------
    def keygen(self, rng, seed=None):
        seed = os.urandom(32) if seed is None else seed
        A = expand_A(self.ring, seed, self.k, self.l)
        s = sample_secret(rng, self.eta, self.l, self.n)
        t = self._matvec_ntt(A, self.ring.ntt(s))
        return (seed, t), s

    def _matvec_ntt(self, A_ntt, s_ntt):
        q = self.q
        # A: (k,l,n) NTT domain, s_ntt: (l,n)
        acc = np.zeros((self.k, self.n), dtype=np.int64)
        for j in range(self.l):
            acc = (acc + A_ntt[:, j, :] * s_ntt[j][None, :]) % q
        return self.ring.intt(acc)

    def verify(self, pk, token):
        seed, t = pk
        s = token
        if s.shape != (self.l, self.n) or np.max(np.abs(s)) > self.eta_v:
            return False
        A = expand_A(self.ring, seed, self.k, self.l)
        return bool(np.array_equal(self._matvec_ntt(A, self.ring.ntt(s)), t))

    # sizes (bytes) with bit packing -------------------------------
    def sizes(self):
        n, q, k, l, eta = self.n, self.q, self.k, self.l, self.eta
        bq = math.ceil(math.log2(q))
        bs = math.ceil(math.log2(2 * eta + 1))
        pk = 32 + math.ceil(k * n * bq / 8)
        sk = math.ceil(l * n * bs / 8)
        return dict(pk=pk, sk=sk, token=sk)


# ---------------------------------------------------------------- leakage maths
def log2_secret_space(n, l, eta):
    return n * l * math.log2(2 * eta + 1)


def lambda0(n, k, l, q, eta):
    """secret entropy minus image entropy: log2|S| - nk log2 q  (bits)."""
    return log2_secret_space(n, l, eta) - n * k * math.log2(q)


def rho(k, l, q, eta):
    return q ** k / (2 * eta + 1) ** l


def gap_split(n, k, l, q, eta):
    return n * math.log2(1 + rho(k, l, q, eta))


def gap_inert(n, k, l, q, eta):
    r = rho(k, l, q, eta)
    return math.log2(1 + r ** n) if r < 1e6 else float('inf')


# ---------------------------------------------------------------- MSIS estimator
def delta_bkz(b):
    return ((math.pi * b) ** (1.0 / b) * b / (2 * math.pi * math.e)) ** (1.0 / (2 * (b - 1)))


def msis_core_svp(n, k, l, q, beta, bmax=4000):
    """Smallest BKZ block size b that finds a vector of l2-norm <= beta in the q-ary lattice
    (dim <= n*l, det q^{nk}); returns (b, classical bits, quantum bits).  Standard root-Hermite model."""
    dim_max = n * l
    nk = n * k
    if beta >= q:
        return (2, 0.0, 0.0)
    for b in range(40, bmax):
        d = delta_bkz(b)
        ld = math.log(d)
        m = min(dim_max, math.sqrt(nk * math.log(q) / ld))
        m = max(m, nk + 1)
        if m > dim_max:
            continue
        length = d ** m * q ** (nk / m)
        if length <= beta and b <= m:
            return (b, 0.292 * b, 0.265 * b)
        if b > m:
            break
    return (None, float('inf'), float('inf'))


def beta_of(n, l, eta, eta_v=None):
    ev = eta if eta_v is None else eta_v
    return (eta + ev) * math.sqrt(n * l)


# ---------------------------------------------------------------- Algorithm 3: parameter selection
def select_params(sec_bits, lam_target, kappa=128, ns=(256,), qs=(8380417, 786433, 12289, 7681),
                  etas=(1, 2, 3, 4, 5, 6, 8, 10, 16), ks=(1, 2, 3, 4), lmax=96):
    best = None
    for n in ns:
        for q in qs:
            if (q - 1) % (2 * n) or not is_prime(q):
                continue
            bq = math.ceil(math.log2(q))
            for eta in etas:
                bs = math.ceil(math.log2(2 * eta + 1))
                for k in ks:
                    for l in range(k + 1, lmax):
                        lam = lambda0(n, k, l, q, eta) - kappa
                        if lam_target is not None and lam < lam_target:
                            continue
                        b, cl, qu = msis_core_svp(n, k, l, q, beta_of(n, l, eta))
                        sec = cl
                        if b is None:
                            continue  # no short vector exists / attack model vacuous: not a meaningful regime
                        if sec < sec_bits:
                            continue
                        if lam_target is None and lam > -kappa - 1e9:
                            pass
                        size = 32 + math.ceil(k * n * bq / 8) + math.ceil(l * n * bs / 8)
                        cand = dict(n=n, q=q, k=k, l=l, eta=eta, lam=lam, lam0=lambda0(n, k, l, q, eta), sec=min(sec, 999),
                                    pk=32 + math.ceil(k * n * bq / 8), sk=math.ceil(l * n * bs / 8), total=size,
                                    rate=lambda0(n, k, l, q, eta) / log2_secret_space(n, l, eta))
                        if best is None or size < best['total']:
                            best = cand
                        break  # larger l only increases size for this (eta,k)
    return best

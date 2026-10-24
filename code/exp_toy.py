"""Exhaustive verification of Theorems 1-2 on toy module parameters."""
import itertools, json, math, sys
import numpy as np
rng = np.random.default_rng(20261024)

def negacyclic(a, n, q):
    M = np.zeros((n, n), dtype=np.int64)       # M[i,j]: coefficient i of a*X^j
    for j in range(n):
        for i in range(n):
            idx = i - j
            M[i, j] = a[idx] if idx >= 0 else (-a[idx + n]) % q
    return M % q

def all_secrets(n, l, eta):
    base = 2 * eta + 1
    N = base ** (n * l)
    idx = np.arange(N, dtype=np.int64)
    digs = np.empty((N, n * l), dtype=np.int8)
    for d in range(n * l):
        digs[:, d] = (idx % base) - eta
        idx //= base
    return digs

def split_or_inert(n, q):
    # roots of X^n+1 mod q
    roots = sum(1 for x in range(q) if (pow(x, n, q) + 1) % q == 0)
    if roots == n: return 'split'
    if roots == 0 and all(True for _ in [0]):
        # inert iff X^n+1 irreducible: check no factor of degree < n (brute force for tiny n)
        return 'inert?' 
    return 'mixed'

def run(n, q, k, l, eta, trials):
    S = all_secrets(n, l, eta).astype(np.int64)
    NS = S.shape[0]
    qb = q ** (n * k)
    ims, colls = [], []
    succ = {}
    lam_grid = list(range(0, int(math.log2(NS)) + 1))
    for lam in lam_grid: succ[lam] = 0.0
    for _ in range(trials):
        A = rng.integers(0, q, size=(k, l, n))
        # big matrix  (l*n) x (k*n)
        B = np.zeros((l * n, k * n), dtype=np.int64)
        for i in range(k):
            for j in range(l):
                B[j * n:(j + 1) * n, i * n:(i + 1) * n] = negacyclic(A[i, j], n, q).T
        T = (S @ B) % q
        key = np.zeros(NS, dtype=np.int64)
        for c in range(k * n):
            key = key * q + T[:, c]
        _, counts = np.unique(key, return_counts=True)
        ims.append(len(counts))
        colls.append(float(np.sum(counts.astype(np.float64) ** 2)) / NS ** 2)
        for lam in lam_grid:
            succ[lam] += np.minimum(counts, 2 ** lam).sum() / NS
    ims = np.array(ims); 
    EIm = ims.mean(); Ecoll = float(np.mean(colls))
    r = q ** k / (2 * eta + 1) ** l
    out = dict(n=n, q=q, k=k, l=l, eta=eta, NS=NS, log2S=math.log2(NS), nklogq=n * k * math.log2(q),
               rho=r, lam0=math.log2(NS) - n * k * math.log2(q),
               H_emp=math.log2(NS) - math.log2(EIm), EIm=EIm, Ecoll=Ecoll,
               gap_emp=math.log2(q ** (n * k) / EIm),
               gap_split=n * math.log2(1 + r), gap_inert=math.log2(1 + r ** n),
               succ={lam: succ[lam] / trials for lam in lam_grid},
               kind=split_or_inert(n, q), trials=trials)
    # theorem checks (tolerate 1e-9)
    ok1 = all(out['succ'][lam] <= 2 ** lam * EIm / NS + 1e-9 for lam in lam_grid)
    ok1b = all(2 ** lam * EIm / NS <= 2 ** (lam - out['lam0']) + 1e-9 for lam in lam_grid)
    ok2 = all(out['succ'][lam] >= 1 - NS * Ecoll / (4 * 2 ** lam) - 1e-9 for lam in lam_grid)
    out.update(ok_thm1=bool(ok1 and ok1b), ok_thm2=bool(ok2))
    return out

cfgs = [  # n,q,k,l,eta,trials
    (2, 5, 1, 3, 1, 60), (2, 5, 1, 4, 1, 60), (2, 13, 1, 3, 2, 40), (2, 13, 1, 4, 2, 12),
    (2, 5, 2, 5, 1, 12), (4, 17, 1, 3, 1, 4),
    (2, 3, 1, 3, 1, 60), (2, 7, 1, 3, 2, 40), (2, 7, 1, 4, 2, 12), (2, 11, 1, 3, 2, 40),
]
res = []
for c in cfgs:
    r = run(*c)
    res.append(r)
    print(c, r['kind'], 'rho=%.3f lam0=%.2f Hemp=%.2f gap=%.3f (split %.2f inert %.3f) ok=%s %s' % (
        r['rho'], r['lam0'], r['H_emp'], r['gap_emp'], r['gap_split'], r['gap_inert'], r['ok_thm1'], r['ok_thm2']), flush=True)
json.dump(res, open('../data/toy_results.json', 'w'))

import json, math, csv
import numpy as np
from lemos import log2_secret_space
rng = np.random.default_rng(11)

def hw(x): return bin(x & 0xFF).count('1')

def postmax(eta, sig):
    """E[max posterior] = (1/m) * sum over distinct means of the Gaussian mass inside its Voronoi cell."""
    mus = sorted(set(hw(s % 256) for s in range(-eta, eta + 1)))
    m = 2 * eta + 1
    Phi = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))
    tot = 0.0
    for i, mu in enumerate(mus):
        lo = -math.inf if i == 0 else (mus[i - 1] + mu) / 2
        hi = math.inf if i == len(mus) - 1 else (mus[i + 1] + mu) / 2
        tot += Phi((hi - mu) / sig) - Phi((lo - mu) / sig) if hi != math.inf and lo != -math.inf else (
            (1 - Phi((lo - mu) / sig)) if hi == math.inf and lo != -math.inf else
            (Phi((hi - mu) / sig) if lo == -math.inf and hi != math.inf else 1.0))
    return tot / m

def h_bits(eta, sig):
    return -math.log2(postmax(eta, sig))

def mc_accuracy(eta, sig, N):
    s = rng.integers(-eta, eta + 1, size=N)
    mu_tab = np.array([hw(v % 256) for v in range(-eta, eta + 1)], dtype=float)
    obs = mu_tab[s + eta] + rng.normal(0, sig, size=N)
    # MAP among values; ties (equal means) broken uniformly at random
    d = np.abs(obs[:, None] - mu_tab[None, :])
    best = d.min(axis=1)
    cand = (d <= best[:, None] + 1e-12)
    # random pick among candidates
    r = rng.random(cand.shape) * cand
    guess = r.argmax(axis=1)
    return float(np.mean(guess == (s + eta)))

# validation (box plot data)
val = []
for eta in (1, 3):
    for sig in (0.25, 0.5, 1.0, 2.0):
        th = postmax(eta, sig)
        for b in range(25):
            val.append(dict(eta=eta, sigma=sig, batch=b, acc=mc_accuracy(eta, sig, 20000), theory=th))
import pandas as pd
pd.DataFrame(val).to_csv('../data/side_validation.csv', index=False)
print(pd.DataFrame(val).groupby(['eta', 'sigma']).agg(acc=('acc', 'mean'), sd=('acc', 'std'), th=('theory', 'first')).round(4))

# trace budget
prof = json.load(open('../data/profiles.json'))
kappa = 128
rows = []
for name in ('P1', 'P2', 'P3', 'P4'):
    p = prof[name]
    n, k, l, q, eta = p['n'], p['k'], p['l'], p['q'], p['eta']
    need = n * k * math.log2(q) + kappa            # bits of residual min-entropy required in the secret
    for sig in (0.5, 1, 2, 4, 8, 16, 32):
        # largest T with n*l*h(sig/sqrt(T)) >= need  (h decreases with T)
        def ok(T): return n * l * h_bits(eta, sig / math.sqrt(T)) >= need
        if not ok(1):
            Tstar = 0
        else:
            lo, hi = 1, 2
            while ok(hi) and hi < 1e9: lo, hi = hi, hi * 2
            if hi >= 1e9: Tstar = float('inf')
            else:
                for _ in range(30):
                    mid = math.sqrt(lo * hi)
                    if ok(mid): lo = mid
                    else: hi = mid
                Tstar = lo
        rows.append(dict(profile=name, eta=eta, sigma=sig, Tstar=Tstar, need=need, h1=h_bits(eta, sig)))
        print(name, sig, round(Tstar, 2) if Tstar != float('inf') else 'inf', round(need))
pd.DataFrame(rows).to_csv('../data/side_Tstar.csv', index=False)

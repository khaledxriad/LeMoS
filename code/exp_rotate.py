"""Simulate Algorithm 2 (leakage-budget monitor) for profile P2/P4 under Poisson trace exposure."""
import json, math
import numpy as np, pandas as pd
from side_lib import h_bits
rng = np.random.default_rng(99)
prof = json.load(open('../data/profiles.json'))
KAPPA = 128; DAYS = 30; RUNS = 100
rows = []; per_run = []
for name in ('P2', 'P4'):
    p = prof[name]; n, k, l, q, eta = p['n'], p['k'], p['l'], p['q'], p['eta']
    lam0 = n * l * math.log2(2 * eta + 1) - n * k * math.log2(q)
    lam_max = lam0 - KAPPA - math.log2(1)       # T = 1 key per epoch
    full = n * l * math.log2(2 * eta + 1)
    for sigma in (8, 16, 32, 64):
        # B after t traces of the same key
        def B(t): return 0.0 if t == 0 else full - n * l * h_bits(eta, sigma / math.sqrt(t))
        cap = 0
        while B(cap + 1) < lam_max and cap < 100000: cap += 1      # traces a key can absorb
        for rate in (0.1, 1, 10, 100):
            rots = []
            for r in range(RUNS):
                events = rng.poisson(rate * 24 * DAYS)
                # Algorithm 2: an event that would push B to lam_max is served by a fresh key after rotation
                rotations = events // (cap if cap > 0 else 1) if cap > 0 else events
                rots.append(rotations / DAYS)
                per_run.append(dict(profile=name, sigma=sigma, rate=rate, run=r, rot_per_day=rotations / DAYS))
            rows.append(dict(profile=name, sigma=sigma, rate=rate, cap=cap, mean_rot_day=np.mean(rots), sd=np.std(rots),
                             bytes_day=np.mean(rots) * (p['pk'] + p['sk'])))
df = pd.DataFrame(rows); df.to_csv('../data/rotation_summary.csv', index=False)
pd.DataFrame(per_run).to_csv('../data/rotation_runs.csv', index=False)
print(df.round(2).to_string())

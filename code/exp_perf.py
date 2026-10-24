import time, hashlib, os, json, csv, math
import numpy as np
from lemos import *
rng = np.random.default_rng(7)
profiles = {}
for name, sec, lam in [("P0", 128, 0), ("P1", 128, 512), ("P2", 128, 2048), ("P3", 192, 2048), ("P4", 256, 4096)]:
    p = select_params(sec, lam)
    profiles[name] = (sec, lam, p)
    print(name, p)
json.dump({k: v[2] | {"sec_target": v[0], "lam_target": v[1]} for k, v in profiles.items()}, open('../data/profiles.json', 'w'))

REPS = 300
rows = []
for name, (sec, lam, p) in profiles.items():
    sch = LeMoS(p['n'], p['q'], p['k'], p['l'], p['eta'])
    # warm up
    for _ in range(10):
        pk, sk = sch.keygen(rng); sch.verify(pk, sk)
    for r in range(REPS):
        t0 = time.perf_counter(); pk, sk = sch.keygen(rng); t1 = time.perf_counter()
        ok = sch.verify(pk, sk); t2 = time.perf_counter()
        assert ok
        A = expand_A(sch.ring, pk[0], sch.k, sch.l)
        t3 = time.perf_counter()
        rows.append((name, 'KeyGen', (t1 - t0) * 1e6)); rows.append((name, 'Verify', (t2 - t1) * 1e6))
        rows.append((name, 'ExpandA', (t3 - t2) * 1e6))
# Lamport baseline (SHA3-256)
def lam_keygen():
    sk = [[os.urandom(32) for _ in range(256)] for _ in range(2)]
    pk = [[hashlib.sha3_256(x).digest() for x in row] for row in sk]
    return pk, sk
def lam_token(sk, ch):
    bits = [(ch[i // 8] >> (i % 8)) & 1 for i in range(256)]
    return [sk[b][i] for i, b in enumerate(bits)]
def lam_verify(pk, tok, ch):
    bits = [(ch[i // 8] >> (i % 8)) & 1 for i in range(256)]
    return all(hashlib.sha3_256(tok[i]).digest() == pk[b][i] for i, b in enumerate(bits))
for r in range(REPS):
    t0 = time.perf_counter(); pk, sk = lam_keygen(); t1 = time.perf_counter()
    ch = os.urandom(32); tok = lam_token(sk, ch)
    t2 = time.perf_counter(); ok = lam_verify(pk, tok, ch); t3 = time.perf_counter()
    assert ok
    rows.append(('Lamport', 'KeyGen', (t1 - t0) * 1e6)); rows.append(('Lamport', 'Verify', (t3 - t2) * 1e6))
with open('../data/perf_raw.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['scheme', 'op', 'usec']); w.writerows(rows)
import pandas as pd
df = pd.DataFrame(rows, columns=['scheme', 'op', 'usec'])
print(df.groupby(['scheme', 'op']).usec.agg(['median', 'mean', 'std', lambda x: np.percentile(x, 95)]).round(1))

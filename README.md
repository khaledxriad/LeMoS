# LeMoS: Leakage-Bounded Module-SIS Credentials with Tight Bounds

Code, data, figure sources and LaTeX sources for the paper

> Khaled Riad, "LeMoS: Leakage-Bounded Module-SIS Credentials with Tight Bounds," manuscript, 2026.

The paper proves how many leaked bits a module-lattice key pair `(A, t = A s mod q)` can absorb,
shows that the bound is tight in the key-hiding sense, extends it to noisy side-channel traces and
many keys, and packages it as a one-time credential scheme (LeMoS) with a device-side
leakage-budget monitor and a parameter-selection routine.

## Repository layout

| Path | Content |
|---|---|
| `code/lemos.py` | Reference implementation: negacyclic NTT, ExpandA (SHAKE-128), KeyGen / Token / Verify, leakage-margin formulas, MSIS core-SVP estimator, parameter selection (Algorithm 3) |
| `code/side_lib.py` | Hamming-weight / Gaussian side-channel model and the closed-form per-coefficient accuracy |
| `code/exp_*.py` | Experiments (toy-ring enumeration, parameter profiles, latency and sizes, side-channel trace budget, rotation planning, ring-dimension sweep) |
| `data/` | All CSV / JSON files behind every table and chart |
| `figs/` | All figures as PDF |
| `tests/test_lemos.py` | Smoke tests |

## Requirements

* Python 3.10 or newer (developed with Python 3.13, NumPy 2.5, SciPy 1.18, pandas 3.0, Matplotlib 3.11): `pip install -r requirements.txt`

All experiments use fixed seeds (7, 11, 20261024). Experiments must run in the order given by
`make experiments`, because `exp_perf.py` writes `data/profiles.json`, which the side-channel
and rotation experiments read.


## Notes on interpretation

* Timings are measured in Python on a 2 vCPU server. They rank the schemes but do not predict
  microcontroller performance.
* The side-channel study uses a Hamming-weight model with independent Gaussian noise, not
  measured traces.
* MSIS strength is a core-SVP estimate that ignores ring structure; it is a planning number.

## License and citation

Code: MIT License (see `LICENSE`). Please cite the paper; see `CITATION.cff`.

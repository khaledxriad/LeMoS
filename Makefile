.PHONY: help test figs experiments
# Usage: make help
PY ?= python3

help:
	@echo "make test        - smoke tests of the reference implementation"
	@echo "make figs        - regenerate the matplotlib figures from data/*.csv"
	@echo "make experiments - rerun every experiment and overwrite data/ (slow)"

test:
	$(PY) tests/test_lemos.py

figs:
	cd code && $(PY) make_figs.py

experiments:
	cd code && $(PY) exp_toy.py && $(PY) exp_params.py && $(PY) exp_perf.py && $(PY) exp_side.py && $(PY) exp_rotate.py && $(PY) exp_nsweep.py

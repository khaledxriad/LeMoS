.PHONY: help test figs tikz bbl editions paper experiments
# Usage: make help
PY ?= python3

help:
	@echo "make test        - smoke tests of the reference implementation"
	@echo "make figs        - regenerate the matplotlib figures from data/*.csv"
	@echo "make tikz        - regenerate pgfplots sources and render all TikZ figures to figs/*.pdf"
	@echo "make bbl         - regenerate paper/tikz_edition/IEEE-Article.bbl from references.bib"
	@echo "make editions    - regenerate paper/pdf_edition from paper/tikz_edition"
	@echo "make paper       - compile both editions (needs pdflatex)"
	@echo "make experiments - rerun every experiment and overwrite data/ (slow)"

test:
	$(PY) tests/test_lemos.py

figs:
	cd code && $(PY) make_figs.py

tikz:
	cd code && $(PY) make_tikz_plots.py
	sh code/build_tikz_pdfs.sh

bbl:
	cd code && $(PY) make_bbl.py

editions: bbl
	$(PY) code/build_editions.py

paper: editions
	cd paper/tikz_edition && pdflatex -interaction=nonstopmode IEEE-Article.tex && pdflatex -interaction=nonstopmode IEEE-Article.tex
	cd paper/pdf_edition && pdflatex -interaction=nonstopmode IEEE-Article.tex && pdflatex -interaction=nonstopmode IEEE-Article.tex

experiments:
	cd code && $(PY) exp_toy.py && $(PY) exp_params.py && $(PY) exp_perf.py && $(PY) exp_side.py && $(PY) exp_rotate.py && $(PY) exp_nsweep.py

# One command per artefact, and one command for all of them.
#
# The plan's slice 6 asks for a validation report regenerable from a single
# command. That is `make report`. The rest are here because the same rule
# applies to everything else the project claims: a figure or a table nobody can
# reproduce is a figure nobody can check.
#
# PY is the project venv, which carries the pinned oracles (disba) and the
# analysis stack. Create it with `make venv`.

PY      := py/.venv/bin/python
PIP     := py/.venv/bin/pip
OUT     := out
GOTEST  := go test

.PHONY: all test vet report report-short hierarchy sobol sweep v5 dispersion venv clean

all: vet test report

# --- the suite -------------------------------------------------------------

vet:
	go vet ./...

test:
	$(GOTEST) ./...

# Everything except the grid runs, which are minutes each.
test-short:
	$(GOTEST) -short ./...

# --- the validation report -------------------------------------------------

# The full report runs V5's grid comparisons, so it takes a few minutes.
report:
	$(PY) py/analysis/validation_report.py

report-short:
	$(PY) py/analysis/validation_report.py --short

# --- the tables and figures ------------------------------------------------

$(OUT):
	mkdir -p $(OUT)

# What each modelling level costs, and what it costs you.
hierarchy: | $(OUT)
	go run ./cmd/geohier -model testdata/dispersion/soft_over_stiff.json \
		-ranges 2,5,10,20 -band 120 -o $(OUT)/hierarchy.csv | tee $(OUT)/hierarchy.txt

# V5: the wavenumber solve against the time-domain grid, with the figure.
v5: | $(OUT)
	go run ./cmd/geofdtd -model testdata/dispersion/soft_over_stiff.json \
		-ranges 3,6,10 -compare -ppw 16 -o $(OUT)/v5.csv
	$(PY) py/analysis/plot_v5.py $(OUT)/v5.csv -o $(OUT)/v5.png

# One axis at a time: the gradient at the reference point.
sweep: | $(OUT)
	go run ./cmd/geosweep -o $(OUT)/sweep.csv
	$(PY) py/analysis/plot_sweep.py $(OUT)/sweep.csv -o $(OUT)/sweep.png

# The joint decomposition. N=512 over ten axes is 6144 runs, about a quarter of
# an hour; the estimator is checked against Ishigami's exact indices first,
# because there is no point spending that on an estimator nobody has tested.
SOBOL_N ?= 512
sobol: | $(OUT)
	$(PY) py/analysis/sobol.py selftest
	go run ./cmd/geosweep -axes $(OUT)/axes.json
	$(PY) py/analysis/sobol.py design -a $(OUT)/axes.json -n $(SOBOL_N) -o $(OUT)/design.csv
	go run ./cmd/geosweep -design $(OUT)/design.csv -o $(OUT)/results.csv
	$(PY) py/analysis/sobol.py analyse -r $(OUT)/results.csv -o $(OUT)/sobol | tee $(OUT)/sobol.txt

# Dispersion curves against the disba golden files.
dispersion: | $(OUT)
	go run ./cmd/geodisp -model testdata/dispersion/three_layer_site.json -o $(OUT)/dispersion.csv
	$(PY) py/analysis/plot_dispersion.py $(OUT)/dispersion.csv \
		-g testdata/dispersion/three_layer_site.json -o $(OUT)/dispersion.png

# --- housekeeping ----------------------------------------------------------

venv:
	python3 -m venv py/.venv
	$(PIP) install -r py/requirements.txt

clean:
	rm -rf $(OUT)

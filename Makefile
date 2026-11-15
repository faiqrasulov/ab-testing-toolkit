PYTHON ?= python
MEMCAP ?=

.PHONY: install data run test lint clean

install:
	$(PYTHON) -m pip install -e ".[dev]"

# Regenerate the simulated example CSVs in examples/ (no external data is used).
data:
	$(MEMCAP) $(PYTHON) examples/make_example_data.py --out examples

# Full reproduction: all simulation studies -> reports/metrics.json + reports/figures/*.png,
# then the CLI report for the example experiment.
run: data
	$(MEMCAP) $(PYTHON) -m abkit.studies --out reports
	$(PYTHON) -m abkit.cli analyze examples/simulated_ab_test.csv --group variant \
		--metric revenue --pre-metric pre_revenue --control control

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

clean:
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info

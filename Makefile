# Convenience wrappers. On Windows without `make`, run the underlying
# `python -m vulnprio.cli ...` commands directly (shown in each recipe).
.PHONY: setup lint test data features baselines evaluate all clean

setup:
	python -m pip install -r requirements.txt && python -m pip install -e .

lint:
	ruff check src tests
	ruff format --check src tests

test:
	pytest -m "not network"

data:
	python -m vulnprio.cli fetch --config configs/default.yaml
	python -m vulnprio.cli build-dataset --config configs/default.yaml

features:
	python -m vulnprio.cli build-features --config configs/default.yaml

baselines:
	python -m vulnprio.cli evaluate --config configs/experiments/baseline_cvss.yaml

evaluate:
	python -m vulnprio.cli evaluate --config configs/default.yaml

all: data features baselines

clean:
	rm -rf data/interim/* data/processed/* reports/figures/* reports/results/*

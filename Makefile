.PHONY: install-dev test lint prepare-eval validate-eval

install-dev:
	python -m pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .

prepare-eval:
	afri-reasoning prepare-eval --config configs/baseline.yaml

validate-eval:
	afri-reasoning validate-eval --config configs/baseline.yaml

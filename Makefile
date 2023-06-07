default:
	pip install .

install-dev:
	pip install -e .[dev]

.PHONY: tests
tests:
	PY_MAJOR_VERSION=py`python -c 'import sys; print(sys.version_info[0])'` pytest -v --cov=cellulus --cov-config=.coveragerc tests
	pre-commit run --all-files

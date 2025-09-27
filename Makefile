# ===== Config =====
SHELL := /bin/bash
.ONESHELL:
.DEFAULT_GOAL := help

# venv
VENV ?= venv
PYTHON := $(VENV)/bin/python
PIP    := $(VENV)/bin/pip

# project files
MAIN ?= main.py
REQ  ?= requirements.txt

# defaults (override with: make pipeline DATA=... MODEL=...)
DATA  ?= gaming_100mb.csv
MODEL ?= models/churn_model.joblib

# ensure tests can import modules from repo root
TEST_ENV := PYTHONPATH=.

# ===== Phony targets =====
.PHONY: help all venv install lint flake8 pylint mypy bandit format test \
        prepare_data train_model evaluate_model save_model load_model pipeline clean clean-pyc

# ===== Help =====
help:
	@echo "Usage: make <target>"
	@echo
	@echo "Setup:"
	@echo "  make install           Create venv and install requirements"
	@echo
	@echo "Quality:"
	@echo "  make format            Autoformat code with black (line length = 100)"
	@echo "  make lint              Run flake8, pylint, mypy, bandit"
	@echo
	@echo "Pipeline steps (DATA=$(DATA), MODEL=$(MODEL)):"
	@echo "  make prepare_data      -> python $(MAIN) --prepare_data --data \$$DATA"
	@echo "  make train_model       -> python $(MAIN) --train_model  --data \$$DATA"
	@echo "  make evaluate_model    -> python $(MAIN) --evaluate_model --data \$$DATA --model \$$MODEL"
	@echo "  make save_model        -> python $(MAIN) --save_model --model \$$MODEL"
	@echo "  make load_model        -> python $(MAIN) --load_model --model \$$MODEL"
	@echo "  make pipeline          Prepare + Train + Evaluate + Save"
	@echo
	@echo "Tests:"
	@echo "  make test              Run pytest if tests/ exists (skips otherwise)"
	@echo
	@echo "Cleanup:"
	@echo "  make clean             Remove venv, model artifacts, __pycache__"
	@echo "  make clean-pyc         Remove *.pyc/*.pyo"

# ===== Setup =====
all: install

venv: $(VENV)/bin/activate
$(VENV)/bin/activate:
	python3 -m venv $(VENV)
	@echo "Venv created at $(VENV)"

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r $(REQ)

# ===== Quality =====
format: venv
	$(PIP) install black
	$(VENV)/bin/black --line-length 100 $(MAIN) model_pipeline.py

lint: flake8 pylint mypy bandit

flake8: venv
	$(PIP) install flake8
	$(VENV)/bin/flake8 $(MAIN) model_pipeline.py

pylint: venv
	$(PIP) install pylint
	$(VENV)/bin/pylint $(MAIN) model_pipeline.py

mypy: venv
	$(PIP) install mypy
	$(VENV)/bin/mypy $(MAIN) model_pipeline.py

bandit: venv
	$(PIP) install bandit
	$(VENV)/bin/bandit -r model_pipeline.py

# ===== Tests & Quality =====
test: format flake8 pylint mypy bandit
	@echo "All quality checks completed."


# ===== Pipeline steps =====
prepare_data: venv
	$(PYTHON) $(MAIN) --prepare_data --data $(DATA)

train_model: venv
	$(PYTHON) $(MAIN) --train_model --data $(DATA)

evaluate_model: venv
	$(PYTHON) $(MAIN) --evaluate_model --data $(DATA) --model $(MODEL)

save_model: venv
	mkdir -p $(dir $(MODEL))
	$(PYTHON) $(MAIN) --save_model --model $(MODEL)

load_model: venv
	$(PYTHON) $(MAIN) --load_model --model $(MODEL)

# One-shot pipeline: prepare -> train -> evaluate -> save
pipeline: venv
	mkdir -p $(dir $(MODEL))
	$(PYTHON) $(MAIN) --prepare_data --train_model --evaluate_model --save_model \
	                  --data $(DATA) --model $(MODEL)

# ===== Cleanup =====
clean:
	-rm -rf $(VENV)
	-rm -f $(MODEL)
	-find . -name "__pycache__" -type d -exec rm -rf {} +
	@echo "Cleaned venv, model artifacts, __pycache__."

clean-pyc:
	-find . -name "*.pyc" -o -name "*.pyo" -o -name "*~" -delete

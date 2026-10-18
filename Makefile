# MOSAIC / BioNexus MLOps Makefile
.PHONY: setup lint test repro serve up down help

PYTHON ?= python

help:
	@echo "MOSAIC / BioNexus MLOps commands:"
	@echo "  make setup   - Install package in editable mode with dev dependencies"
	@echo "  make lint    - Run ruff lint and format check"
	@echo "  make test    - Run pytest test suite"
	@echo "  make repro   - Run DVC pipeline reproduction"
	@echo "  make serve   - Start serving API locally"
	@echo "  make up      - Start Docker Compose services (mlflow, api, train)"
	@echo "  make down    - Stop Docker Compose services"

setup:
	$(PYTHON) -m pip install -e .[dev]

lint:
	ruff check .
	ruff format --check .

test:
	pytest tests/ -v

repro:
	dvc repro

serve:
	uvicorn bio_nexus.serving.api:app --host 0.0.0.0 --port 8000 --reload

up:
	docker compose up -d

down:
	docker compose down

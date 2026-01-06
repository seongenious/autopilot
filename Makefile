# Autopilot Makefile
# Common commands for development and training

.PHONY: help install install-dev train eval test lint format clean

# Default target
help:
	@echo "Autopilot - Autonomous Driving Perception Framework"
	@echo ""
	@echo "Available commands:"
	@echo "  make install      Install package in development mode"
	@echo "  make install-dev  Install with development dependencies"
	@echo "  make train        Run training with default config"
	@echo "  make eval         Run evaluation"
	@echo "  make test         Run unit tests"
	@echo "  make lint         Run linters (flake8, mypy)"
	@echo "  make format       Format code (black, isort)"
	@echo "  make clean        Clean build artifacts"
	@echo ""
	@echo "Training examples:"
	@echo "  make train ARGS='dataset=waymo'"
	@echo "  make train ARGS='training.max_epochs=12 hardware.gpus=2'"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

install-nuscenes:
	pip install -e ".[nuscenes]"

install-waymo:
	pip install -e ".[waymo]"

install-all:
	pip install -e ".[all]"

# Training
ARGS ?=
train:
	python scripts/train.py $(ARGS)

train-nuscenes:
	python scripts/train.py dataset=nuscenes $(ARGS)

train-waymo:
	python scripts/train.py dataset=waymo $(ARGS)

# Evaluation
CHECKPOINT ?=
eval:
	python scripts/eval.py checkpoint=$(CHECKPOINT) $(ARGS)

# Inference
INPUT ?=
infer:
	python scripts/inference.py checkpoint=$(CHECKPOINT) input=$(INPUT) $(ARGS)

# Testing
test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=autopilot --cov-report=html

# Code quality
lint:
	flake8 autopilot/ scripts/ tests/
	mypy autopilot/ --ignore-missing-imports

format:
	black autopilot/ scripts/ tests/
	isort autopilot/ scripts/ tests/

format-check:
	black --check autopilot/ scripts/ tests/
	isort --check-only autopilot/ scripts/ tests/

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

clean-outputs:
	rm -rf outputs/
	rm -rf checkpoints/
	rm -rf logs/

# Data preparation
convert-nuscenes:
	python tools/data_converter/nuscenes_converter.py \
		--data-root /data/nuscenes \
		--out-dir /data/nuscenes \
		--version v1.0-trainval

convert-waymo:
	python tools/data_converter/waymo_converter.py \
		--data-root /data/waymo \
		--out-dir /data/waymo

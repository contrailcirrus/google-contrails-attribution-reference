# Variables
VENV_DIR := .venv
PYTHON := python3
PIP := $(VENV_DIR)/bin/pip
KERNEL_NAME := pycontrails-env
DISPLAY_NAME := Python (Pycontrails)

.PHONY: help setup kernel clean

help:
	@echo "Available commands:"
	@echo "  make setup  - Create venv and install dependencies from requirements.txt"
	@echo "  make kernel - Register the virtual environment as a Jupyter kernel"
	@echo "  make test   - Run unit tests using pytest"
	@echo "  make clean  - Remove the virtual environment and the Jupyter kernel"

setup:
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV_DIR)
	@echo "Upgrading pip..."
	$(PIP) install --upgrade pip
	@echo "Installing dependencies from requirements.txt..."
	$(PIP) install -r requirements.txt --extra-index-url https://pypi.org/simple
	@echo "Installing project in editable mode..."
	$(PIP) install -e . --no-build-isolation
	@echo "Setup complete."

kernel: setup
	@echo "Installing Jupyter kernel..."
	$(VENV_DIR)/bin/python -m ipykernel install --user --name=$(KERNEL_NAME) --display-name="$(DISPLAY_NAME)"
	@echo "Kernel '$(DISPLAY_NAME)' is ready!"

clean:
	@echo "Removing Jupyter kernel..."
	-jupyter kernelspec uninstall $(KERNEL_NAME) -y
	@echo "Removing virtual environment..."
	rm -rf $(VENV_DIR)
	@echo "Clean up complete."

test: setup
	@echo "Running unit tests..."
	$(VENV_DIR)/bin/pytest tests/ -v
	@echo "Tests complete."
	

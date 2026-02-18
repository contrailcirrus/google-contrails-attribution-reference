# Variables
VENV_DIR := .venv
PYTHON := python3
PIP := $(VENV_DIR)/bin/pip
KERNEL_NAME := pycontrails-env
DISPLAY_NAME := Python (Pycontrails)

.PHONY: help setup kernel clean

# Default target
help:
	@echo "Available commands:"
	@echo "  make setup  - Create venv and install dependencies from requirements.txt"
	@echo "  make kernel - Register the virtual environment as a Jupyter kernel"
	@echo "  make clean  - Remove the virtual environment and the Jupyter kernel"

# 1. Create virtual environment and install dependencies
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

# 2. Register the environment as a runtime kernel
kernel: setup
	@echo "Installing Jupyter kernel..."
	$(VENV_DIR)/bin/python -m ipykernel install --user --name=$(KERNEL_NAME) --display-name="$(DISPLAY_NAME)"
	@echo "Kernel '$(DISPLAY_NAME)' is ready!"

# 3. Clean up
clean:
	@echo "Removing Jupyter kernel..."
	-jupyter kernelspec uninstall $(KERNEL_NAME) -y
	@echo "Removing virtual environment..."
	rm -rf $(VENV_DIR)
	@echo "Clean up complete."

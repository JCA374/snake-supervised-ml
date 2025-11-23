#!/bin/bash
# Script to run tests with proper virtual environment

# Activate virtual environment
source .venv/bin/activate

# Run pytest
python -m pytest tests/ -v

# Deactivate
deactivate

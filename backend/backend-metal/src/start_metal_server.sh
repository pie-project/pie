#!/bin/bash
# Metal backend server launcher with virtual environment activation
# This ensures the Metal server starts with the correct Python environment

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

# Activate the Python virtual environment
source "$SCRIPT_DIR/../../backend-python/.venv/bin/activate"

# Run the Metal backend server with all passed arguments
exec python "$SCRIPT_DIR/server.py" "$@"
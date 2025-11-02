#! /bin/bash

set -e

# Run format checks.
echo "Running format checks on Python backend..."
./tests/format/backend-python-format.sh
echo "Done."

# Run pylint checks.
echo "Running Pylint checks on Python backend..."
./tests/format/backend-python-pylint.sh
echo "Done."

# Run pyright checks.
echo "Running Pyright checks on Python backend..."
./tests/format/backend-python-pyright.sh
echo "Done."

echo "All checks passed."

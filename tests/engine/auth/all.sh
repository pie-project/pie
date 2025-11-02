#!/bin/bash

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run key algorithm test
echo "Running key algorithm test..."
"${SCRIPT_DIR}/key_algorithm.sh"
echo "Done."

echo "All authentication tests passed."

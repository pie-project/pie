#!/bin/bash

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run key algorithm test
echo "Running key algorithm test..."
"${SCRIPT_DIR}/key_algorithm.sh"

# Run disable authentication test
echo "Running disable authentication test..."
"${SCRIPT_DIR}/disable_auth.sh"

echo "All authentication tests passed."

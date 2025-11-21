#!/bin/bash

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run standalone link test
echo "Running standalone link test..."
"${SCRIPT_DIR}/standalone.sh"

echo "All link tests passed."


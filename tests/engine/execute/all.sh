#!/bin/bash

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run submit attached test
echo "Running submit attached test..."
"${SCRIPT_DIR}/submit_attached.sh"

# Run submit detached test
echo "Running submit detached test..."
"${SCRIPT_DIR}/submit_detached.sh"

# Run abort test
echo "Running abort test..."
"${SCRIPT_DIR}/abort.sh"

# Run attach test
echo "Running attach test..."
"${SCRIPT_DIR}/attach.sh"

echo "All execution tests passed."


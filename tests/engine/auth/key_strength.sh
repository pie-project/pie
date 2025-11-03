#!/bin/bash

# SSH Key Strength Validation Test
#
# This test script verifies that the pie engine correctly rejects insecure SSH keys
# that do not meet minimum strength requirements. It tests the following:
#   - RSA 1536 bits (should be rejected as insecure)
#
# Test Procedure:
#   1. Generates an insecure RSA 1536 bit key pair
#   2. Attempts to add the public key to the pie authorized users list for user "pie-test"
#   3. Verifies that the add operation fails (key should be rejected)
#   4. Cleans up all generated files and auth keys
#
# The script uses randomized file names for both SSH keys and configuration files
# to prevent naming collisions when multiple test instances run concurrently.
#
# All cleanup (removing keys, config files) happens automatically via an EXIT trap,
# ensuring no test artifacts remain even if the script fails or is interrupted.

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source utility functions
source "${SCRIPT_DIR}/utils.sh"

# Cleanup function
cleanup() {
    # Remove authorized keys from the engine
    yes 2>/dev/null | pie auth remove "pie-test" >/dev/null 2>&1 || true
    
    # Clean up generated key
    if [ -n "$RSA_WEAK_KEY" ]; then
        rm -f "$RSA_WEAK_KEY"
        rm -f "${RSA_WEAK_KEY}.pub"
    fi
}

# Set trap to cleanup on exit
trap cleanup EXIT

echo "Generating insecure RSA 1536 bit key..."

# Generate insecure RSA 1536 bit key
RSA_WEAK_KEY=$(generate_unique_key_path "rsa-1536")
ssh-keygen -t rsa -b 1536 -f "$RSA_WEAK_KEY" -N "" -C "test-rsa-1536" -q

echo "Attempting to add insecure key to authorized users list..."

# Try to add the key - this should fail
if pie auth add "pie-test" "test-rsa-1536" < "${RSA_WEAK_KEY}.pub" >/dev/null 2>&1; then
    echo "Error: Insecure RSA 1536 bit key was unexpectedly accepted"
    exit 1
fi

echo "Test passed: Insecure RSA 1536 bit key was correctly rejected"

# Cleanup will be called automatically by the trap


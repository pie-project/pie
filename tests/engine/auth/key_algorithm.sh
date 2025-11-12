#!/bin/bash

# SSH Key Algorithm Compatibility Test
#
# This test script verifies that the pie engine correctly supports authentication
# using SSH keys with different cryptographic algorithms. It tests the following
# key types:
#   - RSA 4096 bits
#   - ED25519
#   - ECDSA P-256 (NIST curve)
#   - ECDSA P-384 (NIST curve)
#
# Test Procedure:
#   1. Generates SSH key pairs for each algorithm type
#   2. Adds each public key to the pie authorized users list with a randomized username
#   3. Initializes pie engine and pie-cli configurations with unique random names
#   4. Starts the pie engine in the background
#   5. Tests each key by configuring pie-cli to use it and executing a ping command
#   6. Cleans up all generated files, auth keys, and background processes
#
# The script uses randomized file names for both SSH keys and configuration files
# to prevent naming collisions when multiple test instances run concurrently.
#
# All cleanup (removing keys, config files, and stopping the server) happens
# automatically via an EXIT trap, ensuring no test artifacts remain even if the
# script fails or is interrupted.

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source utility functions
source "${SCRIPT_DIR}/../utils.sh"

# Generate randomized username
TEST_USERNAME="pie-test-$(generate_random_string)"

# Array to track generated key files for cleanup
GENERATED_KEYS=()
PIE_SERVE_PID=""

# Generate unique config file paths
PIE_CONFIG=$(generate_unique_config_path "pie-test-config")
PIE_CLI_CONFIG=$(generate_unique_config_path "pie-cli-test-config")

# Cleanup function
cleanup() {
    # Kill the background pie serve process
    if [ -n "$PIE_SERVE_PID" ]; then
        kill "$PIE_SERVE_PID" 2>/dev/null || true
        wait "$PIE_SERVE_PID" 2>/dev/null || true
    fi
    
    # Remove authorized keys from the engine
    yes | pie auth remove "$TEST_USERNAME" || true
    
    # Clean up generated keys
    for key in "${GENERATED_KEYS[@]}"; do
        if [ -f "$key" ]; then
            rm -f "$key"
        fi
        if [ -f "${key}.pub" ]; then
            rm -f "${key}.pub"
        fi
    done

    # Clean up test config files
    rm -f "$PIE_CONFIG"
    rm -f "$PIE_CLI_CONFIG"
}

# Set trap to cleanup on exit
trap cleanup EXIT

echo "Generating keys and adding to authorized users list..."

# Key 1: RSA 4096 bits
RSA_KEY=$(generate_unique_key_path "rsa")
ssh-keygen -t rsa -b 4096 -f "$RSA_KEY" -N "" -C "test-rsa-4096" -q
GENERATED_KEYS+=("$RSA_KEY")
cat "${RSA_KEY}.pub" | pie auth add "$TEST_USERNAME" "test-rsa-4096"

# Key 2: ED25519
ED25519_KEY=$(generate_unique_key_path "ed25519")
ssh-keygen -t ed25519 -f "$ED25519_KEY" -N "" -C "test-ed25519" -q
GENERATED_KEYS+=("$ED25519_KEY")
cat "${ED25519_KEY}.pub" | pie auth add "$TEST_USERNAME" "test-ed25519"

# Key 3: ECDSA P-256
ECDSA_256_KEY=$(generate_unique_key_path "ecdsa-p256")
ssh-keygen -t ecdsa -b 256 -f "$ECDSA_256_KEY" -N "" -C "test-ecdsa-p256" -q
GENERATED_KEYS+=("$ECDSA_256_KEY")
cat "${ECDSA_256_KEY}.pub" | pie auth add "$TEST_USERNAME" "test-ecdsa-p256"

# Key 4: ECDSA P-384
ECDSA_384_KEY=$(generate_unique_key_path "ecdsa-p384")
ssh-keygen -t ecdsa -b 384 -f "$ECDSA_384_KEY" -N "" -C "test-ecdsa-p384" -q
GENERATED_KEYS+=("$ECDSA_384_KEY")
cat "${ECDSA_384_KEY}.pub" | pie auth add "$TEST_USERNAME" "test-ecdsa-p384"

echo "All keys generated and added to authorized users list"

# Create config files
pie config init dummy --path "$PIE_CONFIG"
pie-cli config init --path "$PIE_CLI_CONFIG"

# Start the Pie engine and wait for it to be ready
start_pie_engine "$PIE_CONFIG" "PIE_SERVE_PID" || exit 1

pie-cli config update --path "$PIE_CLI_CONFIG" --username "$TEST_USERNAME"

# Test Key 1: RSA 4096 bits
echo "Testing RSA 4096 bits key..."
pie-cli config update --path "$PIE_CLI_CONFIG" --private-key-path "$RSA_KEY"
timeout 10 pie-cli ping --config "$PIE_CLI_CONFIG" || { echo "Error: ping with RSA key failed"; exit 1; }
echo "RSA 4096 bits key test passed"

# Test Key 2: ED25519
echo "Testing ED25519 key..."
pie-cli config update --path "$PIE_CLI_CONFIG" --private-key-path "$ED25519_KEY"
timeout 10 pie-cli ping --config "$PIE_CLI_CONFIG" || { echo "Error: ping with ED25519 key failed"; exit 1; }
echo "ED25519 key test passed"

# Test Key 3: ECDSA P-256
echo "Testing ECDSA P-256 key..."
pie-cli config update --path "$PIE_CLI_CONFIG" --private-key-path "$ECDSA_256_KEY"
timeout 10 pie-cli ping --config "$PIE_CLI_CONFIG" || { echo "Error: ping with ECDSA P-256 key failed"; exit 1; }
echo "ECDSA P-256 key test passed"

# Test Key 4: ECDSA P-384
echo "Testing ECDSA P-384 key..."
pie-cli config update --path "$PIE_CLI_CONFIG" --private-key-path "$ECDSA_384_KEY"
timeout 10 pie-cli ping --config "$PIE_CLI_CONFIG" || { echo "Error: ping with ECDSA P-384 key failed"; exit 1; }
echo "ECDSA P-384 key test passed"

echo "All tests passed"

# Cleanup will be called automatically by the trap

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
#   2. Adds each public key to the pie authorized users list for user "pie-test"
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
source "${SCRIPT_DIR}/utils.sh"

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
    yes | pie auth remove "pie-test" || true
    
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
cat "${RSA_KEY}.pub" | pie auth add "pie-test" "test-rsa-4096"

# Key 2: ED25519
ED25519_KEY=$(generate_unique_key_path "ed25519")
ssh-keygen -t ed25519 -f "$ED25519_KEY" -N "" -C "test-ed25519" -q
GENERATED_KEYS+=("$ED25519_KEY")
cat "${ED25519_KEY}.pub" | pie auth add "pie-test" "test-ed25519"

# Key 3: ECDSA P-256
ECDSA_256_KEY=$(generate_unique_key_path "ecdsa-p256")
ssh-keygen -t ecdsa -b 256 -f "$ECDSA_256_KEY" -N "" -C "test-ecdsa-p256" -q
GENERATED_KEYS+=("$ECDSA_256_KEY")
cat "${ECDSA_256_KEY}.pub" | pie auth add "pie-test" "test-ecdsa-p256"

# Key 4: ECDSA P-384
ECDSA_384_KEY=$(generate_unique_key_path "ecdsa-p384")
ssh-keygen -t ecdsa -b 384 -f "$ECDSA_384_KEY" -N "" -C "test-ecdsa-p384" -q
GENERATED_KEYS+=("$ECDSA_384_KEY")
cat "${ECDSA_384_KEY}.pub" | pie auth add "pie-test" "test-ecdsa-p384"

echo "All keys generated and added to authorized users list"

# Create config files
pie config init dummy --path "$PIE_CONFIG"
pie-cli config init --path "$PIE_CLI_CONFIG"

pie serve --config "$PIE_CONFIG" &
PIE_SERVE_PID=$!

# Wait for the server to be ready
echo "Waiting for pie serve to start..."
sleep 3

# Check if process is still running
if ! kill -0 "$PIE_SERVE_PID" 2>/dev/null; then
    echo "Error: pie serve process died immediately"
    exit 1
fi

# Try to connect with a timeout, retry up to 15 times
MAX_RETRIES=15
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if nc -z 127.0.0.1 8080 2>/dev/null; then
        echo "Server is ready"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "Error: Server failed to become ready after $MAX_RETRIES attempts"
        exit 1
    fi
    sleep 1
done

pie-cli config update --path "$PIE_CLI_CONFIG" --username "pie-test"

# Test Key 1: RSA 4096 bits
pie-cli config update --path "$PIE_CLI_CONFIG" --private-key-path "$RSA_KEY"
timeout 10 pie-cli ping --config "$PIE_CLI_CONFIG" || { echo "Error: ping with RSA key failed"; exit 1; }

# Test Key 2: ED25519
pie-cli config update --path "$PIE_CLI_CONFIG" --private-key-path "$ED25519_KEY"
timeout 10 pie-cli ping --config "$PIE_CLI_CONFIG" || { echo "Error: ping with ED25519 key failed"; exit 1; }

# Test Key 3: ECDSA P-256
pie-cli config update --path "$PIE_CLI_CONFIG" --private-key-path "$ECDSA_256_KEY"
timeout 10 pie-cli ping --config "$PIE_CLI_CONFIG" || { echo "Error: ping with ECDSA P-256 key failed"; exit 1; }

# Test Key 4: ECDSA P-384
pie-cli config update --path "$PIE_CLI_CONFIG" --private-key-path "$ECDSA_384_KEY"
timeout 10 pie-cli ping --config "$PIE_CLI_CONFIG" || { echo "Error: ping with ECDSA P-384 key failed"; exit 1; }

echo "All tests passed"

# Cleanup will be called automatically by the trap

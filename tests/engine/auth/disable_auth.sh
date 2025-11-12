#!/bin/bash

# Disable Auth Test
#
# This test script verifies the pie engine's authentication behavior when
# authentication is enabled or disabled on the client and engine sides in
# various combinations. It uses ED25519 keys for testing.
#
# Test Procedure:
#   1. Generates an ED25519 SSH key pair
#   2. Adds the public key to the pie authorized users list with a randomized username
#   3. Initializes pie engine and pie-cli configurations with unique random names
#   4. Test when both ends enable authentication; ping should succeed
#   5. Test when both ends disable authentication; ping should succeed
#   6. Test when client enables authentication and engine disables authentication; ping should succeed
#   7. Test when client disables authentication and engine enables authentication; ping should fail
#   8. Cleans up all generated files and auth keys
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
    
    # Clean up generated key
    if [ -n "$ED25519_KEY" ]; then
        rm -f "$ED25519_KEY"
        rm -f "${ED25519_KEY}.pub"
    fi

    # Clean up test config files
    rm -f "$PIE_CONFIG"
    rm -f "$PIE_CLI_CONFIG"
}

# Set trap to cleanup on exit
trap cleanup EXIT


### STEP 1: Prepare the environment
echo "Generating key and adding to authorized users list..."

# Generate ED25519 key and add to authorized users list
ED25519_KEY=$(generate_unique_key_path "ed25519")
ssh-keygen -t ed25519 -f "$ED25519_KEY" -N "" -C "test-ed25519" -q
cat "${ED25519_KEY}.pub" | pie auth add "$TEST_USERNAME" "test-ed25519"

echo "Key generated and added to authorized users list"

# Create config files
pie config init dummy --path "$PIE_CONFIG"
pie-cli config init --path "$PIE_CLI_CONFIG"

# Configure client username
pie-cli config update --path "$PIE_CLI_CONFIG" --username "$TEST_USERNAME"

# Configure client private key
pie-cli config update --path "$PIE_CLI_CONFIG" --private-key-path "$ED25519_KEY"


### STEP 2: Test when both ends enable authentication
pie-cli config update --path "$PIE_CLI_CONFIG" --enable-auth true
pie config update --path "$PIE_CONFIG" --enable-auth true

# Start the Pie engine and wait for it to be ready
start_pie_engine "$PIE_CONFIG" "PIE_SERVE_PID" || exit 1

# Test authentication
echo "Testing: Engine auth == true and client auth == true..."
timeout 10 pie-cli ping --config "$PIE_CLI_CONFIG" || { echo "Error: ping with ED25519 key failed"; exit 1; }
echo "Test passed: Engine auth == true and client auth == true"

# Stop the Pie engine
kill -INT "$PIE_SERVE_PID"
wait "$PIE_SERVE_PID" 2>/dev/null || true
PIE_SERVE_PID=""


### STEP 3: Test when both ends disable authentication
pie-cli config update --path "$PIE_CLI_CONFIG" --enable-auth false
pie config update --path "$PIE_CONFIG" --enable-auth false

# Start the Pie engine and wait for it to be ready
start_pie_engine "$PIE_CONFIG" "PIE_SERVE_PID" || exit 1

# Test authentication
echo "Testing: Engine auth == false and client auth == false..."
timeout 10 pie-cli ping --config "$PIE_CLI_CONFIG" || { echo "Error: ping with ED25519 key failed"; exit 1; }
echo "Test passed: Engine auth == false and client auth == false"

# Stop the Pie engine
kill -INT "$PIE_SERVE_PID"
wait "$PIE_SERVE_PID" 2>/dev/null || true
PIE_SERVE_PID=""


### STEP 4: Test when client enables authentication and engine disables authentication
pie-cli config update --path "$PIE_CLI_CONFIG" --enable-auth true
pie config update --path "$PIE_CONFIG" --enable-auth false

# Start the Pie engine and wait for it to be ready
start_pie_engine "$PIE_CONFIG" "PIE_SERVE_PID" || exit 1

# Test authentication
echo "Testing: Engine auth == false and client auth == true..."
timeout 10 pie-cli ping --config "$PIE_CLI_CONFIG" || { echo "Error: ping with ED25519 key failed"; exit 1; }
echo "Test passed: Engine auth == false and client auth == true"

# Stop the Pie engine
kill -INT "$PIE_SERVE_PID"
wait "$PIE_SERVE_PID" 2>/dev/null || true
PIE_SERVE_PID=""


### STEP 5: Test when client disables authentication and engine enables authentication
pie-cli config update --path "$PIE_CLI_CONFIG" --enable-auth false
pie config update --path "$PIE_CONFIG" --enable-auth true

# Start the Pie engine and wait for it to be ready
start_pie_engine "$PIE_CONFIG" "PIE_SERVE_PID" || exit 1

# Test authentication - this should fail because engine requires auth but client disabled it
echo "Testing: Engine auth == true and client auth == false..."
if timeout 10 pie-cli ping --config "$PIE_CLI_CONFIG" >/dev/null 2>&1; then
    echo "Error: ping unexpectedly succeeded when client disables authentication and engine enables authentication"
    exit 1
fi
echo "Test passed: Engine auth == true and client auth == false"

# Stop the Pie engine
kill -INT "$PIE_SERVE_PID"
wait "$PIE_SERVE_PID" 2>/dev/null || true
PIE_SERVE_PID=""

echo "All tests passed"

# Cleanup will be called automatically by the trap

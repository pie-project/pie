#!/bin/bash

# Submit Attached Inferlet Test
#
# This test script verifies that the pie engine correctly handles submitting
# an attached inferlet via the pie-cli submit command.
#
# Usage:
#   ./submit_attached.sh [INFERLET_DIR]
#
# Arguments:
#   INFERLET_DIR - Optional. Path to the directory containing the inferlet.
#                  Defaults to: ../../../test-apps/target/wasm32-wasip2/release/
#
# Test Procedure:
#   1. Generates an ED25519 SSH key pair for authentication
#   2. Adds the public key to the pie authorized users list with a randomized
#      username
#   3. Initializes pie engine and pie-cli configurations with unique random
#      names
#   4. Configures the engine to use the dummy backend
#   5. Starts the pie engine in the background
#   6. Runs three test cases to verify inferlet submission:
#      - Case 1: Submit inferlet with no text input and no delay
#        Verifies that the submit command completes successfully
#      - Case 2: Submit inferlet with random text input and no delay
#        Verifies that the output contains the input text (echo
#        functionality)
#      - Case 3: Submit inferlet with no text input but with 5-second delay
#        Verifies that the command respects the delay (takes at least
#        3 seconds)
#   7. Cleans up all generated files, auth keys, and background processes
#
# The script uses randomized file names for both SSH keys and configuration
# files to prevent naming collisions when multiple test instances run
# concurrently.
#
# All cleanup (removing keys, config files, and stopping the server) happens
# automatically via an EXIT trap, ensuring no test artifacts remain even if
# the script fails or is interrupted.

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse command-line arguments
INFERLET_DIR="${1:-${SCRIPT_DIR}/../../../test-apps/target/wasm32-wasip2/release}"

# Source utility functions
source "${SCRIPT_DIR}/../utils.sh"
source "${SCRIPT_DIR}/utils.sh"

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

echo "Generating ED25519 key and adding to authorized users list..."

# Generate ED25519 key
ED25519_KEY=$(generate_unique_key_path "ed25519")
ssh-keygen -t ed25519 -f "$ED25519_KEY" -N "" -C "test-ed25519" -q
cat "${ED25519_KEY}.pub" | pie auth add "$TEST_USERNAME" "test-ed25519"

echo "Key generated and added to authorized users list"

# Create config files
pie config init dummy --path "$PIE_CONFIG"
pie-cli config init --path "$PIE_CLI_CONFIG"

# Start the Pie engine and wait for it to be ready
start_pie_engine "$PIE_CONFIG" "PIE_SERVE_PID" || exit 1

# Configure pie-cli with username and private key
pie-cli config update --path "$PIE_CLI_CONFIG" --username "$TEST_USERNAME"
pie-cli config update --path "$PIE_CLI_CONFIG" --private-key-path "$ED25519_KEY"

# Configure the inferlet to use
echo "Using inferlet directory: $INFERLET_DIR"
INFERLET="$INFERLET_DIR/echo.wasm"


# Case 1: No text, no delay
echo "Test submitting inferlet (no text, no delay)"
submit_inferlet_attached "$INFERLET" "$PIE_CLI_CONFIG" > /dev/null
echo "Submit test (no text, no delay) passed"


# Case 2: With text, no delay
echo "Test submitting inferlet (with text, no delay)"
INPUT_TEXT="test-$(generate_random_string)"
OUTPUT=$(submit_inferlet_attached "$INFERLET" "$PIE_CLI_CONFIG" --text "$INPUT_TEXT")
echo "$OUTPUT" | grep -q "$INPUT_TEXT"
echo "Submit test (with text, no delay) passed"


# Case 3: No text, with delay
echo "Test submitting inferlet (no text, with delay)"
START_TIME=$SECONDS
submit_inferlet_attached "$INFERLET" "$PIE_CLI_CONFIG" --delay 5000 > /dev/null
ELAPSED_TIME=$((SECONDS - START_TIME))

if [ $ELAPSED_TIME -ge 3 ]; then
    echo "Submit test (no text, with delay) passed"
else
    echo "Error: inferlet completed too quickly"
    exit 1
fi

# Cleanup will be called automatically by the trap

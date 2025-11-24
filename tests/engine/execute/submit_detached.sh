#!/bin/bash

# Submit Detached Inferlet Test
#
# This test script verifies that the pie engine correctly handles submitting
# a detached inferlet via the pie-cli submit command.
#
# Usage:
#   ./submit_detached.sh [INFERLET_DIR]
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
#   6. Runs three test cases to verify detached inferlet behavior:
#      - Case 1: Submit inferlet in detached mode using --detached flag
#        Submits an inferlet with --detached, extracts the inferlet ID from
#        output, and verifies it appears in the list with "Detached" status
#      - Case 2: Submit inferlet in attached mode, then detach with Ctrl-D (EOF)
#        Submits an inferlet in attached mode (no --detached flag), closes stdin
#        after 1 second to signal EOF, waits for the engine to detect
#        disconnection, and verifies the inferlet automatically transitions to
#        "Detached" status
#      - Case 3: Submit inferlet in attached mode, then kill the client process
#        Submits an inferlet in attached mode, waits 2 seconds for it to launch,
#        kills the pie-cli process to simulate unexpected termination, waits for
#        the engine to detect the disconnection, and verifies the inferlet
#        automatically transitions to "Detached" status
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

### Case 1: Submit in detached mode
echo "Test submitting inferlet in detached mode"
OUTPUT=$(submit_inferlet_detached "$INFERLET" "$PIE_CLI_CONFIG" --delay 100000)
INFERLET_ID=$(extract_inferlet_id "$OUTPUT")
verify_inferlet_id "$INFERLET_ID" "$OUTPUT"

LIST_OUTPUT=$(get_inferlet_list "$PIE_CLI_CONFIG" --full)
verify_inferlet_status "$INFERLET_ID" "Detached" "$LIST_OUTPUT"

echo "Submit test in detached mode passed"


### Case 2: Submit in attached mode and then detach
echo "Test submitting inferlet in attached mode and then detaching"
OUTPUT=$(timeout 10 pie-cli submit "$INFERLET" --config "$PIE_CLI_CONFIG" \
    -- --delay 100000 < <(sleep 1)) || \
    { echo "Error: submit command failed"; exit 1; }
INFERLET_ID=$(extract_inferlet_id "$OUTPUT")
verify_inferlet_id "$INFERLET_ID" "$OUTPUT"

# Wait a moment for the engine to detect the client disconnection and update status
wait_for_state_change

LIST_OUTPUT=$(get_inferlet_list "$PIE_CLI_CONFIG" --full)
verify_inferlet_status "$INFERLET_ID" "Detached" "$LIST_OUTPUT"

echo "Submit test in attached mode and then detaching passed"


### Case 3: Submit in attached mode and kill the client process
echo "Test submitting inferlet in attached mode and killing the client"

# Create a temporary file to capture output
TEMP_OUTPUT=$(mktemp)

# Start submit command in background, redirecting output to temp file
timeout 10 pie-cli submit "$INFERLET" --config "$PIE_CLI_CONFIG" \
    -- --delay 100000 < <(sleep infinity) > "$TEMP_OUTPUT" 2>&1 &
SUBMIT_PID=$!

# Wait for the inferlet to launch and output to be written
sleep 2

# Kill the submit process to simulate unexpected client termination
kill "$SUBMIT_PID" 2>/dev/null || true
wait "$SUBMIT_PID" 2>/dev/null || true

# Read the output from the temp file
OUTPUT=$(cat "$TEMP_OUTPUT")
rm -f "$TEMP_OUTPUT"

INFERLET_ID=$(extract_inferlet_id "$OUTPUT")
verify_inferlet_id "$INFERLET_ID" "$OUTPUT"

# Wait a moment for the engine to detect the client disconnection and update status
wait_for_state_change

LIST_OUTPUT=$(get_inferlet_list "$PIE_CLI_CONFIG" --full)
verify_inferlet_status "$INFERLET_ID" "Detached" "$LIST_OUTPUT"

echo "Submit test in attached mode and killing the client passed"

# Cleanup will be called automatically by the trap

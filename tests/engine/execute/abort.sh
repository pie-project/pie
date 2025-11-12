#!/bin/bash

# Abort Inferlet Test
#
# This test script verifies that the pie engine correctly handles aborting
# inferlets in different states (detached, finished, and attached) via the
# pie-cli abort command.
#
# Usage:
#   ./abort.sh [INFERLET_DIR]
#
# Arguments:
#   INFERLET_DIR - Optional. Path to the directory containing the inferlet.
#                  Defaults to: ../../../example-apps/target/wasm32-wasip2/release
#
# Test Procedure:
#   1. Generates an ED25519 SSH key pair for authentication
#   2. Adds the public key to the pie authorized users list with a randomized
#      username
#   3. Initializes pie engine and pie-cli configurations with unique random
#      names
#   4. Configures the engine to use the dummy backend
#   5. Starts the pie engine in the background
#   6. Runs three test cases to verify abort functionality:
#      - Case 1: Abort a detached inferlet
#        Submits an inferlet in detached mode with a long delay, verifies it
#        appears in the list with "Detached" status, aborts it, and verifies
#        the inferlet no longer appears in the list
#      - Case 2: Abort a finished inferlet
#        Submits an inferlet in detached mode with no delay so it finishes
#        immediately, verifies it appears in the list with "Finished" status,
#        aborts it to clean it up, and verifies it no longer appears in the
#        list
#      - Case 3: Abort a running attached inferlet
#        Submits an inferlet in attached mode (in background) with a long
#        delay, verifies it appears in the list with "Attached" status, aborts
#        it, verifies it no longer appears in the list, and confirms the
#        background client process terminates
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
INFERLET_DIR="${1:-${SCRIPT_DIR}/../../../example-apps/target/wasm32-wasip2/release}"

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


### Case 1: Abort a detached inferlet
echo "Test aborting a detached inferlet"

# Submit inferlet in detached mode
OUTPUT=$(submit_inferlet_detached "$INFERLET" "$PIE_CLI_CONFIG" --delay 100000)
INFERLET_ID=$(extract_inferlet_id "$OUTPUT")
verify_inferlet_id "$INFERLET_ID" "$OUTPUT"

# Verify the inferlet appears in the list with "Detached" status
LIST_OUTPUT=$(get_inferlet_list "$PIE_CLI_CONFIG" --full)
verify_inferlet_status "$INFERLET_ID" "Detached" "$LIST_OUTPUT"

# Abort the inferlet
echo "Aborting inferlet $INFERLET_ID"
abort_inferlet "$INFERLET_ID" "$PIE_CLI_CONFIG"

# Wait for the engine to process the abort
wait_for_state_change

# Verify the inferlet no longer appears in the list
LIST_OUTPUT=$(get_inferlet_list "$PIE_CLI_CONFIG")
verify_inferlet_not_in_list "$INFERLET_ID" "$LIST_OUTPUT"

echo "Abort a detached inferlet passed"


### Case 2: Abort a finished inferlet
echo "Test aborting a finished inferlet"

# Submit inferlet in detached mode
# The inferlet will finish immediately because it has no delay.
OUTPUT=$(submit_inferlet_detached "$INFERLET" "$PIE_CLI_CONFIG")
INFERLET_ID=$(extract_inferlet_id "$OUTPUT")
verify_inferlet_id "$INFERLET_ID" "$OUTPUT"

# Wait for the inferlet to finish
wait_for_state_change

# Verify the inferlet appears in the list with "Finished" status
LIST_OUTPUT=$(get_inferlet_list "$PIE_CLI_CONFIG" --full)
verify_inferlet_status "$INFERLET_ID" "Finished" "$LIST_OUTPUT"

# Abort the inferlet
echo "Aborting inferlet $INFERLET_ID"
abort_inferlet "$INFERLET_ID" "$PIE_CLI_CONFIG"

# Wait for the engine to process the abort
wait_for_state_change

# Verify the inferlet no longer appears in the list
LIST_OUTPUT=$(get_inferlet_list "$PIE_CLI_CONFIG")
verify_inferlet_not_in_list "$INFERLET_ID" "$LIST_OUTPUT"

echo "Abort a finished inferlet passed"


### Case 3: Abort a running attached inferlet
echo "Test aborting a running attached inferlet"

# Create a temporary file to capture output
TEMP_OUTPUT=$(mktemp)

# Start submit command in background (attached mode, no --detached flag)
timeout 10 pie-cli submit "$INFERLET" --config "$PIE_CLI_CONFIG" \
    -- --delay 100000 < <(sleep infinity) > "$TEMP_OUTPUT" 2>&1 &
SUBMIT_PID=$!

# Wait for the inferlet to launch and output to be written
sleep 2

# Read the output to extract inferlet ID
OUTPUT=$(cat "$TEMP_OUTPUT")

INFERLET_ID=$(extract_inferlet_id "$OUTPUT")
if [ -z "$INFERLET_ID" ]; then
    echo "Error: Could not extract inferlet ID from output"
    echo "Output was: $OUTPUT"
    rm -f "$TEMP_OUTPUT"
    kill "$SUBMIT_PID" 2>/dev/null || true
    exit 1
fi

# Verify the inferlet appears in the list with "Attached" status
LIST_OUTPUT=$(get_inferlet_list "$PIE_CLI_CONFIG" --full)
if ! echo "$LIST_OUTPUT" | grep "$INFERLET_ID" | grep -q "Attached"; then
    echo "Error: List output does not contain inferlet $INFERLET_ID" \
        "with status Attached"
    echo "List output was:"
    echo "$LIST_OUTPUT"
    rm -f "$TEMP_OUTPUT"
    kill "$SUBMIT_PID" 2>/dev/null || true
    exit 1
fi

# Abort the inferlet
echo "Aborting inferlet $INFERLET_ID"
abort_inferlet "$INFERLET_ID" "$PIE_CLI_CONFIG"

# Wait for the engine to process the abort
wait_for_state_change

# Verify the inferlet no longer appears in the list
LIST_OUTPUT=$(get_inferlet_list "$PIE_CLI_CONFIG")
if echo "$LIST_OUTPUT" | grep -q "$INFERLET_ID"; then
    echo "Error: Inferlet $INFERLET_ID still appears in the list after abort"
    echo "List output was:"
    echo "$LIST_OUTPUT"
    rm -f "$TEMP_OUTPUT"
    exit 1
fi

# Verify the background submit command has finished
if kill -0 "$SUBMIT_PID" 2>/dev/null; then
    echo "Error: Background submit process still running after abort"
    kill "$SUBMIT_PID" 2>/dev/null || true
    rm -f "$TEMP_OUTPUT"
    exit 1
fi

# Wait for the process to fully clean up
wait "$SUBMIT_PID" 2>/dev/null || true

# Clean up temp file
rm -f "$TEMP_OUTPUT"

echo "Abort a running attached inferlet passed"

# Cleanup will be called automatically by the trap

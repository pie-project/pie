#!/bin/bash

# Attach to Inferlet Test
#
# This test script verifies that the pie engine correctly handles attaching
# to inferlets via the pie-cli attach command.
#
# Usage:
#   ./attach.sh [INFERLET_DIR]
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
#   6. Runs four test cases to verify attach functionality:
#      - Case 1: Attach to a finished inferlet and stream buffered output
#        Submits an inferlet in detached mode with random text input and no
#        delay so it finishes immediately, waits for it to complete, verifies
#        it appears in the list with "Finished" status, attaches to it to
#        stream back the buffered output, verifies the output contains the
#        input text, and confirms the inferlet no longer appears in the list
#      - Case 2: Attempt to attach to an already-attached inferlet
#        Submits an inferlet in attached mode (in background) with a 5-second
#        delay, verifies it appears in the list with "Attached" status,
#        attempts to attach to it using pie-cli attach, and verifies the
#        attach command fails as expected
#      - Case 3: Attach to a detached inferlet and stream new output
#        Submits an inferlet in detached mode with a 5-second delay and text
#        (output generated after delay), verifies it appears as "Detached",
#        attaches to it, waits for the inferlet to complete, verifies the
#        attach command finishes by itself and the output contains the text,
#        and confirms the inferlet is cleaned up after completion
#      - Case 4: Attach to a running detached inferlet and stream back buffered
#        output
#        Submits an inferlet in detached mode with text-before-delay and a
#        long delay, verifies it appears as "Detached", attaches to it to
#        stream back the buffered output, verifies it transitions to "Attached"
#        status during the attach, verifies the output contains the buffered
#        text, and confirms it transitions back to "Detached" after detaching
#        again
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


### Case 1: Attach to a finished inferlet and stream buffered output
echo "Test attaching to a finished inferlet and streaming buffered output"

# Generate random text for the inferlet
INPUT_TEXT="test-$(generate_random_string)"

# Submit inferlet in detached mode with no delay (will finish immediately)
OUTPUT=$(submit_inferlet_detached "$INFERLET" "$PIE_CLI_CONFIG" --text "$INPUT_TEXT")
INFERLET_ID=$(extract_inferlet_id "$OUTPUT")
verify_inferlet_id "$INFERLET_ID" "$OUTPUT"

# Wait for the inferlet to finish
wait_for_state_change

# Verify the inferlet appears in the list with "Finished" status
LIST_OUTPUT=$(get_inferlet_list "$PIE_CLI_CONFIG" --full)
verify_inferlet_status "$INFERLET_ID" "Finished" "$LIST_OUTPUT"

# Attach to the finished inferlet
echo "Attaching to inferlet $INFERLET_ID"
ATTACH_OUTPUT=$(timeout 10 pie-cli attach "$INFERLET_ID" --config "$PIE_CLI_CONFIG" \
    < <(sleep infinity)) || \
    { echo "Error: attach command failed"; exit 1; }

# Verify the attach output contains the input text
if ! echo "$ATTACH_OUTPUT" | grep -q "$INPUT_TEXT"; then
    echo "Error: Attach output does not contain input text '$INPUT_TEXT'"
    echo "Attach output was:"
    echo "$ATTACH_OUTPUT"
    exit 1
fi

# Wait for the engine to terminate the inferlet after attach
wait_for_state_change

# Verify the inferlet no longer appears in the list
LIST_OUTPUT=$(get_inferlet_list "$PIE_CLI_CONFIG")
verify_inferlet_not_in_list "$INFERLET_ID" "$LIST_OUTPUT"

echo "Attach to a finished inferlet and stream buffered output passed"


### Case 2: Attempt to attach to an already-attached inferlet
echo "Test attempting to attach to an already-attached inferlet"

# Create a temporary file to capture output
TEMP_OUTPUT=$(mktemp)

# Start submit command in background (attached mode, no --detached flag)
timeout 10 pie-cli submit "$INFERLET" --config "$PIE_CLI_CONFIG" \
    -- --delay 5000 < <(sleep infinity) > "$TEMP_OUTPUT" 2>&1 &
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

# Attempt to attach to the already-attached inferlet (should fail)
echo "Attempting to attach to already-attached inferlet $INFERLET_ID"
if timeout 10 pie-cli attach "$INFERLET_ID" --config "$PIE_CLI_CONFIG" \
    < <(sleep infinity) >/dev/null 2>&1; then
    echo "Error: Attach command unexpectedly succeeded for already-attached inferlet"
    rm -f "$TEMP_OUTPUT"
    kill "$SUBMIT_PID" 2>/dev/null || true
    exit 1
fi

echo "Attach command correctly failed for already-attached inferlet"

# Clean up
wait "$SUBMIT_PID" 2>/dev/null || true
rm -f "$TEMP_OUTPUT"

echo "Attempt to attach to an already-attached inferlet passed"


### Case 3: Attach to a detached inferlet and stream new output
echo "Test attaching to a detached inferlet and streaming new output"

# Generate random text for the inferlet
INPUT_TEXT="test-$(generate_random_string)"

# Submit inferlet in detached mode with 5-second delay and text (output comes after delay)
OUTPUT=$(submit_inferlet_detached "$INFERLET" "$PIE_CLI_CONFIG" --delay 5000 --text "$INPUT_TEXT")
INFERLET_ID=$(extract_inferlet_id "$OUTPUT")
verify_inferlet_id "$INFERLET_ID" "$OUTPUT"

# Verify the inferlet appears in the list with "Detached" status
LIST_OUTPUT=$(get_inferlet_list "$PIE_CLI_CONFIG" --full)
verify_inferlet_status "$INFERLET_ID" "Detached" "$LIST_OUTPUT"

# Attach to the detached inferlet
# The command should finish by itself after the inferlet completes (5 seconds)
echo "Attaching to inferlet $INFERLET_ID (will wait for inferlet to complete)"
ATTACH_OUTPUT=$(timeout 10 pie-cli attach "$INFERLET_ID" --config "$PIE_CLI_CONFIG" \
    < <(sleep infinity)) || \
    { echo "Error: attach command failed"; exit 1; }

# Verify the attach output contains the input text
if ! echo "$ATTACH_OUTPUT" | grep -q "$INPUT_TEXT"; then
    echo "Error: Attach output does not contain input text '$INPUT_TEXT'"
    echo "Attach output was:"
    echo "$ATTACH_OUTPUT"
    exit 1
fi

# Wait for the engine to clean up the finished inferlet
wait_for_state_change

# Verify the inferlet no longer appears in the list (it should be cleaned up after completion)
LIST_OUTPUT=$(get_inferlet_list "$PIE_CLI_CONFIG")
verify_inferlet_not_in_list "$INFERLET_ID" "$LIST_OUTPUT"

echo "Attach to a detached inferlet and stream new output passed"


### Case 4: Attach to a running detached inferlet and stream back buffered output
echo "Test attaching to a running detached inferlet and streaming back buffered output"

# Generate random text for the inferlet
INPUT_TEXT="test-$(generate_random_string)"

# Submit inferlet in detached mode with long delay and text before delay
OUTPUT=$(submit_inferlet_detached "$INFERLET" "$PIE_CLI_CONFIG" \
    --text-before-delay "$INPUT_TEXT" --delay 100000)
INFERLET_ID=$(extract_inferlet_id "$OUTPUT")
verify_inferlet_id "$INFERLET_ID" "$OUTPUT"

# Wait for the inferlet to start running
wait_for_state_change

# Verify the inferlet appears in the list with "Detached" status
LIST_OUTPUT=$(get_inferlet_list "$PIE_CLI_CONFIG" --full)
verify_inferlet_status "$INFERLET_ID" "Detached" "$LIST_OUTPUT"

# Create a temporary file to capture attach output
TEMP_OUTPUT=$(mktemp)

# Start attach command in background and detach after 2 seconds
timeout 10 pie-cli attach "$INFERLET_ID" --config "$PIE_CLI_CONFIG" \
    < <(sleep 5) > "$TEMP_OUTPUT" 2>&1 &
ATTACH_PID=$!

# Wait a moment for the attach to establish
wait_for_state_change

# Verify the inferlet now appears in the list with "Attached" status
LIST_OUTPUT=$(get_inferlet_list "$PIE_CLI_CONFIG" --full)
if ! echo "$LIST_OUTPUT" | grep "$INFERLET_ID" | grep -q "Attached"; then
    echo "Error: List output does not contain inferlet $INFERLET_ID" \
        "with status Attached"
    echo "List output was:"
    echo "$LIST_OUTPUT"
    kill "$ATTACH_PID" 2>/dev/null || true
    rm -f "$TEMP_OUTPUT"
    exit 1
fi

# Wait for the attach command to finish (after stdin closes)
wait "$ATTACH_PID" 2>/dev/null || true

# Read and verify the attach output contains the input text
ATTACH_OUTPUT=$(cat "$TEMP_OUTPUT")
rm -f "$TEMP_OUTPUT"

if ! echo "$ATTACH_OUTPUT" | grep -q "$INPUT_TEXT"; then
    echo "Error: Attach output does not contain input text '$INPUT_TEXT'"
    echo "Attach output was:"
    echo "$ATTACH_OUTPUT"
    exit 1
fi

# Wait for the engine to transition the inferlet back to detached state
wait_for_state_change

# Verify the inferlet is back to "Detached" status
LIST_OUTPUT=$(get_inferlet_list "$PIE_CLI_CONFIG" --full)
verify_inferlet_status "$INFERLET_ID" "Detached" "$LIST_OUTPUT"

echo "Attach to a running detached inferlet and stream back buffered output passed"

# Cleanup will be called automatically by the trap


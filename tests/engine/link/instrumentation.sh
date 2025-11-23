#!/bin/bash

# Instrumentation Link Test
#
# This test script verifies that the Pie engine correctly handles linking
# a component inferlet via the pie-cli submit command with the --link flag
# when the library instruments the Pie engine's runtime API.
#
# Usage:
#   ./instrumentation.sh [INFERLET_DIR]
#
# Arguments:
#   INFERLET_DIR - Optional. Path to the directory containing the inferlets.
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
#   6. Runs three test cases to verify runtime instrumentation linking:
#      - Case 1: Submit inferlet without linked runtime instrumentation library
#        Submits version.wasm without the --link flag and verifies that the
#        output does NOT contain "Instrumented", confirming baseline behavior
#      - Case 2: Submit inferlet with linked runtime instrumentation library
#        Submits version.wasm with runtime_instrument_lib.wasm linked via the
#        --link flag, verifies that the submit command completes successfully
#        and the output contains "Instrumented", demonstrating that the runtime
#        instrumentation API is working correctly
#      - Case 3: Submit inferlet with nested instrumentation
#        Submits version.wasm with runtime_instrument_lib.wasm linked twice via
#        two --link flags, verifies that the output contains "Instrumented:
#        Instrumented:", confirming that nested/multiple instrumentation layers
#        work correctly
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

# Configure the inferlets to use
echo "Using inferlet directory: $INFERLET_DIR"
VERSION_INFERLET="$INFERLET_DIR/version.wasm"
RUNTIME_INSTRUMENT_LIB_INFERLET="$INFERLET_DIR/runtime_instrument_lib.wasm"

### Case 1: Submit inferlet without linked runtime instrumentation library
echo "Test submitting inferlet without linked library"
OUTPUT=$(timeout 10 pie-cli submit "$VERSION_INFERLET" \
    --config "$PIE_CLI_CONFIG" < <(sleep infinity) || \
    { echo "Error: submit command failed (Case 1)"; exit 1; })

# Verify the output does NOT contain "Instrumented"
if echo "$OUTPUT" | grep -q "Instrumented"; then
    echo "Error: Output contains 'Instrumented' when it should not (Case 1)"
    echo "Output was:"
    echo "$OUTPUT"
    exit 1
else
    echo "Case 1 passed: output does not contain 'Instrumented' when not linked"
fi


### Case 2: Submit inferlet with linked runtime instrumentation library
echo "Test submitting inferlet with linked runtime instrumentation library"
OUTPUT=$(timeout 10 pie-cli submit "$VERSION_INFERLET" --link "$RUNTIME_INSTRUMENT_LIB_INFERLET" \
    --config "$PIE_CLI_CONFIG" < <(sleep infinity) || \
    { echo "Error: submit command failed (Case 2)"; exit 1; })

# Verify the output contains "Instrumented"
if echo "$OUTPUT" | grep -q "Instrumented"; then
    echo "Case 2 passed: output contains 'Instrumented' when linked"
else
    echo "Error: Output does not contain 'Instrumented' (Case 2)"
    echo "Output was:"
    echo "$OUTPUT"
    exit 1
fi


### Case 3: Submit inferlet with nested instrumentation
echo "Test submitting inferlet with nested instrumentation"
OUTPUT=$(timeout 10 pie-cli submit "$VERSION_INFERLET" \
    --link "$RUNTIME_INSTRUMENT_LIB_INFERLET" \
    --link "$RUNTIME_INSTRUMENT_LIB_INFERLET" \
    --config "$PIE_CLI_CONFIG" < <(sleep infinity) || \
    { echo "Error: submit command failed (Case 3)"; exit 1; })

# Verify the output contains "Instrumented: Instrumented:"
if echo "$OUTPUT" | grep -q "Instrumented: Instrumented:"; then
    echo "Case 3 passed: output contains 'Instrumented: Instrumented:' with nested instrumentation"
else
    echo "Error: Output does not contain 'Instrumented: Instrumented:' (Case 3)"
    echo "Output was:"
    echo "$OUTPUT"
    exit 1
fi

# Cleanup will be called automatically by the trap

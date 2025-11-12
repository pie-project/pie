#!/bin/bash

# Utility functions for inferlet execution testing

# Extract inferlet ID from submit/attach output
# Usage: extract_inferlet_id "$OUTPUT"
# Returns: The inferlet UUID, or empty string if not found
extract_inferlet_id() {
    local output=$1
    echo "$output" | grep "Inferlet launched with ID:" | \
        sed -E 's/.*Inferlet launched with ID: ([0-9a-f-]+).*/\1/'
}

# Verify that an inferlet ID was successfully extracted
# Usage: verify_inferlet_id "$INFERLET_ID" "$OUTPUT"
# Exits with error if ID is empty
verify_inferlet_id() {
    local inferlet_id=$1
    local output=$2
    
    if [ -z "$inferlet_id" ]; then
        echo "Error: Could not extract inferlet ID from output"
        echo "Output was: $output"
        exit 1
    fi
}

# Get list output from pie-cli
# Usage: get_inferlet_list "$PIE_CLI_CONFIG" [--full]
# Returns: The list output
get_inferlet_list() {
    local config=$1
    local full_flag=$2
    
    if [ "$full_flag" = "--full" ]; then
        timeout 10 pie-cli list --full --config "$config" || \
            { echo "Error: list command failed"; exit 1; }
    else
        timeout 10 pie-cli list --config "$config" || \
            { echo "Error: list command failed"; exit 1; }
    fi
}

# Verify that an inferlet appears in the list with a specific status
# Usage: verify_inferlet_status "$INFERLET_ID" "$STATUS" "$LIST_OUTPUT"
# Exits with error if the inferlet doesn't have the expected status
verify_inferlet_status() {
    local inferlet_id=$1
    local expected_status=$2
    local list_output=$3
    
    if ! echo "$list_output" | grep "$inferlet_id" | grep -q "$expected_status"; then
        echo "Error: List output does not contain inferlet $inferlet_id" \
            "with status $expected_status"
        echo "List output was:"
        echo "$list_output"
        exit 1
    fi
}

# Verify that an inferlet does NOT appear in the list
# Usage: verify_inferlet_not_in_list "$INFERLET_ID" "$LIST_OUTPUT"
# Exits with error if the inferlet is found in the list
verify_inferlet_not_in_list() {
    local inferlet_id=$1
    local list_output=$2
    
    if echo "$list_output" | grep -q "$inferlet_id"; then
        echo "Error: Inferlet $inferlet_id still appears in the list"
        echo "List output was:"
        echo "$list_output"
        exit 1
    fi
}

# Wait for the engine to process state changes
# Usage: wait_for_state_change [SECONDS]
# Default: 2 seconds
wait_for_state_change() {
    local seconds=${1:-2}
    sleep "$seconds"
}

# Submit an inferlet in detached mode
# Usage: submit_inferlet_detached "$INFERLET" "$CONFIG" [EXTRA_ARGS...]
# Returns: The submit command output
submit_inferlet_detached() {
    local inferlet=$1
    local config=$2
    shift 2
    local extra_args="$@"
    
    if [ -n "$extra_args" ]; then
        timeout 10 pie-cli submit "$inferlet" --config "$config" \
            --detached -- $extra_args < <(sleep infinity) || \
            { echo "Error: submit command failed"; exit 1; }
    else
        timeout 10 pie-cli submit "$inferlet" --config "$config" \
            --detached < <(sleep infinity) || \
            { echo "Error: submit command failed"; exit 1; }
    fi
}

# Submit an inferlet in attached mode (without --detached)
# Usage: submit_inferlet_attached "$INFERLET" "$CONFIG" [EXTRA_ARGS...]
# Returns: The submit command output
submit_inferlet_attached() {
    local inferlet=$1
    local config=$2
    shift 2
    local extra_args="$@"
    
    if [ -n "$extra_args" ]; then
        timeout 10 pie-cli submit "$inferlet" --config "$config" \
            -- $extra_args < <(sleep infinity) || \
            { echo "Error: submit command failed"; exit 1; }
    else
        timeout 10 pie-cli submit "$inferlet" --config "$config" \
            < <(sleep infinity) || \
            { echo "Error: submit command failed"; exit 1; }
    fi
}

# Abort an inferlet
# Usage: abort_inferlet "$INFERLET_ID" "$CONFIG"
abort_inferlet() {
    local inferlet_id=$1
    local config=$2
    
    timeout 10 pie-cli abort "$inferlet_id" --config "$config" || \
        { echo "Error: abort command failed"; exit 1; }
}

# Attach to an inferlet
# Usage: attach_to_inferlet "$INFERLET_ID" "$CONFIG" [STDIN_SOURCE]
# STDIN_SOURCE: Optional, defaults to <(sleep infinity)
# Returns: The attach command output
attach_to_inferlet() {
    local inferlet_id=$1
    local config=$2
    local stdin_source=${3:-<(sleep infinity)}
    
    timeout 10 pie-cli attach "$inferlet_id" --config "$config" \
        < "$stdin_source" || \
        { echo "Error: attach command failed"; exit 1; }
}


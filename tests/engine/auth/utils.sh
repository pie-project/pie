#!/bin/bash

# Utility functions for SSH key testing

# Generate a random 8-character string
generate_random_string() {
      xxd -p -l 4 /dev/urandom
}

# Generate a unique key path (ensures no collision with existing files)
# Usage: generate_unique_key_path <algorithm>
# Example: generate_unique_key_path "rsa"
# Returns: /tmp/pie-test-<algorithm>-<random_string>
generate_unique_key_path() {
    local algorithm=$1
    local key_path
    local max_attempts=100
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        key_path="/tmp/pie-test-${algorithm}-$(generate_random_string)"
        
        # Check if neither the private key nor public key exists
        if [ ! -f "$key_path" ] && [ ! -f "${key_path}.pub" ]; then
            echo "$key_path"
            return 0
        fi
        
        attempt=$((attempt + 1))
    done
    
    echo "Error: Could not generate unique key path after $max_attempts attempts" >&2
    return 1
}

# Generate a unique config file path (ensures no collision with existing files)
# Usage: generate_unique_config_path <config_name>
# Example: generate_unique_config_path "pie-test-config"
# Returns: /tmp/<config_name>-<random_string>.toml
generate_unique_config_path() {
    local config_name=$1
    local config_path
    local max_attempts=100
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        config_path="/tmp/${config_name}-$(generate_random_string).toml"
        
        # Check if the config file doesn't exist
        if [ ! -f "$config_path" ]; then
            echo "$config_path"
            return 0
        fi
        
        attempt=$((attempt + 1))
    done
    
    echo "Error: Could not generate unique config path after $max_attempts attempts" >&2
    return 1
}

# Start the Pie engine and wait for it to be ready
# Usage: start_pie_engine <config_path> [pid_variable_name]
# Example: start_pie_engine "$PIE_CONFIG" "PIE_SERVE_PID"
# Sets the specified variable (default: PIE_SERVE_PID) to the process ID
# The engine will start on port 8080
# Returns 0 on success, non-zero on failure
start_pie_engine() {
    local config_path=$1
    local pid_var=${2:-PIE_SERVE_PID}
    local port=8080
    
    if [ -z "$config_path" ]; then
        echo "Error: config_path is required" >&2
        return 1
    fi
    
    if [ ! -f "$config_path" ]; then
        echo "Error: config file not found: $config_path" >&2
        return 1
    fi
    
    # Start pie serve in background
    pie serve --config "$config_path" >/dev/null 2>&1 &
    local pid=$!
    
    # Set the PID to the specified variable name
    # Since this function is sourced, the variable will be in the caller's scope
    eval "${pid_var}=${pid}"
    
    # Wait for the server to be ready
    echo "Waiting for pie serve to start..."
    sleep 3
    
    # Check if process is still running
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "Error: pie serve process died immediately" >&2
        return 1
    fi
    
    # Try to connect with a timeout, retry up to 15 times
    local max_retries=15
    local retry_count=0
    while [ $retry_count -lt $max_retries ]; do
        # Use bash built-in TCP check (more portable than nc)
        if timeout 1 bash -c "echo > /dev/tcp/127.0.0.1/$port" 2>/dev/null; then
            echo "Server is ready"
            return 0
        fi
        retry_count=$((retry_count + 1))
        if [ $retry_count -ge $max_retries ]; then
            echo "Error: Server failed to become ready after $max_retries attempts" >&2
            return 1
        fi
        sleep 1
    done
    
    return 1
}

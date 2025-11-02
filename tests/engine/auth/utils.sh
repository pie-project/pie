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


# Makefile for simplifying the build process

# Variables
BUILD_DIR := build

# Optional override: set CUDA_COMPILER to force a specific nvcc path.
# Example: make CUDA_COMPILER=/usr/local/cuda/bin/nvcc
# If unset, CMake will auto-detect the CUDA toolkit.
CUDA_COMPILER ?=

# Construct optional CMake argument only if provided
CMAKE_CUDA_ARG := $(if $(strip $(CUDA_COMPILER)),-DCMAKE_CUDA_COMPILER=$(CUDA_COMPILER))

# Default target
.PHONY: all
all: build

# Target to build the project
.PHONY: build
build:
	@echo "Configuring and building the project..."; \
	cmake -S . -B $(BUILD_DIR) $(CMAKE_CUDA_ARG); \
	$(MAKE) -C $(BUILD_DIR) -j$(shell nproc)

## (removed) test data generator target was here; generator deleted.

# Target to clean the build directory
.PHONY: clean
clean:
	@echo "Cleaning the build directory..."
	@rm -rf $(BUILD_DIR)

.PHONY: unit-tests run-unit-tests

# List of current unit test targets (CMake adds them as separate executables)
TEST_TARGETS := \
	test_sum_reduction \
	test_embedding_lookup \
	test_embedding_lookup_bf16 \
	test_extract_k_values \
	test_extract_k_values_bf16 \
	test_cast_type \
	test_cast_type_half \
	test_gemm \
	test_tensor_view \
	test_add_residual_full \
	test_act_and_mul_full \
	test_forward_text_full

unit-tests: build
	@echo "Building unit test executables..."
	@$(MAKE) -C $(BUILD_DIR) $(TEST_TARGETS)
	@echo "Done. Binaries are in $(BUILD_DIR)/bin"

# Convenience target to build (if needed) and then run all unit tests sequentially
run-unit-tests: unit-tests
	@echo "Running unit tests..."
	@set -e; for t in $(TEST_TARGETS); do \
	  echo "Running $$t"; \
	  ./$(BUILD_DIR)/bin/$$t || { echo "$$t FAILED"; exit 1; }; \
	done; \
	echo "All unit tests passed"

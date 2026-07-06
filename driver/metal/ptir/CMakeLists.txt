# Standalone build for the PTIR sampling-IR Metal parity harness.
#
# NOT wired into the main MLX metal driver build (no MLX dependency) — this is
# the cross-backend op-parity track, kept standalone for fast iteration and
# zero build/GPU contention with the model-forward path. Mirrors the raw_metal
# Phase-0 standalone pattern.
#
#   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
#   cmake --build build
#   ./build/ptir_metal_test        # prints PTIR_METAL_TEST_OK
#
# Or direct (CLT-only, no full Xcode):
#   clang++ -std=c++17 -fobjc-arc -O2 -DPTIR_KERNELS_DIR='"'$PWD/kernels'"' \
#     -x objective-c++ metal_harness.mm ptir_metal_test.mm \
#     -framework Metal -framework Foundation -o ptir_metal_test

cmake_minimum_required(VERSION 3.20)
project(ptir_metal LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)

add_executable(ptir_metal_test
  metal_harness.mm
  ptir_metal_test.mm)

set_source_files_properties(
  metal_harness.mm ptir_metal_test.mm
  PROPERTIES COMPILE_FLAGS "-fobjc-arc")

target_compile_definitions(ptir_metal_test PRIVATE
  PTIR_KERNELS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/kernels")

target_link_libraries(ptir_metal_test PRIVATE
  ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})

# Cross-backend golden cert: reuses charlie's CUDA-free PTIR decoder
# (driver/cuda/src/ptir/container.hpp) to verify container identity, then runs
# the Metal ops against echo's golden vectors.
add_executable(ptir_golden_cert
  metal_harness.mm
  ptir_golden_cert.mm)

set_source_files_properties(
  metal_harness.mm ptir_golden_cert.mm
  PROPERTIES COMPILE_FLAGS "-fobjc-arc")

target_include_directories(ptir_golden_cert PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../../cuda/src")

target_compile_definitions(ptir_golden_cert PRIVATE
  PTIR_KERNELS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/kernels"
  PTIR_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../../interface/sampling-ir/tests/golden-ptir")

target_link_libraries(ptir_golden_cert PRIVATE
  ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})

enable_testing()
add_test(NAME ptir_metal_test COMMAND ptir_metal_test)
set_tests_properties(ptir_metal_test PROPERTIES
  PASS_REGULAR_EXPRESSION "PTIR_METAL_TEST_OK")
add_test(NAME ptir_golden_cert COMMAND ptir_golden_cert)
set_tests_properties(ptir_golden_cert PROPERTIES
  PASS_REGULAR_EXPRESSION "PTIR_GOLDEN_CERT_OK")

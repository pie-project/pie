# Standalone build for the Metal KV/attention parity harness (M3).
# Reuses the ptir MetalHarness (../ptir/metal_harness.*). No MLX. CLT-only OK
# (kernels runtime-compiled).
#
#   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
#   cmake --build build
#   ./build/kvattn_test        # prints KVATTN_TEST_OK

cmake_minimum_required(VERSION 3.20)
project(kvattn_metal LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(MPS_FRAMEWORK MetalPerformanceShaders REQUIRED)

add_executable(kvattn_test
  ${CMAKE_CURRENT_SOURCE_DIR}/../ptir/metal_harness.mm
  kvattn_test.mm)

set_source_files_properties(
  ${CMAKE_CURRENT_SOURCE_DIR}/../ptir/metal_harness.mm kvattn_test.mm
  PROPERTIES COMPILE_FLAGS "-fobjc-arc")

target_include_directories(kvattn_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../ptir)

target_compile_definitions(kvattn_test PRIVATE
  KVATTN_KERNELS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/kernels")

target_link_libraries(kvattn_test PRIVATE
  ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK} ${MPS_FRAMEWORK})

enable_testing()
add_test(NAME kvattn_test COMMAND kvattn_test)
set_tests_properties(kvattn_test PROPERTIES
  PASS_REGULAR_EXPRESSION "KVATTN_TEST_OK")

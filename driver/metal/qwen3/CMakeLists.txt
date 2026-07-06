# Standalone build for the Qwen3-0.6B Metal decoder-layer parity harness.
# Reuses the ptir MetalHarness (../ptir/metal_harness.*). No MLX. CLT-only OK.
#
#   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
#   cmake --build build
#   ./build/qwen3_test        # prints QWEN3_TEST_OK

cmake_minimum_required(VERSION 3.20)
project(qwen3_metal LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)

add_executable(qwen3_test
  ${CMAKE_CURRENT_SOURCE_DIR}/../ptir/metal_harness.mm
  qwen3_test.mm)

set_source_files_properties(
  ${CMAKE_CURRENT_SOURCE_DIR}/../ptir/metal_harness.mm qwen3_test.mm
  PROPERTIES COMPILE_FLAGS "-fobjc-arc")

target_include_directories(qwen3_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../ptir)

target_compile_definitions(qwen3_test PRIVATE
  QWEN3_KERNELS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/kernels")

target_link_libraries(qwen3_test PRIVATE
  ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})

enable_testing()
add_test(NAME qwen3_test COMMAND qwen3_test)
set_tests_properties(qwen3_test PROPERTIES
  PASS_REGULAR_EXPRESSION "QWEN3_TEST_OK")

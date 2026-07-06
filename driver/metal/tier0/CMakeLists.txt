# Standalone build for the PTIR tier-0 interpreter cert on Metal.
# Reuses charlie's CUDA-free decoder (driver/cuda/src/ptir) + the ptir MetalHarness.
#
#   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
#   cmake --build build
#   ./build/tier0_test        # prints TIER0_TEST_OK

cmake_minimum_required(VERSION 3.20)
project(tier0_metal LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(MPS_FRAMEWORK MetalPerformanceShaders REQUIRED)

add_executable(tier0_test
  ${CMAKE_CURRENT_SOURCE_DIR}/../ptir/metal_harness.mm
  tier0_test.mm)

set_source_files_properties(
  ${CMAKE_CURRENT_SOURCE_DIR}/../ptir/metal_harness.mm tier0_test.mm
  PROPERTIES COMPILE_FLAGS "-fobjc-arc")

target_include_directories(tier0_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../ptir
  ${CMAKE_CURRENT_SOURCE_DIR}/../../cuda/src)

target_compile_definitions(tier0_test PRIVATE
  TIER0_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../../interface/sampling-ir/tests/golden-ptir"
  TIER0_KERNELS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../ptir/kernels")

target_link_libraries(tier0_test PRIVATE
  ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK} ${MPS_FRAMEWORK})

# PTIR-native MTP spec-verify cert (real Qwen3.5 MTP drafts through the tier-0
# Metal op kernels: de-hardwiring + the spec_verify_greedy sentinel accept-tail).
add_executable(mtpverify_test
  ${CMAKE_CURRENT_SOURCE_DIR}/../ptir/metal_harness.mm
  mtpverify_test.mm)

set_source_files_properties(
  ${CMAKE_CURRENT_SOURCE_DIR}/../ptir/metal_harness.mm mtpverify_test.mm
  PROPERTIES COMPILE_FLAGS "-fobjc-arc")

target_include_directories(mtpverify_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../ptir
  ${CMAKE_CURRENT_SOURCE_DIR}/../../cuda/src)

target_compile_definitions(mtpverify_test PRIVATE
  TIER0_KERNELS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../ptir/kernels")

target_link_libraries(mtpverify_test PRIVATE
  ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK} ${MPS_FRAMEWORK})

# pentathlon_iter cross-backend cert (6-technique composition, 2 programs,
# envelope_dot kernel host + sink assertions).
add_executable(pentathlon_test
  ${CMAKE_CURRENT_SOURCE_DIR}/../ptir/metal_harness.mm
  pentathlon_test.mm)

set_source_files_properties(
  ${CMAKE_CURRENT_SOURCE_DIR}/../ptir/metal_harness.mm pentathlon_test.mm
  PROPERTIES COMPILE_FLAGS "-fobjc-arc")

target_include_directories(pentathlon_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../ptir
  ${CMAKE_CURRENT_SOURCE_DIR}/../../cuda/src)

target_compile_definitions(pentathlon_test PRIVATE
  TIER0_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../../interface/sampling-ir/tests/golden-ptir"
  TIER0_KERNELS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../ptir/kernels")

target_link_libraries(pentathlon_test PRIVATE
  ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK} ${MPS_FRAMEWORK})

enable_testing()
add_test(NAME tier0_test COMMAND tier0_test)
set_tests_properties(tier0_test PROPERTIES PASS_REGULAR_EXPRESSION "TIER0_TEST_OK")
add_test(NAME mtpverify_test COMMAND mtpverify_test)
set_tests_properties(mtpverify_test PROPERTIES PASS_REGULAR_EXPRESSION "MTPVERIFY_OK")
add_test(NAME pentathlon_test COMMAND pentathlon_test)
set_tests_properties(pentathlon_test PROPERTIES PASS_REGULAR_EXPRESSION "PENTATHLON_OK")
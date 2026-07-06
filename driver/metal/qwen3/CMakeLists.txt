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
find_library(MPS_FRAMEWORK MetalPerformanceShaders REQUIRED)

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
  ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK} ${MPS_FRAMEWORK})

# Real-weight forward driver (not auto-tested — needs Qwen3-0.6B model.safetensors):
#   ./build/qwen3_forward <model.safetensors> kernels qwen3_golden.txt
add_executable(qwen3_forward
  ${CMAKE_CURRENT_SOURCE_DIR}/../ptir/metal_harness.mm
  qwen3_forward.mm)
set_source_files_properties(
  ${CMAKE_CURRENT_SOURCE_DIR}/../ptir/metal_harness.mm qwen3_forward.mm
  PROPERTIES COMPILE_FLAGS "-fobjc-arc")
target_include_directories(qwen3_forward PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../ptir)
target_compile_definitions(qwen3_forward PRIVATE
  QWEN3_KERNELS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/kernels")
target_link_libraries(qwen3_forward PRIVATE
  ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK} ${MPS_FRAMEWORK})

# Greedy autoregressive generation driver (needs model.safetensors):
#   ./build/qwen3_generate <model.safetensors> kernels 20 qwen3_generate.txt
add_executable(qwen3_generate
  ${CMAKE_CURRENT_SOURCE_DIR}/../ptir/metal_harness.mm
  qwen3_generate.mm)
set_source_files_properties(
  ${CMAKE_CURRENT_SOURCE_DIR}/../ptir/metal_harness.mm qwen3_generate.mm
  PROPERTIES COMPILE_FLAGS "-fobjc-arc")
target_include_directories(qwen3_generate PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../ptir)
target_compile_definitions(qwen3_generate PRIVATE
  QWEN3_KERNELS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/kernels")
target_link_libraries(qwen3_generate PRIVATE
  ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK} ${MPS_FRAMEWORK})

# Algorithm loops (need model.safetensors): beam search + constrained spec-decode.
add_executable(qwen3_beam
  ${CMAKE_CURRENT_SOURCE_DIR}/../ptir/metal_harness.mm qwen3_beam.mm)
add_executable(qwen3_specdecode
  ${CMAKE_CURRENT_SOURCE_DIR}/../ptir/metal_harness.mm qwen3_specdecode.mm)
add_executable(qwen3_mcts
  ${CMAKE_CURRENT_SOURCE_DIR}/../ptir/metal_harness.mm qwen3_mcts.mm)
foreach(algo qwen3_beam qwen3_specdecode qwen3_mcts)
  set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/../ptir/metal_harness.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
  target_include_directories(${algo} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../ptir)
  target_compile_definitions(${algo} PRIVATE
    QWEN3_KERNELS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/kernels"
    PTIR_KERNELS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../ptir/kernels")
  target_link_libraries(${algo} PRIVATE
    ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK} ${MPS_FRAMEWORK})
endforeach()
set_source_files_properties(qwen3_beam.mm qwen3_specdecode.mm qwen3_mcts.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")

enable_testing()
add_test(NAME qwen3_test COMMAND qwen3_test)
set_tests_properties(qwen3_test PROPERTIES
  PASS_REGULAR_EXPRESSION "QWEN3_TEST_OK")

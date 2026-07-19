cmake_minimum_required(VERSION 3.23)

if(APPLE)
  # driver/metal — MLX + native Metal backend for Pie (Apple Silicon, macOS).
  # Structure mirrors driver/cuda. OBJCXX is enabled so the compute layer can
  # add `.mm` (Objective-C++) translation units for Metal interop.
  project(PIE_DRIVER_METAL CXX OBJCXX)
else()
  # Linux/CI builds compile the direct-ABI stub surface (abi.cpp) so the
  # driver still validates, unit-tests its bind layout, and stays in sync with
  # the canonical ABI even when Metal frameworks are unavailable.
  project(PIE_DRIVER_METAL CXX)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

find_package(Threads REQUIRED)
enable_testing()

# Direct driver ABI header. The worker build forwards this when needed; local
# standalone builds fall back to the committed in-tree generated header.
if(NOT PIE_DRIVER_ABI_INCLUDE_DIR)
  set(PIE_DRIVER_ABI_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../interface/driver/include")
endif()

# Optional MLX dependency. OFF by default so the direct entry surface
# compiles/links fast; the compute layer flips it ON
# (see cmake/FetchMLX.cmake). The worker build.rs forwards
# -DPIE_METAL_WITH_MLX from the PIE_METAL_WITH_MLX env var.
option(PIE_METAL_WITH_MLX "Fetch and link MLX C++" OFF)
if(NOT APPLE AND PIE_METAL_WITH_MLX)
  message(FATAL_ERROR "PIE_METAL_WITH_MLX requires APPLE")
endif()

include(cmake/CPM.cmake)

# Use a persistent CPM cache when the env var is set so incremental builds
# don't re-clone from GitHub.
if(DEFINED ENV{CPM_SOURCE_CACHE} AND NOT CPM_SOURCE_CACHE)
  set(CPM_SOURCE_CACHE $ENV{CPM_SOURCE_CACHE})
endif()

cpmaddpackage(URI "gh:marzer/tomlplusplus@3.4.0")
cpmaddpackage(URI "gh:CLIUtils/CLI11@2.5.0")
cpmaddpackage(URI "gh:nlohmann/json@3.12.0" OPTIONS "JSON_BuildTests OFF")

# -----------------------------------------------------------------------------
# Sources
#
# Explicit direct-ABI entry sources.
# -----------------------------------------------------------------------------
# These compile WITHOUT MLX so the direct ABI stays available even when the
# optional MLX backend is disabled.
set(PIE_METAL_EXPLICIT_SOURCES
  src/abi.cpp
  src/context.cpp
  src/pipeline/registry.cpp
  src/batch/compose.cpp
  src/batch/forward.cpp)
if(NOT APPLE)
  list(APPEND PIE_METAL_EXPLICIT_SOURCES src/pipeline/shared_storage.cpp)
endif()

# Native Metal-4 compute pipeline and default backend.
# Keep this list explicit: standalone tools and not-yet-live architecture
# bring-up code must never enter the shipped archive by filename convention.
set(PIE_METAL_LIVE_SOURCES)
if(APPLE)
  set(PIE_METAL_LIVE_SOURCES
    src/mtl4_context.mm
    src/pipeline/shared_storage.mm
    src/pipeline/m1_codegen.cpp
    src/pipeline/m1_runtime.cpp
    src/batch/decode_timing.cpp
    src/batch/scratch.cpp
    src/model/qwen3_5/decode_consts.cpp
    src/model/qwen3_5/decode_step.cpp
    src/model/qwen3_5/decode_step_mb.cpp
    src/kernels/decode_psos.cpp
    src/loader/heap_bind.cpp
    src/loader/safetensors_view.cpp)
endif()

add_library(pie_driver_metal_lib STATIC
  ${PIE_METAL_EXPLICIT_SOURCES}
  ${PIE_METAL_LIVE_SOURCES})

if(APPLE)
  set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mtl4_context.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pipeline/shared_storage.mm
    PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
endif()

target_include_directories(pie_driver_metal_lib
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/batch
    ${CMAKE_CURRENT_SOURCE_DIR}/src/model/qwen3_5
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels
    ${CMAKE_CURRENT_SOURCE_DIR}/src/store
    ${CMAKE_CURRENT_SOURCE_DIR}/src/loader
    # ../common/include also carries the shared pure-host PTIR decode headers
    # (pie_native/ptir/{container,bound,trace,op_table,fire_geometry}.hpp —
    # CUDA-free, namespace pie_native::ptir). The host interpreter
    # (src/pipeline/interp.hpp) consumes them; Metal no longer reaches into
    # driver/cuda/src for anything.
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/include
    ${PIE_DRIVER_ABI_INCLUDE_DIR})

target_compile_options(pie_driver_metal_lib PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wno-unused-parameter>
  $<$<COMPILE_LANGUAGE:OBJCXX>:-Wall -Wextra -Wno-unused-parameter>)

# Compiled-in fallback for the native .metal kernel sources (runtime PSO
# compilation). Overridable at run time via PIE_METAL_KERNELS_DIR. PUBLIC so
# every target linking pie_driver_metal_lib (e.g. ptir_checkpoint_e2e_test,
# which drives an independent MetalExecutor::setup outside Context) sees
# the same default when PIE_METAL_KERNELS_DIR is unset. Installed packaging
# of the kernels dir is a follow-up.
if(APPLE)
  target_compile_definitions(pie_driver_metal_lib PUBLIC
    PIE_METAL_KERNELS_DIR_DEFAULT="${CMAKE_CURRENT_SOURCE_DIR}/src/kernels")
endif()

target_link_libraries(pie_driver_metal_lib PUBLIC
  CLI11::CLI11
  tomlplusplus::tomlplusplus
  nlohmann_json::nlohmann_json
  Threads::Threads)

# Apple frameworks MLX + native Metal shaders pull. Linked PUBLIC so both
# the standalone exe and the worker binary that embeds the static lib
# resolve them. (The worker build.rs also adds these via rustc-link-arg;
# duplicate -framework flags are harmless.)
if(APPLE)
  target_link_libraries(pie_driver_metal_lib PUBLIC
    "-framework Metal"
    "-framework MetalKit"
    "-framework Foundation"
    "-framework Accelerate")
endif()

# -----------------------------------------------------------------------------
# Native Metal shader compilation (optional, AOT)
#
# Default path is MLX JIT (`mlx::core::fast::metal_kernel`), where kernel
# source lives as a string in a `.cpp` and any `.metal` file is just a
# tooling mirror — nothing to compile here. Set -DPIE_METAL_BUILD_METALLIB=ON
# to instead AOT-compile every standalone-compilable `src/kernels/*.metal`
# into a default.metallib next to the binary.
# -----------------------------------------------------------------------------
option(PIE_METAL_BUILD_METALLIB "AOT-compile src/kernels/*.metal into default.metallib" OFF)

set(PIE_METAL_SHADER_NAMES
  argmax
  attn_gate
  bf16_gemv
  dense_gemv
  embed_gather
  gated_rms
  gdn_core
  gdn_prep
  geglu_tanh
  gemv_demo
  kv_append
  kv_append_paged
  layer_scalar
  logit_softcap
  ple_combine
  quantized_qmv
  residual_add
  rms_norm
  rope
  sdpa_paged
  sdpa_sliding
  sdpa_vector
  silu_mul
  vnorm)

if(PIE_METAL_BUILD_METALLIB)
  find_program(XCRUN_EXECUTABLE xcrun REQUIRED)
  set(PIE_METAL_AIR_FILES "")
  foreach(shader_name ${PIE_METAL_SHADER_NAMES})
    set(shader ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/${shader_name}.metal)
    set(air_file ${CMAKE_BINARY_DIR}/metal/${shader_name}.air)
    add_custom_command(
      OUTPUT ${air_file}
      COMMAND ${XCRUN_EXECUTABLE} -sdk macosx metal -c ${shader} -o ${air_file}
      DEPENDS ${shader}
      COMMENT "Compiling Metal shader ${shader_name}.metal")
    list(APPEND PIE_METAL_AIR_FILES ${air_file})
  endforeach()
  set(PIE_METAL_METALLIB ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib)
  add_custom_command(
    OUTPUT ${PIE_METAL_METALLIB}
    COMMAND ${XCRUN_EXECUTABLE} -sdk macosx metallib ${PIE_METAL_AIR_FILES}
            -o ${PIE_METAL_METALLIB}
    DEPENDS ${PIE_METAL_AIR_FILES}
    COMMENT "Linking default.metallib")
  add_custom_target(pie_driver_metal_shaders ALL DEPENDS ${PIE_METAL_METALLIB})
  add_dependencies(pie_driver_metal_lib pie_driver_metal_shaders)
endif()

# -----------------------------------------------------------------------------
# Standalone executable — thin shim around the lib's entry point. Consumed
# for build/config self-checks and local driver development.
# -----------------------------------------------------------------------------
add_executable(pie_driver_metal src/main.cpp)
target_link_libraries(pie_driver_metal PRIVATE pie_driver_metal_lib)

# The `--features driver-metal` build in worker/build.rs consumes
# `pie_driver_metal_lib` directly.

option(PIE_METAL_BUILD_TOOLS "Build native Metal bring-up and diagnostic tools" OFF)
if(PIE_METAL_BUILD_TOOLS AND APPLE)
  add_subdirectory(tools/rawmetal)
endif()

# -----------------------------------------------------------------------------
# Loader smoke test (delta-owned, off by default). Builds an end-to-end loader→
# graph→KV→forward check; requires MLX. Enable with -DPIE_METAL_BUILD_TESTS=ON.
# -----------------------------------------------------------------------------
option(PIE_METAL_BUILD_TESTS "Build driver/metal MLX reference tests" OFF)
option(PIE_METAL_BUILD_PARITY "Build driver/metal accuracy parity harness" OFF)
if(PIE_METAL_WITH_MLX AND (PIE_METAL_BUILD_TESTS OR PIE_METAL_BUILD_PARITY))
  add_subdirectory(tests)
endif()

option(PIE_METAL_BUILD_STUB_TESTS "Build Linux/non-MLX direct-surface tests" OFF)
if(PIE_METAL_BUILD_STUB_TESTS AND NOT PIE_METAL_WITH_MLX)
  add_executable(metal_direct_stub_test tests/direct_stub_test.cpp)
  target_link_libraries(metal_direct_stub_test PRIVATE pie_driver_metal_lib)
  add_test(NAME metal_direct_stub_test COMMAND metal_direct_stub_test)
  set_tests_properties(metal_direct_stub_test PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)
endif()

function(add_metal_host_test target source)
  add_executable(${target} ${source})
  target_include_directories(${target} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/include)
  add_test(NAME ${target} COMMAND ${target})
endfunction()

function(add_metal_linked_test target source)
  add_executable(${target} ${source})
  target_link_libraries(${target} PRIVATE pie_driver_metal_lib)
  add_test(NAME ${target} COMMAND ${target})
endfunction()

file(GLOB PIE_PTIR_CANONICAL_GOLDENS CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/../../interface/ptir/tests/golden-ptir/*.txt)
add_executable(
  ptir_decoder_limits_test
  ${CMAKE_CURRENT_SOURCE_DIR}/../common/tests/ptir_decoder_limits_test.cpp)
target_include_directories(ptir_decoder_limits_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../common/include)
target_compile_features(ptir_decoder_limits_test PRIVATE cxx_std_20)
add_test(
  NAME ptir_decoder_limits_test
  COMMAND ptir_decoder_limits_test
    ${CMAKE_CURRENT_SOURCE_DIR}/../../interface/ptir/tests/malformed_wire_corpus.txt
    ${PIE_PTIR_CANONICAL_GOLDENS})

add_metal_host_test(pipeline_interp_test tests/pipeline_interp_test.cpp)
add_metal_host_test(
  pipeline_channel_storage_test
  tests/pipeline_channel_storage_test.cpp)
add_metal_host_test(
  pipeline_descriptor_resolve_test
  tests/pipeline_descriptor_resolve_test.cpp)
add_metal_host_test(load_plan_decode_test tests/load_plan_decode_test.cpp)
target_link_libraries(load_plan_decode_test PRIVATE nlohmann_json::nlohmann_json)

add_executable(heap_layout_slot_test tests/heap_layout_slot_test.cpp)
target_include_directories(heap_layout_slot_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/src/batch
  ${CMAKE_CURRENT_SOURCE_DIR}/src/loader)
add_test(NAME heap_layout_slot_test COMMAND heap_layout_slot_test)

add_executable(paged_batch_validation_test tests/paged_batch_validation_test.cpp)
target_include_directories(paged_batch_validation_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/src/batch)
target_link_libraries(paged_batch_validation_test PRIVATE pie_driver_metal_lib)
add_test(NAME paged_batch_validation_test COMMAND paged_batch_validation_test)

add_metal_linked_test(
  pipeline_device_geometry_e2e_test
  tests/pipeline_device_geometry_e2e_test.cpp)
add_metal_linked_test(pipeline_copy_ops_test tests/pipeline_copy_ops_test.cpp)
add_metal_linked_test(caps_honesty_test tests/caps_honesty_test.cpp)
add_metal_linked_test(executor_geometry_test tests/executor_geometry_test.cpp)
add_metal_linked_test(batch_schedule_plan_test tests/batch_schedule_plan_test.cpp)

add_executable(executor_worker_test tests/executor_worker_test.cpp)
target_include_directories(executor_worker_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(executor_worker_test PRIVATE Threads::Threads)
add_test(NAME executor_worker_test COMMAND executor_worker_test)

if(APPLE)
  add_metal_linked_test(m1_generated_test tests/m1_generated_test.cpp)
  target_compile_definitions(m1_generated_test PRIVATE
    PIE_PTIR_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../interface/ptir/tests/golden-ptir")
  add_metal_linked_test(ptir_m0_device_test tests/ptir_m0_device_test.cpp)
  add_metal_linked_test(ptir_checkpoint_e2e_test tests/ptir_checkpoint_e2e_test.cpp)
  add_metal_linked_test(kv_append_paged_pso_test tests/kv_append_paged_pso_test.cpp)
  add_metal_linked_test(decode_step_mb_test tests/decode_step_mb_test.cpp)
  add_metal_linked_test(kv_pool_lifecycle_test tests/kv_pool_lifecycle_test.cpp)
endif()

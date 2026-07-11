cmake_minimum_required(VERSION 3.23)

if(APPLE)
  # driver/metal — MLX + native Metal backend for Pie (Apple Silicon, macOS).
  # Structure mirrors driver/cuda. OBJCXX is enabled so the compute layer can
  # add `.mm` (Objective-C++) translation units for Metal interop.
  project(PIE_DRIVER_METAL CXX OBJCXX)
else()
  # Linux/CI builds compile the direct-ABI stub surface (entry.cpp) so the
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
# GLOBbed (beta/charlie/delta-owned): every .cpp/.mm under the compute,
# executor, kernels, model, and loader trees — so peers add translation
# units WITHOUT editing this file. CONFIGURE_DEPENDS re-globs on rebuild.
# -----------------------------------------------------------------------------
# These compile WITHOUT MLX so the direct ABI stays available even when the
# optional MLX backend is disabled.
set(PIE_METAL_EXPLICIT_SOURCES
  src/entry.cpp
  src/executor/executor.cpp)

# GLOBbed (beta/charlie/delta-owned): every .cpp/.mm under the compute,
# kernels, model, and loader trees — so peers add translation
# units WITHOUT editing this file. CONFIGURE_DEPENDS re-globs on rebuild.
#
# These TUs all `#include` MLX (via ops/tensor.hpp), so they are ONLY compiled
# when PIE_METAL_WITH_MLX is ON.
set(PIE_METAL_GLOB_SOURCES)
if(PIE_METAL_WITH_MLX)
  file(GLOB_RECURSE PIE_METAL_GLOB_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ops/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ops/*.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/*.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/src/model/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/model/*.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/src/loader/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/loader/*.mm)
endif()

# raw_metal (alpha/beta/charlie/delta-owned): the MLX-free raw Metal-4 compute
# pipeline — the DEFAULT forward backend baked into the worker-embedded lib.
# GLOB the compute TUs so peers add .cpp/.mm WITHOUT editing this file, then
# EXCLUDE the standalone driver/bench/probe mains (each carries its own
# `int main()` and must not link into the lib). Convention: any new standalone
# executable TU must match the exclude regex below (`*_run.cpp`, `*_probe.cpp`,
# `harness*.cpp`, `bench_*.cpp`).
set(PIE_METAL_RAW_SOURCES)
if(APPLE)
  file(GLOB PIE_METAL_RAW_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/raw_metal/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/raw_metal/*.mm)
  list(FILTER PIE_METAL_RAW_SOURCES EXCLUDE REGEX
    "/(.*_run|.*_probe|.*_smoke|harness.*|bench_.*)\\.cpp$")
endif()

add_library(pie_driver_metal_lib STATIC
  ${PIE_METAL_EXPLICIT_SOURCES}
  ${PIE_METAL_GLOB_SOURCES}
  ${PIE_METAL_RAW_SOURCES})

if(APPLE)
  # raw_metal's Objective-C++ TU (mtl4_context.mm) needs ARC.
  set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/raw_metal/mtl4_context.mm
    PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
endif()

target_include_directories(pie_driver_metal_lib
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/raw_metal
    # ../common/include also carries the shared pure-host PTIR decode headers
    # (pie_native/ptir/{container,bound,trace,op_table,fire_geometry}.hpp —
    # CUDA-free, namespace pie_native::ptir). The host interpreter
    # (src/ptir/host_interp.hpp) consumes them; Metal no longer reaches into
    # driver/cuda/src for anything.
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/include
    ${PIE_DRIVER_ABI_INCLUDE_DIR})

target_compile_options(pie_driver_metal_lib PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wno-unused-parameter>
  $<$<COMPILE_LANGUAGE:OBJCXX>:-Wall -Wextra -Wno-unused-parameter>)

# Compiled-in fallback for the raw-Metal .metal kernel sources (runtime PSO
# compilation). Overridable at run time via PIE_METAL_KERNELS_DIR. PUBLIC so
# every target linking pie_driver_metal_lib (e.g. ptir_checkpoint_e2e_test,
# which drives an INDEPENDENT RawMetalDecoder::setup outside entry.cpp) sees
# the same default when PIE_METAL_KERNELS_DIR is unset. Installed packaging
# of the kernels dir is a follow-up.
if(APPLE)
  target_compile_definitions(pie_driver_metal_lib PUBLIC
    PIE_METAL_KERNELS_DIR_DEFAULT="${CMAKE_CURRENT_SOURCE_DIR}/src/raw_metal/kernels")
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

if(PIE_METAL_WITH_MLX)
  include(cmake/FetchMLX.cmake)
  pie_metal_fetch_mlx(pie_driver_metal_lib)
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

file(GLOB_RECURSE PIE_METAL_SHADER_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/*.metal)

if(PIE_METAL_BUILD_METALLIB AND PIE_METAL_SHADER_SOURCES)
  find_program(XCRUN_EXECUTABLE xcrun REQUIRED)
  set(PIE_METAL_AIR_FILES "")
  foreach(shader ${PIE_METAL_SHADER_SOURCES})
    get_filename_component(shader_name ${shader} NAME_WE)
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

# -----------------------------------------------------------------------------
# Loader smoke test (delta-owned, off by default). Builds an end-to-end loader→
# graph→KV→forward check; requires MLX. Enable with -DPIE_METAL_BUILD_TESTS=ON.
# -----------------------------------------------------------------------------
option(PIE_METAL_BUILD_TESTS "Build driver/metal loader smoke tests" OFF)
if(PIE_METAL_BUILD_TESTS AND PIE_METAL_WITH_MLX)
  add_subdirectory(tests)
endif()

option(PIE_METAL_BUILD_STUB_TESTS "Build Linux/non-MLX direct-surface tests" OFF)
if(PIE_METAL_BUILD_STUB_TESTS AND NOT PIE_METAL_WITH_MLX)
  enable_testing()
  add_executable(metal_direct_stub_test tests/direct_stub_test.cpp)
  target_link_libraries(metal_direct_stub_test PRIVATE pie_driver_metal_lib)
  add_test(NAME metal_direct_stub_test COMMAND metal_direct_stub_test)
  set_tests_properties(metal_direct_stub_test PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)
endif()

# -----------------------------------------------------------------------------
# PTIR host-interpreter conformance + classification gate (metal_ptir_plan.md
# Phase 1, G1.1/G1.5). Pure host (no Metal/Apple dependency, no checkpoint) —
# always built + run, on every platform, independent of
# PIE_METAL_BUILD_STUB_TESTS / PIE_METAL_WITH_MLX.
# -----------------------------------------------------------------------------
enable_testing()
add_executable(ptir_host_interp_test tests/ptir_host_interp_test.cpp)
target_include_directories(ptir_host_interp_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/../common/include)
add_test(NAME ptir_host_interp_test COMMAND ptir_host_interp_test)

# -----------------------------------------------------------------------------
# PTIR descriptor-resolver pure contract gate (metal_ptir_plan.md Phase 2,
# G2.1-style). Pure host (no Metal/Apple dependency, no checkpoint) — always
# built + run, on every platform.
# -----------------------------------------------------------------------------
add_executable(ptir_descriptor_resolve_test tests/ptir_descriptor_resolve_test.cpp)
target_include_directories(ptir_descriptor_resolve_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/../common/include)
add_test(NAME ptir_descriptor_resolve_test COMMAND ptir_descriptor_resolve_test)

# -----------------------------------------------------------------------------
# heap_layout.hpp plan_heap max_slots invariance gate (Phase 1b review-fix:
# copy_state's real multi-slot GDN state region). Pure host (no Metal/Apple
# dependency, no checkpoint) — always built + run, on every platform.
# -----------------------------------------------------------------------------
add_executable(heap_layout_slot_test tests/heap_layout_slot_test.cpp)
target_include_directories(heap_layout_slot_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/raw_metal)
add_test(NAME heap_layout_slot_test COMMAND heap_layout_slot_test)

add_executable(paged_batch_validation_test tests/paged_batch_validation_test.cpp)
target_include_directories(paged_batch_validation_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/raw_metal)
add_test(NAME paged_batch_validation_test COMMAND paged_batch_validation_test)

# -----------------------------------------------------------------------------
# Device-geometry (Phase 2, C3) driver-level poison/settlement gate
# (metal_ptir_plan.md §9 G2.3-style). Drives the full pie_metal_* ABI surface
# via pie_driver_metal_lib; the not-ready descriptor channel fails before
# MetalExecutor is ever touched, so no checkpoint/Apple/Metal dependency is
# needed — always built + run, on every platform.
# -----------------------------------------------------------------------------
add_executable(ptir_device_geometry_e2e_test tests/ptir_device_geometry_e2e_test.cpp)
target_link_libraries(ptir_device_geometry_e2e_test PRIVATE pie_driver_metal_lib)
add_test(NAME ptir_device_geometry_e2e_test COMMAND ptir_device_geometry_e2e_test)

# -----------------------------------------------------------------------------
# Phase 1b copy_kv/copy_state/resize_pool driver-level ABI gate. Every path
# exercised fails before any real Metal work would happen (no GDN layers, or
# a deliberately incomplete fixture checkpoint dir), so no checkpoint/Apple/
# Metal dependency is needed — always built + run, on every platform.
# -----------------------------------------------------------------------------
add_executable(ptir_copy_ops_test tests/ptir_copy_ops_test.cpp)
target_link_libraries(ptir_copy_ops_test PRIVATE pie_driver_metal_lib)
add_test(NAME ptir_copy_ops_test COMMAND ptir_copy_ops_test)

# -----------------------------------------------------------------------------
# Truthful caps gate (metal_ptir_plan.md §12 "Caps honesty" / §5.5). Exercises
# the full `pie_metal_create` ABI entry point (config.json parsing only — no
# checkpoint load, no Apple/Metal dependency), so it always builds and runs.
# -----------------------------------------------------------------------------
add_executable(caps_honesty_test tests/caps_honesty_test.cpp)
target_link_libraries(caps_honesty_test PRIVATE pie_driver_metal_lib)
add_test(NAME caps_honesty_test COMMAND caps_honesty_test)

# -----------------------------------------------------------------------------
# Phase 1a linear-sequence geometry gate (metal_ptir_plan.md §5.4) — pure
# `validate_linear_sequence_geometry` unit test. Links the full
# pie_driver_metal_lib (executor.cpp's Apple-only RawMetalDecoder path needs
# raw_metal in the same link on Apple; the function under test itself has no
# Metal dependency on either platform).
# -----------------------------------------------------------------------------
add_executable(executor_geometry_test tests/executor_geometry_test.cpp)
target_link_libraries(executor_geometry_test PRIVATE pie_driver_metal_lib)
add_test(NAME executor_geometry_test COMMAND executor_geometry_test)

# -----------------------------------------------------------------------------
# Phase 3 batch-scheduler gate (metal_ptir_plan.md §7) — pure
# `plan_batch_execution` unit test (ring-safe member ordering + honest gating
# of concurrent multi-sequence decode / explicit KV writes). Links the full
# pie_driver_metal_lib (the scheduler itself has no Metal dependency on either
# platform). Always built + run.
# -----------------------------------------------------------------------------
add_executable(batch_schedule_plan_test tests/batch_schedule_plan_test.cpp)
target_link_libraries(batch_schedule_plan_test PRIVATE pie_driver_metal_lib)
add_test(NAME batch_schedule_plan_test COMMAND batch_schedule_plan_test)

# -----------------------------------------------------------------------------
# Phase 3 executor-worker gate (metal_ptir_plan.md §7, D4) — single-owner
# serializer for all RawMetalDecoder/RawMetalContext use (thread-affinity,
# FIFO order, re-entrancy, serialization under concurrent submitters). Pure
# std::thread (no Metal/Apple/checkpoint dependency) — always built + run.
# -----------------------------------------------------------------------------
add_executable(executor_worker_test tests/executor_worker_test.cpp)
target_include_directories(executor_worker_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(executor_worker_test PRIVATE Threads::Threads)
add_test(NAME executor_worker_test COMMAND executor_worker_test)

# -----------------------------------------------------------------------------
# Checkpoint-gated Metal PTIR driver e2e (metal_ptir_plan.md Phase 1, G1.2).
# Apple-only (needs RawMetalDecoder / real Metal); checkpoint-gated at RUN
# time via PIE_METAL_CKPT (prints SKIP + exits 0 when unset), so it always
# builds on a Mac and stays green in CI without a bundled checkpoint fixture.
# -----------------------------------------------------------------------------
if(APPLE)
  add_executable(ptir_checkpoint_e2e_test tests/ptir_checkpoint_e2e_test.cpp)
  target_link_libraries(ptir_checkpoint_e2e_test PRIVATE pie_driver_metal_lib)
  add_test(NAME ptir_checkpoint_e2e_test COMMAND ptir_checkpoint_e2e_test)
endif()

# -----------------------------------------------------------------------------
# Phase 1b/3 paged-KV bridge multi-batch PSO compile gate. REAL Metal shader
# compilation (runtime-compiled, no checkpoint needed — just kernels/*.metal
# + a Metal device, both present on any Mac this builds on). Apple-only
# (RawMetalContext/load_multibatch_psos only exist on Apple). Always builds
# and runs on a Mac, independent of PIE_METAL_CKPT.
# -----------------------------------------------------------------------------
if(APPLE)
  add_executable(kv_append_paged_pso_test tests/kv_append_paged_pso_test.cpp)
  target_link_libraries(kv_append_paged_pso_test PRIVATE pie_driver_metal_lib)
  add_test(NAME kv_append_paged_pso_test COMMAND kv_append_paged_pso_test)

  add_executable(decode_step_mb_test tests/decode_step_mb_test.cpp)
  target_link_libraries(decode_step_mb_test PRIVATE pie_driver_metal_lib)
  add_test(NAME decode_step_mb_test COMMAND decode_step_mb_test)
endif()

# -----------------------------------------------------------------------------
# Phase 3 (review item 4) standalone-buffer allocation lifecycle gate. REAL
# Metal allocation via RawMetalContext (Metal device only — NO checkpoint),
# proving create/release keep an exact count/bytes and a grow/shrink cycle
# stays bounded (no per-resize leak). Apple-only; always builds + runs on a Mac.
# -----------------------------------------------------------------------------
if(APPLE)
  add_executable(kv_pool_lifecycle_test tests/kv_pool_lifecycle_test.cpp)
  target_link_libraries(kv_pool_lifecycle_test PRIVATE pie_driver_metal_lib)
  add_test(NAME kv_pool_lifecycle_test COMMAND kv_pool_lifecycle_test)
endif()

# -----------------------------------------------------------------------------
# Accuracy parity harness (alpha-owned, off by default). Builds `parity_driver`
# (driver logits dumper) for tools/parity/parity_check.py to diff against
# mlx-lm. Requires MLX. Enable with -DPIE_METAL_BUILD_PARITY=ON.
# -----------------------------------------------------------------------------
option(PIE_METAL_BUILD_PARITY "Build driver/metal accuracy parity harness" OFF)
if(PIE_METAL_BUILD_PARITY AND PIE_METAL_WITH_MLX)
  add_subdirectory(tools/parity)
endif()

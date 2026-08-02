cmake_minimum_required(VERSION 3.23)

include(cmake/DetectCudaArchitecture.cmake)
detect_cuda_architectures()
message(STATUS "Using CUDA architecture: ${CMAKE_CUDA_ARCHITECTURES}")

# nvcc discovery: some toolkits install CUDA under /usr/local/cuda without
# putting bin/ on PATH, so workspace builds work without sourcing CUDA's env
# script first.
if(NOT CMAKE_CUDA_COMPILER
   AND NOT DEFINED ENV{CUDACXX}
   AND NOT DEFINED ENV{CMAKE_CUDA_COMPILER})
  find_program(PIE_NVCC nvcc
    HINTS
      /usr/local/cuda/bin
      /usr/local/cuda-13/bin
      /usr/local/cuda-13.0/bin
      /usr/local/cuda-12/bin
      /usr/local/cuda-12.8/bin
      /opt/cuda/bin)
  if(PIE_NVCC)
    set(CMAKE_CUDA_COMPILER "${PIE_NVCC}")
  endif()
endif()

project(PIE_DRIVER_CUDA CUDA CXX)


# ── Where the rest of the repo is ────────────────────────────────────
#
# Every one of these is passed by this driver crate's `build.rs`, which gets
# them from cargo (`links` + `cargo:include`, or its own manifest dir). The
# fallbacks exist so `cmake -S csrc` still configures by hand; they are the
# second answer to a question cargo already answers, so they are written ONCE
# here rather than spelled out at thirty use sites.
if(NOT PIE_DRIVER_INCLUDE_DIR)
  set(PIE_DRIVER_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../driver/include")
endif()
if(NOT PIE_PTIR_INCLUDE_DIR)
  set(PIE_PTIR_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../tensor-compiler/include")
endif()
if(NOT PIE_PTIR_RUNTIME_DIR)
  set(PIE_PTIR_RUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../tensor-compiler/runtime")
endif()
if(NOT PIE_REPO_ROOT)
  set(PIE_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
endif()
# The compiler->driver conformance images. Generated, never committed.
if(NOT PIE_DRIVER_FIXTURES_DIR)
  set(PIE_DRIVER_FIXTURES_DIR "${PIE_REPO_ROOT}/tests/driver-fixtures")
endif()

# Optional compiler cache; speeds clean rebuilds / branch switches.
# Host C/C++ accepts either sccache or ccache.
if(NOT CMAKE_C_COMPILER_LAUNCHER AND NOT CMAKE_CXX_COMPILER_LAUNCHER)
  if(DEFINED ENV{PIE_COMPILER_LAUNCHER} AND NOT "$ENV{PIE_COMPILER_LAUNCHER}" STREQUAL "")
    set(PIE_COMPILER_LAUNCHER "$ENV{PIE_COMPILER_LAUNCHER}")
  else()
    find_program(PIE_COMPILER_LAUNCHER NAMES sccache ccache)
  endif()
  if(PIE_COMPILER_LAUNCHER)
    set(CMAKE_C_COMPILER_LAUNCHER "${PIE_COMPILER_LAUNCHER}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${PIE_COMPILER_LAUNCHER}")
    message(STATUS "Using compiler launcher (C/C++): ${PIE_COMPILER_LAUNCHER}")
  endif()
endif()

# nvcc is launched through ccache ONLY. sccache mishandles nvcc's fatbinary
# temp files (intermittent SIGHUP-killed server); ccache 4.x handles nvcc
# correctly. Caching .cu matters far more than caching host C/C++: .cu
# objects are ~95% of this build's CPU time, so a host-only cache leaves
# nearly all of a rebuild uncached. Set -DCMAKE_CUDA_COMPILER_LAUNCHER=
# explicitly to opt out or to force a different launcher.
if(NOT DEFINED CMAKE_CUDA_COMPILER_LAUNCHER)
  find_program(PIE_CUDA_COMPILER_LAUNCHER NAMES ccache)
  if(PIE_CUDA_COMPILER_LAUNCHER)
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${PIE_CUDA_COMPILER_LAUNCHER}")
    message(STATUS "Using CUDA compiler launcher: ${PIE_CUDA_COMPILER_LAUNCHER}")
  else()
    message(STATUS "ccache not found; nvcc compiles will not be cached")
  endif()
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING
    "Build type (Debug, Release, RelWithDebInfo, MinSizeRel)" FORCE)
endif()

# ---- the kernel archive ----------------------------------------------------
#
# `src/kernels/`, `src/ops/`, kernels.def, the vendored FlashInfer/marlin
# trees and the three substrate headers that go with them are
# `crates/kernels-cuda` now. This file no longer reads kernels.def (the
# head_dim translation units and the MoE activation gate moved with it), no
# longer fetches or patches FlashInfer, and no longer compiles a `.cu` that
# is not part of the shell.
#
# What it needs from over there arrives as CMake variables, set by this
# crate's build.rs out of cargo's `DEP_PIE_KERNELS_CUDA_*` handoff. The
# fallbacks let `cmake -S csrc` still configure by hand, but they can only
# guess the source tree -- FlashInfer's fetch directory is inside the kernels
# crate's build directory and has no stable path -- so a by-hand configure
# gets the kernel headers and not FlashInfer's.
if(NOT PIE_KERNELS_CUDA_SRC)
  set(PIE_KERNELS_CUDA_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../../kernels-cuda/csrc/src")
endif()
# Every include dir the archive was built with: its own src/, FlashInfer's
# headers, and the CCCL that must precede the toolkit's.
list(APPEND PIE_KERNELS_CUDA_INCLUDE_DIRS "${PIE_KERNELS_CUDA_SRC}")
list(REMOVE_DUPLICATES PIE_KERNELS_CUDA_INCLUDE_DIRS)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
# No Python interpreter required to build: FlashInfer's CUTLASS MoE launchers
# are vendored (third_party/flashinfer_generated) and flashinfer itself is
# fetched DOWNLOAD_ONLY for headers.

if(NOT PIE_DRIVER_ABI_INCLUDE_DIR)
  set(PIE_DRIVER_ABI_INCLUDE_DIR "${PIE_REPO_ROOT}/crates/driver-abi/include")
endif()

# `pie_loader.h` is generated by `cargo run -p pie-loader-cbindgen` and
# committed, so a standalone CMake configure works without a Cargo build.
if(NOT PIE_LOADER_INCLUDE_DIR)
  set(PIE_LOADER_INCLUDE_DIR "${PIE_REPO_ROOT}/crates/model-loader-capi/include")
endif()

# NCCL for tensor-parallel collectives. Single-GPU runs never load it (call
# sites gate on tp_size > 1) but we always link so a TP-enabled wrapper can
# talk to the same binary. System NCCL only -- vendoring via CPM is
# impractical (kernel-heavy build).
find_path(NCCL_INCLUDE_DIR nccl.h
  HINTS ${CUDAToolkit_INCLUDE_DIRS} /usr/include /usr/local/include
  REQUIRED)
find_library(NCCL_LIBRARY nccl
  HINTS ${CUDAToolkit_LIBRARY_DIR} /usr/lib/x86_64-linux-gnu /usr/local/lib
  REQUIRED)
message(STATUS "Using NCCL: ${NCCL_LIBRARY} (header: ${NCCL_INCLUDE_DIR})")

# CPM.cmake and DetectCudaArchitecture.cmake are vendored in BOTH csrc trees,
# byte-identical. That is what vendored bootstrap scripts are: each CMake
# project has to be able to configure on its own, and neither can `include()`
# out of the other's source tree without reintroducing exactly the cross-crate
# relative path this split removed. Update them together.
include(cmake/CPM.cmake)

# Persistent CPM cache when set, so incremental builds don't re-clone from
# GitHub. CI sets this via GITHUB_ENV; local builds can export it in .bashrc.
if(DEFINED ENV{CPM_SOURCE_CACHE} AND NOT CPM_SOURCE_CACHE)
  set(CPM_SOURCE_CACHE $ENV{CPM_SOURCE_CACHE})
endif()
CPMAddPackage(URI "gh:marzer/tomlplusplus@3.4.0")
CPMAddPackage(URI "gh:CLIUtils/CLI11@2.5.0")
CPMAddPackage(URI "gh:nlohmann/json@3.12.0" OPTIONS "JSON_BuildTests OFF")

# ---- Helper functions ------------------------------------------------------

# True in <out_var> if CMAKE_CUDA_ARCHITECTURES contains an entry whose
# numeric SM version equals <arch_number> (suffixes like "-real" ignored).
function(pie_cuda_has_arch out_var arch_number)
  set(_found FALSE)
  foreach(_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
    string(REGEX REPLACE "[^0-9].*$" "" _arch_num "${_arch}")
    if(_arch_num STREQUAL "${arch_number}")
      set(_found TRUE)
    endif()
  endforeach()
  set(${out_var} "${_found}" PARENT_SCOPE)
endfunction()

# Adds a driver test/harness executable: NAME SOURCES <src...>
# [INCLUDE_DIRS <dir...>] [LIBS <lib...>] [NO_CTEST | TEST_NAME <name>]
# [TEST_ARGS <arg...>] [WORKING_DIRECTORY <dir>]. Always gets src/ on its
# include path and the standard nvcc relaxed-constexpr/extended-lambda flags
# (a no-op generator expression for pure-C++ sources). Registers a ctest
# named NAME with a leading "test_" stripped, unless TEST_NAME overrides it
# or NO_CTEST marks a build-only manual harness.
function(pie_add_cuda_target NAME)
  cmake_parse_arguments(ARG "NO_CTEST" "TEST_NAME;WORKING_DIRECTORY"
    "SOURCES;INCLUDE_DIRS;LIBS;TEST_ARGS" ${ARGN})
  add_executable(${NAME} ${ARG_SOURCES})
  target_include_directories(${NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src ${ARG_INCLUDE_DIRS})
  target_compile_options(${NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>)
  if(ARG_LIBS)
    target_link_libraries(${NAME} PRIVATE ${ARG_LIBS})
  endif()
  if(NOT ARG_NO_CTEST)
    if(NOT ARG_TEST_NAME)
      string(REGEX REPLACE "^test_" "" ARG_TEST_NAME ${NAME})
    endif()
    add_test(NAME ${ARG_TEST_NAME} COMMAND ${NAME} ${ARG_TEST_ARGS})
    if(ARG_WORKING_DIRECTORY)
      set_tests_properties(${ARG_TEST_NAME} PROPERTIES
        WORKING_DIRECTORY ${ARG_WORKING_DIRECTORY})
    endif()
  endif()
endfunction()

# ---- Architecture-gated source selection -----------------------------------

pie_cuda_has_arch(PIE_HAS_SM90 90)
pie_cuda_has_arch(PIE_HAS_SM100 100)
pie_cuda_has_arch(PIE_HAS_SM120 120)

# Blackwell (sm100/120) has no custom-all-reduce kernel yet; stub it out.
if(PIE_HAS_SM100 OR PIE_HAS_SM120)
  set(PIE_CUDA_CUSTOM_ALL_REDUCE_SOURCE src/comm/custom_all_reduce_stub.cpp)
  message(STATUS "Disabling FlashInfer custom all-reduce for CUDA architecture ${CMAKE_CUDA_ARCHITECTURES}")
else()
  set(PIE_CUDA_CUSTOM_ALL_REDUCE_SOURCE src/comm/custom_all_reduce.cu)
endif()


# The graph-heavy model implementations. The registry (src/model/registry.cpp)
# binds these by architecture name at load time. The kernels they launch used
# to be listed here too; they are in the kernels crate's archive now.
set(PIE_CUDA_MODEL_SOURCES
  src/model/attn_page_mask.cu
  src/model/attn_score.cu
  src/model/descriptor.cpp
  src/model/csm/csm.cpp
  src/model/csm/csm_backbone_forward.cu
  src/model/csm/csm_depth_decoder_forward.cu
  src/model/csm/csm_model.cpp
  src/model/csm/mimi_decoder_adapter.cpp
  src/model/csm/mimi_decoder_forward.cu
  src/model/deepseek_v4/deepseek_v4.cpp
  src/model/deepseek_v4/deepseek_v4_forward.cpp
  src/model/deepseek_v4/deepseek_v4_model.cpp
  src/model/gemma/gemma2.cpp
  src/model/gemma/gemma2_model.cpp
  src/model/gemma3n/gemma3n.cpp
  src/model/gemma3n/gemma3n_model.cpp
  src/model/gemma4/gemma4.cpp
  src/model/gemma4/gemma4_audio_adapter.cpp
  src/model/gemma4/gemma4_audio_forward.cu
  src/model/gemma4/gemma4_model.cpp
  src/model/gemma4/gemma4_vision_adapter.cpp
  src/model/gemma4/gemma4_vision_forward.cu
  src/model/glm5/glm5.cpp
  src/model/glm5/glm5_forward.cpp
  src/model/glm5/glm5_model.cpp
  src/model/kimi_k3/kimi_k3.cpp
  src/model/kimi_k3/kimi_k3_forward.cpp
  src/model/kimi_k3/kimi_k3_model.cpp
  src/model/kimi/kimi.cpp
  src/model/kimi/kimi_forward.cpp
  src/model/kimi/kimi_model.cpp
  src/model/llama_like/llama_like.cpp
  src/model/llama_like/declared_forward.cpp
  src/model/hook_sideband_arena.cpp
  src/batch/supergraph.cu
  src/model/llama_like/llama_like_model.cpp
  src/model/llama_like/mistral3.cpp
  src/model/llama_like/qwen3.cpp
  src/model/loaded_model.cpp
  src/model/mixtral/gpt_oss.cpp
  src/model/mixtral/mixtral.cpp
  src/model/mixtral/mixtral_model.cpp
  src/model/nemotron_h/nemotron_h.cpp
  src/model/nemotron_h/nemotron_h_forward.cpp
  src/model/nemotron_h/nemotron_h_model.cpp
  src/model/gemma4/declared_facts.cpp
  src/model/gemma4/declared_forward.cpp
  src/model/mixtral/declared_facts.cpp
  src/model/mixtral/declared_forward.cpp
  src/model/qwen3_5/declared_facts.cpp
  src/model/qwen3_5/declared_forward.cpp
  src/model/qwen3_5/qwen3_5.cpp
  src/model/qwen3_5/qwen3_5_config.cpp
  src/model/qwen3_5/qwen3_5_forward.cpp
  src/model/qwen3_5/qwen3_5_model.cpp
  src/model/qwen3_5/qwen3_5_moe.cpp
  src/model/qwen3_5/qwen3_5_moe_forward.cpp
  src/model/qwen3_5/qwen3_5_moe_model.cpp
  src/model/qwen3_vl/qwen3_vl.cpp
  src/model/qwen3_vl/qwen3_vl_model.cpp
  src/model/qwen3_vl/qwen3_vl_vision_adapter.cpp
  src/model/qwen3_vl/qwen3_vl_vision_forward.cu
  src/model/registry.cpp
  src/model/weight_store.cpp
  src/model/workspace.cpp
  src/store/dsa_cache.cpp
  src/store/dsv4_compress_cache.cpp
  src/store/mla_cache.cpp)

# ---- pie_driver_cuda_lib ---------------------------------------------------
# Native CUDA backend: the shell. `pie_driver_cuda_lib` is linked in-process by
# the Rust worker; `pie_driver_cuda` below is a thin direct-create diagnostic
# shim. Every production translation unit under src/ is listed exactly once,
# grouped to match the module tree.

add_library(pie_driver_cuda_lib STATIC
  src/abi.cpp
  src/context.cpp
  src/distributed.cpp
  # comm/ -- the TP collective's device half, which is not a compute kernel:
  # it sits on the NCCL/CUDA-IPC comm `distributed.cpp` owns, and that is why
  # it did not go to the kernels crate with the rest of src/kernels.
  ${PIE_CUDA_CUSTOM_ALL_REDUCE_SOURCE}
  # batch/ -- one submitted batch -> device work
  src/batch/brle.cpp
  src/batch/frame.cpp
  src/batch/forward.cpp
  src/batch/persistent_inputs.cpp
  src/batch/planner_calibration.cpp
  src/batch/tp.cpp
  src/batch/workspace.cu
  # pipeline/ -- guest programs (PTIR dispatch, tier0, frame carrier)
  src/pipeline/dispatch.cu
  src/pipeline/frame_carrier.cpp
  src/pipeline/frame_carrier_kernels.cu
  src/pipeline/registry.cpp
  # model/ -- registry, workspace, weight store, HF config, selected families
  ${PIE_CUDA_MODEL_SOURCES}
  # store/ -- long-lived device memory pools
  src/store/kv_cache.cpp
  src/store/kv_cache_format.cpp
  src/store/elastic.cpp
  src/store/memory_planner.cpp
  src/store/planner_profile_cache.cpp
  src/store/recurrent_state_cache.cpp
  src/store/swap_pool.cpp
)

target_compile_options(pie_driver_cuda_lib PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    -Xcompiler=-Wall
    -Xcompiler=-Wextra
    -Wno-unused-parameter
    --extended-lambda
    --expt-relaxed-constexpr
  >
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wno-unused-parameter>
)

target_include_directories(pie_driver_cuda_lib
  PUBLIC
    ${PIE_DRIVER_INCLUDE_DIR} ${PIE_PTIR_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    # tart: the declared-forward FFI headers (pie_forward/*.hpp). Taken from
    # the variable `worker/build.rs` passes, not by walking out of the driver
    # tree: the crate that publishes these headers is located by cargo (its
    # `links = "pie_forward"` + cargo:include), so a relative path here is a
    # second, silently-stale answer to a question cargo already answers.
    ${PIE_FORWARD_INCLUDE_DIR}
    ${PIE_DRIVER_ABI_INCLUDE_DIR} ${PIE_LOADER_INCLUDE_DIR}
    # The kernel declarations, plus the FlashInfer and CCCL trees the kernels
    # crate fetched. `batch/workspace.cu` calls FlashInfer's scheduler
    # directly, so this is not merely a convenience; and CCCL has to precede
    # anything the toolkit ships, which is why it comes over as one ordered
    # list rather than being rebuilt here.
    ${PIE_KERNELS_CUDA_INCLUDE_DIRS}
    # Ahead of NCCL for the same ordering reason.
    ${NCCL_INCLUDE_DIR})

# nvcc cannot parse toml++ 3.4.0's bool-returning for_each detection. The
# library-provided fallback preserves ordinary traversal in CUDA translation
# units; host C++ builds retain early-exit support.
target_compile_definitions(pie_driver_cuda_lib PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:TOML_RETURN_BOOL_FROM_FOR_EACH_BROKEN=1>
  $<$<COMPILE_LANGUAGE:CUDA>:TOML_RETURN_BOOL_FROM_FOR_EACH_BROKEN_ACKNOWLEDGED=1>)

# How a driver reads a config file, talks to its peers, and reaches the device.
#
# `${PIE_KERNELS_CUDA_LIB}` is a PATH, not a CMake target: the kernel archive
# is built by a separate CMake project (a separate cargo crate), and a target
# name does not cross that boundary. build.rs passes the path the kernels
# build produced. It comes after nothing here and before nothing here --
# ordering against `pie_driver_cuda_lib` is settled where the final link
# happens, which for the worker is rustc and for the C++ harnesses below is
# their own `target_link_libraries` line.
target_link_libraries(pie_driver_cuda_lib PUBLIC
  ${PIE_KERNELS_CUDA_LIB}
  CLI11::CLI11
  tomlplusplus::tomlplusplus
  nlohmann_json::nlohmann_json
  rt
  dl
  CUDA::cudart
  CUDA::cublas
  CUDA::cublasLt
  CUDA::cuda_driver
  CUDA::nvrtc
  ${NCCL_LIBRARY}
)
# ---- the loader archive ----------------------------------------------------
#
# Authoring a contract reads the checkpoint's tensor table, which means calling
# into the loader's C ABI — `pie-loader-capi`, an rlib the *worker* links, not
# something a C++ binary can pull in. `cargo rustc -p pie-loader-capi --lib
# --crate-type staticlib` emits an archive that does link. Anything reaching
# `pie_driver_cuda_lib`'s `loaded_model.cpp` needs it. When it is there those
# targets are built; when it is not they are skipped, because failing the whole
# build for everyone who has not run that command would be worse than not
# building a diagnostic tool and two structural tests.
#
# The name is `pie_loader_capi` and NOT `pie_loader`: the C entry points moved
# out of the compiler crate when authoring did, so a `-p pie-loader` staticlib
# still builds and exports none of the eighteen `pie_loader_*` symbols.
# Searching for the old name finds that archive and fails at link instead of
# skipping.
if(NOT PIE_LOADER_STATICLIB)
  find_library(PIE_LOADER_STATICLIB
    NAMES pie_loader_capi
    PATHS ${PIE_REPO_ROOT}/target/debug
          ${PIE_REPO_ROOT}/target/release
    NO_DEFAULT_PATH)
endif()

# ---- pie_driver_cuda (standalone diagnostic executable) --------------------

if(PIE_LOADER_STATICLIB)
  add_executable(pie_driver_cuda src/main.cpp)
  target_link_libraries(pie_driver_cuda PRIVATE pie_driver_cuda_lib
    ${PIE_LOADER_STATICLIB} dl pthread)
else()
  message(STATUS
    "pie_driver_cuda: skipped (no libpie_loader_capi.a; build one with "
    "`cargo rustc -p pie-loader-capi --lib --crate-type staticlib`)")
endif()


# ---- Tests and harnesses ----------------------------------------------------
# Seven of these compile a single kernel translation unit directly, and now
# reach across to ${PIE_KERNELS_CUDA_SRC} to do it. A kernel's unit test
# belongs in the kernels crate beside the kernel; they stayed because moving
# them also means moving `pie_add_cuda_target`, the ctest registration and the
# fixtures directory, and that is a second change rather than part of this one.
enable_testing()

# One rule for test targets: link pie_driver_cuda_lib unless the test needs
# exactly one src/ file. A target naming two or more src/ files is not
# expressing a boundary, it is hand-maintaining a transitive closure, and that
# closure goes stale the moment a listed file grows a new call edge -- an
# undefined-symbol link error that nothing in CI catches, because ci.yml only
# builds two tests. Single-file targets keep their explicit source: there the
# list *is* the unit under test, so it carries real isolation.
#
# Three targets are exempt and say why at their definition: csm_backbone_parity
# and csm_generate_parity (they #include the forward under test, so linking the
# lib would duplicate those symbols) and test_ptir_grouped_dispatch_asan (it
# must compile its own -fsanitize=address copies of what it exercises).

# Runtime fixtures/scripts consumed by tests but not compiled: the load-parity
# harness (tests/load_parity/run.py, driven manually against a live GPU +
# recipe), the qwen3.5 multi-request parity script, and the
# tier0/tier1/masked-attention host reference headers
# (#include-only, no standalone translation unit). Listed as a SOURCES-only
# custom target so IDE-generating generators surface them in the project tree
# and so a reader can tell "not compiled" from "forgotten".
add_custom_target(pie_driver_cuda_test_fixtures SOURCES
  tests/support/host_eval.hpp
  tests/support/tier1_codegen.hpp
  tests/masked_attention_reference.hpp
  tests/parity_qwen3_5_multireq.py
  tests/load_parity/dtypes.py
  tests/load_parity/gen.py
  tests/load_parity/oracle.py
  tests/load_parity/parse_cache.py
  tests/load_parity/run.py
  tests/load_parity/spec.py)

# Mixed prefill/decode batch: a continuing T=1 request must reuse its conv window.
pie_add_cuda_target(test_causal_conv1d
  SOURCES tests/test_causal_conv1d.cu ${PIE_KERNELS_CUDA_SRC}/kernels/causal_conv1d.cu
  LIBS CUDA::cudart)
pie_add_cuda_target(test_gather_tokens
  SOURCES tests/test_gather_tokens.cu ${PIE_KERNELS_CUDA_SRC}/kernels/gather_tokens.cu
  LIBS CUDA::cudart)
pie_add_cuda_target(test_pack_dense_mask
  SOURCES tests/pack_dense_mask_test.cu
  LIBS pie_driver_cuda_lib)
# Names three src/ files (kda, attn_res, swiglu), so by the rule above it links
# the lib rather than hand-maintaining that closure.
pie_add_cuda_target(test_kimi_k3_kernels
  SOURCES tests/kimi_k3_kernels_test.cu
  LIBS pie_driver_cuda_lib)
pie_add_cuda_target(test_envelope_dot
  SOURCES tests/test_envelope_dot.cu ${PIE_KERNELS_CUDA_SRC}/kernels/envelope.cu
  LIBS CUDA::cudart)
pie_add_cuda_target(test_kv_len_geometry
  SOURCES tests/test_kv_len_geometry.cu ${PIE_KERNELS_CUDA_SRC}/kernels/geometry.cu
  LIBS CUDA::cudart)
pie_add_cuda_target(test_page_compact
  SOURCES tests/test_page_compact.cu ${PIE_KERNELS_CUDA_SRC}/kernels/page_compact.cu
  LIBS CUDA::cudart)
pie_add_cuda_target(test_brle SOURCES tests/test_brle.cpp src/batch/brle.cpp)
# Host oracle: fork geometry -> BRLE decode -> masking == prefix truncation.
pie_add_cuda_target(test_masked_attention_geometry
  SOURCES tests/masked_attention_geometry_test.cpp src/batch/brle.cpp
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/tests)
# GPU parity for the geometry test above, through the real flashinfer prefill-custom path.
# Kernel-level decode-attention comparison. NO_CTEST: it is a measurement
# harness, not a pass/fail check, and it needs a Hopper device to say anything
# about the FA3 column.
pie_add_cuda_target(bench_decode_attention
  SOURCES tests/bench_decode_attention.cu
  LIBS pie_driver_cuda_lib CUDA::cudart
  NO_CTEST)
pie_add_cuda_target(masked_attention_parity
  SOURCES tests/masked_attention_parity.cu
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/tests
  LIBS pie_driver_cuda_lib CUDA::cudart)
pie_add_cuda_target(test_decode_batch_parity
  SOURCES tests/test_decode_batch_parity.cu
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/tests
  LIBS pie_driver_cuda_lib CUDA::cudart)
# The three GDN recurrent-decode kernels must agree; the SMEM one is the
# production default wherever the GQA launcher is reached.
pie_add_cuda_target(gdn_recurrent_step_parity
  SOURCES tests/gdn_recurrent_step_parity.cu
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/tests
  LIBS pie_driver_cuda_lib CUDA::cudart)
pie_add_cuda_target(test_attn_score_capture
  SOURCES tests/test_attn_score_capture.cu
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/tests
  LIBS pie_driver_cuda_lib CUDA::cudart)
pie_add_cuda_target(test_attn_prefill_score_capture
  SOURCES tests/test_attn_prefill_score_capture.cu
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/tests
  LIBS pie_driver_cuda_lib CUDA::cudart)
pie_add_cuda_target(test_attn_score_fold
  SOURCES tests/test_attn_score_fold.cu
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/tests
  LIBS pie_driver_cuda_lib CUDA::cudart)
pie_add_cuda_target(test_decode_fused_write_parity
  SOURCES tests/test_decode_fused_write_parity.cu
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/tests
  LIBS pie_driver_cuda_lib CUDA::cudart)
pie_add_cuda_target(test_graph_padding_kv_canary
  SOURCES tests/test_graph_padding_kv_canary.cu
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/tests
  LIBS pie_driver_cuda_lib CUDA::cudart)
pie_add_cuda_target(test_decode_cobatch_chain
  SOURCES tests/test_decode_cobatch_chain.cu
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/tests
  LIBS pie_driver_cuda_lib CUDA::cudart)
pie_add_cuda_target(test_driver_common
  SOURCES tests/test_common.cpp
  LIBS nlohmann_json::nlohmann_json)
pie_add_cuda_target(test_kv_cache_format
  SOURCES tests/test_kv_cache_format.cpp src/store/kv_cache_format.cpp)
pie_add_cuda_target(test_elastic
  SOURCES tests/test_elastic.cpp
  LIBS pie_driver_cuda_lib)
pie_add_cuda_target(test_elastic_converted_pools
  SOURCES tests/test_elastic_converted_pools.cpp
  LIBS pie_driver_cuda_lib CUDA::cudart CUDA::cuda_driver)
pie_add_cuda_target(test_kv_cache_quant
  SOURCES tests/test_kv_cache_quant.cu
  LIBS pie_driver_cuda_lib)
pie_add_cuda_target(test_swap_pool
  SOURCES tests/test_swap_pool.cpp
  LIBS pie_driver_cuda_lib)
pie_add_cuda_target(test_mxfp4_marlin
  SOURCES tests/test_mxfp4_marlin.cu ${PIE_KERNELS_CUDA_SRC}/kernels/mxfp4_marlin.cu
  LIBS CUDA::cudart)
pie_add_cuda_target(test_transcode_fused
  SOURCES tests/test_transcode_fused.cu
  LIBS pie_driver_cuda_lib)
# Model-family full-forward parity harnesses: manual GPU de-risk tools. They
# textually include the forward under test, so they stay standalone -- linking
# pie_driver_cuda_lib would duplicate every symbol in that forward. Compare
# against scripts/<family>_*_dump.py reference dumps at runtime (kept outside
# the repo).
pie_add_cuda_target(csm_backbone_parity
  SOURCES
    tests/csm_backbone_parity.cu
    src/model/csm/csm_depth_decoder_forward.cu
    src/model/csm/mimi_decoder_forward.cu
  LIBS CUDA::cudart
  NO_CTEST)
pie_add_cuda_target(csm_depth_decoder_parity
  SOURCES tests/csm_depth_decoder_parity.cu
  LIBS CUDA::cudart
  NO_CTEST)
pie_add_cuda_target(csm_generate_parity
  SOURCES
    tests/csm_generate_parity.cu
    src/model/csm/csm_depth_decoder_forward.cu
    src/model/csm/mimi_decoder_forward.cu
  LIBS CUDA::cudart
  NO_CTEST)
pie_add_cuda_target(mimi_decoder_full_parity
  SOURCES tests/mimi_decoder_full_parity.cu
  LIBS CUDA::cudart
  NO_CTEST)
pie_add_cuda_target(gemma4_audio_full_parity
  SOURCES tests/gemma4_audio_full_parity.cu
  LIBS CUDA::cudart
  NO_CTEST)
pie_add_cuda_target(gemma4_vision_full_parity
  SOURCES tests/gemma4_vision_full_parity.cu
  LIBS CUDA::cudart
  NO_CTEST)
pie_add_cuda_target(gemma4_vision_full_parity_bf16
  SOURCES tests/gemma4_vision_full_parity_bf16.cu
  LIBS CUDA::cudart
  NO_CTEST)
pie_add_cuda_target(gemma4_vision_patch_parity
  SOURCES tests/gemma4_vision_patch_parity.cu
  LIBS CUDA::cudart
  NO_CTEST)
pie_add_cuda_target(qwen3_vl_vision_full_parity
  SOURCES tests/qwen3_vl_vision_full_parity.cu
  LIBS pie_driver_cuda_lib CUDA::cudart CUDA::cublas
  NO_CTEST)
# Reject dimensions without a compiled FlashInfer decode specialization.
pie_add_cuda_target(test_flashinfer_decode_head_dim_guard
  SOURCES tests/test_flashinfer_decode_head_dim_guard.cpp
  LIBS pie_driver_cuda_lib)
pie_add_cuda_target(test_graph_variant SOURCES tests/graph_variant_test.cpp)
# Reaches `pie/driver/fire/descriptor.hpp`, which pulls in `launch/program.hpp`
# and so `pie_driver_abi.h` — the ABI include dir is not optional here even
# though nothing in the test names it directly.
pie_add_cuda_target(test_decode_envelope_contract
  SOURCES tests/decode_envelope_contract_test.cpp
  INCLUDE_DIRS ${PIE_DRIVER_INCLUDE_DIR} ${PIE_PTIR_INCLUDE_DIR} ${PIE_DRIVER_ABI_INCLUDE_DIR})
pie_add_cuda_target(test_tp_gate
  SOURCES tests/tp_gate_test.cpp
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src
  LIBS Threads::Threads)
pie_add_cuda_target(test_grouped_copy
  SOURCES tests/grouped_copy_test.cpp)
# The eviction rule for streamed group instances. Header-only and CUDA-free on
# purpose: whether a slab thrashes is decided here, and that must be checkable
# without a device.
pie_add_cuda_target(test_group_slot_index
  SOURCES tests/group_slot_index_test.cpp)
# When a pack on disk may be trusted. Header-only, device-free and backend-free
# on purpose: the pack is a property of the load plan, both drivers can use one,
# and every check here stands between a mapping and the wrong weights.
pie_add_cuda_target(test_stream_pack
  SOURCES tests/stream_pack_test.cpp
  INCLUDE_DIRS ${PIE_LOADER_INCLUDE_DIR})
# The slab itself, against the residency it replaces: needs a device, a
# checkpoint on disk and the loader, because the claim it checks is about bytes.
if(PIE_LOADER_STATICLIB)
  pie_add_cuda_target(test_group_stream_cache
    SOURCES tests/group_stream_cache_test.cpp
    INCLUDE_DIRS
      ${PIE_DRIVER_INCLUDE_DIR}
      ${PIE_PTIR_INCLUDE_DIR}
      ${PIE_DRIVER_ABI_INCLUDE_DIR}
      ${PIE_LOADER_INCLUDE_DIR}
    LIBS pie_driver_cuda_lib nlohmann_json::nlohmann_json ${PIE_LOADER_STATICLIB} dl pthread)
endif()
# The materialized-weight artifact's round trip and its page alignment. Needs a
# device because the format's payload is device memory: serializing is a D2H of
# every owned buffer, and the staticlib because the file itself is the loader's.
if(PIE_LOADER_STATICLIB)
  pie_add_cuda_target(test_weight_store_codec
    SOURCES tests/weight_store_codec_test.cpp
    INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src ${PIE_LOADER_INCLUDE_DIR}
    LIBS pie_driver_cuda_lib CUDA::cudart ${PIE_LOADER_STATICLIB} dl pthread)
else()
  message(STATUS
    "test_weight_store_codec: skipped (no libpie_loader_capi.a; build one with "
    "`cargo rustc -p pie-loader-capi --lib --crate-type staticlib`)")
endif()
# The non-compact ExtentWrite path. Header-only: it builds its own one-file
# plan, so it needs the loader's headers but not the staticlib.
pie_add_cuda_target(test_strided_copy
  SOURCES tests/strided_copy_test.cu
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src ${PIE_LOADER_INCLUDE_DIR}
  LIBS CUDA::cudart)
pie_add_cuda_target(test_chunked_argmax
  SOURCES tests/test_chunked_argmax.cu ${PIE_KERNELS_CUDA_SRC}/kernels/argmax.cu
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src ${PIE_LOADER_INCLUDE_DIR}
  LIBS CUDA::cudart)
pie_add_cuda_target(test_nucleus_region
  SOURCES tests/nucleus_region_test.cpp
  INCLUDE_DIRS ${PIE_DRIVER_INCLUDE_DIR} ${PIE_PTIR_INCLUDE_DIR} ${PIE_DRIVER_ABI_INCLUDE_DIR} ${PIE_LOADER_INCLUDE_DIR}
  TEST_ARGS ${PIE_DRIVER_FIXTURES_DIR})
pie_add_cuda_target(test_page_translation
  SOURCES tests/page_translation_test.cpp)
pie_add_cuda_target(test_ptir_graph_key
  SOURCES tests/ptir_graph_key_test.cpp
  INCLUDE_DIRS
    ${PIE_DRIVER_INCLUDE_DIR} ${PIE_PTIR_INCLUDE_DIR}
    ${PIE_DRIVER_ABI_INCLUDE_DIR} ${PIE_LOADER_INCLUDE_DIR}
  LIBS CUDA::cudart tomlplusplus::tomlplusplus)
pie_add_cuda_target(test_frame_carrier_device
  SOURCES tests/test_frame_carrier_device.cpp
  LIBS pie_driver_cuda_lib
  TEST_NAME pipeline_frame_carrier_device)
pie_add_cuda_target(test_pipeline_dispatch_bind
  SOURCES tests/test_pipeline_dispatch_bind.cpp
  LIBS pie_driver_cuda_lib
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)
# Same-instance run-ahead concurrency gate; build with -fsanitize=thread for the TSAN run.
pie_add_cuda_target(test_pipeline_dispatch_race
  SOURCES tests/pipeline_dispatch_race_test.cpp
  LIBS pie_driver_cuda_lib
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)
pie_add_cuda_target(test_ptir_tier0
  SOURCES tests/ptir_tier0_test.cu
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/tests ${PIE_DRIVER_INCLUDE_DIR} ${PIE_PTIR_INCLUDE_DIR} ${PIE_DRIVER_ABI_INCLUDE_DIR} ${PIE_LOADER_INCLUDE_DIR}
  LIBS CUDA::cudart)
pie_add_cuda_target(test_ptir_runner
  SOURCES tests/ptir_runner_test.cu
  INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/tests
    ${PIE_DRIVER_INCLUDE_DIR} ${PIE_PTIR_INCLUDE_DIR}
    ${PIE_DRIVER_ABI_INCLUDE_DIR} ${PIE_LOADER_INCLUDE_DIR}
  LIBS CUDA::cudart)
# Emits + NVRTC-compiles a fused epilogue kernel; asserts tier-1 == tier-0.
pie_add_cuda_target(test_ptir_tier1
  SOURCES tests/ptir_tier1_test.cu
  INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/tests
    ${PIE_DRIVER_INCLUDE_DIR} ${PIE_PTIR_INCLUDE_DIR}
    ${PIE_DRIVER_ABI_INCLUDE_DIR} ${PIE_LOADER_INCLUDE_DIR}
  LIBS CUDA::cudart CUDA::cuda_driver CUDA::nvrtc)
pie_add_cuda_target(test_ptir_grouped_dispatch
  SOURCES tests/ptir_grouped_dispatch_test.cpp
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/tests
  LIBS pie_driver_cuda_lib
  TEST_ARGS ${PIE_DRIVER_FIXTURES_DIR})
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # Keeps its own copies of the sources under test: -fsanitize=address below is
  # PRIVATE, so linking pie_driver_cuda_lib would hand it uninstrumented
  # objects and the sanitizer would see nothing of what this target exists for.
  # The library is still linked, but only to resolve everything these three
  # files call into. Listing that closure by hand does not converge -- it
  # reaches `ops/attention_flashinfer.cu` in two hops -- and an explicit object
  # already defines a symbol, so the archive member that would duplicate it is
  # never pulled in.
  pie_add_cuda_target(test_ptir_grouped_dispatch_asan
    SOURCES
      tests/ptir_grouped_dispatch_test.cpp
      src/pipeline/dispatch.cu
      src/pipeline/frame_carrier.cpp
      src/pipeline/frame_carrier_kernels.cu
    INCLUDE_DIRS
      ${CMAKE_CURRENT_SOURCE_DIR}/tests
      ${PIE_DRIVER_INCLUDE_DIR} ${PIE_PTIR_INCLUDE_DIR}
      ${PIE_DRIVER_ABI_INCLUDE_DIR} ${PIE_LOADER_INCLUDE_DIR}
    LIBS pie_driver_cuda_lib CUDA::cudart CUDA::cuda_driver CUDA::nvrtc
    TEST_ARGS ${PIE_DRIVER_FIXTURES_DIR})
  target_compile_options(test_ptir_grouped_dispatch_asan PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-omit-frame-pointer>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fsanitize=address>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fno-omit-frame-pointer>)
  target_link_options(test_ptir_grouped_dispatch_asan PRIVATE
    -fsanitize=address)
  set_tests_properties(ptir_grouped_dispatch_asan PROPERTIES
    ENVIRONMENT
      "ASAN_OPTIONS=detect_leaks=0:halt_on_error=1:protect_shadow_gap=0")
endif()
pie_add_cuda_target(test_ptir_golden_exec
  SOURCES tests/ptir_golden_exec_test.cu
  INCLUDE_DIRS ${PIE_DRIVER_INCLUDE_DIR} ${PIE_PTIR_INCLUDE_DIR} ${PIE_DRIVER_ABI_INCLUDE_DIR} ${PIE_LOADER_INCLUDE_DIR}
  LIBS CUDA::cudart
  TEST_ARGS ${PIE_DRIVER_FIXTURES_DIR})
# Validation the `extern "C"` entry points owe their caller. Reaches
# `pie_driver_cuda_lib`, and so `loaded_model.cpp`, which calls the loader —
# same reason `test_model_registry` below needs the staticlib.
if(PIE_LOADER_STATICLIB)
  pie_add_cuda_target(test_entry_validation
    SOURCES tests/entry_validation_test.cpp
    LIBS pie_driver_cuda_lib ${PIE_LOADER_STATICLIB} dl pthread)
else()
  message(STATUS
    "test_entry_validation: skipped (no libpie_loader_capi.a; build one with "
    "`cargo rustc -p pie-loader-capi --lib --crate-type staticlib`)")
endif()
pie_add_cuda_target(test_rs_descriptor_validation
  SOURCES tests/rs_descriptor_validation_test.cpp
  INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${PIE_DRIVER_INCLUDE_DIR} ${PIE_PTIR_INCLUDE_DIR}
    ${PIE_DRIVER_ABI_INCLUDE_DIR} ${PIE_LOADER_INCLUDE_DIR})
pie_add_cuda_target(test_precomputed_embeddings
  SOURCES tests/precomputed_embeddings_test.cpp
  INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${PIE_DRIVER_INCLUDE_DIR} ${PIE_PTIR_INCLUDE_DIR}
    ${PIE_DRIVER_ABI_INCLUDE_DIR} ${PIE_LOADER_INCLUDE_DIR}
  LIBS CUDA::cudart)
# Structural proof for model/registry.{hpp,cpp}: every model_type resolves to its
# intended family/binder and declares what it binds; unrecognized archs are
# rejected. No GPU, no weights — but `pie_driver_cuda_lib` reaches
# `loaded_model.cpp`, which calls the loader, so it needs the staticlib.
if(PIE_LOADER_STATICLIB)
  pie_add_cuda_target(test_model_registry
    SOURCES tests/model_registry_test.cpp
    LIBS pie_driver_cuda_lib ${PIE_LOADER_STATICLIB} dl pthread)
else()
  message(STATUS
    "test_model_registry: skipped (no libpie_loader_capi.a; build one with "
    "`cargo rustc -p pie-loader-capi --lib --crate-type staticlib`)")
endif()
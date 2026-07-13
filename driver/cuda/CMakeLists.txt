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

# Optional compiler cache (sccache/ccache) for host C++ compiles; speeds
# clean rebuilds / branch switches. nvcc is NOT auto-launched: sccache
# mishandles nvcc's fatbinary temp files (intermittent SIGHUP-killed
# server). Opt in explicitly with -DCMAKE_CUDA_COMPILER_LAUNCHER=... if your
# sccache build is known-good with nvcc.
if(NOT CMAKE_C_COMPILER_LAUNCHER AND NOT CMAKE_CXX_COMPILER_LAUNCHER)
  if(DEFINED ENV{PIE_COMPILER_LAUNCHER} AND NOT "$ENV{PIE_COMPILER_LAUNCHER}" STREQUAL "")
    set(PIE_COMPILER_LAUNCHER "$ENV{PIE_COMPILER_LAUNCHER}")
  else()
    find_program(PIE_COMPILER_LAUNCHER NAMES sccache ccache)
  endif()
  if(PIE_COMPILER_LAUNCHER)
    set(CMAKE_C_COMPILER_LAUNCHER "${PIE_COMPILER_LAUNCHER}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${PIE_COMPILER_LAUNCHER}")
    message(STATUS "Using compiler launcher (C/C++ only): ${PIE_COMPILER_LAUNCHER}")
  endif()
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
# No Python interpreter required to build: FlashInfer's CUTLASS MoE launchers
# are vendored (third_party/flashinfer_generated) and flashinfer itself is
# fetched DOWNLOAD_ONLY for headers.

# CUDA 13+ ships libcudacxx/CUB/Thrust under a `cccl/` prefix; CUTLASS and
# vendored Marlin still `#include <cuda/std/...>` etc. without that prefix.
# Add the CCCL dir when the non-prefixed headers don't exist.
foreach(_cuda_inc IN LISTS CUDAToolkit_INCLUDE_DIRS)
  if(EXISTS "${_cuda_inc}/cccl" AND NOT EXISTS "${_cuda_inc}/cuda/std")
    list(APPEND PIE_CCCL_INCLUDE_DIRS "${_cuda_inc}/cccl")
  endif()
endforeach()
if(PIE_CCCL_INCLUDE_DIRS)
  message(STATUS "Adding CCCL include dirs for CUDA 13+: ${PIE_CCCL_INCLUDE_DIRS}")
endif()

if(NOT PIE_DRIVER_ABI_INCLUDE_DIR)
  set(PIE_DRIVER_ABI_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../interface/driver/include")
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

include(cmake/CPM.cmake)

# Persistent CPM cache when set, so incremental builds don't re-clone from
# GitHub. CI sets this via GITHUB_ENV; local builds can export it in .bashrc.
if(DEFINED ENV{CPM_SOURCE_CACHE} AND NOT CPM_SOURCE_CACHE)
  set(CPM_SOURCE_CACHE $ENV{CPM_SOURCE_CACHE})
endif()

# flashinfer ships no top-level CMakeLists.txt since v0.3.0; fetch
# DOWNLOAD_ONLY for headers/csrc and compile only the launchers we need
# directly into the driver (see the generated-MoE and Hopper/FA3 blocks below).
cpmaddpackage(
  NAME flashinfer
  GITHUB_REPOSITORY flashinfer-ai/flashinfer
  GIT_TAG v0.6.9
  GIT_SHALLOW TRUE
  DOWNLOAD_ONLY YES)

# v0.6.9 has a narrowing-conversion warning nvcc escalates to an error
# (`state_stride_batch`, signed long, brace-initialized into a uint64_t
# field). Patch once after fetch; a sentinel file avoids re-patching every
# fresh build-hash dir.
set(_FI_PATCH_SENTINEL "${flashinfer_SOURCE_DIR}/.pie_narrowing_patched")
set(_FI_PATCH_FILE
  "${flashinfer_SOURCE_DIR}/include/flashinfer/mamba/kernel_selective_state_update_stp.cuh")
if(EXISTS "${_FI_PATCH_FILE}" AND NOT EXISTS "${_FI_PATCH_SENTINEL}")
  file(READ "${_FI_PATCH_FILE}" _fi_src)
  string(REPLACE
    "DSTATE * DIM, params.state_stride_batch}"
    "DSTATE * DIM, static_cast<uint64_t>(params.state_stride_batch)}"
    _fi_src "${_fi_src}")
  file(WRITE "${_FI_PATCH_FILE}" "${_fi_src}")
  file(WRITE "${_FI_PATCH_SENTINEL}" "patched")
  message(STATUS "Patched FlashInfer narrowing conversion in ${_FI_PATCH_FILE}")
endif()

cpmaddpackage(URI "gh:marzer/tomlplusplus@3.4.0")
cpmaddpackage(URI "gh:CLIUtils/CLI11@2.5.0")
cpmaddpackage(URI "gh:nlohmann/json@3.12.0" OPTIONS "JSON_BuildTests OFF")

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
  set(PIE_CUDA_CUSTOM_ALL_REDUCE_SOURCE src/kernels/custom_all_reduce_stub.cpp)
  message(STATUS "Disabling FlashInfer custom all-reduce for CUDA architecture ${CMAKE_CUDA_ARCHITECTURES}")
else()
  set(PIE_CUDA_CUSTOM_ALL_REDUCE_SOURCE src/kernels/custom_all_reduce.cu)
endif()

# Hopper/FA3 attention + the SM90 XQA GQA8 kernel are SM90-only; every other
# arch gets the stub.
if(PIE_HAS_SM90)
  set(PIE_CUDA_FLASHINFER_HOPPER_SOURCE
    src/ops/attention_flashinfer_hopper.cu
    src/ops/attention_xqa_gqa8_sm90.cu)
  message(STATUS "Enabling FlashInfer Hopper/FA3 attention for CUDA architecture ${CMAKE_CUDA_ARCHITECTURES}")
else()
  set(PIE_CUDA_FLASHINFER_HOPPER_SOURCE src/ops/attention_flashinfer_hopper_stub.cpp)
  message(STATUS "Disabling FlashInfer Hopper/FA3 attention for CUDA architecture ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# FlashInfer's Mamba SSU vertical/horizontal kernels need TMA + memcpy_async_tx
# (sm_90+); the runtime algorithm selector also picks the horizontal kernel
# for sm >= 100 with bf16/fp16 state, so gate the same compile define there.
# Without it, sm_90+ runs would silently fall back to the slow simple
# algorithm even though the runtime SM check would otherwise pick
# vertical/horizontal.
if(PIE_HAS_SM90 OR PIE_HAS_SM100)
  set(PIE_CUDA_FLASHINFER_MAMBA_SM90 ON)
  message(STATUS "Enabling FlashInfer Mamba SM90 vertical/horizontal SSU kernels")
else()
  set(PIE_CUDA_FLASHINFER_MAMBA_SM90 OFF)
  message(STATUS "FlashInfer Mamba SM90 vertical/horizontal SSU kernels disabled (legacy + simple algorithm only)")
endif()

# ---- pie_driver_cuda_lib ---------------------------------------------------
# Native CUDA backend. `pie_driver_cuda_lib` is linked in-process by the Rust
# worker; `pie_driver_cuda` below is a thin direct-create diagnostic shim.
# Every production translation unit under src/ is listed exactly once, grouped
# to match the module tree (see README.md).
add_library(pie_driver_cuda_lib STATIC
  src/abi.cpp
  src/context.cpp
  src/tensor.cpp
  src/distributed.cpp
  # batch/ -- one submitted batch -> device work
  src/batch/brle.cpp
  src/batch/compose.cpp
  src/batch/forward.cpp
  src/batch/persistent_inputs.cpp
  src/batch/tp.cpp
  src/batch/workspace.cpp
  # pipeline/ -- guest programs (PTIR dispatch, tier0, frame carrier)
  src/pipeline/dispatch.cu
  src/pipeline/frame_carrier.cpp
  src/pipeline/frame_carrier_kernels.cu
  src/pipeline/registry.cpp
  # model/ -- registry, workspace, weight store, HF config, one dir per family
  src/model/config.cpp
  src/model/loaded_model.cpp
  src/model/registry.cpp
  src/model/weight_store.cpp
  src/model/workspace.cpp
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
  src/model/kimi/kimi.cpp
  src/model/kimi/kimi_forward.cpp
  src/model/kimi/kimi_model.cpp
  src/model/llama_like/llama_like.cpp
  src/model/llama_like/llama_like_model.cpp
  src/model/llama_like/mistral3.cpp
  src/model/llama_like/qwen3.cpp
  src/model/mixtral/gpt_oss.cpp
  src/model/mixtral/mixtral.cpp
  src/model/mixtral/mixtral_model.cpp
  src/model/nemotron_h/nemotron_h.cpp
  src/model/nemotron_h/nemotron_h_forward.cpp
  src/model/nemotron_h/nemotron_h_model.cpp
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
  # store/ -- long-lived device memory pools
  src/store/dsa_cache.cpp
  src/store/kv_cache.cpp
  src/store/kv_cache_format.cpp
  src/store/memory_planner.cpp
  src/store/mla_cache.cpp
  src/store/recurrent_state_cache.cpp
  src/store/swap_pool.cpp
  # loader/ -- snapshot -> WeightStore
  src/loader/checkpoint_source.cpp
  src/loader/safetensors_manifest.cpp
  # ops/ -- attention / GEMM / MoE / SSM wrappers
  src/ops/attention_flashinfer.cu
  ${PIE_CUDA_FLASHINFER_HOPPER_SOURCE}
  src/ops/attention_mla.cu
  src/ops/attention_naive.cu
  src/ops/attention_naive_paged.cu
  src/ops/attention_workspace.cpp
  src/ops/attention_xqa.cu
  src/ops/attention_xqa_gqa2.cu
  src/ops/attention_xqa_gqa2_p16.cu
  src/ops/attention_xqa_gqa4.cu
  src/ops/attention_xqa_gqa8.cu
  src/ops/flashinfer_mamba.cu
  src/ops/gemm.cpp
  # kernels/ -- fused kernels
  src/kernels/add_bias.cu
  src/kernels/altup.cu
  src/kernels/altup_aux.cu
  src/kernels/argmax.cu
  src/kernels/attn_sink.cu
  src/kernels/causal_conv1d.cu
  ${PIE_CUDA_CUSTOM_ALL_REDUCE_SOURCE}
  src/kernels/deinterleave.cu
  src/kernels/dequant_fp4.cu
  src/kernels/dequant_fp8.cu
  src/kernels/dequant_wna16.cu
  src/kernels/dsa_indexer.cu
  src/kernels/dsv4_compress.cu
  src/kernels/dsv4_hc.cu
  src/kernels/dsv4_routing.cu
  src/kernels/dtype_cast.cu
  src/kernels/embed.cu
  src/kernels/envelope.cu
  src/kernels/gated_delta_net.cu
  src/kernels/gather_rows.cu
  src/kernels/gather_tokens.cu
  src/kernels/gaussian_topk.cu
  src/kernels/geometry.cu
  src/kernels/head_dim_pad.cu
  src/kernels/kimi_mla.cu
  src/kernels/kv_paged.cu
  src/kernels/mla_paged.cu
  src/kernels/moe_dispatch.cu
  src/kernels/mxfp4_marlin.cu
  src/kernels/nemotron_h.cu
  src/kernels/pack_dense_mask.cu
  src/kernels/quant_bf16_to_fp8.cu
  src/kernels/quant_bf16_to_mxfp4.cu
  src/kernels/residual_add.cu
  src/kernels/rmsnorm.cu
  src/kernels/rope.cu
  src/kernels/scalar_mul.cu
  src/kernels/slab_scatter.cu
  src/kernels/softcap.cu
  src/kernels/split_packed.cu
  src/kernels/swiglu.cu
  src/kernels/topk_softmax.cu
  src/kernels/transcode.cu
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
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${PIE_DRIVER_ABI_INCLUDE_DIR}
    ${NCCL_INCLUDE_DIR}
    ${PIE_CCCL_INCLUDE_DIRS}
    ${flashinfer_SOURCE_DIR}/include
    ${flashinfer_SOURCE_DIR}/csrc
    ${flashinfer_SOURCE_DIR}/3rdparty/cutlass/include
    ${flashinfer_SOURCE_DIR}/3rdparty/cutlass/tools/util/include
    ${flashinfer_SOURCE_DIR}/3rdparty/spdlog/include)

# ---- pie_driver_cuda (standalone diagnostic executable) --------------------

add_executable(pie_driver_cuda src/main.cpp)
target_link_libraries(pie_driver_cuda PRIVATE pie_driver_cuda_lib)

# ---- FlashInfer generated CUTLASS MoE (vendored) ---------------------------
# Pre-generated FlashInfer CUTLASS grouped-GEMM MoE launchers, vendored under
# third_party/flashinfer_generated/ (the SM80 set; byte-identical for
# SM80/86/89 and the only set the build compiles). Removes the build-time
# Python codegen and its tvm_ffi/pynvml/tqdm imports -- no external Python
# deps. Refresh after bumping the vendored FlashInfer revision with:
#   FLASHINFER_SRC=<src> driver/cuda/third_party/flashinfer_generated/regenerate.sh
set(PIE_FLASHINFER_MOE_GEN_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/flashinfer_generated")
set(PIE_FLASHINFER_MOE_GENERATED_SOURCES
  third_party/flashinfer_generated/gemm_grouped/80/cutlass_kernel_file_gemm_grouped_sm80_M16_group0.generated.cu
  third_party/flashinfer_generated/gemm_grouped/80/cutlass_kernel_file_gemm_grouped_sm80_M16_group1.generated.cu
  third_party/flashinfer_generated/gemm_grouped/80/cutlass_kernel_file_gemm_grouped_sm80_M16_group2.generated.cu
  third_party/flashinfer_generated/gemm_grouped/80/cutlass_kernel_file_gemm_grouped_sm80_M32_group0.generated.cu
  third_party/flashinfer_generated/gemm_grouped/80/cutlass_kernel_file_gemm_grouped_sm80_M32_group1.generated.cu
  third_party/flashinfer_generated/gemm_grouped/80/cutlass_kernel_file_gemm_grouped_sm80_M64_group0.generated.cu
  third_party/flashinfer_generated/gemm_grouped/80/cutlass_kernel_file_gemm_grouped_sm80_M64_group1.generated.cu
  third_party/flashinfer_generated/gemm_grouped/80/cutlass_kernel_file_gemm_grouped_sm80_M128_group0.generated.cu
  third_party/flashinfer_generated/gemm_grouped/80/cutlass_kernel_file_gemm_grouped_sm80_M128_group1.generated.cu)
list(LENGTH PIE_FLASHINFER_MOE_GENERATED_SOURCES PIE_FLASHINFER_MOE_GENERATED_COUNT)
message(STATUS "Using ${PIE_FLASHINFER_MOE_GENERATED_COUNT} FlashInfer generated MoE launchers")

# FlashInfer/TensorRT-LLM CUTLASS MoE runner. Kept as a separate OBJECT
# library because its dependency slice (nv_internal/) is large and should
# stay a clear opt-in boundary for Nemotron experimentation.
add_library(pie_flashinfer_cutlass_moe OBJECT
  src/ops/flashinfer_moe.cu
  ${flashinfer_SOURCE_DIR}/csrc/nv_internal/cpp/common/envUtils.cpp
  ${flashinfer_SOURCE_DIR}/csrc/nv_internal/cpp/common/logger.cpp
  ${flashinfer_SOURCE_DIR}/csrc/nv_internal/cpp/common/stringUtils.cpp
  ${flashinfer_SOURCE_DIR}/csrc/nv_internal/cpp/common/tllmException.cpp
  ${flashinfer_SOURCE_DIR}/csrc/nv_internal/cpp/common/memoryUtils.cu
  ${flashinfer_SOURCE_DIR}/csrc/nv_internal/tensorrt_llm/kernels/preQuantScaleKernel.cu
  ${flashinfer_SOURCE_DIR}/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.cpp
  ${flashinfer_SOURCE_DIR}/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_tma_warp_specialized_input.cu
  ${flashinfer_SOURCE_DIR}/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_bf16_bf16.cu
  ${flashinfer_SOURCE_DIR}/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.cu
  ${flashinfer_SOURCE_DIR}/csrc/nv_internal/tensorrt_llm/kernels/lora/lora.cpp
  ${PIE_FLASHINFER_MOE_GENERATED_SOURCES})
target_include_directories(pie_flashinfer_cutlass_moe PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CUDAToolkit_INCLUDE_DIRS}
  ${PIE_CCCL_INCLUDE_DIRS}
  ${flashinfer_SOURCE_DIR}/include
  ${flashinfer_SOURCE_DIR}/csrc
  ${flashinfer_SOURCE_DIR}/csrc/fused_moe/cutlass_backend
  ${flashinfer_SOURCE_DIR}/csrc/nv_internal
  ${flashinfer_SOURCE_DIR}/csrc/nv_internal/include
  ${flashinfer_SOURCE_DIR}/csrc/nv_internal/tensorrt_llm/cutlass_extensions/include
  ${flashinfer_SOURCE_DIR}/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/include
  ${flashinfer_SOURCE_DIR}/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels
  ${flashinfer_SOURCE_DIR}/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm
  ${flashinfer_SOURCE_DIR}/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm
  ${PIE_FLASHINFER_MOE_GEN_DIR}
  ${flashinfer_SOURCE_DIR}/3rdparty/cutlass/include
  ${flashinfer_SOURCE_DIR}/3rdparty/cutlass/tools/util/include
  ${flashinfer_SOURCE_DIR}/3rdparty/spdlog/include)
target_compile_definitions(pie_flashinfer_cutlass_moe PRIVATE
  ENABLE_BF16=1
  ENABLE_FP8=1
  USING_OSS_CUTLASS_MOE_GEMM=1)
target_compile_options(pie_flashinfer_cutlass_moe PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>)
set_target_properties(pie_flashinfer_cutlass_moe PROPERTIES
  POSITION_INDEPENDENT_CODE ON)
target_sources(pie_driver_cuda_lib PRIVATE
  $<TARGET_OBJECTS:pie_flashinfer_cutlass_moe>)

# ---- Marlin (W4A16 / MXFP4 GEMM via vendored vLLM kernel) ------------------
# Built by default: GPT-OSS's MXFP4 auto policy should use native
# packed-weight GEMM on supported GPUs and fall back only on legacy targets.
# Vendored under third_party/marlin/; the wrapper exposes a torch-free C++ API.
option(PIE_CUDA_BUILD_MARLIN "Build vendored marlin W4A16/MXFP4 GEMM" ON)
option(PIE_MARLIN_ALL_SHAPES
  "Compile all Marlin dtype combos (fp16/bf16/int8 x u4/u4b8/fp8/mxfp4). \
OFF (default) compiles only bf16x{u4,u4b8,mxfp4} -- the shapes pie dispatches."
  OFF)
# Env overrides (previously forwarded by worker/build.rs as -D...).
if(DEFINED ENV{PIE_CUDA_BUILD_MARLIN})
  string(TOLOWER "$ENV{PIE_CUDA_BUILD_MARLIN}" _pie_marlin)
  if(_pie_marlin MATCHES "^(1|on|true|yes)$")
    set(PIE_CUDA_BUILD_MARLIN ON)
  else()
    set(PIE_CUDA_BUILD_MARLIN OFF)
  endif()
endif()
if(DEFINED ENV{PIE_MARLIN_ALL_SHAPES})
  string(TOLOWER "$ENV{PIE_MARLIN_ALL_SHAPES}" _pie_marlin_all)
  if(_pie_marlin_all MATCHES "^(1|on|true|yes)$")
    set(PIE_MARLIN_ALL_SHAPES ON)
  endif()
endif()
if(PIE_CUDA_BUILD_MARLIN)
  # CUDA 13 generates __global__ template stubs with static linkage,
  # breaking cross-TU explicit instantiations. marlin.cu #includes the
  # sm80_kernel_*.cu files directly so instantiation and selection share a
  # single TU. Do NOT list sm80_kernel_*.cu as separate sources -- they
  # would double-define every template.
  target_sources(pie_driver_cuda_lib PRIVATE
    third_party/marlin/marlin.cu
    third_party/marlin/gptq_marlin_repack.cu
    third_party/marlin/awq_marlin_repack.cu
    third_party/marlin/marlin_wrapper.cpp)
  target_include_directories(pie_driver_cuda_lib PRIVATE
    third_party/marlin)
  target_compile_definitions(pie_driver_cuda_lib PRIVATE PIE_CUDA_HAS_MARLIN=1)
  if(PIE_MARLIN_ALL_SHAPES)
    target_compile_definitions(pie_driver_cuda_lib PRIVATE PIE_MARLIN_ALL_SHAPES=1)
  endif()
endif()

# FlashInfer Mamba SSU on Hopper/Blackwell gates the vertical/horizontal
# producer-consumer kernels behind FLASHINFER_MAMBA_ENABLE_SM90 at compile
# time (TMA + memcpy_async_tx are sm_90+); see the arch-gated selection above.
if(PIE_CUDA_FLASHINFER_MAMBA_SM90)
  target_compile_definitions(pie_driver_cuda_lib PRIVATE
    FLASHINFER_MAMBA_ENABLE_SM90=1)
endif()

target_link_libraries(pie_driver_cuda_lib PUBLIC
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

# ---- Tests and harnesses ----------------------------------------------------
enable_testing()

# Runtime fixtures/scripts consumed by tests but not compiled: golden PTIR
# containers (tests/golden-ptir, read by test_ptir_container/golden_exec via
# TEST_ARGS), the load-parity harness (tests/load_parity/run.py, driven
# manually against a live GPU + recipe), the qwen3.5 multi-request parity
# script, and the tier0/tier1/masked-attention host reference headers
# (#include-only, no standalone translation unit). Grouped into one
# SOURCES-only custom target so `check-structure.sh` rule 6 sees them as
# part of the build graph without compiling them.
add_custom_target(pie_driver_cuda_test_fixtures SOURCES
  tests/support/host_eval.hpp
  tests/support/tier1_codegen.hpp
  tests/masked_attention_reference.hpp
  tests/parity_qwen3_5_multireq.py
  tests/load_parity/README.md
  tests/load_parity/dtypes.py
  tests/load_parity/gen.py
  tests/load_parity/oracle.py
  tests/load_parity/parse_cache.py
  tests/load_parity/run.py
  tests/load_parity/spec.py
  tests/golden-ptir/beam_epilogue.txt
  tests/golden-ptir/counter_pingpong.txt
  tests/golden-ptir/dfa_ingraph.txt
  tests/golden-ptir/greedy_argmax.txt
  tests/golden-ptir/mtp_verify_tail.txt
  tests/golden-ptir/nucleus_sample.txt
  tests/golden-ptir/neg_body_type_error.txt
  tests/golden-ptir/neg_intrinsic_wrong_stage.txt
  tests/golden-ptir/neg_model_gated_missing.txt
  tests/golden-ptir/neg_sink_at_epilogue.txt
  tests/golden-ptir/neg_spsc_second_producer.txt
  tests/golden-ptir/neg_t10_nonreplayable.txt
  tests/golden-ptir/pivot_predicates_multistage.txt
  tests/golden-ptir/section3_masked_gumbel.txt
  tests/golden-ptir/staged_dispatch.txt
  tests/golden-ptir/structured_masks.txt)

file(GLOB PIE_PTIR_CANONICAL_GOLDENS CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/../../interface/ptir/tests/golden-ptir/*.txt)
pie_add_cuda_target(test_ptir_decoder_limits
  SOURCES ../common/tests/ptir_decoder_limits_test.cpp
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../common/include
  TEST_ARGS
    ${CMAKE_CURRENT_SOURCE_DIR}/../../interface/ptir/tests/malformed_wire_corpus.txt
    ${PIE_PTIR_CANONICAL_GOLDENS})

# Mixed prefill/decode batch: a continuing T=1 request must reuse its conv window.
pie_add_cuda_target(test_causal_conv1d
  SOURCES tests/test_causal_conv1d.cu src/kernels/causal_conv1d.cu
  LIBS CUDA::cudart)
pie_add_cuda_target(test_gather_tokens
  SOURCES tests/test_gather_tokens.cu src/kernels/gather_tokens.cu
  LIBS CUDA::cudart)
pie_add_cuda_target(test_pack_dense_mask
  SOURCES tests/pack_dense_mask_test.cu src/kernels/pack_dense_mask.cu
  LIBS CUDA::cudart)
pie_add_cuda_target(test_envelope_dot
  SOURCES tests/test_envelope_dot.cu src/kernels/envelope.cu
  LIBS CUDA::cudart)
pie_add_cuda_target(test_kv_len_geometry
  SOURCES tests/test_kv_len_geometry.cu src/kernels/geometry.cu
  LIBS CUDA::cudart)
pie_add_cuda_target(test_brle SOURCES tests/test_brle.cpp src/batch/brle.cpp)
# Host oracle: fork geometry -> BRLE decode -> masking == prefix truncation.
pie_add_cuda_target(test_masked_attention_geometry
  SOURCES tests/masked_attention_geometry_test.cpp src/batch/brle.cpp
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/tests)
# GPU parity for the geometry test above, through the real flashinfer prefill-custom path.
pie_add_cuda_target(masked_attention_parity
  SOURCES tests/masked_attention_parity.cu
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/tests
  LIBS pie_driver_cuda_lib CUDA::cudart)
pie_add_cuda_target(test_decode_batch_parity
  SOURCES tests/test_decode_batch_parity.cu
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/tests
  LIBS pie_driver_cuda_lib CUDA::cudart)
pie_add_cuda_target(test_decode_fused_write_parity
  SOURCES tests/test_decode_fused_write_parity.cu
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/tests
  LIBS pie_driver_cuda_lib CUDA::cudart)
pie_add_cuda_target(test_decode_cobatch_chain
  SOURCES tests/test_decode_cobatch_chain.cu
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/tests
  LIBS pie_driver_cuda_lib CUDA::cudart)
pie_add_cuda_target(test_driver_common
  SOURCES tests/test_common.cpp src/loader/safetensors_manifest.cpp
  LIBS nlohmann_json::nlohmann_json)
pie_add_cuda_target(test_kv_cache_format
  SOURCES tests/test_kv_cache_format.cpp src/store/kv_cache_format.cpp)
pie_add_cuda_target(test_kv_cache_quant
  SOURCES
    tests/test_kv_cache_quant.cu
    src/tensor.cpp
    src/store/kv_cache.cpp
    src/store/kv_cache_format.cpp
    src/kernels/kv_paged.cu
  LIBS CUDA::cudart)
pie_add_cuda_target(test_swap_pool
  SOURCES
    tests/test_swap_pool.cpp
    src/tensor.cpp
    src/store/kv_cache.cpp
    src/store/kv_cache_format.cpp
    src/store/swap_pool.cpp
  LIBS CUDA::cudart)
pie_add_cuda_target(test_mxfp4_marlin
  SOURCES tests/test_mxfp4_marlin.cu src/kernels/mxfp4_marlin.cu
  LIBS CUDA::cudart)
pie_add_cuda_target(test_transcode_fused
  SOURCES
    tests/test_transcode_fused.cu
    src/kernels/transcode.cu
    src/kernels/dequant_fp8.cu
    src/kernels/quant_bf16_to_mxfp4.cu
  LIBS CUDA::cudart)
# Model-family full-forward parity harnesses: manual GPU de-risk tools. Most
# textually include the forward under test and remain standalone. Compare
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
pie_add_cuda_target(test_tp_gate
  SOURCES tests/tp_gate_test.cpp
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src
  LIBS Threads::Threads)
pie_add_cuda_target(test_program_identity
  SOURCES tests/program_identity_test.cpp
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../common/include)
pie_add_cuda_target(test_grouped_copy
  SOURCES tests/grouped_copy_test.cpp)
pie_add_cuda_target(test_nucleus_region
  SOURCES tests/nucleus_region_test.cpp
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../common/include
  TEST_ARGS ${CMAKE_CURRENT_SOURCE_DIR}/tests/golden-ptir)
pie_add_cuda_target(test_page_translation
  SOURCES tests/page_translation_test.cpp)
pie_add_cuda_target(test_ptir_graph_key
  SOURCES tests/ptir_graph_key_test.cpp
  INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/include
    ${PIE_DRIVER_ABI_INCLUDE_DIR}
  LIBS CUDA::cudart tomlplusplus::tomlplusplus)
pie_add_cuda_target(test_frame_carrier_device
  SOURCES
    tests/test_frame_carrier_device.cpp
    src/pipeline/frame_carrier.cpp
    src/pipeline/frame_carrier_kernels.cu
  LIBS CUDA::cudart
  TEST_NAME pipeline_frame_carrier_device)
pie_add_cuda_target(test_pipeline_dispatch_bind
  SOURCES
    tests/test_pipeline_dispatch_bind.cpp
    src/pipeline/dispatch.cu
    src/pipeline/frame_carrier.cpp
    src/pipeline/frame_carrier_kernels.cu
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../common/include ${PIE_DRIVER_ABI_INCLUDE_DIR}
  LIBS CUDA::cudart CUDA::cuda_driver CUDA::nvrtc
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)
# Same-instance run-ahead concurrency gate; build with -fsanitize=thread for the TSAN run.
pie_add_cuda_target(test_pipeline_dispatch_race
  SOURCES
    tests/pipeline_dispatch_race_test.cpp
    src/pipeline/dispatch.cu
    src/pipeline/frame_carrier.cpp
    src/pipeline/frame_carrier_kernels.cu
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../common/include ${PIE_DRIVER_ABI_INCLUDE_DIR}
  LIBS CUDA::cudart CUDA::cuda_driver CUDA::nvrtc
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)
pie_add_cuda_target(test_ptir_tier0
  SOURCES tests/ptir_tier0_test.cu
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/tests ${CMAKE_CURRENT_SOURCE_DIR}/../common/include
  LIBS CUDA::cudart)
pie_add_cuda_target(test_ptir_runner
  SOURCES tests/ptir_runner_test.cu
  INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/tests
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/include
    ${PIE_DRIVER_ABI_INCLUDE_DIR}
  LIBS CUDA::cudart)
pie_add_cuda_target(test_ptir_container
  SOURCES tests/ptir_container_test.cpp
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../common/include
  TEST_ARGS ${CMAKE_CURRENT_SOURCE_DIR}/tests/golden-ptir)
# Emits + NVRTC-compiles a fused epilogue kernel; asserts tier-1 == tier-0.
pie_add_cuda_target(test_ptir_tier1
  SOURCES tests/ptir_tier1_test.cu
  INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/tests
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/include
    ${PIE_DRIVER_ABI_INCLUDE_DIR}
  LIBS CUDA::cudart CUDA::cuda_driver CUDA::nvrtc)
pie_add_cuda_target(test_ptir_generated_singleton
  SOURCES tests/ptir_generated_singleton_test.cu
  INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/tests
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/include
    ${PIE_DRIVER_ABI_INCLUDE_DIR}
  LIBS CUDA::cudart CUDA::cuda_driver CUDA::nvrtc
  TEST_ARGS
    ${CMAKE_CURRENT_SOURCE_DIR}/../../interface/ptir/tests/golden-ptir)
pie_add_cuda_target(test_ptir_grouped
  SOURCES tests/ptir_grouped_test.cu
  INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/tests
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/include
    ${PIE_DRIVER_ABI_INCLUDE_DIR}
  LIBS CUDA::cudart CUDA::cuda_driver Threads::Threads)
pie_add_cuda_target(test_ptir_grouped_dispatch
  SOURCES
    tests/ptir_grouped_dispatch_test.cpp
    src/pipeline/dispatch.cu
    src/pipeline/frame_carrier.cpp
    src/pipeline/frame_carrier_kernels.cu
  INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/tests
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/include
    ${PIE_DRIVER_ABI_INCLUDE_DIR}
  LIBS CUDA::cudart CUDA::cuda_driver CUDA::nvrtc
  TEST_ARGS ${CMAKE_CURRENT_SOURCE_DIR}/tests/golden-ptir)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  pie_add_cuda_target(test_ptir_grouped_dispatch_asan
    SOURCES
      tests/ptir_grouped_dispatch_test.cpp
      src/pipeline/dispatch.cu
      src/pipeline/frame_carrier.cpp
      src/pipeline/frame_carrier_kernels.cu
    INCLUDE_DIRS
      ${CMAKE_CURRENT_SOURCE_DIR}/tests
      ${CMAKE_CURRENT_SOURCE_DIR}/../common/include
      ${PIE_DRIVER_ABI_INCLUDE_DIR}
    LIBS CUDA::cudart CUDA::cuda_driver CUDA::nvrtc
    TEST_ARGS ${CMAKE_CURRENT_SOURCE_DIR}/tests/golden-ptir)
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
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../common/include ${PIE_DRIVER_ABI_INCLUDE_DIR}
  LIBS CUDA::cudart
  TEST_ARGS ${CMAKE_CURRENT_SOURCE_DIR}/tests/golden-ptir)
pie_add_cuda_target(test_entry_validation
  SOURCES tests/entry_validation_test.cpp
  LIBS pie_driver_cuda_lib)
pie_add_cuda_target(test_rs_descriptor_validation
  SOURCES tests/rs_descriptor_validation_test.cpp
  INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/include
    ${PIE_DRIVER_ABI_INCLUDE_DIR})
# Structural proof for model/registry.{hpp,cpp}: every model_type resolves to its
# intended family/binder; unrecognized archs are rejected. No GPU, no weights.
pie_add_cuda_target(test_model_registry
  SOURCES tests/model_registry_test.cpp
  LIBS pie_driver_cuda_lib)

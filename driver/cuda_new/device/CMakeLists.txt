# libpie_cuda_device — the thin CUDA device library for driver/cuda_new.
#
# Phase 0 builds the ABI seam (context + lifecycle entries) only. Phase 1
# adds the lifted source sets below (kernels/, ops/, cache/, forward/),
# reusing driver/cuda's CPM deps (flashinfer, cutlass) — wired in when
# those TUs land so the dependency graph stays empty until it's needed.
cmake_minimum_required(VERSION 3.24)
project(PIE_CUDA_DEVICE CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)

find_package(CUDAToolkit REQUIRED)

# ── FlashInfer (header-only consume) ────────────────────────────────────────
# The perf attention path (`src/ops/attention_paged.cu`) instantiates
# FlashInfer's BatchPrefill/BatchDecode paged-KV templates. We only need the
# header tree + 3rdparty/cutlass + spdlog headers — NOT flashinfer's csrc TUs
# or its Python codegen (those are the MoE-cutlass / Hopper-FA3 paths we don't
# lift here). Mirrors driver/cuda's CPM fetch (same v0.6.9 pin).
include(cmake/CPM.cmake)
if(DEFINED ENV{CPM_SOURCE_CACHE} AND NOT CPM_SOURCE_CACHE)
  set(CPM_SOURCE_CACHE $ENV{CPM_SOURCE_CACHE})
endif()
# Allow an explicit pre-fetched source dir (set by build.rs via -D, or env) to
# skip the network clone entirely; otherwise CPM fetches v0.6.9.
if(NOT PIE_FLASHINFER_SOURCE_DIR AND DEFINED ENV{PIE_FLASHINFER_SOURCE_DIR})
  set(PIE_FLASHINFER_SOURCE_DIR "$ENV{PIE_FLASHINFER_SOURCE_DIR}")
endif()
if(PIE_FLASHINFER_SOURCE_DIR AND EXISTS "${PIE_FLASHINFER_SOURCE_DIR}")
  set(flashinfer_SOURCE_DIR "${PIE_FLASHINFER_SOURCE_DIR}")
  message(STATUS "Using pre-fetched FlashInfer at ${flashinfer_SOURCE_DIR}")
else()
  cpmaddpackage(
    NAME flashinfer
    GITHUB_REPOSITORY flashinfer-ai/flashinfer
    GIT_TAG v0.6.9
    GIT_SHALLOW TRUE
    DOWNLOAD_ONLY YES)
endif()
set(PIE_FLASHINFER_INCLUDE_DIRS
  ${flashinfer_SOURCE_DIR}/include
  ${flashinfer_SOURCE_DIR}/csrc
  ${flashinfer_SOURCE_DIR}/3rdparty/cutlass/include
  ${flashinfer_SOURCE_DIR}/3rdparty/cutlass/tools/util/include
  ${flashinfer_SOURCE_DIR}/3rdparty/spdlog/include)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  # Frontier-only: default to SM 9.0 (Hopper/H100). `native` was unreliable
  # here — CMake's GPU probe fell back to 75 on this toolchain, which strips the
  # bf16 conversion members (guarded by `__CUDA_ARCH__ >= 800`) in the quant
  # GEMM and fails the build. server/build.rs forwards CMAKE_CUDA_ARCHITECTURES
  # from the env, so CI can still override (e.g. "90;100").
  set(CMAKE_CUDA_ARCHITECTURES 90)
endif()

add_library(pie_cuda_device STATIC
  src/context.cpp
  src/abi.cpp
  # --- phase 1: kernels lifted from driver/cuda, behind the ABI ---
  src/kernels/rmsnorm.cu
  src/kernels/residual_add.cu
  src/kernels/swiglu.cu
  src/kernels/rope.cu
  src/kernels/embed.cu
  src/kernels/argmax.cu
  src/kernels/kv_append.cu
  src/kernels/sample.cu
  src/kernels/dtype_cast.cu
  src/kernels/gather_rows.cu
  src/kernels/gemma.cu
  src/kernels/dequant_fp8.cu
  src/kernels/rope_yarn.cu
  src/kernels/moe.cu
  src/kernels/rope_partial.cu
  src/kernels/causal_conv1d.cu
  src/kernels/dequant_wna16.cu
  src/kernels/altup.cu
  src/kernels/qk_norm.cu
  src/kernels/add_bias.cu
  src/kernels/moe_dispatch.cu
  src/kernels/ssm_scan.cu
  src/kernels/mamba_proj.cu
  src/ops/gemm.cpp
  src/ops/grouped_gemm.cu
  src/ops/attention_naive_paged.cu
  # --- fast paged attention (tensor-core; plan/dispatch scheduler; perf path) + its plan workspace ---
  src/ops/attention_workspace.cpp
  src/ops/attention_paged.cu
  # --- internalized de-branded fused quant GEMM (int4/u4b8 slice) ---
  src/qgemm/qgemm.cu
  src/qgemm/qgemm_kernel_inst.cu
  src/qgemm/qgemm_repack.cu
  src/ops/mla_paged.cu
  src/kernels/mla_write.cu
  src/forward/attn_runtime.cu
  src/forward/llama_layer.cu
  src/forward/llama_forward.cu
  src/forward/moe_mlp.cu
  src/forward/moe_forward.cu
  src/forward/moe_sparse.cu
  src/forward/mla_block.cu
  src/forward/mla_forward.cu
  src/forward/deepseek_forward.cu
  src/forward/gemma_layer.cu
  src/forward/gemma_forward.cu
  src/forward/nemotron_block.cu
  src/forward/nemotron_forward.cu
  # (cache/ + more forward/ lifted incrementally as the body path lands)
)

target_include_directories(pie_cuda_device PUBLIC include)
# src/ is on the private include path so cross-TU quoted includes (e.g. abi.cpp
# pulling qgemm/qgemm.h, and the qgemm TUs' sibling headers) resolve uniformly.
target_include_directories(pie_cuda_device PRIVATE src)
# FlashInfer template headers (attention_paged.cu only). Kept PRIVATE so the
# ABI seam doesn't leak the upstream templates into downstream consumers.
target_include_directories(pie_cuda_device PRIVATE ${PIE_FLASHINFER_INCLUDE_DIRS})
target_link_libraries(pie_cuda_device PUBLIC CUDA::cudart CUDA::cublas)
set_target_properties(pie_cuda_device PROPERTIES POSITION_INDEPENDENT_CODE ON)

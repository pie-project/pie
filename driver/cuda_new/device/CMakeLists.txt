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
  # --- internalized de-branded fused quant GEMM (int4/u4b8 slice) ---
  src/qgemm/qgemm.cu
  src/qgemm/qgemm_kernel_inst.cu
  src/qgemm/qgemm_repack.cu
  src/ops/mla_paged.cu
  src/kernels/mla_write.cu
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
target_link_libraries(pie_cuda_device PUBLIC CUDA::cudart CUDA::cublas)
set_target_properties(pie_cuda_device PROPERTIES POSITION_INDEPENDENT_CODE ON)

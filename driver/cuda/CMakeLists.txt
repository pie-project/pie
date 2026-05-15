cmake_minimum_required(VERSION 3.23)

include(cmake/DetectCudaArchitecture.cmake)
detect_cuda_architectures()
message(STATUS "Using CUDA architecture: ${CMAKE_CUDA_ARCHITECTURES}")

project(PIE_DRIVER_CUDA CUDA CXX)

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

if(NOT PIE_BRIDGE_INCLUDE_DIR)
  set(PIE_BRIDGE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../bridge/include")
endif()

# NCCL for tensor-parallel collectives. Single-GPU runs never load it (the
# call sites are gated on tp_size > 1), but we always link so a TP-enabled
# wrapper can talk to the same binary. System NCCL only — vendoring via CPM
# is impractical (kernel-heavy build).
find_path(NCCL_INCLUDE_DIR nccl.h
  HINTS ${CUDAToolkit_INCLUDE_DIRS} /usr/include /usr/local/include
  REQUIRED)
find_library(NCCL_LIBRARY nccl
  HINTS ${CUDAToolkit_LIBRARY_DIR} /usr/lib/x86_64-linux-gnu /usr/local/lib
  REQUIRED)
message(STATUS "Using NCCL: ${NCCL_LIBRARY} (header: ${NCCL_INCLUDE_DIR})")

include(cmake/CPM.cmake)

# NOTE: flashinfer dropped its top-level CMakeLists.txt after v0.2.7. From
# v0.3.0+ the build is Python-driven (per-arch JIT cubins via build_backend.py).
# We still fetch the source tree for headers and `csrc/`. Kernel integration
# at M1.2 will either (a) compile the csrc TUs we need into our binary
# directly, or (b) consume the precompiled `flashinfer-cubin` artifacts via
# the cuModule API.
cpmaddpackage(
  NAME flashinfer
  GITHUB_REPOSITORY flashinfer-ai/flashinfer
  GIT_TAG v0.6.9
  DOWNLOAD_ONLY YES)

cpmaddpackage(URI "gh:marzer/tomlplusplus@3.4.0")
cpmaddpackage(URI "gh:CLIUtils/CLI11@2.5.0")
cpmaddpackage(URI "gh:nlohmann/json@3.12.0" OPTIONS "JSON_BuildTests OFF")
cpmaddpackage(URI "gh:facebook/zstd@1.5.7")


# pie_driver_cuda — native CUDA backend. Built two ways from the same
# translation units:
#   - `pie_driver_cuda_lib` (STATIC): linked in-process by server/standalone
#     when the `driver-cuda` Cargo feature is on, so the all-Rust binary
#     contains the driver. Per the M3 static-link spike below, this works
#     with `cublas_static + cublasLt_static + cudart_static + culibos +
#     pthread + rt + dl` and yields a binary with zero CUDA shared deps.
#   - `pie_driver_cuda` (executable): a thin shim around the lib's
#     `pie_driver_cuda_run` entry. Consumed by the Python wrapper
#     `pie_driver_cuda_native` via subprocess.
add_library(pie_driver_cuda_lib STATIC
  src/entry.cpp
  src/executor/persistent_inputs.cpp
  src/executor/executor.cpp
  src/service/inproc_service.cpp
  src/response_subpass.cpp
  src/sampling_dispatch.cpp
  src/spec_expansion.cpp
  src/tensor.cpp
  src/model/loaded_model.cpp
  ../common/src/safetensors_manifest.cpp
  src/loader/hf_config.cpp
  src/loader/safetensors.cpp
  src/model/qwen3.cpp
  src/model/qwen3_forward.cpp
  src/model/llama_like.cpp
  src/model/mixtral.cpp
  src/model/gemma2.cpp
  src/model/gemma4.cpp
  src/model/gemma3n.cpp
  src/model/mistral3.cpp
  src/model/gpt_oss.cpp
  src/model/qwen3_5.cpp
  src/kv_cache.cpp
  src/ops/gemm.cpp
  src/ops/attention_naive.cu
  src/ops/attention_naive_paged.cu
  src/kernels/rmsnorm.cu
  src/kernels/embed.cu
  src/kernels/swiglu.cu
  src/kernels/rope.cu
  src/kernels/split_packed.cu
  src/kernels/scatter_int32.cu
  src/kernels/argmax.cu
  src/kernels/kv_paged.cu
  src/kernels/sample_temp.cu
  src/kernels/sample_flashinfer.cu
  src/kernels/entropy.cu
  src/kernels/logprobs.cu
  src/kernels/dist.cu
  src/kernels/gather_rows.cu
  src/kernels/add_bias.cu
  src/kernels/residual_add.cu
  src/kernels/topk_softmax.cu
  src/kernels/moe_dispatch.cu
  src/kernels/softcap.cu
  src/kernels/scalar_mul.cu
  src/kernels/dequant_fp4.cu
  src/kernels/dequant_fp8.cu
  src/kernels/dtype_cast.cu
  src/kernels/quant_bf16_to_fp8.cu
  src/kernels/head_dim_pad.cu
  src/kernels/attn_sink.cu
  src/kernels/deinterleave.cu
  src/kernels/gaussian_topk.cu
  src/kernels/altup.cu
  src/kernels/altup_aux.cu
  src/kernels/causal_conv1d.cu
  src/kernels/gated_delta_net.cu
  src/qwen3_5_state_cache.cpp
  src/slot_allocator.cpp
  src/model/qwen3_5_forward.cpp
  src/model/qwen3_5_moe.cpp
  src/model/qwen3_5_moe_forward.cpp
  src/attention_workspace.cpp
  src/brle.cpp
  src/swap_pool.cpp
  src/distributed.cpp
  src/custom_all_reduce.cu
  src/ops/attention_flashinfer.cu
)

add_executable(pie_driver_cuda src/main.cpp)
target_link_libraries(pie_driver_cuda PRIVATE pie_driver_cuda_lib)

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
# Marlin sources are added to the `pie_driver_cuda` executable below
# (see `PIE_CUDA_BUILD_MARLIN`); they need the same nvcc relaxed-
# constexpr / extended-lambda flags the lib gets, since vendored
# kernels mix `__host__ constexpr` helpers into `__device__` paths.
target_compile_options(pie_driver_cuda PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    --extended-lambda
    --expt-relaxed-constexpr
  >
)

target_include_directories(pie_driver_cuda_lib
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/include
    ${PIE_BRIDGE_INCLUDE_DIR}
    ${NCCL_INCLUDE_DIR}
    ${flashinfer_SOURCE_DIR}/include
    ${flashinfer_SOURCE_DIR}/3rdparty/cutlass/include
    ${flashinfer_SOURCE_DIR}/3rdparty/cutlass/tools/util/include
    ${flashinfer_SOURCE_DIR}/3rdparty/spdlog/include)

# ── Marlin (W4A16 GEMM via vendored vLLM kernel) ─────────────────────
# Off by default while M3 plumbing lands. Vendored sources live under
# `third_party/marlin/`; the wrapper exposes a torch-free C++ API.
# Compile cost is ~3 minutes on first build (heavy templates) so we
# gate it behind an explicit switch until a model actually uses it.
option(PIE_CUDA_BUILD_MARLIN "Build vendored marlin W4A16 GEMM" OFF)
if(PIE_CUDA_BUILD_MARLIN)
    # Include the generated per-arch kernel instantiations alongside the
    # core driver. `generate_kernels.py` produced sm80 variants targeting
    # A100; rerun with a different arch to add Hopper / Ada coverage.
    file(GLOB MARLIN_SM_KERNELS
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/marlin/sm*_kernel_*.cu")
    target_sources(pie_driver_cuda PRIVATE
        third_party/marlin/marlin.cu
        third_party/marlin/gptq_marlin_repack.cu
        third_party/marlin/awq_marlin_repack.cu
        third_party/marlin/marlin_wrapper.cpp
        ${MARLIN_SM_KERNELS})
    target_include_directories(pie_driver_cuda PRIVATE
        third_party/marlin)
    target_compile_definitions(pie_driver_cuda PRIVATE PIE_CUDA_HAS_MARLIN=1)
endif()

# Standalone test for flashinfer sampling debugging.
add_executable(sample_flashinfer_test tests/sample_flashinfer_test.cu)
target_include_directories(sample_flashinfer_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${flashinfer_SOURCE_DIR}/include
    ${flashinfer_SOURCE_DIR}/3rdparty/cutlass/include
    ${flashinfer_SOURCE_DIR}/3rdparty/cutlass/tools/util/include
    ${flashinfer_SOURCE_DIR}/3rdparty/spdlog/include)
target_compile_options(sample_flashinfer_test PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>: --extended-lambda --expt-relaxed-constexpr >)
target_link_libraries(sample_flashinfer_test PRIVATE CUDA::cudart)

# Wire-format unit tests. Pure C++ (no CUDA), short runtime; CTest runs
# them under `cmake --build build --target test`.
add_executable(test_brle tests/test_brle.cpp src/brle.cpp)
target_include_directories(test_brle PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable(test_driver_common
  ../common/tests/test_common.cpp
  ../common/src/safetensors_manifest.cpp)
target_include_directories(test_driver_common PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../common/include)
target_link_libraries(test_driver_common PRIVATE nlohmann_json::nlohmann_json)

enable_testing()
add_test(NAME brle COMMAND test_brle)
add_test(NAME driver_common COMMAND test_driver_common)

target_link_libraries(pie_driver_cuda_lib PUBLIC
  CLI11::CLI11
  tomlplusplus::tomlplusplus
  zstd
  nlohmann_json::nlohmann_json
  rt
  CUDA::cudart
  CUDA::cublas
  CUDA::cublasLt
  CUDA::cuda_driver
  ${NCCL_LIBRARY}
)

# M3 static-link spike notes: a hello-world calling cublasCreate/Destroy
# links cleanly with `-lcublas_static -lcublasLt_static -lcudart_static
# -lculibos -lpthread -lrt -ldl`, runs at runtime, and yields `ldd`
# output with zero non-system shared libs. M4 wires server/standalone's
# `--features driver-cuda` build.rs to consume `pie_driver_cuda_lib`
# with those static link flags (mirror the portable side's `BUILD_SHARED_LIBS=OFF`
# trick, plus explicit `-l:libcublas_static.a` group).

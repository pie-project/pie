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
find_package(Threads REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

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

set(PIE_CUDA_CUSTOM_ALL_REDUCE_SOURCE src/custom_all_reduce.cu)
foreach(PIE_CUDA_ARCH IN LISTS CMAKE_CUDA_ARCHITECTURES)
  string(REGEX REPLACE "[^0-9].*$" "" PIE_CUDA_ARCH_NUMBER "${PIE_CUDA_ARCH}")
  if(PIE_CUDA_ARCH_NUMBER STREQUAL "120")
    set(PIE_CUDA_CUSTOM_ALL_REDUCE_SOURCE src/custom_all_reduce_stub.cpp)
  endif()
endforeach()
if(PIE_CUDA_CUSTOM_ALL_REDUCE_SOURCE MATCHES "stub")
  message(STATUS "Disabling FlashInfer custom all-reduce for CUDA architecture ${CMAKE_CUDA_ARCHITECTURES}")
endif()

set(PIE_CUDA_FLASHINFER_HOPPER_SOURCE src/ops/attention_flashinfer_hopper_stub.cpp)
set(PIE_CUDA_FLASHINFER_MAMBA_SM90 OFF)
foreach(PIE_CUDA_ARCH IN LISTS CMAKE_CUDA_ARCHITECTURES)
  string(REGEX REPLACE "[^0-9].*$" "" PIE_CUDA_ARCH_NUMBER "${PIE_CUDA_ARCH}")
  if(PIE_CUDA_ARCH_NUMBER STREQUAL "90")
    set(PIE_CUDA_FLASHINFER_HOPPER_SOURCE
      src/ops/attention_flashinfer_hopper.cu
      src/ops/attention_xqa_gqa8_sm90.cu
    )
    set(PIE_CUDA_FLASHINFER_MAMBA_SM90 ON)
  endif()
  # FlashInfer's Mamba SSU vertical/horizontal kernels use TMA +
  # memcpy_async_tx, which Blackwell also supports. The runtime
  # algorithm auto-selector in selective_state_update prefers the
  # horizontal kernel for sm >= 10 with bf16/fp16 state — so the same
  # FLASHINFER_MAMBA_ENABLE_SM90 compile gate should be on for sm_100.
  if(PIE_CUDA_ARCH_NUMBER STREQUAL "100")
    set(PIE_CUDA_FLASHINFER_MAMBA_SM90 ON)
  endif()
endforeach()
if(PIE_CUDA_FLASHINFER_HOPPER_SOURCE MATCHES "stub")
  message(STATUS "Disabling FlashInfer Hopper/FA3 attention for CUDA architecture ${CMAKE_CUDA_ARCHITECTURES}")
else()
  message(STATUS "Enabling FlashInfer Hopper/FA3 attention for CUDA architecture ${CMAKE_CUDA_ARCHITECTURES}")
endif()
if(PIE_CUDA_FLASHINFER_MAMBA_SM90)
  message(STATUS "Enabling FlashInfer Mamba SM90 vertical/horizontal SSU kernels")
else()
  message(STATUS "FlashInfer Mamba SM90 vertical/horizontal SSU kernels disabled (legacy + simple algorithm only)")
endif()

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
  src/model/weight_store.cpp
  src/model/loaded_model.cpp
  src/model/bound_model.cpp
  ../common/src/safetensors_manifest.cpp
  src/loader/gguf_source.cpp
  src/loader/hf_config.cpp
  src/loader/safetensors.cpp
  src/model/qwen3.cpp
  src/model/qwen3_forward.cpp
  src/model/llama_like.cpp
  src/model/mixtral.cpp
  src/model/gemma2.cpp
  src/model/gemma4.cpp
  src/model/gemma4_mtp.cpp
  src/model/gemma3n.cpp
  src/model/mistral3.cpp
  src/model/gpt_oss.cpp
  src/model/qwen3_5.cpp
  src/model/nemotron_h.cpp
  src/kv_cache_format.cpp
  src/kv_cache.cpp
  src/ops/gemm.cpp
  src/ops/attention_naive.cu
  src/ops/attention_naive_paged.cu
  src/ops/attention_xqa.cu
  src/ops/attention_xqa_gqa2.cu
  src/ops/attention_xqa_gqa2_p16.cu
  src/ops/attention_xqa_gqa4.cu
  src/ops/attention_xqa_gqa8.cu
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
  src/kernels/mxfp4_marlin.cu
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
  src/kernels/nemotron_h.cu
  src/recurrent_state_cache.cpp
  src/cuda_memory_planner.cpp
  src/driver_startup.cpp
  src/parity_harness.cpp
  src/hf_snapshot.cpp
  src/model/qwen3_5_config.cpp
  src/model/qwen3_5_forward.cpp
  src/model/qwen3_5_moe.cpp
  src/model/qwen3_5_moe_forward.cpp
  src/model/nemotron_h_forward.cpp
  src/model/nemotron_h_model.cpp
  src/model/gemma4_model.cpp
  src/model/gemma2_model.cpp
  src/model/gemma3n_model.cpp
  src/model/mixtral_model.cpp
  src/model/qwen3_5_model.cpp
  src/model/qwen3_5_moe_model.cpp
  src/model/llama_like_model.cpp
  src/attention_workspace.cpp
  src/brle.cpp
  src/swap_pool.cpp
  src/distributed.cpp
  ${PIE_CUDA_CUSTOM_ALL_REDUCE_SOURCE}
  src/ops/attention_flashinfer.cu
  src/ops/flashinfer_mamba.cu
  ${PIE_CUDA_FLASHINFER_HOPPER_SOURCE}
)

find_program(CARGO_EXECUTABLE cargo REQUIRED)
set(PIE_WORKSPACE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(PIE_WEIGHT_LOADER_MANIFEST
    "${PIE_WORKSPACE_ROOT}/driver/weight_loader/Cargo.toml")
set(PIE_WEIGHT_LOADER_STATICLIB_PATH
    "${PIE_WORKSPACE_ROOT}/target/release/libpie_weight_loader.a")
add_custom_target(pie_weight_loader_staticlib
  COMMAND "${CARGO_EXECUTABLE}" build
          --manifest-path "${PIE_WEIGHT_LOADER_MANIFEST}"
          --release
  WORKING_DIRECTORY "${PIE_WORKSPACE_ROOT}"
  BYPRODUCTS "${PIE_WEIGHT_LOADER_STATICLIB_PATH}"
  COMMENT "Building Rust weight-loader static library")
add_library(pie_weight_loader_static STATIC IMPORTED GLOBAL)
set_target_properties(pie_weight_loader_static PROPERTIES
  IMPORTED_LOCATION "${PIE_WEIGHT_LOADER_STATICLIB_PATH}")
add_dependencies(pie_weight_loader_static pie_weight_loader_staticlib)

set(PIE_WEIGHT_LOADER_SYSTEM_LIBS Threads::Threads ${CMAKE_DL_LIBS} m)
if(UNIX AND NOT APPLE)
  list(APPEND PIE_WEIGHT_LOADER_SYSTEM_LIBS rt util)
endif()

add_executable(pie_driver_cuda src/main.cpp)
target_link_libraries(pie_driver_cuda PRIVATE
  pie_driver_cuda_lib
  # The C++ declarations are weak so server builds can link the CUDA driver
  # archive before Cargo supplies Rust symbols. For the standalone parity
  # binary, force the Rust staticlib into the link; weak references alone do
  # not cause archive extraction.
  "-Wl,--whole-archive"
  pie_weight_loader_static
  "-Wl,--no-whole-archive"
  ${PIE_WEIGHT_LOADER_SYSTEM_LIBS})

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
# Marlin sources are added to the static driver library below
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
    ${flashinfer_SOURCE_DIR}/csrc
    ${flashinfer_SOURCE_DIR}/3rdparty/cutlass/include
    ${flashinfer_SOURCE_DIR}/3rdparty/cutlass/tools/util/include
    ${flashinfer_SOURCE_DIR}/3rdparty/spdlog/include)

# FlashInfer's CUTLASS MoE path expects generated launcher instantiations
# beside the static csrc files. Generate only the SM80 grouped-GEMM launchers
# needed by the SM89/Ada BF16 MoE path instead of pulling in the entire TVM
# extension module.
list(GET CMAKE_CUDA_ARCHITECTURES 0 PIE_FLASHINFER_MOE_ARCH)
string(REGEX REPLACE "[^0-9].*$" "" PIE_FLASHINFER_MOE_ARCH_NUMBER
  "${PIE_FLASHINFER_MOE_ARCH}")
set(PIE_FLASHINFER_MOE_GEN_DIR
  "${CMAKE_BINARY_DIR}/flashinfer_generated/cutlass_instantiations/${PIE_FLASHINFER_MOE_ARCH_NUMBER}")
file(MAKE_DIRECTORY "${PIE_FLASHINFER_MOE_GEN_DIR}")
execute_process(
  COMMAND ${CMAKE_COMMAND} -E env
          FLASHINFER_DISABLE_VERSION_CHECK=1
          PYTHONPATH=${flashinfer_SOURCE_DIR}
          ${Python3_EXECUTABLE} -c
          "from pathlib import Path; from flashinfer.jit.gemm.cutlass.generate_kernels import generate_gemm_operations; generate_gemm_operations(Path(r'${PIE_FLASHINFER_MOE_GEN_DIR}'), '${PIE_FLASHINFER_MOE_ARCH_NUMBER};${PIE_FLASHINFER_MOE_ARCH_NUMBER}-real')"
  RESULT_VARIABLE PIE_FLASHINFER_MOE_GEN_RESULT)
if(NOT PIE_FLASHINFER_MOE_GEN_RESULT EQUAL 0)
  message(FATAL_ERROR "Failed to generate FlashInfer CUTLASS MoE launchers")
endif()
file(GLOB PIE_FLASHINFER_MOE_GENERATED_SOURCES CONFIGURE_DEPENDS
  "${PIE_FLASHINFER_MOE_GEN_DIR}/gemm_grouped/80/*.generated.cu")
list(LENGTH PIE_FLASHINFER_MOE_GENERATED_SOURCES PIE_FLASHINFER_MOE_GENERATED_COUNT)
message(STATUS "Using ${PIE_FLASHINFER_MOE_GENERATED_COUNT} FlashInfer generated MoE launchers")

# FlashInfer/TensorRT-LLM CUTLASS MoE runner. Keep this isolated from the
# regular driver TUs because the dependency slice is large and should remain a
# clear opt-in boundary for Nemotron experimentation.
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
  $<$<COMPILE_LANGUAGE:CUDA>:
    --extended-lambda
    --expt-relaxed-constexpr
  >)
set_target_properties(pie_flashinfer_cutlass_moe PROPERTIES
  POSITION_INDEPENDENT_CODE ON)
target_sources(pie_driver_cuda_lib PRIVATE
  $<TARGET_OBJECTS:pie_flashinfer_cutlass_moe>)

# ── Marlin (W4A16 / MXFP4 GEMM via vendored vLLM kernel) ──────────────
# Built by default because GPT-OSS MXFP4 auto policy should use native
# packed-weight GEMM on supported GPUs and fall back only on legacy targets.
# Vendored sources live under `third_party/marlin/`; the wrapper exposes a
# torch-free C++ API.
option(PIE_CUDA_BUILD_MARLIN "Build vendored marlin W4A16/MXFP4 GEMM" ON)
if(PIE_CUDA_BUILD_MARLIN)
	    # Include the generated kernel instantiations alongside the core driver.
    file(GLOB MARLIN_SM_KERNELS
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/marlin/sm*_kernel_*.cu")
    target_sources(pie_driver_cuda_lib PRIVATE
        third_party/marlin/marlin.cu
        third_party/marlin/gptq_marlin_repack.cu
        third_party/marlin/awq_marlin_repack.cu
        third_party/marlin/marlin_wrapper.cpp
        ${MARLIN_SM_KERNELS})
    target_include_directories(pie_driver_cuda_lib PRIVATE
        third_party/marlin)
    target_compile_definitions(pie_driver_cuda_lib PRIVATE PIE_CUDA_HAS_MARLIN=1)
endif()

# FlashInfer Mamba SSU on Hopper / Blackwell gates the vertical and horizontal
# producer-consumer kernels behind FLASHINFER_MAMBA_ENABLE_SM90 at compile
# time (TMA + memcpy_async_tx are sm_90+). Without this define, sm_90+ runs
# would silently fall back to the simple algorithm — the same slow path used
# on sm_89 — even though the runtime SM check would otherwise pick vertical
# or horizontal.
if(PIE_CUDA_FLASHINFER_MAMBA_SM90)
    target_compile_definitions(pie_driver_cuda_lib PRIVATE
        FLASHINFER_MAMBA_ENABLE_SM90=1)
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

# Qwen3.5 linear-attention conv-state regression. This catches mixed
# prefill/decode batches where a continuing T=1 request must consume its
# existing conv window even when the whole fire is not pure decode.
add_executable(test_causal_conv1d
  tests/test_causal_conv1d.cu
  src/kernels/causal_conv1d.cu)
target_include_directories(test_causal_conv1d PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_options(test_causal_conv1d PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>: --extended-lambda --expt-relaxed-constexpr >)
target_link_libraries(test_causal_conv1d PRIVATE CUDA::cudart)

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

add_executable(test_kv_cache_format
  tests/test_kv_cache_format.cpp
  src/kv_cache_format.cpp)
target_include_directories(test_kv_cache_format PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable(test_kv_cache_quant
  tests/test_kv_cache_quant.cu
  src/tensor.cpp
  src/kv_cache.cpp
  src/kv_cache_format.cpp
  src/kernels/kv_paged.cu)
target_include_directories(test_kv_cache_quant PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_options(test_kv_cache_quant PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>: --extended-lambda --expt-relaxed-constexpr >)
target_link_libraries(test_kv_cache_quant PRIVATE CUDA::cudart)

add_executable(test_swap_pool
  tests/test_swap_pool.cpp
  src/tensor.cpp
  src/kv_cache.cpp
  src/kv_cache_format.cpp
  src/swap_pool.cpp)
target_include_directories(test_swap_pool PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(test_swap_pool PRIVATE CUDA::cudart)

add_executable(test_mxfp4_marlin
  tests/test_mxfp4_marlin.cu
  src/kernels/mxfp4_marlin.cu)
target_include_directories(test_mxfp4_marlin PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_options(test_mxfp4_marlin PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>: --extended-lambda --expt-relaxed-constexpr >)
target_link_libraries(test_mxfp4_marlin PRIVATE CUDA::cudart)

enable_testing()
add_test(NAME brle COMMAND test_brle)
add_test(NAME causal_conv1d COMMAND test_causal_conv1d)
add_test(NAME driver_common COMMAND test_driver_common)
add_test(NAME kv_cache_format COMMAND test_kv_cache_format)
add_test(NAME kv_cache_quant COMMAND test_kv_cache_quant)
add_test(NAME swap_pool COMMAND test_swap_pool)
add_test(NAME mxfp4_marlin COMMAND test_mxfp4_marlin)

target_link_libraries(pie_driver_cuda_lib PUBLIC
  CLI11::CLI11
  tomlplusplus::tomlplusplus
  zstd
  nlohmann_json::nlohmann_json
  rt
  dl
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

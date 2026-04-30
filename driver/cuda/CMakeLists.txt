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


# pie_driver_cuda — native CUDA backend, sibling to ../../runtime (Rust) and
# ../../pie (Python driver). Talks to the runtime via the shmem fast path;
# no ZMQ/protobuf needed. Consumed by the Python wrapper `pie_driver_cuda_native`.
add_executable(pie_driver_cuda
  src/main.cpp
  src/shmem_ipc.cpp
  src/shmem_schema.cpp
  src/tensor.cpp
  src/engine.cpp
  src/loader/hf_config.cpp
  src/loader/safetensors.cpp
  src/model/qwen3.cpp
  src/model/qwen3_forward.cpp
  src/kv_cache.cpp
  src/ops/gemm.cpp
  src/ops/attention_naive.cu
  src/ops/attention_paged.cu
  src/response_writer.cpp
  src/kernels/rmsnorm.cu
  src/kernels/embed.cu
  src/kernels/swiglu.cu
  src/kernels/rope.cu
  src/kernels/argmax.cu
  src/kernels/kv_paged.cu
  src/kernels/sample_temp.cu
  src/kernels/sample_flashinfer.cu
  src/attention_workspace.cpp
  src/ops/attention_flashinfer.cu
)

target_compile_options(pie_driver_cuda PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    -Xcompiler=-Wall
    -Xcompiler=-Wextra
    -Wno-unused-parameter
    --extended-lambda
    --expt-relaxed-constexpr
  >
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wno-unused-parameter>
)

target_include_directories(pie_driver_cuda
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${flashinfer_SOURCE_DIR}/include
    ${flashinfer_SOURCE_DIR}/3rdparty/cutlass/include
    ${flashinfer_SOURCE_DIR}/3rdparty/cutlass/tools/util/include
    ${flashinfer_SOURCE_DIR}/3rdparty/spdlog/include)

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

target_link_libraries(pie_driver_cuda PRIVATE
  CLI11::CLI11
  tomlplusplus::tomlplusplus
  zstd
  nlohmann_json::nlohmann_json
  rt
  CUDA::cudart
  CUDA::cublas
  CUDA::cublasLt
)

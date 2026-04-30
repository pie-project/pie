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

cpmaddpackage(
  URI "gh:flashinfer-ai/flashinfer@0.2.7"
  OPTIONS
    "FLASHINFER_ENABLE_FP8 OFF"
    "FLASHINFER_ENABLE_FP8_E4M3 OFF"
    "FLASHINFER_ENABLE_FP8_E5M2 OFF"
    "FLASHINFER_GEN_POS_ENCODING_MODES 0")

cpmaddpackage(URI "gh:marzer/tomlplusplus@3.4.0")
cpmaddpackage(URI "gh:CLIUtils/CLI11@2.5.0")
cpmaddpackage(URI "gh:nlohmann/json@3.12.0" OPTIONS "JSON_BuildTests OFF")
cpmaddpackage(URI "gh:facebook/zstd@1.5.7")

file(GLOB_RECURSE FLASHINFER_GENERATED_SOURCES
     "${flashinfer_SOURCE_DIR}/src/generated/*.cu")

# pie_driver_cuda — native CUDA backend, sibling to ../../runtime (Rust) and
# ../../pie (Python driver). Talks to the runtime via the shmem fast path;
# no ZMQ/protobuf needed. Consumed by the Python wrapper `pie_driver_cuda_native`.
add_executable(pie_driver_cuda
  src/main.cpp
  src/shmem_ipc.cpp
  src/shmem_schema.cpp
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
    ${flashinfer_SOURCE_DIR}/src/generated)

target_link_libraries(pie_driver_cuda PRIVATE
  CLI11::CLI11
  tomlplusplus::tomlplusplus
  zstd
  nlohmann_json::nlohmann_json
  rt
  CUDA::cublas
  CUDA::cublasLt
)

# Set minimum CMake version and define the project. This enables CUDA as a
# first-class language.
cmake_minimum_required(VERSION 3.23)

# --- CUDA compiler auto-detection (runs before project() so it can take effect) ---
# If the user didn't set CMAKE_CUDA_COMPILER / CUDACXX but nvcc exists at a common path,
# set it explicitly to avoid configuration failures on systems where CMake can't locate CUDA.
if(NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED CUDACXX)
  if(EXISTS "/usr/local/cuda/bin/nvcc")
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE FILEPATH "Path to nvcc compiler" FORCE)
    message(STATUS "Auto-detected nvcc at /usr/local/cuda/bin/nvcc")
  elseif(DEFINED ENV{CUDA_HOME} AND EXISTS "$ENV{CUDA_HOME}/bin/nvcc")
    set(CMAKE_CUDA_COMPILER "$ENV{CUDA_HOME}/bin/nvcc" CACHE FILEPATH "Path to nvcc compiler" FORCE)
    message(STATUS "Auto-detected nvcc at $ENV{CUDA_HOME}/bin/nvcc")
  endif()
endif()

# Detect CUDA architecture
include(cmake/DetectCudaArchitecture.cmake)
detect_cuda_architectures()
message(STATUS "Using CUDA architecture: ${CMAKE_CUDA_ARCHITECTURES}")

project(PIE_BACKEND CUDA CXX)

set(CMAKE_BUILD_TYPE Release)

# Set C++ Standard and output directories Ensures C++20 is used, matching the
# ninja file's flag.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Define output directories for binaries, matching the ninja file structure.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

find_package(CUDAToolkit REQUIRED)

# Include CPM.cmake for package management This downloads the CPM script if it's
# not already present.
include(cmake/CPM.cmake)

# Declare and fetch all dependencies using CPM CPM downloads, configures, and
# makes targets available for all dependencies. This replaces manual submodule
# management.
cpmaddpackage(
  URI
  "gh:flashinfer-ai/flashinfer@0.2.7"
  OPTIONS
  "FLASHINFER_ENABLE_FP8 OFF"
  "FLASHINFER_ENABLE_FP8_E4M3 OFF"
  "FLASHINFER_ENABLE_FP8_E5M2 OFF"
  "FLASHINFER_GEN_POS_ENCODING_MODES 0")

cpmaddpackage(
  URI
  "gh:protocolbuffers/protobuf@6.31.1"
  OPTIONS
  "protobuf_FORCE_FETCH_DEPENDENCIES ON"
  "protobuf_BUILD_SHARED_LIBS OFF"
  "protobuf_BUILD_TESTS OFF"
  "protobuf_INSTALL OFF"
  "protobuf_BUILD_EXAMPLES OFF")

cpmaddpackage(URI "gh:zeromq/cppzmq@4.10.0")

cpmaddpackage(URI "gh:machinezone/IXWebSocket@11.4.6")

cpmaddpackage(URI "gh:msgpack/msgpack-c#cpp-7.0.0" OPTIONS "MSGPACK_CXX20 ON"
              "MSGPACK_USE_BOOST OFF" "MSGPACK_BUILD_DOCS OFF")

# for cbor parsing in zTensor maybe i can switch to libcbor later
cpmaddpackage(URI "gh:nlohmann/json@3.12.0" OPTIONS "JSON_BuildTests OFF")

# for zTensor zstd decompression support
cpmaddpackage(URI "gh:facebook/zstd@1.5.7")

cpmaddpackage(URI "gh:marzer/tomlplusplus@3.4.0")

cpmaddpackage(URI "gh:CLIUtils/CLI11@2.5.0")

file(GLOB_RECURSE FLASHINFER_GENERATED_SOURCES
     "${flashinfer_SOURCE_DIR}/src/generated/*.cu")

# ------- self-contained protobuf generation -------
# i prefer this way than find_package(Protobuf REQUIRED) since we have a
# complete control over the protobuf generation & version selection

include(${protobuf_SOURCE_DIR}/cmake/protobuf-generate.cmake)

set(PIE_PROTO_API_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../proto")
set(PROTO_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${PROTO_GENERATED_DIR}")
set(PIE_PROTO_FILES
    "${PIE_PROTO_API_DIR}/l4m.proto" "${PIE_PROTO_API_DIR}/l4m_vision.proto"
    "${PIE_PROTO_API_DIR}/handshake.proto" "${PIE_PROTO_API_DIR}/ping.proto")
add_library(pie_protobuf OBJECT ${PIE_PROTO_FILES})
target_include_directories(pie_protobuf
                           PUBLIC "$<BUILD_INTERFACE:${PROTO_GENERATED_DIR}>")

target_link_libraries(pie_protobuf PUBLIC protobuf::libprotobuf)

protobuf_generate(
  TARGET
  pie_protobuf
  PROTOC_OUT_DIR
  "${PROTO_GENERATED_DIR}"
  IMPORT_DIRS
  "${PIE_PROTO_API_DIR}"
  PROTOC_EXE
  $<TARGET_FILE:protoc>)

# ------- end of protobuf generation -------

# -----

# ----
# Define a static library for the common CUDA kernels
add_library(
  cuda_common_kernels STATIC
  src/common.cu
  src/tensor.cu
  src/model.cu
)

target_include_directories(
  cuda_common_kernels
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${flashinfer_SOURCE_DIR}/include
          ${flashinfer_SOURCE_DIR}/src/generated)

target_link_libraries(cuda_common_kernels INTERFACE zstd)

# Define the main executable target
add_executable(
  pie_cuda_be
  src/main.cpp
  src/l4ma.cu
)

# Link the main executable against the common kernels library
target_link_libraries(pie_cuda_be PRIVATE cuda_common_kernels)


# Set compiler flags and options for the target These flags are applied
# specifically to 'cuda_app'. This mirrors the 'cxxflags' from your ninja file.
target_compile_options(
  pie_cuda_be
  PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
          -Xcompiler=-Wall
          -Xcompiler=-Wextra
          -Wno-unused-parameter
          --extended-lambda
          --expt-relaxed-constexpr
          # -g # Enable host debug symbols -G # Enable device-side debug symbols
          # -gencode=arch=compute_89,code=sm_89
          >)

# Link libraries and include directories CPM automatically handles include
# directories for its packages when you link them. We just need to add our own
# 'include' directory and link the required libraries.
target_include_directories(
  pie_cuda_be
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${flashinfer_SOURCE_DIR}/include
          ${flashinfer_SOURCE_DIR}/src/generated ${zstd_SOURCE_DIR}/lib)

target_link_libraries(
  pie_cuda_be
  PRIVATE # 3rd-party libraries
          cppzmq
          CLI11::CLI11
          tomlplusplus::tomlplusplus
          zstd
          nlohmann_json::nlohmann_json
          ixwebsocket
          msgpack-cxx
          # protobuf bindings
          pie_protobuf
          # from flashinfer
          decode_kernels
          prefill_kernels
          # aux cuda libraries
          CUDA::cublas
          CUDA::cublasLt)

# (Optional) Add a custom 'run' target This allows you to build and run the
# application with a single command. e.g., 'ninja run' or 'make run'
add_custom_target(
  run
  COMMAND $<TARGET_FILE:pie_cuda_be>
  DEPENDS pie_cuda_be
  WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
  COMMENT "Running pie_cuda_be...")

# --- End of CMakeLists.txt ---

add_subdirectory(tests)


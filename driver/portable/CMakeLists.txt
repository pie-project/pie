cmake_minimum_required(VERSION 3.23)

project(PIE_DRIVER_GGML CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

include(cmake/CPM.cmake)

# We vendor llama.cpp for the `ggml` library only. `EXCLUDE_FROM_ALL` keeps
# the `llama` lib (and its ~70 model TUs) out of the default build — only
# targets transitively reached from `pie_driver_portable` get compiled. The
# llama.cpp source tree remains available as a porting reference for the
# per-arch graph builders (src/models/*.cpp) and the per-stream KV cache
# layout (src/llama-kv-cache.cpp).
#
# Backend selection (CPU / CUDA / Metal / Vulkan) is opt-in via GGML_*
# options. Override on the cmake command line, e.g. `-DGGML_CUDA=ON`.
cpmaddpackage(
  NAME llama
  GITHUB_REPOSITORY ggerganov/llama.cpp
  GIT_TAG b8994
  EXCLUDE_FROM_ALL YES
  OPTIONS
    "LLAMA_BUILD_TESTS OFF"
    "LLAMA_BUILD_EXAMPLES OFF"
    "LLAMA_BUILD_SERVER OFF"
    "LLAMA_BUILD_TOOLS OFF"
    "LLAMA_BUILD_COMMON OFF"
    "LLAMA_CURL OFF")

cpmaddpackage(URI "gh:marzer/tomlplusplus@3.4.0")
cpmaddpackage(URI "gh:CLIUtils/CLI11@2.5.0")
cpmaddpackage(URI "gh:nlohmann/json@3.12.0" OPTIONS "JSON_BuildTests OFF")


# pie_driver_portable — portable ggml backend, sibling to ../cuda (native CUDA +
# flashinfer) and ../../runtime (Rust). Talks to the runtime via the shmem
# fast path; no ZMQ/protobuf needed. Loads HuggingFace safetensors directly
# (no GGUF conversion). Consumed by the Python wrapper `pie_driver_portable`.
add_executable(pie_driver_portable
  src/main.cpp
  src/shmem_ipc.cpp
  src/shmem_schema.cpp
  src/safetensors.cpp
  src/hf_config.cpp
  src/gguf_archive.cpp
  src/gguf_hparams.cpp
  src/model.cpp
  src/kv_cache.cpp
  src/state_cache.cpp
  src/arch_spec.cpp
  src/plan.cpp
  src/graph_common.cpp
  src/graph_qwen3.cpp
  src/graph_gemma4.cpp
  src/graph_qwen3_5.cpp
  src/graph_phi3small.cpp
  src/graph_phi3_5moe.cpp
  src/response.cpp
  src/forward.cpp
  src/sampler.cpp
  src/aux_server.cpp
  src/adapter.cpp
)

target_compile_options(pie_driver_portable PRIVATE
  -Wall -Wextra -Wno-unused-parameter
)

target_include_directories(pie_driver_portable
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(pie_driver_portable PRIVATE
  CLI11::CLI11
  tomlplusplus::tomlplusplus
  nlohmann_json::nlohmann_json
  ggml
)
if(NOT APPLE)
  target_link_libraries(pie_driver_portable PRIVATE rt)
endif()

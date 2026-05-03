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
  # patches/ggml-fast-argsort.patch swaps ggml-cpu's O(n²) bubble sort
  # in `ggml_compute_forward_argsort_f32` for `std::sort`. Critical for
  # CPU sampling fast paths: a single `ggml_top_k(probs, K)` over the
  # full vocab (e.g. 152 k for Qwen3) otherwise takes ~22 s per token.
  #
  # patches/ggml-pie-paged-attn-ext.patch adds GGML_OP_PAGED_ATTN_EXT and
  # the `ggml_paged_attn_ext` constructor — the op proper. Backend kernels
  # land in follow-up patches; with no backend advertising support, the
  # Pie graph builders take the materialize+flash_attn_ext fallback path.
  #
  # `-N` makes the patch idempotent so re-runs don't fail; `-t` suppresses
  # interactive prompts on hunk mismatch.
  PATCH_COMMAND patch -p1 -N -t -i ${CMAKE_CURRENT_SOURCE_DIR}/patches/ggml-fast-argsort.patch          || true
        COMMAND patch -p1 -N -t -i ${CMAKE_CURRENT_SOURCE_DIR}/patches/ggml-pie-paged-attn-ext.patch    || true
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
# (no GGUF conversion). Built two ways from the same translation units:
#   - `pie_driver_portable_lib` (STATIC): linked in-process by
#     `server/standalone` so the all-Rust binary contains the driver.
#   - `pie_driver_portable` (executable): a thin shim around the lib's
#     `pie_driver_portable_run` entry. Consumed by the Python wrapper
#     `pie_driver_portable` via subprocess.
add_library(pie_driver_portable_lib STATIC
  src/entry.cpp
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

target_compile_options(pie_driver_portable_lib PRIVATE
  -Wall -Wextra -Wno-unused-parameter
)

# We use `#pragma omp` directly in src/model.cpp (diagnostic probe of
# the OpenMP runtime from our caller thread). Need both the compile
# flag and runtime lib explicitly. ggml-cpu also links libgomp via its
# own `find_package(OpenMP)` — this just makes our usage independent.
find_package(OpenMP REQUIRED)
target_compile_options(pie_driver_portable_lib PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(pie_driver_portable_lib PUBLIC OpenMP::OpenMP_CXX)

target_include_directories(pie_driver_portable_lib
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(pie_driver_portable_lib PUBLIC
  CLI11::CLI11
  tomlplusplus::tomlplusplus
  nlohmann_json::nlohmann_json
  ggml
)
if(NOT APPLE)
  target_link_libraries(pie_driver_portable_lib PUBLIC rt)
endif()

add_executable(pie_driver_portable src/main.cpp)
target_link_libraries(pie_driver_portable PRIVATE pie_driver_portable_lib)

# ── CUDA paged-attention (FlashInfer) ─────────────────────────────────────
#
# When the ggml CUDA backend is enabled, register an out-of-tree paged-
# attention kernel into ggml-cuda's GGML_OP_PAGED_ATTN_EXT dispatch hook
# (added by patches/ggml-pie-paged-attn-ext.patch). The implementation
# wraps FlashInfer's BatchDecodeWithPagedKVCache, which is the
# production-quality kernel for vLLM/SGLang-style paged decode.
#
# Self-contained: no impact on Metal / Vulkan / CPU-only builds.
# FlashInfer is fetched only when GGML_CUDA=ON.
if(GGML_CUDA)
  enable_language(CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Match ggml-cuda's default. Caller can override with -DCMAKE_CUDA_ARCHITECTURES=...
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()

  # FlashInfer dropped its top-level CMakeLists after v0.2.7; pin to that
  # last self-contained release. cuTLASS ships under 3rdparty/.
  cpmaddpackage(
    NAME flashinfer
    GITHUB_REPOSITORY flashinfer-ai/flashinfer
    GIT_TAG v0.2.7
    DOWNLOAD_ONLY YES)

  target_sources(pie_driver_portable_lib PRIVATE
    src/paged_attn_cuda.cu)
  set_source_files_properties(src/paged_attn_cuda.cu
    PROPERTIES LANGUAGE CUDA)
  target_compile_definitions(pie_driver_portable_lib PRIVATE
    PIE_PORTABLE_HAS_CUDA=1)
  target_include_directories(pie_driver_portable_lib PRIVATE
    ${flashinfer_SOURCE_DIR}/include
    ${flashinfer_SOURCE_DIR}/3rdparty/cutlass/include
    ${flashinfer_SOURCE_DIR}/3rdparty/cutlass/tools/util/include)
endif()

cmake_minimum_required(VERSION 3.23)

project(PIE_DRIVER_GGML CXX)

if(MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if(NOT PIE_BRIDGE_INCLUDE_DIR)
  set(PIE_BRIDGE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../bridge/include")
endif()

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
  # patches/ggml-pie-paged-attn-ext.patch adds GGML_OP_PAGED_ATTN_EXT and
  # the `ggml_paged_attn_ext` constructor — the op proper. Backend kernels
  # land in follow-up patches; with no backend advertising support, the
  # Pie graph builders take the materialize+flash_attn_ext fallback path.
  #
  # PATCH_COMMAND idempotency:
  #   * On CI runners with a persistent build cache (our self-hosted
  #     runner keeps `_deps/llama-src/` between jobs), the source tree
  #     can already be patched when ExternalProject re-runs the populate
  #     step — `patch -N` then exits 1 ("Reversed or previously applied")
  #     and breaks the build.
  #   * `--dry-run` lets us probe: if forward-apply succeeds the patch
  #     hasn't been applied yet, so do it; otherwise the tree is already
  #     in the patched state and we skip silently. Either way, the build
  #     proceeds with a correctly-patched ggml.
  #   * We deliberately do NOT swallow real patch failures: if a patch
  #     stops applying cleanly (upstream rev bump, hunk drift), this
  #     fails loudly at configure time rather than ~60 min later as a
  #     missing-symbol error in model.cpp.
  PATCH_COMMAND ${CMAKE_COMMAND} -DPATCH_FILE=${CMAKE_CURRENT_SOURCE_DIR}/patches/ggml-pie-paged-attn-ext.patch
                                 -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/IdempotentPatch.cmake
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


# pie_driver_portable — portable ggml backend, sibling to ../cuda (native
# CUDA + flashinfer) and ../../runtime (Rust). Embedded-only: linked
# in-process by `server/standalone` and invoked via the FFI vtable the
# Rust runtime's `InProcChannel` hands in. Loads HuggingFace safetensors
# directly (no GGUF conversion).
add_library(pie_driver_portable_lib STATIC
  src/entry.cpp
  ../common/src/safetensors_manifest.cpp
  src/safetensors.cpp
  src/hf_config.cpp
  src/gguf_archive.cpp
  src/gguf_hparams.cpp
  src/gguf_tokenizer.cpp
  src/model.cpp
  src/rust_loader.cpp
  src/kv_cache.cpp
  src/kv_cache_quant.cpp
  src/state_cache.cpp
  src/arch_spec.cpp
  src/plan.cpp
  src/graph_common.cpp
  src/graph_qwen3.cpp
  src/graph_vision_qwen3vl.cpp
  src/graph_vision_gemma4.cpp
  src/graph_audio_gemma4.cpp
  src/graph_csm_gen.cpp
  src/graph_csm_mimi.cpp
  src/graph_gemma4.cpp
  src/graph_qwen3_5.cpp
  src/graph_phi3small.cpp
  src/graph_phi3_5moe.cpp
  src/executor/executor.cpp
  src/service/inproc_service.cpp
  src/sampler.cpp
  src/host_swap_pool.cpp
  src/adapter.cpp
)

target_compile_options(pie_driver_portable_lib PRIVATE
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/W4>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Wall -Wextra -Wno-unused-parameter>
)

target_include_directories(pie_driver_portable_lib
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../weight_loader/include
    ${PIE_BRIDGE_INCLUDE_DIR})

target_link_libraries(pie_driver_portable_lib PUBLIC
  CLI11::CLI11
  tomlplusplus::tomlplusplus
  nlohmann_json::nlohmann_json
  ggml
)
if(UNIX AND NOT APPLE)
  target_link_libraries(pie_driver_portable_lib PUBLIC rt)
endif()

enable_testing()
add_executable(test_kv_cache_quant
  tests/test_kv_cache_quant.cpp
  src/kv_cache_quant.cpp
)
target_include_directories(test_kv_cache_quant PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(test_kv_cache_quant PRIVATE ggml)
add_test(NAME portable_kv_cache_quant COMMAND test_kv_cache_quant)

# ── CUDA paged-attention (FlashInfer) ─────────────────────────────────────
#
# When the ggml CUDA backend is enabled, register an out-of-tree paged-
# attention kernel into ggml-cuda's GGML_OP_PAGED_ATTN_EXT dispatch hook
# (added by patches/ggml-pie-paged-attn-ext.patch). The implementation
# wraps FlashInfer's BatchDecodeWithPagedKVCache, which is the
# production-quality kernel for vLLM/SGLang-style paged decode.
#
# Self-contained: no impact on Metal / Vulkan / CPU-only builds. FlashInfer
# 0.2.7 is not MSVC-clean, so Windows CUDA uses ggml-cuda's materialized
# attention path until FlashInfer grows Windows support here.
if(GGML_CUDA AND NOT MSVC)
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
    GIT_SUBMODULES 3rdparty/cutlass
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

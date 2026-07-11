set(PIE_METAL_MLX_TEST_SOURCES
  kernels/sampling.cpp
  loader/hf_config.cpp
  loader/model_loader.cpp
  loader/safetensors_source.cpp
  model/arch.cpp
  model/arch_spec.cpp
  model/gemma.cpp
  model/gemma4.cpp
  model/llama_like.cpp
  model/model_graph.cpp
  model/qwen3_5.cpp
  ops/activation.cpp
  ops/attention.cpp
  ops/compiled.cpp
  ops/elementwise.cpp
  ops/embedding.cpp
  ops/gated_delta.cpp
  ops/gemm.cpp
  ops/moe.cpp
  ops/norm.cpp
  ops/rope.cpp)

add_library(pie_metal_mlx_test_support STATIC ${PIE_METAL_MLX_TEST_SOURCES})
target_include_directories(pie_metal_mlx_test_support
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_compile_options(pie_metal_mlx_test_support PRIVATE
  -Wall
  -Wextra
  -Wno-unused-parameter)
target_link_libraries(pie_metal_mlx_test_support
  PUBLIC
    nlohmann_json::nlohmann_json)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/FetchMLX.cmake)
pie_metal_fetch_mlx(pie_metal_mlx_test_support)

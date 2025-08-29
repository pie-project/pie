cmake_minimum_required(VERSION 3.10)
project(CudaKernelUnitTests)

# Find CUDA and add the required libraries
find_package(CUDA REQUIRED)

# Ensure tests can include core backend headers
include_directories(${CMAKE_SOURCE_DIR}/src)

# Add the test executables
add_executable(test_sum_reduction test_sum_reduction.cu)
add_executable(test_embedding_lookup test_embedding_lookup.cu)
add_executable(test_extract_k_values test_extract_k_values.cu)
add_executable(test_cast_type test_cast_type.cu)
add_executable(test_gemm test_gemm.cu)
add_executable(test_tensor_view test_tensor_view.cu)
add_executable(test_embedding_lookup_bf16 test_embedding_lookup_bf16.cu)
add_executable(test_extract_k_values_bf16 test_extract_k_values_bf16.cu)
add_executable(test_cast_type_half test_cast_type_half.cu)
# Placeholders that exercise presence in cuda_common_kernels only
# Removed lightweight wrapper-based tests; rely on full tests including l4ma.cu
# Full tests that include l4ma.cu directly
add_executable(test_add_residual_full test_add_residual_full.cu)
add_executable(test_act_and_mul_full test_act_and_mul_full.cu)
add_executable(test_forward_text_full test_forward_text_full.cu metadata_loader.cpp)

# Link against the CUDA runtime library and the common kernels library
set(TEST_LIBS CUDA::cudart cuda_common_kernels CUDA::cublas CUDA::cublasLt)

foreach(tgt IN ITEMS test_sum_reduction test_embedding_lookup test_extract_k_values test_cast_type test_gemm test_tensor_view test_embedding_lookup_bf16 test_extract_k_values_bf16 test_cast_type_half)
  target_link_libraries(${tgt} PRIVATE ${TEST_LIBS})
endforeach()

# FlashInfer-dependent tests (both *full and the wrapper-based ones) need headers, kernels, and NVCC flags
if(DEFINED flashinfer_SOURCE_DIR)
  target_include_directories(test_add_residual_full PRIVATE ${flashinfer_SOURCE_DIR}/include ${flashinfer_SOURCE_DIR}/src/generated)
  target_include_directories(test_act_and_mul_full PRIVATE ${flashinfer_SOURCE_DIR}/include ${flashinfer_SOURCE_DIR}/src/generated)
  target_include_directories(test_forward_text_full PRIVATE ${flashinfer_SOURCE_DIR}/include ${flashinfer_SOURCE_DIR}/src/generated)
endif()

# Expose toml++ include dir (fetched via CPM) to test needing metadata parsing

# Link flashinfer kernels for l4ma.cu symbols referenced via wrappers or direct inclusion
# (Removed) wrapper-based tests linking
target_link_libraries(test_add_residual_full PRIVATE ${TEST_LIBS} decode_kernels prefill_kernels)
target_link_libraries(test_act_and_mul_full PRIVATE ${TEST_LIBS} decode_kernels prefill_kernels)
target_link_libraries(test_forward_text_full PRIVATE ${TEST_LIBS} decode_kernels prefill_kernels tomlplusplus::tomlplusplus)

# NVCC options required by flashinfer (match main target)
foreach(tgt IN ITEMS test_add_residual_full test_act_and_mul_full test_forward_text_full)
  target_compile_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>)
endforeach()

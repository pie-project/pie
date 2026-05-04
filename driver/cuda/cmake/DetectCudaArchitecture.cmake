# DetectCudaArchitecture.cmake — auto-detect CUDA architecture from nvidia-smi.

function(detect_cuda_architectures)
  if(DEFINED CMAKE_CUDA_ARCHITECTURES AND CMAKE_CUDA_ARCHITECTURES)
    message(STATUS "Using pre-defined CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    return()
  endif()

  execute_process(
    COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits
    OUTPUT_VARIABLE GPU_COMPUTE_CAPS
    ERROR_VARIABLE NVIDIA_SMI_ERROR
    RESULT_VARIABLE NVIDIA_SMI_RESULT
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(NVIDIA_SMI_RESULT EQUAL 0 AND GPU_COMPUTE_CAPS)
    string(REPLACE "\n" ";" GPU_COMPUTE_LIST ${GPU_COMPUTE_CAPS})
    list(GET GPU_COMPUTE_LIST 0 FIRST_GPU_COMPUTE)
    string(REPLACE "." "" CUDA_ARCH ${FIRST_GPU_COMPUTE})
    message(STATUS "Detected CUDA compute capability: ${FIRST_GPU_COMPUTE} (architecture: ${CUDA_ARCH})")
    set(CMAKE_CUDA_ARCHITECTURES "${CUDA_ARCH}" PARENT_SCOPE)
    return()
  endif()

  message(FATAL_ERROR
    "CUDA Architecture Detection Failed: nvidia-smi unavailable or failed.\n"
    "Set CMAKE_CUDA_ARCHITECTURES manually (e.g., -DCMAKE_CUDA_ARCHITECTURES=\"86;89;90\").")
endfunction()

# DetectCudaArchitecture.cmake — auto-detect CUDA architecture from nvidia-smi.

function(detect_cuda_architectures)
  # Honor the CMAKE_CUDA_ARCHITECTURES env var when the cache var is unset —
  # CI sets it for GPU-less runners, and CMake doesn't auto-initialize the
  # cache var from the same-named env var. (Previously forwarded by build.rs.)
  if((NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR NOT CMAKE_CUDA_ARCHITECTURES)
     AND DEFINED ENV{CMAKE_CUDA_ARCHITECTURES}
     AND NOT "$ENV{CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES "$ENV{CMAKE_CUDA_ARCHITECTURES}")
  endif()

  if(DEFINED CMAKE_CUDA_ARCHITECTURES AND CMAKE_CUDA_ARCHITECTURES)
    string(REPLACE ";" ";" CUDA_ARCH_LIST "${CMAKE_CUDA_ARCHITECTURES}")
    set(CUDA_ARCH_LIST_NORMALIZED "")
    foreach(CUDA_ARCH IN LISTS CUDA_ARCH_LIST)
      # Hopper FA3 kernels require the architecture-accelerated target so GMMA
      # instructions are available. A bare "90" is valid CUDA, but it cannot
      # compile those kernels.
      if(CUDA_ARCH STREQUAL "90")
        list(APPEND CUDA_ARCH_LIST_NORMALIZED "90a")
      else()
        list(APPEND CUDA_ARCH_LIST_NORMALIZED "${CUDA_ARCH}")
      endif()
    endforeach()
    set(CMAKE_CUDA_ARCHITECTURES "${CUDA_ARCH_LIST_NORMALIZED}")
    set(CMAKE_CUDA_ARCHITECTURES "${CUDA_ARCH_LIST_NORMALIZED}" PARENT_SCOPE)
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
    if(CUDA_ARCH STREQUAL "90")
      set(CUDA_ARCH "90a")
    endif()
    message(STATUS "Detected CUDA compute capability: ${FIRST_GPU_COMPUTE} (architecture: ${CUDA_ARCH})")
    set(CMAKE_CUDA_ARCHITECTURES "${CUDA_ARCH}" PARENT_SCOPE)
    return()
  endif()

  message(FATAL_ERROR
    "CUDA Architecture Detection Failed: nvidia-smi unavailable or failed.\n"
    "Set CMAKE_CUDA_ARCHITECTURES manually (e.g., -DCMAKE_CUDA_ARCHITECTURES=\"86;89;90\").")
endfunction()

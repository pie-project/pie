# DetectCudaArchitecture.cmake — auto-detect CUDA architecture from nvidia-smi.

# Architectures whose tensor-core kernels are only available on the
# "architecture-accelerated" target (the `a` suffix). A bare "90"/"100" is
# valid CUDA and compiles, but CUTLASS/CuTe gate TMA, STSM and the tensor-core
# MMA instructions on CUTLASS_ARCH_MMA_SM<xx>A_ENABLED (see
# cutlass/include/cute/arch/config.hpp), so a non-`a` target silently drops
# every Hopper GMMA / Blackwell tcgen05 kernel and falls back to slow paths.
set(PIE_CUDA_ACCELERATED_ARCHS 90 100 101 103 110 120 121)

# <out_var> := <arch> with the `a` suffix appended when that arch needs it.
# Anything already carrying a suffix ("90a", "100-real", ...) is left alone.
function(pie_accelerated_cuda_arch out_var arch)
  if(arch MATCHES "^[0-9]+$" AND arch IN_LIST PIE_CUDA_ACCELERATED_ARCHS)
    set(${out_var} "${arch}a" PARENT_SCOPE)
  else()
    set(${out_var} "${arch}" PARENT_SCOPE)
  endif()
endfunction()

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
      pie_accelerated_cuda_arch(CUDA_ARCH "${CUDA_ARCH}")
      list(APPEND CUDA_ARCH_LIST_NORMALIZED "${CUDA_ARCH}")
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
    pie_accelerated_cuda_arch(CUDA_ARCH "${CUDA_ARCH}")
    message(STATUS "Detected CUDA compute capability: ${FIRST_GPU_COMPUTE} (architecture: ${CUDA_ARCH})")
    set(CMAKE_CUDA_ARCHITECTURES "${CUDA_ARCH}" PARENT_SCOPE)
    return()
  endif()

  message(FATAL_ERROR
    "CUDA Architecture Detection Failed: nvidia-smi unavailable or failed.\n"
    "Set CMAKE_CUDA_ARCHITECTURES manually (e.g., -DCMAKE_CUDA_ARCHITECTURES=\"86;89;90\").")
endfunction()

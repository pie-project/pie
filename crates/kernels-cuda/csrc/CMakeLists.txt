# The CUDA kernel archive: one directory per kernel family under `src/`,
# and the vendored
# third-party kernels they wrap.
#
# This used to be part of `crates/driver-cuda/csrc`, and the split follows a
# line that was already there rather than one drawn for the occasion: nothing
# under `kernels/` or `ops/` includes a single header from `model/`, `batch/`,
# `pipeline/`, `store/` or `loader/`. The dependency already ran one way. What
# it lacked was a boundary that says so.
#
# What came along, and why:
#
#   kernels.def          the manifest both C++ and CMake read; it declares
#   kernels_manifest.hpp which head_dims get a translation unit and which MoE
#                        activations are reachable. Kernel facts.
#   cuda_check.hpp       CUDA_CHECK. The vocabulary any .cu is written in.
#   tensor.hpp/.cpp      DeviceTensor + DType -- the type a kernel's arguments
#                        are expressed in. The shell reads it too, and reads
#                        it from HERE, because a kernel signature cannot be
#                        stated in a type the caller owns.
#
# What went BACK to the driver: `runahead.hpp`, and then the class it was
# there to size. `runahead.hpp` holds the scheduler's run-ahead depth, which
# is shell policy, and it was here only because `attention_workspace.hpp`
# sized a `std::array` by it. That array became a `std::vector` and
# `allocate()` took the depth as an argument, which sent the constant home.
#
# `AttentionWorkspace` itself followed. The measurement that decided it: of
# everything the class offers, `attn/` reads exactly five values and calls
# nothing else, while `allocate` / `begin_plan_update` / `end_plan_update`
# are the driver's alone. So the five stayed as a POD --
# `src/attention_workspace_view.hpp`, which is what every launcher now takes
# -- and the allocation, the move semantics, the pinned staging rotation and
# the events fencing it went to `driver-cuda/csrc/src/attention_workspace.*`.
# Same shape as `runahead`: what crosses is a value, not an object, and the
# dependency is one-way.
#   third_party/         FlashInfer's generated MoE launchers, marlin,
#                        marlin_moe. Kernels, vendored.
#
# What did NOT come: `config.hpp` -- the shell's TOML config, and the one
# header here that genuinely reached into `batch/` and `store/`. It came down
# to four lines: `ops/tuning_cache.hpp` wanted `cache_dir()`, the mutable
# global the config loader publishes and every on-disk cache reads. Those four
# lines are `src/cache_root.hpp` here, and `config.hpp` includes them from
# here. Nor did the NCCL comm in `distributed.*`, nor `custom_all_reduce`,
# which is not a compute kernel but the device half of that comm and now sits
# beside it in the shell's `src/comm/`.
#
# The FlashInfer fetch, its three source patches, and the CCCL selection live
# here because they are kernel machinery -- but `src/batch/workspace.cu` on the
# shell side reaches FlashInfer's scheduler headers too, so the resolved
# include directories are EXPORTED (see the bottom of this file) rather than
# re-derived over there. One fetch, one patch, one answer.


cmake_minimum_required(VERSION 3.23)

# The PTIR contract headers (`rng_contract.generated.h`). `kernels-cuda`'s
# build.rs passes this; the fallback keeps a bare `cmake csrc/` working.
if(NOT PIE_PTIR_INCLUDE_DIR)
  set(PIE_PTIR_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../tensor-compiler/include")
endif()

include(cmake/DetectCudaArchitecture.cmake)
detect_cuda_architectures()
message(STATUS "Using CUDA architecture: ${CMAKE_CUDA_ARCHITECTURES}")

# nvcc discovery: some toolkits install CUDA under /usr/local/cuda without
# putting bin/ on PATH, so workspace builds work without sourcing CUDA's env
# script first.
if(NOT CMAKE_CUDA_COMPILER
   AND NOT DEFINED ENV{CUDACXX}
   AND NOT DEFINED ENV{CMAKE_CUDA_COMPILER})
  find_program(PIE_NVCC nvcc
    HINTS
      /usr/local/cuda/bin
      /usr/local/cuda-13/bin
      /usr/local/cuda-13.0/bin
      /usr/local/cuda-12/bin
      /usr/local/cuda-12.8/bin
      /opt/cuda/bin)
  if(PIE_NVCC)
    set(CMAKE_CUDA_COMPILER "${PIE_NVCC}")
  endif()
endif()

project(PIE_KERNELS_CUDA CUDA CXX)


# The boundary, stated precisely: not one file under `src/` includes a header
# from `driver`, `driver-abi` or `model-loader-capi`. A kernel takes pointers
# and extents.
#
# One exception, and it is a CONTRACT rather than machinery:
# `src/ptir/tier0.cuh` includes `tensor-compiler`'s generated
# `rng_contract.generated.h`. Tier-0 interprets a PTIR trace kernel by kernel
# and tier-1 re-emits the same math through NVRTC; the two are diffed against
# a host golden, so their RNG has to agree bit for bit. That agreement is what
# the generated header IS. Reading it here is the same shape as `kernels.def`
# being read by both C++ and CMake -- one source, two consumers, no copy.

# Optional compiler cache; speeds clean rebuilds / branch switches.
# Host C/C++ accepts either sccache or ccache.
if(NOT CMAKE_C_COMPILER_LAUNCHER AND NOT CMAKE_CXX_COMPILER_LAUNCHER)
  if(DEFINED ENV{PIE_COMPILER_LAUNCHER} AND NOT "$ENV{PIE_COMPILER_LAUNCHER}" STREQUAL "")
    set(PIE_COMPILER_LAUNCHER "$ENV{PIE_COMPILER_LAUNCHER}")
  else()
    find_program(PIE_COMPILER_LAUNCHER NAMES sccache ccache)
  endif()
  if(PIE_COMPILER_LAUNCHER)
    set(CMAKE_C_COMPILER_LAUNCHER "${PIE_COMPILER_LAUNCHER}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${PIE_COMPILER_LAUNCHER}")
    message(STATUS "Using compiler launcher (C/C++): ${PIE_COMPILER_LAUNCHER}")
  endif()
endif()

# nvcc is launched through ccache ONLY. sccache mishandles nvcc's fatbinary
# temp files (intermittent SIGHUP-killed server); ccache 4.x handles nvcc
# correctly. Caching .cu matters far more than caching host C/C++: .cu
# objects are ~95% of this build's CPU time, so a host-only cache leaves
# nearly all of a rebuild uncached. Set -DCMAKE_CUDA_COMPILER_LAUNCHER=
# explicitly to opt out or to force a different launcher.
if(NOT DEFINED CMAKE_CUDA_COMPILER_LAUNCHER)
  find_program(PIE_CUDA_COMPILER_LAUNCHER NAMES ccache)
  if(PIE_CUDA_COMPILER_LAUNCHER)
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${PIE_CUDA_COMPILER_LAUNCHER}")
    message(STATUS "Using CUDA compiler launcher: ${PIE_CUDA_COMPILER_LAUNCHER}")
  else()
    message(STATUS "ccache not found; nvcc compiles will not be cached")
  endif()
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING
    "Build type (Debug, Release, RelWithDebInfo, MinSizeRel)" FORCE)
endif()

# ---- kernels.def ----------------------------------------------------------
# Single source of truth for model-derived kernel variants, read by both C++
# (as an X-macro include) and CMake (here). See src/kernels.def for the rule
# about which axes may live there. CMake needs exactly two facts from it:
# which head_dims get their own translation unit, and which MoE activations
# are reachable.
set(PIE_KERNELS_DEF "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels.def")
if(NOT EXISTS "${PIE_KERNELS_DEF}")
  message(FATAL_ERROR "missing kernel manifest: ${PIE_KERNELS_DEF}")
endif()
set_property(DIRECTORY APPEND
  PROPERTY CMAKE_CONFIGURE_DEPENDS "${PIE_KERNELS_DEF}")

# Collect the arguments of every `<macro_name>(...)` invocation. Only lines
# that *start* with the macro name are matched, so the #ifndef/#define/#undef
# scaffolding and the prose comments are skipped.
function(pie_read_kernels_def macro_name out_var)
  file(STRINGS "${PIE_KERNELS_DEF}" _lines REGEX "^[ \t]*${macro_name}\\(")
  set(_vals "")
  foreach(_line IN LISTS _lines)
    string(REGEX REPLACE "^[ \t]*${macro_name}\\(([^)]*)\\).*$" "\\1" _arg "${_line}")
    string(REGEX REPLACE "[ \t]" "" _arg "${_arg}")
    list(APPEND _vals "${_arg}")
  endforeach()
  set(${out_var} "${_vals}" PARENT_SCOPE)
endfunction()

pie_read_kernels_def(PIE_ATTN_HEAD_DIM PIE_ATTN_HEAD_DIMS)
pie_read_kernels_def(PIE_MOE_ACTIVATION PIE_MOE_ACTIVATIONS)
if(NOT PIE_ATTN_HEAD_DIMS)
  message(FATAL_ERROR
    "no PIE_ATTN_HEAD_DIM entries found in ${PIE_KERNELS_DEF}")
endif()
message(STATUS "Attention head_dims: ${PIE_ATTN_HEAD_DIMS}")
message(STATUS "MoE activations: ${PIE_MOE_ACTIVATIONS}")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
# No Python interpreter required to build: FlashInfer's CUTLASS MoE launchers
# are vendored (third_party/flashinfer_generated) and flashinfer itself is
# fetched DOWNLOAD_ONLY for headers.

# CPM.cmake and DetectCudaArchitecture.cmake are vendored in BOTH csrc trees,
# byte-identical. That is what vendored bootstrap scripts are: each CMake
# project has to be able to configure on its own, and neither can `include()`
# out of the other's source tree without reintroducing exactly the cross-crate
# relative path this split removed. Update them together.
include(cmake/CPM.cmake)

# Persistent CPM cache when set, so incremental builds don't re-clone from
# GitHub. CI sets this via GITHUB_ENV; local builds can export it in .bashrc.
if(DEFINED ENV{CPM_SOURCE_CACHE} AND NOT CPM_SOURCE_CACHE)
  set(CPM_SOURCE_CACHE $ENV{CPM_SOURCE_CACHE})
endif()

# flashinfer ships no top-level CMakeLists.txt since v0.3.0; fetch
# DOWNLOAD_ONLY for headers/csrc and compile only the launchers we need
# directly into the driver (see the generated-MoE and Hopper/FA3 blocks below).
CPMAddPackage(
  NAME flashinfer
  GITHUB_REPOSITORY flashinfer-ai/flashinfer
  GIT_TAG v0.6.15
  GIT_SHALLOW TRUE
  DOWNLOAD_ONLY YES)

# FlashInfer v0.6.15 uses APIs introduced by its vendored CCCL 3.3.2.
# Keep these ahead of the toolkit-bundled CCCL, which may be older even on
# CUDA 13 installations.
set(_FI_CCCL_ROOT "${flashinfer_SOURCE_DIR}/3rdparty/cccl")
set(PIE_CCCL_INCLUDE_DIRS
  "${_FI_CCCL_ROOT}/cub"
  "${_FI_CCCL_ROOT}/libcudacxx/include"
  "${_FI_CCCL_ROOT}/thrust")
foreach(_fi_cccl_include IN LISTS PIE_CCCL_INCLUDE_DIRS)
  if(NOT EXISTS "${_fi_cccl_include}")
    message(FATAL_ERROR
      "FlashInfer vendored CCCL headers are missing: ${_fi_cccl_include}")
  endif()
endforeach()
message(STATUS "Using FlashInfer vendored CCCL: ${_FI_CCCL_ROOT}")

# Upstream bug: `state_stride_batch` is an int64_t stride everywhere else in
# FlashInfer, but buildNdDescriptor() takes `std::vector<uint64_t> const&`, so
# the two brace-initialized call sites narrow. Narrowing an initializer_list
# element is ill-formed, not a warning, so nvcc hard-errors:
#
#   error: invalid narrowing conversion from "signed long" to "unsigned long"
#
# and no -Wno-* or --diag-suppress can demote it. (Were the parameter an
# ordinary aggregate this would be warning #2361-D and nothing to fix.)
#
# Only sm_90+ builds see it: both sites live inside the header's
# FLASHINFER_MAMBA_ENABLE_SM90 block, which is exactly the macro the
# Hopper/Blackwell Mamba gate below defines. Upstream's own builds do not
# compile that path, which is why the bug has survived unnoticed -- as of
# v0.6.15 both sites are still unfixed on main and on today's nightly, and
# v0.6.15.post1 is the newest release, so there is no version to bump to.
# Reported upstream as flashinfer-ai/flashinfer#4155; drop this whole block
# once a release carrying that fix lands and GIT_TAG moves past it.
#
# Upstream has two more identical sites in invoke_selective_state_update_mtp.cuh
# guarded by FLASHINFER_MAMBA_ENABLE_SM100. We do not patch those because we
# never define that macro; enabling sm_100 Mamba means extending this patch.
#
# Patching the fetched source in place is deliberate but not free: it mutates
# the CPM source cache that every build directory on the machine shares, and
# CPM notices ("Cache for flashinfer ... is dirty"). It is kept because the
# alternative is worse -- vendoring cannot shadow this header alone, since its
# only includer reaches it through a relative `#include "..."` that resolves
# next to the includer before any -I path, so a vendored copy would have to
# carry both files (~1460 lines) to fix one cast.
#
# The sentinel avoids re-patching every fresh build-hash dir; the FATAL_ERROR
# fires if a FlashInfer bump ever makes the patch stop applying.
set(_FI_PATCH_SENTINEL "${flashinfer_SOURCE_DIR}/.pie_narrowing_patched")
set(_FI_PATCH_FILE
  "${flashinfer_SOURCE_DIR}/include/flashinfer/mamba/kernel_selective_state_update_stp.cuh")
if(EXISTS "${_FI_PATCH_FILE}" AND NOT EXISTS "${_FI_PATCH_SENTINEL}")
  file(READ "${_FI_PATCH_FILE}" _fi_src)
  string(FIND "${_fi_src}"
    "DSTATE * DIM, params.state_stride_batch}" _fi_patch_offset)
  if(_fi_patch_offset EQUAL -1)
    message(FATAL_ERROR
      "FlashInfer narrowing patch no longer applies to ${_FI_PATCH_FILE}")
  endif()
  string(REPLACE
    "DSTATE * DIM, params.state_stride_batch}"
    "DSTATE * DIM, static_cast<uint64_t>(params.state_stride_batch)}"
    _fi_src "${_fi_src}")
  file(WRITE "${_FI_PATCH_FILE}" "${_fi_src}")
  file(WRITE "${_FI_PATCH_SENTINEL}" "patched")
  message(STATUS "Patched FlashInfer narrowing conversion in ${_FI_PATCH_FILE}")
endif()

# FlashInfer gates its Mamba TMA kernels on FLASHINFER_MAMBA_ENABLE_SM90 and
# nothing else, which is sound for the way upstream builds: a translation unit
# compiled for sm_90 and no other architecture. We compile ONE translation unit
# for every entry in CMAKE_CUDA_ARCHITECTURES, and a compile definition reaches
# every -gencode pass, so defining it hands the TMA kernels to the sm_80/86/89
# passes as well -- and libcu++ does not expose the APIs they call there.
# `nv/detail/__target_macros` defines __CUDA_MINIMUM_ARCH__ as __CUDA_ARCH__
# under nvcc, i.e. PER PASS rather than the minimum of the arch list, and
# `cuda/std/__cccl/ptx_isa.h` withholds __cccl_lib_experimental_ctk12_cp_async_exposure
# below 900. The sub-900 passes then fail on cp_async_bulk_tensor_4d_*,
# cp_async_bulk_commit_group, cp_async_bulk_wait_group_read and
# fence_proxy_async_shared_cta.
#
# So make the guard per-pass as well as per-build. Every site takes the SAME
# condition, and that is what makes it sound rather than merely quieter: the
# horizontal kernel's definition sits at namespace scope and the only code that
# launches it sits in the host launcher, so under one shared condition they
# appear and vanish together in every pass. The vertical kernel keeps its
# definition with an empty body on the passes that cannot launch it. The host
# pass has no __CUDA_ARCH__ at all and therefore keeps everything, which is what
# the launcher needs in order to compile.
#
# This takes nothing away from sm_80/86/89. `src/ssm/flashinfer_mamba.cu` only
# enters FlashInfer's SSU at all when the device's major version is >= 9, and
# FlashInfer's own selector independently sends sm_major < 9 to kSimple, so
# these kernels were unreachable on those architectures before this patch.
#
# The site COUNT is asserted, not just the presence of the text: a FlashInfer
# bump that adds or drops a guard would otherwise patch a different set and
# still configure cleanly, which is the silent half-patch this must not become.
# A missing file is fatal here for the same reason, rather than the skip the
# patches above take -- upstream moving this header would otherwise read as
# "nothing to patch" and hand the TMA kernels straight back to sm_80.
set(_FI_ARCH_SENTINEL "${flashinfer_SOURCE_DIR}/.pie_mamba_arch_guarded")
# Deliberately not _FI_PATCH_FILE: that one belongs to the narrowing patch
# above, which its own comment says to delete once upstream fixes it, and this
# patch must not quietly stop applying when that happens.
set(_FI_ARCH_FILE
  "${flashinfer_SOURCE_DIR}/include/flashinfer/mamba/kernel_selective_state_update_stp.cuh")
set(_FI_ARCH_GUARD_OLD "#ifdef FLASHINFER_MAMBA_ENABLE_SM90")
set(_FI_ARCH_GUARD_NEW
  "#if defined(FLASHINFER_MAMBA_ENABLE_SM90) && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 900)")
if(NOT EXISTS "${_FI_ARCH_SENTINEL}")
  if(NOT EXISTS "${_FI_ARCH_FILE}")
    message(FATAL_ERROR
      "FlashInfer Mamba arch-guard patch cannot find ${_FI_ARCH_FILE}")
  endif()
  file(READ "${_FI_ARCH_FILE}" _fi_arch_src)
  string(REGEX MATCHALL "${_FI_ARCH_GUARD_OLD}" _fi_arch_sites "${_fi_arch_src}")
  list(LENGTH _fi_arch_sites _fi_arch_count)
  if(NOT _fi_arch_count EQUAL 6)
    message(FATAL_ERROR
      "FlashInfer Mamba arch-guard patch expects 6 `${_FI_ARCH_GUARD_OLD}` sites in "
      "${_FI_ARCH_FILE}, found ${_fi_arch_count}")
  endif()
  string(REPLACE "${_FI_ARCH_GUARD_OLD}" "${_FI_ARCH_GUARD_NEW}"
    _fi_arch_src "${_fi_arch_src}")
  file(WRITE "${_FI_ARCH_FILE}" "${_fi_arch_src}")
  file(WRITE "${_FI_ARCH_SENTINEL}" "patched")
  message(STATUS
    "Guarded FlashInfer Mamba SM90 kernels per arch pass in ${_FI_ARCH_FILE}")
endif()

# `finalizeMoeRoutingKernel` unpermutes GEMM2's output and does the topk-weighted
# reduction. Its launcher asks for exactly `num_rows` blocks and the kernel reads
# the token count back out of `gridDim.x`, so the grid is pinned to the batch
# size: at decode that is 8 blocks on a 148-SM B200, and the kernel runs at about
# 57 GB/s (15.5 us per MoE layer, 62 us/step, 3% of GLM at c=8). Upstream already
# knows better -- the sibling `finalizeMoeRoutingNoFillingKernel` launches
# `smCount * 8` blocks with the comment "can fully leverage the memory bandwidth
# (tested on B200)" -- but pie cannot take that branch, which needs `ep_size > 1`
# and all-to-all.
#
# Give the grid a second dimension over column chunks instead. `gridDim.x` still
# equals `num_rows`, so the kernel's own derivation of it is untouched, and every
# output element is still produced by exactly one thread reducing the same topk
# terms in the same order -- the change is bit-exact. Blocks drop to 64 threads
# so a decode batch spreads over ~96 blocks instead of 8.
set(_FI_FINALIZE_SENTINEL "${flashinfer_SOURCE_DIR}/.pie_finalize_grid_patched")
set(_FI_FINALIZE_FILE
  "${flashinfer_SOURCE_DIR}/csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh")
if(EXISTS "${_FI_FINALIZE_FILE}" AND NOT EXISTS "${_FI_FINALIZE_SENTINEL}")
  file(READ "${_FI_FINALIZE_FILE}" _fi_fin_src)
  set(_FI_FIN_KERNEL_OLD [[  int64_t const start_offset = threadIdx.x;
  int64_t const stride = FINALIZE_THREADS_PER_BLOCK;]])
  set(_FI_FIN_KERNEL_NEW [[  int64_t const start_offset = threadIdx.x + blockIdx.y * blockDim.x;
  int64_t const stride = static_cast<int64_t>(blockDim.x) * gridDim.y;]])
  set(_FI_FIN_LAUNCH_OLD [[    int64_t const blocks = num_rows;
    int64_t const threads = FINALIZE_THREADS_PER_BLOCK;
    config.gridDim = blocks;
    config.blockDim = threads;]])
  set(_FI_FIN_LAUNCH_NEW [[    int64_t const threads = 64;
    constexpr int64_t pie_elems_per_thread =
        128 / std::min(sizeof_bits<OutputType>::value, sizeof_bits<GemmOutputType>::value);
    int64_t const pie_vec_cols =
        (unpadded_cols + pie_elems_per_thread - 1) / pie_elems_per_thread;
    int64_t const pie_col_blocks =
        pie_vec_cols > threads ? (pie_vec_cols + threads - 1) / threads : 1;
    config.gridDim = dim3(static_cast<unsigned>(num_rows),
                          static_cast<unsigned>(pie_col_blocks));
    config.blockDim = dim3(static_cast<unsigned>(threads));]])
  string(FIND "${_fi_fin_src}" "${_FI_FIN_KERNEL_OLD}" _fi_fin_k_off)
  string(FIND "${_fi_fin_src}" "${_FI_FIN_LAUNCH_OLD}" _fi_fin_l_off)
  if(_fi_fin_k_off EQUAL -1 OR _fi_fin_l_off EQUAL -1)
    message(FATAL_ERROR
      "FlashInfer finalize-grid patch no longer applies to ${_FI_FINALIZE_FILE}")
  endif()
  string(REPLACE "${_FI_FIN_KERNEL_OLD}" "${_FI_FIN_KERNEL_NEW}"
    _fi_fin_src "${_fi_fin_src}")
  string(REPLACE "${_FI_FIN_LAUNCH_OLD}" "${_FI_FIN_LAUNCH_NEW}"
    _fi_fin_src "${_fi_fin_src}")
  file(WRITE "${_FI_FINALIZE_FILE}" "${_fi_fin_src}")
  file(WRITE "${_FI_FINALIZE_SENTINEL}" "patched")
  message(STATUS "Patched FlashInfer finalizeMoeRouting grid in ${_FI_FINALIZE_FILE}")
endif()

# FlashInfer's `fastdiv.cuh` is a thin wrapper over `cuda::fast_mod_div`, which
# CCCL only grew in 3.1. A CUDA toolkit that ships an older CCCL cannot compile
# any FlashInfer attention kernel at all -- CUDA 13.0 ships CCCL 3.0, so the
# driver fails to build against the current default toolkit.
#
# FlashInfer vendors the CCCL it was developed against, so prefer that over the
# toolkit's when the toolkit's is too old. All three components are taken
# together rather than just libcudacxx, so CUB and Thrust cannot end up a
# different version from the `cuda::` headers they are built on.
set(_PIE_FI_CCCL "${flashinfer_SOURCE_DIR}/3rdparty/cccl")
set(_PIE_CCCL_HAS_FAST_MOD_DIV FALSE)
foreach(_cccl_inc IN LISTS PIE_CCCL_INCLUDE_DIRS CUDAToolkit_INCLUDE_DIRS)
  if(EXISTS "${_cccl_inc}/cuda/__cmath/fast_modulo_division.h")
    set(_PIE_CCCL_HAS_FAST_MOD_DIV TRUE)
  endif()
endforeach()
if(NOT _PIE_CCCL_HAS_FAST_MOD_DIV)
  if(NOT EXISTS "${_PIE_FI_CCCL}/libcudacxx/include/cuda/__cmath/fast_modulo_division.h")
    message(FATAL_ERROR
      "The CUDA toolkit's CCCL predates cuda::fast_mod_div and FlashInfer's "
      "vendored CCCL is missing from ${_PIE_FI_CCCL}; FlashInfer's fastdiv.cuh "
      "cannot compile. Install a toolkit with CCCL 3.1 or newer.")
  endif()
  list(PREPEND PIE_CCCL_INCLUDE_DIRS
    "${_PIE_FI_CCCL}/libcudacxx/include"
    "${_PIE_FI_CCCL}/cub"
    "${_PIE_FI_CCCL}/thrust")
  message(STATUS
    "CUDA toolkit CCCL lacks cuda::fast_mod_div; using FlashInfer's vendored "
    "CCCL at ${_PIE_FI_CCCL}")
endif()


# ---- Helper functions ------------------------------------------------------

# True in <out_var> if CMAKE_CUDA_ARCHITECTURES contains an entry whose
# numeric SM version equals <arch_number> (suffixes like "-real" ignored).
function(pie_cuda_has_arch out_var arch_number)
  set(_found FALSE)
  foreach(_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
    string(REGEX REPLACE "[^0-9].*$" "" _arch_num "${_arch}")
    if(_arch_num STREQUAL "${arch_number}")
      set(_found TRUE)
    endif()
  endforeach()
  set(${out_var} "${_found}" PARENT_SCOPE)
endfunction()
# ---- Architecture-gated source selection -----------------------------------

pie_cuda_has_arch(PIE_HAS_SM90 90)
pie_cuda_has_arch(PIE_HAS_SM100 100)
pie_cuda_has_arch(PIE_HAS_SM120 120)

# Hopper/FA3 attention + the SM90 XQA GQA8 kernel are SM90-only; every other
# arch gets the stub.
if(PIE_HAS_SM90)
  set(PIE_CUDA_FLASHINFER_HOPPER_SOURCE
    src/attn/attention_flashinfer_hopper.cu
    src/attn/attention_xqa_gqa8_sm90.cu)
  message(STATUS "Enabling FlashInfer Hopper/FA3 attention for CUDA architecture ${CMAKE_CUDA_ARCHITECTURES}")
else()
  set(PIE_CUDA_FLASHINFER_HOPPER_SOURCE src/attn/attention_flashinfer_hopper_stub.cpp)
  message(STATUS "Disabling FlashInfer Hopper/FA3 attention for CUDA architecture ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# FlashInfer's Mamba SSU vertical/horizontal kernels need TMA + memcpy_async_tx
# (sm_90+); the runtime algorithm selector also picks the horizontal kernel
# for sm >= 100 with bf16/fp16 state, so gate the same compile define there.
# Without it, sm_90+ runs would silently fall back to the slow simple
# algorithm even though the runtime SM check would otherwise pick
# vertical/horizontal.
#
# "Contains sm_90" is the right question for this define and the wrong one for
# the compiler: the define reaches every -gencode pass, including the ones that
# cannot compile TMA. Which passes keep the kernels is settled per pass by the
# arch guard patched into the FlashInfer header above, so a mixed arch list can
# enable this without the sm_80/86/89 passes failing on it.
if(PIE_HAS_SM90 OR PIE_HAS_SM100)
  set(PIE_CUDA_FLASHINFER_MAMBA_SM90 ON)
  message(STATUS "Enabling FlashInfer Mamba SM90 vertical/horizontal SSU kernels")
else()
  set(PIE_CUDA_FLASHINFER_MAMBA_SM90 OFF)
  message(STATUS "FlashInfer Mamba SM90 vertical/horizontal SSU kernels disabled (legacy + simple algorithm only)")
endif()

# The graph-heavy kernels: the ones a model implementation on the shell
# side launches by name. They were listed with those implementations when
# both lived in one archive; the grouping was about build cost, not about
# ownership, so it did not survive the split.
set(PIE_CUDA_GRAPH_KERNEL_SOURCES
  src/norm/altup.cu
  src/norm/altup_aux.cu
  src/attn/attn_sink.cu
  src/attn/dsa_indexer.cu
  src/attn/dsv4_compress.cu
  src/norm/dsv4_hc.cu
  src/moe/dsv4_routing.cu
  src/mlp/gaussian_topk.cu
  src/attn/kimi_mla.cu
  src/attn/mla_paged.cu
  src/ssm/nemotron_h.cu
  src/attn/page_compact.cu
  src/norm/scalar_mul.cu
  src/attn/softcap.cu
  src/attn/attention_mla.cu
  src/ssm/flashinfer_mamba.cu)

# ---- pie_kernels_cuda ------------------------------------------------------
# The kernel archive. `driver-cuda` links it; nothing else does. Every
# translation unit under src/ is listed exactly once.

# One attention translation unit per head_dim in kernels.def. These cannot be
# generated -- a macro cannot expand to a source file -- so they are checked in
# next to the primary TU and verified here instead.
set(PIE_ATTENTION_HEAD_DIM_SOURCES)
foreach(_pie_hd IN LISTS PIE_ATTN_HEAD_DIMS)
  set(_pie_hd_src "src/attn/attention_flashinfer_hd${_pie_hd}.cu")
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_pie_hd_src}")
    message(FATAL_ERROR
      "kernels.def declares PIE_ATTN_HEAD_DIM(${_pie_hd}) but ${_pie_hd_src} "
      "is missing. Copy an existing attention_flashinfer_hd*.cu and change the "
      "explicit instantiation to AttnHd<${_pie_hd}>.")
  endif()
  list(APPEND PIE_ATTENTION_HEAD_DIM_SOURCES "${_pie_hd_src}")
endforeach()
# ...and the reverse: a stub with no kernels.def entry would silently compile
# nothing, since the header only declares the head_dims kernels.def lists.
# CONFIGURE_DEPENDS so that dropping in a new stub re-runs the check instead of
# passing until someone happens to reconfigure.
file(GLOB _pie_hd_stubs CONFIGURE_DEPENDS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/attn/attention_flashinfer_hd*.cu")
foreach(_pie_hd_src IN LISTS _pie_hd_stubs)
  if(NOT _pie_hd_src IN_LIST PIE_ATTENTION_HEAD_DIM_SOURCES)
    message(FATAL_ERROR
      "${_pie_hd_src} has no matching PIE_ATTN_HEAD_DIM entry in "
      "${PIE_KERNELS_DEF}; add one or delete the file.")
  endif()
endforeach()


# The custom P2P all-reduce, arch-gated. It wraps FlashInfer's vllm-style
# kernel and FUSES a residual add and an RMSNorm into the reduction -- the
# unfused halves of which are `norm/`'s -- so it is a compute kernel that
# happens to be collective, not comm machinery. What it needs from the
# collective is one bootstrap all-gather, taken as a callback
# (`HostAllgather`) rather than an `NcclComm&`.
#
# Blackwell (sm100/120) has no kernel yet; the stub keeps the entry points.
if(PIE_HAS_SM100 OR PIE_HAS_SM120)
  set(PIE_KERNELS_CUSTOM_ALL_REDUCE_SOURCE src/comm/custom_all_reduce_stub.cpp)
  message(STATUS "Disabling FlashInfer custom all-reduce for CUDA architecture ${CMAKE_CUDA_ARCHITECTURES}")
else()
  set(PIE_KERNELS_CUSTOM_ALL_REDUCE_SOURCE src/comm/custom_all_reduce.cu)
endif()

add_library(pie_kernels_cuda STATIC
  src/layout/split_gate_up.cu
  src/moe/topk_sigmoid.cu
  # vision/ -- the multimodal towers, bridged at TOWER granularity (one
  # launcher per tower; the walk and its host prep stay C++ — see the
  # provenance headers and the retirement wiki's VL judgment).
  src/vision/qwen3_vl_tower.cu
  src/vision/qwen3_vl_tower_c.cpp
  src/vision/vis_helpers.cpp
  src/vision/gemma4_vision.cu
  src/vision/gemma4_audio.cu
  src/vision/gemma4_towers_c.cpp
  ${PIE_KERNELS_CUSTOM_ALL_REDUCE_SOURCE}
  src/tensor.cpp
  # ops/ -- attention / GEMM / MoE / SSM wrappers
  src/attn/attention_flashinfer.cu
  ${PIE_ATTENTION_HEAD_DIM_SOURCES}
  ${PIE_CUDA_FLASHINFER_HOPPER_SOURCE}
  # Unconditional, and not part of PIE_CUDA_FLASHINFER_HOPPER_SOURCE above:
  # the decode KV-split path reaches the state merge on every architecture,
  # so it cannot ride the sm90 prefill's stub/real switch.
  src/attn/attention_merge_states.cu
  src/attn/attention_naive.cu
  src/attn/attention_naive_paged.cu
  src/attn/attention_xqa.cu
  src/attn/attention_xqa_gqa2.cu
  src/attn/attention_xqa_gqa2_p16.cu
  src/attn/attention_xqa_gqa4.cu
  src/attn/attention_xqa_gqa8.cu
  src/gemm/gemm.cpp
  # The plan caches' extern "C" lifecycle (`pie_x_*`). Beside the plan
  # types rather than in the driver, which is `cuda.md` §5.E5's second exit
  # — the first (move it to Rust) is closed by the `unique_ptr` with a
  # custom deleter that is the whole reason these entries exist.
  src/attn/plan_lifecycle.cpp
  # kernels/ -- fused kernels
  src/norm/add_bias.cu
  src/sample/argmax.cu
  src/layout/graph_pad.cu
  src/ssm/causal_conv1d.cu
  src/layout/deinterleave.cu
  src/quant/dequant_fp4.cu
  src/quant/dequant_fp8.cu
  src/quant/dequant_wna16.cu
  src/quant/dtype_cast.cu
  src/layout/embed.cu
  src/layout/envelope.cu
  src/ssm/gated_delta_net.cu
  src/ssm/kda.cu
  src/attn/attn_res.cu
  src/layout/gather_rows.cu
  src/gemm/gemv.cu
  src/layout/gather_tokens.cu
  src/layout/geometry.cu
  src/attn/head_dim_pad.cu
  src/attn/kv_paged.cu
  src/moe/moe_dispatch.cu
  src/moe/moe_grouped_gemm.cu
  src/quant/mxfp4_marlin.cu
  src/attn/pack_dense_mask.cu
  src/quant/quant_bf16_to_fp8.cu
  src/quant/quant_bf16_to_mxfp4.cu
  src/norm/residual_add.cu
  src/norm/rmsnorm.cu
  src/rope/rope.cu
  src/attn/qkv_fused.cu
  src/layout/slot_ops.cu
  src/attn/split_packed.cu
  src/mlp/swiglu.cu
  src/moe/topk_softmax.cu
  src/quant/transcode.cu
  # ...and the graph-heavy ones a model implementation launches by name.
  ${PIE_CUDA_GRAPH_KERNEL_SOURCES}
)

target_compile_options(pie_kernels_cuda PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    -Xcompiler=-Wall
    -Xcompiler=-Wextra
    -Wno-unused-parameter
    --extended-lambda
    --expt-relaxed-constexpr
  >
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wno-unused-parameter>
)

# PUBLIC, all of it: `driver-cuda` compiles against these headers too. `src`
# carries the kernel declarations and the three substrate headers that came
# with them; the FlashInfer/CCCL entries are here because the shell's
# `batch/workspace.cu` calls FlashInfer's scheduler directly.
target_include_directories(pie_kernels_cuda
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    # The PTIR contract headers -- see the boundary note at the top.
    ${PIE_PTIR_INCLUDE_DIR}
    # Ahead of anything the toolkit ships: FlashInfer needs its own vendored
    # CCCL, and a toolkit that bundles an older one would shadow it.
    ${PIE_CCCL_INCLUDE_DIRS}
    ${flashinfer_SOURCE_DIR}/include
    ${flashinfer_SOURCE_DIR}/csrc
    ${flashinfer_SOURCE_DIR}/3rdparty/cutlass/include
    ${flashinfer_SOURCE_DIR}/3rdparty/cutlass/tools/util/include
    ${flashinfer_SOURCE_DIR}/3rdparty/spdlog/include)

set_target_properties(pie_kernels_cuda PROPERTIES
  POSITION_INDEPENDENT_CODE ON)

# ---- FlashInfer generated CUTLASS MoE (vendored) ---------------------------
# Pre-generated FlashInfer CUTLASS grouped-GEMM MoE launchers, vendored under
# third_party/flashinfer_generated/ (the SM80 set; byte-identical for
# SM80/86/89 and the only set the build compiles). Removes the build-time
# Python codegen and its tvm_ffi/pynvml/tqdm imports -- no external Python
# deps. Refresh after bumping the vendored FlashInfer revision with:
#   FLASHINFER_SRC=<src> crates/kernels-cuda/csrc/third_party/flashinfer_generated/regenerate.sh
#
# These define sm80_generic_fused_moe_gemm_kernelLauncher, which
# moe_gemm_template_dispatch.h only reaches when the epilogue tag is
# EpilogueOpDefaultSilu or EpilogueOpDefaultFtGelu. With kernels.def enabling
# none of the activations that map onto those tags, every launcher here is
# unreferenced and the whole set drops out of the build -- worth ~5 minutes of
# cicc time, including the second-slowest translation unit in the project.
set(PIE_FLASHINFER_MOE_GEN_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/flashinfer_generated")
set(PIE_MOE_NEEDS_FUSED_LAUNCHERS OFF)
foreach(_pie_act IN LISTS PIE_MOE_ACTIVATIONS)
  # The four ActivationType values that moeGemmBiasAct maps onto those two
  # tags; kept in sync by hand with the dispatch table in
  # third_party/flashinfer_moe/moe_gemm_template_dispatch.h.
  if(_pie_act STREQUAL "Silu" OR _pie_act STREQUAL "Swiglu"
     OR _pie_act STREQUAL "Gelu" OR _pie_act STREQUAL "Geglu")
    set(PIE_MOE_NEEDS_FUSED_LAUNCHERS ON)
  endif()
endforeach()
if(PIE_MOE_NEEDS_FUSED_LAUNCHERS)
  set(PIE_FLASHINFER_MOE_GENERATED_SOURCES
    third_party/flashinfer_generated/gemm_grouped/80/cutlass_kernel_file_gemm_grouped_sm80_M16_group0.generated.cu
    third_party/flashinfer_generated/gemm_grouped/80/cutlass_kernel_file_gemm_grouped_sm80_M16_group1.generated.cu
    third_party/flashinfer_generated/gemm_grouped/80/cutlass_kernel_file_gemm_grouped_sm80_M16_group2.generated.cu
    third_party/flashinfer_generated/gemm_grouped/80/cutlass_kernel_file_gemm_grouped_sm80_M32_group0.generated.cu
    third_party/flashinfer_generated/gemm_grouped/80/cutlass_kernel_file_gemm_grouped_sm80_M32_group1.generated.cu
    third_party/flashinfer_generated/gemm_grouped/80/cutlass_kernel_file_gemm_grouped_sm80_M64_group0.generated.cu
    third_party/flashinfer_generated/gemm_grouped/80/cutlass_kernel_file_gemm_grouped_sm80_M64_group1.generated.cu
    third_party/flashinfer_generated/gemm_grouped/80/cutlass_kernel_file_gemm_grouped_sm80_M128_group0.generated.cu
    third_party/flashinfer_generated/gemm_grouped/80/cutlass_kernel_file_gemm_grouped_sm80_M128_group1.generated.cu)
  list(LENGTH PIE_FLASHINFER_MOE_GENERATED_SOURCES PIE_FLASHINFER_MOE_GENERATED_COUNT)
  message(STATUS "Using ${PIE_FLASHINFER_MOE_GENERATED_COUNT} FlashInfer generated MoE launchers")
else()
  set(PIE_FLASHINFER_MOE_GENERATED_SOURCES)
  message(STATUS
    "FlashInfer fused MoE launchers omitted: no Silu/Gelu-class activation in kernels.def")
endif()

# Blackwell TMA warp-specialized grouped GEMMs. Without
# COMPILE_BLACKWELL_TMA_GROUPED_GEMMS the launcher bodies collapse to a
# "recompile with 100-real" throw, so on sm100 the CutlassMoeFCRunner has no
# usable config at all and MoE falls back to cuBLAS batched GEMM over
# worst-case-padded expert blocks. Requires the `a` architecture target for TMA
# and tcgen05 -- see cmake/DetectCudaArchitecture.cmake.
#
# These come from third_party/flashinfer_generated/regenerate_sm100.sh, and
# unlike the sm80 set they are needed whenever the runner is reachable at all,
# independent of which activations kernels.def enables.
if(PIE_HAS_SM100)
  set(PIE_FLASHINFER_MOE_SM100_SOURCES)
  foreach(_pie_moe_group RANGE 0 23)
    list(APPEND PIE_FLASHINFER_MOE_SM100_SOURCES
      "third_party/flashinfer_generated/gemm_grouped/100/cutlass_kernel_file_gemm_grouped_sm100_group${_pie_moe_group}.generated.cu")
  endforeach()
  list(LENGTH PIE_FLASHINFER_MOE_SM100_SOURCES PIE_FLASHINFER_MOE_SM100_COUNT)
  message(STATUS
    "Enabling Blackwell TMA warp-specialized MoE grouped GEMMs (${PIE_FLASHINFER_MOE_SM100_COUNT} generated units)")
else()
  set(PIE_FLASHINFER_MOE_SM100_SOURCES)
endif()

# Hopper TMA warp-specialized grouped GEMMs. Exactly the same story as the sm100
# block above, for the architecture this box actually is: without
# COMPILE_HOPPER_TMA_GROUPED_GEMMS and these launchers, `getWorkspaceSize` walks
# the TMA warp-specialized configs, finds none with a compiled launcher for
# sm_90, and throws "Could not find valid config when calculating workspace
# size". flashinfer_cutlass_moe_workspace_bytes catches that and reports zero,
# so callers leave `cutlass_ws` empty and EVERY MoE model silently takes the
# unfused expert path.
#
# Generated by third_party/flashinfer_generated/regenerate_sm90.sh (72
# instantiations, 11 units) the same way the sm100 set is: compile the dispatch
# TU, ask the linker which launchers it wants, instantiate exactly those.
if(PIE_HAS_SM90)
  set(PIE_FLASHINFER_MOE_SM90_SOURCES)
  foreach(_pie_moe_group RANGE 0 10)
    list(APPEND PIE_FLASHINFER_MOE_SM90_SOURCES
      "third_party/flashinfer_generated/gemm_grouped/90/cutlass_kernel_file_gemm_grouped_sm90_group${_pie_moe_group}.generated.cu")
  endforeach()
  list(LENGTH PIE_FLASHINFER_MOE_SM90_SOURCES PIE_FLASHINFER_MOE_SM90_COUNT)
  message(STATUS
    "Enabling Hopper TMA warp-specialized MoE grouped GEMMs (${PIE_FLASHINFER_MOE_SM90_COUNT} generated units)")
else()
  set(PIE_FLASHINFER_MOE_SM90_SOURCES)
endif()

# FlashInfer/TensorRT-LLM CUTLASS MoE runner. Kept as a separate OBJECT
# library because its dependency slice (nv_internal/) is large and should
# stay a clear opt-in boundary for Nemotron experimentation.
add_library(pie_flashinfer_cutlass_moe OBJECT
  src/moe/flashinfer_moe.cu
  ${flashinfer_SOURCE_DIR}/csrc/nv_internal/cpp/common/envUtils.cpp
  ${flashinfer_SOURCE_DIR}/csrc/nv_internal/cpp/common/logger.cpp
  ${flashinfer_SOURCE_DIR}/csrc/nv_internal/cpp/common/stringUtils.cpp
  ${flashinfer_SOURCE_DIR}/csrc/nv_internal/cpp/common/tllmException.cpp
  ${flashinfer_SOURCE_DIR}/csrc/nv_internal/cpp/common/memoryUtils.cu
  ${flashinfer_SOURCE_DIR}/csrc/nv_internal/tensorrt_llm/kernels/preQuantScaleKernel.cu
  ${flashinfer_SOURCE_DIR}/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.cpp
  ${flashinfer_SOURCE_DIR}/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_tma_warp_specialized_input.cu
  # Vendored so pie owns the activation dispatch; a quoted #include resolves
  # against the including file's own directory first, so patching the header
  # alone would be ignored by the upstream .cu.
  third_party/flashinfer_moe/moe_gemm_kernels_bf16_bf16.cu
  ${flashinfer_SOURCE_DIR}/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.cu
  ${flashinfer_SOURCE_DIR}/csrc/nv_internal/tensorrt_llm/kernels/lora/lora.cpp
  ${PIE_FLASHINFER_MOE_GENERATED_SOURCES}
  ${PIE_FLASHINFER_MOE_SM90_SOURCES}
  ${PIE_FLASHINFER_MOE_SM100_SOURCES})
target_include_directories(pie_flashinfer_cutlass_moe PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/flashinfer_moe
  ${PIE_CCCL_INCLUDE_DIRS}
  ${CUDAToolkit_INCLUDE_DIRS}
  ${flashinfer_SOURCE_DIR}/include
  ${flashinfer_SOURCE_DIR}/csrc
  ${flashinfer_SOURCE_DIR}/csrc/fused_moe/cutlass_backend
  ${flashinfer_SOURCE_DIR}/csrc/nv_internal
  ${flashinfer_SOURCE_DIR}/csrc/nv_internal/include
  ${flashinfer_SOURCE_DIR}/csrc/nv_internal/tensorrt_llm/cutlass_extensions/include
  ${flashinfer_SOURCE_DIR}/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/include
  ${flashinfer_SOURCE_DIR}/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels
  ${flashinfer_SOURCE_DIR}/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm
  ${flashinfer_SOURCE_DIR}/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm
  ${PIE_FLASHINFER_MOE_GEN_DIR}
  ${flashinfer_SOURCE_DIR}/3rdparty/cutlass/include
  ${flashinfer_SOURCE_DIR}/3rdparty/cutlass/tools/util/include
  ${flashinfer_SOURCE_DIR}/3rdparty/spdlog/include)
target_compile_definitions(pie_flashinfer_cutlass_moe PRIVATE
  ENABLE_BF16=1
  ENABLE_FP8=1
  USING_OSS_CUTLASS_MOE_GEMM=1
  $<$<BOOL:${PIE_HAS_SM90}>:COMPILE_HOPPER_TMA_GROUPED_GEMMS=1>
  $<$<BOOL:${PIE_HAS_SM100}>:COMPILE_BLACKWELL_TMA_GROUPED_GEMMS=1>)
target_compile_options(pie_flashinfer_cutlass_moe PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>)
set_target_properties(pie_flashinfer_cutlass_moe PROPERTIES
  POSITION_INDEPENDENT_CODE ON)
target_sources(pie_kernels_cuda PRIVATE
  $<TARGET_OBJECTS:pie_flashinfer_cutlass_moe>)

# ---- Marlin (W4A16 / MXFP4 GEMM via vendored vLLM kernel) ------------------
# Two vendored kernels with the same lineage and very different build costs, so
# they are two options.
#
# The dense, single-problem GEMM is OFF by default. marlin.cu is the build's
# critical path -- 142s of a 228s clean build, because 150 explicit
# instantiations of a 2082-line __global__ template produce 152 kernels
# averaging 3098 SASS instructions each (91s cicc + 41s ptxas). Nothing else in
# the driver comes close.
#
# What turning it off costs, and why that is an acceptable default:
#   * GPTQ/AWQ INT4 dense GEMM throws at run time (ops/gemm.cpp) with a message
#     naming this option. Those checkpoints are not part of pie's supported set.
# Everything else -- every bf16 and FP8 model -- is untouched.
#
# Vendored under third_party/marlin/; the wrapper exposes a torch-free C++ API.
# Enable with -DPIE_CUDA_BUILD_MARLIN=ON or PIE_CUDA_BUILD_MARLIN=1.
option(PIE_CUDA_BUILD_MARLIN "Build vendored marlin W4A16/MXFP4 GEMM" OFF)
# Env override (previously forwarded by worker/build.rs as -D...).
if(DEFINED ENV{PIE_CUDA_BUILD_MARLIN})
  string(TOLOWER "$ENV{PIE_CUDA_BUILD_MARLIN}" _pie_marlin)
  if(_pie_marlin MATCHES "^(1|on|true|yes)$")
    set(PIE_CUDA_BUILD_MARLIN ON)
  else()
    set(PIE_CUDA_BUILD_MARLIN OFF)
  endif()
endif()

# The expert-indexed MoE variant is ON by default, because it does not share
# the cost that made the dense one opt-in: its generator is pruned to the
# single quant configuration a pie model reaches (MXFP4 / bf16 / E8M0), so it
# is one translation unit rather than fifteen.
#
# It is also what a supported model needs rather than an unsupported one.
# GPT-OSS's MoE is native MXFP4, and without this kernel the driver answers
# `native_mxfp4_moe = false` and the checkpoint falls back to per-route
# dequant. Leaving it off is not neutral for that model; it is a decision to
# serve it on the slow path.
option(PIE_CUDA_BUILD_MARLIN_MOE
  "Build vendored expert-indexed marlin MXFP4 MoE GEMM" ON)
if(DEFINED ENV{PIE_CUDA_BUILD_MARLIN_MOE})
  string(TOLOWER "$ENV{PIE_CUDA_BUILD_MARLIN_MOE}" _pie_marlin_moe)
  if(_pie_marlin_moe MATCHES "^(1|on|true|yes)$")
    set(PIE_CUDA_BUILD_MARLIN_MOE ON)
  else()
    set(PIE_CUDA_BUILD_MARLIN_MOE OFF)
  endif()
endif()

# kernels.def owns the shape list, but it cannot drive marlin.cu directly: a
# macro cannot expand to an #include, and CUDA 13's static-linkage stubs force
# the single-TU #include layout (see below). So the list is written out in
# marlin.cu and reconciled here -- otherwise a drift between the two is
# invisible until it shows up as a missing kernel at run time or as several
# minutes of cicc time spent on shapes nothing dispatches.
#
# Deliberately outside the PIE_CUDA_BUILD_MARLIN gate: this only reads two
# files, and now that marlin is off by default a gated check would never run,
# letting drift accumulate until someone enables it.
pie_read_kernels_def(PIE_MARLIN_SHAPE PIE_MARLIN_SHAPES)
set(_pie_marlin_expected "")
foreach(_pie_shape IN LISTS PIE_MARLIN_SHAPES)
  string(REPLACE "," ";" _pie_parts "${_pie_shape}")
  list(LENGTH _pie_parts _pie_nparts)
  if(NOT _pie_nparts EQUAL 3)
    message(FATAL_ERROR
      "PIE_MARLIN_SHAPE(${_pie_shape}) in ${PIE_KERNELS_DEF} needs exactly "
      "three arguments: (activation dtype, quant type, output dtype)")
  endif()
  list(GET _pie_parts 0 _pie_act)
  list(GET _pie_parts 1 _pie_quant)
  list(GET _pie_parts 2 _pie_out)
  list(APPEND _pie_marlin_expected
    "sm80_kernel_${_pie_act}_${_pie_quant}_${_pie_out}.cu")
endforeach()

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/third_party/marlin/marlin.cu" _pie_marlin_src)
string(REGEX MATCHALL "sm80_kernel_[A-Za-z0-9_]+\\.cu"
  _pie_marlin_actual "${_pie_marlin_src}")
list(SORT _pie_marlin_expected)
list(SORT _pie_marlin_actual)
if(NOT _pie_marlin_expected STREQUAL _pie_marlin_actual)
  message(FATAL_ERROR
    "marlin shape drift.\n"
    "  kernels.def: ${_pie_marlin_expected}\n"
    "  marlin.cu:   ${_pie_marlin_actual}\n"
    "Update the #include block in third_party/marlin/marlin.cu to match "
    "${PIE_KERNELS_DEF}.")
endif()

if(PIE_CUDA_BUILD_MARLIN)
  # CUDA 13 generates __global__ template stubs with static linkage,
  # breaking cross-TU explicit instantiations. marlin.cu #includes the
  # sm80_kernel_*.cu files directly so instantiation and selection share a
  # single TU. Do NOT list sm80_kernel_*.cu as separate sources -- they
  # would double-define every template.
  target_sources(pie_kernels_cuda PRIVATE
    third_party/marlin/marlin.cu
    third_party/marlin/gptq_marlin_repack.cu
    third_party/marlin/awq_marlin_repack.cu
    third_party/marlin/marlin_wrapper.cpp)
  target_include_directories(pie_kernels_cuda PRIVATE
    third_party/marlin)
  target_compile_definitions(pie_kernels_cuda PRIVATE PIE_CUDA_HAS_MARLIN=1)
endif()

# The expert-indexed MoE variant of the same kernel. Separate directory and
# namespace (`marlin_moe_wna16`) so it builds and is revised independently of
# the single-problem path above; `ops.cu` includes its generated instantiation
# for the same CUDA-13 linkage reason marlin.cu does.
#
# The generator is pruned to MXFP4 (bf16 x FE2M1 -> bf16, E8M0 scales): that is
# the only configuration a pie model reaches, and every extra quant config is
# another multi-minute nvcc translation unit. Pruning is what lets this one
# default to ON while the dense path stays opt-in.
if(PIE_CUDA_BUILD_MARLIN_MOE)
  target_sources(pie_kernels_cuda PRIVATE
    third_party/marlin_moe/ops.cu
    third_party/marlin_moe/marlin_moe_wrapper.cpp)
  target_include_directories(pie_kernels_cuda PRIVATE
    third_party/marlin_moe)
  target_compile_definitions(pie_kernels_cuda PRIVATE PIE_CUDA_HAS_MARLIN_MOE=1)
endif()

# FlashInfer Mamba SSU on Hopper/Blackwell gates the vertical/horizontal
# producer-consumer kernels behind FLASHINFER_MAMBA_ENABLE_SM90 at compile
# time (TMA + memcpy_async_tx are sm_90+); see the arch-gated selection above.
if(PIE_CUDA_FLASHINFER_MAMBA_SM90)
  target_compile_definitions(pie_kernels_cuda PRIVATE
    FLASHINFER_MAMBA_ENABLE_SM90=1)
endif()

# The toolkit libraries the kernels themselves call. The shell's own list is
# longer -- CLI11, toml++, nlohmann_json, NCCL, rt, dl -- and stayed with the
# shell: those are how a driver reads a config file and talks to its peers,
# not anything a kernel does. `gemm.cpp` names cuBLAS/cuBLASLt directly and
# the Sampling-IR JIT calls NVRTC, which is why those two are here.
target_link_libraries(pie_kernels_cuda PUBLIC
  CUDA::cudart
  CUDA::cublas
  CUDA::cublasLt
  CUDA::cuda_driver
  CUDA::nvrtc
)


# ---- what driver-cuda needs to know ----------------------------------------
#
# `build.rs` reads this file rather than re-deriving any of it. FlashInfer is
# fetched here and patched here; a second CPMAddPackage on the shell side would
# be a second clone, a second patch pass over the same shared CPM cache, and a
# second chance to disagree. The include dirs are on `pie_kernels_cuda` as
# PUBLIC usage requirements as well -- this file is for the cargo boundary,
# which does not carry CMake targets across.
file(GENERATE
  OUTPUT "${CMAKE_BINARY_DIR}/pie_kernels_cuda_paths.txt"
  CONTENT
"lib=$<TARGET_FILE:pie_kernels_cuda>
include=${CMAKE_CURRENT_SOURCE_DIR}/src
cccl=$<JOIN:${PIE_CCCL_INCLUDE_DIRS},:>
flashinfer=${flashinfer_SOURCE_DIR}/include:${flashinfer_SOURCE_DIR}/csrc:${flashinfer_SOURCE_DIR}/3rdparty/cutlass/include:${flashinfer_SOURCE_DIR}/3rdparty/cutlass/tools/util/include:${flashinfer_SOURCE_DIR}/3rdparty/spdlog/include
marlin=${CMAKE_CURRENT_SOURCE_DIR}/third_party/marlin
marlin_moe=${CMAKE_CURRENT_SOURCE_DIR}/third_party/marlin_moe
has_marlin=$<BOOL:${PIE_CUDA_BUILD_MARLIN}>
has_marlin_moe=$<BOOL:${PIE_CUDA_BUILD_MARLIN_MOE}>
mamba_sm90=$<BOOL:${PIE_CUDA_FLASHINFER_MAMBA_SM90}>
")

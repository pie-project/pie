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
# here.
#
# BOTH OF THOSE HEADERS ARE NOW DELETED. `src/tuning_cache.hpp` (142 lines)
# and `src/cache_root.hpp` (43) came out with `src/gemm/gemm.cpp`, which was
# their last includer once `moe/flashinfer_moe.cu`'s host half became
# `driver-cuda/src/fire/flashinfer_moe.rs`. Sweep for `#include
# "tuning_cache.hpp"` under `csrc/` and it returns nothing. The file format
# they described is not lost -- it is spelled out twice in Rust, in
# `driver-cuda/src/fire/gemm.rs` (`DiskCache`, `tuning_hash`, `cache_path`)
# and in `fire/flashinfer_moe.rs`, and the XDG-then-`$HOME/.cache` derivation
# that was `cache_root.hpp`'s whole content is `fire::gemm::cache_path`. If
# `config.hpp` on the shell side still includes `cache_root.hpp`, it is
# including a path that no longer resolves and wants its own `cache_dir()`.
# Nor did the NCCL comm in `distributed.*` come, nor `custom_all_reduce`,
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
# entered FlashInfer's SSU at all when the device's major version was >= 9, and
# FlashInfer's own selector independently sends sm_major < 9 to kSimple, so
# these kernels were unreachable on those architectures before this patch.
#
# THAT FILE IS NOW DELETED (see `PIE_CUDA_GRAPH_KERNEL_SOURCES` below), so this
# patch and the `FLASHINFER_MAMBA_ENABLE_SM90` define below now guard a header
# that no translation unit in this repository includes. Both stay deliberately.
# The FATAL_ERROR and the asserted site count are load-bearing for the CHECKOUT
# rather than for the include: `driver-cuda` shares this CPM fetch, and a
# FlashInfer bump that moves or reshapes this header must still be loud here
# rather than silently reading as "nothing to patch". Deleting the patch is a
# separate change with its own evidence, and it is cheaper to make after the
# fetch itself has an owner.
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
  # `src/attn/dsa_indexer.cu` was here and IS DELETED, with
  # `attn/dsa_indexer.hpp`. Three `<<<>>>` -- the key norm+rope, the query
  # rope and the causal top-k mask -- now `driver-cuda/src/fire/dsa_indexer.rs`.
  # All three `Execution::Walk` with `Control::Supplies` and named in
  # `execution::RUST_SERVED`. Two device rows were renamed `_dev` (§60.6) and
  # a THIRD was written: `index_q_rope` had no row at all, its block being
  # `round_up(n_heads, 32)` with a one-warp floor, which no `LaunchRule`
  # states. `dsa_index_topk_mask`'s row is fully sourced, so that one moves a
  # live dispatch from the C shim to `bind::service`.
  # `src/attn/dsv4_compress.cu` was here and IS DELETED, with
  # `attn/dsv4_compress.hpp`. Four `<<<>>>` left in it -- the output combine,
  # the two boundary-meta forms and the compressed paged attention -- now
  # `driver-cuda/src/fire/dsv4_compress.rs`. All four `Execution::Walk` with
  # `Control::Supplies`, all four in `execution::RUST_SERVED`, all four rows
  # unsourced so this took the shim entries and nothing else (§60.7). Three
  # device rows were renamed `_dev` (§60.6); a fourth was WRITTEN for
  # `combine_attn_outputs`, deliberately `LaunchRule::Unstated` because the
  # launcher clamps its block to 256 where `PerHeadElementwise` clamps to 128
  # and the disagreement would be a slower kernel rather than a wrong answer.
  # `src/norm/dsv4_hc.cu` was here and IS DELETED with `norm/dsv4_hc.hpp`.
  # Its four launchers -- `hc_pre_postprocess_bf16`, `hc_post_bf16`,
  # `hc_head_postprocess_bf16`, `hc_rmsnorm_to_f32` -- are
  # `execution::RUST_SERVED`, so `abi::emit_c_shim` emits no entry for any of
  # them; the host programs are `driver-cuda/src/fire/dsv4_hc.rs` and the
  # `__global__`s are unmoved in `kernels-cuda-new/csrc/src/norm/dsv4_hc.cuh`.
  # `src/moe/dsv4_routing.cu` was here and IS DELETED with
  # `moe/dsv4_routing.hpp`. Its one surviving launcher, `hash_route_lookup`,
  # is `execution::RUST_SERVED`, so `abi::emit_c_shim` emits no
  # `pie_k_moe_hash_route_lookup` and the shim entry was the whole consumer
  # set: zero C++ callers across all four extensions, no hand-written
  # `ffi::pie_k_*` arm, and `model-compiler/src/dsl.rs:4700`'s
  # `hash_route_lookup` wrapper has no caller in `crates/model/src` --
  # `deepseek_v4/spec.rs:95` names the symbol in PROSE only. The row is
  # unsourced in `table::moe`, so `emit_rust_dispatch` writes no arm either;
  # the host program is `driver-cuda/src/fire/dsv4_routing.rs` and the two
  # `__global__`s are unmoved in
  # `kernels-cuda-new/csrc/src/moe/dsv4_routing.cuh`, where the device row is
  # named `moe::hash_route_lookup_dev`.
  # `src/attn/kimi_mla.cu` was here and the file is DELETED with
  # `attn/kimi_mla.hpp`. Its last launcher, `kimi_split_q_b_bf16`, went the
  # way its sibling `kimi_split_kv_a_norm_bf16` already had:
  # `attn::kimi_split_q_b_bf16` is named in `device::JIT_DISPATCHED`, so
  # `abi::emit_c_shim` emits no `pie_k_attn_kimi_split_q_b_bf16` and the shim
  # entry was the whole consumer set. Both `__global__`s are unmoved in
  # `kernels-cuda-new/csrc/src/attn/kimi_mla.cuh` and are fired through
  # `LaunchRule::Elementwise` and `LaunchRule::Rms`.
  # `src/attn/mla_paged.cu` was here and IS DELETED, with
  # `attn/mla_paged.hpp`. Two `<<<>>>` — `mla_prepare<256>` over
  # `dim3(total_tokens, 1 + q_blocks)` and `write_mla` at `<<<total_tokens,
  # 256>>>` — now `driver-cuda/src/fire/mla_paged.rs`. Both symbols are
  # `Execution::Walk` with `Control::Supplies` and named in
  # `execution::RUST_SERVED`, which is what drops their shim entries; both
  # `table::attn` rows were unsourced on every operand and so never generated
  # a dispatch arm, which is §60.7's case. The file's third function,
  # `write_mla_to_pages_bf16`, was dead on all five channels and is deleted
  # rather than ported.
  # `src/ssm/nemotron_h.cu` was here and IS DELETED with `ssm/nemotron_h.hpp`,
  # and `csrc/src/ssm/` is gone with it. Its last two launchers,
  # `build_nemotron_moe_ptrs_aligned_bf16` and
  # `build_nemotron_moe_ptrs_decode_batched_bf16`, are
  # `execution::RUST_SERVED`, so `abi::emit_c_shim` emits no entry and the
  # shim entry was the whole consumer set -- no `.cu`, `.cpp` or hand-written
  # `ffi::pie_k_*` arm named either. Their `table::ssm` rows STAY unbound:
  # §52.3's missing `Source::Scratch(name, extent)` still has no word for a
  # slab this driver allocated, so no model trace reaches them and
  # `driver-cuda/src/fire/nemotron_h.rs` is reached by hand.
  # `src/attn/page_compact.cu` was here and IS DELETED, with
  # `attn/page_compact.hpp`. Two `<<<num_requests, kBlock>>>` on one stream,
  # the second reading the `scratch_counts` buffer the first fills: now
  # `driver-cuda/src/fire/page_compact.rs`. `attn::compact_page_csr` was
  # already `Execution::Composed` -- the composition that fires end to end --
  # and `execution::RUST_SERVED` is what took the row and the shim entry.
  # This was the only file in the tree that reached into CCCL; the CUB
  # measurement travels in the Rust module header.
  src/attn/attention_mla.cu)
# `src/ssm/flashinfer_mamba.cu` was here and the file is DELETED, which
# re-lands `new-horizon.md` §43. The row `ssm::flashinfer_mamba_ssu_bf16` went
# with `dsl::cuda::flashinfer_mamba_ssu`, no model text ever stated it, and
# `flashinfer_mamba_ssu_enabled()` required sm90 besides. §43 deleted the two
# files under a sweep grant, an upstream merge handed them back, and this
# re-lands the deletion.
#
# It held ZERO `__global__` and zero `<<<>>>`: every kernel it reached was a
# template inside `flashinfer::mamba::invokeSelectiveStateUpdate`. So there was
# no device text to keep, and the host half it did hold -- four `#define`d
# shape constants and a refusal for anything that did not match them -- had no
# caller to serve. Nothing to lift into `kernels-cuda-new/csrc/src` and nothing
# to rewrite in Rust; the whole file was the dead row's launcher.
#
# `kernels-cuda-new/examples/vendor_probe.rs`'s MAMBA candidate is the path
# back and does not depend on this file: its `roots` name the upstream header
# `flashinfer/mamba/selective_state_update.cuh`, and its `launcher:` field has
# read "deleted by new-horizon.md §43" since that sweep. The probe measured the
# text compiling under NVRTC at 396,128 B with 9 of 9 symbols resolving, which
# is what makes re-stating the row cheap.

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


# The custom P2P all-reduce WAS arch-gated here, and both of its files are
# DELETED, which empties `src/comm/`.
#
# It wrapped FlashInfer's vllm-style kernel and FUSED a residual add and an
# RMSNorm into the reduction -- the unfused halves of which are `norm/`'s --
# so it was a compute kernel that happened to be collective, not comm
# machinery. What it needed from the collective was one bootstrap all-gather,
# taken as a callback (`HostAllgather`) rather than an `NcclComm&`.
#
# `src/comm/custom_all_reduce.cu` was then MEASURED: 664 lines, **zero
# `__global__` and zero `<<<>>>`**. It was a host program wearing a `.cu`
# extension for linkage -- lifecycle, IPC exchange, allocation arithmetic, an
# NCCL crossover query and a template dispatch -- and two calls into
# CPM-fetched flashinfer headers this tree does not vendor. All of the first
# is `driver-cuda/src/fire/all_reduce.rs`; the two calls are refusals that
# name the exact template point they want. Both symbols are
# `kernels_cuda_new::execution::RUST_SERVED`, so the generated shim declares
# neither and nothing here needs an entry point.
#
# `src/comm/custom_all_reduce_stub.cpp` -- the sm100/sm120 substitute -- went
# with it, under §48.3: a stub goes when the thing it stands in for goes. It
# could not have compiled if it had ever been selected (its ctor still took
# `NcclComm&`, which the header had replaced with `HostAllgather`), and a
# JIT unit that is never selected costs nothing to not compile, which is
# exactly what an AOT archive cannot do.

add_library(pie_kernels_cuda STATIC
  # vision/ -- the multimodal towers, bridged at TOWER granularity (one
  # launcher per tower; the walk and its host prep stay C++ — see the
  # provenance headers and the retirement wiki's VL judgment).
  # `src/tensor.cpp` WAS NAMED HERE. It is deleted, and it is worth saying
  # what it held because four of its six entry points had NO consumer
  # anywhere in this repository: `alloc_logging_enabled`,
  # `sample_memory_callback`, `set_device_memory_allocator` and
  # `set_device_tensor_memory_callback` -- the whole allocator-BINDING API,
  # whose entire purpose is to let a host bind an allocator, with nothing
  # binding one. The other two, `allocate_device_memory` and
  # `free_device_memory`, were called only from `gemm/gemm.cpp`'s quantized
  # arms, and `DeviceTensor::allocate` / `::view` / `::free_` were called
  # from nowhere at all (the hits under `driver-cuda/tests/oracle/**` are
  # stubs that define their OWN copies, not consumers of this one).
  #
  # So it did not need porting: `bind::quant_gemm` allocates its scratches
  # with `cudarc::runtime::sys::cudaMalloc` directly, which is what the
  # default allocator did once the unbound indirection is removed. The
  # 256-byte alignment argument went with it -- `cudaMalloc` already returns
  # at least 256-byte-aligned memory on every CUDA device, which is why all
  # four call sites passed 256 and the default allocator never read it.
  #
  # `src/tensor.hpp` STAYS: it carries `DType`, `DeviceMemoryBlock` and the
  # `DeviceTensor` declaration that `weight_view.hpp` and the vision headers
  # name. A declaration with no definition is legal until something ODR-uses
  # it, and nothing does.
  # ops/ -- attention / GEMM / MoE / SSM wrappers
  #
  # `src/attn/attention_flashinfer.cu` was named here and IS NOT ON DISK. It
  # was moved to `crates/driver-cuda/csrc/attn/` by the pass before this one,
  # which did not update this list -- so `cmake` configure fails on a
  # nonexistent source and `tests/sources.rs::every_cu_in_the_build_is_on_disk`
  # is red on the same fact. Removing the entry is the smaller half of the
  # repair; the larger half is that the file should not be C++ in either crate.
  # It is a host dispatch over FlashInfer templates, and its home is Rust in
  # `driver-cuda/src/fire/`. Its `AttnHd<HD>::dispatch_decode_capture` is a
  # host `cudaError_t` static member that NVRTC cannot compile, which is an
  # argument for rewriting the host half, not for relocating it.
  ${PIE_ATTENTION_HEAD_DIM_SOURCES}
  ${PIE_CUDA_FLASHINFER_HOPPER_SOURCE}
  # Unconditional, and not part of PIE_CUDA_FLASHINFER_HOPPER_SOURCE above:
  # the decode KV-split path reaches the state merge on every architecture,
  # so it cannot ride the sm90 prefill's stub/real switch.
  src/attn/attention_merge_states.cu
  # `src/attn/attention_naive.cu` was here and IS DELETED, with
  # `attn/attention_naive.hpp`. Two `<<<>>>` left in it -- the MTP state
  # pair, `mtp_shift_hidden<bf16>` at `<<<total_tokens, 256>>>` and
  # `mtp_update_pending_hidden<bf16>` at `<<<num_requests, 256>>>` -- now
  # `driver-cuda/src/fire/attention_naive.rs`. Both are `Execution::Walk`
  # with `Control::Supplies` and named in `execution::RUST_SERVED`; both
  # `table::attn` rows were unsourced, so this took the shim entries and
  # nothing else (§60.7). The device rows were renamed `_dev` in the same
  # change (§60.6), because §52.11 forbids walking a symbol a unit hosts.
  # The deleted three launchers' host `if` on `max_global_tokens +
  # history_steps > 8192` is restated in the Rust module header.
  src/attn/attention_naive_paged.cu
  src/attn/attention_xqa.cu
  src/attn/attention_xqa_gqa2.cu
  src/attn/attention_xqa_gqa2_p16.cu
  src/attn/attention_xqa_gqa4.cu
  src/attn/attention_xqa_gqa8.cu
  # `src/gemm/gemm.cpp` WAS HERE AND IS DELETED, with `gemm/gemm.hpp`, and
  # `src/gemm/` goes with them -- they were the last two files in it.
  #
  # It was the largest `.cpp` in the archive and it held **zero
  # `__global__` and zero `<<<>>>`** from the first day anyone measured it.
  # That is the whole finding: it was never a kernel file, it was a HOST
  # PROGRAM -- a shape ladder, a cuBLASLt plan cache, a private-stream
  # autotuner and an on-disk tactic memo -- wearing a translation unit for
  # linkage reasons. What condemned it is not that kernels compile through
  # NVRTC but the stronger rule: every piece of CPU-side code is Rust.
  #
  # Where it went, in the order it left:
  #
  #   * the four pure-cuBLAS bodies (`act_x_wt_bf16_out_fp32`,
  #     `grouped_act_x_wt_bf16`, the two MLA absorbs) ->
  #     `driver-cuda/src/bind/service.rs`, through `cudarc`.
  #   * the quantized router -- `act_x_w`'s seven-arm switch on `w.dtype`,
  #     the FP8 blockwise W8A8 path, the INT8 W8A8 path, the MXFP4 Marlin
  #     path and their dequant caches -> `driver-cuda/src/bind/quant_gemm.rs`.
  #   * the DENSE autotuner and `gemm_bf16_impl` itself, plus
  #     `gemm_batched_bf16_impl` and the grouped-batched capture memo ->
  #     `driver-cuda/src/fire/gemm.rs`.
  #   * `CublasHandle` -> `driver-cuda/src/device/cublas.rs`, which predates
  #     this deletion; `tests/cublas_handle_parity.rs` holds the golden.
  #
  # The rows are carried by `execution::RUST_SERVED`, which is what drops
  # their shim entries, and each is classified in `execution::SERVED`,
  # `COMPOSED` or `WALKED` first.
  #
  # It was already UNCOMPILABLE when it was deleted: `gemm.cpp:22` still had
  # `#include "gemm/gemv.hpp"` and `src/gemm/gemv.{cu,hpp}` had gone with the
  # NVRTC port below. So this entry named a file that could not have built.
  #
  # `tests/oracle/cublas_handle/run.sh` compiled the real `gemm.cpp` over a
  # cuBLAS declaration stub to regenerate that golden. It cannot any more,
  # and `cublas_handle_parity.rs` says so where a reader will meet it: the
  # transcript is now a permanent record of behaviour that can be READ but
  # not re-derived. (`tests/oracle/memory_planner/` is unaffected -- it
  # includes its own `stub/gemm/gemm.hpp`, not this one.)
  # `src/attn/plan_lifecycle.cpp` was named here and IS NOT ON DISK either --
  # same pass, same omission. Its comment used to argue that the plan caches'
  # `extern "C"` lifecycle belongs beside the plan types rather than in the
  # driver, citing `cuda.md` §5.E5's second exit because the first (move it to
  # Rust) was "closed by the `unique_ptr` with a custom deleter". That is no
  # longer an exit: a `unique_ptr` with a custom deleter is host code, and host
  # code is Rust. The lifecycle wants to be a Rust type owning a raw plan
  # pointer with `Drop`, not a `.cpp` in any crate.
  # kernels/ -- fused kernels
  # `src/sample/argmax.cu` was here and IS DELETED with `sample/argmax.hpp` --
  # the last two files under `src/sample/`, so the directory went too. It held
  # one launcher, `sample::lm_head_gemv_argmax_int8`, and that launcher is the
  # clearest case in the archive of the owner's second clause: *where host
  # code is needed to compose kernels, because kernels produce intermediate
  # results or because device-specific tuning is involved, that host code is
  # all Rust.* Both halves are true of it at once -- an occupancy query sets
  # `grid.x`, and a growable `static` device buffer carries the first kernel's
  # per-tile pairs to the second, which the row's operand list does not
  # mention. It is now `execution::WALKED` + `execution::RUST_SERVED`, with
  # the program in `driver-cuda/src/fire/lm_head_argmax.rs` firing two rows of
  # `kernels-cuda-new`'s `sample/argmax` unit. §43 had already taken eight of
  # this file's twelve launchers and `gemm.cpp`'s `lm_head_argmax_chunked`
  # (zero callers anywhere) took two more.
  # `src/ssm/causal_conv1d.cu` was here and IS DELETED with
  # `ssm/causal_conv1d.hpp`. Two launchers: `causal_conv1d_update_batched_bf16`
  # is in `device::JIT_DISPATCHED` (`families::ssm` states its geometry as
  # `LaunchRule::SplitPacked` and `abi::emit_c_shim` emits no entry), and
  # `causal_conv1d_prefill_batched_bf16` is `execution::RUST_SERVED` with its
  # host program in `driver-cuda/src/fire/causal_conv1d.rs`. Both
  # `__global__`s are unmoved in
  # `kernels-cuda-new/csrc/src/ssm/causal_conv1d.cuh`, and the prefill's host
  # `if (R >= 8)` — a channel-tiled kernel against a block-per-channel one —
  # survives as `Control::Switch`. It is NOT the shape §30 deletes: the two
  # arms open on different indices, so each is correct only under its own grid
  # and there is no shape at which they agree.
  # `src/layout/deinterleave.cu` was here and IS DELETED with
  # `layout/deinterleave.hpp`. Four launchers: `split_q_gate_bf16` is in
  # `device::JIT_DISPATCHED` (NVRTC compiles it out of
  # `kernels-cuda-new/csrc/src/layout/deinterleave.cuh` and the shim forwards
  # to nothing), and `deinterleave_rows_bf16`, `deinterleave_vec_bf16` and
  # `concat_bf16_rows` had NO consumer in any language -- no `crates/model`
  # caller of either the symbol or its `dsl::cuda` wrapper, no C++ caller, no
  # hand-written `ffi::pie_k_*` arm. Their `table::layout` rows and their four
  # DSL wrappers went with the file; see `new-horizon.md` §54.
  # `src/quant/dequant_fp4.cu`, `src/quant/dequant_fp8.cu` and
  # `src/quant/dequant_wna16.cu` were here and are ALL DELETED with their
  # `.hpp`s. Every launcher in the three is in `device::JIT_DISPATCHED` --
  # `quant::dequant_mxfp4_to_bf16`; `quant::dequant_fp8_e4m3_to_bf16`,
  # `..._per_channel`, `..._per_group` and `quant::bf16_to_fp16`;
  # `quant::dequant_wna16_int4b8_to_bf16` -- so `abi::emit_c_shim` emits no
  # entry for any of them, and the shim entry was the whole consumer set. The
  # `norm/rmsnorm.cu:64` call on `dequant_wna16` that the migration notes
  # recorded went with that file, which was deleted a round earlier.
  # `src/quant/dtype_cast.cu` was here and IS DELETED with
  # `quant/dtype_cast.hpp`. Both of its launchers -- `cast_fp32_to_bf16` and
  # `scale_rows_bf16` -- are in `device::JIT_DISPATCHED`, and the two
  # hand-written `ffi::pie_k_quant_*` call sites that held them (in
  # `driver-cuda/src/fire/lora.rs`) now fire the rows through NVRTC via
  # `driver-cuda/src/fire/dtype_cast.rs`. Both geometries were already stated
  # as `LaunchRule`s, so nothing was invented.
  # `src/layout/embed.cu` was here and IS DELETED, with `layout/embed.hpp`.
  # With it, `csrc/src/layout/` is EMPTY — the whole directory has left this
  # build. Two `<<<>>>`, one template with two instantiations, chosen from a
  # host alignment test: now `driver-cuda/src/fire/embed.rs`.
  # `layout/embed.cuh` is a NVRTC unit again (`families/layout.rs`' `EMBED`),
  # which it had to become — the `.cu` was the only thing compiling the flat
  # gather. The `layout::embed_bf16` row moved from `table::driver_internal`
  # to `table::layout` so `execution::RUST_SERVED` could take it; the move is
  # justified by `model-compiler/src/lower.rs:1462`, which lowers `Embed` to
  # that symbol from a statement.
  # `src/layout/envelope.cu` was here and IS DELETED, with
  # `layout/envelope.hpp`. Five `<<<>>>` — three launchers, one of which fired
  # one kernel or two on `num_tokens <= kEnvelopeFuseMaxTokens`. The whole
  # quest envelope tier is `driver-cuda/src/fire/envelope.rs` now, firing
  # `families::layout::ENVELOPE`'s five `LaunchRule::Unstated` rows through a
  # driver-owned `kernels::Launch`; `kernels-cuda-new/csrc/src/layout/
  # envelope.cuh` is unchanged and is that unit's root.
  #
  # CONSUMERS, all five channels. `crates/model/src` and
  # `model-compiler/src/dsl.rs`: none possible — the two rows lived in
  # `table::driver_internal`, which is deliberately not in `TABLES`, so
  # `dsl::cuda` never generated a wrapper. Hand `ffi::pie_k_*` arms in
  # `driver-cuda/src`: one, `bind::abi::seed_envelopes_empty`, rewired to
  # `fire::envelope::seed_empty` with its own signature unchanged. C++:
  # `attn/kv_paged.cu` called two of the three, and both of ITS callers moved
  # to Rust in the same change. The three
  # `driver-cuda/tests/oracle/*/stub/layout/envelope.hpp` files define their
  # own `inline` logging bodies and shadow the real header, so they never
  # linked against this and do not notice it left.
  # `src/ssm/gated_delta_net.cu` and `src/ssm/kda.cu` were here and BOTH ARE
  # DELETED with `ssm/gated_delta_net.hpp` and `ssm/kda.hpp`. Between them
  # they held 21 of the archive's 109 `<<<>>>`.
  #
  # `gated_delta_net.cu`: nine launchers, and only four had reachable bodies
  # left. Five were already routed through `device::JIT_DISPATCHED`
  # (`recurrent_gated_delta_step_batched{,_state_bf16,_gqa}` and
  # `chunk_gated_delta_prefill_batched_warp_tiled_gqa{,_state_bf16}`); the
  # four that remained — `qwen_gdn_post_conv_prep_bf16`, the GQA decode step
  # and the two prefill pairs — are `execution::RUST_SERVED`, with the host
  # program in `driver-cuda/src/fire/gated_delta_net.rs`.
  #
  # EIGHT OF ITS SEVENTEEN `<<<>>>` WERE UNREACHABLE. Three `constexpr bool`
  # selectors in its anonymous namespace — `qwen_gdn_gqa_ilp2_enabled`,
  # `qwen_gdn_k_last_state_enabled` and `qwen_gdn_fused_step_enabled` — were
  # all `false`, so every `_fused` launch and every `KLast = true`
  # instantiation sat behind a branch no build reached. None is ported and
  # none gets a row: a row for an unreachable kernel is a contract with an
  # empty consumer set.
  #
  # Its file-local `gdn_raise_shmem_cap` went too, and NOT into the driver:
  # the cached prefill asks for `K_d * V_d * sizeof(float)` = 65,536 bytes
  # against CUDA's 48 KiB default, and the opt-in is now
  # `runtime::module::raise_dynamic_smem_cap`, called from
  # `KernelModule::fire` for ANY launch over the cap rather than for this one.
  #
  # `kda.cu`: four launchers. `kda_gate_beta_bf16` and `kda_o_norm_gated_bf16`
  # are in `device::JIT_DISPATCHED`; `kda_recurrent_step_batched` and
  # `kda_prefill_batched` are `execution::RUST_SERVED` with their program in
  # `driver-cuda/src/fire/kda.rs`. Those two rows are DELIBERATELY UNSOURCED —
  # `state_base` is a driver-owned slab and `Source` has no `Scratch`, which
  # `new-horizon.md` §52.3 counts as one of ten such rows — so
  # `emit_rust_dispatch` writes no arm for either and they are unreachable
  # from a model trace. Listing them in `RUST_SERVED` is still what drops the
  # shim entries, which is what let this file go. The prefill's block width,
  # `min(32, D) * 32`, carries a measurement that survives in the Rust: 2.2x
  # at T=2048, 26.2 ms -> 12.0 ms per layer at K3's widths.
  # `src/gemm/gemv.cu` was here and IS DELETED. Its two `__global__`
  # templates are `kernels-cuda-new/csrc/src/gemm/gemv.cuh`, which NVRTC
  # compiles; its host launcher `gemv_bf16` -- the compute-capability read,
  # the `N <= 4096` threshold, the `K % 8` / 16-byte refusal and the grid
  # arithmetic -- is `driver-cuda/src/fire/gemv.rs`. `gemm.cpp` still calls
  # `gemv_bf16` and its `#include "gemm/gemv.hpp"` no longer resolves; that
  # call site belongs to the cuBLAS rewrite and is repointed there.
  # `src/layout/gather_tokens.cu` was here and IS DELETED with
  # `layout/gather_tokens.hpp`. The header declared NOTHING: the file's only
  # content was an anonymous-namespace `launch()`, which no other translation
  # unit can name and which nothing in its own called. A `static` function
  # with no caller in its own TU is a translation unit that compiles two
  # kernels and launches neither.
  # `src/attn/head_dim_pad.cu` was here and the file is DELETED with
  # `attn/head_dim_pad.hpp`. Both its launchers -- `pad_head_dim_bf16` and
  # `strip_head_dim_bf16` -- are named in `device::JIT_DISPATCHED`, so
  # `abi::emit_c_shim` emits no entry for either and the two entries were the
  # whole consumer set: no `.cu`/`.cuh`/`.cpp`/`.hpp` included the header and
  # `driver-cuda` has no hand-written `ffi::pie_k_attn_*_head_dim_bf16` arm.
  # The file's own banner still said "Neither has a row"; the rows landed and
  # the launchers did not follow. `LaunchRule::PerHead` states the
  # `dim3(num_heads, num_tokens)` grid at `device::kPadBlock` threads, which
  # is the geometry that was here.
  src/attn/kv_paged.cu
  # `src/moe/moe_dispatch.cu` was here and IS DELETED with
  # `moe/moe_dispatch.hpp`. Its last three launchers --
  # `scatter_add_weighted_bf16`, `moe_bucket_exact` and
  # `add_moe_route_bias_bf16` -- joined the five already gone, and all eight
  # are `execution::RUST_SERVED`, so `abi::emit_c_shim` emits no entry for
  # any of them. For the last three the shim entry was the whole consumer
  # set: zero C++ callers across all four extensions, no hand-written
  # `ffi::pie_k_*` arm, and their `model-compiler/src/dsl.rs` wrappers
  # (`:5669`, `:4995`, `:4174`) have no caller in `crates/model/src`. All
  # three `table::moe` rows are unsourced, so `emit_rust_dispatch` writes no
  # arm either. The host program is
  # `driver-cuda/src/fire/moe_dispatch.rs`; the `__global__`s are unmoved in
  # `kernels-cuda-new/csrc/src/moe/moe_dispatch.cuh` and their device rows
  # carry `_dev` names, because `execution`'s `a_walk_is_only_a_walk` refuses
  # a walked symbol that a unit hosts.
  #
  # The header's `moe_aligned_block()` had no C++ caller and went with it:
  # the function, `kMoeAlignedBlockMin`/`Max` and the kimi26-mini batch-128
  # measurement behind the cap are `fire::moe_dispatch::moe_aligned_block`.
  # `crates/model/src/{glm_5/spec.rs:63,qwen_3_5/forward/mod.rs:12}` and
  # `kernels-cuda-new/csrc/src/moe/moe_grouped_gemm_tile.cuh:79` name it in
  # prose and now point at the Rust.
  #
  # `src/moe/` is left holding `flashinfer_moe.cu` alone, below, which is now
  # ONLY the CUTLASS instantiation seam. That makes this target the LAST
  # ahead-of-time CUDA compilation in the family, which is a state and not a
  # settlement: the principle has no exception any more (no nvcc under any
  # circumstances), so this target, PIE_FLASHINFER_MOE_GENERATED_SOURCES, the
  # `_SM90_`/`_SM100_` lists and the CPM fetch that feeds them all have to go
  # -- and that is no longer blocked on an unmeasured question. A concrete
  # sm90 ptr-array grouped GEMM compiles under NVRTC at compute_90a: 1,245,452
  # B of PTX, exactly one `.entry`. The gap was three names in namespace `std`
  # (`is_pointer_v`, `void_t`, `max`), three specific cub includes instead of
  # `cub/cub.cuh`, and two `griddepcontrol` asm lines -- NOT §13.6's "patch
  # set plus ~39 intrinsics", which was priced for FA2. What is still open is
  # `GemmKernel::Params`, which is host-side and which no probe touched.
  # `driver-cuda/src/fire/flashinfer_moe.rs` carries the probe table. Its
  # 817-line host half (workspace
  # query, arch probe, autotuner, tactic cache, dispatch; 0 `__global__`, 0
  # `<<<`, 0 `__device__`)
  # is `driver-cuda/src/fire/flashinfer_moe.rs`, and the symbol is
  # `execution::RUST_SERVED`'s `moe::flashinfer_cutlass_moe_bf16`. Five
  # `extern "C"` entries remain because `CutlassMoeFCRunner<__nv_bfloat16,
  # __nv_bfloat16>` is a class template no other language can name.
  # `src/moe/moe_grouped_gemm.cu` was here and IS DELETED with
  # `moe/moe_grouped_gemm.hpp`. Its one launcher is
  # `execution::RUST_SERVED`'s `moe::moe_grouped_gemm_bf16`, so
  # `abi::emit_c_shim` emits no entry for it and the body is
  # `driver-cuda/src/fire/moe.rs` — the first Rust service in that list whose
  # body is a `__global__` of ours rather than a library call. The device
  # text is unmoved: `kernels-cuda-new/csrc/src/moe/moe_grouped_gemm.cuh`,
  # which is now also the `MOE_GROUPED_GEMM` NVRTC unit's root. The host
  # support predicate went with the launcher and is
  # `fire::moe::supported`.
  # `src/quant/mxfp4_marlin.cu` was here and IS DELETED with
  # `quant/mxfp4_marlin.hpp`. Its one launcher,
  # `mxfp4_scales_to_marlin_e8m0`, is in `device::JIT_DISPATCHED`. The host
  # VALIDATION it carried (the `row_select` check, the group/stride checks and
  # the `total % 64 != 0` refusal, all `std::runtime_error`) was already lost
  # when the row was routed, not by this deletion; it is written down on the
  # row in `kernels-cuda-new/src/families/quant.rs` and in `new-horizon.md`
  # §54 so that a future guard has a specification to be written from.
  # `src/attn/pack_dense_mask.cu` was here and IS DELETED, with
  # `attn/pack_dense_mask.hpp`. Two `<<<>>>` — `pack_dense_mask` at `<<<B,
  # 128>>>` and `pack_structured_mask` at the same — and an EMPTY CONSUMER
  # SET on all five channels: no `crates/model/src` hit for either symbol or
  # a DSL wrapper (none can exist, both rows were `table::driver_internal`
  # and that table is not in `TABLES`), nothing in `model-compiler`, no hand
  # `ffi::pie_k_*` arm, and no C++ caller — the three `.cuh` hits are device
  # text in `kernels-cuda-new`. Deleted rather than ported, per §60.1: a port
  # of a launcher with an empty consumer set is a contract nobody signed.
  # The two DEVICE rows (`families/attn.rs:1206`, `:1222`) stay.
  # `src/quant/quant_bf16_to_fp8.cu` was here and IS DELETED with
  # `quant/quant_bf16_to_fp8.hpp`, which empties `csrc/src/quant/`.
  #
  # THIS IS THE FILE THE THREE HAND-WRITTEN `ffi::pie_k_quant_*` ARMS HELD
  # ALIVE, and they are invisible to every check that reads generated text:
  # `bind/quant_gemm.rs` called `pie_k_quant_quantize_bf16_to_int8_per_token`,
  # `pie_k_quant_dequant_int32_w8a8_to_bf16` (twice) and
  # `pie_k_quant_quantize_bf16_to_fp8_e4m3_per_token_group`. All three now
  # call `driver-cuda/src/fire/quant_int8.rs`, and the three
  # `table/driver_internal.rs` rows that existed only to give them a shim
  # entry are deleted with them.
  #
  # Of the file's four launchers, three ported and one was DEAD:
  # `launch_dequant_int8_to_bf16_per_channel` had no caller in any language --
  # no `.cu`, `.cpp`, `.cuh` or `.hpp`, no `table::quant` row, no hand arm --
  # so it is deleted rather than ported. Its `__global__` keeps its
  # `LaunchRule::Elementwise` row in `families::quant`.
  # `src/quant/quant_bf16_to_mxfp4.cu` was here and IS DELETED with
  # `quant/quant_bf16_to_mxfp4.hpp`. Its one launcher,
  # `quantize_bf16_to_mxfp4_e2m1_per_block`, is in `device::JIT_DISPATCHED`
  # and `model-loader` reaches it through the generated dispatch.
  # `src/norm/residual_add.cu` was here and IS DELETED with
  # `norm/residual_add.hpp`, which empties `csrc/src/norm/`.
  # `src/norm/rmsnorm.cu` was here and IS DELETED with `norm/rmsnorm.hpp`.
  # Three of its five launchers are `device::JIT_DISPATCHED`
  # (`rmsnorm_bf16`, `rmsnorm_strided_bf16`, `rmsnorm_gated_fp32_in_bf16`)
  # and two are `execution::RUST_SERVED` (`rmsnorm_bf16_with_fp16`,
  # `rmsnorm_residual_add_scale_rmsnorm_bf16`), so the shim emits nothing for
  # the file. `driver-cuda/src/fire/rmsnorm.rs` is the host program.
  #
  # `src/norm/residual_add.cu` above IS DELETED, and the sentence that used to
  # stand here is why it took a round longer than the rest of `norm/`: it read
  # *"STAYS: `gemm/gemm.cpp:1990` calls `residual_add_bf16` from the INT8
  # `beta != 0` arm, and §10.10 is that a launcher goes when its WHOLE
  # consumer set has gone."* That call site is now
  # `bind::jit::fire("norm::residual_add_bf16", ...)` in
  # `driver-cuda/src/bind/quant_gemm.rs`, the row is in
  # `device::JIT_DISPATCHED`, and the consumer set is empty in both languages.
  # The `__global__` is unmoved in
  # `kernels-cuda-new/csrc/src/norm/residual_add.cuh` and the geometry was
  # already `LaunchRule::Elementwise`, so nothing was invented.
  #
  # `src/rope/rope.cu` was here and IS DELETED. Three of its twelve
  # launchers are `device::JIT_DISPATCHED` and nine are
  # `execution::RUST_SERVED`; `driver-cuda/src/fire/rope.rs` is the host
  # program and the `__global__`s are unmoved in
  # `kernels-cuda-new/csrc/src/rope/rope.cuh`. `rope/rope.hpp` SURVIVES the
  # `.cu` on purpose and declares functions nothing now defines --
  # `driver-cuda/tests/launch_abi.rs` is the ABI pilot and reads that header
  # from disk by a hard-coded path. Nothing includes it, so nothing can fail
  # to link; retiring the pilot is what deletes it.
  # `src/attn/qkv_fused.cu` was here and IS DELETED, with
  # `attn/qkv_fused.hpp`. It was the LAST `attn` dispatch, and the one whose
  # host `if` chain no `Specialisation` could state: `head_dim` picks the warp
  # form at 64/128/256 and falls THROUGH to the block form otherwise, and the
  # two forms have different `LaunchRule`s, which `Specialisation::agrees`
  # forbids. `families::attn` had already written that refusal; this change
  # answers it with a host program. Four `<<<>>>` -> the eight instantiations
  # of `driver-cuda/src/fire/qkv_fused.rs`. FOUR device rows were WRITTEN for
  # the `_d64`/`_d256` warp expansions the unit never had -- the absence this
  # file's own header named as what blocked its port. `Execution::Walk` with
  # `Control::Switch` + `execution::RUST_SERVED`; the row is FULLY SOURCED
  # (23/23), so this one moves a LIVE dispatch off the generated C shim.
  # `src/layout/slot_ops.cu` was here and IS DELETED with
  # `layout/slot_ops.hpp`. Its one launcher, `copy_if_valid_slot`, had no
  # consumer in any language. The `table::layout` row and the `dsl::cuda`
  # wrapper went with it; the DEVICE row in `families::layout` STAYS, because
  # it is the only witness `kernels-cuda-new/tests/launch_rules.rs` has for
  # `LaunchRule::Single` and that test resolves through `unit_of`, not through
  # the table.
  # `src/attn/split_packed.cu` was here and IS DELETED, with
  # `attn/split_packed.hpp`. One `<<<dim3(xblocks, n_max), 256>>>` of
  # `split_qkv_devwin<bf16>`: now `driver-cuda/src/fire/split_packed.rs`.
  # `families/attn.rs` argued that no `LaunchRule` may state that geometry --
  # `grid.y` is the fire's lane count and not the statement's rectangle, and
  # the four buffers are base pointers the JIT binder would window twice --
  # and both clauses are answered by a driver-owned `Launch` plus
  # `fire::hand::fire`, which binds what it is given. The row moved from
  # `table::driver_internal` to `table::attn` so `RUST_SERVED` could take it;
  # `model-compiler/src/lower.rs:1503` names the symbol from a statement.
  # `src/mlp/swiglu.cu` was here and IS DELETED with `mlp/swiglu.hpp`, and
  # `src/mlp/` is now an empty directory. Its one remaining launcher,
  # `sigmoid_gate_inplace_bf16`, is in `device::JIT_DISPATCHED`, as are all
  # twelve `table::mlp` rows; the header declared that one function and
  # nothing else.
  # `src/moe/topk_softmax.cu` WAS HERE and IS DELETED with
  # `moe/topk_softmax.hpp`. All three of its launchers were unreachable:
  # `topk_softmax_bf16` and `apply_per_expert_scale_bf16` are named by
  # `device::JIT_DISPATCHED`, so `abi::emit_c_shim` emitted no entry for
  # either; `topk_softmax_bf16_form` had no `table::moe` row at all, so it
  # never had an entry to lose. No `.cu`, `.cuh`, `.cpp` or `.hpp` in either
  # archive called any of the three. The nine templates are unchanged in
  # `kernels-cuda-new/csrc/src/moe/topk_softmax.cuh`, which is what the
  # `TOPK_SOFTMAX` unit compiles. The WARP LADDER the deleted launcher chose
  # between is recorded on `families::moe`'s `topk_softmax` row and specified
  # in `new-horizon.md` §52 — it is a host decision that has not been
  # re-landed, not one that was thrown away.
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
# WHICH `flashinfer/` TREE ANSWERS, since there are two and the spelling of an
# `#include <flashinfer/...>` does not say.
#
# `crates/kernels-cuda-new/csrc/vendor/flashinfer/` carries five of the twelve
# distinct headers this archive's C++ names: `attention/cascade.cuh`,
# `attention/mla.cuh`, `attention/scheduler.cuh`, `layout.cuh` and
# `fastdiv.cuh`. It carries none of the other seven --
# `attention/hopper/default_params.cuh`, `attention/hopper/prefill_sm90.cuh`,
# `cutlass_utils.cuh`, `comm/trtllm_allreduce.cuh`,
# `comm/trtllm_allreduce_fusion.cuh`, `comm/vllm_custom_all_reduce.cuh` and
# `mamba/selective_state_update.cuh`.
#
# The C++ takes the CPM copy for ALL TWELVE, including the five that are also
# vendored. The list above is the whole reason: `${flashinfer_SOURCE_DIR}` is
# the CPM checkout and `csrc/vendor` appears nowhere in it, in any
# configuration, on any branch of this file. There is no `-I` in this
# repository that puts the vendored tree in front of a C++ compiler.
#
# The vendored copies are NVRTC's alone. They are reachable only through
# `Headers::LibraryAndVendor`, and `kernels-cuda-new/src/unit.rs`'s `DEMANDS`
# table is empty -- no unit asks for it -- so `ALL_HEADERS` is referenced by
# that crate's tests and by `examples/vendor_probe.rs` and by nothing on the
# JIT path. `csrc/vendor` therefore has ZERO consumers on either side today,
# and deleting this archive is not what unblocks deleting it.

set_target_properties(pie_kernels_cuda PROPERTIES
  POSITION_INDEPENDENT_CODE ON)

# ---- where the device headers went -----------------------------------------
#
# Every `__global__` these translation units compile is a template in a
# `.cuh`, and the `.cuh` files are not here any more: fifty-seven of them
# moved to `crates/kernels-cuda-new/csrc/src`, the JIT crate, which reads
# them into thirty-eight NVRTC units and roughly a hundred and forty rows
# and is now by far their larger consumer. The `.cu` and `.hpp` files stayed
# -- a translation unit is what this file compiles and a host interface is
# what g++ compiles, and neither is text a JIT header set may carry. It is a
# MOVE: there is still exactly one definition of every kernel in the tree,
# which `tests/sources.rs::no_global_is_defined_twice` fails the build over.
#
# The path is spelled relative to THIS file rather than taken from cargo,
# and that is not a shortcut. It was written when a cargo dependency could
# not have carried it at all: `kernels-cuda-new` depended on THIS crate for
# the signature table, so an edge the other way was a cycle, and `links`
# metadata is a dependency wearing different clothes and would have been the
# same cycle. The tables have since moved and the edge inverted -- this
# crate now depends on `kernels-cuda-new` and re-exports its rows -- so the
# direction is no longer the objection. What is left is the objection that
# was always the stronger one:
#
#   * `kernels-cuda-new` deliberately has no `links` key. Its `src/lib.rs`
#     explains the absence as the design: the crate builds nothing native
#     and so has no native artifact to be unique about, and `links` is how
#     cargo makes that uniqueness enforceable. No `links`, no `DEP_*`, and
#     `DEP_*` is the only channel a dependency has for telling a build
#     script where its files are. The edge exists and still says nothing.
#   * Giving it a `links` key to open that channel would mean claiming a
#     native artifact this crate does not have, so that a CMake file could
#     learn a path it can already compute.
#
# An include directory is a filesystem fact. Both crates live in `crates/`,
# this file's own directory fixes the rest, and nothing cargo could say
# would be more stable than that.
set(PIE_JIT_DEVICE_HEADERS
  "${CMAKE_CURRENT_SOURCE_DIR}/../../kernels-cuda-new/csrc/src")
# The impersonating headers -- `cuda_fp16.h`, `cuda_bf16.h`, `cuda_fp8.h`,
# `cuda_fp4.h`, `cooperative_groups.h`, `cstdint`, `type_traits`, `bit`,
# `cuda.h`, `cuda_runtime.h`, `cuda/{cmath,pipeline,std/limits}` -- moved out
# of that tree when `csrc/` was re-cut by role. They are a SECOND `-iquote`
# and not an addition to the first, and this is load-bearing rather than
# tidy: the traffic crosses in both directions. Two files under `csrc/src`
# reach the shim by QUOTED include (`pie_fp8.cuh` says `#include
# "cuda_fp16.h"` and `#include "cuda_fp8.h"`; `pie_half2.cuh` says the
# first), and both `shim/cuda_fp16.h` and `shim/cuda_bf16.h` come back the
# other way with `#include "pie_device.cuh"`. All five resolved BESIDE the
# includer while the two trees were one. The inward direction fails loudly if
# this is missed -- there is no other `pie_device.cuh`. The outward direction
# does not: a quoted `"cuda_fp16.h"` that misses falls through to the real
# toolkit header, `__half` stops being `device::f16`, and the build SUCCEEDS
# -- the silent 17,744-vs-15,088-byte divergence
# `kernels-cuda-new/src/source.rs` measures.
set(PIE_JIT_SHIM_HEADERS
  "${CMAKE_CURRENT_SOURCE_DIR}/../../kernels-cuda-new/csrc/shim")
if(NOT EXISTS "${PIE_JIT_SHIM_HEADERS}/cuda_fp16.h")
  message(FATAL_ERROR
    "the impersonating headers are not at ${PIE_JIT_SHIM_HEADERS}. Three "
    "files under csrc/src reach them by quoted include, and a quoted include "
    "that misses resolves to the real toolkit header WITHOUT a diagnostic, so "
    "this fails now rather than shipping two incompatible __half types.")
endif()
if(NOT EXISTS "${PIE_JIT_DEVICE_HEADERS}/pie_device.cuh")
  message(FATAL_ERROR
    "the kernel headers are not at ${PIE_JIT_DEVICE_HEADERS}. Every .cu here "
    "includes at least the prelude, so this fails now with one message "
    "instead of once per translation unit with a resolver's.")
endif()

# `-iquote`, not an include directory, and this one is measured: nvcc 13.0.88
# with `-I ${PIE_JIT_DEVICE_HEADERS}` resolves the seventeen
# `#include <cuda_bf16.h>` directives under `src/` to that tree's SHIM of the
# same name -- eighteen names written against the f16x2 instructions so that
# NVRTC, which bundles no device headers before 13.3, has something to answer
# with. The shims wear NVIDIA's filenames on purpose (`new-horizon.md` §13.4:
# a shim impersonates exactly when its includer is upstream source we do not
# own), and an `-I` puts the impersonation ahead of the real toolkit header.
#
# `-iquote` says the narrow thing that is actually wanted: this directory
# answers `#include "..."` -- our own spellings, `pie_device.cuh`,
# `attn/kv_paged.cuh` -- and is never searched for `#include <...>`, so no
# shim can shadow a vendor header no matter what is added to it later. nvcc
# rejects the flag itself and forwards it with `-Xcompiler`, which reaches
# the preprocessor pass that resolves both the host and the device halves of
# a `.cu`.
#
# CUDA only. A `.hpp` here includes no `.cuh` and should not start: device
# text in a host translation unit is a layering mistake, and a build that
# quietly resolved it would be how the mistake survives review.
target_compile_options(pie_kernels_cuda PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-iquote,${PIE_JIT_DEVICE_HEADERS}>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-iquote,${PIE_JIT_SHIM_HEADERS}>)

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
# size". That throw is the ARCH PROBE, not an error: `pie_moe_cutlass_
# workspace_size` catches it (nothing may cross the C ABI) and reports a
# nonzero status, `fire::flashinfer_moe::workspace_bytes` turns that into 0,
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

# ---- Marlin: removed (§47) -----------------------------------------------
# `third_party/marlin` (500 KB) and `third_party/marlin_moe` (156 KB) were
# vendored here behind two options that defaulted APART -- the dense one OFF,
# the MoE one ON and therefore compiled on every build. Neither had a reachable
# caller.
#
#   * marlin_moe exported two functions and NOTHING in the tree called either.
#     Its stated purpose was to let `kernels_manifest.hpp` answer the
#     `native_mxfp4_moe` device capability -- but that answer required
#     PIE_CUDA_HAS_MARLIN too, which was OFF here, and downstream
#     `model-loader/src/plan.rs:194,:214` hard-code `native_mxfp4_moe: false`
#     in both constructors regardless.
#   * marlin exported five functions. Four had zero callers. The fifth,
#     `launch_gptq_gemm_w4a16_bf16`, was called from exactly one place --
#     `gemm.cpp`'s `w.dtype == INT4_PACKED` arm -- and no representation in
#     this driver can produce an INT4 weight: `WeightRepr` (model-compiler
#     `dsl.rs:92`) is `Bf16 | Scaled | Mxfp4Marlin`, and GPTQ/AWQ quant schemes
#     appear only in tests asserting they are refused.
#
# The live `gemm::act_x_wt_mxfp4_marlin` row is NOT affected and never used
# this tree: it sets MXFP4_PACKED and lands in `act_x_w`'s dequant-then-cuBLAS
# arm, which had no marlin `#ifdef`. The name is the only thing they shared.
#
# Gone with them: this section's two options and env overrides, the
# kernels.def/marlin.cu shape-drift reconciliation, `PIE_MARLIN_SHAPE`, and the
# `PIE_CUDA_HAS_MARLIN*` compile definitions.

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
mamba_sm90=$<BOOL:${PIE_CUDA_FLASHINFER_MAMBA_SM90}>
")

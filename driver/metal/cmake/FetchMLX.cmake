# cmake/FetchMLX.cmake — MLX C++ dependency for driver/metal.
#
# Gated behind the `PIE_METAL_WITH_MLX` option (see CMakeLists.txt). The
# bare foundation skeleton (entry + stub service) does NOT need MLX, so the
# option defaults OFF for a fast compile/link/register gate. The compute
# layer (beta's src/ops, src/executor) turns it ON.
#
# delta owns finalizing the exact MLX version / fetch strategy and any
# Accelerate/Metal build knobs.
#
# Two provisioning strategies, selected by `PIE_METAL_MLX_PROVIDER`:
#   * `fetch`  (default) — FetchContent the pinned MLX tag and build it from
#                source. Building MLX's Metal GPU kernels needs the `metal`
#                shader compiler (full Xcode / Metal Toolchain, NOT the Command
#                Line Tools); on a CLT-only host set
#                `PIE_METAL_MLX_BUILD_METAL=OFF` for a CPU-only compile+link.
#   * `system` — `find_package(MLX CONFIG)` against an already-installed MLX
#                (e.g. `brew install mlx`). The brew bottle ships MLX's GPU
#                kernels *precompiled* (`lib/mlx.metallib`) + headers + the
#                CMake config, so real Metal GPU works with NO Xcode and no
#                `xcrun metal`. This is the path for a real GPU build on a
#                CLT-only host.
#
# Both strategies end with the imported target named `mlx` (the brew config
# and the source build use the same target name) plus `PIE_METAL_HAS_MLX=1`,
# so the rest of CMakeLists is identical regardless of provider.

include_guard(GLOBAL)

# Pin a known-good MLX release (used by the `fetch` provider; also the version
# the brew `system` provider is expected to supply). Kept on a tag so the
# shared FetchContent cache stays warm.
#
# Pinned to v0.31.2 by @ingim ruling (see #mac): the real latest MLX (no 0.32
# exists on GitHub tags or PyPI). beta/charlie/delta validate their ops
# against this tag so the API matches at real compile.
set(PIE_MLX_GIT_TAG "v0.31.2" CACHE STRING "MLX git tag to fetch (fetch provider)")

# Provisioning strategy. `fetch` = build MLX from source via FetchContent;
# `system` = link a pre-installed MLX via find_package(MLX CONFIG).
set(PIE_METAL_MLX_PROVIDER "fetch"
    CACHE STRING "How to provide MLX: 'fetch' (source) or 'system' (find_package)")
set_property(CACHE PIE_METAL_MLX_PROVIDER PROPERTY STRINGS "fetch" "system")

# ---------------------------------------------------------------------------
# system provider: link a pre-installed MLX (brew / system CMake package).
# ---------------------------------------------------------------------------
function(pie_metal_link_system_mlx target)
  # Locate the install prefix. Honor an explicit override, else probe Homebrew
  # (`brew --prefix mlx`), else fall back to the default Apple-silicon prefix.
  set(_mlx_prefix "${PIE_METAL_MLX_SYSTEM_PREFIX}")
  if(NOT _mlx_prefix)
    find_program(PIE_BREW_EXECUTABLE brew)
    if(PIE_BREW_EXECUTABLE)
      execute_process(
        COMMAND ${PIE_BREW_EXECUTABLE} --prefix mlx
        OUTPUT_VARIABLE _mlx_prefix
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    endif()
  endif()
  if(NOT _mlx_prefix AND EXISTS "/opt/homebrew/opt/mlx")
    set(_mlx_prefix "/opt/homebrew/opt/mlx")
  endif()

  # The MLX CMake config lives at <prefix>/share/cmake/MLX/MLXConfig.cmake.
  find_package(MLX CONFIG REQUIRED
    PATHS "${_mlx_prefix}/share/cmake/MLX" "${_mlx_prefix}")

  target_link_libraries(${target} PUBLIC mlx)
  target_compile_definitions(${target} PUBLIC PIE_METAL_HAS_MLX=1)
  message(STATUS "driver/metal: linked system MLX ${MLX_VERSION} from ${_mlx_prefix}")
endfunction()

# ---------------------------------------------------------------------------
# fetch provider: build MLX from source via FetchContent.
# ---------------------------------------------------------------------------
function(pie_metal_fetch_mlx_source target)
  include(FetchContent)

  # MLX builds its own Metal kernels; keep tests/examples/python off.
  set(MLX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(MLX_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(MLX_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
  set(MLX_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)

  # Metal GPU backend. Compiling MLX's Metal kernels requires the `metal`
  # shader compiler (full Xcode / Metal Toolchain — NOT the Command Line
  # Tools). On a CLT-only host (`xcrun -find metal` fails), set
  # PIE_METAL_MLX_BUILD_METAL=OFF to build MLX CPU-only (its `no_metal`
  # backend stubs the GPU symbols so the whole driver still compiles+links
  # against the real MLX API surface). For a real GPU build on a CLT-only
  # host, prefer PIE_METAL_MLX_PROVIDER=system (prebuilt brew MLX) instead.
  option(PIE_METAL_MLX_BUILD_METAL "Build MLX's Metal GPU backend (needs xcrun metal)" ON)
  set(MLX_BUILD_METAL ${PIE_METAL_MLX_BUILD_METAL} CACHE BOOL "" FORCE)

  FetchContent_Declare(
    mlx
    GIT_REPOSITORY https://github.com/ml-explore/mlx.git
    GIT_TAG ${PIE_MLX_GIT_TAG}
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(mlx)

  target_link_libraries(${target} PUBLIC mlx)
  target_compile_definitions(${target} PUBLIC PIE_METAL_HAS_MLX=1)
  message(STATUS "driver/metal: built+linked MLX ${PIE_MLX_GIT_TAG} from source")
endfunction()

# Dispatch entry point used by CMakeLists.
function(pie_metal_fetch_mlx target)
  if(PIE_METAL_MLX_PROVIDER STREQUAL "system")
    pie_metal_link_system_mlx(${target})
  elseif(PIE_METAL_MLX_PROVIDER STREQUAL "fetch")
    pie_metal_fetch_mlx_source(${target})
  else()
    message(FATAL_ERROR
      "driver/metal: invalid PIE_METAL_MLX_PROVIDER='${PIE_METAL_MLX_PROVIDER}' "
      "(expected 'fetch' or 'system')")
  endif()
endfunction()

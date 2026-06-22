# cmake/FetchMLX.cmake — MLX C++ dependency for driver/metal.
#
# Gated behind the `PIE_METAL_WITH_MLX` option (see CMakeLists.txt). The
# bare foundation skeleton (entry + stub service) does NOT need MLX, so the
# option defaults OFF for a fast compile/link/register gate. The compute
# layer (beta's src/ops, src/executor) turns it ON.
#
# delta owns finalizing the exact MLX version / fetch strategy and any
# Accelerate/Metal build knobs; this is a working default fetch.

include_guard(GLOBAL)

function(pie_metal_fetch_mlx target)
  include(FetchContent)

  # Pin a known-good MLX release. delta may bump this; keep it on a tag so
  # the shared CPM/FetchContent cache on the build host stays warm.
  #
  # Pinned to v0.31.2 by @ingim ruling (see #mac): the real latest MLX
  # (no 0.32 exists on GitHub tags or PyPI). beta/charlie/delta validate
  # their ops against this same tag so the API matches at real compile.
  set(PIE_MLX_GIT_TAG "v0.31.2" CACHE STRING "MLX git tag to fetch")

  # MLX builds its own Metal kernels; keep tests/examples/python off.
  set(MLX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(MLX_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(MLX_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
  set(MLX_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)

  # Metal GPU backend. Compiling MLX's Metal kernels requires the `metal`
  # shader compiler, which ships ONLY with full Xcode (or the standalone
  # Metal Toolchain) — NOT the Command Line Tools. On a CLT-only host
  # (`xcrun -find metal` fails), set PIE_METAL_MLX_BUILD_METAL=OFF to build
  # MLX CPU-only (its `no_metal` backend stubs the GPU symbols so the whole
  # driver still compiles+links against the real MLX v0.29.1 API surface —
  # GPU execution then requires installing the Metal Toolchain). Defaults ON
  # for real GPU builds on a fully-provisioned host.
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
  message(STATUS "driver/metal: linked MLX ${PIE_MLX_GIT_TAG}")
endfunction()

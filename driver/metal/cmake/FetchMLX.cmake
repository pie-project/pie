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
  # NOTE: v0.31.2 is the latest MLX release on both GitHub tags and PyPI
  # (there is no 0.32 — that version does not exist anywhere). beta
  # validated ops against the newest installed headers, which resolve to
  # 0.31.2. Keep this matched to beta's actual installed <mlx/mlx.h> so
  # the ops type-check against the same API at real compile time.
  set(PIE_MLX_GIT_TAG "v0.31.2" CACHE STRING "MLX git tag to fetch")

  # MLX builds its own Metal kernels; keep tests/examples/python off.
  set(MLX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(MLX_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(MLX_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
  set(MLX_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
  set(MLX_BUILD_METAL ON CACHE BOOL "" FORCE)

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

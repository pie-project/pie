# Loader smoke test — delta-owned. Linked against the driver lib + MLX so it
# exercises the real load → bind → graph → KV → forward path on the GPU.
# Off by default; enable with -DPIE_METAL_BUILD_TESTS=ON.

add_executable(smoke_load smoke_load.cpp)
target_link_libraries(smoke_load PRIVATE pie_driver_metal_lib)

# Loader smoke test — delta-owned. Linked against the driver lib + MLX so it
# exercises the real load → bind → graph → KV → forward path on the GPU.
# Off by default; enable with -DPIE_METAL_BUILD_TESTS=ON.

add_executable(smoke_load smoke_load.cpp)
target_link_libraries(smoke_load PRIVATE pie_driver_metal_lib)

# In-process single-stream decode benchmark (beta) — the driver-vs-runtime
# isolator: bare graph.forward + greedy-sample loop, no server/IPC/scheduler.
add_executable(decode_bench decode_bench.cpp)
target_link_libraries(decode_bench PRIVATE pie_driver_metal_lib)

# Tests for the metal driver (built when PIE_METAL_BUILD_TESTS=ON).

add_executable(smoke_load smoke_load.cpp)
target_link_libraries(smoke_load PRIVATE pie_driver_metal_lib)

# In-process single-stream decode benchmark (beta) — the driver-vs-runtime
# isolator: bare graph.forward + greedy-sample loop, no server/IPC/scheduler.
add_executable(decode_bench decode_bench.cpp)
target_link_libraries(decode_bench PRIVATE pie_driver_metal_lib)

add_executable(kv_perlayer_test kv_perlayer_test.cpp)
target_link_libraries(kv_perlayer_test PRIVATE pie_driver_metal_lib)

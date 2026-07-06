# Tests for the metal driver (built when PIE_METAL_BUILD_TESTS=ON).

add_executable(smoke_load smoke_load.cpp)
target_link_libraries(smoke_load PRIVATE pie_driver_metal_lib)

# In-process single-stream decode benchmark (beta) — the driver-vs-runtime
# isolator: bare graph.forward + greedy-sample loop, no server/IPC/scheduler.
add_executable(decode_bench decode_bench.cpp)
target_link_libraries(decode_bench PRIVATE pie_driver_metal_lib)

add_executable(kv_perlayer_test kv_perlayer_test.cpp)
target_link_libraries(kv_perlayer_test PRIVATE pie_driver_metal_lib)

# Real MTP speculative decoding on Metal (Qwen3.5-0.8B GDN backbone + MTP head).
add_executable(qwen35_mtp qwen35_mtp.cpp)
target_link_libraries(qwen35_mtp PRIVATE pie_driver_metal_lib)

# Honest wall-clock benchmark: regular greedy vs MTP (K=1) spec-decode.
add_executable(qwen35_mtp_bench qwen35_mtp_bench.cpp)
target_link_libraries(qwen35_mtp_bench PRIVATE pie_driver_metal_lib)

# High-concurrency throughput sweep: regular batched vs MTP spec-decode.
add_executable(qwen35_tput_bench qwen35_tput_bench.cpp)
target_link_libraries(qwen35_tput_bench PRIVATE pie_driver_metal_lib)

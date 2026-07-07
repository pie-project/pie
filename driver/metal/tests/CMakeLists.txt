# Tests for the metal driver (built when PIE_METAL_BUILD_TESTS=ON).

add_executable(smoke_load smoke_load.cpp)
target_link_libraries(smoke_load PRIVATE pie_driver_metal_lib)

# In-process single-stream decode benchmark (beta) — the driver-vs-runtime
# isolator: bare graph.forward + greedy-sample loop, no server/IPC/scheduler.
add_executable(decode_bench decode_bench.cpp)
target_link_libraries(decode_bench PRIVATE pie_driver_metal_lib)

add_executable(kv_perlayer_test kv_perlayer_test.cpp)
target_link_libraries(kv_perlayer_test PRIVATE pie_driver_metal_lib)
# Double-buffered (async_eval) pipelined decode executor — Metal vLLM-style
# submit-N+1-before-collect-N; token-identity vs the synchronous path.
add_executable(pipelined_executor pipelined_executor.cpp)
target_link_libraries(pipelined_executor PRIVATE pie_driver_metal_lib)

# End-to-end deferred-response serve loop — the executor double-buffer wired
# through the production InProcServer path (build_request_view → deferred branch
# → defer_send_ → completion thread → collect → send_response); token-identity
# vs the synchronous run_forward path + batched throughput.
add_executable(pipelined_service pipelined_service.cpp)
target_link_libraries(pipelined_service PRIVATE pie_driver_metal_lib)

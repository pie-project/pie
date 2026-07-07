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

# General decode-path throughput (greedy/temp/top-k/top-p/min-p/grammar) under concurrency.
add_executable(qwen35_decode_tput qwen35_decode_tput.cpp)
target_link_libraries(qwen35_decode_tput PRIVATE pie_driver_metal_lib)

# Device-resident MTP spec-decode MACHINERY (std-attention, dependency-clean) —
# the Metal dual of the CUDA fully-pipelined MTP: on-device verify DAG + [K+1]
# window mx::array carrier + back-to-back async_eval; committed==greedy gate.
add_executable(mtp_device_resident mtp_device_resident.cpp)
target_link_libraries(mtp_device_resident PRIVATE pie_driver_metal_lib)

# GDN frozen-checkpointed scan — the core of the GDN spec-decode rewind:
# checkpoint m == clean sub-forward of the first m tokens (lossless commit-advance).
add_executable(gdn_checkpoint_test gdn_checkpoint_test.cpp)
target_link_libraries(gdn_checkpoint_test PRIVATE pie_driver_metal_lib)

# GDN rewind on the REAL Qwen3.5 backbone: speculative window forward + snapshot/
# restore rewind recovers the greedy single-step state (S_fixed==S_ref) exactly.
add_executable(qwen35_gdn_rewind qwen35_gdn_rewind.cpp)
target_link_libraries(qwen35_gdn_rewind PRIVATE pie_driver_metal_lib)

# REAL Qwen3.5 MTP spec-decode + GDN rewind: full autoregressive window loop
# (real MTP head drafter + verify + rewind-on-reject); committed==greedy bit-exact.
add_executable(qwen35_mtp_specdecode qwen35_mtp_specdecode.cpp)
target_link_libraries(qwen35_mtp_specdecode PRIVATE pie_driver_metal_lib)

function(add_rawmetal_tool target)
  add_executable(${target} ${ARGN})
  target_link_libraries(${target} PRIVATE pie_driver_metal_lib)
  target_compile_definitions(${target} PRIVATE
    PIE_METAL_TOOL_KERNELS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../src/kernels")
endfunction()

add_rawmetal_tool(decoder_smoke decoder_smoke.cpp)
add_rawmetal_tool(rawmetal_harness harness_main.cpp harness.cpp)
add_rawmetal_tool(rawmetal_microbench bench_kernels.cpp harness.cpp)
add_rawmetal_tool(sparse_probe sparse_probe.mm)
add_rawmetal_tool(sparse_probe2 sparse_probe2.mm)
add_rawmetal_tool(sparse_probe3 sparse_probe3.mm)

# `st_probe` and `gemma4_decode_run` were removed with §6: both were bring-up
# harnesses built on `SafetensorsView`, the driver-side name-addressed
# checkpoint reader. The plan now carries the byte offsets, so nothing in a
# driver resolves a tensor by name any more and the capability they were written
# against no longer exists.
add_rawmetal_tool(
  gemma4_encode_probe
  gemma4_encode_probe.cpp
  gemma4_encode.cpp)

add_executable(dvfs_probe dvfs_probe.mm)
set_source_files_properties(dvfs_probe.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
target_link_libraries(dvfs_probe PRIVATE
  "-framework Metal"
  "-framework Foundation")

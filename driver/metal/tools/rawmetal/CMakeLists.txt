function(add_rawmetal_tool target)
  add_executable(${target} ${ARGN})
  target_link_libraries(${target} PRIVATE pie_driver_metal_lib)
  target_compile_definitions(${target} PRIVATE
    PIE_METAL_TOOL_KERNELS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../src/kernels")
endfunction()

add_rawmetal_tool(decode_run decode_run.cpp)
add_rawmetal_tool(decoder_smoke decoder_smoke.cpp)
add_rawmetal_tool(rawmetal_harness harness_main.cpp harness.cpp)
add_rawmetal_tool(rawmetal_microbench bench_kernels.cpp harness.cpp)
add_rawmetal_tool(heap_bind_probe heap_bind_probe.cpp)
add_rawmetal_tool(st_probe st_probe.cpp)

add_rawmetal_tool(
  gemma4_decode_run
  gemma4_decode_run.cpp
  gemma4_encode.cpp
  gemma4_heap_bind.cpp
  gemma4_consts.cpp)
add_rawmetal_tool(
  gemma4_encode_probe
  gemma4_encode_probe.cpp
  gemma4_encode.cpp)

add_executable(dvfs_probe dvfs_probe.mm)
set_source_files_properties(dvfs_probe.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
target_link_libraries(dvfs_probe PRIVATE
  "-framework Metal"
  "-framework Foundation")

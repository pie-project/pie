cmake_minimum_required(VERSION 3.20)
# Standalone Phase-0 raw-Metal scaffolding (NOT wired into the main driver build yet).
# Builds a single harness executable: Metal-4 wrappers + latency/micro-bench rig.
project(pie_raw_metal LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ── Phase-0 shader strategy: RUNTIME MSL compile (newLibraryWithSource:) ──
# Decided in #mac (manager + delta): the offline `metal`/`metallib` toolchain is NOT on
# the box (CommandLineTools only; full Xcode + MetalToolchain absent). Runtime compile
# sidesteps it entirely and keeps de-risk iteration fast. The AOT .metal->.metallib path
# is the PRODUCTION option, DEFERRED to Phase 1 (needs `xcodebuild -downloadComponent
# MetalToolchain`). Do NOT wire AOT for the prototype.
option(PIE_RAW_METAL_AOT "AOT-compile .metal -> .metallib (Phase-1/production; needs Metal Toolchain)" OFF)

add_executable(raw_metal_harness
  harness_main.cpp
  harness.cpp
  mtl4_context.mm
)

# ARC only for the Obj-C++ translation unit.
set_source_files_properties(mtl4_context.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")

target_include_directories(raw_metal_harness PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(raw_metal_harness PRIVATE
  RAW_METAL_KERNELS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/kernels")

target_link_libraries(raw_metal_harness PRIVATE
  "-framework Metal"
  "-framework Foundation")

# nlohmann/json for the safetensors header parse (reuse the driver's fetched copy).
find_path(NLOHMANN_JSON_INCLUDE nlohmann/json.hpp
  PATHS
    ${CMAKE_CURRENT_SOURCE_DIR}/../../build-gpu/_deps/json-src/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../build-nomlx/_deps/json-src/include
    /opt/homebrew/include
  NO_DEFAULT_PATH)

# Micro-bench for delta's real ported kernels (first gpu-exec-ms datapoint).
add_executable(raw_metal_microbench
  bench_kernels.cpp
  harness.cpp
  mtl4_context.mm
)
set_source_files_properties(mtl4_context.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
target_include_directories(raw_metal_microbench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(raw_metal_microbench PRIVATE
  RAW_METAL_KERNELS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/kernels")
target_link_libraries(raw_metal_microbench PRIVATE
  "-framework Metal"
  "-framework Foundation")

# Raw safetensors byte reader (MLX-free) + probe — the weight-staging handoff for
# delta's heap_bind. Pure C++/POSIX, no Metal.
if(NLOHMANN_JSON_INCLUDE)
  add_executable(st_probe st_probe.cpp safetensors_view.cpp)
  target_include_directories(st_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${NLOHMANN_JSON_INCLUDE})
else()
  message(WARNING "nlohmann/json.hpp not found — st_probe (safetensors handoff) skipped. "
                  "Set -DNLOHMANN_JSON_INCLUDE=<dir>.")
endif()

if(PIE_RAW_METAL_AOT)
  message(WARNING "PIE_RAW_METAL_AOT is a Phase-1 stub: requires the Metal Toolchain "
                  "component (xcodebuild -downloadComponent MetalToolchain). Phase 0 uses "
                  "runtime newLibraryWithSource: — AOT is intentionally not implemented yet.")
endif()

cmake_minimum_required(VERSION 3.23)

# driver/metal — MLX + native Metal backend for Pie (Apple Silicon, macOS).
# Structure mirrors driver/cuda. OBJCXX is enabled so the compute layer can
# add `.mm` (Objective-C++) translation units for Metal interop.
project(PIE_DRIVER_METAL CXX OBJCXX)

if(NOT APPLE)
  message(FATAL_ERROR "driver/metal targets macOS / Apple Silicon only")
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

find_package(Threads REQUIRED)

# Schema + in-proc IPC headers. The worker's build.rs passes these in via
# -D...; standalone driver builds fall back to the in-tree paths.
if(NOT PIE_SCHEMA_INCLUDE_DIR)
  set(PIE_SCHEMA_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../protocol/schema/include")
endif()
if(NOT PIE_IPC_INCLUDE_DIR)
  set(PIE_IPC_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../ipc/include")
endif()

# Optional MLX dependency. OFF by default so the foundation skeleton
# (entry + stub service) compiles/links fast; the compute layer flips it ON
# (see cmake/FetchMLX.cmake). The worker build.rs forwards
# -DPIE_METAL_WITH_MLX from the PIE_METAL_WITH_MLX env var.
option(PIE_METAL_WITH_MLX "Fetch and link MLX C++" OFF)

include(cmake/CPM.cmake)

# Use a persistent CPM cache when the env var is set so incremental builds
# don't re-clone from GitHub.
if(DEFINED ENV{CPM_SOURCE_CACHE} AND NOT CPM_SOURCE_CACHE)
  set(CPM_SOURCE_CACHE $ENV{CPM_SOURCE_CACHE})
endif()

cpmaddpackage(URI "gh:marzer/tomlplusplus@3.4.0")
cpmaddpackage(URI "gh:CLIUtils/CLI11@2.5.0")
cpmaddpackage(URI "gh:nlohmann/json@3.12.0" OPTIONS "JSON_BuildTests OFF")

# -----------------------------------------------------------------------------
# Sources
#
# Explicit (alpha-owned): entry, driver startup, the stub in-proc service.
# GLOBbed (beta/charlie/delta-owned): every .cpp/.mm under the compute,
# executor, kernels, model, and loader trees — so peers add translation
# units WITHOUT editing this file. CONFIGURE_DEPENDS re-globs on rebuild.
# -----------------------------------------------------------------------------
set(PIE_METAL_EXPLICIT_SOURCES
  src/entry.cpp
  src/driver_startup.cpp
  src/service/inproc_service.cpp)

file(GLOB_RECURSE PIE_METAL_GLOB_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/ops/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/ops/*.mm
  ${CMAKE_CURRENT_SOURCE_DIR}/src/executor/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/executor/*.mm
  ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/*.mm
  ${CMAKE_CURRENT_SOURCE_DIR}/src/model/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/model/*.mm
  ${CMAKE_CURRENT_SOURCE_DIR}/src/loader/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/loader/*.mm)

add_library(pie_driver_metal_lib STATIC
  ${PIE_METAL_EXPLICIT_SOURCES}
  ${PIE_METAL_GLOB_SOURCES})

target_include_directories(pie_driver_metal_lib
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/include
    ${PIE_SCHEMA_INCLUDE_DIR}
    ${PIE_IPC_INCLUDE_DIR})

target_compile_options(pie_driver_metal_lib PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wno-unused-parameter>
  $<$<COMPILE_LANGUAGE:OBJCXX>:-Wall -Wextra -Wno-unused-parameter>)

target_link_libraries(pie_driver_metal_lib PUBLIC
  CLI11::CLI11
  tomlplusplus::tomlplusplus
  nlohmann_json::nlohmann_json
  Threads::Threads)

# Apple frameworks MLX + native Metal shaders pull. Linked PUBLIC so both
# the standalone exe and the worker binary that embeds the static lib
# resolve them. (The worker build.rs also adds these via rustc-link-arg;
# duplicate -framework flags are harmless.)
target_link_libraries(pie_driver_metal_lib PUBLIC
  "-framework Metal"
  "-framework MetalKit"
  "-framework Foundation"
  "-framework Accelerate")

if(PIE_METAL_WITH_MLX)
  include(cmake/FetchMLX.cmake)
  pie_metal_fetch_mlx(pie_driver_metal_lib)
endif()

# -----------------------------------------------------------------------------
# Native Metal shader compilation (optional, AOT)
#
# Default path is MLX JIT (`mlx::core::fast::metal_kernel`), where kernel
# source lives as a string in a `.cpp` and any `.metal` file is just a
# tooling mirror — nothing to compile here. Set -DPIE_METAL_BUILD_METALLIB=ON
# to instead AOT-compile every standalone-compilable `src/kernels/*.metal`
# into a default.metallib next to the binary.
# -----------------------------------------------------------------------------
option(PIE_METAL_BUILD_METALLIB "AOT-compile src/kernels/*.metal into default.metallib" OFF)

file(GLOB_RECURSE PIE_METAL_SHADER_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/*.metal)

if(PIE_METAL_BUILD_METALLIB AND PIE_METAL_SHADER_SOURCES)
  find_program(XCRUN_EXECUTABLE xcrun REQUIRED)
  set(PIE_METAL_AIR_FILES "")
  foreach(shader ${PIE_METAL_SHADER_SOURCES})
    get_filename_component(shader_name ${shader} NAME_WE)
    set(air_file ${CMAKE_BINARY_DIR}/metal/${shader_name}.air)
    add_custom_command(
      OUTPUT ${air_file}
      COMMAND ${XCRUN_EXECUTABLE} -sdk macosx metal -c ${shader} -o ${air_file}
      DEPENDS ${shader}
      COMMENT "Compiling Metal shader ${shader_name}.metal")
    list(APPEND PIE_METAL_AIR_FILES ${air_file})
  endforeach()
  set(PIE_METAL_METALLIB ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib)
  add_custom_command(
    OUTPUT ${PIE_METAL_METALLIB}
    COMMAND ${XCRUN_EXECUTABLE} -sdk macosx metallib ${PIE_METAL_AIR_FILES}
            -o ${PIE_METAL_METALLIB}
    DEPENDS ${PIE_METAL_AIR_FILES}
    COMMENT "Linking default.metallib")
  add_custom_target(pie_driver_metal_shaders ALL DEPENDS ${PIE_METAL_METALLIB})
  add_dependencies(pie_driver_metal_lib pie_driver_metal_shaders)
endif()

# -----------------------------------------------------------------------------
# Standalone executable — thin shim around the lib's entry point. Consumed
# for build/config self-checks and local driver development.
# -----------------------------------------------------------------------------
add_executable(pie_driver_metal src/main.cpp)
target_link_libraries(pie_driver_metal PRIVATE pie_driver_metal_lib)

# The `--features driver-metal` build in worker/build.rs consumes
# `pie_driver_metal_lib` directly.

# -----------------------------------------------------------------------------
# Loader smoke test (delta-owned, off by default). Builds an end-to-end loader→
# graph→KV→forward check; requires MLX. Enable with -DPIE_METAL_BUILD_TESTS=ON.
# -----------------------------------------------------------------------------
option(PIE_METAL_BUILD_TESTS "Build driver/metal loader smoke tests" OFF)
if(PIE_METAL_BUILD_TESTS AND PIE_METAL_WITH_MLX)
  add_subdirectory(tests)
endif()

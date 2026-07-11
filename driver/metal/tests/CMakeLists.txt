# MLX reference implementation and its consumers.

add_subdirectory(mlx)

if(PIE_METAL_BUILD_TESTS)
  add_executable(smoke_load smoke_load.cpp)
  target_link_libraries(smoke_load PRIVATE pie_metal_mlx_test_support)

  add_executable(kv_perlayer_test kv_perlayer_test.cpp)
  target_link_libraries(kv_perlayer_test PRIVATE pie_metal_mlx_test_support)
  add_test(NAME kv_perlayer_test COMMAND kv_perlayer_test)
endif()

if(PIE_METAL_BUILD_PARITY)
  add_subdirectory(parity)
endif()

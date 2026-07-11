# Parity harness (alpha-owned, off by default). Builds `parity_driver`, which
# dumps the driver's last-position logits for an exact token-id prompt so
# tests/parity/parity_check.py can diff them against mlx-lm (the accuracy gate).
# Requires MLX. Enable with -DPIE_METAL_BUILD_PARITY=ON.

add_executable(parity_driver parity_driver.cpp)
target_link_libraries(parity_driver PRIVATE pie_metal_mlx_test_support)

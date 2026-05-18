#pragma once

#include "model.hpp"

namespace pie_portable_driver {

// Canonical Rust storage-program loader for the portable driver. This is
// backend-agnostic over ggml tensors, so CPU/CUDA/Metal/Vulkan weight loading
// all enter through the same compiled storage program.
void load_with_rust_storage_program(Model& model, const char* planner_mode);

}  // namespace pie_portable_driver

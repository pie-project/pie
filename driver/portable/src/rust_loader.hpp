#pragma once

#include "model.hpp"

namespace pie_portable_driver {

// Opt-in Rust storage-program loader for the portable driver.
//
// Returns true when the Rust program loaded every declared tensor. Returns
// false for cpp/dual modes when C++ loading should continue. Throws in rust
// mode if coverage is incomplete or an executable instruction is unsupported.
bool try_load_with_rust_storage_program(Model& model, const char* planner_mode);

}  // namespace pie_portable_driver

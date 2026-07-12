#pragma once

#include <span>

#include "pie_native/storage_program.hpp"

namespace pie_cuda_driver {

using RustStorageProgram = pie_weight_loader::StorageProgram;

inline RustStorageProgram deserialize_rust_storage_program(
    std::span<const std::uint8_t> bytes,
    std::uint64_t expected_compiler_version) {
    return RustStorageProgram::deserialize(bytes, expected_compiler_version);
}

}  // namespace pie_cuda_driver

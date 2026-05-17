#pragma once

#include <memory>
#include <vector>

#include "loader/storage_program.hpp"

namespace pie_cuda_driver {

class CheckpointByteSource;

struct StorageWriteDestination {
    const ExtentWrite* write = nullptr;
    void* dst_base = nullptr;
};

class StorageWriteExecutor {
public:
    virtual ~StorageWriteExecutor() = default;

    virtual const char* name() const noexcept = 0;
    virtual void execute(
        const std::vector<StorageWriteDestination>& destinations) = 0;
};

std::unique_ptr<StorageWriteExecutor> make_storage_write_executor(
    CheckpointByteSource& byte_source);

}  // namespace pie_cuda_driver

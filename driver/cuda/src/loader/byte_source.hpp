#pragma once

#include "loader/physical_load_plan.hpp"
#include "loader/safetensors.hpp"

namespace pie_cuda_driver {

// Physical IO backend for checkpoint bytes. Implementations decide whether
// writes come from mmap+cudaMemcpy, GPUDirect Storage, or a test fixture.
class CheckpointByteSource {
public:
    virtual ~CheckpointByteSource() = default;

    virtual void write_to_device(
        const ByteRangeWrite& write,
        void* dst_base) = 0;
};

class MmapByteSource final : public CheckpointByteSource {
public:
    explicit MmapByteSource(SafetensorsLoader& loader) noexcept
        : loader_(loader) {}

    void write_to_device(
        const ByteRangeWrite& write,
        void* dst_base) override;

private:
    SafetensorsLoader& loader_;
};

}  // namespace pie_cuda_driver

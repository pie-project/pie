#pragma once

#include <memory>
#include <string>

#include "loader/storage_program.hpp"
#include "loader/safetensors.hpp"

namespace pie_cuda_driver {

enum class CheckpointIoPolicy {
    Auto,
    Mmap,
    Gds,
};

CheckpointIoPolicy parse_checkpoint_io_policy(const std::string& value);
const char* checkpoint_io_policy_name(CheckpointIoPolicy policy) noexcept;

// Storage IO backend for checkpoint bytes. Implementations decide whether
// writes come from mmap+cudaMemcpy, GPUDirect Storage, or a test fixture.
class CheckpointByteSource {
public:
    virtual ~CheckpointByteSource() = default;

    virtual const char* name() const noexcept = 0;

    virtual void write_to_device(
        const ExtentWrite& write,
        void* dst_base) = 0;
    virtual bool supports_async_writes() const noexcept { return false; }
    virtual void write_to_device_async(
        const ExtentWrite& write,
        void* dst_base,
        void* stream);
};

class MmapByteSource final : public CheckpointByteSource {
public:
    explicit MmapByteSource(SafetensorsCheckpointSource& loader) noexcept
        : loader_(loader) {}

    const char* name() const noexcept override { return "mmap"; }

    void write_to_device(
        const ExtentWrite& write,
        void* dst_base) override;
    bool supports_async_writes() const noexcept override { return true; }
    void write_to_device_async(
        const ExtentWrite& write,
        void* dst_base,
        void* stream) override;

private:
    SafetensorsCheckpointSource& loader_;
};

class GdsByteSource final : public CheckpointByteSource {
public:
    GdsByteSource(SafetensorsCheckpointSource& loader, bool required, bool verbose);
    ~GdsByteSource() override;

    GdsByteSource(const GdsByteSource&) = delete;
    GdsByteSource& operator=(const GdsByteSource&) = delete;

    const char* name() const noexcept override {
        return direct_enabled_ ? "gds" : "mmap";
    }

    void write_to_device(
        const ExtentWrite& write,
        void* dst_base) override;
    bool supports_async_writes() const noexcept override;
    void write_to_device_async(
        const ExtentWrite& write,
        void* dst_base,
        void* stream) override;

    bool direct_enabled() const noexcept { return direct_enabled_; }

private:
    class Impl;

    SafetensorsCheckpointSource& loader_;
    MmapByteSource fallback_;
    std::unique_ptr<Impl> impl_;
    bool required_ = false;
    bool verbose_ = false;
    bool direct_enabled_ = false;
};

std::unique_ptr<CheckpointByteSource> make_checkpoint_byte_source(
    CheckpointIoPolicy policy,
    SafetensorsCheckpointSource& loader,
    bool verbose);

}  // namespace pie_cuda_driver

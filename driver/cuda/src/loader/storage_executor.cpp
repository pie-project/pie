#include "loader/storage_executor.hpp"

#include <algorithm>
#include <stdexcept>

#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "loader/byte_source.hpp"

namespace pie_cuda_driver {

namespace {

class SyncStorageWriteExecutor final : public StorageWriteExecutor {
public:
    explicit SyncStorageWriteExecutor(CheckpointByteSource& byte_source) noexcept
        : byte_source_(byte_source) {}

    const char* name() const noexcept override { return "sync"; }

    void execute(
        const std::vector<StorageWriteDestination>& destinations) override
    {
        for (const auto& destination : destinations) {
            if (destination.write == nullptr || destination.dst_base == nullptr) {
                throw std::runtime_error(
                    "storage executor: extent write destination is incomplete");
            }
            byte_source_.write_to_device(
                *destination.write, destination.dst_base);
        }
    }

private:
    CheckpointByteSource& byte_source_;
};

class PipelinedMmapWriteExecutor final : public StorageWriteExecutor {
public:
    explicit PipelinedMmapWriteExecutor(
        CheckpointByteSource& byte_source,
        int stream_count = 4)
        : byte_source_(byte_source)
    {
        stream_count = std::max(stream_count, 1);
        streams_.resize(static_cast<std::size_t>(stream_count));
        for (auto& stream : streams_) {
            CUDA_CHECK(cudaStreamCreateWithFlags(
                &stream, cudaStreamNonBlocking));
        }
    }

    ~PipelinedMmapWriteExecutor() override
    {
        for (auto stream : streams_) {
            if (stream != nullptr) {
                cudaStreamDestroy(stream);
            }
        }
    }

    PipelinedMmapWriteExecutor(const PipelinedMmapWriteExecutor&) = delete;
    PipelinedMmapWriteExecutor& operator=(
        const PipelinedMmapWriteExecutor&) = delete;

    const char* name() const noexcept override { return "pipelined-mmap"; }

    void execute(
        const std::vector<StorageWriteDestination>& destinations) override
    {
        if (destinations.empty()) return;
        for (std::size_t i = 0; i < destinations.size(); ++i) {
            const auto& destination = destinations[i];
            if (destination.write == nullptr || destination.dst_base == nullptr) {
                throw std::runtime_error(
                    "storage executor: extent write destination is incomplete");
            }
            cudaStream_t stream = streams_[i % streams_.size()];
            byte_source_.write_to_device_async(
                *destination.write, destination.dst_base, stream);
        }
        for (auto stream : streams_) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    }

private:
    CheckpointByteSource& byte_source_;
    std::vector<cudaStream_t> streams_;
};

}  // namespace

std::unique_ptr<StorageWriteExecutor> make_storage_write_executor(
    CheckpointByteSource& byte_source)
{
    if (byte_source.supports_async_writes()) {
        return std::make_unique<PipelinedMmapWriteExecutor>(byte_source);
    }
    return std::make_unique<SyncStorageWriteExecutor>(byte_source);
}

}  // namespace pie_cuda_driver

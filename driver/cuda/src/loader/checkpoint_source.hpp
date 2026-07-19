#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace pie_cuda_driver {

class CheckpointSource {
  public:
    static CheckpointSource open(const std::filesystem::path& snapshot_dir);

    CheckpointSource() = default;
    ~CheckpointSource();
    CheckpointSource(const CheckpointSource&) = delete;
    CheckpointSource& operator=(const CheckpointSource&) = delete;
    CheckpointSource(CheckpointSource&&) noexcept = default;
    CheckpointSource& operator=(CheckpointSource&&) noexcept = default;

    std::size_t file_count() const noexcept { return files_.size(); }
    void copy_storage_bytes_to_device(
        std::uint32_t file_id,
        std::uint64_t file_offset,
        std::uint64_t span_bytes,
        void* dst);
    void copy_storage_bytes_to_device_async(
        std::uint32_t file_id,
        std::uint64_t file_offset,
        std::uint64_t span_bytes,
        void* dst,
        void* stream);
    const std::uint8_t* storage_host_ptr(
        std::uint32_t file_id,
        std::uint64_t file_offset,
        std::uint64_t span_bytes);
    void read_storage_bytes_to_host(
        std::uint32_t file_id,
        std::uint64_t file_offset,
        std::uint64_t span_bytes,
        void* dst);

  private:
    struct File {
        std::filesystem::path path;
        int fd = -1;
        std::size_t mapped_size = 0;
        const std::uint8_t* data = nullptr;
    };

    void open_file_(File& file) const;
    void map_file_(File& file) const;
    File& checked_file_(
        std::uint32_t file_id,
        std::uint64_t file_offset,
        std::uint64_t span_bytes);

    std::vector<File> files_;
};

}  // namespace pie_cuda_driver

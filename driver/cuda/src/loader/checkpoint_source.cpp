#include "loader/checkpoint_source.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <limits>
#include <stdexcept>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "loader/safetensors_manifest.hpp"

namespace pie_cuda_driver {

CheckpointSource CheckpointSource::open(
    const std::filesystem::path& snapshot_dir) {
    CheckpointSource source;
    const auto manifest = pie_driver_common::discover_safetensors_manifest(
        snapshot_dir,
        pie_driver_common::SafetensorsLayoutPreference::SingleFile);
    source.files_.reserve(manifest.shard_paths.size());
    for (const auto& path : manifest.shard_paths) {
        File file;
        file.path = path;
        source.files_.push_back(std::move(file));
    }
    return source;
}

CheckpointSource::~CheckpointSource() {
    for (auto& file : files_) {
        if (file.data != nullptr && file.mapped_size != 0) {
            ::munmap(const_cast<std::uint8_t*>(file.data), file.mapped_size);
        }
        if (file.fd >= 0) ::close(file.fd);
    }
}

void CheckpointSource::open_file_(File& file) const {
    if (file.fd >= 0) return;
    file.fd = ::open(file.path.c_str(), O_RDONLY);
    if (file.fd < 0) {
        throw std::runtime_error(
            "open(" + file.path.string() + ") failed: " + std::strerror(errno));
    }
    struct stat stat {};
    if (::fstat(file.fd, &stat) != 0) {
        ::close(file.fd);
        file.fd = -1;
        throw std::runtime_error("fstat(" + file.path.string() + ") failed");
    }
    file.mapped_size = static_cast<std::size_t>(stat.st_size);
}

void CheckpointSource::map_file_(File& file) const {
    if (file.data != nullptr) return;
    open_file_(file);
    void* mapping = ::mmap(
        nullptr, file.mapped_size, PROT_READ, MAP_SHARED, file.fd, 0);
    if (mapping == MAP_FAILED) {
        throw std::runtime_error(
            std::string("checkpoint mmap failed: ") + std::strerror(errno));
    }
    file.data = static_cast<const std::uint8_t*>(mapping);
    (void)::madvise(mapping, file.mapped_size, MADV_SEQUENTIAL);
}

CheckpointSource::File& CheckpointSource::checked_file_(
    std::uint32_t file_id,
    std::uint64_t file_offset,
    std::uint64_t span_bytes) {
    if (file_id >= files_.size()) {
        throw std::runtime_error("checkpoint file id is out of range");
    }
    File& file = files_[file_id];
    open_file_(file);
    if (file_offset > file.mapped_size ||
        span_bytes > file.mapped_size - file_offset) {
        throw std::runtime_error("checkpoint byte range exceeds file size");
    }
    return file;
}

void CheckpointSource::copy_storage_bytes_to_device(
    std::uint32_t file_id,
    std::uint64_t file_offset,
    std::uint64_t span_bytes,
    void* dst) {
    if (dst == nullptr && span_bytes != 0) {
        throw std::runtime_error("checkpoint destination is null");
    }
    File& file = checked_file_(file_id, file_offset, span_bytes);
    map_file_(file);
    CUDA_CHECK(cudaMemcpy(
        dst,
        file.data + file_offset,
        span_bytes,
        cudaMemcpyHostToDevice));
}

void CheckpointSource::copy_storage_bytes_to_device_async(
    std::uint32_t file_id,
    std::uint64_t file_offset,
    std::uint64_t span_bytes,
    void* dst,
    void* stream) {
    if (dst == nullptr && span_bytes != 0) {
        throw std::runtime_error("checkpoint destination is null");
    }
    File& file = checked_file_(file_id, file_offset, span_bytes);
    map_file_(file);
    CUDA_CHECK(cudaMemcpyAsync(
        dst,
        file.data + file_offset,
        span_bytes,
        cudaMemcpyHostToDevice,
        static_cast<cudaStream_t>(stream)));
}

const std::uint8_t* CheckpointSource::storage_host_ptr(
    std::uint32_t file_id,
    std::uint64_t file_offset,
    std::uint64_t span_bytes) {
    File& file = checked_file_(file_id, file_offset, span_bytes);
    map_file_(file);
    return file.data + file_offset;
}

void CheckpointSource::read_storage_bytes_to_host(
    std::uint32_t file_id,
    std::uint64_t file_offset,
    std::uint64_t span_bytes,
    void* dst) {
    if (dst == nullptr && span_bytes != 0) {
        throw std::runtime_error("checkpoint host destination is null");
    }
    File& file = checked_file_(file_id, file_offset, span_bytes);
    auto* output = static_cast<std::uint8_t*>(dst);
    std::uint64_t done = 0;
    while (done < span_bytes) {
        const std::size_t chunk = static_cast<std::size_t>(
            std::min<std::uint64_t>(
                span_bytes - done,
                std::numeric_limits<std::size_t>::max()));
        const ssize_t read = ::pread(
            file.fd,
            output + done,
            chunk,
            static_cast<off_t>(file_offset + done));
        if (read < 0 && errno == EINTR) continue;
        if (read <= 0) {
            throw std::runtime_error(
                read == 0 ? "checkpoint short pread"
                          : std::string("checkpoint pread failed: ") +
                                std::strerror(errno));
        }
        done += static_cast<std::uint64_t>(read);
    }
}

}  // namespace pie_cuda_driver

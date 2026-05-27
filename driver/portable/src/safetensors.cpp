#include "safetensors.hpp"

#include <cerrno>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <utility>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <nlohmann/json.hpp>
#include <pie_driver_common/safetensors_manifest.hpp>

namespace pie_portable_driver {

namespace {

constexpr std::size_t MAX_HEADER_SIZE = 128ull * 1024 * 1024;  // 128 MiB sanity cap

#ifdef _WIN32
std::vector<std::uint8_t> read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error("safetensors: open(" + path.string() + ") failed");
    }
    const auto end = in.tellg();
    if (end < 0) {
        throw std::runtime_error("safetensors: tellg failed: " + path.string());
    }
    std::vector<std::uint8_t> data(static_cast<std::size_t>(end));
    in.seekg(0, std::ios::beg);
    if (!data.empty() &&
        !in.read(reinterpret_cast<char*>(data.data()),
                 static_cast<std::streamsize>(data.size()))) {
        throw std::runtime_error("safetensors: read failed: " + path.string());
    }
    return data;
}
#endif

StDtype parse_dtype(const std::string& s) {
    // Names per safetensors spec: F16, BF16, F32, F64, I8, U8, I16, U16,
    // I32, U32, I64, U64, BOOL, F8_E4M3, F8_E5M2.
    if (s == "F32")     return StDtype::F32;
    if (s == "F16")     return StDtype::F16;
    if (s == "BF16")    return StDtype::BF16;
    if (s == "F64")     return StDtype::F64;
    if (s == "I8")      return StDtype::I8;
    if (s == "U8")      return StDtype::U8;
    if (s == "I16")     return StDtype::I16;
    if (s == "U16")     return StDtype::U16;
    if (s == "I32")     return StDtype::I32;
    if (s == "U32")     return StDtype::U32;
    if (s == "I64")     return StDtype::I64;
    if (s == "U64")     return StDtype::U64;
    if (s == "BOOL")    return StDtype::BOOL;
    if (s == "F8_E4M3") return StDtype::F8_E4M3;
    if (s == "F8_E5M2") return StDtype::F8_E5M2;
    throw std::runtime_error("safetensors: unknown dtype '" + s + "'");
}

}  // namespace

std::size_t st_dtype_size(StDtype dt) {
    switch (dt) {
        case StDtype::F64: case StDtype::I64: case StDtype::U64: return 8;
        case StDtype::F32: case StDtype::I32: case StDtype::U32: return 4;
        case StDtype::F16: case StDtype::BF16:
        case StDtype::I16: case StDtype::U16: return 2;
        case StDtype::I8: case StDtype::U8: case StDtype::BOOL:
        case StDtype::F8_E4M3: case StDtype::F8_E5M2: return 1;
    }
    return 0;
}

const char* st_dtype_name(StDtype dt) {
    switch (dt) {
        case StDtype::F32: return "F32";
        case StDtype::F16: return "F16";
        case StDtype::BF16: return "BF16";
        case StDtype::F64: return "F64";
        case StDtype::I8: return "I8";
        case StDtype::U8: return "U8";
        case StDtype::I16: return "I16";
        case StDtype::U16: return "U16";
        case StDtype::I32: return "I32";
        case StDtype::U32: return "U32";
        case StDtype::I64: return "I64";
        case StDtype::U64: return "U64";
        case StDtype::BOOL: return "BOOL";
        case StDtype::F8_E4M3: return "F8_E4M3";
        case StDtype::F8_E5M2: return "F8_E5M2";
    }
    return "?";
}

SafetensorsShard::SafetensorsShard(const std::filesystem::path& path)
    : path_(path) {
#ifdef _WIN32
    owned_data_ = read_file(path_);
    mmap_size_ = owned_data_.size();
    if (mmap_size_ < 8) {
        throw std::runtime_error("safetensors: file too small: " +
                                 path_.string());
    }
    base_ = owned_data_.data();
#else
    fd_ = ::open(path_.c_str(), O_RDONLY);
    if (fd_ < 0) {
        throw std::runtime_error("safetensors: open(" + path_.string() +
                                 ") failed: " + std::strerror(errno));
    }

    struct stat st{};
    if (::fstat(fd_, &st) != 0) {
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("safetensors: fstat failed: " +
                                 std::string(std::strerror(errno)));
    }
    mmap_size_ = static_cast<std::size_t>(st.st_size);
    if (mmap_size_ < 8) {
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("safetensors: file too small: " +
                                 path_.string());
    }

    void* p = ::mmap(nullptr, mmap_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (p == MAP_FAILED) {
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("safetensors: mmap failed: " +
                                 std::string(std::strerror(errno)));
    }
    base_ = static_cast<const std::uint8_t*>(p);

    // Hint sequential / will-need pages — significant for cold-start latency
    // on large models. POSIX_MADV_WILLNEED is best-effort.
    ::posix_madvise(const_cast<void*>(static_cast<const void*>(base_)),
                    mmap_size_, POSIX_MADV_WILLNEED);
#endif

    // Parse header.
    std::uint64_t header_size = 0;
    std::memcpy(&header_size, base_, 8);
    if (header_size == 0 || header_size > MAX_HEADER_SIZE ||
        8 + header_size > mmap_size_) {
        close_mmap();
        throw std::runtime_error(
            "safetensors: invalid header size " + std::to_string(header_size));
    }

    nlohmann::json header;
    try {
        header = nlohmann::json::parse(base_ + 8, base_ + 8 + header_size);
    } catch (const std::exception& e) {
        close_mmap();
        throw std::runtime_error(
            "safetensors: header parse failed: " + std::string(e.what()));
    }
    if (!header.is_object()) {
        close_mmap();
        throw std::runtime_error("safetensors: header is not an object");
    }

    const std::size_t data_origin = 8 + static_cast<std::size_t>(header_size);

    for (auto it = header.begin(); it != header.end(); ++it) {
        if (it.key() == "__metadata__") continue;
        const auto& obj = it.value();
        if (!obj.is_object()) {
            throw std::runtime_error(
                "safetensors: entry '" + it.key() + "' is not an object");
        }

        StTensor t;
        t.dtype = parse_dtype(obj.at("dtype").get<std::string>());

        for (const auto& d : obj.at("shape")) {
            t.shape.push_back(d.get<std::int64_t>());
        }

        const auto& off = obj.at("data_offsets");
        if (!off.is_array() || off.size() != 2) {
            throw std::runtime_error(
                "safetensors: bad data_offsets for '" + it.key() + "'");
        }
        const auto begin = off[0].get<std::uint64_t>();
        const auto end   = off[1].get<std::uint64_t>();
        if (end < begin || data_origin + end > mmap_size_) {
            close_mmap();
            throw std::runtime_error(
                "safetensors: tensor '" + it.key() + "' out of file bounds");
        }
        t.data = base_ + data_origin + begin;
        t.nbytes = static_cast<std::size_t>(end - begin);

        // Sanity: nbytes must equal product(shape) * dtype_size, modulo
        // empty tensors.
        std::size_t expected = st_dtype_size(t.dtype);
        for (auto d : t.shape) {
            if (d < 0) {
                throw std::runtime_error(
                    "safetensors: negative dim in '" + it.key() + "'");
            }
            expected *= static_cast<std::size_t>(d);
        }
        if (expected != t.nbytes) {
            throw std::runtime_error(
                "safetensors: tensor '" + it.key() + "' size mismatch (" +
                std::to_string(expected) + " vs " + std::to_string(t.nbytes) +
                ")");
        }

        tensors_.emplace(it.key(), std::move(t));
    }
}

SafetensorsShard::SafetensorsShard(SafetensorsShard&& other) noexcept
    : path_(std::move(other.path_)),
#ifdef _WIN32
      owned_data_(std::move(other.owned_data_)),
#else
      fd_(other.fd_),
#endif
      mmap_size_(other.mmap_size_),
      base_(other.base_),
      tensors_(std::move(other.tensors_)) {
#ifdef _WIN32
    base_ = owned_data_.empty() ? nullptr : owned_data_.data();
#else
    other.fd_ = -1;
#endif
    other.mmap_size_ = 0;
    other.base_ = nullptr;
}

SafetensorsShard& SafetensorsShard::operator=(SafetensorsShard&& other) noexcept {
    if (this != &other) {
        close_mmap();
        path_ = std::move(other.path_);
#ifdef _WIN32
        owned_data_ = std::move(other.owned_data_);
        base_ = owned_data_.empty() ? nullptr : owned_data_.data();
#else
        fd_ = other.fd_;
        base_ = other.base_;
#endif
        mmap_size_ = other.mmap_size_;
        tensors_ = std::move(other.tensors_);
#ifndef _WIN32
        other.fd_ = -1;
#endif
        other.mmap_size_ = 0;
        other.base_ = nullptr;
    }
    return *this;
}

SafetensorsShard::~SafetensorsShard() { close_mmap(); }

void SafetensorsShard::close_mmap() noexcept {
#ifdef _WIN32
    owned_data_.clear();
    owned_data_.shrink_to_fit();
#else
    if (base_ && mmap_size_ > 0) {
        ::munmap(const_cast<void*>(static_cast<const void*>(base_)),
                 mmap_size_);
    }
    if (fd_ >= 0) {
        ::close(fd_);
    }
    fd_ = -1;
#endif
    base_ = nullptr;
    mmap_size_ = 0;
}

// -----------------------------------------------------------------------------

SafetensorsArchive::SafetensorsArchive(const std::filesystem::path& snapshot_dir) {
    if (!std::filesystem::is_directory(snapshot_dir)) {
        throw std::runtime_error(
            "safetensors: not a directory: " + snapshot_dir.string());
    }

    const auto manifest =
        pie_driver_common::discover_safetensors_manifest(snapshot_dir);
    const auto& shard_paths = manifest.shard_paths;

    for (const auto& p : shard_paths) {
        auto shard = std::make_unique<SafetensorsShard>(p);
        const std::size_t shard_idx = shards_.size();
        for (const auto& [name, _t] : shard->tensors()) {
            if (index_.count(name)) {
                throw std::runtime_error(
                    "safetensors: duplicate tensor across shards: " + name);
            }
            index_.emplace(name, shard_idx);
        }
        shards_.push_back(std::move(shard));
    }
}

const StTensor* SafetensorsArchive::find(const std::string& name) const noexcept {
    auto it = index_.find(name);
    if (it == index_.end()) return nullptr;
    const auto& tensors = shards_[it->second]->tensors();
    auto t_it = tensors.find(name);
    return t_it == tensors.end() ? nullptr : &t_it->second;
}

const StTensor& SafetensorsArchive::at(const std::string& name) const {
    if (auto* t = find(name)) return *t;
    throw std::runtime_error("safetensors: tensor not found: " + name);
}

std::size_t SafetensorsArchive::num_tensors() const noexcept {
    return index_.size();
}

}  // namespace pie_portable_driver

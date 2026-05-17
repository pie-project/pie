#include "loader/byte_source.hpp"

#include <algorithm>
#include <cstdint>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>

#if __has_include(<cufile.h>)
#include <cufile.h>
#define PIE_CUDA_BYTE_SOURCE_HAS_CUFILE_HEADER 1
#else
#define PIE_CUDA_BYTE_SOURCE_HAS_CUFILE_HEADER 0
#endif

namespace pie_cuda_driver {

namespace {

std::uint64_t tensor_nbytes(DType dtype, const std::vector<std::int64_t>& shape)
{
    std::uint64_t n = 1;
    for (const auto dim : shape) {
        n *= static_cast<std::uint64_t>(dim);
    }
    return n * static_cast<std::uint64_t>(dtype_bytes(dtype));
}

bool can_direct_read_contiguous(
    const TensorInfo& info,
    const ExtentWrite& write,
    std::uint64_t& source_delta)
{
    source_delta = 0;
    if (!write.contiguous) return false;
    if (write.slices.empty()) return true;
    if (write.slices.size() != 1 || write.slices[0].axis != 0) {
        return false;
    }
    if (info.shape.empty()) return false;
    std::vector<std::int64_t> inner_shape(
        info.shape.begin() + 1, info.shape.end());
    const std::uint64_t row_bytes = tensor_nbytes(info.dtype, inner_shape);
    source_delta =
        static_cast<std::uint64_t>(write.slices[0].start) * row_bytes;
    return true;
}

}  // namespace

CheckpointIoPolicy parse_checkpoint_io_policy(const std::string& value) {
    if (value.empty() || value == "auto") return CheckpointIoPolicy::Auto;
    if (value == "mmap") return CheckpointIoPolicy::Mmap;
    if (value == "gds") return CheckpointIoPolicy::Gds;
    throw std::runtime_error(
        "byte source: checkpoint_io must be one of {auto,mmap,gds}");
}

const char* checkpoint_io_policy_name(CheckpointIoPolicy policy) noexcept {
    switch (policy) {
    case CheckpointIoPolicy::Auto: return "auto";
    case CheckpointIoPolicy::Mmap: return "mmap";
    case CheckpointIoPolicy::Gds: return "gds";
    }
    return "?";
}

void CheckpointByteSource::write_to_device_async(
    const ExtentWrite& write,
    void* dst_base,
    void* stream)
{
    (void)stream;
    write_to_device(write, dst_base);
}

void MmapByteSource::write_to_device(
    const ExtentWrite& write,
    void* dst_base)
{
    if (dst_base == nullptr) {
        throw std::runtime_error(
            "byte source: null destination for '" + write.output_name + "'");
    }
    auto* dst = static_cast<std::uint8_t*>(dst_base) + write.dst_offset_bytes;
    loader_.copy_strided_to_device(
        write.raw_name,
        write.slices,
        dst,
        write.dst_shape);
}

void MmapByteSource::write_to_device_async(
    const ExtentWrite& write,
    void* dst_base,
    void* stream)
{
    if (dst_base == nullptr) {
        throw std::runtime_error(
            "byte source: null destination for '" + write.output_name + "'");
    }
    auto* dst = static_cast<std::uint8_t*>(dst_base) + write.dst_offset_bytes;
    loader_.copy_strided_to_device_async(
        write.raw_name,
        write.slices,
        dst,
        write.dst_shape,
        stream);
}

#if PIE_CUDA_BYTE_SOURCE_HAS_CUFILE_HEADER
class GdsByteSource::Impl {
public:
    Impl(bool required, bool verbose)
        : required_(required), verbose_(verbose)
    {
        open_library();
    }

    ~Impl()
    {
        for (auto& [_, handle] : handles_) {
            if (handle.fh != nullptr && api_.cuFileHandleDeregister != nullptr) {
                api_.cuFileHandleDeregister(handle.fh);
            }
            if (handle.fd >= 0) {
                ::close(handle.fd);
            }
        }
        if (driver_open_ && api_.cuFileDriverClose != nullptr) {
            api_.cuFileDriverClose();
        }
        if (lib_ != nullptr) {
            dlclose(lib_);
        }
    }

    bool available() const noexcept { return available_; }

    bool read_direct(
        const TensorStorageInfo& storage,
        std::uint64_t source_delta,
        std::uint64_t bytes,
        void* dst)
    {
        if (!available_ || dst == nullptr || bytes == 0) return false;

        RegisteredHandle* handle = handle_for(storage.path);
        if (handle == nullptr) return false;

        const auto file_offset = static_cast<off_t>(
            storage.file_offset + source_delta);
        std::uint64_t remaining = bytes;
        std::uint64_t copied = 0;
        while (remaining > 0) {
            const std::size_t chunk = static_cast<std::size_t>(
                std::min<std::uint64_t>(
                    remaining,
                    static_cast<std::uint64_t>(
                        std::numeric_limits<std::size_t>::max())));
            const ssize_t rc = api_.cuFileRead(
                handle->fh,
                static_cast<std::uint8_t*>(dst) + copied,
                chunk,
                file_offset + static_cast<off_t>(copied),
                /*bufPtr_offset=*/0);
            if (rc < 0 || static_cast<std::size_t>(rc) != chunk) {
                if (required_) {
                    throw std::runtime_error(
                        "byte source: cuFileRead failed for '" +
                        storage.path.string() + "' at offset " +
                        std::to_string(static_cast<std::uint64_t>(
                            file_offset + static_cast<off_t>(copied))) +
                        ": " + std::strerror(errno));
                }
                return false;
            }
            remaining -= static_cast<std::uint64_t>(chunk);
            copied += static_cast<std::uint64_t>(chunk);
        }
        return true;
    }

private:
    using DriverOpenFn = CUfileError_t (*)();
    using DriverCloseFn = CUfileError_t (*)();
    using HandleRegisterFn = CUfileError_t (*)(CUfileHandle_t*, CUfileDescr_t*);
    using HandleDeregisterFn = void (*)(CUfileHandle_t);
    using ReadFn = ssize_t (*)(CUfileHandle_t, void*, size_t, off_t, off_t);

    struct Api {
        DriverOpenFn cuFileDriverOpen = nullptr;
        DriverCloseFn cuFileDriverClose = nullptr;
        HandleRegisterFn cuFileHandleRegister = nullptr;
        HandleDeregisterFn cuFileHandleDeregister = nullptr;
        ReadFn cuFileRead = nullptr;
    };

    struct RegisteredHandle {
        int fd = -1;
        CUfileHandle_t fh = nullptr;
    };

    template <typename Fn>
    Fn load_symbol(const char* name)
    {
        auto* sym = dlsym(lib_, name);
        if (sym == nullptr) {
            return nullptr;
        }
        return reinterpret_cast<Fn>(sym);
    }

    static bool ok(CUfileError_t err) noexcept {
        return err.err == CU_FILE_SUCCESS ||
               err.err == CU_FILE_DRIVER_ALREADY_OPEN ||
               err.err == CU_FILE_HANDLE_ALREADY_REGISTERED;
    }

    void open_library()
    {
        lib_ = dlopen("libcufile.so", RTLD_NOW | RTLD_LOCAL);
        if (lib_ == nullptr) {
            lib_ = dlopen("libcufile.so.0", RTLD_NOW | RTLD_LOCAL);
        }
        if (lib_ == nullptr) {
            if (required_) {
                throw std::runtime_error(
                    "byte source: checkpoint_io='gds' requested but "
                    "libcufile.so is not available");
            }
            if (verbose_) {
                std::cerr << "[pie-driver-cuda] checkpoint_io=auto: "
                          << "libcufile.so not found; using mmap\n";
            }
            return;
        }

        api_.cuFileDriverOpen =
            load_symbol<DriverOpenFn>("cuFileDriverOpen");
        api_.cuFileDriverClose =
            load_symbol<DriverCloseFn>("cuFileDriverClose_v2");
        if (api_.cuFileDriverClose == nullptr) {
            api_.cuFileDriverClose =
                load_symbol<DriverCloseFn>("cuFileDriverClose");
        }
        api_.cuFileHandleRegister =
            load_symbol<HandleRegisterFn>("cuFileHandleRegister");
        api_.cuFileHandleDeregister =
            load_symbol<HandleDeregisterFn>("cuFileHandleDeregister");
        api_.cuFileRead = load_symbol<ReadFn>("cuFileRead");

        if (api_.cuFileDriverOpen == nullptr ||
            api_.cuFileDriverClose == nullptr ||
            api_.cuFileHandleRegister == nullptr ||
            api_.cuFileHandleDeregister == nullptr ||
            api_.cuFileRead == nullptr) {
            if (required_) {
                throw std::runtime_error(
                    "byte source: libcufile.so is missing required symbols");
            }
            if (verbose_) {
                std::cerr << "[pie-driver-cuda] checkpoint_io=auto: "
                          << "libcufile symbols unavailable; using mmap\n";
            }
            return;
        }

        const CUfileError_t err = api_.cuFileDriverOpen();
        if (!ok(err)) {
            if (required_) {
                throw std::runtime_error(
                    "byte source: cuFileDriverOpen failed with status " +
                    std::to_string(static_cast<int>(err.err)));
            }
            if (verbose_) {
                std::cerr << "[pie-driver-cuda] checkpoint_io=auto: "
                          << "cuFileDriverOpen failed with status "
                          << static_cast<int>(err.err)
                          << "; using mmap\n";
            }
            return;
        }
        driver_open_ = true;
        available_ = true;
    }

    RegisteredHandle* handle_for(const std::filesystem::path& path)
    {
        const std::string key = path.string();
        if (auto it = handles_.find(key); it != handles_.end()) {
            return &it->second;
        }

        int fd = ::open(path.c_str(), O_RDONLY | O_DIRECT);
        if (fd < 0) {
            if (required_) {
                throw std::runtime_error(
                    "byte source: open(O_DIRECT) failed for '" + key +
                    "': " + std::strerror(errno));
            }
            return nullptr;
        }

        CUfileDescr_t descr{};
        descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
        descr.handle.fd = fd;
        CUfileHandle_t fh = nullptr;
        const CUfileError_t err = api_.cuFileHandleRegister(&fh, &descr);
        if (!ok(err)) {
            ::close(fd);
            if (required_) {
                throw std::runtime_error(
                    "byte source: cuFileHandleRegister failed for '" + key +
                    "' with status " +
                    std::to_string(static_cast<int>(err.err)));
            }
            return nullptr;
        }

        auto [it, inserted] = handles_.emplace(
            key, RegisteredHandle{.fd = fd, .fh = fh});
        (void)inserted;
        return &it->second;
    }

    bool required_ = false;
    bool verbose_ = false;
    bool available_ = false;
    bool driver_open_ = false;
    void* lib_ = nullptr;
    Api api_;
    std::unordered_map<std::string, RegisteredHandle> handles_;
};
#else
class GdsByteSource::Impl {
public:
    Impl(bool required, bool)
    {
        if (required) {
            throw std::runtime_error(
                "byte source: checkpoint_io='gds' requested but this build "
                "was compiled without cufile.h");
        }
    }
    bool available() const noexcept { return false; }
    bool read_direct(
        const TensorStorageInfo&,
        std::uint64_t,
        std::uint64_t,
        void*)
    {
        return false;
    }
};
#endif

GdsByteSource::GdsByteSource(
    SafetensorsCheckpointSource& loader,
    bool required,
    bool verbose)
    : loader_(loader),
      fallback_(loader),
      impl_(std::make_unique<Impl>(required, verbose)),
      required_(required),
      verbose_(verbose),
      direct_enabled_(impl_->available())
{
    if (verbose_ && direct_enabled_) {
        std::cerr << "[pie-driver-cuda] checkpoint_io="
                  << (required_ ? "gds" : "auto")
                  << ": using GPUDirect Storage for contiguous extent writes\n";
    }
}

GdsByteSource::~GdsByteSource() = default;

void GdsByteSource::write_to_device(
    const ExtentWrite& write,
    void* dst_base)
{
    if (dst_base == nullptr) {
        throw std::runtime_error(
            "byte source: null destination for '" + write.output_name + "'");
    }
    if (direct_enabled_) {
        const TensorInfo& info = loader_.info(write.raw_name);
        std::uint64_t source_delta = 0;
        if (can_direct_read_contiguous(info, write, source_delta)) {
            auto* dst = static_cast<std::uint8_t*>(dst_base) +
                        write.dst_offset_bytes;
            if (impl_->read_direct(
                    loader_.storage_info(write.raw_name),
                    source_delta,
                    write.bytes,
                    dst)) {
                return;
            }
        } else if (required_) {
            throw std::runtime_error(
                "byte source: checkpoint_io='gds' cannot directly serve "
                "non-contiguous write for '" + write.output_name + "'");
        }
    }
    fallback_.write_to_device(write, dst_base);
}

bool GdsByteSource::supports_async_writes() const noexcept
{
    return !direct_enabled_ && fallback_.supports_async_writes();
}

void GdsByteSource::write_to_device_async(
    const ExtentWrite& write,
    void* dst_base,
    void* stream)
{
    if (!direct_enabled_) {
        fallback_.write_to_device_async(write, dst_base, stream);
        return;
    }
    write_to_device(write, dst_base);
}

std::unique_ptr<CheckpointByteSource> make_checkpoint_byte_source(
    CheckpointIoPolicy policy,
    SafetensorsCheckpointSource& loader,
    bool verbose)
{
    switch (policy) {
    case CheckpointIoPolicy::Mmap:
        return std::make_unique<MmapByteSource>(loader);
    case CheckpointIoPolicy::Gds:
        return std::make_unique<GdsByteSource>(
            loader, /*required=*/true, verbose);
    case CheckpointIoPolicy::Auto:
        return std::make_unique<GdsByteSource>(
            loader, /*required=*/false, verbose);
    }
    return std::make_unique<MmapByteSource>(loader);
}

}  // namespace pie_cuda_driver

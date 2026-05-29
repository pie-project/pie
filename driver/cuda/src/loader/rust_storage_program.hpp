#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../../weight_loader/include/weight_loader.h"

extern "C" {
#if defined(__GNUC__) || defined(__clang__)
#define PIE_CUDA_RUST_LOADER_WEAK __attribute__((weak))
#else
#define PIE_CUDA_RUST_LOADER_WEAK
#endif

pie_weight_loader::PieLoaderStatus pie_loader_compile(
    const pie_weight_loader::PieLoaderCompileInput* input,
    pie_weight_loader::PieLoaderProgramHandle** out_program,
    pie_weight_loader::PieLoaderError* out_error) PIE_CUDA_RUST_LOADER_WEAK;
pie_weight_loader::PieLoaderStatus pie_loader_program_serialized_len(
    const pie_weight_loader::PieLoaderProgramHandle* program,
    std::size_t* out_len,
    pie_weight_loader::PieLoaderError* out_error) PIE_CUDA_RUST_LOADER_WEAK;
pie_weight_loader::PieLoaderStatus pie_loader_program_serialize(
    const pie_weight_loader::PieLoaderProgramHandle* program,
    std::uint8_t* dst,
    std::size_t dst_len,
    pie_weight_loader::PieLoaderError* out_error) PIE_CUDA_RUST_LOADER_WEAK;
pie_weight_loader::PieLoaderStatus pie_loader_program_deserialize(
    const std::uint8_t* bytes,
    std::size_t bytes_len,
    pie_weight_loader::PieLoaderProgramHandle** out_program,
    pie_weight_loader::PieLoaderError* out_error) PIE_CUDA_RUST_LOADER_WEAK;
pie_weight_loader::PieLoaderStorageProgramView pie_loader_program_view(
    const pie_weight_loader::PieLoaderProgramHandle* program)
    PIE_CUDA_RUST_LOADER_WEAK;
void pie_loader_program_free(
    pie_weight_loader::PieLoaderProgramHandle* program)
    PIE_CUDA_RUST_LOADER_WEAK;
void pie_loader_error_free(pie_weight_loader::PieLoaderError* error)
    PIE_CUDA_RUST_LOADER_WEAK;
// Build-time content hash of the loader's Rust compiler source (build.rs).
// Folded into the storage-program cache key so a compiler-logic change auto-
// invalidates the on-disk cache.
std::uint64_t pie_loader_compiler_version(void) PIE_CUDA_RUST_LOADER_WEAK;
}

#undef PIE_CUDA_RUST_LOADER_WEAK

namespace pie_cuda_driver {

class RustStorageProgram {
public:
    explicit RustStorageProgram(
        pie_weight_loader::PieLoaderProgramHandle* handle) noexcept
        : handle_(handle)
    {}

    RustStorageProgram(const RustStorageProgram&) = delete;
    RustStorageProgram& operator=(const RustStorageProgram&) = delete;

    RustStorageProgram(RustStorageProgram&& other) noexcept
        : handle_(other.handle_)
    {
        other.handle_ = nullptr;
    }

    RustStorageProgram& operator=(RustStorageProgram&& other) noexcept
    {
        if (this != &other) {
            reset();
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    ~RustStorageProgram() { reset(); }

    pie_weight_loader::PieLoaderStorageProgramView view() const noexcept
    {
        if (::pie_loader_program_view == nullptr) {
            return pie_weight_loader::PieLoaderStorageProgramView{};
        }
        return ::pie_loader_program_view(handle_);
    }

    explicit operator bool() const noexcept { return handle_ != nullptr; }

    const pie_weight_loader::PieLoaderProgramHandle* get() const noexcept
    {
        return handle_;
    }

    pie_weight_loader::PieLoaderProgramHandle* release() noexcept
    {
        auto* out = handle_;
        handle_ = nullptr;
        return out;
    }

    void reset() noexcept
    {
        if (handle_ != nullptr && ::pie_loader_program_free != nullptr) {
            ::pie_loader_program_free(handle_);
            handle_ = nullptr;
        }
    }

private:
    pie_weight_loader::PieLoaderProgramHandle* handle_ = nullptr;
};

class RustLoaderError {
public:
    RustLoaderError() = default;
    RustLoaderError(const RustLoaderError&) = delete;
    RustLoaderError& operator=(const RustLoaderError&) = delete;

    ~RustLoaderError()
    {
        if (::pie_loader_error_free != nullptr) {
            ::pie_loader_error_free(&error_);
        }
    }

    pie_weight_loader::PieLoaderError* out() noexcept { return &error_; }

    std::string message() const
    {
        return error_.message == nullptr ? std::string{} : std::string(error_.message);
    }

private:
    pie_weight_loader::PieLoaderError error_{};
};

inline RustStorageProgram compile_rust_storage_program(
    const pie_weight_loader::PieLoaderCompileInput& input)
{
    pie_weight_loader::PieLoaderProgramHandle* handle = nullptr;
    RustLoaderError error;
    if (::pie_loader_compile == nullptr) {
        throw std::runtime_error(
            "rust weight loader compile failed: pie-weight-loader symbols "
            "are not linked into this binary");
    }
    const auto status = ::pie_loader_compile(
        &input, &handle, error.out());
    if (status != pie_weight_loader::PieLoaderStatus::Ok) {
        throw std::runtime_error(
            "rust weight loader compile failed: " + error.message());
    }
    return RustStorageProgram(handle);
}

inline std::vector<std::uint8_t> serialize_rust_storage_program(
    const RustStorageProgram& program)
{
    if (::pie_loader_program_serialized_len == nullptr ||
        ::pie_loader_program_serialize == nullptr) {
        throw std::runtime_error(
            "rust weight loader serialize failed: pie-weight-loader cache "
            "symbols are not linked into this binary");
    }
    std::size_t len = 0;
    RustLoaderError error;
    auto status = ::pie_loader_program_serialized_len(
        program.get(), &len, error.out());
    if (status != pie_weight_loader::PieLoaderStatus::Ok) {
        throw std::runtime_error(
            "rust weight loader serialize length failed: " + error.message());
    }
    std::vector<std::uint8_t> bytes(len);
    status = ::pie_loader_program_serialize(
        program.get(), bytes.data(), bytes.size(), error.out());
    if (status != pie_weight_loader::PieLoaderStatus::Ok) {
        throw std::runtime_error(
            "rust weight loader serialize failed: " + error.message());
    }
    return bytes;
}

inline RustStorageProgram deserialize_rust_storage_program(
    const std::vector<std::uint8_t>& bytes)
{
    if (::pie_loader_program_deserialize == nullptr) {
        throw std::runtime_error(
            "rust weight loader deserialize failed: pie-weight-loader cache "
            "symbols are not linked into this binary");
    }
    pie_weight_loader::PieLoaderProgramHandle* handle = nullptr;
    RustLoaderError error;
    const auto status = ::pie_loader_program_deserialize(
        bytes.data(), bytes.size(), &handle, error.out());
    if (status != pie_weight_loader::PieLoaderStatus::Ok) {
        throw std::runtime_error(
            "rust weight loader deserialize failed: " + error.message());
    }
    return RustStorageProgram(handle);
}

}  // namespace pie_cuda_driver

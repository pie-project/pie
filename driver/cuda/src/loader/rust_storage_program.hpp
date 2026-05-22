#pragma once

#include <stdexcept>
#include <string>

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
pie_weight_loader::PieLoaderStorageProgramView pie_loader_program_view(
    const pie_weight_loader::PieLoaderProgramHandle* program)
    PIE_CUDA_RUST_LOADER_WEAK;
void pie_loader_program_free(
    pie_weight_loader::PieLoaderProgramHandle* program)
    PIE_CUDA_RUST_LOADER_WEAK;
void pie_loader_error_free(pie_weight_loader::PieLoaderError* error)
    PIE_CUDA_RUST_LOADER_WEAK;
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

}  // namespace pie_cuda_driver

#pragma once

// DeviceTensor — owning RAII wrapper over a `cudaMalloc`'d buffer with
// dtype, shape, and stride metadata. No automatic dtype casting; the model
// graph is responsible for matching dtypes between operations.

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pie_cuda_driver {

enum class DType : std::uint8_t {
    BF16 = 0,
    FP16 = 1,
    FP32 = 2,
    INT8 = 3,
    INT32 = 4,
    INT64 = 5,
    UINT8 = 6,
};

inline std::size_t dtype_bytes(DType d) {
    switch (d) {
        case DType::BF16:  return 2;
        case DType::FP16:  return 2;
        case DType::FP32:  return 4;
        case DType::INT8:  return 1;
        case DType::INT32: return 4;
        case DType::INT64: return 8;
        case DType::UINT8: return 1;
    }
    throw std::runtime_error("unknown dtype");
}

inline const char* dtype_name(DType d) {
    switch (d) {
        case DType::BF16:  return "bf16";
        case DType::FP16:  return "fp16";
        case DType::FP32:  return "fp32";
        case DType::INT8:  return "int8";
        case DType::INT32: return "int32";
        case DType::INT64: return "int64";
        case DType::UINT8: return "u8";
    }
    return "?";
}

// Parse the safetensors dtype string ("BF16", "F16", "F32", "I8", "U8", …).
DType dtype_from_safetensors(const std::string& s);

class DeviceTensor {
public:
    DeviceTensor() = default;

    // Allocate uninitialised device memory. Caller writes into `data()`.
    static DeviceTensor allocate(DType dtype, std::vector<std::int64_t> shape);

    DeviceTensor(const DeviceTensor&) = delete;
    DeviceTensor& operator=(const DeviceTensor&) = delete;

    DeviceTensor(DeviceTensor&& other) noexcept
        : ptr_(other.ptr_),
          dtype_(other.dtype_),
          shape_(std::move(other.shape_)),
          numel_(other.numel_),
          nbytes_(other.nbytes_) {
        other.ptr_ = nullptr;
        other.numel_ = 0;
        other.nbytes_ = 0;
    }

    DeviceTensor& operator=(DeviceTensor&& other) noexcept {
        if (this != &other) {
            free_();
            ptr_ = other.ptr_;
            dtype_ = other.dtype_;
            shape_ = std::move(other.shape_);
            numel_ = other.numel_;
            nbytes_ = other.nbytes_;
            other.ptr_ = nullptr;
            other.numel_ = 0;
            other.nbytes_ = 0;
        }
        return *this;
    }

    ~DeviceTensor() { free_(); }

    void*       data()       noexcept { return ptr_; }
    const void* data() const noexcept { return ptr_; }

    DType                              dtype()  const noexcept { return dtype_; }
    const std::vector<std::int64_t>&   shape()  const noexcept { return shape_; }
    std::size_t                        numel()  const noexcept { return numel_; }
    std::size_t                        nbytes() const noexcept { return nbytes_; }

    bool empty() const noexcept { return ptr_ == nullptr; }

private:
    void free_() noexcept;

    void* ptr_ = nullptr;
    DType dtype_ = DType::BF16;
    std::vector<std::int64_t> shape_;
    std::size_t numel_ = 0;
    std::size_t nbytes_ = 0;
};

}  // namespace pie_cuda_driver

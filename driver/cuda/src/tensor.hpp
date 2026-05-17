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
    // Quantized scalar types. Storage size = 1 byte per logical element.
    // The kernels that consume these read the raw bytes via the matching
    // CUDA fp8 storage type (`__nv_fp8_storage_t` for E4M3 / E5M2). The
    // dtype tag exists so the GEMM dispatcher can pick the right cuBLAS
    // path; everything else (loader, scale tensor, dequant kernel) is
    // unchanged.
    FP8_E4M3 = 7,
    FP8_E5M2 = 8,
    // Marlin-packed INT4. Storage convention matches marlin's expected
    // input to its W4A16 GEMM: a 2-D tensor whose logical shape is the
    // marlin tile-packed layout produced by `gptq_marlin_repack` (i.e.
    // `[K / tile_k_size, N * tile_k_size / pack_factor]` where
    // tile_k_size = 16 and pack_factor = 8 for int4). Each storage
    // element is one byte holding two 4-bit values; the GEMM dispatcher
    // reads `(M, N, K)` from the QuantMeta companion (group_size /
    // channel_axis) plus the tensor shape rather than from the dtype.
    INT4_PACKED = 9,
};

inline std::size_t dtype_bytes(DType d) {
    switch (d) {
        case DType::BF16:     return 2;
        case DType::FP16:     return 2;
        case DType::FP32:     return 4;
        case DType::INT8:     return 1;
        case DType::INT32:    return 4;
        case DType::INT64:    return 8;
        case DType::UINT8:    return 1;
        case DType::FP8_E4M3: return 1;
        case DType::FP8_E5M2: return 1;
        case DType::INT4_PACKED: return 1;  // 1 byte holds 2 nibbles
    }
    throw std::runtime_error("unknown dtype");
}

inline const char* dtype_name(DType d) {
    switch (d) {
        case DType::BF16:     return "bf16";
        case DType::FP16:     return "fp16";
        case DType::FP32:     return "fp32";
        case DType::INT8:     return "int8";
        case DType::INT32:    return "int32";
        case DType::INT64:    return "int64";
        case DType::UINT8:    return "u8";
        case DType::FP8_E4M3: return "fp8e4m3";
        case DType::FP8_E5M2: return "fp8e5m2";
        case DType::INT4_PACKED: return "int4-packed";
    }
    return "?";
}

// Parse the safetensors dtype string ("BF16", "F16", "F32", "I8", "U8", …).
DType dtype_from_safetensors(const std::string& s);

using DeviceTensorMemoryCallback = void (*)(void* context);

// Thread-local hook used by the loader to capture CUDA memory high-water
// during materialization, including transient transform scratch allocated
// inside helper kernels. Passing nullptr clears the callback.
void set_device_tensor_memory_callback(
    DeviceTensorMemoryCallback callback,
    void* context) noexcept;

class DeviceTensor {
public:
    DeviceTensor() = default;

    // Allocate uninitialised device memory. Caller writes into `data()`.
    static DeviceTensor allocate(DType dtype, std::vector<std::int64_t> shape);

    // Non-owning view into existing device memory (e.g. a slice of an
    // already-loaded fused tensor). The caller is responsible for
    // keeping the backing allocation alive at least as long as the
    // view; the destructor here intentionally does not free.
    static DeviceTensor view(void* ptr, DType dtype,
                             std::vector<std::int64_t> shape);

    DeviceTensor(const DeviceTensor&) = delete;
    DeviceTensor& operator=(const DeviceTensor&) = delete;

    DeviceTensor(DeviceTensor&& other) noexcept
        : ptr_(other.ptr_),
          dtype_(other.dtype_),
          shape_(std::move(other.shape_)),
          numel_(other.numel_),
          nbytes_(other.nbytes_),
          owns_memory_(other.owns_memory_) {
        other.ptr_ = nullptr;
        other.numel_ = 0;
        other.nbytes_ = 0;
        other.owns_memory_ = false;
    }

    DeviceTensor& operator=(DeviceTensor&& other) noexcept {
        if (this != &other) {
            free_();
            ptr_ = other.ptr_;
            dtype_ = other.dtype_;
            shape_ = std::move(other.shape_);
            numel_ = other.numel_;
            nbytes_ = other.nbytes_;
            owns_memory_ = other.owns_memory_;
            other.ptr_ = nullptr;
            other.numel_ = 0;
            other.nbytes_ = 0;
            other.owns_memory_ = false;
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
    bool                               owns_memory() const noexcept { return owns_memory_; }

    bool empty() const noexcept { return ptr_ == nullptr; }

private:
    void free_() noexcept;

    void* ptr_ = nullptr;
    DType dtype_ = DType::BF16;
    std::vector<std::int64_t> shape_;
    std::size_t numel_ = 0;
    std::size_t nbytes_ = 0;
    // True for `allocate`d tensors (own + free on destruct), false for
    // non-owning views (`view(...)`). Move semantics propagate ownership.
    bool owns_memory_ = false;
};

}  // namespace pie_cuda_driver

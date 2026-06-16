#include "tensor.hpp"

#include <cuda_runtime.h>
#include <iostream>

#include "cuda_check.hpp"

namespace pie_cuda_driver {

namespace {

thread_local DeviceTensorMemoryCallback g_memory_callback = nullptr;
thread_local void* g_memory_callback_context = nullptr;

void sample_memory_callback() noexcept {
    if (g_memory_callback != nullptr) {
        g_memory_callback(g_memory_callback_context);
    }
}

}  // namespace

void set_device_tensor_memory_callback(
    DeviceTensorMemoryCallback callback,
    void* context) noexcept
{
    g_memory_callback = callback;
    g_memory_callback_context = context;
}

DType dtype_from_safetensors(const std::string& s) {
    if (s == "BF16") return DType::BF16;
    if (s == "F16")  return DType::FP16;
    if (s == "F32")  return DType::FP32;
    if (s == "I8")   return DType::INT8;
    if (s == "I32")  return DType::INT32;
    if (s == "I64")  return DType::INT64;
    if (s == "U8")   return DType::UINT8;
    // FP8 carries its own dtype tag so the GEMM dispatcher can route to a
    // native FP8 cuBLAS path (or a dequant-on-load fallback in mistral3).
    if (s == "F8_E4M3") return DType::FP8_E4M3;
    if (s == "F8_E5M2") return DType::FP8_E5M2;
    // MXFP4 (F4_E2M1) rides on UINT8 storage: two nibbles per byte plus a
    // side E8M0 scale tensor described by QuantSpec. Target lowering decides
    // whether those bytes remain native QuantPacked runtime tensors or
    // dequantize to BF16 during materialization.
    if (s == "F4_E2M1") return DType::UINT8;
    if (s == "F8_E8M0") return DType::UINT8;
    throw std::runtime_error("unsupported safetensors dtype: " + s);
}

DeviceTensor DeviceTensor::allocate(DType dtype, std::vector<std::int64_t> shape) {
    DeviceTensor t;
    t.dtype_ = dtype;
    t.shape_ = std::move(shape);
    t.numel_ = 1;
    for (auto d : t.shape_) {
        if (d < 0) throw std::runtime_error("DeviceTensor: negative shape");
        t.numel_ *= static_cast<std::size_t>(d);
    }
    t.nbytes_ = t.numel_ * dtype_bytes(dtype);
    if (t.nbytes_ > 0) {
        CUDA_CHECK(cudaMalloc(&t.ptr_, t.nbytes_));
        sample_memory_callback();
    }
    t.owns_memory_ = true;
    return t;
}

DeviceTensor DeviceTensor::view(void* ptr, DType dtype,
                                std::vector<std::int64_t> shape) {
    DeviceTensor t;
    t.ptr_ = ptr;
    t.dtype_ = dtype;
    t.shape_ = std::move(shape);
    t.numel_ = 1;
    for (auto d : t.shape_) {
        if (d < 0) throw std::runtime_error("DeviceTensor::view: negative shape");
        t.numel_ *= static_cast<std::size_t>(d);
    }
    t.nbytes_ = t.numel_ * dtype_bytes(dtype);
    t.owns_memory_ = false;
    return t;
}

void DeviceTensor::free_() noexcept {
    if (ptr_ && owns_memory_) {
        sample_memory_callback();
        // Best-effort free; never throw from a destructor.
        if (cudaFree(ptr_) != cudaSuccess) {
            // Pre-shutdown errors are common (driver torn down). Stay quiet
            // unless we're mid-run.
        }
        sample_memory_callback();
    }
    ptr_ = nullptr;
}

}  // namespace pie_cuda_driver

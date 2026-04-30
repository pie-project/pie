#include "tensor.hpp"

#include <cuda_runtime.h>
#include <iostream>

#include "cuda_check.hpp"

namespace pie_cuda_driver {

DType dtype_from_safetensors(const std::string& s) {
    if (s == "BF16") return DType::BF16;
    if (s == "F16")  return DType::FP16;
    if (s == "F32")  return DType::FP32;
    if (s == "I8")   return DType::INT8;
    if (s == "I32")  return DType::INT32;
    if (s == "I64")  return DType::INT64;
    if (s == "U8")   return DType::UINT8;
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
    }
    return t;
}

void DeviceTensor::free_() noexcept {
    if (ptr_) {
        // Best-effort free; never throw from a destructor.
        if (cudaFree(ptr_) != cudaSuccess) {
            // Pre-shutdown errors are common (driver torn down). Stay quiet
            // unless we're mid-run.
        }
        ptr_ = nullptr;
    }
}

}  // namespace pie_cuda_driver

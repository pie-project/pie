#include "tensor.hpp"

#include <cuda_runtime.h>
#include <iostream>

#include "cuda_check.hpp"

namespace pie_cuda_driver {

namespace {

thread_local DeviceTensorMemoryCallback g_memory_callback = nullptr;
thread_local void* g_memory_callback_context = nullptr;
thread_local DeviceMemoryAllocatorBinding g_memory_allocator{};

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

DeviceMemoryAllocatorBinding set_device_memory_allocator(
    DeviceMemoryAllocateCallback allocate,
    void* context) noexcept {
    const DeviceMemoryAllocatorBinding previous = g_memory_allocator;
    g_memory_allocator = {allocate, context};
    return previous;
}

DeviceMemoryBlock allocate_device_memory(
    std::size_t bytes,
    std::size_t alignment) {
    if (bytes == 0) return {};
    DeviceMemoryBlock block;
    if (g_memory_allocator.allocate != nullptr) {
        block.ptr = g_memory_allocator.allocate(
            g_memory_allocator.context,
            bytes,
            alignment);
        block.arena_owned = true;
    } else {
        CUDA_CHECK(cudaMalloc(&block.ptr, bytes));
    }
    sample_memory_callback();
    return block;
}

void free_device_memory(DeviceMemoryBlock block) noexcept {
    if (block.ptr == nullptr || block.arena_owned) return;
    sample_memory_callback();
    cudaFree(block.ptr);
    sample_memory_callback();
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
        const DeviceMemoryBlock block =
            allocate_device_memory(t.nbytes_, 256);
        t.ptr_ = block.ptr;
        t.arena_owned_ = block.arena_owned;
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
        free_device_memory({ptr_, arena_owned_});
    }
    ptr_ = nullptr;
    arena_owned_ = false;
}

}  // namespace pie_cuda_driver

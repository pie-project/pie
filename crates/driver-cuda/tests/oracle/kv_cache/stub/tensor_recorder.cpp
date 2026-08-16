// The ONE thing this oracle replaces: `DeviceTensor::allocate`.
//
// `tensor.cpp` in the archive crate's `csrc/src` -- deleted with that crate
// at `85c6c674b` -- ends in a `cudaMalloc`, so every shape the shipping
// `kv_cache.cpp` computes is consumed by the driver and leaves only a pointer
// behind. The class is declared in the REAL `tensor.hpp`, which is copied
// verbatim; only this implementation differs, and all it does is record the
// (dtype, shape) pair it was handed.
//
// That is the entire boundary. Every decision about WHICH tensors exist, what
// extents they have, and in what order they are created is made by the
// shipping code.
#include "tensor.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace pie_cuda_driver {

std::vector<std::string> g_alloc_log;

namespace {
// A distinct, non-null, never-dereferenced address per allocation. The cache
// classes test `t.empty()` (i.e. `ptr_ == nullptr`) to tell a real layer from
// an aliased placeholder, so the fake pointer must be non-null for anything
// with bytes -- and, like the real allocator, null for a zero-byte request.
std::uintptr_t g_next = 0x1000;

}  // namespace

// Symbolic names for the fabricated addresses, `t#K` in allocation order.
// Added for the kv_cache_live oracle, which reports POINTER WIRING (which
// tensor lands in which view field); the layout oracles never consult it,
// and it changes no existing row.
namespace {
std::map<const void*, std::string> g_tensor_names;
}  // namespace

std::string tensor_name(const void* ptr) {
    if (ptr == nullptr) return "null";
    auto it = g_tensor_names.find(ptr);
    return it == g_tensor_names.end() ? "unknown" : it->second;
}

void reset_alloc_log() {
    g_alloc_log.clear();
    g_next = 0x1000;
    g_tensor_names.clear();
}

const std::vector<std::string>& alloc_log() { return g_alloc_log; }

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

    std::string row = dtype_name(dtype);
    row += '[';
    for (std::size_t i = 0; i < t.shape_.size(); ++i) {
        if (i != 0) row += ',';
        row += std::to_string(t.shape_[i]);
    }
    row += "]=" + std::to_string(t.nbytes_);
    g_alloc_log.push_back(row);

    if (t.nbytes_ > 0) {
        t.ptr_ = reinterpret_cast<void*>(g_next);
        g_next += 4096;
        t.arena_owned_ = false;
        g_tensor_names[t.ptr_] =
            "t#" + std::to_string(g_tensor_names.size());
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
    ptr_ = nullptr;
    numel_ = 0;
    nbytes_ = 0;
    owns_memory_ = false;
    arena_owned_ = false;
}

// The allocator binding, which `allocate_envelopes_` swaps out and restores.
//
// Real in the sense that matters here: it stores and returns the previous
// value, so the RAII restore in the shipping code is observable. Nothing
// routes through it, because this recorder replaces `DeviceTensor::allocate`
// above the point where the binding is consulted.
namespace {
DeviceMemoryAllocatorBinding g_binding{};
}

DeviceMemoryAllocatorBinding set_device_memory_allocator(
    DeviceMemoryAllocateCallback allocate, void* context) noexcept {
    const DeviceMemoryAllocatorBinding prev = g_binding;
    g_binding.allocate = allocate;
    g_binding.context = context;
    g_alloc_log.push_back(std::string("bind(") +
                          (allocate ? "custom" : "default") + ")");
    return prev;
}

void set_device_tensor_memory_callback(DeviceTensorMemoryCallback,
                                       void*) noexcept {}

}  // namespace pie_cuda_driver

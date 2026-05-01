#pragma once

// Owning RAII wrapper around `cudaMalloc`'d memory. Replaces the scattered
// raw `T* d_x = nullptr; cudaMalloc(&d_x, …); … cudaFree(d_x);` pattern in
// the request handler so cleanup happens implicitly on scope exit and
// exception paths.
//
// Move-only. `data()` returns the device pointer for kernel calls. Build
// from a host `std::span<const T>` via `from_host(...)` to alloc + memcpy
// in one call. Read back via `to_host()`.
//
// Not a general-purpose CUDA RAII layer — kept small and dependency-light
// so it can replace existing patterns mechanically. Streams are not
// modeled; all allocations / copies use the default stream, matching the
// pre-RAII code.

#include <cstddef>
#include <cstring>
#include <span>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_check.hpp"

namespace pie_cuda_driver {

template <class T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t count) {
        if (count > 0) {
            CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
            count_ = count;
        }
    }

    ~DeviceBuffer() noexcept { reset(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& o) noexcept
        : ptr_(o.ptr_), count_(o.count_) {
        o.ptr_ = nullptr;
        o.count_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) {
            reset();
            ptr_ = o.ptr_;
            count_ = o.count_;
            o.ptr_ = nullptr;
            o.count_ = 0;
        }
        return *this;
    }

    // Allocate `count` Ts, no copy. Useful for output buffers the kernel
    // will fill before D2H.
    static DeviceBuffer<T> alloc(std::size_t count) {
        return DeviceBuffer<T>(count);
    }

    // Allocate + upload. Returns an empty buffer when `host.empty()` so
    // callers can pass through without special-casing zero-length.
    static DeviceBuffer<T> from_host(std::span<const T> host) {
        DeviceBuffer<T> buf(host.size());
        if (!host.empty()) {
            CUDA_CHECK(cudaMemcpy(buf.ptr_, host.data(),
                                  host.size() * sizeof(T),
                                  cudaMemcpyHostToDevice));
        }
        return buf;
    }

    // Same as `from_host(span<const T>)` but takes a raw byte view for
    // the BPIQ wire-format case where the source array is encoded as a
    // bytes blob the schema decoder aliases to `T`. Length must be a
    // multiple of `sizeof(T)`.
    static DeviceBuffer<T> from_bytes(std::span<const std::uint8_t> bytes) {
        DeviceBuffer<T> buf(bytes.size() / sizeof(T));
        if (!bytes.empty()) {
            CUDA_CHECK(cudaMemcpy(buf.ptr_, bytes.data(), bytes.size(),
                                  cudaMemcpyHostToDevice));
        }
        return buf;
    }

    // Read back to a fresh host vector. Synchronous — issues against the
    // default stream and waits.
    std::vector<T> to_host() const {
        std::vector<T> result(count_);
        if (count_ > 0) {
            CUDA_CHECK(cudaMemcpy(result.data(), ptr_,
                                  count_ * sizeof(T),
                                  cudaMemcpyDeviceToHost));
        }
        return result;
    }

    T*       data()       noexcept { return ptr_; }
    const T* data() const noexcept { return ptr_; }
    std::size_t size()    const noexcept { return count_; }
    bool        empty()   const noexcept { return count_ == 0; }

    // Surrender ownership; caller is responsible for `cudaFree`.
    T* release() noexcept {
        T* tmp = ptr_;
        ptr_ = nullptr;
        count_ = 0;
        return tmp;
    }

private:
    void reset() noexcept {
        if (ptr_) {
            // Best effort — driver shutdown may have already torn the
            // context down; we don't surface errors from a destructor.
            cudaFree(ptr_);
            ptr_ = nullptr;
            count_ = 0;
        }
    }

    T* ptr_ = nullptr;
    std::size_t count_ = 0;
};

}  // namespace pie_cuda_driver

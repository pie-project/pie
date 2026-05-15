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
#include <stdexcept>
#include <string>
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
        : ptr_(o.ptr_), count_(o.count_), h_pinned_(o.h_pinned_) {
        o.ptr_ = nullptr;
        o.count_ = 0;
        o.h_pinned_ = nullptr;
    }

    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) {
            reset();
            ptr_ = o.ptr_;
            count_ = o.count_;
            h_pinned_ = o.h_pinned_;
            o.ptr_ = nullptr;
            o.count_ = 0;
            o.h_pinned_ = nullptr;
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
    // the wire-format case where the source array is encoded as a
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

    // Copy from a host span into the existing device allocation (no
    // alloc). Throws if `src.size() > size()`. Used by the persistent-
    // buffer path that pre-allocates capacity at startup and refills
    // contents per fire — gives kernels stable device pointers across
    // fires (a prerequisite for CUDA-graph capture).
    //
    // Issues against the default stream; the kernel queue is
    // in-order, so subsequent kernel launches see the new contents.
    //
    // Stages through a lazily-allocated pinned host buffer so the
    // `cudaMemcpyAsync` from `src` is truly async. Without the staging,
    // `cudaMemcpyAsync` on pageable host memory blocks the host until
    // CUDA's internal staging completes — adding ~1-2 ms per call. With
    // 13+ per-fire copies of this kind, that dominates the wall time at
    // small per-fire GPU work (~5× HtoD-vs-vllm gap shown in nsys
    // profiles). The pinned scratch is allocated at the buffer's full
    // capacity once on first use and reused thereafter.
    void copy_from_host(std::span<const T> src) {
        if (src.size() > count_) {
            throw std::runtime_error(
                "DeviceBuffer::copy_from_host: src size " +
                std::to_string(src.size()) + " > capacity " +
                std::to_string(count_));
        }
        if (src.empty()) return;
        ensure_pinned_staging();
        std::memcpy(h_pinned_, src.data(), src.size() * sizeof(T));
        CUDA_CHECK(cudaMemcpyAsync(ptr_, h_pinned_,
                                   src.size() * sizeof(T),
                                   cudaMemcpyHostToDevice));
    }

    // Same as `copy_from_host(span<const T>)` but takes a raw byte view —
    // the wire-format case where the source bytes alias `T`.
    // Length must be a multiple of `sizeof(T)`.
    void copy_from_bytes(std::span<const std::uint8_t> bytes) {
        if (bytes.size() / sizeof(T) > count_) {
            throw std::runtime_error(
                "DeviceBuffer::copy_from_bytes: src elements " +
                std::to_string(bytes.size() / sizeof(T)) +
                " > capacity " + std::to_string(count_));
        }
        if (bytes.empty()) return;
        ensure_pinned_staging();
        std::memcpy(h_pinned_, bytes.data(), bytes.size());
        CUDA_CHECK(cudaMemcpyAsync(ptr_, h_pinned_, bytes.size(),
                                   cudaMemcpyHostToDevice));
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
        if (h_pinned_) {
            cudaFreeHost(h_pinned_);
            h_pinned_ = nullptr;
        }
    }

    // Lazily allocate the pinned host staging buffer at the device
    // buffer's full capacity. Pinned alloc isn't cheap (single ~µs
    // syscall per buffer), but it's one-shot — amortised across all
    // fires that ever touch this buffer.
    void ensure_pinned_staging() {
        if (h_pinned_ != nullptr) return;
        if (count_ == 0) return;
        CUDA_CHECK(cudaMallocHost(&h_pinned_, count_ * sizeof(T)));
    }

    T* ptr_ = nullptr;
    std::size_t count_ = 0;
    T* h_pinned_ = nullptr;
};

}  // namespace pie_cuda_driver

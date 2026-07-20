#pragma once

// Owning RAII wrapper around `cudaMalloc`'d memory. Replaces the scattered
// raw `T* d_x = nullptr; cudaMalloc(&d_x, …); … cudaFree(d_x);` pattern in
// the executor so cleanup happens implicitly on scope exit and
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

#include <array>
#include <cstddef>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "runahead.hpp"

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
        : ptr_(o.ptr_),
          count_(o.count_),
          h_pinned_(o.h_pinned_),
          h_pinned_copy_done_(o.h_pinned_copy_done_),
          h_pinned_copy_pending_(o.h_pinned_copy_pending_),
          next_pinned_slot_(o.next_pinned_slot_) {
        o.ptr_ = nullptr;
        o.count_ = 0;
        o.h_pinned_.fill(nullptr);
        o.h_pinned_copy_done_.fill(nullptr);
        o.h_pinned_copy_pending_.fill(false);
        o.next_pinned_slot_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) {
            reset();
            ptr_ = o.ptr_;
            count_ = o.count_;
            h_pinned_ = o.h_pinned_;
            h_pinned_copy_done_ = o.h_pinned_copy_done_;
            h_pinned_copy_pending_ = o.h_pinned_copy_pending_;
            next_pinned_slot_ = o.next_pinned_slot_;
            o.ptr_ = nullptr;
            o.count_ = 0;
            o.h_pinned_.fill(nullptr);
            o.h_pinned_copy_done_.fill(nullptr);
            o.h_pinned_copy_pending_.fill(false);
            o.next_pinned_slot_ = 0;
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
    // profiles). Two pinned slots let the configured depth-2 runahead enqueue
    // the next fire without overwriting the previous fire's DMA source.
    void copy_from_host(std::span<const T> src) {
        if (src.size() > count_) {
            throw std::runtime_error(
                "DeviceBuffer::copy_from_host: src size " +
                std::to_string(src.size()) + " > capacity " +
                std::to_string(count_));
        }
        if (src.empty()) return;
        const std::size_t slot = acquire_pinned_staging();
        std::memcpy(h_pinned_[slot], src.data(), src.size() * sizeof(T));
        CUDA_CHECK(cudaMemcpyAsync(ptr_, h_pinned_[slot],
                                   src.size() * sizeof(T),
                                   cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(h_pinned_copy_done_[slot], nullptr));
        h_pinned_copy_pending_[slot] = true;
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
        const std::size_t slot = acquire_pinned_staging();
        std::memcpy(h_pinned_[slot], bytes.data(), bytes.size());
        CUDA_CHECK(cudaMemcpyAsync(ptr_, h_pinned_[slot], bytes.size(),
                                   cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(h_pinned_copy_done_[slot], nullptr));
        h_pinned_copy_pending_[slot] = true;
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
    // Single-sourced from runahead.hpp: a slot is held from its H2D
    // enqueue (on the legacy default stream, which serializes behind all
    // GPU work) until the copy retires, one cycle per in-flight wave.
    // The stale depth-2 pool re-serialized every wire-geometry fire's
    // ~13 copy_from_host calls once the run-ahead pipe filled.
    static constexpr std::size_t kPinnedStagingSlots = kUploadStagingDepth;

    void reset() noexcept {
        for (std::size_t slot = 0; slot < kPinnedStagingSlots; ++slot) {
            if (h_pinned_copy_pending_[slot]) {
                cudaEventSynchronize(h_pinned_copy_done_[slot]);
                h_pinned_copy_pending_[slot] = false;
            }
            if (h_pinned_copy_done_[slot]) {
                cudaEventDestroy(h_pinned_copy_done_[slot]);
                h_pinned_copy_done_[slot] = nullptr;
            }
        }
        if (ptr_) {
            // Best effort — driver shutdown may have already torn the
            // context down; we don't surface errors from a destructor.
            cudaFree(ptr_);
            ptr_ = nullptr;
            count_ = 0;
        }
        for (T*& pinned : h_pinned_) {
            if (pinned != nullptr) {
                cudaFreeHost(pinned);
                pinned = nullptr;
            }
        }
        next_pinned_slot_ = 0;
    }

    void ensure_pinned_staging(std::size_t slot) {
        if (h_pinned_[slot] != nullptr || count_ == 0) return;
        CUDA_CHECK(cudaMallocHost(
            &h_pinned_[slot], count_ * sizeof(T)));
        try {
            CUDA_CHECK(cudaEventCreateWithFlags(
                &h_pinned_copy_done_[slot], cudaEventDisableTiming));
        } catch (...) {
            cudaFreeHost(h_pinned_[slot]);
            h_pinned_[slot] = nullptr;
            throw;
        }
    }

    std::size_t acquire_pinned_staging() {
        const std::size_t slot = next_pinned_slot_;
        next_pinned_slot_ =
            (next_pinned_slot_ + 1) % kPinnedStagingSlots;
        ensure_pinned_staging(slot);
        if (h_pinned_copy_pending_[slot]) {
            CUDA_CHECK(cudaEventSynchronize(
                h_pinned_copy_done_[slot]));
            h_pinned_copy_pending_[slot] = false;
        }
        return slot;
    }

    T* ptr_ = nullptr;
    std::size_t count_ = 0;
    std::array<T*, kPinnedStagingSlots> h_pinned_{};
    std::array<cudaEvent_t, kPinnedStagingSlots>
        h_pinned_copy_done_{};
    std::array<bool, kPinnedStagingSlots>
        h_pinned_copy_pending_{};
    std::size_t next_pinned_slot_ = 0;
};

template <class T>
class PinnedHostBuffer {
public:
    PinnedHostBuffer() = default;

    explicit PinnedHostBuffer(std::size_t count) {
        if (count > 0) {
            CUDA_CHECK(cudaMallocHost(&ptr_, count * sizeof(T)));
            count_ = count;
        }
    }

    ~PinnedHostBuffer() noexcept { reset(); }

    PinnedHostBuffer(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;

    PinnedHostBuffer(PinnedHostBuffer&& o) noexcept
        : ptr_(o.ptr_), count_(o.count_) {
        o.ptr_ = nullptr;
        o.count_ = 0;
    }

    PinnedHostBuffer& operator=(PinnedHostBuffer&& o) noexcept {
        if (this != &o) {
            reset();
            ptr_ = o.ptr_;
            count_ = o.count_;
            o.ptr_ = nullptr;
            o.count_ = 0;
        }
        return *this;
    }

    static PinnedHostBuffer<T> alloc(std::size_t count) {
        return PinnedHostBuffer<T>(count);
    }

    T* data() { return ptr_; }
    const T* data() const { return ptr_; }
    std::size_t size() const { return count_; }
    bool empty() const { return count_ == 0; }

    T& operator[](std::size_t i) { return ptr_[i]; }
    const T& operator[](std::size_t i) const { return ptr_[i]; }

    void reset() noexcept {
        if (ptr_) {
            cudaFreeHost(ptr_);
            ptr_ = nullptr;
        }
        count_ = 0;
    }

private:
    T* ptr_ = nullptr;
    std::size_t count_ = 0;
};

}  // namespace pie_cuda_driver

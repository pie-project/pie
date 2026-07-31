#pragma once

// RAII helpers for thread_local CUDA scratch. TP ranks are threads that each
// own a device; when the thread exits, these destructors free memory / handles
// while that thread's CUDA context is still the intended one. Best-effort:
// if the context is already gone (process teardown), free is skipped.

#include <cuda_runtime.h>

#include <cstddef>
#include <utility>

namespace pie_cuda_driver {

inline int tls_cuda_current_device() noexcept
{
    int d = -1;
    return (cudaGetDevice(&d) == cudaSuccess) ? d : -1;
}

inline void tls_cuda_free_device(void* p, int device) noexcept
{
    if (p == nullptr) return;
    int prev = -1;
    if (device >= 0) {
        if (cudaGetDevice(&prev) != cudaSuccess) return;
        if (cudaSetDevice(device) != cudaSuccess) return;
    }
    (void)cudaFree(p);
    if (device >= 0 && prev >= 0) (void)cudaSetDevice(prev);
}

inline void tls_cuda_free_host(void* p) noexcept
{
    if (p != nullptr) (void)cudaFreeHost(p);
}

// Growable device buffer owned by a thread_local.
template <typename T>
struct TlsDeviceBuf {
    T* ptr = nullptr;
    std::size_t capacity = 0;  // element count
    int device = -1;

    TlsDeviceBuf() = default;
    TlsDeviceBuf(const TlsDeviceBuf&) = delete;
    TlsDeviceBuf& operator=(const TlsDeviceBuf&) = delete;
    TlsDeviceBuf(TlsDeviceBuf&& o) noexcept
        : ptr(o.ptr), capacity(o.capacity), device(o.device)
    {
        o.ptr = nullptr;
        o.capacity = 0;
        o.device = -1;
    }
    TlsDeviceBuf& operator=(TlsDeviceBuf&& o) noexcept
    {
        if (this == &o) return *this;
        reset();
        ptr = o.ptr;
        capacity = o.capacity;
        device = o.device;
        o.ptr = nullptr;
        o.capacity = 0;
        o.device = -1;
        return *this;
    }

    ~TlsDeviceBuf() { reset(); }

    void reset() noexcept
    {
        tls_cuda_free_device(ptr, device);
        ptr = nullptr;
        capacity = 0;
        device = -1;
    }

    // Ensure capacity >= want_elems. Throws cudaError via CUDA_CHECK only when
    // the caller wraps malloc; here returns false on alloc failure so .cu
    // files without CUDA_CHECK can decide.
    bool ensure(std::size_t want_elems)
    {
        if (want_elems <= capacity) return ptr != nullptr || want_elems == 0;
        reset();
        if (want_elems == 0) return true;
        if (cudaMalloc(&ptr, want_elems * sizeof(T)) != cudaSuccess) {
            ptr = nullptr;
            return false;
        }
        capacity = want_elems;
        device = tls_cuda_current_device();
        return true;
    }
};

// Growable pinned host buffer owned by a thread_local.
template <typename T>
struct TlsHostPinnedBuf {
    T* ptr = nullptr;
    std::size_t capacity = 0;

    TlsHostPinnedBuf() = default;
    TlsHostPinnedBuf(const TlsHostPinnedBuf&) = delete;
    TlsHostPinnedBuf& operator=(const TlsHostPinnedBuf&) = delete;

    ~TlsHostPinnedBuf() { reset(); }

    void reset() noexcept
    {
        tls_cuda_free_host(ptr);
        ptr = nullptr;
        capacity = 0;
    }

    bool ensure(std::size_t want_elems)
    {
        if (want_elems <= capacity) return ptr != nullptr || want_elems == 0;
        reset();
        if (want_elems == 0) return true;
        if (cudaMallocHost(&ptr, want_elems * sizeof(T)) != cudaSuccess) {
            ptr = nullptr;
            return false;
        }
        capacity = want_elems;
        return true;
    }
};

}  // namespace pie_cuda_driver

// Activation dumping for cross-engine numerical debugging.
//
// Set `PIE_ACT_DUMP_DIR` to a directory and every instrumented forward writes
// its intermediate tensors there as float32 `.npy` files named
// `s<step>_<tag>.npy`, where <step> counts calls to `act_dump_step_begin()`
// (i.e. forward passes). `benches/act_parity.py` produces the matching dump
// from vLLM and reports the first tag that disagrees, which is how a
// "pie and vLLM emit different tokens" report gets turned into a specific
// broken kernel.
//
// This is a debugging facility, not a runtime feature: when the variable is
// unset every entry point is a predictable branch on a cached bool, and when
// it is set the dump synchronises the stream and round-trips through host
// memory. Never enable it in a measured run.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace pie_cuda_driver::model {

inline const char* act_dump_dir() {
    static const char* dir = [] {
        const char* v = std::getenv("PIE_ACT_DUMP_DIR");
        return (v != nullptr && v[0] != '\0') ? v : nullptr;
    }();
    return dir;
}

inline bool act_dump_enabled() { return act_dump_dir() != nullptr; }

/// The CUDA device this thread is on, as a filename suffix.
///
/// Every rank of a TP group runs in one process and writes to one directory,
/// so without this they collide: the file that survives is whichever rank
/// finished last, and a dump that silently mixes ranks is worse than no dump.
/// It also makes the ranks comparable to each other, which is the question
/// worth asking about a tensor the model replicates.
inline int act_dump_device() {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return 0;
    return device;
}

inline int& act_dump_step() {
    // Per-thread, because the embedded TP launcher runs every rank as a thread
    // in ONE process. A process-wide counter would be advanced once per rank
    // per forward, so `PIE_ACT_DUMP_STEPS=1` would stop dumping partway through
    // the very first prefill at tp=8.
    static thread_local int step = -1;
    return step;
}

/// Starts a new forward pass. Tags written after this call are prefixed with
/// the new step index; the first pass of a request is step 0 (the prefill).
inline bool act_dump_capturing(cudaStream_t stream);

/// Graph-capture passes must not consume a step index: the upfront capture
/// lattice runs the body several times before the first real prefill, which
/// would otherwise push the step counter past PIE_ACT_DUMP_STEPS.
inline void act_dump_step_begin(cudaStream_t stream = nullptr) {
    if (act_dump_enabled() && !act_dump_capturing(stream)) ++act_dump_step();
}

/// True while the calling thread is recording into a CUDA graph. Dumping
/// there is illegal (it synchronises and round-trips through host memory)
/// and poisons the capture, so every entry point bails out.
inline bool act_dump_capturing(cudaStream_t stream) {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &status) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return status != cudaStreamCaptureStatusNone;
}

/// Only dump the first `PIE_ACT_DUMP_STEPS` passes (default 1: prefill only).
/// Later decode steps would otherwise bury the prefill under hundreds of files.
inline bool act_dump_active() {
    if (!act_dump_enabled()) return false;
    constexpr int limit = 1;
    const int step = act_dump_step();
    return step >= 0 && step < limit;
}

inline void act_dump_write_npy(
    const std::string& tag, const std::vector<float>& host,
    long rows, long cols) {
    char path[1024];
    std::snprintf(path, sizeof(path), "%s/d%d_s%d_%s.npy",
                  act_dump_dir(), act_dump_device(), act_dump_step(), tag.c_str());
    std::FILE* f = std::fopen(path, "wb");
    if (f == nullptr) {
        std::fprintf(stderr, "[pie-act-dump] cannot open %s\n", path);
        return;
    }
    char header[128];
    const int n = std::snprintf(
        header, sizeof(header),
        "{'descr': '<f4', 'fortran_order': False, 'shape': (%ld, %ld), }",
        rows, cols);
    // v1.0 requires the magic+header to be a multiple of 64 bytes.
    int total = 10 + n + 1;
    const int pad = (64 - (total % 64)) % 64;
    const std::uint16_t header_len =
        static_cast<std::uint16_t>(n + 1 + pad);
    std::fwrite("\x93NUMPY\x01\x00", 1, 8, f);
    std::fwrite(&header_len, sizeof(header_len), 1, f);
    std::fwrite(header, 1, static_cast<std::size_t>(n), f);
    for (int i = 0; i < pad; ++i) std::fputc(' ', f);
    std::fputc('\n', f);
    std::fwrite(host.data(), sizeof(float), host.size(), f);
    std::fclose(f);
}

inline void act_dump_bf16(
    const char* tag, const void* device_ptr, long rows, long cols,
    cudaStream_t stream) {
    if (!act_dump_active() || act_dump_capturing(stream) ||
        device_ptr == nullptr || rows <= 0 || cols <= 0) {
        return;
    }
    const std::size_t n = static_cast<std::size_t>(rows) * cols;
    std::vector<std::uint16_t> raw(n);
    cudaStreamSynchronize(stream);
    if (cudaMemcpy(raw.data(), device_ptr, n * sizeof(std::uint16_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return;
    }
    std::vector<float> host(n);
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint32_t bits = static_cast<std::uint32_t>(raw[i]) << 16;
        std::memcpy(&host[i], &bits, sizeof(float));
    }
    act_dump_write_npy(tag, host, rows, cols);
}

inline void act_dump_f32(
    const char* tag, const void* device_ptr, long rows, long cols,
    cudaStream_t stream) {
    if (!act_dump_active() || act_dump_capturing(stream) ||
        device_ptr == nullptr || rows <= 0 || cols <= 0) {
        return;
    }
    const std::size_t n = static_cast<std::size_t>(rows) * cols;
    std::vector<float> host(n);
    cudaStreamSynchronize(stream);
    if (cudaMemcpy(host.data(), device_ptr, n * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return;
    }
    act_dump_write_npy(tag, host, rows, cols);
}

inline void act_dump_i32(
    const char* tag, const void* device_ptr, long rows, long cols,
    cudaStream_t stream) {
    if (!act_dump_active() || act_dump_capturing(stream) ||
        device_ptr == nullptr || rows <= 0 || cols <= 0) {
        return;
    }
    const std::size_t n = static_cast<std::size_t>(rows) * cols;
    std::vector<std::int32_t> raw(n);
    cudaStreamSynchronize(stream);
    if (cudaMemcpy(raw.data(), device_ptr, n * sizeof(std::int32_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return;
    }
    std::vector<float> host(n);
    for (std::size_t i = 0; i < n; ++i) {
        host[i] = static_cast<float>(raw[i]);
    }
    act_dump_write_npy(tag, host, rows, cols);
}

inline std::string act_dump_layer_tag(const char* what, int layer) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "L%02d_%s", layer, what);
    return std::string(buf);
}

}  // namespace pie_cuda_driver::model

#pragma once

// Single source of truth for the CUDA weight loader's tunable constants and the
// PIE_CUDA_* environment knobs. Every default lives here, and every knob is read
// through one of the helpers below so the parse convention is stated once rather
// than re-spelled at each call site. Conventions:
//   env_present(name)        -> set to anything            (TRACE/DEBUG-style toggles)
//   env_truthy(name)         -> set, non-empty, not "0"    (DISABLE_*/ENABLE_* knobs)
//   env_u64(name, fallback)  -> parsed if set and > 0, else fallback   (reject 0)
//   env_u64_or(name,fallback)-> fallback only if unset; if set, the parsed value (0 ok)

#include <cstdint>
#include <cstdlib>

namespace pie_cuda_driver::loader_config {

inline constexpr std::uint64_t kKiB = 1024ull;
inline constexpr std::uint64_t kMiB = 1024ull * 1024ull;

// --- copy engine ---
inline constexpr std::size_t kCopyStreamsDefault = 8;
inline constexpr std::size_t kCopyStreamsMax = 32;
inline constexpr std::size_t kBatchChunk = 1024;       // tensors per batched H2D call
inline constexpr std::size_t kMaxPendingCopies = 512;  // pending-copy flush threshold

// --- pinned staging pool ---
inline constexpr std::uint64_t kPinnedMinBytesDefault = 256ull * kKiB;
inline constexpr std::uint64_t kPinnedPoolBytesDefault = 512ull * kMiB;
inline constexpr std::size_t kPinnedSlotsMax = 128;

// --- parallel host reader lanes ---
inline constexpr std::size_t kReaderThreadsDefault = 4;
// Per-lane double-buffer size. The staged H2D is host-memcpy-bound (mmap page
// cache -> pinned -> DMA), so a large buffer buys no transfer throughput once
// the chunk clears the PCIe small-copy cliff (~1 MiB); it only inflates the
// one-time cudaMallocHost page-locking cost, which is paid on every load
// (lanes*2*buf pinned). Profiling on an L40 (Qwen3-1.7B, page cache warm):
// 32 MiB -> ~665 ms load (256 MiB pinned, ~158 ms page-lock, high variance);
// 2 MiB -> ~268 ms load (16 MiB pinned, ~12 ms page-lock) at the same
// ~13-15 GiB/s copy rate. 2 MiB is also robust for many-small-tensor (FP8)
// checkpoints where 1 MiB starts to fragment.
inline constexpr std::uint64_t kReaderBufBytesDefault = 2ull * kMiB;

// --- transcode / quant ---
inline constexpr std::uint64_t kFallbackTileBytes = 64ull * kMiB;  // when max_tile_bytes==0
inline constexpr int kE8M0Bias = 127;                              // E8M0 exponent bias
inline constexpr int kMxfp4Group = 32;        // MXFP4 values per E8M0 block
inline constexpr int kMxfp4PackedPerByte = 2;  // E2M1 nibbles packed per output byte
// (host mirror of kernels::transcode::EncodeMxfp4::{kGroup, kPackedPerByte})

inline bool env_present(const char* name)
{
    return std::getenv(name) != nullptr;
}

inline bool env_truthy(const char* name)
{
    const char* v = std::getenv(name);
    return v != nullptr && v[0] != '\0' && v[0] != '0';
}

inline std::uint64_t env_u64(const char* name, std::uint64_t fallback)
{
    if (const char* v = std::getenv(name)) {
        const unsigned long long parsed = std::strtoull(v, nullptr, 10);
        if (parsed > 0) {
            return static_cast<std::uint64_t>(parsed);
        }
    }
    return fallback;
}

inline std::uint64_t env_u64_or(const char* name, std::uint64_t fallback)
{
    const char* v = std::getenv(name);
    return v == nullptr ? fallback
                        : static_cast<std::uint64_t>(std::strtoull(v, nullptr, 10));
}

// Parallel host-reader-lane knobs, shared by the storage executor's bulk flush
// and the artifact-cache restore so both stage H2D the same way. A lane count of
// 0 means "no reader lanes" (the executor's direct mmap->device fallback).
inline std::size_t reader_lane_count()
{
    return static_cast<std::size_t>(
        env_u64_or("PIE_CUDA_WEIGHT_READER_THREADS", kReaderThreadsDefault));
}

inline std::uint64_t reader_buf_bytes()
{
    const std::uint64_t mb = env_u64("PIE_CUDA_WEIGHT_READER_BUF_MB", 0);
    return mb != 0 ? mb * kMiB : kReaderBufBytesDefault;
}

}  // namespace pie_cuda_driver::loader_config

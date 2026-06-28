#pragma once

// Canonical program hash — FNV-1a 64-bit over raw bytecode.
//
// This is THE program identity used everywhere: the `get_or_compile` /
// #9-compile-cache key, the driver `ProgramHandle`, and the #12 driver-side
// program→kind recognizer (`program_recognizer.hpp`). One impl, header-inline
// (a trivial pure function — no CUDA), so host-only consumers (the recognizer
// table + its round-trip test) need not link the CUDA JIT TU. Mirrors the
// canonical Rust `pie_sampling_ir::program_hash` (same FNV-1a) so SDK-emitted
// bytecode and driver-baked bytecode hash identically — the basis of #12's
// behavior-preserving recognition.

#include <cstddef>
#include <cstdint>

namespace pie_cuda_driver::sampling_ir::jit {

// FNV-1a 64-bit hash of `len` bytes at `data`.
inline std::uint64_t fnv1a64(const void* data, std::size_t len) {
    const auto* p = static_cast<const unsigned char*>(data);
    std::uint64_t h = 0xcbf29ce484222325ULL;
    for (std::size_t i = 0; i < len; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

}  // namespace pie_cuda_driver::sampling_ir::jit

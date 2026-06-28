#pragma once

// THE single-source program-identity hash — the shared key for the cost /
// grouping / dedup contract. THREE consumers key on this identical value (one
// helper, no per-consumer re-derivation, no drift):
//
//   * #11 JIT compile-dedup  — this == the engine's `keyed`/`ProgramHandle`,
//                              minus the process-constant `batched` bit.
//   * echo's M-batch fire-grouping — two requests share one `num_rows=N` kernel
//                              iff they have the same `program_identity_hash`.
//   * alpha's #10 distinct-program count in the accumulation-window policy.
//
// Partition granularity = (bytecode, manifest). v4 bytecode is binding-free, so
// the manifest (bindings) IS part of program identity: same bytecode + DIFFERENT
// bindings = a DISTINCT compiled program = a distinct fire. Keying on raw
// `fnv1a64(bytecode)` alone would under-count/under-group that case → the silent
// mis-pricing (policy assumes a coalesce the driver won't do).
//
// The M=1-vs-M-batch `batched` lowering bit is intentionally NOT part of this
// identity: it is process-constant (a constant XOR offset that never changes the
// equality classes) AND it is a downstream lowering artifact of the coalesce
// decision, not an input to it. #11's `keyed` folds it on top for its compile
// cache (M=1 vs M-batch PTX are distinct artifacts); each group's M-batched fire
// is one compile, so the layering is consistent.
//
// Header-inline + pure (no CUDA), so any C++ consumer (driver JIT, executor
// M-batch grouping) can key on it without linking the JIT TU.

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "sampling_ir/codegen.hpp"       // ProgramManifest, InputBind
#include "sampling_ir/program_hash.hpp"  // jit::fnv1a64

namespace pie_cuda_driver::sampling_ir {

// Hash the v4 binding manifest. v4 bytecode is binding-free, so two programs
// that share bytecode but differ in their bindings are distinct compiled
// programs. Byte-exact with the legacy jit_backend cache-key folding (kind,
// readiness, host_key per slot, in slot order).
inline std::uint64_t manifest_hash(const ProgramManifest& manifest) {
    std::vector<std::uint8_t> buf;
    buf.reserve(manifest.size() * 6);
    for (const InputBind& b : manifest) {
        buf.push_back(static_cast<std::uint8_t>(b.kind));
        buf.push_back(static_cast<std::uint8_t>(b.ready));
        const std::uint32_t k = b.host_key;
        buf.push_back(static_cast<std::uint8_t>(k & 0xFFu));
        buf.push_back(static_cast<std::uint8_t>((k >> 8) & 0xFFu));
        buf.push_back(static_cast<std::uint8_t>((k >> 16) & 0xFFu));
        buf.push_back(static_cast<std::uint8_t>((k >> 24) & 0xFFu));
    }
    return jit::fnv1a64(buf.data(), buf.size());
}

// THE single-source program identity (see the header banner). Equals the engine
// dedup key / ProgramHandle without the process-constant batched bit:
//   program_identity_hash = fnv1a64(bytecode) ^ manifest_hash(manifest)
inline std::uint64_t program_identity_hash(std::span<const std::uint8_t> bytecode,
                                           const ProgramManifest& manifest) {
    std::uint64_t h = jit::fnv1a64(bytecode.data(), bytecode.size());
    if (!manifest.empty()) h ^= manifest_hash(manifest);
    return h;
}

}  // namespace pie_cuda_driver::sampling_ir

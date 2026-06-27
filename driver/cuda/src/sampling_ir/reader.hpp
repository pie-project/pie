#pragma once

// PSIR v2/v3 bytecode reader: decodes a flat little-endian program buffer
// (interface/sampling-ir/BYTECODE.md) into the in-memory IR graph (ir.hpp).
//
// The reader is a forward, bounds-checked cursor (BYTECODE.md §7). It performs
// *structural* validation only — magic / version / EOF / unknown tag bytes —
// because the host already runs pie-sampling-ir's full semantic validator
// before submit. It is exception-free: failures set `err` and return false.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "sampling_ir/ir.hpp"

namespace pie_cuda_driver::sampling_ir {

struct DecodeError {
    enum Code {
        None,
        BadMagic,
        BadVersion,
        UnexpectedEof,
        UnknownOpcode,
        UnknownTag,    // bad dtype / shape / binding / predicate / rng / intrinsic / avail tag
        BadInputId,    // Input.id != its index
    };
    Code code = None;
    std::string detail;
};

// Decode `len` bytes at `data` into `out`. On failure returns false and (if
// non-null) fills `err`. On success `out` is fully populated and op result ids
// are assigned per the SSA counter rule (§3).
bool decode(const std::uint8_t* data, std::size_t len, Program& out, DecodeError* err = nullptr);

// Decode binding-free PSIR v4 bytecode + its per-slot attach manifest
// (`slot_bindings`, one Binding per input slot in slot order — Intrinsic logits
// or Host tensor) into the in-memory Program. Shims the v4 flat/Input-op SSA
// form into the inputs-first representation the codegen consumes.
bool decode_v4(const std::uint8_t* data, std::size_t len,
               const std::vector<Binding>& slot_bindings, Program& out, DecodeError* err = nullptr);

// #12 phase-2 op-shape canonicalization. Returns a copy of the program bytecode
// with every `PivotThreshold` `RankLe(k)` predicate immediate zeroed (RankLe(0)),
// so the k-bearing standard samplers (top-k / top-k-top-p) — whose ONLY parametric
// byte is the baked `k` — hash to one k-invariant value regardless of `k`. Mirrors
// `pie_sampling_ir::canonicalize_op_shape` (zero the immediate, not re-encode —
// equivalent because the wire is canonical `ir::encode`). The value-id predicates
// `CummassLe`/`ProbGe` are left untouched (precise, no false-match). A non-v4 /
// malformed buffer is returned unchanged.
std::vector<std::uint8_t> canonicalize_op_shape(const std::uint8_t* data, std::size_t len);

// Extract the baked top-k cutoff `k` (the single `RankLe` immediate) from a
// k-bearing standard program's op-shape. `nullopt` if no `RankLe` is present.
std::optional<std::uint32_t> extract_rank_le_k(const std::uint8_t* data, std::size_t len);

}  // namespace pie_cuda_driver::sampling_ir

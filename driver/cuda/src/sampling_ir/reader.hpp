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

}  // namespace pie_cuda_driver::sampling_ir

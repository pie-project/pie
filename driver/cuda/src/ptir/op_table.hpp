#pragma once

// PTIR first-party op table — the CUDA driver's mirror of the closed core op
// set (docs/ptir/overview.md appendix). This is the tier-0/1/2 backend's view
// of the op set: opcode bytes + per-op metadata (family, arity, result kind).
//
// SOURCE OF TRUTH: echo owns the canonical op-table (a shared Rust crate +
// generated C++ header, thrust-3 P0.1). Until that lands, this header is the
// PROVISIONAL mirror the CUDA tier-0 library builds against; the opcode bytes
// are grouped by family with gaps so echo's frozen byte assignments slot in
// mechanically. Every op here comes from the overview appendix, nothing else
// (D5: the first-party core is closed and fusable).
//
// Deliberately header-only and dependency-free (no CUDA, no std beyond
// <cstdint>/<string_view>) so it compiles into host tools, the reference
// evaluator, NVRTC codegen, and .cu kernels alike.

#include <cstdint>
#include <string_view>

namespace pie_cuda_driver::ptir {

// Element dtype. Wire tags frozen to match the Sampling-IR mirror
// (sampling_ir/ir.hpp): F32=0, I32=1, U32=2, Bool=3. `bool` travels packed on
// the wire (D1) but is materialized one-byte per element on device.
enum class DType : std::uint8_t { F32 = 0, I32 = 1, U32 = 2, Bool = 3 };

inline constexpr std::uint32_t dtype_size(DType d) {
    switch (d) {
        case DType::F32: return 4;
        case DType::I32: return 4;
        case DType::U32: return 4;
        case DType::Bool: return 1;
    }
    return 4;
}

inline constexpr bool dtype_is_float(DType d) { return d == DType::F32; }
inline constexpr bool dtype_is_int(DType d) { return d == DType::I32 || d == DType::U32; }

// The closed first-party core (overview appendix). Bytes are grouped by family
// on 0x10 boundaries with intra-family gaps; echo's generated table may reassign
// them, at which point this enum is regenerated from that single source.
enum class OpCode : std::uint8_t {
    // map — element-wise, broadcasting by shape
    Add  = 0x01, Sub = 0x02, Mul = 0x03, Div = 0x04, Rem = 0x05,
    Neg  = 0x06, Exp = 0x07, Log = 0x08, Cast = 0x09,

    // compare / logic — bool results (packed on the wire, D1)
    Eq = 0x10, Ne = 0x11, Lt = 0x12, Le = 0x13, Gt = 0x14, Ge = 0x15,
    And = 0x16, Or = 0x17, Not = 0x18,

    // choice — the data-dependent branch (§2)
    Select = 0x20,

    // shape — metadata only, no data movement (Broadcast/Transpose materialize
    // in a buffer backend; Reshape is a pure alias)
    Reshape = 0x28, Broadcast = 0x29, Transpose = 0x2A,

    // index — scatter_set duplicates resolve in index order, last wins (§6.2)
    Iota = 0x30, Gather = 0x31, ScatterSet = 0x32,

    // reduce / scan — row-local over the last axis (§7.3)
    ReduceSum = 0x40, ReduceMax = 0x41, ReduceArgmax = 0x42,
    CumSum = 0x43, CumProd = 0x44,

    // normalize — composed reduce + map; fuse in tier 1
    Softmax = 0x50, LogSoftmax = 0x51, L2Norm = 0x52,

    // order — rank_le(k) is the rank predicate pivot_threshold cuts at; top_k
    // links as a library kernel (§7.3), never generated
    TopK = 0x60, PivotThreshold = 0x61, RankLe = 0x62,

    // linear — the core's one GEMM; a library kernel (§7.3)
    Matmul = 0x70,

    // sampling — composed over rng state + map (§3, §4); replay-deterministic
    Gumbel = 0x78, MaskApply = 0x79,
};

enum class OpFamily : std::uint8_t {
    Map, Compare, Choice, Shape, Index, Reduce, Normalize, Order, Linear, Sampling,
};

// Result-element kind an op yields regardless of input dtype:
//   SameAsInput — preserves the first operand's dtype (map/select/reduce_sum…)
//   Bool        — compare/logic always yield bool
//   Index       — reduce_argmax / iota / rank predicates yield u32 indices
//   Custom      — dtype is carried explicitly on the op (cast, gather, top_k…)
enum class ResultKind : std::uint8_t { SameAsInput, Bool, Index, Custom };

// How the op maps threads to data — the tier-0 launcher reads this to pick a
// launch shape, and tier-1 fusion reads it to decide the cut points.
//   Elementwise — grid-stride over numel; a pure per-element map
//   RowLocal    — one CTA per row; a reduction/scan/normalize over the last axis
//   Gather      — grid-stride over the index/output length
//   Scatter     — grid-stride over the source length; last-writer-wins
//   Library     — a prebuilt library kernel (top_k, matmul); never generated
//   Materialize — a shape op that copies with an index remap (broadcast/transpose)
//   Alias       — a pure metadata op (reshape); no kernel launched
enum class LaunchClass : std::uint8_t {
    Elementwise, RowLocal, Gather, Scatter, Library, Materialize, Alias,
};

struct OpInfo {
    OpCode      code;
    OpFamily    family;
    LaunchClass launch;
    ResultKind  result;
    std::uint8_t arity;      // number of value operands
    std::string_view name;   // canonical spelling (matches the overview)
};

// Metadata lookup. constexpr so it folds at compile time in both host tools and
// device-side codegen. Returns a sentinel (arity 0xFF) for an unknown byte.
inline constexpr OpInfo op_info(OpCode c) {
    using F = OpFamily;
    using L = LaunchClass;
    using R = ResultKind;
    switch (c) {
        case OpCode::Add:  return {c, F::Map, L::Elementwise, R::SameAsInput, 2, "add"};
        case OpCode::Sub:  return {c, F::Map, L::Elementwise, R::SameAsInput, 2, "sub"};
        case OpCode::Mul:  return {c, F::Map, L::Elementwise, R::SameAsInput, 2, "mul"};
        case OpCode::Div:  return {c, F::Map, L::Elementwise, R::SameAsInput, 2, "div"};
        case OpCode::Rem:  return {c, F::Map, L::Elementwise, R::SameAsInput, 2, "rem"};
        case OpCode::Neg:  return {c, F::Map, L::Elementwise, R::SameAsInput, 1, "neg"};
        case OpCode::Exp:  return {c, F::Map, L::Elementwise, R::SameAsInput, 1, "exp"};
        case OpCode::Log:  return {c, F::Map, L::Elementwise, R::SameAsInput, 1, "log"};
        case OpCode::Cast: return {c, F::Map, L::Elementwise, R::Custom,      1, "cast"};

        case OpCode::Eq: return {c, F::Compare, L::Elementwise, R::Bool, 2, "eq"};
        case OpCode::Ne: return {c, F::Compare, L::Elementwise, R::Bool, 2, "ne"};
        case OpCode::Lt: return {c, F::Compare, L::Elementwise, R::Bool, 2, "lt"};
        case OpCode::Le: return {c, F::Compare, L::Elementwise, R::Bool, 2, "le"};
        case OpCode::Gt: return {c, F::Compare, L::Elementwise, R::Bool, 2, "gt"};
        case OpCode::Ge: return {c, F::Compare, L::Elementwise, R::Bool, 2, "ge"};
        case OpCode::And: return {c, F::Compare, L::Elementwise, R::Bool, 2, "and"};
        case OpCode::Or:  return {c, F::Compare, L::Elementwise, R::Bool, 2, "or"};
        case OpCode::Not: return {c, F::Compare, L::Elementwise, R::Bool, 1, "not"};

        case OpCode::Select: return {c, F::Choice, L::Elementwise, R::SameAsInput, 3, "select"};

        case OpCode::Reshape:   return {c, F::Shape, L::Alias,       R::SameAsInput, 1, "reshape"};
        case OpCode::Broadcast: return {c, F::Shape, L::Materialize, R::SameAsInput, 1, "broadcast"};
        case OpCode::Transpose: return {c, F::Shape, L::Materialize, R::SameAsInput, 1, "transpose"};

        case OpCode::Iota:       return {c, F::Index, L::Elementwise, R::Index,       0, "iota"};
        case OpCode::Gather:     return {c, F::Index, L::Gather,      R::SameAsInput, 2, "gather"};
        case OpCode::ScatterSet: return {c, F::Index, L::Scatter,     R::SameAsInput, 3, "scatter_set"};

        case OpCode::ReduceSum:    return {c, F::Reduce, L::RowLocal, R::SameAsInput, 1, "reduce_sum"};
        case OpCode::ReduceMax:    return {c, F::Reduce, L::RowLocal, R::SameAsInput, 1, "reduce_max"};
        case OpCode::ReduceArgmax: return {c, F::Reduce, L::RowLocal, R::Index,       1, "reduce_argmax"};
        case OpCode::CumSum:       return {c, F::Reduce, L::RowLocal, R::SameAsInput, 1, "cumsum"};
        case OpCode::CumProd:      return {c, F::Reduce, L::RowLocal, R::SameAsInput, 1, "cumprod"};

        case OpCode::Softmax:    return {c, F::Normalize, L::RowLocal, R::SameAsInput, 1, "softmax"};
        case OpCode::LogSoftmax: return {c, F::Normalize, L::RowLocal, R::SameAsInput, 1, "log_softmax"};
        case OpCode::L2Norm:     return {c, F::Normalize, L::RowLocal, R::SameAsInput, 1, "l2norm"};

        case OpCode::TopK:           return {c, F::Order, L::Library,     R::Custom,      1, "top_k"};
        case OpCode::PivotThreshold: return {c, F::Order, L::RowLocal,    R::SameAsInput, 1, "pivot_threshold"};
        case OpCode::RankLe:         return {c, F::Order, L::RowLocal,    R::Bool,        1, "rank_le"};

        case OpCode::Matmul: return {c, F::Linear, L::Library, R::SameAsInput, 2, "matmul"};

        case OpCode::Gumbel:    return {c, F::Sampling, L::Elementwise, R::SameAsInput, 1, "gumbel"};
        case OpCode::MaskApply: return {c, F::Sampling, L::Elementwise, R::SameAsInput, 2, "mask_apply"};
    }
    return {c, OpFamily::Map, LaunchClass::Alias, ResultKind::Custom, 0xFF, "?"};
}

inline constexpr bool op_is_known(OpCode c) { return op_info(c).arity != 0xFF; }
inline constexpr std::string_view op_name(OpCode c) { return op_info(c).name; }

}  // namespace pie_cuda_driver::ptir

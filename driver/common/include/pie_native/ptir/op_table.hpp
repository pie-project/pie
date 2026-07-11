#pragma once

// PTIR op table — the shared pure-host view of the closed first-party op set
// (CUDA-free; consumed by both the CUDA and Metal drivers).
//
// SOURCE OF TRUTH: echo's generated `ptir_abi.h` (from pie-sampling-ir
// `src/ptir/{op,registry}.rs`) — op tag bytes come from its `PTIR_OP_*`
// constants, NOT hand-copied (echo P0.1 rule: "include the header, don't
// hand-copy ids"). Container byte layout: `interface/sampling-ir/PTIR-CONTAINER.md`.
// The header is currently vendored beside this file; once it lands on dev under
// `interface/sampling-ir/include/`, swap the include to that path (byte-identical,
// so it is a no-op sync).
//
// This header adds the driver-side annotations echo's table does not carry:
// per-op OpFamily / LaunchClass / ResultKind, which drive tier-0 launch-shape
// selection and tier-1 fusion cut points. Composite ops (softmax, gumbel, …) are
// NOT container ops — the container carries their core-op expansion (echo's
// `expand.rs`); we keep them as tier-1 fused-kernel forms at private tags 0xE0+.

#include <cstdint>
#include <string_view>

#include "pie_native/ptir/ptir_abi.h"   // PTIR_OP_* tag constants, PtirDType/Stage/Port/… enums

namespace pie_native::ptir {

// Element dtype — values mirror PtirDType (F32=0,I32=1,U32=2,Bool=3, Act=4).
// Act is a channel-decl-only late-bound activation dtype; program ops see F32.
enum class DType : std::uint8_t {
    F32 = PTIR_DT_F32, I32 = PTIR_DT_I32, U32 = PTIR_DT_U32, Bool = PTIR_DT_BOOL, Act = PTIR_DT_ACT,
};

inline constexpr std::uint32_t dtype_size(DType d) {
    switch (d) {
        case DType::F32: case DType::I32: case DType::U32: return 4;
        case DType::Bool: return 1;
        case DType::Act: return 4;   // materialized as F32 program-side
    }
    return 4;
}
inline constexpr bool dtype_is_float(DType d) { return d == DType::F32 || d == DType::Act; }
inline constexpr bool dtype_is_int(DType d) { return d == DType::I32 || d == DType::U32; }

// Op codes. Container ops take their byte from echo's ptir_abi.h constants;
// composite (fusion-only, not-in-container) ops occupy the private 0xE0+ band.
enum class OpCode : std::uint8_t {
    // map / element-wise
    Exp = PTIR_OP_EXP, Log = PTIR_OP_LOG, Neg = PTIR_OP_NEG, Recip = PTIR_OP_RECIP,
    Abs = PTIR_OP_ABS, Sign = PTIR_OP_SIGN, Cast = PTIR_OP_CAST,
    Add = PTIR_OP_ADD, Sub = PTIR_OP_SUB, Mul = PTIR_OP_MUL, Div = PTIR_OP_DIV,
    MaxElem = PTIR_OP_MAX_ELEM, MinElem = PTIR_OP_MIN_ELEM,
    // compare / logic (bool results)
    Gt = PTIR_OP_GT, Ge = PTIR_OP_GE, Eq = PTIR_OP_EQ, Ne = PTIR_OP_NE, Lt = PTIR_OP_LT, Le = PTIR_OP_LE,
    And = PTIR_OP_AND, Or = PTIR_OP_OR, Not = PTIR_OP_NOT,
    Rem = PTIR_OP_REM,
    // choice
    Select = PTIR_OP_SELECT,
    // reduce / scan (row-local)
    ReduceSum = PTIR_OP_REDUCE_SUM, ReduceMax = PTIR_OP_REDUCE_MAX,
    ReduceMin = PTIR_OP_REDUCE_MIN, ReduceArgmax = PTIR_OP_REDUCE_ARGMAX,
    CumSum = PTIR_OP_CUMSUM, CumProd = PTIR_OP_CUMPROD,
    // shape
    Broadcast = PTIR_OP_BROADCAST, Reshape = PTIR_OP_RESHAPE, Transpose = PTIR_OP_TRANSPOSE,
    // order (sort_desc/top_k = 2 results value-first; pivot_threshold = bool mask)
    SortDesc = PTIR_OP_SORT_DESC, TopK = PTIR_OP_TOP_K, PivotThreshold = PTIR_OP_PIVOT_THRESHOLD,
    // linear (library)
    Matmul = PTIR_OP_MATMUL,
    // index
    Gather = PTIR_OP_GATHER, GatherRow = PTIR_OP_GATHER_ROW,
    ScatterAdd = PTIR_OP_SCATTER_ADD, ScatterSet = PTIR_OP_SCATTER_SET,
    Iota = PTIR_OP_IOTA,
    // sampling primitives
    MaskApplyPacked = PTIR_OP_MASK_APPLY_PACKED,
    Rng = PTIR_OP_RNG, RngKeyed = PTIR_OP_RNG_KEYED,
    // structural (container roots/effects; translated by the reader, not launched)
    Const = PTIR_OP_CONST,
    ChanTake = PTIR_OP_CHAN_TAKE, ChanRead = PTIR_OP_CHAN_READ, ChanPut = PTIR_OP_CHAN_PUT,
    IntrinsicVal = PTIR_OP_INTRINSIC_VAL, KernelCall = PTIR_OP_KERNEL_CALL, SinkCall = PTIR_OP_SINK_CALL,

    // ── composite / fusion-only (NOT container ops; core-op expansions, §7.3) ──
    Softmax = 0xE0, LogSoftmax = 0xE1, L2Norm = 0xE2,
    GumbelNoise = 0xE3,     // gumbel = -log(-log(rng_uniform))
    MaskApplyBool = 0xE4,   // unpacked bool mask variant of mask_apply
    RankLe = 0xE5,          // the rank predicate pivot_threshold cuts at, as a bool op
};

enum class OpFamily : std::uint8_t {
    Map, Compare, Choice, Shape, Index, Reduce, Order, Linear, Sampling, Structural,
};

enum class ResultKind : std::uint8_t { SameAsInput, Bool, Index, Custom, None };

enum class LaunchClass : std::uint8_t {
    Elementwise, RowLocal, Gather, Scatter, Library, Materialize, Alias, Structural,
};

struct OpInfo {
    OpCode      code;
    OpFamily    family;
    LaunchClass launch;
    ResultKind  result;
    std::uint8_t arity;      // value-operand count (0xFE = variadic, 0xFF = unknown)
    std::uint8_t results;    // SSA ids defined (0 for chan_put/sink_call, 2 for sort/top_k)
    std::string_view name;
};

inline constexpr OpInfo op_info(OpCode c) {
    using F = OpFamily; using L = LaunchClass; using R = ResultKind;
    switch (c) {
        case OpCode::Exp:   return {c, F::Map, L::Elementwise, R::SameAsInput, 1, 1, "exp"};
        case OpCode::Log:   return {c, F::Map, L::Elementwise, R::SameAsInput, 1, 1, "log"};
        case OpCode::Neg:   return {c, F::Map, L::Elementwise, R::SameAsInput, 1, 1, "neg"};
        case OpCode::Recip: return {c, F::Map, L::Elementwise, R::SameAsInput, 1, 1, "recip"};
        case OpCode::Abs:   return {c, F::Map, L::Elementwise, R::SameAsInput, 1, 1, "abs"};
        case OpCode::Sign:  return {c, F::Map, L::Elementwise, R::SameAsInput, 1, 1, "sign"};
        case OpCode::Cast:  return {c, F::Map, L::Elementwise, R::Custom,      1, 1, "cast"};
        case OpCode::Add:   return {c, F::Map, L::Elementwise, R::SameAsInput, 2, 1, "add"};
        case OpCode::Sub:   return {c, F::Map, L::Elementwise, R::SameAsInput, 2, 1, "sub"};
        case OpCode::Mul:   return {c, F::Map, L::Elementwise, R::SameAsInput, 2, 1, "mul"};
        case OpCode::Div:   return {c, F::Map, L::Elementwise, R::SameAsInput, 2, 1, "div"};
        case OpCode::MaxElem: return {c, F::Map, L::Elementwise, R::SameAsInput, 2, 1, "max_elem"};
        case OpCode::MinElem: return {c, F::Map, L::Elementwise, R::SameAsInput, 2, 1, "min_elem"};

        case OpCode::Gt: return {c, F::Compare, L::Elementwise, R::Bool, 2, 1, "gt"};
        case OpCode::Ge: return {c, F::Compare, L::Elementwise, R::Bool, 2, 1, "ge"};
        case OpCode::Eq: return {c, F::Compare, L::Elementwise, R::Bool, 2, 1, "eq"};
        case OpCode::Ne: return {c, F::Compare, L::Elementwise, R::Bool, 2, 1, "ne"};
        case OpCode::Lt: return {c, F::Compare, L::Elementwise, R::Bool, 2, 1, "lt"};
        case OpCode::Le: return {c, F::Compare, L::Elementwise, R::Bool, 2, 1, "le"};
        case OpCode::And: return {c, F::Compare, L::Elementwise, R::Bool, 2, 1, "and"};
        case OpCode::Or:  return {c, F::Compare, L::Elementwise, R::Bool, 2, 1, "or"};
        case OpCode::Not: return {c, F::Compare, L::Elementwise, R::Bool, 1, 1, "not"};
        case OpCode::Rem: return {c, F::Map, L::Elementwise, R::SameAsInput, 2, 1, "rem"};

        case OpCode::Select: return {c, F::Choice, L::Elementwise, R::SameAsInput, 3, 1, "select"};

        case OpCode::ReduceSum:    return {c, F::Reduce, L::RowLocal, R::SameAsInput, 1, 1, "reduce_sum"};
        case OpCode::ReduceMax:    return {c, F::Reduce, L::RowLocal, R::SameAsInput, 1, 1, "reduce_max"};
        case OpCode::ReduceMin:    return {c, F::Reduce, L::RowLocal, R::SameAsInput, 1, 1, "reduce_min"};
        case OpCode::ReduceArgmax: return {c, F::Reduce, L::RowLocal, R::Index,       1, 1, "reduce_argmax"};
        case OpCode::CumSum:       return {c, F::Reduce, L::RowLocal, R::SameAsInput, 1, 1, "cumsum"};
        case OpCode::CumProd:      return {c, F::Reduce, L::RowLocal, R::SameAsInput, 1, 1, "cumprod"};

        case OpCode::Broadcast: return {c, F::Shape, L::Materialize, R::SameAsInput, 1, 1, "broadcast"};
        case OpCode::Reshape:   return {c, F::Shape, L::Alias,       R::SameAsInput, 1, 1, "reshape"};
        case OpCode::Transpose: return {c, F::Shape, L::Materialize, R::SameAsInput, 1, 1, "transpose"};

        case OpCode::SortDesc:       return {c, F::Order, L::Library,  R::Custom, 1, 2, "sort_desc"};
        case OpCode::TopK:           return {c, F::Order, L::Library,  R::Custom, 1, 2, "top_k"};
        case OpCode::PivotThreshold: return {c, F::Order, L::RowLocal, R::Bool,   2, 1, "pivot_threshold"};

        case OpCode::Matmul: return {c, F::Linear, L::Library, R::SameAsInput, 2, 1, "matmul"};

        case OpCode::Gather:     return {c, F::Index, L::Gather,  R::SameAsInput, 2, 1, "gather"};
        case OpCode::GatherRow:  return {c, F::Index, L::Gather,  R::SameAsInput, 2, 1, "gather_row"};
        case OpCode::ScatterAdd: return {c, F::Index, L::Scatter, R::SameAsInput, 3, 1, "scatter_add"};
        case OpCode::ScatterSet: return {c, F::Index, L::Scatter, R::SameAsInput, 3, 1, "scatter_set"};
        case OpCode::Iota:       return {c, F::Index, L::Elementwise, R::Index,   0, 1, "iota"};

        case OpCode::MaskApplyPacked: return {c, F::Sampling, L::Elementwise, R::SameAsInput, 2, 1, "mask_apply_packed"};
        case OpCode::Rng:      return {c, F::Sampling, L::Elementwise, R::SameAsInput, 0, 1, "rng"};
        case OpCode::RngKeyed: return {c, F::Sampling, L::Elementwise, R::SameAsInput, 1, 1, "rng_keyed"};

        case OpCode::Const:        return {c, F::Structural, L::Structural, R::Custom,      0, 1, "const"};
        case OpCode::ChanTake:     return {c, F::Structural, L::Structural, R::Custom,      0, 1, "chan_take"};
        case OpCode::ChanRead:     return {c, F::Structural, L::Structural, R::Custom,      0, 1, "chan_read"};
        case OpCode::ChanPut:      return {c, F::Structural, L::Structural, R::None,        1, 0, "chan_put"};
        case OpCode::IntrinsicVal: return {c, F::Structural, L::Structural, R::Custom,      0, 1, "intrinsic_val"};
        case OpCode::KernelCall:   return {c, F::Structural, L::Structural, R::Custom,   0xFE, 1, "kernel_call"};
        case OpCode::SinkCall:     return {c, F::Structural, L::Structural, R::None,     0xFE, 0, "sink_call"};

        case OpCode::Softmax:       return {c, F::Sampling, L::RowLocal,    R::SameAsInput, 1, 1, "softmax"};
        case OpCode::LogSoftmax:    return {c, F::Sampling, L::RowLocal,    R::SameAsInput, 1, 1, "log_softmax"};
        case OpCode::L2Norm:        return {c, F::Sampling, L::RowLocal,    R::SameAsInput, 1, 1, "l2norm"};
        case OpCode::GumbelNoise:   return {c, F::Sampling, L::Elementwise, R::SameAsInput, 1, 1, "gumbel"};
        case OpCode::MaskApplyBool: return {c, F::Sampling, L::Elementwise, R::SameAsInput, 2, 1, "mask_apply"};
        case OpCode::RankLe:        return {c, F::Order,    L::RowLocal,    R::Bool,        1, 1, "rank_le"};
    }
    return {c, OpFamily::Map, LaunchClass::Alias, ResultKind::Custom, 0xFF, 1, "?"};
}

inline constexpr bool op_is_known(OpCode c) { return op_info(c).arity != 0xFF; }
inline constexpr std::string_view op_name(OpCode c) { return op_info(c).name; }
inline constexpr std::uint32_t op_result_count(OpCode c) { return op_info(c).results; }

}  // namespace pie_native::ptir

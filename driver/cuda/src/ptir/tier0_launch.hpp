#pragma once

// PTIR tier-0 op-launch dispatcher — maps one decoded trace op (trace.hpp Op)
// onto the matching prebuilt kernel in tier0_kernels.cuh, resolving element
// dtype and the row/len decomposition. The stage-runner (tier0_runner.hpp) fills
// a LaunchOp per op from the value table and calls launch_op; this is the single
// OpCode→kernel switch (the tier-0 "interpret" step, overview §7.3).
//
// Reshape is an ALIAS (no launch — the runner aliases the buffer). top_k/matmul
// are LIBRARY kernels (T9). Everything else is one prebuilt row-parallel launch.

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "pie_native/ptir/op_table.hpp"
#include "ptir/tier0_kernels.cuh"
#include "pie_native/ptir/trace.hpp"

namespace pie_cuda_driver::ptir {

// Shared pure-host PTIR decode model (trace/op-table/container/bound/
// fire-geometry) now lives in pie_native::ptir (driver/common); bring it into
// scope so the CUDA-side tier-0/1 code below can use it unqualified.
using namespace pie_native::ptir;

// The runner-resolved launch descriptor for one op.
struct LaunchOp {
    OpCode code = OpCode::Add;
    std::vector<const void*> in;   // operand device pointers, in Op::args order
    void*  out = nullptr;          // first result device pointer
    void*  out2 = nullptr;         // second result (top_k indices)

    DType  elem_dtype = DType::F32;  // element dtype of the primary operand (map/index/reduce)
    DType  out_dtype = DType::F32;   // result element dtype (cast target; compare→Bool)

    std::uint32_t rows = 1;         // row (CTA) count of the primary shape
    std::uint32_t len = 1;          // per-row length of the primary shape
    std::uint64_t numel = 1;        // total elements of the result (elementwise)

    std::uint32_t k = 0;            // top_k / rank_le (standalone) immediate
    int           bcast_mode = 0;   // broadcast: 0 scalar, 1 per-row
    std::uint32_t rng_stream = 0;   // gumbel stream salt
    const void*   row_seeds = nullptr;  // gumbel per-row seed buffer
    std::uint32_t n_scatter = 0;    // scatter_set index count
    int           a_scalar = 0;     // broadcast: operand 0 is a scalar (index 0)
    int           b_scalar = 0;     // broadcast: operand 1 is a scalar (index 0)
    const void*   bcast_meta = nullptr;  // general broadcast: [tdims(4), sstride(4)] device buf
    std::uint32_t bcast_rank = 0;        // general broadcast: target rank

    // pivot_threshold predicate (interface/ptir interp.rs Op::PivotThreshold):
    // the payload is ALWAYS a resolved trace value (scalar or per-row
    // [rows] vector) — never a host immediate. The runner resolves it to a
    // device pointer + its dtype + element count before launch (tier0_runner.hpp
    // build_launch); `pred_numel<=1` means broadcast (scalar), else per-row.
    PredTag       pred_tag = PredTag::RankLe;
    const void*   pred_ptr = nullptr;
    DType         pred_dtype = DType::F32;
    std::uint32_t pred_numel = 1;

    cudaStream_t stream = nullptr;
};

namespace detail {

constexpr int gs(std::uint64_t n, int b = kTier0Block) { return (int)((n + b - 1) / b); }

// Elementwise binary over a math dtype.
template <class T>
inline void run_binary(const LaunchOp& o, BinKind k) {
    k_binary<T><<<gs(o.numel), kTier0Block, 0, o.stream>>>(
        (const T*)o.in[0], (const T*)o.in[1], (T*)o.out, o.numel, k, o.a_scalar, o.b_scalar);
}
template <class T>
inline void run_unary(const LaunchOp& o, UnKind k) {
    k_unary<T><<<gs(o.numel), kTier0Block, 0, o.stream>>>((const T*)o.in[0], (T*)o.out, o.numel, k);
}
template <class T>
inline void run_compare(const LaunchOp& o, CmpKind k) {
    k_compare<T><<<gs(o.numel), kTier0Block, 0, o.stream>>>(
        (const T*)o.in[0], (const T*)o.in[1], (std::uint8_t*)o.out, o.numel, k, o.a_scalar, o.b_scalar);
}
template <class T>
inline void run_select(const LaunchOp& o) {
    k_select<T><<<gs(o.numel), kTier0Block, 0, o.stream>>>(
        (const std::uint8_t*)o.in[0], (const T*)o.in[1], (const T*)o.in[2], (T*)o.out, o.numel, o.a_scalar, o.b_scalar);
}
template <class T>
inline void run_gather(const LaunchOp& o) {
    k_gather<T><<<gs(o.numel), kTier0Block, 0, o.stream>>>(
        (const T*)o.in[0], (const std::uint32_t*)o.in[1], (T*)o.out, o.numel);
}
template <class T>
inline void run_scatter(const LaunchOp& o) {
    // base already copied into out by the runner; scatter idx/vals into it.
    k_scatter_set_serial<T><<<1, 1, 0, o.stream>>>(
        (T*)o.out, (const std::uint32_t*)o.in[1], (const T*)o.in[2], o.n_scatter);
}
template <class T>
inline void run_reduce(const LaunchOp& o, RedKind k) {
    k_reduce<T><<<o.rows, kTier0Block, 0, o.stream>>>((const T*)o.in[0], (T*)o.out, o.rows, o.len, k);
}
template <class T>
inline void run_scan(const LaunchOp& o, ScanKind k) {
    k_scan<T><<<o.rows, kTier0Block, 0, o.stream>>>((const T*)o.in[0], (T*)o.out, o.rows, o.len, k);
}
template <class T>
inline void run_broadcast(const LaunchOp& o) {
    if (o.bcast_meta) {
        k_broadcast_general<T><<<gs(o.numel), kTier0Block, 0, o.stream>>>(
            (const T*)o.in[0], (T*)o.out, (const std::uint32_t*)o.bcast_meta, o.bcast_rank, o.numel);
    } else {
        k_broadcast<T><<<gs(o.numel), kTier0Block, 0, o.stream>>>(
            (const T*)o.in[0], (T*)o.out, o.rows, o.len, o.bcast_mode);
    }
}
template <class T>
inline void run_transpose(const LaunchOp& o) {
    dim3 blk(16, 16), grd((o.len + 15) / 16, (o.rows + 15) / 16);
    k_transpose<T><<<grd, blk, 0, o.stream>>>((const T*)o.in[0], (T*)o.out, o.rows, o.len);
}

// Dispatch a family that is generic over {F32,I32,U32} by dtype.
template <template <class> class Fn, class... Args>
inline bool by_math_dtype(DType d, Args&&... a) {
    switch (d) {
        case DType::F32: Fn<float>{}(std::forward<Args>(a)...); return true;
        case DType::I32: Fn<std::int32_t>{}(std::forward<Args>(a)...); return true;
        case DType::U32: Fn<std::uint32_t>{}(std::forward<Args>(a)...); return true;
        case DType::Bool: return false;  // handled by the logic/bool path
    }
    return false;
}

}  // namespace detail

// Launch one op. Returns false if the op/dtype combo is not covered by the
// tier-0 library (the runner then fails loud). top_k/matmul are library kernels
// handled here too.
inline bool launch_op(const LaunchOp& o) {
    using namespace detail;
    switch (o.code) {
        // ── map ──
        case OpCode::Add: case OpCode::Sub: case OpCode::Mul: case OpCode::Div: case OpCode::Rem:
        case OpCode::MaxElem: case OpCode::MinElem: {
            BinKind k = o.code == OpCode::Add ? BinKind::Add : o.code == OpCode::Sub ? BinKind::Sub
                      : o.code == OpCode::Mul ? BinKind::Mul : o.code == OpCode::Div ? BinKind::Div
                      : o.code == OpCode::Rem ? BinKind::Rem
                      : o.code == OpCode::MaxElem ? BinKind::MaxElem : BinKind::MinElem;
            switch (o.elem_dtype) {
                case DType::F32: run_binary<float>(o, k); return true;
                case DType::I32: run_binary<std::int32_t>(o, k); return true;
                case DType::U32: run_binary<std::uint32_t>(o, k); return true;
                default: return false;
            }
        }
        case OpCode::Neg: case OpCode::Exp: case OpCode::Log:
        case OpCode::Recip: case OpCode::Abs: case OpCode::Sign: {
            UnKind k = o.code == OpCode::Neg ? UnKind::Neg : o.code == OpCode::Exp ? UnKind::Exp
                     : o.code == OpCode::Log ? UnKind::Log : o.code == OpCode::Recip ? UnKind::Recip
                     : o.code == OpCode::Abs ? UnKind::Abs : UnKind::Sign;
            switch (o.elem_dtype) {
                case DType::F32: run_unary<float>(o, k); return true;
                case DType::I32: if (k == UnKind::Neg || k == UnKind::Abs || k == UnKind::Sign) { run_unary<std::int32_t>(o, k); return true; } return false;
                default: return false;
            }
        }
        case OpCode::Cast: {
            // Covered pairs (extend as needed). Result dtype = out_dtype.
            if (o.elem_dtype == DType::F32 && o.out_dtype == DType::I32) {
                k_cast<float, std::int32_t><<<gs(o.numel), kTier0Block, 0, o.stream>>>((const float*)o.in[0], (std::int32_t*)o.out, o.numel); return true; }
            if (o.elem_dtype == DType::F32 && o.out_dtype == DType::U32) {
                k_cast<float, std::uint32_t><<<gs(o.numel), kTier0Block, 0, o.stream>>>((const float*)o.in[0], (std::uint32_t*)o.out, o.numel); return true; }
            if (o.elem_dtype == DType::I32 && o.out_dtype == DType::F32) {
                k_cast<std::int32_t, float><<<gs(o.numel), kTier0Block, 0, o.stream>>>((const std::int32_t*)o.in[0], (float*)o.out, o.numel); return true; }
            if (o.elem_dtype == DType::U32 && o.out_dtype == DType::F32) {
                k_cast<std::uint32_t, float><<<gs(o.numel), kTier0Block, 0, o.stream>>>((const std::uint32_t*)o.in[0], (float*)o.out, o.numel); return true; }
            if (o.elem_dtype == DType::Bool && o.out_dtype == DType::F32) {
                k_cast<std::uint8_t, float><<<gs(o.numel), kTier0Block, 0, o.stream>>>((const std::uint8_t*)o.in[0], (float*)o.out, o.numel); return true; }
            if (o.elem_dtype == DType::U32 && o.out_dtype == DType::I32) {
                k_cast<std::uint32_t, std::int32_t><<<gs(o.numel), kTier0Block, 0, o.stream>>>((const std::uint32_t*)o.in[0], (std::int32_t*)o.out, o.numel); return true; }
            if (o.elem_dtype == DType::I32 && o.out_dtype == DType::U32) {
                k_cast<std::int32_t, std::uint32_t><<<gs(o.numel), kTier0Block, 0, o.stream>>>((const std::int32_t*)o.in[0], (std::uint32_t*)o.out, o.numel); return true; }
            return false;
        }
        // ── compare / logic ──
        case OpCode::Eq: case OpCode::Ne: case OpCode::Lt: case OpCode::Le: case OpCode::Gt: case OpCode::Ge: {
            CmpKind k = o.code == OpCode::Eq ? CmpKind::Eq : o.code == OpCode::Ne ? CmpKind::Ne
                      : o.code == OpCode::Lt ? CmpKind::Lt : o.code == OpCode::Le ? CmpKind::Le
                      : o.code == OpCode::Gt ? CmpKind::Gt : CmpKind::Ge;
            switch (o.elem_dtype) {
                case DType::F32: run_compare<float>(o, k); return true;
                case DType::I32: run_compare<std::int32_t>(o, k); return true;
                case DType::U32: run_compare<std::uint32_t>(o, k); return true;
                default: return false;
            }
        }
        case OpCode::And: case OpCode::Or:
            k_logic<<<gs(o.numel), kTier0Block, 0, o.stream>>>(
                (const std::uint8_t*)o.in[0], (const std::uint8_t*)o.in[1], (std::uint8_t*)o.out, o.numel,
                o.code == OpCode::And ? LogicKind::And : LogicKind::Or);
            return true;
        case OpCode::Not:
            k_not<<<gs(o.numel), kTier0Block, 0, o.stream>>>((const std::uint8_t*)o.in[0], (std::uint8_t*)o.out, o.numel);
            return true;
        // ── choice ──
        case OpCode::Select:
            switch (o.elem_dtype) {
                case DType::F32: run_select<float>(o); return true;
                case DType::I32: run_select<std::int32_t>(o); return true;
                case DType::U32: run_select<std::uint32_t>(o); return true;
                case DType::Bool: run_select<std::uint8_t>(o); return true;
            }
            return false;
        // ── shape ──
        case OpCode::Broadcast:
            switch (o.elem_dtype) {
                case DType::F32: run_broadcast<float>(o); return true;
                case DType::I32: run_broadcast<std::int32_t>(o); return true;
                case DType::U32: run_broadcast<std::uint32_t>(o); return true;
                case DType::Bool: run_broadcast<std::uint8_t>(o); return true;
            }
            return false;
        case OpCode::Transpose:
            switch (o.elem_dtype) {
                case DType::F32: run_transpose<float>(o); return true;
                case DType::I32: run_transpose<std::int32_t>(o); return true;
                case DType::U32: run_transpose<std::uint32_t>(o); return true;
                default: return false;
            }
        case OpCode::Reshape: return true;  // alias — handled by the runner, no launch
        // ── index ──
        case OpCode::Iota:
            k_iota<<<gs(o.numel), kTier0Block, 0, o.stream>>>((std::uint32_t*)o.out, (std::uint32_t)o.numel);
            return true;
        case OpCode::Gather:
            switch (o.elem_dtype) {
                case DType::F32: run_gather<float>(o); return true;
                case DType::I32: run_gather<std::int32_t>(o); return true;
                case DType::U32: run_gather<std::uint32_t>(o); return true;
                case DType::Bool: run_gather<std::uint8_t>(o); return true;
            }
            return false;
        case OpCode::GatherRow:
            // rows = index count (m), len = per-row length.
            switch (o.elem_dtype) {
                case DType::F32: k_gather_row<float><<<gs(o.numel), kTier0Block, 0, o.stream>>>((const float*)o.in[0], (const std::uint32_t*)o.in[1], (float*)o.out, o.rows, o.len); return true;
                case DType::U32: k_gather_row<std::uint32_t><<<gs(o.numel), kTier0Block, 0, o.stream>>>((const std::uint32_t*)o.in[0], (const std::uint32_t*)o.in[1], (std::uint32_t*)o.out, o.rows, o.len); return true;
                case DType::I32: k_gather_row<std::int32_t><<<gs(o.numel), kTier0Block, 0, o.stream>>>((const std::int32_t*)o.in[0], (const std::uint32_t*)o.in[1], (std::int32_t*)o.out, o.rows, o.len); return true;
                case DType::Bool: k_gather_row<std::uint8_t><<<gs(o.numel), kTier0Block, 0, o.stream>>>((const std::uint8_t*)o.in[0], (const std::uint32_t*)o.in[1], (std::uint8_t*)o.out, o.rows, o.len); return true;
                default: return false;
            }
        case OpCode::ScatterSet:
            switch (o.elem_dtype) {
                case DType::F32: run_scatter<float>(o); return true;
                case DType::I32: run_scatter<std::int32_t>(o); return true;
                case DType::U32: run_scatter<std::uint32_t>(o); return true;
                default: return false;
            }
        case OpCode::ScatterAdd:
            switch (o.elem_dtype) {
                case DType::F32: k_scatter_add_serial<float><<<1, 1, 0, o.stream>>>((float*)o.out, (const std::uint32_t*)o.in[1], (const float*)o.in[2], o.n_scatter); return true;
                case DType::I32: k_scatter_add_serial<std::int32_t><<<1, 1, 0, o.stream>>>((std::int32_t*)o.out, (const std::uint32_t*)o.in[1], (const std::int32_t*)o.in[2], o.n_scatter); return true;
                case DType::U32: k_scatter_add_serial<std::uint32_t><<<1, 1, 0, o.stream>>>((std::uint32_t*)o.out, (const std::uint32_t*)o.in[1], (const std::uint32_t*)o.in[2], o.n_scatter); return true;
                default: return false;
            }
        // ── reduce / scan ──
        case OpCode::ReduceSum: case OpCode::ReduceMax: case OpCode::ReduceMin: {
            RedKind k = o.code == OpCode::ReduceSum ? RedKind::Sum
                      : o.code == OpCode::ReduceMax ? RedKind::Max : RedKind::Min;
            switch (o.elem_dtype) {
                case DType::F32: run_reduce<float>(o, k); return true;
                case DType::I32: run_reduce<std::int32_t>(o, k); return true;
                case DType::U32: run_reduce<std::uint32_t>(o, k); return true;
                default: return false;
            }
        }
        case OpCode::ReduceArgmax:
            k_reduce_argmax<<<o.rows, kTier0Block, 0, o.stream>>>((const float*)o.in[0], (std::uint32_t*)o.out, o.rows, o.len);
            return true;
        case OpCode::CumSum: case OpCode::CumProd: {
            ScanKind k = o.code == OpCode::CumSum ? ScanKind::Sum : ScanKind::Prod;
            switch (o.elem_dtype) {
                case DType::F32: run_scan<float>(o, k); return true;
                case DType::I32: run_scan<std::int32_t>(o, k); return true;
                case DType::U32: run_scan<std::uint32_t>(o, k); return true;
                default: return false;
            }
        }
        // ── normalize (composite / fusion-only) ──
        case OpCode::Softmax: case OpCode::LogSoftmax: case OpCode::L2Norm: {
            NormKind k = o.code == OpCode::Softmax ? NormKind::Softmax
                       : o.code == OpCode::LogSoftmax ? NormKind::LogSoftmax : NormKind::L2Norm;
            k_normalize<<<o.rows, kTier0Block, 0, o.stream>>>((const float*)o.in[0], (float*)o.out, o.rows, o.len, k);
            return true;
        }
        // ── sampling ──
        case OpCode::MaskApplyBool:
            k_mask_apply<<<gs(o.numel), kTier0Block, 0, o.stream>>>(
                (const float*)o.in[0], (const std::uint8_t*)o.in[1], (float*)o.out, o.numel);
            return true;
        case OpCode::MaskApplyPacked:
            // len = vocab; k carries mask words per row (ceil(len/32)).
            k_mask_apply_packed<<<gs(o.numel), kTier0Block, 0, o.stream>>>(
                (const float*)o.in[0], (const std::uint32_t*)o.in[1], (float*)o.out, o.rows, o.len, o.k);
            return true;
        case OpCode::GumbelNoise:
            k_gumbel<<<gs((std::uint64_t)o.rows * o.len), kTier0Block, 0, o.stream>>>(
                (const std::uint32_t*)o.row_seeds, o.rng_stream, (float*)o.out, o.rows, o.len);
            return true;
        case OpCode::Rng:   // ambient seed; rng_stream carries stream, bcast_mode = gumbel flag
            k_rng_ambient<<<gs((std::uint64_t)o.rows * o.len), kTier0Block, 0, o.stream>>>(
                (const std::uint32_t*)o.row_seeds, o.rng_stream, (float*)o.out, o.rows, o.len, o.bcast_mode);
            return true;
        case OpCode::RngKeyed:   // state=[key,ctr] in in[0]; bcast_mode = gumbel flag
            k_rng_keyed<<<gs(o.numel), kTier0Block, 0, o.stream>>>(
                (const std::uint32_t*)o.in[0], (float*)o.out, o.numel, o.bcast_mode);
            return true;
        // ── order ──
        case OpCode::RankLe:
            // Composite/fusion-only form (0xE5, never emitted by container
            // decode today): `k` really is a host immediate here (op.imm),
            // unlike pivot_threshold's dynamic predicate below.
            k_rank_le<<<o.rows, kTier0Block, 0, o.stream>>>((const float*)o.in[0], (std::uint8_t*)o.out, o.rows, o.len, o.k);
            return true;
        case OpCode::PivotThreshold:
            // Container pivot_threshold(input, predicate) → bool selection
            // mask. The predicate payload is a resolved trace value (scalar
            // or per-row), never an immediate (interface/ptir interp.rs).
            switch (o.pred_tag) {
                case PredTag::RankLe:
                    switch (o.pred_dtype) {
                        case DType::I32:
                            k_pivot_rankle<std::int32_t><<<o.rows, kTier0Block, 0, o.stream>>>(
                                (const float*)o.in[0], (std::uint8_t*)o.out, o.rows, o.len,
                                (const std::int32_t*)o.pred_ptr, o.pred_numel);
                            return true;
                        case DType::U32:
                            k_pivot_rankle<std::uint32_t><<<o.rows, kTier0Block, 0, o.stream>>>(
                                (const float*)o.in[0], (std::uint8_t*)o.out, o.rows, o.len,
                                (const std::uint32_t*)o.pred_ptr, o.pred_numel);
                            return true;
                        default:
                            return false;   // RankLe's k must be I32/U32 (infer.rs dtype_is_int)
                    }
                case PredTag::CummassLe:
                    k_pivot_cummassle<<<o.rows, kTier0Block, 0, o.stream>>>(
                        (const float*)o.in[0], (std::uint8_t*)o.out, o.rows, o.len,
                        (const float*)o.pred_ptr, o.pred_numel);
                    return true;
                case PredTag::ProbGe:
                    k_pivot_probge<<<gs((std::uint64_t)o.rows * o.len), kTier0Block, 0, o.stream>>>(
                        (const float*)o.in[0], (std::uint8_t*)o.out, o.rows, o.len,
                        (const float*)o.pred_ptr, o.pred_numel);
                    return true;
            }
            return false;
        // ── library kernels ──
        case OpCode::SortDesc:
            // full per-row sort = top_k with k = len (2 results, value-first).
            k_topk_rows<<<o.rows, kTier0Block, 0, o.stream>>>(
                (const float*)o.in[0], (float*)o.out, (std::uint32_t*)o.out2, o.rows, o.len, o.len);
            return true;
        case OpCode::TopK:
            k_topk_rows<<<o.rows, kTier0Block, 0, o.stream>>>(
                (const float*)o.in[0], (float*)o.out, (std::uint32_t*)o.out2, o.rows, o.len, o.k);
            return true;
        case OpCode::Matmul:
            // rows=M, len=K encoded by the runner; k=N.
            {
                dim3 blk(32, 1), grd((o.k + 31) / 32, o.rows);
                k_matmul<<<grd, blk, 0, o.stream>>>((const float*)o.in[0], (const float*)o.in[1], (float*)o.out, o.rows, o.len, o.k);
            }
            return true;
        default:
            return false;   // structural ops (chan_*/const/intrinsic_val/kernel_call/sink_call) are not launched
    }
    return false;
}

}  // namespace pie_cuda_driver::ptir

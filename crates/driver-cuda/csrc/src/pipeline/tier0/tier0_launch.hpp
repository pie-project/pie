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

#include "pie/driver/launch/op_table.hpp"
#include "ptir/tier0.cuh"
#include "pie/driver/launch/program.hpp"

namespace pie_cuda_driver::pipeline {

// Shared pure-host PTIR decode model (trace/op-table/container/bound/
// fire-geometry) now lives in pie::driver::launch (driver/common); bring it into
// scope so the CUDA-side tier-0/1 code below can use it unqualified.
using namespace pie::driver::launch;

// `DType` is NOT safe to take from that using-directive. There are two of
// them: PTIR's (F32/I32/U32/Bool/Act) and kernels-cuda's tensor dtype in
// `pie_cuda_driver` (BF16/FP16/FP32/...). A using-directive injects names at
// the nearest common enclosing namespace -- the global one -- while
// `pie_cuda_driver::DType` sits closer to this code than that, so it WINS
// unqualified lookup from inside `pie_cuda_driver::pipeline`.
//
// That only became possible when the tier-0 kernels moved into kernels-cuda
// and pulled tensor.hpp into these TUs. It surfaced as an error solely
// because the two enums spell their members differently (F32 vs FP32). Had
// they agreed, this file would have silently switched to the wrong dtype
// enum. Naming it explicitly is what makes that impossible rather than lucky.
using DType = pie::driver::launch::DType;

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
    std::uint32_t imm = 0;
    std::uint32_t imm2 = 0;
    std::uint32_t imm3 = 0;
    int           bcast_mode = 0;   // broadcast: 0 scalar, 1 per-row
    std::uint32_t rng_stream = 0;   // gumbel stream salt
    const void*   row_seeds = nullptr;  // gumbel per-row seed buffer
    std::uint32_t n_scatter = 0;    // scatter_set index count
    std::uint32_t axis0 = 0;        // index-family source/base leading extent
    std::uint32_t inner = 1;        // product of source/base trailing extents
    DType         index_dtype = DType::U32;
    bool          scalar_vals = false;
    int           a_scalar = 0;     // broadcast: operand 0 is a scalar (index 0)
    int           b_scalar = 0;     // broadcast: operand 1 is a scalar (index 0)
    const void*   bcast_meta = nullptr;  // general broadcast: [tdims(4), sstride(4)] device buf
    std::uint32_t bcast_rank = 0;        // general broadcast: target rank

    // pivot_threshold predicate (tensor-compiler's eval/src/interp.rs Op::PivotThreshold):
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

constexpr int gs(std::uint64_t n, int b = kernels::ptir::kTier0Block) { return (int)((n + b - 1) / b); }

// Elementwise binary over a math dtype.
template <class T>
inline void run_binary(const LaunchOp& o, kernels::ptir::BinKind k) {
    kernels::ptir::k_binary<T><<<gs(o.numel), kernels::ptir::kTier0Block, 0, o.stream>>>(
        (const T*)o.in[0], (const T*)o.in[1], (T*)o.out, o.numel, k, o.a_scalar, o.b_scalar);
}
template <class T>
inline void run_unary(const LaunchOp& o, kernels::ptir::UnKind k) {
    kernels::ptir::k_unary<T><<<gs(o.numel), kernels::ptir::kTier0Block, 0, o.stream>>>((const T*)o.in[0], (T*)o.out, o.numel, k);
}
template <class T>
inline void run_compare(const LaunchOp& o, kernels::ptir::CmpKind k) {
    kernels::ptir::k_compare<T><<<gs(o.numel), kernels::ptir::kTier0Block, 0, o.stream>>>(
        (const T*)o.in[0], (const T*)o.in[1], (std::uint8_t*)o.out, o.numel, k, o.a_scalar, o.b_scalar);
}
template <class T>
inline void run_select(const LaunchOp& o) {
    kernels::ptir::k_select<T><<<gs(o.numel), kernels::ptir::kTier0Block, 0, o.stream>>>(
        (const std::uint8_t*)o.in[0], (const T*)o.in[1], (const T*)o.in[2], (T*)o.out, o.numel, o.a_scalar, o.b_scalar);
}
template <class T, class Index>
inline void run_gather(const LaunchOp& o) {
    kernels::ptir::k_gather_axis0<T, Index><<<gs(o.numel), kernels::ptir::kTier0Block, 0, o.stream>>>(
        static_cast<const T*>(o.in[0]),
        static_cast<const Index*>(o.in[1]),
        static_cast<T*>(o.out),
        o.n_scatter,
        o.axis0,
        o.inner);
}
template <class T>
inline bool run_gather_indexed(const LaunchOp& o) {
    if (o.index_dtype == DType::I32) {
        run_gather<T, std::int32_t>(o);
        return true;
    }
    if (o.index_dtype == DType::U32) {
        run_gather<T, std::uint32_t>(o);
        return true;
    }
    return false;
}
template <class T>
inline bool run_gather_row_indexed(const LaunchOp& o) {
    if (o.index_dtype == DType::I32) {
        kernels::ptir::k_gather_row<T, std::int32_t><<<
            gs(o.numel), kernels::ptir::kTier0Block, 0, o.stream>>>(
            static_cast<const T*>(o.in[0]),
            static_cast<const std::int32_t*>(o.in[1]),
            static_cast<T*>(o.out), o.rows, o.len);
        return true;
    }
    if (o.index_dtype == DType::U32) {
        kernels::ptir::k_gather_row<T, std::uint32_t><<<
            gs(o.numel), kernels::ptir::kTier0Block, 0, o.stream>>>(
            static_cast<const T*>(o.in[0]),
            static_cast<const std::uint32_t*>(o.in[1]),
            static_cast<T*>(o.out), o.rows, o.len);
        return true;
    }
    return false;
}
template <class T, class Index, bool Add>
inline void run_scatter(const LaunchOp& o) {
    kernels::ptir::k_scatter_axis0_serial<T, Index, Add><<<1, 1, 0, o.stream>>>(
        static_cast<T*>(o.out),
        static_cast<const Index*>(o.in[1]),
        static_cast<const T*>(o.in[2]),
        o.n_scatter,
        o.axis0,
        o.inner,
        o.scalar_vals);
}
template <class T, bool Add>
inline bool run_scatter_indexed(const LaunchOp& o) {
    if (o.index_dtype == DType::I32) {
        run_scatter<T, std::int32_t, Add>(o);
        return true;
    }
    if (o.index_dtype == DType::U32) {
        run_scatter<T, std::uint32_t, Add>(o);
        return true;
    }
    return false;
}
template <class T>
inline void run_reduce(const LaunchOp& o, kernels::ptir::RedKind k) {
    kernels::ptir::k_reduce<T><<<o.rows, kernels::ptir::kCanonicalReduceWidth, 0, o.stream>>>(
        (const T*)o.in[0], (T*)o.out, o.rows, o.len, k);
}
template <class T>
inline void run_scan(const LaunchOp& o, kernels::ptir::ScanKind k) {
    kernels::ptir::k_scan<T><<<o.rows, kernels::ptir::kTier0Block, 0, o.stream>>>((const T*)o.in[0], (T*)o.out, o.rows, o.len, k);
}
template <class T>
inline void run_broadcast(const LaunchOp& o) {
    if (o.bcast_meta) {
        kernels::ptir::k_broadcast_general<T><<<gs(o.numel), kernels::ptir::kTier0Block, 0, o.stream>>>(
            (const T*)o.in[0], (T*)o.out, (const std::uint32_t*)o.bcast_meta, o.bcast_rank, o.numel);
    } else {
        kernels::ptir::k_broadcast<T><<<gs(o.numel), kernels::ptir::kTier0Block, 0, o.stream>>>(
            (const T*)o.in[0], (T*)o.out, o.rows, o.len, o.bcast_mode);
    }
}
template <class T>
inline void run_transpose(const LaunchOp& o) {
    dim3 blk(16, 16), grd((o.len + 15) / 16, (o.rows + 15) / 16);
    kernels::ptir::k_transpose<T><<<grd, blk, 0, o.stream>>>((const T*)o.in[0], (T*)o.out, o.rows, o.len);
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
            kernels::ptir::BinKind k = o.code == OpCode::Add ? kernels::ptir::BinKind::Add : o.code == OpCode::Sub ? kernels::ptir::BinKind::Sub
                      : o.code == OpCode::Mul ? kernels::ptir::BinKind::Mul : o.code == OpCode::Div ? kernels::ptir::BinKind::Div
                      : o.code == OpCode::Rem ? kernels::ptir::BinKind::Rem
                      : o.code == OpCode::MaxElem ? kernels::ptir::BinKind::MaxElem : kernels::ptir::BinKind::MinElem;
            switch (o.elem_dtype) {
                case DType::F32: run_binary<float>(o, k); return true;
                case DType::I32: run_binary<std::int32_t>(o, k); return true;
                case DType::U32: run_binary<std::uint32_t>(o, k); return true;
                default: return false;
            }
        }
        case OpCode::Neg: case OpCode::Exp: case OpCode::Log:
        case OpCode::Recip: case OpCode::Abs: case OpCode::Sign: {
            kernels::ptir::UnKind k = o.code == OpCode::Neg ? kernels::ptir::UnKind::Neg : o.code == OpCode::Exp ? kernels::ptir::UnKind::Exp
                     : o.code == OpCode::Log ? kernels::ptir::UnKind::Log : o.code == OpCode::Recip ? kernels::ptir::UnKind::Recip
                     : o.code == OpCode::Abs ? kernels::ptir::UnKind::Abs : kernels::ptir::UnKind::Sign;
            switch (o.elem_dtype) {
                case DType::F32: run_unary<float>(o, k); return true;
                case DType::I32: if (k == kernels::ptir::UnKind::Neg || k == kernels::ptir::UnKind::Abs || k == kernels::ptir::UnKind::Sign) { run_unary<std::int32_t>(o, k); return true; } return false;
                case DType::U32: if (k == kernels::ptir::UnKind::Neg || k == kernels::ptir::UnKind::Abs || k == kernels::ptir::UnKind::Sign) { run_unary<std::uint32_t>(o, k); return true; } return false;
                default: return false;
            }
        }
        case OpCode::Cast: {
            // Covered pairs (extend as needed). Result dtype = out_dtype.
            if (o.elem_dtype == DType::F32 && o.out_dtype == DType::I32) {
                kernels::ptir::k_cast<float, std::int32_t><<<gs(o.numel), kernels::ptir::kTier0Block, 0, o.stream>>>((const float*)o.in[0], (std::int32_t*)o.out, o.numel); return true; }
            if (o.elem_dtype == DType::F32 && o.out_dtype == DType::U32) {
                kernels::ptir::k_cast<float, std::uint32_t><<<gs(o.numel), kernels::ptir::kTier0Block, 0, o.stream>>>((const float*)o.in[0], (std::uint32_t*)o.out, o.numel); return true; }
            if (o.elem_dtype == DType::I32 && o.out_dtype == DType::F32) {
                kernels::ptir::k_cast<std::int32_t, float><<<gs(o.numel), kernels::ptir::kTier0Block, 0, o.stream>>>((const std::int32_t*)o.in[0], (float*)o.out, o.numel); return true; }
            if (o.elem_dtype == DType::U32 && o.out_dtype == DType::F32) {
                kernels::ptir::k_cast<std::uint32_t, float><<<gs(o.numel), kernels::ptir::kTier0Block, 0, o.stream>>>((const std::uint32_t*)o.in[0], (float*)o.out, o.numel); return true; }
            if (o.elem_dtype == DType::Bool && o.out_dtype == DType::F32) {
                kernels::ptir::k_cast<std::uint8_t, float><<<gs(o.numel), kernels::ptir::kTier0Block, 0, o.stream>>>((const std::uint8_t*)o.in[0], (float*)o.out, o.numel); return true; }
            if (o.elem_dtype == DType::Bool && o.out_dtype == DType::I32) {
                kernels::ptir::k_cast<std::uint8_t, std::int32_t><<<gs(o.numel), kernels::ptir::kTier0Block, 0, o.stream>>>((const std::uint8_t*)o.in[0], (std::int32_t*)o.out, o.numel); return true; }
            if (o.elem_dtype == DType::Bool && o.out_dtype == DType::U32) {
                kernels::ptir::k_cast<std::uint8_t, std::uint32_t><<<gs(o.numel), kernels::ptir::kTier0Block, 0, o.stream>>>((const std::uint8_t*)o.in[0], (std::uint32_t*)o.out, o.numel); return true; }
            if (o.out_dtype == DType::Bool) {
                if (o.elem_dtype == DType::F32) {
                    kernels::ptir::k_cast_bool<float><<<gs(o.numel), kernels::ptir::kTier0Block, 0, o.stream>>>((const float*)o.in[0], (std::uint8_t*)o.out, o.numel); return true; }
                if (o.elem_dtype == DType::I32) {
                    kernels::ptir::k_cast_bool<std::int32_t><<<gs(o.numel), kernels::ptir::kTier0Block, 0, o.stream>>>((const std::int32_t*)o.in[0], (std::uint8_t*)o.out, o.numel); return true; }
                if (o.elem_dtype == DType::U32) {
                    kernels::ptir::k_cast_bool<std::uint32_t><<<gs(o.numel), kernels::ptir::kTier0Block, 0, o.stream>>>((const std::uint32_t*)o.in[0], (std::uint8_t*)o.out, o.numel); return true; }
            }
            if (o.elem_dtype == DType::U32 && o.out_dtype == DType::I32) {
                kernels::ptir::k_cast<std::uint32_t, std::int32_t><<<gs(o.numel), kernels::ptir::kTier0Block, 0, o.stream>>>((const std::uint32_t*)o.in[0], (std::int32_t*)o.out, o.numel); return true; }
            if (o.elem_dtype == DType::I32 && o.out_dtype == DType::U32) {
                kernels::ptir::k_cast<std::int32_t, std::uint32_t><<<gs(o.numel), kernels::ptir::kTier0Block, 0, o.stream>>>((const std::int32_t*)o.in[0], (std::uint32_t*)o.out, o.numel); return true; }
            if (o.elem_dtype == o.out_dtype) {
                if (o.elem_dtype == DType::F32) {
                    kernels::ptir::k_cast<float, float><<<gs(o.numel), kernels::ptir::kTier0Block, 0, o.stream>>>((const float*)o.in[0], (float*)o.out, o.numel); return true; }
                if (o.elem_dtype == DType::I32) {
                    kernels::ptir::k_cast<std::int32_t, std::int32_t><<<gs(o.numel), kernels::ptir::kTier0Block, 0, o.stream>>>((const std::int32_t*)o.in[0], (std::int32_t*)o.out, o.numel); return true; }
                if (o.elem_dtype == DType::U32) {
                    kernels::ptir::k_cast<std::uint32_t, std::uint32_t><<<gs(o.numel), kernels::ptir::kTier0Block, 0, o.stream>>>((const std::uint32_t*)o.in[0], (std::uint32_t*)o.out, o.numel); return true; }
                kernels::ptir::k_cast<std::uint8_t, std::uint8_t><<<gs(o.numel), kernels::ptir::kTier0Block, 0, o.stream>>>((const std::uint8_t*)o.in[0], (std::uint8_t*)o.out, o.numel); return true;
            }
            return false;
        }
        // ── compare / logic ──
        case OpCode::Eq: case OpCode::Ne: case OpCode::Lt: case OpCode::Le: case OpCode::Gt: case OpCode::Ge: {
            kernels::ptir::CmpKind k = o.code == OpCode::Eq ? kernels::ptir::CmpKind::Eq : o.code == OpCode::Ne ? kernels::ptir::CmpKind::Ne
                      : o.code == OpCode::Lt ? kernels::ptir::CmpKind::Lt : o.code == OpCode::Le ? kernels::ptir::CmpKind::Le
                      : o.code == OpCode::Gt ? kernels::ptir::CmpKind::Gt : kernels::ptir::CmpKind::Ge;
            switch (o.elem_dtype) {
                case DType::F32: run_compare<float>(o, k); return true;
                case DType::I32: run_compare<std::int32_t>(o, k); return true;
                case DType::U32: run_compare<std::uint32_t>(o, k); return true;
                default: return false;
            }
        }
        case OpCode::And: case OpCode::Or:
            kernels::ptir::k_logic<<<gs(o.numel), kernels::ptir::kTier0Block, 0, o.stream>>>(
                (const std::uint8_t*)o.in[0], (const std::uint8_t*)o.in[1], (std::uint8_t*)o.out, o.numel,
                o.code == OpCode::And ? kernels::ptir::LogicKind::And : kernels::ptir::LogicKind::Or);
            return true;
        case OpCode::Not:
            kernels::ptir::k_not<<<gs(o.numel), kernels::ptir::kTier0Block, 0, o.stream>>>((const std::uint8_t*)o.in[0], (std::uint8_t*)o.out, o.numel);
            return true;
        // ── choice ──
        case OpCode::Select:
            switch (o.elem_dtype) {
                case DType::F32: run_select<float>(o); return true;
                case DType::I32: run_select<std::int32_t>(o); return true;
                case DType::U32: run_select<std::uint32_t>(o); return true;
                case DType::Bool: run_select<std::uint8_t>(o); return true;
                // Act is program-side (materialised as F32 before it reaches
                // tier-0); no kernel takes it. Listed rather than defaulted so
                // -Wswitch still reports the NEXT dtype someone adds.
                case DType::Act: return false;
            }
            return false;
        // ── shape ──
        case OpCode::Broadcast:
            switch (o.elem_dtype) {
                case DType::F32: run_broadcast<float>(o); return true;
                case DType::I32: run_broadcast<std::int32_t>(o); return true;
                case DType::U32: run_broadcast<std::uint32_t>(o); return true;
                case DType::Bool: run_broadcast<std::uint8_t>(o); return true;
                // Act is program-side (materialised as F32 before it reaches
                // tier-0); no kernel takes it. Listed rather than defaulted so
                // -Wswitch still reports the NEXT dtype someone adds.
                case DType::Act: return false;
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
            kernels::ptir::k_iota<<<gs(o.numel), kernels::ptir::kTier0Block, 0, o.stream>>>((std::uint32_t*)o.out, (std::uint32_t)o.numel);
            return true;
        case OpCode::Gather:
            switch (o.elem_dtype) {
                case DType::F32: return run_gather_indexed<float>(o);
                case DType::I32: return run_gather_indexed<std::int32_t>(o);
                case DType::U32: return run_gather_indexed<std::uint32_t>(o);
                case DType::Bool: return run_gather_indexed<std::uint8_t>(o);
                // Act is program-side (materialised as F32 before it reaches
                // tier-0); no kernel takes it. Listed rather than defaulted so
                // -Wswitch still reports the NEXT dtype someone adds.
                case DType::Act: return false;
            }
            return false;
        case OpCode::GatherRow:
            switch (o.elem_dtype) {
                case DType::F32: return run_gather_row_indexed<float>(o);
                case DType::U32: return run_gather_row_indexed<std::uint32_t>(o);
                case DType::I32: return run_gather_row_indexed<std::int32_t>(o);
                case DType::Bool: return run_gather_row_indexed<std::uint8_t>(o);
                default: return false;
            }
        case OpCode::ScatterSet:
            switch (o.elem_dtype) {
                case DType::F32: return run_scatter_indexed<float, false>(o);
                case DType::I32: return run_scatter_indexed<std::int32_t, false>(o);
                case DType::U32: return run_scatter_indexed<std::uint32_t, false>(o);
                case DType::Bool: return run_scatter_indexed<std::uint8_t, false>(o);
                default: return false;
            }
        case OpCode::ScatterAdd:
            switch (o.elem_dtype) {
                case DType::F32: return run_scatter_indexed<float, true>(o);
                case DType::I32: return run_scatter_indexed<std::int32_t, true>(o);
                case DType::U32: return run_scatter_indexed<std::uint32_t, true>(o);
                default: return false;
            }
        // ── reduce / scan ──
        case OpCode::ReduceSum: case OpCode::ReduceMax: case OpCode::ReduceMin: {
            kernels::ptir::RedKind k = o.code == OpCode::ReduceSum ? kernels::ptir::RedKind::Sum
                      : o.code == OpCode::ReduceMax ? kernels::ptir::RedKind::Max : kernels::ptir::RedKind::Min;
            switch (o.elem_dtype) {
                case DType::F32: run_reduce<float>(o, k); return true;
                case DType::I32: run_reduce<std::int32_t>(o, k); return true;
                case DType::U32: run_reduce<std::uint32_t>(o, k); return true;
                default: return false;
            }
        }
        case OpCode::ReduceArgmax:
            switch (o.elem_dtype) {
                case DType::F32:
                    kernels::ptir::k_reduce_argmax<float><<<
                        o.rows, kernels::ptir::kTier0Block, 0, o.stream>>>(
                        static_cast<const float*>(o.in[0]),
                        static_cast<std::uint32_t*>(o.out), o.rows, o.len);
                    return true;
                case DType::I32:
                    kernels::ptir::k_reduce_argmax<std::int32_t><<<
                        o.rows, kernels::ptir::kTier0Block, 0, o.stream>>>(
                        static_cast<const std::int32_t*>(o.in[0]),
                        static_cast<std::uint32_t*>(o.out), o.rows, o.len);
                    return true;
                case DType::U32:
                    kernels::ptir::k_reduce_argmax<std::uint32_t><<<
                        o.rows, kernels::ptir::kTier0Block, 0, o.stream>>>(
                        static_cast<const std::uint32_t*>(o.in[0]),
                        static_cast<std::uint32_t*>(o.out), o.rows, o.len);
                    return true;
                default:
                    return false;
            }
        case OpCode::CumSum: case OpCode::CumProd: {
            kernels::ptir::ScanKind k = o.code == OpCode::CumSum ? kernels::ptir::ScanKind::Sum : kernels::ptir::ScanKind::Prod;
            switch (o.elem_dtype) {
                case DType::F32: run_scan<float>(o, k); return true;
                case DType::I32: run_scan<std::int32_t>(o, k); return true;
                case DType::U32: run_scan<std::uint32_t>(o, k); return true;
                default: return false;
            }
        }
        // ── sampling ──
        case OpCode::MaskApplyPacked:
            // len = vocab; k carries mask words per row (ceil(len/32)).
            kernels::ptir::k_mask_apply_packed<<<gs(o.numel), kernels::ptir::kTier0Block, 0, o.stream>>>(
                (const float*)o.in[0], (const std::uint32_t*)o.in[1], (float*)o.out, o.rows, o.len, o.k);
            return true;
        case OpCode::CausalMask:
        case OpCode::SlidingWindowMask:
        case OpCode::SinkWindowMask: {
            const kernels::ptir::Tier0StructuredMaskKind kind =
                o.code == OpCode::CausalMask
                    ? kernels::ptir::Tier0StructuredMaskKind::Causal
                    : o.code == OpCode::SlidingWindowMask
                        ? kernels::ptir::Tier0StructuredMaskKind::SlidingWindow
                        : kernels::ptir::Tier0StructuredMaskKind::SinkWindow;
            const std::uint32_t window =
                o.code == OpCode::SinkWindowMask ? o.imm3 : o.imm2;
            const std::uint32_t sink =
                o.code == OpCode::SinkWindowMask ? o.imm2 : 0;
            kernels::ptir::k_structured_position_mask<<<
                gs(static_cast<std::uint64_t>(o.rows) * o.len),
                kernels::ptir::kTier0Block,
                0,
                o.stream>>>(
                static_cast<const std::uint32_t*>(o.in[0]),
                static_cast<std::uint8_t*>(o.out),
                o.rows,
                o.len,
                kind,
                window,
                sink);
            return true;
        }
        case OpCode::Rng:   // ambient seed; rng_stream carries stream, bcast_mode = gumbel flag
            kernels::ptir::k_rng_ambient<<<gs((std::uint64_t)o.rows * o.len), kernels::ptir::kTier0Block, 0, o.stream>>>(
                (const std::uint32_t*)o.row_seeds, o.rng_stream, (float*)o.out, o.rows, o.len, o.bcast_mode);
            return true;
        case OpCode::RngKeyed:   // state=[key,ctr] in in[0]; bcast_mode = gumbel flag
            kernels::ptir::k_rng_keyed<<<gs(o.numel), kernels::ptir::kTier0Block, 0, o.stream>>>(
                (const std::uint32_t*)o.in[0], (float*)o.out, o.numel, o.bcast_mode);
            return true;
        // ── order ──
        case OpCode::PivotThreshold:
            // Container pivot_threshold(input, predicate) → bool selection
            // mask. The predicate payload is a resolved trace value (scalar
            // or per-row), never an immediate (tensor-compiler's eval/src/interp.rs).
            switch (o.pred_tag) {
                case PredTag::RankLe:
                    switch (o.pred_dtype) {
                        case DType::I32:
                            kernels::ptir::k_pivot_rankle<std::int32_t><<<o.rows, kernels::ptir::kTier0Block, 0, o.stream>>>(
                                (const float*)o.in[0], (std::uint8_t*)o.out, o.rows, o.len,
                                (const std::int32_t*)o.pred_ptr, o.pred_numel);
                            return true;
                        case DType::U32:
                            kernels::ptir::k_pivot_rankle<std::uint32_t><<<o.rows, kernels::ptir::kTier0Block, 0, o.stream>>>(
                                (const float*)o.in[0], (std::uint8_t*)o.out, o.rows, o.len,
                                (const std::uint32_t*)o.pred_ptr, o.pred_numel);
                            return true;
                        default:
                            return false;   // RankLe's k must be I32/U32 (infer.rs dtype_is_int)
                    }
                case PredTag::CummassLe:
                    kernels::ptir::k_pivot_cummassle<<<o.rows, kernels::ptir::kTier0Block, 0, o.stream>>>(
                        (const float*)o.in[0], (std::uint8_t*)o.out, o.rows, o.len,
                        (const float*)o.pred_ptr, o.pred_numel);
                    return true;
                case PredTag::ProbGe:
                    kernels::ptir::k_pivot_probge<<<gs((std::uint64_t)o.rows * o.len), kernels::ptir::kTier0Block, 0, o.stream>>>(
                        (const float*)o.in[0], (std::uint8_t*)o.out, o.rows, o.len,
                        (const float*)o.pred_ptr, o.pred_numel);
                    return true;
            }
            return false;
        // ── library kernels ──
        case OpCode::SortDesc:
            // full per-row sort = top_k with k = len (2 results, value-first).
            kernels::ptir::k_topk_rows<<<o.rows, kernels::ptir::kTier0Block, 0, o.stream>>>(
                (const float*)o.in[0], (float*)o.out, (std::uint32_t*)o.out2, o.rows, o.len, o.len);
            return true;
        case OpCode::TopK:
            kernels::ptir::k_topk_rows<<<o.rows, kernels::ptir::kTier0Block, 0, o.stream>>>(
                (const float*)o.in[0], (float*)o.out, (std::uint32_t*)o.out2, o.rows, o.len, o.k);
            return true;
        case OpCode::Matmul:
            // rows=M, len=K encoded by the runner; k=N.
            {
                dim3 blk(32, 1), grd((o.k + 31) / 32, o.rows);
                kernels::ptir::k_matmul<<<grd, blk, 0, o.stream>>>((const float*)o.in[0], (const float*)o.in[1], (float*)o.out, o.rows, o.len, o.k);
            }
            return true;
        default:
            return false;   // structural ops (chan_*/const/intrinsic_val/kernel_call/sink_call) are not launched
    }
    return false;
}

}  // namespace pie_cuda_driver::pipeline

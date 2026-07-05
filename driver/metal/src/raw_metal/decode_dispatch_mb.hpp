#pragma once
// decode_dispatch_mb.hpp — beta's M>1 launch-geometry helpers (multi-batch lane).
//
// The N>1 generalization of decode_dispatch.hpp. Pie's batch dim is N=total_tokens; the
// raw-Metal activations are token-major [N, feature], so every per-row kernel just stacks N
// rows on the batch axis. KEY INSIGHT (quantized_qmv.metal): affine_qmv_fast ALREADY batches
// over tid.x (`x += tid.x*in_vec_size; y += tid.x*out_vec_size`) — the M=1 path launches with
// 1 threadgroup in x. So an M>1 batched GEMV is just `grid.x *= N`, BIT-EXACT by construction
// (each row reproduces the M=1 accumulation) and reducing to the shipped path at N=1. No new
// GEMM kernel is needed for CORRECTNESS; the tiled affine_qmm (weight reuse across rows) is a
// PERF lever layered on AFTER the parity gate is green.
//
// Pure (no Metal types beyond Grid/Threadgroup). dispatchThreads semantics: Grid = TOTAL
// THREADS, tg = threads/threadgroup, threadgroups = ceil(Grid/tg) per axis (matches
// decode_dispatch.hpp + RawMetalContext::dispatch).

#include "decode_abi.hpp"
#include "decode_dispatch.hpp"  // M=1 helpers (qmv_dispatch, rms_dispatch, ...)
#include "mtl4_context.hpp"     // Grid, Threadgroup

namespace pie_metal_driver::raw_metal {

// affine_qmv_fast over N token rows (batched GEMV). tid.x = token row (0..N-1), tid.y = out-row
// block. grid threads = (32*N, out/4, 1) → N*(out/8) threadgroups, tg=(32,2,1). At N=1 this is
// exactly qmv_dispatch (the sealed M=1 fast path). out%8==0 holds for every qwen3.6 projection.
inline void qmv_mb_dispatch(int out_vec, int N, Grid& g, Threadgroup& tg) {
    g  = Grid{32u * uint32_t(N), uint32_t(out_vec) / 4, 1};
    tg = Threadgroup{32, 2, 1};
}

// rms_single_row over N tokens × n_rows rows-per-token (e.g. per-head q/k norm). One
// threadgroup per row; rows stack token-major [N*n_rows, row_size]. grid.x = (row_size/4)*n_rows*N.
inline void rms_mb_dispatch(int row_size, int n_rows, int N, Grid& g, Threadgroup& tg) {
    const uint32_t t = uint32_t(row_size) / 4;  // N_READS = 4
    g  = Grid{t * uint32_t(n_rows) * uint32_t(N), 1, 1};
    tg = Threadgroup{t, 1, 1};
}

// Elementwise over N rows × `width` channels (residual_add / silu_mul / attn_gate). Token-major
// [N, width]; one thread per (row, channel) folded onto grid.x. tg 256.
inline void elementwise_mb_dispatch(int width, int N, Grid& g, Threadgroup& tg) {
    g  = Grid{uint32_t(width) * uint32_t(N), 1, 1};
    tg = Threadgroup{256, 1, 1};
}

// embed_gather_mb over N tokens: thread (channel k, token m). Token m gathers id[m].
// out token-major [N, hidden]. grid=(hidden, N, 1), tg=(256,1,1).
inline void embed_mb_dispatch(int hidden, int N, Grid& g, Threadgroup& tg) {
    g  = Grid{uint32_t(hidden), uint32_t(N), 1};
    tg = Threadgroup{256, 1, 1};
}

// rope over N tokens: pos.x = freq idx, pos.y = head, pos.z = token row. Token m reads
// position_ids[m] (per-row IO read). grid=(rotary/2, n_heads, N), tg=(rotary/2,1,1).
inline void rope_mb_dispatch(int rotary_dims, int n_heads, int N, Grid& g, Threadgroup& tg) {
    const uint32_t half = uint32_t(rotary_dims) / 2;
    g  = Grid{half, uint32_t(n_heads), uint32_t(N)};
    tg = Threadgroup{half, 1, 1};
}

// sdpa_paged_decode: one threadgroup per (q_head, query row). grid=(n_q_heads*1024, N, 1),
// tg=(1024,1,1). Causal bound per row = position_ids[row]; request = req_of_token[row].
inline void sdpa_paged_dispatch(int n_q_heads, int N, Grid& g, Threadgroup& tg) {
    g  = Grid{uint32_t(n_q_heads) * 1024u, uint32_t(N), 1};
    tg = Threadgroup{1024, 1, 1};
}

// kv_append (paged, delta's kernel): one thread per (channel, kv_head, token). grid=
// (head_dim, n_kv_heads, N). Token m scatters to its phys_slot(position_ids[m]).
inline void kv_append_mb_dispatch(int head_dim, int n_kv_heads, int N, Grid& g, Threadgroup& tg) {
    g  = Grid{uint32_t(head_dim), uint32_t(n_kv_heads), uint32_t(N)};
    tg = Threadgroup{uint32_t(head_dim), 1, 1};
}

}  // namespace pie_metal_driver::raw_metal

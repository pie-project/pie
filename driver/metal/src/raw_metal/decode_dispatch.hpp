#pragma once
// decode_dispatch.hpp — per-kernel launch geometry (Grid/Threadgroup) for the
// raw-Metal decode DAG at qwen3.6 shapes. delta owns this (derived from the
// ported kernels' [[thread_position]] index contracts).
//
// dispatchThreads semantics: Grid = TOTAL THREADS (matches RawMetalContext::
// dispatch + alpha's harness `bench_kernel`), tg = threads per threadgroup.
// Threadgroups = ceil(Grid / tg) per axis.
//
// Validated running via alpha's microbench: rms_single_row + affine_qmv_fast.
// The rest are derived from each kernel's index derivation (see comments) and
// gate against charlie's pos-7 golden on the integration run.
//
// GdnCore is beta's kernel — its grid {32,Vd,R·Vh}/tg {32,4,1} lives in
// gdn_core.metal; beta confirms the dispatchThreads conversion. Not emitted here.

#include "decode_abi.hpp"
#include "mtl4_context.hpp"  // Grid, Threadgroup

namespace pie_metal_driver::raw_metal {

// affine_qmv_fast (all Qmv* kinds): tg=(32,2,1); grid threads=(32, N/4, 1)
// → N/8 threadgroups. Requires N%8==0 (holds for every qwen3.6 projection,
// incl. lm_head N=vocab=248320). K is a bound constant, not a launch dim.
inline void qmv_dispatch(int N, Grid& g, Threadgroup& tg) {
    g  = Grid{32, uint32_t(N) / 4, 1};
    tg = Threadgroup{32, 2, 1};
}

// rms_single_row (Rms / FinalRms / QNorm / KNorm): one threadgroup per row,
// row_size/N_READS threads (N_READS=4). The row index is the threadgroup-x
// position (gid), so multi-row norms (per-head Q/K) stack rows on grid.x.
//   * Rms/FinalRms: n_rows=1, row_size=hidden(1024) → grid=(256,1,1) tg=(256,1,1)
//   * QNorm: n_rows=n_q_heads(8), row_size=head_dim(256) → grid=(512,1,1) tg=(64,1,1)
//   * KNorm: n_rows=n_kv_heads(2), row_size=head_dim(256) → grid=(128,1,1) tg=(64,1,1)
inline void rms_dispatch(int row_size, int n_rows, Grid& g, Threadgroup& tg) {
    const uint32_t t = uint32_t(row_size) / 4;  // N_READS = 4
    g  = Grid{t * uint32_t(n_rows), 1, 1};
    tg = Threadgroup{t, 1, 1};
}

// embed_gather_4bit: one thread per output channel (hidden), tg flexible.
inline void embed_dispatch(int hidden, Grid& g, Threadgroup& tg) {
    g  = Grid{uint32_t(hidden), 1, 1};
    tg = Threadgroup{256, 1, 1};
}

// sdpa_vector_decode: tid.x = q_batch_head_idx (threadgroup-x), one threadgroup
// (1024 threads) per query head → grid=(n_q_heads*1024, 1, 1), tg=(1024,1,1).
inline void sdpa_dispatch(int n_q_heads, Grid& g, Threadgroup& tg) {
    g  = Grid{uint32_t(n_q_heads) * 1024, 1, 1};
    tg = Threadgroup{1024, 1, 1};
}

// rope_neox_decode: pos.x = freq index (0..rotary_dims/2-1), pos.y = head.
// In-place single-tensor → dispatched once for Q (n_q_heads) and once for K
// (n_kv_heads). grid=(rotary_dims/2, n_heads, 1), tg=(rotary_dims/2, 1, 1).
inline void rope_dispatch(int rotary_dims, int n_heads, Grid& g, Threadgroup& tg) {
    const uint32_t half = uint32_t(rotary_dims) / 2;
    g  = Grid{half, uint32_t(n_heads), 1};
    tg = Threadgroup{half, 1, 1};
}

// kv_append: tid=(head_dim, n_kv_heads) elementwise scatter to the page.
inline void kv_append_dispatch(int head_dim, int n_kv_heads, Grid& g, Threadgroup& tg) {
    g  = Grid{uint32_t(head_dim), uint32_t(n_kv_heads), 1};
    tg = Threadgroup{uint32_t(head_dim), 1, 1};
}

}  // namespace pie_metal_driver::raw_metal

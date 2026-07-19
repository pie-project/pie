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

namespace pie::metal {

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

// q_gate_split: deinterleave qg -> Q + gate. one thread per (channel, query head).
inline void q_split_dispatch(int head_dim, int n_q, Grid& g, Threadgroup& tg) {
    g  = Grid{uint32_t(head_dim), uint32_t(n_q), 1};
    tg = Threadgroup{uint32_t(head_dim), 1, 1};
}

// attn_gate: attn *= sigmoid(gate), elementwise over n_q*head_dim (head-major).
inline void attn_gate_dispatch(int n_q, int head_dim, Grid& g, Threadgroup& tg) {
    g  = Grid{uint32_t(n_q) * uint32_t(head_dim), 1, 1};
    tg = Threadgroup{256, 1, 1};
}

// residual_add (AttnResid / LayerOut): Out = X + Residual, elementwise over hidden.
inline void residual_dispatch(int hidden, Grid& g, Threadgroup& tg) {
    g  = Grid{uint32_t(hidden), 1, 1};
    tg = Threadgroup{256, 1, 1};
}

// silu_mul (Swiglu): Out = silu(gate)*up, elementwise over the MLP intermediate.
inline void silu_mul_dispatch(int intermediate, Grid& g, Threadgroup& tg) {
    g  = Grid{uint32_t(intermediate), 1, 1};
    tg = Threadgroup{256, 1, 1};
}

// gated_rms (GatedRms -> golden gdn_core): one threadgroup per value-head, V_d lanes
// cooperatively reduce. grid=(V_d, V_h, 1), tg=(V_d, 1, 1). head=tgpos.y, lane=lid.
inline void gated_rms_dispatch(int v_heads, int v_dim, Grid& g, Threadgroup& tg) {
    g  = Grid{uint32_t(v_dim), uint32_t(v_heads), 1};
    tg = Threadgroup{uint32_t(v_dim), 1, 1};
}

// dense_gemv (GdnInA / GdnInB): cooperative simdgroup K-reduction — ONE simdgroup
// (32 lanes) per output row. The lanes stride K (coalesced loads, latency hidden
// across 32 lanes) then simd_sum reduces. Replaces the old serial grid=(N,1,1)
// tg=(N,1,1) (one 16-thread group serially walking K=1024 → ~93µs/disp, 50× the
// launch floor, 43% of the decode step across GdnInA/B×36). Bit-identical output
// (bf16 round absorbs the sub-ULP reassociation diff; verified 0/16). grid=(32,N,1)
// tg=(32,1,1) → threadgroup_position.y = output row. Pairs with dense_gemv_coop.
inline void dense_gemv_dispatch(int N, Grid& g, Threadgroup& tg) {
    g  = Grid{32u, uint32_t(N), 1};
    tg = Threadgroup{32u, 1, 1};
}

}  // namespace pie::metal

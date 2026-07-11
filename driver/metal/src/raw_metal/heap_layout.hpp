#pragma once
// heap_layout.hpp — delta's MTLHeap region layout for the raw-Metal decode path.
//
// Computes fixed byte offsets for the single resident heap (decode_abi Region:
// Weights / KV / State / Scratch / IO) from DecodeGeometry. Pure C++ (no Metal):
// the offset math is testable standalone; the actual heap_alloc/arg_bind wiring
// (heap_bind.cpp) layers on top via alpha's RawMetalContext.
//
// Sizing rules:
//   * Weights — sized from the LOADER manifest (actual per-tensor bytes), not
//     re-derived shapes (the GDN in/out projections are model-specific; trust the
//     loader). Caller feeds total weight bytes from the manifest.
//   * KV / State / Scratch / IO — derived EXACTLY from geometry here (I4: KV
//     append-only, GDN state resident in-place, scratch ping-pong pool).
//
// All regions 256-aligned (Metal buffer-offset alignment). One heap, fixed offsets,
// resident once (I2).

#include <cstddef>
#include <cstdint>
#include "decode_abi.hpp"

namespace pie_metal_driver::raw_metal {

inline constexpr size_t align_up(size_t n, size_t a = 256) {
    return (n + a - 1) / a * a;
}

// Bytes for one affine-quantized linear [out, in] at (group, bits=4).
struct QLinearBytes {
    size_t weight;   // packed nibbles: out * in/2  (8 per uint32)
    size_t scales;   // out * (in/group) * 2  (bf16)
    size_t biases;   // out * (in/group) * 2  (bf16)
    size_t total() const { return align_up(weight) + align_up(scales) + align_up(biases); }
};
inline QLinearBytes qlinear_bytes(int out, int in, int group, int bits = 4) {
    const size_t vals = size_t(out) * in;
    return QLinearBytes{
        /*weight*/ vals * bits / 8,
        /*scales*/ size_t(out) * (in / group) * 2,
        /*biases*/ size_t(out) * (in / group) * 2,
    };
}

// Per-region byte sizes + base offsets within the single heap.
struct HeapPlan {
    size_t weights_off = 0, weights_bytes = 0;
    size_t kv_off = 0,      kv_bytes = 0;
    size_t state_off = 0,   state_bytes = 0;
    size_t scratch_off = 0, scratch_bytes = 0;
    size_t io_off = 0,      io_bytes = 0;
    // Phase 1b/3 paged-KV bridge (additive; both 0 at total_pages==0 &&
    // max_requests<=1, preserving the sealed M=1 `total` byte-for-byte).
    size_t mb_io_off = 0,      mb_io_bytes = 0;
    size_t kv_pool_off = 0,    kv_pool_bytes = 0;
    size_t total = 0;

    // sizing inputs surfaced for tests / logging
    size_t scratch_slot_bytes = 0;  // one ping-pong buffer
    size_t kv_per_layer = 0;        // k+v for one full-attn layer (M=1 HND ring)
    size_t state_per_layer = 0;     // conv+recurrent for one GDN layer
    size_t kv_pool_per_layer = 0;   // k+v for one full-attn layer (paged NHD pool)
    size_t max_page_refs = 0;       // flattened CSR capacity for one paged fire
};

// max_ctx: Phase-0 single-stream context window (KV capacity per stream).
// state_dtype_bytes: GDN recurrent/conv state element size (fp32 = 4).
// act_dtype_bytes: activation element size (bf16 = 2).
inline HeapPlan plan_heap(const DecodeGeometry& g,
                          size_t weights_bytes_from_manifest,
                          int max_ctx = 4096,
                          int state_dtype_bytes = 4,
                          int act_dtype_bytes = 2) {
    HeapPlan p;

    // ── Weights (load-once RO) ── sized from the loader manifest.
    p.weights_bytes = align_up(weights_bytes_from_manifest);

    // ── KV (append-only, full-attn layers only) ──
    int n_full = 0;
    for (int L = 0; L < g.n_layers; ++L) n_full += DecodeGeometry::is_full_attn(L) ? 1 : 0;
    // per full-attn layer: k + v, each [n_kv_heads, max_ctx, head_dim] act-dtype.
    p.kv_per_layer = size_t(2) * g.n_kv_heads * max_ctx * g.head_dim * act_dtype_bytes;
    p.kv_bytes = align_up(size_t(n_full) * p.kv_per_layer);

    // ── State (GDN resident; per-slot slabs, S = g.max_slots; in-place at S=1) ──
    int n_gdn = g.n_layers - n_full;
    // conv_state [gdn_conv_dim, gdn_conv_k] is PING-PONG (RO in + new_conv_state out; beta's
    // co-fix — in-place conv shift races the Kc-tap reads). recurrent_state [Vh,Vd,Kd] is
    // in-place (each (v-head,v-dim) row owned by one threadgroup → race-free).
    // S>1: each slot's state is packed at the NATURAL (unpadded) per-slot stride so beta's
    // gdn_core_slotted indexes slot*(Kc*CDIM)/(Hv*Vd*Dk); only the whole slab is align_up'd.
    // At max_slots=1 this is byte-identical to the sealed single-slot layout.
    const size_t conv_state = size_t(g.gdn_conv_dim) * g.gdn_conv_k * state_dtype_bytes;
    const size_t recur_state =
        size_t(g.gdn_v_heads) * g.gdn_v_dim * g.gdn_k_dim * state_dtype_bytes;
    const size_t slots = size_t(g.max_slots);
    p.state_per_layer =
        2 * align_up(slots * conv_state)   // ConvState + ConvStateOut (ping-pong), S slots each
        + align_up(slots * recur_state);   // RecurrentState (in-place), S slots
    p.state_bytes = align_up(size_t(n_gdn) * p.state_per_layer);

    // ── Scratch (activation ping-pong pool) ──
    // One slot must hold the largest M=1 activation that ping-pongs through the DAG.
    // Logits (vocab) live in IO, not scratch; the largest scratch activation is the
    // widest intermediate projection output.
    int widest = g.intermediate;                 // MLP gate/up out
    widest = widest > g.gdn_conv_dim ? widest : g.gdn_conv_dim;   // GDN in-proj out
    widest = widest > g.n_q_heads * g.head_dim ? widest : g.n_q_heads * g.head_dim; // q
    // The sealed M=1 allocation stays byte-identical.  Paged fires store
    // token-major [max_tokens,width] activations in the same colored buffers.
    const size_t scratch_rows = g.paged_kv_enabled
                                  ? size_t(g.max_tokens < 1 ? 1 : g.max_tokens)
                                  : 1u;
    p.scratch_slot_bytes = align_up(size_t(widest) * scratch_rows * act_dtype_bytes);
    p.scratch_bytes = align_up(size_t(SCRATCH_POOL) * p.scratch_slot_bytes);

    // ── IO (per-token scalars + logits; scalars widen to max_tokens at M>1) ──
    // TokenId/Position/SeqLen/NextToken: u32[max_tokens] each (slot-aligned). Logits: f32[vocab].
    // At max_tokens=1 each scalar is align_up(4) — byte-identical to the sealed single-token IO.
    const size_t scalars = 4 * align_up(size_t(4) * g.max_tokens);
    // lm_head writes bf16.  The historical M=1 allocation was four bytes per
    // logit; retain it exactly there, while paged output is densely [N,vocab].
    const size_t logits = g.paged_kv_enabled
                              ? align_up(size_t(g.vocab) * scratch_rows * act_dtype_bytes)
                              : align_up(size_t(g.vocab) * 4);
    p.io_bytes = align_up(scalars + logits);

    // ── MbIo (Phase 1b/3; additive — zero bytes unless explicitly opted
    //    into via `g.paged_kv_enabled`, byte-identical to no region at all
    //    for the sealed M=1 path) — the multi-batch CSR buffers (IoSlot::
    //    QoIndptr..ReqOfToken). Sized from `g.max_requests` (R) / `g.
    //    max_tokens` (N) / `g.total_pages` (the pool's physical page COUNT
    //    — KvPageIndices' flat per-batch list can reference at most that
    //    many distinct physical pages). ──
    if (g.paged_kv_enabled) {
        const size_t r1 = size_t(g.max_requests) + 1;
        const size_t qo_indptr        = align_up(r1 * 4);
        const size_t kv_page_indptr   = align_up(r1 * 4);
        // A physical page may legitimately occur in multiple request CSR
        // segments (shared prefixes/forks).  Capacity is therefore references,
        // not unique physical pages.
        p.max_page_refs = size_t(g.max_requests) * size_t(g.total_pages);
        const size_t kv_page_indices  = align_up(p.max_page_refs * 4);
        const size_t kv_last_page_len = align_up(size_t(g.max_requests) * 4);
        const size_t rs_slot_ids      = align_up(size_t(g.max_requests) * 4);
        const size_t rs_slot_flags    = align_up(size_t(g.max_requests) * 1);
        const size_t req_of_token     = align_up(size_t(g.max_tokens) * 4);
        const size_t slot_of_token    = align_up(size_t(g.max_tokens) * 4);
        const size_t w_page           = align_up(size_t(g.max_tokens) * 4);
        const size_t w_off            = align_up(size_t(g.max_tokens) * 4);
        p.mb_io_bytes = align_up(qo_indptr + kv_page_indptr + kv_page_indices +
                                 kv_last_page_len + rs_slot_ids + rs_slot_flags +
                                 req_of_token + slot_of_token + w_page + w_off);
    }

    // ── KvPagePool (Phase 1b/3; additive — zero bytes unless explicitly
    //    opted into via `g.paged_kv_enabled`, byte-identical to no region
    //    at all for the sealed M=1 path). A SEPARATE NHD pool (page-major:
    //    [num_pages, page_size, n_kv_heads, head_dim]) from the M=1 HND
    //    contiguous ring above — kv_append_paged / sdpa_paged read/write
    //    THIS region; the M=1 ring is untouched. ──
    if (g.paged_kv_enabled) {
        p.kv_pool_per_layer = size_t(2) * size_t(g.total_pages) * size_t(g.kv_page_size) *
                              g.n_kv_heads * g.head_dim * act_dtype_bytes;
        p.kv_pool_bytes = align_up(size_t(n_full) * p.kv_pool_per_layer);
    }

    // ── Lay out regions back-to-back, each 256-aligned ──
    size_t off = 0;
    p.weights_off = off; off = align_up(off + p.weights_bytes);
    p.kv_off      = off; off = align_up(off + p.kv_bytes);
    p.state_off   = off; off = align_up(off + p.state_bytes);
    p.scratch_off = off; off = align_up(off + p.scratch_bytes);
    p.io_off      = off; off = align_up(off + p.io_bytes);
    p.mb_io_off   = off; off = align_up(off + p.mb_io_bytes);
    p.kv_pool_off = off; off = align_up(off + p.kv_pool_bytes);
    p.total = off;
    return p;
}

}  // namespace pie_metal_driver::raw_metal

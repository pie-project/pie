// PTIR sampling-IR foundational ops — Metal compute kernels.
//
// Bit-exact ports of the canonical Rust reference semantics in
// interface/sampling-ir/src/eval.rs. These are the Metal duals of charlie's
// CUDA sampling-IR primitives; both backends must agree byte-for-byte with
// echo's golden reference vectors (the cross-backend oracle).
//
// Compiled at RUNTIME via [MTLDevice newLibraryWithSource:] — this box is
// Command-Line-Tools-only (no offline `metal`/`metallib` compiler), which is
// also the raw_metal Phase-0 constraint.

#include <metal_stdlib>
using namespace metal;

// ── mask_apply_packed (Op::MaskApply) ───────────────────────────────────────
// Vector form: out[j] = bit_j(mask) ? logits[j] : -inf,
//   bit_j = (mask[j>>5] >> (j&31)) & 1  (word j/32, bit j%32).
// `mask` is a packed [ceil(n/32)] u32 bitmap. Tail bits >= n are never read.
kernel void mask_apply_packed(
    device const float* logits [[buffer(0)]],
    device const uint*  mask   [[buffer(1)]],
    device float*       out    [[buffer(2)]],
    constant uint&      n      [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    uint bit = (mask[gid >> 5] >> (gid & 31u)) & 1u;
    out[gid] = (bit == 1u) ? logits[gid] : -INFINITY;
}

// Matrix form: logits [rows, vocab] row-major; mask [rows, ceil(vocab/32)]
// row-major (per-row packed bitmap, row stride = words_per_row). Applies the
// per-row bitmask independently to each row. grid = rows * vocab.
kernel void mask_apply_packed_matrix(
    device const float* logits        [[buffer(0)]],
    device const uint*  mask          [[buffer(1)]],
    device float*       out           [[buffer(2)]],
    constant uint&      vocab         [[buffer(3)]],
    constant uint&      words_per_row [[buffer(4)]],
    constant uint&      total         [[buffer(5)]],  // rows * vocab
    uint gid [[thread_position_in_grid]]) {
    if (gid >= total) return;
    uint row = gid / vocab;
    uint col = gid % vocab;
    device const uint* row_mask = mask + row * words_per_row;
    uint bit = (row_mask[col >> 5] >> (col & 31u)) & 1u;
    out[gid] = (bit == 1u) ? logits[gid] : -INFINITY;
}

// ── dselect (Op::Select) ─────────────────────────────────────────────────────
// out[i] = cond[i] ? a[i] : b[i], with per-operand length-1 broadcast
// (len == 1 => index 0, else index i). F32 payload. `cond` is a byte-per-lane
// bool (uchar: 0/1). Matches eval.rs `select` for the F32 arm.
kernel void dselect_f32(
    device const uchar* cond  [[buffer(0)]],
    device const float* a     [[buffer(1)]],
    device const float* b     [[buffer(2)]],
    device float*       out   [[buffer(3)]],
    constant uint&      n     [[buffer(4)]],
    constant uint&      lc    [[buffer(5)]],  // cond length
    constant uint&      la    [[buffer(6)]],  // a length
    constant uint&      lb    [[buffer(7)]],  // b length
    uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    uint ic = (lc == 1u) ? 0u : gid;
    uint ia = (la == 1u) ? 0u : gid;
    uint ib = (lb == 1u) ? 0u : gid;
    out[gid] = (cond[ic] != 0) ? a[ia] : b[ib];
}

// ── broadcast_matrix (Op::Broadcast) ─────────────────────────────────────────
// Row-major rank<=2 shape-directed broadcast, left-aligned: a source of
// extents (src_rows, src_cols) fills a destination (dst_rows, dst_cols); any
// source axis of extent 1 replicates. Covers scalar->[k,vocab] (1,1),
// per-row [m]->[m,n] (src as (m,1)), and row-vector [n]->[m,n] (src as (1,n)).
kernel void broadcast_matrix_f32(
    device const float* src      [[buffer(0)]],
    device float*       out      [[buffer(1)]],
    constant uint&      src_rows [[buffer(2)]],
    constant uint&      src_cols [[buffer(3)]],
    constant uint&      dst_rows [[buffer(4)]],
    constant uint&      dst_cols [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    uint total = dst_rows * dst_cols;
    if (gid >= total) return;
    uint r = gid / dst_cols;
    uint c = gid % dst_cols;
    uint sr = (src_rows == 1u) ? 0u : r;
    uint sc = (src_cols == 1u) ? 0u : c;
    out[gid] = src[sr * src_cols + sc];
}

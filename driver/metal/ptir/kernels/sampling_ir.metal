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

// Matrix form: logits [rows, vocab] row-major; mask is a SINGLE packed word-row
// [ceil(vocab/32)] broadcast across ALL rows (the pinned PTIR op contract, tag
// 0x65) — bit index = column (j % vocab), the SAME words for every row. (Per-row
// DISTINCT masks are the composed bool `dselect` form, not this packed op.)
// grid = rows * vocab.
kernel void mask_apply_packed_matrix(
    device const float* logits [[buffer(0)]],
    device const uint*  mask   [[buffer(1)]],
    device float*       out    [[buffer(2)]],
    constant uint&      vocab  [[buffer(3)]],
    constant uint&      total  [[buffer(4)]],  // rows * vocab
    uint gid [[thread_position_in_grid]]) {
    if (gid >= total) return;
    uint col = gid % vocab;
    uint bit = (mask[col >> 5] >> (col & 31u)) & 1u;
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

// ── elementwise: unary + binary (scalar/len-1 broadcast) ─────────────────────
// eval.rs: unary map_f32; binary zip_f32 with per-operand len-1 broadcast
// (len == 1 => index 0). IEEE-exact (fast-math is disabled in the harness).

kernel void neg_f32(
    device const float* a   [[buffer(0)]],
    device float*       out [[buffer(1)]],
    constant uint&      n   [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    out[gid] = -a[gid];
}

#define PTIR_BINARY(NAME, EXPR)                                            \
kernel void NAME(                                                         \
    device const float* a   [[buffer(0)]],                               \
    device const float* b   [[buffer(1)]],                               \
    device float*       out [[buffer(2)]],                               \
    constant uint&      n   [[buffer(3)]],                               \
    constant uint&      la  [[buffer(4)]],                               \
    constant uint&      lb  [[buffer(5)]],                               \
    uint gid [[thread_position_in_grid]]) {                              \
    if (gid >= n) return;                                                 \
    float x = a[(la == 1u) ? 0u : gid];                                  \
    float y = b[(lb == 1u) ? 0u : gid];                                  \
    out[gid] = (EXPR);                                                    \
}
PTIR_BINARY(add_f32, x + y)
PTIR_BINARY(sub_f32, x - y)
PTIR_BINARY(mul_f32, x * y)
PTIR_BINARY(div_f32, x / y)
PTIR_BINARY(max_elem_f32, fmax(x, y))
PTIR_BINARY(min_elem_f32, fmin(x, y))
#undef PTIR_BINARY

// Comparisons → bool (uchar 0/1), same broadcast rule.
#define PTIR_CMP(NAME, OP)                                                \
kernel void NAME(                                                        \
    device const float* a   [[buffer(0)]],                               \
    device const float* b   [[buffer(1)]],                               \
    device uchar*       out [[buffer(2)]],                               \
    constant uint&      n   [[buffer(3)]],                               \
    constant uint&      la  [[buffer(4)]],                               \
    constant uint&      lb  [[buffer(5)]],                               \
    uint gid [[thread_position_in_grid]]) {                              \
    if (gid >= n) return;                                                 \
    float x = a[(la == 1u) ? 0u : gid];                                  \
    float y = b[(lb == 1u) ? 0u : gid];                                  \
    out[gid] = (x OP y) ? 1 : 0;                                          \
}
PTIR_CMP(gt_f32, >)
PTIR_CMP(ge_f32, >=)
PTIR_CMP(eq_f32, ==)
#undef PTIR_CMP

// ── reductions over the last axis (per-row for rank >= 2) ────────────────────
// One thread per row scans sequentially to match the Rust fold's accumulation
// order exactly (tree reductions would reassociate and diverge). grid = rows.

kernel void reduce_sum_rows(
    device const float* in  [[buffer(0)]],
    device float*       out [[buffer(1)]],
    constant uint&      rows [[buffer(2)]],
    constant uint&      len  [[buffer(3)]],
    uint r [[thread_position_in_grid]]) {
    if (r >= rows) return;
    float acc = 0.0f;
    for (uint j = 0; j < len; ++j) acc = acc + in[r * len + j];
    out[r] = acc;
}

kernel void reduce_max_rows(
    device const float* in  [[buffer(0)]],
    device float*       out [[buffer(1)]],
    constant uint&      rows [[buffer(2)]],
    constant uint&      len  [[buffer(3)]],
    uint r [[thread_position_in_grid]]) {
    if (r >= rows) return;
    float acc = -INFINITY;
    for (uint j = 0; j < len; ++j) acc = fmax(acc, in[r * len + j]);
    out[r] = acc;
}

kernel void reduce_min_rows(
    device const float* in  [[buffer(0)]],
    device float*       out [[buffer(1)]],
    constant uint&      rows [[buffer(2)]],
    constant uint&      len  [[buffer(3)]],
    uint r [[thread_position_in_grid]]) {
    if (r >= rows) return;
    float acc = INFINITY;
    for (uint j = 0; j < len; ++j) acc = fmin(acc, in[r * len + j]);
    out[r] = acc;
}

// Per-row argmax -> I32 token; strict '>' with first-max-wins (init best=-inf,
// bi=0) exactly matching eval.rs `argmax`.
kernel void reduce_argmax_rows(
    device const float* in  [[buffer(0)]],
    device int*         out [[buffer(1)]],
    constant uint&      rows [[buffer(2)]],
    constant uint&      len  [[buffer(3)]],
    uint r [[thread_position_in_grid]]) {
    if (r >= rows) return;
    float best = -INFINITY;
    int bi = 0;
    for (uint j = 0; j < len; ++j) {
        float x = in[r * len + j];
        if (x > best) { best = x; bi = (int)j; }
    }
    out[r] = bi;
}

// ── scans over the last axis (per-row for rank >= 2) ─────────────────────────
kernel void cumsum_rows(
    device const float* in  [[buffer(0)]],
    device float*       out [[buffer(1)]],
    constant uint&      rows [[buffer(2)]],
    constant uint&      len  [[buffer(3)]],
    uint r [[thread_position_in_grid]]) {
    if (r >= rows) return;
    float acc = 0.0f;
    for (uint j = 0; j < len; ++j) { acc = acc + in[r * len + j]; out[r * len + j] = acc; }
}

kernel void cumprod_rows(
    device const float* in  [[buffer(0)]],
    device float*       out [[buffer(1)]],
    constant uint&      rows [[buffer(2)]],
    constant uint&      len  [[buffer(3)]],
    uint r [[thread_position_in_grid]]) {
    if (r >= rows) return;
    float acc = 1.0f;
    for (uint j = 0; j < len; ++j) { acc = acc * in[r * len + j]; out[r * len + j] = acc; }
}

// ── indexing: gather / gather_row (invalid index -> fill 0) ───────────────────
// Op::Gather: out[j] = (0 <= idx[j] < src_len) ? src[idx[j]] : 0.  grid = k.
kernel void gather_f32(
    device const float* src     [[buffer(0)]],
    device const int*   idx     [[buffer(1)]],
    device float*       out     [[buffer(2)]],
    constant uint&      src_len [[buffer(3)]],
    constant uint&      k       [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= k) return;
    int i = idx[gid];
    out[gid] = (i >= 0 && (uint)i < src_len) ? src[i] : 0.0f;
}

// Op::GatherRow: per-row column pick out[r] = src[r, idx[r]] (invalid col -> 0).
// The lossless spec-verify accept-ratio p[i, draft[i]]. grid = rows.
kernel void gather_row_f32(
    device const float* src  [[buffer(0)]],
    device const int*   idx  [[buffer(1)]],
    device float*       out  [[buffer(2)]],
    constant uint&      rows [[buffer(3)]],
    constant uint&      n    [[buffer(4)]],
    uint r [[thread_position_in_grid]]) {
    if (r >= rows) return;
    int c = idx[r];
    out[r] = (c >= 0 && (uint)c < n) ? src[r * n + c] : 0.0f;
}

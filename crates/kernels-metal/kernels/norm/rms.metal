// Raw-Metal port of MLX rms_single_row, scoped to Phase-0 decode (M=1, one token row).
//
// Source: mlx/backend/metal/kernels/rms_norm.metal (rms_single_row).
// Port notes:
//   * Decode is single-row -> only the single_row variant is needed (no rms_looped,
//     no vjp/backward kernels). gid selects the row; for B=1 decode gid==0.
//   * eps/axis_size/w_stride are STATIC geometry (hidden=1024, w_stride=1, eps=1e-6),
//     NOT per-token IO scalars, so they may stay as a constant params buffer with no
//     impact on the byte-identical-CB / encode-overlap invariant (decode_abi I1).
//   * bfloat is native on Metal 4 (macOS 26) -> MLX's bf16.h emulation is dropped.
//   * Output is RMSNorm(x) * w, matching MLX semantics (weight always applied here).
//   * The gemma `(1 + weight)` convention is PER MODEL and per norm, read off
//     the deployment row rather than assumed: `norm_unit_offset` is `true` for
//     gemma_3 and `false` for qwen_3_5 and everything llama-like (see
//     `model/src/*/project.rs`), and `batch/geometry.rs` states it from the row
//     instead of letting `DecodeGeometry::default`'s `true` fall through. The
//     C++ port note this replaces claimed qwen3.5/qwen36 folded the +1 into
//     EVERY RMSNorm, citing a `model/qwen36.cpp` that no longer exists; its
//     weights are absolute (input_layernorm averages 1.24, `model.norm` 4.31 on
//     the 0.8B checkpoint), so a `1 + w` gain there is finite and quiet --
//     a ~80% per-norm error the residual stream compounds. MLX folds the +1 by
//     materializing `mx::add(weight, 1.0f)` (float) before fast::rms_norm, so
//     the effective gain is float (1.0f + weight); we apply it in float here
//     when plus_one != 0. The gated-RMSNorm (gated_rms.metal) never folds
//     (raw gate_norm weight).
//
// THE FIVE SCALARS ARE FIVE SCALARS, NOT A STRUCT.
//
// They took `constant RmsParams& p [[buffer(3)]]` -- MLX's layout out of
// `norm/rms_params.h`, and the shape `kernels-vulkan` and `kernels-wgpu` then
// copied. Every one of the five words is live, so nothing is deleted here:
// `norm::rms_single_row` and its four siblings state them as `eps: Const<f32>`,
// `axis: Const<i32>`, `w_stride: Const<u32>`, `plus_one: Const<u32>` and
// `gain: Const<f32>` rather than forwarding `ctx.params()` whole, and bind five
// `setBytes` where they bound a staged block. Words 0 through 4 of the
// statement's run are the same five numbers they always were, reached by index
// instead of by struct field. Three of them -- the stride, the gemma `+1` flag
// and the gain -- were read HERE and named nowhere in Rust until now, which is
// the whole of what this change buys.
//
// The buffer indices ASCEND IN THE STRUCT'S ORDER, because that order is the
// statement's, and everything the struct used to sit in front of keeps the END
// of its list: `row_pitch` was buffer 4 behind a block of five words and is
// buffer 8 in front of five scalars, `r` moves 4 -> 8 and `s` 5 -> 9. That is
// the same place in each case -- a conditional binding after the unconditional
// ones -- and it is what keeps the five scalars at buffers 3..7 in ALL FIVE
// entrypoints rather than at a different index in each.
//
// `axis_size`, `w_stride` and `plus_one` are `uint` here while their marks are
// `Const<i32>`, `Const<u32>` and `Const<u32>`: the run is a `Vec<u32>` and the
// BITS are the value, and the mark's Rust type is about what the BODY may do
// with the number -- the axis is signed because `rms_threads` refuses a
// non-positive extent, and the other two are never read in Rust at all.

#include <metal_stdlib>
using namespace metal;

#include "rms_reduce.h"

// One threadgroup owns one row, striding it in chunks of `tg_size * N_READS`.
//
// It used to own the row in ONE chunk, which made the threadgroup
// `ceil(axis_size / N_READS)` threads and silently required
// `axis_size <= N_READS * 1024`. Every model here cleared that until a hidden
// of 5120 did not: `rms_mb_dispatch` asked for 1280 threads, the pipeline
// allows 1024, and the dispatch that could not be made is the whole reason
// Qwen3.6-27B and gemma-4-31b answered nonsense.
//
// The stride costs a loop bound at every width that used to fit, which runs
// once; nothing else about the arithmetic moves, and the reduction is over the
// same values in the same order within a lane.
template <typename T, int N_READS>
METAL_FUNC void rms_row_body(
    const device T* x, const device T* w, device T* out,
    float eps, uint axis_size, uint w_stride, uint plus_one, float gain,
    size_t row_base,
    threadgroup float* inv_rms, threadgroup float* partials,
    uint lid, uint simd_lane, uint simd_group, uint tg_size) {
  const uint span = tg_size * uint(N_READS);

  const device T* xr = x + row_base;
  float acc = 0.0f;
  for (uint start = lid * uint(N_READS); start < axis_size; start += span) {
    acc += rms_lane_square_sum_at<T, N_READS>(xr + start, axis_size, start);
  }
  const float inv = rms_inv_from_lane_sum(
      acc, axis_size, eps, inv_rms, partials, simd_lane, simd_group);

  device T* outr = out + row_base;
  for (uint start = lid * uint(N_READS); start < axis_size; start += span) {
    const device T* xs = xr + start;
    const device T* ws = w + w_stride * start;
    device T* os = outr + start;
    for (int i = 0; i < N_READS; i++) {
      if (start + uint(i) < axis_size) {
        T wv = T(gain * (plus_one ? (1.0f + float(ws[w_stride * i]))
                                  : float(ws[w_stride * i])));
        os[i] = wv * static_cast<T>(xs[i] * inv);
      }
    }
  }
}

template <typename T, int N_READS>
[[kernel]] void rms_single_row(
    const device T* x          [[buffer(0)]],
    const device T* w          [[buffer(1)]],
    device T* out              [[buffer(2)]],
    const constant float& eps  [[buffer(3)]],
    const constant uint& axis_size [[buffer(4)]],
    const constant uint& w_stride  [[buffer(5)]],
    const constant uint& plus_one  [[buffer(6)]],
    const constant float& gain     [[buffer(7)]],
    uint gid                   [[threadgroup_position_in_grid]],
    uint lid                   [[thread_position_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_group            [[simdgroup_index_in_threadgroup]],
    uint tg_size               [[threads_per_threadgroup]]) {
  threadgroup float inv_rms[1], partials[32];
  rms_row_body<T, N_READS>(
      x, w, out, eps, axis_size, w_stride, plus_one, gain,
      size_t(gid) * axis_size,
      inv_rms, partials, lid, simd_lane, simd_group, tg_size);
}

// Prefill variant: the prompt's scratch rows are a uniform `row_pitch` elements
// apart (the widest tensor in the layout), not `axis_size`, so a whole prompt can
// run as one dispatch instead of one per token.  Arithmetic is byte-identical to
// `rms_single_row` -- only the row's base address is computed differently.
template <typename T, int N_READS>
[[kernel]] void rms_strided_row(
    const device T* x          [[buffer(0)]],
    const device T* w          [[buffer(1)]],
    device T* out              [[buffer(2)]],
    const constant float& eps  [[buffer(3)]],
    const constant uint& axis_size [[buffer(4)]],
    const constant uint& w_stride  [[buffer(5)]],
    const constant uint& plus_one  [[buffer(6)]],
    const constant float& gain     [[buffer(7)]],
    const constant int& row_pitch  [[buffer(8)]],
    uint gid                   [[threadgroup_position_in_grid]],
    uint lid                   [[thread_position_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_group            [[simdgroup_index_in_threadgroup]],
    uint tg_size               [[threads_per_threadgroup]]) {
  threadgroup float inv_rms[1], partials[32];
  rms_row_body<T, N_READS>(
      x, w, out, eps, axis_size, w_stride, plus_one, gain,
      size_t(gid) * size_t(row_pitch),
      inv_rms, partials, lid, simd_lane, simd_group, tg_size);
}

#define instantiate_rms_strided_row(name, itype, n_reads)              \
  template [[host_name("rms_strided_row_" #name)]] [[kernel]] void      \
  rms_strided_row<itype, n_reads>(                                      \
      const device itype*, const device itype*, device itype*,          \
      const constant float&, const constant uint&, const constant uint&, \
      const constant uint&, const constant float&, const constant int&,  \
      uint, uint, uint, uint, uint);

instantiate_rms_strided_row(bfloat16, bfloat, 4)

// The per-HEAD norms (q/k) over a whole prompt.  `rms_strided_row` cannot take
// these: a token holds `n_rows` of them packed `axis_size` apart, and the next
// token is a uniform `row_pitch` away, so the row base is two-level and the
// single grid dimension it walks cannot carry both terms.
//
// Rather than pass `n_rows`, the launch carries it: grid is (axis_threads,
// n_rows, N) so the threadgroup's own position IS the (head, token) pair. That
// keeps the argument table identical to `rms_strided_row`'s -- same buffer 8,
// same pitch value the other prefill kernels already bind.
template <typename T, int N_READS>
[[kernel]] void rms_strided_head_row(
    const device T* x          [[buffer(0)]],
    const device T* w          [[buffer(1)]],
    device T* out              [[buffer(2)]],
    const constant float& eps  [[buffer(3)]],
    const constant uint& axis_size [[buffer(4)]],
    const constant uint& w_stride  [[buffer(5)]],
    const constant uint& plus_one  [[buffer(6)]],
    const constant float& gain     [[buffer(7)]],
    const constant int& row_pitch  [[buffer(8)]],
    // All three position attributes are uint3: Metal rejects a signature that
    // mixes scalar and vector ones, and this kernel needs the threadgroup's
    // full 3-D position to carry the (head, token) pair.
    uint3 gid                  [[threadgroup_position_in_grid]],
    uint3 lid                  [[thread_position_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_group            [[simdgroup_index_in_threadgroup]],
    uint3 tg_size              [[threads_per_threadgroup]]) {
  threadgroup float inv_rms[1], partials[32];
  rms_row_body<T, N_READS>(
      x, w, out, eps, axis_size, w_stride, plus_one, gain,
      size_t(gid.z) * size_t(row_pitch) + size_t(gid.y) * axis_size,
      inv_rms, partials, lid.x, simd_lane, simd_group, tg_size.x);
}

#define instantiate_rms_strided_head_row(name, itype, n_reads)         \
  template [[host_name("rms_strided_head_row_" #name)]] [[kernel]] void \
  rms_strided_head_row<itype, n_reads>(                                 \
      const device itype*, const device itype*, device itype*,          \
      const constant float&, const constant uint&, const constant uint&, \
      const constant uint&, const constant float&, const constant int&,  \
      uint3, uint3, uint, uint, uint3);

instantiate_rms_strided_head_row(bfloat16, bfloat, 4)

#define instantiate_rms_single_row(name, itype, n_reads)               \
  template [[host_name("rms_single_row_" #name)]] [[kernel]] void       \
  rms_single_row<itype, n_reads>(                                       \
      const device itype*, const device itype*, device itype*,          \
      const constant float&, const constant uint&, const constant uint&, \
      const constant uint&, const constant float&,                       \
      uint, uint, uint, uint, uint);

instantiate_rms_single_row(bfloat16, bfloat, 4)

// ── Fused norm + residual (+ optional layer scalar) — gemma4 ─────────────────
//
// gemma4 wraps each sublayer in a norm sandwich: the BLOCK's output is
// normalised before it rejoins the residual stream. So `rms_single_row` is
// immediately followed by `residual_add` three times a layer, and the per-layer
// embedding path adds a learned scalar after its own add. Each of those is a
// separate dispatch, and a dispatch that is barrier-separated costs ~5.8 us on
// this machine against a step of 8.46 ms -- measured, by running the same DAG
// with 723, 833 and 1 barriers.
//
// The pair fuses for free: the threadgroup that computes the row's inverse RMS
// already holds every element of the row, so adding the residual in its
// write-back costs one extra load and no synchronisation at all.
//
//   out = rms_norm(x) * w + r            (`rms_residual`)
//   out = (rms_norm(x) * w + r) * s[0]   (`rms_residual_scaled`)
//
// Arithmetic is otherwise identical to `rms_single_row` followed by
// `residual_add`: the same float accumulate, the same `precise::rsqrt`, and the
// same single bf16 round on the way out. It is NOT bit-identical to the
// two-dispatch form, which rounds the norm to bf16 before the add reads it back;
// this keeps that intermediate in float, so it is strictly closer to the
// reference. The parity walk is what says so.
template <typename T, int N_READS, bool SCALED>
METAL_FUNC void rms_residual_impl(
    const device T* x,
    const device T* w,
    const device T* r,
    const device T* s,
    device T* out,
    float eps, uint axis_size, uint w_stride, uint plus_one, float gain,
    threadgroup float* local_inv_mean,
    threadgroup float* local_sums,
    uint gid,
    uint lid,
    uint simd_lane_id,
    uint simd_group_id,
    uint tg_size) {
  const uint span = tg_size * uint(N_READS);

  const size_t row = size_t(gid) * size_t(axis_size);
  const device T* xr = x + row;
  float acc = 0.0f;
  for (uint start = lid * uint(N_READS); start < axis_size; start += span) {
    acc += rms_lane_square_sum_at<T, N_READS>(xr + start, axis_size, start);
  }
  const float inv = rms_inv_from_lane_sum(
      acc, axis_size, eps, local_inv_mean, local_sums,
      simd_lane_id, simd_group_id);

  const float scale = SCALED ? float(s[0]) : 1.0f;
  const device T* rr = r + row;
  device T* outr = out + row;
  for (uint start = lid * uint(N_READS); start < axis_size; start += span) {
    for (int i = 0; i < N_READS; i++) {
      if (start + uint(i) < axis_size) {
        const float wv = gain * (plus_one ? (1.0f + float(w[w_stride * (start + uint(i))]))
                                          : float(w[w_stride * (start + uint(i))]));
        const float normed = wv * (float(xr[start + uint(i)]) * inv);
        outr[start + uint(i)] = static_cast<T>((normed + float(rr[start + uint(i)])) * scale);
      }
    }
  }
}

template <typename T, int N_READS>
[[kernel]] void rms_residual(
    const device T* x          [[buffer(0)]],
    const device T* w          [[buffer(1)]],
    device T* out              [[buffer(2)]],
    const constant float& eps  [[buffer(3)]],
    const constant uint& axis_size [[buffer(4)]],
    const constant uint& w_stride  [[buffer(5)]],
    const constant uint& plus_one  [[buffer(6)]],
    const constant float& gain     [[buffer(7)]],
    const device T* r          [[buffer(8)]],
    uint gid                   [[threadgroup_position_in_grid]],
    uint lid                   [[thread_position_in_threadgroup]],
    uint simd_lane_id          [[thread_index_in_simdgroup]],
    uint simd_group_id         [[simdgroup_index_in_threadgroup]],
    uint tg_size               [[threads_per_threadgroup]]) {
  threadgroup float local_inv_mean[1];
  threadgroup float local_sums[32];
  rms_residual_impl<T, N_READS, false>(x, w, r, nullptr, out,
                                       eps, axis_size, w_stride, plus_one, gain,
                                       local_inv_mean, local_sums,
                                       gid, lid, simd_lane_id, simd_group_id, tg_size);
}

template <typename T, int N_READS>
[[kernel]] void rms_residual_scaled(
    const device T* x          [[buffer(0)]],
    const device T* w          [[buffer(1)]],
    device T* out              [[buffer(2)]],
    const constant float& eps  [[buffer(3)]],
    const constant uint& axis_size [[buffer(4)]],
    const constant uint& w_stride  [[buffer(5)]],
    const constant uint& plus_one  [[buffer(6)]],
    const constant float& gain     [[buffer(7)]],
    const device T* r          [[buffer(8)]],
    const device T* s          [[buffer(9)]],
    uint gid                   [[threadgroup_position_in_grid]],
    uint lid                   [[thread_position_in_threadgroup]],
    uint simd_lane_id          [[thread_index_in_simdgroup]],
    uint simd_group_id         [[simdgroup_index_in_threadgroup]],
    uint tg_size               [[threads_per_threadgroup]]) {
  threadgroup float local_inv_mean[1];
  threadgroup float local_sums[32];
  rms_residual_impl<T, N_READS, true>(x, w, r, s, out,
                                      eps, axis_size, w_stride, plus_one, gain,
                                      local_inv_mean, local_sums,
                                      gid, lid, simd_lane_id, simd_group_id, tg_size);
}

#define instantiate_rms_residual(name, itype, nreads)                    \
  template [[host_name("rms_residual_" #name)]]                          \
  [[kernel]] void rms_residual<itype, nreads>(                           \
      const device itype*, const device itype*, device itype*,           \
      const constant float&, const constant uint&, const constant uint&,  \
      const constant uint&, const constant float&, const device itype*,   \
      uint, uint, uint, uint, uint);                                           \
  template [[host_name("rms_residual_scaled_" #name)]]                   \
  [[kernel]] void rms_residual_scaled<itype, nreads>(                    \
      const device itype*, const device itype*, device itype*,           \
      const constant float&, const constant uint&, const constant uint&,  \
      const constant uint&, const constant float&, const device itype*,   \
      const device itype*, uint, uint, uint, uint, uint);

instantiate_rms_residual(bfloat16, bfloat, 4)

// Raw-Metal device argmax + EOS-compare (optional I3 substrate; sampling lane).
//
//   next_token[r] = argmax_i logits[r, i]      (lowest-index tie-break)
//   eos_flag[r]   = (next_token[r] ∈ eos_ids)  ? 1 : 0
//
// LOAD-BEARING for two paths (alpha's GPU-resident loop + the M>1 throughput path):
//   (a) Resident loop: argmax→NextToken bound DIRECTLY as the next step's TokenId
//       (zero host readback) is the ENABLER of GPU-residency — host argmax is what
//       FORCES the per-token drain. EOS-on-GPU cuts the loop without a host roundtrip.
//   (b) Throughput M>1: one threadgroup per request row (grid.y = r) samples all N
//       rows in one dispatch — host argmax would scale N× over the [N,vocab] logits.
//
// BIT-EXACT to host `RawMetalDecoder::argmax()` (decoder.cpp): lm_head writes bf16,
// host widens bf16→f32 and scans ascending with strict `>` (keeps the FIRST/lowest
// index of the max). This kernel reproduces that exactly — `float(bfloat)` is the
// same exact widening, and the reduction keeps the lower index on every value tie.
//
// bind::Argmax (proposed extension, pending alpha's ordinal assignment):
//   Logits=0 (in, [n_rows,vocab] row-major), NextToken=1 (out u32[n_rows]),
//   Params=2 (ArgmaxParams: vocab + inline eos id list), EosFlag=3 (out u32[n_rows]).
// The locked 2-buffer bind (Logits=0/NextToken=1) is unchanged; Params/EosFlag are
// ADD-ONLY (inert until the resident-loop / M>1 wiring binds them).
//
// Dispatch: Grid{1024, n_rows, 1}, Threadgroup{1024,1,1} (M=1 → grid.y=1 → row 0,
// byte-identical single-row reduction). One threadgroup (32 simdgroups) per row.

#include <metal_stdlib>
using namespace metal;

struct ArgmaxParams {
  uint vocab;        // logits row width
  uint n_eos;        // number of valid stop-token ids in eos_ids[] (0 ⇒ eos_flag always 0)
  uint eos_ids[8];   // stop-token ids; next_token compared against these → eos_flag
};

template <typename T>
[[kernel]] void argmax_logits(
    const device T* logits      [[buffer(0)]],
    device uint* next_token     [[buffer(1)]],
    constant ArgmaxParams& p    [[buffer(2)]],
    device uint* eos_flag       [[buffer(3)]],
    uint3 tg_pos                [[threadgroup_position_in_grid]],
    uint3 lid3                  [[thread_position_in_threadgroup]],
    uint3 tg_size3              [[threads_per_threadgroup]],
    uint simd_lane_id           [[thread_index_in_simdgroup]],
    uint simd_group_id          [[simdgroup_index_in_threadgroup]]) {
  constexpr uint SIMD_SIZE = 32;
  const uint row = tg_pos.y;                       // request row (M=1 → 0)
  const uint lid = lid3.x;
  const uint tg_size = tg_size3.x;
  const uint vocab = p.vocab;
  const device T* row_logits = logits + size_t(row) * vocab;

  // Per-thread local argmax over a strided slice (ascending within a thread ⇒ strict
  // `>` already keeps that thread's lowest-index max).
  float best_v = -INFINITY;
  uint  best_i = 0;
  for (uint i = lid; i < vocab; i += tg_size) {
    float v = float(row_logits[i]);
    if (v > best_v) { best_v = v; best_i = i; }
  }

  // Intra-simdgroup reduction: higher value wins; on a value tie the LOWER index wins
  // (matches the host's first-occurrence semantics).
  for (uint off = SIMD_SIZE / 2; off > 0; off >>= 1) {
    float ov = simd_shuffle_down(best_v, off);
    uint  oi = simd_shuffle_down(best_i, off);
    if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
  }

  threadgroup float tg_v[SIMD_SIZE];
  threadgroup uint  tg_i[SIMD_SIZE];
  if (simd_lane_id == 0) { tg_v[simd_group_id] = best_v; tg_i[simd_group_id] = best_i; }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Final reduction across simdgroups (done by simdgroup 0).
  if (simd_group_id == 0) {
    const uint n_simd = (tg_size + SIMD_SIZE - 1) / SIMD_SIZE;
    best_v = simd_lane_id < n_simd ? tg_v[simd_lane_id] : -INFINITY;
    best_i = simd_lane_id < n_simd ? tg_i[simd_lane_id] : 0u;
    for (uint off = SIMD_SIZE / 2; off > 0; off >>= 1) {
      float ov = simd_shuffle_down(best_v, off);
      uint  oi = simd_shuffle_down(best_i, off);
      if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
    }
    if (simd_lane_id == 0) {
      next_token[row] = best_i;
      uint flag = 0;
      for (uint e = 0; e < p.n_eos; ++e) { if (best_i == p.eos_ids[e]) { flag = 1; break; } }
      eos_flag[row] = flag;
    }
  }
}

#define instantiate_argmax(name, itype)                                    \
  template [[host_name("argmax_logits_" #name)]]                           \
  [[kernel]] void argmax_logits<itype>(                                    \
      const device itype*, device uint*, constant ArgmaxParams&,           \
      device uint*, uint3, uint3, uint3, uint, uint);

instantiate_argmax(float32, float)
instantiate_argmax(float16, half)
instantiate_argmax(bfloat16, bfloat)

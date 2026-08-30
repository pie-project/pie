#include <metal_stdlib>

using namespace metal;

// The correction class's device half (palo design §8, decision 17).
//
// One statement: `y[row] += B[a] · (A[a] · x[row])`, where `a = routes[row]`
// is the adapter the row's lane registered and `-1` is the base model. The
// LoRA scale is not here and is not an argument: `α/r` is folded into the up
// bank's contents when the adapter is registered, which is where every
// per-adapter number belongs.
//
// ONE LAUNCH, AND THE WAIST STAYS IN THREADGROUP MEMORY. The CUDA plane
// spells this as two launches over a `rows x rank` scratch slab, because its
// projection half IS `moe`'s routed GEMV and a second copy of that ladder
// would drift. This plane has no scratch to hand a kernel entry — an
// `Encode` sink binds the handles an op names and nothing else — so the rank-
// wide waist lives where its lifetime actually is: `rank` floats in
// threadgroup memory between the two halves of one threadgroup's work. That
// is what the CUDA file's own note names as the form a rank-diverse
// deployment should measure next, arrived at here by the plane's own
// arithmetic rather than by preference.
//
// One threadgroup per token row, and the row is never split across
// threadgroups: a second block over the output columns would have to
// recompute the whole projection half to have a waist to read, and the
// correction rides on a trunk that already paid `O(h·i)` — `O(r·(h+n))` on
// one threadgroup is the whole budget design §8 measures at 1.01×.
//
// A row whose route is negative returns before it reads anything, and it
// returns UNIFORMLY: every thread of the threadgroup reads the same
// `routes[row]`, so the barrier below is never reached by a divided group.
// That is what an adapterless row inside an adapter window costs — one
// predicated load and a branch — and a fire with no adapter lane at all costs
// less, because the walk skips the zero-row region and this kernel never
// launches.
//
// `waist` is read by every thread of the second half at the same `rank`
// addresses; `rank` is the BANK's declared capacity, and an adapter
// registered shorter than that was zero-padded at registration, which
// contributes exactly zero to the accumulator.
//
// The accumulate is the whole of the correction class. Every other routed op
// on this plane assigns its output row, because a routed expert owns the row
// it computes; a correction does not own `y` — it rides on a value the trunk
// already materialised — so this reads, adds, and writes back.

constant constexpr uint kLoraMaxRank = 128;

[[kernel]] void lora_correct(
    const device bfloat* x      [[buffer(0)]],
    const device bfloat* bank_a [[buffer(1)]],
    const device bfloat* bank_b [[buffer(2)]],
    const device int* routes    [[buffer(3)]],
    device bfloat* y            [[buffer(4)]],
    const constant uint& in_width  [[buffer(5)]],
    const constant uint& out_width [[buffer(6)]],
    const constant uint& rank      [[buffer(7)]],
    uint3 lid3    [[thread_position_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint3 tgsize  [[threads_per_threadgroup]],
    uint3 tgid    [[threadgroup_position_in_grid]]) {
  const uint row = tgid.y;
  const int adapter = routes[row];
  if (adapter < 0) return;

  const uint r = min(rank, kLoraMaxRank);
  const device bfloat* down =
      bank_a + size_t(uint(adapter)) * size_t(r) * size_t(in_width);
  const device bfloat* up =
      bank_b + size_t(uint(adapter)) * size_t(out_width) * size_t(r);
  const device bfloat* a = x + size_t(row) * size_t(in_width);
  device bfloat* out = y + size_t(row) * size_t(out_width);

  threadgroup float waist[kLoraMaxRank];

  // ── half one: the projection, one simdgroup per rank row ─────────────
  const uint n_simd = max((tgsize.x + 31u) / 32u, 1u);
  for (uint i = simd_gid; i < r; i += n_simd) {
    const device bfloat* w = down + size_t(i) * size_t(in_width);
    float acc = 0.0f;
    for (uint c = simd_lid; c < in_width; c += 32u) {
      acc += float(w[c]) * float(a[c]);
    }
    acc = simd_sum(acc);
    if (simd_lid == 0) waist[i] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ── half two: the accumulate, the up bank read out-major ─────────────
  for (uint n = lid3.x; n < out_width; n += tgsize.x) {
    const device bfloat* brow = up + size_t(n) * size_t(r);
    float acc = 0.0f;
    for (uint i = 0; i < r; ++i) {
      acc += float(brow[i]) * waist[i];
    }
    out[n] = static_cast<bfloat>(float(out[n]) + acc);
  }
}

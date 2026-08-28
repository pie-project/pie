#pragma once

#include "prelude/device.cuh"

// The correction class's device half (palo design §8, decision 17).
//
// One statement: `y[row] += B[a] · (A[a] · x[row])`, where `a = routes[row]`
// is the adapter the row's lane registered and `-1` is the base model.
//
// TWO LAUNCHES, AND THE FIRST ONE IS SOMEBODY ELSE'S. The projection half —
// `t[row] = A[a] · x[row]` — is a routed matmul-select at fan-out one, which
// is exactly `pie::linear::moe_matmul_select_gemv_by_route`: one warp per
// output column, the bank's first axis indexed by the route, a zero row for a
// negative route. It is fired verbatim, under the correction's own op name, so
// this file carries only what MoE has no shape for: the ACCUMULATE.
//
// `moe`'s selects assign (`out[..] = acc`) because a routed expert owns its
// result row. A correction does not own `y` — it rides on a value the trunk
// already materialised — so the combine reads, adds, and writes back, and that
// one difference is the whole of the correction class.

namespace pie::linear {

[[maybe_unused]] constexpr int kLoraBlock = 256;

// `y[row, n] += Σ_r B[a][n, r] · t[row, r]`, one block row per token row.
//
// A row whose route is negative returns before it reads anything: an
// adapterless row inside an adapter window (a lane that stated an id the
// caller then cleared) costs one predicated load and a branch, which is the
// cheapest honest answer there is. A fire with no adapter lane at all costs
// less than that — `driver::fire::walk` skips the zero-row region and this
// kernel is never launched.
//
// `t` is read by every thread of the block at the same `rank` addresses, so it
// stays in L1 and no shared-memory staging is written for it; `rank` is the
// bank's declared capacity, and an adapter registered shorter than that was
// zero-padded at registration, which contributes exactly zero to `acc`.
//
// ── `segments`: `Fallback::Grouped`, and why it is NULLABLE ──────────────
//
// P4 lays the fire's rows out in one class order and cannot always make a
// windowed consumer's classes an interval of it. When it cannot, the
// correction's rows are SEVERAL intervals of the rectangle this launch was
// handed, with rows belonging to other classes standing in the gaps. The
// answer this file serves is design §3's `Fallback::Grouped`: instead of `r`
// launches over `r` sub-rectangles, ONE launch over the union, told where the
// intervals are.
//
// `segments` is `[segs][2]` — `(first row of this launch's rectangle, how many
// rows)` — and `nullptr` means what it has always meant: the rows are
// `[0, gridDim.y)` and the caller is a window P4 seated. That null arm is not
// a compatibility shim, it is the common case, and it must stay byte-identical
// because it is the oracle the grouped arm is checked against.
//
// **THE GRID IS MAX-GRID PLUS EARLY EXIT** (decision #15). `z` is the segment
// and its extent is the artifact's load-time bound on the segment count
// (`driver::fire::max_runs`), not this fire's; `y` is the row within the
// segment and its extent is the longest segment. Both overshoot and both
// return before reading anything, which is a handful of empty blocks against
// a binary search per block or a host-side grid that moves per fire.
//
// **THE GAPS ARE NEVER TOUCHED, AND THAT IS THE WHOLE SAFETY ARGUMENT.** This
// kernel writes exactly the rows the segments name, so the foreign rows inside
// the union are read by nobody and written by nobody. A kernel that wrote
// densely over the extent it was handed could not take this treatment — which
// is exactly why `Attention::PrefillLse`'s split-kv fold cannot, and why the
// profile names ops rather than backends.
template <class T>
__global__ void lora_combine(
    const i32* __restrict__ routes,
    const T* __restrict__ t,
    const T* __restrict__ bank_b,
    T* __restrict__ y,
    const i32* __restrict__ segments,
    int segs,
    int rank, int out_width, long long adapter_stride)
{
    int row = blockIdx.y;
    if (segments != nullptr) {
        const int seg = blockIdx.z;
        if (seg >= segs) return;
        if ((int)blockIdx.y >= segments[2 * seg + 1]) return;
        row = segments[2 * seg] + (int)blockIdx.y;
    }
    const int adapter = routes[row];
    if (adapter < 0) return;

    const T* b = bank_b + (long long)adapter * adapter_stride;
    const T* tv = t + (long long)row * rank;
    T* out = y + (long long)row * out_width;

    for (int n = blockIdx.x * blockDim.x + threadIdx.x; n < out_width;
         n += gridDim.x * blockDim.x) {
        const T* brow = b + (long long)n * rank;
        float acc = 0.f;
        for (int r = 0; r < rank; ++r) {
            acc += Elem<T>::to_f32(brow[r]) * Elem<T>::to_f32(tv[r]);
        }
        out[n] = Elem<T>::from_f32(Elem<T>::to_f32(out[n]) + acc);
    }
}

}  // namespace pie::linear

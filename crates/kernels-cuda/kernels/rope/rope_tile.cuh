//===-- rope_tile.cuh - the rotation in CuTile: faster AND more accurate --===//
//
// An ALTERNATIVE to `rope.cuh`'s `rotate_partial`. It is 1.2-1.4x faster and
// its outputs are CLOSER to an fp64 reference, which is an unusual pair and
// is the whole reason this file has a long header.
//
//     tokens     tile        rotate_partial    ratio
//          1     5.27 us        6.65 us        1.26x
//        128     5.59           6.85           1.23x
//      1,024    12.98          18.37           1.42x
//      8,192   108.15         127.11           1.18x
//
// L40S sm_89, 32 Q heads + 8 KV heads, head_dim 128, rotary_dim 128, bf16.
//
// # The outputs differ, and the incumbent is the one that is wrong
//
// 400,887 of 33,554,432 elements differ at 8,192 tokens -- 1.2% -- which is
// far too many for a rounding difference and was NOT assumed to be either
// kernel's fault. The bench recomputes the worst mismatch in fp64 on the
// host:
//
//     worst at tok=474 head=23 d=0 pos=3867 angle=3867.000
//     fp64 reference  0.000345
//     tile            0.000345   err 1.52e-07
//     scalar         -0.000117   err 4.62e-04
//
// and at 128 tokens, angle 3156: tile err 1.01e-05, scalar err 1.69e-04.
//
// The cause is `__sincosf`. The scalar form uses it, and it is the FAST
// intrinsic -- accurate for small arguments and degrading through argument
// reduction as the angle grows. A rope angle is `pos * theta^(-2d/D)`, so at
// position 3,867 and `d = 0` the angle IS 3,867 radians. `ct::sin`/`ct::cos`
// do not make that trade.
//
// So the tile kernel is not "close enough" to the incumbent; it is closer to
// the truth, by three orders of magnitude at the worst point. **That is a
// behaviour change and it needs a decision**, not a merge: the scalar form's
// choice may be deliberate, and rope error feeds attention scores. The
// predicate below is therefore the only one in this tree bounded on
// something other than speed.
//
// The relative figures look alarming -- 3.94 at the worst -- because the true
// values are near zero (3.4e-4) and cancellation amplifies. The ABSOLUTE
// errors above are the ones to read.
//
// # What it settles about the census
//
// `kernels/tile/alternatives.cuh` measures the ELEM bucket with
// `mlp/swiglu_tile.cuh`, which is CONTIGUOUS elementwise. Rope is PERMUTED
// elementwise -- element `d` pairs with `d + half`, two gathers a stride
// apart -- and the COPY bucket showed that a permuted access with no
// arithmetic is a wash. This is the case with both, and it lands with ELEM:
// **the arithmetic decides, not the access pattern.**
//
// # Two shapes the tile API forced
//
// The natural lane count is `(NQ + NKV) * head_dim/2` = 2,560 for a 32+8
// model, and a tile extent must be a power of two -- `tile<int, shape<2560>>`
// is `concept is false`. So the pairs are walked in `BS`-wide chunks, which
// is the same stride loop the scalar form runs with `threadIdx.x`.
//
// And the angle is computed PER LANE rather than cached. `rope.cuh` hoists
// sin/cos into shared memory because 65 heads share 32 transcendentals and
// recomputing them "made this kernel cost more than the attention it feeds".
// A tile lane already holds its own angle in a register, so the cache has
// nothing to cache.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cuda_bf16.h>
#include <crt/cuda_tile.h>

namespace pie::rope {

namespace ct = ::cuda::tiles;

/// Whether to prefer this ALTERNATIVE to `rotate_partial`.
///
/// **Not on speed, which it wins at every size measured.** It changes the
/// OUTPUT: `ct::sin`/`ct::cos` against the scalar form's `__sincosf`, which
/// is three orders of magnitude closer to fp64 at large rope angles and
/// therefore a different model unless someone decides otherwise. Rope error
/// feeds attention scores.
///
/// So this answers false until that decision is made, and it is the only
/// predicate in this tree bounded on something other than speed.
constexpr bool rope_partial_tile_preferred(int /*tokens*/)
{
    return false;
}

#ifndef HD
#define HD 128            // head_dim
#endif
#ifndef RD
#define RD 128            // rotary_dim
#endif
#ifndef NQ
#define NQ 32             // q heads
#endif
#ifndef NKV
#define NKV 8             // kv heads
#endif
constexpr int kHalf = HD / 2;
constexpr int kAngles = RD / 2;
// The tile shape must be a power of two, and (NQ+NKV)*half is 2,560 for a
// 32+8-head model. So the pairs are walked in BS-wide chunks -- the same
// stride loop the scalar form runs, with the lane index doing the work
// `threadIdx.x + i * blockDim.x` did.
#ifndef BS
#define BS 512
#endif
constexpr int kPairs = (NQ + NKV) * kHalf;
using i1d = ct::tile<int, ct::shape<BS>>;
using f1d = ct::tile<float, ct::shape<BS>>;
using b1d = ct::tile<__nv_bfloat16, ct::shape<BS>>;

/// One block per token; one LANE per (head, dim pair), which is the scalar
/// form's `t` exactly. The pair `(d, d + half)` is a permuted access -- two
/// gathers a stride apart -- and the rotation is real arithmetic, so this
/// kernel is the test of which of those decides an elementwise result.
__tile_global__ void rope_partial_tile(
    __nv_bfloat16* __restrict__ _q,
    __nv_bfloat16* __restrict__ _k,
    const int* __restrict__ positions,
    int position_delta,
    int /*nq*/, int /*nkv*/, int /*hd*/, int /*rd*/, float theta)
{
    const int n = static_cast<int>(ct::bid().x);
    const int pos = positions[n] + position_delta;
    auto q = ct::assume_aligned<16>(_q + (long long)n * NQ * HD);
    auto k = ct::assume_aligned<16>(_k + (long long)n * NKV * HD);

    constexpr int nb = (kPairs + BS - 1) / BS;
    for (auto ci : ct::irange(0, nb)) {
    auto t = ct::iota<i1d>() + ci * BS;
    auto head = t / ct::full<i1d>(kHalf);
    auto dp = t - head * ct::full<i1d>(kHalf);
    auto live = (dp < ct::full<i1d>(kAngles)) && (t < ct::full<i1d>(kPairs));

    // The angle, computed per lane rather than cached in shared memory --
    // the scalar form hoists it because 65 heads share 32 transcendentals;
    // a tile lane already has it in a register.
    auto fdp = ct::element_cast<float>(dp);
    auto freq = ct::pow(ct::full<f1d>(theta),
                        fdp * ct::full<f1d>(-2.f / (float)HD));
    auto ang = ct::full<f1d>((float)pos) * freq;
    auto sn = ct::sin(ang);
    auto cs = ct::cos(ang);

    auto isq = head < ct::full<i1d>(NQ);
    auto kvh = head - ct::full<i1d>(NQ);
    // Byte offset of the pair's first element, in whichever buffer.
    auto qoff = head * ct::full<i1d>(HD) + dp;
    auto koff = kvh * ct::full<i1d>(HD) + dp;

    b1d qa, qb, ka, kb;
    [[ using cutile : hint(1000, latency=1) ]]
    qa = ct::load_masked(q + qoff, isq && live, ct::zeros<b1d>());
    [[ using cutile : hint(1000, latency=1) ]]
    qb = ct::load_masked(q + qoff + ct::full<i1d>(kHalf), isq && live, ct::zeros<b1d>());
    [[ using cutile : hint(1000, latency=1) ]]
    ka = ct::load_masked(k + koff, !isq && live, ct::zeros<b1d>());
    [[ using cutile : hint(1000, latency=1) ]]
    kb = ct::load_masked(k + koff + ct::full<i1d>(kHalf), !isq && live, ct::zeros<b1d>());

    auto qav = ct::element_cast<float>(qa), qbv = ct::element_cast<float>(qb);
    auto kav = ct::element_cast<float>(ka), kbv = ct::element_cast<float>(kb);
    ct::store_masked(q + qoff,
        ct::element_cast<__nv_bfloat16>(qav * cs - qbv * sn), isq && live);
    ct::store_masked(q + qoff + ct::full<i1d>(kHalf),
        ct::element_cast<__nv_bfloat16>(qbv * cs + qav * sn), isq && live);
    ct::store_masked(k + koff,
        ct::element_cast<__nv_bfloat16>(kav * cs - kbv * sn), !isq && live);
    ct::store_masked(k + koff + ct::full<i1d>(kHalf),
        ct::element_cast<__nv_bfloat16>(kbv * cs + kav * sn), !isq && live);
    }
}

}  // namespace pie::rope

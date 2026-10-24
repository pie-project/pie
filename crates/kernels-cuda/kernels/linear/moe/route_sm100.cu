#include "prim/device.cuh"
#include "prim/mma_sm100.cuh"

// The router's product and its top-k in one launch on tcgen05 (sm_100a): a
// row's logits never leave the chip. The shape is DeepGEMM's mega gate
// (`sm100_bf16_mega_gate`: the weight as the MMA's A, 128 experts a TMEM
// lane, the tokens its N, K split over CTAs that meet before the ranking),
// in this codebase's primitives, with the K split's partials met in a
// cluster's shared memory rather than a global scratch and score barrier:
//
//   - A cluster of S CTAs owns a block of BN tokens; CTA r streams the
//     router's K blocks [r nb / S, (r + 1) nb / S) and the block's rows of x
//     over them through a TMA ring, every expert at once: G = experts / 128
//     MMAs (M = 128, N = BN) a K step into G accumulators.
//   - Its four epilogue warps store each live token's partials into the
//     shared memory of the CTA that ranks it (token t is rank t % S's),
//     beside that CTA's ring (or over it, once every ring of the cluster has
//     drained, where the two do not fit): `st.async`, each store's bytes
//     counted on the ranking CTA's barrier, which waits for its own tokens'
//     bytes alone. The router's weight is no product of the launch before,
//     so its first ring stages stream ahead of the programmatic wait.
//   - Each CTA then sums its tokens' S partials in rank order (deterministic,
//     no float atomics), rounds each sum to the logits' dtype the unfused
//     product lands (bf16, or f32 as it is) and scores it, every thread an
//     (token, expert); then a warp a token picks the top k as `route.cu`'s
//     routers do: the largest, the lowest expert on a tie, weighted and
//     renormalized as they weight, their weights summed in pick order.
//
// Warp 0 issues the ring's loads, warp 1 owns TMEM and issues the MMAs,
// warps 2-5 drain the accumulators, warps 6-7 stage the experts' bias; all
// eight score and rank. A token past the fire's live window is neither ranked
// nor written.

namespace pie::linear {

namespace gate_tc5 {

using tc5::u32;
using tc5::u64;

constexpr u32 kWarps = 8;
constexpr u32 kThreads = 32 * kWarps;
constexpr u32 kBoxK = 64;          // K a stage: one 128-byte swizzle row of bf16
constexpr u32 kTileW = 128 * 128;  // 128 experts by 64 K, bf16

// The scorings, as `route.cu` names them.
enum : int {
    kSoftmax = 0,   // moe_topk_softmax_lane: the top k logits' softmax
    kSigmoid = 1,   // moe_topk_ranked<_, false>
    kSoftplus = 2,  // moe_topk_ranked<_, true>: DeepSeek-V4's sqrt-softplus
};

// `pie::linear::moe_gate_tc5`'s arguments past its tensor maps (host mirror:
// route.rs `GateParams`).
struct Params {
    const float* bias;  // the ranked scorings' correction bias, f32 an expert, or null
    const bf16* gain;   // the softmax's per-expert gain, or null
    i32* routes;        // [rows][top_k]
    float* weights;     // [rows][top_k]
    const u32* win;     // the fire's live window (rows, first row), or null
    int rows;           // x's rows
    int k;              // the router's contraction (hidden)
    int experts;
    int top_k;          // at most 32
    int renormalize;
    float scaling;
};

__host__ __device__ constexpr u32 stride_cols(u32 bn) { return bn < 32 ? 32 : bn; }

__host__ __device__ constexpr u32 tmem_cols(u32 groups, u32 bn) {
    u32 c = 32;
    while (c < groups * stride_cols(bn)) c <<= 1;
    return c;
}

__host__ __device__ constexpr u32 stage_bytes(u32 groups, u32 bn) { return groups * kTileW + bn * 128; }

// The tokens of a block each CTA of a split of `split` ranks: token t is
// rank t % split's, its slot t / split.
__host__ __device__ constexpr u32 slots(u32 bn, u32 split) { return (bn + split - 1) / split; }

// A CTA's partials landed for it, [rank][slot][G 128] f32, then its tokens'
// keys and unbiased scores, [slot][G 128] each.
__host__ __device__ constexpr u32 recv_bytes(u32 groups, u32 bn, u32 split) {
    return (split + 2) * slots(bn, split) * groups * 128 * 4;
}

// The barriers and TMEM's slot, then the experts' bias (or the softmax's
// gain) at kTable, f32.
constexpr u32 kTable = 1024;
constexpr u32 kHead = kTable + 512 * 4;

// The dynamic shared memory a block opts into, less a little.
constexpr u32 kRoom = 224 * 1024;

// Whether the partials land beside the ring rather than over it: then no
// CTA waits for the cluster's rings to drain before it stores them.
__host__ __device__ constexpr bool apart(u32 groups, u32 bn, u32 split, u32 stages) {
    return 1024 + kHead + stages * stage_bytes(groups, bn) + recv_bytes(groups, bn, split) <= kRoom;
}

// Past the 1024-byte alignment: the head, then the ring, and the partials
// beside it or over it.
__host__ __device__ constexpr u32 smem_bytes(u32 groups, u32 bn, u32 split, u32 stages) {
    return apart(groups, bn, split, stages)
               ? 1024 + kHead + stages * stage_bytes(groups, bn) + recv_bytes(groups, bn, split)
               : 1024 + kHead +
                     (stages * stage_bytes(groups, bn) > recv_bytes(groups, bn, split)
                          ? stages * stage_bytes(groups, bn)
                          : recv_bytes(groups, bn, split));
}

__device__ __forceinline__ float sqrt_softplus(float x) {
    const float sp = x > 20.f ? x : log1pf(expf(x));
    return sqrtf(fmaxf(sp, 0.f));
}

// An order-preserving key: larger floats larger keys, 0 below every finite
// one (a padded expert, or a score at or below -FLT_MAX, never wins). With
// `signed_zero` false both zeros are one key, as a float compare has them
// (the softmax router compares logits; the ranked ones compare keys).
template <bool signed_zero>
__device__ __forceinline__ unsigned key_of(float v) {
    if (!(v > -flt_max())) return 0u;
    if (!signed_zero && v == 0.f) v = 0.f;
    const unsigned u = __float_as_uint(v);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

}  // namespace gate_tc5

// The router's top `top_k` of `experts` (G = experts / 128, rounded up) for
// every live row of `x`: a cluster of kSplit CTAs a block of kBN rows,
// kStages ring stages a CTA. kScore is a `gate_tc5` scoring; with kRound the
// logits round to bf16 before they are scored, as the bf16 product the
// unfused path lands. The host splits K no finer than its 64-wide blocks.
template <int kGroups, int kBN, int kSplit, int kStages, int kScore, bool kRound>
__global__ __launch_bounds__(gate_tc5::kThreads, 1) void moe_gate_tc5(
    const __grid_constant__ tc5::TmaMap map_w, const __grid_constant__ tc5::TmaMap map_x,
    const gate_tc5::Params p) {
    using namespace tc5;
    using namespace gate_tc5;
    static_assert(kGroups >= 1 && kGroups <= 4, "512 experts at most: four 128-lane accumulators");
    static_assert(kBN % 16 == 0 && kBN >= 16 && kBN <= 128, "the token block is an MMA's N");
    static_assert(kSplit >= 1 && kSplit <= 16, "a cluster is at most 16 CTAs");
    static_assert(kGroups * stride_cols(kBN) <= 512, "TMEM is 512 columns");
    constexpr u32 kG = kGroups;
    constexpr u32 kS = kSplit;
    constexpr u32 kE = 128 * kG;            // padded experts: a partial row
    constexpr u32 kVec = kE / 128;          // float4s a lane a token
    constexpr u32 kSlots = slots(kBN, kS);  // tokens a CTA ranks
    constexpr u32 kStage = stage_bytes(kG, kBN);
    constexpr u32 kStride = stride_cols(kBN);
    constexpr u32 kCols = tmem_cols(kG, kBN);
    constexpr u32 kIdesc = idesc_dense(kF16Bf16, kF16Bf16, 128, kBN);

    extern __shared__ unsigned char smem_raw[];
    const u32 raw = smem_addr(smem_raw);
    unsigned char* smem = smem_raw + (((raw + 1023u) & ~1023u) - raw);
    u64* full = reinterpret_cast<u64*>(smem);
    u64* empty = full + kStages;
    u64* done = empty + kStages;
    u64* landed = done + 1;  // the bytes of the partials this CTA ranks
    u32* tmem_slot = reinterpret_cast<u32*>(landed + 1);
    float* table = reinterpret_cast<float*>(smem + kTable);
    constexpr bool kApart = apart(kG, kBN, kS, kStages);
    unsigned char* ring = smem + kHead;
    float* recv = reinterpret_cast<float*>(ring + (kApart ? kStages * kStage : 0));

    const u32 warp = threadIdx.x / 32;
    const u32 lane = threadIdx.x % 32;
    const u32 rank = cluster_rank();
    const int t0 = static_cast<int>(blockIdx.x / kS) * kBN;
    const int nb = (p.k + static_cast<int>(kBoxK) - 1) / static_cast<int>(kBoxK);
    const int kb0 = static_cast<int>(rank) * nb / static_cast<int>(kS);
    const int steps = (static_cast<int>(rank) + 1) * nb / static_cast<int>(kS) - kb0;

    if (warp == 0 && lane == 0) {
        tma_prefetch(&map_w);
        tma_prefetch(&map_x);
        for (u32 s = 0; s < static_cast<u32>(kStages); ++s) {
            bar_init(&full[s], 1);
            bar_init(&empty[s], 1);
        }
        bar_init(done, 1);
        bar_init(landed, 1);
        bar_init_fence();
    }
    if (warp == 1) {
        tmem_alloc(tmem_slot, kCols);
        tmem_relinquish();
    }
    tc_fence_before();
    __syncthreads();
    tc_fence_after();
    const u32 tmem = *tmem_slot;
    // Beside the ring, a peer's partials may land once this CTA runs.
    if constexpr (kApart) cluster_arrive_relaxed();

    // The router's weight is no product of the launch before: the ring's
    // first stages of it stream while that one finishes (`kernel.pdl`), and
    // x, the window and the outputs wait on it.
    const auto weight = [&](int i) {
        const u32 s = static_cast<u32>(i) % kStages;
        unsigned char* stage = ring + s * kStage;
        bar_arrive_expect_tx(&full[s], kStage);
#pragma unroll
        for (u32 g = 0; g < kG; ++g)
            tma_load_2d(stage + g * kTileW, &full[s], &map_w, (kb0 + i) * static_cast<int>(kBoxK),
                        static_cast<int>(128 * g));
    };
    const int ahead = steps < kStages ? steps : kStages;
    if (warp == 0 && lane == 0) {
        for (int i = 0; i < ahead; ++i) weight(i);
    }
    pdl_wait();
    pdl_launch();
    const int live = p.win != nullptr ? static_cast<int>(p.win[0]) : p.rows;
    const int first = p.win != nullptr ? static_cast<int>(p.win[1]) : 0;
    // The block's live tokens, and the slots of them this CTA ranks: every
    // rank's partials of those land here, counted in bytes on `landed`.
    const int held = min(kBN, live - t0);
    const u32 ranked = held > static_cast<int>(rank) ? (static_cast<u32>(held) - rank + kS - 1) / kS : 0u;
    if (threadIdx.x == 0 && ranked > 0) bar_arrive_expect_tx(landed, ranked * kS * kE * 4);

    if (warp == 0) {
        // A block past the window streams as any other, its rows unranked.
        if (lane == 0) {
            const int row = first + t0;
            for (int i = 0; i < steps; ++i) {
                const u32 s = static_cast<u32>(i) % kStages;
                const u32 round = static_cast<u32>(i) / kStages;
                if (i >= ahead) {
                    bar_wait(&empty[s], (round - 1) & 1);
                    weight(i);
                }
                tma_load_2d(ring + s * kStage + kG * kTileW, &full[s], &map_x,
                            (kb0 + i) * static_cast<int>(kBoxK), row);
            }
        }
        __syncwarp();
    } else if (warp == 1) {
        for (int i = 0; i < steps; ++i) {
            const u32 s = static_cast<u32>(i) % kStages;
            bar_wait(&full[s], (static_cast<u32>(i) / kStages) & 1);
            tc_fence_after();
            if (elect_one()) {
                const u32 base = smem_addr(ring + s * kStage);
#pragma unroll
                for (u32 kk = 0; kk < kBoxK / 16; ++kk) {
                    const u64 b = sdesc_sw128(base + kG * kTileW + kk * 32);
#pragma unroll
                    for (u32 g = 0; g < kG; ++g) {
                        const u64 a = sdesc_sw128(base + g * kTileW + kk * 32);
                        mma_f16(tmem + g * kStride, a, b, kIdesc, i > 0 || kk > 0);
                    }
                }
                tc_commit(&empty[s]);
                if (i == steps - 1) tc_commit(done);
            }
            __syncwarp();
        }
    } else if (warp < 6) {
        bar_wait(done, 0);
        tc_fence_after();
    } else {
        // While the ring streams: the experts' bias, or the softmax's gain,
        // to the table the ranking reads (visible past the block's sync).
        for (u32 e = threadIdx.x - 192; e < kE; e += 64) {
            const bool real = static_cast<int>(e) < p.experts;
            if constexpr (kScore == kSoftmax)
                table[e] = real && p.gain != nullptr ? Elem<bf16>::to_f32(p.gain[e]) : 1.f;
            else
                table[e] = real && p.bias != nullptr ? p.bias[e] : 0.f;
        }
    }
    // Every CTA of the cluster running (beside the ring) or its ring drained
    // (over it): the partials may land.
    if constexpr (kApart)
        cluster_wait();
    else
        cluster_sync();
    if (warp >= 2 && warp < 6) {
        // Warp w drains TMEM lanes 32 (w % 4) on: experts g 128 + 32 (w % 4)
        // + lane, a token a column, each to its ranking CTA's slot.
        const u32 quarter = warp % 4;
        const u32 e = quarter * 32 + lane;
        const u32 mine = smem_addr(recv) + (rank * kSlots * kE + e) * 4;
        const u32 bar = smem_addr(landed);
#pragma unroll
        for (u32 g = 0; g < kG; ++g) {
#pragma unroll
            for (u32 c0 = 0; c0 < static_cast<u32>(kBN); c0 += 32) {
                u32 v[32];
                tmem_ld32(tmem + ((quarter * 32) << 16) + g * kStride + c0, v);
                tmem_wait_ld();
#pragma unroll
                for (u32 c = 0; c < 32; ++c) {
                    const u32 t = c0 + c;
                    if (t < static_cast<u32>(kBN) && t0 + static_cast<int>(t) < live)
                        st_async_f32(mine + ((t / kS) * kE + g * 128) * 4, bar, t % kS, __uint_as_float(v[c]));
                }
            }
        }
        tc_fence_before();
    }
    __syncthreads();
    if (warp == 1) {
        tc_fence_after();
        tmem_dealloc(tmem, kCols);
    }
    // A CTA with no token to rank is done; one with tokens waits for their
    // partials alone, every rank's.
    if (ranked == 0) return;
    bar_wait(landed, 0);

    // Every (token, expert) this CTA ranks scored by the whole block: its S
    // partials summed in rank order, rounded, scored and keyed past them.
    unsigned* keys = reinterpret_cast<unsigned*>(recv + kS * kSlots * kE);
    float* origs = reinterpret_cast<float*>(keys + kSlots * kE);
    for (u32 at = threadIdx.x; at < ranked * kE; at += kThreads) {
        const int ex = static_cast<int>(at % kE);
        float x = recv[at];
#pragma unroll
        for (u32 r = 1; r < kS; ++r) x += recv[r * kSlots * kE + at];
        if constexpr (kRound) x = bf16_to_f32(f32_to_bf16(x));
        float sc = x;
        if constexpr (kScore == kSigmoid) sc = 1.f / (1.f + expf(-x));
        if constexpr (kScore == kSoftplus) sc = sqrt_softplus(x);
        origs[at] = sc;
        const float v = (kScore != kSoftmax && p.bias != nullptr) ? sc + table[ex] : sc;
        keys[at] = ex < p.experts ? key_of<kScore != kSoftmax>(v) : 0u;
    }
    __syncthreads();
    const int K = p.top_k;
    for (u32 slot = warp; slot < kSlots; slot += kWarps) {
        const int t = static_cast<int>(slot * kS + rank);
        if (t >= kBN || t0 + t >= live) break;
        const int row = first + t0 + t;
        // A lane's experts, 4 lane + 128 m + j (j < 4), sorted once: the
        // best key first, the lower expert first on a tie. A pick is then
        // the best of the lanes' heads, the lowest expert on a tie, and its
        // lane's pop.
        constexpr u32 kN = 4 * kVec;
        unsigned key[kN];
        int idx[kN];
        float orig[kN];
#pragma unroll
        for (u32 m = 0; m < kVec; ++m) {
            const uint4 kv = *reinterpret_cast<const uint4*>(keys + slot * kE + 128 * m + 4 * lane);
            const float4 ov = *reinterpret_cast<const float4*>(origs + slot * kE + 128 * m + 4 * lane);
            const unsigned kk[4] = {kv.x, kv.y, kv.z, kv.w};
            const float oo[4] = {ov.x, ov.y, ov.z, ov.w};
#pragma unroll
            for (u32 j = 0; j < 4; ++j) {
                key[4 * m + j] = kk[j];
                idx[4 * m + j] = static_cast<int>(128 * m + 4 * lane + j);
                orig[4 * m + j] = oo[j];
            }
        }
#pragma unroll
        for (u32 r = 0; r < kN; ++r) {
#pragma unroll
            for (u32 i = r % 2; i + 1 < kN; i += 2) {
                // Branchless: selects, not a branch a compare.
                const bool swap = (key[i + 1] > key[i]) | ((key[i + 1] == key[i]) & (idx[i + 1] < idx[i]));
                const unsigned k0 = key[i], k1 = key[i + 1];
                const int i0 = idx[i], i1 = idx[i + 1];
                const float o0 = orig[i], o1 = orig[i + 1];
                key[i] = swap ? k1 : k0;
                key[i + 1] = swap ? k0 : k1;
                idx[i] = swap ? i1 : i0;
                idx[i + 1] = swap ? i0 : i1;
                orig[i] = swap ? o1 : o0;
                orig[i + 1] = swap ? o0 : o1;
            }
        }
        int my_e = -1;
        float my_o = 0.f;
        bool my_real = false;
        for (int k = 0; k < K; ++k) {
            const unsigned hi = __reduce_max_sync(0xffffffffu, key[0]);
            const unsigned at = __ballot_sync(0xffffffffu, key[0] == hi);
            int owner = __ffs(at) - 1;
            int i = __shfl_sync(0xffffffffu, idx[0], owner);
            if ((at & (at - 1)) != 0) {
                i = static_cast<int>(
                    __reduce_min_sync(0xffffffffu, key[0] == hi ? static_cast<unsigned>(idx[0]) : 0x7fffffffu));
                owner = (i % 128) / 4;
            }
            const float o = __shfl_sync(0xffffffffu, orig[0], owner);
            const bool real = hi != 0u;
            const bool pop = static_cast<int>(lane) == owner;
#pragma unroll
            for (u32 n = 0; n + 1 < kN; ++n) {
                key[n] = pop ? key[n + 1] : key[n];
                idx[n] = pop ? idx[n + 1] : idx[n];
                orig[n] = pop ? orig[n + 1] : orig[n];
            }
            key[kN - 1] = pop ? 0u : key[kN - 1];
            if (static_cast<int>(lane) == k) {
                my_e = real ? i : (kScore == kSigmoid ? 0 : -1);
                my_o = o;
                my_real = real;
            }
        }
        // Each pick's weight-to-be (the softmax's: exp against the best
        // logit), summed in pick order as the routers sum them.
        float my_w;
        if constexpr (kScore == kSoftmax) {
            const float top = __shfl_sync(0xffffffffu, my_o, 0);
            my_w = my_real ? expf(my_o - top) : 0.f;
        } else {
            my_w = my_real ? my_o : 0.f;
        }
        float sum = 0.f;
        for (int k = 0; k < K; ++k) sum += __shfl_sync(0xffffffffu, my_w, k);
        if (static_cast<int>(lane) < K) {
            float w;
            if constexpr (kScore == kSoftmax) {
                const float inv = 1.f / sum;
                w = my_w * inv;
                if (p.gain != nullptr && my_e >= 0) w *= table[my_e];
            } else {
                const float scale = p.renormalize != 0 && sum > 0.f ? p.scaling / sum : p.scaling;
                w = my_w * scale;
            }
            p.routes[static_cast<long long>(row) * K + lane] = my_e;
            p.weights[static_cast<long long>(row) * K + lane] = w;
        }
    }
}

}  // namespace pie::linear

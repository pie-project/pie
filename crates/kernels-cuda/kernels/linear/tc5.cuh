#pragma once

#include "prelude/device.cuh"
#include "prelude/tc5.cuh"

// One persistent, warp-specialized tcgen05 GEMM for every operand kind:
//
//     D[p][q] = alpha * sum_k P[p][k] * Q[q][k]        (both K-major)
//
// P is the MMA's M side (128 rows, or 256 across a CTA pair), Q its kBN-row
// side. Unswapped, P is the activation and Q the weight, and D lands at
// y[p * ldy + q] by TMA store; swapped (a decode's few rows), P is the
// weight and D lands at y[q * ldy + p].
//
// Warps 0-3 drain the accumulator (one TMEM lane quarter each), warp 4's
// lane 0 is the TMA producer, warp 5 owns TMEM and its lane 0 issues every
// MMA, warp 6's lane 0 copies block scales into TMEM. A stage is 128 bytes
// of K for each row of both tiles, so every kind steps the K-major
// descriptor by 32 bytes four times a stage. When the tiles are too few to
// fill the SMs, each splits K and the splits sum in f32 at `part`, which
// tc5_finalize rounds to y.
//
// A pair (kPair 2, a cluster of two) runs cta_group::2: each CTA loads its
// 128 rows of P and half of Q's rows, rank 0 issues the 256-row MMA over
// both, and each drains its own half of D. Both CTAs' loads complete rank
// 0's barriers; rank 0's commits arrive in both.
//
// Grouped (kGroup, a mixture of experts' bank of weights [E][n_per][K]):
//   1  contiguous: the activation's rows sorted by expert, expert e's at
//      [group[e], group[e+1]), each start a multiple of 128 (pad rows are
//      computed and land, unread). P is the activation, tiles never cross
//      an expert.
//   2  masked: expert e's rows at [e * m_max, e * m_max + group[e]) of a
//      [E * m_max] activation, its scales in row blocks padded per expert
//      to whole 128s. Swapped: P is the expert's weight, Q its few rows.
// alpha_e, when given, scales expert e's product too.
//
// Kinds:  0  bf16 x bf16            kind::f16           K 64 a stage
//         1  fp8 x fp8               kind::f8f6f4        K 128
//         2  {e4m3, e2m1} x {..}     kind::mxf8f6f4      K 128, ue8m0 scales
//         3  nvfp4 x nvfp4           kind::mxf4nvf4      K 256, ue4m3 scales a 16
//
// Block scales (kinds 2, 3) arrive in 512-byte atoms: 128 rows x 4 scale
// columns, byte (r % 32) * 16 + (r / 32) * 4 + c -- the layout cuBLASLt and
// the NVFP4 checkpoints use, and exactly what tcgen05.cp 32x128b.warpx4
// lands as TMEM columns. Atoms of a row block run along K; TMA reads them
// as a [row block][atom][128 words] tensor, zero past either end. A scale
// column covers kGran K: 16 (nvfp4), 32 (mx) or 128 (DeepSeek's 1x128 and
// 128x128 blocks, four K-blocks to a 32-bit column, picked by the scale id).

namespace pie::linear {

namespace tc5k {

__host__ __device__ constexpr unsigned stage_k(int kind) {
    return kind == 0 ? 64u : kind == 3 ? 256u : 128u;
}

__host__ __device__ constexpr unsigned atoms_k(int kind, int gran) {
    return kind < 2 ? 0u : (stage_k(kind) / gran / 4 > 0 ? stage_k(kind) / gran / 4 : 1u);
}

__host__ __device__ constexpr unsigned pow2_cols(unsigned n) {
    unsigned c = 32;
    while (c < n) c <<= 1;
    return c;
}

}  // namespace tc5k

template <int kKind, int kFmtP, int kFmtQ, int kGranP, int kGranQ, int kBN, int kStages,
          int kSwap, int kPair, int kGroup>
__global__ __launch_bounds__(224, 1) void tc5_gemm(
    const __grid_constant__ tc5::TmaMap map_p, const __grid_constant__ tc5::TmaMap map_q,
    const __grid_constant__ tc5::TmaMap map_y, const __grid_constant__ tc5::TmaMap map_sfp,
    const __grid_constant__ tc5::TmaMap map_sfq, bf16* __restrict__ y, int rows_p, int rows_q,
    int k, int ldy, float alpha, float* __restrict__ part, int splits,
    const int* __restrict__ group, int experts, int n_per, int m_max,
    const float* __restrict__ alpha_e) {
    using namespace tc5;
    static_assert(kBN % 32 == 0 && kBN >= 32 && kBN <= 256, "the Q tile is 32..256 rows");
    static_assert(kKind >= 0 && kKind <= 3, "four kinds");
    static_assert(kPair == 1 || (kPair == 2 && !kSwap && kBN >= 64), "a pair splits a wide Q");
    static_assert(kGroup == 0 || (kPair == 1 && (kGroup == 1) == !kSwap),
                  "contiguous groups run unswapped, masked swapped, both on lone CTAs");

    constexpr u32 kM = 128 * kPair;
    constexpr u32 kQRows = kBN / kPair;
    constexpr u32 kStageK = tc5k::stage_k(kKind);
    constexpr u32 kInstrK = kStageK / 4;
    constexpr u32 kInner = kKind == 0 ? 64 : 128;  // TMA box width, in the map's elements
    constexpr bool kScaled = kKind >= 2;
    constexpr u32 kTileP = 128 * 128;
    constexpr u32 kTileQ = kQRows * 128;
    // TMA counts the bytes it reads: an fp4 operand unpacked to a byte an
    // element (16U4_ALIGN16B) fills its tile from half as many.
    constexpr u32 kTxP = kKind == 2 && kFmtP == kE2M1 ? kTileP / 2 : kTileP;
    constexpr u32 kTxQ = kKind == 2 && kFmtQ == kE2M1 ? kTileQ / 2 : kTileQ;
    constexpr u32 kAtomsP = tc5k::atoms_k(kKind, kGranP);
    constexpr u32 kAtomsQ = tc5k::atoms_k(kKind, kGranQ);
    constexpr u32 kRowAtomsQ = (kBN + 127) / 128;
    constexpr u32 kSfP = kAtomsP * 512;               // this CTA's P rows
    constexpr u32 kSfQ = kAtomsQ * kRowAtomsQ * 512;  // all of Q's rows, in each CTA
    constexpr u32 kTx = kPair * (kTxP + kTxQ + kSfP + kSfQ);
    constexpr u32 kSfCols = 4 * kAtomsP + 4 * kAtomsQ * kRowAtomsQ;
    // Scales go to TMEM from their own warp, up to four K steps ahead in
    // their own slots: a copy in the MMA thread's stream would hold every
    // MMA behind it until it lands, and a shallower ring would hold the
    // copier on the MMAs.
    constexpr u32 kSfSlots = kStages < 4 ? kStages : 4;
    constexpr u32 kSfAll = kScaled ? kSfSlots * kSfCols : 0;
    // Two accumulator stages always. When they and the scale slots overrun
    // TMEM the stages share kOv columns -- stage 0's tail is stage 1's head --
    // and a tile's first MMA waits until the previous tile's epilogue has
    // read them. The scale slots sit past both stages.
    constexpr u32 kOvRaw = 2 * kBN + kSfAll > 512 ? 2 * kBN + kSfAll - 512 : 0;
    constexpr u32 kOv = (kOvRaw + 15) / 16 * 16;
    constexpr u32 kOvChunks = (kOv + 31) / 32;
    constexpr u32 kStride = kBN - kOv;
    constexpr u32 kCols = tc5k::pow2_cols(kStride + kBN + kSfAll);
    static_assert(kCols <= 512, "TMEM is 512 columns");

    constexpr u32 kOffQ = kStages * kTileP;
    constexpr u32 kOffSfP = kOffQ + kStages * kTileQ;
    constexpr u32 kOffSfQ = kOffSfP + kStages * kSfP;
    // Each epilogue warp stages two 32 x 32 bf16 boxes for the TMA store.
    constexpr u32 kOffOut = (kOffSfQ + kStages * kSfQ + 1023) / 1024 * 1024;
    constexpr u32 kOut = kSwap ? 0 : 4 * 2 * 2048;
    constexpr u32 kOffBar = kOffOut + kOut;
    // A grouped launch holds the routing (offsets or counts) in shared memory:
    // every role reads it once a unit, empty units most.
    constexpr u32 kGroupInts = kGroup == 0 ? 0 : 2 * 1028;
    constexpr u32 kBytes = kOffBar + (2 * kStages + 16) * 8 + 8 + kGroupInts * 4;

    extern __shared__ unsigned char smem_raw[];
    const u32 raw = smem_addr(smem_raw);
    unsigned char* smem = smem_raw + (((raw + 1023u) & ~1023u) - raw);

    u64* full = reinterpret_cast<u64*>(smem + kOffBar);
    u64* empty = full + kStages;
    u64* tfull = empty + kStages;
    u64* tempty = tfull + 2;
    u64* tover = tempty + 2;
    u64* sf_ready = tover + 2;
    u32* tmem_slot = reinterpret_cast<u32*>(sf_ready + 8);
    int* routes = reinterpret_cast<int*>(tmem_slot + 2);
    int* starts = routes + 1028;

    const u32 warp = threadIdx.x / 32;
    const u32 lane = threadIdx.x % 32;
    const u32 rank = kPair == 2 ? cluster_rank() : 0;
    const bool leader = rank == 0;

    if (threadIdx.x == 0) {
        u32 have;
        asm volatile("mov.u32 %0, %%dynamic_smem_size;" : "=r"(have));
        if (have < kBytes + 1024) {
            asm volatile("trap;");
        }
    }
    if (warp == 4 && lane == 0) {
        tma_prefetch(&map_p);
        tma_prefetch(&map_q);
        if constexpr (kScaled) {
            tma_prefetch(&map_sfp);
            tma_prefetch(&map_sfq);
        }
        for (u32 s = 0; s < kStages; ++s) {
            bar_init(&full[s], 1);
            bar_init(&empty[s], 1);
        }
        for (u32 a = 0; a < 2; ++a) {
            bar_init(&tfull[a], 1);
            bar_init(&tempty[a], 4 * kPair);
            bar_init(&tover[a], 4 * kPair);
        }
        for (u32 a = 0; a < kSfSlots; ++a) {
            bar_init(&sf_ready[a], 1);
        }
        bar_init_fence();
    }
    if (warp == 5) {
        tmem_alloc<kPair>(tmem_slot, kCols);
        tmem_relinquish<kPair>();
    }
    // A masked launch counts each expert's tiles up front, so units map
    // straight onto the busy experts: walking empty ones costs every role
    // a unit's bookkeeping each.
    const int masked_p = (n_per + 127) / 128;
    if (kGroup != 0 && warp == 0) {
        for (int i = static_cast<int>(lane); i < experts + (kGroup == 1 ? 1 : 0); i += 32) {
            routes[i] = __ldg(&group[i]);
        }
        if constexpr (kGroup == 2) {
            __syncwarp();
            const int per = (experts + 31) / 32;
            const int lo = min(experts, static_cast<int>(lane) * per);
            const int hi = min(experts, lo + per);
            auto tiles_of = [&](int e) {
                return (min(routes[e], m_max) + kBN - 1) / kBN * masked_p;
            };
            int sum = 0;
            for (int i = lo; i < hi; ++i) {
                sum += tiles_of(i);
            }
            int incl = sum;
            for (int d = 1; d < 32; d <<= 1) {
                const int up = __shfl_up_sync(0xffffffffu, incl, d);
                incl += lane >= static_cast<u32>(d) ? up : 0;
            }
            int run = incl - sum;
            for (int i = lo; i < hi; ++i) {
                starts[i] = run;
                run += tiles_of(i);
            }
            if (lane == 31) {
                starts[experts] = incl;
            }
        }
    }
    tc_fence_before();
    if constexpr (kPair == 2) {
        cluster_sync();
    } else {
        __syncthreads();
    }
    tc_fence_after();
    const u32 tmem = *tmem_slot;

    // Tiles along P and Q (of one expert, when grouped), and the experts.
    // Contiguous P tiles stop at the routed rows.
    const int num_p = kGroup == 2   ? masked_p
                      : kGroup == 1 ? (routes[experts] + 127) / 128
                                    : (rows_p + static_cast<int>(kM) - 1) / static_cast<int>(kM);
    const int num_q = kGroup == 0 ? (rows_q + kBN - 1) / kBN
                      : kGroup == 1 ? (n_per + kBN - 1) / kBN
                                    : (m_max + kBN - 1) / kBN;
    const int num_kb = (k + static_cast<int>(kStageK) - 1) / static_cast<int>(kStageK);
    // A unit is one tile's share of K: `splits` of them a tile when a
    // decode's few tiles would leave SMs idle, summed in f32 at `part`.
    const int kb_per = (num_kb + splits - 1) / splits;
    const int tiles = (kGroup == 2 ? starts[experts] : num_p * num_q) * splits;
    const int first_tile = static_cast<int>(blockIdx.x) / kPair;
    const int tile_step = static_cast<int>(gridDim.x) / kPair;

    // Where a unit reads and lands. D's tile-local [i][j] lands at y[out_p
    // + i][out_q + j] unswapped, y[out_q + j][out_p + i] swapped, for i <
    // lim_p and j < lim_q.
    struct Unit {
        int p_row, q_row, p_block, q_block;
        u32 quarter;
        int kb0, kb1;
        float alpha;
        int out_p, out_q, lim_p, lim_q;
    };
    // Every role walks the same units and skips the same empty ones (a
    // contiguous tile past the routed rows, a masked tile past its count).
    auto unit_of = [&](int u, Unit& x) -> bool {
        const int t = u / splits;
        const int ks = u - t * splits;
        x.kb0 = ks * kb_per;
        x.kb1 = min(num_kb, x.kb0 + kb_per);
        x.alpha = alpha;
        int e = 0, pt, qt;
        if constexpr (kGroup == 2) {
            // The last expert whose tiles start at or before t: an empty
            // expert shares its start with the next.
            int lo = 0, hi = experts - 1;
            while (lo < hi) {
                const int mid = (lo + hi + 1) / 2;
                if (starts[mid] <= t) {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }
            e = lo;
            const int q_tiles = (min(routes[e], m_max) + kBN - 1) / kBN;
            const int r = t - starts[e];
            pt = r / q_tiles;
            qt = r - pt * q_tiles;
        } else {
            // Groups of 8 P tiles walk Q together, so a wave shares both
            // operands.
            constexpr int kRaster = 8;
            const int band = t / (kRaster * num_q);
            const int first = band * kRaster;
            const int size = min(num_p - first, kRaster);
            const int r = t - band * kRaster * num_q;
            pt = first + r % size;
            qt = r / size;
        }
        if constexpr (kGroup == 0) {
            x.p_row = pt * static_cast<int>(kM);
            x.q_row = qt * kBN;
            x.p_block = pt * kPair;
            x.out_p = x.p_row;
            x.out_q = x.q_row;
            x.lim_p = rows_p - x.p_row;
            x.lim_q = rows_q - x.q_row;
        } else if constexpr (kGroup == 1) {
            x.p_row = pt * 128;
            if (x.p_row >= routes[experts]) {
                return false;
            }
            int lo = 0, hi = experts - 1;
            while (lo < hi) {
                const int mid = (lo + hi + 1) / 2;
                if (routes[mid] <= x.p_row) {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }
            e = lo;
            if (routes[e] % 128 != 0) {
                asm volatile("trap;");
            }
            x.q_row = e * n_per + qt * kBN;
            x.p_block = pt;
            x.out_p = x.p_row;
            x.out_q = qt * kBN;
            x.lim_p = 128;
            x.lim_q = n_per - qt * kBN;
        } else {
            const int count = min(routes[e], m_max);
            if (qt * kBN >= count) {
                return false;
            }
            x.p_row = e * n_per + pt * 128;
            x.q_row = e * m_max + qt * kBN;
            x.p_block = x.p_row / 128;
            x.out_p = pt * 128;
            x.out_q = x.q_row;
            x.lim_p = n_per - pt * 128;
            x.lim_q = count - qt * kBN;
        }
        // A Q tile narrower than its 128-row scale atom reads the atom's
        // columns from its own 32-row quarter on; masked rows' scales are
        // padded to whole row blocks an expert.
        const int q_scale_row =
            kGroup == 2 ? e * ((m_max + 127) / 128) * 128 + qt * kBN : x.q_row;
        x.q_block = q_scale_row / 128;
        x.quarter = static_cast<u32>(q_scale_row % 128) / 32;
        if (kGroup != 0 && alpha_e != nullptr) {
            x.alpha *= __ldg(&alpha_e[e]);
        }
        return true;
    };

    // Every arrival a pair counts lands on rank 0's barrier.
    auto arrive_leader = [&](u64* bar) {
        if constexpr (kPair == 2) {
            bar_arrive_at(bar, 0);
        } else {
            bar_arrive(bar);
        }
    };
    auto commit = [&](u64* bar) {
        if constexpr (kPair == 2) {
            tc_commit_pair(bar);
        } else {
            tc_commit(bar);
        }
    };

    if (warp == 4) {
        if (lane == 0) {
            u32 s = 0, phase = 0;
            for (int t = first_tile; t < tiles; t += tile_step) {
                Unit x;
                if (!unit_of(t, x)) {
                    continue;
                }
                const int p0 = x.p_row + static_cast<int>(rank) * 128;
                const int q0 = x.q_row + static_cast<int>(rank * kQRows);
                for (int kb = x.kb0; kb < x.kb1; ++kb) {
                    bar_wait(&empty[s], phase ^ 1);
                    if (leader) {
                        bar_arrive_expect_tx(&full[s], kTx);
                    }
                    unsigned char* tile_p = smem + s * kTileP;
                    unsigned char* tile_q = smem + kOffQ + s * kTileQ;
                    if constexpr (kPair == 2) {
                        tma_load_2d_pair(tile_p, &full[s], &map_p, kb * kInner, p0);
                        tma_load_2d_pair(tile_q, &full[s], &map_q, kb * kInner, q0);
                    } else {
                        tma_load_2d(tile_p, &full[s], &map_p, kb * kInner, p0);
                        tma_load_2d(tile_q, &full[s], &map_q, kb * kInner, q0);
                    }
                    if constexpr (kScaled) {
                        const int atom_p = static_cast<int>(kb * kStageK / kGranP / 4);
                        const int atom_q = static_cast<int>(kb * kStageK / kGranQ / 4);
                        const int block_p = x.p_block + static_cast<int>(rank);
                        const int block_q = x.q_block;
                        unsigned char* sf_p = smem + kOffSfP + s * kSfP;
                        unsigned char* sf_q = smem + kOffSfQ + s * kSfQ;
                        if constexpr (kPair == 2) {
                            tma_load_3d_pair(sf_p, &full[s], &map_sfp, 0, atom_p, block_p);
                            tma_load_3d_pair(sf_q, &full[s], &map_sfq, 0, atom_q, block_q);
                        } else {
                            tma_load_3d(sf_p, &full[s], &map_sfp, 0, atom_p, block_p);
                            tma_load_3d(sf_q, &full[s], &map_sfq, 0, atom_q, block_q);
                        }
                    }
                    if (++s == kStages) {
                        s = 0;
                        phase ^= 1;
                    }
                }
            }
        }
    } else if (warp == 6) {
        // The scale copier: each K step's scales into slot (step %
        // kSfSlots) once the MMAs kSfSlots steps back have read it -- the
        // commit that empties that step's stage says so, and the MMA thread
        // commits nothing else for it. Shared memory holds [row block][atom];
        // TMEM takes [atom][row block], a step's row blocks adjacent.
        if (kScaled && lane == 0 && leader) {
            u32 s = 0, phase = 0, step = 0;
            for (int t = first_tile; t < tiles; t += tile_step) {
                Unit x;
                if (!unit_of(t, x)) {
                    continue;
                }
                for (int kb = x.kb0; kb < x.kb1; ++kb, ++step) {
                    const u32 slot = step % kSfSlots;
                    // A slot is free once the MMAs of the step that last read it
                    // are done: the commit that empties that step's stage says so.
                    if (step >= kSfSlots) {
                        const u32 last = step - kSfSlots;
                        bar_wait(&empty[last % kStages], (last / kStages) & 1);
                    }
                    bar_wait(&full[s], phase);
                    tc_fence_after();
                    const u32 sf_tp = tmem + kStride + kBN + slot * kSfCols;
                    const u32 sf_tq = sf_tp + 4 * kAtomsP;
                    const u32 sf_p = smem_addr(smem + kOffSfP + s * kSfP);
                    const u32 sf_q = smem_addr(smem + kOffSfQ + s * kSfQ);
                    for (u32 a = 0; a < kAtomsP; ++a) {
                        tc_cp_sf_atom<kPair>(sf_tp + 4 * a, sdesc_sf(sf_p + a * 512));
                    }
                    for (u32 r = 0; r < kRowAtomsQ; ++r) {
                        for (u32 a = 0; a < kAtomsQ; ++a) {
                            tc_cp_sf_atom<kPair>(sf_tq + 4 * (a * kRowAtomsQ + r),
                                                 sdesc_sf(sf_q + (r * kAtomsQ + a) * 512));
                        }
                    }
                    tc_commit<kPair>(&sf_ready[slot]);
                    if (++s == kStages) {
                        s = 0;
                        phase ^= 1;
                    }
                }
            }
        }
    } else if (warp == 5) {
        if (lane == 0 && leader) {
            u32 s = 0, phase = 0, step = 0;
            int it = 0;
            for (int t = first_tile; t < tiles; t += tile_step) {
                Unit x;
                if (!unit_of(t, x)) {
                    continue;
                }
                const int kb0 = x.kb0;
                const u32 acc = static_cast<u32>(it) % 2;
                bar_wait(&tempty[acc], ((static_cast<u32>(it) / 2) & 1) ^ 1);
                tc_fence_after();
                const u32 d = tmem + acc * kStride;
                const u32 quarter_q = x.quarter;
                for (int kb = kb0; kb < x.kb1; ++kb, ++step) {
                    bar_wait(&full[s], phase);
                    const u32 slot = step % kSfSlots;
                    if constexpr (kScaled) {
                        bar_wait(&sf_ready[slot], (step / kSfSlots) & 1);
                    }
                    tc_fence_after();
                    const u32 tile_p = smem_addr(smem + s * kTileP);
                    const u32 tile_q = smem_addr(smem + kOffQ + s * kTileQ);
                    const u32 sf_tp = tmem + kStride + kBN + slot * kSfCols;
                    const u32 sf_tq = sf_tp + 4 * kAtomsP;
                    if (kOv > 0 && kb == kb0 && it > 0) {
                        const u32 prev = static_cast<u32>(it - 1);
                        bar_wait(&tover[prev % 2], (prev / 2) & 1);
                        tc_fence_after();
                    }
                    const u64 desc_p = sdesc_sw128(tile_p);
                    const u64 desc_q = sdesc_sw128(tile_q);
#pragma unroll
                    for (u32 j = 0; j < 4; ++j) {
                        const u64 a = desc_p + 2 * j;
                        const u64 b = desc_q + 2 * j;
                        const bool accumulate = kb > kb0 || j > 0;
                        if constexpr (kKind == 0) {
                            mma_f16<kPair>(d, a, b, idesc_dense(kF16Bf16, kF16Bf16, kM, kBN),
                                           accumulate);
                        } else if constexpr (kKind == 1) {
                            mma_f8f6f4<kPair>(d, a, b, idesc_dense(kFmtP, kFmtQ, kM, kBN),
                                              accumulate);
                        } else {
                            const u32 k0 = static_cast<u32>(kb) * kStageK + j * kInstrK;
                            const u32 cp = k0 / kGranP;
                            const u32 cq = k0 / kGranQ;
                            const u32 col_p =
                                sf_tp + 4 * (cp / 4 - static_cast<u32>(kb) * kStageK / kGranP / 4);
                            const u32 col_q =
                                sf_tq +
                                4 * kRowAtomsQ *
                                    (cq / 4 - static_cast<u32>(kb) * kStageK / kGranQ / 4) +
                                quarter_q;
                            const u32 desc = idesc_with_ids(
                                idesc_scaled(kFmtP, kFmtQ, kM, kBN, kKind == 2), cp % 4, cq % 4);
                            if constexpr (kKind == 2) {
                                mma_mxf8f6f4<kPair>(d, a, b, desc, accumulate, col_p, col_q);
                            } else {
                                mma_nvf4<kPair>(d, a, b, desc, accumulate, col_p, col_q);
                            }
                        }
                    }
                    commit(&empty[s]);
                    if (++s == kStages) {
                        s = 0;
                        phase ^= 1;
                    }
                }
                commit(&tfull[acc]);
                ++it;
            }
            // Both CTAs' epilogues have read the last accumulator, so every
            // commit has landed before either CTA retires its barriers.
            if (it > 0) {
                const u32 last = static_cast<u32>(it - 1);
                bar_wait(&tempty[last % 2], (last / 2) & 1);
            }
        }
    } else {
        int it = 0;
        const u32 row = warp * 32 + lane;
        u32 out_buf = 0;
        for (int t = first_tile; t < tiles; t += tile_step) {
            Unit x;
            if (!unit_of(t, x)) {
                continue;
            }
            const u32 acc = static_cast<u32>(it) % 2;
            bar_wait(&tfull[acc], (static_cast<u32>(it) / 2) & 1);
            tc_fence_after();
            ++it;
            const float alpha = x.alpha;
            // This thread's tile-local P row, and where it lands.
            const int i = static_cast<int>(rank * 128 + row);
            const bool live = i < x.lim_p;
            const int p = x.out_p + i;
            const u32 taddr = tmem + acc * kStride + ((warp * 32) << 16);
            // The shared columns go first: stage 0 drains tail first, stage
            // 1 head first, and the next tile's MMA starts once they are read.
            const bool reverse = kOv > 0 && acc == 0;
#pragma unroll 1
            for (u32 chunk = 0; chunk < kBN / 32; ++chunk) {
                const u32 c0 = reverse ? kBN - 32 * (chunk + 1) : 32 * chunk;
                u32 v[32];
                tmem_ld32(taddr + c0, v);
                tmem_wait_ld();
                if (kOv > 0 && chunk + 1 == kOvChunks) {
                    tc_fence_before();
                    __syncwarp();
                    if (lane == 0) {
                        arrive_leader(&tover[acc]);
                    }
                }
                if (chunk + 1 == kBN / 32) {
                    tc_fence_before();
                    __syncwarp();
                    if (lane == 0) {
                        arrive_leader(&tempty[acc]);
                    }
                }
                const int j0 = static_cast<int>(c0);
                const int q0 = x.out_q + j0;
                if constexpr (kSwap) {
                    // A warp's lanes are 32 consecutive weight rows: each
                    // column is one coalesced 64-byte run of y.
                    if (live && splits > 1) {
#pragma unroll
                        for (int j = 0; j < 32; ++j) {
                            if (j0 + j < x.lim_q) {
                                red_add(part + static_cast<size_t>(q0 + j) * ldy + p,
                                        __uint_as_float(v[j]) * alpha);
                            }
                        }
                    } else if (live) {
#pragma unroll
                        for (int j = 0; j < 32; ++j) {
                            if (j0 + j < x.lim_q) {
                                y[static_cast<size_t>(q0 + j) * ldy + p] =
                                    f32_to_bf16(__uint_as_float(v[j]) * alpha);
                            }
                        }
                    }
                } else if (splits > 1) {
                    // A thread's 32 sums of its row, four at a time.
                    if (live) {
                        float* at = part + static_cast<size_t>(p) * ldy + q0;
#pragma unroll
                        for (int j = 0; j < 32; j += 4) {
                            if (j0 + j < x.lim_q) {
                                red_add4(at + j, __uint_as_float(v[j]) * alpha,
                                         __uint_as_float(v[j + 1]) * alpha,
                                         __uint_as_float(v[j + 2]) * alpha,
                                         __uint_as_float(v[j + 3]) * alpha);
                            }
                        }
                    }
                } else {
                    // The warp's 32 x 32 box goes out by TMA from a 64-byte
                    // swizzled stage, which also drops rows and columns past y.
                    unsigned char* box = smem + kOffOut + (warp * 2 + out_buf) * 2048;
                    if (lane == 0) {
                        bulk_wait_read<1>();
                    }
                    __syncwarp();
#pragma unroll
                    for (u32 j = 0; j < 4; ++j) {
                        uint4 w;
                        w.x = pack_bf16x2(__uint_as_float(v[8 * j + 0]) * alpha,
                                          __uint_as_float(v[8 * j + 1]) * alpha);
                        w.y = pack_bf16x2(__uint_as_float(v[8 * j + 2]) * alpha,
                                          __uint_as_float(v[8 * j + 3]) * alpha);
                        w.z = pack_bf16x2(__uint_as_float(v[8 * j + 4]) * alpha,
                                          __uint_as_float(v[8 * j + 5]) * alpha);
                        w.w = pack_bf16x2(__uint_as_float(v[8 * j + 6]) * alpha,
                                          __uint_as_float(v[8 * j + 7]) * alpha);
                        *reinterpret_cast<uint4*>(box + lane * 64 + 16 * (j ^ ((lane >> 1) & 3))) =
                            w;
                    }
                    fence_async_shared();
                    __syncwarp();
                    if (lane == 0) {
                        tma_store_2d(&map_y, box, q0, x.out_p + static_cast<int>(rank * 128 + warp * 32));
                        bulk_commit();
                    }
                    out_buf ^= 1;
                }
            }
        }
        if (!kSwap && lane == 0) {
            bulk_wait_all();
        }
    }

    tc_fence_before();
    if constexpr (kPair == 2) {
        cluster_sync();
    } else {
        __syncthreads();
    }
    if (warp == 5) {
        tc_fence_after();
        tmem_dealloc<kPair>(tmem, kCols);
    }
}

// The split-K sum in f32, four at a time, to bf16, leaving the sum zero for
// the next launch's splits.
__global__ __launch_bounds__(256) void tc5_finalize(float4* __restrict__ part,
                                                    uint2* __restrict__ y, int quads) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < quads) {
        const float4 v = part[i];
        part[i] = make_float4(0.f, 0.f, 0.f, 0.f);
        y[i] = make_uint2(pack_bf16x2(v.x, v.y), pack_bf16x2(v.z, v.w));
    }
}

// As tc5_finalize for a grouped launch's routed rows alone (below
// group[experts] when contiguous, under each count when masked; the rest
// were never summed into): a block a row of `n`.
__global__ __launch_bounds__(128) void tc5_finalize_routed(float4* __restrict__ part,
                                                           uint2* __restrict__ y,
                                                           const int* __restrict__ counts,
                                                           int experts, int m_max, int n) {
    const int row = static_cast<int>(blockIdx.x);
    if (m_max == 0 ? row >= __ldg(&counts[experts]) : row % m_max >= __ldg(&counts[row / m_max])) {
        return;
    }
    const size_t base = static_cast<size_t>(row) * (n / 4);
    for (int i = static_cast<int>(threadIdx.x); i < n / 4; i += blockDim.x) {
        const float4 v = part[base + i];
        part[base + i] = make_float4(0.f, 0.f, 0.f, 0.f);
        y[base + i] = make_uint2(pack_bf16x2(v.x, v.y), pack_bf16x2(v.z, v.w));
    }
}

}  // namespace pie::linear

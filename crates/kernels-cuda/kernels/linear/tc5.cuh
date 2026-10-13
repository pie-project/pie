#pragma once

#include "prelude/device.cuh"
#include "prelude/tc5.cuh"

// One persistent, warp-specialized tcgen05 GEMM for every operand kind:
//
//     D[p][q] = alpha * sum_k P[p][k] * Q[q][k]        (both K-major)
//
// P is the MMA's 128-row side, Q its kBN-row side. Unswapped, P is the
// activation and Q the weight, and D lands at y[p * ldy + q]; swapped
// (a decode's few rows), P is the weight and D lands at y[q * ldy + p].
//
// Warps 0-3 drain the accumulator (one TMEM lane quarter each), warp 4's
// lane 0 is the TMA producer, warp 5 owns TMEM and its lane 0 issues every
// MMA. A stage is 128 bytes of K for each row of both tiles, so every kind
// steps the K-major descriptor by 32 bytes four times a stage.
//
// Kinds:  0  bf16 x bf16             kind::f16           K 64 a stage
//         1  fp8 x fp8               kind::f8f6f4        K 128
//         2  {e4m3, e2m1} x {..}     kind::mxf8f6f4      K 128, ue8m0 scales
//         3  nvfp4 x nvfp4           kind::mxf4nvf4      K 256, ue4m3 scales a 16
//
// Block scales (kinds 2, 3) arrive in 512-byte atoms: 128 rows x 4 scale
// columns, byte (r % 32) * 16 + (r / 32) * 4 + c -- the layout cuBLASLt and
// the NVFP4 checkpoints use, and exactly what tcgen05.cp 32x128b.warpx4
// lands as TMEM columns. Atoms of a row block run along K. A scale column
// covers kGran K: 16 (nvfp4), 32 (mx) or 128 (DeepSeek's 1x128 and 128x128
// blocks, four K-blocks to a 32-bit column, picked by the scale id).

namespace pie::linear {

namespace tc5k {

__host__ __device__ constexpr unsigned stage_k(int kind) { return kind == 0 ? 64u : kind == 3 ? 256u : 128u; }

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
          int kSwap>
__global__ __launch_bounds__(192, 1) void tc5_gemm(
    const __grid_constant__ tc5::TmaMap map_p, const __grid_constant__ tc5::TmaMap map_q,
    const __grid_constant__ tc5::TmaMap map_y,
    const unsigned char* __restrict__ sf_p, const unsigned char* __restrict__ sf_q,
    bf16* __restrict__ y, int rows_p, int rows_q, int k, int ldy, float alpha, int sf_atoms_p,
    int sf_atoms_q) {
    using namespace tc5;
    static_assert(kBN % 32 == 0 && kBN >= 32 && kBN <= 256, "the Q tile is 32..256 rows");
    static_assert(kKind >= 0 && kKind <= 3, "four kinds");

    constexpr u32 kStageK = tc5k::stage_k(kKind);
    constexpr u32 kInstrK = kStageK / 4;
    constexpr u32 kInner = kKind == 0 ? 64 : 128;  // TMA box width, in the map's elements
    constexpr bool kScaled = kKind >= 2;
    constexpr u32 kTileP = 128 * 128;
    constexpr u32 kTileQ = kBN * 128;
    // TMA counts the bytes it reads: an fp4 operand unpacked to a byte an
    // element (16U4_ALIGN16B) fills its tile from half as many.
    constexpr u32 kTxP = kKind == 2 && kFmtP == kE2M1 ? kTileP / 2 : kTileP;
    constexpr u32 kTxQ = kKind == 2 && kFmtQ == kE2M1 ? kTileQ / 2 : kTileQ;
    constexpr u32 kAtomsP = tc5k::atoms_k(kKind, kGranP);
    constexpr u32 kAtomsQ = tc5k::atoms_k(kKind, kGranQ);
    constexpr u32 kRowAtomsQ = (kBN + 127) / 128;
    constexpr u32 kSfP = kAtomsP * 512;
    constexpr u32 kSfQ = kAtomsQ * kRowAtomsQ * 512;
    constexpr u32 kSfCols = 4 * kAtomsP + 4 * kAtomsQ * kRowAtomsQ;
    // Two accumulator stages always. When they and the scale columns overrun
    // TMEM the stages share kOv columns -- stage 0's tail is stage 1's head --
    // and a tile's first MMA waits until the previous tile's epilogue has
    // read them. The scale columns sit past both stages.
    constexpr u32 kOvRaw = 2 * kBN + kSfCols > 512 ? 2 * kBN + kSfCols - 512 : 0;
    constexpr u32 kOv = (kOvRaw + 15) / 16 * 16;
    constexpr u32 kOvChunks = (kOv + 31) / 32;
    constexpr u32 kAcc = 2;
    constexpr u32 kStride = kBN - kOv;
    constexpr u32 kCols = tc5k::pow2_cols(kStride + kBN + kSfCols);
    static_assert(kCols <= 512, "TMEM is 512 columns");

    constexpr u32 kOffQ = kStages * kTileP;
    constexpr u32 kOffSfP = kOffQ + kStages * kTileQ;
    constexpr u32 kOffSfQ = kOffSfP + kStages * kSfP;
    // Each epilogue warp stages two 32 x 32 bf16 boxes for the TMA store.
    constexpr u32 kOffOut = (kOffSfQ + kStages * kSfQ + 1023) / 1024 * 1024;
    constexpr u32 kOut = kSwap ? 0 : 4 * 2 * 2048;
    constexpr u32 kOffBar = kOffOut + kOut;
    constexpr u32 kBytes = kOffBar + (2 * kStages + 6) * 8 + 8;

    extern __shared__ unsigned char smem_raw[];
    const u32 raw = smem_addr(smem_raw);
    unsigned char* smem = smem_raw + (((raw + 1023u) & ~1023u) - raw);

    u64* full = reinterpret_cast<u64*>(smem + kOffBar);
    u64* empty = full + kStages;
    u64* tfull = empty + kStages;
    u64* tempty = tfull + 2;
    u64* tover = tempty + 2;
    u32* tmem_slot = reinterpret_cast<u32*>(tover + 2);

    const u32 warp = threadIdx.x / 32;
    const u32 lane = threadIdx.x % 32;

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
        for (u32 s = 0; s < kStages; ++s) {
            bar_init(&full[s], 1);
            bar_init(&empty[s], 1);
        }
        for (u32 a = 0; a < 2; ++a) {
            bar_init(&tfull[a], 1);
            bar_init(&tempty[a], 4);
            bar_init(&tover[a], 4);
        }
        bar_init_fence();
    }
    if (warp == 5) {
        tmem_alloc(tmem_slot, kCols);
        tmem_relinquish();
    }
    tc_fence_before();
    __syncthreads();
    tc_fence_after();
    const u32 tmem = *tmem_slot;

    const int num_p = (rows_p + 127) / 128;
    const int num_q = (rows_q + kBN - 1) / kBN;
    const int row_atoms_q = (rows_q + 127) / 128;
    const int tiles = num_p * num_q;
    const int num_kb = (k + static_cast<int>(kStageK) - 1) / static_cast<int>(kStageK);

    // Groups of 8 P tiles walk Q together, so a wave shares both operands.
    auto tile_of = [&](int t, int& pt, int& qt) {
        constexpr int kGroup = 8;
        const int group = t / (kGroup * num_q);
        const int first = group * kGroup;
        const int size = min(num_p - first, kGroup);
        const int r = t - group * kGroup * num_q;
        pt = first + r % size;
        qt = r / size;
    };

    if (warp == 4) {
        if (lane == 0) {
            u32 s = 0, phase = 0;
            for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
                int pt, qt;
                tile_of(t, pt, qt);
                for (int kb = 0; kb < num_kb; ++kb) {
                    bar_wait(&empty[s], phase ^ 1);
                    unsigned char* tile_p = smem + s * kTileP;
                    unsigned char* tile_q = smem + kOffQ + s * kTileQ;
                    if constexpr (kScaled) {
                        const int first_p = static_cast<int>(kb * kStageK / kGranP / 4);
                        const int first_q = static_cast<int>(kb * kStageK / kGranQ / 4);
                        unsigned char* atoms_p = smem + kOffSfP + s * kSfP;
                        unsigned char* atoms_q = smem + kOffSfQ + s * kSfQ;
                        // An atom past the tensor (a K tail, a row block past
                        // Q) scales zeros the TMA filled; it is zeroed here,
                        // since a stale NaN scale would poison them.
                        u32 bytes = kTxP + kTxQ;
                        bool zeroed = false;
                        auto zero = [&](unsigned char* atom) {
                            for (u32 i = 0; i < 32; ++i) {
                                reinterpret_cast<uint4*>(atom)[i] = make_uint4(0, 0, 0, 0);
                            }
                            zeroed = true;
                        };
                        for (u32 a = 0; a < kAtomsP; ++a) {
                            if (first_p + static_cast<int>(a) < sf_atoms_p) {
                                bytes += 512;
                            } else {
                                zero(atoms_p + a * 512);
                            }
                        }
                        for (u32 a = 0; a < kAtomsQ; ++a) {
                            for (u32 ra = 0; ra < kRowAtomsQ; ++ra) {
                                const int gra = qt * kBN / 128 + static_cast<int>(ra);
                                if (first_q + static_cast<int>(a) < sf_atoms_q && gra < row_atoms_q) {
                                    bytes += 512;
                                } else {
                                    zero(atoms_q + (a * kRowAtomsQ + ra) * 512);
                                }
                            }
                        }
                        if (zeroed) {
                            fence_async_shared();
                        }
                        bar_arrive_expect_tx(&full[s], bytes);
                        for (u32 a = 0; a < kAtomsP; ++a) {
                            const int at = first_p + static_cast<int>(a);
                            if (at < sf_atoms_p) {
                                bulk_load(atoms_p + a * 512,
                                          sf_p + (static_cast<size_t>(pt) * sf_atoms_p + at) * 512,
                                          512, &full[s]);
                            }
                        }
                        for (u32 a = 0; a < kAtomsQ; ++a) {
                            for (u32 ra = 0; ra < kRowAtomsQ; ++ra) {
                                const int at = first_q + static_cast<int>(a);
                                const int gra = qt * kBN / 128 + static_cast<int>(ra);
                                if (at < sf_atoms_q && gra < row_atoms_q) {
                                    bulk_load(
                                        atoms_q + (a * kRowAtomsQ + ra) * 512,
                                        sf_q + (static_cast<size_t>(gra) * sf_atoms_q + at) * 512,
                                        512, &full[s]);
                                }
                            }
                        }
                    } else {
                        bar_arrive_expect_tx(&full[s], kTxP + kTxQ);
                    }
                    tma_load_2d(tile_p, &full[s], &map_p, kb * kInner, pt * 128);
                    tma_load_2d(tile_q, &full[s], &map_q, kb * kInner, qt * kBN);
                    if (++s == kStages) {
                        s = 0;
                        phase ^= 1;
                    }
                }
            }
        }
    } else if (warp == 5) {
        if (lane == 0) {
            u32 s = 0, phase = 0;
            const u32 sf_tp = tmem + kStride + kBN;
            const u32 sf_tq = sf_tp + 4 * kAtomsP;
            int it = 0;
            for (int t = blockIdx.x; t < tiles; t += gridDim.x, ++it) {
                int pt, qt;
                tile_of(t, pt, qt);
                const u32 acc = static_cast<u32>(it) % kAcc;
                bar_wait(&tempty[acc], ((static_cast<u32>(it) / kAcc) & 1) ^ 1);
                tc_fence_after();
                const u32 d = tmem + acc * kStride;
                // A Q tile narrower than its 128-row scale atom reads the
                // atom's columns from its own 32-row quarter on.
                const u32 quarter_q = static_cast<u32>(qt * kBN % 128) / 32;
                for (int kb = 0; kb < num_kb; ++kb) {
                    bar_wait(&full[s], phase);
                    tc_fence_after();
                    const u32 tile_p = smem_addr(smem + s * kTileP);
                    const u32 tile_q = smem_addr(smem + kOffQ + s * kTileQ);
                    if constexpr (kScaled) {
                        const u32 atoms_p = smem_addr(smem + kOffSfP + s * kSfP);
                        const u32 atoms_q = smem_addr(smem + kOffSfQ + s * kSfQ);
                        for (u32 a = 0; a < kAtomsP; ++a) {
                            tc_cp_sf_atom(sf_tp + 4 * a, sdesc_sf(atoms_p + a * 512));
                        }
                        for (u32 a = 0; a < kAtomsQ * kRowAtomsQ; ++a) {
                            tc_cp_sf_atom(sf_tq + 4 * a, sdesc_sf(atoms_q + a * 512));
                        }
                    }
                    if (kOv > 0 && kb == 0 && it > 0) {
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
                        const bool accumulate = kb > 0 || j > 0;
                        if constexpr (kKind == 0) {
                            mma_f16(d, a, b, idesc_dense(kF16Bf16, kF16Bf16, 128, kBN), accumulate);
                        } else if constexpr (kKind == 1) {
                            mma_f8f6f4(d, a, b, idesc_dense(kFmtP, kFmtQ, 128, kBN), accumulate);
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
                                idesc_scaled(kFmtP, kFmtQ, 128, kBN, kKind == 2), cp % 4, cq % 4);
                            if constexpr (kKind == 2) {
                                mma_mxf8f6f4(d, a, b, desc, accumulate, col_p, col_q);
                            } else {
                                mma_nvf4(d, a, b, desc, accumulate, col_p, col_q);
                            }
                        }
                    }
                    tc_commit(&empty[s]);
                    if (++s == kStages) {
                        s = 0;
                        phase ^= 1;
                    }
                }
                tc_commit(&tfull[acc]);
            }
        }
    } else {
        int it = 0;
        const u32 row = warp * 32 + lane;
        u32 out_buf = 0;
        for (int t = blockIdx.x; t < tiles; t += gridDim.x, ++it) {
            int pt, qt;
            tile_of(t, pt, qt);
            const u32 acc = static_cast<u32>(it) % kAcc;
            bar_wait(&tfull[acc], (static_cast<u32>(it) / kAcc) & 1);
            tc_fence_after();
            const int p = pt * 128 + static_cast<int>(row);
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
                        bar_arrive(&tover[acc]);
                    }
                }
                if (chunk + 1 == kBN / 32) {
                    tc_fence_before();
                    __syncwarp();
                    if (lane == 0) {
                        bar_arrive(&tempty[acc]);
                    }
                }
                const int q0 = qt * kBN + static_cast<int>(c0);
                if constexpr (kSwap) {
                    // A warp's lanes are 32 consecutive weight rows: each
                    // column is one coalesced 64-byte run of y.
                    if (p < rows_p) {
#pragma unroll
                        for (int j = 0; j < 32; ++j) {
                            if (q0 + j < rows_q) {
                                y[static_cast<size_t>(q0 + j) * ldy + p] =
                                    f32_to_bf16(__uint_as_float(v[j]) * alpha);
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
                        *reinterpret_cast<uint4*>(box + lane * 64 + 16 * (j ^ ((lane >> 1) & 3))) = w;
                    }
                    fence_async_shared();
                    __syncwarp();
                    if (lane == 0) {
                        tma_store_2d(&map_y, box, q0, pt * 128 + static_cast<int>(warp) * 32);
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
    __syncthreads();
    if (warp == 5) {
        tc_fence_after();
        tmem_dealloc(tmem, kCols);
    }
}

}  // namespace pie::linear

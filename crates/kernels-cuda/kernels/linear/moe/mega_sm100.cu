#include "linear/act_quant.cuh"
#include "linear/moe/mega.cuh"
#include "prim/mma_sm100.cuh"

// The mega-MoE on tcgen05: a routed MLP (the gate-up product, its GLU, the
// down product and the fold of each token's routes) in one persistent
// launch, a CTA an SM. The shape is DeepGEMM's sm100 mega MoE
// (`sm100_fp8_fp4_mega_moe`, `sm100_bf16_mega_moe`) on one GPU, in this
// codebase's primitives:
//
//   - The routes are laid out by expert in every CTA alike
//     (`linear/moe/mega.cuh`), blocks of kBN pool rows an expert.
//   - The MMA is swapped: the weight's 128 rows on M, a block's routes on N,
//     both K-major from shared memory. A gate-up tile reads 128 gate rows and
//     the 128 up rows `inter` on into two accumulators, so a thread holds a
//     column's gate and up values side by side; a down tile reads 128 rows.
//   - The weights stream through a ring by TMA from their own warp, never
//     waiting on another CTA; the routes' rows come by TMA from the pools
//     once their producers' flags say so (a block's quantized rows, the
//     gate-up tiles that land the K a down step reads).
//   - W4A8 (kKind 1, DeepSeek-V4's experts): MXFP4 weights against e4m3
//     rows in 1x128 blocks, kind::mxf8f6f4 under the weights' ue8m0 a 32
//     and the rows' a 128. The gate-up epilogue lands the GLU's rows as e4m3
//     (the down product's operand) exactly as the apart GLU reads them back.
//     bf16 (kKind 0): kind::f16, the pools bf16.
//   - The down epilogue lands each route's row in bf16; once a column tile's
//     routes have all landed, each token's are folded in route order (the
//     apart combine's sum, element for element).
//
// Warps: 0 loads the weights, 1 the routes' rows, 2 issues the MMAs (and
// owns TMEM), 3 quantizes the routes' rows into the pool, 4-7 run the
// epilogue; 3-7 fold the routes at the end.

namespace pie::linear {

namespace mega {

// `pie::linear::mega::Params`'s host mirror: `linear/moe/mega.rs` `Params`.
struct Params {
    const bf16* x;          // [tokens][hidden]
    const i32* routes;      // [tokens][top]
    const float* weights;   // [tokens][top]
    const bf16* shared;     // [tokens][hidden], added to the fold, or null
    bf16* y;                // [tokens][hidden]
    const u8* up_atoms;     // W4A8: the gate-up bank's ue8m0 scales in atoms
    const u8* down_atoms;   // W4A8: the down bank's
    u8* xpool;              // [pool rows][hidden]: e4m3, or bf16 (two bytes)
    u8* xpool_sf;           // W4A8: its ue8m0 scales a 128, in atoms
    u8* hpool;              // [pool rows][inter]
    u8* hpool_sf;           // W4A8
    bf16* routed;           // [tokens * top][hidden]: each route's down row
    u32* sync;              // ws_words() words, zero at rest
    const u32* win;         // the live window (tokens, first token), or null
    int tokens, top, experts, hidden, inter;
    float limit;            // the GLU's clamp (SwiGLU-clamp)
};

constexpr u32 kTileA = 128 * 128;  // a 128-row, 128-byte box
// A block's pool rows start 64 rows apart: its scales at an even quarter of
// their 128-row atom, as the block-scaled MMA reads them.
constexpr int kPoolStride = 64;
constexpr u32 kStageA = 2 * kTileA;

template <int kBN>
__host__ __device__ constexpr u32 stage_b() { return 2u * kBN * 128u; }

template <int kBN>
__host__ __device__ constexpr u32 stage_bytes() { return kStageA + stage_b<kBN>() + 1024u + 512u; }

// The fixed tail past the ring: barriers, the TMEM slot, the epilogue's
// amax exchange and the routing.
template <int kBN>
__host__ __device__ constexpr u32 tail_bytes() {
    return 256u + 4u * 4u * kBN + static_cast<u32>(sizeof(Routing));
}

template <int kBN, int kStages>
__host__ __device__ constexpr u32 smem_bytes() {
    return kStages * stage_bytes<kBN>() + tail_bytes<kBN>() + 1024u;
}

__device__ __forceinline__ float bf16_round(float v) { return bf16_to_f32(f32_to_bf16(v)); }

// The largest of the warp's u32s (non-negative floats' bits order as theirs).
__device__ __forceinline__ u32 warp_max_u32(u32 v) {
    u32 r;
    asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(r) : "r"(v));
    return r;
}

}  // namespace mega

// kKind: 0 bf16, 1 W4A8. kGlu: 1 SwiGLU, 2 SwiGLU clamped at `limit`.
// kFirst: each route's weight scales its GLU row (DeepSeek's), the fold a
// plain sum plus the shared row; else the fold weighs each route.
template <int kKind, int kBN, int kGlu, int kFirst, int kStages>
__global__ void __launch_bounds__(mega::kThreads, 1) mega_moe(
    const __grid_constant__ tc5::TmaMap map_up, const __grid_constant__ tc5::TmaMap map_down,
    const __grid_constant__ tc5::TmaMap map_x, const __grid_constant__ tc5::TmaMap map_h,
    const __grid_constant__ mega::Params p) {
    using namespace tc5;
    using namespace mega;
    static_assert(kBN == 32 || kBN == 64, "a block is 32 or 64 routes");
    static_assert(kKind == 0 || kKind == 1, "bf16 or W4A8");
    constexpr bool kW4 = kKind == 1;
    // A box's K: 128 bytes of a row (fp4 lands a byte an element).
    constexpr int kBoxK = kW4 ? 128 : 64;
    constexpr u32 kStageB = stage_b<kBN>();
    constexpr u32 kOffB = kStages * kStageA;
    constexpr u32 kOffSfA = kOffB + kStages * kStageB;
    constexpr u32 kOffSfB = kOffSfA + kStages * 1024;
    constexpr u32 kOffTail = kOffSfB + kStages * 512;
    // TMEM: two accumulators of a gate and an up tile, then one scale slot
    // (the weights' two atoms, the rows' one).
    constexpr u32 kAcc = 2 * kBN;
    constexpr u32 kSfCol = 2 * kAcc;
    constexpr u32 kCols = kSfCol + 12 <= 256 ? 256 : 512;
    // TMA counts the bytes it reads: an fp4 box half its tile.
    constexpr u32 kTxA = kW4 ? 2 * (kTileA / 2) + 1024 : 2 * kTileA;
    constexpr u32 kTxB1 = kBN * 128 + (kW4 ? 512 : 0);
    constexpr u32 kTxB2 = 2 * kBN * 128 + (kW4 ? 512 : 0);

    extern __shared__ unsigned char smem_raw[];
    const u32 raw = smem_addr(smem_raw);
    unsigned char* smem = smem_raw + (((raw + 1023u) & ~1023u) - raw);
    u64* full = reinterpret_cast<u64*>(smem + kOffTail);
    u64* empty = full + kStages;
    u64* tfull = empty + kStages;
    u64* tempty = tfull + 2;
    u32* tmem_slot = reinterpret_cast<u32*>(tempty + 2);
    u32* last = tmem_slot + 1;
    float* wmax = reinterpret_cast<float*>(smem + kOffTail + 256);
    Routing& R = *reinterpret_cast<Routing*>(smem + kOffTail + 256 + 4 * 4 * kBN);

    const int warp = static_cast<int>(threadIdx.x) / 32;
    const int lane = static_cast<int>(threadIdx.x) % 32;
    const int H = p.hidden, I = p.inter, K = p.top;
    const int j1 = I / 128, j2 = H / 128;
    const int G = static_cast<int>(gridDim.x);

    if (threadIdx.x == 0) {
        u32 have;
        asm volatile("mov.u32 %0, %%dynamic_smem_size;" : "=r"(have));
        if (have < smem_bytes<kBN, kStages>()) asm volatile("trap;");
        tma_prefetch(&map_up);
        tma_prefetch(&map_down);
        tma_prefetch(&map_x);
        tma_prefetch(&map_h);
        for (int s = 0; s < kStages; ++s) {
            // The weights' arrival and the rows'.
            bar_init(&full[s], 2);
            bar_init(&empty[s], 1);
        }
        for (int a = 0; a < 2; ++a) {
            bar_init(&tfull[a], 1);
            bar_init(&tempty[a], 4);
        }
        bar_init_fence();
    }
    if (warp == 2) {
        tmem_alloc(tmem_slot, kCols);
        tmem_relinquish();
    }
    {
        const int t0 = p.win != nullptr ? static_cast<int>(__ldg(&p.win[1])) : 0;
        const int tl = p.win != nullptr ? static_cast<int>(__ldg(&p.win[0])) : p.tokens;
        // The ring is free until the roles start: its first bytes hold the
        // layout's scratch.
        lay_out<kBN>(R, p.routes, t0, tl, K, p.experts, reinterpret_cast<unsigned short*>(smem));
    }
    tc_fence_before();
    __syncthreads();
    tc_fence_after();
    const u32 tmem = *tmem_slot;
    const int blocks = R.bstart[p.experts];
    const int live_routes = R.start[p.experts];
    const int tasks = blocks * (j1 + j2);
    const long long route0 = static_cast<long long>(R.t0) * K;
    u32* xready = p.sync + ws_xready();
    u32* hready = p.sync + ws_hready();
    u32* done = p.sync + ws_done();

    if (warp == 0) {
        // The weights: a stage is a gate-up tile's 128 K of its gate and up
        // rows, or a down tile's 256 K (W4A8; bf16 half those), and their
        // scales' atoms.
        if (lane == 0) {
            u32 s = 0, ph = 0;
            for (int t = static_cast<int>(blockIdx.x); t < tasks; t += G) {
                const Task x = task_of<kBN>(R, t, blocks, j1, j2);
                const int nk = x.l2 ? I / (2 * kBoxK) : H / kBoxK;
                const int row = x.l2 ? x.e * H + 128 * x.j : x.e * 2 * I + 128 * x.j;
                for (int kb = 0; kb < nk; ++kb) {
                    bar_wait(&empty[s], ph ^ 1);
                    bar_arrive_expect_tx(&full[s], kTxA);
                    unsigned char* a = smem + s * kStageA;
                    if (x.l2) {
                        tma_load_2d(a, &full[s], &map_down, 2 * kb * kBoxK, row);
                        tma_load_2d(a + kTileA, &full[s], &map_down, (2 * kb + 1) * kBoxK, row);
                        if constexpr (kW4) {
                            const long long at = (static_cast<long long>(row / 128) * (I / 128) + 2 * kb) * 512;
                            bulk_load(smem + kOffSfA + s * 1024, p.down_atoms + at, 1024, &full[s]);
                        }
                    } else {
                        tma_load_2d(a, &full[s], &map_up, kb * kBoxK, row);
                        tma_load_2d(a + kTileA, &full[s], &map_up, kb * kBoxK, row + I);
                        if constexpr (kW4) {
                            const long long g = (static_cast<long long>(row / 128) * (H / 128) + kb) * 512;
                            const long long u = (static_cast<long long>((row + I) / 128) * (H / 128) + kb) * 512;
                            bulk_load(smem + kOffSfA + s * 1024, p.up_atoms + g, 512, &full[s]);
                            bulk_load(smem + kOffSfA + s * 1024 + 512, p.up_atoms + u, 512, &full[s]);
                        }
                    }
                    if (++s == kStages) {
                        s = 0;
                        ph ^= 1;
                    }
                }
            }
        }
    } else if (warp == 1) {
        // The routes' rows from the pools, each once its producers say.
        if (lane == 0) {
            u32 s = 0, ph = 0;
            for (int t = static_cast<int>(blockIdx.x); t < tasks; t += G) {
                const Task x = task_of<kBN>(R, t, blocks, j1, j2);
                const int nk = x.l2 ? I / (2 * kBoxK) : H / kBoxK;
                const int p0 = x.b * kPoolStride;
                if (!x.l2) {
                    wait_count(&xready[x.b], static_cast<u32>(x.rows));
                    fence_proxy_global();
                }
                u32 have = 0;
                for (int kb = 0; kb < nk; ++kb) {
                    if (x.l2) {
                        // A down step's K: the gate-up tiles 2 kb and 2 kb + 1
                        // (W4A8), or tile kb (bf16).
                        const u32 need = kW4 ? 3u << (2 * kb) : 1u << kb;
                        if ((have & need) != need) {
                            wait_bits(&hready[x.b], need);
                            have = ld_acquire(&hready[x.b]);
                            fence_proxy_global();
                        }
                    }
                    bar_wait(&empty[s], ph ^ 1);
                    unsigned char* bt = smem + kOffB + s * kStageB;
                    if (x.l2) {
                        bar_arrive_expect_tx(&full[s], kTxB2);
                        tma_load_2d(bt, &full[s], &map_h, 2 * kb * kBoxK, p0);
                        tma_load_2d(bt + kBN * 128, &full[s], &map_h, (2 * kb + 1) * kBoxK, p0);
                        if constexpr (kW4) {
                            const long long at = (static_cast<long long>(p0 / 128) * (I / 512) + kb / 2) * 512;
                            bulk_load(smem + kOffSfB + s * 512, p.hpool_sf + at, 512, &full[s]);
                        }
                    } else {
                        bar_arrive_expect_tx(&full[s], kTxB1);
                        tma_load_2d(bt, &full[s], &map_x, kb * kBoxK, p0);
                        if constexpr (kW4) {
                            const long long at = (static_cast<long long>(p0 / 128) * (H / 512) + kb / 4) * 512;
                            bulk_load(smem + kOffSfB + s * 512, p.xpool_sf + at, 512, &full[s]);
                        }
                    }
                    if (++s == kStages) {
                        s = 0;
                        ph ^= 1;
                    }
                }
            }
        }
    } else if (warp == 2) {
        // The MMAs: the warp walks the steps, its elected lane issues. The
        // scales go to TMEM from the same thread ahead of the MMAs that
        // read them, in the tensor pipe's order, into one slot.
        u32 s = 0, ph = 0;
        int it = 0;
        const u32 sfa = tmem + kSfCol;
        const u32 sfb = sfa + 8;
        for (int t = static_cast<int>(blockIdx.x); t < tasks; t += G) {
            const Task x = task_of<kBN>(R, t, blocks, j1, j2);
            const int nk = x.l2 ? I / (2 * kBoxK) : H / kBoxK;
            const u32 acc = static_cast<u32>(it) % 2;
            bar_wait(&tempty[acc], ((static_cast<u32>(it) / 2) & 1) ^ 1);
            tc_fence_after();
            const u32 d0 = tmem + acc * kAcc;
            // The block's rows from their own half of the atom (the MMA faults
            // on scales at an odd quarter).
            const u32 quarter = static_cast<u32>((x.b * kPoolStride) % 128) / 32;
            for (int kb = 0; kb < nk; ++kb) {
                bar_wait(&full[s], ph);
                tc_fence_after();
                __syncwarp();
                if (elect_one()) {
                    const u32 ta = smem_addr(smem + s * kStageA);
                    const u32 tb = smem_addr(smem + kOffB + s * kStageB);
                    if constexpr (kW4) {
                        const u32 sa = smem_addr(smem + kOffSfA + s * 1024);
                        tc_cp_sf_atom(sfa, sdesc_sf(sa));
                        tc_cp_sf_atom(sfa + 4, sdesc_sf(sa + 512));
                        tc_cp_sf_atom(sfb, sdesc_sf(smem_addr(smem + kOffSfB + s * 512)));
                    }
                    const u64 db = sdesc_sw128(tb);
                    if (x.l2) {
                        const u64 da = sdesc_sw128(ta);
#pragma unroll
                        for (u32 j = 0; j < 8; ++j) {
                            const u64 a = da + (j / 4) * (kTileA >> 4) + 2 * (j % 4);
                            const u64 b = db + (j / 4) * ((kBN * 128) >> 4) + 2 * (j % 4);
                            const bool more = kb > 0 || j > 0;
                            if constexpr (kW4) {
                                const u32 cq = 2 * static_cast<u32>(kb) + j / 4;
                                const u32 desc = idesc_with_ids(
                                    idesc_scaled(kE2M1, kE4M3, 128, kBN, true), j % 4, cq % 4);
                                mma_mxf8f6f4(d0, a, b, desc, more, sfa + 4 * (j / 4), sfb + quarter);
                            } else {
                                mma_f16(d0, a, b, idesc_dense(kF16Bf16, kF16Bf16, 128, kBN), more);
                            }
                        }
                    } else {
#pragma unroll
                        for (u32 h = 0; h < 2; ++h) {
                            const u64 da = sdesc_sw128(ta + h * kTileA);
#pragma unroll
                            for (u32 j = 0; j < 4; ++j) {
                                const bool more = kb > 0 || j > 0;
                                if constexpr (kW4) {
                                    const u32 desc = idesc_with_ids(
                                        idesc_scaled(kE2M1, kE4M3, 128, kBN, true), j,
                                        static_cast<u32>(kb) % 4);
                                    mma_mxf8f6f4(d0 + h * kBN, da + 2 * j, db + 2 * j, desc, more,
                                                 sfa + 4 * h, sfb + quarter);
                                } else {
                                    mma_f16(d0 + h * kBN, da + 2 * j, db + 2 * j,
                                            idesc_dense(kF16Bf16, kF16Bf16, 128, kBN), more);
                                }
                            }
                        }
                    }
                    tc_commit(&empty[s]);
                }
                __syncwarp();
                if (++s == kStages) {
                    s = 0;
                    ph ^= 1;
                }
            }
            if (elect_one()) {
                tc_commit(&tfull[acc]);
            }
            __syncwarp();
            ++it;
        }
        // Every commit has landed before the barriers retire.
        if (it > 0) {
            const u32 l = static_cast<u32>(it - 1);
            bar_wait(&tempty[l % 2], (l / 2) & 1);
        }
    } else {
        // Warps 3-7 first quantize (or copy) the routes' rows of this CTA's
        // blocks into the activation pool, a row a warp at a time.
        {
            const int qw = warp - 3;
            int k = 0;
            for (int b = static_cast<int>(blockIdx.x); b < blocks; b += G) {
                const int e = R.block_expert[b];
                const int m = b - R.bstart[e];
                const int rows = min(R.count[e] - m * kBN, kBN);
                for (int n = 0; n < rows; ++n, ++k) {
                    if (k % 5 != qw) continue;
                    const int r = R.sorted[R.start[e] + m * kBN + n];
                    const bf16* src = p.x + (route0 + r) / K * H;
                    const long long prow = static_cast<long long>(b) * kPoolStride + n;
                    if constexpr (kW4) {
                        // Two 128-blocks a pass, sixteen lanes a block, four
                        // passes' loads in flight.
                        const int half = lane / 16, sub = lane % 16;
                        for (int c0 = 0; c0 < H / 128; c0 += 8) {
                            uint4 w[4];
#pragma unroll
                            for (int q = 0; q < 4; ++q) {
                                const int c = c0 + 2 * q + half;
                                w[q] = c < H / 128 ? *reinterpret_cast<const uint4*>(src + c * 128 + sub * 8)
                                                   : make_uint4(0, 0, 0, 0);
                            }
#pragma unroll
                            for (int q = 0; q < 4; ++q) {
                                const int c = c0 + 2 * q + half;
                                E4m3Block<128> blk;
                                const unsigned words[4] = {w[q].x, w[q].y, w[q].z, w[q].w};
                                float amax = 0.f;
#pragma unroll
                                for (int i = 0; i < 4; ++i) {
                                    blk.v[2 * i] = __uint_as_float(words[i] << 16);
                                    blk.v[2 * i + 1] = __uint_as_float(words[i] & 0xFFFF0000u);
                                    amax = fmaxf(amax, fmaxf(fabsf(blk.v[2 * i]), fabsf(blk.v[2 * i + 1])));
                                }
#pragma unroll
                                for (int o = 8; o > 0; o >>= 1) {
                                    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
                                }
                                blk.e = e4m3_block_exp(amax);
                                if (c < H / 128) {
                                    *reinterpret_cast<unsigned long long*>(p.xpool + prow * H + c * 128 + sub * 8) =
                                        blk.codes();
                                    if (sub == 0) {
                                        p.xpool_sf[atom_offset(static_cast<int>(prow), c, H / 128)] =
                                            static_cast<u8>(blk.e + 127);
                                    }
                                }
                            }
                        }
                    } else {
                        const uint4* s4 = reinterpret_cast<const uint4*>(src);
                        uint4* d4 = reinterpret_cast<uint4*>(p.xpool + prow * H * 2);
                        for (int c = lane; c < H / 8; c += 32) d4[c] = s4[c];
                    }
                    fence_proxy_global();
                    __syncwarp();
                    if (lane == 0) add_release(&xready[b], 1);
                }
            }
        }
        if (warp >= 4) {
            // The epilogue: thread i holds row i of the tile (a GLU column, or
            // an output column) for the block's kBN routes.
            const int q = warp - 4;
            const int i = q * 32 + lane;
            int it = 0;
            for (int t = static_cast<int>(blockIdx.x); t < tasks; t += G) {
                const Task x = task_of<kBN>(R, t, blocks, j1, j2);
                const u32 acc = static_cast<u32>(it) % 2;
                bar_wait(&tfull[acc], (static_cast<u32>(it) / 2) & 1);
                tc_fence_after();
                const u32 base = tmem + acc * kAcc + ((static_cast<u32>(q) * 32) << 16);
                const long long p0 = static_cast<long long>(x.b) * kPoolStride;
                if (!x.l2) {
                    for (int c0 = 0; c0 < kBN; c0 += 32) {
                        u32 g[32], u[32];
                        tmem_ld32(base + c0, g);
                        tmem_ld32(base + kBN + c0, u);
                        tmem_wait_ld();
                        if (c0 + 32 == kBN) {
                            tc_fence_before();
                            __syncwarp();
                            if (lane == 0) bar_arrive(&tempty[acc]);
                        }
                        // Lane n's route: column c0 + n.
                        const int n_me = c0 + lane;
                        float w_me = 1.f;
                        if (kFirst && n_me < x.rows) {
                            w_me = __ldg(&p.weights[route0 + R.sorted[x.s0 + n_me]]);
                        }
                        float v[32];
#pragma unroll
                        for (int n = 0; n < 32; ++n) {
                            // The gate-up product lands bf16 apart, the GLU
                            // reading it so.
                            float gv = bf16_round(__uint_as_float(g[n]));
                            float uv = bf16_round(__uint_as_float(u[n]));
                            float y;
                            if constexpr (kGlu == 2) {
                                gv = fminf(gv, p.limit);
                                uv = fminf(fmaxf(uv, -p.limit), p.limit);
                                y = (gv / (1.f + expf(-gv))) * uv;
                            } else {
                                y = gv / (1.f + __expf(-gv)) * uv;
                            }
                            if constexpr (kFirst) {
                                y = y * __shfl_sync(0xffffffffu, w_me, n);
                            }
                            v[n] = bf16_round(y);
                        }
                        const int col = 128 * x.j + i;
                        if constexpr (kW4) {
                            // A 128-column block's amax: the warps' maxima met
                            // in shared memory.
                            float mine = 0.f;
#pragma unroll
                            for (int n = 0; n < 32; ++n) {
                                const u32 m = warp_max_u32(__float_as_uint(fabsf(v[n])));
                                if (lane == n) mine = __uint_as_float(m);
                            }
                            wmax[q * kBN + c0 + lane] = mine;
                            epi_sync();
#pragma unroll
                            for (int n = 0; n < 32; ++n) {
                                if (c0 + n >= x.rows) break;
                                const float amax = fmaxf(fmaxf(wmax[c0 + n], wmax[kBN + c0 + n]),
                                                         fmaxf(wmax[2 * kBN + c0 + n], wmax[3 * kBN + c0 + n]));
                                const int e = e4m3_block_exp(amax);
                                const float inv = __uint_as_float(static_cast<unsigned>(127 - e) << 23);
                                const float s = fminf(fmaxf(v[n] * inv, -448.f), 448.f);
                                const unsigned short pair = static_cast<unsigned short>(
                                    __nv_cvt_float2_to_fp8x2(make_float2(s, 0.f), __NV_SATFINITE, __NV_E4M3));
                                p.hpool[(p0 + c0 + n) * I + col] = static_cast<u8>(pair & 0xFF);
                                if (i == n) {
                                    p.hpool_sf[atom_offset(static_cast<int>(p0 + c0 + n), x.j, I / 128)] =
                                        static_cast<u8>(e + 127);
                                }
                            }
                            epi_sync();
                        } else {
                            bf16* h = reinterpret_cast<bf16*>(p.hpool);
#pragma unroll
                            for (int n = 0; n < 32; ++n) {
                                if (c0 + n >= x.rows) break;
                                h[(p0 + c0 + n) * I + col] = f32_to_bf16(v[n]);
                            }
                        }
                    }
                    fence_proxy_global();
                    epi_sync();
                    if (i == 0) or_release(&hready[x.b], 1u << x.j);
                } else {
                    for (int c0 = 0; c0 < kBN; c0 += 32) {
                        u32 d[32];
                        tmem_ld32(base + c0, d);
                        tmem_wait_ld();
                        if (c0 + 32 == kBN) {
                            tc_fence_before();
                            __syncwarp();
                            if (lane == 0) bar_arrive(&tempty[acc]);
                        }
                        const int n_me = c0 + lane;
                        const long long r_me = n_me < x.rows ? route0 + R.sorted[x.s0 + n_me] : 0;
                        const int col = 128 * x.j + i;
#pragma unroll
                        for (int n = 0; n < 32; ++n) {
                            if (c0 + n >= x.rows) break;
                            const long long r = __shfl_sync(0xffffffffu, r_me, n);
                            p.routed[r * H + col] = f32_to_bf16(__uint_as_float(d[n]));
                        }
                    }
                    __threadfence();
                    epi_sync();
                    if (i == 0) add_release(&done[x.j], static_cast<u32>(x.rows));
                }
                ++it;
            }
        }
        // The fold: token t's 128-column tile j once every route has landed
        // its rows there, eight columns a lane, the routes in order.
        {
            const int fw = warp - 3;
            const int items = R.tokens * j2;
            for (int item = static_cast<int>(blockIdx.x) + G * fw; item < items; item += 5 * G) {
                const int t = item / j2, j = item - t * j2;
                wait_count(&done[j], static_cast<u32>(live_routes));
                if (lane < 16) {
                    const int col = 128 * j + 8 * lane;
                    const long long tok = R.t0 + t;
                    float acc[8];
#pragma unroll
                    for (int c = 0; c < 8; ++c) acc[c] = 0.f;
                    for (int k = 0; k < K; ++k) {
                        const long long r = tok * K + k;
                        const int e = __ldg(&p.routes[r]);
                        uint4 v = make_uint4(0, 0, 0, 0);
                        if (e >= 0 && e < p.experts) {
                            v = __ldcg(reinterpret_cast<const uint4*>(p.routed + r * H + col));
                        }
                        const unsigned wv[4] = {v.x, v.y, v.z, v.w};
                        if constexpr (kFirst) {
#pragma unroll
                            for (int c = 0; c < 4; ++c) {
                                acc[2 * c] += __uint_as_float(wv[c] << 16);
                                acc[2 * c + 1] += __uint_as_float(wv[c] & 0xFFFF0000u);
                            }
                        } else {
                            const float wt = __ldg(&p.weights[r]);
#pragma unroll
                            for (int c = 0; c < 4; ++c) {
                                acc[2 * c] += wt * __uint_as_float(wv[c] << 16);
                                acc[2 * c + 1] += wt * __uint_as_float(wv[c] & 0xFFFF0000u);
                            }
                        }
                    }
                    if (kFirst && p.shared != nullptr) {
                        const uint4 sv = *reinterpret_cast<const uint4*>(p.shared + tok * H + col);
                        const unsigned sw[4] = {sv.x, sv.y, sv.z, sv.w};
#pragma unroll
                        for (int c = 0; c < 4; ++c) {
                            acc[2 * c] = acc[2 * c] + __uint_as_float(sw[c] << 16);
                            acc[2 * c + 1] = acc[2 * c + 1] + __uint_as_float(sw[c] & 0xFFFF0000u);
                        }
                    }
                    uint4 out;
                    out.x = pack_bf16x2(acc[0], acc[1]);
                    out.y = pack_bf16x2(acc[2], acc[3]);
                    out.z = pack_bf16x2(acc[4], acc[5]);
                    out.w = pack_bf16x2(acc[6], acc[7]);
                    *reinterpret_cast<uint4*>(p.y + tok * H + col) = out;
                }
                __syncwarp();
            }
        }
    }

    // The last CTA out zeroes the flags for the next launch.
    tc_fence_before();
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        *last = atomicAdd(&p.sync[ws_finish()], 1u) == static_cast<u32>(G - 1) ? 1u : 0u;
    }
    __syncthreads();
    if (*last != 0) {
        __threadfence();
        for (int b = static_cast<int>(threadIdx.x); b < blocks; b += kThreads) {
            xready[b] = 0;
            hready[b] = 0;
        }
        for (int j = static_cast<int>(threadIdx.x); j < j2; j += kThreads) done[j] = 0;
        if (threadIdx.x == 0) p.sync[ws_finish()] = 0;
    }
    if (warp == 2) {
        tc_fence_after();
        tmem_dealloc(tmem, kCols);
    }
}

}  // namespace pie::linear

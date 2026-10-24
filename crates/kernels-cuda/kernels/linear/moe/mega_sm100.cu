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
//     (`linear/moe/mega.cuh`), blocks of kBN routes an expert, or for a
//     wide fire read where `moe_route_order_scan` laid them in global
//     memory; the CTAs claim the tasks from one count.
//   - The MMA is swapped: the weight's 128 rows on M, a block's routes on N,
//     both K-major from shared memory. A gate-up tile reads 128 gate rows and
//     the 128 up rows `inter` on into two accumulators, so a thread holds a
//     column's gate and up values side by side; a down tile reads 128 rows.
//   - The weights stream through a ring by TMA from their own warp, never
//     waiting on another CTA; the routes' rows come by TMA from the pools
//     once their producers' flags say so (a block's quantized rows, the
//     gate-up tiles that land the K a down step reads).
//   - W4A8 (kKind 1, DeepSeek-V4's experts): MXFP4 weights against e4m3
//     rows in 1x128 blocks, kind::mxf8f6f4 under the weights' ue8m0 a 32 and
//     the rows' a 128. The gate-up epilogue lands the GLU's rows as e4m3 (the
//     down product's operand) exactly as the apart GLU reads them back.
//     bf16 (kKind 0): kind::f16, the pools bf16.
//   - Two epilogue warpgroups take alternate tasks, each its own
//     accumulator. The down epilogue lands each route's row in bf16; once a
//     column tile's routes have all landed, each token's are folded in route
//     order (the apart combine's sum, element for element).
//
// Warps: 0 claims the tasks and loads the weights, 1 the routes' rows, 2
// issues the MMAs (and owns TMEM), 3 quantizes, 4-7 and 8-11 run the
// epilogue; 3-11 quantize the routes' rows into the pool first and fold the
// routes last.

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
    const i32* order;       // a wide fire's layout (`moe_route_order_scan`'s), or null
    const i32* offsets;
    const i32* work;
    int tokens, top, experts, hidden, inter;
    float limit;            // the GLU's clamp (SwiGLU-clamp)
};

constexpr u32 kTileA = 128 * 128;  // a 128-row, 128-byte box
constexpr u32 kStageA = 2 * kTileA;
// A block's pool rows start 64 rows apart: its scales at an even quarter of
// their 128-row atom, as the block-scaled MMA reads them.
constexpr int kPoolStride = 64;
// Task slots between the claiming warp and the rest.
constexpr int kSlots = 4;
// Columns a quantizer pass reads of a row: eight 128-blocks, two at once
// across the warp's halves, four passes' loads in flight.
constexpr int kChunk = 1024;
// The most tokens a fire folds in its down epilogue, the last route of a
// column tile in folding the tile; a wider one's fold is spread over the
// CTAs at the end.
constexpr int kFoldInline = 16;

template <int kBN>
__host__ __device__ constexpr u32 stage_b() { return 2u * kBN * 128u; }

template <int kBN>
__host__ __device__ constexpr u32 stage_bytes() { return kStageA + stage_b<kBN>() + 1024u + 512u; }

// The fixed tail past the ring: barriers and task slots, the epilogue
// warpgroups' amax exchanges and the routing.
template <int kBN>
__host__ __device__ constexpr u32 tail_bytes() {
    return 256u + 2u * 4u * 4u * kBN + static_cast<u32>(sizeof(Routing));
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
    // A box's K: 128 bytes of a row (fp4 lands a byte an element). A gate-up
    // stage is one box of its gate rows and one of its up rows, a down stage
    // two boxes along K.
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
    static_assert(2 * kStages + 4 + 2 * kSlots <= 28, "the tail holds the barriers");

    extern __shared__ unsigned char smem_raw[];
    const u32 raw = smem_addr(smem_raw);
    unsigned char* smem = smem_raw + (((raw + 1023u) & ~1023u) - raw);
    u64* full = reinterpret_cast<u64*>(smem + kOffTail);
    u64* empty = full + kStages;
    u64* tfull = empty + kStages;
    u64* tempty = tfull + 2;
    u64* task_full = tempty + 2;
    u64* task_empty = task_full + kSlots;
    int* task_slot = reinterpret_cast<int*>(task_empty + kSlots);
    u32* tmem_slot = reinterpret_cast<u32*>(task_slot + kSlots);
    u32* flag = tmem_slot + 1;
    u32* wg_flag = flag + 1;  // an epilogue warpgroup's own, two
    float* wmax_all = reinterpret_cast<float*>(smem + kOffTail + 256);
    Routing& R = *reinterpret_cast<Routing*>(smem + kOffTail + 256 + 2 * 4 * 4 * kBN);

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
        for (int a = 0; a < kSlots; ++a) {
            bar_init(&task_full[a], 1);
            // The row loader, the MMA warp and the task's epilogue warpgroup.
            bar_init(&task_empty[a], 6);
        }
        bar_init_fence();
    }
    if (warp == 2) {
        tmem_alloc(tmem_slot, kCols);
        tmem_relinquish();
    }
    // As a programmatic dependent (`kernel.pdl`) the set-up above overlapped
    // the producer's tail; the routes, rows and flags are read past this.
    pdl_wait();
    {
        const int t0 = p.win != nullptr ? static_cast<int>(__ldg(&p.win[1])) : 0;
        const int tl = p.win != nullptr ? static_cast<int>(__ldg(&p.win[0])) : p.tokens;
        // The ring is free until the roles start: its first bytes hold the
        // layout's scratch.
        if (p.order != nullptr) {
            lay_out_global(R, p.order, p.offsets, p.work, t0, tl, K, p.experts);
        } else {
            lay_out<kBN>(R, p.routes, t0, tl, K, p.experts, reinterpret_cast<unsigned short*>(smem));
        }
    }
    tc_fence_before();
    __syncthreads();
    tc_fence_after();
    const u32 tmem = *tmem_slot;
    const int blocks = R.bstart[p.experts];
    const int live_routes = R.start[p.experts];
    const int tasks = blocks * (j1 + j2);
    const long long route0 = static_cast<long long>(R.t0) * K;
    const u32 chunks = static_cast<u32>(H / kChunk);
    u32* xready = p.sync + ws_xready();
    u32* hready = p.sync + ws_hready();
    u32* done = p.sync + ws_done();

    // Token t's 128-column tile j folded by a warp, eight columns a lane, the
    // routes in order (with `wait`, once every route has landed there).
    auto fold = [&](int t, int j, bool wait) {
            const int col = 128 * j + 8 * (lane % 16);
            const long long tok = R.t0 + t;
            // The token's experts and weights (lane k route k's), and its
            // shared row, read while its routes land.
            const int e_lane = lane < K ? __ldg(&p.routes[tok * K + lane]) : -1;
            const float w_lane = !kFirst && lane < K ? __ldg(&p.weights[tok * K + lane]) : 0.f;
            uint4 sv = make_uint4(0, 0, 0, 0);
            if (kFirst && p.shared != nullptr && lane < 16) {
                sv = *reinterpret_cast<const uint4*>(p.shared + tok * H + col);
            }
            if (wait) {
                if (lane == 0) wait_count(&done[j], static_cast<u32>(live_routes));
                __syncwarp();
            }
            float acc[8];
#pragma unroll
            for (int c = 0; c < 8; ++c) acc[c] = 0.f;
            // Eight routes' rows in flight at once, summed in route order.
            for (int k0 = 0; k0 < K; k0 += 8) {
                uint4 v[8];
                float wt[8];
#pragma unroll
                for (int u = 0; u < 8; ++u) {
                    const int k = k0 + u;
                    const int e = __shfl_sync(0xffffffffu, e_lane, k % 32);
                    wt[u] = __shfl_sync(0xffffffffu, w_lane, k % 32);
                    v[u] = make_uint4(0, 0, 0, 0);
                    if (k < K && e >= 0 && e < p.experts && lane < 16) {
                        v[u] = __ldcg(reinterpret_cast<const uint4*>(p.routed + (tok * K + k) * H + col));
                    }
                }
#pragma unroll
                for (int u = 0; u < 8; ++u) {
                    if (k0 + u >= K) break;
                    const unsigned wv[4] = {v[u].x, v[u].y, v[u].z, v[u].w};
#pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        if constexpr (kFirst) {
                            acc[2 * c] += __uint_as_float(wv[c] << 16);
                            acc[2 * c + 1] += __uint_as_float(wv[c] & 0xFFFF0000u);
                        } else {
                            acc[2 * c] += wt[u] * __uint_as_float(wv[c] << 16);
                            acc[2 * c + 1] += wt[u] * __uint_as_float(wv[c] & 0xFFFF0000u);
                        }
                    }
                }
            }
            if (lane < 16) {
                if (kFirst && p.shared != nullptr) {
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
    };

    // A role's walk over the claimed tasks: slot n % kSlots, its phase.
    auto claimed = [&](int n) {
        const int s = n % kSlots;
        bar_wait(&task_full[s], (n / kSlots) & 1);
        return task_slot[s];
    };

    if (warp == 0) {
        // Claims each task as the ring frees, then streams its weights: a
        // gate-up stage's 128 K of its gate and up rows, a down stage's 256 K
        // (W4A8; bf16 half those), and their scales' atoms. Past the last
        // task, two slots say so (each epilogue warpgroup reads every other).
        if (lane == 0) {
            u32 s = 0, ph = 0;
            int ended = 0;
            for (int n = 0; ended < 2; ++n) {
                const int slot = n % kSlots;
                bar_wait(&task_empty[slot], ((n / kSlots) & 1) ^ 1);
                const u32 got = ended ? static_cast<u32>(tasks) : atomicAdd(p.sync + ws_next(), 1u);
                const int t = got < static_cast<u32>(tasks) ? static_cast<int>(got) : -1;
                task_slot[slot] = t;
                bar_arrive(&task_full[slot]);
                if (t < 0) {
                    ++ended;
                    continue;
                }
                const Task x = task_of<kBN>(R, t, blocks, j1, j2);
                const int nk = x.l2 ? I / (2 * kBoxK) : H / kBoxK;
                const int row = x.l2 ? x.e * H + 128 * x.j : x.e * 2 * I + 128 * x.j;
                for (int kb = 0; kb < nk; ++kb) {
                    bar_wait(&empty[s], ph ^ 1);
                    bar_arrive_expect_tx(&full[s], kTxA);
                    unsigned char* a = smem + s * kStageA;
                    if (x.l2) {
                        // Its two K boxes, one load (the map's K in box chunks).
                        tma_load_3d(a, &full[s], &map_down, 0, row, 2 * kb);
                        if constexpr (kW4) {
                            const long long at = (static_cast<long long>(row / 128) * (I / 128) + 2 * kb) * 512;
                            bulk_load(smem + kOffSfA + s * 1024, p.down_atoms + at, 1024, &full[s]);
                        }
                    } else {
                        // Its gate box and up box, one load (the map's rows
                        // in an expert's two halves).
                        tma_load_3d(a, &full[s], &map_up, kb * kBoxK, 128 * x.j, 2 * x.e);
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
            for (int n = 0;; ++n) {
                const int t = claimed(n);
                bar_arrive(&task_empty[n % kSlots]);
                if (t < 0) break;
                const Task x = task_of<kBN>(R, t, blocks, j1, j2);
                const int nk = x.l2 ? I / (2 * kBoxK) : H / kBoxK;
                const int p0 = x.b * kPoolStride;
                if (!x.l2) {
                    wait_count(&xready[x.b], static_cast<u32>(x.rows) * chunks);
                    fence_proxy_global();
                }
                u32 have = 0;
                for (int kb = 0; kb < nk; ++kb) {
                    if (x.l2) {
                        // A down step's K: the gate-up tiles 2 kb and 2 kb + 1
                        // (W4A8), or tile kb (bf16).
                        const u32 need = kW4 ? 3u << (2 * kb) : 1u << kb;
                        if ((have & need) != need) {
                            have = wait_bits(&hready[x.b], need);
                            fence_proxy_global();
                        }
                    }
                    bar_wait(&empty[s], ph ^ 1);
                    unsigned char* bt = smem + kOffB + s * kStageB;
                    if (x.l2) {
                        bar_arrive_expect_tx(&full[s], kTxB2);
                        tma_load_3d(bt, &full[s], &map_h, 0, p0, 2 * kb);
                        if constexpr (kW4) {
                            const long long at = (static_cast<long long>(p0 / 128) * ((I / 128 + 3) / 4) + kb / 2) * 512;
                            bulk_load(smem + kOffSfB + s * 512, p.hpool_sf + at, 512, &full[s]);
                        }
                    } else {
                        bar_arrive_expect_tx(&full[s], kTxB1);
                        tma_load_2d(bt, &full[s], &map_x, kb * kBoxK, p0);
                        if constexpr (kW4) {
                            const long long at = (static_cast<long long>(p0 / 128) * ((H / 128 + 3) / 4) + kb / 4) * 512;
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
        for (int n = 0;; ++n) {
            const int t = claimed(n);
            __syncwarp();
            if (lane == 0) bar_arrive(&task_empty[n % kSlots]);
            if (t < 0) break;
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
                                // The weights' scale column kb * 8 + j (atom
                                // j / 4), the rows' kb * 2 + j / 4.
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
                                    // The gate's (h 0) or up's atom, column j;
                                    // the rows' column kb.
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
        for (int l = max(0, it - 2); l < it; ++l) {
            bar_wait(&tempty[l % 2], (static_cast<u32>(l) / 2) & 1);
        }
    } else {
        // Warps 3-11 first quantize (or copy) the routes' rows of this CTA's
        // blocks into the activation pool, a 1024-column chunk a warp at a
        // time.
        {
            const int qw = warp - 3;
            int k = 0;
            for (int b = static_cast<int>(blockIdx.x); b < blocks; b += G) {
                int e, m;
                block_of(R, b, e, m);
                const int rows = min(R.count[e] - m * kBN, kBN);
                for (int item = 0; item < rows * static_cast<int>(chunks); ++item, ++k) {
                    if (k % kHelpers != qw) continue;
                    const int n = item / static_cast<int>(chunks);
                    const int c0 = (item - n * static_cast<int>(chunks)) * (kChunk / 128);
                    const bf16* src = p.x + static_cast<long long>(route_at(R, R.start[e] + m * kBN + n)) / K * H;
                    const long long prow = static_cast<long long>(b) * kPoolStride + n;
                    const int half = lane / 16, sub = lane % 16;
                    uint4 w[4];
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const int c = c0 + 2 * q + half;
                        w[q] = *reinterpret_cast<const uint4*>(src + c * 128 + sub * 8);
                    }
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const int c = c0 + 2 * q + half;
                        if constexpr (kW4) {
                            // `E4m3Block<128>::load`'s amax over the block's
                            // sixteen lanes, from the words already loaded.
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
                            *reinterpret_cast<unsigned long long*>(p.xpool + prow * H + c * 128 + sub * 8) =
                                blk.codes();
                            if (sub == 0) {
                                p.xpool_sf[atom_offset(static_cast<int>(prow), c, H / 128)] =
                                    static_cast<u8>(blk.e + 127);
                            }
                        } else {
                            *reinterpret_cast<uint4*>(p.xpool + (prow * H + c * 128 + sub * 8) * 2) = w[q];
                        }
                    }
                    fence_proxy_global();
                    __syncwarp();
                    if (lane == 0) add_release(&xready[b], 1);
                }
            }
        }
        if (warp >= 4) {
            // The epilogue: warpgroup wg takes the tasks claimed at n = wg, wg
            // + 2, ... (accumulator wg); thread i holds row i of the tile (a
            // GLU column, or an output column) for the block's kBN routes.
            const int wg = (warp - 4) / 4;
            const int q = warp % 4;
            const int i = q * 32 + lane;
            float* wmax = wmax_all + wg * 4 * kBN;
            for (int n = wg;; n += 2) {
                const int t = claimed(n);
                __syncwarp();
                if (lane == 0) bar_arrive(&task_empty[n % kSlots]);
                if (t < 0) break;
                const Task x = task_of<kBN>(R, t, blocks, j1, j2);
                const u32 acc = static_cast<u32>(wg);
                bar_wait(&tfull[acc], (static_cast<u32>(n) / 2) & 1);
                tc_fence_after();
                const u32 base = tmem + acc * kAcc + ((static_cast<u32>(q) * 32) << 16);
                const long long p0 = static_cast<long long>(x.b) * kPoolStride;
                // Lane l's routes: columns l and 32 + l (and, gate-up, their
                // weights, DeepSeek's).
                long long r_lane[kBN / 32];
                float w_lane[kBN / 32];
#pragma unroll
                for (int h = 0; h < kBN / 32; ++h) {
                    const int c = 32 * h + lane;
                    r_lane[h] = c < x.rows ? route_at(R, x.s0 + c) : 0;
                    w_lane[h] = kFirst && !x.l2 && c < x.rows ? __ldg(&p.weights[r_lane[h]]) : 1.f;
                }
                // Column c's own of a lane's two (a dynamic index stays in
                // registers so).
                auto route_of = [&](int c) -> long long {
                    long long v = r_lane[0];
                    if constexpr (kBN > 32) {
                        if (c >= 32) v = r_lane[1];
                    }
                    return __shfl_sync(0xffffffffu, v, c % 32);
                };
                auto weight_of = [&](int c) -> float {
                    float v = w_lane[0];
                    if constexpr (kBN > 32) {
                        if (c >= 32) v = w_lane[1];
                    }
                    return __shfl_sync(0xffffffffu, v, c % 32);
                };
                const int col = 128 * x.j + i;
                // The live columns alone, a TMEM column at a time: a loop
                // unrolled over every column runs past the instruction cache.
                if (!x.l2) {
                    // Column c's GLU, the gate-up product read as the apart
                    // select lands it, in bf16.
                    auto glu_of = [&](int c) {
                        u32 g, u;
                        tmem_ld1(base + c, g);
                        tmem_ld1(base + kBN + c, u);
                        tmem_wait_ld();
                        float gv = bf16_round(__uint_as_float(g));
                        float uv = bf16_round(__uint_as_float(u));
                        float y;
                        if constexpr (kGlu == 2) {
                            gv = fminf(gv, p.limit);
                            uv = fminf(fmaxf(uv, -p.limit), p.limit);
                            y = (gv / (1.f + expf(-gv))) * uv;
                        } else {
                            y = gv / (1.f + __expf(-gv)) * uv;
                        }
                        if constexpr (kFirst) {
                            y = y * weight_of(c);
                        }
                        return bf16_round(y);
                    };
                    if constexpr (kW4) {
                        // A 128-column block's amax a route: the warps' maxima
                        // met in shared memory; the GLU taken again to land.
                        for (int c = 0; c < x.rows; ++c) {
                            const u32 m = warp_max_u32(__float_as_uint(fabsf(glu_of(c))));
                            if (lane == 0) wmax[q * kBN + c] = __uint_as_float(m);
                        }
                        wg_sync(wg);
                        for (int c = 0; c < x.rows; ++c) {
                            const float v = glu_of(c);
                            const float amax = fmaxf(fmaxf(wmax[c], wmax[kBN + c]),
                                                     fmaxf(wmax[2 * kBN + c], wmax[3 * kBN + c]));
                            const int e = e4m3_block_exp(amax);
                            const float inv = __uint_as_float(static_cast<unsigned>(127 - e) << 23);
                            const float sv = fminf(fmaxf(v * inv, -448.f), 448.f);
                            const unsigned short code = static_cast<unsigned short>(
                                __nv_cvt_float2_to_fp8x2(make_float2(sv, 0.f), __NV_SATFINITE, __NV_E4M3));
                            p.hpool[(p0 + c) * I + col] = static_cast<u8>(code & 0xFF);
                            if (i == 0) {
                                p.hpool_sf[atom_offset(static_cast<int>(p0 + c), x.j, I / 128)] =
                                    static_cast<u8>(e + 127);
                            }
                        }
                    } else {
                        bf16* hp = reinterpret_cast<bf16*>(p.hpool);
                        for (int c = 0; c < x.rows; ++c) hp[(p0 + c) * I + col] = f32_to_bf16(glu_of(c));
                    }
                    tc_fence_before();
                    __syncwarp();
                    if (lane == 0) bar_arrive(&tempty[acc]);
                    fence_proxy_global();
                    wg_sync(wg);
                    if (i == 0) or_release(&hready[x.b], 1u << x.j);
                } else {
                    for (int c = 0; c < x.rows; ++c) {
                        u32 d;
                        tmem_ld1(base + c, d);
                        tmem_wait_ld();
                        const long long r = route_of(c);
                        p.routed[r * H + col] = f32_to_bf16(__uint_as_float(d));
                    }
                    tc_fence_before();
                    __syncwarp();
                    if (lane == 0) bar_arrive(&tempty[acc]);
                    wg_sync(wg);
                    // A narrow fire's tile is folded by the epilogue that lands
                    // its last routes, every token's.
                    if (i == 0) {
                        const u32 had = add_acq_rel(&done[x.j], static_cast<u32>(x.rows));
                        wg_flag[wg] = had + static_cast<u32>(x.rows) == static_cast<u32>(live_routes) ? 1u : 0u;
                    }
                    wg_sync(wg);
                    if (R.tokens <= kFoldInline && wg_flag[wg] != 0) {
                        fence_acquire();
                        for (int t = q; t < R.tokens; t += 4) fold(t, x.j, false);
                    }
                }
            }
        }
        // The fold of a wide fire's tiles, spread over the CTAs once each
        // tile's routes have landed (a narrow fire's ran in the down epilogue).
        if (R.tokens > kFoldInline) {
            const int fw = warp - 3;
            const int items = R.tokens * j2;
            for (int item = static_cast<int>(blockIdx.x) + G * fw; item < items; item += kHelpers * G) {
                const int t = item / j2, j = item - t * j2;
                fold(t, j, true);
                __syncwarp();
            }
        }
    }

    // This CTA's work is done: the next launch's CTAs may take its SM.
    pdl_launch();
    // The last CTA out zeroes the flags for the next launch.
    tc_fence_before();
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        *flag = atomicAdd(&p.sync[ws_finish()], 1u) == static_cast<u32>(G - 1) ? 1u : 0u;
    }
    __syncthreads();
    if (*flag != 0) {
        __threadfence();
        for (int b = static_cast<int>(threadIdx.x); b < blocks; b += kThreads) {
            xready[b] = 0;
            hready[b] = 0;
        }
        for (int j = static_cast<int>(threadIdx.x); j < j2; j += kThreads) done[j] = 0;
        if (threadIdx.x == 0) {
            p.sync[ws_next()] = 0;
            p.sync[ws_finish()] = 0;
        }
    }
    if (warp == 2) {
        tc_fence_after();
        tmem_dealloc(tmem, kCols);
    }
}

}  // namespace pie::linear

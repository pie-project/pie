#include "prim/device.cuh"
#include "prim/mma_sm100.cuh"
#include "elemwise/hc.cuh"
#include "linear/act_quant.cuh"

// DeepSeek-V4's hyper-connection seam on tcgen05 in one launch: a
// sublayer's fold (`hc_fold`: each stream j the post gate times the
// sublayer's output plus the combine's mix of the streams) and the next
// sublayer's whole pre-mix (`hc::pre`: the stream norm and the 24-mix
// projection, the gates and the Sinkhorn walk, the pre-mixed row and its
// RMS norm). The shape is DeepGEMM's mega mHC (`sm100_mega_mhc`, DeepSeek,
// MIT), in this codebase's primitives:
//
//   - A task is 128 rows by a split of the row: K-blocks of 32 columns of
//     each of the four streams. Warp 0 hands tasks out and brings a
//     K-block's x, its four streams (one 3D box, stream-major) and the f32
//     mix weights' box in by TMA, the activations 64-byte swizzled; with a
//     task's first K-block its rows' post and combine gates.
//   - The fold (warps 4-7, a 16-byte run of columns each, its lanes rows
//     32 apart) folds in registers in `hc_fold`'s order and lands the folded
//     bf16 back over the streams' tile, where it is both the MMA's A and the
//     box warp 14 stores out.
//   - The projection is the dense bf16 MMA at M = 128, N = 48 (warp 1):
//     warps 2-3 cut the f32 weights into a high and a low bf16 half (rows
//     0-23 and 24-47 of B), as `hc_fn_split` cuts them, and the two halves
//     of a mix are summed out of TMEM. A row's sum of squares comes off the
//     folded values.
//   - A split's 24 dots and sum of squares land as one 128-byte row of a
//     partial slab; the fold epilogue and the store warp (its boxes landed)
//     each count the split in for its 128-row block with a release add.
//     Once every split of a block is in, its rows are finished: a 128-thread
//     finisher (warps 8-11; warps 4-7 and 0-3 once out of tasks) takes a
//     row, sums its splits' pre logits, pre-mixes the folded row back out of
//     L2, norms it and lands it (with EMIT its e4m3 1x128 operand too); the
//     Sinkhorn warps (12-13) take one or two rows a warp, sum their splits
//     across the lanes, and land the post gates and the combine, each lane
//     walking the whole 4x4 in its registers.
//   - A block's first task is its own, so its first weight boxes go to L2
//     before `pdl_wait`, under the producer kernel's tail.
//
// A staged window (`win`: its rows and their offset, written at replay) is
// known only on the device, so each block rebases its row maps there before
// any copy: a row past the window is neither read (TMA fills zeros) nor
// written (the store drops it). A block's first task is its own; the rest,
// the finisher rows and the Sinkhorn rows come off counters in order. Every
// block calls `pdl_launch` on entry, so no dependent takes an SM before all
// blocks are resident, and a finisher's wait is only on running blocks.
// The counters and the per-block arrivals live in a stream's zeroed slab,
// and the last block to retire puts every one back to zero: a graph
// replays its launch on the same zeros, and no count from one launch is
// read by another (DeepGEMM tags its split barriers with an epoch for the
// same reason, the fix of its graph replay).
//
// The fold is `hc_fold`'s bit for bit. The mixes are the projection's sums
// in another order, the weights to 16 significant bits (the tensor-core
// prenorm projection's precision), so the gates, the combine and the
// normed row match the unfused composition to rounding, not bit for bit.

namespace pie::elemwise {

constexpr int kMegaRows = 128;
constexpr int kMegaCols = 32;
constexpr int kMegaStages = 3;
constexpr int kMegaStreams = 4;
constexpr int kMegaMix = 24;
constexpr int kMegaN = 2 * kMegaMix;
// A split's partial row: the 24 dots, the sum of squares, padding to 128 bytes.
constexpr int kMegaPart = 32;
constexpr int kMegaThreads = 15 * 32;
// The TMEM columns of one accumulator (N = 48, two of them in flight).
constexpr unsigned kMegaAccCols = 64;

constexpr unsigned kMegaResBytes = kMegaStreams * kMegaRows * 64;
constexpr unsigned kMegaXBytes = kMegaRows * 64;
constexpr unsigned kMegaFnBytes = kMegaStreams * kMegaMix * kMegaCols * 4;
constexpr unsigned kMegaBBytes = kMegaStreams * kMegaN * 64;
// A task's gates: 128 rows of the four post and the 16 combine gates.
constexpr unsigned kMegaPostBytes = kMegaRows * kMegaStreams * 4;
constexpr unsigned kMegaCombBytes = kMegaRows * kMegaStreams * kMegaStreams * 4;
constexpr unsigned kMegaGateBytes = kMegaPostBytes + kMegaCombBytes;
constexpr unsigned kMegaOffX = kMegaStages * kMegaResBytes;
constexpr unsigned kMegaOffFn = kMegaOffX + kMegaStages * kMegaXBytes;
constexpr unsigned kMegaOffB = kMegaOffFn + kMegaStages * kMegaFnBytes;
constexpr unsigned kMegaOffGate = kMegaOffB + kMegaStages * kMegaBBytes;
constexpr unsigned kMegaOffMisc = kMegaOffGate + 2 * kMegaGateBytes;
constexpr unsigned kMegaMiscBytes = 8192;
constexpr unsigned kMegaStageTx = kMegaResBytes + kMegaXBytes + kMegaFnBytes;

// The counters' words: the task, finisher-row and Sinkhorn-row tickets, the
// retired blocks, then a 128-row block's arrivals (two a split) from word 8.
constexpr int kSyncTasks = 0;
constexpr int kSyncNormRows = 1;
constexpr int kSyncSinkRows = 2;
constexpr int kSyncExit = 3;
constexpr int kSyncBlocks = 8;

// A stage's task flags.
constexpr int kMegaFirst = 1;
constexpr int kMegaLast = 2;
constexpr int kMegaDone = 4;

// The finisher warpgroups, and the row maps a block rebases.
constexpr int kMegaNorms = 3;
constexpr int kMegaMaps = 5;

struct HcMegaParams {
    const bf16* x;
    const bf16* residual;
    const float* post_in;
    const float* comb_in;
    bf16* folded;
    const float* scale;
    const float* base;
    float* post_out;
    float* comb_out;
    const bf16* weight;
    bf16* y;
    u8* codes;
    u8* atoms;
    float* part;
    unsigned* sync;
    tc5::TmaMap* maps;
    const u32* win;
    int rows;
    int splits;
    int sinkhorn;
    float mix_eps;
    float gate_eps;
    float alpha;
    float norm_eps;
};

struct HcMegaMisc {
    tc5::TmaMap maps[kMegaMaps];
    unsigned long long full[kMegaStages];
    unsigned long long posted[kMegaStages];
    unsigned long long conv[kMegaStages];
    unsigned long long fold[kMegaStages];
    unsigned long long empty[kMegaStages];
    unsigned long long tfull[2];
    unsigned long long tempty[2];
    int meta[kMegaStages][4];
    unsigned tmem;
    int row[kMegaNorms];
    float sqs[4][kMegaRows];
    float red[kMegaNorms][16][8];
    float pre[kMegaNorms][4];
    float wsum[kMegaNorms][4];
};

static_assert(sizeof(HcMegaMisc) <= kMegaMiscBytes, "the barriers and scratch fit their region");

__host__ __device__ constexpr unsigned hc_mega_smem() { return kMegaOffMisc + kMegaMiscBytes + 1024; }

// Byte offset of 16-byte chunk `c` of row `r` in a 64-byte-swizzled tile.
__device__ __forceinline__ unsigned hc_sw64(int r, int c) {
    return static_cast<unsigned>(r * 64 + ((c ^ ((r >> 1) & 3)) << 4));
}

// Whole-warp: waits until block `mb`'s splits are all in, two arrivals
// each. Every lane reads and the warp leaves on one vote, so it leaves
// converged: a lane alone spinning leaves the warp to run the shuffles
// after it on the divergent path, an order of magnitude slower. The spin
// reads relaxed (an acquire each time would invalidate L1 under the fold's
// warps) and acquires once it has seen the count.
__device__ __forceinline__ void hc_mega_wait(const HcMegaParams& p, int mb) {
    const unsigned* at = p.sync + kSyncBlocks + mb;
    const unsigned want = 2u * static_cast<unsigned>(p.splits);
    while (!__all_sync(0xffffffffu, ld_relaxed_gpu(at) >= want)) __nanosleep(64);
    static_cast<void>(ld_acquire_gpu(at));
}

// The Sinkhorn warp: a row a group of G lanes (two rows a round at G = 16,
// one at 32), lane g of a group summing columns 4-24 (post, combine, the sum
// of squares) of the splits g and g + G, the group folded by butterfly.
// Every lane then holds the row's whole combine and walks it in its own
// registers; lane l lands entry l, lanes l < 4 the post gates.
template <int H, int G>
__device__ __forceinline__ void hc_mega_sink(const HcMegaParams& p, int lane, int live, int base) {
    // A row a group of G lanes: two rows a warp where the splits are few
    // enough that each lane holds at most two, one where they are more.
    const int S = p.splits;
    const int h = lane / G;
    const int g = lane % G;
    const int l = lane & 15;
    const float post_s = p.scale[1];
    const float comb_s = p.scale[2];
    const float post_b = p.base[kMegaStreams + (l & 3)];
    float comb_b[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) comb_b[i] = p.base[2 * kMegaStreams + i];
    const unsigned per = 32u / static_cast<unsigned>(G);
    while (true) {
        int r0 = 0;
        if (lane == 0) r0 = static_cast<int>(atomicAdd(p.sync + kSyncSinkRows, per));
        r0 = __shfl_sync(0xffffffffu, r0, 0);
        if (r0 >= live) return;
        hc_mega_wait(p, r0 >> 7);
        if (per > 1 && r0 + 1 < live && ((r0 + 1) >> 7) != (r0 >> 7)) hc_mega_wait(p, (r0 + 1) >> 7);
        const int r = r0 + h;
        const bool ok = r < live;
        const int rc = ok ? r : r0;
        const float* pr = p.part + (static_cast<long long>(rc >> 7) * S * kMegaRows + (rc & 127)) * kMegaPart + 4;
        // Columns 4-27 as six float4 of the splits g and g + G, both in
        // flight, then the group's lanes folded by butterfly.
        float acc[24];
        {
            float4 v[2][6];
#pragma unroll
            for (int u = 0; u < 2; ++u) {
                const int s = g + G * u;
                const float4* q = reinterpret_cast<const float4*>(pr + static_cast<long long>(s) * kMegaRows * kMegaPart);
#pragma unroll
                for (int i = 0; i < 6; ++i) v[u][i] = s < S ? __ldcg(q + i) : make_float4(0.f, 0.f, 0.f, 0.f);
            }
#pragma unroll
            for (int i = 0; i < 6; ++i) {
                acc[4 * i] = v[0][i].x + v[1][i].x;
                acc[4 * i + 1] = v[0][i].y + v[1][i].y;
                acc[4 * i + 2] = v[0][i].z + v[1][i].z;
                acc[4 * i + 3] = v[0][i].w + v[1][i].w;
            }
        }
#pragma unroll
        for (int i = 0; i < 21; ++i) {
#pragma unroll
            for (int o = 1; o < 16; o <<= 1) acc[i] += __shfl_xor_sync(0xffffffffu, acc[i], o);
            if (G == 32) acc[i] += __shfl_xor_sync(0xffffffffu, acc[i], 16);
        }
        const float sc = rsqrtf(acc[20] / static_cast<float>(kMegaStreams * H) + p.mix_eps);
        float vp = 0.f;
#pragma unroll
        for (int i = 0; i < 4; ++i) vp = i == l ? acc[i] : vp;
        const long long row = base + rc;
        if (ok && g < 4) {
            const float pl = vp * sc * post_s + post_b;
            p.post_out[row * kMegaStreams + l] = 1.f / (1.f + expf(-pl)) * p.alpha;
        }
        float c[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) c[i] = acc[4 + i] * sc * comb_s + comb_b[i];
        hc_sinkhorn4_regs(c, p.gate_eps, p.sinkhorn);
        float mine = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i) mine = i == l ? c[i] : mine;
        if (ok && g < 16) p.comb_out[row * kMegaStreams * kMegaStreams + l] = mine;
    }
}

// A finisher warpgroup (`t` its thread, `wg` its scratch, `bar` its
// barrier): a row a round, its pre gates from its splits, the pre-mixed row
// (eight columns a thread, H / 1024 runs) rounded to bf16 and normed.
template <int H, bool EMIT>
__device__ __forceinline__ void hc_mega_norm(const HcMegaParams& p, HcMegaMisc& m, int wg, unsigned bar,
                                             int t, int live, int base) {
    constexpr int RUNS = H / 1024;
    const int lane = t & 31;
    const int w = t >> 5;
    const int S = p.splits;
    // What no row changes, read before the first wait.
    uint4 wv[RUNS];
#pragma unroll
    for (int i = 0; i < RUNS; ++i) wv[i] = *reinterpret_cast<const uint4*>(p.weight + (i * 128 + t) * 8);
    const float pre_s = p.scale[0];
    const float pre_b = p.base[lane & 3];
    while (true) {
        if (t == 0) m.row[wg] = static_cast<int>(atomicAdd(p.sync + kSyncNormRows, 1u));
        named_bar_sync(bar, 128);
        const int r = m.row[wg];
        if (r >= live) return;
        const int mb = r >> 7;
        if (w == 0) hc_mega_wait(p, mb);
        named_bar_sync(bar, 128);
        // The folded row's first runs in flight under the splits' sums: the
        // pre-mix's reads do not wait on its gates.
        constexpr int EARLY = RUNS < 4 ? RUNS : 4;
        const long long row = base + r;
        const bf16* src = p.folded + row * kMegaStreams * H;
        uint4 early[EARLY][kMegaStreams];
#pragma unroll
        for (int i = 0; i < EARLY; ++i)
#pragma unroll
            for (int j = 0; j < kMegaStreams; ++j)
                early[i][j] = __ldcg(reinterpret_cast<const uint4*>(src + static_cast<long long>(j) * H + (i * 128 + t) * 8));
        {
            // The pre logits and the sum of squares: column k of the splits
            // g, g + 16, g + 32, g + 48, then the 16 groups in order.
            const int k = t & 7;
            const int g = t >> 3;
            if (k < 5) {
                const int col = k < 4 ? k : kMegaMix;
                const float* pr = p.part + (static_cast<long long>(mb) * S * kMegaRows + (r & 127)) * kMegaPart + col;
                float v[4];
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const int s = g + 16 * u;
                    v[u] = s < S ? __ldcg(pr + static_cast<long long>(s) * kMegaRows * kMegaPart) : 0.f;
                }
                m.red[wg][g][k] = ((v[0] + v[1]) + v[2]) + v[3];
            }
        }
        named_bar_sync(bar, 128);
        if (w == 0) {
            float v = 0.f;
            if (lane < 5) {
#pragma unroll
                for (int g = 0; g < 16; ++g) v += m.red[wg][g][lane];
            }
            const float sq = __shfl_sync(0xffffffffu, v, 4);
            if (lane < 4) {
                const float mix = v * rsqrtf(sq / static_cast<float>(kMegaStreams * H) + p.mix_eps);
                const float logit = mix * pre_s + pre_b;
                m.pre[wg][lane] = 1.f / (1.f + expf(-logit)) + p.gate_eps;
            }
        }
        named_bar_sync(bar, 128);
        float pre[kMegaStreams];
#pragma unroll
        for (int k = 0; k < kMegaStreams; ++k) pre[k] = m.pre[wg][k];
        uint4 xb[RUNS];
        float local = 0.f;
#pragma unroll
        for (int i = 0; i < RUNS; ++i) {
            const int q = i * 128 + t;
            uint4 rw[kMegaStreams];
#pragma unroll
            for (int j = 0; j < kMegaStreams; ++j)
                rw[j] = i < EARLY ? early[i < EARLY ? i : 0][j]
                                  : __ldcg(reinterpret_cast<const uint4*>(src + static_cast<long long>(j) * H + q * 8));
            float acc[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[e] = 0.f;
#pragma unroll
            for (int j = 0; j < kMegaStreams; ++j) {
                const bf16* r8 = reinterpret_cast<const bf16*>(&rw[j]);
#pragma unroll
                for (int e = 0; e < 8; ++e) acc[e] += pre[j] * bf16_to_f32(r8[e]);
            }
            bf16* o8 = reinterpret_cast<bf16*>(&xb[i]);
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                o8[e] = f32_to_bf16(acc[e]);
                const float v = bf16_to_f32(o8[e]);
                local += v * v;
            }
        }
        local = ::pie::warp_sum(local);
        if (lane == 0) m.wsum[wg][w] = local;
        named_bar_sync(bar, 128);
        const float total = ((m.wsum[wg][0] + m.wsum[wg][1]) + m.wsum[wg][2]) + m.wsum[wg][3];
        const float inv_rms = rsqrtf(total / static_cast<float>(H) + p.norm_eps);
        bf16* yr = p.y + row * H;
#pragma unroll
        for (int i = 0; i < RUNS; ++i) {
            const int q = i * 128 + t;
            const bf16* w8 = reinterpret_cast<const bf16*>(&wv[i]);
            const bf16* x8 = reinterpret_cast<const bf16*>(&xb[i]);
            float f8[8];
            uint4 yw;
            bf16* y8 = reinterpret_cast<bf16*>(&yw);
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                f8[e] = bf16_to_f32(x8[e]) * inv_rms * bf16_to_f32(w8[e]);
                y8[e] = f32_to_bf16(f8[e]);
            }
            *reinterpret_cast<uint4*>(yr + q * 8) = yw;
            if constexpr (EMIT) {
                // A 128-wide block is 16 adjacent lanes' runs.
                ::pie::linear::E4m3Block<128> eb;
                eb.hold(f8);
                eb.store(p.codes, p.atoms, static_cast<int>(row), q / 16, H, 0, lane);
            }
        }
        named_bar_sync(bar, 128);
    }
}

// The fold and the next pre-mix, `H` wide streams, four of them. Launched
// over at most one block an SM (its shared memory), 480 threads. The maps:
// the f32 mix weights `[4][24][H]` (boxes of 32 x 24 x 4); x `[rows][H]`
// (32 x 128), the streams and the folded streams `[4][rows][H]`
// (32 x 128 x 4), 64-byte swizzled; the f32 post and combine gates
// `[rows][4]` and `[rows][16]` (4 x 128, 16 x 128).
template <int H, bool EMIT>
__global__ void __launch_bounds__(kMegaThreads, 1) hc_fold_pre_tc5(
    const __grid_constant__ tc5::TmaMap map_fn,
    const __grid_constant__ tc5::TmaMap map_x,
    const __grid_constant__ tc5::TmaMap map_res,
    const __grid_constant__ tc5::TmaMap map_out,
    const __grid_constant__ tc5::TmaMap map_post,
    const __grid_constant__ tc5::TmaMap map_comb,
    const __grid_constant__ HcMegaParams p)
{
    using namespace tc5;
    static_assert(H % 1024 == 0, "a finisher thread takes whole runs of eight, 128 a pass");
    constexpr int KB = H / kMegaCols;
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    const unsigned raw = smem_addr(smem_raw);
    unsigned char* sm = smem_raw + (((raw + 1023u) & ~1023u) - raw);
    HcMegaMisc& m = *reinterpret_cast<HcMegaMisc*>(sm + kMegaOffMisc);
    const int warp = static_cast<int>(threadIdx.x >> 5);
    const int lane = static_cast<int>(threadIdx.x & 31);

    if (threadIdx.x == 0) {
#pragma unroll
        for (int s = 0; s < kMegaStages; ++s) {
            bar_init(&m.full[s], 1);
            bar_init(&m.posted[s], 1);
            bar_init(&m.conv[s], 2);
            bar_init(&m.fold[s], 4);
            bar_init(&m.empty[s], 2);
        }
#pragma unroll
        for (int b = 0; b < 2; ++b) {
            bar_init(&m.tfull[b], 1);
            bar_init(&m.tempty[b], 4);
        }
        bar_init_fence();
        tma_prefetch(&map_fn);
        // The block's own task's first weight boxes into L2 while the
        // producer kernel finishes: weights do not wait on it.
        const int t = static_cast<int>(blockIdx.x);
        if (t < (p.rows + kMegaRows - 1) / kMegaRows * p.splits) {
            const int per = KB / p.splits;
            const int sp = t % p.splits;
            const int kb0 = sp * per + min(sp, KB % p.splits);
            const int n = per + (sp < KB % p.splits ? 1 : 0);
            for (int i = 0; i < n && i < kMegaStages; ++i) tma_prefetch_3d(&map_fn, (kb0 + i) * kMegaCols, 0, 0);
        }
    }
    if (warp == 1) {
        tmem_alloc(&m.tmem, 2 * kMegaAccCols);
        tmem_relinquish();
    }
    // A programmatic dependent (`kernel.pdl`) touches nothing until its
    // producer is done; its own dependents may launch from here.
    pdl_wait();
    pdl_launch();

    const int live = p.win != nullptr ? min(static_cast<int>(p.win[0]), p.rows) : p.rows;
    const int base = p.win != nullptr ? static_cast<int>(p.win[1]) : 0;
    // The row maps, rebased at the window where there is one.
    const TmaMap* mx = &map_x;
    const TmaMap* mr = &map_res;
    const TmaMap* mo = &map_out;
    const TmaMap* mp = &map_post;
    const TmaMap* mc = &map_comb;
    if (p.win != nullptr) {
        TmaMap* slot = p.maps + kMegaMaps * blockIdx.x;
        if (warp == 0) {
            const u64* from[kMegaMaps] = {map_x.raw, map_res.raw, map_out.raw, map_post.raw, map_comb.raw};
            const void* at[kMegaMaps] = {
                p.x + static_cast<long long>(base) * H,
                p.residual + static_cast<long long>(base) * kMegaStreams * H,
                p.folded + static_cast<long long>(base) * kMegaStreams * H,
                p.post_in + static_cast<long long>(base) * kMegaStreams,
                p.comb_in + static_cast<long long>(base) * kMegaStreams * kMegaStreams,
            };
#pragma unroll
            for (int k = 0; k < kMegaMaps; ++k)
                if (lane < 16) m.maps[k].raw[lane] = from[k][lane];
            __syncwarp();
            if (lane == 0) {
                const unsigned extent = static_cast<unsigned>(max(live, 1));
#pragma unroll
                for (int k = 0; k < kMegaMaps; ++k) {
                    tmap_set_address(&m.maps[k], reinterpret_cast<u64>(at[k]));
                    tmap_set_dim<1>(&m.maps[k], extent);
                }
            }
            __syncwarp();
#pragma unroll
            for (int k = 0; k < kMegaMaps; ++k) tmap_publish(slot + k, &m.maps[k]);
        }
        mx = slot;
        mr = slot + 1;
        mo = slot + 2;
        mp = slot + 3;
        mc = slot + 4;
    }
    tc_fence_before();
    __syncthreads();
    tc_fence_after();
    const u32 tmem = m.tmem;

    const int S = p.splits;
    const int tasks = (live + kMegaRows - 1) / kMegaRows * S;
    // Split `sp`'s K-blocks: KB / S each, the first KB % S one more.
    auto split = [&](int sp, int& kb0, int& n) {
        const int per = KB / S;
        const int longer = KB % S;
        kb0 = sp * per + min(sp, longer);
        n = per + (sp < longer ? 1 : 0);
    };

    if (warp == 0) {
        // Tasks in order: the block's own, then tickets past the grid's, the
        // next one's taken while this one loads. Each K-block's stage posted
        // (its task and flags) once the MMA and the store have released it,
        // then its boxes loaded; a task's gates with its first.
        if (p.win != nullptr && lane == 0) {
            tmap_acquire(mx);
            tmap_acquire(mr);
            tmap_acquire(mp);
            tmap_acquire(mc);
        }
        int it = 0;
        int tq = 0;
        int t = static_cast<int>(blockIdx.x);
        while (true) {
            unsigned next = 0;
            if (lane == 0 && t < tasks) next = atomicAdd(p.sync + kSyncTasks, 1u);
            if (t >= tasks) {
                const int s = it % kMegaStages;
                bar_wait(&m.empty[s], ((it / kMegaStages) & 1) ^ 1);
                if (lane == 0) {
                    m.meta[s][2] = kMegaDone;
                    bar_arrive(&m.posted[s]);
                    bar_arrive(&m.full[s]);
                }
                break;
            }
            int kb0 = 0, n = 0;
            split(t % S, kb0, n);
            const int row0 = t / S * kMegaRows;
            unsigned char* gate = sm + kMegaOffGate + (tq & 1) * kMegaGateBytes;
            for (int i = 0; i < n; ++i, ++it) {
                const int s = it % kMegaStages;
                bar_wait(&m.empty[s], ((it / kMegaStages) & 1) ^ 1);
                if (lane == 0) {
                    m.meta[s][0] = t;
                    m.meta[s][1] = kb0 + i;
                    m.meta[s][2] = (i == 0 ? kMegaFirst : 0) | (i == n - 1 ? kMegaLast : 0);
                    bar_arrive(&m.posted[s]);
                    bar_arrive_expect_tx(&m.full[s], kMegaStageTx + (i == 0 ? kMegaGateBytes : 0));
                    const int col = (kb0 + i) * kMegaCols;
                    if (i == 0) {
                        tma_load_2d(gate, &m.full[s], mp, 0, row0);
                        tma_load_2d(gate + kMegaPostBytes, &m.full[s], mc, 0, row0);
                    }
                    tma_load_3d(sm + s * kMegaResBytes, &m.full[s], mr, col, row0, 0);
                    tma_load_2d(sm + kMegaOffX + s * kMegaXBytes, &m.full[s], mx, col, row0);
                    tma_load_3d(sm + kMegaOffFn + s * kMegaFnBytes, &m.full[s], &map_fn, col, 0, 0);
                }
                __syncwarp();
            }
            ++tq;
            t = static_cast<int>(gridDim.x + __shfl_sync(0xffffffffu, next, 0));
        }
    } else if (warp == 1) {
        // The MMAs: a K-block's four streams, two K steps each, into the
        // task's accumulator; its stage released, a task's last committed.
        constexpr u32 idesc = idesc_dense(kF16Bf16, kF16Bf16, kMegaRows, kMegaN);
        int tc = 0;
        for (int it = 0;; ++it) {
            const int s = it % kMegaStages;
            const u32 ph = (it / kMegaStages) & 1;
            bar_wait(&m.full[s], ph);
            const int flags = m.meta[s][2];
            if (flags & kMegaDone) break;
            bar_wait(&m.conv[s], ph);
            bar_wait(&m.fold[s], ph);
            const int buf = tc & 1;
            if (flags & kMegaFirst) bar_wait(&m.tempty[buf], ((tc >> 1) & 1) ^ 1);
            tc_fence_after();
            if (elect_one()) {
                const u32 d = tmem + static_cast<u32>(buf) * kMegaAccCols;
                const u32 a0 = smem_addr(sm + s * kMegaResBytes);
                const u32 b0 = smem_addr(sm + kMegaOffB + s * kMegaBBytes);
#pragma unroll
                for (int j = 0; j < kMegaStreams; ++j) {
                    const u64 ad = sdesc_sw64(a0 + j * (kMegaRows * 64));
                    const u64 bd = sdesc_sw64(b0 + j * (kMegaN * 64));
#pragma unroll
                    for (int k = 0; k < 2; ++k) {
                        const bool acc = !(flags & kMegaFirst) || j > 0 || k > 0;
                        mma_f16(d, ad + 2 * k, bd + 2 * k, idesc, acc);
                    }
                }
                tc_commit(&m.empty[s]);
                if (flags & kMegaLast) tc_commit(&m.tfull[buf]);
            }
            __syncwarp();
            if (flags & kMegaLast) ++tc;
        }
    } else if (warp < 4) {
        // The weights' box cut into its high and low bf16 halves, B's rows
        // o and 24 + o, as `hc_fn_split` cuts them.
        const int ct = static_cast<int>(threadIdx.x) - 64;
        for (int it = 0;; ++it) {
            const int s = it % kMegaStages;
            bar_wait(&m.full[s], (it / kMegaStages) & 1);
            if (m.meta[s][2] & kMegaDone) break;
            const float* src = reinterpret_cast<const float*>(sm + kMegaOffFn + s * kMegaFnBytes);
            unsigned char* dst = sm + kMegaOffB + s * kMegaBBytes;
#pragma unroll
            for (int i = ct; i < kMegaStreams * kMegaMix * 4; i += 64) {
                const int j = i / (kMegaMix * 4);
                const int o = (i >> 2) % kMegaMix;
                const int c = i & 3;
                const float* w = src + (j * kMegaMix + o) * kMegaCols + 8 * c;
                const float4 w0 = *reinterpret_cast<const float4*>(w);
                const float4 w1 = *reinterpret_cast<const float4*>(w + 4);
                const float f[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
                uint4 hi, lo;
                bf16* h8 = reinterpret_cast<bf16*>(&hi);
                bf16* l8 = reinterpret_cast<bf16*>(&lo);
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    h8[e] = f32_to_bf16(f[e]);
                    l8[e] = f32_to_bf16(f[e] - bf16_to_f32(h8[e]));
                }
                unsigned char* bj = dst + j * (kMegaN * 64);
                *reinterpret_cast<uint4*>(bj + hc_sw64(o, c)) = hi;
                *reinterpret_cast<uint4*>(bj + hc_sw64(kMegaMix + o, c)) = lo;
            }
            fence_async_shared();
            __syncwarp();
            if (lane == 0) bar_arrive(&m.conv[s]);
        }
    } else if (warp < 8) {
        // The fold in `hc_fold`'s order: warp c the K-block's 16-byte run c,
        // lane l rows l, l + 32, l + 64, l + 96 (a decode's few rows on every
        // warp), landed over the stream tile for the MMA and the store. Its
        // epilogue thread is row r of TMEM's lanes.
        const int c = warp - 4;
        const int r = static_cast<int>(threadIdx.x) - 128;
        int tc = 0;
        float sq[4] = {0.f, 0.f, 0.f, 0.f};
        for (int it = 0;; ++it) {
            const int s = it % kMegaStages;
            bar_wait(&m.full[s], (it / kMegaStages) & 1);
            const int flags = m.meta[s][2];
            if (flags & kMegaDone) break;
            const int t = m.meta[s][0];
            const int mb = t / S;
            const int lim = live - mb * kMegaRows;
            const unsigned char* gate = sm + kMegaOffGate + (tc & 1) * kMegaGateBytes;
            if (flags & kMegaFirst) {
#pragma unroll
                for (int k = 0; k < 4; ++k) sq[k] = 0.f;
            }
            unsigned char* xt = sm + kMegaOffX + s * kMegaXBytes;
            unsigned char* rt = sm + s * kMegaResBytes;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int row = lane + 32 * k;
                if (row >= lim) continue;
                const float4 pw = *reinterpret_cast<const float4*>(gate + row * 16);
                const float post[4] = {pw.x, pw.y, pw.z, pw.w};
                float comb[16];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float4 cw = *reinterpret_cast<const float4*>(gate + kMegaPostBytes + row * 64 + 16 * i);
                    comb[4 * i] = cw.x;
                    comb[4 * i + 1] = cw.y;
                    comb[4 * i + 2] = cw.z;
                    comb[4 * i + 3] = cw.w;
                }
                const unsigned at = hc_sw64(row, c);
                const uint4 xw = *reinterpret_cast<const uint4*>(xt + at);
                uint4 rw[kMegaStreams];
#pragma unroll
                for (int j = 0; j < kMegaStreams; ++j)
                    rw[j] = *reinterpret_cast<const uint4*>(rt + j * (kMegaRows * 64) + at);
                const bf16* x8 = reinterpret_cast<const bf16*>(&xw);
                float xf[8];
                float rf[kMegaStreams][8];
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    xf[e] = bf16_to_f32(x8[e]);
#pragma unroll
                    for (int j = 0; j < kMegaStreams; ++j)
                        rf[j][e] = bf16_to_f32(reinterpret_cast<const bf16*>(&rw[j])[e]);
                }
#pragma unroll
                for (int j = 0; j < kMegaStreams; ++j) {
                    uint4 ow;
                    bf16* o8 = reinterpret_cast<bf16*>(&ow);
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        // Each product rounded, then summed: `hc_fold`
                        // under NVRTC's `--fmad=false`, not a fused
                        // multiply-add.
                        float acc = __fmul_rn(post[j], xf[e]);
#pragma unroll
                        for (int i = 0; i < kMegaStreams; ++i)
                            acc = __fadd_rn(acc, __fmul_rn(comb[i * 4 + j], rf[i][e]));
                        o8[e] = f32_to_bf16(acc);
                        const float v = bf16_to_f32(o8[e]);
                        sq[k] = fmaf(v, v, sq[k]);
                    }
                    *reinterpret_cast<uint4*>(rt + j * (kMegaRows * 64) + at) = ow;
                }
            }
            fence_async_shared();
            __syncwarp();
            if (lane == 0) bar_arrive(&m.fold[s]);
            if (flags & kMegaLast) {
                // The runs' sums of squares met per row, the task's dots out
                // of TMEM with the high and low halves summed, its partial
                // row landed and the split counted in.
#pragma unroll
                for (int k = 0; k < 4; ++k) m.sqs[c][lane + 32 * k] = sq[k];
                named_bar_sync(1, 128);
                const float rsq = ((m.sqs[0][r] + m.sqs[1][r]) + m.sqs[2][r]) + m.sqs[3][r];
                const int buf = tc & 1;
                bar_wait(&m.tfull[buf], (tc >> 1) & 1);
                tc_fence_after();
                u32 v0[32], v1[32];
                const u32 ta = tmem + (static_cast<u32>(warp & 3) * 32u << 16) + static_cast<u32>(buf) * kMegaAccCols;
                tmem_ld32(ta, v0);
                tmem_ld32(ta + 32, v1);
                tmem_wait_ld();
                tc_fence_before();
                __syncwarp();
                if (lane == 0) bar_arrive(&m.tempty[buf]);
                if (r < lim) {
                    float mix[kMegaMix];
#pragma unroll
                    for (int o = 0; o < kMegaMix; ++o) {
                        const u32 lo = o + kMegaMix < 32 ? v0[o + kMegaMix] : v1[o + kMegaMix - 32];
                        mix[o] = __uint_as_float(v0[o]) + __uint_as_float(lo);
                    }
                    float* pr = p.part + (static_cast<long long>(t) * kMegaRows + r) * kMegaPart;
#pragma unroll
                    for (int o = 0; o < kMegaMix; o += 4)
                        *reinterpret_cast<float4*>(pr + o) = make_float4(mix[o], mix[o + 1], mix[o + 2], mix[o + 3]);
                    pr[kMegaMix] = rsq;
                }
                // The barrier orders every thread's row before the one release,
                // which is cumulative: the finisher's acquire sees them all.
                named_bar_sync(1, 128);
                if (r == 0) red_add_release_gpu(p.sync + kSyncBlocks + mb, 1u);
                ++tc;
            }
        }
    } else if (warp < 12) {
        hc_mega_norm<H, EMIT>(p, m, 0, 2, static_cast<int>(threadIdx.x) - 256, live, base);
    } else if (warp < 14) {
        if (p.splits <= 32) {
            hc_mega_sink<H, 16>(p, lane, live, base);
        } else {
            hc_mega_sink<H, 32>(p, lane, live, base);
        }
    } else {
        // The store: a K-block's folded tile out once the fold has landed
        // it, the stage released once the copy has read it, and once a
        // task's boxes have landed in memory the split counted in.
        if (p.win != nullptr && lane == 0) tmap_acquire(mo);
        for (int it = 0;; ++it) {
            const int s = it % kMegaStages;
            const u32 ph = (it / kMegaStages) & 1;
            bar_wait(&m.posted[s], ph);
            const int flags = m.meta[s][2];
            if (flags & kMegaDone) break;
            const int t = m.meta[s][0];
            const int kb = m.meta[s][1];
            bar_wait(&m.fold[s], ph);
            if (lane == 0) {
                tma_store_3d(mo, sm + s * kMegaResBytes, kb * kMegaCols, t / S * kMegaRows, 0);
                bulk_commit();
                bulk_wait_read<0>();
                bar_arrive(&m.empty[s]);
                if (flags & kMegaLast) {
                    bulk_wait_all();
                    fence_async_global();
                    red_add_release_gpu(p.sync + kSyncBlocks + t / S, 1u);
                }
            }
            __syncwarp();
        }
    }

    // Out of tasks, warps 4-7 and 0-3 finish rows too.
    if (warp >= 4 && warp < 8) {
        hc_mega_norm<H, EMIT>(p, m, 1, 3, static_cast<int>(threadIdx.x) - 128, live, base);
    } else if (warp < 4) {
        hc_mega_norm<H, EMIT>(p, m, 2, 4, static_cast<int>(threadIdx.x), live, base);
    }

    tc_fence_before();
    __syncthreads();
    tc_fence_after();
    if (warp == 1) tmem_dealloc(tmem, 2 * kMegaAccCols);
    // The last block out puts the counters back to zero for the next launch.
    if (threadIdx.x == 0) {
        __threadfence();
        const unsigned out = atomicAdd(p.sync + kSyncExit, 1u);
        if (out == gridDim.x - 1) {
            __threadfence();
            const int blocks = (p.rows + kMegaRows - 1) / kMegaRows;
            for (int b = 0; b < blocks; ++b) p.sync[kSyncBlocks + b] = 0;
            p.sync[kSyncTasks] = 0;
            p.sync[kSyncNormRows] = 0;
            p.sync[kSyncSinkRows] = 0;
            p.sync[kSyncExit] = 0;
        }
    }
}

}

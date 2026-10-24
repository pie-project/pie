#pragma once

// The mega-MoE's layout and scheduler (`linear/moe/mega_sm100.cu`): every
// CTA of the persistent grid lays the fire's routes out by expert itself, in
// the same order, so no CTA waits on another to learn where a route's rows
// go; the tasks are then a list each CTA walks at its own stride.
//
// The layout: an expert's routes in route order, cut into blocks of kBN
// pool rows (an expert's last block padded), the blocks expert after
// expert. Pool row p of block b is b * 64 + its place in the block; the
// activation pool (the routes' rows quantized, or copied) and the
// intermediate pool (the GLU's rows) are both laid so.
//
// The tasks: first the gate-up tiles (`L1`: a block by 128 intermediate
// columns, their gate and up rows), then the down tiles (`L2`: a block by
// 128 output columns), each block's tiles adjacent. A CTA takes tasks
// blockIdx.x, + gridDim.x, ...; a gate-up task waits on nothing but its
// rows' quantization, a down task on the gate-up tiles that land the K it
// reads, so every gate-up task finishes and no wait closes a cycle.
//
// Between them, global flags (zero at rest, the last CTA to leave zeroing
// them): a block's quantized rows (`xready`, a count), its intermediate
// tiles (`hready`, a bit a tile), an output column tile's routes landed
// (`done`, a count), and the CTAs out (`finish`).

#include "prim/device.cuh"
#include "prim/smem.cuh"

namespace pie::linear::mega {

constexpr int kMaxExperts = 512;
constexpr int kMaxRoutes = 2048;
// Output column tiles (128 columns each): a hidden width of 8192 at most.
constexpr int kMaxColTiles = 64;
// Warps: 0 the weight loader, 1 the row loader, 2 the MMA, 3 the quantizer,
// 4-7 the epilogue (TMEM lane quarters 0-3); 3-7 also fold the routes.
constexpr int kThreads = 256;

// The workspace's words.
__host__ __device__ constexpr int ws_xready() { return 0; }
__host__ __device__ constexpr int ws_hready() { return kMaxRoutes; }
__host__ __device__ constexpr int ws_done() { return 2 * kMaxRoutes; }
__host__ __device__ constexpr int ws_finish() { return 2 * kMaxRoutes + kMaxColTiles; }
__host__ __device__ constexpr int ws_words() { return 2 * kMaxRoutes + kMaxColTiles + 1; }

// ---- cross-CTA flags ---------------------------------------------------------

__device__ __forceinline__ u32 ld_acquire(const u32* p) {
    u32 v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void add_release(u32* p, u32 v) {
    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ void or_release(u32* p, u32 v) {
    asm volatile("red.release.gpu.global.or.b32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

// Orders this thread's generic writes to global memory before the async
// proxy's reads (another CTA's TMA), and its acquired flags before its own.
__device__ __forceinline__ void fence_proxy_global() {
    asm volatile("fence.proxy.async.global;" ::: "memory");
}

// Until `*p` reaches `want` (counters only grow within a launch).
__device__ __forceinline__ void wait_count(const u32* p, u32 want) {
    while (ld_acquire(p) < want) {
    }
}

// Until every bit of `need` is set in `*p`.
__device__ __forceinline__ void wait_bits(const u32* p, u32 need) {
    while ((ld_acquire(p) & need) != need) {
    }
}

// The epilogue warps' own barrier (4 warps).
__device__ __forceinline__ void epi_sync() { asm volatile("bar.sync 1, 128;" ::: "memory"); }

// ---- the layout --------------------------------------------------------------

struct Routing {
    int count[kMaxExperts + 1];   // an expert's routes
    int start[kMaxExperts + 1];   // its first sorted route; start[E] the routes
    int bstart[kMaxExperts + 1];  // its first block; bstart[E] the blocks
    unsigned short block_expert[kMaxRoutes];
    unsigned short sorted[kMaxRoutes];  // window-relative route ids, by expert
    int t0, tokens;                     // the live window's first token and its tokens
};

// The fire's layout, every thread of the CTA taking part: routes
// `[t0 * top, (t0 + tokens) * top)` (those naming no expert left out) by
// expert, warp w laying out the w-th eighth of them in route order. `seg`:
// 8 * experts shorts of scratch.
template <int kBN>
__device__ __forceinline__ void lay_out(Routing& R, const i32* __restrict__ routes, int t0,
                                        int tokens, int top, int experts,
                                        unsigned short* seg) {
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid / 32, lane = tid % 32;
    const int n = tokens * top;
    const int per = (n + 7) / 8;
    const int lo = min(n, warp * per), hi = min(n, lo + per);
    for (int i = tid; i < 8 * experts; i += kThreads) seg[i] = 0;
    if (tid == 0) {
        R.t0 = t0;
        R.tokens = tokens;
    }
    __syncthreads();
    unsigned short* mine = seg + warp * experts;
    auto expert_of = [&](int r) {
        const int e = r < hi ? __ldg(&routes[static_cast<long long>(t0) * top + r]) : -1;
        return e >= 0 && e < experts ? e : -1;
    };
    for (int c = lo; c < hi; c += 32) {
        const int e = expert_of(c + lane);
        const u32 same = __match_any_sync(0xffffffffu, e);
        if (e >= 0 && static_cast<int>(__ffs(same)) - 1 == lane) {
            mine[e] = static_cast<unsigned short>(mine[e] + __popc(same));
        }
        __syncwarp();
    }
    __syncthreads();
    // Each expert's count, and its warps' first ranks.
    for (int e = tid; e < experts; e += kThreads) {
        int run = 0;
        for (int w = 0; w < 8; ++w) {
            const int c = seg[w * experts + e];
            seg[w * experts + e] = static_cast<unsigned short>(run);
            run += c;
        }
        R.count[e] = run;
    }
    __syncthreads();
    // The routes' and blocks' prefix sums, warp 0 alone (512 experts at most).
    if (warp == 0) {
        int routes_before = 0, blocks_before = 0;
        for (int e0 = 0; e0 < experts; e0 += 32) {
            const int e = e0 + lane;
            const int c = e < experts ? R.count[e] : 0;
            const int b = (c + kBN - 1) / kBN;
            int ic = c, ib = b;
            for (int d = 1; d < 32; d <<= 1) {
                const int uc = __shfl_up_sync(0xffffffffu, ic, d);
                const int ub = __shfl_up_sync(0xffffffffu, ib, d);
                if (lane >= d) {
                    ic += uc;
                    ib += ub;
                }
            }
            if (e < experts) {
                R.start[e] = routes_before + ic - c;
                R.bstart[e] = blocks_before + ib - b;
            }
            routes_before += __shfl_sync(0xffffffffu, ic, 31);
            blocks_before += __shfl_sync(0xffffffffu, ib, 31);
        }
        if (lane == 0) {
            R.start[experts] = routes_before;
            R.bstart[experts] = blocks_before;
            R.count[experts] = 0;
        }
    }
    __syncthreads();
    for (int c = lo; c < hi; c += 32) {
        const int r = c + lane;
        const int e = expert_of(r);
        const u32 same = __match_any_sync(0xffffffffu, e);
        if (e >= 0) {
            const int rank = mine[e] + __popc(same & ((1u << lane) - 1u));
            R.sorted[R.start[e] + rank] = static_cast<unsigned short>(r);
        }
        __syncwarp();
        if (e >= 0 && static_cast<int>(__ffs(same)) - 1 == lane) {
            mine[e] = static_cast<unsigned short>(mine[e] + __popc(same));
        }
        __syncwarp();
    }
    for (int e = tid; e < experts; e += kThreads) {
        for (int b = R.bstart[e]; b < R.bstart[e + 1]; ++b) {
            R.block_expert[b] = static_cast<unsigned short>(e);
        }
    }
    __syncthreads();
}

// ---- the tasks ---------------------------------------------------------------

struct Task {
    int l2;    // 0 a gate-up tile, 1 a down tile
    int b;     // the block
    int j;     // its 128-column tile
    int e;     // the block's expert
    int rows;  // its live rows
    int s0;    // its first sorted route
};

// Task `t` of `blocks * (j1 + j2)`: the gate-up tiles (j1 a block), then
// the down tiles (j2 a block).
template <int kBN>
__device__ __forceinline__ Task task_of(const Routing& R, int t, int blocks, int j1, int j2) {
    Task x;
    const int n1 = blocks * j1;
    x.l2 = t >= n1 ? 1 : 0;
    const int u = x.l2 ? t - n1 : t;
    const int per = x.l2 ? j2 : j1;
    x.b = u / per;
    x.j = u - x.b * per;
    x.e = R.block_expert[x.b];
    const int m = x.b - R.bstart[x.e];
    x.rows = min(R.count[x.e] - m * kBN, kBN);
    x.s0 = R.start[x.e] + m * kBN;
    return x;
}

}  // namespace pie::linear::mega

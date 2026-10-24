#pragma once

// The mega-MoE's layout and scheduler (`linear/moe/mega_sm100.cu`): every
// CTA of the persistent grid lays the fire's routes out by expert itself, in
// the same order, so no CTA waits on another to learn where a route's rows
// go; the tasks are then one list the CTAs claim from as they free.
//
// The layout: an expert's routes in route order, cut into blocks of kBN
// (an expert's last block short), the blocks expert after expert. Block b's
// pool rows start at b * 64; the activation pool (the routes' rows
// quantized, or copied) and the intermediate pool (the GLU's rows) are both
// laid so.
//
// The tasks: first the gate-up tiles (`L1`: a block's 128 intermediate
// columns, their gate rows and their up rows; a decode's few in shares of
// K, the last share in summing them), then the down tiles (`L2`: a
// block's 128 output columns). A CTA claims the next task from a global
// count as its ring frees: a gate-up task waits on nothing but its rows'
// quantization, and a down task (claimed only once every gate-up task is) on
// the gate-up tiles that land the K it reads, so every gate-up task
// finishes and no wait closes a cycle.
//
// Between them, global flags (zero at rest, the last CTA out zeroing them):
// a block's quantized row chunks (`xready`, a count), its intermediate
// tiles (`hready`, a bit a tile), an output column tile's routes landed
// (`done`, a count), the tasks claimed (`next`) and the CTAs out
// (`finish`).

#include "prim/device.cuh"
#include "prim/smem.cuh"

namespace pie::linear::mega {

constexpr int kMaxExperts = 512;
constexpr int kMaxRoutes = 2048;
// Output column tiles (128 columns each): a hidden width of 8192 at most.
constexpr int kMaxColTiles = 64;
// Warps: 0 claims tasks and loads the weights, 1 the routes' rows, 2 the
// MMA, 3 the quantizer, 4-7 and 8-11 two epilogue warpgroups taking
// alternate tasks (each warp the TMEM lane quarter warp % 4); 3-11 also
// quantize first and fold last.
constexpr int kThreads = 384;
constexpr int kHelpers = 9;  // warps 3-11

// The workspace's words.
__host__ __device__ constexpr int ws_xready() { return 0; }
__host__ __device__ constexpr int ws_hready() { return kMaxRoutes; }
__host__ __device__ constexpr int ws_done() { return 2 * kMaxRoutes; }
__host__ __device__ constexpr int ws_finish() { return 2 * kMaxRoutes + kMaxColTiles; }
__host__ __device__ constexpr int ws_next() { return ws_finish() + 1; }
// A split gate-up tile's parts landed: its last part zeroes the count as it
// sums them. A split fire's blocks' tiles, kMaxSplitTiles at most.
constexpr int kMaxSplitTiles = 8192;
__host__ __device__ constexpr int ws_parts() { return ws_next() + 1; }
__host__ __device__ constexpr int ws_words() { return ws_parts() + kMaxSplitTiles; }

// ---- cross-CTA flags ---------------------------------------------------------

__device__ __forceinline__ void add_release(u32* p, u32 v) {
    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ u32 add_acq_rel(u32* p, u32 v) {
    u32 old;
    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory");
    return old;
}

__device__ __forceinline__ void or_release(u32* p, u32 v) {
    asm volatile("red.release.gpu.global.or.b32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

// Orders this thread's generic writes to global memory before the async
// proxy's reads (another CTA's TMA), and its acquired flags before its own.
__device__ __forceinline__ void fence_proxy_global() {
    asm volatile("fence.proxy.async.global;" ::: "memory");
}

__device__ __forceinline__ u32 ld_relaxed(const u32* p) {
    u32 v;
    asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void fence_acquire() { asm volatile("fence.acq_rel.gpu;" ::: "memory"); }

// Until `*p` reaches `want` (counters only grow within a launch), then
// acquired. A waiter polls relaxed and backs off between reads: an acquire
// a read invalidates the SM's L1 each time, stalling its other warps'
// shared-memory traffic, and many waiters on one line slow the stores they
// wait for.
__device__ __forceinline__ void wait_count(const u32* p, u32 want) {
    for (u32 ns = 32; ld_relaxed(p) < want; ns = min(2 * ns, 256u)) __nanosleep(ns);
    fence_acquire();
}

// Until every bit of `need` is set in `*p`; the word, acquired.
__device__ __forceinline__ u32 wait_bits(const u32* p, u32 need) {
    u32 v;
    for (u32 ns = 32; ((v = ld_relaxed(p)) & need) != need; ns = min(2 * ns, 256u)) __nanosleep(ns);
    fence_acquire();
    return v;
}

// An epilogue warpgroup's own barrier (its four warps), 1 or 2.
__device__ __forceinline__ void wg_sync(int wg) {
    asm volatile("bar.sync %0, 128;" ::"r"(1 + wg) : "memory");
}

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
// expert, warp w < 8 laying out the w-th eighth of them in route order.
// `seg`: 8 * experts shorts of scratch.
template <int kBN>
__device__ __forceinline__ void lay_out(Routing& R, const i32* __restrict__ routes, int t0,
                                        int tokens, int top, int experts,
                                        unsigned short* seg) {
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid / 32, lane = tid % 32;
    const int n = tokens * top;
    const int per = (n + 7) / 8;
    const int lo = warp < 8 ? min(n, warp * per) : n;
    const int hi = warp < 8 ? min(n, lo + per) : n;
    // This warp's routes' experts, read once (2048 routes at most: eight
    // chunks of 32 a warp).
    int mine_e[8];
#pragma unroll
    for (int c = 0; c < 8; ++c) {
        const int r = lo + 32 * c + lane;
        const int e = r < hi ? __ldg(&routes[static_cast<long long>(t0) * top + r]) : -1;
        mine_e[c] = e >= 0 && e < experts ? e : -1;
    }
    for (int i = tid; i < 8 * experts; i += kThreads) seg[i] = 0;
    if (tid == 0) {
        R.t0 = t0;
        R.tokens = tokens;
    }
    __syncthreads();
    unsigned short* mine = seg + (warp < 8 ? warp : 0) * experts;
#pragma unroll
    for (int c = 0; c < 8; ++c) {
        if (lo + 32 * c >= hi) break;
        const int e = mine_e[c];
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
#pragma unroll
    for (int c = 0; c < 8; ++c) {
        if (lo + 32 * c >= hi) break;
        const int r = lo + 32 * c + lane;
        const int e = mine_e[c];
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
    int part;  // a gate-up tile's share of K (of the fire's `split`)
    int e;     // the block's expert
    int rows;  // its live rows
    int s0;    // its first sorted route
};

// Task `t` of `blocks * (j1 * split + j2)`: the gate-up tiles (j1 a block,
// each in `split` shares of K, adjacent), then the down tiles (j2 a block).
template <int kBN>
__device__ __forceinline__ Task task_of(const Routing& R, int t, int blocks, int j1, int j2,
                                        int split) {
    Task x;
    const int n1 = blocks * j1 * split;
    x.l2 = t >= n1 ? 1 : 0;
    const int u = x.l2 ? t - n1 : t / split;
    x.part = x.l2 ? 0 : t - u * split;
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

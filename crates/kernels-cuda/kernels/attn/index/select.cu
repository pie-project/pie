#include "attn/index/index.cuh"
#include "prim/smem.cuh"

// The ranking of a scored index's rows with a row's keys spread over a
// cluster of CTAs (a decode's few rows, each thousands to tens of thousands
// of blocks): the picks `select_top_radix` makes, bit for bit, in its order
// (every score above the k-th largest's order bits, then the first of its
// ties by index; ascending). FlashInfer's and DeepSeek's multi-CTA radix
// top-k, rewritten:
//
//   - A row's `ranks` CTAs (one cluster, or one CTA) split its `nkeys`
//     scores into contiguous chunks of whole quads, each CTA's held in its
//     shared memory as order bits (`order_of`), read once from L2.
//   - The k-th largest score's bits are found 11, 11, then 10 bits a pass:
//     each CTA counts its survivors into shared bins and adds the bins it
//     filled into the cluster's sums for the pass (distributed shared
//     memory, each rank a span of the bins, so no one CTA serves the
//     cluster); after the cluster's barrier every CTA reads the sums and
//     walks them from the top bin down to the same digit and need. A pass
//     whose bin is taken whole ends the search.
//   - Each CTA counts its quads' scores above the threshold and its ties,
//     hands the two counts to rank 0 (one more barrier), and places every
//     score above and the first `need` ties in index order at their rank: a
//     chunk's place is the counts of the chunks before it, a quad's in its
//     tile the counts of the (item, warp)s before it and of the lanes before
//     it in its warp.
//
// Clusters need sm_90; below it a row is one CTA. Nothing else here is
// arch-specific.
namespace pie::attn::index {

namespace split {

constexpr int kThreads = 512;
constexpr int kItems = 16;
constexpr int kTile = kThreads * kItems;  // quads a tile of a CTA's chunk
constexpr int kWarps = kThreads / 32;
constexpr int kBins = 2048;               // the widest pass's
constexpr int kPasses = 3;
constexpr int kMaxRanks = 16;

// A CTA's shared memory ahead of its chunk.
struct alignas(16) Shared {
    u32 sums[kPasses][kBins];   // the cluster's bins a pass, a span a rank
    u32 hist[kBins];            // this CTA's
    u32 counts[kItems * kWarps];  // each (item, warp)'s counts, then their prefix
    int tally[kMaxRanks][2];    // rank 0's: each chunk's scores above and ties
    int warp_sums[kWarps];
    int found[2][3];
    u32 s_tile;
    int s_before[2];
};

constexpr int kHeader = (static_cast<int>(sizeof(Shared)) + 15) / 16 * 16;
static_assert(kHeader == 34032, "the host's `HEADER`");

}  // namespace split

struct SplitParams {
    const u32* qo_indptr;
    const u32* page_indptr;
    const u32* last_page_lens;
    const float* scores;  // [rows][score_stride]
    const u32* win;
    int requests;
    int page_size;
    int score_stride;
    int top_k;
    int ratio;
    int rows;
    int ranks;            // CTAs a row, a cluster of them
};

// A shared address's twin in CTA `rank` of the cluster, and what reads and
// writes it there; then the cluster's barrier halves.
__device__ __forceinline__ u32 at_rank(const void* p, u32 rank) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    u32 remote;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(smem_addr(p)), "r"(rank));
    return remote;
#else
    return smem_addr(p);
#endif
}

__device__ __forceinline__ void add_at(u32 remote, u32 v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("red.relaxed.cluster.shared::cluster.add.u32 [%0], %1;" ::"r"(remote), "r"(v) : "memory");
#endif
}

__device__ __forceinline__ void st_at(u32 remote, int v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("st.shared::cluster.s32 [%0], %1;" ::"r"(remote), "r"(v) : "memory");
#endif
}

__device__ __forceinline__ int ld_at(u32 remote) {
    int v = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("ld.shared::cluster.s32 %0, [%1];" : "=r"(v) : "r"(remote) : "memory");
#endif
    return v;
}

__device__ __forceinline__ uint4 ld4_at(u32 remote) {
    uint4 v = make_uint4(0u, 0u, 0u, 0u);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("ld.shared::cluster.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "r"(remote)
                 : "memory");
#endif
    return v;
}

__device__ __forceinline__ void ranks_arrive() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
#endif
}

__device__ __forceinline__ void ranks_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
#endif
}

// From the top bin down, the bin where the count reaches `need`, the need
// left within it, and whether the bin is taken whole. Thread `tid` holds the
// counts of bins `top - i`, `top = bins - 1 - 4 * tid` (none past the
// bottom): a warp's counts summed, then the warp that holds the bin walks
// its lanes'. `found` alternates between two slots a pass, so no barrier
// guards it.
__device__ __forceinline__ void find_digit(const int (&c)[4], int top, int need, int tid, int* warp_sums,
                                           int* found, int& digit, int& left, bool& whole) {
    const int lane = tid & 31, warp = tid >> 5;
    const int sum = c[0] + c[1] + c[2] + c[3];
    const int total = warp_sum(sum);
    if (lane == 0) warp_sums[warp] = total;
    __syncthreads();
    int before = 0;
#pragma unroll
    for (int w = 0; w < split::kWarps; ++w) before += w < warp ? warp_sums[w] : 0;
    if (before < need && before + total >= need) {
        int incl = sum;
#pragma unroll
        for (int sh = 1; sh < 32; sh <<= 1) {
            const int x = __shfl_up_sync(0xffffffffu, incl, sh);
            if (lane >= sh) incl += x;
        }
        int run = before + incl - sum;
        if (run < need && run + sum >= need) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                if (run < need && run + c[i] >= need) {
                    found[0] = top - i;
                    found[1] = need - run;
                    found[2] = need - run == c[i];
                }
                run += c[i];
            }
        }
    }
    __syncthreads();
    digit = found[0];
    left = found[1];
    whole = found[2] != 0;
}

__global__ void __launch_bounds__(split::kThreads, 1) index_select_split(const SplitParams p,
                                                                       i32* __restrict__ selection) {
    using namespace split;
    // All of it dynamic, so a launch past 48 KB opts in whatever the split.
    extern __shared__ __align__(16) unsigned char smem[];
    Shared& sh = *reinterpret_cast<Shared*>(smem);
    u32* const held = reinterpret_cast<u32*>(smem + kHeader);  // the chunk's order bits
    u32 (&hist)[kBins] = sh.hist;
    u32 (&sums)[kPasses][kBins] = sh.sums;
    int (&tally)[kMaxRanks][2] = sh.tally;
    int* const warp_sums = sh.warp_sums;
    int (&found)[2][3] = sh.found;
    u32* const counts = sh.counts;
    u32& s_tile = sh.s_tile;
    int* const s_before = sh.s_before;

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31, warp = tid >> 5;
    const int ranks = p.ranks;
    const int c = static_cast<int>(blockIdx.x) % ranks;
    const int t = static_cast<int>(blockIdx.x) / ranks;
    // A programmatic dependent of the scorer (`kernel.pdl`): nothing it
    // writes is read before it has finished.
    pdl_wait();
    pdl_launch();
    const int live = p.win != nullptr ? min(static_cast<int>(p.win[0]), p.rows) : p.rows;
    // A row's CTAs leave together: none waits on a barrier another skips.
    if (t >= live) return;
    const int base_row = p.win != nullptr ? static_cast<int>(p.win[1]) : 0;
    const int R = (p.win != nullptr && static_cast<int>(p.win[2]) < p.requests) ? static_cast<int>(p.win[2])
                                                                               : p.requests;
    const RowInfo ri = row_info(p, t, R);
    const int nkeys = ri.nkeys;
    i32* srow = selection + static_cast<long long>(base_row + t) * p.top_k;
    if (nkeys <= p.top_k) {
        for (int k = c * kThreads + tid; k < p.top_k; k += ranks * kThreads) srow[k] = (k < nkeys) ? k : -1;
        return;
    }
    const bool cluster = ranks > 1;
    if (cluster) {
        // Every rank's sums zero before any rank adds into them.
        for (int b = tid; b < kPasses * kBins; b += kThreads) (&sums[0][0])[b] = 0u;
        ranks_arrive();
    }
    // A chunk a whole number of quads, so a quad never straddles two.
    const int chunk = ((nkeys + ranks - 1) / ranks + 3) & ~3;
    const int lo = min(c * chunk, nkeys);
    const int n = min(lo + chunk, nkeys) - lo;
    const int quads = (n + 3) / 4;
    {
        const u32* frow = reinterpret_cast<const u32*>(p.scores) + static_cast<long long>(t) * p.score_stride + lo;
#pragma unroll 4
        for (int k = tid; k < 4 * quads; k += kThreads) held[k] = k < n ? order_of(__ldg(frow + k)) : 0u;
    }
    const uint4* held4 = reinterpret_cast<const uint4*>(held);
    __syncthreads();

    // The k-th largest's bits under `mask`, and the ties at it to take.
    u32 prefix = 0, mask = 0;
    int need = p.top_k;
#pragma unroll 1
    for (int pass = 0; pass < kPasses; ++pass) {
        const int shift = pass == 0 ? 21 : (pass == 1 ? 10 : 0);
        const int bins = pass == 2 ? 1024 : 2048;
        const u32 low = static_cast<u32>(bins - 1);
        for (int b = tid; b < bins; b += kThreads) hist[b] = 0;
        __syncthreads();
#pragma unroll 2
        for (int q = tid; q < quads; q += kThreads) {
            const uint4 v = held4[q];
            const u32 e[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                if (4 * q + i < n && (e[i] & mask) == prefix) atomicAdd(&hist[(e[i] >> shift) & low], 1u);
            }
        }
        __syncthreads();
        const int top = bins - 1 - 4 * tid;
        int cnt[4] = {0, 0, 0, 0};
        if (!cluster) {
            if (top >= 0) {
#pragma unroll
                for (int i = 0; i < 4; ++i) cnt[i] = static_cast<int>(hist[top - i]);
            }
        } else {
            if (pass == 0) ranks_wait();
            // Rank r sums bins [r * span, (r + 1) * span): no one CTA's
            // memory serves the whole cluster.
            const int span = bins / ranks;
            for (int b = tid; b < bins; b += kThreads) {
                const u32 v = hist[b];
                if (v != 0u) add_at(at_rank(&sums[pass][b], static_cast<u32>(b / span)), v);
            }
            ranks_arrive();
            ranks_wait();
            if (top >= 0) {
                const int b = top - 3;
                const uint4 v = ld4_at(at_rank(&sums[pass][b], static_cast<u32>(b / span)));
                cnt[0] = static_cast<int>(v.w);
                cnt[1] = static_cast<int>(v.z);
                cnt[2] = static_cast<int>(v.y);
                cnt[3] = static_cast<int>(v.x);
            }
        }
        int digit, left;
        bool whole;
        find_digit(cnt, top, need, tid, warp_sums, found[pass & 1], digit, left, whole);
        prefix |= static_cast<u32>(digit) << shift;
        mask |= low << shift;
        need = left;
        if (whole) break;
    }

    // A quad's scores above the threshold and its ties, as two bit sets.
    auto marks = [&](int q, unsigned& gt, unsigned& tie) {
        gt = 0u;
        tie = 0u;
        if (q < quads) {
            const uint4 v = held4[q];
            const u32 e[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const u32 m = e[i] & mask;
                const bool in = 4 * q + i < n;
                gt |= (in && m > prefix) ? 1u << i : 0u;
                tie |= (in && m == prefix) ? 1u << i : 0u;
            }
        }
    };
    // A tile is `kItems` items of `kThreads` quads; item `i` of thread `tid`
    // is the chunk's quad `tile * kTile + i * kThreads + tid`, so index order
    // is (item, warp, lane). Counts go packed, scores above in the low 16
    // bits and ties in the high (a tile's at most 32768 of each): a lane's
    // before it in its warp (`before[i]`), each (item, warp)'s before it in
    // the tile (`counts`), and the tile's (`s_tile`).
    const int tiles = (quads + kTile - 1) / kTile;
    u32 before[kItems];
    auto scan = [&](int tile) {
#pragma unroll
        for (int i = 0; i < kItems; ++i) {
            before[i] = 0u;
            const int q0 = tile * kTile + i * kThreads;
            if (q0 >= quads) {
                if (lane == 0) counts[i * kWarps + warp] = 0u;
                continue;
            }
            unsigned gt, tie;
            marks(q0 + tid, gt, tie);
            const u32 own = static_cast<u32>(__popc(gt)) | (static_cast<u32>(__popc(tie)) << 16);
            u32 incl = own;
#pragma unroll
            for (int sh = 1; sh < 32; sh <<= 1) {
                const u32 x = __shfl_up_sync(0xffffffffu, incl, sh);
                if (lane >= sh) incl += x;
            }
            before[i] = incl - own;
            if (lane == 31) counts[i * kWarps + warp] = incl;
        }
        __syncthreads();
        if (warp == 0) {
            constexpr int kPer = kItems * kWarps / 32;
            u32 v[kPer], sum = 0u;
#pragma unroll
            for (int k = 0; k < kPer; ++k) {
                v[k] = counts[lane * kPer + k];
                sum += v[k];
            }
            u32 incl = sum;
#pragma unroll
            for (int sh = 1; sh < 32; sh <<= 1) {
                const u32 x = __shfl_up_sync(0xffffffffu, incl, sh);
                if (lane >= sh) incl += x;
            }
            u32 r = incl - sum;
#pragma unroll
            for (int k = 0; k < kPer; ++k) {
                counts[lane * kPer + k] = r;
                r += v[k];
            }
            if (lane == 31) s_tile = incl;
        }
        __syncthreads();
    };
    // Every score above and the first `mine` ties of the chunk at their
    // places: `base` the chunk's, `run_*` the tiles' before this one.
    auto place = [&](int tile, int base, int mine, int run_gt, int run_tie) {
#pragma unroll
        for (int i = 0; i < kItems; ++i) {
            const int q = tile * kTile + i * kThreads + tid;
            if (tile * kTile + i * kThreads >= quads) break;
            unsigned gt, tie;
            marks(q, gt, tie);
            if ((gt | tie) == 0u) continue;
            const u32 at = counts[i * kWarps + warp] + before[i];
            int gr = run_gt + static_cast<int>(at & 0xffffu);
            int tr = run_tie + static_cast<int>(at >> 16);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const bool g = (gt >> e) & 1u, k = (tie >> e) & 1u;
                if (g || (k && tr < mine)) srow[base + gr + min(tr, mine)] = lo + 4 * q + e;
                gr += g;
                tr += k;
            }
        }
    };

    // The chunk's counts: from its one tile's scan, else counted.
    int gt = 0, tie = 0;
    if (tiles == 1) {
        scan(0);
        gt = static_cast<int>(s_tile & 0xffffu);
        tie = static_cast<int>(s_tile >> 16);
    } else {
#pragma unroll 2
        for (int q = tid; q < quads; q += kThreads) {
            unsigned g, k;
            marks(q, g, k);
            gt += __popc(g);
            tie += __popc(k);
        }
        gt = warp_sum(gt);
        tie = warp_sum(tie);
        if (lane == 0) {
            counts[warp] = static_cast<u32>(gt);
            counts[kWarps + warp] = static_cast<u32>(tie);
        }
        __syncthreads();
        gt = 0;
        tie = 0;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) {
            gt += static_cast<int>(counts[w]);
            tie += static_cast<int>(counts[kWarps + w]);
        }
        __syncthreads();
    }
    // The chunks' before this one, by way of rank 0.
    int gt_before = 0, tie_before = 0;
    if (cluster) {
        if (tid == 0) {
            st_at(at_rank(&tally[c][0], 0), gt);
            st_at(at_rank(&tally[c][1], 0), tie);
        }
        ranks_arrive();
        ranks_wait();
        if (warp == 0) {
            int g = 0, k = 0;
            if (lane < c) {
                g = ld_at(at_rank(&tally[lane][0], 0));
                k = ld_at(at_rank(&tally[lane][1], 0));
            }
            g = warp_sum(g);
            k = warp_sum(k);
            if (lane == 0) {
                s_before[0] = g;
                s_before[1] = k;
            }
        }
        // Done with rank 0's memory; its exit waits on the others' arrival.
        ranks_arrive();
        __syncthreads();
        gt_before = s_before[0];
        tie_before = s_before[1];
    }
    const int mine = max(0, min(need - tie_before, tie));  // this chunk's ties taken
    const int base = gt_before + min(need, tie_before);
    if (tiles == 1) {
        place(0, base, mine, 0, 0);
    } else {
        int run_gt = 0, run_tie = 0;
#pragma unroll 1
        for (int tile = 0; tile < tiles; ++tile) {
            scan(tile);
            place(tile, base, mine, run_gt, run_tie);
            run_gt += static_cast<int>(s_tile & 0xffffu);
            run_tie += static_cast<int>(s_tile >> 16);
            __syncthreads();
        }
    }
    if (cluster) ranks_wait();
}

}  // namespace pie::attn::index

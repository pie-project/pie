#include "prim/device.cuh"
#include "prim/peer.cuh"

namespace pie::collective {

using peer::kMaxRanks;

// Expert-parallel dispatch inside one process whose ranks map one another's
// memory: every rank sends each of its live tokens once to every rank that
// holds one of its routes' experts, with the token's routes as that rank's
// local slots (-1 for a route held elsewhere). The shape is DeepEP's
// intranode dispatch (count, notify, copy) in this codebase's primitives.
//
// A receiving rank's rows are ordered by source rank, then token: rank s's
// rows start past every lower rank's, and within them keep the source's
// token order. That order, and the replica a route picks, depend only on
// (rank, token, route), so one rank and N ranks lay the same rows.
//
// Three launches on each rank's stream, every rank firing all three:
//   dispatch_count  (one block)  counts rows per destination, tells each
//                                destination its count, meets the ranks,
//                                and reads where its rows start there;
//   dispatch_send   (a warp a token) copies the rows into the destinations;
//   dispatch_settle (one block)  meets the ranks again, so every row has
//                                landed, then marks the tail as padding and
//                                counts this rank's rows per local slot
//                                (and adds them to the group's loads).

// Where rank s tells rank d how many rows it sends: d's inbox, s's column.
struct Inbox {
    i32 sent[kMaxRanks];
};

// A rank's EP signal block: the barrier's counters, then the inbox.
constexpr int kInboxAt = 3072;
static_assert(sizeof(peer::Signals) <= kInboxAt, "the inbox sits past the barrier counters");

// Barrier slots, one per single-block launch that meets the ranks (2 is
// `combine_settle`'s).
constexpr int kSlotCount = 0;
constexpr int kSlotSettle = 1;

constexpr int kCountThreads = 1024;
constexpr int kSendWarps = 8;

struct DispatchParams {
    const int4* x;
    const i32* routes;
    const float* weights;
    // The live window (`win[0]` tokens from token `win[1]`), or null: the
    // first `live` tokens are.
    const u32* win;
    // `[experts][stride]`: a replica count, then that many physical slots
    // (rank * slots + local slot).
    const i32* place;
    i32* pos;
    i32* offs;
    i32* counts;
    // Each local slot's received rows summed over the dispatches since the
    // group last read them (the rebalance's loads), or null.
    u32* loads;
    int4* rows[kMaxRanks];
    i32* local[kMaxRanks];
    float* row_weights[kMaxRanks];
    i32* origin[kMaxRanks];
    peer::Group group;
    peer::Signals* self;
    i32 rank;
    i32 ranks;
    // The step's tokens, every rank's: rows are `ranks * tokens` and an
    // origin `rank * tokens + t`. `x` holds at least the live ones.
    i32 tokens;
    i32 live;
    i32 packs;
    i32 top_k;
    i32 experts;
    i32 slots;
    i32 stride;
};

__device__ __forceinline__ Inbox* inbox(peer::Signals* s) {
    return reinterpret_cast<Inbox*>(reinterpret_cast<char*>(s) + kInboxAt);
}

__device__ __forceinline__ bool live(const DispatchParams& p, int t) {
    if (p.win == nullptr) return t < p.live;
    const int at = t - static_cast<int>(p.win[1]);
    return at >= 0 && at < static_cast<int>(p.win[0]);
}

// Route j of token t's physical slot, or -1 for no expert.
__device__ __forceinline__ int physical(const DispatchParams& p, int t, int j) {
    const int e = p.routes[t * p.top_k + j];
    if (e < 0 || e >= p.experts) return -1;
    const i32* row = p.place + static_cast<long long>(e) * p.stride;
    return row[1 + (p.rank + t * p.top_k + j) % row[0]];
}

__global__ void __launch_bounds__(kCountThreads) dispatch_count(const DispatchParams p) {
    __shared__ int warp_base[kCountThreads / 32][kMaxRanks];
    __shared__ int total[kMaxRanks];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (threadIdx.x < kMaxRanks) total[threadIdx.x] = 0;
    __syncthreads();
    // Each chunk's tokens scanned per destination: a token's row there is
    // the tokens before it that go there.
    for (int t0 = 0; t0 < p.tokens; t0 += kCountThreads) {
        const int t = t0 + static_cast<int>(threadIdx.x);
        unsigned mask = 0;
        if (t < p.tokens && live(p, t)) {
            for (int j = 0; j < p.top_k; ++j) {
                const int ph = physical(p, t, j);
                if (ph >= 0) mask |= 1u << (ph / p.slots);
            }
        }
        int below[kMaxRanks];
#pragma unroll
        for (int d = 0; d < kMaxRanks; ++d) {
            const unsigned b = __ballot_sync(~0u, (mask >> d) & 1u);
            below[d] = __popc(b & ((1u << lane) - 1u));
            if (lane == 0) warp_base[warp][d] = __popc(b);
        }
        __syncthreads();
        if (threadIdx.x < static_cast<unsigned>(p.ranks)) {
            const int d = threadIdx.x;
            int run = total[d];
            for (int w = 0; w < kCountThreads / 32; ++w) {
                const int n = warp_base[w][d];
                warp_base[w][d] = run;
                run += n;
            }
            total[d] = run;
        }
        __syncthreads();
        if (t < p.tokens) {
            for (int d = 0; d < p.ranks; ++d)
                p.pos[t * p.ranks + d] = (mask >> d) & 1u ? warp_base[warp][d] + below[d] : -1;
        }
        __syncthreads();
    }
    if (threadIdx.x < static_cast<unsigned>(p.ranks))
        peer::st_release_i32(&inbox(p.group.signals[threadIdx.x])->sent[p.rank], total[threadIdx.x]);
    peer::barrier(p.group, p.self, p.rank, p.ranks, kSlotCount);
    if (threadIdx.x < static_cast<unsigned>(p.ranks)) {
        const int d = threadIdx.x;
        int off = 0;
        for (int s = 0; s < p.rank; ++s) off += peer::ld_acquire_i32(&inbox(p.group.signals[d])->sent[s]);
        p.offs[d] = off;
    }
}

__global__ void __launch_bounds__(32 * kSendWarps) dispatch_send(const DispatchParams p) {
    const int lane = threadIdx.x & 31;
    const int t = blockIdx.x * kSendWarps + (threadIdx.x >> 5);
    if (t >= p.tokens || !live(p, t)) return;
    const int4* src = p.x + static_cast<long long>(t) * p.packs;
    const int ph = lane < p.top_k ? physical(p, t, lane) : -1;
    for (int d = 0; d < p.ranks; ++d) {
        const int at = p.pos[t * p.ranks + d];
        if (at < 0) continue;
        const long long row = p.offs[d] + at;
        peer::warp_copy(p.rows[d] + row * p.packs, src, p.packs, lane);
        if (lane < p.top_k) {
            const bool here = ph >= 0 && ph / p.slots == d;
            p.local[d][row * p.top_k + lane] = here ? ph % p.slots : -1;
            if (p.weights != nullptr)
                p.row_weights[d][row * p.top_k + lane] = here ? p.weights[t * p.top_k + lane] : 0.f;
        }
        if (lane == 0) p.origin[d][row] = p.rank * p.tokens + t;
    }
    // The rows reach the peers before `dispatch_settle`'s release says so:
    // a kernel boundary orders them on this device, the fence across it.
    __threadfence_system();
}

__global__ void __launch_bounds__(kCountThreads) dispatch_settle(const DispatchParams p) {
    __shared__ int received;
    peer::barrier(p.group, p.self, p.rank, p.ranks, kSlotSettle);
    if (threadIdx.x == 0) {
        int n = 0;
        for (int s = 0; s < p.ranks; ++s) n += peer::ld_acquire_i32(&inbox(p.self)->sent[s]);
        received = n;
    }
    for (int s = threadIdx.x; s < p.slots; s += blockDim.x) p.counts[s] = 0;
    __syncthreads();
    i32* local = p.local[p.rank];
    const int cap = p.ranks * p.tokens;
    for (int i = threadIdx.x; i < cap * p.top_k; i += blockDim.x) {
        const int r = i / p.top_k;
        if (r >= received) {
            local[i] = -1;
            if (i == r * p.top_k) p.origin[p.rank][r] = -1;
        } else if (local[i] >= 0) {
            atomicAdd(&p.counts[local[i]], 1);
        }
    }
    if (p.loads == nullptr) return;
    __syncthreads();
    for (int s = threadIdx.x; s < p.slots; s += blockDim.x) p.loads[s] += static_cast<u32>(p.counts[s]);
}

}  // namespace pie::collective

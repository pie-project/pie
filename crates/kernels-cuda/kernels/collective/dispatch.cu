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
//   dispatch_count  (a block a chunk of tokens) counts rows per
//                                destination; the last block to finish lays
//                                the chunks end to end, tells each
//                                destination its count, meets the ranks, and
//                                reads where its rows start there;
//   dispatch_send   (a block a token at a time) copies the row into its
//                                destinations;
//   dispatch_settle (one block)  meets the ranks again, so every row has
//                                landed, then marks the tail as padding and
//                                counts this rank's rows per local slot
//                                (and adds them to the group's loads).

// Where rank s tells rank d how many rows it sends: d's inbox, s's column.
struct Inbox {
    i32 sent[kMaxRanks];
};

// A rank's EP signal block: the barrier's counters, then the inbox, then
// the ticket `dispatch_count`'s blocks take as each finishes (back at 0
// once the last has).
constexpr int kInboxAt = 3072;
constexpr int kTicketAt = kInboxAt + sizeof(Inbox);
static_assert(sizeof(peer::Signals) <= kInboxAt, "the inbox sits past the barrier counters");
static_assert(kTicketAt + sizeof(unsigned) <= 4096, "the ticket sits in the signal block");

// Barrier slots, one per single-block launch that meets the ranks (2 is
// `combine_settle`'s).
constexpr int kSlotCount = 0;
constexpr int kSlotSettle = 1;

// `dispatch_count`'s block: a chunk of this many tokens, a thread each.
constexpr int kCountThreads = 1024;

// `dispatch_settle` counts a rank's slots in shared memory up to this many
// (in global atomics past it), each thread with this many routes' loads out.
constexpr int kSettleSlots = 4096;
constexpr int kSettleLoads = 4;

// `dispatch_send`'s block, a token at a time: a warp a destination's routes.
constexpr int kSendThreads = 32 * kMaxRanks;

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
    // Token t's row among its chunk's for each destination (-1: none), each
    // chunk's first row there among the rank's, and where the rank's rows
    // start on each destination.
    i32* pos;
    i32* base;
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

__device__ __forceinline__ unsigned* ticket(peer::Signals* s) {
    return reinterpret_cast<unsigned*>(reinterpret_cast<char*>(s) + kTicketAt);
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

// The ranks token t's routes go to, a bit a rank. Eight routes' loads go
// out together: the walk from route to table to replica is three loads
// deep, and a count block waits on each.
__device__ __forceinline__ unsigned destinations(const DispatchParams& p, int t) {
    unsigned mask = 0;
    for (int j0 = 0; j0 < p.top_k; j0 += 8) {
        int e[8];
        int n[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            const int j = j0 + u;
            e[u] = j < p.top_k ? p.routes[t * p.top_k + j] : -1;
            if (e[u] >= p.experts) e[u] = -1;
        }
#pragma unroll
        for (int u = 0; u < 8; ++u)
            n[u] = e[u] >= 0 ? p.place[static_cast<long long>(e[u]) * p.stride] : 0;
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            if (e[u] < 0) continue;
            const i32* row = p.place + static_cast<long long>(e[u]) * p.stride;
            const int ph = row[1 + (p.rank + t * p.top_k + j0 + u) % n[u]];
            if (ph >= 0) mask |= 1u << (ph / p.slots);
        }
    }
    return mask;
}

__global__ void __launch_bounds__(kCountThreads) dispatch_count(const DispatchParams p) {
    __shared__ int warp_base[kCountThreads / 32][kMaxRanks];
    __shared__ bool last;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int chunk = blockIdx.x;
    // The chunk's tokens scanned per destination: a token's row there is
    // the chunk's tokens before it that go there.
    const int t = chunk * kCountThreads + static_cast<int>(threadIdx.x);
    const unsigned mask = t < p.tokens && live(p, t) ? destinations(p, t) : 0u;
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
        int run = 0;
        for (int w = 0; w < kCountThreads / 32; ++w) {
            const int n = warp_base[w][d];
            warp_base[w][d] = run;
            run += n;
        }
        p.base[chunk * kMaxRanks + d] = run;
        __threadfence();
    }
    __syncthreads();
    if (t < p.tokens) {
        for (int d = 0; d < p.ranks; ++d)
            p.pos[t * p.ranks + d] = (mask >> d) & 1u ? warp_base[warp][d] + below[d] : -1;
    }
    if (threadIdx.x == 0) {
        last = atomicAdd(ticket(p.self), 1u) == gridDim.x - 1;
        if (last) *ticket(p.self) = 0;
        __threadfence();
    }
    __syncthreads();
    if (!last) return;
    // Every chunk's count is in: each chunk's rows start past the chunks
    // before it.
    if (threadIdx.x < static_cast<unsigned>(p.ranks)) {
        const int d = threadIdx.x;
        int run = 0;
        for (int c = 0; c < static_cast<int>(gridDim.x); ++c) {
            const int n = __ldcg(&p.base[c * kMaxRanks + d]);
            p.base[c * kMaxRanks + d] = run;
            run += n;
        }
        peer::st_release_i32(&inbox(p.group.signals[d])->sent[p.rank], run);
    }
    peer::barrier(p.group, p.self, p.rank, p.ranks, kSlotCount);
    if (threadIdx.x < static_cast<unsigned>(p.ranks)) {
        const int d = threadIdx.x;
        int off = 0;
        for (int s = 0; s < p.rank; ++s) off += peer::ld_acquire_i32(&inbox(p.group.signals[d])->sent[s]);
        p.offs[d] = off;
    }
}

__global__ void __launch_bounds__(kSendThreads) dispatch_send(const DispatchParams p) {
    __shared__ int4* dst[kMaxRanks];
    __shared__ long long row_of[kMaxRanks];
    __shared__ int ways;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (int t = blockIdx.x; t < p.tokens; t += gridDim.x) {
        if (!live(p, t)) continue;
        // The destinations this token goes to, packed: warp 0 asks each rank.
        if (warp == 0) {
            const int at = lane < p.ranks ? p.pos[t * p.ranks + lane] : -1;
            const unsigned to = __ballot_sync(~0u, at >= 0);
            if (at >= 0) {
                const long long row =
                    p.offs[lane] + p.base[t / kCountThreads * kMaxRanks + lane] + at;
                dst[__popc(to & ((1u << lane) - 1u))] = p.rows[lane] + row * p.packs;
                row_of[lane] = row;
            } else if (lane < kMaxRanks) {
                row_of[lane] = -1;
            }
            if (lane == 0) ways = __popc(to);
        }
        __syncthreads();
        // Warp d writes the token's routes, weights and origin into rank d's
        // landing, while every thread copies the row into each.
        if (warp < p.ranks && row_of[warp] >= 0 && lane < p.top_k) {
            const int d = warp;
            const long long row = row_of[d];
            const int ph = physical(p, t, lane);
            const bool here = ph >= 0 && ph / p.slots == d;
            p.local[d][row * p.top_k + lane] = here ? ph % p.slots : -1;
            if (p.weights != nullptr)
                p.row_weights[d][row * p.top_k + lane] = here ? p.weights[t * p.top_k + lane] : 0.f;
            if (lane == 0) p.origin[d][row] = p.rank * p.tokens + t;
        }
        peer::fan_copy(dst, ways, p.x + static_cast<long long>(t) * p.packs, p.packs, threadIdx.x,
                       kSendThreads);
        __syncthreads();
    }
    // The rows reach the peers before `dispatch_settle`'s release says so:
    // a kernel boundary orders them on this device, the fence across it. A
    // block strides over the tokens so each thread waits on it once.
    __threadfence_system();
}

__global__ void __launch_bounds__(kCountThreads) dispatch_settle(const DispatchParams p) {
    __shared__ int received;
    __shared__ int tally[kSettleSlots];
    int* counts = p.slots <= kSettleSlots ? tally : p.counts;
    peer::barrier(p.group, p.self, p.rank, p.ranks, kSlotSettle);
    if (threadIdx.x == 0) {
        int n = 0;
        for (int s = 0; s < p.ranks; ++s) n += peer::ld_acquire_i32(&inbox(p.self)->sent[s]);
        received = n;
    }
    for (int s = threadIdx.x; s < p.slots; s += blockDim.x) counts[s] = 0;
    __syncthreads();
    i32* local = p.local[p.rank];
    const int cap = p.ranks * p.tokens;
    const int routes = received * p.top_k;
    for (int i = threadIdx.x; i < routes; i += kCountThreads * kSettleLoads) {
        int slot[kSettleLoads];
#pragma unroll
        for (int u = 0; u < kSettleLoads; ++u)
            slot[u] = i + u * kCountThreads < routes ? local[i + u * kCountThreads] : -1;
#pragma unroll
        for (int u = 0; u < kSettleLoads; ++u)
            if (slot[u] >= 0) atomicAdd(&counts[slot[u]], 1);
    }
    for (int i = routes + threadIdx.x; i < cap * p.top_k; i += blockDim.x) local[i] = -1;
    for (int r = received + threadIdx.x; r < cap; r += blockDim.x) p.origin[p.rank][r] = -1;
    __syncthreads();
    for (int s = threadIdx.x; s < p.slots; s += blockDim.x) {
        if (counts == tally) p.counts[s] = tally[s];
        if (p.loads != nullptr) p.loads[s] += static_cast<u32>(counts[s]);
    }
}

}  // namespace pie::collective

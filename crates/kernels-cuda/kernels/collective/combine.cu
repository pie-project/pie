#include "prim/device.cuh"
#include "prim/peer.cuh"

namespace pie::collective {

using peer::kMaxRanks;

// The way back from `dispatch.cu`: each received row's expert outputs go to
// the rank and token they came from, route j of token t landing at row
// t * top_k + j of that rank's `routed`, as the routes were laid before the
// dispatch. Every live route is held by exactly one rank, so each live row of
// `routed` is written once; a padding token's rows are left as they are. The
// outputs are unweighted: weighting is the caller's sum, as without EP.
//
// Two launches on each rank's stream: `combine_send` (a warp a received
// row at a time) copies, `combine_settle` (one block) meets the ranks so
// every rank's rows have landed before `routed` is read. Nothing opens with
// a barrier: the dispatch before it met the ranks after every earlier read
// of `routed`.

constexpr int kSlotCombine = 2;
constexpr int kSendWarps = 8;

struct CombineParams {
    const int4* y;
    const i32* origin;
    const i32* local;
    int4* routed[kMaxRanks];
    peer::Group group;
    peer::Signals* self;
    i32 rank;
    i32 ranks;
    i32 tokens;
    i32 packs;
    i32 top_k;
};

__global__ void __launch_bounds__(32 * kSendWarps) combine_send(const CombineParams p) {
    const int lane = threadIdx.x & 31;
    const int cap = p.ranks * p.tokens;
    for (long long i = blockIdx.x * kSendWarps + (threadIdx.x >> 5); i < cap;
         i += static_cast<long long>(gridDim.x) * kSendWarps) {
        const int o = p.origin[i];
        if (o < 0) continue;
        const int s = o / p.tokens;
        const long long t = o - s * p.tokens;
        // Each live route's output, its loads in flight across the warp.
        unsigned live = __ballot_sync(~0u, lane < p.top_k && p.local[i * p.top_k + lane] >= 0);
        for (; live != 0; live &= live - 1) {
            const int j = __ffs(live) - 1;
            int4* back = p.routed[s] + (t * p.top_k + j) * p.packs;
            peer::fan_copy(&back, 1, p.y + (i * p.top_k + j) * p.packs, p.packs, lane, 32);
        }
    }
    // As `dispatch_send`: the rows reach the peers before the settle's
    // release. A warp strides over the rows so each thread waits on it once.
    __threadfence_system();
}

__global__ void __launch_bounds__(32) combine_settle(const CombineParams p) {
    peer::barrier(p.group, p.self, p.rank, p.ranks, kSlotCombine);
}

}  // namespace pie::collective

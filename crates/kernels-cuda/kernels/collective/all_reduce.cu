#include <cuda_bf16.h>
#include <cstdint>

#include "collective/p2p.cuh"

namespace pie::collective {

using DType = __nv_bfloat16;

// One-stage reduce over the ranks' stages, which every rank of the group maps
// directly (one process, peer access on). The barriers fence the stages: the
// opening one waits for every rank's copy into its stage, the closing one for
// every rank's reads of it, so the next call may overwrite it.
template <class T, int ngpus>
__global__ void __launch_bounds__(512, 1) all_reduce_peers(
    const p2p::RankData* __restrict__ stages, const p2p::RankSignals* __restrict__ signals,
    p2p::Signal* self, T* __restrict__ result, int rank, int packs)
{
    using P = typename p2p::packed_t<T>::P;
    using A = typename p2p::packed_t<T>::A;
    const p2p::RankSignals sg = *signals;
    const P* ptrs[ngpus];
#pragma unroll
    for (int i = 0; i < ngpus; i++) ptrs[i] = static_cast<const P*>(stages->ptrs[i]);
    p2p::multi_gpu_barrier<ngpus, true>(sg, self, rank);
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < packs; idx += gridDim.x * blockDim.x) {
        reinterpret_cast<P*>(result)[idx] = p2p::packed_reduce<P, ngpus, A>(ptrs, idx);
    }
    p2p::multi_gpu_barrier<ngpus, false>(sg, self, rank);
}

// `rows` rows of every rank's `width`-pack shard, laid side by side in rank
// order: y[r, k * width + w] = stage_k[r, w].
template <class T, int ngpus>
__global__ void __launch_bounds__(512, 1) all_gather_peers(
    const p2p::RankData* __restrict__ stages, const p2p::RankSignals* __restrict__ signals,
    p2p::Signal* self, T* __restrict__ y, int rank, int rows, int width)
{
    using P = typename p2p::packed_t<T>::P;
    const p2p::RankSignals sg = *signals;
    p2p::multi_gpu_barrier<ngpus, true>(sg, self, rank);
    const long long shard = static_cast<long long>(rows) * width;
    const long long total = shard * ngpus;
    for (long long idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += static_cast<long long>(gridDim.x) * blockDim.x) {
        const long long k = idx / shard;
        const long long at = idx - k * shard;
        const long long r = at / width;
        const long long w = at - r * width;
        reinterpret_cast<P*>(y)[(r * ngpus + k) * width + w] =
            static_cast<const P*>(stages->ptrs[k])[at];
    }
    p2p::multi_gpu_barrier<ngpus, false>(sg, self, rank);
}


// The group's open checks the path before any collective spins on it. Each
// rank posts a token to every peer's inbox (the tail of its signal block),
// then looks once for the peers' tokens and the word each peer wrote at the
// head of its stage. Run twice with every device synchronized in between, the
// second run finds every token the link delivered; nothing here waits, so a
// link that does not carry peer traffic leaves the context usable.
__global__ void peers_answer(
    const p2p::RankData* __restrict__ stages, const p2p::RankSignals* __restrict__ signals,
    p2p::Signal* self, int rank, int ngpus, unsigned* status)
{
    constexpr int kInbox = 4096 - 64;
    auto inbox = [](p2p::Signal* sg) {
        return reinterpret_cast<p2p::FlagType*>(reinterpret_cast<char*>(sg) + kInbox);
    };
    for (int k = 0; k < ngpus; k++) {
        if (k != rank) p2p::st_flag_release(inbox(signals->signals[k]) + rank, 1u);
    }
    unsigned verdict = 0;
    for (int k = 0; k < ngpus; k++) {
        if (k == rank) continue;
        const unsigned want = 0x01010101u * static_cast<unsigned>(0x40 + k);
        if (p2p::ld_flag_acquire(inbox(self) + k) != 1u) verdict = 1;
        else if (*static_cast<const volatile unsigned*>(stages->ptrs[k]) != want) verdict = 2;
    }
    *status = verdict;
}

}

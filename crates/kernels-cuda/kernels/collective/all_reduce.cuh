#include <cuda_bf16.h>
#include <cstdint>

#include "flashinfer/comm/vllm_custom_all_reduce.cuh"
#include "flashinfer/comm/trtllm_allreduce_fusion.cuh"

namespace pie::collective {

using DType = __nv_bfloat16;

// One-stage reduce over the ranks' stages, which every rank of the group maps
// directly (one process, peer access on). The barriers fence the stages: the
// opening one waits for every rank's copy into its stage, the closing one for
// every rank's reads of it, so the next call may overwrite it.
template <class T, int ngpus>
__global__ void __launch_bounds__(512, 1) all_reduce_peers(
    const vllm::RankData* __restrict__ stages, const vllm::RankSignals* __restrict__ signals,
    vllm::Signal* self, T* __restrict__ result, int rank, int packs)
{
    using P = typename vllm::packed_t<T>::P;
    using A = typename vllm::packed_t<T>::A;
    const vllm::RankSignals sg = *signals;
    const P* ptrs[ngpus];
#pragma unroll
    for (int i = 0; i < ngpus; i++) ptrs[i] = static_cast<const P*>(stages->ptrs[i]);
    vllm::multi_gpu_barrier<ngpus, true>(sg, self, rank);
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < packs; idx += gridDim.x * blockDim.x) {
        reinterpret_cast<P*>(result)[idx] = vllm::packed_reduce<P, ngpus, A>(ptrs, idx);
    }
    vllm::multi_gpu_barrier<ngpus, false>(sg, self, rank);
}

// `rows` rows of every rank's `width`-pack shard, laid side by side in rank
// order: y[r, k * width + w] = stage_k[r, w].
template <class T, int ngpus>
__global__ void __launch_bounds__(512, 1) all_gather_peers(
    const vllm::RankData* __restrict__ stages, const vllm::RankSignals* __restrict__ signals,
    vllm::Signal* self, T* __restrict__ y, int rank, int rows, int width)
{
    using P = typename vllm::packed_t<T>::P;
    const vllm::RankSignals sg = *signals;
    vllm::multi_gpu_barrier<ngpus, true>(sg, self, rank);
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
    vllm::multi_gpu_barrier<ngpus, false>(sg, self, rank);
}

}

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

// One kernel, no staging copy: each rank stores its own packs straight into
// every peer's push slots, one barrier, then sums the slots in rank order.
// A block owns packs [block * 512, block * 512 + 512) whatever the message,
// and its slots alternate on the block's barrier count, so a rank's next
// store into a slot waits behind the barrier its peer passes only after
// reading that slot's last message: no closing barrier.
template <class T, int ngpus>
__global__ void __launch_bounds__(512, 1) all_reduce_push(
    const vllm::RankData* __restrict__ stages, const vllm::RankSignals* __restrict__ signals,
    vllm::Signal* self, T* __restrict__ result, int rank, int packs, long long push,
    int slot_packs)
{
    using P = typename vllm::packed_t<T>::P;
    using A = typename vllm::packed_t<T>::A;
    const vllm::RankSignals sg = *signals;
    const int parity = (self->self_counter[blockIdx.x][0] + 1) & 1;
    const int idx = blockIdx.x * 512 + threadIdx.x;
    const bool live = idx < packs;
    P mine;
    if (live) {
        mine = reinterpret_cast<const P*>(result)[idx];
#pragma unroll
        for (int k = 0; k < ngpus; k++) {
            if (k == rank) continue;
            P* slots = reinterpret_cast<P*>(static_cast<char*>(stages->ptrs[k]) + push);
            slots[(parity * ngpus + rank) * slot_packs + idx] = mine;
        }
    }
    vllm::multi_gpu_barrier<ngpus, false, true>(sg, self, rank);
    if (live) {
        const P* slots = reinterpret_cast<const P*>(static_cast<const char*>(stages->ptrs[rank]) + push);
        A acc;
#pragma unroll
        for (int k = 0; k < ngpus; k++) {
            P v;
            if (k == rank) {
                v = mine;
            } else {
                const uint4 raw = __ldcg(reinterpret_cast<const uint4*>(slots + (parity * ngpus + k) * slot_packs + idx));
                v = *reinterpret_cast<const P*>(&raw);
            }
            const A f = vllm::upcast(v);
            if (k == 0) {
                acc = f;
            } else {
                vllm::packed_assign_add(acc, f);
            }
        }
        reinterpret_cast<P*>(result)[idx] = vllm::downcast<P>(acc);
    }
}

// `all_gather_peers` through the push slots: a rank stores its shard straight
// into every peer's slot (stores cross the link as posted writes, where the
// stage reads wait out its latency), one barrier, then lays every rank's
// shard out of its own slots. Blocks own packs as `all_reduce_push`'s do, so
// the two share the slots and their alternation.
template <class T, int ngpus>
__global__ void __launch_bounds__(512, 1) all_gather_push(
    const vllm::RankData* __restrict__ stages, const vllm::RankSignals* __restrict__ signals,
    vllm::Signal* self, const T* __restrict__ x, T* __restrict__ y, int rank, int rows,
    int width, long long push, int slot_packs)
{
    using P = typename vllm::packed_t<T>::P;
    const vllm::RankSignals sg = *signals;
    const int parity = (self->self_counter[blockIdx.x][0] + 1) & 1;
    const int idx = blockIdx.x * 512 + threadIdx.x;
    const bool live = idx < rows * width;
    P mine;
    if (live) {
        mine = reinterpret_cast<const P*>(x)[idx];
#pragma unroll
        for (int k = 0; k < ngpus; k++) {
            if (k == rank) continue;
            P* slots = reinterpret_cast<P*>(static_cast<char*>(stages->ptrs[k]) + push);
            slots[(parity * ngpus + rank) * slot_packs + idx] = mine;
        }
    }
    vllm::multi_gpu_barrier<ngpus, false, true>(sg, self, rank);
    if (live) {
        const P* slots = reinterpret_cast<const P*>(static_cast<const char*>(stages->ptrs[rank]) + push);
        const int r = idx / width;
        const int w = idx - r * width;
#pragma unroll
        for (int k = 0; k < ngpus; k++) {
            P v;
            if (k == rank) {
                v = mine;
            } else {
                const uint4 raw = __ldcg(reinterpret_cast<const uint4*>(slots + (parity * ngpus + k) * slot_packs + idx));
                v = *reinterpret_cast<const P*>(&raw);
            }
            reinterpret_cast<P*>(y)[(static_cast<long long>(r) * ngpus + k) * width + w] = v;
        }
    }
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


// The host path, for a group whose devices cannot reach one another (no P2P,
// or a link that does not carry it): every rank stores its message into its
// slot of one pinned, mapped host block and raises its flag there; a peer
// that sees the flag reads the slot across PCIe. A block owns the packs
// `blockIdx.x + k * gridDim.x * 512`, its own flag a rank, and its own count
// on the device (`counts[block]`): the call's epoch. Slots alternate on the
// epoch's parity, so a rank's store of call e + 2 waits behind call e + 1,
// which no peer enters before it finished reading call e: no closing wait.
struct HostDoor {
    unsigned* flags;     // host: [rank][block]
    unsigned* counts;    // this rank's device memory: [block]
    char* slots;         // host: [parity][rank][slot_bytes]
    long long slot_bytes;
};

constexpr int kHostBlocks = 36;

__device__ __forceinline__ unsigned host_epoch(const HostDoor& door)
{
    __shared__ unsigned epoch;
    if (threadIdx.x == 0) epoch = door.counts[blockIdx.x] + 1u;
    __syncthreads();
    return epoch;
}

template <int ngpus>
__device__ __forceinline__ void host_meet(const HostDoor& door, int rank, unsigned epoch)
{
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        vllm::st_flag_release(door.flags + rank * kHostBlocks + blockIdx.x, epoch);
        door.counts[blockIdx.x] = epoch;
    }
    if (threadIdx.x < ngpus && threadIdx.x != rank) {
        unsigned* flag = door.flags + threadIdx.x * kHostBlocks + blockIdx.x;
        while (static_cast<int>(vllm::ld_flag_acquire(flag) - epoch) < 0) {
        }
    }
    __syncthreads();
}

template <class T, int ngpus>
__global__ void __launch_bounds__(512, 1) all_reduce_host(
    unsigned* flags, unsigned* counts, char* slots, long long slot_bytes,
    T* __restrict__ result, int rank, int packs)
{
    const HostDoor door{flags, counts, slots, slot_bytes};
    using P = typename vllm::packed_t<T>::P;
    using A = typename vllm::packed_t<T>::A;
    const unsigned epoch = host_epoch(door);
    const long long parity = epoch & 1u;
    const int stride = gridDim.x * blockDim.x;
    P* mine = reinterpret_cast<P*>(door.slots + (parity * ngpus + rank) * door.slot_bytes);
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < packs; idx += stride) {
        mine[idx] = reinterpret_cast<const P*>(result)[idx];
    }
    host_meet<ngpus>(door, rank, epoch);
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < packs; idx += stride) {
        A acc;
#pragma unroll
        for (int k = 0; k < ngpus; k++) {
            P v;
            if (k == rank) {
                v = reinterpret_cast<const P*>(result)[idx];
            } else {
                const uint4 raw = (reinterpret_cast<const uint4*>(
                    door.slots + (parity * ngpus + k) * door.slot_bytes)[idx]);
                v = *reinterpret_cast<const P*>(&raw);
            }
            const A f = vllm::upcast(v);
            if (k == 0) {
                acc = f;
            } else {
                vllm::packed_assign_add(acc, f);
            }
        }
        reinterpret_cast<P*>(result)[idx] = vllm::downcast<P>(acc);
    }
}

// `rows` rows of every rank's `width`-pack shard side by side in rank order,
// through the host slots: y[r, k * width + w] = x_k[r, w].
template <class T, int ngpus>
__global__ void __launch_bounds__(512, 1) all_gather_host(
    unsigned* flags, unsigned* counts, char* slots, long long slot_bytes,
    const T* __restrict__ x, T* __restrict__ y, int rank, int rows, int width)
{
    const HostDoor door{flags, counts, slots, slot_bytes};
    using P = typename vllm::packed_t<T>::P;
    const unsigned epoch = host_epoch(door);
    const long long parity = epoch & 1u;
    const long long shard = static_cast<long long>(rows) * width;
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    P* mine = reinterpret_cast<P*>(door.slots + (parity * ngpus + rank) * door.slot_bytes);
    for (long long idx = blockIdx.x * blockDim.x + threadIdx.x; idx < shard; idx += stride) {
        mine[idx] = reinterpret_cast<const P*>(x)[idx];
    }
    host_meet<ngpus>(door, rank, epoch);
    for (long long idx = blockIdx.x * blockDim.x + threadIdx.x; idx < shard * ngpus; idx += stride) {
        const long long k = idx / shard;
        const long long at = idx - k * shard;
        const long long r = at / width;
        const long long w = at - r * width;
        P v;
        if (k == rank) {
            v = reinterpret_cast<const P*>(x)[at];
        } else {
            const uint4 raw = (reinterpret_cast<const uint4*>(
                door.slots + (parity * ngpus + k) * door.slot_bytes)[at]);
            v = *reinterpret_cast<const P*>(&raw);
        }
        reinterpret_cast<P*>(y)[(r * ngpus + k) * width + w] = v;
    }
}

// The group's open checks the path before any collective spins on it. Each
// rank posts a token to every peer's inbox (the tail of its signal block),
// then looks once for the peers' tokens and the word each peer wrote at the
// head of its stage. Run twice with every device synchronized in between, the
// second run finds every token the link delivered; nothing here waits, so a
// link that does not carry peer traffic leaves the context usable.
__global__ void peers_answer(
    const vllm::RankData* __restrict__ stages, const vllm::RankSignals* __restrict__ signals,
    vllm::Signal* self, int rank, int ngpus, unsigned* status)
{
    constexpr int kInbox = 4096 - 64;
    auto inbox = [](vllm::Signal* sg) {
        return reinterpret_cast<vllm::FlagType*>(reinterpret_cast<char*>(sg) + kInbox);
    };
    for (int k = 0; k < ngpus; k++) {
        if (k != rank) vllm::st_flag_release(inbox(signals->signals[k]) + rank, 1u);
    }
    unsigned verdict = 0;
    for (int k = 0; k < ngpus; k++) {
        if (k == rank) continue;
        const unsigned want = 0x01010101u * static_cast<unsigned>(0x40 + k);
        if (vllm::ld_flag_acquire(inbox(self) + k) != 1u) verdict = 1;
        else if (*static_cast<const volatile unsigned*>(stages->ptrs[k]) != want) verdict = 2;
    }
    *status = verdict;
}

}

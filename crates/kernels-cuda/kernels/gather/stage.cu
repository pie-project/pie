#include "prim/device.cuh"

namespace pie::gather {

// One copy into a staging slot: `bytes` from `src` (mapped pinned host
// memory, read zero-copy over the bus, or device memory) to `dst`, then
// `dst` stored at `entry` (an expert or row-block table entry), when not 0.
struct alignas(16) StageCopy {
    u64 src;
    u64 dst;
    u64 bytes;
    u64 entry;
};

constexpr int kStageUnroll = 4;

// Copies `count[0]` of `list`'s copies (at most `max`): copy `blockIdx.y`
// split across `gridDim.x` blocks, 16-byte loads where its ends are on 16
// bytes, `kStageUnroll` of them in flight a thread. The list and its count
// live on the device, so a graph captured once replays any list up to `max`.
__global__ void stage_copies(
    const StageCopy* __restrict__ list,
    const u32* __restrict__ count,
    int max)
{
    const int at = static_cast<int>(blockIdx.y);
    if (at >= max || at >= static_cast<int>(count[0])) return;
    const StageCopy c = list[at];
    const u64 lanes = static_cast<u64>(gridDim.x) * blockDim.x;
    const u64 lane = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
    const bool vec = ((c.src | c.dst | c.bytes) & 15u) == 0;
    if (vec) {
        const uint4* src = reinterpret_cast<const uint4*>(c.src);
        uint4* dst = reinterpret_cast<uint4*>(c.dst);
        const u64 words = c.bytes / 16;
        u64 w = lane;
        for (; w + (kStageUnroll - 1) * lanes < words; w += kStageUnroll * lanes) {
            uint4 v[kStageUnroll];
#pragma unroll
            for (int u = 0; u < kStageUnroll; ++u) v[u] = src[w + u * lanes];
#pragma unroll
            for (int u = 0; u < kStageUnroll; ++u) dst[w + u * lanes] = v[u];
        }
        for (; w < words; w += lanes) dst[w] = src[w];
    } else {
        const u8* src = reinterpret_cast<const u8*>(c.src);
        u8* dst = reinterpret_cast<u8*>(c.dst);
        for (u64 b = lane; b < c.bytes; b += lanes) dst[b] = src[b];
    }
    if (c.entry != 0 && blockIdx.x == 0 && threadIdx.x == 0) {
        *reinterpret_cast<u64*>(c.entry) = c.dst;
    }
}

}

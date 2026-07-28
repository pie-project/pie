#include "kernels/embed.hpp"

#include <cuda_bf16.h>

#include <cstdint>

namespace pie_cuda_driver::kernels {

namespace {

// One block per token. Threads stride across `hidden`. Bounds-clamp the
// token id so a runaway wire payload can't OOB-read. (Out-of-vocab → 0 row.)
//
// `VEC` gathers eight bf16 per thread through a 16-byte load. The row this
// reads is a random offset into a table that is the largest tensor in the
// model, so the access is a cold TLB miss whose latency only a wide grid
// hides: at decode the token-per-block form issued 24 dependent 2-byte loads
// from 8 blocks and ran at 8 GB/s. Flattening (token, chunk) into one grid
// gives the memory system every row at once. Pure copy either way.
template <bool VEC>
__global__ void embed_bf16_kernel(
    const std::int32_t* __restrict__ token_ids,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ y,
    int hidden, int vocab, int num_tokens, int per_row)
{
    const int idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= num_tokens * per_row) return;
    const int n = idx / per_row;
    const int h = idx % per_row;

    const std::int32_t tid_raw = token_ids[n];
    const int tid = (tid_raw >= 0 && tid_raw < vocab) ? tid_raw : 0;
    const __nv_bfloat16* row = weight + static_cast<long long>(tid) * hidden;
    __nv_bfloat16* out = y + static_cast<long long>(n) * hidden;

    if constexpr (VEC) {
        reinterpret_cast<float4*>(out)[h] =
            reinterpret_cast<const float4*>(row)[h];
    } else {
        out[h] = row[h];
    }
}

__global__ void embed_bf16_vocab_shard_kernel(
    const std::int32_t* __restrict__ token_ids,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ y,
    int hidden, int local_vocab, int vocab_offset)
{
    const int n = blockIdx.x;
    const std::int32_t tid_raw = token_ids[n];
    const int local_tid = tid_raw - vocab_offset;
    const bool in_shard = local_tid >= 0 && local_tid < local_vocab;
    const __nv_bfloat16* row =
        weight + static_cast<long long>(in_shard ? local_tid : 0) * hidden;
    __nv_bfloat16* out = y + static_cast<long long>(n) * hidden;

    for (int h = threadIdx.x; h < hidden; h += blockDim.x) {
        out[h] = in_shard ? row[h] : __float2bfloat16(0.0f);
    }
}

}  // namespace

void launch_embed_bf16(
    const std::int32_t* token_ids,
    const void* weight,
    void* y,
    int num_tokens, int hidden, int vocab,
    cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    if (num_tokens <= 0 || hidden <= 0) return;
    const bool vec = (hidden % 8) == 0 &&
                     (reinterpret_cast<std::uintptr_t>(weight) % 16) == 0 &&
                     (reinterpret_cast<std::uintptr_t>(y) % 16) == 0;
    const int per_row = vec ? hidden / 8 : hidden;
    const long long total = static_cast<long long>(num_tokens) * per_row;
    dim3 grid(static_cast<unsigned>((total + BLOCK - 1) / BLOCK));
    dim3 block(BLOCK);
    if (vec) {
        embed_bf16_kernel<true><<<grid, block, 0, stream>>>(
            token_ids,
            static_cast<const __nv_bfloat16*>(weight),
            static_cast<__nv_bfloat16*>(y),
            hidden, vocab, num_tokens, per_row);
    } else {
        embed_bf16_kernel<false><<<grid, block, 0, stream>>>(
            token_ids,
            static_cast<const __nv_bfloat16*>(weight),
            static_cast<__nv_bfloat16*>(y),
            hidden, vocab, num_tokens, per_row);
    }
}

void launch_embed_bf16_vocab_shard(
    const std::int32_t* token_ids,
    const void* weight,
    void* y,
    int num_tokens, int hidden, int local_vocab, int vocab_offset,
    cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    dim3 grid(num_tokens);
    dim3 block(BLOCK);
    embed_bf16_vocab_shard_kernel<<<grid, block, 0, stream>>>(
        token_ids,
        static_cast<const __nv_bfloat16*>(weight),
        static_cast<__nv_bfloat16*>(y),
        hidden, local_vocab, vocab_offset);
}

}  // namespace pie_cuda_driver::kernels

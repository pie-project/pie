//===-- embed.cu - the token-embedding launchers ---------------------===//
//
// Two host launchers and not one `__global__`: the device text is in
// `layout/embed.cuh`, which this file includes so the archive and the JIT
// header set hold the SAME definition rather than two that drift.
//
// `embed_bf16` keeps its host alignment test -- the `VEC` choice cannot be a
// row, because no `Source` produces "is this pointer 16-byte aligned".
// `embed_bf16_vocab_shard` is also a row, and the launcher stays anyway: this
// migration extracts device text and adds rows, it deletes nothing.
//
//===----------------------------------------------------------------------===//

// The scalar layer and the fixed-width integer names, out of the prelude.
#include "pie_device.cuh"
#include "layout/embed.hpp"

// The `__global__`s these launchers fire. ONE definition of each.
#include "layout/embed.cuh"

#include <cstdint>

namespace pie_cuda_driver::kernels::layout {

void embed_bf16(
    const device::i32* token_ids,
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
        device::embed<true><<<grid, block, 0, stream>>>(
            token_ids,
            static_cast<const device::bf16*>(weight),
            static_cast<device::bf16*>(y),
            hidden, vocab, num_tokens, per_row);
    } else {
        device::embed<false><<<grid, block, 0, stream>>>(
            token_ids,
            static_cast<const device::bf16*>(weight),
            static_cast<device::bf16*>(y),
            hidden, vocab, num_tokens, per_row);
    }
}

void embed_bf16_vocab_shard(
    const device::i32* token_ids,
    const void* weight,
    void* y,
    int num_tokens, int hidden, int local_vocab, int vocab_offset,
    cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    dim3 grid(num_tokens);
    dim3 block(BLOCK);
    device::embed_vocab_shard<device::bf16><<<grid, block, 0, stream>>>(
        token_ids,
        static_cast<const device::bf16*>(weight),
        static_cast<device::bf16*>(y),
        hidden, local_vocab, vocab_offset);
}

}  // namespace pie_cuda_driver::kernels::layout

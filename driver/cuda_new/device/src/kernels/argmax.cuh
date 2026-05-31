#pragma once

// Per-row greedy argmax over [num_rows, vocab] bf16 logits → [num_rows] i32
// token ids. Used as the greedy sampler (temperature=0). Launcher declaration;
// the kernel bodies live in argmax.cu, lifted verbatim from
// driver/cuda/src/kernels/argmax.cu (the base `launch_argmax_bf16` variant and
// its dependency closure). The compact_scatter / partitioned_pairs / lm_head /
// fp32 / tile_pair / masked_embedding / topk_centroids / gemv / int8 variants
// are lifted later as the forward bodies that need them land.

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_device::kernels {

// out[r] = argmax_i logits[r, i]. Tie-break: lowest index wins
// (matches torch.argmax / numpy.argmax).
// logits: [num_rows, vocab] bf16 row-major; token_ids: [num_rows] i32.
void argmax_bf16(const void* logits, std::int32_t* token_ids,
                 int num_rows, int vocab, cudaStream_t stream);

}  // namespace pie_cuda_device::kernels

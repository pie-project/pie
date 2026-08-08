// Attention-state merge: folds the per-split partial outputs and their log-sum-exps
// back into one result.
//
// This lives in its own translation unit rather than beside the sm90 prefill
// because its CALLERS are not sm90-only. It used to sit in
// `attention_flashinfer_hopper.cu`, on the reasoning that the KV split
// producing its inputs was the sm90 prefill's, so nothing could reach it on an
// architecture where that unit is stubbed out. That reasoning held on sm_80
// and sm_90 and is false on sm_100: the DECODE KV-split path
// (`use_full_split` in mixtral.cpp, and the two sliding-window sites in
// gemma4.cpp) calls `dispatch_attention_flashinfer_decode_bf16` -- which is
// built on every architecture -- and then merges. On Blackwell the dispatch
// succeeded and the merge threw, poisoning the driver on the first fire and
// taking gpt-oss and gemma-4 down with it.
//
// It only ever needed `cascade.cuh`, which carries no architecture gate; the
// Hopper unit's other includes are what made that file arch-specific. So the
// fix is a move, not a port.
#include "ops/attention_flashinfer_hopper.hpp"

#include <cstdint>

#include <cuda_bf16.h>

#include <flashinfer/attention/cascade.cuh>

#include "cuda_check.hpp"

namespace pie_cuda_driver::ops {

void merge_attention_states_bf16(
    const void* v, const float* s,
    void* v_merged, float* s_merged,
    int num_index_sets, int seq_len, int num_heads, int head_dim,
    cudaStream_t stream) {
    if (num_index_sets <= 1 || seq_len <= 0 || num_heads <= 0) return;
    CUDA_CHECK((::flashinfer::MergeStates<__nv_bfloat16, __nv_bfloat16>(
        const_cast<__nv_bfloat16*>(static_cast<const __nv_bfloat16*>(v)),
        const_cast<float*>(s),
        static_cast<__nv_bfloat16*>(v_merged), s_merged,
        static_cast<std::uint32_t>(num_index_sets),
        static_cast<std::uint32_t>(seq_len),
        static_cast<std::uint32_t>(num_heads),
        static_cast<std::uint32_t>(head_dim), stream)));
}

}  // namespace pie_cuda_driver::ops

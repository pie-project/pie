#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "attention_workspace_view.hpp"

namespace pie_cuda_driver::kernels::attn {

struct HopperPrefillPlan {
    std::int64_t qo_tile_indices_offset = 0;
    std::int64_t qo_indptr_offset = 0;
    std::int64_t kv_indptr_offset = 0;
    std::int64_t qo_len_offset = 0;
    std::int64_t kv_len_offset = 0;
    std::int64_t head_indices_offset = 0;
    std::int64_t work_indptr_offset = 0;
    std::int64_t batch_indices_offset = 0;
    bool same_schedule_for_all_heads = false;
    int total_tokens = 0;
    int num_requests = 0;
    int num_q_heads = 0;
    int num_kv_heads = 0;
    int head_dim = 0;
    int page_size = 0;
    int window_left = -1;
    bool causal = true;
    bool valid = false;
};

bool hopper_prefill_supported(int head_dim,
                              int window_left,
                              int total_tokens,
                              int num_requests);
void plan_attention_flashinfer_prefill_sm90_bf16(
    HopperPrefillPlan& plan,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* kv_page_indptr_h,
    const std::uint32_t* kv_last_page_lens_h,
    int total_tokens,
    int num_requests,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    AttentionWorkspaceView workspace,
    cudaStream_t stream,
    bool enable_cuda_graph,
    bool causal,
    int window_left,
    std::size_t int_base_bytes = 0);

void dispatch_attention_flashinfer_prefill_sm90_bf16(
    const HopperPrefillPlan& plan,
    const void* q,
    void* k_pages,
    void* v_pages,
    void* o,
    const std::uint32_t* kv_page_indices_d,
    AttentionWorkspaceView workspace,
    cudaStream_t stream,
    float logits_soft_cap = 0.f,
    float sm_scale = -1.f,
    float* lse_out = nullptr,
    // Every request reads the same Q row. Used by the KV split, whose
    // pseudo-requests are one query against different slices of the pages --
    // the outputs stay per-request, only the input is shared.
    bool broadcast_q = false);

// DECLARED HERE AND DEFINED NOWHERE, as of the pass that emptied this
// directory of everything but XQA and MLA.
//
// `attention_merge_states.cu` held the only definition — one call to
// flashinfer's cascade `MergeStates` — and it is deleted, because
// `attn::merge_attention_states_bf16` lost its `table::attn` row to §38 and
// the body had been unreachable ever since. The declaration is kept, and
// kept HERE rather than moved, for two reasons: it is the argument list a
// re-added row has to match, and `attention_flashinfer_common.cuh` still
// includes this header (it has no compiler now, but it is the anchor forty
// Rust doc comments cite by line).
//
// ── BOTH REASONS HAVE NOW BEEN OVERTAKEN, AND THE SECOND ONE TWICE ──────────
//
// FIRST: "the argument list a re-added row has to match" is superseded.
// `driver-cuda/src/fire/merge_states.rs` exists and implements BOTH cascade
// launchers in Rust, and it went further than the recipe below — it found
// that the specification named the wrong one. `MergeStates` serves only the
// single-request paths; both batched dispatches call
// `VariableLengthMergeStates`, and folding a ragged batch with a uniform
// chunk count reads another row's partials and returns a wrong answer no
// assertion catches. So the declaration below is no longer the shape to
// match; it is the shape that would have been silently wrong to match.
//
// SECOND: `attention_flashinfer_common.cuh` includes this header and HAS NO
// COMPILER — measured since, and stronger than "now": it has zero `#include`
// consumers anywhere in the workspace, and this header plus
// `attn/attention_flashinfer.hpp` are the only two files whose ONLY includer
// it is. The chain's head is dangling, so this whole subtree is unreachable
// as code and reachable only as text. That also discharges a constraint the
// XQA pass recorded as live — *"`attention_flashinfer_hopper.hpp` is not
// chained to the stub and must outlive it"* — which was true while the
// non-sm90 arm was a compile target and is now true of nothing.
//
// THE ONE THING HERE THAT MUST NOT DIE WITH THIS FILE is the paragraph
// below's last sentence, and it is a measurement rather than a plan: the
// decode KV-split path reaches a merge on EVERY architecture, so whatever
// serves this row next must not be arch-gated. `CMakeLists.txt:1071-1082`
// holds the long form and `CMakeLists.txt` is in this crate, so both copies
// were inside the dying tree. It is now carried in
// `driver-cuda/src/fire/merge_states.rs`, which is not.
//
// A declaration with no definition is legal until something ODR-uses it, and
// nothing does. If you are re-adding this: the row wants a RUST body over
// `flashinfer/attention/cascade.cuh`, which `kernels-cuda-new/csrc/vendor`
// already carries, and it must not be arch-gated — the decode KV-split path
// reaches a merge on every architecture, which is the sm_100 defect the
// deleted file's header recorded at length.
//
// Folds `num_index_sets` partial attention outputs and their log-sum-exps into
// one. Wraps flashinfer's cascade `MergeStates`; `v` is
// [num_index_sets, seq_len, num_heads, head_dim] bf16 and `s` the matching
// [num_index_sets, seq_len, num_heads] floats the dispatch writes to `lse_out`.
void merge_attention_states_bf16(
    const void* v, const float* s,
    void* v_merged, float* s_merged,
    int num_index_sets, int seq_len, int num_heads, int head_dim,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels::attn

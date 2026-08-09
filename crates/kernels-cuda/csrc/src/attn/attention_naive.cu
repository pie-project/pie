// The host launchers, and nothing else. All five `__global__`s and the two
// `__device__` helpers live in `attn/attention_naive.cuh` -- ONE definition,
// read by nvcc here and by NVRTC from the same text at run time.
//
// What stayed behind is everything a `LaunchRule` cannot say: the
// shared-memory budgets, which are sized on a KV extent; the `scale`, which
// is `1/sqrt(head_dim)` computed once on the host; and the FALLBACK in
// `attention_mtp_paged_history_bf16`, which chooses a different kernel when
// the global window will not fit in shared memory. A rule selects a
// rectangle, not a kernel.
//
// `<cuda_bf16.h>` and `<cmath>` went with the device text -- see the header
// for what NVRTC answered when it was asked for them. `sqrtf` here is the
// host's, out of `<cuda_runtime.h>` by way of the `.hpp`.
#include "attn/attention_naive.cuh"
#include "attn/attention_naive.hpp"

namespace pie_cuda_driver::kernels::attn {

namespace {

using bf16 = ::pie_cuda_driver::kernels::device::bf16;

constexpr int BLOCK = device::BLOCK;

}  // namespace

// THREE LAUNCHERS WERE DELETED HERE, and the deletion is the second half of
// a decision `new-horizon.md` §38 left open.
//
// `attention_naive_bf16`, `attention_mtp_history_bf16` and
// `attention_mtp_paged_history_bf16` held one `<<<>>>` each. §38 deleted the
// TABLE ROW for `attn::attention_mtp_paged_history_bf16` and kept the
// launcher, on a keeper that named it "the ONLY caller of
// `attention_mtp_history_bf16`" -- true, and a reason to delete the pair
// together rather than a reason to keep either. `csrc-reachability-audit.py`
// now reports all three unreachable from every shim root, which is the
// measurement §38's keeper was waiting for: the cluster's whole consumer set
// is the cluster.
//
// The fallback in `attention_mtp_paged_history_bf16` -- a host `if` on
// `max_global_tokens + history_steps > 8192` choosing a different kernel --
// went with it, and it is worth recording what it was, because it is the
// same shape as the FlashInfer host dispatch this migration could not state:
// a predicate over two operands selecting between two kernels with different
// shared-memory budgets. No `LaunchRule` says that. It did not need one,
// because nothing called it.
//
// The KERNELS are untouched. `attn/attention_naive.cuh` still carries
// `attn_naive`, `attn_mtp_history` and `attn_mtp_paged_history`, and
// `families::attn::ATTENTION_NAIVE` still compiles them -- a launcher's
// deletion is a deletion of host code, and the device text outlives it.

void mtp_shift_hidden_bf16(
    const void* target_hidden,
    const void* pending_hidden,
    const std::uint32_t* qo_indptr,
    const std::int32_t* slot_ids,
    void* out,
    int total_tokens,
    int num_requests,
    int hidden_size,
    cudaStream_t stream)
{
    if (total_tokens <= 0 || num_requests <= 0 || hidden_size <= 0 ||
        pending_hidden == nullptr) {
        return;
    }
    device::mtp_shift_hidden<bf16><<<total_tokens, BLOCK, 0, stream>>>(
        static_cast<const bf16*>(target_hidden),
        static_cast<const bf16*>(pending_hidden),
        qo_indptr, slot_ids,
        static_cast<bf16*>(out),
        num_requests, hidden_size);
}

void mtp_update_pending_hidden_bf16(
    const void* target_hidden,
    void* pending_hidden,
    const std::uint32_t* qo_indptr,
    const std::int32_t* slot_ids,
    int num_requests,
    int hidden_size,
    cudaStream_t stream)
{
    if (num_requests <= 0 || hidden_size <= 0 || pending_hidden == nullptr) {
        return;
    }
    device::mtp_update_pending_hidden<bf16><<<num_requests, BLOCK, 0, stream>>>(
        static_cast<const bf16*>(target_hidden),
        static_cast<bf16*>(pending_hidden),
        qo_indptr, slot_ids, num_requests, hidden_size);
}

}  // namespace pie_cuda_driver::kernels::attn

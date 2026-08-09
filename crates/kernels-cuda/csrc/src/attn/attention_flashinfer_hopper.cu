#include "attn/attention_flashinfer_hopper.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <cuda_bf16.h>

#include <flashinfer/attention/cascade.cuh>

#include <flashinfer/attention/hopper/default_params.cuh>
#include <flashinfer/attention/hopper/prefill_sm90.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/cutlass_utils.cuh>
#include <flashinfer/layout.cuh>

#include "cuda_check.hpp"

namespace pie_cuda_driver::kernels::attn {

namespace {

using DTypeQ = ::flashinfer::cutlass_dtype_t<__nv_bfloat16>;
using DTypeKV = ::flashinfer::cutlass_dtype_t<__nv_bfloat16>;
using DTypeO = ::flashinfer::cutlass_dtype_t<__nv_bfloat16>;
using IdType = std::int32_t;
using HopperParams =
    ::flashinfer::BatchPrefillPagedParams<DTypeQ, DTypeKV, DTypeO, IdType>;

template <typename T>
inline T* offset_ptr(void* base, std::int64_t off) {
    return reinterpret_cast<T*>(reinterpret_cast<std::uint8_t*>(base) + off);
}

int current_device_major() {
    thread_local int cached_device = -1;
    thread_local int cached_major = 0;
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    if (dev != cached_device) {
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        cached_device = dev;
        cached_major = prop.major;
    }
    return cached_major;
}

template <std::uint32_t HeadDim, bool SameSchedule, bool Softcap, bool Causal,
          bool Window>
cudaError_t dispatch_hopper_prefill(HopperParams& params,
                                    bool enable_pdl,
                                    cudaStream_t stream) {
    using Variant = ::flashinfer::DefaultAttention<Softcap>;
    constexpr auto Mask =
        Causal ? ::flashinfer::MaskMode::kCausal : ::flashinfer::MaskMode::kNone;
    return ::flashinfer::BatchPrefillWithPagedKVCacheDispatched<
        HeadDim,
        HeadDim,
        Mask,
        Window,
        SameSchedule,
        Variant,
        HopperParams>(params, enable_pdl, stream);
}

template <std::uint32_t HeadDim, bool Softcap, bool Causal, bool Window>
cudaError_t dispatch_hopper_prefill_schedule(HopperParams& params,
                                             bool same_schedule,
                                             bool enable_pdl,
                                             cudaStream_t stream) {
    if (same_schedule) {
        return dispatch_hopper_prefill<HeadDim, true, Softcap, Causal, Window>(
            params, enable_pdl, stream);
    }
    return dispatch_hopper_prefill<HeadDim, false, Softcap, Causal, Window>(
        params, enable_pdl, stream);
}

template <bool Softcap, bool Causal, bool Window>
cudaError_t dispatch_hopper_prefill_dim(HopperParams& params,
                                        int head_dim,
                                        bool same_schedule,
                                        bool enable_pdl,
                                        cudaStream_t stream) {
    switch (head_dim) {
        case 128:
            return dispatch_hopper_prefill_schedule<128, Softcap, Causal,
                                                    Window>(
                params, same_schedule, enable_pdl, stream);
        // Gemma-4's sliding layers and Qwen3.6's full-attention layers both
        // attend at 256. They are the shapes this driver serves from FA2
        // today and the ones vLLM serves from FlashAttention.
        case 256:
            return dispatch_hopper_prefill_schedule<256, Softcap, Causal,
                                                    Window>(
                params, same_schedule, enable_pdl, stream);
    }
    return cudaErrorNotSupported;
}

template <bool Softcap, bool Causal>
cudaError_t dispatch_hopper_prefill_dim_window(HopperParams& params,
                                               int head_dim,
                                               bool same_schedule,
                                               bool enable_pdl,
                                               cudaStream_t stream) {
    if (params.window_left >= 0) {
        return dispatch_hopper_prefill_dim<Softcap, Causal, true>(
            params, head_dim, same_schedule, enable_pdl, stream);
    }
    return dispatch_hopper_prefill_dim<Softcap, Causal, false>(
        params, head_dim, same_schedule, enable_pdl, stream);
}

template <bool Softcap>
cudaError_t dispatch_hopper_prefill_dim_mask(HopperParams& params,
                                             int head_dim,
                                             bool same_schedule,
                                             bool causal,
                                             bool enable_pdl,
                                             cudaStream_t stream) {
    if (causal) {
        return dispatch_hopper_prefill_dim_window<Softcap, true>(
            params, head_dim, same_schedule, enable_pdl, stream);
    }
    return dispatch_hopper_prefill_dim_window<Softcap, false>(
        params, head_dim, same_schedule, enable_pdl, stream);
}

}  // namespace

// The shapes this wrapper grew into: head_dim 256, a sliding window, and
// decode-shaped fires. On by default -- measured on gemma-4-26B-A4B at 1k
// context, routing the sliding layers' decode here takes attention from
// 4.19 to 2.73 ms and the model from 122.5 to 144.1 tok/s, output unchanged.
//
// # This was `getenv("PIE_CUDA_DISABLE_HOPPER_EXTENDED")`, read per call
//
// It answers a CAPABILITY -- "does this deployment serve head_dim 256, a
// sliding window, a decode-shaped prefill" -- and a capability answered per
// launch is answered too late. A deployment either supports the path or it
// does not; that is a fact about the deployment, settled before the first
// fire, and a refusal is only visible if it is settled somewhere a refusal
// can be reported from.
//
// So the environment read is gone and the fact is stated instead: the choice
// NAMES ITS DEFAULT. `true` is precisely `getenv(...) == nullptr`, and nothing
// in this repository has ever set the variable, so every deployment that
// exists today gets the identical answer from the identical predicate below.
//
// # Why there is no setter here yet
//
// The load-time shape wants a seam -- the driver holds the fact, states it
// once at load, and a refusal is reportable at the place that can report it.
// A `void set_hopper_extended_shapes(bool)` was written, and then withdrawn,
// because it had no caller: the driver's bridge does not reach this header,
// and its one plausible home (`driver-cuda`'s `Boot`, "every boot knob this
// driver reads, parsed once") is a separate change in a separate crate.
// `driver-cuda`'s `launch_abi.rs` refuses a `void` in an `attn` header that
// is neither a table row nor a decided exception, and it was right to: a
// symbol added ahead of its caller is not a decision, it is a backlog entry
// wearing a signature. When `Boot` grows the fact, the constant below becomes
// that seam and the exception becomes `DriverInternal`. Until then the
// override mechanism is an edit, which is at least an edit a replay can see.
//
// The measurement this did NOT get, said plainly rather than implied: the box
// this was audited on is an L40S, sm_89. `PIE_HAS_SM90` is false there, so
// CMake compiles `attention_flashinfer_hopper_stub.cpp` in place of this file
// and `hopper_prefill_supported` is the stub's `return false` -- the variable
// is not merely unreachable on that hardware, it is not in the binary. The
// two arms it chose between are an FA3 dispatch and the FA2 decode path, and
// neither can be fired here. This file was compiled for sm_90 to check the
// change -- SASS identical before and after -- and that is all the evidence
// there is for it; the fold is deliberately behaviour-preserving because it
// could not be anything else. §36 has the reasoning and the compile.
namespace {
constexpr bool kHopperExtendedShapes = true;
}  // namespace

bool hopper_extended_shapes_enabled() { return kHopperExtendedShapes; }

bool hopper_prefill_supported(int head_dim,
                              int window_left,
                              int total_tokens,
                              int num_requests) {
    if (current_device_major() < 9) return false;
    // A sliding window is a template parameter now, not a disqualifier, and a
    // decode fire (one token per request) is the shape this path most needs to
    // cover -- vLLM serves exactly that from FlashAttention and pays nothing
    // for context growth, where the FA2 decode kernel this replaces pays
    // 2.8 ms per token at 1k on gemma-4.
    if (!hopper_extended_shapes_enabled()) {
        if (window_left >= 0) return false;
        if (total_tokens <= num_requests) return false;
        return head_dim == 128;
    }
    if (total_tokens < num_requests) return false;
    return head_dim == 128 || head_dim == 256;
}
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
    std::size_t int_base_bytes) {
    if (!hopper_prefill_supported(
            head_dim, window_left, total_tokens, num_requests)) {
        throw std::runtime_error("flashinfer sm90 prefill: unsupported shape");
    }
    if (num_kv_heads <= 0 || num_q_heads % num_kv_heads != 0) {
        throw std::runtime_error("flashinfer sm90 prefill: invalid head layout");
    }

    std::vector<IdType> qo_h(num_requests + 1);
    std::vector<IdType> kv_h(num_requests + 1);
    std::vector<IdType> kv_lens_h(num_requests);
    for (int r = 0; r <= num_requests; ++r) {
        qo_h[r] = static_cast<IdType>(qo_indptr_h[r]);
        kv_h[r] = static_cast<IdType>(kv_page_indptr_h[r]);
    }
    for (int r = 0; r < num_requests; ++r) {
        const int pages =
            static_cast<int>(kv_page_indptr_h[r + 1] - kv_page_indptr_h[r]);
        kv_lens_h[r] = pages > 0
            ? static_cast<IdType>((pages - 1) * page_size +
                                  static_cast<int>(kv_last_page_lens_h[r]))
            : 0;
    }

    // Every planner in this driver carves from offset 0 of the shared int
    // workspace. Two plans can coexist there only while their contents are
    // identical -- which is true of the two decode plans gemma-4 builds, and
    // is NOT true of this one. `int_base_bytes` moves this plan's descriptor
    // clear of them; the offsets flashinfer returns are relative to the base
    // it was handed, so they are re-biased below to stay meaningful against
    // `workspace.int_buffer`.
    if (int_base_bytes >= workspace.int_bytes) {
        throw std::runtime_error("flashinfer sm90 prefill: int base past end");
    }
    auto* int_base =
        static_cast<std::uint8_t*>(workspace.int_buffer) + int_base_bytes;
    auto* host_base =
        static_cast<std::uint8_t*>(workspace.page_locked_int) + int_base_bytes;
    ::flashinfer::PrefillPlanSM90Info plan_info;
    CUDA_CHECK(::flashinfer::PrefillSM90Plan<IdType>(
        workspace.float_buffer,
        workspace.float_bytes,
        int_base,
        host_base,
        workspace.int_bytes - int_base_bytes,
        plan_info,
        qo_h.data(),
        kv_h.data(),
        kv_lens_h.data(),
        static_cast<std::uint32_t>(total_tokens),
        static_cast<std::uint32_t>(num_requests),
        static_cast<std::uint32_t>(num_q_heads),
        static_cast<std::uint32_t>(num_kv_heads),
        static_cast<std::uint32_t>(head_dim),
        static_cast<std::uint32_t>(head_dim),
        static_cast<std::uint32_t>(page_size),
        causal,
        enable_cuda_graph,
        sizeof(DTypeO),
        stream));

    const std::int64_t bias = static_cast<std::int64_t>(int_base_bytes);
    plan.qo_tile_indices_offset = plan_info.qo_tile_indices_offset + bias;
    plan.qo_indptr_offset = plan_info.qo_indptr_offset + bias;
    plan.kv_indptr_offset = plan_info.kv_indptr_offset + bias;
    plan.qo_len_offset = plan_info.qo_len_offset + bias;
    plan.kv_len_offset = plan_info.kv_len_offset + bias;
    plan.head_indices_offset = plan_info.head_indices_offset + bias;
    plan.work_indptr_offset = plan_info.work_indptr_offset + bias;
    plan.batch_indices_offset = plan_info.batch_indices_offset + bias;
    plan.same_schedule_for_all_heads = plan_info.same_schedule_for_all_heads;
    plan.total_tokens = total_tokens;
    plan.num_requests = num_requests;
    plan.num_q_heads = num_q_heads;
    plan.num_kv_heads = num_kv_heads;
    plan.head_dim = head_dim;
    plan.page_size = page_size;
    plan.window_left = window_left;
    plan.causal = causal;
    plan.valid = true;
}

void dispatch_attention_flashinfer_prefill_sm90_bf16(
    const HopperPrefillPlan& plan,
    const void* q,
    void* k_pages,
    void* v_pages,
    void* o,
    const std::uint32_t* kv_page_indices_d,
    AttentionWorkspaceView workspace,
    cudaStream_t stream,
    float logits_soft_cap,
    float sm_scale,
    float* lse_out,
    bool broadcast_q) {
    if (!plan.valid) {
        throw std::runtime_error("flashinfer sm90 prefill: empty plan");
    }

    HopperParams params{};
    params.q_ptr = const_cast<DTypeQ*>(static_cast<const DTypeQ*>(q));
    params.k_ptr = static_cast<DTypeKV*>(k_pages);
    params.v_ptr = static_cast<DTypeKV*>(v_pages);
    params.o_ptr = static_cast<DTypeO*>(o);
    params.lse_ptr = lse_out;

    void* int_buf = workspace.int_buffer;
    params.qo_tile_indices =
        offset_ptr<IdType>(int_buf, plan.qo_tile_indices_offset);
    params.qo_indptr = offset_ptr<IdType>(int_buf, plan.qo_indptr_offset);
    params.kv_indptr = offset_ptr<IdType>(int_buf, plan.kv_indptr_offset);
    params.kv_indices =
        const_cast<IdType*>(reinterpret_cast<const IdType*>(kv_page_indices_d));
    params.qo_lens = offset_ptr<IdType>(int_buf, plan.qo_len_offset);
    params.kv_lens = offset_ptr<IdType>(int_buf, plan.kv_len_offset);
    params.head_indices = offset_ptr<IdType>(int_buf, plan.head_indices_offset);
    params.work_indptr = offset_ptr<IdType>(int_buf, plan.work_indptr_offset);
    params.batch_indices = offset_ptr<IdType>(int_buf, plan.batch_indices_offset);

    params.additional_params.logits_soft_cap = logits_soft_cap;
    params.additional_params.sm_scale = sm_scale > 0.f
        ? sm_scale
        : 1.0f / std::sqrt(static_cast<float>(plan.head_dim));
    params.additional_params.maybe_prefix_len_ptr = nullptr;
    params.additional_params.maybe_token_pos_in_items_ptr = nullptr;
    params.additional_params.token_pos_in_items_len = 0;
    params.additional_params.maybe_max_item_len_ptr = nullptr;

    params.q_stride_n =
        static_cast<std::int64_t>(plan.num_q_heads) * plan.head_dim;
    const std::int64_t o_stride_n = params.q_stride_n;
    if (broadcast_q) params.q_stride_n = 0;
    params.k_stride_n =
        static_cast<std::int64_t>(plan.num_kv_heads) * plan.head_dim;
    params.v_stride_n =
        static_cast<std::int64_t>(plan.num_kv_heads) * plan.head_dim;
    params.o_stride_n = o_stride_n;
    params.q_stride_h = plan.head_dim;
    params.k_stride_h = plan.head_dim;
    params.v_stride_h = plan.head_dim;
    params.o_stride_h = plan.head_dim;
    params.nnz_qo = plan.total_tokens;
    params.k_page_stride =
        static_cast<std::int64_t>(plan.page_size) * plan.num_kv_heads *
        plan.head_dim;
    params.v_page_stride = params.k_page_stride;
    params.num_qo_heads = plan.num_q_heads;
    params.num_kv_heads = plan.num_kv_heads;
    params.group_size = plan.num_q_heads / plan.num_kv_heads;
    params.page_size = plan.page_size;
    params.window_left = plan.window_left;
    params.causal = plan.causal;

    cudaError_t status;
    if (logits_soft_cap > 0.f) {
        status = dispatch_hopper_prefill_dim_mask<true>(
            params, plan.head_dim, plan.same_schedule_for_all_heads,
            plan.causal,
            /*enable_pdl=*/current_device_major() >= 9, stream);
    } else {
        status = dispatch_hopper_prefill_dim_mask<false>(
            params, plan.head_dim, plan.same_schedule_for_all_heads,
            plan.causal,
            /*enable_pdl=*/current_device_major() >= 9, stream);
    }
    CUDA_CHECK(status);
}

}  // namespace pie_cuda_driver::kernels::attn

// `merge_attention_states_bf16` used to live here. It moved to
// `attention_merge_states.cu` because its callers are not sm90-only: the
// decode KV-split path reaches it on every architecture, so stubbing it out
// with the rest of this unit broke Blackwell. See that file for the details.

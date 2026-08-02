#include "model/deepseek_v4/deepseek_v4_model.hpp"

#include "loader/group_stream_cache.hpp"

#include <cstdlib>
#include <utility>

namespace pie_cuda_driver::model {

DsV4Model::DsV4Model(
    DsV4Weights weights,
    const HfConfig& hf_config,
    DsV4Workspace& ws,
    DsV4CompressCache& comp_cache,
    KvCache& kv_cache,
    int tp_size,
    int tp_rank,
    NcclComm* tp_comm,
    bool emit_logits)
    : weights_(std::move(weights)),
      hf_config_(hf_config),
      ws_(ws),
      comp_cache_(comp_cache),
      kv_cache_(kv_cache)
{
    fwd_cfg_.tp_size = tp_size;
    fwd_cfg_.tp_rank = tp_rank;
    fwd_cfg_.tp_comm = tp_comm;
    fwd_cfg_.emit_logits = emit_logits;

    caps_.supports_compact_logits = true;
    // Pure-decode fires are fully device-side: the compressor boundary list
    // is built by `launch_dsv4_boundary_meta_decode` (fixed length, `-1`
    // marks non-boundaries) instead of a host scan behind a D2H sync, and the
    // FlashInfer plan is built in `prepare()`. The prefill path still syncs,
    // but graphs are only captured for pure-decode shapes.
    {
        bool on = true;
        // Paging experts is host work in the middle of the forward -- a routing
        // table read back off the device, a slot chosen, a plan run -- and none
        // of it is capturable.
        //
        // Residency does not decide this. A streamed contract publishes groups
        // instead of stacks, so `stacked` is false and both device-side
        // dispatches are off the table; the forward takes the per-expert path,
        // which reads `topk_idx` back and synchronizes the stream on every
        // layer whether or not the slab can miss. Under capture that
        // synchronize is `cudaErrorStreamCaptureUnsupported`. Qwen3.5 already
        // gates on the cache existing at all; this is the same rule.
        //
        // Only the capture cap. `graph_padding_kv_write_safe` says this
        // family's KV writes are gated on `row_valid`, which is a property of
        // its kernels and is not changed by where the expert weights live --
        // and the engine refuses to build a context when a family that aliases
        // KV page 0 for padding rows does not claim it, so clearing it here
        // would turn streaming on DeepSeek-V4 into `PIE_STATUS_UNSUPPORTED`
        // rather than into a lost optimisation.
        bool capturable = on;
        for (const auto& L : weights_.layers) {
            if (L.expert_cache != nullptr) {
                capturable = false;
                break;
            }
        }
        caps_.graph_safe = capturable;
        caps_.graph_padding_kv_write_safe = on;
    }
}

void DsV4Model::prepare(AttentionWorkspace& attn_ws,
                        const ForwardFn::PrepareInputs& in) {
    // Runs OUTSIDE any cuStreamCapture region — see ForwardFn's contract.
    ws_.swa_plan_valid = false;
    if (hf_config_.head_dim != 512 || in.qo_indptr_h == nullptr ||
        in.kv_page_indptr_h == nullptr || in.total_tokens <= 0) {
        return;   // fall back to the naive paged kernel in the body
    }
    if (!ws_.swa_plan) ws_.swa_plan = ops::make_prefill_plan();
    const int window_left = hf_config_.dsv4_sliding_window > 0
        ? hf_config_.dsv4_sliding_window - 1 : -1;
    ops::plan_attention_flashinfer_prefill_bf16(
        *ws_.swa_plan, in.qo_indptr_h, in.kv_page_indptr_h,
        /*kv_last_page_lens_h=*/nullptr,
        in.total_tokens, in.num_requests,
        // Full heads, not `/ tp_size`. DeepSeek-V4 replicates its attention
        // weights -- `dsv4_shard_axis` shards the FFN and nothing else -- so
        // every rank computes every head, and the forward says so where it
        // reads `num_attention_heads` straight. Planning for half of them
        // described a geometry the attention never runs, and the kernel
        // answered accordingly: identical q and kv, an `attn_out` that
        // disagreed with TP=1 from the first layer, and both ranks agreeing
        // with each other on the wrong number.
        hf_config_.num_attention_heads,
        /*num_kv_heads=*/1, hf_config_.head_dim,
        kv_cache_.page_size(), attn_ws, /*stream=*/nullptr,
        fwd_cfg_.decode_plan_cuda_graph, window_left,
        /*full_attention_variant=*/false, /*hnd_layout=*/false,
        /*causal_mask=*/true);
    ws_.swa_plan_valid = true;
}

std::uint32_t DsV4Model::graph_layout() {
    if (!ws_.swa_plan_valid || !ws_.swa_plan) return 0;
    return ops::prefill_plan_graph_layout(*ws_.swa_plan);
}

void DsV4Model::body(Workspace& ws,
                     KvCache& kv,
                     AttentionWorkspace& attn_ws,
                     ops::CublasHandle& cublas,
                     const ForwardFn::ForwardInputs& in) {
    dsv4_forward_paged(
        weights_, hf_config_, fwd_cfg_, ws_, kv, comp_cache_, attn_ws, cublas,
        ws.logits.data(),
        in.token_ids, in.positions,
        in.qo_indptr_d, in.kv_page_indices_d, in.kv_page_indptr_d,
        in.kv_last_page_lens_d,
        in.qo_indptr_h, in.kv_page_indptr_h,
        in.total_tokens, in.num_requests, in.is_pure_decode,
        in.row_valid_d,
        in.logit_row_indices_d, in.num_logit_rows,
        in.stage_hooks);
}

}  // namespace pie_cuda_driver::model

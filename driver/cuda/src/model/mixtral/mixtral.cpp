#include "model/mixtral/mixtral.hpp"
#include "model/stage_hooks.hpp"

#include <algorithm>
#include <atomic>
#include <unordered_map>
#include <mutex>
#include <memory>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "device_buffer.hpp"
#include "kernels/add_bias.hpp"
#include "kernels/attn_sink.hpp"
#include "kernels/dequant_fp4.hpp"
#include "kernels/deinterleave.hpp"
#include "kernels/embed.hpp"
#include "kernels/gather_rows.hpp"
#include "kernels/residual_add.hpp"
#include "kernels/kv_paged.hpp"
#include "kernels/moe_dispatch.hpp"
#include "kernels/rmsnorm.hpp"
#include "kernels/rope.hpp"
#include "kernels/swiglu.hpp"
#include "kernels/topk_softmax.hpp"
#include "ops/gemm.hpp"
#include "ops/attention_flashinfer.hpp"
#include "batch/tp.hpp"

namespace pie_cuda_driver::model {

namespace {

const DeviceTensor& must(const LoadedModel& e, const std::string& name) {
    if (!e.has(name)) {
        throw std::runtime_error("mixtral: missing weight '" + name + "'");
    }
    return e.get(name);
}

}  // namespace

MixtralWeights bind_mixtral(const LoadedModel& engine) {
    const auto& cfg = engine.hf_config();
    const int E = cfg.num_experts;
    if (E <= 0) {
        throw std::runtime_error(
            "mixtral: hf_config.num_experts must be > 0; check the loader");
    }

    MixtralWeights w;
    w.embed      = &must(engine, "model.embed_tokens.weight");
    w.final_norm = &must(engine, "model.norm.weight");
    if (engine.has("lm_head.weight")) {
        w.lm_head = &engine.get("lm_head.weight");
    } else if (cfg.tie_word_embeddings) {
        w.lm_head = w.embed;
    } else {
        throw std::runtime_error(
            "mixtral: lm_head missing and tie_word_embeddings=false");
    }

    w.layers.resize(static_cast<std::size_t>(cfg.num_hidden_layers));
    for (int i = 0; i < cfg.num_hidden_layers; ++i) {
        const std::string p = "model.layers." + std::to_string(i) + ".";
        auto& L = w.layers[i];
        L.attn_norm = &must(engine, p + "input_layernorm.weight");
        L.mlp_norm  = &must(engine, p + "post_attention_layernorm.weight");
        L.q_proj    = &must(engine, p + "self_attn.q_proj.weight");
        L.k_proj    = &must(engine, p + "self_attn.k_proj.weight");
        L.v_proj    = &must(engine, p + "self_attn.v_proj.weight");
        L.o_proj    = &must(engine, p + "self_attn.o_proj.weight");

        L.router    = &must(engine, p + "block_sparse_moe.gate.weight");
        L.experts.resize(static_cast<std::size_t>(E));
        for (int e = 0; e < E; ++e) {
            const std::string ep = p + "block_sparse_moe.experts." +
                                   std::to_string(e) + ".";
            // HF Mixtral expert weight layout: w1=gate, w2=down, w3=up.
            L.experts[e].w_gate = &must(engine, ep + "w1.weight");
            L.experts[e].w_down = &must(engine, ep + "w2.weight");
            L.experts[e].w_up   = &must(engine, ep + "w3.weight");
        }
    }
    return w;
}

namespace {

// Build per-expert (token, weight) lists from a [N, K] topk decision.
// Returns one vector<int32> of token indices and one matching vector<float>
// of routing weights per expert. CPU-side because routing is O(N·K) and
// the per-expert dense GEMMs dominate runtime.
struct ExpertRouting {
    std::vector<std::vector<std::int32_t>> token_idx;
    std::vector<std::vector<float>>        weights;
};

ExpertRouting build_routing(
    const std::vector<std::int32_t>& topk_idx_h,
    const std::vector<float>& topk_w_h,
    int N, int K, int E)
{
    ExpertRouting r;
    r.token_idx.assign(E, {});
    r.weights.assign(E, {});
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            const int e = topk_idx_h[n * K + k];
            if (e < 0 || e >= E) continue;
            r.token_idx[e].push_back(n);
            r.weights[e].push_back(topk_w_h[n * K + k]);
        }
    }
    return r;
}

}  // namespace

// Grow-only per-fire scratch, owned per CUDA device.
//
// The forward used to allocate nine DeviceBuffers per fire. That is fatal
// under TP for the reason spelled out at the call site: an allocator call
// synchronizes the context while the peer spins in a collective. Growth
// still allocates, but only when a fire exceeds every previous high-water
// mark, and the caller warms it before the TP group starts issuing
// collectives (see `mixtral_scratch_reserve`).
class MixtralScratch {
  public:
    float* lse(std::size_t n)                 { return grow(lse_, n); }
    std::int32_t* topk_idx(std::size_t n)     { return grow(topk_idx_, n); }
    float* topk_w(std::size_t n)              { return grow(topk_w_, n); }
    std::uint16_t* expert_in(std::size_t n)   { return grow(expert_in_, n); }
    std::uint16_t* expert_gate(std::size_t n) { return grow(expert_gate_, n); }
    std::uint16_t* expert_up(std::size_t n)   { return grow(expert_up_, n); }
    std::uint16_t* expert_out(std::size_t n)  { return grow(expert_out_, n); }
    std::int32_t* expert_idx(std::size_t n)   { return grow(expert_idx_, n); }
    float* expert_w(std::size_t n)            { return grow(expert_w_, n); }

  private:
    template <typename T>
    T* grow(DeviceBuffer<T>& buffer, std::size_t count) {
        if (count == 0) return buffer.data();
        if (buffer.size() < count) {
            // Round up so a ramp of growing shapes stops reallocating early.
            std::size_t target = buffer.size() == 0 ? count : buffer.size();
            while (target < count) target *= 2;
            buffer = DeviceBuffer<T>::alloc(target);
        }
        return buffer.data();
    }

    DeviceBuffer<float> lse_;
    DeviceBuffer<std::int32_t> topk_idx_;
    DeviceBuffer<float> topk_w_;
    DeviceBuffer<std::uint16_t> expert_in_;
    DeviceBuffer<std::uint16_t> expert_gate_;
    DeviceBuffer<std::uint16_t> expert_up_;
    DeviceBuffer<std::uint16_t> expert_out_;
    DeviceBuffer<std::int32_t> expert_idx_;
    DeviceBuffer<float> expert_w_;
};

MixtralScratch& mixtral_scratch() {
    // Per device: TP ranks are threads of one process on different devices,
    // and a DeviceBuffer belongs to the device that allocated it.
    static std::mutex mu;
    static std::unordered_map<int, std::unique_ptr<MixtralScratch>> by_device;
    int device = 0;
    cudaGetDevice(&device);
    std::lock_guard<std::mutex> lock(mu);
    auto& slot = by_device[device];
    if (!slot) slot = std::make_unique<MixtralScratch>();
    return *slot;
}


void mixtral_scratch_reserve(int max_tokens, int top_k, int hidden,
                             int intermediate_padded, int q_heads_local) {
    if (max_tokens <= 0 || top_k <= 0) return;
    const std::size_t tokens = static_cast<std::size_t>(max_tokens);
    const std::size_t routed = tokens * static_cast<std::size_t>(top_k);
    MixtralScratch& s = mixtral_scratch();
    s.lse(tokens * static_cast<std::size_t>(std::max(1, q_heads_local)));
    s.topk_idx(routed);
    s.topk_w(routed);
    s.expert_in(routed * static_cast<std::size_t>(hidden));
    s.expert_gate(routed * static_cast<std::size_t>(intermediate_padded));
    s.expert_up(routed * static_cast<std::size_t>(intermediate_padded));
    s.expert_out(routed * static_cast<std::size_t>(hidden));
    s.expert_idx(routed);
    s.expert_w(routed);
}

void mixtral_forward_paged(
    const MixtralWeights& w,
    const HfConfig& cfg,
    const LlamaLikeForwardCfg& fwd_cfg,
    int num_experts,
    int top_k,
    Workspace& ws,
    KvCache& cache,
    AttentionWorkspace& attn_ws,
    ops::CublasHandle& cublas,
    const std::int32_t* token_ids,
    const std::int32_t* positions,
    const std::uint32_t* qo_indptr,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* kv_page_indptr_h,
    int N,
    int R,
    bool is_pure_decode,
    const std::int32_t* logit_row_indices_d,
    int num_logit_rows,
    const std::uint8_t* custom_mask_d,
    const std::int32_t* custom_mask_indptr_d,
    const std::uint8_t* row_valid_d)
{
    // TP-local dims. tp_size == 1 keeps the original single-GPU shapes.
    // For Mixtral we shard *within* each expert (per-expert TP), not
    // across experts: every rank still runs the full expert routing but
    // each expert's gate/up/down weights are split along axis 0 / axis 1.
    const int T  = (fwd_cfg.tp_size > 0) ? fwd_cfg.tp_size : 1;
    const int H = cfg.hidden_size;
    const int Hq = (cfg.num_attention_heads * cfg.head_dim) / T;
    const int Hk = (cfg.num_key_value_heads * cfg.head_dim) / T;
    const int I  = cfg.intermediate_size / T;
    const int Ip = (w.mxfp4_intermediate_padded > I)
        ? w.mxfp4_intermediate_padded
        : I;
    const int num_q_heads_local  = cfg.num_attention_heads / T;
    const int num_kv_heads_local = cfg.num_key_value_heads / T;
    const int V  = cfg.vocab_size;
    const int d  = cfg.head_dim;
    const float eps = cfg.rms_norm_eps;
    cudaStream_t stream = nullptr;
    NcclComm* tp = (T > 1) ? fwd_cfg.tp_comm : nullptr;
    const bool tp_is_leader = (T == 1) || (tp != nullptr && tp->rank() == 0);

    const int beacon_rank = (tp != nullptr) ? tp->rank() : 0;
    // Global per-rank all-reduce sequence. Two ranks must issue the SAME
    // sequence of collectives; NCCL pairs them by ORDER, so one extra call
    // on either side shifts every later pairing and hangs the group on the
    // first size disagreement. Printing (seq, shape, layer, phase) makes the
    // first divergence visible instead of inferred.
    static std::atomic<long> ar_seq_global{0};
    static const bool ar_trace = std::getenv("PIE_MIXTRAL_AR_TRACE") != nullptr;
    const auto ar_log = [&](int layer, int phase) {
        if (!ar_trace) return;
        std::fprintf(stderr,
                     "[ar] rank=%d seq=%ld N=%d R=%d pure=%d L=%d p=%d\n",
                     beacon_rank,
                     ar_seq_global.fetch_add(1, std::memory_order_relaxed),
                     N, R, is_pure_decode ? 1 : 0, layer, phase);
    };
    const auto beacon = [&](int layer, int phase) {
        if (T > 1) tp_watchdog_mark_model_progress(beacon_rank, layer, phase);
    };
    const bool use_decode_path = is_pure_decode && !fwd_cfg.force_prefill_path;
    int ar_count = 0;  // per-fire all-reduce census (PIE_MIXTRAL_AR_TRACE)
    const bool any_sinks = [&]{
        for (const auto& L : w.layers) {
            if (L.attn_sinks != nullptr) return true;
        }
        return false;
    }();
    // [N, num_attention_heads] fp32 — written by flashinfer per layer when
    // sinks are active, then consumed by the rescale post-pass. Per-layer
    // overwrite is fine; we only need the layer's own lse during its own
    // rescale step. Allocate once per fire instead of once per layer.
    // NOTHING on this path may allocate. `cudaMalloc`/`cudaFree`
    // synchronize the whole CUDA context, and under TP the peer rank is
    // simultaneously spinning inside a collective kernel that only this
    // rank's matching call can release — a context-synchronizing call here
    // deadlocks the pair. (Same class as the fire-path pinned alloc/free
    // fixed in pipeline/dispatch.cu; it is why gpt-oss hung at tp2 the
    // moment a fire grew past the warmup shape and re-sized this scratch.)
    // Allocation happens once per (process, shape high-water) in a
    // grow-only cache instead, so steady state never calls the allocator.
    float* lse_ptr = nullptr;
    if (any_sinks) {
        lse_ptr = mixtral_scratch().lse(
            static_cast<std::size_t>(N) * num_q_heads_local);
    }

    kernels::launch_embed_bf16(
        token_ids, w.embed->data(), ws.y.data(), N, H, V, stream);

    ops::DecodePlanCachePtr decode_plan;
    if (use_decode_path) {
        decode_plan = ops::make_decode_plan();
        ops::plan_attention_flashinfer_decode(
            *decode_plan, kv_page_indptr_h, R,
            num_q_heads_local, num_kv_heads_local, d,
            cache.page_size(), attn_ws, stream,
            /*enable_cuda_graph=*/true,
            /*full_attention_variant=*/false,
            cache.hnd_layout());
    }

    // Per-fire scratch for MoE routing. Sized for the worst case (N
    // tokens × K experts each); reallocated per call which is fine
    // since N changes, but the host-side vectors avoid touching cuda
    // alloc. Device-side topk buffers go through DeviceBuffer for ABI
    // simplicity; if profiling shows alloc latency we can hoist these
    // into Workspace.
    // Worst case: a single expert receives all N*K routes.
    const std::size_t max_routed = static_cast<std::size_t>(N) * top_k;
    MixtralScratch& scratch = mixtral_scratch();
    auto* d_topk_idx_p    = scratch.topk_idx(max_routed);
    auto* d_topk_w_p      = scratch.topk_w(max_routed);
    auto* d_expert_in_p   = scratch.expert_in(max_routed * H);
    auto* d_expert_gate_p = scratch.expert_gate(max_routed * Ip);
    auto* d_expert_up_p   = scratch.expert_up(max_routed * Ip);
    auto* d_expert_out_p  = scratch.expert_out(max_routed * H);
    auto* d_expert_idx_p  = scratch.expert_idx(max_routed);
    auto* d_expert_w_p    = scratch.expert_w(max_routed);

    for (int L = 0; L < cfg.num_hidden_layers; ++L) {
        const auto& layer = w.layers[L];

        // Per-layer attention window: full causal (-1) for plain Mixtral;
        // GPT-OSS alternates sliding/full per `fwd_cfg.per_layer_window_left`.
        const int layer_window =
            (L < (int)fwd_cfg.per_layer_window_left.size())
                ? fwd_cfg.per_layer_window_left[L]
                : fwd_cfg.sliding_window;

        // ── Attention block (identical to llama_like pre-norm path) ──
        kernels::launch_rmsnorm_bf16(
            ws.y.data(), layer.attn_norm->data(), ws.norm_x.data(),
            N, H, eps, stream);
        ops::gemm_act_x_wt_bf16(cublas.handle(),
            ws.norm_x.data(), layer.q_proj->data(), ws.q.data(), N, Hq, H);
        ops::gemm_act_x_wt_bf16(cublas.handle(),
            ws.norm_x.data(), layer.k_proj->data(), ws.k.data(), N, Hk, H);
        ops::gemm_act_x_wt_bf16(cublas.handle(),
            ws.norm_x.data(), layer.v_proj->data(), ws.v.data(), N, Hk, H);
        if (layer.q_bias) kernels::launch_add_bias_bf16(
            ws.q.data(), layer.q_bias->data(), N, Hq, stream);
        if (layer.k_bias) kernels::launch_add_bias_bf16(
            ws.k.data(), layer.k_bias->data(), N, Hk, stream);
        if (layer.v_bias) kernels::launch_add_bias_bf16(
            ws.v.data(), layer.v_bias->data(), N, Hk, stream);

        kernels::launch_rope_bf16(
            ws.q.data(), ws.k.data(), positions,
            N, num_q_heads_local, num_kv_heads_local, d,
            cfg.rope_theta, stream);
        // Fires POST-rope (and post q/k-norm): the query a PTIR program
        // observes here is the one that actually enters attention, so an
        // observer scoring it against the cached keys -- which are stored
        // post-rope -- compares in the same space. Placing it on the raw
        // projection instead would silently mis-rank pages for Quest.
        invoke_stage_hook(
            StageHookPoint::OnAttnProj, ws.q.data(),
            static_cast<std::uint32_t>(N),
            static_cast<std::uint32_t>(Hq),
            static_cast<std::uint32_t>(L), stream);

        auto kv_view = cache.layer_view(L);
        kernels::launch_write_kv_to_pages(
            kv_view, ws.k.data(), ws.v.data(),
            qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
            N, R, stream, row_valid_d);

        // Only ask flashinfer for lse on layers that actually use sinks.
        // Saves a per-layer kernel write on plain Mixtral, and on
        // gpt-oss layers that turn out to have nullptr sinks.
        float* layer_lse = (layer.attn_sinks != nullptr) ? lse_ptr : nullptr;

        if (use_decode_path) {
            ops::dispatch_attention_flashinfer_decode(
                *decode_plan,
                ws.q.data(), kv_view, ws.attn_out.data(),
                kv_page_indices, kv_page_indptr, kv_last_page_lens,
                attn_ws, stream,
                /*window_left=*/layer_window,
                /*logits_soft_cap=*/0.f,
                /*sm_scale=*/-1.f,
                layer_lse);
        } else if (custom_mask_d) {
            ops::launch_attention_flashinfer_prefill_custom(
                ws.q.data(), kv_view, ws.attn_out.data(),
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                custom_mask_d, custom_mask_indptr_d,
                qo_indptr_h, kv_page_indptr_h,
                N, R, num_q_heads_local, attn_ws, stream,
                /*window_left=*/-1,
                /*logits_soft_cap=*/0.f, /*sm_scale=*/-1.f,
                layer_lse);
        } else {
            ops::launch_attention_flashinfer_prefill(
                ws.q.data(), kv_view, ws.attn_out.data(),
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                qo_indptr_h, kv_page_indptr_h,
                N, R, num_q_heads_local, attn_ws, stream,
                /*window_left=*/layer_window,
                /*logits_soft_cap=*/0.f,
                /*sm_scale=*/-1.f,
                layer_lse);
        }
        // GPT-OSS: rescale o by `sigmoid(lse - sink_h)` to apply the
        // softmax-denominator extension that flashinfer's DefaultAttention
        // doesn't emit natively. Per-rank shard count under TP.
        if (layer.attn_sinks != nullptr) {
            kernels::launch_attention_sink_rescale_bf16(
                ws.attn_out.data(), layer_lse, layer.attn_sinks->data(),
                N, num_q_heads_local, d, stream);
        }
        invoke_stage_hook(
            StageHookPoint::OnAttn, ws.q.data(),
            static_cast<std::uint32_t>(N),
            static_cast<std::uint32_t>(Hq),
            static_cast<std::uint32_t>(L), stream);

        // o_proj is row-parallel under TP: write to scratch, all-reduce,
        // residual-add into y. o_bias (replicated; e.g. GPT-OSS) only goes
        // in once on the leader so the all-reduce sums it exactly once.
        if (T == 1) {
            ops::gemm_act_x_wt_bf16(cublas.handle(),
                ws.attn_out.data(), layer.o_proj->data(), ws.y.data(),
                N, H, Hq, /*beta=*/1.f);
            if (layer.o_bias) kernels::launch_add_bias_bf16(
                ws.y.data(), layer.o_bias->data(), N, H, stream);
        } else {
            ops::gemm_act_x_wt_bf16(cublas.handle(),
                ws.attn_out.data(), layer.o_proj->data(), ws.norm_x.data(),
                N, H, Hq, /*beta=*/0.f);
            if (layer.o_bias && tp_is_leader) {
                kernels::launch_add_bias_bf16(
                    ws.norm_x.data(), layer.o_bias->data(), N, H, stream);
            }
            ++ar_count;
            beacon(L, 0);
            ar_log(L, 0);
            tp->all_reduce_bf16(ws.norm_x.data(),
                static_cast<std::size_t>(N) * H, ncclSum, stream);
            beacon(L, 1);
            kernels::launch_residual_add_bf16(
                ws.y.data(), ws.norm_x.data(), N * H, stream);
        }

        // ── Sparse-MoE block ──
        kernels::launch_rmsnorm_bf16(
            ws.y.data(), layer.mlp_norm->data(), ws.norm_y.data(),
            N, H, eps, stream);

        // 1. Router logits, top-K + softmax + renormalize. We piggy-back
        // on `ws.gate` as scratch for the [N, num_experts] router logits
        // — its allocation is `[max_tokens, intermediate]` which is
        // always ≥ [N, num_experts] for any production config (E ≤ 64,
        // I ≥ 4096).
        ops::gemm_act_x_wt_bf16(cublas.handle(),
            ws.norm_y.data(), layer.router->data(), ws.gate.data(),
            N, num_experts, H);
        if (layer.router_bias) kernels::launch_add_bias_bf16(
            ws.gate.data(), layer.router_bias->data(), N, num_experts, stream);
        kernels::launch_topk_softmax_bf16(
            ws.gate.data(), d_topk_idx_p, d_topk_w_p,
            N, num_experts, top_k, stream);

        // 2. D2H copy of routing decisions; build per-expert lists.
        std::vector<std::int32_t> topk_idx_h(static_cast<std::size_t>(N) * top_k);
        std::vector<float>        topk_w_h  (static_cast<std::size_t>(N) * top_k);
        CUDA_CHECK(cudaMemcpyAsync(topk_idx_h.data(), d_topk_idx_p,
                                   topk_idx_h.size() * sizeof(std::int32_t),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(topk_w_h.data(), d_topk_w_p,
                                   topk_w_h.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        beacon(L, 2);
        const auto routing = build_routing(topk_idx_h, topk_w_h,
                                           N, top_k, num_experts);

        // Under TP every rank computes 1/T of every expert's down_proj.
        // Each scatter_add accumulates a *partial* contribution; we
        // collect those in ws.norm_x (zero-initialised), all-reduce after
        // all experts, then residual-add the full MoE delta into ws.y.
        // tp_size == 1 keeps the original "scatter directly into ws.y"
        // path so single-GPU performance is unchanged.
        void* moe_target = ws.y.data();
        if (T > 1) {
            CUDA_CHECK(cudaMemsetAsync(
                ws.norm_x.data(), 0,
                static_cast<std::size_t>(N) * H * sizeof(std::uint16_t),
                stream));
            moe_target = ws.norm_x.data();
        }

        // 3. Per-expert dispatch.
        for (int e = 0; e < num_experts; ++e) {
            const auto& tok_idx = routing.token_idx[e];
            const auto& weights = routing.weights[e];
            const int Ne = static_cast<int>(tok_idx.size());
            if (Ne == 0) continue;

            CUDA_CHECK(cudaMemcpyAsync(
                d_expert_idx_p, tok_idx.data(),
                Ne * sizeof(std::int32_t), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(
                d_expert_w_p, weights.data(),
                Ne * sizeof(float), cudaMemcpyHostToDevice, stream));

            // Gather norm_y rows routed to this expert.
            kernels::launch_gather_bf16_rows(
                static_cast<const std::uint16_t*>(ws.norm_y.data()),
                d_expert_idx_p,
                d_expert_in_p,
                Ne, H, stream);

            // SwiGLU MLP.
            const auto& expert = layer.experts[e];
            const void* gate_w = nullptr;
            const void* up_w = nullptr;
            const void* down_w = nullptr;
            if (expert.format == MixtralExpertWeightFormat::Mxfp4NativeGemm) {
                if (!expert.w_gate_mxfp4 || !expert.w_gate_mxfp4_scale ||
                    !expert.w_up_mxfp4 || !expert.w_up_mxfp4_scale ||
                    !expert.w_down_mxfp4 || !expert.w_down_mxfp4_scale) {
                    throw std::runtime_error(
                        "mixtral/gpt_oss: incomplete native MXFP4 expert backend");
                }
                ops::gemm_act_x_w(cublas.handle(),
                    d_expert_in_p,
                    ops::WeightView::mxfp4_marlin(
                        *expert.w_gate_mxfp4, *expert.w_gate_mxfp4_scale),
                    d_expert_gate_p, Ne, Ip, H);
                ops::gemm_act_x_w(cublas.handle(),
                    d_expert_in_p,
                    ops::WeightView::mxfp4_marlin(
                        *expert.w_up_mxfp4, *expert.w_up_mxfp4_scale),
                    d_expert_up_p, Ne, Ip, H);
                if (expert.b_gate) kernels::launch_add_bias_bf16_strided(
                    d_expert_gate_p, expert.b_gate->data(), Ne, I, Ip,
                    stream);
                if (expert.b_up) kernels::launch_add_bias_bf16_strided(
                    d_expert_up_p, expert.b_up->data(), Ne, I, Ip,
                    stream);
                if (cfg.swiglu_limit > 0.f) {
                    kernels::launch_gpt_oss_glu_bf16(
                        d_expert_gate_p, d_expert_up_p,
                        d_expert_gate_p,
                        static_cast<int>(static_cast<std::size_t>(Ne) * Ip), stream,
                        /*limit=*/cfg.swiglu_limit);
                } else {
                    kernels::launch_swiglu_bf16(
                        d_expert_gate_p, d_expert_up_p,
                        d_expert_gate_p,
                        static_cast<std::size_t>(Ne) * Ip, stream);
                }
                ops::gemm_act_x_w(cublas.handle(),
                    d_expert_gate_p,
                    ops::WeightView::mxfp4_marlin(
                        *expert.w_down_mxfp4, *expert.w_down_mxfp4_scale),
                    d_expert_out_p, Ne, H, Ip);
                if (expert.b_down && tp_is_leader) kernels::launch_add_bias_bf16(
                    d_expert_out_p, expert.b_down->data(), Ne, H, stream);
                kernels::launch_scatter_add_weighted_bf16(
                    moe_target, d_expert_out_p,
                    d_expert_idx_p, d_expert_w_p,
                    Ne, H, stream);
                continue;
            }
            if (expert.format == MixtralExpertWeightFormat::Mxfp4RoutedDequant) {
                if (!expert.w_gate_up || !expert.w_gate_up_scale ||
                    !expert.w_down_packed || !expert.w_down_scale ||
                    w.mxfp4_gate_up_bf16_scratch.empty() ||
                    w.mxfp4_gate_bf16_scratch.empty() ||
                    w.mxfp4_up_bf16_scratch.empty() ||
                    w.mxfp4_down_bf16_scratch.empty()) {
                    throw std::runtime_error(
                        "mixtral/gpt_oss: incomplete MXFP4 expert backend");
                }
                kernels::launch_dequant_mxfp4_to_bf16(
                    static_cast<const std::uint8_t*>(expert.w_gate_up->data()),
                    static_cast<const std::uint8_t*>(
                        expert.w_gate_up_scale->data()),
                    w.mxfp4_gate_up_bf16_scratch.data(),
                    2 * I, H, stream);
                kernels::launch_dequant_mxfp4_to_bf16(
                    static_cast<const std::uint8_t*>(
                        expert.w_down_packed->data()),
                    static_cast<const std::uint8_t*>(
                        expert.w_down_scale->data()),
                    w.mxfp4_down_bf16_scratch.data(),
                    H, I, stream);
                kernels::launch_deinterleave_rows_bf16(
                    w.mxfp4_gate_up_bf16_scratch.data(),
                    w.mxfp4_gate_bf16_scratch.data(),
                    w.mxfp4_up_bf16_scratch.data(),
                    I, H, stream);
                gate_w = w.mxfp4_gate_bf16_scratch.data();
                up_w = w.mxfp4_up_bf16_scratch.data();
                down_w = w.mxfp4_down_bf16_scratch.data();
            } else {
                gate_w = expert.w_gate->data();
                up_w = expert.w_up->data();
                down_w = expert.w_down->data();
            }
            ops::gemm_act_x_wt_bf16(cublas.handle(),
                d_expert_in_p, gate_w,
                d_expert_gate_p, Ne, I, H);
            ops::gemm_act_x_wt_bf16(cublas.handle(),
                d_expert_in_p, up_w,
                d_expert_up_p, Ne, I, H);
            if (expert.b_gate) kernels::launch_add_bias_bf16(
                d_expert_gate_p, expert.b_gate->data(), Ne, I, stream);
            if (expert.b_up) kernels::launch_add_bias_bf16(
                d_expert_up_p, expert.b_up->data(), Ne, I, stream);
            if (cfg.swiglu_limit > 0.f) {
                kernels::launch_gpt_oss_glu_bf16(
                    d_expert_gate_p, d_expert_up_p,
                    d_expert_gate_p,
                    static_cast<int>(static_cast<std::size_t>(Ne) * I), stream,
                    /*limit=*/cfg.swiglu_limit);
            } else {
                kernels::launch_swiglu_bf16(
                    d_expert_gate_p, d_expert_up_p,
                    d_expert_gate_p,
                    static_cast<std::size_t>(Ne) * I, stream);
            }
            ops::gemm_act_x_wt_bf16(cublas.handle(),
                d_expert_gate_p, down_w,
                d_expert_out_p, Ne, H, I);
            // b_down is replicated across ranks; only the leader applies
            // it so the all-reduce sums it once. Plain Mixtral has no
            // b_down so this branch is dead until GPT-OSS.
            if (expert.b_down && tp_is_leader) kernels::launch_add_bias_bf16(
                d_expert_out_p, expert.b_down->data(), Ne, H, stream);

            // Scatter into ws.y (TP=1) or moe_target scratch (TP>1) with
            // routing weight, residual-add style.
            kernels::launch_scatter_add_weighted_bf16(
                moe_target, d_expert_out_p,
                d_expert_idx_p, d_expert_w_p,
                Ne, H, stream);
        }

        if (T > 1) {
            ++ar_count;
            beacon(L, 3);
            ar_log(L, 3);
            tp->all_reduce_bf16(ws.norm_x.data(),
                static_cast<std::size_t>(N) * H, ncclSum, stream);
            kernels::launch_residual_add_bf16(
                ws.y.data(), ws.norm_x.data(), N * H, stream);
        }
    }

    // Per-fire collective census (PIE_MIXTRAL_AR_TRACE=1). Ranks must issue
    // the SAME number of all-reduces in the same order; a divergence hangs
    // the group with one rank spinning in NCCL. Printing the count with the
    // fire's shape is what makes an asymmetry visible instead of inferred.
    if (T > 1) {
        static const bool trace =
            std::getenv("PIE_MIXTRAL_AR_TRACE") != nullptr;
        if (trace) {
            std::fprintf(stderr,
                         "[mixtral-ar] rank=%d N=%d R=%d pure_decode=%d "
                         "layers=%d ar_this_fire=%d\n",
                         tp != nullptr ? tp->rank() : 0, N, R,
                         is_pure_decode ? 1 : 0,
                         static_cast<int>(w.layers.size()), ar_count);
        }
    }
    if (!fwd_cfg.emit_logits) {
        return;
    }
    // Compact logits: gather only the rows that will be sampled before the
    // lm_head, instead of materializing [N, vocab]. Every other family already
    // declares this; without it the batch engine hands the device-side sampler
    // an empty row list and its descriptor channel is never produced.
    const bool compact_logits =
        logit_row_indices_d != nullptr && num_logit_rows > 0 &&
        num_logit_rows < N;
    const int lm_head_rows = compact_logits ? num_logit_rows : N;
    if (compact_logits) {
        kernels::launch_gather_bf16_rows(
            static_cast<const std::uint16_t*>(ws.y.data()),
            logit_row_indices_d,
            static_cast<std::uint16_t*>(ws.norm_x.data()),
            num_logit_rows, H, stream);
        kernels::launch_rmsnorm_bf16(
            ws.norm_x.data(), w.final_norm->data(), ws.norm_y.data(),
            num_logit_rows, H, eps, stream);
        ops::gemm_act_x_wt_bf16(cublas.handle(),
            ws.norm_y.data(), w.lm_head->data(), ws.logits.data(),
            lm_head_rows, V, H);
        return;
    }
    kernels::launch_rmsnorm_bf16(
        ws.y.data(), w.final_norm->data(), ws.norm_x.data(),
        N, H, eps, stream);
    ops::gemm_act_x_wt_bf16(cublas.handle(),
        ws.norm_x.data(), w.lm_head->data(), ws.logits.data(),
        N, V, H);
}

}  // namespace pie_cuda_driver::model

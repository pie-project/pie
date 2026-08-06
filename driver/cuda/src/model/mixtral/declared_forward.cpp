#include "model/mixtral/declared_forward.hpp"

#include <atomic>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "kernels/add_bias.hpp"
#include "kernels/attn_sink.hpp"
#include "kernels/dequant_fp4.hpp"
#include "kernels/dequant_wna16.hpp"
#include "kernels/embed.hpp"
#include "kernels/gather_rows.hpp"
#include "kernels/kv_paged.hpp"
#include "kernels/moe_dispatch.hpp"
#include "kernels/residual_add.hpp"
#include "kernels/rmsnorm.hpp"
#include "kernels/rope.hpp"
#include "kernels/swiglu.hpp"
#include "kernels/topk_softmax.hpp"
#include "ops/attention_flashinfer.hpp"
#include "ops/gemm.hpp"

namespace pie_cuda_driver::model {

namespace {

using pie_forward::PieForwardOp;
using pie_forward::PieForwardOpKind;

// One enum value per symbol the decode plan states. EXHAUSTIVE against
// that plan: `gpt_oss_validate_stated_kernels` walks it at load and a
// symbol outside this list is a model-load failure, so this list and
// `family::gpt_oss_cuda` are two spellings of one vocabulary.
enum class GoKernel {
    GemmBias,
    WriteKvToPages,
    AttnFlashinferDecode,
    AttnSinkRescale,
    TopkSoftmax,
    Bf16ToFp16,
    Mxfp4GateUp,
    Mxfp4Down,
    GptOssGlu,
    WeightedSum,
    ResidualAdd,
};

GoKernel resolve_go_kernel(std::string_view k) {
    if (k == "ops::gemm_act_x_wt_bias_bf16") return GoKernel::GemmBias;
    if (k == "launch_write_kv_to_pages") return GoKernel::WriteKvToPages;
    if (k == "dispatch_attention_flashinfer_decode")
        return GoKernel::AttnFlashinferDecode;
    if (k == "launch_attention_sink_rescale_bf16")
        return GoKernel::AttnSinkRescale;
    if (k == "launch_topk_softmax_bf16") return GoKernel::TopkSoftmax;
    if (k == "launch_bf16_to_fp16") return GoKernel::Bf16ToFp16;
    if (k == "launch_mxfp4_moe_gate_up_decode_bf16")
        return GoKernel::Mxfp4GateUp;
    if (k == "launch_mxfp4_moe_down_decode_bf16") return GoKernel::Mxfp4Down;
    if (k == "launch_gpt_oss_glu_bf16") return GoKernel::GptOssGlu;
    if (k == "launch_token_batched_weighted_sum_bf16")
        return GoKernel::WeightedSum;
    if (k == "launch_residual_add_bf16") return GoKernel::ResidualAdd;
    throw std::runtime_error(
        "declared gptoss: stated kernel '" + std::string(k) +
        "' is not in this executor's registry (the trace and the driver "
        "drifted)");
}

[[noreturn]] void throw_drift(const std::string& what) {
    throw std::runtime_error("declared gptoss: " + what +
                             " has no emission rule");
}

// `layer.field` -> (layer, field). A model-level name has no dot.
struct ParsedName {
    int layer = -1;
    std::string_view field;
};

ParsedName parse_name(std::string_view name) {
    constexpr std::string_view prefix = "layer.";
    if (name.substr(0, prefix.size()) != prefix) return {-1, name};
    const std::size_t dot = name.find('.', prefix.size());
    if (dot == std::string_view::npos) {
        throw_drift("weight name '" + std::string(name) + "'");
    }
    int layer = 0;
    for (std::size_t i = prefix.size(); i < dot; ++i) {
        if (name[i] < '0' || name[i] > '9') {
            throw_drift("weight name '" + std::string(name) + "'");
        }
        layer = layer * 10 + (name[i] - '0');
    }
    return {layer, name.substr(dot + 1)};
}

const DeviceTensor* bind(const MixtralWeights& w, std::string_view name) {
    const ParsedName nm = parse_name(name);
    if (nm.layer < 0) {
        if (nm.field == "embed") return w.embed;
        if (nm.field == "final_norm") return w.final_norm;
        if (nm.field == "lm_head") return w.lm_head;
        throw_drift("unknown model weight '" + std::string(name) + "'");
    }
    if (nm.layer >= static_cast<int>(w.layers.size())) {
        throw_drift("weight names layer " + std::to_string(nm.layer));
    }
    const MixtralLayerWeights& l = w.layers[static_cast<std::size_t>(nm.layer)];
    if (nm.field == "attn_norm") return l.attn_norm;
    if (nm.field == "mlp_norm") return l.mlp_norm;
    if (nm.field == "q_proj") return l.q_proj;
    if (nm.field == "k_proj") return l.k_proj;
    if (nm.field == "v_proj") return l.v_proj;
    if (nm.field == "o_proj") return l.o_proj;
    if (nm.field == "q_bias") return l.q_bias;
    if (nm.field == "k_bias") return l.k_bias;
    if (nm.field == "v_bias") return l.v_bias;
    if (nm.field == "o_bias") return l.o_bias;
    if (nm.field == "attn_sinks") return l.attn_sinks;
    if (nm.field == "router") return l.router;
    if (nm.field == "router_bias") return l.router_bias;
    // The two expert BANKS are not tensors: they name the layer's
    // per-expert pointer arrays, which the arms reach through `w.layers`
    // directly. Naming them here would be a lie about what they are.
    throw_drift("unknown layer weight '" + std::string(name) + "'");
}

const DeviceTensor& require(const MixtralWeights& w, std::string_view name) {
    const DeviceTensor* t = bind(w, name);
    if (t == nullptr) {
        throw std::runtime_error("declared gptoss: weight '" +
                                 std::string(name) +
                                 "' is named by the trace but not bound");
    }
    return *t;
}

}  // namespace

void gpt_oss_validate_stated_kernels(const pie_forward::ForwardPlan& plan) {
    const std::size_t n = plan.op_count();
    for (std::size_t i = 0; i < n; ++i) {
        const auto& op = plan.op(i);
        if (op.kind != pie_forward::PieForwardOpKind::Launch) continue;
        (void)resolve_go_kernel(plan.weight_name(op));
    }
}

bool gpt_oss_forward_declared(
    const GptOssDeclaredPlan& declared,
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
    const std::uint32_t* kv_page_indptr_h,
    int total_tokens,
    int num_requests,
    bool is_pure_decode,
    const std::uint8_t* row_valid_d,
    const std::int32_t* logit_row_indices_d,
    int num_logit_rows)
{
    if (!declared.usable) return false;
    // The declaration is the DECODE class only, and `use_decode_path` is
    // the hand pass's own test for it. Asking it here rather than
    // restating it keeps the two from drifting apart.
    if (!is_pure_decode || fwd_cfg.force_prefill_path) return false;

    const int N = total_tokens;
    const int R = num_requests;
    const int H = cfg.hidden_size;
    const int I = cfg.intermediate_size;
    const int V = cfg.vocab_size;
    const int d = cfg.head_dim;
    const int Hq = cfg.num_attention_heads * d;
    const int Hk = cfg.num_key_value_heads * d;
    const float eps = cfg.rms_norm_eps;
    cudaStream_t stream = cublas.stream();

    // The fused leg's admission threshold, in ROUTES. Past it the hand
    // pass materializes its experts through a host walk this declaration
    // refuses — so past it the drive declines and that pass runs.
    const int routes = N * top_k;
    if (routes > declared.max_routes) return false;

    const pie_forward::ForwardPlan& plan = declared.decode;

    // Per-fire scratch, the same set and the same sizes
    // `mixtral_forward_paged` allocates. The drive threads the hand
    // pass's buffers rather than an arena, so a value's home is that
    // pass's home for it.
    auto d_lse = DeviceBuffer<float>::alloc(
        static_cast<std::size_t>(N) * cfg.num_attention_heads);
    auto d_topk_idx = DeviceBuffer<std::int32_t>::alloc(
        static_cast<std::size_t>(N) * top_k);
    auto d_topk_w = DeviceBuffer<float>::alloc(
        static_cast<std::size_t>(N) * top_k);
    auto d_act_fp16 = DeviceBuffer<std::uint16_t>::alloc(
        static_cast<std::size_t>(N) * H);
    auto d_route_gate = DeviceBuffer<std::uint16_t>::alloc(
        static_cast<std::size_t>(routes) * I);
    auto d_route_up = DeviceBuffer<std::uint16_t>::alloc(
        static_cast<std::size_t>(routes) * I);
    auto d_route_act_fp16 = DeviceBuffer<std::uint16_t>::alloc(
        static_cast<std::size_t>(routes) * I);
    auto d_route_out = DeviceBuffer<std::uint16_t>::alloc(
        static_cast<std::size_t>(routes) * H);
    auto d_moe_out = DeviceBuffer<std::uint16_t>::alloc(
        static_cast<std::size_t>(N) * H);

    // The decode plan the dispatch's contract obligates. One per fire,
    // shared by every layer — mixtral's shape (one head geometry, so no
    // full/sliding split the way gemma-4 has).
    ops::DecodePlanCachePtr decode_plan = ops::make_decode_plan();
    ops::plan_attention_flashinfer_decode(
        *decode_plan, kv_page_indptr_h, R,
        cfg.num_attention_heads, cfg.num_key_value_heads, d,
        cache.page_size(), attn_ws, stream,
        /*enable_cuda_graph=*/true,
        /*full_attention_variant=*/false,
        cache.hnd_layout());

    int lm_head_rows = N;
    int cur_layer = -1;
    const auto enter = [&](int l) {
        if (l >= 0) cur_layer = l;
    };
    // The layer's attention window: a scalar argument, not a kernel, so
    // the declaration never states it and the executor reads it where
    // the hand pass does.
    const auto window_of = [&](int l) {
        return (l < static_cast<int>(fwd_cfg.per_layer_window_left.size()))
                   ? fwd_cfg.per_layer_window_left[static_cast<std::size_t>(l)]
                   : fwd_cfg.sliding_window;
    };

    const auto execute_op = [&](const PieForwardOp& op) {
        enter(op.layer);
        switch (op.kind) {
        case PieForwardOpKind::Embed:
            kernels::launch_embed_bf16(
                token_ids, require(w, plan.weight_name(op)).data(),
                ws.y.data(), N, H, V, stream);
            break;
        case PieForwardOpKind::Rmsnorm: {
            const std::string_view name = plan.weight_name(op);
            const ParsedName nm = parse_name(name);
            // The attention norm lands in `norm_x`, the MLP norm in
            // `norm_y` — the hand pass's two scratch slots, and the MoE
            // block reads `norm_y` twice (router input and cast input).
            void* out = nm.field == "mlp_norm"    ? ws.norm_y.data()
                        : nm.field == "attn_norm" ? ws.norm_x.data()
                                                  : ws.norm_x.data();
            if (nm.field != "attn_norm" && nm.field != "mlp_norm" &&
                nm.field != "final_norm") {
                throw_drift("rmsnorm on '" + std::string(name) + "'");
            }
            kernels::launch_rmsnorm_bf16(
                ws.y.data(), require(w, name).data(), out, N, H, eps, stream);
            break;
        }
        case PieForwardOpKind::Matmul: {
            const std::string_view name = plan.weight_name(op);
            const ParsedName nm = parse_name(name);
            if (nm.field == "o_proj") {
                // beta=1: the residual folds into the projection, which
                // is why `o_bias` is a separate add and q/k/v's are not.
                ops::gemm_act_x_wt_bf16(
                    cublas.handle(), ws.attn_out.data(),
                    require(w, name).data(), ws.y.data(), N, H, Hq,
                    /*beta=*/1.f);
            } else {
                throw_drift("matmul on '" + std::string(name) + "'");
            }
            break;
        }
        case PieForwardOpKind::AddBias: {
            const std::string_view name = plan.weight_name(op);
            const ParsedName nm = parse_name(name);
            if (nm.field != "o_bias") {
                throw_drift("add_bias on '" + std::string(name) + "'");
            }
            kernels::launch_add_bias_bf16(
                ws.y.data(), require(w, name).data(), N, H, stream);
            break;
        }
        case PieForwardOpKind::Rope:
            kernels::launch_rope_bf16(
                ws.q.data(), ws.k.data(), positions, N,
                cfg.num_attention_heads, cfg.num_key_value_heads, d,
                cfg.rope_theta, stream);
            break;
        case PieForwardOpKind::LmHead: {
            const std::string_view name = plan.weight_name(op);
            const void* input = ws.norm_x.data();
            int rows = N;
            if (logit_row_indices_d != nullptr && num_logit_rows > 0 &&
                num_logit_rows < N) {
                kernels::launch_gather_bf16_rows(
                    static_cast<const std::uint16_t*>(ws.norm_x.data()),
                    logit_row_indices_d,
                    static_cast<std::uint16_t*>(ws.norm_y.data()),
                    num_logit_rows, H, stream);
                input = ws.norm_y.data();
                rows = num_logit_rows;
            }
            lm_head_rows = rows;
            (void)lm_head_rows;
            ops::gemm_act_x_wt_bf16(
                cublas.handle(), input, require(w, name).data(),
                ws.logits.data(), rows, V, H, /*beta=*/0.f);
            break;
        }
        case PieForwardOpKind::Launch: {
            const std::string_view sym = plan.weight_name(op);
            const auto names = plan.aux_names(op);
            const auto aux = [&](std::size_t i) { return plan.name(names[i]); };
            const MixtralLayerWeights& layer =
                w.layers[static_cast<std::size_t>(cur_layer)];
            switch (resolve_go_kernel(sym)) {
            case GoKernel::GemmBias: {
                // Four sites, told apart by the projection they name:
                // q/k/v write the attention staging buffers and read the
                // attention norm; the router reads the MLP norm and
                // writes `ws.gate`, which the hand pass borrows as
                // `[N, E]` scratch.
                const std::string_view proj = aux(0);
                const ParsedName nm = parse_name(proj);
                const void* in = ws.norm_x.data();
                void* out = nullptr;
                int cols = 0;
                if (nm.field == "q_proj") {
                    out = ws.q.data(); cols = Hq;
                } else if (nm.field == "k_proj") {
                    out = ws.k.data(); cols = Hk;
                } else if (nm.field == "v_proj") {
                    out = ws.v.data(); cols = Hk;
                } else if (nm.field == "router") {
                    in = ws.norm_y.data();
                    out = ws.gate.data();
                    cols = num_experts;
                } else {
                    throw_drift("biased projection on '" + std::string(proj) + "'");
                }
                ops::gemm_act_x_wt_bias_bf16(
                    cublas.handle(), in, require(w, proj).data(),
                    require(w, aux(1)).data(), out, N, cols, H, stream);
                break;
            }
            case GoKernel::WriteKvToPages: {
                auto kv_view = cache.layer_view(cur_layer);
                kernels::launch_write_kv_to_pages(
                    kv_view, ws.k.data(), ws.v.data(),
                    qo_indptr, kv_page_indices, kv_page_indptr,
                    kv_last_page_lens, N, R, stream, row_valid_d);
                break;
            }
            case GoKernel::AttnFlashinferDecode: {
                auto kv_view = cache.layer_view(cur_layer);
                // The LSE is the second OUTPUT, and asking for it is the
                // whole difference between this call and the one every
                // other family makes.
                ops::dispatch_attention_flashinfer_decode(
                    *decode_plan, ws.q.data(), kv_view, ws.attn_out.data(),
                    kv_page_indices, kv_page_indptr, kv_last_page_lens,
                    attn_ws, stream,
                    /*window_left=*/window_of(cur_layer),
                    /*logits_soft_cap=*/0.f, /*sm_scale=*/-1.f,
                    d_lse.data());
                break;
            }
            case GoKernel::AttnSinkRescale:
                kernels::launch_attention_sink_rescale_bf16(
                    ws.attn_out.data(), d_lse.data(),
                    require(w, aux(0)).data(), N, cfg.num_attention_heads, d,
                    stream);
                break;
            case GoKernel::TopkSoftmax:
                kernels::launch_topk_softmax_bf16(
                    ws.gate.data(), d_topk_idx.data(), d_topk_w.data(),
                    N, num_experts, top_k, stream);
                break;
            case GoKernel::Bf16ToFp16:
                // TWO sites over different extents, told apart by the
                // op's own OUTPUT SHAPE rather than by a counter: the
                // block input is `[Tokens, hidden]` (rank 2), the
                // post-activation routes are `[Tokens, k, intermediate]`
                // (rank 3). A per-layer "first or second" counter would
                // be a restatement of the trace's order, and would go
                // silently wrong the day a layer states one of them
                // twice.
                if (plan.value(plan.outputs(op)[0]).rank == 2) {
                    kernels::launch_bf16_to_fp16(
                        ws.norm_y.data(), d_act_fp16.data(),
                        static_cast<std::size_t>(N) * H, stream);
                } else {
                    kernels::launch_bf16_to_fp16(
                        d_route_gate.data(), d_route_act_fp16.data(),
                        static_cast<std::size_t>(routes) * I, stream);
                }
                break;
            case GoKernel::Mxfp4GateUp:
                kernels::launch_mxfp4_moe_gate_up_decode_bf16(
                    d_act_fp16.data(), d_topk_idx.data(),
                    layer.expert_gate_up_packed_ptrs.data(),
                    layer.expert_gate_up_scale_ptrs.data(),
                    layer.expert_gate_bias_ptrs.data(),
                    layer.expert_up_bias_ptrs.data(),
                    d_route_gate.data(), d_route_up.data(),
                    N, top_k, H, I, stream);
                break;
            case GoKernel::GptOssGlu:
                kernels::launch_gpt_oss_glu_bf16(
                    d_route_gate.data(), d_route_up.data(),
                    d_route_gate.data(),
                    static_cast<int>(static_cast<std::size_t>(routes) * I),
                    stream, /*limit=*/cfg.swiglu_limit);
                break;
            case GoKernel::Mxfp4Down:
                kernels::launch_mxfp4_moe_down_decode_bf16(
                    d_route_act_fp16.data(), d_topk_idx.data(),
                    layer.expert_down_packed_ptrs.data(),
                    layer.expert_down_scale_ptrs.data(),
                    layer.expert_down_bias_ptrs.data(),
                    d_route_out.data(), N, top_k, H, I, stream);
                break;
            case GoKernel::WeightedSum:
                kernels::launch_token_batched_weighted_sum_bf16(
                    d_moe_out.data(), d_route_out.data(),
                    static_cast<const float*>(d_topk_w.data()),
                    N, top_k, H, stream);
                break;
            case GoKernel::ResidualAdd:
                kernels::launch_residual_add_bf16(
                    ws.y.data(), d_moe_out.data(), N * H, stream);
                break;
            }
            break;
        }
        default:
            throw_drift("op kind " +
                        std::to_string(static_cast<std::uint32_t>(op.kind)));
        }
    };

    // Say ONCE that this drive took a fire. Without it, coherent output
    // is evidence about the hand-written pass as easily as about this
    // one.
    {
        static std::atomic<bool> said{false};
        if (!said.exchange(true)) {
            std::fprintf(stderr,
                         "[declared-gptoss] first DECODE fire: N=%d R=%d "
                         "routes=%d ops=%zu\n",
                         N, R, routes, plan.op_count());
        }
    }

    std::vector<pie_forward::PieForwardRow> rows(static_cast<std::size_t>(N));
    for (int r = 0; r < N; ++r) {
        auto& row = rows[static_cast<std::size_t>(r)];
        row.multi_token = 0;
        row.custom_mask = 0;
        row.hooked = 0;
        row.lora = 0;
        row.write_desc = 0;
        row.wants_scores = 0;
        row.samples = 1;
        row._pad = 0;
        row.depth_k = -1;
    }
    if (logit_row_indices_d != nullptr && num_logit_rows > 0 &&
        num_logit_rows < N) {
        for (int r = num_logit_rows; r < N; ++r) {
            rows[static_cast<std::size_t>(r)].samples = 0;
        }
    }
    const pie_forward::PieForwardLowered flat =
        plan.lower(rows.data(), rows.size());
    if (flat.uncovered != pie_forward::PieForwardUncovered::None) return false;

    std::size_t next_site = 0;
    std::size_t at = 0;
    while (at < flat.launches_len || next_site < flat.structural_len) {
        const bool site_first =
            at >= flat.launches_len ||
            (next_site < flat.structural_len &&
             flat.structural[next_site].at_op < flat.launches[at].at_op);
        if (site_first) {
            execute_op(plan.op(flat.structural[next_site].at_op));
            ++next_site;
            continue;
        }
        const std::uint32_t at_op = flat.launches[at].at_op;
        while (at < flat.launches_len && flat.launches[at].at_op == at_op) ++at;
        execute_op(plan.op(at_op));
    }
    return true;
}

}  // namespace pie_cuda_driver::model

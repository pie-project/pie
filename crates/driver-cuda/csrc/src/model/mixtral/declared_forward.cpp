#include "model/mixtral/declared_forward.hpp"

#include <atomic>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "norm/add_bias.hpp"
#include "attn/attn_sink.hpp"
#include "quant/dequant_fp4.hpp"
#include "quant/dequant_wna16.hpp"
#include "layout/embed.hpp"
#include "layout/gather_rows.hpp"
#include "attn/kv_paged.hpp"
#include "moe/moe_dispatch.hpp"
#include "norm/residual_add.hpp"
#include "norm/rmsnorm.hpp"
#include "rope/rope.hpp"
#include "mlp/swiglu.hpp"
#include "moe/topk_softmax.hpp"
#include "attn/attention_flashinfer.hpp"
#include "gemm/gemm.hpp"
#include "model/declared/arms.hpp"
#include "model/declared/execute.hpp"
#include "model/declared/registry.hpp"
#include "model/declared/weights.hpp"
#include "model/declared/value_arena.hpp"

namespace pie_cuda_driver::model {

namespace {

using pie_forward::PieForwardOp;
using pie_forward::PieForwardOpKind;

// `PIE_DECLARED_HOST_ARENA=0` puts this family's pin table back in
// charge; the host assigns otherwise. Shared spelling with gemma-4,
// which is the point — one switch turns the whole driver back to
// conventions, and the A/B it gives is per-family because the pin
// tables are.
//
// gpt-oss NEEDS that A/B in a way gemma-4 did not. Its family-parity
// gate is red for a reason that predates any of this: the HAND-WRITTEN
// pass is nondeterministic. One prompt at a fixed seed, four launches,
// four different and degenerate texts ("piece piece piece"), while the
// declared path returns the same coherent sample every time. So the
// hand pass cannot be the reference this conversion is checked against,
// and the honest gate is the one this switch makes possible: the
// declared path against ITSELF, pins versus arena.
bool gpt_oss_host_arena_enabled() {
    const char* v = std::getenv("PIE_DECLARED_HOST_ARENA");
    return v == nullptr || v[0] != '0';
}

// `PIE_DECLARED_HOST_ARENA_LO` / `_HI`: host-place only the values whose
// OWNER falls in `[lo, hi)`, and pin the rest. gemma-4's cut, and the
// reasoning for the axis is written out there — an id range splits an
// alias chain, an offset range cannot separate two chains that reuse a
// slot, an owner range does both.
std::size_t host_arena_lo() {
    const char* v = std::getenv("PIE_DECLARED_HOST_ARENA_LO");
    return v != nullptr ? static_cast<std::size_t>(std::atoll(v)) : 0;
}

std::size_t host_arena_hi() {
    const char* v = std::getenv("PIE_DECLARED_HOST_ARENA_HI");
    return v != nullptr ? static_cast<std::size_t>(std::atoll(v))
                        : static_cast<std::size_t>(-1);
}

// One enum value per symbol the decode plan states. EXHAUSTIVE against
// that plan: `gpt_oss_validate_stated_kernels` walks it at load and a
// symbol outside this list is a model-load failure, so this list and
// `family::gpt_oss_cuda` are two spellings of one vocabulary.


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

// This family's half of `declared::WeightBinder`: a traced name against
// MixtralWeights. The signature is the binder's -- a plain function
// pointer plus context -- so no arm that goes through it names a struct
// field, which is what lets an arm be shared with a family whose field
// is spelled differently.
const DeviceTensor* bind_gpt_oss_weight(
    const void* ctx, const declared::ParsedWeightName& nm,
    std::string_view name) {
    const auto& w = *static_cast<const MixtralWeights*>(ctx);
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


}  // namespace

std::string gpt_oss_validate_stated_weights(
    const pie_forward::ForwardPlan& plan, const MixtralWeights& w) {
    // qwen3_5's name-resolution dry walk. An unbound weight found at the
    // first fire takes the model load down; found here, the plan declines
    // and the hand-written pass runs.
    const auto resolves = [&](std::string_view name) {
        if (name.empty()) return true;
        // The two expert BANKS are not tensors — they name the layer's
        // per-expert pointer arrays, which the arms reach through
        // `w.layers` directly.
        if (name.find("expert_gate_up_bank") != std::string_view::npos ||
            name.find("expert_down_bank") != std::string_view::npos) {
            return true;
        }
        try {
            return declared::WeightBinder{&bind_gpt_oss_weight, &w}
                       .find(name) != nullptr;
        } catch (const std::exception&) {
            return false;
        }
    };
    for (std::size_t i = 0; i < plan.op_count(); ++i) {
        const auto& op = plan.op(i);
        // On a LAUNCH op the weight slot holds the KERNEL SYMBOL, not a
        // weight — the arms read `aux_names` for the weights. Checking
        // the symbol here declined every deployment for a bogus reason,
        // which a fault-injection run caught: the drive was silently off
        // and the parity gate still said PASS, because both sides were
        // then the hand-written pass.
        const std::string_view name =
            op.kind == pie_forward::PieForwardOpKind::Launch
                ? std::string_view{}
                : plan.weight_name(op);
        if (!resolves(name)) {
            return "weight '" + std::string(name) + "' unresolvable";
        }
        if (op.kind == pie_forward::PieForwardOpKind::Launch) {
            const auto aux = plan.aux_names(op);
            for (std::size_t j = 0; j < aux.size; ++j) {
                const std::string_view a = plan.name(aux[j]);
                if (!resolves(a)) {
                    return "weight '" + std::string(a) + "' unresolvable";
                }
            }
        }
    }
    return {};
}

void gpt_oss_validate_stated_kernels(const pie_forward::ForwardPlan& plan) {
    const std::size_t n = plan.op_count();
    for (std::size_t i = 0; i < n; ++i) {
        const auto& op = plan.op(i);
        if (op.kind != pie_forward::PieForwardOpKind::Launch) continue;
        (void)declared::resolve_kernel(plan.weight_name(op));
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
    kernels::gemm::CublasHandle& cublas,
    const std::int32_t* token_ids,
    const std::int32_t* positions,
    const std::uint32_t* qo_indptr,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* kv_page_indptr_h,
    int total_tokens,
    int num_requests,
    bool is_pure_decode,
    const std::uint8_t* row_valid_d,
    const std::int32_t* logit_row_indices_d,
    int num_logit_rows)
{
    if (!declared.usable) return false;
    // WHICH CLASS — `use_decode_path` is the hand pass's own test, asked
    // rather than restated. Both classes are stated now: the fused MXFP4
    // leg is admitted by ROUTES and not by class, so a prefill under the
    // cap runs the same MoE block the decode class does.
    const bool decode_class = is_pure_decode && !fwd_cfg.force_prefill_path;
    if (!decode_class && (qo_indptr_h == nullptr || kv_page_indptr_h == nullptr)) {
        return false;  // the plan-free prefill dispatch reads host indptrs
    }

    const int N = total_tokens;
    const int R = num_requests;
    const int H = cfg.hidden_size;
    const int I = cfg.intermediate_size;
    // Every weight an arm reads goes through the binder: the arms name
    // what the TRACE names, never a struct field.
    const declared::WeightBinder wb{&bind_gpt_oss_weight, &w};
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

    const pie_forward::ForwardPlan& plan =
        decode_class ? declared.decode : declared.prefill;

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
    // The prefill dispatch builds its own plan on the way in, so only the
    // decode class owes one.
    kernels::attn::DecodePlanCachePtr decode_plan;
    if (decode_class) {
    decode_plan = kernels::attn::make_decode_plan();
    kernels::attn::plan_attention_flashinfer_decode(
        *decode_plan, kv_page_indptr_h, R,
        cfg.num_attention_heads, cfg.num_key_value_heads, d,
        cache.page_size(), attn_ws.view(), stream,
        /*enable_cuda_graph=*/true,
        /*full_attention_variant=*/false,
        cache.hnd_layout());
    }

    int lm_head_rows = N;
    int cur_layer = -1;
    const auto enter = [&](int l) {
        if (l >= 0) cur_layer = l;
    };

    // THE ROWS, and the lowering they key. Both moved ahead of the arms:
    // the arms read `values`, `values` reads the placement, and the
    // placement is what `lower` returns. Nothing here depends on a fire
    // having run.
    std::vector<pie_forward::PieForwardRow> rows(static_cast<std::size_t>(N));
    for (int r = 0; r < N; ++r) {
        auto& row = rows[static_cast<std::size_t>(r)];
        row.multi_token = decode_class ? 0 : 1;
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

    declared::ValueArena values;
    values.reset_pins_only(plan.value_count());
    values.bind_offsets(ws.declared_values.data(), ws.declared_values.nbytes(),
                        flat);
    declared::trace_arena("gptoss", plan, flat, ws.declared_values.nbytes(),
                          N, R);

    // WHAT THE CONVENTION WAS. Every entry here is a buffer some arm
    // above used to name, and the table is the record of that — the
    // next family converts by reading its own, not by rederiving it
    // from a hand-written pass.
    //
    // A pin WINS over the host's offset, so `PIE_DECLARED_HOST_ARENA=0`
    // is the whole table taking charge again and the A/B is exact. With
    // the host assigning, what survives is only what it DECLINED to
    // place: the values a seam exposes, which machinery outside this
    // walk reaches by name.
    {
        const bool host_arena = gpt_oss_host_arena_enabled();
        const std::size_t arena_lo = host_arena_lo();
        const std::size_t arena_hi = host_arena_hi();
        const std::size_t op_count = plan.op_count();
        for (std::size_t i = 0; i < op_count; ++i) {
            const PieForwardOp& op = plan.op(i);
            const auto outs = plan.outputs(op);
            if (outs.size == 0) continue;
            const auto place = [&](std::size_t which, void* ptr) {
                if (which >= outs.size) return;
                const std::uint32_t v = outs[which];
                if (host_arena && v < flat.value_offsets_len &&
                    flat.value_offsets[v] != declared::ValueArena::kNamed) {
                    const std::size_t owner =
                        v < flat.value_owners_len
                            ? static_cast<std::size_t>(flat.value_owners[v])
                            : static_cast<std::size_t>(v);
                    if (owner >= arena_lo && owner < arena_hi) return;
                }
                values.pin(v, ptr);
            };
            switch (op.kind) {
            case PieForwardOpKind::Embed:
                place(0, ws.y.data());
                break;
            case PieForwardOpKind::Rmsnorm:
                // The row norms are `Launch` now; their entry moved to
                // the Launch case below with the statement.
                break;
            case PieForwardOpKind::Matmul:
            case PieForwardOpKind::AddBias:
                place(0, ws.y.data());
                break;
            case PieForwardOpKind::Rope:
                place(0, ws.q.data());
                place(1, ws.k.data());
                break;
            case PieForwardOpKind::LmHead:
                place(0, ws.logits.data());
                break;
            case PieForwardOpKind::Launch: {
                const auto names = plan.aux_names(op);
                switch (declared::resolve_kernel(plan.weight_name(op))) {
                case declared::Kernel::RmsnormRow:
                case declared::Kernel::RmsnormRowGemma: {
                    // The MLP norm's result is read TWICE by the MoE
                    // block (router input and cast input), which is why
                    // it had a convention worth recording; the
                    // attention norm's is the projections'.
                    const ParsedName nm = parse_name(plan.name(names[0]));
                    place(0, nm.field == "mlp_norm" ? ws.norm_y.data()
                                                    : ws.norm_x.data());
                    break;
                }
                case declared::Kernel::GemmBias: {
                    const ParsedName nm = parse_name(plan.name(names[0]));
                    if (nm.field == "q_proj")      place(0, ws.q.data());
                    else if (nm.field == "k_proj") place(0, ws.k.data());
                    else if (nm.field == "v_proj") place(0, ws.v.data());
                    else if (nm.field == "router") place(0, ws.gate.data());
                    else throw_drift("biased projection on '" +
                                     std::string(plan.name(names[0])) + "'");
                    break;
                }
                case declared::Kernel::AttnFlashinferDecode:
                case declared::Kernel::AttnFlashinferPrefillPlanless:
                    place(0, ws.attn_out.data());
                    place(1, d_lse.data());
                    break;
                case declared::Kernel::AttnSinkRescale:
                    place(0, ws.attn_out.data());
                    break;
                case declared::Kernel::RopeYarnOriginal:
                    place(0, ws.q.data());
                    place(1, ws.k.data());
                    break;
                case declared::Kernel::TopkSoftmax:
                    place(0, d_topk_idx.data());
                    place(1, d_topk_w.data());
                    break;
                case declared::Kernel::Bf16ToFp16:
                    // The rank test the arm no longer needs, kept HERE
                    // because a pin table is exactly the place a
                    // convention's per-site choice belongs.
                    place(0, plan.value(outs[0]).rank == 2
                                 ? d_act_fp16.data()
                                 : d_route_act_fp16.data());
                    break;
                case declared::Kernel::Mxfp4GateUp:
                    place(0, d_route_gate.data());
                    place(1, d_route_up.data());
                    break;
                case declared::Kernel::GptOssGlu:
                    place(0, d_route_gate.data());
                    break;
                case declared::Kernel::Mxfp4Down:
                    place(0, d_route_out.data());
                    break;
                case declared::Kernel::WeightedSum:
                    place(0, d_moe_out.data());
                    break;
                case declared::Kernel::ResidualAdd:
                    place(0, ws.y.data());
                    break;
                case declared::Kernel::WriteKvToPages:
                    break;  // writes the cache, not a value
                }
                break;
            }
            default:
                break;
            }
        }
    }

    // An arm indexes operands positionally, and a span that is SHORTER
    // than the arm assumes is not a crash — indexing past its end reads
    // the next statement's operands and hands the arm a plausible
    // pointer to the wrong buffer. gemma-4's guard, for gemma-4's
    // reason.
    const auto need = [&](const auto& span, std::size_t n, const char* what) {
        if (span.size < n) {
            throw std::runtime_error(
                std::string("declared gptoss: ") + what + " states " +
                std::to_string(span.size) + " operands, needs " +
                std::to_string(n));
        }
    };

    // A value's trailing dims ARE its row width. Which is how `H`, `Hq`,
    // `Hk`, `num_experts` and `V` stop being executor bookkeeping: every
    // one of them is some traced value's row width, already stated.
    const auto row_width = [&](std::uint32_t id) {
        const auto& val = plan.value(id);
        std::uint32_t out = 1;
        for (std::uint32_t d = 1; d < val.rank; ++d) {
            if (val.dims[d].kind != pie_forward::PieForwardDimKind::Const) {
                return 0;
            }
            out *= val.dims[d].value;
        }
        return static_cast<int>(out);
    };

    const auto execute_op = [&](const PieForwardOp& op) {
        enter(op.layer);
        switch (op.kind) {
        case PieForwardOpKind::Embed: {
            // ISLAND (value arena). `token_ids` stays a driver input --
            // it is the fire's, not a traced value -- and everything
            // else the site named is the trace's.
            const auto outs = plan.outputs(op);
            need(outs, 1, "embed outputs");
            declared::arm_embed({plan, values, N, 0, stream}, op,
                                token_ids, wb.require(plan.weight_name(op)).data(), V);
            break;
        }
        case PieForwardOpKind::Rmsnorm:
            // RUNG 5: the semantic cascade is deleted -- a class
            // trace states which FOLD it runs (`cuda::rmsnorm`),
            // so this kind reaching the walk means the trace and
            // this executor drifted. Choosing here from a param
            // is what the statement now says instead.
            throw_drift("semantic Rmsnorm in a class trace "
                    "(the declaration states the fold)");

        case PieForwardOpKind::Matmul: {
            // ISLAND (value arena). `beta=1` is the whole of what this
            // arm was for: o_proj folds the residual, and the fold is
            // now STATED -- `Matmul { beta_one }` aliases output 0 over
            // input 1 (`kernels::semantic_in_place`), so the arena puts
            // the projection's result on the residual's own bytes and
            // the accumulate lands where it did under the pin table.
            //
            // Without that alias this conversion would have been the
            // rope bug again: `C = A·Bᵀ + C` reads C, and C would have
            // been a buffer nothing had written.
            const std::string_view name = plan.weight_name(op);
            const auto ins = plan.inputs(op);
            const auto outs = plan.outputs(op);
            need(ins, 1, "matmul inputs");
            need(outs, 1, "matmul outputs");
            kernels::gemm::act_x_wt_bf16(
                cublas.handle(), values.slot(ins[0]),
                wb.require(name).data(), values.slot(outs[0]), N,
                row_width(outs[0]), row_width(ins[0]),
                /*beta=*/op.param0 != 0 ? 1.f : 0.f);
            break;
        }
        case PieForwardOpKind::AddBias: {
            // ISLAND (value arena). The site checked for `o_bias` only
            // because that is the one bias gpt-oss adds separately; the
            // buffer and the width were never the name's to give.
            const std::string_view name = plan.weight_name(op);
            const auto outs = plan.outputs(op);
            need(outs, 1, "add_bias outputs");
            kernels::norm::add_bias_bf16(
                values.slot(outs[0]), wb.require(name).data(), N,
                row_width(outs[0]), stream);
            break;
        }
        case PieForwardOpKind::Rope: {
            // ISLAND (value arena). Rope rotates where q and k lie, and
            // the trace now says so, so `outs` and `ins` name one buffer
            // each -- see `kernels::semantic_in_place`. The head COUNTS
            // stay read from config: the rotation's head geometry is not
            // something a row width alone can give.
            const auto outs = plan.outputs(op);
            need(outs, 2, "rope outputs");
            kernels::rope::rope_bf16(
                values.slot(outs[0]), values.slot(outs[1]), positions, N,
                cfg.num_attention_heads, cfg.num_key_value_heads, d,
                cfg.rope_theta, stream);
            break;
        }
        case PieForwardOpKind::LmHead: {
            const std::string_view name = plan.weight_name(op);
            const auto ins = plan.inputs(op);
            const auto outs = plan.outputs(op);
            need(ins, 1, "lm_head inputs");
            need(outs, 1, "lm_head outputs");
            // SHARED ARM (D1): the compaction is identical in three
            // executors; only the head weight's resolution is not.
            int rows = N;
            const void* const input = declared::arm_epilogue_gather(
                {plan, values, N, 0, stream}, op, values.epilogue_gather(flat),
                logit_row_indices_d, num_logit_rows, &rows);
            lm_head_rows = rows;
            (void)lm_head_rows;
            kernels::gemm::act_x_wt_bf16(
                cublas.handle(), input, wb.require(name).data(),
                values.slot(outs[0]), rows, row_width(outs[0]),
                row_width(ins[0]), /*beta=*/0.f);
            break;
        }
        case PieForwardOpKind::Launch: {
            const std::string_view sym = plan.weight_name(op);
            const auto names = plan.aux_names(op);
            const auto aux = [&](std::size_t i) { return plan.name(names[i]); };
            const MixtralLayerWeights& layer =
                w.layers[static_cast<std::size_t>(cur_layer)];
            // THE SHARED SWITCH FIRST (D1). Every symbol whose arm is
            // family-blind lives in `declared/execute.hpp`; what remains
            // below is this family's RESIDUE. A `false` is an answer --
            // "stated, and this family executes it its own way".
            const declared::ExecCtx ectx{
                {plan, values, N, 0, stream},
                wb, cache, attn_ws, cublas, fwd_cfg.tp_comm,
                /*state_cache=*/nullptr,
                positions, qo_indptr, kv_page_indices, kv_page_indptr,
                kv_last_page_lens, row_valid_d,
                qo_indptr_h, kv_page_indptr_h,
                nullptr, nullptr, R,
                nullptr, false,
                eps, /*sm_scale=*/-1.f, d_lse.data(),
                cfg.rope_theta,
                cfg.num_attention_heads, cfg.num_key_value_heads, d, d,
                cur_layer,
                // One decode plan for the fire, built on the way in.
                // This family states no planned prefill (its prefill is
                // the plan-free wrapper) and every dispatch here
                // declares its own result, so there is no fallback to
                // hand over.
                decode_plan.get(), /*prefill_plan=*/nullptr,
                /*region_dst=*/nullptr,
            };
            if (declared::execute_shared(ectx, op)) break;
            switch (declared::resolve_kernel(sym)) {
            case declared::Kernel::RmsnormRow:
            case declared::Kernel::GemmBias: {
                // ISLAND (value arena). Four sites told apart by the
                // projection they name, and every branch chose a buffer
                // pair and a column count the trace already states: q/k/v
                // read the attention norm and write their staging
                // buffers, the router reads the MLP norm and writes
                // `ws.gate`. The `[N, E]` scratch the hand pass borrowed
                // for the router is just the router's output value.
                //
                // `num_experts`, `Hq` and `Hk` were the executor's way
                // of knowing those column counts; they are the output's
                // row width, per site, without the executor knowing
                // which site it is in.
                const std::string_view proj = aux(0);
                const auto ins = plan.inputs(op);
                const auto outs = plan.outputs(op);
                need(ins, 1, "biased projection inputs");
                need(outs, 1, "biased projection outputs");
                kernels::gemm::act_x_wt_bias_bf16(
                    cublas.handle(), values.slot(ins[0]),
                    wb.require(proj).data(), wb.require(aux(1)).data(),
                    values.slot(outs[0]), N, row_width(outs[0]),
                    row_width(ins[0]), stream);
                break;
            }
            // The decode dispatch left with the plan. It was already
            // calling the shared `arm_attention_decode` -- what kept the
            // case here was that the arm needed a plan, a cache view and
            // an LSE fallback handed to it, and all three are context
            // fields now.
            // The SINK RESCALE generates. Its LSE is the dispatch's
            // second RESULT -- only a sink layer declares one -- so it is
            // operand 1 and traced, rather than a scratch this executor
            // remembered handing the dispatch.
            case declared::Kernel::RopeYarnOriginal: {
                // ISLAND (value arena). Argument for argument the hand
                // pass's `apply_rope` arm; the yarn params come off the
                // shared cfg, which had resolved them all along, and the
                // two buffers come off the statement.
                const auto outs = plan.outputs(op);
                need(outs, 2, "yarn rope outputs");
                kernels::rope::rope_yarn_original_bf16(
                    values.slot(outs[0]), values.slot(outs[1]), positions, N,
                    cfg.num_attention_heads, cfg.num_key_value_heads, d,
                    cfg.rope_theta, fwd_cfg.yarn_factor,
                    fwd_cfg.yarn_beta_fast, fwd_cfg.yarn_beta_slow,
                    fwd_cfg.yarn_attention_factor,
                    fwd_cfg.yarn_original_max_position, stream);
                break;
            }
            case declared::Kernel::Mxfp4GateUp: {
                // ISLAND (value arena). The expert BANKS stay reached
                // through `w.layers`: they are per-expert pointer arrays
                // and not tensors, which is the same reason `bind`
                // refuses to name them.
                const auto ins = plan.inputs(op);
                const auto outs = plan.outputs(op);
                need(ins, 2, "mxfp4 gate_up inputs");
                need(outs, 2, "mxfp4 gate_up outputs");
                kernels::quant::mxfp4_moe_gate_up_decode_bf16(
                    values.slot(ins[1]),
                    static_cast<const std::int32_t*>(values.slot(ins[0])),
                    layer.expert_gate_up_packed_ptrs.data(),
                    layer.expert_gate_up_scale_ptrs.data(),
                    layer.expert_gate_bias_ptrs.data(),
                    layer.expert_up_bias_ptrs.data(),
                    values.slot(outs[0]), values.slot(outs[1]),
                    N, top_k, H, I, stream);
                break;
            }
            // The clamped GLU GENERATES. `gate = glu(gate, up)` is why
            // the pointer appeared twice and the alias is stated; the
            // CLAMP was the last thing left, reached out of a config
            // struct, and it is a number the host has — so it rides the
            // param channel like every other load-time constant.
            case declared::Kernel::Mxfp4Down: {
                // ISLAND (value arena).
                const auto ins = plan.inputs(op);
                const auto outs = plan.outputs(op);
                need(ins, 2, "mxfp4 down inputs");
                need(outs, 1, "mxfp4 down outputs");
                kernels::quant::mxfp4_moe_down_decode_bf16(
                    values.slot(ins[1]),
                    static_cast<const std::int32_t*>(values.slot(ins[0])),
                    layer.expert_down_packed_ptrs.data(),
                    layer.expert_down_scale_ptrs.data(),
                    layer.expert_down_bias_ptrs.data(),
                    values.slot(outs[0]), N, top_k, H, I, stream);
                break;
            }
            }
            break;
        }
        case PieForwardOpKind::HookSite:
            // Observation-only, and arc 2 admits no hooked fire
            // (`in.stage_hooks == nullptr` is an eligibility term), so
            // there is never a program to invoke. The op is STATED
            // because the seam is real and its position is checked; this
            // arm is what makes the trace executable instead of a load
            // failure. When gpt-oss grows hook support, the invoke goes
            // here, next to qwen3_5's.
            break;
        default:
            throw_drift("op kind " +
                        std::to_string(static_cast<std::uint32_t>(op.kind)));
        }
    };

    // Say ONCE that this drive took a fire. Without it, coherent output
    // is evidence about the hand-written pass as easily as about this
    // one.
    {
        static std::atomic<bool> said[2] = {{false}, {false}};
        if (!said[decode_class ? 0 : 1].exchange(true)) {
            std::fprintf(stderr,
                         "[declared-gptoss] first %s fire: N=%d R=%d "
                         "routes=%d ops=%zu\n",
                         decode_class ? "DECODE" : "PREFILL",
                         N, R, routes, plan.op_count());
        }
    }

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

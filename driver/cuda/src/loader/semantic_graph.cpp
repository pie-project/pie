#include "loader/semantic_graph.hpp"

#include <algorithm>
#include <cstring>
#include <unordered_set>
#include <utility>

#include <pie_driver_common/tensor_names.hpp>

#include "loader/runtime_abi.hpp"

namespace pie_cuda_driver {

namespace {

bool ends_with(const std::string& s, const char* suffix) {
    const auto n = std::char_traits<char>::length(suffix);
    return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

bool can_pack_2d_bf16_group(
    const CheckpointSource& loader,
    const std::vector<std::string>& raw_names)
{
    if (raw_names.empty()) return false;
    std::int64_t cols = -1;
    for (const auto& raw : raw_names) {
        if (!loader.contains(raw)) return false;
        const auto& info = loader.info(raw);
        if (info.dtype != DType::BF16 || info.shape.size() != 2) return false;
        if (cols < 0) {
            cols = info.shape[1];
        } else if (info.shape[1] != cols) {
            return false;
        }
    }
    return true;
}

struct PerExpertMoeName {
    std::string base;
    int expert = -1;
    std::string projection;
};

bool try_parse_per_expert_moe_weight(
    const std::string& name,
    PerExpertMoeName& out)
{
    const std::string marker = ".experts.";
    const auto marker_pos = name.find(marker);
    if (marker_pos == std::string::npos) return false;
    const std::size_t idx_begin = marker_pos + marker.size();
    const auto idx_end = name.find('.', idx_begin);
    if (idx_end == std::string::npos || idx_end == idx_begin) return false;
    const std::string idx_text = name.substr(idx_begin, idx_end - idx_begin);
    if (!std::all_of(idx_text.begin(), idx_text.end(), [](unsigned char ch) {
            return ch >= '0' && ch <= '9';
        })) {
        return false;
    }

    const std::string suffix = name.substr(idx_end);
    std::string projection;
    if (suffix == ".gate_proj.weight") {
        projection = "gate";
    } else if (suffix == ".up_proj.weight") {
        projection = "up";
    } else if (suffix == ".down_proj.weight") {
        projection = "down";
    } else {
        return false;
    }

    out.base = name.substr(0, marker_pos);
    out.expert = std::stoi(idx_text);
    out.projection = std::move(projection);
    return true;
}

std::string per_expert_raw_name(
    const std::string& base,
    int expert,
    const char* projection)
{
    return base + ".experts." + std::to_string(expert) + "." +
           projection + "_proj.weight";
}

void add_semantic_group_once(
    SemanticGraph& graph,
    std::unordered_set<std::string>& seen,
    SemanticGroupKind kind,
    std::string runtime_base,
    std::vector<std::string> raw_names,
    std::vector<std::string> runtime_names,
    std::vector<SemanticRole> raw_roles = {},
    std::vector<SemanticRole> runtime_roles = {})
{
    const std::string key =
        std::to_string(static_cast<int>(kind)) + ":" + runtime_base;
    if (!seen.insert(key).second) return;
    graph.groups.push_back(SemanticGroup{
        .kind = kind,
        .runtime_base = std::move(runtime_base),
        .raw_names = std::move(raw_names),
        .runtime_names = std::move(runtime_names),
        .raw_roles = std::move(raw_roles),
        .runtime_roles = std::move(runtime_roles),
    });
}

void discover_semantic_tensor_groups(
    SemanticGraph& graph,
    const HfConfig& hf,
    const CheckpointSource& loader)
{
    std::unordered_set<std::string> seen;
    seen.reserve(graph.tensors.size());

    for (const auto& semantic : graph.tensors) {
        constexpr const char* q_suffix = ".self_attn.q_proj.weight";
        constexpr const char* k_suffix = ".self_attn.k_proj.weight";
        constexpr const char* v_suffix = ".self_attn.v_proj.weight";
        constexpr const char* gate_suffix = ".mlp.gate_proj.weight";
        constexpr const char* up_suffix = ".mlp.up_proj.weight";

        if (ends_with(semantic.runtime_name, q_suffix) ||
            ends_with(semantic.runtime_name, k_suffix) ||
            ends_with(semantic.runtime_name, v_suffix)) {
            const char* matched = ends_with(semantic.runtime_name, q_suffix)
                ? q_suffix
                : (ends_with(semantic.runtime_name, k_suffix) ? k_suffix : v_suffix);
            const std::string raw_base =
                semantic.raw_name.substr(
                    0, semantic.raw_name.size() - std::strlen(matched));
            const std::string runtime_base =
                semantic.runtime_name.substr(
                    0, semantic.runtime_name.size() - std::strlen(matched));
            const std::vector<std::string> raw_names = {
                raw_base + q_suffix,
                raw_base + k_suffix,
                raw_base + v_suffix,
            };
            if (can_pack_2d_bf16_group(loader, raw_names)) {
                add_semantic_group_once(
                    graph, seen, SemanticGroupKind::PackedQkv,
                    runtime_base + ".self_attn", raw_names,
                    {runtime_base + q_suffix,
                     runtime_base + k_suffix,
                     runtime_base + v_suffix},
                    {SemanticRole::AttentionQ,
                     SemanticRole::AttentionK,
                     SemanticRole::AttentionV},
                    {SemanticRole::AttentionQ,
                     SemanticRole::AttentionK,
                     SemanticRole::AttentionV});
            }
        }

        if (ends_with(semantic.runtime_name, gate_suffix) ||
            ends_with(semantic.runtime_name, up_suffix)) {
            const char* matched = ends_with(semantic.runtime_name, gate_suffix)
                ? gate_suffix
                : up_suffix;
            const std::string raw_base =
                semantic.raw_name.substr(
                    0, semantic.raw_name.size() - std::strlen(matched));
            const std::string runtime_base =
                semantic.runtime_name.substr(
                    0, semantic.runtime_name.size() - std::strlen(matched));
            const std::vector<std::string> raw_names = {
                raw_base + gate_suffix,
                raw_base + up_suffix,
            };
            if (can_pack_2d_bf16_group(loader, raw_names)) {
                add_semantic_group_once(
                    graph, seen, SemanticGroupKind::PackedGateUp,
                    runtime_base + ".mlp", raw_names,
                    {runtime_base + gate_suffix,
                     runtime_base + up_suffix},
                    {SemanticRole::MlpGate,
                     SemanticRole::MlpUp},
                    {SemanticRole::MlpGate,
                     SemanticRole::MlpUp});
            }
        }

        if (semantic.role == SemanticRole::AttentionQkv) {
            constexpr const char* qkv_suffix = ".self_attn.qkv_proj.weight";
            if (ends_with(semantic.runtime_name, qkv_suffix)) {
                const std::string runtime_base =
                    semantic.runtime_name.substr(
                        0, semantic.runtime_name.size() - std::strlen(qkv_suffix));
                add_semantic_group_once(
                    graph, seen, SemanticGroupKind::RowRangeSplit,
                    semantic.runtime_name,
                    {semantic.raw_name},
                    {runtime_base + ".self_attn.q_proj.weight",
                     runtime_base + ".self_attn.k_proj.weight",
                     runtime_base + ".self_attn.v_proj.weight"},
                    {SemanticRole::AttentionQkv},
                    {SemanticRole::AttentionQ,
                     SemanticRole::AttentionK,
                     SemanticRole::AttentionV});
            }
        }

        if (semantic.role == SemanticRole::MlpGateUp) {
            constexpr const char* gate_up_suffix = ".mlp.gate_up_proj.weight";
            if (ends_with(semantic.runtime_name, gate_up_suffix)) {
                const std::string runtime_base =
                    semantic.runtime_name.substr(
                        0, semantic.runtime_name.size() -
                               std::strlen(gate_up_suffix));
                add_semantic_group_once(
                    graph, seen, SemanticGroupKind::RowRangeSplit,
                    semantic.runtime_name,
                    {semantic.raw_name},
                    {runtime_base + ".mlp.gate_proj.weight",
                     runtime_base + ".mlp.up_proj.weight"},
                    {SemanticRole::MlpGateUp},
                    {SemanticRole::MlpGate,
                     SemanticRole::MlpUp});
            }
        }

        PerExpertMoeName expert;
        PerExpertMoeName raw_expert;
        if (try_parse_per_expert_moe_weight(semantic.runtime_name, expert) &&
            try_parse_per_expert_moe_weight(semantic.raw_name, raw_expert)) {
            int expert_count = hf.num_experts;
            if (expert_count <= 0) {
                expert_count = 0;
                for (;;) {
                    const std::string gate =
                        per_expert_raw_name(raw_expert.base, expert_count, "gate");
                    const std::string up =
                        per_expert_raw_name(raw_expert.base, expert_count, "up");
                    const std::string down =
                        per_expert_raw_name(raw_expert.base, expert_count, "down");
                    if (!loader.contains(gate) ||
                        !loader.contains(up) ||
                        !loader.contains(down)) {
                        break;
                    }
                    ++expert_count;
                }
            }
            std::vector<std::string> raw_names;
            std::vector<SemanticRole> raw_roles;
            raw_names.reserve(static_cast<std::size_t>(std::max(expert_count, 0)) * 3);
            raw_roles.reserve(raw_names.capacity());
            for (int e = 0; e < expert_count; ++e) {
                raw_names.push_back(per_expert_raw_name(raw_expert.base, e, "gate"));
                raw_roles.push_back(SemanticRole::MoeExpertGate);
                raw_names.push_back(per_expert_raw_name(raw_expert.base, e, "up"));
                raw_roles.push_back(SemanticRole::MoeExpertUp);
                raw_names.push_back(per_expert_raw_name(raw_expert.base, e, "down"));
                raw_roles.push_back(SemanticRole::MoeExpertDown);
            }
            const auto expert_bank =
                pie_cuda_runtime_abi().fused_expert_bank(expert.base);
            add_semantic_group_once(
                graph, seen, SemanticGroupKind::PerExpertMoe,
                expert.base + ".experts", std::move(raw_names),
                {expert_bank.gate_up_name, expert_bank.down_name},
                std::move(raw_roles),
                {SemanticRole::MoeExpertsGateUp,
                 SemanticRole::MoeExpertsDown});
        }

        if (ends_with(semantic.runtime_name, ".experts.gate_up_proj") ||
            ends_with(semantic.runtime_name, ".experts.down_proj")) {
            const auto experts_pos = semantic.runtime_name.rfind(".experts.");
            const std::string runtime_base =
                semantic.runtime_name.substr(0, experts_pos);
            add_semantic_group_once(
                graph, seen, SemanticGroupKind::FusedMoeExperts,
                runtime_base + ".experts", {}, {},
                {},
                {SemanticRole::MoeExpertsGateUp,
                 SemanticRole::MoeExpertsDown});
        }

        if (hf.model_type == "gpt_oss" &&
            (ends_with(semantic.runtime_name,
                       ".mlp.experts.gate_up_proj_blocks") ||
             ends_with(semantic.runtime_name,
                       ".mlp.experts.down_proj_blocks"))) {
            const std::string raw_base =
                semantic.raw_name.substr(
                    0, semantic.raw_name.rfind("_blocks"));
            const std::string runtime_base =
                semantic.runtime_name.substr(
                    0, semantic.runtime_name.rfind("_blocks"));
            add_semantic_group_once(
                graph, seen, SemanticGroupKind::GptOssMxfp4,
                runtime_base,
                {raw_base + "_blocks",
                 raw_base + "_scales",
                 raw_base + "_bias"},
                {runtime_base + ".weight",
                 runtime_base + ".weight_scale",
                 runtime_base + ".bias"},
                {SemanticRole::QuantPackedData,
                 SemanticRole::QuantScale,
                 SemanticRole::Bias},
                {SemanticRole::QuantPackedData,
                 SemanticRole::QuantScale,
                 SemanticRole::Bias});
        }

        if (semantic.checkpoint_dtype == DType::FP8_E4M3 &&
            loader.contains(semantic.raw_name + "_scale_inv")) {
            add_semantic_group_once(
                graph, seen, SemanticGroupKind::Fp8ScaleInv,
                semantic.runtime_name,
                {semantic.raw_name, semantic.raw_name + "_scale_inv"},
                {semantic.runtime_name,
                 pie_cuda_runtime_abi().quant_scale_inv_name(
                     semantic.runtime_name)});
        }
    }
}

}  // namespace

SemanticRole infer_semantic_role(const std::string& name) {
    if (ends_with(name, ".embed_tokens.weight")) {
        return SemanticRole::Embedding;
    }
    if (name == "lm_head.weight" || ends_with(name, ".lm_head.weight")) {
        return SemanticRole::LmHead;
    }
    if (ends_with(name, ".input_layernorm.weight") ||
        ends_with(name, ".post_attention_layernorm.weight") ||
        ends_with(name, ".norm.weight") ||
        ends_with(name, ".q_norm.weight") ||
        ends_with(name, ".k_norm.weight")) {
        return SemanticRole::Norm;
    }
    if (ends_with(name, ".self_attn.q_proj.weight")) {
        return SemanticRole::AttentionQ;
    }
    if (ends_with(name, ".self_attn.k_proj.weight")) {
        return SemanticRole::AttentionK;
    }
    if (ends_with(name, ".self_attn.v_proj.weight")) {
        return SemanticRole::AttentionV;
    }
    if (ends_with(name, ".self_attn.o_proj.weight")) {
        return SemanticRole::AttentionO;
    }
    if (ends_with(name, ".self_attn.qkv_proj.weight")) {
        return SemanticRole::AttentionQkv;
    }
    if (ends_with(name, ".mlp.gate_proj.weight")) {
        return SemanticRole::MlpGate;
    }
    if (ends_with(name, ".mlp.up_proj.weight")) {
        return SemanticRole::MlpUp;
    }
    if (ends_with(name, ".mlp.down_proj.weight")) {
        return SemanticRole::MlpDown;
    }
    if (ends_with(name, ".mlp.gate_up_proj.weight")) {
        return SemanticRole::MlpGateUp;
    }
    if (ends_with(name, ".experts.gate_up_proj")) {
        return SemanticRole::MoeExpertsGateUp;
    }
    if (ends_with(name, ".experts.down_proj")) {
        return SemanticRole::MoeExpertsDown;
    }
    if (ends_with(name, "_blocks")) {
        return SemanticRole::QuantPackedData;
    }
    if (ends_with(name, "_bias") || ends_with(name, ".bias")) {
        return SemanticRole::Bias;
    }
    if (ends_with(name, ".gate_proj.weight")) {
        return SemanticRole::MoeExpertGate;
    }
    if (ends_with(name, ".up_proj.weight")) {
        return SemanticRole::MoeExpertUp;
    }
    if (ends_with(name, ".down_proj.weight")) {
        return SemanticRole::MoeExpertDown;
    }
    if (ends_with(name, ".w1.weight")) {
        return SemanticRole::MoeExpertGate;
    }
    if (ends_with(name, ".w3.weight")) {
        return SemanticRole::MoeExpertUp;
    }
    if (ends_with(name, ".w2.weight")) {
        return SemanticRole::MoeExpertDown;
    }
    if (ends_with(name, ".weight_scale") ||
        ends_with(name, ".weight_scale_inv") ||
        ends_with(name, ".scales")) {
        return SemanticRole::QuantScale;
    }
    if (ends_with(name, ".weight_zero_point") ||
        ends_with(name, ".qzeros") ||
        ends_with(name, ".zero_point")) {
        return SemanticRole::QuantZeroPoint;
    }
    return SemanticRole::Unknown;
}

SemanticGraph build_semantic_graph(
    const HfConfig& hf,
    const CheckpointSource& loader)
{
    SemanticGraph graph;
    graph.tensors.reserve(loader.num_tensors());

    const std::string& mm_strip = hf.mm_lm_strip_prefix;
    const auto& mm_skip = hf.mm_skip_prefixes;
    for (const auto& raw_name : loader.tensor_names()) {
        if (pie_driver_common::starts_with_any(raw_name, mm_skip)) continue;
        const std::string runtime_name =
            pie_driver_common::strip_prefix(raw_name, mm_strip);
        const auto& info = loader.info(raw_name);
        graph.tensors.push_back(SemanticTensor{
            .raw_name = raw_name,
            .runtime_name = runtime_name,
            .role = infer_semantic_role(runtime_name),
            .checkpoint_dtype = info.dtype,
            .checkpoint_shape = info.shape,
            .shard_axis = llama_like_shard_axis(runtime_name),
        });
    }
    discover_semantic_tensor_groups(graph, hf, loader);
    return graph;
}

bool semantic_graph_has_group(
    const SemanticGraph& graph,
    SemanticGroupKind kind)
{
    return std::any_of(
        graph.groups.begin(), graph.groups.end(),
        [kind](const SemanticGroup& group) {
            return group.kind == kind;
        });
}

int llama_like_shard_axis(const std::string& name) {
    if (ends_with(name, ".q_proj.weight") || ends_with(name, ".q_proj.bias") ||
        ends_with(name, ".k_proj.weight") || ends_with(name, ".k_proj.bias") ||
        ends_with(name, ".v_proj.weight") || ends_with(name, ".v_proj.bias") ||
        ends_with(name, ".gate_proj.weight") ||
        ends_with(name, ".up_proj.weight") ||
        ends_with(name, ".sinks")) {
        return 0;
    }
    if (ends_with(name, ".o_proj.weight") || ends_with(name, ".down_proj.weight")) {
        return 1;
    }
    if (ends_with(name, ".w1.weight") || ends_with(name, ".w3.weight") ||
        ends_with(name, ".w1.bias")   || ends_with(name, ".w3.bias")) {
        return 0;
    }
    if (ends_with(name, ".w2.weight")) {
        return 1;
    }
    if (ends_with(name, ".q_proj.weight_scale") ||
        ends_with(name, ".q_proj.weight_scale_inv") ||
        ends_with(name, ".k_proj.weight_scale") ||
        ends_with(name, ".k_proj.weight_scale_inv") ||
        ends_with(name, ".v_proj.weight_scale") ||
        ends_with(name, ".v_proj.weight_scale_inv") ||
        ends_with(name, ".gate_proj.weight_scale") ||
        ends_with(name, ".gate_proj.weight_scale_inv") ||
        ends_with(name, ".up_proj.weight_scale") ||
        ends_with(name, ".up_proj.weight_scale_inv")) {
        return 0;
    }
    if (ends_with(name, ".linear_attn.in_proj_z.weight") ||
        ends_with(name, ".linear_attn.in_proj_b.weight") ||
        ends_with(name, ".linear_attn.in_proj_a.weight") ||
        ends_with(name, ".linear_attn.dt_bias") ||
        ends_with(name, ".linear_attn.A_log")) {
        return 0;
    }
    if (ends_with(name, ".linear_attn.out_proj.weight")) {
        return 1;
    }
    return -1;
}

}  // namespace pie_cuda_driver

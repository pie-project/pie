#pragma once

/// What each model family asks the loader to build.
///
/// This is the *program* half of `plan = compile(source_facts, program,
/// target)`, and it is the half that knows what a model is. It used to live in
/// the loader as `loader/src/arch/`, which made `compile` a function of a
/// model's name — a family the loader had never heard of could not be loaded
/// even when the driver knew exactly what it wanted, and a family the driver
/// bound differently from the loader's guess failed at bind time with a missing
/// weight rather than at compile time with a message.
///
/// Everything here reads two things and nothing else: the checkpoint's tensor
/// table (`pie_loader::Checkpoint`, which is the same parse the compile will
/// use) and the config facts the driver already parsed. Neither is a second
/// reading of anything, which is the point: the previous arrangement had the
/// driver parse `config.json` for its own model and the loader parse it again
/// for the contract, and two readings that can disagree is one too many.
///
/// It is shared between the CUDA and Metal drivers because the knowledge is
/// shared. "Qwen3-MoE binds q/k/v separately" is a fact about Qwen3-MoE, not
/// about a GPU. What *is* backend-specific is stated as such — the fused dense
/// projections are a CUDA GEMM decision, and the Metal Qwen3.5 schema renames
/// every tensor because MLX's binder expects different names.

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "pie_loader.h"
#include "pie_loader/model_contract.hpp"
#include "pie_loader/request.hpp"
#include "pie_loader/source_checkpoint.hpp"

namespace pie_driver {

using pie_loader::Checkpoint;
using pie_loader::ModelContract;
using pie_loader::Node;
using pie_loader::SourceTensor;

// The generated ABI types live in `pie_loader` too. Naming them one by one
// rather than opening the whole namespace keeps this header from leaking
// `pie_loader` into every translation unit that binds a model.
using pie_loader::PieLoaderBackendKind;
using pie_loader::PieLoaderDType;
using pie_loader::PieLoaderEncodingKind;
using pie_loader::PieLoaderEncodingSpec;
using pie_loader::PieLoaderQuantScheme;
using pie_loader::PieLoaderQuantSpecView;
using pie_loader::PieLoaderRepackLayout;
using pie_loader::PieLoaderRepackSpecView;
using pie_loader::PieLoaderRowMap;

/// Which half of a multimodal checkpoint to load.
///
/// A driver concept, not a loader one: the loader is told a contract, and a
/// contract that should not load the vision tower simply does not declare it.
/// The distinction lives here because it is the *author* that has to decide
/// what to declare.
enum class Component : std::uint32_t {
    Full = 0,
    Text = 1,
    Encode = 2,
};

/// How this driver wants MXFP4 expert weights lowered.
///
/// Also a driver concept for the same reason: an expert weight is MXFP4 in the
/// plan because a contract node says so, not because a flag on the side said the
/// model was one of the ones that should be.
enum class Mxfp4MoePolicy : std::uint32_t {
    RoutedDecode = 0,
    NativeGemm = 1,
    EagerBf16 = 2,
};

/// What the caller asked for, before the device has had its say.
enum class Mxfp4MoeRequest : std::uint32_t {
    Auto = 0,
    RoutedDecode = 1,
    NativeGemm = 2,
    EagerBf16 = 3,
};

/// The config facts a family needs beyond the checkpoint itself.
///
/// Five values. Everything else a contract needs — which tensors exist, what
/// shape they are, how they are encoded, whether the experts ship stacked or
/// per-expert — is in the checkpoint, and reading it there rather than
/// predicting it from `config.json` is why `expect(shape)` can be an assertion
/// instead of a guess.
struct ModelFacts {
    std::string model_type;
    /// `quantization_config.quant_method`, empty for an unquantized checkpoint.
    std::string quant_method;
    std::uint32_t num_hidden_layers = 0;
    std::uint32_t num_experts = 0;
};

namespace contract_detail {

// -- vocabulary over the FFI's encoding POD -----------------------------------

inline PieLoaderEncodingKind kind_of(const PieLoaderEncodingSpec& e) {
    return static_cast<PieLoaderEncodingKind>(e.kind);
}

inline bool is_raw(const PieLoaderEncodingSpec& e, PieLoaderDType dtype) {
    return kind_of(e) == PieLoaderEncodingKind::Raw &&
           static_cast<PieLoaderDType>(e.dtype) == dtype;
}

/// The dtype a consumer sees, whatever the storage format is.
inline PieLoaderDType logical_dtype(const PieLoaderEncodingSpec& e) {
    return kind_of(e) == PieLoaderEncodingKind::Raw
               ? static_cast<PieLoaderDType>(e.dtype)
               : static_cast<PieLoaderDType>(e.quant.logical_dtype);
}

inline bool same_encoding(const PieLoaderEncodingSpec& a, const PieLoaderEncodingSpec& b) {
    if (a.kind != b.kind) {
        return false;
    }
    if (kind_of(a) == PieLoaderEncodingKind::Raw) {
        return a.dtype == b.dtype;
    }
    return a.quant.scheme == b.quant.scheme && a.quant.logical_dtype == b.quant.logical_dtype &&
           a.quant.bits_per_element == b.quant.bits_per_element &&
           a.quant.group_size == b.quant.group_size &&
           a.quant.channel_axis == b.quant.channel_axis &&
           a.quant.scale_dtype == b.quant.scale_dtype &&
           a.quant.zero_point_dtype == b.quant.zero_point_dtype;
}

inline std::uint8_t default_bits(PieLoaderQuantScheme scheme) {
    switch (scheme) {
        case PieLoaderQuantScheme::AwqInt4:
        case PieLoaderQuantScheme::GptqInt4:
        case PieLoaderQuantScheme::Mxfp4E2M1E8M0:
        case PieLoaderQuantScheme::MlxAffineU4:
        case PieLoaderQuantScheme::GgufQ4_0:
        case PieLoaderQuantScheme::GgufQ4K:
            return 4;
        case PieLoaderQuantScheme::GgufQ5_0:
        case PieLoaderQuantScheme::GgufQ5K:
            return 5;
        default:
            return 8;
    }
}

/// Whether this encoding lays elements out at a whole number of bytes each.
///
/// The passes below all build byte-addressed views — a row band, a per-expert
/// stack, a strided column window — and none of them is meaningful over an
/// encoding whose elements straddle byte boundaries. Checking it here turns
/// that into a message naming the tensor rather than a wrong extent later.
inline bool is_dense_addressable(const PieLoaderEncodingSpec& e) {
    if (kind_of(e) == PieLoaderEncodingKind::Raw) {
        return true;
    }
    const std::uint8_t bits =
        e.quant.bits_per_element != 0
            ? e.quant.bits_per_element
            : default_bits(static_cast<PieLoaderQuantScheme>(e.quant.scheme));
    return bits % 8 == 0;
}

// -- name-pattern policy ------------------------------------------------------

inline bool ends_with_any(std::string_view value, std::initializer_list<std::string_view> tails) {
    for (std::string_view tail : tails) {
        if (value.size() >= tail.size() && value.compare(value.size() - tail.size(), tail.size(),
                                                         tail) == 0) {
            return true;
        }
    }
    return false;
}

inline bool ends_with(std::string_view value, std::string_view tail) {
    return ends_with_any(value, {tail});
}

inline bool contains(std::string_view value, std::string_view needle) {
    return value.find(needle) != std::string_view::npos;
}

/// No axis. `std::optional<std::uint8_t>` rather than a sentinel because "not
/// sharded" and "sharded on axis 0" are answers a bug can swap.
using ShardAxis = std::optional<std::uint8_t>;

inline ShardAxis llama_like_shard_axis(std::string_view name) {
    if (contains(name, ".mlp.experts.")) {
        if (ends_with_any(name, {".gate_proj.weight_packed", ".gate_proj.weight_scale",
                                 ".up_proj.weight_packed", ".up_proj.weight_scale"})) {
            return std::uint8_t{0};
        }
        if (ends_with_any(name, {".down_proj.weight_packed", ".down_proj.weight_scale"})) {
            return std::uint8_t{1};
        }
    }
    if (ends_with_any(
            name,
            {".q_proj.weight",        ".q_proj.bias",
             ".k_proj.weight",        ".k_proj.bias",
             ".v_proj.weight",        ".v_proj.bias",
             ".gate_proj.weight",     ".up_proj.weight",
             ".sinks",                ".w1.weight",
             ".w3.weight",            ".w1.bias",
             ".w3.bias",              ".q_proj.weight_scale",
             ".q_proj.weight_scale_inv", ".k_proj.weight_scale",
             ".k_proj.weight_scale_inv", ".v_proj.weight_scale",
             ".v_proj.weight_scale_inv", ".gate_proj.weight_scale",
             ".gate_proj.weight_scale_inv", ".up_proj.weight_scale",
             ".up_proj.weight_scale_inv", ".linear_attn.in_proj_z.weight",
             ".linear_attn.in_proj_b.weight", ".linear_attn.in_proj_a.weight",
             ".linear_attn.dt_bias",  ".linear_attn.A_log",
             ".self_attn.q_b_proj.weight", ".self_attn.kv_b_proj.weight"})) {
        return std::uint8_t{0};
    }
    if (ends_with_any(name, {".o_proj.weight", ".down_proj.weight", ".w2.weight",
                             ".linear_attn.out_proj.weight"})) {
        return std::uint8_t{1};
    }
    if (ends_with(name, ".experts.down_proj") || ends_with(name, ".mlp.experts.down_proj")) {
        return std::uint8_t{2};
    }
    return std::nullopt;
}

inline ShardAxis dsv4_shard_axis(std::string_view name) {
    // Routed experts: shard the intermediate dim within each expert. w1/w3 on
    // axis 0 (gate/up out dim), w2 on axis 1 (down in dim). Each rank computes
    // a partial expert output; the results are combined by an all-reduce.
    if (contains(name, ".ffn.experts.")) {
        if (ends_with_any(name, {".w1.weight", ".w1.scale", ".w3.weight", ".w3.scale"})) {
            return std::uint8_t{0};
        }
        if (ends_with_any(name, {".w2.weight", ".w2.scale"})) {
            return std::uint8_t{1};
        }
    }
    if (ends_with_any(name, {".shared_experts.w1.weight", ".shared_experts.w1.scale",
                             ".shared_experts.w3.weight", ".shared_experts.w3.scale"})) {
        return std::uint8_t{0};
    }
    if (ends_with_any(name, {".shared_experts.w2.weight", ".shared_experts.w2.scale"})) {
        return std::uint8_t{1};
    }
    // Everything else replicated, which avoids TP communication in the main path.
    return std::nullopt;
}

inline bool is_glm_expert_weight(std::string_view name) {
    return (contains(name, ".mlp.experts.") || contains(name, ".mlp.shared_experts.")) &&
           ends_with_any(name, {".gate_proj.weight", ".up_proj.weight", ".down_proj.weight"});
}

inline bool runtime_quantizable_name(std::string_view name, PieLoaderQuantScheme scheme) {
    if (scheme == PieLoaderQuantScheme::Mxfp4E2M1E8M0) {
        // For FP4 only GLM-5.1's routed and shared experts are touched.
        // Attention projections stay FP8-plus-scale (block-scaled GEMM) because
        // there is no FP4 GEMM path for them on this hardware.
        return is_glm_expert_weight(name);
    }
    return ends_with_any(name, {".self_attn.q_proj.weight", ".self_attn.k_proj.weight",
                                ".self_attn.v_proj.weight", ".self_attn.o_proj.weight",
                                ".self_attn.q_a_proj.weight", ".self_attn.q_b_proj.weight",
                                ".self_attn.kv_a_proj_with_mqa.weight",
                                ".self_attn.kv_b_proj.weight", ".mlp.gate_proj.weight",
                                ".mlp.up_proj.weight", ".mlp.down_proj.weight"}) ||
           is_glm_expert_weight(name);
}

// -- the per-architecture facts table -----------------------------------------

/// Declarative per-architecture facts, so that adding a model is one table edit
/// rather than a new `model_type ==` in half a dozen places.
struct ArchProfile {
    /// Source tensors live under this prefix; it is stripped from output names.
    std::string_view source_prefix;
    /// MLA attention: fuse q_a + kv_a and the shared gate+up (DeepSeek/Kimi).
    bool mla_fused_joins = false;
    /// Phi-3 fused qkv / gate_up splits.
    bool phi3_fused_splits = false;
    /// Nemotron-H packed-expert views.
    bool nemotron_packed_experts = false;
    /// GPT-OSS native MXFP4 expert groups.
    bool gpt_oss_mxfp4_groups = false;
    /// Shard `embed_tokens` on axis 0 under TP, to save per-rank memory.
    bool shard_embed_tokens = false;
    /// Keep `lm_head` replicated under TP. Costs ~1.7 GB a rank on Kimi K2.6
    /// and buys not needing a TP greedy argmax on the logits path.
    bool replicate_lm_head = false;
    /// FP4/MXFP4 runtime quant of routed experts is wired for this arch.
    bool mxfp4_runtime_quant = false;
    /// BF16 -> FP8/INT8 runtime quant is supported for this arch.
    bool bf16_runtime_quant = false;
    /// Skip the dense fused-QKV join. Qwen3-MoE and GPT-OSS bind attention
    /// q/k/v as separate tensors and never read a fused qkv, so the generic
    /// join would consume q/k/v_proj and leave the bind path with a missing
    /// weight.
    bool skip_dense_qkv_fusion = false;
    /// Stack per-expert MoE weights into the fused 3-D tensors the qwen3_5_moe
    /// forward expects. Plain Qwen3-MoE ships per-expert weights; qwen3_5_moe
    /// checkpoints ship them pre-fused, which makes this a no-op.
    bool stack_per_expert_moe = false;
    /// Only stack when the per-expert weights are raw BF16/F16. Quantised
    /// experts carry companion scale tensors that the stack would orphan, so
    /// archs that can ship either layout set this and fall back to the
    /// per-expert path for quantised checkpoints.
    bool stack_per_expert_moe_float_only = false;
    /// The canonical Metal Qwen3.5/GDN storage schema is defined for this arch.
    bool metal_qwen35 = false;
    /// Tensor-parallel shard-axis strategy, keyed by tensor name. Defaults to
    /// the llama-family rules; archs with a different expert or attention
    /// layout (DeepSeek-V4's native `.ffn.experts.w*`) register their own.
    ShardAxis (*shard_axis_fn)(std::string_view) = llama_like_shard_axis;
};

/// One row of the profile table.
///
/// `names` is a `vector`, not an `initializer_list`: an `initializer_list` kept
/// as a member outlives the backing array the braces created, which is a
/// dangling read the first time the table is consulted. It cost half a day here
/// — the phi3 and gpt-oss profiles resolved correctly at tp 1 and silently
/// stopped matching at tp 2.
struct ArchEntry {
    std::vector<std::string_view> names;
    ArchProfile profile;
};

inline bool iequals(std::string_view a, std::string_view b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        const char lhs = static_cast<char>(std::tolower(static_cast<unsigned char>(a[i])));
        const char rhs = static_cast<char>(std::tolower(static_cast<unsigned char>(b[i])));
        if (lhs != rhs) {
            return false;
        }
    }
    return true;
}

/// The generic-dense fallback covers llama-family models that need nothing
/// beyond the name-pattern detectors that always run.
inline ArchProfile arch_profile(std::string_view model_type) {
    static const std::vector<ArchEntry> kProfiles = {
        {{"kimi_k2", "kimi_k25"},
         ArchProfile{.source_prefix = "language_model.",
                     .mla_fused_joins = true,
                     .shard_embed_tokens = true,
                     .replicate_lm_head = true}},
        {{"deepseek_v2", "deepseek_v3"}, ArchProfile{.mla_fused_joins = true}},
        // Native `.ffn.experts.w*` naming plus per-expert intermediate sharding.
        {{"deepseek_v4"}, ArchProfile{.shard_axis_fn = dsv4_shard_axis}},
        {{"phi3"}, ArchProfile{.phi3_fused_splits = true}},
        {{"nemotron_h"}, ArchProfile{.nemotron_packed_experts = true}},
        {{"gpt_oss", "gpt-oss", "gptoss"},
         ArchProfile{.gpt_oss_mxfp4_groups = true, .skip_dense_qkv_fusion = true}},
        {{"glm_moe_dsa"},
         ArchProfile{.shard_embed_tokens = true,
                     .mxfp4_runtime_quant = true,
                     .bf16_runtime_quant = true,
                     .stack_per_expert_moe = true,
                     .stack_per_expert_moe_float_only = true}},
        {{"qwen3_moe", "qwen3_5_moe", "qwen3_5_moe_text"},
         ArchProfile{.bf16_runtime_quant = true,
                     .skip_dense_qkv_fusion = true,
                     .stack_per_expert_moe = true}},
        {{"qwen3_5", "qwen3_5_text", "qwen3_next", "qwen3_next_text", "qwen3_6"},
         ArchProfile{.bf16_runtime_quant = true, .metal_qwen35 = true}},
        {{"qwen3", "qwen2", "llama", "llama3", "mistral"},
         ArchProfile{.bf16_runtime_quant = true}},
    };
    for (const ArchEntry& entry : kProfiles) {
        for (std::string_view name : entry.names) {
            if (iequals(model_type, name)) {
                return entry.profile;
            }
        }
    }
    return ArchProfile{};
}

inline bool runtime_quant_model_supported(std::string_view model_type,
                                          PieLoaderQuantScheme scheme) {
    const ArchProfile profile = arch_profile(model_type);
    return scheme == PieLoaderQuantScheme::Mxfp4E2M1E8M0 ? profile.mxfp4_runtime_quant
                                                         : profile.bf16_runtime_quant;
}

// -- small arithmetic with a message ------------------------------------------

[[noreturn]] inline void fail(const std::string& what) { throw std::runtime_error(what); }

inline std::int64_t align_up(std::int64_t value, std::int64_t alignment) {
    if (value < 0 || alignment <= 0) {
        fail("contract: align_up needs a non-negative value and a positive alignment");
    }
    return (value + alignment - 1) / alignment * alignment;
}

inline std::uint32_t u32_dim(std::int64_t value, std::string_view context) {
    if (value < 0 || value > static_cast<std::int64_t>(UINT32_MAX)) {
        fail(std::string(context) + ": dimension " + std::to_string(value) + " does not fit u32");
    }
    return static_cast<std::uint32_t>(value);
}

/// The `[start, len)` this rank owns of a `full`-long axis.
///
/// `what` names the thing being split, because this is the message a user gets
/// for "tp_size does not divide this model".
inline std::pair<std::int64_t, std::int64_t> local_range(std::int64_t full,
                                                         const pie_loader::DeviceTarget& target,
                                                         const std::string& what) {
    const std::int64_t world = std::max<std::int64_t>(1, target.tp_size);
    if (full % world != 0) {
        fail(what + " is " + std::to_string(full) + ", which tp_size " +
             std::to_string(target.tp_size) +
             " does not divide; use a tp_size that divides it or run single-GPU");
    }
    const std::int64_t local = full / world;
    return {static_cast<std::int64_t>(target.tp_rank) * local, local};
}

inline PieLoaderEncodingSpec mxfp4_encoding(ModelContract& contract, std::uint8_t channel_axis) {
    PieLoaderQuantSpecView quant =
        pie_loader::quant_spec(PieLoaderQuantScheme::Mxfp4E2M1E8M0, PieLoaderDType::BF16);
    quant.bits_per_element = 4;
    quant.group_size = 32;
    quant.channel_axis = channel_axis;
    quant.scale_dtype = static_cast<std::uint32_t>(PieLoaderDType::U8);
    // One MXFP4 block scale covers 32 contiguous elements along K. The block
    // shape says so; the contract owns the number because the POD only borrows.
    return pie_loader::quantized(contract.with_block_shape(quant, {32}));
}

inline PieLoaderRepackSpecView repack_spec(PieLoaderRepackLayout layout, PieLoaderRowMap row_map) {
    PieLoaderRepackSpecView spec{};
    spec.layout = static_cast<std::uint32_t>(layout);
    spec.row_map = static_cast<std::uint32_t>(row_map);
    return spec;
}

inline std::vector<std::int64_t> shape_of(const SourceTensor& raw) {
    return std::vector<std::int64_t>(raw.shape.begin(), raw.shape.end());
}

/// Which tensors an encode-scoped rank materializes.
///
/// The multimodal towers and nothing else, so an encoder rank never allocates
/// the language model. Under a driver-authored contract this is not a filter
/// applied to a finished contract — it is simply which tensors get declared.
inline bool is_tower_output(std::string_view name) {
    return name.rfind("model.vision_tower.", 0) == 0 ||
           name.rfind("model.embed_vision.", 0) == 0 ||
           name.rfind("model.audio_tower.", 0) == 0 || name.rfind("model.embed_audio.", 0) == 0;
}

/// The Metal Qwen3.5 output name for a checkpoint tensor, or none to skip it.
inline std::optional<std::string> metal_qwen35_runtime_name(std::string_view raw_name) {
    if (raw_name.rfind("model.visual.", 0) == 0 || raw_name.rfind("mtp.", 0) == 0) {
        return std::nullopt;
    }
    if (raw_name.rfind("lm_head.", 0) == 0) {
        return "shared_embedding." + std::string(raw_name.substr(std::string_view("lm_head.").size()));
    }
    if (raw_name == "model.language_model.norm.weight") {
        return std::string("final_norm.weight");
    }
    constexpr std::string_view kLayers = "model.language_model.layers.";
    if (raw_name.rfind(kLayers, 0) != 0) {
        fail("Metal Qwen3.5 schema has no declared mapping or skip for '" + std::string(raw_name) +
             "'");
    }
    const std::string_view rest = raw_name.substr(kLayers.size());
    const std::size_t dot = rest.find('.');
    if (dot == std::string_view::npos) {
        fail("Metal Qwen3.5 layer tensor '" + std::string(raw_name) + "' is malformed");
    }
    const std::string_view layer = rest.substr(0, dot);
    if (layer.empty() || !std::all_of(layer.begin(), layer.end(), [](char c) {
            return std::isdigit(static_cast<unsigned char>(c)) != 0;
        })) {
        fail("Metal Qwen3.5 layer tensor '" + std::string(raw_name) +
             "' has an invalid layer index");
    }
    return "layers." + std::string(layer) + "." + std::string(rest.substr(dot + 1));
}

// -- the author ---------------------------------------------------------------

/// Walks the checkpoint once, applying each family's rules, and declares what
/// this driver will bind.
///
/// The shape is the same as the loader's `arch/` was: special-case passes run
/// first and mark the tensors they consumed, then everything left over is
/// declared directly. That order is not incidental — a pass that fuses q/k/v
/// into one buffer has to claim them before the generic loop publishes them
/// separately.
class ContractAuthor {
public:
    ContractAuthor(const Checkpoint& checkpoint, const ModelFacts& facts,
                   const pie_loader::DeviceTarget& target, std::string_view runtime_quant,
                   Mxfp4MoePolicy mxfp4_moe, Component component)
        : checkpoint_(checkpoint),
          facts_(facts),
          target_(target),
          // A runtime quantization request the device cannot execute is dropped
          // rather than refused: it is a preference, and the checkpoint's own
          // format is always a valid answer. The driver is where this belongs —
          // `fp8_native` is something it measured.
          runtime_quant_(runtime_quant == "fp8" && !target.fp8_native ? std::string_view()
                                                                     : runtime_quant),
          mxfp4_moe_(mxfp4_moe),
          component_(component),
          profile_(arch_profile(facts.model_type)),
          tensors_(checkpoint.tensors()) {
        by_name_.reserve(tensors_.size());
        for (const SourceTensor& raw : tensors_) {
            by_name_.emplace(raw.name, &raw);
        }
    }

    void build(ModelContract& out) {
        contract_ = &out;
        out.align(std::max<std::uint32_t>(1, target_.preferred_alignment));

        if (target_.backend == PieLoaderBackendKind::Metal) {
            if (!profile_.metal_qwen35) {
                fail("Metal storage schema does not support model_type='" + facts_.model_type +
                     "'");
            }
            add_metal_qwen35_contracts();
            return;
        }

        const std::optional<PieLoaderQuantScheme> runtime_quant = runtime_quant_scheme();
        add_phi3_fused_splits();
        add_gpt_oss_mxfp4_groups();
        add_fused_moe_gate_up_tp_slices();
        add_qwen_moe_expert_stacks();
        add_nemotron_h_packed_expert_views();
        add_dense_fused_projection_joins(runtime_quant.has_value());
        add_mla_fused_projection_joins();

        for (const SourceTensor& raw : tensors_) {
            if (consumed_.count(raw.id) != 0 || !source_name_allowed(raw.name)) {
                continue;
            }
            if (runtime_quant.has_value() && runtime_quantizable_name(raw.name, *runtime_quant)) {
                push_runtime_quant(raw, std::string(raw.name), *runtime_quant);
            } else {
                push_direct(raw, output_name(raw.name), shard_axis(raw.name));
            }
        }
    }

private:
    // -- publishing ----------------------------------------------------------

    /// Declare `expr` under `name`, unless this rank is not supposed to hold it.
    ///
    /// The component filter is here rather than as a post-pass over a finished
    /// contract because "declare only what you want" is the whole mechanism a
    /// contract offers. `retain_outputs` existed because the loader authored
    /// first and narrowed afterwards.
    void define(std::string name, Node expr, PieLoaderEncodingSpec encoding,
                std::optional<std::vector<std::int64_t>> shape) {
        if (component_ == Component::Encode && !is_tower_output(name)) {
            return;
        }
        auto defined = contract_->define(std::move(name), expr, encoding);
        if (shape.has_value()) {
            defined.expect(std::move(*shape));
        }
    }

    void push_direct(const SourceTensor& raw, std::string output, ShardAxis axis) {
        auto [expr, shape] = shard(contract_->src(std::string(raw.name)), shape_of(raw), axis);
        define(std::move(output), expr, raw.encoding, std::move(shape));
    }

    /// Partition an expression and its shape across ranks along `axis`.
    ///
    /// The expression records *that* the tensor is split; the shape records what
    /// this rank ends up holding. Both are left alone when there is nothing to
    /// split, so a single-GPU build is identical to one authored without
    /// sharding, and a non-divisible extent is left for the loader to reject
    /// with a message that names the tensor.
    std::pair<Node, std::vector<std::int64_t>> shard(Node expr, std::vector<std::int64_t> shape,
                                                     ShardAxis axis) {
        const std::int64_t world = target_.tp_size;
        if (!axis.has_value() || world <= 1) {
            return {expr, std::move(shape)};
        }
        const std::size_t index = *axis;
        if (index < shape.size() && shape[index] % world == 0) {
            shape[index] /= world;
        }
        return {contract_->shard(expr, *axis), std::move(shape)};
    }

    ShardAxis shard_axis(std::string_view name) const {
        if (target_.tp_size <= 1) {
            return std::nullopt;
        }
        // Shard embed_tokens on axis 0 to save per-rank memory (Kimi, GLM-5.1).
        if (profile_.shard_embed_tokens && ends_with(name, ".embed_tokens.weight")) {
            return std::uint8_t{0};
        }
        if (profile_.replicate_lm_head && ends_with(name, ".lm_head.weight")) {
            return std::nullopt;
        }
        return profile_.shard_axis_fn(name);
    }

    bool source_name_allowed(std::string_view raw_name) const {
        return profile_.source_prefix.empty() ||
               raw_name.rfind(profile_.source_prefix, 0) == 0;
    }

    std::string output_name(std::string_view raw_name) const {
        if (!profile_.source_prefix.empty() && raw_name.rfind(profile_.source_prefix, 0) == 0) {
            return std::string(raw_name.substr(profile_.source_prefix.size()));
        }
        return std::string(raw_name);
    }

    void push_expr(std::string output, const SourceTensor& raw, std::vector<std::int64_t> shape,
                   Node expr) {
        define(std::move(output), expr, pie_loader::raw(logical_dtype(raw.encoding)),
               std::move(shape));
        consumed_.insert(raw.id);
    }

    void push_repack(std::string output, const SourceTensor& raw, PieLoaderEncodingSpec encoding,
                     std::vector<std::int64_t> shape, PieLoaderRepackSpecView spec) {
        const Node node =
            contract_->repack(contract_->src(std::string(raw.name)), spec, shape, encoding);
        define(std::move(output), node, encoding, std::move(shape));
    }

    const SourceTensor* find(const std::string& name) const {
        const auto it = by_name_.find(name);
        return it == by_name_.end() ? nullptr : it->second;
    }

    // -- runtime quantization ------------------------------------------------

    std::optional<PieLoaderQuantScheme> runtime_quant_scheme() const {
        if (runtime_quant_.empty()) {
            return std::nullopt;
        }
        PieLoaderQuantScheme scheme{};
        if (runtime_quant_ == "fp8") {
            scheme = PieLoaderQuantScheme::Fp8E4M3;
        } else if (runtime_quant_ == "int8") {
            scheme = PieLoaderQuantScheme::Int8Symmetric;
        } else if (runtime_quant_ == "fp4" || runtime_quant_ == "mxfp4") {
            scheme = PieLoaderQuantScheme::Mxfp4E2M1E8M0;
        } else {
            fail("unsupported runtime_quant '" + std::string(runtime_quant_) +
                 "'; expected 'fp8', 'int8', or 'fp4'");
        }
        // For FP4 a pre-quantized checkpoint is accepted (GLM-5.1 ships FP8
        // experts). For FP8/INT8 only BF16 weights are re-quantized, never an
        // already-quantized checkpoint.
        if (!facts_.quant_method.empty() && scheme != PieLoaderQuantScheme::Mxfp4E2M1E8M0) {
            return std::nullopt;
        }
        if (!runtime_quant_model_supported(facts_.model_type, scheme)) {
            fail("runtime_quant=" + std::string(runtime_quant_) +
                 " is not supported for model_type='" + facts_.model_type + "'");
        }
        return scheme;
    }

    void push_runtime_quant(const SourceTensor& raw, std::string output,
                            PieLoaderQuantScheme scheme) {
        if (raw.shape.size() != 2) {
            fail("runtime_quant source '" + std::string(raw.name) + "' must be 2-D");
        }
        // Allowed sources are BF16/FP16/FP32 raw, handled by the executor's
        // bf16 cast path, and FP8 (E4M3) raw, used by GLM-5.1 routed experts:
        // those weights ship quantized, the executor dequants them to bf16 with
        // a sibling `_scale_inv` at materialize time, then re-encodes to the
        // target scheme. Only meaningful when the target is a smaller scheme.
        if (!(is_raw(raw.encoding, PieLoaderDType::BF16) ||
              is_raw(raw.encoding, PieLoaderDType::F16) ||
              is_raw(raw.encoding, PieLoaderDType::F32) ||
              is_raw(raw.encoding, PieLoaderDType::F8E4M3))) {
            fail("runtime_quant source '" + std::string(raw.name) +
                 "' must be BF16/FP16/FP32/F8E4M3");
        }

        PieLoaderQuantSpecView quant{};
        switch (scheme) {
            case PieLoaderQuantScheme::Fp8E4M3:
                quant = pie_loader::quant_spec(scheme, PieLoaderDType::F8E4M3);
                quant.bits_per_element = 8;
                quant.group_size = 1;
                quant.channel_axis = 0;
                quant.scale_dtype = static_cast<std::uint32_t>(PieLoaderDType::F32);
                break;
            case PieLoaderQuantScheme::Int8Symmetric:
                quant = pie_loader::quant_spec(scheme, PieLoaderDType::I8);
                quant.bits_per_element = 8;
                quant.group_size = 1;
                quant.channel_axis = 0;
                quant.scale_dtype = static_cast<std::uint32_t>(PieLoaderDType::F32);
                break;
            case PieLoaderQuantScheme::Mxfp4E2M1E8M0:
                // The K dimension (columns, for a 2-D weight) must be a
                // 32-multiple because an MXFP4 block scale covers 32
                // contiguous elements.
                if (raw.shape[1] % 32 != 0) {
                    fail("runtime_quant Mxfp4 source '" + std::string(raw.name) + "' cols " +
                         std::to_string(raw.shape[1]) + " must be a multiple of 32");
                }
                quant = pie_loader::quant_spec(scheme, PieLoaderDType::BF16);
                quant.bits_per_element = 4;
                quant.group_size = 32;
                quant.channel_axis = 1;
                quant.scale_dtype = static_cast<std::uint32_t>(PieLoaderDType::U8);
                quant = contract_->with_block_shape(quant, {32});
                break;
            default:
                fail("unsupported runtime_quant scheme");
        }

        auto [expr, shape] =
            shard(contract_->src(std::string(raw.name)), shape_of(raw), shard_axis(raw.name));
        define(std::move(output), expr, pie_loader::quantized(quant), std::move(shape));
    }

    // -- Phi-3 fused splits ---------------------------------------------------

    void add_phi3_fused_splits() {
        if (!profile_.phi3_fused_splits) {
            return;
        }
        for (const SourceTensor& raw : tensors_) {
            if (ends_with(raw.name, ".self_attn.qkv_proj.weight")) {
                add_phi3_qkv_split(raw);
            } else if (ends_with(raw.name, ".mlp.gate_up_proj.weight")) {
                add_phi3_gate_up_split(raw);
            }
        }
    }

    void add_phi3_qkv_split(const SourceTensor& raw) {
        if (raw.shape.size() != 2) {
            fail("Phi-3 fused QKV '" + std::string(raw.name) + "' must be 2-D");
        }
        const std::int64_t q_rows = raw.shape[1];
        const std::int64_t kv_rows = (raw.shape[0] - q_rows) / 2;
        if (q_rows <= 0 || kv_rows <= 0 || q_rows + 2 * kv_rows != raw.shape[0]) {
            fail("Phi-3 fused QKV '" + std::string(raw.name) + "' has an unsupported shape");
        }
        const std::int64_t cols = raw.shape[1];
        const std::string base =
            std::string(raw.name.substr(0, raw.name.size() -
                                               std::string_view(".self_attn.qkv_proj.weight").size()));
        const struct {
            std::string_view proj;
            std::int64_t start;
            std::int64_t rows;
        } specs[] = {{"q_proj", 0, q_rows},
                     {"k_proj", q_rows, kv_rows},
                     {"v_proj", q_rows + kv_rows, kv_rows}};
        for (const auto& spec : specs) {
            const auto [local_start, local_rows] =
                local_range(spec.rows, target_, "the row count of '" + std::string(raw.name) + "'");
            push_expr(base + ".self_attn." + std::string(spec.proj) + ".weight", raw,
                      {local_rows, cols},
                      contract_->slice(contract_->src(std::string(raw.name)), 0,
                                       spec.start + local_start, local_rows));
        }
    }

    void add_phi3_gate_up_split(const SourceTensor& raw) {
        if (raw.shape.size() != 2 || raw.shape[0] % 2 != 0) {
            fail("Phi-3 fused gate/up '" + std::string(raw.name) + "' has an unsupported shape");
        }
        const std::int64_t half_rows = raw.shape[0] / 2;
        const std::int64_t cols = raw.shape[1];
        const std::string base = std::string(
            raw.name.substr(0, raw.name.size() -
                                   std::string_view(".mlp.gate_up_proj.weight").size()));
        const struct {
            std::string_view proj;
            std::int64_t start;
        } specs[] = {{"gate_proj", 0}, {"up_proj", half_rows}};
        for (const auto& spec : specs) {
            const auto [local_start, local_rows] = local_range(
                half_rows, target_, "half the row count of '" + std::string(raw.name) + "'");
            push_expr(base + ".mlp." + std::string(spec.proj) + ".weight", raw,
                      {local_rows, cols},
                      contract_->slice(contract_->src(std::string(raw.name)), 0,
                                       spec.start + local_start, local_rows));
        }
    }

    // -- GPT-OSS MXFP4 expert groups -----------------------------------------

    void add_gpt_oss_mxfp4_groups() {
        if (!profile_.gpt_oss_mxfp4_groups) {
            return;
        }
        const bool native = mxfp4_moe_ == Mxfp4MoePolicy::NativeGemm;
        if (native && !target_.native_mxfp4_moe) {
            fail("GPT-OSS native MXFP4 requested, but target does not support native MXFP4 MoE");
        }
        for (const SourceTensor& raw : tensors_) {
            if (!ends_with(raw.name, "_blocks")) {
                continue;
            }
            const SourceTensor& block = raw;
            const std::string base =
                std::string(block.name.substr(0, block.name.size() - std::string_view("_blocks").size()));
            const SourceTensor* scale = find(base + "_scales");
            const SourceTensor* bias = find(base + "_bias");
            if (scale == nullptr || bias == nullptr) {
                continue;
            }
            if (native) {
                add_gpt_oss_native_group(block, *scale, *bias, base);
            } else {
                push_direct(block, base + ".weight", std::nullopt);
                push_direct(*scale, base + ".weight_scale", std::nullopt);
                push_direct(*bias, base + ".bias", std::nullopt);
            }
            consumed_.insert(block.id);
            consumed_.insert(scale->id);
            consumed_.insert(bias->id);
        }
    }

    void add_gpt_oss_native_group(const SourceTensor& block, const SourceTensor& scale,
                                  const SourceTensor& bias, const std::string& base) {
        if (ends_with(base, "gate_up_proj")) {
            add_gpt_oss_native_gate_up(block, scale, bias, base);
        } else if (ends_with(base, "down_proj")) {
            add_gpt_oss_native_down(block, scale, bias, base);
        } else {
            fail("GPT-OSS MXFP4 tensor '" + std::string(block.name) +
                 "' is not gate_up_proj or down_proj");
        }
    }

    void add_gpt_oss_native_gate_up(const SourceTensor& block, const SourceTensor& scale,
                                    const SourceTensor& bias, const std::string& base) {
        if (block.shape.size() != 4 || scale.shape.size() != 3 || bias.shape.size() != 2) {
            fail("GPT-OSS native gate/up '" + base + "' has an unsupported block/scale/bias rank");
        }
        const std::int64_t experts = block.shape[0];
        const std::int64_t fused_rows = block.shape[1];
        const std::int64_t groups = block.shape[2];
        if (fused_rows % 2 != 0 || block.shape[3] != 16) {
            fail("GPT-OSS native gate/up '" + base + "' expected [E, 2I, H/32, 16]");
        }
        if (scale.shape.size() != 3 || scale.shape[0] != experts || scale.shape[1] != fused_rows ||
            scale.shape[2] != groups || bias.shape[0] != experts || bias.shape[1] != fused_rows) {
            fail("GPT-OSS native gate/up '" + base + "' scale/bias shape mismatch");
        }
        const std::int64_t full_intermediate = fused_rows / 2;
        const std::int64_t hidden = groups * 32;
        const auto [local_start, local_intermediate] =
            local_range(full_intermediate, target_, "the intermediate size of '" + base + "'");
        const std::int64_t intermediate_native = align_up(local_intermediate, 128);
        const std::string prefix =
            base.substr(0, base.size() - std::string_view("gate_up_proj").size());

        const struct {
            std::string_view name;
            PieLoaderRowMap row_map;
        } halves[] = {{"gate_proj", PieLoaderRowMap::Even}, {"up_proj", PieLoaderRowMap::Odd}};
        for (const auto& half : halves) {
            const std::string out_base = prefix + std::string(half.name);

            PieLoaderRepackSpecView weight =
                repack_spec(PieLoaderRepackLayout::MarlinMxfp4Weight, half.row_map);
            weight.batch = u32_dim(experts, "GPT-OSS experts");
            weight.source_rows = u32_dim(fused_rows, "GPT-OSS gate/up source rows");
            weight.source_row_offset = u32_dim(local_start, "GPT-OSS gate/up source row offset");
            weight.target_rows = u32_dim(intermediate_native, "GPT-OSS gate/up target rows");
            weight.valid_rows = u32_dim(local_intermediate, "GPT-OSS gate/up valid rows");
            weight.source_stride_cols = u32_dim(hidden, "GPT-OSS hidden stride");
            weight.source_col_offset = 0;
            weight.source_cols = u32_dim(hidden, "GPT-OSS hidden size");
            weight.target_cols = u32_dim(hidden, "GPT-OSS hidden size");
            push_repack(out_base + ".weight", block, mxfp4_encoding(*contract_, 1),
                        {experts, intermediate_native, hidden}, weight);

            PieLoaderRepackSpecView scale_spec =
                repack_spec(PieLoaderRepackLayout::MarlinMxfp4Scale, half.row_map);
            scale_spec.batch = u32_dim(experts, "GPT-OSS experts");
            scale_spec.source_rows = u32_dim(fused_rows, "GPT-OSS gate/up source rows");
            scale_spec.source_row_offset =
                u32_dim(local_start, "GPT-OSS gate/up source row offset");
            scale_spec.target_rows = u32_dim(intermediate_native, "GPT-OSS gate/up target rows");
            scale_spec.valid_rows = u32_dim(local_intermediate, "GPT-OSS gate/up valid rows");
            scale_spec.source_stride_cols = u32_dim(groups, "GPT-OSS hidden group stride");
            scale_spec.source_col_offset = 0;
            scale_spec.source_cols = u32_dim(groups, "GPT-OSS hidden groups");
            scale_spec.target_cols = u32_dim(groups, "GPT-OSS hidden groups");
            push_repack(out_base + ".weight_scale", scale,
                        pie_loader::raw(PieLoaderDType::U8),
                        {experts, intermediate_native, groups}, scale_spec);

            PieLoaderRepackSpecView bias_spec =
                repack_spec(PieLoaderRepackLayout::DenseRowGather, half.row_map);
            bias_spec.batch = u32_dim(experts, "GPT-OSS experts");
            bias_spec.source_rows = u32_dim(fused_rows, "GPT-OSS gate/up bias rows");
            bias_spec.source_row_offset =
                u32_dim(local_start, "GPT-OSS gate/up bias source row offset");
            bias_spec.target_rows = u32_dim(local_intermediate, "GPT-OSS gate/up bias target rows");
            bias_spec.valid_rows = u32_dim(local_intermediate, "GPT-OSS gate/up bias valid rows");
            bias_spec.source_stride_cols = 1;
            bias_spec.source_col_offset = 0;
            bias_spec.source_cols = 1;
            bias_spec.target_cols = 1;
            push_repack(out_base + ".bias", bias, pie_loader::raw(PieLoaderDType::BF16),
                        {experts, local_intermediate}, bias_spec);
        }
    }

    void add_gpt_oss_native_down(const SourceTensor& block, const SourceTensor& scale,
                                 const SourceTensor& bias, const std::string& base) {
        if (block.shape.size() != 4 || scale.shape.size() != 3 || bias.shape.size() != 2) {
            fail("GPT-OSS native down '" + base + "' has an unsupported block/scale/bias rank");
        }
        const std::int64_t experts = block.shape[0];
        const std::int64_t hidden = block.shape[1];
        const std::int64_t groups = block.shape[2];
        if (block.shape[3] != 16) {
            fail("GPT-OSS native down '" + base + "' expected [E, H, I/32, 16]");
        }
        if (scale.shape[0] != experts || scale.shape[1] != hidden || scale.shape[2] != groups ||
            bias.shape[0] != experts || bias.shape[1] != hidden) {
            fail("GPT-OSS native down '" + base + "' scale/bias shape mismatch");
        }
        const std::int64_t full_intermediate = groups * 32;
        const auto [local_start, local_intermediate] =
            local_range(full_intermediate, target_, "the intermediate size of '" + base + "'");
        if (local_start % 32 != 0 || local_intermediate % 32 != 0) {
            fail("GPT-OSS native down '" + base +
                 "' TP shard must align to MXFP4 32-wide groups");
        }
        const std::int64_t local_groups = local_intermediate / 32;
        const std::int64_t source_group_offset = local_start / 32;
        const std::int64_t intermediate_native = align_up(local_intermediate, 128);

        PieLoaderRepackSpecView weight =
            repack_spec(PieLoaderRepackLayout::MarlinMxfp4Weight, PieLoaderRowMap::Identity);
        weight.batch = u32_dim(experts, "GPT-OSS experts");
        weight.source_rows = u32_dim(hidden, "GPT-OSS down source rows");
        weight.source_row_offset = 0;
        weight.target_rows = u32_dim(hidden, "GPT-OSS down target rows");
        weight.valid_rows = u32_dim(hidden, "GPT-OSS down valid rows");
        weight.source_stride_cols = u32_dim(full_intermediate, "GPT-OSS down source stride");
        weight.source_col_offset = u32_dim(local_start, "GPT-OSS down source column offset");
        weight.source_cols = u32_dim(local_intermediate, "GPT-OSS intermediate size");
        weight.target_cols = u32_dim(intermediate_native, "GPT-OSS padded intermediate size");
        push_repack(base + ".weight", block, mxfp4_encoding(*contract_, 2),
                    {experts, hidden, intermediate_native}, weight);

        PieLoaderRepackSpecView scale_spec =
            repack_spec(PieLoaderRepackLayout::MarlinMxfp4Scale, PieLoaderRowMap::Identity);
        scale_spec.batch = u32_dim(experts, "GPT-OSS experts");
        scale_spec.source_rows = u32_dim(hidden, "GPT-OSS down source rows");
        scale_spec.source_row_offset = 0;
        scale_spec.target_rows = u32_dim(hidden, "GPT-OSS down target rows");
        scale_spec.valid_rows = u32_dim(hidden, "GPT-OSS down valid rows");
        scale_spec.source_stride_cols = u32_dim(groups, "GPT-OSS down source group stride");
        scale_spec.source_col_offset = u32_dim(source_group_offset, "GPT-OSS down source group offset");
        scale_spec.source_cols = u32_dim(local_groups, "GPT-OSS down source groups");
        scale_spec.target_cols = u32_dim(intermediate_native / 32, "GPT-OSS down target groups");
        push_repack(base + ".weight_scale", scale, pie_loader::raw(PieLoaderDType::U8),
                    {experts, hidden, intermediate_native / 32}, scale_spec);

        push_direct(bias, base + ".bias", std::nullopt);
    }

    // -- Qwen MoE -------------------------------------------------------------

    void add_fused_moe_gate_up_tp_slices() {
        if (target_.tp_size <= 1) {
            return;
        }
        for (const SourceTensor& raw : tensors_) {
            if (!ends_with(raw.name, ".experts.gate_up_proj") &&
                !ends_with(raw.name, ".mlp.experts.gate_up_proj")) {
                continue;
            }
            if (raw.shape.size() != 3 || raw.shape[1] % 2 != 0) {
                continue;
            }
            if (!is_dense_addressable(raw.encoding)) {
                fail("fused MoE gate/up '" + std::string(raw.name) +
                     "' has a non-affine packed encoding");
            }
            const std::int64_t experts = raw.shape[0];
            const std::int64_t full_i = raw.shape[1] / 2;
            const std::int64_t hidden = raw.shape[2];
            const auto [local_start, local_i] = local_range(
                full_i, target_, "the intermediate size of '" + std::string(raw.name) + "'");
            // Gate rows [0, I) and up rows [I, 2I) are sharded independently
            // and re-joined, so the local halves stay adjacent per expert.
            const Node src = contract_->src(std::string(raw.name));
            push_expr(std::string(raw.name), raw, {experts, 2 * local_i, hidden},
                      contract_->cat({contract_->slice(src, 1, local_start, local_i),
                                      contract_->slice(src, 1, full_i + local_start, local_i)},
                                     1));
        }
    }

    /// Stack per-expert MoE weights into the fused 3-D tensors the qwen3_5_moe
    /// forward consumes. The HF Qwen3-MoE source layout is, per expert `e`,
    /// `mlp.experts.{e}.gate_proj.weight` / `.up_proj.weight` as `[I, H]` and
    /// `.down_proj.weight` as `[H, I]`. Output:
    ///   `mlp.experts.gate_up_proj` -> `[E, 2I, H]`, rows `[0,I)` gate, `[I,2I)` up
    ///   `mlp.experts.down_proj`    -> `[E, H, I]`
    /// Built as an expression over the sources, so no GPU-side duplicate exists.
    /// A no-op when the checkpoint already ships the fused tensors.
    void add_qwen_moe_expert_stacks() {
        if (!profile_.stack_per_expert_moe || target_.tp_size != 1) {
            // TP>1 per-expert sharding is not wired; the bind then fails loudly
            // on the missing fused tensor rather than loading silently wrong.
            return;
        }
        const std::int64_t num_experts = facts_.num_experts;
        if (num_experts <= 0) {
            return;
        }
        for (std::uint32_t layer = 0; layer < facts_.num_hidden_layers; ++layer) {
            const std::string prefix =
                "model.layers." + std::to_string(layer) + ".mlp.experts.";
            if (find(prefix + "gate_up_proj") != nullptr) {
                continue;  // already pre-fused; the direct and TP-slice paths take it.
            }
            const SourceTensor* gate0 = find(prefix + "0.gate_proj.weight");
            if (gate0 == nullptr) {
                continue;  // not a per-expert checkpoint at this layer.
            }
            if (gate0->shape.size() != 2) {
                fail("qwen moe expert stack: '" + std::string(gate0->name) + "' expected 2-D");
            }
            const std::int64_t inter = gate0->shape[0];
            const std::int64_t hidden = gate0->shape[1];
            const PieLoaderDType dtype = logical_dtype(gate0->encoding);
            if (profile_.stack_per_expert_moe_float_only &&
                dtype != PieLoaderDType::BF16 && dtype != PieLoaderDType::F16) {
                continue;  // quantised experts keep the per-expert layout.
            }
            if (!is_dense_addressable(gate0->encoding)) {
                fail("qwen moe expert stack: '" + std::string(gate0->name) +
                     "' has a non-affine packed encoding");
            }

            std::vector<Node> gate_up_parts;
            std::vector<Node> down_parts;
            std::vector<std::uint32_t> consumed;
            gate_up_parts.reserve(static_cast<std::size_t>(num_experts));
            down_parts.reserve(static_cast<std::size_t>(num_experts));
            for (std::int64_t e = 0; e < num_experts; ++e) {
                const std::string tag = prefix + std::to_string(e) + ".";
                const SourceTensor* g = find(tag + "gate_proj.weight");
                const SourceTensor* u = find(tag + "up_proj.weight");
                const SourceTensor* d = find(tag + "down_proj.weight");
                if (g == nullptr || u == nullptr || d == nullptr) {
                    fail("qwen moe expert stack: layer " + std::to_string(layer) + " expert " +
                         std::to_string(e) + " missing gate/up/down");
                }
                const bool shapes_ok = g->shape.size() == 2 && u->shape.size() == 2 &&
                                       d->shape.size() == 2 && g->shape[0] == inter &&
                                       g->shape[1] == hidden && u->shape[0] == inter &&
                                       u->shape[1] == hidden && d->shape[0] == hidden &&
                                       d->shape[1] == inter;
                if (!shapes_ok) {
                    fail("qwen moe expert stack: layer " + std::to_string(layer) + " expert " +
                         std::to_string(e) + " shape mismatch");
                }
                if (logical_dtype(g->encoding) != dtype || logical_dtype(u->encoding) != dtype ||
                    logical_dtype(d->encoding) != dtype) {
                    fail("qwen moe expert stack: layer " + std::to_string(layer) + " expert " +
                         std::to_string(e) + " dtype mismatch");
                }
                // One expert slab is gate over up; the leading 1 makes the
                // per-expert concatenation a stack.
                gate_up_parts.push_back(contract_->reshape(
                    contract_->cat({contract_->src(std::string(g->name)),
                                    contract_->src(std::string(u->name))},
                                   0),
                    {1, 2 * inter, hidden}));
                down_parts.push_back(contract_->reshape(
                    contract_->src(std::string(d->name)), {1, hidden, inter}));
                consumed.push_back(g->id);
                consumed.push_back(u->id);
                consumed.push_back(d->id);
            }

            define(prefix + "gate_up_proj", contract_->cat(gate_up_parts, 0),
                   pie_loader::raw(dtype),
                   std::vector<std::int64_t>{num_experts, 2 * inter, hidden});
            define(prefix + "down_proj", contract_->cat(down_parts, 0), pie_loader::raw(dtype),
                   std::vector<std::int64_t>{num_experts, hidden, inter});
            for (std::uint32_t id : consumed) {
                consumed_.insert(id);
            }
        }
    }

    // -- Nemotron-H packed experts -------------------------------------------

    void add_nemotron_h_packed_expert_views() {
        if (target_.backend != PieLoaderBackendKind::Cuda || !profile_.nemotron_packed_experts ||
            facts_.num_experts == 0) {
            return;
        }
        for (std::uint32_t layer = 0; layer < facts_.num_hidden_layers; ++layer) {
            const std::string base =
                "language_model.backbone.layers." + std::to_string(layer) + ".mixer.experts";
            if (find(base + ".up_proj.packed.weight") != nullptr ||
                find(base + ".down_proj.packed.weight") != nullptr) {
                continue;
            }
            std::vector<const SourceTensor*> up;
            std::vector<const SourceTensor*> down;
            bool complete = true;
            for (std::uint32_t expert = 0; expert < facts_.num_experts; ++expert) {
                const std::string tag = base + "." + std::to_string(expert) + ".";
                const SourceTensor* u = find(tag + "up_proj.weight");
                const SourceTensor* d = find(tag + "down_proj.weight");
                if (u == nullptr || d == nullptr) {
                    complete = false;
                    break;
                }
                up.push_back(u);
                down.push_back(d);
            }
            if (complete) {
                add_nemotron_h_layer_packed_experts(base, up, down);
            }
        }
    }

    void add_nemotron_h_layer_packed_experts(const std::string& base,
                                             const std::vector<const SourceTensor*>& up,
                                             const std::vector<const SourceTensor*>& down) {
        if (up.empty() || down.empty()) {
            return;
        }
        const SourceTensor& first_up = *up.front();
        const SourceTensor& first_down = *down.front();
        if (first_up.shape.size() != 2 || first_down.shape.size() != 2 ||
            !is_raw(first_up.encoding, PieLoaderDType::BF16) ||
            !is_raw(first_down.encoding, PieLoaderDType::BF16)) {
            return;
        }
        const std::int64_t full_intermediate = first_up.shape[0];
        const std::int64_t hidden = first_up.shape[1];
        if (first_down.shape[0] != hidden || first_down.shape[1] != full_intermediate) {
            return;
        }
        for (const SourceTensor* raw : up) {
            if (raw->shape.size() != 2 || raw->shape[0] != full_intermediate ||
                raw->shape[1] != hidden || !same_encoding(raw->encoding, first_up.encoding)) {
                return;
            }
        }
        for (const SourceTensor* raw : down) {
            if (raw->shape.size() != 2 || raw->shape[0] != hidden ||
                raw->shape[1] != full_intermediate ||
                !same_encoding(raw->encoding, first_down.encoding)) {
                return;
            }
        }

        const auto [local_start, local_intermediate] =
            local_range(full_intermediate, target_, "the intermediate size of '" + base + "'");
        const std::int64_t expert_count = static_cast<std::int64_t>(up.size());

        // Each expert contributes its local row band; the pack is their
        // concatenation. The sharding is in the expression, not in a flag.
        const std::string up_name = base + ".up_proj.packed.weight";
        std::vector<Node> up_parts;
        up_parts.reserve(up.size());
        for (const SourceTensor* raw : up) {
            up_parts.push_back(contract_->slice(contract_->src(std::string(raw->name)), 0,
                                                local_start, local_intermediate));
        }
        define(up_name, contract_->cat(up_parts, 0), pie_loader::raw(PieLoaderDType::BF16),
               std::vector<std::int64_t>{expert_count * local_intermediate, hidden});

        std::vector<Node> down_parts;
        down_parts.reserve(down.size());
        for (const SourceTensor* raw : down) {
            down_parts.push_back(contract_->src(std::string(raw->name)));
        }
        auto [down_expr, down_shape] =
            shard(contract_->cat(down_parts, 0),
                  {expert_count * hidden, full_intermediate}, std::uint8_t{1});
        const std::string down_name = base + ".down_proj.packed.weight";
        define(down_name, down_expr, pie_loader::raw(PieLoaderDType::BF16), std::move(down_shape));

        for (std::size_t expert = 0; expert < up.size(); ++expert) {
            const std::int64_t index = static_cast<std::int64_t>(expert);
            define(std::string(up[expert]->name),
                   contract_->slice(contract_->out(up_name), 0, index * local_intermediate,
                                    local_intermediate),
                   pie_loader::raw(PieLoaderDType::BF16),
                   std::vector<std::int64_t>{local_intermediate, hidden});
            consumed_.insert(up[expert]->id);
        }
        for (std::size_t expert = 0; expert < down.size(); ++expert) {
            const std::int64_t index = static_cast<std::int64_t>(expert);
            define(std::string(down[expert]->name),
                   contract_->slice(contract_->out(down_name), 0, index * hidden, hidden),
                   pie_loader::raw(PieLoaderDType::BF16),
                   std::vector<std::int64_t>{hidden, local_intermediate});
            consumed_.insert(down[expert]->id);
        }
    }

    // -- fused dense projections ---------------------------------------------

    struct FusedCandidate {
        std::string output_name;
        std::vector<std::uint32_t> ids;
        std::vector<std::string> names;
        std::int64_t rows = 0;
        std::int64_t cols = 0;
        std::uint64_t bytes = 0;
    };

    std::optional<FusedCandidate> fused_join_candidate(const std::string& output_name,
                                                       const std::vector<std::string>& inputs) {
        if (find(output_name) != nullptr) {
            return std::nullopt;
        }
        FusedCandidate candidate;
        candidate.output_name = output_name;
        std::optional<std::int64_t> cols;
        for (const std::string& name : inputs) {
            const SourceTensor* raw = find(name);
            if (raw == nullptr || raw->shape.size() != 2 ||
                !is_raw(raw->encoding, PieLoaderDType::BF16)) {
                return std::nullopt;
            }
            if (cols.has_value() && raw->shape[1] != *cols) {
                return std::nullopt;
            }
            cols = raw->shape[1];
            candidate.ids.push_back(raw->id);
            candidate.names.emplace_back(raw->name);
            candidate.rows += raw->shape[0];
            candidate.bytes += raw->span_bytes;
        }
        candidate.cols = cols.value_or(0);
        return candidate;
    }

    void publish_fused(const std::vector<FusedCandidate>& candidates) {
        for (const FusedCandidate& candidate : candidates) {
            for (std::uint32_t id : candidate.ids) {
                consumed_.insert(id);
            }
            std::vector<Node> parts;
            parts.reserve(candidate.names.size());
            for (const std::string& name : candidate.names) {
                parts.push_back(contract_->src(name));
            }
            define(candidate.output_name, contract_->cat(parts, 0),
                   pie_loader::raw(PieLoaderDType::BF16),
                   std::vector<std::int64_t>{candidate.rows, candidate.cols});
        }
    }

    void add_dense_fused_projection_joins(bool runtime_quant_enabled) {
        if (target_.backend != PieLoaderBackendKind::Cuda || target_.tp_size != 1 ||
            runtime_quant_enabled || profile_.skip_dense_qkv_fusion) {
            return;
        }
        std::vector<FusedCandidate> qkv;
        std::vector<FusedCandidate> gate_up;
        std::uint64_t qkv_bytes = 0;
        std::uint64_t gate_up_bytes = 0;
        for (std::uint32_t layer = 0; layer < facts_.num_hidden_layers; ++layer) {
            const std::string p = "model.layers." + std::to_string(layer) + ".";
            if (auto candidate = fused_join_candidate(
                    p + "self_attn.qkv_proj.fused.weight",
                    {p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight",
                     p + "self_attn.v_proj.weight"})) {
                qkv_bytes += candidate->bytes;
                qkv.push_back(std::move(*candidate));
            }
            if (auto candidate = fused_join_candidate(
                    p + "mlp.gate_up_proj.fused.weight",
                    {p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"})) {
                gate_up_bytes += candidate->bytes;
                gate_up.push_back(std::move(*candidate));
            }
        }
        if (qkv.empty() && gate_up.empty()) {
            return;
        }

        // Fused dense projections replace the original TP1 BF16 tensors, and
        // the unfused fallback binds non-owning views into the fused buffer, so
        // this is not a persistent duplicate-memory budget. It selects which
        // groups get a fused GEMM: all groups through 8B-class Qwen models, and
        // QKV-only above that, where gate/up fusion has regressed.
        constexpr std::uint64_t kBudgetBytes = 10ull * 1024 * 1024 * 1024;
        std::vector<FusedCandidate> chosen;
        if (qkv_bytes + gate_up_bytes <= kBudgetBytes) {
            chosen.insert(chosen.end(), qkv.begin(), qkv.end());
            chosen.insert(chosen.end(), gate_up.begin(), gate_up.end());
        } else {
            // QKV first when the full set does not fit: it is much smaller than
            // gate/up on Qwen-style models and it is what enables the fused
            // decode postprocess, without giving up large-model KV capacity.
            std::uint64_t used = 0;
            if (qkv_bytes <= kBudgetBytes) {
                used = qkv_bytes;
                chosen.insert(chosen.end(), qkv.begin(), qkv.end());
            }
            if (gate_up_bytes <= kBudgetBytes - used) {
                chosen.insert(chosen.end(), gate_up.begin(), gate_up.end());
            }
        }
        publish_fused(chosen);
    }

    void add_mla_fused_projection_joins() {
        if (target_.backend != PieLoaderBackendKind::Cuda || !profile_.mla_fused_joins) {
            return;
        }
        std::vector<FusedCandidate> candidates;
        for (std::uint32_t layer = 0; layer < facts_.num_hidden_layers; ++layer) {
            const std::string p = "model.layers." + std::to_string(layer) + ".";
            // q_a_proj + kv_a_proj_with_mqa share an input (norm_x, unsharded).
            if (auto candidate = fused_join_candidate(
                    p + "self_attn.q_kv_a_proj.fused.weight",
                    {p + "self_attn.q_a_proj.weight", p + "self_attn.kv_a_proj_with_mqa.weight"})) {
                candidates.push_back(std::move(*candidate));
            }
            // Shared gate + up share an input (norm_y).
            if (auto candidate = fused_join_candidate(
                    p + "mlp.shared_experts.gate_up_proj.fused.weight",
                    {p + "mlp.shared_experts.gate_proj.weight",
                     p + "mlp.shared_experts.up_proj.weight"})) {
                candidates.push_back(std::move(*candidate));
            }
        }
        publish_fused(candidates);
    }

    // -- Metal Qwen3.5 --------------------------------------------------------

    void add_metal_qwen35_contracts() {
        std::size_t declared = 0;
        for (const SourceTensor& raw : tensors_) {
            std::optional<std::string> output = metal_qwen35_runtime_name(raw.name);
            if (!output.has_value()) {
                continue;
            }
            if (ends_with(raw.name, ".weight") && is_raw(raw.encoding, PieLoaderDType::U32)) {
                const std::string base = std::string(
                    raw.name.substr(0, raw.name.size() - std::string_view(".weight").size()));
                const SourceTensor* scales = find(base + ".scales");
                const SourceTensor* biases = find(base + ".biases");
                if (scales == nullptr || biases == nullptr) {
                    fail("Metal affine-U4 weight '" + std::string(raw.name) +
                         "' is missing scales or biases");
                }
                push_mlx_affine_u4(raw, *scales, *biases, std::move(*output));
            } else {
                push_direct(raw, std::move(*output), std::nullopt);
            }
            ++declared;
        }
        if (declared == 0) {
            fail("Metal Qwen3.5 schema found no text-decoder tensors");
        }
    }

    void push_mlx_affine_u4(const SourceTensor& raw, const SourceTensor& scales,
                            const SourceTensor& biases, std::string output) {
        if (raw.shape.size() != 2 || scales.shape.size() != 2 ||
            biases.shape.size() != scales.shape.size() ||
            !std::equal(biases.shape.begin(), biases.shape.end(), scales.shape.begin())) {
            fail("MLX affine-U4 triplet '" + std::string(raw.name) +
                 "' has incompatible shapes");
        }
        const std::int64_t rows = raw.shape[0];
        const std::int64_t logical_cols = raw.shape[1] * 8;
        if (rows != scales.shape[0] || scales.shape[1] <= 0 ||
            logical_cols % scales.shape[1] != 0) {
            fail("MLX affine-U4 triplet '" + std::string(raw.name) +
                 "' cannot derive a group size");
        }
        const std::uint32_t group_size =
            u32_dim(logical_cols / scales.shape[1], "MLX affine-U4 group size");

        PieLoaderQuantSpecView quant =
            pie_loader::quant_spec(PieLoaderQuantScheme::MlxAffineU4, PieLoaderDType::BF16);
        quant.bits_per_element = 4;
        quant.group_size = group_size;
        quant.channel_axis = 1;
        quant.scale_dtype = static_cast<std::uint32_t>(logical_dtype(scales.encoding));
        quant.zero_point_dtype = static_cast<std::uint32_t>(logical_dtype(biases.encoding));
        const PieLoaderEncodingSpec encoding = pie_loader::quantized(
            contract_->with_block_shape(quant, {static_cast<std::int64_t>(group_size)}));

        // The checkpoint stores these 4-bit weights eight to a u32 word. The
        // contract names them for what they are; no byte moves.
        define(std::move(output),
               contract_->bitcast(contract_->src(std::string(raw.name)), {rows, logical_cols},
                                  encoding),
               encoding, std::vector<std::int64_t>{rows, logical_cols});
    }

    const Checkpoint& checkpoint_;
    const ModelFacts& facts_;
    const pie_loader::DeviceTarget& target_;
    std::string_view runtime_quant_;
    Mxfp4MoePolicy mxfp4_moe_;
    Component component_;
    ArchProfile profile_;
    std::vector<SourceTensor> tensors_;
    std::unordered_map<std::string_view, const SourceTensor*> by_name_;
    std::unordered_set<std::uint32_t> consumed_;
    ModelContract* contract_ = nullptr;
};

}  // namespace contract_detail

/// Resolve the caller's MoE request against what the device can do.
///
/// The driver measured `native_mxfp4_moe`, so the driver is what turns `Auto`
/// into an answer. It used to be the loader, which meant the loader needed the
/// three-way request; under a contract it needs neither, because a contract
/// either declares an MXFP4 expert weight or it does not.
inline Mxfp4MoePolicy resolve_mxfp4_moe(Mxfp4MoeRequest request,
                                                 bool native_mxfp4_moe) {
    switch (request) {
        case Mxfp4MoeRequest::RoutedDecode:
            return Mxfp4MoePolicy::RoutedDecode;
        case Mxfp4MoeRequest::EagerBf16:
            return Mxfp4MoePolicy::EagerBf16;
        case Mxfp4MoeRequest::NativeGemm:
            if (!native_mxfp4_moe) {
                contract_detail::fail(
                    "mxfp4_moe=NativeGemm requested, but the target reports "
                    "native_mxfp4_moe=false");
            }
            return Mxfp4MoePolicy::NativeGemm;
        case Mxfp4MoeRequest::Auto:
        default:
            // Auto follows the device: a native MXFP4 GEMM is always the better
            // path when the kernels exist, and decoding on the routed path is
            // the fallback that works everywhere.
            return native_mxfp4_moe ? Mxfp4MoePolicy::NativeGemm
                                    : Mxfp4MoePolicy::RoutedDecode;
    }
}

/// Author the contract this driver will bind, against this checkpoint.
///
/// Throws `std::runtime_error` with a message naming the tensor when the
/// checkpoint does not match what the family expects. The contract is left in
/// `out`, which must outlive the compile call that consumes its `view()`.
inline void author_model_contract(const Checkpoint& checkpoint, const ModelFacts& facts,
                                  const pie_loader::DeviceTarget& target,
                                  std::string_view runtime_quant,
                                  Mxfp4MoeRequest mxfp4_moe,
                                  Component component, ModelContract& out) {
    if (component == Component::Encode &&
        target.backend == PieLoaderBackendKind::Cuda) {
        if (target.tp_size != 1) {
            contract_detail::fail(
                "encode-scoped CUDA loading does not support tensor parallelism");
        }
        // Scoping a contract down to the towers is only meaningful for a family
        // whose towers the encode pipeline actually binds. Any other family
        // would silently get a contract of whatever happened to match the
        // prefixes, which for a multimodal checkpoint is not empty and not
        // right.
        if (facts.model_type != "gemma4" && facts.model_type != "gemma4_text") {
            contract_detail::fail("encode-scoped CUDA loading currently supports Gemma-4, got " +
                                  facts.model_type);
        }
    } else if (component == Component::Encode) {
        // Encode scoping is a CUDA path; elsewhere the request is a no-op
        // rather than an error, so a backend that has no encode pipeline still
        // gets the contract it would have got without asking.
        component = Component::Full;
    }
    contract_detail::ContractAuthor author(
        checkpoint, facts, target, runtime_quant,
        resolve_mxfp4_moe(mxfp4_moe, target.native_mxfp4_moe), component);
    author.build(out);
    if (out.empty()) {
        contract_detail::fail("no contract was authored for model_type='" + facts.model_type +
                              "'; the driver must declare what it binds");
    }
}

}  // namespace pie_driver

#pragma once

/// The toolkit a model family uses to state what it binds.
///
/// A contract is the *program* half of `plan = compile(source_facts, program,
/// target)`. It names every tensor this driver wants and gives, for each, an
/// expression over the checkpoint's tensors that produces it. The loader
/// type-checks the expression against what the files actually contain, lowers
/// it to storage instructions, and schedules them; it does not decide *what* to
/// build. That is why a family the loader has never heard of loads exactly as
/// well as one it has (`loader/architecture.md` §12 row 12).
///
/// This header is the mechanism and nothing else. The knowledge — that Phi-3
/// ships a fused QKV, that Kimi hides the decoder under `language_model.`, that
/// GPT-OSS wants Marlin-repacked MXFP4 groups — lives in each family's own
/// directory next to its forward pass, and reaches the loader through the
/// `author_contract` hook on that family's `ArchEntry` row. One row, one
/// `model_type`, one statement of what the family binds and how it is bound.
///
/// It is CUDA's. Metal authors its own contract in
/// `driver/metal/src/model/qwen3_5/`, because a contract is what *this driver*
/// will bind, and the two drivers bind different things: Metal renames every
/// tensor for MLX's binder, fuses nothing, and has no tensor parallelism. A
/// header shared between them made each carry the other's vocabulary.
///
/// Everything here reads two things and nothing else: the checkpoint's tensor
/// table (`pie_loader::Checkpoint`, the same parse the compile will use) and the
/// config facts the driver already parsed. Neither is a second reading of
/// anything, which is the point — the arrangement before this had the driver
/// parse `config.json` for its own model and the loader parse it again for the
/// contract, and two readings that can disagree is one too many.

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

namespace pie_cuda_driver::model {

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
/// Four values. Everything else a contract needs — which tensors exist, what
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
    /// `head_dim`. TP splits an attention projection by rows, and whether a
    /// row split lands on a head boundary is not a question the row count can
    /// answer -- see `ContractBuilder::check_head_granularity`.
    std::uint32_t head_dim = 0;
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
           a.quant.channel_axis == b.quant.channel_axis;
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

/// The axis tensor parallelism splits a tensor on, read off its name.
///
/// Not any one family's rule, despite the family-flavoured tails in the list.
/// It is the HF naming convention's answer: a name ending `.q_proj.weight`
/// denotes a column-parallel projection whatever model ships it, and `.sinks`,
/// `.w1/.w2/.w3` and `.linear_attn.*` are conventions several families adopted,
/// not identities. A family whose checkpoint uses a *different* convention for
/// the same operator says so with `shard_axis_fn` — DeepSeek-V4 is the one such
/// family, and its rule lives in its own header.
inline ShardAxis hf_shard_axis(std::string_view name) {
    // A companion scale splits exactly like the weight it scales, so ask about
    // the weight. Stating it as two lists instead -- every `<proj>.weight` in
    // one, every `<proj>.weight_scale_inv` beside it -- is what this was, and
    // the axis-0 list grew its scale entries while the axis-1 list never did:
    // `o_proj`, `down_proj` and `w2` handed a rank its slice of the weight and
    // the whole model's scale next to it. Whether the scale has an axis to
    // split at all is a question about its shape, and
    // `ContractBuilder::splittable_axis` answers that one.
    for (const std::string_view suffix :
         {".weight_scale_inv", ".weight_scale", ".weight_packed", ".scale"}) {
        if (ends_with(name, suffix)) {
            const std::string_view base = name.substr(0, name.size() - suffix.size());
            const ShardAxis of_weight = hf_shard_axis(std::string(base) + ".weight");
            return of_weight.has_value() ? of_weight : hf_shard_axis(base);
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
             ".w3.bias",              ".linear_attn.in_proj_z.weight",
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

/// A routed or shared expert's projection, under the HF MoE naming convention.
inline bool is_expert_projection(std::string_view name) {
    return (contains(name, ".mlp.experts.") || contains(name, ".mlp.shared_experts.")) &&
           ends_with_any(name, {".gate_proj.weight", ".up_proj.weight", ".down_proj.weight"});
}

/// Which weights a runtime-quant request re-encodes, given the target scheme.
///
/// A statement about this driver's GEMMs, not about any model: the allowlist is
/// exactly the projections a quantized kernel exists for.
inline bool runtime_quantizable_name(std::string_view name, PieLoaderQuantScheme scheme) {
    if (scheme == PieLoaderQuantScheme::Mxfp4E2M1E8M0) {
        // FP4 reaches experts only. Attention projections stay FP8-plus-scale
        // (block-scaled GEMM) because this hardware has no FP4 GEMM for them.
        return is_expert_projection(name);
    }
    return ends_with_any(name, {".self_attn.q_proj.weight", ".self_attn.k_proj.weight",
                                ".self_attn.v_proj.weight", ".self_attn.o_proj.weight",
                                ".self_attn.q_a_proj.weight", ".self_attn.q_b_proj.weight",
                                ".self_attn.kv_a_proj_with_mqa.weight",
                                ".self_attn.kv_b_proj.weight", ".mlp.gate_proj.weight",
                                ".mlp.up_proj.weight", ".mlp.down_proj.weight"}) ||
           is_expert_projection(name);
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
    // One MXFP4 block scale covers 32 contiguous elements along K, which
    // `group_size` above already says.
    return pie_loader::quantized(quant);
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
/// the language model. The four prefixes are the HF multimodal convention, so
/// this reads as a rule rather than as a list of the families using it today.
/// Under a driver-authored contract it is not a filter applied to a finished
/// contract — it is simply which tensors get declared.
inline bool is_tower_output(std::string_view name) {
    return name.rfind("model.vision_tower.", 0) == 0 ||
           name.rfind("model.embed_vision.", 0) == 0 ||
           name.rfind("model.audio_tower.", 0) == 0 || name.rfind("model.embed_audio.", 0) == 0;
}


// -- the builder --------------------------------------------------------------

/// Accumulates one family's declarations against one checkpoint.
///
/// A family's `author_contract` receives this, sets the handful of knobs its
/// layout needs, runs the shared passes it wants, adds whatever is peculiar to
/// it, and finishes with `publish_remaining()`. The order is not incidental: a
/// pass that fuses q/k/v into one buffer has to claim those tensors before the
/// generic loop publishes them separately.
///
/// There used to be an `ArchProfile` table here — thirteen booleans keyed by
/// `model_type`, consulted by passes that all ran for every model and returned
/// early for most of them. It was a second list of `model_type` strings beside
/// the arch registry's, and the two had drifted: six strings appeared in one and
/// not the other. A family that states its own rules cannot drift from itself.
class ContractBuilder {
public:
    ContractBuilder(const Checkpoint& checkpoint, const ModelFacts& facts,
                    const pie_loader::DeviceTarget& target, std::string_view runtime_quant,
                    Mxfp4MoePolicy mxfp4_moe, Component component, ModelContract& out)
        : facts_(facts),
          target_(target),
          runtime_quant_(runtime_quant),
          mxfp4_moe_(mxfp4_moe),
          component_(component),
          tensors_(checkpoint.tensors()),
          contract_(out) {
        out.align(std::max<std::uint32_t>(1, target_.preferred_alignment));
        by_name_.reserve(tensors_.size());
        for (const SourceTensor& raw : tensors_) {
            by_name_.emplace(raw.name, &raw);
        }
    }

    // -- what the family is authoring against --------------------------------

    const ModelFacts& facts() const noexcept { return facts_; }
    const pie_loader::DeviceTarget& target() const noexcept { return target_; }
    Mxfp4MoePolicy mxfp4_moe() const noexcept { return mxfp4_moe_; }
    ModelContract& contract() noexcept { return contract_; }
    const std::vector<SourceTensor>& tensors() const noexcept { return tensors_; }

    // -- knobs, each one a claim about this family ---------------------------

    /// Source tensors *may* live under this prefix; it is stripped from output
    /// names, and every pass that goes looking for a source name by hand puts
    /// it back through `source_name`.
    ///
    /// A prefix the checkpoint does not use is not a prefix. Kimi ships both
    /// shapes -- the released text checkpoints name the decoder
    /// `model.layers.*` and the multimodal ones nest it under
    /// `language_model.` -- and one arch row covers both only if this is a
    /// question asked of the checkpoint rather than a declaration about the
    /// family. Declaring it unconditionally is what it was, and against a text
    /// checkpoint it filtered every tensor out and failed authoring outright.
    void source_prefix(std::string_view prefix) {
        source_prefix_ = {};
        for (const SourceTensor& raw : tensors_) {
            if (raw.name.rfind(prefix, 0) == 0) {
                source_prefix_ = prefix;
                return;
            }
        }
    }

    /// The name a source tensor is published under, given the prefix.
    ///
    /// The counterpart to `source_name`: a pass that walks `tensors()` already
    /// holds the raw name and publishes it through this.
    std::string output_name(std::string_view raw_name) const {
        if (!source_prefix_.empty() && raw_name.rfind(source_prefix_, 0) == 0) {
            return std::string(raw_name.substr(source_prefix_.size()));
        }
        return std::string(raw_name);
    }

    /// The name a source tensor has in the checkpoint, given the prefix.
    ///
    /// A pass that goes looking for one specific tensor names it the way the
    /// bind path does and lets this put the prefix back. A pass that instead
    /// walks `tensors()` already has the raw name and publishes it through
    /// `output_name`. Getting either half wrong is silent: `fused_join_candidate`
    /// and friends return "no candidate" for a name that is not there.
    std::string source_name(std::string_view bound_name) const {
        return std::string(source_prefix_) + std::string(bound_name);
    }

    /// Where the decoder's layers are named, up to the layer index.
    ///
    /// Only the fused-projection pass needs it, and only because it goes
    /// looking for specific names instead of walking `tensors()`. Matching on
    /// the `.self_attn.q_proj.weight` suffix alone would be prefix-free but
    /// wrong: Gemma-4's vision and audio towers have projections of that name
    /// too, and fusing one would consume weights its bind path still reads.
    void decoder_layer_prefix(std::string_view prefix) { decoder_layer_prefix_ = prefix; }

    /// Tensor-parallel shard-axis strategy, keyed by tensor name. Defaults to
    /// the HF convention; a family whose checkpoint names the same operator
    /// differently (DeepSeek-V4's native `.ffn.experts.w*`) registers its own.
    void shard_axis_fn(ShardAxis (*fn)(std::string_view)) { shard_axis_fn_ = fn; }

    /// Shard `embed_tokens` on axis 0 under TP, to save per-rank memory.
    void shard_embed_tokens() { shard_embed_tokens_ = true; }

    /// Keep `lm_head` replicated under TP. Costs ~1.7 GB a rank on Kimi K2.6
    /// and buys not needing a TP greedy argmax on the logits path.
    void replicate_lm_head() { replicate_lm_head_ = true; }

    /// BF16 -> FP8/INT8 runtime quant is wired for this family.
    void allow_bf16_runtime_quant() { allow_bf16_rq_ = true; }

    /// FP4/MXFP4 runtime quant of routed experts is wired for this family.
    void allow_mxfp4_runtime_quant() { allow_mxfp4_rq_ = true; }

    /// This family's bind path can be scoped to the multimodal towers alone.
    ///
    /// Only meaningful for a family whose towers the encode pipeline actually
    /// binds. Any other family would silently get a contract of whatever
    /// happened to match the prefixes, which for a multimodal checkpoint is not
    /// empty and not right.
    void allow_encode_scope() {
        if (component_ == Component::Encode && target_.tp_size != 1) {
            fail("encode-scoped loading does not support tensor parallelism");
        }
        encode_scope_allowed_ = true;
    }

    // -- lookups -------------------------------------------------------------

    const SourceTensor* find(const std::string& name) const {
        const auto it = by_name_.find(name);
        return it == by_name_.end() ? nullptr : it->second;
    }

    /// Claim `id` for a pass that has published it under some other name, so
    /// the generic tail does not publish it a second time.
    void consume(std::uint32_t id) { consumed_.insert(id); }

    /// The `[start, len)` this rank owns of a `full`-long axis, as numbers.
    ///
    /// **`Repack` only.** A repack spec carries `source_row_offset` as an
    /// integer a kernel reads, so the escape hatch has nowhere to put a `Shard`
    /// node and the rank has to be resolved here. Everything affine must use
    /// [`band`] or [`split`] instead: baking `start + tp_rank * len` into a
    /// `Slice` makes the contract a second per-rank input to compilation, and
    /// §6.3's rank-uniformity guarantee rests on the target being the only one.
    std::pair<std::int64_t, std::int64_t> local_range(std::int64_t full,
                                                      const std::string& what) const {
        return contract_detail::local_range(full, target_, what);
    }

    /// Record that `expr` is split across ranks along `axis`.
    ///
    /// Left alone when there is nothing to split, so a single-GPU contract is
    /// identical to one authored without sharding.
    Node split(Node expr, std::uint8_t axis) {
        return target_.tp_size <= 1 ? expr : contract_.shard(expr, axis);
    }

    /// This rank's share of a `full`-long axis.
    ///
    /// A declared shape, never an offset. Divisibility is not checked here: the
    /// loader rejects an indivisible `Shard` with a message that names the
    /// tensor and the axis, and checking the same fact twice is how the
    /// per-family divisibility table over `config.json` used to drift.
    std::int64_t local_extent(std::int64_t full) const {
        const std::int64_t world = std::max<std::int64_t>(1, target_.tp_size);
        return full % world == 0 ? full / world : full;
    }

    /// The `[start, start + len)` band of `expr`, split across ranks.
    ///
    /// Returns the expression and the extent this rank ends up holding, so a
    /// caller states a declared shape without ever computing an offset. This is
    /// what a fused source wants: slice the band out first — that part is the
    /// same on every rank — and let `Resolver::specialize`, the loader's one
    /// rank-aware pass, do the arithmetic.
    std::pair<Node, std::int64_t> band(Node expr, std::uint8_t axis, std::int64_t start,
                                       std::int64_t len) {
        return {split(contract_.slice(expr, axis, start, len), axis), local_extent(len)};
    }


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
        auto defined = contract_.define(std::move(name), expr, encoding);
        if (shape.has_value()) {
            defined.expect(std::move(*shape));
        }
    }

    void push_direct(const SourceTensor& raw, std::string output, ShardAxis axis) {
        std::vector<std::int64_t> raw_shape = shape_of(raw);
        axis = splittable_axis(raw.name, raw_shape, axis);
        check_head_granularity(raw.name, raw_shape, axis);
        auto [expr, shape] =
            shard(contract_.src(std::string(raw.name)), std::move(raw_shape), axis);
        define(std::move(output), expr, raw.encoding, std::move(shape));
    }

    /// True for a tensor that only scales another one.
    static bool is_companion_scale(std::string_view name) {
        return ends_with_any(name, {".weight_scale_inv", ".weight_scale", ".scale"});
    }

    /// Demote a companion scale to "replicate" when it has nothing to split.
    ///
    /// A scale follows its weight, but only a block or per-channel scale has
    /// an axis of its own: a per-tensor scale is one number for a whole
    /// projection and follows it vacuously. Asking `hf_shard_axis` about that
    /// is asking a question about the name when the answer is in the shape.
    ///
    /// A scale that *does* have the axis and does not divide is left alone on
    /// purpose -- the loader rejects it by name, which is how a 1536-wide FP8
    /// expert says it has 12 blocks and cannot go to 8 ranks.
    static ShardAxis splittable_axis(std::string_view name,
                                     const std::vector<std::int64_t>& shape, ShardAxis axis) {
        if (!axis.has_value() || !is_companion_scale(name)) {
            return axis;
        }
        const std::size_t index = *axis;
        if (index >= shape.size() || shape[index] <= 1) {
            return std::nullopt;
        }
        return axis;
    }

    /// A row-parallel attention projection has to split on a head boundary.
    ///
    /// The loader asks whether `tp_size` divides the row count, and for a
    /// projection of `heads * head_dim` rows that is the weaker question: four
    /// KV heads of 256 make 1024 rows, 1024 divides eight ways, and the same
    /// forward pass then computes `num_key_value_heads / tp_size` and gets
    /// zero heads a rank. Both divisions succeed and they disagree, so the
    /// sharper one belongs here, where `head_dim` is known and the loader's is
    /// not -- 1024 is just 1024 by the time it gets there.
    void check_head_granularity(std::string_view name, const std::vector<std::int64_t>& shape,
                                ShardAxis axis) const {
        const std::int64_t d = facts_.head_dim;
        const std::int64_t world = target_.tp_size;
        if (!axis.has_value() || *axis != 0 || world <= 1 || d <= 0 || shape.empty()) {
            return;
        }
        if (!ends_with_any(name, {".q_proj.weight", ".k_proj.weight", ".v_proj.weight",
                                  ".q_proj.bias", ".k_proj.bias", ".v_proj.bias"})) {
            return;
        }
        // A fused or otherwise non-head-shaped projection is not this rule's
        // business; `local_range` still has the row count covered.
        if (shape[0] % d != 0) {
            return;
        }
        const std::int64_t heads = shape[0] / d;
        if (heads % world == 0) {
            return;
        }
        fail(std::string(name) + " is " + std::to_string(heads) + " head(s) of " +
             std::to_string(d) + ", which tp_size " + std::to_string(target_.tp_size) +
             " does not divide; a rank cannot hold part of an attention head, so use a "
             "tp_size that divides the head count or run single-GPU");
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
        return {contract_.shard(expr, *axis), std::move(shape)};
    }

    ShardAxis shard_axis(std::string_view name) const {
        if (target_.tp_size <= 1) {
            return std::nullopt;
        }
        // Shard embed_tokens on axis 0 to save per-rank memory (Kimi, GLM-5.1).
        if (shard_embed_tokens_ && ends_with(name, ".embed_tokens.weight")) {
            return std::uint8_t{0};
        }
        if (replicate_lm_head_ && ends_with(name, ".lm_head.weight")) {
            return std::nullopt;
        }
        return shard_axis_fn_(name);
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
            contract_.repack(contract_.src(std::string(raw.name)), spec, shape, encoding);
        define(std::move(output), node, encoding, std::move(shape));
    }

    // -- shared passes -------------------------------------------------------

    /// Re-join each rank's gate and up bands of a pre-fused expert tensor.
    /// `gate_second` publishes each expert's halves as `[up|gate]` instead of
    /// the checkpoint's `[gate|up]`. That is the order flashinfer's CUTLASS
    /// MoE reads fc1's output in, and declaring it here makes the reorder a
    /// load-time view of the source rather than a runtime pass that mutates a
    /// resident weight. It applies with or without sharding, so unlike the
    /// slicing this runs at tp_size == 1 too.
    void fused_moe_gate_up_tp_slices(bool gate_second = false) {
        const bool sharding = target_.tp_size > 1;
        if (!sharding && !gate_second) {
            return;
        }
        for (const SourceTensor& raw : tensors_) {
            if (!source_name_allowed(raw.name)) {
                continue;
            }
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
            // Gate rows [0, I) and up rows [I, 2I) are sharded independently
            // and re-joined, so the local halves stay adjacent per expert.
            const Node src = contract_.src(std::string(raw.name));
            // Gate rows [0, I) and up rows [I, 2I) are taken separately so
            // that a shard splits each half independently and the local
            // halves stay adjacent per expert. Build only the nodes that are
            // used: an unreferenced slice still enters the DAG.
            const std::pair<Node, std::int64_t> gate_part =
                sharding ? band(src, 1, 0, full_i)
                         : std::pair<Node, std::int64_t>{
                               contract_.slice(src, 1, 0, full_i), full_i};
            const std::pair<Node, std::int64_t> up_part =
                sharding ? band(src, 1, full_i, full_i)
                         : std::pair<Node, std::int64_t>{
                               contract_.slice(src, 1, full_i, full_i), full_i};
            const Node gate = gate_part.first;
            const Node up = up_part.first;
            const std::int64_t local_i = gate_part.second;
            push_expr(output_name(raw.name), raw, {experts, 2 * local_i, hidden},
                      contract_.cat(gate_second ? std::vector<Node>{up, gate}
                                                : std::vector<Node>{gate, up},
                                    1));
        }
    }

    struct FusedCandidate {
        std::string output_name;
        /// The source tensors, in join order. Borrowed from `tensors_`, which
        /// outlives every candidate.
        std::vector<const SourceTensor*> parts;
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
            candidate.parts.push_back(raw);
            candidate.rows += raw->shape[0];
            candidate.bytes += raw->span_bytes;
        }
        candidate.cols = cols.value_or(0);
        return candidate;
    }

    void publish_fused(const std::vector<FusedCandidate>& candidates) {
        for (const FusedCandidate& candidate : candidates) {
            std::vector<Node> parts;
            std::vector<std::int64_t> local_rows;
            parts.reserve(candidate.parts.size());
            local_rows.reserve(candidate.parts.size());
            std::int64_t rows = 0;
            for (const SourceTensor* raw : candidate.parts) {
                // Shard each part, then join: q/k/v have different row counts,
                // so a row-shard of the *concatenation* would cut across the
                // q/k boundary and hand a rank a mix of two projections.
                // Sharding first gives every rank its own band of each, and
                // their join is exactly this rank's fused weight.
                check_head_granularity(raw->name, shape_of(*raw), ShardAxis{0});
                parts.push_back(split(contract_.src(std::string(raw->name)), 0));
                local_rows.push_back(local_extent(raw->shape[0]));
                rows += local_rows.back();
            }
            define(candidate.output_name, contract_.cat(parts, 0),
                   pie_loader::raw(PieLoaderDType::BF16),
                   std::vector<std::int64_t>{rows, candidate.cols});

            // Re-publish each projection as a view into the bank. A bind path
            // that reads q/k/v individually -- a shared Gemma-4 layer, an
            // unfused GEMM -- then finds them under their own names at every
            // `tp_size`, and the offset of a rank's band is stated once, here,
            // instead of being recomputed in pointer arithmetic per family.
            std::int64_t at = 0;
            for (std::size_t part = 0; part < candidate.parts.size(); ++part) {
                const SourceTensor* raw = candidate.parts[part];
                define(output_name(raw->name),
                       contract_.slice(contract_.out(candidate.output_name), 0, at,
                                       local_rows[part]),
                       pie_loader::raw(PieLoaderDType::BF16),
                       std::vector<std::int64_t>{local_rows[part], candidate.cols});
                consumed_.insert(raw->id);
                at += local_rows[part];
            }
        }
    }

    /// Fuse q/k/v and gate/up into one buffer each, where the GEMM wants it.
    ///
    /// A CUDA kernel decision, not a fact about any model: the same
    /// checkpoint on a backend without a fused GEMM declares the three
    /// projections separately. A family whose bind path reads q/k/v
    /// individually (Qwen3-MoE, GPT-OSS) simply does not call this.
    void dense_fused_projection_joins() {
        if (runtime_quant_scheme().has_value()) {
            return;
        }
        std::vector<FusedCandidate> qkv;
        std::vector<FusedCandidate> gate_up;
        std::uint64_t qkv_bytes = 0;
        std::uint64_t gate_up_bytes = 0;
        for (std::uint32_t layer = 0; layer < facts_.num_hidden_layers; ++layer) {
            const std::string p =
                std::string(decoder_layer_prefix_) + std::to_string(layer) + ".";
            const std::string s = source_name(p);
            if (auto candidate = fused_join_candidate(
                    p + "self_attn.qkv_proj.fused.weight",
                    {s + "self_attn.q_proj.weight", s + "self_attn.k_proj.weight",
                     s + "self_attn.v_proj.weight"})) {
                qkv_bytes += candidate->bytes;
                qkv.push_back(std::move(*candidate));
            }
            if (auto candidate = fused_join_candidate(
                    p + "mlp.gate_up_proj.fused.weight",
                    {s + "mlp.gate_proj.weight", s + "mlp.up_proj.weight"})) {
                gate_up_bytes += candidate->bytes;
                gate_up.push_back(std::move(*candidate));
            }
        }
        if (qkv.empty() && gate_up.empty()) {
            return;
        }

        // Fused dense projections replace the original BF16 tensors, and the
        // unfused fallback binds non-owning views into the fused buffer, so
        // this is not a persistent duplicate-memory budget. It selects which
        // groups get a fused GEMM: all groups through 8B-class Qwen models, and
        // QKV-only above that, where gate/up fusion has regressed. Measured on
        // whole-checkpoint bytes so a model lands in the same class at every
        // `tp_size` — it is a proxy for model class, not for device memory.
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

    // -- the generic tail ----------------------------------------------------

    /// Declare every source tensor no pass has claimed, under its own name.
    ///
    /// Always last. A family that publishes nothing else still gets a complete
    /// contract from this, which is why most rows in the arch table point
    /// straight at `author_dense_contract`.
    void publish_remaining() {
        const std::optional<PieLoaderQuantScheme> scheme = runtime_quant_scheme();
        for (const SourceTensor& raw : tensors_) {
            if (consumed_.count(raw.id) != 0 || !source_name_allowed(raw.name)) {
                continue;
            }
            if (scheme.has_value() && runtime_quantizable_name(raw.name, *scheme)) {
                push_runtime_quant(raw, std::string(raw.name), *scheme);
            } else {
                push_direct(raw, output_name(raw.name), shard_axis(raw.name));
            }
        }
    }

    /// Check what only the whole contract can answer, after the family is done.
    void finish() const {
        if (component_ == Component::Encode && !encode_scope_allowed_) {
            fail("encode-scoped loading is not supported for model_type='" + facts_.model_type +
                 "'");
        }
        if (contract_.empty()) {
            fail("no contract was authored for model_type='" + facts_.model_type +
                 "'; the driver must declare what it binds");
        }
    }

private:

    // -- the source-prefix rule ----------------------------------------------
    //
    // A family that sets `source_prefix` is saying its checkpoint nests the
    // decoder under a prefix the bind path does not use. Both halves of that
    // rule are read only by `publish_remaining`; a family that wants a
    // prefix-stripped name for one tensor states the name it wants.

    bool source_name_allowed(std::string_view raw_name) const {
        return source_prefix_.empty() ||
               raw_name.rfind(source_prefix_, 0) == 0;
    }

    // -- runtime quantization ------------------------------------------------

    /// The scheme this author's runtime-quant request resolves to, or none.
    ///
    /// Resolved on each call rather than once in the constructor, and it has to
    /// stay that way: `runtime_quant_allowed` reads `allow_bf16_rq_` and
    /// `allow_mxfp4_rq_`, which the *family* sets after construction. Caching
    /// this at build time would answer with the knobs unset and silently refuse
    /// a scheme the family goes on to allow.

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
        if (!runtime_quant_allowed(scheme)) {
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
                break;
            case PieLoaderQuantScheme::Int8Symmetric:
                quant = pie_loader::quant_spec(scheme, PieLoaderDType::I8);
                quant.bits_per_element = 8;
                quant.group_size = 1;
                quant.channel_axis = 0;
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
                break;
            default:
                fail("unsupported runtime_quant scheme");
        }

        auto [expr, shape] =
            shard(contract_.src(std::string(raw.name)), shape_of(raw), shard_axis(raw.name));
        define(std::move(output), expr, pie_loader::quantized(quant), std::move(shape));
    }

    bool runtime_quant_allowed(PieLoaderQuantScheme scheme) const {
        return scheme == PieLoaderQuantScheme::Mxfp4E2M1E8M0 ? allow_mxfp4_rq_ : allow_bf16_rq_;
    }

    const ModelFacts& facts_;
    const pie_loader::DeviceTarget& target_;
    std::string_view runtime_quant_;
    Mxfp4MoePolicy mxfp4_moe_;
    Component component_;
    std::vector<SourceTensor> tensors_;
    std::unordered_map<std::string_view, const SourceTensor*> by_name_;
    std::unordered_set<std::uint32_t> consumed_;
    ModelContract& contract_;

    std::string_view source_prefix_;
    ShardAxis (*shard_axis_fn_)(std::string_view) = hf_shard_axis;
    std::string_view decoder_layer_prefix_ = "model.layers.";
    bool shard_embed_tokens_ = false;
    bool replicate_lm_head_ = false;
    bool allow_bf16_rq_ = false;
    bool allow_mxfp4_rq_ = false;
    bool encode_scope_allowed_ = false;
};

}  // namespace contract_detail

using contract_detail::ContractBuilder;
using contract_detail::ShardAxis;


/// Resolve the caller's runtime-quantization request against what the device can
/// execute.
///
/// A request the device cannot run is dropped rather than refused: it is a
/// preference, and the checkpoint's own format is always a valid answer.
///
/// Resolved here, beside `resolve_mxfp4_moe`, because `fp8_native` is a fact
/// *this driver measured* off `cudaDeviceProp` and nothing on the far side of
/// the FFI ever read it. Sending it across so the answer could come back would
/// be asking the loader a question only the asker can answer (§8.1).
inline std::string_view resolve_runtime_quant(std::string_view request, bool fp8_native) {
    return request == "fp8" && !fp8_native ? std::string_view() : request;
}

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

/// The contract for a family that needs nothing beyond the name-pattern rules.
///
/// A dense decoder whose checkpoint already ships tensors under the names the
/// bind path reads: declare them, fuse the projections the GEMM wants, done.
/// Most rows in the arch table are this.
inline void author_dense_contract(ContractBuilder& b) {
    // Order is load-bearing, and it is the same order for every family: a pass
    // that fuses tensors has to claim them before `publish_remaining` declares
    // them one by one. The MoE slice runs first because a family can have both
    // a fused expert weight and fused dense projections (Gemma-4 does).
    b.fused_moe_gate_up_tp_slices();
    b.dense_fused_projection_joins();
    b.publish_remaining();
}

}  // namespace pie_cuda_driver::model

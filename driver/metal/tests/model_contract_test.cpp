// What this driver declares, and what the MLX affine-U4 path owes its kernels.
//
// The contract is the `program` half of `plan = compile(source_facts, program,
// target)` (`loader/architecture.md` §12 row 12): `author_model_contract` reads
// the checkpoint's tensor table and writes down every tensor the Metal binder
// will bind, as an expression over what is on disk. Nothing checked that.
//
// The loader v2 refactor is why this file exists. It
// deleted `QuantSpec::{scale_dtype, zero_point_dtype, block_shape}`, and
// `qwen3_5_contract.hpp::push_mlx_affine_u4` authored all three. It was fixed by
// deletion and then verified by *syntax only* — the names lined up, and nothing
// executed the arm. CUDA caught the same removal immediately because it has
// `driver/cuda/tests/model_contract_test.cpp`; this is Metal's.
//
// Two halves, as on the CUDA side. First the authoring primitives, against
// synthetic `SourceTensor`s — no checkpoint, no GPU, no Metal framework, so this
// half runs anywhere the loader archive links. Second, when `PIE_TEST_SNAPSHOT`
// points at a real checkpoint, the whole schema: every `Src` names a tensor that
// exists, no tensor is declared twice, the node arena is topologically sorted,
// and the contract compiles and verifies.
//
// The first half is the one that matters for M3. `push_mlx_affine_u4` derives a
// group size from two shapes and hands the result to kernels that accept
// exactly g64/b4 — `heap_bind.cpp` throws on anything else — so the quant spec
// it authors is a contract between two files that no single file states.

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "model_facts.hpp"
#include "pie_loader.h"
#include "pie_loader/model_contract.hpp"
#include "pie_loader/plan.hpp"
#include "pie_loader/request.hpp"
#include "pie_loader/source_checkpoint.hpp"

#include "loader/load_plan.hpp"
#include "model/contract.hpp"

namespace {

int failures = 0;

void check(bool ok, std::string_view what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << "\n";
        ++failures;
    }
}

std::string_view view_of(const pie_loader::PieLoaderBytes& b) {
    if (b.ptr == nullptr) return {};
    return {reinterpret_cast<const char*>(b.ptr), b.len};
}

pie_loader::PieLoaderExprKind kind_of(const pie_loader::PieLoaderExprNode& n) {
    return static_cast<pie_loader::PieLoaderExprKind>(n.kind);
}

/// A checkpoint tensor that never was. `shape` is borrowed, so the vector the
/// caller passes has to outlive the `SourceTensor` — every use below keeps it
/// in the same scope.
pie_loader::SourceTensor tensor_of(std::string_view name,
                                   const std::vector<std::int64_t>& shape,
                                   pie_loader::PieLoaderEncodingSpec encoding) {
    pie_loader::SourceTensor t;
    t.name = name;
    t.shape = std::span<const std::int64_t>(shape.data(), shape.size());
    t.encoding = encoding;
    return t;
}

// -- the MLX affine-U4 triplet ----------------------------------------------

// The checkpoint stores 4-bit weights eight to a u32 word, so a `[rows, cols/8]`
// u32 tensor *is* a `[rows, cols]` 4-bit one. The contract has to say so without
// moving a byte, which is a `Bitcast` carrying the logical extents and the quant
// spec — and the spec is what the field removal could have silently emptied.
void test_affine_u4_authors_the_quant_spec_the_kernels_need() {
    const std::int64_t rows = 2048;
    const std::int64_t group = 64;
    const std::int64_t logical_cols = 4096;

    const std::vector<std::int64_t> raw_shape{rows, logical_cols / 8};
    const std::vector<std::int64_t> group_shape{rows, logical_cols / group};
    const auto raw = tensor_of("w.weight", raw_shape,
                               pie_loader::raw(pie_loader::PieLoaderDType::U32));
    const auto scales = tensor_of("w.scales", group_shape,
                                  pie_loader::raw(pie_loader::PieLoaderDType::BF16));
    const auto biases = tensor_of("w.biases", group_shape,
                                  pie_loader::raw(pie_loader::PieLoaderDType::BF16));

    pie_loader::ModelContract c;
    pie::metal::model::contract_detail::push_mlx_affine_declared(c, raw, scales, biases, 4, 0, "layers.0.w");
    const auto v = c.view();

    check(v.tensors.len == 1, "the triplet is one declaration");
    if (v.tensors.len != 1) return;
    const auto& t = v.tensors.ptr[0];
    check(view_of(t.name) == "layers.0.w", "declared under the runtime name");

    // `expect` is the logical shape, not the packed one: a driver that declared
    // the on-disk `[rows, cols/8]` would type-check against the wrong tensor.
    check(t.shape.len == 2 && t.shape.ptr[0] == rows && t.shape.ptr[1] == logical_cols,
          "the declared shape is the unpacked one");

    const auto& enc = t.encoding;
    check(enc.kind == static_cast<std::uint32_t>(pie_loader::PieLoaderEncodingKind::Quant),
          "the declaration is quantized");
    check(enc.quant.scheme ==
              static_cast<std::uint32_t>(pie_loader::PieLoaderQuantScheme::MlxAffineU4),
          "the scheme is MLX affine-U4");
    check(enc.quant.logical_dtype ==
              static_cast<std::uint32_t>(pie_loader::PieLoaderDType::BF16),
          "the logical dtype is bf16");
    // These three are the whole point of the file. `bits_per_element` and
    // `group_size` used to sit beside `scale_dtype`, `zero_point_dtype` and
    // `block_shape`; deleting the latter three must not have disturbed them,
    // and `0` here would mean "the scheme's default" rather than a fault.
    check(enc.quant.bits_per_element == 4, "four bits per element");
    check(enc.quant.group_size == static_cast<std::uint32_t>(group),
          "the group size is derived from the scales, not defaulted");
    check(enc.quant.channel_axis == 1, "the quantized axis is the column axis");

    // Transmute(Src) and nothing else: the contract renames, it does not compute.
    check(v.nodes.len == 2, "the triplet costs two nodes");
    if (v.nodes.len != 2) return;
    check(kind_of(v.nodes.ptr[0]) == pie_loader::PieLoaderExprKind::Src, "the operand is a Src");
    check(view_of(v.nodes.ptr[0].name) == "w.weight", "the Src names the packed tensor");
    check(t.root == 1 && kind_of(v.nodes.ptr[1]) == pie_loader::PieLoaderExprKind::Transmute,
          "the root reinterprets the packed tensor");
    const auto& transmute = v.nodes.ptr[1];
    check(transmute.out_shape.len == 2 && transmute.out_shape.ptr[0] == rows &&
              transmute.out_shape.ptr[1] == logical_cols,
          "the transmute carries the unpacked extents");
    check(transmute.out_encoding.quant.group_size == static_cast<std::uint32_t>(group) &&
              transmute.out_encoding.quant.bits_per_element == 4,
          "the node's own encoding agrees with the declaration's");
}

// `heap_bind.cpp` rejects any plan whose MLX affine-U4 tensors are not g64/b4,
// because the qmv kernels index the group stride as a constant. That guard and
// the derivation above are in different files and neither states the other, so
// the pairing is only true if something checks it: a 4096-column weight with
// 64-column groups must derive 64, and a 128-column-group checkpoint must
// derive something the guard will refuse rather than something it will accept.
void test_the_derived_group_size_is_what_heap_bind_demands() {
    struct Case {
        std::int64_t logical_cols;
        std::int64_t groups;
        std::uint32_t expected;
    };
    for (const Case& c : {Case{4096, 64, 64}, Case{4096, 32, 128}, Case{1536, 24, 64}}) {
        const std::vector<std::int64_t> raw_shape{8, c.logical_cols / 8};
        const std::vector<std::int64_t> group_shape{8, c.groups};
        const auto raw = tensor_of("w.weight", raw_shape,
                                   pie_loader::raw(pie_loader::PieLoaderDType::U32));
        const auto scales = tensor_of("w.scales", group_shape,
                                      pie_loader::raw(pie_loader::PieLoaderDType::BF16));
        const auto biases = tensor_of("w.biases", group_shape,
                                      pie_loader::raw(pie_loader::PieLoaderDType::BF16));
        pie_loader::ModelContract contract;
        pie::metal::model::contract_detail::push_mlx_affine_declared(contract, raw, scales, biases, 4, 0, "w");
        const auto v = contract.view();
        check(v.tensors.len == 1 && v.tensors.ptr[0].encoding.quant.group_size == c.expected,
              "group size " + std::to_string(c.expected) + " derived from " +
                  std::to_string(c.logical_cols) + " columns in " + std::to_string(c.groups) +
                  " groups");
    }
}

// The one thing the tensors cannot say.
//
// A weight quantized 8-bit in groups of 64 and one quantized 4-bit in groups of
// 128 have identical shapes: the same u32 count against the same number of
// scales. So a schema that assumes the width and solves for the group cannot be
// wrong in a way the shapes reveal -- it just names a group nobody used, and
// mlx-community's 8-bit Llama-3.2-1B was refused as "g128/b4", a quantization
// that does not exist. The width has to come from `config.json`, and this is
// what says the two are then cross-checked rather than one being trusted.
void test_the_declared_width_is_checked_against_the_shapes() {
    // 4096 logical columns at 8 bits is 1024 u32 words; at 4 bits it is 512.
    const std::vector<std::int64_t> packed8{8, 1024};
    const std::vector<std::int64_t> groups64{8, 64};
    const auto bf16 = pie_loader::raw(pie_loader::PieLoaderDType::BF16);
    const auto raw = tensor_of("w.weight", packed8, pie_loader::raw(pie_loader::PieLoaderDType::U32));
    const auto scales = tensor_of("w.scales", groups64, bf16);
    const auto biases = tensor_of("w.biases", groups64, bf16);

    {
        pie_loader::ModelContract contract;
        pie::metal::model::contract_detail::push_mlx_affine_declared(contract, raw, scales, biases,
                                                                     8, 64, "w");
        const auto v = contract.view();
        check(v.tensors.len == 1 && v.tensors.ptr[0].encoding.quant.group_size == 64 &&
                  v.tensors.ptr[0].encoding.quant.bits_per_element == 8,
              "an 8-bit g64 triplet is described as 8-bit g64 when the config says 8 bits");
    }
    {
        // The same bytes with the config's width set to 4 -- which is what a
        // file-wide `quantization` block says about a checkpoint whose per
        // tensor overrides push a few projections to 8. The GROUP is the
        // number that holds for the whole file, so it wins, and the width is
        // read back out of the shapes as 8 rather than believed as 4.
        pie_loader::ModelContract contract;
        pie::metal::model::contract_detail::push_mlx_affine_declared(contract, raw, scales, biases,
                                                                     4, 64, "w");
        const auto v = contract.view();
        check(v.tensors.len == 1 && v.tensors.ptr[0].encoding.quant.group_size == 64 &&
                  v.tensors.ptr[0].encoding.quant.bits_per_element == 8,
              "a file-wide 4-bit declaration does not override an 8-bit tensor's own shapes");
    }
    {
        // The group is now the one told number, so a wrong one is the one way
        // left to mis-describe these bytes -- and it has to fail rather than
        // resolve, because g32 over these shapes implies a 16-bit width that
        // no affine kernel here reads.
        pie_loader::ModelContract contract;
        bool threw = false;
        try {
            pie::metal::model::contract_detail::push_mlx_affine_declared(contract, raw, scales,
                                                                         biases, 8, 32, "w");
        } catch (const std::exception&) {
            threw = true;
        }
        check(threw, "a declared group the shapes contradict is refused, not silently widened");
    }
}

bool authoring_throws(const pie_loader::SourceTensor& raw, const pie_loader::SourceTensor& scales,
                      const pie_loader::SourceTensor& biases) {
    pie_loader::ModelContract contract;
    try {
        pie::metal::model::contract_detail::push_mlx_affine_declared(contract, raw, scales, biases, 4, 0, "w");
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

// A triplet the schema cannot read has to fail here, where the message names the
// tensor. The alternative is a contract that type-checks against a group size
// the quantizer never used, and a plan that loads garbage without complaint.
void test_a_malformed_triplet_is_refused() {
    const std::vector<std::int64_t> packed{8, 512};       // 4096 logical columns
    const std::vector<std::int64_t> groups{8, 64};
    const std::vector<std::int64_t> other_groups{8, 32};
    const std::vector<std::int64_t> rank3{8, 64, 1};
    const std::vector<std::int64_t> wrong_rows{4, 64};
    const std::vector<std::int64_t> indivisible{8, 48};   // 4096 % 48 != 0
    const auto bf16 = pie_loader::raw(pie_loader::PieLoaderDType::BF16);
    const auto u32 = pie_loader::raw(pie_loader::PieLoaderDType::U32);

    check(authoring_throws(tensor_of("w.weight", packed, u32), tensor_of("w.scales", groups, bf16),
                           tensor_of("w.biases", other_groups, bf16)),
          "scales and biases must agree");
    check(authoring_throws(tensor_of("w.weight", packed, u32), tensor_of("w.scales", rank3, bf16),
                           tensor_of("w.biases", rank3, bf16)),
          "a rank-3 scale tensor is refused");
    check(authoring_throws(tensor_of("w.weight", packed, u32),
                           tensor_of("w.scales", wrong_rows, bf16),
                           tensor_of("w.biases", wrong_rows, bf16)),
          "scales must have one row per weight row");
    check(authoring_throws(tensor_of("w.weight", packed, u32),
                           tensor_of("w.scales", indivisible, bf16),
                           tensor_of("w.biases", indivisible, bf16)),
          "a group count that does not divide the columns is refused");
}

// -- the naming table --------------------------------------------------------

// Every checkpoint name is mapped or explicitly skipped, and an unrecognised one
// throws rather than passing through: a tensor declared under its checkpoint
// name would be bound by nothing and reported by nothing.
void test_every_name_is_mapped_or_refused() {
    namespace detail = pie::metal::model::qwen3_5::contract_detail;
    const auto mapped = [](std::string_view from, std::string_view to) {
        const auto got = detail::runtime_name(from);
        check(got.has_value() && *got == to,
              std::string(from) + " -> " + std::string(to));
    };
    mapped("lm_head.weight", "shared_embedding.weight");
    // The untied member of the same family, spelled by the repack. 0.8B ships
    // tied and has no `lm_head`; 35B-A3B ships untied and has one, under the
    // wrapper. Authoring stopped on this name.
    mapped("language_model.lm_head.weight", "shared_embedding.weight");
    mapped("language_model.lm_head.biases", "shared_embedding.biases");
    // ── and the same two names when the head is NOT the table ──
    // A tied checkpoint has one tensor and one slot, so both ends land on
    // `shared_embedding`. An untied one has two, and mapping them both onto
    // that slot declares it twice -- which is how this was found: every routed
    // member of the family is untied, and the load stopped with a duplicate.
    // `LmHeadUntied` and `EmbedUntied` bind these names.
    {
        const detail::HeadTying untied{false};
        const auto mapped_untied = [&untied](std::string_view from, std::string_view to) {
            const auto got = detail::runtime_name(from, untied);
            check(got.has_value() && *got == to,
                  std::string("untied: ") + std::string(from) + " -> " + std::string(to));
        };
        mapped_untied("lm_head.weight", "lm_head.weight");
        mapped_untied("language_model.lm_head.scales", "lm_head.scales");
        mapped_untied("model.language_model.embed_tokens.weight", "embed_tokens.weight");
        mapped_untied("language_model.model.embed_tokens.biases", "embed_tokens.biases");
        // Everything else is indifferent to tying, and must stay so.
        mapped_untied("model.language_model.norm.weight", "final_norm.weight");
    }
    // The tied half: this family ships tied, so its checkpoint has an
    // `embed_tokens` and no `lm_head`, and both land in the one shared slot
    // `EmbedGather` and `QmvLmHead` bind.
    mapped("model.language_model.embed_tokens.weight", "shared_embedding.weight");
    mapped("model.language_model.embed_tokens.scales", "shared_embedding.scales");
    mapped("model.language_model.norm.weight", "final_norm.weight");
    mapped("model.language_model.layers.31.self_attn.q_proj.weight",
           "layers.31.self_attn.q_proj.weight");
    mapped("model.language_model.layers.0.mlp.gate.scales", "layers.0.mlp.gate.scales");

    check(!detail::runtime_name("model.visual.blocks.0.attn.qkv.weight").has_value(),
          "the vision tower is skipped");
    check(!detail::runtime_name("mtp.layers.0.weight").has_value(),
          "the MTP head is skipped");

    // The same checkpoint, spelled by mlx_lm's repack rather than by the HF
    // release: the wrapper and the language tower swap places. Both are real
    // and both are this family, and the driver read only the first -- so
    // authoring against an actual mlx Qwen3.5 stopped on its first tensor.
    // These are the same six names as above with the prefix swapped, which is
    // the point: only the prefix may differ.
    mapped("language_model.model.embed_tokens.weight", "shared_embedding.weight");
    mapped("language_model.model.embed_tokens.scales", "shared_embedding.scales");
    mapped("language_model.model.norm.weight", "final_norm.weight");
    mapped("language_model.model.layers.31.self_attn.q_proj.weight",
           "layers.31.self_attn.q_proj.weight");
    mapped("language_model.model.layers.0.linear_attn.in_proj_qkv.weight",
           "layers.0.linear_attn.in_proj_qkv.weight");
    mapped("language_model.model.layers.3.mlp.switch_mlp.up_proj.weight",
           "layers.3.mlp.experts.up_proj.weight");
    check(!detail::runtime_name("vision_tower.blocks.0.attn.qkv.weight").has_value(),
          "and its vision tower is skipped under that spelling too");

    const auto refused = [](std::string_view name) {
        try {
            (void)detail::runtime_name(name);
        } catch (const std::exception&) {
            return true;
        }
        return false;
    };
    // The mixture. `mlx_lm` wraps a routed FFN in a `SwitchGLU`, so a
    // Qwen3-Next MoE checkpoint spells its stacked banks `mlp.switch_mlp.*`;
    // the driver's dispatches read `mlp.experts.*`. Until the family learned
    // this, the pass-through mapped `switch_mlp` onto itself: every expert was
    // declared, loaded into the heap, and read by nothing, and the router --
    // whose `mlp.gate` DID map -- routed tokens to weights the driver never
    // bound. Note that `mlp.gate` is asserted above and `mlp.gate_proj` is a
    // different tensor; the two differ by a suffix and one is a router.
    mapped("model.language_model.layers.3.mlp.switch_mlp.gate_proj.weight",
           "layers.3.mlp.experts.gate_proj.weight");
    mapped("model.language_model.layers.3.mlp.switch_mlp.down_proj.scales",
           "layers.3.mlp.experts.down_proj.scales");
    // And the fused export's spelling is already the driver's, so it passes
    // through unchanged rather than being rewritten twice.
    mapped("model.language_model.layers.3.mlp.experts.up_proj.weight",
           "layers.3.mlp.experts.up_proj.weight");

    check(refused("model.embed_tokens.weight"), "an unmapped name is an error, not a pass-through");
    check(refused("model.language_model.layers.x.weight"), "a non-numeric layer index is an error");
    check(refused("model.language_model.layers.7"), "a layer tensor with no suffix is an error");
    // A mixture this driver cannot run, refused at load with the tensor's name
    // on it rather than loaded and ignored: a stock HF checkpoint ships the
    // experts unstacked, and binding those would need a load-time gather this
    // driver does not do -- guessing gives every token expert 0, which is
    // fluent.
    check(refused("model.language_model.layers.3.mlp.experts.0.gate_proj.weight"),
          "an unstacked expert bank is refused, not silently bound to expert 0");
    // The shared expert, which this driver now DISPATCHES. Its four tensors
    // pass through under their own names -- `weights_for_kind` asks for exactly
    // these -- and the assertion is that they are mapped and not merely
    // tolerated, because a name that reaches the heap under a spelling no
    // dispatch reads is the failure the refusal used to prevent.
    for (const char* n : {"mlp.shared_expert.gate_proj.weight",
                          "mlp.shared_expert.up_proj.scales",
                          "mlp.shared_expert.down_proj.biases",
                          "mlp.shared_expert_gate.weight"}) {
        mapped(std::string("model.language_model.layers.3.") + n,
               std::string("layers.3.") + n);
    }
    // The PLURAL spelling is still refused. `mlp.shared_experts.` is DeepSeek's,
    // where several shared experts are stacked on an axis this block has no
    // index for; taking the first would be one expert of several, silently.
    check(refused("model.language_model.layers.3.mlp.shared_experts.gate_proj.weight"),
          "a STACK of shared experts is still refused");
}

void test_the_schema_owns_its_model_types() {
    namespace model = pie::metal::model;
    for (std::string_view yes : {"qwen3_5", "qwen3_5_text", "qwen3_next", "qwen3_next_text",
                                 "qwen3_6"}) {
        check(model::is_supported_model_type(yes), std::string(yes) + " is this schema's");
    }
    for (std::string_view no : {"qwen3_5_vision", ""}) {
        check(!model::is_supported_model_type(no), std::string(no) + " is nobody's");
    }
    // `llama` and `qwen3` ARE supported -- by the llama schema, not this one.
    // Which is the point of asking: one table answers "whose", so a name that
    // two schemas both claimed would be a routing bug rather than a coin toss.
    for (std::string_view ll : {"llama", "llama3", "mistral", "qwen2", "qwen3", "qwen3_moe"}) {
        check(model::model_family_of(ll) == model::ModelFamily::Llama,
              std::string(ll) + " routes to the llama schema");
    }
    check(model::model_family_of("qwen3_5") == model::ModelFamily::Qwen35,
          "qwen3_5 still routes to this one");

    // The refusal has to happen before anything is authored, or a caller gets a
    // half-built contract and a message about a missing tensor.
    pie_loader::Checkpoint empty;
    pie_loader::ModelContract contract;
    bool threw = false;
    try {
        model::author_model_contract(empty, "gpt2", pie::metal::metal_device_target(), contract);
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "an unsupported model_type is refused");
    check(contract.view().tensors.len == 0, "nothing is authored before the refusal");
}

// -- the whole schema, against a real checkpoint -----------------------------

/// Everything checkable about a contract without running the compiler.
void check_well_formed(const pie_loader::Checkpoint& checkpoint,
                       const pie_loader::PieLoaderModelContractView& v) {
    check(v.tensors.len > 0, "the contract declares at least one tensor");
    check(v.alignment == pie::metal::kMetalPreferredAlignment,
          "the contract carries this device's alignment");

    std::unordered_set<std::string_view> declared;
    for (std::size_t i = 0; i < v.tensors.len; ++i) {
        const auto& t = v.tensors.ptr[i];
        check(t.root < v.nodes.len, "every root is in the node arena");
        check(declared.insert(view_of(t.name)).second, "no tensor is declared twice");
    }

    std::size_t sources = 0;
    for (std::size_t i = 0; i < v.nodes.len; ++i) {
        const auto& n = v.nodes.ptr[i];
        // Topological order: an operand is always an earlier node, which makes
        // the graph acyclic by construction rather than by a check.
        if (n.src != pie_loader::PIE_LOADER_NO_NODE) {
            check(n.src < i, "an operand precedes its use");
        }
        for (std::size_t p = 0; p < n.parts.len; ++p) {
            check(n.parts.ptr[p] < i, "every concat part precedes the concat");
        }
        if (kind_of(n) == pie_loader::PieLoaderExprKind::Src) {
            ++sources;
            check(checkpoint.has(view_of(n.name)),
                  std::string("Src '") + std::string(view_of(n.name)) +
                      "' names a tensor the checkpoint has");
        }
        // This driver renames, reinterprets, and narrows a float that is not
        // already BF16; it shards nothing and packs nothing. Anything else
        // appearing here is a schema that grew a transform the Metal binder has
        // no kernel for.
        check(kind_of(n) == pie_loader::PieLoaderExprKind::Src ||
                  kind_of(n) == pie_loader::PieLoaderExprKind::Transmute ||
                  kind_of(n) == pie_loader::PieLoaderExprKind::Cast,
              "the Metal schema authors only Src, Transmute and Cast");
    }
    check(sources > 0, "the contract reads the checkpoint");
}

/// Author the snapshot's schema and push it through the real compiler.
void author_real_contract(const std::string& snapshot, const std::string& model_type) {
    std::string open_error;
    pie_loader::Checkpoint checkpoint = pie_loader::Checkpoint::open(snapshot, &open_error);
    check(static_cast<bool>(checkpoint), "the snapshot opens: " + open_error);
    if (!checkpoint) return;

    const auto target = pie::metal::metal_device_target();
    pie_loader::ModelContract contract;
    try {
        pie::metal::model::author_model_contract(checkpoint, model_type, target, contract);
    } catch (const std::exception& error) {
        check(false, std::string("authoring: ") + error.what());
        return;
    }
    const auto v = contract.view();
    check_well_formed(checkpoint, v);

    // Compiling is what turns "the names line up" into "the load is buildable":
    // `verify` shares no code with the compiler and stats every file the plan
    // declares.
    const auto request = pie_loader::build_contract_request(checkpoint, target, v);
    try {
        const pie_loader::LoadPlan plan = pie_loader::LoadPlan::compile(request);
        plan.verify(request);
        std::cout << "authored " << v.tensors.len << " tensor(s) for " << model_type
                  << "; plan has " << plan.view().schedule.len << " instruction(s)\n";
    } catch (const std::exception& error) {
        check(false, std::string("compiling: ") + error.what());
    }
}

}  // namespace

int main() {
    test_affine_u4_authors_the_quant_spec_the_kernels_need();
    test_the_derived_group_size_is_what_heap_bind_demands();
    test_the_declared_width_is_checked_against_the_shapes();
    test_a_malformed_triplet_is_refused();
    test_every_name_is_mapped_or_refused();
    test_the_schema_owns_its_model_types();

    const char* snapshot = std::getenv("PIE_TEST_SNAPSHOT");
    if (snapshot != nullptr && *snapshot != '\0') {
        // Read from the checkpoint, not defaulted. This used to fall back to
        // "qwen3_5", so pointing it at a Llama snapshot ran the WRONG schema and
        // reported "Qwen3.5 schema has no mapping for model.embed_tokens.biases"
        // -- a true sentence about a question nobody asked. `PIE_TEST_MODEL_TYPE`
        // still overrides, for the case where the point is to run a schema
        // against a checkpoint it does not claim.
        const char* forced = std::getenv("PIE_TEST_MODEL_TYPE");
        std::string model_type = forced != nullptr ? forced
                                                   : pie::metal::read_model_facts(snapshot).model_type;
        if (model_type.empty()) {
            check(false, "the snapshot's config.json names no model_type");
        } else {
            author_real_contract(snapshot, model_type);
        }
    }

    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "model_contract_test: OK\n";
    return 0;
}

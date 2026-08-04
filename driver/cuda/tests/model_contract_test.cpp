// The contract this driver authors, and the invariants the authoring DSL owes.
//
// Since §12 row 12 the contract is not a *claim about* the load — it is the
// load's input. `plan = compile(source_facts, program, target)`, and this file
// covers the `program`: the `author_contract` hook on this family's arch-table
// row reads the checkpoint's tensor table plus the config facts the driver
// already parsed and writes down every tensor it will bind, as an expression
// over what is on disk.
//
// Two things are checked here. First the builder — that a handle names one node
// however many times it is used, that a shape may be declined, and that the
// borrowed view stays valid as the contract grows past any initial capacity.
// Second, when `PIE_TEST_SNAPSHOT` points at a real checkpoint, that the real
// family authors a real contract: every `Src` names a tensor that exists, every
// declaration is unique, and the node arena is topologically sorted.
//
// The last of those is the one that used to be impossible. The old test could
// only compare a covered count against a demanded one, and both were names for
// `view.tensors.len`; it could not evaluate the contract because the contract
// was a list of names with no expressions in it.
//
// Under `PIE_TEST_CONTRACT_DUMP` the authored contract *and the plan it
// compiles to* are written out as JSON, at every rank of every TP size. That
// dump is what makes an authoring change checkable: a contract may legitimately
// be rewritten — `Shard(Slice(..))` and a `Slice` with the rank folded into its
// offset are different graphs — while the plan must not move at all.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <iostream>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

#include "pie_loader.h"
#include "pie_loader/model_contract.hpp"
#include "pie_loader/plan.hpp"
#include "pie_loader/request.hpp"
#include "pie_loader/source_checkpoint.hpp"

#include "loader/load_plan_bridge.hpp"
#include "model/contract.hpp"
#include "model/csm/csm_contract.hpp"
#include "model/deepseek_v4/deepseek_v4_contract.hpp"
#include "model/glm5/glm5_contract.hpp"
#include "model/mixtral/mixtral_contract.hpp"
#include "model/kimi/kimi_contract.hpp"
#include "model/mixtral/mixtral_contract.hpp"
#include "model/nemotron_h/nemotron_h_contract.hpp"
#include "model/qwen3_5/qwen3_5_contract.hpp"
#include "model/registry.hpp"

#include "loader/load_plan.hpp"
#include "model/config.hpp"

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

// -- the builder itself ------------------------------------------------------

void test_a_handle_names_one_node() {
    pie_loader::ModelContract c;
    auto q = c.src("q");
    // Naming `q` three times must build one `Src`, not three: a `concat` of three
    // slices of one tensor has to read that tensor once, and the compiler
    // decides that by node identity.
    c.define("qkv", c.concat({q, q, q}, 0), pie_loader::raw(pie_loader::PieLoaderDType::BF16));

    const auto v = c.view();
    std::size_t srcs = 0;
    for (std::size_t i = 0; i < v.nodes.len; ++i) {
        if (v.nodes.ptr[i].kind ==
            static_cast<std::uint32_t>(pie_loader::PieLoaderExprKind::Src)) {
            ++srcs;
        }
    }
    check(srcs == 1, "one Src node for a handle used three times");
    check(v.tensors.len == 1, "one declaration");
    check(view_of(v.tensors.ptr[0].name) == "qkv", "declaration name");
    check(v.tensors.ptr[0].shape.len == 0, "a declaration without expect states no shape");
}

void test_expect_states_a_shape() {
    pie_loader::ModelContract c;
    c.define("w", c.src("w"), pie_loader::raw(pie_loader::PieLoaderDType::BF16))
        .expect({4, 8});
    const auto v = c.view();
    check(v.tensors.len == 1 && v.tensors.ptr[0].shape.len == 2, "expect states a rank-2 shape");
    check(v.tensors.ptr[0].shape.ptr[1] == 8, "the stated extents come back");
}

// A group is `arity` interchangeable declarations. What the builder owes is
// that they land in storage of their own -- a `define` on a group must not
// reach the contract's tensor list -- and that `expect` and `internal` still
// address the declaration they were chained onto once there are two lists they
// could mean.
void test_a_group_declares_its_own_tensors() {
    pie_loader::ModelContract c;
    c.define("norm", c.src("norm"), pie_loader::raw(pie_loader::PieLoaderDType::BF16));

    auto experts = c.group("experts", 128);
    auto bank = c.src("experts.gate_up");
    experts.define("gate_up", c.select(bank, 0, 1, 1),
                   pie_loader::raw(pie_loader::PieLoaderDType::BF16))
        .expect({1, 64, 128});
    experts.define("down", c.index_src("experts.{}.down"),
                   pie_loader::raw(pie_loader::PieLoaderDType::BF16))
        .internal();

    const auto v = c.view();
    check(v.tensors.len == 1, "a group's declarations stay out of the contract's");
    check(v.groups.len == 1, "one group");
    check(view_of(v.groups.ptr[0].name) == "experts", "group name");
    check(v.groups.ptr[0].arity == 128, "group arity");
    check(v.groups.ptr[0].tensors.len == 2, "two declarations in the group");
    check(v.groups.ptr[0].tensors.ptr[0].shape.len == 3,
          "expect on a group's declaration reaches that declaration");
    check(v.groups.ptr[0].tensors.ptr[1].visibility ==
              pie_loader::PieLoaderVisibility::Internal,
          "internal on a group's declaration reaches that declaration");
    // The index nodes share the one node pool, as every other node does.
    std::size_t indexed = 0;
    for (std::size_t i = 0; i < v.nodes.len; ++i) {
        const std::uint32_t kind = v.nodes.ptr[i].kind;
        if (kind == static_cast<std::uint32_t>(pie_loader::PieLoaderExprKind::SrcIndexed) ||
            kind == static_cast<std::uint32_t>(pie_loader::PieLoaderExprKind::Select)) {
            ++indexed;
        }
    }
    check(indexed == 2, "both index nodes are in the shared pool");
}

// A group whose declarations were added after the view was taken must not be
// read through the stale view -- but a view taken *after* must see them, which
// is the growth rule the contract's own tensors already have.
void test_a_group_view_survives_growth() {
    pie_loader::ModelContract c;
    for (int i = 0; i < 64; ++i) {
        auto g = c.group("g" + std::to_string(i), 4);
        g.define("w", c.index_src("t.{}.w"), pie_loader::raw(pie_loader::PieLoaderDType::BF16));
    }
    const auto v = c.view();
    check(v.groups.len == 64, "every group is in the view");
    for (std::size_t i = 0; i < v.groups.len; ++i) {
        check(v.groups.ptr[i].tensors.len == 1, "each group kept its declaration");
        check(view_of(v.groups.ptr[i].name) == "g" + std::to_string(i),
              "each group kept its name");
    }
}

// A scale tensor states which weight it scales. The name it states is stored
// by the builder, not borrowed from the caller, so it has to outlive the
// argument it came from.
void test_scaling_states_the_weight() {
    pie_loader::ModelContract c;
    c.define("w", c.src("w"), pie_loader::raw(pie_loader::PieLoaderDType::F8E4M3))
        .expect({256, 512});
    {
        std::string weight = "w";
        c.define("w_scale_inv", c.src("w_scale_inv"),
                 pie_loader::raw(pie_loader::PieLoaderDType::F32))
            .expect({2, 4})
            .scaling(weight, pie_loader::PieLoaderQuantGranularity::PerGroup, 128, 0,
                     pie_loader::PieLoaderScaleForm::F32Factors);
    }
    const auto v = c.view();
    check(v.tensors.len == 2, "two declarations");
    check(view_of(v.tensors.ptr[0].scales.of).empty(), "a weight states no scales of its own");
    const auto& sc = v.tensors.ptr[1].scales;
    check(view_of(sc.of) == "w", "the scale states the weight it belongs to");
    check(sc.group_size == 128 && sc.channel_axis == 0, "the stated block size comes back");
    check(sc.granularity ==
                  static_cast<std::uint32_t>(pie_loader::PieLoaderQuantGranularity::PerGroup) &&
              sc.form == static_cast<std::uint32_t>(pie_loader::PieLoaderScaleForm::F32Factors),
          "the stated granularity and form come back");
}

// The one node that changes a value. The factor crosses the boundary as a bit
// pattern, so what this checks is that `scale()` produced the bits of the float
// it was handed and nothing reinterpreted them on the way.
void test_scale_carries_its_factor() {
    pie_loader::ModelContract c;
    const float factor = 1.f / std::sqrt(2048.f);
    c.define("router.scale", c.scale(c.src("router.scale"), factor),
             pie_loader::raw(pie_loader::PieLoaderDType::BF16))
        .expect({2048});
    const auto v = c.view();
    check(v.tensors.len == 1, "one declaration");
    const auto& node = v.nodes.ptr[v.tensors.ptr[0].root];
    check(node.kind == static_cast<std::uint32_t>(pie_loader::PieLoaderExprKind::Scale),
          "the node is a Scale");
    float round_tripped = 0.f;
    std::memcpy(&round_tripped, &node.scale_factor_bits, sizeof(round_tripped));
    check(round_tripped == factor, "the factor comes back bit for bit");
    check(v.nodes.ptr[node.src].kind ==
              static_cast<std::uint32_t>(pie_loader::PieLoaderExprKind::Src),
          "the operand is the source it was given");
}

// The reason the builder stores names, shapes and nodes in deques. With a
// vector, growing past the initial capacity would move everything and leave the
// already-borrowed view pointing at freed memory — a use-after-free the loader
// would dereference, and one that only appears above some tensor count.
void test_the_view_survives_growth() {
    pie_loader::ModelContract c;
    std::vector<std::string> expected;
    for (int i = 0; i < 512; ++i) {
        expected.push_back("model.layers." + std::to_string(i) + ".weight");
        c.define(expected.back(), c.src(expected.back()),
                 pie_loader::raw(pie_loader::PieLoaderDType::BF16))
            .expect({static_cast<std::int64_t>(i + 1)});
    }
    const auto v = c.view();
    check(v.tensors.len == expected.size(), "every declaration is present after growth");
    bool intact = true;
    for (std::size_t i = 0; i < v.tensors.len; ++i) {
        const auto& t = v.tensors.ptr[i];
        if (view_of(t.name) != expected[i] || t.shape.len != 1 ||
            t.shape.ptr[0] != static_cast<std::int64_t>(i + 1) ||
            view_of(v.nodes.ptr[t.root].name) != expected[i]) {
            intact = false;
        }
    }
    check(intact, "every name, shape and node still readable after growth");
}

// -- the real declaration ----------------------------------------------------

/// One entry for `write_typed_checkpoint`: a name, a shape, and the safetensors
/// dtype string the header should carry.
struct TypedTensor {
    std::string name;
    std::vector<std::int64_t> shape;
    std::string dtype;
};

/// Flush and close, and say so when the bytes did not land.
///
/// These fixtures are written to `$TMPDIR`, which on a shared box is a tmpfs
/// several users are competing for. A stream that runs out of room drops its
/// whole buffer silently, leaving a zero-byte file, and the test then fails
/// several frames later reading a header that was never written. Naming the
/// real cause here is the difference between a two-minute diagnosis and an
/// afternoon spent suspecting the contract.
void close_or_throw(std::ofstream& out, const std::filesystem::path& path) {
    out.close();
    if (!out) {
        throw std::runtime_error("could not write the fixture " + path.string() +
                                 " (is the filesystem holding $TMPDIR full?)");
    }
}

/// Write a safetensors file whose tensors do *not* all share a dtype.
///
/// A packed-MXFP4 checkpoint is the case that needs this: the weight is `I8`
/// and its block scales are `U8`, and a pass that reads one as the other is
/// exactly the mistake worth pinning.
std::filesystem::path write_typed_checkpoint(const std::string& tag,
                                             const std::vector<TypedTensor>& tensors) {
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / ("pie_contract_test_" + tag);
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);

    nlohmann::json header = nlohmann::json::object();
    std::uint64_t offset = 0;
    for (const TypedTensor& t : tensors) {
        std::uint64_t bytes =
            (t.dtype == "F32" || t.dtype == "I32") ? 4 : (t.dtype == "BF16" ? 2 : 1);
        for (std::int64_t extent : t.shape) bytes *= static_cast<std::uint64_t>(extent);
        header[t.name] = {{"dtype", t.dtype},
                          {"shape", t.shape},
                          {"data_offsets", {offset, offset + bytes}}};
        offset += bytes;
    }
    const std::string text = header.dump();
    std::ofstream out(dir / "model.safetensors", std::ios::binary);
    const std::uint64_t len = text.size();
    out.write(reinterpret_cast<const char*>(&len), sizeof(len));
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
    const std::vector<char> zeros(static_cast<std::size_t>(offset), 0);
    out.write(zeros.data(), static_cast<std::streamsize>(zeros.size()));
    close_or_throw(out, dir / "model.safetensors");
    return dir;
}

/// Write a safetensors file holding `tensors`, zero-filled.
///
/// The container is 8 bytes of little-endian header length, that many bytes of
/// JSON, then the data. Small enough to write by hand, which is the point: a
/// pass that reshards a fused tensor is checkable against a checkpoint whose
/// only real content is its *shapes*.
std::filesystem::path write_synthetic_checkpoint(
    const std::string& tag,
    const std::vector<std::pair<std::string, std::vector<std::int64_t>>>& tensors,
    const std::string& dtype = "BF16") {
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / ("pie_contract_test_" + tag);
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);

    nlohmann::json header = nlohmann::json::object();
    std::uint64_t offset = 0;
    for (const auto& [name, shape] : tensors) {
        std::uint64_t elems = dtype == "F32" ? 4 : 2;
        for (std::int64_t extent : shape) elems *= static_cast<std::uint64_t>(extent);
        header[name] = {{"dtype", dtype},
                        {"shape", shape},
                        {"data_offsets", {offset, offset + elems}}};
        offset += elems;
    }
    const std::string text = header.dump();
    std::ofstream out(dir / "model.safetensors", std::ios::binary);
    const std::uint64_t len = text.size();
    out.write(reinterpret_cast<const char*>(&len), sizeof(len));
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
    const std::vector<char> zeros(static_cast<std::size_t>(offset), 0);
    out.write(zeros.data(), static_cast<std::streamsize>(zeros.size()));
    close_or_throw(out, dir / "model.safetensors");
    return dir;
}

/// True when the compiled plan declares `name` as a name for the contract's own
/// use rather than a weight the driver will bind.
///
/// The declaration is still there -- the dequantizing multiply reads its dtype
/// and shape -- so its absence from the plan is not what to check for.
inline bool planned_internal(const pie_loader::LoadPlan& plan, std::string_view name) {
    const auto& v = plan.view();
    for (std::size_t i = 0; i < v.tensors.len; ++i) {
        const auto& t = v.tensors.ptr[i];
        if (view_of(t.name) == name) {
            return t.visibility == pie_loader::PieLoaderVisibility::Internal;
        }
    }
    return false;
}

/// DeepSeek-V4 dequantizes and stacks its routed experts in the contract.
///
/// The checkpoint stores each expert separately as packed MXFP4 -- an `I8`
/// tensor of two E2M1 nibbles per byte, beside `U8` E8M0 block scales -- and
/// the batched expert GEMM wants one dense bf16 slab per layer. That used to
/// be a bind-time loop calling `launch_dequant_mxfp4_to_bf16` 3E times per
/// layer into slabs outside the storage arena, leaving the packed originals
/// resident because an arena view cannot be freed.
///
/// What this pins is that the algebra states it instead, and states it *under*
/// tensor parallelism: the scale node's operand carries the shard, so each
/// rank dequantizes only the slice it keeps, and the declared slab shapes are
/// this rank's share rather than the whole expert.
constexpr std::int64_t kDsV4Experts = 2;
constexpr std::int64_t kDsV4Hidden = 64;
constexpr std::int64_t kDsV4InterFull = 64;
constexpr std::int64_t kDsV4Group = 32;

/// One layer of packed-MXFP4 routed experts, as DeepSeek-V4 ships them.
std::filesystem::path dsv4_mxfp4_checkpoint(const char* name) {
    std::vector<TypedTensor> tensors{
        {"model.embed_tokens.weight", {16, kDsV4Hidden}, "BF16"},
        {"model.norm.weight", {kDsV4Hidden}, "BF16"},
        {"lm_head.weight", {16, kDsV4Hidden}, "BF16"},
        {"model.layers.0.ffn.gate.weight", {kDsV4Experts, kDsV4Hidden}, "BF16"},
    };
    const std::string ffn = "model.layers.0.ffn.";
    for (int e = 0; e < kDsV4Experts; ++e) {
        const std::string ep = ffn + "experts." + std::to_string(e) + ".";
        // `w1`/`w3` are `[I_full, H/2]`; `w2` is `[H, I_full/2]`. Halved
        // because two 4-bit elements share a stored byte.
        for (const char* gate_up : {"w1", "w3"}) {
            tensors.push_back(
                {ep + gate_up + ".weight", {kDsV4InterFull, kDsV4Hidden / 2}, "I8"});
            tensors.push_back(
                {ep + gate_up + ".scale", {kDsV4InterFull, kDsV4Hidden / kDsV4Group}, "U8"});
        }
        tensors.push_back({ep + "w2.weight", {kDsV4Hidden, kDsV4InterFull / 2}, "I8"});
        tensors.push_back(
            {ep + "w2.scale", {kDsV4Hidden, kDsV4InterFull / kDsV4Group}, "U8"});
    }
    return write_typed_checkpoint(name, tensors);
}

void test_deepseek_v4_stacks_experts_as_bf16() {
    constexpr std::int64_t kExperts = kDsV4Experts;
    constexpr std::int64_t kHidden = kDsV4Hidden;
    constexpr std::int64_t kInterFull = kDsV4InterFull;
    constexpr std::int64_t kGroup = kDsV4Group;

    const std::string ffn = "model.layers.0.ffn.";
    const std::filesystem::path dir = dsv4_mxfp4_checkpoint("dsv4_mxfp4_experts");

    std::string open_error;
    pie_loader::Checkpoint checkpoint = pie_loader::Checkpoint::open(dir.string(), &open_error);
    check(static_cast<bool>(checkpoint), "the synthetic checkpoint opens: " + open_error);
    if (!checkpoint) return;

    const pie_cuda_driver::model::ModelFacts facts{
        .model_type = "deepseek_v4",
        .quant_method = "",
        .num_hidden_layers = 1,
        .num_experts = kExperts,
        .head_dim = 0,
        .mamba_groups = 0,
    };
    namespace model = pie_cuda_driver::model;

    for (int tp : {1, 2}) {
        for (int rank = 0; rank < tp; ++rank) {
            const std::int64_t inter = kInterFull / tp;
            const std::string at =
                " (tp " + std::to_string(tp) + " rank " + std::to_string(rank) + ")";

            auto target = pie_cuda_driver::cuda_device_target();
            target.tp_size = static_cast<std::uint32_t>(tp);
            target.tp_rank = static_cast<std::uint32_t>(rank);

            pie_loader::ModelContract contract;
            try {
                model::ContractBuilder builder(checkpoint, facts, target, "",
                                               model::Mxfp4MoeRequest::Auto,
                                               model::Component::Full, /*stream_routed_experts=*/false, contract);
                model::author_deepseek_v4_contract(builder);
                builder.finish();
                // The family answers for itself: there is no native MXFP4 GEMM
                // to fall back from, so `Auto` is eager whatever the device
                // reports.
                check(builder.mxfp4_moe() == model::Mxfp4MoePolicy::EagerBf16,
                      "Auto resolves to eager for this family" + at);
            } catch (const std::exception& error) {
                check(false, "authoring" + at + ": " + error.what());
                continue;
            }

            std::map<std::string_view, std::vector<std::int64_t>> declared;
            std::map<std::string_view, std::uint32_t> dtypes;
            const auto v = contract.view();
            for (std::size_t i = 0; i < v.tensors.len; ++i) {
                const auto& t = v.tensors.ptr[i];
                declared[view_of(t.name)] =
                    std::vector<std::int64_t>(t.shape.ptr, t.shape.ptr + t.shape.len);
                dtypes[view_of(t.name)] = t.encoding.dtype;
            }
            const auto declares = [&](const std::string& name,
                                      const std::vector<std::int64_t>& want,
                                      pie_loader::PieLoaderDType dtype) {
                const auto found = declared.find(name);
                check(found != declared.end() && found->second == want &&
                          dtypes[name] == static_cast<std::uint32_t>(dtype),
                      "'" + name + "' is this rank's share" + at);
            };
            declares(ffn + "experts.gate_up.weight", {kExperts, 2 * inter, kHidden},
                     pie_loader::PieLoaderDType::BF16);
            declares(ffn + "experts.down.weight", {kExperts, kHidden, inter},
                     pie_loader::PieLoaderDType::BF16);
            declares(ffn + "experts.gate_up.scale", {kExperts, 2 * inter, kHidden / kGroup},
                     pie_loader::PieLoaderDType::E8M0);
            declares(ffn + "experts.down.scale", {kExperts, kHidden, inter / kGroup},
                     pie_loader::PieLoaderDType::E8M0);

            // The packed originals are the stacks' input, not a second copy of
            // the model: nothing should still publish them under their own
            // names.
            for (int e = 0; e < kExperts; ++e) {
                const std::string ep = ffn + "experts." + std::to_string(e) + ".";
                for (const char* part : {"w1", "w2", "w3"}) {
                    check(declared.find(ep + part + ".weight") == declared.end(),
                          "'" + ep + part + ".weight' was consumed by the stack" + at);
                }
            }

            // Authoring is only half of it: the shapes above are what the
            // contract *claims*, and only compilation checks the claim against
            // the expression that is supposed to produce it.
            try {
                const pie_loader::PieLoaderContractRequest request =
                    pie_loader::build_contract_request(checkpoint, target, v);
                const pie_loader::LoadPlan plan = pie_loader::LoadPlan::compile(request);
                plan.verify(request);
                // The stacked factors are a name, not a weight: the dequantize
                // that consumes them happens at load, so binding them would
                // pin bytes in the persistent arena that nothing reads.
                for (const char* half : {"gate_up", "down"}) {
                    const std::string name = ffn + "experts." + half + ".scale";
                    check(planned_internal(plan, name), "'" + name + "' is not bound" + at);
                }
            } catch (const std::exception& error) {
                check(false, "compiling" + at + ": " + error.what());
            }
        }
    }
}

/// DeepSeek-V4 declares its routed experts as a group when they are streamed.
///
/// The same expression, one rank lower. A stack says "every expert of this
/// layer, concatenated"; a group says "one expert of this layer", with the
/// index left standing -- so what this pins is that removing the outer concat
/// removed the *only* thing that was per-expert, and that the dequantization,
/// the shard and the scale pairing all survived the move.
///
/// Two facts matter more than the shapes. The group's declarations are not the
/// contract's, so nothing publishes a slab: the resident plan gets smaller by
/// exactly the experts. And the group's plan prices one expert, not E, which
/// is the number a driver sizes a slot against -- if that ever equalled the
/// stack's, streaming would be buying nothing.
///
/// The loader proves interchangeability by compiling every instance and
/// requiring them equal, so `LoadPlan::compile` succeeding here is not a
/// formality: it is the statement that these E experts really are one program
/// read at E offsets.
void test_deepseek_v4_streams_experts_as_a_group() {
    constexpr std::int64_t kExperts = kDsV4Experts;
    constexpr std::int64_t kHidden = kDsV4Hidden;
    constexpr std::int64_t kInterFull = kDsV4InterFull;
    constexpr std::int64_t kGroup = kDsV4Group;

    const std::string ffn = "model.layers.0.ffn.";
    const std::filesystem::path dir = dsv4_mxfp4_checkpoint("dsv4_mxfp4_streamed");

    std::string open_error;
    pie_loader::Checkpoint checkpoint = pie_loader::Checkpoint::open(dir.string(), &open_error);
    check(static_cast<bool>(checkpoint), "the synthetic checkpoint opens: " + open_error);
    if (!checkpoint) return;

    const pie_cuda_driver::model::ModelFacts facts{
        .model_type = "deepseek_v4",
        .quant_method = "",
        .num_hidden_layers = 1,
        .num_experts = kExperts,
        .head_dim = 0,
        .mamba_groups = 0,
    };
    namespace model = pie_cuda_driver::model;

    // Under TP too: a group and a shard index on different axes -- one picks
    // the tensor, the other the slice of it -- so a sharded group needs no
    // special care from either side, and this is where that is checked.
    for (int tp : {1, 2}) {
        for (int rank = 0; rank < tp; ++rank) {
            const std::int64_t inter = kInterFull / tp;
            const std::string at =
                " (tp " + std::to_string(tp) + " rank " + std::to_string(rank) + ")";

            auto target = pie_cuda_driver::cuda_device_target();
            target.tp_size = static_cast<std::uint32_t>(tp);
            target.tp_rank = static_cast<std::uint32_t>(rank);

            pie_loader::ModelContract contract;
            try {
                model::ContractBuilder builder(checkpoint, facts, target, "",
                                               model::Mxfp4MoeRequest::Auto,
                                               model::Component::Full,
                                               /*stream_routed_experts=*/true, contract);
                model::author_deepseek_v4_contract(builder);
                builder.finish();
            } catch (const std::exception& error) {
                check(false, "authoring" + at + ": " + error.what());
                continue;
            }

            const auto v = contract.view();
            check(v.groups.len == 1, "one group, for the one layer" + at);
            if (v.groups.len != 1) continue;
            const auto& g = v.groups.ptr[0];
            check(view_of(g.name) == ffn + "experts",
                  "the group is named for its layer" + at);
            check(g.arity == kExperts, "one instance per expert" + at);

            std::map<std::string_view, std::vector<std::int64_t>> declared;
            std::map<std::string_view, std::uint32_t> dtypes;
            std::map<std::string_view, bool> internal;
            for (std::size_t i = 0; i < g.tensors.len; ++i) {
                const auto& t = g.tensors.ptr[i];
                declared[view_of(t.name)] =
                    std::vector<std::int64_t>(t.shape.ptr, t.shape.ptr + t.shape.len);
                dtypes[view_of(t.name)] = t.encoding.dtype;
                internal[view_of(t.name)] =
                    t.visibility == pie_loader::PieLoaderVisibility::Internal;
            }
            const auto declares = [&](const std::string& name,
                                      const std::vector<std::int64_t>& want,
                                      pie_loader::PieLoaderDType dtype) {
                const auto found = declared.find(name);
                check(found != declared.end() && found->second == want &&
                          dtypes[name] == static_cast<std::uint32_t>(dtype),
                      "the group declares '" + name + "' as this rank's share" + at);
            };
            // No layer and no expert in the name: there is one plan, and it is
            // the same plan for every instance.
            declares("gate_up.weight", {2 * inter, kHidden}, pie_loader::PieLoaderDType::BF16);
            declares("down.weight", {kHidden, inter}, pie_loader::PieLoaderDType::BF16);
            declares("gate_up.scale", {2 * inter, kHidden / kGroup},
                     pie_loader::PieLoaderDType::E8M0);
            declares("down.scale", {kHidden, inter / kGroup},
                     pie_loader::PieLoaderDType::E8M0);
            check(internal["gate_up.scale"] && internal["down.scale"],
                  "the factors are consumed by the dequantize, not bound" + at);

            for (std::size_t i = 0; i < v.tensors.len; ++i) {
                const auto name = view_of(v.tensors.ptr[i].name);
                check(name.find("experts") == std::string_view::npos,
                      "no expert slab is published outside the group" + at);
            }

            try {
                const pie_loader::PieLoaderContractRequest request =
                    pie_loader::build_contract_request(checkpoint, target, v);
                // Compiling *is* the interchangeability proof: every instance
                // is compiled and required to equal instance 0 everywhere but
                // the three fields that locate bytes.
                const pie_loader::LoadPlan plan = pie_loader::LoadPlan::compile(request);
                plan.verify(request);

                const auto& pv = plan.view();
                check(pv.groups.len == 1, "the plan carries the group" + at);
                if (pv.groups.len != 1) continue;
                const auto& pg = pv.groups.ptr[0];
                check(pg.arity == kExperts, "the plan agrees on the arity" + at);
                check(pg.plan != nullptr, "the group carries a whole plan" + at);
                check(pg.bindings_per_instance > 0,
                      "an expert is read from somewhere" + at);
                check(pg.bindings.len ==
                          pg.bindings_per_instance * static_cast<std::size_t>(kExperts),
                      "every instance is bound" + at);

                // A slot holds one expert. Both halves of one expert, this
                // rank's share: gate_up is `[2I, H]` and down is `[H, I]`, two
                // bytes an element. Alignment may round it up, never down.
                const std::uint64_t one_expert = static_cast<std::uint64_t>(
                    (2 * inter * kHidden + kHidden * inter) * 2);
                check(pg.plan->memory.persistent_bytes >= one_expert &&
                          pg.plan->memory.persistent_bytes <
                              one_expert + 4096,
                      "a slot is priced at one expert, not " +
                          std::to_string(kExperts) + at);
                // And the resident plan no longer carries them at all.
                check(pv.memory.persistent_bytes < one_expert,
                      "the experts left the resident plan" + at);
            } catch (const std::exception& error) {
                check(false, "compiling" + at + ": " + error.what());
            }
        }
    }
}

/// The shared expert's gate and up projections are joined by the contract.
///
/// Additive, not a replacement: `try_fold_shared_into_routed` still reads the
/// two projections separately, and which path runs is a per-step decision, so
/// the sources must survive. Sharded on axis 0, so the join's row count has to
/// follow the rank rather than the checkpoint.
/// GLM-5.1 ships `kv_b_proj` as FP8 with a two-dimensional block scale, and
/// the kernels that read it want BF16.
///
/// This is the case that forced the scale factor's `group`/`axis` pair to be
/// replaced by the factors' own shape. A single group size along a single axis
/// can name a row-wise blocking and nothing else, but the kernel this driver
/// already ships (`dequant_fp8_e4m3_blocked_kernel`) indexes
/// `scale[row / row_block][col / col_block]` -- so the checkpoint's actual
/// layout was one the algebra could not spell.
///
/// What the test pins is that the FP8 pair leaves the checkpoint and one BF16
/// tensor arrives: `kv_b_proj.weight` is declared BF16 at this rank's row
/// count, and the factors are declared but not bound. `bind_glm5` used to
/// dequantize into a side buffer and keep the FP8 original resident, so both
/// halves of that are what changed.
void test_glm5_dequantizes_kv_b_proj() {
    constexpr std::int64_t kHidden = 64;
    constexpr std::int64_t kKvBRows = 128;
    constexpr std::int64_t kKvLora = 64;
    // A 64x32 block: deliberately not square and not the whole row, so a
    // reader that assumes either is wrong about this fixture.
    constexpr std::int64_t kScaleRows = 2;
    constexpr std::int64_t kScaleCols = 2;

    const std::string lp = "model.layers.0.";
    const std::string ap = lp + "self_attn.";
    std::vector<TypedTensor> tensors{
        {"model.embed_tokens.weight", {16, kHidden}, "BF16"},
        {"model.norm.weight", {kHidden}, "BF16"},
        {"lm_head.weight", {16, kHidden}, "BF16"},
        {lp + "input_layernorm.weight", {kHidden}, "BF16"},
        {lp + "post_attention_layernorm.weight", {kHidden}, "BF16"},
        {ap + "kv_b_proj.weight", {kKvBRows, kKvLora}, "F8_E4M3"},
        {ap + "kv_b_proj.weight_scale_inv", {kScaleRows, kScaleCols}, "F32"},
        {ap + "o_proj.weight", {kHidden, kHidden}, "BF16"},
        {lp + "mlp.gate_proj.weight", {kHidden, kHidden}, "BF16"},
        {lp + "mlp.up_proj.weight", {kHidden, kHidden}, "BF16"},
        {lp + "mlp.down_proj.weight", {kHidden, kHidden}, "BF16"},
    };

    const std::filesystem::path dir = write_typed_checkpoint("glm5_kv_b_fp8", tensors);
    std::string open_error;
    pie_loader::Checkpoint checkpoint = pie_loader::Checkpoint::open(dir.string(), &open_error);
    check(static_cast<bool>(checkpoint), "the synthetic checkpoint opens: " + open_error);
    if (!checkpoint) {
        std::filesystem::remove_all(dir);
        return;
    }

    const pie_cuda_driver::model::ModelFacts facts{
        .model_type = "glm5",
        .quant_method = "",
        .num_hidden_layers = 1,
        .num_experts = 0,
        .head_dim = 0,
        .mamba_groups = 0,
    };
    namespace model = pie_cuda_driver::model;

    for (int tp : {1, 2}) {
        for (int rank = 0; rank < tp; ++rank) {
            const std::string at =
                " (tp " + std::to_string(tp) + " rank " + std::to_string(rank) + ")";
            auto target = pie_cuda_driver::cuda_device_target();
            target.tp_size = static_cast<std::uint32_t>(tp);
            target.tp_rank = static_cast<std::uint32_t>(rank);

            pie_loader::ModelContract contract;
            try {
                model::ContractBuilder builder(checkpoint, facts, target, "",
                                               model::Mxfp4MoeRequest::Auto,
                                               model::Component::Full,
                                               /*stream_routed_experts=*/false, contract);
                model::author_glm5_contract(builder);
                builder.finish();
            } catch (const std::exception& error) {
                check(false, "authoring" + at + ": " + error.what());
                continue;
            }

            const auto v = contract.view();
            bool found = false;
            for (std::size_t i = 0; i < v.tensors.len; ++i) {
                const auto& t = v.tensors.ptr[i];
                if (view_of(t.name) != ap + "kv_b_proj.weight") continue;
                found = true;
                const std::vector<std::int64_t> shape(t.shape.ptr, t.shape.ptr + t.shape.len);
                check(shape == std::vector<std::int64_t>{kKvBRows / tp, kKvLora},
                      "kv_b_proj is this rank's rows" + at);
                check(t.encoding.dtype ==
                          static_cast<std::uint32_t>(pie_loader::PieLoaderDType::BF16),
                      "kv_b_proj reaches the bind dequantized" + at);
            }
            check(found, "kv_b_proj is published" + at);

            try {
                const pie_loader::PieLoaderContractRequest request =
                    pie_loader::build_contract_request(checkpoint, target, v);
                const pie_loader::LoadPlan plan = pie_loader::LoadPlan::compile(request);
                plan.verify(request);
                // The factors are consumed by the dequant, so binding them
                // would pin FP32 bytes nothing reads for the process's life.
                check(planned_internal(plan, ap + "kv_b_proj.weight_scale_inv"),
                      "the block scales are not bound" + at);
            } catch (const std::exception& error) {
                check(false, "compiling" + at + ": " + error.what());
            }
        }
    }
    std::filesystem::remove_all(dir);
}

void test_qwen3_5_moe_joins_the_shared_expert() {
    constexpr std::int64_t kExperts = 4;
    constexpr std::int64_t kHidden = 64;
    constexpr std::int64_t kInter = 32;
    constexpr std::int64_t kShared = 16;

    std::vector<TypedTensor> tensors{
        {"model.embed_tokens.weight", {16, kHidden}, "BF16"},
        {"model.norm.weight", {kHidden}, "BF16"},
        {"lm_head.weight", {16, kHidden}, "BF16"},
    };
    const std::string lp = "model.layers.0.";
    tensors.push_back({lp + "input_layernorm.weight", {kHidden}, "BF16"});
    tensors.push_back({lp + "post_attention_layernorm.weight", {kHidden}, "BF16"});
    for (const char* proj : {"q_proj", "k_proj", "v_proj", "o_proj"}) {
        tensors.push_back({lp + "self_attn." + proj + ".weight", {kHidden, kHidden}, "BF16"});
    }
    tensors.push_back({lp + "mlp.gate.weight", {kExperts, kHidden}, "BF16"});
    tensors.push_back({lp + "mlp.experts.gate_up_proj", {kExperts, 2 * kInter, kHidden}, "BF16"});
    tensors.push_back({lp + "mlp.experts.down_proj", {kExperts, kHidden, kInter}, "BF16"});
    tensors.push_back({lp + "mlp.shared_expert.gate_proj.weight", {kShared, kHidden}, "BF16"});
    tensors.push_back({lp + "mlp.shared_expert.up_proj.weight", {kShared, kHidden}, "BF16"});
    tensors.push_back({lp + "mlp.shared_expert.down_proj.weight", {kHidden, kShared}, "BF16"});
    tensors.push_back({lp + "mlp.shared_expert_gate.weight", {1, kHidden}, "BF16"});

    const std::filesystem::path dir =
        write_typed_checkpoint("qwen3_5_moe_shared", tensors);

    std::string open_error;
    pie_loader::Checkpoint checkpoint = pie_loader::Checkpoint::open(dir.string(), &open_error);
    check(static_cast<bool>(checkpoint), "the synthetic checkpoint opens: " + open_error);
    if (!checkpoint) {
        std::filesystem::remove_all(dir);
        return;
    }

    const pie_cuda_driver::model::ModelFacts facts{
        .model_type = "qwen3_5_moe",
        .quant_method = "",
        .num_hidden_layers = 1,
        .num_experts = kExperts,
        .head_dim = 0,
        .mamba_groups = 0,
    };
    namespace model = pie_cuda_driver::model;
    const bool folded = model::qwen35_fused_shared_scalar_gate_enabled();
    const std::string joined =
        lp + "mlp.shared_expert." + (folded ? "gate_up_gate_proj.weight" : "gate_up_proj.weight");

    for (std::uint32_t tp : {1u, 2u}) {
        const std::string at = " (tp " + std::to_string(tp) + ")";
        auto target = pie_cuda_driver::cuda_device_target();
        target.tp_size = tp;
        target.tp_rank = 0;

        pie_loader::ModelContract contract;
        try {
            model::ContractBuilder builder(checkpoint, facts, target, "",
                                           model::Mxfp4MoeRequest::Auto,
                                           model::Component::Full,
                                           /*stream_routed_experts=*/false, contract);
            model::author_qwen3_5_moe_contract(builder);
            builder.finish();
        } catch (const std::exception& error) {
            check(false, "authoring" + at + ": " + error.what());
            continue;
        }

        const auto v = contract.view();
        std::map<std::string_view, std::vector<std::int64_t>> shapes;
        for (std::size_t i = 0; i < v.tensors.len; ++i) {
            const auto& t = v.tensors.ptr[i];
            shapes[view_of(t.name)] =
                std::vector<std::int64_t>(t.shape.ptr, t.shape.ptr + t.shape.len);
        }

        const std::int64_t local = kShared / static_cast<std::int64_t>(tp);
        check(shapes[joined] ==
                  std::vector<std::int64_t>{2 * local + (folded ? 1 : 0), kHidden},
              "the join is this rank's two bands, in row order" + at);
        // Additive: the fold-into-routed path reads these separately.
        check(shapes.count(lp + "mlp.shared_expert.gate_proj.weight") == 1 &&
                  shapes.count(lp + "mlp.shared_expert.up_proj.weight") == 1,
              "the unfused projections survive beside it" + at);
        check(shapes[lp + "mlp.shared_expert.gate_proj.weight"] ==
                  std::vector<std::int64_t>{local, kHidden},
              "and are themselves sharded" + at);

        try {
            const pie_loader::PieLoaderContractRequest request =
                pie_loader::build_contract_request(checkpoint, target, v);
            const pie_loader::LoadPlan plan = pie_loader::LoadPlan::compile(request);
            plan.verify(request);
        } catch (const std::exception& error) {
            check(false, "compiling" + at + ": " + error.what());
        }
    }
    std::filesystem::remove_all(dir);
}

/// Qwen3-MoE declares its per-expert weights as a group when asked to stream.
///
/// The plain HF MoE source layout -- one tensor per expert per projection --
/// is the case `index_src` exists for, and this is the whole of it: the
/// streaming branch is the stacking pass with the outer concatenation removed,
/// so the instance it leaves behind is what the stack was already building per
/// expert. GLM-5 shares the helper, so this covers it too.
void test_qwen3_moe_streams_experts_as_a_group() {
    constexpr std::int64_t kExperts = 4;
    constexpr std::int64_t kHidden = 64;
    constexpr std::int64_t kInter = 32;
    constexpr std::int64_t kLayers = 2;

    std::vector<TypedTensor> tensors{
        {"model.embed_tokens.weight", {16, kHidden}, "BF16"},
        {"model.norm.weight", {kHidden}, "BF16"},
        {"lm_head.weight", {16, kHidden}, "BF16"},
    };
    for (int layer = 0; layer < kLayers; ++layer) {
        const std::string lp = "model.layers." + std::to_string(layer) + ".mlp.";
        tensors.push_back({lp + "gate.weight", {kExperts, kHidden}, "BF16"});
        for (int e = 0; e < kExperts; ++e) {
            const std::string ep = lp + "experts." + std::to_string(e) + ".";
            tensors.push_back({ep + "gate_proj.weight", {kInter, kHidden}, "BF16"});
            tensors.push_back({ep + "up_proj.weight", {kInter, kHidden}, "BF16"});
            tensors.push_back({ep + "down_proj.weight", {kHidden, kInter}, "BF16"});
        }
    }
    const std::filesystem::path dir =
        write_typed_checkpoint("qwen3_moe_streamed", tensors);

    std::string open_error;
    pie_loader::Checkpoint checkpoint = pie_loader::Checkpoint::open(dir.string(), &open_error);
    check(static_cast<bool>(checkpoint), "the synthetic checkpoint opens: " + open_error);
    if (!checkpoint) return;

    const pie_cuda_driver::model::ModelFacts facts{
        .model_type = "qwen3_moe",
        .quant_method = "",
        .num_hidden_layers = kLayers,
        .num_experts = kExperts,
        .head_dim = 0,
        .mamba_groups = 0,
    };
    namespace model = pie_cuda_driver::model;

    pie_loader::ModelContract contract;
    try {
        model::ContractBuilder builder(checkpoint, facts,
                                       pie_cuda_driver::cuda_device_target(), "",
                                       model::Mxfp4MoeRequest::Auto,
                                       model::Component::Full,
                                       /*stream_routed_experts=*/true, contract);
        model::author_qwen3_5_moe_contract(builder);
        builder.finish();
    } catch (const std::exception& error) {
        check(false, std::string("authoring: ") + error.what());
        return;
    }

    const auto v = contract.view();
    // One group per layer, not one for the model: a name template carries a
    // single `{}` and this checkpoint spells an expert with two indices.
    check(v.groups.len == kLayers, "one group per layer");
    if (v.groups.len != kLayers) return;

    for (std::size_t layer = 0; layer < v.groups.len; ++layer) {
        const auto& g = v.groups.ptr[layer];
        const std::string at = " (layer " + std::to_string(layer) + ")";
        check(view_of(g.name) ==
                  "model.layers." + std::to_string(layer) + ".mlp.experts",
              "the group is named where the bind looks for it" + at);
        check(g.arity == kExperts, "one instance per expert" + at);

        std::map<std::string_view, std::vector<std::int64_t>> declared;
        for (std::size_t i = 0; i < g.tensors.len; ++i) {
            const auto& t = g.tensors.ptr[i];
            declared[view_of(t.name)] =
                std::vector<std::int64_t>(t.shape.ptr, t.shape.ptr + t.shape.len);
        }
        check(declared.size() == 2, "two tensors, gate_up and down" + at);
        // Rank 2, not 3: the stack's leading 1 existed only to make the
        // per-expert slabs concatenable, and there is no concatenation left.
        check(declared["gate_up_proj"] == std::vector<std::int64_t>{2 * kInter, kHidden},
              "gate and up, joined" + at);
        check(declared["down_proj"] == std::vector<std::int64_t>{kHidden, kInter},
              "down, as the checkpoint stores it" + at);
    }

    for (std::size_t i = 0; i < v.tensors.len; ++i) {
        const auto name = view_of(v.tensors.ptr[i].name);
        check(name.find("experts.") == std::string_view::npos,
              "no per-expert tensor is published outside the group");
    }

    try {
        const pie_loader::PieLoaderContractRequest request =
            pie_loader::build_contract_request(
                checkpoint, pie_cuda_driver::cuda_device_target(), v);
        // Compiling *is* the interchangeability proof.
        const pie_loader::LoadPlan plan = pie_loader::LoadPlan::compile(request);
        plan.verify(request);

        const auto& pv = plan.view();
        check(pv.groups.len == kLayers, "the plan carries every group");
        if (pv.groups.len != kLayers) return;
        const std::uint64_t one_expert =
            static_cast<std::uint64_t>((2 * kInter * kHidden + kHidden * kInter) * 2);
        for (std::size_t layer = 0; layer < pv.groups.len; ++layer) {
            const auto& pg = pv.groups.ptr[layer];
            const std::string at = " (layer " + std::to_string(layer) + ")";
            check(pg.arity == kExperts, "the plan agrees on the arity" + at);
            check(pg.bindings.len ==
                      pg.bindings_per_instance * static_cast<std::size_t>(kExperts),
                  "every instance is bound" + at);
            check(pg.plan->memory.persistent_bytes >= one_expert &&
                      pg.plan->memory.persistent_bytes < one_expert + 4096,
                  "a slot is priced at one expert" + at);
        }
        check(pv.memory.persistent_bytes < one_expert * kExperts,
              "the experts left the resident plan");
    } catch (const std::exception& error) {
        check(false, std::string("compiling: ") + error.what());
    }
}

/// GPT-OSS streams its experts out of a fused bank, which is what `select` is.
///
/// The counterpart to the Qwen test and the other half of what a group has to
/// cover: there the checkpoint names a tensor per expert and an instance is
/// picked by substituting into a template; here one tensor per layer holds all
/// of them and an instance is a band whose start is the only thing the index
/// decides. The type of that band cannot depend on the index -- a slice of
/// fixed length has a fixed shape -- so interchangeability is structural, and
/// the compile is the proof.
void test_gpt_oss_streams_experts_as_a_group() {
    constexpr std::int64_t kExperts = 4;
    constexpr std::int64_t kHidden = 64;
    constexpr std::int64_t kInter = 32;
    constexpr std::int64_t kLayers = 2;

    std::vector<TypedTensor> tensors{
        {"model.embed_tokens.weight", {16, kHidden}, "BF16"},
        {"model.norm.weight", {kHidden}, "BF16"},
        {"lm_head.weight", {16, kHidden}, "BF16"},
    };
    for (int layer = 0; layer < kLayers; ++layer) {
        const std::string ep =
            "model.layers." + std::to_string(layer) + ".mlp.experts.";
        tensors.push_back({ep + "gate_up_proj_blocks",
                           {kExperts, 2 * kInter, kHidden / 32, 16}, "U8"});
        tensors.push_back({ep + "gate_up_proj_scales",
                           {kExperts, 2 * kInter, kHidden / 32}, "U8"});
        tensors.push_back({ep + "gate_up_proj_bias", {kExperts, 2 * kInter}, "BF16"});
        tensors.push_back({ep + "down_proj_blocks",
                           {kExperts, kHidden, kInter / 32, 16}, "U8"});
        tensors.push_back({ep + "down_proj_scales",
                           {kExperts, kHidden, kInter / 32}, "U8"});
        tensors.push_back({ep + "down_proj_bias", {kExperts, kHidden}, "BF16"});
    }
    const std::filesystem::path dir =
        write_typed_checkpoint("gpt_oss_streamed", tensors);

    std::string open_error;
    pie_loader::Checkpoint checkpoint = pie_loader::Checkpoint::open(dir.string(), &open_error);
    check(static_cast<bool>(checkpoint), "the synthetic checkpoint opens: " + open_error);
    if (!checkpoint) return;

    const pie_cuda_driver::model::ModelFacts facts{
        .model_type = "gpt_oss",
        .quant_method = "",
        .num_hidden_layers = kLayers,
        .num_experts = kExperts,
        .head_dim = 0,
        .mamba_groups = 0,
    };
    namespace model = pie_cuda_driver::model;

    pie_loader::ModelContract contract;
    try {
        model::ContractBuilder builder(checkpoint, facts,
                                       pie_cuda_driver::cuda_device_target(), "",
                                       model::Mxfp4MoeRequest::RoutedDecode,
                                       model::Component::Full,
                                       /*stream_routed_experts=*/true, contract);
        model::author_gpt_oss_contract(builder);
        builder.finish();
    } catch (const std::exception& error) {
        check(false, std::string("authoring: ") + error.what());
        return;
    }

    const auto v = contract.view();
    check(v.groups.len == kLayers, "one group per layer");
    if (v.groups.len != kLayers) return;

    for (std::size_t layer = 0; layer < v.groups.len; ++layer) {
        const auto& g = v.groups.ptr[layer];
        const std::string at = " (layer " + std::to_string(layer) + ")";
        check(view_of(g.name) ==
                  "model.layers." + std::to_string(layer) + ".mlp.experts",
              "the group is named where the bind looks for it" + at);
        check(g.arity == kExperts, "one instance per expert" + at);

        std::map<std::string_view, std::vector<std::int64_t>> declared;
        for (std::size_t i = 0; i < g.tensors.len; ++i) {
            const auto& t = g.tensors.ptr[i];
            declared[view_of(t.name)] =
                std::vector<std::int64_t>(t.shape.ptr, t.shape.ptr + t.shape.len);
        }
        check(declared.size() == 4, "weights and scales, both halves" + at);
        // The leading 1 survives: a `Select` lowers to a `Slice`, and a slice
        // keeps its rank.
        check(declared["gate_up_proj.weight"] ==
                  std::vector<std::int64_t>{1, 2 * kInter, kHidden / 32, 16},
              "one expert's packed gate/up band" + at);
        check(declared["down_proj.weight"] ==
                  std::vector<std::int64_t>{1, kHidden, kInter / 32, 16},
              "one expert's packed down band" + at);
        check(declared["gate_up_proj.weight_scale"] ==
                  std::vector<std::int64_t>{1, 2 * kInter, kHidden / 32},
              "its factors come with it" + at);
    }

    // The biases are small and need a de-interleave the contract has no node
    // for, so they stay resident under the names the bind already reads.
    bool bias_resident = false;
    for (std::size_t i = 0; i < v.tensors.len; ++i) {
        const auto name = view_of(v.tensors.ptr[i].name);
        if (name == "model.layers.0.mlp.experts.gate_up_proj.bias") {
            bias_resident = true;
        }
        check(name.find("_blocks") == std::string_view::npos &&
                  name.find("_scales") == std::string_view::npos,
              "no expert bank is published outside the group");
    }
    check(bias_resident, "the expert biases stayed resident");

    try {
        const pie_loader::PieLoaderContractRequest request =
            pie_loader::build_contract_request(
                checkpoint, pie_cuda_driver::cuda_device_target(), v);
        // Compiling is the interchangeability proof, and for `select` it is
        // also the bounds check: an arity wider than the bank fails here
        // rather than producing a slot of whatever followed it.
        const pie_loader::LoadPlan plan = pie_loader::LoadPlan::compile(request);
        plan.verify(request);

        const auto& pv = plan.view();
        check(pv.groups.len == kLayers, "the plan carries every group");
        if (pv.groups.len != kLayers) return;
        const std::uint64_t one_expert = static_cast<std::uint64_t>(
            2 * kInter * (kHidden / 32) * 16 + 2 * kInter * (kHidden / 32) +
            kHidden * (kInter / 32) * 16 + kHidden * (kInter / 32));
        for (std::size_t layer = 0; layer < pv.groups.len; ++layer) {
            const auto& pg = pv.groups.ptr[layer];
            const std::string at = " (layer " + std::to_string(layer) + ")";
            check(pg.arity == kExperts, "the plan agrees on the arity" + at);
            check(pg.bindings.len ==
                      pg.bindings_per_instance * static_cast<std::size_t>(kExperts),
                  "every instance is bound" + at);
            check(pg.plan->memory.persistent_bytes >= one_expert &&
                      pg.plan->memory.persistent_bytes < one_expert + 4096,
                  "a slot is priced at one expert, not " +
                      std::to_string(kExperts) + at);
        }
    } catch (const std::exception& error) {
        check(false, std::string("compiling: ") + error.what());
    }
}

/// Kimi dequantizes and stacks its routed experts in the contract.
///
/// The checkpoint stores each expert separately as W4A16: `I32` words of eight
/// 4-bit codes beside `BF16` group scales, an element being `code - 8` times
/// its factor. That is `QuantScheme::Int4B8`, and it is a different element
/// format under the same `Scale` node MXFP4 uses.
///
/// Unlike DeepSeek-V4 the packed originals are *kept*: Kimi picks between the
/// W4A16 GEMVs and the batched bf16 path per step, by token count, so both
/// forms have to be resident. What this pins is that the stacks are additive
/// and correctly sharded, and that nothing consumed the packed weights the
/// other path still reads.
void test_kimi_stacks_experts_as_bf16() {
    constexpr std::int64_t kExperts = 2;
    constexpr std::int64_t kHidden = 64;
    constexpr std::int64_t kInterFull = 64;
    constexpr std::int64_t kGroup = 32;
    constexpr std::int64_t kCodesPerWord = 8;

    const std::string prefix = "language_model.";
    std::vector<TypedTensor> tensors{
        {prefix + "model.embed_tokens.weight", {16, kHidden}, "BF16"},
        {prefix + "model.norm.weight", {kHidden}, "BF16"},
        {prefix + "lm_head.weight", {16, kHidden}, "BF16"},
        {prefix + "model.layers.0.mlp.gate.weight", {kExperts, kHidden}, "BF16"},
    };
    const std::string mlp = "model.layers.0.mlp.";
    for (int e = 0; e < kExperts; ++e) {
        const std::string ep = prefix + mlp + "experts." + std::to_string(e) + ".";
        for (const char* fc1 : {"gate_proj", "up_proj"}) {
            tensors.push_back(
                {ep + fc1 + ".weight_packed", {kInterFull, kHidden / kCodesPerWord}, "I32"});
            tensors.push_back(
                {ep + fc1 + ".weight_scale", {kInterFull, kHidden / kGroup}, "BF16"});
        }
        tensors.push_back(
            {ep + "down_proj.weight_packed", {kHidden, kInterFull / kCodesPerWord}, "I32"});
        tensors.push_back(
            {ep + "down_proj.weight_scale", {kHidden, kInterFull / kGroup}, "BF16"});
    }
    const std::filesystem::path dir = write_typed_checkpoint("kimi_w4a16_experts", tensors);

    std::string open_error;
    pie_loader::Checkpoint checkpoint = pie_loader::Checkpoint::open(dir.string(), &open_error);
    check(static_cast<bool>(checkpoint), "the synthetic checkpoint opens: " + open_error);
    if (!checkpoint) return;

    const pie_cuda_driver::model::ModelFacts facts{
        .model_type = "kimi_k2",
        .quant_method = "",
        .num_hidden_layers = 1,
        .num_experts = kExperts,
        .head_dim = 0,
        .mamba_groups = 0,
    };
    namespace model = pie_cuda_driver::model;

    for (int tp : {1, 2}) {
        for (int rank = 0; rank < tp; ++rank) {
            const std::int64_t inter = kInterFull / tp;
            const std::string at =
                " (tp " + std::to_string(tp) + " rank " + std::to_string(rank) + ")";

            auto target = pie_cuda_driver::cuda_device_target();
            target.tp_size = static_cast<std::uint32_t>(tp);
            target.tp_rank = static_cast<std::uint32_t>(rank);

            pie_loader::ModelContract contract;
            try {
                model::ContractBuilder builder(checkpoint, facts, target, "",
                                               model::Mxfp4MoeRequest::Auto,
                                               model::Component::Full, /*stream_routed_experts=*/false, contract);
                model::author_kimi_contract(builder);
                builder.finish();
            } catch (const std::exception& error) {
                check(false, "authoring" + at + ": " + error.what());
                continue;
            }

            std::map<std::string_view, std::vector<std::int64_t>> declared;
            std::map<std::string_view, std::uint32_t> dtypes;
            const auto v = contract.view();
            for (std::size_t i = 0; i < v.tensors.len; ++i) {
                const auto& t = v.tensors.ptr[i];
                declared[view_of(t.name)] =
                    std::vector<std::int64_t>(t.shape.ptr, t.shape.ptr + t.shape.len);
                dtypes[view_of(t.name)] = t.encoding.dtype;
            }
            const auto declares = [&](const std::string& name,
                                      const std::vector<std::int64_t>& want) {
                const auto found = declared.find(name);
                check(found != declared.end() && found->second == want &&
                          dtypes[name] ==
                              static_cast<std::uint32_t>(pie_loader::PieLoaderDType::BF16),
                      "'" + name + "' is this rank's share" + at);
            };
            declares(mlp + "experts.gate_up.weight", {kExperts, 2 * inter, kHidden});
            declares(mlp + "experts.down.weight", {kExperts, kHidden, inter});
            declares(mlp + "experts.gate_up.scale", {kExperts, 2 * inter, kHidden / kGroup});
            declares(mlp + "experts.down.scale", {kExperts, kHidden, inter / kGroup});

            // The per-step W4A16 GEMV path reads the packed weights, so the
            // stacks are additive and must not have consumed them.
            for (int e = 0; e < kExperts; ++e) {
                const std::string ep = mlp + "experts." + std::to_string(e) + ".";
                check(declared.count(ep + "gate_proj.weight_packed") == 1 &&
                          declared.count(ep + "down_proj.weight_packed") == 1,
                      "'" + ep + "' is still published packed" + at);
            }

            // Authoring is only half of it: the shapes above are what the
            // contract *claims*, and only compilation checks the claim against
            // the expression that is supposed to produce it.
            try {
                const pie_loader::PieLoaderContractRequest request =
                    pie_loader::build_contract_request(checkpoint, target, v);
                const pie_loader::LoadPlan plan = pie_loader::LoadPlan::compile(request);
                plan.verify(request);
                // The stacked factors are a name, not a weight: the dequantize
                // that consumes them happens at load, so binding them would
                // pin bytes in the persistent arena that nothing reads.
                for (const char* half : {"gate_up", "down"}) {
                    const std::string name = mlp + "experts." + half + ".scale";
                    check(planned_internal(plan, name), "'" + name + "' is not bound" + at);
                }
            } catch (const std::exception& error) {
                check(false, "compiling" + at + ": " + error.what());
            }
        }
    }
}

/// CSM narrows the whole checkpoint to bf16, in the contract.
///
/// eustlb/csm-1b ships fp32 and every CSM kernel is bf16-store/fp32-compute.
/// The bind used to allocate a bf16 copy of each weight while the fp32
/// original stayed loaded, so peak memory was three times the bf16 model. What
/// this pins is that the declaration alone does it -- a dtype is not an
/// operator, so nothing but the declared encoding had to change -- and that
/// the resulting plan compiles.
void test_csm_narrows_fp32_weights_to_bf16() {
    constexpr std::int64_t kHidden = 8;
    constexpr std::int64_t kInter = 16;
    const std::string p = "backbone_model.";
    const std::filesystem::path dir = write_synthetic_checkpoint(
        "csm_fp32",
        {{"embed_text_tokens.weight", {32, kHidden}},
         {p + "norm.weight", {kHidden}},
         {"lm_head.weight", {32, kHidden}},
         {p + "layers.0.input_layernorm.weight", {kHidden}},
         {p + "layers.0.self_attn.q_proj.weight", {kHidden, kHidden}},
         {p + "layers.0.self_attn.o_proj.weight", {kHidden, kHidden}},
         {p + "layers.0.mlp.gate_proj.weight", {kInter, kHidden}},
         {p + "layers.0.mlp.up_proj.weight", {kInter, kHidden}},
         {p + "layers.0.mlp.down_proj.weight", {kHidden, kInter}}},
        "F32");

    std::string open_error;
    pie_loader::Checkpoint checkpoint = pie_loader::Checkpoint::open(dir.string(), &open_error);
    check(static_cast<bool>(checkpoint), "the synthetic checkpoint opens: " + open_error);
    if (!checkpoint) return;

    const pie_cuda_driver::model::ModelFacts facts{
        .model_type = "csm",
        .quant_method = "",
        .num_hidden_layers = 1,
        .num_experts = 0,
        .head_dim = 0,
        .mamba_groups = 0,
    };
    namespace model = pie_cuda_driver::model;

    auto target = pie_cuda_driver::cuda_device_target();
    target.tp_size = 1;
    target.tp_rank = 0;

    pie_loader::ModelContract contract;
    try {
        model::ContractBuilder builder(checkpoint, facts, target, "",
                                       model::Mxfp4MoeRequest::RoutedDecode,
                                       model::Component::Full, /*stream_routed_experts=*/false, contract);
        model::author_csm_contract(builder);
        builder.finish();
    } catch (const std::exception& error) {
        check(false, std::string("authoring: ") + error.what());
        return;
    }

    const auto v = contract.view();
    std::size_t declared = 0;
    for (std::size_t i = 0; i < v.tensors.len; ++i) {
        const auto& t = v.tensors.ptr[i];
        ++declared;
        check(t.encoding.kind ==
                      static_cast<std::uint32_t>(pie_loader::PieLoaderEncodingKind::Raw) &&
                  t.encoding.dtype ==
                      static_cast<std::uint32_t>(pie_loader::PieLoaderDType::BF16),
              "'" + std::string(view_of(t.name)) + "' is declared bf16");
    }
    check(declared == 9, "every source tensor is still published");

    try {
        const pie_loader::PieLoaderContractRequest request =
            pie_loader::build_contract_request(checkpoint, target, v);
        const pie_loader::LoadPlan plan = pie_loader::LoadPlan::compile(request);
        plan.verify(request);
    } catch (const std::exception& error) {
        check(false, std::string("compiling: ") + error.what());
    }
    std::filesystem::remove_all(dir);
}

/// Nemotron-H's Mamba mixer is split by the contract, on head and group
/// boundaries.
///
/// The mixer's `in_proj` stacks five bands that split by different units, so a
/// rank's share is not a row range of it -- it is the concatenation of five
/// independently sharded bands. Bind used to build that with `cudaMemcpy2D`
/// after the load, which meant every rank still read and kept the whole mixer.
///
/// What this pins is that the pass says it in the algebra and that the result
/// compiles: the local extents are exact, `A_log` and friends follow the same
/// head split, and `out_proj` splits by column instead of by row.
void test_nemotron_h_mamba_splits_on_unit_boundaries() {
    constexpr std::int64_t kHeads = 8;
    constexpr std::int64_t kHeadDim = 4;
    constexpr std::int64_t kGroups = 4;
    constexpr std::int64_t kState = 2;
    constexpr std::int64_t kHidden = 6;
    constexpr std::int64_t kConvKernel = 4;
    constexpr std::int64_t kIntermediate = kHeads * kHeadDim;
    constexpr std::int64_t kGroupState = kGroups * kState;
    constexpr std::int64_t kConvDim = kIntermediate + 2 * kGroupState;
    constexpr std::int64_t kProjection = 2 * kIntermediate + 2 * kGroupState + kHeads;

    const std::string mixer = "language_model.backbone.layers.0.mixer.";
    const std::filesystem::path dir = write_synthetic_checkpoint(
        "nemotron_h_mamba",
        {{"language_model.backbone.embeddings.weight", {16, kHidden}},
         {"language_model.backbone.norm_f.weight", {kHidden}},
         {"language_model.lm_head.weight", {16, kHidden}},
         {"language_model.backbone.layers.0.norm.weight", {kHidden}},
         {mixer + "in_proj.weight", {kProjection, kHidden}},
         {mixer + "conv1d.weight", {kConvDim, kConvKernel}},
         {mixer + "conv1d.bias", {kConvDim}},
         {mixer + "A_log", {kHeads}},
         {mixer + "D", {kHeads}},
         {mixer + "dt_bias", {kHeads}},
         {mixer + "norm.weight", {kIntermediate}},
         {mixer + "out_proj.weight", {kHidden, kIntermediate}}});

    std::string open_error;
    pie_loader::Checkpoint checkpoint = pie_loader::Checkpoint::open(dir.string(), &open_error);
    check(static_cast<bool>(checkpoint), "the synthetic checkpoint opens: " + open_error);
    if (!checkpoint) return;

    const pie_cuda_driver::model::ModelFacts facts{
        .model_type = "nemotron_h",
        .quant_method = "",
        .num_hidden_layers = 1,
        .num_experts = 0,
        .head_dim = 0,
        .mamba_groups = static_cast<std::uint32_t>(kGroups),
    };
    namespace model = pie_cuda_driver::model;

    for (std::uint32_t tp : {1u, 2u, 4u}) {
        for (std::uint32_t rank = 0; rank < tp; ++rank) {
            const std::string at =
                " (tp " + std::to_string(tp) + " rank " + std::to_string(rank) + ")";
            const std::int64_t local_heads = kHeads / tp;
            const std::int64_t local_groups = kGroups / tp;
            const std::int64_t local_intermediate = local_heads * kHeadDim;
            const std::int64_t local_group_state = local_groups * kState;

            auto target = pie_cuda_driver::cuda_device_target();
            target.tp_size = tp;
            target.tp_rank = rank;

            pie_loader::ModelContract contract;
            try {
                model::ContractBuilder builder(checkpoint, facts, target, "",
                                               model::Mxfp4MoeRequest::RoutedDecode,
                                               model::Component::Full, /*stream_routed_experts=*/false, contract);
                model::author_nemotron_h_contract(builder);
                builder.finish();
            } catch (const std::exception& error) {
                check(false, "authoring" + at + ": " + error.what());
                continue;
            }

            std::map<std::string_view, std::vector<std::int64_t>> declared;
            const auto v = contract.view();
            for (std::size_t i = 0; i < v.tensors.len; ++i) {
                const auto& t = v.tensors.ptr[i];
                declared[view_of(t.name)] =
                    std::vector<std::int64_t>(t.shape.ptr, t.shape.ptr + t.shape.len);
            }
            const auto shape_is = [&](const std::string& name,
                                      const std::vector<std::int64_t>& want) {
                const auto found = declared.find(name);
                check(found != declared.end() && found->second == want,
                      "'" + name + "' is this rank's share" + at);
            };
            shape_is(mixer + "in_proj.weight",
                     {2 * local_intermediate + 2 * local_group_state + local_heads, kHidden});
            shape_is(mixer + "conv1d.weight",
                     {local_intermediate + 2 * local_group_state, kConvKernel});
            shape_is(mixer + "conv1d.bias", {local_intermediate + 2 * local_group_state});
            shape_is(mixer + "A_log", {local_heads});
            shape_is(mixer + "D", {local_heads});
            shape_is(mixer + "dt_bias", {local_heads});
            shape_is(mixer + "norm.weight", {local_intermediate});
            shape_is(mixer + "out_proj.weight", {kHidden, local_intermediate});

            // Authoring is only half of it: the shapes above are what the
            // contract *claims*, and only compilation checks the claim against
            // the expression that is supposed to produce it.
            try {
                const pie_loader::PieLoaderContractRequest request =
                    pie_loader::build_contract_request(checkpoint, target, v);
                const pie_loader::LoadPlan plan = pie_loader::LoadPlan::compile(request);
                plan.verify(request);
            } catch (const std::exception& error) {
                check(false, "compiling" + at + ": " + error.what());
            }
        }
    }
    std::filesystem::remove_all(dir);
}

/// Qwen3.5's Gated DeltaNet tensors are split by block, under the prefix the
/// checkpoint actually uses.
///
/// `in_proj_qkv`, `conv1d.weight` and `conv1d.bias` stack `[K | K | V]` on axis
/// 0, so a uniform row shard hands a rank part of K where it needs V -- and
/// because none of those names is in the HF shard-axis table, the fallback is
/// not a wrong split but no split at all, which is a rank holding the whole
/// tensor while the kernel indexes it as its share.
///
/// The pass that says this went in matching `model.layers.`, and Qwen3.5 names
/// its decoder `model.language_model.layers.`: it never fired, and nothing
/// noticed because the only thing that could have is a test with the real
/// names in it. Which is what this is.
/// The speculative head reads the same `lm_head` as the main path, in int8.
///
/// Env-gated, so this asserts whichever side of `PIE_QWEN35_MTP_INT8_LM_HEAD`
/// this process is on. Both are load-bearing: with the flag off there must be
/// no second view at all, and with it on the bf16 original must survive -- the
/// main path still reads it, so a re-encode here would be a regression rather
/// than an optimization.
void test_qwen3_5_mtp_int8_lm_head() {
    constexpr std::int64_t kHidden = 8;
    constexpr std::int64_t kVocab = 32;
    const std::string p = "model.";
    const std::string l0 = p + "layers.0.";
    const std::string m0 = "mtp.layers.0.";
    const std::filesystem::path dir = write_synthetic_checkpoint(
        "qwen3_5_mtp_int8", {{p + "embed_tokens.weight", {kVocab, kHidden}},
                             {p + "norm.weight", {kHidden}},
                             {"lm_head.weight", {kVocab, kHidden}},
                             {l0 + "input_layernorm.weight", {kHidden}},
                             {l0 + "post_attention_layernorm.weight", {kHidden}},
                             {l0 + "self_attn.q_proj.weight", {kHidden, kHidden}},
                             {l0 + "self_attn.k_proj.weight", {kHidden, kHidden}},
                             {l0 + "self_attn.v_proj.weight", {kHidden, kHidden}},
                             {l0 + "self_attn.o_proj.weight", {kHidden, kHidden}},
                             {l0 + "mlp.gate_proj.weight", {kHidden, kHidden}},
                             {l0 + "mlp.up_proj.weight", {kHidden, kHidden}},
                             {l0 + "mlp.down_proj.weight", {kHidden, kHidden}},
                             {"mtp.fc.weight", {kHidden, 2 * kHidden}},
                             {"mtp.norm.weight", {kHidden}},
                             {"mtp.pre_fc_norm_embedding.weight", {kHidden}},
                             {"mtp.pre_fc_norm_hidden.weight", {kHidden}},
                             {m0 + "input_layernorm.weight", {kHidden}},
                             {m0 + "post_attention_layernorm.weight", {kHidden}},
                             {m0 + "self_attn.q_proj.weight", {kHidden, kHidden}},
                             {m0 + "self_attn.k_proj.weight", {kHidden, kHidden}},
                             {m0 + "self_attn.v_proj.weight", {kHidden, kHidden}},
                             {m0 + "self_attn.o_proj.weight", {kHidden, kHidden}},
                             {m0 + "mlp.gate_proj.weight", {kHidden, kHidden}},
                             {m0 + "mlp.up_proj.weight", {kHidden, kHidden}},
                             {m0 + "mlp.down_proj.weight", {kHidden, kHidden}}});

    std::string open_error;
    pie_loader::Checkpoint checkpoint = pie_loader::Checkpoint::open(dir.string(), &open_error);
    check(static_cast<bool>(checkpoint), "the synthetic checkpoint opens: " + open_error);
    if (!checkpoint) return;

    const pie_cuda_driver::model::ModelFacts facts{
        .model_type = "qwen3_5",
        .quant_method = "",
        .num_hidden_layers = 1,
        .num_experts = 0,
        .head_dim = 0,
        .mamba_groups = 0,
    };
    namespace model = pie_cuda_driver::model;

    auto target = pie_cuda_driver::cuda_device_target();
    target.tp_size = 1;
    target.tp_rank = 0;

    pie_loader::ModelContract contract;
    try {
        model::ContractBuilder builder(checkpoint, facts, target, "",
                                       model::Mxfp4MoeRequest::RoutedDecode,
                                       model::Component::Full, /*stream_routed_experts=*/false,
                                       contract);
        model::author_qwen3_5_contract(builder);
        builder.finish();
    } catch (const std::exception& error) {
        check(false, std::string("authoring: ") + error.what());
        std::filesystem::remove_all(dir);
        return;
    }

    const auto v = contract.view();
    const pie_loader::PieLoaderTensorContractView* head = nullptr;
    const pie_loader::PieLoaderTensorContractView* original = nullptr;
    for (std::size_t i = 0; i < v.tensors.len; ++i) {
        const auto& t = v.tensors.ptr[i];
        if (view_of(t.name) == "mtp.lm_head") head = &t;
        if (view_of(t.name) == "lm_head.weight") original = &t;
    }

    if (model::qwen35_mtp_int8_lm_head_enabled()) {
        check(head != nullptr, "the int8 speculative head is published");
        if (head != nullptr) {
            check(head->encoding.kind ==
                          static_cast<std::uint32_t>(pie_loader::PieLoaderEncodingKind::Quant) &&
                      head->encoding.quant.scheme ==
                          static_cast<std::uint32_t>(pie_loader::PieLoaderQuantScheme::Int8Symmetric),
                  "the speculative head is int8-symmetric");
        }
        // A view, not a re-encode: the main path still reads bf16.
        check(original != nullptr &&
                  original->encoding.dtype ==
                      static_cast<std::uint32_t>(pie_loader::PieLoaderDType::BF16),
              "the bf16 head survives beside it");
    } else {
        check(head == nullptr, "no speculative head without the flag");
        check(original != nullptr, "the bf16 head is published");
    }

    try {
        const pie_loader::PieLoaderContractRequest request =
            pie_loader::build_contract_request(checkpoint, target, v);
        const pie_loader::LoadPlan plan = pie_loader::LoadPlan::compile(request);
        plan.verify(request);

        // The bind reads `engine.quant_meta("mtp.lm_head")`, which the weight
        // store keys off exactly this table. The scale is named by appending a
        // suffix to the declaration, and "mtp.lm_head" does not end in
        // ".weight", so this is the assertion that the name carries.
        bool attached = false;
        for (const auto& a : pie_cuda_driver::resolve_quant_attachments(plan.view())) {
            if (a.tensor_name == "mtp.lm_head") attached = true;
        }
        check(attached == model::qwen35_mtp_int8_lm_head_enabled(),
              "the speculative head's scale reaches the driver's bind table");
    } catch (const std::exception& error) {
        check(false, std::string("compiling: ") + error.what());
    }
    std::filesystem::remove_all(dir);
}

void test_qwen3_5_gdn_splits_by_block() {
    constexpr std::int64_t kKey = 8;
    constexpr std::int64_t kValue = 16;
    constexpr std::int64_t kHidden = 6;
    constexpr std::int64_t kConvKernel = 4;
    constexpr std::int64_t kConvDim = 2 * kKey + kValue;

    const std::string p = "model.language_model.";
    const std::string la = p + "layers.0.linear_attn.";
    const std::filesystem::path dir = write_synthetic_checkpoint(
        "qwen3_5_gdn", {{p + "embed_tokens.weight", {16, kHidden}},
                        {p + "norm.weight", {kHidden}},
                        {"lm_head.weight", {16, kHidden}},
                        {p + "layers.0.input_layernorm.weight", {kHidden}},
                        {p + "layers.0.post_attention_layernorm.weight", {kHidden}},
                        {p + "layers.0.mlp.gate_proj.weight", {kValue, kHidden}},
                        {p + "layers.0.mlp.up_proj.weight", {kValue, kHidden}},
                        {p + "layers.0.mlp.down_proj.weight", {kHidden, kValue}},
                        {la + "in_proj_qkv.weight", {kConvDim, kHidden}},
                        {la + "conv1d.weight", {kConvDim, kConvKernel}},
                        {la + "conv1d.bias", {kConvDim}},
                        {la + "in_proj_z.weight", {kValue, kHidden}},
                        {la + "in_proj_b.weight", {kValue, kHidden}},
                        {la + "in_proj_a.weight", {kValue, kHidden}},
                        {la + "dt_bias", {kValue}},
                        {la + "A_log", {kValue}},
                        {la + "norm.weight", {kValue}},
                        {la + "out_proj.weight", {kHidden, kValue}}});

    std::string open_error;
    pie_loader::Checkpoint checkpoint = pie_loader::Checkpoint::open(dir.string(), &open_error);
    check(static_cast<bool>(checkpoint), "the synthetic checkpoint opens: " + open_error);
    if (!checkpoint) return;

    const pie_cuda_driver::model::ModelFacts facts{
        .model_type = "qwen3_5",
        .quant_method = "",
        .num_hidden_layers = 1,
        .num_experts = 0,
        .head_dim = 0,
        .mamba_groups = 0,
    };
    namespace model = pie_cuda_driver::model;

    for (std::uint32_t tp : {1u, 2u}) {
        for (std::uint32_t rank = 0; rank < tp; ++rank) {
            const std::string at =
                " (tp " + std::to_string(tp) + " rank " + std::to_string(rank) + ")";
            const std::int64_t local = static_cast<std::int64_t>(2 * kKey / tp + kValue / tp);

            auto target = pie_cuda_driver::cuda_device_target();
            target.tp_size = tp;
            target.tp_rank = rank;

            pie_loader::ModelContract contract;
            try {
                model::ContractBuilder builder(checkpoint, facts, target, "",
                                               model::Mxfp4MoeRequest::RoutedDecode,
                                               model::Component::Full, /*stream_routed_experts=*/false, contract);
                model::author_qwen3_5_contract(builder);
                builder.finish();
            } catch (const std::exception& error) {
                check(false, "authoring" + at + ": " + error.what());
                continue;
            }

            std::map<std::string_view, std::vector<std::int64_t>> declared;
            std::map<std::string_view, std::uint32_t> dtypes;
            const auto v = contract.view();
            for (std::size_t i = 0; i < v.tensors.len; ++i) {
                const auto& t = v.tensors.ptr[i];
                declared[view_of(t.name)] =
                    std::vector<std::int64_t>(t.shape.ptr, t.shape.ptr + t.shape.len);
                dtypes[view_of(t.name)] = t.encoding.dtype;
            }
            const auto shape_is = [&](const std::string& name,
                                      const std::vector<std::int64_t>& want) {
                const auto found = declared.find(name);
                check(found != declared.end() && found->second == want,
                      "'" + name + "' is this rank's share" + at);
            };
            // The gated-delta-net kernels take these two as `const float*`.
            // The checkpoint above is bf16, so the widening has to be the
            // contract's: bind used to allocate an fp32 copy of each and leave
            // the bf16 original resident beside it. `dt_bias` is the control --
            // it sits in the same module, is read as bf16, and must not widen.
            const auto dtype_is = [&](const std::string& name,
                                      pie_loader::PieLoaderDType want) {
                const auto found = dtypes.find(name);
                check(found != dtypes.end() &&
                          found->second == static_cast<std::uint32_t>(want),
                      "'" + name + "' is declared " +
                          (want == pie_loader::PieLoaderDType::F32 ? "fp32" : "bf16") + at);
            };
            dtype_is(la + "A_log", pie_loader::PieLoaderDType::F32);
            dtype_is(la + "norm.weight", pie_loader::PieLoaderDType::F32);
            dtype_is(la + "dt_bias", pie_loader::PieLoaderDType::BF16);
            // A_log is per-value-head, so it shards; the gated norm is over
            // head_v_dim, which every rank holds whole.
            shape_is(la + "A_log", {kValue / static_cast<std::int64_t>(tp)});
            shape_is(la + "norm.weight", {kValue});
            shape_is(la + "conv1d.weight", {local, kConvKernel});
            shape_is(la + "conv1d.bias", {local});
            // Under `PIE_QWEN35_FUSED_GDN_PROJ` the shard is the first leg of
            // the join rather than a tensor of its own, so assert whichever
            // layout this process selected. Both must be this rank's share.
            if (model::qwen35_fused_gdn_projection_enabled()) {
                shape_is(la + "in_proj_qkvz.weight",
                         {local + kValue / static_cast<std::int64_t>(tp), kHidden});
            } else {
                shape_is(la + "in_proj_qkv.weight", {local, kHidden});
            }

            // The same prefix decides whether the generic dense join fires, so
            // a wrong one is invisible here and expensive there: bind falls
            // back to concatenating gate/up on the device, after the load.
            check(declared.count(p + "layers.0.mlp.gate_up_proj.fused.weight") == 1,
                  "the dense gate/up join fired" + at);

            try {
                const pie_loader::PieLoaderContractRequest request =
                    pie_loader::build_contract_request(checkpoint, target, v);
                const pie_loader::LoadPlan plan = pie_loader::LoadPlan::compile(request);
                plan.verify(request);
            } catch (const std::exception& error) {
                check(false, "compiling" + at + ": " + error.what());
            }
        }
    }
    std::filesystem::remove_all(dir);
}

nlohmann::json shape_to_json(const pie_loader::PieLoaderI64Slice& s) {
    nlohmann::json out = nlohmann::json::array();
    for (std::size_t i = 0; i < s.len; ++i) out.push_back(s.ptr[i]);
    return out;
}

nlohmann::json quant_to_json(const pie_loader::PieLoaderQuantSpecView& q) {
    return {{"scheme", q.scheme},
            {"logical_dtype", q.logical_dtype},
            {"bits_per_element", q.bits_per_element},
            {"group_size", q.group_size},
            {"channel_axis", q.channel_axis}};
}

nlohmann::json contract_to_json(const pie_loader::PieLoaderModelContractView& v) {
    nlohmann::json nodes = nlohmann::json::array();
    for (std::size_t i = 0; i < v.nodes.len; ++i) {
        const auto& n = v.nodes.ptr[i];
        nlohmann::json parts = nlohmann::json::array();
        for (std::size_t p = 0; p < n.parts.len; ++p) parts.push_back(n.parts.ptr[p]);
        nodes.push_back({{"kind", n.kind},
                         {"name", std::string(view_of(n.name))},
                         {"src", n.src},
                         {"parts", std::move(parts)},
                         {"axis", n.axis},
                         {"start", n.start},
                         {"len", n.len},
                         {"step", n.step},
                         {"fill_bits", n.fill_bits},
                         {"out_shape", shape_to_json(n.out_shape)},
                         {"repack_layout", n.repack_layout},
                         {"out_encoding", {{"kind", n.out_encoding.kind},
                                           {"dtype", n.out_encoding.dtype},
                                           {"quant", quant_to_json(n.out_encoding.quant)}}}});
    }
    nlohmann::json tensors = nlohmann::json::array();
    for (std::size_t i = 0; i < v.tensors.len; ++i) {
        const auto& t = v.tensors.ptr[i];
        nlohmann::json entry = {{"name", std::string(view_of(t.name))},
                                {"root", t.root},
                                {"shape", shape_to_json(t.shape)}};
        // Which weight a scale belongs to is now the contract's to say, so a
        // dump that omitted it would be blind to the pairing changing — the
        // same reason `repack` above is spelled out.
        if (t.scales.of.len != 0) {
            entry["scales"] = {{"of", std::string(view_of(t.scales.of))},
                               {"granularity", static_cast<std::uint32_t>(t.scales.granularity)},
                               {"group_size", t.scales.group_size},
                               {"channel_axis", t.scales.channel_axis},
                               {"form", static_cast<std::uint32_t>(t.scales.form)}};
        }
        tensors.push_back(std::move(entry));
    }
    // Groups are dumped beside the tensors because the rank-blindness identity
    // is only as wide as what it compares: a streamed contract declares its
    // experts *here* and nowhere in `tensors`, so a group that started folding
    // the rank into its own expression would have gone unseen.
    nlohmann::json groups = nlohmann::json::array();
    for (std::size_t i = 0; i < v.groups.len; ++i) {
        const auto& g = v.groups.ptr[i];
        nlohmann::json members = nlohmann::json::array();
        for (std::size_t t = 0; t < g.tensors.len; ++t) {
            const auto& m = g.tensors.ptr[t];
            members.push_back({{"name", std::string(view_of(m.name))},
                               {"root", m.root},
                               {"shape", shape_to_json(m.shape)}});
        }
        groups.push_back({{"name", std::string(view_of(g.name))},
                          {"arity", g.arity},
                          {"tensors", std::move(members)}});
    }
    return {{"alignment", v.alignment},
            {"nodes", std::move(nodes)},
            {"tensors", std::move(tensors)},
            {"groups", std::move(groups)}};
}

/// Everything that can be checked about a contract without a compiler.
///
/// The loader re-checks all of it and more, but it does so in Rust behind an FFI
/// this binary cannot call; catching a malformed contract here names the
/// authoring bug instead of a diagnostic string.
void check_well_formed(const pie_loader::Checkpoint& checkpoint,
                       const pie_loader::PieLoaderModelContractView& v,
                       std::uint32_t tp) {
    const std::string at = " (tp " + std::to_string(tp) + ")";
    check(v.tensors.len > 0, "the contract declares at least one tensor" + at);

    std::unordered_set<std::string_view> declared;
    for (std::size_t i = 0; i < v.tensors.len; ++i) {
        const auto& t = v.tensors.ptr[i];
        check(t.root < v.nodes.len, "every root is in the node arena" + at);
        check(declared.insert(view_of(t.name)).second,
              "no tensor is declared twice" + at);
    }

    std::unordered_set<std::string_view> published;
    std::size_t sources = 0;
    for (std::size_t i = 0; i < v.nodes.len; ++i) {
        const auto& n = v.nodes.ptr[i];
        const auto kind = static_cast<pie_loader::PieLoaderExprKind>(n.kind);
        // Topological order: an operand is always an earlier node, which is what
        // makes the graph acyclic by construction rather than by a check.
        if (n.src != pie_loader::PIE_LOADER_NO_NODE) {
            check(n.src < i, "an operand precedes its use" + at);
        }
        for (std::size_t p = 0; p < n.parts.len; ++p) {
            check(n.parts.ptr[p] < i, "every concat part precedes the concat" + at);
        }
        if (kind == pie_loader::PieLoaderExprKind::Src) {
            ++sources;
            check(checkpoint.has(view_of(n.name)),
                  std::string("Src '") + std::string(view_of(n.name)) +
                      "' names a tensor the checkpoint has" + at);
        }
    }
    check(sources > 0, "the contract reads the checkpoint" + at);

    // `Out` may only name a declaration that came earlier, which is the rule
    // that lets the resolver run in one pass.
    std::unordered_set<std::string_view> seen_before;
    for (std::size_t i = 0; i < v.tensors.len; ++i) {
        const auto& t = v.tensors.ptr[i];
        std::vector<std::uint32_t> stack{t.root};
        while (!stack.empty()) {
            const auto& n = v.nodes.ptr[stack.back()];
            stack.pop_back();
            if (n.kind == static_cast<std::uint32_t>(pie_loader::PieLoaderExprKind::Out)) {
                check(seen_before.count(view_of(n.name)) != 0,
                      std::string("Out '") + std::string(view_of(n.name)) +
                          "' names an earlier declaration" + at);
            }
            if (n.src != pie_loader::PIE_LOADER_NO_NODE) stack.push_back(n.src);
            for (std::size_t p = 0; p < n.parts.len; ++p) stack.push_back(n.parts.ptr[p]);
        }
        seen_before.insert(view_of(t.name));
    }
    (void)published;
}

/// The TP sizes to sweep. `PIE_TEST_TP_SWEEP=1,3,8` overrides, which is how a
/// tp that divides nothing gets exercised.
std::vector<std::uint32_t> pie_tp_sweep() {
    const char* spec = std::getenv("PIE_TEST_TP_SWEEP");
    if (spec == nullptr) return {1u, 2u, 4u};
    std::vector<std::uint32_t> out;
    std::string text(spec);
    std::size_t start = 0;
    while (start <= text.size()) {
        const std::size_t comma = text.find(',', start);
        const std::string piece = text.substr(start, comma - start);
        if (!piece.empty()) out.push_back(static_cast<std::uint32_t>(std::stoul(piece)));
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return out;
}

/// A canonical rendering of the compiled plan, for comparing two authorings.
///
/// The contract is the wrong thing to diff across an authoring change: a shard
/// may be written as `Shard(Slice(..))` or as a `Slice` with the rank folded
/// into its offset, and those are different graphs that denote the same load.
/// The plan is where the question is answerable, because `compile` has by then
/// evaluated the expression down to byte movement.
nlohmann::json plan_to_json(const pie_loader::LoadPlanView& p) {
    nlohmann::json instrs = nlohmann::json::array();
    for (std::size_t i = 0; i < p.instrs.len; ++i) {
        const auto& in = p.instrs.ptr[i];
        using Tag = pie_loader::PieLoaderStorageOp::Tag;
        nlohmann::json entry{{"tag", static_cast<int>(in.op.tag)}, {"id", in.id}};
        switch (in.op.tag) {
        case Tag::Allocate:
            entry["buffer"] = in.op.allocate.buffer_id;
            break;
        case Tag::Fill:
            entry["buffer"] = in.op.fill.buffer_id;
            break;
        case Tag::ExtentWrite: {
            const auto& op = in.op.extent_write;
            entry["src_tensor"] = op.source.tensor_id;
            entry["src_file"] = op.source.file_id;
            entry["src_offset"] = op.source.file_offset;
            entry["src_span"] = op.source.span_bytes;
            entry["dest_buffer"] = op.dest.buffer_id;
            entry["dest_offset"] = op.dest.offset;
            break;
        }
        case Tag::BulkExtentWrite: {
            const auto& op = in.op.bulk_extent_write;
            entry["src_tensor"] = op.source.tensor_id;
            entry["src_file"] = op.source.file_id;
            entry["src_offset"] = op.source.file_offset;
            entry["src_span"] = op.source.span_bytes;
            entry["dest_offset"] = op.dest_offset;
            break;
        }
        case Tag::TileMap: {
            const auto& op = in.op.tile_map;
            entry["tile_kind"] = static_cast<int>(op.tile_kind);
            entry["rows_per_tile"] = op.rows_per_tile;
            entry["has_source"] = op.has_source;
            entry["src_tensor"] = op.source.tensor_id;
            entry["src_file"] = op.source.file_id;
            entry["src_offset"] = op.source.file_offset;
            entry["src_span"] = op.source.span_bytes;
            entry["has_dest"] = op.has_dest;
            entry["dest_buffer"] = op.dest.buffer_id;
            entry["dest_offset"] = op.dest.offset;
            entry["repack_layout"] = static_cast<int>(op.repack_layout);
            entry["t_batch"] = op.transform_batch;
            entry["t_source_rows"] = op.transform_source_rows;
            entry["t_target_rows"] = op.transform_target_rows;
            entry["t_source_cols"] = op.transform_source_cols;
            entry["t_target_cols"] = op.transform_target_cols;
            break;
        }
        case Tag::CreateView: {
            const auto& op = in.op.create_view;
            entry["input_buffer"] = op.input_buffer;
            entry["output_buffer"] = op.output_buffer;
            entry["dest_offset"] = op.view.offset;
            break;
        }
        case Tag::Finalize:
            entry["buffer"] = in.op.finalize.buffer_id;
            entry["name"] = std::string(view_of(in.op.finalize.name));
            break;
        }
        instrs.push_back(std::move(entry));
    }
    nlohmann::json buffers = nlohmann::json::array();
    for (std::size_t i = 0; i < p.buffers.len; ++i) {
        const auto& b = p.buffers.ptr[i];
        buffers.push_back({{"id", b.id},
                           {"bytes", b.bytes},
                           {"align", b.alignment},
                           {"temporary", b.temporary}});
    }
    nlohmann::json tensors = nlohmann::json::array();
    for (std::size_t i = 0; i < p.tensors.len; ++i) {
        const auto& t = p.tensors.ptr[i];
        nlohmann::json shape = nlohmann::json::array();
        for (std::size_t d = 0; d < t.shape.len; ++d) shape.push_back(t.shape.ptr[d]);
        tensors.push_back(
            {{"name", std::string(view_of(t.name))}, {"shape", std::move(shape)}});
    }
    nlohmann::json schedule = nlohmann::json::array();
    for (std::size_t i = 0; i < p.schedule.len; ++i) schedule.push_back(p.schedule.ptr[i]);
    nlohmann::json attachments = nlohmann::json::array();
    for (std::size_t i = 0; i < p.attachments.len; ++i) {
        const auto& a = p.attachments.ptr[i];
        auto name_of = [&](std::uint32_t id) {
            for (std::size_t t = 0; t < p.tensors.len; ++t)
                if (p.tensors.ptr[t].id == id)
                    return std::string(view_of(p.tensors.ptr[t].name));
            return std::string("<unknown>");
        };
        attachments.push_back(
            {{"tensor", name_of(a.tensor_id)},
             {"scale_tensor", name_of(a.scale_tensor_id)},
             {"granularity", static_cast<int>(a.granularity)},
             {"group_size", a.group_size},
             {"channel_axis", a.channel_axis},
             {"scale_form", static_cast<int>(a.scale_form)}});
    }
    return {{"instrs", std::move(instrs)},
            {"buffers", std::move(buffers)},
            {"tensors", std::move(tensors)},
            {"schedule", std::move(schedule)},
            {"attachments", std::move(attachments)},
            {"read_bytes", p.memory.checkpoint_read_bytes},
            {"write_bytes", p.memory.device_write_bytes},
            {"persistent_bytes", p.memory.persistent_bytes}};
}

/// Authors the snapshot's family at every rank of several TP sizes and, if
/// asked, writes the result to `PIE_TEST_CONTRACT_DUMP`.
///
/// Every rank, not just rank 0. An authoring bug in the rank term is invisible
/// at rank 0, where it is multiplied by zero — which is what every version of
/// this test used to check and nothing else did.
void author_real_contract(const std::string& snapshot, const char* dest) {
    const auto cfg = std::filesystem::path(snapshot) / "config.json";
    const pie_cuda_driver::HfConfig hf = pie_cuda_driver::parse_hf_config(cfg);

    std::string open_error;
    pie_loader::Checkpoint checkpoint = pie_loader::Checkpoint::open(snapshot, &open_error);
    check(static_cast<bool>(checkpoint), "the snapshot opens: " + open_error);
    if (!checkpoint) return;

    const pie_cuda_driver::model::ModelFacts facts{
        .model_type = hf.model_type,
        .quant_method = hf.quant_method,
        .num_hidden_layers = static_cast<std::uint32_t>(std::max(0, hf.num_hidden_layers)),
        .num_experts = static_cast<std::uint32_t>(std::max(0, hf.num_experts)),
        .head_dim = static_cast<std::uint32_t>(std::max(0, hf.head_dim)),
        .mamba_groups = static_cast<std::uint32_t>(std::max(0, hf.mamba_n_groups)),
    };

    namespace model = pie_cuda_driver::model;
    const model::ArchEntry* arch = model::find_arch_entry(hf.model_type);
    check(arch != nullptr && static_cast<bool>(arch->author_contract),
          "the arch table has a row for model_type '" + hf.model_type + "'");
    if (arch == nullptr || !arch->author_contract) return;

    nlohmann::json by_tp = nlohmann::json::object();
    for (std::uint32_t tp : pie_tp_sweep()) {
        nlohmann::json by_rank = nlohmann::json::object();
        for (std::uint32_t rank = 0; rank < tp; ++rank) {
            auto target = pie_cuda_driver::cuda_device_target();
            target.tp_size = tp;
            target.tp_rank = rank;
            // The native-MXFP4 repack path is the one place a rank cannot be
            // deferred, so it needs reaching on a GPU that does not offer it.
            if (std::getenv("PIE_TEST_NATIVE_MXFP4") != nullptr) {
                target.native_mxfp4_moe = true;
            }

            const std::string at =
                " (tp " + std::to_string(tp) + " rank " + std::to_string(rank) + ")";

            pie_loader::ModelContract contract;
            try {
                model::ContractBuilder builder(
                    checkpoint, facts, target, "", model::Mxfp4MoeRequest::Auto,
                    model::Component::Full, /*stream_routed_experts=*/false, contract);
                arch->author_contract(builder);
                builder.finish();
            } catch (const std::exception& error) {
                check(false, "authoring" + at + ": " + error.what());
                continue;
            }
            const auto v = contract.view();
            check_well_formed(checkpoint, v, tp);

            // Compiling is what makes the rank sweep worth running: two ranks
            // may author the same graph and still disagree once `Shard` is
            // specialized, and nothing above this line would notice.
            nlohmann::json entry = {{"contract", contract_to_json(v)}};
            const pie_loader::PieLoaderContractRequest request =
                pie_loader::build_contract_request(checkpoint, target, v);
            try {
                const pie_loader::LoadPlan plan = pie_loader::LoadPlan::compile(request);
                plan.verify(request);
                entry["plan"] = plan_to_json(plan.view());
            } catch (const std::exception& error) {
                check(false, "compiling" + at + ": " + error.what());
                continue;
            }
            by_rank[std::to_string(rank)] = std::move(entry);
        }
        // §6.3 lets the leader own `TargetSpec` because it is the only per-rank
        // input to compilation. That holds only if authoring is rank-blind, and
        // for a while it was not: `local_range` put `tp_rank * local` into a
        // `Slice` offset. `Expr::Shard` says the same thing without the rank, so
        // every affine site can be written that way and the contract comes out
        // identical on every rank.
        //
        // `Repack` used to be an exception -- it carried `source_row_offset` as
        // an integer a kernel reads, so there was nowhere to hang a `Shard` --
        // and this check had to let a repacking family through. It does not any
        // more: a repack names a layout and takes an expression, so the shard
        // goes where every other family puts it. There is no exception left.
        if (by_rank.size() > 1 && by_rank.contains("0")) {
            const auto& first = by_rank["0"]["contract"];
            for (const auto& [rank, entry] : by_rank.items()) {
                check(entry["contract"] == first,
                      "authoring does not read the rank: tp " + std::to_string(tp) +
                          " rank " + rank + " must author what rank 0 does");
            }
        }
        by_tp[std::to_string(tp)] = std::move(by_rank);
    }

    std::cout << "authored "
              << by_tp["1"]["0"]["contract"]["tensors"].size() << " tensor(s) for "
              << hf.model_type << "\n";

    // Set-but-empty is how a caller sweeping many snapshots turns the dump off
    // for the big ones, and `std::ofstream("")` fails, so it has to mean "no".
    if (dest != nullptr && *dest != '\0') {
        nlohmann::json doc = {{"model_type", hf.model_type}, {"by_tp_size", std::move(by_tp)}};
        std::ofstream out(dest);
        check(out.good(), "contract dump destination is writable");
        out << doc.dump(2) << "\n";
        std::cout << "wrote " << dest << "\n";
    }
}

}  // namespace

/// Does the expression published under `name` contain an `Expr::Shard`?
///
/// Walks the whole subtree, not just the spine, because the shard sits under a
/// `Stride` for the interleaved halves and under nothing for the down columns.
bool expr_contains_shard(const nlohmann::json& contract, const std::string& name) {
    const auto& nodes = contract["nodes"];
    std::int64_t root = -1;
    for (const auto& tensor : contract["tensors"]) {
        if (tensor["name"] == name) root = tensor["root"].get<std::int64_t>();
    }
    if (root < 0) return false;

    std::vector<std::int64_t> stack{root};
    while (!stack.empty()) {
        const std::int64_t at = stack.back();
        stack.pop_back();
        if (at < 0 || at >= static_cast<std::int64_t>(nodes.size())) continue;
        const auto& node = nodes[static_cast<std::size_t>(at)];
        if (node["kind"] ==
            static_cast<std::uint32_t>(pie_loader::PieLoaderExprKind::Shard)) {
            return true;
        }
        stack.push_back(node["src"].get<std::int64_t>());
        for (const auto& part : node["parts"]) stack.push_back(part.get<std::int64_t>());
    }
    return false;
}

/// The escape hatch's last rank.
///
/// A slot holds this rank's share, and only where the resident path sharded too.
///
/// Streaming is a residency decision, so a group under TP has to declare
/// whatever the resident path would have declared on that rank -- no more and
/// no less. The two streamed families sit on opposite sides of that:
///
///   - Qwen stores a tensor per expert, and the family's TP path bands the
///     intermediate axis. The group shards each half before the gate/up join,
///     so a slot is priced at `2*(I/W)*H + H*(I/W)` -- half of what one rank
///     holds at `tp=1`.
///   - GPT-OSS packed publishes its expert banks with `push_direct`, unsharded:
///     every rank holds every expert and the MoE is replicated. A group that
///     took a band here would *not* match what it replaces, so its slot stays
///     full width.
///
/// Both are rank-blind in the contract: a shard names an axis, and
/// `Resolver::specialize` is the one pass that knows which rank is asking.
void test_streamed_expert_groups_follow_the_resident_tp_layout() {
    constexpr std::int64_t kExperts = 4;
    constexpr std::int64_t kHidden = 64;
    constexpr std::int64_t kInter = 32;
    constexpr std::int64_t kLayers = 2;
    constexpr std::uint32_t kWorld = 2;
    constexpr std::int64_t kLocalInter = kInter / kWorld;
    namespace model = pie_cuda_driver::model;

    std::vector<TypedTensor> common{
        {"model.embed_tokens.weight", {16, kHidden}, "BF16"},
        {"model.norm.weight", {kHidden}, "BF16"},
        {"lm_head.weight", {16, kHidden}, "BF16"},
    };
    std::vector<TypedTensor> qwen = common;
    std::vector<TypedTensor> oss = common;
    for (int layer = 0; layer < kLayers; ++layer) {
        const std::string lp = "model.layers." + std::to_string(layer) + ".mlp.";
        qwen.push_back({lp + "gate.weight", {kExperts, kHidden}, "BF16"});
        for (int e = 0; e < kExperts; ++e) {
            const std::string ep = lp + "experts." + std::to_string(e) + ".";
            qwen.push_back({ep + "gate_proj.weight", {kInter, kHidden}, "BF16"});
            qwen.push_back({ep + "up_proj.weight", {kInter, kHidden}, "BF16"});
            qwen.push_back({ep + "down_proj.weight", {kHidden, kInter}, "BF16"});
        }
        const std::string ep = lp + "experts.";
        oss.push_back({ep + "gate_up_proj_blocks",
                       {kExperts, 2 * kInter, kHidden / 32, 16}, "U8"});
        oss.push_back({ep + "gate_up_proj_scales",
                       {kExperts, 2 * kInter, kHidden / 32}, "U8"});
        oss.push_back({ep + "gate_up_proj_bias", {kExperts, 2 * kInter}, "BF16"});
        oss.push_back({ep + "down_proj_blocks",
                       {kExperts, kHidden, kInter / 32, 16}, "U8"});
        oss.push_back({ep + "down_proj_scales",
                       {kExperts, kHidden, kInter / 32}, "U8"});
        oss.push_back({ep + "down_proj_bias", {kExperts, kHidden}, "BF16"});
    }

    // What one slot is worth on a rank. Spelled from the shapes rather than
    // from a measured number, so a layout that quietly changed could not drag
    // the expectation along with it.
    struct Family {
        const char* label;
        const char* model_type;
        const std::vector<TypedTensor>* tensors;
        model::Mxfp4MoeRequest mxfp4;
        void (*author)(model::ContractBuilder&);
        const char* gate_up_name;
        const char* down_name;
        std::vector<std::int64_t> gate_up;
        std::vector<std::int64_t> down;
        std::uint64_t slot_bytes;
    };
    const Family families[] = {
        {"qwen3_moe", "qwen3_moe", &qwen, model::Mxfp4MoeRequest::Auto,
         &model::author_qwen3_5_moe_contract, "gate_up_proj", "down_proj",
         {2 * kLocalInter, kHidden}, {kHidden, kLocalInter},
         static_cast<std::uint64_t>(
             (2 * kLocalInter * kHidden + kHidden * kLocalInter) * 2)},
        {"gpt_oss", "gpt_oss", &oss, model::Mxfp4MoeRequest::RoutedDecode,
         &model::author_gpt_oss_contract, "gate_up_proj.weight", "down_proj.weight",
         {1, 2 * kInter, kHidden / 32, 16}, {1, kHidden, kInter / 32, 16},
         static_cast<std::uint64_t>(
             2 * kInter * (kHidden / 32) * 16 + 2 * kInter * (kHidden / 32) +
             kHidden * (kInter / 32) * 16 + kHidden * (kInter / 32))},
    };

    for (const Family& family : families) {
        const std::filesystem::path dir = write_typed_checkpoint(
            std::string("tp_streamed_") + family.label, *family.tensors);
        std::string open_error;
        pie_loader::Checkpoint checkpoint =
            pie_loader::Checkpoint::open(dir.string(), &open_error);
        check(static_cast<bool>(checkpoint), std::string(family.label) +
                  ": the synthetic checkpoint opens: " + open_error);
        if (!checkpoint) continue;

        const model::ModelFacts facts{
            .model_type = family.model_type,
            .quant_method = "",
            .num_hidden_layers = kLayers,
            .num_experts = kExperts,
            .head_dim = 0,
            .mamba_groups = 0,
        };

        nlohmann::json first_contract;
        for (std::uint32_t rank = 0; rank < kWorld; ++rank) {
            const std::string at = std::string(" (") + family.label + " rank " +
                                   std::to_string(rank) + " of " +
                                   std::to_string(kWorld) + ")";
            auto target = pie_cuda_driver::cuda_device_target();
            target.tp_size = kWorld;
            target.tp_rank = rank;

            pie_loader::ModelContract contract;
            try {
                model::ContractBuilder builder(checkpoint, facts, target, "",
                                               family.mxfp4, model::Component::Full,
                                               /*stream_routed_experts=*/true, contract);
                family.author(builder);
                builder.finish();
            } catch (const std::exception& error) {
                check(false, "authoring" + at + ": " + error.what());
                continue;
            }

            const auto v = contract.view();
            check(v.groups.len == kLayers, "one group per layer" + at);
            if (v.groups.len != kLayers) continue;
            for (std::size_t layer = 0; layer < v.groups.len; ++layer) {
                std::map<std::string_view, std::vector<std::int64_t>> declared;
                const auto& g = v.groups.ptr[layer];
                for (std::size_t i = 0; i < g.tensors.len; ++i) {
                    const auto& t = g.tensors.ptr[i];
                    declared[view_of(t.name)] = std::vector<std::int64_t>(
                        t.shape.ptr, t.shape.ptr + t.shape.len);
                }
                check(declared[family.gate_up_name] == family.gate_up,
                      "gate/up is declared at this rank's width" + at);
                check(declared[family.down_name] == family.down,
                      "down is declared at this rank's width" + at);
            }

            const nlohmann::json dumped = contract_to_json(v);
            if (rank == 0) {
                first_contract = dumped;
            } else {
                check(dumped == first_contract,
                      "every rank authors the same contract" + at);
            }

            try {
                const pie_loader::PieLoaderContractRequest request =
                    pie_loader::build_contract_request(checkpoint, target, v);
                const pie_loader::LoadPlan plan = pie_loader::LoadPlan::compile(request);
                plan.verify(request);
                const auto& pv = plan.view();
                check(pv.groups.len == kLayers, "the plan carries every group" + at);
                for (std::size_t layer = 0; layer < pv.groups.len; ++layer) {
                    const auto& pg = pv.groups.ptr[layer];
                    check(pg.arity == kExperts, "the plan agrees on the arity" + at);
                    check(pg.plan->memory.persistent_bytes >= family.slot_bytes &&
                              pg.plan->memory.persistent_bytes <
                                  family.slot_bytes + 4096,
                          "a slot is priced at this rank's share: expected " +
                              std::to_string(family.slot_bytes) + ", got " +
                              std::to_string(pg.plan->memory.persistent_bytes) + at);
                }
            } catch (const std::exception& error) {
                check(false, std::string("compiling") + at + ": " + error.what());
            }
        }
    }
}

/// GPT-OSS's native MXFP4 path was the only place a `tp_rank` reached a
/// contract: a repack spec carried `source_row_offset` as an integer, so the
/// driver computed `rank * local` while authoring and the contract was valid for
/// exactly one rank. The cross-rank check in `author_real_contract` had to let
/// that through, and it only runs against a real snapshot anyway -- so the one
/// family that broke the rule was also the one nothing checked by default.
///
/// The selection is `Shard` and `Stride` now, so this asserts the rule with no
/// exception: every rank authors byte-identical bytes, and the rank still
/// reaches the *plan*, which is where it belongs.
void test_gpt_oss_native_repack_is_rank_blind() {
    constexpr std::int64_t kExperts = 2;
    constexpr std::int64_t kHidden = 64;
    constexpr std::int64_t kInterFull = 128;
    constexpr std::int64_t kHiddenGroups = kHidden / 32;
    constexpr std::int64_t kInterGroups = kInterFull / 32;

    const std::string ep = "model.layers.0.mlp.experts.";
    const std::vector<TypedTensor> tensors{
        {"model.embed_tokens.weight", {16, kHidden}, "BF16"},
        {"model.norm.weight", {kHidden}, "BF16"},
        {"lm_head.weight", {16, kHidden}, "BF16"},
        // gate and up are interleaved row by row on the fused axis.
        {ep + "gate_up_proj_blocks", {kExperts, 2 * kInterFull, kHiddenGroups, 16}, "U8"},
        {ep + "gate_up_proj_scales", {kExperts, 2 * kInterFull, kHiddenGroups}, "U8"},
        {ep + "gate_up_proj_bias", {kExperts, 2 * kInterFull}, "BF16"},
        {ep + "down_proj_blocks", {kExperts, kHidden, kInterGroups, 16}, "U8"},
        {ep + "down_proj_scales", {kExperts, kHidden, kInterGroups}, "U8"},
        {ep + "down_proj_bias", {kExperts, kHidden}, "BF16"},
    };
    const std::filesystem::path dir = write_typed_checkpoint("gpt_oss_native_mxfp4", tensors);

    std::string open_error;
    pie_loader::Checkpoint checkpoint = pie_loader::Checkpoint::open(dir.string(), &open_error);
    check(static_cast<bool>(checkpoint), "the synthetic checkpoint opens: " + open_error);
    if (!checkpoint) return;

    const pie_cuda_driver::model::ModelFacts facts{
        .model_type = "gpt_oss",
        .quant_method = "mxfp4",
        .num_hidden_layers = 1,
        .num_experts = kExperts,
        .head_dim = 0,
        .mamba_groups = 0,
    };
    namespace model = pie_cuda_driver::model;

    for (int tp : {1, 2}) {
        nlohmann::json first_contract;
        std::vector<nlohmann::json> reads;
        for (int rank = 0; rank < tp; ++rank) {
            const std::string at =
                " (tp " + std::to_string(tp) + " rank " + std::to_string(rank) + ")";
            auto target = pie_cuda_driver::cuda_device_target();
            target.tp_size = static_cast<std::uint32_t>(tp);
            target.tp_rank = static_cast<std::uint32_t>(rank);
            target.native_mxfp4_moe = true;

            pie_loader::ModelContract contract;
            try {
                model::ContractBuilder builder(checkpoint, facts, target, "",
                                               model::Mxfp4MoeRequest::NativeGemm,
                                               model::Component::Full,
                                               /*stream_routed_experts=*/false, contract);
                model::author_gpt_oss_contract(builder);
                builder.finish();
            } catch (const std::exception& error) {
                check(false, "authoring" + at + ": " + error.what());
                return;
            }
            const auto v = contract.view();
            const nlohmann::json dumped = contract_to_json(v);

            if (rank == 0) {
                // Without this the identity below would pass on a contract that
                // never took the native path at all.
                int repacks = 0;
                for (const auto& node : dumped["nodes"]) {
                    if (node["kind"] == static_cast<std::uint32_t>(
                                            pie_loader::PieLoaderExprKind::Repack)) {
                        ++repacks;
                    }
                }
                check(repacks == 6,
                      "the native path authors a weight and a scale repack for gate, up "
                      "and down" + at + ": got " + std::to_string(repacks));
                first_contract = dumped;

                // Each sharded tensor must carry the shard in its *own*
                // expression. Comparing whole plans is not enough: down_proj's
                // column shard alone makes two ranks read different bytes, so a
                // gate_up that quietly stopped sharding would still pass.
                if (tp > 1) {
                    for (const char* leaf : {"gate_proj.weight", "gate_proj.bias",
                                             "up_proj.weight_scale", "down_proj.weight"}) {
                        check(expr_contains_shard(dumped, ep + leaf),
                              std::string(leaf) + " selects this rank's band with a Shard" + at);
                    }
                }
            } else {
                check(dumped == first_contract,
                      "every rank authors the same contract" + at);
            }

            const pie_loader::PieLoaderContractRequest request =
                pie_loader::build_contract_request(checkpoint, target, v);
            try {
                const pie_loader::LoadPlan plan = pie_loader::LoadPlan::compile(request);
                plan.verify(request);
                nlohmann::json sources = nlohmann::json::array();
                const nlohmann::json dumped_plan = plan_to_json(plan.view());
                for (const auto& instr : dumped_plan["instrs"]) {
                    if (instr.contains("src_tensor")) {
                        sources.push_back({instr["src_tensor"], instr["src_offset"],
                                           instr["src_span"]});
                    }
                }
                std::sort(sources.begin(), sources.end(),
                          [](const nlohmann::json& a, const nlohmann::json& b) {
                              return a.dump() < b.dump();
                          });
                reads.push_back(std::move(sources));
            } catch (const std::exception& error) {
                check(false, "compiling" + at + ": " + error.what());
                return;
            }
        }
        // ...and the rank is not lost, it moved: the target is what selects the
        // band now, so two ranks must read two different sets of bytes.
        if (tp > 1) {
            check(!reads[0].empty(), "the plan reads the checkpoint at all");
            check(reads[0] != reads[1],
                  "tp " + std::to_string(tp) + " ranks read different bytes");
        }
    }
}

int main() {
    test_a_handle_names_one_node();
    test_expect_states_a_shape();
    test_scaling_states_the_weight();
    test_scale_carries_its_factor();
    test_the_view_survives_growth();
    test_a_group_declares_its_own_tensors();
    test_a_group_view_survives_growth();
    test_csm_narrows_fp32_weights_to_bf16();
    test_deepseek_v4_stacks_experts_as_bf16();
    test_deepseek_v4_streams_experts_as_a_group();
    test_kimi_stacks_experts_as_bf16();
    test_nemotron_h_mamba_splits_on_unit_boundaries();
    test_qwen3_5_gdn_splits_by_block();
    test_qwen3_5_mtp_int8_lm_head();
    test_gpt_oss_native_repack_is_rank_blind();
    test_qwen3_5_moe_joins_the_shared_expert();
    test_glm5_dequantizes_kv_b_proj();
    test_qwen3_moe_streams_experts_as_a_group();
    test_gpt_oss_streams_experts_as_a_group();
    test_streamed_expert_groups_follow_the_resident_tp_layout();

    const char* snapshot = std::getenv("PIE_TEST_SNAPSHOT");
    if (snapshot != nullptr) {
        author_real_contract(snapshot, std::getenv("PIE_TEST_CONTRACT_DUMP"));
    }

    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "model_contract_test: OK\n";
    return 0;
}

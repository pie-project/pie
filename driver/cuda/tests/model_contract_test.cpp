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

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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

#include "model/contract.hpp"
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
    // Naming `q` three times must build one `Src`, not three: a `cat` of three
    // slices of one tensor has to read that tensor once, and the compiler
    // decides that by node identity.
    c.define("qkv", c.cat({q, q, q}, 0), pie_loader::raw(pie_loader::PieLoaderDType::BF16));

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
    check(sc.granularity == pie_loader::PieLoaderQuantGranularity::PerGroup &&
              sc.form == pie_loader::PieLoaderScaleForm::F32Factors,
          "the stated granularity and form come back");
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
                         {"before", n.before},
                         {"after", n.after},
                         {"shape", shape_to_json(n.shape)},
                         {"out_shape", shape_to_json(n.out_shape)},
                         // A `Repack` carries its offsets as integers, so this
                         // is where a `tp_rank` can still be baked into a
                         // contract. Omitting it made the cross-rank diff below
                         // blind to the one case it must not miss.
                         {"repack", {{"layout", n.repack.layout},
                                     {"row_map", n.repack.row_map},
                                     {"batch", n.repack.batch},
                                     {"source_rows", n.repack.source_rows},
                                     {"source_row_offset", n.repack.source_row_offset},
                                     {"target_rows", n.repack.target_rows},
                                     {"valid_rows", n.repack.valid_rows},
                                     {"source_stride_cols", n.repack.source_stride_cols},
                                     {"source_col_offset", n.repack.source_col_offset},
                                     {"source_cols", n.repack.source_cols},
                                     {"target_cols", n.repack.target_cols}}},
                         {"quant", quant_to_json(n.quant)},
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
    return {{"alignment", v.alignment},
            {"nodes", std::move(nodes)},
            {"tensors", std::move(tensors)}};
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
            check(n.parts.ptr[p] < i, "every cat part precedes the cat" + at);
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
            entry["t_source_row_offset"] = op.transform_source_row_offset;
            entry["t_target_rows"] = op.transform_target_rows;
            entry["t_valid_rows"] = op.transform_valid_rows;
            entry["t_source_col_offset"] = op.transform_source_col_offset;
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
                    checkpoint, facts, target, "",
                    model::resolve_mxfp4_moe(model::Mxfp4MoeRequest::Auto,
                                             target.native_mxfp4_moe),
                    model::Component::Full, contract);
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
        // `Repack` is the exception and cannot be one of these by accident: it
        // carries `source_row_offset` as an integer a kernel reads, so there is
        // nowhere to hang a `Shard`. Naming the exception this precisely is the
        // point — anything else that starts baking a rank fails here.
        if (by_rank.size() > 1 && by_rank.contains("0")) {
            const auto& first = by_rank["0"]["contract"];
            bool repacks = false;
            for (const auto& node : first["nodes"]) {
                if (node["kind"] ==
                    static_cast<std::uint32_t>(pie_loader::PieLoaderExprKind::Repack)) repacks = true;
            }
            for (const auto& [rank, entry] : by_rank.items()) {
                const bool same = entry["contract"] == first;
                check(same || repacks,
                      "authoring does not read the rank: tp " + std::to_string(tp) +
                          " rank " + rank + " authors what rank 0 does, or repacks");
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

int main() {
    test_a_handle_names_one_node();
    test_expect_states_a_shape();
    test_scaling_states_the_weight();
    test_the_view_survives_growth();

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

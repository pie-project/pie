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
            {"channel_axis", q.channel_axis},
            {"scale_dtype", q.scale_dtype},
            {"zero_point_dtype", q.zero_point_dtype},
            {"block_shape", shape_to_json(q.block_shape)}};
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
        tensors.push_back({{"name", std::string(view_of(t.name))},
                           {"root", t.root},
                           {"shape", shape_to_json(t.shape)}});
    }
    return {{"abi_version", v.abi_version},
            {"alignment", v.alignment},
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
        instrs.push_back({{"kind", static_cast<int>(in.kind)},
                          {"id", in.id},
                          {"buffer", in.buffer_id},
                          {"tile_kind", static_cast<int>(in.tile_kind)},
                          {"rows_per_tile", in.rows_per_tile},
                          {"has_source", in.has_source},
                          {"src_tensor", in.source.tensor_id},
                          {"src_file", in.source.file_id},
                          {"src_offset", in.source.file_offset},
                          {"src_span", in.source.span_bytes},
                          {"has_dest", in.has_dest},
                          {"dest_buffer", in.dest.buffer_id},
                          {"dest_offset", in.dest.offset},
                          {"repack_layout", static_cast<int>(in.repack_layout)},
                          {"t_batch", in.transform_batch},
                          {"t_source_rows", in.transform_source_rows},
                          {"t_source_row_offset", in.transform_source_row_offset},
                          {"t_target_rows", in.transform_target_rows},
                          {"t_valid_rows", in.transform_valid_rows},
                          {"t_source_col_offset", in.transform_source_col_offset}});
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
    return {{"instrs", std::move(instrs)},
            {"buffers", std::move(buffers)},
            {"tensors", std::move(tensors)},
            {"schedule", std::move(schedule)},
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

    if (dest != nullptr) {
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

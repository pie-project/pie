// m1_codegen_dump.cpp — TRANSITIONAL oracle for the Metal MSL emitters.
//
// This harness links the C++ emitter (`driver/metal/src/pipeline/m1_codegen.cpp`)
// and dumps every one of its ten public entry points over a fixed corpus, in a
// deterministic diffable text format. The dump is checked in under
// `compiler/tests/golden-msl/`; the Rust port in `compiler/codegen/src/metal/`
// reproduces it byte for byte (`compiler/tests/tests/metal_msl_golden.rs`).
//
// DELETE THIS FILE when the C++ emitter is deleted. Nothing links it into a
// build; it exists so a human can re-derive the goldens from the C++ oracle and
// see an empty `git diff`.
//
// Build + run (from the repository root):
//
//   g++ -std=c++20 -O1 -o /path/to/scratch/m1_codegen_dump \
//     compiler/tests/oracle/m1_codegen_dump.cpp \
//     driver/metal/src/pipeline/m1_codegen.cpp \
//     -I driver/metal/src -I driver/common/include -I compiler/codegen/include
//   /path/to/scratch/m1_codegen_dump . compiler/tests/golden-msl
//
// argv[1] is the repository root, argv[2] the output directory.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "pie_native/ptir/plan.hpp"
#include "pipeline/m1_codegen.hpp"

using namespace pie::metal::pipeline;
namespace plan = pie_native::ptir::plan;

namespace {

// ── shared dump plumbing ───────────────────────────────────────────────────

std::string read_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        std::fprintf(stderr, "cannot read %s\n", path.c_str());
        std::exit(1);
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

void write_file(const std::string& path, const std::string& text) {
    std::ofstream output(path, std::ios::binary);
    output << text;
}

std::string hex64(std::uint64_t value) {
    std::ostringstream stream;
    stream << std::hex << std::setfill('0') << std::setw(16) << value;
    return stream.str();
}

std::vector<std::uint8_t> unhex(const std::string& text) {
    std::vector<std::uint8_t> bytes;
    bytes.reserve(text.size() / 2);
    for (std::size_t index = 0; index + 1 < text.size(); index += 2) {
        bytes.push_back(static_cast<std::uint8_t>(
            std::stoul(text.substr(index, 2), nullptr, 16)));
    }
    return bytes;
}

// The dump factors two constant fragments out of every emitted source: the
// 34 KB runtime template and the grouped struct preamble. Both are pinned by
// length + hash in each file's header, so nothing is lost.
std::string g_runtime;          // runtime template + "\n"
std::string g_grouped_prefix;   // runtime template + "\n" + grouped preamble

struct Dump {
    std::string emitter;
    std::ostringstream body;
    std::size_t cases = 0;

    void open_case(const std::string& id) {
        ++cases;
        body << "=== " << id << "\n";
    }

    void field(const std::string& key, const std::string& value) {
        body << key << ": " << value << "\n";
    }

    void error(const std::string& message) {
        body << "ok: false\n";
        body << "error: " << message << "\n";
        body << "=== end\n";
    }

    void source(const std::string& text) {
        body << "ok: true\n";
        std::string tail = text;
        const char* prefix = "none";
        if (!g_grouped_prefix.empty() && tail.rfind(g_grouped_prefix, 0) == 0) {
            prefix = "@runtime+@grouped";
            tail = tail.substr(g_grouped_prefix.size());
        } else if (!g_runtime.empty() && tail.rfind(g_runtime, 0) == 0) {
            prefix = "@runtime";
            tail = tail.substr(g_runtime.size());
        }
        body << "source: bytes=" << text.size() << " prefix=" << prefix
             << " tail_bytes=" << tail.size() << "\n";
        body << tail;
        if (!tail.empty() && tail.back() != '\n') body << "\n\\no-trailing-newline\n";
        body << "=== end\n";
    }
};

std::uint64_t fnv1a64(const std::string& text) {
    // 0xcbf29ce484222325 — the same basis as `pie_ir::program_hash`.
    std::uint64_t hash = 14695981039346656037ull;
    for (const char byte : text) {
        hash ^= static_cast<std::uint8_t>(byte);
        hash *= 1099511628211ull;
    }
    return hash;
}

void finish(const std::string& directory, Dump& dump) {
    std::ostringstream head;
    head << "# oracle dump: pie::metal::pipeline::" << dump.emitter << "\n";
    head << "# source of truth: driver/metal/src/pipeline/m1_codegen.cpp"
         << " (transitional; the Rust port is compiler/codegen/src/metal/)\n";
    head << "# emitter_version: " << kMetalM1EmitterVersion << "\n";
    head << "# @runtime: bytes=" << g_runtime.size()
         << " fnv1a64=0x" << hex64(fnv1a64(g_runtime)) << "\n";
    head << "# @grouped: bytes=" << (g_grouped_prefix.size() - g_runtime.size())
         << " fnv1a64=0x"
         << hex64(fnv1a64(g_grouped_prefix.substr(g_runtime.size()))) << "\n";
    head << "# cases: " << dump.cases << "\n";
    write_file(directory + "/" + dump.emitter + ".txt", head.str() + dump.body.str());
    std::printf("  %-32s %4zu cases\n", dump.emitter.c_str(), dump.cases);
}

// ── channel-effect corpus ──────────────────────────────────────────────────

M1ChannelEffect effect_from_flags(unsigned flags, std::uint32_t capacity) {
    M1ChannelEffect effect;
    effect.requires_full = (flags & 1u) != 0u;
    effect.requires_empty = (flags & 2u) != 0u;
    effect.take = (flags & 4u) != 0u;
    effect.put = (flags & 8u) != 0u;
    effect.capacity = capacity;
    return effect;
}

struct EffectCase {
    std::string id;
    std::vector<M1ChannelEffect> channels;
};

std::vector<EffectCase> effect_cases() {
    std::vector<EffectCase> cases;
    cases.push_back({"empty", {}});
    for (unsigned flags = 0; flags < 16u; ++flags) {
        std::ostringstream id;
        id << "flags_" << flags;
        cases.push_back({id.str(), {effect_from_flags(flags, 1)}});
    }
    const std::uint32_t capacities[] = {0u, 1u, 2u, 3u, 7u, 64u, 65535u, 4294967295u};
    for (const std::uint32_t capacity : capacities) {
        std::ostringstream id;
        id << "capacity_" << capacity;
        cases.push_back({id.str(), {effect_from_flags(13u, capacity)}});
    }
    for (unsigned flags = 0; flags < 16u; ++flags) {
        std::ostringstream id;
        id << "pair_" << flags;
        cases.push_back({
            id.str(),
            {
                effect_from_flags(flags, 1),
                effect_from_flags(15u - flags, 3),
            },
        });
    }
    cases.push_back({
        "mix5",
        {
            effect_from_flags(1u, 1),
            effect_from_flags(8u, 2),
            effect_from_flags(12u, 4),
            effect_from_flags(2u, 8),
            effect_from_flags(15u, 16),
        },
    });
    for (const std::size_t count : {kMetalM1MaxChannels - 1,
                                    kMetalM1MaxChannels,
                                    kMetalM1MaxChannels + 1}) {
        std::vector<M1ChannelEffect> channels;
        for (std::size_t channel = 0; channel < count; ++channel) {
            channels.push_back(effect_from_flags(
                static_cast<unsigned>((channel * 7u + 1u) % 16u),
                static_cast<std::uint32_t>(channel + 1)));
        }
        std::ostringstream id;
        id << "wide_" << count;
        cases.push_back({id.str(), channels});
    }
    return cases;
}

// ── stage-plan corpus ──────────────────────────────────────────────────────

struct CorpusStage {
    std::string id;
    plan::StagePlan plan;
};

std::vector<CorpusStage> read_corpus(const std::string& path) {
    std::vector<CorpusStage> stages;
    std::ifstream input(path);
    if (!input) {
        std::fprintf(stderr, "cannot read %s\n", path.c_str());
        std::exit(1);
    }
    std::string line;
    while (std::getline(input, line)) {
        if (line.rfind("plan: ", 0) != 0) continue;
        auto field = [&](const std::string& key) {
            const std::size_t at = line.find(" " + key + "=");
            const std::size_t begin = at + key.size() + 2;
            const std::size_t end = line.find(' ', begin);
            return line.substr(
                begin, end == std::string::npos ? std::string::npos : end - begin);
        };
        const std::string id = field("id");
        const std::vector<std::uint8_t> bytes = unhex(field("bytes"));
        CorpusStage stage;
        stage.id = id;
        std::string error;
        if (!plan::decode(bytes.data(), bytes.size(), stage.plan, &error)) {
            std::fprintf(stderr, "corpus plan %s failed to decode: %s\n",
                         id.c_str(), error.c_str());
            std::exit(1);
        }
        stages.push_back(std::move(stage));
    }
    if (stages.empty()) {
        std::fprintf(stderr, "corpus %s is empty\n", path.c_str());
        std::exit(1);
    }
    return stages;
}

// `validate_singleton_plan` rejects things a well-formed compiler plan never
// produces, and `plan::decode` already range-checks most of them, so its error
// paths are only reachable by damaging a decoded plan. These mutations are
// applied identically on the Rust side.
const char* const kMutations[] = {
    "none",
    "zero_signature_hash",
    "flip_signature_byte",
    "singleton_kind",
    "whole_stage_fallback",
    "drop_last_singleton_region",
    "singleton_region_node",
    "swap_singleton_region_nodes",
    "unordered_region_nodes",
    "region_input_out_of_range",
    "region_output_out_of_range",
    "region_sink_out_of_range",
    "library_scan_claim",
    "extra_value_type",
    "drop_last_value_type",
    "drop_last_value_type_and_refs",
    "rank5_value_type",
    "zero_static_dim",
    "overflow_static_dims",
    "reverse_ops",
    "bad_channel_slot",
    "pivot_payload_out_of_range",
    "rename_names",
    "clear_names",
};
// Deliberately absent: mutations for the "invalid symbolic extent role",
// "invalid normalized value type" (bad dtype) and "unsupported singleton op"
// branches. All three are reachable off the wire but unrepresentable in the
// Rust plan types, where `SymbolicExtent`, `DType` and `Op` are closed enums
// whose variants are exactly the legal roles / dtypes / known op tags. Adding
// them here would make the dump untranslatable.

// Returns false when the mutation does not apply to this plan.
bool mutate(plan::StagePlan& stage, const std::string& mutation) {
    if (mutation == "none") return true;
    if (mutation == "zero_signature_hash") {
        if (stage.signature_hash == 0) return false;
        stage.signature_hash = 0;
        return true;
    }
    if (mutation == "flip_signature_byte") {
        if (stage.signature.empty()) return false;
        stage.signature[0] ^= 1u;
        return true;
    }
    if (mutation == "singleton_kind") {
        stage.singleton.kind = 1;
        return true;
    }
    if (mutation == "whole_stage_fallback") {
        stage.singleton.whole_stage_fallback = true;
        return true;
    }
    if (mutation == "drop_last_singleton_region") {
        if (stage.singleton.regions.empty()) return false;
        stage.singleton.regions.pop_back();
        return true;
    }
    if (mutation == "singleton_region_node") {
        if (stage.singleton.regions.empty() ||
            stage.singleton.regions[0].nodes.empty()) {
            return false;
        }
        stage.singleton.regions[0].nodes[0] =
            static_cast<std::uint32_t>(stage.ops.size());
        return true;
    }
    if (mutation == "swap_singleton_region_nodes") {
        if (stage.singleton.regions.size() < 2 ||
            stage.singleton.regions[0].nodes.empty() ||
            stage.singleton.regions[1].nodes.empty()) {
            return false;
        }
        std::swap(
            stage.singleton.regions[0].nodes[0],
            stage.singleton.regions[1].nodes[0]);
        return true;
    }
    if (mutation == "unordered_region_nodes") {
        if (stage.fused.regions.empty() ||
            stage.fused.regions[0].nodes.size() < 2) {
            return false;
        }
        std::swap(
            stage.fused.regions[0].nodes[0], stage.fused.regions[0].nodes[1]);
        return true;
    }
    if (mutation == "region_input_out_of_range") {
        if (stage.fused.regions.empty()) return false;
        stage.fused.regions[0].inputs.push_back(
            static_cast<std::uint32_t>(stage.value_types.size()));
        return true;
    }
    if (mutation == "region_output_out_of_range") {
        if (stage.fused.regions.empty()) return false;
        stage.fused.regions[0].outputs.push_back(
            static_cast<std::uint32_t>(stage.value_types.size()));
        return true;
    }
    if (mutation == "region_sink_out_of_range") {
        if (stage.fused.regions.empty()) return false;
        stage.fused.regions[0].sinks.push_back(
            {static_cast<std::uint32_t>(stage.channel_bindings.size()), 0});
        return true;
    }
    if (mutation == "library_scan_claim") {
        if (stage.fused.regions.empty()) return false;
        stage.fused.regions[0].library = true;
        stage.fused.regions[0].library_op = PTIR_LIBRARY_SCAN;
        return true;
    }
    if (mutation == "extra_value_type") {
        plan::ValueType type;
        type.dtype = PTIR_DT_F32;
        type.dims.push_back({false, 1});
        stage.value_types.push_back(type);
        return true;
    }
    if (mutation == "drop_last_value_type") {
        if (stage.value_types.empty()) return false;
        stage.value_types.pop_back();
        return true;
    }
    if (mutation == "drop_last_value_type_and_refs") {
        if (stage.value_types.empty()) return false;
        stage.value_types.pop_back();
        const std::uint32_t dropped =
            static_cast<std::uint32_t>(stage.value_types.size());
        for (auto* partition : {&stage.singleton, &stage.fused}) {
            for (auto& region : partition->regions) {
                std::erase(region.inputs, dropped);
                std::erase(region.outputs, dropped);
                std::erase_if(region.sinks, [&](const plan::ChannelSink& sink) {
                    return sink.value == dropped;
                });
            }
        }
        return true;
    }
    if (mutation == "rank5_value_type") {
        if (stage.value_types.empty()) return false;
        stage.value_types[0].dims.assign(5, {false, 2});
        return true;
    }
    if (mutation == "zero_static_dim") {
        for (auto& type : stage.value_types) {
            for (auto& dimension : type.dims) {
                if (!dimension.symbolic) {
                    dimension.value = 0;
                    return true;
                }
            }
        }
        return false;
    }
    if (mutation == "overflow_static_dims") {
        if (stage.value_types.empty()) return false;
        stage.value_types[0].dims.assign(3, {false, 65536});
        return true;
    }
    if (mutation == "reverse_ops") {
        if (stage.ops.size() < 2) return false;
        std::reverse(stage.ops.begin(), stage.ops.end());
        return true;
    }
    if (mutation == "bad_channel_slot") {
        for (auto& planned : stage.ops) {
            if (planned.op.chan >= 0) {
                planned.op.chan =
                    static_cast<std::int64_t>(stage.channel_bindings.size());
                return true;
            }
        }
        return false;
    }
    if (mutation == "pivot_payload_out_of_range") {
        for (auto& planned : stage.ops) {
            if (planned.op.tag == PTIR_OP_PIVOT_THRESHOLD) {
                planned.op.pred_payload =
                    static_cast<std::uint32_t>(stage.value_types.size());
                return true;
            }
        }
        return false;
    }
    if (mutation == "rename_names") {
        if (stage.names.empty()) return false;
        for (auto& name : stage.names) name = "metal.bogus";
        return true;
    }
    if (mutation == "clear_names") {
        if (stage.names.empty()) return false;
        stage.names.clear();
        return true;
    }
    std::fprintf(stderr, "unknown mutation %s\n", mutation.c_str());
    std::exit(1);
}

std::string tag_hex(std::uint8_t tag) {
    std::ostringstream stream;
    stream << "0x" << std::hex << std::setfill('0') << std::setw(2)
           << unsigned(tag);
    return stream.str();
}

}  // namespace

int main(int argc, char** argv) {
    const std::string root = argc > 1 ? argv[1] : ".";
    const std::string out = argc > 2 ? argv[2] : "compiler/tests/golden-msl";

    const std::string runtime_template =
        read_file(root + "/driver/metal/src/kernels/ptir_m1_runtime.metal");
    g_runtime = runtime_template + "\n";
    // The grouped preamble is whatever `emit_grouped_fused_region_msl` puts
    // between the runtime template and the kernel; recover it from a probe so
    // the harness never restates the emitter's own text.
    {
        plan::StagePlan empty;
        plan::Region region;
        std::string probe;
        std::string ignored;
        emit_grouped_fused_region_msl(
            runtime_template, "probe", empty, region, probe, ignored);
        const std::string marker = "kernel void probe(";
        g_grouped_prefix = probe.substr(0, probe.find(marker));
    }

    const std::vector<CorpusStage> corpus =
        read_corpus(root + "/compiler/tests/oracle/corpus/stage_plans.txt");
    std::printf("corpus: %zu stage plans\n", corpus.size());

    // ── emit_singleton_region_msl: every tag byte (the function is total) ──
    {
        Dump dump;
        dump.emitter = "emit_singleton_region_msl";
        for (unsigned tag = 0; tag < 256u; ++tag) {
            dump.open_case("tag_" + tag_hex(static_cast<std::uint8_t>(tag)));
            const std::string name =
                "ptir_m1_" + hex64(0xfeedfacecafebeefull) + "_r" +
                std::to_string(tag);
            dump.field("function_name", name);
            dump.source(emit_singleton_region_msl(
                runtime_template, name, static_cast<std::uint8_t>(tag)));
        }
        finish(out, dump);
    }

    // ── emit_readiness_msl / emit_commit_msl ──────────────────────────────
    {
        Dump readiness;
        readiness.emitter = "emit_readiness_msl";
        Dump commit;
        commit.emitter = "emit_commit_msl";
        for (const EffectCase& item : effect_cases()) {
            const std::string ready_name = "ptir_m1_" + item.id + "_ready";
            const std::string commit_name = "ptir_m1_" + item.id + "_commit";
            readiness.open_case(item.id);
            readiness.field("function_name", ready_name);
            readiness.field("channels", std::to_string(item.channels.size()));
            for (std::size_t channel = 0; channel < item.channels.size();
                 ++channel) {
                const M1ChannelEffect& effect = item.channels[channel];
                std::ostringstream line;
                line << "requires_full=" << effect.requires_full
                     << " requires_empty=" << effect.requires_empty
                     << " take=" << effect.take << " put=" << effect.put
                     << " capacity=" << effect.capacity;
                readiness.field(
                    "channel[" + std::to_string(channel) + "]", line.str());
            }
            readiness.source(emit_readiness_msl(ready_name, item.channels));
            commit.open_case(item.id);
            commit.field("function_name", commit_name);
            commit.field("channels", std::to_string(item.channels.size()));
            commit.source(emit_commit_msl(commit_name, item.channels));
        }
        finish(out, readiness);
        finish(out, commit);
    }

    // ── emit_grouped_readiness_msl / emit_grouped_commit_msl ──────────────
    {
        Dump readiness;
        readiness.emitter = "emit_grouped_readiness_msl";
        Dump commit;
        commit.emitter = "emit_grouped_commit_msl";
        const std::string names[] = {
            "ptir_m3_generic_ready_v" + std::to_string(kMetalM1EmitterVersion),
            "ptir_m3_generic_commit_v" + std::to_string(kMetalM1EmitterVersion),
            "k",
            "ptir_m3_" + hex64(0x0123456789abcdefull) + "_effects",
        };
        for (const std::string& name : names) {
            readiness.open_case(name);
            readiness.field("function_name", name);
            readiness.source(emit_grouped_readiness_msl(name));
            commit.open_case(name);
            commit.field("function_name", name);
            commit.source(emit_grouped_commit_msl(name));
        }
        finish(out, readiness);
        finish(out, commit);
    }

    // ── validate_singleton_plan over the corpus × mutations ───────────────
    {
        Dump dump;
        dump.emitter = "validate_singleton_plan";
        for (const CorpusStage& stage : corpus) {
            for (const char* mutation : kMutations) {
                dump.open_case(stage.id + " " + mutation);
                plan::StagePlan damaged = stage.plan;
                if (!mutate(damaged, mutation)) {
                    dump.field("applicable", "false");
                    dump.body << "=== end\n";
                    continue;
                }
                dump.field("applicable", "true");
                std::vector<M1OpMeta> operations;
                std::string error;
                const bool ok =
                    validate_singleton_plan(damaged, operations, error);
                dump.field("ok", ok ? "true" : "false");
                dump.field("error", error);
                dump.field("operations", std::to_string(operations.size()));
                for (const M1OpMeta& meta : operations) {
                    std::ostringstream line;
                    line << "node=" << meta.node
                         << " result_base=" << meta.result_base
                         << " tag=" << tag_hex(meta.op.tag)
                         << " results=" << meta.op.results
                         << " chan=" << meta.op.chan
                         << " args=" << meta.op.args.size();
                    dump.field("op", line.str());
                }
                dump.body << "=== end\n";
            }
        }
        finish(out, dump);
    }

    // ── the four plan-taking emitters over every region of every stage ────
    {
        Dump fused;
        fused.emitter = "emit_fused_region_msl";
        Dump grouped;
        grouped.emitter = "emit_grouped_fused_region_msl";
        Dump nucleus;
        nucleus.emitter = "emit_grouped_nucleus_msl";
        Dump topk;
        topk.emitter = "emit_grouped_topk_msl";
        for (const CorpusStage& stage : corpus) {
            const std::string signature = hex64(stage.plan.signature_hash);
            struct PartitionRef {
                const char* name;
                const plan::Partition* partition;
            };
            const PartitionRef partitions[] = {
                {"singleton", &stage.plan.singleton},
                {"fused", &stage.plan.fused},
            };
            for (const PartitionRef& part : partitions) {
                // Singleton regions are 1:1 with ops, and their emitted text
                // is a function of the op's tag/arity, so one region per
                // distinct tag per stage covers the partition without dumping
                // dozens of near-identical 10 KB kernels. Fused regions are
                // all dumped.
                const bool singleton = std::strcmp(part.name, "singleton") == 0;
                bool seen[256] = {};
                for (std::size_t index = 0;
                     index < part.partition->regions.size();
                     ++index) {
                    const plan::Region& region = part.partition->regions[index];
                    if (singleton) {
                        if (region.nodes.size() != 1 ||
                            region.nodes[0] >= stage.plan.ops.size()) {
                            continue;
                        }
                        const std::uint8_t tag =
                            stage.plan.ops[region.nodes[0]].op.tag;
                        if (seen[tag]) continue;
                        seen[tag] = true;
                    }
                    const std::string id = stage.id + " " + part.name + "#" +
                        std::to_string(index);
                    std::ostringstream shape;
                    shape << "library=" << region.library
                          << " library_op=" << unsigned(region.library_op)
                          << " schedule=" << unsigned(region.schedule)
                          << " nodes=" << region.nodes.size()
                          << " inputs=" << region.inputs.size()
                          << " outputs=" << region.outputs.size()
                          << " sinks=" << region.sinks.size();

                    struct Target {
                        Dump* dump;
                        const char* prefix;
                        bool (*emit)(
                            const std::string&,
                            const std::string&,
                            const plan::StagePlan&,
                            const plan::Region&,
                            std::string&,
                            std::string&);
                    };
                    const Target targets[] = {
                        {&fused, "ptir_m2_", &emit_fused_region_msl},
                        {&grouped,
                         std::strcmp(part.name, "singleton") == 0
                             ? "ptir_m3s_"
                             : "ptir_m3_",
                         &emit_grouped_fused_region_msl},
                        {&nucleus, "ptir_m3_", &emit_grouped_nucleus_msl},
                        {&topk, "ptir_m3_", &emit_grouped_topk_msl},
                    };
                    for (const Target& target : targets) {
                        const std::string function = std::string(target.prefix) +
                            signature + "_r" + std::to_string(index);
                        target.dump->open_case(id);
                        target.dump->field("function_name", function);
                        target.dump->field("region", shape.str());
                        // `emit_grouped_nucleus_msl` reads `region.inputs[0]`
                        // and `region.outputs[0]` after a guard that forgot
                        // `!region.library` (its TopK sibling has it), so a
                        // Generated region — whose `library_op` byte is 0 ==
                        // PTIR_LIBRARY_NUCLEUS_SAMPLE — walks straight into an
                        // out-of-bounds read. Skip only where the C++ is
                        // undefined; everything else is dumped.
                        if (target.dump == &nucleus && !region.library &&
                            (region.inputs.size() < 3 ||
                             region.outputs.empty())) {
                            target.dump->field(
                                "skipped",
                                "cxx-oracle-undefined "
                                "(emit_grouped_nucleus_msl indexes "
                                "region.inputs[0]/outputs[0] without checking "
                                "region.library)");
                            target.dump->body << "=== end\n";
                            continue;
                        }
                        std::string source;
                        std::string error;
                        if (target.emit(
                                runtime_template,
                                function,
                                stage.plan,
                                region,
                                source,
                                error)) {
                            target.dump->source(source);
                        } else {
                            target.dump->error(error);
                        }
                    }
                }
            }
        }
        finish(out, fused);
        finish(out, grouped);
        finish(out, nucleus);
        finish(out, topk);
    }
    return 0;
}

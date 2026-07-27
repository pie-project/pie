// cuda_codegen_dump.cpp — TRANSITIONAL oracle for the CUDA kernel emitters.
//
// The sibling of `m1_codegen_dump.cpp`, for the CUDA backend. It links the
// header-only C++ emitters (`driver/cuda/src/pipeline/generated/`) and dumps
// their public entry points over the shared corpus in the same diffable format.
// The dump is checked in under `compiler/tests/golden-cuda/`; the Rust port in
// `compiler/codegen/src/cuda/` reproduces it byte for byte
// (`compiler/tests/tests/cuda_golden.rs`).
//
// DELETE THIS FILE when the C++ emitters are deleted. Nothing links it into a
// build; it exists so a human can re-derive the goldens from the C++ oracle and
// see an empty `git diff`. No CUDA toolkit is needed — the emitters are string
// builders, and the headers they include are CUDA-free.
//
// Build + run (from the repository root):
//
//   g++ -std=c++20 -O1 -o /path/to/scratch/cuda_codegen_dump \
//     compiler/tests/oracle/cuda_codegen_dump.cpp \
//     -I driver/cuda/src -I driver/common/include
//   /path/to/scratch/cuda_codegen_dump . compiler/tests/golden-cuda
//
// argv[1] is the repository root, argv[2] the output directory.

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "pie_native/ptir/plan.hpp"
#include "pipeline/generated/fused_codegen.hpp"

namespace generated = pie_cuda_driver::pipeline::generated;
namespace plan = pie_native::ptir::plan;

namespace {

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
    std::ostringstream out;
    out << std::hex << std::setw(16) << std::setfill('0') << value;
    return out.str();
}

std::string tag_hex(std::uint8_t tag) {
    std::ostringstream out;
    out << "0x" << std::hex << std::setw(2) << std::setfill('0')
        << unsigned(tag);
    return out.str();
}

std::vector<std::uint8_t> unhex(const std::string& text) {
    std::vector<std::uint8_t> bytes;
    bytes.reserve(text.size() / 2);
    for (std::size_t at = 0; at + 1 < text.size(); at += 2) {
        bytes.push_back(static_cast<std::uint8_t>(
            std::stoul(text.substr(at, 2), nullptr, 16)));
    }
    return bytes;
}

std::uint64_t fnv1a64(const std::string& text) {
    // 0xcbf29ce484222325 — the same basis as `pie_ir::program_hash`.
    std::uint64_t hash = 14695981039346656037ull;
    for (const char byte : text) {
        hash ^= static_cast<std::uint8_t>(byte);
        hash *= 1099511628211ull;
    }
    return hash;
}

// The runtime preamble every emitted kernel starts with, elided from the dump
// so a 30 KB constant is not repeated 200 times.
std::string g_runtime;

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

    void end() { body << "=== end\n"; }

    // A fused kernel is 30-60 KB and there are 320 of them, so the body is
    // pinned by hash rather than checked in whole; `verbatim` below keeps a
    // handful of representative ones readable.
    void kernel_digest(const generated::GeneratedKernelSource& result) {
        body << "ok: " << (result.ok ? "true" : "false") << "\n";
        body << "error: " << result.error << "\n";
        body << "entry_name: " << result.entry_name << "\n";
        if (result.ok) {
            std::string tail = result.source;
            const char* prefix = "none";
            if (!g_runtime.empty() && tail.rfind(g_runtime, 0) == 0) {
                prefix = "@runtime";
                tail = tail.substr(g_runtime.size());
            }
            body << "source: bytes=" << result.source.size()
                 << " prefix=" << prefix << " tail_bytes=" << tail.size()
                 << " fnv1a64=0x" << hex64(fnv1a64(tail)) << "\n";
        }
        end();
    }

    // `GeneratedKernelSource` carries its own ok/error, so unlike the Metal
    // dump there is no separate failure path.
    void kernel(const generated::GeneratedKernelSource& result) {
        body << "ok: " << (result.ok ? "true" : "false") << "\n";
        body << "error: " << result.error << "\n";
        body << "entry_name: " << result.entry_name << "\n";
        body << "op_tag: " << tag_hex(result.op_tag) << "\n";
        if (!result.ok) {
            end();
            return;
        }
        std::string tail = result.source;
        const char* prefix = "none";
        if (!g_runtime.empty() && tail.rfind(g_runtime, 0) == 0) {
            prefix = "@runtime";
            tail = tail.substr(g_runtime.size());
        }
        body << "source: bytes=" << result.source.size() << " prefix=" << prefix
             << " tail_bytes=" << tail.size() << "\n";
        body << tail;
        if (!tail.empty() && tail.back() != '\n') {
            body << "\n\\no-trailing-newline\n";
        }
        end();
    }
};

void finish(const std::string& directory, Dump& dump) {
    std::ostringstream head;
    head << "# oracle dump: pie_cuda_driver::pipeline::generated::"
         << dump.emitter << "\n";
    head << "# source of truth: driver/cuda/src/pipeline/generated/"
         << " (transitional; the Rust port is compiler/codegen/src/cuda/)\n";
    head << "# emitter_version: " << generated::kCudaGeneratedEmitterVersion
         << "\n";
    head << "# @runtime: bytes=" << g_runtime.size() << " fnv1a64=0x"
         << hex64(fnv1a64(g_runtime)) << "\n";
    head << "# cases: " << dump.cases << "\n";
    write_file(
        directory + "/" + dump.emitter + ".txt", head.str() + dump.body.str());
    std::printf("  %-32s %4zu cases\n", dump.emitter.c_str(), dump.cases);
}

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
                begin,
                end == std::string::npos ? std::string::npos : end - begin);
        };
        CorpusStage stage;
        stage.id = field("id");
        const std::vector<std::uint8_t> bytes = unhex(field("bytes"));
        std::string error;
        if (!plan::decode(bytes.data(), bytes.size(), stage.plan, &error)) {
            std::fprintf(
                stderr,
                "corpus plan %s failed to decode: %s\n",
                stage.id.c_str(),
                error.c_str());
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

// Entry names the emitters must accept or reject on their own terms; both
// check `valid_identifier` before doing any work.
const char* const kEntryNames[] = {
    "ptir_fused_0123456789abcdef_r0",
    "",
    "0starts_with_digit",
    "has-a-dash",
    "has space",
    "_underscore_start",
    "A",
};

}  // namespace

int main(int argc, char** argv) {
    const std::string root = argc > 1 ? argv[1] : ".";
    const std::string out = argc > 2 ? argv[2] : "compiler/tests/golden-cuda";

    g_runtime = generated::singleton_runtime_cuda_source();

    const std::vector<CorpusStage> corpus =
        read_corpus(root + "/compiler/tests/oracle/corpus/stage_plans.txt");
    std::printf("corpus: %zu stage plans\n", corpus.size());

    // ── emit_singleton_region_cuda: every tag byte × the entry-name cases ──
    {
        Dump dump;
        dump.emitter = "emit_singleton_region_cuda";
        for (unsigned tag = 0; tag < 256u; ++tag) {
            const std::string name =
                "ptir_singleton_" + tag_hex(static_cast<std::uint8_t>(tag));
            dump.open_case("tag_" + tag_hex(static_cast<std::uint8_t>(tag)));
            dump.field("entry_name", name);
            dump.kernel(generated::emit_singleton_region_cuda(
                name, static_cast<std::uint8_t>(tag)));
        }
        for (const char* name : kEntryNames) {
            dump.open_case(std::string("entry_name[") + name + "]");
            dump.field("entry_name", name);
            // 0x10 is a real op tag, so the only thing under test is the name.
            dump.kernel(generated::emit_singleton_region_cuda(name, 0x10));
        }
        finish(out, dump);
    }

    // ── the region-shaped entry points over every region of every stage ────
    {
        Dump fused;
        fused.emitter = "emit_fused_region_cuda";
        Dump validate;
        validate.emitter = "validate_generated_region";
        Dump second_party;
        second_party.emitter = "second_party_region_supported";
        Dump verbatim;
        verbatim.emitter = "emit_fused_region_cuda_verbatim";

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
                for (std::size_t index = 0;
                     index < part.partition->regions.size();
                     ++index) {
                    const plan::Region& region = part.partition->regions[index];
                    const std::string id =
                        stage.id + " " + part.name + "#" + std::to_string(index);
                    std::ostringstream shape;
                    shape << "library=" << region.library
                          << " library_op=" << unsigned(region.library_op)
                          << " schedule=" << unsigned(region.schedule)
                          << " nodes=" << region.nodes.size()
                          << " inputs=" << region.inputs.size()
                          << " outputs=" << region.outputs.size()
                          << " sinks=" << region.sinks.size();

                    const std::string entry =
                        "ptir_fused_" + signature + "_r" + std::to_string(index);
                    const generated::GeneratedKernelSource emitted =
                        generated::emit_fused_region_cuda(
                            entry, stage.plan, region);
                    fused.open_case(id);
                    fused.field("entry_name", entry);
                    fused.field("region", shape.str());
                    fused.kernel_digest(emitted);
                    // One region per stage is also kept verbatim, so a diff can
                    // be read rather than only detected.
                    if (index == 0) {
                        verbatim.open_case(id);
                        verbatim.field("entry_name", entry);
                        verbatim.field("region", shape.str());
                        verbatim.kernel(emitted);
                    }

                    validate.open_case(id);
                    validate.field("region", shape.str());
                    std::string error;
                    const bool ok = generated::validate_generated_region(
                        stage.plan, region, error);
                    validate.field("ok", ok ? "true" : "false");
                    validate.field("error", error);
                    validate.end();

                    second_party.open_case(id);
                    second_party.field("region", shape.str());
                    second_party.field(
                        "supported",
                        generated::second_party_region_supported(
                            stage.plan, region)
                            ? "true"
                            : "false");
                    second_party.end();
                }
            }
        }
        finish(out, fused);
        finish(out, verbatim);
        finish(out, validate);
        finish(out, second_party);
    }

    // ── the runtime preamble itself ───────────────────────────────────────
    {
        Dump dump;
        dump.emitter = "singleton_runtime_cuda_source";
        dump.open_case("runtime");
        dump.field("bytes", std::to_string(g_runtime.size()));
        dump.field("fnv1a64", "0x" + hex64(fnv1a64(g_runtime)));
        dump.end();
        finish(out, dump);
    }

    return 0;
}

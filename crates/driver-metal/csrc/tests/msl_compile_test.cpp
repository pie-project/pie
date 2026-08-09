// The host MSL emitter's output, through a real Metal compiler.
//
// Since the driver's own emitter was deleted, `register_program` fails outright
// when the host does not supply MSL for a fused region: `tensor-compiler's codegen/src/
// metal/` is now the only implementation, and every kernel this driver runs is
// text that crate produced. Nothing checked that text is *valid MSL*.
//
// `compiler/tests/tests/metal_msl_golden.rs` checks the emitter byte for byte
// against a frozen dump of the C++ oracle, which is a strong check on the port
// and no check at all on the language: the oracle was only ever dumped, never
// compiled, so a construct Apple's compiler rejects would sit in both sides of
// that comparison and pass. The whole corpus reached a Metal compiler for the
// first time here.
//
// This reads the same goldens the Rust test asserts against, so the sources are
// the emitter's own bytes: the Rust test pins Rust == golden, and this pins
// golden == compilable. The dump elides the two shared prefixes, which is why
// `grouped_preamble()` had to move out of a Rust string literal and into
// `crates/tensor-compiler/runtime/metal/ptir_m1_grouped.metal` — a literal in that
// crate is reachable from nothing but that crate, and the golden header's
// length + FNV field is what proves the move changed no byte.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "mtl4_context.hpp"

using pie::metal::RawMetalContext;

namespace {

int g_pass = 0, g_fail = 0;

bool expect(bool ok, const std::string& what) {
    if (ok) {
        ++g_pass;
        std::printf("  PASS  %s\n", what.c_str());
    } else {
        ++g_fail;
        std::printf("  FAIL  %s\n", what.c_str());
    }
    return ok;
}

bool read_file(const std::string& path, std::string& out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    std::ostringstream buffer;
    buffer << in.rdbuf();
    out = buffer.str();
    return true;
}

/// One emitted kernel recovered from the dump.
struct Emitted {
    std::string golden;      // which golden file it came from
    std::string entry;       // `function_name:` — the MSL entry point
    std::string source;      // prefix + tail, the bytes the emitter produced
    std::size_t declared;    // `bytes=` from the dump, for an integrity check
};

std::string_view trim_cr(std::string_view line) {
    if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
    return line;
}

/// Parse one golden dump.
///
/// Format (`metal_msl_golden.rs`'s `Dump`): `=== <id>` opens a case, `key:
/// value` lines carry fields, `ok: true` is followed by `source: bytes=N
/// prefix=P tail_bytes=T` and then the tail verbatim, and `=== end` closes.
/// A tail that does not end in a newline is followed by a `\no-trailing-newline`
/// marker line, which is not part of the source.
bool parse_golden(const std::string& name, const std::string& text, const std::string& runtime,
                  const std::string& grouped, std::vector<Emitted>& out, std::string& error) {
    std::istringstream in(text);
    std::string line;
    std::string entry;
    while (std::getline(in, line)) {
        const std::string_view view = trim_cr(line);
        if (view.rfind("function_name: ", 0) == 0) {
            entry = std::string(view.substr(std::string_view("function_name: ").size()));
            continue;
        }
        if (view != "ok: true") continue;

        if (!std::getline(in, line)) {
            error = name + ": truncated after `ok: true`";
            return false;
        }
        const std::string header(trim_cr(line));
        if (header.rfind("source: ", 0) != 0) {
            error = name + ": expected a `source:` line, got '" + header + "'";
            return false;
        }
        const auto field = [&header](std::string_view key) -> std::size_t {
            const std::size_t at = header.find(key);
            if (at == std::string::npos) return SIZE_MAX;
            return static_cast<std::size_t>(std::strtoull(header.data() + at + key.size(), nullptr, 10));
        };
        const std::size_t declared = field("bytes=");
        const std::size_t tail_bytes = field("tail_bytes=");
        const std::size_t prefix_at = header.find("prefix=");
        if (declared == SIZE_MAX || tail_bytes == SIZE_MAX || prefix_at == std::string::npos) {
            error = name + ": malformed `source:` line '" + header + "'";
            return false;
        }
        const std::string prefix_name = header.substr(prefix_at + 7, header.find(' ', prefix_at) - prefix_at - 7);

        std::string tail;
        bool no_trailing_newline = false;
        while (std::getline(in, line)) {
            const std::string_view body = trim_cr(line);
            if (body == "=== end") break;
            if (body == "\\no-trailing-newline") {
                no_trailing_newline = true;
                continue;
            }
            tail.append(body);
            tail.push_back('\n');
        }
        if (no_trailing_newline && !tail.empty()) tail.pop_back();

        std::string source;
        if (prefix_name == "@runtime") {
            source = runtime;
        } else if (prefix_name == "@runtime+@grouped") {
            source = runtime + grouped;
        } else if (prefix_name != "none") {
            error = name + ": unknown prefix '" + prefix_name + "'";
            return false;
        }
        source += tail;

        if (tail.size() != tail_bytes || source.size() != declared) {
            error = name + ": reconstruction of " + entry + " is " + std::to_string(source.size()) +
                    " bytes, the dump says " + std::to_string(declared) +
                    " (tail " + std::to_string(tail.size()) + " vs " + std::to_string(tail_bytes) +
                    ") — the prefixes have drifted from the dump";
            return false;
        }
        out.push_back({name, entry, std::move(source), declared});
    }
    return true;
}

/// Expand `#include "ptir/ptir_rng.generated.metal"`, exactly as the driver does at
/// library-build time (`m1_runtime.cpp`): Metal's runtime shader compiler does
/// no filesystem include lookup, so an unexpanded include is a compile error
/// rather than a missing symbol.
void splice_rng(std::string& source, const std::string& preamble) {
    static constexpr std::string_view kInclude = "#include \"ptir_rng.generated.metal\"";
    for (std::size_t at = source.find(kInclude); at != std::string::npos;
         at = source.find(kInclude, at + preamble.size())) {
        source.replace(at, kInclude.size(), preamble);
    }
}

}  // namespace

int main() {
    std::printf("[host-emitted MSL through the Metal compiler]\n");

    const std::string golden_dir = PIE_MSL_GOLDEN_DIR;
    const std::string runtime_path = PIE_MSL_RUNTIME_TEMPLATE;
    const std::string grouped_path = PIE_MSL_GROUPED_PREAMBLE;
    const std::string rng_path = PIE_MSL_RNG_PREAMBLE;

    std::string runtime, grouped, rng;
    if (!expect(read_file(runtime_path, runtime), "the embedded runtime template reads") ||
        !expect(read_file(grouped_path, grouped), "the grouped preamble reads") ||
        !expect(read_file(rng_path, rng), "the generated RNG preamble reads")) {
        std::printf("\n==== msl_compile_test: %d passed, %d failed ====\n", g_pass, g_fail);
        return 1;
    }
    // `runtime_prefix()` in the Rust dump appends one newline to the file.
    runtime.push_back('\n');

    // Every emitter's dump. Named rather than globbed so a golden that stops
    // being produced fails here instead of silently shrinking the corpus.
    const char* goldens[] = {
        "emit_commit_msl.txt",
        "emit_fused_region_msl.txt",
        "emit_grouped_commit_msl.txt",
        "emit_grouped_fused_region_msl.txt",
        "emit_grouped_nucleus_msl.txt",
        "emit_grouped_readiness_msl.txt",
        "emit_grouped_topk_msl.txt",
        "emit_readiness_msl.txt",
        "emit_singleton_region_msl.txt",
    };

    std::vector<Emitted> emitted;
    for (const char* name : goldens) {
        std::string text;
        if (!expect(read_file(golden_dir + "/" + name, text), std::string("golden ") + name + " reads")) {
            continue;
        }
        std::string error;
        if (!parse_golden(name, text, runtime, grouped, emitted, error)) {
            expect(false, error);
        }
    }
    if (!expect(!emitted.empty(), "the corpus is not empty")) {
        std::printf("\n==== msl_compile_test: %d passed, %d failed ====\n", g_pass, g_fail);
        return 1;
    }
    expect(emitted.size() >= 600,
           "the corpus is the whole dump (" + std::to_string(emitted.size()) + " kernels)");

    auto ctx = RawMetalContext::create(1u << 20);
    if (!expect(ctx != nullptr, "RawMetalContext::create succeeds")) {
        std::printf("\n==== msl_compile_test: %d passed, %d failed ====\n", g_pass, g_fail);
        return 1;
    }

    // Identical bytes compile identically, and there are a lot of them: the
    // corpus sweeps region shapes that often differ only above the tail.
    std::unordered_set<std::string> seen;
    std::size_t compiled = 0, skipped = 0;
    std::vector<std::string> failed;
    for (Emitted& kernel : emitted) {
        splice_rng(kernel.source, rng);
        if (!seen.insert(kernel.source + "\0" + kernel.entry).second) {
            ++skipped;
            continue;
        }
        std::string error;
        const auto pso = ctx->compile_ptir_pso(kernel.source, kernel.entry, &error);
        if (pso.valid()) {
            ++compiled;
            continue;
        }
        // Metal reports the whole translation unit's diagnostics; the first
        // error line is the one that names the construct it rejected.
        std::string first_line = error.substr(0, error.find('\n'));
        const std::size_t at = error.find("error: ");
        if (at != std::string::npos) {
            first_line = error.substr(at, error.find('\n', at) - at);
        }
        failed.push_back(kernel.golden + " :: " + kernel.entry + ": " + first_line);
    }

    for (const std::string& what : failed) {
        std::printf("  ---   %s\n", what.c_str());
    }
    expect(failed.empty(),
           "every emitted kernel compiles (" + std::to_string(compiled) + " distinct, " +
               std::to_string(skipped) + " duplicate, " + std::to_string(failed.size()) +
               " rejected)");

    std::printf("\n==== msl_compile_test: %d passed, %d failed ====\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

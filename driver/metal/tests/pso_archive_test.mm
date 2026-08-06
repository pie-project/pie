// Metal 4 pipeline archives for the statically-compiled kernel set.
//
// The driver used to rebuild every kernel pipeline from source on each start.
// Measured on an M1 Max, that was ~380 ms of pure pipeline creation before the
// first token, every single run, even with Metal's own shader cache fully warm:
// the cache saves the source parse, not the pipeline build. `MTL4Archive` was
// already wired up for the PTIR path but not for the ~40 kernel pipelines that
// dominate startup, which took the same batch down to ~9 ms.
//
// The risk an archive introduces is serving a stale binary, so the archive is
// keyed on the batch contents *and* on the RESOLVED source of every file in it
// -- resolved because a `.metal` file may `#include` another, and Metal's
// runtime compiler resolves none of those itself, so the driver splices them in
// before it compiles. A key built from the includer alone would keep its value
// when the included file changed and serve a pipeline compiled from the old
// definition, which is worse than a slow start because it looks like it worked.
//
// So the key is the bytes and not the clock. This test states that in the three
// places it can be observed: a rerun hits, a rewrite misses, and a rewrite of
// something the source merely INCLUDES misses too. It runs against sources it
// writes itself, because the alternative is editing a checked-in kernel and
// putting it back.

#import <Foundation/Foundation.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <utime.h>

#include "mtl4_context.hpp"

using pie::metal::Pso;
using pie::metal::RawMetalContext;

namespace {
int g_pass = 0, g_fail = 0;
bool expect(bool ok, const std::string& what) {
    if (ok) { ++g_pass; std::printf("  PASS  %s\n", what.c_str()); }
    else    { ++g_fail; std::printf("  FAIL  %s\n", what.c_str()); }
    return ok;
}

int count_archives(const std::string& dir) {
    NSArray<NSString*>* entries =
        [[NSFileManager defaultManager] contentsOfDirectoryAtPath:@(dir.c_str())
                                                            error:nil];
    int n = 0;
    for (NSString* e in entries) if ([e hasSuffix:@".mtl4archive"]) ++n;
    return n;
}

int finish() {
    std::printf("\n==== pso_archive_test: %d passed, %d failed ====\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

/// Write `text` to `path`, failing the test rather than the process.
bool put(const std::string& path, const std::string& text) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) return false;
    const bool ok = std::fwrite(text.data(), 1, text.size(), f) == text.size();
    std::fclose(f);
    return ok;
}

}  // namespace

int main() {
    std::printf("[Metal 4 pipeline archives: cache hit + source-change invalidation]\n");

    // A private cache directory so the test neither reads nor disturbs the
    // developer's real one.
    const std::string cache = "/tmp/pie_pso_archive_test";
    [[NSFileManager defaultManager] removeItemAtPath:@(cache.c_str()) error:nil];
    setenv("PIE_METAL_PSO_CACHE", cache.c_str(), 1);
    expect(RawMetalContext::pso_archive_dir() == cache,
           "PIE_METAL_PSO_CACHE selects the archive directory");

    const std::string dir = "/tmp/pie_pso_archive_test_src";
    [[NSFileManager defaultManager] removeItemAtPath:@(dir.c_str()) error:nil];
    [[NSFileManager defaultManager] createDirectoryAtPath:@(dir.c_str())
                              withIntermediateDirectories:YES
                                               attributes:nil
                                                    error:nil];
    const std::string header = dir + "/shared.h";
    const std::string src = dir + "/probe.metal";
    const std::string kernel =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "#include \"shared.h\"\n"
        "[[kernel]] void archive_probe(device float* out [[buffer(0)]],\n"
        "                              uint i [[thread_position_in_grid]]) {\n"
        "  out[i] = archive_probe_constant();\n"
        "}\n";
    if (!expect(put(header, "inline float archive_probe_constant() { return 1.0f; }\n") &&
                    put(src, kernel),
                "a kernel source and the header it includes are written")) {
        return finish();
    }

    std::vector<RawMetalContext::PsoFileRequest> requests = {{src, "archive_probe"}};

    auto build = [&](RawMetalContext& ctx) {
        std::vector<std::string> errors;
        auto psos = ctx.compile_psos_from_files(requests, &errors);
        bool ok = !psos.empty() && psos[0].valid();
        if (!ok && !errors.empty()) std::printf("        (%s)\n", errors[0].c_str());
        return ok;
    };

    {
        auto ctx = RawMetalContext::create(4u << 20);
        if (!expect(ctx != nullptr && build(*ctx),
                    "the source compiles, include and all")) {
            return finish();
        }
    }
    expect(count_archives(cache) == 1, "the first build wrote exactly one archive");

    {
        auto ctx = RawMetalContext::create(4u << 20);
        expect(ctx != nullptr && build(*ctx), "second build succeeds off the archive");
    }
    expect(count_archives(cache) == 1,
           "the second build reused the archive instead of writing another");

    // The clock alone is not a change. This is the direction the key USED to
    // go -- size and mtime -- and it cost a full rebuild on every `touch`, on
    // every checkout that rewrote a file to the same bytes, and on the `cp` out
    // of a backup that this repo's own workflow does.
    {
        struct stat st {};
        expect(stat(src.c_str(), &st) == 0, "the source can be stat'd");
        struct utimbuf times {};
        times.actime = st.st_atime;
        times.modtime = st.st_mtime + 120;
        expect(utime(src.c_str(), &times) == 0, "the source's mtime is moved forward");
        auto ctx = RawMetalContext::create(4u << 20);
        expect(ctx != nullptr && build(*ctx), "build after the touch succeeds");
    }
    expect(count_archives(cache) == 1, "an untouched-content source still hits the archive");

    // Changed bytes must miss.
    {
        expect(put(src, kernel + "// a byte that was not there before\n"),
               "the source's bytes change");
        auto ctx = RawMetalContext::create(4u << 20);
        expect(ctx != nullptr && build(*ctx), "build after the rewrite succeeds");
    }
    expect(count_archives(cache) == 2,
           "a changed source invalidates the archive and writes a new one");

    // And the whole reason the key is the RESOLVED source: the includer is
    // untouched here, and what it compiles to is different anyway.
    {
        expect(put(header, "inline float archive_probe_constant() { return 2.0f; }\n"),
               "the INCLUDED header changes and the source does not");
        auto ctx = RawMetalContext::create(4u << 20);
        expect(ctx != nullptr && build(*ctx), "build after the header changed succeeds");
    }
    expect(count_archives(cache) == 3,
           "a changed include invalidates the archive too");

    [[NSFileManager defaultManager] removeItemAtPath:@(cache.c_str()) error:nil];

    // Opting out has to actually skip the cache, since that is the escape hatch
    // when a pipeline is suspected of being served stale.
    {
        auto ctx = RawMetalContext::create(4u << 20);
        std::vector<std::string> errors;
        auto psos = ctx->compile_psos_from_files(requests, &errors,
                                                 /*use_archive_cache=*/false);
        expect(!psos.empty() && psos[0].valid(), "uncached build succeeds");
        expect(count_archives(cache) == 0, "uncached build writes no archive");
    }

    [[NSFileManager defaultManager] removeItemAtPath:@(cache.c_str()) error:nil];
    [[NSFileManager defaultManager] removeItemAtPath:@(dir.c_str()) error:nil];
    return finish();
}

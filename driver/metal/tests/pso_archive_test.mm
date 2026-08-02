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
// keyed on the batch contents *and* on the size and mtime of each source file.
// This test covers both halves: a second run must hit the archive, and touching
// a source must miss it.

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
}  // namespace

int main(int argc, char** argv) {
    std::printf("[Metal 4 pipeline archives: cache hit + source-change invalidation]\n");

    // A private cache directory so the test neither reads nor disturbs the
    // developer's real one.
    const std::string cache = "/tmp/pie_pso_archive_test";
    [[NSFileManager defaultManager] removeItemAtPath:@(cache.c_str()) error:nil];
    setenv("PIE_METAL_PSO_CACHE", cache.c_str(), 1);
    expect(RawMetalContext::pso_archive_dir() == cache,
           "PIE_METAL_PSO_CACHE selects the archive directory");

    const std::string dir = argc > 1 ? argv[1] : "src/kernels";
    // Two entrypoints out of one file is enough to exercise the whole path.
    const std::string src = dir + "/residual_add.metal";
    struct stat st {};
    if (!expect(stat(src.c_str(), &st) == 0, "kernel source found at " + src)) {
        return finish();
    }

    std::vector<RawMetalContext::PsoFileRequest> requests = {
        {src, "residual_add_bfloat16"},
    };

    auto build = [&](RawMetalContext& ctx) {
        std::vector<std::string> errors;
        auto psos = ctx.compile_psos_from_files(requests, &errors);
        bool ok = !psos.empty() && psos[0].valid();
        if (!ok && !errors.empty()) std::printf("        (%s)\n", errors[0].c_str());
        return ok;
    };

    {
        auto ctx = RawMetalContext::create(4u << 20);
        if (!expect(ctx != nullptr && build(*ctx), "first build compiles from source")) {
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

    // Roll the source's mtime forward. The bytes are untouched, but the driver
    // has no cheap way to know that, and treating any change as a miss is the
    // safe direction: the alternative is running a stale pipeline.
    struct utimbuf times {};
    times.actime = st.st_atime;
    times.modtime = st.st_mtime + 120;
    expect(utime(src.c_str(), &times) == 0, "kernel source mtime moved forward");

    {
        auto ctx = RawMetalContext::create(4u << 20);
        expect(ctx != nullptr && build(*ctx), "build after the source changed succeeds");
    }
    expect(count_archives(cache) == 2,
           "a changed source invalidates the archive and writes a new one");

    // Restore the mtime so a rerun starts from the same state.
    times.modtime = st.st_mtime;
    utime(src.c_str(), &times);
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
    return finish();
}

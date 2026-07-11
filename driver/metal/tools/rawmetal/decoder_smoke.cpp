// decoder_smoke — proves MetalExecutor drives the qwen3.6 native Metal path
// pipeline correctly: setup() once, step() the golden prompt token-by-token (mirroring the
// prefill→decode accumulation), and assert the final argmax == 264 (golden qwen36-pos7).
// This is the standalone correctness gate for the reusable host wrapper; the
// e2e number itself comes from pie + benches/pie_bench.py.
//
// Usage: decoder_smoke <checkpoint_dir> <kernels_dir> [comma_prompt_ids] [expect_argmax]

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "native_access.hpp"

using namespace pie::metal;
using pie::metal::batch::MetalExecutor;
using pie::metal::batch::NativeAccess;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: %s <checkpoint_dir> <kernels_dir> [comma_prompt_ids] [expect_argmax]\n",
            argv[0]);
        return 2;
    }
    const std::string ckpt_dir    = argv[1];
    const std::string kernels_dir = argv[2];
    const std::string ids_csv = (argc > 3) ? argv[3] : "785,6722,315,9625,374,264,3460,304";
    const uint32_t expect = (argc > 4) ? uint32_t(std::stoul(argv[4])) : 264u;

    std::vector<uint32_t> ids;
    for (size_t p = 0; p < ids_csv.size();) {
        size_t c = ids_csv.find(',', p);
        if (c == std::string::npos) c = ids_csv.size();
        ids.push_back(uint32_t(std::stoul(ids_csv.substr(p, c - p))));
        p = c + 1;
    }

    MetalExecutor dec;
    std::string err;
    if (!NativeAccess::setup(
            dec, ckpt_dir, kernels_dir, DecodeGeometry{}, &err)) {
        std::fprintf(stderr, "[decoder_smoke] setup failed: %s\n", err.c_str());
        return 1;
    }
    std::printf(
        "[decoder_smoke] setup OK (vocab=%d). prompt ids (%zu):",
        NativeAccess::vocab(dec),
        ids.size());
    for (uint32_t id : ids) std::printf(" %u", id);
    std::printf("\n");

    // Fresh sequence: zero GDN + KV state, then feed each prompt id as an M=1 step at its
    // absolute position (prefill semantics; argmax of the LAST step is the next-token).
    NativeAccess::reset_state(dec);
    uint32_t last_argmax = 0;
    for (size_t i = 0; i < ids.size(); ++i) {
        StepTiming t = NativeAccess::step(dec, ids[i], uint32_t(i));
        last_argmax = NativeAccess::argmax(dec);
        std::printf("[decoder_smoke] step %zu (id=%u pos=%zu): gpu_exec_ms=%.4f argmax=%u\n",
                    i, ids[i], i, t.gpu_exec_ms, last_argmax);
    }

    const bool ok = (last_argmax == expect);
    std::printf("[decoder_smoke] final argmax=%u expect=%u -> %s\n",
                last_argmax, expect, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

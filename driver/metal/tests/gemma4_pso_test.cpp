// Gemma 4 PSO compile gate — REAL Metal shader compilation on-device.
//
// Six `.metal` files shipped with this family's bring-up and had no PSO entry,
// no `bind::` enum and no host-side params mirror: they compiled in the abstract
// and nothing could reach them. `build_gemma4_psos` is the other half, and this
// proves every entry point it names actually exists and builds against the real
// sources — including both head widths of the sliding-window attention, which
// this family needs because its head_dim is per attention type (256 sliding,
// 512 full).
//
// No checkpoint: just `kernels/*.metal` on disk and a Metal device.

#include <cstdio>
#include <cstdlib>
#include <string>

#include "model/gemma4/kernels.hpp"
#include "mtl4_context.hpp"

using pie::metal::RawMetalContext;
using pie::metal::gemma4::build_gemma4_psos;
using pie::metal::gemma4::Gemma4Psos;

namespace {

int g_failures = 0;

void expect(bool ok, const std::string& what) {
    std::printf("  %s  %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++g_failures;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::string kernels_dir = PIE_METAL_KERNELS_DIR_FOR_TEST;
    if (argc > 1) kernels_dir = argv[1];
    std::printf("gemma4 kernel PSOs (%s)\n", kernels_dir.c_str());

    std::string error;
    auto ctx = RawMetalContext::create(16u << 20);
    if (!ctx) {
        std::printf("  FAIL  RawMetalContext::create\n");
        return 1;
    }
    expect(true, "RawMetalContext::create succeeds");

    Gemma4Psos psos;
    const bool built = build_gemma4_psos(*ctx, kernels_dir, psos, &error);
    expect(built, "every gemma4 PSO compiles: " + (built ? std::string("ok") : error));
    if (built) {
        // Named individually so a regression says which kernel broke.
        expect(psos.sdpa_swa_d256.valid(), "sdpa_vector_decode_swa d=256 (sliding layers)");
        expect(psos.sdpa_swa_d512.valid(), "sdpa_vector_decode_swa d=512 (full layers)");
        expect(psos.geglu_tanh.valid(), "geglu_tanh (gemma's FFN nonlinearity)");
        expect(psos.logit_softcap.valid(), "logit_softcap (cap * tanh(x / cap))");
        expect(psos.layer_scalar.valid(), "layer_scalar_mul (learned per-layer gain)");
        expect(psos.ple_combine.valid(), "ple_combine (per-layer embeddings)");
        expect(psos.vnorm.valid(), "vnorm_single_row (weightless V-norm)");
        expect(psos.valid(), "the table reports itself complete");
    }

    std::printf("\n==== gemma4_pso_test: %s ====\n", g_failures == 0 ? "all passed" : "FAILURES");
    return g_failures == 0 ? 0 : 1;
}

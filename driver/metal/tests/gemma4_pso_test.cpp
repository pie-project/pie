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

#include "model/gemma4/decode_step.hpp"
#include "model/gemma4/encode.hpp"
#include "model/gemma4/decode_consts.hpp"
#include "model/gemma4/geometry.hpp"
#include "model/gemma4/kernels.hpp"
#include "decode_psos.hpp"
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

    // Every dispatch in the step has to resolve to a pipeline that exists and a
    // grid the hardware will accept. `pso_max_threads` is 832 on an M1 Max, and
    // exceeding it is NOT an error — it silently computes something else.
    if (built) {
        std::printf("[whole step resolves]\n");
        pie::metal::DecodeStepPsos base;
        std::string base_err;
        const bool base_ok = pie::metal::load_decode_psos(*ctx, kernels_dir, base,
                                                          /*with_argmax=*/true, &base_err);
        expect(base_ok, "the shared decode PSOs compile: " + (base_ok ? std::string("ok")
                                                                     : base_err));
        if (base_ok) {
            const pie::metal::gemma4::Gemma4Geometry g;
            const auto dag = pie::metal::gemma4::build_gemma4_dag(g);
            int missing = 0, oversized = 0, empty_grid = 0;
            std::uint32_t worst_cap = 0;
            for (const auto& d : dag) {
                const auto pso = pie::metal::gemma4::pso_for(d, base, psos);
                if (!pso.valid()) {
                    ++missing;
                    continue;
                }
                pie::metal::Grid grid;
                pie::metal::Threadgroup tg;
                pie::metal::gemma4::launch_shape(d, g, grid, tg);
                const std::uint32_t threads = tg.x * tg.y * tg.z;
                // Per PIPELINE, not per device: the cap depends on the kernel's
                // register use, and exceeding it is not an error — it silently
                // computes something else.
                const std::uint32_t cap = ctx->pso_max_threads(pso);
                if (cap > worst_cap) worst_cap = cap;
                if (threads == 0 || threads > cap) ++oversized;
                if (grid.x == 0 || grid.y == 0 || grid.z == 0) ++empty_grid;
            }
            std::printf("  (%zu dispatches, widest pipeline allows %u threads)\n", dag.size(),
                        worst_cap);
            expect(missing == 0, "every dispatch resolves to a compiled pipeline");
            expect(oversized == 0, "no threadgroup exceeds what this device allows");
            expect(empty_grid == 0, "no dispatch has an empty grid");

            // The same dispatches at M>1. At rows==1 EVERY one of them must
            // reduce to the decode shape -- including the projections, whose
            // `qmv_mb_dispatch(N, 1)` is `qmv_dispatch(N)` by construction. This
            // used to assert the opposite for projections, which passed only
            // because decode's matvec shape was itself wrong: `Grid{1, N/8, 1}`
            // against tg `{64,1,1}` gave every threadgroup one thread, so half
            // the output rows were never written and the rest summed 1/32 of K.
            // A rows==1 divergence between the two paths is the symptom.
            std::printf("[M>1 launch shapes]\n");
            int mb_empty = 0, mb_oversized = 0, unexpected_diff = 0;
            for (int rows : {1, 2, 17, 128}) {
                for (const auto& d : dag) {
                    pie::metal::Grid mg{}; pie::metal::Threadgroup mt{};
                    pie::metal::gemma4::launch_shape_mb(d, g, rows, mg, mt);
                    if (mg.x == 0 || mg.y == 0 || mg.z == 0) ++mb_empty;
                    const pie::metal::Pso p = pie::metal::gemma4::pso_for(d, base, psos);
                    const std::uint32_t threads = mt.x * mt.y * mt.z;
                    if (p.valid() && threads > ctx->pso_max_threads(p)) ++mb_oversized;
                    if (rows != 1) continue;
                    pie::metal::Grid dg{}; pie::metal::Threadgroup dt{};
                    pie::metal::gemma4::launch_shape(d, g, dg, dt);
                    const bool same = mg.x == dg.x && mg.y == dg.y && mg.z == dg.z &&
                                      mt.x == dt.x && mt.y == dt.y && mt.z == dt.z;
                    if (!same) ++unexpected_diff;
                }
            }
            expect(mb_empty == 0, "no M>1 dispatch has an empty grid");
            expect(mb_oversized == 0, "no M>1 threadgroup exceeds what this device allows");
            expect(unexpected_diff == 0,
                   "at rows==1 every M>1 shape reduces to decode's (" +
                       std::to_string(unexpected_diff) + " do not)");

            // And that matmul PSO must exist. A dispatch whose grid was computed
            // for one tiling and whose pipeline was compiled for another is not
            // a crash -- it is wrong numbers -- so every M>1 dispatch has to
            // resolve, and resolve to something other than the M=1 pipeline
            // wherever the kernel genuinely changes.
            pie::metal::MultiBatchPsos mb;
            std::string mb_err;
            const bool mb_ok = pie::metal::load_multibatch_psos(
                *ctx, kernels_dir, mb, /*with_d512=*/true, &mb_err);
            expect(mb_ok, "the multi-batch PSOs compile: " + (mb_ok ? std::string("ok")
                                                                    : mb_err));
            if (mb_ok) {
                int unresolved = 0, still_m1 = 0;
                for (const auto& d : dag) {
                    const pie::metal::Pso p =
                        pie::metal::gemma4::pso_for_mb(d, g, 128, base, mb, psos);
                    if (!p.valid()) ++unresolved;
                    const bool changes =
                        pie::metal::gemma4::qmv_kn(d.kind, g, d.layer).N != 0 ||
                        d.kind == pie::metal::gemma4::Kind::EmbedGather ||
                        d.kind == pie::metal::gemma4::Kind::PleTokenGather ||
                        d.kind == pie::metal::gemma4::Kind::RopeQ ||
                        d.kind == pie::metal::gemma4::Kind::RopeK ||
                        d.kind == pie::metal::gemma4::Kind::KvAppend ||
                        d.kind == pie::metal::gemma4::Kind::Sdpa;
                    if (changes) {
                        const pie::metal::Pso m1 =
                            pie::metal::gemma4::pso_for(d, base, psos);
                        if (p.obj == m1.obj) ++still_m1;
                    }
                }
                expect(unresolved == 0, "every M>1 dispatch resolves to a pipeline");
                expect(still_m1 == 0,
                       "and every kind whose kernel changes at M>1 got a different one (" +
                           std::to_string(still_m1) + " did not)");
            }
        }
    }

    std::printf("\n==== gemma4_pso_test: %s ====\n", g_failures == 0 ? "all passed" : "FAILURES");
    return g_failures == 0 ? 0 : 1;
}

// Phase 1b/3 paged-KV bridge multi-batch PSO compile gate — REAL Metal
// shader compilation (runs on-device via `[MTLDevice newLibraryWithSource:]`,
// The driver compiles shaders at runtime, but this test needs
// NO checkpoint at all — just the `kernels/*.metal` sources on disk and a
// Metal device, both available on this Mac. Gated on Apple (this file only
// builds/links where RawMetalContext is real; a non-Apple CI box has nothing
// to compile shaders against, so this test is Apple-only, unlike the pure
// heap_layout/executor_geometry gates).
//
// Proves `load_multibatch_psos` — including the kv_append_paged PSO this
// session ADDED (metal_ptir_plan.md Phase 1b/3 review: "kv_append_paged has
// no PSO entry" was cited as a concrete missing piece) — compiles
// successfully against the real kernel sources: embed_gather_mb, rope_mb,
// gdn_core_slotted, sdpa_paged (d256 required, d512/gemma4 optional), and
// kv_append_paged. A real, decisive, on-hardware check that these shaders
// are syntactically valid Metal and bind-index-consistent with the
// `MultiBatchPsos`/`bind::` contracts in decode_abi.hpp — the actual
// encoder/DAG wiring to DISPATCH them in a live forward is the genuinely
// remaining gap (see decode_step_mb — not yet implemented this session);
// this test proves the compile-time half of that gap is now closed.

#include <cstdio>
#include <cstdlib>
#include <string>

#include "decode_psos.hpp"
#include "mtl4_context.hpp"

using pie::metal::load_multibatch_psos;
using pie::metal::load_decode_psos;
using pie::metal::DecodeStepPsos;
using pie::metal::MultiBatchPsos;
using pie::metal::RawMetalContext;

namespace {
int g_pass = 0, g_fail = 0;
bool expect(bool ok, const std::string& what) {
    if (ok) { ++g_pass; std::printf("  PASS  %s\n", what.c_str()); }
    else    { ++g_fail; std::printf("  FAIL  %s\n", what.c_str()); }
    return ok;
}
}  // namespace

int main() {
    std::printf("[multi-batch PSO compile gate: real Metal shader compilation]\n");

    std::string kernels_dir;
    if (const char* kd = std::getenv("PIE_METAL_KERNELS_DIR")) kernels_dir = kd;
#ifdef PIE_METAL_KERNELS_DIR_DEFAULT
    if (kernels_dir.empty()) kernels_dir = PIE_METAL_KERNELS_DIR_DEFAULT;
#endif
    if (!expect(!kernels_dir.empty(), "kernels_dir resolved (env or compiled-in default)")) {
        std::printf("\n==== kv_append_paged_pso_test: %d passed, %d failed ====\n", g_pass, g_fail);
        return g_fail == 0 ? 0 : 1;
    }

    // A small heap is enough — this test only compiles PSOs, it never
    // allocates/binds a real decode heap.
    auto ctx = RawMetalContext::create(16u << 20);
    if (!expect(ctx != nullptr, "RawMetalContext::create succeeds")) {
        std::printf("\n==== kv_append_paged_pso_test: %d passed, %d failed ====\n", g_pass, g_fail);
        return g_fail == 0 ? 0 : 1;
    }

    MultiBatchPsos psos;
    std::string err;
    DecodeStepPsos base;
    expect(load_decode_psos(*ctx, kernels_dir, base, /*with_argmax=*/false, &err,
                            /*fuse_residual=*/false, /*gdn_prep=*/true),
           "load_decode_psos compiles base kernels after MB row ABI additions (" + err + ")");
    const bool ok = load_multibatch_psos(*ctx, kernels_dir, psos, /*with_d512=*/true, &err);
    expect(ok, "load_multibatch_psos compiles successfully (" + err + ")");
    expect(psos.embed_mb.valid(), "embed_gather_mb_4bit_bfloat16_gs_64_b_4 compiled");
    expect(psos.rope_mb.valid(), "rope_neox_mb_bfloat16 compiled");
    expect(psos.gdn_slotted.valid(), "gdn_core_slotted_bfloat16 compiled");
    expect(psos.sdpa_paged.valid(), "sdpa_paged_decode_bfloat16_d_256 compiled");
    expect(psos.sdpa_paged_d512.valid(), "sdpa_paged_decode_bfloat16_d_512 (gemma4) compiled");
    expect(psos.kv_append_paged.valid(),
          "kv_append_paged_bfloat16 compiled — Phase 1b/3 review's cited gap "
          "(\"kv_append_paged has no PSO\") is closed at the compile level");
    expect(psos.valid(), "MultiBatchPsos::valid() (all required paged/slotted PSOs) is true");

    std::printf("\n==== kv_append_paged_pso_test: %d passed, %d failed ====\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

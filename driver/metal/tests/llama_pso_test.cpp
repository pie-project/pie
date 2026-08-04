// `build_llama_psos` against the real Metal compiler.
//
// Every PSO this family needs beyond the shared decode set, compiled on device.
// That set is small on purpose -- the shared table already carried the dense
// llama decoder -- so what is actually being proved here is the three claims
// that made it small:
//
//   1. A 128-wide attention head is an INSTANTIATION of `sdpa_vector` and
//      `sdpa_paged`, not a new kernel. If the templates did not really
//      generalise over D, they fail to compile at 128 and this catches it.
//   2. `router_topk` and `expert_combine` are generic, despite living in
//      `gptoss.metal`. If either had a gpt-oss assumption baked in, reusing
//      them from a Qwen geometry is where that shows.
//   3. The unbiased routed matvec is `qmv_gptoss_impl` with BIASED off.
//
// And one claim about what is NOT compiled: a dense checkpoint must not be held
// to the MoE PSOs. Compiling shaders a model will never dispatch means an
// unrelated error in them fails a load that would otherwise have worked.

#include <cstdio>
#include <cstdlib>
#include <string>

#include "model/llama/kernels.hpp"
#include "mtl4_context.hpp"

using pie::metal::Grid;
using pie::metal::RawMetalContext;
using pie::metal::Threadgroup;
using pie::metal::llama::LlamaGeometry;
using pie::metal::llama::LlamaPsos;
using pie::metal::llama::build_llama_psos;
using pie::metal::llama::expert_combine_dispatch;
using pie::metal::llama::expert_silu_dispatch;
using pie::metal::llama::routed_qmv_dispatch;
using pie::metal::llama::router_topk_dispatch;

namespace {
int g_pass = 0, g_fail = 0;
void expect(bool ok, const std::string& what) {
    if (ok) { ++g_pass; std::printf("  PASS  %s\n", what.c_str()); }
    else    { ++g_fail; std::printf("  FAIL  %s\n", what.c_str()); }
}
void expect_eq(long long got, long long want, const std::string& what) {
    if (got == want) { ++g_pass; std::printf("  PASS  %s (%lld)\n", what.c_str(), got); }
    else { ++g_fail; std::printf("  FAIL  %s: got %lld, want %lld\n", what.c_str(), got, want); }
}
}  // namespace

int main() {
    std::printf("llama_pso_test — the family's kernels, compiled\n");

    std::string kernels_dir;
    if (const char* kd = std::getenv("PIE_METAL_KERNELS_DIR")) kernels_dir = kd;
#ifdef PIE_METAL_KERNELS_DIR_FOR_TEST
    if (kernels_dir.empty()) kernels_dir = PIE_METAL_KERNELS_DIR_FOR_TEST;
#endif
    if (!(g_fail == 0 && !kernels_dir.empty())) {
        std::printf("  FAIL  kernels_dir resolved\n");
        return 1;
    }

    auto ctx = RawMetalContext::create(8u << 20);
    if (ctx == nullptr) { std::printf("  FAIL  RawMetalContext::create\n"); return 1; }

    LlamaGeometry dense;
    dense.head_dim = 128;
    LlamaPsos dense_psos;
    std::string err;
    expect(build_llama_psos(*ctx, kernels_dir, dense, dense_psos, &err),
           "dense llama PSOs compile: " + err);
    expect(dense_psos.dense_valid(), "sdpa d128, sdpa_paged d128 and row_gather are valid");
    expect(dense_psos.valid_for(dense), "a dense geometry is satisfied by the dense set");
    expect(!dense_psos.moe_valid(),
           "a dense load does not compile the routed set it will never dispatch");

    LlamaGeometry moe = dense;
    moe.n_experts = 128;
    moe.experts_per_token = 8;
    moe.moe_intermediate = 768;
    expect(!dense_psos.valid_for(moe),
           "the dense set does NOT satisfy a routed geometry");

    // Llama-3.2-1B is 32 query heads of 64, not 8 of 128, and a set compiled
    // for 128 handed 64-wide heads does not fail -- it strides past the end of
    // every head and writes zeros, which is a model that runs and says nothing.
    // So the width has to be an instantiation that is actually ASKED for.
    LlamaGeometry narrow = dense;
    narrow.head_dim = 64;
    narrow.n_q_heads = 32;
    narrow.n_kv_heads = 8;
    LlamaPsos narrow_psos;
    err.clear();
    expect(build_llama_psos(*ctx, kernels_dir, narrow, narrow_psos, &err),
           "a 64-wide head compiles its own attention: " + err);
    expect(narrow_psos.valid_for(narrow), "and satisfies the geometry that asked for it");
    expect(narrow_psos.sdpa.obj != dense_psos.sdpa.obj,
           "which is a DIFFERENT pipeline from the 128-wide one");
    expect(narrow_psos.sdpa_paged.obj != dense_psos.sdpa_paged.obj,
           "paged too");
    expect(narrow_psos.sdpa_paged_tiled.obj != dense_psos.sdpa_paged_tiled.obj,
           "and tiled");

    // A width nothing instantiates is a named compile failure at load, not a
    // pipeline built for some other width.
    LlamaGeometry unknown = dense;
    unknown.head_dim = 96;
    LlamaPsos unknown_psos;
    err.clear();
    expect(!build_llama_psos(*ctx, kernels_dir, unknown, unknown_psos, &err),
           "an uninstantiated head width is refused, not approximated");
    expect(err.find("_d_96") != std::string::npos,
           "and the refusal names the width it could not find [" + err + "]");

    // Llama 3.1's rope is a frequency TABLE, and the two kernels that read one
    // are compiled only for a geometry that says so.
    LlamaGeometry tabled = dense;
    tabled.rope_freq_table = true;
    expect(!dense_psos.valid_for(tabled),
           "the base set does NOT satisfy a table-driven rope");
    LlamaPsos tabled_psos;
    err.clear();
    expect(build_llama_psos(*ctx, kernels_dir, tabled, tabled_psos, &err),
           "the table-driven rope compiles: " + err);
    expect(tabled_psos.rope_table_valid() && tabled_psos.valid_for(tabled),
           "and satisfies the geometry that asked for it");
    expect(!dense_psos.rope_table_valid(),
           "while a geometric-series checkpoint compiles neither");

    LlamaPsos moe_psos;
    err.clear();
    expect(build_llama_psos(*ctx, kernels_dir, moe, moe_psos, &err),
           "routed llama PSOs compile: " + err);
    expect(moe_psos.moe_valid(), "router_topk, expert_combine and the unbiased routed qmv");
    expect(moe_psos.valid_for(moe), "a routed geometry is satisfied by the full set");

    // ── Launch shapes ──
    std::printf("\n-- launch geometry --\n");
    Grid g{};
    Threadgroup tg{};

    router_topk_dispatch(128, g, tg, 4);
    expect_eq(tg.x, 128, "router: 128 experts is four whole simdgroups");
    expect_eq(g.y, 4, "router: one threadgroup per row");
    router_topk_dispatch(48, g, tg);
    expect_eq(tg.x, 64, "router: 48 experts rounds up to two simdgroups");
    router_topk_dispatch(4096, g, tg);
    expect_eq(tg.x, 1024, "router: capped at the threadgroup limit");

    // The slot axis is the whole difference from a dense matvec, and getting it
    // wrong means every row computes one expert instead of k.
    //
    // These numbers were WRONG here before `llama_numerics_test` ran the
    // kernel: this asked for `g.y == N` and one simdgroup, and the kernel
    // derives `out_row` from `tid.y * 8` across TWO simdgroups. That shape
    // leaves four output rows in every eight unwritten and indexes up to 8N.
    // Pinning a launch shape only pins it to whatever was believed, which is
    // why the arithmetic below is stated against the kernel's own constants.
    routed_qmv_dispatch(768, 8, g, tg, 3);
    expect_eq(g.y, 768 / 4, "routed qmv: 8 output rows per threadgroup, 4 per simdgroup");
    expect_eq(g.z, 8, "routed qmv: the slot axis is experts_per_token");
    expect_eq(g.x, 32 * 3, "routed qmv: the token row rides tid.x, not the slot axis");
    expect_eq(tg.x, 32, "routed qmv: a simdgroup is 32 lanes");
    expect_eq(tg.y, 2, "routed qmv: two simdgroups, which is what out_row assumes");

    expert_combine_dispatch(2048, g, tg, 5);
    expect_eq(g.x, 2048, "combine: one thread per hidden element");
    expect_eq(g.y, 5, "combine: one row per gid.y");
    expect(g.z == 1, "combine: the k slots are summed inside, not in the grid");

    // The expert SwiGLU is flat over the stack precisely because gate, up and
    // out share a layout.
    expert_silu_dispatch(768, 8, g, tg, 2);
    expect_eq(static_cast<long long>(g.x) * g.y * g.z >= 768LL * 8 * 2, 1,
              "expert silu covers rows x k x moe_intermediate");

    std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

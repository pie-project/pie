// `entrypoint.h`, through a real compiler and against the real shader set.
//
// This is the C++ half of `.wiki/kernel-metal-refactor.md` §6 invariant (2);
// the Rust half is `crates/kernels-metal/tests/entrypoints.rs`. Neither needs
// an Apple toolchain: a `.metal` is data to the host, so the name grammar and
// the set it checks against are ordinary C++.
//
// Run by `scripts/metal-kernel-audit.py --cpp`, which compiles it with the one
// include root it needs. Paths below are relative to the repo root, which is
// where that script runs it from.

#include "pie/kernels/entrypoint.h"
#include "pie/kernels/moe.h"
#include <cstdio>
#include <fstream>
#include <vector>
using namespace pie::kernels;
int fail = 0;
void ok(bool c, const char* what) { if (!c) { std::printf("FAIL %s\n", what); ++fail; } }
int main() {
    // 1. every generated name is accepted by is_entrypoint
    std::ifstream in("crates/kernels-metal/entrypoints.generated.txt");
    std::string line; int n = 0;
    while (std::getline(in, line)) { if (!line.empty()) { ok(is_entrypoint(line), line.c_str()); ++n; } }
    std::printf("checked %d generated names\n", n);
    ok(n > 400, "the generated set is not empty or truncated");
    // 2. the builders reproduce real names
    pie::metal::AffineFormat g64b4{4, 64};
    ok(entrypoint("affine_qmv_fast", {affine(g64b4)}) == "affine_qmv_fast_bfloat16_gs_64_b_4", "qmv_fast");
    ok(entrypoint("affine_qmm_t", {affine(g64b4), tile(16, 32)}) == "affine_qmm_t_bfloat16_gs_64_b_4_bm_16_bn_32", "qmm_t");
    ok(entrypoint("sdpa_vector_decode", {bf16(), head_dim(128)}) == "sdpa_vector_decode_bfloat16_d_128", "sdpa d128");
    ok(entrypoint("rms_single_row", {bf16()}) == "rms_single_row_bfloat16", "rms");
    ok(entrypoint("route_sort") == "route_sort", "axisless");
    ok(entrypoint("affine_qmv_wide_strided", {affine(g64b4), rows(4), k_unroll(8)})
           == "affine_qmv_wide_strided_bfloat16_gs_64_b_4_v_4_kl_8", "wide strided");
    ok(entrypoint("gdn_core_recurrent_prefill", {bf16(), chunk(32), rows(4)})
           == "gdn_core_recurrent_prefill_bfloat16_l_32_v_4", "gdn prefill");
    // 3. The refusal, on the kernel this exists for. `affine_qmv_routed` is
    // compiled for ONE affine format BY DESIGN -- `AffineQ::group_size` is a
    // constant, so a second point would name an instantiation that dequantises
    // at 64 whatever it claims. A routed checkpoint at another group is meant
    // to fail by name, and this is that failure, at the call rather than in
    // the Metal compiler.
    //
    // An earlier version of this test asserted the opposite, after the gap was
    // "fixed" by adding five instantiations. They were five DUPLICATE explicit
    // instantiations of two specializations -- the macro puts `gs` in the
    // `host_name` string only -- so the file would not have compiled, and had
    // it compiled the numbers would have been wrong under a name that promised
    // otherwise. Reverted; this asserts the design.
    ok(is_entrypoint("affine_qmv_routed_bfloat16_gs_64_b_4"), "routed qmv at g64/b4");
    ok(!is_entrypoint("affine_qmv_routed_bfloat16_gs_32_b_4"), "and at no other group");
    bool threw = false;
    try { entrypoint("affine_qmv_routed", {affine(pie::metal::AffineFormat{4, 32})}); }
    catch (const std::exception& e) { threw = true; std::printf("  refusal: %s\n", e.what()); }
    ok(threw, "a group with no instantiation is refused, by name");
    // 4. a half-spelled name (a tile short) is refused
    threw = false;
    try { entrypoint("affine_qmm_t", {affine(g64b4)}); }
    catch (const std::exception&) { threw = true; }
    ok(threw, "qmm_t without a tile is refused");
    // ── the launch shapes that came down with §7 step 3 ──────────────────
    //
    // Pinned rather than described, because they are the numbers a wrong move
    // would change silently: a threadgroup that is not a whole simdgroup
    // leaves the router's cross-simdgroup reduction reading an uninitialised
    // slot, and a routed matvec at grid.y = N runs `out_row` to 8N.
    {
        using namespace pie::kernels;
        using namespace pie::kernels::moe;
        Grid g; Threadgroup tg;

        ok(router_lane_width(32) == 32, "router width, 32 experts");
        ok(router_lane_width(128) == 128, "router width, 128 experts");
        ok(router_lane_width(33) == 64, "router width rounds to a simdgroup");
        ok(router_lane_width(0) == 32, "router width has a floor");
        ok(router_lane_width(5000) == 1024, "router width caps at the tg limit");

        router_topk_dispatch(128, g, tg, 4);
        ok(g.x == 128 && g.y == 4 && tg.x == 128, "router_topk shape");

        expert_combine_dispatch(4096, g, tg, 2);
        ok(g.x == 4096 && g.y == 2 && tg.x == 256, "expert_combine caps tg at 256");

        // N=100 -> ceil(100/4) = 25 on y; 32 lanes per row on x; slot on z.
        routed_qmv_dispatch(100, 8, g, tg, 3);
        ok(g.x == 96 && g.y == 25 && g.z == 8, "routed qmv grid");
        ok(tg.x == 32 && tg.y == 2, "routed qmv needs TWO simdgroups");

        route_sort_dispatch(64, g, tg);
        ok(g.x == 64 && g.y == 1 && tg.x == 64, "route_sort is one threadgroup");

        route_rows_dispatch(2048, 48, g, tg);
        ok(g.x == 2048 && g.y == 48 && tg.x == 256, "route_rows shape");

        ok(bm_slot(16) == 0 && bm_slot(32) == 1 && bm_slot(64) == 2, "bm slots");

        // The bound: every touched expert can waste tile-1 rows, rounded up.
        ok(sorted_rows(8, 128, 1) == 8, "tile 1 wastes nothing");
        ok(sorted_rows(64, 8, 16) == 192, "64 pairs over 8 experts at tile 16");
        ok(sorted_rows(0, 128, 16) == 0, "empty batch");

        elementwise_dispatch(10, g, tg);
        ok(g.x == 10 && tg.x == 10, "elementwise below the cap");
        elementwise_dispatch(100000, g, tg);
        ok(g.x == 100000 && tg.x == 256, "elementwise above the cap");
    }

    std::printf(fail ? "\n%d FAILED\n" : "\nall ok\n", fail);
    return fail != 0;
}

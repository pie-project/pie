// #24 graph_variant bitfield — host unit test (deterministic, no GPU, no RNG).
//
// A GPU graph-replay verify CANNOT exhibit this bug: the cache is decode-only
// and real `graph_layout` ∈ 48..63, so `63<<3 = 0x1F8 < 0x200` — no real config
// ever aliases. The honest "prove-the-bug-would-have-fired" check for a LATENT
// collision is a boundary unit test: at `graph_layout=64` the OLD encoding
// `(64<<3)==0x200` aliases `small_spec`; the NEW `make_graph_variant` keeps them
// distinct. We also sweep the full layout range to (a) prove the OLD encoding
// actually collides over it (non-degenerate) and (b) prove the NEW one is
// collision-free.

#include <cstdint>
#include <cstdio>
#include <set>

#include "executor/graph_variant.hpp"

using pie_cuda_driver::gv_old_encode_for_proof;
using pie_cuda_driver::make_graph_variant;

namespace {
int g_failures = 0;
void check(bool cond, const char* what) {
    std::printf("%s %s\n", cond ? "[ ok ]" : "[FAIL]", what);
    if (!cond) ++g_failures;
}
}  // namespace

int main() {
    // ── 1. The bug WAS real at the one-bump-away boundary (graph_layout=64) ──
    // OLD: {layout=64, no flags} and {layout=0, small_spec} both hash to 0x200.
    check(gv_old_encode_for_proof(64u, false, false) ==
              gv_old_encode_for_proof(0u, true, false),
          "OLD encoding ALIASES graph_layout=64 with small_spec (the latent bug)");
    // OLD: layout=128 reaches bit 10 → aliases rs_verify (0x400).
    check(gv_old_encode_for_proof(128u, false, false) ==
              gv_old_encode_for_proof(0u, false, true),
          "OLD encoding ALIASES graph_layout=128 with rs_verify");

    // ── 2. The FIX keeps those distinct ──
    check(make_graph_variant(false, false, false, false, false, 64u) !=
              make_graph_variant(false, false, false, true, false, 0u),
          "NEW: graph_layout=64 is DISTINCT from small_spec");
    check(make_graph_variant(false, false, false, false, false, 128u) !=
              make_graph_variant(false, false, false, false, true, 0u),
          "NEW: graph_layout=128 is DISTINCT from rs_verify");

    // ── 3. NEW encoding: exhaustive uniqueness over (5 flags) × (layout 0..255) ──
    {
        std::set<std::uint32_t> seen;
        bool unique = true;
        for (std::uint32_t layout = 0; layout <= 255u; ++layout) {
            for (int f = 0; f < 32; ++f) {
                const std::uint32_t v = make_graph_variant(
                    (f & 1) != 0, (f & 2) != 0, (f & 4) != 0,
                    (f & 8) != 0, (f & 16) != 0, layout);
                if (!seen.insert(v).second) unique = false;
            }
        }
        check(unique, "NEW: all (flags × layout 0..255) graph_variants are UNIQUE");
    }

    // ── 4. OLD encoding MUST collide over the same range (non-degenerate proof:
    //       the test exercises a range that genuinely aliased pre-fix) ──
    {
        std::set<std::uint32_t> seen;
        bool collides = false;
        for (std::uint32_t layout = 0; layout <= 255u; ++layout) {
            for (int f = 0; f < 4; ++f) {  // small_spec(0x200) × rs_verify(0x400)
                const std::uint32_t v =
                    gv_old_encode_for_proof(layout, (f & 1) != 0, (f & 2) != 0);
                if (!seen.insert(v).second) collides = true;
            }
        }
        check(collides,
              "OLD encoding DOES collide over layout 0..255 (proof is non-degenerate)");
    }

    std::printf("\n%s (%d failures)\n",
                g_failures == 0 ? "ALL PASS" : "FAILURES", g_failures);
    return g_failures == 0 ? 0 : 1;
}

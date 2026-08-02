// llama_mb_test — the M>1 path's shape and tiling invariants, without a GPU.
//
// The llama family has ONE `launch_shape` for M=1 and M>1, where gpt-oss and
// gemma4 each carry a second `launch_shape_mb`. That is the cheaper structure
// only if the single function really does reduce to the M=1 shapes at rows=1,
// so the first thing here pins that: every kind, compared against the shapes
// the M=1 path is already known-good at, because `llama_numerics_test` runs
// them on a device.
//
// The second thing is the invariant that the tiling decision is made in two
// places -- `launch_shape` computes a grid for it and `pso_for` selects a
// pipeline compiled for it -- and those two must never disagree. A grid built
// for BN=32 running a BN=64 pipeline is not a crash. It is wrong numbers, in a
// GEMM, on a prefill, which is the hardest kind of wrong to find later.
//
// The third is that the GEMM's row padding stays inside the pool. The padding
// rows are real writes to real memory; if the scratch is sized at the batch
// rather than at the padded batch, they land past the end of the buffer.

#include <cstdio>
#include <string>
#include <vector>

#include "../src/model/llama/bind.hpp"
#include "../src/model/llama/decode_consts.hpp"
#include "../src/model/llama/decode_step.hpp"
#include "../src/model/llama/encode.hpp"
#include "../src/model/llama/geometry.hpp"
#include "../src/model/llama/scratch.hpp"
#include "../src/model/qwen3_5/decode_dispatch.hpp"
#include "../src/model/qwen3_5/decode_dispatch_mb.hpp"

namespace {

using namespace pie::metal;
using pie::metal::llama::Dispatch;
using pie::metal::llama::Kind;
using pie::metal::llama::KN;
using pie::metal::llama::LlamaGeometry;
using pie::metal::llama::ScratchPlan;
using pie::metal::llama::build_llama_dag;
using pie::metal::llama::build_llama_scratch;
using pie::metal::llama::color_llama_scratch;
using pie::metal::llama::launch_shape;
using pie::metal::llama::llama_pool_elems;
using pie::metal::llama::llama_qmm_bn;
using pie::metal::llama::llama_qmm_pool_rows;
using pie::metal::llama::llama_qmm_rows;
using pie::metal::llama::qmv_kn;
using pie::metal::llama::routed_qmv_dispatch;
using pie::metal::llama::elementwise_dispatch;

int g_pass = 0;
int g_fail = 0;

void expect(bool ok, const std::string& what) {
    if (ok) {
        ++g_pass;
        std::printf("  PASS  %s\n", what.c_str());
    } else {
        ++g_fail;
        std::printf("  FAIL  %s\n", what.c_str());
    }
}

void expect_eq(long got, long want, const std::string& what) {
    if (got == want) {
        ++g_pass;
        std::printf("  PASS  %s\n", what.c_str());
    } else {
        ++g_fail;
        std::printf("  FAIL  %s: got %ld, want %ld\n", what.c_str(), got, want);
    }
}

LlamaGeometry base() {
    LlamaGeometry g;
    g.hidden = 512;
    g.n_layers = 2;
    g.vocab = 1024;
    g.n_q_heads = 4;
    g.n_kv_heads = 2;
    g.head_dim = 128;
    g.intermediate = 512;
    g.kv_max_ctx = 64;
    g.rope_theta = 10000.0f;
    return g;
}

LlamaGeometry moe() {
    LlamaGeometry g = base();
    g.qk_norm = true;
    g.n_experts = 8;
    g.experts_per_token = 2;
    g.moe_intermediate = 512;
    return g;
}

bool same(const Grid& a, const Grid& b) { return a.x == b.x && a.y == b.y && a.z == b.z; }
bool same(const Threadgroup& a, const Threadgroup& b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

/// At one row the unified shape must equal the shape the M=1 path is verified
/// at. This is the whole licence for having one function instead of two.
void check_degenerates_to_m1(const char* who, const LlamaGeometry& g) {
    std::printf("\n-- %s: rows=1 is the M=1 shape --\n", who);
    const std::vector<Dispatch> dag = build_llama_dag(g, /*with_argmax=*/false);

    int mismatched = 0;
    for (const Dispatch& d : dag) {
        Grid got{};
        Threadgroup got_tg{};
        launch_shape(d, g, got, got_tg, /*rows=*/1, /*head_rows=*/1);

        // The independently-stated M=1 answer, from the shared primitives.
        Grid want{};
        Threadgroup want_tg{};
        const KN kn = qmv_kn(d.kind, g);
        if (kn.N != 0) {
            // The routed three run on the SORTED rows -- one per (token, slot)
            // pair, with no expert axis, because `moe_route_sort` put the pair
            // in the row index and `row_expert` replaced `tid.z`. At one row
            // that is eight rows and one slot where it used to be one row and
            // eight slots: the same threads, decomposed by what the kernel now
            // indexes by.
            const bool routed = d.kind == Kind::ExpertGate || d.kind == Kind::ExpertUp ||
                                d.kind == Kind::ExpertDown;
            routed_qmv_dispatch(kn.N, 1, want, want_tg,
                                routed ? llama_moe_sorted_rows(g, 1) : 1);
        } else {
            switch (d.kind) {
                case Kind::EmbedGather: embed_dispatch(g.hidden, want, want_tg); break;
                case Kind::AttnNorm:
                case Kind::FfnNorm:
                case Kind::FinalRms: rms_dispatch(g.hidden, 1, want, want_tg); break;
                case Kind::QNorm: rms_dispatch(g.head_dim, g.n_q_heads, want, want_tg); break;
                case Kind::KNorm: rms_dispatch(g.head_dim, g.n_kv_heads, want, want_tg); break;
                case Kind::AttnResidual:
                case Kind::FfnResidual: residual_dispatch(g.hidden, want, want_tg); break;
                case Kind::RopeQ:
                    rope_dispatch(g.rotary_dims(), g.n_q_heads, want, want_tg);
                    break;
                case Kind::RopeK:
                    rope_dispatch(g.rotary_dims(), g.n_kv_heads, want, want_tg);
                    break;
                case Kind::KvAppend:
                    kv_append_dispatch(g.head_dim, g.n_kv_heads, want, want_tg);
                    break;
                case Kind::Sdpa: sdpa_dispatch(g.n_q_heads, want, want_tg); break;
                case Kind::SiluMul: elementwise_dispatch(g.intermediate, want, want_tg); break;
                case Kind::ExpertSiluMul:
                    elementwise_dispatch(g.moe_intermediate * llama_moe_sorted_rows(g, 1), want,
                                         want_tg);
                    break;
                default: continue;  // routed odds and ends, covered by their own arms
            }
        }
        if (!same(got, want) || !same(got_tg, want_tg)) {
            if (mismatched < 8) {
                std::printf("    kind=%d ord=%d: got grid(%u,%u,%u) tg(%u,%u,%u) "
                            "want grid(%u,%u,%u) tg(%u,%u,%u)\n",
                            int(d.kind), d.ordinal, got.x, got.y, got.z, got_tg.x, got_tg.y,
                            got_tg.z, want.x, want.y, want.z, want_tg.x, want_tg.y, want_tg.z);
            }
            ++mismatched;
        }
    }
    expect(mismatched == 0, std::string(who) + ": every kind reduces to its M=1 shape");
}

/// The grid and the pipeline are chosen by two functions and must agree.
void check_tiling_agrees(const char* who, const LlamaGeometry& g) {
    std::printf("\n-- %s: the grid and the pipeline agree on the tile --\n", who);
    const std::vector<Dispatch> dag = build_llama_dag(g, /*with_argmax=*/false);

    int disagreed = 0;
    int gemm_rows_seen = 0;
    for (int rows = 1; rows <= 64; ++rows) {
        for (const int requests : {1, rows == 64 ? 64 : 1}) {
            for (const Dispatch& d : dag) {
                const int m = rows;
                const int bn = llama_qmm_bn(d.kind, g, m, requests);
                if (bn == 0) continue;
                ++gemm_rows_seen;
                // `launch_shape` builds its grid from exactly these; recompute them
                // the way `pso_for` does and require the same answer.
                Grid grid{};
                Threadgroup tg{};
                launch_shape(d, g, grid, tg, rows, rows, requests);
                Grid want{};
                Threadgroup want_tg{};
                // The row block comes from the BATCH. Asking it of the padded count
                // is a different question once there are three rungs -- 40 rows pad
                // to 64, and 64 answers "BM=64" while the grid was built for 32.
                //
                // And the split comes first, because a split projection launches a
                // grid that is `split` deep in z against a pipeline that writes
                // partials: choosing the split here and the plain grid there leaves
                // the projection's real output buffer untouched.
                const int bm = rows == 64 && requests == 64 ? 32 : qmm_bm(m);
                if (const int split = llama_qmm_split(d.kind, g, m, requests); split > 1) {
                    qmm_t_splitk_dispatch(qmv_kn(d.kind, g).N,
                                          llama_qmm_rows(m, requests), bm, split,
                                          want, want_tg);
                } else {
                    qmm_t_dispatch(qmv_kn(d.kind, g).N,
                                   llama_qmm_rows(m, requests), bn, bm, want, want_tg);
                }
                if (!same(grid, want) || !same(tg, want_tg)) ++disagreed;
            }
        }
    }
    expect(gemm_rows_seen > 0, std::string(who) + ": the GEMM engages somewhere in 1..64 rows");
    expect(disagreed == 0, std::string(who) + ": every GEMM grid matches its selected tile");
}

/// The routed matvecs must NEVER become a GEMM: each row picks its own experts,
/// so a tile spanning rows would span weights.
void check_routed_never_gemm(const LlamaGeometry& g) {
    std::printf("\n-- routed projections stay matvecs at every batch --\n");
    int became = 0;
    for (int rows = 1; rows <= 128; ++rows) {
        became += llama_qmm_bn(Kind::ExpertGate, g, rows) != 0;
        became += llama_qmm_bn(Kind::ExpertUp, g, rows) != 0;
        became += llama_qmm_bn(Kind::ExpertDown, g, rows) != 0;
        became += llama_qmm_bn(Kind::Router, g, rows) != 0;
    }
    expect_eq(became, 0, "no routed kind, and not the router, ever tiles");
}

/// The GEMM writes its padding rows for real. They must fit.
void check_padding_fits() {
    std::printf("\n-- the GEMM's padding stays inside the pool --\n");
    int over = 0;
    for (int max_rows = 1; max_rows <= 256; ++max_rows) {
        const int pool = llama_qmm_pool_rows(max_rows);
        if (pool < max_rows) ++over;
        for (int rows = 1; rows <= max_rows; ++rows) {
            if (llama_qmm_rows(rows) > pool) ++over;
        }
    }
    expect_eq(over, 0, "every batch's padded row count fits the pool sized for the maximum");

    // Below the GEMM's minimum batch the matvec runs and there is no padding at
    // all -- padding there would be pure waste on the decode path, which is
    // every step of a normal generation.
    expect_eq(llama_qmm_rows(1), 1, "a decode is not padded");
    expect_eq(llama_qmm_rows(kQmmMinBatch - 1), kQmmMinBatch - 1,
              "nothing below the GEMM's minimum batch is padded");
    expect(llama_qmm_rows(kQmmMinBatch) >= kQmmMinBatch,
           "at the minimum batch the rows round up to a tile");
}

/// The row axis really does widen the row-parallel kinds.
void check_rows_widen(const LlamaGeometry& g) {
    std::printf("\n-- the row count reaches the row-parallel kinds --\n");
    const std::vector<Dispatch> dag = build_llama_dag(g, /*with_argmax=*/false);
    auto shape_of = [&](Kind k, int rows) {
        Grid grid{};
        Threadgroup tg{};
        for (const Dispatch& d : dag) {
            if (d.kind != k) continue;
            launch_shape(d, g, grid, tg, rows, rows);
            break;
        }
        return grid;
    };
    expect_eq(shape_of(Kind::AttnResidual, 8).x, g.hidden * 8,
              "residual: one thread per (row, channel)");
    expect_eq(shape_of(Kind::AttnNorm, 8).x, (g.hidden / 4) * 8,
              "norm: one threadgroup per row");
    expect_eq(shape_of(Kind::EmbedGather, 8).y, 8, "embed: the row is grid.y");
    expect_eq(shape_of(Kind::RopeQ, 8).z, 8, "rope: the row is grid.z");
    expect_eq(shape_of(Kind::KvAppend, 8).z, 8, "kv_append: the row is grid.z");
    expect_eq(shape_of(Kind::Sdpa, 8).y, 8, "sdpa: the query row is grid.y");
    expect_eq(shape_of(Kind::SiluMul, 8).x, g.intermediate * 8,
              "silu_mul: flat over rows x intermediate");
}

/// The tail runs on the COMPACTED rows, not on every token. Getting this wrong
/// is the most expensive dispatch in the step run 100x too wide.
void check_head_rows(const LlamaGeometry& g) {
    std::printf("\n-- the tail runs on the sampled rows --\n");
    const std::vector<Dispatch> dag = build_llama_dag(g, /*with_argmax=*/false);
    auto shape_of = [&](Kind k, int rows, int head_rows) {
        Grid grid{};
        Threadgroup tg{};
        for (const Dispatch& d : dag) {
            if (d.kind != k) continue;
            launch_shape(d, g, grid, tg, rows, head_rows);
            break;
        }
        return grid;
    };
    expect_eq(shape_of(Kind::FinalRms, 64, 2).x, (g.hidden / 4) * 2,
              "final norm: two sampled rows, not sixty-four");
    expect_eq(shape_of(Kind::RowGather, 64, 2).y, 2, "row_gather: gathers the sampled rows");
    // The body still runs on all of them.
    expect_eq(shape_of(Kind::AttnNorm, 64, 2).x, (g.hidden / 4) * 64,
              "the body is unaffected by the sample count");
}

/// Sizing the activation pool at the batch the family will actually run.
void check_pool_scales(const LlamaGeometry& g) {
    std::printf("\n-- the pool scales with the batch --\n");
    const std::vector<Dispatch> dag = build_llama_dag(g, /*with_argmax=*/false);
    const ScratchPlan plan = build_llama_scratch(dag, g);
    const model::ScratchColoring col = color_llama_scratch(dag, plan);
    const std::vector<std::size_t> one = llama_pool_elems(dag, plan, col, g, 1, 1);
    const std::vector<std::size_t> many = llama_pool_elems(dag, plan, col, g, 16, 2);
    expect_eq((long)one.size(), (long)many.size(), "the same colours at any batch");
    bool grew = true;
    for (std::size_t i = 0; i < one.size(); ++i) {
        if (many[i] < one[i]) grew = false;
    }
    expect(grew, "no colour shrinks when the batch grows");
}

}  // namespace

int main() {
    std::printf("llama_mb_test — the M>1 shapes, and the tiling they must agree on\n");

    check_degenerates_to_m1("llama-3 (dense)", base());
    check_degenerates_to_m1("qwen3-moe (routed)", moe());
    check_tiling_agrees("llama-3 (dense)", base());
    check_routed_never_gemm(moe());
    check_padding_fits();
    check_rows_widen(base());
    check_head_rows(base());
    check_pool_scales(moe());

    std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

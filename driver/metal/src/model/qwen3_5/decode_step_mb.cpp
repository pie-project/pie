#include <cstdio>
#include <cstdlib>
#include <iostream>
#include "decode_step_mb.hpp"

#include <limits>
#include <stdexcept>

#include "decode_dispatch.hpp"
#include "decode_consts.hpp"
#include "decode_dispatch_mb.hpp"
#include "heap_bind.hpp"
#include "batch/scratch.hpp"
#include "batch/decode_timing.hpp"

namespace pie::metal {
namespace {

Kernel mb_kind(Kernel k) {
    switch (k) {
        case Kernel::GdnPrep: return Kernel::GdnPrepSlotted;
        case Kernel::GdnCore: return Kernel::GdnCoreSlotted;
        case Kernel::KvAppend: return Kernel::KvAppendPaged;
        case Kernel::Sdpa: return Kernel::SdpaPaged;
        default: return k;
    }
}

}  // namespace

int qmv_out_size(Kernel k, const DecodeGeometry& g) {
    switch (k) {
        case Kernel::QmvIn: return g.gdn_conv_dim;
        case Kernel::QmvInZ: return g.gdn_v_total;
        case Kernel::GdnInA:
        case Kernel::GdnInB: return g.gdn_v_heads;
        case Kernel::QmvOut:
        case Kernel::QmvO:
        case Kernel::QmvDown:
        case Kernel::LlSharedDown: return g.hidden;
        case Kernel::QmvQ: return 2 * g.n_q_heads * g.head_dim;
        case Kernel::QmvK:
        case Kernel::QmvV: return g.n_kv_heads * g.head_dim;
        case Kernel::QmvGate:
        case Kernel::QmvUp: return g.intermediate;
        case Kernel::QmvLmHead:
        case Kernel::LmHeadUntied: return g.vocab;
        // The router is a DENSE matvec into one logit per expert: it runs over
        // the tokens like any other projection, and only what follows it is
        // routed. The three expert projections are deliberately absent -- they
        // run over the SORTED rows, and answering here would launch them over
        // the token count instead, computing the first `n` sorted rows and
        // leaving the rest of the stack holding the previous layer's output.
        case Kernel::LlRouter: return g.n_experts;
        // The shared expert is dense in every sense -- it runs over the tokens,
        // one row each, so it answers here like any other projection.
        case Kernel::LlSharedGate:
        case Kernel::LlSharedUp: return g.shared_intermediate;
        case Kernel::LlSharedGateProj: return 1;
        default: return 0;
    }
}

namespace {
bool g_ab_arm = false;
}  // namespace

bool ab_all_barriers() {
    static const bool on = std::getenv("PIE_METAL_AB_BARRIERS") != nullptr;
    return on;
}

bool ab_enabled() {
    static const bool on = std::getenv("PIE_METAL_AB") != nullptr;
    return on;
}
bool ab_arm() { return g_ab_arm; }
void ab_set_arm(bool b) { g_ab_arm = b; }

namespace {

void mb_geometry(Dispatch& d, const DecodeGeometry& g, int n) {
    auto rms = [&](int row, int rows) { rms_mb_dispatch(row, rows, n, d.grid, d.tg); };
    if (const int out = qmv_out_size(d.kind, g); out != 0) {
        // `qmm_bn_unsplit`, not `qmm_bn`: the widest-tile rule is correct only
        // where split-K supplies the threadgroups the wide tile gives up, and
        // this family dispatches no split -- see the comment on `qmm_split`
        // just below, which is the same reason gemma4 and gpt-oss have.
        // Padded to a whole row tile: `qmm_mb_rows` explains why, and why the
        // padding is asked for the GEMM's three questions (does a tile exist,
        // which rung, what grid) and for nothing else in this dispatch.
        const int qn = qmm_mb_rows(n, g.max_tokens, qmm_min_batch(g.is_moe(), qwen35_fp16_format(g)));
        d.qmm_bn = qmm_bn_unsplit(out, qn, qmm_min_batch(g.is_moe(), qwen35_fp16_format(g)));
        d.qmm_bm = qmm_bm(qn);
        // The second quantized set has a matvec and no GEMM: a checkpoint that
        // spares its two routing projections at 8 bits gets one extra pipeline
        // table, not an extra table of every batched shape.
        //
        // Measured, because a router runs once per prompt ROW and 40 layers of
        // them is not obviously free: building the strided GEMM at the second
        // format too and putting the router back on the batched path moved a
        // 128-token prefill of Qwen3.6-35B-A3B from 222.4 to 223.7 tok/s, which
        // is inside the run-to-run spread. The dispatch trace names it 8.5% of
        // GPU time and that share is a decode's, where the batched path does
        // not apply. So the second table stays two pipelines.
        if (qwen35_uses_alt_quant(d.kind, g)) d.qmm_bn = 0;
        // NO split-K, for exactly the reason gemma4's `launch_shape_mb` gives:
        // the split GEMM writes `split_k` partial [M, N] slices into a side
        // buffer and needs a reduce pass to sum them, and NOTHING IN THIS
        // DRIVER EVER DISPATCHES THAT PASS. `qmm_splitk_reduce` is compiled and
        // has no caller; `affine_qmm_t_splitk` writes its result to buffer 8,
        // the partials, and buffer 4 -- the projection's real output -- keeps
        // whatever the last fire left in it.
        //
        // It survived because the split only engages at `qmm_bn != 0`, which
        // needs a batch of at least `qmm_min_batch()`, and nothing fired one until
        // the throughput harness did. With the harness's fleet check it is a
        // one-line reproduction: sixteen copies of one prompt in one fire
        // answer 74088 and 1125 at step 0 with the split on, and agree with the
        // single-sequence decode for all 64 steps with it off.
        //
        // The measured cost of turning it off is the honest version of a number
        // that was never real: 717 tok/s becomes 520 at sixteen lanes, and 717
        // was the speed of computing the wrong answer.
        //
        // The fix is to emit the reduce, which this family cannot do here --
        // `mb_geometry` decides the split while walking a DAG that is already
        // built, so it has no way to insert a dispatch after itself.
        d.qmm_split = 1;
        if (d.qmm_split > 1)
            qmm_t_splitk_dispatch(out, qn, d.qmm_bm, d.qmm_split, d.grid, d.tg);
        else if (d.qmm_bn > 0)
            qmm_t_dispatch(out, qn, d.qmm_bn, d.qmm_bm, d.grid, d.tg);
        else
            qmv_mb_dispatch(out, n, d.grid, d.tg);
        return;
    }
    switch (d.kind) {
        case Kernel::EmbedUntied:
        case Kernel::EmbedGather:
            embed_mb_dispatch(g.hidden, n, d.grid, d.tg); break;
        case Kernel::Rms:
        case Kernel::FfnRms:
        case Kernel::FinalRms:
            rms(g.hidden, 1); break;
        case Kernel::QNorm:
            rms(g.head_dim, g.n_q_heads); break;
        case Kernel::KNorm:
            rms(g.head_dim, g.n_kv_heads); break;
        case Kernel::GdnPrepSlotted:
            d.grid = Grid{32u, 1u, uint32_t(n * g.gdn_v_heads)};
            d.tg = Threadgroup{32, 1, 1};
            break;
        case Kernel::GdnCoreSlotted:
            d.grid = Grid{32u, uint32_t(g.gdn_v_dim), uint32_t(n * g.gdn_v_heads)};
            d.tg = Threadgroup{32, 4, 1};
            break;
        case Kernel::GatedRms:
            d.grid = Grid{uint32_t(g.gdn_v_dim), uint32_t(g.gdn_v_heads), uint32_t(n)};
            d.tg = Threadgroup{uint32_t(g.gdn_v_dim), 1, 1};
            break;
        case Kernel::QSplit:
            d.grid = Grid{uint32_t(g.head_dim), uint32_t(g.n_q_heads), uint32_t(n)};
            d.tg = Threadgroup{uint32_t(g.head_dim), 1, 1};
            break;
        case Kernel::Rope:
            rope_mb_dispatch(g.rotary_dims, g.n_q_heads, n, d.grid, d.tg); break;
        case Kernel::RopeK:
            rope_mb_dispatch(g.rotary_dims, g.n_kv_heads, n, d.grid, d.tg); break;
        case Kernel::KvAppendPaged:
            kv_append_mb_dispatch(g.head_dim, g.n_kv_heads, n, d.grid, d.tg); break;
        case Kernel::SdpaPaged:
            sdpa_paged_dispatch(g.n_q_heads, n, d.grid, d.tg); break;
        case Kernel::AttnGate:
            elementwise_mb_dispatch(g.n_q_heads * g.head_dim, n, d.grid, d.tg); break;
        case Kernel::Residual:
        case Kernel::LayerOut:
            elementwise_mb_dispatch(g.hidden, n, d.grid, d.tg); break;
        case Kernel::SiluMul:
            // Routed, the dense SwiGLU that remains is the SHARED expert's --
            // one row a token, at its own width. The mixture's own SwiGLU is
            // `LlExpertSiluMul` below, over the sorted stack, and the two were
            // split precisely because this line cannot say both.
            elementwise_mb_dispatch(g.is_moe() ? g.shared_intermediate : g.intermediate,
                                    n, d.grid, d.tg);
            break;
        case Kernel::LlExpertSiluMul:
            // The slot axis is gone -- a sorted row IS a slot -- so this is the
            // same elementwise shape over a taller batch.
            elementwise_mb_dispatch(g.moe_intermediate,
                                    moe_sorted_rows(g, n, qwen35_routed_decode_batched()),
                                    d.grid, d.tg);
            break;

        // ── the mixture ──
        // The three expert projections first, because `qmv_out_size` does not
        // answer for them: they run over the sorted rows, which is neither `n`
        // nor `n * k`, because the sort pads every expert's run to a whole
        // tile. Once the batch fills a tile they become matmuls, and the tile
        // is `moe_tile_rows`'s answer -- the same number the sort padded to,
        // asked of the same function rather than restated.
        case Kernel::LlExpertGate:
        case Kernel::LlExpertUp:
        case Kernel::LlExpertDown: {
            const int N = d.kind == Kernel::LlExpertDown ? g.hidden : g.moe_intermediate;
            // Asked with the SAME answer `bind_token_consts` gives the sort, so
            // the stack the dispatch walks is the stack the sort built.
            const int sorted = moe_sorted_rows(g, n, qwen35_routed_decode_batched());
            // WRONG ANSWERS, so this arm is currently shut. NOT a fleet bug and
            // NOT a width bug, whatever the note that used to stand here said.
            //
            // A ONE-ROW GREEDY DECODE REPRODUCES IT:
            //
            //   PIE_METAL_MOE_BATCH_MIN_PER_EXPERT=0 \
            //   PIE_METAL_QWEN_ROUTED_DECODE_GEMM=1 \
            //   PIE_METAL_EXPERT_SLAB_MB=1024 ./bin/llama_bench <q35> 32 32
            //
            // answers `220 96834 45163 149960 149960 97282` where mlx-lm says
            // `220 24 11 220 16 15`. Either env alone PASSES; only the pair
            // fails, which is the definition of this branch and nothing else.
            //
            // The bisection this comment used to carry -- "30 and 31 lanes
            // PASS, 32 FAILS", "MIN_PER=2 makes 32 pass", "forced on at 8 lanes
            // it reproduces mlx-lm", "reproducing the wrong answer needs a
            // bigger machine" -- was ENTIRELY FALSE, and both halves of why are
            // worth carrying:
            //
            //   * `env_int` rejected 0, so every `MIN_PER=0` run silently used
            //     the default 1 and never left the matvec. That is why "forced
            //     on at 8 lanes" passed: nothing was forced. See
            //     `env_int_allow_zero` in device_tuning.cpp.
            //   * The widths that "passed" never reached this branch either.
            //     Two independent filters gate it, and 32 is where they happen
            //     to coincide on THIS mixture: `moe_should_batch` first admits
            //     at 256 pairs = 32 lanes at top-8, and `qmm_bn` returns 0 --
            //     silently demoting to the matvec below -- whenever `sorted` is
            //     not BN-aligned. 32 lanes sort to 4096 rows (aligned, GEMM);
            //     33 sort to 4112 = 16*257 (not aligned, matvec). Traced with
            //     `PIE_METAL_MOE_TRACE=1`: 33 lanes report `bn=0`.
            //
            // So the honest statement is that the qwen3.5 routed decode GEMM
            // has NEVER produced a right answer at any width tested. 32 was an
            // artefact of the two gates, not a threshold.
            //
            // What has been ruled out since, all measured:
            //   * NOT expert paging: fails with and without the slab.
            //   * NOT the tile width: at one row `moe_tile_rows` pins BM=16,
            //     and `PIE_METAL_MOE_TILE_MID_PER=1` moves the PREFILL to
            //     BM=32 while the decode stays at 16 and stays wrong.
            //   * NOT a silent PSO fallback to the matvec: the trace reports
            //     `routed=1`, so `mb.qmm_routed[0][2]` is valid and selected.
            //   * NOT the kernel. The PREFILL fires the SAME PSO over
            //     `sorted=5376` and reproduces mlx-lm. Same scratch schedule,
            //     same weights, same `tile_expert` contract.
            //   * NOT the shared kernels: gpt-oss (32 experts, top-4) and
            //     gemma-4-26b-a4b both PASS at one row with `MIN_PER=0` now
            //     that the knob is honoured.
            //   * NOT the recurrent state, and NOT the dense GEMM
            //     (`PIE_METAL_QMM_MIN_BATCH=64` does not help).
            //
            // That leaves what the decode binds or passes, which is where to
            // look next. The obvious instrument does not work here: golden taps
            // turn pool recycling off, which takes scratch from 0.184 to 2.616
            // GiB and puts an 18.16 GiB checkpoint over the GPU's limit
            // (`kIOGPUCommandBufferCallbackErrorOutOfMemory`).
            // `PIE_METAL_TAPS_RECYCLE=1` dumps, but only the LAST layer's
            // values survive the recycling and by then both arms have diverged.
            // Truncating the checkpoint to one layer via `num_hidden_layers`
            // does not help either: the heap is reserved from the checkpoint's
            // tensors, not from the geometry, so a 1-layer DAG still asks for
            // 18.26 GiB and the paging staging refuses `1 routed layers in the
            // DAG against 40 staged`.
            //
            // Not a reason to raise `moe_batch_min_per_expert` past 1, which
            // is the shape of fix that suggests itself: that constant carries
            // a measured table in `device_tuning.hpp` where 1 beats 4 by up to
            // 132% on the families whose arm IS on. This is a qwen3.5 bug, and
            // the crossover is not the thing that is wrong.
            //
            // Shutting it costs a mixture's decode the batched form wherever
            // both gates would have admitted it, which on this mixture is 32
            // lanes and up. That is a throughput loss on one family and it is
            // the correct trade against a wrong answer.
            //
            // The prefill is deliberately unaffected: its routed batching is
            // `routed_group_shape`, a different call site, and these DAGs are
            // built at `n_tokens == 1` where the predicate is false anyway.
            if (const int bn = qmm_bn(N, sorted, qmm_min_batch(true, qwen35_fp16_format(g)));
                qwen35_routed_decode_batched() && bn > 0 &&
                shared_kernels::moe_should_batch(n * g.experts_per_token, g.n_experts)) {
                d.qmm_bn = bn;
                d.qmm_bm = shared_kernels::moe_tile_rows(n * g.experts_per_token, g.n_experts);
                d.qmm_split = 1;
                qmm_t_dispatch(N, sorted, bn, d.qmm_bm, d.grid, d.tg);
            } else {
                // One sorted row per (token, slot) pair and no expert axis: the
                // pair's expert is `row_expert[p]`, not `tid.z`.
                shared_kernels::routed_qmv_dispatch(N, 1, d.grid, d.tg, sorted);
            }
            if (moe_trace_enabled() && d.kind == Kernel::LlExpertGate) {
                std::fprintf(stderr,
                             "[moe] gemm  n=%d N=%d sorted=%d bm=%d bn=%d"
                             " grid=%ux%ux%u tg=%ux%ux%u\n",
                             n, N, sorted, d.qmm_bm, d.qmm_bn, d.grid.x, d.grid.y, d.grid.z,
                             d.tg.x, d.tg.y, d.tg.z);
            }
            break;
        }
        case Kernel::GoRouterTopK:
            shared_kernels::router_topk_dispatch(g.n_experts, d.grid, d.tg, n); break;
        case Kernel::LlMoeSort:
            shared_kernels::moe_route_sort_dispatch(g.n_experts, d.grid, d.tg); break;
        case Kernel::LlMoeGather:
            // The stack this fills is the one the sort writes, so it has to
            // ask the same question the sort did. Launched over the padded
            // count with an unpadded sort behind it, the gather walks rows the
            // sort never grouped -- which at 512 tokens is 12800 rows of copy
            // for 4096 of content, and is exactly the cost
            // `qwen35_routed_decode_batched` was turned off to stop paying.
            shared_kernels::moe_route_rows_dispatch(
                g.hidden, moe_sorted_rows(g, n, qwen35_routed_decode_batched()),
                d.grid, d.tg);
            break;
        case Kernel::LlMoeCombine:
            shared_kernels::expert_combine_dispatch(g.hidden, d.grid, d.tg, n); break;
        case Kernel::LlSharedCombine:
            // The SAME helper the M=1 DAG emits this with, and not the flat
            // `elementwise_mb_dispatch` its neighbours take: `shared_expert_combine`
            // reads its row from `gid.y`, so a flat (width * N, 1, 1) grid puts
            // every thread on row 0 and leaves rows 1.. holding whatever the
            // pool did. At N=1 the two shapes are the same grid, which is why
            // this survived every single-sequence gate and broke a fleet of two
            // at its first step.
            shared_kernels::moe_route_rows_dispatch(g.hidden, n, d.grid, d.tg); break;

        default:
            throw std::runtime_error("missing multi-batch launch geometry");
    }
}

/// The dispatches the mixture's routing owns, in DAG order.
///
/// Everything from the top-k down to the combine: what they share is that their
/// extent is the SORTED stack or the pair list, neither of which is a row count
/// the per-token walk can supply. The router's own projection is not here -- it
/// is an ordinary dense matvec and the strided GEMM already batches it -- and
/// neither is the shared expert's, for the same reason.
bool is_routed_group(Kernel k) {
    switch (k) {
        case Kernel::GoRouterTopK:
        case Kernel::LlMoeSort:
        case Kernel::LlMoeGather:
        case Kernel::LlExpertGate:
        case Kernel::LlExpertUp:
        case Kernel::LlExpertDown:
        case Kernel::LlExpertSiluMul:
        case Kernel::LlMoeCombine:
            return true;
        default:
            return false;
    }
}

/// This kind's launch shape and pipeline over `rows` prompt rows at once.
///
/// Deliberately the same expressions `mb_geometry` uses at the same row count,
/// because the batched prefill and the batched decode ARE the same computation
/// -- what differs is only that the prefill's token-major values are a pitch
/// apart. Two derivations of one shape is how a sort pads to one tile and a
/// matmul reads another.
bool routed_group_shape(const Dispatch& d, const DecodeGeometry& g, int rows,
                        const MultiBatchPsos& mb, Grid& grid, Threadgroup& tg, Pso& pso) {
    // A PREFILL still batches, and correctly: the same kernel over 8192 pairs
    // at 2048 tokens reproduces mlx-lm. Only the decode's is shut.
    const int sorted = moe_sorted_rows(g, rows, /*batched=*/true);
    switch (d.kind) {
        case Kernel::GoRouterTopK:
            shared_kernels::router_topk_dispatch(g.n_experts, grid, tg, rows);
            return true;
        case Kernel::LlMoeSort:
            // One threadgroup whatever the batch is; only its params change.
            shared_kernels::moe_route_sort_dispatch(g.n_experts, grid, tg);
            return true;
        case Kernel::LlMoeGather:
            shared_kernels::moe_route_rows_dispatch(g.hidden, sorted, grid, tg);
            return true;
        case Kernel::LlExpertSiluMul:
            elementwise_mb_dispatch(g.moe_intermediate, sorted, grid, tg);
            return true;
        case Kernel::LlMoeCombine:
            shared_kernels::expert_combine_dispatch(g.hidden, grid, tg, rows);
            return true;
        case Kernel::LlExpertGate:
        case Kernel::LlExpertUp:
        case Kernel::LlExpertDown: {
            const int N = d.kind == Kernel::LlExpertDown ? g.hidden : g.moe_intermediate;
            const int pairs = rows * g.experts_per_token;
            if (const int bn = qmm_bn(N, sorted, qmm_min_batch(true, qwen35_fp16_format(g)));
                bn > 0 && shared_kernels::moe_should_batch(pairs, g.n_experts)) {
                const int bm = shared_kernels::moe_tile_rows(pairs, g.n_experts);
                const Pso& gemm = mb.qmm_routed[shared_kernels::moe_bm_slot(bm)]
                                              [bn == 64 ? 2 : (bn == 32 ? 1 : 0)];
                if (gemm.valid()) {
                    qmm_t_dispatch(N, sorted, bn, bm, grid, tg);
                    pso = gemm;
                    return true;
                }
            }
            // One sorted row per pair. Still worth batching: the sort put the
            // rows that share an expert next to each other, so the slice one
            // reads is the slice the next three read.
            shared_kernels::routed_qmv_dispatch(N, 1, grid, tg, sorted);
            return true;
        }
        default:
            return false;
    }
}

bool barrier_after_mb(const std::vector<Dispatch>& dag, size_t i,
                      const std::vector<int>& run_ends) {
    if (i + 1 >= dag.size()) return true;
    return run_ends[i] == int(i);
}

Pso mb_pso(const Dispatch& d, const DecodeStepPsos& base, const MultiBatchPsos& mb,
           bool fp16_ok = true) {
    switch (d.kind) {
        case Kernel::EmbedUntied:
        case Kernel::EmbedGather: return mb.embed_mb;
        case Kernel::Rope:
        case Kernel::RopeK: return mb.rope_mb;
        case Kernel::GdnPrepSlotted: return mb.gdn_prep_slotted;
        case Kernel::GdnCoreSlotted: return mb.gdn_recurrent_slotted;
        case Kernel::KvAppendPaged: return mb.kv_append_paged;
        case Kernel::SdpaPaged: return mb.sdpa_paged;
        // The mixture's projections, asked BEFORE the shared GEMM below and
        // for the same reason `qmv_out_size` declines to answer for them: they
        // carry a `qmm_bn` like any batched projection, so the default arm
        // would hand them the DENSE GEMM -- which indexes one weight for the
        // whole dispatch and would run every expert's rows through expert 0's
        // slice. Fluent, and wrong.
        case Kernel::LlExpertGate:
        case Kernel::LlExpertUp:
        case Kernel::LlExpertDown: {
            if (d.qmm_bn > 0) {
                const int slot = d.qmm_bn == 64 ? 2 : (d.qmm_bn == 32 ? 1 : 0);
                const int bm = shared_kernels::moe_bm_slot(d.qmm_bm);
                if (moe_trace_enabled()) {
                    std::fprintf(stderr, "[moe] pso   bm=%d(slot %d) bn=%d(slot %d) routed=%d\n",
                                 d.qmm_bm, bm, d.qmm_bn, slot,
                                 int(mb.qmm_routed[bm][slot].valid()));
                }
                if (mb.qmm_routed[bm][slot].valid()) return mb.qmm_routed[bm][slot];
            }
            return base[d.kind];
        }
        default: {
            const int wide_ = qmm_bm_slot(d.qmm_bm);
            if (d.qmm_split > 1 && mb.qmm_t_splitk[wide_].valid())
                return mb.qmm_t_splitk[wide_];
            if (d.qmm_bn > 0) {
                const int slot = d.qmm_bn == 64 ? 2 : (d.qmm_bn == 32 ? 1 : 0);
                const int wide = wide_;
                // FP16 first, where the tables exist. Storage and every
                // neighbouring kernel stay BF16; what changes is that the MMA
                // is an instruction on this device rather than a sequence the
                // compiler writes to stand in for one.
                const Pso& fp16 = d.fuse_residual
                                      ? mb.qmm_t_residual_fp16_precast[wide][slot]
                                      : mb.qmm_t_fp16_precast[wide][slot];
                if (fp16_ok && fp16.valid() && mb.qmm_cast_bf16_f16.valid()) return fp16;
                const Pso& gemm = d.fuse_residual ? mb.qmm_t_residual[wide][slot]
                                                  : mb.qmm_t[wide][slot];
                if (gemm.valid()) return gemm;
            }
            return d.fuse_residual ? base.qmv_residual : base[d.kind];
        }
    }
}

/// The number of elements this dispatch's staging cast has to write, or 0
/// where it has none.
///
/// The one predicate: `mb_pso` is told FP16 is available exactly when this
/// answers nonzero, so the kernel that reads the staged copy cannot be
/// selected in a fire that did not write one.
int mb_fp16_cast_elems(const Dispatch& d, const MultiBatchPsos& mb,
                       const DecodeGeometry* g, int n_tokens) {
    if (g == nullptr || d.qmm_bn <= 0 || d.qmm_split > 1) return 0;
    if (!mb.qmm_cast_bf16_f16.valid()) return 0;
    if (is_routed_group(d.kind)) return 0;
    const int wide = qmm_bm_slot(d.qmm_bm);
    const int slot = d.qmm_bn == 64 ? 2 : (d.qmm_bn == 32 ? 1 : 0);
    const Pso& fp16 = d.fuse_residual ? mb.qmm_t_residual_fp16_precast[wide][slot]
                                      : mb.qmm_t_fp16_precast[wide][slot];
    if (!fp16.valid()) return 0;
    const int K = int(qmv_kn(d.kind, *g).K);
    if (K <= 0) return 0;
    // The padded row count, matching what the host bound at buffer 13 and what
    // the GEMM's row tiles actually read.
    return qmm_mb_rows(n_tokens, g->max_tokens, qmm_min_batch(g->is_moe(), qwen35_fp16_format(*g))) * K;
}

inline void bind_slot(RawMetalContext& ctx, int ord, uint8_t idx, const SlotHandle& slot) {
    ctx.arg_bind_ordinal(ord, idx, slot);
}

}  // namespace

std::size_t paged_attention_mask_pitch_bytes(
    const DecodeGeometry& geometry) {
    if (geometry.total_pages <= 0 ||
        geometry.kv_page_size <= 0 ||
        static_cast<std::size_t>(geometry.total_pages) >
            std::numeric_limits<std::size_t>::max() /
                static_cast<std::size_t>(
                    geometry.kv_page_size)) {
        return 0;
    }
    return static_cast<std::size_t>(geometry.total_pages) *
           static_cast<std::size_t>(geometry.kv_page_size);
}

bool paged_pool_size_supported(
    const DecodeGeometry& geometry,
    std::uint32_t pages) {
    return pages != 0 &&
           geometry.total_pages > 0 &&
           pages <=
               static_cast<std::uint32_t>(
                   geometry.total_pages);
}

std::vector<Dispatch> build_decode_dag_mb(const DecodeGeometry& g, int n_tokens,
                                          int ordinal_base, bool fuse_residual, bool gdn_prep) {
    if (n_tokens <= 0) throw std::runtime_error("multi-batch DAG requires at least one token");
    std::vector<Dispatch> dag = build_decode_dag(g, false, fuse_residual, gdn_prep);
    for (Dispatch& d : dag) {
        d.kind = mb_kind(d.kind);
        d.ordinal += ordinal_base;
        mb_geometry(d, g, n_tokens);
    }
    return dag;
}

std::vector<std::vector<Dispatch>> build_decode_prefill_dags(
    const DecodeGeometry& g, int n_tokens, bool fuse_residual, bool gdn_prep) {
    if (n_tokens <= 0) throw std::runtime_error("prefill DAG stream requires at least one token");
    std::vector<std::vector<Dispatch>> out;
    out.reserve(size_t(n_tokens));
    for (int t = 0; t < n_tokens; ++t) {
        auto dag = build_decode_dag_mb(
            g, 1, kPrefillOrdinalBase + t * kPrefillOrdinalStride, fuse_residual, gdn_prep);
        if (dag.size() >= size_t(kPrefillOrdinalStride)) {
            throw std::runtime_error("prefill DAG exceeds its argument-table ordinal stride");
        }
        out.push_back(std::move(dag));
    }
    return out;
}

void bind_decode_dag_mb(RawMetalContext& ctx, const BoundDecode& b,
                        const std::vector<Dispatch>& dag, const DecodeGeometry& g,
                        const std::vector<SlotHandle>& k_pages,
                        const std::vector<SlotHandle>& v_pages, bool gdn_prep,
                        const MbBindOffsets& offsets) {
    auto io = [&](IoSlot s) -> const SlotHandle& { return b.io[static_cast<int>(s)]; };
    if (k_pages.size() < size_t(g.n_layers) || v_pages.size() < size_t(g.n_layers))
        throw std::runtime_error("paged KV bindings do not cover all layers");
    for (const Dispatch& d : dag) {
        const int ord = d.ordinal;
        const int L = d.layer;
        for (const WeightBind& wb : weight_binds(d.kind, L, g, gdn_prep)) {
            const auto it = b.weights.find(wb.tensor);
            if (it == b.weights.end()) throw std::runtime_error("MB bind: unstaged weight " + wb.tensor);
            bind_slot(ctx, ord, wb.bind_index, it->second);
        }
        switch (d.kind) {
            case Kernel::EmbedUntied:
            case Kernel::EmbedGather:
                ctx.arg_bind_ordinal(ord, uint8_t(bind::Embed::TokenId), io(IoSlot::TokenId),
                                     offsets.token_row * sizeof(uint32_t));
                break;
            case Kernel::GdnPrepSlotted: {
                const auto& s = b.gdn[L];
                bind_slot(ctx, ord, uint8_t(bind::GdnPrep::ConvState), s.conv_state);
                bind_slot(ctx, ord, uint8_t(bind::GdnPrep::ConvStateOut), s.conv_state_out);
                bind_slot(ctx, ord, uint8_t(bind::GdnPrep::ConvB), s.conv_bias_zero);
                ctx.arg_bind_ordinal(ord, uint8_t(bind::GdnPrep::SlotOfToken),
                                     io(IoSlot::SlotOfToken),
                                     offsets.token_row * sizeof(uint32_t));
                break;
            }
            case Kernel::GdnCoreSlotted: {
                const auto& s = b.gdn[L];
                bind_slot(ctx, ord, uint8_t(bind::GdnCoreRecurrent::ConvState), s.conv_state);
                bind_slot(ctx, ord, uint8_t(bind::GdnCoreRecurrent::RecurrentState), s.recurrent_state);
                bind_slot(ctx, ord, uint8_t(bind::GdnCoreRecurrent::ConvStateOut), s.conv_state_out);
                bind_slot(ctx, ord, uint8_t(bind::GdnCoreRecurrent::ConvB), s.conv_bias_zero);
                ctx.arg_bind_ordinal(ord, uint8_t(bind::GdnCoreRecurrent::SlotOfToken),
                                     io(IoSlot::SlotOfToken),
                                     offsets.token_row * sizeof(uint32_t));
                break;
            }
            case Kernel::KvAppendPaged:
                bind_slot(ctx, ord, uint8_t(bind::KvAppendPaged::KPages), k_pages[L]);
                bind_slot(ctx, ord, uint8_t(bind::KvAppendPaged::VPages), v_pages[L]);
                ctx.arg_bind_ordinal(ord, uint8_t(bind::KvAppendPaged::PositionIds),
                                     io(IoSlot::Position),
                                     offsets.token_row * sizeof(uint32_t));
                bind_slot(ctx, ord, uint8_t(bind::KvAppendPaged::KvPageIndices),
                          io(IoSlot::KvPageIndices));
                bind_slot(ctx, ord, uint8_t(bind::KvAppendPaged::KvPageIndptr),
                          io(IoSlot::KvPageIndptr));
                ctx.arg_bind_ordinal(ord, uint8_t(bind::KvAppendPaged::ReqOfToken),
                                     io(IoSlot::ReqOfToken), offsets.token_row * sizeof(uint32_t));
                ctx.arg_bind_ordinal(ord, uint8_t(bind::KvAppendPaged::WPage), io(IoSlot::WPage),
                                     offsets.token_row * sizeof(uint32_t));
                ctx.arg_bind_ordinal(ord, uint8_t(bind::KvAppendPaged::WOff), io(IoSlot::WOff),
                                     offsets.token_row * sizeof(uint32_t));
                break;
            case Kernel::SdpaPaged:
                bind_slot(ctx, ord, uint8_t(bind::SdpaPaged::KPages), k_pages[L]);
                bind_slot(ctx, ord, uint8_t(bind::SdpaPaged::VPages), v_pages[L]);
                ctx.arg_bind_ordinal(ord, uint8_t(bind::SdpaPaged::PositionIds),
                                     io(IoSlot::Position),
                                     offsets.token_row * sizeof(uint32_t));
                ctx.arg_bind_ordinal(ord, uint8_t(bind::SdpaPaged::ReqOfToken),
                                     io(IoSlot::ReqOfToken),
                                     offsets.token_row * sizeof(uint32_t));
                bind_slot(ctx, ord, uint8_t(bind::SdpaPaged::KvPageIndices),
                          io(IoSlot::KvPageIndices));
                bind_slot(ctx, ord, uint8_t(bind::SdpaPaged::KvPageIndptr),
                          io(IoSlot::KvPageIndptr));
                ctx.arg_bind_ordinal(
                    ord,
                    uint8_t(bind::SdpaPaged::AttnMask),
                    io(IoSlot::AttnMask),
                    offsets.token_row *
                        paged_attention_mask_pitch_bytes(g));
                bind_slot(
                    ctx, ord,
                    uint8_t(bind::SdpaPaged::AttnMaskStride),
                    io(IoSlot::AttnMaskStride));
                ctx.arg_bind_ordinal(
                    ord,
                    uint8_t(bind::SdpaPaged::AttnMaskEnabled),
                    io(IoSlot::AttnMaskEnabled),
                    offsets.token_row);
                break;
            case Kernel::Rope:
            case Kernel::RopeK:
                ctx.arg_bind_ordinal(ord, uint8_t(bind::Rope::Position), io(IoSlot::Position),
                                     offsets.token_row * sizeof(uint32_t));
                break;
            case Kernel::LmHeadUntied:
            case Kernel::QmvLmHead:
                ctx.arg_bind_ordinal(ord, uint8_t(bind::Qmv::Out), io(IoSlot::Logits),
                                     offsets.logits_bytes);
                break;
            default:
                break;
        }
    }
}

/// Point the GDN pair's conv ping-pong at one of its two halves.
///
/// `even` selects the half holding the history this fire will READ: true binds
/// `conv_state` in and `conv_state_out` out, false the reverse.
///
/// The two halves may not be the same buffer, which is the whole reason this
/// exists. `gdn_prep` runs one threadgroup per VALUE head, and `rep = Hv/Hk`
/// value heads share a key head: that head's conv window is READ by all `rep`
/// of them and WRITTEN by the first. Alias the output over the input and the
/// writer shifts the window while its siblings are still reading it -- a race
/// whose two outcomes are both finite numbers, so it reports as a fleet member
/// that quietly disagrees with its identical neighbours rather than as a
/// fault. `rep` is 1 on Qwen3.6-35B-A3B and 3 on Qwen3.6-27B, which is why
/// only the 27B showed it.
void bind_gdn_conv_parity(RawMetalContext& ctx, const BoundDecode& b,
                          const std::vector<Dispatch>& dag, bool even) {
    for (const Dispatch& d : dag) {
        if (d.kind != Kernel::GdnPrepSlotted && d.kind != Kernel::GdnCoreSlotted) continue;
        const auto& s = b.gdn[size_t(d.layer)];
        if (!s.conv_state.valid() || !s.conv_state_out.valid()) continue;
        const SlotHandle& in = even ? s.conv_state : s.conv_state_out;
        const SlotHandle& out = even ? s.conv_state_out : s.conv_state;
        if (d.kind == Kernel::GdnPrepSlotted) {
            ctx.arg_bind_ordinal(d.ordinal, uint8_t(bind::GdnPrep::ConvState), in);
            ctx.arg_bind_ordinal(d.ordinal, uint8_t(bind::GdnPrep::ConvStateOut), out);
        } else {
            ctx.arg_bind_ordinal(d.ordinal, uint8_t(bind::GdnCoreRecurrent::ConvState), in);
            ctx.arg_bind_ordinal(d.ordinal, uint8_t(bind::GdnCoreRecurrent::ConvStateOut), out);
        }
    }
}

namespace {

// The only state a prompt token hands to the next one: the GDN recurrent and
// convolution slots, which every token reads and writes in place (and which
// ping-pong, so token t's output buffer is token t+1's input). Paged KV is not
// on this list -- each token appends at its own position, and the group barrier
// below already orders the whole append group ahead of the whole attention
// group.
// The tail that exists only to produce a row's logits.
bool produces_logits(Kernel kind) {
    return kind == Kernel::QmvLmHead || kind == Kernel::LmHeadUntied ||
           kind == Kernel::Argmax;
}

bool carries_cross_token_state(Kernel kind) {
    switch (kind) {
        case Kernel::GdnPrep:
        case Kernel::GdnPrepSlotted:
        case Kernel::GdnCore:
        case Kernel::GdnCoreSlotted:
            return true;
        default:
            return false;
    }
}

}  // namespace

// A prefill walked token by token streams the whole checkpoint once per prompt
// token -- 34 passes over 405MB for a 34-token prompt, which measured 118ms of
// GPU against a 1.0ms bandwidth floor. Every token's DAG is the same DAG with
// different argument tables, so walking it dispatch by dispatch instead reads
// each weight tensor once and hands it to all N tokens back to back, small
// enough to stay in cache, and gives the GPU N-way work where it had one narrow
// dispatch.
//
// The reorder is legal because a barrier closes every dispatch group: token t's
// dispatch i is ordered ahead of any token's dispatch i+1, which is every
// dependency inside a token and every dependency through paged KV. The one
// exception is the GDN state above, which is serialized token by token.
int prefill_routed_group(const DecodeGeometry& g, int n_tokens, int max_rows) {
    if (!g.is_moe() || n_tokens <= 1 || max_rows <= 0) return 0;
    const std::size_t cap = scratch_slot_elems(g, max_rows);
    const std::size_t need =
        std::size_t(moe_sorted_rows(g, n_tokens)) * std::size_t(g.hidden);
    return need <= cap ? n_tokens : 0;
}

void encode_prefill_dags_mb(StepEncoder& se,
                            const std::vector<std::vector<Dispatch>>& dags,
                            int n_tokens,
                            const DecodeStepPsos& base_psos,
                            const MultiBatchPsos& mb_psos,
                            bool force_barriers,
                            const std::vector<std::uint8_t>& row_needs_logits,
                            const DecodeGeometry* geometry,
                            int max_rows,
                            const std::vector<GdnScanSegment>& gdn_scans,
                            const MultiBatchPsos* mb_alt_psos,
                            std::size_t begin,
                            std::size_t end) {
    const size_t n = n_tokens > 0 ? size_t(n_tokens) : 0;
    if (n == 0 || dags.size() < n) return;
    const size_t length = dags[0].size();
    // Token-major is the contract this reorders; if the DAGs ever stop being the
    // same shape, that assumption is gone and so is the reorder.
    for (size_t t = 1; t < n; ++t) {
        if (dags[t].size() != length) {
            // The sub-range goes through unchanged. It indexes a dispatch
            // within one token's DAG either way, and if the shapes disagree a
            // cut derived from `dags[0]` means nothing anywhere -- which is a
            // paging refusal made at plan time, not a case to reinterpret here.
            for (size_t k = 0; k < n; ++k)
                encode_decode_step_mb(se, dags[k], base_psos, mb_psos, force_barriers,
                                      nullptr, 0, begin, end);
            return;
        }
    }
    // Every group ends in a barrier. Letting the adjacent-group hazard rule
    // relax some of them was measured and moved nothing (93.6ms against 93.0ms):
    // N-way concurrency inside a group already saturates the machine.
    // lm_head reads the whole output embedding -- 127MB for this checkpoint,
    // far past any cache -- so running it for a prompt row nobody samples is the
    // single most expensive wasted dispatch in a prefill.
    const bool skip_unsampled_logits = row_needs_logits.size() >= n;
    // A projection does not depend on any other token's row, so the whole prompt
    // can go through the GEMM in one dispatch instead of one GEMV per token --
    // which is what turns N reads of the checkpoint into one. lm_head is left
    // out: it writes the logits buffer, whose rows are the fire's readout rows
    // and are not padded, and it already runs only for sampled rows.
    const int strided_rows =
        geometry != nullptr ? qmm_strided_rows(int(n), max_rows) : 0;
    // How many prompt rows the mixture runs over at once, or 0 for one at a
    // time. Bounded by the pool slot, which was sized for a per-token layout:
    // a batched sort pads every touched expert's run to a whole tile, so its
    // stack can be several times the rows that are real. A prompt whose stack
    // would not fit keeps the per-token walk rather than overrunning the
    // colour into the next value.
    const int routed_group =
        geometry != nullptr ? prefill_routed_group(*geometry, int(n), max_rows) : 0;
    for (size_t i = begin; i < std::min(end, length); ++i) {
        const Dispatch& d0 = dags[0][i];
        // Priced by ablation rather than by the trace; see `kernel_ablated`.
        // Skipping the WHOLE kind here -- not one token's copy of it -- is what
        // makes the wall-clock difference the kind's serial cost.
        if (kernel_ablated(d0.kind)) continue;
        // ── the alternate affine format ──
        //
        // A checkpoint may spare individual tensors from the model-wide format
        // -- mlx-lm's predicate names them -- and Qwen3.6-35B-A3B leaves the MoE
        // router and the shared expert's gate at 8 bits inside a 4-bit build.
        // The strided arm below has to decline those two, because its pipelines
        // read 4-bit bytes and running them over 8-bit ones is not a crash, it
        // is a wrong answer: gemma4's router did exactly that once and produced
        // logits at cosine 0.10 to mlx-lm's with every tensor feeding them at
        // 0.9999.
        //
        // Declining is not the same as having nothing to run, though, and until
        // now it was: both kinds fell all the way through to the per-token walk,
        // so a prefill dispatched them once per row -- forty layers times two
        // kinds times every token. In a 1024-row fire that was the largest line
        // in the profile at 30-37% of GPU time, against 1.75% for the real
        // GEMMs. The wide matvec is the batched shape they were missing: it is
        // parametric in the row count, guards its last partial group, and reads
        // each decoded weight chunk once for four token vectors. Asked of the
        // ALT table, so the bytes and the pipeline agree.
        if (strided_rows > 0 && geometry != nullptr && mb_alt_psos != nullptr &&
            qwen35_uses_alt_quant(d0.kind, *geometry) && !d0.fuse_residual &&
            mb_alt_psos->qmv_wide_strided.valid()) {
            const int out = qmv_out_size(d0.kind, *geometry);
            if (out != 0) {
                constexpr int vecs = 4;
                constexpr int lanes = 8;
                se.set_pso(mb_alt_psos->qmv_wide_strided);
                se.set_argtable(d0.kind, d0.ordinal);
                se.dispatch(
                    Grid{32u * std::uint32_t((int(n) + vecs - 1) / vecs),
                         2u * std::uint32_t(
                             (out + (64 / lanes) - 1) / (64 / lanes)), 1},
                    Threadgroup{32, 2, 1});
                se.barrier();
                continue;
            }
        }
        if (strided_rows > 0 && d0.kind != Kernel::QmvLmHead &&
            d0.kind != Kernel::LmHeadUntied &&
            !qwen35_uses_alt_quant(d0.kind, *geometry)) {
            const int out = qmv_out_size(d0.kind, *geometry);
            // The GEMM below tiles the output 32 columns at a time, so it can
            // only take a projection whose width is a whole number of tiles.
            // Everything else needs a batched primitive of its own, and the
            // wide matvec IS that primitive: it is parametric in N, guards its
            // last partial row group, and reads each decoded weight chunk once
            // for four token vectors instead of once per token.
            //
            // This used to ask `out == 16`, which is the value Qwen3-Next's
            // `linear_num_value_heads` has, and every projection of any other
            // width that the GEMM also declined fell through to ONE MATVEC PER
            // TOKEN. Qwen3.6-27B has 48 value heads, so its two per-head decay
            // projections did exactly that: 2 kinds x 48 GDN layers x N tokens,
            // 3072 dispatches in a 32-token prefill, each moving a 123 KB
            // weight -- and a 32-token prefill encoded 8435 dispatches where it
            // now encodes 5459. Prefill tok/s: 67.4 -> 84.2 at 32 tokens, 81.1
            // -> 90.0 at 64, 94.2 -> 94.0 at 128. It is a per-token cost, so it
            // is the SHORT prompts it was taxing, and by 128 tokens the GEMMs
            // have grown past it. The width the GEMM cannot take is the width
            // that needs this path most, so the question to ask is the GEMM's,
            // negated.
            if (out != 0 && out % 32 != 0 && !d0.fuse_residual &&
                mb_psos.qmv_wide_strided.valid()) {
                constexpr int vecs = 4;
                constexpr int lanes = 8;
                se.set_pso(mb_psos.qmv_wide_strided);
                se.set_argtable(d0.kind, d0.ordinal);
                se.dispatch(
                    Grid{32u * std::uint32_t((int(n) + vecs - 1) / vecs),
                         2u * std::uint32_t(
                             (out + (64 / lanes) - 1) / (64 / lanes)), 1},
                    Threadgroup{32, 2, 1});
                se.barrier();
                continue;
            }
            // A GDN projection narrow enough to fit one tile stays a matvec
            // even where the GEMM could take it. A BN16 strided GEMM replaced
            // 768 dispatches with six and end-to-end prefill fell from 1408 to
            // 1396 tok/s: at that width a matrix tile's setup costs more than
            // the four-vector reuse above saves.
            if (out != 0 && out % 32 == 0) {
                static_assert(kQmmBMCount == 3,
                              "the strided PSO arrays in DecodeStepPsos are sized 3 "
                              "and indexed by qmm_bm_slot; a fourth rung has to widen "
                              "them in the same commit");
                const bool fp16 = mb_psos.qmm_t_strided_cast.valid() &&
                                  mb_psos.qmm_t_strided_fp16_precast[0].valid();
                const Pso* rungs = fp16
                    ? (d0.fuse_residual ? mb_psos.qmm_t_strided_fp16_precast_residual
                                        : mb_psos.qmm_t_strided_fp16_precast)
                    : (d0.fuse_residual ? mb_psos.qmm_t_strided_residual
                                        : mb_psos.qmm_t_strided);
                // The rung the row count earns, narrowed to one that actually
                // loaded. Narrowing has to move the GRID with it -- the kernel
                // takes no M and reaches its row count through grid.y alone, so
                // a BM=64 pipeline dispatched on a BM=32 grid writes sixty-four
                // rows into a thirty-two-row tile. A 32-row prefill did exactly
                // that and fell from 84.2 tok/s to 51.7 while still answering
                // correctly, because the rows it trampled were the padding.
                int bm = qmm_strided_bm(strided_rows);
                while (bm > kQmmBM && !rungs[qmm_bm_slot(bm)].valid()) bm /= 2;
                const Pso& gemm = rungs[qmm_bm_slot(bm)];
                if (gemm.valid()) {
                    Grid grid;
                    Threadgroup tg;
                    qmm_t_strided_dispatch(out, strided_rows, bm, grid, tg);
                    if (fp16) {
                        se.set_pso(mb_psos.qmm_t_strided_cast);
                        se.set_argtable(d0.kind, d0.ordinal);
                        se.dispatch(
                            Grid{std::uint32_t(qmv_kn(d0.kind, *geometry).K),
                                 std::uint32_t(strided_rows), 1},
                            Threadgroup{256, 1, 1});
                        se.barrier();
                    }
                    se.set_pso(gemm);
                    se.set_argtable(d0.kind, d0.ordinal);
                    se.dispatch(grid, tg);
                    se.barrier();
                    continue;
                }
            }
        }
        // ── the mixture, over the whole prompt at once ──
        //
        // The routing group is the one part of a prefill that CANNOT be batched
        // by taking a kernel's strided variant, because what it batches over is
        // not rows: the sort groups (row, slot) pairs by EXPERT, and a group of
        // one row can only ever be one token's eight pairs. Run per token, every
        // pair read its expert's whole weight slice -- 1.5 MB for a (2048, 512)
        // triple -- and a 128-token prefill of Qwen3.6-35B-A3B read 1.6 GB a
        // layer where 256 experts hold 402 MB. `affine_qmv_routed` was 30% of
        // the fire's GPU time, and it was mostly the same bytes.
        //
        // Batched, the group runs on dag[0]'s argument tables: the sorted stack
        // is packed from the base either way, and the three values that are
        // TOKEN-major -- the router's logits in, the mixture's output out, and
        // the hidden rows the gather reads -- carry the prefill's row pitch as a
        // parameter rather than being assumed packed.
        if (routed_group > 0 && is_routed_group(d0.kind)) {
            Grid grid = d0.grid;
            Threadgroup tg = d0.tg;
            Pso pso = mb_pso(d0, base_psos, mb_psos);
            if (!routed_group_shape(d0, *geometry, routed_group, mb_psos, grid, tg, pso)) {
                // A kind whose batched form this does not know: fall through to
                // the per-token walk rather than guess a shape for it.
            } else {
                se.set_pso(pso);
                se.set_argtable(d0.kind, d0.ordinal);
                se.dispatch(grid, tg);
                se.barrier();
                continue;
            }
        }
        // ── the attention, over the whole prompt at once ──
        //
        // Priced by ablation (`PIE_METAL_ABLATE=sdpa_paged`) at 8.2% of a
        // 128-token Qwen3.6-35B-A3B prefill and 16.5% of a 2048-token one --
        // and at 2048 it is the WHOLE remaining gap to mlx-lm, whose 847.5
        // tok/s this fire meets at 840.7 with the attention taken out.
        //
        // Two wins, and the second is the larger. One dispatch replaces N, and
        // the tiled kernel stages a key tile ONCE PER 32 ROWS where the per-row
        // kernel reads it once per row. `sdpa_should_tile`'s own note is that a
        // prefill "is all one run and wins outright" -- every row shares a key
        // span -- which is the opposite of a fleet of decodes, where 32 rows
        // are 32 runs and tiling loses badly. A prefill is exactly one request,
        // so this asks the predicate rather than assuming.
        if (strided_rows > 0 && geometry != nullptr && d0.kind == Kernel::SdpaPaged &&
            mb_psos.sdpa_paged_tiled_strided.valid() &&
            sdpa_should_tile(int(n), /*requests=*/1)) {
            Grid grid;
            Threadgroup tg;
            // `geometry->n_q_heads`, not `d0.grid.x`: the per-token dispatch
            // already folded the 1024-thread threadgroup into x, so grid.x is
            // `n_q_heads * 1024`. The kernel reads its head count back out of
            // `threadgroups_per_grid.x`, so handing this the wrong number is a
            // wrong answer rather than a refusal.
            sdpa_paged_tiled_dispatch(geometry->n_q_heads, int(n), grid, tg);
            se.set_pso(mb_psos.sdpa_paged_tiled_strided);
            se.set_argtable(d0.kind, d0.ordinal);
            se.dispatch(grid, tg);
            se.barrier();
            continue;
        }
        // Row-independent kernels: the prefill's scratch rows are a uniform pitch
        // apart, so one dispatch over the whole prompt replaces one per token.
        // The row-blocked variants take that pitch explicitly and are otherwise
        // byte-for-byte the same arithmetic.
        {
            Pso strided{};
            Grid grid = d0.grid;
            switch (d0.kind) {
                case Kernel::Rms:
                case Kernel::FfnRms:
                case Kernel::FinalRms:
                    // One threadgroup per row only; the per-head norms (q/k) stack
                    // several rows inside one dispatch and do not have a uniform pitch.
                    if (d0.grid.x == d0.tg.x && d0.grid.y == 1 && d0.grid.z == 1) {
                        strided = mb_psos.rms_strided;
                        grid.x = d0.grid.x * uint32_t(n);
                    }
                    break;
                case Kernel::QNorm:
                case Kernel::KNorm:
                    // `rms_mb_dispatch` lays the heads along x as
                    // (axis_threads * n_rows); the strided form wants them as
                    // their own dimension so the threadgroup's position carries
                    // the (head, token) pair. tg.x IS the axis thread count, so
                    // the head count divides out.
                    if (d0.grid.y == 1 && d0.grid.z == 1 && d0.tg.x > 0 &&
                        d0.grid.x % d0.tg.x == 0) {
                        strided = mb_psos.rms_strided_head;
                        grid = Grid{d0.tg.x, d0.grid.x / d0.tg.x, uint32_t(n)};
                    }
                    break;
                case Kernel::Rope:
                case Kernel::RopeK:
                    // Already (half, n_heads, 1) per token, and the kernel reads
                    // `position[m]` from the same array dag[0] points at, so the
                    // token row is the only thing the launch has to add.
                    if (d0.grid.z == 1) {
                        strided = mb_psos.rope_strided;
                        grid.z = uint32_t(n);
                    }
                    break;
                case Kernel::SiluMul:
                    if (d0.grid.y == 1 && d0.grid.z == 1) {
                        strided = mb_psos.silu_mul_strided;
                        grid.y = uint32_t(n);
                    }
                    break;
                // `elementwise_mb_dispatch` folds the row count onto x, so at
                // the per-token DAG's n==1 the grid is exactly one row wide and
                // the strided form only has to add the row axis.
                case Kernel::Residual:
                case Kernel::LayerOut:
                    if (d0.grid.y == 1 && d0.grid.z == 1) {
                        strided = mb_psos.residual_add_strided;
                        grid.y = uint32_t(n);
                    }
                    break;
                // Already (width, 1, 1) from `expert_combine_dispatch`, which
                // reads its row from gid.y -- so this one was ALREADY row-aware
                // and only lacked the pitch.
                case Kernel::LlSharedCombine:
                    if (d0.grid.y == 1 && d0.grid.z == 1) {
                        strided = mb_psos.shared_expert_combine_strided;
                        grid.y = uint32_t(n);
                    }
                    break;
                case Kernel::GatedRms:
                    if (d0.grid.z == 1) {
                        strided = mb_psos.gated_rms_strided;
                        grid.z = uint32_t(n);
                    }
                    break;
                case Kernel::QSplit:
                    // z was always the token row; the prompt just never went
                    // there, so a 1024-row prefill deinterleaved one row at a
                    // time -- 2820 dispatches in the fire this was found in.
                    if (d0.grid.z == 1) {
                        strided = mb_pso(d0, base_psos, mb_psos);
                        grid.z = uint32_t(n);
                    }
                    break;
                case Kernel::AttnGate:
                    if (d0.grid.y == 1 && d0.grid.z == 1) {
                        strided = mb_pso(d0, base_psos, mb_psos);
                        grid.y = uint32_t(n);
                    }
                    break;
                case Kernel::KvAppendPaged:
                    // z is the token, and w_page/w_off are already arrays row
                    // zero's table points at the front of. Only the source
                    // pitch stood between this and the whole prompt.
                    if (d0.grid.z == 1) {
                        strided = mb_pso(d0, base_psos, mb_psos);
                        grid.z = uint32_t(n);
                    }
                    break;
                default:
                    break;
            }
            if (strided.valid()) {
                se.set_pso(strided);
                se.set_argtable(d0.kind, d0.ordinal);
                se.dispatch(grid, d0.tg);
                se.barrier();
                continue;
            }
        }
        const bool serialize = carries_cross_token_state(dags[0][i].kind);
        // The GDN pair is the prefill's only strict chain -- a barrier per token,
        // 34 of them per layer.  When the caller says the prompt is one request's
        // and hands us an even scan length (so the ping-pong lands where the
        // trailing per-token dispatch expects it), the whole prompt goes through
        // in one dispatch each: prep is token-parallel once its conv window comes
        // off `mixed`, and the recurrent scan runs in registers.
        // Each request's recurrence is independent, so one scan per request
        // replaces that request's whole per-token chain; rows a scan does not
        // cover (an even tail) fall through to the per-token path below.
        std::vector<bool> scanned(n, false);
        if (serialize && !gdn_scans.empty()) {
            const Pso& scan = dags[0][i].kind == Kernel::GdnPrepSlotted
                                  ? mb_psos.gdn_prep_prefill
                                  : mb_psos.gdn_core_prefill;
            if (scan.valid()) {
                se.set_pso(scan);
                for (const GdnScanSegment& seg : gdn_scans) {
                    if (seg.rows < 2 || size_t(seg.start) >= n) continue;
                    const Dispatch& d = dags[size_t(seg.start)][i];
                    Grid grid = d.grid;
                    // prep is (dk, 1, rows*heads); the recurrent kernel keeps
                    // its (dk, dv, heads) grid and walks the rows internally.
                    if (d.kind == Kernel::GdnPrepSlotted) {
                        grid.z = d.grid.z * uint32_t(seg.rows);
                    } else {
                        // The scan puts 32/`gdn_scan_lanes` dv rows in one
                        // simdgroup so each token's two reductions run that
                        // many lanes wide instead of 32. Its x extent therefore
                        // covers that many rows, not one; a Dv the rows do not
                        // divide rounds up and the kernel masks the spares.
                        const uint32_t rows_per_simd =
                            uint32_t(32 / gdn_scan_lanes()) * uint32_t(gdn_scan_rows());
                        grid.y = (grid.y + rows_per_simd - 1) / rows_per_simd;
                    }
                    se.set_argtable(d.kind, d.ordinal);
                    se.dispatch(grid, d.tg);
                    for (int r = 0; r < seg.rows; ++r)
                        scanned[size_t(seg.start + r)] = true;
                }
                se.barrier();
            }
        }
        const bool logits_tail = skip_unsampled_logits && produces_logits(dags[0][i].kind);
        bool prev_emitted = false;
        for (size_t t = 0; t < n; ++t) {
            if (serialize && scanned[t]) continue;
            if (logits_tail && row_needs_logits[t] == 0) continue;
            const Dispatch& d = dags[t][i];
            if (d.kind != dags[0][i].kind) {
                for (size_t k = 0; k < n; ++k)
                    encode_decode_step_mb(se, dags[k], base_psos, mb_psos, force_barriers);
                return;
            }
            se.set_pso(mb_pso(d, base_psos, mb_psos));
            se.set_argtable(d.kind, d.ordinal);
            se.dispatch(d.grid, d.tg);
            // Barrier between consecutive per-token GDN dispatches only; a
            // scanned row in between has already been ordered by the scan's
            // own barrier above.
            if (serialize && prev_emitted) se.barrier();
            prev_emitted = true;
        }
        se.barrier();
    }
}

void encode_decode_step_mb(StepEncoder& se, const std::vector<Dispatch>& dag,
                           const DecodeStepPsos& base_psos, const MultiBatchPsos& mb_psos,
                           bool force_barriers, const DecodeGeometry* g, int n_tokens,
                           std::size_t begin, std::size_t end) {
    const std::vector<int> run_ends = concurrent_run_ends(dag);
    const std::size_t stop = std::min(end, dag.size());
    for (size_t i = begin; i < stop; ++i) {
        const Dispatch& d = dag[i];
        // Priced by ablation, same as the prefill walk; see `kernel_ablated`.
        if (kernel_ablated(d.kind)) continue;
        const int cast_elems = mb_fp16_cast_elems(d, mb_psos, g, n_tokens);
        // The staging cast, ahead of the GEMM that reads it. It rides the
        // GEMM's own argument table -- the source is already bound at buffer 3
        // -- so it needs no DAG entry, and the barrier after it is what makes
        // the cast visible to the tile loader.
        //
        // Per projection rather than grouped: a batched decode produces each
        // value in the dispatch immediately before its consumer, so there is no
        // window in which several are simultaneously final and nothing to
        // group. It touches n*K elements against a GEMM that reads them N/BN
        // times over.
        if (cast_elems > 0) {
            se.set_pso(mb_psos.qmm_cast_bf16_f16);
            se.set_argtable(d.kind, d.ordinal);
            se.dispatch(Grid{std::uint32_t(cast_elems), 1, 1}, Threadgroup{256, 1, 1});
            se.barrier();
        }
        se.set_pso(mb_pso(d, base_psos, mb_psos, cast_elems > 0));
        se.set_argtable(d.kind, d.ordinal);
        se.dispatch(d.grid, d.tg);
        // Arm B of the interleaved A/B is a CONTROL by default: identical to
        // arm A, so a nonzero A-B difference means the harness is biased and
        // nothing else. Point it at whatever is being evaluated by adding a
        // term here.
        //
        // It used to default to "barrier after EVERY dispatch", which is a fine
        // question but a terrible default: any other question asked of this
        // harness came back dominated by it, and one such answer (a 12% win
        // that was really 0) reached a commit before being caught.
        // `PIE_METAL_AB_BARRIERS=1` asks the old question explicitly.
        if (force_barriers || barrier_after_mb(dag, i, run_ends) ||
            (ab_enabled() && ab_arm() && ab_all_barriers()))
            se.barrier();
    }
}

}  // namespace pie::metal

//! GPT-OSS's per-token dispatch DAG, in the shared [`Kernel`] vocabulary.
//!
//! A pure function, exactly like the qwen builder: it emits the ordered
//! dispatch list for one token and touches no Metal, so the step's shape —
//! dispatch count, the sliding/full split, what an MoE layer costs — is
//! checked with no GPU and no checkpoint.
//!
//! The C++ kept this family's kinds in their own namespace until the
//! family could be bound; the shared enum has since absorbed them (the
//! `Go*` kinds carry the family's weight names in `weight_binds`), so this
//! port builds directly in the shared vocabulary and reuses the whole M=1
//! machinery. The mixture's movers are the SHARED sort and gather — the
//! same kernels every routed family runs — and the residuals are the
//! shared adds.
//!
//! What is deliberately absent at M=1: the C++ `RowGather`, which compacts
//! the rows a fire will SAMPLE so the tail runs on that prefix. At one
//! token the compaction is the identity, and emitting an identity mover
//! would cost a dispatch and a scratch colour for nothing; it returns with
//! the prefill, where it is most of the fire's savings.

use crate::tuning::Tuning;

use super::abi::Kernel;
use super::dispatch::{
    Dispatch, Launch, kv_append, qmv, residual, rms, rope, route_rows, route_sort,
    router_lane_width, router_topk, sdpa,
};
use super::dispatch_mb::{qmm_bm, qmm_bn_unsplit};
use super::gptoss::GptOssGeometry;
use super::gptoss_consts::gptoss_qmv_kn;
use super::psos_mb::QMM_BMS;
use super::sizing::sorted_rows;

fn elementwise(width: u32) -> Launch {
    Launch {
        grid: [width, 1, 1],
        tg: [256, 1, 1],
    }
}

/// Emit the ordered per-token DAG for `g`.
///
/// Within a stage the order clusters independent dispatches (q/k/v
/// together, then the ropes) — hazard-neutral, and what lets a concurrency
/// group form.
#[must_use]
pub fn build_gptoss_dag(g: &GptOssGeometry, with_argmax: bool) -> Vec<Dispatch> {
    let mut dag: Vec<Dispatch> = Vec::with_capacity(g.n_layers as usize * 21 + 4);
    let emit = |dag: &mut Vec<Dispatch>, kind: Kernel, layer: Option<u32>, launch: Launch| {
        let ordinal = u32::try_from(dag.len()).expect("a DAG is hundreds of dispatches");
        dag.push(Dispatch {
            kind,
            ordinal,
            layer,
            launch,
            fuse_residual: false,
            qmm_bn: 0,
            qmm_split: 1,
            qmm_bm: 16,
        });
    };
    // At decode the sort is a pure grouping: tile 1, one row per
    // (token, slot) pair.
    let sorted = u32::try_from(sorted_rows(g.experts_per_token, g.n_experts, 1))
        .expect("a decode sort is small");

    emit(
        &mut dag,
        Kernel::EmbedUntied,
        None,
        Launch {
            grid: [g.hidden, 1, 1],
            tg: [256, 1, 1],
        },
    );

    for layer in 0..g.n_layers {
        let at = Some(layer);
        emit(&mut dag, Kernel::Rms, at, rms(g.hidden, 1));
        emit(&mut dag, Kernel::GoQmvQ, at, qmv(g.q_dim()));
        emit(&mut dag, Kernel::GoQmvK, at, qmv(g.kv_dim()));
        emit(&mut dag, Kernel::GoQmvV, at, qmv(g.kv_dim()));
        // Full rotary: every head dim rotates (no partial factor here).
        emit(&mut dag, Kernel::Rope, at, rope(g.head_dim, g.n_q_heads));
        emit(&mut dag, Kernel::RopeK, at, rope(g.head_dim, g.n_kv_heads));
        emit(
            &mut dag,
            Kernel::KvAppend,
            at,
            kv_append(g.head_dim, g.n_kv_heads),
        );
        // The sink attention: same launch as the plain decode SDPA — the
        // sink is one more denominator term, not one more thread.
        emit(&mut dag, Kernel::GoSdpaSink, at, sdpa(g.n_q_heads));
        emit(&mut dag, Kernel::GoQmvO, at, qmv(g.hidden));
        emit(&mut dag, Kernel::Residual, at, residual(g.hidden));

        emit(&mut dag, Kernel::FfnRms, at, rms(g.hidden, 1));
        emit(&mut dag, Kernel::GoRouter, at, qmv(g.n_experts));
        emit(&mut dag, Kernel::GoRouterTopK, at, router_topk(g.n_experts));
        emit(&mut dag, Kernel::LlMoeSort, at, route_sort(g.n_experts));
        emit(
            &mut dag,
            Kernel::LlMoeGather,
            at,
            route_rows(g.hidden, sorted),
        );
        emit(&mut dag, Kernel::GoExpertGate, at, {
            let mut launch = qmv(g.intermediate);
            launch.grid[0] *= sorted;
            launch
        });
        emit(&mut dag, Kernel::GoExpertUp, at, {
            let mut launch = qmv(g.intermediate);
            launch.grid[0] *= sorted;
            launch
        });
        // The clamped SwiGLU over the sorted stack: gate*sigmoid(alpha*gate)
        // * (up + 1), both operands clamped — the +1 and the clamp are why
        // this cannot reuse silu_mul.
        emit(
            &mut dag,
            Kernel::GoSwiGlu,
            at,
            elementwise(g.intermediate * sorted),
        );
        emit(&mut dag, Kernel::GoExpertDown, at, {
            let mut launch = qmv(g.hidden);
            launch.grid[0] *= sorted;
            launch
        });
        emit(&mut dag, Kernel::GoExpertCombine, at, {
            let w = g.hidden.max(1);
            Launch {
                grid: [w, 1, 1],
                tg: [w.min(256), 1, 1],
            }
        });
        emit(&mut dag, Kernel::LayerOut, at, residual(g.hidden));
    }

    emit(&mut dag, Kernel::FinalRms, None, rms(g.hidden, 1));
    emit(&mut dag, Kernel::LmHeadUntied, None, qmv(g.vocab));
    if with_argmax {
        emit(
            &mut dag,
            Kernel::Argmax,
            None,
            Launch {
                grid: [1024, 1, 1],
                tg: [1024, 1, 1],
            },
        );
    }
    dag
}

/// What the step costs, countable with no GPU.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GptOssDagStats {
    /// Every dispatch.
    pub total: usize,
    /// Layers attending the full context.
    pub full_attn_layers: u32,
    /// Layers attending the sliding window.
    pub sliding_attn_layers: u32,
    /// The matvecs — the step's bandwidth.
    pub gemv: usize,
    /// Of those, the ones whose weights the router picks.
    pub routed: usize,
}

/// Count the DAG's costs.
#[must_use]
pub fn gptoss_dag_stats(dag: &[Dispatch], g: &GptOssGeometry) -> GptOssDagStats {
    let mut s = GptOssDagStats {
        total: dag.len(),
        ..GptOssDagStats::default()
    };
    for layer in 0..g.n_layers {
        if g.is_full_attn(layer) {
            s.full_attn_layers += 1;
        } else {
            s.sliding_attn_layers += 1;
        }
    }
    for d in dag {
        match d.kind {
            Kernel::GoQmvQ
            | Kernel::GoQmvK
            | Kernel::GoQmvV
            | Kernel::GoQmvO
            | Kernel::GoRouter
            | Kernel::LmHeadUntied => s.gemv += 1,
            Kernel::GoExpertGate | Kernel::GoExpertUp | Kernel::GoExpertDown => {
                s.gemv += 1;
                s.routed += 1;
            }
            _ => {}
        }
    }
    s
}

// ─── The M>1 fire path ───
//
// The C++ split this across `pso_for_mb_rows` and `launch_shape_mb`, two
// switches that had to answer four tiling questions IDENTICALLY — which
// SDPA shape, whether the mixture tiles, whether a dense projection tiles,
// at which block — and said so itself, twice: "a grid computed for one
// tiling against a pipeline compiled for another is not a crash, it is
// wrong numbers." Here the questions are asked once, in the builder, and
// the answers ride the [`Dispatch`] (`qmm_bn`, `qmm_bm`, the launch): the
// pipeline lookup and the grid read the same decision because there is
// only one.

/// The multibatch form of each gpt-oss kind: the paged KV pair. Everything
/// else keeps its name — an `_mb` pipeline choice is the PSO table's
/// business, not the kind's.
#[must_use]
pub fn gptoss_mb_kind(kind: Kernel) -> Kernel {
    match kind {
        Kernel::KvAppend => Kernel::KvAppendPaged,
        Kernel::GoSdpaSink => Kernel::GoSdpaSinkPaged,
        other => other,
    }
}

/// The dense projections — the ones a batched GEMM can serve.
///
/// NOT the routed three: each row picks its own experts, so their weight
/// is chosen on the GPU and a tile spanning rows would span weights. And
/// NOT the router: its output is `n_experts` wide, below every column
/// tile.
#[must_use]
pub fn gptoss_is_dense_proj(kind: Kernel) -> bool {
    matches!(
        kind,
        Kernel::GoQmvQ | Kernel::GoQmvK | Kernel::GoQmvV | Kernel::GoQmvO | Kernel::LmHeadUntied
    )
}

/// The widest activation an N-token gpt-oss fire ping-pongs through the
/// pool — the MB twin of
/// [`gptoss_scratch_elems`](super::gptoss_scratch_elems), which it
/// collapses to at one row. The mixture's sorted stack dominates every
/// real fire; the dense term uses the POOL's padded row count
/// ([`gptoss_qmm_pool_rows`]) because the GEMM writes its padding rows,
/// and a pool sized to the true batch would put those writes in the next
/// colour's slot.
#[must_use]
pub fn gptoss_scratch_elems_mb(g: &GptOssGeometry, tuning: &Tuning, rows: u32) -> u64 {
    let stack = u64::from(gptoss_moe_sorted_rows(g, tuning, rows))
        * u64::from(g.intermediate.max(g.hidden));
    let dense = u64::from(gptoss_qmm_pool_rows(rows)) * u64::from(g.q_dim().max(g.hidden));
    let router = u64::from(rows.max(1)) * u64::from(g.n_experts);
    stack.max(dense).max(router)
}

/// How many rows a dense projection's GEMM runs over: whole row tiles.
///
/// Padded UNCONDITIONALLY once past the crossover, unlike the shared
/// [`qmm_mb_rows`](super::qmm_mb_rows) which refuses when padding would
/// overrun the pool — this family sizes its pool to the widest tile up
/// front ([`gptoss_qmm_pool_rows`]), so the refusal has nothing to refuse.
/// The padding computes discardable values into pool rows the fire does
/// not read.
#[must_use]
pub fn gptoss_qmm_rows(rows: u32, min_batch: u32) -> u32 {
    let n = rows.max(1);
    if n < min_batch {
        return n;
    }
    n.div_ceil(qmm_bm(n)) * qmm_bm(n)
}

/// The activation pool's row count for a fire of up to `max_rows`: padded
/// to the WIDEST row tile, so [`gptoss_qmm_rows`] can pad any batch
/// without a bounds question.
#[must_use]
pub fn gptoss_qmm_pool_rows(max_rows: u32) -> u32 {
    let wide = QMM_BMS[QMM_BMS.len() - 1];
    max_rows.max(1).div_ceil(wide) * wide
}

/// The mixture's sorted-stack extent for an `rows`-token fire: every
/// `(token, slot)` pair, padded to the tile the sort will group by. The
/// tile is [`Tuning::moe_tile_rows`]'s answer — the SAME answer the GEMM
/// tile choice reads, which is the point.
#[must_use]
pub fn gptoss_moe_sorted_rows(g: &GptOssGeometry, tuning: &Tuning, rows: u32) -> u32 {
    let pairs = rows.max(1).saturating_mul(g.experts_per_token);
    let tile = tuning.moe_tile_rows(pairs, g.n_experts);
    u32::try_from(sorted_rows(pairs, g.n_experts, tile)).expect("a sort is bounded")
}

/// The routed GEMM's column tile for an `rows`-token fire, or 0 to keep
/// the matvec.
///
/// Only the MXFP4 bank has a routed GEMM instantiation, and only a sort
/// that actually tiles (tile > 1) produces the padded runs the GEMM's row
/// blocks assume. The widest tile that divides IS right here, unlike the
/// dense rule: the sorted mixture supplies the threadgroups a wide tile
/// gives up, because every expert with rows contributes its own (measured
/// at 448 rows — BN=16 374.7 tok/s, BN=32 420.8, BN=64 457.9).
#[must_use]
pub fn gptoss_moe_qmm_bn(kind: Kernel, g: &GptOssGeometry, tuning: &Tuning, rows: u32) -> u32 {
    let routed = matches!(
        kind,
        Kernel::GoExpertGate | Kernel::GoExpertUp | Kernel::GoExpertDown
    );
    let pairs = rows.max(1).saturating_mul(g.experts_per_token);
    if !routed || !g.mxfp4_experts || tuning.moe_tile_rows(pairs, g.n_experts) <= 1 {
        return 0;
    }
    let n = gptoss_qmv_kn(kind, g).n;
    let mut bn = 0;
    for candidate in [16, 32, 64] {
        if n.is_multiple_of(candidate) {
            bn = candidate;
        }
    }
    bn
}

/// This family's GEMM crossover. `is_moe` is TRUE unconditionally: gpt-oss
/// is a mixture in every checkpoint there is, so there is no dense gpt-oss
/// to ask about. The FP16 flag is the tuning's alone for the same reason —
/// the routed MXFP4 GEMM stages its tile as half whatever the projections
/// are quantized at.
#[must_use]
pub fn gptoss_qmm_min_batch(tuning: &Tuning) -> u32 {
    tuning.qmm_min_batch_for(true, tuning.fp16_qmm)
}

/// A dense projection's GEMM column tile for `rows` rows, or 0 to keep the
/// matvec. `qmm_bn_unsplit`, not `qmm_bn`: the widest-tile rule is correct
/// only where split-K supplies the threadgroups a wide tile gives up, and
/// this family dispatches no split at all.
#[must_use]
pub fn gptoss_qmm_bn(kind: Kernel, g: &GptOssGeometry, tuning: &Tuning, rows: u32) -> u32 {
    if !gptoss_is_dense_proj(kind) {
        return 0;
    }
    let n = gptoss_qmv_kn(kind, g).n;
    if n == 0 {
        return 0;
    }
    let min_batch = gptoss_qmm_min_batch(tuning);
    qmm_bn_unsplit(
        n,
        gptoss_qmm_rows(rows, min_batch),
        min_batch,
        tuning.qmm_bn_crossover_tg,
    )
}

fn rms_mb(row_size: u32, n: u32) -> Launch {
    let t = row_size.div_ceil(4).min(1024);
    Launch {
        grid: [t * n, 1, 1],
        tg: [t, 1, 1],
    }
}

fn elementwise_mb(width: u32, n: u32) -> Launch {
    Launch {
        grid: [width * n, 1, 1],
        tg: [256, 1, 1],
    }
}

fn qmv_mb(out_vec: u32, n: u32) -> Launch {
    Launch {
        grid: [32 * n.max(1), out_vec.div_ceil(4), 1],
        tg: [32, 2, 1],
    }
}

fn qmm_t(out_vec: u32, rows: u32, bn: u32, bm: u32) -> Launch {
    Launch {
        grid: [32 * (out_vec / bn), 2 * (rows / bm), 2],
        tg: [32, 2, 2],
    }
}

/// The M>1 DAG: the M=1 order with the paged kinds, the sampled-row
/// compaction restored to the tail, and every tiling decision made HERE,
/// once, carried on the dispatch.
///
/// `head_rows` is how many rows the fire will SAMPLE — what the row gather
/// compacts to, and the extent of everything after it. 0 means "every row"
/// (a fleet of decodes samples them all). The LM head is `hidden × vocab`
/// per row, so on a prefill that compaction is most of the fire's cost.
#[must_use]
pub fn build_gptoss_dag_mb(
    g: &GptOssGeometry,
    tuning: &Tuning,
    n_tokens: u32,
    head_rows: u32,
    ordinal_base: u32,
    with_argmax: bool,
) -> Vec<Dispatch> {
    assert!(n_tokens > 0, "a multibatch DAG carries at least one token");
    let n = n_tokens;
    let s = if head_rows == 0 { n } else { head_rows.min(n) };
    let sorted = gptoss_moe_sorted_rows(g, tuning, n);
    let min_batch = gptoss_qmm_min_batch(tuning);

    let mut dag: Vec<Dispatch> = Vec::new();
    for base in build_gptoss_dag(g, with_argmax) {
        // The compaction sits where the C++ emits it: after the last
        // layer, before the tail norm.
        if base.kind == Kernel::FinalRms {
            dag.push(Dispatch {
                kind: Kernel::G4RowGather,
                ordinal: 0,
                layer: None,
                launch: Launch {
                    grid: [g.hidden, s, 1],
                    tg: [64, 1, 1],
                },
                fuse_residual: false,
                qmm_bn: 0,
                qmm_split: 1,
                qmm_bm: 16,
            });
        }
        let mut d = base;
        d.kind = gptoss_mb_kind(d.kind);
        // The tail runs on the rows the fire will SAMPLE, which the row
        // gather compacted.
        let m = if d.kind == Kernel::LmHeadUntied { s } else { n };
        if let bn @ 1.. = gptoss_moe_qmm_bn(d.kind, g, tuning, n) {
            let pairs = n.saturating_mul(g.experts_per_token);
            d.qmm_bn = bn;
            d.qmm_bm = tuning.moe_tile_rows(pairs, g.n_experts);
            d.launch = qmm_t(gptoss_qmv_kn(d.kind, g).n, sorted, bn, d.qmm_bm);
        } else if let bn @ 1.. = gptoss_qmm_bn(d.kind, g, tuning, m) {
            let rows = gptoss_qmm_rows(m, min_batch);
            d.qmm_bn = bn;
            d.qmm_bm = qmm_bm(rows);
            d.launch = qmm_t(gptoss_qmv_kn(d.kind, g).n, rows, bn, d.qmm_bm);
        } else {
            d.launch = match d.kind {
                Kernel::EmbedUntied => Launch {
                    grid: [g.hidden, n, 1],
                    tg: [256, 1, 1],
                },
                Kernel::Rms | Kernel::FfnRms => rms_mb(g.hidden, n),
                Kernel::FinalRms => rms_mb(g.hidden, s),
                Kernel::Rope => Launch {
                    grid: [g.head_dim / 2, g.n_q_heads, n],
                    tg: [g.head_dim / 2, 1, 1],
                },
                Kernel::RopeK => Launch {
                    grid: [g.head_dim / 2, g.n_kv_heads, n],
                    tg: [g.head_dim / 2, 1, 1],
                },
                Kernel::KvAppendPaged => Launch {
                    grid: [g.head_dim, g.n_kv_heads, n],
                    tg: [g.head_dim, 1, 1],
                },
                // The scalar paged sink at every fleet width; the tiled
                // and matrix shapes are ledgered with the shared family's
                // — the same deferral for the same reason.
                Kernel::GoSdpaSinkPaged => Launch {
                    grid: [g.n_q_heads * 1024, n, 1],
                    tg: [1024, 1, 1],
                },
                Kernel::Residual | Kernel::LayerOut => elementwise_mb(g.hidden, n),
                Kernel::GoRouterTopK => {
                    let w = router_lane_width(g.n_experts);
                    Launch {
                        grid: [w, n, 1],
                        tg: [w, 1, 1],
                    }
                }
                Kernel::LlMoeSort => route_sort(g.n_experts),
                Kernel::LlMoeGather => route_rows(g.hidden, sorted),
                Kernel::GoSwiGlu => elementwise_mb(g.intermediate, sorted),
                Kernel::GoExpertCombine => {
                    let w = g.hidden.max(1);
                    Launch {
                        grid: [w, n, 1],
                        tg: [w.min(256), 1, 1],
                    }
                }
                // The routed matvec over the sorted stack, the router and
                // the dense matvecs below the crossover: rows on the first
                // grid axis, exactly the M=1 sorted wiring.
                Kernel::GoExpertGate | Kernel::GoExpertUp => qmv_mb(g.intermediate, sorted),
                Kernel::GoExpertDown => qmv_mb(g.hidden, sorted),
                Kernel::GoQmvQ => qmv_mb(g.q_dim(), n),
                Kernel::GoQmvK | Kernel::GoQmvV => qmv_mb(g.kv_dim(), n),
                Kernel::GoQmvO => qmv_mb(g.hidden, n),
                Kernel::GoRouter => qmv_mb(g.n_experts, n),
                Kernel::LmHeadUntied => qmv_mb(g.vocab, s),
                // The argmax and the gather keep their shapes.
                other => {
                    debug_assert!(
                        matches!(other, Kernel::Argmax),
                        "missing multibatch launch geometry for {other:?}"
                    );
                    d.launch
                }
            };
        }
        dag.push(d);
    }
    for (i, d) in dag.iter_mut().enumerate() {
        d.ordinal = ordinal_base + u32::try_from(i).expect("a DAG is hundreds of dispatches");
    }
    dag
}

/// One value's shape, off its WRITER — this family's half of
/// [`pool_colour_elems`](super::pool_colour_elems). The int32 traps are
/// the shared ones: ids count two two-byte elements per entry, one kind
/// writes ids and weights, the sort's index outputs size as the
/// tallest.
#[must_use]
pub fn gptoss_value_extent(d: &Dispatch, g: &GptOssGeometry) -> super::sizing::ValueExtent {
    use super::sizing::{RowAxis, ValueExtent};
    let e = |elems: u32, axis: RowAxis| ValueExtent { elems, axis };
    match d.kind {
        Kernel::EmbedUntied
        | Kernel::Rms
        | Kernel::FfnRms
        | Kernel::Residual
        | Kernel::LayerOut
        | Kernel::GoQmvO
        | Kernel::GoExpertCombine => e(g.hidden, RowAxis::Body),
        Kernel::FinalRms | Kernel::G4RowGather => e(g.hidden, RowAxis::Tail),
        Kernel::GoQmvQ | Kernel::Rope | Kernel::GoSdpaSink | Kernel::GoSdpaSinkPaged => {
            e(g.q_dim(), RowAxis::Body)
        }
        Kernel::GoQmvK | Kernel::GoQmvV | Kernel::RopeK => e(g.kv_dim(), RowAxis::Body),
        Kernel::GoRouter => e(g.n_experts, RowAxis::Body),
        Kernel::GoRouterTopK => e(g.experts_per_token * 2, RowAxis::Body),
        Kernel::LlMoeSort => e(2, RowAxis::Sorted),
        Kernel::LlMoeGather => e(g.hidden, RowAxis::Sorted),
        Kernel::GoExpertGate | Kernel::GoExpertUp | Kernel::GoSwiGlu => {
            e(g.intermediate, RowAxis::Sorted)
        }
        Kernel::GoExpertDown => e(g.hidden, RowAxis::Sorted),
        _ => e(0, RowAxis::Body),
    }
}

/// Each pool colour's element count for a `rows`-token fire — one
/// composition of the same pure pieces the fire uses.
#[must_use]
pub fn gptoss_pool_elems(
    g: &GptOssGeometry,
    tuning: &Tuning,
    rows: u32,
    head_rows: u32,
) -> Vec<u64> {
    let dag = build_gptoss_dag_mb(g, tuning, rows.max(1), head_rows, 0, false);
    let (uses, values) = super::dataflow::build_scratch_uses(&dag);
    let ends = super::dispatch::concurrent_run_ends(&dag);
    let coloring = super::color::color_live_ranges(&uses, &ends, values, false)
        .expect("the gpt-oss DAG colours");
    super::sizing::pool_colour_elems(
        &dag,
        &uses,
        &coloring,
        |d| gptoss_value_extent(d, g),
        rows,
        head_rows,
        gptoss_moe_sorted_rows(g, tuning, rows),
        u64::from(rows.max(1)) * u64::from(g.hidden),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_20b_step_is_twenty_one_dispatches_a_layer_plus_the_ends() {
        let g = GptOssGeometry::default();
        let dag = build_gptoss_dag(&g, true);
        // 1 embed + 24 x 21 + final norm + head + argmax.
        assert_eq!(dag.len(), 1 + 24 * 21 + 3);
        assert!(dag.iter().enumerate().all(|(i, d)| d.ordinal as usize == i));
        let stats = gptoss_dag_stats(&dag, &g);
        assert_eq!(stats.full_attn_layers, 12);
        assert_eq!(stats.sliding_attn_layers, 12);
        assert_eq!(
            stats.gemv,
            24 * 8 + 1,
            "seven per layer plus the router, plus the head"
        );
        assert_eq!(stats.routed, 24 * 3);
    }

    #[test]
    fn the_routed_projections_launch_over_the_sorted_stack() {
        let g = GptOssGeometry::default();
        let dag = build_gptoss_dag(&g, false);
        let gate = dag
            .iter()
            .find(|d| d.kind == Kernel::GoExpertGate)
            .expect("every layer routes");
        // Four sorted rows at decode (top-4, tile 1).
        assert_eq!(gate.launch.grid[0], 32 * 4);
        let plain = dag.iter().find(|d| d.kind == Kernel::GoQmvQ).unwrap();
        assert_eq!(plain.launch.grid[0], 32, "the dense matvec stays one row");
    }

    #[test]
    fn the_mb_dag_swaps_the_ring_for_pages_and_compacts_the_sampled_rows() {
        let g = GptOssGeometry::default();
        let tuning = Tuning::default();
        let dag = build_gptoss_dag_mb(&g, &tuning, 16, 2, 100, true);
        assert!(dag.iter().any(|d| d.kind == Kernel::KvAppendPaged));
        assert!(dag.iter().any(|d| d.kind == Kernel::GoSdpaSinkPaged));
        assert!(
            dag.iter()
                .all(|d| d.kind != Kernel::KvAppend && d.kind != Kernel::GoSdpaSink),
            "the ring kinds do not survive into the paged fire"
        );
        // The compaction sits immediately before the tail norm, sized to
        // the SAMPLED rows, and everything after it runs on that prefix.
        let gather = dag
            .iter()
            .position(|d| d.kind == Kernel::G4RowGather)
            .expect("the prefill tail compacts");
        assert_eq!(dag[gather + 1].kind, Kernel::FinalRms);
        assert_eq!(dag[gather].launch.grid, [g.hidden, 2, 1]);
        let head = dag.iter().find(|d| d.kind == Kernel::LmHeadUntied).unwrap();
        assert_eq!(head.launch.grid[0], 32 * 2, "the head runs on two rows");
        let norm = dag.iter().rfind(|d| d.kind == Kernel::FinalRms).unwrap();
        assert_eq!(norm.launch.grid[0], (g.hidden / 4) * 2);
        // Ordinals renumber from the base, gather included.
        assert!(
            dag.iter()
                .enumerate()
                .all(|(i, d)| d.ordinal as usize == 100 + i)
        );
        assert_eq!(dag.len(), 1 + 24 * 21 + 4, "one more than the M=1 step");
        // And the schedule covers it: the dataflow walk knows every kind
        // this DAG dispatches, compaction and paged attention included.
        crate::batch::build_scratch_schedule(&dag, false).expect("the MB DAG colours");
    }

    #[test]
    fn a_sixteen_row_fire_decides_its_tiles_once_and_writes_them_down() {
        let g = GptOssGeometry {
            mxfp4_experts: true,
            ..GptOssGeometry::default()
        };
        let tuning = Tuning::default();
        let dag = build_gptoss_dag_mb(&g, &tuning, 16, 0, 0, false);
        // The dense q projection tiles: 16 rows is one whole 16-row block,
        // and 4096 columns past the crossover take the 32 tile.
        let q = dag.iter().find(|d| d.kind == Kernel::GoQmvQ).unwrap();
        assert_eq!((q.qmm_bn, q.qmm_bm), (32, 16));
        assert_eq!(
            q.launch.grid,
            [32 * (g.q_dim() / 32), 2, 2],
            "one row block"
        );
        // The narrow kv projections stay on the narrow tile: 512 columns
        // is 32 threadgroups, under the crossover.
        let k = dag.iter().find(|d| d.kind == Kernel::GoQmvK).unwrap();
        assert_eq!(k.qmm_bn, 16);
        // The mixture tiles at the width the SORT pads to — 64 pairs over
        // 32 experts is two rows an expert, the 16 tile — and the widest
        // dividing column tile, which is the routed rule, not the dense
        // one.
        let sorted = gptoss_moe_sorted_rows(&g, &tuning, 16);
        assert_eq!(sorted, 544, "64 pairs, every expert padded to 16 rows");
        let gate = dag.iter().find(|d| d.kind == Kernel::GoExpertGate).unwrap();
        assert_eq!((gate.qmm_bn, gate.qmm_bm), (64, 16));
        assert_eq!(
            gate.launch.grid,
            [32 * (g.intermediate / 64), 2 * (sorted / 16), 2]
        );
        // The router is not a dense projection: 32 columns fit no tile,
        // and the decision is 0 rather than a lie.
        let router = dag.iter().find(|d| d.kind == Kernel::GoRouter).unwrap();
        assert_eq!(router.qmm_bn, 0);
        assert_eq!(router.launch.grid[0], 32 * 16, "the matvec rows ride x");
    }

    #[test]
    fn below_the_crossover_and_without_the_bank_every_matvec_survives() {
        let tuning = Tuning::default();
        // Two rows pay for no tile anywhere.
        let g = GptOssGeometry {
            mxfp4_experts: true,
            ..GptOssGeometry::default()
        };
        let two = build_gptoss_dag_mb(&g, &tuning, 2, 0, 0, false);
        assert!(two.iter().all(|d| d.qmm_bn == 0));
        // The affine bank has no routed GEMM instantiation: sixteen rows
        // tile the dense projections and leave the mixture alone.
        let affine = GptOssGeometry::default();
        assert!(!affine.mxfp4_experts, "the default bank is affine");
        let dag = build_gptoss_dag_mb(&affine, &tuning, 16, 0, 0, false);
        let gate = dag.iter().find(|d| d.kind == Kernel::GoExpertGate).unwrap();
        assert_eq!(gate.qmm_bn, 0);
        assert!(
            dag.iter().any(|d| d.qmm_bn > 0),
            "the dense side still tiles"
        );
        // The pool padding rule that lets the row padding skip the bounds
        // question: any batch, padded, fits a pool padded to the widest.
        for rows in [1u32, 7, 16, 33, 100] {
            assert!(
                gptoss_qmm_rows(rows, gptoss_qmm_min_batch(&tuning)) <= gptoss_qmm_pool_rows(rows)
            );
        }
    }
}

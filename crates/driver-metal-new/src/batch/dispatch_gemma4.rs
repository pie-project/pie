//! Gemma 4's per-token dispatch DAG, in the shared [`Kernel`]
//! vocabulary.
//!
//! A pure function like the other three builders, and the family where
//! the DAG carries the most schedule: which dispatches EXIST moves per
//! layer. A KV-shared layer rotates its own Q and reads the pages its
//! source wrote — no k/v projection, no k/v norm, no append. A k-eq-v
//! layer projects K and takes V from it, so it has no `v_proj` weight
//! at all; suppressing the dispatch is not an optimisation, it is which
//! tensors the checkpoint ships. The mixture's branch sits BESIDE the
//! dense MLP, both reading the post-attention residual — five norms
//! round one block.
//!
//! Sliding and full attention are two KINDS ([`Kernel::G4SdpaSliding`]
//! vs [`Kernel::Sdpa`]) rather than a flag: the port's rule that a
//! per-fire choice the C++ made at encode time becomes data on the
//! dispatch.
//!
//! `RowGather` deliberately absent at M=1, as with the other families.

use super::abi::Kernel;
use super::dispatch::{
    Dispatch, Launch, embed, kv_append, qmv, residual, rms, rope, route_rows, route_sort,
    routed_qmv, router_topk, sdpa,
};
use super::dispatch_mb::{qmm_t, rms_mb};
use super::gemma4::{Gemma4Geometry, gemma4_qmv_kn};
use crate::tuning::Tuning;

use super::sizing::sorted_rows;

fn elementwise(width: u32) -> Launch {
    Launch {
        grid: [width, 1, 1],
        tg: [256, 1, 1],
    }
}

/// Emit the ordered per-token DAG for `g`.
#[must_use]
#[allow(clippy::too_many_lines)] // one walk; splitting hides the layer schedule
pub fn build_gemma4_dag(g: &Gemma4Geometry, with_argmax: bool) -> Vec<Dispatch> {
    let mut dag: Vec<Dispatch> = Vec::with_capacity(g.n_layers as usize * 32 + 16);
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
    let mv = |kind: Kernel, layer: Option<u32>| qmv(gemma4_qmv_kn(kind, g, layer).n);
    // At decode the sort is a pure grouping: tile 1.
    let sorted = u32::try_from(sorted_rows(
        g.experts_per_token.max(1),
        g.n_experts.max(1),
        1,
    ))
    .expect("a decode sort is small");

    // The PLE precompute, once per step, layer-less.
    emit(&mut dag, Kernel::EmbedGather, None, embed(g.hidden));
    if g.per_layer_emb_dim > 0 {
        emit(
            &mut dag,
            Kernel::G4PleTokenGather,
            None,
            embed(g.n_layers * g.per_layer_emb_dim),
        );
        emit(
            &mut dag,
            Kernel::G4PleProjGemv,
            None,
            mv(Kernel::G4PleProjGemv, None),
        );
        emit(
            &mut dag,
            Kernel::G4PleProjNorm,
            None,
            rms(g.per_layer_emb_dim, g.n_layers),
        );
        emit(
            &mut dag,
            Kernel::G4PleCombine,
            None,
            elementwise(g.n_layers * g.per_layer_emb_dim),
        );
    }

    for layer in 0..g.n_layers {
        let at = Some(layer);
        let hd = g.head_dim_of(layer);
        let kv_heads = g.n_kv_heads_of(layer);
        // A KV-shared layer rotates its own Q and reads its source's
        // pages: no k/v projection, no k/v norm, no append.
        let owns_kv = !g.is_kv_shared(layer);
        let owns_v = owns_kv && !g.k_is_v(layer);

        emit(&mut dag, Kernel::Rms, at, rms(g.hidden, 1));
        emit(&mut dag, Kernel::QmvQ, at, mv(Kernel::QmvQ, at));
        if owns_kv {
            emit(&mut dag, Kernel::QmvK, at, mv(Kernel::QmvK, at));
            if owns_v {
                emit(&mut dag, Kernel::QmvV, at, mv(Kernel::QmvV, at));
            }
        }
        emit(&mut dag, Kernel::QNorm, at, rms(hd, g.n_q_heads));
        if owns_kv {
            if owns_v {
                emit(&mut dag, Kernel::KNorm, at, rms(hd, kv_heads));
                emit(&mut dag, Kernel::G4VNorm, at, rms(hd, kv_heads));
            } else {
                // V first: it reads the projection `KNorm` is about to
                // overwrite.
                emit(&mut dag, Kernel::G4VNormFromK, at, rms(hd, kv_heads));
                emit(&mut dag, Kernel::KNorm, at, rms(hd, kv_heads));
            }
        }
        emit(
            &mut dag,
            Kernel::Rope,
            at,
            rope(g.rotary_dims_of(layer), g.n_q_heads),
        );
        if owns_kv {
            emit(
                &mut dag,
                Kernel::RopeK,
                at,
                rope(g.rotary_dims_of(layer), kv_heads),
            );
            emit(&mut dag, Kernel::KvAppend, at, kv_append(hd, kv_heads));
        }
        emit(
            &mut dag,
            if g.is_sliding(layer) {
                Kernel::G4SdpaSliding
            } else {
                Kernel::Sdpa
            },
            at,
            sdpa(g.n_q_heads),
        );
        emit(&mut dag, Kernel::QmvO, at, mv(Kernel::QmvO, at));
        // The sandwich: `rms(block)·w + resid`, fused.
        emit(&mut dag, Kernel::G4AttnPostResidual, at, rms(g.hidden, 1));

        emit(&mut dag, Kernel::G4FfnPreNorm, at, rms(g.hidden, 1));
        emit(&mut dag, Kernel::QmvGate, at, mv(Kernel::QmvGate, at));
        emit(&mut dag, Kernel::QmvUp, at, mv(Kernel::QmvUp, at));
        emit(
            &mut dag,
            Kernel::G4Geglu,
            at,
            elementwise(g.intermediate_of(layer)),
        );
        emit(&mut dag, Kernel::QmvDown, at, mv(Kernel::QmvDown, at));
        if g.is_moe() {
            emit(&mut dag, Kernel::G4DenseBranchNorm, at, rms(g.hidden, 1));
            emit(&mut dag, Kernel::G4RouterNorm, at, rms(g.hidden, 1));
            emit(&mut dag, Kernel::G4Router, at, mv(Kernel::G4Router, at));
            emit(&mut dag, Kernel::G4RouterTopK, at, router_topk(g.n_experts));
            emit(&mut dag, Kernel::G4MoeNorm, at, rms(g.hidden, 1));
            emit(&mut dag, Kernel::G4MoeSort, at, route_sort(g.n_experts));
            emit(
                &mut dag,
                Kernel::G4MoeGather,
                at,
                route_rows(g.hidden, sorted),
            );
            emit(
                &mut dag,
                Kernel::G4ExpertGate,
                at,
                routed_qmv(g.moe_intermediate, 1, sorted),
            );
            emit(
                &mut dag,
                Kernel::G4ExpertUp,
                at,
                routed_qmv(g.moe_intermediate, 1, sorted),
            );
            emit(
                &mut dag,
                Kernel::G4ExpertGeglu,
                at,
                elementwise(g.moe_intermediate * sorted),
            );
            emit(
                &mut dag,
                Kernel::G4ExpertDown,
                at,
                routed_qmv(g.hidden, 1, sorted),
            );
            emit(&mut dag, Kernel::G4ExpertCombine, at, {
                let w = g.hidden.max(1);
                Launch {
                    grid: [w, 1, 1],
                    tg: [w.min(256), 1, 1],
                }
            });
            emit(&mut dag, Kernel::G4MoeBranchNorm, at, rms(g.hidden, 1));
            // NOT the norms' shape: `residual_add` reads one element per
            // thread where the norm reads four.
            emit(&mut dag, Kernel::G4BranchAdd, at, residual(g.hidden));
        }
        emit(&mut dag, Kernel::G4FfnPostResidual, at, rms(g.hidden, 1));

        if g.per_layer_emb_dim > 0 {
            emit(
                &mut dag,
                Kernel::G4PleGateGemv,
                at,
                mv(Kernel::G4PleGateGemv, at),
            );
            emit(&mut dag, Kernel::G4PleGeglu, at, {
                let w = g.per_layer_emb_dim;
                Launch {
                    grid: [w, 1, 1],
                    tg: [w.min(256), 1, 1],
                }
            });
            emit(
                &mut dag,
                Kernel::G4PleProjLayerGemv,
                at,
                mv(Kernel::G4PleProjLayerGemv, at),
            );
            emit(&mut dag, Kernel::G4PleResidualScaled, at, rms(g.hidden, 1));
        } else {
            emit(&mut dag, Kernel::G4LayerScalar, at, elementwise(g.hidden));
        }
    }

    emit(&mut dag, Kernel::FinalRms, None, rms(g.hidden, 1));
    emit(
        &mut dag,
        if g.tied_embeddings {
            Kernel::QmvLmHead
        } else {
            Kernel::LmHeadUntied
        },
        None,
        qmv(g.vocab),
    );
    if g.final_softcap > 0.0 {
        emit(&mut dag, Kernel::G4Softcap, None, elementwise(g.vocab));
    }
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
pub struct Gemma4DagStats {
    /// Every dispatch.
    pub total: usize,
    /// Layers that own KV pages.
    pub kv_owning_layers: u32,
    /// Layers re-attending another's pages.
    pub kv_shared_layers: u32,
    /// Layers attending the full context.
    pub full_attn_layers: u32,
    /// Layers attending the window.
    pub sliding_attn_layers: u32,
    /// The matvecs — the step's bandwidth.
    pub gemv: usize,
}

/// Count the DAG's costs.
#[must_use]
pub fn gemma4_dag_stats(dag: &[Dispatch], g: &Gemma4Geometry) -> Gemma4DagStats {
    let mut s = Gemma4DagStats {
        total: dag.len(),
        ..Gemma4DagStats::default()
    };
    for layer in 0..g.n_layers {
        if g.is_kv_shared(layer) {
            s.kv_shared_layers += 1;
        } else {
            s.kv_owning_layers += 1;
        }
        if g.is_full_attn(layer) {
            s.full_attn_layers += 1;
        } else {
            s.sliding_attn_layers += 1;
        }
    }
    for d in dag {
        if gemma4_qmv_kn(d.kind, g, d.layer).n != 0 {
            s.gemv += 1;
        }
    }
    s
}

/// One value's shape, off its WRITER — per LAYER, as everything in this
/// family is.
#[must_use]
#[allow(clippy::match_same_arms)] // grouped by meaning, not by width
pub fn gemma4_value_extent(d: &Dispatch, g: &Gemma4Geometry) -> super::sizing::ValueExtent {
    use super::sizing::{RowAxis, ValueExtent};
    let e = |elems: u32, axis: RowAxis| ValueExtent { elems, axis };
    let l = d.layer;
    let hd = l.map_or(g.global_head_dim.max(g.head_dim), |l| g.head_dim_of(l));
    let kv = l.map_or(g.n_kv_heads, |l| g.n_kv_heads_of(l));
    match d.kind {
        Kernel::EmbedGather
        | Kernel::Rms
        | Kernel::G4FfnPreNorm
        | Kernel::G4AttnPostResidual
        | Kernel::G4FfnPostResidual
        | Kernel::G4PleResidualScaled
        | Kernel::G4LayerScalar
        | Kernel::QmvO
        | Kernel::QmvDown
        | Kernel::G4DenseBranchNorm
        | Kernel::G4RouterNorm
        | Kernel::G4MoeNorm
        | Kernel::G4MoeBranchNorm
        | Kernel::G4BranchAdd
        | Kernel::G4ExpertCombine
        | Kernel::G4PleProjLayerGemv => e(g.hidden, RowAxis::Body),
        Kernel::FinalRms | Kernel::G4RowGather => e(g.hidden, RowAxis::Tail),
        Kernel::QmvQ | Kernel::QNorm | Kernel::Rope | Kernel::Sdpa | Kernel::G4SdpaSliding => {
            e(g.n_q_heads * hd, RowAxis::Body)
        }
        Kernel::QmvK
        | Kernel::QmvV
        | Kernel::KNorm
        | Kernel::RopeK
        | Kernel::G4VNorm
        | Kernel::G4VNormFromK => e(kv * hd, RowAxis::Body),
        Kernel::QmvGate | Kernel::QmvUp | Kernel::G4Geglu => e(
            l.map_or(g.intermediate, |l| g.intermediate_of(l)),
            RowAxis::Body,
        ),
        Kernel::G4PleTokenGather
        | Kernel::G4PleProjGemv
        | Kernel::G4PleProjNorm
        | Kernel::G4PleCombine => e(g.n_layers * g.per_layer_emb_dim, RowAxis::Body),
        Kernel::G4PleGateGemv | Kernel::G4PleGeglu => e(g.per_layer_emb_dim, RowAxis::Body),
        Kernel::G4Router => e(g.n_experts, RowAxis::Body),
        Kernel::G4RouterTopK => e(g.experts_per_token * 2, RowAxis::Body),
        Kernel::G4MoeSort => e(2, RowAxis::Sorted),
        Kernel::G4MoeGather => e(g.hidden, RowAxis::Sorted),
        Kernel::G4ExpertGate | Kernel::G4ExpertUp | Kernel::G4ExpertGeglu => {
            e(g.moe_intermediate, RowAxis::Sorted)
        }
        Kernel::G4ExpertDown => e(g.hidden, RowAxis::Sorted),
        _ => e(0, RowAxis::Body),
    }
}

/// Each pool colour's element count for a `rows`-token fire. The C++
/// asks its MB DAG, which is the SAME dispatch list with shifted
/// ordinals — the M=1 list serves, and the rows ride the axes.
#[must_use]
pub fn gemma4_pool_elems(
    g: &Gemma4Geometry,
    tuning: &Tuning,
    rows: u32,
    head_rows: u32,
) -> Vec<u64> {
    let dag = build_gemma4_dag(g, false);
    let (uses, values) = super::dataflow::build_scratch_uses(&dag);
    let ends = super::dispatch::concurrent_run_ends(&dag);
    let coloring = super::color::color_live_ranges(&uses, &ends, values, false)
        .expect("the gemma4 DAG colours");
    let pairs = rows.max(1).saturating_mul(g.experts_per_token.max(1));
    let sorted = if g.is_moe() {
        u32::try_from(super::sizing::sorted_rows(
            pairs,
            g.n_experts,
            tuning.moe_tile_rows(pairs, g.n_experts),
        ))
        .expect("a sort is bounded")
    } else {
        rows.max(1)
    };
    super::sizing::pool_colour_elems(
        &dag,
        &uses,
        &coloring,
        |d| gemma4_value_extent(d, g),
        rows,
        head_rows,
        sorted,
        u64::from(rows.max(1)) * u64::from(g.hidden),
    )
}

// ─── The M>1 fire path ───

/// This family's GEMM crossover — dense or routed by what the FFN is,
/// FP16 by the model format.
#[must_use]
pub fn gemma4_qmm_min_batch(g: &Gemma4Geometry, tuning: &Tuning) -> u32 {
    tuning.qmm_min_batch_for(
        g.is_moe(),
        tuning.fp16_gemm_format(g.quant.bits, g.quant.group),
    )
}

/// Whole row blocks past the crossover; the pool is pre-padded to the
/// widest tile, as in the other families.
#[must_use]
pub fn gemma4_qmm_rows(g: &Gemma4Geometry, tuning: &Tuning, rows: u32) -> u32 {
    let n = rows.max(1);
    if n < gemma4_qmm_min_batch(g, tuning) {
        return n;
    }
    n.div_ceil(super::dispatch_mb::qmm_bm(n)) * super::dispatch_mb::qmm_bm(n)
}

/// A dense projection's column tile, or 0 for the matvec —
/// `qmm_bn_unsplit`, because this family dispatches no split at all
/// (the C++ records a split that wrote partials nobody summed: a
/// 16-row prefill answered 147040 where the oracle says 476).
#[must_use]
pub fn gemma4_qmm_bn(
    kind: Kernel,
    g: &Gemma4Geometry,
    tuning: &Tuning,
    rows: u32,
    layer: Option<u32>,
) -> u32 {
    let routed = matches!(
        kind,
        Kernel::G4ExpertGate | Kernel::G4ExpertUp | Kernel::G4ExpertDown
    );
    if routed || matches!(kind, Kernel::G4Router) {
        return 0;
    }
    let n = gemma4_qmv_kn(kind, g, layer).n;
    if n == 0 {
        return 0;
    }
    super::dispatch_mb::qmm_bn_unsplit(
        n,
        gemma4_qmm_rows(g, tuning, rows),
        gemma4_qmm_min_batch(g, tuning),
        tuning.qmm_bn_crossover_tg,
    )
}

/// The mixture's sorted-stack extent, tile and padding from the same
/// tuning answer the sort reads.
#[must_use]
pub fn gemma4_moe_sorted_rows(g: &Gemma4Geometry, tuning: &Tuning, rows: u32) -> u32 {
    if !g.is_moe() {
        return rows.max(1);
    }
    let pairs = rows.max(1).saturating_mul(g.experts_per_token);
    let tile = tuning.moe_tile_rows(pairs, g.n_experts);
    u32::try_from(super::sizing::sorted_rows(pairs, g.n_experts, tile)).expect("a sort is bounded")
}

/// The routed GEMM's column tile, or 0 when the batch stays a matvec.
#[must_use]
pub fn gemma4_moe_qmm_bn(kind: Kernel, g: &Gemma4Geometry, tuning: &Tuning, rows: u32) -> u32 {
    if !matches!(
        kind,
        Kernel::G4ExpertGate | Kernel::G4ExpertUp | Kernel::G4ExpertDown
    ) {
        return 0;
    }
    let pairs = rows.max(1).saturating_mul(g.experts_per_token.max(1));
    if tuning.moe_tile_rows(pairs, g.n_experts) <= 1 {
        return 0;
    }
    let n = gemma4_qmv_kn(kind, g, None).n;
    if n == 0 {
        return 0;
    }
    super::dispatch_mb::qmm_bn(
        n,
        gemma4_moe_sorted_rows(g, tuning, rows),
        tuning.qmm_min_batch_for(true, tuning.fp16_gemm_format(g.quant.bits, g.quant.group)),
    )
}

/// The M>1 DAG: the M=1 order with the paged kinds, the compaction in
/// the tail, and every tiling decision made here, once — the shape the
/// gpt-oss and llama arcs closed, at this family's per-layer widths.
/// Both attention types become [`Kernel::SdpaPaged`]: which paged
/// INSTANTIATION serves a layer is its head width, resolved by the
/// step's table from `d.layer`, and the window is the geometry's per
/// layer — the kind no longer needs to carry it.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn build_gemma4_dag_mb(
    g: &Gemma4Geometry,
    tuning: &Tuning,
    n_tokens: u32,
    head_rows: u32,
    ordinal_base: u32,
    with_argmax: bool,
) -> Vec<Dispatch> {
    assert!(n_tokens > 0, "a multibatch DAG carries at least one token");
    let n = n_tokens;
    let s = if head_rows == 0 { n } else { head_rows.min(n) };
    let sorted = gemma4_moe_sorted_rows(g, tuning, n);

    let mut dag: Vec<Dispatch> = Vec::new();
    for base in build_gemma4_dag(g, with_argmax) {
        if base.kind == Kernel::G4RowGather {
            // The M=1 builder omits it; nothing to do — but the M=1
            // list never emits it, so insert before the tail norm.
            continue;
        }
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
        d.kind = match d.kind {
            Kernel::KvAppend => Kernel::KvAppendPaged,
            Kernel::Sdpa | Kernel::G4SdpaSliding => Kernel::SdpaPaged,
            other => other,
        };
        let layer = d.layer;
        let hd = layer.map_or(g.head_dim, |l| g.head_dim_of(l));
        let kv_heads = layer.map_or(g.n_kv_heads, |l| g.n_kv_heads_of(l));
        let m = if matches!(d.kind, Kernel::QmvLmHead | Kernel::LmHeadUntied) {
            s
        } else {
            n
        };
        if let bn @ 1.. = gemma4_moe_qmm_bn(d.kind, g, tuning, n) {
            let pairs = n.saturating_mul(g.experts_per_token.max(1));
            d.qmm_bn = bn;
            d.qmm_bm = tuning.moe_tile_rows(pairs, g.n_experts);
            d.launch = qmm_t(gemma4_qmv_kn(d.kind, g, None).n, sorted, bn, d.qmm_bm);
        } else if let bn @ 1.. = gemma4_qmm_bn(d.kind, g, tuning, m, layer) {
            let rows = gemma4_qmm_rows(g, tuning, m);
            d.qmm_bn = bn;
            d.qmm_bm = super::dispatch_mb::qmm_bm(rows);
            d.launch = qmm_t(gemma4_qmv_kn(d.kind, g, layer).n, rows, bn, d.qmm_bm);
        } else {
            d.launch = match d.kind {
                Kernel::EmbedGather => Launch {
                    grid: [g.hidden, n, 1],
                    tg: [256, 1, 1],
                },
                Kernel::G4PleTokenGather => Launch {
                    grid: [g.n_layers * g.per_layer_emb_dim, n, 1],
                    tg: [256, 1, 1],
                },
                Kernel::Rms
                | Kernel::G4FfnPreNorm
                | Kernel::G4AttnPostResidual
                | Kernel::G4FfnPostResidual
                | Kernel::G4PleResidualScaled
                | Kernel::G4RouterNorm
                | Kernel::G4MoeNorm
                | Kernel::G4DenseBranchNorm
                | Kernel::G4MoeBranchNorm => rms_mb(g.hidden, 1, n),
                Kernel::FinalRms => rms_mb(g.hidden, 1, s),
                Kernel::QNorm => rms_mb(hd, g.n_q_heads, n),
                Kernel::KNorm | Kernel::G4VNorm | Kernel::G4VNormFromK => rms_mb(hd, kv_heads, n),
                Kernel::G4PleProjNorm => rms_mb(g.per_layer_emb_dim, g.n_layers, n),
                Kernel::Rope => Launch {
                    grid: [
                        layer.map_or(g.head_dim / 2, |l| g.rotary_dims_of(l) / 2),
                        g.n_q_heads,
                        n,
                    ],
                    tg: [
                        layer.map_or(g.head_dim / 2, |l| g.rotary_dims_of(l) / 2),
                        1,
                        1,
                    ],
                },
                Kernel::RopeK => Launch {
                    grid: [
                        layer.map_or(g.head_dim / 2, |l| g.rotary_dims_of(l) / 2),
                        kv_heads,
                        n,
                    ],
                    tg: [
                        layer.map_or(g.head_dim / 2, |l| g.rotary_dims_of(l) / 2),
                        1,
                        1,
                    ],
                },
                Kernel::KvAppendPaged => Launch {
                    grid: [hd, kv_heads, n],
                    tg: [hd.min(1024), 1, 1],
                },
                Kernel::SdpaPaged => Launch {
                    grid: [g.n_q_heads * 1024, n, 1],
                    tg: [1024, 1, 1],
                },
                Kernel::G4Geglu => super::dispatch_mb::elementwise_mb(
                    layer.map_or(g.intermediate, |l| g.intermediate_of(l)),
                    n,
                ),
                Kernel::G4LayerScalar | Kernel::G4BranchAdd => {
                    super::dispatch_mb::elementwise_mb(g.hidden, n)
                }
                Kernel::G4PleCombine => {
                    super::dispatch_mb::elementwise_mb(g.n_layers * g.per_layer_emb_dim, n)
                }
                // One thread per (channel, token row): the strided kernel
                // indexes both, because its `up` operand strides by the
                // whole PLE table.
                Kernel::G4PleGeglu => Launch {
                    grid: [g.per_layer_emb_dim, n, 1],
                    tg: [g.per_layer_emb_dim.min(256), 1, 1],
                },
                Kernel::G4RouterTopK => {
                    let w = super::dispatch::router_lane_width(g.n_experts);
                    Launch {
                        grid: [w, n, 1],
                        tg: [w, 1, 1],
                    }
                }
                Kernel::G4MoeSort => route_sort(g.n_experts),
                Kernel::G4MoeGather => route_rows(g.hidden, sorted),
                Kernel::G4ExpertGate | Kernel::G4ExpertUp => {
                    routed_qmv(g.moe_intermediate, 1, sorted)
                }
                Kernel::G4ExpertGeglu => {
                    super::dispatch_mb::elementwise_mb(g.moe_intermediate, sorted)
                }
                Kernel::G4ExpertDown => routed_qmv(g.hidden, 1, sorted),
                Kernel::G4ExpertCombine => {
                    let w = g.hidden.max(1);
                    Launch {
                        grid: [w, n, 1],
                        tg: [w.min(256), 1, 1],
                    }
                }
                Kernel::QmvQ => qmv(g.n_q_heads * hd),
                Kernel::QmvK | Kernel::QmvV => qmv(kv_heads * hd),
                _ => {
                    // The matvecs below the crossover ride the shared MB
                    // matvec shape; the softcap and argmax scale on S.
                    match d.kind {
                        Kernel::QmvO | Kernel::QmvDown | Kernel::G4PleProjLayerGemv => {
                            super::dispatch_mb::qmv_mb(g.hidden, n)
                        }
                        Kernel::QmvGate | Kernel::QmvUp => super::dispatch_mb::qmv_mb(
                            layer.map_or(g.intermediate, |l| g.intermediate_of(l)),
                            n,
                        ),
                        Kernel::G4PleProjGemv => {
                            super::dispatch_mb::qmv_mb(g.n_layers * g.per_layer_emb_dim, n)
                        }
                        Kernel::G4PleGateGemv => super::dispatch_mb::qmv_mb(g.per_layer_emb_dim, n),
                        Kernel::G4Router => super::dispatch_mb::qmv_mb(g.n_experts, n),
                        Kernel::QmvLmHead | Kernel::LmHeadUntied => {
                            super::dispatch_mb::qmv_mb(g.vocab, s)
                        }
                        Kernel::G4Softcap => super::dispatch_mb::elementwise_mb(g.vocab, s),
                        other => {
                            debug_assert!(
                                matches!(other, Kernel::Argmax),
                                "missing multibatch launch geometry for {other:?}"
                            );
                            d.launch
                        }
                    }
                }
            };
        }
        // The dense-projection matvecs above went through qmv() at their
        // per-layer width for rows=1 semantics; at MB they ride qmv_mb.
        if matches!(d.kind, Kernel::QmvQ | Kernel::QmvK | Kernel::QmvV) && d.qmm_bn == 0 {
            let width = if d.kind == Kernel::QmvQ {
                g.n_q_heads * hd
            } else {
                kv_heads * hd
            };
            d.launch = super::dispatch_mb::qmv_mb(width, n);
        }
        dag.push(d);
    }
    for (i, d) in dag.iter_mut().enumerate() {
        d.ordinal = ordinal_base + u32::try_from(i).expect("a DAG is hundreds of dispatches");
    }
    dag
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_e2b_step_carries_the_kv_shared_skips_and_the_ple_chain() {
        let g = Gemma4Geometry::default();
        let dag = build_gemma4_dag(&g, true);
        let stats = gemma4_dag_stats(&dag, &g);
        assert_eq!(stats.kv_owning_layers, 15);
        assert_eq!(stats.kv_shared_layers, 20);
        assert_eq!(stats.full_attn_layers, 7);
        // An owning layer runs 23 dispatches; a shared one drops the
        // SIX KV-owner dispatches (k, v, k-norm, v-norm, rope-k, append).
        // 1 embed + 4 PLE + 15×23 + 20×17 + norm + head + softcap + argmax.
        assert_eq!(dag.len(), 5 + 15 * 23 + 20 * 17 + 4);
        assert!(dag.iter().enumerate().all(|(i, d)| d.ordinal as usize == i));
        // A shared layer emits no KV write; an owner does.
        let of =
            |layer: u32, kind: Kernel| dag.iter().any(|d| d.layer == Some(layer) && d.kind == kind);
        assert!(of(3, Kernel::KvAppend) && !of(16, Kernel::KvAppend));
        assert!(of(3, Kernel::QmvK) && !of(16, Kernel::QmvK));
        // Sliding and full attention are two kinds, per layer.
        assert!(of(3, Kernel::G4SdpaSliding) && !of(3, Kernel::Sdpa));
        assert!(of(4, Kernel::Sdpa) && !of(4, Kernel::G4SdpaSliding));
        // The wide-head full layer's q projection is wider.
        let q_at = |layer: u32| {
            dag.iter()
                .find(|d| d.layer == Some(layer) && d.kind == Kernel::QmvQ)
                .unwrap()
                .launch
                .grid[1]
        };
        assert_eq!(q_at(4), (8 * 512u32).div_ceil(4));
        assert_eq!(q_at(3), (8 * 256u32).div_ceil(4));
        // The softcap sits between the head and the argmax.
        let cap = dag
            .iter()
            .position(|d| d.kind == Kernel::G4Softcap)
            .unwrap();
        assert_eq!(dag[cap - 1].kind, Kernel::QmvLmHead);
        assert_eq!(dag[cap + 1].kind, Kernel::Argmax);
    }

    #[test]
    fn the_26b_layer_swaps_v_projection_for_the_k_reading_norm() {
        let g = Gemma4Geometry {
            n_kv_heads: 8,
            attention_k_eq_v: true,
            n_global_kv_heads: 2,
            enable_moe: true,
            n_experts: 128,
            experts_per_token: 4,
            moe_intermediate: 704,
            ..Gemma4Geometry::default()
        };
        let dag = build_gemma4_dag(&g, false);
        let at = |layer: u32, kind: Kernel| {
            dag.iter()
                .position(|d| d.layer == Some(layer) && d.kind == kind)
        };
        // Full layer 4: no v projection, and the k-reading V norm runs
        // BEFORE KNorm — it reads the projection KNorm overwrites.
        assert!(at(4, Kernel::QmvV).is_none());
        let v = at(4, Kernel::G4VNormFromK).expect("the k-reading norm");
        let k = at(4, Kernel::KNorm).expect("the k norm");
        assert!(v < k, "V must read k_proj before KNorm rewrites it");
        // Sliding layer 3 keeps the dense arrangement.
        assert!(at(3, Kernel::QmvV).is_some());
        assert!(at(3, Kernel::G4VNormFromK).is_none());
        // The mixture rides BESIDE the dense MLP: both exist on layer 0.
        assert!(at(0, Kernel::QmvGate).is_some());
        assert!(at(0, Kernel::G4ExpertGate).is_some());
        assert!(at(0, Kernel::G4BranchAdd).is_some());
        // Fourteen extra dispatches per mixture layer.
        let dense = Gemma4Geometry::default();
        let base = build_gemma4_dag(&dense, false);
        // The 26B shape also drops one v-projection+swap per full layer:
        // same count either way there (VNormFromK replaces QmvV+VNorm
        // minus one).
        let full_owning = (0..g.n_layers)
            .filter(|&l| g.is_full_attn(l) && !g.is_kv_shared(l))
            .count();
        assert_eq!(
            dag.len(),
            base.len() + 35 * 14 - full_owning,
            "the mixture adds 14; each owning full layer loses its v projection"
        );
        crate::batch::build_scratch_schedule(&dag, false).expect("the 26B DAG colours");
    }
}

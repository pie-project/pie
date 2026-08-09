//! The llama families' per-token dispatch DAG, in the shared [`Kernel`]
//! vocabulary.
//!
//! A pure function, exactly like the qwen and gpt-oss builders. Almost
//! every kind here IS an existing kind — this family's whole point is
//! that the driver already had the pieces — but the DAG cannot be the
//! shared builder under a flag: that walk is qwen3.5's, whose q
//! projection is the 2×-wide `[query | gate]` that `QSplit` then halves
//! and `AttnGate` consumes. llama's q is plain, its gate nonexistent,
//! and its QK-norm OPTIONAL where qwen's is structural. A builder that
//! tried to be both behind options would carry every difference as a
//! branch the other family must dodge; two builders each state one
//! shape, over one set of launch helpers that cannot drift.
//!
//! What is deliberately absent at M=1: the C++ `RowGather` — the same
//! argument as gpt-oss's builder — and any fused residual, which the
//! C++ llama path never had.

use crate::tuning::Tuning;

use super::abi::Kernel;
use super::dispatch::{
    Dispatch, Launch, embed, kv_append, qmv, residual, rms, rope, route_rows, route_sort,
    routed_qmv, router_topk, sdpa, silu_mul,
};
use super::dispatch_mb::{elementwise_mb, qmm_bm, qmm_bn, qmm_bn_unsplit, qmm_t, qmv_mb, rms_mb};
use super::llama::{LlamaGeometry, llama_qmv_kn};
use super::sizing::{pool_colour_elems, sorted_rows};

/// Emit the ordered per-token DAG for `g`.
///
/// Within a stage the order clusters independent dispatches (q/k/v
/// together, then the norms, then the ropes) — hazard-neutral, and what
/// lets a concurrency group form.
#[must_use]
pub fn build_llama_dag(g: &LlamaGeometry, tuning: &Tuning, with_argmax: bool) -> Vec<Dispatch> {
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
    // The sort runs at M=1 too, where the tile the tuning answers for
    // one row's pairs is 1 — a grouping with no padding. The same
    // deliberate choice the shared builder records: one routed dataflow
    // shape, not a decode shape and a prefill shape kept agreeing.
    let tile = tuning.moe_tile_rows(g.experts_per_token, g.n_experts);
    let sorted = u32::try_from(sorted_rows(g.experts_per_token, g.n_experts, tile))
        .expect("an M=1 sort is small");

    // Tied checkpoints read `shared_embedding`, untied ones their own
    // `embed_tokens` — a kind is a weight name, so they are two kinds.
    emit(
        &mut dag,
        if g.tied_embeddings {
            Kernel::EmbedGather
        } else {
            Kernel::EmbedUntied
        },
        None,
        embed(g.hidden),
    );

    for layer in 0..g.n_layers {
        let at = Some(layer);
        emit(&mut dag, Kernel::Rms, at, rms(g.hidden, 1));
        emit(&mut dag, Kernel::QmvQ, at, qmv(g.q_width()));
        emit(&mut dag, Kernel::QmvK, at, qmv(g.kv_width()));
        emit(&mut dag, Kernel::QmvV, at, qmv(g.kv_width()));
        // Qwen3 only: per-head RMS over head_dim, before the rotation.
        // Not emitting the pair on a checkpoint that has one would be a
        // wrong model that still produces fluent text — but that is the
        // geometry's refusal to make, not this builder's: `qk_norm` came
        // from whether the checkpoint ships `self_attn.q_norm`.
        if g.qk_norm {
            emit(&mut dag, Kernel::QNorm, at, rms(g.head_dim, g.n_q_heads));
            emit(&mut dag, Kernel::KNorm, at, rms(g.head_dim, g.n_kv_heads));
        }
        emit(
            &mut dag,
            Kernel::Rope,
            at,
            rope(g.rotary_dims(), g.n_q_heads),
        );
        emit(
            &mut dag,
            Kernel::RopeK,
            at,
            rope(g.rotary_dims(), g.n_kv_heads),
        );
        emit(
            &mut dag,
            Kernel::KvAppend,
            at,
            kv_append(g.head_dim, g.n_kv_heads),
        );
        emit(&mut dag, Kernel::Sdpa, at, sdpa(g.n_q_heads));
        emit(&mut dag, Kernel::QmvO, at, qmv(g.hidden));
        emit(&mut dag, Kernel::Residual, at, residual(g.hidden));

        emit(&mut dag, Kernel::FfnRms, at, rms(g.hidden, 1));
        if g.is_moe() {
            // The same nine dispatches the shared builder emits for
            // qwen3.6's mixture, minus the shared expert this family
            // does not have. `mlp.gate` is `[n_experts, hidden]` — the
            // same shape as a narrow attention projection, so the router
            // is an ordinary matvec rather than a new kernel.
            emit(&mut dag, Kernel::LlRouter, at, qmv(g.n_experts));
            emit(&mut dag, Kernel::GoRouterTopK, at, router_topk(g.n_experts));
            emit(&mut dag, Kernel::LlMoeSort, at, route_sort(g.n_experts));
            emit(
                &mut dag,
                Kernel::LlMoeGather,
                at,
                route_rows(g.hidden, sorted),
            );
            emit(
                &mut dag,
                Kernel::LlExpertGate,
                at,
                routed_qmv(g.moe_intermediate, 1, sorted),
            );
            emit(
                &mut dag,
                Kernel::LlExpertUp,
                at,
                routed_qmv(g.moe_intermediate, 1, sorted),
            );
            emit(
                &mut dag,
                Kernel::LlExpertSiluMul,
                at,
                route_rows(g.moe_intermediate, sorted),
            );
            emit(
                &mut dag,
                Kernel::LlExpertDown,
                at,
                routed_qmv(g.hidden, 1, sorted),
            );
            emit(&mut dag, Kernel::LlMoeCombine, at, route_rows(g.hidden, 1));
        } else {
            emit(&mut dag, Kernel::QmvGate, at, qmv(g.intermediate));
            emit(&mut dag, Kernel::QmvUp, at, qmv(g.intermediate));
            emit(&mut dag, Kernel::SiluMul, at, silu_mul(g.intermediate));
            emit(&mut dag, Kernel::QmvDown, at, qmv(g.hidden));
        }
        emit(&mut dag, Kernel::LayerOut, at, residual(g.hidden));
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
pub struct LlamaDagStats {
    /// Every dispatch.
    pub total: usize,
    /// Decoder layers.
    pub layers: u32,
    /// The matvecs — the step's bandwidth, and what any performance
    /// claim about this family is really about.
    pub gemv: usize,
    /// Of those, the ones whose weights the router picks.
    pub routed: usize,
}

/// Count the DAG's costs.
#[must_use]
pub fn llama_dag_stats(dag: &[Dispatch], g: &LlamaGeometry) -> LlamaDagStats {
    let mut s = LlamaDagStats {
        total: dag.len(),
        layers: g.n_layers,
        ..LlamaDagStats::default()
    };
    for d in dag {
        match d.kind {
            Kernel::QmvQ
            | Kernel::QmvK
            | Kernel::QmvV
            | Kernel::QmvO
            | Kernel::QmvGate
            | Kernel::QmvUp
            | Kernel::QmvDown
            | Kernel::LlRouter
            | Kernel::QmvLmHead
            | Kernel::LmHeadUntied => s.gemv += 1,
            Kernel::LlExpertGate | Kernel::LlExpertUp | Kernel::LlExpertDown => {
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
// The C++ keeps ONE launch switch for both fires — "every *_mb_dispatch
// is its M=1 counterpart with a row count folded in, and at rows == 1
// they agree element for element; a second switch over the same thirty
// kinds would be a second place for the three shape bugs the numerics
// test just found to grow back" — and the port keeps the stronger form
// the gpt-oss arc established: the tiling questions are answered ONCE,
// here, and ride the Dispatch.
//
// Deferred with the shared family's rungs, ledgered: split-K (so every
// dispatch is unsplit — which by the C++'s own measurement makes
// `qmm_bn_unsplit` the RIGHT width: the widest-tile rule is correct only
// where a split supplies the threadgroups a wide tile gives up), the
// FP16 staging pair, and the tiled paged attention.

/// Whether this checkpoint's GEMM reaches the FP16 matrix path — the
/// crossover question; the staging itself is deferred.
#[must_use]
pub fn llama_fp16_format(g: &LlamaGeometry, tuning: &Tuning) -> bool {
    tuning.fp16_gemm_format(g.quant.bits, g.quant.group)
}

/// This family's GEMM crossover: dense or routed by what the FFN is.
#[must_use]
pub fn llama_qmm_min_batch(g: &LlamaGeometry, tuning: &Tuning) -> u32 {
    tuning.qmm_min_batch_for(g.is_moe(), llama_fp16_format(g, tuning))
}

/// The dense GEMM's row block. A 64-request decode fleet has enough
/// independent rows that BM=32's extra threadgroups beat BM=64's weight
/// reuse; a 64-token prefill is one request and keeps BM=64 — row count
/// alone cannot distinguish the two, which is why `requests` exists.
#[must_use]
pub fn llama_dense_qmm_bm(rows: u32, requests: u32) -> u32 {
    if rows == 64 && requests >= 64 {
        return 32;
    }
    qmm_bm(rows)
}

/// How many rows a dense projection's GEMM runs over: whole row blocks,
/// padded unconditionally past the crossover — the pool is pre-padded to
/// the widest tile ([`llama_qmm_pool_rows`]), as with gpt-oss.
#[must_use]
pub fn llama_qmm_rows(g: &LlamaGeometry, tuning: &Tuning, rows: u32, requests: u32) -> u32 {
    let n = rows.max(1);
    if n < llama_qmm_min_batch(g, tuning) {
        return n;
    }
    let bm = llama_dense_qmm_bm(n, requests);
    n.div_ceil(bm) * bm
}

/// The activation pool's row count for a fire of up to `max_rows`.
#[must_use]
pub fn llama_qmm_pool_rows(max_rows: u32) -> u32 {
    let wide = super::psos_mb::QMM_BMS[super::psos_mb::QMM_BMS.len() - 1];
    max_rows.max(1).div_ceil(wide) * wide
}

/// The dense projections a batched GEMM can serve. Unlike gpt-oss, whose
/// FFN is always a mixture, a dense llama's gate/up/down are ordinary
/// projections and the LARGEST matrices in the layer — excluding them
/// would leave most of a dense prefill running as a matvec. The router
/// is deliberately absent: its N is the expert count, tens of columns
/// against a hidden of thousands, so a tile is mostly padding — and it
/// is the one projection whose output every later dispatch waits on.
#[must_use]
pub fn llama_is_dense_proj(kind: Kernel) -> bool {
    matches!(
        kind,
        Kernel::QmvQ
            | Kernel::QmvK
            | Kernel::QmvV
            | Kernel::QmvO
            | Kernel::QmvGate
            | Kernel::QmvUp
            | Kernel::QmvDown
            | Kernel::QmvLmHead
            | Kernel::LmHeadUntied
    )
}

/// A dense projection's column tile for `rows` rows, or 0 for the
/// matvec.
///
/// Existence is the shared `qmm_bn` question; the WIDTH is the unsplit
/// rule, because with split-K deferred no dispatch here has the
/// threadgroup supply the widest-tile rule assumes. That is not a
/// compromise — it is the C++'s own measurement (BN=32 beats 16 and 64
/// on every dense projection of the very checkpoint the smoke runs) made
/// unconditional. The head keeps its pinned 32: `llama_qmm_bn` pins it
/// in the C++ and the vocabulary is wide enough that both rules agree.
#[must_use]
pub fn llama_qmm_bn(
    kind: Kernel,
    g: &LlamaGeometry,
    tuning: &Tuning,
    rows: u32,
    requests: u32,
) -> u32 {
    if !llama_is_dense_proj(kind) {
        return 0;
    }
    let n = llama_qmv_kn(kind, g).n;
    if n == 0 {
        return 0;
    }
    let min_batch = llama_qmm_min_batch(g, tuning);
    let padded = llama_qmm_rows(g, tuning, rows, requests);
    let exists = qmm_bn(n, padded, min_batch);
    if exists == 0 {
        return 0;
    }
    if matches!(kind, Kernel::QmvLmHead | Kernel::LmHeadUntied) {
        return 32;
    }
    let unsplit = qmm_bn_unsplit(n, padded, min_batch, tuning.qmm_bn_crossover_tg);
    if unsplit > 0 { unsplit } else { exists }
}

/// The mixture's sorted-stack extent for an `rows`-token fire, tile and
/// padding from the SAME tuning answer the sort and the GEMM read.
#[must_use]
pub fn llama_moe_sorted_rows(g: &LlamaGeometry, tuning: &Tuning, rows: u32) -> u32 {
    let pairs = rows.max(1).saturating_mul(g.experts_per_token.max(1));
    let tile = tuning.moe_tile_rows(pairs, g.n_experts);
    u32::try_from(sorted_rows(pairs, g.n_experts, tile)).expect("a sort is bounded")
}

/// The routed GEMM's column tile, or 0 when the batch stays a matvec.
#[must_use]
pub fn llama_moe_qmm_bn(kind: Kernel, g: &LlamaGeometry, tuning: &Tuning, rows: u32) -> u32 {
    if !matches!(
        kind,
        Kernel::LlExpertGate | Kernel::LlExpertUp | Kernel::LlExpertDown
    ) {
        return 0;
    }
    let pairs = rows.max(1).saturating_mul(g.experts_per_token.max(1));
    if tuning.moe_tile_rows(pairs, g.n_experts) <= 1 {
        return 0;
    }
    let n = llama_qmv_kn(kind, g).n;
    if n == 0 {
        return 0;
    }
    qmm_bn(
        n,
        llama_moe_sorted_rows(g, tuning, rows),
        tuning.qmm_min_batch_for(true, llama_fp16_format(g, tuning)),
    )
}

/// The M>1 DAG: the M=1 order with the paged kinds, the sampled-row
/// compaction restored to the tail, and every tiling decision made here,
/// once, carried on the dispatch — the shape the gpt-oss arc closed.
#[must_use]
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn build_llama_dag_mb(
    g: &LlamaGeometry,
    tuning: &Tuning,
    n_tokens: u32,
    head_rows: u32,
    requests: u32,
    ordinal_base: u32,
    with_argmax: bool,
) -> Vec<Dispatch> {
    assert!(n_tokens > 0, "a multibatch DAG carries at least one token");
    let n = n_tokens;
    let s = if head_rows == 0 { n } else { head_rows.min(n) };
    let sorted = llama_moe_sorted_rows(g, tuning, n);

    let mut dag: Vec<Dispatch> = Vec::new();
    for base in build_llama_dag(g, tuning, with_argmax) {
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
            Kernel::Sdpa => Kernel::SdpaPaged,
            other => other,
        };
        let m = if matches!(d.kind, Kernel::QmvLmHead | Kernel::LmHeadUntied) {
            s
        } else {
            n
        };
        if let bn @ 1.. = llama_moe_qmm_bn(d.kind, g, tuning, n) {
            let pairs = n.saturating_mul(g.experts_per_token.max(1));
            d.qmm_bn = bn;
            d.qmm_bm = tuning.moe_tile_rows(pairs, g.n_experts);
            d.launch = qmm_t(llama_qmv_kn(d.kind, g).n, sorted, bn, d.qmm_bm);
        } else if let bn @ 1.. = llama_qmm_bn(d.kind, g, tuning, m, requests) {
            let rows = llama_qmm_rows(g, tuning, m, requests);
            d.qmm_bn = bn;
            d.qmm_bm = llama_dense_qmm_bm(rows, requests);
            d.launch = qmm_t(llama_qmv_kn(d.kind, g).n, rows, bn, d.qmm_bm);
        } else {
            d.launch = match d.kind {
                Kernel::EmbedGather | Kernel::EmbedUntied => Launch {
                    grid: [g.hidden, n, 1],
                    tg: [256, 1, 1],
                },
                Kernel::Rms | Kernel::FfnRms => rms_mb(g.hidden, 1, n),
                Kernel::QNorm => rms_mb(g.head_dim, g.n_q_heads, n),
                Kernel::KNorm => rms_mb(g.head_dim, g.n_kv_heads, n),
                Kernel::FinalRms => rms_mb(g.hidden, 1, s),
                Kernel::Rope => Launch {
                    grid: [g.rotary_dims() / 2, g.n_q_heads, n],
                    tg: [g.rotary_dims() / 2, 1, 1],
                },
                Kernel::RopeK => Launch {
                    grid: [g.rotary_dims() / 2, g.n_kv_heads, n],
                    tg: [g.rotary_dims() / 2, 1, 1],
                },
                Kernel::KvAppendPaged => Launch {
                    grid: [g.head_dim, g.n_kv_heads, n],
                    tg: [g.head_dim, 1, 1],
                },
                // The scalar paged attention at every fleet width; the
                // tiled shape is deferred with the shared family's.
                Kernel::SdpaPaged => Launch {
                    grid: [g.n_q_heads * 1024, n, 1],
                    tg: [1024, 1, 1],
                },
                Kernel::Residual | Kernel::LayerOut => elementwise_mb(g.hidden, n),
                Kernel::QmvGate | Kernel::QmvUp => qmv_mb(g.intermediate, n),
                Kernel::SiluMul => elementwise_mb(g.intermediate, n),
                Kernel::QmvDown => qmv_mb(g.hidden, n),
                Kernel::QmvQ => qmv_mb(g.q_width(), n),
                Kernel::QmvK | Kernel::QmvV => qmv_mb(g.kv_width(), n),
                Kernel::QmvO => qmv_mb(g.hidden, n),
                Kernel::LlRouter => qmv_mb(g.n_experts, n),
                Kernel::QmvLmHead | Kernel::LmHeadUntied => qmv_mb(g.vocab, s),
                Kernel::GoRouterTopK => {
                    let w = super::dispatch::router_lane_width(g.n_experts);
                    Launch {
                        grid: [w, n, 1],
                        tg: [w, 1, 1],
                    }
                }
                Kernel::LlMoeSort => route_sort(g.n_experts),
                Kernel::LlMoeGather => route_rows(g.hidden, sorted),
                Kernel::LlExpertGate | Kernel::LlExpertUp => {
                    routed_qmv(g.moe_intermediate, 1, sorted)
                }
                Kernel::LlExpertSiluMul => route_rows(g.moe_intermediate, sorted),
                Kernel::LlExpertDown => routed_qmv(g.hidden, 1, sorted),
                Kernel::LlMoeCombine => route_rows(g.hidden, n),
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

/// One value's shape, off its WRITER — the per-family half of
/// [`pool_colour_elems`](super::pool_colour_elems). The two int32 traps
/// are the C++'s, kept verbatim: the router's ids are `int32` — TWO of
/// this pool's two-byte elements per entry — and one kind produces both
/// ids and weights, so the claim is the wider and the weights ride along
/// at twice their need (k extra elements per routed layer). The sort's
/// four index outputs are sized as the tallest of them for the same
/// reason.
#[must_use]
pub fn llama_value_extent(d: &Dispatch, g: &LlamaGeometry) -> super::sizing::ValueExtent {
    use super::sizing::{RowAxis, ValueExtent};
    let e = |elems: u32, axis: RowAxis| ValueExtent { elems, axis };
    match d.kind {
        Kernel::EmbedGather
        | Kernel::EmbedUntied
        | Kernel::Rms
        | Kernel::FfnRms
        | Kernel::Residual
        | Kernel::LayerOut
        | Kernel::QmvO
        | Kernel::QmvDown
        | Kernel::LlMoeCombine => e(g.hidden, RowAxis::Body),
        Kernel::FinalRms | Kernel::G4RowGather => e(g.hidden, RowAxis::Tail),
        Kernel::QmvQ | Kernel::QNorm | Kernel::Rope | Kernel::Sdpa | Kernel::SdpaPaged => {
            e(g.q_width(), RowAxis::Body)
        }
        Kernel::QmvK | Kernel::QmvV | Kernel::KNorm | Kernel::RopeK => {
            e(g.kv_width(), RowAxis::Body)
        }
        Kernel::QmvGate | Kernel::QmvUp | Kernel::SiluMul => e(g.intermediate, RowAxis::Body),
        Kernel::LlRouter => e(g.n_experts, RowAxis::Body),
        Kernel::GoRouterTopK => e(g.experts_per_token * 2, RowAxis::Body),
        Kernel::LlMoeSort => e(2, RowAxis::Sorted),
        Kernel::LlMoeGather => e(g.hidden, RowAxis::Sorted),
        Kernel::LlExpertGate | Kernel::LlExpertUp | Kernel::LlExpertSiluMul => {
            e(g.moe_intermediate, RowAxis::Sorted)
        }
        Kernel::LlExpertDown => e(g.hidden, RowAxis::Sorted),
        _ => e(0, RowAxis::Body),
    }
}

/// Each pool colour's element count for a `rows`-token fire — the number
/// [`extra-heap budgeting`](super::pool_colour_elems) and the engine's
/// staging must agree on, produced by ONE composition of the same pure
/// pieces the fire itself uses.
#[must_use]
pub fn llama_pool_elems(g: &LlamaGeometry, tuning: &Tuning, rows: u32, head_rows: u32) -> Vec<u64> {
    let dag = build_llama_dag_mb(g, tuning, rows.max(1), head_rows, 1, 0, false);
    let (uses, values) = super::dataflow::build_scratch_uses(&dag);
    let ends = super::dispatch::concurrent_run_ends(&dag);
    let coloring = super::color::color_live_ranges(&uses, &ends, values, false)
        .expect("the llama DAG colours");
    pool_colour_elems(
        &dag,
        &uses,
        &coloring,
        |d| llama_value_extent(d, g),
        rows,
        head_rows,
        llama_moe_sorted_rows(g, tuning, rows),
        u64::from(rows.max(1)) * u64::from(g.hidden),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch::build_scratch_schedule;

    #[test]
    fn the_8b_step_is_sixteen_dispatches_a_layer_plus_the_ends() {
        let g = LlamaGeometry::default();
        let dag = build_llama_dag(&g, &Tuning::default(), true);
        // 1 embed + 32 × 16 + final norm + head + argmax.
        assert_eq!(dag.len(), 1 + 32 * 16 + 3);
        assert!(dag.iter().enumerate().all(|(i, d)| d.ordinal as usize == i));
        // llama has no q|gate fusion and no attention gate — the qwen
        // kinds must not leak in.
        assert!(dag.iter().all(|d| d.kind != Kernel::QSplit
            && d.kind != Kernel::AttnGate
            && d.kind != Kernel::QNorm));
        // Untied: the 8B ships both matrices.
        assert!(dag.iter().any(|d| d.kind == Kernel::LmHeadUntied));
        assert!(dag.iter().any(|d| d.kind == Kernel::EmbedUntied));
        let q = dag.iter().find(|d| d.kind == Kernel::QmvQ).unwrap();
        assert_eq!(
            q.launch.tg,
            [32, 2, 1],
            "the plain q projection is a matvec at its own width"
        );
        let stats = llama_dag_stats(&dag, &g);
        assert_eq!(stats.gemv, 32 * 7 + 1, "seven a layer plus the head");
        assert_eq!(stats.routed, 0);
        build_scratch_schedule(&dag, false).expect("the dense DAG colours");
    }

    #[test]
    fn qwen3_gets_the_norm_pair_between_v_and_the_rotation() {
        let g = LlamaGeometry {
            qk_norm: true,
            tied_embeddings: true,
            ..LlamaGeometry::default()
        };
        let dag = build_llama_dag(&g, &Tuning::default(), false);
        assert_eq!(dag.len(), 1 + 32 * 18 + 2);
        let v = dag.iter().position(|d| d.kind == Kernel::QmvV).unwrap();
        assert_eq!(dag[v + 1].kind, Kernel::QNorm);
        assert_eq!(dag[v + 2].kind, Kernel::KNorm);
        assert_eq!(dag[v + 3].kind, Kernel::Rope);
        // Tied: one table serves both ends.
        assert!(dag.iter().any(|d| d.kind == Kernel::EmbedGather));
        assert!(dag.iter().any(|d| d.kind == Kernel::QmvLmHead));
        build_scratch_schedule(&dag, false).expect("the qk-norm DAG colours");
    }

    #[test]
    fn the_mb_dag_decides_its_tiles_once_and_compacts_the_sampled_rows() {
        let g = LlamaGeometry::default();
        let tuning = Tuning::default();
        let dag = build_llama_dag_mb(&g, &tuning, 16, 2, 1, 0, false);
        // The paged kinds replace the ring kinds; the compaction joins
        // the tail at the SAMPLED size.
        assert!(dag.iter().any(|d| d.kind == Kernel::KvAppendPaged));
        assert!(dag.iter().any(|d| d.kind == Kernel::SdpaPaged));
        assert!(
            dag.iter()
                .all(|d| d.kind != Kernel::KvAppend && d.kind != Kernel::Sdpa)
        );
        let gather = dag
            .iter()
            .position(|d| d.kind == Kernel::G4RowGather)
            .expect("the tail compacts");
        assert_eq!(dag[gather + 1].kind, Kernel::FinalRms);
        assert_eq!(dag[gather].launch.grid, [g.hidden, 2, 1]);
        // Sixteen rows tile every dense projection at the UNSPLIT width:
        // with split-K deferred, no dispatch has the threadgroup supply
        // the widest-tile rule assumes. The 8B's q is 4096 wide — 256
        // narrow threadgroups, past the crossover, the 32 tile; its kv
        // is 1024 wide — 64 threadgroups, under it, the 16 tile.
        let q = dag.iter().find(|d| d.kind == Kernel::QmvQ).unwrap();
        assert_eq!((q.qmm_bn, q.qmm_bm), (32, 16));
        assert_eq!(q.launch.grid, [32 * (g.q_width() / 32), 2, 2]);
        let k = dag.iter().find(|d| d.kind == Kernel::QmvK).unwrap();
        assert_eq!(k.qmm_bn, 16);
        let gate = dag.iter().find(|d| d.kind == Kernel::QmvGate).unwrap();
        assert_eq!(gate.qmm_bn, 32, "14336 columns stays past the crossover");
        // The head runs on the two sampled rows — below the crossover,
        // so it keeps the matvec.
        let head = dag.iter().find(|d| d.kind == Kernel::LmHeadUntied).unwrap();
        assert_eq!(head.qmm_bn, 0);
        assert_eq!(head.launch.grid[0], 32 * 2);
        crate::batch::build_scratch_schedule(&dag, false).expect("the MB DAG colours");
        // The 64-request fleet takes BM=32 where a 64-row prefill keeps
        // 64 — row count alone cannot distinguish the two.
        assert_eq!(llama_dense_qmm_bm(64, 64), 32);
        assert_eq!(llama_dense_qmm_bm(64, 1), 64);
    }

    #[test]
    fn a_colour_is_sized_by_its_widest_value_not_its_kind() {
        let tuning = Tuning::default();
        // Dense 8B at 16 rows: every colour holds at least one 16-row
        // value, the widest holds the 14336-wide FFN pair.
        let g = LlamaGeometry::default();
        let elems = llama_pool_elems(&g, &tuning, 16, 0);
        assert!(!elems.is_empty());
        assert_eq!(
            elems.iter().max().copied().unwrap(),
            u64::from(16u32) * u64::from(g.intermediate),
            "the FFN width dominates a dense fire"
        );
        // Routed: the sorted stack is TALLER than rows×k (128 pairs pad
        // to 2048 at the 16 tile), and the colour holding the gathered
        // stack must be sized for it — sizing by kind would hand the
        // last expert's projection a buffer 16× short.
        let moe = LlamaGeometry {
            n_experts: 128,
            experts_per_token: 8,
            moe_intermediate: 768,
            ..LlamaGeometry::default()
        };
        let sorted = llama_moe_sorted_rows(&moe, &tuning, 16);
        assert_eq!(sorted, 2048);
        let elems = llama_pool_elems(&moe, &tuning, 16, 0);
        assert!(
            elems
                .iter()
                .any(|&e| e >= u64::from(sorted) * u64::from(moe.hidden)),
            "some colour holds the hidden-wide gathered stack at the SORTED height"
        );
        // The tail scales on the SAMPLED rows: at head_rows=1 no colour
        // needs 16 vocab-rows… the head writes IO, but the gather's
        // output is [S, hidden] — its colour may still be dominated by
        // body values sharing it, so pin the SHRINK instead: sampling
        // fewer rows never grows any colour.
        let all = llama_pool_elems(&g, &tuning, 16, 0);
        let one = llama_pool_elems(&g, &tuning, 16, 1);
        assert_eq!(all.len(), one.len());
        assert!(one.iter().zip(&all).all(|(a, b)| a <= b));
    }

    #[test]
    fn the_routed_mixture_tiles_when_the_pairs_pay_for_it() {
        let g = LlamaGeometry {
            qk_norm: true,
            n_experts: 128,
            experts_per_token: 8,
            moe_intermediate: 768,
            ..LlamaGeometry::default()
        };
        let tuning = Tuning::default();
        // Sixteen tokens route 128 pairs over 128 experts: one row an
        // expert pays for the 16 tile, and the sort pads to it.
        let sorted = llama_moe_sorted_rows(&g, &tuning, 16);
        assert_eq!(sorted, 2048, "128 pairs, every expert padded to 16 rows");
        let dag = build_llama_dag_mb(&g, &tuning, 16, 0, 1, 0, false);
        let gate = dag.iter().find(|d| d.kind == Kernel::LlExpertGate).unwrap();
        assert_eq!((gate.qmm_bn, gate.qmm_bm), (64, 16));
        assert_eq!(gate.launch.grid, [32 * (768 / 64), 2 * (sorted / 16), 2]);
        // Two tokens do not: the mixture stays a matvec over the pure
        // grouping.
        let two = build_llama_dag_mb(&g, &tuning, 2, 0, 1, 0, false);
        let gate = two.iter().find(|d| d.kind == Kernel::LlExpertGate).unwrap();
        assert_eq!(gate.qmm_bn, 0);
        crate::batch::build_scratch_schedule(&dag, false).expect("the routed MB DAG colours");
    }

    #[test]
    fn the_mixture_swaps_four_dense_dispatches_for_the_nine_routed_ones() {
        let g = LlamaGeometry {
            qk_norm: true,
            n_experts: 128,
            experts_per_token: 8,
            moe_intermediate: 768,
            ..LlamaGeometry::default()
        };
        let dag = build_llama_dag(&g, &Tuning::default(), true);
        assert_eq!(dag.len(), 1 + 32 * (18 - 4 + 9) + 3);
        let stats = llama_dag_stats(&dag, &g);
        assert_eq!(stats.routed, 32 * 3);
        assert_eq!(stats.gemv, 32 * 8 + 1, "q k v o, router, three routed");
        // Eight sorted rows at decode: top-8, tile 1 — a grouping with
        // no padding.
        let gate = dag.iter().find(|d| d.kind == Kernel::LlExpertGate).unwrap();
        assert_eq!(gate.launch.grid[0], 32 * 8);
        // No shared expert in this family.
        assert!(dag.iter().all(|d| d.kind != Kernel::LlSharedGate));
        build_scratch_schedule(&dag, false).expect("the routed DAG colours");
    }
}

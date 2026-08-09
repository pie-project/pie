//! The M>1 decode DAG: the M=1 walk with paged/slotted kinds and batched
//! launch geometry.
//!
//! [`build_decode_dag_mb`] reuses `build_decode_dag` — the DAG's ORDER is
//! the family's and does not change with the batch — then swaps each kind
//! for its multibatch form ([`mb_kind`]) and rewrites the launch
//! ([`mb_geometry`]). The decisions the C++ kept in `decode_dispatch_mb.hpp`
//! (which GEMM tile, whether a projection batches at all) read [`Tuning`]
//! and live here; the SHAPES are the kernels' own.
//!
//! Two carried findings, condensed from the C++'s longer records:
//!
//! * **No split-K.** The split GEMM writes partial slices a reduce pass
//!   must sum, and nothing in this driver ever dispatched that pass — the
//!   projection's real output kept whatever the last fire left in it. It
//!   survived because the split engages only past `qmm_min_batch`, which
//!   nothing fired until the throughput harness did. 717 tok/s became 520
//!   with it off, and 717 was the speed of computing the wrong answer.
//! * **The routed decode GEMM is SHUT** ([`ROUTED_DECODE_BATCHED`]). At
//!   exactly the fleet width where `n_pairs >= n_experts` first holds, a
//!   32-lane fleet answered `220 0 0 0 0 0` against mlx-lm's
//!   `220 24 11 220 16 15`, members disagreeing among themselves — rows not
//!   written, not rounding. The bisect record lives in the C++; the arm
//!   stays shut until it is reproduced and fixed on a machine big enough to
//!   hold the tripled scratch, and shutting it costs only the batched form
//!   above 32 lanes, where the matvec is the fallback. The PREFILL's routed
//!   batching is a different call site and stays on.

use crate::tuning::Tuning;

use super::abi::Kernel;
use super::consts::qmv_kn;
use super::dispatch::{DagOptions, Dispatch, Launch, build_decode_dag};
use super::geometry::DecodeGeometry;
use super::psos_mb::QMM_BMS;
pub use crate::model::grid::{elementwise_mb, qmm_bm, qmm_t, qmv_mb, rms_mb};
use super::sizing::{RoutedProjection, moe_sorted_rows};

/// The routed decode GEMM's arm; see the module docs for why it is shut.
pub const ROUTED_DECODE_BATCHED: bool = false;

/// One threadgroup covers this many query rows in `sdpa_paged_tiled`.
pub const SDPA_QUERY_TILE: u32 = 32;

/// Where prefill DAG ordinals start, and how far apart per-token tables
/// sit. Disjoint from the M=1 range by construction.
pub const PREFILL_ORDINAL_BASE: u32 = 1 << 20;
/// See [`PREFILL_ORDINAL_BASE`].
pub const PREFILL_ORDINAL_STRIDE: u32 = 2048;

/// The multibatch form of each kind: paged attention, slotted GDN. The
/// rest keep their names — an `_mb` pipeline choice is the PSO table's
/// business, not the kind's.
#[must_use]
pub fn mb_kind(kind: Kernel) -> Kernel {
    match kind {
        Kernel::GdnPrep => Kernel::GdnPrepSlotted,
        Kernel::GdnCore => Kernel::GdnCoreSlotted,
        Kernel::KvAppend => Kernel::KvAppendPaged,
        Kernel::Sdpa => Kernel::SdpaPaged,
        other => other,
    }
}

/// A projection's output width for the PER-TOKEN batched path, or zero.
///
/// The three expert projections are deliberately absent: they run over the
/// SORTED rows, and answering here would launch them over the token count —
/// computing the first `n` sorted rows and leaving the rest of the stack
/// holding the previous layer's output. The router and the shared expert
/// are dense in every sense and answer like any projection.
#[must_use]
pub fn qmv_out_size(kind: Kernel, g: &DecodeGeometry) -> u32 {
    match kind {
        Kernel::LlExpertGate | Kernel::LlExpertUp | Kernel::LlExpertDown => 0,
        _ => qmv_kn(kind, g).n,
    }
}


/// Which [`QMM_BMS`] rung a chosen row block is — the dispatch records the
/// block it launched, not the batch it came from, so the PSO lookup comes
/// back the other way.
#[must_use]
pub fn qmm_bm_slot(bm: u32) -> usize {
    QMM_BMS.iter().position(|&b| b == bm).unwrap_or(0)
}

/// The GEMM's column tile when split-K supplies threadgroups: the widest
/// tile that divides the output, full stop — wider is strictly fewer
/// dequantizations of each weight tile. Zero refuses the GEMM (matvec).
#[must_use]
pub fn qmm_bn(out_vec: u32, n: u32, min_batch: u32) -> u32 {
    let bm = qmm_bm(n);
    if n < min_batch || !n.is_multiple_of(bm) {
        return 0;
    }
    let mut best = 0;
    for bn in [16, 32, 64] {
        if out_vec.is_multiple_of(bn) {
            best = bn;
        }
    }
    best
}

/// [`qmm_bn`] for a family whose GEMM has no split-K behind it: the narrow
/// tile until there is enough work to fill the machine, 32 after, and
/// never 64 — that is the finding, not an omission (a decode fire is one
/// row tile, so BN does not change how many times a weight is
/// dequantized; what it changes is how much of `x` each threadgroup
/// re-reads).
#[must_use]
pub fn qmm_bn_unsplit(out_vec: u32, n: u32, min_batch: u32, crossover_tg: u32) -> u32 {
    let bm = qmm_bm(n);
    if n < min_batch || !n.is_multiple_of(bm) || !out_vec.is_multiple_of(16) {
        return 0;
    }
    let row_tiles = n / bm;
    let narrow_tg = (out_vec / 16) * row_tiles;
    if narrow_tg <= crossover_tg || !out_vec.is_multiple_of(32) {
        16
    } else {
        32
    }
}

/// The padded row count a batched projection launches over: whole row
/// tiles, refused back to the true count when padding would talk the
/// dispatch past the measured crossover (a 2-row fire padded to 16 would
/// launch eight times the arithmetic it needs) or past the pool's depth
/// (a wider write runs into the next activation's slot).
#[must_use]
pub fn qmm_mb_rows(n: u32, max_tokens: u32, min_batch: u32) -> u32 {
    let rows = n.max(1);
    if rows < min_batch {
        return rows;
    }
    let bm = qmm_bm(rows);
    let padded = rows.div_ceil(bm) * bm;
    if padded <= max_tokens.max(1) {
        padded
    } else {
        rows
    }
}





/// Whether this checkpoint's GEMM reaches the FP16 matrix path.
#[must_use]
pub fn fp16_format(g: &DecodeGeometry, tuning: &Tuning) -> bool {
    tuning.fp16_qmm && g.quant.bits == 4 && g.quant.group == 64
}

/// Whether a kind's weights live in the checkpoint's second affine format
/// (the spared routing projections). Those get one extra matvec pipeline,
/// not a table of every batched shape: measured, putting the router back
/// on the batched path moved a 128-token prefill by less than the
/// run-to-run spread, because a router's share of GPU time is a decode's,
/// where the batched path does not apply.
#[must_use]
pub fn uses_alt_quant(kind: Kernel, g: &DecodeGeometry) -> bool {
    g.has_alt_quant() && matches!(kind, Kernel::LlRouter | Kernel::LlSharedGateProj)
}

/// Rewrite one dispatch's launch (and GEMM fields) for `n` tokens.
#[must_use]
#[allow(clippy::too_many_lines)] // one switch, one rewrite; splitting hides the mapping
pub fn mb_geometry(d: &Dispatch, g: &DecodeGeometry, tuning: &Tuning, n: u32) -> Dispatch {
    let mut d = *d;
    let min_batch = tuning.qmm_min_batch_for(g.is_moe(), fp16_format(g, tuning));
    let out = qmv_out_size(d.kind, g);
    if out != 0 {
        let qn = qmm_mb_rows(n, g.max_tokens, min_batch);
        d.qmm_bn = qmm_bn_unsplit(out, qn, min_batch, tuning.qmm_bn_crossover_tg);
        d.qmm_bm = qmm_bm(qn);
        if uses_alt_quant(d.kind, g) {
            d.qmm_bn = 0;
        }
        // NO split-K; see the module docs.
        d.qmm_split = 1;
        d.launch = if d.qmm_bn > 0 {
            qmm_t(out, qn, d.qmm_bn, d.qmm_bm)
        } else {
            qmv_mb(out, n)
        };
        return d;
    }
    let sorted =
        |run| u32::try_from(moe_sorted_rows(g, tuning, n, run)).expect("a sort is bounded");
    let routed_run = if ROUTED_DECODE_BATCHED {
        RoutedProjection::Matmul
    } else {
        RoutedProjection::Matvec
    };
    d.launch = match d.kind {
        Kernel::EmbedUntied | Kernel::EmbedGather => Launch {
            grid: [g.hidden, n, 1],
            tg: [256, 1, 1],
        },
        Kernel::Rms | Kernel::FfnRms | Kernel::FinalRms => rms_mb(g.hidden, 1, n),
        Kernel::QNorm => rms_mb(g.head_dim, g.n_q_heads, n),
        Kernel::KNorm => rms_mb(g.head_dim, g.n_kv_heads, n),
        Kernel::GdnPrepSlotted => Launch {
            grid: [32, 1, n * g.gdn_v_heads],
            tg: [32, 1, 1],
        },
        Kernel::GdnCoreSlotted => Launch {
            grid: [32, g.gdn_v_dim, n * g.gdn_v_heads],
            tg: [32, 4, 1],
        },
        Kernel::GatedRms => Launch {
            grid: [g.gdn_v_dim, g.gdn_v_heads, n],
            tg: [g.gdn_v_dim, 1, 1],
        },
        Kernel::QSplit => Launch {
            grid: [g.head_dim, g.n_q_heads, n],
            tg: [g.head_dim, 1, 1],
        },
        Kernel::Rope => Launch {
            grid: [g.rotary_dims / 2, g.n_q_heads, n],
            tg: [g.rotary_dims / 2, 1, 1],
        },
        Kernel::RopeK => Launch {
            grid: [g.rotary_dims / 2, g.n_kv_heads, n],
            tg: [g.rotary_dims / 2, 1, 1],
        },
        Kernel::KvAppendPaged => Launch {
            grid: [g.head_dim, g.n_kv_heads, n],
            tg: [g.head_dim, 1, 1],
        },
        Kernel::SdpaPaged => Launch {
            grid: [g.n_q_heads * 1024, n, 1],
            tg: [1024, 1, 1],
        },
        Kernel::AttnGate => elementwise_mb(g.n_q_heads * g.head_dim, n),
        Kernel::Residual | Kernel::LayerOut => elementwise_mb(g.hidden, n),
        // Routed, the dense SwiGLU that remains is the SHARED expert's —
        // one row a token at its own width; the mixture's own is
        // LlExpertSiluMul over the sorted stack, split precisely because
        // one line cannot say both.
        Kernel::SiluMul => elementwise_mb(
            if g.is_moe() {
                g.shared_intermediate
            } else {
                g.intermediate
            },
            n,
        ),
        Kernel::LlExpertSiluMul => elementwise_mb(g.moe_intermediate, sorted(routed_run)),
        // The expert projections run over the sorted rows — neither `n`
        // nor `n * k` — and take the matvec while the batched arm is shut.
        Kernel::LlExpertGate | Kernel::LlExpertUp | Kernel::LlExpertDown => {
            let width = if d.kind == Kernel::LlExpertDown {
                g.hidden
            } else {
                g.moe_intermediate
            };
            let rows = sorted(routed_run);
            Launch {
                grid: [32 * rows.max(1), width.max(1).div_ceil(4), 1],
                tg: [32, 2, 1],
            }
        }
        Kernel::GoRouterTopK => {
            let w = super::dispatch::router_lane_width(g.n_experts);
            Launch {
                grid: [w, n.max(1), 1],
                tg: [w, 1, 1],
            }
        }
        Kernel::LlMoeSort => super::dispatch::route_sort(g.n_experts),
        // The stack this fills is the one the sort writes, so it asks the
        // SAME question: launched over the padded count with an unpadded
        // sort behind it, the gather walks rows the sort never grouped.
        Kernel::LlMoeGather => super::dispatch::route_rows(g.hidden, sorted(routed_run)),
        Kernel::LlMoeCombine => {
            let w = g.hidden.max(1);
            Launch {
                grid: [w, n.max(1), 1],
                tg: [w.min(256), 1, 1],
            }
        }
        // The SAME helper the M=1 DAG uses, not the flat elementwise its
        // neighbours take: the kernel reads its row from gid.y, so a flat
        // grid puts every thread on row 0 and leaves rows 1.. holding
        // whatever the pool did. At N=1 the two shapes coincide, which is
        // why this survived every single-sequence gate and broke a fleet
        // of two at its first step.
        Kernel::LlSharedCombine => super::dispatch::route_rows(g.hidden, n),
        other => panic!("missing multi-batch launch geometry for {other:?}"),
    };
    d
}

/// The M>1 DAG: the M=1 order with multibatch kinds and launches, ordinals
/// offset by `ordinal_base` so per-fire tables stay disjoint.
#[must_use]
pub fn build_decode_dag_mb(
    g: &DecodeGeometry,
    tuning: &Tuning,
    n_tokens: u32,
    ordinal_base: u32,
    options: DagOptions,
) -> Vec<Dispatch> {
    assert!(n_tokens > 0, "a multibatch DAG carries at least one token");
    build_decode_dag(
        g,
        tuning,
        DagOptions {
            with_argmax: false,
            ..options
        },
    )
    .iter()
    .map(|d| {
        let mut d = *d;
        d.kind = mb_kind(d.kind);
        let mut d = mb_geometry(&d, g, tuning, n_tokens);
        d.ordinal += ordinal_base;
        d
    })
    .collect()
}

/// One single-token DAG per prompt row, each on its own ordinal stride —
/// the per-token prefill over a shared scratch pool.
#[must_use]
pub fn build_decode_prefill_dags(
    g: &DecodeGeometry,
    tuning: &Tuning,
    n_tokens: u32,
    options: DagOptions,
) -> Vec<Vec<Dispatch>> {
    assert!(n_tokens > 0, "a prefill carries at least one token");
    (0..n_tokens)
        .map(|t| {
            let dag = build_decode_dag_mb(
                g,
                tuning,
                1,
                PREFILL_ORDINAL_BASE + t * PREFILL_ORDINAL_STRIDE,
                options,
            );
            assert!(
                dag.len() < PREFILL_ORDINAL_STRIDE as usize,
                "a prefill DAG exceeds its ordinal stride"
            );
            dag
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_gemm_rules_hold_their_measured_shape() {
        assert_eq!(qmm_bm(1), 16);
        assert_eq!(qmm_bm(48), 32);
        assert_eq!(qmm_bm(200), 64);
        assert_eq!(qmm_bm_slot(64), 2);
        // Below the crossover the GEMM refuses; on it, whole tiles only.
        assert_eq!(qmm_bn(1024, 4, 8), 0);
        assert_eq!(qmm_bn(1024, 16, 8), 64, "widest tile dividing the output");
        assert_eq!(qmm_bn(1000, 16, 8), 0, "no tile divides 1000");
        // Unsplit: narrow until the machine fills, 32 after, never 64.
        assert_eq!(qmm_bn_unsplit(1024, 16, 8, 160), 16);
        assert_eq!(qmm_bn_unsplit(4096, 64, 8, 160), 32);
        // Padding refuses past the pool and below the crossover.
        assert_eq!(
            qmm_mb_rows(2, 128, 8),
            2,
            "padding cannot talk past the crossover"
        );
        assert_eq!(qmm_mb_rows(17, 128, 8), 32);
        assert_eq!(qmm_mb_rows(17, 24, 8), 17, "the pool is only that deep");
    }

    #[test]
    fn the_mb_dag_is_the_m1_order_in_paged_kinds() {
        let g = DecodeGeometry {
            max_tokens: 64,
            paged_kv_enabled: true,
            ..DecodeGeometry::default()
        };
        let tuning = Tuning::default();
        let dag = build_decode_dag_mb(&g, &tuning, 16, 0, DagOptions::default());
        assert!(dag.iter().any(|d| d.kind == Kernel::SdpaPaged));
        assert!(dag.iter().any(|d| d.kind == Kernel::KvAppendPaged));
        assert!(
            dag.iter()
                .all(|d| d.kind != Kernel::Sdpa && d.kind != Kernel::KvAppend)
        );
        // A 16-row fire batches its projections: some dispatch carries a
        // GEMM tile.
        assert!(dag.iter().any(|d| d.qmm_bn > 0));
        // The prefill stream keeps each token's table on its own stride.
        let prefill = build_decode_prefill_dags(&g, &tuning, 3, DagOptions::default());
        assert_eq!(prefill.len(), 3);
        assert_eq!(
            prefill[1][0].ordinal,
            PREFILL_ORDINAL_BASE + PREFILL_ORDINAL_STRIDE
        );
    }
}

//! What a plan costs on the wafer: a decode step's time ([`time`]) and each PE image's memory
//! ([`memory`]), read off the plan and the program it builds. The search ([`crate::search`])
//! ranks partitions by these; a model is calibrated against what the chip and the linker
//! measured, and says how far off it was there (`ARCHITECTURE.md`, "Cost").

pub mod memory;
pub mod time;

use serde::Serialize;

use crate::plan::{LayerKind, Plan, Shape};

/// The plan's facts a step's time is a function of, per decode step.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Features {
    /// Decoder layers.
    pub layers: f64,
    /// The weight elements one PE multiplies a step, summed over the layers (the busiest PE of
    /// each phase: a phase's matrix over the PEs that hold it; routed experts: the top-k).
    pub matvec: f64,
    /// Reduction hops a step within the groups: each layer's two column and two row reductions,
    /// a two-level chain (`group` members, then `span / group` group roots).
    pub group_hops: f64,
    /// The reductions' hops between the group roots.
    pub root_hops: f64,
    /// The fabric distance a step's collectives cross: each layer's multicasts and reductions
    /// span the unit's rows and its columns.
    pub span: f64,
    /// KV elements one row PE reads a step (keys and values, the attention layers' entries times
    /// a column's KV dims).
    pub kv: f64,
    /// The head's elements one head PE multiplies a step (its vocabulary rows times its share of
    /// the hidden).
    pub head: f64,
}

impl Features {
    /// The features of `plan`.
    #[must_use]
    pub fn of(plan: &Plan) -> Features {
        let s = &plan.shape;
        let p = &plan.partition;
        let (rows, cols) = (f64::from(p.rows), f64::from(p.cols));
        // the attention half and the MLP half on their own columns (gpt-oss), else on the unit's
        let (att_cols, mlp_cols) = match p.attn_cols {
            Some(a) => (f64::from(a), cols - f64::from(a)),
            None => (cols, cols),
        };
        let h = f64::from(s.hidden);
        let mut matvec = 0.0;
        let mut kv = 0.0;
        for (l, kind) in s.layers.iter().enumerate() {
            let (mix, own_cache) = mixer_elements(s, *kind);
            matvec += mix / (rows * att_cols);
            let mlp = match s.moe {
                Some(m) => {
                    f64::from(m.top_k) * 3.0 * h * f64::from(m.expert_intermediate)
                        + h * f64::from(m.experts)
                }
                None => 3.0 * h * f64::from(s.mlp.width(l as u32)),
            };
            matvec += mlp / (rows * mlp_cols);
            if own_cache || matches!(kind, LayerKind::SharedKv { .. }) {
                let (heads, dim) = kv_heads(s, *kind);
                // a column's KV dims: the heads' over the attention columns
                let dims = (heads * dim / att_cols).max(1.0);
                let entries = match (kind, p.kv_entries.sliding, p.kv_entries.global) {
                    (LayerKind::Sliding { .. }, Some(n), _)
                    | (LayerKind::SharedKv { sliding: true, .. }, Some(n), _) => f64::from(n),
                    (_, _, Some(n)) => f64::from(n),
                    _ => f64::from(p.kv_entries.per_row),
                };
                kv += 2.0 * entries * dims;
            }
        }
        let layers = s.layers.len() as f64;
        let span_c = att_cols.min(mlp_cols).max(1.0);
        let (gc, gr) = (
            f64::from(p.reduce.col.max(1)).min(rows),
            f64::from(p.reduce.row.max(1)).min(span_c),
        );
        let group_hops = layers * 2.0 * (gc + gr);
        let span = layers * (rows + span_c);
        let root_hops = layers * 2.0 * ((rows / gc).ceil() + (span_c / gr).ceil());
        let head = f64::from(p.head.vocab_rows) * (h / f64::from(p.head.cols.max(1))).ceil();
        Features {
            layers,
            matvec,
            group_hops,
            root_hops,
            span,
            kv,
            head,
        }
    }
}

/// A layer's mixer: its projections' weight elements, and whether it keeps a cache of its own.
fn mixer_elements(s: &Shape, kind: LayerKind) -> (f64, bool) {
    let h = f64::from(s.hidden);
    match kind {
        LayerKind::Gdn => {
            let g = s.gdn.expect("a GDN layer has a GDN spec");
            let (k, v) = (
                f64::from(g.key_heads * g.key_dim),
                f64::from(g.value_heads * g.value_dim),
            );
            // q, k, v, z in; a, b gates; o out
            (
                h * (2.0 * k + 2.0 * v + 2.0 * f64::from(g.value_heads)) + v * h,
                false,
            )
        }
        _ => {
            let (kvh, dim) = kv_heads(s, kind);
            let q = f64::from(s.attn.q_heads) * head_dim(s, kind);
            let gate = if s.attn.output_gate { 2.0 } else { 1.0 };
            let shared = matches!(kind, LayerKind::SharedKv { .. });
            let kvp = if shared { 0.0 } else { 2.0 * kvh * dim };
            (h * (gate * q + kvp) + q * h, !shared)
        }
    }
}

fn head_dim(s: &Shape, kind: LayerKind) -> f64 {
    match kind {
        LayerKind::Global | LayerKind::SharedKv { sliding: false, .. } => {
            f64::from(s.attn.global_head_dim.unwrap_or(s.attn.head_dim))
        }
        _ => f64::from(s.attn.head_dim),
    }
}

/// A layer's KV heads and their dim.
fn kv_heads(s: &Shape, kind: LayerKind) -> (f64, f64) {
    let global = matches!(
        kind,
        LayerKind::Global | LayerKind::SharedKv { sliding: false, .. }
    );
    let heads = if global {
        s.attn.global_kv_heads.unwrap_or(s.attn.kv_heads)
    } else {
        s.attn.kv_heads
    };
    (f64::from(heads), head_dim(s, kind))
}

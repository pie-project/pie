//! What a plan costs on the wafer: a decode step's time ([`time`]) and each PE image's memory
//! ([`memory`]). The search ([`crate::compile::search`]) ranks partitions by these.

/// The CS-3's PE clock, Hz, measured against host wall time; every cycle count the cost model
/// and the stamps give is seconds at this rate.
pub const CLOCK_HZ: f64 = 750e6;

mod anchors;
pub mod gemm;
pub mod grid;
pub mod memory;
pub mod stream;
pub mod tiled;
pub mod time;

use serde::Serialize;

use crate::compile::plan::{LayerKind, Plan, Shape};
use crate::tuning;

/// The plan's facts a step's time is a function of, per decode step.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Features {
    /// Decoder layers.
    pub layers: f64,
    /// The weight elements one PE multiplies a step, summed over the layers (the busiest PE of
    /// each phase: a phase's matrix over the PEs that hold it; routed experts: the top-k).
    pub matvec: f64,
    /// Cycles a step the reductions' chains take end to end ([`chain_cycles`]): each layer's two
    /// row chains into CX and two column chains into CY, a chain the slower of its two halves.
    pub chain: f64,
    /// The fabric distance a step's collectives cross down the columns: each layer's multicasts
    /// and reductions span the unit's rows.
    pub span_rows: f64,
    /// The distance they cross along the rows: the unit's columns (the attention or the MLP
    /// half's, the narrower).
    pub span_cols: f64,
    /// KV elements one row PE reads a step (keys and values, the attention layers' entries times
    /// a column's KV dims).
    pub kv: f64,
    /// The head's elements one head PE multiplies a step (its vocabulary rows times its share of
    /// the hidden).
    pub head: f64,
    /// Cycles a step the token loop's max-reduce takes ([`loop_reduce_cycles`]), the head
    /// control line's slower half into its middle.
    pub loop_reduce: f64,
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
        let (span_rows, span_cols) = (layers * rows, layers * span_c);
        // a unit's halves either side of its roots: CX after the first cols / 2 columns (gpt-oss:
        // after the attention half's), CY after the first rows / 2 rows
        let (cl, ch) = match p.attn_cols {
            Some(a) => (a, p.cols - a),
            None => (p.cols / 2, p.cols - p.cols / 2),
        };
        let (rl, rh) = (p.rows / 2, p.rows - p.rows / 2);
        let half = |a: u32, b: u32, g: u32| chain_cycles(a, g).max(chain_cycles(b, g));
        let chain = layers * 2.0 * (half(cl, ch, p.reduce.row) + half(rl, rh, p.reduce.col));
        let head = f64::from(p.head.vocab_rows) * (h / f64::from(p.head.cols.max(1))).ceil();
        // the control line: the band's row of head columns (gpt-oss), else a column of head rows
        let line = if p.attn_cols.is_some() {
            p.head.cols
        } else {
            p.head.rows
        };
        let loop_reduce = if plan.contract.token_loop {
            loop_reduce_cycles(line, p.reduce.token_loop, p.reduce.token_loop3)
        } else {
            0.0
        };
        Features {
            layers,
            matvec,
            chain,
            span_rows,
            span_cols,
            kv,
            head,
            loop_reduce,
        }
    }
}

/// The cycles the token loop's max-reduce over a control line of `n` PEs takes into its middle
/// (`n / 2`): the two-level chain in groups of `g1` (`G2W`), or with `g2` (`G3W`) > 0 the
/// three-level tree ([`crate::compile::wiring::three_level`]). Each PE merges once its children's pairs
/// are in ([`tree_cycles`]).
#[must_use]
pub fn loop_reduce_cycles(n: u32, g1: u32, g2: u32) -> f64 {
    let (n, root) = (n as usize, (n / 2) as usize);
    let parent = if g2 > 0 && g1 > 0 {
        crate::compile::wiring::three_level(n, root, g1 as usize, g2 as usize)
    } else {
        crate::compile::wiring::two_level(n, root, g1 as usize)
    };
    tree_cycles(
        &parent,
        root,
        tuning::get().loop_merge,
        tuning::get().loop_wire,
    )
}

/// The cycles a reduction tree over a line (`parent[i]`: the point `i` sends to, `None` at
/// `root`) takes from every point's own value to the last child's sum at the root: a point
/// sends `merge` cycles after the last of its children's sums is in (a leaf `merge` after the
/// start), a sum `wire` cycles a hop on its way.
#[must_use]
pub fn tree_cycles(parent: &[Option<usize>], root: usize, merge: f64, wire: f64) -> f64 {
    let n = parent.len();
    let mut order: Vec<usize> = (0..n).filter(|&i| i != root).collect();
    // children are farther from the root than their parents: the farthest first
    order.sort_by_key(|&i| std::cmp::Reverse(i.abs_diff(root)));
    let mut ready = vec![0.0f64; n];
    let mut sent = vec![0.0f64; n];
    for &i in &order {
        sent[i] = ready[i] + merge;
        if let Some(p) = parent[i] {
            let at = sent[i] + wire * i.abs_diff(p) as f64;
            ready[p] = ready[p].max(at);
        }
    }
    ready[root]
}

/// The cycles a two-level chain's half of `members` takes in groups of `group` (0: one group),
/// from every member's partial to the sum leaving the half for the root. Groups count from the
/// root out, the remainder farthest; each group end merges its sum into the level-2 sum once
/// both are in, so a short far group starts the level-2 sum early.
#[must_use]
pub fn chain_cycles(members: u32, group: u32) -> f64 {
    if members == 0 {
        return 0.0;
    }
    let g = if group == 0 {
        members
    } else {
        group.min(members)
    };
    let n = members.div_ceil(g);
    let mut end: Option<f64> = None;
    for j in 0..n {
        // far first: the remainder, then full groups
        let size = if j == 0 { members - (n - 1) * g } else { g };
        let level1 = f64::from(size - 1) * (tuning::get().chain_add + tuning::get().chain_wire)
            + tuning::get().chain_wire;
        let level2 = end.map_or(f64::NEG_INFINITY, |e| {
            e + f64::from(size) * tuning::get().chain_wire
        });
        end = Some(level1.max(level2) + tuning::get().chain_merge);
    }
    end.unwrap_or(0.0)
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

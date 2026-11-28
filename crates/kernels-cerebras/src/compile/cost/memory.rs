//! Each PE image's memory, against the 48 KiB a PE has ([`crate::compile::build::STACK_ROOM`] must stay
//! free). The weight and head images move with the partition: their bytes are an anchor's as the
//! linker measured them (`ANCHORS`) plus what the plan's tiles and caches add over the anchor
//! plan's. The other images are as the anchor's.

use crate::compile::kernels::KernelSet;
use crate::compile::plan::{LayerKind, Plan, Shape};

/// A PE's memory ([`crate::compile::check::PE_BYTES`]; the stack grows down from `0xC000`).
pub const PE_BYTES: f64 = crate::compile::check::PE_BYTES as f64;

/// The bytes of a plan that move with the partition: the weight PE's tiles and KV cache, the head
/// PE's tiles and embeddings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Moving {
    pub weight: f64,
    pub head: f64,
}

impl Moving {
    #[must_use]
    pub fn of(plan: &Plan) -> Moving {
        let s = &plan.shape;
        let p = &plan.partition;
        let ceil = |a: u32, b: u32| f64::from(a.div_ceil(b.max(1)));
        let k = p.layers_per_unit as usize;
        let (att_cols, mlp_cols) = match p.attn_cols {
            Some(a) => (a, p.cols - a),
            None => (p.cols, p.cols),
        };
        let split = p.attn_cols.is_some();
        // the scores' buffers (f32 partials and scores, their f16 copy): each KV head's q heads
        // times the entries a row keeps
        let entries = [
            p.kv_entries.per_row,
            p.kv_entries.global.unwrap_or(0),
            p.kv_entries.sliding.unwrap_or(0),
        ]
        .into_iter()
        .max()
        .unwrap_or(0);
        let qh = s.attn.q_heads / s.attn.kv_heads.max(1);
        // each run of rows on its own (its x dims; the cache and the scores only where its rows
        // attend): a plain row's wider dims and an attention row's cache are not one PE's
        let mut weight: f64 = 0.0;
        for run in &p.row_dims {
            let hc = f64::from(run.dims);
            // the busiest unit: its K layers' tiles (f16; routed experts E5M2) and caches, the
            // attention and MLP halves on their own PEs where they split
            for (u, unit) in s.layers.chunks(k.max(1)).enumerate() {
                let (mut att, mut mlp) = (0.0, 0.0);
                for (i, kind) in unit.iter().enumerate() {
                    let l = u * k + i;
                    let (outs, q) = projections(plan, *kind);
                    att += 2.0 * hc * (ceil(outs, att_cols) + ceil(q, att_cols));
                    mlp += match s.moe {
                        Some(m) => {
                            f64::from(m.experts)
                                * 3.0
                                * f64::from(s.hidden)
                                * f64::from(m.expert_intermediate)
                                / (f64::from(p.rows.max(1)) * f64::from(mlp_cols.max(1)))
                        }
                        None => 2.0 * hc * 3.0 * ceil(s.mlp.width(l as u32), mlp_cols),
                    };
                    if let (true, Some(entries)) = (run.attention, cache_entries(plan, *kind)) {
                        let (heads, dim) = kv(plan, *kind);
                        let dc = ceil(heads * dim, att_cols);
                        // keys and values (f16), the keys' inverse norms (f32)
                        att += entries * (4.0 * dc + 4.0);
                    }
                }
                // a ring's other seats: their GDN states and log-sum-exp references (f32)
                if plan.seats > 1 {
                    let (gdn, nla) = seat_state(plan, unit);
                    att += f64::from(plan.seats - 1) * 4.0 * (gdn + nla * f64::from(qh));
                }
                let mut w = if split { att.max(mlp) } else { att + mlp };
                if !split && run.attention {
                    w += 10.0 * f64::from(qh * entries);
                }
                weight = weight.max(w);
            }
        }
        let hdc = ceil(s.hidden, p.head.cols);
        let head = 4.0 * f64::from(p.head.vocab_rows) * hdc;
        Moving { weight, head }
    }
}

/// A unit's GDN state words a weight PE (its layers' v dims a column) and its attention layers.
fn seat_state(plan: &Plan, unit: &[LayerKind]) -> (f64, f64) {
    let dv = plan.shape.gdn.map_or(0, |g| {
        (g.value_heads * g.value_dim).div_ceil(plan.partition.cols.max(1))
    });
    let gdn = unit.iter().filter(|l| matches!(l, LayerKind::Gdn)).count() as f64;
    (gdn * f64::from(dv), unit.len() as f64 - gdn)
}

/// A layer's projection outputs (q, k, v and gates, or GDN's) and its q heads' dims (the
/// o-projection's inputs).
fn projections(plan: &Plan, kind: LayerKind) -> (u32, u32) {
    let s = &plan.shape;
    match kind {
        LayerKind::Gdn => {
            let g = s.gdn.expect("a GDN layer has a GDN spec");
            let (k, v) = (g.key_heads * g.key_dim, g.value_heads * g.value_dim);
            (2 * k + 2 * v + 2 * g.value_heads, v)
        }
        _ => {
            let (heads, dim) = kv(plan, kind);
            let q = s.attn.q_heads * dim;
            let gate = if s.attn.output_gate { 2 } else { 1 };
            let shared = matches!(kind, LayerKind::SharedKv { .. });
            (gate * q + if shared { 0 } else { 2 * heads * dim }, q)
        }
    }
}

/// The entries a row keeps of a layer's own cache (`None`: it reads another's, or none).
fn cache_entries(plan: &Plan, kind: LayerKind) -> Option<f64> {
    let e = &plan.partition.kv_entries;
    match kind {
        LayerKind::Gdn | LayerKind::SharedKv { .. } => None,
        // (a ring: a ring a seat, or each seat's within its window where those take fewer)
        LayerKind::Sliding { .. } if plan.seats > 1 => {
            let pool = e.global.unwrap_or(e.per_row);
            Some(f64::from(
                e.sliding.map_or(pool, |n| (n * plan.seats).min(pool)),
            ))
        }
        LayerKind::Sliding { .. } => Some(f64::from(e.sliding.unwrap_or(e.per_row))),
        LayerKind::Global => Some(f64::from(e.global.unwrap_or(e.per_row))),
    }
}

/// A layer's KV heads and head dim.
fn kv(plan: &Plan, kind: LayerKind) -> (u32, u32) {
    let a = &plan.shape.attn;
    let global = matches!(
        kind,
        LayerKind::Global | LayerKind::SharedKv { sliding: false, .. }
    );
    if global {
        (
            a.global_kv_heads.unwrap_or(a.kv_heads),
            a.global_head_dim.unwrap_or(a.head_dim),
        )
    } else {
        (a.kv_heads, a.head_dim)
    }
}

/// Code a weight image links to beside its anchor's where a plan's comptime sizes take another
/// path ([`Anchor::code`]): `bytes` where `on` holds of the plan and not of the anchor's (less
/// where the anchor's held and the plan's does not).
pub struct Code {
    pub what: &'static str,
    pub on: fn(&Plan) -> bool,
    /// Whether the anchor's plan takes the path.
    pub anchor: bool,
    pub bytes: f64,
}

impl Code {
    /// The bytes `plan`'s weight image takes past (or short of) the anchor's for this path.
    #[must_use]
    pub fn over(&self, plan: &Plan) -> f64 {
        match ((self.on)(plan), self.anchor) {
            (true, false) => self.bytes,
            (false, true) => -self.bytes,
            _ => 0.0,
        }
    }
}

/// A kernel set's program as the linker measured it: its weight and head images' bytes, the
/// least stack room of its other images, and its plan's moving bytes. A kernel change that moves
/// code or buffers re-measures them.
pub struct Anchor {
    pub kernels: KernelSet,
    pub weight_used: f64,
    pub head_used: f64,
    pub other_room: f64,
    pub moving: Moving,
    /// Whether the head's bytes follow [`Moving::head`] (the column head); the grouped and band
    /// heads stay as measured.
    pub head_moves: bool,
    /// The weight image's code paths that move with the plan.
    pub code: &'static [Code],
    /// The attention rows' moving bytes, where the weight image measured is an attention row's
    /// beside plain rows that move more (0: none): a ring's pool grows the attention rows'.
    pub attention: f64,
}

/// The anchor a model's plans are measured off: its kernel set's.
#[must_use]
pub fn anchor(shape: &Shape) -> &'static Anchor {
    let id = KernelSet::of_shape(shape);
    ANCHORS
        .iter()
        .find(|a| a.kernels == id)
        .expect("every kernel set has an anchor")
}

/// The least stack room `plan`'s images leave, measured off `anchor`.
#[must_use]
pub fn room(plan: &Plan, anchor: &Anchor) -> f64 {
    let m = Moving::of(plan);
    let code: f64 = anchor.code.iter().map(|c| c.over(plan)).sum();
    let mut over = m.weight - anchor.moving.weight;
    if plan.seats > 1 && anchor.attention > 0.0 {
        let mut rows = plan.clone();
        rows.partition.row_dims.retain(|r| r.attention);
        over = over.max(Moving::of(&rows).weight - anchor.attention);
    }
    let weight = PE_BYTES - (anchor.weight_used + over + code);
    let head = if anchor.head_moves {
        PE_BYTES - (anchor.head_used + m.head - anchor.moving.head)
    } else {
        PE_BYTES - anchor.head_used
    };
    weight.min(head).min(anchor.other_room)
}

pub use super::anchors::ANCHORS;

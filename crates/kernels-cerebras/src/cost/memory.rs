//! Each PE image's memory, against the 48 KiB a PE has (the stack grows down from `0xC000` past
//! what the linker counts; [`crate::build::STACK_ROOM`] must stay free).
//!
//! The images a partition moves are the weight PE's (its K layers' tiles and its KV cache) and the
//! head PE's (its vocabulary rows' tiles and embeddings); code and the small buffers do not move
//! with the partition. So an image's bytes are an anchor's, as the linker measured them for a
//! program of the same family (`ANCHORS`), plus what the plan's tiles and caches add over the
//! anchor plan's. The other images (the roots, the head control, the relays) are as the anchor's.

use crate::plan::{FamilyId, LayerKind, Plan, Shape};

/// A PE's memory below the stack (`0xC000`).
pub const PE_BYTES: f64 = 49152.0;

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
        // a weight PE's x dims: the widest row run's
        let hc = p
            .row_dims
            .iter()
            .map(|r| f64::from(r.dims))
            .fold(0.0, f64::max);
        let (att_cols, mlp_cols) = match p.attn_cols {
            Some(a) => (a, p.cols - a),
            None => (p.cols, p.cols),
        };
        // the busiest unit: its K layers' tiles (f16; routed experts E5M2) and its caches, the
        // attention half's and the MLP half's on their own PEs where they split (gpt-oss)
        let split = p.attn_cols.is_some();
        let mut weight: f64 = 0.0;
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
                if let Some(entries) = cache_entries(plan, *kind) {
                    let (heads, dim) = kv(plan, *kind);
                    let dc = ceil(heads * dim, att_cols);
                    // keys and values (f16), the keys' inverse norms (f32)
                    att += entries * (4.0 * dc + 4.0);
                }
            }
            weight = weight.max(if split { att.max(mlp) } else { att + mlp });
        }
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
        if !split {
            weight += 10.0 * f64::from(qh * entries);
        }
        let hdc = ceil(s.hidden, p.head.cols);
        let head = 4.0 * f64::from(p.head.vocab_rows) * hdc;
        Moving { weight, head }
    }
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

/// A program as the linker measured it: the bytes its weight and head images use, the least
/// stack room of its other images, and its plan's moving bytes. Linked at d0e8fe23c
/// (`wafer-compile link`, `cs-readelf -m`: `scratch/genC/mem`); a kernel change that moves code
/// or buffers re-measures them.
pub struct Anchor {
    pub family: FamilyId,
    /// The model's shape signature and the preset (with its levers) the program was built from.
    pub signature: [u32; 3],
    pub preset: &'static str,
    pub weight_used: f64,
    pub head_used: f64,
    pub other_room: f64,
    pub moving: Moving,
    /// Whether the head's bytes follow [`Moving::head`] (the column head); the grouped and band
    /// heads stay as measured.
    pub head_moves: bool,
}

/// The anchor a model's plans are measured off: the one linked from the model itself, else its
/// family's widest (a model no program of was linked: its code and buffers as that one's).
#[must_use]
pub fn anchor(shape: &Shape) -> &'static Anchor {
    let id = crate::family::serving(shape);
    let sig = shape.signature();
    let of = ANCHORS.iter().filter(|a| a.family == id);
    of.clone()
        .find(|a| a.signature == sig)
        .or_else(|| of.max_by(|a, b| a.weight_used.total_cmp(&b.weight_used)))
        .expect("every family has an anchor")
}

/// The least stack room `plan`'s images leave, measured off `anchor`.
#[must_use]
pub fn room(plan: &Plan, anchor: &Anchor) -> f64 {
    let m = Moving::of(plan);
    let weight = PE_BYTES - (anchor.weight_used + m.weight - anchor.moving.weight);
    let head = if anchor.head_moves {
        PE_BYTES - (anchor.head_used + m.head - anchor.moving.head)
    } else {
        PE_BYTES - anchor.head_used
    };
    weight.min(head).min(anchor.other_room)
}

include!("anchors.rs");

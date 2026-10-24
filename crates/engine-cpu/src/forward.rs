//! A poem trace's forward on the host, one row at a time: what a rank of
//! a group computes, the collectives among ranks run through an in-memory
//! [`Exchange`]. It serves one lane a step and holds its keys and values by
//! position; it runs the ops a plain MoE trunk is written in (embedding,
//! norms, products, rotary, paged attention over its own lane, routed
//! experts, an expert-parallel dispatch and combine) and refuses the rest.
//!
//! Each value is f32 on the host, rounded to bf16 where the trace types it
//! so. A row's result depends only on that row (and, in attention, on its
//! lane's earlier rows), never on which other rows a step or a rank holds:
//! one rank holding every expert and `N` ranks holding a part each answer
//! the same bits, which is what [`Forward`] is for.
//!
//! A dispatch follows `Collective::Dispatch`: every rank's tokens, `T` the
//! most any rank brings, land on the ranks holding their routes (by the
//! engine's uniform placement, a replica picked by source rank, token and
//! route), `ep * T` rows in source rank then token order, padded with NaN
//! rows whose routes are -1. A combine sends each landed route's row back
//! to its token; a route nothing returns stays NaN.

use std::sync::Arc;

use engine::placement::Placement;
use poem_ir::{
    Attn, Collective, Def, Dim, Dtype, Elemwise, Gather, GluAct, Guard, Linear, NormAxis,
    NormWeight, Operands, Operation, ParamSource, RouteScore, RuntimeInput, Trace, Turn, Ty,
    ValueId,
};

pub use crate::exchange::{Exchange, Post};

/// A value as the forward holds it: rows of `width` elements.
#[derive(Clone, Debug)]
enum Held {
    Floats {
        rows: usize,
        width: usize,
        v: Vec<f32>,
    },
    Ints {
        rows: usize,
        width: usize,
        v: Vec<i32>,
    },
    /// A plan or a cache handle: read by nothing here.
    Opaque,
}

/// A float value's view.
struct Rows<'a> {
    rows: usize,
    width: usize,
    v: &'a [f32],
}

impl Rows<'_> {
    fn row(&self, r: usize) -> &[f32] {
        &self.v[r * self.width..(r + 1) * self.width]
    }
}

/// An int value's view.
struct Ints<'a> {
    rows: usize,
    width: usize,
    v: &'a [i32],
}

/// One lane's keys and values for one cache, by position.
#[derive(Default)]
struct Lane {
    k: Vec<Option<Vec<f32>>>,
    v: Vec<Option<Vec<f32>>>,
}

/// A layer's placement: the dispatch's table (`stride` i32 an expert) and
/// the experts each rank's slots hold.
struct Placed {
    table: Vec<i32>,
    stride: usize,
    held: Vec<u32>,
}

/// This rank's part of a group.
pub struct Group {
    pub rank: usize,
    pub exchange: Arc<Exchange>,
}

/// One rank's forward of `trace`, its weights read at construction.
pub struct Forward {
    trace: Trace,
    group: Option<Group>,
    weights: Vec<Option<Held>>,
    lanes: Vec<Lane>,
    placed: std::collections::BTreeMap<u32, Placed>,
}

fn bf16(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    let bits = x.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    f32::from_bits(rounded & 0xffff_0000)
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// A value's elements a row: its shape past the first axis.
fn width(ty: &Ty) -> usize {
    match ty {
        Ty::Tensor { shape, .. } => (shape.iter().skip(1))
            .map(|d| match d {
                Dim::Const(n) => *n as usize,
                _ => 1,
            })
            .product(),
        Ty::Struct(_) => 0,
    }
}

impl Forward {
    /// `trace` as rank `group.rank` runs it, or alone without a group.
    /// `plane(name, elements)` is a checkpoint plane's values, whole: a
    /// rank keeps the experts its slots hold of a bank the group spreads.
    /// Registered planes (adapters) are zero.
    pub fn new(
        trace: Trace,
        group: Option<Group>,
        plane: &dyn Fn(&str, usize) -> Vec<f32>,
    ) -> Result<Forward, String> {
        let world = group.as_ref().map_or(1, |g| g.exchange.world());
        let rank = group.as_ref().map_or(0, |g| g.rank);
        let mut placed = std::collections::BTreeMap::new();
        for node in &trace.nodes {
            if let Operation::Collective(Collective::Dispatch {
                ep,
                experts,
                slots,
                layer,
                ..
            }) = node.op
            {
                if ep as usize != world {
                    return Err(format!("a dispatch over {ep} ranks in a group of {world}"));
                }
                placed
                    .entry(layer)
                    .or_insert(Self::place(experts, ep, slots)?);
            }
        }
        let mut weights = vec![None; trace.values.len()];
        for (id, decl) in trace.values.iter().enumerate() {
            let Def::Weight(p) = decl.def else {
                continue;
            };
            let param = &trace.params[p as usize];
            let rows = *param.shape.first().unwrap_or(&1) as usize;
            let each: usize = param.shape.iter().skip(1).product::<u64>() as usize;
            let v = match (param.source, param.experts) {
                (ParamSource::Registered, _) => vec![0.0; rows * each],
                (ParamSource::Checkpoint, None) => plane(&param.name, rows * each),
                (ParamSource::Checkpoint, Some(cut)) => {
                    let whole = plane(&param.name, cut.experts as usize * each);
                    let held = &placed
                        .get(&cut.layer)
                        .ok_or_else(|| format!("`{}`: no dispatch places its layer", param.name))?
                        .held;
                    (0..cut.slots as usize)
                        .flat_map(|s| {
                            let e = held[rank * cut.slots as usize + s] as usize;
                            whole[e * each..(e + 1) * each].iter().copied()
                        })
                        .collect()
                }
            };
            if v.len() != rows * each {
                return Err(format!(
                    "`{}`: {} values for {rows} x {each}",
                    param.name,
                    v.len()
                ));
            }
            weights[id] = Some(Held::Floats {
                rows,
                width: each,
                v: v.into_iter().map(bf16).collect(),
            });
        }
        let lanes = (0..trace.caches.len()).map(|_| Lane::default()).collect();
        Ok(Forward {
            trace,
            group,
            weights,
            lanes,
            placed,
        })
    }

    /// The engine's uniform placement of `experts` over `ranks` ranks of
    /// `slots` (expert-parallel `EpRank::new`'s).
    fn place(experts: u32, ranks: u32, slots: u32) -> Result<Placed, String> {
        let placement =
            Placement::uniform(experts, ranks, slots).map_err(|why| format!("{why:?}"))?;
        let stride = Placement::stride_for(ranks * slots - experts + 1);
        Ok(Placed {
            table: placement.table(stride),
            stride: stride as usize,
            held: placement.held,
        })
    }

    /// One step of this rank's lane: `tokens` at positions from `start`,
    /// the trace's arms chosen by `word`. Returns the last row's logits
    /// (empty for a step of no tokens, which a rank fires to serve its
    /// peers' routes).
    pub fn step(&mut self, tokens: &[u32], start: u32, word: u64) -> Result<Vec<f32>, String> {
        let n = tokens.len();
        let positions: Vec<i32> = (0..n).map(|t| (start as usize + t) as i32).collect();
        let mut vals: Vec<Option<Held>> = self.weights.clone();
        for (id, decl) in self.trace.values.iter().enumerate() {
            let held = match &decl.def {
                Def::Input(RuntimeInput::Tokens) => Held::Ints {
                    rows: n,
                    width: 1,
                    v: tokens.iter().map(|&t| t as i32).collect(),
                },
                Def::Input(RuntimeInput::Positions) => Held::Ints {
                    rows: n,
                    width: 1,
                    v: positions.clone(),
                },
                Def::Input(RuntimeInput::AdapterRoutes) => Held::Ints {
                    rows: n,
                    width: 1,
                    v: vec![-1; n],
                },
                Def::Input(RuntimeInput::ReadoutRows) => Held::Ints {
                    rows: usize::from(n > 0),
                    width: 1,
                    v: if n > 0 {
                        vec![n as i32 - 1]
                    } else {
                        Vec::new()
                    },
                },
                Def::Input(RuntimeInput::Mask { .. } | RuntimeInput::Geometry { .. })
                | Def::Cache(_) => Held::Opaque,
                Def::Input(other) => return Err(format!("no input {other:?} here")),
                _ => continue,
            };
            vals[id] = Some(held);
        }
        let mut run = Run {
            trace: &self.trace,
            vals,
            word,
            positions: &positions,
            lanes: &mut self.lanes,
            group: self.group.as_ref(),
            placed: &self.placed,
        };
        let mut aliases = Vec::new();
        for (at, node) in self.trace.nodes.iter().enumerate() {
            if !node.guard.holds(word) {
                // An op the word skips leaves what it writes in place as
                // it found it.
                aliases.clear();
                node.op.aliases(&mut aliases);
                for &(out, into) in &aliases {
                    let kept = run.held(into).ok().cloned();
                    run.vals[out.0 as usize] = kept;
                }
                continue;
            }
            run.op(&node.op)
                .map_err(|why| format!("node {at}: {why}"))?;
        }
        let out = (self.trace.seams.iter())
            .find(|s| s.seam == "out")
            .and_then(|s| s.values.first().copied())
            .ok_or("the trace has no out seam")?;
        let logits = run.floats(out)?;
        Ok(if logits.rows == 0 {
            Vec::new()
        } else {
            logits.row(logits.rows - 1).to_vec()
        })
    }
}

struct Run<'a> {
    trace: &'a Trace,
    vals: Vec<Option<Held>>,
    word: u64,
    positions: &'a [i32],
    lanes: &'a mut Vec<Lane>,
    group: Option<&'a Group>,
    placed: &'a std::collections::BTreeMap<u32, Placed>,
}

impl Run<'_> {
    /// The value `id` reads as: a merge's arm the word takes.
    fn resolve(&self, mut id: ValueId) -> Result<ValueId, String> {
        while let Def::Merge(arms) = &self.trace.values[id.0 as usize].def {
            id = (arms.iter())
                .find(|(_, g): &&(ValueId, Guard)| g.holds(self.word))
                .map(|(v, _)| *v)
                .ok_or_else(|| format!("no arm of value {} holds", id.0))?;
        }
        Ok(id)
    }

    fn held(&self, id: ValueId) -> Result<&Held, String> {
        let at = self.resolve(id)?;
        self.vals[at.0 as usize]
            .as_ref()
            .ok_or_else(|| format!("value {} is read before it is made", at.0))
    }

    fn floats(&self, id: ValueId) -> Result<Rows<'_>, String> {
        match self.held(id)? {
            Held::Floats { rows, width, v } => Ok(Rows {
                rows: *rows,
                width: *width,
                v,
            }),
            _ => Err(format!("value {} is not floats", id.0)),
        }
    }

    fn ints(&self, id: ValueId) -> Result<Ints<'_>, String> {
        match self.held(id)? {
            Held::Ints { rows, width, v } => Ok(Ints {
                rows: *rows,
                width: *width,
                v,
            }),
            _ => Err(format!("value {} is not ints", id.0)),
        }
    }

    fn opt_floats(&self, id: Option<ValueId>) -> Result<Option<Rows<'_>>, String> {
        id.map(|v| self.floats(v)).transpose()
    }

    /// Lands `v` (`rows` of the value's width) as value `id`, rounded to
    /// bf16 where the trace types it so.
    fn put(&mut self, id: ValueId, rows: usize, v: Vec<f32>) -> Result<(), String> {
        let ty = &self.trace.values[id.0 as usize].ty;
        let width = v.len().checked_div(rows).unwrap_or_else(|| width(ty));
        if v.len() != rows * width {
            return Err(format!(
                "value {}: {} values for {rows} rows",
                id.0,
                v.len()
            ));
        }
        let v = match ty {
            Ty::Tensor {
                dtype: Dtype::Bf16, ..
            } => v.into_iter().map(bf16).collect(),
            _ => v,
        };
        self.vals[id.0 as usize] = Some(Held::Floats { rows, width, v });
        Ok(())
    }

    fn put_ints(&mut self, id: ValueId, rows: usize, width: usize, v: Vec<i32>) {
        self.vals[id.0 as usize] = Some(Held::Ints { rows, width, v });
    }

    fn op(&mut self, op: &Operation) -> Result<(), String> {
        match op {
            Operation::Attn(op) => self.attn(op),
            Operation::Elemwise(op) => self.elemwise(op),
            Operation::Gather(op) => self.gather(op),
            Operation::Linear(op) => self.linear(op),
            Operation::Collective(op) => self.collective(op),
            other => Err(format!("`{}` has no host forward", Operands::name(other))),
        }
    }

    fn attn(&mut self, op: &Attn) -> Result<(), String> {
        match *op {
            Attn::PlanPrefill { plan, .. } | Attn::PlanDecode { plan, .. } => {
                self.vals[plan.0 as usize] = Some(Held::Opaque);
                Ok(())
            }
            Attn::KvAppend {
                k,
                v: Some(v),
                cache,
                ..
            } => {
                let Def::Cache(c) = self.trace.values[cache.0 as usize].def else {
                    return Err("a kv append into no cache".into());
                };
                let (keys, values) = (self.floats(k)?, self.floats(v)?);
                let rows: Vec<_> = (0..keys.rows)
                    .map(|t| (keys.row(t).to_vec(), values.row(t).to_vec()))
                    .collect();
                let lane = &mut self.lanes[c as usize];
                for (t, (k, v)) in rows.into_iter().enumerate() {
                    let p = self.positions[t] as usize;
                    if lane.k.len() <= p {
                        lane.k.resize(p + 1, None);
                        lane.v.resize(p + 1, None);
                    }
                    lane.k[p] = Some(k);
                    lane.v[p] = Some(v);
                }
                Ok(())
            }
            Attn::Prefill {
                q,
                cache,
                mask: None,
                selection: None,
                bias: None,
                floor: None,
                window: None,
                head_dim,
                causal: true,
                sm_scale,
                selected: None,
                rel: None,
                o,
                lse,
                ..
            }
            | Attn::Decode {
                q,
                cache,
                selection: None,
                bias: None,
                window: None,
                head_dim,
                sm_scale,
                selected: None,
                rel: None,
                o,
                lse,
                ..
            } => self.attend(q, cache, head_dim as usize, sm_scale, o, lse),
            Attn::Prefill { mask: Some(_), .. } => {
                Err("a custom mask, which this forward does not read".into())
            }
            _ => Err(format!("`{}` has no host forward", Operands::name(op))),
        }
    }

    /// Each row's query against its lane's keys up to its position.
    fn attend(
        &mut self,
        q: ValueId,
        cache: ValueId,
        hd: usize,
        scale: f32,
        o: ValueId,
        lse: Option<ValueId>,
    ) -> Result<(), String> {
        let Def::Cache(c) = self.trace.values[cache.0 as usize].def else {
            return Err("attention over no cache".into());
        };
        let q = self.floats(q)?;
        let lane = &self.lanes[c as usize];
        let heads = q.width / hd;
        let kv_heads = (lane.k.iter().flatten().next()).map_or(1, |k| k.len() / hd);
        let group = heads / kv_heads.max(1);
        let mut out = vec![0.0; q.rows * q.width];
        let mut lses = vec![0.0; q.rows * heads];
        for t in 0..q.rows {
            let p = self.positions[t] as usize;
            for h in 0..heads {
                let qh = &q.row(t)[h * hd..(h + 1) * hd];
                let kh = h / group;
                let mut scores = Vec::with_capacity(p + 1);
                for at in 0..=p {
                    let k = lane.k.get(at).and_then(Option::as_ref);
                    let k = k.ok_or_else(|| format!("no key at position {at}"))?;
                    scores.push(dot(qh, &k[kh * hd..(kh + 1) * hd]) * scale);
                }
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = scores.iter().map(|s| (s - max).exp()).sum();
                let row = &mut out[t * q.width + h * hd..][..hd];
                for (at, s) in scores.iter().enumerate() {
                    let w = (s - max).exp() / sum;
                    let v = lane.v[at].as_ref().ok_or("a key without its value")?;
                    for (o, v) in row.iter_mut().zip(&v[kh * hd..(kh + 1) * hd]) {
                        *o += w * v;
                    }
                }
                lses[t * heads + h] = max + sum.ln();
            }
        }
        let rows = q.rows;
        self.put(o, rows, out)?;
        if let Some(lse) = lse {
            self.put(lse, rows, lses)?;
        }
        Ok(())
    }

    fn elemwise(&mut self, op: &Elemwise) -> Result<(), String> {
        match *op {
            Elemwise::Rmsnorm {
                x,
                weight,
                axis,
                eps,
                y,
            } => {
                let xs = self.floats(x)?;
                let run = match axis {
                    NormAxis::Row => xs.width,
                    NormAxis::Head(d) => d as usize,
                    NormAxis::Group(_) => return Err(format!("an rms norm over {axis:?}")),
                };
                let (w, plus) = match weight {
                    NormWeight::None => (None, 0.0),
                    NormWeight::Scale(w) => (Some(self.floats(w)?), 0.0),
                    NormWeight::PlusOne(w) => (Some(self.floats(w)?), 1.0),
                };
                let mut out = Vec::with_capacity(xs.v.len());
                for chunk in xs.v.chunks(run) {
                    let ms = chunk.iter().map(|v| v * v).sum::<f32>() / run as f32;
                    let inv = 1.0 / (ms + eps).sqrt();
                    for (i, v) in chunk.iter().enumerate() {
                        let g = w.as_ref().map_or(1.0, |w| w.v[i] + plus);
                        out.push(v * inv * g);
                    }
                }
                let rows = xs.rows;
                self.put(y, rows, out)
            }
            Elemwise::Rope {
                q,
                k,
                positions,
                head_dim,
                theta,
                turn,
                freqs: None,
                q_out,
                k_out,
            } => {
                let hd = head_dim as usize;
                let (pairs, span) = match turn {
                    Turn::Partial { rotary_dim, span } => (rotary_dim as usize / 2, span as usize),
                    Turn::Full { interleaved: false } => (hd / 2, hd / 2),
                    other => return Err(format!("a rotary turn {other:?}")),
                };
                let pos = self.ints(positions)?.v.to_vec();
                let turned = |rows: &Rows<'_>| {
                    let mut v = rows.v.to_vec();
                    for (t, &p) in pos.iter().enumerate().take(rows.rows) {
                        for head in 0..rows.width / hd {
                            let at = t * rows.width + head * hd;
                            for i in 0..pairs {
                                let inv = theta.powf(-(i as f32) / span as f32);
                                let (sin, cos) = (p as f32 * inv).sin_cos();
                                let (a, b) = (v[at + i], v[at + i + span]);
                                v[at + i] = a * cos - b * sin;
                                v[at + i + span] = b * cos + a * sin;
                            }
                        }
                    }
                    (rows.rows, v)
                };
                let (rows, qv) = turned(&self.floats(q)?);
                let kv = k.map(|k| self.floats(k).map(|k| turned(&k))).transpose()?;
                self.put(q_out, rows, qv)?;
                if let (Some((rows, kv)), Some(k_out)) = (kv, k_out) {
                    self.put(k_out, rows, kv)?;
                }
                Ok(())
            }
            Elemwise::GateSigmoidMul {
                x,
                gate,
                heads: None,
                x_out,
            } => {
                let (xs, g) = (self.floats(x)?, self.floats(gate)?);
                let out = xs.v.iter().zip(g.v).map(|(x, g)| x * sigmoid(*g)).collect();
                let rows = xs.rows;
                self.put(x_out, rows, out)
            }
            Elemwise::ResidualAdd {
                x,
                gate: None,
                y,
                y_out,
            } => {
                let (xs, ys) = (self.floats(x)?, self.floats(y)?);
                let out = xs.v.iter().zip(ys.v).map(|(x, y)| x + y).collect();
                let rows = xs.rows;
                self.put(y_out, rows, out)
            }
            _ => Err(format!("`{}` has no host forward", Operands::name(op))),
        }
    }

    fn gather(&mut self, op: &Gather) -> Result<(), String> {
        match *op {
            Gather::Embed {
                ids,
                weights: None,
                table,
                scales: None,
                concat: false,
                y,
                ..
            } => {
                let (ids, table) = (self.ints(ids)?, self.floats(table)?);
                let mut out = Vec::with_capacity(ids.rows * table.width);
                for &id in ids.v {
                    let id = usize::try_from(id).map_err(|_| "a negative token id")?;
                    if id >= table.rows {
                        return Err(format!("token {id} past the table"));
                    }
                    out.extend_from_slice(table.row(id));
                }
                let rows = ids.rows;
                self.put(y, rows, out)
            }
            Gather::SplitQGate {
                packed,
                head_dim,
                q,
                gate,
            } => {
                let p = self.floats(packed)?;
                let hd = head_dim as usize;
                let (mut qs, mut gs) = (Vec::new(), Vec::new());
                for unit in p.v.chunks(2 * hd) {
                    qs.extend_from_slice(&unit[..hd]);
                    gs.extend_from_slice(&unit[hd..]);
                }
                let rows = p.rows;
                self.put(q, rows, qs)?;
                self.put(gate, rows, gs)
            }
            Gather::Rows { x, rows, y } => {
                let (xs, picks) = (self.floats(x)?, self.ints(rows)?);
                let mut out = Vec::with_capacity(picks.rows * xs.width);
                for &r in picks.v {
                    out.extend_from_slice(xs.row(r as usize));
                }
                let rows = picks.rows;
                self.put(y, rows, out)
            }
            _ => Err(format!("`{}` has no host forward", Operands::name(op))),
        }
    }

    fn linear(&mut self, op: &Linear) -> Result<(), String> {
        match *op {
            Linear::Matmul {
                act,
                w,
                input_scales: None,
                scales: None,
                quant: None,
                y,
                ..
            } => {
                let (a, w) = (self.floats(act)?, self.floats(w)?);
                if a.width != w.width {
                    return Err(format!("a product of {} by {}", a.width, w.width));
                }
                let mut out = Vec::with_capacity(a.rows * w.rows);
                for t in 0..a.rows {
                    out.extend((0..w.rows).map(|o| dot(a.row(t), w.row(o))));
                }
                let rows = a.rows;
                self.put(y, rows, out)
            }
            Linear::LoraCorrect {
                routes, y, y_out, ..
            } => {
                if self.ints(routes)?.v.iter().any(|&r| r >= 0) {
                    return Err("an adapter's correction, which this forward does not run".into());
                }
                let ys = self.floats(y)?;
                let (rows, v) = (ys.rows, ys.v.to_vec());
                self.put(y_out, rows, v)
            }
            Linear::MoeRoute {
                ids: None,
                table: None,
                logits,
                bias: None,
                scale: None,
                hint: None,
                experts,
                top_k,
                score: RouteScore::Softmax,
                routes,
                weights,
            } => {
                let l = self.floats(logits)?;
                let (e, k) = (experts as usize, top_k as usize);
                let (mut picks, mut ws) = (Vec::new(), Vec::new());
                for t in 0..l.rows {
                    let mut row = l.row(t)[..e].to_vec();
                    let mut best = Vec::with_capacity(k);
                    for _ in 0..k {
                        let (at, v) = (row.iter().copied().enumerate()).fold(
                            (usize::MAX, f32::NEG_INFINITY),
                            |b, (i, v)| {
                                if v > b.1 { (i, v) } else { b }
                            },
                        );
                        row[at] = f32::NEG_INFINITY;
                        best.push((at, v));
                    }
                    let max = best[0].1;
                    let sum: f32 = best.iter().map(|(_, v)| (v - max).exp()).sum();
                    for (at, v) in best {
                        picks.push(at as i32);
                        ws.push((v - max).exp() / sum);
                    }
                }
                let rows = l.rows;
                self.put_ints(routes, rows, k, picks);
                self.put(weights, rows, ws)
            }
            Linear::MoeSelect {
                x,
                bank,
                bias: None,
                routes,
                input_scales: None,
                quantized: false,
                quant: None,
                y,
            } => {
                let (xs, bank, r) = (self.floats(x)?, self.floats(bank)?, self.ints(routes)?);
                let count = r.rows * r.width;
                let by_token = if xs.rows == count {
                    false
                } else if xs.rows == r.rows {
                    true
                } else {
                    return Err(format!(
                        "a select of {} rows over {} routes of {} tokens",
                        xs.rows, count, r.rows
                    ));
                };
                let n = bank.width / xs.width.max(1);
                let mut out = vec![0.0; count * n];
                for (route, &e) in r.v.iter().enumerate() {
                    let Ok(e) = usize::try_from(e) else {
                        continue;
                    };
                    if e >= bank.rows {
                        return Err(format!("route to expert {e} of {}", bank.rows));
                    }
                    let row = xs.row(if by_token { route / r.width } else { route });
                    let plane = bank.row(e);
                    for (o, w) in out[route * n..(route + 1) * n]
                        .iter_mut()
                        .zip(plane.chunks(xs.width))
                    {
                        *o = dot(row, w);
                    }
                }
                self.put(y, count, out)
            }
            Linear::Glu {
                packed,
                gate,
                up,
                weights: None,
                intermediate,
                act: GluAct::Swiglu,
                limit: None,
                y,
            } => {
                let i = intermediate as usize;
                let (rows, out) = match (packed, gate, up) {
                    (Some(p), None, None) => {
                        let p = self.floats(p)?;
                        let out = (0..p.rows)
                            .flat_map(|t| {
                                let row = p.row(t);
                                (0..i).map(move |c| row[c] * sigmoid(row[c]) * row[i + c])
                            })
                            .collect();
                        (p.rows, out)
                    }
                    (None, Some(g), Some(u)) => {
                        let (g, u) = (self.floats(g)?, self.floats(u)?);
                        let out = (g.v.iter().zip(u.v))
                            .map(|(g, u)| g * sigmoid(*g) * u)
                            .collect();
                        (g.rows, out)
                    }
                    _ => return Err("a glu of neither one packed plane nor two".into()),
                };
                self.put(y, rows, out)
            }
            Linear::MoeCombine {
                routed,
                weights: Some(weights),
                shared: None,
                y,
            } => {
                let (r, w) = (self.floats(routed)?, self.floats(weights)?);
                let k = w.width;
                let mut out = vec![0.0; w.rows * r.width];
                for t in 0..w.rows {
                    let o = &mut out[t * r.width..(t + 1) * r.width];
                    for j in 0..k {
                        let wt = w.v[t * k + j];
                        for (o, v) in o.iter_mut().zip(r.row(t * k + j)) {
                            *o += wt * v;
                        }
                    }
                }
                let rows = w.rows;
                self.put(y, rows, out)
            }
            Linear::MoeSigmoidGateAdd {
                routed,
                shared,
                gate,
                y,
            } => {
                let (r, s, g) = (
                    self.floats(routed)?,
                    self.floats(shared)?,
                    self.floats(gate)?,
                );
                let mut out = Vec::with_capacity(r.v.len());
                for t in 0..r.rows {
                    for c in 0..r.width {
                        let gv = if g.width == 1 { g.v[t] } else { g.row(t)[c] };
                        out.push(r.row(t)[c] + sigmoid(gv) * s.row(t)[c]);
                    }
                }
                let rows = r.rows;
                self.put(y, rows, out)
            }
            _ => Err(format!("`{}` has no host forward", Operands::name(op))),
        }
    }

    fn collective(&mut self, op: &Collective) -> Result<(), String> {
        let group = self.group.ok_or("a collective on a forward in no group")?;
        match *op {
            Collective::Dispatch {
                x,
                routes,
                weights,
                top_k,
                layer,
                slots,
                rows,
                local,
                row_weights,
                origin,
                counts,
                ..
            } => {
                let xs = self.floats(x)?;
                let r = self.ints(routes)?;
                let w = self.opt_floats(weights)?;
                let k = top_k as usize;
                if r.width != k || r.rows != xs.rows {
                    return Err("routes are not [rows, top_k]".into());
                }
                let post = Post::Tokens {
                    rows: xs.rows,
                    x: xs.v.to_vec(),
                    routes: r.v.to_vec(),
                    weights: w.map_or(Vec::new(), |w| w.v.to_vec()),
                };
                let all = group.exchange.all(group.rank, post);
                let place = self
                    .placed
                    .get(&layer)
                    .ok_or("a dispatch of no placement")?;
                let slots = slots as usize;
                let tokens = (all.iter())
                    .map(|p| match &**p {
                        Post::Tokens { rows, .. } => *rows,
                        Post::Outputs { .. } => 0,
                    })
                    .max()
                    .unwrap_or(0);
                let cap = all.len() * tokens;
                let width = xs.width;
                let mut land = Vec::with_capacity(cap * width);
                let mut slot_of = Vec::with_capacity(cap * k);
                let mut landed_w = Vec::with_capacity(cap * k);
                let mut from = Vec::with_capacity(cap);
                let mut tally = vec![0i32; slots];
                for (s, post) in all.iter().enumerate() {
                    let Post::Tokens {
                        rows: n,
                        x,
                        routes,
                        weights,
                    } = &**post
                    else {
                        return Err(format!("rank {s} fired another collective"));
                    };
                    for t in 0..*n {
                        let here: Vec<i32> = (0..k)
                            .map(|j| {
                                let e = routes[t * k + j];
                                if e < 0 {
                                    return -1;
                                }
                                let row = &place.table[e as usize * place.stride..];
                                let replica = (s + t * k + j) % row[0] as usize;
                                let p = row[1 + replica] as usize;
                                if p / slots == group.rank {
                                    (p % slots) as i32
                                } else {
                                    -1
                                }
                            })
                            .collect();
                        if here.iter().all(|&l| l < 0) {
                            continue;
                        }
                        land.extend_from_slice(&x[t * width..(t + 1) * width]);
                        for (j, &l) in here.iter().enumerate() {
                            if l >= 0 {
                                tally[l as usize] += 1;
                            }
                            let wt = weights.get(t * k + j).copied().unwrap_or(0.0);
                            landed_w.push(if l >= 0 { wt } else { 0.0 });
                        }
                        slot_of.extend(here);
                        from.push((s * tokens + t) as i32);
                    }
                }
                land.resize(cap * width, f32::NAN);
                slot_of.resize(cap * k, -1);
                landed_w.resize(cap * k, 0.0);
                from.resize(cap, -1);
                self.put(rows, cap, land)?;
                self.put_ints(local, cap, k, slot_of);
                if let Some(rw) = row_weights {
                    self.put(rw, cap, landed_w)?;
                }
                self.put_ints(origin, cap, 1, from);
                self.put_ints(counts, slots, 1, tally);
                Ok(())
            }
            Collective::Combine {
                y,
                origin,
                local,
                top_k,
                routed,
                ..
            } => {
                let ys = self.floats(y)?;
                let k = top_k as usize;
                let post = Post::Outputs {
                    y: ys.v.to_vec(),
                    origin: self.ints(origin)?.v.to_vec(),
                    local: self.ints(local)?.v.to_vec(),
                };
                let width = ys.width;
                let all = group.exchange.all(group.rank, post);
                let cap = match &*all[group.rank] {
                    Post::Outputs { origin, .. } => origin.len(),
                    Post::Tokens { .. } => 0,
                };
                let tokens = cap / all.len();
                // This rank's tokens, `tokens` the most any rank brought:
                // its rows are read off the origins every rank returns.
                let mine = self.positions.len();
                if mine > tokens {
                    return Err(format!(
                        "{mine} tokens past the {tokens} a dispatch carried"
                    ));
                }
                let mut out = vec![f32::NAN; mine * k * width];
                for (d, post) in all.iter().enumerate() {
                    let Post::Outputs { y, origin, local } = &**post else {
                        return Err(format!("rank {d} fired another collective"));
                    };
                    for (i, &o) in origin.iter().enumerate() {
                        if o < 0 || o as usize / tokens != group.rank {
                            continue;
                        }
                        let t = o as usize % tokens;
                        for j in 0..k {
                            if local[i * k + j] < 0 {
                                continue;
                            }
                            let src = &y[(i * k + j) * width..(i * k + j + 1) * width];
                            out[(t * k + j) * width..(t * k + j + 1) * width].copy_from_slice(src);
                        }
                    }
                }
                self.put(routed, mine * k, out)
            }
            _ => Err(format!("`{}` has no host forward", Operands::name(op))),
        }
    }
}

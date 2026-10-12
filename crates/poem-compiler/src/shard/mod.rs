//! Tensor parallelism: a model is written at its whole size, its weights and
//! caches state how ranks split them, and this pass turns that one trace into
//! the trace each of `world` ranks runs, inserting the collectives a split
//! value needs where an op wants it whole.
//!
//! Every value is replicated, split across the ranks along its last axis, or
//! a partial sum each rank holds a term of. A block of heads fewer than the
//! ranks is cut once per head, each head then held by a group of ranks in a
//! row (the kv heads of grouped-query attention, whose query heads the ranks
//! split one more level). An op's rule (`rules.rs`) says which of these it
//! takes and what it yields; a partial or split value that reaches an op
//! wanting it whole is all-reduced or all-gathered right after the op that
//! made it.

mod rules;
#[cfg(test)]
mod tests;

use std::collections::BTreeMap;

use poem_ir::{
    CacheRow, Collective, Def, Dim, Node, Operands, Operation, Shard, Trace, Ty, ValueDecl, ValueId,
};

#[derive(Debug, thiserror::Error)]
#[error("`{trace}` does not split {world} ways: {why}")]
pub struct Unshardable {
    pub trace: String,
    pub world: u32,
    pub why: String,
}

/// Whether the runtime reads a model's answer from `seam`, which every rank
/// then holds whole: every export but the attention scores, a tap each rank
/// reads its own share of like the other seams.
fn output(seam: &str) -> bool {
    seam != poem::seam::SCORES.name && crate::EXPORT_SEAMS.contains(&seam)
}

/// How the ranks hold a value.
#[derive(Clone, Debug, PartialEq)]
enum Dist {
    Whole,
    /// Split along the last axis; each block is split on its own, so a rank
    /// holds one share of every block.
    Split(Vec<Block>),
    /// Each rank holds one term of a sum.
    Partial,
}

/// A block of a split value's last axis: its whole width, and how many ways
/// the ranks cut it.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Block {
    width: u64,
    parts: u64,
}

impl Dist {
    /// Split as one block `width` wide, cut once per rank.
    fn even(width: u64, world: u32) -> Dist {
        Dist::Split(vec![Block {
            width,
            parts: u64::from(world),
        }])
    }

    /// Whether every block is cut once per rank.
    fn is_even(&self, world: u32) -> bool {
        match self {
            Dist::Split(blocks) => blocks.iter().all(|b| b.parts == u64::from(world)),
            _ => true,
        }
    }
}

/// The trace each of `world` ranks runs.
pub fn shard(trace: Trace, world: u32) -> Result<Trace, Unshardable> {
    if world == 1 {
        return Ok(trace);
    }
    let name = trace.name.clone();
    Pass::new(trace, world).run().map_err(|why| Unshardable {
        trace: name,
        world,
        why,
    })
}

struct Pass {
    world: u32,
    trace: Trace,
    dists: Vec<Option<Dist>>,
    /// Each node of the input trace, rewritten for one rank, and the
    /// collectives that follow it.
    nodes: Vec<Option<Node>>,
    after: Vec<Vec<Node>>,
    whole: BTreeMap<ValueId, ValueId>,
}

impl Pass {
    fn new(trace: Trace, world: u32) -> Pass {
        let n = trace.nodes.len();
        Pass {
            world,
            dists: vec![None; trace.values.len()],
            nodes: vec![None; n],
            after: vec![Vec::new(); n],
            whole: BTreeMap::new(),
            trace,
        }
    }

    fn run(mut self) -> Result<Trace, String> {
        for at in 0..self.trace.nodes.len() {
            let node = self.trace.nodes[at].clone();
            if matches!(node.op, Operation::Collective(_)) {
                return Err("the trace already states a collective".into());
            }
            let (op, outs) = rules::apply(&mut self, &node.op)?;
            let mut ids = Vec::new();
            op.outputs(&mut ids);
            for (id, dist) in ids.into_iter().zip(outs) {
                self.dists[id.0 as usize] = Some(dist);
            }
            self.nodes[at] = Some(Node { op, ..node });
        }
        for s in 0..self.trace.seams.len() {
            for v in 0..self.trace.seams[s].values.len() {
                let id = self.trace.seams[s].values[v];
                let answer = output(&self.trace.seams[s].seam);
                let held = self.dist(id)?;
                if held == Dist::Partial || answer && held != Dist::Whole {
                    self.trace.seams[s].values[v] = self.whole(id)?;
                }
            }
        }
        self.finish()
    }

    fn dist(&self, v: ValueId) -> Result<Dist, String> {
        match &self.trace.values[v.0 as usize].def {
            Def::Input(_) | Def::Weight(_) | Def::Cache(_) => Ok(Dist::Whole),
            Def::Op(_) => self.dists[v.0 as usize]
                .clone()
                .ok_or_else(|| format!("value {} is read before it is made", v.0)),
            Def::Merge(arms) => {
                let mut seen: Option<Dist> = None;
                for (arm, _) in arms {
                    let d = self.dist(*arm)?;
                    match &seen {
                        Some(s) if *s != d => {
                            return Err(format!("the arms of merge {} are held two ways", v.0));
                        }
                        _ => seen = Some(d),
                    }
                }
                Ok(seen.unwrap_or(Dist::Whole))
            }
        }
    }

    /// How the ranks split weight `v`: the axis and its blocks.
    fn weight_cut(&self, v: ValueId) -> Result<Option<(u32, Vec<Block>)>, String> {
        let Def::Weight(p) = self.trace.values[v.0 as usize].def else {
            return Ok(None);
        };
        let param = &self.trace.params[p as usize];
        match &param.shard {
            Shard::Cut {
                axis,
                segments,
                heads,
            } => {
                let blocks = segments
                    .iter()
                    .enumerate()
                    .map(|(i, width)| {
                        Ok(Block {
                            width: *width,
                            parts: Shard::parts(heads.get(i).copied(), u64::from(self.world))
                                .map_err(|why| format!("`{}`: {why}", param.name))?,
                        })
                    })
                    .collect::<Result<_, String>>()?;
                Ok(Some((*axis, blocks)))
            }
            Shard::Replicated => Ok(None),
        }
    }

    /// The ways the ranks cut the kv cache `v`'s heads, if they split it.
    fn cache_parts(&self, v: ValueId) -> Result<Option<u64>, String> {
        let Def::Cache(c) = self.trace.values[v.0 as usize].def else {
            return Ok(None);
        };
        match &self.trace.caches[c as usize] {
            CacheRow::Kv {
                name,
                planes,
                head_dim,
                shard: Shard::Cut { .. },
                ..
            } => kv_parts(planes, *head_dim, u64::from(self.world))
                .map(Some)
                .map_err(|why| format!("`{name}`: {why}")),
            _ => Ok(None),
        }
    }

    /// Whether the attention reading `plan` reads a cache the ranks split.
    fn planned_kv_split(&self, plan: ValueId) -> bool {
        let mut ins = Vec::new();
        self.trace.nodes.iter().any(|node| {
            ins.clear();
            node.op.inputs(&mut ins);
            ins.contains(&plan) && ins.iter().any(|v| self.cache_split(*v) == Some(true))
        })
    }

    /// Whether a grouped matmul reading `routes` groups a row the ranks split.
    fn grouped_row_split(&self, routes: ValueId) -> bool {
        self.trace.nodes.iter().any(|node| match &node.op {
            Operation::Linear(poem_ir::Linear::MatmulGrouped { x, routes: r, .. }) => {
                *r == routes && matches!(self.dist(*x), Ok(Dist::Split(_)))
            }
            _ => false,
        })
    }

    /// The op that makes `v`, for an error to name.
    fn maker(&self, v: ValueId) -> &'static str {
        match self.trace.values[v.0 as usize].def {
            Def::Op(i) => self
                .trace
                .nodes
                .get(i as usize)
                .map_or("?", |node| node.op.name()),
            _ => "no op",
        }
    }

    fn is_weight(&self, v: ValueId) -> bool {
        matches!(self.trace.values[v.0 as usize].def, Def::Weight(_))
    }

    fn cache_split(&self, v: ValueId) -> Option<bool> {
        let Def::Cache(c) = self.trace.values[v.0 as usize].def else {
            return None;
        };
        Some(matches!(
            &self.trace.caches[c as usize],
            CacheRow::Kv {
                shard: Shard::Cut { .. },
                ..
            } | CacheRow::State {
                shard: Shard::Cut { .. },
                ..
            }
        ))
    }

    fn width(&self, v: ValueId) -> Option<u64> {
        match &self.trace.values[v.0 as usize].ty {
            Ty::Tensor { shape, .. } => match shape.last() {
                Some(Dim::Const(w)) => Some(*w),
                _ => None,
            },
            Ty::Struct(_) => None,
        }
    }

    /// `v` made whole on every rank: itself, or the output of the
    /// collective that follows the op that made it.
    fn whole(&mut self, v: ValueId) -> Result<ValueId, String> {
        let dist = self.dist(v)?;
        if dist == Dist::Whole {
            return Ok(v);
        }
        if let Some(seen) = self.whole.get(&v) {
            return Ok(*seen);
        }
        let Def::Op(at) = self.trace.values[v.0 as usize].def else {
            return Err(format!("value {} is held {dist:?} and no op makes it", v.0));
        };
        let out = ValueId(self.trace.values.len() as u32);
        self.trace.values.push(ValueDecl {
            def: Def::Op(u32::MAX),
            ty: self.trace.values[v.0 as usize].ty.clone(),
        });
        self.dists.push(Some(Dist::Whole));
        let op = match dist {
            Dist::Partial => Collective::AllReduce {
                buf: v,
                buf_out: out,
            },
            Dist::Split(blocks) if blocks.len() == 1 && dist.is_even(self.world) => {
                Collective::AllGather { x: v, y: out }
            }
            Dist::Split(blocks) => {
                return Err(format!(
                    "value {} is split in blocks {blocks:?}, which no gather puts back in order",
                    v.0
                ));
            }
            Dist::Whole => unreachable!(),
        };
        let guard = self.trace.nodes[at as usize].guard.clone();
        let layer = self.trace.nodes[at as usize].layer;
        self.after[at as usize].push(Node {
            op: Operation::Collective(op),
            guard,
            layer,
        });
        self.whole.insert(v, out);
        Ok(out)
    }

    fn finish(mut self) -> Result<Trace, String> {
        for v in 0..self.trace.values.len() {
            if matches!(self.trace.values[v].def, Def::Merge(_)) {
                self.dists[v] = Some(self.dist(ValueId(v as u32))?);
            }
        }
        let before = self.trace.nodes.len();
        let mut nodes = Vec::with_capacity(before);
        let mut landed = vec![0u32; before];
        for (at, node) in self.nodes.into_iter().enumerate() {
            landed[at] = nodes.len() as u32;
            nodes.push(node.expect("every node was rewritten"));
            nodes.extend(std::mem::take(&mut self.after[at]));
        }
        for decl in &mut self.trace.values {
            if let Def::Op(i) = &mut decl.def
                && (*i as usize) < before
            {
                *i = landed[*i as usize];
            }
        }
        let mut outs = Vec::new();
        for (at, node) in nodes.iter().enumerate() {
            if matches!(node.op, Operation::Collective(_)) {
                outs.clear();
                node.op.outputs(&mut outs);
                for v in &outs {
                    self.trace.values[v.0 as usize].def = Def::Op(at as u32);
                }
            }
        }
        self.trace.nodes = nodes;

        let world = u64::from(self.world);
        // Each param's cut extent, whole and one rank's share.
        let mut shares = vec![None; self.trace.params.len()];
        for (at, param) in self.trace.params.iter_mut().enumerate() {
            if let Shard::Cut {
                axis,
                segments,
                heads,
            } = &mut param.shard
            {
                let whole: u64 = segments.iter().sum();
                for (i, segment) in segments.iter_mut().enumerate() {
                    let parts = Shard::parts(heads.get(i).copied(), world)
                        .map_err(|why| format!("`{}`: {why}", param.name))?;
                    *segment = split(*segment, parts, &param.name)?;
                }
                let local: u64 = segments.iter().sum();
                param.shape[*axis as usize] = local;
                shares[at] = Some((*axis as usize, whole, local));
            }
        }
        for (v, decl) in self.trace.values.iter_mut().enumerate() {
            let Ty::Tensor { shape, .. } = &mut decl.ty else {
                continue;
            };
            match (&decl.def, self.dists.get(v)) {
                (_, Some(Some(Dist::Split(blocks)))) => {
                    let local = blocks
                        .iter()
                        .map(|b| split(b.width, b.parts, &format!("value {v}")))
                        .sum::<Result<u64, String>>()?;
                    if let Some(Dim::Const(extent)) = shape.last_mut() {
                        *extent = local;
                    }
                }
                // A weight's value may be wider or narrower than the plane
                // its param stores; it keeps the share its param keeps.
                (Def::Weight(p), _) => {
                    if let Some((axis, whole, local)) = shares[*p as usize]
                        && let Some(Dim::Const(extent)) = shape.get_mut(axis)
                    {
                        *extent = *extent * local / whole;
                    }
                }
                _ => {}
            }
        }
        for row in &mut self.trace.caches {
            match row {
                CacheRow::Kv {
                    name,
                    planes,
                    head_dim,
                    shard: Shard::Cut { segments, .. },
                    ..
                } => {
                    let parts = kv_parts(planes, *head_dim, world)
                        .map_err(|why| format!("`{name}`: {why}"))?;
                    for plane in planes.iter_mut() {
                        *plane = split(*plane, parts, name)?;
                    }
                    segments.clone_from(planes);
                }
                CacheRow::State {
                    name,
                    slab,
                    shard: Shard::Cut { axis, segments, .. },
                    ..
                } => {
                    let extent = &mut slab[*axis as usize];
                    *extent = split(*extent, world, name)?;
                    *segments = vec![*extent];
                }
                _ => {}
            }
        }
        Ok(self.trace)
    }
}

/// The ways `world` ranks cut a kv cache whose planes hold heads
/// `head_dim` wide: once per rank, or once per head when the ranks outnumber
/// them.
fn kv_parts(planes: &[u64], head_dim: u32, world: u64) -> Result<u64, String> {
    let head_dim = u64::from(head_dim);
    let heads = planes
        .iter()
        .map(|plane| (head_dim > 0 && plane.is_multiple_of(head_dim)).then(|| plane / head_dim))
        .min()
        .flatten();
    Shard::parts(heads, world)
}

fn split(whole: u64, world: u64, name: &str) -> Result<u64, String> {
    if whole.is_multiple_of(world) {
        Ok(whole / world)
    } else {
        Err(format!("`{name}`: {whole} does not split {world} ways"))
    }
}

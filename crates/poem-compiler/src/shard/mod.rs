//! Tensor parallelism: a model is written at its whole size, its weights and
//! caches state how ranks split them, and this pass turns that one trace into
//! the trace each of `world` ranks runs, inserting the collectives a split
//! value needs where an op wants it whole.
//!
//! Every value is replicated, split across the ranks along its last axis, or
//! a partial sum each rank holds a term of. An op's rule (`rules.rs`) says
//! which of these it takes and what it yields; a partial or split value that
//! reaches an op wanting it whole is all-reduced or all-gathered right after
//! the op that made it.

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

/// The seams the runtime reads a model's answer from, which every rank holds
/// whole; the rest are taps a rank reads its own share of.
const OUTPUTS: &[&str] = &["out", "mtp", "mtp.drafts", "velocity", "hidden", "pixels"];

/// How the ranks hold a value.
#[derive(Clone, Debug, PartialEq)]
enum Dist {
    Whole,
    /// Split along the last axis; each block of `segments` (whole widths) is
    /// split on its own, so a rank holds one share of every block.
    Split(Vec<u64>),
    /// Each rank holds one term of a sum.
    Partial,
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
                let output = OUTPUTS.contains(&self.trace.seams[s].seam.as_str());
                let held = self.dist(id)?;
                if held == Dist::Partial || output && held != Dist::Whole {
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
    fn weight_cut(&self, v: ValueId) -> Option<(u32, Vec<u64>)> {
        let Def::Weight(p) = self.trace.values[v.0 as usize].def else {
            return None;
        };
        match &self.trace.params[p as usize].shard {
            Shard::Cut { axis, segments } => Some((*axis, segments.clone())),
            Shard::Replicated => None,
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
            Dist::Split(segments) if segments.len() == 1 => Collective::AllGather { x: v, y: out },
            Dist::Split(segments) => {
                return Err(format!(
                    "value {} is split in blocks {segments:?}, which no gather puts back in order",
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
        for (v, decl) in self.trace.values.iter_mut().enumerate() {
            let cut = match (&decl.def, self.dists.get(v)) {
                (_, Some(Some(Dist::Split(_)))) => Some(match &decl.ty {
                    Ty::Tensor { shape, .. } => shape.len().saturating_sub(1),
                    Ty::Struct(_) => 0,
                }),
                (Def::Weight(p), _) => match &self.trace.params[*p as usize].shard {
                    Shard::Cut { axis, .. } => Some(*axis as usize),
                    Shard::Replicated => None,
                },
                _ => None,
            };
            if let (Some(axis), Ty::Tensor { shape, .. }) = (cut, &mut decl.ty)
                && let Some(Dim::Const(extent)) = shape.get_mut(axis)
            {
                *extent /= world;
            }
        }
        for param in &mut self.trace.params {
            if let Shard::Cut { axis, segments } = &mut param.shard {
                for segment in segments.iter_mut() {
                    *segment = split(*segment, world, &param.name)?;
                }
                param.shape[*axis as usize] = segments.iter().sum();
            }
        }
        for row in &mut self.trace.caches {
            match row {
                CacheRow::Kv {
                    name,
                    planes,
                    shard: Shard::Cut { segments, .. },
                    ..
                } => {
                    for plane in planes.iter_mut() {
                        *plane = split(*plane, world, name)?;
                    }
                    segments.clone_from(planes);
                }
                CacheRow::State {
                    name,
                    slab,
                    shard: Shard::Cut { axis, segments },
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

fn split(whole: u64, world: u64, name: &str) -> Result<u64, String> {
    if whole.is_multiple_of(world) {
        Ok(whole / world)
    } else {
        Err(format!("`{name}`: {whole} does not split {world} ways"))
    }
}

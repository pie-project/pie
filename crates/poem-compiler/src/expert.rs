//! Expert parallelism: each of `ep` ranks runs the model on its own tokens,
//! and a MoE layer's routed experts are spread across them. This pass turns
//! the one trace into the trace every rank runs: around each layer's routed
//! experts (the selects that read the layer's routes, and the row ops between
//! them) it inserts a [`Collective::Dispatch`] that sends each token to the
//! ranks holding its experts and a [`Collective::Combine`] that brings the
//! expert rows back in their routes' order. Everything outside, the route,
//! the shared expert and the weighted sum, runs where it ran.
//!
//! A rank holds `slots` of a layer's experts: the experts over the ranks,
//! rounded up, plus `extra` replica slots. Which ones is a load decision
//! (the engine's placement); the trace only states how many. A layer whose
//! experts do not take this form keeps every expert on every rank, each rank
//! computing its own tokens' routes: the program runs as written.

use std::collections::{BTreeMap, BTreeSet};

use poem_ir::{
    Collective, Def, Dim, Dtype, ExpertSlots, Linear, Node, Operands, Operation, Trace, Ty,
    ValueDecl, ValueId,
};

use crate::shard::Unshardable;

/// The trace each of `ep` ranks runs, `extra` replica slots on each.
pub fn lower(trace: Trace, ep: u32, extra: u32) -> Result<Trace, Unshardable> {
    if ep == 1 {
        return Ok(trace);
    }
    let name = trace.name.clone();
    Pass { trace, ep, extra }.run().map_err(|why| Unshardable {
        trace: name,
        world: ep,
        why,
    })
}

struct Pass {
    trace: Trace,
    ep: u32,
    extra: u32,
}

/// One layer's routed experts, as found in the trace.
struct Layer {
    experts: u32,
    top_k: u32,
    routes: ValueId,
    weights: ValueId,
    /// The token rows every entry select reads.
    x: ValueId,
    /// The nodes between dispatch and combine, ascending: the selects and
    /// the row ops between them.
    nodes: BTreeSet<usize>,
    /// The values those nodes make.
    made: BTreeSet<ValueId>,
    /// The values made inside and read outside, in the order they are made.
    exits: Vec<ValueId>,
    /// Whether an op inside reads the route weights.
    weighted: bool,
    /// The params of the banks (and their per-expert companions) the
    /// selects read.
    banks: BTreeSet<u32>,
}

impl Pass {
    fn run(mut self) -> Result<Trace, String> {
        let layers: Vec<Layer> = (0..self.trace.nodes.len())
            .filter_map(|at| self.layer(at))
            .collect();
        if layers.is_empty() {
            return Ok(self.trace);
        }
        let mut before: BTreeMap<usize, Vec<Node>> = BTreeMap::new();
        let mut after: BTreeMap<usize, Vec<Node>> = BTreeMap::new();
        // Layers that read one bank (a drafter block run at each draft step)
        // share its placement, so they take one layer's number.
        let mut numbered: BTreeMap<u32, u32> = BTreeMap::new();
        let mut next = 0;
        for layer in &layers {
            let index = match layer.banks.iter().find_map(|p| numbered.get(p)) {
                Some(index) => *index,
                None => {
                    next += 1;
                    next - 1
                }
            };
            numbered.extend(layer.banks.iter().map(|p| (*p, index)));
            self.lower_layer(index, layer, &mut before, &mut after)?;
        }
        self.splice(before, after);
        Ok(self.trace)
    }

    fn ty(&self, v: ValueId) -> &Ty {
        &self.trace.values[v.0 as usize].ty
    }

    fn rows(&self, v: ValueId) -> Option<Dim> {
        match self.ty(v) {
            Ty::Tensor { shape, .. } => shape.first().copied(),
            Ty::Struct(_) => None,
        }
    }

    fn param(&self, v: ValueId) -> Option<u32> {
        match self.trace.values[v.0 as usize].def {
            Def::Weight(p) => Some(p),
            _ => None,
        }
    }

    /// The routed experts of the MoE layer whose route is node `at`, where
    /// they take the form this pass lowers.
    fn layer(&self, at: usize) -> Option<Layer> {
        let Operation::Linear(Linear::MoeRoute {
            routes, weights, ..
        }) = &self.trace.nodes[at].op
        else {
            return None;
        };
        let (routes, weights) = (*routes, *weights);
        // A sinked route is wider than its pick, its sink slots planes of the
        // banks past the routed experts: the experts a dispatch places are
        // the banks' planes, each token's routes the route's width.
        let top_k = match self.ty(routes) {
            Ty::Tensor { shape, .. } => match shape.get(1) {
                Some(Dim::Const(width)) => u32::try_from(*width).ok()?,
                _ => return None,
            },
            Ty::Struct(_) => return None,
        };
        let routed = Dim::TokensTimes(top_k);
        let mut experts = None::<u32>;
        let mut nodes = BTreeSet::new();
        let mut made = BTreeSet::new();
        let mut x = None;
        let mut banks = BTreeSet::new();
        let mut weighted = false;
        let mut ins = Vec::new();
        let mut outs = Vec::new();
        for (i, node) in self.trace.nodes.iter().enumerate().skip(at + 1) {
            ins.clear();
            node.op.inputs(&mut ins);
            match &node.op {
                Operation::Linear(Linear::MoeSelect {
                    x: input,
                    bank,
                    bias,
                    routes: r,
                    input_scales,
                    y,
                    ..
                }) if *r == routes => {
                    if !made.contains(input) {
                        if x.is_some_and(|seen| seen != *input) {
                            return None;
                        }
                        x = Some(*input);
                    }
                    for companion in [Some(*bank), *bias, *input_scales].into_iter().flatten() {
                        let p = self.param(companion)?;
                        let planes =
                            u32::try_from(*self.trace.params[p as usize].shape.first()?).ok()?;
                        let planes = match self.trace.params[p as usize].experts {
                            Some(cut) => cut.experts,
                            None => planes,
                        };
                        if experts.is_some_and(|seen| seen != planes) {
                            return None;
                        }
                        experts = Some(planes);
                        banks.insert(p);
                    }
                    nodes.insert(i);
                    made.insert(*y);
                }
                _ if ins.iter().any(|v| made.contains(v)) => {
                    // A row op between the selects: every output on the
                    // routed rows, and nothing read but those rows, the
                    // route weights and weights.
                    outs.clear();
                    node.op.outputs(&mut outs);
                    let rowwise = outs.iter().all(|v| self.rows(*v) == Some(routed))
                        && ins
                            .iter()
                            .all(|v| made.contains(v) || *v == weights || self.param(*v).is_some());
                    let between = matches!(
                        node.op,
                        Operation::Linear(
                            Linear::Glu { .. } | Linear::Relu2 { .. } | Linear::GeluTanh { .. }
                        ) | Operation::Elemwise(_)
                    );
                    if !(rowwise && between) {
                        continue;
                    }
                    weighted |= ins.contains(&weights);
                    nodes.insert(i);
                    made.extend(outs.iter().copied());
                }
                _ => {}
            }
        }
        let x = x?;
        let experts = experts?;
        // Only the selects and row ops read the routes inside; what reads a
        // value made inside from outside reads it through the combine.
        let mut exits = Vec::new();
        for (i, node) in self.trace.nodes.iter().enumerate() {
            if nodes.contains(&i) {
                continue;
            }
            ins.clear();
            node.op.inputs(&mut ins);
            for v in &ins {
                if made.contains(v) && !exits.contains(v) {
                    exits.push(*v);
                }
            }
        }
        for seam in &self.trace.seams {
            for v in &seam.values {
                if made.contains(v) && !exits.contains(v) {
                    exits.push(*v);
                }
            }
        }
        if exits.is_empty() || exits.iter().any(|v| self.rows(*v) != Some(routed)) {
            return None;
        }
        Some(Layer {
            experts,
            top_k,
            routes,
            weights,
            x,
            nodes,
            made,
            exits,
            weighted,
            banks,
        })
    }

    fn fresh(&mut self, shape: Vec<Dim>, dtype: Dtype) -> ValueId {
        let id = ValueId(self.trace.values.len() as u32);
        self.trace.values.push(ValueDecl {
            def: Def::Op(u32::MAX),
            ty: Ty::Tensor { shape, dtype },
        });
        id
    }

    fn lower_layer(
        &mut self,
        index: u32,
        layer: &Layer,
        before: &mut BTreeMap<usize, Vec<Node>>,
        after: &mut BTreeMap<usize, Vec<Node>>,
    ) -> Result<(), String> {
        let (ep, k) = (self.ep, layer.top_k);
        let slots = layer.experts.div_ceil(ep) + self.extra;
        let Ty::Tensor {
            shape,
            dtype: x_dtype,
        } = self.ty(layer.x).clone()
        else {
            return Err("a routed select reads a struct".into());
        };
        let hidden = shape.get(1).copied().unwrap_or(Dim::Const(1));
        let received = Dim::TokensTimes(ep);
        let rows = self.fresh(vec![received, hidden], x_dtype);
        let local = self.fresh(vec![received, Dim::Const(u64::from(k))], Dtype::I32);
        let row_weights = layer
            .weighted
            .then(|| self.fresh(vec![received, Dim::Const(u64::from(k))], Dtype::F32));
        let origin = self.fresh(vec![received], Dtype::I32);
        let counts = self.fresh(vec![Dim::Const(u64::from(slots))], Dtype::I32);

        let first = *layer.nodes.first().expect("a layer holds a select");
        let guard = self.trace.nodes[first].guard.clone();
        let at_layer = self.trace.nodes[first].layer;
        before.entry(first).or_default().push(Node {
            op: Operation::Collective(Collective::Dispatch {
                x: layer.x,
                routes: layer.routes,
                weights: layer.weighted.then_some(layer.weights),
                ep,
                experts: layer.experts,
                slots,
                top_k: k,
                layer: index,
                rows,
                local,
                row_weights,
                origin,
                counts,
            }),
            guard,
            layer: at_layer,
        });

        // Inside: the token rows are the received rows, the routes their
        // local slots, the route weights the received ones; every value
        // made inside is `ep` times as tall.
        let mut swap = BTreeMap::from([(layer.x, rows), (layer.routes, local)]);
        if let Some(w) = row_weights {
            swap.insert(layer.weights, w);
        }
        for &i in &layer.nodes {
            let op = &self.trace.nodes[i].op;
            self.trace.nodes[i].op =
                crate::tree::remap(op, &|v| swap.get(&v).copied().unwrap_or(v));
        }
        for v in &layer.made {
            if let Ty::Tensor { shape, .. } = &mut self.trace.values[v.0 as usize].ty
                && let Some(Dim::TokensTimes(n)) = shape.first_mut()
            {
                *n *= ep;
            }
        }

        // Out: each exit read through a combine right after the op that
        // makes it, at the type it had.
        let mut back = BTreeMap::new();
        for &exit in &layer.exits {
            let Def::Op(maker) = self.trace.values[exit.0 as usize].def else {
                return Err(format!("value {} is made inside and by no op", exit.0));
            };
            let Ty::Tensor { shape, dtype } = self.ty(exit).clone() else {
                return Err("a routed row is a struct".into());
            };
            let mut whole = shape;
            whole[0] = Dim::TokensTimes(k);
            let routed = self.fresh(whole, dtype);
            let node = &self.trace.nodes[maker as usize];
            let (guard, at_layer) = (node.guard.clone(), node.layer);
            after.entry(maker as usize).or_default().push(Node {
                op: Operation::Collective(Collective::Combine {
                    y: exit,
                    origin,
                    local,
                    ep,
                    top_k: k,
                    routed,
                }),
                guard,
                layer: at_layer,
            });
            back.insert(exit, routed);
        }
        for i in 0..self.trace.nodes.len() {
            if layer.nodes.contains(&i) {
                continue;
            }
            let op = &self.trace.nodes[i].op;
            let mut ins = Vec::new();
            op.inputs(&mut ins);
            if ins.iter().any(|v| back.contains_key(v)) {
                self.trace.nodes[i].op =
                    crate::tree::remap(op, &|v| back.get(&v).copied().unwrap_or(v));
            }
        }
        for seam in &mut self.trace.seams {
            for v in &mut seam.values {
                if let Some(routed) = back.get(v) {
                    *v = *routed;
                }
            }
        }

        // The banks: `slots` planes on each rank, and so the planes stored
        // beside a bank (`{bank}.scales`, ...), which hold one row an expert.
        let mut banks = layer.banks.clone();
        for &p in &layer.banks {
            let name = self.trace.params[p as usize].name.clone();
            for (at, param) in self.trace.params.iter().enumerate() {
                let beside = param
                    .name
                    .strip_prefix(name.as_str())
                    .is_some_and(|rest| rest.starts_with('.'));
                if beside && param.shape.first() == Some(&u64::from(layer.experts)) {
                    banks.insert(at as u32);
                }
            }
        }
        for &p in &banks {
            let param = &mut self.trace.params[p as usize];
            if param.experts.is_some() {
                continue;
            }
            if param.shape.first() != Some(&u64::from(layer.experts)) {
                return Err(format!(
                    "`{}` is read as a bank of {} experts and is {:?}",
                    param.name, layer.experts, param.shape
                ));
            }
            param.shape[0] = u64::from(slots);
            param.experts = Some(ExpertSlots {
                layer: index,
                ep,
                experts: layer.experts,
                slots,
            });
        }
        for decl in &mut self.trace.values {
            if let Def::Weight(p) = decl.def
                && banks.contains(&p)
                && let Ty::Tensor { shape, .. } = &mut decl.ty
                && let Some(Dim::Const(n)) = shape.first_mut()
                && *n == u64::from(layer.experts)
            {
                *n = u64::from(slots);
            }
        }
        Ok(())
    }

    /// The trace's nodes with each inserted one in its place, every value's
    /// maker renumbered.
    fn splice(
        &mut self,
        mut before: BTreeMap<usize, Vec<Node>>,
        mut after: BTreeMap<usize, Vec<Node>>,
    ) {
        let old = std::mem::take(&mut self.trace.nodes);
        let mut landed = vec![0u32; old.len()];
        let mut nodes = Vec::with_capacity(old.len() + before.len() + after.len());
        for (at, node) in old.into_iter().enumerate() {
            nodes.extend(before.remove(&at).unwrap_or_default());
            landed[at] = nodes.len() as u32;
            nodes.push(node);
            nodes.extend(after.remove(&at).unwrap_or_default());
        }
        for decl in &mut self.trace.values {
            if let Def::Op(i) = &mut decl.def
                && *i != u32::MAX
            {
                *i = landed[*i as usize];
            }
        }
        let mut outs = Vec::new();
        for (at, node) in nodes.iter().enumerate() {
            if matches!(
                node.op,
                Operation::Collective(Collective::Dispatch { .. } | Collective::Combine { .. })
            ) {
                outs.clear();
                node.op.outputs(&mut outs);
                for v in &outs {
                    self.trace.values[v.0 as usize].def = Def::Op(at as u32);
                }
            }
        }
        self.trace.nodes = nodes;
    }
}

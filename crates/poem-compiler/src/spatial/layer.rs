//! The layer program: a trace read as the phases a spatial target runs one decoded token in.
//!
//! A spatial target holds a decoder layer's matrices tiled over a grid of cores: a tile is a
//! row chunk by a column chunk, a control core at the end of each row (the row root) and of each
//! column (the column root). So a vector lives on an [`Axis`]: chunked over the rows (the row
//! roots hold it; the residual stream lives there) or over the columns. A matvec flips the axis
//! its input lives on (a row vector times a tile reduces into the column roots, and back), and
//! everything else runs where its inputs are: elementwise ops at the roots of their axis, the
//! stateful mixers (attention's KV, the delta rule's state) on the tiles themselves, their inputs
//! cast out from the column roots. [`lower`] reads every node of a decoder layer that runs into
//! that machine's [`Phase`]s, assigns each value its axis (a 2-colouring of the dataflow) and
//! refuses by name any node it has no phase for. Layers whose phases are the same share a
//! [`LayerBody`]: a target builds one code path for each.
//!
//! The program is target-neutral: which nodes run (a spatial target serves one token a request
//! at a time: [`single_token`]) is the caller's, and the placement, kernels and tiling of the
//! phases are the target's.

use std::collections::{HashMap, HashSet};

use poem_ir::{
    Attn, Builtin, Def, Dim, Fact, Gather, GluAct, Guard, Linear, Operands, Operation, Ssm, Trace,
    Ty, ValueId,
};
use serde::Serialize;

use crate::analysis::Analysis;

/// Where a vector is chunked: over the grid's rows (the row roots hold it) or its columns.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub enum Axis {
    Row,
    Col,
}

impl Axis {
    /// The other axis: where a matvec of a vector on this one reduces.
    #[must_use]
    pub fn flip(self) -> Axis {
        match self {
            Axis::Row => Axis::Col,
            Axis::Col => Axis::Row,
        }
    }
}

/// A stateful mixer, on the tiles.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub enum Mixer {
    /// Attention over a KV cache: the layer's own (`shared_from` none) or the one layer
    /// `shared_from` appends; over the whole sequence or the last `window` positions.
    Attn {
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        window: Option<u32>,
        shared_from: Option<u32>,
    },
    /// The gated delta rule: a state slice a tile.
    Gdn {
        k_heads: u32,
        v_heads: u32,
        k_dim: u32,
        v_dim: u32,
    },
}

/// An op of a phase: its name, its output's width, the params it reads (by name, a layer's
/// number written `*`, so layers of one body name the same).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Op {
    pub op: &'static str,
    pub width: u32,
    pub params: Vec<String>,
}

/// A cross-column reduction over a segment of a row's columns.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Reduce {
    Sum,
    Max,
    Gather,
}

/// One step of a layer.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub enum Phase {
    /// The roots of `axis` multicast a vector of `width` to the tiles.
    Cast { axis: Axis, width: u32 },
    /// The tiles multiply a vector of width `input` on `from` by the matrices `weights`, each
    /// reduced into the roots of the other axis; `experts`, a routed bank: (experts, top k).
    Matvec {
        from: Axis,
        to: Axis,
        input: u32,
        weights: Vec<Op>,
        experts: Option<(u32, u32)>,
    },
    /// Elementwise ops at the roots of `axis`.
    Root { axis: Axis, ops: Vec<Op> },
    /// A mixer's ops on the tiles (its epilogue ops on its output, such as a sink or a gated
    /// norm, with them).
    Mixer { mixer: Mixer, ops: Vec<Op> },
    /// A reduction across the columns of `axis`'s roots.
    Segment {
        axis: Axis,
        reduce: Reduce,
        width: u32,
    },
    /// A per-token input read from a table (a gather) and the ops on it before it meets the
    /// layer's vectors.
    SideIn { ops: Vec<Op> },
}

/// The phases of a layer, shared by every layer that has the same.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct LayerBody {
    pub phases: Vec<Phase>,
    /// The gate's activation of the layer's gated MLP (its experts' where they are routed).
    pub glu: Option<GluAct>,
}

impl LayerBody {
    /// The body's mixers, in order.
    pub fn mixers(&self) -> impl Iterator<Item = &Mixer> {
        self.phases.iter().filter_map(|p| match p {
            Phase::Mixer { mixer, .. } => Some(mixer),
            _ => None,
        })
    }

    /// Whether any phase has an op named `op`.
    #[must_use]
    pub fn has(&self, op: &str) -> bool {
        self.phases.iter().any(|p| match p {
            Phase::Matvec { weights: o, .. }
            | Phase::Root { ops: o, .. }
            | Phase::Mixer { ops: o, .. }
            | Phase::SideIn { ops: o } => o.iter().any(|o| o.op == op),
            _ => false,
        })
    }
}

/// The model's head: the logits' matvec over `vocab` rows, its matrix the token embedding's
/// (`tied`) or its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Head {
    pub vocab: u32,
    pub tied: bool,
}

/// The trace as a spatial target runs a token: the ops before the first layer (from the token
/// embedding on), each layer's body, the ops after the last.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct LayerProgram {
    pub hidden: u32,
    pub prologue: Vec<Op>,
    pub bodies: Vec<LayerBody>,
    /// Each layer's body, by index into `bodies`.
    pub layers: Vec<usize>,
    pub epilogue: Vec<Op>,
    pub head: Head,
}

impl LayerProgram {
    /// Layer `l`'s body.
    #[must_use]
    pub fn body(&self, l: usize) -> &LayerBody {
        &self.bodies[self.layers[l]]
    }
}

/// Whether a node guarded by `g` runs when every request is one token at a time with no mask,
/// adapter, media, drafts or score capture: what a spatial program serves. A fact that this
/// does not fix (a fresh state, a switch) may take either value.
#[must_use]
pub fn single_token(trace: &Trace, g: &Guard) -> bool {
    let (mut word, mut free) = (0u64, Vec::new());
    for bit in g.referenced_bits() {
        match trace.facts.table.get(usize::from(bit)) {
            Some(Fact::Builtin(Builtin::SingleToken)) => word |= 1 << bit,
            Some(Fact::Builtin(Builtin::Fresh) | Fact::Switch { .. }) => free.push(bit),
            _ => {}
        }
    }
    (0..1u64 << free.len()).any(|a| {
        let w = free
            .iter()
            .enumerate()
            .filter(|(i, _)| a & (1 << i) != 0)
            .fold(word, |w, (_, b)| w | 1 << b);
        g.holds(w)
    })
}

/// The ops outside the layers a program carries, from the token embedding on: the embedding,
/// a per-layer input's projection (an activation, a matvec, a norm), the final norm, the last
/// row, the head, its softcap and the argmax, and the attention's plan (the host's).
const OUTER: &[&str] = &[
    "attn.plan_decode",
    "elemwise.act",
    "elemwise.rmsnorm",
    "elemwise.softcap",
    "gather.embed",
    "gather.rows",
    "linear.matmul",
    "sampling.argmax",
];

type R<T> = Result<T, String>;

/// Where a value is while a layer runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum At {
    /// At the roots of the axis (a matvec's reduction, an elementwise op's result).
    Root(Axis),
    /// On the tiles, chunked over the axis (a cast, a mixer's output).
    Tiles(Axis),
    /// Wherever it is needed: the host's (positions, the attention plan), a side input, a
    /// router's choice (replicated by its gather).
    Any,
}

struct Reader<'t> {
    trace: &'t Trace,
    runs: Vec<bool>,
    hidden: u32,
    /// The layer that appends to each KV cache.
    owner: HashMap<ValueId, u32>,
    layers: &'t [Option<u32>],
}

impl Reader<'_> {
    fn def(&self, v: ValueId) -> Option<&Def> {
        self.trace.values.get(v.0 as usize).map(|d| &d.def)
    }

    fn width(&self, v: ValueId) -> u32 {
        match self.trace.values.get(v.0 as usize).map(|d| &d.ty) {
            Some(Ty::Tensor { shape, .. }) => match shape.last() {
                Some(Dim::Const(n)) => *n as u32,
                _ => 0,
            },
            _ => 0,
        }
    }

    /// The value `v` is when only running nodes run: a node that does not run passes on the
    /// input its output aliases; of a merge's arms, the one a running node makes.
    fn resolve(&self, mut v: ValueId) -> ValueId {
        for _ in 0..self.trace.nodes.len() {
            match self.def(v) {
                Some(Def::Op(n)) if !self.runs[*n as usize] => {
                    let mut pairs = Vec::new();
                    self.trace.nodes[*n as usize].op.aliases(&mut pairs);
                    match pairs.iter().find(|(out, _)| *out == v) {
                        Some(&(_, of)) => v = of,
                        None => return v,
                    }
                }
                Some(Def::Merge(arms)) => {
                    let live: Vec<ValueId> = arms
                        .iter()
                        .map(|(a, _)| self.resolve(*a))
                        .filter(
                            |a| matches!(self.def(*a), Some(Def::Op(n)) if self.runs[*n as usize]),
                        )
                        .collect();
                    match live.as_slice() {
                        [one] => v = *one,
                        _ => return v,
                    }
                }
                _ => return v,
            }
        }
        v
    }

    fn maker(&self, v: ValueId) -> Option<&Operation> {
        match self.def(self.resolve(v)) {
            Some(Def::Op(n)) => Some(&self.trace.nodes[*n as usize].op),
            _ => None,
        }
    }

    /// `p`'s name, layer `l`'s number written `*`.
    fn param(&self, p: u32, l: u32) -> String {
        let l = l.to_string();
        self.trace.params[p as usize]
            .name
            .split('.')
            .map(|s| if s == l { "*" } else { s })
            .collect::<Vec<_>>()
            .join(".")
    }

    /// Where a value made outside layer `l` is: the residual stream (a layer's output, or an
    /// activation of the model's width before the first) at the row roots; else anywhere.
    fn outside(&self, v: ValueId) -> At {
        match self.def(v) {
            Some(Def::Op(n)) if self.layers[*n as usize].is_some() => At::Root(Axis::Row),
            Some(Def::Op(_)) if self.width(v) == self.hidden => At::Root(Axis::Row),
            _ => At::Any,
        }
    }
}

/// One layer's phases as they are read.
struct Body<'r, 't> {
    r: &'r Reader<'t>,
    layer: u32,
    phases: Vec<Phase>,
    at: HashMap<ValueId, At>,
    /// The phase that made each value.
    made: HashMap<ValueId, usize>,
    cast: HashSet<ValueId>,
    /// The input of the last phase where it is a matvec (a matvec of the same input joins it).
    matvec_of: Option<ValueId>,
    glu: Option<GluAct>,
}

impl Body<'_, '_> {
    fn at(&self, v: ValueId) -> At {
        self.at
            .get(&v)
            .copied()
            .unwrap_or_else(|| self.r.outside(v))
    }

    fn set(&mut self, v: ValueId, at: At) {
        self.at.insert(v, at);
        self.made.insert(v, self.phases.len().saturating_sub(1));
    }

    fn push(&mut self, p: Phase) {
        self.phases.push(p);
        self.matvec_of = None;
    }

    /// Cast `v` from the roots of `axis` to the tiles, once; before phase `before` where given
    /// (a mixer's op reading a value made before the mixer began).
    fn cast(&mut self, v: ValueId, axis: Axis, before: Option<usize>) {
        if !self.cast.insert(v) {
            return;
        }
        let width = self.r.width(v);
        let p = Phase::Cast { axis, width };
        match before {
            Some(i) if self.made.get(&v).is_none_or(|&m| m < i) => {
                // one multicast carries every vector cast together
                if let Some(Phase::Cast { axis: a, width: w }) =
                    i.checked_sub(1).map(|j| &mut self.phases[j])
                    && *a == axis
                {
                    *w += width;
                    return;
                }
                self.phases.insert(i, p);
                for m in self.made.values_mut() {
                    if *m >= i {
                        *m += 1;
                    }
                }
            }
            _ => match self.phases.last_mut() {
                Some(Phase::Cast { axis: a, width: w }) if *a == axis => *w += width,
                _ => self.push(p),
            },
        }
    }

    fn op(&self, name: &'static str, ins: &[ValueId], outs: &[ValueId]) -> Op {
        let params = ins
            .iter()
            .filter_map(|&v| match self.r.def(v) {
                Some(Def::Weight(p)) => Some(self.r.param(*p, self.layer)),
                _ => None,
            })
            .collect();
        Op {
            op: name,
            width: outs.first().map_or(0, |&o| self.r.width(o)),
            params,
        }
    }

    /// The activations among `ins` (no weights, caches or host inputs), resolved.
    fn acts(&self, ins: &[ValueId]) -> Vec<ValueId> {
        ins.iter()
            .filter(|&&v| matches!(self.r.def(v), Some(Def::Op(_) | Def::Merge(_))))
            .map(|&v| self.r.resolve(v))
            .collect()
    }

    fn matvec(
        &mut self,
        x: ValueId,
        op: Op,
        experts: Option<(u32, u32)>,
        outs: &[ValueId],
    ) -> R<()> {
        let (from, cast) = match self.at(x) {
            At::Root(a) => (a, true),
            At::Tiles(a) => (a, false),
            At::Any => return Err(format!("{}'s input is on no axis", op.op)),
        };
        if self.matvec_of == Some(x)
            && let Some(Phase::Matvec { weights, .. }) = self.phases.last_mut()
        {
            weights.push(op);
        } else {
            if cast {
                self.cast(x, from, None);
            }
            self.push(Phase::Matvec {
                from,
                to: from.flip(),
                input: self.r.width(x),
                weights: vec![op],
                experts,
            });
            self.matvec_of = Some(x);
        }
        for &o in outs {
            self.set(o, At::Root(from.flip()));
        }
        Ok(())
    }

    /// A mixer's op: its inputs cast from the column roots to the tiles, its outputs on them.
    fn mixer(&mut self, op: Op, kind: Option<Mixer>, acts: &[ValueId], outs: &[ValueId]) -> R<()> {
        let open = match self.phases.last() {
            Some(Phase::Mixer { .. }) => Some(self.phases.len() - 1),
            _ => None,
        };
        for &v in acts {
            match self.at(v) {
                At::Root(Axis::Col) => self.cast(v, Axis::Col, open),
                At::Root(Axis::Row) => {
                    return Err(format!("{} reads a row vector on the tiles", op.op));
                }
                _ => {}
            }
        }
        match self.phases.last_mut() {
            Some(Phase::Mixer { mixer, ops }) => {
                if let Some(k) = kind {
                    *mixer = k;
                }
                ops.push(op);
            }
            _ => {
                let mixer = kind.unwrap_or(Mixer::Gdn {
                    k_heads: 0,
                    v_heads: 0,
                    k_dim: 0,
                    v_dim: 0,
                });
                self.push(Phase::Mixer {
                    mixer,
                    ops: vec![op],
                });
            }
        }
        for &o in outs {
            self.set(o, At::Tiles(Axis::Col));
        }
        Ok(())
    }

    /// An elementwise op: at the roots of its inputs' axis; on the tiles with the mixer whose
    /// output it reads; a side input's where it reads only side inputs.
    fn elementwise(&mut self, op: Op, acts: &[ValueId], outs: &[ValueId]) -> R<()> {
        let mut axis = None;
        let mut tiles = false;
        for &v in acts {
            let a = match self.at(v) {
                At::Root(a) => a,
                At::Tiles(a) => {
                    tiles = true;
                    a
                }
                At::Any => continue,
            };
            if axis.is_some_and(|b| b != a) {
                return Err(format!(
                    "{}: value %{} is needed on rows and columns (no transpose route)",
                    op.op, v.0
                ));
            }
            axis = Some(a);
        }
        if tiles {
            if !matches!(self.phases.last(), Some(Phase::Mixer { .. })) {
                return Err(format!("{} reads the tiles after their mixer", op.op));
            }
            return self.mixer(op, None, acts, outs);
        }
        let Some(axis) = axis else {
            match self.phases.last_mut() {
                Some(Phase::SideIn { ops }) => ops.push(op),
                _ => self.push(Phase::SideIn { ops: vec![op] }),
            }
            for &o in outs {
                self.set(o, At::Any);
            }
            return Ok(());
        };
        if op.op == "linear.moe_route" {
            self.push(Phase::Segment {
                axis,
                reduce: Reduce::Gather,
                width: self.r.width(acts[0]),
            });
        }
        match self.phases.last_mut() {
            Some(Phase::Root { axis: a, ops }) if *a == axis => ops.push(op),
            _ => self.push(Phase::Root {
                axis,
                ops: vec![op],
            }),
        }
        let at = if op_routes(self.phases.last()) {
            At::Any
        } else {
            At::Root(axis)
        };
        for &o in outs {
            self.set(o, at);
        }
        Ok(())
    }

    fn node(&mut self, at: usize) -> R<()> {
        let node = &self.r.trace.nodes[at];
        let (mut ins, mut outs) = (Vec::new(), Vec::new());
        node.op.inputs(&mut ins);
        node.op.outputs(&mut outs);
        let name = node.op.name();
        let op = self.op(name, &ins, &outs);
        let acts = self.acts(&ins);
        match &node.op {
            Operation::Gather(
                Gather::SplitQkv { .. } | Gather::SplitQGate { .. } | Gather::SplitRows { .. },
            ) => {
                let a = acts.first().map_or(At::Any, |&v| self.at(v));
                for &o in &outs {
                    self.set(o, a);
                }
                Ok(())
            }
            Operation::Linear(Linear::Matmul {
                act, logits: false, ..
            }) => self.matvec(self.r.resolve(*act), op, None, &outs),
            Operation::Linear(Linear::MoeSelect { x, routes, .. }) => {
                let experts = match self.r.maker(*routes) {
                    Some(Operation::Linear(Linear::MoeRoute { experts, top_k, .. })) => {
                        Some((*experts, *top_k))
                    }
                    _ => return Err(format!("{name}: its routes are no router's")),
                };
                self.matvec(self.r.resolve(*x), op, experts, &outs)
            }
            Operation::Attn(Attn::Decode {
                plan,
                cache,
                window,
                ..
            }) => {
                let Some(Operation::Attn(Attn::PlanDecode {
                    q_heads,
                    kv_heads,
                    head_dim,
                    ..
                })) = self.r.maker(*plan)
                else {
                    return Err(format!("{name}: its plan is no decode plan"));
                };
                let shared_from = match self.r.owner.get(cache) {
                    Some(&o) if o == self.layer => None,
                    Some(&o) => Some(o),
                    None => return Err(format!("{name} reads a cache no layer appends")),
                };
                let kind = Mixer::Attn {
                    q_heads: *q_heads,
                    kv_heads: *kv_heads,
                    head_dim: *head_dim,
                    window: *window,
                    shared_from,
                };
                self.mixer(op, Some(kind), &acts, &outs)
            }
            Operation::Ssm(Ssm::GatedDelta {
                k_heads,
                v_heads,
                k_dim,
                v_dim,
                ..
            }) => {
                let kind = Mixer::Gdn {
                    k_heads: *k_heads,
                    v_heads: *v_heads,
                    k_dim: *k_dim,
                    v_dim: *v_dim,
                };
                self.mixer(op, Some(kind), &acts, &outs)
            }
            Operation::Attn(Attn::KvAppend { .. } | Attn::Sink { .. })
            | Operation::Ssm(Ssm::CausalConv1d { .. } | Ssm::GdnGates { .. }) => {
                self.mixer(op, None, &acts, &outs)
            }
            Operation::Gather(Gather::Embed { .. } | Gather::Select { .. }) => {
                match self.phases.last_mut() {
                    Some(Phase::SideIn { ops }) => ops.push(op),
                    _ => self.push(Phase::SideIn { ops: vec![op] }),
                }
                for &o in &outs {
                    self.set(o, At::Any);
                }
                Ok(())
            }
            Operation::Elemwise(e)
                if matches!(
                    e.name(),
                    "elemwise.act"
                        | "elemwise.add_bias"
                        | "elemwise.gate_sigmoid_mul"
                        | "elemwise.residual_add"
                        | "elemwise.rmsnorm"
                        | "elemwise.rmsnorm_gated"
                        | "elemwise.rope"
                ) =>
            {
                self.elementwise(op, &acts, &outs)
            }
            Operation::Linear(Linear::Glu { act, .. }) => {
                self.glu = Some(*act);
                self.elementwise(op, &acts, &outs)
            }
            Operation::Linear(
                Linear::MoeRoute { .. } | Linear::MoeCombine { .. } | Linear::MoeBiasSum { .. },
            ) => self.elementwise(op, &acts, &outs),
            _ => Err(format!("{name} has no spatial lowering")),
        }
    }

    fn finish(self) -> R<LayerBody> {
        for p in &self.phases {
            if let Phase::Mixer {
                mixer: Mixer::Gdn { k_heads: 0, .. },
                ops,
            } = p
            {
                let names: Vec<_> = ops.iter().map(|o| o.op).collect();
                return Err(format!(
                    "mixer ops {} with no attention or delta rule",
                    names.join(", ")
                ));
            }
        }
        Ok(LayerBody {
            phases: self.phases,
            glu: self.glu,
        })
    }
}

/// Whether the phase is a router's choice (replicated over the roots by its gather).
fn op_routes(p: Option<&Phase>) -> bool {
    matches!(p, Some(Phase::Root { ops, .. }) if ops.last().is_some_and(|o| o.op == "linear.moe_route"))
}

/// The layer program of `trace`, the nodes `runs` keeps (by guard) only: the decoder layers'
/// phases from their nodes, the ops before and after them from the token embedding (the last
/// plain gather before the first layer: a multimodal trace's encoders run before it) on. A node
/// no phase runs is refused by name: `<trace>: layer L node N: <op> has no spatial lowering`.
pub fn lower(
    trace: &Trace,
    analysis: &Analysis,
    runs: &dyn Fn(&Guard) -> bool,
) -> Result<LayerProgram, String> {
    let name = &trace.name;
    let running: Vec<bool> = trace.nodes.iter().map(|n| runs(&n.guard)).collect();
    let first = analysis.layers.iter().position(Option::is_some);
    let embed = trace
        .nodes
        .iter()
        .enumerate()
        .take(first.unwrap_or(trace.nodes.len()))
        .filter(|(at, n)| {
            running[*at] && matches!(n.op, Operation::Gather(Gather::Embed { weights: None, .. }))
        })
        .map(|(at, _)| at)
        .next_back()
        .ok_or_else(|| format!("{name}: no token embedding"))?;
    let (table, vocab, hidden) = match &trace.nodes[embed].op {
        Operation::Gather(Gather::Embed { table, vocab, .. }) => {
            let hidden = match trace.values.get(table.0 as usize).map(|d| &d.def) {
                Some(Def::Weight(p)) => {
                    trace.params[*p as usize].shape.get(1).copied().unwrap_or(0) as u32
                }
                _ => 0,
            };
            (*table, *vocab, hidden)
        }
        _ => unreachable!("the embedding is a gather"),
    };
    let mut owner = HashMap::new();
    for (at, n) in trace.nodes.iter().enumerate() {
        if let (true, Some(l), Operation::Attn(Attn::KvAppend { cache, .. })) =
            (running[at], analysis.layers[at], &n.op)
        {
            owner.insert(*cache, l);
        }
    }
    let r = Reader {
        trace,
        runs: running,
        hidden,
        owner,
        layers: &analysis.layers,
    };
    let n_layers = analysis.layers.iter().flatten().max().map_or(0, |&l| l + 1);
    let mut bodies: Vec<Body> = (0..n_layers)
        .map(|layer| Body {
            r: &r,
            layer,
            phases: Vec::new(),
            at: HashMap::new(),
            made: HashMap::new(),
            cast: HashSet::new(),
            matvec_of: None,
            glu: None,
        })
        .collect();
    let (mut prologue, mut epilogue, mut head) = (Vec::new(), Vec::new(), None);
    for at in embed..trace.nodes.len() {
        if !r.runs[at] {
            continue;
        }
        let node = &trace.nodes[at];
        let op = node.op.name();
        match analysis.layers[at] {
            Some(l) => bodies[l as usize]
                .node(at)
                .map_err(|e| format!("{name}: layer {l} node {at}: {e}"))?,
            None if !OUTER.contains(&op) => {
                return Err(format!("{name}: node {at}: {op} has no spatial lowering"));
            }
            None if op == "attn.plan_decode" => {}
            None => {
                let (mut ins, mut outs) = (Vec::new(), Vec::new());
                node.op.inputs(&mut ins);
                node.op.outputs(&mut outs);
                let o = Op {
                    op,
                    width: outs.first().map_or(0, |&v| r.width(v)),
                    params: ins
                        .iter()
                        .filter_map(|&v| match r.def(v) {
                            Some(Def::Weight(p)) => Some(trace.params[*p as usize].name.clone()),
                            _ => None,
                        })
                        .collect(),
                };
                if let Operation::Linear(Linear::Matmul {
                    w, logits: true, ..
                }) = &node.op
                {
                    head = Some(Head {
                        vocab: o.width,
                        tied: *w == table,
                    });
                }
                let before = first.is_none_or(|f| at < f);
                if before { &mut prologue } else { &mut epilogue }.push(o);
            }
        }
    }
    let head = head.ok_or_else(|| format!("{name}: no logits"))?;
    if hidden == 0 || vocab == 0 || n_layers == 0 {
        return Err(format!("{name}: no embedding or no layers"));
    }
    let mut out = LayerProgram {
        hidden,
        prologue,
        bodies: Vec::new(),
        layers: Vec::new(),
        epilogue,
        head,
    };
    for (l, b) in bodies.into_iter().enumerate() {
        let body = b.finish().map_err(|e| format!("{name}: layer {l}: {e}"))?;
        let at = match out.bodies.iter().position(|o| *o == body) {
            Some(i) => i,
            None => {
                out.bodies.push(body);
                out.bodies.len() - 1
            }
        };
        out.layers.push(at);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::analyze;

    fn program(name: &str) -> Result<LayerProgram, String> {
        let t = crate::catalog::repository()
            .parse(name)
            .unwrap_or_else(|| panic!("{name} parses"))
            .trace(poem_ir::Platform::Cerebras);
        let a = analyze(&t).unwrap();
        lower(&t, &a, &|g| single_token(&t, g))
    }

    /// A short form of a body: each phase's letter (C cast, M matvec, R root, X mixer, S
    /// segment, I side input) and axis.
    fn form(b: &LayerBody) -> String {
        let ax = |a: &Axis| match a {
            Axis::Row => "r",
            Axis::Col => "c",
        };
        b.phases
            .iter()
            .map(|p| match p {
                Phase::Cast { axis, .. } => format!("C{}", ax(axis)),
                Phase::Matvec { from, weights, .. } => format!("M{}{}", ax(from), weights.len()),
                Phase::Root { axis, ops } => format!("R{}{}", ax(axis), ops.len()),
                Phase::Mixer { ops, .. } => format!("X{}", ops.len()),
                Phase::Segment { axis, .. } => format!("S{}", ax(axis)),
                Phase::SideIn { ops } => format!("I{}", ops.len()),
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Llama: one body; the residual's norm at the rows, q / k / v one matvec into the columns,
    /// RoPE there, cast to the tiles' attention, o back to the rows; the MLP the same way.
    #[test]
    fn a_dense_layer_is_the_unit_snake() {
        let p = program("llama31-8b-bf16-kv-bf16").unwrap();
        assert_eq!(p.bodies.len(), 1);
        assert_eq!(p.layers.len(), 32);
        assert_eq!(
            form(&p.bodies[0]),
            "Rr1 Cr Mr1 Rc1 Cc X2 Mc1 Rr2 Cr Mr1 Rc1 Cc Mc1 Rr1"
        );
        assert_eq!(
            p.bodies[0].mixers().next(),
            Some(&Mixer::Attn {
                q_heads: 32,
                kv_heads: 8,
                head_dim: 128,
                window: None,
                shared_from: None
            })
        );
        assert_eq!((p.hidden, p.head.vocab, p.head.tied), (4096, 128256, false));
    }

    /// Qwen3.5's GDN layer: in_qkvz and in_ba one matvec (they read the same x), the conv,
    /// gates, delta rule and gated norm one mixer. gpt-oss's: biases at the roots, the sinks with
    /// the attention, the router's gather, the experts' banks, their combine at the rows.
    #[test]
    fn gdn_and_routed_layers_have_their_phases() {
        let q = program("qwen35-d0.8b-bf16-kv-bf16").unwrap();
        assert_eq!(
            form(q.body(0)),
            "Rr1 Cr Mr2 Cc X4 Mc1 Rr2 Cr Mr1 Rc1 Cc Mc1 Rr1"
        );
        let g = program("gptoss-20b-bf16-mxfp4-kv-bf16").unwrap();
        assert_eq!(
            form(g.body(0)),
            "Rr1 Cr Mr1 Rc2 Cc X3 Mc1 Rr3 Cr Mr1 Rc1 Sc Rc1 Mr1 Rc1 Cc Mc1 Rr3"
        );
    }

    /// Every shipping model and the Qwen3 MoE miniature lower; their bodies.
    #[test]
    fn shipping_models_lower() {
        for (name, bodies) in [
            ("qwen35-d9b-bf16-kv-bf16", 2),
            ("qwen35-d4b-bf16-kv-bf16", 2),
            ("qwen35-d0.8b-bf16-kv-bf16", 2),
            ("qwen3-8b-bf16-kv-bf16", 1),
            ("llama31-8b-bf16-kv-bf16", 1),
            ("gemma4-e2b-bf16-kv-bf16", 4),
            ("gemma4-e4b-bf16-kv-bf16", 4),
            ("gemma4-12b-bf16-kv-bf16", 2),
            ("gptoss-20b-bf16-mxfp4-kv-bf16", 2),
            ("qwen3-30b-a3b-mini-bf16-kv-bf16", 1),
        ] {
            let p = program(name).unwrap_or_else(|e| panic!("{e}"));
            let forms: Vec<String> = p.bodies.iter().map(form).collect();
            assert_eq!(p.bodies.len(), bodies, "{name}: {forms:#?}");
        }
    }

    /// Routed experts: the router's matvec into the columns, a gather of its logits over the
    /// column roots, the choice; the experts' bank one matvec each way (the miniature keeps 16 of
    /// its 128 experts).
    #[test]
    fn routed_experts_lower_to_a_gather_and_banks() {
        let p = program("qwen3-30b-a3b-mini-bf16-kv-bf16").unwrap();
        let b = &p.bodies[0];
        assert!(b.phases.iter().any(|ph| matches!(
            ph,
            Phase::Segment {
                axis: Axis::Col,
                reduce: Reduce::Gather,
                ..
            }
        )));
        let banks: Vec<_> = b
            .phases
            .iter()
            .filter_map(|ph| match ph {
                Phase::Matvec {
                    experts: Some(e), ..
                } => Some(*e),
                _ => None,
            })
            .collect();
        assert_eq!(banks, vec![(16, 8), (16, 8)]);
    }

    /// Gemma 4's later layers read an earlier layer's cache; per-layer inputs are side inputs.
    #[test]
    fn shared_caches_and_side_inputs() {
        let p = program("gemma4-e2b-bf16-kv-bf16").unwrap();
        let last = p.body(p.layers.len() - 1);
        assert!(matches!(
            last.mixers().next(),
            Some(Mixer::Attn {
                shared_from: Some(_),
                ..
            })
        ));
        assert!(
            last.phases
                .iter()
                .any(|ph| matches!(ph, Phase::SideIn { .. }))
        );
        assert!(p.head.tied);
    }

    /// An op no phase runs is refused by name.
    #[test]
    fn an_op_without_a_phase_is_refused_by_name() {
        let e = program("dsv4-flash-mini-bf16-kv-bf16").unwrap_err();
        assert!(
            e.contains("elemwise.hc_expand has no spatial lowering"),
            "{e}"
        );
    }
}

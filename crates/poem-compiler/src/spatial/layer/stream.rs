//! The stream regime's layer program: a reading that runs whole sequences a step (a diffusion
//! transformer's denoise, an encoder's prefill) read into the phases a spatial target runs.
//!
//! Decode serves one token a request a step; a stream step carries `T` rows, and they flow
//! through each layer's unit one after another with the weights held still. So a stream program
//! has the same phases as the decode layer program ([`super::lower`]), each row going through
//! them as a decode token would, with four differences:
//!
//! - **Row classes.** A step's rows come in classes (a text stream and an image stream, each a
//!   request of its own word): a node runs for the classes whose word its guard holds. Each
//!   class is read on its own, and the classes whose phases (and so whose weights) are the same
//!   share a [`Side`] of the layer: one unit. Classes with their own weights (FLUX's double
//!   blocks: `.on(text)` / `.on(~text)`) take a unit each.
//! - **Attention over the step's rows.** `attn.ragged` is [`Mixer::Joint`]: the rows' keys and
//!   values are stored on the tiles (`gather.pack_rows`: the order they are stored in), a
//!   [`Phase::Barrier`] waits for every row's, then the queries attend (`gather.unpack_rows`
//!   puts each row's output back). No cache outlives the step.
//! - **Lane vectors.** Values of one row a request (the timestep's embedding, the adaLN
//!   modulations) are computed once a step, before the rows flow: the program's `lanes`, a body
//!   in the decode regime. The rows' phases read them where they are (they are resident at
//!   every root, as a decode layer's positions are).
//! - **Host permutations.** Ops that only reorder rows or compute tables from the host's own
//!   inputs (`conv.patchify` / `unpatchify`, `conv.cells`, a position table's splits) run on the
//!   host, as the order rows enter and leave the units ([`Host`]).
//!
//! The rows' ops before the first layer (the input projections), between two runs of layers and
//! after the last (the output norm and projection) are stages of their own ([`Stage`]), with
//! sides as a layer's.

use poem_ir::{Attn, Def, Dim, Gather, Linear, Operands, Operation, Trace, Ty};
use serde::Serialize;

use super::{At, Axis, Body, LayerBody, Mixer, Op, Phase, R, Reader};
use crate::analysis::Analysis;

/// A class of a stream step's rows: its name, the word of the request its rows are (the
/// trace's facts, [`poem_ir::Facts::word`]) and the rows it brings a step.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct RowClass {
    pub name: String,
    pub word: u64,
    pub rows: u32,
}

/// Where in the reading a stage of units runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Stage {
    /// The rows' ops before the first layer.
    Entry,
    /// Layer `layer` of the `stack`-th run of layers (a poem's `fold_layers`, in order).
    Layer { stack: u32, layer: u32 },
    /// The rows' ops between the `stack`-th run of layers and the next.
    Between { stack: u32 },
    /// The rows' ops after the last layer.
    Exit,
}

/// The classes whose rows run one body at a stage: one unit.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Side {
    /// Indices into [`StreamProgram::classes`].
    pub classes: Vec<usize>,
    /// Index into [`StreamProgram::bodies`].
    pub body: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct StageProgram {
    pub stage: Stage,
    pub sides: Vec<Side>,
}

/// What a host op does to the step's rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum HostKind {
    /// It reorders rows the units read or write: the order they are streamed in or out.
    Rows,
    /// A table the units read where they are, as a decode layer's positions.
    Table,
}

/// An op the host runs on the step's rows.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Host {
    pub op: Op,
    pub kind: HostKind,
}

/// A stream reading as a spatial target runs a step.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct StreamProgram {
    /// The residual's width: what a layer's unit hands the next.
    pub hidden: u32,
    pub classes: Vec<RowClass>,
    /// The host's ops, in trace order.
    pub host: Vec<Host>,
    /// The lane vectors' phases, once a step before the rows flow.
    pub lanes: LayerBody,
    pub bodies: Vec<LayerBody>,
    /// The stages the rows flow through, in order: entry, each layer (and what runs between two
    /// runs of layers), exit.
    pub stages: Vec<StageProgram>,
}

impl StreamProgram {
    /// The units the rows flow through: a side of each stage.
    #[must_use]
    pub fn units(&self) -> usize {
        self.stages.iter().map(|s| s.sides.len()).sum()
    }

    /// The bodies the layers' sides run (not the entry's or exit's), distinct.
    #[must_use]
    pub fn layer_bodies(&self) -> Vec<usize> {
        let mut out: Vec<usize> = Vec::new();
        for s in &self.stages {
            if matches!(s.stage, Stage::Layer { .. }) {
                for side in &s.sides {
                    if !out.contains(&side.body) {
                        out.push(side.body);
                    }
                }
            }
        }
        out
    }
}

/// The ops that only reorder a step's rows: on the host, the order rows enter and leave a unit.
const PERMUTES: &[&str] = &[
    "conv.patchify",
    "conv.unpatchify",
    "conv.pixel_shuffle",
    "conv.pixel_unshuffle",
    "gather.pack_rows",
    "gather.unpack_rows",
];

/// Whether `ty` is a row a request (a lane vector).
fn per_lane(ty: &Ty) -> bool {
    matches!(ty, Ty::Tensor { shape, .. } if matches!(shape.first(), Some(Dim::Lanes | Dim::LanesPlus(_))))
}

impl Reader<'_> {
    /// Where a value made outside the body reading it is, in the stream regime: a lane vector or
    /// a host table anywhere; a host input's rows, a host permutation's, or a unit's output
    /// (handed to the next unit) at the row roots.
    pub(super) fn streamed(&self, v: poem_ir::ValueId) -> At {
        if self
            .trace
            .values
            .get(v.0 as usize)
            .is_some_and(|d| per_lane(&d.ty))
        {
            return At::Any;
        }
        match self.def(v) {
            Some(Def::Input(poem_ir::RuntimeInput::Guest {
                axis: poem_ir::InputAxis::Row,
                ..
            })) => At::Root(Axis::Row),
            Some(Def::Op(n)) => match self.host.get(*n as usize).copied().flatten() {
                Some(false) => At::Any,
                _ => At::Root(Axis::Row),
            },
            _ => At::Any,
        }
    }
}

impl Body<'_, '_> {
    /// A node the stream regime adds a phase for.
    pub(super) fn stream_node(
        &mut self,
        node: &Operation,
        op: Op,
        acts: &[poem_ir::ValueId],
        outs: &[poem_ir::ValueId],
    ) -> R<()> {
        match node {
            Operation::Elemwise(e)
                if matches!(
                    e.name(),
                    "elemwise.layernorm"
                        | "elemwise.modulate"
                        | "elemwise.rope_axes"
                        | "elemwise.binary"
                        | "elemwise.sinusoid"
                ) =>
            {
                self.elementwise(op, acts, outs)
            }
            // the order keys and values are stored in
            Operation::Gather(Gather::PackRows { .. }) => self.mixer(op, None, acts, outs),
            // each row's output back in its place: with the attention that made it
            Operation::Gather(Gather::UnpackRows { .. }) => self.elementwise(op, acts, outs),
            Operation::Attn(Attn::Ragged {
                q,
                head_dim,
                kv_heads,
                ..
            }) => {
                let kind = Mixer::Joint {
                    q_heads: self.r.width(*q) / (*head_dim).max(1),
                    kv_heads: *kv_heads,
                    head_dim: *head_dim,
                    causal: false,
                };
                // the stored keys and values are the store phase's; every row's is in before
                // any query reads them
                match self.phases.last_mut() {
                    Some(Phase::Mixer { mixer, .. }) => *mixer = kind.clone(),
                    _ => return Err(format!("{} attends no stored rows", op.op)),
                }
                self.push(Phase::Barrier);
                self.push(Phase::Mixer {
                    mixer: kind,
                    ops: vec![op],
                });
                for &o in outs {
                    self.set(o, At::Tiles(Axis::Col));
                }
                Ok(())
            }
            _ => Err(format!("{} has no spatial lowering", op.op)),
        }
    }
}

/// `trace`'s stream program for a step of `classes`' rows: the nodes each class's word runs.
/// A node no phase runs is refused by name: `<trace>: <stage> <class> node N: <op> has no
/// spatial lowering`.
pub fn lower(
    trace: &Trace,
    analysis: &Analysis,
    classes: &[RowClass],
) -> Result<StreamProgram, String> {
    let name = &trace.name;
    if classes.is_empty() {
        return Err(format!("{name}: a stream step of no rows"));
    }
    let nodes = trace.nodes.len();
    let per: Vec<Vec<bool>> = classes
        .iter()
        .map(|c| trace.nodes.iter().map(|n| n.guard.holds(c.word)).collect())
        .collect();
    let any: Vec<bool> = (0..nodes).map(|at| per.iter().any(|r| r[at])).collect();

    // each running node's stage: a layer number below the last starts the next stack
    let mut stage_of: Vec<Option<Stage>> = vec![None; nodes];
    let (mut stack, mut prev) = (0u32, None);
    for at in (0..nodes).filter(|&at| any[at]) {
        if let Some(l) = analysis.layers[at] {
            if prev.is_some_and(|p| l < p) {
                stack += 1;
            }
            prev = Some(l);
            stage_of[at] = Some(Stage::Layer { stack, layer: l });
        }
    }
    let first = stage_of.iter().position(Option::is_some);
    let last = stage_of.iter().rposition(Option::is_some);
    let (Some(first), Some(last)) = (first, last) else {
        return Err(format!("{name}: the stream step runs no layers"));
    };

    let out_ty = |at: usize| {
        let mut outs = Vec::new();
        trace.nodes[at].op.outputs(&mut outs);
        outs.first()
            .and_then(|v| trace.values.get(v.0 as usize))
            .map(|d| &d.ty)
    };
    let lane: Vec<bool> = (0..nodes)
        .map(|at| any[at] && out_ty(at).is_some_and(per_lane))
        .collect();

    // the host's nodes: outside the layers, no weight read, every input the host's own
    let mut host: Vec<Option<bool>> = vec![None; nodes];
    let mut ins = Vec::new();
    for at in 0..nodes {
        if !any[at] || lane[at] || analysis.layers[at].is_some() {
            continue;
        }
        let node = &trace.nodes[at];
        let op = node.op.name();
        let permutes = PERMUTES.contains(&op);
        ins.clear();
        node.op.inputs(&mut ins);
        let own = |v: &poem_ir::ValueId| match trace.values.get(v.0 as usize).map(|d| &d.def) {
            Some(Def::Input(_)) => true,
            Some(Def::Op(n)) => host[*n as usize].is_some(),
            Some(Def::Merge(arms)) => {
                arms.iter().all(
                    |(a, _)| match trace.values.get(a.0 as usize).map(|d| &d.def) {
                        Some(Def::Op(n)) => !any[*n as usize] || host[*n as usize].is_some(),
                        Some(Def::Input(_)) => true,
                        _ => false,
                    },
                )
            }
            _ => false,
        };
        let matmul = matches!(node.op, Operation::Linear(Linear::Matmul { .. }));
        // a permutation of a unit's rows (the velocity out) is the host's too
        if permutes || (!matmul && ins.iter().all(own)) {
            host[at] = Some(permutes);
        }
    }

    let hidden_of = |at: usize| match out_ty(at) {
        Some(Ty::Tensor { shape, .. }) => match shape.last() {
            Some(Dim::Const(n)) => *n as u32,
            _ => 0,
        },
        _ => 0,
    };
    let first_stage = stage_of[first];
    let hidden = (0..nodes)
        .rev()
        .find(|&at| stage_of[at] == first_stage)
        .map_or(0, hidden_of);

    let reader = |runs: Vec<bool>| Reader {
        trace,
        runs,
        hidden,
        owner: std::collections::HashMap::new(),
        layers: &analysis.layers,
        host: host.clone(),
        stream: true,
    };
    let body = |r, layer| Body {
        r,
        layer,
        phases: Vec::new(),
        at: std::collections::HashMap::new(),
        made: std::collections::HashMap::new(),
        cast: std::collections::HashSet::new(),
        matvec_of: None,
        glu: None,
    };

    // each row node's stage, the stages in order; the lane vectors once
    let mut row_stage: Vec<Option<Stage>> = vec![None; nodes];
    let mut order: Vec<Stage> = Vec::new();
    let mut after = 0;
    for at in 0..nodes {
        if let Some(Stage::Layer { stack, .. }) = stage_of[at] {
            after = stack;
        }
        if !any[at] || host[at].is_some() || lane[at] {
            continue;
        }
        let s = match stage_of[at] {
            Some(s) => s,
            None if at < first => Stage::Entry,
            None if at > last => Stage::Exit,
            None => Stage::Between { stack: after },
        };
        row_stage[at] = Some(s);
        if !order.contains(&s) {
            order.push(s);
        }
    }
    let layer_of = |s: Stage| match s {
        Stage::Layer { layer, .. } => layer,
        Stage::Entry | Stage::Between { .. } | Stage::Exit => u32::MAX,
    };
    let describe = |s: Stage| match s {
        Stage::Entry => "entry".to_string(),
        Stage::Layer { stack, layer } => format!("stack {stack} layer {layer}"),
        Stage::Between { stack } => format!("after stack {stack}"),
        Stage::Exit => "exit".to_string(),
    };

    let any_reader = reader(any.clone());
    let mut lanes = body(&any_reader, u32::MAX);
    for at in (0..nodes).filter(|&at| lane[at]) {
        lanes
            .node(at)
            .map_err(|e| format!("{name}: lanes node {at}: {e}"))?;
    }
    let lanes = lanes.finish().map_err(|e| format!("{name}: lanes: {e}"))?;

    let readers: Vec<Reader> = per.iter().map(|r| reader(r.clone())).collect();
    // each class's body at each stage
    let mut read: Vec<Vec<Body>> = readers
        .iter()
        .map(|r| order.iter().map(|&s| body(r, layer_of(s))).collect())
        .collect();
    for at in 0..nodes {
        let Some(s) = row_stage[at] else {
            continue;
        };
        let i = order.iter().position(|&o| o == s).unwrap_or(0);
        for (c, class) in classes.iter().enumerate() {
            if !per[c][at] {
                continue;
            }
            read[c][i]
                .node(at)
                .map_err(|e| format!("{name}: {} {} node {at}: {e}", describe(s), class.name))?;
        }
    }

    let mut bodies: Vec<LayerBody> = Vec::new();
    let mut stages: Vec<StageProgram> = order
        .iter()
        .map(|&stage| StageProgram {
            stage,
            sides: Vec::new(),
        })
        .collect();
    for (c, class_bodies) in read.into_iter().enumerate() {
        for (i, b) in class_bodies.into_iter().enumerate() {
            if b.phases.is_empty() {
                continue;
            }
            let body = b
                .finish()
                .map_err(|e| format!("{name}: {} {}: {e}", describe(order[i]), classes[c].name))?;
            let at = match bodies.iter().position(|o| *o == body) {
                Some(at) => at,
                None => {
                    bodies.push(body);
                    bodies.len() - 1
                }
            };
            let sides = &mut stages[i].sides;
            match sides.iter_mut().find(|s| s.body == at) {
                Some(side) => side.classes.push(c),
                None => sides.push(Side {
                    classes: vec![c],
                    body: at,
                }),
            }
        }
    }

    let host = (0..nodes)
        .filter_map(|at| {
            let kind = match host[at]? {
                true => HostKind::Rows,
                false => HostKind::Table,
            };
            Some(Host {
                op: Op {
                    op: trace.nodes[at].op.name(),
                    width: hidden_of(at),
                    params: Vec::new(),
                },
                kind,
            })
        })
        .collect();
    Ok(StreamProgram {
        hidden,
        classes: classes.to_vec(),
        host,
        lanes,
        bodies,
        stages,
    })
}

/// The conventional row classes of a diffusion poem's `reading` = `reading` step: one a value
/// of its `stream` switch among `context` (the text rows) and `latents` (the image rows), each
/// `rows` rows; a poem with no `stream` switch, one class of the reading alone.
#[must_use]
pub fn classes(trace: &Trace, reading: &str, rows: u32) -> Vec<RowClass> {
    let request = |stream: Option<&str>| {
        let r = poem_ir::Request::new(rows, false).switched("reading", reading);
        match stream {
            Some(s) => r.switched("stream", s),
            None => r,
        }
    };
    let streams: Vec<&str> = ["context", "latents"]
        .into_iter()
        .filter(|v| {
            trace.facts.table.iter().any(|f| {
                matches!(f, poem_ir::Fact::Switch { name, value } if name == "stream" && value == v)
            }) || trace
                .facts
                .declared("stream")
                .is_some_and(|d| d.values.iter().any(|x| x == v))
        })
        .collect();
    if streams.is_empty() {
        return vec![RowClass {
            name: reading.to_string(),
            word: trace.facts.word(&request(None)),
            rows,
        }];
    }
    streams
        .into_iter()
        .map(|s| RowClass {
            name: s.to_string(),
            word: trace.facts.word(&request(Some(s))),
            rows,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::tests::form;
    use super::*;
    use crate::analysis::analyze;

    fn program(name: &str) -> Result<StreamProgram, String> {
        let t = crate::catalog::repository()
            .parse(name)
            .unwrap_or_else(|| panic!("{name} parses"))
            .trace(poem_ir::Platform::Cerebras);
        let a = analyze(&t).unwrap();
        lower(&t, &a, &classes(&t, "denoise", 64))
    }

    /// The miniature's denoise: the input projections a unit a class; two double blocks a
    /// body a class (their own weights), joint attention between a barrier; two single blocks
    /// one body both classes share; the output norm and projection the image rows' only.
    #[test]
    fn flux_2_minis_denoise_lowers_a_unit_a_layer_side() {
        let p = program("flux2-mini-bf16-kv-bf16").unwrap_or_else(|e| panic!("{e}"));
        let names: Vec<&str> = p.classes.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["context", "latents"]);
        assert_eq!(p.hidden, 256);
        let stages: Vec<Stage> = p.stages.iter().map(|s| s.stage).collect();
        assert_eq!(
            stages,
            [
                Stage::Entry,
                Stage::Layer { stack: 0, layer: 0 },
                Stage::Layer { stack: 0, layer: 1 },
                Stage::Layer { stack: 1, layer: 0 },
                Stage::Layer { stack: 1, layer: 1 },
                Stage::Exit,
            ]
        );
        let sides: Vec<usize> = p.stages.iter().map(|s| s.sides.len()).collect();
        assert_eq!(sides, [2, 2, 2, 1, 1, 1]);
        assert_eq!(p.units(), 9);
        assert_eq!(p.layer_bodies().len(), 3);
        // the double blocks share their bodies across layers, a body a class
        assert_eq!(p.stages[1].sides, p.stages[2].sides);
        assert_eq!(p.stages[3].sides, p.stages[4].sides);
        assert_eq!(p.stages[3].sides[0].classes, [0, 1]);
        assert_eq!(p.stages[5].sides[0].classes, [1]);

        let double = p.bodies[p.stages[1].sides[0].body].clone();
        assert_eq!(
            form(&double),
            "Rr2 Cr Mr1 Rc4 Cc X3 B X2 Mc1 Rr3 Cr Mr1 Rc1 Cc Mc1 Rr1"
        );
        assert_eq!(
            double.mixers().next(),
            Some(&Mixer::Joint {
                q_heads: 2,
                kv_heads: 2,
                head_dim: 128,
                causal: false
            })
        );
        let single = &p.bodies[p.stages[3].sides[0].body];
        assert_eq!(form(single), "Rr2 Cr Mr1 Rc4 Cc X3 B X2 Mc1 Rc1 Cc Mc1 Rr2");
        assert_eq!(form(&p.bodies[p.stages[5].sides[0].body]), "Rr2 Cr Mr1");
        // the timestep's and guidance's embeddings, the modulations: once a step
        assert!(p.lanes.has("elemwise.sinusoid"));
        let hosts: Vec<(&str, HostKind)> = p.host.iter().map(|h| (h.op.op, h.kind)).collect();
        assert_eq!(
            hosts,
            [
                ("conv.cells", HostKind::Table),
                ("gather.split_rows", HostKind::Table),
                ("conv.patchify", HostKind::Rows),
                ("conv.unpatchify", HostKind::Rows),
            ]
        );
    }

    /// Klein's denoise (its context the text encoder's, laid out by the host) has the
    /// miniature's layer bodies, over 5 double and 20 single blocks.
    #[test]
    fn klein_s_denoise_lowers_to_the_miniature_s_bodies() {
        let p = program("flux2-klein-4b-bf16-kv-bf16").unwrap_or_else(|e| panic!("{e}"));
        let mini = program("flux2-mini-bf16-kv-bf16").unwrap();
        assert_eq!(p.layer_bodies().len(), mini.layer_bodies().len());
        assert_eq!(p.stages.len(), 1 + 5 + 20 + 1);
        assert_eq!(p.hidden, 3072);
        let forms = |p: &StreamProgram| -> Vec<String> {
            p.layer_bodies()
                .iter()
                .map(|&b| form(&p.bodies[b]))
                .collect()
        };
        assert_eq!(forms(&p), forms(&mini));
    }

    /// A decode trace has no stream reading's rows: its layers' nodes run for no class.
    #[test]
    fn a_reading_with_no_rows_is_refused() {
        let t = crate::catalog::repository()
            .parse("flux2-mini-bf16-kv-bf16")
            .unwrap()
            .trace(poem_ir::Platform::Cerebras);
        let a = analyze(&t).unwrap();
        let e = lower(&t, &a, &classes(&t, "nothing", 64)).unwrap_err();
        assert!(e.contains("runs no layers"), "{e}");
        assert!(lower(&t, &a, &[]).is_err());
    }
}

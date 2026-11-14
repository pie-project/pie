//! A stream unit's steps: what one side of a stage (or the lane prologue) does to each row it
//! runs, read node by node off the trace (the side's nodes, [`StageProgram::nodes`]).
//!
//! A unit holds its weights still and its rows flow through it. Each row enters (from the unit
//! before it on the row stream, or from the host), meets the unit's matvecs (P1-stream, from the
//! CX roots to the CY roots or back) and root regions (P4: a run of elementwise nodes lowered by
//! `fuse::chain::lower_run`, at the roots of the side its rows are on), stores its q / k / v in
//! the K / V column of its block (P7) and reads its attention back, and leaves. Splits, packs and
//! unpacks are no step: a value they make is columns of another ([`Cols`]).
//!
//! The joint attention's barrier cuts a unit's steps in two: those that read nothing the
//! attention makes run for every row first ([`Unit::split`]), the rest after every row has
//! stored. A value made before the barrier and read after it is held a row at the root that
//! has it ([`Unit::held`]).
//!
//! [`StageProgram::nodes`]: poem_compiler::spatial::layer::stream::StageProgram::nodes

use std::collections::{BTreeMap, BTreeSet};

use poem_compiler::fuse::chain::{self, Chain};
use poem_ir::{
    Attn, Def, Dim, Dtype, Gather, InputAxis, Linear, Operands, Operation, RuntimeInput, Trace, Ty,
    ValueId,
};

type R<T> = Result<T, String>;

/// The roots a row-side or column-side value is cut over: a unit's CX column (the row roots) or
/// its CY row (the column roots).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
pub enum Side {
    Row,
    Col,
}

impl Side {
    #[must_use]
    pub fn flip(self) -> Side {
        match self {
            Side::Row => Side::Col,
            Side::Col => Side::Row,
        }
    }
}

/// `width` columns of `base` from column `from`: what a split, a pack or an unpack makes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Cols {
    pub base: ValueId,
    pub from: u32,
    pub width: u32,
}

/// One step of a unit, as each row runs it.
#[derive(Clone, Debug)]
pub enum Step {
    /// The row enters at `side`: from the unit before it on the row stream (the CX roots), or
    /// the host's (`host`).
    In {
        value: ValueId,
        side: Side,
        host: bool,
    },
    /// A root region: `chain`'s region at the roots of `side`.
    Region { side: Side, chain: Chain },
    /// A matvec from the roots of `from` to the other side's: `n` x `k` weights.
    Matvec {
        node: u32,
        from: Side,
        input: Cols,
        weight: ValueId,
        output: ValueId,
        k: u32,
        n: u32,
        /// The input is f32: the tile multiplies its two 16-bit halves.
        x32: bool,
        /// The output is a 16-bit value: rounded where it lands.
        round: bool,
    },
    /// The row's q, k and v into its block's K / V column.
    Store { q: Cols, k: Cols, v: Cols },
    /// The row's attention output back from the K / V column.
    Attend { out: ValueId, round: bool },
    /// The row leaves from `side`: to the next unit, the host (`host`), or the lane bus (`lane`).
    Out {
        value: Cols,
        side: Side,
        host: bool,
        lane: bool,
    },
}

/// The joint attention a unit's rows take part in.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Joint {
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub scale: f32,
}

use serde::Serialize;

/// A unit: one side of a stage, or the lane prologue.
#[derive(Clone, Debug)]
pub struct Unit {
    pub name: String,
    /// The row classes it runs (indices into the program's classes); none: the lane prologue.
    pub classes: Vec<usize>,
    /// Its rows a step.
    pub rows: u32,
    pub steps: Vec<Step>,
    /// `steps[..split]` run for every row before the barrier, the rest after it (all, without
    /// one).
    pub split: usize,
    pub joint: Option<Joint>,
    /// The values made before the barrier (or entering) that a step after it reads.
    pub held: Vec<ValueId>,
    /// Where each value it makes (or takes in) is.
    pub sides: BTreeMap<ValueId, Side>,
}

/// The trace read for one class's word.
pub struct Reader<'t> {
    pub trace: &'t Trace,
    pub word: u64,
    /// The lane prologue's: its lane vectors are its rows.
    pub prologue: bool,
}

/// Where a value a unit reads comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Src {
    /// Made by a step of this unit.
    Unit,
    /// A host row input (or a host permutation's rows).
    Host,
    /// Made before this unit on the row stream.
    Prev,
    /// A lane vector.
    Lane,
    /// A weight, a host table, a geometry: what the host writes at the roots.
    Static,
}

impl Reader<'_> {
    fn def(&self, v: ValueId) -> Option<&Def> {
        self.trace.values.get(v.0 as usize).map(|d| &d.def)
    }

    #[must_use]
    pub fn width(&self, v: ValueId) -> u32 {
        match self.trace.values.get(v.0 as usize).map(|d| &d.ty) {
            Some(Ty::Tensor { shape, .. }) => match shape.last() {
                Some(Dim::Const(n)) => *n as u32,
                _ => 0,
            },
            _ => 0,
        }
    }

    #[must_use]
    pub fn dtype(&self, v: ValueId) -> Option<Dtype> {
        match self.trace.values.get(v.0 as usize).map(|d| &d.ty) {
            Some(Ty::Tensor { dtype, .. }) => Some(*dtype),
            _ => None,
        }
    }

    /// Whether `v` is a row a request (a lane vector).
    #[must_use]
    pub fn lane(&self, v: ValueId) -> bool {
        !self.prologue
            && matches!(self.trace.values.get(v.0 as usize).map(|d| &d.ty),
            Some(Ty::Tensor { shape, .. }) if matches!(shape.first(), Some(Dim::Lanes | Dim::LanesPlus(_))))
    }

    /// The merge arm this class's word takes.
    #[must_use]
    pub fn resolve(&self, mut v: ValueId) -> ValueId {
        while let Some(Def::Merge(arms)) = self.def(v) {
            match arms.iter().find(|(_, g)| g.holds(self.word)) {
                Some((a, _)) => v = *a,
                None => break,
            }
        }
        v
    }

    /// The columns `v` is of the value it was split, packed or unpacked from.
    #[must_use]
    pub fn cols(&self, v: ValueId) -> Cols {
        let v = self.resolve(v);
        let width = self.width(v);
        let Some(Def::Op(n)) = self.def(v) else {
            return Cols {
                base: v,
                from: 0,
                width,
            };
        };
        let up = |x: ValueId, off: u32| {
            let c = self.cols(x);
            Cols {
                base: c.base,
                from: c.from + off,
                width,
            }
        };
        match &self.trace.nodes[*n as usize].op {
            Operation::Gather(Gather::SplitRows {
                x, width: w, right, ..
            }) => up(*x, if *right == v { *w } else { 0 }),
            Operation::Gather(Gather::SplitQkv {
                packed,
                q_width,
                kv_width,
                k,
                v: vv,
                ..
            }) => {
                let off = if *k == v {
                    *q_width
                } else if *vv == v {
                    q_width + kv_width
                } else {
                    0
                };
                up(*packed, off)
            }
            Operation::Gather(Gather::PackRows { x, .. } | Gather::UnpackRows { x, .. }) => {
                up(*x, 0)
            }
            _ => Cols {
                base: v,
                from: 0,
                width,
            },
        }
    }

    /// Where `v` (a base) comes from for a unit running `nodes`.
    #[must_use]
    pub fn src(&self, v: ValueId, nodes: &BTreeSet<u32>, host: &BTreeSet<u32>) -> Src {
        if self.lane(v) {
            return Src::Lane;
        }
        match self.def(v) {
            Some(Def::Op(n)) if nodes.contains(n) => Src::Unit,
            Some(Def::Op(n)) if host.contains(n) => Src::Host,
            Some(Def::Op(_)) => Src::Prev,
            Some(Def::Input(RuntimeInput::Guest {
                axis: InputAxis::Row,
                ..
            })) => Src::Host,
            _ => Src::Static,
        }
    }

    fn round(&self, v: ValueId) -> bool {
        matches!(self.dtype(v), Some(Dtype::Bf16 | Dtype::F16))
    }
}

/// The nodes a row op's run may hold: the elementwise ops the root regions lower.
fn rooted(op: &Operation) -> bool {
    matches!(
        op,
        Operation::Elemwise(_) | Operation::Linear(Linear::Glu { .. })
    )
}

/// No step: the value it makes is columns of another.
fn alias(op: &Operation) -> bool {
    matches!(
        op,
        Operation::Gather(
            Gather::SplitRows { .. }
                | Gather::SplitQkv { .. }
                | Gather::PackRows { .. }
                | Gather::UnpackRows { .. }
        )
    )
}

/// The region of `run`, split where a run does not lower as one (each part lowered alone).
fn regions(trace: &Trace, run: &[u32]) -> R<Vec<Chain>> {
    if run.is_empty() {
        return Ok(Vec::new());
    }
    if let Some(c) = chain::lower_run(trace, run) {
        return Ok(vec![c]);
    }
    if run.len() == 1 {
        return Err(format!(
            "node {}: {} lowers to no root region",
            run[0],
            trace.nodes[run[0] as usize].op.name()
        ));
    }
    let (a, b) = run.split_at(run.len() / 2);
    let mut out = regions(trace, a)?;
    out.extend(regions(trace, b)?);
    Ok(out)
}

/// The unit running `nodes` (in trace order) for the classes of `words` (the first's merges
/// taken; `rows` rows a step;
/// `lane`: the lane prologue, its rows the host's). `host` are the host's nodes (the stream
/// program's [`poem_compiler::spatial::layer::stream::Host`] ops), `ext` every node that runs
/// outside the unit (what reads a value it leaves).
pub fn lower(
    trace: &Trace,
    words: &[u64],
    name: &str,
    nodes: &[u32],
    host: &BTreeSet<u32>,
    rows: u32,
    lane: bool,
) -> R<Unit> {
    let word = words[0];
    let r = Reader {
        trace,
        word,
        prologue: lane,
    };
    let own: BTreeSet<u32> = nodes.iter().copied().collect();
    let refuse = |at: u32, what: String| format!("{name}: node {at}: {what}");
    let mut steps: Vec<Step> = Vec::new();
    let mut sides: BTreeMap<ValueId, Side> = BTreeMap::new();
    let mut joint = None;

    // which side a base is on; an input entering where it is first read
    let side_of = |sides: &BTreeMap<ValueId, Side>, c: Cols| sides.get(&c.base).copied();

    // the values a node reads that enter the unit
    let enter =
        |steps: &mut Vec<Step>, sides: &mut BTreeMap<ValueId, Side>, c: Cols, at: Side| -> R<()> {
            if sides.contains_key(&c.base) {
                return Ok(());
            }
            let host_in = match r.src(c.base, &own, host) {
                Src::Host => true,
                Src::Prev => false,
                // the lane prologue's inputs are the host's (the sinusoids of its scalars)
                _ if lane => true,
                _ => return Ok(()),
            };
            if !host_in && at != Side::Row {
                return Err(format!(
                    "{name}: value %{} enters on the columns: the row stream runs on the CX roots",
                    c.base.0
                ));
            }
            sides.insert(c.base, at);
            steps.push(Step::In {
                value: c.base,
                side: at,
                host: host_in,
            });
            Ok(())
        };

    let mut pending: Vec<u32> = Vec::new();
    let mut ins = Vec::new();
    let mut i = 0;
    while i <= nodes.len() {
        let next = nodes.get(i).copied();
        let rooted_next = next.is_some_and(|n| rooted(&trace.nodes[n as usize].op));
        let alias_next = next.is_some_and(|n| alias(&trace.nodes[n as usize].op));
        if rooted_next {
            pending.push(next.unwrap_or(0));
            i += 1;
            continue;
        }
        if alias_next {
            i += 1;
            continue;
        }
        // flush the run of root ops before the next step
        if !pending.is_empty() {
            for c in regions(trace, &pending)? {
                // the side its row inputs are on (a lane prologue's sinusoid: the rows)
                let mut side = None;
                for (k, input) in c.ins.iter().enumerate() {
                    if c.region.params[k] == kernels_ir::Shape::Scalar
                        || input.shared
                        || input.by.is_some()
                    {
                        continue;
                    }
                    let cc = r.cols(input.value);
                    let s = match side_of(&sides, cc) {
                        Some(s) => s,
                        None => {
                            enter(&mut steps, &mut sides, cc, side.unwrap_or(Side::Row))?;
                            match side_of(&sides, cc) {
                                Some(s) => s,
                                None => continue,
                            }
                        }
                    };
                    if side.is_some_and(|x| x != s) {
                        return Err(refuse(
                            c.nodes[0],
                            "a root region over values on both sides (no transpose)".into(),
                        ));
                    }
                    side = Some(s);
                }
                let side = side.unwrap_or(Side::Row);
                for o in &c.outs {
                    sides.insert(o.value, side);
                }
                steps.push(Step::Region { side, chain: c });
            }
            pending.clear();
        }
        let Some(at) = next else { break };
        let node = &trace.nodes[at as usize];
        match &node.op {
            Operation::Linear(Linear::Matmul {
                act,
                w,
                input_scales: None,
                scales: None,
                quant: None,
                logits: false,
                y,
            }) => {
                let x = r.cols(*act);
                if !sides.contains_key(&x.base) {
                    // a host row: on the side whose matvec lands the output where the row
                    // stream (or the next step) wants it
                    let leaves = !lane && reads_outside(trace, words, *y, &own);
                    let at_side = if leaves { Side::Col } else { Side::Row };
                    enter(&mut steps, &mut sides, x, at_side)?;
                }
                let from = side_of(&sides, x)
                    .ok_or_else(|| refuse(at, "linear.matmul reads a value no root has".into()))?;
                sides.insert(*y, from.flip());
                steps.push(Step::Matvec {
                    node: at,
                    from,
                    input: x,
                    weight: *w,
                    output: *y,
                    k: r.width(*act),
                    n: r.width(*y),
                    x32: r.dtype(*act) == Some(Dtype::F32),
                    round: r.round(*y),
                });
            }
            Operation::Attn(Attn::Ragged {
                q,
                k,
                v,
                head_dim,
                kv_heads,
                sm_scale,
                o,
                ..
            }) => {
                let (q, k, v) = (r.cols(*q), r.cols(*k), r.cols(*v));
                for c in [q, k, v] {
                    if side_of(&sides, c) != Some(Side::Col) {
                        return Err(refuse(
                            at,
                            "attn.ragged reads a q / k / v not at the column roots".into(),
                        ));
                    }
                }
                joint = Some(Joint {
                    q_heads: q.width / head_dim,
                    kv_heads: *kv_heads,
                    head_dim: *head_dim,
                    scale: *sm_scale,
                });
                steps.push(Step::Store { q, k, v });
                sides.insert(*o, Side::Col);
                steps.push(Step::Attend {
                    out: *o,
                    round: r.round(*o),
                });
            }
            op => {
                return Err(refuse(at, format!("{} has no stream unit step", op.name())));
            }
        }
        i += 1;
    }

    // what leaves: every value made here (or entering) read outside the unit
    let mut outs: Vec<Cols> = Vec::new();
    for at in nodes {
        ins.clear();
        trace.nodes[*at as usize].op.outputs(&mut ins);
        for &v in &ins {
            if reads_outside(trace, words, v, &own) {
                let c = r.cols(v);
                if sides.contains_key(&c.base) && !outs.contains(&c) {
                    outs.push(c);
                }
            }
        }
    }
    for c in outs {
        let side = sides[&c.base];
        let host_out = !lane
            && trace.nodes.iter().enumerate().any(|(n, node)| {
                host.contains(&(n as u32)) && {
                    let mut i = Vec::new();
                    node.op.inputs(&mut i);
                    i.iter().any(|x| r.cols(*x).base == c.base)
                }
            });
        if !lane && !host_out && side != Side::Row {
            return Err(format!(
                "{name}: value %{} leaves from the column roots: the row stream runs on the CX roots",
                c.base.0
            ));
        }
        steps.push(Step::Out {
            value: c,
            side,
            host: host_out,
            lane,
        });
    }

    // the barrier: what reads the attention's output (or what does) runs after it
    let mut late: BTreeSet<ValueId> = BTreeSet::new();
    let mut is_late = vec![false; steps.len()];
    for (i, s) in steps.iter().enumerate() {
        let reads: Vec<ValueId> = step_reads(&r, s);
        let l = matches!(s, Step::Attend { .. }) || reads.iter().any(|v| late.contains(v));
        is_late[i] = l;
        if l {
            late.extend(step_makes(s));
        }
    }
    let mut early: Vec<Step> = Vec::new();
    let mut after: Vec<Step> = Vec::new();
    for (s, l) in steps.into_iter().zip(&is_late) {
        if *l { after.push(s) } else { early.push(s) }
    }
    let split = early.len();
    let made_early: BTreeSet<ValueId> = early.iter().flat_map(step_makes).collect();
    let mut held: Vec<ValueId> = Vec::new();
    for s in &after {
        for v in step_reads(&r, s) {
            if made_early.contains(&v) && !held.contains(&v) {
                held.push(v);
            }
        }
    }
    let mut steps = early;
    steps.extend(after);
    Ok(Unit {
        name: name.to_string(),
        classes: Vec::new(),
        rows,
        steps,
        split,
        joint,
        held,
        sides,
    })
}

/// Whether a node not in `own` reads `v` (or a value `v` is columns of: a split read outside).
fn reads_outside(trace: &Trace, words: &[u64], v: ValueId, own: &BTreeSet<u32>) -> bool {
    let mut ins = Vec::new();
    for (n, node) in trace.nodes.iter().enumerate() {
        if own.contains(&(n as u32)) || !words.iter().any(|&w| node.guard.holds(w)) {
            continue;
        }
        ins.clear();
        node.op.inputs(&mut ins);
        if ins.contains(&v) {
            return true;
        }
        // through a merge
        for &i in &ins {
            if let Some(Def::Merge(arms)) = trace.values.get(i.0 as usize).map(|d| &d.def)
                && arms.iter().any(|(a, _)| *a == v)
            {
                return true;
            }
        }
    }
    false
}

/// The bases a step reads (row and column values only; lanes and statics are resident).
pub fn step_reads(r: &Reader, s: &Step) -> Vec<ValueId> {
    match s {
        Step::In { .. } => Vec::new(),
        Step::Region { chain, .. } => chain
            .ins
            .iter()
            .enumerate()
            .filter(|(k, i)| {
                chain.region.params[*k] != kernels_ir::Shape::Scalar
                    && !i.shared
                    && i.by.is_none()
                    && !r.lane(i.value)
            })
            .map(|(_, i)| r.cols(i.value).base)
            .collect(),
        Step::Matvec { input, .. } => vec![input.base],
        Step::Store { q, k, v } => vec![q.base, k.base, v.base],
        Step::Attend { .. } => Vec::new(),
        Step::Out { value, .. } => vec![value.base],
    }
}

/// The bases a step makes.
pub fn step_makes(s: &Step) -> Vec<ValueId> {
    match s {
        Step::In { value, .. } => vec![*value],
        Step::Region { chain, .. } => chain.outs.iter().map(|o| o.value).collect(),
        Step::Matvec { output, .. } => vec![*output],
        Step::Attend { out, .. } => vec![*out],
        Step::Store { .. } | Step::Out { .. } => Vec::new(),
    }
}

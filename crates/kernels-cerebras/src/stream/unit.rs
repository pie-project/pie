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
    Attn, Def, Dim, Dtype, Gather, InputAxis, Linear, Operands, Operation, RuntimeInput, Ssm,
    Trace, Ty, ValueId,
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
    /// A matvec from the roots of `from` to the other side's: `n` x `k` weights, `weight` read
    /// as `view` lays them.
    Matvec {
        node: u32,
        from: Side,
        input: Cols,
        weight: ValueId,
        view: View,
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

/// How a matvec's `n` x `k` weights lie in its weight value (`W[o, i]`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum View {
    /// Row-major `[n, k]`.
    #[default]
    Plain,
    /// Output `o` the value's row `(o / block) stride + off + o % block`: a bank's experts'
    /// `block` rows each, `off` into each expert's `stride` (an expert's gate or up half).
    Rows { block: u32, stride: u32, off: u32 },
    /// Input `i` in block `i / block` of `[k / block, n, block]`: each block its expert's own
    /// `[n, block]` (a bank's down projections side by side).
    Blocks { block: u32 },
    /// Constant: one where `o / rep == i` (an expert's gate over its `rep` columns).
    Spread { rep: u32 },
    /// Constant: one where `i / rep == o` (a group's sum of its `rep` columns).
    Sum { rep: u32 },
}

impl View {
    /// The element of the weight value `W[o, i]` is, of `k` inputs and `n` outputs; `None`
    /// for a constant view's.
    #[must_use]
    pub fn at(self, o: usize, i: usize, k: usize, n: usize) -> Option<usize> {
        match self {
            View::Plain => Some(o * k + i),
            View::Rows { block, stride, off } => {
                let (b, s, f) = (block as usize, stride as usize, off as usize);
                Some(((o / b) * s + f + o % b) * k + i)
            }
            View::Blocks { block } => {
                let b = block as usize;
                Some((i / b) * n * b + o * b + i % b)
            }
            View::Spread { .. } | View::Sum { .. } => None,
        }
    }
}

/// The joint attention a unit's rows take part in.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Joint {
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub scale: f32,
    /// A row attends the rows stored before it alone (a prefill's mask).
    pub causal: bool,
    /// And of them, the last `window` alone (a sliding window's), where one is stated.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub window: Option<u32>,
    /// A recurrence's state in place of the stored rows (a gated delta rule: P8, its column's
    /// PEs holding each head's state), where one is.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub state: Option<State>,
    /// Each q head's sink logit (a weight of `q_heads`), joining its softmax's sum (gpt-oss's
    /// `attn.sink` folded into the attention), where one is.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sink: Option<ValueId>,
    /// Each stored K / V head replicated over this many columns, a column its share of the
    /// head's q heads (`kv_heads` the replicas': a head root's q heads and tiles that many times
    /// fewer; the search's choice where a head root's tiles do not fit).
    #[serde(skip_serializing_if = "is_one")]
    pub kv_rep: u32,
}

/// A gated delta rule's column (P8, ops/sst.csl, ops/cy_sst.csl): each head's root also takes
/// its `b` and `a` (the beta and decay logits, a column a head), and holds the conv's taps
/// (`conv`, `taps` a channel) and the decay's `a_log` and `dt` (one a head).
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct State {
    pub b: Cols,
    pub a: Cols,
    pub conv: ValueId,
    pub taps: u32,
    pub a_log: ValueId,
    pub dt: ValueId,
    /// The q / k channels' first column and the v channels' in the conv's input (its `[q | k |
    /// v]` layout).
    pub k_from: u32,
    pub v_from: u32,
    /// Value heads a key head (a head root `h`'s q / k the key head `h / ratio`'s).
    #[serde(skip_serializing_if = "is_one")]
    pub ratio: u32,
    /// The q channels' first column in the conv's input (0: `[q | k | v]`).
    #[serde(skip_serializing_if = "is_zero")]
    pub q_from: u32,
    /// A value head's width where it is not the key head's (Mamba-2's head beside its state).
    #[serde(skip_serializing_if = "is_zero")]
    pub v_dim: u32,
    /// Mamba-2's recurrence in place of the delta rule (ops/cy_sst.csl's MAMBA): `d` a head's
    /// skip, `bias` the conv's a channel.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mamba: Option<Mamba>,
    /// Kimi's delta attention in place of Gated DeltaNet's (ops/cy_sst.csl's KDA): a decay a key
    /// channel (`a` its forget logits, `head_dim` a head; `dt` the same).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kda: Option<Kda>,
}

/// Kimi's delta attention's own parameters of a state column ([`State::kda`]).
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Kda {
    pub head_dim: u32,
    /// The norms' eps (q's and k's l2 norms).
    pub eps: f32,
    /// The decay's floor (`g = floor sigmoid(..)`), 0 where unbounded.
    pub floor: f32,
}

impl State {
    /// The decay logits a head root takes: one (a head's), or a key's each under KDA.
    #[must_use]
    pub fn a_width(&self) -> u32 {
        self.kda.map_or(1, |k| k.head_dim)
    }
}

/// Mamba-2's own parameters of a state column ([`State::mamba`]).
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Mamba {
    pub d: ValueId,
    pub bias: ValueId,
}

fn is_one(r: &u32) -> bool {
    *r == 1
}

fn is_zero(r: &u32) -> bool {
    *r == 0
}

impl Joint {
    /// The q head (group) a head root `h` reads: a recurrence's key head over several value
    /// heads, else its own.
    #[must_use]
    pub fn q_head(&self, h: u32) -> u32 {
        h / self.state.map_or(1, |s| s.ratio.max(1))
    }

    /// The k head a head root `h` stores: a recurrence's key head, else the K / V head it
    /// replicates.
    #[must_use]
    pub fn k_head(&self, h: u32) -> u32 {
        match self.state {
            Some(s) => h / s.ratio.max(1),
            None => h / self.kv_rep.max(1),
        }
    }

    /// The v head a head root `h` stores: a recurrence's own, else the K / V head it
    /// replicates.
    #[must_use]
    pub fn v_head(&self, h: u32) -> u32 {
        match self.state {
            Some(_) => h,
            None => h / self.kv_rep.max(1),
        }
    }

    /// A head root's v (and output) width: a recurrence's value head's, else its q heads'.
    #[must_use]
    pub fn v_dim(&self) -> u32 {
        match self.state {
            Some(s) if s.v_dim > 0 => s.v_dim,
            Some(_) => self.head_dim,
            None => self.head_dim,
        }
    }

    /// A head root's output width: its q heads' (attention) or its value head's (a
    /// recurrence).
    #[must_use]
    pub fn out_dim(&self) -> u32 {
        match self.state {
            Some(_) => self.v_dim(),
            None => self.q_heads / self.kv_heads.max(1) * self.head_dim,
        }
    }
}

impl Serialize for Cols {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        (self.base.0, self.from, self.width).serialize(s)
    }
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

/// A value's copy on the other side of its unit, moved across by an identity matvec (a root
/// region over values of both sides runs where the row stream is; a column value the row
/// stream takes on leaves from the rows): its id is the value's with this bit set.
pub const MOVED: u32 = 1 << 30;

/// The copy of `v` moved across its unit ([`MOVED`]).
#[must_use]
pub fn moved(v: ValueId) -> ValueId {
    ValueId(v.0 | MOVED)
}

/// The value `v` is (itself, or what it is a moved copy of).
#[must_use]
pub fn unmoved(v: ValueId) -> ValueId {
    ValueId(v.0 & !MOVED)
}

/// A copy of a row value's columns laid out as a value of its own (a layer's slice of a
/// stacked row value, re-laid through the columns so the row roots hold it as its own width).
pub const RELAID: u32 = 1 << 28;

/// The value `v` names in the trace (a moved or re-laid copy's original).
fn traced(v: ValueId) -> ValueId {
    ValueId(v.0 & !(MOVED | RELAID))
}

impl Reader<'_> {
    fn def(&self, v: ValueId) -> Option<&Def> {
        self.trace.values.get(unmoved(v).0 as usize).map(|d| &d.def)
    }

    #[must_use]
    pub fn width(&self, v: ValueId) -> u32 {
        if let Some((w, _)) = super::moe::ty(self.trace, unmoved(v)) {
            return w;
        }
        match self.trace.values.get(traced(v).0 as usize).map(|d| &d.ty) {
            Some(Ty::Tensor { shape, .. }) => match shape.last() {
                Some(Dim::Const(n)) => *n as u32,
                _ => 0,
            },
            _ => 0,
        }
    }

    #[must_use]
    pub fn dtype(&self, v: ValueId) -> Option<Dtype> {
        if let Some((_, d)) = super::moe::ty(self.trace, unmoved(v)) {
            return Some(d);
        }
        match self.trace.values.get(traced(v).0 as usize).map(|d| &d.ty) {
            Some(Ty::Tensor { dtype, .. }) => Some(*dtype),
            _ => None,
        }
    }

    /// Whether `v` is a row a request (a lane vector).
    #[must_use]
    pub fn lane(&self, v: ValueId) -> bool {
        !self.prologue
            && matches!(self.trace.values.get(unmoved(v).0 as usize).map(|d| &d.ty),
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
        // a moved copy is a value of its own, whole
        if v.0 & MOVED != 0 {
            return Cols {
                base: v,
                from: 0,
                width: self.width(v),
            };
        }
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
            // (a q | k | v whose q is cut into its heads' halves: each part its own matvec's,
            // the packed product never made)
            Operation::Gather(Gather::SplitQkv { q, .. })
                if self.trace.nodes.iter().any(|n| {
                    matches!(&n.op, Operation::Gather(Gather::SplitQGate { packed, .. }) if packed == q)
                }) =>
            {
                Cols {
                    base: v,
                    from: 0,
                    width,
                }
            }
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
            Operation::Gather(
                Gather::PackRows { x, .. } | Gather::UnpackRows { x, .. } | Gather::Rows { x, .. },
            ) => up(*x, 0),
            // (a layer's columns of a stacked value: a per-layer embedding's projection)
            Operation::Gather(Gather::Select {
                table,
                layer,
                width: w,
                ..
            }) => up(*table, layer * w),
            // (a sink's output is its attention's, the sink folded into the Attend)
            Operation::Attn(Attn::Sink { o, .. }) => up(*o, 0),
            // (an adapter's correction: its product, no adapter on the wafer)
            Operation::Linear(Linear::LoraCorrect { y, .. }) => up(*y, 0),
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
        Operation::Elemwise(_)
            | Operation::Linear(
                Linear::Glu { .. } | Linear::MoeSigmoidGateAdd { .. } | Linear::Relu2 { .. }
            )
    )
}

/// No step: the value it makes is columns of another (an adapter's correction is its product
/// unchanged: the wafer serves no adapter).
fn alias(op: &Operation) -> bool {
    matches!(
        op,
        Operation::Gather(
            Gather::SplitRows { .. }
                | Gather::SplitQkv { .. }
                | Gather::PackRows { .. }
                | Gather::UnpackRows { .. }
                | Gather::Rows { .. }
                | Gather::Select { .. }
        ) | Operation::Linear(Linear::LoraCorrect { .. })
    )
}

/// Whether `c`'s row values (each read whole) are of different widths: head regions over values
/// of several head counts (a q of 2 heads a kv head beside its k).
fn mixed(trace: &Trace, c: &Chain) -> bool {
    let width = |v: ValueId| match trace.values.get(v.0 as usize).map(|d| &d.ty) {
        Some(Ty::Tensor { shape, .. }) => match shape.last() {
            Some(Dim::Const(n)) => *n as u32,
            _ => 0,
        },
        _ => 0,
    };
    let ws: BTreeSet<u32> = (c.ins.iter().enumerate())
        .filter(|(k, i)| {
            c.region.params[*k] != kernels_ir::Shape::Scalar
                && !i.shared
                && i.by.is_none()
                && i.from == 0
        })
        .map(|(_, i)| width(i.value))
        .collect();
    ws.len() > 1
}

/// The region of `run`, split where a run does not lower as one (each part lowered alone).
fn regions(trace: &Trace, run: &[u32]) -> R<Vec<Chain>> {
    if run.is_empty() {
        return Ok(Vec::new());
    }
    // (a part over row values of different widths — a q of 2 heads a kv head beside its v —
    // places on no one set of roots: its nodes apart)
    if let Some(c) = chain::lower_run_parts(trace, run)
        && (run.len() == 1 || !c.iter().any(|c| mixed(trace, c)))
    {
        return Ok(c);
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
    lower_through(trace, words, name, nodes, host, rows, lane, &[])
}

/// [`lower`], the unit also taking `through` in on the row stream and sending them on, read
/// or not (values a later unit reads that the row stream carries past this one: an
/// encoder's earlier tapped layers).
#[allow(clippy::too_many_arguments)]
pub fn lower_through(
    trace: &Trace,
    words: &[u64],
    name: &str,
    nodes: &[u32],
    host: &BTreeSet<u32>,
    rows: u32,
    lane: bool,
    through: &[ValueId],
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
            // the row stream runs on the CX roots: what it brings enters there (an op over it
            // and a column value moves the column value across)
            let at = if host_in { at } else { Side::Row };
            sides.insert(c.base, at);
            steps.push(Step::In {
                value: c.base,
                side: at,
                host: host_in,
            });
            Ok(())
        };

    for &v in through {
        if let std::collections::btree_map::Entry::Vacant(e) = sides.entry(v) {
            e.insert(Side::Row);
            steps.push(Step::In {
                value: v,
                side: Side::Row,
                host: false,
            });
        }
    }
    let mut pending: Vec<u32> = Vec::new();
    let mut appended: Vec<(ValueId, Cols, Cols)> = Vec::new();
    let mut ins = Vec::new();
    let mut i = 0;
    // the nodes a routed layer's lowering stands for, past its route
    let mut skip: BTreeSet<u32> = BTreeSet::new();
    // a routed layer's group, by its first select
    let mut routed: BTreeMap<u32, super::moe::Group> = BTreeMap::new();
    while i <= nodes.len() {
        let next = nodes.get(i).copied();
        if next.is_some_and(|n| skip.contains(&n)) {
            i += 1;
            continue;
        }
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
            for mut c in regions(trace, &pending)? {
                // a layer's slice of a stacked row value (a per-layer embedding's projection):
                // the row roots hold the stacked value's columns, not the slice's as a value of
                // its own width; re-laid through the columns (two identity matvecs) first
                for k in 0..c.ins.len() {
                    let v = c.ins[k].value;
                    let sliced = matches!(r.def(v), Some(Def::Op(n))
                        if matches!(trace.nodes[*n as usize].op, Operation::Gather(Gather::Select { .. })));
                    if !sliced || c.region.params[k] == kernels_ir::Shape::Scalar {
                        continue;
                    }
                    let cc = r.cols(v);
                    if side_of(&sides, cc).is_none() {
                        enter(&mut steps, &mut sides, cc, Side::Row)?;
                    }
                    if side_of(&sides, cc) != Some(Side::Row) {
                        continue;
                    }
                    // the stacked value across whole (once), the slice back from its columns
                    let all = ValueId(cc.base.0 | RELAID);
                    if let std::collections::btree_map::Entry::Vacant(e) = sides.entry(all) {
                        e.insert(Side::Col);
                        let w = r.width(cc.base);
                        steps.push(Step::Matvec {
                            node: c.nodes[0],
                            from: Side::Row,
                            input: Cols {
                                base: cc.base,
                                from: 0,
                                width: w,
                            },
                            weight: ValueId(super::host::UNIT),
                            view: View::Plain,
                            output: all,
                            k: w,
                            n: w,
                            x32: r.dtype(cc.base) == Some(Dtype::F32),
                            round: r.round(cc.base),
                        });
                    }
                    let a = ValueId(v.0 | RELAID);
                    let part = Cols { base: all, ..cc };
                    move_across(&r, &mut steps, &mut sides, a, part, c.nodes[0]);
                    c.ins[k].value = moved(a);
                }
                // the side its row inputs are on (a lane prologue's sinusoid: the rows)
                let mut side = None;
                let mut on: Vec<(usize, Cols, Side)> = Vec::new();
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
                    on.push((k, cc, s));
                    side = side.or(Some(s));
                }
                // over values of both sides: it runs on the rows, each column value moved
                // across first (an identity matvec)
                if on.iter().any(|x| x.2 != side.unwrap_or(Side::Row)) {
                    for &(k, cc, s) in &on {
                        if s == Side::Row {
                            continue;
                        }
                        // the columns it reads moved across: the value whole, or its part (a
                        // split's half: its own moved copy)
                        let key = if cc.from == 0 && cc.width == r.width(cc.base) {
                            cc.base
                        } else {
                            c.ins[k].value
                        };
                        move_across(&r, &mut steps, &mut sides, key, cc, c.nodes[0]);
                        c.ins[k].value = moved(key);
                    }
                    side = Some(Side::Row);
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
                y,
                ..
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
                // a product its heads' halves are cut from (a query and its gate a head, maybe
                // after a q | k | v split): a matvec a part, each over its rows of the weight (a
                // strided split of the output is a view of the weight)
                let find = |pick: &dyn Fn(&Operation) -> bool| {
                    nodes
                        .iter()
                        .copied()
                        .find(|&n| pick(&trace.nodes[n as usize].op))
                };
                let qkv = find(
                    &|op| matches!(op, Operation::Gather(Gather::SplitQkv { packed, .. }) if packed == y),
                );
                let qsrc = match qkv {
                    Some(n) => match &trace.nodes[n as usize].op {
                        Operation::Gather(Gather::SplitQkv { q, .. }) => *q,
                        _ => unreachable!(),
                    },
                    None => *y,
                };
                let halves = find(
                    &|op| matches!(op, Operation::Gather(Gather::SplitQGate { packed, .. }) if *packed == qsrc),
                );
                if let Some(hn) = halves {
                    let Operation::Gather(Gather::SplitQGate {
                        head_dim: d,
                        q,
                        gate,
                        ..
                    }) = &trace.nodes[hn as usize].op
                    else {
                        unreachable!()
                    };
                    let mut parts = vec![
                        (
                            *q,
                            View::Rows {
                                block: *d,
                                stride: 2 * d,
                                off: 0,
                            },
                        ),
                        (
                            *gate,
                            View::Rows {
                                block: *d,
                                stride: 2 * d,
                                off: *d,
                            },
                        ),
                    ];
                    skip.insert(hn);
                    if let Some(n) = qkv {
                        let Operation::Gather(Gather::SplitQkv {
                            q_width,
                            kv_width,
                            k,
                            v,
                            ..
                        }) = &trace.nodes[n as usize].op
                        else {
                            unreachable!()
                        };
                        skip.insert(n);
                        for (out, off) in [(*k, *q_width), (*v, q_width + kv_width)] {
                            parts.push((
                                out,
                                View::Rows {
                                    block: *kv_width,
                                    stride: *kv_width,
                                    off,
                                },
                            ));
                        }
                    }
                    for (out, view) in parts {
                        sides.insert(out, from.flip());
                        steps.push(Step::Matvec {
                            node: at,
                            from,
                            input: x,
                            weight: *w,
                            view,
                            output: out,
                            k: r.width(*act),
                            n: r.width(out),
                            x32: r.dtype(*act) == Some(Dtype::F32),
                            round: r.round(out),
                        });
                    }
                    i += 1;
                    continue;
                }
                sides.insert(*y, from.flip());
                steps.push(Step::Matvec {
                    node: at,
                    from,
                    input: x,
                    weight: *w,
                    view: View::Plain,
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
                    causal: false,
                    window: None,
                    state: None,
                    sink: None,
                    kv_rep: 1,
                });
                steps.push(Step::Store { q, k, v });
                sides.insert(*o, Side::Col);
                steps.push(Step::Attend {
                    out: *o,
                    round: r.round(*o),
                });
            }
            // the step's keys and values into the pages its prefill reads: stored with the
            // row's query when the prefill comes
            Operation::Attn(Attn::KvAppend {
                k,
                v: Some(v),
                cache,
                ..
            }) => {
                appended.push((*cache, r.cols(*k), r.cols(*v)));
            }
            Operation::Attn(Attn::Prefill {
                q,
                cache,
                head_dim,
                kv_heads,
                sm_scale,
                causal,
                o,
                ..
            }) => {
                let Some(&(_, k, v)) = appended.iter().find(|(c, _, _)| c == cache) else {
                    return Err(refuse(
                        at,
                        "attn.prefill reads pages no step of its unit appends".into(),
                    ));
                };
                let q = r.cols(*q);
                for c in [q, k, v] {
                    if side_of(&sides, c) != Some(Side::Col) {
                        return Err(refuse(
                            at,
                            "attn.prefill reads a q / k / v not at the column roots".into(),
                        ));
                    }
                }
                joint = Some(Joint {
                    q_heads: q.width / head_dim,
                    kv_heads: *kv_heads,
                    head_dim: *head_dim,
                    scale: *sm_scale,
                    causal: *causal,
                    window: None,
                    state: None,
                    sink: None,
                    kv_rep: 1,
                });
                steps.push(Step::Store { q, k, v });
                sides.insert(*o, Side::Col);
                steps.push(Step::Attend {
                    out: *o,
                    round: r.round(*o),
                });
            }
            // routed experts: their gated sum over every expert ([`super::moe`])
            // (lowered where its first select comes: its input made by then)
            Operation::Linear(Linear::MoeRoute { .. }) => {
                let g = super::moe::group(trace, at, nodes).map_err(|e| refuse(at, e))?;
                skip.extend(g.nodes[2..].iter().copied());
                routed.insert(g.nodes[1], g);
            }
            Operation::Linear(Linear::MoeSelect { .. }) if routed.contains_key(&at) => {
                use super::moe::{Kind, synth};
                let g = routed.remove(&at).unwrap_or_else(|| unreachable!());
                let at = g.route;
                let l = r.cols(g.logits);
                let gs = side_of(&sides, l).ok_or_else(|| {
                    refuse(at, "linear.moe_route reads logits no root has".into())
                })?;
                let (e, ei, h) = (g.experts, g.experts * g.inter, g.hidden);
                let gate = synth(at, Kind::Gate);
                steps.push(Step::Region {
                    side: gs,
                    chain: super::moe::router(trace, &g).map_err(|e| refuse(at, e))?,
                });
                sides.insert(gate, gs);
                // the experts' columns over the column roots (the most PEs), from the rows: the
                // gate and the layer's input on the rows, moved there where they are not
                let (rows, o) = (Side::Row, Side::Col);
                let to_rows = |steps: &mut Vec<Step>,
                               sides: &mut BTreeMap<ValueId, Side>,
                               c: Cols,
                               from: Side,
                               kind: Kind,
                               x32: bool,
                               round: bool| {
                    if from == rows {
                        return c;
                    }
                    let moved = synth(at, kind);
                    steps.push(Step::Matvec {
                        node: at,
                        from,
                        input: c,
                        weight: ValueId(super::host::UNIT),
                        view: View::Plain,
                        output: moved,
                        k: c.width,
                        n: c.width,
                        x32,
                        round,
                    });
                    sides.insert(moved, rows);
                    Cols {
                        base: moved,
                        from: 0,
                        width: c.width,
                    }
                };
                let gin = to_rows(
                    &mut steps,
                    &mut sides,
                    Cols {
                        base: gate,
                        from: 0,
                        width: e,
                    },
                    gs,
                    Kind::GateRows,
                    true,
                    false,
                );
                let spread = synth(at, Kind::Spread);
                steps.push(Step::Matvec {
                    node: at,
                    from: rows,
                    input: gin,
                    weight: ValueId(super::host::UNIT),
                    view: View::Spread { rep: g.inter },
                    output: spread,
                    k: e,
                    n: ei,
                    x32: true,
                    round: false,
                });
                sides.insert(spread, o);
                let xc = r.cols(g.x);
                let xs = side_of(&sides, xc).ok_or_else(|| {
                    refuse(at, "linear.moe_select reads a value no root has".into())
                })?;
                let xin = to_rows(
                    &mut steps,
                    &mut sides,
                    xc,
                    xs,
                    Kind::Input,
                    r.dtype(g.x) == Some(Dtype::F32),
                    r.round(g.x),
                );
                let narrow = r.dtype(synth(at, Kind::Gates)).unwrap_or(Dtype::Bf16);
                // (a non-gated activation's experts: their ups alone)
                let banks: Vec<(Kind, (ValueId, u32, u32))> = g
                    .gate
                    .map(|gt| (Kind::Gates, gt))
                    .into_iter()
                    .chain([(Kind::Ups, g.up)])
                    .collect();
                for (kind, (bank, stride, off)) in banks {
                    steps.push(Step::Matvec {
                        node: at,
                        from: rows,
                        input: xin,
                        weight: bank,
                        view: View::Rows {
                            block: g.inter,
                            stride,
                            off,
                        },
                        output: synth(at, kind),
                        k: h,
                        n: ei,
                        x32: r.dtype(g.x) == Some(Dtype::F32),
                        round: true,
                    });
                    sides.insert(synth(at, kind), o);
                }
                steps.push(Step::Region {
                    side: o,
                    chain: super::moe::activation(&g, narrow).map_err(|e| refuse(at, e))?,
                });
                let acts = synth(at, Kind::Acts);
                sides.insert(acts, o);
                steps.push(Step::Matvec {
                    node: at,
                    from: o,
                    input: Cols {
                        base: acts,
                        from: 0,
                        width: ei,
                    },
                    weight: g.down,
                    view: View::Blocks { block: g.inter },
                    output: g.y,
                    k: ei,
                    n: h,
                    x32: false,
                    round: r.round(g.y),
                });
                sides.insert(g.y, rows);
            }
            // a gated delta rule (its conv, its gates, its recurrence): a step's row through
            // its heads' state columns (P8), the conv's history and the gates at their roots
            // Mamba-2 (its conv, its scan, its gated group norm): a step's row through its heads'
            // state columns (P8's MAMBA), the norm as root regions and constant matvecs
            // (super::mamba)
            Operation::Ssm(Ssm::CausalConv1d {
                x,
                weight,
                bias: Some(cb),
                conv_width,
                dilation: 1,
                y: conv_y,
                ..
            }) if nodes.iter().any(|&n| {
                matches!(&trace.nodes[n as usize].op,
                    Operation::Ssm(Ssm::Mamba2 { xbc, .. }) if xbc == conv_y)
            }) =>
            {
                use super::mamba::{Kind as MK, synth as msynth};
                let refuse_at = |what: &str| refuse(at, format!("ssm.mamba2 {what}"));
                let mn = nodes
                    .iter()
                    .copied()
                    .find(|&n| {
                        matches!(&trace.nodes[n as usize].op,
                            Operation::Ssm(Ssm::Mamba2 { xbc, .. }) if xbc == conv_y)
                    })
                    .unwrap_or_else(|| unreachable!());
                let Operation::Ssm(Ssm::Mamba2 {
                    dt,
                    z,
                    a_log,
                    dt_bias,
                    d,
                    norm,
                    eps,
                    y,
                    ..
                }) = &trace.nodes[mn as usize].op
                else {
                    unreachable!()
                };
                let (h, g, n, p, inner) =
                    super::mamba::dims(trace, mn).ok_or_else(|| refuse_at("of no inner width"))?;
                if g == 0 || !h.is_multiple_of(g) || !inner.is_multiple_of(g) {
                    return Err(refuse_at(&format!("of {h} heads over {g} groups")));
                }
                skip.insert(mn);
                let xc = r.cols(*x);
                let dtc = r.cols(*dt);
                for c in [xc, dtc] {
                    if side_of(&sides, c) != Some(Side::Col) {
                        return Err(refuse_at("reads a value not at the column roots"));
                    }
                }
                // xbc: [x (inner) | B (a group's n each) | C]: q C, k B, v x
                let q = Cols {
                    from: xc.from + inner + g * n,
                    width: g * n,
                    ..xc
                };
                let k = Cols {
                    from: xc.from + inner,
                    width: g * n,
                    ..xc
                };
                let v = Cols { width: inner, ..xc };
                let dth = Cols { width: h, ..dtc };
                joint = Some(Joint {
                    q_heads: h,
                    kv_heads: h,
                    head_dim: n,
                    scale: 1.0,
                    causal: true,
                    window: None,
                    state: Some(State {
                        b: dth,
                        a: dth,
                        conv: *weight,
                        taps: *conv_width,
                        a_log: *a_log,
                        dt: *dt_bias,
                        k_from: inner,
                        v_from: 0,
                        ratio: h / g,
                        q_from: inner + g * n,
                        v_dim: p,
                        mamba: Some(super::unit::Mamba { d: *d, bias: *cb }),
                        kda: None,
                    }),
                    sink: None,
                    kv_rep: 1,
                });
                steps.push(Step::Store { q, k, v });
                let raw = msynth(mn, MK::Raw);
                sides.insert(raw, Side::Col);
                steps.push(Step::Attend {
                    out: raw,
                    round: false,
                });
                // the gate, each group's sum of squares and its spread, the norm
                let zd = r.dtype(*z).unwrap_or(Dtype::Bf16);
                steps.push(Step::Region {
                    side: Side::Col,
                    chain: super::mamba::gate(mn, *z, zd, inner).map_err(|e| refuse(at, e))?,
                });
                for kind in [MK::Gated, MK::Squares] {
                    sides.insert(msynth(mn, kind), Side::Col);
                }
                let gw = inner / g;
                steps.push(Step::Matvec {
                    node: mn,
                    from: Side::Col,
                    input: Cols {
                        base: msynth(mn, MK::Squares),
                        from: 0,
                        width: inner,
                    },
                    weight: ValueId(super::host::UNIT),
                    view: View::Sum { rep: gw },
                    output: msynth(mn, MK::Sums),
                    k: inner,
                    n: g,
                    x32: true,
                    round: false,
                });
                sides.insert(msynth(mn, MK::Sums), Side::Row);
                steps.push(Step::Matvec {
                    node: mn,
                    from: Side::Row,
                    input: Cols {
                        base: msynth(mn, MK::Sums),
                        from: 0,
                        width: g,
                    },
                    weight: ValueId(super::host::UNIT),
                    view: View::Spread { rep: gw },
                    output: msynth(mn, MK::Spread),
                    k: g,
                    n: inner,
                    x32: true,
                    round: false,
                });
                sides.insert(msynth(mn, MK::Spread), Side::Col);
                let wd = r.dtype(*norm).unwrap_or(Dtype::Bf16);
                let yd = r.dtype(*y).unwrap_or(Dtype::Bf16);
                steps.push(Step::Region {
                    side: Side::Col,
                    chain: super::mamba::norm(mn, gw, *eps, *norm, wd, *y, yd, inner)
                        .map_err(|e| refuse(at, e))?,
                });
                sides.insert(*y, Side::Col);
            }
            Operation::Ssm(Ssm::CausalConv1d {
                x,
                weight,
                bias: None,
                conv_width,
                dilation: 1,
                y: conv_y,
                ..
            }) => {
                let refuse_at = |what: &str| refuse(at, format!("ssm.causal_conv1d {what}"));
                let find = |pick: &dyn Fn(&Operation) -> bool| {
                    nodes
                        .iter()
                        .copied()
                        .find(|&n| n > at && pick(&trace.nodes[n as usize].op))
                };
                // Kimi's delta attention: a gated delta rule of a decay a key channel
                if let Some(kn) = find(
                    &|op| matches!(op, Operation::Ssm(Ssm::Kda { mixed, .. }) if mixed == conv_y),
                ) {
                    let Operation::Ssm(Ssm::Kda {
                        f,
                        b,
                        dt_bias,
                        a_log,
                        heads,
                        head_dim,
                        norm_eps,
                        gate_floor,
                        y,
                        ..
                    }) = &trace.nodes[kn as usize].op
                    else {
                        unreachable!()
                    };
                    skip.insert(kn);
                    let (h, d) = (*heads, *head_dim);
                    let xc = r.cols(*x);
                    let bc = r.cols(*b);
                    for c in [xc, bc] {
                        if side_of(&sides, c) != Some(Side::Col) {
                            return Err(refuse_at("reads a value not at the column roots"));
                        }
                    }
                    // the forget logits (a low-rank projection's, at the row roots) to the
                    // column roots: a constant matvec (one where its output is its input)
                    let mut fc = r.cols(*f);
                    match side_of(&sides, fc) {
                        Some(Side::Col) => {}
                        Some(Side::Row) => {
                            let moved = super::mamba::synth(kn, super::mamba::Kind::Moved);
                            steps.push(Step::Matvec {
                                node: kn,
                                from: Side::Row,
                                input: fc,
                                weight: ValueId(super::host::UNIT),
                                view: View::Spread { rep: 1 },
                                output: moved,
                                k: h * d,
                                n: h * d,
                                x32: true,
                                round: false,
                            });
                            sides.insert(moved, Side::Col);
                            fc = Cols {
                                base: moved,
                                from: 0,
                                width: h * d,
                            };
                        }
                        None => return Err(refuse_at("reads forget logits on no roots")),
                    }
                    let q = Cols { width: h * d, ..xc };
                    let k = Cols {
                        from: xc.from + h * d,
                        width: h * d,
                        ..xc
                    };
                    let v = Cols {
                        from: xc.from + 2 * h * d,
                        width: h * d,
                        ..xc
                    };
                    joint = Some(Joint {
                        q_heads: h,
                        kv_heads: h,
                        head_dim: d,
                        scale: 1.0,
                        causal: true,
                        window: None,
                        state: Some(State {
                            b: Cols { width: h, ..bc },
                            a: fc,
                            conv: *weight,
                            taps: *conv_width,
                            a_log: *a_log,
                            dt: *dt_bias,
                            k_from: h * d,
                            v_from: 2 * h * d,
                            ratio: 1,
                            q_from: 0,
                            v_dim: 0,
                            mamba: None,
                            kda: Some(Kda {
                                head_dim: d,
                                eps: *norm_eps,
                                floor: *gate_floor,
                            }),
                        }),
                        sink: None,
                        kv_rep: 1,
                    });
                    steps.push(Step::Store { q, k, v });
                    sides.insert(*y, Side::Col);
                    steps.push(Step::Attend {
                        out: *y,
                        round: r.round(*y),
                    });
                    i += 1;
                    continue;
                }
                let gd = find(&|op| {
                    matches!(op, Operation::Ssm(Ssm::GatedDelta { qkv, .. }) if qkv == conv_y)
                })
                .ok_or_else(|| refuse_at("feeds no gated delta rule"))?;
                let Operation::Ssm(Ssm::GatedDelta {
                    gates,
                    k_heads,
                    v_heads,
                    k_dim,
                    v_dim,
                    y,
                    ..
                }) = &trace.nodes[gd as usize].op
                else {
                    unreachable!()
                };
                // (a state column a value head; several value heads a key head share its q / k)
                if k_dim != v_dim || *k_heads == 0 || !v_heads.is_multiple_of(*k_heads) {
                    return Err(refuse_at(&format!(
                        "into {k_heads} key heads of {k_dim} beside {v_heads} value heads of \
                         {v_dim}: a state column a value head, its key head's q / k"
                    )));
                }
                let gn = find(&|op| {
                    matches!(op, Operation::Ssm(Ssm::GdnGates { gates: g, .. }) if g == gates)
                })
                .or_else(|| {
                    // (the gates made before the conv)
                    nodes.iter().copied().find(|&n| {
                        matches!(&trace.nodes[n as usize].op,
                            Operation::Ssm(Ssm::GdnGates { gates: g, .. }) if g == gates)
                    })
                })
                .ok_or_else(|| refuse_at("gated by no gdn_gates"))?;
                let Operation::Ssm(Ssm::GdnGates {
                    ba, dt_bias, a_log, ..
                }) = &trace.nodes[gn as usize].op
                else {
                    unreachable!()
                };
                skip.extend([gn, gd]);
                let (h, hk, d) = (*v_heads, *k_heads, *k_dim);
                let xc = r.cols(*x);
                let bac = r.cols(*ba);
                for c in [xc, bac] {
                    if side_of(&sides, c) != Some(Side::Col) {
                        return Err(refuse_at("reads a value not at the column roots"));
                    }
                }
                let q = Cols {
                    width: hk * d,
                    ..xc
                };
                let k = Cols {
                    from: xc.from + hk * d,
                    width: hk * d,
                    ..xc
                };
                let v = Cols {
                    from: xc.from + 2 * hk * d,
                    width: h * d,
                    ..xc
                };
                joint = Some(Joint {
                    q_heads: h,
                    kv_heads: h,
                    head_dim: d,
                    scale: 1.0,
                    causal: true,
                    window: None,
                    state: Some(State {
                        b: Cols { width: h, ..bac },
                        a: Cols {
                            from: bac.from + h,
                            width: h,
                            ..bac
                        },
                        conv: *weight,
                        taps: *conv_width,
                        a_log: *a_log,
                        dt: *dt_bias,
                        k_from: hk * d,
                        v_from: 2 * hk * d,
                        ratio: h / hk,
                        q_from: 0,
                        v_dim: 0,
                        mamba: None,
                        kda: None,
                    }),
                    sink: None,
                    kv_rep: 1,
                });
                steps.push(Step::Store { q, k, v });
                sides.insert(*y, Side::Col);
                steps.push(Step::Attend {
                    out: *y,
                    round: r.round(*y),
                });
            }
            // a decode step's attention over its layer's cache: the row stored with its query
            // (at its position), then attending every row stored at or before it
            Operation::Attn(Attn::Decode {
                q,
                plan,
                cache,
                head_dim,
                sm_scale,
                window,
                o,
                lse,
                ..
            }) => {
                // its log-sum-exp read by a sink alone: the sink folded in (the Attend's)
                let sink = match lse {
                    None => None,
                    Some(l) => {
                        let sn = nodes
                            .iter()
                            .copied()
                            .find(|&n| {
                                matches!(&trace.nodes[n as usize].op,
                                    Operation::Attn(Attn::Sink { o: so, lse: sl, .. })
                                        if so == o && sl == l)
                            })
                            .ok_or_else(|| {
                                refuse(at, "attn.decode's log-sum-exp is read by no sink".into())
                            })?;
                        let Operation::Attn(Attn::Sink { sink, .. }) = &trace.nodes[sn as usize].op
                        else {
                            unreachable!()
                        };
                        skip.insert(sn);
                        Some(*sink)
                    }
                };
                let Some(&(_, k, v)) = appended.iter().find(|(c, _, _)| c == cache) else {
                    return Err(refuse(
                        at,
                        "attn.decode reads pages no step of its unit appends".into(),
                    ));
                };
                let kv_heads = match r.def(*plan) {
                    Some(Def::Op(n)) => match &trace.nodes[*n as usize].op {
                        Operation::Attn(Attn::PlanDecode { kv_heads, .. }) => *kv_heads,
                        _ => return Err(refuse(at, "attn.decode's plan is no decode plan".into())),
                    },
                    _ => return Err(refuse(at, "attn.decode's plan is no decode plan".into())),
                };
                let q = r.cols(*q);
                for c in [q, k, v] {
                    if side_of(&sides, c) != Some(Side::Col) {
                        return Err(refuse(
                            at,
                            format!(
                                "attn.decode reads a q / k / v not at the column roots (value \
                                 %{} at {:?})",
                                c.base.0,
                                side_of(&sides, c)
                            ),
                        ));
                    }
                }
                joint = Some(Joint {
                    q_heads: q.width / head_dim,
                    kv_heads,
                    head_dim: *head_dim,
                    scale: *sm_scale,
                    causal: true,
                    window: *window,
                    state: None,
                    sink,
                    kv_rep: 1,
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
            if reads_outside(trace, words, v, &own) || read_out(trace, v) {
                let c = r.cols(v);
                if sides.contains_key(&c.base) && !outs.contains(&c) {
                    outs.push(c);
                }
            }
        }
    }
    for &v in through {
        let c = Cols {
            base: v,
            from: 0,
            width: r.width(v),
        };
        if !outs.iter().any(|o| unmoved(o.base) == v) {
            outs.push(c);
        }
    }
    for c in outs {
        let side = sides[&c.base];
        // to the host: what a host op reads, or what the reading reads out itself
        let host_out = !lane
            && (read_out(trace, c.base)
                || trace.nodes.iter().enumerate().any(|(n, node)| {
                    host.contains(&(n as u32)) && {
                        let mut i = Vec::new();
                        node.op.inputs(&mut i);
                        i.iter().any(|x| r.cols(*x).base == c.base)
                    }
                }));
        // a column value the row stream takes on leaves from the rows, moved across first
        let (c, side) = if !lane && !host_out && side != Side::Row {
            if c.from != 0 || c.width != r.width(c.base) {
                return Err(format!(
                    "{name}: part of value %{} leaves from the column roots: the row stream runs on the CX roots",
                    c.base.0
                ));
            }
            move_across(
                &r,
                &mut steps,
                &mut sides,
                c.base,
                Cols {
                    base: c.base,
                    from: 0,
                    width: c.width,
                },
                u32::MAX,
            );
            (
                Cols {
                    base: moved(c.base),
                    ..c
                },
                Side::Row,
            )
        } else {
            (c, side)
        };
        steps.push(Step::Out {
            value: c,
            side,
            host: host_out,
            lane,
        });
    }
    // several values on the row stream: in and out in one order (by value), each row's in at
    // its start and out at its end
    let streamed_in = |s: &Step| {
        matches!(
            s,
            Step::In {
                side: Side::Row,
                host: false,
                ..
            }
        )
    };
    let streamed_out = |s: &Step| {
        matches!(
            s,
            Step::Out {
                side: Side::Row,
                host: false,
                lane: false,
                ..
            }
        )
    };
    if steps.iter().filter(|s| streamed_in(s)).count() > 1 {
        let mut ins: Vec<Step> = Vec::new();
        steps.retain(|s| {
            if streamed_in(s) {
                ins.push(s.clone());
                false
            } else {
                true
            }
        });
        ins.sort_by_key(|s| match s {
            Step::In { value, .. } => unmoved(*value).0,
            _ => 0,
        });
        steps.splice(0..0, ins);
    }
    if steps.iter().filter(|s| streamed_out(s)).count() > 1 {
        let mut outs: Vec<Step> = Vec::new();
        steps.retain(|s| {
            if streamed_out(s) {
                outs.push(s.clone());
                false
            } else {
                true
            }
        });
        outs.sort_by_key(|s| match s {
            Step::Out { value, .. } => unmoved(value.base).0,
            _ => 0,
        });
        steps.extend(outs);
    }

    let several_out = steps.iter().filter(|s| streamed_out(s)).count() > 1;
    // the barrier: what reads the attention's output (or what does) runs after it
    let mut late: BTreeSet<ValueId> = BTreeSet::new();
    let mut is_late = vec![false; steps.len()];
    for (i, s) in steps.iter().enumerate() {
        let reads: Vec<ValueId> = step_reads(&r, s);
        // what a unit with a barrier sends on, it sends after it (each row's values together)
        let l = matches!(s, Step::Attend { .. })
            || reads.iter().any(|v| late.contains(v))
            || (joint.is_some() && streamed_out(s) && several_out);
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

/// `v` (a column value) moved onto the rows by an identity matvec, where it is not yet: the
/// step before the one that reads it (made for node `node`).
fn move_across(
    r: &Reader,
    steps: &mut Vec<Step>,
    sides: &mut BTreeMap<ValueId, Side>,
    v: ValueId,
    input: Cols,
    node: u32,
) {
    let m = moved(v);
    if sides.contains_key(&m) {
        return;
    }
    let w = input.width;
    steps.push(Step::Matvec {
        node,
        from: Side::Col,
        input,
        weight: ValueId(super::host::UNIT),
        view: View::Plain,
        output: m,
        k: w,
        n: w,
        x32: r.dtype(v) == Some(Dtype::F32),
        round: r.round(v),
    });
    sides.insert(m, Side::Row);
}

/// Whether the trace reads `v` out (one of its outputs, or an arm of one).
fn read_out(trace: &Trace, v: ValueId) -> bool {
    trace.outputs.iter().any(|o| {
        o.value == v
            || matches!(trace.values.get(o.value.0 as usize).map(|d| &d.def),
                Some(Def::Merge(arms)) if arms.iter().any(|(a, _)| *a == v))
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

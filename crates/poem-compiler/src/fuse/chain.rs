//! Vertical fusion into generated kernels: the elementwise and row ops a
//! trace runs back to back, formed into chains and each lowered to a
//! region (kernels-ir's [`Region`], a function of one row) that kernels-*
//! `synth` builds one kernel for. The compiler hands over the region alone:
//! whether a chain is one the row core schedules is kernels-ir's question
//! ([`kernels_ir::schedule`]), and each backend prints its own kernel.
//!
//! **Forming** is conservative, as the fuse rules are: a chain is a run of
//! ops, each one this module lowers, each reading a value an earlier op of
//! the run computes, all under one guard and over rows of one width (a run
//! may cross a layer's end, as a layer's last residual add and the next
//! layer's input norm do); at most [`MAX_CHAIN`] ops, and at least two (a
//! lone op keeps its hand kernel). The run's ops need not be consecutive:
//! the ops between them run before it, and none reads what the run computes
//! or writes what it reads, so the region lands where the run's last op was,
//! in its layer, as a fuse rule's match does. Of the run from an op, the
//! longest part the row frame holds is the chain. Nothing else is searched,
//! so no unplanned combination forms.
//!
//! **Lowering** turns each op into the primitive math it states, in f32. A value
//! the run reads but does not compute is a parameter; a value it computes
//! that anything past the run may read (another op, a seam, a merge, or
//! nothing at all, as the trace's last values) is a result, stored in its
//! own dtype. A value only the run reads never leaves the registers, but it
//! is rounded to its dtype where it is defined, as the ops it replaces
//! store it ([`Region::rounds`]): a trace value's dtype is a rounding point
//! the trace states, so fusing never changes what a chain lands. What one
//! op computes inside itself (a norm's sum of squares) stays f32.
//!
//! **Folding** ([`fold`]) replaces each chain's ops with one
//! `fused.region` op reading its parameters and landing its results, the
//! values only the chain read gone from the trace. No result lands over a
//! parameter, so the arena gives each its own plane: a row frame's threads
//! read their row whole before any writes it.
//!
//! Chains are formed and folded only under `compiler.synth_regions` (off by
//! default).

use std::collections::BTreeMap;

use kernels_ir::{Binary, Literal, Op, Reduce, Region, Shape, Unary};
use poem_ir::{
    ActFn, BinaryOp, Bounds, Def, Dim, Dtype, Elemwise, Fused, Node, NormAxis, NormWeight,
    Operands, Operation, Trace, Ty, ValueId,
};

/// The most ops one chain holds.
pub const MAX_CHAIN: usize = 8;

/// `gelu`'s tanh form: `0.5 x (1 + tanh(√(2/π) (x + 0.044715 x³)))`.
const GELU_SQRT_2_OVER_PI: f32 = 0.797_884_6;
const GELU_CUBIC: f32 = 0.044_715;

/// A chain of ops lowered to one region.
#[derive(Debug, Clone, PartialEq)]
pub struct Chain {
    /// The trace's ops the region replaces, in order; the ops between them
    /// run before the region, where its last op was.
    pub nodes: Vec<u32>,
    /// `ins[k]` is the region's parameter `k`.
    pub ins: Vec<In>,
    /// `outs[k]` is the region's result `k`.
    pub outs: Vec<Out>,
    pub region: Region,
}

/// A value a chain reads and how it is stored: the kernels' `In`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct In {
    pub value: ValueId,
    pub dtype: Dtype,
    /// One row every row reads (a norm's weight), not one a row.
    pub shared: bool,
    /// One element every row reads (a layer's scale, a weight of one
    /// element): a scalar parameter.
    pub one: bool,
}

/// A value a chain writes, a row a row in its own dtype: the kernels' `Out`.
/// One the trace computes in place (a residual's stream) is stored over
/// its input's plane, as the trace's placement says.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Out {
    pub value: ValueId,
    pub dtype: Dtype,
}

/// `trace` with each chain the row core schedules replaced by one
/// `fused.region` op, and the regions its ops name, by key
/// ([`Region::key`]). Two chains of one region (a layer's and the next's)
/// share one entry.
#[must_use]
pub fn fold(trace: Trace) -> (Trace, BTreeMap<u64, Region>) {
    let mut regions = BTreeMap::new();
    let mut folded: Vec<(Vec<u32>, Fused)> = Vec::new();
    for chain in chains(&trace) {
        if kernels_ir::schedule(&chain.region).is_err() {
            continue;
        }
        let width = chain.region.width;
        if !frames(&trace, &chain, width) {
            continue;
        }
        folded.push((
            chain.nodes.clone(),
            Fused::Region {
                ins: chain.ins.iter().map(|input| input.value).collect(),
                width,
                key: chain.region.key(),
                outs: chain.outs.iter().map(|out| out.value).collect(),
            },
        ));
        regions.entry(chain.region.key()).or_insert(chain.region);
    }
    if folded.is_empty() {
        return (trace, regions);
    }
    (super::search::compact(replace(trace, folded)), regions)
}

/// The widest row the frame holds in registers: a block of
/// [`FRAME_BLOCK`] threads, [`FRAME_PER`] eight-wide runs a thread.
pub const FRAME_BLOCK: u32 = 256;
pub const FRAME_PER: u32 = 4;
/// The planes a frame holds at most, read and written together, at its
/// widest: each plane is [`FRAME_PER`] sixteen-byte runs a thread, and past
/// five at the widest the row no longer stays in registers.
pub const FRAME_PLANES: usize = 5;

/// Whether the row frame a region's body runs in (`row_frame` in
/// kernels-cuda's `elemwise/frame.cuh`) holds `chain`: every row it reads
/// and writes bf16, the width a whole number of eight-wide runs within one
/// block's registers, at most [`FRAME_PLANES`] planes each way (a scalar
/// parameter, one element, holds none). A chain the frame does not hold is
/// never folded, so an engine never meets a region it cannot launch.
#[must_use]
pub fn frames(trace: &Trace, chain: &Chain, width: u32) -> bool {
    let bf16 = |v: poem_ir::ValueId| {
        matches!(
            &trace.values[v.0 as usize].ty,
            poem_ir::Ty::Tensor {
                dtype: poem_ir::Dtype::Bf16,
                ..
            }
        )
    };
    width.is_multiple_of(8)
        && width <= FRAME_BLOCK * FRAME_PER * 8
        && planes_fit(
            chain.ins.iter().filter(|input| !input.one).count() + chain.outs.len(),
            width,
        )
        && chain.ins.iter().all(|input| bf16(input.value))
        && chain.outs.iter().all(|out| bf16(out.value))
}

/// Whether `planes` rows of `width` stay in a frame's registers: as many
/// planes as [`FRAME_PLANES`] at the widest, more as the row narrows (the
/// runs a thread holds scale with the width).
fn planes_fit(planes: usize, width: u32) -> bool {
    let per = width.div_ceil(FRAME_BLOCK * 8).max(1) as usize;
    planes * per <= FRAME_PLANES * FRAME_PER as usize
}

/// `trace` with each run's ops replaced by its op, landed where the run's
/// last op was, under the run's guard and in the layer of its last op that
/// has one (a fuse rule's match lands so too); the runs are apart.
fn replace(mut trace: Trace, runs: Vec<(Vec<u32>, Fused)>) -> Trace {
    // Each node's run, and the run each run's last node lands.
    let mut run_of: Vec<Option<usize>> = vec![None; trace.nodes.len()];
    for (r, (nodes, _)) in runs.iter().enumerate() {
        for &n in nodes {
            run_of[n as usize] = Some(r);
        }
    }
    let lasts: Vec<u32> = runs
        .iter()
        .map(|(nodes, _)| *nodes.last().expect("a run holds ops"))
        .collect();
    let layers: Vec<Option<u32>> = runs
        .iter()
        .map(|(nodes, _)| {
            nodes
                .iter()
                .rev()
                .find_map(|&n| trace.nodes[n as usize].layer)
        })
        .collect();
    // Where each node lands: a run's ops all where its last lands.
    let mut at_of: Vec<u32> = vec![0; trace.nodes.len()];
    let mut landed_at = vec![0u32; runs.len()];
    let mut next = 0u32;
    for (n, run) in run_of.iter().enumerate() {
        match run {
            Some(r) if lasts[*r] as usize != n => {}
            Some(r) => {
                landed_at[*r] = next;
                next += 1;
            }
            None => {
                at_of[n] = next;
                next += 1;
            }
        }
    }
    for (n, run) in run_of.iter().enumerate() {
        if let Some(r) = run {
            at_of[n] = landed_at[*r];
        }
    }
    let mut ops: Vec<Option<Fused>> = runs.into_iter().map(|(_, op)| Some(op)).collect();
    let mut nodes = Vec::with_capacity(next as usize);
    for (n, node) in std::mem::take(&mut trace.nodes).into_iter().enumerate() {
        match run_of[n] {
            None => nodes.push(node),
            Some(r) if lasts[r] as usize == n => nodes.push(Node {
                op: Operation::Fused(ops[r].take().expect("a run lands once")),
                guard: node.guard,
                layer: layers[r],
            }),
            Some(_) => {}
        }
    }
    for decl in &mut trace.values {
        if let Def::Op(i) = &mut decl.def
            && let Some(&to) = at_of.get(*i as usize)
        {
            *i = to;
        }
    }
    trace.nodes = nodes;
    trace
}

/// How far past a chain's first op its run is looked for.
const REACH: u32 = 64;

/// Every chain of `trace` the row frame holds, by its first op; ops outside
/// them run as they are.
#[must_use]
pub fn chains(trace: &Trace) -> Vec<Chain> {
    let escapes = Escapes::of(trace);
    let mut out = Vec::new();
    let count = u32::try_from(trace.nodes.len()).unwrap_or(u32::MAX);
    let mut taken = vec![false; trace.nodes.len()];
    for at in 0..count {
        if taken[at as usize] {
            continue;
        }
        let Some(row) = row_of(trace, at) else {
            continue;
        };
        // The run from `at`: each op that joins it, up to the first op that
        // reads what it computes and does not join it.
        let mut run = vec![at];
        let mut next = at + 1;
        while next < count && next <= at + REACH && run.len() < MAX_CHAIN {
            if !taken[next as usize] && joins(trace, &run, next, row) {
                run.push(next);
            } else if reads_from(trace, &run, next) {
                break;
            }
            next += 1;
        }
        if let Some(chain) = held(trace, &run, &escapes) {
            for &n in &chain.nodes {
                taken[n as usize] = true;
            }
            out.push(chain);
        }
    }
    out
}

/// The longest part of `run` (the latest of the longest, so a chain that
/// ends in a norm keeps it, as the rules' matches do) that is a chain the
/// row frame holds: its ops lower, the ops between them keep their order,
/// and the frame takes its region.
fn held(trace: &Trace, run: &[u32], escapes: &Escapes) -> Option<Chain> {
    for len in (2..=run.len()).rev() {
        for from in (0..=run.len() - len).rev() {
            let part = &run[from..from + len];
            if !keeps_order(trace, part) {
                continue;
            }
            let Some(chain) = lower(trace, part, escapes) else {
                continue;
            };
            let framed = kernels_ir::schedule(&chain.region).is_ok()
                && frames(trace, &chain, chain.region.width);
            if framed {
                return Some(chain);
            }
        }
    }
    None
}

/// Whether `node` reads a value an op of `run` computes.
fn reads_from(trace: &Trace, run: &[u32], node: u32) -> bool {
    let mut ins = Vec::new();
    trace.nodes[node as usize].op.inputs(&mut ins);
    ins.iter()
        .any(|v| matches!(trace.values[v.0 as usize].def, Def::Op(by) if run.contains(&by)))
}

/// Whether the ops between the first and the last of `run` keep their
/// order once the run lands where its last op was: none reads what the run
/// computes or a cache it reads, and none writes in place what it reads.
fn keeps_order(trace: &Trace, run: &[u32]) -> bool {
    let (Some(&first), Some(&last)) = (run.first(), run.last()) else {
        return false;
    };
    let mut read = Vec::new();
    let mut written = Vec::new();
    for &n in run {
        let op = &trace.nodes[n as usize].op;
        op.inputs(&mut read);
        op.outputs(&mut written);
    }
    let caches: Vec<ValueId> = read
        .iter()
        .copied()
        .filter(|v| matches!(trace.values[v.0 as usize].def, Def::Cache(_)))
        .collect();
    let (mut ids, mut aliases) = (Vec::new(), Vec::new());
    (first + 1..last).filter(|n| !run.contains(n)).all(|n| {
        let op = &trace.nodes[n as usize].op;
        ids.clear();
        op.inputs(&mut ids);
        aliases.clear();
        op.aliases(&mut aliases);
        ids.iter()
            .all(|v| !written.contains(v) && !caches.contains(v))
            && aliases.iter().all(|(_, into)| !read.contains(into))
    })
}

/// The rows a lowerable op runs over: their leading dim and width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Rows {
    dim: Dim,
    width: u64,
}

/// The rows `node` runs over, if this module lowers it: every value it
/// reads or writes is a row of one width (or a weight that wide).
fn row_of(trace: &Trace, node: u32) -> Option<Rows> {
    let op = &trace.nodes[node as usize].op;
    if !lowers(op) {
        return None;
    }
    // A device scale is its first element: a region reads one of a weight
    // of one element, and no other.
    if let Operation::Elemwise(Elemwise::Act { s: Some(s), .. }) = op
        && !matches!(form(trace, *s), Some(Form::One))
    {
        return None;
    }
    // The rows are its outputs'; an input may be rows of another axis (a
    // readout's rows read beside the tokens'), row k read for row k, as
    // the op's own kernel reads it.
    let (mut ins, mut outs) = (Vec::new(), Vec::new());
    op.inputs(&mut ins);
    op.outputs(&mut outs);
    let mut rows: Option<Rows> = None;
    let mut width: Option<u64> = None;
    for (v, output) in ins
        .into_iter()
        .map(|v| (v, false))
        .chain(outs.into_iter().map(|v| (v, true)))
    {
        let (dim, w) = match form(trace, v)? {
            Form::Rows { dim, width } => (Some(dim), width),
            Form::Shared { width } => (None, width),
            Form::One => continue,
        };
        if width.is_some_and(|held| held != w) {
            return None;
        }
        width = Some(w);
        if let Some(dim) = dim
            && output
        {
            match rows {
                Some(held) if held.dim != dim => return None,
                _ => rows = Some(Rows { dim, width: w }),
            }
        }
    }
    rows
}

/// Whether `node` extends the run `run` over `row`: it lowers over the same
/// rows, under the run's guard, and reads what the run computes. Its layer
/// may be the next one's: a layer's last residual add and the next layer's
/// input norm are one chain, as they are one fused op by a rule.
fn joins(trace: &Trace, run: &[u32], node: u32, row: Rows) -> bool {
    if row_of(trace, node) != Some(row) {
        return false;
    }
    let first = &trace.nodes[run[0] as usize];
    let next = &trace.nodes[node as usize];
    next.guard == first.guard && reads_from(trace, run, node)
}

/// How a value a chain touches is laid out.
enum Form {
    /// One row of `width` a row of `dim`.
    Rows { dim: Dim, width: u64 },
    /// One row of `width` every row reads: a weight.
    Shared { width: u64 },
    /// One element every row reads: a weight of one element.
    One,
}

fn form(trace: &Trace, v: ValueId) -> Option<Form> {
    let decl = &trace.values[v.0 as usize];
    let Ty::Tensor { shape, dtype } = &decl.ty else {
        return None;
    };
    let stored = match decl.def {
        Def::Weight(i) => trace.params[i as usize].dtype,
        _ => *dtype,
    };
    if !matches!(stored, Dtype::Bf16 | Dtype::F16 | Dtype::F32) {
        return None;
    }
    match (shape.as_slice(), &decl.def) {
        ([Dim::Const(1)], Def::Weight(_)) => Some(Form::One),
        ([Dim::Const(width)], Def::Weight(_)) => Some(Form::Shared { width: *width }),
        ([dim, Dim::Const(width)], _) if !matches!(dim, Dim::Const(_)) => Some(Form::Rows {
            dim: *dim,
            width: *width,
        }),
        _ => None,
    }
}

fn dtype_of(trace: &Trace, v: ValueId) -> Dtype {
    let decl = &trace.values[v.0 as usize];
    match (&decl.def, &decl.ty) {
        (Def::Weight(i), _) => trace.params[*i as usize].dtype,
        (_, Ty::Tensor { dtype, .. }) => *dtype,
        (_, Ty::Struct(_)) => unreachable!("a chain's values are tensors"),
    }
}

/// Whether this module lowers `op` (its rows aside).
fn lowers(op: &Operation) -> bool {
    match op {
        Operation::Elemwise(e) => match e {
            Elemwise::ResidualAdd { gate, .. } => gate.is_none(),
            Elemwise::AddBias { .. } => true,
            Elemwise::Binary { .. } => true,
            Elemwise::Rmsnorm { axis, .. } => *axis == NormAxis::Row,
            Elemwise::Layernorm { .. } => true,
            Elemwise::Act { act, .. } => !matches!(act, ActFn::Gelu { tanh: false }),
            Elemwise::Softcap { .. } => true,
            Elemwise::GateSigmoidMul { heads, .. } => heads.is_none(),
            Elemwise::Clamp { bounds, .. } => matches!(bounds, Bounds::Fixed { .. }),
            _ => false,
        },
        Operation::Fused(Fused::ResidualAddRmsnorm { .. }) => true,
        _ => false,
    }
}

/// Who may read each value past a chain: another op, a seam or a merge.
struct Escapes {
    readers: Vec<Vec<u32>>,
    held: Vec<bool>,
}

impl Escapes {
    fn of(trace: &Trace) -> Escapes {
        let mut readers = vec![Vec::new(); trace.values.len()];
        let mut held = vec![false; trace.values.len()];
        let mut ins = Vec::new();
        for (at, node) in (0u32..).zip(&trace.nodes) {
            ins.clear();
            node.op.inputs(&mut ins);
            for v in &ins {
                readers[v.0 as usize].push(at);
            }
        }
        for decl in &trace.values {
            if let Def::Merge(arms) = &decl.def {
                for (arm, _) in arms {
                    held[arm.0 as usize] = true;
                }
            }
        }
        for seam in &trace.seams {
            for v in &seam.values {
                held[v.0 as usize] = true;
            }
        }
        Escapes { readers, held }
    }

    /// Whether `v`, computed inside `run`, may be read past it. A value
    /// nothing reads is the trace's to keep, so it leaves too.
    fn leaves(&self, v: ValueId, run: &[u32]) -> bool {
        let readers = &self.readers[v.0 as usize];
        self.held[v.0 as usize] || readers.is_empty() || readers.iter().any(|r| !run.contains(r))
    }
}

/// The region `run` lowers to, or `None` where an op of it does not lower.
fn lower(trace: &Trace, run: &[u32], escapes: &Escapes) -> Option<Chain> {
    let row = row_of(trace, run[0])?;
    let width = u32::try_from(row.width).ok()?;

    // Parameters: what the run reads and does not compute, in first-read
    // order.
    let mut computed: Vec<ValueId> = Vec::new();
    let mut ins: Vec<In> = Vec::new();
    let mut ids = Vec::new();
    for &node in run {
        let op = &trace.nodes[node as usize].op;
        ids.clear();
        op.inputs(&mut ids);
        for &v in &ids {
            if !computed.contains(&v) && !ins.iter().any(|held| held.value == v) {
                ins.push(In {
                    value: v,
                    dtype: dtype_of(trace, v),
                    shared: matches!(form(trace, v)?, Form::Shared { .. }),
                    one: matches!(form(trace, v)?, Form::One),
                });
            }
        }
        ids.clear();
        op.outputs(&mut ids);
        computed.extend_from_slice(&ids);
    }

    let mut b = Builder {
        params: u32::try_from(ins.len()).ok()?,
        width,
        ops: Vec::new(),
        of: BTreeMap::new(),
    };
    for (k, input) in (0u32..).zip(&ins) {
        b.of.insert(input.value, k);
    }
    for &node in run {
        b.node(&trace.nodes[node as usize].op)?;
    }

    let outs: Vec<Out> = computed
        .iter()
        .filter(|&&v| escapes.leaves(v, run))
        .map(|&v| Out {
            value: v,
            dtype: dtype_of(trace, v),
        })
        .collect();
    let results = outs
        .iter()
        .map(|out| b.of.get(&out.value).copied())
        .collect::<Option<Vec<u32>>>()?;
    // Every value an op of the run defines is rounded to its dtype where it
    // is defined, as the unfused ops store it, so the region lands what
    // they land; only what one op computes inside itself stays f32.
    let mut rounds: Vec<(u32, Dtype)> = Vec::new();
    for &v in &computed {
        let narrow = dtype_of(trace, v);
        if !matches!(narrow, Dtype::Bf16 | Dtype::F16) {
            continue;
        }
        let id = b.get(v)?;
        if id >= b.params && !rounds.iter().any(|(held, _)| *held == id) {
            rounds.push((id, narrow));
        }
    }
    let region = Region {
        width,
        params: ins
            .iter()
            .map(|input| if input.one { Shape::Scalar } else { Shape::Row })
            .collect(),
        ops: b.ops,
        results,
        rounds,
    };
    region.types().ok()?;
    Some(Chain {
        nodes: run.to_vec(),
        ins,
        outs,
        region,
    })
}

/// A region's ops for a run, one trace op at a time.
struct Builder {
    params: u32,
    width: u32,
    ops: Vec<Op>,
    /// The region's value each trace value is.
    of: BTreeMap<ValueId, u32>,
}

impl Builder {
    fn push(&mut self, op: Op) -> u32 {
        let id = self.params + u32::try_from(self.ops.len()).unwrap_or(u32::MAX);
        self.ops.push(op);
        id
    }

    fn konst(&mut self, x: f32) -> u32 {
        self.push(Op::Const(Literal::F32(x)))
    }

    fn get(&self, v: ValueId) -> Option<u32> {
        self.of.get(&v).copied()
    }

    fn set(&mut self, v: ValueId, id: u32) {
        self.of.insert(v, id);
    }

    /// `1 / (1 + e^-t)`.
    fn sigmoid(&mut self, t: u32) -> u32 {
        let one = self.konst(1.0);
        let neg = self.push(Op::Unary(Unary::Neg, t));
        let exp = self.push(Op::Unary(Unary::Exp, neg));
        let den = self.push(Op::Binary(Binary::Add, one, exp));
        self.push(Op::Unary(Unary::Recip, den))
    }

    /// `1 - 2 / (e^2t + 1)`: a region has no tanh.
    fn tanh(&mut self, t: u32) -> u32 {
        let (one, two) = (self.konst(1.0), self.konst(2.0));
        let twice = self.push(Op::Binary(Binary::Mul, t, two));
        let exp = self.push(Op::Unary(Unary::Exp, twice));
        let den = self.push(Op::Binary(Binary::Add, exp, one));
        let frac = self.push(Op::Binary(Binary::Div, two, den));
        self.push(Op::Binary(Binary::Sub, one, frac))
    }

    /// The row's mean: its sum over the width.
    fn mean(&mut self, x: u32) -> u32 {
        let sum = self.push(Op::Reduce(Reduce::Sum, x));
        let n = self.konst(self.width as f32);
        self.push(Op::Binary(Binary::Div, sum, n))
    }

    /// `x · rsqrt(mean(x²) + eps)`, as the hand kernels compute it.
    fn rms_normed(&mut self, x: u32, eps: f32) -> u32 {
        let sq = self.push(Op::Binary(Binary::Mul, x, x));
        let ms = self.mean(sq);
        let eps = self.konst(eps);
        let shifted = self.push(Op::Binary(Binary::Add, ms, eps));
        let r = self.push(Op::Unary(Unary::Rsqrt, shifted));
        self.push(Op::Binary(Binary::Mul, x, r))
    }

    /// `n · w`, or `n · (w + 1)`.
    fn scaled(&mut self, n: u32, w: u32, plus_one: bool) -> u32 {
        let w = if plus_one {
            let one = self.konst(1.0);
            self.push(Op::Binary(Binary::Add, w, one))
        } else {
            w
        };
        self.push(Op::Binary(Binary::Mul, n, w))
    }

    fn node(&mut self, op: &Operation) -> Option<()> {
        match op {
            Operation::Elemwise(e) => self.elemwise(e),
            Operation::Fused(Fused::ResidualAddRmsnorm {
                x,
                y,
                y_out,
                weight,
                plus_one,
                eps,
                out,
            }) => {
                let h = self.push(Op::Binary(Binary::Add, self.get(*y)?, self.get(*x)?));
                self.set(*y_out, h);
                let n = self.rms_normed(h, *eps);
                let o = self.scaled(n, self.get(*weight)?, *plus_one);
                self.set(*out, o);
                Some(())
            }
            _ => None,
        }
    }

    fn elemwise(&mut self, e: &Elemwise) -> Option<()> {
        match e {
            Elemwise::ResidualAdd {
                x,
                gate: None,
                y,
                y_out,
            } => {
                let h = self.push(Op::Binary(Binary::Add, self.get(*y)?, self.get(*x)?));
                self.set(*y_out, h);
            }
            Elemwise::AddBias { bias, out, out_out } => {
                let z = self.push(Op::Binary(Binary::Add, self.get(*out)?, self.get(*bias)?));
                self.set(*out_out, z);
            }
            Elemwise::Binary { x, y, op, z } => {
                let (x, y) = (self.get(*x)?, self.get(*y)?);
                let id = self.push(match op {
                    BinaryOp::Add => Op::Binary(Binary::Add, x, y),
                    BinaryOp::Mul => Op::Binary(Binary::Mul, x, y),
                    BinaryOp::Div => Op::Binary(Binary::Div, x, y),
                    // floored, as Python's `%`: the region has no floor
                    BinaryOp::Mod => return None,
                });
                self.set(*z, id);
            }
            Elemwise::Rmsnorm {
                x,
                weight,
                axis: NormAxis::Row,
                eps,
                y,
            } => {
                let n = self.rms_normed(self.get(*x)?, *eps);
                let out = match weight {
                    NormWeight::None => n,
                    NormWeight::Scale(w) => self.scaled(n, self.get(*w)?, false),
                    NormWeight::PlusOne(w) => self.scaled(n, self.get(*w)?, true),
                };
                self.set(*y, out);
            }
            Elemwise::Layernorm {
                x,
                weight,
                bias,
                eps,
                y,
            } => {
                let x = self.get(*x)?;
                let mean = self.mean(x);
                let d = self.push(Op::Binary(Binary::Sub, x, mean));
                let sq = self.push(Op::Binary(Binary::Mul, d, d));
                let var = self.mean(sq);
                let eps = self.konst(*eps);
                let shifted = self.push(Op::Binary(Binary::Add, var, eps));
                let r = self.push(Op::Unary(Unary::Rsqrt, shifted));
                let mut out = self.push(Op::Binary(Binary::Mul, d, r));
                if let Some(w) = weight {
                    out = self.push(Op::Binary(Binary::Mul, out, self.get(*w)?));
                }
                if let Some(b) = bias {
                    out = self.push(Op::Binary(Binary::Add, out, self.get(*b)?));
                }
                self.set(*y, out);
            }
            Elemwise::Act {
                s,
                x,
                act,
                scale,
                x_out,
            } => {
                let mut t = self.get(*x)?;
                if let Some(k) = scale {
                    let k = self.konst(*k);
                    t = self.push(Op::Binary(Binary::Mul, t, k));
                }
                // a device scalar's first element: a weight of one element
                // the region reads as a scalar parameter
                if let Some(s) = s {
                    t = self.push(Op::Binary(Binary::Mul, t, self.get(*s)?));
                }
                let out = match act {
                    ActFn::Identity => t,
                    ActFn::Silu => {
                        let s = self.sigmoid(t);
                        self.push(Op::Binary(Binary::Mul, t, s))
                    }
                    ActFn::Tanh => self.tanh(t),
                    ActFn::Gelu { tanh: true } => {
                        let (half, one) = (self.konst(0.5), self.konst(1.0));
                        let (c, cubic) = (self.konst(GELU_SQRT_2_OVER_PI), self.konst(GELU_CUBIC));
                        let t2 = self.push(Op::Binary(Binary::Mul, t, t));
                        let t3 = self.push(Op::Binary(Binary::Mul, t2, t));
                        let k3 = self.push(Op::Binary(Binary::Mul, cubic, t3));
                        let inner = self.push(Op::Binary(Binary::Add, t, k3));
                        let arg = self.push(Op::Binary(Binary::Mul, c, inner));
                        let th = self.tanh(arg);
                        let up = self.push(Op::Binary(Binary::Add, one, th));
                        let ht = self.push(Op::Binary(Binary::Mul, half, t));
                        self.push(Op::Binary(Binary::Mul, ht, up))
                    }
                    ActFn::Gelu { tanh: false } => return None,
                    ActFn::Sqrt => self.push(Op::Unary(Unary::Sqrt, t)),
                };
                self.set(*x_out, out);
            }
            Elemwise::Softcap { x, cap, x_out } => {
                let cap = self.konst(*cap);
                let t = self.push(Op::Binary(Binary::Div, self.get(*x)?, cap));
                let th = self.tanh(t);
                let out = self.push(Op::Binary(Binary::Mul, cap, th));
                self.set(*x_out, out);
            }
            Elemwise::GateSigmoidMul {
                x,
                gate,
                heads: None,
                x_out,
            } => {
                let s = self.sigmoid(self.get(*gate)?);
                let out = self.push(Op::Binary(Binary::Mul, self.get(*x)?, s));
                self.set(*x_out, out);
            }
            Elemwise::Clamp {
                x,
                lo: None,
                hi: None,
                bounds: Bounds::Fixed { lo, hi },
                x_out,
            } => {
                let (lo, hi) = (self.konst(*lo), self.konst(*hi));
                let up = self.push(Op::Binary(Binary::Max, self.get(*x)?, lo));
                let out = self.push(Op::Binary(Binary::Min, up, hi));
                self.set(*x_out, out);
            }
            _ => return None,
        }
        Some(())
    }
}

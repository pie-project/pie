//! Vertical fusion into generated kernels: the elementwise and row ops a
//! trace runs back to back, formed into chains and each lowered to an eta
//! region (`eta_compiler::region::Region`, a function of one row) that
//! kernels-* `synth` builds one kernel for.
//!
//! **Forming** is conservative, as the fuse rules are: a chain is a run of
//! consecutive ops, each one this module lowers, each reading a value an
//! earlier op of the run computes, all under one guard and layer and over
//! rows of one width; at most [`MAX_CHAIN`] ops, and at least two (a lone op
//! keeps its hand kernel). Nothing is searched, so no unplanned combination
//! forms.
//!
//! **Lowering** turns each op into the eta math it states, in f32. A value
//! the run reads but does not compute is a parameter; a value it computes
//! that anything past the run may read (another op, a seam, a merge, or
//! nothing at all, as the trace's last values) is a result, stored in its
//! own dtype. A value only the run reads never leaves the registers, so the
//! rounding it had between two kernels is gone: a generated kernel is held
//! to the ops it replaces within a tolerance, not bit for bit.
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
use std::ops::Range;

use eta_compiler::plan::{Dimension, SymbolicType};
use eta_compiler::region::{Body, Region};
use eta_ir::op::Op;
use eta_ir::types::Literal;
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
    /// The trace's ops the region replaces, in order.
    pub nodes: Range<u32>,
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
}

/// A value a chain writes, a row a row in its own dtype: the kernels' `Out`.
/// One the trace computes in place (a residual's stream) is stored over
/// its input's plane, as the trace's placement says.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Out {
    pub value: ValueId,
    pub dtype: Dtype,
}

/// A generated kernel's region and the body the stream core wrote for it:
/// what a `fused.region` op's `body` hash names.
#[derive(Debug, Clone, PartialEq)]
pub struct Synth {
    pub region: Region,
    pub body: Body,
}

/// `trace` with each chain whose body the stream core writes replaced by
/// one `fused.region` op, and the regions its ops name, by body hash. Two
/// chains of one body (a layer's and the next's) share one entry.
#[must_use]
pub fn fold(trace: Trace) -> (Trace, BTreeMap<u64, Synth>) {
    let mut bodies = BTreeMap::new();
    let mut folded: Vec<(Range<u32>, Fused)> = Vec::new();
    for chain in chains(&trace) {
        let Ok(body) = chain.region.body() else {
            continue;
        };
        let Ok(Some(width)) = chain.region.width() else {
            continue;
        };
        folded.push((
            chain.nodes.clone(),
            Fused::Region {
                ins: chain.ins.iter().map(|input| input.value).collect(),
                width,
                body: body.hash,
                outs: chain.outs.iter().map(|out| out.value).collect(),
            },
        ));
        bodies.entry(body.hash).or_insert(Synth {
            region: chain.region,
            body,
        });
    }
    if folded.is_empty() {
        return (trace, bodies);
    }
    (super::search::compact(replace(trace, folded)), bodies)
}

/// `trace` with each run's ops replaced by its op, landed where the run
/// began, under the run's guard and layer; the runs are in order and apart.
fn replace(mut trace: Trace, runs: Vec<(Range<u32>, Fused)>) -> Trace {
    let mut landed = Vec::with_capacity(trace.nodes.len());
    let mut nodes = Vec::with_capacity(trace.nodes.len());
    let mut runs = runs.into_iter().peekable();
    let mut inside: Option<(u32, u32)> = None;
    for (at, node) in (0u32..).zip(std::mem::take(&mut trace.nodes)) {
        if let Some((end, to)) = inside {
            if at < end {
                landed.push(to);
                continue;
            }
            inside = None;
        }
        let here = u32::try_from(nodes.len()).unwrap_or(u32::MAX);
        landed.push(here);
        if runs.peek().is_some_and(|(run, _)| run.start == at) {
            let (run, op) = runs.next().expect("peeked");
            inside = Some((run.end, here));
            nodes.push(Node {
                op: Operation::Fused(op),
                guard: node.guard,
                layer: node.layer,
            });
        } else {
            nodes.push(node);
        }
    }
    for decl in &mut trace.values {
        if let Def::Op(i) = &mut decl.def
            && let Some(&to) = landed.get(*i as usize)
        {
            *i = to;
        }
    }
    trace.nodes = nodes;
    trace
}

/// Every chain of `trace`, in order; ops outside them run as they are.
#[must_use]
pub fn chains(trace: &Trace) -> Vec<Chain> {
    let escapes = Escapes::of(trace);
    let mut out = Vec::new();
    let mut at = 0u32;
    let count = u32::try_from(trace.nodes.len()).unwrap_or(u32::MAX);
    while at < count {
        let Some(row) = row_of(trace, at) else {
            at += 1;
            continue;
        };
        let mut end = at + 1;
        while end < count && ((end - at) as usize) < MAX_CHAIN && joins(trace, at..end, end, row) {
            end += 1;
        }
        if end - at >= 2
            && let Some(chain) = lower(trace, at..end, &escapes)
        {
            out.push(chain);
            at = end;
        } else {
            at += 1;
        }
    }
    out
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
    let mut ids = Vec::new();
    op.inputs(&mut ids);
    op.outputs(&mut ids);
    let mut rows: Option<Rows> = None;
    let mut width: Option<u64> = None;
    for v in ids {
        let (dim, w) = match form(trace, v)? {
            Form::Rows { dim, width } => (Some(dim), width),
            Form::Shared { width } => (None, width),
        };
        if width.is_some_and(|held| held != w) {
            return None;
        }
        width = Some(w);
        if let Some(dim) = dim {
            match rows {
                Some(held) if held.dim != dim => return None,
                _ => rows = Some(Rows { dim, width: w }),
            }
        }
    }
    rows
}

/// Whether `node` extends the run `run` over `row`: it lowers over the same
/// rows, under the run's guard and layer, and reads what the run computes.
fn joins(trace: &Trace, run: Range<u32>, node: u32, row: Rows) -> bool {
    if row_of(trace, node) != Some(row) {
        return false;
    }
    let first = &trace.nodes[run.start as usize];
    let next = &trace.nodes[node as usize];
    if next.guard != first.guard || next.layer != first.layer {
        return false;
    }
    let mut ins = Vec::new();
    next.op.inputs(&mut ins);
    ins.iter()
        .any(|v| matches!(trace.values[v.0 as usize].def, Def::Op(by) if run.contains(&by)))
}

/// How a value a chain touches is laid out.
enum Form {
    /// One row of `width` a row of `dim`.
    Rows { dim: Dim, width: u64 },
    /// One row of `width` every row reads: a weight.
    Shared { width: u64 },
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
            Elemwise::Act { s, act, .. } => {
                s.is_none() && !matches!(act, ActFn::Gelu { tanh: false })
            }
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
    fn leaves(&self, v: ValueId, run: &Range<u32>) -> bool {
        let readers = &self.readers[v.0 as usize];
        self.held[v.0 as usize] || readers.is_empty() || readers.iter().any(|r| !run.contains(r))
    }
}

/// The region `run` lowers to, or `None` where an op of it does not lower.
fn lower(trace: &Trace, run: Range<u32>, escapes: &Escapes) -> Option<Chain> {
    let row = row_of(trace, run.start)?;
    let width = u32::try_from(row.width).ok()?;

    // Parameters: what the run reads and does not compute, in first-read
    // order.
    let mut computed: Vec<ValueId> = Vec::new();
    let mut ins: Vec<In> = Vec::new();
    let mut ids = Vec::new();
    for node in run.clone() {
        let op = &trace.nodes[node as usize].op;
        ids.clear();
        op.inputs(&mut ids);
        for &v in &ids {
            if !computed.contains(&v) && !ins.iter().any(|held| held.value == v) {
                ins.push(In {
                    value: v,
                    dtype: dtype_of(trace, v),
                    shared: matches!(form(trace, v)?, Form::Shared { .. }),
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
    for node in run.clone() {
        b.node(&trace.nodes[node as usize].op)?;
    }

    let outs: Vec<Out> = computed
        .iter()
        .filter(|&&v| escapes.leaves(v, &run))
        .map(|&v| Out {
            value: v,
            dtype: dtype_of(trace, v),
        })
        .collect();
    let results = outs
        .iter()
        .map(|out| b.of.get(&out.value).copied())
        .collect::<Option<Vec<u32>>>()?;
    let row_type = SymbolicType {
        dtype: eta_ir::Dtype::F32,
        dims: vec![Dimension::Static(width)],
    };
    let region = Region {
        params: vec![row_type; ins.len()],
        ops: b.ops,
        results,
    };
    region.types().ok()?;
    Some(Chain {
        nodes: run,
        ins,
        outs,
        region,
    })
}

/// Eta ops for a run, one trace op at a time.
struct Builder {
    params: u32,
    width: u32,
    ops: Vec<Op>,
    /// The eta value each trace value is.
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
        let neg = self.push(Op::Neg(t));
        let exp = self.push(Op::Exp(neg));
        let den = self.push(Op::Add(one, exp));
        self.push(Op::Recip(den))
    }

    /// `1 - 2 / (e^2t + 1)`: eta has no tanh.
    fn tanh(&mut self, t: u32) -> u32 {
        let (one, two) = (self.konst(1.0), self.konst(2.0));
        let twice = self.push(Op::Mul(t, two));
        let exp = self.push(Op::Exp(twice));
        let den = self.push(Op::Add(exp, one));
        let frac = self.push(Op::Div(two, den));
        self.push(Op::Sub(one, frac))
    }

    /// The row's mean: its sum over the width.
    fn mean(&mut self, x: u32) -> u32 {
        let sum = self.push(Op::ReduceSum(x));
        let n = self.konst(self.width as f32);
        self.push(Op::Div(sum, n))
    }

    /// `x · rsqrt(mean(x²) + eps)`, as the hand kernels compute it.
    fn rms_normed(&mut self, x: u32, eps: f32) -> u32 {
        let sq = self.push(Op::Mul(x, x));
        let ms = self.mean(sq);
        let eps = self.konst(eps);
        let shifted = self.push(Op::Add(ms, eps));
        let r = self.push(Op::Rsqrt(shifted));
        self.push(Op::Mul(x, r))
    }

    /// `n · w`, or `n · (w + 1)`.
    fn scaled(&mut self, n: u32, w: u32, plus_one: bool) -> u32 {
        let w = if plus_one {
            let one = self.konst(1.0);
            self.push(Op::Add(w, one))
        } else {
            w
        };
        self.push(Op::Mul(n, w))
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
                let h = self.push(Op::Add(self.get(*y)?, self.get(*x)?));
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
                let h = self.push(Op::Add(self.get(*y)?, self.get(*x)?));
                self.set(*y_out, h);
            }
            Elemwise::AddBias { bias, out, out_out } => {
                let z = self.push(Op::Add(self.get(*out)?, self.get(*bias)?));
                self.set(*out_out, z);
            }
            Elemwise::Binary { x, y, op, z } => {
                let (x, y) = (self.get(*x)?, self.get(*y)?);
                let id = self.push(match op {
                    BinaryOp::Add => Op::Add(x, y),
                    BinaryOp::Mul => Op::Mul(x, y),
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
                let d = self.push(Op::Sub(x, mean));
                let sq = self.push(Op::Mul(d, d));
                let var = self.mean(sq);
                let eps = self.konst(*eps);
                let shifted = self.push(Op::Add(var, eps));
                let r = self.push(Op::Rsqrt(shifted));
                let mut out = self.push(Op::Mul(d, r));
                if let Some(w) = weight {
                    out = self.push(Op::Mul(out, self.get(*w)?));
                }
                if let Some(b) = bias {
                    out = self.push(Op::Add(out, self.get(*b)?));
                }
                self.set(*y, out);
            }
            Elemwise::Act {
                s: None,
                x,
                act,
                scale,
                x_out,
            } => {
                let mut t = self.get(*x)?;
                if let Some(k) = scale {
                    let k = self.konst(*k);
                    t = self.push(Op::Mul(t, k));
                }
                let out = match act {
                    ActFn::Identity => t,
                    ActFn::Silu => {
                        let s = self.sigmoid(t);
                        self.push(Op::Mul(t, s))
                    }
                    ActFn::Tanh => self.tanh(t),
                    ActFn::Gelu { tanh: true } => {
                        let (half, one) = (self.konst(0.5), self.konst(1.0));
                        let (c, cubic) = (self.konst(GELU_SQRT_2_OVER_PI), self.konst(GELU_CUBIC));
                        let t2 = self.push(Op::Mul(t, t));
                        let t3 = self.push(Op::Mul(t2, t));
                        let k3 = self.push(Op::Mul(cubic, t3));
                        let inner = self.push(Op::Add(t, k3));
                        let arg = self.push(Op::Mul(c, inner));
                        let th = self.tanh(arg);
                        let up = self.push(Op::Add(one, th));
                        let ht = self.push(Op::Mul(half, t));
                        self.push(Op::Mul(ht, up))
                    }
                    ActFn::Gelu { tanh: false } => return None,
                };
                self.set(*x_out, out);
            }
            Elemwise::Softcap { x, cap, x_out } => {
                let cap = self.konst(*cap);
                let t = self.push(Op::Div(self.get(*x)?, cap));
                let th = self.tanh(t);
                let out = self.push(Op::Mul(cap, th));
                self.set(*x_out, out);
            }
            Elemwise::GateSigmoidMul {
                x,
                gate,
                heads: None,
                x_out,
            } => {
                let s = self.sigmoid(self.get(*gate)?);
                let out = self.push(Op::Mul(self.get(*x)?, s));
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
                let up = self.push(Op::MaxElem(self.get(*x)?, lo));
                let out = self.push(Op::MinElem(up, hi));
                self.set(*x_out, out);
            }
            _ => return None,
        }
        Some(())
    }
}

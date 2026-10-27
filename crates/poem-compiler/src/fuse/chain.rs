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
//! longest part the backend serves (`Serves`) is the chain. Nothing else is searched,
//! so no unplanned combination forms.
//!
//! **Lowering** turns each op into the primitive math it states, in f32. A value
//! the run reads but does not compute is a parameter; a value it computes
//! that anything past the run may read (another op, an output, a merge, or
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
//! parameter, so the arena gives each its own plane: a row skeleton's threads
//! read their row whole before any writes it.
//!
//! Chains are formed and folded only under `compiler.synth_regions` (off by
//! default).

use std::collections::BTreeMap;

use kernels_ir::build::{self, Ops};
use kernels_ir::{Binary, Op, Reduce, Region, Shape, Unary};
use poem_ir::{
    ActFn, BinaryOp, Bounds, Def, Dim, Dtype, Elemwise, Fused, Gather, GluAct, Linear,
    ModulateForm, Node, NormAxis, NormWeight, Operands, Operation, RegionRead, ResidualGate, Trace,
    Turn, Ty, ValueId,
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
    /// One element a row, its own (a position): a scalar parameter.
    pub each: bool,
    /// The ids its rows are picked by (a request lane's modulation, read by
    /// each of the lane's rows): another parameter, one element a row.
    pub by: Option<ValueId>,
    /// The first of its columns read (one half of a packed modulation).
    pub from: u32,
}

/// A value a chain writes, a row a row in its own dtype: the kernels' `Out`,
/// on its own plane, never over an input's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Out {
    pub value: ValueId,
    pub dtype: Dtype,
}

/// Whether a backend's row kernels serve a region, its parameters read and
/// its results written as the chain stores them (a region states no
/// storage). A chain it does not serve is never folded, so an engine never
/// meets a region it cannot launch.
pub type Serves<'a> = &'a dyn Fn(&Region, &[In], &[Out]) -> bool;

/// `trace` with each chain the row core schedules and `serves` takes
/// replaced by one `fused.region` op, and the regions its ops name, by key
/// ([`Region::key`]). Two chains of one region (a layer's and the next's)
/// share one entry.
#[must_use]
pub fn fold(trace: Trace, serves: Serves<'_>) -> (Trace, BTreeMap<u64, Region>) {
    let mut regions = BTreeMap::new();
    let mut folded: Vec<(Vec<u32>, Fused)> = Vec::new();
    for chain in chains(&trace, serves) {
        let width = chain.region.width;
        let reads: Vec<RegionRead> = chain
            .ins
            .iter()
            .map(|input| RegionRead {
                by: input.by.and_then(|by| {
                    chain
                        .ins
                        .iter()
                        .position(|held| held.value == by && held.each)
                        .and_then(|k| u32::try_from(k).ok())
                }),
                from: input.from,
            })
            .collect();
        let plain = reads.iter().all(|read| *read == RegionRead::default());
        folded.push((
            chain.nodes.clone(),
            Fused::Region {
                ins: chain.ins.iter().map(|input| input.value).collect(),
                reads: if plain { Vec::new() } else { reads },
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

/// `trace` with each run's ops replaced by its op, landed where the run's
/// last op was, under the run's guard and in the layer of its last op that
/// has one (a fuse rule's match lands so too); the runs are apart.
pub(super) fn replace(mut trace: Trace, runs: Vec<(Vec<u32>, Fused)>) -> Trace {
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
pub(super) const REACH: u32 = 64;

/// Every chain of `trace` the row core schedules and `serves` takes, by its
/// first op; ops outside them run as they are.
#[must_use]
pub fn chains(trace: &Trace, serves: Serves<'_>) -> Vec<Chain> {
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
        if let Some(chain) = held(trace, &run, &escapes, serves) {
            for &n in &chain.nodes {
                taken[n as usize] = true;
            }
            out.push(chain);
        }
    }
    // A run's chain may be a later part of it, and the next run's lie
    // before it: in trace order.
    out.sort_by_key(|chain| chain.nodes[0]);
    out
}

/// The longest part of `run` (the latest of the longest, so a chain that
/// ends in a norm keeps it, as the rules' matches do) that is a chain the
/// backend serves: its ops lower, the ops between them keep their order,
/// the row core schedules its region and `serves` takes it.
fn held(trace: &Trace, run: &[u32], escapes: &Escapes, serves: Serves<'_>) -> Option<Chain> {
    for len in (2..=run.len()).rev() {
        for from in (0..=run.len() - len).rev() {
            let part = &run[from..from + len];
            if !keeps_order(trace, part) {
                continue;
            }
            let Some(chain) = lower(trace, part, escapes) else {
                continue;
            };
            let fits = kernels_ir::schedule(&chain.region).is_ok()
                && serves(&chain.region, &chain.ins, &chain.outs)
                && lands_apart(trace, &chain);
            if fits {
                return Some(chain);
            }
        }
    }
    None
}

/// Whether each value `chain` writes lands on a plane apart from every one
/// it reads, once it is one op: no result is a merge's arm (or tied in
/// place by an op past the chain) beside a value it reads, the plane the
/// merge's arms all land on. The ties the chain's own ops make in place
/// are its op's to break, each result a plane of its own.
fn lands_apart(trace: &Trace, chain: &Chain) -> bool {
    lands_apart_of(trace, &chain.nodes, &chain.ins, &chain.outs)
}

/// [`lands_apart`] of the values `ins` and `outs` an op replacing `nodes`
/// reads and writes.
pub(super) fn lands_apart_of(trace: &Trace, nodes: &[u32], ins: &[In], outs: &[Out]) -> bool {
    let mut tied: Vec<u32> = (0..u32::try_from(trace.values.len()).unwrap_or(u32::MAX)).collect();
    fn find(tied: &mut [u32], mut v: u32) -> u32 {
        while tied[v as usize] != v {
            tied[v as usize] = tied[tied[v as usize] as usize];
            v = tied[v as usize];
        }
        v
    }
    let mut tie = |a: ValueId, b: ValueId| {
        let (a, b) = (find(&mut tied, a.0), find(&mut tied, b.0));
        tied[a as usize] = b;
    };
    for (at, decl) in (0u32..).zip(&trace.values) {
        if let Def::Merge(arms) = &decl.def {
            for (arm, _) in arms {
                tie(ValueId(at), *arm);
            }
        }
    }
    let mut pairs = Vec::new();
    for (at, node) in (0u32..).zip(&trace.nodes) {
        if nodes.contains(&at) {
            continue;
        }
        pairs.clear();
        node.op.aliases(&mut pairs);
        for &(out, overwritten) in &pairs {
            tie(out, overwritten);
        }
    }
    outs.iter().all(|out| {
        ins.iter()
            .all(|input| find(&mut tied, out.value.0) != find(&mut tied, input.value.0))
    })
}

/// Whether `node` reads a value an op of `run` computes.
pub(super) fn reads_from(trace: &Trace, run: &[u32], node: u32) -> bool {
    let mut ins = Vec::new();
    trace.nodes[node as usize].op.inputs(&mut ins);
    ins.iter()
        .any(|v| matches!(trace.values[v.0 as usize].def, Def::Op(by) if run.contains(&by)))
}

/// Whether the ops between the first and the last of `run` keep their
/// order once the run lands where its last op was: none reads what the run
/// computes or a cache it reads, and none writes in place what it reads.
pub(super) fn keeps_order(trace: &Trace, run: &[u32]) -> bool {
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
pub(super) struct Rows {
    pub(super) dim: Dim,
    pub(super) width: u64,
}

/// The rows `node` runs over, if this module lowers it: every value it
/// reads or writes is a row of one width (or a weight that wide).
pub(super) fn row_of(trace: &Trace, node: u32) -> Option<Rows> {
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
    // what the op reads of a row by id or from a column on is a row's part
    // of another plane, its width that plane's
    let parted = reads_of(op, 0)
        .into_iter()
        .filter(|read| read.by.is_some() || read.from > 0 || read.half)
        .map(|read| read.value)
        .collect::<Vec<_>>();
    ins.retain(|v| !parted.contains(v));
    let mut rows: Option<Rows> = None;
    let mut width: Option<u64> = None;
    for (v, output) in ins
        .into_iter()
        .map(|v| (v, false))
        .chain(outs.into_iter().map(|v| (v, true)))
    {
        let (dim, w) = match form(trace, v)? {
            Form::Rows { dim, width } => (Some(dim), width),
            // a head's weight, every head reading it, is narrower than the
            // row; a row's own width is its outputs'
            Form::Shared { .. } | Form::One | Form::Each | Form::Table { .. } => continue,
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
    // A weight is the row's width, or one head's of it.
    let held = rows?;
    for v in ins_of(op) {
        if let Some(Form::Shared { width }) = form(trace, v)
            && (width == 0 || held.width % width != 0)
        {
            return None;
        }
    }
    Some(held)
}

fn ins_of(op: &Operation) -> Vec<ValueId> {
    let mut ins = Vec::new();
    op.inputs(&mut ins);
    ins
}

/// One value an op reads, and how: as its own rows, or by id (`by`, one
/// element a row) and from a column on; `half`, a part of a wider row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Read {
    pub(super) value: ValueId,
    pub(super) by: Option<ValueId>,
    pub(super) from: u32,
    pub(super) half: bool,
}

impl Read {
    const fn plain(value: ValueId) -> Self {
        Read {
            value,
            by: None,
            from: 0,
            half: false,
        }
    }
}

/// What `op` reads, over rows of `width`: each input as its own rows, but a
/// modulation's vectors (scale, then shift, of a row twice as wide) and a
/// residual's gate, each its row's or picked by the row's lane.
pub(super) fn reads_of(op: &Operation, width: u32) -> Vec<Read> {
    let parted = |value: ValueId, by: Option<ValueId>, from: u32| Read {
        value,
        by,
        from,
        half: true,
    };
    match op {
        Operation::Elemwise(Elemwise::Modulate {
            x,
            m,
            lane_of_row,
            form,
            ..
        }) => {
            let mut reads = vec![Read::plain(*x), parted(*m, *lane_of_row, 0)];
            if *form == ModulateForm::ScaleShift {
                reads.push(parted(*m, *lane_of_row, width));
            }
            reads.extend(lane_of_row.map(Read::plain));
            reads
        }
        Operation::Elemwise(Elemwise::ResidualAdd {
            x,
            gate: Some(ResidualGate { g, lane_of_row }),
            y,
            ..
        }) => {
            let mut reads = vec![
                Read::plain(*y),
                parted(*g, *lane_of_row, 0),
                Read::plain(*x),
            ];
            reads.extend(lane_of_row.map(Read::plain));
            reads
        }
        Operation::Gather(Gather::Embed { ids, table, .. }) => {
            vec![parted(*table, Some(*ids), 0), Read::plain(*ids)]
        }
        Operation::Gather(Gather::Select {
            table,
            layer,
            width: part,
            ..
        }) => vec![parted(*table, None, layer.saturating_mul(*part))],
        _ => ins_of(op).into_iter().map(Read::plain).collect(),
    }
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
pub(super) enum Form {
    /// One row of `width` a row of `dim`.
    Rows { dim: Dim, width: u64 },
    /// One row of `width` every row reads: a weight (the row's width, or
    /// one head's, every head reading it).
    Shared { width: u64 },
    /// One element every row reads: a weight of one element.
    One,
    /// One element a row, its own: a position.
    Each,
    /// A weight's rows of `width`, each read where an id picks it (an
    /// embedding's table).
    Table { width: u64 },
}

pub(super) fn form(trace: &Trace, v: ValueId) -> Option<Form> {
    let decl = &trace.values[v.0 as usize];
    let Ty::Tensor { shape, dtype } = &decl.ty else {
        return None;
    };
    let stored = match decl.def {
        Def::Weight(i) => trace.params[i as usize].dtype,
        _ => *dtype,
    };
    let rowed = |dim: &Dim| !matches!(dim, Dim::Const(_));
    let each = match shape.as_slice() {
        [dim] | [dim, Dim::Const(1)] => rowed(dim) && !matches!(decl.def, Def::Weight(_)),
        _ => false,
    };
    if each && matches!(stored, Dtype::Bf16 | Dtype::F32 | Dtype::I32) {
        return Some(Form::Each);
    }
    if !matches!(stored, Dtype::Bf16 | Dtype::F16 | Dtype::F32) {
        return None;
    }
    match (shape.as_slice(), &decl.def) {
        ([Dim::Const(1)], Def::Weight(_)) => Some(Form::One),
        ([Dim::Const(width)], Def::Weight(_)) => Some(Form::Shared { width: *width }),
        ([Dim::Const(_), Dim::Const(width)], Def::Weight(_)) => Some(Form::Table { width: *width }),
        ([dim, Dim::Const(width)], _) if !matches!(dim, Dim::Const(_)) => Some(Form::Rows {
            dim: *dim,
            width: *width,
        }),
        _ => None,
    }
}

pub(super) fn dtype_of(trace: &Trace, v: ValueId) -> Dtype {
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
            Elemwise::ResidualAdd { .. } => true,
            // a modulation by its scale (and shift), not tanh's gate (a
            // tanh the kernels take as `tanhf`)
            Elemwise::Modulate { form, .. } => {
                matches!(form, ModulateForm::ScaleShift | ModulateForm::Scale)
            }
            Elemwise::AddBias { .. } => true,
            Elemwise::Binary { .. } => true,
            Elemwise::Rmsnorm { axis, .. } => matches!(axis, NormAxis::Row | NormAxis::Head(_)),
            // a rotation of q alone at the frequencies it states (`Last`'s
            // ramped by YaRN where given)
            Elemwise::Rope {
                k: None,
                freqs: None,
                turn,
                ..
            } => matches!(turn, Turn::Partial { .. } | Turn::Last { .. }),
            Elemwise::Layernorm { .. } => true,
            Elemwise::Act { act, .. } => !matches!(act, ActFn::Gelu { tanh: false }),
            Elemwise::Softcap { .. } => true,
            Elemwise::GateSigmoidMul { heads, .. } => heads.is_none(),
            Elemwise::Clamp { bounds, .. } => matches!(bounds, Bounds::Fixed { .. }),
            // AttnRes's blend of a row's closed blocks and its prefix
            Elemwise::ResBlend { .. } => true,
            _ => false,
        },
        Operation::Fused(Fused::ResidualAddRmsnorm { .. }) => true,
        // an embedding's rows, read by token id (a dense table, no weights
        // or scales on its rows), and one layer's row of a stacked plane
        Operation::Gather(Gather::Embed {
            weights: None,
            scales: None,
            concat: false,
            ..
        })
        | Operation::Gather(Gather::Select { .. }) => true,
        _ => false,
    }
}

/// The rotation `turn` states of `head_dim`-wide heads of a row of `width`
/// at `theta`, as kernels-ir's builder turns it; `None` for a turn it does not.
pub(super) fn rotation(width: u32, head_dim: u32, theta: f32, turn: &Turn) -> Option<build::Turn> {
    if head_dim == 0 || !width.is_multiple_of(head_dim) {
        return None;
    }
    let mut mscale = None;
    let (pairs, inverse, ramp) = match *turn {
        // every pair at YaRN's stretched frequencies (`theta^(-2d /
        // head_dim)` ramped), cos and sin times the attention factor, as the
        // hand YaRN turn lands it
        Turn::Yarn {
            yarn,
            attention_factor,
            interleaved,
        } if head_dim.is_multiple_of(2) && yarn.original_max_position > 0 => {
            let (low, high) = ramp_bounds(
                i32::try_from(head_dim).ok()?,
                theta,
                yarn.beta_fast,
                yarn.beta_slow,
                i32::try_from(yarn.original_max_position).ok()?,
            );
            mscale = Some(attention_factor);
            (
                build::Pairs::Tail {
                    rotary: head_dim,
                    interleaved,
                },
                false,
                Some(build::Ramp {
                    factor: yarn.factor,
                    low,
                    high,
                }),
            )
        }
        // every pair: the halves `i`, `i + head_dim / 2` at `theta^(-i /
        // (head_dim / 2))`, or `2i`, `2i + 1` at `theta^(-2i / head_dim)`
        Turn::Full { interleaved } if head_dim.is_multiple_of(2) => (
            if interleaved {
                build::Pairs::Tail {
                    rotary: head_dim,
                    interleaved: true,
                }
            } else {
                build::Pairs::Halves {
                    rotary: head_dim,
                    span: head_dim / 2,
                }
            },
            false,
            None,
        ),
        Turn::Partial { rotary_dim, span } => {
            if rotary_dim % 2 != 0 || span < rotary_dim / 2 || span + rotary_dim / 2 > head_dim {
                return None;
            }
            (
                build::Pairs::Halves {
                    rotary: rotary_dim,
                    span,
                },
                false,
                None,
            )
        }
        Turn::Last {
            rotary_dim,
            interleaved,
            inverse,
            yarn,
        } => {
            if rotary_dim % 2 != 0 || rotary_dim == 0 || rotary_dim > head_dim {
                return None;
            }
            // The kernels ramp only a stretched rotation.
            let ramp = match yarn {
                Some(y) if y.factor > 1.0 => {
                    if y.original_max_position == 0 {
                        return None;
                    }
                    let (low, high) = ramp_bounds(
                        i32::try_from(rotary_dim).ok()?,
                        theta,
                        y.beta_fast,
                        y.beta_slow,
                        i32::try_from(y.original_max_position).ok()?,
                    );
                    Some(build::Ramp {
                        factor: y.factor,
                        low,
                        high,
                    })
                }
                _ => None,
            };
            (
                build::Pairs::Tail {
                    rotary: rotary_dim,
                    interleaved,
                },
                inverse,
                ramp,
            )
        }
        _ => return None,
    };
    Some(build::Turn {
        head_dim,
        pairs,
        theta,
        inverse,
        ramp,
        mscale,
    })
}

/// The dims YaRN's ramp runs between, as kernels-cuda's `ramp_bounds`
/// works them out (the same f32 arithmetic, so the same bounds).
#[allow(clippy::cast_precision_loss)]
fn ramp_bounds(
    head_dim: i32,
    theta: f32,
    beta_fast: f32,
    beta_slow: f32,
    original_max_position: i32,
) -> (f32, f32) {
    const TWO_PI: f32 = core::f32::consts::TAU;
    let ln_theta = theta.ln();
    let corr_dim = |rot: f32| -> f32 {
        head_dim as f32 * (original_max_position as f32 / (rot * TWO_PI)).ln() / (2.0 * ln_theta)
    };
    let mut low_dim = corr_dim(beta_fast).floor();
    let mut high_dim = corr_dim(beta_slow).ceil();
    if low_dim < 0.0 {
        low_dim = 0.0;
    }
    let max_pair = (head_dim / 2) as f32 - 1.0;
    if high_dim > max_pair {
        high_dim = max_pair;
    }
    if high_dim < low_dim {
        high_dim = low_dim;
    }
    (low_dim, high_dim)
}

/// Who may read each value past a chain: another op, an output or a merge.
pub(super) struct Escapes {
    readers: Vec<Vec<u32>>,
    held: Vec<bool>,
}

impl Escapes {
    pub(super) fn of(trace: &Trace) -> Escapes {
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
        for output in &trace.outputs {
            for v in output.values() {
                held[v.0 as usize] = true;
            }
        }
        Escapes { readers, held }
    }

    /// Whether `v`, computed inside `run`, may be read past it. A value
    /// nothing reads is the trace's to keep, so it leaves too.
    pub(super) fn leaves(&self, v: ValueId, run: &[u32]) -> bool {
        let readers = &self.readers[v.0 as usize];
        self.held[v.0 as usize] || readers.is_empty() || readers.iter().any(|r| !run.contains(r))
    }
}

/// The region `run` lowers to, or `None` where an op of it does not lower.
fn lower(trace: &Trace, run: &[u32], escapes: &Escapes) -> Option<Chain> {
    // An embedding's table holds its whole vocabulary, so an id past it
    // picks row 0 in the skeleton as in the gather (a table cut by its rows
    // gathers each rank's share).
    for &node in run {
        if let Operation::Gather(Gather::Embed { table, vocab, .. }) =
            &trace.nodes[node as usize].op
            && !matches!(&trace.values[table.0 as usize].ty,
                Ty::Tensor { shape, .. } if matches!(shape.as_slice(), [Dim::Const(rows), _] if *rows == u64::from(*vocab)))
        {
            return None;
        }
    }
    let row = row_of(trace, run[0])?;
    let row_width = u32::try_from(row.width).ok()?;
    // The region's row: a head, where every op is a head's alone (each
    // row cut into its heads, each the kernel's row), else the row.
    let width = part(trace, run, row_width).unwrap_or(row_width);

    // Parameters: what the run reads and does not compute, in first-read
    // order.
    let mut computed: Vec<ValueId> = Vec::new();
    let mut ins: Vec<In> = Vec::new();
    let mut ids = Vec::new();
    for &node in run {
        let op = &trace.nodes[node as usize].op;
        for read in reads_of(op, width) {
            let v = read.value;
            let held = ins
                .iter()
                .any(|held| held.value == v && held.by == read.by && held.from == read.from);
            if computed.contains(&v) || held {
                continue;
            }
            let form = form(trace, v)?;
            if read.half {
                // a row's part of a plane of rows (its own, or a lane's, or
                // a table's row an id picks)
                let (Form::Rows { width: whole, .. } | Form::Table { width: whole }) = form else {
                    return None;
                };
                if matches!(form, Form::Table { .. }) && read.by.is_none() {
                    return None;
                }
                if u64::from(read.from) + u64::from(width) > whole {
                    return None;
                }
            }
            ins.push(In {
                value: v,
                dtype: dtype_of(trace, v),
                shared: matches!(form, Form::Shared { .. }),
                one: matches!(form, Form::One),
                each: matches!(form, Form::Each) && !read.half,
                by: read.by,
                from: read.from,
            });
        }
        ids.clear();
        op.outputs(&mut ids);
        computed.extend_from_slice(&ids);
    }

    let mut b = Builder::new(u32::try_from(ins.len()).ok()?, width);
    for (k, input) in (0u32..).zip(&ins) {
        if input.by.is_some() || input.from > 0 || reads_parted(trace, run, input.value) {
            b.parts.insert((input.value, input.from), k);
        } else {
            b.of.insert(input.value, k);
        }
    }
    for &node in run {
        b.node(trace, &trace.nodes[node as usize].op)?;
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
        if id >= b.b.params && !rounds.iter().any(|(held, _)| *held == id) {
            rounds.push((id, narrow));
        }
    }
    // A parameter is a row of the chain's width (a head's weight repeated
    // across the row's heads by its plane), or a scalar.
    let params = ins
        .iter()
        .map(|input| {
            Some(match form(trace, input.value)? {
                Form::One => Shape::Scalar,
                Form::Each if input.by.is_none() && input.from == 0 => Shape::Scalar,
                _ => Shape::Row,
            })
        })
        .collect::<Option<Vec<_>>>()?;
    let region = Region {
        width,
        params,
        ops: b.b.ops,
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

/// The heads `run` cuts each row into, `head_dim` wide: every op of it
/// computes each head alone (a head's norm, a rotation of heads) or each
/// element (an add, an activation), all of one head width narrower than the
/// row, and every weight it reads one head wide. A whole row's op (a row's
/// norm, a weight as wide as the row) keeps the row whole.
fn part(trace: &Trace, run: &[u32], width: u32) -> Option<u32> {
    let mut part: Option<u32> = None;
    for &node in run {
        let op = &trace.nodes[node as usize].op;
        let head = match op {
            Operation::Elemwise(Elemwise::Rmsnorm {
                axis: NormAxis::Head(head),
                ..
            })
            | Operation::Elemwise(Elemwise::Rope { head_dim: head, .. }) => Some(*head),
            Operation::Elemwise(
                Elemwise::ResidualAdd { gate: None, .. }
                | Elemwise::Binary { .. }
                | Elemwise::Act { .. }
                | Elemwise::Softcap { .. }
                | Elemwise::GateSigmoidMul { heads: None, .. }
                | Elemwise::Clamp { .. },
            ) => None,
            _ => return None,
        };
        match (part, head) {
            (_, None) => {}
            (None, Some(head)) => part = Some(head),
            (Some(held), Some(head)) if held == head => {}
            _ => return None,
        }
        for v in ins_of(op) {
            if let Some(Form::Shared { width }) = form(trace, v)
                && Some(width) != part.map(u64::from)
            {
                return None;
            }
        }
    }
    part.filter(|&head| head > 0 && head < width && width.is_multiple_of(head))
}

/// Whether an op of `run` reads `v` as a part of a wider row (a
/// modulation's half, a lane's gate).
fn reads_parted(trace: &Trace, run: &[u32], v: ValueId) -> bool {
    run.iter().any(|&node| {
        reads_of(&trace.nodes[node as usize].op, 0)
            .iter()
            .any(|read| read.value == v && read.half)
    })
}

/// A region's ops for a run, one trace op at a time.
pub(super) struct Builder {
    pub(super) b: Ops,
    width: u32,
    /// The region's value each trace value is.
    pub(super) of: BTreeMap<ValueId, u32>,
    /// The parameter each part of a wider row is (a value and its first
    /// column).
    pub(super) parts: BTreeMap<(ValueId, u32), u32>,
}

impl Builder {
    /// A region's ops over rows `width` wide, of `params` parameters.
    pub(super) fn new(params: u32, width: u32) -> Builder {
        Builder {
            b: Ops::new(params),
            width,
            of: BTreeMap::new(),
            parts: BTreeMap::new(),
        }
    }

    fn push(&mut self, op: Op) -> u32 {
        self.b.push(op)
    }

    fn konst(&mut self, x: f32) -> u32 {
        self.b.f32(x)
    }

    pub(super) fn get(&self, v: ValueId) -> Option<u32> {
        self.of.get(&v).copied()
    }

    /// The part of `v`'s row from column `from`, a row of the region's.
    fn part(&self, v: ValueId, from: u32) -> Option<u32> {
        self.parts.get(&(v, from)).copied()
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

    /// `tanh(t)`, as the hand kernels' `tanhf`.
    fn tanh(&mut self, t: u32) -> u32 {
        self.push(Op::Unary(Unary::Tanh, t))
    }

    /// `0.5 t (1 + tanh(√(2/π) (t + 0.044715 t³)))`.
    fn gelu_tanh(&mut self, t: u32) -> u32 {
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

    /// The row's mean: its sum over the width.
    fn mean(&mut self, x: u32) -> u32 {
        let sum = self.push(Op::Reduce(Reduce::Sum, x));
        let n = self.konst(self.width as f32);
        self.push(Op::Binary(Binary::Div, sum, n))
    }

    /// `x · rsqrt(mean(x²) + eps)`, as the hand kernels compute it.
    fn rms_normed(&mut self, x: u32, eps: f32) -> u32 {
        self.b.rms_normed(x, self.width, eps)
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

    pub(super) fn node(&mut self, trace: &Trace, op: &Operation) -> Option<()> {
        match op {
            Operation::Elemwise(e) => self.elemwise(trace, e),
            // the row an id picks of the table, a parameter read so
            Operation::Gather(Gather::Embed { table, y, .. }) => {
                let row = self.part(*table, 0)?;
                self.set(*y, row);
                Some(())
            }
            Operation::Gather(Gather::Select {
                table,
                layer,
                width,
                y,
            }) => {
                let row = self.part(*table, layer.saturating_mul(*width))?;
                self.set(*y, row);
                Some(())
            }
            Operation::Linear(linear) => self.linear(linear),
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

    /// A product's activations, each pointwise over its tile: a GLU of a
    /// packed `[gate | up]` row's two halves (the row's parts, from column
    /// 0 and from the width on), and the ungated ones. Only a contraction's
    /// epilogue reads a product so (`fuse::anchor`); a chain lowers none.
    fn linear(&mut self, op: &Linear) -> Option<()> {
        match *op {
            Linear::Glu {
                packed: Some(packed),
                gate: None,
                up: None,
                weights: None,
                intermediate,
                act,
                limit: None,
                y,
            } if intermediate == self.width => {
                let (gate, up) = (self.part(packed, 0)?, self.part(packed, self.width)?);
                let gated = match act {
                    GluAct::Swiglu => {
                        let s = self.sigmoid(gate);
                        self.push(Op::Binary(Binary::Mul, gate, s))
                    }
                    GluAct::GegluTanh => self.gelu_tanh(gate),
                    _ => return None,
                };
                let out = self.push(Op::Binary(Binary::Mul, gated, up));
                self.set(y, out);
            }
            Linear::GeluTanh { x, y } => {
                let out = self.gelu_tanh(self.get(x)?);
                self.set(y, out);
            }
            Linear::Relu2 { x, y } => {
                let zero = self.konst(0.0);
                let r = self.push(Op::Binary(Binary::Max, self.get(x)?, zero));
                let out = self.push(Op::Binary(Binary::Mul, r, r));
                self.set(y, out);
            }
            _ => return None,
        }
        Some(())
    }

    fn elemwise(&mut self, trace: &Trace, e: &Elemwise) -> Option<()> {
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
            // `y + g ⊙ x`, one rounding, as the gated add's `fmaf` lands it
            Elemwise::ResidualAdd {
                x,
                gate: Some(ResidualGate { g, .. }),
                y,
                y_out,
            } => {
                let h = self.push(Op::Fma(self.part(*g, 0)?, self.get(*x)?, self.get(*y)?));
                self.set(*y_out, h);
            }
            // `x (1 + scale) + shift` in one rounding, or `x (1 + scale)`,
            // as the modulation kernels land it
            Elemwise::Modulate { x, m, form, y, .. } => {
                let x = self.get(*x)?;
                let one = self.konst(1.0);
                let grown = self.push(Op::Binary(Binary::Add, one, self.part(*m, 0)?));
                let out = match form {
                    ModulateForm::ScaleShift => {
                        self.push(Op::Fma(x, grown, self.part(*m, self.width)?))
                    }
                    ModulateForm::Scale => self.push(Op::Binary(Binary::Mul, x, grown)),
                    ModulateForm::TanhGate => return None,
                };
                self.set(*y, out);
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
            // A head as wide as the row is the row's norm.
            Elemwise::Rmsnorm {
                x,
                weight,
                axis: NormAxis::Head(head_dim),
                eps,
                y,
            } if *head_dim == self.width => {
                let n = self.rms_normed(self.get(*x)?, *eps);
                let out = match weight {
                    NormWeight::None => n,
                    NormWeight::Scale(w) => self.scaled(n, self.get(*w)?, false),
                    NormWeight::PlusOne(w) => self.scaled(n, self.get(*w)?, true),
                };
                self.set(*y, out);
            }
            // Each head normed on its own: the row read as its heads, one
            // weight one head wide across every head.
            Elemwise::Rmsnorm {
                x,
                weight,
                axis: NormAxis::Head(head_dim),
                eps,
                y,
            } => {
                let (width, head) = (self.width, *head_dim);
                if head == 0 || width % head != 0 {
                    return None;
                }
                let n = self.b.rms_normed_heads(self.get(*x)?, head, *eps);
                let n = match weight {
                    NormWeight::None => n,
                    NormWeight::Scale(w) | NormWeight::PlusOne(w) => {
                        let w = self.get(*w)?;
                        let w = if matches!(weight, NormWeight::PlusOne(_)) {
                            let one = self.konst(1.0);
                            self.push(Op::Binary(Binary::Add, w, one))
                        } else {
                            w
                        };
                        // one head's weight, its plane repeating it across the heads
                        self.push(Op::Binary(Binary::Mul, n, w))
                    }
                };
                self.set(*y, n);
            }
            Elemwise::Rope {
                q,
                k: None,
                positions,
                head_dim,
                theta,
                turn,
                freqs: None,
                q_out,
                k_out: None,
            } => {
                let turn = rotation(self.width, *head_dim, *theta, turn)?;
                let out = self.b.rotate(self.get(*q)?, self.get(*positions)?, &turn);
                self.set(*q_out, out);
            }
            // AttnRes's blend: each candidate (the closed blocks, then the
            // prefix) scaled by the norm's weight and the projection and
            // summed, times its RMS norm's scale, a logit a candidate (as
            // the hand kernels compute it); their softmax; and the
            // candidates summed under it.
            Elemwise::ResBlend {
                prefix,
                blocks,
                weight,
                proj,
                eps,
                y,
            } => {
                let candidates = blocks
                    .iter()
                    .chain(core::iter::once(prefix))
                    .map(|v| self.get(*v))
                    .collect::<Option<Vec<u32>>>()?;
                let (w, p) = (self.get(*weight)?, self.get(*proj)?);
                let logits: Vec<u32> = candidates
                    .iter()
                    .map(|&x| {
                        // the norm a scale of the row's sum, so a candidate's
                        // square sum and its dot are one pass's
                        let sq = self.push(Op::Binary(Binary::Mul, x, x));
                        let ss = self.push(Op::Reduce(Reduce::Sum, sq));
                        let n = self.konst(self.width as f32);
                        let ms = self.push(Op::Binary(Binary::Div, ss, n));
                        let e = self.konst(*eps);
                        let shifted = self.push(Op::Binary(Binary::Add, ms, e));
                        let r = self.push(Op::Unary(Unary::Rsqrt, shifted));
                        let xw = self.push(Op::Binary(Binary::Mul, x, w));
                        let d = self.push(Op::Binary(Binary::Mul, xw, p));
                        let dot = self.push(Op::Reduce(Reduce::Sum, d));
                        self.push(Op::Binary(Binary::Mul, dot, r))
                    })
                    .collect();
                let most = logits[1..]
                    .iter()
                    .fold(logits[0], |m, &l| self.push(Op::Binary(Binary::Max, m, l)));
                let exps: Vec<u32> = logits
                    .iter()
                    .map(|&l| {
                        let shifted = self.push(Op::Binary(Binary::Sub, l, most));
                        self.push(Op::Unary(Unary::Exp, shifted))
                    })
                    .collect();
                let sum = exps[1..]
                    .iter()
                    .fold(exps[0], |a, &e| self.push(Op::Binary(Binary::Add, a, e)));
                let mut blend = None;
                for (&x, &e) in candidates.iter().zip(&exps) {
                    let share = self.push(Op::Binary(Binary::Div, e, sum));
                    let term = self.push(Op::Binary(Binary::Mul, share, x));
                    blend = Some(match blend {
                        None => term,
                        Some(acc) => self.push(Op::Binary(Binary::Add, acc, term)),
                    });
                }
                self.set(*y, blend?);
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
                // weight and bias in one rounding, as the kernels' `fmaf`
                match (weight, bias) {
                    (Some(w), Some(b)) => {
                        out = self.push(Op::Fma(out, self.get(*w)?, self.get(*b)?));
                    }
                    (Some(w), None) => out = self.push(Op::Binary(Binary::Mul, out, self.get(*w)?)),
                    (None, Some(b)) => out = self.push(Op::Binary(Binary::Add, out, self.get(*b)?)),
                    (None, None) => {}
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
                    // a bare scale multiplies by itself as x's dtype holds
                    // it (`mul_scalar`), a scaled silu by the f32
                    let k = if *act == ActFn::Identity && dtype_of(trace, *x) == Dtype::Bf16 {
                        kernels_ir::skeleton::round_bf16(*k)
                    } else {
                        *k
                    };
                    let k = self.konst(k);
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
                    ActFn::Gelu { tanh: true } => self.gelu_tanh(t),
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

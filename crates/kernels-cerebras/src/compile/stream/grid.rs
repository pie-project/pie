//! The grid regime: a reading over a clip's voxel rows (a VAE's decode or encode) read off its
//! trace into passes over one unit, every grid op lowered into the stream regime's existing
//! modules. A convolution is a row gather (its im2col, from a line buffer at the roots) feeding
//! the tile matvec; upsampling and pixel (un)shuffling are gathers too; a group norm sums its
//! input's statistics over a pass and applies in a later one; the mid-block attention uses a K / V
//! column under the unit, stores one pass and queries the next.
//!
//! A pass's head makes the rows its gather reads into the line buffer, its tail the rows the
//! gather makes. A value read in a later pass is held in the unit's weight PEs (row `t` by column
//! or row `t % P`) and fetched back up its chain. The unit is `P` CX roots, `P` CY roots and
//! `P x P` weight PEs; `P` is the most roots (a power of two) every value cuts over evenly.

use std::collections::{BTreeMap, BTreeSet};

use poem_compiler::fuse::chain::{self, Chain};
use poem_ir::{
    Attn, Conv, Def, Dim, Dtype, InputAxis, Linear, Operands, Operation, RuntimeInput, Trace, Ty,
    ValueId,
};
use serde::Serialize;

use crate::compile::emit::region::{self, ParamAt, Place, Printed};
use crate::compile::fabric::{Fabric, Palette};
use crate::compile::floorplan::Requests;
use crate::compile::floorplan::stream::{Bounds, Stream, UnitAsk};
use crate::compile::stream::unit::Side;

type R<T> = Result<T, String>;

/// A row value of the reading: its width, rows (its grid's), side and cut.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Val {
    pub width: u32,
    pub grid: [u32; 3],
    pub side: Side,
    /// Whole at the side's first root (an attention's q / k / v / o), not cut evenly.
    pub head: bool,
    /// A 16-bit value: rounded where it is made.
    pub round: bool,
}

impl Val {
    #[must_use]
    pub fn rows(&self) -> u32 {
        self.grid[0] * self.grid[1] * self.grid[2]
    }
}

/// A gather's re-ordering of its source rows into each output row.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Kind {
    /// A `k` convolution's taps (stride `s`, front padding `p`), over its input grid, that grid
    /// the source's upsampled by `up` (1: the source's own).
    Conv {
        k: [u32; 2],
        s: [u32; 2],
        p: [u32; 2],
        up: u32,
    },
    /// Nearest upsampling by `f` an axis (h, w).
    Up(u32),
    /// Pixel shuffle by `r` (h, w): an output row is a channel of every `r²` of its source row.
    Shuffle(u32),
    /// Pixel unshuffle by `r`: an output row is `r²` source rows interleaved.
    Unshuffle(u32),
}

/// A pass's gather: `src`'s rows (on `side`, `src_grid`) re-ordered into the rows of `out`
/// (the gathered value: an im2col row, or the op's own output).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Gather {
    pub src: ValueId,
    pub out: ValueId,
    pub side: Side,
    pub kind: Kind,
    pub src_grid: [u32; 3],
    /// The grid the taps index (a convolution's input, the source upsampled).
    pub in_grid: [u32; 3],
    pub out_grid: [u32; 3],
    /// The line buffer's rows (a power of two): the most source rows a row's taps span.
    pub ring: u32,
}

/// The taps of a gather a row reads: (tap, source row) for each one inside the grid.
impl Gather {
    #[must_use]
    pub fn taps(&self) -> u32 {
        match self.kind {
            Kind::Conv { k, .. } => k[0] * k[1],
            Kind::Up(_) | Kind::Shuffle(_) => 1,
            Kind::Unshuffle(r) => r * r,
        }
    }

    /// Output row `(y, x)`'s taps: `(tap, source row)` for those inside the grid.
    #[must_use]
    pub fn sources(&self, y: u32, x: u32) -> Vec<(u32, u32)> {
        let ws = self.src_grid[2];
        let mut out = Vec::new();
        match self.kind {
            Kind::Conv { k, s, p, up } => {
                let [_, hi, wi] = self.in_grid;
                for a in 0..k[0] {
                    for b in 0..k[1] {
                        let iy = i64::from(y * s[0] + a) - i64::from(p[0]);
                        let ix = i64::from(x * s[1] + b) - i64::from(p[1]);
                        if iy < 0 || ix < 0 || iy >= i64::from(hi) || ix >= i64::from(wi) {
                            continue;
                        }
                        let (sy, sx) = (iy as u32 / up, ix as u32 / up);
                        out.push((a * k[1] + b, sy * ws + sx));
                    }
                }
            }
            Kind::Up(f) => out.push((0, (y / f) * ws + x / f)),
            Kind::Shuffle(r) => out.push((0, (y / r) * ws + x / r)),
            Kind::Unshuffle(r) => {
                for i in 0..r {
                    for j in 0..r {
                        out.push((i * r + j, (y * r + i) * ws + x * r + j));
                    }
                }
            }
        }
        out
    }

    /// The source rows pulled before output row `(y, x)` (its last tap's, plus one).
    #[must_use]
    pub fn need(&self, y: u32, x: u32) -> u32 {
        self.sources(y, x)
            .iter()
            .map(|s| s.1 + 1)
            .max()
            .unwrap_or(0)
    }
}

/// A resident parameter of a region: a weight's columns, or a group norm's scale or shift made
/// on the PE.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Res {
    Value { v: ValueId, from: u32 },
    GnScale(usize),
    GnShift(usize),
}

/// A region parameter: a row value's chunk, or a resident.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum RegIn {
    Row(ValueId),
    Res(Res),
}

/// A printed root region and what it reads and makes.
#[derive(Clone, Debug)]
pub struct Reg {
    pub side: Side,
    pub head: bool,
    pub region: kernels_ir::Region,
    pub ins: Vec<RegIn>,
    pub outs: Vec<ValueId>,
    pub printed: Option<Printed>,
    pub params: Vec<ParamAt>,
}

/// A matvec's input: a row value, or a gather's row (its `taps` of a `c_in`-wide source).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum MvIn {
    Value(ValueId),
    Gathered { taps: u32, c_in: u32 },
}

/// A matvec: `n` x `k` weights from the roots of `from` to the other side's.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Mv {
    pub node: u32,
    /// The weight (none for a move: an identity, its tile the constant one on the diagonal).
    pub weight: ValueId,
    pub ident: bool,
    pub k: u32,
    pub n: u32,
    pub from: Side,
    pub input: MvIn,
    pub output: ValueId,
}

/// A group norm: its input, output and parameters.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Gn {
    pub x: ValueId,
    pub y: ValueId,
    pub groups: u32,
    pub eps: f32,
    pub silu: bool,
    pub gamma: ValueId,
    pub beta: ValueId,
    pub reg: usize,
}

/// One step a row of a pass.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub enum Step {
    /// A held (or the host's) value's row in, up the chain it was kept on.
    Fetch(ValueId),
    Region(usize),
    Matvec(usize),
    /// The row kept in the weight PEs for a later pass.
    Hold(ValueId),
    /// The row summed into its value's statistics (a group norm's input).
    Stats(ValueId),
    /// The row to the host (kept in the weight PEs for it to read).
    Out(ValueId),
    /// The head's row into the line buffer.
    Ring(ValueId),
    /// The tail's row gathered from the line buffer.
    Gather,
    /// q / k / v into the K / V column.
    Store {
        q: ValueId,
        k: ValueId,
        v: ValueId,
    },
    /// The attention's output back.
    Attend(ValueId),
}

/// A sweep over a pass's rows.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Pass {
    /// The tail's rows' grid.
    pub grid: [u32; 3],
    pub gather: Option<Gather>,
    pub head: Vec<Step>,
    pub tail: Vec<Step>,
    /// The group norms whose scale and shift are made at the pass's end.
    pub finals: Vec<usize>,
}

impl Pass {
    #[must_use]
    pub fn rows(&self) -> u32 {
        self.grid[0] * self.grid[1] * self.grid[2]
    }

    /// The head's rows (the gather's source's).
    #[must_use]
    pub fn src_rows(&self) -> u32 {
        self.gather
            .as_ref()
            .map_or(0, |g| g.src_grid[0] * g.src_grid[1] * g.src_grid[2])
    }
}

/// The attention: its rows, its head's width and scale, the K / V column's PEs.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Attention {
    pub rows: u32,
    pub width: u32,
    pub scale: f32,
    pub pya: u32,
}

/// A held value's place in its side's store (16-bit words a weight PE), the host's rows' and the
/// outputs' (f32): its offset there.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Slots {
    /// Both sides' held values in one store (a row-side and a column-side value of a pass apart
    /// in it by their lifetimes alone).
    pub store: BTreeMap<u32, u32>,
    pub hin: BTreeMap<u32, u32>,
    pub hout: BTreeMap<u32, u32>,
    pub store_words: u32,
    pub hin_words: u32,
    pub hout_words: u32,
}

/// A grid reading's program.
#[derive(Clone, Debug)]
pub struct GridProgram {
    pub trace: Trace,
    pub reading: String,
    pub word: u64,
    pub clip: [u32; 3],
    /// Roots a side (and the weight grid's side).
    pub p: u32,
    pub vals: BTreeMap<ValueId, Val>,
    pub passes: Vec<Pass>,
    pub regs: Vec<Reg>,
    pub mvs: Vec<Mv>,
    pub gns: Vec<Gn>,
    pub attn: Option<Attention>,
    /// The host's row values (written into the weight PEs) and the reading's outputs.
    pub hostv: Vec<ValueId>,
    pub outs: Vec<ValueId>,
    pub slots: Slots,
    /// Each matvec's tile in a weight PE: (16-bit offset, rows, columns).
    pub tiles: Vec<(u32, u32, u32)>,
    /// The passes' weight groups, where every pass's tiles do not fit a weight PE at once: each
    /// a run of passes `(first, count)` whose tiles the host writes before a launch of them
    /// alone (their offsets the group's own); empty: one launch of every pass.
    pub groups: Vec<(usize, usize)>,
    pub stream: Stream,
    pub palette: Palette,
    pub fabric: Fabric,
    pub round16: bool,
}

impl GridProgram {
    /// `v`'s chunk at a root of its side (a head value's whole width).
    #[must_use]
    pub fn chunk(&self, v: ValueId) -> u32 {
        let x = &self.vals[&v];
        if x.head {
            x.width
        } else {
            x.width.div_ceil(self.p)
        }
    }

    /// `v`'s real columns at root `k` of its side (a padded value's last roots have fewer).
    #[must_use]
    pub fn real(&self, v: ValueId, k: u32) -> u32 {
        let x = &self.vals[&v];
        if x.head {
            return if k == 0 { x.width } else { 0 };
        }
        let c = self.chunk(v);
        x.width.saturating_sub(k * c).min(c)
    }

    /// Matvec `m`'s input piece at root `k` of its `from` side, as indices into its weight's
    /// columns.
    #[must_use]
    pub fn piece(&self, m: usize, k: u32) -> Vec<u32> {
        let mv = &self.mvs[m];
        match mv.input {
            MvIn::Value(x) => {
                let v = &self.vals[&x];
                if v.head {
                    if k == 0 {
                        (0..v.width).collect()
                    } else {
                        Vec::new()
                    }
                } else {
                    let c = self.chunk(x);
                    (k * c..(k + 1) * c).collect()
                }
            }
            MvIn::Gathered { taps, c_in } => {
                let c = c_in / self.p;
                (0..taps)
                    .flat_map(|t| (k * c..(k + 1) * c).map(move |e| t * c_in + e))
                    .collect()
            }
        }
    }

    /// Matvec `m`'s outputs at root `k` of the side it lands on: each output row of its weight,
    /// `None` past its real ones (a padded value).
    #[must_use]
    pub fn slice(&self, m: usize, k: u32) -> Vec<Option<u32>> {
        let mv = &self.mvs[m];
        let v = &self.vals[&mv.output];
        if v.head {
            return if k == 0 {
                (0..mv.n).map(Some).collect()
            } else {
                Vec::new()
            };
        }
        let c = self.chunk(mv.output);
        (k * c..(k + 1) * c)
            .map(|o| (o < mv.n).then_some(o))
            .collect()
    }

    /// Whether a value's row `t` is kept by the weight PE at (`i`, `j`) of the unit (row-side:
    /// column `t % P`; column-side: row `t % P`), and the chunk it keeps (its root's).
    #[must_use]
    pub fn keeper(&self, v: ValueId, i: u32, j: u32) -> (u32, u32) {
        match self.vals[&v].side {
            Side::Row => (i, j),
            Side::Col => (j, i),
        }
    }
}

/// Whether `ty` is a voxel row value.
pub(crate) fn voxel_rows(ty: &Ty) -> bool {
    matches!(ty, Ty::Tensor { shape, .. } if matches!(shape.first(), Some(Dim::Voxels | Dim::VoxelsScaled(..))))
}

pub(crate) fn width(trace: &Trace, v: ValueId) -> u32 {
    match &trace.values[v.0 as usize].ty {
        Ty::Tensor { shape, .. } => match shape.last() {
            Some(Dim::Const(n)) => *n as u32,
            _ => 0,
        },
        Ty::Struct(_) => 0,
    }
}

pub(crate) fn dtype(trace: &Trace, v: ValueId) -> Option<Dtype> {
    match &trace.values[v.0 as usize].ty {
        Ty::Tensor { dtype, .. } => Some(*dtype),
        Ty::Struct(_) => None,
    }
}

/// Whether `trace`'s `reading` is a grid reading: its nodes run no layer, and some of them are
/// of the grid family (a convolution, a group norm).
#[must_use]
pub fn is_grid(trace: &Trace, reading: &str) -> bool {
    let word = trace
        .facts
        .word(&poem_ir::Request::new(0, false).switched("reading", reading));
    let runs = |n: &poem_ir::Node| n.guard.holds(word);
    let Ok(a) = poem_compiler::analysis::analyze(trace) else {
        return false;
    };
    let any_layer = trace
        .nodes
        .iter()
        .enumerate()
        .any(|(at, n)| runs(n) && a.layers[at].is_some());
    !any_layer
        && trace.nodes.iter().any(|n| {
            runs(n)
                && matches!(
                    n.op,
                    Operation::Conv(Conv::Conv3d { .. } | Conv::GroupNorm { .. })
                )
        })
}

/// The clip a grid reading is compiled for at `rows` rows a class: the square whose voxels the
/// trace's patchify of the same voxel input makes `rows` token rows of (a VAE decode's latent
/// at the denoiser's image rows), the square of `rows` voxels where nothing patchifies it.
#[must_use]
pub fn clip_of(trace: &Trace, reading: &str, rows: u32) -> [u32; 3] {
    let word = trace
        .facts
        .word(&poem_ir::Request::new(0, false).switched("reading", reading));
    let mut voxels: BTreeSet<ValueId> = BTreeSet::new();
    let mut ins = Vec::new();
    for n in trace.nodes.iter().filter(|n| n.guard.holds(word)) {
        ins.clear();
        n.op.inputs(&mut ins);
        for v in &ins {
            if matches!(
                trace.values[v.0 as usize].def,
                Def::Input(RuntimeInput::Guest {
                    axis: InputAxis::Voxel,
                    ..
                })
            ) {
                voxels.insert(*v);
            }
        }
    }
    let area = trace
        .nodes
        .iter()
        .find_map(|n| match &n.op {
            Operation::Conv(Conv::Patchify { x, p, .. }) if voxels.contains(x) => Some(p[1] * p[2]),
            _ => None,
        })
        .unwrap_or(1);
    let n = rows * area;
    let s = (f64::from(n)).sqrt().round() as u32;
    [1, s.max(1), s.max(1)]
}

/// The voxels a token row of the trace's latent holds: the area its patchify of a voxel input
/// takes a row of (1 where nothing patchifies one).
#[must_use]
pub fn voxels_a_row(trace: &Trace) -> u32 {
    let voxel = |v: &ValueId| {
        matches!(
            trace.values[v.0 as usize].def,
            Def::Input(RuntimeInput::Guest {
                axis: InputAxis::Voxel,
                ..
            })
        )
    };
    trace
        .nodes
        .iter()
        .find_map(|n| match &n.op {
            Operation::Conv(Conv::Patchify { x, p, .. }) if voxel(x) => Some(p[1] * p[2]),
            _ => None,
        })
        .unwrap_or(1)
        .max(1)
}

/// Whether a node `word` runs reads a voxel input (a latent).
#[must_use]
pub fn reads_voxels(trace: &Trace, word: u64) -> bool {
    let mut ins = Vec::new();
    trace.nodes.iter().filter(|n| n.guard.holds(word)).any(|n| {
        ins.clear();
        n.op.inputs(&mut ins);
        ins.iter().any(|v| {
            matches!(
                trace.values[v.0 as usize].def,
                Def::Input(RuntimeInput::Guest {
                    axis: InputAxis::Voxel,
                    ..
                })
            )
        })
    })
}

/// The program builder's state.
struct B<'t> {
    t: &'t Trace,
    word: u64,
    clip: [u32; 3],
    /// Each node the host runs (its inputs the host's own).
    host: Vec<bool>,
    grids: BTreeMap<ValueId, [u32; 3]>,
    vals: BTreeMap<ValueId, Val>,
    passes: Vec<Pass>,
    /// Where each value is made: (pass, in its head).
    made: BTreeMap<ValueId, (usize, bool)>,
    fetched: BTreeSet<(usize, bool, ValueId)>,
    held: BTreeSet<ValueId>,
    hostv: BTreeSet<ValueId>,
    stats: BTreeMap<ValueId, Vec<usize>>,
    heads: BTreeSet<ValueId>,
    regs: Vec<Reg>,
    mvs: Vec<Mv>,
    gns: Vec<Gn>,
    attn: Option<Attention>,
    next: u32,
}

impl B<'_> {
    fn def(&self, v: ValueId) -> &Def {
        &self.t.values[v.0 as usize].def
    }

    fn pseudo(&mut self, val: Val) -> ValueId {
        let v = ValueId(self.next);
        self.next += 1;
        self.vals.insert(v, val);
        v
    }

    /// A value's grid: the clip's, or the grid its op makes it on.
    fn grid_of(&self, v: ValueId) -> R<[u32; 3]> {
        if let Some(x) = self.vals.get(&v) {
            return Ok(x.grid);
        }
        let g = |gv: &ValueId| {
            self.grids
                .get(gv)
                .copied()
                .ok_or_else(|| format!("value %{}: a grid the compile does not know", gv.0))
        };
        match self.def(v) {
            Def::Input(RuntimeInput::Guest {
                axis: InputAxis::Voxel,
                ..
            }) => Ok(self.clip),
            Def::Op(n) => {
                let op = &self.t.nodes[*n as usize].op;
                match op {
                    Operation::Conv(
                        Conv::Conv3d { y_grid, .. }
                        | Conv::Upsample { y_grid, .. }
                        | Conv::PixelShuffle { y_grid, .. }
                        | Conv::PixelUnshuffle { y_grid, .. },
                    ) => g(y_grid),
                    Operation::Conv(Conv::GroupNorm { grid, .. })
                    | Operation::Attn(Attn::Voxel { grid, .. }) => g(grid),
                    _ => {
                        let mut ins = Vec::new();
                        op.inputs(&mut ins);
                        let x = ins
                            .iter()
                            .find(|i| voxel_rows(&self.t.values[i.0 as usize].ty))
                            .ok_or_else(|| format!("value %{}: on no voxel grid", v.0))?;
                        self.grid_of(*x)
                    }
                }
            }
            _ => Err(format!("value %{}: on no voxel grid", v.0)),
        }
    }

    fn cur(&self) -> usize {
        self.passes.len().saturating_sub(1)
    }

    fn cut(&mut self, grid: [u32; 3]) {
        self.passes.push(Pass {
            grid,
            gather: None,
            head: Vec::new(),
            tail: Vec::new(),
            finals: Vec::new(),
        });
    }

    fn push(&mut self, head: bool, s: Step) {
        let p = self.cur();
        if head {
            self.passes[p].head.push(s);
        } else {
            self.passes[p].tail.push(s);
        }
    }

    /// The host's value `v` (rows the host computes from the step's inputs): kept in the
    /// weight PEs by the host, fetched as a held value is.
    fn host_value(&mut self, v: ValueId) -> R<()> {
        if self.vals.contains_key(&v) {
            return Ok(());
        }
        let grid = self.grid_of(v)?;
        let val = Val {
            width: width(self.t, v),
            grid,
            side: Side::Row,
            head: false,
            round: matches!(dtype(self.t, v), Some(Dtype::Bf16 | Dtype::F16)),
        };
        self.vals.insert(v, val);
        self.hostv.insert(v);
        Ok(())
    }

    /// Whether `v` is the host's (an input, or a host node's output).
    fn is_host(&self, v: ValueId) -> bool {
        match self.def(v) {
            Def::Input(_) => true,
            Def::Op(n) => self.host[*n as usize],
            _ => false,
        }
    }

    /// `v` held from the pass that makes it on.
    fn hold(&mut self, v: ValueId) -> R<()> {
        if self.hostv.contains(&v) || !self.held.insert(v) {
            return Ok(());
        }
        if self.vals[&v].head {
            return Err(format!(
                "value %{}: a value whole at one root held over a pass (an attention's q / k / v)",
                v.0
            ));
        }
        let (p, h) = self.made[&v];
        if h {
            self.passes[p].head.push(Step::Hold(v));
        } else {
            self.passes[p].tail.push(Step::Hold(v));
        }
        Ok(())
    }

    /// `ins` (row values) readable in the current pass's tail: fetched where they were made in
    /// an earlier pass (or are the host's); a pass cut where one is the current pass's head's.
    fn avail(&mut self, ins: &[ValueId]) -> R<()> {
        let mut grid = None;
        for &v in ins {
            if self.is_host(v) {
                self.host_value(v)?;
            }
            grid = Some(self.grid_of(v)?);
        }
        let Some(grid) = grid else { return Ok(()) };
        let cur = self.cur();
        let cut = self.passes.is_empty()
            || self.passes[cur].grid != grid
            || ins.iter().any(|v| self.made.get(v) == Some(&(cur, true)));
        if cut {
            self.cut(grid);
        }
        let cur = self.cur();
        for &v in ins {
            match self.made.get(&v).copied() {
                Some((p, false)) if p == cur => {}
                Some((p, _)) if p < cur => {
                    self.hold(v)?;
                    if self.fetched.insert((cur, false, v)) {
                        self.passes[cur].tail.push(Step::Fetch(v));
                    }
                }
                None if self.hostv.contains(&v) => {
                    if self.fetched.insert((cur, false, v)) {
                        self.passes[cur].tail.push(Step::Fetch(v));
                    }
                }
                other => {
                    return Err(format!(
                        "value %{} is read in pass {cur} but made at {other:?}",
                        v.0
                    ));
                }
            }
        }
        Ok(())
    }

    fn side_of(&self, ins: &[ValueId]) -> R<Side> {
        let mut side = None;
        for v in ins {
            let s = self.vals[v].side;
            if side.is_some_and(|x| x != s) {
                return Err(format!(
                    "value %{}: an op over values on both sides (no transpose)",
                    v.0
                ));
            }
            side = Some(s);
        }
        side.ok_or_else(|| "an op over no row value".into())
    }

    fn make(&mut self, v: ValueId, val: Val) {
        let cur = self.cur();
        self.vals.insert(v, val);
        self.made.insert(v, (cur, false));
    }

    /// A run of elementwise nodes: each part `lower_run` lowers a printed region.
    fn run(&mut self, nodes: &[u32]) -> R<()> {
        if nodes.is_empty() {
            return Ok(());
        }
        for c in split_runs(self.t, nodes)? {
            self.chain(c)?;
        }
        Ok(())
    }

    fn chain(&mut self, c: Chain) -> R<()> {
        let mut ins = Vec::new();
        let mut rows = Vec::new();
        for (k, i) in c.ins.iter().enumerate() {
            if c.region.params[k] == kernels_ir::Shape::Scalar || i.by.is_some() {
                return Err(format!(
                    "node {}: a region over a scalar a row: no grid row carries one",
                    c.nodes[0]
                ));
            }
            if i.shared {
                ins.push(RegIn::Res(Res::Value {
                    v: i.value,
                    from: i.from,
                }));
            } else {
                if i.from != 0 {
                    return Err(format!(
                        "node {}: a region reading a value from column {}",
                        c.nodes[0], i.from
                    ));
                }
                rows.push(i.value);
                ins.push(RegIn::Row(i.value));
            }
        }
        self.avail(&rows)?;
        // values on both sides (a resnet's shortcut convolution, one matvec off its main
        // branch's two): each off the last one's side moved across, an identity matvec
        if self.side_of(&rows).is_err() {
            let to = self.vals[&rows[rows.len() - 1]].side;
            for (k, v) in rows.clone().into_iter().enumerate() {
                if self.vals[&v].side != to {
                    let m = self.moved(v, to)?;
                    rows[k] = m;
                    for i in &mut ins {
                        if *i == RegIn::Row(v) {
                            *i = RegIn::Row(m);
                        }
                    }
                }
            }
        }
        let side = self.side_of(&rows)?;
        let head = rows.iter().any(|v| self.vals[v].head);
        if head && !rows.iter().all(|v| self.vals[v].head) {
            return Err(format!(
                "node {}: a region over a value whole at one root and one cut over the roots",
                c.nodes[0]
            ));
        }
        let grid = self.grid_of(rows[0])?;
        let outs: Vec<ValueId> = c.outs.iter().map(|o| o.value).collect();
        for o in &c.outs {
            let val = Val {
                width: width(self.t, o.value),
                grid,
                side,
                head,
                round: matches!(o.dtype, Dtype::Bf16 | Dtype::F16),
            };
            self.make(o.value, val);
        }
        let r = self.regs.len();
        self.regs.push(Reg {
            side,
            head,
            region: c.region,
            ins,
            outs,
            printed: None,
            params: Vec::new(),
        });
        self.push(false, Step::Region(r));
        Ok(())
    }

    /// `v` (made in the current pass's tail) on the other side `to`: an identity matvec.
    fn moved(&mut self, v: ValueId, to: Side) -> R<ValueId> {
        let x = self.vals[&v].clone();
        if x.head {
            return Err(format!(
                "value %{}: a move of a value whole at one root",
                v.0
            ));
        }
        let m = self.pseudo(Val {
            side: to,
            ..x.clone()
        });
        let cur = self.cur();
        self.made.insert(m, (cur, false));
        let i = self.mvs.len();
        self.mvs.push(Mv {
            node: u32::MAX,
            weight: v,
            ident: true,
            k: x.width,
            n: x.width,
            from: x.side,
            input: MvIn::Value(v),
            output: m,
        });
        self.push(false, Step::Matvec(i));
        Ok(m)
    }

    /// A region of the program's own (a bias, a group norm's apply): `region` over `ins`.
    fn own_region(&mut self, region: kernels_ir::Region, ins: Vec<RegIn>, out: ValueId) {
        let side = self.vals[&out].side;
        let head = self.vals[&out].head;
        let r = self.regs.len();
        self.regs.push(Reg {
            side,
            head,
            region,
            ins,
            outs: vec![out],
            printed: None,
            params: Vec::new(),
        });
        self.push(false, Step::Region(r));
    }

    /// A gather of `src`'s rows into `out`'s: the current pass's head made of what makes `src`
    /// where it is that pass's (and it gathers nothing yet), else a new pass fetching `src`.
    fn gather(&mut self, src: ValueId, out: ValueId, kind: Kind, in_grid: [u32; 3]) -> R<()> {
        if self.is_host(src) {
            self.host_value(src)?;
        }
        let src_grid = self.grid_of(src)?;
        let out_grid = self.vals[&out].grid;
        let side = self.vals[&src].side;
        let cur = self.cur();
        let convert = !self.passes.is_empty()
            && self.passes[cur].gather.is_none()
            && self.made.get(&src) == Some(&(cur, false));
        if convert {
            let tail = std::mem::take(&mut self.passes[cur].tail);
            self.passes[cur].head = tail;
            for m in self.made.values_mut() {
                if *m == (cur, false) {
                    *m = (cur, true);
                }
            }
            self.fetched = std::mem::take(&mut self.fetched)
                .into_iter()
                .map(|(p, h, v)| if p == cur { (p, true, v) } else { (p, h, v) })
                .collect();
        } else {
            self.cut(src_grid);
            let cur = self.cur();
            if !self.hostv.contains(&src) {
                match self.made.get(&src) {
                    Some((p, _)) if *p < cur => self.hold(src)?,
                    other => {
                        return Err(format!(
                            "value %{}: gathered in pass {cur} but made at {other:?}",
                            src.0
                        ));
                    }
                }
            }
            self.fetched.insert((cur, true, src));
            self.passes[cur].head.push(Step::Fetch(src));
        }
        let cur = self.cur();
        let mut g = Gather {
            src,
            out,
            side,
            kind,
            src_grid,
            in_grid,
            out_grid,
            ring: 1,
        };
        g.ring = ring_rows(&g);
        self.passes[cur].head.push(Step::Ring(src));
        self.passes[cur].gather = Some(g);
        self.passes[cur].grid = out_grid;
        self.passes[cur].tail.push(Step::Gather);
        self.made.insert(out, (cur, false));
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    fn node(&mut self, at: u32) -> R<()> {
        let t = self.t;
        let node = &t.nodes[at as usize];
        let refuse = |why: String| format!("{}: node {at}: {why}", t.name);
        let round = |v: ValueId| matches!(dtype(t, v), Some(Dtype::Bf16 | Dtype::F16));
        match &node.op {
            Operation::Conv(Conv::Grid { .. }) => Ok(()),
            Operation::Conv(Conv::Conv3d {
                x,
                w,
                bias,
                k,
                stride,
                pad,
                causal_t,
                dilation,
                depthwise,
                cache,
                y,
                ..
            }) => {
                if k[0] != 1 || stride[0] != 1 || pad[0] != 0 {
                    return Err(refuse(format!(
                        "conv.conv3d of {k:?} taps, stride {stride:?}: a kernel across frames"
                    )));
                }
                if *causal_t || *depthwise || cache.is_some() || *dilation != [1, 1, 1] {
                    return Err(refuse(
                        "conv.conv3d causal, depthwise, dilated or cached: a plain convolution's \
                         taps alone"
                            .into(),
                    ));
                }
                if self.is_host(*x) {
                    self.host_value(*x)?;
                }
                let (c_in, c_out) = (width(t, *x), width(t, *y));
                let taps = k[1] * k[2];
                let grid = self.grid_of(*y)?;
                let xs = self.vals.get(x).map(|v| v.side);
                let pointwise = taps == 1 && *stride == [1, 1, 1] && *pad == [0, 0, 0];
                let (input, from) = if pointwise {
                    self.avail(&[*x])?;
                    (MvIn::Value(*x), self.vals[x].side)
                } else {
                    let from = xs.ok_or_else(|| refuse("a convolution of no row".into()))?;
                    let raw_in = Val {
                        width: taps * c_in,
                        grid,
                        side: from,
                        head: false,
                        round: false,
                    };
                    // an upsampling just gathered (and read by nothing else) folds into the taps
                    let cur = self.cur();
                    let fold = self.passes.get(cur).and_then(|p| match &p.gather {
                        Some(g)
                            if g.out == *x
                                && matches!(g.kind, Kind::Up(_))
                                && p.tail == [Step::Gather]
                                && readers(t, self.word, *x) == 1 =>
                        {
                            Some(g.clone())
                        }
                        _ => None,
                    });
                    let in_grid = self.grid_of(*x)?;
                    let kind = |up: u32| Kind::Conv {
                        k: [k[1], k[2]],
                        s: [stride[1], stride[2]],
                        p: [pad[1], pad[2]],
                        up,
                    };
                    let gv = self.pseudo(raw_in);
                    if let Some(g) = fold {
                        let Kind::Up(f) = g.kind else { unreachable!() };
                        let mut ng = Gather {
                            out: gv,
                            kind: kind(f),
                            in_grid,
                            out_grid: grid,
                            ..g
                        };
                        ng.ring = ring_rows(&ng);
                        self.passes[cur].gather = Some(ng);
                        self.passes[cur].grid = grid;
                        self.made.insert(gv, (cur, false));
                    } else {
                        self.gather(*x, gv, kind(1), in_grid)?;
                    }
                    (MvIn::Gathered { taps, c_in }, from)
                };
                let side = from.flip();
                let raw = self.pseudo(Val {
                    width: c_out,
                    grid,
                    side,
                    head: false,
                    round: bias.is_none() && round(*y),
                });
                let cur = self.cur();
                self.made.insert(raw, (cur, false));
                let m = self.mvs.len();
                self.mvs.push(Mv {
                    node: at,
                    weight: *w,
                    ident: false,
                    k: taps * c_in,
                    n: c_out,
                    from,
                    input,
                    output: raw,
                });
                self.push(false, Step::Matvec(m));
                let val = Val {
                    width: c_out,
                    grid,
                    side,
                    head: false,
                    round: round(*y),
                };
                match bias {
                    Some(b) => {
                        self.make(*y, val.clone());
                        let reg = add_region(c_out, val.round);
                        self.own_region(
                            reg,
                            vec![RegIn::Row(raw), RegIn::Res(Res::Value { v: *b, from: 0 })],
                            *y,
                        );
                    }
                    None => {
                        // the matvec's output is the convolution's
                        self.vals.remove(&raw);
                        self.made.remove(&raw);
                        self.mvs[m].output = *y;
                        self.make(*y, val);
                    }
                }
                Ok(())
            }
            Operation::Conv(Conv::GroupNorm {
                x,
                groups,
                weight,
                bias,
                eps,
                silu,
                y,
                ..
            }) => {
                if self.is_host(*x) {
                    return Err(refuse(
                        "conv.group_norm over the host's rows: its statistics are summed where \
                         a pass makes its input"
                            .into(),
                    ));
                }
                // the norm reads its input's every row: a pass after the one that makes it
                let grid = self.grid_of(*x)?;
                if self.made.get(x).is_some_and(|m| m.0 == self.cur()) {
                    self.cut(grid);
                }
                self.avail(&[*x])?;
                let g = self.gns.len();
                let (p, h) = self.made[x];
                if !self.stats.contains_key(x) {
                    if h {
                        self.passes[p].head.push(Step::Stats(*x));
                    } else {
                        self.passes[p].tail.push(Step::Stats(*x));
                    }
                }
                self.stats.entry(*x).or_default().push(g);
                self.passes[p].finals.push(g);
                let side = self.vals[x].side;
                let val = Val {
                    width: width(t, *y),
                    grid,
                    side,
                    head: false,
                    round: round(*y),
                };
                self.make(*y, val.clone());
                let reg = self.regs.len();
                self.gns.push(Gn {
                    x: *x,
                    y: *y,
                    groups: *groups,
                    eps: *eps,
                    silu: *silu,
                    gamma: *weight,
                    beta: *bias,
                    reg,
                });
                self.own_region(
                    norm_region(val.width, *silu, val.round),
                    vec![
                        RegIn::Row(*x),
                        RegIn::Res(Res::GnScale(g)),
                        RegIn::Res(Res::GnShift(g)),
                    ],
                    *y,
                );
                Ok(())
            }
            Operation::Conv(
                op @ (Conv::Upsample { x, y, .. }
                | Conv::PixelShuffle { x, y, .. }
                | Conv::PixelUnshuffle { x, y, .. }),
            ) => {
                let kind = match op {
                    Conv::Upsample { factor, .. } if factor[0] == 1 && factor[1] == factor[2] => {
                        Kind::Up(factor[1])
                    }
                    Conv::PixelShuffle { r, trim_t: 0, .. } if r[0] == 1 && r[1] == r[2] => {
                        Kind::Shuffle(r[1])
                    }
                    Conv::PixelUnshuffle { r, .. } if r[0] == 1 && r[1] == r[2] => {
                        Kind::Unshuffle(r[1])
                    }
                    _ => {
                        return Err(refuse(format!(
                            "{}: a resampling across frames or of unequal axes",
                            node.op.name()
                        )));
                    }
                };
                let f = match kind {
                    Kind::Up(f) | Kind::Shuffle(f) | Kind::Unshuffle(f) => f,
                    Kind::Conv { .. } => 1,
                };
                if !f.is_power_of_two() {
                    return Err(refuse(format!(
                        "{} by {f}: a factor of two's powers alone (a shift)",
                        node.op.name()
                    )));
                }
                if self.is_host(*x) {
                    self.host_value(*x)?;
                }
                let side = self
                    .vals
                    .get(x)
                    .map(|v| v.side)
                    .ok_or_else(|| refuse("a resampling of no row".into()))?;
                let grid = self.grid_of(*y)?;
                self.vals.insert(
                    *y,
                    Val {
                        width: width(t, *y),
                        grid,
                        side,
                        head: false,
                        round: round(*y),
                    },
                );
                let in_grid = self.grid_of(*x)?;
                self.gather(*x, *y, kind, in_grid)
            }
            Operation::Linear(Linear::Matmul {
                act,
                w,
                input_scales: None,
                scales: None,
                quant: None,
                logits: false,
                y,
            }) => {
                self.avail(&[*act])?;
                let from = self.vals[act].side;
                let grid = self.grid_of(*act)?;
                let val = Val {
                    width: width(t, *y),
                    grid,
                    side: from.flip(),
                    head: self.heads.contains(y),
                    round: round(*y),
                };
                self.make(*y, val);
                let m = self.mvs.len();
                self.mvs.push(Mv {
                    node: at,
                    weight: *w,
                    ident: false,
                    k: width(t, *act),
                    n: width(t, *y),
                    from,
                    input: MvIn::Value(*act),
                    output: *y,
                });
                self.push(false, Step::Matvec(m));
                Ok(())
            }
            Operation::Attn(Attn::Voxel {
                q,
                k,
                v,
                segment,
                sm_scale,
                o,
                ..
            }) => {
                if *segment != poem_ir::VoxelSegment::Clip {
                    return Err(refuse(
                        "attn.voxel over blocks of frames: a clip's every row alone".into(),
                    ));
                }
                if self.attn.is_some() {
                    return Err(refuse(
                        "a second attn.voxel: one K / V column a unit".into(),
                    ));
                }
                self.avail(&[*q, *k, *v])?;
                for x in [q, k, v] {
                    if self.vals[x].side != Side::Col || !self.vals[x].head {
                        return Err(refuse(
                            "attn.voxel's q / k / v not whole at the first column root: the \
                             K / V column hangs below it"
                                .into(),
                        ));
                    }
                }
                let grid = self.grid_of(*q)?;
                let w = width(t, *q);
                self.push(
                    false,
                    Step::Store {
                        q: *q,
                        k: *k,
                        v: *v,
                    },
                );
                // the queries read every row's keys: the next pass
                self.cut(grid);
                self.make(
                    *o,
                    Val {
                        width: w,
                        grid,
                        side: Side::Col,
                        head: true,
                        round: round(*o),
                    },
                );
                self.push(false, Step::Attend(*o));
                let rows = grid[0] * grid[1] * grid[2];
                self.attn = Some(Attention {
                    rows,
                    width: w,
                    scale: *sm_scale,
                    pya: crate::compile::stream::build::kv_pya(rows, 1, w),
                });
                Ok(())
            }
            op => Err(refuse(format!("{} has no grid lowering", op.name()))),
        }
    }
}

/// The readers of `v` among the nodes `word` runs.
pub(crate) fn readers(t: &Trace, word: u64, v: ValueId) -> usize {
    let mut ins = Vec::new();
    t.nodes
        .iter()
        .filter(|n| n.guard.holds(word))
        .filter(|n| {
            ins.clear();
            n.op.inputs(&mut ins);
            ins.contains(&v)
        })
        .count()
}

/// `run` as regions, split where it does not lower as one (as a stream unit's).
pub(crate) fn split_runs(t: &Trace, run: &[u32]) -> R<Vec<Chain>> {
    if let Some(c) = chain::lower_run(t, run) {
        return Ok(vec![c]);
    }
    if run.len() == 1 {
        return Err(format!(
            "node {}: {} lowers to no root region",
            run[0],
            t.nodes[run[0] as usize].op.name()
        ));
    }
    let (a, b) = run.split_at(run.len() / 2);
    let mut out = split_runs(t, a)?;
    out.extend(split_runs(t, b)?);
    Ok(out)
}

/// `x + b` over a row of `width` (a convolution's bias), rounded where `round`.
pub(crate) fn add_region(width: u32, round: bool) -> kernels_ir::Region {
    use kernels_ir::{Binary, Op, Region, Shape};
    Region {
        width,
        params: vec![Shape::Row, Shape::Row],
        ops: vec![Op::Binary(Binary::Add, 0, 1)],
        results: vec![2],
        rounds: if round {
            vec![(2, Dtype::Bf16)]
        } else {
            Vec::new()
        },
    }
}

/// A group norm's apply over a row of `width`: `x a + b` (the scale and shift its statistics
/// made a channel), then SiLU where `silu`, rounded where `round`.
pub(crate) fn norm_region(width: u32, silu: bool, round: bool) -> kernels_ir::Region {
    use kernels_ir::{Binary, Literal, Op, Region, Shape, Unary};
    let mut ops = vec![Op::Fma(0, 1, 2)];
    let last = if silu {
        ops.extend([
            Op::Unary(Unary::Neg, 3),
            Op::Unary(Unary::Exp, 4),
            Op::Const(Literal::F32(1.0)),
            Op::Binary(Binary::Add, 6, 5),
            Op::Unary(Unary::Recip, 7),
            Op::Binary(Binary::Mul, 3, 8),
        ]);
        9
    } else {
        3
    };
    Region {
        width,
        params: vec![Shape::Row, Shape::Row, Shape::Row],
        ops,
        results: vec![last],
        rounds: if round {
            vec![(last, Dtype::Bf16)]
        } else {
            Vec::new()
        },
    }
}

/// The line buffer's rows a gather needs: the most source rows from a row's first tap to the
/// last pulled for it, a power of two.
fn ring_rows(g: &Gather) -> u32 {
    let [_, h, w] = g.out_grid;
    let mut pulled = 0u32;
    let mut most = 1u32;
    for y in 0..h {
        for x in 0..w {
            pulled = pulled.max(g.need(y, x));
            if let Some(lo) = g.sources(y, x).iter().map(|s| s.1).min() {
                most = most.max(pulled - lo);
            }
        }
    }
    most.next_power_of_two()
}

/// The roots a side a grid program may take at most.
pub const MAX_P: u32 = 16;
/// 16-bit words a weight PE gives its tiles and stores (of 24,576).
pub const WEIGHT_WORDS: u32 = 15_360;

/// `trace`'s grid `reading` on the clip `clip` (the step's one): its passes, placement and
/// fabric. A node no grid step lowers is refused by name.
#[allow(clippy::too_many_lines)]
pub fn program(trace: &Trace, reading: &str, clip: [u32; 3], round16: bool) -> R<GridProgram> {
    program_upto(trace, reading, clip, round16, None)
}

/// [`program`] over its first `upto` passes only (a part of a reading on the simulator): the
/// value the last pass's tail makes last is read out in place of the reading's outputs.
#[allow(clippy::too_many_lines)]
pub fn program_upto(
    trace: &Trace,
    reading: &str,
    clip: [u32; 3],
    round16: bool,
    upto: Option<usize>,
) -> R<GridProgram> {
    build(trace, reading, clip, round16, upto, true, WEIGHT_WORDS)
}

/// What a grid reading's lowering starts from, in either regime ([`build`]'s passes, or the
/// tiled regime's ops): its one class's word, the nodes it runs, those the host runs (their
/// inputs all the host's), each grid value's extent on `clip`.
pub(crate) struct Prelude {
    pub word: u64,
    pub runs: Vec<bool>,
    pub host: Vec<bool>,
    pub grids: BTreeMap<ValueId, [u32; 3]>,
}

pub(crate) fn prelude(trace: &Trace, reading: &str, clip: [u32; 3]) -> R<Prelude> {
    let name = &trace.name;
    let classes = poem_compiler::spatial::layer::stream::classes(trace, reading, 0);
    let [class] = &classes[..] else {
        return Err(format!(
            "{name}: grid reading `{reading}` of {} row classes: one clip a step",
            classes.len()
        ));
    };
    let word = class.word;
    let runs: Vec<bool> = trace.nodes.iter().map(|n| n.guard.holds(word)).collect();
    if !runs.iter().any(|r| *r) {
        return Err(format!("{name}: grid reading `{reading}` runs no node"));
    }
    // the host's nodes: every input the host's own and no weight read (a permutation of the
    // latents, a grid's rule)
    let mut host = vec![false; trace.nodes.len()];
    let mut ins = Vec::new();
    for (at, n) in trace.nodes.iter().enumerate() {
        if !runs[at] {
            continue;
        }
        ins.clear();
        n.op.inputs(&mut ins);
        let own = |v: &ValueId| match &trace.values[v.0 as usize].def {
            Def::Input(_) => true,
            Def::Op(m) => host[*m as usize],
            _ => false,
        };
        let matmul = matches!(n.op, Operation::Linear(_));
        host[at] = !matmul && ins.iter().all(own);
    }
    // the grids, from the clip
    let mut grids: BTreeMap<ValueId, [u32; 3]> = BTreeMap::new();
    for (i, d) in trace.values.iter().enumerate() {
        if matches!(d.def, Def::Input(RuntimeInput::Grid)) {
            grids.insert(ValueId(i as u32), clip);
        }
    }
    for (at, n) in trace.nodes.iter().enumerate() {
        if let (true, Operation::Conv(Conv::Grid { grid, rule, y })) = (runs[at], &n.op)
            && let Some(g) = grids.get(grid).copied()
        {
            let out = rule
                .out_extent(g)
                .ok_or_else(|| format!("{name}: node {at}: a grid {g:?} its rule does not map"))?;
            grids.insert(*y, out);
        }
    }
    Ok(Prelude {
        word,
        runs,
        host,
        grids,
    })
}

#[allow(clippy::too_many_lines)]
/// `budget`: the 16-bit words a weight PE gives its tiles and stores ([`WEIGHT_WORDS`]; less
/// to put a reading that fits in weight groups all the same).
fn build(
    trace: &Trace,
    reading: &str,
    clip: [u32; 3],
    round16: bool,
    upto: Option<usize>,
    fit: bool,
    budget: u32,
) -> R<GridProgram> {
    let name = &trace.name;
    let Prelude {
        word,
        runs,
        host,
        grids,
    } = prelude(trace, reading, clip)?;
    // the values whole at the first column root: an attention's q / k / v (and what makes
    // them after the matvec) and its output
    let mut heads: BTreeSet<ValueId> = BTreeSet::new();
    for (at, n) in trace.nodes.iter().enumerate() {
        if let (true, Operation::Attn(Attn::Voxel { q, k, v, o, .. })) = (runs[at], &n.op) {
            heads.extend([*q, *k, *v, *o]);
        }
    }
    loop {
        let before = heads.len();
        for (at, n) in trace.nodes.iter().enumerate() {
            if !runs[at] || !matches!(n.op, Operation::Elemwise(_)) {
                continue;
            }
            let (mut i, mut o) = (Vec::new(), Vec::new());
            n.op.inputs(&mut i);
            n.op.outputs(&mut o);
            let rows: Vec<ValueId> = i
                .into_iter()
                .filter(|v| voxel_rows(&trace.values[v.0 as usize].ty))
                .collect();
            if o.iter().any(|v| heads.contains(v)) {
                heads.extend(rows.iter().copied());
            }
        }
        if heads.len() == before {
            break;
        }
    }
    let mut b = B {
        t: trace,
        word,
        clip,
        host,
        grids,
        vals: BTreeMap::new(),
        passes: Vec::new(),
        made: BTreeMap::new(),
        fetched: BTreeSet::new(),
        held: BTreeSet::new(),
        hostv: BTreeSet::new(),
        stats: BTreeMap::new(),
        heads,
        regs: Vec::new(),
        mvs: Vec::new(),
        gns: Vec::new(),
        attn: None,
        next: trace.values.len() as u32,
    };
    let mut pending: Vec<u32> = Vec::new();
    for (at, n) in trace.nodes.iter().enumerate() {
        if !runs[at] || b.host[at] {
            continue;
        }
        if matches!(n.op, Operation::Elemwise(_)) {
            pending.push(at as u32);
            continue;
        }
        let run = std::mem::take(&mut pending);
        b.run(&run).map_err(|e| format!("{name}: {e}"))?;
        b.node(at as u32)?;
    }
    let run = std::mem::take(&mut pending);
    b.run(&run).map_err(|e| format!("{name}: {e}"))?;
    // a value the reading reads out leaves from the weight PEs
    let mut outs = Vec::new();
    if let Some(n) = upto.filter(|&n| n < b.passes.len()) {
        b.passes.truncate(n.max(1));
        if !b
            .passes
            .iter()
            .any(|p| p.tail.iter().any(|s| matches!(s, Step::Attend(_))))
        {
            // the K / V column's stores and queries cut too
            b.attn = None;
            for p in &mut b.passes {
                p.tail.retain(|s| !matches!(s, Step::Store { .. }));
            }
        }
        let last = b.passes.len() - 1;
        let made = b.passes[last].tail.iter().rev().find_map(|s| match s {
            Step::Region(r) => b.regs[*r].outs.first().copied(),
            Step::Matvec(m) => Some(b.mvs[*m].output),
            Step::Gather => b.passes[last].gather.as_ref().map(|g| g.out),
            Step::Attend(o) => Some(*o),
            _ => None,
        });
        let v = made.ok_or_else(|| format!("{name}: pass {last} makes no row"))?;
        if b.vals[&v].head {
            return Err(format!(
                "{name}: pass {last}'s last value is whole at one root"
            ));
        }
        outs.push(v);
        b.passes[last].tail.push(Step::Out(v));
    }
    let reads_out = outs.is_empty();
    for o in trace.outputs.iter().filter(|_| reads_out) {
        let v = o.value;
        if let Some(&(p, h)) = b.made.get(&v) {
            if outs.contains(&v) {
                continue;
            }
            if b.vals[&v].head {
                return Err(format!(
                    "{name}: output `{}` whole at one root: read out from the even cut",
                    o.name
                ));
            }
            outs.push(v);
            if h {
                b.passes[p].head.push(Step::Out(v));
            } else {
                b.passes[p].tail.push(Step::Out(v));
            }
        }
    }
    if outs.is_empty() {
        return Err(format!(
            "{name}: grid reading `{reading}` reads nothing out"
        ));
    }
    if b.passes.is_empty() {
        return Err(format!("{name}: grid reading `{reading}` runs no pass"));
    }
    // the roots a side: the most a group norm's groups and a shuffle's channels cut over evenly
    // (every other value padded to a whole chunk a root: its padding columns read zero weights)
    let fits = |p: u32| -> Option<String> {
        for gn in &b.gns {
            let w = b.vals[&gn.x].width;
            if !w.is_multiple_of(p) || !(w / p).is_multiple_of(w / gn.groups.max(1)) {
                return Some(format!("a group norm of {} groups over {w}", gn.groups));
            }
        }
        for pass in &b.passes {
            if let Some(gt) = &pass.gather
                && matches!(gt.kind, Kind::Shuffle(_) | Kind::Unshuffle(_))
            {
                for v in [gt.src, gt.out] {
                    if !b.vals[&v].width.is_multiple_of(p) {
                        return Some(format!(
                            "a shuffle of %{} over {} columns",
                            v.0, b.vals[&v].width
                        ));
                    }
                }
            }
        }
        None
    };
    let mut p = MAX_P;
    while p > 1 && fits(p).is_some() {
        p /= 2;
    }
    if let Some(why) = fits(p) {
        return Err(format!("{name}: no roots cut the reading's values: {why}"));
    }
    // (priced unplaced: the unit's fabric without its K / V column)
    let (stream, palette, fabric) = floorplan(p, b.attn.as_ref().filter(|_| fit), reading)?;
    let mut prog = GridProgram {
        trace: trace.clone(),
        reading: reading.to_string(),
        word,
        clip,
        p,
        vals: b.vals,
        passes: b.passes,
        regs: b.regs,
        mvs: b.mvs,
        gns: b.gns,
        attn: b.attn,
        hostv: b.hostv.into_iter().collect(),
        outs,
        slots: Slots {
            store: BTreeMap::new(),
            hin: BTreeMap::new(),
            hout: BTreeMap::new(),
            store_words: 0,
            hin_words: 0,
            hout_words: 0,
        },
        tiles: Vec::new(),
        groups: Vec::new(),
        stream,
        palette,
        fabric,
        round16,
    };
    print_regions(&mut prog)?;
    place(&mut prog, fit, budget)?;
    Ok(prog)
}

/// Every region printed at its chunk.
fn print_regions(g: &mut GridProgram) -> R<()> {
    let p = g.p;
    for ri in 0..g.regs.len() {
        let r = &g.regs[ri];
        let out = r.outs[0];
        let chunk = g.chunk(out);
        let roots = if r.head { 1 } else { p };
        let params: Vec<ParamAt> = r
            .ins
            .iter()
            .map(|i| match i {
                RegIn::Row(_) => ParamAt::Row,
                RegIn::Res(_) => ParamAt::Resident,
            })
            .collect();
        // a padded value's region over its padding columns too (its residents' zeros there)
        let mut reg = r.region.clone();
        reg.width = reg.width.max(chunk * roots);
        let printed = region::print(
            &reg,
            &Place {
                chunk,
                roots,
                params: params.clone(),
                round16: g.round16,
                rope_lv: 1,
                rope_bits: 3,
                straddle: false,
                columns: None,
                dup: 1,
            },
        )
        .map_err(|e| format!("{}: region {ri}: {e}", g.trace.name))?;
        if !printed.tables.is_empty() || printed.all_reduces + printed.head_reduces > 0 {
            return Err(format!(
                "{}: region {ri}: a region with tables or a row's reduction on a grid row",
                g.trace.name
            ));
        }
        g.regs[ri].printed = Some(printed);
        g.regs[ri].params = params;
    }
    Ok(())
}

/// The tiles' and stores' places in a weight PE.
fn place(g: &mut GridProgram, fit: bool, budget: u32) -> R<()> {
    let p = g.p;
    let mut off = 1u32;
    let mut tiles = Vec::new();
    for m in 0..g.mvs.len() {
        let kc = (0..p)
            .map(|k| g.piece(m, k).len() as u32)
            .max()
            .unwrap_or(0);
        let nc = (0..p)
            .map(|k| g.slice(m, k).len() as u32)
            .max()
            .unwrap_or(0);
        tiles.push((off, kc, nc));
        off += kc * nc;
    }
    g.tiles = tiles;
    // each held value's lifetime: the pass that makes it to the last that fetches it
    let mut life: BTreeMap<ValueId, (usize, usize)> = BTreeMap::new();
    for (pi, pass) in g.passes.iter().enumerate() {
        for s in pass.head.iter().chain(&pass.tail) {
            match s {
                Step::Hold(v) | Step::Fetch(v) => {
                    let e = life.entry(*v).or_insert((pi, pi));
                    e.0 = e.0.min(pi);
                    e.1 = e.1.max(pi);
                }
                _ => {}
            }
        }
    }
    let words = |g: &GridProgram, v: ValueId| g.vals[&v].rows().div_ceil(p) * g.chunk(v);
    let mut placed: Vec<(u32, u32, usize, usize)> = Vec::new();
    let mut map = BTreeMap::new();
    let mut top = 0u32;
    for (&v, &(a, b)) in &life {
        if g.hostv.contains(&v) {
            continue;
        }
        let n = words(g, v);
        let mut at = 0u32;
        loop {
            let clash = placed
                .iter()
                .find(|(o, w, pa, pb)| *pa <= b && a <= *pb && at < o + w && *o < at + n);
            match clash {
                Some((o, w, _, _)) => at = o + w,
                None => break,
            }
        }
        placed.push((at, n, a, b));
        map.insert(v.0, at);
        top = top.max(at + n);
    }
    g.slots.store = map;
    g.slots.store_words = top;
    let mut at = 0;
    for &v in &g.hostv {
        if g.vals[&v].side != Side::Row || g.vals[&v].head {
            return Err(format!("host value %{} not on the row side", v.0));
        }
        g.slots.hin.insert(v.0, at);
        at += words(g, v);
    }
    g.slots.hin_words = at;
    let mut at = 0;
    for &v in &g.outs.clone() {
        g.slots.hout.insert(v.0, at);
        at += words(g, v);
    }
    g.slots.hout_words = at;
    let mut used = off + g.slots.store_words + 2 * (g.slots.hin_words + g.slots.hout_words);
    if used > budget {
        // every pass's tiles at once do not fit: the passes in weight groups, each run of
        // passes a launch whose tiles the host writes before it (the stores stay)
        let room = budget
            .saturating_sub(g.slots.store_words + 2 * (g.slots.hin_words + g.slots.hout_words));
        if let Some((groups, top)) = weight_groups(g, room) {
            g.groups = groups;
            used = top + g.slots.store_words + 2 * (g.slots.hin_words + g.slots.hout_words);
        }
    }
    if fit && used > budget {
        return Err(format!(
            "{}: a weight PE's tiles and stores are {used} 16-bit words (at most {budget}): \
             {} of tiles, {} held, {} + {} the host's",
            g.trace.name, off, g.slots.store_words, g.slots.hin_words, g.slots.hout_words
        ));
    }
    Ok(())
}

/// The passes cut into runs whose tiles each fit `room` words of a weight PE (an attention's
/// stores and queries in one run), every matvec's tile placed from its run's first word on:
/// the runs (`(first, count)`) and the most words a run's tiles take; `None` where a pass's
/// own tiles do not fit.
fn weight_groups(g: &mut GridProgram, room: u32) -> Option<(Vec<(usize, usize)>, u32)> {
    let words = |g: &GridProgram, pi: usize| -> u32 {
        let pass = &g.passes[pi];
        pass.head
            .iter()
            .chain(&pass.tail)
            .filter_map(|s| match s {
                Step::Matvec(m) => Some(g.tiles[*m].1 * g.tiles[*m].2),
                _ => None,
            })
            .sum()
    };
    let attends = |g: &GridProgram, pi: usize| {
        g.passes[pi]
            .tail
            .iter()
            .chain(&g.passes[pi].head)
            .any(|s| matches!(s, Step::Attend(_)))
    };
    let mut groups: Vec<(usize, usize)> = Vec::new();
    let (mut first, mut held) = (0usize, 1u32);
    for pi in 0..g.passes.len() {
        let w = words(g, pi);
        if 1 + w > room {
            return None;
        }
        // a pass that attends stays with the pass before it (its stores)
        if held + w > room && pi > first && !attends(g, pi) {
            groups.push((first, pi - first));
            (first, held) = (pi, 1);
        }
        held += w;
    }
    groups.push((first, g.passes.len() - first));
    if groups.iter().any(|&(f, n)| {
        let mut h = 1;
        for pi in f..f + n {
            h += words(g, pi);
        }
        h > room
    }) {
        return None;
    }
    // each run's tiles from its first word, in matvec order
    let mut top = 1;
    for &(f, n) in &groups {
        let mut off = 1;
        for m in 0..g.mvs.len() {
            let in_run = (f..f + n).any(|pi| {
                g.passes[pi]
                    .head
                    .iter()
                    .chain(&g.passes[pi].tail)
                    .any(|s| *s == Step::Matvec(m))
            });
            if in_run {
                let (_, kc, nc) = g.tiles[m];
                g.tiles[m].0 = off;
                off += kc * nc;
            }
        }
        top = top.max(off);
    }
    Some((groups, top))
}

/// The unit (`p` roots a side) and the K / V column of its attention on the fabric.
fn floorplan(p: u32, attn: Option<&Attention>, reading: &str) -> R<(Stream, Palette, Fabric)> {
    let requests = Requests {
        stream: true,
        joint: attn.is_some(),
        ..Requests::default()
    };
    let asks = [UnitAsk {
        name: format!("{reading}.grid"),
        heads: u32::from(attn.is_some()),
        cols: p,
        segment: false,
        col_segment: None,
        on_path: true,
        lanes: (false, false),
        helpers: 0,
    }];
    let blocks: Vec<Vec<usize>> = if attn.is_some() {
        vec![vec![0]]
    } else {
        Vec::new()
    };
    let pya = attn.map_or(crate::compile::stream::build::PYA, |a| a.pya);
    let stream = Stream::of(&requests, p, &asks, &blocks, 1, pya, 1, Bounds::system())?;
    let wiring = stream.wiring();
    let floor = stream.floor();
    let palette = Palette::allocate(Vec::new(), &floor, &wiring)?;
    let fabric = Fabric::lay(&floor, &wiring, &palette)?;
    fabric.validate(&floor)?;
    Ok((stream, palette, fabric))
}

/// Compiles `trace`'s grid `reading` on `clip` into `out`: the program's CSL (`layout.csl`, the
/// kernels, the printed regions), `fabric.txt` and the host's files (`host.json`, the static
/// arrays' spans over the checkpoint's values, the host rows' gathers, the program's own words).
pub fn compile(
    trace: &Trace,
    reading: &str,
    clip: [u32; 3],
    round16: bool,
    out: &std::path::Path,
) -> R<GridProgram> {
    compile_upto(trace, reading, clip, round16, None, out)
}

/// [`compile`] of [`program_upto`].
pub fn compile_upto(
    trace: &Trace,
    reading: &str,
    clip: [u32; 3],
    round16: bool,
    upto: Option<usize>,
    out: &std::path::Path,
) -> R<GridProgram> {
    compile_within(trace, reading, clip, round16, upto, WEIGHT_WORDS, out)
}

/// [`compile_upto`] at `budget` words a weight PE for tiles and stores (below
/// [`WEIGHT_WORDS`]: a reading that fits in weight groups all the same, as a larger one is).
pub fn compile_within(
    trace: &Trace,
    reading: &str,
    clip: [u32; 3],
    round16: bool,
    upto: Option<usize>,
    budget: u32,
    out: &std::path::Path,
) -> R<GridProgram> {
    let g = build(trace, reading, clip, round16, upto, true, budget)?;
    let e = crate::compile::emit::grid::emit(&g)?;
    std::fs::create_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))?;
    for (name, text) in &e.files {
        std::fs::write(out.join(name), text).map_err(|e| format!("{name}: {e}"))?;
    }
    crate::compile::check::written(&e.placed, &e.files, None);
    std::fs::write(
        out.join("fabric.txt"),
        format!("{} {} {}\n", e.w, e.h, u8::from(round16)),
    )
    .map_err(|e| e.to_string())?;
    let meta = serde_json::json!({
        "fabric": [e.w, e.h],
        "p": g.p,
        "passes": g.passes.len(),
        "matvecs": g.mvs.len(),
        "norms": g.gns.len(),
        "slots": g.slots,
        "attn": g.attn,
        "clip": g.clip,
    });
    let mut meta = meta;
    if !g.groups.is_empty() {
        meta["groups"] = serde_json::json!(g.groups);
    }
    std::fs::write(
        out.join("stream.json"),
        serde_json::to_string_pretty(&meta).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    write_host(&g, &e, out)?;
    Ok(g)
}

/// The `h2d` and `d2h` items of `host.json` for `arrays`, their files written under `out`: every
/// array a memcpy rectangle of PEs with its words' source (the program's own words on file, a
/// static array's spans, an entry's gather), the rows read back (a PE each, or with `rect` the
/// rectangles of PEs reading back alike).
pub(crate) fn host_arrays(
    arrays: &[crate::compile::emit::grid::GridArray],
    out: &std::path::Path,
    rect: bool,
) -> R<(Vec<serde_json::Value>, Vec<serde_json::Value>)> {
    use crate::compile::emit::grid::GridFill;
    use crate::compile::emit::stream::{Fill, HostArray};
    use serde_json::json;
    for d in ["h2d", "entries", "statics"] {
        std::fs::create_dir_all(out.join(d)).map_err(|e| e.to_string())?;
    }
    let ins: Vec<&crate::compile::emit::grid::GridArray> = arrays
        .iter()
        .filter(|a| !matches!(a.fill, GridFill::Back { .. }))
        .collect();
    // (a group's arrays a rectangle apart from another group's at the same PEs)
    let proxies: Vec<HostArray> = ins
        .iter()
        .map(|a| HostArray {
            at: a.at,
            sym: match a.group {
                Some(k) => format!("{}@{k}", a.sym),
                None => a.sym.clone(),
            },
            u16: a.u16,
            len: a.len,
            fill: Fill::Stamps,
        })
        .collect();
    let refs: Vec<&HostArray> = proxies.iter().collect();
    let mut h2d = Vec::new();
    for (k, (x, y, w, h, members)) in crate::compile::stream::host::rects(&refs)
        .into_iter()
        .enumerate()
    {
        let a0 = ins[members[0]];
        let mut item =
            json!({"sym": a0.sym, "x": x, "y": y, "w": w, "h": h, "len": a0.len, "u16": a0.u16});
        if let Some(k) = a0.group {
            item["group"] = json!(k);
        }
        let shape = [h as usize, w as usize, a0.len as usize];
        match &a0.fill {
            GridFill::Words(_) => {
                let mut words: Vec<u32> = Vec::new();
                for &m in &members {
                    let GridFill::Words(v) = &ins[m].fill else {
                        return Err(format!("`{}`: words beside other fills", a0.sym));
                    };
                    words.extend(v);
                }
                let file = format!("h2d/{k}.npy");
                let data: Vec<f32> = words.iter().map(|&x| f32::from_bits(x)).collect();
                crate::compile::npy::write_f32(&out.join(&file), &shape, &data)?;
                item["file"] = json!(file);
            }
            GridFill::Spans(_) => {
                let mut spans: Vec<u32> = Vec::new();
                let mut n = 0;
                for &m in &members {
                    let GridFill::Spans(s) = &ins[m].fill else {
                        return Err(format!("`{}`: spans beside other fills", a0.sym));
                    };
                    let words = crate::compile::stream::host::span_words(s);
                    if words != a0.len {
                        return Err(format!(
                            "`{}` at ({}, {}): spans of {words} words for {}",
                            a0.sym, ins[m].at.x, ins[m].at.y, a0.len
                        ));
                    }
                    spans.extend(s.iter().flatten());
                    n += s.len();
                }
                let file = format!("statics/{k}.npy");
                crate::compile::npy::write_u32(&out.join(&file), &[n, 8], &spans)?;
                item["static"] = json!(file);
            }
            GridFill::Entry(_) => {
                let mut words: Vec<u32> = Vec::new();
                // (each row's PEs taking the same words, the row's first PE's: a write of a
                // row's words, broadcast along it, `rowbcast`)
                let mut rows: BTreeMap<u32, &Vec<crate::compile::stream::host::Word>> =
                    BTreeMap::new();
                let mut along = w > 1;
                for &m in &members {
                    let GridFill::Entry(v) = &ins[m].fill else {
                        return Err(format!("`{}`: an entry beside other fills", a0.sym));
                    };
                    words.extend(v.iter().flatten());
                    along &= rows.entry(ins[m].at.y).or_insert(v) == &v;
                }
                if along {
                    item["rowbcast"] = json!(true);
                }
                let file = format!("entries/{k}.npy");
                crate::compile::npy::write_u32(
                    &out.join(&file),
                    &[h as usize, w as usize, a0.len as usize, 4],
                    &words,
                )?;
                item["entry"] = json!(file);
            }
            GridFill::Back { .. } => {}
        }
        h2d.push(item);
    }
    let mut d2h = Vec::new();
    if rect {
        // the Backs of one sym, length and parts, a rectangle at a time (row-major)
        let backs: Vec<&crate::compile::emit::grid::GridArray> = arrays
            .iter()
            .filter(|a| matches!(a.fill, GridFill::Back { .. }))
            .collect();
        let proxies: Vec<HostArray> = backs
            .iter()
            .map(|a| {
                let GridFill::Back { parts, .. } = &a.fill else {
                    unreachable!()
                };
                HostArray {
                    at: a.at,
                    sym: format!("{}@{parts:?}", a.sym),
                    u16: false,
                    len: a.len,
                    fill: Fill::Stamps,
                }
            })
            .collect();
        let refs: Vec<&HostArray> = proxies.iter().collect();
        for (x, y, w, h, members) in crate::compile::stream::host::rects(&refs) {
            let a0 = backs[members[0]];
            let GridFill::Back { parts, .. } = &a0.fill else {
                unreachable!()
            };
            let mut rows: Vec<usize> = Vec::new();
            for &m in &members {
                if let GridFill::Back { rows: r, .. } = &backs[m].fill {
                    rows.extend(r);
                }
            }
            d2h.push(json!({"sym": a0.sym, "x": x, "y": y, "w": w, "h": h, "len": a0.len, "rows": rows, "parts": parts}));
        }
    } else {
        for a in arrays {
            if let GridFill::Back { rows, parts } = &a.fill {
                d2h.push(json!({"sym": a.sym, "x": a.at.x, "y": a.at.y, "len": a.len, "rows": rows, "parts": parts}));
            }
        }
    }
    Ok((h2d, d2h))
}

fn write_host(
    g: &GridProgram,
    e: &crate::compile::emit::grid::Emitted,
    out: &std::path::Path,
) -> R<()> {
    use serde_json::json;
    let (h2d, d2h) = host_arrays(&e.arrays, out, false)?;
    let outs: Vec<serde_json::Value> = g
        .outs
        .iter()
        .map(|v| json!({"value": v.0, "rows": g.vals[v].rows(), "width": g.vals[v].width}))
        .collect();
    let host = json!({
        "model": g.trace.name, "reading": g.reading,
        "classes": [{"name": g.reading, "word": g.word.to_string(), "rows": 0}],
        "clip": g.clip,
        "outs": outs, "w": g.stream.w, "h": g.stream.h, "h2d": h2d, "d2h": d2h, "refs": [],
        "round16": g.round16, "causal": false,
        "units": [{"name": g.stream.units[0].name, "x": 0, "cols": g.p, "rows": g.p}],
    });
    let mut host = host;
    if !g.groups.is_empty() {
        // a launch a group, its arrays written before it
        host["groups"] = json!(g.groups.len());
    }
    std::fs::write(
        out.join("host.json"),
        serde_json::to_string_pretty(&host).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

/// A value's rows as [`emulate`] reads them: (rows, width, elements).
pub type Rows = crate::compile::stream::host::Rows;

/// The program run on the host, pass by pass, as its kernels compute (the tiles each weight PE
/// holds, from the compile's own spans over the weights; each matvec's input rounded to the
/// build's 16-bit type as a tile multiplies it; a group norm's scale and shift from its input's
/// every row; the gathers' taps): what checks a lowering against the reference before a
/// simulator runs it. `get` gives a value's rows by id (a weight, or a host row value); returns
/// each output's rows.
#[allow(clippy::too_many_lines)]
pub fn emulate(g: &GridProgram, get: &dyn Fn(u32) -> R<Rows>) -> R<BTreeMap<u32, Vec<f32>>> {
    use kernels_ir::interp::{Arg, eval};
    let b16 = |x: f32| -> f32 {
        if g.round16 {
            let b = x.to_bits();
            f32::from_bits(b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff_0000)
        } else {
            crate::compile::stream::host::half_f32(crate::compile::stream::host::half_bits(x))
        }
    };
    let mut vals: BTreeMap<u32, Vec<f32>> = BTreeMap::new();
    for &v in &g.hostv {
        vals.insert(v.0, get(v.0)?.2.to_vec());
    }
    let mut scale: BTreeMap<usize, (Vec<f32>, Vec<f32>)> = BTreeMap::new();
    let weight = |v: ValueId| -> R<std::rc::Rc<Vec<f32>>> { Ok(get(v.0)?.2) };
    let width = |v: ValueId| g.vals[&v].width as usize;
    for (pi, pass) in g.passes.iter().enumerate() {
        let mut steps: Vec<(&Step, u32)> = Vec::new();
        if let Some(gt) = &pass.gather {
            let n = gt.src_grid[0] * gt.src_grid[1] * gt.src_grid[2];
            steps.extend(pass.head.iter().map(|s| (s, n)));
        }
        let head_n = steps.len();
        steps.extend(pass.tail.iter().map(|s| (s, pass.rows())));
        for (si, &(s, rows)) in steps.iter().enumerate() {
            let rows = rows as usize;
            match s {
                Step::Region(ri) => {
                    let r = &g.regs[*ri];
                    let w = r.region.width as usize;
                    let mut outs: Vec<Vec<f32>> = vec![Vec::new(); r.outs.len()];
                    for t in 0..rows {
                        let mut args = Vec::new();
                        for input in &r.ins {
                            args.push(Arg::Row(match input {
                                RegIn::Row(v) => {
                                    let x = vals
                                        .get(&v.0)
                                        .ok_or_else(|| format!("pass {pi}: %{} not made", v.0))?;
                                    x[t * w..(t + 1) * w].to_vec()
                                }
                                RegIn::Res(Res::Value { v, from }) => {
                                    let x = weight(*v)?;
                                    (0..w).map(|e| x[(*from as usize + e) % x.len()]).collect()
                                }
                                RegIn::Res(Res::GnScale(gi)) => scale[gi].0.clone(),
                                RegIn::Res(Res::GnShift(gi)) => scale[gi].1.clone(),
                            }));
                        }
                        let got =
                            eval(&r.region, &args).map_err(|e| format!("region {ri}: {e:?}"))?;
                        for (o, row) in outs.iter_mut().zip(got) {
                            o.extend(row);
                        }
                    }
                    for (v, o) in r.outs.iter().zip(outs) {
                        vals.insert(v.0, o);
                    }
                }
                Step::Gather => {
                    let gt = pass.gather.as_ref().ok_or("a gather with none")?;
                    let src = &vals[&gt.src.0];
                    let cs = width(gt.src);
                    let [_, h, w] = gt.out_grid;
                    let ow = width(gt.out);
                    let mut out = vec![0f32; (h * w) as usize * ow];
                    for y in 0..h {
                        for x in 0..w {
                            let r = (y * w + x) as usize;
                            for (tap, sr) in gt.sources(y, x) {
                                let row = &src[sr as usize * cs..(sr as usize + 1) * cs];
                                match gt.kind {
                                    Kind::Conv { .. } => {
                                        out[r * ow + tap as usize * cs..][..cs].copy_from_slice(row)
                                    }
                                    Kind::Up(_) => out[r * ow..][..cs].copy_from_slice(row),
                                    Kind::Shuffle(f) => {
                                        let rr = (f * f) as usize;
                                        let k = ((y % f) * f + x % f) as usize;
                                        for c in 0..ow {
                                            out[r * ow + c] = row[c * rr + k];
                                        }
                                    }
                                    Kind::Unshuffle(f) => {
                                        let rr = (f * f) as usize;
                                        for c in 0..cs {
                                            out[r * ow + c * rr + tap as usize] = row[c];
                                        }
                                    }
                                }
                            }
                        }
                    }
                    vals.insert(gt.out.0, out);
                }
                Step::Matvec(mi) => {
                    let mv = &g.mvs[*mi];
                    let (kin, xin) = match mv.input {
                        MvIn::Value(x) => (width(x), vals[&x.0].clone()),
                        MvIn::Gathered { .. } => {
                            let gt = pass.gather.as_ref().ok_or("a matvec of no gather")?;
                            (width(gt.out), vals[&gt.out.0].clone())
                        }
                    };
                    let n = width(mv.output);
                    let mut out = vec![0f32; rows * n];
                    let w = if mv.ident {
                        std::rc::Rc::new(Vec::new())
                    } else {
                        weight(mv.weight)?
                    };
                    let wget = |v: u32| -> R<Rows> {
                        if v == mv.weight.0 {
                            Ok((1, w.len(), std::rc::Rc::clone(&w)))
                        } else {
                            get(v)
                        }
                    };
                    let (_, kc, nc) = g.tiles[*mi];
                    for i in 0..g.p {
                        for j in 0..g.p {
                            let (pk, sk) = match mv.from {
                                Side::Row => (j, i),
                                Side::Col => (i, j),
                            };
                            let piece = g.piece(*mi, pk);
                            let slice = g.slice(*mi, sk);
                            if piece.is_empty() || slice.is_empty() {
                                continue;
                            }
                            let sp = crate::compile::emit::grid::tile_spans(g, *mi, i, j);
                            let words = crate::compile::stream::host::apply_spans(&sp, &wget)?;
                            let tile: Vec<f32> = words
                                .iter()
                                .map(|&x| {
                                    if g.round16 {
                                        f32::from_bits(x << 16)
                                    } else {
                                        crate::compile::stream::host::half_f32(x as u16)
                                    }
                                })
                                .collect();
                            for t in 0..rows {
                                for (kk, &ki) in piece.iter().enumerate().take(kc as usize) {
                                    let xv = if (ki as usize) < kin {
                                        b16(xin[t * kin + ki as usize])
                                    } else {
                                        0.0
                                    };
                                    for (nn, o) in slice.iter().enumerate().take(nc as usize) {
                                        if let Some(o) = o {
                                            out[t * n + *o as usize] +=
                                                tile[kk * nc as usize + nn] * xv;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if g.vals[&mv.output].round {
                        for x in &mut out {
                            *x = b16(*x);
                        }
                    }
                    vals.insert(mv.output.0, out);
                }
                Step::Attend(o) => {
                    // the store pass's q / k / v: the previous pass's
                    let Some(Step::Store { q, k, v }) = g.passes[pi - 1]
                        .tail
                        .iter()
                        .find(|s| matches!(s, Step::Store { .. }))
                    else {
                        return Err("an attend with no store".into());
                    };
                    let a = g.attn.as_ref().ok_or("no attention")?;
                    let w = a.width as usize;
                    let (qs, ks, vs) = (&vals[&q.0], &vals[&k.0], &vals[&v.0]);
                    let mut out = vec![0f32; rows * w];
                    for t in 0..rows {
                        let sc: Vec<f64> = (0..rows)
                            .map(|u| {
                                (0..w)
                                    .map(|e| f64::from(qs[t * w + e]) * f64::from(ks[u * w + e]))
                                    .sum::<f64>()
                                    * f64::from(a.scale)
                            })
                            .collect();
                        let m = sc.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                        let ex: Vec<f64> = sc.iter().map(|x| (x - m).exp()).collect();
                        let den: f64 = ex.iter().sum();
                        for e in 0..w {
                            let acc: f64 =
                                (0..rows).map(|u| ex[u] * f64::from(vs[u * w + e])).sum();
                            out[t * w + e] = (acc / den) as f32;
                        }
                    }
                    if g.vals[o].round {
                        for x in &mut out {
                            *x = b16(*x);
                        }
                    }
                    vals.insert(o.0, out);
                }
                Step::Fetch(_)
                | Step::Hold(_)
                | Step::Stats(_)
                | Step::Out(_)
                | Step::Ring(_)
                | Step::Store { .. } => {}
            }
            let _ = (si, head_n);
        }
        // the group norms' scale and shift a channel, from their input's every row
        for &gi in &pass.finals {
            let gn = &g.gns[gi];
            let x = &vals[&gn.x.0];
            let c = width(gn.x);
            let rows = x.len() / c;
            let cpg = c / gn.groups as usize;
            let (ga, be) = (weight(gn.gamma)?, weight(gn.beta)?);
            let (mut a, mut b) = (vec![0f32; c], vec![0f32; c]);
            for q in (0..c).step_by(cpg) {
                let n = (rows * cpg) as f64;
                let mean: f64 = (0..rows)
                    .flat_map(|t| (q..q + cpg).map(move |e| (t, e)))
                    .map(|(t, e)| f64::from(x[t * c + e]))
                    .sum::<f64>()
                    / n;
                let var: f64 = (0..rows)
                    .flat_map(|t| (q..q + cpg).map(move |e| (t, e)))
                    .map(|(t, e)| (f64::from(x[t * c + e]) - mean).powi(2))
                    .sum::<f64>()
                    / n;
                let rs = 1.0 / (var + f64::from(gn.eps)).sqrt();
                for e in q..q + cpg {
                    a[e] = (rs * f64::from(ga[e])) as f32;
                    b[e] = (f64::from(be[e]) - mean * rs * f64::from(ga[e])) as f32;
                }
            }
            scale.insert(gi, (a, b));
        }
    }
    Ok(g.outs
        .iter()
        .map(|v| (v.0, vals.get(&v.0).cloned().unwrap_or_default()))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trace() -> Trace {
        poem_compiler::catalog::repository()
            .parse("flux2-mini-bf16-kv-bf16")
            .expect("flux2-mini parses")
            .trace(poem_ir::Platform::Cerebras)
    }

    /// flux2-mini's VAE decode lowers into passes over one unit: every convolution a matvec, a
    /// pass a group norm's input, the attention's stores and queries two passes.
    #[test]
    fn flux_2_minis_vae_decode_lowers_into_passes() {
        let t = trace();
        assert!(is_grid(&t, "vae.decode"));
        assert!(!is_grid(&t, "denoise"));
        let clip = clip_of(&t, "vae.decode", 16);
        assert_eq!(clip, [1, 8, 8]);
        let g = program(&t, "vae.decode", clip, true).unwrap_or_else(|e| panic!("{e}"));
        eprintln!(
            "P {} passes {} matvecs {} norms {} slots {:?} fabric {} x {}",
            g.p,
            g.passes.len(),
            g.mvs.len(),
            g.gns.len(),
            (g.slots.store_words, g.slots.hin_words, g.slots.hout_words),
            g.stream.w,
            g.stream.h
        );
        for (i, p) in g.passes.iter().enumerate() {
            eprintln!(
                "pass {i}: {:?} gather {:?} head {:?} tail {:?}",
                p.grid,
                p.gather.as_ref().map(|g| (g.kind, g.ring)),
                p.head,
                p.tail
            );
        }
        assert_eq!(g.p, 8);
        assert_eq!(g.mvs.len(), 26 + 4);
        assert_eq!(g.gns.len(), 2 * 10 + 1 + 1);
        assert!(g.attn.is_some());
        assert_eq!(g.outs.len(), 1);
    }
}

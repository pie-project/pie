//! The tiled regime: a grid reading (a VAE's decode) with its image spread over the fabric. The
//! reading becomes ops over the whole image, each finishing everywhere before the next reads its
//! output, each lowered into the existing modules: convolutions (input staged with a halo,
//! weights streamed past every PE a block at a time), group norms (all-reduces), printed regions,
//! upsamplings, and attention (keys and values broadcast from a hub, an online softmax).
//!
//! An image array of `gx x gy` PEs holds each resolution an axis at a time: a tile of pixels a
//! PE, or every few rows of PEs a pixel row (a stride). The weights stream from a tape of PEs up a
//! spine and along every row.

use std::collections::BTreeMap;

use poem_ir::{Attn, Conv, Def, Dtype, Linear, Operands, Operation, Trace, ValueId};
use serde::Serialize;

use crate::compile::emit::region::{self, ParamAt, Place, Printed};
use crate::compile::stream::grid::{self, Prelude, RegIn, Res};
use crate::compile::stream::host::{BF16, FLAT, HALF, LIT, NONE, PACK, Span};

type R<T> = Result<T, String>;

/// A row value of the reading: its width, grid and 16-bit rounding.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TVal {
    pub width: u32,
    pub grid: [u32; 3],
    /// Rounded to the build's 16-bit type where it is made (held f32).
    pub round: bool,
}

/// One op over the whole image.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub enum TOp {
    /// A host value written into its slot.
    In(ValueId),
    /// A convolution of `k x k` taps (1 or 3, stride 1, "same" padding) or a matmul (`k` 1).
    Conv {
        node: u32,
        x: ValueId,
        y: ValueId,
        w: ValueId,
        bias: Option<ValueId>,
        k: u32,
        cin: u32,
        cout: u32,
    },
    /// A group norm; its apply the printed region `reg`.
    Norm {
        x: ValueId,
        y: ValueId,
        groups: u32,
        eps: f32,
        gamma: ValueId,
        beta: ValueId,
        reg: usize,
    },
    /// A printed region.
    Region(usize),
    /// Nearest upsampling by two along y (`axis` 0) or x (1); with `sel`, a pixel shuffle's
    /// half of it (the child at offset `a` along the axis takes channels `2 c + a`).
    Up {
        x: ValueId,
        y: ValueId,
        axis: usize,
        sel: bool,
    },
    /// Attention of every pixel's q over every pixel's k / v.
    Attend {
        q: ValueId,
        k: ValueId,
        v: ValueId,
        o: ValueId,
        scale: f32,
    },
    /// A value read out (kept for the host to read).
    Out(ValueId),
}

/// A printed region and what it reads and makes.
#[derive(Clone, Debug)]
pub struct TReg {
    pub region: kernels_ir::Region,
    pub ins: Vec<RegIn>,
    pub outs: Vec<ValueId>,
    /// The columns a call of the module runs (the row a chunk at a time).
    pub chunk: u32,
    pub printed: Option<Printed>,
    /// Its module (regions printed alike share one).
    pub module: usize,
}

/// A resolution's layout over the image array, an axis each: a tile's pixels (`th`, `tw`) and
/// the stride's log (`lsy`, `lsx`), its blocks (`nrb` x `ncb`, a block a busy PE).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Lay {
    pub th: u32,
    pub tw: u32,
    pub lsy: u32,
    pub lsx: u32,
    pub nrb: u32,
    pub ncb: u32,
}

impl Lay {
    /// A busy PE's pixels.
    #[must_use]
    pub fn px(&self) -> u32 {
        self.th * self.tw
    }
}

/// What an arena slot holds: a value, or an op's scratch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum Key {
    Val(ValueId),
    Temp(usize),
}

/// A tiled reading's program.
#[derive(Clone, Debug)]
pub struct TileProgram {
    pub trace: Trace,
    pub reading: String,
    pub word: u64,
    pub clip: [u32; 3],
    pub vals: BTreeMap<ValueId, TVal>,
    pub ops: Vec<TOp>,
    pub regs: Vec<TReg>,
    /// The printed modules (name, text).
    pub modules: Vec<(String, String)>,
    pub hostv: Vec<ValueId>,
    pub outs: Vec<ValueId>,
    /// The image array.
    pub gx: u32,
    pub gy: u32,
    /// Each convolution's block rows (by op), the rest 0.
    pub kb: Vec<u32>,
    /// Each convolution's staged channels: its input staged (and its halo exchanged) a chunk
    /// of `kc` channels at a time, every tap's blocks of the chunk then.
    pub kc: Vec<u32>,
    /// Arena offsets (f32 words), its size, the staging buffer (16-bit) and weight block (words).
    pub slots: BTreeMap<Key, u32>,
    pub arena: u32,
    pub hx: u32,
    pub wb: u32,
    /// The weight stream (every record in op order), its words, the tape's PEs.
    pub stream: Vec<Span>,
    pub stream_words: u32,
    pub tape: u32,
    pub round16: bool,
    /// The PE's buffers packed tightly ([`Chunks::tight`]).
    pub tight: bool,
    /// Each op's words in the stream ahead of its records (their 16-bit words, an even count),
    /// none where the op table is a PE's constant.
    pub op_words: Vec<u32>,
}

/// A weight block's words at most (two 16-bit weights a word).
pub const WB_WORDS: u32 = 512;
/// A convolution's staged channels at most.
pub const KC: u32 = 128;
/// f32 words a PE gives its arena, staging buffer and weight block (of 12,288).
pub const DATA_WORDS: u32 = 8_448;
/// A printed region's chunk at most.
pub const CHUNK: u32 = 8;

/// The lowering's state.
struct L<'t> {
    t: &'t Trace,
    pre: Prelude,
    vals: BTreeMap<ValueId, TVal>,
    ops: Vec<TOp>,
    regs: Vec<TReg>,
    hostv: Vec<ValueId>,
    next: u32,
}

fn refuse(t: &Trace, at: usize, why: impl std::fmt::Display) -> String {
    format!("{}: node {at}: the tiled regime: {why}", t.name)
}

impl L<'_> {
    fn round(&self, v: ValueId) -> bool {
        matches!(grid::dtype(self.t, v), Some(Dtype::Bf16 | Dtype::F16))
    }

    fn is_host(&self, v: ValueId) -> bool {
        match &self.t.values[v.0 as usize].def {
            Def::Input(_) => true,
            Def::Op(n) => self.pre.host[*n as usize],
            _ => false,
        }
    }

    /// The grid `v` is on: its op's grid value's, else its first voxel-row input's.
    fn grid_of(&self, v: ValueId) -> R<[u32; 3]> {
        if let Some(x) = self.vals.get(&v) {
            return Ok(x.grid);
        }
        let g = |gv: &ValueId| {
            self.pre
                .grids
                .get(gv)
                .copied()
                .ok_or_else(|| format!("value %{}: a grid the compile does not know", gv.0))
        };
        match &self.t.values[v.0 as usize].def {
            Def::Input(poem_ir::RuntimeInput::Guest {
                axis: poem_ir::InputAxis::Voxel,
                ..
            }) => Err(format!("value %{}: a voxel input the host keeps", v.0)),
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
                            .find(|i| grid::voxel_rows(&self.t.values[i.0 as usize].ty))
                            .ok_or_else(|| format!("value %{}: on no voxel grid", v.0))?;
                        self.grid_of(*x)
                    }
                }
            }
            _ => Err(format!("value %{}: on no voxel grid", v.0)),
        }
    }

    /// `v` readable: made on the wafer, or the host's (written in by an `In`).
    fn avail(&mut self, v: ValueId) -> R<()> {
        if self.vals.contains_key(&v) {
            return Ok(());
        }
        if !self.is_host(v) {
            return Err(format!("value %{}: read before it is made", v.0));
        }
        let grid = self.grid_of(v)?;
        let val = TVal {
            width: grid::width(self.t, v),
            grid,
            round: self.round(v),
        };
        self.vals.insert(v, val);
        self.hostv.push(v);
        self.ops.push(TOp::In(v));
        Ok(())
    }

    fn make(&mut self, v: ValueId, width: u32, grid: [u32; 3], round: bool) {
        self.vals.insert(v, TVal { width, grid, round });
    }

    fn pseudo(&mut self, val: TVal) -> ValueId {
        let v = ValueId(self.next);
        self.next += 1;
        self.vals.insert(v, val);
        v
    }

    fn run(&mut self, nodes: &[u32]) -> R<()> {
        if nodes.is_empty() {
            return Ok(());
        }
        for c in grid::split_runs(self.t, nodes).map_err(|e| format!("{}: {e}", self.t.name))? {
            let mut ins = Vec::new();
            let mut grid = None;
            for (k, i) in c.ins.iter().enumerate() {
                if c.region.params[k] == kernels_ir::Shape::Scalar || i.by.is_some() {
                    return Err(refuse(
                        self.t,
                        c.nodes[0] as usize,
                        "a region over a scalar a row",
                    ));
                }
                if i.shared {
                    ins.push(RegIn::Res(Res::Value {
                        v: i.value,
                        from: i.from,
                    }));
                } else {
                    if i.from != 0 {
                        return Err(refuse(
                            self.t,
                            c.nodes[0] as usize,
                            format!("a region reading a value from column {}", i.from),
                        ));
                    }
                    self.avail(i.value)?;
                    let x = &self.vals[&i.value];
                    if x.width != c.region.width {
                        return Err(refuse(
                            self.t,
                            c.nodes[0] as usize,
                            format!("a region of {} over a row of {}", c.region.width, x.width),
                        ));
                    }
                    grid = Some(x.grid);
                    ins.push(RegIn::Row(i.value));
                }
            }
            let grid =
                grid.ok_or_else(|| refuse(self.t, c.nodes[0] as usize, "a region of no row"))?;
            let outs: Vec<ValueId> = c.outs.iter().map(|o| o.value).collect();
            for o in &c.outs {
                let w = grid::width(self.t, o.value);
                self.make(
                    o.value,
                    w,
                    grid,
                    matches!(o.dtype, Dtype::Bf16 | Dtype::F16),
                );
            }
            let r = self.regs.len();
            self.regs.push(TReg {
                region: c.region,
                ins,
                outs,
                chunk: 0,
                printed: None,
                module: 0,
            });
            self.ops.push(TOp::Region(r));
        }
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    fn node(&mut self, at: usize) -> R<()> {
        let t = self.t;
        let node = &t.nodes[at];
        match &node.op {
            Operation::Conv(Conv::Grid { .. }) => Ok(()),
            Operation::Conv(Conv::Conv3d {
                x,
                w,
                bias,
                k,
                stride,
                pad,
                pad_back,
                causal_t,
                dilation,
                depthwise,
                cache,
                y,
                ..
            }) => {
                if *causal_t || *depthwise || cache.is_some() || *dilation != [1, 1, 1] {
                    return Err(refuse(
                        t,
                        at,
                        "a causal, depthwise, dilated or cached convolution",
                    ));
                }
                let kk = match (k, stride, pad, pad_back) {
                    ([1, 1, 1], [1, 1, 1], [0, 0, 0], [0, 0, 0]) => 1,
                    ([1, 3, 3], [1, 1, 1], [0, 1, 1], [0, 1, 1]) => 3,
                    _ => {
                        return Err(refuse(
                            t,
                            at,
                            format!(
                                "a convolution of {k:?} taps, stride {stride:?}, padding {pad:?} / {pad_back:?}: \
                                 1 x 1 or 3 x 3 \"same\" at stride 1 alone"
                            ),
                        ));
                    }
                };
                self.avail(*x)?;
                let grid = self.grid_of(*y)?;
                if self.vals[x].grid != grid {
                    return Err(refuse(t, at, "a convolution changing its grid"));
                }
                let (cin, cout) = (grid::width(t, *x), grid::width(t, *y));
                self.make(*y, cout, grid, self.round(*y));
                self.ops.push(TOp::Conv {
                    node: at as u32,
                    x: *x,
                    y: *y,
                    w: *w,
                    bias: *bias,
                    k: kk,
                    cin,
                    cout,
                });
                Ok(())
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
                self.avail(*act)?;
                let grid = self.vals[act].grid;
                let (cin, cout) = (grid::width(t, *act), grid::width(t, *y));
                self.make(*y, cout, grid, self.round(*y));
                self.ops.push(TOp::Conv {
                    node: at as u32,
                    x: *act,
                    y: *y,
                    w: *w,
                    bias: None,
                    k: 1,
                    cin,
                    cout,
                });
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
                    return Err(refuse(t, at, "a group norm over the host's rows"));
                }
                self.avail(*x)?;
                let xv = self.vals[x].clone();
                if *groups == 0 || !xv.width.is_multiple_of(*groups) {
                    return Err(refuse(
                        t,
                        at,
                        format!("{} groups over {}", groups, xv.width),
                    ));
                }
                let round = self.round(*y);
                self.make(*y, xv.width, xv.grid, round);
                let reg = self.regs.len();
                let gi = self.ops.len();
                self.regs.push(TReg {
                    region: grid::norm_region(xv.width, *silu, round),
                    ins: vec![
                        RegIn::Row(*x),
                        RegIn::Res(Res::GnScale(gi)),
                        RegIn::Res(Res::GnShift(gi)),
                    ],
                    outs: vec![*y],
                    chunk: 0,
                    printed: None,
                    module: 0,
                });
                self.ops.push(TOp::Norm {
                    x: *x,
                    y: *y,
                    groups: *groups,
                    eps: *eps,
                    gamma: *weight,
                    beta: *bias,
                    reg,
                });
                Ok(())
            }
            Operation::Conv(Conv::Upsample { x, factor, y, .. }) => {
                if factor[0] != 1 || factor[1] != factor[2] || !factor[1].is_power_of_two() {
                    return Err(refuse(
                        t,
                        at,
                        format!("an upsampling by {factor:?}: powers of two in h and w alike"),
                    ));
                }
                self.avail(*x)?;
                let xv = self.vals[x].clone();
                let mut cur = *x;
                let mut g = xv.grid;
                let steps = factor[1].trailing_zeros();
                for s in 0..steps {
                    for axis in 0..2 {
                        g[1 + axis] *= 2;
                        let last = s + 1 == steps && axis == 1;
                        let next = if last {
                            self.make(*y, xv.width, g, xv.round);
                            *y
                        } else {
                            self.pseudo(TVal {
                                width: xv.width,
                                grid: g,
                                round: xv.round,
                            })
                        };
                        self.ops.push(TOp::Up {
                            x: cur,
                            y: next,
                            axis,
                            sel: false,
                        });
                        cur = next;
                    }
                }
                Ok(())
            }
            Operation::Conv(Conv::PixelShuffle {
                x, r, trim_t, y, ..
            }) => {
                if *r != [1, 2, 2] || *trim_t != 0 {
                    return Err(refuse(
                        t,
                        at,
                        format!("a pixel shuffle by {r:?}: by two in h and w alone"),
                    ));
                }
                self.avail(*x)?;
                let xv = self.vals[x].clone();
                if !xv.width.is_multiple_of(4) {
                    return Err(refuse(t, at, "a pixel shuffle of a width not four times"));
                }
                // along x (channel 2m + b to column child b), then along y (2c + a)
                let mut g = xv.grid;
                g[2] *= 2;
                let mid = self.pseudo(TVal {
                    width: xv.width / 2,
                    grid: g,
                    round: xv.round,
                });
                g[1] *= 2;
                self.make(*y, xv.width / 4, g, self.round(*y));
                self.ops.push(TOp::Up {
                    x: *x,
                    y: mid,
                    axis: 1,
                    sel: true,
                });
                self.ops.push(TOp::Up {
                    x: mid,
                    y: *y,
                    axis: 0,
                    sel: true,
                });
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
                    return Err(refuse(t, at, "attn.voxel over blocks of frames"));
                }
                for x in [q, k, v] {
                    self.avail(*x)?;
                }
                let qv = self.vals[q].clone();
                self.make(*o, qv.width, qv.grid, self.round(*o));
                self.ops.push(TOp::Attend {
                    q: *q,
                    k: *k,
                    v: *v,
                    o: *o,
                    scale: *sm_scale,
                });
                Ok(())
            }
            op => Err(refuse(
                t,
                at,
                format!("{} has no tiled lowering", op.name()),
            )),
        }
    }
}

/// `trace`'s grid `reading` on `clip`, lowered into ops (no placement yet).
#[allow(clippy::too_many_lines)]
pub fn lower(trace: &Trace, reading: &str, clip: [u32; 3], round16: bool) -> R<TileProgram> {
    let pre = grid::prelude(trace, reading, clip)?;
    let word = pre.word;
    let mut l = L {
        t: trace,
        pre,
        vals: BTreeMap::new(),
        ops: Vec::new(),
        regs: Vec::new(),
        hostv: Vec::new(),
        next: trace.values.len() as u32,
    };
    let mut pending: Vec<u32> = Vec::new();
    for (at, n) in trace.nodes.iter().enumerate() {
        if !l.pre.runs[at] || l.pre.host[at] {
            continue;
        }
        if matches!(n.op, Operation::Elemwise(_)) {
            pending.push(at as u32);
            continue;
        }
        let run = std::mem::take(&mut pending);
        l.run(&run)?;
        l.node(at)?;
    }
    let run = std::mem::take(&mut pending);
    l.run(&run)?;
    let mut outs = Vec::new();
    for o in &trace.outputs {
        if l.vals.contains_key(&o.value) && !l.hostv.contains(&o.value) && !outs.contains(&o.value)
        {
            outs.push(o.value);
            l.ops.push(TOp::Out(o.value));
        }
    }
    if outs.is_empty() {
        return Err(format!(
            "{}: grid reading `{reading}` reads nothing out",
            trace.name
        ));
    }
    for v in l.vals.values() {
        let [t, h, w] = v.grid;
        if t != 1 || !h.is_power_of_two() || !w.is_power_of_two() {
            return Err(format!(
                "{}: the tiled regime: a grid {:?} (one frame, powers of two)",
                trace.name, v.grid
            ));
        }
    }
    let ops = l.ops;
    let mut p = TileProgram {
        trace: trace.clone(),
        reading: reading.to_string(),
        word,
        clip,
        vals: l.vals,
        kb: vec![0; ops.len()],
        kc: vec![0; ops.len()],
        ops,
        regs: l.regs,
        modules: Vec::new(),
        hostv: l.hostv,
        outs,
        gx: 0,
        gy: 0,
        slots: BTreeMap::new(),
        arena: 0,
        hx: 0,
        wb: 0,
        stream: Vec::new(),
        stream_words: 0,
        tape: 0,
        round16,
        op_words: Vec::new(),
        tight: false,
    };
    print_regions(&mut p)?;
    Ok(p)
}

/// Each region printed at its chunk (the most columns up to [`CHUNK`] that cut its row evenly),
/// modules printed alike (but for their comments) shared.
fn print_regions(p: &mut TileProgram) -> R<()> {
    let mut texts: BTreeMap<String, usize> = BTreeMap::new();
    for ri in 0..p.regs.len() {
        let r = &p.regs[ri];
        let w = r.region.width;
        let mut chunk = CHUNK.min(w);
        while !w.is_multiple_of(chunk) {
            chunk -= 1;
        }
        let params: Vec<ParamAt> = r
            .ins
            .iter()
            .map(|i| match i {
                RegIn::Row(_) => ParamAt::Row,
                RegIn::Res(_) => ParamAt::Resident,
            })
            .collect();
        let printed = region::print(
            &r.region,
            &Place {
                chunk,
                roots: w / chunk,
                params,
                round16: p.round16,
                rope_lv: 1,
                rope_bits: 3,
                straddle: false,
                columns: None,
                dup: 1,
            },
        )
        .map_err(|e| format!("{}: region {ri}: {e}", p.trace.name))?;
        if !printed.tables.is_empty() || printed.all_reduces + printed.head_reduces > 0 {
            return Err(format!(
                "{}: region {ri}: a region with tables or a row's reduction on a grid row",
                p.trace.name
            ));
        }
        // printed alike but for their comments (a region's row width): one module
        let key: String = printed
            .csl
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        let n = texts.len();
        let m = *texts.entry(key).or_insert(n);
        if m == n {
            p.modules
                .push((format!("trg_{m}.csl"), printed.csl.clone()));
        }
        let r = &mut p.regs[ri];
        r.chunk = chunk;
        r.printed = Some(printed);
        r.module = m;
    }
    Ok(())
}

impl TileProgram {
    /// The layout of a value on `grid`.
    #[must_use]
    pub fn lay(&self, grid: [u32; 3]) -> Lay {
        let [_, h, w] = grid;
        let (th, lsy) = if h >= self.gy {
            (h / self.gy, 0)
        } else {
            (1, (self.gy / h).trailing_zeros())
        };
        let (tw, lsx) = if w >= self.gx {
            (w / self.gx, 0)
        } else {
            (1, (self.gx / w).trailing_zeros())
        };
        Lay {
            th,
            tw,
            lsy,
            lsx,
            nrb: h / th,
            ncb: w / tw,
        }
    }

    #[must_use]
    pub fn lay_of(&self, v: ValueId) -> Lay {
        self.lay(self.vals[&v].grid)
    }

    /// A value's words a busy PE.
    #[must_use]
    pub fn words(&self, v: ValueId) -> u32 {
        self.lay_of(v).px() * self.vals[&v].width
    }

    /// The largest grid's (h, w).
    #[must_use]
    pub fn top(&self) -> (u32, u32) {
        let h = self.vals.values().map(|v| v.grid[1]).max().unwrap_or(1);
        let w = self.vals.values().map(|v| v.grid[2]).max().unwrap_or(1);
        (h, w)
    }

    /// The values op `i` reads and makes.
    #[must_use]
    pub fn io(&self, i: usize) -> (Vec<ValueId>, Vec<ValueId>) {
        match &self.ops[i] {
            TOp::In(v) => (Vec::new(), vec![*v]),
            TOp::Out(v) => (vec![*v], Vec::new()),
            TOp::Conv { x, y, .. } | TOp::Norm { x, y, .. } | TOp::Up { x, y, .. } => {
                (vec![*x], vec![*y])
            }
            TOp::Region(r) => {
                let g = &self.regs[*r];
                let ins = g
                    .ins
                    .iter()
                    .filter_map(|i| match i {
                        RegIn::Row(v) => Some(*v),
                        RegIn::Res(_) => None,
                    })
                    .collect();
                (ins, g.outs.clone())
            }
            TOp::Attend { q, k, v, o, .. } => (vec![*q, *k, *v], vec![*o]),
        }
    }

    /// Op `i`'s scratch words in the arena.
    #[must_use]
    pub fn temp(&self, i: usize) -> u32 {
        match &self.ops[i] {
            TOp::Conv { bias, cout, .. } => {
                if bias.is_some() {
                    *cout
                } else {
                    0
                }
            }
            TOp::Norm { x, groups, .. } => 3 * self.vals[x].width + 3 * groups,
            TOp::Region(r) => {
                let g = &self.regs[*r];
                g.ins.iter().filter(|i| matches!(i, RegIn::Res(_))).count() as u32 * g.region.width
            }
            TOp::Attend { q, .. } => {
                // a key widened (then its value), a product where a PE holds several queries,
                // a query's max, sum and score
                let w = self.vals[q].width;
                let px = self.lay_of(*q).px();
                w + if px > 1 { w } else { 0 } + 3 * px
            }
            TOp::In(_) | TOp::Out(_) | TOp::Up { .. } => 0,
        }
    }

    /// The tape's records of op `i`, in the order it reads them.
    #[must_use]
    pub fn records(&self, i: usize) -> Vec<Span> {
        let cast = if self.round16 { BF16 } else { HALF };
        let f32s = |v: ValueId, from: u32, n: u32| [v.0, 0, from, n, 1, FLAT, 1, 0];
        match &self.ops[i] {
            TOp::Conv {
                w,
                bias,
                k,
                cin,
                cout,
                ..
            } => {
                let (kb, kc) = (self.kb[i], self.kc[i]);
                let full = k * k * cin;
                let mut s = Vec::new();
                for ch in 0..cin / kc {
                    for t in 0..k * k {
                        for b in 0..kc / kb {
                            for r in 0..kb {
                                s.push([
                                    w.0,
                                    0,
                                    t * cin + ch * kc + b * kb + r,
                                    *cout,
                                    full,
                                    FLAT | cast | PACK,
                                    1,
                                    0,
                                ]);
                            }
                        }
                    }
                }
                if let Some(b) = bias {
                    s.push(f32s(*b, 0, *cout));
                }
                s
            }
            TOp::Norm { x, gamma, beta, .. } => {
                let c = self.vals[x].width;
                vec![f32s(*gamma, 0, c), f32s(*beta, 0, c)]
            }
            TOp::Region(r) => {
                let g = &self.regs[*r];
                g.ins
                    .iter()
                    .filter_map(|i| match i {
                        RegIn::Res(Res::Value { v, from }) => Some(f32s(*v, *from, g.region.width)),
                        _ => None,
                    })
                    .collect()
            }
            _ => Vec::new(),
        }
    }

    /// The words the hub relays a launch, in op order: each norm's two all-reduces, each
    /// attention's keys and values (two 16-bit values a word).
    #[must_use]
    pub fn hub(&self) -> Vec<u32> {
        let mut out = Vec::new();
        for op in &self.ops {
            match op {
                TOp::Norm { groups, .. } => out.extend([*groups, *groups]),
                TOp::Attend { q, .. } => {
                    let v = &self.vals[q];
                    out.push(v.grid[1] * v.grid[2] * v.width);
                }
                _ => {}
            }
        }
        out
    }

    /// The program's fabric: the spine column, the image array, the hub east of its last row;
    /// the tape's rows below.
    #[must_use]
    pub fn fabric(&self) -> (u32, u32) {
        let rows = self.tape.div_ceil(self.gx + 1).max(1);
        (self.gx + 2, self.gy + rows)
    }

    /// Tape PE `k`'s place: a snake from below the spine east, then west, a row each.
    #[must_use]
    pub fn tape_at(&self, k: u32) -> (u32, u32) {
        let n = self.gx + 1;
        let (r, c) = (k / n, k % n);
        let x = if r % 2 == 0 { c } else { self.gx - c };
        (x, self.gy + r)
    }
}

/// A convolution's staged channels and weight block at most: a point's choice (smaller, a PE
/// holds less staging and block for more passes and blocks).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Chunks {
    /// A convolution's staged channels at most.
    pub kc: u32,
    /// A weight block's words at most (two 16-bit weights a word).
    pub wb: u32,
    /// The op table streams in with the weights, an op's words ahead of its records (a PE holds
    /// one op's), not held whole on every PE.
    pub stream_ops: bool,
    /// The PE's buffers as tight as the compiler packs them: a rounding staged a chunk at a
    /// time (the staging no longer holds a whole value), the arena the least of several orders
    /// of first fit.
    pub tight: bool,
}

impl Default for Chunks {
    fn default() -> Self {
        Chunks {
            kc: KC,
            wb: WB_WORDS,
            stream_ops: false,
            tight: false,
        }
    }
}

/// The program at a top tile of `th0 x tw0` pixels (the image array the top grid over it), with
/// `chunks` for the staged channels and the weight block: its slots, buffers and tape, or why it
/// does not fit.
pub fn place_with(
    p: &mut TileProgram,
    th0: u32,
    tw0: u32,
    bounds: Option<(u32, u32)>,
    chunks: Chunks,
) -> R<()> {
    let (kc_most, wb_most) = (chunks.kc, chunks.wb);
    p.tight = chunks.tight;
    let (h, w) = p.top();
    if !h.is_multiple_of(th0) || !w.is_multiple_of(tw0) {
        return Err(format!("a {th0} x {tw0} tile over {h} x {w}"));
    }
    p.gy = h / th0;
    p.gx = w / tw0;
    // each convolution's staged chunk (the most channels up to KC that cut c_in), its block
    // rows (the most cutting the chunk whose block fits the weight block, an even count)
    let mut wb = 1u32;
    for i in 0..p.ops.len() {
        if let TOp::Conv { cin, cout, .. } = p.ops[i] {
            let k3 = matches!(p.ops[i], TOp::Conv { k: 3, .. });
            let mut kc = cin.min(kc_most);
            while kc > 1 && (!cin.is_multiple_of(kc) || (k3 && kc % 2 != 0)) {
                kc -= 1;
            }
            let mut kb = kc;
            while kb > 1
                && (kb * cout > 2 * wb_most || !kc.is_multiple_of(kb) || (kb * cout) % 2 != 0)
            {
                kb -= 1;
            }
            if (kb * cout) % 2 != 0 || kb * cout > 2 * wb_most {
                return Err(format!(
                    "op {i}: a convolution to {cout} channels: no block of rows fits {wb_most} words"
                ));
            }
            p.kb[i] = kb;
            p.kc[i] = kc;
            wb = wb.max(kb * cout / 2);
        }
    }
    p.wb = wb;
    // halos and keys go two 16-bit values a wavelet
    for (i, op) in p.ops.iter().enumerate() {
        let w = match op {
            TOp::Conv { k: 3, cin, .. } => *cin,
            TOp::Attend { q, .. } => p.vals[q].width,
            _ => 0,
        };
        if w % 2 != 0 {
            return Err(format!(
                "op {i}: an odd width {w} sent two 16-bit values a wavelet"
            ));
        }
    }
    // the staging buffer (16-bit): a convolution's input and halo, an attention's key and
    // value, a rounding's
    let mut hx = 2u32;
    for i in 0..p.ops.len() {
        match &p.ops[i] {
            TOp::Conv { x, y, k, cin, .. } => {
                let l = p.lay_of(*x);
                let kc = p.kc[i];
                let n = if *k == 3 {
                    (l.th + 2) * (l.tw + 2) * kc
                } else {
                    l.px() * kc
                };
                let _ = cin;
                hx = hx.max(n);
                if !chunks.tight {
                    hx = hx.max(p.words(*y));
                }
            }
            TOp::Attend { q, o, .. } => hx = hx.max(2 * p.vals[q].width).max(p.words(*o)),
            op => {
                let (_, outs) = p.io(i);
                for v in outs {
                    if p.vals[&v].round && !chunks.tight {
                        hx = hx.max(p.words(v));
                    }
                }
                let _ = op;
            }
        }
    }
    p.hx = hx.next_multiple_of(2);
    // the arena: each value from the op making it to its last reader, each op's scratch its own
    let n = p.ops.len();
    let mut life: BTreeMap<Key, (usize, usize, u32)> = BTreeMap::new();
    for i in 0..n {
        let (ins, outs) = p.io(i);
        for v in outs {
            life.insert(Key::Val(v), (i, i, p.words(v)));
        }
        for v in ins {
            let e = life
                .get_mut(&Key::Val(v))
                .ok_or_else(|| format!("op {i} reads %{} before it is made", v.0))?;
            e.1 = i;
        }
        let tw = p.temp(i);
        if tw > 0 {
            life.insert(Key::Temp(i), (i, i, tw));
        }
    }
    let items: Vec<(Key, (usize, usize, u32))> = life.into_iter().collect();
    // first fit in an order: by start, the wider first (the tight choice: the least of that and
    // the widest first, the longest-lived first)
    let fit = |order: &Order| {
        let mut items = items.clone();
        items.sort_by_key(|it| (order(it), it.0));
        let mut placed: Vec<(u32, u32, usize, usize)> = Vec::new();
        let mut slots = BTreeMap::new();
        let mut arena = 0u32;
        for (k, (s, e, w)) in items {
            // the lowest offset free over [s, e] (4-word aligned)
            let mut busy: Vec<(u32, u32)> = placed
                .iter()
                .filter(|(_, _, ps, pe)| *ps <= e && s <= *pe)
                .map(|(o, w, _, _)| (*o, o + w))
                .collect();
            busy.sort_unstable();
            let mut off = 0u32;
            for (a, b) in busy {
                if off + w <= a {
                    break;
                }
                off = off.max(b.next_multiple_of(4));
            }
            placed.push((off, w, s, e));
            slots.insert(k, off);
            arena = arena.max(off + w);
        }
        (arena, slots)
    };
    let by_start = |(_, (s, _, w)): &(Key, (usize, usize, u32))| (*s as i64, -i64::from(*w), 0i64);
    let (mut arena, mut slots) = fit(&by_start);
    if chunks.tight {
        let widest =
            |(_, (s, _, w)): &(Key, (usize, usize, u32))| (-i64::from(*w), *s as i64, 0i64);
        let longest = |(_, (s, e, w)): &(Key, (usize, usize, u32))| {
            (-((e - s) as i64), -i64::from(*w), *s as i64)
        };
        for order in [&widest as &Order, &longest] {
            let (a, sl) = fit(order);
            if a < arena {
                (arena, slots) = (a, sl);
            }
        }
    }
    p.slots = slots;
    p.arena = arena.next_multiple_of(4).max(4);
    let data = p.arena + p.hx / 2 + p.wb;
    if data > DATA_WORDS {
        return Err(format!(
            "a PE's arena {} + staging {} + weight block {} words, past {DATA_WORDS}",
            p.arena,
            p.hx / 2,
            p.wb
        ));
    }
    if p.arena > 32_000 || p.hx > 32_000 {
        return Err("an arena past a DSD offset".into());
    }
    // the stream (each op's words ahead of its records where the op table streams)
    p.stream.clear();
    p.op_words = Vec::new();
    let table = if chunks.stream_ops {
        crate::compile::emit::tiled::op_words(p)?
    } else {
        Vec::new()
    };
    for i in 0..n {
        if let Some(w) = table.get(i) {
            let even = w.len().next_multiple_of(2);
            p.op_words.push(even as u32);
            for k in 0..even {
                let x = w.get(k).copied().unwrap_or(0);
                p.stream.push([LIT, x, 0, 1, 0, PACK, 1, 0]);
            }
        }
        let r = p.records(i);
        p.stream.extend(r);
    }
    p.stream_words = crate::compile::stream::host::span_words(&p.stream);
    p.tape = p
        .stream_words
        .div_ceil(crate::compile::emit::tiled::tape_words())
        .max(1);
    if let Some((bw, bh)) = bounds {
        let (fw, fh) = p.fabric();
        if fw > bw || fh > bh {
            return Err(format!(
                "a fabric of {fw} x {fh} PEs (array {} x {}, tape {}) past {bw} x {bh}",
                p.gx, p.gy, p.tape
            ));
        }
    }
    Ok(())
}

/// Tape PE `k`'s spans: its words of the stream (zeros past the stream's end).
pub fn tape_spans(p: &TileProgram, k: u32) -> Vec<Span> {
    let lo = 2 * u64::from(k) * u64::from(crate::compile::emit::tiled::tape_words());
    let hi = lo + 2 * u64::from(crate::compile::emit::tiled::tape_words());
    let mut out: Vec<Span> = Vec::new();
    let mut at = 0u64; // in halves
    for s in &p.stream {
        let [v, row, col, len, stride, flags, count, step] = *s;
        debug_assert_eq!(count, 1);
        let per = if flags & PACK != 0 { 1u64 } else { 2 };
        let n = u64::from(len) * per;
        let (a, b) = (at, at + n);
        at = b;
        if b <= lo || a >= hi {
            continue;
        }
        let from = (lo.max(a) - a) / per;
        let to = (hi.min(b) - a) / per;
        out.push([
            v,
            row,
            col + from as u32 * stride,
            (to - from) as u32,
            stride,
            flags,
            count,
            step,
        ]);
    }
    let used = crate::compile::stream::host::span_words(&out);
    if used < crate::compile::emit::tiled::tape_words() {
        out.push([
            NONE,
            0,
            0,
            crate::compile::emit::tiled::tape_words() - used,
            0,
            0,
            1,
            0,
        ]);
    }
    out
}

/// The search over the top tile: every power-of-two tile whose array, memory and fabric fit,
/// the cost model's cheapest (or `pin`'s).
pub fn program(
    trace: &Trace,
    reading: &str,
    clip: [u32; 3],
    round16: bool,
    pin: Option<(u32, u32)>,
    bounds: Option<(u32, u32)>,
) -> R<TileProgram> {
    program_upto(trace, reading, clip, round16, pin, bounds, None)
}

/// The program at top tile `tile` with `chunks` (a point [`points`] lists).
pub fn program_with(
    trace: &Trace,
    reading: &str,
    clip: [u32; 3],
    round16: bool,
    tile: (u32, u32),
    chunks: Chunks,
    bounds: Option<(u32, u32)>,
) -> R<TileProgram> {
    let mut p = lower(trace, reading, clip, round16)?;
    place_with(&mut p, tile.0, tile.1, bounds, chunks)?;
    Ok(p)
}

/// [`program`] over its first `upto` ops alone (a part of a reading on the simulator), the value
/// the last of them makes read out.
pub fn program_upto(
    trace: &Trace,
    reading: &str,
    clip: [u32; 3],
    round16: bool,
    pin: Option<(u32, u32)>,
    bounds: Option<(u32, u32)>,
    upto: Option<usize>,
) -> R<TileProgram> {
    let mut base = lower(trace, reading, clip, round16)?;
    if let Some(n) = upto {
        base.ops.retain(|o| !matches!(o, TOp::Out(_)));
        base.ops.truncate(n.max(1));
        let last = base.ops.len() - 1;
        let v = *base
            .io(last)
            .1
            .first()
            .ok_or("the last op makes no value")?;
        base.ops.push(TOp::Out(v));
        base.outs = vec![v];
        base.kb = vec![0; base.ops.len()];
        base.kc = vec![0; base.ops.len()];
    }
    let (h, w) = base.top();
    let mut tiles = Vec::new();
    match pin {
        Some(t) => tiles.push(t),
        None => {
            let mut a = 1;
            while a <= h.min(16) {
                let mut b = 1;
                while b <= w.min(16) {
                    tiles.push((a, b));
                    b *= 2;
                }
                a *= 2;
            }
        }
    }
    let mut best: Option<(f64, TileProgram)> = None;
    let mut why = String::new();
    for (a, b) in tiles {
        let mut p = base.clone();
        match place_with(&mut p, a, b, bounds, Chunks::default()) {
            Ok(()) => {
                let c = crate::compile::cost::tiled::time(&p).cycles;
                if best.as_ref().is_none_or(|(bc, _)| c < *bc) {
                    best = Some((c, p));
                }
            }
            Err(e) => why = format!("{a} x {b}: {e}"),
        }
    }
    best.map(|b| b.1).ok_or_else(|| {
        format!(
            "{}: grid reading `{reading}` places on no tile ({why})",
            trace.name
        )
    })
}

/// A top tile's price (cycles), the tile, its chunks and the program's fabric (w, h).
pub type TilePoint = (f64, (u32, u32), Chunks, (u32, u32));

/// Every top tile `reading` places on within `bounds`, each at the first of [`CHUNK_LADDER`]
/// whose every PE, as the combined program links it, leaves the stack's room past the estimate's
/// miss ([`crate::compile::check::miss`]): its price (cycles), tile, chunks and fabric (w, h). Where no
/// tile does, why, by its nearest miss.
pub fn points(
    trace: &Trace,
    reading: &str,
    clip: [u32; 3],
    round16: bool,
    bounds: (u32, u32),
) -> R<Vec<TilePoint>> {
    let base = lower(trace, reading, clip, round16)?;
    let (h, w) = base.top();
    let mut out = Vec::new();
    // the nearest miss: bytes short, and what fell short
    let mut nearest: Option<(i64, String)> = None;
    let mut a = 1;
    while a <= h.min(16) {
        let mut b = 1;
        while b <= w.min(16) {
            for &chunks in CHUNK_LADDER {
                let mut p = base.clone();
                if place_with(&mut p, a, b, Some(bounds), chunks).is_err() {
                    continue;
                }
                let Ok(e) = crate::compile::emit::tiled::emit(&p) else {
                    continue;
                };
                let files = super::coresident::gated_files(&e.files);
                let r = crate::compile::check::placed(&e.placed, &files, None);
                if r.ok().is_err() {
                    continue;
                }
                // each kind of PE's room past the stack's and its estimate's miss
                let Some((short, (file, d, c))) = r
                    .codes
                    .iter()
                    .map(|(f, d, c)| {
                        let text = files.iter().find(|(n, _)| n == f).map(|(_, t)| t.as_str());
                        let need = i64::from(crate::compile::build::STACK_ROOM)
                            + crate::compile::check::miss(f, text);
                        (
                            need - (crate::compile::check::PE_BYTES - d - c),
                            (f.clone(), *d, *c),
                        )
                    })
                    .max_by_key(|(s, _)| *s)
                else {
                    continue;
                };
                if short > 0 {
                    if nearest.as_ref().is_none_or(|(s, _)| short < *s) {
                        nearest = Some((
                            short,
                            format!(
                                "tile {a},{b} at {chunks:?}: `{file}` {d} B data + ~{c} B code, \
                                 {short} B short of the stack's {} B and the estimate's miss",
                                crate::compile::build::STACK_ROOM
                            ),
                        ));
                    }
                    continue;
                }
                out.push((
                    crate::compile::cost::tiled::time(&p).cycles,
                    (a, b),
                    chunks,
                    p.fabric(),
                ));
                break;
            }
            b *= 2;
        }
        a *= 2;
    }
    if out.is_empty() {
        return Err(format!(
            "{reading}: no top tile's PEs hold its program as the combined program links it{}",
            nearest.map_or_else(String::new, |(_, n)| format!(" (the nearest: {n})"))
        ));
    }
    Ok(out)
}

/// The chunkings a tile is tried at, in turn: the defaults, then less staging and block, then
/// the op table streamed and the buffers packed tightly, then less again (each a PE's memory for
/// a little more time: more passes and blocks).
pub const CHUNK_LADDER: &[Chunks] = &[
    Chunks {
        kc: KC,
        wb: WB_WORDS,
        stream_ops: false,
        tight: false,
    },
    Chunks {
        kc: KC / 2,
        wb: WB_WORDS / 2,
        stream_ops: false,
        tight: false,
    },
    Chunks {
        kc: KC,
        wb: WB_WORDS,
        stream_ops: true,
        tight: true,
    },
    Chunks {
        kc: KC / 2,
        wb: WB_WORDS / 2,
        stream_ops: true,
        tight: true,
    },
    Chunks {
        kc: KC / 4,
        wb: WB_WORDS / 2,
        stream_ops: true,
        tight: true,
    },
];

/// An order of the arena's first fit: a lifetime's (key, (first op, last op, words)) to its sort
/// key.
type Order = dyn Fn(&(Key, (usize, usize, u32))) -> (i64, i64, i64);

/// A value's rows as [`emulate`] reads them.
pub type Rows = crate::compile::stream::host::Rows;

/// The program run on the host as its kernels compute: each convolution's weights read off the
/// compile's own stream (its spans over the checkpoint's values, unpacked), its input rounded to
/// the build's 16-bit type, summed in f32 in the order a PE sums (taps, then input rows); the
/// norms' statistics in f64; the regions by `kernels_ir::interp`; the attention's keys and values
/// 16-bit. `get` gives a value's rows by id (a weight, or a host value); returns each output's.
#[allow(clippy::too_many_lines)]
pub fn emulate(p: &TileProgram, get: &dyn Fn(u32) -> R<Rows>) -> R<BTreeMap<u32, Vec<f32>>> {
    let b16 = |x: f32| -> f32 {
        if p.round16 {
            let b = x.to_bits();
            f32::from_bits(b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff_0000)
        } else {
            crate::compile::stream::host::half_f32(crate::compile::stream::host::half_bits(x))
        }
    };
    let h16 = |x: u32| -> f32 {
        if p.round16 {
            f32::from_bits(x << 16)
        } else {
            crate::compile::stream::host::half_f32(x as u16)
        }
    };
    let words = crate::compile::stream::host::apply_spans(&p.stream, get)?;
    let mut cur = 0usize; // the stream's word read next
    let mut vals: BTreeMap<u32, Vec<f32>> = BTreeMap::new();
    let mut scales: BTreeMap<usize, (Vec<f32>, Vec<f32>)> = BTreeMap::new();
    let wd = |v: ValueId| p.vals[&v].width as usize;
    for (i, op) in p.ops.iter().enumerate() {
        // (an op's words ahead of its records, where the op table streams)
        cur += p.op_words.get(i).map_or(0, |&n| n as usize / 2);
        match op {
            TOp::In(v) => {
                vals.insert(v.0, get(v.0)?.2.to_vec());
            }
            TOp::Out(_) => {}
            TOp::Conv {
                x,
                y,
                k,
                cin,
                cout,
                bias,
                ..
            } => {
                let [_, h, w] = p.vals[y].grid;
                let (h, w) = (h as usize, w as usize);
                let (cin, cout, k) = (*cin as usize, *cout as usize, *k as usize);
                let xs: Vec<f32> = vals[&x.0].iter().map(|&a| b16(a)).collect();
                let (kb, kc) = (p.kb[i] as usize, p.kc[i] as usize);
                let mut acc = vec![0f32; h * w * cout];
                for (ch, t) in (0..cin / kc).flat_map(|ch| (0..k * k).map(move |t| (ch, t))) {
                    let (dy, dx) = (
                        (t / k) as i64 - (k / 2) as i64,
                        (t % k) as i64 - (k / 2) as i64,
                    );
                    for b in (0..kc / kb).map(|b| ch * (kc / kb) + b) {
                        let n = kb * cout;
                        let blk: Vec<f32> = (0..n)
                            .map(|e| {
                                let wv = words[cur + e / 2];
                                h16(if e % 2 == 0 { wv & 0xffff } else { wv >> 16 })
                            })
                            .collect();
                        cur += n / 2;
                        for yy in 0..h {
                            for xx in 0..w {
                                let (sy, sx) = (yy as i64 + dy, xx as i64 + dx);
                                if sy < 0 || sx < 0 || sy >= h as i64 || sx >= w as i64 {
                                    continue;
                                }
                                let src = (sy as usize * w + sx as usize) * cin + b * kb;
                                let dst = (yy * w + xx) * cout;
                                for r in 0..kb {
                                    let xv = xs[src + r];
                                    for c in 0..cout {
                                        acc[dst + c] += blk[r * cout + c] * xv;
                                    }
                                }
                            }
                        }
                    }
                }
                if bias.is_some() {
                    let bs: Vec<f32> = (0..cout).map(|c| f32::from_bits(words[cur + c])).collect();
                    cur += cout;
                    for (e, a) in acc.iter_mut().enumerate() {
                        *a += bs[e % cout];
                    }
                }
                if p.vals[y].round {
                    for a in &mut acc {
                        *a = b16(*a);
                    }
                }
                vals.insert(y.0, acc);
            }
            TOp::Norm {
                x,
                groups,
                eps,
                reg,
                ..
            } => {
                let c = wd(*x);
                let xs = &vals[&x.0];
                let rows = xs.len() / c;
                let cpg = c / *groups as usize;
                let ga: Vec<f32> = (0..c).map(|e| f32::from_bits(words[cur + e])).collect();
                let be: Vec<f32> = (0..c).map(|e| f32::from_bits(words[cur + c + e])).collect();
                cur += 2 * c;
                let (mut a, mut b) = (vec![0f32; c], vec![0f32; c]);
                for q in (0..c).step_by(cpg) {
                    let n = (rows * cpg) as f64;
                    let it = || (0..rows).flat_map(move |t| (q..q + cpg).map(move |e| t * c + e));
                    let mean = it().map(|j| f64::from(xs[j])).sum::<f64>() / n;
                    let var = it().map(|j| (f64::from(xs[j]) - mean).powi(2)).sum::<f64>() / n;
                    let rs = 1.0 / (var + f64::from(*eps)).sqrt();
                    for e in q..q + cpg {
                        a[e] = (rs * f64::from(ga[e])) as f32;
                        b[e] = (f64::from(be[e]) - mean * rs * f64::from(ga[e])) as f32;
                    }
                }
                scales.insert(i, (a, b));
                let r = &p.regs[*reg];
                region_rows(p, r, &mut vals, &[], &scales, &eval_args)?;
            }
            TOp::Region(ri) => {
                let r = &p.regs[*ri];
                let mut res: Vec<Vec<f32>> = Vec::new();
                for input in &r.ins {
                    if let RegIn::Res(Res::Value { .. }) = input {
                        let w = r.region.width as usize;
                        res.push((0..w).map(|e| f32::from_bits(words[cur + e])).collect());
                        cur += w;
                    }
                }
                region_rows(p, r, &mut vals, &res, &scales, &eval_args)?;
            }
            TOp::Up { x, y, axis, sel } => {
                let c = wd(*x);
                let co = wd(*y);
                let [_, _, w] = p.vals[x].grid;
                let [_, h2, w2] = p.vals[y].grid;
                let xs = &vals[&x.0];
                let mut out = vec![0f32; (h2 * w2) as usize * co];
                for yy in 0..h2 {
                    for xx in 0..w2 {
                        let (sy, sx) = if *axis == 0 {
                            (yy / 2, xx)
                        } else {
                            (yy, xx / 2)
                        };
                        let a = (if *axis == 0 { yy } else { xx } % 2) as usize;
                        let s = ((sy * w + sx) as usize) * c;
                        let d = ((yy * w2 + xx) as usize) * co;
                        for e in 0..co {
                            out[d + e] = if *sel { xs[s + 2 * e + a] } else { xs[s + e] };
                        }
                    }
                }
                if p.vals[y].round && !p.vals[x].round {
                    for a in &mut out {
                        *a = b16(*a);
                    }
                }
                vals.insert(y.0, out);
            }
            TOp::Attend { q, k, v, o, scale } => {
                let c = wd(*q);
                let (qs, ks, vs) = (&vals[&q.0], &vals[&k.0], &vals[&v.0]);
                let rows = qs.len() / c;
                let k16: Vec<f32> = ks.iter().map(|&a| b16(a)).collect();
                let v16: Vec<f32> = vs.iter().map(|&a| b16(a)).collect();
                let mut out = vec![0f32; rows * c];
                for t in 0..rows {
                    let sc: Vec<f64> = (0..rows)
                        .map(|u| {
                            (0..c)
                                .map(|e| f64::from(qs[t * c + e]) * f64::from(k16[u * c + e]))
                                .sum::<f64>()
                                * f64::from(*scale)
                        })
                        .collect();
                    let m = sc.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let ex: Vec<f64> = sc.iter().map(|x| (x - m).exp()).collect();
                    let den: f64 = ex.iter().sum();
                    for e in 0..c {
                        let acc: f64 = (0..rows).map(|u| ex[u] * f64::from(v16[u * c + e])).sum();
                        out[t * c + e] = (acc / den) as f32;
                    }
                }
                if p.vals[o].round {
                    for a in &mut out {
                        *a = b16(*a);
                    }
                }
                vals.insert(o.0, out);
            }
        }
    }
    if cur != words.len() {
        return Err(format!("the stream's {} words, {cur} read", words.len()));
    }
    Ok(p.outs
        .iter()
        .map(|v| (v.0, vals.get(&v.0).cloned().unwrap_or_default()))
        .collect())
}

fn eval_args(r: &kernels_ir::Region, args: &[kernels_ir::interp::Arg]) -> R<Vec<Vec<f32>>> {
    kernels_ir::interp::eval(r, args).map_err(|e| format!("{e:?}"))
}

type Eval = dyn Fn(&kernels_ir::Region, &[kernels_ir::interp::Arg]) -> R<Vec<Vec<f32>>>;

/// Region `r` over every row of its inputs: its residents `res` (in order), a norm's scale and
/// shift from `scales`.
fn region_rows(
    p: &TileProgram,
    r: &TReg,
    vals: &mut BTreeMap<u32, Vec<f32>>,
    res: &[Vec<f32>],
    scales: &BTreeMap<usize, (Vec<f32>, Vec<f32>)>,
    ev: &Eval,
) -> R<()> {
    use kernels_ir::interp::Arg;
    let w = r.region.width as usize;
    let rows = r
        .ins
        .iter()
        .find_map(|i| match i {
            RegIn::Row(v) => Some(vals[&v.0].len() / w),
            RegIn::Res(_) => None,
        })
        .ok_or("a region of no row")?;
    let mut outs: Vec<Vec<f32>> = vec![Vec::new(); r.outs.len()];
    for t in 0..rows {
        let mut args = Vec::new();
        let mut ri = 0;
        for input in &r.ins {
            args.push(Arg::Row(match input {
                RegIn::Row(v) => vals[&v.0][t * w..(t + 1) * w].to_vec(),
                RegIn::Res(Res::Value { .. }) => {
                    ri += 1;
                    res[ri - 1].clone()
                }
                RegIn::Res(Res::GnScale(g)) => scales[g].0.clone(),
                RegIn::Res(Res::GnShift(g)) => scales[g].1.clone(),
            }));
        }
        let got = ev(&r.region, &args)?;
        for (o, row) in outs.iter_mut().zip(got) {
            o.extend(row);
        }
    }
    for (v, o) in r.outs.iter().zip(outs) {
        vals.insert(v.0, o);
    }
    let _ = p;
    Ok(())
}

/// Writes `p`'s program into `out`: its kernels and `layout.csl`, `stream.json` (its ops and
/// price), `host.json` and the arrays' files (as a rows program's, the rows read back a
/// rectangle at a time).
pub fn compile_program(p: &TileProgram, out: &std::path::Path) -> R<()> {
    use serde_json::json;
    let e = crate::compile::emit::tiled::emit(p)?;
    std::fs::create_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))?;
    for (name, text) in &e.files {
        std::fs::write(out.join(name), text).map_err(|e| format!("{name}: {e}"))?;
    }
    crate::compile::check::written(&e.placed, &e.files, None);
    std::fs::write(
        out.join("fabric.txt"),
        format!("{} {} {}\n", e.w, e.h, u8::from(p.round16)),
    )
    .map_err(|e| e.to_string())?;
    let c = crate::compile::cost::tiled::time(p);
    let meta = json!({
        "regime": "tiled",
        "fabric": [e.w, e.h],
        "array": [p.gx, p.gy],
        "ops": p.ops.len(),
        "arena": p.arena, "staging": p.hx, "block": p.wb,
        "stream_words": p.stream_words, "tape": p.tape,
        "clip": p.clip,
        "cycles": c.cycles,
    });
    std::fs::write(
        out.join("stream.json"),
        serde_json::to_string_pretty(&meta).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let (h2d, d2h) = grid::host_arrays(&e.arrays, out, true)?;
    let outs: Vec<serde_json::Value> = p
        .outs
        .iter()
        .map(|v| json!({"value": v.0, "rows": p.vals[v].grid[1] * p.vals[v].grid[2], "width": p.vals[v].width}))
        .collect();
    let host = json!({
        "model": p.trace.name, "reading": p.reading,
        "classes": [{"name": p.reading, "word": p.word.to_string(), "rows": 0}],
        "clip": p.clip,
        "outs": outs, "w": e.w, "h": e.h, "h2d": h2d, "d2h": d2h, "refs": [],
        "round16": p.round16, "causal": false,
        "units": [{"name": "image", "x": 1, "cols": p.gx, "rows": p.gy}],
    });
    std::fs::write(
        out.join("host.json"),
        serde_json::to_string_pretty(&host).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())
}

/// [`program_upto`] and [`compile_program`]: the reading compiled into `out`.
#[allow(clippy::too_many_arguments)]
pub fn compile(
    trace: &Trace,
    reading: &str,
    clip: [u32; 3],
    round16: bool,
    pin: Option<(u32, u32)>,
    bounds: Option<(u32, u32)>,
    upto: Option<usize>,
    out: &std::path::Path,
) -> R<TileProgram> {
    let p = program_upto(trace, reading, clip, round16, pin, bounds, upto)?;
    compile_program(&p, out)?;
    Ok(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trace(name: &str) -> Trace {
        poem_compiler::catalog::repository()
            .parse(name)
            .unwrap_or_else(|| panic!("{name} parses"))
            .trace(poem_ir::Platform::Cerebras)
    }

    /// flux2-mini's decode lowers into ops over the whole image: its latent's standardisation
    /// and pixel shuffle (two halves), its convolutions (the bias each one's own), its norms,
    /// its attention, the upsamplings an axis at a time, the pixels out.
    #[test]
    fn flux_2_minis_vae_decode_lowers_into_tiled_ops() {
        let p = lower(
            &trace("flux2-mini-bf16-kv-bf16"),
            "vae.decode",
            [1, 8, 8],
            true,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        let count = |f: &dyn Fn(&TOp) -> bool| p.ops.iter().filter(|o| f(o)).count();
        assert_eq!(count(&|o| matches!(o, TOp::In(_))), 1);
        assert_eq!(count(&|o| matches!(o, TOp::Conv { .. })), 30);
        assert_eq!(count(&|o| matches!(o, TOp::Norm { .. })), 22);
        assert_eq!(count(&|o| matches!(o, TOp::Attend { .. })), 1);
        // the shuffle's two halves, three upsamplings by two in y and x
        assert_eq!(count(&|o| matches!(o, TOp::Up { sel: true, .. })), 2);
        assert_eq!(count(&|o| matches!(o, TOp::Up { sel: false, .. })), 6);
        assert_eq!(p.outs.len(), 1);
        assert_eq!(p.top(), (64, 64));
    }

    /// klein-4B's decode at 1024² (latent 128²) places within a system's fabric: the search's
    /// array 256 x 512 (a 2 x 2 top tile), each PE's data within its budget.
    #[test]
    fn kleins_vae_decode_places_at_1024() {
        let t = trace("flux2-klein-4b-bf16-kv-bf16");
        let b = crate::compile::floorplan::stream::Bounds::system();
        let p = program(
            &t,
            "vae.decode",
            [1, 128, 128],
            true,
            None,
            Some((b.w, b.h)),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!((p.gx, p.gy), (256, 512));
        let (w, h) = p.fabric();
        assert!(w <= b.w && h <= b.h);
        assert!(p.arena + p.hx / 2 + p.wb <= DATA_WORDS);
    }
}

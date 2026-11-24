//! The GEMM regime: a whole-sequence reading (a DiT's blocks, a text encoder's layers) over the
//! whole fabric, layer by layer, every PE on the current op. The reading is lowered as the stream
//! regime lowers it ([`build::lower_units`]); only the placement differs: each unit's steps become
//! ops over an array of `C x R` PEs, column `c` a channel block and row `r` a token block. A
//! matvec is a GEMM (inputs along the row, weights streamed down the column), a root region runs
//! at every PE over its chunk with sums all-reduced along the row, and the attention streams keys
//! and values along the column with an online softmax.

use std::collections::{BTreeMap, BTreeSet};

use poem_ir::{Def, Dtype, Trace, ValueId};

use super::build::{self, Lowered};
use super::host::{BF16, CLAMP, FLAT, HALF, LIT, NONE, ONE, PACK, Span, Word};
use super::unit::{Cols, Reader, Step, Unit, unmoved};
use super::{Options, StreamShape};
use crate::compile::emit::region::{self, ParamAt, Place, Printed};

type R<T> = Result<T, String>;

/// How a region reads its parameter `k`.
#[derive(Clone, Debug, PartialEq)]
pub enum RArg {
    /// A row's chunk: columns `from ..` of `v`, the region's width.
    Row { v: ValueId, from: u32 },
    /// A chunk every row reads, made on the wafer (a lane vector): columns `from ..` of `v`'s
    /// one row.
    Lane { v: ValueId, from: u32 },
    /// A chunk every row reads, the checkpoint's (a norm's gain): `v`'s elements from `from`.
    Static { v: ValueId, from: u32 },
    /// One f32 a row, the host's (a position): `v`'s column `col` of the row.
    Each { v: ValueId, col: u32 },
    /// One f32 every row reads, the checkpoint's: `v`'s element `col`.
    One { v: ValueId, col: u32 },
}

/// A printed region over the array.
#[derive(Clone, Debug)]
pub struct GReg {
    pub region: kernels_ir::Region,
    pub args: Vec<RArg>,
    pub outs: Vec<ValueId>,
    pub params: Vec<ParamAt>,
    /// The classes whose rows run it (a bit a class); `lane`: a lane vector's one row.
    pub mask: u16,
    pub lane: bool,
    /// Placed: its print at the program's chunk, and its module.
    pub printed: Option<Printed>,
    pub module: usize,
    /// Placed: the columns of its row a PE holds; a head region (its row a head, run over every
    /// head, its sums the head's band's).
    pub chunk: u32,
    pub head: bool,
}

/// The joint attention's operands for one unit's rows.
#[derive(Clone, Debug, PartialEq)]
pub struct ASide {
    pub mask: u16,
    pub q: (ValueId, u32),
    pub k: (ValueId, u32),
    pub v: (ValueId, u32),
    pub o: ValueId,
}

/// One op over the array.
#[derive(Clone, Debug)]
pub enum GOp {
    /// The host's value written into its slot (a step's entry).
    In {
        v: ValueId,
        mask: u16,
        lane: bool,
    },
    /// `y = x[:, from .. from + k] W^T` (`w` is `[n, k]`).
    Gemm {
        node: u32,
        x: ValueId,
        from: u32,
        k: u32,
        w: ValueId,
        y: ValueId,
        n: u32,
        x32: bool,
        round: bool,
        mask: u16,
        lane: bool,
    },
    Region(usize),
    Attend {
        sides: Vec<ASide>,
        head_dim: u32,
        heads: u32,
        scale: f32,
        causal: bool,
        round: bool,
    },
    /// Columns `from .. from + width` of `v` read back by the host.
    Out {
        v: ValueId,
        from: u32,
        width: u32,
        mask: u16,
    },
}

/// A value of the program.
#[derive(Clone, Debug, PartialEq)]
pub struct GVal {
    pub width: u32,
    /// Every column a reader starts or ends at (0 and the width among them).
    pub cuts: BTreeSet<u32>,
    pub lane: bool,
    /// Rounded to the build's 16-bit type where made (held f32).
    pub round: bool,
    /// The classes whose rows hold it.
    pub mask: u16,
    /// Read only by GEMMs and the host (its pieces may be uneven).
    pub free: bool,
    /// The segments (by their first column) laid out other than in order.
    pub maps: BTreeMap<u32, SegMap>,
}

/// A segment laid out other than its columns in order: grouped-query attention's k and v a
/// copy a q head (`expand` q heads a kv head of `hd` columns), and a NeoX rotation's halves
/// side by side in each PE's chunk (`perm`, over blocks of its length: position `p`'s column).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SegMap {
    pub expand: u32,
    pub hd: u32,
    pub perm: Option<Vec<u32>>,
}

impl SegMap {
    /// The segment's own column (from its start) at layout position `p`.
    #[must_use]
    pub fn src(&self, p: u32) -> u32 {
        let u = match &self.perm {
            Some(m) => {
                let n = m.len() as u32;
                (p / n) * n + m[(p % n) as usize]
            }
            None => p,
        };
        if self.expand > 1 {
            (u / self.hd / self.expand) * self.hd + u % self.hd
        } else {
            u
        }
    }

    /// The layout's width of a segment of `w` columns.
    #[must_use]
    pub fn width(&self, w: u32) -> u32 {
        w * self.expand.max(1)
    }
}

impl GVal {
    /// Segment `from`'s layout width (its columns, or a copy a q head of each).
    #[must_use]
    pub fn lwidth(&self, from: u32, w: u32) -> u32 {
        self.maps.get(&from).map_or(w, |m| m.width(w))
    }

    /// Its segments: (from, width).
    #[must_use]
    pub fn segs(&self) -> Vec<(u32, u32)> {
        let c: Vec<u32> = self.cuts.iter().copied().collect();
        c.windows(2).map(|w| (w[0], w[1] - w[0])).collect()
    }
}

/// Columns `[lo, hi)` of `w` PE `i` of `n` holds (the floors of an even cut).
#[must_use]
pub fn piece(i: u32, n: u32, w: u32) -> (u32, u32) {
    let lo = (u64::from(i) * u64::from(w) / u64::from(n)) as u32;
    let hi = (u64::from(i + 1) * u64::from(w) / u64::from(n)) as u32;
    (lo, hi)
}

/// The lowered reading, before placement.
#[derive(Clone, Debug)]
pub struct Lowering {
    pub trace: Trace,
    pub reading: String,
    pub shape: StreamShape,
    pub class_rows: Vec<(u32, u32)>,
    pub vals: BTreeMap<ValueId, GVal>,
    pub ops: Vec<GOp>,
    pub regs: Vec<GReg>,
    pub round16: bool,
    /// Every value's alias (a merge of class arms: one slot).
    pub alias: BTreeMap<ValueId, ValueId>,
    /// Requests a launch: the step's rows `batch` requests' rows in turn (each its classes'
    /// rows, its lane vectors its own row), its attention its own rows'.
    pub batch: u32,
}

impl Lowering {
    /// A request's rows (every class's).
    #[must_use]
    pub fn request_rows(&self) -> u32 {
        self.class_rows.iter().map(|x| x.0).sum()
    }
}

/// A placed program: the lowering at `c x r` PEs.
#[derive(Clone, Debug)]
pub struct GemmProgram {
    pub l: Lowering,
    pub c: u32,
    pub r: u32,
    pub tb: u32,
    /// Each PE row's class and request.
    pub row_class: Vec<usize>,
    pub row_req: Vec<u32>,
    /// The printed modules (name, text).
    pub modules: Vec<(String, String)>,
    /// Arena offsets (f32 words) by value, its size.
    pub slots: BTreeMap<ValueId, u32>,
    pub arena: u32,
    /// A column's weight stream (column `c`'s spans) and its words.
    pub wstream: Vec<Vec<Span>>,
    /// Where each span of a column's stream starts (16-bit halves from the column's start).
    pub wat: Vec<Vec<u64>>,
    /// Where each op's words start in column 0's stream (32-bit words: its own words, then its
    /// residents or blocks), the stream's end last: a profile's op by where the tape has got to.
    pub wop: Vec<u64>,
    pub wwords: u32,
    /// Each column's stream words (an uneven output's columns differ).
    pub wcol: Vec<u32>,
    /// The tapes' rows below the array: a column's stream over its tape PEs,
    /// [`crate::compile::emit::gemm::tape_words`] a PE, each forwarding the words of the PEs past it.
    pub tape: u32,
    /// Key-block members of the attention (`kbm x tb` keys a block).
    pub kbm: u32,
    /// The attention's band (columns a head) and q's chunk.
    pub band: u32,
    pub qchunk: u32,
    /// Buffer sizes (words): a weight block, an input piece (16-bit halves), the residents.
    pub wbn: u32,
    pub xbn: u32,
    /// Some weight block holds an odd count of 16-bit weights (each padded to whole words).
    pub odd: bool,
}

fn refuse(t: &Trace, what: impl std::fmt::Display) -> String {
    format!("{}: the GEMM regime: {what}", t.name)
}

struct Lw<'t> {
    t: &'t Trace,
    vals: BTreeMap<ValueId, GVal>,
    ops: Vec<GOp>,
    regs: Vec<GReg>,
    alias: BTreeMap<ValueId, ValueId>,
    entered: BTreeSet<ValueId>,
    nclasses: usize,
}

impl Lw<'_> {
    fn canon(&self, v: ValueId) -> ValueId {
        let v = unmoved(v);
        *self.alias.get(&v).unwrap_or(&v)
    }

    fn width(&self, v: ValueId) -> u32 {
        match self.t.values.get(v.0 as usize).map(|d| &d.ty) {
            Some(poem_ir::Ty::Tensor { shape, .. }) => match shape.last() {
                Some(poem_ir::Dim::Const(n)) => *n as u32,
                _ => 0,
            },
            _ => 0,
        }
    }

    fn round(&self, v: ValueId) -> bool {
        matches!(
            self.t.values.get(v.0 as usize).map(|d| &d.ty),
            Some(poem_ir::Ty::Tensor {
                dtype: Dtype::Bf16 | Dtype::F16,
                ..
            })
        )
    }

    /// `v` made (or entered) on `mask`'s rows.
    fn make(&mut self, v: ValueId, mask: u16, lane: bool) {
        let v = self.canon(v);
        let width = self.width(v);
        let round = self.round(v);
        let e = self.vals.entry(v).or_insert_with(|| GVal {
            width,
            cuts: [0, width].into_iter().collect(),
            lane,
            round,
            mask: 0,
            free: true,
            maps: BTreeMap::new(),
        });
        e.mask |= mask;
        e.lane |= lane;
    }

    /// A read of `width` columns of `v` from `from`; `free`: by a GEMM or the host.
    fn read(&mut self, v: ValueId, from: u32, width: u32, free: bool) -> R<()> {
        let v = self.canon(v);
        let t = self.t;
        let e = self
            .vals
            .get_mut(&v)
            .ok_or_else(|| refuse(t, format!("value %{} read before it is made", v.0)))?;
        e.cuts.insert(from);
        e.cuts.insert(from + width);
        e.free &= free;
        Ok(())
    }

    fn mask_of(u: &Unit) -> u16 {
        u.classes.iter().fold(0, |m, &c| m | (1 << c))
    }

    fn all(&self) -> u16 {
        ((1u32 << self.nclasses) - 1) as u16
    }

    #[allow(clippy::too_many_lines)]
    fn step(&mut self, r: &Reader, u: &Unit, s: &Step, joint: &mut Vec<ASide>) -> R<()> {
        let t = self.t;
        let lane = u.classes.is_empty();
        let mask = if lane { self.all() } else { Self::mask_of(u) };
        match s {
            Step::In { value, host, .. } => {
                if *host {
                    let v = self.canon(*value);
                    if self.entered.insert(v) {
                        self.make(v, mask, lane);
                        self.ops.push(GOp::In { v, mask, lane });
                    } else if let Some(e) = self.vals.get_mut(&v) {
                        e.mask |= mask;
                    }
                }
                Ok(())
            }
            Step::Region { chain, .. } => {
                // a root kernel's step runs its own module, which the GEMM PE has no form of
                if crate::kernel::root_of(r.trace, chain).is_some() {
                    let n = chain.nodes[0] as usize;
                    return Err(format!(
                        "{} has no GEMM form",
                        poem_ir::Operands::name(&r.trace.nodes[n].op)
                    ));
                }
                let mut args = Vec::new();
                let mut params = Vec::new();
                // a head region (a head's ops, its row a head) runs over every head of the
                // values it reads: its rows' reads `reps` heads wide
                let reps = chain
                    .outs
                    .first()
                    .map_or(1, |o| self.width(o.value) / chain.region.width.max(1))
                    .max(1);
                let w = chain.region.width * reps;
                for (k, input) in chain.ins.iter().enumerate() {
                    let base = self.canon(r.cols(input.value).base);
                    let from = r.cols(input.value).from + input.from;
                    let scalar = chain.region.params[k] == kernels_ir::Shape::Scalar;
                    let weight = matches!(
                        t.values.get(base.0 as usize).map(|d| &d.def),
                        Some(Def::Weight(_))
                    );
                    let made = self.vals.get(&base).is_some_and(|x| x.lane);
                    let (a, at) = if scalar {
                        if input.each {
                            (RArg::Each { v: base, col: from }, ParamAt::Scalar)
                        } else if weight {
                            (RArg::One { v: base, col: from }, ParamAt::Scalar)
                        } else {
                            return Err(refuse(
                                t,
                                format!(
                                    "{}: node {}: a scalar parameter of no row and no weight",
                                    u.name, chain.nodes[0]
                                ),
                            ));
                        }
                    } else if input.shared
                        || input.by.is_some()
                        || (!lane && (r.lane(input.value) || made))
                    {
                        if weight {
                            (RArg::Static { v: base, from }, ParamAt::Resident)
                        } else {
                            self.read(base, from, w, false)?;
                            (RArg::Lane { v: base, from }, ParamAt::Resident)
                        }
                    } else {
                        if !self.vals.contains_key(&base) {
                            // a host row read where it is (an entry's table)
                            if self.entered.insert(base) {
                                self.make(base, mask, lane);
                                self.ops.push(GOp::In {
                                    v: base,
                                    mask,
                                    lane,
                                });
                            }
                        }
                        self.read(base, from, w, false)?;
                        (RArg::Row { v: base, from }, ParamAt::Row)
                    };
                    args.push(a);
                    params.push(at);
                }
                let mut outs = Vec::new();
                for o in &chain.outs {
                    let v = self.canon(o.value);
                    self.make(v, mask, lane);
                    if let Some(x) = self.vals.get_mut(&v) {
                        x.free = false;
                    }
                    outs.push(v);
                }
                let ri = self.regs.len();
                self.regs.push(GReg {
                    region: chain.region.clone(),
                    args,
                    outs,
                    params,
                    mask,
                    lane,
                    printed: None,
                    module: 0,
                    chunk: 0,
                    head: false,
                });
                self.ops.push(GOp::Region(ri));
                Ok(())
            }
            Step::Matvec {
                node,
                input,
                weight,
                output,
                k,
                n,
                x32,
                round,
                ..
            } => {
                let x = self.canon(input.base);
                if weight.0 == super::host::UNIT {
                    // an identity matvec moves a value across a unit: here it is where it is
                    let y = unmoved(*output);
                    if self.canon(y) != x {
                        self.alias.insert(y, x);
                    }
                    return Ok(());
                }
                self.read(x, input.from, *k, true)?;
                let y = self.canon(*output);
                self.make(y, mask, lane);
                self.ops.push(GOp::Gemm {
                    node: *node,
                    x,
                    from: input.from,
                    k: *k,
                    w: *weight,
                    y,
                    n: *n,
                    x32: *x32,
                    round: *round,
                    mask,
                    lane,
                });
                Ok(())
            }
            Step::Store { q, k, v } => {
                let mut c = |x: &Cols| -> R<(ValueId, u32)> {
                    let b = self.canon(x.base);
                    self.read(b, x.from, x.width, false)?;
                    Ok((b, x.from))
                };
                let (q, k, v) = (c(q)?, c(k)?, c(v)?);
                joint.push(ASide {
                    mask,
                    q,
                    k,
                    v,
                    o: ValueId(u32::MAX),
                });
                Ok(())
            }
            Step::Attend { out, .. } => {
                let o = self.canon(*out);
                self.make(o, mask, lane);
                if let Some(j) = joint.iter_mut().rev().find(|j| j.mask == mask) {
                    j.o = o;
                }
                Ok(())
            }
            Step::Out {
                value, host, lane, ..
            } => {
                if *host && !*lane {
                    let b = self.canon(value.base);
                    self.read(b, value.from, value.width, true)?;
                    self.ops.push(GOp::Out {
                        v: b,
                        from: value.from,
                        width: value.width,
                        mask,
                    });
                }
                Ok(())
            }
        }
    }
}

/// `trace`'s stream reading `o` lowered into ops over the array (no placement yet).
pub fn lower(trace: &Trace, o: &Options, class_rows: &[(String, u32)]) -> R<Lowering> {
    let s = super::shape(
        trace,
        &o.reading,
        &super::with_rows(
            poem_compiler::spatial::layer::stream::classes(trace, &o.reading, o.rows),
            class_rows,
        ),
    )?;
    let Lowered {
        class_rows: crows,
        units,
        blocks,
        ..
    } = build::lower_units(trace, &s.program, o.upto, &o.taps)?;
    let nclasses = s.program.classes.len();
    if nclasses > 16 {
        return Err(refuse(trace, "more than 16 row classes"));
    }
    // a merge of the classes' arms is one value
    let mut alias = BTreeMap::new();
    for (i, d) in trace.values.iter().enumerate() {
        if let Def::Merge(arms) = &d.def {
            let rep = arms.first().map_or(ValueId(i as u32), |a| a.0);
            alias.insert(ValueId(i as u32), rep);
            for (a, _) in arms {
                if *a != rep {
                    alias.insert(*a, rep);
                }
            }
        }
    }
    let mut lw = Lw {
        t: trace,
        vals: BTreeMap::new(),
        ops: Vec::new(),
        regs: Vec::new(),
        alias,
        entered: BTreeSet::new(),
        nclasses,
    };
    let in_block: BTreeMap<usize, usize> = blocks
        .iter()
        .enumerate()
        .flat_map(|(b, us)| us.iter().map(move |&u| (u, b)))
        .collect();
    let mut i = 0;
    while i < units.len() {
        let group: Vec<usize> = match in_block.get(&i) {
            Some(&b) => blocks[b].clone(),
            None => vec![i],
        };
        let mut joint = Vec::new();
        let readers: Vec<Reader> = group
            .iter()
            .map(|&g| Reader {
                trace,
                word: units[g].1,
                prologue: units[g].0.classes.is_empty(),
            })
            .collect();
        for (gi, &g) in group.iter().enumerate() {
            let u = &units[g].0;
            for s in &u.steps[..u.split] {
                lw.step(&readers[gi], u, s, &mut joint)?;
            }
        }
        // each unit's attention output (its first step after the barrier)
        for &g in &group {
            let u = &units[g].0;
            let mask = Lw::mask_of(u);
            for st in &u.steps[u.split..] {
                if let Step::Attend { out, .. } = st {
                    let o = lw.canon(*out);
                    lw.make(o, mask, false);
                    if let Some(j) = joint.iter_mut().rev().find(|j| j.mask == mask) {
                        j.o = o;
                    }
                }
            }
        }
        if !joint.is_empty() {
            let j = units[group[0]]
                .0
                .joint
                .ok_or_else(|| refuse(trace, "a block's stores and no attention"))?;
            if joint.iter().any(|s| s.o.0 == u32::MAX) {
                return Err(refuse(
                    trace,
                    "a unit stores rows its attention never reads",
                ));
            }
            let round = joint
                .iter()
                .any(|s| lw.vals.get(&s.o).is_some_and(|v| v.round));
            lw.ops.push(GOp::Attend {
                sides: joint.clone(),
                head_dim: j.head_dim,
                heads: j.q_heads,
                scale: j.scale,
                causal: j.causal,
                round,
            });
        }
        for (gi, &g) in group.iter().enumerate() {
            let u = &units[g].0;
            for s in &u.steps[u.split..] {
                lw.step(&readers[gi], u, s, &mut joint)?;
            }
        }
        i = group.iter().max().map_or(i, |m| m + 1);
    }
    // grouped-query attention: k and v laid out a copy a q head (every head's band then holds
    // its q, k and v alike), and every region over them so (a head region runs over the copies)
    if s.attn.kv_heads > 0 && s.attn.q_heads > s.attn.kv_heads {
        let (g, hd) = (s.attn.q_heads / s.attn.kv_heads, s.attn.head_dim);
        if g * s.attn.kv_heads != s.attn.q_heads {
            return Err(refuse(
                trace,
                "q heads that are no multiple of the kv heads",
            ));
        }
        let m = SegMap {
            expand: g,
            hd,
            perm: None,
        };
        let mut seed: Vec<(ValueId, u32)> = Vec::new();
        for op in &lw.ops {
            if let GOp::Attend { sides, .. } = op {
                for x in sides {
                    seed.push(x.k);
                    seed.push(x.v);
                }
            }
        }
        for (v, f) in seed {
            if let Some(x) = lw.vals.get_mut(&v) {
                x.maps.insert(f, m.clone());
            }
        }
        loop {
            let mut more = Vec::new();
            for g2 in &lw.regs {
                let rows: Vec<(ValueId, u32)> = g2
                    .args
                    .iter()
                    .filter_map(|a| match a {
                        RArg::Row { v, from } => Some((*v, *from)),
                        _ => None,
                    })
                    .chain(g2.outs.iter().map(|o| (*o, 0)))
                    .collect();
                let any = rows
                    .iter()
                    .any(|(v, f)| lw.vals.get(v).is_some_and(|x| x.maps.contains_key(f)));
                if any {
                    more.extend(
                        rows.into_iter().filter(|(v, f)| {
                            lw.vals.get(v).is_some_and(|x| !x.maps.contains_key(f))
                        }),
                    );
                }
            }
            if more.is_empty() {
                break;
            }
            for (v, f) in more {
                if let Some(x) = lw.vals.get_mut(&v) {
                    x.maps.insert(f, m.clone());
                }
            }
        }
    }
    Ok(Lowering {
        trace: trace.clone(),
        reading: o.reading.clone(),
        shape: s,
        class_rows: crows,
        vals: lw.vals,
        ops: lw.ops,
        regs: lw.regs,
        round16: o.round16,
        alias: lw.alias,
        batch: 1,
    })
}

/// What a GEMM reads and makes, by op (for lifetimes).
fn op_io(l: &Lowering, op: &GOp) -> (Vec<ValueId>, Vec<ValueId>) {
    match op {
        GOp::In { v, .. } => (vec![], vec![*v]),
        GOp::Gemm { x, y, .. } => (vec![*x], vec![*y]),
        GOp::Region(r) => {
            let g = &l.regs[*r];
            let ins = g
                .args
                .iter()
                .filter_map(|a| match a {
                    RArg::Row { v, .. } | RArg::Lane { v, .. } => Some(*v),
                    _ => None,
                })
                .collect();
            (ins, g.outs.clone())
        }
        GOp::Attend { sides, .. } => (
            sides.iter().flat_map(|s| [s.q.0, s.k.0, s.v.0]).collect(),
            sides.iter().map(|s| s.o).collect(),
        ),
        GOp::Out { v, .. } => (vec![*v], vec![]),
    }
}

/// The chunk a region runs at: its width over the columns.
fn chunk_of(width: u32, c: u32) -> Option<u32> {
    (width.is_multiple_of(c) && width >= c).then(|| width / c)
}

/// A PE's local columns of `v` (its pieces of each segment, side by side) at column `ci`:
/// (segment from, segment width, local offset, piece lo, piece len).
#[must_use]
pub fn local(v: &GVal, c: u32, ci: u32) -> Vec<(u32, u32, u32, u32, u32)> {
    let mut off = 0;
    v.segs()
        .into_iter()
        .map(|(f, w)| {
            let w = v.lwidth(f, w);
            let (lo, hi) = piece(ci, c, w);
            let r = (f, w, off, lo, hi - lo);
            off += hi - lo;
            r
        })
        .collect()
}

/// `v`'s local width at column `ci`.
#[must_use]
pub fn local_width(v: &GVal, c: u32, ci: u32) -> u32 {
    local(v, c, ci).iter().map(|x| x.4).sum()
}

/// Runs of consecutive numbers: (first, length).
fn runs(xs: impl Iterator<Item = u32>) -> Vec<(u32, u32)> {
    let mut out: Vec<(u32, u32)> = Vec::new();
    for x in xs {
        match out.last_mut() {
            Some(r) if r.0 + r.1 == x => r.1 += 1,
            _ => out.push((x, 1)),
        }
    }
    out
}

/// The local offset of the segment of `v` starting at `from` (every column alike: even
/// values), or an error.
fn seg_off(t: &Trace, vid: ValueId, v: &GVal, c: u32, from: u32, width: u32) -> R<u32> {
    let segs = local(v, c, 0);
    let s = segs.iter().find(|s| s.0 == from).ok_or_else(|| {
        refuse(
            t,
            format!("value %{}: no segment starts at column {from}", vid.0),
        )
    })?;
    if s.1 != width {
        return Err(refuse(
            t,
            format!(
                "value %{}: columns {from} .. {} span segments (one is {} wide)",
                vid.0,
                from + width,
                s.1
            ),
        ));
    }
    Ok(s.2)
}

impl GemmProgram {
    /// The fabric's size: the array, the tapes below it.
    #[must_use]
    pub fn fabric(&self) -> (u32, u32) {
        (self.c, self.r + self.tape)
    }

    /// The step's rows: every request's, all classes.
    #[must_use]
    pub fn step_rows(&self) -> u32 {
        self.l.request_rows() * self.l.batch.max(1)
    }
}

/// The 16-bit flags of a weight in this build.
fn w16(round16: bool) -> u32 {
    if round16 { BF16 } else { HALF }
}

/// Why [`placed`] refuses a shape, by what the search does next: a refusal tied to the columns
/// ends that column count's candidates; one of a cut that does not divide evenly is the reason
/// given last (a reason past it, the shape's own, before it).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// The rows do not cut evenly over the rows of PEs (or a class over the blocks).
    Rows(String),
    /// A value's segment does not cut evenly over the columns.
    Segment(String),
    /// An attention's q does not cut over the columns.
    Q(String),
    /// A region does not print at the columns' chunk.
    Region(String),
    /// Any other.
    Other(String),
    /// A line's window open ([`crate::compile::check::hazard::lines`]): the program would hang; the
    /// reason given before any other.
    Window(String),
}

impl Refusal {
    /// Its message.
    #[must_use]
    pub fn text(&self) -> &str {
        match self {
            Refusal::Rows(m)
            | Refusal::Segment(m)
            | Refusal::Q(m)
            | Refusal::Region(m)
            | Refusal::Other(m)
            | Refusal::Window(m) => m,
        }
    }

    /// No row count places at these columns.
    fn ends_columns(&self) -> bool {
        matches!(
            self,
            Refusal::Segment(_) | Refusal::Q(_) | Refusal::Region(_)
        )
    }

    /// A cut that does not divide evenly (the least telling reason).
    fn uneven(&self) -> bool {
        matches!(self, Refusal::Segment(_) | Refusal::Rows(_))
    }
}

impl From<String> for Refusal {
    fn from(m: String) -> Self {
        Refusal::Other(m)
    }
}

impl From<&str> for Refusal {
    fn from(m: &str) -> Self {
        Refusal::Other(m.to_string())
    }
}

impl From<Refusal> for String {
    fn from(r: Refusal) -> Self {
        match r {
            Refusal::Rows(m)
            | Refusal::Segment(m)
            | Refusal::Q(m)
            | Refusal::Region(m)
            | Refusal::Other(m)
            | Refusal::Window(m) => m,
        }
    }
}

/// `l` placed at `c` columns and `r` rows of PEs with `kbm` key-block members, its weight
/// streams written, or why not.
pub fn place(l: &Lowering, c: u32, r: u32, kbm: u32) -> R<GemmProgram> {
    let mut p = placed(l, c, r, kbm).map_err(String::from)?;
    streams(&mut p)?;
    Ok(p)
}

/// [`place`] but the weight streams (their words counted, a search's candidate). Refused where
/// a line's queue is left open to a colour it is not bound to ([`crate::compile::check::hazard::lines`]).
pub fn placed(l: &Lowering, c: u32, r: u32, kbm: u32) -> Result<GemmProgram, Refusal> {
    let p = laid(l, c, r, kbm)?;
    let open = crate::compile::check::hazard::lines(&p, crate::compile::emit::gemm::GPE_OPS);
    if let Some(w) = open.first() {
        return Err(Refusal::Window(refuse(
            &l.trace,
            format!(
                "{w}{}",
                if open.len() > 1 {
                    format!(" (and {} more)", open.len() - 1)
                } else {
                    String::new()
                }
            ),
        )));
    }
    Ok(p)
}

/// [`placed`] but the line windows' check.
#[allow(clippy::too_many_lines)]
pub(crate) fn laid(l: &Lowering, c: u32, r: u32, kbm: u32) -> Result<GemmProgram, Refusal> {
    let t = &l.trace;
    let per = l.request_rows();
    let total = per * l.batch.max(1);
    if c == 0 || r == 0 || !total.is_multiple_of(r) {
        return Err(Refusal::Rows(refuse(
            t,
            format!("{total} rows over {r} rows of PEs"),
        )));
    }
    let tb = total / r;
    // (a PE row's rows one request's, one class's)
    let (mut row_class, mut row_req) = (Vec::new(), Vec::new());
    for b in 0..l.batch.max(1) {
        for (ci, &(rows, _)) in l.class_rows.iter().enumerate() {
            if !rows.is_multiple_of(tb) {
                return Err(Refusal::Rows(refuse(
                    t,
                    format!("class {ci}'s {rows} rows over blocks of {tb}"),
                )));
            }
            row_class.extend(std::iter::repeat_n(ci, (rows / tb) as usize));
            row_req.extend(std::iter::repeat_n(b, (rows / tb) as usize));
        }
    }
    // (a request's rows of PEs)
    let rq = r / l.batch.max(1);
    // every value read by a region or attention cuts evenly
    for (vid, v) in &l.vals {
        if v.free {
            continue;
        }
        for (f, w) in v.segs() {
            let w = v.lwidth(f, w);
            if !w.is_multiple_of(c) {
                return Err(Refusal::Segment(refuse(
                    t,
                    format!("value %{}: a segment of {w} over {c} columns", vid.0),
                )));
            }
        }
    }
    // the attention's band: a head's columns over the PEs of a row
    let (mut band, mut qchunk) = (1, 1);
    for op in &l.ops {
        if let GOp::Attend {
            sides, head_dim, ..
        } = op
        {
            for s in sides {
                let qv = &l.vals[&s.q.0];
                let qw = qv
                    .segs()
                    .iter()
                    .find(|x| x.0 == s.q.1)
                    .map_or(0, |x| qv.lwidth(x.0, x.1));
                let qc = chunk_of(qw, c)
                    .ok_or_else(|| Refusal::Q(refuse(t, format!("q of {qw} columns over {c}"))))?;
                if !head_dim.is_multiple_of(qc) {
                    return Err(Refusal::Other(refuse(
                        t,
                        format!("heads of {head_dim} over chunks of {qc} columns"),
                    )));
                }
                band = head_dim / qc;
                qchunk = qc;
            }
            // (a key block's keys are the dim-major tile's stride: a DSD stride is 8 bits)
            if !rq.is_multiple_of(kbm) || !(kbm * tb).is_multiple_of(band) || kbm * tb > 127 {
                return Err(Refusal::Other(refuse(
                    t,
                    format!(
                        "key blocks of {kbm} rows of PEs over {rq} ({} keys over a band of {band})",
                        kbm * tb
                    ),
                )));
            }
        }
    }
    // the regions printed at their chunk: a region's span (the layout width of what its rows
    // read) over the columns, its row the region's own width (a head region's a head, run over
    // every head: its sums over the head's band); a rotation whose mates another PE holds has its
    // halves gathered into each chunk (the columns permuted, and every row segment it touches)
    let top = l.class_rows.iter().map(|x| x.0).max().unwrap_or(1).max(2) - 1;
    let lv = (32 - top.leading_zeros()).div_ceil(3);
    let mut l2 = l.clone();
    let rows_of = |g: &GReg| -> Vec<(ValueId, u32)> {
        g.args
            .iter()
            .filter_map(|a| match a {
                RArg::Row { v, from } => Some((*v, *from)),
                _ => None,
            })
            .chain(g.outs.iter().map(|o| (*o, 0)))
            .collect()
    };
    let span_of = |vals: &BTreeMap<ValueId, GVal>, g: &GReg| -> u32 {
        rows_of(g)
            .first()
            .and_then(|(v, f)| {
                let x = vals.get(v)?;
                x.segs()
                    .into_iter()
                    .find(|s| s.0 == *f)
                    .map(|s| x.lwidth(s.0, s.1))
            })
            .unwrap_or(g.region.width)
    };
    let place_of = |g: &GReg, span: u32, columns: Option<Vec<u32>>| -> Result<Place, Refusal> {
        let ch = chunk_of(span, c)
            .filter(|ch| g.region.width.is_multiple_of(*ch))
            .ok_or_else(|| {
                Refusal::Region(refuse(
                    t,
                    format!("a region of {} over {span} / {c}", g.region.width),
                ))
            })?;
        Ok(Place {
            chunk: ch,
            roots: g.region.width / ch,
            params: g.params.clone(),
            round16: l.round16,
            rope_lv: lv.max(1),
            rope_bits: 3,
            straddle: true,
            columns,
            dup: 1,
        })
    };
    // the rotations' layouts first (a region's rows alike: a reader of a permuted row prints
    // permuted too), to a fixed point
    loop {
        let mut changed = false;
        for ri in 0..l2.regs.len() {
            let g = &l2.regs[ri];
            let rows = rows_of(g);
            let mut perms: Vec<Vec<u32>> = rows
                .iter()
                .filter_map(|(v, f)| l2.vals[v].maps.get(f).and_then(|m| m.perm.clone()))
                .collect();
            perms.dedup();
            if perms.len() > 1 {
                return Err(Refusal::Region(refuse(
                    t,
                    format!("region {ri}: rows of two layouts"),
                )));
            }
            let span = span_of(&l2.vals, g);
            let perm = match perms.pop() {
                Some(p) => Some(p),
                None if region::print(&g.region, &place_of(g, span, None)?).is_err() => {
                    // the mates gathered into each chunk: chunk k the row's pairs k ch / 2 ..
                    // in order, their firsts then their seconds
                    let h = (span / c / 2) as usize;
                    region::mates(&g.region).first().and_then(|m| {
                        let pairs: Vec<(u32, u32)> = (0..m.len() as u32)
                            .filter(|&i| m[i as usize] > i)
                            .map(|i| (i, m[i as usize]))
                            .collect();
                        (h > 0 && 2 * pairs.len() == m.len()).then(|| {
                            pairs
                                .chunks(h)
                                .flat_map(|b| b.iter().map(|x| x.0).chain(b.iter().map(|x| x.1)))
                                .collect()
                        })
                    })
                }
                None => None,
            };
            if let Some(perm) = perm {
                for (v, f) in rows {
                    let x = l2.vals.get_mut(&v).ok_or("a region's value")?;
                    let e = x.maps.entry(f).or_default();
                    if e.perm.as_ref() != Some(&perm) {
                        e.perm = Some(perm.clone());
                        e.hd = e.hd.max(perm.len() as u32);
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    let mut modules: Vec<(String, String)> = Vec::new();
    for ri in 0..l2.regs.len() {
        let g = l2.regs[ri].clone();
        let span = span_of(&l2.vals, &g);
        let head = span > g.region.width;
        let held = rows_of(&g)
            .iter()
            .find_map(|(v, f)| l2.vals[v].maps.get(f).and_then(|m| m.perm.clone()));
        let mut pr = region::print(&g.region, &place_of(&g, span, held)?)
            .map_err(|e| Refusal::Region(refuse(t, format!("region {ri}: {e}"))))?;
        if head {
            // a head region's row sums are its band's
            if pr.head_reduces > 0 || (pr.all_reduces > 0 && g.region.width / (span / c) != band) {
                return Err(Refusal::Region(refuse(
                    t,
                    format!("region {ri}: a head region's sums over other than its band"),
                )));
            }
            pr.csl = pr.csl.replace("seg.all_reduce(", "seg.head_reduce(");
        }
        let m = match modules.iter().position(|(_, s)| *s == pr.csl) {
            Some(m) => m,
            None => {
                modules.push((format!("greg{}.csl", modules.len()), pr.csl.clone()));
                modules.len() - 1
            }
        };
        let gm = &mut l2.regs[ri];
        gm.chunk = span / c;
        gm.head = head;
        gm.printed = Some(pr);
        gm.module = m;
    }
    // what the host writes and reads, and a GEMM's input, are in the columns' order; a head's
    // q and k alike
    for op in &l2.ops {
        let mapped = |v: &ValueId| l2.vals.get(v).is_some_and(|x| !x.maps.is_empty());
        match op {
            GOp::In { v, .. } | GOp::Out { v, .. } if mapped(v) => {
                return Err(Refusal::Other(refuse(
                    t,
                    format!("value %{}: the host's, laid out otherwise", v.0),
                )));
            }
            GOp::Gemm { x, .. } if mapped(x) => {
                return Err(Refusal::Other(refuse(
                    t,
                    format!("value %{}: a GEMM's input laid out otherwise", x.0),
                )));
            }
            GOp::Attend { sides, .. } => {
                for sd in sides {
                    let perm = |(v, f): (ValueId, u32)| {
                        l2.vals[&v].maps.get(&f).and_then(|m| m.perm.clone())
                    };
                    if perm(sd.q) != perm(sd.k) || perm(sd.v).is_some() {
                        return Err(Refusal::Other(refuse(
                            t,
                            "an attention's q and k laid out apart",
                        )));
                    }
                }
            }
            _ => {}
        }
    }
    // slots: each value's words a PE, first fit over the ops' intervals
    let mut first: BTreeMap<ValueId, usize> = BTreeMap::new();
    let mut last: BTreeMap<ValueId, usize> = BTreeMap::new();
    for (i, op) in l2.ops.iter().enumerate() {
        let (ins, outs) = op_io(&l2, op);
        for v in outs {
            first.entry(v).or_insert(i);
            last.entry(v).or_insert(i);
        }
        for v in ins {
            first.entry(v).or_insert(i);
            *last.entry(v).or_default() = i;
        }
    }
    let size = |v: &GVal| -> u32 {
        let rows = if v.lane { 1 } else { tb };
        rows * (0..c).map(|ci| local_width(v, c, ci)).max().unwrap_or(0)
    };
    let mut order: Vec<ValueId> = first.keys().copied().collect();
    order.sort_by_key(|v| (first[v], v.0));
    let mut placed: Vec<(u32, u32, usize, usize)> = Vec::new();
    let mut slots = BTreeMap::new();
    let mut arena = 0u32;
    for v in order {
        let n = size(&l2.vals[&v]).max(1).next_multiple_of(2);
        let (a, b) = (first[&v], last[&v]);
        let mut off = 0u32;
        loop {
            let clash = placed
                .iter()
                .find(|&&(o, s, x, y)| off < o + s && o < off + n && a <= y && x <= b);
            match clash {
                Some(&(o, s, _, _)) => off = o + s,
                None => break,
            }
        }
        placed.push((off, n, a, b));
        slots.insert(v, off);
        arena = arena.max(off + n);
    }
    // each column's weight-stream words and the buffers' sizes, counted (the streams
    // themselves are written for the program a search keeps: [`streams`])
    let mut wcol = vec![0u64; c as usize];
    let (mut wbn, mut xbn) = (2u32, 2u32);
    for op in &l2.ops {
        if let GOp::Region(ri) = op {
            let g = &l2.regs[*ri];
            for a in &g.args {
                let add = match a {
                    RArg::Static { .. } => u64::from(g.chunk),
                    RArg::One { .. } => 1,
                    _ => 0,
                };
                for w in &mut wcol {
                    *w += add;
                }
            }
            continue;
        }
        let GOp::Gemm {
            x,
            from,
            k,
            y,
            x32,
            lane,
            ..
        } = op
        else {
            continue;
        };
        let yv = &l2.vals[y];
        let xv = &l2.vals[x];
        if !xv.free || xv.segs().len() > 1 {
            seg_off(t, *x, xv, c, *from, *k)?;
        }
        let ysegs: Vec<(u32, u32)> = yv
            .segs()
            .into_iter()
            .map(|(f, w)| (f, yv.lwidth(f, w)))
            .collect();
        // an output of one segment read only by GEMMs and the host may cut unevenly
        let un = ysegs.len() == 1 && yv.free && !ysegs[0].1.is_multiple_of(c);
        for &(_, sw) in &ysegs {
            if !sw.is_multiple_of(c) && !un {
                return Err(Refusal::Segment(refuse(
                    t,
                    format!("value %{}: a GEMM's output segment of {sw} over {c}", y.0),
                )));
            }
        }
        let rows = if *lane { 1 } else { tb };
        // the input's pieces over the columns: the widest
        let pieces: Vec<u32> = (0..c)
            .map(|cp| {
                let (k0, k1) = piece(cp, c, *k);
                k1 - k0
            })
            .collect();
        let kmax = pieces.iter().copied().max().unwrap_or(0);
        if kmax == 0 {
            continue;
        }
        xbn = xbn.max(rows * kmax * if *x32 { 2 } else { 1 });
        for &(_, sw) in &ysegs {
            for (ci, wc) in wcol.iter_mut().enumerate() {
                let (lo, hi) = piece(ci as u32, c, sw);
                let p = hi - lo;
                // (a block of an odd count of 16-bit weights is padded to whole words)
                wbn = wbn.max((kmax * p).div_ceil(2));
                *wc += pieces
                    .iter()
                    .map(|&kc| u64::from((kc * p).div_ceil(2)))
                    .sum::<u64>();
            }
        }
    }
    let _ = (NONE, ONE, CLAMP);
    let mut p = GemmProgram {
        l: l2,
        c,
        r,
        tb,
        row_class,
        row_req,
        modules,
        slots,
        arena,
        wstream: Vec::new(),
        wat: Vec::new(),
        wop: Vec::new(),
        wwords: 0,
        wcol: Vec::new(),
        tape: 0,
        kbm,
        band,
        qchunk,
        wbn,
        xbn,
        odd: false,
    };
    // the program streams in with its weights: each op's words (16-bit, an even count) where its
    // mark is, in every column's stream
    let words = crate::compile::emit::gemm::op_words(&p)?;
    let own: u64 = words.iter().map(|w| w.len().div_ceil(2) as u64).sum();
    for wc in &mut wcol {
        *wc += own;
    }
    let wwords = wcol.iter().copied().max().unwrap_or(0);
    if wwords > u64::from(u32::MAX) {
        return Err(Refusal::Other(refuse(
            t,
            "a column's weight stream past 2^32 words",
        )));
    }
    p.wwords = wwords as u32;
    p.wcol = wcol.iter().map(|&w| w as u32).collect();
    p.tape = (wwords.div_ceil(u64::from(crate::compile::emit::gemm::tape_words())) as u32).max(1);
    p.wbn = p
        .wbn
        .max(words.iter().map(Vec::len).max().unwrap_or(2).div_ceil(2) as u32);
    Ok(p)
}

/// The columns' weight streams of `p` (placed by [`placed`]): a column's blocks a GEMM, an
/// input piece and an output segment each, each op's own words where its mark is; their words
/// the ones [`placed`] counted.
#[allow(clippy::too_many_lines)]
pub fn streams(p: &mut GemmProgram) -> R<()> {
    let (l2, c, tb) = (&p.l, p.c, p.tb);
    let t = &l2.trace;
    let rows_of = |g: &GReg| -> Vec<(ValueId, u32)> {
        g.args
            .iter()
            .filter_map(|a| match a {
                RArg::Row { v, from } => Some((*v, *from)),
                _ => None,
            })
            .chain(g.outs.iter().map(|o| (*o, 0)))
            .collect()
    };
    // the weight stream: a column's blocks a GEMM, an input piece and an output segment each
    let f16 = w16(l2.round16) | PACK;
    let mut wstream: Vec<Vec<Span>> = vec![Vec::new(); c as usize];
    let mut wcol = vec![0u64; c as usize];
    let (mut wbn, mut xbn) = (2u32, 2u32);
    let mut any_odd = false;
    for (oi, op) in l2.ops.iter().enumerate() {
        // the op's own words first (a mark, its words once the program is placed)
        for ws in &mut wstream {
            ws.push([OPMARK, oi as u32, 0, 0, 0, 0, 0, 0]);
        }
        // a region's static residents (a norm's gain) come down the stream too, f32, the PE's
        // chunk each
        if let GOp::Region(ri) = op {
            let g = &l2.regs[*ri];
            let ch = g.chunk;
            let roots = g.region.width / ch;
            // (its columns in the rows' layout: a rotation's halves gathered)
            let perm = rows_of(g)
                .iter()
                .find_map(|(v, f)| l2.vals[v].maps.get(f).and_then(|m| m.perm.clone()));
            for a in &g.args {
                for (ci, ws) in wstream.iter_mut().enumerate() {
                    match a {
                        RArg::Static { v, from } => {
                            let p0 = (ci as u32 % roots) * ch;
                            let src = |p: u32| perm.as_ref().map_or(p, |m| m[p as usize]);
                            for (a0, n) in runs((p0..p0 + ch).map(src)) {
                                ws.push([v.0, 0, from + a0, n, 1, FLAT, 1, 0]);
                            }
                            wcol[ci] += u64::from(ch);
                        }
                        RArg::One { v, col } => {
                            ws.push([v.0, 0, *col, 1, 1, FLAT, 1, 0]);
                            wcol[ci] += 1;
                        }
                        _ => {}
                    }
                }
            }
            continue;
        }
        let GOp::Gemm {
            x,
            from,
            k,
            w,
            y,
            n,
            x32,
            lane,
            ..
        } = op
        else {
            continue;
        };
        let yv = &l2.vals[y];
        let xv = &l2.vals[x];
        if !xv.free || xv.segs().len() > 1 {
            seg_off(t, *x, xv, c, *from, *k)?;
        }
        let ysegs: Vec<(u32, u32)> = yv
            .segs()
            .into_iter()
            .map(|(f, w)| (f, yv.lwidth(f, w)))
            .collect();
        // an output of one segment read only by GEMMs and the host may cut unevenly
        let un = ysegs.len() == 1 && yv.free && !ysegs[0].1.is_multiple_of(c);
        for &(_, sw) in &ysegs {
            if !sw.is_multiple_of(c) && !un {
                return Err(refuse(
                    t,
                    format!("value %{}: a GEMM's output segment of {sw} over {c}", y.0),
                ));
            }
        }
        let rows = if *lane { 1 } else { tb };
        for cp in 0..c {
            let (k0, k1) = piece(cp, c, *k);
            let kc = k1 - k0;
            if kc == 0 {
                continue;
            }
            xbn = xbn.max(rows * kc * if *x32 { 2 } else { 1 });
            for &(sf, sw) in &ysegs {
                for (ci, ws) in wstream.iter_mut().enumerate() {
                    let (lo, hi) = piece(ci as u32, c, sw);
                    let p = hi - lo;
                    // (a block of an odd count of 16-bit weights is padded to whole words)
                    let odd = (kc * p) % 2 != 0;
                    any_odd |= odd && p > 0;
                    wbn = wbn.max((kc * p).div_ceil(2));
                    wcol[ci] += u64::from((kc * p).div_ceil(2));
                    match yv.maps.get(&sf) {
                        _ if p == 0 => {}
                        None => ws.push([w.0, sf + lo, k0, p, *k, f16, kc, 1]),
                        // (an output laid out otherwise: its weights' rows in that order, a run
                        // of them a span, an input column's runs after another's)
                        Some(m) => {
                            let rs = runs((lo..hi).map(|q| m.src(q)));
                            for j in k0..k1 {
                                for &(a0, n) in &rs {
                                    ws.push([w.0, sf + a0, j, n, *k, f16, 1, 0]);
                                }
                            }
                        }
                    }
                    if odd && p > 0 {
                        ws.push([NONE, 0, 0, 1, 0, f16, 1, 0]);
                    }
                }
            }
        }
        let _ = n;
    }

    // merge the spans that run on (a GEMM of one output segment: its input pieces' blocks)
    for ws in &mut wstream {
        let mut m: Vec<Span> = Vec::with_capacity(ws.len());
        for s in ws.drain(..) {
            if let Some(p) = m.last_mut()
                && p[0] == s[0]
                && p[1] == s[1]
                && p[3] == s[3]
                && p[4] == s[4]
                && p[5] == s[5]
                && p[7] == 1
                && s[7] == 1
                && p[2] + p[6] == s[2]
            {
                p[6] += s[6];
                continue;
            }
            m.push(s);
        }
        *ws = m;
    }
    let words = crate::compile::emit::gemm::op_words(p)?;
    // (column 0's span at which each op starts)
    let mut op_span: Vec<usize> = Vec::new();
    for (ci, (ws, wc)) in wstream.iter_mut().zip(wcol.iter_mut()).enumerate() {
        let mut out = Vec::with_capacity(ws.len() * 2);
        for s in ws.drain(..) {
            if s[0] != OPMARK {
                out.push(s);
                continue;
            }
            if ci == 0 {
                op_span.push(out.len());
            }
            let w = &words[s[1] as usize];
            for &x in w {
                out.push([LIT, x, 0, 1, 0, PACK, 1, 0]);
            }
            if w.len() % 2 == 1 {
                out.push([LIT, 0, 0, 1, 0, PACK, 1, 0]);
            }
            *wc += w.len().div_ceil(2) as u64;
        }
        *ws = out;
    }
    if wcol.iter().zip(&p.wcol).any(|(a, b)| *a != u64::from(*b)) {
        return Err(refuse(
            t,
            "the weight streams' words are not the ones counted",
        ));
    }
    p.odd = any_odd;
    p.wat = wstream.iter().map(|ws| starts(ws)).collect();
    p.wop = op_span
        .iter()
        .map(|&i| p.wat[0].get(i).copied().unwrap_or(0) / 2)
        .chain(std::iter::once(u64::from(p.wcol[0])))
        .collect();
    p.wstream = wstream;
    Ok(())
}

/// Where each of a column's spans starts (16-bit halves: a packed element a half, an f32 a
/// word, starting on a whole word).
fn starts(ws: &[Span]) -> Vec<u64> {
    let mut at = 0u64;
    ws.iter()
        .map(|s| {
            let h = if s[5] & PACK != 0 { 1u64 } else { 2 };
            if h == 2 {
                at = at.next_multiple_of(2);
            }
            let a = at;
            at += u64::from(s[3]) * u64::from(s[6]) * h;
            a
        })
        .collect()
}

/// A placeholder in a column's stream for op `row`'s words.
const OPMARK: u32 = u32::MAX - 3;

/// The words tape PE `ri` of column `ci` holds: its run of the column's stream, padded to
/// [`crate::compile::emit::gemm::tape_words`].
#[must_use]
pub fn tape_spans(p: &GemmProgram, ci: u32, ri: u32) -> Vec<Span> {
    let per = crate::compile::emit::gemm::tape_words();
    let lo = 2 * u64::from(ri) * u64::from(per);
    let hi = (lo + 2 * u64::from(per)).min(2 * u64::from(p.wcol[ci as usize]));
    let mut out: Vec<Span> = Vec::new();
    let (ws, at) = (&p.wstream[ci as usize], &p.wat[ci as usize]);
    // (the spans from the last one starting at or before `lo`, to the first past `hi`)
    let first = at.partition_point(|&a| a <= lo).saturating_sub(1);
    for (s, &a) in ws.iter().zip(at).skip(first) {
        let [v, row, col, len, stride, flags, count, step] = *s;
        let h = if flags & PACK != 0 { 1u64 } else { 2 };
        let b = a + u64::from(len) * u64::from(count) * h;
        if a >= hi {
            break;
        }
        if b <= lo {
            continue;
        }
        // cut at whole elements of the run: (j, i) with element j * len + i
        let (from, to) = ((lo.max(a) - a) / h, (hi.min(b) - a) / h);
        let mut e = from;
        while e < to {
            let (j, i) = ((e / u64::from(len)) as u32, (e % u64::from(len)) as u32);
            if i == 0 && to - e >= u64::from(len) {
                // whole rows of the run from j
                let rows = ((to - e) / u64::from(len)) as u32;
                out.push([v, row, col + j * step, len, stride, flags, rows, step]);
                e += u64::from(rows) * u64::from(len);
            } else {
                let take = (u64::from(len - i)).min(to - e) as u32;
                // (a weight's run steps down its rows; a flat one along its elements)
                let (r0, c0) = if flags & FLAT != 0 {
                    (row, col + j * step + i * stride)
                } else {
                    (row + i, col + j * step)
                };
                out.push([v, r0, c0, take, stride, flags, 1, step]);
                e += u64::from(take);
            }
        }
    }
    let used = super::host::span_words(&out);
    if used < per {
        out.push([NONE, 0, 0, per - used, 0, 0, 1, 0]);
    }
    out
}

/// The search: every (columns, rows of PEs, key block) whose program places within `bounds`
/// and fits a PE, the cost model's cheapest (or the pinned one).
pub fn program(
    trace: &Trace,
    o: &Options,
    class_rows: &[(String, u32)],
    pin: Option<(u32, u32, u32)>,
    bounds: (u32, u32),
    batch: u32,
) -> R<GemmProgram> {
    let (all, why) = candidates(trace, o, class_rows, pin, bounds, batch)?;
    let mut p = all
        .into_iter()
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, p)| p)
        .ok_or_else(|| {
            if why.is_empty() {
                refuse(trace, "no placement")
            } else {
                why
            }
        })?;
    streams(&mut p)?;
    Ok(p)
}

/// Every (columns, rows of PEs, key block) at `batch` requests a launch whose program places
/// within `bounds` and fits a PE (or the pinned one), with its price (cycles a step), its weight
/// streams not yet written; and why the last refused one was refused.
pub fn candidates(
    trace: &Trace,
    o: &Options,
    class_rows: &[(String, u32)],
    pin: Option<(u32, u32, u32)>,
    bounds: (u32, u32),
    batch: u32,
) -> R<(Vec<(f64, GemmProgram)>, String)> {
    let mut l = lower(trace, o, class_rows)?;
    l.batch = batch.max(1);
    let total = l.request_rows() * l.batch;
    let mut cands: Vec<(u32, u32, u32)> = Vec::new();
    match pin {
        Some(p) => cands.push(p),
        None => {
            for c in 1..=bounds.0 {
                for r in 1..=bounds.1.min(total) {
                    if !total.is_multiple_of(r) {
                        continue;
                    }
                    for kbm in [1, 2, 4, 8, 16, 32, 64] {
                        if r.is_multiple_of(kbm) {
                            cands.push((c, r, kbm));
                        }
                    }
                }
            }
        }
    }
    let mut all: Vec<(f64, GemmProgram)> = Vec::new();
    // the reason given: its message, and whether it is a cut that does not divide evenly
    let mut why: Option<(String, bool)> = None;
    let mut window: Option<String> = None;
    // the columns first: a candidate's regions print at its chunk
    let mut ok_c: BTreeMap<u32, bool> = BTreeMap::new();
    for (c, r, kbm) in cands {
        if ok_c.get(&c) == Some(&false) {
            continue;
        }
        match placed(&l, c, r, kbm) {
            Ok(p) if p.fabric().1 > bounds.1 => {
                ok_c.insert(c, true);
                why = Some((
                    format!("{c} x {r}: {} rows of tapes past the fabric", p.tape),
                    false,
                ));
            }
            Ok(p) => {
                ok_c.insert(c, true);
                let mem = crate::compile::cost::gemm::memory(&p);
                if mem.total > crate::compile::cost::gemm::PE_WORDS {
                    if why.as_ref().is_none_or(|w| w.1) {
                        why = Some((
                            format!(
                                "{c} x {r}: {} words a PE (of {}): {mem:?}",
                                mem.total,
                                crate::compile::cost::gemm::PE_WORDS
                            ),
                            false,
                        ));
                    }
                    continue;
                }
                // (its array PE's constant DSDs: past a PE's low data it cannot link)
                let lo = crate::compile::emit::gemm::pe_files(&p).map_or(0, |f| {
                    crate::compile::check::lo_dsds(&f, "gpe.csl", &BTreeMap::new())
                });
                if lo > crate::compile::check::LO_DSDS {
                    if why.as_ref().is_none_or(|w| w.1) {
                        why = Some((
                            format!(
                                "{c} x {r}: ~{lo} constant DSD values a PE (of {})",
                                crate::compile::check::LO_DSDS
                            ),
                            false,
                        ));
                    }
                    continue;
                }
                all.push((crate::compile::cost::gemm::time(&p).cycles, p));
            }
            Err(e) => {
                if e.ends_columns() {
                    ok_c.insert(c, false);
                }
                // (a line's window first: no other cut mends it; then the reason a cut that
                // divides evenly fails, before one that does not)
                if let Refusal::Window(m) = &e {
                    window = Some(m.clone());
                }
                if why.as_ref().is_none_or(|w| w.1 && !e.uneven()) {
                    let uneven = e.uneven();
                    why = Some((e.into(), uneven));
                }
            }
        }
    }
    let why = window.or(why.map(|w| w.0)).unwrap_or_default();
    Ok((all, why))
}

/// The batch of at most `most` requests a launch whose program serves a request cheapest (its
/// step's price over its requests; the fewer requests on a tie), and that program.
pub fn batched(
    trace: &Trace,
    o: &Options,
    class_rows: &[(String, u32)],
    bounds: (u32, u32),
    most: u32,
) -> R<GemmProgram> {
    let mut best: Option<(f64, GemmProgram)> = None;
    let mut why = String::new();
    for b in 1..=most.max(1) {
        match program(trace, o, class_rows, None, bounds, b) {
            Ok(p) => {
                let each = crate::compile::cost::gemm::time(&p).cycles / f64::from(b);
                if best.as_ref().is_none_or(|(c, _)| each < *c) {
                    best = Some((each, p));
                }
            }
            Err(e) => why = e,
        }
    }
    best.map(|(_, p)| p).ok_or(why)
}

/// Compiles `p` into `out`: its CSL and its host's files (`host.json`: the classes, the arrays'
/// spans and gathers, the values read back), as the other regimes' readings.
pub fn compile_program(
    p: &GemmProgram,
    out: &std::path::Path,
    values: Option<&std::path::Path>,
) -> R<()> {
    use serde_json::json;
    let e = crate::compile::emit::gemm::emit(p)?;
    std::fs::create_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))?;
    for (name, text) in &e.files {
        std::fs::write(out.join(name), text).map_err(|e| format!("{name}: {e}"))?;
    }
    crate::compile::check::written(&e.placed, &e.files, None);
    std::fs::write(
        out.join("fabric.txt"),
        format!("{} {} {}\n", e.w, e.h, u8::from(p.l.round16)),
    )
    .map_err(|e| e.to_string())?;
    let c = crate::compile::cost::gemm::time(p);
    let m = crate::compile::cost::gemm::memory(p);
    let meta = json!({
        "regime": "gemm",
        "fabric": [e.w, e.h],
        "array": [p.c, p.r], "tb": p.tb, "kbm": p.kbm, "band": p.band,
        "ops": p.l.ops.len(),
        "modules": p.modules.len(),
        "stream_words": p.wwords, "tape_rows": p.tape,
        "memory": m,
        "cycles": c.cycles,
        "by": c.by(),
    });
    std::fs::write(
        out.join("stream.json"),
        serde_json::to_string_pretty(&meta).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let (mut h2d, d2h) = super::grid::host_arrays(&e.arrays, out, true)?;
    // (the step's entries, which the PEs read and never write: what the device holds is what
    // was last written, a step's write its changed words alone)
    for item in h2d.iter_mut() {
        if item["entry"].is_string() && matches!(item["sym"].as_str(), Some("hin" | "hsc")) {
            item["kept"] = serde_json::json!(true);
        }
    }
    // given a step's capture, every static array's words on file too (a runner with no
    // checkpoint writes them)
    if let Some(dir) = values {
        let vals = super::host::Values::open(dir)?;
        for (k, item) in h2d.iter_mut().enumerate() {
            // (an entry: its gather over the capture's rows)
            if let Some(f) = item["entry"].as_str() {
                let (shape, g) = crate::compile::npy::read_f32_or_i32(&out.join(f))?;
                let words: Vec<Word> = g
                    .chunks(4)
                    .map(|x| {
                        [
                            x[0].to_bits(),
                            x[1].to_bits(),
                            x[2].to_bits(),
                            x[3].to_bits(),
                        ]
                    })
                    .collect();
                let data = super::host::apply(&words, &|id| {
                    let (s, d) = vals.get(ValueId(id))?;
                    let w = *s.last().unwrap_or(&1);
                    Ok((d.len() / w.max(1), w, d))
                })?;
                let file = format!("h2d/{k}.npy");
                crate::compile::npy::write_f32(
                    &out.join(&file),
                    &shape[..shape.len().saturating_sub(1)],
                    &data,
                )?;
                item["file"] = json!(file);
                continue;
            }
            let Some(f) = item["static"].as_str() else {
                continue;
            };
            let (_, sp) = crate::compile::npy::read_f32_or_i32(&out.join(f))?;
            let spans: Vec<Span> = sp
                .chunks(8)
                .map(|c| {
                    let mut s = [0u32; 8];
                    for (i, x) in c.iter().enumerate() {
                        s[i] = x.to_bits();
                    }
                    s
                })
                .collect();
            let words = super::host::apply_spans(&spans, &|id| {
                let (s, d) = vals.get(ValueId(id))?;
                let w = *s.last().unwrap_or(&1);
                Ok((d.len() / w.max(1), w, d))
            })?;
            let (h, w, len) = (
                item["h"].as_u64().unwrap_or(1) as usize,
                item["w"].as_u64().unwrap_or(1) as usize,
                item["len"].as_u64().unwrap_or(1) as usize,
            );
            let file = format!("h2d/s{k}.npy");
            let data: Vec<f32> = words.iter().map(|&x| f32::from_bits(x)).collect();
            crate::compile::npy::write_f32(&out.join(&file), &[h, w, len], &data)?;
            item["file"] = json!(file);
        }
    }
    let mut outs: BTreeMap<u32, u32> = BTreeMap::new();
    for op in &p.l.ops {
        if let GOp::Out { v, from, width, .. } = op {
            let w = outs.entry(v.0).or_default();
            *w = (*w).max(from + width);
        }
    }
    let outs: Vec<serde_json::Value> = outs
        .iter()
        .map(|(v, w)| json!({"value": v, "rows": p.step_rows(), "width": w}))
        .collect();
    let classes: Vec<serde_json::Value> =
        p.l.shape
            .program
            .classes
            .iter()
            .map(|c| json!({"name": c.name, "word": c.word.to_string(), "rows": c.rows}))
            .collect();
    let host = json!({
        "model": p.l.trace.name, "reading": p.l.reading, "classes": classes, "outs": outs.clone(),
        "w": e.w, "h": e.h, "h2d": h2d, "d2h": d2h, "refs": [],
        "round16": p.l.round16, "causal": p.l.shape.attn.causal, "batch": p.l.batch,
        "units": [{"name": "array", "x": 0, "cols": p.c, "rows": p.r}],
    });
    let mut host = host;
    if let Some(dir) = values {
        let gold = std::fs::canonicalize(dir).unwrap_or_else(|_| dir.to_path_buf());
        host["gold"] = json!(gold.display().to_string());
        // the capture's values the rows read back are held to
        let vals = super::host::Values::open(dir)?;
        let refs: Vec<serde_json::Value> = outs
            .iter()
            .filter_map(|o| {
                let v = o["value"].as_u64()? as u32;
                Some(json!({"value": v, "file": vals.file(ValueId(v))?}))
            })
            .collect();
        host["refs"] = json!(refs);
    }
    std::fs::write(
        out.join("host.json"),
        serde_json::to_string_pretty(&host).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())
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

    /// flux2-mini's denoise lowers into ops over the array, every matmul a GEMM, the joint
    /// attentions one op a block, and places at a head-aligned cut.
    #[test]
    fn flux_2_minis_denoise_lowers_and_places() {
        let o = Options {
            reading: "denoise".into(),
            rows: 16,
            round16: false,
            upto: None,
            taps: Vec::new(),
            anneal: Default::default(),
        };
        let t = trace("flux2-mini-bf16-kv-bf16");
        let l = lower(&t, &o, &[]).unwrap_or_else(|e| panic!("{e}"));
        let gemms = l
            .ops
            .iter()
            .filter(|o| matches!(o, GOp::Gemm { .. }))
            .count();
        let attn = l
            .ops
            .iter()
            .filter(|o| matches!(o, GOp::Attend { .. }))
            .count();
        assert_eq!(attn, 4);
        assert!(gemms > 20, "{gemms} GEMMs");
        let p = place(&l, 32, 8, 8).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!((p.tb, p.band, p.qchunk), (4, 16, 8));
        // the q / k norms' heads (128) straddle the chunks (8): each head's sums all-reduced
        // over its band; the printer refuses them where a regime does not ask for it
        let g =
            p.l.regs
                .iter()
                .find(|g| g.printed.as_ref().is_some_and(|x| x.head_reduces > 0))
                .expect("a region of straddling heads");
        let off = region::print(
            &g.region,
            &Place {
                chunk: 8,
                roots: 32,
                params: g.params.clone(),
                round16: false,
                rope_lv: 4,
                rope_bits: 3,
                straddle: false,
                columns: None,
                dup: 1,
            },
        );
        assert!(off.is_err());
        // every PE's memory within its 48 KB, the step priced
        assert!(
            crate::compile::cost::gemm::memory(&p).total <= crate::compile::cost::gemm::PE_WORDS
        );
        assert!(crate::compile::cost::gemm::time(&p).cycles > 0.0);
    }

    /// A placement past a PE's low data (`data.lo`, 128 DSD descriptors) does not link, so the
    /// search refuses it, pinned or not: flux2-mini's denoise at 16 x 128 prints its regions at a
    /// 16-column chunk and spells more constant DSDs than fit.
    #[test]
    fn a_placement_past_a_pes_low_data_is_refused() {
        let o = Options {
            reading: "denoise".into(),
            rows: 16,
            round16: true,
            upto: None,
            taps: Vec::new(),
            anneal: Default::default(),
        };
        let t = trace("flux2-mini-bf16-kv-bf16");
        let at = |c: u32| candidates(&t, &o, &[], Some((c, 128, 32)), (762, 1172), 4);
        let (wide, _) = at(32).unwrap_or_else(|e| panic!("{e}"));
        let p = &wide.first().expect("32 x 128 places").1;
        let files = crate::compile::emit::gemm::pe_files(p).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            crate::compile::check::lo_dsds(&files, "gpe.csl", &BTreeMap::new()),
            126
        );
        let (narrow, why) = at(16).unwrap_or_else(|e| panic!("{e}"));
        assert!(narrow.is_empty(), "16 x 128 refused");
        assert!(why.contains("constant DSD values"), "{why}");
    }
}

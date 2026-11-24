//! The root region: a kernels-ir [`Region`] (a root phase's fused ops) printed as a CSL module a
//! root PE runs on its chunk of every row, the row cut across `roots` roots in order. Values are
//! classed as **static** (no parameter: evaluated by the host into [`Table`]s), **rows** (an f32
//! buffer each, an op one DSD op where it can), **scalars** (one f32 a row) and **heads**. A row
//! reduction across roots is a segment all-reduce (ops/seg.csl); cos / sin of an angle come from
//! rotation tables (ops/srope.csl). Anything else is refused by name ([`Refused`]).

use std::collections::BTreeMap;
use std::fmt::Write as _;

use kernels_ir::interp::{self, Arg};
use kernels_ir::{Binary, Elem, Literal, Op, Reduce, Region, Shape, Unary, ValueId};

/// How a parameter reaches the root.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParamAt {
    /// A row's chunk, every row (a stream row).
    Row,
    /// A chunk the root holds for every row (a norm's weight, a lane vector).
    Resident,
    /// One f32 a row (a position).
    Scalar,
}

/// Where the region runs.
#[derive(Clone, Debug)]
pub struct Place {
    pub chunk: u32,
    pub roots: u32,
    pub params: Vec<ParamAt>,
    /// Land the region's roundings in the build's 16-bit type (a bf16 build).
    pub round16: bool,
    /// The rotation tables' digits: positions below `2^(lv bits)`.
    pub rope_lv: u32,
    pub rope_bits: u32,
    /// A head may be wider than the chunk (cut over whole chunks of consecutive roots): its sums
    /// all-reduced over those roots (`seg.head_reduce`). Otherwise such a head is refused.
    pub straddle: bool,
    /// The row's columns in the order the roots hold them (`columns[p]` the region's column at
    /// position `p`; `None`: in order): a rotation's mates gathered into one root's chunk.
    pub columns: Option<Vec<u32>>,
    /// Consecutive roots that run the same chunk (a replicated K / V head): a row's all-reduce sums
    /// each column `dup` times, so a root's part is taken `1 / dup` (a power of two: exact).
    pub dup: u32,
}

/// A decode rotation's anchor period (`srope.csl`'s STEP): positions between two DIRECT
/// computations advance by one f32 rotation each, adding at most [`ROPE_STEP_ERR`] (~4.83 u)
/// a step. The budget is a sixteenth of fp16's half ulp (`2^-15`), so drift never moves a
/// rounded value by more than a sixteenth of its rounding (`rope_anchor_drift_in_budget`).
pub const ROPE_ANCHOR: u32 = rope_anchor();
pub const ROPE_STEP_ERR: f64 = (2.0 * std::f64::consts::SQRT_2 + 2.0) / 16_777_216.0;
pub const ROPE_DIRECT_ERR: f64 = 1e-6;
pub const ROPE_BUDGET: f64 = 1.0 / 32_768.0;

/// The largest power of two `N` with `ROPE_DIRECT_ERR + (N - 1) ROPE_STEP_ERR <= ROPE_BUDGET`.
const fn rope_anchor() -> u32 {
    let mut n = 1u32;
    while ROPE_DIRECT_ERR + (2 * n - 1) as f64 * ROPE_STEP_ERR <= ROPE_BUDGET {
        n *= 2;
    }
    n
}

/// Most runs a static select is printed as (DSD moves, a run each), else its two mask tables
/// and three operations over the chunk.
const SELECT_RUNS: usize = 32;

/// A DSD operand as `operand` prints it (a buffer, a parameter row, a table), not a scalar.
fn is_dsd(x: &str) -> bool {
    ["B", "IN", "T_"].iter().any(|p| {
        x.strip_prefix(p).is_some_and(|r| {
            !r.is_empty() && r.chars().all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
        })
    })
}

/// Most DSD moves a rotation's cos / sin gather is printed as (else a scalar loop over the
/// slot table).
const SLOT_RUNS: usize = 8;

/// A rotation's slot gather (`b[e] = rc[slot[e]]`) as runs `(e, j, n)`: columns `e .. e + n`
/// from slots `j .. j + n`, the same for every chunk, at most [`SLOT_RUNS`] of them.
fn slot_runs(slot_of: &[Vec<f32>]) -> Option<Vec<(usize, usize, usize)>> {
    let first = slot_of.first()?;
    if slot_of.iter().any(|m| m != first) {
        return None;
    }
    let mut runs: Vec<(usize, usize, usize)> = Vec::new();
    for (e, &j) in first.iter().enumerate() {
        let j = j as usize;
        match runs.last_mut() {
            Some((e0, j0, n)) if *e0 + *n == e && *j0 + *n == j => *n += 1,
            _ => runs.push((e, j, 1)),
        }
        if runs.len() > SLOT_RUNS {
            return None;
        }
    }
    Some(runs)
}

/// The mates `region`'s static gathers name, each a row over the region's columns (a rotation's
/// pair mate, a NeoX head's other half).
#[must_use]
pub fn mates(region: &Region) -> Vec<Vec<u32>> {
    let ats: Vec<ValueId> = region
        .ops
        .iter()
        .filter_map(|o| match *o {
            Op::Peer(_, at) => Some(at),
            _ => None,
        })
        .collect();
    if ats.is_empty() {
        return Vec::new();
    }
    let mut probe = region.clone();
    probe.results = ats;
    probe.rounds.clear();
    let args: Vec<Arg> = (region.params.iter())
        .map(|s| match s {
            Shape::Scalar => Arg::Scalar(0.0),
            _ => Arg::Row(vec![0.0; region.width as usize]),
        })
        .collect();
    interp::eval(&probe, &args)
        .map(|rows| {
            rows.into_iter()
                .map(|r| r.into_iter().map(|x| x as u32).collect())
                .collect()
        })
        .unwrap_or_default()
}

/// A static table the host writes into each root: `of_chunk[r]` root `r`'s words.
#[derive(Clone, Debug, PartialEq)]
pub struct Table {
    pub name: String,
    pub u16: bool,
    pub of_chunk: Vec<Vec<f32>>,
}

/// The printed module and what its runtime and host fill.
#[derive(Clone, Debug)]
pub struct Printed {
    pub csl: String,
    /// Parameter `k`'s array or scalar in the module: `in<k>` ([C]f32) or `sc<k>` (f32).
    pub params: Vec<(String, ParamAt)>,
    /// Result `k`'s array: `out<k>` ([C]f32).
    pub results: Vec<String>,
    pub tables: Vec<Table>,
    /// Segment all-reduces a row.
    pub all_reduces: u32,
    /// All-reduces a row of a head's sums over the roots its head straddles (`seg.head_reduce`:
    /// a head wider than the chunk, cut over consecutive roots).
    pub head_reduces: u32,
}

/// Why a region does not print.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refused(pub String);

impl std::fmt::Display for Refused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "the root region printer: {}", self.0)
    }
}

type R<T> = Result<T, Refused>;

fn refuse<T>(why: impl Into<String>) -> R<T> {
    Err(Refused(why.into()))
}

/// A value as the printed code reads it.
#[derive(Clone, Debug, PartialEq)]
enum Val {
    /// A row buffer's DSD.
    Buf(usize),
    /// A parameter row's DSD.
    In(u32),
    /// A scalar expression (a variable).
    Scal(String),
    /// A head value: an array of one f32 a head.
    Head(String, u32),
}

struct Printer<'a> {
    region: &'a Region,
    place: &'a Place,
    types: Vec<(Shape, Elem)>,
    nparams: u32,
    statics: Vec<bool>,
    /// Every static value's elements over the whole row.
    host: BTreeMap<ValueId, Vec<f32>>,
    vals: BTreeMap<ValueId, Val>,
    last_use: Vec<usize>,
    bufs_free: Vec<usize>,
    nbufs: usize,
    tables: Vec<Table>,
    decls: String,
    body: String,
    all_reduces: u32,
    head_reduces: u32,
    /// Cos / sin rows already made from an angle: angle -> (cos buf, sin buf).
    trig: BTreeMap<ValueId, (usize, usize)>,
    rope: Option<RopeUse>,
    /// The angles the rotation tables were made for, and their cos / sin rows: a second
    /// rotation of the same angles (q's and k's) reads them again.
    angles: Option<Angles>,
    /// Buffers never freed (the cos / sin rows a later rotation may read).
    pinned: Vec<usize>,
    scratch16: bool,
    exp: bool,
    recip: bool,
    sums: bool,
}

/// Angles' terms (source, frequency bits) and the cos / sin rows made of them.
type Angles = (Vec<(ValueId, u32)>, (usize, usize));

struct RopeUse {
    slots: u32,
    sources: Vec<ValueId>,
}

/// Prints `region` for a root at `place`.
pub fn print(region: &Region, place: &Place) -> R<Printed> {
    let types = region.types().map_err(|e| Refused(e.to_string()))?;
    let c = place.chunk;
    // (the roots' chunks hold the row, the last ones padded: a padded value's padding zero)
    if c == 0 || place.roots == 0 || c * place.roots < region.width {
        return refuse(format!(
            "{} roots of {c} columns do not hold a row of {}",
            place.roots, region.width
        ));
    }
    if place.params.len() != region.params.len() {
        return refuse("a parameter without its place");
    }
    let nparams = region.params.len() as u32;
    let total = types.len();
    let mut statics = vec![false; total];
    for (k, op) in region.ops.iter().enumerate() {
        let v = nparams as usize + k;
        statics[v] = match op {
            Op::Const(_) | Op::Column => true,
            op => op.args().iter().all(|&a| statics[a as usize]),
        };
    }
    let mut last_use = vec![0usize; total];
    for (k, op) in region.ops.iter().enumerate() {
        for a in op.args() {
            last_use[a as usize] = k;
        }
    }
    for &r in &region.results {
        last_use[r as usize] = usize::MAX;
    }
    let mut p = Printer {
        region,
        place,
        types,
        nparams,
        statics,
        host: BTreeMap::new(),
        vals: BTreeMap::new(),
        last_use,
        bufs_free: Vec::new(),
        nbufs: 0,
        tables: Vec::new(),
        decls: String::new(),
        body: String::new(),
        all_reduces: 0,
        head_reduces: 0,
        trig: BTreeMap::new(),
        rope: None,
        angles: None,
        pinned: Vec::new(),
        scratch16: false,
        exp: false,
        recip: false,
        sums: false,
    };
    p.eval_statics()?;
    let skip = p.angle_only();
    let mut params = Vec::new();
    for (k, (&at, &shape)) in (0u32..).zip(place.params.iter().zip(&region.params)) {
        match (at, shape) {
            (ParamAt::Row | ParamAt::Resident, Shape::Row) => {
                p.vals.insert(k, Val::In(k));
                params.push((format!("in{k}"), at));
            }
            (ParamAt::Scalar, Shape::Scalar) => {
                p.vals.insert(k, Val::Scal(format!("sc{k}")));
                params.push((format!("sc{k}"), at));
            }
            _ => return refuse(format!("parameter {k}'s place is not its shape")),
        }
    }
    for (k, op) in region.ops.iter().enumerate() {
        let v = nparams + k as u32;
        if p.statics[v as usize] || skip.contains(&v) {
            continue;
        }
        p.op(v, op, k)?;
        if p.place.round16
            && region.rounding(v).is_some()
            && let Some(Val::Buf(b)) = p.vals.get(&v)
        {
            let b = *b;
            p.scratch16 = true;
            let _ = writeln!(p.body, "  @fs2h(H16, B{b}); @fh2s(B{b}, H16);");
        }
        // free what dies here
        for a in op.args() {
            if p.last_use[a as usize] == k
                && let Some(Val::Buf(b)) = p.vals.get(&a)
            {
                let b = *b;
                if !p.bufs_free.contains(&b) && !p.pinned.contains(&b) {
                    p.bufs_free.push(b);
                }
            }
        }
    }
    let mut results = Vec::new();
    for (k, &r) in region.results.iter().enumerate() {
        let name = format!("out{k}");
        let src = p.row_of(r)?;
        let _ = writeln!(p.body, "  @fmovs(OUT{k}, {src});");
        results.push(name);
    }
    let (tables, all_reduces, head_reduces) = (p.tables.clone(), p.all_reduces, p.head_reduces);
    Ok(Printed {
        csl: p.finish(),
        params,
        results,
        tables,
        all_reduces,
        head_reduces,
    })
}

impl Printer<'_> {
    fn c(&self) -> u32 {
        self.place.chunk
    }

    fn eval_statics(&mut self) -> R<()> {
        let wanted: Vec<ValueId> = (0..self.statics.len() as u32)
            .filter(|&v| self.statics[v as usize])
            .filter(|&v| !matches!(self.op_of(v), Some(Op::Const(_))))
            .collect();
        if wanted.is_empty() {
            return Ok(());
        }
        let mut probe = self.region.clone();
        probe.results = wanted.clone();
        probe.rounds.clear();
        let args: Vec<Arg> = (self.region.params.iter())
            .map(|s| match s {
                Shape::Scalar => Arg::Scalar(0.0),
                _ => Arg::Row(vec![0.0; self.region.width as usize]),
            })
            .collect();
        let rows =
            interp::eval(&probe, &args).map_err(|e| Refused(format!("static values: {e}")))?;
        // the columns in the roots' order: each row's elements moved, a gather's index its
        // mate's new place
        let at_of: Vec<ValueId> = self
            .region
            .ops
            .iter()
            .filter_map(|o| match *o {
                Op::Peer(_, at) => Some(at),
                _ => None,
            })
            .collect();
        let w = self.region.width as usize;
        for (v, row) in wanted.into_iter().zip(rows) {
            let row = match &self.place.columns {
                Some(cols) if row.len() == w => {
                    let mut inv = vec![0u32; w];
                    for (p, &c) in cols.iter().enumerate() {
                        inv[c as usize] = p as u32;
                    }
                    cols.iter()
                        .map(|&c| {
                            let x = row[c as usize];
                            if at_of.contains(&v) {
                                inv[(x as usize).min(w - 1)] as f32
                            } else {
                                x
                            }
                        })
                        .collect()
                }
                _ => row,
            };
            self.host.insert(v, row);
        }
        Ok(())
    }

    fn op_of(&self, v: ValueId) -> Option<&Op> {
        v.checked_sub(self.nparams)
            .and_then(|k| self.region.ops.get(k as usize))
    }

    fn shape(&self, v: ValueId) -> Shape {
        self.types[v as usize].0
    }

    fn new_buf(&mut self) -> usize {
        if let Some(b) = self.bufs_free.pop() {
            return b;
        }
        self.nbufs += 1;
        self.nbufs - 1
    }

    /// A static row's table (by chunk), declared once.
    fn table(&mut self, v: ValueId, u16: bool) -> String {
        let name = format!("t{v}");
        if !self.tables.iter().any(|t| t.name == name) {
            let row = self.host[&v].clone();
            let c = self.c() as usize;
            self.tables.push(Table {
                name: name.clone(),
                u16,
                of_chunk: row.chunks(c).map(<[f32]>::to_vec).collect(),
            });
        }
        name
    }

    /// The operand text of `v` in a DSD op: a DSD, or a scalar.
    fn operand(&mut self, v: ValueId) -> R<String> {
        if self.statics[v as usize] {
            if let Some(Op::Const(l)) = self.op_of(v) {
                return Ok(match *l {
                    Literal::F32(x) => lit(x),
                    Literal::U32(x) => lit(x as f32),
                    Literal::Bool(b) => lit(if b { 1.0 } else { 0.0 }),
                });
            }
            return match self.shape(v) {
                Shape::Scalar => Ok(lit(self.host[&v][0])),
                _ => {
                    let t = self.table(v, false);
                    Ok(format!("T_{t}"))
                }
            };
        }
        Ok(match self.vals.get(&v).cloned() {
            Some(Val::Buf(b)) => format!("B{b}"),
            Some(Val::In(k)) => format!("IN{k}"),
            Some(Val::Scal(s)) => s,
            Some(Val::Head(..)) => return refuse(format!("value {v}, a head's, read as a row")),
            None => return refuse(format!("value {v} read before it is made")),
        })
    }

    /// A row value's array, as a pointer (a scalar loop's).
    fn ptr_of(&mut self, v: ValueId) -> R<String> {
        let d = self.row_of(v)?;
        Ok(if let Some(b) = d.strip_prefix('B') {
            format!("&b{b}")
        } else if let Some(k) = d.strip_prefix("IN") {
            format!("&in{k}")
        } else if let Some(t) = d.strip_prefix("T_") {
            format!("&T_{t}_a")
        } else {
            return refuse(format!("value {v} has no array"));
        })
    }

    /// A value's element `i` in a scalar loop over the chunk (a scalar's text as it is).
    fn elem_of(&mut self, v: ValueId) -> R<String> {
        if self.is_scalar_text(v) {
            return self.operand(v);
        }
        let d = self.row_of(v)?;
        Ok(if let Some(b) = d.strip_prefix('B') {
            format!("b{b}[i]")
        } else if let Some(k) = d.strip_prefix("IN") {
            format!("in{k}[i]")
        } else if let Some(t) = d.strip_prefix("T_") {
            format!("T_{t}_a[i]")
        } else {
            return refuse(format!("value {v} has no array"));
        })
    }

    /// A row value's DSD text.
    fn row_of(&mut self, v: ValueId) -> R<String> {
        let s = self.operand(v)?;
        if self.is_scalar_text(v) {
            return refuse(format!("value {v} is no row"));
        }
        Ok(s)
    }

    fn is_scalar_text(&self, v: ValueId) -> bool {
        if self.statics[v as usize] {
            return self.shape(v) == Shape::Scalar;
        }
        matches!(self.vals.get(&v), Some(Val::Scal(_)))
    }

    fn is_head(&self, v: ValueId) -> Option<(String, u32)> {
        match self.vals.get(&v) {
            Some(Val::Head(n, s)) => Some((n.clone(), *s)),
            _ => None,
        }
    }

    fn op(&mut self, v: ValueId, op: &Op, k: usize) -> R<()> {
        let shape = self.shape(v);
        let elem = self.types[v as usize].1;
        // a row's comparisons (and their joins) over its values: 0 / 1 words, a scalar loop
        let data_bool = elem == Elem::Bool
            && shape == Shape::Row
            && matches!(op, Op::Binary(b, ..) if b.compares() || b.logic());
        if elem != Elem::F32 && !matches!(op, Op::Broadcast(_)) && !data_bool {
            return refuse(format!(
                "op {k} ({op:?}): a row's bools or indices that depend on its values"
            ));
        }
        match (*op, shape) {
            (Op::Broadcast(a), _) => {
                // a scalar read across the row stays a scalar; a head's value its head's
                let val = self.vals.get(&a).cloned();
                match val {
                    Some(val) => {
                        self.vals.insert(v, val);
                    }
                    None => return refuse(format!("op {k}: broadcast of an unmade value")),
                }
            }
            (_, Shape::Scalar) => self.scalar(v, op, k)?,
            (_, Shape::Head(size)) => self.head(v, op, k, size)?,
            (_, Shape::Row) => self.row(v, op, k)?,
        }
        Ok(())
    }

    fn scalar(&mut self, v: ValueId, op: &Op, k: usize) -> R<()> {
        let name = format!("s{v}");
        let _ = writeln!(self.decls, "var {name}: f32 = 0.0;");
        let expr = match *op {
            Op::Reduce(Reduce::Sum, a) => {
                let src = self.ptr_of(a)?;
                self.sums = true;
                let mut e = format!("vsum({src}, 0, C)");
                if self.place.roots > 1 && self.place.dup > 1 {
                    e = format!("({e}) * {}", lit(1.0 / self.place.dup as f32));
                }
                if self.place.roots > 1 {
                    self.all_reduces += 1;
                    let _ = writeln!(self.body, "  sred[0] = {e};\n  seg.all_reduce(SRED, 1);");
                    e = "sred[0]".into();
                }
                e
            }
            // a row's maximum or minimum: a scalar loop over the chunk, at one root
            Op::Reduce(r @ (Reduce::Max | Reduce::Min), a) => {
                if self.place.roots > 1 {
                    return refuse(format!(
                        "op {k}: a row's {r:?} over {} roots",
                        self.place.roots
                    ));
                }
                let x = self.elem_of(a)?;
                let (init, cmp) = if r == Reduce::Max {
                    ("-3.4e38", ">")
                } else {
                    ("3.4e38", "<")
                };
                let _ = writeln!(
                    self.body,
                    "  {name} = {init}; {{ var i: u16 = 0; while (i < C) : (i += 1) {{ if ({x} {cmp} {name}) {{ {name} = {x}; }} }} }}"
                );
                self.vals.insert(v, Val::Scal(name));
                return Ok(());
            }
            Op::Unary(u, a) => {
                let x = self.operand(a)?;
                match u {
                    Unary::Neg => format!("-({x})"),
                    Unary::Abs => format!("(if (({x}) < 0.0) -({x}) else ({x}))"),
                    Unary::Rsqrt => format!("math.rsqrt({x})"),
                    Unary::Sqrt => format!("({x}) * math.rsqrt({x})"),
                    Unary::Recip => format!("math.recip({x})"),
                    Unary::Exp => format!("math.exp1({x})"),
                    Unary::Log => format!("math.ln1({x})"),
                    u => return refuse(format!("op {k}: a scalar's {u:?}")),
                }
            }
            Op::Binary(b, x, y) => {
                let (xs, ys) = (self.operand(x)?, self.operand(y)?);
                match b {
                    Binary::Add => format!("({xs}) + ({ys})"),
                    Binary::Sub => format!("({xs}) - ({ys})"),
                    Binary::Mul => format!("({xs}) * ({ys})"),
                    Binary::Max => format!("(if (({xs}) > ({ys})) ({xs}) else ({ys}))"),
                    Binary::Min => format!("(if (({xs}) < ({ys})) ({xs}) else ({ys}))"),
                    Binary::Div if self.statics[y as usize] => {
                        format!("({xs}) * {}", lit(1.0 / self.host_scalar(y)))
                    }
                    Binary::Div => format!("({xs}) * math.recip({ys})"),
                    b => return refuse(format!("op {k}: a scalar's {b:?}")),
                }
            }
            Op::Fma(a, b, c) => {
                let (a, b, c) = (self.operand(a)?, self.operand(b)?, self.operand(c)?);
                format!("({a}) * ({b}) + ({c})")
            }
            ref op => return refuse(format!("op {k}: a scalar's {op:?}")),
        };
        let _ = writeln!(self.body, "  {name} = {expr};");
        self.vals.insert(v, Val::Scal(name));
        Ok(())
    }

    fn host_scalar(&self, v: ValueId) -> f32 {
        match self.op_of(v) {
            Some(Op::Const(Literal::F32(x))) => *x,
            Some(Op::Const(Literal::U32(x))) => *x as f32,
            _ => self.host[&v][0],
        }
    }

    fn head(&mut self, v: ValueId, op: &Op, k: usize, size: u32) -> R<()> {
        let c = self.c();
        // a head wider than the chunk, cut over whole chunks: one head value a root, its sums
        // all-reduced over the roots the head straddles
        let straddle = self.place.straddle && size > c && size.is_multiple_of(c);
        if !c.is_multiple_of(size) && !straddle {
            return refuse(format!("op {k}: heads of {size} straddle chunks of {c}"));
        }
        let (nh, size) = if straddle { (1, c) } else { (c / size, size) };
        // (`hd`: `h16` is the 16-bit scratch row)
        let name = format!("hd{v}");
        let _ = writeln!(self.decls, "var {name} = @zeros([{nh}]f32);");
        let el = |p: &mut Self, x: ValueId| -> R<String> {
            if let Some((h, _)) = p.is_head(x) {
                Ok(format!("{h}[j]"))
            } else {
                p.operand(x)
            }
        };
        let expr = match *op {
            Op::ReduceHeads(Reduce::Sum, a, s) if straddle && s > size => {
                let src = self.ptr_of(a)?;
                self.sums = true;
                self.head_reduces += 1;
                let _ = writeln!(
                    self.body,
                    "  sred[0] = vsum({src}, 0, C);\n  seg.head_reduce(SRED, 1);"
                );
                "sred[0]".to_string()
            }
            Op::ReduceHeads(Reduce::Sum, a, s) if s == size => {
                let src = self.ptr_of(a)?;
                self.sums = true;
                format!("vsum({src}, j * {size}, {size})")
            }
            Op::Unary(u, a) => {
                let x = el(self, a)?;
                match u {
                    Unary::Rsqrt => format!("math.rsqrt({x})"),
                    Unary::Recip => format!("math.recip({x})"),
                    Unary::Neg => format!("-({x})"),
                    u => return refuse(format!("op {k}: a head's {u:?}")),
                }
            }
            Op::Binary(b, x, y) => {
                let (xs, ys) = (el(self, x)?, el(self, y)?);
                match b {
                    Binary::Add => format!("({xs}) + ({ys})"),
                    Binary::Sub => format!("({xs}) - ({ys})"),
                    Binary::Mul => format!("({xs}) * ({ys})"),
                    Binary::Div if self.statics[y as usize] => {
                        format!("({xs}) * {}", lit(1.0 / self.host_scalar(y)))
                    }
                    Binary::Div => format!("({xs}) * math.recip({ys})"),
                    b => return refuse(format!("op {k}: a head's {b:?}")),
                }
            }
            ref op => return refuse(format!("op {k}: a head's {op:?}")),
        };
        let _ = writeln!(
            self.body,
            "  {{ var j: u16 = 0; while (j < {nh}) : (j += 1) {{ {name}[j] = {expr}; }} }}"
        );
        self.vals.insert(v, Val::Head(name, size));
        Ok(())
    }

    /// A row op with a head operand: the op a head at a time over its columns.
    fn per_head(&mut self, dst: usize, f: &str, x: ValueId, y: ValueId) -> R<bool> {
        let (hx, hy) = (self.is_head(x), self.is_head(y));
        if hx.is_none() && hy.is_none() {
            return Ok(false);
        }
        let size = hx.as_ref().or(hy.as_ref()).map(|h| h.1).unwrap_or(1);
        let nh = self.c() / size;
        let sub = |s: &str| {
            format!(
                "@set_dsd_length(@increment_dsd_offset({s}, @as(i16, j * {size}), f32), {size})"
            )
        };
        let xs = match hx {
            Some((h, _)) => format!("{h}[j]"),
            None => sub(&self.operand(x)?),
        };
        let ys = match hy {
            Some((h, _)) => format!("{h}[j]"),
            None => sub(&self.operand(y)?),
        };
        let d = sub(&format!("B{dst}"));
        let _ = writeln!(
            self.body,
            "  {{ var j: u16 = 0; while (j < {nh}) : (j += 1) {{ {f}({d}, {xs}, {ys}); }} }}"
        );
        Ok(true)
    }

    fn row(&mut self, v: ValueId, op: &Op, k: usize) -> R<()> {
        // cos / sin of an angle: rotation tables
        if let Op::Unary(u @ (Unary::Cos | Unary::Sin), a) = *op {
            let (cb, sb) = self.trig_of(a, k)?;
            self.vals
                .insert(v, Val::Buf(if u == Unary::Cos { cb } else { sb }));
            return Ok(());
        }
        let dst = self.new_buf();
        let d = format!("B{dst}");
        match *op {
            Op::Unary(u, a) => {
                let x = self.row_of(a)?;
                match u {
                    Unary::Neg => {
                        let _ = writeln!(self.body, "  @fnegs({d}, {x});");
                    }
                    Unary::Abs => {
                        let _ = writeln!(self.body, "  @fnegs({d}, {x}); @fmaxs({d}, {d}, {x});");
                    }
                    Unary::Exp => {
                        self.exp = true;
                        let _ = writeln!(self.body, "  @fmovs({d}, {x}); vexp({d});");
                    }
                    Unary::Recip => {
                        self.recip = true;
                        let _ = writeln!(self.body, "  vrecip({d}, {x});");
                    }
                    // tanh(x) = 1 - 2 / (e^2x + 1): the row's exp and reciprocal
                    Unary::Tanh => {
                        (self.exp, self.recip) = (true, true);
                        let t = self.new_buf();
                        let _ = writeln!(
                            self.body,
                            "  @fadds({d}, {x}, {x}); vexp({d}); @fadds({d}, {d}, @as(f32, 1.0)); \
                             vrecip(B{t}, {d}); @fmuls(B{t}, B{t}, @as(f32, -2.0)); \
                             @fadds({d}, B{t}, @as(f32, 1.0));"
                        );
                        self.bufs_free.push(t);
                    }
                    // a row's (reciprocal) square root: a scalar loop
                    Unary::Rsqrt | Unary::Sqrt => {
                        let xs = self.elem_of(a)?;
                        let e = if u == Unary::Rsqrt {
                            format!("math.rsqrt({xs})")
                        } else {
                            format!("({xs}) * math.rsqrt({xs})")
                        };
                        let _ = writeln!(
                            self.body,
                            "  {{ var i: u16 = 0; while (i < C) : (i += 1) {{ b{dst}[i] = {e}; }} }}"
                        );
                    }
                    u => return refuse(format!("op {k}: a row's {u:?}")),
                }
            }
            // a comparison or a join of comparisons over the row's values: 0 / 1 words
            Op::Binary(b, x, y) if b.compares() || b.logic() => {
                let (xs, ys) = (self.elem_of(x)?, self.elem_of(y)?);
                let e = match b {
                    Binary::Gt => format!("{xs} > {ys}"),
                    Binary::Ge => format!("{xs} >= {ys}"),
                    Binary::Lt => format!("{xs} < {ys}"),
                    Binary::Le => format!("{xs} <= {ys}"),
                    Binary::Eq => format!("{xs} == {ys}"),
                    Binary::Ne => format!("{xs} != {ys}"),
                    Binary::And => format!("{xs} != 0.0 and {ys} != 0.0"),
                    _ => format!("{xs} != 0.0 or {ys} != 0.0"),
                };
                let _ = writeln!(
                    self.body,
                    "  {{ var i: u16 = 0; while (i < C) : (i += 1) {{ b{dst}[i] = if ({e}) 1.0 else 0.0; }} }}"
                );
            }
            Op::Binary(b, x, y) => {
                let f = match b {
                    Binary::Add => "@fadds",
                    Binary::Sub => "@fsubs",
                    Binary::Mul => "@fmuls",
                    Binary::Max => "@fmaxs",
                    Binary::Min | Binary::Div => "",
                    b => return refuse(format!("op {k}: a row's {b:?}")),
                };
                if !f.is_empty() && self.per_head(dst, f, x, y)? {
                } else if b == Binary::Min {
                    let (xs, ys) = (self.operand(x)?, self.operand(y)?);
                    let t = self.new_buf();
                    let _ = writeln!(
                        self.body,
                        "  @fnegs({d}, {xs}); @fnegs(B{t}, {ys}); @fmaxs({d}, {d}, B{t}); @fnegs({d}, {d});"
                    );
                    self.bufs_free.push(t);
                } else if b == Binary::Div {
                    let xs = self.operand(x)?;
                    if self.is_scalar_text(y) {
                        let r = if self.statics[y as usize] {
                            lit(1.0 / self.host_scalar(y))
                        } else {
                            format!("math.recip({})", self.operand(y)?)
                        };
                        let _ = writeln!(self.body, "  @fmuls({d}, {xs}, {r});");
                    } else {
                        self.recip = true;
                        let ys = self.row_of(y)?;
                        let _ = writeln!(self.body, "  vrecip({d}, {ys}); @fmuls({d}, {d}, {xs});");
                    }
                } else {
                    let (xs, ys) = (self.operand(x)?, self.operand(y)?);
                    if self.is_scalar_text(x) && self.is_scalar_text(y) {
                        return refuse(format!("op {k}: a row of two scalars"));
                    }
                    let _ = writeln!(self.body, "  {f}({d}, {xs}, {ys});");
                }
            }
            Op::Fma(a, b, c) => {
                let (sa, sb) = (self.is_scalar_text(a), self.is_scalar_text(b));
                let (xa, xb, xc) = (self.operand(a)?, self.operand(b)?, self.operand(c)?);
                if !self.is_scalar_text(c) && (sa || sb) {
                    let (row, s) = if sb { (xa, xb) } else { (xb, xa) };
                    let _ = writeln!(self.body, "  @fmacs({d}, {xc}, {row}, {s});");
                } else {
                    let _ = writeln!(
                        self.body,
                        "  @fmuls({d}, {xa}, {xb}); @fadds({d}, {d}, {xc});"
                    );
                }
            }
            // a select on the row's values (its 0 / 1 words): a scalar loop
            Op::Select(cnd, a, b) if !self.statics[cnd as usize] => {
                let (c, xa, xb) = (self.elem_of(cnd)?, self.elem_of(a)?, self.elem_of(b)?);
                let _ = writeln!(
                    self.body,
                    "  {{ var i: u16 = 0; while (i < C) : (i += 1) {{ b{dst}[i] = if ({c} != 0.0) {xa} else {xb}; }} }}"
                );
            }
            Op::Select(cnd, a, b) => {
                if self.shape(cnd) == Shape::Scalar {
                    let pick = if self.host[&cnd][0] != 0.0 { a } else { b };
                    let s = self.operand(pick)?;
                    let _ = writeln!(self.body, "  @fmovs({d}, {s});");
                } else if let Some(runs) = self.select_runs(cnd)
                    && let (xa, xb) = (self.operand(a)?, self.operand(b)?)
                    && is_dsd(&xa)
                    && is_dsd(&xb)
                {
                    // a static condition of few runs (a NeoX head's halves, a partial
                    // rotation's rest): each run moved from its side, no mask tables
                    for (e, n, hold) in runs {
                        let src = if hold { &xa } else { &xb };
                        if *src == d {
                            continue;
                        }
                        let _ = writeln!(
                            self.body,
                            "  @fmovs(@increment_dsd_offset(@set_dsd_length({d}, {n}), {e}, f32), @increment_dsd_offset(@set_dsd_length({src}, {n}), {e}, f32));"
                        );
                    }
                } else {
                    let m = self.mask(cnd, false);
                    let nm = self.mask(cnd, true);
                    let (xa, xb) = (self.operand(a)?, self.operand(b)?);
                    let t = self.new_buf();
                    let _ = writeln!(
                        self.body,
                        "  @fmuls({d}, T_{m}, {xa}); @fmuls(B{t}, T_{nm}, {xb}); @fadds({d}, {d}, B{t});"
                    );
                    self.bufs_free.push(t);
                }
            }
            Op::Peer(src, at) => {
                if !self.statics[at as usize] {
                    return refuse(format!("op {k}: a gather by the row's values"));
                }
                let idx = self.host[&at].clone();
                let c = self.c() as usize;
                let swap = idx
                    .iter()
                    .enumerate()
                    .all(|(e, &m)| m as usize == (e ^ 1) && c.is_multiple_of(2));
                // the mate `hop` columns on in the first half of each `2 hop` block, back in
                // the second (NeoX's halves of a head)
                let hop = idx.first().map_or(0, |&m| m as usize);
                let halves = !swap
                    && hop > 1
                    && c.is_multiple_of(2 * hop)
                    && idx.iter().enumerate().all(|(e, &m)| {
                        m as usize
                            == if e % (2 * hop) < hop {
                                e + hop
                            } else {
                                e - hop
                            }
                    });
                // a head's halves swapped and its rest kept (a partial rotation's): blocks of
                // `per` columns, the first `2 hop` of each its halves' mates, the rest their own
                let per = (!swap && !halves && hop > 0)
                    .then(|| (2 * hop..=c).find(|&p| p == c || idx[p] as usize != p))
                    .flatten()
                    .filter(|&per| {
                        per > 2 * hop
                            && c.is_multiple_of(per)
                            && idx.iter().enumerate().all(|(e, &m)| {
                                let r = e % per;
                                m as usize
                                    == if r < hop {
                                        e + hop
                                    } else if r < 2 * hop {
                                        e - hop
                                    } else {
                                        e
                                    }
                            })
                    });
                if !swap && !halves && per.is_none() {
                    // the general gather: each column's mate in its own chunk, by the chunk's
                    // index table (a scalar loop); a mate in another chunk is refused
                    let s = self.row_of(src)?;
                    let arr = s
                        .strip_prefix('B')
                        .map(|n| format!("b{n}"))
                        .or_else(|| s.strip_prefix("IN").map(|n| format!("in{n}")));
                    let within = idx
                        .iter()
                        .enumerate()
                        .all(|(e, &m)| m as usize / c == e / c);
                    let (Some(arr), true) = (arr, within) else {
                        return refuse(format!(
                            "op {k}: a gather of a mate in another chunk, or of no row"
                        ));
                    };
                    let name = format!("g{v}");
                    self.tables.push(Table {
                        name: name.clone(),
                        u16: true,
                        of_chunk: idx
                            .chunks(c)
                            .map(|ch| ch.iter().map(|&m| (m as usize % c) as f32).collect())
                            .collect(),
                    });
                    let _ = writeln!(
                        self.body,
                        "  {{ var e: u16 = 0; while (e < C) : (e += 1) {{ b{dst}[e] = {arr}[T_{name}_a[e]]; }} }}"
                    );
                    self.vals.insert(v, Val::Buf(dst));
                    return Ok(());
                }
                let s = self.row_of(src)?;
                if let Some(per) = per {
                    let part = |t: &str, o: usize, n: usize| {
                        format!("@increment_dsd_offset(@set_dsd_length({t}, {n}), {o}, f32)")
                    };
                    for b in (0..c).step_by(per) {
                        let _ = writeln!(
                            self.body,
                            "  @fmovs({}, {}); @fmovs({}, {}); @fmovs({}, {});",
                            part(&d, b, hop),
                            part(&s, b + hop, hop),
                            part(&d, b + hop, hop),
                            part(&s, b, hop),
                            part(&d, b + 2 * hop, per - 2 * hop),
                            part(&s, b + 2 * hop, per - 2 * hop)
                        );
                    }
                    self.vals.insert(v, Val::Buf(dst));
                    return Ok(());
                }
                if halves {
                    let part = |t: &str, o: usize| {
                        format!("@increment_dsd_offset(@set_dsd_length({t}, {hop}), {o}, f32)")
                    };
                    for b in (0..c).step_by(2 * hop) {
                        let _ = writeln!(
                            self.body,
                            "  @fmovs({}, {}); @fmovs({}, {});",
                            part(&d, b),
                            part(&s, b + hop),
                            part(&d, b + hop),
                            part(&s, b)
                        );
                    }
                    self.vals.insert(v, Val::Buf(dst));
                    return Ok(());
                }
                let h = c / 2;
                let half = |t: &str, o: u32| {
                    format!(
                        "@increment_dsd_offset(@set_dsd_stride(@set_dsd_length({t}, {h}), 2), {o}, f32)"
                    )
                };
                let _ = writeln!(
                    self.body,
                    "  @fmovs({}, {}); @fmovs({}, {});",
                    half(&d, 0),
                    half(&s, 1),
                    half(&d, 1),
                    half(&s, 0)
                );
            }
            ref op => return refuse(format!("op {k}: a row's {op:?}")),
        }
        self.vals.insert(v, Val::Buf(dst));
        Ok(())
    }

    /// The values only an angle reads (the rotation tables make the cos / sin; the angles are never
    /// computed): each cos / sin's angle tree, where nothing outside it reads its values.
    fn angle_only(&self) -> Vec<ValueId> {
        let n = self.nparams;
        let mut users: BTreeMap<ValueId, Vec<ValueId>> = BTreeMap::new();
        for (k, op) in self.region.ops.iter().enumerate() {
            for a in op.args() {
                users.entry(a).or_default().push(n + k as u32);
            }
        }
        let mut tree: Vec<ValueId> = Vec::new();
        let mut trig: Vec<ValueId> = Vec::new();
        for (k, op) in self.region.ops.iter().enumerate() {
            if let Op::Unary(Unary::Cos | Unary::Sin, a) = *op
                && self.angle_terms(a).is_some()
            {
                trig.push(n + k as u32);
                let mut stack = vec![a];
                while let Some(v) = stack.pop() {
                    if self.statics[v as usize] || tree.contains(&v) {
                        continue;
                    }
                    tree.push(v);
                    if let Some(Op::Select(_, x, y)) = self.op_of(v) {
                        stack.extend([*x, *y]);
                    }
                }
            }
        }
        let inside = |u: &ValueId| tree.contains(u) || trig.contains(u);
        let mut keep = tree.clone();
        keep.retain(|v| {
            users.get(v).is_none_or(|us| us.iter().all(inside)) && !self.region.results.contains(v)
        });
        keep
    }

    /// A static bool row's runs `(e, n, holds)` over a chunk, the same for every chunk, at most
    /// [`SELECT_RUNS`] of them (else `None`: its masks).
    fn select_runs(&self, v: ValueId) -> Option<Vec<(usize, usize, bool)>> {
        let c = self.c() as usize;
        let row = self.host.get(&v)?;
        let chunks: Vec<&[f32]> = row.chunks(c).collect();
        let first = chunks.first()?;
        if first.len() != c
            || chunks.iter().any(|k| {
                k.iter()
                    .zip(first.iter())
                    .any(|(x, y)| (*x != 0.0) != (*y != 0.0))
            })
        {
            return None;
        }
        let mut runs: Vec<(usize, usize, bool)> = Vec::new();
        for (e, &x) in first.iter().enumerate() {
            let h = x != 0.0;
            match runs.last_mut() {
                Some((_, n, hh)) if *hh == h => *n += 1,
                _ => runs.push((e, 1, h)),
            }
            if runs.len() > SELECT_RUNS {
                return None;
            }
        }
        Some(runs)
    }

    /// A static bool row's mask (1 where it holds), or its complement.
    fn mask(&mut self, v: ValueId, not: bool) -> String {
        let name = format!("m{v}{}", if not { "n" } else { "" });
        if !self.tables.iter().any(|t| t.name == name) {
            let c = self.c() as usize;
            let row: Vec<f32> = self.host[&v]
                .iter()
                .map(|&b| if (b != 0.0) != not { 1.0 } else { 0.0 })
                .collect();
            self.tables.push(Table {
                name: name.clone(),
                u16: false,
                of_chunk: row.chunks(c).map(<[f32]>::to_vec).collect(),
            });
        }
        name
    }

    /// Each column's (position scalar, frequency) of an angle: a select over static conditions of
    /// `position x frequency` terms.
    fn angle_terms(&self, a: ValueId) -> Option<Vec<(ValueId, f32)>> {
        let w = self.region.width as usize;
        match self.op_of(a)? {
            Op::Select(c, x, y) if self.statics[*c as usize] && self.shape(*c) != Shape::Scalar => {
                let (tx, ty) = (self.angle_terms(*x)?, self.angle_terms(*y)?);
                let cond = &self.host[c];
                Some(
                    (0..w)
                        .map(|e| if cond[e] != 0.0 { tx[e] } else { ty[e] })
                        .collect(),
                )
            }
            Op::Binary(Binary::Mul, x, y) => {
                let (p, f) = if self.statics[*y as usize] {
                    (*x, *y)
                } else {
                    (*y, *x)
                };
                if !self.statics[f as usize] || self.statics[p as usize] {
                    return None;
                }
                if self.shape(p) != Shape::Scalar
                    && !matches!(self.vals.get(&p), Some(Val::Scal(_)))
                {
                    return None;
                }
                let freq = self
                    .host
                    .get(&f)
                    .cloned()
                    .unwrap_or_else(|| vec![self.host_scalar(f); w]);
                Some(freq.into_iter().map(|fr| (p, fr)).collect())
            }
            _ => None,
        }
    }

    /// cos / sin rows of angle `a` from rotation tables.
    fn trig_of(&mut self, a: ValueId, k: usize) -> R<(usize, usize)> {
        if let Some(&cs) = self.trig.get(&a) {
            return Ok(cs);
        }
        let Some(terms) = self.angle_terms(a) else {
            return refuse(format!("op {k}: a cos / sin of no position x frequency"));
        };
        let key: Vec<(ValueId, u32)> = terms.iter().map(|&(s, f)| (s, f.to_bits())).collect();
        if let Some((held, cs)) = &self.angles
            && *held == key
        {
            let cs = *cs;
            self.trig.insert(a, cs);
            return Ok(cs);
        }
        if self.rope.is_some() {
            return refuse(format!("op {k}: a second rotation's angles"));
        }
        let c = self.c() as usize;
        // sources in order of first use; each chunk's slots: its (source, frequency) pairs
        // source-major, in column order
        let mut sources: Vec<ValueId> = Vec::new();
        for &(s, _) in &terms {
            if !sources.contains(&s) {
                sources.push(s);
            }
        }
        let chunks: Vec<&[(ValueId, f32)]> = terms.chunks(c).collect();
        let mut slot_of: Vec<Vec<f32>> = Vec::new();
        let mut freqs: Vec<Vec<f32>> = Vec::new();
        let mut ranges: Vec<Vec<f32>> = Vec::new();
        for chunk in &chunks {
            let mut slots: Vec<(ValueId, u32)> = Vec::new();
            let mut rng = Vec::new();
            for &s in &sources {
                let from = slots.len();
                for &(cs, f) in chunk.iter() {
                    if cs == s && !slots.contains(&(s, f.to_bits())) {
                        slots.push((s, f.to_bits()));
                    }
                }
                rng.extend([from as f32, (slots.len() - from) as f32]);
            }
            slot_of.push(
                chunk
                    .iter()
                    .map(|&(s, f)| {
                        slots
                            .iter()
                            .position(|&x| x == (s, f.to_bits()))
                            .unwrap_or(0) as f32
                    })
                    .collect(),
            );
            freqs.push(slots.iter().map(|&(_, f)| f32::from_bits(f)).collect());
            ranges.push(rng);
        }
        let ns = freqs.iter().map(Vec::len).max().unwrap_or(1).max(1);
        let (lv, bits) = (self.place.rope_lv, self.place.rope_bits);
        let rb = 1usize << bits;
        // rot[base + ((lv rb + d) 2) ns + j]: cos, then sin at + ns (ops/rope.csl); at BITS 0 the
        // angles are computed at the root and the table holds each frequency's cos / sin step
        let rot: Vec<Vec<f32>> = freqs
            .iter()
            .map(|fs| {
                if bits == 0 {
                    let mut t = vec![0f32; 3 * ns];
                    for (j, &f) in fs.iter().enumerate() {
                        t[j] = f;
                        t[ns + j] = f64::from(f).cos() as f32;
                        t[2 * ns + j] = f64::from(f).sin() as f32;
                    }
                    return t;
                }
                let mut t = vec![0f32; lv as usize * rb * 2 * ns];
                for l in 0..lv as usize {
                    for d in 0..rb {
                        let scale =
                            (d as f64) * f64::from(1u32 << (bits as usize * (lv as usize - 1 - l)));
                        for (j, &f) in fs.iter().enumerate() {
                            let ang = scale * f64::from(f);
                            t[((l * rb + d) * 2) * ns + j] = ang.cos() as f32;
                            t[((l * rb + d) * 2) * ns + ns + j] = ang.sin() as f32;
                        }
                    }
                }
                t
            })
            .collect();
        let pairs = slot_of
            .iter()
            .all(|m| m.iter().enumerate().all(|(e, &j)| j as usize == e / 2));
        let runs = if pairs { None } else { slot_runs(&slot_of) };
        self.tables.push(Table {
            name: "rot".into(),
            u16: false,
            of_chunk: rot,
        });
        self.tables.push(Table {
            name: "rng".into(),
            u16: false,
            of_chunk: ranges,
        });
        if !pairs && runs.is_none() {
            self.tables.push(Table {
                name: "slot".into(),
                u16: true,
                of_chunk: slot_of,
            });
        }
        let (cb, sb) = (self.new_buf(), self.new_buf());
        let mut code = String::new();
        for (i, &s) in sources.iter().enumerate() {
            let p = self.operand(s)?;
            let _ = writeln!(
                code,
                "  {{ const n = @as(u16, T_rng_a[{}]); if (n > 0) {{ const b = @as(u16, T_rng_a[{}]); rp.tables(@as(u32, {p}), b, b, n); }} }}",
                2 * i + 1,
                2 * i
            );
        }
        if pairs {
            let h = c / 2;
            for (buf, src) in [(cb, "RC"), (sb, "RS")] {
                for o in 0..2 {
                    let _ = writeln!(
                        code,
                        "  @fmovs(@increment_dsd_offset(@set_dsd_stride(@set_dsd_length(B{buf}, {h}), 2), {o}, f32), @set_dsd_length({src}, {h}));"
                    );
                }
            }
        } else if let Some(runs) = &runs {
            for (buf, src) in [(cb, "RC"), (sb, "RS")] {
                for &(e, j, n) in runs {
                    let _ = writeln!(
                        code,
                        "  @fmovs(@increment_dsd_offset(@set_dsd_length(B{buf}, {n}), {e}, f32), @increment_dsd_offset(@set_dsd_length({src}, {n}), {j}, f32));"
                    );
                }
            }
        } else {
            let _ = writeln!(
                code,
                "  {{ var e: u16 = 0; while (e < {c}) : (e += 1) {{ b{cb}[e] = rc[T_slot_a[e]]; b{sb}[e] = rs[T_slot_a[e]]; }} }}"
            );
        }
        self.body.push_str(&code);
        self.rope = Some(RopeUse {
            slots: ns as u32,
            sources,
        });
        self.trig.insert(a, (cb, sb));
        self.angles = Some((key, (cb, sb)));
        self.pinned.extend([cb, sb]);
        Ok((cb, sb))
    }

    fn finish(self) -> String {
        let c = self.c();
        let mut s = String::new();
        let _ = writeln!(
            s,
            "// A root region printed by kernels-cerebras emit::region: the region's\n\
             // row cut across {} roots of {c} columns. Interface: in<k> / sc<k> (the parameters), out<k>\n\
             // (the results), the host's tables (T_<name>), row() a row.\n\
             //\n\
             // Params: math (ops/math.csl), seg (an ops/seg.csl: the row reductions' all-reduce over the roots).",
            self.place.roots
        );
        let _ = writeln!(
            s,
            "param math: imported_module;\nparam seg: imported_module;\n"
        );
        let _ = writeln!(s, "const C: u16 = {c};");
        for (k, (&at, &shape)) in (0u32..).zip(self.place.params.iter().zip(&self.region.params)) {
            if shape == Shape::Scalar {
                let _ = writeln!(s, "var sc{k}: f32 = 0.0;");
            } else {
                let _ = writeln!(
                    s,
                    "var in{k}: [C]f32 align(16) = @zeros([C]f32);   // {}\nconst IN{k} = @get_dsd(mem1d_dsd, .{{ .tensor_access = |i|{{C}} -> in{k}[i] }});",
                    if at == ParamAt::Resident {
                        "resident"
                    } else {
                        "a row's"
                    }
                );
            }
        }
        for k in 0..self.region.results.len() {
            let _ = writeln!(
                s,
                "var out{k}: [C]f32 align(16) = @zeros([C]f32);\nconst OUT{k} = @get_dsd(mem1d_dsd, .{{ .tensor_access = |i|{{C}} -> out{k}[i] }});"
            );
        }
        for b in 0..self.nbufs {
            let _ = writeln!(
                s,
                "var b{b}: [C]f32 align(16) = @zeros([C]f32);\nconst B{b} = @get_dsd(mem1d_dsd, .{{ .tensor_access = |i|{{C}} -> b{b}[i] }});"
            );
        }
        for t in &self.tables {
            let n = t.of_chunk.iter().map(Vec::len).max().unwrap_or(1).max(1);
            let ty = if t.u16 { "u16" } else { "f32" };
            let _ = writeln!(s, "var T_{}_a = @zeros([{n}]{ty});", t.name);
            if !t.u16 && t.name != "rot" && t.name != "rng" {
                let _ = writeln!(
                    s,
                    "const T_{0} = @get_dsd(mem1d_dsd, .{{ .tensor_access = |i|{{C}} -> T_{0}_a[i] }});",
                    t.name
                );
            }
        }
        if let Some(r) = &self.rope {
            let _ = writeln!(
                s,
                "var rc = @zeros([{0}]f32);\nvar rs = @zeros([{0}]f32);\n\
                 const RC = @get_dsd(mem1d_dsd, .{{ .tensor_access = |i|{{{0}}} -> rc[i] }});\n\
                 const RS = @get_dsd(mem1d_dsd, .{{ .tensor_access = |i|{{{0}}} -> rs[i] }});\n\
                 const rp = @import_module(\"ops/srope.csl\", .{{ .rot = &T_rot_a, .rc = &rc, .rs = &rs, .NROT = {0}, .LV = {1}, .BITS = {2}{4} }});\n\
                 // (sources: {3:?})",
                r.slots,
                self.place.rope_lv,
                self.place.rope_bits,
                r.sources,
                if self.place.rope_bits == 0 {
                    format!(", .DIRECT = true, .STEP = {ROPE_ANCHOR}")
                } else {
                    String::new()
                }
            );
            if !self.tables.iter().any(|t| t.name == "slot") {
                let _ = writeln!(s, "var T_slot_a = @zeros([1]u16);");
            }
        }
        if self.scratch16 {
            let _ = writeln!(
                s,
                "var h16 = @zeros([C]@fp16());\nconst H16 = @get_dsd(mem1d_dsd, .{{ .tensor_access = |i|{{C}} -> h16[i] }});"
            );
        }
        if self.place.roots > 1 && self.all_reduces + self.head_reduces > 0 {
            let _ = writeln!(
                s,
                "var sred = @zeros([1]f32);\nconst SRED = @get_dsd(mem1d_dsd, .{{ .tensor_access = |i|{{1}} -> sred[i] }});"
            );
        }
        if self.sums {
            s.push_str(
                "// n words' sum from p[o]: a halving tree of DSD adds over a scratch row (a stride-0\n\
                 // accumulating DSD op is wrong on the CS-3); an odd run's middle word waits a level\n\
                 var sumt: [C]f32 align(16) = @zeros([C]f32);\n\
                 const SUMT = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{C} -> sumt[i] });\n\
                 fn vsum(p: [*]f32, o: u16, n: u16) f32 {\n\
                 \x20 if (n == 1) { return p[o]; }\n\
                 \x20 const src = @increment_dsd_offset(@set_dsd_base_addr(SUMT, p), @as(i16, o), f32);\n\
                 \x20 var h: u16 = n / 2;\n\
                 \x20 @fadds(@set_dsd_length(SUMT, h), @set_dsd_length(src, h),\n\
                 \x20        @set_dsd_length(@increment_dsd_offset(src, @as(i16, n - h), f32), h));\n\
                 \x20 var m: u16 = h;\n\
                 \x20 if (n % 2 == 1) { sumt[h] = p[o + h]; m = h + 1; }\n\
                 \x20 while (m > 1) {\n\
                 \x20   h = m / 2;\n\
                 \x20   @fadds(@set_dsd_length(SUMT, h), @set_dsd_length(SUMT, h),\n\
                 \x20          @set_dsd_length(@increment_dsd_offset(SUMT, @as(i16, m - h), f32), h));\n\
                 \x20   m -= h;\n\
                 \x20 }\n\
                 \x20 return sumt[0];\n}\n",
            );
        }
        if self.exp {
            s.push_str(
                "var ep: [C]f32 align(16) = @zeros([C]f32);\nvar ek = @zeros([C]u16);\n\
                 const EP = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{C} -> ep[i] });\n\
                 const EK = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{C} -> ek[i] });\nvar EPL = EK;\n\
                 fn vexp(b: mem1d_dsd) void { math.vexp(b, EP, EK, EPL, 88.0, true); }\n",
            );
        }
        if self.recip {
            s.push_str(
                "// 1 / x: the high half's bit guess (0x7ef4 - |x|'s), three Newton steps, x's sign\n\
                 var ra: [C]f32 align(16) = @zeros([C]f32);\nvar ry: [C]f32 align(16) = @zeros([C]f32);\nvar rsg = @zeros([C]u16);\n\
                 const RA = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{C} -> ra[i] });\n\
                 const RY = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{C} -> ry[i] });\n\
                 const RSG = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{C} -> rsg[i] });\n\
                 var RAH = RSG;\nvar RYH = RSG;\nvar RYL = RSG;\n\
                 fn vrecip(dst: mem1d_dsd, src: mem1d_dsd) void {\n\
                 \x20 @fmovs(RA, src);\n  @and16(RSG, RAH, @as(u16, 0x8000));\n  @and16(RAH, RAH, @as(u16, 0x7fff));\n\
                 \x20 @xor16(RYH, RAH, @as(u16, 0xffff));\n  @add16(RYH, RYH, @as(u16, 0x7ef4));\n  @mov16(RYL, @as(u16, 0));\n\
                 \x20 var k: u16 = 0;\n  while (k < 3) : (k += 1) { @fmuls(dst, RA, RY); @fsubs(dst, 2.0, dst); @fmuls(RY, RY, dst); }\n\
                 \x20 @or16(RYH, RYH, RSG);\n  @fmovs(dst, RY);\n}\n",
            );
        }
        // the harness's packing: a row's record (its rows' chunks, then its scalars), the resident
        // chunks, the results' chunks, the tables (in `tables` order; u16 ones as f32 words)
        let (mut rows, mut scal, mut res) = (Vec::new(), Vec::new(), Vec::new());
        for (k, (&at, &shape)) in (0u32..).zip(self.place.params.iter().zip(&self.region.params)) {
            match (at, shape) {
                (_, Shape::Scalar) => scal.push(k),
                (ParamAt::Resident, _) => res.push(k),
                _ => rows.push(k),
            }
        }
        let ntab: usize = self
            .tables
            .iter()
            .map(|t| t.of_chunk.iter().map(Vec::len).max().unwrap_or(0))
            .sum();
        let _ = writeln!(
            s,
            "const NROW: u16 = {};\nconst NSC: u16 = {};\nconst NRES: u16 = {};\nconst NOUT: u16 = {};\nconst NTAB: u16 = {};",
            rows.len(),
            scal.len(),
            res.len(),
            self.region.results.len(),
            ntab.max(1)
        );
        s.push_str("fn load_row(p: [*]f32) void {\n  const src = @set_dsd_base_addr(OUT0, p);\n");
        for (i, k) in rows.iter().enumerate() {
            let _ = writeln!(
                s,
                "  @fmovs(IN{k}, @increment_dsd_offset(src, @as(i16, {}), f32));",
                i as u32 * c
            );
        }
        for (i, k) in scal.iter().enumerate() {
            let _ = writeln!(s, "  sc{k} = p[{}];", rows.len() as u32 * c + i as u32);
        }
        s.push_str(
            "}\nfn load_res(p: [*]f32) void {\n  const src = @set_dsd_base_addr(OUT0, p);\n",
        );
        for (i, k) in res.iter().enumerate() {
            let _ = writeln!(
                s,
                "  @fmovs(IN{k}, @increment_dsd_offset(src, @as(i16, {}), f32));",
                i as u32 * c
            );
        }
        s.push_str(
            "}\nfn store_row(p: [*]f32) void {\n  const dst = @set_dsd_base_addr(OUT0, p);\n",
        );
        for k in 0..self.region.results.len() {
            let _ = writeln!(
                s,
                "  @fmovs(@increment_dsd_offset(dst, @as(i16, {}), f32), OUT{k});",
                k as u32 * c
            );
        }
        // the tables a word at a time: word g of the packed tables (load_tabs: n words from g = off)
        s.push_str("}\nfn tab_put(g: u16, x: f32) void {\n");
        let mut off = 0usize;
        let mut first = true;
        for t in &self.tables {
            let n = t.of_chunk.iter().map(Vec::len).max().unwrap_or(0);
            let conv = if t.u16 { "@as(u16, x)" } else { "x" };
            let _ = writeln!(
                s,
                "  {}if (g < {}) {{ T_{}_a[g - {off}] = {conv}; }}",
                if first { "" } else { "else " },
                off + n,
                t.name
            );
            first = false;
            off += n;
        }
        s.push_str("}\nfn load_tabs(p: [*]f32, off: u16, n: u16) void {\n  var e: u16 = 0;\n  while (e < n) : (e += 1) { tab_put(off + e, p[e]); }\n}\n");
        s.push_str("\nfn init() void {\n");
        if self.exp {
            s.push_str(
                "  EPL = @set_dsd_stride(@set_dsd_base_addr(EK, @ptrcast([*]u16, &ep)), 2);\n",
            );
        }
        if self.recip {
            s.push_str(
                "  RAH = @increment_dsd_offset(@set_dsd_stride(@set_dsd_base_addr(RSG, @ptrcast([*]u16, &ra)), 2), 1, u16);\n\
                 \x20 RYH = @increment_dsd_offset(@set_dsd_stride(@set_dsd_base_addr(RSG, @ptrcast([*]u16, &ry)), 2), 1, u16);\n\
                 \x20 RYL = @set_dsd_stride(@set_dsd_base_addr(RSG, @ptrcast([*]u16, &ry)), 2);\n",
            );
        }
        s.push_str("}\n\n");
        s.push_str(&self.decls);
        s.push_str("\n// one row: in<k> / sc<k> in, out<k> out\nfn row() void {\n");
        s.push_str(&self.body);
        s.push_str("}\n");
        s
    }
}

/// An f32 literal CSL reads back to the same bits.
fn lit(x: f32) -> String {
    if x.is_finite() {
        let s = format!("{x:?}");
        if s.contains('.') || s.contains('e') {
            format!("@as(f32, {s})")
        } else {
            format!("@as(f32, {s}.0)")
        }
    } else if x.is_nan() {
        "@bitcast(f32, @as(u32, 0x7fc00000))".into()
    } else if x > 0.0 {
        "@bitcast(f32, @as(u32, 0x7f800000))".into()
    } else {
        "@bitcast(f32, @as(u32, 0xff800000))".into()
    }
}

/// A printed module's scratch rows (its row buffers, the sum's, exp's, recip's and rounding's
/// rows, and its row parameters): nothing held across `row()`, so a root's regions share them.
fn scratch_name(n: &str) -> bool {
    matches!(n, "sumt" | "ep" | "ek" | "ra" | "ry" | "rsg" | "h16")
        || n.strip_prefix('b')
            .is_some_and(|k| !k.is_empty() && k.bytes().all(|c| c.is_ascii_digit()))
}

/// A scratch row as `pooled` takes it in: its name and element type.
pub type ScratchRow = (String, &'static str);

/// `csl` (a printed module) with its scratch rows taken as pointer parameters in place of its
/// own arrays, the importing root passing rows of its pool. Each row's name and element type,
/// and the module's chunk (`C`).
#[must_use]
pub fn pooled(csl: &str) -> (String, Vec<ScratchRow>, u32) {
    let mut out = String::with_capacity(csl.len());
    let mut rows: Vec<ScratchRow> = Vec::new();
    let mut c = 0;
    for line in csl.split_inclusive('\n') {
        if let Some(v) = line.strip_prefix("const C: u16 = ") {
            c = v.trim_end().trim_end_matches(';').parse().unwrap_or(0);
        }
        let decl = line.strip_prefix("var ").and_then(|r| {
            let name = r.split([':', ' ']).next()?;
            let ty = if r.contains("@zeros([C]f32)") {
                "f32"
            } else if r.contains("@zeros([C]u16)") {
                "u16"
            } else if r.contains("@zeros([C]@fp16())") {
                "@fp16()"
            } else {
                return None;
            };
            // (a row's parameter: the root writes it just before `row()`, never reads it)
            let row_in = name
                .strip_prefix("in")
                .is_some_and(|k| !k.is_empty() && k.bytes().all(|c| c.is_ascii_digit()))
                && r.contains("// a row's");
            (scratch_name(name) || row_in).then(|| (name.to_string(), ty))
        });
        match decl {
            Some((name, ty)) => {
                out.push_str(&format!("param {name}: [*]{ty};\n"));
                rows.push((name, ty));
            }
            None => out.push_str(line),
        }
    }
    // a pointer parameter is its row's address already
    let mut text = String::with_capacity(out.len());
    let mut rest = out.as_str();
    while let Some(i) = rest.find('&') {
        text.push_str(&rest[..i]);
        let after = &rest[i + 1..];
        let n = after
            .find(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_'))
            .unwrap_or(after.len());
        if !rows.iter().any(|(r, _)| r == &after[..n]) {
            text.push('&');
        }
        rest = after;
    }
    text.push_str(rest);
    (text, rows, c)
}

#[cfg(test)]
mod tests {

    /// srope's STEP in f32 (a DIRECT anchor each period, one rotation a position between) stays
    /// within the budget against f64 over 64k positions of three bases' 64 pairs.
    #[test]
    fn rope_anchor_drift_in_budget() {
        let n = super::ROPE_ANCHOR;
        assert!(n.is_power_of_two() && n >= 2, "{n}");
        assert_eq!(n, 64);
        let mut worst = 0f64;
        for theta in [1e4f32, 5e5, 1e6] {
            for j in 0..64 {
                let f = 1.0 / theta.powf(2.0 * j as f32 / 128.0);
                let (c1, s1) = (f64::from(f).cos() as f32, f64::from(f).sin() as f32);
                let (mut c, mut s) = (1f32, 0f32);
                for p in 0u32..65_536 {
                    let a = f64::from(p) * f64::from(f);
                    if p % n == 0 {
                        (c, s) = (a.cos() as f32, a.sin() as f32);
                    } else {
                        // srope's advance: a = s s1, b = c s1, c = c c1 - a, s = s c1 + b
                        let (x, y) = (s * s1, c * s1);
                        (c, s) = (c * c1 - x, s * c1 + y);
                    }
                    let e = (f64::from(c) - a.cos()).hypot(f64::from(s) - a.sin());
                    worst = worst.max(e);
                }
            }
        }
        eprintln!("rope drift: worst {worst:.3e} at an anchor every {n}");
        assert!(
            worst <= super::ROPE_BUDGET - super::ROPE_DIRECT_ERR,
            "{worst:e}"
        );
    }

    /// A NeoX head's gather is two runs a head; interleaved slots (pairs) or chunks that differ
    /// are not runs.
    #[test]
    fn slot_runs_of_a_neox_head() {
        let neox: Vec<f32> = (0..256).map(|e| (e % 64) as f32).collect();
        assert_eq!(
            super::slot_runs(&[neox.clone(), neox.clone()]),
            Some(vec![(0, 0, 64), (64, 0, 64), (128, 0, 64), (192, 0, 64)])
        );
        let mut other = neox.clone();
        other.swap(0, 1);
        assert_eq!(super::slot_runs(&[neox, other]), None);
        let pairs: Vec<f32> = (0..128).map(|e| (e / 2) as f32).collect();
        assert_eq!(super::slot_runs(&[pairs]), None);
    }
}

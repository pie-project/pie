//! The reference interpreter of both forms: a region op by op, and a
//! kernel phase by phase over a frame. Each takes the same math (the host's
//! f32, a reduction's sum in f64, rounded to f32 once), so a kernel run here
//! lands its region's bits: the schedule moves where a value is computed
//! and kept, never what it is. A device's kernel sums in its own order, in
//! f32, and is held to this reference within each op's tolerance.

use std::collections::BTreeMap;
use std::fmt;

use dtype::Dtype;

use crate::frame::{Frame, round_bf16};
use crate::kernel::{Expr, Kernel, Operand, Table};
use crate::region::{
    Binary, Elem, Literal, Op, Reduce, Region, RegionError, Shape, Unary, ValueId,
};

/// A parameter's value at one row.
#[derive(Clone, Debug, PartialEq)]
pub enum Arg {
    Row(Vec<f32>),
    Scalar(f32),
}

/// Why the interpreter does not run a region.
#[derive(Clone, Debug, PartialEq)]
pub enum InterpError {
    Region(RegionError),
    /// The arguments are not the parameters' shapes or the row's width.
    Arguments {
        param: u32,
    },
    /// A rounding to a dtype the interpreter does not round to.
    Rounding(Dtype),
}

impl fmt::Display for InterpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InterpError::Region(why) => write!(f, "{why}"),
            InterpError::Arguments { param } => {
                write!(f, "argument {param} is not of its parameter's shape")
            }
            InterpError::Rounding(dtype) => {
                write!(f, "the interpreter rounds no value to {dtype:?}")
            }
        }
    }
}

impl std::error::Error for InterpError {}

/// One element of a value: a number or a bool.
#[derive(Clone, Copy, Debug, PartialEq)]
enum V {
    F(f32),
    B(bool),
    U(u32),
}

impl V {
    fn f(self) -> f32 {
        match self {
            V::F(x) => x,
            V::B(b) => f32::from(u8::from(b)),
            #[allow(clippy::cast_precision_loss)]
            V::U(x) => x as f32,
        }
    }

    fn b(self) -> bool {
        match self {
            V::B(b) => b,
            V::F(x) => x != 0.0,
            V::U(x) => x != 0,
        }
    }

    fn u(self) -> u32 {
        match self {
            V::U(x) => x,
            V::B(b) => u32::from(b),
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            V::F(x) => x as u32,
        }
    }
}

fn literal(l: Literal) -> V {
    match l {
        Literal::F32(x) => V::F(x),
        Literal::Bool(b) => V::B(b),
        Literal::U32(x) => V::U(x),
    }
}

/// The host's math for one op: what every interpreted kernel computes.
fn unary(op: Unary, x: V) -> V {
    match (op, x) {
        (Unary::Not, x) => V::B(!x.b()),
        (Unary::Abs | Unary::Sign, V::U(u)) => V::U(if op == Unary::Abs {
            u
        } else {
            u32::from(u > 0)
        }),
        (Unary::Neg, V::U(u)) => V::U(u.wrapping_neg()),
        (op, x) => {
            let x = x.f();
            V::F(match op {
                Unary::Exp => x.exp(),
                Unary::Log => x.ln(),
                Unary::Recip => 1.0 / x,
                Unary::Sin => x.sin(),
                Unary::Cos => x.cos(),
                Unary::Sqrt => x.sqrt(),
                Unary::Rsqrt => 1.0 / x.sqrt(),
                Unary::Neg => -x,
                Unary::Abs => x.abs(),
                Unary::Sign => {
                    if x > 0.0 {
                        1.0
                    } else if x < 0.0 {
                        -1.0
                    } else {
                        0.0
                    }
                }
                Unary::Not => unreachable!("taken above"),
            })
        }
    }
}

fn binary_u(op: Binary, a: u32, b: u32) -> V {
    match op {
        Binary::Add => V::U(a.wrapping_add(b)),
        Binary::Sub => V::U(a.wrapping_sub(b)),
        Binary::Mul => V::U(a.wrapping_mul(b)),
        Binary::Div => V::U(a.checked_div(b).unwrap_or(u32::MAX)),
        Binary::Rem => V::U(a.checked_rem(b).unwrap_or(a)),
        Binary::Max => V::U(a.max(b)),
        Binary::Min => V::U(a.min(b)),
        Binary::Gt => V::B(a > b),
        Binary::Ge => V::B(a >= b),
        Binary::Eq => V::B(a == b),
        Binary::Ne => V::B(a != b),
        Binary::Lt => V::B(a < b),
        Binary::Le => V::B(a <= b),
        Binary::And | Binary::Or | Binary::Pow => unreachable!("no op over u32s"),
    }
}

fn binary(op: Binary, a: V, b: V) -> V {
    match op {
        Binary::And => V::B(a.b() && b.b()),
        Binary::Or => V::B(a.b() || b.b()),
        op if op != Binary::Pow && matches!((a, b), (V::U(_), V::U(_))) => {
            binary_u(op, a.u(), b.u())
        }
        op => {
            let (a, b) = (a.f(), b.f());
            match op {
                Binary::Add => V::F(a + b),
                Binary::Sub => V::F(a - b),
                Binary::Mul => V::F(a * b),
                Binary::Div => V::F(a / b),
                Binary::Max => V::F(a.max(b)),
                Binary::Min => V::F(a.min(b)),
                Binary::Rem => V::F(a % b),
                Binary::Gt => V::B(a > b),
                Binary::Ge => V::B(a >= b),
                Binary::Eq => V::B(a == b),
                Binary::Ne => V::B(a != b),
                Binary::Lt => V::B(a < b),
                Binary::Le => V::B(a <= b),
                Binary::Pow => V::F(a.powf(b)),
                Binary::And | Binary::Or => unreachable!("taken above"),
            }
        }
    }
}

fn cast(x: V, to: Elem) -> V {
    match to {
        Elem::F32 => V::F(x.f()),
        Elem::Bool => V::B(x.b()),
        Elem::U32 => V::U(x.u()),
    }
}

/// The reduction's identity.
#[must_use]
pub fn identity(op: Reduce) -> f64 {
    match op {
        Reduce::Sum => 0.0,
        Reduce::Max => f64::NEG_INFINITY,
        Reduce::Min => f64::INFINITY,
    }
}

/// Two of a reduction's values combined, in f64.
#[must_use]
pub fn combine(op: Reduce, a: f64, b: f64) -> f64 {
    match op {
        Reduce::Sum => a + b,
        Reduce::Max => a.max(b),
        Reduce::Min => a.min(b),
    }
}

/// `xs` combined in order, in f64.
fn fold(op: Reduce, xs: impl Iterator<Item = f32>) -> f64 {
    xs.fold(identity(op), |acc, x| combine(op, acc, f64::from(x)))
}

/// Partials combined over runs of `seg`, in place: index `i` joined with
/// `i ^ off`, `off` halving from `seg / 2`.
pub fn butterfly(op: Reduce, seg: u32, partials: &mut [f64]) {
    let mut off = (seg / 2) as usize;
    while off > 0 {
        let before = partials.to_vec();
        for (i, x) in partials.iter_mut().enumerate() {
            *x = combine(op, before[i], before[i ^ off]);
        }
        off /= 2;
    }
}

#[allow(clippy::cast_possible_truncation)]
fn narrowed(x: f64) -> f32 {
    x as f32
}

fn rounded(dtype: Dtype, x: f32) -> Result<f32, InterpError> {
    match dtype {
        Dtype::Bf16 => Ok(round_bf16(x)),
        other => Err(InterpError::Rounding(other)),
    }
}

/// The region at one row: `args[k]` is parameter `k`'s row (or scalar), and
/// `results[k]`'s row comes back at `k`. Each value rounds where it is
/// defined; a reduction sums in f64 and rounds to f32 once.
pub fn eval(region: &Region, args: &[Arg]) -> Result<Vec<Vec<f32>>, InterpError> {
    let types = region.types().map_err(InterpError::Region)?;
    let width = region.width as usize;
    if args.len() != region.params.len() {
        return Err(InterpError::Arguments {
            param: u32::try_from(args.len().min(region.params.len())).unwrap_or(u32::MAX),
        });
    }
    // Each value as a row of elements, a scalar its one.
    let mut vals: Vec<Vec<V>> = Vec::with_capacity(types.len());
    for (param, (arg, shape)) in (0u32..).zip(args.iter().zip(&region.params)) {
        vals.push(match (arg, shape) {
            (Arg::Row(row), Shape::Row) if row.len() == width => {
                row.iter().map(|&x| V::F(x)).collect()
            }
            (Arg::Scalar(x), Shape::Scalar) => vec![V::F(*x)],
            _ => return Err(InterpError::Arguments { param }),
        });
    }
    let at = |vals: &[Vec<V>], v: ValueId, e: usize| -> V {
        let row = &vals[v as usize];
        if row.len() == 1 { row[0] } else { row[e] }
    };
    for (op, k) in region.ops.iter().zip(0u32..) {
        let out = region.params.len() as u32 + k;
        let n = match types[out as usize].0 {
            Shape::Row | Shape::Head(_) => width,
            Shape::Scalar => 1,
        };
        let mut row: Vec<V> = match *op {
            Op::Reduce(reduce, input) => {
                let total = fold(reduce, (0..width).map(|e| at(&vals, input, e).f()));
                vec![V::F(narrowed(total))]
            }
            Op::ReduceHeads(reduce, input, size) => {
                let mut partials: Vec<f64> = (0..width / 8)
                    .map(|p| fold(reduce, (p * 8..p * 8 + 8).map(|e| at(&vals, input, e).f())))
                    .collect();
                butterfly(reduce, size / 8, &mut partials);
                (0..width)
                    .map(|e| V::F(narrowed(partials[e / 8])))
                    .collect()
            }
            _ => (0..n)
                .map(|e| match *op {
                    Op::Const(l) => literal(l),
                    Op::Unary(u, a) => unary(u, at(&vals, a, e)),
                    Op::Binary(b, x, y) => binary(b, at(&vals, x, e), at(&vals, y, e)),
                    Op::Select(c, a, b) => {
                        if at(&vals, c, e).b() {
                            at(&vals, a, e)
                        } else {
                            at(&vals, b, e)
                        }
                    }
                    Op::Cast(a, to) => cast(at(&vals, a, e), to),
                    Op::Broadcast(a) => at(&vals, a, e),
                    Op::Fma(a, b, c) => V::F(
                        at(&vals, a, e)
                            .f()
                            .mul_add(at(&vals, b, e).f(), at(&vals, c, e).f()),
                    ),
                    Op::Column => V::U(u32::try_from(e).unwrap_or(u32::MAX)),
                    Op::Peer(src, idx) => {
                        let c = at(&vals, idx, e).u() as usize;
                        at(&vals, src, if c < width { c } else { 0 })
                    }
                    Op::Reduce(..) | Op::ReduceHeads(..) => unreachable!("taken above"),
                })
                .collect(),
        };
        if let Some(dtype) = region.rounding(out) {
            for x in &mut row {
                *x = V::F(rounded(dtype, x.f())?);
            }
        }
        vals.push(row);
    }
    Ok(region
        .results
        .iter()
        .map(|&r| (0..width).map(|e| at(&vals, r, e).f()).collect())
        .collect())
}

/// The frame of one row on the host: one thread holding the whole row, its
/// results' planes each of the dtype the region rounds them to (f32 where
/// it does not).
#[derive(Clone, Debug)]
pub struct OneRow {
    pub rows: Vec<Vec<f32>>,
    pub ones: Vec<f32>,
    pub stored: Vec<Option<Dtype>>,
    pub out: Vec<Vec<f32>>,
    pub width: usize,
    pub shared: Vec<f32>,
}

impl OneRow {
    /// The frame over `args`, the results of `region`.
    pub fn of(region: &Region, args: &[Arg]) -> Result<OneRow, InterpError> {
        let width = region.width as usize;
        let mut rows = Vec::new();
        let mut ones = Vec::new();
        for (param, (arg, shape)) in (0u32..).zip(args.iter().zip(&region.params)) {
            match (arg, shape) {
                (Arg::Row(row), Shape::Row) if row.len() == width => {
                    rows.push(row.clone());
                    ones.push(0.0);
                }
                (Arg::Scalar(x), Shape::Scalar) => {
                    rows.push(Vec::new());
                    ones.push(*x);
                }
                _ => return Err(InterpError::Arguments { param }),
            }
        }
        Ok(OneRow {
            rows,
            ones,
            stored: region.results.iter().map(|&r| region.rounding(r)).collect(),
            out: vec![vec![0.0; width]; region.results.len()],
            width,
            shared: vec![0.0; width],
        })
    }
}

impl Frame for OneRow {
    fn elements(&self) -> usize {
        self.width
    }

    fn valid(&self, e: usize) -> bool {
        e < self.width
    }

    fn load(&self, k: u32, e: usize) -> f32 {
        self.rows[k as usize][e]
    }

    fn one(&self, k: u32) -> f32 {
        self.ones[k as usize]
    }

    fn store(&mut self, k: u32, e: usize, v: f32) -> f32 {
        let v = match self.stored[k as usize] {
            Some(Dtype::Bf16) => round_bf16(v),
            _ => v,
        };
        self.out[k as usize][e] = v;
        v
    }

    fn flush(&mut self, _k: u32) {}

    fn reduce(&mut self, _op: Reduce, partial: f64) -> f32 {
        narrowed(partial)
    }

    fn round(&self, dtype: Dtype, v: f32) -> f32 {
        match dtype {
            Dtype::Bf16 => round_bf16(v),
            _ => v,
        }
    }

    fn column(&self, e: usize) -> u32 {
        u32::try_from(e).unwrap_or(u32::MAX)
    }

    fn reduce_heads(&mut self, op: Reduce, seg: u32, partials: &mut [f64]) {
        butterfly(op, seg, partials);
    }

    fn share(&mut self, e: usize, v: f32) {
        self.shared[e] = v;
    }

    fn peer(&self, c: u32) -> f32 {
        self.shared
            .get(c as usize)
            .copied()
            .unwrap_or(self.shared[0])
    }
}

/// What a kernel has computed so far, by how it holds it.
#[derive(Default)]
struct State {
    uniforms: BTreeMap<ValueId, V>,
    kept: BTreeMap<ValueId, Vec<V>>,
    heads: BTreeMap<ValueId, Vec<V>>,
    table: Vec<f32>,
}

/// `kernel` run over `frame`, phase by phase: what a printed kernel does on
/// a device, on the host.
pub fn run(kernel: &Kernel, frame: &mut impl Frame) {
    let n = frame.elements();
    let mut st = State {
        kept: kernel
            .kept
            .iter()
            .map(|(v, _)| (*v, vec![V::F(0.0); n]))
            .collect(),
        heads: kernel
            .heads
            .iter()
            .map(|(v, _)| (*v, vec![V::F(0.0); n.div_ceil(8)]))
            .collect(),
        table: vec![0.0; kernel.table as usize],
        ..State::default()
    };
    let none = BTreeMap::new();
    for (at, phase) in kernel.phases.iter().enumerate() {
        for u in &phase.uniforms {
            let mut x = eval_expr(&u.expr, frame, &st, &none, 0);
            if let Some(dtype) = u.round {
                x = V::F(frame.round(dtype, x.f()));
            }
            st.uniforms.insert(u.value, x);
        }
        if at == 0 {
            for t in &kernel.tables {
                fill(t, frame, &mut st);
            }
        }
        for h in &phase.heads {
            for p in 0..n.div_ceil(8) {
                let mut x = eval_expr(&h.expr, frame, &st, &none, p * 8);
                if let Some(dtype) = h.round {
                    x = V::F(frame.round(dtype, x.f()));
                }
                st.heads.get_mut(&h.value).expect("a value one a head")[p] = x;
            }
        }
        let Some(pass) = &phase.pass else {
            continue;
        };
        let mut acc: Vec<Vec<f64>> = pass
            .partials
            .iter()
            .map(|p| vec![identity(p.op); if p.seg.is_some() { n.div_ceil(8) } else { 1 }])
            .collect();
        for e in 0..n {
            let mut local: BTreeMap<ValueId, V> = BTreeMap::new();
            for row in &pass.rows {
                let mut x = eval_expr(&row.expr, frame, &st, &local, e);
                if let Some(dtype) = row.round {
                    x = V::F(frame.round(dtype, x.f()));
                }
                if let Some(k) = row.store {
                    x = V::F(frame.store(k, e, x.f()));
                }
                for &k in &row.also {
                    frame.store(k, e, x.f());
                }
                if row.kept {
                    st.kept.get_mut(&row.value).expect("a kept value")[e] = x;
                }
                local.insert(row.value, x);
            }
            for &(k, from) in &pass.copies {
                let x = frame.load(from, e);
                frame.store(k, e, x);
            }
            for (i, p) in pass.partials.iter().enumerate() {
                let x = operand(p.input, frame, &st, &local, e).f();
                let slot = if p.seg.is_some() { e / 8 } else { 0 };
                if frame.valid(e) {
                    acc[i][slot] = combine(p.op, acc[i][slot], f64::from(x));
                }
            }
            for &share in &pass.shares {
                let x = operand(share, frame, &st, &local, e).f();
                frame.share(e, x);
            }
        }
        for &k in &pass.flushes {
            frame.flush(k);
        }
        for (i, t) in pass.totals.iter().enumerate() {
            if let Some(seg) = t.seg {
                frame.reduce_heads(t.op, seg, &mut acc[i]);
                let heads: Vec<V> = acc[i]
                    .iter()
                    .map(|&x| narrowed(x))
                    .map(|x| V::F(t.round.map_or(x, |dtype| frame.round(dtype, x))))
                    .collect();
                st.heads.insert(t.value, heads);
                continue;
            }
            let mut x = frame.reduce(t.op, acc[i][0]);
            if let Some(dtype) = t.round {
                x = frame.round(dtype, x);
            }
            st.uniforms.insert(t.value, V::F(x));
        }
    }
}

/// A table's entries, each computed from its column.
fn fill(t: &Table, frame: &impl Frame, st: &mut State) {
    for c in 0..t.period {
        let mut entries: BTreeMap<ValueId, V> = BTreeMap::new();
        for &(value, _, expr) in &t.entries {
            let x = entry_expr(&expr, st, &entries, c);
            entries.insert(value, x);
        }
        st.table[(t.at + c) as usize] = entries[&t.value].f();
    }
    let _ = frame;
}

fn entry_expr(expr: &Expr, st: &State, entries: &BTreeMap<ValueId, V>, c: u32) -> V {
    let o = |x: Operand| match x {
        Operand::Entry(v) => entries[&v],
        Operand::Local(v) => st.uniforms[&v],
        other => unreachable!("a table reads its entries and uniforms, not {other:?}"),
    };
    match *expr {
        Expr::Column => V::U(c),
        _ => apply(expr, o),
    }
}

/// An expression over its operands' values (a column or peer aside).
fn apply(expr: &Expr, o: impl Fn(Operand) -> V) -> V {
    match *expr {
        Expr::Literal(l) => literal(l),
        Expr::Copy(a) => o(a),
        Expr::Unary(u, a) => unary(u, o(a)),
        Expr::Binary(b, x, y) => binary(b, o(x), o(y)),
        Expr::Select(c, a, b) => {
            if o(c).b() {
                o(a)
            } else {
                o(b)
            }
        }
        Expr::Cast { value, to, .. } => cast(o(value), to),
        Expr::Fma(a, b, c) => V::F(o(a).f().mul_add(o(b).f(), o(c).f())),
        Expr::Column | Expr::Peer(_) => unreachable!("taken by the caller"),
    }
}

fn operand(
    o: Operand,
    frame: &impl Frame,
    st: &State,
    local: &BTreeMap<ValueId, V>,
    e: usize,
) -> V {
    match o {
        Operand::Param(k) => V::F(frame.load(k, e)),
        Operand::One(k) => V::F(frame.one(k)),
        Operand::Kept(v) => st.kept[&v][e],
        Operand::Heads(v) => st.heads[&v][e / 8],
        Operand::Table { at, period } => V::F(st.table[(at + frame.column(e) % period) as usize]),
        Operand::Local(v) | Operand::Entry(v) => local
            .get(&v)
            .or_else(|| st.uniforms.get(&v))
            .copied()
            .expect("a local value its pass computed or a uniform an earlier phase did"),
    }
}

fn eval_expr(
    expr: &Expr,
    frame: &impl Frame,
    st: &State,
    local: &BTreeMap<ValueId, V>,
    e: usize,
) -> V {
    match *expr {
        Expr::Column => V::U(frame.column(e)),
        Expr::Peer(at) => V::F(frame.peer(operand(at, frame, st, local, e).u())),
        _ => apply(expr, |x| operand(x, frame, st, local, e)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schedule::schedule;

    /// `h = y + x`, `rmsnorm(h) · w`, as poem-compiler lowers a residual
    /// norm, the stream and the normed row rounded to bf16.
    fn residual_norm(width: u32) -> Region {
        Region {
            width,
            params: vec![Shape::Row; 3],
            ops: vec![
                Op::Binary(Binary::Add, 1, 0),
                Op::Binary(Binary::Mul, 3, 3),
                Op::Reduce(Reduce::Sum, 4),
                Op::Const(Literal::F32(width as f32)),
                Op::Binary(Binary::Div, 5, 6),
                Op::Const(Literal::F32(1e-6)),
                Op::Binary(Binary::Add, 7, 8),
                Op::Unary(Unary::Rsqrt, 9),
                Op::Binary(Binary::Mul, 3, 10),
                Op::Binary(Binary::Mul, 11, 2),
            ],
            results: vec![3, 12],
            rounds: vec![(3, Dtype::Bf16), (12, Dtype::Bf16)],
        }
    }

    fn row(width: u32, seed: u32) -> Vec<f32> {
        (0..width)
            .map(|i| round_bf16(((i * 37 + seed * 11) % 23) as f32 / 8.0 - 1.4))
            .collect()
    }

    fn both(region: &Region, args: &[Arg]) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let want = eval(region, args).expect("the region runs");
        let kernel = schedule(region).expect("the row core schedules it");
        let mut frame = OneRow::of(region, args).expect("a frame of the arguments");
        run(&kernel, &mut frame);
        (frame.out, want)
    }

    fn bits(rows: &[Vec<f32>]) -> Vec<Vec<u32>> {
        rows.iter()
            .map(|row| row.iter().map(|x| x.to_bits()).collect())
            .collect()
    }

    #[test]
    fn a_scheduled_residual_norm_lands_its_regions_bits() {
        for width in [8, 64, 1544] {
            let region = residual_norm(width);
            let args: Vec<Arg> = (1..=3).map(|s| Arg::Row(row(width, s))).collect();
            let (got, want) = both(&region, &args);
            assert_eq!(bits(&got), bits(&want), "{width}");
            let kernel = schedule(&region).expect("scheduled");
            assert_eq!(kernel.passes(), 2, "the row held across its sum");
            assert_eq!(
                kernel.kept,
                vec![(3, Elem::F32)],
                "the stream kept in registers"
            );
        }
    }

    /// A norm over a value rounded where it is defined, a scalar parameter,
    /// a maximum, a select, a result that is a parameter copied through and
    /// one stored twice.
    #[test]
    fn every_step_kind_lands_its_regions_bits() {
        let width = 64;
        let region = Region {
            width,
            params: vec![Shape::Row, Shape::Row, Shape::Scalar],
            ops: vec![
                Op::Binary(Binary::Mul, 0, 2),   // 3: x · s, rounded
                Op::Reduce(Reduce::Max, 3),      // 4
                Op::Broadcast(4),                // 5
                Op::Binary(Binary::Sub, 3, 5),   // 6
                Op::Unary(Unary::Exp, 6),        // 7
                Op::Reduce(Reduce::Sum, 7),      // 8
                Op::Binary(Binary::Div, 7, 8),   // 9: softmax
                Op::Binary(Binary::Gt, 1, 6),    // 10
                Op::Select(10, 9, 1),            // 11
                Op::Reduce(Reduce::Min, 11),     // 12
                Op::Binary(Binary::Add, 11, 12), // 13
            ],
            results: vec![13, 1, 13],
            rounds: vec![(3, Dtype::Bf16), (13, Dtype::Bf16)],
        };
        let args = vec![
            Arg::Row(row(width, 4)),
            Arg::Row(row(width, 5)),
            Arg::Scalar(0.75),
        ];
        let (got, want) = both(&region, &args);
        assert_eq!(bits(&got), bits(&want));
        let kernel = schedule(&region).expect("scheduled");
        assert_eq!(kernel.passes(), 4);
    }

    /// Each head of `hd` normed by its own mean square, then rotated by a
    /// position over its first `rot` columns (NeoX halves), the rest kept.
    fn head_norm_rope(heads: u32, hd: u32, rot: u32) -> Region {
        let half = rot / 2;
        Region {
            width: heads * hd,
            params: vec![Shape::Row, Shape::Row, Shape::Scalar],
            ops: vec![
                Op::Binary(Binary::Mul, 0, 0),        // 3
                Op::ReduceHeads(Reduce::Sum, 3, hd),  // 4
                Op::Const(Literal::F32(hd as f32)),   // 5
                Op::Binary(Binary::Div, 4, 5),        // 6
                Op::Const(Literal::F32(1e-6)),        // 7
                Op::Binary(Binary::Add, 6, 7),        // 8
                Op::Unary(Unary::Rsqrt, 8),           // 9
                Op::Broadcast(9),                     // 10
                Op::Binary(Binary::Mul, 0, 10),       // 11
                Op::Binary(Binary::Mul, 11, 1),       // 12: normed, bf16
                Op::Column,                           // 13
                Op::Const(Literal::U32(hd)),          // 14
                Op::Binary(Binary::Rem, 13, 14),      // 15: i
                Op::Const(Literal::U32(half)),        // 16
                Op::Binary(Binary::Rem, 15, 16),      // 17: d
                Op::Binary(Binary::Lt, 15, 16),       // 18: first half
                Op::Cast(17, Elem::F32),              // 19
                Op::Unary(Unary::Neg, 19),            // 20
                Op::Const(Literal::F32(half as f32)), // 21
                Op::Binary(Binary::Div, 20, 21),      // 22
                Op::Const(Literal::F32(10000.0)),     // 23
                Op::Binary(Binary::Pow, 23, 22),      // 24: freq
                Op::Binary(Binary::Mul, 2, 24),       // 25: angle
                Op::Unary(Unary::Cos, 25),            // 26
                Op::Unary(Unary::Sin, 25),            // 27
                Op::Binary(Binary::Add, 13, 16),      // 28
                Op::Binary(Binary::Sub, 13, 16),      // 29
                Op::Select(18, 28, 29),               // 30: mate
                Op::Peer(12, 30),                     // 31
                Op::Binary(Binary::Mul, 12, 26),      // 32
                Op::Binary(Binary::Mul, 31, 27),      // 33
                Op::Binary(Binary::Sub, 32, 33),      // 34
                Op::Binary(Binary::Add, 32, 33),      // 35
                Op::Select(18, 34, 35),               // 36
                Op::Const(Literal::U32(rot)),         // 37
                Op::Binary(Binary::Lt, 15, 37),       // 38
                Op::Select(38, 36, 12),               // 39
            ],
            results: vec![39],
            rounds: vec![(12, Dtype::Bf16), (39, Dtype::Bf16)],
        }
    }

    #[test]
    fn a_scheduled_head_norm_rope_lands_its_regions_bits() {
        for (heads, hd, rot) in [(2, 64, 64), (4, 128, 64), (1, 256, 256)] {
            let region = head_norm_rope(heads, hd, rot);
            let width = heads * hd;
            let args = vec![
                Arg::Row(row(width, 1)),
                Arg::Row((0..width).map(|c| row(hd, 2)[(c % hd) as usize]).collect()),
                Arg::Scalar(37.0),
            ];
            let (got, want) = both(&region, &args);
            assert_eq!(bits(&got), bits(&want), "{heads}x{hd}/{rot}");
            let kernel = schedule(&region).expect("scheduled");
            assert_eq!(kernel.tables.len(), 1, "the frequencies tabled");
            assert_eq!(kernel.tables[0].period, rot / 2);
            assert!(kernel.shares, "the mate gathered within the row");
            assert_eq!(
                kernel.heads.len(),
                4,
                "the mean square to its rsqrt, one a head"
            );
        }
    }

    #[test]
    fn a_scalar_result_is_not_scheduled() {
        let mut region = residual_norm(64);
        region.results = vec![9];
        assert!(schedule(&region).is_err());
    }

    #[test]
    fn a_regions_key_is_its_function() {
        let (a, b) = (residual_norm(64), residual_norm(64));
        assert_eq!(a.key(), b.key());
        assert_ne!(a.key(), residual_norm(128).key());
    }
}

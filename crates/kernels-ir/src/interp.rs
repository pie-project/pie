//! The reference interpreter of both forms: a region op by op, and a
//! kernel phase by phase over a frame. Each takes the same math (the host's
//! f32) and the same reduction order (a row's elements two at a time, in
//! order), so a kernel lands its region's bits: the schedule moves where a
//! value is computed and kept, never what it is.

use std::collections::BTreeMap;
use std::fmt;

use dtype::Dtype;

use crate::frame::{Frame, round_bf16};
use crate::kernel::{Expr, Kernel, Operand};
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
}

impl V {
    fn f(self) -> f32 {
        match self {
            V::F(x) => x,
            V::B(b) => f32::from(u8::from(b)),
        }
    }

    fn b(self) -> bool {
        match self {
            V::B(b) => b,
            V::F(x) => x != 0.0,
        }
    }
}

fn literal(l: Literal) -> V {
    match l {
        Literal::F32(x) => V::F(x),
        Literal::Bool(b) => V::B(b),
    }
}

/// The host's math for one op: what every interpreted kernel computes.
fn unary(op: Unary, x: V) -> V {
    match op {
        Unary::Not => V::B(!x.b()),
        op => {
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

fn binary(op: Binary, a: V, b: V) -> V {
    match op {
        Binary::And => V::B(a.b() && b.b()),
        Binary::Or => V::B(a.b() || b.b()),
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
                Binary::And | Binary::Or => unreachable!("taken above"),
            }
        }
    }
}

fn cast(x: V, to: Elem) -> V {
    match to {
        Elem::F32 => V::F(x.f()),
        Elem::Bool => V::B(x.b()),
    }
}

/// The reduction's identity.
#[must_use]
pub fn identity(op: Reduce) -> f32 {
    match op {
        Reduce::Sum => 0.0,
        Reduce::Max => f32::NEG_INFINITY,
        Reduce::Min => f32::INFINITY,
    }
}

/// Two of a reduction's values combined.
#[must_use]
pub fn combine(op: Reduce, a: f32, b: f32) -> f32 {
    match op {
        Reduce::Sum => a + b,
        Reduce::Max => a.max(b),
        Reduce::Min => a.min(b),
    }
}

fn rounded(dtype: Dtype, x: f32) -> Result<f32, InterpError> {
    match dtype {
        Dtype::Bf16 => Ok(round_bf16(x)),
        other => Err(InterpError::Rounding(other)),
    }
}

/// The region at one row: `args[k]` is parameter `k`'s row (or scalar), and
/// `results[k]`'s row comes back at `k`. Each value rounds where it is
/// defined; a reduction takes the row two at a time, in order.
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
            Shape::Row => width,
            Shape::Scalar => 1,
        };
        let mut row: Vec<V> = match *op {
            Op::Reduce(reduce, input) => {
                let mut acc = identity(reduce);
                let mut pending = 0.0;
                for e in 0..width {
                    let x = at(&vals, input, e).f();
                    if e % 2 == 0 {
                        pending = x;
                    } else {
                        acc = combine(reduce, acc, combine(reduce, pending, x));
                    }
                }
                vec![V::F(acc)]
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
                    Op::Reduce(..) => unreachable!("taken above"),
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

    fn reduce(&mut self, _op: Reduce, partial: f32) -> f32 {
        partial
    }

    fn round(&self, dtype: Dtype, v: f32) -> f32 {
        match dtype {
            Dtype::Bf16 => round_bf16(v),
            _ => v,
        }
    }
}

/// `kernel` run over `frame`, phase by phase: what a printed kernel does on
/// a device, on the host.
pub fn run(kernel: &Kernel, frame: &mut impl Frame) {
    let n = frame.elements();
    let mut uniforms: BTreeMap<ValueId, V> = BTreeMap::new();
    let mut kept: BTreeMap<ValueId, Vec<V>> = kernel
        .kept
        .iter()
        .map(|(v, _)| (*v, vec![V::F(0.0); n]))
        .collect();
    for phase in &kernel.phases {
        for u in &phase.uniforms {
            let mut x = eval_expr(&u.expr, frame, &uniforms, &BTreeMap::new(), &kept, 0);
            if let Some(dtype) = u.round {
                x = V::F(frame.round(dtype, x.f()));
            }
            uniforms.insert(u.value, x);
        }
        let Some(pass) = &phase.pass else {
            continue;
        };
        let mut acc: Vec<f32> = pass.partials.iter().map(|p| identity(p.op)).collect();
        let mut pending: Vec<f32> = pass.partials.iter().map(|p| identity(p.op)).collect();
        for e in 0..n {
            let mut local: BTreeMap<ValueId, V> = BTreeMap::new();
            for row in &pass.rows {
                let mut x = eval_expr(&row.expr, frame, &uniforms, &local, &kept, e);
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
                    kept.get_mut(&row.value).expect("a kept value")[e] = x;
                }
                local.insert(row.value, x);
            }
            for &(k, from) in &pass.copies {
                let x = frame.load(from, e);
                frame.store(k, e, x);
            }
            for (i, p) in pass.partials.iter().enumerate() {
                let x = operand(p.input, frame, &uniforms, &local, &kept, e).f();
                if e % 2 == 0 {
                    pending[i] = x;
                } else if frame.valid(e) {
                    acc[i] = combine(p.op, acc[i], combine(p.op, pending[i], x));
                }
            }
        }
        for &k in &pass.flushes {
            frame.flush(k);
        }
        for (i, t) in pass.totals.iter().enumerate() {
            let mut x = frame.reduce(t.op, acc[i]);
            if let Some(dtype) = t.round {
                x = frame.round(dtype, x);
            }
            uniforms.insert(t.value, V::F(x));
        }
    }
}

fn operand(
    o: Operand,
    frame: &impl Frame,
    uniforms: &BTreeMap<ValueId, V>,
    local: &BTreeMap<ValueId, V>,
    kept: &BTreeMap<ValueId, Vec<V>>,
    e: usize,
) -> V {
    match o {
        Operand::Param(k) => V::F(frame.load(k, e)),
        Operand::One(k) => V::F(frame.one(k)),
        Operand::Kept(v) => kept[&v][e],
        Operand::Local(v) => local
            .get(&v)
            .or_else(|| uniforms.get(&v))
            .copied()
            .expect("a local value its pass computed or a uniform an earlier phase did"),
    }
}

fn eval_expr(
    expr: &Expr,
    frame: &impl Frame,
    uniforms: &BTreeMap<ValueId, V>,
    local: &BTreeMap<ValueId, V>,
    kept: &BTreeMap<ValueId, Vec<V>>,
    e: usize,
) -> V {
    let o = |x: Operand| operand(x, frame, uniforms, local, kept, e);
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

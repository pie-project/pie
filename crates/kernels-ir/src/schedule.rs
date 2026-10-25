//! Region to kernel: the row core's plan.
//!
//! Each op is a row op (computed per element, in the pass of its latest
//! operand), a uniform (one a row, between passes) or a reduction (its
//! partials accumulated in its input's pass, its total one phase on). A
//! pass does not end at the first reader of a reduction: the row stays in
//! registers across it, so a residual norm is two passes over registers,
//! not two over memory. A row value read in a later phase than its own is
//! kept in registers; one read only in its own is local to that pass.

use std::fmt;

use dtype::Dtype;

use crate::kernel::{Expr, Kernel, Loop, Operand, Partial, Phase, RowAssign, Total, Uniform};
use crate::region::{Elem, Op, Region, Shape, ValueId};

/// Why the row core does not schedule a region: the op it stops at (`None`
/// for the region as a whole) and what it lacks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unscheduled {
    pub op: Option<u32>,
    pub why: &'static str,
}

impl fmt::Display for Unscheduled {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.op {
            Some(at) => write!(f, "op {at}: {}", self.why),
            None => f.write_str(self.why),
        }
    }
}

impl std::error::Error for Unscheduled {}

/// How the plan holds a value.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Held {
    Param(u32),
    One(u32),
    Uniform,
    Local,
    Kept,
}

#[derive(Clone, Copy)]
struct Value {
    phase: u32,
    held: Held,
}

enum Step {
    Uniform { value: ValueId, op: usize },
    Row { value: ValueId, op: usize },
    Reduce { value: ValueId, op: usize },
}

fn whole(why: &'static str) -> Unscheduled {
    Unscheduled { op: None, why }
}

/// `region` scheduled into passes over the row, or why the row core does
/// not cover it: every result an f32 row, every rounding of a value that
/// does not leave one the frame rounds to (bf16).
pub fn schedule(region: &Region) -> Result<Kernel, Unscheduled> {
    let types = region
        .types()
        .map_err(|_| whole("the region is not a function of one row"))?;
    let params = u32::try_from(region.params.len()).unwrap_or(u32::MAX);
    let results = &region.results;
    if region
        .rounds
        .iter()
        .any(|(v, dtype)| !results.contains(v) && *dtype != Dtype::Bf16)
    {
        return Err(whole("a value rounded to a dtype no hook rounds to"));
    }
    for &r in results {
        if types[r as usize] != (Shape::Row, Elem::F32) {
            return Err(whole(
                "a result that is not an f32 row, and `f.st` writes one",
            ));
        }
    }
    let count = types.len();
    let mut values: Vec<Option<Value>> = vec![None; count];
    for (k, shape) in (0u32..).zip(&region.params) {
        values[k as usize] = Some(Value {
            phase: 0,
            held: match shape {
                Shape::Row => Held::Param(k),
                Shape::Scalar => Held::One(k),
            },
        });
    }

    // Each op's step and phase, in program order.
    let mut steps: Vec<(u32, Step)> = Vec::new();
    for (at, op) in region.ops.iter().enumerate() {
        let out = params + u32::try_from(at).unwrap_or(u32::MAX);
        let phase = op
            .args()
            .iter()
            .map(|&a| values[a as usize].map_or(0, |v| v.phase))
            .max()
            .unwrap_or(0);
        let (step, value, step_phase) = match (op, types[out as usize].0) {
            (Op::Reduce(..), _) => (
                Step::Reduce { value: out, op: at },
                Value {
                    phase: phase + 1,
                    held: Held::Uniform,
                },
                phase,
            ),
            (_, Shape::Row) => (
                Step::Row { value: out, op: at },
                Value {
                    phase,
                    held: Held::Local,
                },
                phase,
            ),
            (_, Shape::Scalar) => (
                Step::Uniform { value: out, op: at },
                Value {
                    phase,
                    held: Held::Uniform,
                },
                phase,
            ),
        };
        values[out as usize] = Some(value);
        steps.push((step_phase, step));
    }

    // A row value read in a later phase than its own is kept.
    for (phase, step) in &steps {
        let op = match step {
            Step::Row { op, .. } | Step::Uniform { op, .. } | Step::Reduce { op, .. } => *op,
        };
        for a in region.ops[op].args() {
            if let Some(v) = values[a as usize].as_mut()
                && v.held == Held::Local
                && v.phase < *phase
            {
                v.held = Held::Kept;
            }
        }
    }
    let mut readers = vec![false; count];
    for op in &region.ops {
        for a in op.args() {
            readers[a as usize] = true;
        }
    }
    let last = values.iter().flatten().map(|v| v.phase).max().unwrap_or(0);

    let operand = |v: ValueId| -> Operand {
        match values[v as usize].map(|v| v.held) {
            Some(Held::Param(k)) => Operand::Param(k),
            Some(Held::One(k)) => Operand::One(k),
            Some(Held::Kept) => Operand::Kept(v),
            _ => Operand::Local(v),
        }
    };
    let expr = |op: &Op, out: ValueId| -> Expr {
        match *op {
            Op::Const(literal) => Expr::Literal(literal),
            Op::Broadcast(a) => Expr::Copy(operand(a)),
            Op::Unary(unary, a) => Expr::Unary(unary, operand(a)),
            Op::Binary(binary, a, b) => Expr::Binary(binary, operand(a), operand(b)),
            Op::Select(c, a, b) => Expr::Select(operand(c), operand(a), operand(b)),
            Op::Cast(a, _) => Expr::Cast {
                value: operand(a),
                from: types[a as usize].1,
                to: types[out as usize].1,
            },
            Op::Reduce(..) => unreachable!("a reduction is no expression"),
        }
    };
    // A result rounds by its store; any other value where it is defined.
    let round = |v: ValueId| -> Option<Dtype> {
        if results.contains(&v) {
            None
        } else {
            region.rounding(v)
        }
    };

    let mut phases = Vec::new();
    for phase in 0..=last {
        let mut uniforms = Vec::new();
        let mut partials = Vec::new();
        let mut totals = Vec::new();
        let mut rows = Vec::new();
        for (p, step) in &steps {
            if *p != phase {
                continue;
            }
            match *step {
                Step::Uniform { value, op } => uniforms.push(Uniform {
                    value,
                    elem: types[value as usize].1,
                    expr: expr(&region.ops[op], value),
                    round: round(value),
                }),
                Step::Reduce { value, op } => {
                    let Op::Reduce(reduce, input) = region.ops[op] else {
                        unreachable!("a reduction's step is a reduction");
                    };
                    partials.push(Partial {
                        value,
                        op: reduce,
                        input: operand(input),
                    });
                    totals.push(Total {
                        value,
                        op: reduce,
                        round: round(value),
                    });
                }
                Step::Row { value, op } => {
                    let mut stores = results
                        .iter()
                        .enumerate()
                        .filter(|(_, r)| **r == value)
                        .map(|(k, _)| u32::try_from(k).unwrap_or(u32::MAX));
                    let store = stores.next();
                    rows.push(RowAssign {
                        value,
                        elem: types[value as usize].1,
                        expr: expr(&region.ops[op], value),
                        round: round(value),
                        store,
                        also: stores.collect(),
                        kept: values[value as usize].is_some_and(|v| v.held == Held::Kept),
                        unread: !readers[value as usize],
                    });
                }
            }
        }
        // The results that land in this pass: its rows', and the
        // parameters a result copies through (in the first).
        let stored: Vec<(u32, ValueId)> = results
            .iter()
            .enumerate()
            .filter(|(_, r)| {
                values[**r as usize].is_some_and(|v| {
                    v.phase == phase && matches!(v.held, Held::Local | Held::Kept | Held::Param(_))
                })
            })
            .map(|(k, r)| (u32::try_from(k).unwrap_or(u32::MAX), *r))
            .collect();
        let pass = if rows.is_empty() && partials.is_empty() && stored.is_empty() {
            None
        } else {
            Some(Loop {
                copies: stored
                    .iter()
                    .filter_map(|(k, r)| match values[*r as usize].map(|v| v.held) {
                        Some(Held::Param(from)) => Some((*k, from)),
                        _ => None,
                    })
                    .collect(),
                flushes: stored.iter().map(|(k, _)| *k).collect(),
                partials,
                rows,
                totals,
            })
        };
        phases.push(Phase { uniforms, pass });
    }
    let kept = values
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_some_and(|v| v.held == Held::Kept))
        .map(|(at, _)| {
            let at = u32::try_from(at).unwrap_or(u32::MAX);
            (at, types[at as usize].1)
        })
        .collect();
    Ok(Kernel {
        params,
        results: u32::try_from(results.len()).unwrap_or(u32::MAX),
        kept,
        phases,
    })
}

//! Region to kernel: the row core's plan.
//!
//! Each op is a row op (computed per element), a uniform (one a row,
//! between passes), a value one a head (between passes, a vector at a
//! time) or a reduction (its partials accumulated in its input's pass, its
//! total one phase on). A pass does not end at the first reader of a
//! reduction: the row stays in registers across it, so a residual norm is
//! two passes over registers, not two over memory. A gather within the row
//! reads the row a pass shared, one phase on.
//!
//! - **Where a row value is computed.** In the phase its first reader runs
//!   in, not the first it could be: a column's index, a rotation's angle, a
//!   weight's copy are computed where they are read rather than held in
//!   registers across a reduction. A result is stored where it is
//!   computed, a gather's row shared where it is.
//! - **What is kept.** A row value read in a later phase than its own is
//!   kept in registers; one read only in its own lives in that pass.
//! - **What is tabled.** What a column alone computes dearly (a power, a
//!   transcendental), the same every `P` columns past a remainder by `P`,
//!   is computed once a block for its `P` columns and read back by column.

use std::fmt;

use dtype::Dtype;

use crate::kernel::{
    Expr, HeadAssign, Kernel, Loop, Operand, Partial, Phase, RowAssign, Table, Total, Uniform,
};
use crate::region::{Binary, Elem, Literal, Op, Region, Shape, Unary, ValueId};

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

/// The most entries one table holds.
const MAX_TABLE: u32 = 1024;

/// How the plan holds a value.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Held {
    Param(u32),
    One(u32),
    Vector(u32),
    Uniform,
    Heads,
    Local,
    Kept,
    Table { at: u32, period: u32 },
}

#[derive(Clone, Copy)]
struct Value {
    phase: u32,
    held: Held,
}

#[derive(Clone, Copy)]
enum Step {
    Uniform { value: ValueId, op: usize },
    Head { value: ValueId, op: usize },
    Row { value: ValueId, op: usize },
    Reduce { value: ValueId, op: usize },
    Share { src: ValueId },
}

/// How a value depends on the column alone: alike at every column (a
/// constant), each column's own, or the same every `P` columns.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Column {
    Alike,
    Each,
    Every(u32),
}

fn whole(why: &'static str) -> Unscheduled {
    Unscheduled { op: None, why }
}

/// `region` scheduled into passes over the row, or why the row core does
/// not cover it: every result an f32 row, every rounding of a value that
/// does not leave one the skeleton rounds to (bf16), a gather's row shared once
/// a phase.
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
        if !matches!(
            types[r as usize],
            (Shape::Row | Shape::Vector(_), Elem::F32)
        ) {
            return Err(whole(
                "a result that is not an f32 row or small vector, and `f.st` writes one",
            ));
        }
    }
    let count = types.len();
    let op_of =
        |v: ValueId| -> Option<&Op> { v.checked_sub(params).map(|at| &region.ops[at as usize]) };
    let mut values: Vec<Option<Value>> = vec![None; count];
    for (k, shape) in (0u32..).zip(&region.params) {
        values[k as usize] = Some(Value {
            phase: 0,
            held: match shape {
                Shape::Scalar => Held::One(k),
                Shape::Vector(_) => Held::Vector(k),
                _ => Held::Param(k),
            },
        });
    }

    // Each op's step and phase, in program order. A repeat's body runs
    // whole in one phase: the latest any value it reads from outside it (its
    // initial value among them) lands in.
    let mut steps: Vec<(u32, Step)> = Vec::new();
    let mut looping: Option<u32> = None;
    for (at, op) in region.ops.iter().enumerate() {
        let out = params + u32::try_from(at).unwrap_or(u32::MAX);
        let phase_of =
            |values: &[Option<Value>], v: ValueId| values[v as usize].map_or(0, |v| v.phase);
        if let Op::Carry(_) = op {
            let close = region.ops[at..]
                .iter()
                .position(|op| matches!(op, Op::Repeat(..)))
                .map_or(region.ops.len(), |k| at + k);
            looping = Some(
                region.ops[at..close]
                    .iter()
                    .flat_map(Op::args)
                    .filter(|&a| a < out)
                    .map(|a| phase_of(&values, a))
                    .max()
                    .unwrap_or(0),
            );
        }
        let phase = looping.unwrap_or_else(|| {
            op.args()
                .iter()
                .map(|&a| phase_of(&values, a))
                .max()
                .unwrap_or(0)
        });
        if let Op::Repeat(..) = op {
            looping = None;
        }
        let (step, value, step_phase) = match (op, types[out as usize].0) {
            (Op::Reduce(..) | Op::ReduceHeads(..), shape) => (
                Step::Reduce { value: out, op: at },
                Value {
                    phase: phase + 1,
                    held: if matches!(shape, Shape::Head(_)) {
                        Held::Heads
                    } else {
                        Held::Uniform
                    },
                },
                phase,
            ),
            (Op::Peer(src, _), _) => {
                // the row shared in its own phase, read back one on
                let shared = phase_of(&values, *src);
                steps.push((shared, Step::Share { src: *src }));
                let phase = phase.max(shared + 1);
                (
                    Step::Row { value: out, op: at },
                    Value {
                        phase,
                        held: Held::Local,
                    },
                    phase,
                )
            }
            (_, Shape::Row) => (
                Step::Row { value: out, op: at },
                Value {
                    phase,
                    held: Held::Local,
                },
                phase,
            ),
            (_, Shape::Head(_)) => (
                Step::Head { value: out, op: at },
                Value {
                    phase,
                    held: Held::Heads,
                },
                phase,
            ),
            (_, Shape::Scalar | Shape::Vector(_)) => (
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

    let reads_of = |step: &Step| -> Vec<ValueId> {
        match *step {
            Step::Row { op, .. }
            | Step::Uniform { op, .. }
            | Step::Head { op, .. }
            | Step::Reduce { op, .. } => region.ops[op].args(),
            Step::Share { src } => vec![src],
        }
    };

    // What a column alone computes dearly, tabled.
    let column = column_values(region, params, &op_of);
    let dear = dear_values(region, params, &op_of);
    let mut tables: Vec<(ValueId, u32, u32)> = Vec::new();
    let mut table = 0u32;
    for (_, step) in &steps {
        let Step::Row { value, .. } = *step else {
            continue;
        };
        let Some(Column::Every(period)) = column[value as usize] else {
            continue;
        };
        let frontier = steps.iter().any(|(_, reader)| match reader {
            Step::Row { value: r, .. } => {
                column[*r as usize].is_none() && reads_of(reader).contains(&value)
            }
            Step::Reduce { .. } | Step::Share { .. } => reads_of(reader).contains(&value),
            _ => false,
        });
        if period > MAX_TABLE
            || region.width % period != 0
            || types[value as usize].1 != Elem::F32
            || results.contains(&value)
            || region.rounding(value).is_some()
            || !dear[value as usize]
            || !frontier
        {
            continue;
        }
        tables.push((value, table, period));
        values[value as usize] = Some(Value {
            phase: 0,
            held: Held::Table { at: table, period },
        });
        table += period;
    }
    // A tabled value is no row's step, nor is what only tables read.
    if !tables.is_empty() {
        let mut dropped: Vec<ValueId> = tables.iter().map(|(v, ..)| *v).collect();
        for at in (0..steps.len()).rev() {
            let Step::Row { value, .. } = steps[at].1 else {
                continue;
            };
            let read = steps.iter().any(|(_, step)| {
                !matches!(step, Step::Row { value: v, .. } if dropped.contains(v))
                    && reads_of(step).contains(&value)
            });
            if dropped.contains(&value)
                || (column[value as usize].is_some() && !read && !results.contains(&value))
            {
                if !dropped.contains(&value) {
                    dropped.push(value);
                }
                steps.remove(at);
            }
        }
    }

    // A row value computed where its first reader runs.
    let sources: Vec<ValueId> = steps
        .iter()
        .filter_map(|(_, step)| match step {
            Step::Share { src } => Some(*src),
            _ => None,
        })
        .collect();
    for at in (0..steps.len()).rev() {
        let (phase, Step::Row { value, .. }) = steps[at] else {
            continue;
        };
        if results.contains(&value) || sources.contains(&value) {
            continue;
        }
        let need = steps[at + 1..]
            .iter()
            .filter(|(_, step)| reads_of(step).contains(&value))
            .map(|(p, _)| *p)
            .min();
        if let Some(need) = need
            && need > phase
        {
            steps[at].0 = need;
            if let Some(v) = values[value as usize].as_mut() {
                v.phase = need;
            }
        }
    }
    // A row value read in a later phase than its own is kept.
    for (phase, step) in &steps {
        for a in reads_of(step) {
            if let Some(v) = values[a as usize].as_mut()
                && v.held == Held::Local
                && v.phase < *phase
            {
                v.held = Held::Kept;
            }
        }
    }
    // The row is shared once a phase, and read back only after: a second
    // share waits on the first one's reads.
    let mut last_peer: Option<u32> = None;
    let mut shared_at: Vec<u32> = Vec::new();
    for (phase, step) in &steps {
        match step {
            Step::Share { .. } => {
                if shared_at.contains(phase) || last_peer.is_some_and(|p| p >= *phase) {
                    return Err(whole(
                        "two gathers whose shares of the row overlap their reads",
                    ));
                }
                shared_at.push(*phase);
            }
            Step::Row { op, .. } if matches!(region.ops[*op], Op::Peer(..)) => {
                last_peer = Some(last_peer.map_or(*phase, |p| p.max(*phase)));
            }
            _ => {}
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
            Some(Held::Vector(k)) => Operand::Vector(k),
            Some(Held::Kept) => Operand::Kept(v),
            Some(Held::Heads) => Operand::Heads(v),
            Some(Held::Table { at, period }) => Operand::Table { at, period },
            _ => Operand::Local(v),
        }
    };
    let expr_with = |op: &Op, out: ValueId, operand: &dyn Fn(ValueId) -> Operand| -> Expr {
        match *op {
            Op::Const(literal) => Expr::Literal(literal),
            Op::Broadcast(a) => Expr::Copy(operand(a)),
            Op::Unary(unary, a) => Expr::Unary(unary, operand(a)),
            Op::Binary(binary, a, b) => Expr::Binary(binary, operand(a), operand(b)),
            Op::Select(c, a, b) => Expr::Select(operand(c), operand(a), operand(b)),
            Op::Fma(a, b, c) => Expr::Fma(operand(a), operand(b), operand(c)),
            Op::Cast(a, _) => Expr::Cast {
                value: operand(a),
                from: types[a as usize].1,
                to: types[out as usize].1,
            },
            Op::Column => Expr::Column,
            Op::Peer(_, idx) => Expr::Peer(operand(idx)),
            Op::Pick(v, i) => Expr::Pick(operand(v), i),
            Op::Insert(v, i, x) => Expr::Insert(operand(v), i, operand(x)),
            Op::Splat(x, _) => Expr::Splat(operand(x)),
            Op::Carry(init) => Expr::Carry(operand(init)),
            Op::Repeat(count, carry, next) => Expr::Repeat {
                count,
                carry,
                next: operand(next),
            },
            Op::Reduce(..) | Op::ReduceHeads(..) => unreachable!("a reduction is no expression"),
        }
    };
    let expr = |op: &Op, out: ValueId| expr_with(op, out, &operand);
    // A result rounds by its store; any other value where it is defined.
    let round = |v: ValueId| -> Option<Dtype> {
        if results.contains(&v) {
            None
        } else {
            region.rounding(v)
        }
    };

    // Each table's entries: what its value is computed from, in order.
    let tables: Vec<Table> = tables
        .iter()
        .map(|&(value, at, period)| {
            let cone = cone(region, params, &op_of, value);
            let entries = (params..)
                .zip(&region.ops)
                .filter(|(v, op)| cone[*v as usize] && !matches!(op, Op::Const(_)))
                .filter(|(v, _)| {
                    values[*v as usize].is_none_or(|h| h.held != Held::Uniform) || *v == value
                })
                .map(|(v, op)| {
                    let entry = |a: ValueId| match values[a as usize].map(|h| h.held) {
                        Some(Held::Uniform) if !cone_row(&types, a) => Operand::Local(a),
                        _ => Operand::Entry(a),
                    };
                    (v, types[v as usize].1, expr_with(op, v, &entry))
                })
                .collect();
            Table {
                value,
                at,
                period,
                entries,
            }
        })
        .collect();

    let mut phases = Vec::new();
    let mut shared_before = false;
    for phase in 0..=last {
        let mut uniforms = Vec::new();
        let mut heads = Vec::new();
        let mut partials = Vec::new();
        let mut totals = Vec::new();
        let mut rows = Vec::new();
        let mut shares = Vec::new();
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
                    vector: match types[value as usize].0 {
                        Shape::Vector(n) => Some(n),
                        _ => None,
                    },
                    stores: results
                        .iter()
                        .enumerate()
                        .filter(|(_, r)| **r == value)
                        .map(|(k, _)| u32::try_from(k).unwrap_or(u32::MAX))
                        .collect(),
                }),
                Step::Head { value, op } => heads.push(HeadAssign {
                    value,
                    elem: types[value as usize].1,
                    expr: expr(&region.ops[op], value),
                    round: round(value),
                }),
                Step::Reduce { value, op } => {
                    let (reduce, input, seg) = match region.ops[op] {
                        Op::Reduce(reduce, input) => (reduce, input, None),
                        Op::ReduceHeads(reduce, input, size) => (reduce, input, Some(size / 8)),
                        _ => unreachable!("a reduction's step is a reduction"),
                    };
                    partials.push(Partial {
                        value,
                        op: reduce,
                        input: operand(input),
                        seg,
                    });
                    totals.push(Total {
                        value,
                        op: reduce,
                        round: round(value),
                        seg,
                    });
                }
                Step::Share { src } => shares.push(operand(src)),
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
        let pass =
            if rows.is_empty() && partials.is_empty() && stored.is_empty() && shares.is_empty() {
                None
            } else {
                let sync_first = !shares.is_empty() && shared_before;
                if !shares.is_empty() {
                    shared_before = true;
                }
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
                    shares,
                    sync_first,
                })
            };
        phases.push(Phase {
            uniforms,
            heads,
            pass,
        });
    }
    let held_as = |held: Held| -> Vec<(ValueId, Elem)> {
        values
            .iter()
            .enumerate()
            .filter(|(_, v)| v.is_some_and(|v| v.held == held))
            .map(|(at, _)| {
                let at = u32::try_from(at).unwrap_or(u32::MAX);
                (at, types[at as usize].1)
            })
            .collect()
    };
    Ok(Kernel {
        params,
        results: u32::try_from(results.len()).unwrap_or(u32::MAX),
        kept: held_as(Held::Kept),
        heads: held_as(Held::Heads),
        shares: !shared_at.is_empty(),
        tables,
        table,
        phases,
    })
}

/// Whether a value is a row (a table's entry), not a scalar it reads.
fn cone_row(types: &[(Shape, Elem)], v: ValueId) -> bool {
    types[v as usize].0 == Shape::Row
}

/// How each value depends on the column alone, if it does.
fn column_values<'r>(
    region: &'r Region,
    params: u32,
    op_of: &dyn Fn(ValueId) -> Option<&'r Op>,
) -> Vec<Option<Column>> {
    let count = region.params.len() + region.ops.len();
    let mut column: Vec<Option<Column>> = vec![None; count];
    for (v, op) in (params..).zip(&region.ops) {
        let of = |a: ValueId| column[a as usize];
        column[v as usize] = match *op {
            Op::Const(_) => Some(Column::Alike),
            Op::Column => Some(Column::Each),
            Op::Binary(Binary::Rem, a, b) => match (of(a), op_of(b)) {
                (Some(Column::Alike), _) => Some(Column::Alike),
                (Some(Column::Every(q)), Some(Op::Const(Literal::U32(p))))
                    if *p > 0 && q % p != 0 =>
                {
                    Some(Column::Every(q))
                }
                (Some(_), Some(Op::Const(Literal::U32(p)))) if *p > 0 => Some(Column::Every(*p)),
                _ => None,
            },
            Op::Unary(..) | Op::Binary(..) | Op::Select(..) | Op::Cast(..) | Op::Fma(..) => {
                op.args().iter().try_fold(Column::Alike, |held, &a| {
                    Some(match (held, of(a)?) {
                        (Column::Alike, c) | (c, Column::Alike) => c,
                        (Column::Every(p), Column::Every(q)) if p == q => Column::Every(p),
                        _ => Column::Each,
                    })
                })
            }
            _ => None,
        };
    }
    column
}

/// Whether each value is computed dearly: a power or a transcendental in
/// what it is computed from.
fn dear_values<'r>(
    region: &'r Region,
    params: u32,
    _op_of: &dyn Fn(ValueId) -> Option<&'r Op>,
) -> Vec<bool> {
    let count = region.params.len() + region.ops.len();
    let mut dear = vec![false; count];
    for (v, op) in (params..).zip(&region.ops) {
        let own = matches!(
            op,
            Op::Binary(Binary::Pow, ..)
                | Op::Unary(
                    Unary::Exp
                        | Unary::Log
                        | Unary::Sin
                        | Unary::Cos
                        | Unary::Sqrt
                        | Unary::Rsqrt
                        | Unary::Tanh,
                    _
                )
        );
        dear[v as usize] = own || op.args().iter().any(|&a| dear[a as usize]);
    }
    dear
}

/// The values `of` is computed from (and `of`), by value.
fn cone<'r>(
    region: &'r Region,
    _params: u32,
    op_of: &dyn Fn(ValueId) -> Option<&'r Op>,
    of: ValueId,
) -> Vec<bool> {
    let mut held = vec![false; region.params.len() + region.ops.len()];
    let mut stack = vec![of];
    while let Some(v) = stack.pop() {
        if held[v as usize] {
            continue;
        }
        held[v as usize] = true;
        if let Some(op) = op_of(v) {
            stack.extend(op.args());
        }
    }
    held
}

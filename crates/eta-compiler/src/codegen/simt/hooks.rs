//! The stream core's body in hook form: one fused region (a function of one
//! row) as `template <class F> __device__ void body(F& f)`, pure computation
//! over the hooks a kernels-* frame provides (`crate::region::Body` states
//! them). The frame owns everything else: which thread holds which element,
//! loads, stores, vector width, the block-wide reductions.
//!
//! The plan is the guest stream core's (`Streams::role` admits each op) with
//! one difference: a pass does not end at the first reader of a reduction.
//! The row stays in registers across it, so a body runs in phases: each phase
//! is one unrolled loop over the thread's elements, and a reduction's partial
//! accumulates in the loop of its input's phase and is combined over the row
//! right after it. A value that only one row's worth of reductions and
//! constants define (a mean, an `rsqrt`) is uniform: computed once, between
//! loops. A row value a later phase reads is kept in a register array; a
//! parameter is the frame's to keep (`f.ld` reads its registers).

use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt::Write as _;

use eta_ir::Dtype;
use eta_ir::op::tags;

use super::stream::{Role, RowClass, Streams};
use crate::plan::Dimension;

/// Why the stream core cannot write a region's body: the op it stopped at
/// (`None` for the region as a whole) and what it lacks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Uncovered {
    pub op: Option<u32>,
    pub why: &'static str,
}

/// How a value reaches a body expression.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Held {
    /// A parameter, read through `f.ld`.
    Param { k: u32 },
    /// A scalar parameter, one value every row reads: `f.one(k)`.
    One { k: u32 },
    /// A uniform: one per row, a local named `v{id}`.
    Uniform,
    /// A row value local to the loop that computes it.
    Local,
    /// A row value a later phase reads: `v{id}[e]`.
    Kept,
}

struct Value {
    phase: u32,
    held: Held,
}

/// One op as the body computes it.
enum Step {
    /// A uniform computed before the loop of its phase.
    Uniform { value: u32, node: usize },
    /// A row value computed in its phase's loop.
    Row { value: u32, node: usize },
    /// A reduction's partial, accumulated in its input's phase.
    Reduce { value: u32, tag: u8, input: u32 },
}

/// The body of the region `streams` views: `params` parameters (values
/// `0..params`) and `results` the values that leave, `results[k]` written
/// through `f.st(k, ..)`, and `rounds` the values rounded where they are
/// defined. Returned with the passes it makes over the row.
pub(crate) fn emit_body(
    streams: &Streams<'_>,
    params: u32,
    results: &[u32],
    rounds: &[(u32, Dtype)],
) -> Result<(String, u32), Uncovered> {
    // `expr`, the value `value`, as the region holds it: rounded where it
    // rounds.
    let held = |value: u32, expr: String| -> Result<String, Uncovered> {
        match rounds.iter().find(|(v, _)| *v == value) {
            None => Ok(expr),
            Some((_, Dtype::Bf16)) => Ok(format!("f.round_bf16({expr})")),
            Some(_) => Err(Uncovered {
                op: None,
                why: "a value rounded to a dtype no hook rounds to",
            }),
        }
    };
    let count = streams.view.value_types.len();
    let mut values: Vec<Option<Value>> = (0..count).map(|_| None).collect();
    for k in 0..params {
        let held = match streams.class(k) {
            RowClass::Full(_) => Held::Param { k },
            RowClass::Scalar => Held::One { k },
            _ => {
                return Err(whole(
                    "a parameter that is neither a row nor a scalar, and a frame's planes are \
                     one of those",
                ));
            }
        };
        if streams.dtype(k) != Dtype::F32 {
            return Err(whole("a parameter that is not f32, and `f.ld` reads f32"));
        }
        values[k as usize] = Some(Value { phase: 0, held });
    }
    for &r in results {
        if r as usize >= count
            || streams.dtype(r) != Dtype::F32
            || !matches!(streams.class(r), RowClass::Full(_))
        {
            return Err(whole(
                "a result that is not an f32 row, and `f.st` writes one",
            ));
        }
    }

    // Each op's step and phase, in program order (a region's ops are).
    let mut width: Option<Dimension> = None;
    let mut steps: Vec<(u32, Step)> = Vec::new();
    for &node in &streams.view.nodes {
        let op = &streams.ops[node];
        let at = Some(node as u32);
        if op.results != 1 {
            return Err(Uncovered {
                op: at,
                why: "an op of two results",
            });
        }
        let out = streams.bases[node];
        if !matches!(streams.dtype(out), Dtype::F32 | Dtype::Bool) {
            return Err(Uncovered {
                op: at,
                why: "a value that is neither f32 nor bool",
            });
        }
        let phase_of = |values: &[Option<Value>], v: u32| -> Result<u32, Uncovered> {
            values
                .get(v as usize)
                .and_then(Option::as_ref)
                .map(|held| held.phase)
                .ok_or(Uncovered {
                    op: at,
                    why: "an operand no earlier op defines",
                })
        };
        let mut phase = 0;
        for &a in &op.args {
            phase = phase.max(phase_of(&values, a)?);
        }
        let (step, value) = match streams.role(node, &mut width) {
            Some(Role::Map | Role::Broadcast) => {
                map_expr(streams, node, &values, at)?;
                (
                    Step::Row { value: out, node },
                    Value {
                        phase,
                        held: Held::Local,
                    },
                )
            }
            Some(Role::Reduce) => (
                Step::Reduce {
                    value: out,
                    tag: op.tag,
                    input: op.args[0],
                },
                Value {
                    phase: phase + 1,
                    held: Held::Uniform,
                },
            ),
            Some(Role::Argmax | Role::Rng | Role::Intrinsic) => {
                return Err(Uncovered {
                    op: at,
                    why: "an argmax, a draw or an intrinsic, which no hook serves",
                });
            }
            None if streams.class(out) == RowClass::Scalar => {
                if op.tag == tags::CONST {
                    literal(op, at)?;
                } else if pointwise(op.tag)
                    && op
                        .args
                        .iter()
                        .all(|&a| streams.class(a) == RowClass::Scalar)
                {
                    map_expr(streams, node, &values, at)?;
                } else {
                    return Err(Uncovered {
                        op: at,
                        why: "a scalar the stream core does not compute",
                    });
                };
                (
                    Step::Uniform { value: out, node },
                    Value {
                        phase,
                        held: Held::Uniform,
                    },
                )
            }
            None => {
                return Err(Uncovered {
                    op: at,
                    why: "an op the stream core does not cover",
                });
            }
        };
        let step_phase = match step {
            Step::Reduce { .. } => phase,
            _ => value.phase,
        };
        values[out as usize] = Some(value);
        steps.push((step_phase, step));
    }

    // A row value read in a later phase than its own is kept.
    for (phase, step) in &steps {
        let reads: Vec<u32> = match step {
            Step::Row { node, .. } | Step::Uniform { node, .. } => streams.ops[*node].args.clone(),
            Step::Reduce { input, .. } => vec![*input],
        };
        for a in reads {
            if let Some(v) = values[a as usize].as_mut()
                && v.held == Held::Local
                && v.phase < *phase
            {
                v.held = Held::Kept;
            }
        }
    }
    // The last phase any value lands in: a reduction's own result lands one
    // past its loop, and may be all that phase holds.
    let last = values.iter().flatten().map(|v| v.phase).max().unwrap_or(0);

    let mut s = String::new();
    for (at, v) in values.iter().enumerate() {
        if let Some(v) = v
            && v.held == Held::Kept
        {
            let _ = writeln!(s, "  {} v{at}[F::N];", c_type(streams.dtype(at as u32)));
        }
    }
    let mut loops = 0u32;
    for phase in 0..=last {
        // The uniforms this phase's loop reads.
        for (p, step) in &steps {
            if *p != phase {
                continue;
            }
            if let Step::Uniform { value, node } = step {
                let expr = held(*value, uniform_expr(streams, *node, &values)?)?;
                let _ = writeln!(
                    s,
                    "  const {} v{value} = {expr};",
                    c_type(streams.dtype(*value))
                );
            }
        }
        let reductions: Vec<(u32, u8, u32)> = steps
            .iter()
            .filter(|(p, _)| *p == phase)
            .filter_map(|(_, step)| match step {
                Step::Reduce { value, tag, input } => Some((*value, *tag, *input)),
                _ => None,
            })
            .collect();
        let mut rows: Vec<(u32, String)> = Vec::new();
        for (p, step) in &steps {
            if let Step::Row { value, node } = step
                && *p == phase
            {
                rows.push((
                    *value,
                    held(
                        *value,
                        map_expr(streams, *node, &values, Some(*node as u32))?,
                    )?,
                ));
            }
        }
        let stored: Vec<(usize, u32)> = results
            .iter()
            .enumerate()
            .filter(|(_, r)| {
                values[**r as usize].as_ref().is_some_and(|v| {
                    v.phase == phase
                        && matches!(v.held, Held::Local | Held::Kept | Held::Param { .. })
                })
            })
            .map(|(k, r)| (k, *r))
            .collect();
        if rows.is_empty() && reductions.is_empty() && stored.is_empty() {
            continue;
        }

        loops += 1;
        // A reduction takes its elements two at a time, as the hand row
        // kernels take a bf16 pair (`a += x₀ + x₁`): an even element waits
        // in `p`, and its odd neighbour (in the same vector of 8, so valid
        // with it) joins it into the partial. `e` is constant once the loop
        // unrolls, so the choice folds away.
        for (value, tag, _) in &reductions {
            let _ = writeln!(s, "  float a{value} = {};", identity(*tag));
            let _ = writeln!(s, "  float p{value} = {};", identity(*tag));
        }
        s.push_str("#pragma unroll\n");
        s.push_str("  for (int e = 0; e < F::N; ++e) {\n");
        for (value, expr) in &rows {
            let ty = c_type(streams.dtype(*value));
            let first = results.iter().position(|r| r == value);
            let expr = match first {
                Some(k) => format!("f.st({k}, e, {expr})"),
                None => expr.clone(),
            };
            let unread = streams.readers[*value as usize].is_empty();
            match values[*value as usize].as_ref().map(|v| v.held) {
                Some(Held::Kept) => {
                    let _ = writeln!(s, "    v{value}[e] = {expr};");
                }
                _ if unread => {
                    let _ = writeln!(s, "    {expr};");
                }
                _ => {
                    let _ = writeln!(s, "    const {ty} v{value} = {expr};");
                }
            }
            for (k, r) in results.iter().enumerate() {
                if r == value && Some(k) != first {
                    let _ = writeln!(s, "    f.st({k}, e, {});", name(&values, *value));
                }
            }
        }
        for (k, r) in &stored {
            if let Some(Held::Param { k: from, .. }) = values[*r as usize].as_ref().map(|v| v.held)
            {
                let _ = writeln!(s, "    f.st({k}, e, f.ld({from}, e));");
            }
        }
        for (value, tag, input) in &reductions {
            let x = name(&values, *input);
            let pair = combine(*tag, &format!("p{value}"), &x);
            let _ = writeln!(
                s,
                "    if (e % 2 == 0) p{value} = {x}; else if (f.valid(e)) a{value} = {};",
                combine(*tag, &format!("a{value}"), &format!("({pair})"))
            );
        }
        s.push_str("  }\n");
        for (k, _) in &stored {
            let _ = writeln!(s, "  f.template flush<{k}>();");
        }
        for (value, tag, _) in &reductions {
            let total = held(*value, finish(*tag, *value))?;
            let _ = writeln!(s, "  const float v{value} = {total};");
        }
    }
    let mut source = format!(
        "// eta region body: {params} parameter(s), {} result(s), {loops} pass(es) over the row\n",
        results.len()
    );
    source.push_str("struct Body {\n");
    source.push_str("  template <class F>\n");
    source.push_str("  __device__ __forceinline__ void operator()(F& f) const {\n");
    // Nested in the struct and its operator: two levels in, pragmas aside.
    for line in s.lines() {
        if !line.starts_with('#') {
            source.push_str("  ");
        }
        source.push_str(line);
        source.push('\n');
    }
    source.push_str("  }\n};\n");
    Ok((source, loops))
}

fn whole(why: &'static str) -> Uncovered {
    Uncovered { op: None, why }
}

/// A uniform's expression: a constant's literal, or a map over uniforms.
fn uniform_expr(
    streams: &Streams<'_>,
    node: usize,
    values: &[Option<Value>],
) -> Result<String, Uncovered> {
    let op = &streams.ops[node];
    let at = Some(node as u32);
    if op.tag == tags::CONST {
        literal(op, at)
    } else {
        map_expr(streams, node, values, at)
    }
}

fn pointwise(tag: u8) -> bool {
    matches!(
        tag,
        tags::EXP
            | tags::LOG
            | tags::RECIP
            | tags::SIN
            | tags::COS
            | tags::SQRT
            | tags::RSQRT
            | tags::CAST
            | tags::NOT
            | tags::NEG
            | tags::ABS
            | tags::SIGN
            | tags::ADD
            | tags::SUB
            | tags::MUL
            | tags::DIV
            | tags::MAX_ELEM
            | tags::MIN_ELEM
            | tags::REM
            | tags::GT
            | tags::GE
            | tags::EQ
            | tags::NE
            | tags::LT
            | tags::LE
            | tags::AND
            | tags::OR
            | tags::SELECT
    )
}

fn c_type(dtype: Dtype) -> &'static str {
    match dtype {
        Dtype::F32 => "float",
        _ => "bool",
    }
}

/// How a body names `value` where it is read.
fn name(values: &[Option<Value>], value: u32) -> String {
    match values[value as usize].as_ref().map(|v| v.held) {
        Some(Held::Param { k }) => format!("f.ld({k}, e)"),
        Some(Held::One { k }) => format!("f.one({k})"),
        Some(Held::Kept) => format!("v{value}[e]"),
        _ => format!("v{value}"),
    }
}

fn literal(op: &eta_ir::wire::OpWire, at: Option<u32>) -> Result<String, Uncovered> {
    match eta_ir::types::from_wire(op.lit_dtype) {
        Some(Dtype::F32) => Ok(float(f32::from_bits(op.lit_bits))),
        Some(Dtype::Bool) => Ok(if op.lit_bits != 0 { "true" } else { "false" }.into()),
        _ => Err(Uncovered {
            op: at,
            why: "an integer constant",
        }),
    }
}

/// An f32 as a C++ literal that reads back to the same bits.
fn float(x: f32) -> String {
    if x.is_nan() {
        "(0.0f / 0.0f)".into()
    } else if x.is_infinite() {
        if x > 0.0 {
            "(1.0f / 0.0f)"
        } else {
            "(-1.0f / 0.0f)"
        }
        .into()
    } else {
        format!("{x:?}f")
    }
}

fn map_expr(
    streams: &Streams<'_>,
    node: usize,
    values: &[Option<Value>],
    at: Option<u32>,
) -> Result<String, Uncovered> {
    let op = &streams.ops[node];
    let arg = |i: usize| name(values, op.args[i]);
    let bools = |v: u32| streams.dtype(v) == Dtype::Bool;
    Ok(match op.tag {
        tags::BROADCAST => arg(0),
        tags::EXP => format!("expf({})", arg(0)),
        tags::LOG => format!("logf({})", arg(0)),
        tags::SIN => format!("sinf({})", arg(0)),
        tags::COS => format!("cosf({})", arg(0)),
        tags::SQRT => format!("sqrtf({})", arg(0)),
        tags::RSQRT => format!("rsqrtf({})", arg(0)),
        tags::RECIP => format!("(1.0f / {})", arg(0)),
        tags::NEG => format!("(-{})", arg(0)),
        tags::ABS => format!("fabsf({})", arg(0)),
        tags::SIGN => {
            let x = arg(0);
            format!("({x} > 0.0f ? 1.0f : ({x} < 0.0f ? -1.0f : 0.0f))")
        }
        tags::NOT => format!("(!{})", arg(0)),
        tags::CAST => {
            let from = streams.dtype(op.args[0]);
            let to = streams.dtype(streams.bases[node]);
            match (from, to) {
                (a, b) if a == b => arg(0),
                (Dtype::Bool, Dtype::F32) => format!("({} ? 1.0f : 0.0f)", arg(0)),
                (Dtype::F32, Dtype::Bool) => format!("({} != 0.0f)", arg(0)),
                _ => {
                    return Err(Uncovered {
                        op: at,
                        why: "a cast through an integer",
                    });
                }
            }
        }
        tags::SELECT => format!("({} ? {} : {})", arg(0), arg(1), arg(2)),
        tags::AND => format!("({} && {})", arg(0), arg(1)),
        tags::OR => format!("({} || {})", arg(0), arg(1)),
        tag => {
            if op.args.iter().any(|&a| bools(a)) {
                return Err(Uncovered {
                    op: at,
                    why: "arithmetic or a comparison over bools",
                });
            }
            let (a, b) = (arg(0), arg(1));
            match tag {
                tags::ADD => format!("({a} + {b})"),
                tags::SUB => format!("({a} - {b})"),
                tags::MUL => format!("({a} * {b})"),
                tags::DIV => format!("({a} / {b})"),
                tags::MAX_ELEM => format!("fmaxf({a}, {b})"),
                tags::MIN_ELEM => format!("fminf({a}, {b})"),
                tags::REM => format!("fmodf({a}, {b})"),
                tags::GT => format!("({a} > {b})"),
                tags::GE => format!("({a} >= {b})"),
                tags::EQ => format!("({a} == {b})"),
                tags::NE => format!("({a} != {b})"),
                tags::LT => format!("({a} < {b})"),
                tags::LE => format!("({a} <= {b})"),
                _ => {
                    return Err(Uncovered {
                        op: at,
                        why: "an op the stream core does not cover",
                    });
                }
            }
        }
    })
}

fn identity(tag: u8) -> &'static str {
    match tag {
        tags::REDUCE_SUM => "0.0f",
        tags::REDUCE_MAX => "(-1.0f / 0.0f)",
        _ => "(1.0f / 0.0f)",
    }
}

fn combine(tag: u8, acc: &str, x: &str) -> String {
    match tag {
        tags::REDUCE_SUM => format!("{acc} + {x}"),
        tags::REDUCE_MAX => format!("fmaxf({acc}, {x})"),
        _ => format!("fminf({acc}, {x})"),
    }
}

/// A reduction combined over the row: `reduce_sum` and `reduce_max` are the
/// frame's; a minimum is the negated maximum of the negated partials.
fn finish(tag: u8, value: u32) -> String {
    match tag {
        tags::REDUCE_SUM => format!("f.reduce_sum(a{value})"),
        tags::REDUCE_MAX => format!("f.reduce_max(a{value})"),
        _ => format!("(-f.reduce_max(-a{value}))"),
    }
}

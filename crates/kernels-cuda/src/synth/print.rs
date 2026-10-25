//! A kernel (kernels-ir's kernel form) printed as CUDA C++ over the row
//! frame's hooks (`elemwise/frame.cuh`): one struct, pure computation.
//!
//! ```cpp
//! struct Body {
//!   template <class F> __device__ __forceinline__ void operator()(F& f) const;
//! };
//! ```
//!
//! fired as `::pie::elemwise::row_frame<Body, BLOCK, PER, IN, OUT, ...>`. The
//! frame's primitives ([`kernels_ir::frame::Frame`]) are its hooks:
//!
//! | primitive | hook |
//! |---|---|
//! | `elements` | `F::N`, the elements a thread holds (`PER * 8`) |
//! | `load(k, e)`, `one(k)` | `f.ld(k, e)`, `f.one(k)` |
//! | `store(k, e, v)` | `f.st(k, e, v)`, `v` as its plane holds it |
//! | `valid(e)` | `f.valid(e)` |
//! | `flush(k)` | `f.template flush<k>()` |
//! | `reduce(Sum or Max, a)` | `f.reduce_sum(a)`, `f.reduce_max(a)`; a minimum `-f.reduce_max(-a)` |
//! | `round(Bf16, v)` | `f.round_bf16(v)` |
//!
//! Every `k` and `e` the body passes is a constant once its loops unroll, so
//! the frame's arrays stay in registers. The math is f32 through `expf`,
//! `logf`, `sinf`, `cosf`, `sqrtf`, `rsqrtf`, `fmaxf`, `fminf` and `fmodf`,
//! each op as the kernel states it, in its order (the JIT compiles with
//! `fmad=false`, so a product and a sum stay two roundings).

use std::fmt::Write as _;

use dtype::Dtype;
use kernels_ir::{Binary, Elem, Expr, Kernel, Literal, Operand, Reduce, Unary};

/// Why a kernel does not print: a rounding to a dtype no hook rounds to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Unprinted(pub Dtype);

/// `kernel`'s body, its source.
pub fn body(kernel: &Kernel) -> Result<String, Unprinted> {
    let mut s = String::new();
    for (v, elem) in &kernel.kept {
        let _ = writeln!(s, "  {} v{v}[F::N];", c_type(*elem));
    }
    for phase in &kernel.phases {
        for u in &phase.uniforms {
            let expr = held(u.round, expr(&u.expr, "e"))?;
            let _ = writeln!(s, "  const {} v{} = {expr};", c_type(u.elem), u.value);
        }
        let Some(pass) = &phase.pass else {
            continue;
        };
        // A reduction takes its elements two at a time, as the hand row
        // kernels take a bf16 pair (`a += x₀ + x₁`): an even element waits
        // in `p`, and its odd neighbour (in the same vector of 8, so valid
        // with it) joins it into the partial. `e` is constant once the loop
        // unrolls, so the choice folds away.
        for p in &pass.partials {
            let _ = writeln!(s, "  float a{} = {};", p.value, identity(p.op));
            let _ = writeln!(s, "  float p{} = {};", p.value, identity(p.op));
        }
        s.push_str("#pragma unroll\n");
        s.push_str("  for (int e = 0; e < F::N; ++e) {\n");
        for row in &pass.rows {
            let value = row.value;
            let mut e = held(row.round, expr(&row.expr, "e"))?;
            if let Some(k) = row.store {
                e = format!("f.st({k}, e, {e})");
            }
            if row.kept {
                let _ = writeln!(s, "    v{value}[e] = {e};");
            } else if row.unread {
                let _ = writeln!(s, "    {e};");
            } else {
                let _ = writeln!(s, "    const {} v{value} = {e};", c_type(row.elem));
            }
            let name = if row.kept {
                format!("v{value}[e]")
            } else {
                format!("v{value}")
            };
            for k in &row.also {
                let _ = writeln!(s, "    f.st({k}, e, {name});");
            }
        }
        for (k, from) in &pass.copies {
            let _ = writeln!(s, "    f.st({k}, e, f.ld({from}, e));");
        }
        for p in &pass.partials {
            let x = operand(p.input, "e");
            let pair = combine(p.op, &format!("p{}", p.value), &x);
            let _ = writeln!(
                s,
                "    if (e % 2 == 0) p{v} = {x}; else if (f.valid(e)) a{v} = {};",
                combine(p.op, &format!("a{}", p.value), &format!("({pair})")),
                v = p.value
            );
        }
        s.push_str("  }\n");
        for k in &pass.flushes {
            let _ = writeln!(s, "  f.template flush<{k}>();");
        }
        for t in &pass.totals {
            let total = held(t.round, finish(t.op, t.value))?;
            let _ = writeln!(s, "  const float v{} = {total};", t.value);
        }
    }
    let mut source = format!(
        "// region body: {} parameter(s), {} result(s), {} pass(es) over the row\n",
        kernel.params,
        kernel.results,
        kernel.passes()
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
    Ok(source)
}

/// `expr` rounded as the kernel says.
fn held(round: Option<Dtype>, expr: String) -> Result<String, Unprinted> {
    match round {
        None => Ok(expr),
        Some(Dtype::Bf16) => Ok(format!("f.round_bf16({expr})")),
        Some(other) => Err(Unprinted(other)),
    }
}

const fn c_type(elem: Elem) -> &'static str {
    match elem {
        Elem::F32 => "float",
        Elem::Bool => "bool",
    }
}

/// How a body names an operand at element `index`.
fn operand(o: Operand, index: &str) -> String {
    match o {
        Operand::Param(k) => format!("f.ld({k}, {index})"),
        Operand::One(k) => format!("f.one({k})"),
        Operand::Kept(v) => format!("v{v}[{index}]"),
        Operand::Local(v) => format!("v{v}"),
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

fn expr(e: &Expr, index: &str) -> String {
    let o = |x: Operand| operand(x, index);
    match *e {
        Expr::Literal(Literal::F32(x)) => float(x),
        Expr::Literal(Literal::Bool(b)) => if b { "true" } else { "false" }.into(),
        Expr::Copy(a) => o(a),
        Expr::Unary(u, a) => {
            let a = o(a);
            match u {
                Unary::Exp => format!("expf({a})"),
                Unary::Log => format!("logf({a})"),
                Unary::Sin => format!("sinf({a})"),
                Unary::Cos => format!("cosf({a})"),
                Unary::Sqrt => format!("sqrtf({a})"),
                Unary::Rsqrt => format!("rsqrtf({a})"),
                Unary::Recip => format!("(1.0f / {a})"),
                Unary::Neg => format!("(-{a})"),
                Unary::Abs => format!("fabsf({a})"),
                Unary::Sign => format!("({a} > 0.0f ? 1.0f : ({a} < 0.0f ? -1.0f : 0.0f))"),
                Unary::Not => format!("(!{a})"),
            }
        }
        Expr::Cast { value, from, to } => match (from, to) {
            (a, b) if a == b => o(value),
            (Elem::Bool, Elem::F32) => format!("({} ? 1.0f : 0.0f)", o(value)),
            (Elem::F32, Elem::Bool) => format!("({} != 0.0f)", o(value)),
            _ => unreachable!("every pair of elements taken above"),
        },
        Expr::Select(c, a, b) => format!("({} ? {} : {})", o(c), o(a), o(b)),
        Expr::Binary(op, a, b) => {
            let (a, b) = (o(a), o(b));
            match op {
                Binary::Add => format!("({a} + {b})"),
                Binary::Sub => format!("({a} - {b})"),
                Binary::Mul => format!("({a} * {b})"),
                Binary::Div => format!("({a} / {b})"),
                Binary::Max => format!("fmaxf({a}, {b})"),
                Binary::Min => format!("fminf({a}, {b})"),
                Binary::Rem => format!("fmodf({a}, {b})"),
                Binary::Gt => format!("({a} > {b})"),
                Binary::Ge => format!("({a} >= {b})"),
                Binary::Eq => format!("({a} == {b})"),
                Binary::Ne => format!("({a} != {b})"),
                Binary::Lt => format!("({a} < {b})"),
                Binary::Le => format!("({a} <= {b})"),
                Binary::And => format!("({a} && {b})"),
                Binary::Or => format!("({a} || {b})"),
            }
        }
    }
}

const fn identity(op: Reduce) -> &'static str {
    match op {
        Reduce::Sum => "0.0f",
        Reduce::Max => "(-1.0f / 0.0f)",
        Reduce::Min => "(1.0f / 0.0f)",
    }
}

fn combine(op: Reduce, acc: &str, x: &str) -> String {
    match op {
        Reduce::Sum => format!("{acc} + {x}"),
        Reduce::Max => format!("fmaxf({acc}, {x})"),
        Reduce::Min => format!("fminf({acc}, {x})"),
    }
}

/// A reduction combined over the row: `reduce_sum` and `reduce_max` are the
/// frame's; a minimum is the negated maximum of the negated partials.
fn finish(op: Reduce, value: u32) -> String {
    match op {
        Reduce::Sum => format!("f.reduce_sum(a{value})"),
        Reduce::Max => format!("f.reduce_max(a{value})"),
        Reduce::Min => format!("(-f.reduce_max(-a{value}))"),
    }
}

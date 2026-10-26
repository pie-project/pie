//! A kernel (kernels-ir's kernel form) printed as HIP C++ over the row
//! skeleton's hooks (`elemwise/skeleton.hip`): one struct, pure computation.
//! kernels-cuda prints the same body for its skeleton; this is its own copy.
//!
//! ```cpp
//! struct Body {
//!   template <class F> __device__ __forceinline__ void operator()(F& f) const;
//! };
//! ```
//!
//! fired as `::pie::elemwise::row_skeleton<Body, BLOCK, PER, IN, OUT, ...>`. The
//! skeleton's primitives ([`kernels_ir::skeleton::Skeleton`]) are its hooks:
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
//! | `column(e)` | `f.col(e)` |
//! | `reduce_heads(Sum or Max, seg, a)` | `f.template reduce_sum_heads<seg>(a)`, `reduce_max_heads` |
//! | `share(e, v)`, `peer(c)` | `f.share(e, v)`, `f.peer(c)`, the row complete between (`f.sync()`) |
//!
//! A table is `f.table()`, its entries written a column a thread
//! (`f.thread()` on by `F::THREADS`) and complete for the block
//! (`f.sync_block()`) before the first pass. A value one a head is an array
//! of `F::PER`, one a vector of 8. Sines and cosines are the skeleton's
//! (`f.sin`, `f.cos`: an angle as its kernels turn one).
//!
//! Every `k` and `e` the body passes is a constant once its loops unroll, so
//! the skeleton's arrays stay in registers. The math is f32 through `expf`,
//! `logf`, `sinf`, `cosf`, `sqrtf`, `rsqrtf`, `fmaxf`, `fminf` and `fmodf`,
//! each op as the kernel states it, in its order (the JIT compiles with
//! `-ffp-contract=off`, so a product and a sum stay two roundings).

use std::fmt::Write as _;

use dtype::Dtype;
use kernels_ir::{Binary, Elem, Expr, Kernel, Literal, Operand, Reduce, Unary};

/// Where an expression is printed: a pass's element `e`, a head loop's
/// vector `p`, or a table's column `c`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum At {
    Elem,
    Vector,
    Column,
}

/// Why a kernel does not print.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unprinted {
    /// A rounding to a dtype no hook rounds to.
    Rounding(Dtype),
    /// A small vector one a row, or a repeat over one: this skeleton holds
    /// neither yet.
    Vectors,
}

/// Whether `kernel` holds a small vector one a row or a repeat, which this
/// printer does not print.
fn vectors(kernel: &Kernel) -> bool {
    let reads = |e: &Expr| {
        matches!(
            e,
            Expr::Pick(..)
                | Expr::Insert(..)
                | Expr::Splat(_)
                | Expr::Carry(_)
                | Expr::Repeat { .. }
        ) || expr_operands(e)
            .iter()
            .any(|o| matches!(o, Operand::Vector(_)))
    };
    kernel.phases.iter().any(|phase| {
        phase
            .uniforms
            .iter()
            .any(|u| u.vector.is_some() || !u.stores.is_empty() || reads(&u.expr))
            || phase.heads.iter().any(|h| reads(&h.expr))
            || phase
                .pass
                .as_ref()
                .is_some_and(|pass| pass.rows.iter().any(|r| reads(&r.expr)))
    })
}

/// The operands `e` reads.
fn expr_operands(e: &Expr) -> Vec<Operand> {
    match *e {
        Expr::Literal(_) | Expr::Column => Vec::new(),
        Expr::Copy(a)
        | Expr::Unary(_, a)
        | Expr::Cast { value: a, .. }
        | Expr::Peer(a)
        | Expr::Pick(a, _)
        | Expr::Splat(a)
        | Expr::Carry(a) => vec![a],
        Expr::Binary(_, a, b) | Expr::Insert(a, _, b) => vec![a, b],
        Expr::Select(a, b, c) | Expr::Fma(a, b, c) => vec![a, b, c],
        Expr::Repeat { next, .. } => vec![next],
    }
}

/// `kernel`'s body, its source; refused where it holds a small vector or a
/// repeat.
pub fn body(kernel: &Kernel) -> Result<String, Unprinted> {
    if vectors(kernel) {
        return Err(Unprinted::Vectors);
    }
    let mut s = String::new();
    for (v, elem) in &kernel.kept {
        let _ = writeln!(s, "  {} v{v}[F::N];", c_type(*elem));
    }
    for (v, elem) in &kernel.heads {
        let _ = writeln!(s, "  {} v{v}[F::PER];", c_type(*elem));
    }
    for (at, phase) in kernel.phases.iter().enumerate() {
        for u in &phase.uniforms {
            let expr = held(u.round, expr(&u.expr, At::Elem, u.elem))?;
            let _ = writeln!(s, "  const {} v{} = {expr};", c_type(u.elem), u.value);
        }
        // The tables, once the first phase's uniforms (their constants) are.
        if at == 0 && !kernel.tables.is_empty() {
            for t in &kernel.tables {
                let _ = writeln!(
                    s,
                    "  for (unsigned c = f.thread(); c < {}u; c += F::THREADS) {{",
                    t.period
                );
                for (value, elem, e) in &t.entries {
                    let _ = writeln!(
                        s,
                        "    const {} t{value} = {};",
                        c_type(*elem),
                        expr(e, At::Column, *elem)
                    );
                }
                let _ = writeln!(s, "    f.table()[{}u + c] = t{};", t.at, t.value);
                s.push_str("  }\n");
            }
            s.push_str("  f.sync_block();\n");
        }
        for h in &phase.heads {
            let e = held(h.round, expr(&h.expr, At::Vector, h.elem))?;
            s.push_str("#pragma unroll\n");
            let _ = writeln!(
                s,
                "  for (int p = 0; p < F::PER; ++p) v{}[p] = {e};",
                h.value
            );
        }
        let Some(pass) = &phase.pass else {
            continue;
        };
        // A second share overwrites what the first one's gathers read, all
        // of them in earlier passes.
        if pass.sync_first {
            s.push_str("  f.sync();\n");
        }
        // A reduction takes its elements two at a time, as the hand row
        // kernels take a bf16 pair (`a += x₀ + x₁`): an even element waits
        // in `p`, and its odd neighbour (in the same vector of 8, so valid
        // with it) joins it into the partial. `e` is constant once the loop
        // unrolls, so the choice folds away.
        // Over heads, a vector's pairs join its own partial, one a vector.
        for p in &pass.partials {
            if p.seg.is_some() {
                let _ = writeln!(s, "  float a{}[F::PER];", p.value);
                s.push_str("#pragma unroll\n");
                let _ = writeln!(
                    s,
                    "  for (int p = 0; p < F::PER; ++p) a{}[p] = {};",
                    p.value,
                    identity(p.op)
                );
            } else {
                let _ = writeln!(s, "  float a{} = {};", p.value, identity(p.op));
            }
            let _ = writeln!(s, "  float p{} = {};", p.value, identity(p.op));
        }
        s.push_str("#pragma unroll\n");
        s.push_str("  for (int e = 0; e < F::N; ++e) {\n");
        for row in &pass.rows {
            let value = row.value;
            let mut e = held(row.round, expr(&row.expr, At::Elem, row.elem))?;
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
            let x = operand(p.input, At::Elem);
            let pair = combine(p.op, &format!("p{}", p.value), &x);
            let acc = if p.seg.is_some() {
                format!("a{}[e / 8]", p.value)
            } else {
                format!("a{}", p.value)
            };
            let _ = writeln!(
                s,
                "    if (e % 2 == 0) p{v} = {x}; else if (f.valid(e)) {acc} = {};",
                combine(p.op, &acc, &format!("({pair})")),
                v = p.value
            );
        }
        for share in &pass.shares {
            let _ = writeln!(s, "    f.share(e, {});", operand(*share, At::Elem));
        }
        s.push_str("  }\n");
        for k in &pass.flushes {
            let _ = writeln!(s, "  f.template flush<{k}>();");
        }
        if !pass.shares.is_empty() {
            s.push_str("  f.sync();\n");
        }
        for t in &pass.totals {
            let Some(seg) = t.seg else {
                let total = held(t.round, finish(t.op, t.value))?;
                let _ = writeln!(s, "  const float v{} = {total};", t.value);
                continue;
            };
            let v = t.value;
            // a minimum the negated maximum of the negated partials
            let total = match t.op {
                Reduce::Sum => {
                    let _ = writeln!(s, "  f.template reduce_sum_heads<{seg}>(a{v});");
                    held(t.round, format!("a{v}[p]"))?
                }
                Reduce::Max => {
                    let _ = writeln!(s, "  f.template reduce_max_heads<{seg}>(a{v});");
                    held(t.round, format!("a{v}[p]"))?
                }
                Reduce::Min => {
                    s.push_str("#pragma unroll\n");
                    let _ = writeln!(s, "  for (int p = 0; p < F::PER; ++p) a{v}[p] = -a{v}[p];");
                    let _ = writeln!(s, "  f.template reduce_max_heads<{seg}>(a{v});");
                    held(t.round, format!("(-a{v}[p])"))?
                }
            };
            s.push_str("#pragma unroll\n");
            let _ = writeln!(s, "  for (int p = 0; p < F::PER; ++p) v{v}[p] = {total};");
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
        Some(other) => Err(Unprinted::Rounding(other)),
    }
}

const fn c_type(elem: Elem) -> &'static str {
    match elem {
        Elem::F32 => "float",
        Elem::Bool => "bool",
        Elem::U32 => "unsigned",
    }
}

/// How a body names an operand where it is read.
fn operand(o: Operand, at: At) -> String {
    match (o, at) {
        (Operand::Param(k), _) => format!("f.ld({k}, e)"),
        (Operand::Vector(_), _) => unreachable!("`body` refuses a small vector"),
        (Operand::One(k), _) => format!("f.one({k})"),
        (Operand::Kept(v), _) => format!("v{v}[e]"),
        (Operand::Heads(v), At::Vector) => format!("v{v}[p]"),
        (Operand::Heads(v), _) => format!("v{v}[e / 8]"),
        (Operand::Table { at, period }, _) => format!("f.table()[{at}u + f.col(e) % {period}u]"),
        (Operand::Entry(v), _) => format!("t{v}"),
        (Operand::Local(v), _) => format!("v{v}"),
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

/// `e` as C++, its value of `elem` (an integer's op is the integer's).
fn expr(e: &Expr, at: At, elem: Elem) -> String {
    let o = |x: Operand| operand(x, at);
    let int = elem == Elem::U32;
    match *e {
        Expr::Literal(Literal::F32(x)) => float(x),
        Expr::Literal(Literal::Bool(b)) => if b { "true" } else { "false" }.into(),
        Expr::Literal(Literal::U32(n)) => format!("{n}u"),
        Expr::Column if at == At::Column => "c".into(),
        Expr::Column => "f.col(e)".into(),
        Expr::Peer(c) => format!("f.peer({})", o(c)),
        Expr::Pick(..)
        | Expr::Insert(..)
        | Expr::Splat(_)
        | Expr::Carry(_)
        | Expr::Repeat { .. } => {
            unreachable!("`body` refuses a small vector or a repeat")
        }
        Expr::Fma(a, b, c) => format!("fmaf({}, {}, {})", o(a), o(b), o(c)),
        Expr::Copy(a) => o(a),
        Expr::Unary(u, a) if int => {
            let a = o(a);
            match u {
                Unary::Neg => format!("(0u - {a})"),
                Unary::Sign => format!("({a} > 0u ? 1u : 0u)"),
                _ => a,
            }
        }
        Expr::Unary(u, a) => {
            let a = o(a);
            match u {
                Unary::Exp => format!("expf({a})"),
                Unary::Log => format!("logf({a})"),
                Unary::Sin => format!("f.sin({a})"),
                Unary::Cos => format!("f.cos({a})"),
                Unary::Sqrt => format!("sqrtf({a})"),
                Unary::Rsqrt => format!("rsqrtf({a})"),
                Unary::Tanh => format!("tanhf({a})"),
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
            (Elem::U32, Elem::Bool) => format!("({} != 0u)", o(value)),
            (Elem::Bool, Elem::U32) => format!("({} ? 1u : 0u)", o(value)),
            (_, to) => format!("static_cast<{}>({})", c_type(to), o(value)),
        },
        Expr::Select(c, a, b) => format!("({} ? {} : {})", o(c), o(a), o(b)),
        Expr::Binary(op, a, b) => {
            let (a, b) = (o(a), o(b));
            match op {
                Binary::Add => format!("({a} + {b})"),
                Binary::Sub => format!("({a} - {b})"),
                Binary::Mul => format!("({a} * {b})"),
                Binary::Div => format!("({a} / {b})"),
                Binary::Max if int => format!("max({a}, {b})"),
                Binary::Min if int => format!("min({a}, {b})"),
                Binary::Rem if int => format!("({a} % {b})"),
                Binary::Max => format!("fmaxf({a}, {b})"),
                Binary::Min => format!("fminf({a}, {b})"),
                Binary::Rem => format!("fmodf({a}, {b})"),
                Binary::Pow => format!("powf({a}, {b})"),
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
/// skeleton's; a minimum is the negated maximum of the negated partials.
fn finish(op: Reduce, value: u32) -> String {
    match op {
        Reduce::Sum => format!("f.reduce_sum(a{value})"),
        Reduce::Max => format!("f.reduce_max(a{value})"),
        Reduce::Min => format!("(-f.reduce_max(-a{value}))"),
    }
}

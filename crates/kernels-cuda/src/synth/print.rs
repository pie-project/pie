//! A kernel (kernels-ir's kernel form) printed as CUDA C++ over the row
//! skeleton's hooks (`elemwise/skeleton.cuh`): one struct, pure computation.
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
//! `fmad=false`, so a product and a sum stay two roundings).

use std::fmt::Write as _;

use dtype::Dtype;
use kernels_ir::{Binary, Elem, Expr, Kernel, Literal, Loop, Operand, Reduce, Unary};

/// Where an expression is printed: a pass's element `e`, a head loop's
/// vector `p`, or a table's column `c`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum At {
    Elem,
    Vector,
    Column,
}

/// Why a kernel does not print: a rounding to a dtype no hook rounds to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Unprinted(pub Dtype);

/// The row parameters `kernel` first reads in a pass past its first, by
/// bit: each one's vectors are issued (`f.template load<k>()`) once the
/// pass before is done with its own, ahead of that pass's reductions, so
/// they hold no registers through the passes that never read them (a
/// modulation's scale and shift through its norm's sums).
#[must_use]
pub fn late(kernel: &Kernel) -> u32 {
    first_reads(kernel)
        .iter()
        .enumerate()
        .filter(|(_, first)| matches!(first, Some(pass) if *pass > 0))
        .fold(0, |mask, (k, _)| mask | 1 << k)
}

/// The pass (counted among the phases that make one) each parameter is
/// first read in as a row; a parameter read outside a pass's elements (a
/// value one a head, a table) counts as read in the first.
fn first_reads(kernel: &Kernel) -> Vec<Option<usize>> {
    let mut first: Vec<Option<usize>> = vec![None; kernel.params as usize];
    let mut mark = |o: Operand, pass: usize| {
        if let Operand::Param(k) = o
            && let Some(at) = first.get_mut(k as usize)
        {
            *at = Some(at.map_or(pass, |was| was.min(pass)));
        }
    };
    let mut pass = 0usize;
    for phase in &kernel.phases {
        for o in phase
            .uniforms
            .iter()
            .flat_map(|u| operands(&u.expr))
            .chain(phase.heads.iter().flat_map(|h| operands(&h.expr)))
        {
            mark(o, 0);
        }
        let Some(lp) = &phase.pass else {
            continue;
        };
        for o in lp
            .rows
            .iter()
            .flat_map(|r| operands(&r.expr))
            .chain(lp.partials.iter().map(|p| p.input))
            .chain(lp.shares.iter().copied())
            .chain(lp.copies.iter().map(|(_, from)| Operand::Param(*from)))
        {
            mark(o, pass);
        }
        pass += 1;
    }
    for t in &kernel.tables {
        for o in t.entries.iter().flat_map(|(_, _, e)| operands(e)) {
            mark(o, 0);
        }
    }
    first
}

/// The operands `e` reads.
fn operands(e: &Expr) -> Vec<Operand> {
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

/// A value one a row at element `j` (a C++ expression): a small vector's
/// `j`th, a scalar's one.
fn lane(o: Operand, j: &str, vectors: &std::collections::HashMap<u32, u32>) -> String {
    match o {
        Operand::Vector(k) => format!("f.vec({k}, {j})"),
        Operand::Local(v) if vectors.contains_key(&v) => format!("v{v}[{j}]"),
        other => operand(other, At::Elem),
    }
}

/// A phase's uniforms printed: scalars one a row, small vectors as arrays a
/// thread holds (every thread computing its row's alike), a repeat as a
/// loop around its body, a small-vector result stored by the row's first
/// thread (`f.st_vec`).
fn uniforms(
    s: &mut String,
    us: &[kernels_ir::Uniform],
    vectors: &std::collections::HashMap<u32, u32>,
    indent: &str,
) -> Result<(), Unprinted> {
    let mut i = 0;
    while i < us.len() {
        let u = &us[i];
        let ty = c_type(u.elem);
        let v = u.value;
        if let Expr::Carry(init) = u.expr {
            let close = (i + 1..us.len())
                .find(|&k| matches!(us[k].expr, Expr::Repeat { carry, .. } if carry == v))
                .expect("a carry's repeat closes it in its phase");
            let Expr::Repeat { count, next, .. } = us[close].expr else {
                unreachable!("found above");
            };
            let r = us[close].value;
            match u.vector {
                Some(n) => {
                    let _ = writeln!(s, "{indent}{ty} v{v}[{n}];");
                    let _ = writeln!(s, "{indent}#pragma unroll");
                    let _ = writeln!(
                        s,
                        "{indent}for (int j = 0; j < {n}; ++j) v{v}[j] = {};",
                        lane(init, "j", vectors)
                    );
                }
                None => {
                    let _ = writeln!(s, "{indent}{ty} v{v} = {};", lane(init, "0", vectors));
                }
            }
            let _ = writeln!(
                s,
                "{indent}for (int it{v} = 0; it{v} < {count}; ++it{v}) {{"
            );
            let inner = format!("{indent}  ");
            uniforms(s, &us[i + 1..close], vectors, &inner)?;
            match u.vector {
                Some(n) => {
                    let _ = writeln!(s, "{inner}#pragma unroll");
                    let _ = writeln!(
                        s,
                        "{inner}for (int j = 0; j < {n}; ++j) v{v}[j] = {};",
                        lane(next, "j", vectors)
                    );
                }
                None => {
                    let _ = writeln!(s, "{inner}v{v} = {};", lane(next, "0", vectors));
                }
            }
            let _ = writeln!(s, "{indent}}}");
            match u.vector {
                Some(n) => {
                    let _ = writeln!(s, "{indent}{ty} v{r}[{n}];");
                    let _ = writeln!(s, "{indent}#pragma unroll");
                    let _ = writeln!(
                        s,
                        "{indent}for (int j = 0; j < {n}; ++j) v{r}[j] = v{v}[j];"
                    );
                }
                None => {
                    let _ = writeln!(s, "{indent}const {ty} v{r} = v{v};");
                }
            }
            stores(s, &us[close], vectors, indent);
            i = close + 1;
            continue;
        }
        match u.vector {
            Some(n) => {
                let _ = writeln!(s, "{indent}{ty} v{v}[{n}];");
                let _ = writeln!(s, "{indent}#pragma unroll");
                let element = match u.expr {
                    Expr::Insert(src, _, _) => lane(src, "j", vectors),
                    Expr::Splat(x) => lane(x, "0", vectors),
                    ref e => expr_of(e, At::Elem, u.elem, &|x| lane(x, "j", vectors)),
                };
                let _ = writeln!(
                    s,
                    "{indent}for (int j = 0; j < {n}; ++j) v{v}[j] = {};",
                    held(u.round, element)?
                );
                if let Expr::Insert(_, at, x) = u.expr {
                    let _ = writeln!(
                        s,
                        "{indent}v{v}[{at}] = {};",
                        held(u.round, lane(x, "0", vectors))?
                    );
                }
            }
            None => {
                let e = match u.expr {
                    Expr::Pick(src, at) => lane(src, &at.to_string(), vectors),
                    ref e => expr_of(e, At::Elem, u.elem, &|x| lane(x, "0", vectors)),
                };
                let _ = writeln!(s, "{indent}const {ty} v{v} = {};", held(u.round, e)?);
            }
        }
        stores(s, u, vectors, indent);
        i += 1;
    }
    Ok(())
}

/// A small vector's stores as the results it is.
fn stores(
    s: &mut String,
    u: &kernels_ir::Uniform,
    _vectors: &std::collections::HashMap<u32, u32>,
    indent: &str,
) {
    let Some(n) = u.vector else {
        return;
    };
    for k in &u.stores {
        let _ = writeln!(s, "{indent}#pragma unroll");
        let _ = writeln!(
            s,
            "{indent}for (int j = 0; j < {n}; ++j) f.template st_vec<{k}>(j, v{}[j]);",
            u.value
        );
    }
}

/// `kernel`'s body, its source.
pub fn body(kernel: &Kernel) -> Result<String, Unprinted> {
    body_with(kernel, None)
}

/// A pair-packed turn: the row's vectors laid out so the few a select turns
/// (a wide head's rotated pairs, where the rotary dims are a part of it) sit
/// on the first threads, whole warps of them, and the select's turned arm
/// computed only where it is taken. A warp holding none of them skips the
/// turn; SIMT otherwise runs both arms on every lane.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Packing {
    /// The vectors of 8 the select takes its arm at, in order: the first
    /// threads' (slot `s < turned.len()` holds `turned[s]`).
    pub turned: Vec<u32>,
    /// The row's vectors of 8.
    pub vectors: u32,
    /// The phase whose pass makes the select, and its row there.
    pub phase: usize,
    pub select: usize,
}

impl Packing {
    /// The vector slot `s` holds: the turned ones first, then the rest in
    /// order.
    #[must_use]
    pub fn vector(&self, s: u32) -> u32 {
        let k = self.turned.len() as u32;
        if s < k {
            return self.turned[s as usize];
        }
        let mut rest = s - k;
        for v in 0..self.vectors {
            if !self.turned.contains(&v) {
                if rest == 0 {
                    return v;
                }
                rest -= 1;
            }
        }
        s
    }

    /// `vector` as a C++ function of the slot: piecewise over the runs of
    /// consecutive slots that hold consecutive vectors.
    fn source(&self) -> String {
        let mut runs: Vec<(u32, u32, u32)> = Vec::new(); // (first slot, slots, first vector)
        for slot in 0..self.vectors {
            let v = self.vector(slot);
            match runs.last_mut() {
                Some((first, n, at)) if *first + *n == slot && *at + *n == v => *n += 1,
                _ => runs.push((slot, 1, v)),
            }
        }
        let mut s = String::from(
            "  // The vector of 8 slot `s` holds: the turned vectors on the first threads.
               __device__ __forceinline__ static unsigned vector(unsigned s) {
",
        );
        for (first, n, at) in &runs {
            let _ = writeln!(
                s,
                "    if (s < {}u) return s + {}u;",
                first + n,
                at.wrapping_sub(*first)
            );
        }
        s.push_str(
            "    return s;
  }
",
        );
        s
    }
}

/// The pair-packed turn of `kernel` over rows `width` wide, where one fits:
/// a pass's select whose condition is a function of the column alone, the
/// same over each vector of 8, holding at under half of the row's vectors
/// and those more than a warp's worth from filling it (a row past a warp's
/// vectors, so whole warps hold none), and whose taken arm computes a turn
/// (a sine, a cosine or a gather within the row) no other value reads. A
/// kernel that reduces over heads keeps its order (a head's vectors sit on
/// adjacent threads).
#[must_use]
pub fn packing(kernel: &Kernel, width: u32) -> Option<Packing> {
    let vectors = width / 8;
    if vectors <= 32 || !width.is_multiple_of(8) {
        return None;
    }
    let by_heads = kernel.phases.iter().any(|phase| {
        phase
            .pass
            .as_ref()
            .is_some_and(|pass| pass.partials.iter().any(|p| p.seg.is_some()))
    });
    if by_heads || !kernel.heads.is_empty() {
        return None;
    }
    let mut uniforms: Vec<(u32, Expr)> = Vec::new();
    for (at_phase, phase) in kernel.phases.iter().enumerate() {
        uniforms.extend(phase.uniforms.iter().map(|u| (u.value, u.expr)));
        let Some(pass) = &phase.pass else {
            continue;
        };
        let lazy = Lazy::of(pass);
        for (j, row) in pass.rows.iter().enumerate() {
            let Expr::Select(Operand::Local(c), _, _) = row.expr else {
                continue;
            };
            let turned_arm = lazy.members(j, true);
            if !turned_arm.iter().any(|&m| turns(&pass.rows[m].expr)) {
                continue;
            }
            let at = |col: u32| column_only(c, col, pass, &uniforms);
            let mut turned = Vec::new();
            let mut uniform = true;
            for v in 0..vectors {
                let first = at(v * 8);
                if (1..8).any(|e| at(v * 8 + e) != first) {
                    uniform = false;
                    break;
                }
                match first {
                    Some(true) => turned.push(v),
                    Some(false) => {}
                    None => {
                        uniform = false;
                        break;
                    }
                }
            }
            if uniform && !turned.is_empty() && 2 * turned.len() as u32 <= vectors {
                return Some(Packing {
                    turned,
                    vectors,
                    phase: at_phase,
                    select: j,
                });
            }
        }
    }
    None
}

/// Whether `e` turns: a sine or cosine, or a gather within the row.
fn turns(e: &Expr) -> bool {
    matches!(e, Expr::Unary(Unary::Sin | Unary::Cos, _) | Expr::Peer(_))
}

/// Value `v` of a pass at column `col`, where it is a function of the
/// column (and the constants) alone; `None` where it reads anything else.
fn column_only(v: u32, col: u32, pass: &Loop, uniforms: &[(u32, Expr)]) -> Option<bool> {
    #[derive(Clone, Copy, PartialEq)]
    enum X {
        U(u32),
        B(bool),
        F(f32),
    }
    fn eval(v: u32, col: u32, pass: &Loop, uniforms: &[(u32, Expr)], depth: u32) -> Option<X> {
        if depth > 64 {
            return None;
        }
        let e = pass
            .rows
            .iter()
            .find(|r| r.value == v)
            .map(|r| r.expr)
            .or_else(|| uniforms.iter().find(|(u, _)| *u == v).map(|(_, e)| *e))?;
        let o = |o: Operand| match o {
            Operand::Local(w) => eval(w, col, pass, uniforms, depth + 1),
            _ => None,
        };
        Some(match e {
            Expr::Literal(Literal::U32(n)) => X::U(n),
            Expr::Literal(Literal::Bool(b)) => X::B(b),
            Expr::Literal(Literal::F32(x)) => X::F(x),
            Expr::Column => X::U(col),
            Expr::Copy(a) => o(a)?,
            Expr::Unary(Unary::Not, a) => match o(a)? {
                X::B(b) => X::B(!b),
                _ => return None,
            },
            Expr::Select(c, a, b) => match o(c)? {
                X::B(true) => o(a)?,
                X::B(false) => o(b)?,
                _ => return None,
            },
            Expr::Binary(op, a, b) => match (o(a)?, o(b)?) {
                (X::U(a), X::U(b)) => match op {
                    Binary::Add => X::U(a.wrapping_add(b)),
                    Binary::Sub => X::U(a.wrapping_sub(b)),
                    Binary::Mul => X::U(a.wrapping_mul(b)),
                    Binary::Div if b != 0 => X::U(a / b),
                    Binary::Rem if b != 0 => X::U(a % b),
                    Binary::Max => X::U(a.max(b)),
                    Binary::Min => X::U(a.min(b)),
                    Binary::Gt => X::B(a > b),
                    Binary::Ge => X::B(a >= b),
                    Binary::Eq => X::B(a == b),
                    Binary::Ne => X::B(a != b),
                    Binary::Lt => X::B(a < b),
                    Binary::Le => X::B(a <= b),
                    _ => return None,
                },
                (X::B(a), X::B(b)) => match op {
                    Binary::And => X::B(a && b),
                    Binary::Or => X::B(a || b),
                    Binary::Eq => X::B(a == b),
                    Binary::Ne => X::B(a != b),
                    _ => return None,
                },
                _ => return None,
            },
            _ => return None,
        })
    }
    match eval(v, col, pass, uniforms, 0)? {
        X::B(b) => Some(b),
        _ => None,
    }
}

/// Which rows of a pass an arm of a select alone reads, so they move into
/// that arm (`if (c) { ... } else { ... }`): a row stored, kept for a later
/// pass, summed or shared stays where it is, as does one either arm, the
/// condition or anything past the select reads.
struct Lazy {
    /// Each row's select and arm (`true` the taken one), where it moved.
    owner: Vec<Option<(usize, bool)>>,
}

impl Lazy {
    fn of(pass: &Loop) -> Lazy {
        let n = pass.rows.len();
        let index = |v: u32| pass.rows.iter().position(|r| r.value == v);
        // Each row's readers among the rows; `pinned` where anything else
        // reads it or it leaves the pass.
        let mut readers: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut pinned = vec![false; n];
        for (j, row) in pass.rows.iter().enumerate() {
            for o in operands(&row.expr) {
                if let Operand::Local(v) = o
                    && let Some(i) = index(v)
                {
                    readers[i].push(j);
                }
            }
            pinned[j] = row.kept || row.store.is_some() || !row.also.is_empty() || row.unread;
        }
        for o in pass
            .partials
            .iter()
            .map(|p| p.input)
            .chain(pass.shares.iter().copied())
        {
            if let Operand::Local(v) = o
                && let Some(i) = index(v)
            {
                pinned[i] = true;
            }
        }
        let mut owner: Vec<Option<(usize, bool)>> = vec![None; n];
        // The last select first, so an arm's own select claims the rows only
        // it reads after the select around it has.
        for j in (0..n).rev() {
            let Expr::Select(_, a, b) = pass.rows[j].expr else {
                continue;
            };
            for (arm, taken) in [(a, true), (b, false)] {
                let Operand::Local(v) = arm else {
                    continue;
                };
                let Some(seed) = index(v) else {
                    continue;
                };
                if pinned[seed] || readers[seed] != [j] {
                    continue;
                }
                let mut members = vec![seed];
                loop {
                    let mut grew = false;
                    for m in members.clone() {
                        for o in operands(&pass.rows[m].expr) {
                            let Operand::Local(w) = o else {
                                continue;
                            };
                            let Some(i) = index(w) else {
                                continue;
                            };
                            if members.contains(&i) || pinned[i] || i >= j {
                                continue;
                            }
                            if readers[i].iter().all(|r| members.contains(r)) {
                                members.push(i);
                                grew = true;
                            }
                        }
                    }
                    if !grew {
                        break;
                    }
                }
                for m in members {
                    owner[m] = Some((j, taken));
                }
            }
        }
        Lazy { owner }
    }

    /// The rows arm `taken` of the select at row `j` computes, in order,
    /// nested selects' own rows among them.
    fn members(&self, j: usize, taken: bool) -> Vec<usize> {
        let mut out = Vec::new();
        for (m, owner) in self.owner.iter().enumerate() {
            let mut at = *owner;
            while let Some((sel, arm)) = at {
                if sel == j && arm == taken {
                    out.push(m);
                    break;
                }
                at = self.owner[sel];
            }
        }
        out
    }
}

/// Row `j` of `pass` printed; the select at `decided`'s row as its arm
/// there (`arm`, the turned vectors' or the rest's), the rows only that arm
/// reads before it.
fn print_row(
    s: &mut String,
    pass: &Loop,
    decided: Option<(&Lazy, usize, bool)>,
    j: usize,
    indent: &str,
) -> Result<(), Unprinted> {
    let row = &pass.rows[j];
    let value = row.value;
    let mut e = match (decided, row.expr) {
        (Some((_, sel, arm)), Expr::Select(_, a, b)) if sel == j => {
            operand(if arm { a } else { b }, At::Elem)
        }
        _ => expr(&row.expr, At::Elem, row.elem),
    };
    e = held(row.round, e)?;
    if let Some(k) = row.store {
        e = format!("f.st({k}, e, {e})");
    }
    if row.kept {
        let _ = writeln!(s, "{indent}v{value}[e] = {e};");
    } else if row.unread && row.also.is_empty() {
        // a value stored and read by nothing else; one stored to more
        // results is named, the later stores reading it
        let _ = writeln!(s, "{indent}{e};");
    } else {
        let _ = writeln!(s, "{indent}const {} v{value} = {e};", c_type(row.elem));
    }
    let name = if row.kept {
        format!("v{value}[e]")
    } else {
        format!("v{value}")
    };
    for k in &row.also {
        let _ = writeln!(s, "{indent}f.st({k}, e, {name});");
    }
    Ok(())
}

/// Whether a pass shares its row (`f.sync()` after it) before any pass
/// reads a table.
fn shared_before_table(kernel: &Kernel) -> bool {
    let tabled = |e: &Expr| {
        operands(e)
            .iter()
            .any(|o| matches!(o, Operand::Table { .. }))
    };
    for phase in &kernel.phases {
        let Some(pass) = &phase.pass else {
            if phase.uniforms.iter().any(|u| tabled(&u.expr))
                || phase.heads.iter().any(|h| tabled(&h.expr))
            {
                return false;
            }
            continue;
        };
        if phase.heads.iter().any(|h| tabled(&h.expr))
            || pass.rows.iter().any(|r| tabled(&r.expr))
            || pass
                .partials
                .iter()
                .any(|p| matches!(p.input, Operand::Table { .. }))
            || pass
                .shares
                .iter()
                .any(|o| matches!(o, Operand::Table { .. }))
        {
            return false;
        }
        if !pass.shares.is_empty() {
            return true;
        }
    }
    false
}

/// The row parameters `pass` reads, by bit: what it fetches a vector at a
/// time where the skeleton streams them (`f.template fetch<m>(p)`).
fn pass_reads(pass: &Loop) -> u32 {
    pass.rows
        .iter()
        .flat_map(|r| operands(&r.expr))
        .chain(pass.partials.iter().map(|p| p.input))
        .chain(pass.shares.iter().copied())
        .chain(pass.copies.iter().map(|(_, from)| Operand::Param(*from)))
        .fold(0u32, |mask, o| match o {
            Operand::Param(k) if k < 32 => mask | 1 << k,
            _ => mask,
        })
}

/// A pass's work at element `e`: its rows (the select at `decided`'s row
/// as its arm there, the other arm's own rows left out), copies, partial
/// sums and shares.
fn elements(
    s: &mut String,
    pass: &Loop,
    decided: Option<(&Lazy, usize, bool)>,
    indent: &str,
) -> Result<(), Unprinted> {
    let skipped = decided.map(|(lazy, sel, arm)| lazy.members(sel, !arm));
    for j in 0..pass.rows.len() {
        if skipped.as_ref().is_some_and(|skip| skip.contains(&j)) {
            continue;
        }
        print_row(s, pass, decided, j, indent)?;
    }
    for (k, from) in &pass.copies {
        let _ = writeln!(s, "{indent}f.st({k}, e, f.ld({from}, e));");
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
            "{indent}if (e % 2 == 0) p{v} = {x}; else if (f.valid(e)) {acc} = {};",
            combine(p.op, &acc, &format!("({pair})")),
            v = p.value
        );
    }
    for share in &pass.shares {
        let _ = writeln!(s, "{indent}f.share(e, {});", operand(*share, At::Elem));
    }
    Ok(())
}

/// [`body`], its vectors laid out and its turn computed where taken as
/// `packing` says.
pub fn body_with(kernel: &Kernel, packing: Option<&Packing>) -> Result<String, Unprinted> {
    let first = first_reads(kernel);
    // The small vectors one a row, by value, and their lengths.
    let vectors: std::collections::HashMap<u32, u32> = kernel
        .phases
        .iter()
        .flat_map(|phase| &phase.uniforms)
        .filter_map(|u| u.vector.map(|n| (u.value, n)))
        .collect();
    let mut pass_at = 0usize;
    let mut s = String::new();
    for (v, elem) in &kernel.kept {
        let _ = writeln!(s, "  {} v{v}[F::N];", c_type(*elem));
    }
    for (v, elem) in &kernel.heads {
        let _ = writeln!(s, "  {} v{v}[F::PER];", c_type(*elem));
    }
    for (at, phase) in kernel.phases.iter().enumerate() {
        uniforms(&mut s, &phase.uniforms, &vectors, "  ")?;
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
            // A row's share before the table's first read meets every
            // thread of the block where the row is (more than a warp's
            // threads), so the table is whole by then without a barrier
            // of its own.
            if shared_before_table(kernel) {
                s.push_str("  f.sync_table();\n");
            } else {
                s.push_str("  f.sync_block();\n");
            }
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
        let here = packing.filter(|pk| pk.phase == at);
        match here {
            None => {
                s.push_str("#pragma unroll\n");
                s.push_str("  for (int e = 0; e < F::N; ++e) {\n");
                let reads = pass_reads(pass);
                if reads != 0 {
                    let _ = writeln!(s, "    if (e % 8 == 0) f.template fetch<{reads}u>(e / 8);");
                }
                elements(&mut s, pass, None, "    ")?;
                s.push_str("  }\n");
            }
            // A pair-packed turn: each vector the turned arm's alone or the
            // rest's, one branch a vector, which a warp of the rest skips.
            Some(pk) => {
                let lazy = Lazy::of(pass);
                s.push_str("#pragma unroll\n");
                s.push_str("  for (int q = 0; q < F::PER; ++q) {\n");
                let reads = pass_reads(pass);
                if reads != 0 {
                    let _ = writeln!(s, "    f.template fetch<{reads}u>(q);");
                }
                let _ = writeln!(s, "    if (f.turned(q, {}u)) {{", pk.turned.len());
                for (arm, close) in [(true, "    } else {\n"), (false, "    }\n")] {
                    s.push_str("#pragma unroll\n");
                    s.push_str("      for (int r = 0; r < 8; ++r) {\n");
                    s.push_str("        const int e = q * 8 + r;\n");
                    elements(&mut s, pass, Some((&lazy, pk.select, arm)), "        ")?;
                    s.push_str("      }\n");
                    s.push_str(close);
                }
                s.push_str("  }\n");
            }
        }
        for k in &pass.flushes {
            let _ = writeln!(s, "  f.template flush<{k}>();");
        }
        if !pass.shares.is_empty() {
            s.push_str("  f.sync();\n");
        }
        // The next pass's own parameters, issued ahead of this one's sums.
        pass_at += 1;
        for (k, at) in first.iter().enumerate() {
            if *at == Some(pass_at) {
                let _ = writeln!(s, "  f.template load<{k}>();");
            }
        }
        // A pass's several sums over the row (a blend's candidates' square
        // sums and dots) combine in one batch of the block's barriers.
        let sums: Vec<_> = pass
            .totals
            .iter()
            .filter(|t| t.seg.is_none() && t.op == Reduce::Sum)
            .collect();
        let batched = sums.len() > 1;
        if batched {
            let first = sums[0].value;
            let parts: Vec<String> = sums.iter().map(|t| format!("a{}", t.value)).collect();
            let _ = writeln!(
                s,
                "  float r{first}[{}] = {{{}}};",
                sums.len(),
                parts.join(", ")
            );
            let _ = writeln!(s, "  f.template reduce_sums<{}>(r{first});", sums.len());
            for (j, t) in sums.iter().enumerate() {
                let total = held(t.round, format!("r{first}[{j}]"))?;
                let _ = writeln!(s, "  const float v{} = {total};", t.value);
            }
        }
        for t in &pass.totals {
            if batched && t.seg.is_none() && t.op == Reduce::Sum {
                continue;
            }
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
    if let Some(packing) = packing {
        source.push_str(&packing.source());
    }
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
        Elem::U32 => "unsigned",
    }
}

/// How a body names an operand where it is read.
fn operand(o: Operand, at: At) -> String {
    match (o, at) {
        (Operand::Param(k), _) => format!("f.ld({k}, e)"),
        (Operand::One(k), _) => format!("f.one({k})"),
        (Operand::Kept(v), _) => format!("v{v}[e]"),
        (Operand::Heads(v), At::Vector) => format!("v{v}[p]"),
        (Operand::Heads(v), _) => format!("v{v}[e / 8]"),
        (Operand::Table { at, period }, _) => format!("f.table()[{at}u + f.col(e) % {period}u]"),
        (Operand::Entry(v), _) => format!("t{v}"),
        (Operand::Local(v), _) => format!("v{v}"),
        (Operand::Vector(k), _) => format!("f.vec({k}, 0)"),
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
    expr_of(e, at, elem, &|x: Operand| operand(x, at))
}

/// [`expr`] with each operand named by `o`.
fn expr_of(e: &Expr, at: At, elem: Elem, o: &dyn Fn(Operand) -> String) -> String {
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
            unreachable!("a small vector's own exprs are printed with its uniform")
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

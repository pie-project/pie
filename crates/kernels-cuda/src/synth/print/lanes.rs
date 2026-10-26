//! A kernel's small vectors one a row printed over a warp's lanes: value
//! `i` of an `n`-vector in lane `i % F::SPAN` of each warp the row takes
//! (slot `i / F::SPAN`, one slot where a row takes whole warps), a vector
//! op one op in every lane, a scalar (one a row) the same in every lane, a
//! pick of one value a shuffle to every lane (`f.pick`).
//!
//! A vector laid out of scalars (a [`Expr::Splat`] and the
//! [`Expr::Insert`]s over it) whose values are each the same op over
//! operands that differ only in the values they pick of one vector (a
//! normalization's sums laid beside the values they divide) is computed in
//! each lane from the values its own picks gather (`f.gather`): each lane
//! its own value of it, by the region's ops, in their order.
//!
//! What a pass, a head or a table reads, and what it reads in turn, every
//! warp of the row computes; what only lands (a small-vector result no
//! pass reads, a repeat over it) the row's first warp computes alone
//! (`f.lead()`), and lands it (`f.st_lane`). Where all of that is the
//! first phase's (it reads no row and nothing a pass computes), a kernel
//! with side blocks (`F::SIDE`, a decode's spread rows) computes it there
//! instead, beside the row's block, as a hand kernel walks its gates on a
//! block of its own.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

use kernels_ir::{Expr, Kernel, Operand, Uniform};

use super::{At, Unprinted, c_type, expr_of, held, operand, operands};

/// The most ops a lane's value of a laid-out vector expands to; past it
/// the vector is laid value by value.
const MOST_LIFTED: u32 = 256;

/// A kernel's uniforms, each by its value; the vectors laid out of
/// scalars that each lane computes its own value of; which every warp
/// needs, and which are printed at all.
pub(super) struct Lanes<'k> {
    defs: HashMap<u32, &'k Uniform>,
    /// Each lifted vector's lane value and the values it reads.
    lifted: HashMap<u32, (String, Vec<u32>)>,
    needed: HashSet<u32>,
    /// What a pass reads or lands, and what that reads as printed: a
    /// scalar only a lifted vector's ops read is not.
    live: HashSet<u32>,
    /// The phase whose uniforms the lead warp's own follow, if any.
    pub(super) lead_at: Option<usize>,
}

/// Whether `kernel` prints in the lane form: it holds a small vector or a
/// repeat.
pub(super) fn wanted(kernel: &Kernel) -> bool {
    kernel
        .phases
        .iter()
        .flat_map(|phase| &phase.uniforms)
        .any(|u| u.vector.is_some() || matches!(u.expr, Expr::Carry(_)))
}

impl<'k> Lanes<'k> {
    pub(super) fn of(kernel: &'k Kernel) -> Lanes<'k> {
        let defs: HashMap<u32, &Uniform> = kernel
            .phases
            .iter()
            .flat_map(|phase| &phase.uniforms)
            .map(|u| (u.value, u))
            .collect();
        let mut lanes = Lanes {
            defs,
            lifted: HashMap::new(),
            needed: HashSet::new(),
            live: HashSet::new(),
            lead_at: None,
        };
        for u in kernel.phases.iter().flat_map(|phase| &phase.uniforms) {
            if let Some(lifted) = lanes.laid(u) {
                lanes.lifted.insert(u.value, lifted);
            }
        }
        // What the row's passes, heads and tables read.
        let mut roots: Vec<u32> = Vec::new();
        let mut read = |o: Operand| {
            if let Operand::Local(v) = o {
                roots.push(v);
            }
        };
        for phase in &kernel.phases {
            for h in &phase.heads {
                operands(&h.expr).into_iter().for_each(&mut read);
            }
            if let Some(pass) = &phase.pass {
                for r in &pass.rows {
                    operands(&r.expr).into_iter().for_each(&mut read);
                }
                pass.partials.iter().for_each(|p| read(p.input));
                pass.shares.iter().copied().for_each(&mut read);
            }
        }
        for t in &kernel.tables {
            for (_, _, e) in &t.entries {
                operands(e).into_iter().for_each(&mut read);
            }
        }
        lanes.needed = lanes.closure(roots.clone());
        // and what lands
        roots.extend(
            lanes
                .defs
                .values()
                .filter(|u| !u.stores.is_empty())
                .map(|u| u.value),
        );
        lanes.live = lanes.closure(roots);
        lanes.lead_at = kernel.phases.iter().rposition(|phase| {
            blocks(&phase.uniforms).any(|(at, _)| {
                lanes.lead(&phase.uniforms, at) && lanes.live.contains(&head(&phase.uniforms, at))
            })
        });
        lanes
    }

    /// The values `roots` and those they read as printed, in turn.
    fn closure(&self, mut work: Vec<u32>) -> HashSet<u32> {
        let mut seen = HashSet::new();
        while let Some(v) = work.pop() {
            if !seen.insert(v) {
                continue;
            }
            let Some(u) = self.defs.get(&v) else {
                continue;
            };
            if let Some((_, reads)) = self.lifted.get(&v) {
                work.extend(reads);
                continue;
            }
            if let Expr::Repeat { carry, .. } = u.expr {
                work.push(carry);
            }
            for o in operands(&u.expr) {
                if let Operand::Local(w) = o {
                    work.push(w);
                }
            }
        }
        seen
    }

    fn vector(&self, v: u32) -> Option<u32> {
        self.defs.get(&v).and_then(|u| u.vector)
    }

    /// Whether what only lands may run on side blocks beside the rows'
    /// (`F::SIDE`): all of it in the first phase, so it reads no row and
    /// nothing a pass computes.
    pub(super) fn side(&self) -> bool {
        self.lead_at == Some(0)
    }

    /// A side block's work, of the first phase's uniforms `us`: what only
    /// lands, and what it reads that every warp computes, in order.
    pub(super) fn print_side(
        &self,
        s: &mut String,
        us: &[Uniform],
        indent: &str,
    ) -> Result<(), Unprinted> {
        let landing: Vec<u32> = blocks(us)
            .filter(|&(at, _)| self.lead(us, at))
            .map(|(at, _)| head(us, at))
            .filter(|v| self.live.contains(v))
            .collect();
        let reads = self.closure(landing);
        for (at, end) in blocks(us) {
            let v = head(us, at);
            if !reads.contains(&v) || !self.live.contains(&v) {
                continue;
            }
            if end > at + 1 {
                self.repeat(s, &us[at..end], indent)?;
            } else {
                self.uniform(s, &us[at], indent)?;
            }
        }
        Ok(())
    }

    /// Whether the uniform at `us[at]` (a repeat's carry standing for its
    /// whole body) only lands.
    fn lead(&self, us: &[Uniform], at: usize) -> bool {
        !self.needed.contains(&head(us, at))
    }

    /// The uniforms of `us` the row's every warp computes (`lead` false),
    /// or those the first warp alone does (`lead` true), printed.
    pub(super) fn print(
        &self,
        s: &mut String,
        us: &[Uniform],
        lead: bool,
        indent: &str,
    ) -> Result<(), Unprinted> {
        for (at, end) in blocks(us) {
            if self.lead(us, at) != lead || !self.live.contains(&head(us, at)) {
                continue;
            }
            if end > at + 1 {
                self.repeat(s, &us[at..end], indent)?;
            } else {
                self.uniform(s, &us[at], indent)?;
            }
        }
        Ok(())
    }

    /// A repeat, its carry first and its close last, as a loop around its
    /// body.
    fn repeat(&self, s: &mut String, us: &[Uniform], indent: &str) -> Result<(), Unprinted> {
        let (open, shut) = (&us[0], &us[us.len() - 1]);
        let (Expr::Carry(init), Expr::Repeat { count, next, .. }) = (open.expr, shut.expr) else {
            unreachable!("a repeat's block opens with its carry and closes with its repeat");
        };
        let (c, r) = (open.value, shut.value);
        let ty = c_type(open.elem);
        match open.vector {
            Some(n) => {
                let _ = writeln!(s, "{indent}{ty} v{c}[{}];", slots(n));
                self.each(
                    s,
                    n,
                    indent,
                    &format!("v{c}[t] = {}", self.element(init, n)),
                );
            }
            None => {
                let _ = writeln!(s, "{indent}{ty} v{c} = {};", scalar(init));
            }
        }
        let _ = writeln!(
            s,
            "{indent}for (int it{c} = 0; it{c} < {count}; ++it{c}) {{"
        );
        let inner = format!("{indent}  ");
        for u in &us[1..us.len() - 1] {
            if self.live.contains(&u.value) {
                self.uniform(s, u, &inner)?;
            }
        }
        match open.vector {
            Some(n) => self.each(
                s,
                n,
                &inner,
                &format!("v{c}[t] = {}", self.element(next, n)),
            ),
            None => {
                let _ = writeln!(s, "{inner}v{c} = {};", scalar(next));
            }
        }
        let _ = writeln!(s, "{indent}}}");
        match open.vector {
            Some(n) => {
                let _ = writeln!(s, "{indent}{ty} v{r}[{}];", slots(n));
                self.each(s, n, indent, &format!("v{r}[t] = v{c}[t]"));
            }
            None => {
                let _ = writeln!(s, "{indent}const {ty} v{r} = v{c};");
            }
        }
        self.stores(s, shut, indent);
        Ok(())
    }

    fn uniform(&self, s: &mut String, u: &Uniform, indent: &str) -> Result<(), Unprinted> {
        let (v, ty) = (u.value, c_type(u.elem));
        match u.vector {
            Some(n) => {
                let value = match u.expr {
                    Expr::Splat(x) => scalar(x),
                    Expr::Insert(src, at, x) => match self.lifted.get(&v) {
                        Some((lifted, _)) => lifted.clone(),
                        None => format!("(i == {at} ? {} : {})", scalar(x), self.element(src, n)),
                    },
                    ref e => expr_of(e, At::Elem, u.elem, &|o| self.element(o, n)),
                };
                let _ = writeln!(s, "{indent}{ty} v{v}[{}];", slots(n));
                self.each(
                    s,
                    n,
                    indent,
                    &format!("v{v}[t] = {}", held(u.round, value)?),
                );
            }
            None => {
                let value = match u.expr {
                    Expr::Pick(Operand::Vector(k), at) => format!("f.vec({k}, {at})"),
                    Expr::Pick(Operand::Local(w), at) => format!("f.pick(v{w}, {at})"),
                    ref e => expr_of(e, At::Elem, u.elem, &scalar),
                };
                let _ = writeln!(s, "{indent}const {ty} v{v} = {};", held(u.round, value)?);
            }
        }
        self.stores(s, u, indent);
        Ok(())
    }

    /// `body` (a statement over slot `t`, its value `i`) for each slot of an
    /// `n`-vector this thread holds.
    fn each(&self, s: &mut String, n: u32, indent: &str, body: &str) {
        let _ = writeln!(s, "{indent}#pragma unroll");
        let _ = writeln!(s, "{indent}for (int t = 0; t < {}; ++t) {{", slots(n));
        let _ = writeln!(
            s,
            "{indent}  [[maybe_unused]] const int i = f.vlane() + t * F::SPAN;"
        );
        let _ = writeln!(s, "{indent}  {body};");
        let _ = writeln!(s, "{indent}}}");
    }

    /// A small vector's stores, each value from the lane that holds it.
    fn stores(&self, s: &mut String, u: &Uniform, indent: &str) {
        let Some(n) = u.vector else {
            return;
        };
        for k in &u.stores {
            self.each(
                s,
                n,
                indent,
                &format!("if (i < {n}) f.template st_lane<{k}>(i, v{}[t])", u.value),
            );
        }
    }

    /// Operand `o` of an `n`-vector's op at this lane's value `i`.
    fn element(&self, o: Operand, n: u32) -> String {
        match o {
            Operand::Vector(k) => format!("(i < {n} ? f.vec({k}, i) : 0.0f)"),
            Operand::Local(w) if self.vector(w).is_some() => format!("v{w}[t]"),
            other => scalar(other),
        }
    }

    /// A vector laid out of scalars (`u` the last insert over a splat),
    /// each lane computing its own value where the values are one op over
    /// the picks of a vector ([`Lanes::lift`]).
    fn laid(&self, u: &Uniform) -> Option<(String, Vec<u32>)> {
        let n = u.vector?;
        let mut values: Vec<Option<u32>> = vec![None; n as usize];
        let mut at = u;
        loop {
            match at.expr {
                Expr::Insert(Operand::Local(src), i, Operand::Local(x)) => {
                    let slot = values.get_mut(i as usize)?;
                    if slot.is_none() {
                        *slot = Some(x);
                    }
                    at = self.defs.get(&src)?;
                }
                Expr::Splat(Operand::Local(x)) => {
                    for slot in &mut values {
                        slot.get_or_insert(x);
                    }
                    break;
                }
                _ => return None,
            }
        }
        let values: Vec<u32> = values.into_iter().collect::<Option<_>>()?;
        let mut budget = MOST_LIFTED;
        let mut reads = Vec::new();
        let lifted = self.lift(&values, n, &mut budget, &mut reads)?;
        Some((lifted, reads))
    }

    /// Scalars `values` (one a lane's value of an `n`-vector) as one
    /// expression of this lane's `i`: a value they all are; the gather of
    /// the values they pick of one vector; or the op they all are, over
    /// its operands lifted alike.
    fn lift(
        &self,
        values: &[u32],
        n: u32,
        budget: &mut u32,
        reads: &mut Vec<u32>,
    ) -> Option<String> {
        *budget = budget.checked_sub(1)?;
        if values.iter().all(|&v| v == values[0]) {
            reads.push(values[0]);
            return Some(format!("v{}", values[0]));
        }
        let us: Vec<&Uniform> = values
            .iter()
            .map(|v| self.defs.get(v).copied())
            .collect::<Option<_>>()?;
        let first = us[0];
        if us
            .iter()
            .any(|u| u.round != first.round || u.elem != first.elem || u.vector.is_some())
        {
            return None;
        }
        if let Expr::Pick(src, _) = first.expr {
            let picks: Vec<u32> = us
                .iter()
                .map(|u| match u.expr {
                    Expr::Pick(of, at) if of == src => Some(at),
                    _ => None,
                })
                .collect::<Option<_>>()?;
            let j = index(&picks);
            let gathered = match src {
                Operand::Vector(k) => format!("(i < {n} ? f.vec({k}, {j}) : 0.0f)"),
                Operand::Local(w) => {
                    reads.push(w);
                    format!("f.gather(v{w}, {j})")
                }
                _ => return None,
            };
            return held(first.round, gathered).ok();
        }
        if !liftable(&first.expr) || us.iter().any(|u| !same_op(&u.expr, &first.expr)) {
            return None;
        }
        let arity = operands(&first.expr).len();
        let mut names = Vec::with_capacity(arity);
        for at in 0..arity {
            let column: Vec<Operand> = us.iter().map(|u| operands(&u.expr)[at]).collect();
            let name = if column.iter().all(|o| *o == column[0]) {
                if let Operand::Local(v) = column[0] {
                    reads.push(v);
                }
                scalar(column[0])
            } else {
                let locals: Vec<u32> = column
                    .iter()
                    .map(|o| match o {
                        Operand::Local(v) if self.vector(*v).is_none() => Some(*v),
                        _ => None,
                    })
                    .collect::<Option<_>>()?;
                self.lift(&locals, n, budget, reads)?
            };
            names.push(name);
        }
        // the op over placeholders, each named by its lifted operand
        let mut at = 0u32;
        let shape = rebuild(&first.expr, &mut |_| {
            at += 1;
            Operand::Local(u32::MAX - (at - 1))
        });
        let e = expr_of(&shape, At::Elem, first.elem, &|o| match o {
            Operand::Local(p) => names[(u32::MAX - p) as usize].clone(),
            _ => unreachable!("every operand a placeholder"),
        });
        held(first.round, e).ok()
    }
}

/// The slots of an `n`-vector a thread holds, as C++.
fn slots(n: u32) -> String {
    format!("({n} + F::SPAN - 1) / F::SPAN")
}

/// A scalar operand where every lane reads the same.
fn scalar(o: Operand) -> String {
    operand(o, At::Elem)
}

/// The uniforms of `us` as blocks to print whole: each its own, a repeat's
/// carry through its close.
fn blocks(us: &[Uniform]) -> impl Iterator<Item = (usize, usize)> + '_ {
    let mut at = 0;
    core::iter::from_fn(move || {
        if at >= us.len() {
            return None;
        }
        let start = at;
        at = match us[at].expr {
            Expr::Carry(_) => close(us, start) + 1,
            _ => at + 1,
        };
        Some((start, at))
    })
}

/// The value that stands for the block at `us[at]`: a repeat's its close's.
fn head(us: &[Uniform], at: usize) -> u32 {
    match us[at].expr {
        Expr::Carry(_) => us[close(us, at)].value,
        _ => us[at].value,
    }
}

/// The repeat that closes the carry at `us[at]`.
fn close(us: &[Uniform], at: usize) -> usize {
    let carry = us[at].value;
    (at + 1..us.len())
        .find(|&k| matches!(us[k].expr, Expr::Repeat { carry: c, .. } if c == carry))
        .expect("a carry's repeat closes it in its phase")
}

/// Lane `i`'s pick of `picks` (`picks[i]`, 0 past them), as C++: an offset
/// of `i` where they are one, else a table of them.
fn index(picks: &[u32]) -> String {
    let offset = picks[0];
    if (0u32..).zip(picks).all(|(i, &p)| p == i + offset) {
        return if offset == 0 {
            "i".into()
        } else {
            format!("(i + {offset})")
        };
    }
    let mut words = [0u64; 3];
    for (i, &p) in picks.iter().enumerate() {
        words[i / 12] |= u64::from(p & 31) << (5 * (i % 12));
    }
    format!(
        "f.index(i, {}ull, {}ull, {}ull)",
        words[0], words[1], words[2]
    )
}

/// Whether a lane may compute `e` from its operands lifted: an op of
/// scalars alone.
fn liftable(e: &Expr) -> bool {
    matches!(
        e,
        Expr::Copy(_)
            | Expr::Unary(..)
            | Expr::Binary(..)
            | Expr::Select(..)
            | Expr::Cast { .. }
            | Expr::Fma(..)
    )
}

/// Whether `a` and `b` are the same op, their operands aside.
fn same_op(a: &Expr, b: &Expr) -> bool {
    let blank = |e: &Expr| rebuild(e, &mut |_| Operand::Local(0));
    blank(a) == blank(b)
}

/// `e` with each of its operands (in [`operands`]' order) replaced.
fn rebuild(e: &Expr, f: &mut dyn FnMut(Operand) -> Operand) -> Expr {
    match *e {
        Expr::Copy(a) => Expr::Copy(f(a)),
        Expr::Unary(u, a) => Expr::Unary(u, f(a)),
        Expr::Binary(op, a, b) => {
            let a = f(a);
            Expr::Binary(op, a, f(b))
        }
        Expr::Select(c, a, b) => {
            let c = f(c);
            let a = f(a);
            Expr::Select(c, a, f(b))
        }
        Expr::Cast { value, from, to } => Expr::Cast {
            value: f(value),
            from,
            to,
        },
        Expr::Fma(a, b, c) => {
            let a = f(a);
            let b = f(b);
            Expr::Fma(a, b, f(c))
        }
        other => other,
    }
}

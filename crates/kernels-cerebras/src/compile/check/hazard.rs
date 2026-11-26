//! A PE's code read for what runs it off its rails before or during a launch ([`super::code`]):
//!
//! - **Startup tasks**: a data task (`@bind_data_task`) or an activation at startup runs over
//!   state the host has not written yet.
//! - **Host words unbounded**: a DSD's length, offset or base from a word the host writes a step
//!   ([`STEP_WORDS`]) with no `below(v, n)` or `v % N` between.
//! - **Overlap**: a DSD operation whose destination and a source overlap at different elements.
//! - **Line windows** ([`lines`]): a GEMM-regime queue rebound among a line's colours: H1-H4.
//! - **Hub loops** ([`hub_loops`]): a tiled-regime blocking send of more words than the hub's
//!   loop holds before its receive is posted.
//! - **Re-binds**: an output queue re-coloured during a launch with no drain wait in the
//!   function (a send still in it would leave on the new colour).
//! - **Late inputs** ([`late_inputs`]): a ring's input queue re-coloured during a launch to a
//!   colour another unit (or the head) sends on: nothing holds the next pass's words back.
//! - **Seats** ([`seats`]): a ring of more seats than the units' hand-offs hold; ([`seat_pool`]) a
//!   KV pool of more positions than CY's word to the attention carries.

use std::collections::{BTreeMap, BTreeSet};

use super::code::{Code, Module};
use super::{call_args, eval};
use crate::compile::abi::Kernels;
use crate::compile::fabric::Colour;
use crate::compile::floor::Coord;
use crate::compile::program::Program;
use crate::compile::wiring::Link;

/// The hazards of `file` at `params` (its imports from `files`), one line each.
#[must_use]
pub fn hazards(
    files: &BTreeMap<String, String>,
    file: &str,
    params: &BTreeMap<String, i64>,
) -> Vec<String> {
    let c = super::code::code(files, file, params);
    let mut out = BTreeSet::new();
    startup(&c, file, &mut out);
    host_words(&c, file, &mut out);
    overlaps(&c, file, &mut out);
    rebinds(&c, file, &mut out);
    out.into_iter().collect()
}

/// A ring of `m` seats over `units` units: every hand-off is synchronous, so a pass waits for the
/// next unit to take the one before; `m` passes in flight over `units` units and the head close
/// the loop only while one unit is free (`m <= units - 1`), else they deadlock.
#[must_use]
pub fn seats(m: u32, units: u32) -> Option<String> {
    (m > 1 && m >= units).then(|| {
        format!(
            "a ring of {m} seats over {units} units: past units - 1, the synchronous hand-offs \
             deadlock"
        )
    })
}

/// A ring's KV pool of `pool` positions: CY's word to the attention carries a window's position
/// in its low 16 bits (2 seat + its reset above), so a position past them reads as another seat's.
#[must_use]
pub fn seat_pool(pool: u32) -> Option<String> {
    (pool > 1 << 16).then(|| {
        format!("a KV pool of {pool} positions: past the 65536 the attention's seat word holds")
    })
}

/// The input queues of a ring of seats re-coloured during a launch to a colour whose words come
/// from outside the PE's unit, one line a kernel's host word. A pass ahead sends on that colour
/// whenever its own unit is done (the hand-offs carry no credit), so its words may be at the
/// queue when it is re-coloured; an input from within the unit waits for the PE's own pass.
/// Such an input holds a queue bound at the start. A colour the unit's own CX sends on after the
/// PE's last partial ([`Late::Posted`]) may be bound once no later word of the queue's own colour
/// can come, which is before that partial is sent: bound after it, the order is only timing's.
pub fn late_inputs(
    prog: &Program,
    kinds: &Kernels,
    own: &[(String, String)],
) -> Result<Vec<String>, String> {
    let files: BTreeMap<String, String> = own.iter().cloned().collect();
    let mut binds: BTreeMap<&str, (String, BTreeMap<i64, bool>)> = BTreeMap::new();
    let mut out = BTreeSet::new();
    for (at, colour, late) in late_links(prog)? {
        let kind = prog.floor.roles.get(at).kind;
        if !binds.contains_key(kind.stem) {
            let file = format!("{}.csl", kind.stem);
            let params = (kinds.get(&kind).into_iter())
                .flat_map(|k| &k.params.items)
                .filter_map(|(n, v)| Some((n.to_string(), eval(&v.render(), &BTreeMap::new())?)))
                .collect();
            let words = bound_words(&super::code::code(&files, &file, &params))
                .map_err(|e| format!("`{file}` {e}"))?;
            binds.insert(kind.stem, (file, words));
        }
        let (file, words) = &binds[kind.stem];
        if words.is_empty() {
            continue;
        }
        let rcfg = crate::compile::emit::rcfg::of(prog, at)?;
        for (&i, &before) in words {
            if late == Late::Posted && before {
                continue;
            }
            if usize::try_from(i).ok().and_then(|i| rcfg.get(i)) == Some(&u32::from(colour.0)) {
                let why = match late {
                    Late::Outside => {
                        "a colour from outside the PE's unit (a later pass's words may come on it \
                         before the queue is restored)"
                    }
                    Late::Posted => {
                        "a colour the unit's CX sends on, after the PE's last partial (a later \
                         pass's words may come on the queue's own colour before it is re-coloured)"
                    }
                };
                out.insert(format!(
                    "`{file}`: an input queue re-coloured during a launch to rcfg[{i}], {why}"
                ));
            }
        }
    }
    Ok(out.into_iter().collect())
}

/// When a unit PE's input from another PE comes, seen from the PE's own pass.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Late {
    /// From outside the unit: at any time.
    Outside,
    /// The unit's CX sends it once it holds the PE's last partial of a pass.
    Posted,
}

/// The unit PEs a link from outside their unit, or a path from their unit's CX, delivers to, and
/// its colour.
fn late_links(prog: &Program) -> Result<Vec<(Coord, Colour, Late)>, String> {
    let unit = |c: Coord| prog.floor.units.iter().position(|u| u.rect.contains(c));
    let mut out = Vec::new();
    for l in &prog.wiring.links {
        let (src, class, to): (Option<&Coord>, _, Vec<Coord>) = match l {
            Link::Path(p) => (
                p.pts.first(),
                p.class,
                p.pts.last().copied().into_iter().collect(),
            ),
            Link::Multicast(m) => (m.pts.get(m.src), m.class, m.pts.clone()),
            Link::Broadcast(b) => {
                let to = b.pts.iter().zip(&b.recv).filter(|(_, r)| **r);
                (b.pts.get(b.src), b.class, to.map(|(c, _)| *c).collect())
            }
            _ => continue,
        };
        let Some(&src) = src else { continue };
        let colour = prog.palette.get(class)?;
        let path = matches!(l, Link::Path(_));
        for c in to {
            let Some(u) = unit(c) else { continue };
            if unit(src) != Some(u) {
                out.push((c, colour, Late::Outside));
            } else if path && src.x == prog.floor.units[u].cx_x() {
                out.push((c, colour, Late::Posted));
            }
        }
    }
    Ok(out)
}

/// The rcfg words a task's input-queue re-binds (`iqc`) take their colours from, a function's
/// parameter by each of its call sites: whether a row partial's send (`send_x`) follows at every
/// one, in the function or after its call.
fn bound_words(c: &Code) -> Result<BTreeMap<i64, bool>, String> {
    let reached = c.task_reached();
    let mut out = BTreeMap::new();
    for (m, func) in &reached {
        let f = &c.modules[*m].funcs[func];
        for (at, _) in f.body.match_indices("iqc(") {
            let args = call_args(&f.body[at..]).unwrap_or_default();
            let Some(colour) = args.get(1).filter(|_| args.len() == 2) else {
                continue;
            };
            for (w, _) in colour.match_indices("rcfg[") {
                let e = &colour[w + 5..];
                let e = &e[..e.find(']').unwrap_or(e.len())];
                for (mut env, after) in site_envs(c, &reached, *m, func) {
                    if ruled_out(&f.body, at, &env) {
                        continue;
                    }
                    // (a name of an imported module: `pe.RQ`)
                    let mut flat = e.to_string();
                    for (alias, inst) in &c.modules[*m].aliases {
                        for (n, v) in &c.modules[*inst].env {
                            let q = format!("{alias}.{n}");
                            if flat.contains(&q) {
                                flat = flat.replace(&q, n);
                                env.insert(n.clone(), *v);
                            }
                        }
                    }
                    let word = eval(&flat, &env)
                        .ok_or_else(|| format!("{func}(): a queue's colour from rcfg[{e}]"))?;
                    *out.entry(word).or_insert(true) &= after || sends_row(c, *m, &f.body[at..]);
                }
            }
        }
    }
    Ok(out)
}

/// A row partial's send.
const SEND_ROW: &str = "send_x(";

/// Whether `text` (module `m`'s) sends a row partial: itself, or by a call of a function of the
/// module that does.
fn sends_row(c: &Code, m: usize, text: &str) -> bool {
    text.contains(SEND_ROW)
        || (c.modules[m].funcs.iter())
            .filter(|(_, f)| f.body.contains(SEND_ROW))
            .any(|(n, _)| calls(text, n))
}

/// Whether `text` calls `name` (unqualified).
fn calls(text: &str, name: &str) -> bool {
    text.match_indices(&format!("{name}("))
        .any(|(i, _)| !text[..i].ends_with(|ch: char| ident(ch) || ch == '.'))
}

/// Whether the call at `i` of `body` sits in an `if` whose condition is a conjunction with a term
/// false at `env` (a branch the params rule out).
fn ruled_out(body: &str, i: usize, env: &BTreeMap<String, i64>) -> bool {
    let b = body.as_bytes();
    let mut depth = 0i32;
    let mut at = i;
    while at > 0 {
        at -= 1;
        match b[at] {
            b'}' => depth += 1,
            b'{' if depth > 0 => depth -= 1,
            b'{' => {
                let head = body[..at].trim_end();
                if let Some(h) = head.strip_suffix(')') {
                    let mut d = 1;
                    let open = h.char_indices().rev().find(|&(_, ch)| {
                        d += match ch {
                            ')' => 1,
                            '(' => -1,
                            _ => 0,
                        };
                        d == 0
                    });
                    if let Some((o, _)) = open {
                        let cond = &h[o + 1..];
                        if h[..o].trim_end().ends_with("if")
                            && !cond.contains(" or ")
                            && cond.split(" and ").any(|t| eval(t.trim(), env) == Some(0))
                        {
                            return true;
                        }
                    }
                }
            }
            _ => {}
        }
    }
    false
}

/// The values of `func`'s (module `m`'s) names at each call of it from what a task reaches, and
/// whether its caller sends a row partial after it.
fn site_envs(
    c: &Code,
    reached: &BTreeSet<(usize, String)>,
    m: usize,
    func: &str,
) -> Vec<(BTreeMap<String, i64>, bool)> {
    let (base, params) = (&c.modules[m].env, &c.modules[m].funcs[func].params);
    let mut out = Vec::new();
    let mut ruled = false;
    for (at, caller) in reached {
        let body = &c.modules[*at].funcs[caller].body;
        for (i, _) in body.match_indices(&format!("{func}(")) {
            let before = &body[..i];
            if before.ends_with(ident) {
                continue;
            }
            let qual = before.strip_suffix('.').map(|b| {
                let s = b.rfind(|ch| !ident(ch)).map_or(0, |p| p + 1);
                &b[s..]
            });
            if c.resolve(*at, qual, func) != Some((m, func.to_string())) {
                continue;
            }
            if ruled_out(body, i, &c.modules[*at].env) {
                ruled = true;
                continue;
            }
            let args = call_args(&body[i..]).unwrap_or_default();
            let mut env = base.clone();
            for (p, a) in params.iter().zip(&args) {
                if let Some(v) = eval(a, &c.modules[*at].env) {
                    env.insert(p.clone(), v);
                }
            }
            out.push((env, sends_row(c, *at, &body[i..])));
        }
    }
    if out.is_empty() && !ruled {
        out.push((base.clone(), false));
    }
    out
}

/// Output queues re-coloured by what a task runs (`oqc`), in a function with no drain wait
/// (`output_queue_status`) of its own or of a function it calls.
fn rebinds(c: &Code, file: &str, out: &mut BTreeSet<String>) {
    let spins = |m: usize, f: &str| c.modules[m].funcs[f].body.contains("output_queue_status");
    for (m, func) in c.task_reached() {
        let body = &c.modules[m].funcs[&func].body;
        if !body.contains("oqc(") {
            continue;
        }
        if spins(m, &func) || c.callees(m, &func).iter().any(|(t, g)| spins(*t, g)) {
            continue;
        }
        out.insert(format!(
            "`{file}` {func}(): an output queue re-bound during a launch with no drain wait (a \
             send still in it leaves on the new colour)"
        ));
    }
}

/// Data tasks and activations at startup.
fn startup(c: &Code, file: &str, out: &mut BTreeSet<String>) {
    for m in &c.modules {
        if m.outside.contains("@bind_data_task(") {
            out.insert(format!(
                "`{file}`: a data task (it runs on any wavelet its colour brings, before the \
                 launch too)"
            ));
        }
        for block in comptime_blocks(&m.outside) {
            if block.contains("@activate(") {
                out.insert(format!(
                    "`{file}`: an activation at startup (a task runs before the host writes its \
                     state)"
                ));
            }
        }
    }
}

/// The bodies of the `comptime { .. }` blocks in `text`.
fn comptime_blocks(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for (at, _) in text.match_indices("comptime {") {
        let cs: Vec<char> = text[at + 9..].chars().collect();
        let mut depth = 0;
        for (i, &ch) in cs.iter().enumerate() {
            match ch {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        out.push(cs[1..i].iter().collect());
                        break;
                    }
                }
                _ => {}
            }
        }
    }
    out
}

/// The statements of a body: its text cut at `;`, `{` and `}`.
fn fragments(body: &str) -> Vec<&str> {
    body.split([';', '{', '}'])
        .map(str::trim)
        .filter(|f| !f.is_empty())
        .collect()
}

fn ident(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// The assignments in a statement: (qualifier, name, declared here, right-hand side).
fn assignments(f: &str) -> Vec<(Option<String>, String, bool, String)> {
    let mut out = Vec::new();
    for op in [" = ", " += ", " -= ", " *= "] {
        for (at, _) in f.match_indices(op) {
            let lhs = f[..at].trim_end();
            let rhs = f[at + op.len()..].trim();
            // the declared name: `const n`, `var n: T`
            let (lhs, declared) = match lhs.rsplit_once(':') {
                Some((l, ty)) if !ty.contains('(') && !l.contains("while") => (l.trim_end(), true),
                _ => (lhs, false),
            };
            let name: String = lhs
                .chars()
                .rev()
                .take_while(|&c| ident(c) || c == '.')
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            if name.is_empty() || name.ends_with(']') {
                continue;
            }
            let before = lhs[..lhs.len() - name.len()].trim_end();
            let declared = declared || before.ends_with("const") || before.ends_with("var");
            let (q, n) = match name.rsplit_once('.') {
                Some((q, n)) => (Some(q.to_string()), n.to_string()),
                None => (None, name),
            };
            out.push((q, n, declared, rhs.to_string()));
        }
    }
    out
}

/// The names `text` reads, each with its qualifier (an address taken, `&x`, reads no value).
fn names(text: &str) -> Vec<(Option<String>, String)> {
    let cs: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < cs.len() {
        if cs[i] == '@' {
            i += 1;
            while i < cs.len() && ident(cs[i]) {
                i += 1;
            }
        } else if cs[i].is_ascii_alphabetic() || cs[i] == '_' {
            let s = i;
            while i < cs.len() && (ident(cs[i]) || cs[i] == '.') {
                i += 1;
            }
            let w: String = cs[s..i].iter().collect();
            if s > 0 && cs[s - 1] == '&' {
                continue;
            }
            match w.rsplit_once('.') {
                Some((q, n)) => out.push((Some(q.to_string()), n.to_string())),
                None => out.push((None, w)),
            }
        } else {
            i += 1;
        }
    }
    out
}

/// A taint's key: (module, the function a local is of, name).
type Key = (usize, Option<String>, String);

/// The key name `(q, n)` read in function `func` of module `m` refers to.
fn key(c: &Code, m: usize, func: &str, q: Option<&str>, n: &str) -> Option<Key> {
    if let Some(q) = q {
        let t = *c.modules[m].aliases.get(q)?;
        return Some((t, None, n.to_string()));
    }
    let f = &c.modules[m].funcs[func];
    let local = f.params.iter().any(|p| p == n)
        || f.body.contains(&format!("const {n} "))
        || f.body.contains(&format!("var {n}:"))
        || f.body.contains(&format!("var {n} "));
    Some((m, local.then(|| func.to_string()), n.to_string()))
}

/// Whether `rhs` bounds what it reads: `below(..)`, or a remainder by a comptime value.
fn bounded(rhs: &str, env: &BTreeMap<String, i64>) -> bool {
    if rhs.contains("below(") {
        return true;
    }
    // the divisor: what follows the last `%`, its parentheses closed as far as they open
    rhs.rsplit_once('%').is_some_and(|(_, d)| {
        let d = d.trim().trim_end_matches(';');
        let mut depth = 0i32;
        let mut end = d.len();
        for (i, ch) in d.char_indices() {
            match ch {
                '(' => depth += 1,
                ')' if depth == 0 => {
                    end = i;
                    break;
                }
                ')' => depth -= 1,
                _ => {}
            }
        }
        eval(&d[..end], env).is_some()
    })
}

/// The exported arrays the host writes a step, by their symbols.
pub const STEP_WORDS: [&str; 4] = ["rowp", "ptab", "pos", "scal"];

/// The exported arrays the host writes a step: their keys.
fn exported(c: &Code) -> Vec<Key> {
    let mut out = Vec::new();
    for (mi, m) in c.modules.iter().enumerate() {
        for (at, _) in m.outside.match_indices("@export_symbol(") {
            let Some(a) = call_args(&m.outside[at..]) else {
                continue;
            };
            if a.len() < 2 || !STEP_WORDS.contains(&a[1].trim().trim_matches('"')) {
                continue;
            }
            let p = a[0].trim();
            // `var p: [*]T = &x;` or `&mod.x`
            let Some(d) = m
                .outside
                .split(';')
                .find(|s| s.trim().starts_with(&format!("var {p}:")))
            else {
                continue;
            };
            let Some((_, rhs)) = d.split_once('=') else {
                continue;
            };
            let Some(target) = rhs.trim().strip_prefix('&') else {
                continue;
            };
            let (q, n) = match target.trim().rsplit_once('.') {
                Some((q, n)) => (Some(q), n),
                None => (None, target.trim()),
            };
            let t = match q {
                Some(q) => match m.aliases.get(q) {
                    Some(&t) => t,
                    None => continue,
                },
                None => mi,
            };
            out.push((t, None, n.to_string()));
        }
    }
    out
}

/// DSD lengths, offsets and bases from host words with no bound between.
fn host_words(c: &Code, file: &str, out: &mut BTreeSet<String>) {
    let mut taint: BTreeSet<Key> = exported(c).into_iter().collect();
    if taint.is_empty() {
        return;
    }
    let tainted = |taint: &BTreeSet<Key>, m: usize, func: &str, text: &str| -> Option<String> {
        names(text).into_iter().find_map(|(q, n)| {
            let k = key(c, m, func, q.as_deref(), &n)?;
            taint.contains(&k).then_some(n)
        })
    };
    // the taint to its fixpoint: assignments, and calls' arguments into their parameters
    loop {
        let before = taint.len();
        for (m, func) in &c.reached {
            let (m, f) = (*m, &c.modules[*m].funcs[func]);
            let env = &c.modules[m].env;
            for frag in fragments(&f.body) {
                for (q, n, _, rhs) in assignments(frag) {
                    if tainted(&taint, m, func, &rhs).is_some()
                        && !bounded(&rhs, env)
                        && let Some(k) = key(c, m, func, q.as_deref(), &n)
                    {
                        taint.insert(k);
                    }
                }
                for (q, n) in names(frag) {
                    let Some((t, callee)) = c.resolve(m, q.as_deref(), &n) else {
                        continue;
                    };
                    let call = format!("{}{n}(", q.map(|q| format!("{q}.")).unwrap_or_default());
                    let Some(at) = frag.find(&call) else { continue };
                    let Some(args) = call_args(&frag[at..]) else {
                        continue;
                    };
                    let ps = &c.modules[t].funcs[&callee].params;
                    for (a, p) in args.iter().zip(ps) {
                        if tainted(&taint, m, func, a).is_some() && !bounded(a, env) {
                            taint.insert((t, Some(callee.clone()), p.clone()));
                        }
                    }
                }
            }
        }
        if taint.len() == before {
            break;
        }
    }
    for (m, func) in &c.reached {
        let f = &c.modules[*m].funcs[func];
        for b in [
            "@set_dsd_length(",
            "@increment_dsd_offset(",
            "@set_dsd_base_addr(",
        ] {
            for (at, _) in f.body.match_indices(b) {
                let Some(args) = call_args(&f.body[at..]) else {
                    continue;
                };
                for a in args.iter().skip(1) {
                    if let Some(n) = tainted(&taint, *m, func, a) {
                        out.insert(format!(
                            "`{file}` {func}(): `{}` from a host word (`{n}`) with no bound",
                            b.trim_end_matches('(')
                        ));
                    }
                }
            }
        }
    }
}

/// A DSD's extent: its array (module, name), element bytes, offset and stride (elements),
/// length.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Dsd {
    arr: (usize, String),
    eb: i64,
    off: i64,
    stride: i64,
    len: i64,
}

impl Dsd {
    /// The bytes it covers (`None` past a size worth enumerating).
    fn bytes(&self) -> Option<BTreeSet<i64>> {
        if self.len <= 0 || self.len > 1 << 14 || self.stride <= 0 {
            return None;
        }
        let mut s = BTreeSet::new();
        for k in 0..self.len {
            let at = (self.off + k * self.stride) * self.eb;
            for b in 0..self.eb {
                s.insert(at + b);
            }
        }
        Some(s)
    }
}

fn type_bytes(t: &str) -> Option<i64> {
    let t = t.trim();
    if t.ends_with("f32") || t.ends_with("u32") || t.ends_with("i32") {
        Some(4)
    } else if t.ends_with("u16")
        || t.ends_with("i16")
        || t.ends_with("f16")
        || t.contains("@fp16()")
    {
        Some(2)
    } else {
        None
    }
}

/// Array `n`'s element bytes in module `m`.
fn elem(m: &Module, n: &str) -> Option<i64> {
    let d = m.outside.split(';').find(|s| {
        let s = s.trim();
        s.starts_with(&format!("var {n} ")) || s.starts_with(&format!("var {n}:"))
    })?;
    let (_, rhs) = d.split_once('=')?;
    let rhs = rhs.trim();
    let ty = if let Some(z) = rhs.strip_prefix("@zeros([") {
        z.split_once(']')?.1.trim_end_matches(')').to_string()
    } else {
        d.split_once(':')?.1.split_once(']')?.1.to_string()
    };
    type_bytes(
        ty.trim_end_matches(|c: char| c == ')' || c.is_whitespace())
            .split(' ')
            .next()?,
    )
}

/// Where DSD names resolve: a function's bound parameters and locals, over its module's.
struct Scope<'a> {
    c: &'a Code,
    m: usize,
    local: BTreeMap<String, Option<Dsd>>,
}

impl Scope<'_> {
    fn module(&self) -> &Module {
        &self.c.modules[self.m]
    }

    /// Global `n`'s DSD: its initialiser where nothing reached sets it.
    fn global(&self, m: usize, n: &str) -> Option<Dsd> {
        let md = &self.c.modules[m];
        let d = md.outside.split(';').find_map(|s| {
            let s = s.trim();
            let rest = s
                .strip_prefix(&format!("const {n} = "))
                .or_else(|| s.strip_prefix(&format!("var {n} = ")))?;
            Some((s.starts_with("var "), rest.to_string()))
        })?;
        if d.0 {
            // a var: set by anything reached, unknown
            for (fm, func) in &self.c.reached {
                let body = &self.c.modules[*fm].funcs[func].body;
                for frag in fragments(body) {
                    for (q, an, decl, _) in assignments(frag) {
                        let target = match &q {
                            Some(q) => self.c.modules[*fm].aliases.get(q).copied(),
                            None if !decl => Some(*fm),
                            None => None,
                        };
                        if an == n && target == Some(m) {
                            return None;
                        }
                    }
                }
            }
        }
        let s = Scope {
            c: self.c,
            m,
            local: BTreeMap::new(),
        };
        s.dsd(&d.1)
    }

    /// The DSD expression `e` makes, where its extent is known.
    fn dsd(&self, e: &str) -> Option<Dsd> {
        let e = e.trim();
        let env = &self.module().env;
        if let Some(rest) = e.strip_prefix("@get_dsd(") {
            let ta = rest.split_once(".tensor_access")?.1;
            let len = eval(ta.split_once('{')?.1.split_once('}')?.0, env)?;
            let (arr, idx) = ta.split_once("->")?.1.trim().split_once('[')?;
            let idx = idx.rsplit_once(']')?.0.trim();
            let (off, stride) = if idx == "i" {
                (0, 1)
            } else if let Some(s) = idx.strip_prefix("i * ").or_else(|| idx.strip_prefix("i*")) {
                (0, eval(s, env)?)
            } else {
                (eval(idx.strip_prefix("i + ")?, env)?, 1)
            };
            let arr = arr.trim();
            return Some(Dsd {
                arr: (self.m, arr.to_string()),
                eb: elem(self.module(), arr)?,
                off,
                stride,
                len,
            });
        }
        for (b, n) in [
            ("@set_dsd_length(", 2),
            ("@increment_dsd_offset(", 3),
            ("@set_dsd_stride(", 2),
            ("@set_dsd_base_addr(", 2),
        ] {
            if !e.starts_with(b) {
                continue;
            }
            let a = call_args(e)?;
            if a.len() != n {
                return None;
            }
            let mut d = self.dsd(&a[0])?;
            match b {
                "@set_dsd_length(" => d.len = eval(&a[1], env)?,
                "@set_dsd_stride(" => d.stride = eval(&a[1], env)?,
                "@increment_dsd_offset(" => {
                    let k = eval(&a[1], env)?;
                    let tb = type_bytes(&a[2])?;
                    if (k * tb) % d.eb != 0 {
                        return None;
                    }
                    d.off += k * tb / d.eb;
                }
                _ => {
                    // @ptrcast([*]T, &arr)
                    let p = a[1].strip_prefix("@ptrcast(")?;
                    let pa = call_args(&format!("f({p}"))?;
                    let arr = pa.get(1)?.trim().strip_prefix('&')?.to_string();
                    d.arr = (self.m, arr);
                    d.eb = type_bytes(pa[0].trim_start_matches("[*]"))?;
                    d.off = 0;
                }
            }
            return Some(d);
        }
        if e.chars().all(|c| ident(c) || c == '.') {
            if let Some(d) = self.local.get(e) {
                return d.clone();
            }
            return match e.rsplit_once('.') {
                Some((q, n)) => self.global(*self.module().aliases.get(q)?, n),
                None => self.global(self.m, e),
            };
        }
        None
    }
}

/// The DSD operations a fabric checks for overlap (destination first).
const OPS: [&str; 26] = [
    "fmovs", "fmovh", "fadds", "faddh", "fsubs", "fsubh", "fmuls", "fmulh", "fmacs", "fmach",
    "fnegs", "fnegh", "fmaxs", "fmaxh", "fs2h", "fh2s", "fabss", "fabsh", "add16", "sub16",
    "sll16", "slr16", "sra16", "mov16", "mov32", "and16",
];

/// Same-op destination / source overlaps whose extents are known.
fn overlaps(c: &Code, file: &str, out: &mut BTreeSet<String>) {
    for (m, func) in &c.reached {
        let s = Scope {
            c,
            m: *m,
            local: BTreeMap::new(),
        };
        body_overlaps(c, s, func, file, out, 0);
    }
}

fn body_overlaps(
    c: &Code,
    mut s: Scope<'_>,
    func: &str,
    file: &str,
    out: &mut BTreeSet<String>,
    depth: usize,
) {
    let m = s.m;
    let body = c.modules[m].funcs[func].body.clone();
    for frag in fragments(&body) {
        // a local's DSD (a set one unknown from here on)
        for (q, n, _, rhs) in assignments(frag) {
            if q.is_none() && rhs.trim_start().starts_with('@')
                || q.is_none() && s.local.contains_key(&n)
            {
                let d = s.dsd(&rhs);
                s.local.insert(n, d);
            }
        }
        for op in OPS {
            let pat = format!("@{op}(");
            for (at, _) in frag.match_indices(&pat) {
                let Some(args) = call_args(&frag[at..]) else {
                    continue;
                };
                let Some(dst) = args.first().and_then(|a| s.dsd(a)) else {
                    continue;
                };
                let Some(db) = dst.bytes() else { continue };
                for a in args.iter().skip(1) {
                    let Some(src) = s.dsd(a) else { continue };
                    if src.arr != dst.arr || src == dst {
                        continue;
                    }
                    if src.bytes().is_some_and(|sb| !sb.is_disjoint(&db)) {
                        out.insert(format!(
                            "`{file}` {func}(): `@{op}` writes `{}` over its source `{}` \
                             (elements of `{}` overlapping, not the same)",
                            args[0].trim(),
                            a.trim(),
                            dst.arr.1
                        ));
                    }
                }
            }
        }
        // a call: its callee's body with the DSDs passed bound
        if depth >= 3 {
            continue;
        }
        for (q, n) in names(frag) {
            let Some((t, callee)) = c.resolve(m, q.as_deref(), &n) else {
                continue;
            };
            let call = format!("{}{n}(", q.map(|q| format!("{q}.")).unwrap_or_default());
            let Some(at) = frag.find(&call) else { continue };
            let Some(args) = call_args(&frag[at..]) else {
                continue;
            };
            let ps = c.modules[t].funcs[&callee].params.clone();
            let local: BTreeMap<String, Option<Dsd>> = args
                .iter()
                .zip(&ps)
                .map(|(a, p)| (p.clone(), s.dsd(a)))
                .collect();
            if local.values().all(Option::is_none) {
                continue;
            }
            let inner = Scope { c, m: t, local };
            body_overlaps(c, inner, &callee, file, out, depth + 1);
        }
    }
}

/// A PE's input queue the GEMM regime's code rebinds among colours, and the function that does:
/// a line (H1-H4).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Line {
    pub queue: u32,
    pub colours: BTreeSet<String>,
    pub func: String,
}

/// The body of `fn name(` in `text` (its braces' inside), if there is one.
fn body<'a>(text: &'a str, name: &str) -> Option<&'a str> {
    let at = text.find(&format!("fn {name}("))?;
    let open = at + text[at..].find('{')?;
    let mut depth = 0usize;
    for (i, ch) in text[open..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&text[open + 1..open + i]);
                }
            }
            _ => {}
        }
    }
    None
}

/// The lines of `pe` (a GEMM PE's code): every input queue a function rebinds to more than
/// one named colour (`fab.iqc(q, C_..)`), by the function that does.
#[must_use]
pub fn line_queues(pe: &str) -> Vec<Line> {
    let mut by: BTreeMap<(u32, String), BTreeSet<String>> = BTreeMap::new();
    for func in crate::kernel::gemm_forms().map(|f| f.func) {
        let Some(b) = body(pe, func) else { continue };
        for frag in b.split("fab.iqc(").skip(1) {
            let Some(end) = frag.find(')') else { continue };
            let args: Vec<&str> = frag[..end].split(',').map(str::trim).collect();
            if let [q, c] = args[..]
                && let Ok(q) = q.parse::<u32>()
                && c.starts_with("C_")
            {
                by.entry((q, func.to_string()))
                    .or_default()
                    .insert(c.to_string());
            }
        }
    }
    by.into_iter()
        .filter(|(_, c)| c.len() > 1)
        .map(|((queue, func), colours)| Line {
            queue,
            colours,
            func,
        })
        .collect()
}

/// A line `l` whose function alternates the colour it starts on, one run to the next (its first
/// rebind picks between two colours by the run's parity): each run starts on the colour the last
/// ended on, so there is no window between runs.
fn line_alternates(pe: &str, l: &Line) -> bool {
    let Some(b) = body(pe, &l.func) else {
        return false;
    };
    let bind = format!("fab.iqc({}, ", l.queue);
    let Some(at) = b.find(&bind) else {
        return false;
    };
    let before = b[..at].trim_end();
    let cond = before.ends_with('{')
        && before
            .rsplit('\n')
            .next()
            .is_some_and(|ln| ln.trim_start().starts_with("if ("));
    let rest = &b[at + bind.len()..];
    let Some(end) = rest.find(')') else {
        return false;
    };
    let first = rest[..end].trim();
    let other = rest[end..]
        .strip_prefix(");")
        .map(str::trim_start)
        .and_then(|r| r.strip_prefix('}'))
        .map(str::trim_start)
        .and_then(|r| r.strip_prefix("else {"))
        .map(str::trim_start)
        .and_then(|r| r.strip_prefix(bind.as_str()))
        .and_then(|r| r.find(')').map(|e| r[..e].trim()));
    cond && other.is_some_and(|o| o != first && l.colours.contains(o) && l.colours.contains(first))
}

/// The attention ends in a row-wide ordering point: an all-reduce over the row after its last
/// key block (`gseg.allred(.., false)` or `gseg.all_reduce(`), so no PE of the row leaves it
/// before every other has reached its end.
pub(crate) fn attention_orders_row(pe: &str) -> bool {
    body(pe, "attend").is_some_and(|b| {
        let tail = b.rfind("gseg.allred(").map_or("", |i| &b[i..]);
        let last = tail.find(';').map_or(tail, |e| &tail[..e]);
        b.contains("gseg.all_reduce(")
            || last.trim_end_matches(')').trim().ends_with("false")
            || b.contains("row_barrier(")
    })
}

/// The row-line windows of a GEMM-regime program `p` whose PEs run `pe` (`emit::gemm::GPE_OPS`),
/// one line each: where a queue rebound among a line's colours can meet a wavelet of a colour it
/// is not bound to (the wavelet holds the PE's ramp and what the PE waits on never comes).
///
/// Each op orders a row (`row`), lets its bands drift apart (`skew`: the attention, unless it
/// ends in a row-wide ordering point), or neither.
///
/// - **H1** (inside an op): a rebind no later sender is ordered after: covered by H2.
/// - **H2** (between ops): a row-line op while an op since the last row-wide ordering point is
///   `skew`. Column line: an attention after another where the line leaves `KB` bound and the
///   next starts on `KF`, unless the attentions alternate their order (`line_alternates`).
/// - **H3** (no overtaking): covered by H2's ordering point.
/// - **H4** (row drift): an attention with ops after it and no row-wide ordering point at its end.
///
/// A line on queues of its own (no queue rebound) has no window.
#[must_use]
pub fn lines(p: &crate::compile::stream::gemm::GemmProgram, pe: &str) -> Vec<String> {
    use crate::compile::stream::gemm::GOp;
    use crate::kernel::LineAlong;
    let ls = line_queues(pe);
    // each op's kernel's GEMM form and what it declares of `pe`: its line, whether it orders
    // the row, whether a column line it rebinds alternates its first colour
    struct Op<'f> {
        form: &'f crate::kernel::GemmForm,
        line: Option<&'f Line>,
        orders: bool,
        shut: bool,
    }
    let forms: Vec<Option<Op>> = (p.l.ops.iter())
        .map(|op| {
            let form = crate::kernel::gemm_form(op)?;
            let line = ls.iter().find(|l| l.func == form.func);
            Some(Op {
                form,
                line,
                orders: (form.orders_row)(pe),
                shut: line.is_some_and(|l| line_alternates(pe, l)),
            })
        })
        .collect();
    let mut out = Vec::new();
    let classes: BTreeSet<usize> = p.row_class.iter().copied().collect();
    // (a region's all-reduces over the row, as printed)
    let reduces = |r: usize| p.l.regs[r].printed.as_ref().map_or(0, |x| x.all_reduces);
    for &c in &classes {
        let bit = 1u16 << c;
        // (the last op since a row-wide ordering point that couples only a band)
        let mut skew: Option<(usize, &str)> = None;
        // (each form's last op, by its function)
        let mut last: BTreeMap<&str, usize> = BTreeMap::new();
        for (i, op) in p.l.ops.iter().enumerate() {
            if let GOp::Region(r) = op {
                if p.l.regs[*r].mask & bit != 0 && reduces(*r) > 0 {
                    skew = None;
                }
                continue;
            }
            let Some(o) = &forms[i] else { continue };
            let mask = match op {
                GOp::Gemm { mask, .. } => *mask,
                _ => u16::MAX,
            };
            if o.form.masked && mask & bit == 0 {
                continue;
            }
            match (o.form.line, o.line) {
                (LineAlong::Row, Some(l)) => {
                    if let Some((j, why)) = skew {
                        out.push(format!(
                            "class {c}: op {i}, {}'s row line (iq{} rebound {:?}), follows op \
                             {j} ({why}) with no row-wide ordering point between: its first \
                             pieces can reach PEs still in op {j} on a colour iq{} is not bound \
                             to (H2)",
                            o.form.what, l.queue, l.colours, l.queue
                        ));
                    }
                }
                // (no op orders a column: the tape's words are buffered, not a barrier)
                (LineAlong::Column, Some(l)) => {
                    if let (Some(&j), false) = (last.get(o.form.func), o.shut) {
                        out.push(format!(
                            "op {i}: {} after op {j}'s with no column-wide ordering \
                             point between: the column line (iq{} rebound {:?}) leaves its PEs \
                             on its last colour, and the next one's first pieces can reach PEs \
                             still in op {j} (H2)",
                            o.form.what, l.queue, l.colours
                        ));
                    }
                }
                (_, None) => {}
            }
            last.insert(o.form.func, i);
            skew = if o.orders {
                None
            } else {
                Some((i, o.form.drift))
            };
        }
    }
    // H4: a row's bands drift apart over an op that does not order it; with no row-wide ordering
    // point after, a band ahead meets one behind and the row stops, own queues or not
    for (i, o) in forms.iter().enumerate() {
        if let Some(o) = o
            && !o.orders
            && i + 1 < p.l.ops.len()
        {
            out.push(format!(
                "op {i}: {} with no row-wide ordering point at its end, ops after it: {} (H4)",
                o.form.what, o.form.drift_after
            ));
        }
    }
    out.sort();
    out.dedup();
    out
}

/// Words the hub's loop holds for certain (the routers and queues from a PE's send to the hub
/// and back along the broadcast tree to its own receive).
pub const HUB_LOOP_WORDS: u32 = 32;

/// The statements of `body` before its first that receives from `queue` (`fab.fin(<queue>`).
fn before_receive<'a>(body: &'a str, queue: &str) -> Option<&'a str> {
    body.find(&format!("fab.fin({queue}")).map(|at| &body[..at])
}

/// The blocking sends of `text` (a function's statements, cut at `;`: an async op's options
/// `.{ .. }` stay with it): a fabric send (`fab.fout(`) or a forward (`fwd(`) not posted async.
fn blocking_sends(text: &str) -> Vec<String> {
    text.split(';')
        .map(|f| f.trim().trim_start_matches(['{', '}', ' ', '\n']).trim())
        .filter(|f| {
            (f.contains("fab.fout(") || f.starts_with("fwd(") || f.contains(" fwd("))
                && !f.contains(".async = true")
        })
        .map(str::to_string)
        .collect()
}

/// The hub's loops of a tiled-regime program `p` whose image PEs run `pe`, one line each: an op
/// whose PEs send towards the hub and receive back on the broadcast tree, where a send blocks
/// before the receive is posted and sends more than [`HUB_LOOP_WORDS`]: the broadcast fills the
/// PE's queue, stalling the tree, the hub and the send.
#[must_use]
pub fn hub_loops(p: &crate::compile::stream::tiled::TileProgram, pe: &str) -> Vec<String> {
    let blocking = |func: &str| {
        body(pe, func)
            .and_then(|b| before_receive(b, "q_bi"))
            .map(blocking_sends)
            .unwrap_or_default()
    };
    let mut out = Vec::new();
    for (i, op) in p.ops.iter().enumerate() {
        let Some(f) = crate::kernel::tiled_form(op).and_then(|f| f.hub.as_ref()) else {
            continue;
        };
        let (words, sends) = ((f.words)(p, op), blocking(f.func));
        if words > HUB_LOOP_WORDS && !sends.is_empty() {
            out.push(format!(
                "op {i}: {}, {words} words a PE sends before its receive of the broadcast is \
                 posted (`{}`), past the {HUB_LOOP_WORDS} the hub's loop holds",
                f.what, sends[0]
            ));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn files(pe: &str) -> BTreeMap<String, String> {
        let mut f: BTreeMap<String, String> = BTreeMap::new();
        for (n, s) in crate::sources::OPS
            .iter()
            .chain(crate::sources::stream_ops())
        {
            f.insert(format!("ops/{n}"), (*s).to_string());
        }
        f.insert("pe.csl".into(), pe.into());
        f
    }

    const SATT: &str = "param C: u16;\nconst fab = @import_module(\"ops/fabric.csl\");\n\
        const math = @import_module(\"ops/math.csl\");\nconst q = @get_input_queue(2);\n\
        const red = @import_module(\"ops/schain.csl\", .{ .fab = fab, .IN1 = q, .OUT = @get_output_queue(2) });\n\
        const att = @import_module(\"ops/satt.csl\", .{ .fab = fab, .math = math, .red = red, .YM = q, .Qh = 2, .Dc = 128, .NE = 16, .PYA = 8, .SC = 0.1, .QH = true, .PAGED = true, .EP = 8, .LANES = 1, .MAXP = 2, .ROWS = 1, .CAUSAL = true });\n\
        var pcfg = @zeros([2]u32);\nvar rowp_p: [*]u32 = &att.rowp;\nvar ptab_p: [*]u32 = &att.ptab;\n\
        task go() void {\n  SETUP\n  att.store_t(0);\n  att.query_t(0);\n}\n\
        comptime {\n  @bind_local_task(go, @get_local_task_id(8));\n  @export_symbol(rowp_p, \"rowp\");\n  @export_symbol(ptab_p, \"ptab\");\n}\n";

    #[test]
    fn a_paged_column_without_its_bind_overlaps_and_its_host_words_are_bounded() {
        let bad = SATT.replace("SETUP", "att.myrow = @as(u16, pcfg[0]);");
        let h = hazards(&files(&bad), "pe.csl", &BTreeMap::new());
        assert!(
            h.iter().any(|l| l.contains("@add16") && l.contains("`kk`")),
            "{h:?}"
        );
        let good = SATT.replace("SETUP", "att.bind(@as(u16, pcfg[0]));");
        let h = hazards(&files(&good), "pe.csl", &BTreeMap::new());
        assert!(h.is_empty(), "{h:?}");
    }

    #[test]
    fn a_host_word_into_an_offset_and_startup_tasks_are_found() {
        let pe = "var rows = @zeros([4]u32);\nvar rows_p: [*]u32 = &rows;\nvar buf = @zeros([64]f32);\n\
                  const b_d = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{8} -> buf[i] });\n\
                  fn at(s: u32) void { @fmovs(@increment_dsd_offset(b_d, @as(i16, s), f32), 0.0); }\n\
                  fn ok(s: u32) void { @fmovs(@increment_dsd_offset(b_d, @as(i16, s), f32), 0.0); }\n\
                  task go() void {\n  const s = rows[0] * 8;\n  at(s);\n  const t = below(rows[1], 7) * 8;\n  ok(t);\n}\n\
                  comptime {\n  @bind_data_task(go, @get_data_task_id(3));\n  @activate(@get_local_task_id(9));\n  @export_symbol(rows_p, \"rowp\");\n}\n";
        let h = hazards(&files(pe), "pe.csl", &BTreeMap::new());
        assert_eq!(h.len(), 3, "{h:?}");
        assert!(
            h.iter()
                .any(|l| l.contains("at(): `@increment_dsd_offset` from a host word"))
        );
        assert!(h.iter().any(|l| l.contains("a data task")));
        assert!(h.iter().any(|l| l.contains("activation at startup")));
    }

    #[test]
    fn a_queue_rebound_in_a_task_without_a_drain_and_too_many_seats_are_found() {
        let pe = "const fab = @import_module(\"ops/fabric.csl\");\n\
                  const tile_config = @import_module(\"<tile_config>\");\n\
                  fn init() void { fab.oqc(4, 9); }\n\
                  fn send() void { fab.oqc(4, 7); }\n\
                  fn wait() void { while ((tile_config.output_queue_status.get_raw() & 0x1000) == 0) {} }\n\
                  fn safe() void { wait(); fab.oqc(4, 7); }\n\
                  task go() void { send(); safe(); }\n\
                  comptime {\n  @bind_local_task(go, @get_local_task_id(8));\n  @export_symbol(init);\n}\n";
        let h = hazards(&files(pe), "pe.csl", &BTreeMap::new());
        assert_eq!(h.len(), 1, "{h:?}");
        assert!(h[0].contains("send(): an output queue re-bound"), "{h:?}");
        assert_eq!(seats(1, 1), None);
        assert_eq!(seats(7, 8), None);
        assert!(seats(8, 8).is_some());
        assert_eq!(seat_pool(1 << 16), None);
        assert!(seat_pool((1 << 16) + 1).is_some());
    }

    #[test]
    fn a_handoff_taken_on_a_queue_rebound_in_a_ring_is_found() {
        let mut plan = crate::compile::target::searched("llama31-8b-bf16-kv-bf16").plan;
        plan.seats = 2;
        let levers = crate::compile::emit::params::Levers::of(&plan.levers, "").unwrap();
        let prog = Program::of(plan).unwrap();
        let kinds = crate::compile::emit::params::kernels(&prog, &levers).unwrap();
        let mut own: Vec<(String, String)> = kinds
            .iter()
            .filter(|(_, kb)| !kb.source.is_empty())
            .map(|(k, kb)| {
                let text = prog.source(kb.source, crate::sources::get(kb.source).unwrap());
                (format!("{}.csl", k.stem), text.unwrap())
            })
            .collect();
        for (n, s) in crate::sources::OPS {
            own.extend(prog.source(n, s).map(|s| (format!("ops/{n}"), s)));
        }
        assert_eq!(late_inputs(&prog, &kinds, &own), Ok(Vec::new()));
        // the pass-end turn's colour bound after the last partial, and the pass-start turn's during
        // a launch
        let send = "pe.send_x(Hc);\n  }\n  return bo";
        for (from, to, found) in [
            (
                format!("if (l + 1 == K) {{ pe.post_in(); }}\n  {send}"),
                send.replace("send_x(Hc);", "send_x(Hc); pe.post_in();"),
                "rcfg[55]",
            ),
            (
                "pe.relay_in(18)".into(),
                "pe.relay(18, 1)".into(),
                "rcfg[52]",
            ),
        ] {
            let mut own = own.clone();
            for (_, s) in &mut own {
                *s = s.replace(&from, &to);
                if found == "rcfg[52]" {
                    let oq = "fab.oqc(4, rcfg[RQ + o + 2]);";
                    *s = s.replacen(oq, &format!("fab.iqc(6, rcfg[RQ + o + 1]); {oq}"), 1);
                }
            }
            let h = late_inputs(&prog, &kinds, &own).unwrap();
            assert!(h.iter().any(|h| h.contains(found)), "{from}: {h:?}");
        }
    }

    /// klein's tiled decode at a 4 x 4 latent, first 19 ops: with the sends blocking, its
    /// attention's keys are a hub loop and its norms none; as emitted (async sends), none.
    #[test]
    fn kleins_attention_keys_are_a_hub_loop_unless_sent_async() {
        let t = poem_compiler::catalog::repository()
            .parse("flux2-klein-4b-bf16-kv-bf16")
            .unwrap()
            .trace(poem_ir::Platform::Cerebras);
        let p = crate::compile::stream::tiled::program_upto(
            &t,
            "vae.decode",
            [1, 4, 4],
            true,
            Some((1, 1)),
            None,
            Some(19),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        let pe = crate::compile::emit::tiled::TPE_OPS;
        assert_eq!(hub_loops(&p, pe), Vec::<String>::new(), "as emitted");
        let blocking = pe
            .replace(", .{ .async = true, .ut_id = ut_e }", "")
            .replace(", .{ .async = true, .ut_id = ut_s }", "")
            .replace(", .{ .async = true, .ut_id = ut }", "");
        assert_ne!(blocking, pe);
        let open = hub_loops(&p, &blocking);
        assert_eq!(open.len(), 1, "{open:?}");
        assert!(
            open[0].starts_with("op 18: an attention's key, 512 words"),
            "{open:?}"
        );
    }

    /// flux2-mini's denoise in the GEMM regime, laid at 32 x 8 PEs, key blocks of 8 (as `m2`),
    /// as emitted, its line windows not checked.
    fn mini_gemm() -> crate::compile::stream::gemm::GemmProgram {
        let o = crate::compile::stream::Options {
            reading: "denoise".into(),
            rows: 16,
            round16: false,
            upto: None,
            taps: Vec::new(),
            anneal: Default::default(),
        };
        let t = poem_compiler::catalog::repository()
            .parse("flux2-mini-bf16-kv-bf16")
            .unwrap()
            .trace(poem_ir::Platform::Cerebras);
        let l = crate::compile::stream::gemm::lower(&t, &o, &[]).unwrap_or_else(|e| panic!("{e}"));
        crate::compile::stream::gemm::laid(&l, 32, 8, 8).unwrap_or_else(|e| panic!("{}", e.text()))
    }

    /// The GEMM PE's lines: the column line (iq6, `KF` / `KB`) in its attention; the row line's
    /// colours each on a queue of its own (no line).
    #[test]
    fn the_gemm_pes_lines() {
        let ls = line_queues(crate::compile::emit::gemm::GPE_OPS);
        let named: Vec<(u32, &str, Vec<&str>)> = ls
            .iter()
            .map(|l| {
                (
                    l.queue,
                    l.func.as_str(),
                    l.colours.iter().map(String::as_str).collect(),
                )
            })
            .collect();
        assert_eq!(named, vec![(6, "attend", vec!["C_KB", "C_KF"])]);
    }

    /// With the row line rebound on iq2, every GEMM after an attention is a window; with a
    /// row-wide all-reduce at the attention's end, none; as emitted (own queues), none.
    #[test]
    fn the_row_lines_windows_open_when_rebound_and_shut_by_v14_or_own_queues() {
        let p = mini_gemm();
        let pe = crate::compile::emit::gemm::GPE_OPS;
        // as emitted: XF and XB each on a queue bound once
        assert!(
            line_queues(pe).iter().all(|l| l.func != "gemm"),
            "the emitted row line rebinds"
        );
        assert_eq!(lines(&p, pe), Vec::<String>::new(), "as emitted");
        // the row line rebound on iq2 at a member's turn and back at the GEMM's end
        let own = "      if (cp < X) { @mov32(xhw(0, nw), fab.fin(q_xi, nw)); } else { @mov32(xhw(0, nw), fab.fin(q_xb, nw)); }\n";
        let tail = "  if ((fl & 1) != 0) { rnd(ys, rows * ylw); }\n}";
        let turn =
            "    const kc = if (even) kd else plen(cp, GX, K);\n    if (kc == 0) { continue; }";
        assert!(pe.contains(own) && pe.contains(tail) && pe.contains(turn));
        // (as built before the row ordering point at each attention's end)
        let rebound = pe
            .replace("  gseg.allred(md(RT, 1), 1, false);\n", "")
            .replace(own, "      @mov32(xhw(0, nw), fab.fin(q_xi, nw));\n")
            .replace(tail, &format!("  fab.iqc(2, C_XF);\n{tail}"))
            .replacen(
                turn,
                &format!("    if (cp == X) {{ fab.iqc(2, C_XB); }}\n{turn}"),
                1,
            );
        let open = lines(&p, &rebound);
        assert!(
            open.iter().any(|w| w.contains("(H2)"))
                && open
                    .iter()
                    .all(|w| w.contains("(H2)") || w.contains("(H4)")),
            "{open:?}"
        );
        eprintln!("{} windows; {}", open.len(), open[0]);
        // a one-word row all-reduce after the last key block
        let end = "  t = 0;\n  while (t < TB) : (t += 1) {\n    @fmuls(md(oo + t * lo, nc), md(oo + t * lo, nc), math.recip(ll[t]));";
        assert!(rebound.contains(end));
        let v14 = rebound.replace(end, &format!("  gseg.allred(md(RT, 1), 1, false);\n{end}"));
        assert_eq!(lines(&p, &v14), Vec::<String>::new(), "v14");
    }

    /// H4: as emitted every attention ends in a row-wide ordering point; without it, every
    /// attention with an op after it is a window, the lines on their own queues or not.
    #[test]
    fn an_attention_with_no_row_ordering_point_is_a_window() {
        let p = mini_gemm();
        let pe = crate::compile::emit::gemm::GPE_OPS;
        assert!(attention_orders_row(pe));
        let bar = "  gseg.allred(md(RT, 1), 1, false);\n";
        assert_eq!(pe.matches(bar).count(), 1);
        let open = lines(&p, &pe.replace(bar, ""));
        let after =
            p.l.ops
                .iter()
                .enumerate()
                .filter(|(i, o)| {
                    matches!(o, crate::compile::stream::gemm::GOp::Attend { .. })
                        && i + 1 < p.l.ops.len()
                })
                .count();
        assert!(after > 0);
        assert_eq!(
            open.iter().filter(|w| w.contains("(H4)")).count(),
            after,
            "{open:?}"
        );
    }

    /// The column line: as emitted, the attentions alternate their members' order, so none is a
    /// window; with the order fixed (`KF` bound at each attention's start), every attention after
    /// the first is one, also with `KF` rebound after the last piece.
    #[test]
    fn the_column_lines_windows_open_unless_the_attentions_alternate() {
        let p = mini_gemm();
        let pe = crate::compile::emit::gemm::GPE_OPS;
        let col = line_queues(pe)
            .into_iter()
            .find(|l| l.func == "attend")
            .expect("the column line");
        assert!(line_alternates(pe, &col), "as emitted");
        assert_eq!(lines(&p, pe), Vec::<String>::new(), "as emitted");
        let attends =
            p.l.ops
                .iter()
                .filter(|o| matches!(o, crate::compile::stream::gemm::GOp::Attend { .. }))
                .count();
        assert!(attends > 1, "{attends} attentions");
        let start = "  if (rev) { fab.iqc(6, C_KB); } else { fab.iqc(6, C_KF); }\n";
        assert_eq!(pe.matches(start).count(), 1);
        let fixed = pe.replace(start, "  fab.iqc(6, C_KF);\n");
        let open = lines(&p, &fixed);
        assert_eq!(open.len(), attends - 1, "{open:?}");
        assert!(
            open.iter()
                .all(|w| w.contains("column line") && w.contains("(H2)"))
        );
        let turn = "        if (rev) { fab.iqc(6, C_KF); } else { fab.iqc(6, C_KB); }\n";
        assert_eq!(fixed.matches(turn).count(), 1);
        let v17 = fixed.replace(
            turn,
            &format!("{turn}        if (Y + 1 == R0 + RQ) {{ fab.iqc(6, C_KF); }}\n"),
        );
        assert_eq!(lines(&p, &v17).len(), attends - 1, "v17");
    }
}

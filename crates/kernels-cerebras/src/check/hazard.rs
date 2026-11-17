//! A PE's code read for what runs it off its rails before or during a launch, over what it
//! reaches ([`super::code`]):
//!
//! - **Before launch**: a data task (`@bind_data_task`: it runs on any wavelet its colour
//!   brings, the host's writes and the memcpy halo's traffic before the program's launch among
//!   them) or an activation at startup (`@activate` in a comptime block) runs over state the
//!   host has not written yet.
//! - **Host words unbounded**: a DSD's length, offset or base computed from a word the host
//!   writes a step ([`STEP_WORDS`]: the state root's rows, page tables and positions, a row's
//!   scalars) with no bound between (the compile's own statics, written once, are not):
//!   a word the host has not written, or a wrong one, makes a DSD past its array (on the
//!   simulator, a fault far from its cause). A value is bounded by `below(v, n)` (`ops/
//!   fabric.csl`) or a remainder by a comptime value (`v % N`).
//! - **Overlap**: a DSD operation whose destination and a source are the same array at
//!   overlapping elements, not the same ones (the fabric's "src dest operand overlap"), where
//!   both DSDs' extents are known: a DSD a `var` holds is its initialiser where nothing the PE
//!   reaches sets it (ops/satt.csl's exp view, unset where a paged column skipped its init).

use std::collections::{BTreeMap, BTreeSet};

use super::code::{Code, Module};
use super::{call_args, eval};

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
    out.into_iter().collect()
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

#[cfg(test)]
mod tests {
    use super::*;

    fn files(pe: &str) -> BTreeMap<String, String> {
        let mut f: BTreeMap<String, String> = BTreeMap::new();
        for (n, s) in crate::sources::OPS.iter().chain(crate::sources::STREAM_OPS) {
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
}

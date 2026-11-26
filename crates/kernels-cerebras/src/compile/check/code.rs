//! A PE's code, from what cslc compiles of its files: the functions its tasks, its exports and
//! its comptime bindings reach (an unreached function is not compiled), their statements
//! after the branches their comptime conditions cut, an `inline fn` once a call site.

use std::collections::{BTreeMap, BTreeSet};

use super::{call_args, eval, fields, uncomment};

/// A function's (or task's) measures and the names it refers to (`(qualifier, name)`).
#[derive(Clone, Debug, Default)]
pub(super) struct Func {
    inline: bool,
    task: bool,
    measure: Measure,
    refs: Vec<(Option<String>, String)>,
    /// Its parameters' names, and its body (its comptime branches cut).
    pub(super) params: Vec<String>,
    pub(super) body: String,
}

/// One instance of a file: its functions, the names its declarations and comptime blocks refer
/// to (its roots), and its imports' instances by alias.
#[derive(Clone, Debug, Default)]
pub(super) struct Module {
    pub(super) funcs: BTreeMap<String, Func>,
    roots: Vec<(Option<String>, String)>,
    pub(super) aliases: BTreeMap<String, usize>,
    /// Its comptime names' values, and its text outside functions (declarations, comptime
    /// blocks; branches cut).
    pub(super) env: BTreeMap<String, i64>,
    pub(super) outside: String,
}

/// Code's measures: statements, DSD builtins, loops, `if`s (an unrolled loop's body each
/// time, an inline function's each call site).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Measure {
    pub statements: i64,
    pub dsd: i64,
    pub loops: i64,
    pub ifs: i64,
}

impl Measure {
    /// `o`'s measures `times` over (its statements once: an unrolled body's are a sequence of
    /// its DSD operations and branches).
    fn add(&mut self, o: &Measure, times: i64) {
        self.statements += o.statements;
        self.dsd += o.dsd * times;
        self.loops += o.loops;
        self.ifs += o.ifs * times;
    }
}

/// What a PE's code reaches: the functions compiled and their measures.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Reached {
    pub functions: i64,
    pub measure: Measure,
}

const KEYWORDS: [&str; 12] = [
    "if", "else", "while", "for", "return", "const", "var", "fn", "task", "and", "or", "break",
];

/// The identifiers `text` refers to, each with the identifier before a `.` it follows.
fn refs(text: &str) -> Vec<(Option<String>, String)> {
    let cs: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    let mut prev: Option<String> = None;
    while i < cs.len() {
        let c = cs[i];
        if c == '@' || c == '"' {
            // a builtin's name, a string: not a reference
            let q = c == '"';
            i += 1;
            while i < cs.len()
                && (if q {
                    cs[i] != '"'
                } else {
                    cs[i].is_ascii_alphanumeric() || cs[i] == '_'
                })
            {
                i += 1;
            }
            i += usize::from(q);
            prev = None;
        } else if c.is_ascii_alphabetic() || c == '_' {
            let s = i;
            while i < cs.len() && (cs[i].is_ascii_alphanumeric() || cs[i] == '_') {
                i += 1;
            }
            let id: String = cs[s..i].iter().collect();
            let qual = (s > 0 && cs[s - 1] == '.').then(|| prev.clone()).flatten();
            if !KEYWORDS.contains(&id.as_str()) {
                out.push((qual, id.clone()));
            }
            prev = Some(id);
        } else {
            if c != '.' {
                prev = None;
            }
            i += 1;
        }
    }
    out
}

/// A body's measures: statements, DSD builtins, loops, `if`s; a loop cslc unrolls (trip count
/// within [`UNROLL_TRIP`], body within [`UNROLL_CHARS`]) counted that many times.
fn measure(body: &str, env: &BTreeMap<String, i64>) -> Measure {
    let cs: Vec<char> = body.chars().collect();
    let mut out = Measure::default();
    let mut flat = String::new();
    let mut i = 0;
    while i < cs.len() {
        if cs[i..].starts_with(&['w', 'h', 'i', 'l', 'e', ' ', '('])
            && (i == 0 || !(cs[i - 1].is_ascii_alphanumeric() || cs[i - 1] == '_'))
            && let Some((trip, inner, end)) = a_loop(&cs, i, env)
        {
            let chars = inner.chars().filter(|c| !c.is_whitespace()).count() as i64;
            let times = match trip {
                Some(t) if t <= UNROLL_TRIP && t * chars <= UNROLL_CHARS => t,
                _ => 1,
            };
            out.add(&measure(&inner, env), times);
            out.loops += 1;
            i = end;
            continue;
        }
        flat.push(cs[i]);
        i += 1;
    }
    let n = |p: &str| flat.matches(p).count() as i64;
    out.add(
        &Measure {
            statements: n(";"),
            dsd: n("@f") + n("@mov"),
            loops: 0,
            ifs: n("if (") + n("if("),
        },
        1,
    );
    out
}

/// The longest trip count of a loop cslc unrolls, and its unrolled body's characters at most.
const UNROLL_TRIP: i64 = 64;
const UNROLL_CHARS: i64 = 4000;

/// The loop `while (v < E) : (v += 1) { .. }` at `at`: its trip count where `E` is known and
/// the body does not set `v`, its body, and where it ends.
fn a_loop(
    cs: &[char],
    at: usize,
    env: &BTreeMap<String, i64>,
) -> Option<(Option<i64>, String, usize)> {
    let open = at + 6;
    let mut depth = 0;
    let mut close = None;
    for (i, &c) in cs.iter().enumerate().skip(open) {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    close = Some(i);
                    break;
                }
            }
            _ => {}
        }
    }
    let close = close?;
    let cond: String = cs[open + 1..close].iter().collect();
    let brace = cs[close..].iter().position(|&c| c == '{')? + close;
    let end = closing(cs, brace)?;
    let body: String = cs[brace + 1..end].iter().collect();
    let trip = cond.split_once(" < ").and_then(|(v, e)| {
        let v = v.trim();
        let sets = [" = ", " += ", " -= "]
            .iter()
            .any(|op| body.contains(&format!("{v}{op}")));
        if sets { None } else { eval(e, env) }
    });
    Some((trip, body, end + 1))
}

/// The index of the brace closing the one at `open`.
fn closing(cs: &[char], open: usize) -> Option<usize> {
    let mut depth = 0;
    for (i, &c) in cs.iter().enumerate().skip(open) {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

/// `body` with its `if (c) { .. } else { .. }` cut to the branch taken where `c` evaluates
/// under `env` (a comptime condition: the other branch is not compiled).
pub(super) fn prune(body: &str, env: &BTreeMap<String, i64>) -> String {
    let cs: Vec<char> = body.chars().collect();
    let mut out = String::with_capacity(body.len());
    let mut i = 0;
    while i < cs.len() {
        let at_if = cs[i..].starts_with(&['i', 'f', ' ', '('])
            && (i == 0 || !(cs[i - 1].is_ascii_alphanumeric() || cs[i - 1] == '_'));
        if at_if && let Some(cut) = cut_if(&cs, i, env) {
            out.push_str(&prune(&cut.0, env));
            i = cut.1;
            continue;
        }
        out.push(cs[i]);
        i += 1;
    }
    out
}

/// The `if` at `at`, its condition known: the branch taken and where the statement ends.
fn cut_if(cs: &[char], at: usize, env: &BTreeMap<String, i64>) -> Option<(String, usize)> {
    let open = at + 3;
    let mut depth = 0;
    let mut close = None;
    for (i, &c) in cs.iter().enumerate().skip(open) {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    close = Some(i);
                    break;
                }
            }
            _ => {}
        }
    }
    let close = close?;
    let cond: String = cs[open + 1..close].iter().collect();
    let v = eval(&cond, env)?;
    let mut j = close + 1;
    while j < cs.len() && cs[j].is_whitespace() {
        j += 1;
    }
    if cs.get(j) != Some(&'{') {
        return None;
    }
    let end = closing(cs, j)?;
    let then: String = cs[j + 1..end].iter().collect();
    let mut k = end + 1;
    while k < cs.len() && cs[k].is_whitespace() {
        k += 1;
    }
    let (other, stop) = if cs[k..].starts_with(&['e', 'l', 's', 'e']) {
        let mut m = k + 4;
        while m < cs.len() && cs[m].is_whitespace() {
            m += 1;
        }
        if cs.get(m) == Some(&'{') {
            let e = closing(cs, m)?;
            (cs[m + 1..e].iter().collect::<String>(), e + 1)
        } else if cs[m..].starts_with(&['i', 'f', ' ', '(']) {
            // `else if`: the rest is one statement; known, it cuts the same way
            match cut_if(cs, m, env) {
                Some((s, e)) => (s, e),
                None => return None,
            }
        } else {
            return None;
        }
    } else {
        (String::new(), end + 1)
    };
    Some((if v != 0 { then } else { other }, stop))
}

/// The modules a PE's code instantiates from `file` at `params` (the first its own), each
/// import its own instance.
fn modules(
    files: &BTreeMap<String, String>,
    file: &str,
    params: &BTreeMap<String, i64>,
    passed: &BTreeMap<String, usize>,
    out: &mut Vec<Module>,
    depth: usize,
) -> usize {
    let me = out.len();
    out.push(Module::default());
    let Some(text) = files.get(file) else {
        return me;
    };
    let src = uncomment(text);
    // the comptime names (parameters, consts; not vars, which a branch may read at run time)
    let mut env = params.clone();
    for stmt in src.split(';') {
        let s = super::decl(stmt);
        if let Some(rest) = s
            .strip_prefix("const ")
            .or_else(|| s.strip_prefix("param "))
            && let Some((name, v)) = rest.split_once('=')
        {
            let name = name.split(':').next().unwrap_or("").trim().to_string();
            if !env.contains_key(&name)
                && let Some(v) = eval(v, &env)
            {
                env.insert(name, v);
            }
        }
    }
    let mut m = Module {
        aliases: passed.clone(),
        ..Module::default()
    };
    let cs: Vec<char> = src.chars().collect();
    let mut outside = String::new();
    let (mut stmt_start, mut depth_out) = (0usize, 0i32);
    let mut i = 0;
    let boundary = |i: usize| i == 0 || !(cs[i - 1].is_ascii_alphanumeric() || cs[i - 1] == '_');
    while i < cs.len() {
        let rest = &cs[i..];
        let kind = if !boundary(i) || depth_out != 0 {
            None
        } else if rest.starts_with(&"inline fn ".chars().collect::<Vec<_>>()) {
            Some((true, false, 10))
        } else if rest.starts_with(&['f', 'n', ' ']) {
            Some((false, false, 3))
        } else if rest.starts_with(&['t', 'a', 's', 'k', ' ']) {
            Some((false, true, 5))
        } else {
            None
        };
        if let Some((inline, task, skip)) = kind {
            let name: String = cs[i + skip..]
                .iter()
                .take_while(|c| c.is_ascii_alphanumeric() || **c == '_')
                .collect();
            if let Some(open) = cs[i..].iter().position(|&c| c == '{').map(|p| p + i)
                && let Some(end) = closing(&cs, open)
            {
                let body: String = cs[open + 1..end].iter().collect();
                let body = prune(&body, &env);
                let head: String = cs[i + skip..open].iter().collect();
                let params = call_args(&head)
                    .unwrap_or_default()
                    .iter()
                    .filter_map(|a| a.split(':').next())
                    .map(|n| n.trim().trim_start_matches("comptime ").trim().to_string())
                    .filter(|n| !n.is_empty())
                    .collect();
                m.funcs.insert(
                    name,
                    Func {
                        inline,
                        task,
                        measure: measure(&body, &env),
                        refs: refs(&body),
                        params,
                        body,
                    },
                );
                i = end + 1;
                continue;
            }
        }
        // a declaration at the top: an import's instance (a comptime block's statements alike)
        let c = cs[i];
        outside.push(c);
        i += 1;
        match c {
            '(' | '{' | '[' => depth_out += 1,
            ')' | '}' | ']' => depth_out -= 1,
            _ => {}
        }
        if c == ';' && depth_out == 0 {
            let stmt = super::decl(&outside[stmt_start..]).to_string();
            declared(files, &stmt, &env, &mut m, out, depth);
            stmt_start = outside.len();
        } else if c == '}' && depth_out == 0 {
            stmt_start = outside.len();
        }
    }
    // (a function the comptime blocks call runs at compile time: the roots are what they bind
    // to tasks and export)
    let outside = prune(&outside, &env);
    m.env = env;
    for key in [
        "@bind_local_task(",
        "@bind_data_task(",
        "@bind_control_task(",
        "@export_symbol(",
    ] {
        for (at, _) in outside.match_indices(key) {
            if let Some(a) = call_args(&outside[at..])
                && let Some(first) = a.first()
            {
                m.roots.extend(refs(first));
            }
        }
    }
    m.outside = outside;
    out[me] = m;
    me
}

/// A top-level statement's import: its instance into the module's aliases (the import's
/// parameters evaluated under `env`, a module passed by its alias).
fn declared(
    files: &BTreeMap<String, String>,
    stmt: &str,
    env: &BTreeMap<String, i64>,
    m: &mut Module,
    out: &mut Vec<Module>,
    depth: usize,
) {
    let Some(rest) = stmt.strip_prefix("const ") else {
        return;
    };
    let Some((name, v)) = rest.split_once('=') else {
        return;
    };
    let name = name.split(':').next().unwrap_or("").trim().to_string();
    let v = v.trim();
    if let Some(at) = v.find("@import_module(\"")
        && depth < 8
        && let Some(a) = call_args(&v[at..])
    {
        let f = a[0].trim_matches('"');
        if files.contains_key(f) {
            let mut p = BTreeMap::new();
            let mut passed = BTreeMap::new();
            for (k, e) in a.get(1).map(|f| fields(f)).unwrap_or_default() {
                if let Some(x) = eval(&e, env) {
                    p.insert(k, x);
                } else if let Some(&inst) = m.aliases.get(e.trim()) {
                    passed.insert(k, inst);
                }
            }
            let inst = modules(files, f, &p, &passed, out, depth + 1);
            m.aliases.insert(name, inst);
        }
    }
}

/// A PE's code: its modules (the first its own) and the functions it reaches (module, name),
/// with each one's call sites.
pub(super) struct Code {
    pub(super) modules: Vec<Module>,
    pub(super) reached: BTreeSet<(usize, String)>,
    sites: BTreeMap<(usize, String), i64>,
}

impl Code {
    /// The functions a task reaches (the tasks too): what runs during a launch.
    pub(super) fn task_reached(&self) -> BTreeSet<(usize, String)> {
        let mut work: Vec<(usize, String)> = (self.modules.iter().enumerate())
            .flat_map(|(at, m)| {
                (m.funcs.iter())
                    .filter(|(_, f)| f.task)
                    .map(move |(n, _)| (at, n.clone()))
            })
            .collect();
        let mut seen = BTreeSet::new();
        while let Some(k) = work.pop() {
            if !seen.insert(k.clone()) {
                continue;
            }
            for (q, n) in &self.modules[k.0].funcs[&k.1].refs {
                if let Some(t) = self.resolve(k.0, q.as_deref(), n) {
                    work.push(t);
                }
            }
        }
        seen
    }

    /// The functions `f` of module `m` calls.
    pub(super) fn callees(&self, m: usize, f: &str) -> Vec<(usize, String)> {
        (self.modules[m].funcs[f].refs.iter())
            .filter_map(|(q, n)| self.resolve(m, q.as_deref(), n))
            .collect()
    }

    /// The function `(q, n)` names from module `at`.
    pub(super) fn resolve(&self, at: usize, q: Option<&str>, n: &str) -> Option<(usize, String)> {
        let target = match q {
            Some(q) => *self.modules[at].aliases.get(q)?,
            None => at,
        };
        self.modules[target]
            .funcs
            .contains_key(n)
            .then(|| (target, n.to_string()))
    }
}

/// The code of `file` at `params`, its imports read from `files`.
pub(super) fn code(
    files: &BTreeMap<String, String>,
    file: &str,
    params: &BTreeMap<String, i64>,
) -> Code {
    let mut ms = Vec::new();
    modules(files, file, params, &BTreeMap::new(), &mut ms, 0);
    // a reference from module `at` to a function: (module, name)
    let resolve =
        |ms: &[Module], at: usize, (q, n): &(Option<String>, String)| -> Option<(usize, String)> {
            let target = match q {
                Some(q) => *ms[at].aliases.get(q)?,
                None => at,
            };
            ms[target]
                .funcs
                .contains_key(n)
                .then(|| (target, n.clone()))
        };
    let mut seen: BTreeSet<(usize, String)> = BTreeSet::new();
    let mut sites: BTreeMap<(usize, String), i64> = BTreeMap::new();
    let mut work: Vec<(usize, String)> = Vec::new();
    for (at, m) in ms.iter().enumerate() {
        for (n, f) in &m.funcs {
            if f.task {
                work.push((at, n.clone()));
            }
        }
        for r in &m.roots {
            if let Some(k) = resolve(&ms, at, r) {
                work.push(k);
            }
        }
    }
    while let Some(k) = work.pop() {
        if !seen.insert(k.clone()) {
            continue;
        }
        for r in &ms[k.0].funcs[&k.1].refs {
            if let Some(t) = resolve(&ms, k.0, r) {
                *sites.entry(t.clone()).or_default() += 1;
                work.push(t);
            }
        }
    }
    Code {
        modules: ms,
        reached: seen,
        sites,
    }
}

/// What the code of `file` at `params` reaches, its imports read from `files`.
#[must_use]
pub fn reached(
    files: &BTreeMap<String, String>,
    file: &str,
    params: &BTreeMap<String, i64>,
) -> Reached {
    let Code {
        modules: ms,
        reached: seen,
        sites,
    } = code(files, file, params);
    let mut out = Reached::default();
    for k in &seen {
        let f = &ms[k.0].funcs[&k.1];
        let times = if f.inline {
            sites.get(k).copied().unwrap_or(1).max(1)
        } else {
            1
        };
        out.measure.add(&f.measure, times);
        out.functions += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_what_is_reached_and_the_branch_taken_count() {
        let pe = "param A: u16;\nconst m = @import_module(\"m.csl\", .{ .B = A });\n\
                  fn unused() void { x(); y(); }\n\
                  fn go() void { if (A == 1) { m.f(); m.f(); } else { a(); b(); c(); } }\n\
                  task t() void { go(); }\n";
        let m = "param B: u16;\nfn f() void { p(); q(); }\nfn g() void { r(); }\n";
        let files: BTreeMap<String, String> =
            [("pe.csl".into(), pe.into()), ("m.csl".into(), m.into())].into();
        let one = reached(&files, "pe.csl", &[("A".into(), 1)].into());
        // t (1) + go's taken branch (2) + m.f (2)
        assert_eq!((one.measure.statements, one.functions), (5, 3));
        let two = reached(&files, "pe.csl", &[("A".into(), 2)].into());
        assert_eq!((two.measure.statements, two.functions), (4, 2));
    }

    #[test]
    fn a_short_loop_of_a_known_count_is_unrolled_and_comptime_calls_are_not_code() {
        let pe = "param N: u16;\nconst fab = @import_module(\"f.csl\");\n\
                  fn go() void { var i: u16 = 0; while (i < N) : (i += 1) { @fmovs(a, b); } }\n\
                  comptime { fab.setup(); @bind_local_task(go, t); }\n";
        let f = "fn setup() void { x(); y(); z(); }\n";
        let files: BTreeMap<String, String> =
            [("pe.csl".into(), pe.into()), ("f.csl".into(), f.into())].into();
        let at = |n: i64| reached(&files, "pe.csl", &[("N".into(), n)].into());
        // go alone (the comptime block's call runs at compile time); 16 trips unrolled
        assert_eq!(at(16).functions, 1);
        assert_eq!((at(16).measure.dsd, at(16).measure.loops), (16, 1));
        assert_eq!(at(1000).measure.dsd, 1);
    }
}

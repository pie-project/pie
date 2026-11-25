//! The instantiations the host names, read off its source: each
//! `kernel(unit, name)` under `src/` and each kernel name spelled elsewhere
//! (a helper's, a table's), filled from the literals, consts, `let`s, arms
//! and helpers that spell it. A placeholder the source does not spell takes
//! its template parameter's stand-in: `bf16` a type, `false` a bool, its
//! default, else a [`HOLE`] the compile tries at each of [`STAND_INS`].

use std::collections::{BTreeMap, BTreeSet};

use crate::source::{Header, LIBRARY};

/// A `kernel(..)` call the scan could not read a kernel off.
pub(crate) struct Unread {
    pub at: String,
    pub name: String,
}

pub(crate) struct Launches {
    /// The instantiations named, by unit, in the order the source names
    /// them.
    pub named: BTreeMap<&'static str, Vec<String>>,
    pub unread: Vec<Unread>,
}

/// A `__global__` a unit defines: its name and its template parameters,
/// each its kind (`class`, `bool`, `int`, ...) and default; `None` for a
/// plain function.
pub(crate) struct Global<'a> {
    pub name: &'a str,
    pub params: Option<Vec<(&'a str, Option<&'a str>)>>,
    /// The namespaces the defining file opens, each path segment.
    pub spaces: Vec<&'a str>,
}

/// The `__global__` functions `text` defines.
pub(crate) fn globals(text: &str) -> Vec<Global<'_>> {
    let spaces: Vec<&str> = text
        .lines()
        .filter_map(|line| line.trim_start().strip_prefix("namespace "))
        .flat_map(|path| path.split('{').next().unwrap_or("").split("::"))
        .map(str::trim)
        .filter(|seg| !seg.is_empty())
        .collect();
    let mut found = Vec::new();
    for (at, _) in text.match_indices("__global__") {
        let line = text[..at].rsplit('\n').next().unwrap_or("");
        if line.trim_start().starts_with("//") {
            continue;
        }
        let Some(name) = name_after(&text[at + "__global__".len()..]) else {
            continue;
        };
        found.push(Global {
            name,
            params: template_before(text, at),
            spaces: spaces.clone(),
        });
    }
    found
}

fn name_after(mut rest: &str) -> Option<&str> {
    loop {
        rest = rest.trim_start();
        let word = rest
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .unwrap_or(rest.len());
        if word == 0 {
            return None;
        }
        let (name, after) = rest.split_at(word);
        if !after.trim_start().starts_with('(') {
            rest = after;
            continue;
        }
        if name != "__launch_bounds__" {
            return Some(name);
        }
        let open = after.find('(')?;
        rest = &after[close(after, open)?..];
    }
}

/// The template parameter kinds of the `template <..>` that `at` closes, if
/// only whitespace stands between them.
fn template_before(text: &str, at: usize) -> Option<Vec<(&str, Option<&str>)>> {
    let head = text[..at].trim_end();
    let shut = head.strip_suffix('>')?;
    let mut depth = 1;
    let open = shut
        .char_indices()
        .rev()
        .find(|&(_, c)| {
            depth += i32::from(c == '>') - i32::from(c == '<');
            depth == 0
        })?
        .0;
    if !shut[..open].trim_end().ends_with("template") {
        return None;
    }
    let kinds = split_args(&shut[open + 1..])
        .into_iter()
        .map(|p| {
            let (p, default) = p
                .split_once('=')
                .map_or((p, None), |(p, d)| (p, Some(d.trim())));
            let p = p.trim();
            let kind = p
                .rsplit_once(char::is_whitespace)
                .map_or(p, |(k, _)| k.trim());
            (kind, default)
        })
        .collect();
    Some(kinds)
}

/// Every unit's `__global__`s, its includes' among them.
pub(crate) fn defined(header: &Header) -> Vec<Global<'static>> {
    let mut seen = BTreeSet::new();
    let mut todo = vec![header.name];
    let mut all = Vec::new();
    while let Some(name) = todo.pop() {
        if !seen.insert(name) {
            continue;
        }
        let Some(h) = LIBRARY.iter().find(|h| h.name == name) else {
            continue;
        };
        all.extend(globals(h.text));
        todo.extend(
            h.text
                .lines()
                .filter_map(|line| line.strip_prefix("#include \"")?.split('"').next()),
        );
    }
    all
}

/// The `kernel(..)` calls under `src/`.
pub(crate) fn scan() -> Launches {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    walk(&root, &mut files);
    let texts: Vec<(String, String)> = files
        .iter()
        .map(|p| {
            let text = std::fs::read_to_string(p).expect("a host source reads");
            let at = p.strip_prefix(&root).unwrap_or(p).display().to_string();
            (at, uncommented(&text))
        })
        .collect();
    let mut units: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (_, text) in &texts {
        for (name, value) in consts(text) {
            if let Some(v) = literal(value) {
                units.entry(name.to_owned()).or_default().insert(v);
            }
        }
    }
    let globals: Vec<(&'static str, Vec<Global<'static>>)> =
        LIBRARY.iter().map(|h| (h.name, defined(h))).collect();

    let mut launches = Launches {
        named: BTreeMap::new(),
        unread: Vec::new(),
    };
    let mut found: Vec<(bool, &'static str, String)> = Vec::new();
    for (file, text) in &texts {
        // Each `kernel(unit, name)`, and each kernel name spelled anywhere
        // else (a table of them, a helper's), whose unit is the one that
        // defines it.
        let mut sites: Vec<(usize, Option<&str>, &str)> = Vec::new();
        let mut calls = Vec::new();
        for (at, _) in text.match_indices("kernel(") {
            let before = &text[..at];
            if before.ends_with(|c: char| c.is_ascii_alphanumeric() || c == '_')
                || before.trim_end().ends_with("fn")
            {
                continue;
            }
            let open = at + "kernel".len();
            let Some(end) = close(text, open) else {
                continue;
            };
            if let [unit, name, ..] = split_top(&text[open + 1..end - 1])[..] {
                sites.push((at, Some(unit), name));
                calls.push(at..end);
            }
        }
        for (at, _) in text.match_indices("\"::pie::") {
            if calls.iter().any(|call| call.contains(&at)) {
                continue;
            }
            let before = text[..at].trim_end();
            if let Some(open) = before.strip_suffix("format!(").map(str::len) {
                if let Some(end) = close(text, open + "format!".len()) {
                    sites.push((at, None, &text[open..end]));
                }
            } else if let Some(end) = text[at + 1..].find('"').map(|e| at + e + 2)
                && !text[at..end].contains('{')
            {
                sites.push((at, None, &text[at..end]));
            }
        }
        for (at, unit, name) in sites {
            let line = text[..at].matches('\n').count() + 1;
            let mut unread = |name: &str| {
                if unit.is_some() {
                    launches.unread.push(Unread {
                        at: format!("{file}:{line}"),
                        name: name.split_whitespace().collect::<Vec<_>>().join(" "),
                    });
                }
            };
            let spelled = eval(name, text, at, 0);
            if spelled.is_empty() {
                unread(name);
            }
            for spelled in spelled {
                let base = base_of(&spelled);
                let defines = |(_, gs): &&(&'static str, Vec<Global<'static>>)| {
                    gs.iter().any(|g| g.name == base && spaced(g, &spelled))
                };
                let named = unit.and_then(|u| unit_of(u, text, &units));
                let mut owners: Vec<_> = globals
                    .iter()
                    .filter(|(h, _)| named.as_deref() == Some(*h))
                    .filter(defines)
                    .collect();
                if owners.is_empty() {
                    owners = globals.iter().filter(defines).collect();
                }
                let mut read = false;
                for (owner, defined) in owners {
                    if let Some(filled) = holes(&spelled, defined) {
                        found.push((unit.is_some(), *owner, filled));
                        read = true;
                    }
                }
                if !read {
                    unread(&spelled.replace(HOLE, "{?}"));
                }
            }
        }
    }
    // A name spelled outside a launch stands for its kernel only where no
    // launch names that kernel: a helper's, a table's, whose launch the
    // scan did not follow. Its includers would take it otherwise.
    let launched: BTreeSet<String> = found
        .iter()
        .filter(|(at_launch, ..)| *at_launch)
        .map(|(_, _, name)| base_of(name).to_owned())
        .collect();
    for (at_launch, owner, name) in found {
        if !at_launch && launched.contains(base_of(&name)) {
            continue;
        }
        let named = launches.named.entry(owner).or_default();
        if !named.contains(&name) {
            named.push(name);
        }
    }
    launches
}

fn walk(dir: &std::path::Path, into: &mut Vec<std::path::PathBuf>) {
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .expect("src/ lists")
        .map(|e| e.expect("an entry").path())
        .collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            walk(&path, into);
        } else if path.extension().is_some_and(|e| e == "rs") {
            into.push(path);
        }
    }
}

/// Whether `g`'s file opens the namespace `name` qualifies it in.
fn spaced(g: &Global<'_>, name: &str) -> bool {
    let bare = name.split('<').next().unwrap_or(name).trim();
    let mut path = bare.rsplit("::");
    path.next();
    path.next().is_none_or(|space| g.spaces.contains(&space))
}

/// The kernel's unqualified name: `::pie::x::f<T>` is `f`.
pub(crate) fn base_of(name: &str) -> &str {
    let bare = name.split('<').next().unwrap_or(name).trim();
    bare.rsplit("::").next().unwrap_or(bare)
}

/// `{unit}` of a `kernel(unit, ..)`: a literal, this file's const of that
/// name, or the one const of that name the crate has.
fn unit_of(arg: &str, text: &str, units: &BTreeMap<String, BTreeSet<String>>) -> Option<String> {
    if let Some(v) = literal(arg) {
        return Some(v);
    }
    let ident = arg.rsplit("::").next()?.trim();
    if let Some(v) = consts(text)
        .into_iter()
        .find(|(n, _)| *n == ident)
        .and_then(|(_, v)| literal(v))
    {
        return Some(v);
    }
    let all = units.get(ident)?;
    (all.len() == 1)
        .then(|| all.iter().next().cloned())
        .flatten()
}

/// Every `const NAME: ty = value;` in `text`.
fn consts(text: &str) -> Vec<(&str, &str)> {
    let mut found = Vec::new();
    for (at, _) in text.match_indices("const ") {
        let rest = &text[at + "const ".len()..];
        let Some((name, rest)) = rest.split_once(':') else {
            continue;
        };
        let name = name.trim();
        if !is_ident(name) || name.contains("fn") {
            continue;
        }
        let Some((_, value)) = rest.split_once('=') else {
            continue;
        };
        let value = statement(value);
        found.push((name, value.trim()));
    }
    found
}

fn is_ident(s: &str) -> bool {
    !s.is_empty()
        && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !s.starts_with(|c: char| c.is_ascii_digit())
}

/// `text` up to the `;` that ends its statement.
fn statement(text: &str) -> &str {
    let mut depth = 0i32;
    let mut skip = Skip::default();
    for (i, c) in text.char_indices() {
        if skip.over(text, i, c) {
            continue;
        }
        match c {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            ';' if depth <= 0 => return &text[..i],
            _ => {}
        }
    }
    text
}

/// The most names one expression is read as: every arm of every `if` and
/// `match` that spells it, crossed, up to this many.
const ALTERNATIVES: usize = 16;

/// The values `expr` may take at `pos` of `text`, as far as its source
/// spells them: each arm of an `if`, a `match` or a dtype dispatch. Empty
/// where it spells none.
fn eval(expr: &str, text: &str, pos: usize, depth: u32) -> Vec<String> {
    if depth > 8 {
        return Vec::new();
    }
    let expr = expr.trim();
    let expr = expr.strip_prefix('&').unwrap_or(expr).trim();
    let expr = expr.strip_suffix('?').unwrap_or(expr).trim();
    if let Some(v) = literal(expr) {
        return vec![v];
    }
    let digits = expr.trim_end_matches(|c: char| c.is_ascii_alphabetic());
    if !digits.is_empty() && digits.replace('_', "").bytes().all(|b| b.is_ascii_digit()) {
        return vec![digits.replace('_', "")];
    }
    if expr == "true" || expr == "false" {
        return vec![expr.to_owned()];
    }
    if let Some(inner) = expr.strip_prefix("format!(") {
        return inner
            .strip_suffix(')')
            .map_or_else(Vec::new, |inner| fill(inner, text, pos, depth));
    }
    for adaptor in [
        ".ok_or_else(",
        ".ok_or(",
        ".expect(",
        ".to_string(",
        ".to_owned(",
        ".into(",
    ] {
        if let Some(at) = expr.rfind(adaptor)
            && close(expr, at + adaptor.len() - 1) == Some(expr.len())
        {
            return eval(&expr[..at], text, pos, depth + 1);
        }
    }
    for wrap in ["Ok(", "Some("] {
        if let Some(inner) = expr.strip_prefix(wrap)
            && close(expr, wrap.len() - 1) == Some(expr.len())
        {
            return eval(&inner[..inner.len() - 1], text, pos, depth + 1);
        }
    }
    for from in ["u32::from(", "i32::from(", "u8::from(", "usize::from("] {
        if let Some(inner) = expr.strip_prefix(from).and_then(|e| e.strip_suffix(')')) {
            return eval(inner, text, pos, depth + 1)
                .into_iter()
                .map(|v| match v.as_str() {
                    "true" => "1".to_owned(),
                    "false" => "0".to_owned(),
                    _ => v,
                })
                .collect();
        }
    }
    if expr.starts_with("if ") || expr.starts_with("match ") || is_macro(expr) {
        return arms(expr, text, pos, depth, None);
    }
    for op in [" / ", " * ", " + ", " - "] {
        if let Some((a, b)) = expr.rsplit_once(op) {
            let (a, b) = (eval(a, text, pos, depth + 1), eval(b, text, pos, depth + 1));
            let mut out = Vec::new();
            for a in a.iter().filter_map(|a| a.parse::<i64>().ok()) {
                for b in b.iter().filter_map(|b| b.parse::<i64>().ok()) {
                    let v = match op {
                        " / " => a.checked_div(b),
                        " * " => a.checked_mul(b),
                        " + " => a.checked_add(b),
                        _ => a.checked_sub(b),
                    };
                    out.extend(v.map(|v| v.to_string()));
                }
            }
            return capped(out);
        }
    }
    if expr.ends_with(')')
        && let Some(open) = expr.find('(')
        && close(expr, open) == Some(expr.len())
    {
        let callee = &expr[..open];
        let name = callee.rsplit('.').next().unwrap_or(callee);
        if is_ident(name) {
            return called(name, text, depth);
        }
    }
    if !is_ident(expr) {
        return Vec::new();
    }
    if let Some((rhs, part)) = binding(expr, &text[..pos]) {
        let rhs = rhs.trim();
        return match part {
            None => eval(rhs, text, pos, depth + 1),
            Some(i) if rhs.starts_with('(') => {
                tuple_part(rhs, i).map_or_else(Vec::new, |v| eval(v, text, pos, depth + 1))
            }
            Some(i) => arms(rhs, text, pos, depth, Some(i)),
        };
    }
    consts(text)
        .into_iter()
        .find(|(n, _)| *n == expr)
        .map_or_else(Vec::new, |(_, v)| eval(v, text, pos, depth + 1))
}

/// `name!(..)`, a macro call (a dtype dispatch).
fn is_macro(expr: &str) -> bool {
    expr.split_once("!(")
        .is_some_and(|(name, _)| is_ident(name) && name != "format")
}

/// The values each arm of the `if`, `match` or dispatching macro `expr`
/// takes: its top-level blocks, a `match`'s split at its arms; with `part`,
/// that element of each arm's tuple.
fn arms(expr: &str, text: &str, pos: usize, depth: u32, part: Option<usize>) -> Vec<String> {
    let value = |v: &str| -> Vec<String> {
        let v = tail(v);
        match part {
            None => eval(v, text, pos, depth + 1),
            Some(i) => tuple_part(v, i).map_or_else(Vec::new, |v| eval(v, text, pos, depth + 1)),
        }
    };
    let mut out = Vec::new();
    let mut at = 0;
    while let Some(open) = expr[at..].find('{').map(|o| at + o) {
        let Some(end) = close(expr, open) else {
            break;
        };
        let block = &expr[open + 1..end - 1];
        let is_match = expr.starts_with("match ") || is_macro(expr);
        if is_match {
            for arm in split_top(block) {
                if let Some((_, value_of)) = arm.split_once("=>") {
                    out.extend(value(value_of));
                }
            }
        } else {
            out.extend(value(block));
        }
        at = end;
    }
    capped(out)
}

/// Element `i` of the tuple expression `expr`.
fn tuple_part(expr: &str, i: usize) -> Option<&str> {
    let expr = expr.trim();
    if !expr.starts_with('(') || close(expr, 0) != Some(expr.len()) {
        return None;
    }
    split_top(&expr[1..expr.len() - 1]).get(i).copied()
}

/// A block's value: its text after the last statement, braces shed.
fn tail(block: &str) -> &str {
    let block = block.trim();
    let block = block
        .strip_prefix('{')
        .and_then(|b| b.strip_suffix('}'))
        .unwrap_or(block);
    let mut depth = 0i32;
    let mut last = 0;
    let mut skip = Skip::default();
    for (i, c) in block.char_indices() {
        if skip.over(block, i, c) {
            continue;
        }
        match c {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            ';' if depth == 0 => last = i + 1,
            _ => {}
        }
    }
    block[last..].trim()
}

fn capped(mut values: Vec<String>) -> Vec<String> {
    let mut seen = BTreeSet::new();
    values.retain(|v| seen.insert(v.clone()));
    values.truncate(ALTERNATIVES);
    values
}

/// What the helper `fn name` in `text` returns: its body's value, else the
/// first kernel name its body spells; its parameters are holes.
fn called(name: &str, text: &str, depth: u32) -> Vec<String> {
    let Some(at) = text.find(&format!("fn {name}(")) else {
        return Vec::new();
    };
    let Some(body) = text[at..].find('{').map(|b| at + b) else {
        return Vec::new();
    };
    let Some(end) = close(text, body) else {
        return Vec::new();
    };
    let inner = &text[body..end];
    let value = eval(tail(inner), text, body, depth + 1);
    if !value.is_empty() {
        return value;
    }
    for (f, _) in inner.match_indices("format!(") {
        let open = f + "format!".len();
        if inner[open + 1..].trim_start().starts_with("\"::pie::")
            && let Some(shut) = close(inner, open)
        {
            return eval(&inner[f..shut], text, body + f, depth + 1);
        }
    }
    inner
        .find("\"::pie::")
        .and_then(|lit| {
            let shut = lit + 1 + inner[lit + 1..].find('"')?;
            literal(&inner[lit..=shut])
        })
        .filter(|name| !name.contains('{'))
        .into_iter()
        .collect()
}

/// The value of the last `let name = ..;` in `text`, and `name`'s place
/// where the `let` destructures a tuple.
fn binding<'a>(name: &str, text: &'a str) -> Option<(&'a str, Option<usize>)> {
    let mut last = None;
    for (at, _) in text.match_indices("let ") {
        let rest = text[at + "let ".len()..].trim_start();
        let (rest, part) = if rest.starts_with('(') {
            let Some(shut) = close(rest, 0) else {
                continue;
            };
            let place = split_top(&rest[1..shut - 1])
                .iter()
                .position(|p| p.strip_prefix("mut ").unwrap_or(p).trim() == name);
            let Some(place) = place else {
                continue;
            };
            (&rest[shut..], Some(place))
        } else {
            let rest = rest.strip_prefix("mut ").unwrap_or(rest).trim_start();
            let Some(rest) = rest.strip_prefix(name) else {
                continue;
            };
            if rest.starts_with(|c: char| c.is_ascii_alphanumeric() || c == '_') {
                continue;
            }
            (rest, None)
        };
        let rest = rest.trim_start();
        let rest = if let Some(typed) = rest.strip_prefix(':') {
            let Some((_, value)) = typed.split_once('=') else {
                continue;
            };
            value
        } else if let Some(value) = rest.strip_prefix('=') {
            value
        } else {
            continue;
        };
        last = Some((statement(rest), part));
    }
    last
}

/// A hole: a placeholder the source does not spell.
pub(crate) const HOLE: char = '\u{1}';

/// What an integer hole is tried at, in turn, every hole of a name at once.
const STAND_INS: &[&str] = &["1", "2", "4", "8", "16", "32", "64", "128", "256", "0"];

/// The names `pattern` stands for: itself without a hole, else one a
/// stand-in.
pub(crate) fn stand_ins(pattern: &str) -> Vec<String> {
    if pattern.contains(HOLE) {
        STAND_INS.iter().map(|s| pattern.replace(HOLE, s)).collect()
    } else {
        vec![pattern.to_owned()]
    }
}

/// `format!(inner)` filled each way its arguments read, with a [`HOLE`]
/// where one is unspelled.
fn fill(inner: &str, text: &str, pos: usize, depth: u32) -> Vec<String> {
    let args = split_top(inner);
    let Some(template) = args.first().and_then(|a| literal(a)) else {
        return Vec::new();
    };
    let mut named = BTreeMap::new();
    let mut positional = Vec::new();
    for arg in &args[1..] {
        match arg.split_once('=') {
            Some((k, v)) if is_ident(k.trim()) && !v.starts_with('=') => {
                named.insert(k.trim(), v);
            }
            _ => positional.push(*arg),
        }
    }
    let mut outs = vec![String::new()];
    let mut rest = template.as_str();
    let mut next = 0;
    while let Some(open) = rest.find('{') {
        let Some(shut) = rest[open..].find('}').map(|s| open + s) else {
            return Vec::new();
        };
        let key = rest[open + 1..shut].split(':').next().unwrap_or("");
        let values = if key.is_empty() {
            let Some(arg) = positional.get(next) else {
                return Vec::new();
            };
            next += 1;
            eval(arg, text, pos, depth + 1)
        } else if let Some(arg) = named.get(key) {
            eval(arg, text, pos, depth + 1)
        } else {
            eval(key, text, pos, depth + 1)
        };
        let values = if values.is_empty() {
            vec![HOLE.to_string()]
        } else {
            values
        };
        let lead = &rest[..open];
        outs = capped(
            outs.iter()
                .flat_map(|o| values.iter().map(move |v| format!("{o}{lead}{v}")))
                .collect(),
        );
        rest = &rest[shut + 1..];
    }
    outs.into_iter().map(|o| o + rest).collect()
}

/// `name` with each hole taken by its template parameter's stand-in, where
/// its kernel's parameter at that place is a type or a bool.
fn holes(name: &str, defined: &[Global<'_>]) -> Option<String> {
    if !name.contains(HOLE) {
        return Some(name.to_owned());
    }
    let base = base_of(name);
    if base.contains(HOLE) {
        return None;
    }
    let open = name.find('<')?;
    let shut = name.rfind('>')?;
    let params = defined
        .iter()
        .find(|g| g.name == base && g.params.is_some())?
        .params
        .as_ref()?;
    let filled: Option<Vec<String>> = split_args(&name[open + 1..shut])
        .into_iter()
        .enumerate()
        .map(|(i, arg)| {
            if !arg.contains(HOLE) {
                return Some(arg.to_owned());
            }
            let (kind, default) = *params.get(i)?;
            let whole = arg == HOLE.to_string();
            match (kind, default) {
                ("class" | "typename", _) if whole => Some("::pie::bf16".to_owned()),
                ("bool", _) if whole => Some("false".to_owned()),
                ("class" | "typename" | "bool", _) => None,
                (_, Some(d)) if whole => Some(d.to_owned()),
                // An integer the launch computes: [`STAND_INS`] try it.
                _ => Some(arg.to_owned()),
            }
        })
        .collect();
    Some(format!(
        "{}<{}>{}",
        &name[..open],
        filled?.join(", "),
        &name[shut + 1..]
    ))
}

/// A Rust string literal's value: escapes read, a `\` line continuation
/// joined.
fn literal(s: &str) -> Option<String> {
    let s = s.trim();
    let body = s.strip_prefix('"')?.strip_suffix('"')?;
    let mut out = String::new();
    let mut chars = body.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next()? {
            '\n' => {
                while chars.peek().is_some_and(|c| c.is_whitespace()) {
                    chars.next();
                }
            }
            'n' => out.push('\n'),
            other => out.push(other),
        }
    }
    (!out.contains('"')).then_some(out)
}

/// Splits Rust `s` at its top-level commas.
fn split_top(s: &str) -> Vec<&str> {
    split(s, false)
}

/// Splits a template argument list at its top-level commas.
fn split_args(s: &str) -> Vec<&str> {
    split(s, true)
}

fn split(s: &str, angles: bool) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut start = 0;
    let mut skip = Skip::default();
    for (i, c) in s.char_indices() {
        if skip.over(s, i, c) {
            continue;
        }
        match c {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            '<' if angles => depth += 1,
            '>' if angles => depth -= 1,
            ',' if depth == 0 => {
                parts.push(s[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }
    if !s[start..].trim().is_empty() {
        parts.push(s[start..].trim());
    }
    parts
}

/// One past the bracket that closes the one at `open`.
fn close(s: &str, open: usize) -> Option<usize> {
    let mut depth = 0i32;
    let mut skip = Skip::default();
    for (i, c) in s[open..].char_indices() {
        if skip.over(&s[open..], i, c) {
            continue;
        }
        match c {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(open + i + 1);
                }
            }
            _ => {}
        }
    }
    None
}

/// Steps over string and char literals while a scan walks `s`.
#[derive(Default)]
struct Skip {
    string: bool,
    escaped: bool,
    until: usize,
}

impl Skip {
    fn over(&mut self, s: &str, i: usize, c: char) -> bool {
        if i < self.until {
            return true;
        }
        if self.string {
            if self.escaped {
                self.escaped = false;
            } else if c == '\\' {
                self.escaped = true;
            } else if c == '"' {
                self.string = false;
            }
            return true;
        }
        if c == '"' {
            self.string = true;
            return true;
        }
        if c == '\'' {
            let b = s.as_bytes();
            if b.get(i + 2) == Some(&b'\'') {
                self.until = i + 3;
                return true;
            }
            if b.get(i + 1) == Some(&b'\\') {
                self.until = s[i + 2..].find('\'').map_or(s.len(), |e| i + 2 + e + 1);
                return true;
            }
        }
        false
    }
}

/// `text` with its `//` comments blanked, so a launch a comment names is
/// not read as one.
fn uncommented(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut skip = Skip::default();
    let mut comment = false;
    for (i, c) in text.char_indices() {
        if comment {
            comment = c != '\n';
            if !comment {
                out.push(c);
            }
            continue;
        }
        if !skip.over(text, i, c) && text[i..].starts_with("//") {
            comment = true;
            continue;
        }
        out.push(c);
    }
    out
}

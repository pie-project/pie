//! A static check of an emitted wafer program, every regime alike: what `layout.csl` (and a
//! decode program's `rcfg.txt`) route and each PE's code holds, read off the files cslc takes.
//!
//! - **Routes**: one route a colour at a PE (a colour set twice to different words is a clash),
//!   colours in range, and every send received: a route transmitting toward a neighbour needs
//!   that neighbour's route of the colour to receive from it (a wavelet sent into a PE that
//!   does not take it, or off the program's rectangle, stalls the sender).
//! - **Queues**: queue ids in range; a PE's input queue bound to a colour its route at the PE
//!   carries but never delivers to the ramp is an error (the queue waits on what never comes:
//!   K1's band bug); bound to a colour the PE does not route at all, a warning (a queue a
//!   program binds and never reads is legal).
//! - **Memory**: each PE's data (its files' arrays, their sizes evaluated from the tile's
//!   parameters) and code (an estimate from its statements, [`CODE_BASE`] + [`CODE_PER`] a
//!   statement, fitted to linked ELFs) within [`PE_BYTES`], the margin reported. Against cslc's
//!   ELFs (flux2-mini's GEMM program) the estimate is 1.2 KB (compute PE) and 3.0 KB (tape PE)
//!   over: a margin it reports is at most the real one.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

type R<T> = Result<T, String>;

/// Colours a WSE-3 router routes.
pub const COLOURS: i64 = 24;
/// Input / output queues a PE has.
pub const QUEUES: i64 = 8;
/// A PE's memory.
pub const PE_BYTES: i64 = 48 * 1024;
/// A PE's code past its statements (the runtime, memcpy's handlers, the task table), and a
/// statement's: fitted to cslc's ELFs (GENERIC.md: the GEMM regime's compute PE 31,174 bytes of
/// text for its interpreter and regions, its tape PE 4,556).
pub const CODE_BASE: i64 = 3_000;
pub const CODE_PER: i64 = 24;
/// A PE's data past its files' arrays (the runtime's and memcpy's): the GEMM regime's compute
/// PE's ELF holds 1,430 bytes of data past what its files declare, its tape PE 326.
pub const DATA_BASE: i64 = 1_500;

/// Every PE's routes, `(x, y, colour)` -> route word (rx `0x00` W, `0x20` E, `0x40` S, `0x60` N,
/// `0x80` the ramp; tx `1` W, `2` E, `4` S, `8` N, `0x10` the ramp).
#[derive(Clone, Debug, Default)]
pub struct Routes {
    pub w: i64,
    pub h: i64,
    pub at: BTreeMap<(i64, i64, i64), u32>,
    pub errors: Vec<String>,
}

impl Routes {
    fn set(&mut self, x: i64, y: i64, c: i64, word: u32) {
        if !(0..COLOURS).contains(&c) {
            self.errors.push(format!(
                "({x}, {y}): colour {c} past the router's {COLOURS}"
            ));
            return;
        }
        match self.at.get(&(x, y, c)) {
            Some(&w) if w != word => self.errors.push(format!(
                "({x}, {y}) colour {c}: routed twice, {w:#x} and {word:#x}"
            )),
            _ => {
                self.at.insert((x, y, c), word);
            }
        }
    }

    /// A decode program's routes (`rcfg.txt`: `x y kind n` and `n` pairs `colour word`).
    pub fn of_rcfg(text: &str, w: i64, h: i64) -> R<Routes> {
        let mut r = Routes {
            w,
            h,
            ..Routes::default()
        };
        for line in text.lines() {
            let f: Vec<&str> = line.split_whitespace().collect();
            if f.len() < 4 {
                continue;
            }
            let num = |s: &str| s.parse::<i64>().map_err(|e| format!("rcfg `{line}`: {e}"));
            let (x, y, n) = (num(f[0])?, num(f[1])?, num(f[3])? as usize);
            for k in 0..n {
                let (c, word) = (num(f[4 + 2 * k])?, num(f[5 + 2 * k])?);
                r.set(x, y, c, word as u32);
            }
        }
        Ok(r)
    }
}

const DIRS: [(u32, u32, i64, i64, &str); 4] = [
    (0x01, 0x00, -1, 0, "west"),
    (0x02, 0x20, 1, 0, "east"),
    (0x04, 0x40, 0, 1, "south"),
    (0x08, 0x60, 0, -1, "north"),
];

/// The receive field a neighbour across direction `tx` (its bit) takes from us on.
fn opposite_rx(tx: u32) -> u32 {
    match tx {
        0x01 => 0x20,
        0x02 => 0x00,
        0x04 => 0x60,
        _ => 0x40,
    }
}

/// Every send received: each route's transmissions toward a neighbour, that neighbour's route of
/// the colour receiving from this PE.
#[must_use]
pub fn flow(r: &Routes) -> Vec<String> {
    let mut errs = Vec::new();
    for (&(x, y, c), &word) in &r.at {
        for &(bit, _, dx, dy, name) in &DIRS {
            if word & bit == 0 {
                continue;
            }
            let (nx, ny) = (x + dx, y + dy);
            if nx < 0 || ny < 0 || nx >= r.w || ny >= r.h {
                errs.push(format!(
                    "({x}, {y}) colour {c}: sends {name}, off the {} x {} rectangle",
                    r.w, r.h
                ));
                continue;
            }
            match r.at.get(&(nx, ny, c)) {
                Some(&n) if n & 0xe0 == opposite_rx(bit) => {}
                Some(&n) => errs.push(format!(
                    "({x}, {y}) colour {c}: sends {name} to ({nx}, {ny}), which receives \
                     colour {c} from elsewhere ({n:#x})"
                )),
                None => errs.push(format!(
                    "({x}, {y}) colour {c}: sends {name} to ({nx}, {ny}), which does not route \
                     colour {c}"
                )),
            }
        }
    }
    errs
}

/// Every PE's tile, by `(x, y)`.
pub type Tiles = BTreeMap<(i64, i64), Tile>;

/// A PE's code: its file and its parameters' values (those the layout states as integers).
#[derive(Clone, Debug, Default)]
pub struct Tile {
    pub file: String,
    pub params: BTreeMap<String, i64>,
}

/// `text` with its comments dropped.
fn uncomment(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for line in text.lines() {
        out.push_str(line.split("//").next().unwrap_or(""));
        out.push('\n');
    }
    out
}

/// An integer expression of `env`'s names: literals, names, `+ - * / %`, parentheses,
/// `@as(T, e)`; `None` where it is anything else.
#[must_use]
pub fn eval(e: &str, env: &BTreeMap<String, i64>) -> Option<i64> {
    let toks = tokens(e);
    let mut p = 0;
    let v = expr(&toks, &mut p, env)?;
    (p == toks.len()).then_some(v)
}

fn tokens(e: &str) -> Vec<String> {
    let mut out = Vec::new();
    let cs: Vec<char> = e.chars().collect();
    let mut i = 0;
    while i < cs.len() {
        let c = cs[i];
        if c.is_whitespace() {
            i += 1;
        } else if c.is_ascii_alphanumeric() || c == '_' || c == '@' {
            let s = i;
            while i < cs.len() && (cs[i].is_ascii_alphanumeric() || cs[i] == '_' || cs[i] == '@') {
                i += 1;
            }
            out.push(cs[s..i].iter().collect());
        } else {
            out.push(c.to_string());
            i += 1;
        }
    }
    out
}

fn expr(t: &[String], p: &mut usize, env: &BTreeMap<String, i64>) -> Option<i64> {
    // `if (c) a else b`, the lowest
    if t.get(*p).is_some_and(|s| s == "if") {
        *p += 1;
        (t.get(*p)? == "(").then(|| *p += 1)?;
        let c = expr(t, p, env)?;
        (t.get(*p)? == ")").then(|| *p += 1)?;
        let a = expr(t, p, env)?;
        (t.get(*p)? == "else").then(|| *p += 1)?;
        let b = expr(t, p, env)?;
        return Some(if c != 0 { a } else { b });
    }
    binary(t, p, env, 0)
}

/// The binary operators by precedence, loosest first (`and` / `or` as CSL spells them).
const LEVELS: [&[&str]; 7] = [
    &["or"],
    &["and"],
    &["==", "!=", "<", ">", "<=", ">="],
    &["|", "&", "^"],
    &["<<", ">>"],
    &["+", "-"],
    &["*", "/", "%"],
];

fn binary(t: &[String], p: &mut usize, env: &BTreeMap<String, i64>, level: usize) -> Option<i64> {
    if level == LEVELS.len() {
        return atom(t, p, env);
    }
    let mut v = binary(t, p, env, level + 1)?;
    loop {
        // (two-character operators come as two tokens)
        let one = t.get(*p).cloned().unwrap_or_default();
        let two = format!("{one}{}", t.get(*p + 1).cloned().unwrap_or_default());
        let op = if LEVELS[level].contains(&two.as_str()) && two.len() == 2 {
            *p += 2;
            two
        } else if LEVELS[level].contains(&one.as_str())
            && !((one == "<" || one == ">")
                && t.get(*p + 1)
                    .is_some_and(|n| n == "<" || n == ">" || n == "="))
        {
            *p += 1;
            one
        } else {
            return Some(v);
        };
        let r = binary(t, p, env, level + 1)?;
        v = match op.as_str() {
            "or" => i64::from(v != 0 || r != 0),
            "and" => i64::from(v != 0 && r != 0),
            "==" => i64::from(v == r),
            "!=" => i64::from(v != r),
            "<" => i64::from(v < r),
            ">" => i64::from(v > r),
            "<=" => i64::from(v <= r),
            ">=" => i64::from(v >= r),
            "|" => v | r,
            "&" => v & r,
            "^" => v ^ r,
            "<<" => v.checked_shl(u32::try_from(r).ok()?)?,
            ">>" => v.checked_shr(u32::try_from(r).ok()?)?,
            "+" => v + r,
            "-" => v - r,
            "*" => v * r,
            "/" => v.checked_div(r)?,
            _ => v.checked_rem(r)?,
        };
    }
}

fn atom(t: &[String], p: &mut usize, env: &BTreeMap<String, i64>) -> Option<i64> {
    let s = t.get(*p)?.clone();
    *p += 1;
    if s == "(" {
        let v = expr(t, p, env)?;
        (t.get(*p)? == ")").then(|| *p += 1)?;
        return Some(v);
    }
    if s == "@as" {
        // @as(T, e)
        (t.get(*p)? == "(").then(|| *p += 1)?;
        *p += 1;
        (t.get(*p)? == ",").then(|| *p += 1)?;
        let v = expr(t, p, env)?;
        (t.get(*p)? == ")").then(|| *p += 1)?;
        return Some(v);
    }
    if s == "-" {
        return atom(t, p, env).map(|v| -v);
    }
    match s.as_str() {
        "true" => Some(1),
        "false" => Some(0),
        _ => s.parse::<i64>().ok().or_else(|| env.get(&s).copied()),
    }
}

/// `s` cut at its top-level commas (outside parentheses, braces and brackets).
fn split_top(s: &str) -> Vec<String> {
    let (mut out, mut cur, mut depth) = (Vec::new(), String::new(), 0i32);
    for c in s.chars() {
        match c {
            '(' | '{' | '[' => depth += 1,
            ')' | '}' | ']' => depth -= 1,
            ',' if depth == 0 => {
                out.push(cur.trim().to_string());
                cur.clear();
                continue;
            }
            _ => {}
        }
        cur.push(c);
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

/// The arguments of the call at the start of `s` (`name(...)`).
fn call_args(s: &str) -> Option<Vec<String>> {
    let open = s.find('(')?;
    let mut depth = 0;
    for (i, c) in s[open..].char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(split_top(&s[open + 1..open + i]));
                }
            }
            _ => {}
        }
    }
    None
}

/// A struct literal's fields (`.{ .a = e, .b = f }`).
fn fields(s: &str) -> Vec<(String, String)> {
    let s = s.trim();
    let inner = s
        .strip_prefix(".{")
        .and_then(|r| r.strip_suffix('}'))
        .unwrap_or("");
    split_top(inner)
        .into_iter()
        .filter_map(|f| {
            let (k, v) = f.split_once('=')?;
            Some((
                k.trim().trim_start_matches('.').to_string(),
                v.trim().to_string(),
            ))
        })
        .collect()
}

fn dirs_word(list: &str, rx: bool) -> u32 {
    let mut w = 0;
    for d in list.split(',').map(str::trim) {
        w |= match (d, rx) {
            ("WEST", true) => 0x00,
            ("EAST", true) => 0x20,
            ("SOUTH", true) => 0x40,
            ("NORTH", true) => 0x60,
            ("RAMP", true) => 0x80,
            ("WEST", false) => 0x01,
            ("EAST", false) => 0x02,
            ("SOUTH", false) => 0x04,
            ("NORTH", false) => 0x08,
            ("RAMP", false) => 0x10,
            _ => 0,
        };
    }
    w
}

/// `.rx = .{ ... }` / `.tx = .{ ... }`'s list.
fn route_list<'a>(cfg: &'a str, key: &str) -> &'a str {
    let Some(at) = cfg.find(key) else { return "" };
    let rest = &cfg[at + key.len()..];
    let Some(open) = rest.find('{') else {
        return "";
    };
    let Some(close) = rest[open..].find('}') else {
        return "";
    };
    &rest[open + 1..open + close]
}

/// A layout's statements run: the rectangle, each PE's tile and every route it sets.
pub fn layout(text: &str) -> R<(Routes, Tiles)> {
    let src = uncomment(text);
    let cs: Vec<char> = src.chars().collect();
    let mut st = Layout::default();
    let mut env = BTreeMap::new();
    st.block(&cs, &mut 0, &mut env, false)?;
    Ok((st.routes, st.tiles))
}

#[derive(Default)]
struct Layout {
    routes: Routes,
    tiles: BTreeMap<(i64, i64), Tile>,
}

impl Layout {
    /// Statements up to the block's closing brace (or the end at the top).
    fn block(
        &mut self,
        cs: &[char],
        p: &mut usize,
        env: &mut BTreeMap<String, i64>,
        inner: bool,
    ) -> R<()> {
        loop {
            while *p < cs.len() && cs[*p].is_whitespace() {
                *p += 1;
            }
            if *p >= cs.len() {
                return Ok(());
            }
            if cs[*p] == '}' {
                *p += 1;
                if inner {
                    return Ok(());
                }
                continue;
            }
            // a statement: up to `;` or a block's `{` at depth 0
            let start = *p;
            let mut depth = 0i32;
            while *p < cs.len() {
                match cs[*p] {
                    '(' | '[' => depth += 1,
                    ')' | ']' => depth -= 1,
                    '{' if depth == 0 => break,
                    '{' | '}' if depth > 0 => {}
                    ';' if depth == 0 => break,
                    _ => {}
                }
                *p += 1;
            }
            let head: String = cs[start..(*p).min(cs.len())].iter().collect();
            let head = head.trim().to_string();
            if *p < cs.len() && cs[*p] == '{' {
                *p += 1;
                if let Some(rest) = head.strip_prefix("for") {
                    let args = call_args(rest.trim_start().trim_start_matches('(').trim_start())
                        .unwrap_or_default();
                    let var = head
                        .rsplit('|')
                        .nth(1)
                        .map(str::trim)
                        .unwrap_or("")
                        .to_string();
                    let (a, b, s) = match args.as_slice() {
                        [_, a, b, s] => (eval(a, env), eval(b, env), eval(s, env)),
                        _ => (None, None, None),
                    };
                    let (Some(a), Some(b), Some(s)) = (a, b, s) else {
                        return Err(format!("layout: a loop it cannot run: `{head}`"));
                    };
                    let body = *p;
                    let mut i = a;
                    let mut end = body;
                    if a >= b {
                        // (an empty loop: its body skipped)
                        let mut e = body;
                        self.skip(cs, &mut e);
                        end = e;
                    }
                    while i < b {
                        let mut q = body;
                        let mut inner_env = env.clone();
                        inner_env.insert(var.clone(), i);
                        self.block(cs, &mut q, &mut inner_env, true)?;
                        end = q;
                        i += s.max(1);
                    }
                    *p = end;
                } else {
                    let mut inner_env = env.clone();
                    self.block(cs, p, &mut inner_env, true)?;
                }
                continue;
            }
            *p += 1;
            self.statement(&head, env);
        }
    }

    fn skip(&self, cs: &[char], p: &mut usize) {
        let mut depth = 1;
        while *p < cs.len() && depth > 0 {
            match cs[*p] {
                '{' => depth += 1,
                '}' => depth -= 1,
                _ => {}
            }
            *p += 1;
        }
    }

    fn statement(&mut self, s: &str, env: &mut BTreeMap<String, i64>) {
        if let Some(rest) = s
            .strip_prefix("const ")
            .or_else(|| s.strip_prefix("param "))
        {
            if let Some((name, v)) = rest.split_once('=') {
                let name = name.split(':').next().unwrap_or("").trim().to_string();
                if let Some(v) = eval(v, env) {
                    env.insert(name, v);
                }
            }
        } else if s.starts_with("@set_rectangle") {
            if let Some(a) = call_args(s)
                && a.len() == 2
            {
                self.routes.w = eval(&a[0], env).unwrap_or(0);
                self.routes.h = eval(&a[1], env).unwrap_or(0);
            }
        } else if s.starts_with("@set_tile_code") {
            if let Some(a) = call_args(s)
                && a.len() >= 3
                && let (Some(x), Some(y)) = (eval(&a[0], env), eval(&a[1], env))
            {
                let params = a
                    .get(3)
                    .map(|f| fields(f))
                    .unwrap_or_default()
                    .into_iter()
                    .filter_map(|(k, v)| eval(&v, env).map(|v| (k, v)))
                    .collect();
                self.tiles.insert(
                    (x, y),
                    Tile {
                        file: a[2].trim_matches('"').to_string(),
                        params,
                    },
                );
            }
        } else if s.starts_with("@set_color_config")
            && let Some(a) = call_args(s)
            && a.len() == 4
            && let (Some(x), Some(y)) = (eval(&a[0], env), eval(&a[1], env))
        {
            let c = call_args(&a[2])
                .and_then(|g| g.first().and_then(|e| eval(e, env)))
                .unwrap_or(-1);
            let word = dirs_word(route_list(&a[3], ".rx"), true)
                | dirs_word(route_list(&a[3], ".tx"), false);
            self.routes.set(x, y, c, word);
        }
    }
}

/// What a PE's files hold: data bytes, statements, queue bindings (`input`, id, colour).
#[derive(Clone, Debug, Default)]
struct Held {
    data: i64,
    statements: i64,
    unknown: Vec<String>,
    queues: Vec<(bool, i64, Option<i64>)>,
}

fn type_bytes(t: &str) -> i64 {
    let t = t.trim();
    if t.contains("f32") || t.contains("u32") || t.contains("i32") {
        4
    } else if t.contains("u8") || t.contains("i8") {
        1
    } else {
        2
    }
}

/// A statement's text from its declaration on (`const`, `param`, `var` opening a line; what a
/// `;` split leaves before it is the end of a block).
fn decl(stmt: &str) -> &str {
    let mut at = 0;
    for line in stmt.split_inclusive('\n') {
        let t = line.trim_start();
        if t.starts_with("const ") || t.starts_with("param ") || t.starts_with("var ") {
            return stmt[at..].trim();
        }
        at += line.len();
    }
    stmt.trim()
}

/// `file`'s holdings at `params` (and its imports', each at the parameters it passes), the
/// files read from `files`.
fn held(
    files: &BTreeMap<String, String>,
    file: &str,
    params: &BTreeMap<String, i64>,
    depth: usize,
) -> Held {
    let mut h = Held {
        data: if depth == 0 { DATA_BASE } else { 0 },
        ..Held::default()
    };
    let Some(text) = files.get(file) else {
        return h;
    };
    let src = uncomment(text);
    let mut env = params.clone();
    // the file's integer names, in order (a parameter's default where the tile states none)
    for stmt in src.split(';') {
        let s = decl(stmt);
        let rest = s
            .strip_prefix("const ")
            .or_else(|| s.strip_prefix("param "))
            .or_else(|| s.strip_prefix("var "));
        if let Some(rest) = rest
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
    h.statements = src.matches(';').count() as i64;
    let mut qnames: BTreeMap<String, (bool, i64)> = BTreeMap::new();
    for stmt in src.split(';') {
        let s = decl(stmt);
        // arrays: `[N]T` in a var's type or its `@zeros` / `@constants`, a const's literal
        let is_data = s.starts_with("var ") || (s.starts_with("const ") && s.contains("]f32 {"));
        if is_data
            && let Some(open) = s.find('[')
            && !s[open..].starts_with("[*]")
        {
            let close = s[open..].find(']').map(|c| open + c);
            if let Some(close) = close {
                let n = &s[open + 1..close];
                let ty: String = s[close + 1..]
                    .chars()
                    .take_while(|c| c.is_ascii_alphanumeric() || *c == '@' || *c == '(')
                    .collect();
                match eval(n, &env) {
                    Some(n) => h.data += n * type_bytes(&ty),
                    None => h
                        .unknown
                        .push(format!("{file}: `{}`", s.lines().next().unwrap_or(s))),
                }
            }
        }
        // queues
        for (key, input) in [("@get_input_queue(", true), ("@get_output_queue(", false)] {
            if let Some(at) = s.find(key)
                && let Some(name) = s
                    .strip_prefix("const ")
                    .and_then(|r| r.split(['=', ':']).next())
                && let Some(a) = call_args(&s[at..])
                && let Some(id) = a.first().and_then(|e| eval(e, &env))
            {
                qnames.insert(name.trim().to_string(), (input, id));
            }
        }
        if let Some(at) = s.find("@initialize_queue(")
            && let Some(a) = call_args(&s[at..])
            && a.len() == 2
            && let Some(&(input, id)) = qnames.get(a[0].trim())
        {
            let colour = fields(&a[1])
                .into_iter()
                .find(|(k, _)| k == "color")
                .and_then(|(_, v)| {
                    call_args(&v).and_then(|g| g.first().and_then(|e| eval(e, &env)))
                });
            h.queues.push((input, id, colour));
        }
        // imports: their holdings at the parameters passed
        if depth < 8
            && let Some(at) = s.find("@import_module(\"")
            && let Some(a) = call_args(&s[at..])
        {
            let name = a[0].trim_matches('"');
            if files.contains_key(name) {
                let p: BTreeMap<String, i64> = a
                    .get(1)
                    .map(|f| fields(f))
                    .unwrap_or_default()
                    .into_iter()
                    .filter_map(|(k, v)| eval(&v, &env).map(|v| (k, v)))
                    .collect();
                let sub = held(files, name, &p, depth + 1);
                h.data += sub.data;
                h.statements += sub.statements;
                h.unknown.extend(sub.unknown);
            }
        }
    }
    h
}

/// A check's findings.
#[derive(Clone, Debug, Default)]
pub struct Report {
    pub pes: usize,
    pub routes: usize,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    /// The PE most full: its file, data bytes, code bytes (estimated), and the margin left.
    pub fullest: Option<(String, i64, i64)>,
    /// Array sizes the check could not evaluate (counted as none).
    pub unknown: Vec<String>,
    /// Each PE code's (file, data bytes, code bytes estimated), the fullest first.
    pub codes: Vec<(String, i64, i64)>,
    /// A regime's own memory model's least room a PE leaves (its sizes are its comptime
    /// functions', past this check's arithmetic: a decode program's, `cost::memory::room`),
    /// in place of the estimate's margin.
    pub model_room: Option<i64>,
}

impl Report {
    /// The margin the fullest PE leaves (bytes).
    #[must_use]
    pub fn margin(&self) -> i64 {
        if let Some(r) = self.model_room {
            return r;
        }
        self.fullest
            .as_ref()
            .map_or(PE_BYTES, |(_, d, c)| PE_BYTES - d - c)
    }

    /// One line: its PEs, routes, margin and findings.
    #[must_use]
    pub fn line(&self) -> String {
        let mut s = format!(
            "check: {} PEs, {} routes, {} errors, {} warnings",
            self.pes,
            self.routes,
            self.errors.len(),
            self.warnings.len()
        );
        if let Some(r) = self.model_room {
            let _ = write!(s, "; the memory model's least room a PE leaves {r} B");
        } else if let Some((f, d, c)) = &self.fullest {
            let _ = write!(
                s,
                "; fullest PE `{f}` {d} B data + ~{c} B code of {PE_BYTES}, margin {} B",
                self.margin()
            );
        }
        s
    }

    /// `Err` with its errors (the first few) where it has any.
    pub fn ok(&self) -> R<()> {
        if self.errors.is_empty() {
            return Ok(());
        }
        Err(format!(
            "{}: {}",
            self.line(),
            self.errors
                .iter()
                .take(8)
                .cloned()
                .collect::<Vec<_>>()
                .join("; ")
        ))
    }
}

/// The checks over a program's files (`layout.csl`, its PE files and the ops they import),
/// its routes `routes` where the layout does not set them (a decode program's `rcfg.txt`).
#[must_use]
pub fn program(files: &BTreeMap<String, String>, routes: Option<Routes>) -> Report {
    let mut rep = Report::default();
    let Some(text) = files.get("layout.csl") else {
        rep.errors.push("no layout.csl".into());
        return rep;
    };
    let (mut r, tiles) = match layout(text) {
        Ok(x) => x,
        Err(e) => {
            rep.errors.push(e);
            return rep;
        }
    };
    // (a decode program's queues are bound at its start from its rcfg words, which the
    // fabric's own check holds to its routes: the files' bindings are placeholders)
    let runtime_queues = routes.is_some();
    if let Some(mut given) = routes {
        given.w = given.w.max(r.w);
        given.h = given.h.max(r.h);
        r.at.extend(given.at);
        r.errors.extend(given.errors);
    }
    rep.pes = tiles.len();
    rep.routes = r.at.len();
    rep.errors.extend(r.errors.iter().cloned());
    rep.errors.extend(flow(&r));
    // each distinct (file, parameters) once
    let mut seen: BTreeMap<(String, Vec<(String, i64)>), Held> = BTreeMap::new();
    for (&(x, y), t) in &tiles {
        let key = (
            t.file.clone(),
            t.params.iter().map(|(k, v)| (k.clone(), *v)).collect(),
        );
        let h = seen
            .entry(key)
            .or_insert_with(|| held(files, &t.file, &t.params, 0))
            .clone();
        for &(input, id, colour) in h.queues.iter().filter(|_| !runtime_queues) {
            if !(0..QUEUES).contains(&id) {
                rep.errors.push(format!(
                    "({x}, {y}) `{}`: {} queue {id} past {QUEUES}",
                    t.file,
                    if input { "input" } else { "output" }
                ));
            }
            let Some(c) = colour else { continue };
            if !input {
                continue;
            }
            match r.at.get(&(x, y, c)) {
                Some(&w) if w & 0x10 == 0 => rep.errors.push(format!(
                    "({x}, {y}) `{}`: input queue {id} on colour {c}, which the PE routes but \
                     never to its ramp ({w:#x})",
                    t.file
                )),
                Some(_) => {}
                None => rep.warnings.push(format!(
                    "({x}, {y}) `{}`: input queue {id} on colour {c}, which the PE does not route",
                    t.file
                )),
            }
        }
    }
    for ((f, _), h) in &seen {
        let code = CODE_BASE + CODE_PER * h.statements;
        if rep
            .fullest
            .as_ref()
            .is_none_or(|(_, d, c)| h.data + code > d + c)
        {
            rep.fullest = Some((f.clone(), h.data, code));
        }
        rep.unknown.extend(h.unknown.iter().cloned());
        rep.codes.push((f.clone(), h.data, code));
    }
    rep.codes.sort_by_key(|(_, d, c)| std::cmp::Reverse(d + c));
    rep.codes.dedup();
    rep.unknown.sort();
    rep.unknown.dedup();
    if rep.margin() < 0 {
        let (f, d, c) = rep.fullest.clone().unwrap_or_default();
        rep.errors.push(format!(
            "`{f}`: {d} B data + ~{c} B code past a PE's {PE_BYTES} B"
        ));
    }
    rep
}

/// [`program`] over the files in `dir` (and `ops/` beside them, else the crate's ops), its
/// `rcfg.txt` where it has one.
pub fn dir(dir: &Path) -> R<Report> {
    let mut files: BTreeMap<String, String> = BTreeMap::new();
    for (n, s) in crate::sources::OPS.iter().chain(crate::sources::STREAM_OPS) {
        files.insert(format!("ops/{n}"), (*s).to_string());
    }
    for sub in ["", "ops"] {
        let d = dir.join(sub);
        let Ok(rd) = std::fs::read_dir(&d) else {
            continue;
        };
        for e in rd.flatten() {
            let p = e.path();
            if p.extension().is_some_and(|x| x == "csl") {
                let name = if sub.is_empty() {
                    e.file_name().to_string_lossy().to_string()
                } else {
                    format!("{sub}/{}", e.file_name().to_string_lossy())
                };
                files.insert(
                    name,
                    std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?,
                );
            }
        }
    }
    let rcfg = dir.join("../rcfg.txt");
    let routes = if rcfg.is_file() {
        let t = std::fs::read_to_string(&rcfg).map_err(|e| e.to_string())?;
        Some(Routes::of_rcfg(&t, 0, 0)?)
    } else {
        None
    };
    Ok(program(&files, routes))
}

/// [`program`] over files in memory (an emitter's), the crate's ops beside them.
#[must_use]
pub fn files(own: &[(String, String)]) -> Report {
    let mut files: BTreeMap<String, String> = BTreeMap::new();
    for (n, s) in crate::sources::OPS.iter().chain(crate::sources::STREAM_OPS) {
        files.insert(format!("ops/{n}"), (*s).to_string());
    }
    for (n, s) in own {
        files.insert(n.clone(), s.clone());
    }
    program(&files, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expressions_evaluate_as_csl_reads_them() {
        let env: BTreeMap<String, i64> = [("A".to_string(), 3), ("B".to_string(), 1)].into();
        assert_eq!(eval("if (B == 1) A * 4 else 1", &env), Some(12));
        assert_eq!(eval("(A + 1) << 2", &env), Some(16));
        assert_eq!(eval("@as(u16, A) - 1", &env), Some(2));
        assert_eq!(eval("if (A > 2 and B != 0) 5 else 6", &env), Some(5));
        assert_eq!(eval("C + 1", &env), None);
    }

    #[test]
    fn a_send_into_a_pe_that_does_not_take_it_is_found() {
        let lay = "layout {\n  @set_rectangle(3, 1);\n  for (@range(u16, 0, 2, 1)) |x| { @set_color_config(x, 0, @get_color(4), .{ .routes = .{ .rx = .{ WEST }, .tx = .{ EAST, RAMP } } }); }\n  @set_color_config(2, 0, @get_color(4), .{ .routes = .{ .rx = .{ NORTH }, .tx = .{ RAMP } } });\n}\n";
        let (r, _) = layout(lay).expect("it parses");
        assert_eq!(r.at.len(), 3);
        let e = flow(&r);
        assert_eq!(e.len(), 1, "{e:?}");
        assert!(e[0].contains("(1, 0) colour 4: sends east to (2, 0)"));
    }

    #[test]
    fn a_queue_on_a_colour_never_delivered_is_found_and_memory_is_counted() {
        let pe = "param C: u16;\nconst N: u16 = 100;\nvar buf = @zeros([N * 2]f32);\nconst q = @get_input_queue(3);\ncomptime { @initialize_queue(q, .{ .color = @get_color(C) }); }\n";
        let lay = "layout {\n  @set_rectangle(2, 1);\n  @set_tile_code(0, 0, \"pe.csl\", .{ .C = 5 });\n  @set_tile_code(1, 0, \"pe.csl\", .{ .C = 6 });\n  @set_color_config(0, 0, @get_color(5), .{ .routes = .{ .rx = .{ RAMP }, .tx = .{ EAST } } });\n  @set_color_config(1, 0, @get_color(5), .{ .routes = .{ .rx = .{ WEST }, .tx = .{ RAMP } } });\n}\n";
        let rep = files(&[
            ("layout.csl".into(), lay.into()),
            ("pe.csl".into(), pe.into()),
        ]);
        assert_eq!(rep.errors.len(), 1, "{:?}", rep.errors);
        assert!(rep.errors[0].contains("(0, 0)"));
        assert_eq!(rep.warnings.len(), 1, "{:?}", rep.warnings);
        assert_eq!(rep.fullest.as_ref().map(|f| f.1), Some(800 + DATA_BASE));
    }

    /// A scratch directory of its own.
    fn scratch(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("pie-check-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).expect("a scratch directory");
        d
    }

    /// Every catalog model's decode program of the decode identity's configs (planned as the
    /// search plans them, built as `wafer-compile build` builds them) passes the check (the tiny
    /// t08's, a shape file of the tools', is `wafer-compile build`'s own check).
    #[test]
    fn the_decode_programs_pass_the_check() {
        let models = [
            "qwen35-d0.8b-bf16-kv-bf16",
            "qwen35-d9b-bf16-kv-bf16",
            "llama31-8b-bf16-kv-bf16",
            "gptoss-20b-bf16-mxfp4-kv-bf16",
            "gemma4-e2b-bf16-kv-bf16",
        ];
        let mut plans = Vec::new();
        for m in models {
            let t = poem_compiler::catalog::repository()
                .parse(m)
                .unwrap_or_else(|| panic!("{m} parses"))
                .trace(poem_ir::Platform::Cerebras);
            let p = crate::target::plan(&t, &crate::target::Ask::default())
                .unwrap_or_else(|e| panic!("{m}: {e}"));
            plans.push((m.to_string(), p.plan));
        }
        for (m, plan) in plans {
            let out = scratch(&m.replace(['/', '.'], "_"));
            let room = crate::cost::memory::room(&plan, crate::cost::memory::anchor(&plan.shape));
            let levers = crate::emit::params::Levers::of(&plan.levers, "").expect("levers");
            let prog = crate::program::Program::of(plan).unwrap_or_else(|e| panic!("{m}: {e}"));
            crate::build::emit(&prog, &levers, None, &out, &crate::build::EMITS)
                .unwrap_or_else(|e| panic!("{m}: {e}"));
            let mut rep = dir(&out.join("layout")).expect("the check reads it");
            rep.model_room = Some(room as i64);
            eprintln!("{m}: {}", rep.line());
            assert!(
                rep.errors.is_empty(),
                "{m}: {:?}",
                &rep.errors[..rep.errors.len().min(5)]
            );
            assert!(rep.routes > 0 && rep.margin() > 0, "{m}: {}", rep.line());
            let _ = std::fs::remove_dir_all(&out);
        }
    }

    /// flux2-mini's programs in every regime and klein's 1024² GEMM program pass the check.
    #[test]
    fn the_flux_programs_pass_the_check() {
        use crate::stream::{self, Options};
        let o = |reading: &str| Options {
            reading: reading.into(),
            rows: 16,
            round16: true,
            upto: None,
            taps: Vec::new(),
        };
        let trace = |m: &str| {
            poem_compiler::catalog::repository()
                .parse(m)
                .unwrap_or_else(|| panic!("{m} parses"))
                .trace(poem_ir::Platform::Cerebras)
        };
        let mini = trace("flux2-mini-bf16-kv-bf16");
        // compile on load's text and denoise, the stream regime's denoise, the VAE's grid and
        // tiled programs
        for reading in ["text", "denoise"] {
            let out = scratch(&format!("mini-{reading}"));
            stream::compile_whole(&mini, &o(reading), &[], &out).expect("it compiles");
            let rep = dir(&out).expect("the check reads it");
            eprintln!("flux2-mini {reading}: {}", rep.line());
            assert!(
                rep.errors.is_empty(),
                "{reading}: {:?}",
                &rep.errors[..rep.errors.len().min(5)]
            );
            let _ = std::fs::remove_dir_all(&out);
        }
        let out = scratch("mini-stream");
        stream::compile(&mini, &o("denoise"), &out, None).expect("the stream regime's");
        let rep = dir(&out).expect("the check reads it");
        eprintln!("flux2-mini denoise (stream): {}", rep.line());
        assert!(
            rep.errors.is_empty(),
            "{:?}",
            &rep.errors[..rep.errors.len().min(5)]
        );
        let _ = std::fs::remove_dir_all(&out);
        let out = scratch("mini-vae");
        stream::compile_reading(&mini, &o("vae.decode"), &out).expect("the VAE");
        let rep = dir(&out).expect("the check reads it");
        eprintln!("flux2-mini vae.decode: {}", rep.line());
        assert!(
            rep.errors.is_empty(),
            "{:?}",
            &rep.errors[..rep.errors.len().min(5)]
        );
        let _ = std::fs::remove_dir_all(&out);
        // klein at 1024²: its program in memory (its tapes' files are not the check's)
        let klein = trace("flux2-klein-4b-bf16-kv-bf16");
        let rows = [("context".to_string(), 512), ("latents".to_string(), 4096)];
        let b = crate::floorplan::stream::Bounds::system();
        let mut ko = o("denoise");
        ko.rows = 64;
        let p =
            stream::gemm::program(&klein, &ko, &rows, None, (b.w, b.h), 1).expect("klein places");
        let e = crate::emit::gemm::emit(&p).expect("it emits");
        let rep = files(&e.files);
        eprintln!("klein 1024² denoise: {}", rep.line());
        assert!(
            rep.errors.is_empty(),
            "{:?}",
            &rep.errors[..rep.errors.len().min(5)]
        );
        assert!(rep.margin() > 0, "{}", rep.line());
    }
}

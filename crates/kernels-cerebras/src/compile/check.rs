//! A static check of an emitted wafer program, every regime alike, over its placement
//! ([`Placed`], and a decode program's rcfg) and each PE's files:
//!
//! - **Routes**: one route a colour at a PE, colours in range, and every send received (a
//!   wavelet sent to a PE that does not take it, or off the rectangle, stalls the sender).
//! - **Queues**: an input queue bound to a colour its route carries but never delivers to the
//!   ramp is an error; bound to a colour the PE does not route, a warning.
//! - **Memory**: each PE's data and code ([`code`], [`code_bytes`]) within [`PE_BYTES`]: past it
//!   by more than [`OVER`] an error, within [`UNDER`] of it a warning.

pub mod code;
pub mod hazard;

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use crate::compile::emit::placed::Placed;

type R<T> = Result<T, String>;

/// Colours a WSE-3 router routes.
pub const COLOURS: i64 = 24;
/// Input / output queues a PE has.
pub const QUEUES: i64 = 8;
/// A PE's memory.
pub const PE_BYTES: i64 = 48 * 1024;
/// A PE's code: its runtime's, and a statement's, a DSD operation's, a loop's and an `if`'s of
/// what it reaches, fitted by least squares to cslc's ELFs.
pub const CODE_BASE: f64 = 3_091.0;
pub const CODE_STATEMENT: f64 = 21.4;
pub const CODE_DSD: f64 = 13.3;
pub const CODE_LOOP: f64 = 78.2;
pub const CODE_IF: f64 = 12.2;
/// A PE's data past its files' arrays (the runtime's and memcpy's): the ELFs' median.
pub const DATA_BASE: i64 = 434;
/// The estimate's worst miss against the ELFs it is fitted to, data and code together: a PE
/// holds at most [`UNDER`] more than it estimates and at least [`OVER`] less.
pub const UNDER: i64 = 2_437;
pub const OVER: i64 = 2_361;

/// A PE's memory as the check estimates it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PeBytes {
    pub data: i64,
    pub code: i64,
}

impl PeBytes {
    #[must_use]
    pub fn total(self) -> i64 {
        self.data + self.code
    }

    /// It cannot fit whatever the estimate's miss (past [`PE_BYTES`] by more than [`OVER`]).
    #[must_use]
    pub fn overflows(self) -> bool {
        self.total() - OVER > PE_BYTES
    }
}

/// The code bytes of what a PE's code reaches.
#[must_use]
pub fn code_bytes(r: &code::Reached) -> i64 {
    let m = r.measure;
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    let b = CODE_BASE
        + CODE_STATEMENT * m.statements as f64
        + CODE_DSD * m.dsd as f64
        + CODE_LOOP * m.loops as f64
        + CODE_IF * m.ifs as f64;
    #[allow(clippy::cast_possible_truncation)]
    let b = b.round() as i64;
    b
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CodeKind {
    /// Its file (a `*` in it any run of characters).
    pub file: &'static str,
    /// Text its file holds (`None`: any), telling a variant of the file apart.
    pub holding: Option<&'static str>,
    /// The bytes its images held past the estimate (their mean miss), added to its code.
    pub bytes: i64,
    /// The estimate's worst miss below its images after (bytes; [`miss`]).
    pub miss: i64,
}

/// Kinds of PE the fit misses alike, each refitted on its own linked images; the first a
/// file's matches. A tiled program's top-tile PE (`tpe.csl`) misses by whether its op table
/// streams in (`D_progw`) or is held; a stream unit's weight PE (`w_*.csl`) on its own.
pub const CODE_KINDS: &[CodeKind] = &[
    CodeKind {
        file: "tpe.csl",
        holding: Some("D_progw"),
        bytes: 952,
        miss: 73,
    },
    CodeKind {
        file: "tpe.csl",
        holding: None,
        bytes: 1_329,
        miss: 12,
    },
    CodeKind {
        file: "ttape.csl",
        holding: None,
        bytes: 512,
        miss: 15,
    },
    CodeKind {
        file: "w_*.csl",
        holding: Some("ops/smv.csl"),
        bytes: 476,
        miss: 640,
    },
];

/// `file`'s kind ([`CODE_KINDS`]), its text `text` where known.
fn kind_of(file: &str, text: Option<&str>) -> Option<&'static CodeKind> {
    let base = file.rsplit('/').next().unwrap_or(file);
    CODE_KINDS.iter().find(|k| {
        named(k.file, base)
            && k.holding
                .is_none_or(|h| text.is_some_and(|t| t.contains(h)))
    })
}

/// Whether `file` is `pattern`'s (a `*` in it any run of characters).
fn named(pattern: &str, file: &str) -> bool {
    match pattern.split_once('*') {
        Some((a, b)) => file.len() >= a.len() + b.len() && file.starts_with(a) && file.ends_with(b),
        None => pattern == file,
    }
}

/// The code a kind of PE holds past the fit's ([`CODE_KINDS`]; none for a kind not refitted).
#[must_use]
pub fn kind_code(files: &BTreeMap<String, String>, file: &str) -> i64 {
    kind_of(file, files.get(file).map(String::as_str)).map_or(0, |k| k.bytes)
}

/// The estimate's worst miss below a PE of `file`'s linked image (bytes), its text `text`:
/// its kind's ([`CODE_KINDS`]), else the fit's over every image ([`UNDER`]).
#[must_use]
pub fn miss(file: &str, text: Option<&str>) -> i64 {
    kind_of(file, text).map_or(UNDER, |k| k.miss)
}

/// The DSD descriptors a PE image's low data holds at most: cslc keeps every constant DSD value
/// (8 bytes each) in `.data.lo`, 1,024 bytes; past it the link fails.
pub const LO_DSDS: i64 = 128;
/// [`lo_dsds`]' offset, fitted to linked images: what memcpy's runtime adds less what the
/// compiler folds onto a constant's own descriptor.
const LO_FOLDED: i64 = 17;

/// A PE image's constant DSD values ([`LO_DSDS`] the most it links with): every top-level
/// `@get_dsd` constant of each module instance, and each distinct DSD-builtin expression over
/// such a constant whose other arguments are literals.
#[must_use]
pub fn lo_dsds(
    files: &BTreeMap<String, String>,
    file: &str,
    params: &BTreeMap<String, i64>,
) -> i64 {
    const BUILTINS: [&str; 4] = [
        "@increment_dsd_offset(",
        "@set_dsd_length(",
        "@set_dsd_stride(",
        "@set_dsd_base_addr(",
    ];
    let ident = |c: char| c.is_ascii_alphanumeric() || c == '_';
    let c = code::code(files, file, params);
    let mut tops = 0i64;
    let mut exprs: BTreeSet<(usize, String)> = BTreeSet::new();
    for (mi, m) in c.modules.iter().enumerate() {
        let consts: BTreeSet<String> = m
            .outside
            .match_indices("const ")
            .filter_map(|(at, _)| {
                let rest = &m.outside[at + 6..];
                let name: String = rest.chars().take_while(|&ch| ident(ch)).collect();
                // (a fabric DSD constant takes no low-data descriptor: flux2-mini's 32 x 128
                // GEMM PE with the line sends' two linked at 1,008 B, as without them)
                let rhs = rest[name.len()..].trim_start();
                (rhs.starts_with("= @get_dsd(")
                    && !rhs.starts_with("= @get_dsd(fabout_dsd")
                    && !rhs.starts_with("= @get_dsd(fabin_dsd"))
                .then_some(name)
            })
            .collect();
        tops += consts.len() as i64;
        let constant = |e: &str| {
            let cs: Vec<char> = e.chars().collect();
            let mut any = false;
            let mut i = 0;
            while i < cs.len() {
                if (cs[i].is_ascii_alphabetic() || cs[i] == '_')
                    && (i == 0 || !(ident(cs[i - 1]) || cs[i - 1] == '@'))
                {
                    let w: String = cs[i..].iter().take_while(|&&ch| ident(ch)).collect();
                    i += w.len();
                    if consts.contains(&w) {
                        any = true;
                    } else if !matches!(
                        w.as_str(),
                        "f32" | "f16" | "i16" | "u16" | "i32" | "u32" | "as"
                    ) {
                        return false;
                    }
                    continue;
                }
                i += 1;
            }
            any
        };
        let mut found: Vec<String> = Vec::new();
        for text in std::iter::once(&m.outside).chain(m.funcs.values().map(|f| &f.body)) {
            for b in BUILTINS {
                for (at, _) in text.match_indices(b) {
                    if let Some(args) = call_args(&text[at..]) {
                        let e: String = format!("{b}{})", args.join(","))
                            .chars()
                            .filter(|ch| !ch.is_whitespace())
                            .collect();
                        if constant(&e) {
                            found.push(e);
                        }
                    }
                }
            }
        }
        // (an operand is the outermost expression: one nested in another is not its own)
        for e in &found {
            if !found.iter().any(|o| o != e && o.contains(e.as_str())) {
                exprs.insert((mi, e.clone()));
            }
        }
    }
    tops + exprs.len() as i64 - LO_FOLDED
}

/// The words a tape PE's array may take: the room its image leaves past its own bytes (`own`),
/// the stack's [`crate::compile::build::STACK_ROOM`] and the search's [`crate::compile::search::MARGIN`].
#[must_use]
pub fn tape_words(own: i64) -> u32 {
    #[allow(clippy::cast_possible_truncation)]
    let margin = crate::compile::search::MARGIN as i64;
    let room = PE_BYTES - own - i64::from(crate::compile::build::STACK_ROOM) - margin;
    u32::try_from(room.max(0) / 4).unwrap_or(0)
}

/// The code room a search holds back on every PE beside its data, until the code estimate
/// bounds cslc's: the most `.text` a stream PE of over 10 KB data links to.
pub const CODE_RESERVE: i64 = 7_680;

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

/// What a PE's files hold: data bytes, queue bindings (`input`, id, colour).
#[derive(Clone, Debug, Default)]
struct Held {
    data: i64,
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

/// A file's one-expression functions (`fn f(a: T, ..) R { return e; }`, a comptime size's
/// helper): name -> (parameters, expression).
fn returns(src: &str) -> BTreeMap<String, (Vec<String>, String)> {
    let mut out = BTreeMap::new();
    for (at, _) in src.match_indices("fn ") {
        if at > 0 && src[..at].ends_with(|c: char| c.is_ascii_alphanumeric() || c == '_') {
            continue;
        }
        let rest = &src[at + 3..];
        let Some(open) = rest.find('(') else { continue };
        let name = rest[..open].trim().to_string();
        let Some(args) = call_args(rest) else {
            continue;
        };
        let Some(body) = rest.find('{').map(|b| &rest[b + 1..]) else {
            continue;
        };
        let Some(end) = body.find('}') else { continue };
        let body = body[..end].trim();
        if let Some(e) = body
            .strip_prefix("return ")
            .and_then(|e| e.strip_suffix(';'))
            && !e.contains(';')
        {
            let ps = args
                .iter()
                .map(|a| a.split(':').next().unwrap_or("").trim().to_string())
                .collect();
            out.insert(name, (ps, e.to_string()));
        }
    }
    out
}

/// `e` with its calls of `fns` replaced by their expressions over the call's arguments.
fn inline_calls(e: &str, fns: &BTreeMap<String, (Vec<String>, String)>) -> String {
    if fns.is_empty() {
        return e.to_string();
    }
    let mut out = e.to_string();
    for _ in 0..8 {
        let mut changed = false;
        for (name, (ps, body)) in fns {
            let key = format!("{name}(");
            let Some(at) = out.find(&key) else { continue };
            if at > 0
                && out[..at].ends_with(|c: char| c.is_ascii_alphanumeric() || c == '_' || c == '.')
            {
                continue;
            }
            let Some(args) = call_args(&out[at..]) else {
                continue;
            };
            if args.len() != ps.len() {
                continue;
            }
            // the call's extent: its name to its closing parenthesis
            let mut depth = 0;
            let mut end = at;
            for (i, c) in out[at..].char_indices() {
                match c {
                    '(' => depth += 1,
                    ')' => {
                        depth -= 1;
                        if depth == 0 {
                            end = at + i + 1;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            let sub: Vec<String> = tokens(body)
                .into_iter()
                .map(|t| match ps.iter().position(|p| *p == t) {
                    Some(i) => format!("({})", args[i]),
                    None => t,
                })
                .collect();
            out = format!("{}({}){}", &out[..at], sub.join(" "), &out[end..]);
            changed = true;
        }
        if !changed {
            break;
        }
    }
    out
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
    let fns = returns(&src);
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
                && let Some(v) = eval(&inline_calls(v, &fns), &env)
            {
                env.insert(name, v);
            }
        }
    }
    let mut qnames: BTreeMap<String, (bool, i64)> = BTreeMap::new();
    for stmt in src.split(';') {
        let s = decl(stmt);
        // arrays: `[N]T` in a var's type or its `@zeros` / `@constants`, a const's literal
        let is_data = s.starts_with("var ")
            || (s.starts_with("const ")
                && s.split_once('=')
                    .is_some_and(|(_, v)| v.trim_start().starts_with('[') && v.contains('{')));
        if is_data
            && let Some(open) = s.find('[')
            && !s[open..].starts_with("[*]")
        {
            let close = s[open..].find(']').map(|c| open + c);
            // (`[N]T`: a subscript `a[i]` is no array)
            if let Some(close) = close
                && s[close + 1..].starts_with(|c: char| c.is_ascii_alphabetic() || c == '@')
            {
                let n = &s[open + 1..close];
                let ty: String = s[close + 1..]
                    .chars()
                    .take_while(|c| c.is_ascii_alphanumeric() || *c == '@' || *c == '(')
                    .collect();
                match eval(&inline_calls(n, &fns), &env) {
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
    /// A regime's own memory model's least room a PE leaves (a decode program's,
    /// `cost::memory::room`), in place of the estimate's margin.
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

/// A placement's routes and tiles as the check reads them (no `layout.csl` to parse).
#[must_use]
pub fn layout_ir(p: &Placed) -> (Routes, Tiles) {
    let mut r = Routes {
        w: i64::from(p.w),
        h: i64::from(p.h),
        ..Routes::default()
    };
    for rt in &p.routes {
        let word = dirs_word(rt.rx, true) | dirs_word(&rt.tx.join(", "), false);
        for (x, y) in Placed::pes(rt.at) {
            r.set(i64::from(x), i64::from(y), i64::from(rt.colour), word);
        }
    }
    let mut tiles = Tiles::new();
    for t in &p.tiles {
        let params: BTreeMap<String, i64> = t
            .params
            .iter()
            .filter_map(|(k, v)| eval(v, &BTreeMap::new()).map(|v| (k.clone(), v)))
            .collect();
        for (x, y) in Placed::pes(t.at) {
            tiles.insert(
                (i64::from(x), i64::from(y)),
                Tile {
                    file: t.file.clone(),
                    params: params.clone(),
                },
            );
        }
    }
    (r, tiles)
}

/// The checks over a placement: its PE files in `own` (the crate's ops beside them), `routes`
/// a decode program's (its rcfg's: its placement sets none).
#[must_use]
pub fn placed(p: &Placed, own: &[(String, String)], routes: Option<Routes>) -> Report {
    let mut files: BTreeMap<String, String> = BTreeMap::new();
    for (n, s) in crate::sources::OPS
        .iter()
        .chain(crate::sources::stream_ops())
    {
        files.insert(format!("ops/{n}"), (*s).to_string());
    }
    for (n, s) in own {
        files.insert(n.clone(), s.clone());
    }
    let (r, tiles) = layout_ir(p);
    placed_with(&files, r, &tiles, routes)
}

thread_local! {
    /// Whether a written program's check is kept for [`last`] (this thread's), and the last.
    static KEEP: Cell<bool> = const { Cell::new(false) };
    static LAST: RefCell<Option<Report>> = const { RefCell::new(None) };
}

/// From now on, every program this thread writes is checked, its report kept for [`last`].
pub fn keep() {
    KEEP.with(|k| k.set(true));
}

/// Whether [`keep`] asked.
pub(crate) fn keeping() -> bool {
    KEEP.with(Cell::get)
}

/// A program's writer: its placement and files checked where [`keep`] asks.
pub(crate) fn written(p: &Placed, own: &[(String, String)], routes: Option<Routes>) {
    if keeping() {
        let rep = placed(p, own, routes);
        LAST.with(|l| *l.borrow_mut() = Some(rep));
    }
}

/// Findings of a program's own rules, added to its report ([`written`]).
pub(crate) fn found(errors: impl IntoIterator<Item = String>) {
    LAST.with(|l| {
        if let Some(rep) = l.borrow_mut().as_mut() {
            rep.errors.extend(errors);
        }
    });
}

/// The check of the program written last (since [`keep`]), once.
pub fn last() -> Option<Report> {
    LAST.with(|l| l.borrow_mut().take())
}

/// The checks over routes `r` and tiles `tiles`, the PEs' files in `files`, `routes` a decode
/// program's from its rcfg.
fn placed_with(
    files: &BTreeMap<String, String>,
    mut r: Routes,
    tiles: &Tiles,
    routes: Option<Routes>,
) -> Report {
    let mut rep = Report::default();
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
    for (&(x, y), t) in tiles {
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
    for ((f, p), h) in &seen {
        let params: BTreeMap<String, i64> = p.iter().cloned().collect();
        for h in hazard::hazards(files, f, &params) {
            rep.errors.push(h);
        }
        let lo = lo_dsds(files, f, &params);
        if lo > LO_DSDS {
            rep.errors.push(format!(
                "`{f}`: ~{lo} constant DSD values, past the {LO_DSDS} a PE's low data holds \
                 (data.lo): it cannot link"
            ));
        }
        let reached = code::reached(files, f, &params);
        let code = code_bytes(&reached) + kind_code(files, f);
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
    // (a decode program's memory is its own model's: the estimate misses its comptime-sized
    // arrays; reported only)
    if !runtime_queues && let Some((f, d, c)) = rep.fullest.clone() {
        let pe = PeBytes { data: d, code: c };
        if pe.overflows() {
            rep.errors.push(format!(
                "`{f}`: {d} B data + ~{c} B code past a PE's {PE_BYTES} B by more than the \
                 estimate's {OVER} B miss"
            ));
        } else if pe.total() + miss(&f, files.get(&f).map(String::as_str)) > PE_BYTES {
            rep.warnings.push(format!(
                "`{f}`: {d} B data + ~{c} B code within the estimate's miss of a PE's \
                 {PE_BYTES} B: it may not link"
            ));
        }
    }
    rep
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
        use crate::compile::emit::placed::At;
        let mut pl = Placed::new(String::new(), 3, 1);
        pl.route(
            4,
            At::Region((0, 2, 1), (0, 1, 1)),
            "WEST",
            &["EAST", "RAMP"],
        );
        pl.route(4, At::Pe(2, 0), "NORTH", &["RAMP"]);
        let (r, _) = layout_ir(&pl);
        assert_eq!(r.at.len(), 3);
        let e = flow(&r);
        assert_eq!(e.len(), 1, "{e:?}");
        assert!(e[0].contains("(1, 0) colour 4: sends east to (2, 0)"));
    }

    #[test]
    fn a_queue_on_a_colour_never_delivered_is_found_and_memory_is_counted() {
        let pe = "param C: u16;\nconst N: u16 = 100;\nvar buf = @zeros([N * 2]f32);\nconst q = @get_input_queue(3);\ncomptime { @initialize_queue(q, .{ .color = @get_color(C) }); }\n";
        use crate::compile::emit::placed::At;
        let mut pl = Placed::new(String::new(), 2, 1);
        pl.tile(At::Pe(0, 0), "pe.csl", &[("C", "5".into())]);
        pl.tile(At::Pe(1, 0), "pe.csl", &[("C", "6".into())]);
        pl.route(5, At::Pe(0, 0), "RAMP", &["EAST"]);
        pl.route(5, At::Pe(1, 0), "WEST", &["RAMP"]);
        let rep = placed(&pl, &[("pe.csl".into(), pe.into())], None);
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

    /// Every catalog model's decode program of the decode identity's configs, planned and built
    /// as the tools do, passes the check.
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
            let p = crate::compile::target::plan(&t, &crate::compile::target::Ask::default())
                .unwrap_or_else(|e| panic!("{m}: {e}"));
            plans.push((m.to_string(), p.plan));
        }
        keep();
        for (m, plan) in plans {
            let out = scratch(&m.replace(['/', '.'], "_"));
            let room = crate::compile::cost::memory::room(
                &plan,
                crate::compile::cost::memory::anchor(&plan.shape),
            );
            let levers =
                crate::compile::emit::params::Levers::of(&plan.levers, "").expect("levers");
            let prog =
                crate::compile::program::Program::of(plan).unwrap_or_else(|e| panic!("{m}: {e}"));
            crate::compile::build::emit(&prog, &levers, None, &out, &crate::compile::build::EMITS)
                .unwrap_or_else(|e| panic!("{m}: {e}"));
            let mut rep = last().expect("the build's check");
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
        use crate::compile::stream::{self, Options};
        let o = |reading: &str| Options {
            reading: reading.into(),
            rows: 16,
            round16: true,
            upto: None,
            taps: Vec::new(),
            anneal: Default::default(),
        };
        let trace = |m: &str| {
            poem_compiler::catalog::repository()
                .parse(m)
                .unwrap_or_else(|| panic!("{m} parses"))
                .trace(poem_ir::Platform::Cerebras)
        };
        let mini = trace("flux2-mini-bf16-kv-bf16");
        keep();
        // compile on load's text and denoise, the stream regime's denoise, the VAE's grid and
        // tiled programs
        for reading in ["text", "denoise"] {
            let out = scratch(&format!("mini-{reading}"));
            stream::compile_whole(&mini, &o(reading), &[], &out).expect("it compiles");
            let rep = last().expect("the program's check");
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
        let rep = last().expect("the program's check");
        eprintln!("flux2-mini denoise (stream): {}", rep.line());
        assert!(
            rep.errors.is_empty(),
            "{:?}",
            &rep.errors[..rep.errors.len().min(5)]
        );
        let _ = std::fs::remove_dir_all(&out);
        let out = scratch("mini-vae");
        stream::compile_reading(&mini, &o("vae.decode"), &out).expect("the VAE");
        let rep = last().expect("the program's check");
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
        let b = crate::compile::floorplan::stream::Bounds::system();
        let mut ko = o("denoise");
        ko.rows = 64;
        let p =
            stream::gemm::program(&klein, &ko, &rows, None, (b.w, b.h), 1).expect("klein places");
        let e = crate::compile::emit::gemm::emit(&p).expect("it emits");
        // (the check reads the placement as the emitter states it)
        let rep = placed(&e.placed, &e.files, None);
        eprintln!("klein 1024² denoise: {}", rep.line());
        assert!(
            rep.errors.is_empty(),
            "{:?}",
            &rep.errors[..rep.errors.len().min(5)]
        );
        assert!(rep.margin() > 0, "{}", rep.line());
    }

    #[test]
    fn a_fabric_dsd_constant_takes_no_low_data_slot() {
        let est = |extra: &str| {
            let src = format!(
                "var a = @zeros([4]f32);\nconst D_a = @get_dsd(mem1d_dsd, .{{ .tensor_access = |i|{{4}} -> a[i] }});\n{extra}"
            );
            let files: BTreeMap<String, String> =
                [("pe.csl".to_string(), src)].into_iter().collect();
            lo_dsds(&files, "pe.csl", &BTreeMap::new())
        };
        let base = est("");
        // (a fabout constant, as a line send's advancing-switch DSD: none; 1,008 B linked either way)
        assert_eq!(
            est(
                "const D_o = @get_dsd(fabout_dsd, .{ .extent = 1, .output_queue = q, .advance_switch = true });\n"
            ),
            base
        );
        assert_eq!(
            est(
                "var b = @zeros([4]f32);\nconst D_b = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{4} -> b[i] });\n"
            ),
            base + 1
        );
    }
}

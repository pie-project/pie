//! The `when` language: comparisons on declared variables joined by `and`,
//! `or` and `not`.
//!
//! ```text
//! expr := or
//! or   := and ("or" and)*
//! and  := not ("and" not)*
//! not  := "not" not | atom
//! atom := "(" expr ")"
//!       | var cmp int                    integer variables
//!       | var "in" int ".." int          inclusive both ends
//!       | var ("==" | "!=") name         enumerated and boolean variables
//!       | var "in" "(" name ("," name)* ")"
//!       | var                            a boolean variable that holds
//! cmp  := < | <= | == | != | >= | >
//! ```
//!
//! Integers only, with `_` separators. No arithmetic, no floats, no strings,
//! no variable-to-variable comparison.

use crate::region::{Region, Set};
use crate::spec::{Domain, Var, space};

/// A parse or type error, with the byte offset into the expression.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    pub at: usize,
    pub message: String,
}

impl Error {
    fn new(at: usize, message: impl Into<String>) -> Error {
        Error {
            at,
            message: message.into(),
        }
    }
}

/// A compiled `when`: its text, the region it holds, and the variables it
/// names.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct When {
    pub text: String,
    pub region: Region,
    pub names: Vec<&'static str>,
}

impl When {
    /// Parse `text` and normalise it over `vars`.
    pub fn compile(text: &str, vars: &[Var]) -> Result<When, Error> {
        let tokens = lex(text)?;
        let mut parser = Parser {
            tokens: &tokens,
            pos: 0,
            vars,
            space: space(vars),
            names: Vec::new(),
            end: text.len(),
        };
        let region = parser.or()?;
        if let Some(t) = parser.peek() {
            return Err(Error::new(
                t.at,
                format!(
                    "expected `and`, `or` or the end, found {}",
                    t.tok.describe()
                ),
            ));
        }
        let mut names = parser.names;
        names.sort_unstable();
        names.dedup();
        Ok(When {
            text: text.to_string(),
            region,
            names,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Tok {
    Name(String),
    Int(i64),
    Cmp(Cmp),
    DotDot,
    Open,
    Close,
    Comma,
}

impl Tok {
    fn describe(&self) -> String {
        match self {
            Tok::Name(n) => format!("`{n}`"),
            Tok::Int(v) => format!("`{v}`"),
            Tok::Cmp(c) => format!("`{}`", c.text()),
            Tok::DotDot => "`..`".to_string(),
            Tok::Open => "`(`".to_string(),
            Tok::Close => "`)`".to_string(),
            Tok::Comma => "`,`".to_string(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cmp {
    Lt,
    Le,
    Eq,
    Ne,
    Ge,
    Gt,
}

impl Cmp {
    fn text(self) -> &'static str {
        match self {
            Cmp::Lt => "<",
            Cmp::Le => "<=",
            Cmp::Eq => "==",
            Cmp::Ne => "!=",
            Cmp::Ge => ">=",
            Cmp::Gt => ">",
        }
    }
}

#[derive(Clone, Debug)]
struct Token {
    tok: Tok,
    at: usize,
}

fn lex(text: &str) -> Result<Vec<Token>, Error> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        let at = i;
        match c {
            b' ' | b'\t' | b'\n' | b'\r' => {
                i += 1;
                continue;
            }
            b'(' | b')' | b',' => {
                let tok = match c {
                    b'(' => Tok::Open,
                    b')' => Tok::Close,
                    _ => Tok::Comma,
                };
                out.push(Token { tok, at });
                i += 1;
            }
            b'.' if bytes.get(i + 1) == Some(&b'.') => {
                out.push(Token {
                    tok: Tok::DotDot,
                    at,
                });
                i += 2;
            }
            b'<' | b'>' | b'=' | b'!' => {
                let two = bytes.get(i + 1) == Some(&b'=');
                let cmp = match (c, two) {
                    (b'<', false) => Cmp::Lt,
                    (b'<', true) => Cmp::Le,
                    (b'>', false) => Cmp::Gt,
                    (b'>', true) => Cmp::Ge,
                    (b'=', true) => Cmp::Eq,
                    (b'!', true) => Cmp::Ne,
                    (b'=', false) => {
                        return Err(Error::new(at, "`=` is not a comparison; write `==`"));
                    }
                    _ => {
                        return Err(Error::new(
                            at,
                            "`!` is not an operator; write `not` or `!=`",
                        ));
                    }
                };
                out.push(Token {
                    tok: Tok::Cmp(cmp),
                    at,
                });
                i += if two { 2 } else { 1 };
            }
            b'0'..=b'9' | b'-' => {
                let start = i;
                i += 1;
                while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'_') {
                    i += 1;
                }
                if i < bytes.len() && (bytes[i] == b'.' && bytes.get(i + 1) != Some(&b'.')) {
                    return Err(Error::new(start, "`when` compares integers only"));
                }
                let digits: String = text[start..i].chars().filter(|&c| c != '_').collect();
                let v = digits.parse::<i64>().map_err(|_| {
                    Error::new(start, format!("`{}` is not an integer", &text[start..i]))
                })?;
                out.push(Token {
                    tok: Tok::Int(v),
                    at: start,
                });
            }
            b'a'..=b'z' | b'A'..=b'Z' | b'_' => {
                let start = i;
                while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                    i += 1;
                }
                out.push(Token {
                    tok: Tok::Name(text[start..i].to_string()),
                    at: start,
                });
            }
            b'"' | b'\'' => return Err(Error::new(at, "`when` has no strings; write a bare name")),
            b'+' | b'*' | b'/' | b'%' => {
                return Err(Error::new(at, "`when` has no arithmetic"));
            }
            _ => {
                return Err(Error::new(
                    at,
                    format!(
                        "`{}` is not part of `when`",
                        text[at..].chars().next().unwrap_or('?')
                    ),
                ));
            }
        }
    }
    Ok(out)
}

struct Parser<'a> {
    tokens: &'a [Token],
    pos: usize,
    vars: &'a [Var],
    space: Vec<Set>,
    names: Vec<&'static str>,
    end: usize,
}

impl Parser<'_> {
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn here(&self) -> usize {
        self.peek().map_or(self.end, |t| t.at)
    }

    fn keyword(&mut self, word: &str) -> bool {
        if matches!(self.peek(), Some(Token { tok: Tok::Name(n), .. }) if n == word) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn expect(&mut self, tok: &Tok, what: &str) -> Result<usize, Error> {
        match self.peek() {
            Some(t) if &t.tok == tok => {
                let at = t.at;
                self.pos += 1;
                Ok(at)
            }
            Some(t) => Err(Error::new(
                t.at,
                format!("expected {what}, found {}", t.tok.describe()),
            )),
            None => Err(Error::new(
                self.end,
                format!("expected {what}, found the end"),
            )),
        }
    }

    fn or(&mut self) -> Result<Region, Error> {
        let mut r = self.and()?;
        while self.keyword("or") {
            r = r.union(&self.and()?);
        }
        Ok(r)
    }

    fn and(&mut self) -> Result<Region, Error> {
        let mut r = self.not()?;
        while self.keyword("and") {
            r = r.intersect(&self.not()?);
        }
        Ok(r)
    }

    fn not(&mut self) -> Result<Region, Error> {
        if self.keyword("not") {
            Ok(self.not()?.complement(&self.space))
        } else {
            self.atom()
        }
    }

    fn int(&mut self) -> Result<i64, Error> {
        match self.peek().cloned() {
            Some(Token {
                tok: Tok::Int(v), ..
            }) => {
                self.pos += 1;
                Ok(v)
            }
            Some(t) => Err(Error::new(
                t.at,
                format!("expected an integer, found {}", t.tok.describe()),
            )),
            None => Err(Error::new(self.end, "expected an integer, found the end")),
        }
    }

    fn word(&mut self, var: &Var) -> Result<i64, Error> {
        match self.peek().cloned() {
            Some(Token {
                tok: Tok::Name(n),
                at,
            }) => {
                self.pos += 1;
                var.domain.word(&n).ok_or_else(|| {
                    Error::new(
                        at,
                        format!(
                            "`{n}` is not a value of `{}`; declared: {}",
                            var.name,
                            values(&var.domain)
                        ),
                    )
                })
            }
            Some(t) => Err(Error::new(
                t.at,
                format!(
                    "`{}` compares by name, found {}",
                    var.name,
                    t.tok.describe()
                ),
            )),
            None => Err(Error::new(self.end, "expected a name, found the end")),
        }
    }

    fn atom(&mut self) -> Result<Region, Error> {
        let Some(token) = self.peek().cloned() else {
            return Err(Error::new(self.end, "expected a comparison, found the end"));
        };
        match token.tok {
            Tok::Open => {
                self.pos += 1;
                let r = self.or()?;
                self.expect(&Tok::Close, "`)`")?;
                Ok(r)
            }
            Tok::Name(name) if !matches!(name.as_str(), "and" | "or" | "not" | "in") => {
                self.pos += 1;
                let Some(index) = self.vars.iter().position(|v| v.name == name) else {
                    return Err(Error::new(
                        token.at,
                        format!(
                            "`{name}` is not a variable here; declared: {}",
                            if self.vars.is_empty() {
                                "none".to_string()
                            } else {
                                self.vars
                                    .iter()
                                    .map(|v| v.name)
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            }
                        ),
                    ));
                };
                let var = self.vars[index];
                self.names.push(var.name);
                self.comparison(index, &var, token.at)
            }
            other => Err(Error::new(
                token.at,
                format!("expected a variable, found {}", other.describe()),
            )),
        }
    }

    fn comparison(&mut self, index: usize, var: &Var, at: usize) -> Result<Region, Error> {
        let set = match self.peek().map(|t| t.tok.clone()) {
            Some(Tok::Cmp(cmp)) => {
                self.pos += 1;
                if var.domain.is_ordered() {
                    let v = self.int()?;
                    match cmp {
                        Cmp::Lt => Set::span(i64::MIN, v.saturating_sub(1)),
                        Cmp::Le => Set::span(i64::MIN, v),
                        Cmp::Eq => Set::point(v),
                        Cmp::Ne => Set::span(i64::MIN, i64::MAX).minus(&Set::point(v)),
                        Cmp::Ge => Set::span(v, i64::MAX),
                        Cmp::Gt => Set::span(v.saturating_add(1), i64::MAX),
                    }
                } else {
                    let v = self.word(var)?;
                    match cmp {
                        Cmp::Eq => Set::point(v),
                        Cmp::Ne => var.domain.set().minus(&Set::point(v)),
                        _ => {
                            return Err(Error::new(
                                at,
                                format!(
                                    "`{}` has no order; compare it with `==`, `!=` or `in (...)`",
                                    var.name
                                ),
                            ));
                        }
                    }
                }
            }
            Some(Tok::Name(n)) if n == "in" => {
                self.pos += 1;
                if var.domain.is_ordered() {
                    let lo = self.int()?;
                    self.expect(&Tok::DotDot, "`..`")?;
                    let hi_at = self.here();
                    let hi = self.int()?;
                    if lo > hi {
                        return Err(Error::new(hi_at, format!("`{lo}..{hi}` holds nothing")));
                    }
                    Set::span(lo, hi)
                } else {
                    self.expect(&Tok::Open, "`(`")?;
                    let mut set = Set::point(self.word(var)?);
                    while self.peek().map(|t| &t.tok) == Some(&Tok::Comma) {
                        self.pos += 1;
                        set = set.union(&Set::point(self.word(var)?));
                    }
                    self.expect(&Tok::Close, "`)`")?;
                    set
                }
            }
            _ if var.domain == Domain::Bool => Set::point(1),
            _ => {
                return Err(Error::new(
                    self.here(),
                    format!("`{}` needs a comparison", var.name),
                ));
            }
        };
        Ok(Region::on(&self.space, index, &set))
    }
}

fn values(domain: &Domain) -> String {
    match *domain {
        Domain::Enum(names) => names.join(", "),
        Domain::Bool => "true, false".to_string(),
        _ => domain.describe(),
    }
}

/// Render a region back as a `when`, for dumps and doctor. `None` when the
/// region is the whole space.
#[must_use]
pub fn render(region: &Region, vars: &[Var]) -> Option<String> {
    let space = space(vars);
    if Region::full(&space).is_subset(region, &space) {
        return None;
    }
    if region.is_empty() {
        return Some("false".to_string());
    }
    let mut boxes: Vec<String> = Vec::new();
    for b in region.boxes() {
        let mut parts: Vec<String> = Vec::new();
        for (i, set) in b.iter().enumerate() {
            if space[i].is_subset(set) {
                continue;
            }
            parts.push(render_set(&vars[i], set, &space[i]));
        }
        boxes.push(parts.join(" and "));
    }
    if boxes.len() == 1 {
        Some(boxes.remove(0))
    } else {
        Some(
            boxes
                .into_iter()
                .map(|b| {
                    if b.contains(" and ") {
                        format!("({b})")
                    } else {
                        b
                    }
                })
                .collect::<Vec<_>>()
                .join(" or "),
        )
    }
}

fn render_set(var: &Var, set: &Set, full: &Set) -> String {
    let name = var.name;
    if !var.domain.is_ordered() {
        let names: Vec<String> = set
            .intervals()
            .iter()
            .flat_map(|&(lo, hi)| lo..=hi)
            .map(|v| var.domain.render(v))
            .collect();
        if var.domain == Domain::Bool && names.len() == 1 {
            return if names[0] == "true" {
                name.to_string()
            } else {
                format!("not {name}")
            };
        }
        return if names.len() == 1 {
            format!("{name} == {}", names[0])
        } else {
            format!("{name} in ({})", names.join(", "))
        };
    }
    let (Some(lo_full), Some(hi_full)) = (full.min(), full.max()) else {
        return "false".to_string();
    };
    let one = |lo: i64, hi: i64| -> String {
        if lo == hi {
            format!("{name} == {lo}")
        } else if lo <= lo_full {
            format!("{name} <= {hi}")
        } else if hi >= hi_full {
            format!("{name} >= {lo}")
        } else {
            format!("{name} in {lo}..{hi}")
        }
    };
    let parts: Vec<String> = set
        .intervals()
        .iter()
        .map(|&(lo, hi)| one(lo, hi))
        .collect();
    if parts.len() == 1 {
        parts[0].clone()
    } else {
        format!("({})", parts.join(" or "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DTYPES: &[&str] = &["bf16", "f16", "f32"];
    const VARS: &[Var] = &[
        Var::dynamic("rows", Domain::COUNT),
        Var::fixed("n", Domain::COUNT),
        Var::fixed("bits", Domain::Ints(&[2, 4, 8])),
        Var::fixed("dtype", Domain::Enum(DTYPES)),
    ];

    fn at(rows: i64, n: i64, bits: i64, dtype: &str) -> Vec<i64> {
        vec![rows, n, bits, VARS[3].domain.word(dtype).unwrap()]
    }

    fn holds(text: &str, point: &[i64]) -> bool {
        When::compile(text, VARS).unwrap().region.contains(point)
    }

    #[test]
    fn comparisons_join_by_and_or_not() {
        assert!(holds("rows < 5 and n >= 65_536", &at(4, 65536, 4, "bf16")));
        assert!(!holds("rows < 5 and n >= 65_536", &at(5, 65536, 4, "bf16")));
        assert!(holds("rows < 5 or n >= 65536", &at(9, 70000, 4, "bf16")));
        assert!(holds("not (rows < 5)", &at(5, 0, 4, "bf16")));
        assert!(
            holds("rows in 4..8", &at(8, 0, 4, "bf16")),
            "a range holds both ends"
        );
        assert!(!holds("rows in 4..8", &at(9, 0, 4, "bf16")));
    }

    #[test]
    fn and_binds_tighter_than_or() {
        let text = "rows < 2 or rows > 10 and n == 1";
        assert!(holds(text, &at(1, 0, 4, "bf16")));
        assert!(!holds(text, &at(11, 0, 4, "bf16")));
        assert!(holds(text, &at(11, 1, 4, "bf16")));
    }

    #[test]
    fn enumerations_compare_by_name_only() {
        assert!(holds("dtype == bf16", &at(1, 1, 4, "bf16")));
        assert!(holds("dtype in (f16, f32)", &at(1, 1, 4, "f32")));
        assert!(holds("dtype != bf16", &at(1, 1, 4, "f16")));
        let e = When::compile("dtype < f16", VARS).unwrap_err();
        assert!(e.message.contains("no order"), "{e:?}");
        let e = When::compile("dtype == fp8", VARS).unwrap_err();
        assert_eq!(e.at, 9);
        assert!(e.message.contains("declared: bf16, f16, f32"), "{e:?}");
    }

    #[test]
    fn a_finite_integer_variable_holds_only_its_values() {
        let w = When::compile("bits in 3..8", VARS).unwrap();
        assert!(w.region.contains(&at(0, 0, 4, "bf16")));
        let space = space(VARS);
        let four_or_eight = When::compile("bits == 4 or bits == 8", VARS)
            .unwrap()
            .region;
        assert!(w.region.is_subset(&four_or_eight, &space));
    }

    #[test]
    fn refusals_point_at_the_offending_column() {
        let e = When::compile("keys < 256", VARS).unwrap_err();
        assert_eq!(e.at, 0);
        assert_eq!(
            e.message,
            "`keys` is not a variable here; declared: rows, n, bits, dtype"
        );
        assert_eq!(When::compile("rows < 4.5", VARS).unwrap_err().at, 7);
        for (text, want) in [
            ("rows < n", "integer"),
            ("rows = 4", "`==`"),
            ("rows < 4 +1", "arithmetic"),
            ("rows < 4 rows", "expected `and`"),
            ("rows in 8..4", "holds nothing"),
            ("(rows < 4", "`)`"),
        ] {
            let e = When::compile(text, VARS).unwrap_err();
            assert!(e.message.contains(want), "`{text}`: {e:?}");
        }
    }

    #[test]
    fn a_boolean_fact_reads_bare() {
        const FACTS: &[Var] = &[
            Var::fixed("family", Domain::COUNT),
            Var::fixed("ane", Domain::Bool),
        ];
        let w = When::compile("family >= 10 and ane", FACTS).unwrap();
        assert!(w.region.contains(&[10, 1]));
        assert!(!w.region.contains(&[10, 0]));
        assert!(
            When::compile("not ane", FACTS)
                .unwrap()
                .region
                .contains(&[7, 0])
        );
    }

    #[test]
    fn a_rendered_region_compiles_back_to_itself() {
        let space = space(VARS);
        for text in [
            "rows < 5 and n >= 65536",
            "rows in 5..11 or dtype == f32",
            "bits in 4..8",
            "not (rows < 5)",
            "dtype in (bf16, f16) and rows > 0",
        ] {
            let w = When::compile(text, VARS).unwrap();
            let back = render(&w.region, VARS).unwrap();
            let again = When::compile(&back, VARS).unwrap();
            assert!(
                w.region.is_subset(&again.region, &space)
                    && again.region.is_subset(&w.region, &space),
                "`{text}` rendered as `{back}`"
            );
        }
        assert_eq!(render(&Region::full(&space), VARS), None);
    }
}

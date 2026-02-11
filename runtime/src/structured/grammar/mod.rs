pub mod builder;
pub mod ebnf;
pub(crate) mod normalize;
pub(crate) mod optimizer;

use std::fmt;

/// Index into the grammar's rule list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RuleId(pub u32);

/// Index into the grammar's expression arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExprId(pub u32);

/// A grammar rule: a named production with a body expression.
#[derive(Debug, Clone)]
pub struct Rule {
    pub name: String,
    pub body: ExprId,
    pub lookahead: Option<Lookahead>,
}

/// A lookahead assertion attached to a rule.
#[derive(Debug, Clone)]
pub struct Lookahead {
    pub expr: ExprId,
    pub is_exact: bool,
}

/// A grammar expression node.
///
/// Expressions are stored in an arena (`Grammar.exprs`) and referenced by `ExprId`.
/// This gives cache-friendly access without lifetime issues, while being fully
/// type-safe unlike the C++ CSR-encoded flat array.
#[derive(Debug, Clone)]
pub enum Expr {
    /// The empty string `""`.
    EmptyString,

    /// A literal byte string (UTF-8 encoded).
    ByteString(Vec<u8>),

    /// A character class matching Unicode codepoint ranges, e.g. `[a-z0-9]`.
    /// When `negated` is true, matches any codepoint NOT in the ranges.
    CharacterClass {
        negated: bool,
        /// Inclusive ranges of Unicode codepoints: `(lower, upper)`.
        ranges: Vec<(u32, u32)>,
    },

    /// Kleene star of a character class, e.g. `[a-z]*`.
    /// Optimized to avoid rule recursion during matching.
    CharacterClassStar {
        negated: bool,
        ranges: Vec<(u32, u32)>,
    },

    /// A reference to another rule.
    RuleRef(RuleId),

    /// An ordered sequence of expressions (concatenation).
    Sequence(Vec<ExprId>),

    /// A choice between expressions (alternation / union).
    Choices(Vec<ExprId>),

    /// Bounded repetition of a rule: `rule{min, max}`.
    /// `max = None` means unbounded.
    Repeat {
        rule: RuleId,
        min: u32,
        max: Option<u32>,
    },
}

/// An immutable context-free grammar.
///
/// Constructed via `GrammarBuilder` or `Grammar::from_ebnf()`.
/// Expressions are stored in a flat arena for cache efficiency.
#[derive(Debug, Clone)]
pub struct Grammar {
    pub(crate) rules: Vec<Rule>,
    pub(crate) exprs: Vec<Expr>,
    pub(crate) root_rule: RuleId,
}

impl Grammar {
    /// Get the root rule id.
    pub fn root_rule(&self) -> RuleId {
        self.root_rule
    }

    /// Get a rule by id.
    pub fn get_rule(&self, id: RuleId) -> &Rule {
        &self.rules[id.0 as usize]
    }

    /// Get an expression by id.
    pub fn get_expr(&self, id: ExprId) -> &Expr {
        &self.exprs[id.0 as usize]
    }

    /// Number of rules in the grammar.
    pub fn num_rules(&self) -> usize {
        self.rules.len()
    }

    /// Number of expressions in the grammar.
    pub fn num_exprs(&self) -> usize {
        self.exprs.len()
    }

    /// Iterate over all rules.
    pub fn rules(&self) -> &[Rule] {
        &self.rules
    }

    /// Get the root rule.
    pub fn root(&self) -> &Rule {
        self.get_rule(self.root_rule)
    }
}

impl fmt::Display for Grammar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, rule) in self.rules.iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            write!(f, "{} ::= ", rule.name)?;
            self.fmt_expr(f, rule.body)?;
            if let Some(ref la) = rule.lookahead {
                write!(f, " (= ")?;
                self.fmt_expr(f, la.expr)?;
                write!(f, ")")?;
            }
        }
        Ok(())
    }
}

impl Grammar {
    fn fmt_expr(&self, f: &mut fmt::Formatter<'_>, id: ExprId) -> fmt::Result {
        match self.get_expr(id) {
            Expr::EmptyString => write!(f, "\"\""),
            Expr::ByteString(bytes) => {
                write!(f, "\"")?;
                for &b in bytes {
                    match b {
                        b'\\' => write!(f, "\\\\")?,
                        b'"' => write!(f, "\\\"")?,
                        b'\n' => write!(f, "\\n")?,
                        b'\r' => write!(f, "\\r")?,
                        b'\t' => write!(f, "\\t")?,
                        0x20..=0x7e => write!(f, "{}", b as char)?,
                        _ => write!(f, "\\x{:02x}", b)?,
                    }
                }
                write!(f, "\"")
            }
            Expr::CharacterClass { negated, ranges } => {
                write!(f, "[")?;
                if *negated {
                    write!(f, "^")?;
                }
                for &(lo, hi) in ranges {
                    if lo == hi {
                        Self::fmt_char_class_char(f, lo)?;
                    } else {
                        Self::fmt_char_class_char(f, lo)?;
                        write!(f, "-")?;
                        Self::fmt_char_class_char(f, hi)?;
                    }
                }
                write!(f, "]")
            }
            Expr::CharacterClassStar { negated, ranges } => {
                write!(f, "[")?;
                if *negated {
                    write!(f, "^")?;
                }
                for &(lo, hi) in ranges {
                    if lo == hi {
                        Self::fmt_char_class_char(f, lo)?;
                    } else {
                        Self::fmt_char_class_char(f, lo)?;
                        write!(f, "-")?;
                        Self::fmt_char_class_char(f, hi)?;
                    }
                }
                write!(f, "]*")
            }
            Expr::RuleRef(rule_id) => {
                write!(f, "{}", self.rules[rule_id.0 as usize].name)
            }
            Expr::Sequence(exprs) => {
                write!(f, "(")?;
                for (i, &eid) in exprs.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    self.fmt_expr(f, eid)?;
                }
                write!(f, ")")
            }
            Expr::Choices(exprs) => {
                write!(f, "(")?;
                for (i, &eid) in exprs.iter().enumerate() {
                    if i > 0 {
                        write!(f, " | ")?;
                    }
                    self.fmt_expr(f, eid)?;
                }
                write!(f, ")")
            }
            Expr::Repeat { rule, min, max } => {
                let name = &self.rules[rule.0 as usize].name;
                match max {
                    Some(max) => write!(f, "{}{{{},{}}}", name, min, max),
                    None => write!(f, "{}{{{},}}", name, min),
                }
            }
        }
    }

    fn fmt_char_class_char(f: &mut fmt::Formatter<'_>, cp: u32) -> fmt::Result {
        match cp {
            0x5c => write!(f, "\\\\"),         // backslash
            0x5d => write!(f, "\\]"),           // ]
            0x5e => write!(f, "\\^"),           // ^
            0x2d => write!(f, "\\-"),           // -
            0x09 => write!(f, "\\t"),
            0x0a => write!(f, "\\n"),
            0x0d => write!(f, "\\r"),
            cp if cp >= 0x20 && cp <= 0x7e => {
                if let Some(c) = char::from_u32(cp) {
                    write!(f, "{}", c)
                } else {
                    write!(f, "\\u{{{:04x}}}", cp)
                }
            }
            cp if cp <= 0xffff => write!(f, "\\u{:04x}", cp),
            cp => write!(f, "\\U{:08x}", cp),
        }
    }
}

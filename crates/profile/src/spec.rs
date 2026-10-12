//! The registry's vocabulary: what a decision declares about itself.
//!
//! Everything here is `'static` data, written by `#[derive(Decision)]` or by
//! hand for a backend's facts.

use crate::region::Set;

/// The values a variable or fact takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Domain {
    /// Integers in `lo..=hi`.
    Int { lo: i64, hi: i64 },
    /// One of a finite list of integers.
    Ints(&'static [i64]),
    /// One of a list of names, compared by equality only.
    Enum(&'static [&'static str]),
    /// `true` or `false`; a bare name in `when` means `true`.
    Bool,
}

impl Domain {
    /// Sizes, counts, widths: every integer variable defaults to this.
    pub const COUNT: Domain = Domain::Int {
        lo: 0,
        hi: i64::MAX,
    };

    #[must_use]
    pub fn set(&self) -> Set {
        match *self {
            Domain::Int { lo, hi } => Set::span(lo, hi),
            Domain::Ints(values) => Set::points(values),
            Domain::Enum(names) => Set::span(0, names.len() as i64 - 1),
            Domain::Bool => Set::span(0, 1),
        }
    }

    #[must_use]
    pub fn is_ordered(&self) -> bool {
        matches!(self, Domain::Int { .. } | Domain::Ints(_))
    }

    /// The coordinate a name stands for: an enumeration's index, or a bool.
    #[must_use]
    pub fn word(&self, name: &str) -> Option<i64> {
        match *self {
            Domain::Enum(names) => names.iter().position(|n| *n == name).map(|i| i as i64),
            Domain::Bool => match name {
                "false" => Some(0),
                "true" => Some(1),
                _ => None,
            },
            _ => None,
        }
    }

    #[must_use]
    pub fn render(&self, v: i64) -> String {
        match *self {
            Domain::Enum(names) => usize::try_from(v)
                .ok()
                .and_then(|i| names.get(i))
                .map_or_else(|| v.to_string(), |n| (*n).to_string()),
            Domain::Bool => (v != 0).to_string(),
            _ => v.to_string(),
        }
    }

    #[must_use]
    pub fn describe(&self) -> String {
        match *self {
            Domain::Int {
                lo: 0,
                hi: i64::MAX,
            } => "integer >= 0".to_string(),
            Domain::Int { lo, hi } => format!("integer in {lo}..{hi}"),
            Domain::Ints(values) => format!(
                "one of {}",
                values
                    .iter()
                    .map(i64::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Domain::Enum(names) => format!("one of {}", names.join(", ")),
            Domain::Bool => "true or false".to_string(),
        }
    }
}

/// A shape variable of a decision, or a fact of a backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Var {
    pub name: &'static str,
    pub domain: Domain,
    /// Set per recorded body rather than per compiled op.
    pub dynamic: bool,
    /// A dynamic variable the caller computes from the backend's recording
    /// axes, rather than an axis itself.
    pub derived: bool,
}

impl Var {
    #[must_use]
    pub const fn fixed(name: &'static str, domain: Domain) -> Var {
        Var {
            name,
            domain,
            dynamic: false,
            derived: false,
        }
    }

    #[must_use]
    pub const fn dynamic(name: &'static str, domain: Domain) -> Var {
        Var {
            name,
            domain,
            dynamic: true,
            derived: false,
        }
    }
}

#[must_use]
pub fn space(vars: &[Var]) -> Vec<Set> {
    vars.iter().map(|v| v.domain.set()).collect()
}

/// How a constant reaches the kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sort {
    /// Baked into the compiled kernel: part of the census.
    Specialize,
    /// Passed per launch.
    Launch,
}

/// The values a constant may take.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    Int { lo: i64, hi: i64 },
    Ints(&'static [i64]),
    Float { lo: f64, hi: f64 },
    Bool,
    Word(&'static [&'static str]),
    Words(&'static [&'static str]),
}

impl Kind {
    #[must_use]
    pub fn describe(&self) -> String {
        match *self {
            Kind::Int { lo, hi } if hi == i64::MAX => format!("integer >= {lo}"),
            Kind::Int { lo, hi } => format!("integer in {lo}..{hi}"),
            Kind::Ints(values) => format!(
                "one of {}",
                values
                    .iter()
                    .map(i64::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Kind::Float { lo, hi } => format!("number in {lo}..{hi}"),
            Kind::Bool => "true or false".to_string(),
            Kind::Word(names) => format!("one of \"{}\"", names.join("\", \"")),
            Kind::Words(names) => format!("a list of \"{}\"", names.join("\", \"")),
        }
    }

    /// Whether `value` is one of this kind's values.
    #[must_use]
    pub fn admits(&self, value: &Value) -> bool {
        match (*self, value) {
            (Kind::Int { lo, hi }, Value::Int(v)) => lo <= *v && *v <= hi,
            (Kind::Ints(values), Value::Int(v)) => values.contains(v),
            (Kind::Float { lo, hi }, Value::Float(v)) => lo <= *v && *v <= hi,
            (Kind::Bool, Value::Bool(_)) => true,
            (Kind::Word(names), Value::Word(w)) => names.contains(&w.as_str()),
            (Kind::Words(names), Value::Words(ws)) => {
                ws.iter().all(|w| names.contains(&w.as_str()))
            }
            _ => false,
        }
    }
}

/// A literal in a profile.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
    Word(String),
    Words(Vec<String>),
}

impl Value {
    #[must_use]
    pub fn as_int(&self) -> i64 {
        match self {
            Value::Int(v) => *v,
            other => panic!("a registry-checked integer constant holds {other:?}"),
        }
    }

    #[must_use]
    pub fn as_float(&self) -> f64 {
        match self {
            Value::Float(v) => *v,
            Value::Int(v) => *v as f64,
            other => panic!("a registry-checked float constant holds {other:?}"),
        }
    }

    #[must_use]
    pub fn as_bool(&self) -> bool {
        match self {
            Value::Bool(v) => *v,
            other => panic!("a registry-checked bool constant holds {other:?}"),
        }
    }

    #[must_use]
    pub fn as_word(&self) -> &str {
        match self {
            Value::Word(v) => v,
            other => panic!("a registry-checked word constant holds {other:?}"),
        }
    }

    #[must_use]
    pub fn as_words(&self) -> Vec<String> {
        match self {
            Value::Words(v) => v.clone(),
            other => panic!("a registry-checked word-list constant holds {other:?}"),
        }
    }

    /// The value spelled as TOML.
    #[must_use]
    pub fn toml(&self) -> String {
        match self {
            Value::Int(v) => group_digits(*v),
            Value::Float(v) => {
                if v.fract() == 0.0 && v.abs() < 1e15 {
                    format!("{v:.1}")
                } else {
                    v.to_string()
                }
            }
            Value::Bool(v) => v.to_string(),
            Value::Word(w) => format!("\"{w}\""),
            Value::Words(ws) => format!(
                "[{}]",
                ws.iter()
                    .map(|w| format!("\"{w}\""))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }
}

fn group_digits(v: i64) -> String {
    let digits = v.unsigned_abs().to_string();
    if digits.len() <= 4 {
        return v.to_string();
    }
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push('_');
        }
        out.push(c);
    }
    if v < 0 { format!("-{out}") } else { out }
}

/// A named number an alternative (or a constant decision) carries.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Constant {
    pub name: &'static str,
    pub kind: Kind,
    pub sort: Sort,
    pub doc: &'static str,
    /// The value an unspecified constant takes, spelled as TOML.
    pub default: &'static str,
}

/// A named implementation of a choice.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Alternative {
    pub name: &'static str,
    pub doc: &'static str,
    pub constants: &'static [Constant],
    /// An expression over the backend's facts; empty means baseline.
    pub requires: &'static str,
    /// Holds only for some static shapes; its `fits` says which. A ladder may
    /// not end on one.
    pub partial: bool,
    /// `"<decision id> = <alternative>"`: a rung on this alternative needs a
    /// rung on that one in the same profile.
    pub consistent_with: &'static [&'static str],
}

/// What kind of decision this is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Shape {
    /// A typed value with no alternatives.
    Constant(Constant),
    /// Two or more alternatives.
    Choice(&'static [Alternative]),
    /// On where listed, off where not.
    Flag,
}

/// How `pie tune` treats a decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tune {
    /// Its alternatives agree numerically; the fastest over the grid wins.
    Search,
    /// A number measured directly, such as a launch's fixed cost.
    Price,
    /// Engine behaviour, judged only by end-to-end runs.
    Engine,
    /// Never tuned: it changes numerics, memory or import, which is the
    /// operator's call. Also every decision that says nothing.
    Never,
}

impl Tune {
    #[must_use]
    pub fn word(self) -> &'static str {
        match self {
            Tune::Search => "search",
            Tune::Price => "price",
            Tune::Engine => "engine",
            Tune::Never => "never",
        }
    }
}

/// Everything a decision declares.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spec {
    pub id: &'static str,
    pub doc: &'static str,
    pub shape: Shape,
    pub vars: &'static [Var],
    /// The registry default, spelled as the TOML value a profile would hold.
    pub default: &'static str,
    pub tune: Tune,
}

impl Spec {
    #[must_use]
    pub fn alternatives(&self) -> &'static [Alternative] {
        match self.shape {
            Shape::Choice(alts) => alts,
            _ => &[],
        }
    }

    #[must_use]
    pub fn kind_word(&self) -> &'static str {
        match self.shape {
            Shape::Constant(_) => "constant",
            Shape::Choice(_) => "choice",
            Shape::Flag => "flag",
        }
    }
}

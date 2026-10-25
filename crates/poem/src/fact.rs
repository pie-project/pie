//! The switches a program branches its rows on.
//!
//! A runtime switch is one the runtime derives from what a lane carries, so
//! an inferlet cannot state it apart from the buffers a kernel reads under
//! it. A model's own switch is one it declares with its values and an
//! inferlet sets; its name and values are the model's.

use std::ops::{BitAnd, Not};

use poem_ir::{Builtin, Guard};

pub use poem_ir::Builtin::{Adapted as Adapter, Masked as Mask, Media};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Predicate {
    Builtin(Builtin),
    Switch(String, String),
    Not(Box<Predicate>),
    And(Box<Predicate>, Box<Predicate>),
}

impl Predicate {
    /// The guard over the rows this predicate holds for, its facts given bits
    /// in `facts` if it does not hold them yet.
    pub fn guard(&self, facts: &mut poem_ir::Facts) -> Guard {
        match self {
            Predicate::Builtin(builtin) => facts.builtin(*builtin),
            Predicate::Switch(name, value) => facts.switch(name, value),
            Predicate::Not(a) => !a.guard(facts),
            Predicate::And(a, b) => Guard::and(a.guard(facts), b.guard(facts)),
        }
    }
}

/// The rows whose lane carries `what`.
#[must_use]
pub fn has(what: Builtin) -> Predicate {
    assert!(
        matches!(what, Builtin::Masked | Builtin::Adapted | Builtin::Media),
        "a lane carries a mask, adapter routes or media, not {what:?}"
    );
    Predicate::Builtin(what)
}

/// The rows of lanes that query one token.
#[must_use]
pub fn single_token() -> Predicate {
    Predicate::Builtin(Builtin::SingleToken)
}

/// The rows whose program reads their drafts.
#[must_use]
pub fn drafts() -> Predicate {
    Predicate::Builtin(Builtin::Drafts)
}

/// The rows whose program reads their attention scores.
#[must_use]
pub fn scores() -> Predicate {
    Predicate::Builtin(Builtin::Scores)
}

/// The rows of a prompt only a causal encoder-decoder's encoder reads.
#[must_use]
pub fn encode() -> Predicate {
    Predicate::Builtin(Builtin::Encode)
}

/// The rows whose switch `name` is set to `value`.
#[must_use]
pub fn switch_is(name: &str, value: &str) -> Predicate {
    Predicate::Switch(name.to_string(), value.to_string())
}

impl BitAnd for Predicate {
    type Output = Predicate;

    fn bitand(self, rhs: Predicate) -> Predicate {
        Predicate::And(Box::new(self), Box::new(rhs))
    }
}

impl Not for Predicate {
    type Output = Predicate;

    fn not(self) -> Predicate {
        Predicate::Not(Box::new(self))
    }
}

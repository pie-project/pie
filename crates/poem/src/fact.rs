//! The facts a program branches its rows on.
//!
//! A builtin fact is one the runtime derives from what a lane carries, so an
//! inferlet cannot state it apart from the buffers a kernel reads under it. A
//! custom fact is one the program names and an inferlet sets; the names here
//! are the ones every family shares.

use std::ops::{BitAnd, Not};

use poem_ir::{Builtin, Guard, Request, Stream};

pub use poem_ir::Builtin::{Adapted as Adapter, Masked as Mask, Media};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Predicate {
    Builtin(Builtin),
    Flag(String),
    Choice(String, String),
    Not(Box<Predicate>),
    And(Box<Predicate>, Box<Predicate>),
}

impl Predicate {
    /// The guard over the rows this predicate holds for, its facts given bits
    /// in `facts` if it does not hold them yet.
    pub fn guard(&self, facts: &mut poem_ir::Facts) -> Guard {
        match self {
            Predicate::Builtin(builtin) => facts.builtin(*builtin),
            Predicate::Flag(name) => facts.flag(name),
            Predicate::Choice(name, value) => facts.choice(name, value),
            Predicate::Not(a) => Guard::not(a.guard(facts)),
            Predicate::And(a, b) => Guard::and(a.guard(facts), b.guard(facts)),
        }
    }
}

/// The rows whose lane carries `what`.
#[must_use]
pub fn has(what: Builtin) -> Predicate {
    assert!(
        what != Builtin::SingleToken,
        "a single token is a count, not a buffer; it is `single_token()`"
    );
    Predicate::Builtin(what)
}

/// The rows of lanes that query one token.
#[must_use]
pub fn single_token() -> Predicate {
    Predicate::Builtin(Builtin::SingleToken)
}

/// The rows whose inferlet turned the flag `name` on.
#[must_use]
pub fn flag(name: &str) -> Predicate {
    Predicate::Flag(name.to_string())
}

/// The rows whose inferlet set the choice `name` to `value`.
#[must_use]
pub fn choice(name: &str, value: &str) -> Predicate {
    Predicate::Choice(name.to_string(), value.to_string())
}

#[must_use]
pub fn drafts() -> Predicate {
    flag(Request::DRAFTS)
}

#[must_use]
pub fn block_draft() -> Predicate {
    flag(Request::BLOCK_DRAFT)
}

#[must_use]
pub fn scores() -> Predicate {
    flag(Request::SCORES)
}

#[must_use]
pub fn bidirectional() -> Predicate {
    flag(Request::BIDIRECTIONAL)
}

/// The rows of the stream `stream`.
#[must_use]
pub fn stream(stream: Stream) -> Predicate {
    choice(Request::STREAM, stream.name())
}

/// The rows of the reading `name`.
#[must_use]
pub fn reading(name: &str) -> Predicate {
    choice(Request::READING, name)
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

//! Fire-time facts: model-owned bits, classified per request.

use std::ops::{BitAnd, Not};

/// The per-request view `Classify` reads; the engine constructs one per
/// admitted request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Request {
    query_len: u32,
    custom_mask: bool,
}

impl Request {
    #[must_use]
    pub fn new(query_len: u32, custom_mask: bool) -> Request {
        Request {
            query_len,
            custom_mask,
        }
    }

    #[must_use]
    pub fn query_len(&self) -> u32 {
        self.query_len
    }

    #[must_use]
    pub fn has_custom_mask(&self) -> bool {
        self.custom_mask
    }
}

/// The declared facts: bit-ordered names and the word packing, field
/// order; `#[derive(Facts)]` writes both.
pub trait FactWord {
    const NAMES: &'static [&'static str];
    fn word(&self) -> u64;
}

pub trait Classify: FactWord {
    fn of(r: &Request) -> Self;
}

/// A condition over the fact word, recorded as data and swept at finish;
/// the derive's constructors are the atoms.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Predicate {
    Fact { bit: u8, name: &'static str },
    Not(Box<Predicate>),
    And(Box<Predicate>, Box<Predicate>),
    Rest,
}

impl Predicate {
    #[must_use]
    pub fn fact(bit: u8, name: &'static str) -> Predicate {
        Predicate::Fact { bit, name }
    }

    #[must_use]
    pub fn rest() -> Predicate {
        Predicate::Rest
    }
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

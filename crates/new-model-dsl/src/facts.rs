//! The fact-bit predicate algebra a model splits on. Unchanged in spirit from
//! the old `facts.rs`: `#[derive(Facts)]` on a struct of bools hands the model
//! one `Predicate` constructor and one word bit per field, and `Value::split`
//! lowers predicates to `Cond` trees on the nodes they guard.

use std::ops::{BitAnd, Not};

/// A formula over fact bits, stated at trace time. `Rest` is the n-way
/// split's catch-all arm and legal nowhere else.
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

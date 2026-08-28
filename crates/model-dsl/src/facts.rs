//! The fact-bit predicate algebra a model splits on. A model names its own
//! bits — `Facts::qo_one()` is a hand-written constructor over
//! [`Predicate::fact(0)`](Predicate::fact) — and `Value::split` lowers
//! predicates to `Guard` trees on the nodes they guard.

use std::ops::{BitAnd, Not};

/// A formula over fact bits, stated at trace time. `Rest` is the n-way
/// split's catch-all arm and legal nowhere else.
///
/// A bit is a POSITION AND NOTHING ELSE. The name a model calls it by lives in
/// that model's own `Facts` impl, where it is the name of the constructor and
/// of the struct field, and it never travels into the plan: `Guard::Fact(bit)`
/// is what a guard is, `fact_width` reads the plan's F off those guards, and a
/// second list of names would be a second statement of the same fact, free to
/// disagree with them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Predicate {
    Fact { bit: u8 },
    Not(Box<Predicate>),
    And(Box<Predicate>, Box<Predicate>),
    Rest,
}

impl Predicate {
    #[must_use]
    pub fn fact(bit: u8) -> Predicate {
        Predicate::Fact { bit }
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

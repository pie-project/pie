use std::ops::{BitAnd, Not};

use poem_ir::Stream;

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

    #[must_use]
    pub fn stream(base: u8, stream: Stream) -> Predicate {
        Predicate::fact(base + stream.code())
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

//! The fact-bit predicate algebra a model splits on. Unchanged in spirit from
//! the old `facts.rs`: [`facts!`](crate::facts!) on a list of field names hands
//! the model one `Predicate` constructor and one word bit per field, and
//! `Value::split` lowers predicates to `Cond` trees on the nodes they guard.

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

/// Declares a model's fact vocabulary. Every field is a bool, and its bit in
/// the fact word is its position in the list:
///
/// ```
/// model_dsl::facts! {
///     pub struct Facts { qo_one, masked }
/// }
/// ```
///
/// generates the struct itself, the [`FactWord`](crate::FactWord) impl —
/// `NAMES` in field order, `word` packing each bool into its bit — and one
/// predicate constructor per field: `Facts::qo_one()` is the
/// [`Predicate`](crate::Predicate) testing bit 0. A fact word is one `u64`,
/// so a struct of more than 64 fields is a compile-time error.
#[macro_export]
macro_rules! facts {
    ($vis:vis struct $name:ident { $($field:ident),+ $(,)? }) => {
        $vis struct $name {
            $(pub $field: bool,)+
        }

        impl $name {
            $crate::facts!(@predicates [] $($field)+);
        }

        impl $crate::FactWord for $name {
            const NAMES: &'static [&'static str] = &[$(stringify!($field)),+];

            fn word(&self) -> u64 {
                let mut word = 0u64;
                for (bit, &fact) in [$(self.$field),+].iter().enumerate() {
                    word |= (fact as u64) << bit;
                }
                word
            }
        }

        const _: () = assert!(
            <$name as $crate::FactWord>::NAMES.len() <= 64,
            "a fact word is one u64: at most 64 fields",
        );
    };

    // One predicate constructor, its bit the count of the fields before it.
    (@predicates [$($seen:ident)*] $field:ident $($rest:ident)*) => {
        #[must_use]
        pub fn $field() -> $crate::Predicate {
            const BIT: u8 = <[&str]>::len(&[$(stringify!($seen)),*]) as u8;
            $crate::Predicate::fact(BIT, stringify!($field))
        }

        $crate::facts!(@predicates [$($seen)* $field] $($rest)*);
    };

    (@predicates [$($seen:ident)*]) => {};
}

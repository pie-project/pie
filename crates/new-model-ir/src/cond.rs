//! Guard conditions over the plan's fact word. Ported from the old
//! `model-ir::plan::Cond` unchanged in meaning: `Fact(bit)` indexes
//! `Plan::facts`, and the driver evaluates each node's guard against the
//! fire's fact word.

use serde::{Deserialize, Serialize};

/// A boolean formula over fact bits. Kept as a tree, not a truth table, so a
/// plan prints the same structure the model text stated.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Cond {
    Always,
    Fact(u8),
    Not(Box<Cond>),
    And(Box<Cond>, Box<Cond>),
    Or(Box<Cond>, Box<Cond>),
}

impl Cond {
    /// `Always` is the identity, folded here so traces do not accrete
    /// `And(Always, ..)` wrappers.
    #[must_use]
    pub fn and(a: Cond, b: Cond) -> Cond {
        match (a, b) {
            (Cond::Always, x) | (x, Cond::Always) => x,
            (a, b) => Cond::And(Box::new(a), Box::new(b)),
        }
    }

    #[must_use]
    pub fn or(a: Cond, b: Cond) -> Cond {
        Cond::Or(Box::new(a), Box::new(b))
    }

    #[must_use]
    #[allow(clippy::should_implement_trait)]
    pub fn not(a: Cond) -> Cond {
        Cond::Not(Box::new(a))
    }

    /// Evaluate against a fire's fact word.
    #[must_use]
    pub fn holds(&self, word: u64) -> bool {
        match self {
            Cond::Always => true,
            Cond::Fact(bit) => word & (1 << bit) != 0,
            Cond::Not(a) => !a.holds(word),
            Cond::And(a, b) => a.holds(word) && b.holds(word),
            Cond::Or(a, b) => a.holds(word) || b.holds(word),
        }
    }

    fn bits_into(&self, bits: &mut Vec<u8>) {
        match self {
            Cond::Always => {}
            Cond::Fact(bit) => bits.push(*bit),
            Cond::Not(a) => a.bits_into(bits),
            Cond::And(a, b) | Cond::Or(a, b) => {
                a.bits_into(bits);
                b.bits_into(bits);
            }
        }
    }

    /// The fact bits this condition reads, sorted and deduplicated.
    #[must_use]
    pub fn referenced_bits(&self) -> Vec<u8> {
        let mut bits = Vec::new();
        self.bits_into(&mut bits);
        bits.sort_unstable();
        bits.dedup();
        bits
    }

    /// Collapse a tautology to `Always` by exhausting assignments over the
    /// referenced bits — merges of complementary branches produce these, and
    /// `Always` is what downstream passes test for.
    #[must_use]
    pub fn simplified(self) -> Cond {
        let bits = self.referenced_bits();
        if bits.is_empty() {
            return self;
        }
        assert!(bits.len() <= 20, "a condition over {} facts", bits.len());
        let every = (0..1u64 << bits.len()).all(|assignment| {
            let mut word = 0u64;
            for (i, bit) in bits.iter().enumerate() {
                if assignment & (1 << i) != 0 {
                    word |= 1 << bit;
                }
            }
            self.holds(word)
        });
        if every { Cond::Always } else { self }
    }
}

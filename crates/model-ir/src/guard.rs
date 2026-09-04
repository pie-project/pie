//! Guard conditions over the plan's fact word: `Fact(bit)` names one bit of
//! the word the model's `Classify` computed, and the engine evaluates each
//! node's guard against the fire's fact word.

use serde::{Deserialize, Serialize};

/// A boolean formula over fact bits. Kept as a tree, not a truth table, so a
/// plan prints the same structure the model text stated.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Guard {
    Always,
    Fact(u8),
    Not(Box<Guard>),
    And(Box<Guard>, Box<Guard>),
    Or(Box<Guard>, Box<Guard>),
}

impl Guard {
    /// `Always` is the identity, folded here so traces do not accrete
    /// `And(Always, ..)` wrappers.
    #[must_use]
    pub fn and(a: Guard, b: Guard) -> Guard {
        match (a, b) {
            (Guard::Always, x) | (x, Guard::Always) => x,
            (a, b) => Guard::And(Box::new(a), Box::new(b)),
        }
    }

    #[must_use]
    pub fn or(a: Guard, b: Guard) -> Guard {
        Guard::Or(Box::new(a), Box::new(b))
    }

    #[must_use]
    #[allow(clippy::should_implement_trait)]
    pub fn not(a: Guard) -> Guard {
        Guard::Not(Box::new(a))
    }

    /// Evaluate against a fire's fact word.
    #[must_use]
    pub fn holds(&self, word: u64) -> bool {
        match self {
            Guard::Always => true,
            Guard::Fact(bit) => word & (1 << bit) != 0,
            Guard::Not(a) => !a.holds(word),
            Guard::And(a, b) => a.holds(word) && b.holds(word),
            Guard::Or(a, b) => a.holds(word) || b.holds(word),
        }
    }

    fn bits_into(&self, bits: &mut Vec<u8>) {
        match self {
            Guard::Always => {}
            Guard::Fact(bit) => bits.push(*bit),
            Guard::Not(a) => a.bits_into(bits),
            Guard::And(a, b) | Guard::Or(a, b) => {
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

    /// Whether two guards admit exactly the same lanes — a truth table over
    /// every fact either of them names, not an algebraic argument.
    #[must_use]
    pub fn equivalent(&self, other: &Guard) -> bool {
        let mut bits = self.referenced_bits();
        for bit in other.referenced_bits() {
            if !bits.contains(&bit) {
                bits.push(bit);
            }
        }
        if bits.is_empty() {
            return self.holds(0) == other.holds(0);
        }
        assert!(bits.len() <= 20, "a condition over {} facts", bits.len());
        (0..1u64 << bits.len()).all(|assignment| {
            let mut word = 0u64;
            for (i, bit) in bits.iter().enumerate() {
                if assignment & (1 << i) != 0 {
                    word |= 1 << bit;
                }
            }
            self.holds(word) == other.holds(word)
        })
    }

    /// This guard's conjuncts, flattened; one that is not an `And` is its
    /// own only conjunct.
    fn conjuncts<'a>(&'a self, out: &mut Vec<&'a Guard>) {
        match self {
            Guard::And(a, b) => {
                a.conjuncts(out);
                b.conjuncts(out);
            }
            other => out.push(other),
        }
    }

    /// The conjuncts every one of `arms` carries, `And`ed back together —
    /// [`Always`](Guard::Always) when they share none.
    ///
    /// The arms of one split of a value guarded by `G` are each `And(G, pᵢ)`,
    /// so this recovers `G`. [`Value::merge`](../../model_dsl) needs it
    /// because a merge is compared to its siblings by EQUALITY: nested in an
    /// outer split, the join of the arms must come back spelled as that
    /// outer guard or the next node reading both looks like it mixed arms.
    #[must_use]
    pub fn common(arms: &[Guard]) -> Guard {
        let Some((first, rest)) = arms.split_first() else {
            return Guard::Always;
        };
        let mut shared: Vec<&Guard> = Vec::new();
        first.conjuncts(&mut shared);
        for arm in rest {
            let mut theirs: Vec<&Guard> = Vec::new();
            arm.conjuncts(&mut theirs);
            shared.retain(|c| theirs.iter().any(|t| t == c));
        }
        shared
            .into_iter()
            .cloned()
            .fold(Guard::Always, |a, b| Guard::and(a, b))
    }

    /// Collapse a tautology to `Always` by exhausting assignments over the
    /// referenced bits — merges of complementary branches produce these, and
    /// `Always` is what downstream passes test for.
    #[must_use]
    pub fn simplified(self) -> Guard {
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
        if every { Guard::Always } else { self }
    }
}

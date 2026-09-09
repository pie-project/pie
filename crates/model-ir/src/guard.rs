use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Guard {
    Always,
    Fact(u8),
    Not(Box<Guard>),
    And(Box<Guard>, Box<Guard>),
    Or(Box<Guard>, Box<Guard>),
}

impl Guard {
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
    pub fn narrow(outer: Guard, inner: Guard) -> Guard {
        if matches!(outer, Guard::Always) {
            return inner;
        }
        let joined = Guard::and(outer, inner.clone());
        if joined.equivalent(&inner) {
            inner
        } else {
            joined
        }
    }

    #[must_use]
    #[allow(clippy::should_implement_trait)]
    pub fn not(a: Guard) -> Guard {
        Guard::Not(Box::new(a))
    }

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

    #[must_use]
    pub fn referenced_bits(&self) -> Vec<u8> {
        let mut bits = Vec::new();
        self.bits_into(&mut bits);
        bits.sort_unstable();
        bits.dedup();
        bits
    }

    #[must_use]
    pub fn equivalent(&self, other: &Guard) -> bool {
        self.agree(other, |mine, theirs| mine == theirs)
    }

    #[must_use]
    pub fn implies(&self, outer: &Guard) -> bool {
        self.agree(outer, |mine, theirs| !mine || theirs)
    }

    fn agree(&self, other: &Guard, agree: impl Fn(bool, bool) -> bool) -> bool {
        let mut bits = self.referenced_bits();
        for bit in other.referenced_bits() {
            if !bits.contains(&bit) {
                bits.push(bit);
            }
        }
        if bits.is_empty() {
            return agree(self.holds(0), other.holds(0));
        }
        assert!(bits.len() <= 20, "a condition over {} facts", bits.len());
        (0..1u64 << bits.len()).all(|assignment| {
            let mut word = 0u64;
            for (i, bit) in bits.iter().enumerate() {
                if assignment & (1 << i) != 0 {
                    word |= 1 << bit;
                }
            }
            agree(self.holds(word), other.holds(word))
        })
    }

    fn conjuncts<'a>(&'a self, out: &mut Vec<&'a Guard>) {
        match self {
            Guard::And(a, b) => {
                a.conjuncts(out);
                b.conjuncts(out);
            }
            other => out.push(other),
        }
    }

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
        shared.into_iter().cloned().fold(Guard::Always, Guard::and)
    }

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

//! The supergraph plan: one trace of the text, conditions as data.
//!
//! Tier-1 statements name a role point (`attention.decode`); a plane-gated
//! statement names its plane's symbol (`cuda::...`). Resolution is the
//! lowering's lookup, never the text's.

use serde::{Deserialize, Serialize};

pub type ValueId = u32;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Plan {
    /// The catalog SKU this plan monomorphizes.
    pub name: String,
    pub plane: crate::kernels::Backend,
    /// Declared fact names, bit-ordered; conditions index into this.
    pub facts: Vec<String>,
    /// The load contract: every weight the text touched, canonical.
    pub params: Vec<Param>,
    pub caches: Vec<CacheRow>,
    pub values: Vec<ValueDef>,
    pub ops: Vec<Op>,
    pub seams: Vec<Seam>,
}

/// One weight: canonical zt name and shape, the rank cut, the storage repr.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Param {
    pub name: String,
    pub shape: Vec<u64>,
    pub shard: Shard,
    pub repr: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Shard {
    Replicated,
    Columns,
    Rows,
    Packed(Vec<u64>),
    Experts,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CacheRow {
    /// Paged; one `row` appended per token, discardable.
    Kv { name: String, row: Vec<u64> },
    /// One slab per request, folded in place.
    State { name: String, slab: Vec<u64> },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValueDef {
    /// A driver-provided tensor by its runtime name.
    Runtime(String),
    /// The output of the op at this index.
    Stmt(u32),
    /// The join of split arms: same data, one identity downstream.
    Merge(Vec<(ValueId, Cond)>),
}

/// A predicate over the fact word; `Always` is the unsplit text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Cond {
    Always,
    Fact(u8),
    Not(Box<Cond>),
    And(Box<Cond>, Box<Cond>),
    Or(Box<Cond>, Box<Cond>),
}

impl Cond {
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
    pub fn not(a: Cond) -> Cond {
        Cond::Not(Box::new(a))
    }

    /// Evaluate against a fact word — the lowering's whole interface to
    /// conditions.
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

    #[must_use]
    pub fn referenced_bits(&self) -> Vec<u8> {
        let mut bits = Vec::new();
        self.bits_into(&mut bits);
        bits.sort_unstable();
        bits.dedup();
        bits
    }

    /// Collapse a condition that holds under every assignment of its bits —
    /// what an exhaustive split's merge reconstitutes.
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Op {
    /// A role point, or `cuda::`-prefixed plane symbol behind a plane gate.
    pub kernel: String,
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
    /// Weight params by canonical name, operand order.
    pub weights: Vec<String>,
    /// Scalar params in statement order, bits of the stated values.
    pub params: Vec<u64>,
    /// The cache row this statement reads or writes.
    pub cache: Option<String>,
    pub layer: Option<u32>,
    pub cond: Cond,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Seam {
    pub seam: String,
    pub values: Vec<ValueId>,
    pub layer: Option<u32>,
}

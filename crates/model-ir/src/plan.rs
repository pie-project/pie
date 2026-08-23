use serde::{Deserialize, Serialize};

pub type ValueId = u32;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Plan {
    pub name: String,
    pub plane: crate::kernels::Backend,

    pub facts: Vec<String>,

    pub params: Vec<Param>,
    pub caches: Vec<CacheRow>,
    pub values: Vec<ValueDef>,
    pub ops: Vec<Op>,
    pub seams: Vec<Seam>,
}

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
    Kv { name: String, row: Vec<u64> },

    State { name: String, slab: Vec<u64> },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValueDef {
    Runtime(String),

    Stmt(u32),

    Merge(Vec<(ValueId, Cond)>),
}

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
    pub kernel: String,
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,

    pub weights: Vec<String>,

    pub params: Vec<u64>,

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

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

/// One weight a text stated, as the plan holds it.
///
/// `shape` IS WHAT THIS RANK HOLDS. A tensor-parallel row states its own cut
/// widths — a `-tp2` text divides the dims its shard marks name and every
/// shape, statement param and cache row falls out narrower — so the walk
/// sizes a rank's rectangles by reading this column and nothing else. There
/// is no degree beside it: a plan that stated both the sharded width AND the
/// way it was cut would be saying one fact twice, and the two could disagree.
/// The degree a load needs is the ratio between the checkpoint's own extent
/// and this one, along [`Shard`]'s axis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Param {
    pub name: String,
    pub shape: Vec<u64>,
    pub shard: Shard,
    pub repr: String,
}

/// WHICH AXIS of a weight a rank cut runs along — never how far.
///
/// The extent is already in [`Param::shape`]; this says where the knife went,
/// which is what a load needs to pick this rank's slice out of the canonical
/// checkpoint tensor and what holds the text's own dims honest (a dim that
/// divides must be a dim some mark cuts, and a mark must cut a dim that
/// divides — `model/tests/a_rank_cut_is_the_shard_column.rs` is that check).
///
/// # The axis is a NUMBER, and it used to be a naming convention
///
/// `Columns`, `Rows` and `Packed(segments)` stood here, and which axis each
/// one meant was left to whoever read them. Two of the three are recoverable
/// — a column cut is the leading axis, a row cut the trailing one — and the
/// third is not: a packed axis was "the one the segments add up to", and
/// deepseek-v4's `experts_gate_up` is `[64, 2048, 2048]` with segments
/// `[1024, 1024]`, where BOTH the out axis and the hidden axis are 2048. A
/// rule that cannot answer that tensor is a rule a load cannot run.
/// So the mark answers with the axis, the builders in `model_dsl` compute it
/// from the shape they already hold, and the texts keep reading
/// `.columns()` / `.rows()` / `.packed(..)` / `.bank(..)`.
///
/// # `Experts` also stood here, and no text can fire it
///
/// It named the expert axis of a routed `[E, N, K]` bank: expert
/// parallelism, a rank holding `E/world` of the experts. Four texts stated it
/// and none could fire it. `moe.matmul_select` indexes the bank with the
/// ROUTER'S OWN id, which ranges over all `E`, and no statement in this tree
/// remaps a global expert to a local slot or masks the rows a rank does not
/// hold; and the `dist.all_reduce` those same texts state over the whole
/// feed-forward is the combine of a TENSOR cut, which sums partial rows —
/// over replicated expert output it would multiply the answer by the world
/// size. deepseek-v4 already said the cut its reduce closes; the other four
/// now say it too, and the variant no text states is gone.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Shard {
    /// No axis is cut: every rank holds the whole tensor.
    Replicated,
    /// This rank holds `1/world` of `axis`, and `segments` is how that axis
    /// is partitioned — EACH segment cut, so a rank's half of a `[gate | up]`
    /// bank is `[gate/2 | up/2]` and not the whole gate followed by nothing.
    /// A plain cut is one segment, and it is the whole axis by construction:
    /// the segments sum to `shape[axis]`.
    Cut { axis: u32, segments: Vec<u64> },
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

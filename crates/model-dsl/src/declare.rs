//! What a model declares beside its text: symbolic weights, cache rows,
//! shard cuts.

use std::marker::PhantomData;

use crate::axes::Dtype;

/// A cache row's name symbol; sharing is spelling the same name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CacheRef {
    pub name: String,
}

impl CacheRef {
    pub fn to(name: impl Into<String>) -> CacheRef {
        CacheRef { name: name.into() }
    }
}

pub use model_ir::plan::Shard;

/// YaRN rope numbers, stated once on the attention that uses them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Yarn {
    pub theta: f32,
    pub factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub attention_factor: f32,
    pub original_max_position: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KvRow {
    pub name: String,
    /// Extent appended per token; paged, discardable.
    pub row: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StateRow {
    pub name: String,
    /// One slab per request; folded in place.
    pub slab: Vec<u64>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KvSpec {
    pub rows: Vec<KvRow>,
}

impl KvSpec {
    #[must_use]
    pub fn new() -> KvSpec {
        KvSpec::default()
    }

    pub fn kv(&mut self, name: impl Into<String>, row: impl IntoIterator<Item = u64>) {
        self.rows.push(KvRow {
            name: name.into(),
            row: row.into_iter().collect(),
        });
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StateSpec {
    pub rows: Vec<StateRow>,
}

impl StateSpec {
    #[must_use]
    pub fn new() -> StateSpec {
        StateSpec::default()
    }

    pub fn state(&mut self, name: impl Into<String>, slab: impl IntoIterator<Item = u64>) {
        self.rows.push(StateRow {
            name: name.into(),
            slab: slab.into_iter().collect(),
        });
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HybridSpec {
    pub kv: Vec<KvRow>,
    pub state: Vec<StateRow>,
}

impl HybridSpec {
    #[must_use]
    pub fn new() -> HybridSpec {
        HybridSpec::default()
    }

    pub fn kv(&mut self, name: impl Into<String>, row: impl IntoIterator<Item = u64>) {
        self.kv.push(KvRow {
            name: name.into(),
            row: row.into_iter().collect(),
        });
    }

    pub fn state(&mut self, name: impl Into<String>, slab: impl IntoIterator<Item = u64>) {
        self.state.push(StateRow {
            name: name.into(),
            slab: slab.into_iter().collect(),
        });
    }
}

/// A symbolic weight: the canonical zt name, the canonical shape, the cut.
/// Constructors spell all three; nothing else may.
#[derive(Clone, Debug)]
pub struct Tensor<W: Dtype> {
    pub name: String,
    pub shape: Vec<u64>,
    pub shard: Shard,
    _axis: PhantomData<W>,
}

impl<W: Dtype> Tensor<W> {
    #[must_use]
    pub fn sym(name: impl Into<String>, shape: impl IntoIterator<Item = u64>) -> Tensor<W> {
        Tensor {
            name: name.into(),
            shape: shape.into_iter().collect(),
            shard: Shard::Replicated,
            _axis: PhantomData,
        }
    }

    #[must_use]
    pub fn columns(mut self) -> Tensor<W> {
        self.shard = Shard::Columns;
        self
    }

    #[must_use]
    pub fn rows(mut self) -> Tensor<W> {
        self.shard = Shard::Rows;
        self
    }

    #[must_use]
    pub fn packed(mut self, segments: impl IntoIterator<Item = u64>) -> Tensor<W> {
        self.shard = Shard::Packed(segments.into_iter().collect());
        self
    }

    #[must_use]
    pub fn experts(mut self) -> Tensor<W> {
        self.shard = Shard::Experts;
        self
    }
}

#[derive(Clone, Debug)]
pub struct Norm<W: Dtype> {
    pub weight: Tensor<W>,
    pub eps: f32,
}

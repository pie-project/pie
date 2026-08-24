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

    /// A COLUMN-PARALLEL cut: the OUT axis, which is the leading one on
    /// every weight this catalog states — `[out, in]` for a projection,
    /// `[out]` for the bias or the per-head scalar beside it.
    #[must_use]
    pub fn columns(self) -> Tensor<W> {
        self.cut(0, None)
    }

    /// A ROW-PARALLEL cut: the IN axis, which is the trailing one — axis 1
    /// of an `[out, in]` projection, axis 2 of an `[experts, out, in]` bank.
    /// The statement that carries it produces PARTIAL rows, which is what the
    /// `dist.all_reduce` after it sums.
    #[must_use]
    pub fn rows(self) -> Tensor<W> {
        // `wrapping_sub` on a scalar lands on an axis no shape has, which
        // `cut` refuses with the tensor's name — the answer a `[]` weight
        // deserves, and one underflow message fewer.
        let last = self.shape.len().wrapping_sub(1);
        self.cut(last, None)
    }

    /// A column-parallel cut of an out axis that is a CONCATENATION —
    /// `[gate | up]`, `[q | k | v]` — where every segment is cut, so that a
    /// rank holds half of each and not the whole of the first.
    #[must_use]
    pub fn packed(self, segments: impl IntoIterator<Item = u64>) -> Tensor<W> {
        self.cut(0, Some(segments.into_iter().collect()))
    }

    /// An expert bank's out axis, which is axis 1: axis 0 is the expert fan,
    /// and a TENSOR cut does not touch it — every rank scores and holds every
    /// expert, at a share of each one's width. Takes the segments because
    /// every routed bank in this catalog that is cut here is a `[gate | up]`
    /// pair; a bank with one segment says so with one.
    #[must_use]
    pub fn bank(self, segments: impl IntoIterator<Item = u64>) -> Tensor<W> {
        self.cut(1, Some(segments.into_iter().collect()))
    }

    /// The one place a mark becomes an axis, and the only place that checks
    /// it: a cut names an axis this tensor has, and a partition of that axis
    /// covers it exactly.
    fn cut(mut self, axis: usize, segments: Option<Vec<u64>>) -> Tensor<W> {
        let extent = *self.shape.get(axis).unwrap_or_else(|| {
            panic!(
                "`{}` is {:?} and a cut names axis {axis}",
                self.name, self.shape,
            )
        });
        let segments = segments.unwrap_or_else(|| vec![extent]);
        let whole: u64 = segments.iter().sum();
        assert_eq!(
            whole, extent,
            "`{}`: the segments of axis {axis} sum to {whole} and the axis is {extent}",
            self.name,
        );
        self.shard = Shard::Cut {
            axis: u32::try_from(axis).expect("an axis inside a shape"),
            segments,
        };
        self
    }
}

/// One rank's share of a width a text's shard marks cut `tp` ways.
///
/// THE ONLY PLACE A DEGREE IS SPENT. A tensor-parallel catalog row is the
/// same text at a different `TP`, and what `TP` buys is this: the dims the
/// marks name come out divided, and every shape, every statement param and
/// every cache row built from them is this rank's own. Nothing downstream
/// ever sees the degree again — the plan states sharded widths and the walk
/// sizes them the way it sizes any others, with no shard rule of its own.
///
/// `what` is the dim's name and it is in the panic for the same reason the
/// catalog's other arithmetic is checked at trace time: a row whose heads do
/// not divide is a deployment nobody can serve, and finding out at the first
/// launch of the first fire is finding out in the worst place.
///
/// # Panics
///
/// `whole` does not divide `tp` ways, or `tp` is zero.
#[must_use]
pub fn per_rank(what: &str, whole: u32, tp: usize) -> u32 {
    let tp = u32::try_from(tp).expect("a world no u32 holds");
    assert!(tp > 0, "a {tp}-way cut of `{what}`");
    assert!(
        whole.is_multiple_of(tp),
        "`{what}` is {whole} and does not cut {tp} ways",
    );
    whole / tp
}

#[derive(Clone, Debug)]
pub struct Norm<W: Dtype> {
    pub weight: Tensor<W>,
    pub eps: f32,
}

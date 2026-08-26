//! Weight declarations without generics. The old `Tensor<W: Dtype>` carried
//! its representation as a phantom type; here `Repr` is a plain field — one
//! `Weight` struct, no monomorphized model trees (design §5).

use new_model_ir::{Dtype, Repr, Shard};

/// A declaration-site name for a cache space; `Input::kv`/`state` resolve it
/// against what the model's `caches()` actually declared.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CacheRef {
    pub name: String,
}

impl CacheRef {
    pub fn to(name: impl Into<String>) -> CacheRef {
        CacheRef { name: name.into() }
    }

    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// Yarn rope scaling, as declaration data — the wrapper spreads it into the
/// `Rope::Yarn` fields.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Yarn {
    pub theta: f32,
    pub factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub attention_factor: f32,
    pub original_max_position: u32,
}

/// One logical weight: name, logical shape, on-device representation, and how
/// it is laid out across ranks. The recorder interns it into `Plan::params` —
/// one param per stored plane — the first time a wrapper touches it.
#[derive(Clone, Debug)]
pub struct Weight {
    pub name: String,
    pub shape: Vec<u64>,
    pub repr: Repr,
    pub shard: Shard,
}

impl Weight {
    #[must_use]
    pub fn sym(name: impl Into<String>, shape: impl IntoIterator<Item = u64>, repr: Repr) -> Weight {
        Weight {
            name: name.into(),
            shape: shape.into_iter().collect(),
            repr,
            shard: Shard::Replicated,
        }
    }

    /// Cut along the output axis: each rank holds a column block.
    #[must_use]
    pub fn columns(self) -> Weight {
        self.cut(0, None)
    }

    /// Cut along the reduction axis: each rank holds a row block and the
    /// matmul's partial sums meet in a collective.
    #[must_use]
    pub fn rows(self) -> Weight {
        let last = self.shape.len().wrapping_sub(1);
        self.cut(last, None)
    }

    /// Cut axis 0 at the stated seams — a fused projection whose halves must
    /// not straddle ranks.
    #[must_use]
    pub fn packed(self, segments: impl IntoIterator<Item = u64>) -> Weight {
        self.cut(0, Some(segments.into_iter().collect()))
    }

    /// Cut axis 1 at the stated seams — the per-expert form of `packed`.
    #[must_use]
    pub fn bank(self, segments: impl IntoIterator<Item = u64>) -> Weight {
        self.cut(1, Some(segments.into_iter().collect()))
    }

    fn cut(mut self, axis: usize, segments: Option<Vec<u64>>) -> Weight {
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

    #[must_use]
    pub fn dim(&self, i: usize) -> u64 {
        *self.shape.get(i).unwrap_or_else(|| {
            panic!("`{}` is {:?} and has no axis {i}", self.name, self.shape)
        })
    }

    /// The dtype activations see through this weight. Mxfp4 banks store codes
    /// and compute in bf16; e8m0 exists only as the scales plane a bank
    /// interns beside itself, never as a weight an author declares.
    #[must_use]
    pub fn dtype(&self) -> Dtype {
        match self.repr {
            Repr::Bf16 => Dtype::Bf16,
            Repr::F16 => Dtype::F16,
            Repr::F32 => Dtype::F32,
            Repr::Mxfp4 => Dtype::Bf16,
            Repr::E8m0 => panic!(
                "`{}`: e8m0 is a scales plane, not a weight an author declares",
                self.name,
            ),
        }
    }

    /// The stored planes behind one logical weight. Every repr stores itself
    /// verbatim except mxfp4, which packs each 32-code block into 16 bytes
    /// and interns an e8m0 scale-per-block companion under `.scales`.
    pub(crate) fn planes(&self) -> Vec<BankPlane> {
        match self.repr {
            Repr::Mxfp4 => {
                let (&k, lead) = self
                    .shape
                    .split_last()
                    .expect("an mxfp4 bank's logical shape ends in its contracted axis");
                assert!(
                    k % 32 == 0,
                    "an mxfp4 bank contracts over {k}, which is not a whole number of \
                     32-code blocks"
                );
                let groups = k / 32;
                let mut codes = lead.to_vec();
                codes.extend([groups, 16]);
                let mut scales = lead.to_vec();
                scales.push(groups);
                vec![
                    BankPlane {
                        suffix: "",
                        shape: codes,
                        repr: Repr::Mxfp4,
                    },
                    BankPlane {
                        suffix: ".scales",
                        shape: scales,
                        repr: Repr::E8m0,
                    },
                ]
            }
            _ => vec![BankPlane {
                suffix: "",
                shape: self.shape.clone(),
                repr: self.repr,
            }],
        }
    }
}

/// One stored plane of a weight: what actually lands in `Plan::params`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BankPlane {
    pub suffix: &'static str,
    pub shape: Vec<u64>,
    pub repr: Repr,
}

/// A normalization's weight and epsilon travel together everywhere.
#[derive(Clone, Debug)]
pub struct Norm {
    pub weight: Weight,
    pub eps: f32,
}

/// The per-rank extent of a head count or width that tensor parallelism cuts.
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

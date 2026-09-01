//! Weight declarations without generics. The old `Tensor<W: Dtype>` carried
//! its representation as a phantom type; here `Dtype` is a plain field — one
//! `Weight` struct, no monomorphized model trees (design §5).

use model_ir::{Dtype, ParamSource, Shard};

/// One logical weight: name, logical shape, on-device representation, how it
/// is laid out across ranks, and where its bytes come from. The recorder
/// interns it into `Trace::params` — one param per stored plane — the first
/// time a wrapper touches it.
#[derive(Clone, Debug)]
pub struct Weight {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: Dtype,
    pub shard: Shard,
    /// The checkpoint's, unless [`registered`](Weight::registered) says
    /// otherwise.
    pub source: ParamSource,
}

impl Weight {
    #[must_use]
    pub fn sym(
        name: impl Into<String>,
        shape: impl IntoIterator<Item = u64>,
        dtype: Dtype,
    ) -> Weight {
        Weight {
            name: name.into(),
            shape: shape.into_iter().collect(),
            dtype,
            shard: Shard::Replicated,
            source: ParamSource::Checkpoint,
        }
    }

    /// **AN ADAPTER BANK: THE BUDGET IS THE SHAPE** (design §8).
    ///
    /// The checkpoint does not publish this plane. It is reserved at load from
    /// the shape declared here, zeroed — and a zeroed low-rank `A` is the
    /// identity correction, so every unwritten row of the bank is the base
    /// model — and filled a row at a time by `Engine::register_adapter`, which
    /// is a pool write and a table row and NOT a recapture: the graph key is
    /// the fire's composition and a bank's contents are not in it.
    ///
    /// Capacity is stated here rather than passed in at load because it is a
    /// SHAPE, and shapes are the model text's. `model_compiler::compile`
    /// refuses a `Budget::max_adapters` this bank cannot seat, which is the
    /// one place the deployment's number and the model's meet.
    #[must_use]
    pub fn registered(mut self) -> Weight {
        self.source = ParamSource::Registered;
        self
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
        *self
            .shape
            .get(i)
            .unwrap_or_else(|| panic!("`{}` is {:?} and has no axis {i}", self.name, self.shape))
    }

    /// The dtype activations see through this weight. Mxfp4 and U4g64 banks
    /// store codes and compute in bf16; the rest of `Dtype` names kv-cache
    /// schemes, index layouts, and the companion planes a bank interns beside
    /// itself — never a weight an author declares.
    #[must_use]
    pub fn compute_dtype(&self) -> Dtype {
        compute_dtype(self.dtype).unwrap_or_else(|| {
            panic!(
                "`{}`: not a weight representation an author declares",
                self.name
            )
        })
    }

    /// The stored planes behind one logical weight. Every dtype stores itself
    /// verbatim except mxfp4, which packs each 32-code block into 16 bytes
    /// and interns an e8m0 scale-per-block companion under `.scales`.
    pub(crate) fn planes(&self) -> Vec<BankPlane> {
        match self.dtype {
            // MLX's affine U4: the codes keep the logical shape and change
            // only their width — four bits an element, which is exactly the
            // eight-codes-to-a-u32-word the checkpoint ships — and TWO
            // companions of one group count each, a scale and an offset,
            // because `code * scale + bias` needs both to be read at all.
            // Publishing the scales and withholding the biases would land a
            // bank that dequantizes to the right spread around the wrong
            // centre, silently.
            // **BOTH AFFINE WIDTHS TAKE THIS ARM, AND THE WIDTH IS THE ONLY
            // THING THAT DIFFERS.** `U8g64` is `U4g64`'s scheme at eight bits
            // a code — `mlx_lm`'s `quant_predicate` raises a MoE router gate
            // to it while the stack around it stays at four (see
            // `dtype::Dtype::U8g64`) — and everything below is a fact about
            // the SCHEME rather than the width or the group: the codes keep
            // the logical shape and both companions are one bf16 per group.
            // So the plane carries `self.dtype` and the arm is shared; a
            // width- or group-specific arm here would be copies of one
            // paragraph with one number changed — and the group IS that one
            // number (`U4g32` is the 160-wide-row reading, see its doc).
            Dtype::U4g64 | Dtype::U8g64 | Dtype::U4g32 => {
                let group = match self.dtype {
                    Dtype::U4g32 => 32,
                    _ => 64,
                };
                let (&k, lead) = self
                    .shape
                    .split_last()
                    .expect("an affine bank's logical shape ends in its contracted axis");
                assert!(
                    k % group == 0,
                    "`{}` is an affine bank contracting over {k}, which is not a whole \
                     number of {group}-code groups",
                    self.name,
                );
                let mut factors = lead.to_vec();
                factors.push(k / group);
                vec![
                    BankPlane {
                        suffix: "",
                        shape: self.shape.clone(),
                        dtype: self.dtype,
                    },
                    BankPlane {
                        suffix: SCALES,
                        shape: factors.clone(),
                        dtype: Dtype::Bf16,
                    },
                    BankPlane {
                        suffix: BIASES,
                        shape: factors,
                        dtype: Dtype::Bf16,
                    },
                ]
            }
            Dtype::Mxfp4 => {
                let (&k, lead) = self
                    .shape
                    .split_last()
                    .expect("an mxfp4 bank's logical shape ends in its contracted axis");
                assert!(
                    k % 32 == 0,
                    "`{}` is an mxfp4 bank contracting over {k}, which is not a whole \
                     number of 32-code blocks",
                    self.name,
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
                        dtype: Dtype::Mxfp4,
                    },
                    BankPlane {
                        suffix: SCALES,
                        shape: scales,
                        dtype: Dtype::E8m0,
                    },
                ]
            }
            // Served formats no model TEXT declares yet: they reach the
            // engine through an import plan, not through a family's weight
            // declaration, so a declaration naming one has no plane story to
            // intern. The day a text declares one, its author lands here and
            // writes that story — which is this panic's whole job.
            Dtype::Nvfp4
            | Dtype::U2g16k
            | Dtype::I3g16k
            | Dtype::U4g32k
            | Dtype::U5g32k
            | Dtype::I6g16k
            | Dtype::E4m3row
            | Dtype::E4m3tile128 => panic!(
                "`{}`: {:?} is served but not yet a weight representation a \
                 model text declares",
                self.name, self.dtype
            ),
            Dtype::Bf16
            | Dtype::F16
            | Dtype::F32
            | Dtype::I32
            | Dtype::U32
            | Dtype::U8
            | Dtype::I8
            | Dtype::E4m3
            | Dtype::E2m1
            | Dtype::E8m0
            | Dtype::E5m2
            | Dtype::I64
            | Dtype::I16
            | Dtype::U64
            | Dtype::U16
            | Dtype::Bool => vec![BankPlane {
                suffix: "",
                shape: self.shape.clone(),
                dtype: self.dtype,
            }],
        }
    }
}

/// One stored plane of a weight: what actually lands in `Trace::params`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BankPlane {
    pub suffix: &'static str,
    pub shape: Vec<u64>,
    pub dtype: Dtype,
}

/// What an mxfp4 bank's e8m0 scales plane is called: the bank's own name and
/// this suffix.
///
/// ONE HOME FOR ONE STRING. Four places write it — the plane `planes` interns,
/// the load contract that declares where the bytes come from, the family
/// import that names the stored tensor, and the loader's runtime-quant encode
/// that publishes the companion it computed — and they must agree exactly or
/// the loader lands a plane nothing reads under a name no contract chose. So
/// they all say it here.
const SCALES: &str = ".scales";

/// What an affine bank's zero-point plane is called. Same argument as
/// [`SCALES`], and the same four writers: an offset nothing can find is an
/// offset nothing subtracts, and a bank read without its offsets is not a
/// worse bank but a wrong one.
const BIASES: &str = ".biases";

#[must_use]
pub fn scales_name(of: &str) -> String {
    format!("{of}{SCALES}")
}

#[must_use]
pub fn biases_name(of: &str) -> String {
    format!("{of}{BIASES}")
}

/// The dtype a weight of this representation is MULTIPLIED as, or `None` for a
/// dtype no weight is declared in.
///
/// Asked of the dtype rather than of the weight so a model text can state a
/// bank's neighbours without owning a bank: a norm beside an U4g64 projection
/// is bf16 BECAUSE the projection computes in bf16, and saying so here is what
/// keeps a quantized SKU from being a second family text whose norms happen to
/// have been remembered. [`Weight::compute_dtype`] is this function plus the
/// weight's name in the panic.
#[must_use]
pub fn compute_dtype(dtype: Dtype) -> Option<Dtype> {
    match dtype {
        Dtype::Bf16 => Some(Dtype::Bf16),
        Dtype::F16 => Some(Dtype::F16),
        Dtype::F32 => Some(Dtype::F32),
        Dtype::Mxfp4 | Dtype::U4g64 | Dtype::U8g64 | Dtype::U4g32 => Some(Dtype::Bf16),
        // Served formats no model text declares (yet): a plan lands them,
        // an author does not, so they have no compute story here either.
        Dtype::Nvfp4
        | Dtype::U2g16k
        | Dtype::I3g16k
        | Dtype::U4g32k
        | Dtype::U5g32k
        | Dtype::I6g16k
        | Dtype::E4m3row
        | Dtype::E4m3tile128 => None,
        Dtype::I32
        | Dtype::U32
        | Dtype::U8
        | Dtype::I8
        | Dtype::E4m3
        | Dtype::E2m1
        | Dtype::E8m0
        | Dtype::E5m2
        | Dtype::I64
        | Dtype::I16
        | Dtype::U64
        | Dtype::U16
        | Dtype::Bool => None,
    }
}

//! Weight declarations without generics: `Dtype` is a plain field, so there
//! is one `Weight` struct and no monomorphized model trees.

use model_ir::{Dtype, Param, ParamSource, Platform, Shard, TILED_BAND, TILED_STEP};

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

    /// An adapter bank: the checkpoint does not publish this plane. It is
    /// reserved at load from the shape declared here, zeroed (a zeroed
    /// low-rank `A` is the identity correction), and filled a row at a time
    /// by `Engine::register_adapter` — a pool write, not a recapture, since
    /// the graph key is the fire's composition and a bank's contents aren't
    /// in it. Capacity lives here because it's a shape, and shapes are the
    /// model text's; `model_compiler::compile` refuses a `Budget::max_adapters`
    /// this bank cannot seat.
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
            // Affine schemes (U4g64, U8g64, U4g32, U2g32/64/128): codes keep
            // the logical shape; two bf16-per-group companions (scale,
            // offset) since `code * scale + bias` needs both. Width and
            // group are the only things that differ between these dtypes,
            // so they share one arm.
            Dtype::U4g64
            | Dtype::U8g64
            | Dtype::U4g32
            | Dtype::U2g32
            | Dtype::U2g64
            | Dtype::U2g128 => {
                let group = match self.dtype {
                    Dtype::U4g32 | Dtype::U2g32 => 32,
                    Dtype::U4g64 | Dtype::U8g64 | Dtype::U2g64 => 64,
                    Dtype::U2g128 => 128,
                    other => unreachable!("`{other:?}` is not an affine row this arm claims"),
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
            // The tiled affine row: same scheme, same group of 64, same two
            // bf16 companions, but codes and factors have gone through
            // `kernels_cuda::linear::tiled`'s relabelling, which pads the
            // output column axis up to a whole 16-column mma band (padded
            // columns decode to a zero weight). A different rectangle from
            // the row-major weight, hence its own arm. Also a projection,
            // not a bank — a routed expert bank has an axis it does not
            // carve, and a repack of one is refused here.
            Dtype::U4g64tiled => {
                let [n, k] = self.shape[..] else {
                    panic!(
                        "`{}` is a tiled affine projection declared {:?}; the tiled point \
                         reads a two-dimensional weight",
                        self.name, self.shape,
                    )
                };
                // Band and step are the layout's own constants; `checkpoint`'s
                // repack checks a declared target against these same two numbers.
                const GROUP: u64 = 64;
                let band = u64::from(TILED_BAND);
                let step = u64::from(TILED_STEP);
                assert!(
                    k % GROUP == 0,
                    "`{}` is an affine bank contracting over {k}, which is not a whole \
                     number of {GROUP}-code groups",
                    self.name,
                );
                assert!(
                    k % step == 0,
                    "`{}` contracts over {k}, which is not a whole number of the \
                     {step}-wide steps the tiled point walks",
                    self.name,
                );
                let rows = n.div_ceil(band) * band;
                let factors = vec![rows, k / GROUP];
                vec![
                    BankPlane {
                        suffix: "",
                        shape: vec![rows, k],
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
            // Served formats no model text declares yet: they reach the
            // engine through an import plan, not a weight declaration.
            Dtype::Nvfp4 | Dtype::E4m3row | Dtype::E4m3tile128 => panic!(
                "`{}`: {} is served but not yet a weight representation a \
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
            // Stored super-block families (GGUF K-quant etc): scales live
            // inside the super-block, so there's no companion to intern —
            // one rectangle of `n` rows by `Dtype::row_bytes(k)` bytes,
            // exactly what `linear::kquant` reads. The text still declares
            // the logical `[n, k]`; this is where it folds into the stored
            // container shape.
            Dtype::U2g16k
            | Dtype::I3g16k
            | Dtype::U4g32k
            | Dtype::U5g32k
            | Dtype::I6g16k => {
                let (&k, lead) = self
                    .shape
                    .split_last()
                    .expect("a quantized bank's logical shape ends in its contracted axis");
                let k = u32::try_from(k).unwrap_or_else(|_| {
                    panic!("`{}` contracts over {k}, which is no row width", self.name)
                });
                let row = self.dtype.row_bytes(k).unwrap_or_else(|| {
                    panic!(
                        "`{}` is `{}` contracting over {k}, which is not a whole \
                         number of the format's {:?} — a row cut mid-group owns a \
                         factor it does not fill",
                        self.name,
                        self.dtype,
                        self.dtype.quantum(),
                    )
                });
                let mut bytes = lead.to_vec();
                bytes.push(row);
                vec![BankPlane {
                    suffix: "",
                    shape: bytes,
                    dtype: self.dtype,
                }]
            }
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

/// What an mxfp4 bank's e8m0 scales plane is called. One home for one
/// string: four places write it and must agree exactly, or the loader lands
/// a plane nothing reads under a name no contract chose.
const SCALES: &str = ".scales";

/// What an affine bank's zero-point plane is called. Same argument as
/// [`SCALES`].
const BIASES: &str = ".biases";

impl Weight {
    /// This weight as `platform` lands it: a placed dtype resolves to the
    /// canonical one where the platform's kernels do not read the placement
    /// ([`Platform::placement`]).
    #[must_use]
    pub fn placed(&self, platform: Platform) -> Weight {
        Weight {
            dtype: platform.placement(self.dtype),
            ..self.clone()
        }
    }
}

impl Weight {
    /// The weight a stored plane of a trace was interned from: the inverse of
    /// [`planes`](Weight::planes) for the codes plane, up to the row padding
    /// a tiled bank carries. A companion plane has no weight of its own.
    #[must_use]
    pub fn of_plane(p: &Param) -> Weight {
        let shape: Vec<u64> = match p.dtype {
            Dtype::Mxfp4 => {
                let [lead @ .., groups, sixteen] = &p.shape[..] else {
                    panic!("`{}` is an mxfp4 codes plane stored {:?}", p.name, p.shape)
                };
                assert_eq!(*sixteen, 16, "`{}` is an mxfp4 codes plane stored {:?}", p.name, p.shape);
                lead.iter().copied().chain([groups * 32]).collect()
            }
            _ => p.shape.clone(),
        };
        let shard = match &p.shard {
            Shard::Replicated => Shard::Replicated,
            Shard::Cut { axis, segments } => {
                let at = *axis as usize;
                let block = shape[at] / p.shape[at];
                Shard::Cut {
                    axis: *axis,
                    segments: segments.iter().map(|s| s * block).collect(),
                }
            }
        };
        Weight {
            name: p.name.clone(),
            shape,
            dtype: p.dtype,
            shard,
            source: p.source,
        }
    }
}

#[must_use]
pub fn scales_name(of: &str) -> String {
    format!("{of}{SCALES}")
}

#[must_use]
pub fn biases_name(of: &str) -> String {
    format!("{of}{BIASES}")
}

/// The dtype a weight of this representation is multiplied as, or `None` for
/// a dtype no weight is declared in. Asked of the dtype rather than the
/// weight, so a model text can state a bank's neighbours without owning a
/// bank: a norm beside a U4g64 projection is bf16 because the projection
/// computes in bf16. [`Weight::compute_dtype`] is this plus the weight's
/// name in the panic.
#[must_use]
pub fn compute_dtype(dtype: Dtype) -> Option<Dtype> {
    match dtype {
        Dtype::Bf16 => Some(Dtype::Bf16),
        Dtype::F16 => Some(Dtype::F16),
        Dtype::F32 => Some(Dtype::F32),
        // A quant term says how a weight is stored, nothing about what it
        // multiplies as: every weight-only quant point decodes inside the
        // dot and accumulates against a bf16 activation.
        Dtype::Mxfp4
        | Dtype::Nvfp4
        | Dtype::U4g64
        | Dtype::U8g64
        | Dtype::U4g32
        | Dtype::U2g32
        | Dtype::U2g64
        | Dtype::U2g128
        | Dtype::U4g64tiled
        | Dtype::U2g16k
        | Dtype::I3g16k
        | Dtype::U4g32k
        | Dtype::U5g32k
        | Dtype::I6g16k
        | Dtype::E4m3row
        | Dtype::E4m3tile128 => Some(Dtype::Bf16),
        // A lookup table computes as itself: `I64` is `ffn.gate.tid2eid`,
        // DeepSeek-V4-Flash's token-id -> expert-id table, which
        // `linear.moe_hash_route` gathers rather than dequantizes.
        Dtype::I64 => Some(Dtype::I64),
        Dtype::I32
        | Dtype::U32
        | Dtype::U8
        | Dtype::I8
        | Dtype::E4m3
        | Dtype::E2m1
        | Dtype::E8m0
        | Dtype::E5m2
        | Dtype::I16
        | Dtype::U64
        | Dtype::U16
        | Dtype::Bool => None,
    }
}

use poem_ir::{
    ActQuant, BIASES, Dtype, GLOBAL_SCALES, INPUT_SCALES, Param, ParamLayout, ParamPlacement,
    ParamSource, Platform, SCALES, Shard, TILED_BAND, TILED_STEP,
};

#[derive(Clone, Debug)]
pub struct Weight {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: Dtype,
    /// The precision a product quantizes its activations to before reading
    /// this weight, or `None` for one that reads them as they are.
    pub act: Option<ActQuant>,
    pub shard: Shard,
    pub source: ParamSource,
    pub layout: ParamLayout,
    /// Where the weight's leading plane (its codes) may be served from; its
    /// companions (a scale) stay on the device.
    pub placement: ParamPlacement,
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
            act: None,
            shard: Shard::Replicated,
            source: ParamSource::Checkpoint,
            layout: ParamLayout::Natural,
            placement: ParamPlacement::Device,
        }
    }

    /// A weight stored at `precision`'s dtype, its products quantizing their
    /// activations as it says.
    #[must_use]
    pub fn at(
        name: impl Into<String>,
        shape: impl IntoIterator<Item = u64>,
        precision: Precision,
    ) -> Weight {
        Weight {
            act: precision.act,
            ..Weight::sym(name, shape, precision.dtype)
        }
    }

    #[must_use]
    pub fn conv_taps_major(mut self, c_in: u32, taps: u32) -> Weight {
        assert!(
            self.shape.len() == 2 && self.shape[1] == u64::from(c_in) * u64::from(taps),
            "`{}` is {:?}; a conv weight is `[C_out, C_in*taps]` = [.., {}]",
            self.name,
            self.shape,
            u64::from(c_in) * u64::from(taps)
        );
        self.layout = ParamLayout::ConvTapsMajor { c_in, taps };
        self
    }

    /// The weight is read only by row gathers, so a card whose profile asks
    /// may serve it from pinned host memory the device reads through.
    #[must_use]
    pub fn host(mut self) -> Weight {
        self.placement = ParamPlacement::Host;
        self
    }

    #[must_use]
    pub fn registered(mut self) -> Weight {
        self.source = ParamSource::Registered;
        self
    }

    #[must_use]
    pub fn columns(self) -> Weight {
        self.cut(0, None)
    }

    #[must_use]
    pub fn rows(self) -> Weight {
        let last = self.shape.len().wrapping_sub(1);
        self.cut(last, None)
    }

    #[must_use]
    pub fn packed(self, segments: impl IntoIterator<Item = u64>) -> Weight {
        self.cut(0, Some(segments.into_iter().collect()))
    }

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
            heads: Vec::new(),
        };
        self
    }

    /// The heads each segment of the cut holds, so ranks outnumbering a
    /// segment's heads each hold a whole head rather than part of one.
    #[must_use]
    pub fn heads(mut self, heads: impl IntoIterator<Item = u64>) -> Weight {
        let Shard::Cut {
            segments,
            heads: held,
            ..
        } = &mut self.shard
        else {
            panic!("`{}` is not cut, so its heads split nothing", self.name);
        };
        let heads: Vec<u64> = heads.into_iter().collect();
        assert_eq!(
            heads.len(),
            segments.len(),
            "`{}` has {} segments and {} head counts",
            self.name,
            segments.len(),
            heads.len(),
        );
        for (segment, h) in segments.iter().zip(&heads) {
            assert!(
                *h > 0 && segment.is_multiple_of(*h),
                "`{}`: a segment of {segment} does not hold {h} heads",
                self.name,
            );
        }
        *held = heads;
        self
    }

    #[must_use]
    pub fn dim(&self, i: usize) -> u64 {
        *self
            .shape
            .get(i)
            .unwrap_or_else(|| panic!("`{}` is {:?} and has no axis {i}", self.name, self.shape))
    }

    #[must_use]
    pub fn compute_dtype(&self) -> Dtype {
        compute_dtype(self.dtype).unwrap_or_else(|| {
            panic!(
                "`{}`: not a weight representation an author declares",
                self.name
            )
        })
    }

    pub(crate) fn planes(&self) -> Vec<BankPlane> {
        match self.dtype {
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
            Dtype::U4g64tiled => {
                let [n, k] = self.shape[..] else {
                    panic!(
                        "`{}` is a tiled affine projection declared {:?}; the tiled point \
                         reads a two-dimensional weight",
                        self.name, self.shape,
                    )
                };
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
            Dtype::Nvfp4 => {
                let (&k, lead) = self
                    .shape
                    .split_last()
                    .expect("an nvfp4 bank's logical shape ends in its contracted axis");
                assert!(
                    k % 16 == 0,
                    "`{}` is an nvfp4 bank contracting over {k}, which is not a whole \
                     number of 16-code groups",
                    self.name,
                );
                let mut scales = lead.to_vec();
                scales.push(k / 16);
                let slabs = lead.split_last().map_or(&[][..], |(_, slabs)| slabs);
                vec![
                    BankPlane {
                        suffix: "",
                        shape: self.shape.clone(),
                        dtype: Dtype::Nvfp4,
                    },
                    BankPlane {
                        suffix: SCALES,
                        shape: scales,
                        dtype: Dtype::E4m3,
                    },
                    BankPlane {
                        suffix: GLOBAL_SCALES,
                        shape: lead.to_vec(),
                        dtype: Dtype::F32,
                    },
                    BankPlane {
                        suffix: INPUT_SCALES,
                        shape: slabs.to_vec(),
                        dtype: Dtype::F32,
                    },
                ]
            }
            Dtype::E4m3tensor => vec![
                BankPlane {
                    suffix: "",
                    shape: self.shape.clone(),
                    dtype: Dtype::E4m3tensor,
                },
                BankPlane {
                    suffix: SCALES,
                    shape: Vec::new(),
                    dtype: Dtype::F32,
                },
            ],
            // Its codes and its block exponents are planes of their own, raw
            // e4m3 and e8m0, which a W8A8 product reads beside each other.
            Dtype::Fp8b128 | Dtype::Fp8b32 | Dtype::Mxfp8 => {
                let (br, bc) = self.dtype.fp8_blocks().unwrap_or((128, 128));
                let (br, bc) = (u64::from(br), u64::from(bc));
                let [n, k] = self.shape[..] else {
                    panic!(
                        "`{}` is a block-fp8 weight declared {:?}; its {br}x{bc} blocks \
                         scale a two-dimensional projection",
                        self.name, self.shape,
                    )
                };
                vec![
                    BankPlane {
                        suffix: "",
                        shape: vec![n, k],
                        dtype: Dtype::E4m3,
                    },
                    BankPlane {
                        suffix: SCALES,
                        shape: vec![n.div_ceil(br), k.div_ceil(bc)],
                        dtype: Dtype::E8m0,
                    },
                ]
            }
            // A block-FP8 release's weights as released: e4m3 codes, and one
            // f32 factor a 128x128 block (`w = code * factor`) of each
            // projection (each expert's, in a bank), raw planes side by side.
            Dtype::E4m3tile128 => {
                let (&k, lead) = self
                    .shape
                    .split_last()
                    .expect("a block-FP8 weight's logical shape ends in its contracted axis");
                let (&n, slabs) = lead
                    .split_last()
                    .expect("a block-FP8 weight is at least a projection");
                let mut factors = slabs.to_vec();
                factors.extend([n.div_ceil(128), k.div_ceil(128)]);
                vec![
                    BankPlane {
                        suffix: "",
                        shape: self.shape.clone(),
                        dtype: Dtype::E4m3,
                    },
                    BankPlane {
                        suffix: SCALES,
                        shape: factors,
                        dtype: Dtype::F32,
                    },
                ]
            }
            Dtype::E4m3row | Dtype::KvU4 => panic!(
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
            Dtype::U2g16k | Dtype::I3g16k | Dtype::U4g32k | Dtype::U5g32k | Dtype::I6g16k => {
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
            Dtype::Ptq1_0 => {
                // Unlike the k-quant block families above, PTQ1_0 keeps its
                // LOGICAL shape in the trace plane, exactly as the affine codes
                // plane does. The engine's `plane_bytes`/`places` reserve
                // `rows * row_bytes(k)` FROM the logical width, and the matmul
                // contracts over the logical `k` — so the plane must carry `k`,
                // not the packed byte count (packing it here would both double the
                // reserved bytes and shrink the contraction to the byte width).
                // Validate the contracted axis is a whole number of 128-wide
                // ternary blocks (a row cut mid-block owns a partial 28-byte block
                // it cannot fill).
                let (&k, _) = self
                    .shape
                    .split_last()
                    .expect("a PTQ1_0 bank's logical shape ends in its contracted axis");
                let k = u32::try_from(k).unwrap_or_else(|_| {
                    panic!("`{}` contracts over {k}, which is no row width", self.name)
                });
                assert!(
                    self.dtype.row_bytes(k).is_some(),
                    "`{}` is `{}` contracting over {k}, which is not a whole number of \
                     the format's {:?} — a row cut mid-block owns a factor it cannot fill",
                    self.name,
                    self.dtype,
                    self.dtype.quantum(),
                );
                vec![BankPlane {
                    suffix: "",
                    shape: self.shape.clone(),
                    dtype: self.dtype,
                }]
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BankPlane {
    pub suffix: &'static str,
    pub shape: Vec<u64>,
    pub dtype: Dtype,
}

impl Weight {
    #[must_use]
    pub fn placed(&self, platform: Platform) -> Weight {
        Weight {
            dtype: platform.placement(self.dtype),
            ..self.clone()
        }
    }
}

impl Weight {
    #[must_use]
    pub fn of_plane(p: &Param) -> Weight {
        let shape: Vec<u64> = match p.dtype {
            Dtype::Mxfp4 => {
                let [lead @ .., groups, sixteen] = &p.shape[..] else {
                    panic!("`{}` is an mxfp4 codes plane stored {:?}", p.name, p.shape)
                };
                assert_eq!(
                    *sixteen, 16,
                    "`{}` is an mxfp4 codes plane stored {:?}",
                    p.name, p.shape
                );
                lead.iter().copied().chain([groups * 32]).collect()
            }
            _ => p.shape.clone(),
        };
        let shard = match &p.shard {
            Shard::Replicated => Shard::Replicated,
            Shard::Cut {
                axis,
                segments,
                heads,
            } => {
                let at = *axis as usize;
                let block = shape[at] / p.shape[at];
                Shard::Cut {
                    axis: *axis,
                    segments: segments.iter().map(|s| s * block).collect(),
                    heads: heads.clone(),
                }
            }
        };
        Weight {
            name: p.name.clone(),
            shape,
            dtype: p.dtype,
            act: None,
            shard,
            source: p.source,
            layout: p.layout,
            placement: p.placement,
        }
    }
}

/// The precision a deployment states for a bank: the dtype its weights are
/// stored at and, for a low-precision product, the precision its activations
/// are quantized to (`nvfp4a4`: NVFP4 weights against NVFP4 activations).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Precision {
    pub dtype: Dtype,
    pub act: Option<ActQuant>,
}

impl Precision {
    /// The stored dtypes that quantize their activations, by the word a
    /// deployment's name spells each: NVFP4 against NVFP4 (W4A4, vLLM's
    /// modelopt product), and MXFP4 (W4A8) and DeepSeek's 128x128-block FP8
    /// (W8A8) against e4m3 in its 1x128 blocks.
    pub const ACTED: [Precision; 6] = [
        Precision {
            dtype: Dtype::Nvfp4,
            act: Some(ActQuant::Nvfp4),
        },
        Precision {
            dtype: Dtype::Mxfp4,
            act: Some(ActQuant::E4m3 { gran: 128 }),
        },
        Precision {
            dtype: Dtype::Fp8b128,
            act: Some(ActQuant::E4m3 { gran: 128 }),
        },
        // A block-FP8 release's experts as released (f32 block factors),
        // against e4m3 in 1x128 blocks: vLLM's W8A8 MoE on them.
        Precision {
            dtype: Dtype::E4m3tile128,
            act: Some(ActQuant::E4m3 { gran: 128 }),
        },
        // DeepSeek-V4.1's release: MXFP8 dense weights (32x32 blocks) and
        // MXFP4 experts, each against e4m3 in 1x32 blocks.
        Precision {
            dtype: Dtype::Fp8b32,
            act: Some(ActQuant::E4m3 { gran: 32 }),
        },
        Precision {
            dtype: Dtype::Mxfp4,
            act: Some(ActQuant::E4m3 { gran: 32 }),
        },
    ];

    #[must_use]
    pub const fn of(dtype: Dtype) -> Precision {
        Precision { dtype, act: None }
    }

    /// What a deployment's name appends to its dtype's word: `a4` for NVFP4
    /// activations, `a8` for e4m3 in 128s, `a8g{n}` in `n`s.
    #[must_use]
    pub fn suffix(self) -> String {
        match self.act {
            None => String::new(),
            Some(ActQuant::Nvfp4) => "a4".to_string(),
            Some(ActQuant::E4m3 { gran: 128 }) => "a8".to_string(),
            Some(ActQuant::E4m3 { gran }) => format!("a8g{gran}"),
        }
    }
}

impl From<Dtype> for Precision {
    fn from(dtype: Dtype) -> Precision {
        Precision::of(dtype)
    }
}

impl PartialEq<Dtype> for Precision {
    fn eq(&self, other: &Dtype) -> bool {
        self.act.is_none() && self.dtype == *other
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

#[must_use]
pub fn global_scales_name(of: &str) -> String {
    format!("{of}{GLOBAL_SCALES}")
}

#[must_use]
pub fn input_scales_name(of: &str) -> String {
    format!("{of}{INPUT_SCALES}")
}

#[must_use]
pub fn compute_dtype(dtype: Dtype) -> Option<Dtype> {
    match dtype {
        Dtype::Bf16 => Some(Dtype::Bf16),
        Dtype::F16 => Some(Dtype::F16),
        Dtype::F32 => Some(Dtype::F32),
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
        | Dtype::E4m3tile128
        | Dtype::E4m3tensor
        | Dtype::Fp8b128
        | Dtype::Fp8b32
        | Dtype::Mxfp8
        | Dtype::Ptq1_0 => Some(Dtype::Bf16),
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
        | Dtype::KvU4
        | Dtype::Bool => None,
    }
}

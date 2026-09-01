//! Weight declarations without generics. The old `Tensor<W: Dtype>` carried
//! its representation as a phantom type; here `Dtype` is a plain field — one
//! `Weight` struct, no monomorphized model trees (design §5).

use std::cell::Cell;

use model_ir::{Dtype, ParamSource, Platform, Shard, TILED_BAND, TILED_STEP};

thread_local! {
    /// The SETUP a declaration is being read for, or `None` for a reading
    /// that states none. Written only by [`placing_for`], read only by
    /// [`place`].
    static PLACING: Cell<Option<Platform>> = const { Cell::new(None) };
}

/// **A PLACED DTYPE, RESOLVED FOR THE SETUP THIS DECLARATION IS BEING READ
/// FOR** ([`Platform::placement`]).
///
/// A model text calls this where it would otherwise state a placed variant
/// outright: `place(Dtype::U4g64tiled)` says *"this weight may be arranged
/// for a tensor-core lane"*, and the answer is that arrangement on a platform
/// whose kernels read it and the canonical row-major sibling on one that does
/// not. Which WEIGHTS may be placed stays the text's — the arrangement is
/// only legal on a two-dimensional projection — and WHETHER stops being the
/// text's, because a text does not know which shell is about to serve.
///
/// **A READING THAT STATES NO SETUP KEEPS THE TEXT'S WORDS.** `None` is not a
/// platform and this does not invent one: a declaration read outside any
/// setup — a census, a shape assertion, a tool that wants to know what the
/// text asked for — resolves to what the text asked for. That is a narrowing
/// that did not happen, never a wrong one that did: a placement no reader can
/// serve is refused BY NAME at the two doors that meet it
/// (`checkpoint::plan::passes::validate`'s `validate_target_support` on a
/// serving plan, `engine_metal::weights::readable_plane_orders` on the trace),
/// with the param named and the command to run. The failure mode of
/// forgetting is a sentence, not a wrong number.
///
/// **AMBIENT FOR THE REASON [`platform`](crate::platform) IS**, and not the
/// same ambient. That one is a fact about a TRACE and panics outside one,
/// because a forward pass is only ever inside one. This is a fact about a
/// SETUP, and the reader that most needs it is not tracing at all: a family's
/// `import` walks the same `Model` to state a load contract, and
/// `models::ImportRow` is `fn(&Source)` — the checkpoint's shape is the only
/// argument a contract has ever taken. Threading a platform through it would
/// put the word in sixty rows across seven families to be ignored by
/// fifty-nine of them, and the two readings would then be free to disagree
/// about a weight neither could see. One ambient, set around the CONSTRUCTION
/// of the model expression, is what keeps the trace's answer and the
/// contract's the same answer.
#[must_use]
pub fn place(dtype: Dtype) -> Dtype {
    match PLACING.with(Cell::get) {
        Some(platform) => platform.placement(dtype),
        None => dtype,
    }
}

/// Read a declaration for `platform`: every [`place`] inside `f` resolves
/// against it.
///
/// **THE SCOPE IS THE MODEL EXPRESSION'S CONSTRUCTION**, which is why this
/// takes a closure rather than a setter. `catalog!` wraps the `$m` it builds
/// per trace; a party that builds one to state an import contract wraps its
/// own call the same way (`runtime::engine::load`'s three import sites are
/// the shipping ones). Restored on the way out, panic or not, so a `place`
/// after the closure is a `place` with no setup and not the last one's.
///
/// Nested with a different platform is refused: two setups on one thread have
/// two answers for one weight, and quietly taking the inner one is how a
/// contract and a plan come to disagree.
pub fn placing_for<R>(platform: Platform, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<Platform>);
    impl Drop for Restore {
        fn drop(&mut self) {
            PLACING.with(|p| p.set(self.0));
        }
    }

    let was = PLACING.with(Cell::get);
    assert!(
        was.is_none_or(|seen| seen == platform),
        "a declaration is already being read for {was:?} and this one says {platform:?}",
    );
    let _restore = Restore(was);
    PLACING.with(|p| p.set(Some(platform)));
    f()
}

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
            // **AND THE TWO-BIT ROWS TAKE IT TOO**, for the same reason and one
            // number further: `U2g{32,64,128}` are the affine scheme at two
            // bits over three groups (`dtype::Dtype::U2g32`), and the group
            // is the ONLY thing below that reads off the dtype.
            Dtype::U4g64
            | Dtype::U8g64
            | Dtype::U4g32
            | Dtype::U2g32
            | Dtype::U2g64
            | Dtype::U2g128 => {
                // A TABLE, because the group is now four answers and not two.
                // The `_ => 64` it replaces was a default that read as "the
                // scheme's group" while a fifth row was being added under it.
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
            // **THE TILED AFFINE ROW IS THAT ARM WITH ITS COLUMNS BANDED**
            // (§J4b). Same scheme, same group of sixty-four, same two bf16
            // companions — the codes and the factors have been through
            // `kernels_cuda::linear::tiled`'s relabelling, which pads the
            // OUTPUT COLUMN axis up to a whole 16-column mma band and writes
            // the tail as zero codes beside zero factors, so that a padded
            // column decodes to a zero weight the epilogue does not store.
            //
            // The padding is why this is its own arm rather than one more
            // name in the list above: the three planes a repacked weight
            // publishes are a different RECTANGLE from the ones the row-major
            // weight publishes, and `plane_bytes` reads that rectangle to
            // size the store.
            //
            // **AND IT IS A PROJECTION, NOT A BANK.** The tiled point serves
            // `y = act x W^T` over a dense two-dimensional weight; a routed
            // expert bank has an axis it does not carve, and a repack of one
            // is a layout nobody has written. Refused here, where the text
            // said it.
            Dtype::U4g64tiled => {
                let [n, k] = self.shape[..] else {
                    panic!(
                        "`{}` is a tiled affine projection declared {:?}; the tiled point \
                         reads a two-dimensional weight",
                        self.name, self.shape,
                    )
                };
                // The group is the affine scheme's, and it reads off the
                // dtype like every other row's above. The band and the step
                // are the LAYOUT's, and they are `dtype`'s own constants —
                // `checkpoint`'s repack checks a declared target against the
                // same two numbers, and this is the rectangle it would check.
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
            // Served formats no model TEXT declares yet: they reach the
            // engine through an import plan rather than a family's weight
            // declaration, so a declaration naming one has no plane story to
            // intern. The day a text declares one, its author lands here and
            // writes that story — which is this panic's whole job.
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
            // **A SELF-CONTAINED TERM IS ONE PLANE, AND THE PLANE IS BYTES.**
            // A GGUF K-quant carries its scales INSIDE the super-block, so
            // there is no companion to intern and nothing to name: the whole
            // format is one rectangle of `n` rows by `Dtype::row_bytes(k)`
            // bytes, which is exactly what `linear::kquant` reads and exactly
            // what serving AS STORED means (alto `next.md` §J).
            //
            // **THE SHAPE IS FOLDED HERE, WHICH IS THE MXFP4 ARM'S OWN
            // MOVE.** The text declares the LOGICAL `[n, k]` — that is what a
            // model text is for, and hard-coding `144 * k / 256` into every
            // family would be exactly the lore QNF exists to delete — and
            // this is the one place that logical shape becomes the container
            // it is stored in. Downstream the rectangle IS the byte count,
            // which is the sentence `engine_cuda::weights::plane_bytes`
            // already says about mxfp4.
            //
            // A leaf-per-plane container is the other legal reading of the
            // same term (`.zt`'s canonical form, §J), and it would be a
            // second arm here rather than a change to this one — the term
            // says which planes there are, and the CONTAINER says whether
            // they are braided.
            // The stored super-block families, whose whole term lives in one
            // braided plane: the row is the term's own byte width and there is
            // no companion to publish (§J5).
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
        Dtype::Mxfp4
        | Dtype::U4g64
        | Dtype::U8g64
        | Dtype::U4g32
        | Dtype::U4g64tiled
        | Dtype::U2g32
        | Dtype::U2g64
        | Dtype::U2g128 => Some(Dtype::Bf16),
        // **A QUANT TERM SAYS HOW A WEIGHT IS STORED AND NOTHING ABOUT WHAT
        // IT MULTIPLIES AS.** Every weight-only quant point in this tree
        // decodes inside the dot and accumulates against a bf16 activation —
        // `linear::kquant` dispatches on `act.dtype` over `{bf16, f16}` and
        // the affine family does the same — so the answer is the four rows
        // above's answer, for the same reason: the codes are a storage fact
        // and the neighbours a bank does not quantize (its norms) are stated
        // in what the bank COMPUTES in.
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
        // **A LOOKUP TABLE COMPUTES AS ITSELF.** `I64` is the one integer row
        // here a model text declares a plane in, and it is not a
        // representation of a number a dot product accumulates — it is
        // `ffn.gate.tid2eid`, DeepSeek-V4-Flash's `[vocab, top_k]` token-id →
        // expert-id table, which `linear.moe_hash_route` GATHERS. A gather
        // reads the element the file stored, so the dtype an activation sees
        // through this plane is the dtype it was stored in; answering `Bf16`
        // here (the arm above) would say the table dequantizes, and answering
        // `None` (the arm below) would say no text may state it, which is the
        // sentence that kept the hash gate interned.
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

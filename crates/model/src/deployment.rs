//! `Deployment` — everything a driver needs to serve a checkpoint, with no
//! family name in it.
//!
//! A type and not a trait: the thirteen questions a `Box<dyn PlannedFamily>`
//! answered had exceptions that were all one family, and `facts_from_hf` ran
//! at the admission of every fire. Gemma-4's two head dims are now
//! `attention[l].head_dim` differing between layers, and a `Vec` of per-layer
//! facts has no opinion about which family produced it.
//!
//! It lives here because a driver reads the answer, never the question.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// How a layer attends.
///
/// PER LAYER, unconditionally — not "per layer for the families that
/// need it". A stack whose layers agree fills this with equal entries,
/// which costs a `Vec` and buys the absence of a special case.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayerAttention {
    /// The kernel-facing head dim for this layer.
    pub head_dim: u32,
    /// Sliding window, or `-1` for a layer that attends the whole
    /// context.
    pub window: i32,
    /// Which layer's KV pages this layer reads.
    ///
    /// Its own index for the ordinary case; gemma-4's trailing layers name
    /// an earlier one and own no pages themselves.
    pub kv_source: u32,
    /// The attention scale. Usually `1/sqrt(head_dim)`; gemma-4 runs
    /// `1.0` because its q/k norms carry the scaling.
    pub sm_scale: f32,
    /// Rope base for this layer. The stacks that use one theta
    /// throughout repeat it.
    pub rope_theta: f32,
    /// Rotary width, or `0` for full rotation at the head dim.
    pub rotary_dim: u32,
    /// The kv-head COUNT for this layer.
    ///
    /// A head shape is two numbers, and a stack-wide count is wrong for any
    /// stack whose layers disagree: the second shape's K would be paged at
    /// the first shape's width. Rows with one shape repeat it.
    pub kv_heads: u32,
    /// Whether this layer's Q projection also carries a per-head OUTPUT
    /// GATE, doubling its published width.
    ///
    /// `Qwen3NextAttention` projects `2 * q_heads * head_dim` and splits
    /// the result per head into a query and a gate, then multiplies the
    /// attention's output by `sigmoid(gate)` before landing it. Twelve
    /// families do not, so this is `false` for all of them.
    ///
    /// It is stated because a projection WIDTH is the one attention
    /// fact a name gate cannot check: every tensor is present and
    /// correctly spelled, and a reader holding the ungated width takes
    /// half of each Q. That is the gemma-4 defect `kv_heads` was added
    /// to this struct for, one field over.
    pub q_gate: bool,
}

/// Which gate the MLP applies to its first projection.
///
/// A driver that receives a shape rather than a text has nowhere else to
/// learn this, and serving a GELU stack on SiLU is a few percent at the
/// origin that diverges from there: finite, plausible, never faulting, wrong.
///
/// The clamp is on the variant rather than beside it, because a limit of zero
/// and a limit of seven are not two settings of one gate — gpt-oss's is a
/// different function, and a row holding `0.0` in a field nothing reads
/// cannot be told from one that forgot to fill it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum MlpGate {
    /// `silu(gate) * up` — llama's, and most of the catalog's.
    Silu,
    /// `gelu_tanh(gate) * up` — every gemma, whose configs spell it
    /// `hidden_activation: "gelu_pytorch_tanh"`.
    GeluTanh,
    /// gpt-oss's clamped SwiGLU: both halves are clipped and the gate
    /// carries an alpha.
    SiluClamped {
        /// `swiglu_limit`, 7.0 on gpt-oss.
        limit: f32,
        /// The gate's alpha, 1.702 on gpt-oss.
        alpha: f32,
    },
}

/// What kind of KV this deployment needs.
///
/// AN ENUM, so a shape the driver has no pool for is an `unimplemented!` arm
/// rather than a row in a registry that loads successfully and dies at its
/// first fire.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvStyle {
    /// Ordinary paged K and V.
    Paged,
    /// Multi-head latent attention: a compressed KV plane and a
    /// positional one, which do not fit the standard k/v pair.
    Mla {
        /// The compressed KV rank.
        kv_lora_rank: u32,
        /// The rope head dim carried beside it.
        qk_rope_head_dim: u32,
    },
    /// A per-layer COMPRESSED KV plane, stated as one compression ratio
    /// per layer.
    ///
    /// Named for the shape and not for the generation that introduced it: a
    /// driver matching on it must not have to spell a family name to ask a
    /// question about a KV layout.
    CompressedPlane {
        /// One ratio per layer; `None` for an uncompressed layer.
        ratios: Vec<i32>,
    },
}

impl KvStyle {
    /// Whether THIS BUILD provisions a store this style can live in.
    ///
    /// A capability question, not a shape one: the style is a fact about the
    /// model and this answer is a fact about the binary, and keeping them
    /// separable is why it is a method rather than a variant.
    #[must_use]
    pub fn has_a_store_in_this_build(&self) -> bool {
        match self {
            Self::Paged => true,
            Self::Mla { .. } | Self::CompressedPlane { .. } => false,
        }
    }

    /// The refusal a build with no store for this style owes its caller,
    /// or `None` when one exists.
    ///
    /// Keyed on the STYLE and not on the family: the missing store is a
    /// property of the shape, and a compressed KV plane has nowhere to live
    /// regardless of which vendor shipped it.
    #[must_use]
    pub fn store_refusal(&self) -> Option<Refusal> {
        match self {
            Self::Paged => None,
            Self::Mla { .. } => Some(Refusal::Unsupported(
                "this build provisions no MLA latent store; a compressed KV \
                 plane and a positional one do not fit the k/v pair the pager \
                 allocates",
            )),
            Self::CompressedPlane { .. } => Some(Refusal::Unsupported(
                "this build provisions no compressed KV plane store; the row's \
                 per-layer compressed entries have nowhere to live",
            )),
        }
    }
}

/// A recurrent stack's slab geometry — what a driver must allocate and
/// stride before it can run one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecurrentShape {
    /// Which layers are linear-attention layers.
    pub linear_layers: Vec<u32>,
    /// Bytes per conv slot.
    pub conv_stride: usize,
    /// Bytes per recurrent-state slot.
    pub state_stride: usize,
    /// Element width of the recurrent state.
    pub state_elem: usize,
    /// Key heads.
    pub k_h: i32,
    /// Value heads.
    pub v_h: i32,
    /// Key head dim.
    pub k_d: i32,
    /// Value head dim.
    pub v_d: i32,
    /// Conv channel count.
    pub conv_dim: i32,
    /// Conv kernel width.
    pub conv_k: i32,
    /// mamba's B/C group count, or `0` for a gated-delta stack.
    ///
    /// The one number of a mamba mixer that NO tensor extent carries: the
    /// checkpoint ships `2 * n_groups * state_size` rows of B and C fused
    /// into one bank, so a loader holding the tensors knows only their
    /// PRODUCT. Its own field rather than riding [`Self::k_h`], because
    /// `GdnCtx::n_groups` divides by it — at zero, that is a divide by zero.
    pub n_groups: i32,
}

/// Whether the prefill path can be planned ahead of the fire.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefillStyle {
    /// The ordinary case: a plan is raised before the fire and bound.
    Planned,
    /// The plan is built inside the fire from the host CSR mirrors, so
    /// there is nothing to raise. Gemma-4's 512-wide layers take the
    /// naive kernel and plan internally.
    Planless,
}

/// Where a layer's norm sits relative to its projections.
///
/// A DRIVER needs it: an adapter's staging reads the projection input, and
/// which buffer that is depends on this. `Pre` ships one input norm; `Post`
/// (olmo2) ships `post_attention` and `post_feedforward` instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormPlacement {
    /// The norm precedes the projections; the input is the normed value.
    Pre,
    /// The norm follows them; the input is the residual stream.
    Post,
}

/// Where the attention output lands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttnOutput {
    /// The guard region records no SSA output, so the driver owns the
    /// landing buffer and pins the query.
    DriverPinned,
    /// The forward states `[q, o]` as SSA args, so there is nothing to
    /// pin.
    StatedArgs,
}

/// The geometry a launch path reads, once, off the value it was already
/// holding.
///
/// Nothing here is derived: they are the row's own numbers, in the row's own
/// units, with the one exception named on [`Self::head_dim_kernel`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Geometry {
    /// Residual width.
    pub hidden: u32,
    /// Query heads.
    pub q_heads: u32,
    /// Key/value heads; equal to [`Self::q_heads`] for MHA.
    pub kv_heads: u32,
    /// One head's width, as the CHECKPOINT states it.
    pub head_dim: u32,
    /// The width a kernel was instantiated at, when it differs.
    ///
    /// The ONE derived number: phi-3's 96-wide heads run on the 128-wide
    /// kernel, so a buffer sized `heads * head_dim` is too small by a third.
    /// Equal to [`Self::head_dim`] when nothing was padded, so a reader that
    /// wants the width to allocate can always use this one.
    pub head_dim_kernel: u32,
    /// The dense MLP's inner width.
    pub intermediate: u32,
    /// One EXPERT's inner width, or `0` for a dense stack.
    ///
    /// Separate from [`Self::intermediate`] because the forward workspace is
    /// ONE buffer both layer kinds share, so what it must hold is the wider.
    /// A planner given only the dense width under-sizes a mixture whose
    /// experts are wider, and that moves bytes out of the KV pool quietly.
    pub moe_intermediate: u32,
    /// How many experts one token visits, or `0` for a dense stack.
    ///
    /// A mixture is not described by a width alone: the router ranks the
    /// experts and this is how deep down that ranking a token goes. A mixture
    /// fired at the wrong top-k routes each token to almost the right experts
    /// and returns fluent nonsense.
    pub experts_per_token: u32,
    /// The dense FFN a routed layer runs BESIDE the bank, gated by one
    /// sigmoid scalar per token. Zero means the routing has none, and it is
    /// `n_shared_experts * moe_intermediate` rather than one expert's width —
    /// the rows already fold the count in. Stated because there is no proxy:
    /// equal to `moe_intermediate` is false for kimi-k2, and zero drops a
    /// whole FFN from every token.
    pub shared_intermediate: u32,
    /// The logit dimension.
    ///
    /// The MODEL's `vocab_size` and never the tokenizer's token count —
    /// they differ (qwen3: 151 669 against 151 936) and using the
    /// smaller one is the vocab-padding device fault.
    pub vocab: u32,
}

/// The attention head dims a build INSTANTIATES.
///
/// `kernels.def`'s `PIE_ATTN_HEAD_DIM` rows on the CUDA side, and the same
/// four points on the Metal side. A property of the BINARY, not of any
/// checkpoint, which is why no row states it.
///
/// Beside [`Geometry`] because [`Geometry::head_dim_alloc`] is the CONSUMER's
/// half of this same question, so a producer that pads and a consumer that
/// allocates cannot disagree.
pub const ATTN_HEAD_DIMS: &[u32] = &[64, 128, 256, 512];

/// Smallest instantiated head dim that can hold `head_dim`, or `head_dim`
/// itself when none can — the caller then surfaces the dispatch error rather
/// than silently mis-sizing.
///
/// The result is never less than `head_dim`, so callers need no `.max()`.
#[must_use]
pub fn round_up_attn_head_dim(head_dim: u32) -> u32 {
    ATTN_HEAD_DIMS
        .iter()
        .copied()
        .filter(|&d| d >= head_dim)
        .min()
        .unwrap_or(head_dim)
}

impl Geometry {
    /// The zeros that go with [`Deployment::empty`]: a shape no fire can
    /// take, so a driver that forgot to fill it in refuses at its first
    /// admission rather than serving a stack it never derived.
    pub const EMPTY: Self = Self {
        hidden: 0,
        q_heads: 0,
        kv_heads: 0,
        head_dim: 0,
        head_dim_kernel: 0,
        intermediate: 0,
        moe_intermediate: 0,
        experts_per_token: 0,
        shared_intermediate: 0,
        vocab: 0,
    };

    /// Heads per KV group — the GQA ratio a decode kernel is instantiated for.
    ///
    /// Zero KV heads answers 0 rather than dividing, because [`Self::EMPTY`]
    /// must be askable.
    #[must_use]
    pub const fn gqa_group(&self) -> u32 {
        // `match` rather than `unwrap_or`, which is not const yet.
        match self.q_heads.checked_div(self.kv_heads) {
            Some(group) => group,
            None => 0,
        }
    }

    /// The width to ALLOCATE for one head: the kernel's, when one was
    /// instantiated wider than the checkpoint's.
    #[must_use]
    pub const fn head_dim_alloc(&self) -> u32 {
        if self.head_dim_kernel > self.head_dim {
            self.head_dim_kernel
        } else {
            self.head_dim
        }
    }

    /// The widest MLP any layer in the stack asks for.
    ///
    /// The forward workspace is sized from this: a `max` rather than a choice
    /// because a mixture's layers share the buffer with its dense ones.
    #[must_use]
    pub const fn widest_mlp(&self) -> u32 {
        if self.moe_intermediate > self.intermediate {
            self.moe_intermediate
        } else {
            self.intermediate
        }
    }
}

/// How a stack rescales its rope frequency ladder, for the stacks that do.
///
/// # Why this is a `Deployment` field and not a derivation
///
/// A factor of zero reads as "no rescaling", so a stack whose numbers never
/// arrive attends with the wrong wavelengths past its original context and
/// degrades fluently. And it is a per-CHECKPOINT fact, not a per-generation
/// one: Llama-3.2's 1B and 3B rescale by `32.0` where 3.1's 8B and 70B
/// rescale by `8.0`, from the same `rope_theta` and the same original
/// context.
///
/// `None` is a statement, not a default: this stack uses its `rope_theta`
/// ladder unrescaled.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RopeScaling {
    /// A piecewise rescaling by wavelength band (`rope_type: "llama3"`).
    ///
    /// Wavelengths shorter than the high-frequency cut pass through
    /// untouched, those longer than the low-frequency cut are divided by
    /// `factor`, and the band between them interpolates. The two cuts are
    /// expressed as divisors of `original_max_position`, which is why they
    /// are factors and not lengths.
    Piecewise {
        /// The divisor applied to the low-frequency end.
        factor: f32,
        /// Divisor of `original_max_position` giving the wavelength
        /// below which nothing is rescaled.
        low_freq_factor: f32,
        /// Divisor of `original_max_position` giving the wavelength
        /// above which everything is rescaled by `factor`.
        high_freq_factor: f32,
        /// The context the checkpoint was TRAINED at, which is the
        /// length the rescaling is measured against — not
        /// [`Advertised::max_model_len`], which is the extended one.
        original_max_position: u32,
    },
    /// YaRN's NTK-by-parts interpolation (`rope_type: "yarn"`).
    ///
    /// `beta_fast` and `beta_slow` bound a ramp in rotations-per-token;
    /// `attention_factor` scales the attention logits to compensate for
    /// the lengthened ladder.
    Yarn {
        /// Context-extension ratio.
        factor: f32,
        /// High-rotation end of the ramp.
        beta_fast: f32,
        /// Low-rotation end of the ramp.
        beta_slow: f32,
        /// The logit scale. HF computes `0.1 * ln(factor) + 1` when a config
        /// omits it, which is what OLMo-3 states explicitly — so a row that
        /// omits it states the same number the formula gives.
        attention_factor: f32,
        /// The context the checkpoint was trained at.
        original_max_position: u32,
        /// Whether the ramp's ends snap to whole channels.
        ///
        /// HF's `find_correction_range` floors and ceils by default. Most
        /// rows omit the flag and mean `true`; every gpt-oss writes
        /// `truncate: false`, which moves the ramp from `(8, 18)` to
        /// `(8.09, 17.40)` and three percent of channel 12.
        truncate: bool,
    },
}

/// Everything a driver needs to serve a checkpoint.
///
/// `Clone`, `Debug`, comparable and derivable once at load — not per
/// fire. Nothing in it is a string naming a family.
#[derive(Debug, Clone, PartialEq)]
pub struct Deployment {
    /// How many layers.
    pub layers: u32,
    /// RMSNorm epsilon.
    ///
    /// A CONSTANT of the checkpoint — `1e-6` for most of the llama lineage,
    /// `1e-5` for the qwen-2 generation — and one no tensor extent carries,
    /// so a row must state it.
    pub norm_eps: f32,
    /// The geometry a driver's LAUNCH path reads.
    ///
    /// A projection of the row like everything else here, so a launch and a
    /// trace cannot disagree about how many heads there are — and so no
    /// driver keeps a normalized config resident to answer them.
    pub shape: Geometry,
    /// Per-layer attention facts.
    pub attention: Vec<LayerAttention>,
    /// What kind of KV to provision.
    pub kv: KvStyle,
    /// The recurrent slabs, for a hybrid stack.
    pub recurrent: Option<RecurrentShape>,
    /// Whether prefill can be planned ahead.
    pub prefill: PrefillStyle,
    /// Where attention output lands.
    pub attn_output: AttnOutput,
    /// Final-logit softcap, `0.0` for none.
    pub logit_softcap: f32,
    /// ATTENTION-logit softcap — `cap * tanh(score / cap)` applied to
    /// the scores, not to the readout — or `0.0` for none.
    ///
    /// gemma-2's `attn_logit_softcapping`, which every published row states as
    /// `50.0` against the readout's `30.0`. Two caps, two numbers, two places
    /// in the fire; this one rides as a DISPATCH parameter, reaching the
    /// kernel as `logits_soft_cap` on every flashinfer entry point.
    ///
    /// A `Deployment` field rather than an argument to `attention_for`,
    /// because the cap is one number for the whole fire — like `sm_scale` and
    /// the window beside it in `AttnCtx` — and a trace that had to pass it at
    /// every layer is a trace that can forget it at one.
    pub attn_logit_softcap: f32,
    /// Per-layer-embedding width, `0` for a stack without one.
    ///
    /// UNSIGNED: `0` is this field's word for "this stack has no per-layer
    /// embeddings", and a narrowing conversion that produced it would switch
    /// the PLE path off silently. Contrast [`LayerAttention::window`], where
    /// `-1` is load-bearing and the signedness is earned.
    pub ple_dim: u32,
    /// Where the norm sits — read by anything that needs to name the
    /// projection input, which today is the adapter staging.
    pub norm: NormPlacement,
    /// Whether the norm's gain is stored as an OFFSET FROM ONE, so that
    /// firing it means `(1 + w) * x` rather than `w * x`.
    ///
    /// Stated rather than derived: the only other place to read it from is
    /// the norm PLACEMENT, and gemma-4 publishes the sandwich while storing a
    /// plain multiplier. Inferring it does not fail loudly — `(1 + w)/w` is
    /// 1.002 where `w` is 444 and 1.38 where `w` is 2.6, so the largest gains
    /// agree to three digits while the ordinary ones are off by a third.
    ///
    /// `false` for every non-gemma row, which is the true answer for them
    /// rather than a default.
    pub norm_unit_offset: bool,
    /// Whether V is RMS-normed, per head, on its way to the KV pool.
    ///
    /// Not implied by the per-head QK norm — gemma-3 carries `q_norm` and
    /// `k_norm` and has no V norm at all. A ROW'S ANSWER rather than a probe
    /// because the module ships NO PARAMETER, so a checkpoint contains
    /// nothing `has_tensor` could be asked about.
    pub v_norm: bool,
    /// Which gate this stack's MLP applies. See [`MlpGate`].
    pub mlp_gate: MlpGate,
    /// Whether the router renormalizes over the SELECTED experts.
    ///
    /// HF's `norm_topk_prob`. True softmaxes the k chosen logits so the
    /// routing weights sum to one; false softmaxes over ALL the experts and
    /// then selects, so they sum to less than one and scale the routed FFN's
    /// whole contribution down with them. Both produce weights, neither
    /// faults, and the difference is a few percent of every routed token.
    ///
    /// A CONVENTION the stack was trained under, not a size, which is why it
    /// is here rather than on [`Geometry`]. A dense stack states `true` and
    /// nothing reads it: a row added later should have to answer.
    pub norm_topk_prob: bool,
    /// `routed_scaling_factor` — what the routing weights are multiplied by
    /// once the router has produced them.
    ///
    /// The other half of [`Self::norm_topk_prob`]; the pair is only
    /// meaningful together, because the scaling is what pays for weights that
    /// were never renormalized. DeepSeek-V3 publishes 2.5 against a
    /// `norm_topk_prob` of false, GLM-4.5 publishes 2.5 against true, and the
    /// families with neither key want 1.0.
    pub routed_scaling: f32,
    /// Named scalar constants the forward refers to by name.
    pub scales: BTreeMap<String, f32>,
    /// What a driver ADVERTISES about this model, as distinct from what
    /// it needs to fire it.
    pub advertised: Advertised,
    /// How the rope ladder is rescaled, `None` for the stacks that use
    /// their [`LayerAttention::rope_theta`] ladder as-is.
    ///
    /// A property of the STACK and not of a layer: no checkpoint here
    /// rescales one layer and not another, while gemma-3 and gemma-4 do run
    /// two different bases.
    pub rope_scaling: Option<RopeScaling>,
    /// The media encoders this row ships, empty for a text-only stack.
    pub towers: Towers,
}

/// The encoder stacks that run BESIDE the decoder.
///
/// A tower is not a layer of the model: it takes waveform or pixels and hands
/// back rows the decoder embeds, on its own kernels, at its own depth. It is
/// here because a `Deployment` is what a driver fires and the driver fires
/// these too.
///
/// Both are `None` on every text-only row. A driver's encode entry refuses on
/// `None` rather than defaulting, because a default tower is a plausible
/// shape for a stack that does not exist.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Towers {
    /// The audio front-end, `None` when the row ships none.
    pub audio: Option<AudioTower>,
    /// The vision front-end, `None` when the row ships none.
    pub vision: Option<VisionTower>,
}

/// A conformer audio encoder's shape.
///
/// Fourteen numbers: every field of a conformer config a driver reads.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AudioTower {
    /// Conformer blocks.
    pub layers: u32,
    /// The residual stream's width.
    pub hidden: u32,
    /// Attention heads per block.
    pub heads: u32,
    /// The depthwise convolution's kernel width.
    pub conv_kernel: u32,
    /// Mel bins per frame — the encoder's input width.
    pub feature_size: u32,
    /// Channels out of the first subsampling convolution.
    pub subsample_channels_0: u32,
    /// Channels out of the second.
    pub subsample_channels_1: u32,
    /// The width the encoder projects onto for the decoder.
    pub output_dims: u32,
    /// Frames per local-attention chunk.
    pub chunk_size: u32,
    /// Chunks of history each chunk may attend.
    pub context_left: u32,
    /// Chunks of lookahead each chunk may attend.
    pub context_right: u32,
    /// The tanh cap on attention logits.
    pub logit_cap: f32,
    /// The residual's weight in the conformer's half-step sum.
    pub residual_weight: f32,
    /// The norm epsilon, the tower's own.
    pub norm_eps: f32,
}

/// A vision encoder's shape.
///
/// Seven numbers, again the ones a driver reads. The patch grid and token
/// count the host needs come from [`crate::multimodal`], which computes them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VisionTower {
    /// Transformer blocks.
    pub layers: u32,
    /// The residual stream's width.
    pub hidden: u32,
    /// Attention heads per block.
    pub heads: u32,
    /// The MLP's inner width.
    pub intermediate: u32,
    /// The average-pool kernel that reduces the patch grid.
    pub pooling_kernel: u32,
    /// The norm epsilon, the tower's own.
    pub norm_eps: f32,
    /// The rotary base, the tower's own.
    pub rope_theta: f32,
}

/// The three answers a driver puts in its capabilities that are facts
/// about the MODEL rather than about the device.
///
/// Here rather than on [`Variant`](crate::catalog::Variant) as three more
/// required methods, so that answering them costs a driver no resident
/// parsed config.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Advertised {
    /// The family label a GUEST PROGRAM matches on.
    ///
    /// Not a dispatch key — that is the whole point of the catalog — but
    /// still a real value, because `engine`'s `model.arch_name()` is a host
    /// function inferlets call and `VisionArch::from_arch_name` selects an
    /// image front-end from it. Deliberately coarser than an id.
    ///
    /// Matched WHOLE by every consumer: a substring match makes `qwen3`
    /// select the front-end belonging to `qwen3_5`.
    pub arch: &'static str,
    /// The published context ceiling, `0` when the row does not state one.
    ///
    /// A TRAINING-time fact and not a deployment one: nothing in a fire reads
    /// it, and a driver serving a shorter context than this is serving
    /// correctly.
    pub max_model_len: u32,
    /// Whether this row ships a tower the driver's encode entry point serves.
    ///
    /// The worker refuses to build an encode executor at all when this is
    /// clear, so a row that ships towers and states `false` has them
    /// unreachable through the engine.
    ///
    /// Qwen3-VL is deliberately NOT this: its tower writes into the fire's
    /// hidden rows rather than handing host rows back, so it is an in-fire
    /// path and not an encode one.
    pub media_encode: bool,
}

impl Deployment {
    /// This plan, or the refusal its KV store draws.
    ///
    /// One place rather than one per MLA generation, and reachable here: a
    /// `Deployment` with a paged layout is one any dense family builds.
    #[must_use = "the refusal is the whole result; dropping it deploys a \
                  row whose store does not exist"]
    pub fn provisioned(self) -> Result<Self, Refusal> {
        match self.kv.store_refusal() {
            Some(no_store) => Err(no_store),
            None => Ok(self),
        }
    }

    /// A placeholder for a driver that must build its model value before
    /// it can derive one.
    ///
    /// Zero layers, a shape no fire can take, so a driver that forgot to fill
    /// it in refuses at its first admission.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            layers: 0,
            norm_eps: 0.0,
            // False, not true, so this is not the value any real row wants:
            // `empty()` exists to be refused, and a driver that served off it
            // would route on unnormalized weights.
            norm_topk_prob: false,
            routed_scaling: 1.0,
            shape: Geometry::EMPTY,
            attention: Vec::new(),
            kv: KvStyle::Paged,
            recurrent: None,
            prefill: PrefillStyle::Planned,
            attn_output: AttnOutput::StatedArgs,
            logit_softcap: 0.0,
            // A stack with no layers has nothing to cap, and `0.0` reads as
            // "none" both here and at `AttnCtx`.
            attn_logit_softcap: 0.0,
            ple_dim: 0,
            norm: NormPlacement::Pre,
            norm_unit_offset: false,
            v_norm: false,
            mlp_gate: MlpGate::Silu,
            scales: BTreeMap::new(),
            advertised: Advertised::default(),
            rope_scaling: None,
            towers: Towers::default(),
        }
    }

    /// The distinct decode head dims this stack needs plans for.
    ///
    /// `None` when every layer agrees, which is the ordinary case;
    /// `Some((a, b))` when two kinds disagree.
    #[must_use]
    pub fn decode_head_dims(&self) -> Option<(u32, u32)> {
        let first = self.attention.first()?.head_dim;
        let other = self
            .attention
            .iter()
            .find(|a| a.head_dim != first)?
            .head_dim;
        Some((first, other))
    }

    /// The FULL-attention layers' head shape, when it differs.
    ///
    /// `None` when every layer agrees; `Some((head_dim, kv_heads,
    /// rotary_dim))` for a stack whose windowed and unwindowed layers are
    /// shaped differently.
    ///
    /// Keyed on the WINDOW rather than on the shape that differs, because a
    /// page is sized per layer and the layer that reads the whole context is
    /// the one that has to be right.
    #[must_use]
    pub fn full_attention_shape(&self) -> Option<(u32, u32, u32)> {
        let first = self.attention.first()?;
        let full = self.attention.iter().find(|a| a.window < 0)?;
        if full.head_dim == first.head_dim && full.kv_heads == first.kv_heads {
            return None;
        }
        Some((full.head_dim, full.kv_heads, full.rotary_dim))
    }

    /// Does any layer read another's KV pages?
    #[must_use]
    pub fn shares_kv(&self) -> bool {
        self.attention
            .iter()
            .enumerate()
            .any(|(l, a)| a.kv_source as usize != l)
    }

    /// CAN THIS BUILD SERVE THIS STACK — asked at load, not at launch.
    ///
    /// FlashInfer's decode instantiates a fixed set of GQA group sizes and
    /// reports anything else by THROWING. A throw crossing the C ABI is
    /// undefined behaviour and a launcher signature has nowhere to put a
    /// failure; a LOAD returns a status code, so the question is asked here.
    ///
    /// A property of the BUILD, so it takes the set as an ARGUMENT: `model`
    /// states the shape, the driver states what it instantiated, and neither
    /// has to know the other's answer.
    ///
    /// The ratio itself is [`Geometry::gqa_group`]. What this adds is
    /// divisibility, which that one cannot answer: it truncates, so 14 over 4
    /// reads as 3 — a group every build instantiates, for a stack no build
    /// can run.
    ///
    /// # Errors
    ///
    /// [`Refusal::Unsupported`] when the head counts do not divide, or
    /// when the resulting group size is not in `groups`.
    pub fn servable_by(&self, groups: &[u32]) -> Result<(), Refusal> {
        let (q, kv) = (self.shape.q_heads, self.shape.kv_heads);
        if kv == 0 || q % kv != 0 {
            return Err(Refusal::Unsupported(
                "the query heads do not divide the kv heads, so this stack \
                 asks for a fractional GQA group no build instantiates",
            ));
        }
        if groups.contains(&self.shape.gqa_group()) {
            Ok(())
        } else {
            Err(Refusal::Unsupported(
                "this build's decode does not instantiate the GQA group size \
                 this stack asks for",
            ))
        }
    }

    /// The sliding window per layer, as the fire path binds it.
    #[must_use]
    pub fn windows(&self) -> Vec<i32> {
        self.attention.iter().map(|a| a.window).collect()
    }

    /// Rope base per layer, or empty when one theta serves the stack.
    ///
    /// EMPTY RATHER THAN REPEATED, because the binder's fast path
    /// checks emptiness — a table of identical values is a table it
    /// would walk for nothing.
    #[must_use]
    pub fn theta_by_layer(&self) -> Vec<f32> {
        let first = self.attention.first().map_or(0.0, |a| a.rope_theta);
        if self.attention.iter().all(|a| a.rope_theta == first) {
            return Vec::new();
        }
        self.attention.iter().map(|a| a.rope_theta).collect()
    }

    /// Rotary width per layer, or empty when every layer rotates fully.
    #[must_use]
    pub fn rotary_by_layer(&self) -> Vec<u32> {
        if self.attention.iter().all(|a| a.rotary_dim == 0) {
            return Vec::new();
        }
        self.attention.iter().map(|a| a.rotary_dim).collect()
    }
}

#[cfg(test)]
mod tests {

    /// Two towers that compare unequal must also PRINT unequal.
    ///
    /// `Debug` is the only way one of these reaches a human, so a field it
    /// omitted would make two differing towers render identically. Perturb
    /// one number at a time: both the comparison and the rendering notice.
    #[test]
    fn a_tower_that_differs_in_one_field_prints_differently() {
        let v = VisionTower {
            layers: 27,
            hidden: 1152,
            heads: 16,
            intermediate: 4304,
            pooling_kernel: 4,
            norm_eps: 1e-6,
            rope_theta: 100.0,
        };
        let vision: [(&str, VisionTower); 7] = [
            ("layers", VisionTower { layers: 28, ..v }),
            ("hidden", VisionTower { hidden: 768, ..v }),
            ("heads", VisionTower { heads: 12, ..v }),
            (
                "intermediate",
                VisionTower {
                    intermediate: 3072,
                    ..v
                },
            ),
            (
                "pooling_kernel",
                VisionTower {
                    pooling_kernel: 2,
                    ..v
                },
            ),
            (
                "norm_eps",
                VisionTower {
                    norm_eps: 1e-5,
                    ..v
                },
            ),
            (
                "rope_theta",
                VisionTower {
                    rope_theta: 10_000.0,
                    ..v
                },
            ),
        ];
        for (field, other) in vision {
            assert_ne!(v, other, "changing {field} did not change the tower");
            assert_ne!(
                format!("{v:?}"),
                format!("{other:?}"),
                "two vision towers differing in {field} print identically, so a \
                 diagnostic naming the difference cannot show it"
            );
        }

        let a = AudioTower {
            layers: 12,
            hidden: 1536,
            heads: 8,
            conv_kernel: 5,
            feature_size: 128,
            subsample_channels_0: 128,
            subsample_channels_1: 32,
            output_dims: 1536,
            chunk_size: 12,
            context_left: 13,
            context_right: 0,
            logit_cap: 50.0,
            residual_weight: 0.5,
            norm_eps: 1e-6,
        };
        let audio: [(&str, AudioTower); 14] = [
            ("layers", AudioTower { layers: 13, ..a }),
            ("hidden", AudioTower { hidden: 768, ..a }),
            ("heads", AudioTower { heads: 4, ..a }),
            (
                "conv_kernel",
                AudioTower {
                    conv_kernel: 3,
                    ..a
                },
            ),
            (
                "feature_size",
                AudioTower {
                    feature_size: 80,
                    ..a
                },
            ),
            (
                "subsample_channels_0",
                AudioTower {
                    subsample_channels_0: 64,
                    ..a
                },
            ),
            (
                "subsample_channels_1",
                AudioTower {
                    subsample_channels_1: 64,
                    ..a
                },
            ),
            (
                "output_dims",
                AudioTower {
                    output_dims: 2048,
                    ..a
                },
            ),
            ("chunk_size", AudioTower { chunk_size: 6, ..a }),
            (
                "context_left",
                AudioTower {
                    context_left: 7,
                    ..a
                },
            ),
            (
                "context_right",
                AudioTower {
                    context_right: 1,
                    ..a
                },
            ),
            (
                "logit_cap",
                AudioTower {
                    logit_cap: 30.0,
                    ..a
                },
            ),
            (
                "residual_weight",
                AudioTower {
                    residual_weight: 1.0,
                    ..a
                },
            ),
            (
                "norm_eps",
                AudioTower {
                    norm_eps: 1e-5,
                    ..a
                },
            ),
        ];
        for (field, other) in audio {
            assert_ne!(a, other, "changing {field} did not change the tower");
            assert_ne!(
                format!("{a:?}"),
                format!("{other:?}"),
                "two audio towers differing in {field} print identically"
            );
        }

        // And what a guest matches on. `Advertised::default()` is what an
        // unstated row advertises, and the empty label is the point: a program
        // matching on it matches nothing rather than a family by accident.
        let d = Advertised::default();
        assert_eq!(d.arch, "");
        assert_eq!(d.max_model_len, 0);
        assert!(!d.media_encode);
        let advertised: [(&str, Advertised); 3] = [
            (
                "arch",
                Advertised {
                    arch: "gemma4",
                    ..d.clone()
                },
            ),
            (
                "max_model_len",
                Advertised {
                    max_model_len: 8192,
                    ..d.clone()
                },
            ),
            (
                "media_encode",
                Advertised {
                    media_encode: true,
                    ..d.clone()
                },
            ),
        ];
        for (field, other) in advertised {
            assert_ne!(
                d, other,
                "changing {field} did not change what is advertised"
            );
            assert_ne!(
                format!("{d:?}"),
                format!("{other:?}"),
                "two rows differing in {field} advertise the same printed thing"
            );
        }
    }

    use super::*;

    fn layer(head_dim: u32) -> LayerAttention {
        LayerAttention {
            head_dim,
            kv_heads: 1,
            window: -1,
            kv_source: 0,
            sm_scale: 1.0,
            rope_theta: 10_000.0,
            rotary_dim: 0,
            q_gate: false,
        }
    }

    fn stack(dims: &[u32]) -> Deployment {
        Deployment {
            layers: dims.len() as u32,
            // The two the launch path reads, stated rather than defaulted so
            // a `stack()` is a whole `Deployment`.
            norm_eps: 1e-5,
            norm_topk_prob: true,
            routed_scaling: 1.0,
            shape: Geometry::EMPTY,
            attention: dims
                .iter()
                .enumerate()
                .map(|(l, &d)| LayerAttention {
                    kv_source: l as u32,
                    ..layer(d)
                })
                .collect(),
            kv: KvStyle::Paged,
            recurrent: None,
            prefill: PrefillStyle::Planned,
            attn_output: AttnOutput::DriverPinned,
            logit_softcap: 0.0,
            attn_logit_softcap: 0.0,
            ple_dim: 0,
            norm: NormPlacement::Pre,
            norm_unit_offset: false,
            v_norm: false,
            mlp_gate: MlpGate::Silu,
            scales: BTreeMap::new(),
            advertised: Advertised::default(),
            rope_scaling: None,
            towers: Towers::default(),
        }
    }

    /// A uniform stack answers `None` without anyone having to know which
    /// family it is.
    #[test]
    fn two_head_dims_is_a_property_of_the_layers_not_of_a_family() {
        assert_eq!(stack(&[128, 128, 128]).decode_head_dims(), None);
        assert_eq!(stack(&[128, 128, 256]).decode_head_dims(), Some((128, 256)));
    }

    /// Likewise KV sharing: gemma-4's trailing layers read an earlier
    /// layer's pages, and that is a fact about a LAYER.
    #[test]
    fn kv_sharing_is_a_property_of_the_layers() {
        assert!(!stack(&[128, 128]).shares_kv());
        let mut shared = stack(&[128, 128]);
        shared.attention[1].kv_source = 0;
        assert!(shared.shares_kv());
    }

    /// A uniform table is EMPTY rather than repeated, because the
    /// binder's fast path checks emptiness and a table of identical
    /// values is one it would walk for nothing.
    #[test]
    fn a_uniform_table_is_empty_rather_than_repeated() {
        assert!(stack(&[128, 128]).theta_by_layer().is_empty());
        let mut mixed = stack(&[128, 128]);
        mixed.attention[1].rope_theta = 1_000_000.0;
        assert_eq!(mixed.theta_by_layer().len(), 2);
    }

    /// The MLA orphan, as a type: a driver matching on `KvStyle` has to
    /// write the arm or refuse.
    #[test]
    fn an_unservable_kv_shape_is_a_variant_rather_than_a_registry_row() {
        let mut d = stack(&[128]);
        d.kv = KvStyle::Mla {
            kv_lora_rank: 512,
            qk_rope_head_dim: 64,
        };
        assert!(matches!(d.kv, KvStyle::Mla { .. }));
    }

    /// The set is the CALLER's, which is what moved this out of the
    /// llama lineage. A stack is servable or not against a build, and
    /// the same stack answers differently to a differently-built pie.
    #[test]
    fn the_same_stack_is_servable_by_one_build_and_not_another() {
        let mut d = stack(&[128]);
        d.shape.q_heads = 24;
        d.shape.kv_heads = 4;
        assert_eq!(d.shape.gqa_group(), 6);
        assert!(
            d.servable_by(&[1, 2, 3, 4, 8]).is_err(),
            "six is not in FlashInfer's instantiated set"
        );
        assert!(
            d.servable_by(&[1, 2, 3, 4, 6, 8]).is_ok(),
            "a build that instantiated six serves the identical stack"
        );
    }

    /// Qwen3.6-27B, which the doc names as the live proof that this is
    /// not one lineage's business: 24 over 4 reaches the same dispatch
    /// from a different generation.
    #[test]
    fn the_hybrids_group_of_six_is_refused_at_the_door() {
        let mut d = stack(&[128]);
        d.shape.q_heads = 24;
        d.shape.kv_heads = 4;
        let why = d
            .servable_by(&[1, 2, 3, 4, 8])
            .expect_err("six is unservable");
        assert!(
            matches!(why, Refusal::Unsupported(_)),
            "a build limit is Unsupported, not Malformed — the checkpoint is fine"
        );
    }

    /// The layout every deployable row in this build uses answers NOTHING.
    ///
    /// A `Some` here stops the load, so a paged layout that ever answered one
    /// would ground the entire build.
    #[test]
    fn the_paged_layout_is_the_one_that_refuses_nothing() {
        assert!(
            KvStyle::Paged.store_refusal().is_none(),
            "the pager allocates a k/v pair and a paged layout is that pair, \
             so a refusal here refuses every dense row in the catalog"
        );
    }

    /// A plan whose store exists is handed back whole.
    ///
    /// It must NOT edit the plan on the way through: a `provisioned` that
    /// reconstructed would be a second place for a deployment to be assembled.
    #[test]
    fn a_paged_plan_passes_through_provisioned_unchanged() {
        let mut paged = Deployment::empty();
        paged.layers = 3;
        paged.kv = KvStyle::Paged;
        assert_eq!(
            paged.clone().provisioned(),
            Ok(paged),
            "a paged layout has a store, so the plan is the result"
        );
    }

    /// A plan whose store does not exist draws the store's own sentence.
    #[test]
    fn an_mla_plan_is_refused_by_the_store_rather_than_by_the_family() {
        let mut mla = Deployment::empty();
        mla.kv = KvStyle::Mla {
            kv_lora_rank: 512,
            qk_rope_head_dim: 64,
        };
        let mut plane = Deployment::empty();
        plane.kv = KvStyle::CompressedPlane { ratios: vec![8, 8] };
        for (what, d) in [("MLA latent store", mla), ("compressed KV plane", plane)] {
            let Err(Refusal::Unsupported(why)) = d.provisioned() else {
                panic!("{what}: this build provisions none, so the plan is refused");
            };
            assert!(why.contains(what), "{why}");
        }
    }

    /// Both refusals name the store they cannot provision.
    ///
    /// A `Refusal` is carried across the FFI as TEXT, so the sentence is the
    /// whole diagnosis a user gets: two layouts refusing with the same words
    /// leave the reader unable to tell them apart.
    #[test]
    fn the_two_refusing_layouts_do_not_refuse_with_the_same_sentence() {
        let mla = KvStyle::Mla {
            kv_lora_rank: 512,
            qk_rope_head_dim: 64,
        };
        let plane = KvStyle::CompressedPlane {
            ratios: vec![8, 8, 8],
        };
        let say = |k: &KvStyle| k.store_refusal().expect("refuses").to_string();
        let (a, b) = (say(&mla), say(&plane));
        assert!(a.contains("MLA latent store"), "{a}");
        assert!(b.contains("compressed KV plane"), "{b}");
        assert_ne!(a, b, "two different missing stores, one sentence");
    }

    /// The variant no row in this build can produce.
    ///
    /// `driver-cuda` maps the two arms to two DIFFERENT errors: `Unsupported`
    /// is a statement about the build, `Malformed` about the checkpoint.
    /// Nothing in this crate constructs the second one, and it is kept
    /// because a checkpoint contradicting its own declared type is a real
    /// category — so its sentence is checked here rather than left unread.
    #[test]
    fn the_refusal_arm_no_row_reaches_still_says_something_a_reader_can_use() {
        let malformed = Refusal::Malformed("a stack of 0 layers").to_string();
        let unsupported = Refusal::Unsupported("a stack of 0 layers").to_string();
        assert!(
            malformed.contains("contradicts its own type"),
            "the checkpoint is at fault and the sentence must say so: {malformed}"
        );
        assert!(
            unsupported.contains("this build cannot serve"),
            "the build is at fault and the sentence must say so: {unsupported}"
        );
        assert_ne!(
            malformed, unsupported,
            "the same detail under two arms must not read identically -- the \
             arms exist so a user can tell 'fix your checkpoint' from 'this \
             binary was not built for it'"
        );
    }

    /// `full_attention_shape` answers only when the two shapes DIFFER.
    ///
    /// A `Some` makes the driver size its pages twice, so a uniform stack
    /// that answered `Some` would have it allocate a second geometry
    /// identical to the first.
    #[test]
    fn a_stack_whose_windowed_and_full_layers_agree_states_no_second_shape() {
        assert_eq!(
            stack(&[]).full_attention_shape(),
            None,
            "no layers, no shape"
        );
        assert_eq!(
            stack(&[128, 128]).full_attention_shape(),
            None,
            "one shape stated twice is not two shapes"
        );

        let mut d = stack(&[128, 128]);
        d.attention[0].window = 512;
        d.attention[0].head_dim = 64;
        d.attention[1].rotary_dim = 32;
        assert_eq!(
            d.full_attention_shape(),
            Some((128, 1, 32)),
            "the UNWINDOWED layer's shape is the one a page is sized by"
        );
    }

    /// Every layer windowed means there is no full layer to describe.
    #[test]
    fn a_stack_with_no_full_layer_at_all_states_no_second_shape() {
        let mut d = stack(&[128, 64]);
        for a in &mut d.attention {
            a.window = 512;
        }
        assert_eq!(
            d.full_attention_shape(),
            None,
            "two shapes but neither reads the whole context, so neither is \
             the one the driver's `global_*` fields mean"
        );
    }

    /// Qwen2.5-1.5B: twelve over two.
    #[test]
    fn the_other_live_example_is_refused_the_same_way() {
        let mut d = stack(&[128]);
        d.shape.q_heads = 12;
        d.shape.kv_heads = 2;
        assert_eq!(d.shape.gqa_group(), 6);
        assert!(d.servable_by(&[1, 2, 3, 4, 8]).is_err());
    }

    #[test]
    fn the_ordinary_ratios_are_served() {
        for (q, kv, group) in [(16u32, 16u32, 1u32), (16, 8, 2), (32, 8, 4), (64, 8, 8)] {
            let mut d = stack(&[128]);
            d.shape.q_heads = q;
            d.shape.kv_heads = kv;
            assert_eq!(d.shape.gqa_group(), group, "{q} over {kv}");
            assert!(d.servable_by(&[1, 2, 3, 4, 8]).is_ok(), "{q} over {kv}");
        }
    }

    /// A ratio that does not divide is not a build question.
    ///
    /// No instantiation set contains a fractional group, so widening the set
    /// cannot fix it, and the refusal says something different.
    #[test]
    fn a_fractional_ratio_is_refused_by_every_build() {
        let mut d = stack(&[128]);
        d.shape.q_heads = 14;
        d.shape.kv_heads = 4;
        assert_eq!(
            d.shape.gqa_group(),
            3,
            "the ratio TRUNCATES to a group every build instantiates, which \
             is why divisibility is asked separately"
        );
        assert!(d.servable_by(&[1, 2, 3, 4, 8]).is_err());
        assert!(
            d.servable_by(&[1, 2, 3, 4, 6, 8, 14]).is_err(),
            "widening the set cannot admit a ratio that does not divide"
        );
    }

    /// Zero kv heads is refused rather than dividing by zero.
    ///
    /// `gqa_group` answers 0 so `EMPTY` stays askable, but the refusal comes
    /// from the divisibility arm, so the sentence says the shape is wrong.
    #[test]
    fn a_stack_that_states_no_kv_heads_does_not_divide_by_zero() {
        let mut d = stack(&[128]);
        d.shape.q_heads = 8;
        d.shape.kv_heads = 0;
        assert_eq!(d.shape.gqa_group(), 0, "askable, not a panic");
        assert!(d.servable_by(&[1, 2, 3, 4, 8]).is_err());
        assert!(
            d.servable_by(&[0, 1, 2, 3, 4, 8]).is_err(),
            "not fixable by widening"
        );
    }

    /// An empty set serves nothing, which is the honest answer for a
    /// build that instantiated no decode at all.
    #[test]
    fn a_build_that_instantiated_nothing_serves_nothing() {
        let mut d = stack(&[128]);
        d.shape.q_heads = 8;
        d.shape.kv_heads = 8;
        assert!(d.servable_by(&[]).is_err());
    }

    /// A stack with no layers is a stack no build can serve, which is what
    /// `empty()` is for: every question asked of it answers without a first
    /// layer to read.
    #[test]
    fn the_empty_stack_answers_every_question_and_is_served_by_nobody() {
        let e = Deployment::empty();
        assert_eq!(e.decode_head_dims(), None);
        assert_eq!(e.full_attention_shape(), None);
        assert!(!e.shares_kv());
        assert!(e.windows().is_empty());
        assert!(e.theta_by_layer().is_empty());
        assert!(e.rotary_by_layer().is_empty());
        assert!(
            e.servable_by(&[1, 2, 4, 8]).is_err(),
            "zero kv heads is a fractional group, not a servable one"
        );
        assert!(
            !e.norm_topk_prob,
            "the routing convention no real row wants: `empty()` exists to be \
             refused, and a driver that served off it would route on \
             unnormalized weights"
        );
    }

    /// Keyed on the WINDOW: a page is sized per layer, and the layer that
    /// reads the whole context is the one that has to be right.
    #[test]
    fn the_full_layers_shape_is_reported_only_when_it_differs() {
        // Every layer unwindowed and identical: one shape serves.
        assert_eq!(stack(&[128, 128]).full_attention_shape(), None);

        // A windowed layer 0 beside an unwindowed layer 1 of the same
        // shape is still one shape.
        let mut same = stack(&[128, 128]);
        same.attention[0].window = 512;
        assert_eq!(same.full_attention_shape(), None);

        // Now the full layer is shaped differently, which is gemma-4.
        let mut differs = stack(&[128, 256]);
        differs.attention[0].window = 512;
        differs.attention[1].rotary_dim = 64;
        assert_eq!(differs.full_attention_shape(), Some((256, 1, 64)));

        // Differing only in kv heads counts too — a page is sized by
        // both, and reporting only on head_dim would bill the wrong
        // width.
        let mut kv_only = stack(&[128, 128]);
        kv_only.attention[0].window = 512;
        kv_only.attention[1].kv_heads = 4;
        assert_eq!(kv_only.full_attention_shape(), Some((128, 4, 0)));

        // A stack with no unwindowed layer has no full layer to report.
        let mut all_windowed = stack(&[128, 256]);
        for a in &mut all_windowed.attention {
            a.window = 512;
        }
        assert_eq!(all_windowed.full_attention_shape(), None);
    }

    /// The three per-layer tables do not share an elision rule.
    ///
    /// `windows` is always full length: a window of `-1` is as much a binding
    /// as `512`, and the fire path indexes it. `theta_by_layer` elides on
    /// UNIFORMITY. `rotary_by_layer` elides on the SENTINEL — `0` means
    /// "rotate the whole head", so a stack whose layers all state 64 gets a
    /// full table even though the values agree.
    #[test]
    fn the_per_layer_tables_elide_by_three_different_rules() {
        let uniform = stack(&[128, 128]);
        assert_eq!(
            uniform.windows(),
            vec![-1, -1],
            "a full-length table even when every layer agrees"
        );
        assert!(uniform.theta_by_layer().is_empty());
        assert!(uniform.rotary_by_layer().is_empty());

        let mut partial = stack(&[128, 128]);
        for a in &mut partial.attention {
            a.rotary_dim = 64;
        }
        assert_eq!(
            partial.rotary_by_layer(),
            vec![64, 64],
            "uniform but not the sentinel, so the table is stated"
        );

        let mut one_full = stack(&[128, 128]);
        one_full.attention[0].rotary_dim = 64;
        assert_eq!(one_full.rotary_by_layer(), vec![64, 0]);
    }

    /// The capability question, asked of every style this enum has.
    #[test]
    fn only_the_paged_style_has_a_store_in_this_build() {
        assert!(KvStyle::Paged.has_a_store_in_this_build());
        assert!(
            !KvStyle::Mla {
                kv_lora_rank: 512,
                qk_rope_head_dim: 64,
            }
            .has_a_store_in_this_build()
        );
        assert!(!KvStyle::CompressedPlane { ratios: vec![4, 4] }.has_a_store_in_this_build());
    }
}

/// Why a checkpoint cannot be served.
///
/// An ENUM rather than an ABI status, because this crate has no ABI: a driver
/// maps it to whatever its own boundary speaks. Both variants carry a reason,
/// because one sentence for nine unrelated causes names neither the model nor
/// the thing that is missing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {
    /// This build cannot serve the row: it has no forward text for it,
    /// or no kernel for a shape the row states.
    ///
    /// A statement about the BUILD, not about the checkpoint, which is why it
    /// is separate from [`Self::Malformed`].
    Unsupported(&'static str),
    /// A row exists and the checkpoint contradicts it.
    Malformed(&'static str),
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported(what) => write!(f, "this build cannot serve it: {what}"),
            Self::Malformed(why) => write!(f, "the checkpoint contradicts its own type: {why}"),
        }
    }
}

impl std::error::Error for Refusal {}

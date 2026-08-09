//! `Deployment` — everything a driver needs to serve a checkpoint, with
//! no family name in it.
//!
//! # Why this is a type and not a trait
//!
//! The drivers used to ask a `Box<dyn PlannedFamily>` thirteen
//! questions, and its own doc comments named the exception in the
//! method: `pins_attention_values()` said *"Only gemma-4 does"*, and
//! `decode_plan_head_dims()` existed because gemma-4's two layer kinds
//! disagree. Then the callers undid the abstraction to get the name
//! back — `let is_gemma4 = family.planless_prefill();` appears twice in
//! the CUDA shell and once in its transfer path.
//!
//! Wrapping a family name in a virtual predicate and then recovering the
//! name at the call site means **the axis was the family all along**.
//!
//! It also cost at run time. `facts_from_hf` was called from the
//! admission of EVERY fire, allocating a box and cloning per-layer
//! `Vec`s — while the lowering it feeds is cached precisely because it
//! costs 3.3 ms. The expensive answer was cached; its input was
//! rederived.
//!
//! # What changes about gemma-4
//!
//! Its two head dims stop being an exception and become
//! `attention[l].head_dim` differing between layers, which is what they
//! are. A `Vec` of per-layer facts has no opinion about which family
//! produced it.
//!
//! # Why it lives HERE
//!
//! `crates/model/tests/one_normalizer.rs` states the rule: *"what a
//! driver reads is the answer, never the question."* The drivers obeyed
//! the letter — they read `pie.model/1`, not `config.json` — while the
//! answer was still SHAPED like the question and they still switched on
//! it: 33 `FACTS_ROWS` rows and 11 derivations in the CUDA shell alone,
//! against the 25 `model_type` conditionals of the C++ normalizer that
//! test was written to hunt.
//!
//! A `Deployment` with no family name in it is a type a driver
//! **cannot** branch on. That is the difference between a guard that
//! must be remembered and one that cannot be routed around.

use std::collections::BTreeMap;

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
    /// Its own index for the ordinary case. Gemma-4's trailing layers
    /// name an earlier one and own no pages themselves — which is a
    /// fact about a LAYER, and was a fact about a family.
    pub kv_source: u32,
    /// The attention scale. Usually `1/sqrt(head_dim)`; gemma-4 runs
    /// `1.0` because its q/k norms carry the scaling.
    pub sm_scale: f32,
    /// Rope base for this layer. The stacks that use one theta
    /// throughout repeat it.
    pub rope_theta: f32,
    /// Rotary width, or `0` for full rotation at the head dim.
    pub rotary_dim: u32,
}

/// What kind of KV this deployment needs.
///
/// AN ENUM, so a shape the driver has no pool for is an
/// `unimplemented!` ARM rather than a row in a registry that loads
/// successfully and dies at its first fire. That was a real defect:
/// the MLA lineage registered in `FACTS_ROWS`, answered `facts_from_hf`
/// happily, and had no forward path at all.
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
    /// DeepSeek-V4's per-layer compression ratios.
    Dsv4 {
        /// One ratio per layer; `None` for an uncompressed layer.
        ratios: Vec<i32>,
    },
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
/// It is on the deployment rather than inside a family's facts because
/// a DRIVER needs it: an adapter's staging reads the projection input,
/// and which buffer that is depends on this. `Pre` ships one input
/// norm; `Post` (olmo2) ships `post_attention` and `post_feedforward`
/// instead.
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
/// Nine numbers. `driver-cuda` read every one of them off `model.hf` —
/// a `HfConfig` kept resident purely so a kernel could ask how many
/// heads there are — and these nine are exactly what it asked for.
/// Nothing here is derived: they are the row's own numbers, in the
/// row's own units, with the one exception named on
/// [`Self::head_dim_kernel`].
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
    /// The ONE derived number, and it is here because sixteen executor
    /// sites re-derived it from config and a shape is not something a
    /// boolean gives: phi-3's 96-wide heads run on the 128-wide kernel,
    /// so a buffer sized `heads * head_dim` is too small by a third.
    /// Equal to [`Self::head_dim`] when nothing was padded, so a reader
    /// that wants "the width to allocate" can always use this one.
    pub head_dim_kernel: u32,
    /// The dense MLP's inner width.
    pub intermediate: u32,
    /// One EXPERT's inner width, or `0` for a dense stack.
    ///
    /// Separate from [`Self::intermediate`] because the two are
    /// genuinely different numbers on a mixture and the forward
    /// workspace is ONE buffer both layer kinds share — so what it must
    /// hold is the wider of them. A planner given only the dense width
    /// under-sizes a mixture whose experts are wider, which does not
    /// fail: it moves bytes out of the KV pool, quietly.
    pub moe_intermediate: u32,
    /// The logit dimension.
    ///
    /// The MODEL's `vocab_size` and never the tokenizer's token count —
    /// they differ (qwen3: 151 669 against 151 936) and using the
    /// smaller one is the vocab-padding device fault.
    pub vocab: u32,
}

impl Geometry {
    /// The zeros that go with [`Deployment::empty`].
    ///
    /// A shape no fire can take, so a driver that forgot to fill it in
    /// refuses at its first admission rather than serving a stack it
    /// never derived.
    pub const EMPTY: Self = Self {
        hidden: 0,
        q_heads: 0,
        kv_heads: 0,
        head_dim: 0,
        head_dim_kernel: 0,
        intermediate: 0,
        moe_intermediate: 0,
        vocab: 0,
    };

    /// Heads per KV group — the GQA ratio a decode kernel is
    /// instantiated for.
    ///
    /// Zero KV heads answers 0 rather than dividing, because
    /// [`Self::EMPTY`] must be askable.
    #[must_use]
    pub const fn gqa_group(&self) -> u32 {
        if self.kv_heads == 0 { 0 } else { self.q_heads / self.kv_heads }
    }

    /// The width to ALLOCATE for one head: the kernel's, when one was
    /// instantiated wider than the checkpoint's.
    #[must_use]
    pub const fn head_dim_alloc(&self) -> u32 {
        if self.head_dim_kernel > self.head_dim { self.head_dim_kernel } else { self.head_dim }
    }

    /// The widest MLP any layer in the stack asks for.
    ///
    /// The forward workspace is sized from this, and it is a `max`
    /// rather than a choice because a mixture's layers share the buffer
    /// with its dense ones.
    #[must_use]
    pub const fn widest_mlp(&self) -> u32 {
        if self.moe_intermediate > self.intermediate { self.moe_intermediate } else { self.intermediate }
    }
}

/// How a stack rescales its rope frequency ladder, for the stacks that
/// do.
///
/// # Why this is a `Deployment` field and not a derivation
///
/// It was neither, for a while, and that was a bug with a blast radius.
/// `driver-metal` read four numbers off the `pie.model/1` descriptor —
/// `rope_scaling.{factor, low_freq_factor, high_freq_factor,
/// original_max_position_embeddings}` — and built its decode ladder from
/// them. Deleting the descriptor deleted the only path those numbers
/// travelled, and `DecodeGeometry` kept the four fields while nothing
/// filled them: every Llama-3.1/3.2/3.3 would have run with a factor of
/// zero, which the derivation reads as "no rescaling". That model does
/// not fail. It attends with the wrong wavelengths past its original
/// 8192 and degrades — fluently, which is the worst way.
///
/// The row states it because it is a per-CHECKPOINT fact and not a
/// per-generation one: Llama-3.2's 1B and 3B rescale by `32.0` where
/// 3.1's 8B and 70B rescale by `8.0`, from the same `rope_theta` and the
/// same original context. A generation constant would have to be wrong
/// for one of them.
///
/// `None` is a statement, not a default: it says this stack uses its
/// `rope_theta` ladder unrescaled, which is what every Qwen, Gemma,
/// Mistral, Phi and OLMo-2 row means.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RopeScaling {
    /// Llama-3's piecewise rescaling (`rope_type: "llama3"`).
    ///
    /// Wavelengths shorter than the high-frequency cut pass through
    /// untouched, those longer than the low-frequency cut are divided by
    /// `factor`, and the band between them interpolates. The two cuts
    /// are expressed as divisors of `original_max_position`, which is
    /// why they are factors and not lengths.
    Llama3 {
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
        /// The logit scale. HF computes `0.1 * ln(factor) + 1` when a
        /// config omits it, which is exactly what OLMo-3 states
        /// explicitly — so a row that omits it is stating the same
        /// number the formula gives, and
        /// `an_omitted_attention_factor_is_the_formula_not_a_guess`
        /// checks that against OLMo-3's published value.
        attention_factor: f32,
        /// The context the checkpoint was trained at.
        original_max_position: u32,
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
    /// Beside [`Self::shape`] and for the same reason: the launch path
    /// read it off a resident `HfConfig`. A CONSTANT of the checkpoint —
    /// `1e-6` for most of the llama lineage, `1e-5` for the qwen-2
    /// generation, `1e-6` for gemma — and one no tensor extent carries,
    /// so a row must state it.
    pub norm_eps: f32,
    /// The geometry a driver's LAUNCH path reads.
    ///
    /// Here for a reason worth stating, because it looks like scope
    /// creep and is the opposite. `driver-cuda`'s `fire/launch.rs` read
    /// thirty of these numbers off a parsed `config.json` it was holding
    /// (`model.hf.num_attention_heads`, `model.hf.head_dim_kernel`,
    /// `model.hf.intermediate_size`, …) — which meant the driver kept a
    /// whole normalized config resident for the life of a model, and
    /// meant a fire's geometry came from a DIFFERENT reading of the
    /// checkpoint than the one its trace was built from. Two readers,
    /// one document, no one holding them together: the same shape of
    /// defect as the three registries.
    ///
    /// It is a projection of the row like everything else here, so a
    /// launch and a trace cannot disagree about how many heads there
    /// are.
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
    /// Per-layer-embedding width, `0` for a stack without one.
    pub ple_dim: i32,
    /// Where the norm sits — read by anything that needs to name the
    /// projection input, which today is the adapter staging.
    pub norm: NormPlacement,
    /// Named scalar constants the forward refers to by name.
    pub scales: BTreeMap<String, f32>,
    /// What a driver ADVERTISES about this model, as distinct from what
    /// it needs to fire it.
    pub advertised: Advertised,
    /// How the rope ladder is rescaled, `None` for the stacks that use
    /// their [`LayerAttention::rope_theta`] ladder as-is.
    ///
    /// Beside `rope_theta` in meaning but not in placement, because it
    /// is a property of the STACK: no checkpoint here rescales one layer
    /// and not another, while gemma-3 and gemma-4 genuinely do run two
    /// different bases. Putting it per-layer would invite a shape no
    /// published model has.
    ///
    /// See [`RopeScaling`] for the regression that put it here.
    pub rope_scaling: Option<RopeScaling>,
    /// The media encoders this row ships, empty for a text-only stack.
    pub towers: Towers,
}

/// The encoder stacks that run BESIDE the decoder.
///
/// A tower is not a layer of the model: it takes waveform or pixels and
/// hands back rows the decoder embeds, on its own kernels, at its own
/// depth. It is here because a `Deployment` is what a driver fires and
/// the driver fires these too — the alternative was the resident
/// `HfConfig` these 21 numbers used to be read from, which is the thing
/// this refactor exists to delete.
///
/// Both are `Option` and both are `None` on every text-only row, which
/// is the ordinary case. A driver's encode entry refuses on `None`
/// rather than defaulting, because a default tower would be a *plausible
/// shape for a stack that does not exist* — the exact failure mode the
/// old `GemmaAudioConfig::default()` had, where a checkpoint with no
/// audio block was handed gemma-4's own 12 layers.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Towers {
    /// The audio front-end, `None` when the row ships none.
    pub audio: Option<AudioTower>,
    /// The vision front-end, `None` when the row ships none.
    pub vision: Option<VisionTower>,
}

/// A conformer audio encoder's shape.
///
/// Fourteen numbers, which is every field of the old
/// `GemmaAudioConfig` that a driver actually read. The fifteenth was
/// `use_clipped_linears`, parsed, normalized, carried across the process
/// boundary and read by nobody.
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
/// Seven numbers, again the ones a driver reads. `patch_size`,
/// `head_dim`, `num_key_value_heads`, `soft_tokens_per_image` and
/// `use_clipped_linears` were parsed and never asked for; the patch grid
/// and token count that the host DOES need come from
/// [`crate::multimodal`], which computes them rather than being told.
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
/// Here rather than on [`Variant`](crate::catalog::Variant) as three
/// more required methods, and here rather than left where they were,
/// which was a resident `HfConfig` the driver kept for the life of a
/// load in order to answer them. Those three reads — `model_type`,
/// `max_position_embeddings`, and whether `gemma_vision`/`gemma_audio`
/// are present — were the last thing keeping a parsed `config.json`
/// alive inside `driver-cuda`, and the last two are why the
/// 845-line normalizer could not be deleted.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Advertised {
    /// The family label a GUEST PROGRAM matches on.
    ///
    /// Not a dispatch key — that is the whole point of the catalog —
    /// but still a real value, because `engine`'s `model.arch_name()` is
    /// a host function inferlets call and `VisionArch::from_arch_name`
    /// selects an image front-end from it. It is a FAMILY, deliberately
    /// coarser than an id: `qwen3` names twelve checkpoints of six
    /// shapes, and a program asking "is this a gemma" wants the coarse
    /// answer.
    ///
    /// Stated by the row rather than read off a config, so the string a
    /// program sees and the row a driver loaded cannot be about
    /// different models.
    ///
    /// Matched WHOLE by every consumer. `VisionArch::from_arch_name`
    /// used to accept a substring, which made `qwen3` — this
    /// generation's own label — select the Qwen3-VL front-end belonging
    /// to `qwen3_5`. A coarse label is still an exact one.
    pub arch: &'static str,
    /// The published context ceiling, `0` when the row does not state
    /// one.
    ///
    /// A TRAINING-time fact and not a deployment one, which is why it
    /// sits here and not beside [`Deployment::shape`]: nothing in a fire
    /// reads it, and a driver serving a shorter context than this is
    /// serving correctly.
    pub max_model_len: u32,
    /// Whether this row ships a tower the driver's encode entry point
    /// serves.
    ///
    /// The bug this replaced: it was hardwired `false` while four GPU
    /// tests fired the entry point and passed. The worker refuses to
    /// build an encode executor at all when this is clear, so gemma-4's
    /// vision and audio towers — ported, bound, and matching HF's
    /// embeddings to cosine — were unreachable through the engine. The
    /// tests never saw it because they call the entry directly, which is
    /// exactly the seam a capability is supposed to cover.
    ///
    /// Qwen3-VL is deliberately NOT this: its tower writes into the
    /// fire's hidden rows rather than handing host rows back, so it is
    /// an in-fire path and not an encode one.
    pub media_encode: bool,
}

impl Deployment {
    /// A placeholder for a driver that must build its model value before
    /// it can derive one.
    ///
    /// Zero layers, which is a shape no fire can take — so a driver that
    /// forgot to fill it in refuses at its first admission rather than
    /// serving a stack it never derived.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            layers: 0,
            norm_eps: 0.0,
            shape: Geometry::EMPTY,
            attention: Vec::new(),
            kv: KvStyle::Paged,
            recurrent: None,
            prefill: PrefillStyle::Planned,
            attn_output: AttnOutput::StatedArgs,
            logit_softcap: 0.0,
            ple_dim: 0,
            norm: NormPlacement::Pre,
            scales: BTreeMap::new(),
            advertised: Advertised::default(),
            rope_scaling: None,
            towers: Towers::default(),
        }
    }

    /// The distinct decode head dims this stack needs plans for.
    ///
    /// `None` when every layer agrees, which is the ordinary case.
    /// `Some((a, b))` when two kinds disagree — which used to be
    /// `decode_plan_head_dims()`, a method that existed because
    /// gemma-4 has two, and is now a question about the `Vec`.
    #[must_use]
    pub fn decode_head_dims(&self) -> Option<(u32, u32)> {
        let first = self.attention.first()?.head_dim;
        let other = self.attention.iter().find(|a| a.head_dim != first)?.head_dim;
        Some((first, other))
    }

    /// Does any layer read another's KV pages?
    #[must_use]
    pub fn shares_kv(&self) -> bool {
        self.attention.iter().enumerate().any(|(l, a)| a.kv_source as usize != l)
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
    use super::*;

    fn layer(head_dim: u32) -> LayerAttention {
        LayerAttention {
            head_dim,
            window: -1,
            kv_source: 0,
            sm_scale: 1.0,
            rope_theta: 10_000.0,
            rotary_dim: 0,
        }
    }

    fn stack(dims: &[u32]) -> Deployment {
        Deployment {
            layers: dims.len() as u32,
            // The two the launch path reads. This helper existed before
            // either was a field and went on compiling only because the
            // struct had not grown them yet; they are stated here rather
            // than defaulted so a `stack()` is a whole `Deployment` and
            // the tests below exercise the same value a driver holds.
            norm_eps: 1e-5,
            shape: Geometry::EMPTY,
            attention: dims
                .iter()
                .enumerate()
                .map(|(l, &d)| LayerAttention { kv_source: l as u32, ..layer(d) })
                .collect(),
            kv: KvStyle::Paged,
            recurrent: None,
            prefill: PrefillStyle::Planned,
            attn_output: AttnOutput::DriverPinned,
            logit_softcap: 0.0,
            ple_dim: 0,
            norm: NormPlacement::Pre,
            scales: BTreeMap::new(),
            advertised: Advertised::default(),
            rope_scaling: None,
            towers: Towers::default(),
        }
    }

    /// The exception that stopped being one. `decode_plan_head_dims()`
    /// existed as a vtable method because gemma-4 has two layer kinds;
    /// it is a question about a `Vec` now, and a uniform stack answers
    /// `None` without anyone having to know which family it is.
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

    /// The MLA orphan, as a type. It used to be a row in `FACTS_ROWS`
    /// that loaded successfully and died at its first fire; a driver
    /// matching on `KvStyle` has to write the arm or refuse.
    #[test]
    fn an_unservable_kv_shape_is_a_variant_rather_than_a_registry_row() {
        let mut d = stack(&[128]);
        d.kv = KvStyle::Mla { kv_lora_rank: 512, qk_rope_head_dim: 64 };
        assert!(matches!(d.kv, KvStyle::Mla { .. }));
    }
}

/// Why a checkpoint cannot be served.
///
/// An ENUM rather than an ABI status, because this crate has no ABI. A
/// driver maps it to whatever its own boundary speaks — which is the
/// point of §4: the derivation used to return `PIE_STATUS_UNSUPPORTED`,
/// the engine's vocabulary, from a crate that has no engine.
///
/// Both variants carry a reason, and `Unsupported` did not used to. It
/// was returned from nine sites in the old derivation and reached the
/// operator as "no deployment derivation for this model type" — a
/// sentence that names neither the model nor the thing that is missing,
/// for nine unrelated causes (a KV store this build did not
/// instantiate, a family with no forward text, an expert bank the
/// tracer has no kernel for). A row that refuses now has to say what it
/// wanted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {
    /// This build cannot serve the row: it has no forward text for it,
    /// or no kernel for a shape the row states.
    ///
    /// A statement about the BUILD, not about the checkpoint — the
    /// checkpoint is fine and a differently-configured pie would serve
    /// it. That is why it is separate from [`Self::Malformed`].
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

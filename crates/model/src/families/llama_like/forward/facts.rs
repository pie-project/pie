//! `llama_like`'s per-backend binding facts.
//!
//! The SHAPE moved to `../spec.rs` (ungated: a row is written in it).
//! What a deployment BOUND -- a fused bank, a kernel's padded head dim,
//! a TP width -- is per-backend and known only when that backend's
//! aspect is compiled, so it stays here.

use serde::{Deserialize, Serialize};
// The shared vocabulary stayed with the toolchain -- more than one family
// is written in these words.
pub use model_compiler::facts::{NormPlacement, QkNorm};
use model_compiler::dsl::WeightRepr;
use model_compiler::trace::{NormVariant, RopeKind};

/// The shape, re-exported so a declaration reaches its facts and the
/// words they are stated in from one place.
pub use super::super::spec::LlamaLikeFacts;


/// CUDA backend facts for a LOWERED llama_like trace
/// (`family::llama_like_cuda`; north-star-dsl.md).
///
/// Everything here is load-time — env defaults, kernel-support predicates
/// over the head geometry, what the deployment's binding materialized —
/// exactly the terms `context.cpp:1419` and the executor's path booleans
/// derive today. The declaration's class arms consume them the way they
/// consume model facts; the driver's job shrinks to VALIDATING at boot
/// that its own derivation agrees (the `declared_facts.cpp` pattern), not
/// choosing.
///
/// Like every fact struct: measured per deployment, provenance-pinned in
/// the constructors, never silently derived.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlamaLikeCudaFacts {
    /// XQA decode eligibility: `PIE_CUDA_XQA_DECODE` (default on) &&
    /// `xqa_decode_bf16_supported(heads, head_dim_kernel, page_size,
    /// window)` && all-full-attention && native-bf16 cache && !HND layout
    /// (context.cpp:1419-1425, llama_like.cpp:693-701).
    pub xqa_decode: bool,
    /// The fused decode-QKV epilogue is live: `decode_fused_post_enabled`
    /// (env, default on) && native-bf16 cache && unpadded head_dim &&
    /// no qkv bias — the load-time terms of `fused_decode_qkv_post`
    /// (declared_forward.cpp:465-479). The trace-time terms (`fused_qkv`,
    /// per-head qk-norm, Standard rope) live on [`LlamaLikeFacts`] and
    /// the declaration checks both.
    pub decode_fused_post: bool,
    /// The workspace carries a rope table (`ws.rope_table` non-empty), so
    /// the fused arm's first layer states [`model_compiler::trace::OpKind::RopeTableBuild`];
    /// without it the fused kernel derives cos/sin from theta in-kernel
    /// and no table launch exists.
    pub rope_table: bool,
    /// FlashInfer's decode kernel set lacks this model's GQA ratio
    /// (`!flashinfer_decode_supports_gqa`, context.cpp:1413-1414): decode
    /// fires fall back to dequant + the prefill kernel
    /// ([`model_compiler::trace::AttnKernel::PrefillDequantDecode`]). XQA, when
    /// eligible, overrides this (context.cpp:1427).
    pub force_prefill_path: bool,
    /// The attention kernels run at a padded `head_dim_kernel` wider than
    /// the logical head dim (Phi-3-mini: 96 → 128). A load-time fact
    /// (`cfg.head_dim != cfg.head_dim_kernel`): the generated form stages
    /// zero-padded q/k/v copies around the KV write, overrides the
    /// softmax scale to `1/sqrt(d)`, and strips the attention output —
    /// the hand-written `head_dim_padded` branches, resolved at emission.
    /// Serde-defaulted (append-only discipline).
    #[serde(default)]
    pub head_dim_padded: bool,
    /// The head width the ATTENTION kernels run at (Phi-3-mini: 128 for
    /// a logical 96), or 0 for a deployment that runs at the logical
    /// one.
    ///
    /// [`Self::head_dim_padded`] is exactly `head_dim_kernel != 0`, and
    /// both are here because the bool is in the digest and the WIDTH is
    /// what a statement needs: `cuda::pad_head_dim` produces a value
    /// whose shape is `heads * head_dim_kernel`, and a shape is not
    /// something a boolean gives. Sixteen executor sites read the bool
    /// and re-derived the width from config; the pads and the strip are
    /// statements now, so the width crosses with them.
    ///
    /// Serde-defaulted (append-only discipline).
    #[serde(default)]
    pub head_dim_kernel: u32,
    /// The checkpoint materialised a packed gate‖up bank
    /// (`w.layers[l].gate_up_proj_fused != nullptr`), so the MLP's packed
    /// GEMM lands in one buffer and the activation is the CHUNKED swiglu
    /// over it; without it the projection writes two buffers and the
    /// activation is the pair form.
    ///
    /// A pure WEIGHT-BINDING fact, which is why it belongs here rather
    /// than in the walk. The executor derived it per layer as
    /// `gate_up_proj_fused != nullptr && !ws.gate_up_fused.empty()`, but
    /// the second term is dead: `workspace.cpp` allocates that buffer
    /// unconditionally ("Always allocated … lets the forward dispatch
    /// decide per layer"). So the binding alone decides, the binding is
    /// known at model construction, and the taxonomy's first row applies
    /// — a load-time fact is a trace-time `match`, erased.
    ///
    /// Stating it deletes a runtime branch from three places at once: the
    /// interpreter's `arm_swiglu`, the generated `.inc`'s
    /// `if (gate_up_fused_N)` per layer, and the flat list's residue.
    /// Serde-defaulted (append-only discipline).
    #[serde(default)]
    pub gate_up_fused: bool,
    /// How this deployment STORES its linear projections — the weight
    /// representation axis ([`model_compiler::dsl::WeightRepr`]).
    ///
    /// A pure binding fact, like [`Self::gate_up_fused`], and the last
    /// one the driver was answering for itself: `make_weight_view` built
    /// a `WeightView` out of a per-layer `QuantMeta` the statement never
    /// mentioned and `gemm::act_x_w` routed on it — ten call sites here
    /// and eight in qwen3.5, every one of them the driver knowing
    /// something the declaration did not.
    ///
    /// ONE repr for the whole deployment rather than one per projection,
    /// because a checkpoint quantizes uniformly and the build gate
    /// refuses a mixed binding by name. Where a checkpoint ever does
    /// mix, this becomes a field per projection and the text asks the
    /// facts per handle — nothing else changes, which is the point of
    /// putting the axis on the WEIGHT.
    ///
    /// Serde-defaulted to dense (append-only discipline), so a fixture
    /// written before this field reads exactly as it did.
    #[serde(default)]
    pub proj_repr: WeightRepr,
    /// How many ranks this deployment shards its layers across
    /// (`LlamaLikeForwardCfg::tp_size`), or 0/1 for a single GPU.
    ///
    /// SHARDING NEEDS NO VOCABULARY: a rank's trace states ITS widths,
    /// and the text divides by this the way it divides by anything
    /// else. What needs vocabulary is the point where the shards are
    /// recombined, because that is a launch — `dist::all_reduce_bf16`
    /// and its two friends, which the text states.
    ///
    /// So this fact does two things and neither is a switch the driver
    /// reads: it narrows the projection widths, and it decides whether
    /// the landing statements exist at all.
    ///
    /// Serde-defaulted (append-only discipline); 0 reads as one rank.
    #[serde(default)]
    pub tp_size: u32,
    /// The SLIDING WINDOW each layer attends over, `-1` for none.
    ///
    /// Empty means every layer is `-1` — a deployment with no window at
    /// all — which is why the accessor and not the field is what texts
    /// read ([`Self::window_left_at`]).
    ///
    /// A load-time fact: a config's `sliding_window`, or its per-layer
    /// list where the architecture alternates (OLMo-3, Mistral). Eleven
    /// executor sites across four families derived it by reaching into
    /// `fwd_cfg.per_layer_window_left` — a per-layer array no statement
    /// mentioned — and the dispatch statements carry it now.
    ///
    /// The per-FIRE override (`runtime_window_left`) is NOT this. That
    /// is a runtime input and wants a guard predicate;
    /// `DeclineReason::SlidingWindow` still names it.
    ///
    /// Serde-defaulted (append-only discipline).
    #[serde(default)]
    pub window_left: Vec<i32>,
    /// Rows below which an all-reduce takes the NVLink P2P kernel
    /// instead of NCCL, or 0 for a deployment that always takes NCCL.
    ///
    /// `NcclComm::all_reduce_bf16` asks `can_handle(bytes)` and routes
    /// on the answer — a driver picking between two implementations,
    /// which is what this replaces. The text states the pair as a guard
    /// and this is its predicate's payload.
    ///
    /// It is a ROW count and the kernel's test is BYTES, which is the
    /// same question once `hidden` is known: a row is `hidden` bf16
    /// elements. Converting here rather than in the arm is the point —
    /// a load-time fact becomes a trace-time constant.
    ///
    /// ZERO also covers the deployment that registered no P2P buffers.
    /// The kernel reads only registered memory, which is a placement
    /// fact rather than a size one, and a deployment that has none has
    /// no threshold either.
    ///
    /// Serde-defaulted (append-only discipline).
    #[serde(default)]
    pub all_reduce_p2p_max_rows: u32,
}

/// The METAL backend's load-time facts — what the Metal deployment
/// resolved before any fire, the way [`LlamaLikeCudaFacts`] carries
/// CUDA's.
///
/// UNVERIFIED (2026-08-05): no Metal deployment has produced these yet.
/// The Metal driver cannot even build on the box we have (`xcrun --find
/// metal` fails — the shader compiler ships with full Xcode), so every
/// field here is read off the driver's SOURCE
/// (`crates/driver-metal/csrc/src/batch/decode_psos.cpp`, `model/qwen3_5/decode_step.hpp`)
/// rather than measured. `.wiki/tart/macos.md` records the ladder; the
/// precedent for refusing to call an unmeasured fact set measured is
/// [`Qwen35CudaFacts::qwen3_5_0_8b_synthetic`].
/// WHICH gated activation a deployment takes. See
/// [`LlamaLikeMetalFacts::activation`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub enum Activation {
    /// `silu_mul` — `silu(gate) * up`, every llama-like deployment's.
    #[default]
    SiluMul,
    /// `gptoss_swiglu` — the gate clamped ABOVE only, the linear branch
    /// clamped both ways and carrying a `+1`.
    SwiGlu {
        /// The clamp.
        limit: f32,
        /// The sigmoid's slope. NOT from a config: it is part of the
        /// activation the way `silu`'s sigmoid is.
        alpha: f32,
    },
    /// `geglu_tanh` — gemma's, and the gelu is the TANH approximation.
    Geglu,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlamaLikeMetalFacts {
    /// The projection GEMV folds the block residual in its epilogue
    /// (`affine_qmv_fast_residual`, `Dispatch::fuse_residual`,
    /// `PIE_FUSE_RESIDUAL`), so a `beta_one` matmul states one launch
    /// instead of a projection plus a `residual_add`.
    pub fuse_residual_gemv: bool,
    /// The M>1 lane addresses the KV cache through a page table
    /// (`sdpa_paged_decode` + `kv_append_paged`) rather than the M=1
    /// contiguous pair (`sdpa_vector_decode` + `kv_append`).
    pub paged_multi_batch: bool,
    /// The M>1 projections take MLX's steel quantized GEMM
    /// (`affine_qmm_t`) instead of the GEMV — the driver's
    /// `kQmmMinBatch` gate, as a load-time fact of the deployment.
    pub qmm_multi_batch: bool,
    /// How this deployment's projections are STORED.
    ///
    /// Same role `LlamaLikeCudaFacts::proj_repr` plays and for the same
    /// reason: `LlamaLikeFacts::shape()` answers `Bf16` because the semantic
    /// facts carry no backend, and a trace with no backend cannot name the
    /// kernel a scaled weight needs. The backend facts do carry one, so this
    /// is where the representation reaches the namespace.
    ///
    /// It is load-bearing beyond the kernel name: an affine kernel reads
    /// THREE tensors, and `MatW::scale_names` is what makes the statement say
    /// so. A text that left this dense named `affine_qmv_fast` while stating
    /// one weight, and the driver would have had to derive the other two from
    /// a naming convention it was never told.
    ///
    /// Serde-defaulted, so a fixture written before this field reads as it did.
    #[serde(default)]
    pub proj_repr: model_compiler::dsl::WeightRepr,
    /// Bits per packed weight element — 4 or 8.
    ///
    /// The affine entrypoints are instantiated over `(group size × bit
    /// width)`, so the SYMBOL a statement names carries both. `proj_repr`
    /// already carries the group; this is the other half, and it is a separate
    /// field because `WeightRepr::Scaled` has nowhere to put it (a bit width
    /// is a property of the weight's dtype, which the trace does not spell).
    ///
    /// This is the `_d_256` lesson generalised: a symbol whose axis point is
    /// wrong does not fail, it reads the wrong bytes. A symbol that is a bare
    /// STEM does not resolve at all, which is the better failure and the one
    /// the runtime compiler reports by listing what the shader does export.
    #[serde(default)]
    pub affine_bits: u32,
    /// How this deployment stores its EXPERT BANKS, when that is not how it
    /// stores its dense projections.
    ///
    /// The DSL's own note beside `proj_repr` says "a checkpoint quantizes
    /// uniformly", and that is very nearly true. `mlx-community/
    /// gpt-oss-20b-MXFP4-Q4` is the exception: its `quantization` block names
    /// 98 tensors as `affine`/64/4 and leaves the expert banks OUT, so they
    /// take the top-level default -- `mxfp4`, group **32**. One checkpoint,
    /// two formats.
    ///
    /// Reading the banks with the dense format is not a near miss. Every scale
    /// comes from the wrong offset, and bf16 garbage is NaN more often than
    /// not: the fire bound everything, ran, and produced 909,207 NaNs
    /// beginning at the first routed projection of layer 0.
    ///
    /// `None` is "the same as `proj_repr`", which is every other checkpoint.
    #[serde(default)]
    pub moe_repr: Option<model_compiler::dsl::WeightRepr>,
    /// [`Self::affine_bits`] for the expert banks; see [`Self::moe_repr`].
    #[serde(default)]
    pub moe_bits: u32,
    /// The GEMM's `(row tile, column tile)`, as the entrypoint spells them.
    ///
    /// `affine_qmm_t` is instantiated over `(group × bits × bm × bn)`, so the
    /// batched projection's symbol carries a TILE — and a tile is chosen from
    /// the ROW COUNT, which is a fire-time fact a trace cannot know.
    ///
    /// So it is a load-time fact instead, and that is the honest reading of
    /// what the driver was doing: `qmm_bm` picks the widest rung at or under
    /// `n`, and a deployment that always fires the same window always picks
    /// the same one. A deployment that wants the tile chosen per fire needs
    /// the row count on a guard axis, which is a change to `Row` and not to
    /// this text.
    ///
    /// Serde-defaulted to `(0, 0)`, which spells no tile — right for the
    /// GEMV-only deployments, wrong loudly for a GEMM one, because a symbol
    /// with no tile does not resolve.
    #[serde(default)]
    pub qmm_tile: (u32, u32),
    /// The deployment bound ONE packed `gate‖up` bank.
    ///
    /// A binding fact, exactly as `LlamaLikeFacts::fused_qkv` is, and the
    /// Metal answer to it is normally **false**: `compile_load_plan` authors
    /// with `Projections::InPlace`, and `dense_fused_projection_joins` returns
    /// before doing anything under that policy. So the MLX path publishes
    /// `mlp.gate_proj` and `mlp.up_proj` separately and the text must state
    /// two projections.
    ///
    /// It matters more than a name: `mlp/gated.metal::silu_mul` takes **gate
    /// and up as two buffers**, so a text that states one packed value binds
    /// the OUTPUT where `up` belongs and leaves the output unbound. That is a
    /// fire that runs.
    #[serde(default)]
    pub gate_up_fused: bool,
    /// `rms_norm_eps`, the epsilon every norm of this deployment carries.
    ///
    /// A load-time fact and not a constant: it comes from the checkpoint's
    /// config, and the shader takes it as a field of `RmsParams` rather than
    /// baking one in. A norm handed zero divides by the root of the mean
    /// square alone, which for a near-zero row is an infinity the next kernel
    /// spreads everywhere.
    #[serde(default)]
    pub rms_eps: f32,
    /// `rope_theta`, the rotary base.
    ///
    /// Stated rather than defaulted for the reason `ModelFacts::rope_theta`
    /// gives about its own: a reader that only knows the flat key finds
    /// nothing on a config that nests it, silently keeps its default, and the
    /// rotated channels come out wrong in a way that compounds layer over
    /// layer until the activations saturate.
    #[serde(default)]
    pub rope_theta: f32,
    /// The rotary base the SLIDING layers take, when a deployment states a
    /// second one, or zero for a stack whose layers all share [`Self::rope_theta`].
    ///
    /// gemma-4 states both — `rope_parameters` gives `full_attention` a base
    /// of 1e6 and `sliding_attention` a base of 1e4 — and it is not a corner:
    /// gemma-4-31b slides fifty of its sixty layers, so the single-base
    /// reading was wrong on 83% of the stack. Two orders of magnitude apart,
    /// which is not a near miss; the rotation is wrong from the second
    /// channel on and compounds layer over layer.
    ///
    /// Keyed off [`Self::window_left_at`] rather than a second per-layer list,
    /// because "does this layer slide" is already answered there and two
    /// lists could disagree.
    #[serde(default)]
    pub rope_theta_sliding: f32,
    /// The per-head width the FULL-attention layers use, or zero for a stack
    /// whose layers all share one.
    ///
    /// gemma-4's `global_head_dim`. Its full layers are twice as wide per
    /// head as its sliding ones and carry a quarter the KV heads, which the
    /// checkpoint states in its tensors: on the 31b, layer 0's `q_norm` is
    /// `[256]` and layer 5's is `[512]`.
    ///
    /// Nothing had ever read it, so ten of that model's sixty layers ran at
    /// half their Q width and a quarter past the end of their K.
    #[serde(default)]
    pub global_head_dim: u32,
    /// The key/value head count the FULL-attention layers use, or zero for
    /// one shape everywhere. See [`Self::global_head_dim`].
    #[serde(default)]
    pub global_kv_heads: u32,
    /// What fraction of each FULL-attention head the rotation covers, or zero
    /// for a deployment that rotates the whole head.
    ///
    /// gemma-4's `partial_rotary_factor: 0.25`: its full layers rotate 128 of
    /// their 512 channels and leave the rest, while its sliding layers rotate
    /// all 256 of theirs. The rotation's extent is therefore a per-layer-type
    /// fact like the head shape, and it reaches the grid rather than the
    /// kernel — `Rule::Rope` launches half of it.
    #[serde(default)]
    pub full_partial_rotary: f32,
    /// Whether the FULL-attention layers take V from the K projection.
    ///
    /// PER LAYER, and measured: `mlx-community/gemma-4-26b-a4b-it-4bit` ships
    /// no `v_proj` for layers 5, 11, 17, 23 and 29 — every sixth, which is
    /// exactly the layers `window_left` marks as full attention. A text that
    /// stated a `v_proj` for one of them would name a tensor that is not
    /// there.
    ///
    /// It also reorders the two norms on those layers: V reads the projection
    /// K's norm is about to overwrite, so V goes first.
    ///
    /// A bool and not a list because the layers it applies to are already
    /// stated — `window_left_at(l) < 0` is the full-attention test — and two
    /// lists that must agree is one more thing to keep agreeing.
    #[serde(default)]
    pub v_from_k: bool,
    /// Whether the mixture sits BESIDE the dense MLP rather than replacing it.
    ///
    /// gemma4's. Both branches read the post-attention residual and their
    /// results are added — five norms round one block. Every other deployment
    /// this text serves runs one FFN or the other, which is why this is a fact
    /// and not the shape of the walk.
    #[serde(default)]
    pub dense_beside_moe: bool,
    /// Whether each layer scales the stream by a learned SCALAR.
    ///
    /// gemma's, for a deployment with no per-layer embeddings: one number per
    /// layer, read from a buffer rather than stated, because which layer is
    /// running is the fire's and not the text's.
    #[serde(default)]
    pub per_layer_scalar: bool,
    /// gemma's PER-LAYER EMBEDDING width, or zero for a deployment with none.
    ///
    /// A SIDE NETWORK: a second embedding table gathered once per step,
    /// projected, normed and joined into `[n_layers, ple_dim]` that each layer
    /// reads its own slice of. Nothing llama-like has a counterpart, which is
    /// why gemma4 needs a text where qwen3-moe and gpt-oss needed a fixture.
    #[serde(default)]
    pub per_layer_emb_dim: u32,
    /// Layers at the END of the stack that SHARE their KV with an earlier one.
    ///
    /// A shared layer rotates its own Q and reads the pages its source wrote:
    /// no k/v projection, no k/v norm, no append. Suppressing those dispatches
    /// is not an optimisation — it is which tensors the checkpoint ships.
    #[serde(default)]
    pub kv_shared_layers: u32,
    /// The readout's SOFTCAP — `cap * tanh(x / cap)` — or zero for none.
    ///
    /// gemma's. Zero is "no softcap" and the text names nothing, rather than
    /// passing a cap so large it does nothing: that would be a kernel run per
    /// fire to compute the identity.
    #[serde(default)]
    pub logit_softcap: f32,
    /// Whether every layer carries an attention SINK.
    ///
    /// A per-head learned logit that joins the softmax without a value behind
    /// it, so a sinked attention normalizes over one more term than it sums.
    /// gpt-oss's; asked of the TENSORS, like the other binding facts.
    #[serde(default)]
    pub attn_sinks: bool,
    /// WHICH gated activation this deployment takes.
    ///
    /// Three symbols, not one with flags, and the difference is not
    /// cosmetic: gpt-oss clamps the gate ABOVE only, clamps the linear branch
    /// both ways and adds one to it; gemma's gelu is the TANH approximation
    /// rather than the erf one. Dropping any of that produces a model that
    /// runs and is wrong, which is why a text names a symbol and a fact
    /// chooses.
    ///
    /// Serde-defaulted (append-only discipline); the default is `silu_mul`,
    /// every llama-like deployment's.
    #[serde(default)]
    pub activation: Activation,
    /// Whether this deployment's rotary frequencies come from a TABLE.
    ///
    /// True for a config that rescales its ladder -- llama-3's `rope_scaling`
    /// with `rope_type: llama3`, YaRN, and anything else that is not a plain
    /// geometric series in a base. A `rope_theta` cannot express those, so a
    /// text that stated one would rotate by the wrong frequencies from the
    /// second channel on, at every position but zero.
    ///
    /// The table itself is the DRIVER's: derived at load from the config and
    /// answered as `Source::RopeFrequencies`. This fact only says which form
    /// the statement takes.
    #[serde(default)]
    pub rope_freq_table: bool,
    /// The SLIDING WINDOW each layer attends over, `-1` for none.
    ///
    /// The same load-time fact [`LlamaLikeCudaFacts::window_left`] carries,
    /// and stated here rather than shared because the two halves are
    /// independently deserialized: a Metal deployment that alternates windows
    /// (OLMo-3, Mistral) has to say so on its own side.
    ///
    /// Empty means every layer is `-1`. Read through
    /// [`Self::window_left_at`].
    #[serde(default)]
    pub window_left: Vec<i32>,
}

impl LlamaLikeMetalFacts {
    /// gpt-oss-20b's Metal facts. A SYNTHETIC fixture like `synthetic`.
    #[must_use]
    pub fn gpt_oss_20b() -> Self {
        Self {
            // Every layer carries one learned logit per head.
            attn_sinks: true,
            // `swiglu_limit: 7.0`, and alpha is the activation's own constant.
            activation: Activation::SwiGlu {
                limit: 7.0,
                alpha: 1.702,
            },
            // `rope_theta: 150000`, a plain geometric ladder.
            rope_theta: 150_000.0,
            rope_freq_table: false,
            rms_eps: 1e-5,
            // `sliding_window: 128`, ALTERNATING: every other layer attends
            // the window and the rest attend everything. `window_left_at`
            // reads the list per layer, which is what the accessor is for.
            window_left: (0..24).map(|l| if l % 2 == 0 { 128 } else { -1 }).collect(),
            ..Self::synthetic()
        }
    }

    /// The three gemma facts that ARE facts, on an otherwise llama-like
    /// deployment.
    ///
    /// The PLE and the KV sharing are here too, which makes this the fixture
    /// a gemma4 text reads. What it is NOT is a measurement: the widths are
    /// plausible rather than any published config's, and a real gemma4
    /// deployment states its own.
    #[must_use]
    pub fn gemma_like() -> Self {
        Self {
            activation: Activation::Geglu,
            logit_softcap: 30.0,
            per_layer_emb_dim: 256,
            kv_shared_layers: 4,
            dense_beside_moe: true,
            window_left: (0..24).map(|l| if l % 6 == 5 { -1 } else { 512 }).collect(),
            rope_theta: 1_000_000.0,
            // The SLIDING layers' base. gemma states both, and this fixture
            // slides twenty of its twenty-four layers.
            rope_theta_sliding: 10_000.0,
            ..Self::synthetic()
        }
    }

    /// This layer's window, `-1` for all of it. See [`Self::window_left`].
    pub fn window_left_at(&self, l: u32) -> i32 {
        model_compiler::facts::window_left_at(&self.window_left, l)
    }

    /// This layer's rotary base. See [`Self::rope_theta_sliding`].
    pub fn rope_theta_at(&self, l: u32) -> f32 {
        if self.rope_theta_sliding > 0.0 && self.window_left_at(l) >= 0 {
            self.rope_theta_sliding
        } else {
            self.rope_theta
        }
    }

    /// Whether layer `l` attends the WHOLE context.
    ///
    /// The one question the two per-layer-type facts below are keyed on, so
    /// they cannot disagree with each other or with the window list.
    pub fn is_full_attention(&self, l: u32) -> bool {
        self.window_left_at(l) < 0
    }

    /// This layer's per-head width, given the deployment's usual one.
    ///
    /// gemma-4 states TWO. Measured on `gemma-4-31b-it-4bit`'s own tensors:
    /// layer 0 (sliding) has `q_norm [256]` and `q_proj [8192, …]` = 32x256,
    /// while layer 5 (full) has `q_norm [512]` and `q_proj [16384, …]` =
    /// 32x512. `global_head_dim: 512` is the config saying so.
    ///
    /// Zero means one shape for the whole stack, which is every family here
    /// but gemma-4.
    pub fn head_dim_at(&self, l: u32, sliding: u32) -> u32 {
        if self.global_head_dim > 0 && self.is_full_attention(l) {
            self.global_head_dim
        } else {
            sliding
        }
    }

    /// This layer's key/value head count. See [`Self::head_dim_at`] — the
    /// 31b's full layers carry 4 where its sliding ones carry 16
    /// (`k_proj [2048, …]` = 4x512 against `[4096, …]` = 16x256).
    pub fn kv_heads_at(&self, l: u32, sliding: u32) -> u32 {
        if self.global_kv_heads > 0 && self.is_full_attention(l) {
            self.global_kv_heads
        } else {
            sliding
        }
    }

    /// How many of this layer's channels the rotation covers.
    ///
    /// The whole head everywhere but gemma-4's full-attention layers, which
    /// state `partial_rotary_factor` and rotate a fraction. Rounded DOWN to
    /// an even number because the rotation pairs channels — `Rule::Rope`
    /// launches `rotary/2` and an odd width would leave a lane without its
    /// partner.
    pub fn rotary_dim_at(&self, l: u32, head_dim: u32) -> u32 {
        let dim = self.head_dim_at(l, head_dim);
        if self.full_partial_rotary <= 0.0 || !self.is_full_attention(l) {
            return dim;
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_precision_loss)]
        let want = (f64::from(dim) * f64::from(self.full_partial_rotary)) as u32;
        (want.min(dim) / 2 * 2).max(2)
    }

    /// A SYNTHETIC fixture, not a measurement — see the struct comment.
    /// These are the driver's own defaults as its source reads them.
    pub fn synthetic() -> Self {
        Self {
            fuse_residual_gemv: true,
            paged_multi_batch: true,
            qmm_multi_batch: true,
            // The symbols this text names are the affine ones
            // (`affine_qmv_fast`, `embed_gather_4bit`), so the deployment
            // they describe is a quantized checkpoint. MLX stores the pair
            // beside the packed weight as `.scales` and `.biases`, which is
            // a zero-point layout.
            proj_repr: model_compiler::dsl::WeightRepr::Scaled {
                layout: model_compiler::dsl::ScaleLayout::PerGroup,
                group: 64,
                axis: 0,
                zero_point: true,
            },
            affine_bits: 4,
            // Uniform: the synthetic fixture's banks are the dense format.
            moe_repr: None,
            moe_bits: 0,
            // The narrowest rung `qmm_bm` can pick, so it is the one a short
            // window fires; `bn = 32` is the only column tile the residual
            // variant is instantiated at.
            qmm_tile: (16, 32),
            // `Projections::InPlace` is what `compile_load_plan` authors with,
            // and the join declines under it.
            gate_up_fused: false,
            // qwen3's `rms_norm_eps`.
            rms_eps: 1e-6,
            // qwen3's `rope_theta`. The shader raises TWO to a base, so the
            // statement hands it `log2(theta)`; handing theta rotates by a
            // frequency ladder that is wrong from the second channel on.
            rope_theta: 1_000_000.0,
            // qwen3 states ONE base for every layer, which is what zero means.
            rope_theta_sliding: 0.0,
            // And ONE attention shape for every layer, likewise, rotating
            // the whole head.
            global_head_dim: 0,
            global_kv_heads: 0,
            full_partial_rotary: 0.0,
            // qwen3's mixture replaces its dense MLP rather than sitting
            // beside it, and it has no per-layer embeddings or scalar and
            // shares no KV.
            // qwen3 projects its own V.
            v_from_k: false,
            dense_beside_moe: false,
            per_layer_scalar: false,
            per_layer_emb_dim: 0,
            kv_shared_layers: 0,
            // qwen3 caps no logits, has no attention sinks, and takes the
            // plain gated activation.
            logit_softcap: 0.0,
            attn_sinks: false,
            activation: Activation::SiluMul,
            // qwen3's ladder is a plain geometric series in `rope_theta`.
            rope_freq_table: false,
            // qwen3 attends over the whole context at every layer.
            window_left: Vec::new(),
        }
    }
}

impl LlamaLikeCudaFacts {
    /// The window layer `l` attends over, `-1` for none.
    ///
    /// A short list is not an error: a deployment whose config carries
    /// ONE `sliding_window` states a one-element list and every layer
    /// reads it, which is what the drivers' `per_layer_window_left`
    /// fallback meant.
    pub fn window_left_at(&self, l: u32) -> i32 {
        model_compiler::facts::window_left_at(&self.window_left, l)
    }

    /// Qwen3-0.6B on L40S, default env — MEASURED 2026-08-02 against the
    /// driver's own derivation via the rung-3 digest print
    /// (`PIE_DECLARED_FORWARD_TRACE=1` + `..._GENERATED=1`; the live
    /// digest is the provenance):
    /// `.../qk1/fq1/te0/xqa0/dfp1/rt1/fpp0`.
    ///
    /// `xqa_decode: false` — the first version of this constructor
    /// guessed `true` from the geometry and called it measured; the
    /// digest mechanism caught the lie on its first live run
    /// (`fwd_cfg.use_xqa_decode` derives false on this deployment).
    /// The decode class therefore states the FlashInfer decode kernel.
    /// Note the deployment's BINDING also unties the lm_head
    /// (`w.lm_head != w.embed`, live `te0`) even though the checkpoint's
    /// config ties it — the model-facts fixture [`LlamaLikeFacts::qwen3_0_6b`]
    /// keeps the config-level fact for the semantic goldens; emission
    /// against the live deployment overrides it (`emit-cuda`).
    pub fn qwen3_0_6b_l40s() -> Self {
        Self {
            xqa_decode: false,
            decode_fused_post: true,
            rope_table: true,
            force_prefill_path: false,
            head_dim_padded: false,
            head_dim_kernel: 0,
            // The loader's `dense_fused_projection_joins` contract packs
            // BF16 dense groups and declines quantized ones, so a plain
            // BF16 deployment carries the bank. VERIFIED LIVE, not
            // assumed — the boot log's declared-facts line is what says
            // so, and the digest refuses the deployment if this fixture
            // and the binding disagree.
            gate_up_fused: true,
            // Dense. The same contract line says so: a group it packs is
            // a BF16 one, so a deployment that carries the bank cannot
            // be quantized.
            proj_repr: WeightRepr::Bf16,
            // One rank: the L40S fixture is a single GPU.
            tp_size: 1,
            // qwen3-0.6B attends over the whole prefix.
            window_left: Vec::new(),
            // Single GPU: no collective, so no threshold.
            all_reduce_p2p_max_rows: 0,
        }
    }
}


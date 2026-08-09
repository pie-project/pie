//! The shared projection every llama-lineage generation's rows go
//! through.
//!
//! Twelve `model_type` strings dispatched to one derivation
//! (`llama_like_facts_from_hf`), and the reason was never that twelve
//! families are alike — it is that they are ONE family whose rows differ
//! in their numbers. Qwen3, Llama 3, Mistral, OLMo, Phi-3, Gemma 3 and
//! the Qwen mixtures are all `LlamaLikeFacts`; what a generation module
//! adds is which chat template speaks for it and which author writes its
//! contract, and those are the two things a shape cannot state.
//!
//! So the projections live here, once, taking a `&LlamaLikeFacts`, and a
//! generation's `impl Variant` calls them. That is the same N:1 the old
//! `HF_ROWS` column expressed — spelled as a call rather than as a table
//! nothing held to the other two tables.

use crate::catalog::{Backend, Deployed, MetalBinding};
use crate::deployment::{
    Advertised,
    AttnOutput, Deployment, Geometry, KvStyle, LayerAttention, NormPlacement, PrefillStyle,
};
use crate::manifest::{Manifest, TensorSpec};

use super::spec::LlamaLikeFacts;

use model_compiler::facts::{NormPlacement as SpecNorm, QkNorm};

/// The attention head dims a CUDA build instantiates.
///
/// `kernels.def`'s `PIE_ATTN_HEAD_DIM` rows. It is a property of the
/// BINARY, not of any checkpoint, which is why a row does not state it
/// and why it was excluded from the descriptor when there was one.
pub const ATTN_HEAD_DIMS: &[u32] = &[64, 128, 256, 512];

/// Smallest instantiated head dim that can hold `head_dim`, or
/// `head_dim` itself when none can — the caller then surfaces the
/// dispatch error rather than silently mis-sizing.
#[must_use]
pub fn round_up_attn_head_dim(head_dim: u32) -> u32 {
    ATTN_HEAD_DIMS.iter().copied().filter(|&d| d >= head_dim).min().unwrap_or(head_dim)
}

/// The GQA group sizes a CUDA decode instantiates.
///
/// FlashInfer's decode reports anything else by THROWING, and a throw
/// crossing a C ABI is undefined behaviour. This was
/// `refuse_unservable_gqa`, and it sat inside the llama lineage's
/// derivation as though it were a property of that lineage. It is a
/// property of the BUILD — every family reaching the same dispatch is
/// subject to the same instantiation set — so it is stated here as a
/// build capability and asked by [`Deployment::servable_by`].
pub const DECODE_GQA_GROUPS: &[u32] = &[1, 2, 3, 4, 8];

/// This row's tensors.
///
/// Every extent is the row's own arithmetic, which is what makes the
/// manifest a check rather than a second statement: `q_proj` is
/// `[q_heads * head_dim, hidden]` because that is what `q_heads` and
/// `head_dim` MEAN.
#[must_use]
pub fn manifest(f: &LlamaLikeFacts) -> Manifest {
    let (hidden, vocab) = (u64::from(f.hidden), u64::from(f.vocab));
    let (q, kv) = (u64::from(f.q_width()), u64::from(f.kv_width()));
    let head_dim = u64::from(f.head_dim);
    let dense = f.n_experts == 0;

    Manifest::new(f.layers)
        .with(TensorSpec::required("embed_tokens", [vocab, hidden]))
        .with(TensorSpec::required("norm", [hidden]))
        // TIED vs UNTIED as presence, which is the only way a manifest
        // can tell them apart: every extent agrees.
        .either(!f.tied_embeddings, "lm_head", [vocab, hidden])
        .with(TensorSpec::required("layer.{}.self_attn.q_proj", [q, hidden]))
        .with(TensorSpec::required("layer.{}.self_attn.k_proj", [kv, hidden]))
        .with(TensorSpec::required("layer.{}.self_attn.v_proj", [kv, hidden]))
        .with(TensorSpec::required("layer.{}.self_attn.o_proj", [hidden, q]))
        // The q/k-norm question the old derivation answered by dividing
        // a byte count: per-head ships `[head_dim]`, global ships the
        // whole projection width, and off ships nothing.
        .with(match f.qk_norm {
            QkNorm::Off => TensorSpec::absent("layer.{}.self_attn.q_norm"),
            QkNorm::PerHead => TensorSpec::required("layer.{}.self_attn.q_norm", [head_dim]),
            QkNorm::Global => TensorSpec::required("layer.{}.self_attn.q_norm", [q]),
        })
        // Pre-norm ships one input norm per sub-layer; post-norm
        // (olmo2/olmo3) ships the pair that follows them instead. This
        // is the `ends_with("input_layernorm.weight")` alias probe, as
        // an expectation.
        //
        // The DISCRIMINATING pair is `input_layernorm` and
        // `post_feedforward_layernorm`, and not the name that reads as
        // though it were: HuggingFace calls the norm in front of the
        // MLP `post_attention_layernorm` because of where it sits in
        // the block, not because of which placement it belongs to, and
        // EVERY row here ships it. A manifest that made it the marker
        // for `Post` refused llama-3.2 on the one tensor llama-3.2
        // certainly has.
        .either(
            f.norm_placement == SpecNorm::Pre,
            "layer.{}.input_layernorm",
            [hidden],
        )
        .with(TensorSpec::required("layer.{}.post_attention_layernorm", [hidden]))
        .either(
            f.norm_placement == SpecNorm::Post,
            "layer.{}.post_feedforward_layernorm",
            [hidden],
        )
        .with_if(f.qkv_bias, TensorSpec::required("layer.{}.self_attn.q_proj.bias", [q]))
        .with_if(
            dense,
            TensorSpec::required("layer.{}.mlp.gate_proj", [u64::from(f.intermediate), hidden]),
        )
        .with_if(
            dense,
            TensorSpec::required("layer.{}.mlp.down_proj", [hidden, u64::from(f.intermediate)]),
        )
        .with_if(!dense, TensorSpec::required("layer.{}.mlp.gate", [u64::from(f.n_experts), hidden]))
        .with_if(!dense, TensorSpec::present("layer.{}.mlp.experts.0.gate_proj"))
}

/// This row's deployment.
///
/// A projection, and a short one: every value below was already in the
/// row. The eleven-function derivation it replaces read the same numbers
/// out of a parsed `config.json`, one family at a time.
#[must_use]
pub fn deployment(
    f: &LlamaLikeFacts,
    rope_theta: f32,
    norm_eps: f32,
    sliding_window: i32,
) -> Deployment {
    let head_dim = round_up_attn_head_dim(f.head_dim).max(f.head_dim);
    let attention = (0..f.layers)
        .map(|l| LayerAttention {
            // One shape for every layer, which is what this row was
            // already saying by having no per-layer count.
            kv_heads: f.kv_heads,
            head_dim,
            window: sliding_window,
            // Every layer owns its pages. KV sharing is gemma-4's, and
            // it is a fact about a LAYER there rather than a family
            // here.
            kv_source: l,
            sm_scale: 1.0 / (head_dim as f32).sqrt(),
            rope_theta,
            // Full rotation at the head dim.
            rotary_dim: 0,
        })
        .collect();
    Deployment {
        layers: f.layers,
        norm_eps,
        // The row's own numbers, so a LAUNCH and a TRACE cannot
        // disagree about how many heads there are. `driver-cuda` read
        // thirty of these off a resident `HfConfig`, which is a second
        // reading of the same document.
        shape: Geometry {
            hidden: f.hidden,
            q_heads: f.q_heads,
            kv_heads: f.kv_heads,
            head_dim: f.head_dim,
            head_dim_kernel: round_up_attn_head_dim(f.head_dim),
            intermediate: f.intermediate,
            moe_intermediate: f.moe_intermediate,
            experts_per_token: f.experts_per_token,
            shared_intermediate: f.shared_intermediate,
            vocab: f.vocab,
        },
        attention,
        kv: KvStyle::Paged,
        recurrent: None,
        prefill: PrefillStyle::Planned,
        // The guard region records no SSA output for this text, so the
        // driver owns the landing buffer. `pins_attention_values()`
        // defaulted to true with the doc "Only gemma-4 does" — the
        // exception is gemma-4's row, and it says so there.
        attn_output: AttnOutput::DriverPinned,
        logit_softcap: 0.0,
        ple_dim: 0,
        norm: match f.norm_placement {
            SpecNorm::Post => NormPlacement::Post,
            SpecNorm::Pre | SpecNorm::Sandwich => NormPlacement::Pre,
        },
        // `false` even for the `Sandwich` arm above, and that is the whole
        // point of the field. The sandwich is an ARCHITECTURE and the
        // offset is a WEIGHT CONVENTION; every stack this shared
        // projection serves stores the multiplier directly, and the
        // gemmas that do not have projections of their own that say so.
        norm_unit_offset: false,
        v_norm: false,
        k_eq_v: false,
        mlp_gate: crate::deployment::MlpGate::Silu,
        scales: std::collections::BTreeMap::new(),
        // Filled by the ROW, not by the shape: a family label and a
        // published context ceiling are facts about a checkpoint, and a
        // projection only sees geometry.
        advertised: Advertised::default(),
        rope_scaling: None,
        towers: Default::default(),
    }
}

/// The CUDA binding facts for this row.
///
/// Every field the old derivation hardcoded is hardcoded here, in one
/// place, with the two that are not constants — the padded head dim and
/// the TP width — coming from the row and the load respectively.
#[cfg(feature = "forward")]
#[must_use]
pub fn cuda_facts(
    f: &LlamaLikeFacts,
    load: Deployed<'_>,
) -> super::forward::facts::LlamaLikeCudaFacts {
    let kernel = round_up_attn_head_dim(f.head_dim);
    super::forward::facts::LlamaLikeCudaFacts {
        xqa_decode: false,
        decode_fused_post: false,
        rope_table: true,
        force_prefill_path: false,
        head_dim_padded: kernel != f.head_dim,
        head_dim_kernel: if kernel == f.head_dim { 0 } else { kernel },
        gate_up_fused: true,
        proj_repr: model_compiler::dsl::WeightRepr::Bf16,
        tp_size: load.tp_size.max(1),
        window_left: Vec::new(),
        all_reduce_p2p_max_rows: 0,
    }
}

/// The GEMM tile this build's Metal shaders were STAMPED at.
///
/// A property of the BINARY and not of any checkpoint, exactly as
/// [`ATTN_HEAD_DIMS`] is. `affine_qmm_t` is instantiated over
/// `(group × bits × bm × bn)`, so a batched projection's SYMBOL carries
/// a tile; `16` is the narrowest row rung `qmm_bm` can pick — the one a
/// short window fires — and `32` is the only column tile the residual
/// variant is instantiated at. Neither number is a row's to state nor a
/// load's to observe, because the shaders were stamped before either
/// existed, which is why this is a `const` here and not a field of
/// [`MetalBinding`].
///
/// A symbol whose tile is WRONG does not fail: it reads the wrong bytes.
/// A symbol with no tile at all does not resolve, which is the better
/// failure and the one the runtime compiler reports by listing what the
/// shader does export — which is why `qmm_tile` is stated rather than
/// left at the serde default of `(0, 0)`.
#[cfg(feature = "forward")]
pub const QMM_TILE: (u32, u32) = (16, 32);

/// Why a SHARDED load has no Metal text here.
///
/// A `const` so the test that asserts the refusal names the missing
/// thing compares against the same sentence the operator is shown,
/// rather than against a paraphrase that can drift away from it — the
/// discipline `csm::project::NO_TRACE` sets.
#[cfg(feature = "forward")]
pub const NO_METAL_SHARD: &str = "this Metal load states a tensor-parallel width above one and \
     `LlamaLikeMetalFacts` has no shard vocabulary: the CUDA facts carry \
     a `tp_size` that narrows every projection width in the text, and the \
     Metal ones carry nothing, so the text would state the WHOLE model's \
     widths against one rank's slice of the weights and read past the end \
     of every projection. Refused rather than traced, because a shard \
     read at full width is arithmetic that runs";

/// The head widths `sdpa_paged.metal` instantiates.
///
/// A row's `head_dim` names the symbol directly — the `_d_256` literal
/// that strode 128-wide heads was fixed by reading it — so a width the
/// shader never compiled names a symbol nothing exports. That used to be
/// left to fail at pipeline construction, and phi-3's 96 is why it
/// cannot be: `model-compiler` validates a declaration against the
/// kernel rows and PANICS on an undeclared launch, so the row never
/// reaches the driver to be refused by name.
///
/// The set is `sdpa_paged_decode`'s own axis, minus the page-shape tails
/// (`_p32`, `_p32_sg8`), which are points of a different axis.
#[cfg(feature = "forward")]
pub const METAL_SDPA_HEAD_DIMS: &[u32] = &[64, 128, 256, 512];

/// [`METAL_SDPA_HEAD_DIMS`] as a refusal.
#[cfg(feature = "forward")]
pub const NO_METAL_HEAD_DIM: &str = "this row's heads are a width `sdpa_paged.metal` does not \
     instantiate: the shader compiles the paged decode at 64, 128, 256 \
     and 512, and the text names `sdpa_paged_decode_bfloat16_d_<width>` \
     from the row. CUDA pads such a row to `head_dim_kernel` and strips \
     the pad back off; Metal has no pad kernel in the text, so the \
     choices here are a symbol no shader exports or an attention that \
     reads 32 columns of whatever the loader staged next. Refused";

/// The one affine point `quant/qmv.metal` instantiates the ROUTED
/// matvec at.
///
/// `AffineQ::group_size` is a template constant, so `affine_qmv_routed`
/// exists at exactly `(64, 4)` and a second point would name an
/// instantiation that dequantises at 64 whatever its name claimed —
/// reading every scale from the wrong offset, which is the 909,207-NaN
/// defect the MXFP4 arm was added to fix. MXFP4 is a different codec
/// rather than another point and has its own symbol at group 32.
#[cfg(feature = "forward")]
pub const METAL_ROUTED_AFFINE: (u32, u32) = (64, 4);

/// [`METAL_ROUTED_AFFINE`] as a refusal.
#[cfg(feature = "forward")]
pub const NO_METAL_ROUTED_ENCODING: &str = "this row's expert bank reached the device at an affine \
     point `quant/qmv.metal` does not instantiate the routed matvec at: \
     the shader compiles `affine_qmv_routed` only at group 64 / 4 bits, \
     because `AffineQ::group_size` is a template constant. A bank at \
     another group dequantised by that kernel reads every scale from the \
     wrong offset and answers bf16 garbage, which is NaN more often than \
     not. Refused";

/// The row's four numbers a Metal text reads and [`LlamaLikeFacts`]
/// does not hold.
///
/// They are gathered here rather than added to the shape because twelve
/// generations share that struct and a field on it is a field every one
/// of them restates. Each generation already holds these four for
/// [`deployment`] — which is the point: a row that FIRES at one theta
/// and PAGES at another would be two readings of one checkpoint, and
/// two readings of one document is the defect the catalog exists to
/// end.
///
/// # Why the CUDA side needs none of them
///
/// `llama_like_cuda` states no epsilon and no rotary base at all: the
/// CUDA driver carries both on its own `fwd_cfg` and the text never
/// names them. `llama_like_metal` names all three — its norms take
/// `RmsParams`, its rope takes a base, and its attention takes a
/// window — because a Metal statement carries every scalar its kernel
/// reads. That asymmetry is not a gap in one of the texts. It is what
/// "a text is written for a backend" means.
#[cfg(feature = "forward")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MetalRow {
    /// `rope_theta`, the rotary base every layer of this row rotates at.
    ///
    /// Stated rather than defaulted for the reason `ModelFacts::rope_theta`
    /// gives about its own: a reader that only knows the flat key finds
    /// nothing on a config that nests it, silently keeps its default, and
    /// the rotated channels come out wrong in a way that compounds layer
    /// over layer until the activations saturate.
    pub rope_theta: f32,
    /// `rms_norm_eps`, the epsilon every norm of this row carries.
    ///
    /// A norm handed zero divides by the root of the mean square alone,
    /// which for a near-zero row is an infinity the next kernel spreads
    /// everywhere. The generation states it — most of them once, as a
    /// constant, because every published config of the generation agrees.
    pub norm_eps: f32,
    /// The window every layer attends over, `-1` for the whole context.
    ///
    /// ONE number because this family's rows state one: the schedules
    /// that ALTERNATE are gemma's and gpt-oss's, and each of those has a
    /// projection of its own that expands its own rule. A row that
    /// alternated and stated a single width here would attend the whole
    /// context on every layer or a window on every layer, and both read
    /// as a working model.
    pub window: i32,
    /// The row states a rope RESCALING, so no base expresses its ladder.
    ///
    /// llama-3's piecewise `rope_scaling` and OLMo-3's YaRN. The table
    /// itself is the DRIVER's — derived at load and answered as
    /// `Source::RopeFrequencies` — and this only says which form the
    /// statement takes. Nothing carried it for the length of this
    /// refactor: `driver-metal` read the four numbers off the deleted
    /// `pie.model/1` descriptor, and a factor of zero reads as "no
    /// rescaling", so every llama-3 would have attended past its trained
    /// 8192 with the wrong wavelengths — degrading rather than failing.
    pub rope_rescaled: bool,
}

/// The METAL binding facts for this row.
///
/// The twin of [`cuda_facts`], and the split is the whole point: six
/// fields come from `bind` because a LOAD observed them, one is this
/// build's stamp ([`QMM_TILE`]), and every remaining field is the ROW's
/// — projected here rather than sniffed from tensors by a driver.
///
/// # What this replaces
///
/// `driver-metal`'s `facts_from_with`, which rebuilt the model's own
/// facts from nine tensor probes: whether a `pre_feedforward_layernorm`
/// exists decided the norm placement AND the `(1 + w)` fold, whether a
/// `q_norm` exists decided the qk-norm, whether a `layer_scalar` exists
/// decided gemma's per-layer scale, and `embed_scale` was keyed off the
/// first of those. Every one of those questions is answered by the row
/// that matched the checkpoint, and answering them twice is how a
/// gemma came to be read as a llama — the `(1 + w)` became `w` and two
/// norms per layer were dropped, with nothing faulting.
///
/// # The zeros are STATED
///
/// Nine of the fields below are gemma's or gpt-oss's and are zero or
/// false for every row this projection serves. They are written out one
/// at a time with the reason, and [`LlamaLikeMetalFacts`] deliberately
/// has no `Default` impl to fall through to, for the reason the
/// fixtures give: a row is a MEASUREMENT of a real checkpoint, and
/// "this one has no per-layer embeddings" is part of the measurement. A
/// default body is a claim about every row that has not been written
/// yet.
///
/// [`LlamaLikeMetalFacts`]: super::forward::facts::LlamaLikeMetalFacts
#[cfg(feature = "forward")]
#[must_use]
pub fn metal_facts(
    f: &LlamaLikeFacts,
    row: MetalRow,
    load: Deployed<'_>,
    bind: &MetalBinding,
) -> super::forward::facts::LlamaLikeMetalFacts {
    // The LOAD's, and unread here: `LlamaLikeMetalFacts` has no shard
    // field, so a Metal deployment is one rank by construction and
    // `trace` refuses a load that says otherwise ([`NO_METAL_SHARD`])
    // rather than letting this projection quietly drop the width. The
    // parameter stays for the reason `csm::project::deployment` keeps
    // its own: the day a Metal collective lands, this grows a body and
    // its callers do not move.
    let _ = load;
    super::forward::facts::LlamaLikeMetalFacts {
        // ── what the LOAD observed ──
        //
        // Six fields, and the shortness of this list is the same point
        // `Deployed`'s is. Three are about the kernels this build
        // compiled and three about the bytes this checkpoint shipped;
        // no row can state either kind.
        fuse_residual_gemv: bind.fuse_residual_gemv,
        paged_multi_batch: bind.paged_multi_batch,
        qmm_multi_batch: bind.qmm_multi_batch,
        // The checkpoint's own affine format. MLX stores the pair beside
        // the packed weight as `.scales` and `.biases`, which is a
        // zero-point layout, and the GROUP is asked of the load rather
        // than inferred from a tensor's shape because g64/b8 and g128/b4
        // pack to identical extents. A pipeline built for the wrong
        // point returns fluent nonsense rather than failing.
        proj_repr: model_compiler::dsl::WeightRepr::Scaled {
            layout: model_compiler::dsl::ScaleLayout::PerGroup,
            group: bind.quant_group,
            axis: 0,
            zero_point: true,
        },
        affine_bits: bind.quant_bits,
        // The expert bank's OWN format, when the loader left it in the
        // checkpoint's MXFP4. `None` is "the same as the dense
        // projections", which is every checkpoint this family serves —
        // and reading a bank with the dense format is not a near miss:
        // every scale comes from the wrong offset, and the fire that did
        // it produced 909,207 NaNs beginning at the first routed
        // projection of layer 0.
        moe_repr: bind
            .moe_mxfp4
            .then_some(model_compiler::dsl::WeightRepr::Mxfp4Marlin),
        // FOUR by the format's own definition — MXFP4 is a four-bit
        // mantissa under a shared block exponent — so it is a constant
        // and not a second thing for the load to observe. It is read
        // only when `moe_repr` is `Some`, and stating it beside the
        // repr rather than folding it in is what `affine_bits` does one
        // field up, for the reason stated there: `WeightRepr::Scaled`
        // has nowhere to put a bit width.
        moe_bits: 4,
        // ── what this BUILD stamped ──
        qmm_tile: QMM_TILE,
        // ── what the ROW states ──
        //
        // FALSE, and it is the row's answer rather than a policy: this
        // family's manifest states `mlp.gate_proj` and `mlp.up_proj` as
        // two tensors, so a checkpoint that matched it published two.
        // `silu_mul` takes gate and up as TWO buffers, so a text that
        // stated one packed value would bind the OUTPUT where `up`
        // belongs and leave the output unbound — a fire that runs. The
        // Metal text refuses the packed arm at trace time for exactly
        // that reason.
        gate_up_fused: false,
        // FALSE for the same reason and by the same evidence.
        // `lowering::resolve` states it outright -- "`qkv` and `gate_up`
        // are FUSED handles, and no Metal deployment has them" -- because
        // `compile_load_plan` authors with `Projections::InPlace` and
        // `dense_fused_projection_joins` returns before doing anything
        // under that policy.
        //
        // The row's own `fused_qkv` says `true` on all eight llama-3 rows,
        // and that is not wrong: its doc calls it "a *binding* fact, not
        // an architecture fact", and the binding it was written against is
        // CUDA's. A binding fact read off the row is a fact one backend
        // stated for all of them, which is why the answer belongs here
        // beside `gate_up_fused` and not there.
        qkv_fused: false,
        // The row's epsilon and the row's rotary base, carried on
        // [`MetalRow`] because `LlamaLikeFacts` states neither — see
        // that struct for why a shape shared by twelve generations
        // holds no scalar a `Deployment` already carries, and why the
        // Metal text needs both anyway. They are the two numbers the
        // deleted `facts_from_with` read off a `DecodeGeometry` that had
        // read them off a `config.json`, which is a checkpoint
        // describing itself a second time.
        rms_eps: row.norm_eps,
        rope_theta: row.rope_theta,
        // ZERO, which is "one base for every layer". Two bases are
        // gemma's — `rope_parameters` gives `full_attention` 1e6 and
        // `sliding_attention` 1e4 — and a gemma states them through a
        // projection of its own. Two orders of magnitude apart is not a
        // near miss: the rotation is wrong from the second channel on
        // and compounds layer over layer.
        rope_theta_sliding: 0.0,
        // ZERO for all three, which is "one attention shape for the
        // whole stack". gemma-4's full layers are twice as wide per head
        // as its sliding ones, carry a quarter the KV heads and rotate
        // only 128 of their 512 channels; every row this projection
        // serves states ONE `head_dim` and ONE `kv_heads`, and its
        // manifest is that claim — `q_proj` is `[q_heads * head_dim,
        // hidden]` at every layer or the checkpoint is not this row.
        global_head_dim: 0,
        global_kv_heads: 0,
        full_partial_rotary: 0.0,
        // FALSE: every row here projects its own V. The manifest
        // requires `layer.{}.self_attn.v_proj` at every layer, so a
        // checkpoint that took V from K — gemma-4's full layers ship no
        // `v_proj` at all — could not have matched this row.
        v_from_k: false,
        // FALSE, and STATED rather than left off. `v_norm` is a norm
        // with NO WEIGHT, so there is no tensor to ask about it and
        // nothing in a checkpoint could ever have contradicted a wrong
        // answer here — every `has_tensor` question about it answers no,
        // correctly and uselessly. It is gemma-4's alone, and gemma-4
        // does not trace Metal at all; the deleted `facts_from_with`
        // carried it from a `DecodeGeometry` that had read it off a
        // `config.json`, which is the second reading this crate exists
        // to remove. `Deployment::v_norm` is where the answer lives now.
        v_norm: false,
        // FALSE: this family's mixture REPLACES the dense MLP rather
        // than sitting beside it. The manifest is that claim too, and it
        // is exclusive — a dense row states `mlp.gate_proj`, a routed
        // row states `mlp.gate` and the expert banks, and no row states
        // both. gemma-4 runs both branches off the post-attention
        // residual and adds them, which is five norms round one block.
        dense_beside_moe: false,
        // FALSE: no row here ships a `layer_scalar`. It is gemma's, for
        // a deployment with no per-layer embeddings, and the driver
        // asked the TENSORS for it because gemma-4-31b states
        // `hidden_size_per_layer_input: 0` and has the scalar anyway —
        // "the gemma-shaped fields are populated" and "has a PLE" are
        // not the same question.
        per_layer_scalar: false,
        // ZERO, which is "scaled not at all". gemma multiplies its
        // gathered embeddings by `sqrt(hidden)`; nothing in this lineage
        // does, and the difference is not subtle — a gemma that got no
        // scale had a widest gathered value of 0.058 where MLX's
        // reference for the same snapshot is about seventy times that.
        embed_scale: 0.0,
        // ZERO, which is "derive `1/sqrt(head_dim)`" — and for this
        // family the derived answer IS the row's answer, so stating a
        // number here would only be a second place for it to be wrong.
        //
        // The head dim the Metal text derives it from is the
        // CHECKPOINT's, which is the right one and differs from the CUDA
        // side's on exactly one row: phi-3-mini's 96-wide heads run on
        // the 128-wide CUDA kernel and `deployment` scales by
        // `1/sqrt(128)` to match it, while Metal takes `head_dim` as an
        // operand and never pads. The two backends divide by different
        // numbers because they attend over different widths.
        attn_scale: 0.0,
        // ZERO: no side network. gemma's PLE is a second embedding table
        // gathered once per step, projected, normed and joined into
        // `[n_layers, ple_dim]` that each layer reads its own slice of,
        // and nothing llama-like has a counterpart.
        per_layer_emb_dim: 0,
        // ZERO: every layer of this family owns its pages. KV sharing is
        // gemma-4's tail, where a shared layer rotates its own Q and
        // reads the pages its source wrote — no k/v projection, no k/v
        // norm, no append — and suppressing those dispatches is not an
        // optimisation but which tensors the checkpoint ships. This
        // family's manifest requires `k_proj` and `v_proj` at every
        // layer, and `LoadShape::dense` states the same zero.
        kv_shared_layers: 0,
        // ZERO, which is "no softcap" and names nothing. gemma-2 caps
        // its readout at 30 and gemma-3 dropped it — `final_logit_
        // softcapping` is null in every published gemma-3 config — and
        // passing a cap so large it does nothing would be a kernel run
        // per fire to compute the identity.
        logit_softcap: 0.0,
        // FALSE: no row here ships `self_attn.sinks`. A sink is a
        // per-head learned logit that joins the softmax without a value
        // behind it, so a sinked attention normalizes over one more term
        // than it sums. It is gpt-oss's, and gpt-oss has a text of its
        // own.
        attn_sinks: false,
        // `silu_mul` — `silu(gate) * up` — every llama-lineage
        // deployment's. Three symbols exist rather than one with flags
        // because the difference is not cosmetic: gpt-oss clamps the
        // gate ABOVE only, clamps the linear branch both ways and adds
        // one to it, and gemma's gelu is the TANH approximation rather
        // than the erf one. Dropping any of that produces a model that
        // runs and is wrong — the two agree to about 2% at the origin
        // and diverge from there.
        activation: super::forward::facts::Activation::SiluMul,
        // Whether the driver hands over a frequency TABLE instead of a
        // base. True for llama-3's piecewise rescaling and OLMo-3's
        // YaRN, which no `rope_theta` expresses — see [`MetalRow`] for
        // the bug this closes, which is that nothing carried either for
        // the length of this refactor and a zero factor reads as "no
        // rescaling".
        rope_freq_table: row.rope_rescaled,
        // The window, as the per-layer list the text reads through
        // `window_left_at`. EMPTY when the row attends the whole
        // context, because empty is what the accessor reads as `-1`
        // everywhere and a list of repeated `-1`s is a table walked for
        // nothing; ONE entry otherwise, because the accessor's last
        // entry covers the tail and this row's layers all agree.
        //
        // Stated, and not left empty the way `cuda_facts` leaves its
        // own: the CUDA driver carries a window per layer on its
        // `fwd_cfg` and the Metal text has only this. Mistral-7B-v0.3
        // and Qwen-2 are the rows it is not `-1` for, and a window
        // dropped is a model that attends its whole prefix — fluent, and
        // about a context the checkpoint was never trained to see.
        window_left: if row.window < 0 { Vec::new() } else { vec![row.window] },
    }
}

/// Trace this row's text for one fire class, on the backend that asked.
///
/// The two arms are two TEXTS over one row, which is the shape the
/// catalog was rewritten for: a model is a row, a backend is a
/// parameter of the question, and neither one is a string a driver
/// looks up. What stood here before was `driver-metal`'s eleven-entry
/// table of architecture NAMES — `llama`, `llama3`, `qwen3_moe`,
/// `gemma4` — reduced by a punctuation-stripping `canonical()` and
/// consulted before any text was traced. It was the third dispatch key,
/// and it disagreed with the other two exactly the way the first three
/// did: `gemma4` was listed as served by a driver that refused it on a
/// different ground, and `gemma3` was absent from a table whose text
/// models it.
///
/// # Errors
///
/// [`Refusal::Unsupported`](crate::deployment::Refusal) carrying
/// [`NO_METAL_SHARD`] when a Metal load says its weights were sharded,
/// [`NO_METAL_HEAD_DIM`] when the row's heads are a width the paged
/// decode shader has no instantiation for, or
/// [`NO_METAL_ROUTED_ENCODING`] when a routed bank arrived affine at
/// some point other than [`METAL_ROUTED_AFFINE`].
///
/// The last two are refusals about the KERNEL SET and not about the row,
/// which is why they are here rather than at a generation: twelve
/// families share this text, so a width or an encoding it cannot name is
/// a gap in the same one sentence for all of them.
#[cfg(feature = "forward")]
pub fn trace(
    f: &LlamaLikeFacts,
    row: MetalRow,
    class: model_compiler::trace::FireClass,
    load: Deployed<'_>,
) -> Result<model_compiler::trace::ForwardPlan, crate::deployment::Refusal> {
    match load.backend {
        Backend::Cuda => Ok(super::forward::llama_like_cuda(f, &cuda_facts(f, load), class)),
        Backend::Metal(bind) => {
            if load.tp_size > 1 {
                return Err(crate::deployment::Refusal::Unsupported(NO_METAL_SHARD));
            }
            if !METAL_SDPA_HEAD_DIMS.contains(&f.head_dim) {
                return Err(crate::deployment::Refusal::Unsupported(NO_METAL_HEAD_DIM));
            }
            // MXFP4 banks take their own symbol at group 32, so the
            // affine point is not asked of them.
            let routed = f.n_experts > 0;
            if routed
                && !bind.moe_mxfp4
                && (bind.quant_group, bind.quant_bits) != METAL_ROUTED_AFFINE
            {
                return Err(crate::deployment::Refusal::Unsupported(NO_METAL_ROUTED_ENCODING));
            }
            Ok(super::forward::llama_like_metal(f, &metal_facts(f, row, load, bind), class))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The epsilon these projections are exercised at.
    ///
    /// One number for every test below, because none of them reads it:
    /// it is a fact about a CHECKPOINT that rides through `deployment`
    /// untouched, and a test that varied it would only be asserting that
    /// a field is copied. The rows' own values are held against their
    /// published configs by `tests/catalog_differential.rs`, which is
    /// where a transcribed number belongs.
    const NORM_EPS: f32 = 1e-6;

    /// The build's instantiation set, as a rounding rule rather than a
    /// table each caller re-derives.
    #[test]
    fn a_head_dim_rounds_up_to_one_this_build_instantiates() {
        assert_eq!(round_up_attn_head_dim(64), 64);
        assert_eq!(round_up_attn_head_dim(96), 128, "phi-3's 96 pads to 128");
        assert_eq!(round_up_attn_head_dim(128), 128);
        // Nothing holds it, so the caller surfaces the dispatch error
        // rather than getting a silently mis-sized answer.
        assert_eq!(round_up_attn_head_dim(768), 768);
    }

    /// The four probes the old derivation made of the LOAD are four
    /// manifest rows, so two variants that differ only in them cannot
    /// both match one checkpoint.
    #[test]
    fn the_load_probes_became_expectations() {
        let per_head = manifest(&LlamaLikeFacts::qwen3_0_6b());
        let global = manifest(&LlamaLikeFacts::olmo2_1b());
        let names = |m: &Manifest| -> Vec<String> {
            m.tensors.iter().map(|t| format!("{}:{:?}:{:?}", t.name, t.extents, t.presence))
                .collect()
        };
        assert_ne!(names(&per_head), names(&global));
    }

    /// olmo2's post-norm placement reaches the deployment, which is
    /// what `model_type.starts_with("olmo")` used to decide — a string
    /// test on the identity, for a fact the row states.
    #[test]
    fn norm_placement_is_stated_rather_than_matched_on_a_name() {
        let olmo = deployment(&LlamaLikeFacts::olmo2_1b(), 500_000.0, NORM_EPS, -1);
        assert_eq!(olmo.norm, NormPlacement::Post);
        let qwen = deployment(&LlamaLikeFacts::qwen3_0_6b(), 1e6, NORM_EPS, -1);
        assert_eq!(qwen.norm, NormPlacement::Pre);
    }

    /// The launch geometry is the row's own numbers, not a second
    /// reading of anything.
    #[test]
    fn the_launch_geometry_is_the_rows_own_numbers() {
        let f = LlamaLikeFacts::qwen3_0_6b();
        let d = deployment(&f, 1e6, NORM_EPS, -1);
        assert_eq!(d.shape.hidden, f.hidden);
        assert_eq!(d.shape.q_heads, f.q_heads);
        assert_eq!(d.shape.kv_heads, f.kv_heads);
        assert_eq!(d.shape.head_dim, f.head_dim);
        assert_eq!(d.shape.intermediate, f.intermediate);
        assert_eq!(d.shape.vocab, f.vocab);
        assert_eq!(d.shape.gqa_group(), 2, "16 q over 8 kv");
        assert_eq!(d.shape.head_dim_alloc(), 128, "128 is instantiated; nothing pads");
    }

    /// Phi-3's 96-wide heads run on the 128-wide kernel, and the
    /// difference is a WIDTH rather than a boolean: a buffer sized
    /// `heads * head_dim` is short by a third.
    #[test]
    fn a_padded_head_reaches_the_geometry_as_a_width() {
        let f = LlamaLikeFacts::phi3_mini();
        let d = deployment(&f, 10_000.0, NORM_EPS, -1);
        assert_eq!(d.shape.head_dim, 96, "the checkpoint's own");
        assert_eq!(d.shape.head_dim_kernel, 128, "the one instantiated");
        assert_eq!(d.shape.head_dim_alloc(), 128, "the one to allocate");
        for a in &d.attention {
            assert_eq!(a.head_dim, 128, "attention is sized for the kernel");
        }
    }

    /// The GQA ratios this build's decode instantiates. Outside the set
    /// FlashInfer THROWS, and a throw crossing a C ABI is undefined
    /// behaviour — which is why the question is asked at the door.
    #[test]
    fn the_gqa_set_is_the_builds_and_not_the_familys() {
        assert_eq!(DECODE_GQA_GROUPS, &[1, 2, 3, 4, 8]);
        let g = deployment(&LlamaLikeFacts::qwen3_0_6b(), 1e6, NORM_EPS, -1).shape.gqa_group();
        assert!(DECODE_GQA_GROUPS.contains(&g), "qwen3-0.6b's 2 is servable");
        assert_eq!(Geometry::EMPTY.gqa_group(), 0, "no division by zero");
    }

    /// A sliding window reaches every layer, and full attention is -1.
    #[test]
    fn the_window_is_stated_per_layer() {
        let f = LlamaLikeFacts::mistral_7b_v03();
        let full = deployment(&f, 1e6, NORM_EPS, -1);
        assert!(full.attention.iter().all(|a| a.window == -1));
        let windowed = deployment(&f, 1e6, NORM_EPS, 4096);
        assert!(windowed.attention.iter().all(|a| a.window == 4096));
    }

    /// Every layer owns its pages here. KV sharing is gemma-4's, and it
    /// is a fact about a LAYER there rather than a family here.
    #[test]
    fn every_layer_owns_its_own_pages() {
        let d = deployment(&LlamaLikeFacts::qwen3_0_6b(), 1e6, NORM_EPS, -1);
        for (l, a) in d.attention.iter().enumerate() {
            assert_eq!(a.kv_source, l as u32);
            assert_eq!(a.rope_theta, 1e6);
            assert_eq!(a.rotary_dim, 0, "full rotation at the head dim");
        }
        assert_eq!(d.kv, KvStyle::Paged);
        assert_eq!(d.prefill, PrefillStyle::Planned);
        assert_eq!(d.attn_output, AttnOutput::DriverPinned);
        assert_eq!(d.logit_softcap, 0.0);
        assert_eq!(d.ple_dim, 0);
        assert!(d.recurrent.is_none());
        assert!(d.scales.is_empty());
    }

    /// A tie is an ABSENCE, and the manifest says so — which is the only
    /// way tied and untied can be told apart when every extent agrees.
    #[test]
    fn a_tie_is_an_absence_the_manifest_expects() {
        use crate::manifest::Presence;
        let tied = manifest(&LlamaLikeFacts::qwen3_0_6b());
        let head = tied.tensors.iter().find(|t| t.name == "lm_head").expect("stated");
        assert_eq!(head.presence, Presence::Absent);

        let untied = manifest(&LlamaLikeFacts::phi3_mini());
        let head = untied.tensors.iter().find(|t| t.name == "lm_head").expect("stated");
        assert_eq!(head.presence, Presence::Required);
    }

    /// A mixture ships a router and a dense block does not, so the
    /// manifest tells them apart without reading a `model_type`.
    #[test]
    fn a_mixture_ships_a_router() {
        let dense = manifest(&LlamaLikeFacts::qwen3_0_6b());
        assert!(dense.tensors.iter().any(|t| t.name.contains("mlp.gate_proj")));
        assert!(!dense.tensors.iter().any(|t| t.name.ends_with("mlp.gate")));

        let moe = manifest(&LlamaLikeFacts::qwen3_30b_a3b());
        assert!(moe.tensors.iter().any(|t| t.name.ends_with("mlp.gate")));
        assert!(!moe.tensors.iter().any(|t| t.name.contains("mlp.gate_proj")));
    }

    /// Attention biases are a Qwen-2 fact, and the manifest expects the
    /// tensor rather than inferring it from a name.
    #[test]
    fn attention_biases_are_expected_when_the_row_says_so() {
        let with = manifest(&LlamaLikeFacts::qwen2_5_1_5b());
        assert!(with.tensors.iter().any(|t| t.name.ends_with("q_proj.bias")));
        let without = manifest(&LlamaLikeFacts::qwen3_0_6b());
        assert!(!without.tensors.iter().any(|t| t.name.ends_with("q_proj.bias")));
    }

    /// The q/k-norm question the old derivation answered by dividing a
    /// byte count is three distinct expectations.
    #[test]
    fn qk_norm_is_three_expectations_and_not_a_division() {
        use crate::manifest::Presence;
        let spec = |f: &LlamaLikeFacts| {
            manifest(f)
                .tensors
                .into_iter()
                .find(|t| t.name.ends_with("q_norm"))
                .expect("every row states it")
        };
        let per_head = spec(&LlamaLikeFacts::qwen3_0_6b());
        assert_eq!(per_head.presence, Presence::Required);
        assert_eq!(per_head.extents, vec![128]);

        let global = spec(&LlamaLikeFacts::olmo2_1b());
        assert_eq!(global.presence, Presence::Required);
        assert_ne!(global.extents, vec![128], "global is the projection width");

        let off = spec(&LlamaLikeFacts::mistral_7b_v03());
        assert_eq!(off.presence, Presence::Absent);
    }

    /// Two bindings of one checkpoint, which is what a `MetalBinding`
    /// is for: the g64/b4 publication and the g128/b8 one.
    #[cfg(feature = "forward")]
    fn binding(group: u32, bits: u32) -> MetalBinding {
        MetalBinding {
            quant_group: group,
            quant_bits: bits,
            moe_mxfp4: false,
            fuse_residual_gemv: true,
            paged_multi_batch: true,
            qmm_multi_batch: true,
        }
    }

    /// qwen3-0.6b's row, as the generation states it — a full-attention
    /// dense stack on one rope base.
    #[cfg(feature = "forward")]
    fn qwen3_row() -> MetalRow {
        MetalRow { rope_theta: 1e6, norm_eps: 1e-6, window: -1, rope_rescaled: false }
    }

    /// THE claim this projection makes: the binding is the LOAD's
    /// answer and everything else is the ROW's.
    ///
    /// It is the whole of what `driver-metal`'s `facts_from_with`
    /// stopped having to do. That function filled twenty-six fields
    /// from a `DecodeGeometry` and nine tensor probes, and eighteen of
    /// them were facts the row already stated — the epsilon, the rotary
    /// base, the head counts, the activation, the softcap. Asking a
    /// checkpoint what it is, twice, is how the answers came to differ:
    /// a gemma read as a llama folded `(1 + w)` as `w` and dropped two
    /// norms per layer, and nothing faulted.
    #[cfg(feature = "forward")]
    #[test]
    fn the_metal_binding_is_the_loads_answer_and_the_rest_is_the_rows() {
        let f = LlamaLikeFacts::qwen3_0_6b();
        let bind = binding(64, 4);
        let m = metal_facts(&f, qwen3_row(), Deployed::metal(&bind), &bind);

        // The LOAD's six, field for field.
        assert!(m.fuse_residual_gemv);
        assert!(m.paged_multi_batch);
        assert!(m.qmm_multi_batch);
        assert_eq!(m.affine_bits, 4, "the checkpoint's own bit width");
        assert_eq!(
            m.proj_repr,
            model_compiler::dsl::WeightRepr::Scaled {
                layout: model_compiler::dsl::ScaleLayout::PerGroup,
                group: 64,
                axis: 0,
                zero_point: true,
            },
            "MLX stores `.scales` and `.biases`, which is a zero-point layout"
        );
        assert_eq!(m.moe_repr, None, "this load left no bank in MXFP4");

        // The ROW's, which no load observed.
        assert_eq!(m.rms_eps, 1e-6);
        assert_eq!(m.rope_theta, 1e6);
        assert!(!m.rope_freq_table, "a plain geometric ladder");
        assert!(m.window_left.is_empty(), "qwen3 attends its whole prefix");
        assert_eq!(m.activation, super::super::forward::facts::Activation::SiluMul);
        assert!(!m.gate_up_fused, "the manifest states gate and up as two tensors");
    }

    /// The same row under a DIFFERENT publication of the same weights.
    ///
    /// `mlx-community` ships the 4-bit and the 8-bit build of one
    /// checkpoint and they differ in nothing else, which is why an
    /// encoding is a policy and not an identity — the module doc's rule,
    /// asked of the one projection that has to hold both.
    #[cfg(feature = "forward")]
    #[test]
    fn a_second_publication_of_one_row_moves_only_the_binding() {
        let f = LlamaLikeFacts::qwen3_0_6b();
        let four = binding(64, 4);
        let eight = binding(128, 8);
        let a = metal_facts(&f, qwen3_row(), Deployed::metal(&four), &four);
        let b = metal_facts(&f, qwen3_row(), Deployed::metal(&eight), &eight);

        assert_ne!(a.affine_bits, b.affine_bits);
        assert_ne!(a.proj_repr, b.proj_repr, "g64/b8 and g128/b4 pack identically");
        // And nothing the ROW states moved, because nothing about the
        // model did.
        assert_eq!(a.rms_eps, b.rms_eps);
        assert_eq!(a.rope_theta, b.rope_theta);
        assert_eq!(a.window_left, b.window_left);
        assert_eq!(a.activation, b.activation);
        assert_eq!(a.qmm_tile, b.qmm_tile);
    }

    /// A second row, so the semantic half is read off THAT row and not
    /// off a constant that happens to match qwen3's.
    ///
    /// Mistral-7B-v0.3 is the sharpest one available: it states a
    /// different rotary base, a different epsilon and — unlike every
    /// other row this projection serves — a WINDOW.
    #[cfg(feature = "forward")]
    #[test]
    fn a_second_row_states_its_own_window_and_its_own_base() {
        let f = LlamaLikeFacts::mistral_7b_v03();
        let bind = binding(64, 4);
        let row = MetalRow { rope_theta: 1e6, norm_eps: 1e-5, window: 4096, rope_rescaled: false };
        let m = metal_facts(&f, row, Deployed::metal(&bind), &bind);

        assert_eq!(m.rms_eps, 1e-5, "mistral's, not qwen3's");
        assert_eq!(m.window_left, vec![4096]);
        // One entry covers the tail, which is what the accessor means by
        // a list shorter than the layer count — every mistral layer
        // slides the same 4096.
        for l in 0..f.layers {
            assert_eq!(m.window_left_at(l), 4096, "layer {l}");
        }
        // And the row with no window states none, which the accessor
        // reads as the whole context rather than as a zero-width one.
        let full = metal_facts(&LlamaLikeFacts::qwen3_0_6b(), qwen3_row(), Deployed::metal(&bind), &bind);
        assert_eq!(full.window_left_at(0), -1);
    }

    /// The gemma-shaped fields are STATED zero, one at a time.
    ///
    /// `LlamaLikeMetalFacts` has no `Default` impl and this projection
    /// spells no `..Default::default()`, for the reason the fixtures
    /// give: a row is a MEASUREMENT of a real checkpoint, and "this one
    /// has no per-layer embeddings" is part of the measurement. This
    /// test is that discipline held rather than described — it fails if
    /// a future field is defaulted into existence.
    #[cfg(feature = "forward")]
    #[test]
    fn the_gemma_shaped_facts_are_stated_rather_than_defaulted() {
        let bind = binding(64, 4);
        let m = metal_facts(&LlamaLikeFacts::qwen3_0_6b(), qwen3_row(), Deployed::metal(&bind), &bind);
        assert_eq!(m.rope_theta_sliding, 0.0, "one base for every layer");
        assert_eq!(m.global_head_dim, 0, "one attention shape");
        assert_eq!(m.global_kv_heads, 0);
        assert_eq!(m.full_partial_rotary, 0.0);
        assert!(!m.v_from_k, "every layer here projects its own V");
        assert!(!m.dense_beside_moe, "the mixture replaces the dense MLP");
        assert!(!m.per_layer_scalar);
        assert_eq!(m.embed_scale, 0.0, "gemma's `sqrt(hidden)` is gemma's");
        assert_eq!(m.attn_scale, 0.0, "derive `1/sqrt(head_dim)`");
        assert_eq!(m.per_layer_emb_dim, 0, "no side network");
        assert_eq!(m.kv_shared_layers, 0, "every layer owns its pages");
        assert_eq!(m.logit_softcap, 0.0);
        assert!(!m.attn_sinks);
        // The accessors read those zeros as "one shape everywhere",
        // which is the reading the text depends on.
        assert_eq!(m.head_dim_at(0, 128), 128);
        assert_eq!(m.kv_heads_at(0, 8), 8);
        assert_eq!(m.rotary_dim_at(0, 128), 128, "the whole head rotates");
        assert_eq!(m.rope_theta_at(0), 1e6);
    }

    /// The tile is the BUILD's stamp and the two bits that are not.
    ///
    /// A symbol whose tile is wrong reads the wrong bytes; a symbol with
    /// no tile does not resolve at all, which is the better failure —
    /// so a `(0, 0)` here would be a fixture written before the field
    /// existed and not a deployment that wants no tile.
    #[cfg(feature = "forward")]
    #[test]
    fn the_gemm_tile_is_the_builds_stamp_and_not_the_rows() {
        let bind = binding(64, 4);
        let m = metal_facts(&LlamaLikeFacts::qwen3_0_6b(), qwen3_row(), Deployed::metal(&bind), &bind);
        assert_eq!(QMM_TILE, (16, 32));
        assert_eq!(m.qmm_tile, QMM_TILE);
        // The narrowest rung `qmm_bm` can pick, and the only column tile
        // the residual variant carries.
        assert_ne!(m.qmm_tile, (0, 0), "a stem does not resolve");
    }

    /// The expert bank states its own format only when the load left it
    /// in MXFP4.
    ///
    /// Reading a bank with the dense format is not a near miss: every
    /// scale comes from the wrong offset, and the fire that did it
    /// produced 909,207 NaNs beginning at the first routed projection of
    /// layer 0. `None` is "the same as the dense projections", which is
    /// every checkpoint this family serves.
    #[cfg(feature = "forward")]
    #[test]
    fn the_expert_bank_names_its_own_format_only_when_the_load_left_one() {
        let f = LlamaLikeFacts::qwen3_30b_a3b();
        let plain = binding(64, 4);
        let mixed = MetalBinding { moe_mxfp4: true, ..binding(64, 4) };

        let uniform = metal_facts(&f, qwen3_row(), Deployed::metal(&plain), &plain);
        assert_eq!(uniform.moe_repr, None);

        let split = metal_facts(&f, qwen3_row(), Deployed::metal(&mixed), &mixed);
        assert_eq!(split.moe_repr, Some(model_compiler::dsl::WeightRepr::Mxfp4Marlin));
        assert_eq!(split.moe_bits, 4, "MXFP4 is four bits by the format's own name");
    }

    /// A rescaled ladder reaches the text as a TABLE, not as a base.
    ///
    /// llama-3's piecewise rescaling and OLMo-3's YaRN. No `rope_theta`
    /// expresses either, so a text that stated one would rotate by the
    /// wrong frequencies from the second channel on, at every position
    /// but zero — degrading rather than failing, which is the shape of
    /// defect this catalog exists to make impossible.
    #[cfg(feature = "forward")]
    #[test]
    fn a_rescaled_ladder_is_a_table_and_a_plain_one_is_a_base() {
        let f = LlamaLikeFacts::qwen3_0_6b();
        let bind = binding(64, 4);
        let plain = metal_facts(&f, qwen3_row(), Deployed::metal(&bind), &bind);
        assert!(!plain.rope_freq_table);

        let rescaled = MetalRow { rope_rescaled: true, ..qwen3_row() };
        let m = metal_facts(&f, rescaled, Deployed::metal(&bind), &bind);
        assert!(m.rope_freq_table, "the driver hands over a table instead");
    }

    /// One row, two backends, two texts — and neither is reached by a
    /// name in a table.
    ///
    /// This is the guard that replaces `driver-metal`'s `LLAMA_LIKE`
    /// list of eleven architecture strings and its `canonical()`
    /// reduction. The row is the dispatch; the backend is a parameter of
    /// the question.
    #[cfg(feature = "forward")]
    #[test]
    fn one_row_traces_on_either_backend_and_says_which() {
        use model_compiler::trace::FireClass;
        let f = LlamaLikeFacts::qwen3_0_6b();
        let bind = binding(64, 4);
        for class in [FireClass::Prefill, FireClass::Decode] {
            let cuda = trace(&f, qwen3_row(), class, Deployed::single())
                .expect("the CUDA text is written");
            let metal = trace(&f, qwen3_row(), class, Deployed::metal(&bind))
                .expect("and so is the Metal one");
            assert!(cuda.family.starts_with("llama_like.cuda."), "{}", cuda.family);
            assert!(metal.family.starts_with("llama_like.metal."), "{}", metal.family);
            assert_ne!(cuda.ops.len(), 0);
            assert_ne!(metal.ops.len(), 0);
        }
    }

    /// A SHARDED Metal load is refused rather than traced at full width.
    ///
    /// The CUDA facts carry a `tp_size` and the text divides every
    /// projection width by it; the Metal facts carry nothing, so the
    /// same trace would state the whole model's widths against one
    /// rank's slice of the weights. That is not a crash — it is a
    /// projection reading past the end of its own tensor — so the door
    /// says no.
    #[cfg(feature = "forward")]
    #[test]
    fn a_sharded_metal_load_is_refused_rather_than_traced_at_full_width() {
        use crate::deployment::Refusal;
        use model_compiler::trace::FireClass;
        let f = LlamaLikeFacts::qwen3_0_6b();
        let bind = binding(64, 4);
        let sharded = Deployed { backend: Backend::Metal(&bind), tp_size: 4, layer_scalars: &[] };
        let err = trace(&f, qwen3_row(), FireClass::Decode, sharded)
            .expect_err("four ranks and no shard vocabulary");
        assert_eq!(err, Refusal::Unsupported(NO_METAL_SHARD));
        // One rank is the shape of every Metal deployment there is, and
        // it traces.
        assert!(trace(&f, qwen3_row(), FireClass::Decode, Deployed::metal(&bind)).is_ok());
        // The CUDA side shards and keeps tracing: `tp_size` is a fact
        // its own facts carry.
        let cuda = Deployed { backend: Backend::Cuda, tp_size: 4, layer_scalars: &[] };
        assert!(trace(&f, qwen3_row(), FireClass::Decode, cuda).is_ok());
    }

    /// A head width no shader compiled is refused, not named.
    ///
    /// phi-3's heads are 96 and `sdpa_paged.metal` instantiates the paged
    /// decode at 64, 128, 256 and 512. The text spells the width into the
    /// symbol -- deliberately, since the `_d_256` literal was a real
    /// defect -- so 96 produced `sdpa_paged_decode_bfloat16_d_96`.
    ///
    /// Nothing caught that at the call, because nothing got there:
    /// `model-compiler` validates a declaration against the kernel rows
    /// and PANICS on an undeclared launch, so `driver-metal`'s
    /// catalog-wide binding gate died with a backtrace instead of
    /// reporting a row. The refusal is what the doc always promised --
    /// "a width no kernel instantiates simply does not resolve, and the
    /// driver's row check reports it by name" -- said early enough to be
    /// a sentence.
    #[cfg(feature = "forward")]
    #[test]
    fn a_head_width_no_metal_shader_compiled_is_refused_by_name() {
        use crate::deployment::Refusal;
        use model_compiler::trace::FireClass;
        let bind = binding(64, 4);
        let phi3 = LlamaLikeFacts::phi3_mini();
        assert_eq!(phi3.head_dim, 96, "the fixture this test reads");
        for class in [FireClass::Prefill, FireClass::Decode] {
            let err = trace(&phi3, qwen3_row(), class, Deployed::metal(&bind))
                .expect_err("96 is not a point sdpa_paged.metal has");
            assert_eq!(err, Refusal::Unsupported(NO_METAL_HEAD_DIM));
        }
        // CUDA pads to `head_dim_kernel` and strips the pad back off, so
        // the same row keeps tracing there. The refusal is about this
        // build's SHADERS and nothing about phi-3.
        assert!(trace(&phi3, qwen3_row(), FireClass::Decode, Deployed::single()).is_ok());
        // And every width the shader does have still traces.
        for w in METAL_SDPA_HEAD_DIMS {
            let mut f = LlamaLikeFacts::qwen3_0_6b();
            f.head_dim = *w;
            assert!(
                trace(&f, qwen3_row(), FireClass::Decode, Deployed::metal(&bind)).is_ok(),
                "d_{w} is instantiated and must not be refused"
            );
        }
    }

    /// A routed bank at an affine point the routed matvec has no
    /// instantiation for is refused.
    ///
    /// `AffineQ::group_size` is a template constant, so
    /// `affine_qmv_routed` exists at group 64 / 4 bits and nowhere else.
    /// The symbol takes the repr the caller states -- which is right, and
    /// is what fixed gpt-oss's MXFP4 bank being read as affine-64 -- so a
    /// g128/b8 bank named `affine_qmv_routed_bfloat16_gs_128_b_8`, which
    /// no shader exports.
    ///
    /// This is the one refusal in this file that moves with the LOAD
    /// rather than the row, which is why `driver-metal`'s
    /// `a_row_is_served_the_same_way_at_every_encoding` had to become
    /// one-directional: a pre-staging probe may be permissive, it may not
    /// be wrong.
    #[cfg(feature = "forward")]
    #[test]
    fn a_routed_bank_at_an_uninstantiated_affine_point_is_refused() {
        use crate::deployment::Refusal;
        use model_compiler::trace::FireClass;
        let moe = LlamaLikeFacts::qwen3_30b_a3b();
        assert!(moe.n_experts > 0, "the fixture this test reads");

        let (g, b) = METAL_ROUTED_AFFINE;
        let good = binding(g, b);
        assert!(trace(&moe, qwen3_row(), FireClass::Decode, Deployed::metal(&good)).is_ok());

        let bad = binding(128, 8);
        let err = trace(&moe, qwen3_row(), FireClass::Decode, Deployed::metal(&bad))
            .expect_err("group 128 has no routed instantiation");
        assert_eq!(err, Refusal::Unsupported(NO_METAL_ROUTED_ENCODING));

        // An MXFP4 bank is a different codec with its own symbol at group
        // 32, so the affine point is not asked of it.
        let mut mxfp4 = binding(128, 8);
        mxfp4.moe_mxfp4 = true;
        assert!(trace(&moe, qwen3_row(), FireClass::Decode, Deployed::metal(&mxfp4)).is_ok());

        // A DENSE row is not asked either: the point governs the expert
        // bank, and a row with no bank has no stake in it.
        let dense = LlamaLikeFacts::qwen3_0_6b();
        assert_eq!(dense.n_experts, 0);
        assert!(trace(&dense, qwen3_row(), FireClass::Decode, Deployed::metal(&bad)).is_ok());
    }
}

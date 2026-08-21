//! The three projections a gemma-4 row makes.
//!
//! Gemma-4 is where the old `PlannedFamily` trait's defaults ran out,
//! and the three projections here are the same four exceptions written
//! as statements instead:
//!
//! * `pins_attention_values()` was a DEFAULT returning `true`, with the
//!   doc *"Only gemma-4 does"* — a default whose documentation names the
//!   one row it is false for. It is [`AttnOutput::StatedArgs`] below,
//!   and this generation is the only one in the catalog that says it.
//! * `planless_prefill()` defaulted to `false` and gemma-4 overrode it.
//!   Its 512-wide sliding layers take the naive kernel, which plans
//!   inside the fire from the host CSR mirrors, so there is no plan to
//!   raise: [`PrefillStyle::Planless`].
//! * `sm_scale()` defaulted to `1/sqrt(head_dim)`. Gemma-4's q and k
//!   norms carry the scaling already, so the softmax scale is 1.0 and a
//!   derived one would apply it twice.
//! * `decode_plan_head_dims()` existed as a vtable method BECAUSE this
//!   generation's two layer kinds disagree. It is
//!   `Deployment::decode_head_dims()` now, a question about the per-layer
//!   table, and the table is filled here.
//!
//! The fifth exception has no old counterpart at all, because the old
//! shape had nowhere to put it: **a KV-shared layer names another
//! layer's pages**, and `LayerAttention::kv_source` exists for it.

use model_dsl::axes::DtypeAxis;
use std::collections::BTreeMap;

use crate::catalog::Deployed;
use crate::deployment::{
    Advertised, AttnOutput, Deployment, Geometry, KvStyle, LayerAttention, NormPlacement,
    PrefillStyle, Towers,
};
use crate::manifest::{Manifest, TensorSpec};

use super::spec::{Gemma4Facts, Gemma4Mixture};

/// The rope base a SLIDING layer rotates at.
///
/// A generation constant rather than a row field because all four
/// published gemma-4 configs state the same pair, and the pair is
/// keyed by LAYER KIND rather than by checkpoint: `rope_parameters`
/// carries a `sliding_attention` object and a `full_attention` one, and
/// every file in the corpus fills them with 10 000 and 1 000 000. A
/// checkpoint that disagreed would need these on the row, and would be
/// a different reading of the generation rather than a fifth row.
pub const ROPE_THETA_LOCAL: f32 = 10_000.0;

/// The rope base a FULL-attention layer rotates at.
pub const ROPE_THETA_GLOBAL: f32 = 1_000_000.0;

/// This row's tensors.
///
/// # Layer 0 is a SLIDING layer, and that is why this is well posed
///
/// Gemma-4's projections are not one width: a full-attention layer's
/// heads are `global_head_dim` wide (512) and a sliding layer's are
/// `head_dim` (256), so `layer.{}.self_attn.q_proj` does not name one
/// extent for the whole stack. [`Manifest`] expands `{}` to layer 0 and
/// compares once — and layer 0 is a sliding layer in every gemma-4,
/// because full attention lands on `l % interval == interval - 1` and
/// the smallest interval any published config states is four. So these
/// rows are the SLIDING geometry, stated deliberately rather than
/// arrived at.
///
/// # The rows that tell this generation from gemma-3n
///
/// Both ship the per-layer embedding trio, which nothing else in the
/// catalog does, so the PLE rows do not separate them. Three rows do,
/// and all three are absences: gemma-3n norms V and gemma-4 does not,
/// gemma-3n carries an AltUp router and a laurel branch and gemma-4
/// carries neither. An absence is a fact a checkpoint can be held to —
/// see [`TensorSpec::absent`] — and stating them here is what stops a
/// gemma-3n from matching this row on its way past.
#[must_use]
pub fn manifest(f: &Gemma4Facts, mixture: Option<Gemma4Mixture>) -> Manifest {
    let (hidden, vocab) = (u64::from(f.hidden), u64::from(f.vocab));
    // Layer 0's geometry, for the reason the doc gives.
    let head_dim = u64::from(f.head_dim_of(0));
    let q = u64::from(f.q_heads) * head_dim;
    let kv = u64::from(f.kv_heads) * head_dim;
    let inter = u64::from(f.intermediate_of(0));
    let ple = u64::from(f.ple_dim);
    let layers = u64::from(f.layers);
    let has_ple = f.ple_dim > 0;

    Manifest::new(f.layers)
        // The shipped projection repr, DERIVED from the family's one
        // axis spelling (`forward::ShippedW1`) — plain bf16 here, so
        // the claim is inert for matching and states only what the
        // catalogued text assumes.
        .holds_projections_as(<super::forward::ShippedW1 as DtypeAxis>::REPR)
        .with(TensorSpec::required("embed_tokens", [vocab, hidden]))
        .with(TensorSpec::required("norm", [hidden]))
        // A tie is an ABSENCE: gemma-4 ships no `lm_head`, and the MLX
        // author renames `embed_tokens` into the head's place rather
        // than binding a second table.
        .tie(f.tied_embeddings, "lm_head", [vocab, hidden])
        // THE PER-LAYER EMBEDDING TABLE, whose second axis multiplies
        // the layer count — the one tensor shape that says "gemma-3n or
        // gemma-4" and nothing else. The A4B mixture drops it
        // (`hidden_size_per_layer_input: 0`), so it is conditional here
        // and its absence is stated rather than merely unlisted.
        .with_if(
            has_ple,
            TensorSpec::required("embed_tokens_per_layer", [vocab, layers * ple]),
        )
        .with_if(
            has_ple,
            TensorSpec::required("per_layer_model_projection", [layers * ple, hidden]),
        )
        .with_if(
            has_ple,
            TensorSpec::required("per_layer_projection_norm", [ple]),
        )
        .with_if(!has_ple, TensorSpec::absent("embed_tokens_per_layer"))
        .with(TensorSpec::required(
            "layer.{}.self_attn.q_proj",
            [q, hidden],
        ))
        .with(TensorSpec::required(
            "layer.{}.self_attn.k_proj",
            [kv, hidden],
        ))
        .with(TensorSpec::required(
            "layer.{}.self_attn.v_proj",
            [kv, hidden],
        ))
        .with(TensorSpec::required(
            "layer.{}.self_attn.o_proj",
            [hidden, q],
        ))
        // Per-HEAD q/k norms: the extent is one head's width, not the
        // projection's, and on layer 0 that is the sliding head dim.
        .with(TensorSpec::required(
            "layer.{}.self_attn.q_norm",
            [head_dim],
        ))
        .with(TensorSpec::required(
            "layer.{}.self_attn.k_norm",
            [head_dim],
        ))
        // gemma-3n norms V as well. This generation does not, and the
        // absence is what keeps a gemma-3n from matching here.
        .with(TensorSpec::absent("layer.{}.self_attn.v_norm"))
        .with(TensorSpec::absent("layer.{}.altup.modality_router"))
        .with(TensorSpec::absent("layer.{}.laurel.linear_left"))
        // The four gemma norms.
        .with(TensorSpec::required("layer.{}.input_layernorm", [hidden]))
        .with(TensorSpec::required(
            "layer.{}.post_attention_layernorm",
            [hidden],
        ))
        .with(TensorSpec::required(
            "layer.{}.pre_feedforward_layernorm",
            [hidden],
        ))
        .with(TensorSpec::required(
            "layer.{}.post_feedforward_layernorm",
            [hidden],
        ))
        .with(TensorSpec::required(
            "layer.{}.mlp.gate_proj",
            [inter, hidden],
        ))
        .with(TensorSpec::required(
            "layer.{}.mlp.up_proj",
            [inter, hidden],
        ))
        .with(TensorSpec::required(
            "layer.{}.mlp.down_proj",
            [hidden, inter],
        ))
        // The PLE epilogue's own weights, present exactly when the table
        // they read is.
        .with_if(
            has_ple,
            TensorSpec::required("layer.{}.per_layer_input_gate", [ple, hidden]),
        )
        .with_if(
            has_ple,
            TensorSpec::required("layer.{}.per_layer_projection", [hidden, ple]),
        )
        .with_if(
            has_ple,
            TensorSpec::required("layer.{}.post_per_layer_input_norm", [hidden]),
        )
        // The router's `[hidden]` scale, which `contract::author_gemma4`
        // folds `1/sqrt(hidden)` into at load — it reads the width off
        // this tensor, so this row states the width that fold assumes.
        // A dense gemma-4 forbids it, which is how the A4B is told from
        // an E4B whose other extents somehow agreed.
        .either(mixture.is_some(), "layer.{}.router.scale", [hidden])
}

/// The four numbers a gemma-4 ROW states and its shape cannot.
///
/// One struct because a row is read once. `deployment` and
/// `metal_facts` used to take these four as loose arguments each — the
/// same four, off the same `Gemma4`, in the same order — and two
/// argument lists spelling one reading is two chances to spell it
/// differently. A deployment paged at a 512-token window and a fire
/// traced at 1024 compile; nothing in the types objects, and the
/// disagreement surfaces as attention that reads the wrong distance
/// back.
///
/// This is the same merge `llama_like::project::RowScalars` is, made
/// for the same reason, and gemma-4 needs its own because two of the
/// four are gemma-4's alone: a mixture the catalog attaches per row,
/// and the `attention_k_eq_v` that decides whether V has a projection
/// at all.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RowScalars {
    /// The routed bank, or `None` for a dense row.
    pub mixture: Option<Gemma4Mixture>,
    /// What a SLIDING layer attends. Not a tensor extent.
    pub sliding_window: i32,
    /// RMSNorm epsilon. Not a tensor extent either.
    pub norm_eps: f32,
    /// Whether the full layers read V out of the K projection, so the
    /// checkpoint ships no `v_proj`.
    pub k_eq_v: bool,
}

/// This row's deployment.
///
/// `sliding_window` and `norm_eps` are the row's rather than the
/// shape's, for [`crate::gemma_4::spec`]'s reason: the shape is the
/// thing a checkpoint's TENSORS can be measured against, and neither of
/// these is a tensor extent. `load` carries the one thing no row can
/// state — the per-layer scalars this rank read to host.
///
/// # Every layer is asked all six questions
///
/// Including the five where a gemma-4 layer's answer depends on which
/// KIND it is. The alternative — a table for the exceptions and a
/// default for the rest — is what `FamilyTables` was, and it is how a
/// full-attention layer came to be deployed with the sliding layers'
/// 512-token window: not a wrong answer, an unasked question.
#[must_use]
pub fn deployment(f: &Gemma4Facts, row: RowScalars, load: Deployed<'_>) -> Deployment {
    let RowScalars {
        mixture,
        sliding_window,
        norm_eps,
        // Deliberately unread, and destructured by name rather than swept
        // under a `..` so that a field added to the row still has to be
        // answered here. `Deployment` used to carry a copy of this "for the
        // driver to read"; no driver read it, because the two branches that
        // need it are elsewhere — `metal_facts` turns it into `v_from_k`,
        // and CUDA refuses on it in `Variant::trace`. What a rank allocates
        // does not change: `k_eq_v` decides whether V has its own
        // PROJECTION, not how many KV heads a layer has.
        k_eq_v: _,
    } = row;
    let attention = (0..f.layers)
        .map(|l| layer_attention(f, sliding_window, l))
        .collect();
    Deployment {
        layers: f.layers,
        norm_eps,
        shape: Geometry {
            hidden: f.hidden,
            q_heads: f.q_heads,
            kv_heads: f.kv_heads,
            // The CHECKPOINT's `head_dim`, which is the sliding layers'.
            // The full layers' 512 reaches a driver through the per-layer
            // table above, because that is where a per-layer fact goes.
            head_dim: f.head_dim,
            // Nothing is padded: 256 is one of the widths the attention
            // kernels are instantiated at.
            head_dim_kernel: f.head_dim,
            intermediate: f.intermediate,
            // ONE EXPERT's width, and 0 for the dense rows — not the
            // dense width repeated. The A4B's experts are 704 wide beside
            // a dense 2112, so a planner told 704 for both would size the
            // forward workspace at a third of what the dense layers ask.
            moe_intermediate: mixture.map_or(0, |m| m.moe_intermediate),
            experts_per_token: mixture.map_or(0, |m| m.experts_per_token),
            shared_intermediate: 0,
            vocab: f.vocab,
        },
        attention,
        kv: KvStyle::Paged,
        // No recurrent slabs: every gemma-4 layer attends.
        recurrent: None,
        prefill: PrefillStyle::Planless,
        attn_output: AttnOutput::StatedArgs,
        logit_softcap: f.logit_softcap,
        // No ATTENTION cap: gemma-2's `attn_logit_softcapping` is
        // gemma-2's alone, and a zero here is "no cap" rather than a
        // cap at zero — which would flatten every score to `tanh(inf)`.
        attn_logit_softcap: 0.0,
        // The row's own `u32`, carried across unchanged. This was
        // `i32::try_from(..).unwrap_or(0)`, which turned an
        // out-of-range width into "no per-layer embeddings".
        ple_dim: f.ple_dim,
        norm: NormPlacement::Pre,
        // THE EXCEPTION, and the reason this is a field rather than a
        // reading of the placement: gemma-4 sandwiches its norms like
        // every gemma before it and stores a plain multiplier anyway.
        // `forward/mod.rs` fires `NormVariant::Plain` at all fourteen of
        // its norm sites for the same reason.
        norm_unit_offset: false,
        // gemma-4 alone. A weightless per-head RMS over V, run
        // before the KV write. See `Deployment::v_norm`.
        v_norm: true,
        // The ROW's, not the shape's: two gemma-4 checkpoints of one
        // geometry disagree about it.
        // As `metal_facts` records: gemma-4-26b-a4b ships no
        // `norm_topk_prob` key, and its router normalizes.
        norm_topk_prob: true,
        // No router of this family states a scaling factor.
        routed_scaling: 1.0,
        mlp_gate: crate::deployment::MlpGate::GeluTanh,
        scales: scales(f, load),
        // The ROW's, not the shape's: a family label and a context
        // ceiling are facts about a checkpoint and this sees geometry.
        advertised: Advertised::default(),
        // Also the ROW's. Two packages of identical stack ship different
        // encoders — the E-series carries an audio tower and the A4B
        // does not — so a tower cannot be projected from the decoder's
        // widths without inventing one, which is the failure the old
        // `GemmaAudioConfig::default()` had.
        rope_scaling: None,
        towers: Towers::default(),
    }
}

/// One layer's attention facts.
///
/// Split out because there are six of them and five differ by kind;
/// inline, the `map` closure would be the longest thing in this file and
/// the five conditionals would read as one.
fn layer_attention(f: &Gemma4Facts, sliding_window: i32, l: u32) -> LayerAttention {
    let full = f.is_full_attn(l);
    LayerAttention {
        head_dim: f.head_dim_of(l),
        // The other half of the head shape, and the reason this whole
        // generation was refused: a full layer is 4x256 where a sliding
        // one is 16x256 on the 31b, so a page sized at the sliding count
        // runs three quarters past the end of a full layer's K.
        kv_heads: f.kv_heads_of(l),
        // A full-attention layer sees the whole context. `max(0)` is the
        // driver's own clamp: a config stating no window means none, and
        // a negative one reaching a kernel as a length is a fault.
        window: if full { -1 } else { sliding_window.max(0) },
        // ITS OWN INDEX unless it shares. `kv_source` answers `None` for
        // a layer that owns its pages AND for a stack whose every layer
        // shares — the second is `gemma-4-E4B-it-assistant`, whose KV
        // comes from a backbone that is not in this stack, and landing it
        // on itself is the only answer a single deployment can give.
        kv_source: f.kv_source(l).unwrap_or(l),
        // 1.0, and this is the only generation in the catalog where it
        // is: the q and k norms have already applied the scaling, so a
        // `1/sqrt(head_dim)` here would apply it a second time.
        sm_scale: 1.0,
        rope_theta: if full {
            ROPE_THETA_GLOBAL
        } else {
            ROPE_THETA_LOCAL
        },
        // The full layers rotate PARTIALLY (128 of 512, from
        // `partial_rotary_factor: 0.25`); the sliding ones rotate fully.
        // A full rotation is spelled as the head dim rather than as the
        // `0` the field documents as meaning the same thing, because the
        // driver's own derivation emits `max(2, 2*int(0.5*f*d))` for
        // every layer and a deployment that differed from it by a
        // sentinel would compare unequal to one that did not.
        rotary_dim: if full {
            f.global_rotary_dim
        } else {
            f.head_dim
        },
        q_gate: false,
    }
}

/// The named constants gemma-4's traced text refers to.
///
/// Four of them are the generation's own arithmetic, spelled here
/// because the trace names them and a name has to resolve to a number
/// somewhere. The fifth is per LAYER and is not arithmetic at all: the
/// `layer_scalar` tensors are `[1]` values read to host at load, so they
/// arrive through [`Deployed::layer_scalars`] and this is the one row in
/// the catalog that reads that field.
fn scales(f: &Gemma4Facts, load: Deployed<'_>) -> BTreeMap<String, f32> {
    let mut scales = BTreeMap::new();
    let hidden = f.hidden as f32;
    scales.insert("sqrt_hidden".to_string(), hidden.sqrt());
    scales.insert("sqrt_ple_dim".to_string(), (f.ple_dim as f32).sqrt());
    scales.insert("rsqrt_hidden".to_string(), 1.0 / hidden.sqrt());
    scales.insert("rsqrt_2".to_string(), 1.0 / 2f32.sqrt());
    for (n, scalar) in load.layer_scalars.iter().enumerate() {
        scales.insert(format!("layer.{n}.ple_norm"), *scalar);
    }
    scales
}

/// This row as the SHAPE `llama_like_metal` reads.
///
/// The sliding layers' widths, because that is what a shape holds: one
/// `head_dim` and one `kv_heads`. The full layers' are per-layer facts
/// and travel on [`metal_facts`] beside the window list that says which
/// layers they belong to — the same split [`deployment`] makes, where
/// `Geometry::head_dim` is the checkpoint's 256 and the 512 reaches a
/// driver through `LayerAttention`.
#[must_use]
pub fn metal_shape(
    f: &Gemma4Facts,
    mixture: Option<Gemma4Mixture>,
) -> crate::shared::llama_like::spec::LlamaLikeFacts {
    use crate::shared::llama_like::spec::LlamaLikeFacts;
    use model_ir::facts::{NormPlacement as SpecNorm, QkNorm};
    use model_ir::trace::{NormVariant, RopeKind};

    LlamaLikeFacts {
        hidden: f.hidden,
        layers: f.layers,
        q_heads: f.q_heads,
        kv_heads: f.kv_heads,
        head_dim: f.head_dim,
        n_experts: mixture.map_or(0, |m| m.num_experts),
        experts_per_token: mixture.map_or(0, |m| m.experts_per_token),
        moe_intermediate: mixture.map_or(0, |m| m.moe_intermediate),
        // No shared expert: gemma-4's second branch is the DENSE MLP,
        // which `dense_beside_moe` states, not an extra bank.
        shared_intermediate: 0,
        intermediate: f.intermediate,
        vocab: f.vocab,
        // Two plain bases, one per layer kind; `rope_theta_sliding`
        // carries the second and `rope_theta_at` picks.
        rope: RopeKind::Standard,
        // PLAIN, and this is the exception `deployment` states as
        // `norm_unit_offset: false`: gemma-4 sandwiches its norms like
        // every gemma before it and stores a plain multiplier anyway.
        // `forward/mod.rs` fires `NormVariant::Plain` at all fourteen of
        // its norm sites for the same reason, and reading the FAMILY
        // instead of the field is how every gemma-4 norm came out off by
        // `(1 + w)/w`.
        norm_variant: NormVariant::Plain,
        norm_placement: SpecNorm::Sandwich,
        qk_norm: QkNorm::PerHead,
        // FALSE, unlike the CUDA row: no Metal deployment publishes a
        // fused bank. Stated here as well as on `metal_facts` because
        // `llama_like`'s SEMANTIC text reads this one.
        fused_qkv: false,
        tied_embeddings: f.tied_embeddings,
        qkv_bias: false,
        o_bias: false,
        router_bias: false,
    }
}

/// This row as the shape and the binding facts `llama_like_metal` reads.
///
/// # This replaces a refusal that was wrong about the text it named
///
/// `NO_METAL` stood here and said `llama_like_metal` "states the widths
/// without the fused-projection split or the shared-cache attention
/// built on them". It states all three. `LlamaLikeMetalFacts` carries
/// `global_head_dim`,
/// `global_kv_heads`, `full_partial_rotary`, `rope_theta_sliding`,
/// `v_from_k`, `kv_shared_layers`, `per_layer_emb_dim`,
/// `per_layer_scalar`, `embed_scale` and `dense_beside_moe`; the text
/// reads them through `head_dim_at`/`kv_heads_at`/`rotary_dim_at`, and
/// its own comments are measurements taken ON gemma-4-31b — "layer 17's
/// k_proj lowered to `@151552w4096` and the fire's first NaN was at
/// element 2048 of exactly that value".
///
/// What was true is that nothing PROJECTED this row into those facts.
/// `driver-metal/src/model/text.rs` did, from tensor probes, and
/// gemma-4-31b passed all twelve real-weight gates at 5d7e05526 —
/// including `one_token_at_position_zero_agrees_with_mlx`. Deleting it
/// in favour of the catalog moved the projection here and it was never
/// written, so the refusal recorded the gap as a property of the text.
///
/// Every number below is the row's own, and the per-layer ones come off
/// [`Gemma4Facts`]'s helpers — `is_full_attn`, `head_dim_of`,
/// `kv_heads_of`, `kv_source` — which are the SAME helpers
/// [`layer_attention`] uses. One list, asked twice, rather than two that
/// can disagree about which layers are full.
///
/// That a row answers for ITSELF is the whole of what replaced
/// `driver-metal`'s `LLAMA_LIKE` — an eleven-entry table of
/// architecture STRINGS, reduced by a punctuation-stripping
/// `canonical()`, consulted before any text was traced and free to
/// disagree with what the tracer would actually do. It listed
/// `gpt_oss`, which no publication of reaches a Metal device here, and
/// omitted `gemma3`, whose text it models. A row that projects itself
/// cannot disagree with a list, because there is no list; the refusal
/// that briefly stood in its place could, and did.
#[must_use]
pub fn metal_facts(
    f: &Gemma4Facts,
    row: RowScalars,
    bind: &crate::catalog::MetalBinding,
) -> crate::shared::llama_like::forward::facts::LlamaLikeMetalFacts {
    let RowScalars {
        mixture,
        sliding_window,
        norm_eps,
        k_eq_v,
    } = row;
    use crate::shared::llama_like::forward::facts::{Activation, LlamaLikeMetalFacts};
    use model_dsl::{ScaleLayout, WeightRepr};

    LlamaLikeMetalFacts {
        qmm_partial_rows: false,
        // TRUE and unread: the row this projection serves is gemma-4's DENSE
        // stack. `gemma-4-26b-a4b` is the generation's mixture and it is
        // untraced on every backend, and its config states `top_k_experts`
        // with no `norm_topk_prob` at all -- so the day it is traced, this
        // is a measurement it has to make rather than inherit.
        norm_topk_prob: true,
        // ── the LOAD's six, identical to the llama-like projection's ──
        fuse_residual_gemv: bind.fuse_residual_gemv,
        paged_multi_batch: bind.paged_multi_batch,
        qmm_multi_batch: bind.qmm_multi_batch,
        add_bias: bind.add_bias,
        fused_qk_rope: bind.fused_qk_rope,
        proj_repr: WeightRepr::Scaled {
            layout: ScaleLayout::PerGroup,
            group: bind.quant_group,
            axis: 0,
            zero_point: true,
        },
        affine_bits: bind.quant_bits,
        // The ROUTER GATE's own format, when the checkpoint published it
        // wider than the stack it routes. `None` is "the same as the dense
        // projections", which is every checkpoint but gpt-oss's -- and
        // getting it wrong is the QUIET failure: a bank read at the wrong
        // format is 909,207 NaNs, and a gate read at the wrong width is a
        // fluent model routing every token to almost the right experts.
        router_repr: (bind.router_quant_group != 0).then_some(WeightRepr::Scaled {
            layout: ScaleLayout::PerGroup,
            group: bind.router_quant_group,
            axis: 0,
            zero_point: true,
        }),
        router_bits: bind.router_quant_bits,
        moe_repr: bind.moe_mxfp4.then_some(WeightRepr::Mxfp4Marlin),
        moe_bits: 4,
        qmm_tile: crate::shared::llama_like::project::QMM_TILE,
        qmm_fp16_precast: bind.qmm_fp16_precast
            && crate::shared::llama_like::project::qmm_fp16_precast(
                bind.quant_group,
                bind.quant_bits,
            ),
        routed_qmm_fp16: crate::shared::llama_like::project::qmm_fp16_precast(
            bind.quant_group,
            bind.quant_bits,
        ),
        moe_tile: Some(crate::shared::llama_like::project::ROUTED_QMM_TILE),
        // No Metal deployment publishes a fused bank; `compile_load_plan`
        // authors with `Projections::InPlace`.
        gate_up_fused: false,
        rms_eps: norm_eps,
        // The FULL layers' base is the model's, and the SLIDING layers'
        // is the second one. Stating both is what lets `rope_theta_at`
        // pick off the window list rather than a second list of its own.
        rope_theta: ROPE_THETA_GLOBAL,
        rope_theta_sliding: ROPE_THETA_LOCAL,
        // The three that make this row's two attention shapes one text.
        // `head_dim`/`kv_heads` on the shape are the SLIDING layers'; a
        // full layer is twice as wide per head, carries a quarter the KV
        // heads and rotates 128 of its 512 channels.
        global_head_dim: f.global_head_dim,
        global_kv_heads: f.global_kv_heads,
        // A division, not a case. The two attention geometries ARE this
        // family -- every gemma-4 row states a full-layer head dim, and a
        // row answering 0 would be a llama. A guard for it is a branch no
        // row can take, and `every_row_states_both_geometries` is what
        // keeps that true.
        full_partial_rotary: f64::from(f.global_rotary_dim) as f32 / f.global_head_dim as f32,
        // The ROW's, not the shape's: two gemma-4 checkpoints of one
        // geometry disagree about it, which is why `deployment` takes it
        // as a parameter and this does too.
        v_from_k: k_eq_v,
        // TRUE, and gemma-4 alone: a weightless per-head RMS over V, run
        // before the KV write. There is no tensor to ask about it, so
        // nothing in a checkpoint could ever contradict a wrong answer —
        // which is why `Deployment::v_norm` is where it is stated.
        v_norm: true,
        // gemma-4 runs the dense MLP and the bank off the SAME
        // post-attention residual, norms each leg's output, adds them, and
        // norms the sum -- the SEVEN norms round one block its forward
        // states.
        dense_beside_moe: mixture.is_some(),
        // Both measured off `mlx-community/gemma-4-26b-a4b-it-4bit`, which
        // publishes `layers.{n}.router.scale` `[2816]` and
        // `layers.{n}.router.per_expert_scale` `[128]` in every mixture
        // layer and neither in a dense one.
        //
        // Asked of the MIXTURE and not of the tensor list, for the reason
        // `per_layer_scalar` just below is: a row states what it is, and a
        // checkpoint that shipped the tensor without the block would then
        // norm nothing twice rather than silently skip a scale.
        router_input_norm: mixture.is_some(),
        router_expert_scale: mixture.is_some(),
        // TRUE, and asked of the row rather than the tensors: gemma-4-31b
        // states `hidden_size_per_layer_input: 0` and has the scalar
        // anyway, which is the trap the deleted probe fell into from the
        // other side -- "the gemma-shaped fields are populated" and "has
        // a PLE" are not the same question.
        per_layer_scalar: f.ple_dim == 0,
        // `sqrt(hidden)`, which the forward states as `sqrt_hidden`. A
        // gemma that got no scale had a widest gathered value of 0.058
        // where MLX's reference for the same snapshot is seventy times
        // that.
        embed_scale: (f64::from(f.hidden) as f32).sqrt(),
        // ONE, and stated rather than derived. This field READ 0.0 --
        // the sentinel for "derive `1/sqrt(head_dim)`" -- under a
        // comment that called the derivation "the same reading" as the
        // per-layer table's `sm_scale: 1.0`. It is not the same reading
        // and it is not the same NUMBER: a sliding layer would have
        // divided every logit by 16 and a full layer by 22.6, on a
        // family whose `q_norm`/`k_norm` have already divided by the
        // thing the derivation divides by again. MLX's `gemma4_text`
        // says `self.scale = 1.0` for every layer of every gemma-4, and
        // the CUDA text in this same crate has said `sm_scale: 1.0`
        // since it was written, with the reason spelled out at the top
        // of this module.
        //
        // The cost of the wrong reading was INVISIBLE at position zero,
        // which is why it survived the reference gate: with one key the
        // softmax over a single logit is 1.0 at every temperature, so
        // `one_token_at_position_zero_agrees_with_mlx` passed while
        // generation produced English-shaped rubbish. Every position
        // after the first attended at a temperature 16 to 22 times too
        // HOT, which flattens the distribution toward uniform and hands
        // the value path an average of the context instead of a lookup
        // into it.
        attn_scale: 1.0,
        per_layer_emb_dim: f.ple_dim,
        kv_shared_layers: f.kv_shared_layers,
        logit_softcap: f.logit_softcap,
        // gpt-oss's alone.
        attn_sinks: false,
        // The TANH approximation, not the erf one. The two agree to about
        // 2% at the origin and diverge from there.
        activation: Activation::Geglu,
        // Two plain bases, both expressible; no rescaling ladder.
        rope_freq_table: false,
        // TRUE, and it is the whole generation's: `rope_parameters`'
        // `full_attention` arm states `rope_type: proportional` beside
        // `partial_rotary_factor: 0.25`, so a full-attention head rotates 128
        // of its 512 channels PAIRED ACROSS the head and not within the
        // rotated slice. The sliding arm rotates all 256 of its own, where
        // the two readings coincide -- which is why one flag covers the
        // stack.
        rope_proportional: true,
        // The per-layer window list every `*_at` reads to decide which
        // layers are full. Built from `is_full_attn`, the same helper
        // `layer_attention` uses, so the two cannot disagree.
        window_left: (0..f.layers)
            .map(|l| {
                if f.is_full_attn(l) {
                    -1
                } else {
                    sliding_window.max(0)
                }
            })
            .collect(),
    }
}

/// Trace this row's CUDA text for one fire class.
///
/// The backend facts are built HERE rather than held on the row, and the
/// three booleans are why: they say what the LOADER did — whether it
/// fused the QKV bank, whether it fused gate‖up, whether the pages are
/// native bf16 — which is a property of a load and not of a model. The
/// per-layer window list is the row's, and it is not empty: gemma-4 is
/// the generation where an empty list would have the trace say "attends
/// everything" while the plan applied a 512-token window.
#[must_use]
pub fn trace(
    f: &Gemma4Facts,
    sliding_window: i32,
    class: model_ir::trace::FireClass,
    layer_scalars: &[f32],
    norm_eps: f32,
) -> model_ir::trace::ForwardPlan {
    let cuda = super::forward::facts::Gemma4CudaFacts {
        fused_qkv: true,
        gate_up_fused: true,
        kv_native_bf16: true,
        // The LOAD's, like the three booleans above: `layer_scalar` is a
        // per-layer `[1]` tensor, so only something that has opened the
        // checkpoint can state it.
        layer_scalars: layer_scalars.to_vec(),
        window_left: (0..f.layers)
            .map(|l| {
                if f.is_full_attn(l) {
                    -1
                } else {
                    sliding_window.max(0)
                }
            })
            .collect(),
    };
    // THE SHIPPED POINT, read off the family's one axis spelling
    // (`forward::Shipped*`, beside its CATALOG). gemma-4 catalogues one
    // SKU today; the table in `forward::CATALOG` is where a second one
    // appears, and the coverage test is what keeps every row loadable.
    super::forward::gemma4_cuda::<
        super::forward::ShippedW1,
        super::forward::ShippedA,
        super::forward::ShippedKv,
    >(f, &cuda, class, norm_eps)
}

#[cfg(test)]
mod tests {
    use super::{
        Advertised, AttnOutput, Deployed, Gemma4Facts, Gemma4Mixture, KvStyle, NormPlacement,
        PrefillStyle, ROPE_THETA_GLOBAL, ROPE_THETA_LOCAL, RowScalars, deployment, manifest,
    };
    use crate::manifest::{Observed, Presence};

    const E4B_WINDOW: i32 = 512;
    const NORM_EPS: f32 = 1e-6;

    fn e4b() -> crate::deployment::Deployment {
        deployment(
            &Gemma4Facts::gemma_4_e4b(),
            RowScalars {
                mixture: None,
                sliding_window: E4B_WINDOW,
                norm_eps: NORM_EPS,
                k_eq_v: false,
            },
            Deployed::single(),
        )
    }

    fn spec(m: &crate::manifest::Manifest, name: &str) -> crate::manifest::TensorSpec {
        m.tensors
            .iter()
            .find(|t| t.name == name)
            .unwrap_or_else(|| panic!("{name} is stated"))
            .clone()
    }

    /// The manifest names layer 0's widths, and layer 0 is a SLIDING
    /// layer — 8 heads of 256 into 2560, not 8 of 512. A manifest that
    /// used the full layers' width would fail to identify every gemma-4
    /// ever published, which is the failure mode worth a test.
    #[test]
    fn the_projection_widths_are_the_sliding_layers_because_layer_zero_is_one() {
        let f = Gemma4Facts::gemma_4_e4b();
        assert!(
            !f.is_full_attn(0),
            "layer 0 is full, and the manifest below assumes otherwise"
        );
        let m = manifest(&f, None);
        assert_eq!(m.layers, 42);
        assert_eq!(
            spec(&m, "layer.{}.self_attn.q_proj").extents,
            vec![2048, 2560]
        );
        assert_eq!(
            spec(&m, "layer.{}.self_attn.k_proj").extents,
            vec![512, 2560]
        );
        assert_eq!(
            spec(&m, "layer.{}.self_attn.v_proj").extents,
            vec![512, 2560]
        );
        assert_eq!(
            spec(&m, "layer.{}.self_attn.o_proj").extents,
            vec![2560, 2048]
        );
        assert_eq!(spec(&m, "layer.{}.self_attn.q_norm").extents, vec![256]);
        assert_eq!(spec(&m, "layer.{}.self_attn.k_norm").extents, vec![256]);
    }

    /// The dense stack's own rows: one embedding table, four norms, a
    /// three-tensor MLP at the width `intermediate_of` gives, and no
    /// `lm_head` because the head is the embedding table.
    #[test]
    fn the_dense_rows_are_the_gemma_vocabulary() {
        let m = manifest(&Gemma4Facts::gemma_4_e4b(), None);
        assert_eq!(spec(&m, "embed_tokens").extents, vec![262_144, 2560]);
        assert_eq!(spec(&m, "norm").extents, vec![2560]);
        assert_eq!(
            spec(&m, "lm_head").presence,
            Presence::Absent,
            "a tie is an absence"
        );
        for norm in [
            "layer.{}.input_layernorm",
            "layer.{}.post_attention_layernorm",
            "layer.{}.pre_feedforward_layernorm",
            "layer.{}.post_feedforward_layernorm",
        ] {
            assert_eq!(spec(&m, norm).extents, vec![2560], "{norm}");
        }
        assert_eq!(
            spec(&m, "layer.{}.mlp.gate_proj").extents,
            vec![10_240, 2560]
        );
        assert_eq!(spec(&m, "layer.{}.mlp.up_proj").extents, vec![10_240, 2560]);
        assert_eq!(
            spec(&m, "layer.{}.mlp.down_proj").extents,
            vec![2560, 10_240]
        );
    }

    /// The PLE table's second axis is `layers * ple_dim` — the one
    /// tensor extent in this catalog that multiplies a layer count, and
    /// the reason a gemma-4 cannot be mistaken for a llama-like stack.
    #[test]
    fn the_per_layer_embedding_table_multiplies_the_layer_count() {
        let m = manifest(&Gemma4Facts::gemma_4_e4b(), None);
        assert_eq!(
            spec(&m, "embed_tokens_per_layer").extents,
            vec![262_144, 42 * 256]
        );
        assert_eq!(
            spec(&m, "per_layer_model_projection").extents,
            vec![42 * 256, 2560]
        );
        assert_eq!(spec(&m, "per_layer_projection_norm").extents, vec![256]);
        assert_eq!(
            spec(&m, "layer.{}.per_layer_input_gate").extents,
            vec![256, 2560]
        );
        assert_eq!(
            spec(&m, "layer.{}.per_layer_projection").extents,
            vec![2560, 256]
        );
        assert_eq!(
            spec(&m, "layer.{}.post_per_layer_input_norm").extents,
            vec![2560]
        );
    }

    /// The three absences that separate this generation from gemma-3n,
    /// which is the only other one that ships a PLE trio. A checkpoint
    /// publishing any of them is not a gemma-4, and saying so here is
    /// what stops it matching on its way past.
    #[test]
    fn the_absences_are_what_tell_gemma_4_from_gemma_3n() {
        let m = manifest(&Gemma4Facts::gemma_4_e4b(), None);
        for absent in [
            "layer.{}.self_attn.v_norm",
            "layer.{}.altup.modality_router",
            "layer.{}.laurel.linear_left",
        ] {
            assert_eq!(spec(&m, absent).presence, Presence::Absent, "{absent}");
        }
    }

    /// The mixture's rows and the dense stack's are mutually exclusive:
    /// a routed gemma-4 requires the router scale and drops the PLE
    /// trio, a dense one forbids the router and requires the trio. No
    /// checkpoint can satisfy both manifests.
    #[test]
    fn the_router_and_the_per_layer_table_are_mutually_exclusive_rows() {
        let dense = manifest(&Gemma4Facts::gemma_4_e4b(), None);
        let routed = manifest(
            &Gemma4Facts::gemma_4_26b_a4b(),
            Some(Gemma4Mixture::gemma_4_26b_a4b()),
        );
        assert_eq!(
            spec(&dense, "layer.{}.router.scale").presence,
            Presence::Absent
        );
        assert_eq!(
            spec(&routed, "layer.{}.router.scale").presence,
            Presence::Required
        );
        assert_eq!(spec(&routed, "layer.{}.router.scale").extents, vec![2816]);
        assert_eq!(
            spec(&dense, "embed_tokens_per_layer").presence,
            Presence::Required
        );
        assert_eq!(
            spec(&routed, "embed_tokens_per_layer").presence,
            Presence::Absent
        );
        assert!(
            !routed
                .tensors
                .iter()
                .any(|t| t.name == "layer.{}.per_layer_input_gate"),
            "a stack with no per-layer table cannot gate one into its residual"
        );
    }

    /// An untied checkpoint would be required to ship the head, and the
    /// row says so through the same `either` that states the tie. No
    /// published gemma-4 unties, so this exercises the branch the
    /// fixtures do not.
    #[test]
    fn an_untied_head_is_required_rather_than_forbidden() {
        let f = Gemma4Facts {
            tied_embeddings: false,
            ..Gemma4Facts::gemma_4_e4b()
        };
        let m = manifest(&f, None);
        assert_eq!(spec(&m, "lm_head").presence, Presence::Required);
        assert_eq!(spec(&m, "lm_head").extents, vec![262_144, 2560]);
    }

    /// A manifest describes the checkpoint its own numbers imply, which
    /// is the property the whole table rests on.
    #[test]
    fn every_manifest_describes_a_checkpoint_it_would_accept() {
        for (name, f, mixture) in [
            ("e4b", Gemma4Facts::gemma_4_e4b(), None),
            ("e2b", Gemma4Facts::gemma_4_e2b(), None),
            (
                "26b-a4b",
                Gemma4Facts::gemma_4_26b_a4b(),
                Some(Gemma4Mixture::gemma_4_26b_a4b()),
            ),
        ] {
            let m = manifest(&f, mixture);
            let implied = Observed::from_pairs(
                m.tensors
                    .iter()
                    .filter(|t| t.presence != Presence::Absent)
                    .map(|t| (t.name.replace("{}", "0"), t.extents.clone())),
            );
            assert!(
                m.check(&implied).is_ok(),
                "{name}: manifest does not describe itself"
            );
        }
    }

    /// The stack's own count, and the geometry a launch path reads.
    #[test]
    fn the_geometry_is_the_rows_own_numbers() {
        let d = e4b();
        assert_eq!(d.layers, 42);
        assert_eq!(
            d.attention.len(),
            42,
            "one entry per layer, unconditionally"
        );
        assert_eq!(d.shape.hidden, 2560);
        assert_eq!(d.shape.q_heads, 8);
        assert_eq!(d.shape.kv_heads, 2);
        assert_eq!(d.shape.gqa_group(), 4);
        assert_eq!(d.shape.head_dim, 256);
        assert_eq!(d.shape.head_dim_kernel, 256);
        assert_eq!(d.shape.head_dim_alloc(), 256, "nothing is padded at 256");
        assert_eq!(d.shape.intermediate, 10_240);
        assert_eq!(d.shape.vocab, 262_144);
        assert_eq!(d.norm_eps, NORM_EPS);
        assert_eq!(d.logit_softcap, 30.0);
        assert_eq!(d.ple_dim, 256);
        assert_eq!(d.norm, NormPlacement::Pre);
        assert_eq!(d.kv, KvStyle::Paged);
        assert!(d.recurrent.is_none(), "every gemma-4 layer attends");
        assert_eq!(
            d.advertised,
            Advertised::default(),
            "the row fills this, not the projection"
        );
        assert_eq!(
            d.towers,
            crate::deployment::Towers::default(),
            "two rows of one shape ship different encoders, so a projection of the \
             shape cannot know which"
        );
    }

    /// A dense row states `moe_intermediate: 0` — the field means ONE
    /// EXPERT's width, and repeating the dense width there would make
    /// `widest_mlp()` right by accident on this row and wrong on the
    /// next.
    #[test]
    fn a_dense_row_states_no_expert_width_and_a_mixture_states_its_own() {
        let dense = e4b();
        assert_eq!(dense.shape.moe_intermediate, 0);
        assert_eq!(dense.shape.widest_mlp(), 10_240);

        let routed = deployment(
            &Gemma4Facts::gemma_4_26b_a4b(),
            RowScalars {
                mixture: Some(Gemma4Mixture::gemma_4_26b_a4b()),
                sliding_window: 1024,
                norm_eps: NORM_EPS,
                k_eq_v: false,
            },
            Deployed::single(),
        );
        assert_eq!(
            routed.shape.intermediate, 2112,
            "the DENSE width stays in `intermediate`"
        );
        assert_eq!(routed.shape.moe_intermediate, 704);
        assert_eq!(
            routed.shape.widest_mlp(),
            2112,
            "the workspace is sized from the wider of the two, which here is the dense one"
        );
    }

    /// ONLY gemma-4 states its attention output as an SSA arg. It was a
    /// default body reading `pins_attention_values() -> true` whose own
    /// doc said *"Only gemma-4 does"* — an exception hiding in a default,
    /// at inverted polarity.
    #[test]
    fn only_gemma_4_states_its_attention_output_as_an_ssa_arg() {
        assert_eq!(e4b().attn_output, AttnOutput::StatedArgs);
    }

    /// Its 512-wide sliding layers take the naive kernel, which builds
    /// its plan inside the fire from the host CSR mirrors — so there is
    /// no plan to raise and nothing for a driver to bind.
    #[test]
    fn gemma_4_prefill_has_no_plan_to_raise() {
        assert_eq!(e4b().prefill, PrefillStyle::Planless);
    }

    /// The q and k norms carry the scaling, so the softmax scale is 1.0
    /// on every layer. The trait this replaced divided by
    /// `sqrt(head_dim)` in a DEFAULT body, which for this generation
    /// would apply the scaling twice — and on the full layers it would
    /// divide by `sqrt(512)`, a width no gemma-4 kernel scales by.
    #[test]
    fn the_softmax_scale_is_one_because_the_norms_already_scaled() {
        for a in &e4b().attention {
            assert_eq!(a.sm_scale, 1.0);
        }
    }

    /// The two head dims, as a question about the per-layer table rather
    /// than a vtable method. `decode_plan_head_dims()` existed BECAUSE
    /// of this generation.
    #[test]
    fn the_two_layer_kinds_need_two_decode_plans() {
        let d = e4b();
        assert_eq!(d.decode_head_dims(), Some((256, 512)));
        assert_eq!(d.attention[0].head_dim, 256, "layer 0 slides");
        assert_eq!(
            d.attention[5].head_dim, 512,
            "layer 5 is the first full one"
        );
    }

    /// The window schedule: full layers see everything, sliding layers
    /// see 512. The generic derivation broadcast one window to every
    /// layer, and a full-attention layer given a 512-token window
    /// produces fluent text about the wrong prefix.
    #[test]
    fn a_full_attention_layer_sees_the_whole_context() {
        let f = Gemma4Facts::gemma_4_e4b();
        let d = e4b();
        for l in 0..f.layers {
            let want = if f.is_full_attn(l) { -1 } else { E4B_WINDOW };
            assert_eq!(d.attention[l as usize].window, want, "layer {l}");
        }
        assert_eq!(d.windows().iter().filter(|&&w| w == -1).count(), 7);
    }

    /// A config stating no window means none, and the clamp is the
    /// driver's own: a negative length reaching a kernel is a fault, so
    /// `max(0)` turns "unstated" into "zero" rather than into garbage.
    #[test]
    fn an_unstated_window_clamps_to_zero_rather_than_reaching_a_kernel_negative() {
        let d = deployment(
            &Gemma4Facts::gemma_4_e4b(),
            RowScalars {
                mixture: None,
                sliding_window: -1,
                norm_eps: NORM_EPS,
                k_eq_v: false,
            },
            Deployed::single(),
        );
        assert_eq!(
            d.attention[0].window, 0,
            "a sliding layer with no stated window"
        );
        assert_eq!(
            d.attention[5].window, -1,
            "a full layer still sees everything"
        );
    }

    /// THE KV-SHARING TABLE, which is the reason `kv_source` is a field
    /// on a LAYER. Every one of E4B's trailing 18 layers reads the pages
    /// of the last EARLIER layer of its own kind: the shared sliding
    /// layers all land on 22 and the shared full ones on 23.
    #[test]
    fn the_shared_tail_reads_the_last_earlier_layer_of_its_own_kind() {
        let f = Gemma4Facts::gemma_4_e4b();
        let d = e4b();
        assert!(d.shares_kv());
        for l in 0..24 {
            assert_eq!(
                d.attention[l as usize].kv_source, l,
                "layer {l} owns its pages"
            );
        }
        for l in 24..42 {
            let src = d.attention[l as usize].kv_source;
            assert!(
                src < 24,
                "layer {l} sources from {src}, which owns no pages itself"
            );
            assert_eq!(
                f.is_full_attn(src),
                f.is_full_attn(l),
                "layer {l} sources from the other attention kind, whose heads are a \
                 different width"
            );
            assert_eq!(
                d.attention[l as usize].head_dim, d.attention[src as usize].head_dim,
                "layer {l} reads pages laid out for a different head width"
            );
        }
        assert_eq!(d.attention[24].kv_source, 22);
        assert_eq!(d.attention[29].kv_source, 23);
        assert_eq!(d.attention[41].kv_source, 23);
    }

    /// A stack whose every layer shares has no source to name, so each
    /// layer lands on itself. That is `gemma-4-E4B-it-assistant`, whose
    /// KV comes from the backbone it rides — a fact one `Deployment`
    /// cannot express, which is why no row claims that checkpoint.
    #[test]
    fn a_stack_shared_end_to_end_lands_every_layer_on_itself() {
        let f = Gemma4Facts {
            layers: 4,
            kv_shared_layers: 4,
            full_attn_interval: 4,
            hidden: 256,
            q_heads: 4,
            kv_heads: 2,
            intermediate: 2048,
            ple_dim: 0,
            logit_softcap: 0.0,
            ..Gemma4Facts::gemma_4_e4b()
        };
        let d = deployment(
            &f,
            RowScalars {
                mixture: None,
                sliding_window: 512,
                norm_eps: NORM_EPS,
                k_eq_v: false,
            },
            Deployed::single(),
        );
        assert!(!d.shares_kv());
        for l in 0..4u32 {
            assert_eq!(d.attention[l as usize].kv_source, l);
        }
    }

    /// A window and a rope base are the same decision seen twice, so a
    /// projection that emits one per layer emits both: 1e6 on the full
    /// layers, 1e4 on the sliding ones.
    #[test]
    fn the_rope_base_follows_the_layer_kind() {
        let f = Gemma4Facts::gemma_4_e4b();
        let d = e4b();
        for l in 0..f.layers {
            let want = if f.is_full_attn(l) {
                ROPE_THETA_GLOBAL
            } else {
                ROPE_THETA_LOCAL
            };
            assert_eq!(d.attention[l as usize].rope_theta, want, "layer {l}");
        }
        assert_eq!(
            d.theta_by_layer().len(),
            42,
            "the two bases differ, so the table is real"
        );
    }

    /// The full layers rotate 128 of their 512 (`partial_rotary_factor
    /// 0.25`); the sliding ones rotate all 256. Full rotation is spelled
    /// as the head dim rather than as the `0` sentinel, because the
    /// driver's own derivation emits the width for every layer.
    #[test]
    fn only_the_full_layers_rotate_partially() {
        let f = Gemma4Facts::gemma_4_e4b();
        let d = e4b();
        for l in 0..f.layers {
            let want = if f.is_full_attn(l) { 128 } else { 256 };
            assert_eq!(d.attention[l as usize].rotary_dim, want, "layer {l}");
        }
        assert_eq!(d.rotary_by_layer().len(), 42);
    }

    /// The four named constants the traced text refers to, and the fifth
    /// that is not a constant at all: `layer.{n}.ple_norm` comes from
    /// the `[1]` tensors this rank read to host, one per layer, and this
    /// is the only row in the catalog that reads `Deployed::layer_scalars`.
    #[test]
    fn the_host_scalars_reach_the_trace_by_name() {
        let bare = e4b();
        assert_eq!(
            bare.scales.len(),
            4,
            "no layer scalars were handed to this load"
        );
        assert_eq!(bare.scales["sqrt_hidden"], 2560f32.sqrt());
        assert_eq!(bare.scales["sqrt_ple_dim"], 256f32.sqrt());
        assert_eq!(bare.scales["rsqrt_hidden"], 1.0 / 2560f32.sqrt());
        assert_eq!(bare.scales["rsqrt_2"], 1.0 / 2f32.sqrt());

        let scalars = [0.5f32, 0.25, 0.125];
        let loaded = deployment(
            &Gemma4Facts::gemma_4_e4b(),
            RowScalars {
                mixture: None,
                sliding_window: E4B_WINDOW,
                norm_eps: NORM_EPS,
                k_eq_v: false,
            },
            Deployed {
                // Stated, not defaulted: this build's gemma-4 text is
                // CUDA-only, and a row that fell through to a default
                // backend would be the same silent assumption the
                // deleted `LLAMA_LIKE` table made from the other side.
                backend: crate::catalog::Backend::Cuda,
                tp_size: 1,
                layer_scalars: &scalars,
            },
        );
        assert_eq!(loaded.scales.len(), 7);
        assert_eq!(loaded.scales["layer.0.ple_norm"], 0.5);
        assert_eq!(loaded.scales["layer.1.ple_norm"], 0.25);
        assert_eq!(loaded.scales["layer.2.ple_norm"], 0.125);
    }

    /// A stack with no PLE says so, and the scale it would have needed
    /// is zero rather than absent — the trace names it either way.
    #[test]
    fn a_stack_with_no_per_layer_table_states_a_zero_width() {
        let d = deployment(
            &Gemma4Facts::gemma_4_26b_a4b(),
            RowScalars {
                mixture: Some(Gemma4Mixture::gemma_4_26b_a4b()),
                sliding_window: 1024,
                norm_eps: NORM_EPS,
                k_eq_v: false,
            },
            Deployed::single(),
        );
        assert_eq!(d.ple_dim, 0);
        assert_eq!(d.scales["sqrt_ple_dim"], 0.0);
    }

    /// E2B's own schedule, which is the fixture where nothing divides
    /// evenly: 35 layers on an interval of 5, so the LAST layer is a
    /// full one, and 20 of the 35 share KV.
    #[test]
    fn the_second_geometry_schedules_its_last_layer_full() {
        let f = Gemma4Facts::gemma_4_e2b();
        let d = deployment(
            &f,
            RowScalars {
                mixture: None,
                sliding_window: 512,
                norm_eps: NORM_EPS,
                k_eq_v: false,
            },
            Deployed::single(),
        );
        assert_eq!(d.attention.len(), 35);
        assert_eq!(d.attention[34].head_dim, 512, "layer 34 is a full layer");
        assert_eq!(d.attention[34].window, -1);
        assert_eq!(d.decode_head_dims(), Some((256, 512)));
        assert!(d.shares_kv());
        for l in 0..15u32 {
            assert_eq!(d.attention[l as usize].kv_source, l);
        }
        assert_eq!(
            d.attention[15].kv_source, 13,
            "the last earlier sliding layer"
        );
        assert_eq!(d.attention[34].kv_source, 14, "the last earlier full layer");
    }

    /// gemma-4 sandwiches its norms and stores a PLAIN multiplier, and it is
    /// the only stack that does both.
    ///
    /// The row is the only place that can say so. Every driver that inferred
    /// this from the norm placement got it wrong here and did not fail
    /// loudly: `(1 + w)/w` is 1.002 where `w` is 444 and 1.38 where `w` is
    /// 2.6, so the largest gains agreed to three digits while the ordinary
    /// ones were off by a third.
    #[test]
    fn the_sandwich_does_not_imply_the_fold() {
        for d in [
            e4b(),
            deployment(
                &Gemma4Facts::gemma_4_e2b(),
                RowScalars {
                    mixture: None,
                    sliding_window: 512,
                    norm_eps: NORM_EPS,
                    k_eq_v: false,
                },
                Deployed::single(),
            ),
        ] {
            assert!(
                !d.norm_unit_offset,
                "gemma-4 stores the multiplier; `forward/mod.rs` fires \
                 `NormVariant::Plain` at all fourteen of its norm sites"
            );
        }
    }

    /// The one row-scalar `deployment` does not read, and the reason a
    /// gemma-4-31b serves on Metal at all.
    ///
    /// `k_eq_v` says the full-attention layers read V out of the K
    /// projection, so the checkpoint ships no `v_proj`. Three consumers
    /// disagree about what to do with that and all three are right:
    /// `metal_facts` hands it on as `v_from_k` and the Metal text serves
    /// the row; `Variant::trace` refuses it on CUDA, whose hand-written
    /// text matmuls a `v_proj` unconditionally; and this projection
    /// ignores it, because what a rank ALLOCATES is unchanged — the KV
    /// head count is a separate measurement.
    ///
    /// It had no test on the `true` side anywhere. Every constructor in
    /// this module passes `false`, so the carry-through that makes the
    /// claim true was only ever exercised at its uninteresting value.
    #[test]
    fn reading_v_out_of_k_reaches_the_metal_text_and_changes_no_allocation() {
        use crate::catalog::MetalBinding;
        let f = Gemma4Facts::gemma_4_e4b();
        let row = |k_eq_v| RowScalars {
            mixture: None,
            sliding_window: E4B_WINDOW,
            norm_eps: NORM_EPS,
            k_eq_v,
        };
        let bind = MetalBinding {
            qmm_partial_rows: false,
            qmm_fp16_precast: true,
            qmm_tile: None,
            quant_group: 64,
            quant_bits: 4,
            router_quant_group: 0,
            router_quant_bits: 0,
            moe_mxfp4: false,
            fuse_residual_gemv: true,
            paged_multi_batch: true,
            qmm_multi_batch: true,
            add_bias: false,
            fused_qk_rope: false,
        };
        for k_eq_v in [false, true] {
            assert_eq!(
                super::metal_facts(&f, row(k_eq_v), &bind).v_from_k,
                k_eq_v,
                "the measurement reaches the text that branches on it"
            );
        }
        assert_eq!(
            deployment(&f, row(true), Deployed::single()),
            deployment(&f, row(false), Deployed::single()),
            "a projection that is not V's does not change what a rank reserves"
        );
    }

    /// The two texts must attend at the SAME temperature, and for eight
    /// months they did not.
    ///
    /// `metal_facts` read `attn_scale: 0.0`, which is the sentinel
    /// `model_dsl::metal::sdpa` reads as "derive `1/sqrt(head_dim)`",
    /// under a comment claiming the derivation was "the same reading" as
    /// the per-layer table's `sm_scale: 1.0`. It is not the same number:
    /// a sliding layer divided every logit by 16 and a full one by
    /// 22.6, on the one family in this catalog whose `q_norm` and
    /// `k_norm` have already applied the scaling.
    ///
    /// The defect was invisible to the reference gate, because with a
    /// SINGLE key the softmax over one logit is 1.0 at any temperature
    /// -- so a position-zero comparison against MLX agreed to the top-5
    /// while generated text was English-shaped rubbish.
    ///
    /// Asserted as a RELATION between the two statements rather than
    /// against a literal, because the number that matters is that CUDA's
    /// table and the Metal facts say the same thing about the same row.
    /// Over every shipped gemma-4, so a row added tomorrow is covered.
    #[test]
    fn the_metal_softmax_scale_is_the_one_the_per_layer_table_states() {
        use crate::catalog::MetalBinding;
        let bind = MetalBinding {
            qmm_partial_rows: false,
            qmm_fp16_precast: true,
            qmm_tile: None,
            quant_group: 64,
            quant_bits: 4,
            router_quant_group: 0,
            router_quant_bits: 0,
            moe_mxfp4: false,
            fuse_residual_gemv: true,
            paged_multi_batch: true,
            qmm_multi_batch: true,
            add_bias: false,
            fused_qk_rope: false,
        };
        for (name, f, mixture, sliding_window, k_eq_v) in [
            ("gemma-4-e4b", Gemma4Facts::gemma_4_e4b(), None, 512, false),
            ("gemma-4-31b", Gemma4Facts::gemma_4_31b(), None, 1024, true),
            (
                "gemma-4-26b-a4b",
                Gemma4Facts::gemma_4_26b_a4b(),
                Some(Gemma4Mixture::gemma_4_26b_a4b()),
                1024,
                true,
            ),
        ] {
            let row = RowScalars {
                mixture,
                sliding_window,
                norm_eps: NORM_EPS,
                k_eq_v,
            };
            let d = deployment(&f, row.clone(), Deployed::single());
            let m = super::metal_facts(&f, row, &bind);
            assert!(
                d.attention.iter().all(|a| a.sm_scale == 1.0),
                "{name}: the per-layer table states 1.0 for every layer"
            );
            assert_eq!(
                m.attn_scale, d.attention[0].sm_scale,
                "{name}: the Metal text must not derive a scale this row STATES"
            );
        }
    }
    /// gemma row is.
    ///
    /// `LlamaLikeMetalFacts::gemma_like()` is what the gemma4 Metal
    /// texts fire, and its own doc says it is not a measurement -- the
    /// widths are plausible rather than any published config's. So
    /// equality with a projection is the wrong relation, and there is no
    /// single row to compare it against anyway.
    ///
    /// The right relation is weaker and still total: a field that every
    /// shipped gemma-4 moves off `synthetic()`, this fixture must move
    /// too. Because a field left at the default is not a smaller value
    /// of that field, it is the OTHER BRANCH -- `embed_scale: 0.0`
    /// selects `embed_gather` over `embed_gather_scaled`,
    /// `global_head_dim: 0` says one attention shape for the whole
    /// stack, `full_partial_rotary: 0.0` says rotate every channel.
    ///
    /// Every-row and not any-row, because the three rows disagree with
    /// each other and a fixture is allowed to pick a side: `e4b` carries
    /// the per-layer embeddings and so states `per_layer_scalar: false`
    /// where the two without a PLE state `true`, and only the A4B norms
    /// its V. Those are the fields no fixture in this crate exercises,
    /// and the honest reason is that one fixture cannot be three rows.
    ///
    /// Read through `serde` rather than by naming fields, because the
    /// defects this fixture family has had were all in fields no
    /// enumeration contained -- `embed_scale` here, `attn_scale` and
    /// `rope_freq_table` in gpt-oss's. A field added to the struct
    /// tomorrow is covered the day it is added.
    #[test]
    fn the_shared_gemma_fixture_moves_wherever_every_gemma_row_moves() {
        use crate::shared::llama_like::forward::facts::LlamaLikeMetalFacts;
        // gemma-4's OWN binding, the one `catalog_backends` publishes --
        // five of these fields are a deployment's observation rather
        // than the row's claim, so reading them off anything else would
        // compare the fixture against a gemma nobody ships.
        let bind = crate::catalog::MetalBinding {
            qmm_partial_rows: false,
            qmm_fp16_precast: true,
            qmm_tile: None,
            quant_group: 64,
            quant_bits: 4,
            router_quant_group: 0,
            router_quant_bits: 0,
            moe_mxfp4: false,
            fuse_residual_gemv: true,
            paged_multi_batch: true,
            qmm_multi_batch: true,
            add_bias: false,
            fused_qk_rope: false,
        };
        let row = RowScalars {
            mixture: None,
            sliding_window: E4B_WINDOW,
            norm_eps: NORM_EPS,
            k_eq_v: false,
        };
        let fields = |m: &LlamaLikeMetalFacts| match serde_json::to_value(m) {
            Ok(serde_json::Value::Object(o)) => o,
            other => panic!("these facts serialise as a struct, not {other:?}"),
        };
        let plain = fields(&LlamaLikeMetalFacts::synthetic());
        let fixture = fields(&LlamaLikeMetalFacts::gemma_like());

        // Two fields the projection moves that nothing then READS,
        // because each is ANDed with a shape fact this deployment
        // states as false. Written as a table whose reason is checked
        // rather than asserted away: if either operand ever becomes
        // true the entry stops being an excuse and this test says so.
        let pair = crate::shared::llama_like::spec::LlamaLikeFacts::qwen3_0_6b();
        let unread = [
            ("add_bias", pair.qkv_bias || pair.router_bias),
            ("moe_bits", bind.moe_mxfp4),
        ];
        for (name, operand) in unread {
            assert!(
                !operand,
                "`{name}` is excused here only because the text never \
                 reaches it; it does now, so the fixture owes a value"
            );
        }

        let mut owed: Vec<String> = Vec::new();
        for name in plain.keys() {
            if unread.iter().any(|(n, _)| n == name) || fixture[name] != plain[name] {
                continue;
            }
            let every = [
                Gemma4Facts::gemma_4_e4b(),
                Gemma4Facts::gemma_4_31b(),
                Gemma4Facts::gemma_4_26b_a4b(),
            ]
            .into_iter()
            .map(|f| fields(&super::metal_facts(&f, row, &bind))[name].clone())
            .collect::<Vec<_>>();
            if every.iter().all(|v| v != &plain[name]) {
                owed.push(format!(
                    "{name}: every gemma moves it ({}), the fixture stays \
                     at {}",
                    every
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join("/"),
                    plain[name],
                ));
            }
        }
        assert!(
            owed.is_empty(),
            "a field left at the default is the OTHER branch, under a \
             gemma name:\n  {}",
            owed.join("\n  "),
        );
    }

    /// `full_partial_rotary` is a FRACTION, and the shape it is divided
    /// out of does not appear beside it — so an inverted division reads
    /// as a plausible float and nothing downstream refuses it. The
    /// consumer clamps with `want.min(dim)`, which turns 4.0 into "rotate
    /// every channel" rather than into an error: a full gemma-4 layer
    /// would rotate 512 channels where the checkpoint rotates 128, and
    /// the model would produce fluent nonsense at long range.
    ///
    /// Asserted where the fraction becomes a WIDTH rather than on the
    /// float, because the width is the thing the kernel is launched with
    /// and the two attention geometries answer it differently.
    #[test]
    fn a_full_layer_rotates_a_quarter_of_its_channels_and_a_sliding_one_all_of_them() {
        use crate::catalog::MetalBinding;
        let f = Gemma4Facts::gemma_4_e4b();
        let m = super::metal_facts(
            &f,
            RowScalars {
                mixture: None,
                sliding_window: E4B_WINDOW,
                norm_eps: NORM_EPS,
                k_eq_v: false,
            },
            &MetalBinding {
                qmm_partial_rows: false,
                qmm_fp16_precast: true,
                qmm_tile: None,
                quant_group: 64,
                quant_bits: 4,
                router_quant_group: 0,
                router_quant_bits: 0,
                moe_mxfp4: false,
                fuse_residual_gemv: true,
                paged_multi_batch: true,
                qmm_multi_batch: true,
                add_bias: false,
                fused_qk_rope: false,
            },
        );
        let full = (0..f.layers)
            .find(|&l| f.is_full_attn(l))
            .expect("gemma-4 interleaves full layers");
        let sliding = (0..f.layers)
            .find(|&l| !f.is_full_attn(l))
            .expect("gemma-4 interleaves sliding layers");
        assert_eq!(
            m.rotary_dim_at(full, f.head_dim),
            f.global_rotary_dim,
            "the full layer rotates the row's stated {} of {} channels",
            f.global_rotary_dim,
            f.global_head_dim,
        );
        assert_eq!(
            m.rotary_dim_at(sliding, f.head_dim),
            f.head_dim,
            "the fraction is the FULL layers' alone; a sliding layer \
             rotates every one of its own channels"
        );
    }
}

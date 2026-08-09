//! The three projections a Kimi-K3 row makes: its tensor manifest, its
//! `Deployment`, and its traced text.
//!
//! The generation is a HYBRID, and every projection here is shaped by
//! that one fact: two thirds of the stack are KDA linear attention with
//! a recurrent state, one third is MLA with a paged latent, and a
//! manifest, a deployment and a trace each have to say so in their own
//! vocabulary. Written against [`KimiK3Facts`] rather than against the
//! row type, for `kimi_k2/project.rs`'s reason: the numbers are the
//! row's and the arithmetic over them is the generation's.

use crate::deployment::{
    Advertised, AttnOutput, Deployment, Geometry, KvStyle, LayerAttention, NormPlacement,
    PrefillStyle, RecurrentShape, Refusal,
};
use crate::manifest::{Manifest, TensorSpec};

use super::spec::KimiK3Facts;

/// This row's tensors.
///
/// # A hybrid's per-layer rows are a UNION, and they have to be
///
/// [`crate::manifest::Observed::logical`] rewrites `layers.<n>.` to
/// `layer.{}.`, so every layer of a stack collapses onto one key and a
/// manifest cannot say "layer 3 attends and layer 2 does not". What a
/// K3 checkpoint publishes under the collapsed name is therefore the
/// UNION over its layer kinds — the KDA block's tensors AND the MLA
/// block's — and that is what is stated below. The SCHEDULE is not lost,
/// it is `full_attn_interval` on the row, and [`deployment`] and the
/// trace are where it is read.
///
/// # What is deliberately NOT named
///
/// **The routed expert bank.** `contract::author_kimi_k3` reads it as
/// `block_sparse_moe.experts.<e>.{w1,w3,w2}.weight_packed` plus a
/// `.weight_scale` beside each — MXFP4, two E2M1 codes per byte, one
/// E8M0 exponent per group of 32. Every one of those extents is a
/// statement about the PACKING and not about the model: the same
/// checkpoint published bf16 would ship `[intermediate, latent]` under
/// different names, and a manifest that named either would key identity
/// on an encoding. The ROUTER is named instead, because it is the one
/// row of a mixture that is never quantized and whose extents ARE the
/// mixture: `[num_experts, hidden]`.
///
/// **`A_log` and `dt_bias`.** They are named — but their extent is the
/// checkpoint's own PADDING and not the row's arithmetic.
/// `contract::author_kimi_k3`'s band exists precisely because `A_log`
/// ships `[head_dim]` entries for `[value_heads]` real heads (F32[128]
/// where the layer has 96), so a manifest that stated `[value_heads]`
/// would fault every real checkpoint and one that stated `[head_dim]`
/// would be stating the allocator's rounding. Neither is a measurement,
/// so this states the bank's presence at the width the CONTRACT keeps —
/// see the `a_log` row below, which is `Optional` for exactly this
/// reason.
/// The width two layer kinds agree on for a name they BOTH publish, or
/// `None` where they disagree.
///
/// A hybrid's per-layer names collapse onto one key — `layer.{}.` covers
/// the whole stack — so `self_attn.o_proj` is one row standing for two
/// different projections: the MLA layers' reads the value width, the KDA
/// layers' reads the delta-net width. Where those numbers agree (they do
/// on every K3 published so far, because both are `heads * head_dim`) the
/// row is checkable and is stated. Where they disagree there is no honest
/// row to write, and stating either width would fault a checkpoint that
/// is perfectly well formed — so nothing is stated, which is a manifest
/// making no claim rather than a manifest making a wrong one.
#[must_use]
fn shared_width(mla: Option<u64>, kda: Option<u64>) -> Option<u64> {
    match (mla, kda) {
        (Some(a), Some(b)) if a == b => Some(a),
        (Some(w), None) | (None, Some(w)) => Some(w),
        _ => None,
    }
}

#[must_use]
pub fn manifest(f: &KimiK3Facts, tied_embeddings: bool) -> Manifest {
    let (hidden, vocab) = (u64::from(f.hidden), u64::from(f.vocab));
    let a = &f.attn;
    let k = &f.kda;
    let latent_q = a.q_lora_rank > 0;
    let q_lora = u64::from(a.q_lora_rank);
    let kv_lora = u64::from(a.kv_lora_rank);
    let q_b_width = u64::from(a.q_b_width());
    let kv_a_width = u64::from(a.kv_a_width());
    // What the latent is read back OUT to: every head's nope half and
    // every head's value. The one extent that states the compression
    // ratio, and the one a sibling MLA row cannot accidentally match.
    let kv_b_width = u64::from(a.heads * (a.qk_nope_head_dim + a.v_head_dim));
    let v_width = u64::from(a.v_width());
    let kda_width = u64::from(k.width());
    let dense_inter = u64::from(f.dense_intermediate);
    let has_dense_prefix = f.dense_layers > 0;
    let all_dense = f.dense_layers >= f.layers;
    // A stack with no periodic full layer ships no MLA tensors at all —
    // which is what an interval of zero says, and the reading
    // `model_compiler::facts::full_attn_at` settled.
    let has_mla = (0..f.layers).any(|l| f.is_full_attn(l));
    let has_kda = (0..f.layers).any(|l| !f.is_full_attn(l));
    // The three names BOTH halves publish. See [`shared_width`]: one
    // logical key, two projections, and a row only where they agree.
    let q_width = shared_width(
        (has_mla && !latent_q).then_some(q_b_width),
        has_kda.then_some(kda_width),
    );
    let o_width = shared_width(has_mla.then_some(v_width), has_kda.then_some(kda_width));
    let g_width = shared_width(
        (has_mla && a.output_gate).then_some(v_width),
        has_kda.then_some(kda_width),
    );

    Manifest::new(f.layers)
        .with(TensorSpec::required("embed_tokens", [vocab, hidden]))
        .with(TensorSpec::required("norm", [hidden]))
        // TIED vs UNTIED as presence, which is the only way a manifest
        // can tell them apart: every extent agrees.
        .either(!tied_embeddings, "lm_head", [vocab, hidden])
        .with(TensorSpec::required("layer.{}.input_layernorm", [hidden]))
        .with(TensorSpec::required(
            "layer.{}.post_attention_layernorm",
            [hidden],
        ))
        // ── The MLA third ────────────────────────────────────────────
        .either(
            has_mla && latent_q,
            "layer.{}.self_attn.q_a_proj",
            [q_lora, hidden],
        )
        .either(
            has_mla && latent_q,
            "layer.{}.self_attn.q_b_proj",
            [q_b_width, q_lora],
        )
        // The straight-projection alternative, forbidden when there is a
        // rank: the two cannot both be published. Claimed as an ABSENCE
        // only where the KDA half is not publishing the same name — in a
        // hybrid, `q_proj` is a KDA layer's own tensor and forbidding it
        // would fault every real checkpoint.
        .with_if(
            has_mla && latent_q && !has_kda,
            TensorSpec::absent("layer.{}.self_attn.q_proj"),
        )
        .with_if(
            q_width.is_some(),
            TensorSpec::required(
                "layer.{}.self_attn.q_proj",
                [q_width.unwrap_or_default(), hidden],
            ),
        )
        .with_if(
            has_mla && latent_q,
            TensorSpec::required("layer.{}.self_attn.q_a_layernorm", [q_lora]),
        )
        .either(
            has_mla,
            "layer.{}.self_attn.kv_a_proj_with_mqa",
            [kv_a_width, hidden],
        )
        .either(has_mla, "layer.{}.self_attn.kv_a_layernorm", [kv_lora])
        .either(
            has_mla,
            "layer.{}.self_attn.kv_b_proj",
            [kv_b_width, kv_lora],
        )
        // `o_proj` reads the VALUE width and not the query width, which
        // is where MLA stops looking like GQA: for this row they differ
        // by 64 per head. Both halves land through this name, so the row
        // exists only where both mean the same width.
        .with_if(
            o_width.is_some(),
            TensorSpec::required(
                "layer.{}.self_attn.o_proj",
                [hidden, o_width.unwrap_or_default()],
            ),
        )
        // The MLA output gate. kimi-k2 and glm-5 land the absorb straight
        // into `o_proj`; K3 gates it first, and the gate is a projection
        // to the VALUE width. An all-MLA stack that does NOT gate forbids
        // the name, which is what tells the two apart — in a hybrid the
        // KDA half publishes `g_proj` regardless, so no absence can be
        // claimed there.
        .with_if(
            has_mla && !a.output_gate && !has_kda,
            TensorSpec::absent("layer.{}.self_attn.g_proj"),
        )
        .with_if(
            g_width.is_some(),
            TensorSpec::required(
                "layer.{}.self_attn.g_proj",
                [g_width.unwrap_or_default(), hidden],
            ),
        )
        // ── The KDA two thirds ───────────────────────────────────────
        //
        // Three projections and three short convolutions, one per
        // projection: q, k and v each go through their own depthwise
        // causal conv before they ever meet, which is why
        // `author_kimi_k3` deliberately does NOT join them into a fused
        // QKV the way a llama-like row would. `q_proj` is stated above,
        // with the other two names both halves share.
        .either(has_kda, "layer.{}.self_attn.k_proj", [kda_width, hidden])
        .either(has_kda, "layer.{}.self_attn.v_proj", [kda_width, hidden])
        // The decay gate through its rank-`value_head_dim` bottleneck —
        // the two tensors that make the decay per-CHANNEL rather than
        // qwen3_5's per-head scalar. `f_a_proj` is replicated under TP
        // for the same reason it is small: every head reads all of it.
        .either(
            has_kda,
            "layer.{}.self_attn.f_a_proj",
            [u64::from(k.value_head_dim), hidden],
        )
        .either(
            has_kda,
            "layer.{}.self_attn.f_b_proj",
            [kda_width, u64::from(k.value_head_dim)],
        )
        // One beta row per KDA head — `[value_heads, hidden]`, and NOT to
        // be confused with `q_b_proj`, which is MLA's and is `[q_b_width,
        // q_lora_rank]`. The contract reads its head count off exactly
        // this tensor, because that is the only place the real (unpadded)
        // head count is written down.
        .either(
            has_kda,
            "layer.{}.self_attn.b_proj",
            [u64::from(k.value_heads), hidden],
        )
        // Per-head-channel, inside a head: this is the norm the gated
        // output passes through, and its width is the head's and not the
        // stack's.
        .either(
            has_kda,
            "layer.{}.self_attn.o_norm",
            [u64::from(k.value_head_dim)],
        )
        // The gate bank ships PADDED — see the module doc — so its
        // extent is the checkpoint's allocation rounding and not the
        // row's arithmetic. Stated as presence at the row's own head
        // count, `Optional` so a checkpoint that rounded it up is not
        // reported as a fault by a rule that has no measurement behind
        // it.
        .with_if(
            has_kda,
            TensorSpec::optional("layer.{}.self_attn.A_log", [u64::from(k.value_heads)]),
        )
        // ── The dense prefix and the mixture ─────────────────────────
        //
        // `first_k_dense_replace` is a fact a checkpoint publishes: a
        // stack with a prefix ships a dense MLP (from its leading layer)
        // AND a router (from the rest), and every logical name collapses
        // over the stack, so both appear. A mixture with no prefix ships
        // no dense MLP at all, and one that is dense all the way ships no
        // router — the same statement read from its two ends.
        .either(
            has_dense_prefix,
            "layer.{}.mlp.gate_proj",
            [dense_inter, hidden],
        )
        .either(
            has_dense_prefix,
            "layer.{}.mlp.down_proj",
            [hidden, dense_inter],
        )
        .either(
            !all_dense,
            "layer.{}.block_sparse_moe.gate",
            [u64::from(f.moe.num_experts), hidden],
        )
        .with_if(
            f.moe.has_shared_expert(),
            TensorSpec::required(
                "layer.{}.block_sparse_moe.shared_expert.gate_proj",
                [u64::from(f.moe.shared_intermediate), hidden],
            ),
        )
        // The attention-residual blend's own two tensors, which exist
        // only where a block opens. They are this generation's alone —
        // no other row in the catalog blends a prefix across layers — so
        // their presence is what tells a K3 checkpoint from an MLA
        // sibling even before an extent is compared.
        .either(
            f.attn_res_block > 0,
            "layer.{}.self_attention_res_proj",
            [1, hidden],
        )
        .either(
            f.attn_res_block > 0,
            "layer.{}.self_attention_res_norm",
            [hidden],
        )
}

/// This row's deployment, or a refusal at the DOOR.
///
/// [`Deployment::advertised`] is left DEFAULT here, and deliberately: a
/// projection sees geometry and nothing else, while none of that
/// struct's three answers is a shape. An arch label is a coarse family
/// name a guest program matches on, and a context ceiling is a
/// training-time fact two checkpoints of identical geometry can disagree
/// about. The ROW states them, over the top of this.
///
/// # Errors
///
/// [`Refusal::Unsupported`] when this build provisions no store for the
/// row's [`KvStyle`] — which for this generation is every build, and is
/// the honest answer rather than a load that dies at its first fire.
pub fn deployment(f: &KimiK3Facts, rope_theta: f32, norm_eps: f32) -> Result<Deployment, Refusal> {
    let planned = plan(f, rope_theta, norm_eps);
    match planned.kv.store_refusal() {
        Some(no_store) => Err(no_store),
        None => Ok(planned),
    }
}

/// The projection itself, which is TOTAL: a row's deployment is a fact
/// about the row and exists whether or not this build can serve it.
///
/// Separate from [`deployment`] so the two statements stay separable —
/// "what this model needs" is the row's, "what this binary provides" is
/// [`KvStyle::has_a_store_in_this_build`]'s, and collapsing them is how a capability
/// question turns back into a family name.
#[must_use]
fn plan(f: &KimiK3Facts, rope_theta: f32, norm_eps: f32) -> Deployment {
    let a = &f.attn;
    // MLA's page row holds the LATENT plus the one shared rope half, and
    // not a head-split key — which is what this generation's
    // `head_dim_of` answered for EVERY layer, KDA ones included. It is
    // right to repeat it: a KDA layer owns no pages at all, so the only
    // number a pool allocator can read off it is the one the MLA layers
    // need.
    let page_row = a.kv_a_width();
    // The scale is over the DOT's width — the query's `nope + rope`
    // (192 here) and not the page row (320). A latent is what is STORED,
    // `qk_head_dim` is what is MULTIPLIED, and the vtable this replaces
    // took `1/sqrt(head_dim_of(l))` for every family that did not
    // override it.
    let sm_scale = 1.0 / (a.qk_head_dim() as f32).sqrt();
    let attention = (0..f.layers)
        .map(|l| LayerAttention {
            // One shape for every layer, which is what this row was
            // already saying by having no per-layer count.
            kv_heads: 1,
            head_dim: page_row,
            // No window anywhere: the KDA layers carry a RECURRENCE
            // rather than a bounded context, and the MLA layers attend
            // the whole of it. A window here would be a driver dropping
            // pages the model reads.
            window: -1,
            kv_source: l,
            sm_scale,
            // ZERO, and deliberately. This generation's MLA carries no
            // rope — `forward::kimi_k3_cuda` says so in its own words
            // ("there is deliberately no `kernels::rope::rope_bf16`
            // here") because the positional information rides the KDA
            // layers instead. The derivation this replaces filled every
            // layer of every family with `config.rope_theta`, so a
            // checkpoint that stated a theta it never rotates by handed
            // the driver a rope table for a rotation the text does not
            // contain.
            rope_theta: if f.is_full_attn(l) { 0.0 } else { rope_theta },
            rotary_dim: 0,
        })
        .collect();

    Deployment {
        layers: f.layers,
        norm_eps,
        shape: Geometry {
            hidden: f.hidden,
            q_heads: a.heads,
            // ONE: MLA reads a single latent plane per token, so every
            // query head shares it.
            kv_heads: 1,
            head_dim: page_row,
            // Nothing pads it — a latent row is not a head width a
            // kernel is instantiated at.
            head_dim_kernel: page_row,
            intermediate: f.dense_intermediate,
            // One expert's inner width, which is a DIFFERENT number from
            // the dense prefix's — 1024 against 5632 — and the forward
            // workspace is one buffer both layer kinds share, so both are
            // stated and the planner takes the wider.
            moe_intermediate: f.moe.moe_intermediate,
            experts_per_token: f.moe.top_k,
            shared_intermediate: f.moe.shared_intermediate,
            vocab: f.vocab,
        },
        attention,
        kv: KvStyle::Mla {
            kv_lora_rank: a.kv_lora_rank,
            qk_rope_head_dim: a.qk_rope_head_dim,
        },
        // The two-thirds of the stack a paged store knows nothing about.
        recurrent: Some(kda_shape(f)),
        prefill: PrefillStyle::Planned,
        attn_output: AttnOutput::DriverPinned,
        logit_softcap: 0.0,
        // No ATTENTION cap: gemma-2's `attn_logit_softcapping` is
        // gemma-2's alone, and a zero here is "no cap" rather than a
        // cap at zero — which would flatten every score to `tanh(inf)`.
        attn_logit_softcap: 0.0,
        ple_dim: 0,
        norm: NormPlacement::Pre,
        // Not a gemma: the gain is the multiplier, stored directly.
        norm_unit_offset: false,
        v_norm: false,
        norm_topk_prob: f.moe.norm_topk_prob,
        routed_scaling: f.moe.routed_scaling,
        mlp_gate: crate::deployment::MlpGate::Silu,
        scales: std::collections::BTreeMap::new(),
        // DEFAULT, and the row writes over it. None of the three
        // answers in here is geometry: an arch label is a coarse family
        // name, a context ceiling is a training-time fact, and whether a
        // tower ships is a question about the checkpoint's files. A
        // projection that filled them would be deriving a family name
        // from a shape, which is the inference that put
        // `Gemma4ForConditionalGeneration` in a table row it did not
        // belong in.
        advertised: Advertised::default(),
        // Unscaled, because nothing in this tree says otherwise:
        // `synthetic--kimi-k3.json` states no `rope_scaling` block at
        // all, and the ladder is used as written. A row that invented a
        // YaRN factor here would lengthen every position by a ratio no
        // published config backs.
        rope_scaling: None,
        towers: Default::default(),
    }
}

/// The recurrent slab geometry the KDA layers need allocated.
///
/// `PlannedFamily::gdn_shape()` for this generation, which was `None` —
/// the DEFAULT, inherited rather than answered, while
/// `PlannedFamily::recurrent()` was overridden to `true`. Two methods on
/// one vtable disagreeing about whether a stack carries recurrent state
/// is exactly the shape of defect the row table removes: a driver that
/// believed the first allocated nothing for two thirds of this stack.
///
/// The KDA state is `[value_heads, key_head_dim, value_head_dim]` per
/// slot, and this generation's key head dim IS its value head dim — the
/// delta rule updates a square state per head, which is what makes the
/// decay a per-channel vector rather than a scalar.
#[must_use]
fn kda_shape(f: &KimiK3Facts) -> RecurrentShape {
    let k = &f.kda;
    let head_dim = k.value_head_dim;
    RecurrentShape {
        linear_layers: (0..f.layers).filter(|l| !f.is_full_attn(*l)).collect(),
        conv_stride: (k.conv_kernel * k.width()) as usize,
        state_stride: (k.value_heads * head_dim * head_dim) as usize,
        // The store is bf16, which is the only recurrent allocator a
        // driver has (`RecurrentStateCache::allocate_bf16_recurrent`).
        state_elem: 2,
        k_h: k.value_heads as i32,
        v_h: k.value_heads as i32,
        k_d: head_dim as i32,
        v_d: head_dim as i32,
        // Three projections share one conv slab — q, k and v each get
        // their own short convolution, and the slab holds all three.
        conv_dim: (3 * k.width()) as i32,
        conv_k: k.conv_kernel as i32,
        // A KDA stack has no B/C groups; mamba's alone.
        n_groups: 0,
    }
}

/// Why this build has no Metal text for a kimi-k3 row.
///
/// A `const` so the test that asserts the refusal NAMES the missing
/// thing compares against the same string the caller is shown, rather
/// than against a paraphrase that can drift away from it — the shape
/// `csm::project::NO_TRACE` set for the same reason.
///
/// Its forward is `kimi_k3_cuda`: kimi-k2's MLA half plus the
/// KDA recurrence, which is a state-carrying layer kind
/// `llama_like_metal` has no operation for at all.
///
/// A `Refusal::Unsupported` and not a `Malformed`: the checkpoint is
/// fine, and a pie whose Metal half had this text would serve the same
/// row unchanged. What is missing is a TEXT in this build, which is a
/// fact about the build.
///
/// Stating it is the whole of what replaces `driver-metal`'s
/// `LLAMA_LIKE` — an eleven-entry table of architecture STRINGS,
/// reduced by a punctuation-stripping `canonical()`, consulted before
/// any text was traced and free to disagree with what the tracer would
/// actually do. It listed `gpt_oss`, which no publication of reaches a
/// Metal device here, and omitted `gemma3`, whose text it models. A row
/// that answers for itself cannot disagree with a list, because there
/// is no list.
pub const NO_METAL: &str = "kimi-k3 has no Metal text in this build: its forward is `kimi_k3_cuda` — \
     latent attention beside the KDA recurrence, which carries state across \
     tokens — and the one Metal text here (`llama_like_metal`) has no recurrent \
     layer kind and takes a different shape; the CUDA backend serves this row";

/// Trace this row's CUDA text for one fire class, or refuse it.
///
/// # The refusal is the point
///
/// `forward::kimi_k3_cuda` ASSERTS that the MLA output gate is clear and
/// panics when it is not, with a comment explaining why: `SigmoidGateMul`
/// is a semantic op whose operands must share a `Shape`, MLA's absorb is
/// rank-3, and filling that hole to make one text pass is how a
/// declaration starts describing the DSL instead of the model. A row,
/// unlike a fixture, states the MODEL — and this generation's MLA does
/// gate its output — so the row and the text disagree, and the disagreement
/// has to surface as a REFUSAL rather than as a panic inside a walk.
///
/// That is the same move `deployment` makes for the unbuilt latent
/// store, one aspect over: a build that cannot serve a row says so at
/// the door.
///
/// # Errors
///
/// [`Refusal::Unsupported`] when the row states a gated MLA output,
/// which this build has no text for.
pub fn trace(
    f: &KimiK3Facts,
    class: model_compiler::trace::FireClass,
) -> Result<model_compiler::trace::ForwardPlan, Refusal> {
    if f.attn.output_gate {
        return Err(Refusal::Unsupported(
            "kimi_k3: the MLA output gate is not stated by this build's text — \
             the semantic SigmoidGateMul wants equal Shapes and MLA's absorb is \
             rank-3",
        ));
    }
    Ok(super::forward::kimi_k3_cuda(f, class))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::Presence;

    fn k3() -> KimiK3Facts {
        KimiK3Facts::kimi_k3_synthetic()
    }

    fn extents(m: &Manifest, name: &str) -> Vec<u64> {
        m.tensors
            .iter()
            .find(|t| t.name == name)
            .unwrap_or_else(|| panic!("{name} is not named by this manifest"))
            .extents
            .clone()
    }

    fn presence(m: &Manifest, name: &str) -> Presence {
        m.tensors
            .iter()
            .find(|t| t.name == name)
            .unwrap_or_else(|| panic!("{name} is not named by this manifest"))
            .presence
    }

    fn named(m: &Manifest, name: &str) -> bool {
        m.tensors.iter().any(|t| t.name == name)
    }

    /// A manifest is a CHECK, so every extent in it has to be
    /// recomputable from the row's own numbers rather than transcribed
    /// beside them. These are the four that carry MLA's arithmetic.
    #[test]
    fn the_latent_extents_are_the_rows_own_arithmetic() {
        let f = k3();
        let m = manifest(&f, false);
        assert_eq!(extents(&m, "layer.{}.self_attn.q_a_proj"), vec![768, 2048]);
        assert_eq!(
            extents(&m, "layer.{}.self_attn.q_b_proj"),
            vec![16 * (128 + 64), 768],
            "every head's nope and rope halves, out of the query's own rank",
        );
        assert_eq!(
            extents(&m, "layer.{}.self_attn.kv_a_proj_with_mqa"),
            vec![256 + 64, 2048],
            "the latent plus the ONE shared rope half — not one per head",
        );
        assert_eq!(
            extents(&m, "layer.{}.self_attn.kv_b_proj"),
            vec![16 * (128 + 128), 256],
            "the compression ratio, stated: one latent read back into every \
             head's nope half and every head's value",
        );
    }

    /// `o_proj` reads the VALUE width, which is where MLA stops looking
    /// like GQA. A row that stated the query width here would load a
    /// tensor of the right rank and the wrong shape.
    #[test]
    fn the_output_projection_reads_the_value_width_and_not_the_query_width() {
        let f = k3();
        let m = manifest(&f, false);
        assert_eq!(extents(&m, "layer.{}.self_attn.o_proj"), vec![2048, 2048]);
        assert_ne!(
            f.attn.v_width(),
            f.attn.q_b_width(),
            "the fixture must keep them apart"
        );
    }

    /// The KDA half's tensors are named at the KDA width, which is NOT
    /// the MLA width. A hybrid that stated one width for both halves
    /// would match a checkpoint that ships neither.
    #[test]
    fn the_linear_half_is_named_at_its_own_width() {
        let f = k3();
        let m = manifest(&f, false);
        let w = u64::from(f.kda.width());
        assert_eq!(extents(&m, "layer.{}.self_attn.k_proj"), vec![w, 2048]);
        assert_eq!(extents(&m, "layer.{}.self_attn.v_proj"), vec![w, 2048]);
        assert_eq!(
            extents(&m, "layer.{}.self_attn.f_a_proj"),
            vec![128, 2048],
            "the decay's bottleneck is rank `value_head_dim`, which is what \
             makes the decay per-channel",
        );
        assert_eq!(extents(&m, "layer.{}.self_attn.f_b_proj"), vec![w, 128]);
        assert_eq!(
            extents(&m, "layer.{}.self_attn.b_proj"),
            vec![16, 2048],
            "one beta row per head, and the only place the unpadded head \
             count is written down",
        );
        assert_eq!(extents(&m, "layer.{}.self_attn.o_norm"), vec![128]);
    }

    /// `b_proj` and `q_b_proj` are different tensors with confusable
    /// names, and the contract says so out loud. If a manifest gave them
    /// the same extents, a checkpoint that shipped one for the other
    /// would load clean.
    #[test]
    fn the_beta_rows_and_the_query_bank_are_not_confusable() {
        let m = manifest(&k3(), false);
        assert_ne!(
            extents(&m, "layer.{}.self_attn.b_proj"),
            extents(&m, "layer.{}.self_attn.q_b_proj"),
        );
    }

    /// The padded gate bank is `Optional`, because its published extent
    /// is an allocator's rounding: `A_log` ships `[head_dim]` entries for
    /// `[value_heads]` real heads. A `Required` row at either width would
    /// fault a checkpoint that is fine.
    #[test]
    fn the_padded_gate_bank_is_not_required_at_a_width_nobody_measured() {
        let m = manifest(&k3(), false);
        assert_eq!(presence(&m, "layer.{}.self_attn.A_log"), Presence::Optional);
    }

    /// The routed BANK is an encoding and the ROUTER is the model. A
    /// manifest that named `experts.0.w1.weight_packed` would report a
    /// bf16 republish of the same checkpoint as a different model.
    #[test]
    fn the_manifest_names_the_router_and_never_the_bank() {
        let m = manifest(&k3(), false);
        assert_eq!(
            extents(&m, "layer.{}.block_sparse_moe.gate"),
            vec![64, 2048],
            "one row per expert over the residual stream",
        );
        for t in &m.tensors {
            assert!(
                !t.name.contains("experts.") && !t.name.contains("weight_packed"),
                "{} names a packing, which is what the catalog divides out",
                t.name,
            );
        }
    }

    /// The dense prefix and the mixture are the same statement read from
    /// its two ends, and a manifest states both because logical names
    /// collapse over the stack.
    #[test]
    fn no_prefix_means_no_dense_mlp_and_all_dense_means_no_router() {
        let f = k3();
        let m = manifest(&f, false);
        assert_eq!(presence(&m, "layer.{}.mlp.gate_proj"), Presence::Required);
        assert_eq!(
            presence(&m, "layer.{}.block_sparse_moe.gate"),
            Presence::Required
        );

        let mut no_prefix = f.clone();
        no_prefix.dense_layers = 0;
        let m = manifest(&no_prefix, false);
        assert_eq!(presence(&m, "layer.{}.mlp.gate_proj"), Presence::Absent);
        assert_eq!(
            presence(&m, "layer.{}.block_sparse_moe.gate"),
            Presence::Required
        );

        let mut all_dense = f.clone();
        all_dense.dense_layers = all_dense.layers;
        let m = manifest(&all_dense, false);
        assert_eq!(presence(&m, "layer.{}.mlp.gate_proj"), Presence::Required);
        assert_eq!(
            presence(&m, "layer.{}.block_sparse_moe.gate"),
            Presence::Absent
        );
    }

    /// A stack with no full-attention layer publishes no MLA tensors,
    /// and one with no linear layer publishes no KDA tensors. The union
    /// is over the layer kinds a row actually schedules — otherwise a
    /// manifest is a description of the generation rather than of the
    /// checkpoint.
    #[test]
    fn the_union_covers_only_the_layer_kinds_this_row_schedules() {
        let f = k3();
        let m = manifest(&f, false);
        assert_eq!(
            presence(&m, "layer.{}.self_attn.kv_b_proj"),
            Presence::Required
        );
        assert_eq!(
            presence(&m, "layer.{}.self_attn.f_a_proj"),
            Presence::Required
        );

        let mut kda_only = f.clone();
        kda_only.full_attn_interval = 0;
        let m = manifest(&kda_only, false);
        assert_eq!(
            presence(&m, "layer.{}.self_attn.kv_b_proj"),
            Presence::Absent
        );
        assert_eq!(
            presence(&m, "layer.{}.self_attn.f_a_proj"),
            Presence::Required
        );

        let mut mla_only = f.clone();
        mla_only.full_attn_interval = 1;
        let m = manifest(&mla_only, false);
        assert_eq!(
            presence(&m, "layer.{}.self_attn.kv_b_proj"),
            Presence::Required
        );
        assert_eq!(
            presence(&m, "layer.{}.self_attn.f_a_proj"),
            Presence::Absent
        );
    }

    /// The gate this generation ships and its text cannot state is still
    /// a TENSOR, and a row that states it names it. In a HYBRID the KDA
    /// half publishes `g_proj` too, so the discriminating statement is
    /// only available on an all-MLA stack — which is exactly what
    /// [`shared_width`] is for.
    #[test]
    fn a_gated_mla_names_the_gate_and_an_ungated_one_does_not() {
        let mut mla_only = k3();
        mla_only.full_attn_interval = 1;
        mla_only.attn.output_gate = false;
        let m = manifest(&mla_only, false);
        assert_eq!(
            presence(&m, "layer.{}.self_attn.g_proj"),
            Presence::Absent,
            "an all-MLA stack with no output gate ships no gate projection",
        );

        mla_only.attn.output_gate = true;
        let m = manifest(&mla_only, false);
        assert_eq!(
            extents(&m, "layer.{}.self_attn.g_proj"),
            vec![2048, 2048],
            "the gate projects to the VALUE width, which is what it multiplies",
        );

        // The hybrid states it as the KDA half's, which is the only
        // width both halves can be held to.
        let m = manifest(&k3(), false);
        assert_eq!(
            presence(&m, "layer.{}.self_attn.g_proj"),
            Presence::Required
        );
    }

    /// Where the two halves disagree about a shared name's width there
    /// is no honest row to write, and the manifest writes none. Stating
    /// either width would fault a checkpoint that is perfectly well
    /// formed — a manifest making a wrong claim, which is worse than one
    /// making no claim.
    #[test]
    fn a_name_both_halves_publish_at_different_widths_is_left_unstated() {
        let mut lopsided = k3();
        // Half the KDA heads, same MLA: 1024 against the value half's
        // 2048, under one collapsed `o_proj` key. The gate is stated too,
        // so both shared names disagree.
        lopsided.kda.value_heads = 8;
        lopsided.attn.output_gate = true;
        assert_ne!(
            u64::from(lopsided.kda.width()),
            u64::from(lopsided.attn.v_width())
        );
        let m = manifest(&lopsided, false);
        assert!(
            !named(&m, "layer.{}.self_attn.o_proj"),
            "two widths, no row"
        );
        assert!(
            !named(&m, "layer.{}.self_attn.g_proj"),
            "two widths, no row"
        );
        // The names only ONE half publishes are unaffected — they still
        // have exactly one width apiece.
        assert_eq!(
            extents(&m, "layer.{}.self_attn.k_proj"),
            vec![1024, 2048],
            "a KDA-only name keeps the KDA width",
        );
        assert_eq!(
            presence(&m, "layer.{}.self_attn.kv_b_proj"),
            Presence::Required
        );
    }

    /// The straight query projection is FORBIDDEN where a latent rank
    /// exists — but only where nothing else publishes the name. A hybrid
    /// cannot forbid `q_proj`, because its KDA layers ship one.
    #[test]
    fn a_ranked_query_forbids_the_straight_projection_only_where_nothing_else_ships_it() {
        let f = k3();
        assert!(f.attn.q_lora_rank > 0, "this generation ranks its query");
        let m = manifest(&f, false);
        assert_eq!(
            presence(&m, "layer.{}.self_attn.q_proj"),
            Presence::Required,
            "the KDA half publishes it, so forbidding it would fault every \
             real checkpoint",
        );
        assert_eq!(extents(&m, "layer.{}.self_attn.q_proj"), vec![2048, 2048]);

        let mut mla_only = f.clone();
        mla_only.full_attn_interval = 1;
        let m = manifest(&mla_only, false);
        assert_eq!(
            presence(&m, "layer.{}.self_attn.q_proj"),
            Presence::Absent,
            "a ranked query and a straight one cannot both be published",
        );

        let mut unranked = mla_only.clone();
        unranked.attn.q_lora_rank = 0;
        let m = manifest(&unranked, false);
        assert_eq!(
            extents(&m, "layer.{}.self_attn.q_proj"),
            vec![u64::from(unranked.attn.q_b_width()), 2048],
        );
        assert_eq!(
            presence(&m, "layer.{}.self_attn.q_a_proj"),
            Presence::Absent
        );
    }

    /// The blend's tensors exist only where a block opens, and they are
    /// this generation's alone — nothing else in the catalog blends a
    /// prefix across layers.
    #[test]
    fn the_attention_residual_tensors_follow_the_block_size() {
        let f = k3();
        let m = manifest(&f, false);
        assert_eq!(
            presence(&m, "layer.{}.self_attention_res_proj"),
            Presence::Required
        );
        assert_eq!(extents(&m, "layer.{}.self_attention_res_norm"), vec![2048]);

        let mut unblended = f.clone();
        unblended.attn_res_block = 0;
        let m = manifest(&unblended, false);
        assert_eq!(
            presence(&m, "layer.{}.self_attention_res_proj"),
            Presence::Absent
        );
    }

    /// A tied head publishes no `lm_head`, and the only way a manifest
    /// can tell the two apart is presence: every extent agrees.
    #[test]
    fn a_tied_head_publishes_no_output_table() {
        assert_eq!(
            presence(&manifest(&k3(), false), "lm_head"),
            Presence::Required
        );
        assert_eq!(
            presence(&manifest(&k3(), true), "lm_head"),
            Presence::Absent
        );
        assert_eq!(
            extents(&manifest(&k3(), false), "lm_head"),
            vec![163_840, 2048]
        );
    }

    /// A manifest that would match a SIBLING generation is a bug, and
    /// the siblings here are the two other MLA rows in the catalog.
    /// Their numbers are held as data rather than reached for by path:
    /// a generation that named `crate::kimi_k2::` would be a dependency
    /// between two rows that are supposed to be independent.
    #[test]
    fn the_manifest_does_not_match_a_sibling_mla_row() {
        // kimi-k2's own measurement: 61 layers, 7168 hidden, 64 heads,
        // q_lora 1536, kv_lora 512, nope 128, rope 64, v 128.
        let sibling = KimiK3Facts {
            layers: 61,
            vocab: 163_840,
            hidden: 7168,
            dense_intermediate: 18_432,
            dense_layers: 1,
            full_attn_interval: 1,
            attn_res_block: 0,
            attn: super::super::spec::KimiK3MlaFacts {
                hidden: 7168,
                heads: 64,
                q_lora_rank: 1536,
                kv_lora_rank: 512,
                qk_nope_head_dim: 128,
                qk_rope_head_dim: 64,
                v_head_dim: 128,
                output_gate: false,
            },
            kda: super::super::spec::KimiK3KdaFacts {
                value_heads: 16,
                value_head_dim: 128,
                conv_kernel: 4,
                gate_lower_bound_milli: 0,
            },
            moe: super::super::spec::KimiK3MoeFacts {
                num_experts: 384,
                top_k: 8,
                norm_topk_prob: false,
                routed_scaling: 2.0,
                moe_intermediate: 2048,
                shared_intermediate: 2048,
            },
        };
        let (mine, theirs) = (manifest(&k3(), false), manifest(&sibling, false));
        assert_ne!(
            extents(&mine, "layer.{}.self_attn.kv_b_proj"),
            extents(&theirs, "layer.{}.self_attn.kv_b_proj"),
            "two MLA rows that agree on the read-back width are one row",
        );
        assert_ne!(mine.layers, theirs.layers);
        assert_eq!(
            presence(&mine, "layer.{}.self_attn.f_a_proj"),
            Presence::Required,
            "the KDA half is this generation's own",
        );
        assert_eq!(
            presence(&theirs, "layer.{}.self_attn.f_a_proj"),
            Presence::Absent,
            "a pure-MLA sibling FORBIDS the decay gate, so no checkpoint \
             satisfies both rows",
        );
    }

    /// The stack's depth, and the per-layer table's length with it. A
    /// deployment whose attention table is shorter than its layer count
    /// is a fire indexing past the end of it.
    #[test]
    fn the_deployment_states_one_attention_row_per_layer() {
        let d = plan(&k3(), 10_000.0, 1e-5);
        assert_eq!(d.layers, 8);
        assert_eq!(d.attention.len(), 8);
        assert!(
            (d.norm_eps - 1e-5).abs() < f32::EPSILON,
            "epsilon is stated, not defaulted"
        );
    }

    /// The geometry, field by field. `head_dim` and `head_dim_kernel`
    /// are the page row and nothing pads it; `kv_heads` is ONE because
    /// every query head reads the same latent plane.
    #[test]
    fn the_geometry_states_the_latent_plane_and_both_mlp_widths() {
        let d = plan(&k3(), 10_000.0, 1e-5);
        let g = &d.shape;
        assert_eq!(g.hidden, 2048);
        assert_eq!(g.q_heads, 16);
        assert_eq!(g.kv_heads, 1);
        assert_eq!(g.head_dim, 320);
        assert_eq!(g.head_dim_kernel, 320);
        assert_eq!(g.intermediate, 5632, "the dense prefix's width");
        assert_eq!(g.moe_intermediate, 1024, "ONE expert's width");
        assert_eq!(g.vocab, 163_840);
        assert_eq!(
            g.widest_mlp(),
            5632,
            "the workspace is one buffer both layer kinds share, so it is \
             sized from the wider",
        );
        assert_eq!(
            g.gqa_group(),
            16,
            "what an MLA decode would have to instantiate"
        );
    }

    /// The per-layer table: no window anywhere, every layer owns its
    /// pages, and the scale is over the DOT width rather than the stored
    /// row.
    #[test]
    fn every_layer_attends_the_whole_context_at_the_dot_width() {
        let d = plan(&k3(), 10_000.0, 1e-5);
        let expected = 1.0 / (192.0_f32).sqrt();
        for (l, a) in d.attention.iter().enumerate() {
            assert_eq!(a.window, -1, "layer {l} must not window a recurrence");
            assert_eq!(a.kv_source, l as u32);
            assert_eq!(a.head_dim, 320);
            assert!(
                (a.sm_scale - expected).abs() < 1e-6,
                "layer {l} scales by the stored row"
            );
        }
        assert!(
            (d.attention[0].sm_scale - 1.0 / (320.0_f32).sqrt()).abs() > 1e-6,
            "scaling by the page row is the defect this states away",
        );
    }

    /// The MLA layers state NO rope, because the text contains none.
    /// A driver that built a rope table for them would rotate a query
    /// the model never rotates.
    #[test]
    fn the_full_attention_layers_state_no_rotation() {
        let f = k3();
        let d = plan(&f, 10_000.0, 1e-5);
        for (l, a) in d.attention.iter().enumerate() {
            assert_eq!(a.rotary_dim, 0, "layer {l} rotates nothing");
            if f.is_full_attn(l as u32) {
                assert!(
                    a.rope_theta.abs() < f32::EPSILON,
                    "layer {l} is MLA and carries no rope"
                );
            } else {
                assert!(
                    (a.rope_theta - 10_000.0).abs() < f32::EPSILON,
                    "layer {l} is KDA and keeps whatever base the row states",
                );
            }
        }
    }

    /// The KV style, and its two numbers. They are STATED by the row —
    /// the derivation this replaces read them back off a resident
    /// `config.json` to fill the very facts row that already held them.
    #[test]
    fn the_kv_style_is_mla_with_the_rows_own_ranks() {
        let d = plan(&k3(), 10_000.0, 1e-5);
        assert_eq!(
            d.kv,
            KvStyle::Mla {
                kv_lora_rank: 256,
                qk_rope_head_dim: 64
            }
        );
    }

    /// The recurrent slabs the KDA layers need, which the vtable this
    /// replaces answered `None` for while claiming `recurrent() == true`.
    #[test]
    fn the_linear_layers_get_slabs_and_the_full_attention_layers_do_not() {
        let f = k3();
        let d = plan(&f, 10_000.0, 1e-5);
        let r = d
            .recurrent
            .as_ref()
            .expect("two thirds of this stack is a recurrence");
        assert_eq!(r.linear_layers, vec![0, 1, 2, 4, 5, 6]);
        assert!(
            r.linear_layers.iter().all(|l| !f.is_full_attn(*l)),
            "a slab for a layer that pages is a slab nothing writes",
        );
        assert_eq!(r.k_h, 16);
        assert_eq!(r.v_h, 16);
        assert_eq!(r.k_d, 128);
        assert_eq!(r.v_d, 128);
        assert_eq!(r.state_stride, 16 * 128 * 128, "a square state per head");
        assert_eq!(
            r.conv_dim,
            3 * 2048,
            "one conv per projection, three projections"
        );
        assert_eq!(r.conv_k, 4);
        assert_eq!(r.conv_stride, 4 * 2048);
        assert_eq!(
            r.state_elem, 2,
            "bf16 is the only recurrent allocator a driver has"
        );
    }

    /// A stack scheduled as all-MLA has no linear layers at all, and the
    /// slab list says so rather than covering the stack out of habit.
    #[test]
    fn an_all_full_attention_schedule_asks_for_no_slabs() {
        let mut f = k3();
        f.full_attn_interval = 1;
        let d = plan(&f, 10_000.0, 1e-5);
        assert!(d.recurrent.expect("still stated").linear_layers.is_empty());
    }

    /// What the row advertises rides through untouched. It is carried
    /// rather than derived because the derivation it replaces read
    /// `model_type` and `max_position_embeddings` off a resident
    /// `HfConfig`.
    #[test]
    fn the_projection_states_no_family_label_because_a_label_is_not_geometry() {
        let d = plan(&k3(), 10_000.0, 1e-5);
        assert_eq!(
            d.advertised,
            Advertised::default(),
            "a projection that fills this in has derived a family name from a shape",
        );
        assert!(
            d.advertised.arch.is_empty(),
            "an empty label is a row that has not spoken yet"
        );
        assert_eq!(d.advertised.max_model_len, 0);
        assert!(!d.advertised.media_encode);
    }

    /// The ladder is used AS WRITTEN and no tower ships, and both are
    /// statements rather than omissions. `synthetic--kimi-k3.json`, the
    /// only Kimi-K3 config committed here, states no `rope_scaling`
    /// block, so `None` is what was READ. A projection that filled in a
    /// factor would stretch every position by a ratio nothing here
    /// measured.
    #[test]
    fn the_rope_ladder_is_unscaled_and_no_tower_ships() {
        let d = plan(&k3(), 10_000.0, 1e-5);
        assert!(
            d.rope_scaling.is_none(),
            "no committed config states a rescaling to read"
        );
        assert_eq!(
            d.towers,
            crate::deployment::Towers::default(),
            "a text-in text-out stack that claims a tower has the worker bind an encoder \
             it does not ship",
        );
    }

    /// The rest of the deployment, which is this generation's answer to
    /// questions other generations answer differently.
    #[test]
    fn the_stack_is_planned_pre_norm_and_caps_nothing() {
        let d = plan(&k3(), 10_000.0, 1e-5);
        assert_eq!(d.prefill, PrefillStyle::Planned);
        assert_eq!(d.attn_output, AttnOutput::DriverPinned);
        assert_eq!(d.norm, NormPlacement::Pre);
        assert!(d.logit_softcap.abs() < f32::EPSILON);
        assert_eq!(d.ple_dim, 0);
        assert!(d.scales.is_empty());
        assert!(d.towers.audio.is_none() && d.towers.vision.is_none());
    }

    /// The refusal fires, and it fires at the DOOR — before a load, not
    /// at the first fire inside a walk.
    #[test]
    fn a_build_with_no_mla_store_refuses_the_row() {
        let err = deployment(&k3(), 10_000.0, 1e-5)
            .expect_err("no MLA store is built in this tree, so the row cannot be served");
        assert!(matches!(err, Refusal::Unsupported(_)));
    }

    /// The two halves of that statement are separable: the PLAN exists
    /// whether or not this build can serve it.
    #[test]
    fn the_plan_exists_even_where_the_build_refuses_it() {
        let planned = plan(&k3(), 10_000.0, 1e-5);
        assert!(!planned.kv.has_a_store_in_this_build());
        assert!(KvStyle::Paged.has_a_store_in_this_build());
        assert!(!KvStyle::CompressedPlane { ratios: Vec::new() }.has_a_store_in_this_build());
    }

    /// The text traces for both fire classes, and names this
    /// generation's own family string — the goldens are keyed on it, so
    /// a rename here is a rename of every recorded plan.
    #[test]
    fn the_text_traces_for_both_fire_classes() {
        use model_compiler::trace::FireClass;
        let f = k3();
        for (class, suffix) in [
            (FireClass::Decode, "decode"),
            (FireClass::Prefill, "prefill"),
        ] {
            let plan = trace(&f, class).expect("the fixture states the shape the text declares");
            assert_eq!(plan.family, format!("kimi_k3.cuda.{suffix}"));
            assert!(!plan.ops.is_empty(), "a traced plan states ops");
        }
    }

    /// A row whose MLA gates its output is refused rather than traced,
    /// because the text asserts on it. Turning that panic into a
    /// refusal is the whole reason `trace` returns a `Result`.
    #[test]
    fn a_gated_mla_output_is_refused_and_not_traced() {
        use model_compiler::trace::FireClass;
        let mut gated = k3();
        gated.attn.output_gate = true;
        for class in [FireClass::Decode, FireClass::Prefill] {
            assert!(
                matches!(trace(&gated, class), Err(Refusal::Unsupported(_))),
                "a text that would panic must refuse at the door instead",
            );
        }
    }
}

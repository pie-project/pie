//! GPT-OSS's projections: what its numbers imply about a checkpoint, a
//! deployment and a trace.
//!
//! `gpt_oss_facts_from_hf` read fourteen numbers out of a parsed
//! `config.json` and the vtable answered three more per fire — the
//! per-layer window came back out of the CUDA facts through
//! `window_by_layer`, which meant the deployment's window table was a
//! second reading of a list the row already implied. The alternation is
//! a RULE here (`is_sliding`), stated once and projected.

// Only the texts name a backend, and only they are gated.
use crate::catalog::Deployed;
use crate::deployment::{
    Advertised, AttnOutput, Deployment, Geometry, KvStyle, LayerAttention, NormPlacement,
    PrefillStyle,
};
use crate::manifest::{Manifest, TensorSpec};

use super::spec::GptOssFacts;

/// This row's tensors.
///
/// # Why no expert WEIGHT is named
///
/// gpt-oss ships its experts two ways. The MXFP4 release publishes a
/// `gate_up_proj_blocks` / `gate_up_proj_scales` / `gate_up_proj_bias`
/// triplet; the dequantized bf16 release publishes `gate_up_proj` and
/// the same bias. Those are two SPELLINGS of one bank, so a spec that
/// required either name would be matching on the encoding — and the
/// catalog's rule is that an encoding is not an identity.
///
/// What both publish, at extents no packing changes, is the BIAS: one
/// row per expert of the projection's output width. That pins the
/// mixture's geometry (`[experts, 2 * intermediate]` and
/// `[experts, hidden]`) without naming a packing, which is what makes
/// the MXFP4 build and the bf16 build one row rather than two.
#[must_use]
pub fn manifest(f: &GptOssFacts) -> Manifest {
    let (hidden, vocab) = (u64::from(f.hidden), u64::from(f.vocab));
    let q = u64::from(f.q_heads * f.head_dim);
    let kv = u64::from(f.kv_heads * f.head_dim);
    let experts = u64::from(f.experts);

    Manifest::new(f.layers)
        .with(TensorSpec::required("embed_tokens", [vocab, hidden]))
        .with(TensorSpec::required("norm", [hidden]))
        .tie(f.tied_embeddings, "lm_head", [vocab, hidden])
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
        // The four biases the row's `attention_bias` states. They are an
        // expectation rather than a probe: the old derivation asked the
        // LOAD whether a bias was bound and branched on the answer.
        .with(TensorSpec::required("layer.{}.self_attn.q_proj.bias", [q]))
        .with(TensorSpec::required("layer.{}.self_attn.k_proj.bias", [kv]))
        .with(TensorSpec::required("layer.{}.self_attn.v_proj.bias", [kv]))
        .with(TensorSpec::required(
            "layer.{}.self_attn.o_proj.bias",
            [hidden],
        ))
        // ATTENTION SINKS: one learned logit per query head, appended to
        // the softmax denominator. It is the tensor that makes gpt-oss
        // gpt-oss — no other row in this build ships one — and the
        // reason the attention statement has to produce an LSE beside
        // its output.
        .with(TensorSpec::required(
            "layer.{}.self_attn.sinks",
            [u64::from(f.q_heads)],
        ))
        .with(TensorSpec::required("layer.{}.input_layernorm", [hidden]))
        .with(TensorSpec::required(
            "layer.{}.post_attention_layernorm",
            [hidden],
        ))
        // The router. `[experts, hidden]` IS the claim that this is a
        // mixture of `experts` experts, and it is never quantized
        // (`modules_to_not_convert` names it), so its extents are safe
        // to state exactly.
        .with(TensorSpec::required(
            "layer.{}.mlp.router",
            [experts, hidden],
        ))
        .with(TensorSpec::required("layer.{}.mlp.router.bias", [experts]))
        // The expert banks, named by the half of each that survives
        // every packing. `2 *` because gate and up ship fused.
        //
        // ...in OpenAI's publication. MLX's divides the same bank three
        // ways and spells the bias with a dot, so neither name reaches
        // it, and an `mlx-community/gpt-oss-20b-MXFP4-Q4` matched NO ROW
        // AT ALL rather than this one. Measured against that checkpoint:
        // those two names were the only faults in the whole manifest —
        // the packed `embed_tokens`, the packed `lm_head` and even a
        // ROUTER MLX quantizes at `[32, 720]` beside a `.scales` all
        // agree already, because `extents_agree` divides the packing
        // out. So the disagreement really was two names wide.
        .with(
            TensorSpec::required(
                "layer.{}.mlp.experts.gate_up_proj_bias",
                [experts, u64::from(2 * f.intermediate)],
            )
            // Gate and up, unfused: two halves of the fused row's last
            // axis, at `[experts, intermediate]` each. Both are required
            // together — one alone would be a checkpoint that publishes
            // a gate and no up, which is not a mixture.
            .or_published_as([
                (
                    "layer.{}.mlp.experts.gate_proj.bias",
                    [experts, u64::from(f.intermediate)],
                ),
                (
                    "layer.{}.mlp.experts.up_proj.bias",
                    [experts, u64::from(f.intermediate)],
                ),
            ]),
        )
        .with(
            TensorSpec::required("layer.{}.mlp.experts.down_proj_bias", [experts, hidden])
                // The same tensor at the same extents under a dot. It is
                // stated here rather than folded into `Observed::logical`
                // because a general `_bias` -> `.bias` rule is FALSE:
                // nemotron-h publishes a `mixer.dt_bias`, whose `dt` is
                // no tensor, and the rule would rename it to something
                // no checkpoint holds.
                .or_published_as([("layer.{}.mlp.experts.down_proj.bias", [experts, hidden])]),
        )
}

/// gpt-oss's gate alpha, the 1.702 that makes `x * sigmoid(alpha * x)`
/// the GELU approximation its MLP is written against.
///
/// A CONSTANT and not a fact, because no published gpt-oss config
/// states it: `swiglu_limit` is a row's number and this is the
/// activation's own. It had no home at all before — the driver carried
/// a `swiglu_alpha` field nothing ever wrote, so every gpt-oss reaching
/// a Metal text would have gated on alpha zero.
pub(crate) const GATE_ALPHA: f32 = 1.702;

/// This row's deployment.
///
/// The window table is the row's alternation rule expanded, not a list
/// read back out of the CUDA facts: `is_sliding` is the statement and
/// this table is its only expansion. The text states no window at all —
/// the driver reads [`LayerAttention::window`] per layer — so there is
/// no second copy left to disagree with this one.
#[must_use]
pub fn deployment(
    f: &GptOssFacts,
    rope_theta: f32,
    norm_eps: f32,
    sliding_window: i32,
) -> Deployment {
    let head_dim = f.head_dim;
    let attention = (0..f.layers)
        .map(|l| LayerAttention {
            // One shape for every layer, which is what this row was
            // already saying by having no per-layer count.
            kv_heads: f.kv_heads,
            head_dim,
            window: if f.is_sliding(l) { sliding_window } else { -1 },
            kv_source: l,
            sm_scale: 1.0 / (head_dim as f32).sqrt(),
            rope_theta,
            // Full rotation at the head dim. gpt-oss's YaRN scaling is a
            // property of the rope TABLE, not of its width.
            rotary_dim: 0,
            q_gate: false,
        })
        .collect();
    Deployment {
        layers: f.layers,
        norm_eps,
        shape: Geometry {
            hidden: f.hidden,
            q_heads: f.q_heads,
            kv_heads: f.kv_heads,
            head_dim,
            // 64 is instantiated, so nothing pads.
            head_dim_kernel: head_dim,
            // ZERO dense width: every layer in gpt-oss is the router's,
            // so there is no dense block for a workspace to be sized
            // by. The per-expert width goes in the field that names it,
            // and `widest_mlp()` — which is what sizes the one buffer —
            // reads the max of the two.
            intermediate: 0,
            moe_intermediate: f.intermediate,
            experts_per_token: f.top_k,
            shared_intermediate: 0,
            vocab: f.vocab,
        },
        attention,
        // Ordinary paged k/v. The sinks change the SOFTMAX, not the
        // store — which is worth stating, because the family's one
        // historical refusal (`UnknownWeight("layer.0.router")`) was a
        // missing forward path and not a missing pool.
        kv: KvStyle::Paged,
        recurrent: None,
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
        // gpt-oss softmaxes the k it selected, so the weights sum to
        // one. No `norm_topk_prob` key is in its config -- the router is
        // written that way.
        norm_topk_prob: true,
        // No router of this family states a scaling factor.
        routed_scaling: 1.0,
        mlp_gate: crate::deployment::MlpGate::SiluClamped {
            limit: f.swiglu_limit,
            alpha: GATE_ALPHA,
        },
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
/// The window list is the deployment's, expanded from the same rule, so
/// the tracer and the launcher cannot hold different schedules. The rest
/// is the engine's default MXFP4 policy: pointer arrays for the fused
/// decode GEMV, the default route ceiling of `32 * experts`, no
/// streaming.
#[must_use]
pub fn cuda_facts(f: &GptOssFacts, load: Deployed<'_>) -> super::forward::facts::GptOssCudaFacts {
    let _ = load;
    super::forward::facts::GptOssCudaFacts {
        mxfp4_decode_gemv: true,
        mxfp4_decode_max_routes: 32 * f.experts,
        streamed_experts: false,
    }
}

/// This row's ARCHITECTURE, in the shared llama-like vocabulary.
///
/// # What this replaces, and what its absence was mistaken for
///
/// There was a `NO_METAL` constant here, and for a long time it named the
/// wrong missing thing. It read: `llama_like_metal` "takes a
/// `LlamaLikeFacts` and this row is a `GptOssFacts`; a shape cast between
/// them would be the tensor-sniffing this catalog replaced". A projection is
/// not a cast — it states each field from the row's own facts, which is the
/// opposite of the `facts_from_with` that read TENSORS.
///
/// Its second reading narrowed the gap correctly, to "no `metal_shape` /
/// `metal_facts` projection is written for this row", and listed what was
/// already present: the text states `attn_sinks`, states the clamped GLU as
/// `Activation::SwiGlu` at gpt-oss's own 7.0 and 1.702, and `routed_qmv`
/// picks `mxfp4_qmv_routed_bias` off `moe_repr`. All true.
///
/// The list was still not complete, and what it left out was not small.
/// gpt-oss biases its attention landing and its ROUTER, and the shared Metal
/// text stated neither — because no Metal kernel added a bias at all until
/// `norm/add_bias.metal`, and no binder resolved that kernel's width until
/// `dispatch::derived`. A projection written before those would have traced
/// a gpt-oss that routes every token to the wrong experts.
///
/// A projection and not a cast, which is the distinction the Metal refusal
/// used to blur: every field below is stated from `GptOssFacts`, so the two
/// structs never have to have the same layout and nothing here reads a
/// tensor. [`crate::gemma_4::project::metal_shape`] is the same shape for
/// the same reason.
#[must_use]
pub fn metal_shape(f: &GptOssFacts) -> crate::shared::llama_like::spec::LlamaLikeFacts {
    use crate::shared::llama_like::spec::LlamaLikeFacts;
    use model_ir::facts::{NormPlacement, QkNorm};
    use model_ir::trace::{NormVariant, RopeKind};

    LlamaLikeFacts {
        hidden: f.hidden,
        layers: f.layers,
        q_heads: f.q_heads,
        kv_heads: f.kv_heads,
        head_dim: f.head_dim,
        n_experts: f.experts,
        experts_per_token: f.top_k,
        // Every expert is one MLP of this width. `GptOssFacts::intermediate`
        // warns that gpt-oss's happens to equal `hidden` and that no text may
        // lean on the coincidence -- stating it in both places is how this
        // projection declines to.
        moe_intermediate: f.intermediate,
        // No shared expert: gpt-oss routes every token to `top_k` of the 32
        // and adds nothing dense alongside.
        shared_intermediate: 0,
        // And no DENSE MLP either. `intermediate` on the llama-like side is
        // the dense branch's width, which this stack does not have -- the
        // expert width above is where gpt-oss's `intermediate_size` goes.
        intermediate: 0,
        vocab: f.vocab,
        // YaRN, which the shared side spells as a KIND. Stated here rather
        // than read off the row: `the_branching_boolean_is_the_one_the_
        // checkpoint_states` in `spec.rs` retired `rope_yarn_original`
        // because every gpt-oss row answered it the same way, and a field
        // two rows fill identically is a place for one of them to be wrong.
        // Wrong here is not a crash: a silently unscaled rotation.
        rope: RopeKind::Yarn,
        norm_variant: NormVariant::Plain,
        norm_placement: NormPlacement::Pre,
        qk_norm: QkNorm::Off,
        // The loader publishes `self_attn.{q,k,v}_proj` separately for every
        // MLX checkpoint, gpt-oss included; see `LlamaLikeMetalFacts::
        // qkv_fused` for why this is the driver's answer and not the row's.
        fused_qkv: false,
        tied_embeddings: f.tied_embeddings,
        // All three biases, stated rather than asked. gpt-oss biases q/k/v/o,
        // the router and the experts -- one publication decision every row
        // answers the same way, which is why `attention_bias` is no longer a
        // field and this projection states it three times instead.
        //
        // The expert banks' own biases are not here: they are not optional
        // on an MXFP4 bank at all. `mxfp4_qmv_routed_bias` is the only routed
        // MXFP4 symbol `qmv.metal` exports, so that leg reads a bias or it
        // does not run.
        qkv_bias: true,
        o_bias: true,
        router_bias: true,
    }
}

/// This row's DEPLOYMENT facts for the Metal text.
///
/// The binding half comes from `bind` and the architecture half from the
/// row, which is the split [`crate::catalog::MetalBinding`] exists to make.
#[must_use]
pub fn metal_facts(
    f: &GptOssFacts,
    bind: &crate::catalog::MetalBinding,
) -> crate::shared::llama_like::forward::facts::LlamaLikeMetalFacts {
    use crate::shared::llama_like::forward::facts::{Activation, LlamaLikeMetalFacts};
    use model_dsl::{ScaleLayout, WeightRepr};

    LlamaLikeMetalFacts {
        // Every gpt-oss layer carries a per-head sink; the row stopped
        // holding a flag for a question with one answer.
        attn_sinks: true,
        // The clamp is a different kernel, so it is the SYMBOL that changes
        // and not a scalar. Alpha is the activation's own constant; the limit
        // is the row's `swiglu_limit`.
        activation: Activation::SwiGlu {
            limit: f.swiglu_limit,
            alpha: 1.702,
        },
        // The base the ladder would use if it were geometric, and the fact
        // that says it is NOT. A rescaled ladder -- llama-3's piecewise one,
        // YaRN's -- is not expressible as a base, so the text takes the table
        // form and the driver derives the table at load.
        //
        // `LlamaLikeMetalFacts::gpt_oss_20b()` said `false` here and called
        // 150000 "a plain geometric ladder", which was true of nothing --
        // the row has carried the YaRN block since it was written. That
        // fixture is synthetic and this is the projection a real load reads,
        // so this one was corrected first and the fixture followed; both say
        // `true` now, and four backends' text tests fire the table form
        // because of it. gpt-oss rescales by YaRN (factor 32 over an
        // original 4096 context) and getting it wrong rotates by unscaled
        // frequencies at every position but zero.
        rope_theta: 150_000.0,
        rope_freq_table: true,
        rms_eps: 1e-5,
        // Alternating: every other layer attends 128 tokens back and the rest
        // attend everything. Built from `GptOssFacts::is_sliding` rather than
        // from a second copy of the parity, so the two backends cannot come
        // to different answers about which layers slide -- `cuda_facts`
        // builds its `window_left` from the same accessor, right above.
        window_left: (0..f.layers)
            .map(|l| if f.is_sliding(l) { 128 } else { -1 })
            .collect(),
        // THE PROJECTIONS' encoding, which is not the banks'. MLX quantises
        // gpt-oss's dense tensors affine/g64/b4 and leaves the expert banks
        // to the top-level mxfp4/32 default; `MetalBinding` carries what the
        // load observed rather than what this row assumes.
        proj_repr: WeightRepr::Scaled {
            layout: ScaleLayout::PerGroup,
            group: bind.quant_group,
            axis: 0,
            zero_point: true,
        },
        affine_bits: bind.quant_bits,
        // And the BANKS'. `binding::observed` probes one expert tensor at
        // load and answers this; `EXPERT_BANK` is the tensor it asks, and the
        // constant exists so the claim "one probe, and it decides an
        // encoding" is checkable by reading a file.
        //
        // `None` is not "unknown", it is "THE SAME AS THE DENSE PROJECTIONS"
        // -- the shared projection's word, and worth taking rather than
        // paraphrasing. Restating it as an explicit `Scaled` built from
        // `bind.quant_group` looks equivalent and is not: at the g128/b8
        // encoding `a_row_is_served_the_same_way_at_every_encoding` walks,
        // it named `affine_qmv_routed_bfloat16_gs_128_b_8`, which no
        // `kernel!` signature declares, and the row stopped being servable at
        // an encoding it must be servable at.
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
        // Four by the format's definition and read only when the repr above
        // is `Some`, exactly as the shared projection states it.
        moe_bits: 4,
        fuse_residual_gemv: bind.fuse_residual_gemv,
        paged_multi_batch: bind.paged_multi_batch,
        qmm_multi_batch: bind.qmm_multi_batch,
        add_bias: bind.add_bias,
        // gpt-oss's config states no `norm_topk_prob` and the reference
        // softmaxes the top-k scores, which is what normalising them means.
        norm_topk_prob: true,
        // YaRN's OTHER number, and the one a rope table cannot carry.
        attn_scale: yarn_softmax_scale(f.head_dim),
        ..LlamaLikeMetalFacts::synthetic()
    }
}

/// The softmax temperature a YaRN deployment attends at.
///
/// `RopeScaling::Yarn::attention_factor` compensates for the lengthened
/// ladder, and HF applies it to `cos` and `sin` — which is to say to the
/// ROTATED `q` and the rotated `k`, both. What reaches the softmax is
/// therefore the dot product times the factor SQUARED, and then times
/// `1/sqrt(head_dim)` as usual.
///
/// Two consequences worth stating, because each is a place this could have
/// gone wrong quietly:
///
///   * It cannot ride the frequency table. `Source::Named(<keys::RopeFrequencies as keys::Fact>::KEY)` carries
///     inverse frequencies and the shader raises its own `cos`/`sin` from
///     them, so an amplitude handed to `model::rope` would simply be dropped.
///     CUDA does not have this problem — `bind::abi` passes all four YaRN
///     numbers to the kernel — so the two backends reach the same product by
///     different routes and this is Metal's.
///   * The attention SINK must not be scaled by it, and is not: the sink is
///     an additive logit the kernel appends after the temperature multiplies
///     `q·k`, which is exactly HF's order. Folding the factor into the
///     temperature keeps that distinction; folding it into `q` alone would
///     not have.
///
/// For gpt-oss: `1.3466² / 8` is `0.2266`, against the `0.125` a derived
/// `1/sqrt(64)` gives. A 1.81x error in the softmax temperature does not
/// fault — it sharpens every distribution in the stack and reads as a model
/// that has become oddly confident.
fn yarn_softmax_scale(head_dim: u32) -> f32 {
    // Destructured in a CONST, so "gpt-oss rescales by YaRN" is checked by
    // the compiler rather than by a branch. The runtime form of this had a
    // fallback arm reading "unreachable while the const is what its name
    // says" -- true, and a fallback is how an unreachable arm becomes a
    // reachable one without anybody deciding to. Its value was the shared
    // "derive `1/sqrt(head_dim)`", so a generation that stopped rescaling
    // would have attended at 0.125 against this row's 0.2266 and nothing
    // would have faulted.
    const ATTENTION_FACTOR: f32 = match super::ROPE_SCALING {
        crate::deployment::RopeScaling::Yarn {
            attention_factor, ..
        } => attention_factor,
        _ => panic!("gpt-oss rescales by YaRN; this scale is that factor squared"),
    };
    ATTENTION_FACTOR * ATTENTION_FACTOR / (head_dim as f32).sqrt()
}

/// Trace this row's CUDA text for one fire class.
#[must_use]
pub fn trace(
    f: &GptOssFacts,
    class: model_ir::trace::FireClass,
    load: Deployed<'_>,
    norm_eps: f32,
    rope_theta: f32,
    sliding_window: i32,
) -> model_ir::trace::ForwardPlan {
    // THE SHIPPED POINT. gpt-oss catalogues one SKU today — MXFP4-Marlin
    // experts around a bf16 stack; the table in `forward::CATALOG` is
    // where a second one appears, and the coverage test is what keeps
    // every row loadable.
    use model_dsl::axes::{Bf16Ax, Mxfp4Ax, NativeKv};
    super::forward::gpt_oss_cuda::<Bf16Ax, Mxfp4Ax, Bf16Ax, NativeKv>(
        f,
        &cuda_facts(f, load),
        class,
        norm_eps,
        rope_theta,
        sliding_window,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{Observed, Presence};

    fn f20b() -> GptOssFacts {
        GptOssFacts::gpt_oss_20b()
    }

    /// What a checkpoint publishes, in the manifest's own vocabulary,
    /// for every row the manifest does not forbid — the same
    /// construction `catalog::every_row_satisfies_its_own_manifest`
    /// makes.
    fn implied(m: &Manifest) -> Vec<(String, Vec<u64>)> {
        m.tensors
            .iter()
            .filter(|t| t.presence != Presence::Absent)
            .map(|t| (t.name.replace("{}", "0"), t.extents.clone()))
            .collect()
    }

    /// YaRN's `attention_factor` reaches the softmax, squared.
    ///
    /// Two independent evaluations of one number: `yarn_softmax_scale` reads
    /// `ROPE_SCALING` and squares it, and this reads the config's formula
    /// (`0.1 * ln(32) + 1`) and squares that. They agree because the const is
    /// the formula, which is the claim
    /// `the_yarn_numbers_are_the_configs_and_the_omitted_one_is_the_formula`
    /// checks one seam earlier.
    ///
    /// A ZERO here is the failure mode worth naming: the shared `sdpa`
    /// statement reads zero as "derive `1/sqrt(head_dim)`", so a text that
    /// forgot this fact would attend at `0.125` instead of `0.2266` and
    /// nothing would fault.
    #[test]
    fn the_yarn_attention_factor_reaches_the_metal_softmax_squared() {
        let f = f20b();
        let bind = crate::catalog::MetalBinding {
            qmm_partial_rows: false,
            qmm_fp16_precast: true,
            qmm_tile: None,
            quant_group: 64,
            quant_bits: 4,
            router_quant_group: 0,
            router_quant_bits: 0,
            moe_mxfp4: true,
            fuse_residual_gemv: true,
            paged_multi_batch: true,
            qmm_multi_batch: true,
            add_bias: true,
            fused_qk_rope: false,
        };
        let m = metal_facts(&f, &bind);
        let factor = 0.1f32.mul_add(32.0f32.ln(), 1.0);
        let want = factor * factor / 8.0;
        assert!(
            (m.attn_scale - want).abs() < 1e-6,
            "stated {}, the formula squared over sqrt(64) is {want}",
            m.attn_scale
        );
        // And it is NOT the derived default, which is the whole point.
        let derived = 1.0f32 / (f.head_dim as f32).sqrt();
        assert!(
            (m.attn_scale / derived - 1.813_26).abs() < 1e-3,
            "the YaRN temperature is 1.81x the plain one, not {}",
            m.attn_scale / derived
        );
    }

    /// The synthetic Metal fixture IS this projection, field for field.
    ///
    /// `LlamaLikeMetalFacts::gpt_oss_20b()` is what four backends' text
    /// tests fire, and `metal_facts` is what a real load reads. Two
    /// readings of one row, so the only honest relation is equality --
    /// and stated as a whole struct rather than a list of fields,
    /// because the two drifts this pair has had were both in fields no
    /// list contained.
    ///
    /// `rope_freq_table` was the first: the fixture said `false` and
    /// called 150000 "a plain geometric ladder", which was true of
    /// nothing. `attn_scale` was the second, and it was 0.0 -- the
    /// "derive `1/sqrt(head_dim)`" sentinel -- where this row rescales
    /// by YaRN. See `yarn_softmax_scale`: a 1.81x error in the softmax
    /// temperature does not fault, it sharpens every distribution in the
    /// stack. Both were found by comparing everything at once, after
    /// the enumerations that were supposed to hold this pair together
    /// had passed.
    ///
    /// The binding half is pinned to the deployment the sibling test
    /// builds, because those five fields are a load's observation and
    /// not the row's claim.
    #[test]
    fn the_synthetic_metal_fixture_is_this_projection() {
        use crate::shared::llama_like::forward::facts::LlamaLikeMetalFacts;
        // The binding a real load OBSERVES, not the sibling test's.
        //
        // This mattered, and it is the reason this test missed a field on
        // its first pass. `model::binding::observed` reads the router
        // gate's affine point off the checkpoint and states it whenever it
        // differs from the stack's, and this row's `config.json` puts 98
        // dense tensors at group 64 / 4 bits and 24 `mlp.router` gates at
        // group 64 / EIGHT. Written against `(0, 0)` this comparison
        // agreed with a fixture that bound no second point -- two readings
        // of a deployment nobody loads.
        let bind = crate::catalog::MetalBinding {
            qmm_partial_rows: false,
            qmm_fp16_precast: true,
            qmm_tile: None,
            quant_group: 64,
            quant_bits: 4,
            router_quant_group: 64,
            router_quant_bits: 8,
            moe_mxfp4: true,
            fuse_residual_gemv: true,
            paged_multi_batch: true,
            qmm_multi_batch: true,
            add_bias: true,
            fused_qk_rope: false,
        };
        assert_eq!(
            LlamaLikeMetalFacts::gpt_oss_20b(),
            metal_facts(&f20b(), &bind),
            "the fixture and the projection are one row"
        );
    }

    /// FOUR attention biases, not three: q, k, v and the LANDING.
    ///
    /// The shared Metal text adds the landing's bias behind `o_bias &&
    /// add_bias`, and gpt-oss is the only row in the catalog that states
    /// `o_bias` -- so the arm belongs to this row and had never been
    /// traced. The sibling test on the shared text asserts three launches
    /// against a qwen row, which is right for qwen and is exactly why it
    /// could not see this one.
    ///
    /// The failure is quiet and the term is not small: `o_proj`'s bias is
    /// added to the attention output before the residual, so dropping it
    /// removes a per-channel offset from every layer of every token. No
    /// shape error and no unbound symbol -- the tensor is declared by the
    /// contract, read off disk, allocated, and never summed.
    ///
    /// Asserted by WEIGHT NAME rather than by count, because a count is
    /// satisfied by any four launches and this row also biases its
    /// router: the number would have been right with the landing missing
    /// and something else doubled.
    #[test]
    fn the_attention_landing_takes_its_own_bias_beside_the_three_projections() {
        use crate::shared::llama_like::forward::llama_like_metal;
        use model_ir::trace::{FireClass, OpKind};
        let f = f20b();
        let bind = crate::catalog::MetalBinding {
            qmm_partial_rows: false,
            qmm_fp16_precast: true,
            qmm_tile: None,
            quant_group: 64,
            quant_bits: 4,
            router_quant_group: 0,
            router_quant_bits: 0,
            moe_mxfp4: true,
            fuse_residual_gemv: true,
            paged_multi_batch: true,
            qmm_multi_batch: true,
            add_bias: true,
            fused_qk_rope: false,
        };
        let shape = metal_shape(&f);
        assert!(shape.o_bias, "the row this test reads states one");
        let biased = |m: &crate::shared::llama_like::forward::facts::LlamaLikeMetalFacts| {
            llama_like_metal(&shape, m, FireClass::Decode)
                .ops
                .iter()
                .filter_map(|op| match &op.kind {
                    OpKind::Launch {
                        kernel, weights, ..
                    } if kernel.contains("add_bias") => Some(weights.clone()),
                    _ => None,
                })
                .flatten()
                .filter(|w| w.starts_with("layer.0."))
                .collect::<Vec<_>>()
        };
        let on = biased(&metal_facts(&f, &bind));
        assert!(
            on.contains(&"layer.0.o_bias".to_string()),
            "the attention landing's bias is never summed: {on:?}"
        );
        for projection in ["q_bias", "k_bias", "v_bias"] {
            assert!(
                on.contains(&format!("layer.0.{projection}")),
                "{projection} went missing while the landing was added: {on:?}"
            );
        }

        let mut off = metal_facts(&f, &bind);
        off.add_bias = false;
        assert!(
            biased(&off).is_empty(),
            "the checkpoint's biases and this build's Metal path are two \
             facts, and either one alone must not produce a launch"
        );
    }

    /// Every extent is the row's own arithmetic.
    #[test]
    fn the_attention_rows_are_the_rows_own_arithmetic() {
        let f = f20b();
        let m = manifest(&f);
        let q = m
            .tensors
            .iter()
            .find(|t| t.name.ends_with("q_proj"))
            .expect("stated");
        assert_eq!(
            q.extents,
            vec![u64::from(f.q_heads * f.head_dim), u64::from(f.hidden)]
        );
        let k = m
            .tensors
            .iter()
            .find(|t| t.name.ends_with("k_proj"))
            .expect("stated");
        assert_eq!(
            k.extents,
            vec![u64::from(f.kv_heads * f.head_dim), u64::from(f.hidden)]
        );
    }

    /// THE SINKS. One learned logit per query head, and the only row in
    /// this build that ships them — which is why the attention
    /// statement here produces an LSE beside its output.
    #[test]
    fn the_sinks_are_one_logit_per_query_head() {
        let f = f20b();
        let sinks = manifest(&f)
            .tensors
            .into_iter()
            .find(|t| t.name.ends_with("self_attn.sinks"))
            .expect("gpt-oss states its sinks");
        assert_eq!(sinks.presence, Presence::Required);
        assert_eq!(sinks.extents, vec![u64::from(f.q_heads)]);

        // Required rather than optional, which is what separates this row
        // from every other MoE decoder of the same geometry: a mixtral
        // that happens to match on width is refused HERE, by tensors it
        // does not carry.
        //
        // The four attention biases are in the same list. The trace folds
        // them into each projection's epilogue, so a checkpoint that
        // matched this row without them would bind an epilogue to a
        // tensor that is not there.
        for absent in [
            "self_attn.sinks",
            "self_attn.q_proj.bias",
            "self_attn.k_proj.bias",
            "self_attn.v_proj.bias",
            "self_attn.o_proj.bias",
        ] {
            let m = manifest(&f);
            let without = Observed::from_pairs(
                m.tensors
                    .iter()
                    .filter(|t| t.presence != Presence::Absent && !t.name.ends_with(absent))
                    .map(|t| (t.name.clone(), t.extents.clone())),
            );
            assert!(
                m.check(&without).is_err(),
                "a decoder with no {absent} is not a gpt-oss, and this row said it was"
            );
        }
    }

    /// THE CLAIM: one row serves the MXFP4 build and the bf16 one.
    ///
    /// The two releases differ by an added `quantization_config` and by
    /// how the expert bank is SPELLED — `gate_up_proj_blocks` plus a
    /// scale plane, or a plain `gate_up_proj`. Both checkpoints satisfy
    /// this row's manifest, because the manifest names neither.
    #[test]
    fn an_mxfp4_checkpoint_and_a_bf16_one_satisfy_the_same_row() {
        let f = f20b();
        let m = manifest(&f);
        let common = implied(&m);

        let mut mxfp4 = common.clone();
        mxfp4.push((
            "layer.0.mlp.experts.gate_up_proj_blocks".into(),
            vec![32, 5760, 90, 16],
        ));
        mxfp4.push((
            "layer.0.mlp.experts.gate_up_proj_scales".into(),
            vec![32, 5760, 90],
        ));
        mxfp4.push((
            "layer.0.mlp.experts.down_proj_blocks".into(),
            vec![32, 2880, 90, 16],
        ));
        mxfp4.push((
            "layer.0.mlp.experts.down_proj_scales".into(),
            vec![32, 2880, 90],
        ));

        let mut bf16 = common;
        bf16.push((
            "layer.0.mlp.experts.gate_up_proj".into(),
            vec![32, 5760, 2880],
        ));
        bf16.push(("layer.0.mlp.experts.down_proj".into(), vec![32, 2880, 2880]));

        for (what, pairs) in [("mxfp4", mxfp4), ("bf16", bf16)] {
            let observed = Observed::from_pairs(pairs);
            assert!(
                m.check(&observed).is_ok(),
                "{what}: {:?}",
                m.check(&observed)
            );
        }
    }

    /// MLX publishes a THIRD gpt-oss and this row now knows it.
    ///
    /// The test above names two releases and [`manifest`]'s doc named
    /// the same two, so "both spellings" read as "every spelling".
    /// `mlx-community/gpt-oss-20b-MXFP4-Q4` is a third: it splits the
    /// expert bank into `gate_proj` / `up_proj` / `down_proj`, and it
    /// splits the BIAS with it — which is the half that mattered,
    /// because the bias is what this manifest pins in order NOT to
    /// match on an encoding. No `gate_up_proj_bias` exists in that
    /// checkpoint at any extent, and the snapshot identified as
    /// NOTHING.
    ///
    /// It was left as a refusal because "either of these names" was a
    /// change to the shape of identity that a test should not make
    /// silently. It is made now, in [`TensorSpec::instead`], and made
    /// narrow: an alternative applies only when EVERY name in it is
    /// published at agreeing extents, so a checkpoint with a gate and
    /// no up is still not a mixture. `Observed::logical` was the wrong
    /// home for it — the two publications DIVIDE the bank, not just
    /// spell it, and even the part that is only spelling (`down_proj_
    /// bias` against `down_proj.bias`) cannot be a rule there, because
    /// a general `_bias` -> `.bias` would rename nemotron-h's
    /// `mixer.dt_bias` to a tensor no checkpoint holds.
    ///
    /// This test now asserts the acceptance, and the layout is stated
    /// at MLX's real extents — `[32, 2880]` per leg, read off the
    /// snapshot's safetensors headers rather than assumed from the
    /// fused width halved.
    #[test]
    fn a_split_expert_bank_is_a_third_publication_this_row_now_knows() {
        let f = f20b();
        let m = manifest(&f);

        let fused = "layer.0.mlp.experts.gate_up_proj_bias";
        let mut split: Vec<(String, Vec<u64>)> = implied(&m)
            .into_iter()
            .filter(|(n, _)| n != fused)
            .collect();
        // MLX's spelling, at MLX's extents: one bias per expert per
        // branch, each the projection's own width rather than the
        // fused `2 *`.
        for leg in ["gate_proj", "up_proj"] {
            split.push((
                format!("layer.0.mlp.experts.{leg}.bias"),
                vec![32, u64::from(f.intermediate)],
            ));
            split.push((
                format!("layer.0.mlp.experts.{leg}.weight"),
                vec![32, u64::from(f.intermediate), u64::from(f.hidden)],
            ));
        }

        // `down_proj` is split the same way and spelled the same way,
        // so the fixture has to carry MLX's name for it too or the test
        // would be measuring one acceptance and calling it two.
        let fused_down = "layer.0.mlp.experts.down_proj_bias";
        let mut split: Vec<(String, Vec<u64>)> =
            split.into_iter().filter(|(n, _)| n != fused_down).collect();
        split.push((
            "layer.0.mlp.experts.down_proj.bias".into(),
            vec![32, u64::from(f.hidden)],
        ));

        assert!(
            m.check(&Observed::from_pairs(split.clone())).is_ok(),
            "an MLX-divided gpt-oss identifies as this row: {:?}",
            m.check(&Observed::from_pairs(split.clone()))
        );

        // Half a layout is not a layout. A checkpoint that publishes a
        // gate bias and no up bias has neither publication's bank, and
        // accepting it would identify a mixture by one of its halves.
        let half: Vec<(String, Vec<u64>)> = split
            .into_iter()
            .filter(|(n, _)| n != "layer.0.mlp.experts.up_proj.bias")
            .collect();
        let err = m
            .check(&Observed::from_pairs(half))
            .expect_err("a gate without an up is not a divided bank");
        assert!(
            format!("{err:?}").contains("gate_up_proj_bias"),
            "the row should name the quantity it could not find under \
             any layout, and said: {err:?}"
        );
    }

    /// Every projection satisfies the checkpoint it implies.
    #[test]
    fn each_manifest_is_satisfied_by_the_checkpoint_it_implies() {
        for f in [
            f20b(),
            GptOssFacts {
                layers: 36,
                experts: 128,
                ..f20b()
            },
        ] {
            let m = manifest(&f);
            let observed = Observed::from_pairs(implied(&m));
            assert!(m.check(&observed).is_ok(), "{:?}", m.check(&observed));
        }
    }

    /// The alternation is a RULE, expanded here and nowhere else. The
    /// old path had the deployment read this list back out of the CUDA
    /// facts, so the tracer's schedule and the launcher's were two
    /// statements of one fact.
    #[test]
    fn the_window_alternates_from_layer_zero() {
        let f = f20b();
        let d = deployment(&f, 150_000.0, 1e-5, 128);
        let windows: Vec<i32> = d.attention.iter().map(|a| a.window).collect();
        assert_eq!(windows[..4], [128, -1, 128, -1]);
        assert_eq!(windows.len(), 24);
        for (l, w) in windows.iter().enumerate() {
            assert_eq!(*w == 128, f.is_sliding(l as u32), "layer {l}");
        }
        assert_eq!(d.windows(), windows, "a mixed table is a real table");
    }

    /// The launch geometry is the row's own numbers, and `intermediate`
    /// is the per-expert width because gpt-oss has no dense block.
    #[test]
    fn the_launch_geometry_is_the_rows_own_numbers() {
        let f = f20b();
        let d = deployment(&f, 150_000.0, 1e-5, 128);
        assert_eq!(d.layers, 24);
        assert_eq!(d.shape.hidden, 2880);
        assert_eq!(d.shape.q_heads, 64);
        assert_eq!(d.shape.kv_heads, 8);
        assert_eq!(d.shape.head_dim, 64);
        assert_eq!(
            d.shape.head_dim_kernel, 64,
            "64 is instantiated; nothing pads"
        );
        assert_eq!(d.shape.gqa_group(), 8, "64 q over 8 kv");
        assert_eq!(
            d.shape.intermediate, 0,
            "no dense block anywhere in the stack"
        );
        assert_eq!(d.shape.moe_intermediate, 2880, "one expert's width");
        assert_eq!(
            d.shape.widest_mlp(),
            2880,
            "what the shared workspace must hold"
        );
        assert_eq!(
            d.norm_eps, 1e-5,
            "gpt-oss's own, and not the llama lineage's 1e-6"
        );
        assert_eq!(d.shape.vocab, 201_088);
        for a in &d.attention {
            assert_eq!(a.rope_theta, 150_000.0);
            assert_eq!(a.rotary_dim, 0, "full rotation at the head dim");
            assert_eq!(a.sm_scale, 1.0 / 8.0, "1/sqrt(64)");
        }
    }

    /// Paged pages, no recurrent state, pre-norm, nothing capped —
    /// stated, because a default body is a claim about rows nobody has
    /// written yet.
    #[test]
    fn the_rest_of_the_deployment_is_stated_rather_than_defaulted() {
        let d = deployment(&f20b(), 150_000.0, 1e-5, 128);
        assert_eq!(d.kv, KvStyle::Paged);
        assert!(d.recurrent.is_none(), "a mixture is not a recurrence");
        assert_eq!(d.prefill, PrefillStyle::Planned);
        assert_eq!(d.attn_output, AttnOutput::DriverPinned);
        assert_eq!(d.norm, NormPlacement::Pre);
        assert_eq!(d.logit_softcap, 0.0);
        assert_eq!(d.ple_dim, 0);
        assert!(
            d.scales.is_empty(),
            "the swiglu clamp is a kernel, not a scalar"
        );
        assert!(!d.shares_kv());
    }

    /// The window lives in ONE place: the deployment's table, expanded
    /// from `is_sliding`. The tracer is handed no copy of it.
    ///
    /// It used to be handed one, and read it, and pass it to a dispatch
    /// that ignores it -- gpt-oss takes the sink spelling on every layer,
    /// and that spelling states no window because the driver reads
    /// `window_left_by_layer` out of the deployment instead.
    #[test]
    fn the_binding_facts_carry_the_same_schedule() {
        let f = f20b();
        let cuda = cuda_facts(&f, Deployed::single());
        assert_eq!(
            deployment(&f, 150_000.0, 1e-5, 128).windows(),
            (0..f.layers)
                .map(|l| if f.is_sliding(l) { 128 } else { -1 })
                .collect::<Vec<_>>(),
            "the deployment is the only place the alternation is written"
        );
        assert_eq!(cuda.mxfp4_decode_max_routes, 32 * f.experts);
        assert!(cuda.mxfp4_decode_gemv);
        assert!(!cuda.streamed_experts);
    }

    /// The text exists for every class a fire can carry — which it did
    /// not, once: gpt-oss had a facts row and only a Prefill arm, so a
    /// checkpoint loaded, reported itself healthy and died at its first
    /// decode.
    #[test]
    fn every_fire_class_traces() {
        use model_ir::trace::FireClass;
        for class in [FireClass::Decode, FireClass::Prefill] {
            let plan = trace(&f20b(), class, Deployed::single());
            assert!(!plan.ops.is_empty(), "{class:?} traced nothing");
        }
    }
}

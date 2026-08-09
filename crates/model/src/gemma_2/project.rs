//! The three projections a gemma-2 row makes.
//!
//! Gemma 2 is not a `llama_like` configuration and cannot borrow that
//! family's projections: its blocks carry a norm PAIR each (pre and
//! post, around both attention and the MLP), its attention alternates
//! local and global on a two-layer cycle, and its logits are capped at
//! the end. So the three answers are written here, once, taking a
//! `&Gemma2Facts` — the same N:1 the old `HF_ROWS` and `FACTS_ROWS`
//! columns expressed for this generation, spelled as a call rather than
//! as two tables keyed on the string `"gemma2"` that nothing held to
//! each other.

// Only the texts name a backend, and only they are gated.
use crate::catalog::Deployed;
use crate::deployment::{
    Advertised, AttnOutput, Deployment, Geometry, KvStyle, LayerAttention, NormPlacement,
    PrefillStyle,
};
use crate::manifest::{Manifest, TensorSpec};

use super::spec::Gemma2Facts;

/// The final-logit cap gemma-2 applies, when it applies one.
///
/// A CONSTANT and not a config read, which is what the old derivation
/// did too (`if self.final_logit_softcap { 30.0 } else { 0.0 }`), and it
/// is worth saying why that is right rather than lazy: every published
/// gemma-2 states `final_logit_softcapping: 30.0`, and the corpus's
/// `synthetic--gemma-null-softcap.json` exists precisely because the
/// field can be `null` — which means NO CAP, not "cap at zero" and not a
/// panic. A `bool` on the shape and a constant here cannot express the
/// third thing.
pub const FINAL_LOGIT_SOFTCAP: f32 = 30.0;

/// The ATTENTION cap gemma-2 applies, when it applies one.
///
/// `cap * tanh(score / cap)` over the attention scores, which is a
/// DIFFERENT cap in a different place from the readout's: every
/// published gemma-2 states `attn_logit_softcapping: 50.0` against
/// `final_logit_softcapping: 30.0`. A constant here for the reason the
/// one above is a constant, and `synthetic--gemma-null-softcap.json`
/// covers the third case for both.
pub const ATTN_LOGIT_SOFTCAP: f32 = 50.0;

/// This row's tensors.
///
/// Every extent is the row's own arithmetic, which is what makes the
/// manifest a check rather than a second statement of the numbers.
///
/// The four norms are the interesting rows. Gemma-2 ships a PAIR around
/// each sub-layer — `input_layernorm` and `post_attention_layernorm`
/// around attention, `pre_feedforward_layernorm` and
/// `post_feedforward_layernorm` around the MLP — and that pair is what
/// tells a gemma-2 checkpoint apart from a llama-like one whose extents
/// otherwise agree. The old loader sniffed for
/// `pre_feedforward_layernorm.weight` to decide the same thing; a sniff
/// inside a derivation is an expectation with nowhere to be written
/// down.
#[must_use]
pub fn manifest(f: &Gemma2Facts) -> Manifest {
    let (hidden, vocab) = (u64::from(f.hidden), u64::from(f.vocab));
    let (q, kv) = (u64::from(f.attn.q_width()), u64::from(f.attn.kv_width()));
    let inter = u64::from(f.intermediate);

    Manifest::new(f.layers)
        .with(TensorSpec::required("embed_tokens", [vocab, hidden]))
        .with(TensorSpec::required("norm", [hidden]))
        // Gemma-2 ties, and a tie is an ABSENCE: it is the only thing
        // that tells tied from untied when every extent agrees.
        .either(!f.tied_embeddings, "lm_head", [vocab, hidden])
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
        // gemma-2 proper has no q/k norm. Stated as an ABSENCE rather
        // than left unsaid, because a later gemma that does ship one
        // must not match this row.
        .either(
            f.attn.qk_norm,
            "layer.{}.self_attn.q_norm",
            [u64::from(f.attn.head_dim)],
        )
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
}

/// This row's deployment.
///
/// A projection, and every value below was already in the row or is a
/// constant of the family. What it replaces is `gemma2_facts_from_hf`
/// plus an `impl PlannedFamily` plus `deployment_of` — three hops that
/// read a parsed `config.json` to arrive at these numbers.
#[must_use]
pub fn deployment(f: &Gemma2Facts, rope_theta: f32, norm_eps: f32) -> Deployment {
    // The checkpoint's own head dim, unpadded: 256 and 128 are both
    // instantiated, so the rounding is the identity here and the
    // question is asked rather than assumed.
    let head_dim = crate::deployment::round_up_attn_head_dim(f.attn.head_dim);
    let attention = (0..f.layers)
        .map(|l| LayerAttention {
            // One shape for every layer, which is what this row was
            // already saying by having no per-layer count.
            kv_heads: f.attn.kv_heads,
            head_dim,
            // The alternation, per layer. This is the vector that used
            // to be a shape field.
            window: f.window_left_at(l),
            // Every layer owns its pages. KV sharing is gemma-4's.
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
        shape: Geometry {
            hidden: f.hidden,
            q_heads: f.attn.heads,
            kv_heads: f.attn.kv_heads,
            head_dim: f.attn.head_dim,
            head_dim_kernel: head_dim,
            intermediate: f.intermediate,
            // Dense throughout; gemma's mixture is gemma-4.
            moe_intermediate: 0,
            experts_per_token: 0,
            shared_intermediate: 0,
            vocab: f.vocab,
        },
        attention,
        kv: KvStyle::Paged,
        recurrent: None,
        prefill: PrefillStyle::Planned,
        // The guard region records no SSA output for this text, so the
        // driver owns the landing buffer. Gemma-4 is the exception and
        // its own row says so.
        attn_output: AttnOutput::DriverPinned,
        logit_softcap: if f.final_logit_softcap {
            FINAL_LOGIT_SOFTCAP
        } else {
            0.0
        },
        // The ATTENTION cap, which the shape has measured all along and
        // nothing carried. `Gemma2AttnFacts::attn_logit_softcap`'s doc
        // said it rides "as a DISPATCH parameter, not a launch: the
        // attention kernel takes it, so nothing states it separately" —
        // and the kernel does take it, `Source::Attn("logits_soft_cap")`
        // on every flashinfer entry point. What nothing did was FILL it:
        // `AttnCtx::logits_soft_cap` was the literal `0.0`, so a gemma-2
        // with the cap and one without attended identically. A constant
        // for the same reason `FINAL_LOGIT_SOFTCAP` is one.
        attn_logit_softcap: if f.attn.attn_logit_softcap {
            ATTN_LOGIT_SOFTCAP
        } else {
            0.0
        },
        // No per-layer embeddings; that is gemma-3n and gemma-4.
        ple_dim: 0,
        // PRE, and gemma's second norm of each pair does not change
        // that: `NormPlacement` answers which buffer a projection reads
        // — the normed value or the residual — and gemma-2 projects the
        // normed one. The post-norms sit on the block's OUTPUT, which is
        // a fact of the traced text and not of the driver's staging.
        norm: NormPlacement::Pre,
        // gemma-2 pairs the sandwich with the offset, which is what made
        // reading this off the placement look sound for three generations.
        norm_unit_offset: true,
        // gemma-3 carries the per-head q/k norm and NO V norm: the two
        // are separate facts, and this row is where that is said.
        v_norm: false,
        // Dense: no router reads this.
        norm_topk_prob: true,
        // No router of this family states a scaling factor.
        routed_scaling: 1.0,
        mlp_gate: crate::deployment::MlpGate::GeluTanh,
        // No named constants: the `sqrt(hidden)` embedding scale is
        // stated inside the trace, not looked up by name.
        scales: std::collections::BTreeMap::new(),
        // Filled by the ROW, not by the shape: a family label and a
        // published context ceiling are facts about a checkpoint, and a
        // projection only sees geometry.
        advertised: Advertised::default(),
        rope_scaling: None,
        towers: Default::default(),
    }
}

/// Why this build has no Metal text for a gemma-2 row.
///
/// A `const` so the test that asserts the refusal NAMES the missing
/// thing compares against the same string the caller is shown, rather
/// than against a paraphrase that can drift away from it — the shape
/// `csm::project::NO_TRACE` set for the same reason.
///
/// Its forward is `gemma2_cuda`, which states gemma-2's caps —
/// the attention logit cap and the final logit cap — and
/// `llama_like_metal`, the one Metal text here, has a `logit_softcap`
/// field it reads for the final one but no attention cap at all. A
/// gemma-2 traced without its attention cap runs, and its softmax
/// saturates differently in every layer.
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
pub const NO_METAL: &str = "gemma-2 has no Metal text in this build: its forward is `gemma2_cuda`, whose \
     attention logit cap has no counterpart in the one Metal text here \
     (`llama_like_metal`), and whose shape is `Gemma2Facts` rather than the \
     `LlamaLikeFacts` that text takes; the CUDA backend serves this row";

/// Trace this row's CUDA text for one fire class.
///
/// `load` is unread and named anyway: gemma-2 binds nothing from the
/// load — no TP-sharded fused bank to ask about, no host scalars — and
/// the three projections keep one shape so a reader can see which
/// generation reads what.
#[must_use]
pub fn trace(
    f: &Gemma2Facts,
    class: model_compiler::trace::FireClass,
    load: Deployed<'_>,
) -> model_compiler::trace::ForwardPlan {
    let _ = load;
    super::forward::gemma2_cuda(f, class)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::Presence;

    fn f() -> Gemma2Facts {
        Gemma2Facts::gemma_2_9b()
    }

    /// The norm PAIR is what a gemma-2 checkpoint has and a llama-like
    /// one does not, so it is an expectation rather than a sniff for
    /// `pre_feedforward_layernorm.weight` inside a derivation.
    #[test]
    fn the_sandwich_pair_is_expected_rather_than_sniffed_for() {
        let m = manifest(&f());
        for name in [
            "layer.{}.input_layernorm",
            "layer.{}.post_attention_layernorm",
            "layer.{}.pre_feedforward_layernorm",
            "layer.{}.post_feedforward_layernorm",
        ] {
            let spec = m.tensors.iter().find(|t| t.name == name).expect("stated");
            assert_eq!(spec.presence, Presence::Required, "{name}");
            assert_eq!(spec.extents, vec![3584]);
        }
    }

    /// Every extent is the row's arithmetic: 16 heads of 256 over a
    /// 3584-wide residual.
    #[test]
    fn the_extents_are_the_rows_own_arithmetic() {
        let m = manifest(&f());
        let ext = |n: &str| {
            m.tensors
                .iter()
                .find(|t| t.name == n)
                .expect("stated")
                .extents
                .clone()
        };
        assert_eq!(ext("layer.{}.self_attn.q_proj"), vec![4096, 3584]);
        assert_eq!(ext("layer.{}.self_attn.k_proj"), vec![2048, 3584]);
        assert_eq!(ext("layer.{}.self_attn.o_proj"), vec![3584, 4096]);
        assert_eq!(ext("layer.{}.mlp.gate_proj"), vec![14336, 3584]);
        assert_eq!(ext("layer.{}.mlp.down_proj"), vec![3584, 14336]);
        assert_eq!(ext("embed_tokens"), vec![256_000, 3584]);
        assert_eq!(m.layers, 42);
    }

    /// A tie is an absence, and gemma-2 ties.
    #[test]
    fn the_tied_head_is_an_absence_the_manifest_expects() {
        let m = manifest(&f());
        let head = m
            .tensors
            .iter()
            .find(|t| t.name == "lm_head")
            .expect("stated");
        assert_eq!(head.presence, Presence::Absent);
    }

    /// The alternation reaches `Deployment` layer by layer — the same
    /// forty-two numbers `window_by_layer()` used to hand `deployment_of`
    /// out of a `Vec` field.
    #[test]
    fn the_alternating_window_reaches_every_layer() {
        let f = f();
        let d = deployment(&f, 10_000.0, 1e-6);
        let windows: Vec<i32> = d.attention.iter().map(|a| a.window).collect();
        assert_eq!(windows, f.window_by_layer());
        assert_eq!(windows.len(), 42);
        assert_eq!(windows[0], 4096);
        assert_eq!(windows[1], -1);
        assert!(windows.contains(&-1) && windows.contains(&4096));
    }

    /// The final cap reaches the deployment, and a checkpoint that
    /// states `null` gets NO cap rather than a cap of zero applied or a
    /// panic — which is what `synthetic--gemma-null-softcap.json` is in
    /// the corpus to say.
    #[test]
    fn a_null_final_softcap_is_no_cap_and_not_a_panic() {
        let capped = deployment(&f(), 10_000.0, 1e-6);
        assert_eq!(capped.logit_softcap, 30.0);

        let mut uncapped = f();
        uncapped.final_logit_softcap = false;
        assert_eq!(deployment(&uncapped, 10_000.0, 1e-6).logit_softcap, 0.0);
    }

    /// The launch geometry is the row's own numbers, so a fire and a
    /// trace cannot disagree about how many heads there are.
    #[test]
    fn the_launch_geometry_is_the_rows_own_numbers() {
        let f = f();
        let d = deployment(&f, 10_000.0, 1e-6);
        assert_eq!(d.shape.hidden, 3584);
        assert_eq!(d.shape.q_heads, 16);
        assert_eq!(d.shape.kv_heads, 8);
        assert_eq!(d.shape.head_dim, 256);
        assert_eq!(
            d.shape.head_dim_kernel, 256,
            "256 is instantiated; nothing pads"
        );
        assert_eq!(d.shape.intermediate, 14336);
        assert_eq!(d.shape.vocab, 256_000);
        assert_eq!(d.shape.gqa_group(), 2);
        assert_eq!(d.layers, 42);
        assert_eq!(d.attention.len(), 42);
    }

    /// The answers the old derivation gave by DEFAULT are stated here,
    /// which is the point of having no default bodies: each of these was
    /// a claim about every family that had not been written yet.
    #[test]
    fn the_defaults_the_old_vtable_supplied_are_stated() {
        let d = deployment(&f(), 10_000.0, 1e-6);
        assert_eq!(d.kv, KvStyle::Paged);
        assert!(d.recurrent.is_none());
        assert_eq!(d.prefill, PrefillStyle::Planned);
        assert_eq!(d.attn_output, AttnOutput::DriverPinned);
        assert_eq!(
            d.ple_dim, 0,
            "per-layer embeddings are gemma-3n's and gemma-4's"
        );
        assert_eq!(d.norm, NormPlacement::Pre);
        assert!(d.scales.is_empty());
        for (l, a) in d.attention.iter().enumerate() {
            assert_eq!(a.kv_source, l as u32, "every layer owns its pages");
            assert_eq!(a.rope_theta, 10_000.0);
            assert_eq!(a.rotary_dim, 0, "full rotation at the head dim");
            assert_eq!(a.head_dim, 256);
            assert!((a.sm_scale - 1.0 / 16.0).abs() < 1e-6, "1/sqrt(256)");
        }
    }

    /// The 27B's narrower heads reach both the geometry and the scale,
    /// so a row that differs only in head width deploys differently.
    #[test]
    fn a_narrower_head_reaches_the_scale_and_the_geometry() {
        let mut f = f();
        f.attn.head_dim = 128;
        f.attn.heads = 32;
        f.attn.kv_heads = 16;
        let d = deployment(&f, 10_000.0, 1e-6);
        assert_eq!(d.shape.head_dim, 128);
        assert!((d.attention[0].sm_scale - 1.0 / 128f32.sqrt()).abs() < 1e-6);
    }
}

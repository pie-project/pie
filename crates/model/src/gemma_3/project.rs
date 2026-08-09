//! The three projections a gemma-3 row makes.
//!
//! Gemma 3 IS a `llama_like` configuration — same attention, same
//! block, and the family's own tracer serves it — so this module is a
//! wrapper and not a fourth family. What it adds is the two things
//! `LlamaLikeFacts` deliberately does not hold, both of them per LAYER:
//!
//! * **Two rope bases.** Gemma 3 rotates its five sliding layers at
//!   10 000 and its sixth, full-attention layer at 1 000 000. The shape
//!   holds one `RopeKind` and no theta at all — theta reaches the driver
//!   through `Deployment`, per layer — so the two bases are ROW fields
//!   and this module builds the schedule.
//! * **The 5:1 window.** `deployment_cuda` reached gemma-3 through
//!   `llama_like_facts_from_hf`, whose `window_by_layer` is empty, so
//!   `deployment_of` broadcast the config's single `sliding_window` to
//!   all sixty-two layers. Every sixth layer then attended 1024 tokens
//!   instead of the whole context — silently, because a window that is
//!   too small produces fluent text about the wrong prefix.
//!
//! The same derivation also read `norm_variant: Plain` and
//! `norm_placement: Pre` for gemma-3, because the generic reader had no
//! way to ask and the row it landed in had no way to say. Both are
//! stated on the rows now, which is the difference the catalog makes:
//! the numbers are not derived from a config, they ARE the model.

use crate::catalog::{Backend, Deployed, MetalBinding};
use crate::deployment::{
    Advertised,
    AttnOutput, Deployment, Geometry, KvStyle, LayerAttention, NormPlacement, PrefillStyle,
};
use crate::shared::llama_like::project as family;
use crate::shared::llama_like::spec::LlamaLikeFacts;
use crate::manifest::{Manifest, TensorSpec};

/// What gemma-3 states about its layers that the family shape cannot.
///
/// Five numbers, all of them from the checkpoint's own `config.json`,
/// and each on the ROW rather than in `LlamaLikeFacts` because twelve
/// generations share that struct and eleven of them have nothing to say
/// here.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Schedule {
    /// `sliding_window`: how far back a LOCAL layer sees.
    pub sliding_window: i32,
    /// `sliding_window_pattern`: every `interval`-th layer attends the
    /// whole context. Six on every published gemma-3 — five local, then
    /// one global.
    pub full_attn_interval: u32,
    /// `rope_local_base_freq`, the base the sliding layers rotate at.
    pub rope_theta_local: f32,
    /// `rope_theta`, the base the full-attention layers rotate at. Two
    /// orders of magnitude apart from the local one, which is why
    /// broadcasting either is not a rounding error.
    pub rope_theta_global: f32,
    /// `query_pre_attn_scalar`: the softmax scale is `1/sqrt` of THIS,
    /// and it is not always the head dim. Gemma-3-27B states 168 with
    /// 128-wide heads, and HF is unambiguous — `scaling =
    /// query_pre_attn_scalar ** -0.5` — so a scale derived from the head
    /// dim is wrong for exactly that row.
    pub query_pre_attn_scalar: u32,
}

impl Schedule {
    /// Whether layer `l` attends the whole context.
    #[must_use]
    pub fn is_full_attn(&self, l: u32) -> bool {
        model_compiler::facts::full_attn_at(self.full_attn_interval, l)
    }

    /// The window layer `l` attends over, `-1` for the whole context.
    #[must_use]
    pub fn window_at(&self, l: u32) -> i32 {
        if self.is_full_attn(l) { -1 } else { self.sliding_window }
    }

    /// The rope base layer `l` rotates at.
    #[must_use]
    pub fn rope_theta_at(&self, l: u32) -> f32 {
        if self.is_full_attn(l) { self.rope_theta_global } else { self.rope_theta_local }
    }
}

/// This row's tensors: the family's, with gemma's norms corrected.
///
/// The family projection expresses norm placement as a PAIR of either-s
/// — `input_layernorm` when `Pre`, `post_feedforward_layernorm` when
/// `Post` — which is exactly right for the two placements that existed
/// when it was written and answers `Absent` to both under `Sandwich`. A
/// gemma-3 checkpoint ships all four norms, so that manifest would
/// refuse every real gemma-3 twice over. The four are restated here,
/// where the generation that has them lives, rather than by teaching a
/// shared projection about a third case it has no other reader for.
///
/// Which means the strip below has to name whatever the family states
/// as placement, and an `Absent` row left standing is not inert: a
/// manifest that says a tensor is absent and then requires it refuses a
/// checkpoint that ships it. When the family's Post marker moved from
/// `post_attention_layernorm` to `post_feedforward_layernorm` — because
/// HF ships the former under BOTH placements and it discriminates
/// nothing — this list moved with it.
#[must_use]
pub fn manifest(f: &LlamaLikeFacts) -> Manifest {
    let hidden = u64::from(f.hidden);
    let head_dim = u64::from(f.head_dim);
    let base = family::manifest(f);

    let mut out = Manifest::new(base.layers);
    for spec in base.tensors {
        // Everything the family states as placement or as always-there;
        // gemma has all of them, restated unconditionally below.
        if spec.name == "layer.{}.input_layernorm"
            || spec.name == "layer.{}.post_attention_layernorm"
            || spec.name == "layer.{}.post_feedforward_layernorm"
        {
            continue;
        }
        out = out.with(spec);
    }
    out.with(TensorSpec::required("layer.{}.input_layernorm", [hidden]))
        .with(TensorSpec::required("layer.{}.post_attention_layernorm", [hidden]))
        .with(TensorSpec::required("layer.{}.pre_feedforward_layernorm", [hidden]))
        .with(TensorSpec::required("layer.{}.post_feedforward_layernorm", [hidden]))
        // Gemma-3 norms K as well as Q, and both are per head. The
        // family states only `q_norm`, because that is the one whose
        // EXTENT answers the three-way question; stating k_norm here
        // costs nothing and is a fact about every gemma-3.
        .with(TensorSpec::required("layer.{}.self_attn.k_norm", [head_dim]))
}

/// This row's deployment: the family's shape, on gemma's schedule.
///
/// Written out rather than delegated to `family::deployment`, which
/// takes ONE theta and ONE window and would therefore have to be given
/// the wrong one five times in six.
#[must_use]
pub fn deployment(f: &LlamaLikeFacts, s: &Schedule, norm_eps: f32) -> Deployment {
    let head_dim = family::round_up_attn_head_dim(f.head_dim).max(f.head_dim);
    let sm_scale = 1.0 / (s.query_pre_attn_scalar as f32).sqrt();
    let attention = (0..f.layers)
        .map(|l| LayerAttention {
            // One shape for every layer, which is what this row was
            // already saying by having no per-layer count.
            kv_heads: f.kv_heads,
            head_dim,
            window: s.window_at(l),
            // Every layer owns its pages. KV sharing is gemma-4's.
            kv_source: l,
            sm_scale,
            rope_theta: s.rope_theta_at(l),
            // Full rotation at the head dim; gemma-3 has no partial
            // rotary factor.
            rotary_dim: 0,
        })
        .collect();
    Deployment {
        layers: f.layers,
        norm_eps,
        shape: Geometry {
            hidden: f.hidden,
            q_heads: f.q_heads,
            kv_heads: f.kv_heads,
            head_dim: f.head_dim,
            head_dim_kernel: family::round_up_attn_head_dim(f.head_dim),
            intermediate: f.intermediate,
            // Dense: gemma-3 publishes no expert block, and the field means
            // ONE expert's width rather than the stack's.
            moe_intermediate: 0,
            experts_per_token: 0,
            shared_intermediate: 0,
            vocab: f.vocab,
        },
        attention,
        kv: KvStyle::Paged,
        recurrent: None,
        prefill: PrefillStyle::Planned,
        attn_output: AttnOutput::DriverPinned,
        // Gemma 3 dropped gemma-2's caps: `final_logit_softcapping` is
        // null in every published gemma-3 config, and null means NO cap.
        logit_softcap: 0.0,
        // Per-layer embeddings are gemma-3n's and gemma-4's.
        ple_dim: 0,
        // The four-norm block is still a PRE placement as far as a
        // driver's staging is concerned: the projections read the normed
        // value. The extra norms sit on the block's output, which is the
        // traced text's business.
        norm: NormPlacement::Pre,
        // `(1 + w)`, the same as gemma-2 — and the forward above says so
        // independently with `NormVariant::Gemma`.
        norm_unit_offset: true,
        // gemma-3 carries the per-head q/k norm and NO V norm: the two
        // are separate facts, and this row is where that is said.
        v_norm: false,
        k_eq_v: false,
        mlp_gate: crate::deployment::MlpGate::GeluTanh,
        scales: std::collections::BTreeMap::new(),
        // Filled by the ROW, not by the shape: a family label and a
        // published context ceiling are facts about a checkpoint, and a
        // projection only sees geometry.
        advertised: Advertised::default(),
        rope_scaling: None,
        towers: Default::default(),
    }
}

/// The METAL binding facts for a gemma-3 row.
///
/// The family's, with the seven fields gemma-3 answers differently
/// written over them — and it is a `..base` struct update rather than a
/// fresh literal for the same reason
/// `LlamaLikeMetalFacts::gemma_like()` is one: the fields that are NOT
/// listed here are the ones gemma-3 shares with qwen3, and restating
/// them would be a second place for them to drift. What is listed is
/// exactly gemma-3's difference, which is the diff a reader wants.
///
/// Six of the seven are things `driver-metal`'s deleted
/// `facts_from_with` derived by asking the TENSORS — it keyed
/// `embed_scale` off the presence of a `pre_feedforward_layernorm`, and
/// took the softmax scale from the head dim because no tensor states
/// `query_pre_attn_scalar`. Gemma-3-27B publishes 168 with 128-wide
/// heads, so that derivation was wrong for exactly the row whose scale
/// is not the head dim's, and wrong by a factor of `sqrt(168/128)` —
/// which is a sharper softmax, not a crash.
///
/// # Errors
///
/// None. Whether this build has a Metal text at all is [`trace`]'s
/// question, and it is answered by the backend rather than by a value
/// here.
#[cfg(feature = "forward")]
#[must_use]
pub fn metal_facts(
    f: &LlamaLikeFacts,
    s: &Schedule,
    norm_eps: f32,
    load: Deployed<'_>,
    bind: &MetalBinding,
) -> crate::shared::llama_like::forward::facts::LlamaLikeMetalFacts {
    // The family's answer for every field neither the schedule nor the
    // generation moves — the binding's six, this build's tile, and the
    // structural zeros. The window is passed as `-1` and the base as the
    // GLOBAL one because both are overridden below; passing the schedule's
    // local values here would put a number in the base that the override
    // then contradicts, and a reader would have to compare two literals to
    // learn which won.
    let base = family::metal_facts(
        f,
        family::MetalRow {
            rope_theta: s.rope_theta_global,
            norm_eps,
            window: -1,
            rope_rescaled: false,
        },
        load,
        bind,
    );
    // `sqrt(hidden)`, computed in f64 and narrowed. 2560's root is not
    // representable in either width, and the f32 rounding of the f64
    // answer is the one every reference implementation lands on — a
    // difference that would otherwise appear as a per-token drift in
    // the last mantissa bits of every embedding, which is exactly the
    // kind of divergence a parity harness cannot attribute.
    #[allow(clippy::cast_possible_truncation)]
    let embed_scale = f64::from(f.hidden).sqrt() as f32;
    crate::shared::llama_like::forward::facts::LlamaLikeMetalFacts {
        // The 5:1 schedule, expanded to the per-layer list the text reads
        // through `window_left_at`. Not one entry: this is the row whose
        // layers DISAGREE, and the accessor's "last entry covers the tail"
        // rule would broadcast whichever width came last to the rest of the
        // stack. That is the `deployment_of` bug restated — every sixth
        // layer attending 1024 tokens instead of its whole context, or five
        // in six attending a context they were never trained on — and both
        // read as a working model.
        window_left: (0..f.layers).map(|l| s.window_at(l)).collect(),
        // The SLIDING base, which turns the field above into a two-base
        // schedule: `rope_theta_at` reads this one for any layer with a
        // window and `rope_theta` for the rest, which is precisely
        // gemma-3's rule. Two orders of magnitude apart — 10 000 against
        // 1 000 000 — so broadcasting either is wrong for five layers in
        // six or for one in six, and there is no choice that is right.
        rope_theta_sliding: s.rope_theta_local,
        // `gelu_tanh(gate) * up`. Gemma's activation is the TANH
        // approximation of gelu and not the erf one; the two agree to about
        // 2% at the origin and diverge from there, which is a model that
        // runs and is wrong rather than one that faults.
        activation: crate::shared::llama_like::forward::facts::Activation::Geglu,
        // Applied at the embedding gather. Gemma's embeddings are stored
        // small and scaled up on read; a text that skipped it starts the
        // stack an order of magnitude below where every downstream norm
        // and softmax was trained, and the output is fluent and wrong.
        embed_scale,
        // The softmax TEMPERATURE, `query_pre_attn_scalar ** -0.5` —
        // HF's own words for it, and the same number `deployment` puts
        // on every `LayerAttention::sm_scale` two functions up, so the
        // two backends divide by one value. Stated rather than left zero
        // (which the text reads as "derive `1/sqrt(head_dim)`") because
        // gemma-3-27B states 168 against 128-wide heads and is the one
        // published row where the derivation and the config part ways.
        //
        // It also DIVERGES from what `driver-metal`'s deleted
        // `facts_from_with` would have produced, deliberately. That
        // function set `attn_scale: 1.0` whenever a checkpoint had
        // sandwich norms AND a per-head `q_norm`, on the reasoning that
        // a deployment which norms its heads has already paid the
        // temperature. That reasoning was written for gemma-4 and is
        // false for gemma-3, which norms its heads AND scales — and it
        // never showed, because `LLAMA_LIKE` did not list `gemma3`, so
        // the branch only ever fired for the family it was written for.
        // A rule inferred from two tensors, holding for the one model it
        // was measured on, is the whole failure mode this catalog
        // replaces.
        attn_scale: 1.0 / (s.query_pre_attn_scalar as f32).sqrt(),
        ..base
    }
}

/// Trace this row's text for one fire class, on the backend that asked.
///
/// Gemma-3 SERVES Metal, and that is a claim worth stating plainly: its
/// CUDA text is the family's `llama_like_cuda`, so the family's
/// `llama_like_metal` is the same model — the Metal text already carries
/// the sandwich norms, the `(1 + w)` fold through `NormVariant::Gemma`,
/// the scaled embedding gather, the two rope bases and the per-layer
/// window, which are the whole of gemma-3's difference. The generation
/// that must REFUSE Metal is the one whose CUDA text is a different
/// function — `gemma2_cuda`, `gemma3n_cuda`, `gpt_oss_cuda` — because
/// there is then no Metal text of that model to reach, and reaching the
/// llama-like one would trace a different model under this row's name.
///
/// The string table this replaces got exactly this wrong in both
/// directions: `driver-metal`'s `LLAMA_LIKE` listed `gemma4`, which the
/// load path refused on other grounds, and did NOT list `gemma3`, whose
/// text it models. A row answering for itself cannot disagree with a
/// list, because there is no list.
///
/// # Errors
///
/// The family's — [`Refusal::Unsupported`](crate::deployment::Refusal)
/// carrying [`family::NO_METAL_SHARD`] for a sharded Metal load.
#[cfg(feature = "forward")]
pub fn trace(
    f: &LlamaLikeFacts,
    s: &Schedule,
    norm_eps: f32,
    class: model_compiler::trace::FireClass,
    load: Deployed<'_>,
) -> Result<model_compiler::trace::ForwardPlan, crate::deployment::Refusal> {
    // The family's scalars, which the CUDA arm carries unread — its text
    // states no epsilon and no base at all — and which the Metal arm
    // takes only for the fields gemma-3 does NOT override. The two arms
    // build it from the same expression rather than from two literals,
    // so a base that moved on the row cannot reach one text and not the
    // other.
    let row = family::MetalRow {
        rope_theta: s.rope_theta_global,
        norm_eps,
        window: -1,
        rope_rescaled: false,
    };
    match load.backend {
        Backend::Cuda => family::trace(f, row, class, load),
        Backend::Metal(bind) => {
            if load.tp_size > 1 {
                return Err(crate::deployment::Refusal::Unsupported(family::NO_METAL_SHARD));
            }
            Ok(crate::shared::llama_like::forward::llama_like_metal(
                f,
                &metal_facts(f, s, norm_eps, load, bind),
                class,
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::Presence;
    use model_compiler::facts::{NormPlacement as SpecNorm, QkNorm};
    use model_compiler::trace::{NormVariant, RopeKind};

    fn shape() -> LlamaLikeFacts {
        // gemma-3-4b's geometry, which is the middle of the generation
        // in every axis these tests read.
        LlamaLikeFacts {
            hidden: 2560,
            layers: 34,
            q_heads: 8,
            kv_heads: 4,
            head_dim: 256,
            n_experts: 0,
            experts_per_token: 0,
            shared_intermediate: 0,
            moe_intermediate: 0,
            intermediate: 10_240,
            vocab: 262_208,
            rope: RopeKind::Standard,
            norm_variant: NormVariant::Gemma,
            norm_placement: SpecNorm::Sandwich,
            qk_norm: QkNorm::PerHead,
            fused_qkv: true,
            tied_embeddings: true,
            qkv_bias: false,
        }
    }

    /// Gemma-3's epsilon, the one every published config of the
    /// generation states — `rms_norm_eps: 1e-6`. Held here as one
    /// number because none of these tests varies it: it rides through
    /// the projections untouched, and the row's own value is checked
    /// against the published config by `tests/catalog_differential.rs`,
    /// which is where a transcribed number belongs.
    const NORM_EPS: f32 = 1e-6;

    fn schedule() -> Schedule {
        Schedule {
            sliding_window: 1024,
            full_attn_interval: 6,
            rope_theta_local: 10_000.0,
            rope_theta_global: 1_000_000.0,
            query_pre_attn_scalar: 256,
        }
    }

    /// Five sliding layers, then one that sees everything — the pattern
    /// the config states as `sliding_window_pattern: 6`, and the one the
    /// generic derivation flattened into "1024 everywhere".
    #[test]
    fn five_local_layers_then_one_global() {
        let d = deployment(&shape(), &schedule(), NORM_EPS);
        let windows: Vec<i32> = d.attention.iter().map(|a| a.window).collect();
        assert_eq!(&windows[..6], &[1024, 1024, 1024, 1024, 1024, -1]);
        assert_eq!(windows.iter().filter(|&&w| w == -1).count(), 34 / 6);
        for (l, w) in windows.iter().enumerate() {
            assert_eq!(*w == -1, (l + 1) % 6 == 0, "layer {l}");
        }
    }

    /// The two rope bases land on the layers that rotate at them. One
    /// theta broadcast to the stack is wrong for five layers in six or
    /// for one in six, depending which one is chosen; there is no
    /// choice that is right.
    #[test]
    fn each_layer_rotates_at_its_own_base() {
        let d = deployment(&shape(), &schedule(), NORM_EPS);
        for (l, a) in d.attention.iter().enumerate() {
            let expected = if (l + 1) % 6 == 0 { 1_000_000.0 } else { 10_000.0 };
            assert_eq!(a.rope_theta, expected, "layer {l}");
        }
        assert_ne!(d.attention[0].rope_theta, d.attention[5].rope_theta);
    }

    /// The scale is `1/sqrt(query_pre_attn_scalar)` and not
    /// `1/sqrt(head_dim)`. They agree for four of the five rows, which
    /// is exactly why the difference is worth a test: gemma-3-27b states
    /// 168 against 128-wide heads, so a head-dim scale is wrong by 15%
    /// on the one row nobody would notice it on.
    #[test]
    fn the_softmax_scale_is_the_configs_scalar_and_not_the_head_dim() {
        let d = deployment(&shape(), &schedule(), NORM_EPS);
        assert!((d.attention[0].sm_scale - 1.0 / 16.0).abs() < 1e-6, "1/sqrt(256)");

        let mut wide = shape();
        wide.head_dim = 128;
        let s = Schedule { query_pre_attn_scalar: 168, ..schedule() };
        let d = deployment(&wide, &s, 1e-6);
        assert!((d.attention[0].sm_scale - 1.0 / 168f32.sqrt()).abs() < 1e-6);
        assert!(
            (d.attention[0].sm_scale - 1.0 / 128f32.sqrt()).abs() > 1e-3,
            "the head-dim scale is a different number, which is the point",
        );
    }

    /// All four norms are expected, where the shared projection would
    /// have expected the ABSENCE of two of them and refused every real
    /// gemma-3.
    #[test]
    fn the_four_norm_block_is_expected_rather_than_forbidden() {
        let m = manifest(&shape());
        for name in [
            "layer.{}.input_layernorm",
            "layer.{}.post_attention_layernorm",
            "layer.{}.pre_feedforward_layernorm",
            "layer.{}.post_feedforward_layernorm",
        ] {
            let spec = m.tensors.iter().find(|t| t.name == name).expect("stated");
            assert_eq!(spec.presence, Presence::Required, "{name}");
            assert_eq!(spec.extents, vec![2560]);
        }
        // And the family's own version of those two rows is gone, not
        // shadowed: a manifest with the same name twice would check
        // both, and one of them says Absent.
        for name in ["layer.{}.input_layernorm", "layer.{}.post_attention_layernorm"] {
            assert_eq!(m.tensors.iter().filter(|t| t.name == name).count(), 1, "{name}");
        }
    }

    /// Q and K are both normed, per head, which is `[head_dim]` on each.
    #[test]
    fn both_query_and_key_norms_are_expected_per_head() {
        let m = manifest(&shape());
        for name in ["layer.{}.self_attn.q_norm", "layer.{}.self_attn.k_norm"] {
            let spec = m.tensors.iter().find(|t| t.name == name).expect("stated");
            assert_eq!(spec.presence, Presence::Required, "{name}");
            assert_eq!(spec.extents, vec![256], "{name}");
        }
    }

    /// The family's own rows survive the fix-up — this is a wrapper, not
    /// a fourth family, and the projections that were already right are
    /// not restated.
    #[test]
    fn the_familys_rows_are_kept() {
        let m = manifest(&shape());
        let ext = |n: &str| m.tensors.iter().find(|t| t.name == n).expect("stated").extents.clone();
        assert_eq!(ext("embed_tokens"), vec![262_208, 2560]);
        assert_eq!(ext("layer.{}.self_attn.q_proj"), vec![2048, 2560]);
        assert_eq!(ext("layer.{}.self_attn.k_proj"), vec![1024, 2560]);
        assert_eq!(ext("layer.{}.mlp.gate_proj"), vec![10_240, 2560]);
        let head = m.tensors.iter().find(|t| t.name == "lm_head").expect("stated");
        assert_eq!(head.presence, Presence::Absent, "gemma-3 ties");
    }

    /// The launch geometry is the row's own numbers, and the rest of the
    /// deployment is stated rather than defaulted.
    #[test]
    fn the_deployment_states_what_the_old_vtable_defaulted() {
        let d = deployment(&shape(), &schedule(), NORM_EPS);
        assert_eq!(d.layers, 34);
        assert_eq!(d.shape.hidden, 2560);
        assert_eq!(d.shape.q_heads, 8);
        assert_eq!(d.shape.kv_heads, 4);
        assert_eq!(d.shape.head_dim, 256);
        assert_eq!(d.shape.head_dim_kernel, 256);
        assert_eq!(d.shape.gqa_group(), 2);
        assert_eq!(d.kv, KvStyle::Paged);
        assert!(d.recurrent.is_none());
        assert_eq!(d.prefill, PrefillStyle::Planned);
        assert_eq!(d.attn_output, AttnOutput::DriverPinned);
        assert_eq!(d.logit_softcap, 0.0, "gemma-3 dropped gemma-2's caps");
        assert_eq!(d.ple_dim, 0);
        assert_eq!(d.norm, NormPlacement::Pre);
        assert!(
            d.norm_unit_offset,
            "gemma-3 stores the gain as an offset from one; the forward says \
             the same with `NormVariant::Gemma`"
        );
        assert!(d.scales.is_empty());
        for (l, a) in d.attention.iter().enumerate() {
            assert_eq!(a.kv_source, l as u32);
            assert_eq!(a.rotary_dim, 0);
            assert_eq!(a.head_dim, 256);
        }
    }

    /// The schedule answers for any layer index without a table to run
    /// past, which is what a rule buys over the `Vec` this replaces.
    #[test]
    fn the_schedule_is_a_rule_and_not_a_table() {
        let s = schedule();
        assert!(!s.is_full_attn(0));
        assert!(s.is_full_attn(5));
        // The 27B is 62 layers, so 61 is its last, and the rule answers
        // there rather than running off a table's end. The answer is
        // SLIDING: `(61 + 1) % 6` is 2, and the 27B's last full layer is
        // 59. A period that does not divide the depth simply leaves the
        // tail sliding, which is what google published.
        assert!(!s.is_full_attn(61), "the 27B's last layer is sliding");
        assert!(s.is_full_attn(59), "the 27B's last FULL layer");
        assert_eq!(s.window_at(5), -1);
        assert_eq!(s.rope_theta_at(5), 1_000_000.0);
        assert_eq!(s.rope_theta_at(4), 10_000.0);
    }

    /// A binding to exercise the Metal projection with. The values are
    /// `mlx-community`'s usual 4-bit publication of a gemma-3.
    #[cfg(feature = "forward")]
    fn binding() -> MetalBinding {
        MetalBinding {
            quant_group: 64,
            quant_bits: 4,
            moe_mxfp4: false,
            fuse_residual_gemv: true,
            paged_multi_batch: true,
            qmm_multi_batch: true,
        }
    }

    /// The 5:1 schedule reaches the METAL text per layer, both the
    /// window and the base that goes with it.
    ///
    /// This is the bug the family's single-window `MetalRow` cannot
    /// hold and the reason gemma-3 has a projection of its own: a row
    /// that stated one width here would attend 1024 tokens on the layer
    /// trained to see everything, or the whole context on five layers
    /// that were not — and both read as a working model.
    #[cfg(feature = "forward")]
    #[test]
    fn the_metal_text_gets_the_schedule_per_layer_and_not_broadcast() {
        let f = shape();
        let b = binding();
        let m = metal_facts(&f, &schedule(), NORM_EPS, Deployed::metal(&b), &b);

        assert_eq!(m.window_left.len(), f.layers as usize, "one entry per layer");
        for l in 0..f.layers {
            let full = (l + 1) % 6 == 0;
            assert_eq!(m.window_left_at(l), if full { -1 } else { 1024 }, "layer {l}");
            assert_eq!(m.is_full_attention(l), full, "layer {l}");
            // And the base the accessor reads for that layer is the one
            // the schedule states for it — the pairing is the point,
            // because `rope_theta_at` keys the base on the WINDOW.
            let want = if full { 1_000_000.0 } else { 10_000.0 };
            assert_eq!(m.rope_theta_at(l), want, "layer {l}");
        }
        assert_eq!(m.rope_theta, 1_000_000.0);
        assert_eq!(m.rope_theta_sliding, 10_000.0);
    }

    /// Gemma's four differences from a llama, stated on the Metal facts.
    ///
    /// Each one is a thing `driver-metal`'s deleted `facts_from_with`
    /// derived by asking the tensors — it keyed `embed_scale` off the
    /// presence of a `pre_feedforward_layernorm` — and each is now the
    /// row's answer. The scale is the sharpest: no tensor states
    /// `query_pre_attn_scalar`, so the derivation used the head dim and
    /// was wrong for the one published row where they differ.
    #[cfg(feature = "forward")]
    #[test]
    fn the_gemma_differences_are_the_rows_and_not_a_tensor_probe() {
        use crate::shared::llama_like::forward::facts::Activation;
        let f = shape();
        let b = binding();
        let m = metal_facts(&f, &schedule(), NORM_EPS, Deployed::metal(&b), &b);

        assert_eq!(m.activation, Activation::Geglu, "the TANH gelu, not silu");
        assert!(
            (m.embed_scale - 50.596_44).abs() < 1e-4,
            "sqrt(2560), the 4B's hidden, and not 1.0: got {}",
            m.embed_scale
        );
        assert_eq!(m.attn_scale, 1.0 / 16.0, "1/sqrt(256)");
        assert_eq!(m.rms_eps, NORM_EPS);

        // The 27B is the row the head dim would have got wrong: 168
        // against 128-wide heads, which is a factor of sqrt(168/128) —
        // a sharper softmax, not a crash.
        let big = Schedule { query_pre_attn_scalar: 168, ..schedule() };
        let m27 = metal_facts(&f, &big, NORM_EPS, Deployed::metal(&b), &b);
        assert_eq!(m27.attn_scale, 1.0 / 168f32.sqrt());
        assert_ne!(m27.attn_scale, m.attn_scale, "the head dim's answer is a different number");
    }

    /// The binding half is the family's, unchanged — gemma-3 overrides
    /// what gemma-3 IS and nothing about how its bytes arrived.
    #[cfg(feature = "forward")]
    #[test]
    fn the_binding_half_is_still_the_familys() {
        let f = shape();
        let b = MetalBinding { quant_group: 128, quant_bits: 8, ..binding() };
        let mine = metal_facts(&f, &schedule(), NORM_EPS, Deployed::metal(&b), &b);
        let family_answer = family::metal_facts(
            &f,
            family::MetalRow {
                rope_theta: 1_000_000.0,
                norm_eps: NORM_EPS,
                window: -1,
                rope_rescaled: false,
            },
            Deployed::metal(&b),
            &b,
        );
        assert_eq!(mine.affine_bits, family_answer.affine_bits);
        assert_eq!(mine.proj_repr, family_answer.proj_repr);
        assert_eq!(mine.qmm_tile, family_answer.qmm_tile);
        assert_eq!(mine.fuse_residual_gemv, family_answer.fuse_residual_gemv);
        assert_eq!(mine.paged_multi_batch, family_answer.paged_multi_batch);
        assert_eq!(mine.qmm_multi_batch, family_answer.qmm_multi_batch);
        assert_eq!(mine.moe_repr, family_answer.moe_repr);
        // And the structural zeros too, because gemma-3 is not gemma-4:
        // one attention shape, no per-layer embeddings, no shared pages.
        assert_eq!(mine.global_head_dim, 0);
        assert_eq!(mine.per_layer_emb_dim, 0);
        assert_eq!(mine.kv_shared_layers, 0);
        assert!(!mine.per_layer_scalar, "layer scalars are gemma-3n's");
        assert_eq!(mine.logit_softcap, 0.0, "gemma-3 dropped gemma-2's caps");
    }

    /// gemma-3 SERVES Metal, and says so on both fire classes.
    ///
    /// The claim the deleted string table got backwards in both
    /// directions: `LLAMA_LIKE` listed `gemma4`, which the load path
    /// refused, and did not list `gemma3`, whose text it models.
    #[cfg(feature = "forward")]
    #[test]
    fn gemma_3_answers_a_metal_load_with_a_metal_text() {
        use model_compiler::trace::FireClass;
        let f = shape();
        let b = binding();
        for class in [FireClass::Prefill, FireClass::Decode] {
            let metal = trace(&f, &schedule(), NORM_EPS, class, Deployed::metal(&b))
                .expect("gemma-3's forward IS llama-like");
            assert!(metal.family.starts_with("llama_like.metal."), "{}", metal.family);
            let cuda = trace(&f, &schedule(), NORM_EPS, class, Deployed::single())
                .expect("and the CUDA text is the one it always was");
            assert!(cuda.family.starts_with("llama_like.cuda."), "{}", cuda.family);
        }
    }

    /// A sharded Metal load is refused with the FAMILY's sentence.
    ///
    /// One sentence rather than a gemma-flavoured copy of it, because
    /// the ground is the family's: `LlamaLikeMetalFacts` carries no
    /// shard width, so the same trace would state the whole model's
    /// projections against one rank's slice of the weights.
    #[cfg(feature = "forward")]
    #[test]
    fn a_sharded_metal_load_is_refused_with_the_familys_sentence() {
        use crate::deployment::Refusal;
        use model_compiler::trace::FireClass;
        let b = binding();
        let sharded = Deployed { backend: Backend::Metal(&b), tp_size: 2, layer_scalars: &[] };
        let err = trace(&shape(), &schedule(), NORM_EPS, FireClass::Decode, sharded)
            .expect_err("two ranks and no shard vocabulary");
        assert_eq!(err, Refusal::Unsupported(family::NO_METAL_SHARD));
    }
}

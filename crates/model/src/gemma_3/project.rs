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

use crate::catalog::Deployed;
use crate::deployment::{
    Advertised,
    AttnOutput, Deployment, Geometry, KvStyle, LayerAttention, NormPlacement, PrefillStyle,
};
use crate::families::llama_like::project as family;
use crate::families::llama_like::spec::LlamaLikeFacts;
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
/// — `input_layernorm` when `Pre`, `post_attention_layernorm` when
/// `Post` — which is exactly right for the two placements that existed
/// when it was written and answers `Absent` to both under `Sandwich`. A
/// gemma-3 checkpoint ships all four norms, so that manifest would
/// refuse every real gemma-3 twice over. The four are restated here,
/// where the generation that has them lives, rather than by teaching a
/// shared projection about a third case it has no other reader for.
#[must_use]
pub fn manifest(f: &LlamaLikeFacts) -> Manifest {
    let hidden = u64::from(f.hidden);
    let head_dim = u64::from(f.head_dim);
    let base = family::manifest(f);

    let mut out = Manifest::new(base.layers);
    for spec in base.tensors {
        // The two the family states as placement; gemma has both, plus
        // the feedforward pair below.
        if spec.name == "layer.{}.input_layernorm"
            || spec.name == "layer.{}.post_attention_layernorm"
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
        scales: std::collections::BTreeMap::new(),
        // Filled by the ROW, not by the shape: a family label and a
        // published context ceiling are facts about a checkpoint, and a
        // projection only sees geometry.
        advertised: Advertised::default(),
        rope_scaling: None,
        towers: Default::default(),
    }
}

/// Trace this row's CUDA text for one fire class — the family's tracer,
/// on the family's shape, because gemma-3's forward IS llama-like.
#[cfg(feature = "forward")]
#[must_use]
pub fn trace(
    f: &LlamaLikeFacts,
    class: model_compiler::trace::FireClass,
    load: Deployed<'_>,
) -> model_compiler::trace::ForwardPlan {
    family::trace(f, class, load)
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
            moe_intermediate: 0,
            shared_intermediate: 0,
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
        let d = deployment(&shape(), &schedule());
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
        let d = deployment(&shape(), &schedule());
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
        let d = deployment(&shape(), &schedule());
        assert!((d.attention[0].sm_scale - 1.0 / 16.0).abs() < 1e-6, "1/sqrt(256)");

        let mut wide = shape();
        wide.head_dim = 128;
        let s = Schedule { query_pre_attn_scalar: 168, ..schedule() };
        let d = deployment(&wide, &s);
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
        let d = deployment(&shape(), &schedule());
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
        assert!(s.is_full_attn(61), "the 27B's last layer");
        assert_eq!(s.window_at(5), -1);
        assert_eq!(s.rope_theta_at(5), 1_000_000.0);
        assert_eq!(s.rope_theta_at(4), 10_000.0);
    }
}

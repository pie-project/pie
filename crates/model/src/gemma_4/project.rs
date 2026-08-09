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
        .with(TensorSpec::required("embed_tokens", [vocab, hidden]))
        .with(TensorSpec::required("norm", [hidden]))
        // A tie is an ABSENCE: gemma-4 ships no `lm_head`, and the MLX
        // author renames `embed_tokens` into the head's place rather
        // than binding a second table.
        .either(!f.tied_embeddings, "lm_head", [vocab, hidden])
        // THE PER-LAYER EMBEDDING TABLE, whose second axis multiplies
        // the layer count — the one tensor shape that says "gemma-3n or
        // gemma-4" and nothing else. The A4B mixture drops it
        // (`hidden_size_per_layer_input: 0`), so it is conditional here
        // and its absence is stated rather than merely unlisted.
        .with_if(has_ple, TensorSpec::required("embed_tokens_per_layer", [vocab, layers * ple]))
        .with_if(
            has_ple,
            TensorSpec::required("per_layer_model_projection", [layers * ple, hidden]),
        )
        .with_if(has_ple, TensorSpec::required("per_layer_projection_norm", [ple]))
        .with_if(!has_ple, TensorSpec::absent("embed_tokens_per_layer"))
        .with(TensorSpec::required("layer.{}.self_attn.q_proj", [q, hidden]))
        .with(TensorSpec::required("layer.{}.self_attn.k_proj", [kv, hidden]))
        .with(TensorSpec::required("layer.{}.self_attn.v_proj", [kv, hidden]))
        .with(TensorSpec::required("layer.{}.self_attn.o_proj", [hidden, q]))
        // Per-HEAD q/k norms: the extent is one head's width, not the
        // projection's, and on layer 0 that is the sliding head dim.
        .with(TensorSpec::required("layer.{}.self_attn.q_norm", [head_dim]))
        .with(TensorSpec::required("layer.{}.self_attn.k_norm", [head_dim]))
        // gemma-3n norms V as well. This generation does not, and the
        // absence is what keeps a gemma-3n from matching here.
        .with(TensorSpec::absent("layer.{}.self_attn.v_norm"))
        .with(TensorSpec::absent("layer.{}.altup.modality_router"))
        .with(TensorSpec::absent("layer.{}.laurel.linear_left"))
        // The four gemma norms.
        .with(TensorSpec::required("layer.{}.input_layernorm", [hidden]))
        .with(TensorSpec::required("layer.{}.post_attention_layernorm", [hidden]))
        .with(TensorSpec::required("layer.{}.pre_feedforward_layernorm", [hidden]))
        .with(TensorSpec::required("layer.{}.post_feedforward_layernorm", [hidden]))
        .with(TensorSpec::required("layer.{}.mlp.gate_proj", [inter, hidden]))
        .with(TensorSpec::required("layer.{}.mlp.up_proj", [inter, hidden]))
        .with(TensorSpec::required("layer.{}.mlp.down_proj", [hidden, inter]))
        // The PLE epilogue's own weights, present exactly when the table
        // they read is.
        .with_if(has_ple, TensorSpec::required("layer.{}.per_layer_input_gate", [ple, hidden]))
        .with_if(has_ple, TensorSpec::required("layer.{}.per_layer_projection", [hidden, ple]))
        .with_if(has_ple, TensorSpec::required("layer.{}.post_per_layer_input_norm", [hidden]))
        // The router's `[hidden]` scale, which `contract::author_gemma4`
        // folds `1/sqrt(hidden)` into at load — it reads the width off
        // this tensor, so this row states the width that fold assumes.
        // A dense gemma-4 forbids it, which is how the A4B is told from
        // an E4B whose other extents somehow agreed.
        .either(mixture.is_some(), "layer.{}.router.scale", [hidden])
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
pub fn deployment(
    f: &Gemma4Facts,
    mixture: Option<Gemma4Mixture>,
    sliding_window: i32,
    norm_eps: f32,
    load: Deployed<'_>,
) -> Deployment {
    let attention = (0..f.layers).map(|l| layer_attention(f, sliding_window, l)).collect();
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
            vocab: f.vocab,
        },
        attention,
        kv: KvStyle::Paged,
        // No recurrent slabs: every gemma-4 layer attends.
        recurrent: None,
        prefill: PrefillStyle::Planless,
        attn_output: AttnOutput::StatedArgs,
        logit_softcap: f.logit_softcap,
        ple_dim: i32::try_from(f.ple_dim).unwrap_or(0),
        norm: NormPlacement::Pre,
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
        rope_theta: if full { ROPE_THETA_GLOBAL } else { ROPE_THETA_LOCAL },
        // The full layers rotate PARTIALLY (128 of 512, from
        // `partial_rotary_factor: 0.25`); the sliding ones rotate fully.
        // A full rotation is spelled as the head dim rather than as the
        // `0` the field documents as meaning the same thing, because the
        // driver's own derivation emits `max(2, 2*int(0.5*f*d))` for
        // every layer and a deployment that differed from it by a
        // sentinel would compare unequal to one that did not.
        rotary_dim: if full { f.global_rotary_dim } else { f.head_dim },
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

/// Trace this row's CUDA text for one fire class.
///
/// The backend facts are built HERE rather than held on the row, and the
/// three booleans are why: they say what the LOADER did — whether it
/// fused the QKV bank, whether it fused gate‖up, whether the pages are
/// native bf16 — which is a property of a load and not of a model. The
/// per-layer window list is the row's, and it is not empty: gemma-4 is
/// the generation where an empty list would have the trace say "attends
/// everything" while the plan applied a 512-token window.
#[cfg(feature = "forward")]
#[must_use]
pub fn trace(
    f: &Gemma4Facts,
    sliding_window: i32,
    class: model_compiler::trace::FireClass,
) -> model_compiler::trace::ForwardPlan {
    let cuda = super::forward::facts::Gemma4CudaFacts {
        fused_qkv: true,
        gate_up_fused: true,
        kv_native_bf16: true,
        window_left: (0..f.layers)
            .map(|l| if f.is_full_attn(l) { -1 } else { sliding_window.max(0) })
            .collect(),
    };
    super::forward::gemma4_cuda(f, &cuda, class)
}

#[cfg(test)]
mod tests {
    use super::{
        Advertised, AttnOutput, Deployed, Gemma4Facts, Gemma4Mixture, KvStyle, NormPlacement,
        PrefillStyle, ROPE_THETA_GLOBAL, ROPE_THETA_LOCAL, deployment, manifest,
    };
    use crate::manifest::{Observed, Presence};

    const E4B_WINDOW: i32 = 512;
    const NORM_EPS: f32 = 1e-6;

    fn e4b() -> crate::deployment::Deployment {
        deployment(&Gemma4Facts::gemma_4_e4b(), None, E4B_WINDOW, NORM_EPS, Deployed::single())
    }

    fn spec(m: &crate::manifest::Manifest, name: &str) -> crate::manifest::TensorSpec {
        m.tensors.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("{name} is stated")).clone()
    }

    /// The manifest names layer 0's widths, and layer 0 is a SLIDING
    /// layer — 8 heads of 256 into 2560, not 8 of 512. A manifest that
    /// used the full layers' width would fail to identify every gemma-4
    /// ever published, which is the failure mode worth a test.
    #[test]
    fn the_projection_widths_are_the_sliding_layers_because_layer_zero_is_one() {
        let f = Gemma4Facts::gemma_4_e4b();
        assert!(!f.is_full_attn(0), "layer 0 is full, and the manifest below assumes otherwise");
        let m = manifest(&f, None);
        assert_eq!(m.layers, 42);
        assert_eq!(spec(&m, "layer.{}.self_attn.q_proj").extents, vec![2048, 2560]);
        assert_eq!(spec(&m, "layer.{}.self_attn.k_proj").extents, vec![512, 2560]);
        assert_eq!(spec(&m, "layer.{}.self_attn.v_proj").extents, vec![512, 2560]);
        assert_eq!(spec(&m, "layer.{}.self_attn.o_proj").extents, vec![2560, 2048]);
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
        assert_eq!(spec(&m, "lm_head").presence, Presence::Absent, "a tie is an absence");
        for norm in [
            "layer.{}.input_layernorm",
            "layer.{}.post_attention_layernorm",
            "layer.{}.pre_feedforward_layernorm",
            "layer.{}.post_feedforward_layernorm",
        ] {
            assert_eq!(spec(&m, norm).extents, vec![2560], "{norm}");
        }
        assert_eq!(spec(&m, "layer.{}.mlp.gate_proj").extents, vec![10_240, 2560]);
        assert_eq!(spec(&m, "layer.{}.mlp.up_proj").extents, vec![10_240, 2560]);
        assert_eq!(spec(&m, "layer.{}.mlp.down_proj").extents, vec![2560, 10_240]);
    }

    /// The PLE table's second axis is `layers * ple_dim` — the one
    /// tensor extent in this catalog that multiplies a layer count, and
    /// the reason a gemma-4 cannot be mistaken for a llama-like stack.
    #[test]
    fn the_per_layer_embedding_table_multiplies_the_layer_count() {
        let m = manifest(&Gemma4Facts::gemma_4_e4b(), None);
        assert_eq!(spec(&m, "embed_tokens_per_layer").extents, vec![262_144, 42 * 256]);
        assert_eq!(spec(&m, "per_layer_model_projection").extents, vec![42 * 256, 2560]);
        assert_eq!(spec(&m, "per_layer_projection_norm").extents, vec![256]);
        assert_eq!(spec(&m, "layer.{}.per_layer_input_gate").extents, vec![256, 2560]);
        assert_eq!(spec(&m, "layer.{}.per_layer_projection").extents, vec![2560, 256]);
        assert_eq!(spec(&m, "layer.{}.post_per_layer_input_norm").extents, vec![2560]);
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
        assert_eq!(spec(&dense, "layer.{}.router.scale").presence, Presence::Absent);
        assert_eq!(spec(&routed, "layer.{}.router.scale").presence, Presence::Required);
        assert_eq!(spec(&routed, "layer.{}.router.scale").extents, vec![2816]);
        assert_eq!(spec(&dense, "embed_tokens_per_layer").presence, Presence::Required);
        assert_eq!(spec(&routed, "embed_tokens_per_layer").presence, Presence::Absent);
        assert!(
            !routed.tensors.iter().any(|t| t.name == "layer.{}.per_layer_input_gate"),
            "a stack with no per-layer table cannot gate one into its residual"
        );
    }

    /// An untied checkpoint would be required to ship the head, and the
    /// row says so through the same `either` that states the tie. No
    /// published gemma-4 unties, so this exercises the branch the
    /// fixtures do not.
    #[test]
    fn an_untied_head_is_required_rather_than_forbidden() {
        let f = Gemma4Facts { tied_embeddings: false, ..Gemma4Facts::gemma_4_e4b() };
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
            assert!(m.check(&implied).is_ok(), "{name}: manifest does not describe itself");
        }
    }

    /// The stack's own count, and the geometry a launch path reads.
    #[test]
    fn the_geometry_is_the_rows_own_numbers() {
        let d = e4b();
        assert_eq!(d.layers, 42);
        assert_eq!(d.attention.len(), 42, "one entry per layer, unconditionally");
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
        assert_eq!(d.advertised, Advertised::default(), "the row fills this, not the projection");
        assert_eq!(
            d.towers,
            Towers::default(),
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
            Some(Gemma4Mixture::gemma_4_26b_a4b()),
            1024,
            NORM_EPS,
            Deployed::single(),
        );
        assert_eq!(routed.shape.intermediate, 2112, "the DENSE width stays in `intermediate`");
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
        assert_eq!(d.attention[5].head_dim, 512, "layer 5 is the first full one");
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
        let d = deployment(&Gemma4Facts::gemma_4_e4b(), None, -1, NORM_EPS, Deployed::single());
        assert_eq!(d.attention[0].window, 0, "a sliding layer with no stated window");
        assert_eq!(d.attention[5].window, -1, "a full layer still sees everything");
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
            assert_eq!(d.attention[l as usize].kv_source, l, "layer {l} owns its pages");
        }
        for l in 24..42 {
            let src = d.attention[l as usize].kv_source;
            assert!(src < 24, "layer {l} sources from {src}, which owns no pages itself");
            assert_eq!(
                f.is_full_attn(src),
                f.is_full_attn(l),
                "layer {l} sources from the other attention kind, whose heads are a \
                 different width"
            );
            assert_eq!(
                d.attention[l as usize].head_dim,
                d.attention[src as usize].head_dim,
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
        let d = deployment(&f, None, 512, NORM_EPS, Deployed::single());
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
            let want =
                if f.is_full_attn(l) { ROPE_THETA_GLOBAL } else { ROPE_THETA_LOCAL };
            assert_eq!(d.attention[l as usize].rope_theta, want, "layer {l}");
        }
        assert_eq!(d.theta_by_layer().len(), 42, "the two bases differ, so the table is real");
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
        assert_eq!(bare.scales.len(), 4, "no layer scalars were handed to this load");
        assert_eq!(bare.scales["sqrt_hidden"], 2560f32.sqrt());
        assert_eq!(bare.scales["sqrt_ple_dim"], 256f32.sqrt());
        assert_eq!(bare.scales["rsqrt_hidden"], 1.0 / 2560f32.sqrt());
        assert_eq!(bare.scales["rsqrt_2"], 1.0 / 2f32.sqrt());

        let scalars = [0.5f32, 0.25, 0.125];
        let loaded = deployment(
            &Gemma4Facts::gemma_4_e4b(),
            None,
            E4B_WINDOW,
            NORM_EPS,
            Deployed { tp_size: 1, layer_scalars: &scalars },
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
            Some(Gemma4Mixture::gemma_4_26b_a4b()),
            1024,
            NORM_EPS,
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
        let d = deployment(&f, None, 512, NORM_EPS, Deployed::single());
        assert_eq!(d.attention.len(), 35);
        assert_eq!(d.attention[34].head_dim, 512, "layer 34 is a full layer");
        assert_eq!(d.attention[34].window, -1);
        assert_eq!(d.decode_head_dims(), Some((256, 512)));
        assert!(d.shares_kv());
        for l in 0..15u32 {
            assert_eq!(d.attention[l as usize].kv_source, l);
        }
        assert_eq!(d.attention[15].kv_source, 13, "the last earlier sliding layer");
        assert_eq!(d.attention[34].kv_source, 14, "the last earlier full layer");
    }
}

//! The three projections a gemma-3n row makes.
//!
//! gemma-3n is not a `llama_like` configuration and cannot borrow that
//! family's projections. What sets it apart is not one thing but four:
//! a rank-4 AltUp residual, a low-rank laurel branch beside attention, a
//! gaussian top-k on the first ten MLPs, and — the one the deployment
//! has to carry — a PER-LAYER EMBEDDING table that gates into every
//! block. So the three answers live here, taking a `&Gemma3nFacts`.
//!
//! # Two things the old derivation got wrong by DEFAULT
//!
//! `impl PlannedFamily for Gemma3nFacts` overrode `trace`, `layers`,
//! `head_dim_of` and `window_by_layer`, and let `tables()` default. The
//! default returns `FamilyTables::default()` — `ple_dim: 0` and
//! `softcap: 0.0` — so the family whose defining feature is a per-layer
//! embedding deployed claiming it had none, and the family whose config
//! states `final_logit_softcapping: 30.0` deployed uncapped. Neither was
//! a decision; both were a trait method nobody wrote, which is exactly
//! what "no default bodies" is in the [`crate::catalog::Variant`]
//! contract to prevent. A row that must answer every question cannot
//! forget to answer one.
//!
//! The rope base is the third correction and a smaller one: gemma-3n
//! states `rope_theta: 1e6` for its full-attention layers and
//! `rope_local_base_freq: 1e4` for its sliding ones, and the default
//! table broadcast the former to all thirty. A window and a rope base
//! are the same decision seen twice; a projection that emits one per
//! layer emits both per layer.

// Only the texts name a backend, and only they are gated.
use crate::catalog::Deployed;
use crate::deployment::{
    Advertised, AttnOutput, Deployment, Geometry, KvStyle, LayerAttention, NormPlacement,
    PrefillStyle,
};
use crate::manifest::{Manifest, TensorSpec};

use super::spec::Gemma3nFacts;

/// The final-logit cap, as [`crate::gemma_2::project::FINAL_LOGIT_SOFTCAP`]
/// is: a constant of the generation rather than a config read. Every
/// published gemma-3n states 30.0.
pub const FINAL_LOGIT_SOFTCAP: f32 = 30.0;

/// This row's tensors.
///
/// # The names that are not the decoder's
///
/// The ordinary decoder rows are the gemma vocabulary
/// [`crate::gemma_4::contract`] maps for this same lineage —
/// `embed_tokens`, `norm`, `self_attn.*`, `mlp.*` and the FOUR gemma
/// norms — and the PLE trio is named there too, layer-less and
/// unwrapped: `embed_tokens_per_layer`, `per_layer_model_projection`,
/// `per_layer_projection_norm`.
///
/// The per-layer AltUp and laurel rows are stated at the extents this
/// row's own numbers imply, which is the point of a manifest, but their
/// CHECKPOINT names are the one thing here no file in this repo pins:
/// gemma-3n loads through `author_dense`, which publishes whatever it
/// is handed rather than renaming it, so there is no mapping table to
/// read them off. They are written as the upstream module tree spells
/// them. If a real gemma-3n fails to identify, these four rows are the
/// place to look, and that is an improvement on the previous
/// arrangement, where the expectation existed only as a load-time
/// `UnknownWeight` in a family nothing had stated.
#[must_use]
pub fn manifest(f: &Gemma3nFacts) -> Manifest {
    let (hidden, vocab) = (u64::from(f.hidden), u64::from(f.vocab));
    let (q, kv) = (u64::from(f.attn.q_width()), u64::from(f.attn.kv_width()));
    let head_dim = u64::from(f.attn.head_dim);
    let inter = u64::from(f.intermediate(0));
    let ple = u64::from(f.ple_width);
    let layers = u64::from(f.layers());

    Manifest::new(f.layers())
        // The shipped projection repr, DERIVED from the family's one
        // axis spelling (`forward::ShippedW1`) — plain bf16, inert for
        // matching; it states only what the catalogued text assumes.
        .holds_projections_as(<super::forward::ShippedW1 as model_dsl::axes::DtypeAxis>::REPR)
        .with(TensorSpec::required("embed_tokens", [vocab, hidden]))
        .with(TensorSpec::required("norm", [hidden]))
        // gemma-3n ties, and a tie is an ABSENCE.
        .with(TensorSpec::absent("lm_head"))
        // THE PER-LAYER EMBEDDING TABLE. One row of `ple_width` per
        // layer per token, which is why the second axis is the product:
        // no other family in this catalog publishes a tensor whose shape
        // multiplies the layer count, and a checkpoint holding one is a
        // gemma-3n or a gemma-4 and nothing else.
        .with(TensorSpec::required(
            "embed_tokens_per_layer",
            [vocab, layers * ple],
        ))
        .with(TensorSpec::required(
            "per_layer_model_projection",
            [layers * ple, hidden],
        ))
        .with(TensorSpec::required("per_layer_projection_norm", [ple]))
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
        // Per-head q/k norms — gemma-3n normalises inside the head, so
        // the extent is the head dim and not the projection width. The
        // old derivation had a special case that measured
        // `elems_of("layer.0.q_norm")` to tell those two apart; the
        // measurement is this row.
        .with(TensorSpec::required(
            "layer.{}.self_attn.q_norm",
            [head_dim],
        ))
        .with(TensorSpec::required(
            "layer.{}.self_attn.k_norm",
            [head_dim],
        ))
        // gemma-3n normalises V as well, which gemma-2 and gemma-3 do
        // not, and it is the cheapest single row that separates this
        // generation from those.
        .with(TensorSpec::required(
            "layer.{}.self_attn.v_norm",
            [head_dim],
        ))
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
        // The laurel branch: down to `laurel_rank` and back.
        .with(TensorSpec::required(
            "layer.{}.laurel.linear_left",
            [u64::from(f.laurel_rank), hidden],
        ))
        .with(TensorSpec::required(
            "layer.{}.laurel.linear_right",
            [hidden, u64::from(f.laurel_rank)],
        ))
        .with(TensorSpec::required(
            "layer.{}.laurel.post_laurel_norm",
            [hidden],
        ))
        // AltUp's router picks which stream the block body runs on, so
        // its width IS the stream count.
        .with(TensorSpec::required(
            "layer.{}.altup.modality_router",
            [u64::from(f.altup.num_streams), hidden],
        ))
        .with(TensorSpec::required("layer.{}.altup.router_norm", [hidden]))
        // Where the per-layer embedding gates in.
        .with(TensorSpec::required(
            "layer.{}.per_layer_input_gate",
            [ple, hidden],
        ))
        .with(TensorSpec::required(
            "layer.{}.per_layer_projection",
            [hidden, ple],
        ))
        .with(TensorSpec::required(
            "layer.{}.post_per_layer_input_norm",
            [hidden],
        ))
}

/// This row's deployment.
///
/// `rope_theta_global` and `rope_theta_local` are the row's, not the
/// shape's: they are the two bases the config states beside a window
/// schedule the shape already holds, and passing them here keeps the
/// shape the thing a checkpoint can be measured against.
#[must_use]
pub fn deployment(
    f: &Gemma3nFacts,
    rope_theta_global: f32,
    rope_theta_local: f32,
    norm_eps: f32,
) -> Deployment {
    let head_dim = crate::deployment::round_up_attn_head_dim(f.attn.head_dim);
    let attention = (0..f.layers())
        .map(|l| {
            let window = model_ir::facts::window_left_at(f.window_left, l);
            LayerAttention {
                // One shape for every layer, which is what this row
                // was already saying by having no per-layer count.
                kv_heads: f.attn.kv_heads,
                head_dim,
                window,
                // Every layer writes and reads its OWN pages. The config
                // states `num_kv_shared_layers: 10`, and this row
                // deliberately does not carry it: the traced text
                // (`gemma3n_cuda`) writes a k/v plane per layer, so a
                // deployment claiming layer 25 reads layer 19's pages
                // would be a claim the driver acts on and the forward
                // contradicts. Sharing is stated where it is
                // implemented, which is gemma-4.
                kv_source: l,
                sm_scale: 1.0 / (head_dim as f32).sqrt(),
                // A window and a rope base are one decision seen twice.
                rope_theta: if window < 0 {
                    rope_theta_global
                } else {
                    rope_theta_local
                },
                // Full rotation at the head dim.
                rotary_dim: 0,
                q_gate: false,
            }
        })
        .collect();
    Deployment {
        layers: f.layers(),
        norm_eps,
        shape: Geometry {
            hidden: f.hidden,
            q_heads: f.attn.heads,
            kv_heads: f.attn.kv_heads,
            head_dim: f.attn.head_dim,
            head_dim_kernel: head_dim,
            // Layer 0's. The config states an MLP width per layer and
            // every published gemma-3n states the same one thirty times;
            // the per-layer list is what the TRACE reads, and this field
            // is the launch geometry, which is uniform or the launch
            // could not be one shape.
            intermediate: f.intermediate(0),
            // Dense. Gemma-3n's sparsity is activation-level — the
            // per-layer `activation_sparsity` threshold zeroes entries
            // of ONE feed-forward, it does not route to experts — so
            // there is no second width for a planner to size against.
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
        logit_softcap: FINAL_LOGIT_SOFTCAP,
        // No ATTENTION cap: gemma-2's `attn_logit_softcapping` is
        // gemma-2's alone, and a zero here is "no cap" rather than a
        // cap at zero — which would flatten every score to `tanh(inf)`.
        attn_logit_softcap: 0.0,
        // THE FIELD THIS GENERATION EXISTS FOR. Nonzero here and zero in
        // every other row of the catalog: the driver sizes the per-layer
        // embedding gather from it, and the old derivation left it at
        // the trait's default of zero.
        // See `Deployment::ple_dim`: the row holds a `u32` and this
        // no longer narrows through a conversion whose failure arm
        // meant "this stack has none".
        ple_dim: f.ple_width,
        norm: NormPlacement::Pre,
        // A gemma-3 derivative, and it kept the fold: this generation's
        // forward fires `NormVariant::Gemma`.
        norm_unit_offset: true,
        // gemma-3 carries the per-head q/k norm and NO V norm: the two
        // are separate facts, and this row is where that is said.
        v_norm: false,
        // Dense: no router reads this.
        norm_topk_prob: true,
        // No router of this family states a scaling factor.
        routed_scaling: 1.0,
        mlp_gate: crate::deployment::MlpGate::GeluTanh,
        // No named host scalars: gemma-3n's `sqrt(hidden)` embedding
        // scale and its AltUp coefficient clip are stated inside the
        // traced text, not looked up by name at fire time. Gemma-4 is
        // the generation that needs the table.
        scales: std::collections::BTreeMap::new(),
        // Filled by the ROW, not by the shape: a family label and a
        // published context ceiling are facts about a checkpoint, and a
        // projection only sees geometry.
        advertised: Advertised::default(),
        rope_scaling: None,
        towers: Default::default(),
    }
}

/// Why this build has no Metal text for a gemma-3n row.
///
/// A `const` so the test that asserts the refusal NAMES the missing
/// thing compares against the same string the caller is shown, rather
/// than against a paraphrase that can drift away from it — the shape
/// `csm::project::NO_TRACE` set for the same reason.
///
/// Its forward is `gemma3n_cuda`: AltUp's four-way hidden
/// bundle, the Laurel residual, per-layer embeddings and the shared-KV
/// tail. `llama_like_metal` carries a `per_layer_emb_dim` and a
/// `kv_shared_layers` — which is exactly the trap, because carrying two
/// of the four would trace a model that is recognisably gemma-3n and is
/// not this one.
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
pub const NO_METAL: &str = "gemma-3n has no Metal text in this build: its forward is `gemma3n_cuda` — \
     AltUp's four-way hidden bundle, the Laurel residual, per-layer embeddings \
     and the shared-KV tail — and the one Metal text here (`llama_like_metal`) \
     states two of those four and takes a different shape; a text that is \
     recognisably gemma-3n and is not this one is the failure to avoid";

/// Trace this row's CUDA text for one fire class.
///
/// `load` is unread and named anyway, as gemma-2's is: gemma-3n binds
/// nothing from the load, and the three projections keep one shape so a
/// reader can see at a glance which generation reads what.
#[must_use]
pub fn trace(
    f: &Gemma3nFacts,
    class: model_ir::trace::FireClass,
    load: Deployed<'_>,
    norm_eps: f32,
    rope_theta_global: f32,
    rope_theta_local: f32,
) -> model_ir::trace::ForwardPlan {
    let _ = load;
    // THE SHIPPED POINT, read off the family's one axis spelling
    // (`forward::Shipped*`, beside its CATALOG). gemma-3n catalogues one
    // SKU today; the table in `forward::CATALOG` is where a second one
    // appears, and the coverage test is what keeps every row loadable.
    super::forward::gemma3n_cuda::<
        super::forward::ShippedW1,
        super::forward::ShippedA,
        super::forward::ShippedKv,
    >(
        f,
        class,
        norm_eps,
        rope_theta_global,
        rope_theta_local,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gemma_3n::spec::{Gemma3nAltUpFacts, Gemma3nAttnFacts, window_schedule};
    use crate::manifest::Presence;

    /// E2B, as the published config states it — built here rather than
    /// imported from the row so this file's tests fail on this file's
    /// arithmetic and not on the row's.
    fn e2b() -> Gemma3nFacts {
        const W: [i32; 30] = window_schedule(5, 512);
        Gemma3nFacts {
            vocab: 262_400,
            hidden: 2048,
            per_layer_intermediate: &[8192; 30],
            laurel_rank: 64,
            ple_width: 256,
            sparsity_layers: 10,
            altup: Gemma3nAltUpFacts {
                num_streams: 4,
                active: 0,
            },
            attn: Gemma3nAttnFacts {
                heads: 8,
                kv_heads: 2,
                head_dim: 256,
            },
            window_left: &W,
        }
    }

    /// The per-layer embedding table is the row that multiplies by the
    /// layer count, and no other manifest in this catalog has one.
    #[test]
    fn the_ple_table_is_sized_by_the_layer_count() {
        let m = manifest(&e2b());
        let ext = |n: &str| {
            m.tensors
                .iter()
                .find(|t| t.name == n)
                .expect("stated")
                .extents
                .clone()
        };
        assert_eq!(ext("embed_tokens_per_layer"), vec![262_400, 30 * 256]);
        assert_eq!(ext("per_layer_model_projection"), vec![30 * 256, 2048]);
        assert_eq!(ext("per_layer_projection_norm"), vec![256]);
    }

    /// Every extent is the row's own arithmetic: 8 heads of 256 and 2 kv
    /// heads of 256 over a 2048-wide residual.
    #[test]
    fn the_extents_are_the_rows_own_arithmetic() {
        let m = manifest(&e2b());
        let ext = |n: &str| {
            m.tensors
                .iter()
                .find(|t| t.name == n)
                .expect("stated")
                .extents
                .clone()
        };
        assert_eq!(ext("layer.{}.self_attn.q_proj"), vec![2048, 2048]);
        assert_eq!(ext("layer.{}.self_attn.k_proj"), vec![512, 2048]);
        assert_eq!(ext("layer.{}.self_attn.o_proj"), vec![2048, 2048]);
        assert_eq!(ext("layer.{}.mlp.gate_proj"), vec![8192, 2048]);
        assert_eq!(ext("layer.{}.laurel.linear_left"), vec![64, 2048]);
        assert_eq!(ext("layer.{}.laurel.linear_right"), vec![2048, 64]);
        assert_eq!(ext("layer.{}.altup.modality_router"), vec![4, 2048]);
        assert_eq!(m.layers, 30);
    }

    /// The q/k/v norms are PER HEAD, which is the measurement
    /// `elems_of("layer.0.q_norm") == head_dim` used to make inside a
    /// derivation with nowhere to write the answer down.
    #[test]
    fn the_qkv_norms_are_stated_at_the_head_dim() {
        let m = manifest(&e2b());
        for name in [
            "layer.{}.self_attn.q_norm",
            "layer.{}.self_attn.k_norm",
            "layer.{}.self_attn.v_norm",
        ] {
            let spec = m.tensors.iter().find(|t| t.name == name).expect("stated");
            assert_eq!(spec.presence, Presence::Required, "{name}");
            assert_eq!(spec.extents, vec![256], "{name}");
        }
    }

    /// A tie is an absence, and gemma-3n ties.
    #[test]
    fn the_tied_head_is_an_absence_the_manifest_expects() {
        let m = manifest(&e2b());
        let head = m
            .tensors
            .iter()
            .find(|t| t.name == "lm_head")
            .expect("stated");
        assert_eq!(head.presence, Presence::Absent);
    }

    /// THE CLAIM. `ple_dim` reaches the deployment nonzero, where the
    /// old path left `FamilyTables::default()` to answer zero for the
    /// generation that is named after having one.
    #[test]
    fn the_per_layer_embedding_width_reaches_the_deployment() {
        let d = deployment(&e2b(), 1_000_000.0, 10_000.0, 1e-6);
        assert_eq!(d.ple_dim, 256);
        assert_eq!(
            d.ple_dim,
            e2b().ple_width,
            "the row's own width, no conversion between"
        );
        assert!(d.ple_dim > 0, "the default this row replaces was 0");
    }

    /// And the cap, which the same default answered wrong: the config
    /// states 30.0 and the deployment carried 0.0.
    #[test]
    fn the_final_softcap_reaches_the_deployment() {
        assert_eq!(
            deployment(&e2b(), 1_000_000.0, 10_000.0, 1e-6).logit_softcap,
            30.0
        );
    }

    /// The window schedule the shape holds is the schedule every layer
    /// deploys with — four sliding then one full, six times over.
    #[test]
    fn the_window_schedule_reaches_every_layer() {
        let f = e2b();
        let d = deployment(&f, 1_000_000.0, 10_000.0, 1e-6);
        let windows: Vec<i32> = d.attention.iter().map(|a| a.window).collect();
        assert_eq!(windows, f.window_left.to_vec());
        assert_eq!(windows.len(), 30);
        let full: Vec<usize> = (0..30).filter(|&l| windows[l] == -1).collect();
        assert_eq!(full, vec![4, 9, 14, 19, 24, 29]);
    }

    /// A window and a rope base are one decision seen twice: the full
    /// layers rotate on 1e6 and the sliding ones on 1e4, where the
    /// default table broadcast 1e6 to all thirty.
    #[test]
    fn the_rope_base_follows_the_window() {
        let d = deployment(&e2b(), 1_000_000.0, 10_000.0, 1e-6);
        for a in &d.attention {
            if a.window < 0 {
                assert_eq!(a.rope_theta, 1_000_000.0);
            } else {
                assert_eq!(a.rope_theta, 10_000.0, "sliding layers take the local base");
            }
        }
        assert_eq!(d.attention[4].rope_theta, 1_000_000.0);
        assert_eq!(d.attention[0].rope_theta, 10_000.0);
    }

    /// KV sharing is stated where it is implemented, and that is not
    /// here: the traced text writes a plane per layer, so the row says
    /// every layer owns its pages even though the config counts ten
    /// shared ones.
    #[test]
    fn every_layer_owns_its_pages() {
        let d = deployment(&e2b(), 1_000_000.0, 10_000.0, 1e-6);
        for (l, a) in d.attention.iter().enumerate() {
            assert_eq!(a.kv_source, l as u32);
        }
    }

    /// The launch geometry is the row's own numbers.
    #[test]
    fn the_launch_geometry_is_the_rows_own_numbers() {
        let d = deployment(&e2b(), 1_000_000.0, 10_000.0, 1e-6);
        assert_eq!(d.shape.hidden, 2048);
        assert_eq!(d.shape.q_heads, 8);
        assert_eq!(d.shape.kv_heads, 2);
        assert_eq!(d.shape.head_dim, 256);
        assert_eq!(
            d.shape.head_dim_kernel, 256,
            "256 is instantiated; nothing pads"
        );
        assert_eq!(d.shape.intermediate, 8192);
        assert_eq!(d.shape.vocab, 262_400);
        assert_eq!(d.shape.gqa_group(), 4);
        assert_eq!(d.layers, 30);
        assert_eq!(d.attention.len(), 30);
    }

    /// The rest of the answers, stated rather than defaulted — which is
    /// the whole point, given that two of this family's defaults were
    /// wrong.
    #[test]
    fn the_defaults_the_old_vtable_supplied_are_stated() {
        let d = deployment(&e2b(), 1_000_000.0, 10_000.0, 1e-6);
        assert_eq!(d.kv, KvStyle::Paged);
        assert!(d.recurrent.is_none());
        assert_eq!(d.prefill, PrefillStyle::Planned);
        assert_eq!(
            d.attn_output,
            AttnOutput::DriverPinned,
            "SSA args are gemma-4's"
        );
        assert_eq!(d.norm, NormPlacement::Pre);
        assert!(d.scales.is_empty());
        for a in &d.attention {
            assert_eq!(a.rotary_dim, 0, "full rotation at the head dim");
            assert_eq!(a.head_dim, 256);
            assert!((a.sm_scale - 1.0 / 16.0).abs() < 1e-6, "1/sqrt(256)");
        }
    }

    /// E4B differs from E2B in exactly two numbers, and both reach the
    /// projections — otherwise the two rows would be one row.
    #[test]
    fn the_e4b_widths_reach_the_manifest_and_the_geometry() {
        let mut f = e2b();
        const W: [i32; 35] = window_schedule(5, 512);
        f.per_layer_intermediate = &[16384; 35];
        f.window_left = &W;
        let m = manifest(&f);
        let ext = |n: &str| {
            m.tensors
                .iter()
                .find(|t| t.name == n)
                .expect("stated")
                .extents
                .clone()
        };
        assert_eq!(ext("layer.{}.mlp.gate_proj"), vec![16384, 2048]);
        assert_eq!(ext("embed_tokens_per_layer"), vec![262_400, 35 * 256]);
        assert_eq!(m.layers, 35);

        let d = deployment(&f, 1_000_000.0, 10_000.0, 1e-6);
        assert_eq!(d.layers, 35);
        assert_eq!(d.shape.intermediate, 16384);
        assert_eq!(d.attention[34].window, -1, "the last layer is a full one");
    }
}

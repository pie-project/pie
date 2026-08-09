//! The Qwen3.5 hybrid's projections: what its numbers imply about a
//! checkpoint, a deployment and a trace.
//!
//! Three functions, and each one used to be a paragraph of
//! `deployment_cuda`. `qwen35_facts_from_hf` read a parsed `config.json`
//! for the twenty numbers below; the vtable's `gdn_shape()` re-derived
//! the recurrent geometry from those facts a second time, at fire time,
//! through a trait whose other twelve methods were defaults. The numbers
//! were always the ROW's — a checkpoint of Qwen3.5-4B has 32 layers
//! whether or not anyone parsed a document to find out — so they are
//! stated once and projected here.
//!
//! Written against [`Qwen35HybridFacts`] rather than against a row type,
//! for `llama_like::project`'s reason: the dense hybrid and the MoE
//! hybrid are the same shape with a different block between the norms,
//! and a projection that took the row would have to be written twice.

// Only the texts name a backend, and only they are gated.
use crate::catalog::Deployed;
use crate::deployment::round_up_attn_head_dim;
use crate::deployment::{
    Advertised, AttnOutput, Deployment, Geometry, KvStyle, LayerAttention, NormPlacement,
    PrefillStyle, RecurrentShape,
};
use crate::manifest::{Manifest, TensorSpec};

use super::spec::{Qwen35HybridFacts, Qwen35MlpKind};

/// This row's tensors.
///
/// # A hybrid's per-layer rows are a UNION, and they have to be
///
/// [`crate::manifest::Observed::logical`] rewrites `layers.<n>.` to
/// `layer.{}.`, so every layer of a stack collapses onto one key and a
/// manifest cannot say "layer 3 attends and layer 2 does not". What a
/// hybrid checkpoint publishes under the collapsed name is therefore the
/// UNION over its layer kinds — the GDN block's tensors AND the
/// full-attention block's — and that is exactly what is stated below.
///
/// The SCHEDULE is not lost, it is just not a manifest's to hold:
/// `full_attn_interval` is the row's own statement of which layers are
/// which, and [`deployment`] and the trace are where it is read. A
/// checkpoint that ships both sets is this family; one that ships only
/// the attention set is a llama-like model and matches a llama-like row.
#[must_use]
pub fn manifest(f: &Qwen35HybridFacts) -> Manifest {
    let (hidden, vocab) = (u64::from(f.hidden()), u64::from(f.vocab));
    let a = &f.attn;
    let g = &f.gdn;
    // The output-gated q bank: `[query | gate]` in one tensor, so the
    // projection is twice the head width. A row that stated `q_width`
    // here would be describing a model whose gate is missing.
    let q2 = u64::from(2 * a.q_width());
    let kv = u64::from(a.kv_width());
    let head_dim = u64::from(a.head_dim);
    let (conv_dim, v_width) = (u64::from(g.conv_dim()), u64::from(g.value_width()));
    let v_heads = u64::from(g.value_heads);

    let m = Manifest::new(f.layers)
        .with(TensorSpec::required("embed_tokens", [vocab, hidden]))
        .with(TensorSpec::required("norm", [hidden]))
        .either(!f.tied_embeddings, "lm_head", [vocab, hidden])
        // The full-attention block.
        .with(TensorSpec::required(
            "layer.{}.self_attn.q_proj",
            [q2, hidden],
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
            [hidden, u64::from(a.q_width())],
        ))
        // Per-head q/k norms, which is what `head_dim` MEANS here: the
        // llama-like derivation divided a byte count to find this out.
        .with(TensorSpec::required(
            "layer.{}.self_attn.q_norm",
            [head_dim],
        ))
        .with(TensorSpec::required(
            "layer.{}.self_attn.k_norm",
            [head_dim],
        ))
        // The GDN block. Four unfused projections: the checkpoint ships
        // `in_proj_{qkv,z,b,a}` and the fused `qkvz`/`ba` banks are a
        // JOIN the loader performs behind an env gate, which is why
        // `Qwen35GdnFacts::fused_in_proj` is a binding fact and gets no
        // row here.
        .with(TensorSpec::required(
            "layer.{}.linear_attn.in_proj_qkv",
            [conv_dim, hidden],
        ))
        .with(TensorSpec::required(
            "layer.{}.linear_attn.in_proj_z",
            [v_width, hidden],
        ))
        .with(TensorSpec::required(
            "layer.{}.linear_attn.in_proj_b",
            [v_heads, hidden],
        ))
        .with(TensorSpec::required(
            "layer.{}.linear_attn.in_proj_a",
            [v_heads, hidden],
        ))
        // `[conv_dim, 1, kernel]` as HF stores it; `extents_agree`
        // squeezes the degenerate axis, so a converter that wrote
        // `[conv_dim, kernel]` still matches.
        .with(TensorSpec::required(
            "layer.{}.linear_attn.conv1d",
            [conv_dim, 1, u64::from(g.conv_kernel)],
        ))
        .with(TensorSpec::required(
            "layer.{}.linear_attn.conv1d.bias",
            [conv_dim],
        ))
        // One decay parameter and one step bias per VALUE head — the
        // scan is per head, and this is the extent that says so.
        .with(TensorSpec::required(
            "layer.{}.linear_attn.A_log",
            [v_heads],
        ))
        .with(TensorSpec::required(
            "layer.{}.linear_attn.dt_bias",
            [v_heads],
        ))
        // The gated norm folds per value-head CHANNEL, not per head.
        .with(TensorSpec::required(
            "layer.{}.linear_attn.norm",
            [u64::from(g.value_head_dim)],
        ))
        .with(TensorSpec::required(
            "layer.{}.linear_attn.out_proj",
            [hidden, v_width],
        ))
        .with(TensorSpec::required("layer.{}.input_layernorm", [hidden]))
        .with(TensorSpec::required(
            "layer.{}.post_attention_layernorm",
            [hidden],
        ));

    match &f.mlp {
        Qwen35MlpKind::Dense { intermediate } => {
            let i = u64::from(*intermediate);
            m.with(TensorSpec::required("layer.{}.mlp.gate_proj", [i, hidden]))
                .with(TensorSpec::required("layer.{}.mlp.up_proj", [i, hidden]))
                .with(TensorSpec::required("layer.{}.mlp.down_proj", [hidden, i]))
                // A dense hybrid has no router, and saying so is how a
                // dense row and a mixture row of the same width stay
                // distinguishable.
                .with(TensorSpec::absent("layer.{}.mlp.gate"))
        }
        Qwen35MlpKind::Moe(moe) => {
            let shared = u64::from(moe.shared_expert_intermediate);
            m.with(TensorSpec::required(
                "layer.{}.mlp.gate",
                [u64::from(moe.num_experts), hidden],
            ))
            // The expert bank's extents are a PACKING decision — a
            // stacked `[experts, 2 * moe_intermediate, hidden]` slab
            // or one tensor per expert, quantized or not — so the
            // spec asks that it exist and says nothing about its
            // shape. This is the same rule that keeps an FP8 build
            // and a bf16 build on one row.
            .with(TensorSpec::present("layer.{}.mlp.experts.0.gate_proj"))
            .with(TensorSpec::present("layer.{}.mlp.experts.0.down_proj"))
            .either(
                shared != 0,
                "layer.{}.mlp.shared_expert.gate_proj",
                [shared, hidden],
            )
            .with(TensorSpec::absent("layer.{}.mlp.gate_proj"))
        }
    }
}

/// This row's deployment.
///
/// The window is `-1` everywhere and stated rather than passed: no
/// Qwen3.5 config ships a `sliding_window`, the full-attention layers
/// attend the whole context, and the GDN layers have no window to
/// attend over — their history is the recurrent state.
#[must_use]
pub fn deployment(f: &Qwen35HybridFacts, rope_theta: f32, norm_eps: f32) -> Deployment {
    let a = &f.attn;
    let head_dim = round_up_attn_head_dim(a.head_dim);
    let attention = (0..f.layers)
        .map(|l| LayerAttention {
            // One shape for every layer, which is what this row was
            // already saying by having no per-layer count.
            kv_heads: a.kv_heads,
            head_dim,
            window: -1,
            kv_source: l,
            sm_scale: 1.0 / (head_dim as f32).sqrt(),
            rope_theta,
            // PARTIAL rotation, and the one row in this crate where the
            // field is not zero: only the leading 64 of 256 channels
            // rotate. The old derivation left it 0 and let the tracer
            // read `rotary_dim` out of the family's own facts, which is
            // the second reading this table exists to remove.
            rotary_dim: a.rotary_dim,
        })
        .collect();
    Deployment {
        layers: f.layers,
        norm_eps,
        shape: Geometry {
            hidden: f.hidden(),
            q_heads: a.q_heads,
            kv_heads: a.kv_heads,
            head_dim: a.head_dim,
            head_dim_kernel: head_dim,
            // A dense row's width is the block's; a mixture's dense
            // width is ZERO because no layer here runs a dense block —
            // Qwen3.5's mixture is uniform, every MLP is the router's.
            // The two are separate fields because `widest_mlp()` sizes
            // ONE forward workspace both kinds share, and a planner
            // given only the dense number under-sizes a mixture whose
            // experts are wider.
            intermediate: match &f.mlp {
                Qwen35MlpKind::Dense { intermediate } => *intermediate,
                Qwen35MlpKind::Moe(_) => 0,
            },
            moe_intermediate: match &f.mlp {
                Qwen35MlpKind::Dense { .. } => 0,
                Qwen35MlpKind::Moe(moe) => moe.moe_intermediate,
            },
            experts_per_token: match &f.mlp {
                Qwen35MlpKind::Dense { .. } => 0,
                Qwen35MlpKind::Moe(moe) => moe.top_k,
            },
            shared_intermediate: match &f.mlp {
                Qwen35MlpKind::Dense { .. } => 0,
                Qwen35MlpKind::Moe(moe) => moe.shared_expert_intermediate,
            },
            vocab: f.vocab,
        },
        attention,
        // Full attention pages ordinarily; the GDN layers hold no pages
        // at all, which `recurrent` below is what states.
        kv: KvStyle::Paged,
        recurrent: Some(gdn_shape(f)),
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
        // `Qwen3-30B-A3B` publishes `true` while the `Qwen3MoeConfig`
        // class default is `False`; the row wins. A dense qwen3.5 states
        // it too and nothing reads it.
        norm_topk_prob: true,
        // No router of this family states a scaling factor.
        routed_scaling: 1.0,
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

/// The recurrent slab geometry the GDN layers need allocated.
///
/// `PlannedFamily::gdn_shape()` verbatim, minus the vtable: it was a
/// method with a `None` default that twelve families inherited and one
/// overrode, which made "does this family carry recurrent state" a
/// question you answered by finding the override. Here it is a `Some`
/// on the row's own deployment.
#[must_use]
fn gdn_shape(f: &Qwen35HybridFacts) -> RecurrentShape {
    let g = &f.gdn;
    RecurrentShape {
        linear_layers: (0..f.layers).filter(|&l| !f.is_full_attn(l)).collect(),
        conv_stride: (g.conv_kernel * g.conv_dim()) as usize,
        state_stride: (g.value_heads * g.key_head_dim * g.value_head_dim) as usize,
        // The store is bf16. `RecurrentStateCache::allocate_bf16_recurrent`
        // is the only allocator a driver has for it, and
        // `Qwen35CudaFacts::state_bf16` is the same decision spelled for
        // the tracer — so the two agree by construction rather than by
        // a fire-time cross-check.
        state_elem: 2,
        k_h: g.key_heads as i32,
        v_h: g.value_heads as i32,
        k_d: g.key_head_dim as i32,
        v_d: g.value_head_dim as i32,
        conv_dim: g.conv_dim() as i32,
        conv_k: g.conv_kernel as i32,
        // A gated delta stack has no B/C groups; mamba's alone.
        n_groups: 0,
    }
}

/// The CUDA binding facts for this row.
///
/// Every field is a deployment's answer rather than a checkpoint's, and
/// the values below are the LIVE ones — the env defaults a boot with no
/// `PIE_*` set resolves to, corrected on first boot by the digest the
/// way `emissions.rs` documents. Two are not constants: `moe_*` follow
/// from whether this row has a mixture, and `moe_residual_fold` follows
/// from the load's TP width, because at `tp > 1` the block writes to
/// scratch and an allreduce follows.
#[must_use]
pub fn cuda_facts(
    f: &Qwen35HybridFacts,
    load: Deployed<'_>,
) -> super::forward::facts::Qwen35CudaFacts {
    let moe = matches!(f.mlp, Qwen35MlpKind::Moe(_));
    let shared_gate = match &f.mlp {
        Qwen35MlpKind::Moe(m) => m.shared_expert_intermediate != 0,
        Qwen35MlpKind::Dense { .. } => false,
    };
    super::forward::facts::Qwen35CudaFacts {
        state_bf16: true,
        // The warp-tiled prefill arm needs `PIE_QWEN35_GDN_WARP_TILED_STATE_PERSIST`,
        // and the cached family needs a non-zero
        // `PIE_QWEN35_GDN_CACHED_PREFILL_MAX_TOKENS`. Both are off by
        // default, which the emission fixture records the digest having
        // caught the synthetic set getting wrong.
        warp_tiled: false,
        warp_tiled_max: 64,
        cached_max: 0,
        verify_stash: true,
        // Zero means "no fused CUTLASS leg on this deployment": the row
        // bound is a WORKSPACE size, which no checkpoint states.
        moe_cutlass_max_rows: 0,
        prefill_decode: true,
        moe_residual_fold: moe && load.tp_size.max(1) == 1,
        moe_shared_gate_dot: shared_gate,
        moe_streamed_experts: false,
        moe_force_general: false,
        gate_up_fused: true,
        proj_repr: model_compiler::dsl::WeightRepr::Bf16,
        // Empty reads as "no window" — see `window_left_at`.
        window_left: Vec::new(),
    }
}

/// Why this build has no Metal text for a qwen-3.5 row.
///
/// A `const` so the test that asserts the refusal NAMES the missing
/// thing compares against the same string the caller is shown, rather
/// than against a paraphrase that can drift away from it — the shape
/// `csm::project::NO_TRACE` set for the same reason.
///
/// Its forward is `qwen3_5_hybrid_cuda`, which interleaves
/// gated DeltaNet layers with attention and states two SERVICE classes
/// beside the two shaped ones. `llama_like_metal` has neither the
/// recurrent layer kind nor the service classes.
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
pub const NO_METAL: &str = "qwen-3.5 has no Metal text in this build: its forward is \
     `qwen3_5_hybrid_cuda`, which interleaves gated DeltaNet layers with \
     attention and states two service classes beside the two shaped ones, and \
     the one Metal text here (`llama_like_metal`) has neither and takes a \
     different shape; the CUDA backend serves this row";

/// Trace this row's CUDA text for one fire class.
#[must_use]
pub fn trace(
    f: &Qwen35HybridFacts,
    class: model_compiler::trace::FireClass,
    load: Deployed<'_>,
) -> model_compiler::trace::ForwardPlan {
    super::forward::qwen3_5_hybrid_cuda(f, &cuda_facts(f, load), class)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{Observed, Presence};

    /// The 0.8B fixture, which every test here projects.
    fn dense() -> Qwen35HybridFacts {
        Qwen35HybridFacts::qwen3_5_0_8b()
    }

    /// The 35B-A3B ROW's shape, so the mixture arm is exercised against
    /// a measured checkpoint rather than an invented one — and against
    /// the same numbers the catalog serves.
    fn mixture() -> Qwen35HybridFacts {
        crate::qwen_3_5::VARIANTS
            .iter()
            .find(|v| v.id == "qwen3.5-35b-a3b")
            .expect("the mixture row is in the table")
            .shape
            .clone()
    }

    /// A manifest is a projection: what it states about the attention
    /// bank is what `q_heads` and `head_dim` MEAN, doubled because the
    /// bank carries the output gate beside the query.
    #[test]
    fn the_attention_rows_are_the_rows_own_arithmetic() {
        let f = dense();
        let m = manifest(&f);
        let q = m
            .tensors
            .iter()
            .find(|t| t.name.ends_with("q_proj"))
            .expect("stated");
        assert_eq!(
            q.extents,
            vec![u64::from(2 * f.attn.q_width()), u64::from(f.hidden())]
        );
        let o = m
            .tensors
            .iter()
            .find(|t| t.name.ends_with("o_proj"))
            .expect("stated");
        assert_eq!(
            o.extents,
            vec![u64::from(f.hidden()), u64::from(f.attn.q_width())]
        );
    }

    /// The GDN block's tensors sit beside the attention block's under
    /// one collapsed layer key, because `Observed::logical` collapses
    /// the stack and a hybrid publishes both sets across it.
    #[test]
    fn a_hybrids_layer_rows_are_the_union_of_its_layer_kinds() {
        let m = manifest(&dense());
        let names: Vec<&str> = m.tensors.iter().map(|t| t.name.as_str()).collect();
        assert!(
            names.contains(&"layer.{}.linear_attn.in_proj_qkv"),
            "{names:?}"
        );
        assert!(names.contains(&"layer.{}.self_attn.q_proj"), "{names:?}");
    }

    /// The conv bank's width is `2 * key_width + value_width`, and the
    /// three legs it packs are why the contract has to shard it
    /// blockwise rather than by rows.
    #[test]
    fn the_conv_bank_is_the_packed_kkv_width() {
        let f = dense();
        let m = manifest(&f);
        let conv = m
            .tensors
            .iter()
            .find(|t| t.name.ends_with("conv1d"))
            .expect("stated");
        assert_eq!(
            conv.extents,
            vec![u64::from(f.gdn.conv_dim()), 1, u64::from(f.gdn.conv_kernel)],
        );
        assert_eq!(
            f.gdn.conv_dim(),
            2 * f.gdn.key_width() + f.gdn.value_width()
        );
    }

    /// A tie is an ABSENCE the manifest expects — the only way tied and
    /// untied tell apart when every extent agrees.
    #[test]
    fn a_tie_is_an_absence() {
        let tied = manifest(&dense());
        let head = tied
            .tensors
            .iter()
            .find(|t| t.name == "lm_head")
            .expect("stated");
        assert_eq!(head.presence, Presence::Absent);
        let untied = manifest(&Qwen35HybridFacts {
            tied_embeddings: false,
            ..dense()
        });
        let head = untied
            .tensors
            .iter()
            .find(|t| t.name == "lm_head")
            .expect("stated");
        assert_eq!(head.presence, Presence::Required);
    }

    /// The dense hybrid and the MoE hybrid differ by a ROUTER, and each
    /// forbids the other's MLP — so no checkpoint satisfies both.
    #[test]
    fn the_mixture_ships_a_router_and_the_dense_row_forbids_one() {
        let d = manifest(&dense());
        let router = d
            .tensors
            .iter()
            .find(|t| t.name.ends_with("mlp.gate"))
            .expect("stated");
        assert_eq!(router.presence, Presence::Absent);

        let m = manifest(&mixture());
        let router = m
            .tensors
            .iter()
            .find(|t| t.name.ends_with("mlp.gate"))
            .expect("stated");
        assert_eq!(router.presence, Presence::Required);
        let dense_mlp = m
            .tensors
            .iter()
            .find(|t| t.name.ends_with("mlp.gate_proj"))
            .expect("stated");
        assert_eq!(dense_mlp.presence, Presence::Absent);
    }

    /// The expert bank is asked to EXIST and nothing more. That is the
    /// mechanism behind "an FP8 build and a bf16 build are one row": a
    /// packed slab's stored extents are a packing decision, and a spec
    /// that stated them would need one row per encoding.
    #[test]
    fn an_expert_bank_is_asked_to_exist_and_not_to_have_a_shape() {
        let m = manifest(&mixture());
        let bank = m
            .tensors
            .iter()
            .find(|t| t.name.contains("experts.0.gate_proj"))
            .expect("stated");
        assert_eq!(bank.presence, Presence::Required);
        assert!(bank.extents.is_empty(), "extents are a packing decision");
    }

    /// Every projection satisfies the checkpoint it implies, which is
    /// what makes a manifest a check rather than a second statement.
    #[test]
    fn each_manifest_is_satisfied_by_the_checkpoint_it_implies() {
        for f in [dense(), mixture(), Qwen35HybridFacts::qwen3_6_27b()] {
            let m = manifest(&f);
            let implied = Observed::from_pairs(
                m.tensors
                    .iter()
                    .filter(|t| t.presence != Presence::Absent)
                    .map(|t| (t.name.replace("{}", "0"), t.extents.clone())),
            );
            assert!(
                m.check(&implied).is_ok(),
                "{}",
                m.check(&implied).unwrap_err()
            );
        }
    }

    /// The recurrent geometry is stated on the DEPLOYMENT, so a driver
    /// allocating conv and state slabs never asks which family this is.
    /// It was a vtable method with a `None` default before, which made
    /// "does this carry recurrent state" a question about overrides.
    #[test]
    fn the_gdn_stack_states_its_recurrent_geometry() {
        let f = dense();
        let d = deployment(&f, 1e7, 1e-6);
        let r = d.recurrent.as_ref().expect("a GDN hybrid carries state");
        assert_eq!(
            r.linear_layers,
            vec![
                0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22
            ]
        );
        assert_eq!(
            r.linear_layers.len(),
            18,
            "24 layers, 6 of them full attention"
        );
        assert_eq!(
            r.conv_stride,
            (f.gdn.conv_kernel * f.gdn.conv_dim()) as usize
        );
        assert_eq!(
            r.state_stride,
            (f.gdn.value_heads * f.gdn.key_head_dim * f.gdn.value_head_dim) as usize,
        );
        assert_eq!(r.state_elem, 2, "the store the driver allocates is bf16");
        assert_eq!(r.conv_dim, f.gdn.conv_dim() as i32);
        assert_eq!(r.conv_k, f.gdn.conv_kernel as i32);
        assert_eq!((r.k_h, r.v_h, r.k_d, r.v_d), (16, 16, 128, 128));
    }

    /// The schedule the manifest cannot hold is the deployment's: a
    /// linear layer is one the interval does not land on.
    #[test]
    fn the_layer_schedule_is_the_interval_and_not_a_list() {
        let f = dense();
        let r = deployment(&f, 1e7, 1e-6).recurrent.expect("stated");
        for l in 0..f.layers {
            assert_eq!(
                !r.linear_layers.contains(&l),
                f.is_full_attn(l),
                "layer {l}"
            );
            assert_eq!(
                f.is_full_attn(l),
                l % f.full_attn_interval == f.full_attn_interval - 1
            );
        }
    }

    /// Partial rotation reaches the deployment as a WIDTH. Every other
    /// row in this crate states 0 (rotate the whole head); qwen3.5
    /// rotates 64 of 256, and a driver that assumed the head dim would
    /// rotate four times too much.
    #[test]
    fn partial_rotation_is_stated_per_layer_rather_than_read_from_the_family() {
        let f = dense();
        let d = deployment(&f, 1e7, 1e-6);
        assert_eq!(d.attention.len(), f.layers as usize);
        for a in &d.attention {
            assert_eq!(a.rotary_dim, 64);
            assert_eq!(a.rope_theta, 1e7);
            assert_eq!(a.window, -1, "no Qwen3.5 config ships a sliding window");
            assert_eq!(a.head_dim, 256);
            assert_eq!(a.sm_scale, 1.0 / 16.0, "1/sqrt(256)");
        }
        assert_eq!(
            d.rotary_by_layer().len(),
            f.layers as usize,
            "a real table, not empty"
        );
    }

    /// The launch geometry is the row's own numbers, and the mixture's
    /// `intermediate` is the per-expert width because that is what
    /// every MLP GEMM in the stack is sized by.
    #[test]
    fn the_launch_geometry_is_the_rows_own_numbers() {
        let f = dense();
        let d = deployment(&f, 1e7, 1e-6);
        assert_eq!(d.shape.hidden, f.hidden());
        assert_eq!(d.shape.q_heads, f.attn.q_heads);
        assert_eq!(d.shape.kv_heads, f.attn.kv_heads);
        assert_eq!(d.shape.head_dim, 256);
        assert_eq!(
            d.shape.head_dim_kernel, 256,
            "256 is instantiated; nothing pads"
        );
        assert_eq!(d.shape.head_dim_alloc(), 256);
        assert_eq!(d.shape.gqa_group(), 4, "8 q over 2 kv");
        assert_eq!(d.shape.vocab, f.vocab);
        assert_eq!(d.shape.intermediate, 3584);
        assert_eq!(
            d.shape.moe_intermediate, 0,
            "a dense row has no experts to be wide"
        );
        assert_eq!(d.shape.widest_mlp(), 3584);
        assert_eq!(
            d.norm_eps, 1e-6,
            "stated by the row; no tensor extent carries it"
        );
        let moe = deployment(&mixture(), 1e7, 1e-6);
        assert_eq!(moe.shape.moe_intermediate, 512, "the per-expert width");
        assert_eq!(
            moe.shape.intermediate, 0,
            "no layer in this stack runs a dense block"
        );
        assert_eq!(
            moe.shape.widest_mlp(),
            512,
            "what the one shared workspace must hold"
        );
    }

    /// The paged store, the pre-norm placement and the empty epilogue —
    /// stated, because a default body is a claim about rows nobody has
    /// written yet.
    #[test]
    fn the_rest_of_the_deployment_is_stated_rather_than_defaulted() {
        let d = deployment(&dense(), 1e7, 1e-6);
        assert_eq!(d.kv, KvStyle::Paged);
        assert_eq!(d.prefill, PrefillStyle::Planned);
        assert_eq!(d.attn_output, AttnOutput::DriverPinned);
        assert_eq!(d.norm, NormPlacement::Pre);
        assert_eq!(d.logit_softcap, 0.0);
        assert_eq!(d.ple_dim, 0);
        assert!(d.scales.is_empty());
        assert!(!d.shares_kv(), "every layer that attends owns its pages");
    }

    /// The binding facts are the LIVE env defaults, and the two that
    /// are not constants follow from the row and the load.
    #[test]
    fn the_binding_facts_are_the_live_defaults() {
        let dense_facts = cuda_facts(&dense(), Deployed::single());
        assert!(dense_facts.state_bf16);
        assert!(
            !dense_facts.warp_tiled,
            "the state-persist env is off by default"
        );
        assert_eq!(dense_facts.cached_max, 0);
        assert!(dense_facts.prefill_decode);
        assert!(dense_facts.gate_up_fused);
        assert!(
            !dense_facts.moe_residual_fold,
            "a dense row reaches no MoE op"
        );
        assert!(!dense_facts.moe_shared_gate_dot);
        assert!(dense_facts.window_left.is_empty());

        let moe_facts = cuda_facts(&mixture(), Deployed::single());
        assert!(
            moe_facts.moe_residual_fold,
            "tp == 1 folds into the residual"
        );
        assert!(
            moe_facts.moe_shared_gate_dot,
            "35B-A3B binds a shared expert"
        );
        let sharded = cuda_facts(
            &mixture(),
            Deployed {
                backend: crate::catalog::Backend::Cuda,
                tp_size: 4,
                layer_scalars: &[],
            },
        );
        assert!(
            !sharded.moe_residual_fold,
            "tp > 1 writes scratch and allreduces"
        );
    }

    /// The trace is the row's, for every class a fire can carry.
    #[test]
    fn every_fire_class_traces() {
        use model_compiler::trace::FireClass;
        for class in [FireClass::Decode, FireClass::Prefill] {
            let plan = trace(&dense(), class, Deployed::single());
            assert!(!plan.ops.is_empty(), "{class:?} traced nothing");
        }
    }
}

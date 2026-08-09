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

use crate::catalog::Deployed;
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
        .either(
            f.norm_placement == SpecNorm::Pre,
            "layer.{}.input_layernorm",
            [hidden],
        )
        .either(
            f.norm_placement == SpecNorm::Post,
            "layer.{}.post_attention_layernorm",
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
) -> super::facts::LlamaLikeCudaFacts {
    let kernel = round_up_attn_head_dim(f.head_dim);
    super::facts::LlamaLikeCudaFacts {
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

/// Trace this row's CUDA text for one fire class.
#[cfg(feature = "forward")]
#[must_use]
pub fn trace(
    f: &LlamaLikeFacts,
    class: model_compiler::trace::FireClass,
    load: Deployed<'_>,
) -> model_compiler::trace::ForwardPlan {
    super::forward::llama_like_cuda(f, &cuda_facts(f, load), class)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let olmo = deployment(&LlamaLikeFacts::olmo2_1b(), 500_000.0, -1);
        assert_eq!(olmo.norm, NormPlacement::Post);
        let qwen = deployment(&LlamaLikeFacts::qwen3_0_6b(), 1e6, -1);
        assert_eq!(qwen.norm, NormPlacement::Pre);
    }

    /// The launch geometry is the row's own numbers, not a second
    /// reading of anything.
    #[test]
    fn the_launch_geometry_is_the_rows_own_numbers() {
        let f = LlamaLikeFacts::qwen3_0_6b();
        let d = deployment(&f, 1e6, -1);
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
        let d = deployment(&f, 10_000.0, -1);
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
        let g = deployment(&LlamaLikeFacts::qwen3_0_6b(), 1e6, -1).shape.gqa_group();
        assert!(DECODE_GQA_GROUPS.contains(&g), "qwen3-0.6b's 2 is servable");
        assert_eq!(Geometry::EMPTY.gqa_group(), 0, "no division by zero");
    }

    /// A sliding window reaches every layer, and full attention is -1.
    #[test]
    fn the_window_is_stated_per_layer() {
        let f = LlamaLikeFacts::mistral_7b_v03();
        let full = deployment(&f, 1e6, -1);
        assert!(full.attention.iter().all(|a| a.window == -1));
        let windowed = deployment(&f, 1e6, 4096);
        assert!(windowed.attention.iter().all(|a| a.window == 4096));
    }

    /// Every layer owns its pages here. KV sharing is gemma-4's, and it
    /// is a fact about a LAYER there rather than a family here.
    #[test]
    fn every_layer_owns_its_own_pages() {
        let d = deployment(&LlamaLikeFacts::qwen3_0_6b(), 1e6, -1);
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
}

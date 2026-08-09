//! GPT-OSS's projections: what its numbers imply about a checkpoint, a
//! deployment and a trace.
//!
//! `gpt_oss_facts_from_hf` read fourteen numbers out of a parsed
//! `config.json` and the vtable answered three more per fire — the
//! per-layer window came back out of the CUDA facts through
//! `window_by_layer`, which meant the deployment's window table was a
//! second reading of a list the row already implied. The alternation is
//! a RULE here (`is_sliding`), stated once and projected.

use crate::catalog::Deployed;
use crate::deployment::{
    Advertised,
    AttnOutput, Deployment, Geometry, KvStyle, LayerAttention, NormPlacement, PrefillStyle,
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
        .either(!f.tied_embeddings, "lm_head", [vocab, hidden])
        .with(TensorSpec::required("layer.{}.self_attn.q_proj", [q, hidden]))
        .with(TensorSpec::required("layer.{}.self_attn.k_proj", [kv, hidden]))
        .with(TensorSpec::required("layer.{}.self_attn.v_proj", [kv, hidden]))
        .with(TensorSpec::required("layer.{}.self_attn.o_proj", [hidden, q]))
        // The four biases the row's `attention_bias` states. They are an
        // expectation rather than a probe: the old derivation asked the
        // LOAD whether a bias was bound and branched on the answer.
        .either(f.attention_bias, "layer.{}.self_attn.q_proj.bias", [q])
        .either(f.attention_bias, "layer.{}.self_attn.k_proj.bias", [kv])
        .either(f.attention_bias, "layer.{}.self_attn.v_proj.bias", [kv])
        .either(f.attention_bias, "layer.{}.self_attn.o_proj.bias", [hidden])
        // ATTENTION SINKS: one learned logit per query head, appended to
        // the softmax denominator. It is the tensor that makes gpt-oss
        // gpt-oss — no other row in this build ships one — and the
        // reason the attention statement has to produce an LSE beside
        // its output.
        .either(f.attn_sinks, "layer.{}.self_attn.sinks", [u64::from(f.q_heads)])
        .with(TensorSpec::required("layer.{}.input_layernorm", [hidden]))
        .with(TensorSpec::required("layer.{}.post_attention_layernorm", [hidden]))
        // The router. `[experts, hidden]` IS the claim that this is a
        // mixture of `experts` experts, and it is never quantized
        // (`modules_to_not_convert` names it), so its extents are safe
        // to state exactly.
        .with(TensorSpec::required("layer.{}.mlp.router", [experts, hidden]))
        .with(TensorSpec::required("layer.{}.mlp.router.bias", [experts]))
        // The expert banks, named by the half of each that survives
        // every packing. `2 *` because gate and up ship fused.
        .with(TensorSpec::required(
            "layer.{}.mlp.experts.gate_up_proj_bias",
            [experts, u64::from(2 * f.intermediate)],
        ))
        .with(TensorSpec::required("layer.{}.mlp.experts.down_proj_bias", [experts, hidden]))
}

/// gpt-oss's gate alpha, the 1.702 that makes `x * sigmoid(alpha * x)`
/// the GELU approximation its MLP is written against.
///
/// A CONSTANT and not a fact, because no published gpt-oss config
/// states it: `swiglu_limit` is a row's number and this is the
/// activation's own. It had no home at all before — the driver carried
/// a `swiglu_alpha` field nothing ever wrote, so every gpt-oss reaching
/// a Metal text would have gated on alpha zero.
const GATE_ALPHA: f32 = 1.702;

/// This row's deployment.
///
/// The window table is the row's alternation rule expanded, not a list
/// read back out of the CUDA facts: `is_sliding` is the statement, and
/// `GptOssCudaFacts::window_left` carries the same answer to the tracer.
/// Both come from here now, so the two cannot disagree.
#[must_use]
pub fn deployment(f: &GptOssFacts, rope_theta: f32, norm_eps: f32, sliding_window: i32) -> Deployment {
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
        ple_dim: 0,
        norm: NormPlacement::Pre,
        // Not a gemma: the gain is the multiplier, stored directly.
        norm_unit_offset: false,
        v_norm: false,
        k_eq_v: false,
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
#[cfg(feature = "forward")]
#[must_use]
pub fn cuda_facts(f: &GptOssFacts, load: Deployed<'_>) -> super::forward::facts::GptOssCudaFacts {
    let _ = load;
    super::forward::facts::GptOssCudaFacts {
        mxfp4_decode_gemv: true,
        mxfp4_decode_max_routes: 32 * f.experts,
        streamed_experts: false,
        window_left: (0..f.layers)
            .map(|l| if f.is_sliding(l) { 128 } else { -1 })
            .collect(),
    }
}

/// Why this build has no Metal text for a gpt-oss row.
///
/// A `const` so the test that asserts the refusal NAMES the missing
/// thing compares against the same string the caller is shown, rather
/// than against a paraphrase that can drift away from it — the shape
/// `csm::project::NO_TRACE` set for the same reason.
///
/// Its forward is `gpt_oss_cuda`, and this is the closest call
/// of the ten: `llama_like_metal` does carry attention sinks, the
/// clamped `SwiGlu` and an MXFP4 expert bank, and `driver-metal`'s
/// deleted `LLAMA_LIKE` table listed `gpt_oss` as served —
/// `LlamaLikeMetalFacts::gpt_oss_20b()` is still there as a fixture.
/// It refuses all the same, on the rule this refactor exists to state:
/// this row's shape is `GptOssFacts`, `llama_like_metal` takes a
/// `LlamaLikeFacts`, and a projection between them would be a
/// derivation of a model's facts from something other than its row —
/// which is precisely what `facts_from_with` was. Wiring it needs a
/// gpt-oss Metal TEXT, not a shape cast.
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
/// actually do. It listed `gemma4`, which the load path refused on
/// other grounds, and omitted `gemma3`, whose text it models. A row
/// that answers for itself cannot disagree with a list, because there
/// is no list.
pub const NO_METAL: &str = "gpt-oss has no Metal text in this build: its forward is `gpt_oss_cuda` — the \
     grouped seven-rectangle expert leg, the clamped GLU and attention sinks — \
     and while the one Metal text here (`llama_like_metal`) states sinks and a \
     clamped GLU of its own, it takes a `LlamaLikeFacts` and this row is a \
     `GptOssFacts`; a shape cast between them would be the tensor-sniffing this \
     catalog replaced, so what is wanted is a gpt-oss Metal text";

/// Trace this row's CUDA text for one fire class.
#[cfg(feature = "forward")]
#[must_use]
pub fn trace(
    f: &GptOssFacts,
    class: model_compiler::trace::FireClass,
    load: Deployed<'_>,
) -> model_compiler::trace::ForwardPlan {
    super::forward::gpt_oss_cuda(f, &cuda_facts(f, load), class)
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

    /// Every extent is the row's own arithmetic.
    #[test]
    fn the_attention_rows_are_the_rows_own_arithmetic() {
        let f = f20b();
        let m = manifest(&f);
        let q = m.tensors.iter().find(|t| t.name.ends_with("q_proj")).expect("stated");
        assert_eq!(q.extents, vec![u64::from(f.q_heads * f.head_dim), u64::from(f.hidden)]);
        let k = m.tensors.iter().find(|t| t.name.ends_with("k_proj")).expect("stated");
        assert_eq!(k.extents, vec![u64::from(f.kv_heads * f.head_dim), u64::from(f.hidden)]);
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

        // And a row without them FORBIDS the tensor, so the fact is a
        // presence rather than a branch inside one manifest.
        let without = manifest(&GptOssFacts { attn_sinks: false, ..f });
        let sinks = without
            .tensors
            .into_iter()
            .find(|t| t.name.ends_with("self_attn.sinks"))
            .expect("stated either way");
        assert_eq!(sinks.presence, Presence::Absent);
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
        mxfp4.push(("layer.0.mlp.experts.gate_up_proj_blocks".into(), vec![32, 5760, 90, 16]));
        mxfp4.push(("layer.0.mlp.experts.gate_up_proj_scales".into(), vec![32, 5760, 90]));
        mxfp4.push(("layer.0.mlp.experts.down_proj_blocks".into(), vec![32, 2880, 90, 16]));
        mxfp4.push(("layer.0.mlp.experts.down_proj_scales".into(), vec![32, 2880, 90]));

        let mut bf16 = common;
        bf16.push(("layer.0.mlp.experts.gate_up_proj".into(), vec![32, 5760, 2880]));
        bf16.push(("layer.0.mlp.experts.down_proj".into(), vec![32, 2880, 2880]));

        for (what, pairs) in [("mxfp4", mxfp4), ("bf16", bf16)] {
            let observed = Observed::from_pairs(pairs);
            assert!(m.check(&observed).is_ok(), "{what}: {:?}", m.check(&observed));
        }
    }

    /// Every projection satisfies the checkpoint it implies.
    #[test]
    fn each_manifest_is_satisfied_by_the_checkpoint_it_implies() {
        for f in [f20b(), GptOssFacts { layers: 36, experts: 128, ..f20b() }] {
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
        assert_eq!(d.shape.head_dim_kernel, 64, "64 is instantiated; nothing pads");
        assert_eq!(d.shape.gqa_group(), 8, "64 q over 8 kv");
        assert_eq!(d.shape.intermediate, 0, "no dense block anywhere in the stack");
        assert_eq!(d.shape.moe_intermediate, 2880, "one expert's width");
        assert_eq!(d.shape.widest_mlp(), 2880, "what the shared workspace must hold");
        assert_eq!(d.norm_eps, 1e-5, "gpt-oss's own, and not the llama lineage's 1e-6");
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
        assert!(d.scales.is_empty(), "the swiglu clamp is a kernel, not a scalar");
        assert!(!d.shares_kv());
    }

    /// The tracer's window list and the deployment's window table are
    /// one rule expanded twice, so they agree by construction.
    #[cfg(feature = "forward")]
    #[test]
    fn the_binding_facts_carry_the_same_schedule() {
        let f = f20b();
        let cuda = cuda_facts(&f, Deployed::single());
        assert_eq!(cuda.window_left, deployment(&f, 150_000.0, 1e-5, 128).windows());
        assert_eq!(cuda.mxfp4_decode_max_routes, 32 * f.experts);
        assert!(cuda.mxfp4_decode_gemv);
        assert!(!cuda.streamed_experts);
    }

    /// The text exists for every class a fire can carry — which it did
    /// not, once: gpt-oss had a facts row and only a Prefill arm, so a
    /// checkpoint loaded, reported itself healthy and died at its first
    /// decode.
    #[cfg(feature = "forward")]
    #[test]
    fn every_fire_class_traces() {
        use model_compiler::trace::FireClass;
        for class in [FireClass::Decode, FireClass::Prefill] {
            let plan = trace(&f20b(), class, Deployed::single());
            assert!(!plan.ops.is_empty(), "{class:?} traced nothing");
        }
    }
}

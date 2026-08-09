//! The three projections a Kimi K2 row makes: its tensor manifest, its
//! `Deployment`, and its traced text.
//!
//! Mirrors `shared/llama_like/project.rs`, and differs from it in the
//! one place this lineage differs: the attention is MULTI-HEAD LATENT,
//! so the tensors are not `{q,k,v}_proj` and the cache row is not a
//! head-split key. Every extent below is still the row's own arithmetic
//! — that is what makes a manifest a CHECK rather than a second
//! statement of the numbers.

use crate::deployment::{
    Advertised, AttnOutput, Deployment, Geometry, KvStyle, LayerAttention, NormPlacement,
    PrefillStyle, Refusal,
};
use crate::manifest::{Manifest, TensorSpec};

use super::spec::KimiFacts;

/// Does this build provision the KV store a deployment asks for?
///
/// A property of the BINARY, not of any checkpoint — which is why it is
/// stated here beside the projection rather than carried on a row. This
/// is where `unbuilt_kv_store()` went. That method existed because a
/// family could hold a `FACTS_ROWS` row, load happily, report itself
/// healthy, and die at its first fire inside a walk; it answered with a
/// STRING, and the caller then decided which store the string meant by
/// asking whether it contained `"compress"`.
///
/// The row states its [`KvStyle`] outright now, this asks one question
/// of the enum, and a store that gets built is one arm changing here.
/// `gpu::pools::mla_cache` is ported and waiting; until an executor arm
/// names an MLA dispatch there is nothing to point it at.
#[must_use]
pub fn kv_store_is_built(kv: &KvStyle) -> bool {
    kv.has_a_store_in_this_build()
}

/// This row's tensors.
///
/// The MLA rows are the interesting ones, and they are interesting
/// because the latent ranks are the only thing that makes them
/// checkable. `kv_b_proj` is `[heads * (qk_nope_head_dim + v_head_dim),
/// kv_lora_rank]` because that is what a COMPRESSED kv means: one
/// `kv_lora_rank`-wide latent per token is read back into every head's
/// nope half and every head's value. A checkpoint whose `kv_b_proj` has
/// some other number of rows is not this row's model, whatever its
/// `config.json` says its `model_type` is.
#[must_use]
pub fn manifest(f: &KimiFacts, tied_embeddings: bool) -> Manifest {
    let (hidden, vocab) = (u64::from(f.hidden), u64::from(f.vocab));
    let a = &f.attn;
    // The query's own latent is OPTIONAL in this lineage: DeepSeek-V2-Lite
    // and Kimi-Linear project the query straight from the residual, the
    // rest route it through a rank. Presence of `q_a_proj` versus
    // `q_proj` is how a checkpoint says which, and it is a discriminator
    // rather than a branch because the two cannot both be published.
    let latent_q = a.q_lora_rank > 0;
    let q_lora = u64::from(a.q_lora_rank);
    let kv_lora = u64::from(a.kv_lora_rank);
    let q_b_width = u64::from(a.q_b_width());
    let kv_a_width = u64::from(a.kv_a_width());
    // What the latent is read back OUT to: every head's nope half and
    // every head's value, which is the one extent that states the
    // compression ratio.
    let kv_b_width = u64::from(a.heads * (a.qk_nope_head_dim + a.v_head_dim));
    let v_width = u64::from(a.v_width());
    let dense_inter = u64::from(f.dense_intermediate);
    let has_dense_prefix = f.dense_layers > 0;
    let all_dense = f.dense_layers >= f.layers;

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
        // ── The latent attention, as arithmetic ──────────────────────
        .either(latent_q, "layer.{}.self_attn.q_a_proj", [q_lora, hidden])
        .either(latent_q, "layer.{}.self_attn.q_b_proj", [q_b_width, q_lora])
        // The straight-projection alternative, forbidden when there is a
        // rank. One of these two rows is `Absent` for every row of this
        // lineage, so no checkpoint can satisfy both spellings.
        .either(!latent_q, "layer.{}.self_attn.q_proj", [q_b_width, hidden])
        .with_if(
            latent_q,
            TensorSpec::required("layer.{}.self_attn.q_a_layernorm", [q_lora]),
        )
        .with(TensorSpec::required(
            "layer.{}.self_attn.kv_a_proj_with_mqa",
            [kv_a_width, hidden],
        ))
        .with(TensorSpec::required(
            "layer.{}.self_attn.kv_a_layernorm",
            [kv_lora],
        ))
        .with(TensorSpec::required(
            "layer.{}.self_attn.kv_b_proj",
            [kv_b_width, kv_lora],
        ))
        // `o_proj` reads the VALUE width and not the query width, which
        // is where MLA stops looking like GQA: the two differ whenever
        // `v_head_dim != qk_nope_head_dim + qk_rope_head_dim`.
        .with(TensorSpec::required(
            "layer.{}.self_attn.o_proj",
            [hidden, v_width],
        ))
        // ── The dense prefix, as tensors ─────────────────────────────
        //
        // `first_k_dense_replace` is a fact a checkpoint publishes: a
        // stack with a prefix ships a dense MLP (from its leading
        // layers) AND a router (from the rest), and every logical name
        // collapses over the stack, so both appear. A mixture with no
        // prefix ships no dense MLP at all, and one that is dense all
        // the way ships no router — which is the same statement read
        // from its two ends.
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
        // The ROUTER is the mixture's identity, and it is the row that
        // can be measured: `[num_experts, hidden]`, never quantized,
        // always spelled `.weight`. The routed BANK is deliberately not
        // named — this family's experts ship as bf16 `.weight` in one
        // upload and as W4A16 `.weight_packed` in another, and a
        // manifest that named either would be keying identity on an
        // ENCODING, which is exactly what the catalog divides out.
        .either(
            !all_dense,
            "layer.{}.mlp.gate",
            [u64::from(f.moe.num_experts), hidden],
        )
        .with_if(
            f.moe.has_shared_expert(),
            TensorSpec::required(
                "layer.{}.mlp.shared_experts.gate_proj",
                [u64::from(f.moe.shared_intermediate), hidden],
            ),
        )
}

/// This row's deployment, or a refusal at the DOOR.
///
/// A projection, and every value in it was already in the row. What the
/// derivation this replaces did instead was re-read `kv_lora_rank` and
/// `qk_rope_head_dim` off a resident `config.json` to fill the very
/// `KvStyle::Mla` the facts row already held — two readings of one
/// document, which is the defect the whole table exists to remove.
///
/// The `Result` is the other half of the point. A row that can be
/// LOADED and cannot be SERVED says so here, once, at the door — rather
/// than being absent from a second table and dying at its first fire
/// inside a walk, which is what an MLA family did.
///
/// `advertised` arrives from the ROW rather than being derived here,
/// because none of its three answers is a shape: a family label, a
/// published context ceiling and whether an encoder ships are facts
/// about a checkpoint, and this function sees only geometry. Deriving
/// them here is exactly what `driver-cuda` did when it copied
/// `hf.model_type` and `hf.max_position_embeddings` off a resident
/// `HfConfig` at load.
///
/// # Errors
///
/// [`Refusal::Unsupported`] when this build provisions no store for the
/// row's [`KvStyle`].
pub fn deployment(
    f: &KimiFacts,
    rope_theta: f32,
    norm_eps: f32,
    advertised: Advertised,
) -> Result<Deployment, Refusal> {
    let planned = plan(f, rope_theta, norm_eps, advertised);
    if kv_store_is_built(&planned.kv) {
        Ok(planned)
    } else {
        Err(Refusal::Unsupported(
            "this build provisions no MLA paged store; the row's compressed \
             latent KV has nowhere to live",
        ))
    }
}

/// The projection itself, which is TOTAL: a row's deployment is a fact
/// about the row and exists whether or not this build can serve it.
///
/// Separate from [`deployment`] so the two statements stay separable —
/// "what this model needs" is the row's, "what this binary provides" is
/// [`kv_store_is_built`]'s, and collapsing them is how a capability
/// question turns back into a family name.
#[must_use]
fn plan(f: &KimiFacts, rope_theta: f32, norm_eps: f32, advertised: Advertised) -> Deployment {
    let a = &f.attn;
    // MLA's page row holds the LATENT plus the one shared rope half —
    // `kv_a_width()` — and not a head-split key. That is what every MLA
    // family's `head_dim_of` answered, said once by the shared geometry.
    let page_row = a.kv_a_width();
    // The scale is over the DOT's width, which is the query's
    // `nope + rope` and not the page row: a latent is what is STORED,
    // `qk_head_dim` is what is MULTIPLIED. The vtable this replaces took
    // `1/sqrt(head_dim_of(l))` — the page row — for every family that
    // did not override it.
    let sm_scale = 1.0 / (a.qk_head_dim() as f32).sqrt();
    let attention = (0..f.layers)
        .map(|l| LayerAttention {
            // One shape for every layer, which is what this row was
            // already saying by having no per-layer count.
            kv_heads: 1,
            head_dim: page_row,
            // Kimi K2 attends the whole context; a sliding window is
            // DeepSeek-V4's, and it says so on its own row.
            window: -1,
            // Every layer owns its pages.
            kv_source: l,
            sm_scale,
            rope_theta,
            // Only the rope half rotates; the nope half is carried
            // straight through, which is what makes the latent cacheable.
            rotary_dim: a.qk_rope_head_dim,
        })
        .collect();

    Deployment {
        layers: f.layers,
        norm_eps,
        // The row's own numbers, so a LAUNCH and a TRACE cannot disagree
        // about how many heads there are.
        shape: Geometry {
            hidden: f.hidden,
            q_heads: a.heads,
            // ONE, and it is not a rounding: MLA reads a single latent
            // plane per token, so every query head shares it. The GQA
            // ratio that follows is `heads`, which is the honest
            // statement of what an MLA decode would have to instantiate.
            kv_heads: 1,
            head_dim: page_row,
            // Nothing pads it. A latent row is not a head width a kernel
            // is instantiated at, and the store that would hold it is
            // not built here anyway — see [`kv_store_is_built`].
            head_dim_kernel: page_row,
            intermediate: f.dense_intermediate,
            // The mixture's inner width is a DIFFERENT number from the
            // dense prefix's, and the forward workspace is one buffer
            // both layer kinds share — so both are stated and the
            // planner takes the wider.
            moe_intermediate: f.moe.moe_intermediate,
            experts_per_token: f.moe.top_k,
            shared_intermediate: f.moe.shared_intermediate,
            vocab: f.vocab,
        },
        attention,
        // STATED, not sniffed. The derivation this replaces chose
        // between `Mla` and `Dsv4` by asking whether a family's NAME
        // contained `"compress"`, and then read the ranks back off a
        // resident `config.json`.
        kv: KvStyle::Mla {
            kv_lora_rank: a.kv_lora_rank,
            qk_rope_head_dim: a.qk_rope_head_dim,
        },
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
        k_eq_v: false,
        norm_topk_prob: f.moe.norm_topk_prob,
        routed_scaling: f.moe.routed_scaling,
        mlp_gate: crate::deployment::MlpGate::Silu,
        scales: std::collections::BTreeMap::new(),
        // From the ROW, not from the shape: a family label and a
        // published context ceiling are facts about a checkpoint, and a
        // projection only sees geometry. Carried through untouched — a
        // projection that edited the label would be inventing one.
        advertised,
        rope_scaling: None,
        towers: Default::default(),
    }
}

/// The CUDA binding facts for this row.
///
/// `q_kv_a_fused` is `true` because the CONTRACT stages the fused bank:
/// `q_a_proj` and `kv_a_proj_with_mqa` share their input, so the
/// authoring pass joins them for every layer that publishes both. The
/// derivation this replaces asked the loaded checkpoint
/// (`alias("layer.0.q_kv_a_fused").is_some()`) — the same answer,
/// obtained by probing the result of a decision this crate made.
#[cfg(feature = "forward")]
#[must_use]
pub fn cuda_facts(rope_yarn_original: bool) -> super::forward::facts::KimiCudaFacts {
    super::forward::facts::KimiCudaFacts {
        q_kv_a_fused: true,
        rope_yarn_original,
    }
}

/// Why this build has no Metal text for a kimi-k2 row.
///
/// A `const` so the test that asserts the refusal NAMES the missing
/// thing compares against the same string the caller is shown, rather
/// than against a paraphrase that can drift away from it — the shape
/// `csm::project::NO_TRACE` set for the same reason.
///
/// Its forward is `kimi_cuda`: MLA over a compressed latent,
/// one planned dispatch per fire, with YaRN on the decoupled rotary
/// half. `llama_like_metal` states dense paged attention over full
/// K and V, which is a different cache and a different kernel.
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
pub const NO_METAL: &str = "kimi-k2 has no Metal text in this build: its forward is `kimi_cuda` — latent \
     attention over a compressed KV with YaRN on the decoupled rotary half — and \
     the one Metal text here (`llama_like_metal`) states dense paged attention \
     over full K and V, a different cache reached through a different shape; the \
     CUDA backend serves this row";

/// Trace this row's CUDA text for one fire class.
#[cfg(feature = "forward")]
#[must_use]
pub fn trace(
    f: &KimiFacts,
    rope_yarn_original: bool,
    class: model_compiler::trace::FireClass,
) -> model_compiler::trace::ForwardPlan {
    super::forward::kimi_cuda(f, &cuda_facts(rope_yarn_original), class)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::Presence;

    fn k2() -> KimiFacts {
        KimiFacts::kimi_k2()
    }

    /// What the row advertises rides through UNTOUCHED.
    ///
    /// It is carried rather than derived because the derivation it
    /// replaces read `model_type` and `max_position_embeddings` off a
    /// resident `HfConfig` at load — two more readings of a document
    /// this table exists to stop re-reading. A projection that rewrote
    /// either would be inventing a fact about a checkpoint out of its
    /// geometry.
    #[test]
    fn the_rows_advertised_label_is_carried_and_not_rewritten() {
        let stated = Advertised {
            arch: "kimi_k2",
            max_model_len: 131_072,
            media_encode: false,
        };
        let d = plan(&k2(), 50_000.0, 1e-6, stated.clone());
        assert_eq!(
            d.advertised, stated,
            "a projection that edits the label is inventing one"
        );
    }

    fn spec(m: &Manifest, name: &str) -> TensorSpec {
        m.tensors
            .iter()
            .find(|t| t.name == name)
            .unwrap_or_else(|| panic!("the manifest states {name}"))
            .clone()
    }

    /// THE claim of an MLA manifest: every attention extent is the
    /// latent rank's arithmetic, so a checkpoint that compresses to a
    /// different rank cannot match this row however its config reads.
    #[test]
    fn the_mla_extents_are_the_latent_ranks_arithmetic() {
        let f = k2();
        let m = manifest(&f, false);
        let a = &f.attn;
        assert_eq!(a.q_lora_rank, 1536);
        assert_eq!(a.kv_lora_rank, 512);

        // The query's rank: hidden -> 1536 -> every head's nope+rope.
        assert_eq!(
            spec(&m, "layer.{}.self_attn.q_a_proj").extents,
            vec![1536, 7168]
        );
        assert_eq!(
            spec(&m, "layer.{}.self_attn.q_b_proj").extents,
            vec![64 * (128 + 64), 1536],
        );
        assert_eq!(
            spec(&m, "layer.{}.self_attn.q_a_layernorm").extents,
            vec![1536]
        );

        // The KV's rank: hidden -> 512 latent + 64 shared rope, then the
        // latent alone back out to every head's nope half and value.
        assert_eq!(
            spec(&m, "layer.{}.self_attn.kv_a_proj_with_mqa").extents,
            vec![512 + 64, 7168],
        );
        assert_eq!(
            spec(&m, "layer.{}.self_attn.kv_a_layernorm").extents,
            vec![512]
        );
        assert_eq!(
            spec(&m, "layer.{}.self_attn.kv_b_proj").extents,
            vec![64 * (128 + 128), 512],
        );
        // And the output reads the VALUE width, which is where MLA
        // stops looking like GQA.
        assert_eq!(
            spec(&m, "layer.{}.self_attn.o_proj").extents,
            vec![7168, 64 * 128]
        );
    }

    /// A query rank is not a branch, it is a DISCRIMINATOR: the two
    /// spellings forbid each other, so no checkpoint can satisfy both
    /// and a row cannot silently accept the wrong one.
    #[test]
    fn a_query_latent_forbids_the_straight_projection() {
        let m = manifest(&k2(), false);
        assert_eq!(
            spec(&m, "layer.{}.self_attn.q_a_proj").presence,
            Presence::Required
        );
        assert_eq!(
            spec(&m, "layer.{}.self_attn.q_proj").presence,
            Presence::Absent
        );

        let mut lite = k2();
        lite.attn.q_lora_rank = 0;
        let m = manifest(&lite, false);
        assert_eq!(
            spec(&m, "layer.{}.self_attn.q_proj").presence,
            Presence::Required
        );
        assert_eq!(
            spec(&m, "layer.{}.self_attn.q_a_proj").presence,
            Presence::Absent
        );
        assert_eq!(
            spec(&m, "layer.{}.self_attn.q_b_proj").presence,
            Presence::Absent
        );
        assert!(
            !m.tensors.iter().any(|t| t.name.ends_with("q_a_layernorm")),
            "there is no rank to norm",
        );
    }

    /// `first_k_dense_replace` as tensors. A stack with a prefix
    /// publishes BOTH a dense MLP and a router; one with none publishes
    /// no dense MLP; one that never leaves the prefix publishes no
    /// router. The manifest says all three without reading a config.
    #[test]
    fn the_dense_prefix_shows_up_as_both_kinds_of_block() {
        let with_prefix = manifest(&k2(), false);
        assert_eq!(
            spec(&with_prefix, "layer.{}.mlp.gate_proj").presence,
            Presence::Required
        );
        assert_eq!(
            spec(&with_prefix, "layer.{}.mlp.gate_proj").extents,
            vec![18_432, 7168],
            "the DENSE inner width, which the mixture's 2048 is not",
        );
        assert_eq!(
            spec(&with_prefix, "layer.{}.mlp.gate").presence,
            Presence::Required
        );
        assert_eq!(
            spec(&with_prefix, "layer.{}.mlp.gate").extents,
            vec![384, 7168]
        );

        let mut no_prefix = k2();
        no_prefix.dense_layers = 0;
        let m = manifest(&no_prefix, false);
        assert_eq!(
            spec(&m, "layer.{}.mlp.gate_proj").presence,
            Presence::Absent
        );
        assert_eq!(spec(&m, "layer.{}.mlp.gate").presence, Presence::Required);

        let mut all_dense = k2();
        all_dense.dense_layers = all_dense.layers;
        let m = manifest(&all_dense, false);
        assert_eq!(
            spec(&m, "layer.{}.mlp.gate_proj").presence,
            Presence::Required
        );
        assert_eq!(spec(&m, "layer.{}.mlp.gate").presence, Presence::Absent);
    }

    /// The mixture is stated by its ROUTER, not by its bank: this family
    /// ships the same experts as bf16 `.weight` and as W4A16
    /// `.weight_packed`, and those are two encodings of one model.
    #[test]
    fn the_mixture_is_stated_by_the_router_and_not_by_an_encoding() {
        let m = manifest(&k2(), false);
        assert!(
            !m.tensors
                .iter()
                .any(|t| t.name.contains("packed") || t.name.contains("experts.0")),
            "a manifest that named the bank would be keying identity on a packing",
        );
        assert_eq!(spec(&m, "layer.{}.mlp.gate").extents, vec![384, 7168]);
        assert_eq!(
            spec(&m, "layer.{}.mlp.shared_experts.gate_proj").extents,
            vec![2048, 7168],
        );
    }

    /// A tie is an ABSENCE, and it is the only thing that separates a
    /// tied row from an untied one when every extent agrees.
    #[test]
    fn a_tie_is_an_absence_the_manifest_expects() {
        assert_eq!(
            spec(&manifest(&k2(), true), "lm_head").presence,
            Presence::Absent
        );
        assert_eq!(
            spec(&manifest(&k2(), false), "lm_head").presence,
            Presence::Required
        );
    }

    /// The KV shape is STATED by the row. The derivation this replaces
    /// picked between `Mla` and `Dsv4` by asking whether a string
    /// contained `"compress"`, and then read the ranks back off a
    /// resident `config.json`.
    #[test]
    fn the_kv_style_is_stated_rather_than_matched_on_a_substring() {
        let d = plan(&k2(), 50_000.0, 1e-6, Advertised::default());
        assert_eq!(
            d.kv,
            KvStyle::Mla {
                kv_lora_rank: 512,
                qk_rope_head_dim: 64
            }
        );
        assert!(
            !kv_store_is_built(&d.kv),
            "no executor arm names an MLA dispatch"
        );
        assert!(kv_store_is_built(&KvStyle::Paged));
        assert!(!kv_store_is_built(&KvStyle::Dsv4 { ratios: vec![1] }));
    }

    /// Loadable and not servable, said at the DOOR. This is why
    /// `deployment` returns a `Result` at all: the MLA lineage
    /// registered in `FACTS_ROWS`, answered `facts_from_hf` happily,
    /// reported itself healthy, and had no forward path — so it died at
    /// its first fire, inside a walk, with the model already resident.
    #[test]
    fn a_row_with_no_store_in_this_build_refuses_before_the_first_fire() {
        let refusal = deployment(&k2(), 50_000.0, 1e-6, Advertised::default())
            .expect_err("no MLA store here");
        assert!(
            matches!(refusal, Refusal::Unsupported(what) if what.contains("MLA")),
            "the refusal names what is missing rather than saying `Unsupported`: {refusal}"
        );
        // And the refusal is about the BUILD, not about the row: the
        // projection is total and still says what the model needs.
        assert_eq!(
            plan(&k2(), 50_000.0, 1e-6, Advertised::default()).layers,
            61
        );
    }

    /// The launch geometry is the row's own numbers, and the two that
    /// are not obvious are stated: one latent plane per token, and a
    /// page row that is the latent plus the shared rope half.
    #[test]
    fn the_launch_geometry_is_the_rows_own_numbers() {
        let f = k2();
        let d = plan(&f, 50_000.0, 1e-6, Advertised::default());
        assert_eq!(d.layers, 61);
        assert_eq!(d.norm_eps, 1e-6);
        assert_eq!(d.attention.len(), 61);
        assert_eq!(d.shape.hidden, 7168);
        assert_eq!(d.shape.q_heads, 64);
        assert_eq!(d.shape.kv_heads, 1);
        assert_eq!(d.shape.head_dim, 512 + 64);
        assert_eq!(d.shape.head_dim_alloc(), 512 + 64);
        assert_eq!(
            d.shape.gqa_group(),
            64,
            "every head reads the one latent plane"
        );
        assert_eq!(d.shape.intermediate, 18_432);
        assert_eq!(d.shape.moe_intermediate, 2048);
        assert_eq!(
            d.shape.widest_mlp(),
            18_432,
            "the dense prefix is the wider block here"
        );
        assert_eq!(d.shape.vocab, 163_840);
        assert_eq!(d.decode_head_dims(), None, "one kind of layer");
        assert!(!d.shares_kv());
        assert!(d.recurrent.is_none());
        assert!(d.theta_by_layer().is_empty(), "one theta serves the stack");
        assert_eq!(d.prefill, PrefillStyle::Planned);
        assert_eq!(d.attn_output, AttnOutput::DriverPinned);
        assert_eq!(d.norm, NormPlacement::Pre);
        assert_eq!(d.logit_softcap, 0.0);
        assert_eq!(d.ple_dim, 0);
        assert!(d.scales.is_empty());
    }

    /// The softmax scale is over what is MULTIPLIED, not over what is
    /// stored. A latent page row is 576 wide here and the dot is 192.
    #[test]
    fn the_scale_is_over_the_dot_and_not_over_the_page_row() {
        let f = k2();
        let d = plan(&f, 50_000.0, 1e-6, Advertised::default());
        let want = 1.0 / (192.0f32).sqrt();
        for a in &d.attention {
            assert!((a.sm_scale - want).abs() < 1e-6, "{}", a.sm_scale);
            assert_eq!(a.head_dim, 576, "the page row is the latent plus rope");
            assert_eq!(a.rotary_dim, 64, "only the rope half rotates");
            assert_eq!(a.window, -1);
            assert_eq!(a.rope_theta, 50_000.0);
        }
    }

    /// The binding facts, which are about the LOAD and not the model:
    /// the contract stages the fused latent bank, so the trace reads one
    /// GEMM where the split binding reads two.
    #[cfg(feature = "forward")]
    #[test]
    fn the_binding_facts_say_what_the_contract_staged() {
        let cuda = cuda_facts(true);
        assert!(
            cuda.q_kv_a_fused,
            "the authoring pass joins the two latents"
        );
        assert!(cuda.rope_yarn_original);
        assert!(!cuda_facts(false).rope_yarn_original);
    }

    /// Every fire class traces, which is the aspect answer a row owes
    /// even though this build refuses to serve it.
    #[cfg(feature = "forward")]
    #[test]
    fn every_fire_class_traces() {
        use model_compiler::trace::FireClass;
        for class in [FireClass::Prefill, FireClass::Decode] {
            let plan = trace(&k2(), true, class);
            assert!(!plan.ops.is_empty(), "{class:?} traced nothing");
        }
    }
}

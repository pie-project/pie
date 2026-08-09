//! The three projections a GLM-5 row makes: its tensor manifest, its
//! `Deployment`, and its traced text.
//!
//! Mirrors `kimi_k2/project.rs`, and differs from it in the one place
//! this lineage differs: beside the MLA there is a DSA INDEXER — a
//! second, smaller attention whose only output is a top-k page mask.
//! Every extent below is still the row's own arithmetic; that is what
//! makes a manifest a CHECK rather than a second statement of the
//! numbers.

use crate::deployment::{
    Advertised, AttnOutput, Deployment, Geometry, KvStyle, LayerAttention, NormPlacement,
    PrefillStyle, Refusal,
};
use crate::manifest::{Manifest, TensorSpec};

use super::spec::Glm5Facts;

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
///
/// # What is deliberately NOT named
///
/// **The DSA indexer's three projections.** `forward` names them
/// `idx_wq_b`, `idx_wk` and `idx_weights_proj`, and those are TRACE
/// names this crate invented — `weight_names::wire` has no glm-5 row, and
/// `tests/seam_names.rs` records all three as names no builder can yet
/// emit. So the checkpoint's own spelling for them is not written down
/// anywhere in this tree, and a manifest row is a CHECK: naming a
/// tensor whose spelling is a guess turns a matching checkpoint into a
/// `Fault::Missing` and the row into one nothing can satisfy. The
/// extents that ARE named already separate this row from every other
/// MLA row in the catalog — see the test that holds it against kimi-k2.
#[must_use]
pub fn manifest(f: &Glm5Facts, tied_embeddings: bool) -> Manifest {
    let (hidden, vocab) = (u64::from(f.hidden), u64::from(f.vocab));
    let a = &f.attn;
    // The query's own latent is OPTIONAL in this lineage: DeepSeek-V2-Lite
    // projects the query straight from the residual, the rest route it
    // through a rank. Presence of `q_a_proj` versus `q_proj` is how a
    // checkpoint says which, and it is a discriminator rather than a
    // branch because the two cannot both be published.
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
        // STATED absent rather than left out when there is no rank: a
        // rank that is not there has no norm over it, and a manifest
        // that omits the row cannot refuse a checkpoint that ships one.
        .either(latent_q, "layer.{}.self_attn.q_a_layernorm", [q_lora])
        .with(TensorSpec::required(
            "layer.{}.self_attn.kv_a_proj_with_mqa",
            [kv_a_width, hidden],
        ))
        .with(TensorSpec::required(
            "layer.{}.self_attn.kv_a_layernorm",
            [kv_lora],
        ))
        // The one tensor this generation's contract DEQUANTIZES: it
        // ships FP8 and `kimi_mla` reads bf16. The extents are the same
        // either way, which is the point — a manifest states the model
        // and not the encoding it arrived in.
        .with(TensorSpec::required(
            "layer.{}.self_attn.kv_b_proj",
            [kv_b_width, kv_lora],
        ))
        // `o_proj` reads the VALUE width and not the query width, which
        // is where MLA stops looking like GQA: the two differ whenever
        // `v_head_dim != qk_nope_head_dim + qk_rope_head_dim`, and for
        // this row they differ by 64 per head.
        .with(TensorSpec::required(
            "layer.{}.self_attn.o_proj",
            [hidden, v_width],
        ))
        // ── The dense prefix, as tensors ─────────────────────────────
        //
        // `first_k_dense_replace` is a fact a checkpoint publishes: a
        // stack with a prefix ships a dense MLP (from its leading three
        // layers) AND a router (from the other 43), and every logical
        // name collapses over the stack, so both appear. A mixture with
        // no prefix ships no dense MLP at all, and one that is dense all
        // the way ships no router — the same statement read from its two
        // ends.
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
        // named — this generation ships bf16 `.weight` per expert in one
        // upload and FP8 in another, and a manifest that named either
        // would be keying identity on an ENCODING, which is exactly what
        // the catalog divides out.
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
/// The `Result` is the other half of the point. A row that can be LOADED
/// and cannot be SERVED says so here, once, at the door — rather than
/// being absent from a second table and dying at its first fire inside a
/// walk, which is what this generation did.
///
/// `advertised` arrives from the ROW as an argument rather than being
/// derived here, because none of its three answers is a shape: an arch
/// label is a coarse family name a guest program matches on, a context
/// ceiling is a training-time fact two checkpoints of identical geometry
/// can disagree about, and whether a tower ships is a question about the
/// checkpoint's files. This function sees geometry and nothing else.
///
/// It is a PARAMETER and not a field the caller writes afterwards. The
/// caller used to project, `?` the refusal, and only then assign the
/// label -- which put the assignment past a refusal this build always
/// takes, so the row's label was carried by no code that ran. Passing it
/// in is what makes the projection total in fact and not only in the
/// sentence above.
///
/// # Errors
///
/// [`Refusal::Unsupported`] when this build provisions no store for the
/// row's [`KvStyle`].
pub fn deployment(
    f: &Glm5Facts,
    rope_theta: f32,
    norm_eps: f32,
    advertised: Advertised,
) -> Result<Deployment, Refusal> {
    let planned = plan(f, rope_theta, norm_eps, advertised);
    planned.provisioned()
}

/// The projection itself, which is TOTAL: a row's deployment is a fact
/// about the row and exists whether or not this build can serve it.
///
/// Separate from [`deployment`] so the two statements stay separable —
/// "what this model needs" is the row's, "what this binary provides" is
/// [`KvStyle::has_a_store_in_this_build`]'s, and collapsing them is how a capability
/// question turns back into a family name.
#[must_use]
fn plan(f: &Glm5Facts, rope_theta: f32, norm_eps: f32, advertised: Advertised) -> Deployment {
    let a = &f.attn;
    // MLA's page row holds the LATENT plus the one shared rope half —
    // `kv_a_width()` — and not a head-split key. That is what this
    // generation's `head_dim_of` answered, said once by the shared
    // geometry.
    let page_row = a.kv_a_width();
    // The scale is over the DOT's width, which is the query's
    // `nope + rope` (192 here) and not the page row (576): a latent is
    // what is STORED, `qk_head_dim` is what is MULTIPLIED. The vtable
    // this replaces took `1/sqrt(head_dim_of(l))` — the page row — for
    // every family that did not override it, which is a scale 1.73x too
    // small on this row.
    let sm_scale = 1.0 / (a.qk_head_dim() as f32).sqrt();
    let attention = (0..f.layers)
        .map(|l| LayerAttention {
            // One shape for every layer, which is what this row was
            // already saying by having no per-layer count.
            kv_heads: 1,
            head_dim: page_row,
            // GLM-5 attends the whole context and SPARSIFIES it with the
            // DSA mask instead of windowing: the indexer's top-k is not
            // a sliding window and a driver that read it as one would
            // drop the far pages the mask exists to keep.
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
            // not built here anyway — see [`KvStyle::has_a_store_in_this_build`].
            head_dim_kernel: page_row,
            intermediate: f.dense_intermediate,
            // The mixture's inner width is a DIFFERENT number from the
            // dense prefix's — 1408 against 10944 — and the forward
            // workspace is one buffer both layer kinds share, so both
            // are stated and the planner takes the wider.
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
        // GLM publishes `true` where its DeepSeek-shaped siblings
        // publish `false` -- see `Glm5MoeFacts`.
        norm_topk_prob: f.moe.norm_topk_prob,
        routed_scaling: f.moe.routed_scaling,
        mlp_gate: crate::deployment::MlpGate::Silu,
        scales: std::collections::BTreeMap::new(),
        // Carried, unread. None of the three answers in here is
        // geometry: an arch label is a coarse family name, a context
        // ceiling is a training-time fact, and whether a tower ships is
        // a question about the checkpoint's files. A projection that
        // filled them from what it can see would be deriving a family
        // name from a shape, which is the inference that put
        // `Gemma4ForConditionalGeneration` in a table row it did not
        // belong in.
        advertised,
        // Unscaled, because nothing in this tree says otherwise:
        // `synthetic--glm-moe-dsa.json` states no `rope_scaling` block at
        // all, and the ladder is used as written. A row that invented a
        // YaRN factor here would lengthen every position by a ratio no
        // published config backs.
        rope_scaling: None,
        towers: Default::default(),
    }
}

/// Why this build has no Metal text for a glm-5 row.
///
/// A `const` so the test that asserts the refusal NAMES the missing
/// thing compares against the same string the caller is shown, rather
/// than against a paraphrase that can drift away from it — the shape
/// `csm::project::NO_TRACE` set for the same reason.
///
/// Its forward is `glm5_cuda`: MLA attention plus the DSA
/// indexer, a sparse-attention selection pass with no counterpart
/// anywhere in `llama_like_metal`, which serves dense paged attention
/// only.
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
pub const NO_METAL: &str = "glm-5 has no Metal text in this build: its forward is `glm5_cuda` — latent \
     attention plus the DSA indexer that selects which keys each query scores — \
     and the one Metal text here (`llama_like_metal`) serves dense paged \
     attention over a `LlamaLikeFacts`, which is neither this attention nor \
     this shape; the CUDA backend serves this row";

/// Trace this row's CUDA text for one fire class.
///
/// No binding facts on the way in, unlike kimi-k2's: this generation's
/// text takes the fused `mla_prepare` unconditionally and states no rope
/// variant, so the shape is the whole input.
#[must_use]
pub fn trace(
    f: &Glm5Facts,
    class: model_compiler::trace::FireClass,
) -> model_compiler::trace::ForwardPlan {
    super::forward::glm5_cuda(f, class)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::Presence;

    fn glm5() -> Glm5Facts {
        Glm5Facts::glm5_106b_a12b()
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
        let f = glm5();
        let m = manifest(&f, false);

        // The query's rank: hidden -> 1536 -> every head's nope+rope.
        assert_eq!(
            spec(&m, "layer.{}.self_attn.q_a_proj").extents,
            vec![1536, 4096]
        );
        assert_eq!(
            spec(&m, "layer.{}.self_attn.q_b_proj").extents,
            vec![96 * (128 + 64), 1536],
        );
        assert_eq!(
            spec(&m, "layer.{}.self_attn.q_a_layernorm").extents,
            vec![1536]
        );

        // The KV's rank: hidden -> 512 latent + 64 shared rope, then the
        // latent alone back out to every head's nope half and value.
        assert_eq!(
            spec(&m, "layer.{}.self_attn.kv_a_proj_with_mqa").extents,
            vec![512 + 64, 4096],
        );
        assert_eq!(
            spec(&m, "layer.{}.self_attn.kv_a_layernorm").extents,
            vec![512]
        );
        assert_eq!(
            spec(&m, "layer.{}.self_attn.kv_b_proj").extents,
            vec![96 * (128 + 128), 512],
        );
        // And the output reads the VALUE width, which is where MLA
        // stops looking like GQA.
        assert_eq!(
            spec(&m, "layer.{}.self_attn.o_proj").extents,
            vec![4096, 96 * 128]
        );
    }

    /// A query rank is not a branch, it is a DISCRIMINATOR: the two
    /// spellings forbid each other, so no checkpoint can satisfy both
    /// and a row cannot silently accept the wrong one.
    #[test]
    fn a_query_latent_forbids_the_straight_projection() {
        let m = manifest(&glm5(), false);
        assert_eq!(
            spec(&m, "layer.{}.self_attn.q_a_proj").presence,
            Presence::Required
        );
        assert_eq!(
            spec(&m, "layer.{}.self_attn.q_proj").presence,
            Presence::Absent
        );

        let mut straight = glm5();
        straight.attn.q_lora_rank = 0;
        let m = manifest(&straight, false);
        assert_eq!(
            spec(&m, "layer.{}.self_attn.q_proj").presence,
            Presence::Required
        );
        assert_eq!(
            spec(&m, "layer.{}.self_attn.q_a_proj").presence,
            Presence::Absent
        );
        assert_eq!(
            spec(&m, "layer.{}.self_attn.q_a_layernorm").presence,
            Presence::Absent,
            "a rank that is not there has no norm over it",
        );
    }

    /// Tied and untied differ in one tensor's PRESENCE and in nothing
    /// else, because every extent agrees — which is why a manifest has
    /// to state absence rather than leaving the row out.
    #[test]
    fn tying_the_head_is_the_absence_of_a_table() {
        let untied = manifest(&glm5(), false);
        assert_eq!(spec(&untied, "lm_head").presence, Presence::Required);
        assert_eq!(spec(&untied, "lm_head").extents, vec![151_552, 4096]);

        let tied = manifest(&glm5(), true);
        assert_eq!(spec(&tied, "lm_head").presence, Presence::Absent);
    }

    /// The dense prefix and the mixture BOTH show, because logical names
    /// collapse over the stack: layers 0..3 publish the dense MLP and
    /// layers 3..46 publish the router, and the collapsed view holds
    /// every name any layer shipped.
    #[test]
    fn a_dense_prefix_publishes_both_an_mlp_and_a_router() {
        let f = glm5();
        let m = manifest(&f, false);
        assert_eq!(
            spec(&m, "layer.{}.mlp.gate_proj").extents,
            vec![10_944, 4096]
        );
        assert_eq!(
            spec(&m, "layer.{}.mlp.down_proj").extents,
            vec![4096, 10_944]
        );
        assert_eq!(spec(&m, "layer.{}.mlp.gate").presence, Presence::Required);
        assert_eq!(spec(&m, "layer.{}.mlp.gate").extents, vec![128, 4096]);
        assert_eq!(
            spec(&m, "layer.{}.mlp.shared_experts.gate_proj").extents,
            vec![1408, 4096],
        );
    }

    /// The two ends of that statement, and both are reachable: a stack
    /// with no prefix ships no dense MLP, and one that is dense all the
    /// way ships no router.
    #[test]
    fn no_prefix_means_no_dense_mlp_and_all_dense_means_no_router() {
        let mut no_prefix = glm5();
        no_prefix.dense_layers = 0;
        let m = manifest(&no_prefix, false);
        assert_eq!(
            spec(&m, "layer.{}.mlp.gate_proj").presence,
            Presence::Absent
        );
        assert_eq!(
            spec(&m, "layer.{}.mlp.down_proj").presence,
            Presence::Absent
        );
        assert_eq!(spec(&m, "layer.{}.mlp.gate").presence, Presence::Required);

        let mut all_dense = glm5();
        all_dense.dense_layers = all_dense.layers;
        let m = manifest(&all_dense, false);
        assert_eq!(
            spec(&m, "layer.{}.mlp.gate_proj").presence,
            Presence::Required
        );
        assert_eq!(
            spec(&m, "layer.{}.mlp.gate").presence,
            Presence::Absent,
            "a stack with no routed layer publishes no router, and a row \
             that required one would match nothing",
        );
    }

    /// A shared expert is a fact about the checkpoint, so its tensor is
    /// stated only when the row has one.
    #[test]
    fn a_shared_expert_is_named_only_when_the_row_has_one() {
        let mut none = glm5();
        none.moe.shared_intermediate = 0;
        let m = manifest(&none, false);
        assert!(
            !m.tensors.iter().any(|t| t.name.contains("shared_experts")),
            "a row with no shared expert states no tensor for one; a \
             `Required` row here would fail every checkpoint that has none",
        );
    }

    /// Two MLA generations state the same tensor NAMES, so what has to
    /// separate them is the EXTENTS — and this is the property the DSA
    /// indexer's absence from the manifest rests on. Written against the
    /// sibling's numbers as data rather than against the sibling's row,
    /// because a generation may not name one.
    #[test]
    fn a_stack_of_the_sibling_geometry_projects_a_different_manifest() {
        // kimi-k2's measurement: 61 layers of 7168 hidden, 64 heads, the
        // same 512/64 latent, 384 experts at 2048 behind an 18432-wide
        // dense prefix. Every NAME it publishes is one this row states.
        let sibling = Glm5Facts {
            layers: 61,
            vocab: 163_840,
            hidden: 7168,
            dense_intermediate: 18_432,
            dense_layers: 1,
            attn: super::super::spec::Glm5MlaFacts {
                hidden: 7168,
                heads: 64,
                q_lora_rank: 1536,
                kv_lora_rank: 512,
                qk_nope_head_dim: 128,
                qk_rope_head_dim: 64,
                v_head_dim: 128,
                output_gate: false,
            },
            dsa: super::super::spec::Glm5DsaFacts {
                index_n_heads: 0,
                index_head_dim: 0,
                index_topk: 0,
            },
            moe: super::super::spec::Glm5MoeFacts {
                hidden: 7168,
                num_experts: 384,
                top_k: 8,
                norm_topk_prob: true,
                routed_scaling: 2.5,
                moe_intermediate: 2048,
                shared_intermediate: 2048,
                aligned_block: 16,
            },
        };
        let mine = manifest(&glm5(), false);
        let theirs = manifest(&sibling, false);
        for name in [
            "embed_tokens",
            "layer.{}.self_attn.kv_b_proj",
            "layer.{}.self_attn.o_proj",
            "layer.{}.mlp.gate",
        ] {
            assert_ne!(
                spec(&mine, name).extents,
                spec(&theirs, name).extents,
                "{name} agrees across the two geometries, so one row's \
                 checkpoint would satisfy the other and `identify` would \
                 have to pick between them",
            );
        }
    }

    /// Every extent the manifest states is the row's own arithmetic and
    /// nothing else — no constant, no second reading of a config. Held
    /// by rebuilding each one from the shape.
    #[test]
    fn every_extent_is_recomputable_from_the_shape_alone() {
        let f = glm5();
        let a = &f.attn;
        let m = manifest(&f, false);
        let h = u64::from(f.hidden);
        for (name, want) in [
            ("embed_tokens", vec![u64::from(f.vocab), h]),
            ("norm", vec![h]),
            ("layer.{}.input_layernorm", vec![h]),
            ("layer.{}.post_attention_layernorm", vec![h]),
            (
                "layer.{}.self_attn.q_a_proj",
                vec![u64::from(a.q_lora_rank), h],
            ),
            (
                "layer.{}.self_attn.q_b_proj",
                vec![u64::from(a.q_b_width()), u64::from(a.q_lora_rank)],
            ),
            (
                "layer.{}.self_attn.kv_a_proj_with_mqa",
                vec![u64::from(a.kv_a_width()), h],
            ),
            (
                "layer.{}.self_attn.kv_a_layernorm",
                vec![u64::from(a.kv_lora_rank)],
            ),
            (
                "layer.{}.self_attn.kv_b_proj",
                vec![
                    u64::from(a.heads * (a.qk_nope_head_dim + a.v_head_dim)),
                    u64::from(a.kv_lora_rank),
                ],
            ),
            ("layer.{}.self_attn.o_proj", vec![h, u64::from(a.v_width())]),
            ("layer.{}.mlp.gate", vec![u64::from(f.moe.num_experts), h]),
            (
                "layer.{}.mlp.shared_experts.gate_proj",
                vec![u64::from(f.moe.shared_intermediate), h],
            ),
        ] {
            assert_eq!(
                spec(&m, name).extents,
                want,
                "{name} is not the row's own arithmetic"
            );
        }
        assert_eq!(
            m.layers, f.layers,
            "the manifest covers the row's own stack"
        );
    }

    /// The mixture is stated by its ROUTER and never by its bank: this
    /// generation ships the same experts as bf16 `.weight` and as FP8,
    /// and those are two encodings of one model.
    #[test]
    fn the_mixture_is_stated_by_the_router_and_not_by_an_encoding() {
        let m = manifest(&glm5(), false);
        assert!(
            !m.tensors
                .iter()
                .any(|t| t.name.contains("packed") || t.name.contains("experts.0")),
            "a manifest that named the expert bank would be keying identity \
             on a packing, which is exactly what the catalog divides out",
        );
    }

    /// The DSA indexer is deliberately unnamed, and the reason is worth
    /// holding: `weight_names::wire` has no glm-5 row, so the
    /// checkpoint's spelling for the three projections is nowhere in
    /// this tree. A guessed name is not a check — it is a row nothing
    /// can satisfy.
    #[test]
    fn the_indexer_is_left_unnamed_rather_than_guessed_at() {
        let m = manifest(&glm5(), false);
        assert!(
            !m.tensors
                .iter()
                .any(|t| t.name.contains("idx") || t.name.contains("indexer")),
            "the manifest names an indexer tensor whose spelling this tree \
             does not state; a checkpoint that has one under another name \
             would be refused for a name nobody wrote down",
        );
    }

    /// The deployment is the row's own numbers, field for field. A
    /// disagreement here is a launch that sizes a buffer from one
    /// reading and a trace that was built from another.
    #[test]
    fn the_geometry_is_the_rows_own_numbers() {
        let f = glm5();
        let d = plan(&f, 10_000.0, 1e-5, Advertised::default());
        assert_eq!(d.layers, 46);
        assert_eq!(d.attention.len(), 46);
        assert_eq!(d.shape.hidden, 4096);
        assert_eq!(d.shape.q_heads, 96);
        assert_eq!(
            d.shape.kv_heads, 1,
            "one latent plane, shared by every head"
        );
        assert_eq!(d.shape.head_dim, 576);
        assert_eq!(d.shape.head_dim_kernel, 576, "a latent row is never padded");
        assert_eq!(d.shape.intermediate, 10_944);
        assert_eq!(d.shape.moe_intermediate, 1408);
        assert_eq!(d.shape.vocab, 151_552);
        assert!(
            (d.norm_eps - 1e-5).abs() < f32::EPSILON,
            "the epsilon a row states is the one the norm runs at",
        );
        assert_eq!(d.ple_dim, 0);
        assert!(d.logit_softcap.abs() < f32::EPSILON);
        assert!(d.recurrent.is_none(), "no layer of this stack is recurrent");
        assert_eq!(d.norm, NormPlacement::Pre);
        assert_eq!(d.prefill, PrefillStyle::Planned);
        assert_eq!(d.attn_output, AttnOutput::DriverPinned);
        assert!(d.scales.is_empty());
    }

    /// The dense width and the expert width are BOTH stated, and the
    /// planner's `widest_mlp` is what reads them: sizing the shared
    /// workspace from the mixture alone would under-size it by 9536
    /// columns on this row and move the difference out of the KV pool.
    #[test]
    fn the_workspace_is_sized_from_the_wider_of_two_mlps() {
        let d = plan(&glm5(), 10_000.0, 1e-5, Advertised::default());
        assert_eq!(d.shape.widest_mlp(), 10_944);
    }

    /// The per-layer table: one entry per layer, and every one of them
    /// states the same six facts because this stack has no schedule.
    #[test]
    fn every_layer_attends_the_whole_context_through_its_own_pages() {
        let f = glm5();
        let d = plan(&f, 10_000.0, 1e-5, Advertised::default());
        let expected = 1.0 / (192.0f32).sqrt();
        for (l, a) in d.attention.iter().enumerate() {
            assert_eq!(a.head_dim, 576, "the page row is the latent plus rope");
            assert_eq!(a.window, -1, "DSA sparsifies, it does not window");
            assert_eq!(a.kv_source, l as u32, "every layer owns its pages");
            assert!((a.sm_scale - expected).abs() < f32::EPSILON);
            assert!((a.rope_theta - 10_000.0).abs() < f32::EPSILON);
            assert_eq!(a.rotary_dim, 64, "only the rope half rotates");
        }
    }

    /// The scale is over the DOT and not over the page row. Stated as
    /// its own property because the vtable this replaces got it wrong by
    /// default, and `1/sqrt(576)` against `1/sqrt(192)` is a silently
    /// worse model rather than a failure.
    #[test]
    fn the_softmax_scale_is_the_dot_width_and_not_the_page_row() {
        let d = plan(&glm5(), 10_000.0, 1e-5, Advertised::default());
        let over_page_row = 1.0 / (576.0f32).sqrt();
        assert!((d.attention[0].sm_scale - over_page_row).abs() > 1e-4);
    }

    /// The KV style is STATED by the row, with the ranks the shape
    /// already holds — not sniffed from a family name and re-read from a
    /// config.
    #[test]
    fn the_kv_style_carries_the_rows_own_ranks() {
        let d = plan(&glm5(), 10_000.0, 1e-5, Advertised::default());
        match d.kv {
            KvStyle::Mla {
                kv_lora_rank,
                qk_rope_head_dim,
            } => {
                assert_eq!(kv_lora_rank, 512);
                assert_eq!(qk_rope_head_dim, 64);
            }
            other => panic!("an MLA row must state an MLA cache, not {other:?}"),
        }
    }

    /// The refusal fires, and it fires at the DOOR — before a load, not
    /// at the first fire inside a walk.
    #[test]
    fn a_build_with_no_mla_store_refuses_the_row() {
        let err = deployment(&glm5(), 10_000.0, 1e-5, Advertised::default())
            .expect_err("no MLA store is built in this tree, so the row cannot be served");
        assert!(matches!(err, Refusal::Unsupported(_)));
    }

    /// What the row advertises rides through the projection untouched.
    ///
    /// It is CARRIED and not derived because the derivation it replaces
    /// read `model_type` and `max_position_embeddings` off a resident
    /// `HfConfig` at load. A projection that filled the label in from
    /// what it can see would be re-inventing the `architectures[0]`
    /// inference that put `Gemma4ForConditionalGeneration` in a table row
    /// it did not belong in.
    #[test]
    fn the_rows_advertised_label_is_carried_and_not_rewritten() {
        let stated = Advertised {
            arch: "glm_moe_dsa",
            max_model_len: 1000000,
            media_encode: false,
        };
        let d = plan(&glm5(), 10_000.0, 1e-5, stated.clone());
        assert_eq!(
            d.advertised, stated,
            "a projection that edits the label is inventing one, and one that \
             defaults it has dropped what the row said"
        );
        // The default has to be distinguishable from the stated value, or
        // the assertion above passes on a projection that ignores its
        // argument entirely.
        assert_ne!(stated, Advertised::default());
    }

    /// The ladder is used AS WRITTEN and no tower ships, and both are
    /// statements rather than omissions. `synthetic--glm-moe-dsa.json`,
    /// the only GLM-5 config committed here, states no `rope_scaling`
    /// block, so `None` is what was READ. A projection that filled in a
    /// factor would stretch every position by a ratio nothing here
    /// measured.
    #[test]
    fn the_rope_ladder_is_unscaled_and_no_tower_ships() {
        let d = plan(&glm5(), 10_000.0, 1e-5, Advertised::default());
        assert!(
            d.rope_scaling.is_none(),
            "no committed config states a rescaling to read"
        );
        assert_eq!(
            d.towers,
            crate::deployment::Towers::default(),
            "a text-in text-out stack that claims a tower has the worker bind an encoder \
             it does not ship",
        );
    }

    /// The two halves of that statement are separable: the PLAN exists
    /// whether or not this build can serve it, which is what lets a
    /// deployment be described and refused in one place.
    #[test]
    fn the_plan_exists_even_where_the_build_refuses_it() {
        let planned = plan(&glm5(), 10_000.0, 1e-5, Advertised::default());
        assert!(!planned.kv.has_a_store_in_this_build());
        assert!(KvStyle::Paged.has_a_store_in_this_build());
        assert!(!KvStyle::CompressedPlane { ratios: Vec::new() }.has_a_store_in_this_build());
    }

    /// The text traces for both fire classes, and names this
    /// generation's own family string — the goldens are keyed on it, so
    /// a rename here is a rename of every recorded plan.
    #[test]
    fn the_text_traces_for_both_fire_classes() {
        use model_compiler::trace::FireClass;
        let f = glm5();
        for (class, suffix) in [
            (FireClass::Decode, "decode"),
            (FireClass::Prefill, "prefill"),
        ] {
            let plan = trace(&f, class);
            assert_eq!(plan.family, format!("glm5.cuda.{suffix}"));
            assert!(!plan.ops.is_empty(), "a traced plan states ops");
        }
    }
}

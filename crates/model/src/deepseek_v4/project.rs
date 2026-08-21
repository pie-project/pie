//! The three projections a DeepSeek-V4 row makes: its tensor manifest,
//! its `Deployment`, and its traced text.
//!
//! Written against [`Dsv4Facts`] rather than against the row type, for
//! `kimi_k2/project.rs`'s reason: the numbers are the row's and the
//! arithmetic over them is the generation's, so a second V4 checkpoint
//! is a second row and not a second projection.
//!
//! # What this generation makes harder than its siblings
//!
//! Its checkpoint vocabulary is barely written down. There is no
//! published V4 config in this tree beyond a hand-written toy, and the
//! one committed checkpoint FIXTURE
//! (`tests/family_contracts.rs::deepseek_v4_checkpoint`) states four
//! spellings and no more: the bare `layers.` prefix,
//! `embed_tokens.weight`, `layers.<L>.attn.…` for the attention block
//! and `layers.<L>.ffn.…` for everything routed. [`manifest`] states
//! what those two prefixes and this generation's own text imply, and
//! deliberately states nothing where they imply nothing — see its doc.

use crate::deployment::{
    Advertised, AttnOutput, Deployment, Geometry, KvStyle, LayerAttention, NormPlacement,
    PrefillStyle, Refusal,
};
use crate::manifest::{Manifest, TensorSpec};

use super::spec::Dsv4Facts;

/// This row's tensors.
///
/// # Where every name below comes from
///
/// Two sources and no invention. The committed contract fixture states
/// the PREFIXES — `layers.<L>.attn.` for attention, `layers.<L>.ffn.`
/// for the mixture, a bare `embed_tokens.weight` and `norm.weight` at
/// the ends — and `author_deepseek_v4` corroborates them by asking for
/// `decoder_layer_prefix_any_of(&["model.layers.", "layers."])` and by
/// keying both expert passes on `.ffn.experts.`. This generation's own
/// text states the LEAVES: `wq_a`, `q_norm`, `wq_b`, `wkv`, `kv_norm`,
/// `wo_a`, `wo_b`. The extents are the row's arithmetic, as everywhere
/// else in the catalog.
///
/// The dense prefix's MLP is the one place the evidence is indirect and
/// it is still evidence: `author_deepseek_v4` runs
/// `dense_fused_projection_joins()`, a pass whose whole body looks for
/// `mlp.gate_proj` / `mlp.up_proj`, and a contract does not run a pass
/// for tensors its checkpoints never ship.
///
/// # What is deliberately NOT named
///
/// **The routed expert bank.** `author_deepseek_v4` reads it as
/// `ffn.experts.<e>.{w1,w3,w2}.weight` with a `.scale` beside each, and
/// the committed golden shows what those extents are: an MXFP4 `w1` is
/// `[intermediate, hidden / 2]`, two codes per byte, beside an E8M0
/// exponent per group of 32. Every one of those numbers is a statement
/// about the PACKING and not about the model — the same checkpoint
/// published bf16 would ship the same weights at twice the width — and a
/// manifest that named them would key identity on an encoding. The
/// ROUTER is named instead: it is the one row of a mixture that is never
/// quantized, and its extents ARE the mixture.
///
/// **The shared experts.** `dsv4_shard_axis` names
/// `shared_experts.{w1,w3,w2}` explicitly, so this generation plainly
/// has them — but [`super::spec::Dsv4MoeFacts`] states no shared width,
/// and a row cannot claim a tensor whose extents it cannot compute.
/// Naming it at no extent would be worse than silence: it would claim
/// every V4 ships one, which the shape does not know.
///
/// **The attention sink and the router bias.** The text names
/// `layer.<L>.attn_sink` and `layer.<L>.router_bias`, and both are trace
/// names with no witnessed checkpoint spelling — the fixture ships
/// neither. A manifest row for them would be a guess wearing a
/// measurement's clothes.
///
/// **The hyper-connection's affine pairs.** Six more of the same:
/// `layer.<L>.hc_{attn,mlp}_{scale,base}` and the tower-wide
/// `hc_head_{scale,base}`. `norm/dsv4_hc.cuh` reads a `scale` and a
/// `base` per mix and dereferences both per token, so they are weights
/// and not facts — but this generation's checkpoints are not in hand and
/// nothing here has seen how one spells them. The EXTENTS are known
/// exactly (`[3]` and `[2M + M*M]` for a pre-mix, `[1]` and `[M]` for
/// the head), which is what makes the omission a naming problem rather
/// than a shape one, and what a later row will need when a spelling
/// arrives. `tests/seam_names.rs` carries all six until then.
#[must_use]
pub fn manifest(f: &Dsv4Facts, tied_embeddings: bool) -> Manifest {
    let (hidden, vocab) = (u64::from(f.hidden), u64::from(f.vocab));
    let a = &f.attn;
    let q_width = u64::from(a.q_width());
    let q_lora = u64::from(a.q_lora_rank);
    let o_lora = u64::from(a.o_lora_rank);
    let dense_inter = u64::from(f.dense_intermediate);
    // A latent query is a PAIR of projections and a straight one is a
    // single tensor. They are alternatives, never both — the fixture's
    // `attn.wq` is what a checkpoint with no rank ships — so each is
    // claimed as an absence under the other's condition, and that is the
    // one place this manifest can tell two V4 checkpoints apart by
    // presence rather than by extent.
    let latent_q = a.q_lora_rank > 0;
    let grouped_o = a.o_lora_rank > 0;
    let has_dense_prefix = f.dense_layers > 0;
    let all_dense = f.dense_layers >= f.layers;

    Manifest::new(f.layers)
        .with(TensorSpec::required("embed_tokens", [vocab, hidden]))
        .with(TensorSpec::required("norm", [hidden]))
        // TIED vs UNTIED as presence, which is the only way a manifest
        // can tell them apart: every extent agrees.
        .tie(tied_embeddings, "lm_head", [vocab, hidden])
        // The two norms a layer opens its halves with. Rank-K residual
        // or not, each half still normalizes the stream it reads.
        .with(TensorSpec::required("layer.{}.attn_norm", [hidden]))
        .with(TensorSpec::required("layer.{}.mlp_norm", [hidden]))
        // ── Attention ────────────────────────────────────────────────
        .either(latent_q, "layer.{}.attn.wq_a", [q_lora, hidden])
        // A norm over the LATENT and not over the query: the text
        // normalizes `q_a` before it expands, which is why this is
        // `q_lora_rank` wide and not `heads * head_dim`.
        .either(latent_q, "layer.{}.attn.q_norm", [q_lora])
        .either(latent_q, "layer.{}.attn.wq_b", [q_width, q_lora])
        .either(!latent_q, "layer.{}.attn.wq", [q_width, hidden])
        // ONE projection for both K and V — the text hands the same
        // tensor to `write_kv_to_pages` twice. That is this generation's
        // sharpest difference from its MLA siblings, which project a
        // latent and read a separate `kv_b` back out of it, and it is
        // why `wkv` is `q_width` wide rather than a rank.
        .with(TensorSpec::required("layer.{}.attn.wkv", [q_width, hidden]))
        .with(TensorSpec::required("layer.{}.attn.kv_norm", [q_width]))
        // The output projection is low-rank AND grouped. Where it is
        // neither, a checkpoint ships the single `wo` instead, and the
        // two spellings exclude each other the same way the query's do.
        .either(grouped_o, "layer.{}.attn.wo_a", [o_lora, q_width])
        .either(grouped_o, "layer.{}.attn.wo_b", [hidden, o_lora])
        .either(!grouped_o, "layer.{}.attn.wo", [hidden, q_width])
        // ── The dense prefix and the mixture ─────────────────────────
        //
        // `first_k_dense_replace` is a fact a checkpoint publishes, and
        // every logical name collapses over the stack, so a model with a
        // prefix ships a dense MLP (from its leading layers) AND a
        // router (from the rest) and both rows appear. A stack that is
        // dense all the way ships no router; one with no prefix ships no
        // dense MLP. The same statement read from its two ends.
        .either(
            has_dense_prefix,
            "layer.{}.mlp.gate_proj",
            [dense_inter, hidden],
        )
        .either(
            has_dense_prefix,
            "layer.{}.mlp.up_proj",
            [dense_inter, hidden],
        )
        .either(
            has_dense_prefix,
            "layer.{}.mlp.down_proj",
            [hidden, dense_inter],
        )
        .either(
            !all_dense,
            "layer.{}.ffn.gate",
            [u64::from(f.moe.num_experts), hidden],
        )
}

/// This row's deployment, or a refusal at the DOOR.
///
/// A projection, and every value in it was already in the row. What the
/// derivation this replaces did instead was re-read a resident
/// `config.json` — and for this generation it did not even reach the
/// numbers: it chose `KvStyle::CompressedPlane` by asking whether a family's
/// `unbuilt_kv_store()` sentence contained the substring `"compress"`,
/// and then filled the ratios with `Vec::new()`.
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
/// row's [`KvStyle`] — which for this generation is every build, and is
/// the honest answer rather than a load that dies at its first fire.
pub fn deployment(
    f: &Dsv4Facts,
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
fn plan(f: &Dsv4Facts, rope_theta: f32, norm_eps: f32, advertised: Advertised) -> Deployment {
    let a = &f.attn;
    // The window is the UNCOMPRESSED reach and nothing else: everything
    // older than it is served by the compressed pass, which attends the
    // pooled entries. `-1` for a checkpoint that states no window, which
    // is `LayerAttention::window`'s own word for "the whole context" —
    // and the vtable's `if w > 0 { w } else { -1 }`, said once here.
    let window = i32::try_from(a.sliding_window).unwrap_or(i32::MAX);
    let window = if window > 0 { window } else { -1 };
    // Over the head's own width, because the dot IS the head: this
    // generation projects K and V straight rather than through a latent,
    // so what is stored and what is multiplied are the same width. Its
    // MLA siblings are where those two numbers come apart.
    let sm_scale = 1.0 / (a.head_dim as f32).sqrt();
    let attention = (0..f.layers)
        .map(|l| LayerAttention {
            // One shape for every layer, which is what this row was
            // already saying by having no per-layer count.
            kv_heads: a.heads,
            head_dim: a.head_dim,
            window,
            // Every layer owns its pages. A compressing layer owns a
            // second, smaller store beside them — which is `kv` below
            // and not a `kv_source`, because those entries are pooled
            // from this layer's own KV and not read from another
            // layer's.
            kv_source: l,
            sm_scale,
            rope_theta,
            // PARTIAL, over the last channels of each head — the text's
            // `rope_partial_last`. Stating `0` here would say "rotate
            // the whole head", and the 64 nope channels this generation
            // carries straight through are what makes the entries it
            // pools comparable across positions.
            rotary_dim: a.qk_rope_head_dim,
            q_gate: false,
        })
        .collect();

    Deployment {
        layers: f.layers,
        norm_eps,
        shape: Geometry {
            hidden: f.hidden,
            q_heads: a.heads,
            // K and V are ONE projection, `heads * head_dim` wide, and
            // the text hands that single tensor to `write_kv_to_pages`
            // as both planes — so every query head has its own KV head
            // and the GQA ratio is 1. Not an MLA's single shared plane,
            // however much the `q_lora_rank` beside it looks like one.
            kv_heads: a.heads,
            head_dim: a.head_dim,
            // Nothing pads it: 128 is a width every attention kernel in
            // the tree is instantiated at.
            head_dim_kernel: a.head_dim,
            intermediate: f.dense_intermediate,
            // One expert's inner width — 1024 against the dense
            // prefix's 5632 — and the forward workspace is one buffer
            // both layer kinds share, so both are stated and the planner
            // takes the wider.
            moe_intermediate: f.moe.moe_intermediate,
            experts_per_token: f.moe.top_k,
            shared_intermediate: 0,
            // Not "no shared expert": `dsv4_shard_axis` names
            // `shared_experts.{w1,w3,w2}`, so this generation plainly has
            // them. `Dsv4MoeFacts` states no width, and `manifest`'s doc
            // above already refuses to name the tensors for exactly that
            // reason — a width invented here would be the same guess with
            // a driver reading it.
            vocab: f.vocab,
        },
        attention,
        // THE SCHEDULE, stated. This is the number the memory planner
        // multiplies by a token budget (`compress_bytes_per_token`), and
        // the derivation this replaces passed `Vec::new()` here on every
        // load — so a V4's compressor cache, three tensors per
        // compressing layer that have to survive across fires, was
        // charged at zero bytes per token and came out of whatever the
        // KV pool did not use.
        kv: KvStyle::CompressedPlane {
            ratios: f.ratios.to_vec(),
        },
        // No recurrence: the compressed history is a CACHE keyed by
        // position, not a state carried per request.
        recurrent: None,
        prefill: PrefillStyle::Planned,
        // Stated rather than inherited. The vtable's default was `true`
        // under a doc-comment reading "Only gemma-4 does" — the default
        // and its own documentation disagreeing, with eleven families
        // taking whichever one the compiler read.
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
        // The row's own, not a class default -- see `Dsv4MoeFacts`.
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
        // Unscaled, because nothing in this tree says otherwise.
        // Notable, because the deepseek lineage is where YaRN is
        // expected — v3 ships `rope_type: "yarn"` in the wild — but
        // neither `synthetic--deepseek-v4.json` nor its v3 sibling
        // states a `rope_scaling` block, so there is no factor to read.
        // Inventing one would rescale every position by a ratio nothing
        // here measured.
        rope_scaling: None,
        towers: Default::default(),
    }
}

/// Why this build has no Metal text for a deepseek-v4 row.
///
/// A `const` so the test that asserts the refusal NAMES the missing
/// thing compares against the same string the caller is shown, rather
/// than against a paraphrase that can drift away from it — the shape
/// `csm::project::NO_TRACE` set for the same reason.
///
/// Its forward is `dsv4_cuda`: MLA with a compressed KV latent, a
/// per-token compression boundary and a 256-expert router. The only
/// Metal text in this build is `llama_like_metal`, which states none of
/// those — and it could not be reached for this row in any case, since
/// it takes a `LlamaLikeFacts` and this row's shape is `Dsv4Facts`.
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
pub const NO_METAL: &str = "deepseek-v4 has no Metal text in this build: its forward is `dsv4_cuda` — \
     multi-head latent attention over a compressed KV, a per-token compression \
     boundary and a 256-expert router — and the one Metal text here \
     (`llama_like_metal`) states none of those and takes a different shape \
     entirely; the CUDA backend serves this row";

/// Trace this row's CUDA text for one fire class.
///
/// No binding facts on the way in, unlike kimi-k2's: this generation's
/// text reads the shape and nothing else, so the shape is the whole
/// input. Both fire classes trace — the compressed pass needs the block
/// boundaries a fire's positions imply, and that is a per-TOKEN fact
/// either class can state.
#[must_use]
pub fn trace(
    f: &Dsv4Facts,
    class: model_ir::trace::FireClass,
    norm_eps: f32,
    rope_theta: f32,
) -> model_ir::trace::ForwardPlan {
    // THE SHIPPED POINT. deepseek-v4 catalogues one SKU today; the table
    // in `forward::CATALOG` is where a second one appears, and the
    // coverage test is what keeps every row loadable.
    use model_dsl::axes::{Bf16Ax, NativeKv};
    super::forward::dsv4_cuda::<Bf16Ax, Bf16Ax, Bf16Ax, NativeKv>(f, class, norm_eps, rope_theta)
}

#[cfg(test)]
mod tests {
    use super::super::spec::Dsv4AttnFacts;
    use super::*;
    use crate::manifest::Presence;

    fn f() -> Dsv4Facts {
        Dsv4Facts::dsv4_synthetic()
    }

    fn extents(m: &Manifest, name: &str) -> Vec<u64> {
        m.tensors
            .iter()
            .find(|t| t.name == name)
            .unwrap_or_else(|| panic!("{name} is not named by this manifest"))
            .extents
            .clone()
    }

    fn presence(m: &Manifest, name: &str) -> Presence {
        m.tensors
            .iter()
            .find(|t| t.name == name)
            .unwrap_or_else(|| panic!("{name} is not named by this manifest"))
            .presence
    }

    fn named(m: &Manifest, name: &str) -> bool {
        m.tensors.iter().any(|t| t.name == name)
    }

    /// The store this generation needs is not one this build has, and
    /// the answer has to be per-STYLE rather than per-family: `Paged` is
    /// the only shape a driver here provisions.
    #[test]
    fn only_the_paged_store_is_built() {
        assert!(KvStyle::Paged.has_a_store_in_this_build());
        assert!(
            !KvStyle::CompressedPlane {
                ratios: vec![1, 2, 4]
            }
            .has_a_store_in_this_build()
        );
        assert!(!KvStyle::CompressedPlane { ratios: Vec::new() }.has_a_store_in_this_build());
        assert!(
            !KvStyle::Mla {
                kv_lora_rank: 512,
                qk_rope_head_dim: 64
            }
            .has_a_store_in_this_build()
        );
    }

    /// The two ends of the stack, which every row in the catalog states
    /// and which are the only rows here a fixture witnesses verbatim.
    #[test]
    fn the_manifest_names_both_ends_of_the_stack() {
        let m = manifest(&f(), false);
        assert_eq!(extents(&m, "embed_tokens"), vec![129_280, 2048]);
        assert_eq!(extents(&m, "norm"), vec![2048]);
        assert_eq!(
            m.layers, 6,
            "a manifest carries the stack it was written for"
        );
    }

    /// Tied and untied differ by PRESENCE and not by extent, so the row
    /// has to claim the absence rather than leave the name unmentioned:
    /// an unmentioned name is one a checkpoint may ship freely.
    #[test]
    fn a_tied_head_forbids_the_name_an_untied_one_requires() {
        assert_eq!(
            presence(&manifest(&f(), false), "lm_head"),
            Presence::Required
        );
        assert_eq!(presence(&manifest(&f(), true), "lm_head"), Presence::Absent);
        assert_eq!(
            extents(&manifest(&f(), false), "lm_head"),
            vec![129_280, 2048],
            "an untied head is its own copy of the table's shape",
        );
    }

    /// The query's latent is a PAIR, and the norm between them is over
    /// the rank rather than over the heads — 768 and not 2048. A row
    /// that stated the wider one would fault every real checkpoint at
    /// the one tensor that proves the latent exists.
    #[test]
    fn a_latent_query_is_two_projections_with_a_norm_over_the_rank() {
        let m = manifest(&f(), false);
        assert_eq!(extents(&m, "layer.{}.attn.wq_a"), vec![768, 2048]);
        assert_eq!(extents(&m, "layer.{}.attn.q_norm"), vec![768]);
        assert_eq!(extents(&m, "layer.{}.attn.wq_b"), vec![2048, 768]);
        assert_eq!(
            presence(&m, "layer.{}.attn.wq"),
            Presence::Absent,
            "a checkpoint cannot ship both the pair and the single projection",
        );
    }

    /// And the other arm: no rank, one tensor, and the pair is then the
    /// forbidden spelling. This is the shape the committed contract
    /// fixture ships (`layers.0.attn.wq.weight`).
    #[test]
    fn a_straight_query_is_one_projection_and_forbids_the_pair() {
        let no_lora = Dsv4Facts {
            attn: Dsv4AttnFacts {
                q_lora_rank: 0,
                ..f().attn
            },
            ..f()
        };
        let m = manifest(&no_lora, false);
        assert_eq!(extents(&m, "layer.{}.attn.wq"), vec![2048, 2048]);
        assert_eq!(presence(&m, "layer.{}.attn.wq_a"), Presence::Absent);
        assert_eq!(presence(&m, "layer.{}.attn.wq_b"), Presence::Absent);
        assert_eq!(presence(&m, "layer.{}.attn.q_norm"), Presence::Absent);
    }

    /// One projection serves K and V, and it is head-wide rather than a
    /// rank. This is the row that tells a V4 from its MLA siblings
    /// before any name is compared: an MLA checkpoint's KV projection is
    /// its `kv_lora_rank` (512 on GLM-5, 256 on K3) and could not be
    /// 2048 without ceasing to be a compression.
    #[test]
    fn one_projection_serves_both_kv_planes() {
        let m = manifest(&f(), false);
        assert_eq!(extents(&m, "layer.{}.attn.wkv"), vec![2048, 2048]);
        assert_eq!(extents(&m, "layer.{}.attn.kv_norm"), vec![2048]);
        // The sibling widths, held as local numbers rather than read
        // from a sibling module — the isolation rule forbids the reach,
        // and the point is only that they are different.
        for latent in [512_u64, 256] {
            assert_ne!(
                extents(&m, "layer.{}.attn.wkv")[0],
                latent,
                "a KV projection this wide is not a latent, and a manifest that \
                 matched one would identify a V4 as an MLA sibling",
            );
        }
    }

    /// The output projection's two spellings exclude each other the same
    /// way the query's do, and the grouped one's rank is the width
    /// between them.
    #[test]
    fn a_grouped_output_is_two_projections_and_forbids_the_single_one() {
        let m = manifest(&f(), false);
        assert_eq!(extents(&m, "layer.{}.attn.wo_a"), vec![512, 2048]);
        assert_eq!(extents(&m, "layer.{}.attn.wo_b"), vec![2048, 512]);
        assert_eq!(presence(&m, "layer.{}.attn.wo"), Presence::Absent);

        let flat = manifest(
            &Dsv4Facts {
                attn: Dsv4AttnFacts {
                    o_lora_rank: 0,
                    ..f().attn
                },
                ..f()
            },
            false,
        );
        assert_eq!(extents(&flat, "layer.{}.attn.wo"), vec![2048, 2048]);
        assert_eq!(presence(&flat, "layer.{}.attn.wo_a"), Presence::Absent);
        assert_eq!(presence(&flat, "layer.{}.attn.wo_b"), Presence::Absent);
    }

    /// A stack with a dense prefix publishes BOTH a dense MLP and a
    /// router, because the layer index is what tells them apart and the
    /// logical name collapses it away.
    #[test]
    fn a_prefixed_stack_names_the_dense_mlp_and_the_router_both() {
        let m = manifest(&f(), false);
        assert_eq!(extents(&m, "layer.{}.mlp.gate_proj"), vec![5632, 2048]);
        assert_eq!(extents(&m, "layer.{}.mlp.up_proj"), vec![5632, 2048]);
        assert_eq!(extents(&m, "layer.{}.mlp.down_proj"), vec![2048, 5632]);
        assert_eq!(extents(&m, "layer.{}.ffn.gate"), vec![64, 2048]);
    }

    /// The two ends of the prefix rule, which are the branches a stack
    /// with no dense layers and a stack with nothing but them take.
    #[test]
    fn a_stack_with_no_prefix_ships_no_dense_mlp_and_an_all_dense_one_no_router() {
        let no_prefix = manifest(
            &Dsv4Facts {
                dense_layers: 0,
                ..f()
            },
            false,
        );
        assert_eq!(
            presence(&no_prefix, "layer.{}.mlp.gate_proj"),
            Presence::Absent
        );
        assert_eq!(
            presence(&no_prefix, "layer.{}.ffn.gate"),
            Presence::Required
        );

        let all_dense = manifest(
            &Dsv4Facts {
                dense_layers: 6,
                ..f()
            },
            false,
        );
        assert_eq!(
            presence(&all_dense, "layer.{}.mlp.gate_proj"),
            Presence::Required
        );
        assert_eq!(
            presence(&all_dense, "layer.{}.ffn.gate"),
            Presence::Absent,
            "a stack that never routes has nothing for a router to route to",
        );
    }

    /// The expert bank is not named at all — not required, not
    /// forbidden. Naming it would key identity on a PACKING: the same
    /// weights ship MXFP4 at `[intermediate, hidden / 2]` and bf16 at
    /// twice that, and a row is about the model.
    #[test]
    fn the_expert_bank_is_not_named_under_any_presence() {
        let m = manifest(&f(), false);
        for name in [
            "layer.{}.ffn.experts.0.w1",
            "layer.{}.ffn.experts.0.w2",
            "layer.{}.ffn.experts.0.w3",
            "layer.{}.ffn.shared_experts.w1",
        ] {
            assert!(!named(&m, name), "{name} states an encoding, not a model");
        }
    }

    /// A V4 manifest could not be mistaken for one of its MLA siblings'.
    /// The `attn.` / `ffn.` prefixes are this generation's own, and no
    /// other row in the catalog spells a projection `wkv`.
    #[test]
    fn the_manifest_is_not_one_a_latent_sibling_could_match() {
        let m = manifest(&f(), false);
        for sibling in [
            "layer.{}.self_attn.kv_a_proj_with_mqa",
            "layer.{}.self_attn.kv_b_proj",
            "layer.{}.self_attn.q_proj",
            "layer.{}.block_sparse_moe.gate",
        ] {
            assert!(
                !named(&m, sibling),
                "{sibling} belongs to another generation"
            );
        }
        assert!(named(&m, "layer.{}.attn.wkv"));
    }

    /// The deployment refuses, and the refusal is the row's own answer
    /// rather than an absence from a second table. `unbuilt_kv_store()`
    /// used to say this in a sentence a caller pattern-matched on by
    /// substring.
    #[test]
    fn the_row_refuses_at_the_door_because_no_compressed_store_is_built() {
        let err = deployment(&f(), 10_000.0, 1e-5, Advertised::default())
            .expect_err("no build here provisions a compressor cache");
        assert!(matches!(err, Refusal::Unsupported(_)));
    }

    /// Everything the refusal hides is still a fact about the row, and
    /// the total projection is where it is stated. A layer count that
    /// disagreed with the shape would size every table in the driver
    /// wrongly.
    #[test]
    fn the_plan_states_one_attention_row_per_layer() {
        let d = plan(&f(), 10_000.0, 1e-5, Advertised::default());
        assert_eq!(d.layers, 6);
        assert_eq!(d.attention.len(), 6);
        assert_eq!(
            d.norm_eps, 1e-5,
            "an epsilon no tensor carries has to be stated"
        );
    }

    /// The geometry, field by field. Every one of these was read off a
    /// resident `config.json` by the launch path, which is how a fire's
    /// geometry came from a different reading of the checkpoint than the
    /// trace it fired.
    #[test]
    fn the_geometry_is_the_rows_own_numbers() {
        let g = plan(&f(), 10_000.0, 1e-5, Advertised::default()).shape;
        assert_eq!(g.hidden, 2048);
        assert_eq!(g.q_heads, 16);
        assert_eq!(g.kv_heads, 16, "K and V are one projection, one head each");
        assert_eq!(g.gqa_group(), 1, "not an MLA's shared plane");
        assert_eq!(g.head_dim, 128);
        assert_eq!(g.head_dim_kernel, 128);
        assert_eq!(g.intermediate, 5632);
        assert_eq!(g.moe_intermediate, 1024);
        assert_eq!(g.vocab, 129_280);
    }

    /// The workspace is sized from the WIDER of the two MLP widths, and
    /// on this row that is the dense prefix's. A planner given only the
    /// mixture's 1024 would under-size the buffer the prefix layers
    /// share with it — which does not fail, it quietly moves bytes out
    /// of the KV pool.
    #[test]
    fn the_widest_mlp_is_the_dense_prefixs_and_not_the_mixtures() {
        let g = plan(&f(), 10_000.0, 1e-5, Advertised::default()).shape;
        assert_eq!(g.widest_mlp(), 5632);
        assert!(
            g.moe_intermediate > 0,
            "a mixture states one expert's width"
        );
    }

    /// The per-layer attention table: one window, one scale, every layer
    /// on its own pages, and a PARTIAL rotation.
    #[test]
    fn every_layer_attends_its_own_pages_through_a_partial_rope() {
        let d = plan(&f(), 10_000.0, 1e-5, Advertised::default());
        let want = 1.0 / (128.0_f32).sqrt();
        for (l, la) in d.attention.iter().enumerate() {
            assert_eq!(la.head_dim, 128);
            assert_eq!(la.window, 2048, "the uncompressed reach this row states");
            assert_eq!(la.kv_source, l as u32, "no layer reads another's pages");
            assert!((la.sm_scale - want).abs() < 1e-9);
            assert_eq!(la.rope_theta, 10_000.0);
            assert_eq!(
                la.rotary_dim, 64,
                "the rope turns the last channels of a head; zero would say the \
                 whole head turns and the nope half is what makes a pooled \
                 entry comparable across positions",
            );
        }
    }

    /// A checkpoint that states no window attends the whole context, and
    /// `-1` is the field's word for that. Zero would be a driver
    /// dropping every page but the current one.
    #[test]
    fn no_window_means_the_whole_context_and_not_none_of_it() {
        let d = plan(
            &Dsv4Facts {
                attn: Dsv4AttnFacts {
                    sliding_window: 0,
                    ..f().attn
                },
                ..f()
            },
            10_000.0,
            1e-5,
            Advertised::default(),
        );
        assert!(d.attention.iter().all(|la| la.window == -1));
    }

    /// The compression schedule reaches the driver, which is the whole
    /// repair: `KvStyle::CompressedPlane { ratios: Vec::new() }` charged a V4's
    /// compressor cache at nothing per token.
    #[test]
    fn the_kv_style_carries_the_schedule_the_row_states() {
        let d = plan(&f(), 10_000.0, 1e-5, Advertised::default());
        match &d.kv {
            KvStyle::CompressedPlane { ratios } => {
                assert_eq!(ratios.as_slice(), &[1, 2, 4]);
                assert!(
                    !ratios.is_empty(),
                    "an empty schedule is a cache sized at zero"
                );
            }
            other => panic!("a V4 attends through compressed entries, not {other:?}"),
        }
    }

    /// Nothing recurs. The compressed history is a cache keyed by
    /// position, and a `Some` here would have a driver allocating
    /// per-request state slabs for a stack that carries none.
    #[test]
    fn the_stack_carries_no_per_request_state() {
        assert!(
            plan(&f(), 10_000.0, 1e-5, Advertised::default())
                .recurrent
                .is_none()
        );
    }

    /// The remaining answers, stated rather than inherited from a vtable
    /// default that disagreed with its own doc comment.
    #[test]
    fn the_serving_answers_are_stated_and_not_defaulted() {
        let d = plan(&f(), 10_000.0, 1e-5, Advertised::default());
        assert_eq!(d.prefill, PrefillStyle::Planned);
        assert_eq!(d.attn_output, AttnOutput::DriverPinned);
        assert_eq!(d.norm, NormPlacement::Pre);
        assert_eq!(d.logit_softcap, 0.0);
        assert_eq!(d.ple_dim, 0);
        assert!(d.scales.is_empty());
        assert!(d.towers.audio.is_none() && d.towers.vision.is_none());
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
            arch: "deepseek_v4",
            max_model_len: 163840,
            media_encode: false,
        };
        let d = plan(&f(), 10_000.0, 1e-5, stated.clone());
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
    /// statements rather than omissions.
    ///
    /// This is the lineage where a rescaling would be expected — v3
    /// ships `rope_type: "yarn"` in the wild — and neither
    /// `synthetic--deepseek-v4.json` nor its v3 sibling states a
    /// `rope_scaling` block, so `None` is what was read. A projection
    /// that filled in a factor would stretch every position by a ratio
    /// nothing here measured.
    #[test]
    fn the_rope_ladder_is_unscaled_and_no_tower_ships() {
        let d = plan(&f(), 10_000.0, 1e-5, Advertised::default());
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

    /// Both fire classes trace, and they are different texts: the
    /// compressed pass reads its block boundaries from the positions a
    /// fire brings, and a decode's are not a prefill's.
    #[test]
    fn both_fire_classes_have_a_text() {
        use model_ir::trace::FireClass;
        let decode = trace(&f(), FireClass::Decode);
        let prefill = trace(&f(), FireClass::Prefill);
        assert!(decode.family.ends_with("decode"), "{}", decode.family);
        assert!(prefill.family.ends_with("prefill"), "{}", prefill.family);
        assert_ne!(decode.family, prefill.family);
    }
}

//! The three projections a CSM row makes — and two of them are
//! refusals.
//!
//! This is the generation the `Result` on
//! [`Variant::deployment`](crate::catalog::Variant::deployment) and
//! [`Variant::trace`](crate::catalog::Variant::trace) was added for.
//! CSM has a contract — [`super::contract::author_csm`] binds every
//! tensor in the package and narrows the fp32 checkpoint to bf16 — and
//! it has no traced forward text at all. There is no `forward/` module
//! in this generation, and there never has been.
//!
//! # Why a refusal, and not a `None`
//!
//! The trait this replaces answered these questions with defaults. A
//! family that had no forward returned `None` from a defaulted method
//! and the caller read that as "nothing special here"; a family with no
//! deployment facts fell through `FACTS_ROWS` to a llama-like
//! derivation, which for CSM would have SUCCEEDED — the backbone states
//! `hidden_size`, `num_attention_heads`, `num_key_value_heads`,
//! `head_dim`, `intermediate_size` and `rms_norm_eps` under exactly the
//! spellings a llama reader wants. It would have produced a servable
//! 16-layer stack over a 2051-entry vocabulary, paged it, fired it, and
//! returned audio codebook indices to a caller that asked for speech.
//!
//! So the refusals below name what is MISSING. A `Refusal::Unsupported`
//! is a statement about this BUILD — the checkpoint is fine, and a pie
//! with a CSM decoder in it would serve the same row — which is why it
//! is not `Malformed`, and why the row still answers
//! [`manifest`] and `load_shape` in full.

use crate::deployment::{Deployment, Refusal};
use crate::manifest::{Manifest, TensorSpec};

use super::spec::CsmFacts;

/// Why a CSM row cannot be deployed by this build.
///
/// A `const` so the test that asserts the refusal NAMES the missing
/// thing compares against the same string the caller is shown, rather
/// than against a paraphrase that can drift away from it.
pub const NO_DEPLOYMENT: &str = "csm is three stacks — a 16-layer backbone, a 4-layer depth \
     decoder that emits 32 residual codebooks per frame, and a Mimi \
     codec — and a Deployment describes one; this build has no speech \
     decode loop to drive them and no `Deployment` field that could \
     carry the depth decoder or the codec";

/// Why a CSM row cannot be traced by this build.
pub const NO_TRACE: &str = "csm has no traced forward text: there is no `csm/forward` module, \
     so neither the backbone's frame pass nor the depth decoder's \
     per-codebook pass has ever been written in the tracing eDSL";

/// This row's tensors — all three stacks of them.
///
/// # Why the codec is in here
///
/// It is 350 of the checkpoint's 537 tensors. A manifest that named
/// only the backbone would match a bare 16-layer llama of the right
/// width with no codec and no depth decoder beside it — which is
/// precisely the mistake the refusals above exist to prevent, made one
/// layer earlier. Identity IS validation here: the thing that makes a
/// checkpoint a CSM rather than a small llama is the two stacks bolted
/// to it, so the two stacks are what the manifest asks for.
///
/// # Why the codec's extents are mostly unstated
///
/// [`TensorSpec::present`] rather than [`TensorSpec::required`] for the
/// convolutional stacks. Mimi's encoder and decoder change channel
/// count and stride at every layer — `[64, 1, 7]` at the mouth,
/// `[1024, 512, 7]` at the throat — so `encoder.layer.{}.conv` does not
/// name ONE extent, and a spec that stated one would be checked against
/// whichever layer the checkpoint happened to publish last. The
/// quantizer's codebooks DO have one shape across all 32, and those are
/// stated.
#[must_use]
pub fn manifest(f: &CsmFacts) -> Manifest {
    let b = &f.backbone;
    let d = &f.depth;
    let c = &f.codec;
    let (hidden, inter) = (u64::from(b.hidden), u64::from(b.intermediate));
    let (dh, dinter) = (u64::from(d.hidden), u64::from(d.intermediate));

    Manifest::new(b.layers)
        // ── The backbone.
        //
        // Its TEXT table only. The audio table is the depth decoder's,
        // shared, and the absence of a second copy is stated below.
        .with(TensorSpec::required("embed_text_tokens", [u64::from(b.text_vocab), hidden]))
        .with(TensorSpec::required("backbone_model.norm", [hidden]))
        // `tie_word_embeddings: false`, so the head is shipped — over
        // ONE codebook's alphabet, not the text vocabulary. That is the
        // extent that says "this stack emits audio codes": a 2051-wide
        // head on a 128 256-entry embedding table is a shape no text
        // model has.
        .either(!f.tied_embeddings, "lm_head", [u64::from(b.audio_vocab), hidden])
        .with(TensorSpec::required(
            "backbone_model.layer.{}.self_attn.q_proj",
            [u64::from(b.q_width()), hidden],
        ))
        .with(TensorSpec::required(
            "backbone_model.layer.{}.self_attn.k_proj",
            [u64::from(b.kv_width()), hidden],
        ))
        .with(TensorSpec::required(
            "backbone_model.layer.{}.self_attn.v_proj",
            [u64::from(b.kv_width()), hidden],
        ))
        .with(TensorSpec::required(
            "backbone_model.layer.{}.self_attn.o_proj",
            [hidden, u64::from(b.q_width())],
        ))
        .with(TensorSpec::required("backbone_model.layer.{}.mlp.gate_proj", [inter, hidden]))
        .with(TensorSpec::required("backbone_model.layer.{}.mlp.up_proj", [inter, hidden]))
        .with(TensorSpec::required("backbone_model.layer.{}.mlp.down_proj", [hidden, inter]))
        .with(TensorSpec::required("backbone_model.layer.{}.input_layernorm", [hidden]))
        .with(TensorSpec::required("backbone_model.layer.{}.post_attention_layernorm", [hidden]))
        // THE TIE, as an absence. `tie_codebooks_embeddings: true`
        // means the backbone reads its audio codes out of the depth
        // decoder's table, so it ships no table of its own — and a
        // checkpoint that DID ship one is an untied build whose codes
        // would be embedded twice, differently.
        .either(
            !f.tied_codebooks,
            "backbone_model.embed_tokens.embed_audio_tokens",
            [u64::from(d.code_table_rows()), hidden],
        )
        // ── The depth decoder.
        //
        // The shared code table: one alphabet per codebook, stacked, at
        // the BACKBONE's width — which is the tie written as an extent.
        // 32 × 2051 = 65 632 rows of 2048.
        .with(TensorSpec::required(
            "depth_decoder.model.embed_tokens",
            [u64::from(d.code_table_rows()), u64::from(d.backbone_hidden)],
        ))
        // The narrowing from a backbone row to this stack's width. The
        // one tensor whose shape states both hidden sizes at once, and
        // therefore the one that cannot match a stack of either width
        // alone.
        .with(TensorSpec::required(
            "depth_decoder.model.inputs_embeds_projector",
            [dh, u64::from(d.backbone_hidden)],
        ))
        .with(TensorSpec::required("depth_decoder.model.norm", [dh]))
        // `codebooks - 1` slices: codebook 0 is the backbone's.
        .with(TensorSpec::required(
            "depth_decoder.codebooks_head",
            [u64::from(d.head_slices()), dh, u64::from(d.vocab)],
        ))
        .with(TensorSpec::required(
            "depth_decoder.model.layer.{}.self_attn.q_proj",
            [u64::from(d.q_width()), dh],
        ))
        .with(TensorSpec::required(
            "depth_decoder.model.layer.{}.self_attn.k_proj",
            [u64::from(d.kv_width()), dh],
        ))
        .with(TensorSpec::required(
            "depth_decoder.model.layer.{}.self_attn.v_proj",
            [u64::from(d.kv_width()), dh],
        ))
        .with(TensorSpec::required(
            "depth_decoder.model.layer.{}.self_attn.o_proj",
            [dh, u64::from(d.q_width())],
        ))
        .with(TensorSpec::required("depth_decoder.model.layer.{}.mlp.gate_proj", [dinter, dh]))
        .with(TensorSpec::required("depth_decoder.model.layer.{}.mlp.up_proj", [dinter, dh]))
        .with(TensorSpec::required("depth_decoder.model.layer.{}.mlp.down_proj", [dh, dinter]))
        .with(TensorSpec::required("depth_decoder.model.layer.{}.input_layernorm", [dh]))
        .with(TensorSpec::required("depth_decoder.model.layer.{}.post_attention_layernorm", [dh]))
        // ── The Mimi codec.
        //
        // The two quantizers, whose codebooks are the one part of the
        // codec with a stable extent: `[codebook_size, codebook_dim]`,
        // identical across all 32. They are also the tensors that carry
        // the alphabet the two decoders emit into, so getting them
        // wrong is getting the model's output space wrong.
        .with(TensorSpec::required(
            "codec_model.quantizer.semantic_residual_vector_quantizer.layer.{}.codebook.embed_sum",
            [u64::from(c.codebook_size), u64::from(c.codebook_dim)],
        ))
        .with(TensorSpec::required(
            "codec_model.quantizer.acoustic_residual_vector_quantizer.layer.{}.codebook.embed_sum",
            [u64::from(c.codebook_size), u64::from(c.codebook_dim)],
        ))
        // The projections in and out of the latent, stated as `[dim,
        // hidden, 1]` — a 1-wide convolution, which `extents_agree`
        // squeezes, so the same row matches a converter that wrote them
        // as plain matrices.
        .with(TensorSpec::required(
            "codec_model.quantizer.semantic_residual_vector_quantizer.input_proj",
            [u64::from(c.codebook_dim), u64::from(c.hidden), 1],
        ))
        .with(TensorSpec::required(
            "codec_model.quantizer.acoustic_residual_vector_quantizer.output_proj",
            [u64::from(c.hidden), u64::from(c.codebook_dim), 1],
        ))
        // The codec transformer, at the codec's width. Present rather
        // than dimensioned only where the width would restate itself.
        .with(TensorSpec::required(
            "codec_model.encoder_transformer.layer.{}.self_attn.q_proj",
            [u64::from(c.hidden), u64::from(c.hidden)],
        ))
        .with(TensorSpec::required(
            "codec_model.decoder_transformer.layer.{}.self_attn.q_proj",
            [u64::from(c.hidden), u64::from(c.hidden)],
        ))
        // Mimi's per-branch learned scale. Nothing else in this
        // checkpoint has one, and nothing else in the catalog has one
        // at all — so it is the single cheapest tensor to look for when
        // asking "is there a Mimi in here".
        .with(TensorSpec::required(
            "codec_model.decoder_transformer.layer.{}.self_attn_layer_scale.scale",
            [u64::from(c.hidden)],
        ))
        // The convolutional stacks, by name only: see the doc above on
        // why their extents are not stated.
        .with(TensorSpec::present("codec_model.encoder.layer.{}.conv"))
        .with(TensorSpec::present("codec_model.decoder.layer.{}.conv"))
        .with(TensorSpec::present("codec_model.upsample.conv"))
        .with(TensorSpec::present("codec_model.downsample.conv"))
}

/// What this build would need to serve a CSM row, which it has not got.
///
/// # Errors
///
/// Always [`Refusal::Unsupported`], carrying [`NO_DEPLOYMENT`].
///
/// The signature still takes the facts. That is deliberate and not
/// dead: the refusal is a statement about the BUILD, so the day a
/// speech decode loop lands, this function grows a body and its callers
/// do not change. A `deployment()` that took no arguments would be
/// admitting the row has no numbers, and it has all of them —
/// [`super::spec::CsmFacts`] is complete, and [`manifest`] projects it.
pub fn deployment(f: &CsmFacts) -> Result<Deployment, Refusal> {
    let _ = f;
    Err(Refusal::Unsupported(NO_DEPLOYMENT))
}

/// # Errors
///
/// Always [`Refusal::Unsupported`], carrying [`NO_TRACE`].
///
/// Not gated on the `forward` feature, because the answer does not
/// depend on it: a build WITH tracing still has no CSM text to trace.
/// Gating it would make the honest answer available only to the
/// configuration that could act on it, which is the shape of the
/// problem this generation is here to illustrate. The `Ok` type is
/// nevertheless [`ForwardPlan`](model_compiler::trace::ForwardPlan) and
/// not a never-type, so the day a `csm/forward` lands this signature
/// does not move.
pub fn trace(f: &CsmFacts) -> Result<model_compiler::trace::ForwardPlan, Refusal> {
    let _ = f;
    Err(Refusal::Unsupported(NO_TRACE))
}

#[cfg(test)]
mod tests {
    use super::{CsmFacts, NO_DEPLOYMENT, NO_TRACE, deployment, manifest, trace};
    use crate::deployment::Refusal;
    use crate::manifest::{Observed, Presence};

    /// What a checkpoint this manifest describes would publish.
    fn implied(f: &CsmFacts) -> Observed {
        Observed::from_pairs(
            manifest(f)
                .tensors
                .iter()
                .filter(|t| t.presence != Presence::Absent)
                .map(|t| (t.name.replace("{}", "0"), t.extents.clone())),
        )
    }

    fn spec_extents(f: &CsmFacts, name: &str) -> Vec<u64> {
        manifest(f)
            .tensors
            .iter()
            .find(|t| t.name == name)
            .unwrap_or_else(|| panic!("no row named '{name}'"))
            .extents
            .clone()
    }

    /// A manifest that cannot match its own arithmetic describes a
    /// stack that does not exist.
    #[test]
    fn a_row_satisfies_the_manifest_it_states() {
        for f in [CsmFacts::csm_1b(), CsmFacts::csm_synthetic()] {
            let m = manifest(&f);
            let seen = implied(&f);
            assert!(m.check(&seen).is_ok(), "{}", m.check(&seen).unwrap_err());
        }
    }

    /// The manifest names all THREE stacks.
    ///
    /// This is the assertion that separates a CSM row from the
    /// llama-like row a naive derivation would have produced: the
    /// backbone alone is a small llama, and a manifest that asked only
    /// for the backbone would say yes to one.
    #[test]
    fn the_manifest_asks_for_the_depth_decoder_and_the_codec_too() {
        let binding = manifest(&CsmFacts::csm_1b());
        let names: Vec<&str> = binding.tensors.iter().map(|t| t.name.as_str()).collect();
        assert!(names.iter().any(|n| n.starts_with("backbone_model.")), "no backbone");
        assert!(names.iter().any(|n| n.starts_with("depth_decoder.")), "no depth decoder");
        assert!(names.iter().any(|n| n.starts_with("codec_model.")), "no codec");
        assert!(
            names.contains(&"depth_decoder.codebooks_head"),
            "without the codebook head this row matches a llama that emits one code per frame"
        );
    }

    /// A bare backbone does not match this row.
    ///
    /// The concrete failure mode: strip the depth decoder and the codec
    /// out of a CSM checkpoint and what is left is a 16-layer, 2048-wide
    /// GQA stack — servable, plausible, and mute. The manifest has to
    /// say no to it, and the fault it reports has to name the stack that
    /// is missing rather than complain about a width.
    #[test]
    fn a_backbone_with_no_depth_decoder_is_not_a_csm() {
        use crate::manifest::Fault;
        let f = CsmFacts::csm_1b();
        let full = manifest(&f);
        let backbone_only = Observed::from_pairs(
            full.tensors
                .iter()
                .filter(|t| t.presence != Presence::Absent)
                .filter(|t| !t.name.starts_with("depth_decoder."))
                .map(|t| (t.name.replace("{}", "0"), t.extents.clone())),
        );
        let err = full.check(&backbone_only).expect_err("a mute backbone must not match");
        assert!(
            err.faults.iter().any(
                |fault| matches!(fault, Fault::Missing(n) if n.starts_with("depth_decoder."))
            ),
            "the mismatch must name the missing stack, not a width: {err}"
        );
    }

    /// The tie is stated as an absence, so an untied build cannot match.
    ///
    /// `tie_codebooks_embeddings` is what says the backbone and the
    /// depth decoder read audio codes out of ONE table. A checkpoint
    /// that ships two would embed the same code twice, differently, and
    /// nothing downstream would notice — so the row forbids the second
    /// table by name.
    #[test]
    fn a_second_audio_table_is_forbidden_by_name() {
        let f = CsmFacts::csm_1b();
        assert!(f.tied_codebooks);
        let row = manifest(&f)
            .tensors
            .into_iter()
            .find(|t| t.name == "backbone_model.embed_tokens.embed_audio_tokens")
            .expect("the tie is stated");
        assert_eq!(row.presence, Presence::Absent);

        // And the other leg of `either`: an untied CSM would REQUIRE it,
        // at the shared table's extents.
        let untied = CsmFacts { tied_codebooks: false, ..CsmFacts::csm_1b() };
        let row = manifest(&untied)
            .tensors
            .into_iter()
            .find(|t| t.name == "backbone_model.embed_tokens.embed_audio_tokens")
            .expect("the tie is stated");
        assert_eq!(row.presence, Presence::Required);
        assert_eq!(row.extents, vec![65_632, 2048]);
    }

    /// The head is over ONE codebook's alphabet, not over the text
    /// vocabulary.
    ///
    /// `[2051, 2048]` against a `[128256, 2048]` embedding table. This
    /// is the single extent that most clearly says the backbone emits
    /// audio: no text model in the catalog has a head narrower than its
    /// embedding table by a factor of sixty.
    #[test]
    fn the_backbone_head_emits_audio_codes_not_text() {
        let f = CsmFacts::csm_1b();
        assert_eq!(spec_extents(&f, "lm_head"), vec![2051, 2048]);
        assert_eq!(spec_extents(&f, "embed_text_tokens"), vec![128_256, 2048]);
        // And a tied build would state its absence instead.
        let tied = CsmFacts { tied_embeddings: true, ..f };
        let row = manifest(&tied)
            .tensors
            .into_iter()
            .find(|t| t.name == "lm_head")
            .expect("the tie is stated either way");
        assert_eq!(row.presence, Presence::Absent);
    }

    /// The measured extents, against the checkpoint's own header.
    ///
    /// Every number here was read out of `model.safetensors` rather
    /// than derived on paper, because the whole value of a manifest is
    /// that it is a measurement — arithmetic that agrees with itself
    /// and with nothing else is what the old `HF_ROWS` table was.
    #[test]
    fn the_extents_are_the_ones_the_checkpoint_ships() {
        let f = CsmFacts::csm_1b();
        for (name, want) in [
            ("backbone_model.layer.{}.self_attn.q_proj", vec![2048, 2048]),
            ("backbone_model.layer.{}.self_attn.k_proj", vec![512, 2048]),
            ("backbone_model.layer.{}.self_attn.v_proj", vec![512, 2048]),
            ("backbone_model.layer.{}.self_attn.o_proj", vec![2048, 2048]),
            ("backbone_model.layer.{}.mlp.gate_proj", vec![8192, 2048]),
            ("backbone_model.layer.{}.mlp.down_proj", vec![2048, 8192]),
            ("backbone_model.norm", vec![2048]),
            ("depth_decoder.model.embed_tokens", vec![65_632, 2048]),
            ("depth_decoder.model.inputs_embeds_projector", vec![1024, 2048]),
            ("depth_decoder.codebooks_head", vec![31, 1024, 2051]),
            ("depth_decoder.model.layer.{}.self_attn.k_proj", vec![256, 1024]),
            ("depth_decoder.model.layer.{}.mlp.down_proj", vec![1024, 8192]),
            (
                "codec_model.quantizer.acoustic_residual_vector_quantizer.layer.{}.codebook.embed_sum",
                vec![2048, 256],
            ),
            (
                "codec_model.quantizer.semantic_residual_vector_quantizer.input_proj",
                vec![256, 512, 1],
            ),
            ("codec_model.decoder_transformer.layer.{}.self_attn.q_proj", vec![512, 512]),
        ] {
            assert_eq!(spec_extents(&f, name), want, "{name} is not the extent the file ships");
        }
    }

    /// The convolutional stacks are asked for by name and not by shape.
    ///
    /// Mimi's encoder widens 1 → 64 → … → 512 across its layers, so
    /// `encoder.layer.{}.conv` has no single extent. A spec that stated
    /// one would be checked against whichever layer the checkpoint
    /// published last, which is a test that passes or fails on
    /// iteration order.
    #[test]
    fn the_codec_convolutions_are_asked_for_without_a_shape() {
        let m = manifest(&CsmFacts::csm_1b());
        for name in [
            "codec_model.encoder.layer.{}.conv",
            "codec_model.decoder.layer.{}.conv",
            "codec_model.upsample.conv",
            "codec_model.downsample.conv",
        ] {
            let row = m.tensors.iter().find(|t| t.name == name).expect("named");
            assert_eq!(row.presence, Presence::Required);
            assert!(row.extents.is_empty(), "{name} states an extent it cannot have");
        }
    }

    /// The manifest's layer count is the BACKBONE's.
    ///
    /// One number, three stacks — so it has to be documented which one
    /// it is. The backbone's, because that is the stack every per-layer
    /// row without a prefix belongs to and the one a `Deployment` would
    /// describe if there were one.
    #[test]
    fn the_stated_layer_count_is_the_backbones() {
        assert_eq!(manifest(&CsmFacts::csm_1b()).layers, 16);
        assert_eq!(manifest(&CsmFacts::csm_synthetic()).layers, 4);
    }

    /// The deployment refuses, and the refusal names what is missing.
    ///
    /// Not "unsupported model" — the text has to be usable by someone
    /// reading a log at three in the morning, so it names the three
    /// stacks and says which of them this build has no loop for.
    #[test]
    fn the_deployment_refuses_and_names_what_is_missing() {
        let err = deployment(&CsmFacts::csm_1b()).expect_err("this build serves no speech");
        assert!(matches!(err, Refusal::Unsupported(_)));
        assert_eq!(err, Refusal::Unsupported(NO_DEPLOYMENT));
        let said = err.to_string();
        for named in ["depth decoder", "codec", "backbone"] {
            assert!(said.contains(named), "the refusal does not name the {named}: {said}");
        }
        assert!(
            said.contains("this build cannot serve it"),
            "a refusal must read as a statement about the build: {said}"
        );
    }

    /// The trace refuses, and the refusal names the module that does
    /// not exist.
    ///
    /// The old shape of this answer was a defaulted method returning
    /// `None`, which a caller read as "no special handling". Here the
    /// answer is a sentence with a path in it.
    #[test]
    fn the_trace_refuses_and_names_the_module_that_is_not_there() {
        let err = trace(&CsmFacts::csm_1b()).expect_err("there is no text to trace");
        assert_eq!(err, Refusal::Unsupported(NO_TRACE));
        let said = err.to_string();
        assert!(said.contains("csm/forward"), "the refusal must name the missing module: {said}");
        assert!(said.contains("depth decoder"), "and both passes that are missing: {said}");
    }

    /// Both refusals are `Unsupported` and neither is `Malformed`.
    ///
    /// The distinction carries a promise: `Malformed` says the
    /// checkpoint contradicts itself and no build will ever serve it,
    /// `Unsupported` says THIS build cannot. A CSM checkpoint is
    /// perfectly well formed, and calling it malformed would send
    /// someone to inspect a file that is fine.
    #[test]
    fn a_csm_checkpoint_is_unserved_and_not_malformed() {
        let f = CsmFacts::csm_synthetic();
        assert!(matches!(deployment(&f), Err(Refusal::Unsupported(_))));
        assert!(matches!(trace(&f), Err(Refusal::Unsupported(_))));
        assert!(!matches!(deployment(&f), Err(Refusal::Malformed(_))));
        assert!(!matches!(trace(&f), Err(Refusal::Malformed(_))));
    }

    /// The synthetic's manifest is the synthetic's numbers.
    ///
    /// The corpus file is the only CSM config in this repository, and a
    /// fixture that transcribed it wrongly would never be caught by the
    /// 1B row's tests — the two share every line of code below and no
    /// number at all.
    #[test]
    fn the_synthetic_projects_its_own_widths() {
        let s = CsmFacts::csm_synthetic();
        assert_eq!(spec_extents(&s, "embed_text_tokens"), vec![1000, 128]);
        assert_eq!(spec_extents(&s, "lm_head"), vec![1000, 128]);
        assert_eq!(spec_extents(&s, "depth_decoder.model.embed_tokens"), vec![8 * 2048, 128]);
        assert_eq!(spec_extents(&s, "depth_decoder.model.inputs_embeds_projector"), vec![64, 128]);
        assert_eq!(spec_extents(&s, "depth_decoder.codebooks_head"), vec![7, 64, 2048]);
        assert_eq!(
            spec_extents(&s, "codec_model.quantizer.acoustic_residual_vector_quantizer.layer.{}.codebook.embed_sum"),
            vec![1024, 32]
        );
    }

    /// The two fixtures' manifests do not claim one another.
    #[test]
    fn the_1b_row_does_not_claim_the_synthetic() {
        let big = manifest(&CsmFacts::csm_1b());
        assert!(
            big.check(&implied(&CsmFacts::csm_synthetic())).is_err(),
            "a 128-wide toy must not load as a 1B speech model"
        );
        let small = manifest(&CsmFacts::csm_synthetic());
        assert!(small.check(&implied(&CsmFacts::csm_1b())).is_err());
    }
}

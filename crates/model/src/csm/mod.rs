//! The CSM lineage: speech — a backbone, a depth decoder, and a Mimi
//! codec.
//!
//! The generation that answers a question with a REFUSAL, and the
//! reason [`Variant::deployment`] and [`Variant::trace`] return a
//! `Result` at all.
//!
//! CSM loads. Its contract is written, it binds every tensor in the
//! package, and [`contract::author_csm`] narrows the fp32 checkpoint to
//! the bf16 its kernels read. What it does not have is forward text:
//! there is no `csm/forward` module in this crate and there never has
//! been. Under the design this replaces, that fact had nowhere to be
//! said. A family with no forward answered `None` from a defaulted
//! trait method, which every caller read as "nothing unusual"; a family
//! with no deployment derivation fell off the end of `FACTS_ROWS`, and
//! for CSM the llama-like fallback would have SUCCEEDED, because its
//! backbone states every key a llama reader wants under exactly the
//! spelling it wants them. The result is a stack that pages, fires, and
//! returns audio codebook indices to a caller that asked for speech.
//!
//! So the row below states what it is and refuses what it cannot do,
//! by name. The refusals live in [`project`] as `const` strings so the
//! tests can hold them to naming the missing thing rather than to being
//! non-empty.
//!
//! Chat is NOT refused, because it is not missing: CSM's turn protocol
//! is published and short, and [`chat`] implements it. A `chat()` that
//! cannot refuse — the trait's signature returns an `Arc`, not a
//! `Result` — is exactly why: a template is always answerable, and the
//! previous answer for this generation was Qwen's ChatML by fallthrough.

#[cfg(feature = "chat")]
pub mod chat;
#[cfg(feature = "contract")]
pub mod contract;

/// What a CSM checkpoint IS — ungated, because a row is written in
/// these words and a row must answer under every aspect.
pub mod spec;

/// What those numbers imply: a manifest, and two refusals.
pub mod project;

use std::sync::{Arc, OnceLock};

use crate::catalog::{Deployed, LoadShape, Variant};
use crate::deployment::{Deployment, Refusal};
use crate::manifest::Manifest;

use self::spec::CsmFacts;

/// One CSM checkpoint.
///
/// No `rope_theta` and no `norm_eps` field, which every other row in
/// the catalog carries. Both are stated by the checkpoint — 500 000 and
/// `1e-5` — and both are numbers a `Deployment` transports, so putting
/// them on a row that cannot produce one would be recording a
/// measurement nothing reads. The place they belong is
/// [`spec::CsmFacts`], the day this generation is served.
pub struct Csm {
    /// The stable name a boundary carries.
    pub id: &'static str,
    /// The numbers the package is.
    pub shape: CsmFacts,
}

/// The generation's rows.
///
/// One. `sesame/csm-1b` is the only published CSM, and the
/// `unsloth/csm-1b` mirror is the same weights under a licence that
/// does not gate downloads — a redistribution is not a second model.
///
/// The corpus's `synthetic--csm.json` gets no row: nothing has ever
/// shipped that stack. It is transcribed as [`CsmFacts::csm_synthetic`]
/// instead, which is a fixture the projections are tested through, so
/// the one CSM config checked into this repository is still read by
/// something.
pub const VARIANTS: &[Csm] = &[Csm { id: "csm-1b", shape: CsmFacts::csm_1b() }];

/// This generation's contribution to [`crate::catalog::catalog`].
///
/// The `OnceLock` is only the widening from `&Csm` to `&dyn Variant`;
/// the rows themselves are `const` and in `.rodata`.
#[must_use]
pub fn rows() -> &'static [&'static dyn Variant] {
    static ROWS: OnceLock<Vec<&'static dyn Variant>> = OnceLock::new();
    ROWS.get_or_init(|| VARIANTS.iter().map(|v| v as &'static dyn Variant).collect())
}

impl Variant for Csm {
    fn id(&self) -> &'static str {
        self.id
    }

    fn manifest(&self) -> Manifest {
        project::manifest(&self.shape)
    }

    /// The BACKBONE's, and it is worth saying which stack that is.
    ///
    /// A `LoadShape` is what a tensor-parallel split and a bind read,
    /// and both of those happen — CSM loads. The depth decoder's four
    /// layers and the codec's convolutions are not in the count because
    /// nothing splits them: `contract::author_csm` narrows every tensor
    /// in the package to bf16 and shards on the axes the loader can
    /// see, and `layers` is used to iterate the backbone's per-layer
    /// passes. A count of 52 here would make a bind walk 36 layers that
    /// do not exist under that name.
    fn load_shape(&self) -> LoadShape {
        LoadShape {
            layers: self.shape.backbone.layers,
            head_dim: self.shape.backbone.head_dim,
            // Dense: CSM routes nothing.
            n_experts: 0,
            // No selective scan.
            mamba_groups: 0,
            // No layer attends through another's pages.
            kv_shared_layers: 0,
            // `tie_word_embeddings: false` — `lm_head` is shipped. The
            // OTHER tie, `tie_codebooks_embeddings`, is between the two
            // stacks and has no field here; it is a manifest absence.
            tied_embeddings: self.shape.tied_embeddings,
        }
    }

    /// # Errors
    ///
    /// Always [`Refusal::Unsupported`]. See
    /// [`project::NO_DEPLOYMENT`] for the sentence an operator is
    /// shown; it names the three stacks and which of them this build
    /// has no loop for.
    fn deployment(&self, load: Deployed<'_>) -> Result<Deployment, Refusal> {
        let _ = load;
        project::deployment(&self.shape)
    }

    /// The one aspect that is fully answered.
    ///
    /// Same author for both namings: the MLX table never held a `csm`
    /// row, and `author_csm` is written against tensor names rather
    /// than against a spelling — it walks whatever the builder resolved
    /// and casts it.
    #[cfg(feature = "contract")]
    fn author(
        &self,
        builder: &mut crate::shared::builder::Builder<'_>,
    ) -> Result<(), model_loader::error::Error> {
        self::contract::author_csm(builder)
    }

    /// # Errors
    ///
    /// Always [`Refusal::Unsupported`], carrying
    /// [`project::NO_TRACE`] — there is no `csm/forward` module to
    /// call into.
    #[cfg(feature = "forward")]
    fn trace(
        &self,
        class: model_compiler::trace::FireClass,
        load: Deployed<'_>,
    ) -> Result<model_compiler::trace::ForwardPlan, Refusal> {
        let _ = (class, load);
        project::trace(&self.shape)
    }

    /// CSM's own speaker-id protocol — `<bos>[0]…<eos>` — and not
    /// ChatML, which is what the registry's `_ =>` arm used to hand it.
    #[cfg(feature = "chat")]
    fn chat(&self, tokenizer: Arc<tokenizer::Tokenizer>) -> Arc<dyn crate::instruct::Instruct> {
        Arc::new(self::chat::CsmInstruct::new(tokenizer))
    }
}

#[cfg(test)]
mod tests {
    use super::{Csm, Deployed, VARIANTS, Variant, project, rows};
    use crate::deployment::Refusal;
    use crate::manifest::{Observed, Presence};

    fn only_row() -> &'static Csm {
        VARIANTS.first().expect("the generation has a row")
    }

    /// Ids are what a boundary carries, so they are held to the shape a
    /// boundary can carry: unique, non-empty, lowercase, hyphenated.
    #[test]
    fn every_row_has_an_id_a_boundary_can_carry() {
        let mut seen = std::collections::BTreeSet::new();
        for v in VARIANTS {
            assert!(!v.id.is_empty(), "a row with no name cannot be asked for");
            assert!(seen.insert(v.id), "'{}' names two rows, and a lookup would pick one", v.id);
            assert!(
                v.id.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-'),
                "'{}' is not a name a URL or a CLI flag carries unescaped",
                v.id
            );
        }
        assert_eq!(only_row().id, "csm-1b");
    }

    /// `rows()` hands the catalog exactly the table, once.
    #[test]
    fn the_catalog_sees_one_entry_per_row() {
        assert_eq!(rows().len(), VARIANTS.len());
        assert_eq!(rows().len(), 1, "one published CSM, one row");
        assert_eq!(rows()[0].id(), only_row().id);
    }

    /// The row and the fixture are the same measurement.
    #[test]
    fn the_row_is_the_committed_fixture() {
        assert_eq!(only_row().shape, super::CsmFacts::csm_1b());
    }

    /// The row IDENTIFIES its checkpoint in full, even though it cannot
    /// serve it.
    ///
    /// This is the distinction the catalog is for: refusing to deploy
    /// is not the same as not knowing what a file is. A CSM checkpoint
    /// arriving at this build gets named, its tensors get checked, and
    /// then it gets a refusal that says what is missing — rather than
    /// "unknown model type", which sends someone to look for a typo.
    #[test]
    fn the_row_identifies_the_checkpoint_it_cannot_serve() {
        let v = only_row();
        let m = v.manifest();
        assert_eq!(m.layers, 16);
        let seen = Observed::from_pairs(
            m.tensors
                .iter()
                .filter(|t| t.presence != Presence::Absent)
                .map(|t| (t.name.replace("{}", "0"), t.extents.clone())),
        );
        assert!(m.check(&seen).is_ok(), "{}", m.check(&seen).unwrap_err());
        assert!(v.deployment(Deployed::single()).is_err());
    }

    /// Every field of the load shape, and each one is the backbone's.
    #[test]
    fn the_load_shape_describes_the_backbone_and_says_so() {
        let s = only_row().load_shape();
        assert_eq!(s.layers, 16, "the backbone's depth, not the package's 16 + 4");
        assert_eq!(s.head_dim, 64);
        assert_eq!(s.n_experts, 0, "csm routes nothing");
        assert_eq!(s.mamba_groups, 0, "csm scans nothing");
        assert_eq!(s.kv_shared_layers, 0);
        assert!(!s.tied_embeddings, "`tie_word_embeddings: false` — lm_head is shipped");
    }

    /// The deployment refuses, and the refusal names what is missing.
    #[test]
    fn the_row_refuses_to_deploy_and_names_the_stacks() {
        let err =
            only_row().deployment(Deployed::single()).expect_err("this build serves no speech");
        assert_eq!(err, Refusal::Unsupported(project::NO_DEPLOYMENT));
        let said = err.to_string();
        for named in ["backbone", "depth", "codec"] {
            assert!(said.contains(named), "the refusal does not name the {named}: {said}");
        }
    }

    /// And it is `Unsupported`, not `Malformed` — a statement about the
    /// BUILD.
    ///
    /// A CSM checkpoint is perfectly well formed. Calling it malformed
    /// would send someone to inspect a file that is fine, which is the
    /// most expensive kind of wrong error message.
    #[test]
    fn the_refusal_blames_the_build_and_not_the_checkpoint() {
        let err = only_row().deployment(Deployed::single()).expect_err("refuses");
        assert!(matches!(err, Refusal::Unsupported(_)));
        assert!(err.to_string().starts_with("this build cannot serve it"));
    }

    /// The refusal carries a SENTENCE, and that is the difference this
    /// variant's payload makes.
    ///
    /// `Unsupported` was a unit variant returned from nine sites, and
    /// all nine reached an operator as "no deployment derivation for
    /// this model type" — which names neither the model nor the missing
    /// thing, and is the same message whether you handed the build a
    /// CSM, a Kimi or a typo. Length is the crude proxy for "a sentence
    /// rather than a label", and the named terms below are the actual
    /// property: the text points at what is absent.
    #[test]
    fn the_refusal_carries_a_sentence_rather_than_a_label() {
        let Refusal::Unsupported(why) =
            only_row().deployment(Deployed::single()).expect_err("refuses")
        else {
            panic!("csm is unsupported, not malformed");
        };
        assert!(
            why.len() > 40,
            "'{why}' reads as a label; the payload exists so an operator learns what this \
             build wanted and did not have"
        );
        assert!(why.contains("csm"), "the refusal must name the model it is about");
    }

    /// Nothing is advertised, because there is no deployment to
    /// advertise it in — and that is the right shape for this fact.
    ///
    /// `Advertised` rides inside a `Deployment` rather than being a
    /// `Variant` method, so a row that cannot be served cannot claim a
    /// context ceiling or a media front end either. Under the design
    /// this replaces the three values were read off a resident
    /// `HfConfig`, which a CSM has just as much as a Qwen does: the
    /// driver would have answered `max_position_embeddings` 2048 and
    /// `model_type` "csm" for a stack it cannot fire, and a guest
    /// program would have believed it.
    #[test]
    fn a_row_that_cannot_deploy_advertises_nothing() {
        assert!(
            only_row().deployment(Deployed::single()).is_err(),
            "if this ever succeeds, this row owes an `advertised` — arch `csm`, ceiling 2048 \
             from `max_position_embeddings`, and `media_encode` false because the codec is \
             not an encode-entry tower"
        );
    }

    /// The trace refuses too, and names the module that is not there.
    #[cfg(feature = "forward")]
    #[test]
    fn the_row_refuses_to_trace_and_names_the_missing_module() {
        use model_compiler::trace::FireClass;
        for class in [FireClass::Prefill, FireClass::Decode] {
            let err = only_row()
                .trace(class, Deployed::single())
                .expect_err("there is no csm forward text");
            assert_eq!(err, Refusal::Unsupported(project::NO_TRACE));
            assert!(
                err.to_string().contains("csm/forward"),
                "the refusal must name the module that does not exist"
            );
        }
    }

    /// Chat is answered, not refused — and it is CSM's protocol.
    ///
    /// The concrete thing this prevents: the registry's `_ =>` arm used
    /// to answer ChatML here, so a CSM was prompted with
    /// `<|im_start|>user` — tokens its llama-3 vocabulary does not
    /// contain — and told to stop on an `<|im_end|>` it can never emit.
    #[cfg(feature = "chat")]
    #[test]
    fn chat_is_answered_with_the_speaker_protocol_and_not_chatml() {
        use crate::instruct::Instruct;
        use std::sync::Arc;
        let vocab: Vec<String> =
            ["<|begin_of_text|>", "<|end_of_text|>", "[0]", "[1]", "Hello"]
                .iter()
                .map(ToString::to_string)
                .collect();
        let tok = Arc::new(tokenizer::Tokenizer::from_vocab(&vocab));
        let inst = only_row().chat(tok.clone());
        let rendered = tok.decode(&inst.user("Hello"), false);
        assert_eq!(rendered, "<|begin_of_text|>[0]Hello<|end_of_text|>");
        assert!(!rendered.contains("<|im_start|>"), "chatml is not this model's protocol");
        assert_eq!(tok.decode(&inst.cue(), false), "<|begin_of_text|>[1]");
    }
}

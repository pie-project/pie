//! Kimi K2.
//!
//! One generation, one directory, and now one ROW. What this generation
//! was before is the three-table shape the catalog exists to remove:
//! `deployment_cuda::FACTS_ROWS` held `"kimi_k2"` and `"kimi_k25"`
//! against a `kimi_k2_facts_from_hf` that re-parsed `config.json` on
//! every load, `contract::HF_ROWS` held the same two strings against
//! `author_kimi`, and `instruct::create` held `"kimi_k2" | "kimi_k25" |
//! "kimi_k3"` against a template that lived in this directory. Three
//! keys, three tables, nothing holding them to each other.
//!
//! Read `VARIANTS` and then `impl Variant`. What this generation adds
//! over its shape is what a shape cannot state: which pass authors it
//! and which template speaks for it — and the second of those is shared
//! with kimi-k3, so it lives in `shared/` and this directory
//! re-exports it.

#[cfg(feature = "chat")]
pub mod chat;
#[cfg(feature = "contract")]
pub mod contract;

/// The declared forward — MLA over a dense-prefix MoE stack with WNA16
/// experts.
pub mod forward;

/// The SHAPE, ungated: a catalog row is written in these words.
pub mod spec;

/// The three projections a row makes — its manifest, its deployment,
/// its text.
pub mod project;

// `Arc` reaches this module only through `Variant::chat`, so the
// import carries that method's gate. It used to ride along with
// `OnceLock`, which `rows()` needed unconditionally until
// `rows_of!` absorbed it.
#[cfg(feature = "chat")]
use std::sync::Arc;

use crate::catalog::{Deployed, LoadShape, Variant};
use crate::deployment::Advertised;
use crate::manifest::Manifest;
use spec::KimiFacts;

/// The coarse family label a GUEST PROGRAM matches on.
///
/// `kimi_k2`, the string `deployment_cuda::FACTS_ROWS`,
/// `contract::HF_ROWS` and `instruct::create` each keyed on separately.
/// What it is NOT any more is a dispatch key — nothing in this crate
/// matches on it — so the one job left is being recognisable to an
/// inferlet through `engine`'s `model.arch_name()`.
///
/// It is the `model_type` and NOT what the worker's stem heuristic
/// produces, and that gap is worth stating rather than papering over:
/// `moonshotai/Kimi-K2-Instruct` publishes
/// `architectures: ["DeepseekV3ForCausalLM"]`, so
/// `read_hf_config_defaults` reduces a real K2 config to `deepseekv3`,
/// which no row here claims. (The corpus's `synthetic--kimi-k2.json`
/// writes `KimiK2ForCausalLM`, which reduces to `kimik2` — a third
/// spelling, and still not this one.) K2 reuses DeepSeek's modelling
/// code and says so in its `architectures`; a label of `deepseekv3`
/// would then tell a guest program this is a DeepSeek, which is the
/// more damaging of the two wrong answers. Sibling `kimi_k3` makes the
/// same choice for the same reason.
const ARCH: &str = "kimi_k2";

/// One Kimi K2 checkpoint.
///
/// A newtype over the shape for the exemplar's reason: `chat` and
/// `author` are answers the numbers cannot give, and this generation's
/// two are its own template and its own MLA authoring pass.
pub struct KimiK2 {
    /// The stable name a boundary carries.
    pub id: &'static str,
    /// The numbers.
    pub shape: KimiFacts,
    /// Rope's base frequency. NOT in the shape, because the tracer takes
    /// it per-layer through `Deployment` rather than through the shape.
    pub rope_theta: f32,
    /// RMSNorm epsilon — a constant of the checkpoint that no tensor
    /// extent carries, so a row must state it.
    pub norm_eps: f32,
    /// The config asks for YaRN, so the CUDA reading binds
    /// `rope_yarn_original_bf16` rather than the plain rope. A BINDING
    /// fact, kept beside the row because the row is where a load reads
    /// it from.
    pub rope_yarn: bool,
    /// The head owns its own table. K2 unties.
    pub tied_embeddings: bool,
    /// The published context ceiling.
    ///
    /// On the row rather than in a shared constant because it is a
    /// MEASUREMENT of a checkpoint, and this generation's only
    /// committed configs are synthetics whose numbers are a parser's
    /// fixtures rather than a release's claim — so the field records
    /// where the value came from by being per row. A second row that
    /// ever joins this one states its own.
    pub max_model_len: u32,
}

/// The generation's rows.
///
/// ONE, and the one is the argument. `deployment_cuda`'s `FACTS_ROWS`
/// carried `"kimi_k2"` and `"kimi_k25"` as separate keys, and
/// `instruct::create` carried both again as separate architecture
/// strings — but `synthetic--kimi-k2.json` and `synthetic--kimi-k25.json`
/// state the SAME geometry, K2.5 differing only in nesting its config
/// under a `text_config` key. A packaging difference is not a model, and
/// a second row here would be two rows one checkpoint matches, which is
/// [`crate::catalog::Unmatched::Ambiguous`] — the catalog refuses to
/// serve rather than picking one. Two keys became one row on purpose.
pub const VARIANTS: &[KimiK2] = &[
    // moonshotai/Kimi-K2-Instruct. 1T total, 32B active: 384 experts at
    // 2048, eight of them per token, over a 61-layer MLA stack whose
    // first layer is dense.
    KimiK2 {
        id: "kimi-k2",
        shape: KimiFacts {
            layers: 61,
            vocab: 163_840,
            hidden: 7168,
            dense_intermediate: 18_432,
            dense_layers: 1,
            attn: spec::KimiMlaFacts {
                hidden: 7168,
                heads: 64,
                q_lora_rank: 1536,
                kv_lora_rank: 512,
                qk_nope_head_dim: 128,
                qk_rope_head_dim: 64,
                v_head_dim: 128,
                output_gate: false,
            },
            moe: spec::KimiMoeFacts {
                num_experts: 384,
                top_k: 8,
                // As `spec.rs` -- K2 publishes `false`.
                norm_topk_prob: false,
                routed_scaling: 2.0,
                moe_intermediate: 2048,
                shared_intermediate: 2048,
            },
        },
        rope_theta: 50_000.0,
        norm_eps: 1e-6,
        rope_yarn: true,
        tied_embeddings: false,
        // From the PUBLISHED `moonshotai/Kimi-K2-Instruct` config,
        // which states `max_position_embeddings: 131072`. Not from
        // `synthetic--kimi-k2.json`: the synthetics in the corpus are a
        // normalizer's fixtures — a handful of layers, a toy vocabulary
        // and a 2048 ceiling to match — and taking a ceiling from one
        // would advertise the fixture instead of the model.
        max_model_len: 131_072,
    },
];

/// The YaRN rescaling `moonshotai/Kimi-K2-Instruct` publishes.
///
/// Four of the five are transcribed from the published `config.json`:
/// `{"beta_fast": 1.0, "beta_slow": 1.0, "factor": 32.0, "mscale": 1.0,
/// "mscale_all_dim": 1.0, "original_max_position_embeddings": 4096,
/// "type": "yarn"}`. NOT from `synthetic--kimi-k2.json`, which states
/// `rope_scaling: null` — the synthetics are a normalizer's fixtures,
/// and reading a rope ladder off one would be reading the fixture.
///
/// `beta_fast: 1.0` is deliberate and looks wrong: the YaRN paper's
/// values are 32 and 1, and gpt-oss states exactly those. K2 publishes
/// 1.0 for both, which HF takes as-is (`rope_parameters_dict.get(
/// "beta_fast") or 32` — 1.0 is truthy), so it is the checkpoint's
/// answer and not a missing field.
///
/// The fifth, `attention_factor`, the config does not state, and it is
/// NOT the `0.1 * ln(factor) + 1` gpt-oss carries. HF infers it as
/// `get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim)`
/// when BOTH `mscale` and `mscale_all_dim` are present, and falls back
/// to the plain formula only when they are not. K2 states `1.0` for
/// both, so numerator and denominator are the same number and the
/// factor is exactly `1.0` — where the formula would have given
/// `1.3466` and lengthened every attention logit by a third.
///
/// `original_max_position` is 4096 and the row's `max_model_len` is
/// 131_072: 4096 * 32 is exactly the extended context, the same
/// arithmetic gpt-oss's pair satisfies.
const ROPE_SCALING: crate::deployment::RopeScaling = crate::deployment::RopeScaling::Yarn {
    factor: 32.0,
    beta_fast: 1.0,
    beta_slow: 1.0,
    attention_factor: 1.0,
    original_max_position: 4_096,
    // OMITTED by the config, which is HF's default.
    truncate: true,
};

crate::rows_of!(KimiK2);

impl Variant for KimiK2 {
    fn id(&self) -> &'static str {
        self.id
    }

    fn manifest(&self) -> Manifest {
        project::manifest(&self.shape, self.tied_embeddings)
    }

    /// The head dim an AUTHORING pass needs, and for MLA that is the
    /// latent page row rather than a per-head width: the projections a
    /// tensor-parallel split cuts are `q_b_proj` and `kv_b_proj`, whose
    /// rows are heads of `qk_nope + qk_rope` and `qk_nope + v` — and
    /// both of those are multiples of this. A split that lands
    /// mid-latent produces a contract that compiles and a model that is
    /// wrong.
    fn load_shape(&self) -> LoadShape {
        LoadShape::mixture(
            self.shape.layers,
            self.shape.attn.kv_a_width(),
            self.shape.moe.num_experts,
            self.tied_embeddings,
        )
    }

    /// # Errors
    ///
    /// [`crate::deployment::Refusal::Unsupported`] — this build
    /// provisions no MLA store, and the row says so at the door.
    fn deployment(
        &self,
        load: Deployed<'_>,
    ) -> Result<crate::deployment::Deployment, crate::deployment::Refusal> {
        let _ = load;
        project::deployment(
            &self.shape,
            self.rope_theta,
            self.norm_eps,
            self.rope_yarn,
            Advertised {
                arch: ARCH,
                max_model_len: self.max_model_len,
                // No tower. K2 is a text stack in every published
                // release, so `Deployment::towers` stays empty and the
                // driver's encode entry has nothing here to serve.
                media_encode: false,
            },
        )
    }

    #[cfg(feature = "contract")]
    fn author(
        &self,
        builder: &mut crate::shared::builder::Builder<'_>,
    ) -> Result<(), model_loader::error::Error> {
        contract::author_kimi(builder)
    }

    /// # Errors
    ///
    /// Whatever [`Self::deployment`] refuses, because it is asked first
    /// and the two questions are one question: "does this build serve
    /// this row". The text itself traces for both fire classes.
    ///
    /// It has to be asked here rather than assumed. This doc used to say
    /// the refusal happened "before anything reaches here" — but nothing
    /// sequenced the two calls, so a build with no MLA latent store
    /// refused at the door and handed out a fire anyway.
    fn trace(
        &self,
        class: model_ir::trace::FireClass,
        load: Deployed<'_>,
    ) -> Result<model_ir::trace::ForwardPlan, crate::deployment::Refusal> {
        // METAL, refused by name. `llama_like_metal` is the only Metal
        // text in this build and it is not this model's — see
        // [`project::NO_METAL`] for what it states instead and why
        // reaching for it would trace a different model under this
        // row's id. The refusal is stated HERE, at the row, rather than
        // consulted from a list of architecture strings a driver keeps:
        // a list is a fourth place for the answer to live and a fourth
        // place for it to be wrong.
        if let crate::catalog::Backend::Metal(_) = load.backend {
            return Err(crate::deployment::Refusal::Unsupported(project::NO_METAL));
        }
        // The plan this row WOULD fire. It cannot run in this build --
        // `deployment` refuses for a store nothing provisions -- and it is
        // written anyway, because the alternative is a row that stops being
        // able to serve the day one is. `tests/a_store_this_build_does_not
        // _provision.rs` holds the refusal so the day it changes is a test
        // failure naming this line rather than a silent revival.
        self.deployment(load)
            .map(|_| project::trace(&self.shape, self.rope_yarn, class))
    }

    /// Kimi's own `<|im_middle|>` protocol, which reads like ChatML and
    /// is not. Stated by the row; the arm that used to state it keyed on
    /// `"kimi_k2" | "kimi_k25" | "kimi_k3"`, three strings for one
    /// answer, in a table that could disagree with the other two.
    #[cfg(feature = "chat")]
    fn chat(&self, tokenizer: Arc<tokenizer::Tokenizer>) -> Arc<dyn crate::instruct::Instruct> {
        Arc::new(crate::shared::kimi::KimiInstruct::new(tokenizer))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(id: &str) -> &'static KimiK2 {
        VARIANTS.iter().find(|v| v.id == id).expect("row present")
    }

    /// The fixture and the row are the same measurement, so if they
    /// disagree one of them is wrong. `KimiFacts::kimi_k2()` was
    /// committed as a measurement of `moonshotai/Kimi-K2-Instruct`; so
    /// is the row, and the row is now the only one a driver reads.
    #[test]
    fn the_row_agrees_with_the_committed_fixture() {
        assert_eq!(row("kimi-k2").shape, KimiFacts::kimi_k2());
    }

    /// The three capability answers come off the ROW.
    ///
    /// They cannot be read back out of a `Deployment` here, because
    /// this build refuses to serve MLA and [`Variant::deployment`]
    /// therefore returns a refusal — so the assertion is on what the
    /// row hands the projection, which is the same statement one step
    /// earlier. `project`'s own
    /// `the_rows_advertised_label_is_carried_and_not_rewritten` holds
    /// the other half: that the projection does not rewrite it.
    #[test]
    fn the_row_answers_what_the_driver_advertises() {
        assert_eq!(ARCH, "kimi_k2", "the string three tables used to key on");
        assert!(
            !ARCH.is_empty(),
            "an empty arch is a model no guest predicate can recognise"
        );
        for v in VARIANTS {
            assert_eq!(
                v.max_model_len, 131_072,
                "{}: the published Kimi-K2-Instruct config states 131072",
                v.id,
            );
            assert_ne!(v.max_model_len, 0, "{}: 0 is 'the row does not say'", v.id);
            assert_ne!(
                v.max_model_len, 2048,
                "{}: 2048 is `synthetic--kimi-k2.json`'s fixture ceiling, not a release's",
                v.id,
            );
        }
    }

    /// The bool that picks the YaRN kernel is the bool that carries its
    /// numbers.
    ///
    /// `rope_yarn: true` says "the config asks for YaRN, so the CUDA
    /// reading binds `rope_yarn_original_bf16`", and the projection said
    /// `rope_scaling: None`, which that field's doc defines as "this
    /// stack uses its `rope_theta` ladder unrescaled". One checkpoint,
    /// two answers. `fire::launch` resolves `None` to `yarn: [0.0; 4]`
    /// on the one kernel that reads it — a degenerate ramp, in that
    /// file's own words — so the row would have attended at the wrong
    /// wavelengths past 4096 and degraded fluently, which is the way
    /// this crate keeps finding.
    ///
    /// The numbers are the published config's, checked here so a
    /// transcription error is a failure rather than a wavelength.
    #[test]
    fn the_bool_that_picks_the_yarn_kernel_carries_its_numbers() {
        for v in VARIANTS {
            // Through the TOTAL projection, not `Variant::deployment`:
            // this build provisions no MLA store, so the row refuses at
            // the door (`the_row_loads_and_refuses_to_serve_in_this_
            // build`) and the statement would be unobservable. The
            // refusal is about what this binary provides; the ladder is
            // a fact about the checkpoint either way.
            let d = project::deployment_unrefused(
                &v.shape,
                v.rope_theta,
                v.norm_eps,
                v.rope_yarn,
                Advertised {
                    arch: ARCH,
                    max_model_len: v.max_model_len,
                    media_encode: false,
                },
            );
            assert_eq!(
                v.rope_yarn,
                d.rope_scaling.is_some(),
                "{}: `rope_yarn` picks the kernel and `rope_scaling` \
                 feeds it; a row that answers these differently sends \
                 the YaRN rope a zeroed ramp",
                v.id,
            );
            let Some(crate::deployment::RopeScaling::Yarn {
                factor,
                beta_fast,
                beta_slow,
                attention_factor,
                original_max_position,
                truncate,
            }) = d.rope_scaling
            else {
                panic!("{}: K2 publishes `type: yarn`", v.id);
            };
            // The published `moonshotai/Kimi-K2-Instruct` config, field
            // for field. `beta_fast` is 1.0 and not the paper's 32.0:
            // K2 states it, and HF takes a stated 1.0 as-is.
            assert_eq!((factor, beta_fast, beta_slow), (32.0, 1.0, 1.0), "{}", v.id);
            assert_eq!(original_max_position, 4_096, "{}", v.id);
            // OMITTED by the config, so the row states HF's default. Unlike
            // `attention_factor` above there is no second field to infer it
            // from -- an absent `truncate` means the fallback and nothing
            // else -- but writing it down is what keeps the row readable
            // beside gpt-oss, which states the opposite.
            assert!(truncate, "{}: K2's config omits `truncate`", v.id);
            // NOT `0.1 * ln(32) + 1`. That formula is HF's fallback for
            // a config that omits `mscale`/`mscale_all_dim`; K2 states
            // both as 1.0, so the inferred ratio is exactly one. Taking
            // gpt-oss's 1.3466 here would scale every attention logit by
            // a third with nothing to fault on.
            assert!(
                (attention_factor - 1.0).abs() < 1e-6,
                "{}: mscale == mscale_all_dim makes the ratio 1.0, not \
                 the {:.4} the plain formula gives",
                v.id,
                0.1f32.mul_add(32.0f32.ln(), 1.0),
            );
            // 4096 * 32 is exactly the advertised ceiling, which is what
            // this pair is supposed to satisfy.
            assert_eq!(
                original_max_position * (factor as u32),
                v.max_model_len,
                "{}: the original context times the factor is the \
                 extended one, or one of them came off the wrong line",
                v.id,
            );
        }
    }

    /// THE LABEL IS THE `model_type` AND THE BOUNDARY CANNOT DERIVE IT,
    /// which is a limitation worth pinning rather than hiding.
    ///
    /// A real K2 config states `architectures: ["DeepseekV3ForCausalLM"]`
    /// — K2 reuses DeepSeek's modelling code — so the worker's stem
    /// heuristic reduces it to `deepseekv3` and never to anything this
    /// row claims. Of the two available wrong answers, telling a guest
    /// program that a Kimi is a DeepSeek is the worse one, so the row
    /// keeps the family's own name. Sibling `kimi_k3` chooses the same
    /// way. If this ever changes, it changes here and in `kimi_k3`
    /// together.
    #[test]
    fn the_label_is_the_family_the_checkpoint_names_itself() {
        let stem = "DeepseekV3ForCausalLM"
            .to_lowercase()
            .strip_suffix("forcausallm")
            .expect("the suffix the worker strips")
            .to_string();
        assert_eq!(stem, "deepseekv3", "what a published K2 config reduces to");
        assert_ne!(
            stem, ARCH,
            "the heuristic cannot reach this label, and the row says so"
        );
        assert!(
            ARCH.starts_with("kimi_"),
            "the family's own name, spelled as `kimi_k3` is"
        );
    }

    /// K2 and K2.5 are ONE row. The two synthetics differ in where the
    /// numbers sit in the JSON, not in what the numbers are — and a row
    /// per packaging would be two rows one checkpoint matches, which the
    /// catalog reports as ambiguous rather than resolving by order.
    #[test]
    fn the_two_keys_are_one_model() {
        assert_eq!(VARIANTS.len(), 1);
        assert_eq!(rows().len(), 1);
        assert_eq!(rows()[0].id(), "kimi-k2");
        assert!(
            !crate::catalog::ids()
                .iter()
                .any(|id| id.contains("k2.5") || id.contains("k25")),
            "a nesting of config keys is not a second model",
        );
    }

    /// Every row answers every question, which is what having no default
    /// bodies buys — and for this generation one of the answers is a
    /// refusal, which is still an answer.
    #[test]
    fn every_row_projects() {
        for v in VARIANTS {
            let m = v.manifest();
            assert_eq!(m.layers, v.shape.layers);
            assert!(!m.tensors.is_empty());
            let ls = v.load_shape();
            assert_eq!(ls.layers, v.shape.layers);
            assert_eq!(ls.n_experts, v.shape.moe.num_experts);
            assert_eq!(ls.head_dim, 576, "the latent page row, unpadded");
            assert_eq!(ls.mamba_groups, 0, "no mixer here");
            assert_eq!(ls.kv_shared_layers, 0, "every layer owns its pages");
            assert!(!ls.tied_embeddings);
            assert!(
                matches!(
                    v.deployment(Deployed::single()),
                    Err(crate::deployment::Refusal::Unsupported(_))
                ),
                "no MLA store is built here",
            );
        }
    }

    /// The id is what a boundary carries, so it says the MODEL and not
    /// an encoding of it: the same checkpoint ships bf16 and W4A16.
    #[test]
    fn the_id_names_a_model_and_not_a_packing() {
        for v in VARIANTS {
            assert_eq!(v.id(), v.id);
            assert!(
                v.id.chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
            );
            for banned in ["fp8", "int4", "bf16", "awq", "w4a16"] {
                assert!(!v.id.contains(banned), "{} names a packing", v.id);
            }
        }
    }

    /// A row that LOADS and does not SERVE says so at the door. The
    /// manifest is complete, the load shape is complete, the authoring
    /// runs — and the deployment refuses, which is the whole reason the
    /// trait's signature carries a `Result`.
    #[test]
    fn the_row_loads_and_refuses_to_serve_in_this_build() {
        let v = row("kimi-k2");
        assert!(
            v.manifest()
                .tensors
                .iter()
                .any(|t| t.name.contains("kv_b_proj"))
        );
        assert_eq!(v.load_shape().layers, 61);
        assert!(
            matches!(
                v.deployment(Deployed::single()),
                Err(crate::deployment::Refusal::Unsupported(_))
            ),
            "no MLA store is built here, and that is not a property of Kimi K2",
        );
    }

    /// The trace is still owed and still given: a build that cannot
    /// serve the row can still compile its text, which is what keeps the
    /// goldens honest.
    ///
    /// Asked of [`project::trace`] and not of [`Variant::trace`], and the
    /// distinction is the same one `plan` draws against `deployment`: the
    /// projection is TOTAL, a fact about the row, while the trait method
    /// is capability-gated and must refuse exactly when the door does.
    /// Asking the gated one here is what let this test and
    /// `catalog::tests::a_row_that_cannot_deploy_cannot_trace_either`
    /// state opposite things about the same call.
    #[test]
    fn every_row_traces_both_fire_classes() {
        use model_ir::trace::FireClass;
        for v in VARIANTS {
            for class in [FireClass::Prefill, FireClass::Decode] {
                let plan = project::trace(&v.shape, v.rope_yarn, class);
                assert!(plan.family.starts_with("kimi.cuda."), "{}", plan.family);
                assert!(!plan.ops.is_empty());
            }
        }
    }

    /// The chat answer is a CALL and not a table row, and it reaches the
    /// same words `kimi_k3` reaches — which is why they live in
    /// `shared/` and not in this directory.
    #[cfg(feature = "chat")]
    #[test]
    fn every_row_answers_the_chat_question() {
        use tokenizer::Tokenizer;

        let words: Vec<String> = ["<|im_user|>", "<|im_middle|>", "<|im_end|>", "user", "Hi"]
            .iter()
            .map(|s| (*s).to_string())
            .collect();
        let tok = Arc::new(Tokenizer::from_vocab(&words));
        for v in VARIANTS {
            let turn = tok.decode(&v.chat(tok.clone()).user("Hi"), false);
            assert_eq!(turn, "<|im_user|>user<|im_middle|>Hi<|im_end|>", "{}", v.id);
            assert!(
                !turn.contains("<|im_start|>"),
                "the `_ =>` ChatML arm is gone"
            );
        }
    }

    /// The author is reached from the ROW. This exercises the dispatch
    /// the catalog replaced rather than the pass itself: `HF_ROWS` kept
    /// `"kimi_k2"` pointing at one author and `FACTS_ROWS` kept the same
    /// string pointing at a shape, with nothing holding the two to each
    /// other.
    #[cfg(feature = "contract")]
    #[test]
    fn every_row_authors() {
        use model_loader::checkpoint::CheckpointMetadata;
        use model_loader::plan::StorageTarget;

        let metadata = CheckpointMetadata {
            files: Vec::new(),
            tensors: Vec::new(),
        };
        let encoding = crate::encoding::Encoding::dense();
        let target = StorageTarget::default();
        let policy = crate::shared::policy::Policy::default();

        for v in VARIANTS {
            let mut builder = crate::shared::builder::Builder::new(
                &metadata,
                v.id(),
                v.load_shape(),
                &encoding,
                &target,
                &policy,
            );
            v.author(&mut builder)
                .unwrap_or_else(|e| panic!("{} refused to author: {e:?}", v.id));
        }
    }

    /// A METAL load is refused BY NAME rather than traced as a llama.
    ///
    /// The guard that replaces `driver-metal`'s `LLAMA_LIKE` table. That
    /// table answered "does this build serve you" from an architecture
    /// STRING reduced by `canonical()`, in a driver, before any text was
    /// traced — so it could say yes to a row this build cannot resolve
    /// (it listed `gpt_oss`, whose every publication either fails this
    /// crate's manifest or names tensors `driver-metal` has no handle
    /// for) and no to one whose text it models (it omitted `gemma3`). The row answers now, and what it answers with is a
    /// sentence naming what is missing.
    ///
    /// The comparison is against [`project::NO_METAL`] itself and not a
    /// paraphrase, so the sentence a caller is shown is the sentence
    /// this test pins — `csm`'s `NO_TRACE` sets the same shape.
    #[test]
    fn a_metal_load_is_refused_by_name_and_not_traced_as_a_llama() {
        use crate::catalog::{Backend, Deployed, MetalBinding};
        use crate::deployment::Refusal;
        use model_ir::trace::FireClass;

        let bind = MetalBinding {
            qmm_partial_rows: false,
            qmm_fp16_precast: true,
            qmm_tile: None,
            quant_group: 64,
            quant_bits: 4,
            router_quant_group: 0,
            router_quant_bits: 0,
            moe_mxfp4: false,
            fuse_residual_gemv: true,
            paged_multi_batch: true,
            qmm_multi_batch: true,
            add_bias: false,
            fused_qk_rope: false,
        };
        assert!(!VARIANTS.is_empty());
        for v in VARIANTS {
            for class in [FireClass::Prefill, FireClass::Decode] {
                let err = v
                    .trace(class, Deployed::metal(&bind))
                    .expect_err("this build has no Metal text for this generation");
                assert_eq!(
                    err,
                    Refusal::Unsupported(project::NO_METAL),
                    "`{}` refused a Metal load with a sentence that is not the \
                     one the row states",
                    v.id
                );
            }
        }
        // And the refusal is about the BACKEND and nothing else.
        //
        // Stated as "not the Metal sentence" rather than `is_ok()`,
        // because this family's CUDA answer is not `Ok` either: MLA has
        // no store in ANY build, `deployment` has always said so, and
        // `trace` now asks it first instead of firing a text for a row
        // the door already turned away. Asserting `is_ok()` here would
        // make this test a second, quieter claim that the MLA store is
        // built -- which is the coupling `Deployed::metal` exists to
        // avoid. What "unchanged" means is that no CUDA caller ever sees
        // a sentence about Metal.
        for v in VARIANTS {
            let cuda = v.trace(FireClass::Decode, Deployed::single());
            assert_ne!(
                cuda.as_ref().err(),
                Some(&Refusal::Unsupported(project::NO_METAL)),
                "`{}` answered a CUDA load with a sentence about Metal",
                v.id
            );
        }
        // A `Backend::Cuda` is what `Deployed::single()` states, so the
        // arm above is reached by every existing caller unchanged.
        assert!(matches!(Deployed::single().backend, Backend::Cuda));
    }
}

/// This generation's tensor names, in every vocabulary that spells them.
#[cfg(feature = "contract")]
pub mod import;

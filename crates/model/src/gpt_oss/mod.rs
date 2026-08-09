//! The GPT-OSS lineage: a mixture with MXFP4 expert triplets, an
//! alternating sliding window, and ATTENTION SINKS.
//!
//! Three facts about this generation are why its rows are worth reading
//! carefully.
//!
//! The sinks first: one learned logit per query head, appended to the
//! softmax denominator. No other row in this build ships them, and they
//! are what makes the attention statement here produce an LSE beside its
//! output. A row states them, so a deployment that has them and a text
//! that expects them come from the same place.
//!
//! Then the window: gpt-oss alternates sliding and full attention from
//! layer 0. The old path derived that list in `gpt_oss_facts_from_hf`,
//! bound it into `GptOssCudaFacts::window_left`, and then had
//! `PlannedFamily::window_by_layer()` read it BACK OUT to build the
//! deployment's table — a rule spelled once, transported through a
//! backend struct, and re-read as data. [`project::deployment`] expands
//! `is_sliding` directly, and it is the only expansion: the text states
//! no window, because every gpt-oss layer takes the sink spelling and
//! the driver reads the window per layer from the deployment.
//!
//! Then the encodings. `openai/gpt-oss-20b` publishes MXFP4 expert
//! blocks; the dequantized release publishes bf16 banks under different
//! NAMES. They are one model, so they are one row: see
//! [`project::manifest`] for how a manifest pins the mixture's geometry
//! without naming either spelling.

// `Arc` reaches this module only through `Variant::chat`, so the
// import carries that method's gate. It used to ride along with
// `OnceLock`, which `rows()` needed unconditionally until
// `rows_of!` absorbed it.
#[cfg(feature = "chat")]
use std::sync::Arc;

use crate::catalog::{Deployed, LoadShape, Variant};
use crate::manifest::Manifest;

use spec::GptOssFacts;

#[cfg(feature = "chat")]
pub mod chat;
#[cfg(feature = "contract")]
pub mod contract;

/// gpt-oss's forward pass.
///
/// Written in `model-compiler`'s tracing eDSL: ordinary Rust that runs at
/// model-load time with the checkpoint's facts in hand and records what one
/// pass computes. The traced form is what a driver executes.
pub mod forward;

/// What a gpt-oss checkpoint IS — ungated, because a row is written in
/// these words and a row must answer under every aspect.
pub mod spec;

/// What those numbers imply: a manifest, a deployment, a trace.
pub mod project;

/// One gpt-oss checkpoint.
///
/// A newtype over [`GptOssFacts`] for [`crate::qwen_3::Qwen3`]'s reason —
/// the shape cannot say which template speaks for it — plus the two
/// numbers the shape deliberately does not hold, because a
/// [`crate::deployment::Deployment`] carries them PER LAYER and the
/// shape is per model.
pub struct GptOss {
    /// The stable name a boundary carries.
    pub id: &'static str,
    /// The numbers.
    pub shape: GptOssFacts,
    /// Rope's base frequency (`rope_theta`), 150 000 across the
    /// generation. The config also asks for YaRN scaling over a 4096
    /// original context; the driver passes this base UNSCALED, which
    /// [`GptOssFacts::rope_yarn_original`] records as the fact it is
    /// rather than the fact it should be.
    pub rope_theta: f32,
    /// RMSNorm epsilon (`rms_norm_eps`): `1e-5`, and NOT the `1e-6` most
    /// of the llama lineage ships. No tensor extent carries it, so a row
    /// is the only place it can live — the launch path used to read it
    /// off a resident `HfConfig`.
    pub norm_eps: f32,
    /// The sliding window's width, applied on the layers
    /// [`GptOssFacts::is_sliding`] names. `128` on every published
    /// gpt-oss, which is small enough that getting it wrong is a quality
    /// regression rather than a crash — the kind of defect a stated
    /// number prevents and a derived one hides.
    pub window: i32,
}

impl GptOss {
    /// Routed experts. gpt-oss has no dense arm at all, so this is never
    /// zero — which is the claim [`Self::load_shape`] and
    /// [`project::manifest`]'s router row are both projections of.
    #[must_use]
    pub const fn experts(&self) -> u32 {
        self.shape.experts
    }
}

/// The family label a GUEST PROGRAM matches on.
///
/// `gptoss` and not `gpt_oss`, which is the string the checkpoints put
/// in `model_type` and the one the OLD driver advertised — it copied
/// `hf.model_type` straight into its capabilities. The label has to be
/// the other spelling now because the boundary CHECKS it:
/// `embedded_driver::read_hf_config_defaults` lowercases
/// `architectures[0]` — `GptOssForCausalLM` — strips the task suffix and
/// refuses any family [`crate::catalog::arches`] does not list. No
/// underscore survives that reduction, so a row advertising `gpt_oss`
/// would have the boundary reject both published gpt-oss releases.
const ARCH: &str = "gptoss";

/// The published context ceiling, shared by both rows.
///
/// One constant because `openai/gpt-oss-20b` and `openai/gpt-oss-120b`
/// state the same `max_position_embeddings: 131072`, and three corpus
/// files agree — the 20b's own config and both `tiny-random--gpt-oss-*`
/// dumps. The two rows differ in layer count and expert count and in
/// nothing else; this is one of the numbers they share.
///
/// It is NOT the 4096 that rides in the same configs under
/// `rope_scaling.original_max_position_embeddings`. That number is
/// about the rotary ladder — the pretraining length the YaRN stretch is
/// measured from — and what a row records of it is the BINDING it
/// forces, [`GptOssFacts::rope_yarn_original`], rather than the length
/// itself. This constant is the different question: what the release
/// advertises, which nothing in a fire reads.
const MAX_MODEL_LEN: u32 = 131_072;

/// gpt-oss's YaRN rescaling, stated once because both rows share it.
///
/// Four of the five numbers come from the corpus's
/// `openai--gpt-oss-20b.json` — `factor: 32.0`, `beta_fast: 32.0`,
/// `beta_slow: 1.0`, `original_max_position_embeddings: 4096` — and both
/// `tiny-random--gpt-oss-*` fixtures state the same four, which is why
/// the 120B row inherits them.
///
/// The fifth, `attention_factor`, the config does NOT state. It is not
/// invented here: HF's default is `0.1 * ln(factor) + 1`, and the value
/// below is that formula at `factor = 32.0`. The formula itself is
/// checked against a checkpoint that DOES write its answer down —
/// OLMo 3 — rather than trusted, which is what
/// `an_omitted_attention_factor_is_the_formula_olmo_3_states` is for.
///
/// `original_max_position` is 4096 and [`MAX_MODEL_LEN`] is 131_072:
/// 4096 * 32 is exactly the extended context, which is the arithmetic
/// this pair of numbers is supposed to satisfy and a cheap check that
/// neither was transcribed from the wrong line.
const ROPE_SCALING: crate::deployment::RopeScaling = crate::deployment::RopeScaling::Yarn {
    factor: 32.0,
    beta_fast: 32.0,
    beta_slow: 1.0,
    attention_factor: 1.346_573_6,
    original_max_position: 4_096,
    // STATED by every gpt-oss config, and the only family in the corpus
    // that writes it. HF would default it to `true`.
    truncate: false,
};

/// The generation's rows.
///
/// `const`, so identity is in `.rodata` and the three questions have one
/// answer each. Both rows are the SAME shape at two sizes: gpt-oss-120b
/// is the 20b with half again the layers and four times the experts, and
/// every other number — the head counts, the 64-wide heads, the 2880
/// expert width, the 201 088-entry vocabulary, the sinks, the biases —
/// is identical. Stating them twice rather than deriving the big one
/// from the small one is deliberate: a row is a MEASUREMENT, and the
/// next release that breaks the pattern must break a number, not a rule.
pub const VARIANTS: &[GptOss] = &[
    // openai/gpt-oss-20b. Equal to `GptOssFacts::gpt_oss_20b()`, which a
    // test below holds to.
    GptOss {
        id: "gpt-oss-20b",
        shape: GptOssFacts {
            hidden: 2880,
            layers: 24,
            q_heads: 64,
            kv_heads: 8,
            head_dim: 64,
            intermediate: 2880,
            experts: 32,
            top_k: 4,
            vocab: 201_088,
            tied_embeddings: false,
            swiglu_limit: 7.0,
        },
        rope_theta: 150_000.0,
        norm_eps: 1e-5,
        window: 128,
    },
    // openai/gpt-oss-120b. 36 layers and 128 experts; the 20b's shape
    // otherwise, down to `intermediate: 2880` — the 120b is bigger by
    // EXPERT COUNT, not by expert width, which is the whole design of
    // the release and the reason its active parameter count barely
    // moves.
    GptOss {
        id: "gpt-oss-120b",
        shape: GptOssFacts {
            hidden: 2880,
            layers: 36,
            q_heads: 64,
            kv_heads: 8,
            head_dim: 64,
            intermediate: 2880,
            experts: 128,
            top_k: 4,
            vocab: 201_088,
            tied_embeddings: false,
            swiglu_limit: 7.0,
        },
        rope_theta: 150_000.0,
        norm_eps: 1e-5,
        window: 128,
    },
];

crate::rows_of!(GptOss);

impl Variant for GptOss {
    fn id(&self) -> &'static str {
        self.id
    }

    fn manifest(&self) -> Manifest {
        project::manifest(&self.shape)
    }

    /// The UNPADDED head dim — 64, which is also what the kernel is
    /// instantiated at — and a nonzero expert count, because gpt-oss has
    /// no dense arm. `mamba_groups` and `kv_shared_layers` are zero and
    /// [`LoadShape::mixture`] states them rather than defaulting them.
    fn load_shape(&self) -> LoadShape {
        LoadShape::mixture(
            self.shape.layers,
            self.shape.head_dim,
            self.experts(),
            self.shape.tied_embeddings,
        )
    }

    /// Servable. The sinks change the SOFTMAX and not the store, so what
    /// this needs is the ordinary paged k/v every build has — worth
    /// stating, because the family's one historical failure looked like
    /// a missing pool and was a missing forward arm.
    fn deployment(
        &self,
        load: Deployed<'_>,
    ) -> Result<crate::deployment::Deployment, crate::deployment::Refusal> {
        let _ = load;
        let mut deployment =
            project::deployment(&self.shape, self.rope_theta, self.norm_eps, self.window);
        deployment.rope_scaling = Some(ROPE_SCALING);
        deployment.advertised = crate::deployment::Advertised {
            arch: ARCH,
            max_model_len: MAX_MODEL_LEN,
            // No tower. gpt-oss is text-only in both releases — there is
            // no vision or audio block in either package — so
            // `Deployment::towers` stays empty and the driver's encode
            // entry has nothing here to serve.
            media_encode: false,
        };
        Ok(deployment)
    }

    /// `author_gpt_oss`, stated. It was reached through
    /// `HF_ROWS["gpt_oss"]` before, which is the same function behind a
    /// string — and the string was also what a SECOND table keyed its
    /// facts on, with nothing holding the two answers together.
    ///
    /// The MLX author (`contract::author_gpt_oss_mlx`) is a different
    /// BACKEND's reading of the same checkpoint, not a different model,
    /// so it is not a row: `crate::shared::policy` is where a build picks a
    /// loader -- which is why this ASKS. Stating the HF author alone
    /// read as though the sentence above had been implemented, and
    /// handed Metal a contract in the checkpoint's own names.
    #[cfg(feature = "contract")]
    fn author(
        &self,
        builder: &mut crate::shared::builder::Builder<'_>,
    ) -> Result<(), model_loader::error::Error> {
        match builder.naming() {
            crate::shared::policy::Naming::Hf => contract::author_gpt_oss(builder),
            crate::shared::policy::Naming::Mlx => contract::author_gpt_oss_mlx(builder),
        }
    }

    fn trace(
        &self,
        class: model_compiler::trace::FireClass,
        load: Deployed<'_>,
    ) -> Result<model_compiler::trace::ForwardPlan, crate::deployment::Refusal> {
        // METAL, through the SHARED text. This arm returned
        // `project::NO_METAL` for as long as the projection below did not
        // exist, and that refusal's own doc had narrowed the gap to one
        // thing: "no `metal_shape` / `metal_facts` pair, so `Variant::trace`
        // has no Metal text to hand back". These are that pair.
        //
        // What it named as already present was present -- the sinks reach the
        // shader, the clamped GLU is a symbol, `routed_qmv` picks
        // `mxfp4_qmv_routed_bias` off `moe_repr` -- but the doc's list was
        // not complete. gpt-oss also biases its attention landing and its
        // ROUTER, and the shared Metal text stated neither, because no Metal
        // kernel added a bias at all until `norm/add_bias.metal` and no
        // binder resolved its width until `dispatch::derived`. Those are the
        // three that landed together.
        if let crate::catalog::Backend::Metal(bind) = load.backend {
            let shape = project::metal_shape(&self.shape);
            let facts = project::metal_facts(&self.shape, bind);
            // The KERNEL SET's refusals, taken from the same function every
            // other llama-like row asks, and not restated here. Two of the
            // three can fire for a gpt-oss: a sharded load, and a routed leg
            // at an affine encoding no `affine_qmv_routed` is instantiated
            // for.
            //
            // Skipping this was not a missing nicety. `the_text_this_driver
            // _runs_is_the_text_the_row_states` walks every row at g128/b8 as
            // well as g64/b4, and without the gate the row named
            // `affine_qmv_routed_bfloat16_gs_128_b_8` -- a symbol no `kernel!`
            // declares -- and the trace panicked rather than refusing. The
            // head-dim arm cannot fire here (gpt-oss is 64) and stating it
            // anyway is the point of sharing the function.
            crate::shared::llama_like::project::metal_kernel_refusal(&shape, &facts, load, bind)?;
            return Ok(crate::shared::llama_like::forward::llama_like_metal(
                &shape, &facts, class,
            ));
        }
        Ok(project::trace(&self.shape, class, load))
    }

    /// gpt-oss's OWN template and not ChatML: the harmony format opens
    /// its turns with `<|start|>`, carries a reasoning channel, and ends
    /// on `<|return|>`. `instruct::create` reached it by matching
    /// `"gpt_oss"`, one arm above a `_ =>` that would have handed a
    /// harmony checkpoint the ChatML template — a mismatch that produces
    /// text rather than an error.
    #[cfg(feature = "chat")]
    fn chat(&self, tokenizer: Arc<tokenizer::Tokenizer>) -> Arc<dyn crate::instruct::Instruct> {
        Arc::new(chat::GptOssInstruct::new(tokenizer))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::Presence;

    fn row(id: &str) -> &'static GptOss {
        VARIANTS.iter().find(|v| v.id == id).expect("row present")
    }

    /// The YaRN numbers are the corpus's, and the fifth is a FORMULA.
    ///
    /// `openai--gpt-oss-20b.json` states `factor`, `beta_fast`,
    /// `beta_slow` and `original_max_position_embeddings`, and both
    /// `tiny-random--gpt-oss-*` fixtures agree — which is why the 120B row
    /// inherits them. It does NOT state `attention_factor`; HF then uses
    /// `0.1 * ln(factor) + 1`, and that formula is checked against a
    /// config that does write its answer down over in `olmo_3`
    /// (`an_omitted_attention_factor_is_the_formula_olmo_3_states`), so
    /// the number below is derived rather than invented.
    #[test]
    fn the_yarn_numbers_are_the_configs_and_the_omitted_one_is_the_formula() {
        let crate::deployment::RopeScaling::Yarn {
            factor,
            beta_fast,
            beta_slow,
            attention_factor,
            original_max_position,
            truncate,
        } = ROPE_SCALING
        else {
            panic!("gpt-oss rescales by YaRN");
        };
        assert_eq!(factor, 32.0);
        assert_eq!(beta_fast, 32.0);
        assert_eq!(beta_slow, 1.0);
        assert_eq!(original_max_position, 4_096);
        // The one number in this block the config writes DOWN rather than
        // omits, and the only family in the corpus that does. HF's default
        // is the opposite, so a row copying the common case would snap the
        // ramp gpt-oss deliberately leaves unsnapped.
        assert!(!truncate, "gpt-oss states `truncate: false`");
        let formula = 0.1 * f64::from(factor).ln() + 1.0;
        assert!(
            (f64::from(attention_factor) - formula).abs() < 1e-6,
            "stated {attention_factor}, formula {formula}"
        );
        // 4096 * 32 is exactly the advertised ceiling. The two numbers come
        // off different lines of the same file, so agreeing here is a cheap
        // check that neither was transcribed from the wrong one.
        assert_eq!(
            original_max_position * factor as u32,
            MAX_MODEL_LEN,
            "the trained context times the extension factor is the ceiling"
        );
    }

    /// Both rows carry it, and a driver that cannot build the table must
    /// refuse rather than serve the ladder unrescaled.
    #[test]
    fn both_rows_carry_the_yarn_rescaling_into_the_deployment() {
        for v in VARIANTS {
            let d = v.deployment(Deployed::single()).expect("gpt-oss deploys");
            assert_eq!(d.rope_scaling, Some(ROPE_SCALING), "{}", v.id);
        }
    }

    /// The fixture and the row are the same measurement of the same
    /// checkpoint, so if they disagree one of them is wrong.
    #[test]
    fn the_20b_row_is_the_committed_fixture() {
        assert_eq!(row("gpt-oss-20b").shape, GptOssFacts::gpt_oss_20b());
    }

    /// The three capability answers come off the ROW.
    ///
    /// They were the last three reads of a resident `HfConfig` inside
    /// `driver-cuda`: `model_type`, `max_position_embeddings` and "does
    /// this ship a tower". Both rows are asserted because both
    /// advertise, and the 120b has no config in the corpus — its ceiling
    /// comes from the published `openai/gpt-oss-120b` config and this is
    /// where that claim is pinned.
    #[test]
    fn the_row_answers_what_the_driver_advertises() {
        for v in VARIANTS {
            let a = v
                .deployment(Deployed::single())
                .expect("gpt-oss deploys")
                .advertised;
            assert_eq!(
                a.arch, "gptoss",
                "{}: the family label a guest program sees",
                v.id
            );
            assert_eq!(
                a.max_model_len, 131_072,
                "{}: both releases state the same ceiling",
                v.id
            );
            assert!(!a.media_encode, "{}: gpt-oss ships no encoder tower", v.id);
        }
    }

    /// The label is the STEM and not the `model_type`, which is the one
    /// place this generation's two spellings can be told apart.
    ///
    /// `model_type` is `gpt_oss` and the old driver advertised exactly
    /// that, having copied `hf.model_type` into its capabilities. The
    /// worker now derives its own label from `architectures[0]` and
    /// refuses what [`crate::catalog::arches`] does not list — so the
    /// underscore spelling would turn a real gpt-oss checkpoint away at
    /// the boundary.
    #[test]
    fn the_label_is_the_architecture_stem_and_not_the_model_type() {
        let stem = "GptOssForCausalLM"
            .to_lowercase()
            .strip_suffix("forcausallm")
            .expect("the suffix the worker strips")
            .to_string();
        assert_eq!(stem, ARCH);
        assert_ne!(
            ARCH, "gpt_oss",
            "the `model_type` spelling never survives the reduction"
        );
        assert!(!ARCH.contains('_'), "no underscore survives it either");
    }

    /// The 120b is the 20b's shape at two different numbers, and this
    /// test names WHICH two — so a future release that also moves the
    /// head count or the expert width fails here rather than silently
    /// serving the wrong geometry under a familiar id.
    #[test]
    fn the_120b_differs_from_the_20b_in_layers_and_experts_only() {
        let small = &row("gpt-oss-20b").shape;
        let big = &row("gpt-oss-120b").shape;
        assert_eq!(big.layers, 36);
        assert_eq!(big.experts, 128);
        assert_eq!(
            *big,
            GptOssFacts {
                layers: big.layers,
                experts: big.experts,
                ..small.clone()
            },
            "nothing else moved between the two sizes",
        );
    }

    /// Both rows are mixtures, and the claim is made in every place that
    /// projects it: the load shape's expert count, the manifest's router
    /// extents, and the deployment's expert width.
    #[test]
    fn every_row_routes_and_says_so_everywhere() {
        for v in VARIANTS {
            assert_ne!(v.experts(), 0, "{}", v.id);
            assert_eq!(v.load_shape().n_experts, v.experts(), "{}", v.id);
            assert_eq!(v.load_shape().layers, v.shape.layers, "{}", v.id);
            assert_eq!(v.load_shape().head_dim, 64, "{}: unpadded", v.id);
            assert_eq!(v.load_shape().mamba_groups, 0, "{}: no mamba mixer", v.id);
            assert_eq!(v.load_shape().kv_shared_layers, 0, "{}", v.id);
            assert!(!v.load_shape().tied_embeddings, "{}: a separate head", v.id);

            let m = v.manifest();
            let router = m
                .tensors
                .iter()
                .find(|t| t.name.ends_with("mlp.router"))
                .expect("a mixture states its router");
            assert_eq!(router.presence, Presence::Required);
            assert_eq!(
                router.extents,
                vec![u64::from(v.experts()), u64::from(v.shape.hidden)],
                "{}: the router IS the claim that this routes",
                v.id,
            );

            let d = v.deployment(Deployed::single()).expect("servable");
            assert_eq!(d.layers, v.shape.layers, "{}", v.id);
            assert_eq!(d.shape.moe_intermediate, v.shape.intermediate, "{}", v.id);
            assert_eq!(d.shape.intermediate, 0, "{}: no dense block", v.id);
            assert_eq!(d.norm_eps, v.norm_eps, "{}", v.id);
            assert_eq!(d.attention.len() as u32, v.shape.layers, "{}", v.id);
        }
    }

    /// THE SINKS, at the row level: every gpt-oss row ships one logit
    /// per query head, and this is the only generation in the catalog
    /// that does. A row that stopped stating them would still load and
    /// would attend with a softmax missing its extra denominator term,
    /// which is a quality regression and not a crash — so it is asserted
    /// rather than assumed.
    #[test]
    fn every_row_ships_attention_sinks() {
        for v in VARIANTS {
            let sinks = v
                .manifest()
                .tensors
                .into_iter()
                .find(|t| t.name.ends_with("self_attn.sinks"))
                .expect("stated");
            assert_eq!(sinks.presence, Presence::Required, "{}", v.id);
            assert_eq!(sinks.extents, vec![u64::from(v.shape.q_heads)], "{}", v.id);
        }
    }

    /// THE CLAIM: a quantization is not an identity.
    ///
    /// `tiny-random--gpt-oss-bf16.json` and
    /// `tiny-random--gpt-oss-mxfp4.json` differ by exactly one key — an
    /// added `quantization_config` — and the released 20b ships both
    /// ways. One row serves both, because the manifest compares LOGICAL
    /// extents and names no expert weight, and because the id vocabulary
    /// forbids an encoding word outright (`catalog::tests::
    /// no_id_names_an_encoding`). Where the packing is decided is
    /// `crate::shared::policy`.
    #[test]
    fn no_row_records_the_expert_encoding() {
        for v in VARIANTS {
            for word in ["mxfp4", "bf16", "fp8", "int4"] {
                assert!(!v.id.contains(word), "{} names an encoding", v.id);
            }
            let named: Vec<&str> = v
                .manifest()
                .tensors
                .iter()
                .filter(|t| t.name.contains("experts."))
                .map(|t| t.name.as_str())
                .map(str::to_owned)
                .map(|s| Box::leak(s.into_boxed_str()) as &str)
                .collect();
            assert!(
                named.iter().all(|n| n.ends_with("_bias")),
                "{}: only the bias survives every packing, but the manifest names {named:?}",
                v.id,
            );
        }
    }

    /// The two rows have distinct ids and distinct manifests — the
    /// second half matters, because two rows a manifest cannot tell
    /// apart are `Unmatched::Ambiguous` for every checkpoint they both
    /// fit.
    #[test]
    fn the_rows_are_distinguishable_by_their_tensors_alone() {
        let a = row("gpt-oss-20b").manifest();
        let b = row("gpt-oss-120b").manifest();
        assert_ne!(a.layers, b.layers, "24 against 36");
        let extents = |m: &Manifest| {
            m.tensors
                .iter()
                .find(|t| t.name.ends_with("mlp.router"))
                .expect("stated")
                .extents
                .clone()
        };
        assert_ne!(extents(&a), extents(&b), "32 experts against 128");
    }

    /// The row and `LlamaLikeFacts::gpt_oss_20b()` are two vocabularies
    /// for one checkpoint, and they agree.
    ///
    /// gpt-oss is NOT a llama-like row — the sinks, the clamped SwiGLU
    /// and the MXFP4 triplets are why it has its own text — but the
    /// fixture exists, other tests read it, and a number that drifted
    /// between the two would be exactly the two-readings-of-one-document
    /// defect this catalog was built to remove.
    #[test]
    fn the_llama_like_fixture_measures_the_same_checkpoint() {
        let f = &row("gpt-oss-20b").shape;
        let l = crate::shared::llama_like::spec::LlamaLikeFacts::gpt_oss_20b();
        assert_eq!(l.hidden, f.hidden);
        assert_eq!(l.layers, f.layers);
        assert_eq!(l.q_heads, f.q_heads);
        assert_eq!(l.kv_heads, f.kv_heads);
        assert_eq!(l.head_dim, f.head_dim);
        assert_eq!(l.vocab, f.vocab);
        assert_eq!(l.n_experts, f.experts);
        assert_eq!(l.experts_per_token, f.top_k);
        assert_eq!(l.moe_intermediate, f.intermediate, "the per-expert width");
        assert_eq!(l.intermediate, 0, "neither vocabulary claims a dense block");
        assert!(
            l.qkv_bias,
            "gpt-oss biases q/k/v, and the epilogue folds it"
        );
        assert_eq!(l.tied_embeddings, f.tied_embeddings);
    }

    /// The window is stated per row and expanded by the row's own rule,
    /// so a launch and a trace cannot hold different schedules.
    #[test]
    fn every_row_alternates_its_window_from_layer_zero() {
        for v in VARIANTS {
            assert_eq!(v.window, 128);
            let d = v.deployment(Deployed::single()).expect("servable");
            for (l, a) in d.attention.iter().enumerate() {
                let expected = if v.shape.is_sliding(l as u32) {
                    v.window
                } else {
                    -1
                };
                assert_eq!(a.window, expected, "{}: layer {l}", v.id);
            }
            assert_eq!(d.attention[0].window, 128, "{}: starts sliding", v.id);
            assert_eq!(d.attention[1].window, -1, "{}: then full", v.id);
        }
    }

    /// Every row answers every question, under every aspect this build
    /// compiled — the property that replaced three tables with different
    /// key sets, where a model present in one and missing from another
    /// was a `None` nobody noticed.
    #[test]
    fn every_row_answers_every_question() {
        assert_eq!(rows().len(), VARIANTS.len());
        for v in rows() {
            assert!(!v.id().is_empty());
            assert!(!v.manifest().tensors.is_empty(), "{}", v.id());
            assert_ne!(v.load_shape().layers, 0, "{}", v.id());
            assert!(v.deployment(Deployed::single()).is_ok(), "{}", v.id());
            for class in [
                model_compiler::trace::FireClass::Decode,
                model_compiler::trace::FireClass::Prefill,
            ] {
                let plan = v.trace(class, Deployed::single()).expect("traceable");
                assert!(!plan.ops.is_empty(), "{}: {class:?}", v.id());
            }
        }
    }

    /// The harmony template, reached from the row rather than from
    /// `instruct::create`'s string match — and it is a DIFFERENT
    /// template from every other row's, which is what the `_ =>` arm
    /// could not have told anyone.
    ///
    /// The fixture names harmony's three stop strings, because `seal()`
    /// resolves them THROUGH THE TOKENIZER and a vocabulary that does not
    /// contain them makes the answer empty for a reason that has nothing
    /// to do with the row. `t0..t63` alone tested that a missing token is
    /// missing.
    #[cfg(feature = "chat")]
    #[test]
    fn every_row_speaks_harmony() {
        let mut vocab: Vec<String> = ["<|endoftext|>", "<|return|>", "<|call|>"]
            .iter()
            .map(|s| (*s).to_string())
            .collect();
        vocab.extend((0..64).map(|i| format!("t{i}")));
        let tokenizer = Arc::new(tokenizer::Tokenizer::from_vocab(&vocab));
        for v in VARIANTS {
            let chat = v.chat(tokenizer.clone());
            assert!(!chat.user("Hi").is_empty(), "{}", v.id);
            assert!(!chat.seal().is_empty(), "{}", v.id);
        }
    }

    /// A METAL load is traced as THIS ROW and not as a llama.
    ///
    /// The guard that replaces `driver-metal`'s `LLAMA_LIKE` table. That
    /// table answered "does this build serve you" from an architecture
    /// STRING reduced by `canonical()`, in a driver, before any text was
    /// traced — so it could say yes to a row this build could not resolve
    /// and no to one whose text it models. The row answers now.
    ///
    /// # This used to assert a REFUSAL, and the refusal was the whole risk
    ///
    /// It pinned `project::NO_METAL` verbatim, on the ground that a row
    /// which cannot be served must say so by name rather than be quietly
    /// handed the shared llama text. That was right while it was true. It
    /// stopped being true when `metal_shape`/`metal_facts` were written,
    /// and the danger the old test guarded against is exactly what this one
    /// now has to rule out: the shared text is `llama_like_metal`, and a
    /// projection that got a field wrong would trace a *llama* under this
    /// row's id and nothing would say so.
    ///
    /// So the assertions are the four things that make this text gpt-oss's
    /// and not any other row's — the sinks, the clamp, the routed MXFP4
    /// bank, and the three biases. Each is a tensor or a symbol no llama
    /// names, and each was separately missing at some point in getting
    /// here.
    #[test]
    fn a_metal_load_is_traced_as_this_row_and_not_as_a_llama() {
        use crate::catalog::{Backend, Deployed, MetalBinding};
        use model_compiler::trace::FireClass;

        // `moe_mxfp4` TRUE, which is what `binding::observed` answers for
        // every gpt-oss published: the checkpoint lists its dense tensors as
        // affine/64/4 and leaves the expert banks to the top-level mxfp4/32.
        let bind = MetalBinding {
            quant_group: 64,
            quant_bits: 4,
            moe_mxfp4: true,
            fuse_residual_gemv: true,
            paged_multi_batch: true,
            qmm_multi_batch: true,
            add_bias: true,
        };
        assert!(!VARIANTS.is_empty());
        for v in VARIANTS {
            for class in [FireClass::Prefill, FireClass::Decode] {
                let plan = v
                    .trace(class, Deployed::metal(&bind))
                    .expect("this build has a Metal text for this generation");
                let text = format!("{plan:?}");
                for (want, why) in [
                    (
                        "attn_sinks",
                        "the learned per-head logit every gpt-oss layer carries",
                    ),
                    (
                        "swiglu",
                        "the CLAMPED gate, which is a different kernel and not a scalar",
                    ),
                    (
                        "mxfp4_qmv_routed_bias",
                        "the expert bank's own encoding -- an affine reading of it \
                         produced 909,207 NaNs",
                    ),
                    ("o_bias", "the attention landing's bias"),
                    ("router_bias", "the ROUTER's bias, which moves a ranking"),
                    ("q_bias", "the projection biases"),
                ] {
                    assert!(
                        text.contains(want),
                        "`{}`'s {class:?} Metal text names no `{want}` -- {why}",
                        v.id
                    );
                }
                assert!(
                    plan.family.contains("metal"),
                    "`{}` was served `{}`, which is not a Metal text",
                    v.id,
                    plan.family
                );
            }
        }
        // And the text is about the BACKEND and nothing else: the same rows
        // keep answering a CUDA load exactly as they did.
        for v in VARIANTS {
            assert!(
                v.trace(FireClass::Decode, Deployed::single()).is_ok(),
                "`{}` stopped serving CUDA",
                v.id
            );
        }
        // A `Backend::Cuda` is what `Deployed::single()` states, so the
        // arm above is reached by every existing caller unchanged.
        assert!(matches!(Deployed::single().backend, Backend::Cuda));
    }
}

//! Gemma 4 — the generation every default in the old family trait was
//! written to hide.
//!
//! Thirteen methods, twelve of them with default bodies, and four of the
//! twelve had gemma-4 in their documentation:
//! `pins_attention_values()` returned `true` with the doc *"Only gemma-4
//! does"*, `planless_prefill()` returned `false` because every family
//! but this one plans ahead, `sm_scale()` divided by `sqrt(head_dim)`
//! because every family but this one wants that, and
//! `decode_plan_head_dims()` existed AT ALL because this generation's
//! two layer kinds disagree about how wide a head is. A default body is
//! a claim about every row not yet written, and here the claim was
//! false four times over — each time about the same row.
//!
//! So these rows answer every question, and [`project`] is where the
//! answers are spelled. What is left here is what a set of widths cannot
//! say: which window the config states, whether the checkpoint ships a
//! routed bank, what a guest program should be told this model is, and
//! which of the two authors writes the load.

#[cfg(feature = "chat")]
pub mod chat;
#[cfg(feature = "contract")]
pub mod contract;

/// The three projections a row of this generation makes: its tensor
/// manifest, its `Deployment`, and its traced text.
pub mod project;

/// The numbers a gemma-4 IS, ungated — see the module's own doc for why
/// it is not inside [`forward`].
pub mod spec;

/// Gemma 4's forward pass.
///
/// Written in `model-compiler`'s tracing eDSL: ordinary Rust that runs at
/// model-load time with the checkpoint's facts in hand and records what one
/// pass computes. The traced form is what a driver executes.
pub mod forward;

// `Arc` reaches this module only through `Variant::chat`, so the
// import carries that method's gate. It used to ride along with
// `OnceLock`, which `rows()` needed unconditionally until
// `rows_of!` absorbed it.
#[cfg(feature = "chat")]
use std::sync::Arc;

use crate::catalog::{Deployed, LoadShape, Variant};
use crate::deployment::{Advertised, AudioTower, Deployment, Refusal, Towers, VisionTower};
use crate::manifest::Manifest;

use self::spec::{Gemma4Facts, Gemma4Mixture};

/// RMSNorm epsilon, shared by the whole generation.
///
/// Stated once rather than per row because all four published gemma-4
/// configs carry the same `1e-6` — a generation-level constant is an
/// honest way to say that, where four copies of a literal invite one of
/// them to drift. It reaches a driver through `Deployment::norm_eps`,
/// which is where the launch path used to read it off a resident
/// `HfConfig`.
const NORM_EPS: f32 = 1e-6;

/// The family label a GUEST PROGRAM matches on.
///
/// `gemma4` and not `Gemma4ForConditionalGeneration`, which is what the
/// old derivation produced by stripping suffixes off `architectures[0]`
/// — and `VisionArch::from_arch_name` looks for exactly this substring,
/// so the string a program sees and the tower a driver selects are one
/// decision now instead of two that happened to agree.
const ARCH: &str = "gemma4";

/// The audio front-end every E-series package ships.
///
/// Identical across `E4B`, `E4B-it` and (by the same reading) `E2B`:
/// twelve conformer blocks over 128 mel bins, chunked 12 frames at a
/// time with 13 chunks of history and none of lookahead. The A4B ships
/// NO audio tower at all, which is why this is a row's field and not the
/// generation's.
const E_SERIES_AUDIO: AudioTower = AudioTower {
    layers: 12,
    hidden: 1024,
    heads: 8,
    conv_kernel: 5,
    feature_size: 128,
    // `subsampling_conv_channels: [128, 32]` — the second convolution is
    // NARROWER than the first, which is why they are two numbers and not
    // a width and a multiplier.
    subsample_channels_0: 128,
    subsample_channels_1: 32,
    output_dims: 1536,
    chunk_size: 12,
    context_left: 13,
    context_right: 0,
    logit_cap: 50.0,
    residual_weight: 0.5,
    norm_eps: NORM_EPS,
};

/// The E-series vision front-end.
const E_SERIES_VISION: VisionTower = VisionTower {
    layers: 16,
    hidden: 768,
    heads: 12,
    intermediate: 3072,
    pooling_kernel: 3,
    norm_eps: NORM_EPS,
    // A rotary base of ONE HUNDRED, not the decoder's 10 000 or
    // 1 000 000. A tower that inherited the decoder's base would rotate
    // a 280-patch grid as if it were a 131 072-token context, and the
    // number is small enough to look like a typo — it is not.
    rope_theta: 100.0,
};

/// The A4B's vision front-end, which is a DIFFERENT tower.
///
/// 27 blocks of 1152 against the E-series' 16 of 768. Two rows of one
/// generation, two encoders, no shared default: this is the pair that
/// makes [`Deployment::towers`] a row's answer.
const A4B_VISION: VisionTower = VisionTower {
    layers: 27,
    hidden: 1152,
    heads: 16,
    intermediate: 4304,
    pooling_kernel: 3,
    norm_eps: NORM_EPS,
    rope_theta: 100.0,
};

/// Both E-series encoders, named once for the two rows that carry them.
const E_SERIES_TOWERS: Towers = Towers {
    audio: Some(E_SERIES_AUDIO),
    vision: Some(E_SERIES_VISION),
};

/// One Gemma 4 checkpoint.
///
/// The shape is the stack; these five fields are what the stack cannot
/// say. `sliding_window` and `max_model_len` are read off the row's own
/// `config.json`, `mixture` is present only on the A4B, `k_eq_v` is the
/// A4B's V-from-K attention mode, and `towers` says which encoders the
/// package ships beside the decoder.
pub struct Gemma4 {
    /// The stable name a boundary carries.
    pub id: &'static str,
    /// The numbers the stack is.
    pub shape: Gemma4Facts,
    /// The routed bank, on the one row that has one.
    pub mixture: Option<Gemma4Mixture>,
    /// `sliding_window` — what a SLIDING layer attends. The full layers
    /// ignore it, which is why it is one number and not a table.
    pub sliding_window: i32,
    /// `attention_k_eq_v`: the A4B reads V from K rather than
    /// projecting it. A different attention, not a different width — and
    /// this build has no text for it.
    pub k_eq_v: bool,
    /// `max_position_embeddings`, advertised and never fired on.
    pub max_model_len: u32,
    /// The encoders this package ships beside the decoder.
    ///
    /// Gemma-4 is the ONLY family the encode pipeline can scope a load
    /// to — see [`contract`] — and the driver's capability read this as
    /// a hardwired `false` while four GPU paths served the towers. It is
    /// derived from this field now rather than stated a second time, so
    /// a row cannot advertise an encoder it does not carry.
    pub towers: Towers,
}

/// The generation's rows.
///
/// Three, for four corpus configs and a fifth checkpoint that is not
/// here:
///
/// * `google/gemma-4-E4B` and `google/gemma-4-E4B-it` are ONE row. Their
///   `text_config` blocks are byte-identical — an instruction tune
///   changes weights and a chat template, not a stack — and a row per
///   tune would be two rows one checkpoint matches, which the catalog
///   reports as ambiguous rather than resolving by order.
/// * `google/gemma-4-E4B-it-assistant` has NO row. It states four layers
///   and `num_kv_shared_layers: 4`, so every layer in it attends KV that
///   an earlier layer wrote — and there is no earlier layer. Its pages
///   come from the E4B backbone it rides beside, which is a fact one
///   `Deployment` has no way to hold. A row that claimed it would have
///   to land every layer's `kv_source` on itself and serve a stack that
///   attends its own empty cache.
pub const VARIANTS: &[Gemma4] = &[
    // google/gemma-4-E2B-it. NOT in the config corpus: the shape is the
    // live-anchored fixture (the driver's parity gate ran this
    // checkpoint), and the three fields below it are the E-series
    // siblings' — E4B states a 512-token window, a 131 072 ceiling and
    // both towers, and every E-series package in the corpus agrees. If a
    // real E2B disagrees about any of the three, this row is where to
    // look; none of them changes what the stack IS.
    Gemma4 {
        id: "gemma-4-e2b",
        shape: Gemma4Facts::gemma_4_e2b(),
        mixture: None,
        sliding_window: 512,
        k_eq_v: false,
        max_model_len: 131_072,
        towers: E_SERIES_TOWERS,
    },
    // google/gemma-4-E4B and google/gemma-4-E4B-it.
    Gemma4 {
        id: "gemma-4-e4b",
        shape: Gemma4Facts::gemma_4_e4b(),
        mixture: None,
        sliding_window: 512,
        k_eq_v: false,
        max_model_len: 131_072,
        towers: E_SERIES_TOWERS,
    },
    // google/gemma-4-31B and google/gemma-4-31B-it — the DENSE 31B, and
    // the row this table did not have. `attention_k_eq_v` is true here
    // and does NOT refuse: this build reads V out of the K projection,
    // and the 31b reproduces MLX's logits exactly doing it. What the
    // 26b still lacks is the routed bank, which is now the only thing
    // `untraced` names.
    Gemma4 {
        id: "gemma-4-31b",
        shape: Gemma4Facts::gemma_4_31b(),
        mixture: None,
        sliding_window: 1024,
        k_eq_v: true,
        max_model_len: 262_144,
        // VISION ONLY, and the same tower the 26b ships — 27 layers of
        // 1152 with a 3-wide pooling kernel, read from this row's own
        // `vision_config`. `audio_config` is `null`.
        towers: Towers {
            audio: None,
            vision: Some(A4B_VISION),
        },
    },
    // google/gemma-4-26B-A4B-it — the row that LOADS and does not SERVE.
    // Its contract authors, its manifest identifies it, and its
    // deployment refuses: `attention_k_eq_v` and a 128-expert routed
    // bank are two legs this build has no traced text for. Stating the
    // row anyway is the point — a checkpoint this build can identify and
    // cannot fire says so at the door, where the old arrangement loaded
    // it happily and died at its first fire.
    Gemma4 {
        id: "gemma-4-26b-a4b",
        shape: Gemma4Facts::gemma_4_26b_a4b(),
        mixture: Some(Gemma4Mixture::gemma_4_26b_a4b()),
        sliding_window: 1024,
        k_eq_v: true,
        max_model_len: 262_144,
        // VISION ONLY. `audio_config` is `null` in this checkpoint's
        // `config.json` where the E-series states a twelve-block
        // conformer, and a row that copied its sibling's towers would
        // advertise an audio encoder whose weights are not in the
        // package.
        towers: Towers {
            audio: None,
            vision: Some(A4B_VISION),
        },
    },
];

crate::rows_of!(Gemma4);

impl Gemma4 {
    /// The four numbers this row states that its shape cannot.
    ///
    /// Both projections take this — `deployment` and `metal_facts` used
    /// to take the same four as loose arguments each, and a row read
    /// twice is a row that can be read differently. Deploying at one
    /// window and tracing at another compiles.
    fn row(&self) -> project::RowScalars {
        project::RowScalars {
            mixture: self.mixture,
            sliding_window: self.sliding_window,
            norm_eps: NORM_EPS,
            k_eq_v: self.k_eq_v,
        }
    }

    /// Why this build cannot fire this row, or `None` when it can.
    ///
    /// ONE predicate for two questions, because a row that cannot be
    /// deployed cannot be traced either and two spellings of that would
    /// be two chances to disagree.
    ///
    /// It USED to be the old derivation's whole test —
    /// `gemma4_attention_k_eq_v || gemma4_enable_moe` — and the first
    /// half stopped being true OF METAL. `llama_like_metal` does have
    /// attention text that reads V out of the K projection;
    /// gemma-4-31b runs it and reproduces MLX's logits exactly. A
    /// refusal kept past the day its reason expired reads exactly like
    /// one that was never wrong, and it deleted the only gemma-4 in the
    /// corpus with real weights to check against.
    ///
    /// The first half is still true of CUDA, and lifting it from here
    /// lifted it from both — `Gemma4LayerW` declares a `v_proj` and
    /// `layer()` matmuls it with nothing to branch on. That refusal now
    /// lives in [`Variant::trace`]'s CUDA arm, where it can name one
    /// backend, which is why this predicate is no longer "one predicate
    /// for two questions" about `k_eq_v`: deployment is backend-blind
    /// and Metal deploys these rows. The routed bank is unwritten on
    /// both, and that is what this still says.
    /// MEASURED against `mlx-community/gemma-4-26b-a4b-it-4bit`, and the
    /// mixture rows serve now. The refusal that stood here is gone, and
    /// every sentence it carried was wrong about something.
    ///
    /// It said "no routed-expert TEXT", and the text was written.
    /// It said the contract publishes none of `experts.switch_glu.*`, read
    /// off a run reporting **90 unpublished names** -- thirty layers times
    /// `router`, `router.scales`, `router.zeros`, which is ONE role whose
    /// module gemma-4 spells `router.proj` where gpt-oss spells
    /// `mlp.router`. Every `experts.switch_glu.*` name resolved. A count of
    /// one spelling repeated per layer was read as a census of what was
    /// missing.
    ///
    /// What was really missing was the SHAPE of the block, which
    /// `mlx_lm/models/gemma4_text.py::DecoderLayer.__call__` settles. Layer
    /// 0 of the A4B ships seven norms where the dense 31b ships four:
    ///
    /// ```text
    /// h1 = post_ffn_norm_1(mlp(pre_ffn_norm(h)))     // dense leg
    /// h2 = post_ffn_norm_2(experts(pre_ffn_norm_2(h), router(h)))
    /// h  = post_ffn_norm(h1 + h2) + residual         // join, THEN norm
    /// ```
    ///
    /// Both legs read the same `h`, each has its own pair of norms, and the
    /// shared `post_feedforward_layernorm` lands on the SUM. The branch that
    /// stood here chained off the post-dense residual, reused `mlp_norm` for
    /// the routed pre-norm, stated no leg post-norm, and added where the
    /// reference joins. Firing it measured finite everywhere and **19456 at
    /// its widest against a 1000 ceiling** -- exactly the missing norms, and
    /// exactly why "it runs and produces no NaN" is not the question.
    ///
    /// The router is its own statement and not a projection:
    /// `Router.__call__` is `rms_norm(x, scale * hidden^-0.5)` -> project ->
    /// top-k -> softmax -> `* per_expert_scale[indices]`. Both weights are in
    /// the checkpoint; `router_input_norm` and `router_expert_scale` state
    /// them, and `moe/route.metal::router_topk_scaled` already had the
    /// buffer.
    ///
    /// Nothing refuses here now. What is NOT yet true is a numeric reference:
    /// no `REFERENCES` row holds this checkpoint, so the claim this doc can
    /// make is "shaped like the reference", not "agrees with it".
    fn untraced(&self) -> Option<Refusal> {
        None
    }
}

impl Variant for Gemma4 {
    fn id(&self) -> &'static str {
        self.id
    }

    fn manifest(&self) -> Manifest {
        project::manifest(&self.shape, self.mixture)
    }

    /// The one row in the catalog that fills `kv_shared_layers`.
    ///
    /// It is on [`LoadShape`] because of this generation: a shared
    /// layer's k and v projections are shipped and DEAD, and a contract
    /// that declared them would bind eighteen layers of weights nothing
    /// reads — while the MLX author, which drops them by name, needs to
    /// know which layers those are. A tensor cannot say "ignore me".
    fn load_shape(&self) -> LoadShape {
        LoadShape {
            layers: self.shape.layers,
            // The CHECKPOINT's head dim, unpadded and unrounded — the
            // sliding layers'. A tensor-parallel row split reads this to
            // land on a head boundary, and layer 0's projections are the
            // ones it is measured against.
            head_dim: self.shape.head_dim,
            n_experts: self.mixture.map_or(0, |m| m.num_experts),
            // No Mamba mixer anywhere in this generation.
            mamba_groups: 0,
            kv_shared_layers: self.shape.kv_shared_layers,
            tied_embeddings: self.shape.tied_embeddings,
        }
    }

    /// # Errors
    ///
    /// [`Refusal::Unsupported`] for the A4B: this build has no text for
    /// `attention_k_eq_v` or for a routed gemma-4 block, and the row
    /// says so at the door rather than at its first fire.
    fn deployment(&self, load: Deployed<'_>) -> Result<Deployment, Refusal> {
        if let Some(refusal) = self.untraced() {
            return Err(refusal);
        }
        let mut deployment = project::deployment(&self.shape, self.row(), load);
        deployment.towers = self.towers.clone();
        deployment.advertised = Advertised {
            arch: ARCH,
            max_model_len: self.max_model_len,
            // DERIVED from the towers this deployment actually carries,
            // and read back off the field rather than off `self` so the
            // two cannot be written from different places. The bug this
            // replaces was a hardwired `false` in `driver-cuda` sitting
            // beside a working encoder: a capability computed from the
            // encoder list cannot drift from the encoder list.
            media_encode: deployment.towers.audio.is_some() || deployment.towers.vision.is_some(),
        };
        Ok(deployment)
    }

    /// The two authors, chosen by NAMING rather than by table.
    ///
    /// This is what `contract::HF_ROWS` and `MLX_ROWS` were: two tables
    /// keyed by the same `model_type`, which could name different
    /// generations for one string and did not have to agree. The choice
    /// is a `match` on the row now, so there is no seam for them to
    /// disagree across.
    #[cfg(feature = "contract")]
    fn author(
        &self,
        builder: &mut crate::shared::builder::Builder<'_>,
    ) -> Result<(), model_loader::error::Error> {
        match builder.naming() {
            crate::shared::policy::Naming::Hf => self::contract::author_gemma4(builder),
            crate::shared::policy::Naming::Mlx => self::contract::author_gemma4_mlx(builder),
        }
    }

    /// # Errors
    ///
    /// [`Refusal::Unsupported`] for the A4B, by the same predicate
    /// [`Self::deployment`] refuses on — a row that cannot be deployed
    /// cannot be traced.
    fn trace(
        &self,
        class: model_ir::trace::FireClass,
        load: Deployed<'_>,
    ) -> Result<model_ir::trace::ForwardPlan, Refusal> {
        if let Some(refusal) = self.untraced() {
            return Err(refusal);
        }
        // METAL, through `llama_like_metal` with THIS row projected into
        // the facts it reads. It used to be refused by name here, and
        // [`project::metal_facts`] records why that refusal was wrong
        // about the text it named: `llama_like_metal` carries every
        // gemma-4 field, its own comments are measurements taken on
        // gemma-4-31b, and what was actually missing was the projection.
        if let crate::catalog::Backend::Metal(bind) = load.backend {
            let shape = project::metal_shape(&self.shape, self.mixture);
            let facts = project::metal_facts(&self.shape, self.row(), bind);
            // The kernel set's three refusals, not this row's. This arm
            // asked only the shard question. Neither of the other two
            // fires for a row published today -- every gemma runs at
            // head width 256 and 26B-A4B is refused earlier for having
            // no routed text -- so this is the door being able to SAY
            // what it cannot run, rather than a mis-serving repaired.
            // Without it an off-axis width aborts in `model-compiler`
            // rather than arriving as a sentence.
            crate::shared::llama_like::project::metal_kernel_refusal(&shape, &facts, load, bind)?;
            return Ok(crate::shared::llama_like::forward::llama_like_metal(
                &shape, &facts, class,
            ));
        }
        // CUDA, and the other half of the refusal this comment says
        // expired. It expired for METAL: `llama_like_metal` reads
        // `v_from_k` and gemma-4-31b reproduces MLX's logits through it.
        // The hand-written CUDA text has no such arm — `Gemma4LayerW`
        // declares `v_proj` and layer() matmuls it unconditionally, and
        // `project::trace` is not even handed `k_eq_v` to branch on. So
        // a row whose checkpoint ships no `v_proj` would be traced
        // against one here.
        //
        // `Gemma4Facts::k_eq_v` states the measurement and this is the
        // reader CUDA never grew. `Deployment` used to carry a copy of
        // it "for the driver to read"; no driver read it and none was
        // going to, because the branch that needs it is HERE. Refusing
        // is the whole of what this
        // file can honestly do about it: the alternative is a plan that
        // binds a tensor the checkpoint does not contain.
        if self.k_eq_v {
            return Err(Refusal::Unsupported(
                "gemma-4 31B/26B-A4B on CUDA: these rows read V out of the K projection (`attention_k_eq_v`) and ship no `v_proj`; the hand-written text projects one. The Metal text reads it (`LlamaLikeMetalFacts::v_from_k`) and serves these rows",
            ));
        }
        Ok(project::trace(&self.shape, self.sliding_window, class))
    }

    /// Gemma-4's own turn protocol, and the reason [`Variant::chat`] is
    /// required rather than defaulted: `instruct::create` ended in
    /// `_ => QwenInstruct`, so this generation got ChatML and generated
    /// fluently, ending turns it was not having with an `<|im_end|>` its
    /// vocabulary does not contain. Gemma-4 delimits a turn with the
    /// single tokens `<|turn>` and `<turn|>`, which are not gemma-3's
    /// `<start_of_turn>` either.
    #[cfg(feature = "chat")]
    fn chat(&self, tokenizer: Arc<tokenizer::Tokenizer>) -> Arc<dyn crate::instruct::Instruct> {
        Arc::new(self::chat::Gemma4Instruct::new(tokenizer))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ARCH, Deployed, Gemma4, Gemma4Facts, Gemma4Mixture, NORM_EPS, Towers, VARIANTS, Variant,
        rows,
    };
    // Only the Metal text's tests call the projection by name.
    use super::project;
    use crate::deployment::{AttnOutput, PrefillStyle, Refusal};

    fn row(id: &str) -> &'static Gemma4 {
        VARIANTS.iter().find(|v| v.id == id).expect("row present")
    }

    /// The fixture and the row are the same measurement, so if they
    /// disagree one of them is wrong. Each of these three constructors
    /// was committed as a reading of a real `config.json`; so is the
    /// row, and the row is now the only one a driver reads.
    #[test]
    fn every_row_agrees_with_the_committed_fixture() {
        assert_eq!(row("gemma-4-e2b").shape, Gemma4Facts::gemma_4_e2b());
        assert_eq!(row("gemma-4-e4b").shape, Gemma4Facts::gemma_4_e4b());
        assert_eq!(row("gemma-4-26b-a4b").shape, Gemma4Facts::gemma_4_26b_a4b());
        assert_eq!(
            row("gemma-4-26b-a4b").mixture,
            Some(Gemma4Mixture::gemma_4_26b_a4b())
        );
    }

    /// The table is reachable through the trait, one entry per row, in
    /// the order the rows are written.
    #[test]
    fn the_rows_are_the_variants() {
        assert_eq!(rows().len(), VARIANTS.len());
        let ids: Vec<&str> = rows().iter().map(|r| r.id()).collect();
        assert_eq!(
            ids,
            vec![
                "gemma-4-e2b",
                "gemma-4-e4b",
                "gemma-4-31b",
                "gemma-4-26b-a4b"
            ]
        );
        assert_eq!(rows().len(), 4);
    }

    /// Ids are unique, non-empty and spelled the way a boundary carries
    /// them: lowercase, hyphenated, and naming a MODEL rather than an
    /// encoding of one.
    #[test]
    fn the_ids_are_unique_and_spelled_for_an_operator() {
        let mut seen = std::collections::BTreeSet::new();
        for v in VARIANTS {
            assert!(!v.id.is_empty(), "a row with no id cannot be asked for");
            assert!(seen.insert(v.id), "'{}' appears twice", v.id);
            assert!(
                v.id.chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-'),
                "'{}' is not lowercase-hyphenated",
                v.id
            );
            assert!(
                v.id.starts_with("gemma-4-"),
                "'{}' does not name its generation",
                v.id
            );
            for word in ["fp8", "int4", "bf16", "mlx", "awq"] {
                assert!(!v.id.contains(word), "'{}' names an encoding", v.id);
            }
        }
    }

    /// Every row states a manifest as long as its own stack, and one
    /// that describes a checkpoint it would accept.
    #[test]
    fn every_row_states_a_manifest_of_its_own_stack() {
        for v in VARIANTS {
            let m = v.manifest();
            assert_eq!(m.layers, v.shape.layers, "{}", v.id);
            assert!(!m.tensors.is_empty(), "{} states no tensors", v.id);
        }
    }

    /// The load shape, field by field — including the two that are
    /// almost always zero and are still stated, and the one field in it
    /// that exists because of this generation.
    #[test]
    fn the_load_shape_states_the_shared_tail_and_the_unpadded_head() {
        let e4b = row("gemma-4-e4b").load_shape();
        assert_eq!(e4b.layers, 42);
        assert_eq!(
            e4b.head_dim, 256,
            "the checkpoint's own, not the full layers' 512"
        );
        assert_eq!(e4b.n_experts, 0);
        assert_eq!(e4b.mamba_groups, 0, "no mixer in this generation");
        assert_eq!(
            e4b.kv_shared_layers, 18,
            "eighteen layers ship dead k/v projections"
        );
        assert!(e4b.tied_embeddings);

        let a4b = row("gemma-4-26b-a4b").load_shape();
        assert_eq!(a4b.layers, 30);
        assert_eq!(a4b.head_dim, 256);
        assert_eq!(a4b.n_experts, 128, "the author splits a bank this wide");
        assert_eq!(a4b.kv_shared_layers, 0, "this one shares nothing");
        assert!(a4b.tied_embeddings);

        let e2b = row("gemma-4-e2b").load_shape();
        assert_eq!(e2b.layers, 35);
        assert_eq!(e2b.kv_shared_layers, 20);
        assert_eq!(e2b.n_experts, 0);
    }

    /// The four exceptions, through the trait: two head dims, an SSA
    /// attention output, a planless prefill and a softmax scale of one.
    /// Every one of them was a default body in the trait this replaced.
    #[test]
    fn the_deployment_states_all_four_of_this_generations_exceptions() {
        let d = row("gemma-4-e4b")
            .deployment(Deployed::single())
            .expect("E4B deploys");
        assert_eq!(d.decode_head_dims(), Some((256, 512)));
        assert_eq!(d.attn_output, AttnOutput::StatedArgs);
        assert_eq!(d.prefill, PrefillStyle::Planless);
        assert!(d.attention.iter().all(|a| a.sm_scale == 1.0));
        assert_eq!(d.norm_eps, NORM_EPS);
        assert!(d.shares_kv());
    }

    /// What a guest program is told, which is the last thing that kept a
    /// parsed `config.json` alive inside the driver. `media_encode` was
    /// hardwired `false` there while four GPU paths served this
    /// generation's towers.
    #[test]
    fn the_row_advertises_its_own_ceiling_and_its_towers() {
        let d = row("gemma-4-e4b")
            .deployment(Deployed::single())
            .expect("E4B deploys");
        assert_eq!(d.advertised.arch, ARCH);
        assert_eq!(d.advertised.max_model_len, 131_072);
        assert!(
            d.advertised.media_encode,
            "a `false` here is not a missing label, it is a missing EXECUTOR: the worker \
             declines to build an encode executor at all when this is clear, so gemma-4's \
             ported, bound and cosine-verified vision and audio towers become unreachable \
             through the engine while the four GPU tests that call the entry point directly \
             keep passing — which is exactly how they were unreachable before this field"
        );

        let e2b = row("gemma-4-e2b")
            .deployment(Deployed::single())
            .expect("E2B deploys");
        assert_eq!(e2b.advertised.max_model_len, 131_072);
    }

    /// The label the row states is the label the front-end selector
    /// accepts — WHOLE, not as a substring.
    ///
    /// `from_arch_name` matches with `==` because a `contains` here is
    /// what handed `qwen3` the Qwen3-VL front end belonging to
    /// `qwen3_5`. Asserting the round trip rather than the spelling is
    /// what keeps this row and that `match` from drifting apart: a
    /// rename on either side fails here instead of silently returning
    /// `None` and leaving an image unencoded.
    #[cfg(feature = "chat")]
    #[test]
    fn the_advertised_label_selects_this_generations_front_ends() {
        use crate::multimodal::{VisionArch, audio_arch_supported};
        let d = row("gemma-4-e4b")
            .deployment(Deployed::single())
            .expect("E4B deploys");
        assert_eq!(
            VisionArch::from_arch_name(d.advertised.arch),
            Some(VisionArch::Gemma4),
            "'{}' selects no vision front end, so the tower loads and nothing routes an \
             image to it",
            d.advertised.arch
        );
        assert!(
            audio_arch_supported(d.advertised.arch),
            "'{}' selects no audio front end",
            d.advertised.arch
        );
    }

    /// One family label over rows that ship different encoders, and the
    /// capability is the encoder list rather than a claim beside it.
    ///
    /// A generation-level `const MAX_MODEL_LEN` would be wrong here and
    /// wrong QUIETLY: the number is only ever advertised, never fired
    /// on, so a 26B row claiming 131 072 truncates a caller's context at
    /// half the published length with no error anywhere.
    #[test]
    fn one_family_label_covers_rows_with_two_different_ceilings() {
        let deployed: Vec<_> = VARIANTS
            .iter()
            .filter_map(|v| v.deployment(Deployed::single()).ok().map(|d| (v.id, d)))
            .collect();
        assert_eq!(
            deployed.len(),
            4,
            "every gemma-4 row advertises now, the A4B included"
        );
        let labels: std::collections::BTreeSet<&str> =
            deployed.iter().map(|(_, d)| d.advertised.arch).collect();
        assert_eq!(labels.len(), 1, "one family, one label");
        assert_eq!(labels.into_iter().next(), Some(ARCH));

        // The invariant, not the value: what is advertised is what is
        // carried. A row can gain or lose a tower and this still holds.
        for (id, d) in &deployed {
            assert_eq!(
                d.advertised.media_encode,
                d.towers.audio.is_some() || d.towers.vision.is_some(),
                "{id} advertises an encode capability its tower list does not back, which is \
                 the `driver-cuda` defect restated one layer up"
            );
        }

        // The A4B cannot be asked through `deployment()`, so its ceiling
        // is held against the row, where the corpus put it.
        let ids: std::collections::BTreeSet<&str> = VARIANTS.iter().map(|v| v.id).collect();
        assert_eq!(ids.len(), VARIANTS.len(), "three ids for one label");
        let a4b = row("gemma-4-26b-a4b");
        assert_eq!(
            a4b.max_model_len, 262_144,
            "`google--gemma-4-26B-A4B-it.json` states it"
        );
        assert_ne!(
            a4b.max_model_len,
            row("gemma-4-e4b").max_model_len,
            "the two series publish different ceilings, which is why this is a row field"
        );
    }

    /// The encoders reach a driver, and they are not the same encoders.
    ///
    /// The E-series carries an audio tower and a 16-block vision tower;
    /// the A4B's `audio_config` is `null` and its vision tower is 27
    /// blocks of 1152. A generation-level default here would hand the
    /// A4B twelve conformer blocks whose weights are not in the package
    /// — which is the `GemmaAudioConfig::default()` defect, restated.
    #[test]
    fn the_two_series_ship_different_encoders() {
        let d = row("gemma-4-e4b")
            .deployment(Deployed::single())
            .expect("E4B deploys");
        let audio = d.towers.audio.expect("the E-series ships a conformer");
        assert_eq!(audio.layers, 12);
        assert_eq!(audio.feature_size, 128, "128 mel bins in, per frame");
        assert_eq!(
            (audio.subsample_channels_0, audio.subsample_channels_1),
            (128, 32)
        );
        assert_eq!(audio.output_dims, 1536);
        assert_eq!(
            (audio.chunk_size, audio.context_left, audio.context_right),
            (12, 13, 0)
        );
        let vision = d.towers.vision.expect("the E-series ships a vision tower");
        assert_eq!((vision.layers, vision.hidden), (16, 768));
        assert_eq!(
            vision.rope_theta, 100.0,
            "the tower's base is its own, not the decoder's"
        );

        let a4b = row("gemma-4-26b-a4b");
        assert!(
            a4b.towers.audio.is_none(),
            "`audio_config` is null in this checkpoint"
        );
        let a4b_vision = a4b.towers.vision.expect("the A4B ships a vision tower");
        assert_eq!((a4b_vision.layers, a4b_vision.hidden), (27, 1152));
        assert_ne!(
            a4b_vision, vision,
            "two rows of one generation carry two different encoders, and a shared \
             default would give one of them the other's"
        );
    }

    /// A capability is COMPUTED from what the row carries.
    ///
    /// Stated beside the towers instead, it was a `false` sitting next
    /// to a working encoder for as long as the towers existed.
    #[test]
    fn a_row_advertises_encoding_exactly_when_it_carries_an_encoder() {
        for v in VARIANTS {
            let carries = v.towers.audio.is_some() || v.towers.vision.is_some();
            assert!(
                carries,
                "every gemma-4 package in the corpus ships at least one tower"
            );
        }
        let text_only = Gemma4 {
            id: "gemma-4-hypothetical-text-only",
            shape: Gemma4Facts::gemma_4_e4b(),
            mixture: None,
            sliding_window: 512,
            k_eq_v: false,
            max_model_len: 131_072,
            towers: Towers::default(),
        };
        let d = text_only
            .deployment(Deployed::single())
            .expect("a text-only stack deploys");
        assert!(
            !d.advertised.media_encode,
            "a row with no tower must not advertise an encode entry point"
        );
    }

    /// The row that used to LOAD and not SERVE.
    ///
    /// This test asserted a refusal and asserted its WORDING, and the
    /// wording is why it is worth keeping in inverted form: the sentence
    /// went through three phrasings and each of the first two guarded a
    /// claim that had already stopped being true. It blamed the routed
    /// text, which was written; then the `switch_glu` contract, which
    /// published every name; and finally the three join norms, which is
    /// what was actually missing and is now stated.
    ///
    /// So what it holds now is the row SERVING, and serving with the
    /// mixture's own shape rather than as a dense row that happens to
    /// deploy.
    #[test]
    fn the_mixture_row_serves_with_the_mixtures_own_shape() {
        let a4b = row("gemma-4-26b-a4b");
        assert!(!a4b.manifest().tensors.is_empty());
        assert_eq!(a4b.load_shape().n_experts, 128);
        assert!(a4b.untraced().is_none(), "nothing refuses the mixture now");
        let d = a4b
            .deployment(Deployed::single())
            .expect("the A4B serves");
        assert_eq!(d.shape.experts_per_token, 8);
        assert_eq!(d.shape.moe_intermediate, 704);
        // The DENSE leg is still there, at its own width, because the
        // mixture sits beside it rather than replacing it. A row whose
        // `intermediate` had collapsed to the routed one would deploy and
        // run the wrong FFN.
        assert_eq!(d.shape.intermediate, 2112);
    }

    /// The three dense rows deploy, so the refusal above is about the
    /// A4B and not about the generation.
    #[test]
    fn the_dense_rows_serve() {
        for id in ["gemma-4-e2b", "gemma-4-e4b", "gemma-4-31b", "gemma-4-26b-a4b"] {
            let d = row(id).deployment(Deployed::single());
            assert!(d.is_ok(), "{id} refused, and its legs are all traced");
        }
    }

    /// One predicate answers both questions, so a row cannot deploy and
    /// then fail to trace.
    ///
    /// The k/v mode used to be half of that predicate, and this test
    /// asserted a hypothetical row carrying it alone would refuse. It
    /// SERVES: gemma-4-31b is exactly that row — dense, `k_eq_v` — and
    /// it reproduces MLX's logits on real weights. The hypothetical is
    /// kept, inverted, because the pairing is the thing worth holding:
    /// whatever `untraced` answers, `deployment` must agree.
    ///
    /// `untraced` answers `None` for EVERY row now — the mixture was the
    /// last refusal and its text is written. The pairing is asserted
    /// across the whole list rather than on one row, because a predicate
    /// that refuses nothing can be paired with anything by accident.
    #[test]
    fn a_row_that_cannot_deploy_cannot_trace_either() {
        for v in VARIANTS {
            assert_eq!(
                v.untraced().is_none(),
                v.deployment(Deployed::single()).is_ok(),
                "'{}' answers `untraced` and `deployment` differently",
                v.id
            );
        }
        assert!(row("gemma-4-26b-a4b").untraced().is_none());
        assert!(row("gemma-4-e4b").untraced().is_none());

        let k_eq_v_only = Gemma4 {
            id: "gemma-4-hypothetical",
            shape: Gemma4Facts::gemma_4_e4b(),
            mixture: None,
            sliding_window: 512,
            k_eq_v: true,
            max_model_len: 131_072,
            towers: Towers::default(),
        };
        assert!(
            k_eq_v_only.untraced().is_none(),
            "V read from K is a different attention and this build has it"
        );
        assert!(k_eq_v_only.deployment(Deployed::single()).is_ok());

    }

    /// A METAL load traces the llama-like Metal text, at THIS row's
    /// widths.
    ///
    /// This test asserted the opposite — that every variant refused with
    /// a `project::NO_METAL` constant — and the refusal it pinned said
    /// `llama_like_metal` "states the widths without the fused-projection
    /// split or the shared-cache attention built on them". It states all
    /// of them, and gemma-4-31b passed all twelve of `driver-metal`'s
    /// real-weight gates through that text at 5d7e05526. What was missing
    /// was the PROJECTION, which `driver-metal/src/model/text.rs` used to
    /// do from tensor probes and which [`project::metal_facts`] does from
    /// the row.
    ///
    /// So the guard changed shape rather than went away. Its point was
    /// that a row must not be traced AS a llama — the `LLAMA_LIKE` table
    /// answered from an architecture string and could say yes to a row
    /// whose text does not exist — and what stops that is not a refusal
    /// but the widths: a gemma-4 traced as a generic llama would carry
    /// ONE attention shape, and this asserts the two.
    #[test]
    fn a_metal_load_traces_this_rows_two_attention_shapes_and_not_a_llamas() {
        use crate::catalog::{Backend, Deployed, MetalBinding};
        use model_ir::trace::FireClass;

        let bind = MetalBinding {
            quant_group: 64,
            quant_bits: 4,
            router_quant_group: 0,
            router_quant_bits: 0,
            moe_mxfp4: false,
            fuse_residual_gemv: true,
            paged_multi_batch: true,
            qmm_multi_batch: true,
            add_bias: false,
        };
        assert!(!VARIANTS.is_empty());
        let mut two_shaped = 0usize;
        for v in VARIANTS {
            if v.untraced().is_some() {
                continue;
            }
            for class in [FireClass::Prefill, FireClass::Decode] {
                let plan = v
                    .trace(class, Deployed::metal(&bind))
                    .expect("this row projects into the llama-like Metal text");
                assert!(
                    plan.family.starts_with("llama_like.metal."),
                    "`{}` traced `{}`",
                    v.id,
                    plan.family
                );
                assert_ne!(plan.ops.len(), 0);
            }

            // The two shapes, which is what "not as a llama" means. A
            // generic llama states ONE `head_dim` and ONE `kv_heads`;
            // this row's full layers are twice as wide per head and
            // carry a quarter the KV heads, and the facts say so.
            //
            // Asserted rather than asked: `metal_facts` divides the
            // rotary dim by this width, so a row that stated 0 would
            // publish an infinite rotary fraction rather than take some
            // other arm.
            let facts = project::metal_facts(&v.shape, v.row(), &bind);
            assert!(
                v.shape.global_head_dim > 0,
                "`{}` states no full-layer head dim, which is a llama and \
                 not a gemma-4",
                v.id
            );
            two_shaped += 1;
            assert_eq!(facts.global_head_dim, v.shape.global_head_dim);
            assert_eq!(facts.global_kv_heads, v.shape.global_kv_heads);
            let full = (0..v.shape.layers).find(|l| v.shape.is_full_attn(*l));
            let slide = (0..v.shape.layers).find(|l| !v.shape.is_full_attn(*l));
            if let (Some(full), Some(slide)) = (full, slide) {
                assert_ne!(
                    facts.head_dim_at(full, v.shape.head_dim),
                    facts.head_dim_at(slide, v.shape.head_dim),
                    "`{}` reads one width at both layer kinds",
                    v.id
                );
            }
            // Two rotary bases, and `rope_theta_at` picks off the same
            // window list the widths do.
            assert_eq!(facts.rope_theta, project::ROPE_THETA_GLOBAL);
            assert_eq!(facts.rope_theta_sliding, project::ROPE_THETA_LOCAL);
            // The three weightless facts no checkpoint could contradict.
            assert!(facts.v_norm, "`{}` dropped the V norm", v.id);
            assert_eq!(
                facts.activation,
                crate::shared::llama_like::forward::facts::Activation::Geglu
            );
            assert!(facts.embed_scale > 0.0, "gemma scales its embeddings");
        }
        assert!(two_shaped > 0, "no variant exercised the two-shape path");

        // A SHARDED Metal load is still refused: `LlamaLikeMetalFacts`
        // has no shard vocabulary, so the text would state the whole
        // model's widths against one rank's slice.
        let sharded = Deployed {
            backend: Backend::Metal(&bind),
            tp_size: 4,
            layer_scalars: &[],
        };
        let served = VARIANTS
            .iter()
            .find(|v| v.untraced().is_none())
            .expect("one row traces");
        assert!(served.trace(FireClass::Decode, sharded).is_err());

        // And CUDA traces exactly the rows whose V has its own
        // projection. The hand-written text declares a `v_proj` and
        // matmuls it unconditionally, so a `k_eq_v` row would be traced
        // against a tensor its checkpoint does not ship — the half of
        // the old `gemma4_attention_k_eq_v || gemma4_enable_moe`
        // predicate that expired for Metal and not for here.
        for v in VARIANTS {
            if v.untraced().is_some() {
                continue;
            }
            let cuda = v.trace(FireClass::Decode, Deployed::single());
            if v.k_eq_v {
                assert!(cuda.is_err(), "{}: no CUDA text reads V from K", v.id);
                continue;
            }
            let cuda = cuda.expect("CUDA still traces");
            assert!(cuda.family.starts_with("gemma4"), "{}", cuda.family);
        }
        // Not vacuous in either direction: the catalog holds rows of
        // both kinds, so this asserts a split rather than a blanket.
        assert!(VARIANTS.iter().any(|v| v.k_eq_v && v.untraced().is_none()));
        assert!(VARIANTS.iter().any(|v| !v.k_eq_v && v.untraced().is_none()));
        assert!(matches!(Deployed::single().backend, Backend::Cuda));
    }
}

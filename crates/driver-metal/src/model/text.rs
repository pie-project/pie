//! Which text the loaded checkpoint is.
//!
//! # Selecting is not choosing
//!
//! "Nothing in the driver may choose a kernel" is the crate's governing rule,
//! and a reader could mistake this module for a breach of it. It is not, and
//! the distinction is worth stating precisely because it will be reached for
//! again when the other three families land.
//!
//! A *choice* is the driver deciding what to run. What happens here is a
//! **lookup**: the checkpoint states its architecture, and this answers with
//! the text written for it. Nothing about which kernels fire is decided here —
//! the text names every symbol, the lowering flattens it, and the executor
//! walks the result. Remove this module and the same kernels would run; you
//! would simply have no way to say which model you loaded.
//!
//! The test is the one `metal.md` gives for the whole crate: *does removing it
//! change which kernels fire?* It does not.
//!
//! # Why the driver and not the engine
//!
//! It could sit in the seam instead. It sits here because running the model it
//! loaded is the driver's job, and because `driver-cuda`'s shell does the
//! same in `pie_cuda_launch` for the same reason. A seam that selected texts
//! would have to know every family, and it would learn a new one every time a
//! driver did.

use model_compiler::trace::{FireClass, ForwardPlan};

/// Why no text could be selected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Unfamiliar {
    /// No text has been written for this architecture.
    ///
    /// Three of four families are in this state today (`gemma4`, `gpt-oss`,
    /// `qwen`), and it is the honest report: their driver code still exists
    /// but the text that would replace it does not.
    NoText {
        /// What the checkpoint called itself.
        arch: String,
        /// The architectures a text does exist for.
        known: Vec<&'static str>,
    },
}

/// The architectures `llama_like`'s text serves.
///
/// One text covers many architectures — that is what makes it a *family* — so
/// this is a list rather than a name. Every entry is a deployment the family's
/// facts can describe.
const LLAMA_LIKE: &[&str] = &[
    "llama",
    "llama3",
    "llama4",
    "mistral",
    "phi3",
    "olmo2",
    "qwen2",
    "qwen3",
    // The mixtures and the branch, which are FACTS of this text and not
    // families beside it -- and which were missing here for exactly as long as
    // nothing asked. A text that models an architecture but whose driver
    // refuses its name serves it in principle and not at all in practice: the
    // seam calls `plan_for` before anything else, so `gpt_oss` and `gemma4`
    // could not reach the device through the generic path, only through the
    // per-family layers this crate is retiring.
    "qwen3_moe",
    "gpt_oss",
    "gemma4",
];

/// One architecture's name, with the punctuation two spellings disagree
/// about removed.
///
/// A checkpoint's `config.json` carries both a `model_type` (`qwen3_moe`) and
/// an `architectures[0]` (`Qwen3MoeForCausalLM`), and the driver used to
/// reduce the second — lowercase it, drop its `ForCausalLM` tail — to reach
/// `qwen3moe`, the same architecture without the underscore.
///
/// The seam passed the reduced stem; the gate in `device_checkpoint_names`
/// passed `model_type`. So the gate proved that every name a text states
/// resolves, over five checkpoints, while the seam refused two of those
/// checkpoints outright — the two whose spellings differ. `llama`, `qwen3`
/// and `gemma4` are the same either way, which is why nothing noticed.
///
/// THE SOURCE OF THAT DISAGREEMENT IS GONE. A catalog row STATES its
/// architecture (`model::deployment::Advertised::arch`) instead of having one
/// derived from whichever half of a config the caller happened to hold, so
/// there is one spelling to reduce rather than two to reconcile. The
/// reduction stays because a label is still typed by hand in the list below
/// and in an operator's boot file, and one rule covers a spelling this driver
/// has never seen where a second table entry would only cover the ones
/// someone thought to add.
fn canonical(arch: &str) -> String {
    arch.chars()
        .filter(|c| *c != '_' && *c != '-')
        .flat_map(char::to_lowercase)
        .collect()
}

/// Every architecture some text serves.
#[must_use]
pub fn known() -> Vec<&'static str> {
    LLAMA_LIKE.to_vec()
}

/// The text for `arch`, traced for `class`.
///
/// `facts` and `metal` are the deployment's, and they are the caller's to
/// supply because they come from the row's projected
/// `model::deployment::Deployment` — not from anything this module could
/// derive.
///
/// # Errors
///
/// [`Unfamiliar::NoText`], naming the architecture and what is known. A driver
/// that guessed a text here would run a different model's program against this
/// checkpoint's weights, which is fluent nonsense rather than a failure.
pub fn plan_for(
    arch: &str,
    class: FireClass,
    facts: &model::families::llama_like::forward::facts::LlamaLikeFacts,
    metal: &model::families::llama_like::forward::facts::LlamaLikeMetalFacts,
) -> Result<ForwardPlan, Unfamiliar> {
    if serves(arch) {
        return Ok(model::families::llama_like::forward::llama_like_metal(
            facts, metal, class,
        ));
    }
    Err(Unfamiliar::NoText {
        arch: arch.to_string(),
        known: known(),
    })
}

/// Whether any text serves `arch`.
#[must_use]
pub fn serves(arch: &str) -> bool {
    let want = canonical(arch);
    LLAMA_LIKE.iter().any(|known| canonical(known) == want)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_text_serves_many_architectures_which_is_what_a_family_is() {
        assert!(serves("qwen3"));
        assert!(serves("llama3"));
        assert!(serves("mistral"));
        // The three that are FACTS of the same text rather than families
        // beside it: a mixture, a mixture with sinks and a clamped
        // activation, and a side network with a branch. One text, nine
        // architectures, no per-family arm anywhere below this line.
        assert!(serves("qwen3_moe"));
        assert!(serves("gpt_oss"));
        assert!(serves("gemma4"));
    }

    /// The row's own spelling is one this list answers for.
    ///
    /// The bug this pins was invisible to everything: a `config.json` carries
    /// a `model_type` (`qwen3_moe`) AND an `architectures[0]`
    /// (`Qwen3MoeForCausalLM`), the driver's `arch_stem` turned the second
    /// into `qwen3moe`, and the SEAM passed the stem while the load-time name
    /// gate passed `model_type`. So the gate reported five checkpoints
    /// resolving every name their texts state, while the seam refused two of
    /// them at `plan_for` — and both reports were true, of different
    /// questions.
    ///
    /// There is ONE spelling now and it is the row's: a catalog row states
    /// `Advertised::arch`, and that string is what `serve/load.rs` hands to
    /// [`plan_for`]. So the question is no longer "do two derivations of a
    /// name agree" — it is the sharper one this asserts, that the label a row
    /// hands out is a label this list answers for. A row whose arch nothing
    /// here serves is a checkpoint that loads and cannot fire, and that is
    /// exactly what `tests/catalog_coverage.rs` enumerates.
    #[test]
    fn a_row_states_one_arch_so_two_spellings_cannot_disagree() {
        // Every arch in the list resolves under the punctuation and case a
        // hand-typed label may carry, which is the whole of what the two
        // spellings used to differ by.
        for known in known() {
            assert!(serves(known), "`{known}` is in the list and must resolve");
            assert!(
                serves(&known.to_uppercase()),
                "`{known}` typed in caps is the same architecture"
            );
            assert!(
                serves(&known.replace('_', "-")),
                "`{known}` hyphenated is the same architecture"
            );
        }
    }

    /// A spelling neither half of a config has produced yet still resolves.
    ///
    /// The list used to carry `qwen3_moe` AND `qwen3moe` as two entries, so
    /// it answered for exactly the spellings someone had thought to add. It
    /// carries one now and reduces before comparing, which is the difference
    /// between a table of answers and a rule: `Qwen3-Moe` is not in the list
    /// and never was, and it is the same architecture.
    #[test]
    fn a_spelling_the_list_does_not_carry_still_resolves() {
        for spelling in ["Qwen3_Moe", "QWEN3MOE", "qwen3-moe", "Gpt-Oss", "GPT_OSS"] {
            assert!(
                serves(spelling),
                "{spelling} names an architecture a text serves"
            );
        }
        // The rule reduces punctuation and case, not letters: a different
        // architecture stays different.
        assert!(!serves("qwen3_5"));
        assert!(!serves("qwen35"));
    }

    #[test]
    fn an_architecture_with_no_text_says_so_and_says_what_is_known() {
        // `qwen3_5` is the GDN family: it interleaves linear attention with
        // full attention, which no Metal text models. A driver that guessed
        // would run another model's program against these weights, which is
        // fluent nonsense rather than a failure.
        assert!(!serves("qwen3_5"));
        let facts = model::families::llama_like::forward::facts::LlamaLikeFacts::qwen3_0_6b();
        let metal = model::families::llama_like::forward::facts::LlamaLikeMetalFacts::synthetic();
        match plan_for("qwen3_5", FireClass::Decode, &facts, &metal) {
            Err(Unfamiliar::NoText { arch, known }) => {
                assert_eq!(arch, "qwen3_5");
                assert!(known.contains(&"qwen3"), "and what IS served: {known:?}");
            }
            Ok(_) => panic!("gemma4 has no metal text yet"),
        }
    }

    #[test]
    fn a_served_architecture_traces_a_plan_of_that_family() {
        let facts = model::families::llama_like::forward::facts::LlamaLikeFacts::qwen3_0_6b();
        let metal = model::families::llama_like::forward::facts::LlamaLikeMetalFacts::synthetic();
        let plan = plan_for("qwen3", FireClass::Decode, &facts, &metal).expect("qwen3 is served");
        assert!(
            plan.family.starts_with("llama_like"),
            "the plan states its family: {}",
            plan.family
        );
    }
}

/// The deployment's facts, derived from the row's projected shape.
///
/// # Why this is not a fixture
///
/// It replaced one. The seam's `launch` took `LlamaLikeFacts::qwen3_0_6b()` —
/// a TEST fixture — for every checkpoint, so a deployment with different head
/// counts would have traced another model's program against its weights and
/// answered fluent nonsense. A fixture standing where a fact belongs is the
/// exact defect this crate keeps finding, and it is worth naming when it is
/// one's own.
///
/// # What is still assumed, and why each is stated rather than hidden
///
/// `DecodeGeometry` carries the shape; three facts it has no field for are
/// stated here with the reason:
///
/// * `qk_norm` — read from the checkpoint's own tensors, because a per-head
///   Q/K norm exists exactly when the weights for it do.
/// * `fused_qkv` — whether the deployment binds ONE packed projection. This
///   asks the tensors too: a checkpoint with `self_attn.qkv_proj` fused it,
///   one with `q_proj`/`k_proj`/`v_proj` did not, and the text traces
///   differently for each.
/// * `qkv_bias` — the qwen-2 attention biases, again by tensor.
///
/// Asking the checkpoint rather than the config is deliberate: the config
/// states an architecture and the tensors state a BINDING, and every one of
/// these three is a binding fact.
#[must_use]
pub fn facts_from(
    geometry: &crate::batch::DecodeGeometry,
    has_tensor: impl Fn(&str) -> bool,
) -> (
    model::families::llama_like::forward::facts::LlamaLikeFacts,
    model::families::llama_like::forward::facts::LlamaLikeMetalFacts,
) {
    facts_from_with(geometry, has_tensor, |_| false)
}

/// [`facts_from`], told which weights the load left in MXFP4.
///
/// The second probe is the one a mixture needs. A checkpoint need not
/// quantize uniformly -- `mlx-community/gpt-oss-20b-MXFP4-Q4` names 98
/// tensors as affine/64/4 and leaves the expert banks out, so they take the
/// top-level default, mxfp4/32 -- and reading a bank with the dense format is
/// 909,207 NaNs rather than a near miss.
///
/// `Loaded::mxfp4` is what answers it: the LOAD gets the bytes onto the
/// device unchanged and says what they are, and this is the binder deciding
/// what they mean.
pub fn facts_from_with(
    geometry: &crate::batch::DecodeGeometry,
    has_tensor: impl Fn(&str) -> bool,
    is_mxfp4: impl Fn(&str) -> bool,
) -> (
    model::families::llama_like::forward::facts::LlamaLikeFacts,
    model::families::llama_like::forward::facts::LlamaLikeMetalFacts,
) {
    use model::families::llama_like::forward::facts::{LlamaLikeFacts, LlamaLikeMetalFacts};
    use model_compiler::dsl::{ScaleLayout, WeightRepr};

    let sandwich_norms = has_tensor("layers.0.pre_feedforward_layernorm.weight");

    let facts = LlamaLikeFacts {
        hidden: geometry.hidden,
        layers: geometry.n_layers,
        q_heads: geometry.n_q_heads,
        kv_heads: geometry.n_kv_heads,
        head_dim: geometry.head_dim,
        intermediate: geometry.intermediate,
        vocab: geometry.vocab,
        rope: model_compiler::trace::RopeKind::Standard,
        // gemma scales by `(1 + w)` and every other family by `w`, and gemma
        // norms BOTH ways round each sub-layer where llama norms once. Both
        // were hardcoded to llama's answer, so a gemma checkpoint passed
        // `serves` and then ran as a llama: the `(1 + w)` became `w`, and the
        // two output norms were dropped while `pre_feedforward_layernorm` was
        // bound where the attention's output norm belonged. Nothing faults.
        // Decided by which norms the checkpoint SHIPS, the same rule
        // `lowering/resolve.rs` already uses to bind them: "which one a
        // checkpoint means is decided by which one it ships." A stack that
        // norms both ways round each sub-layer publishes a
        // `pre_feedforward_layernorm`; one that norms once does not, and has
        // nothing for that name to bind to.
        //
        // The two move together on every checkpoint this driver serves, and
        // that is an observation rather than a law: the `(1 + w)` scaling is
        // a weight CONVENTION and the sandwich is an architecture. A stack
        // that published one without the other would break here, loudly --
        // this comment is the place to look.
        norm_variant: if sandwich_norms {
            model_compiler::trace::NormVariant::Gemma
        } else {
            model_compiler::trace::NormVariant::Plain
        },
        norm_placement: if sandwich_norms {
            model::families::llama_like::forward::facts::NormPlacement::Sandwich
        } else {
            model::families::llama_like::forward::facts::NormPlacement::Pre
        },
        qk_norm: if has_tensor("layers.0.self_attn.q_norm.weight") {
            model_compiler::facts::QkNorm::PerHead
        } else {
            model_compiler::facts::QkNorm::Off
        },
        fused_qkv: has_tensor("layers.0.self_attn.qkv_proj.weight"),
        tied_embeddings: geometry.tied_embeddings,
        qkv_bias: has_tensor("layers.0.self_attn.q_proj.bias"),
        // Asked of the TENSORS, like the other binding facts: a router weight
        // in the checkpoint is a mixture and its absence is a dense FFN. The
        // counts come from the geometry, which the fire already states.
        // Every router spelling, because a mixture that answers to one the
        // probe does not know is read as a DENSE model: the text then states
        // `mlp.gate_proj` against a checkpoint that publishes only expert
        // banks, and every FFN name misses. gpt-oss spells it `mlp.router`.
        n_experts: if ["layers.0.mlp.gate.weight", "layers.0.mlp.router.weight"]
            .into_iter()
            .any(&has_tensor)
        {
            geometry.n_experts
        } else {
            0
        },
        experts_per_token: geometry.experts_per_token,
        moe_intermediate: geometry.moe_intermediate,
        // A shared expert is its own tensor, and qwen3-moe has none.
        shared_intermediate: if has_tensor("layers.0.mlp.shared_expert.up_proj.weight") {
            geometry.moe_intermediate
        } else {
            0
        },
    };
    let metal = LlamaLikeMetalFacts {
        fuse_residual_gemv: true,
        paged_multi_batch: true,
        qmm_multi_batch: true,
        // The checkpoint's own affine format. `AffineFormat` is what the
        // checkpoint's `config.json` stated; a pipeline built for the wrong point returns
        // fluent nonsense rather than failing, which is why neither number is
        // inferred from a tensor's shape (g64/b8 and g128/b4 pack identically).
        proj_repr: WeightRepr::Scaled {
            layout: ScaleLayout::PerGroup,
            group: geometry.quant.group,
            axis: 0,
            zero_point: true,
        },
        affine_bits: geometry.quant.bits,
        // Uniform, for now, and the reason is worth stating because it is a
        // load-time fact and not a text one.
        //
        // `metal_storage_target` sets `native_mxfp4_moe: false`, which tells
        // the loader this driver has no MXFP4 routed kernel and its banks must
        // be TRANSCODED to affine at load. When that transcode runs, one
        // format serves the whole checkpoint and `None` is right.
        //
        // It does not run yet -- the load takes gpt-oss's bytes unchanged --
        // so the banks reach the device as mxfp4/32 while the text reads them
        // as affine/64, which is the 909,207 NaNs
        // `the_first_statement_that_writes_a_nan_says_which_one_it_is` points
        // at `affine_qmv_routed_bfloat16_gs_64_b_4`, layer 0.
        //
        // The mechanism to say otherwise now exists (`moe_repr`), so whichever
        // way that lands -- the loader transcodes, or this driver grows the
        // native kernel and states the bank's own format here -- the text can
        // express it.
        // The expert bank's OWN format, asked of the checkpoint. `None` is
        // "the same as the dense projections", which is every checkpoint but
        // gpt-oss.
        moe_repr: is_mxfp4("layers.0.mlp.experts.gate_proj.weight")
            .then_some(model_compiler::dsl::WeightRepr::Mxfp4Marlin),
        moe_bits: 4,
        // The narrowest rung, which is what a short window fires; `bn = 32` is
        // the only column tile the residual GEMM is instantiated at.
        qmm_tile: (16, 32),
        // Asked of the tensors, like the other binding facts. It is false for
        // every MLX checkpoint -- `compile_load_plan` authors with
        // `Projections::InPlace` and the join declines under it -- but asking
        // is what makes that a finding rather than an assumption.
        gate_up_fused: has_tensor("layers.0.mlp.gate_up_proj.fused.weight"),
        // The checkpoint's own `rms_norm_eps`. A norm handed zero divides by
        // the root of the mean square alone, which for a near-zero row is an
        // infinity the next kernel spreads everywhere.
        rms_eps: geometry.eps,
        // The checkpoint's own rotary base.
        rope_theta: geometry.rope_theta,
        // The SLIDING layers' base, when the deployment states a second one.
        // `rope_theta_at` picks between them off the window list.
        rope_theta_sliding: geometry.rope_theta_sliding,
        // The FULL-attention layers' own attention shape, when it differs.
        // `head_dim_at`/`kv_heads_at` pick off the SAME window list, so the
        // two per-layer-type facts cannot disagree about which layers are
        // full.
        global_head_dim: geometry.global_head_dim,
        global_kv_heads: geometry.global_kv_heads,
        // The rotation's extent on those layers, which reaches the GRID
        // through the rope rows' `grid_param` rather than the kernel.
        full_partial_rotary: geometry.full_partial_rotary,
        // Whether the ladder is RESCALED, in which case no base expresses it
        // and the driver hands over a table instead.
        rope_freq_table: geometry.rope_freq_factor > 0.0,
        // gemma's side network, asked of the TENSORS: a checkpoint that ships
        // a per-layer embedding table is a deployment that has one.
        per_layer_emb_dim: geometry.per_layer_emb_dim,
        // gemma's per-layer SCALAR, asked of the tensors for the same reason
        // the PLE is: `layer_scalar` ships with gemma-4-31b, which states
        // `hidden_size_per_layer_input: 0` and so has no PLE at all. The two
        // are alternatives in the text's tail, and hardcoding this one false
        // dropped the tail entirely for every gemma deployment.
        per_layer_scalar: has_tensor("layers.0.layer_scalar"),
        dense_beside_moe: false,
        // The CONFIG's, not a tensor's. Asking layer 0 was the first draft
        // and it is a SLIDING layer, which ships its `v_proj` — only the full
        // ones do not, and a fact derived from the wrong layer is false for
        // exactly the deployment it describes.
        v_from_k: geometry.attention_k_eq_v,
        kv_shared_layers: geometry.kv_shared_layers,
        // gemma's readout cap. Zero is "none" and the text names nothing.
        logit_softcap: geometry.final_logit_softcap,
        // Asked of the TENSORS: a sink is a weight, and a checkpoint that
        // ships one is a deployment that has them.
        attn_sinks: has_tensor("layers.0.self_attn.sinks"),
        // WHICH activation, and every branch is reachable. A swiglu limit of
        // zero is "this deployment is not gpt-oss" and not a clamp at zero,
        // which would zero the gate branch entirely.
        //
        // `Geglu` is gemma's, and it had no way to be selected at all: this
        // read `swiglu_limit > 0.0` and fell through to `SiluMul`, so every
        // gemma checkpoint ran GELU's gate as SiLU. The two agree to about
        // 2% at the origin and diverge from there — finite, plausible, wrong.
        // `hidden_activation: gelu_pytorch_tanh` is what gemma-4 states, and
        // the driver now READS that instead of inferring it from the family:
        // the importer used to replace the statement with the `silu` default,
        // so a family flag was the only signal left.
        activation: if geometry.swiglu_limit > 0.0 {
            model::families::llama_like::forward::facts::Activation::SwiGlu {
                limit: geometry.swiglu_limit,
                alpha: geometry.swiglu_alpha,
            }
        } else if geometry.gelu_gate {
            model::families::llama_like::forward::facts::Activation::Geglu
        } else {
            model::families::llama_like::forward::facts::Activation::SiluMul
        },
        // Empty is every layer attending the whole context, which is what a
        // llama-like deployment does. `DecodeGeometry` carries no window at
        // all, so this is the honest answer and not a default: the families
        // that alternate (gemma4, gpt-oss) will need the geometry to state one
        // before their texts can, and stating it here from nothing would make
        // that a silent wrong answer instead of a missing one.
        // Which layers slide. A gemma4 stack alternates: every
        // `full_attn_every`-th layer attends everything and the rest attend a
        // window. Empty for a deployment that does not alternate, which is
        // every llama-like one.
        window_left: if geometry.full_attn_every > 1 && geometry.sliding_window > 0 {
            (0..geometry.n_layers)
                .map(|l| {
                    if (l + 1).is_multiple_of(geometry.full_attn_every) {
                        -1
                    } else {
                        i32::try_from(geometry.sliding_window).unwrap_or(-1)
                    }
                })
                .collect()
        } else {
            Vec::new()
        },
    };
    (facts, metal)
}

#[cfg(test)]
mod facts_tests {
    use super::*;

    fn geometry() -> crate::batch::DecodeGeometry {
        crate::batch::DecodeGeometry {
            hidden: 1024,
            n_layers: 28,
            vocab: 151_936,
            n_q_heads: 16,
            n_kv_heads: 8,
            head_dim: 128,
            intermediate: 3072,
            tied_embeddings: true,
            quant: crate::batch::AffineFormat {
                bits: 4,
                group: 64,
            },
            ..crate::batch::DecodeGeometry::default()
        }
    }

    #[test]
    fn the_shape_comes_from_the_row_and_not_from_a_fixture() {
        // This replaced `LlamaLikeFacts::qwen3_0_6b()` standing in the seam's
        // `launch` for every checkpoint — a fixture where a fact belongs, and
        // a deployment with other head counts would have traced another
        // model's program against its weights.
        let (facts, metal) = facts_from(&geometry(), |_| false);
        assert_eq!(facts.hidden, 1024);
        assert_eq!(facts.layers, 28);
        assert_eq!(facts.q_heads, 16);
        assert_eq!(facts.kv_heads, 8);
        assert_eq!(facts.intermediate, 3072);
        assert!(facts.tied_embeddings);
        assert_eq!(metal.affine_bits, 4, "the checkpoint's own affine format");
    }

    #[test]
    fn the_binding_facts_ask_the_tensors_rather_than_the_config() {
        // A config states an architecture; a tensor states a BINDING. Whether
        // this deployment fused its QKV is answerable only by looking.
        let split = facts_from(&geometry(), |_| false).0;
        assert!(!split.fused_qkv, "no fused tensor, so the text traces three");
        assert_eq!(split.qk_norm, model_compiler::facts::QkNorm::Off);
        assert!(!split.qkv_bias);

        let fused = facts_from(&geometry(), |name| {
            name == "layers.0.self_attn.qkv_proj.weight"
                || name == "layers.0.self_attn.q_norm.weight"
                || name == "layers.0.self_attn.q_proj.bias"
        })
        .0;
        assert!(fused.fused_qkv);
        assert_eq!(fused.qk_norm, model_compiler::facts::QkNorm::PerHead);
        assert!(fused.qkv_bias);
    }

    #[test]
    fn the_sandwich_is_read_from_the_norms_the_checkpoint_ships() {
        // Was `geometry.gemma`. The probe name is the load's canonical
        // spelling -- `language_model.model.layers.0.pre_feedforward_
        // layernorm.weight` with the prefix stripped, the same form
        // `layers.0.self_attn.q_norm.weight` takes one field above. A typo
        // here does not fault: it silently reads every checkpoint as
        // single-norm and drops two norms per layer, which is the bug this
        // replaced a family flag to avoid repeating.
        let one = facts_from(&geometry(), |_| false).0;
        assert_eq!(
            one.norm_placement,
            model::families::llama_like::forward::facts::NormPlacement::Pre
        );
        assert_eq!(one.norm_variant, model_compiler::trace::NormVariant::Plain);

        let sandwich =
            facts_from(&geometry(), |n| n == "layers.0.pre_feedforward_layernorm.weight").0;
        assert_eq!(
            sandwich.norm_placement,
            model::families::llama_like::forward::facts::NormPlacement::Sandwich
        );
        assert_eq!(
            sandwich.norm_variant,
            model_compiler::trace::NormVariant::Gemma
        );
    }

    #[test]
    fn the_affine_point_follows_the_checkpoints_group_and_width() {
        // g64/b8 and g128/b4 pack to identical shapes, so neither number can
        // be inferred from a tensor — a pipeline built for the wrong point
        // returns fluent nonsense rather than failing.
        let g = crate::batch::DecodeGeometry {
            quant: crate::batch::AffineFormat {
                bits: 8,
                group: 128,
            },
            ..geometry()
        };
        let (_, metal) = facts_from(&g, |_| false);
        assert_eq!(metal.affine_bits, 8);
        assert_eq!(
            model_compiler::dsl::metal::affine_point(metal.proj_repr, metal.affine_bits),
            "_bfloat16_gs_128_b_8"
        );
    }
}

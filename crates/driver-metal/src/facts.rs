//! What a `pie.model/1` descriptor says about a checkpoint's shape.
//!
//! Portable, and that is the point: the descriptor is text and the facts are
//! integers, so the one reading in the driver that decides a model's geometry
//! can be tested without a GPU, a Metal device or an Apple host. The C++ this
//! replaces (`csrc/src/model_facts.{hpp,cpp}`) is compiled into the driver
//! library and therefore only testable where that library links.
//!
//! # What is deliberately not here
//!
//! * `read_model_facts`, the `config.json` reader. Its own header calls it
//!   test-only and on its way out -- no driver code calls it any more, because
//!   the boot reads a descriptor and normalization happens once, in Rust,
//!   before the driver exists. Porting it would be porting a second answer to
//!   a question that already has one, which is the shape of the disagreement
//!   the descriptor path was introduced to end.
//! * `fill_family_geometry`. It writes into `SetupConfig`, which is the
//!   executor, and the executor is not ported yet. A port of it now would have
//!   to invent a destination type and then be rewritten when the real one
//!   arrives.
//!
//! # Reading rules
//!
//! A key that is present but of the wrong JSON type leaves the default
//! standing rather than becoming zero. That mirrors the C++ helpers
//! (`i32_of` / `u32_of` / `f32_of` all test the value's kind before
//! assigning) and it matters: a default here is a documented number that some
//! geometry was written against, and a silent zero is a shape no checkpoint
//! has.

use serde_json::Value;

/// Which storage schema and decode DAG a checkpoint asks for.
///
/// One place answers this. The geometry and the executor need the same
/// answer, and two independent readings of `model_type` would be two chances
/// to disagree. The lists mirror the `Naming::Mlx` rows of the author
/// registry (`model/src/contract.rs`) -- the loader dispatches on the same
/// strings on its side of the call, and refuses one this table would too.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ModelFamily {
    /// A `model_type` no geometry in this driver is written for.
    #[default]
    Unknown,
    /// Qwen3.5 / Qwen3-Next / Qwen3.6: the gated-delta-net hybrid.
    Qwen35,
    /// Gemma 4, whose shape is stated per attention type.
    Gemma4,
    /// GPT-OSS: flat shape, YaRN rope.
    GptOss,
    /// The llama-shaped families, which differ in values and not in shape.
    Llama,
}

impl ModelFamily {
    /// Classify a `model_type` string.
    ///
    /// Exact string equality, not a prefix or substring test: `qwen3` and
    /// `qwen3_moe` are Llama-shaped while `qwen3_next` is not, so anything
    /// looser than equality puts a GDN hybrid through the dense decoder.
    #[must_use]
    pub fn of(model_type: &str) -> Self {
        const QWEN35: [&str; 7] = [
            "qwen3_5",
            "qwen3_5_text",
            "qwen3_5_moe",
            "qwen3_5_moe_text",
            "qwen3_next",
            "qwen3_next_text",
            "qwen3_6",
        ];
        const LLAMA: [&str; 7] = [
            "llama",
            "llama3",
            "mistral",
            "qwen2",
            "qwen3",
            "qwen3_moe",
            "qwen2_moe",
        ];
        if QWEN35.contains(&model_type) {
            return Self::Qwen35;
        }
        if model_type == "gemma4" || model_type == "gemma4_text" {
            return Self::Gemma4;
        }
        if model_type == "gpt_oss" {
            return Self::GptOss;
        }
        if LLAMA.contains(&model_type) {
            return Self::Llama;
        }
        Self::Unknown
    }

    /// Whether this driver has a geometry for the named `model_type`.
    #[must_use]
    pub fn is_supported(model_type: &str) -> bool {
        Self::of(model_type) != Self::Unknown
    }
}

/// `architectures[0]` reduced to the stem the registries key on.
///
/// HuggingFace names an architecture `<Stem>For<Task>`, and every registry on
/// both sides of the ABI -- this driver's family switch and the runtime's
/// instruct registry -- keys on the STEM, so the task suffix comes off.
///
/// The suffix list is explicit rather than "cut at the first `for`", and that
/// is the whole of the care this needs. `ReformerForCausalLM` lowercases to
/// `reformerforcausallm`, whose first `for` is inside `reformer`: cutting
/// there yields `re`, which is not a family and does not announce itself as
/// one -- it just misses the registry and takes the fallback.
///
/// `forconditionalgeneration` is on the list because leaving it off is what a
/// multimodal release costs. `Gemma4ForConditionalGeneration` stayed whole,
/// missed the `gemma4` row, and took the ChatML fallback. The weights still
/// loaded (the load contract keys on `model_type`, not on this) and the
/// kernels were right, so the model produced fluent text -- of a conversation
/// that was never had, terminated by `<|im_end|>`, a token that is not in
/// gemma's vocabulary and was being spelled out a piece at a time. Both
/// gemma-4 checkpoints ship that architecture name.
///
/// The list is walked longest-first so a suffix containing another cannot be
/// half-stripped.
#[must_use]
pub fn arch_stem(name: &str) -> String {
    let lower = name.to_ascii_lowercase();
    for suffix in ["forconditionalgeneration", "forcausallm"] {
        if lower.len() > suffix.len() && lower.ends_with(suffix) {
            return lower[..lower.len() - suffix.len()].to_string();
        }
    }
    lower
}

/// Everything the driver reads out of a checkpoint's descriptor.
///
/// One flat struct rather than a family enum, because the families overlap in
/// most of their fields and a caller that has to match before it can ask
/// "how many layers" is a caller that grows a match arm per family. The
/// non-zero marker fields (`go_num_hidden_layers`, `ll_num_hidden_layers`,
/// `q35_num_hidden_layers`, `g4_num_hidden_layers`) say which block was read.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelFacts {
    /// The quantization block's width, which the SHAPES cannot recover: a
    /// weight quantized 8-bit in groups of 64 and one quantized 4-bit in
    /// groups of 128 pack to exactly the same u32 count against exactly the
    /// same number of scales. Assuming one of them is how an 8-bit llama
    /// checkpoint was refused as "g128/b4" -- a quantization it does not have
    /// and nobody ships.
    ///
    /// Zero means the config declared none, which is a checkpoint whose
    /// tensors are dense.
    pub quant_bits: i32,
    /// The quantization group size. See [`Self::quant_bits`].
    pub quant_group_size: i32,

    /// Vocabulary size, family-independent.
    pub vocab_size: u32,
    /// `max_position_embeddings`, the context ceiling this build serves.
    pub max_model_len: u32,
    /// The lowercased architecture stem. See [`arch_stem`].
    pub arch_name: String,
    /// Whether any layer is gated-delta-net rather than attention.
    pub has_linear_attn: bool,
    /// The flat rope base.
    ///
    /// Qwen3.5/3.6 carry the RoPE hyperparameters in a nested
    /// `rope_parameters` object rather than at the top level, so a reader
    /// that only knows the flat key finds nothing and silently keeps its
    /// default. Nothing fails: the rotated channels just come out wrong, and
    /// the error compounds layer over layer until the activations saturate.
    pub rope_theta: f32,

    /// GPT-OSS layer count. Non-zero marks "this config was read as gpt-oss".
    pub go_num_hidden_layers: i32,
    /// GPT-OSS hidden size.
    pub go_hidden_size: i32,
    /// GPT-OSS vocabulary size.
    pub go_vocab_size: i32,
    /// GPT-OSS attention head count.
    pub go_num_attention_heads: i32,
    /// GPT-OSS key/value head count.
    pub go_num_key_value_heads: i32,
    /// GPT-OSS per-head dimension.
    pub go_head_dim: i32,
    /// GPT-OSS routed expert count.
    pub go_num_local_experts: i32,
    /// GPT-OSS experts per token.
    pub go_num_experts_per_tok: i32,
    /// GPT-OSS FFN width.
    pub go_intermediate_size: i32,
    /// GPT-OSS RMSNorm epsilon.
    pub go_rms_norm_eps: f32,
    /// GPT-OSS SwiGLU clamp.
    pub go_swiglu_limit: f32,

    /// Gemma 4 layer count. Non-zero marks "this config was read as gemma4".
    pub g4_num_hidden_layers: i32,
    /// Gemma 4 hidden size.
    pub g4_hidden_size: i32,
    /// Gemma 4 FFN width.
    pub g4_intermediate_size: i32,
    /// Gemma 4 attention head count.
    pub g4_num_attention_heads: i32,
    /// Gemma 4 key/value head count.
    pub g4_num_key_value_heads: i32,
    /// Gemma 4 per-head dimension for the SLIDING layers.
    pub g4_head_dim: i32,
    /// Gemma 4 per-head dimension for the FULL layers.
    pub g4_global_head_dim: i32,
    /// Gemma 4 sliding window. Zero means the config omitted one.
    pub g4_sliding_window: i32,
    /// Gemma 4 layers that share their neighbour's KV.
    pub g4_num_kv_shared_layers: i32,
    /// Gemma 4 `hidden_size_per_layer_input`.
    pub g4_per_layer_emb_dim: i32,
    /// One full-attention layer every `interval`, derived from `layer_types`;
    /// -1 for an irregular pattern the geometry refuses rather than
    /// approximates.
    pub g4_full_attn_interval: i32,
    /// Gemma 4 final logit softcap.
    pub g4_final_softcap: f32,
    /// Gemma 4 rope base on the full-attention layers.
    ///
    /// Zero until `gemma_per_layer_rope_theta` states one, which is what
    /// makes it answerable: it was 1e6, and a gemma-4 config carrying a flat
    /// `rope_theta` and no per-layer array had its stated base DISCARDED for
    /// that constant, because the geometry chose the base by asking whether
    /// the gemma4 block had been read rather than whether a per-layer base
    /// had been stated. A wrong base does not fail, it rotates by a factor
    /// that grows with position, so a short prompt agrees and a long one
    /// does not.
    pub g4_rope_theta_full: f32,
    /// Gemma 4 rope base on the sliding layers.
    ///
    /// Zero for a stack that states one base everywhere, which is what
    /// `rope_theta_at` already reads zero as. Same correction as
    /// [`Self::g4_rope_theta_full`]: the old 1e4 was a value no config had
    /// asked for.
    pub g4_rope_theta_sliding: f32,
    /// Gemma 4 partial rotary factor on the full-attention layers.
    pub g4_full_partial_rotary: f32,
    /// Gemma 4 routed expert count.
    pub g4_num_experts: i32,
    /// Gemma 4 experts per token.
    pub g4_experts_per_token: i32,
    /// Gemma 4 expert FFN width.
    pub g4_moe_intermediate: i32,
    /// Gemma 4's k-equals-V attention.
    pub g4_attention_k_eq_v: bool,
    /// Gemma 4 key/value head count on the full-attention layers.
    pub g4_num_global_kv_heads: i32,

    /// Llama-family layer count. Non-zero marks "this config was read as one
    /// of `llama`, `llama3`, `mistral`, `qwen2`, `qwen3` and the two MoE
    /// variants" -- one set of fields, because they differ in VALUES and not
    /// in shape.
    pub ll_num_hidden_layers: i32,
    /// Llama-family hidden size.
    pub ll_hidden_size: i32,
    /// Llama-family vocabulary size.
    pub ll_vocab_size: i32,
    /// Llama-family attention head count.
    pub ll_num_attention_heads: i32,
    /// Llama-family key/value head count.
    pub ll_num_key_value_heads: i32,
    /// Llama-family per-head dimension.
    pub ll_head_dim: i32,
    /// Llama-family FFN width.
    pub ll_intermediate_size: i32,
    /// Llama-family routed expert count.
    pub ll_num_experts: i32,
    /// Llama-family experts per token.
    pub ll_num_experts_per_tok: i32,
    /// Llama-family expert FFN width.
    pub ll_moe_intermediate_size: i32,
    /// Llama-family RMSNorm epsilon.
    pub ll_rms_norm_eps: f32,
    /// Llama-family rope scale (`factor`), shared by the linear and llama3
    /// schedules.
    pub ll_rope_scale: f32,
    /// The rope schedule in HF's spelling: `llama3`, `yarn`, or empty for the
    /// plain geometric series.
    pub ll_rope_scaling_kind: String,
    /// Llama 3.1's piecewise schedule: the low-frequency factor. The default
    /// is HF's, so a config that names the type but omits a knob gets the
    /// schedule everyone else gets rather than a division by zero.
    pub ll_rope_low_freq_factor: f32,
    /// Llama 3.1's high-frequency factor. See
    /// [`Self::ll_rope_low_freq_factor`].
    pub ll_rope_high_freq_factor: f32,
    /// Llama 3.1's original context length.
    pub ll_rope_original_max_position: i32,
    /// Whether the unembedding reuses the embedding matrix.
    pub ll_tied_embeddings: bool,
    /// Whether router probabilities are renormalized over the top-k.
    pub ll_norm_topk_prob: bool,

    /// Qwen3.5 / Qwen3-Next layer count. Non-zero marks "this config was read
    /// as the GDN hybrid". Its shape used to be a struct full of defaults
    /// compiled into the driver, which meant the family ran exactly one
    /// checkpoint and mis-ran any other silently.
    pub q35_num_hidden_layers: i32,
    /// Qwen3.5 hidden size.
    pub q35_hidden_size: i32,
    /// Qwen3.5 vocabulary size.
    pub q35_vocab_size: i32,
    /// Qwen3.5 attention head count.
    pub q35_num_attention_heads: i32,
    /// Qwen3.5 key/value head count.
    pub q35_num_key_value_heads: i32,
    /// Qwen3.5 per-head dimension.
    pub q35_head_dim: i32,
    /// Qwen3.5 FFN width.
    pub q35_intermediate_size: i32,
    /// Linear-attention key heads. `conv_dim` and the value total are DERIVED
    /// from these rather than read, because a config cannot state them
    /// inconsistently with the head counts and this driver should not be able
    /// to either.
    pub q35_linear_key_heads: i32,
    /// Linear-attention value heads. See [`Self::q35_linear_key_heads`].
    pub q35_linear_value_heads: i32,
    /// Linear-attention key head dimension.
    pub q35_linear_key_head_dim: i32,
    /// Linear-attention value head dimension.
    pub q35_linear_value_head_dim: i32,
    /// Linear-attention short-convolution kernel width.
    pub q35_linear_conv_kernel: i32,
    /// One full-attention layer every `interval`; -1 for an irregular pattern
    /// the geometry refuses rather than approximates.
    pub q35_full_attn_interval: i32,
    /// Qwen3.5 routed expert count.
    pub q35_num_experts: i32,
    /// Qwen3.5 experts per token.
    pub q35_num_experts_per_tok: i32,
    /// Qwen3.5 expert FFN width.
    pub q35_moe_intermediate_size: i32,
    /// Qwen3.5 shared-expert FFN width.
    pub q35_shared_expert_intermediate: i32,
    /// Every `step`-th layer is routed. No descriptor spelling, so the
    /// default stands.
    pub q35_decoder_sparse_step: i32,
    /// Layers the config exempts from routing. Carried as a COUNT because the
    /// geometry refuses any non-empty one; the number is for the message.
    pub q35_mlp_only_layer_count: i32,
    /// Qwen3.5 RMSNorm epsilon.
    pub q35_rms_norm_eps: f32,
    /// Whether the Qwen3.5 unembedding reuses the embedding matrix.
    pub q35_tied_embeddings: bool,
    /// Whether Qwen3.5 router probabilities are renormalized over the top-k.
    pub q35_norm_topk_prob: bool,

    /// Which storage schema this driver authors against.
    pub model_type: String,
}

impl Default for ModelFacts {
    fn default() -> Self {
        Self {
            quant_bits: 0,
            quant_group_size: 0,

            vocab_size: 32000,
            max_model_len: 8192,
            arch_name: "llama".to_string(),
            has_linear_attn: false,
            rope_theta: 1.0e7,

            go_num_hidden_layers: 0,
            go_hidden_size: 0,
            go_vocab_size: 0,
            go_num_attention_heads: 0,
            go_num_key_value_heads: 0,
            go_head_dim: 0,
            go_num_local_experts: 0,
            go_num_experts_per_tok: 0,
            go_intermediate_size: 0,
            go_rms_norm_eps: 1e-5,
            go_swiglu_limit: 7.0,

            g4_num_hidden_layers: 0,
            g4_hidden_size: 0,
            g4_intermediate_size: 0,
            g4_num_attention_heads: 0,
            g4_num_key_value_heads: 0,
            g4_head_dim: 0,
            g4_global_head_dim: 0,
            g4_sliding_window: 0,
            g4_num_kv_shared_layers: 0,
            g4_per_layer_emb_dim: 0,
            g4_full_attn_interval: 0,
            g4_final_softcap: 0.0,
            g4_rope_theta_full: 0.0,
            g4_rope_theta_sliding: 0.0,
            g4_full_partial_rotary: 0.25,
            g4_num_experts: 0,
            g4_experts_per_token: 0,
            g4_moe_intermediate: 0,
            g4_attention_k_eq_v: false,
            g4_num_global_kv_heads: 0,

            ll_num_hidden_layers: 0,
            ll_hidden_size: 0,
            ll_vocab_size: 0,
            ll_num_attention_heads: 0,
            ll_num_key_value_heads: 0,
            ll_head_dim: 0,
            ll_intermediate_size: 0,
            ll_num_experts: 0,
            ll_num_experts_per_tok: 0,
            ll_moe_intermediate_size: 0,
            ll_rms_norm_eps: 1e-5,
            ll_rope_scale: 1.0,
            ll_rope_scaling_kind: String::new(),
            ll_rope_low_freq_factor: 1.0,
            ll_rope_high_freq_factor: 4.0,
            ll_rope_original_max_position: 8192,
            ll_tied_embeddings: true,
            ll_norm_topk_prob: true,

            q35_num_hidden_layers: 0,
            q35_hidden_size: 0,
            q35_vocab_size: 0,
            q35_num_attention_heads: 0,
            q35_num_key_value_heads: 0,
            q35_head_dim: 0,
            q35_intermediate_size: 0,
            q35_linear_key_heads: 0,
            q35_linear_value_heads: 0,
            q35_linear_key_head_dim: 0,
            q35_linear_value_head_dim: 0,
            q35_linear_conv_kernel: 0,
            q35_full_attn_interval: 0,
            q35_num_experts: 0,
            q35_num_experts_per_tok: 0,
            q35_moe_intermediate_size: 0,
            q35_shared_expert_intermediate: 0,
            q35_decoder_sparse_step: 1,
            q35_mlp_only_layer_count: 0,
            q35_rms_norm_eps: 1e-6,
            q35_tied_embeddings: true,
            q35_norm_topk_prob: true,

            model_type: String::new(),
        }
    }
}

impl ModelFacts {
    /// The facts, read out of the `pie.model/1` descriptor every boot is
    /// handed.
    ///
    /// Takes the document, not a path to it: the caller has already read the
    /// file because the compile request carries the bytes, and opening it a
    /// second time here would be a second chance to read something else.
    ///
    /// `None` means the document is empty, unparseable, or of a version this
    /// build does not read. That is a refusal at the call site, not a signal
    /// to derive the facts another way: there is no other way left.
    #[must_use]
    pub fn from_descriptor(json: &str) -> Option<Self> {
        if json.is_empty() {
            return None;
        }
        let j: Value = serde_json::from_str(json).ok()?;
        if string_or(&j, "version", "") != "pie.model/1" {
            return None;
        }

        let mut facts = Self {
            model_type: string_or(&j, "model_type", ""),
            ..Self::default()
        };
        u32_of(&j, "vocab_size", &mut facts.vocab_size);
        u32_of(&j, "max_position_embeddings", &mut facts.max_model_len);
        f32_of(&j, "rope_theta", &mut facts.rope_theta);
        // The affine width and group. Leaving them zero is not a smaller set
        // of facts, it is a *wrong* one: every geometry defaults to `{4, 64}`
        // and only overrides when these are non-zero, so an 8-bit checkpoint
        // served as an artifact would be decoded by 4-bit kernels. The C++
        // reads this pair TWICE, in two adjacent blocks with two different
        // comments; the second read is redundant, not meaningful, so it is
        // read once here.
        i32_of(&j, "quant_bits", &mut facts.quant_bits);
        i32_of(&j, "quant_group_size", &mut facts.quant_group_size);

        // `arch_name` is `architectures[0]` verbatim in the descriptor; this
        // driver keys on the lowercased stem, so apply the same reduction --
        // the SAME function, because two copies of the rule is how one of them
        // came to strip only `forcausallm`.
        let arch = arch_stem(&string_or(&j, "arch_name", ""));
        if !arch.is_empty() {
            facts.arch_name = arch;
        }

        if i64_or(&j, "linear_num_value_heads", 0) > 0 {
            facts.has_linear_attn = true;
        }
        if let Some(types) = array_of(&j, "layer_types")
            && types.iter().any(|t| t.as_str() == Some("linear_attention"))
        {
            facts.has_linear_attn = true;
        }

        if facts.model_type == "gpt_oss" {
            i32_of(&j, "num_hidden_layers", &mut facts.go_num_hidden_layers);
            i32_of(&j, "hidden_size", &mut facts.go_hidden_size);
            i32_of(&j, "vocab_size", &mut facts.go_vocab_size);
            i32_of(&j, "num_attention_heads", &mut facts.go_num_attention_heads);
            i32_of(&j, "num_key_value_heads", &mut facts.go_num_key_value_heads);
            i32_of(&j, "head_dim", &mut facts.go_head_dim);
            // The descriptor folds `num_local_experts` / `num_experts` /
            // `n_routed_experts` into one field at import.
            i32_of(&j, "num_experts", &mut facts.go_num_local_experts);
            i32_of(&j, "num_experts_per_tok", &mut facts.go_num_experts_per_tok);
            i32_of(&j, "intermediate_size", &mut facts.go_intermediate_size);
            f32_of(&j, "rms_norm_eps", &mut facts.go_rms_norm_eps);
            f32_of(&j, "swiglu_limit", &mut facts.go_swiglu_limit);
        }

        // Read only when `model_type` names one of the llama-shaped families,
        // so a config that never mentions them cannot accidentally select
        // one. Without this branch every llama-shaped artifact reached the
        // geometry with `ll_num_hidden_layers == 0` and was refused as
        // "config carried no decoder shape" -- from a descriptor that claimed
        // to carry a normalized config and then carried no decoder at all.
        if ModelFamily::of(&facts.model_type) == ModelFamily::Llama {
            i32_of(&j, "num_hidden_layers", &mut facts.ll_num_hidden_layers);
            i32_of(&j, "hidden_size", &mut facts.ll_hidden_size);
            i32_of(&j, "vocab_size", &mut facts.ll_vocab_size);
            i32_of(&j, "num_attention_heads", &mut facts.ll_num_attention_heads);
            i32_of(&j, "num_key_value_heads", &mut facts.ll_num_key_value_heads);
            i32_of(&j, "head_dim", &mut facts.ll_head_dim);
            i32_of(&j, "intermediate_size", &mut facts.ll_intermediate_size);
            i32_of(&j, "num_experts", &mut facts.ll_num_experts);
            i32_of(&j, "num_experts_per_tok", &mut facts.ll_num_experts_per_tok);
            i32_of(
                &j,
                "moe_intermediate_size",
                &mut facts.ll_moe_intermediate_size,
            );
            f32_of(&j, "rms_norm_eps", &mut facts.ll_rms_norm_eps);
            facts.ll_norm_topk_prob = bool_or(&j, "norm_topk_prob", facts.ll_norm_topk_prob);
            facts.ll_tied_embeddings = bool_or(&j, "tie_word_embeddings", facts.ll_tied_embeddings);
            // Import resolves `rope_scaling` into a kind plus already-
            // defaulted knobs; the driver keys on HF's spelling, so translate
            // back. `none` stays empty, which is the plain geometric series
            // the geometry runs for an absent scaling. `original_yarn` becomes
            // `yarn` so a refusal names the schedule the config asked for.
            match string_or(&j, "rope_scaling_kind", "").as_str() {
                "llama3" => facts.ll_rope_scaling_kind = "llama3".to_string(),
                "original_yarn" => facts.ll_rope_scaling_kind = "yarn".to_string(),
                _ => {}
            }
            // `rope_factor` is read whatever the kind. A linear scaling
            // normalizes to kind `none` plus a factor -- it IS the plain
            // geometric series, just with positions divided -- so gating this
            // read on a non-empty kind would drop the factor.
            f32_of(&j, "rope_factor", &mut facts.ll_rope_scale);
            if !facts.ll_rope_scaling_kind.is_empty() {
                f32_of(
                    &j,
                    "rope_low_freq_factor",
                    &mut facts.ll_rope_low_freq_factor,
                );
                f32_of(
                    &j,
                    "rope_high_freq_factor",
                    &mut facts.ll_rope_high_freq_factor,
                );
                i32_of(
                    &j,
                    "rope_original_max_position",
                    &mut facts.ll_rope_original_max_position,
                );
            }
        }

        // The GDN hybrid, and the reason a Qwen3.5 checkpoint imported through
        // `pie model import` could not boot on Metal: the descriptor path had
        // no branch for this family at all, so the geometry saw
        // `q35_num_hidden_layers == 0` and refused the model as carrying no
        // decoder shape -- from a descriptor that carried the whole decoder.
        if ModelFamily::of(&facts.model_type) == ModelFamily::Qwen35 {
            i32_of(&j, "num_hidden_layers", &mut facts.q35_num_hidden_layers);
            i32_of(&j, "hidden_size", &mut facts.q35_hidden_size);
            i32_of(&j, "vocab_size", &mut facts.q35_vocab_size);
            i32_of(
                &j,
                "num_attention_heads",
                &mut facts.q35_num_attention_heads,
            );
            i32_of(
                &j,
                "num_key_value_heads",
                &mut facts.q35_num_key_value_heads,
            );
            i32_of(&j, "head_dim", &mut facts.q35_head_dim);
            i32_of(&j, "intermediate_size", &mut facts.q35_intermediate_size);
            i32_of(&j, "linear_num_key_heads", &mut facts.q35_linear_key_heads);
            i32_of(
                &j,
                "linear_num_value_heads",
                &mut facts.q35_linear_value_heads,
            );
            i32_of(
                &j,
                "linear_key_head_dim",
                &mut facts.q35_linear_key_head_dim,
            );
            i32_of(
                &j,
                "linear_value_head_dim",
                &mut facts.q35_linear_value_head_dim,
            );
            i32_of(
                &j,
                "linear_conv_kernel_dim",
                &mut facts.q35_linear_conv_kernel,
            );
            i32_of(&j, "num_experts", &mut facts.q35_num_experts);
            i32_of(
                &j,
                "num_experts_per_tok",
                &mut facts.q35_num_experts_per_tok,
            );
            i32_of(
                &j,
                "moe_intermediate_size",
                &mut facts.q35_moe_intermediate_size,
            );
            i32_of(
                &j,
                "shared_expert_intermediate_size",
                &mut facts.q35_shared_expert_intermediate,
            );
            f32_of(&j, "rms_norm_eps", &mut facts.q35_rms_norm_eps);
            facts.q35_norm_topk_prob = bool_or(&j, "norm_topk_prob", facts.q35_norm_topk_prob);
            // The multimodal `text_config` split the `config.json` path has to
            // reconcile does not exist here: import already resolved the text
            // decoder's own statement into one top-level field.
            facts.q35_tied_embeddings =
                bool_or(&j, "tie_word_embeddings", facts.q35_tied_embeddings);
            // `full_attention_interval` is not a descriptor field -- import
            // expands the schedule into `layer_types`, one entry per layer.
            // Reduce it including the -1 for an irregular stack: rounding that
            // to a regular interval would put full attention on layers that
            // are linear.
            if let Some(types) = array_of(&j, "layer_types")
                && let Some(interval) = full_attention_interval(types)
            {
                facts.q35_full_attn_interval = interval;
            }
            // `decoder_sparse_step` and `mlp_only_layers` have no descriptor
            // spelling; their defaults stand. Both describe which layers are
            // dense in a routed stack, so they are inert for the dense
            // releases and are the known gap for a MoE one imported through
            // the descriptor.
        }

        if facts.model_type == "gemma4" || facts.model_type == "gemma4_text" {
            i32_of(&j, "num_hidden_layers", &mut facts.g4_num_hidden_layers);
            i32_of(&j, "hidden_size", &mut facts.g4_hidden_size);
            i32_of(&j, "intermediate_size", &mut facts.g4_intermediate_size);
            i32_of(&j, "num_attention_heads", &mut facts.g4_num_attention_heads);
            i32_of(&j, "num_key_value_heads", &mut facts.g4_num_key_value_heads);
            i32_of(&j, "head_dim", &mut facts.g4_head_dim);
            i32_of(&j, "gemma4_global_head_dim", &mut facts.g4_global_head_dim);
            // `sliding_window` is -1 in the descriptor when the config omits
            // it, where reading `config.json` would leave this 0. Guard so
            // "absent" means the same thing on both paths.
            if i64_or(&j, "sliding_window", -1) > 0 {
                i32_of(&j, "sliding_window", &mut facts.g4_sliding_window);
            }
            i32_of(
                &j,
                "num_kv_shared_layers",
                &mut facts.g4_num_kv_shared_layers,
            );
            i32_of(
                &j,
                "gemma_hidden_size_per_layer_input",
                &mut facts.g4_per_layer_emb_dim,
            );
            facts.g4_attention_k_eq_v = bool_or(&j, "gemma4_attention_k_eq_v", false);
            i32_of(&j, "num_experts", &mut facts.g4_num_experts);
            i32_of(&j, "num_experts_per_tok", &mut facts.g4_experts_per_token);
            i32_of(&j, "moe_intermediate_size", &mut facts.g4_moe_intermediate);
            i32_of(
                &j,
                "gemma4_num_global_key_value_heads",
                &mut facts.g4_num_global_kv_heads,
            );
            f32_of(&j, "gemma_final_logit_softcap", &mut facts.g4_final_softcap);

            // The descriptor expands the per-attention-type rope into one
            // entry per layer at import, so the per-type values come back by
            // looking at the first layer of each type rather than by
            // re-reading nested JSON.
            if let Some(types) = array_of(&j, "layer_types") {
                // Each type's values come from the *first* layer of that type,
                // and the two are tracked independently: keying the sliding
                // read off "no full layer yet" would miss a stack that opens
                // with a full-attention layer.
                let mut saw_full = false;
                let mut saw_sliding = false;
                for (i, t) in types.iter().enumerate() {
                    match t.as_str() {
                        Some("full_attention") if !saw_full => {
                            at_f32(
                                &j,
                                "gemma_per_layer_rope_theta",
                                i,
                                &mut facts.g4_rope_theta_full,
                            );
                            at_f32(
                                &j,
                                "gemma_per_layer_partial_rotary_factor",
                                i,
                                &mut facts.g4_full_partial_rotary,
                            );
                            saw_full = true;
                        }
                        Some("sliding_attention") if !saw_sliding => {
                            at_f32(
                                &j,
                                "gemma_per_layer_rope_theta",
                                i,
                                &mut facts.g4_rope_theta_sliding,
                            );
                            saw_sliding = true;
                        }
                        _ => {}
                    }
                }
                // The same schedule check the Qwen3.5 block makes: the
                // interval is the distance between the first two full layers,
                // and an irregular stack is refused (-1) rather than silently
                // mis-scheduled.
                if let Some(interval) = full_attention_interval(types) {
                    facts.g4_full_attn_interval = interval;
                }
            }
        }
        Some(facts)
    }
}

/// The full-attention period of an expanded `layer_types` schedule.
///
/// `None` when the stack has no full-attention layer at all, which leaves the
/// caller's field at its default rather than claiming an interval of zero.
/// `Some(-1)` for an irregular schedule: the geometry refuses that rather than
/// rounding it, because rounding puts full attention on a layer that is not.
fn full_attention_interval(types: &[Value]) -> Option<i32> {
    let full: Vec<i32> = types
        .iter()
        .enumerate()
        .filter(|(_, t)| t.as_str() == Some("full_attention"))
        .map(|(i, _)| i as i32)
        .collect();
    let first = *full.first()?;
    let interval = first + 1;
    let regular = full
        .iter()
        .enumerate()
        .all(|(k, &at)| at == (k as i32 + 1) * interval - 1);
    Some(if regular { interval } else { -1 })
}

/// The integer at `key`, or `None` when the key is absent or is not an
/// integer. A float, a string and a null all read as absent, which is what
/// keeps a mistyped descriptor from replacing a documented default with zero.
fn int_at(j: &Value, key: &str) -> Option<i64> {
    let v = j.get(key)?;
    if v.is_i64() || v.is_u64() {
        v.as_i64().or_else(|| v.as_u64().map(|n| n as i64))
    } else {
        None
    }
}

/// Assign `out` only if `key` is present and an integer.
fn i32_of(j: &Value, key: &str, out: &mut i32) {
    if let Some(n) = int_at(j, key) {
        *out = n as i32;
    }
}

/// Assign `out` only if `key` is present and an integer.
fn u32_of(j: &Value, key: &str, out: &mut u32) {
    if let Some(n) = int_at(j, key) {
        *out = n as u32;
    }
}

/// Assign `out` only if `key` is present and a number of any kind. Unlike the
/// integer readers this accepts `1e6` written as an integer, because every
/// rope base in the wild is spelled both ways.
fn f32_of(j: &Value, key: &str, out: &mut f32) {
    if let Some(n) = j.get(key).and_then(Value::as_f64) {
        *out = n as f32;
    }
}

/// Element `i` of the array at `key` as a float, when both exist and the
/// element is a number.
fn at_f32(j: &Value, key: &str, i: usize, out: &mut f32) {
    if let Some(n) = j
        .get(key)
        .and_then(Value::as_array)
        .and_then(|a| a.get(i))
        .and_then(Value::as_f64)
    {
        *out = n as f32;
    }
}

/// The array at `key`, or `None` when the key is absent or holds something
/// else.
fn array_of<'a>(j: &'a Value, key: &str) -> Option<&'a Vec<Value>> {
    j.get(key).and_then(Value::as_array)
}

/// The integer at `key`, falling back when it is absent or of another type --
/// the C++ `j.value(key, fallback)`.
fn i64_or(j: &Value, key: &str, fallback: i64) -> i64 {
    int_at(j, key).unwrap_or(fallback)
}

/// The bool at `key`, falling back when it is absent or of another type.
fn bool_or(j: &Value, key: &str, fallback: bool) -> bool {
    j.get(key).and_then(Value::as_bool).unwrap_or(fallback)
}

/// The string at `key`, falling back when it is absent or of another type.
fn string_or(j: &Value, key: &str, fallback: &str) -> String {
    j.get(key)
        .and_then(Value::as_str)
        .unwrap_or(fallback)
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::{ModelFacts, ModelFamily, arch_stem};
    use serde_json::json;

    fn descriptor(model_type: &str) -> serde_json::Value {
        json!({ "version": "pie.model/1", "model_type": model_type })
    }

    #[test]
    fn an_empty_document_is_a_refusal_not_a_default() {
        assert!(ModelFacts::from_descriptor("").is_none());
    }

    #[test]
    fn a_malformed_document_is_a_refusal() {
        assert!(ModelFacts::from_descriptor("{\"version\":").is_none());
        assert!(ModelFacts::from_descriptor("not json at all").is_none());
    }

    #[test]
    fn a_document_of_another_version_is_a_refusal() {
        assert!(
            ModelFacts::from_descriptor(&json!({"version": "pie.model/2"}).to_string()).is_none()
        );
        assert!(ModelFacts::from_descriptor(&json!({"model_type": "llama"}).to_string()).is_none());
        // A version key of the wrong type reads as absent, so it refuses too.
        assert!(ModelFacts::from_descriptor(&json!({"version": 1}).to_string()).is_none());
    }

    #[test]
    fn a_minimal_descriptor_keeps_every_documented_default() {
        let facts =
            ModelFacts::from_descriptor(&json!({"version": "pie.model/1"}).to_string()).unwrap();
        assert_eq!(facts, ModelFacts::default());
        assert_eq!(facts.vocab_size, 32000);
        assert_eq!(facts.max_model_len, 8192);
        assert_eq!(facts.arch_name, "llama");
        assert!((facts.rope_theta - 1.0e7).abs() < f32::EPSILON);
        // One base, so one default. There were three — 1e7 here, 500000 on
        // the llama block and 150000 on the gpt-oss one, all three read from
        // the same `rope_theta` key and told apart by a family name at the
        // point of use. Which one a descriptor got was decided by its
        // `model_type` and mattered only when the key was absent, which is a
        // case the descriptor contract says cannot arise.
        assert!(facts.ll_tied_embeddings);
        assert_eq!(facts.q35_decoder_sparse_step, 1);
        assert!((facts.q35_rms_norm_eps - 1e-6).abs() < f32::EPSILON);
    }

    #[test]
    fn a_task_suffix_comes_off_the_architecture_name() {
        assert_eq!(arch_stem("Gemma4ForConditionalGeneration"), "gemma4");
        assert_eq!(arch_stem("LlamaForCausalLM"), "llama");
        assert_eq!(arch_stem("Qwen3MoeForCausalLM"), "qwen3moe");
    }

    #[test]
    fn reformer_keeps_its_own_for_and_does_not_become_re() {
        assert_eq!(arch_stem("ReformerForCausalLM"), "reformer");
    }

    #[test]
    fn a_name_with_no_task_suffix_is_only_lowercased() {
        assert_eq!(arch_stem("Gemma4TextModel"), "gemma4textmodel");
        assert_eq!(arch_stem(""), "");
        // The suffix must be a proper suffix: a name that IS one stays whole.
        assert_eq!(arch_stem("ForCausalLM"), "forcausallm");
    }

    #[test]
    fn every_family_has_at_least_one_member_and_a_stranger_has_none() {
        assert_eq!(ModelFamily::of("qwen3_next"), ModelFamily::Qwen35);
        assert_eq!(ModelFamily::of("qwen3_6"), ModelFamily::Qwen35);
        assert_eq!(ModelFamily::of("gemma4_text"), ModelFamily::Gemma4);
        assert_eq!(ModelFamily::of("gpt_oss"), ModelFamily::GptOss);
        assert_eq!(ModelFamily::of("mistral"), ModelFamily::Llama);
        assert_eq!(ModelFamily::of("qwen2_moe"), ModelFamily::Llama);
        assert_eq!(ModelFamily::of("phi3"), ModelFamily::Unknown);
        assert_eq!(ModelFamily::of(""), ModelFamily::Unknown);
        assert!(ModelFamily::is_supported("llama3"));
        assert!(!ModelFamily::is_supported("qwen3_5_vision"));
    }

    #[test]
    fn a_key_of_the_wrong_type_leaves_the_default_standing() {
        let doc = json!({
            "version": "pie.model/1",
            "model_type": "llama",
            "vocab_size": "128256",
            "max_position_embeddings": null,
            "rope_theta": "500000",
            "num_hidden_layers": 32.5,
            "tie_word_embeddings": "false",
            "arch_name": 7,
        });
        let facts = ModelFacts::from_descriptor(&doc.to_string()).unwrap();
        assert_eq!(facts.vocab_size, 32000);
        assert_eq!(facts.max_model_len, 8192);
        assert!((facts.rope_theta - 1.0e7).abs() < f32::EPSILON);
        assert_eq!(facts.ll_num_hidden_layers, 0);
        assert!(facts.ll_tied_embeddings);
        assert_eq!(facts.arch_name, "llama");
    }

    #[test]
    fn the_llama_block_is_not_read_for_a_gpt_oss_model_type() {
        let mut doc = descriptor("gpt_oss");
        doc["num_hidden_layers"] = json!(24);
        doc["hidden_size"] = json!(2880);
        let facts = ModelFacts::from_descriptor(&doc.to_string()).unwrap();
        assert_eq!(facts.go_num_hidden_layers, 24);
        assert_eq!(facts.go_hidden_size, 2880);
        assert_eq!(facts.ll_num_hidden_layers, 0);
        assert_eq!(facts.q35_num_hidden_layers, 0);
        assert_eq!(facts.g4_num_hidden_layers, 0);
    }

    #[test]
    fn the_gpt_oss_block_is_not_read_for_a_llama_model_type() {
        let mut doc = descriptor("mistral");
        doc["num_hidden_layers"] = json!(32);
        doc["rope_theta"] = json!(1_000_000.0);
        let facts = ModelFacts::from_descriptor(&doc.to_string()).unwrap();
        assert_eq!(facts.ll_num_hidden_layers, 32);
        assert_eq!(facts.go_num_hidden_layers, 0);
        // The base is read ONCE, family-free, so there is no longer a
        // "the other family's theta" to check for contamination. The two
        // assertions that were here — that a llama config leaves
        // `go_rope_theta` at 150000 and fills `ll_rope_theta` with the stated
        // value — were pinning the fact that the same key was read into three
        // fields.
        assert!((facts.rope_theta - 1_000_000.0).abs() < f32::EPSILON);
    }

    // `qk_norm_is_on_for_qwen3_moe_even_without_the_explicit_flag` was here.
    //
    // It asserted `ll_qk_norm` — that `use_qk_norm`, ORed with
    // `model_type == "qwen3" || "qwen3_moe"`, came out true. The
    // REQUIREMENT it encoded is real and still holds; the field it read
    // was not, because nothing anywhere consumed `ll_qk_norm`.
    //
    // What decides qk-norm is `model::text`, and it asks the TENSORS:
    // `has_tensor("layers.0.self_attn.q_norm.weight")`. That is strictly
    // the better question — a checkpoint that ships the weight gets the
    // norm whatever its `model_type` says, and the OR-in-the-family-name
    // this test was written to pin exists precisely because the config
    // key could not be trusted. `model/text.rs` covers both directions
    // (`split.qk_norm == Off`, `fused.qk_norm == PerHead`).
    //
    // So this is a test of a dead parallel implementation of a rule that
    // is met, better, elsewhere. Deleting it removes no coverage.

    #[test]
    fn the_rope_kind_is_translated_and_the_factor_is_read_whatever_it_is() {
        let mut doc = descriptor("llama3");
        doc["rope_scaling_kind"] = json!("none");
        doc["rope_factor"] = json!(8.0);
        doc["rope_low_freq_factor"] = json!(2.0);
        let facts = ModelFacts::from_descriptor(&doc.to_string()).unwrap();
        assert_eq!(facts.ll_rope_scaling_kind, "");
        assert!((facts.ll_rope_scale - 8.0).abs() < f32::EPSILON);
        // The llama3 knobs are gated on a non-empty kind.
        assert!((facts.ll_rope_low_freq_factor - 1.0).abs() < f32::EPSILON);

        doc["rope_scaling_kind"] = json!("llama3");
        let facts = ModelFacts::from_descriptor(&doc.to_string()).unwrap();
        assert_eq!(facts.ll_rope_scaling_kind, "llama3");
        assert!((facts.ll_rope_low_freq_factor - 2.0).abs() < f32::EPSILON);

        doc["rope_scaling_kind"] = json!("original_yarn");
        let facts = ModelFacts::from_descriptor(&doc.to_string()).unwrap();
        assert_eq!(facts.ll_rope_scaling_kind, "yarn");
    }

    #[test]
    fn a_regular_schedule_yields_its_interval() {
        let mut doc = descriptor("qwen3_next");
        doc["layer_types"] = json!([
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention"
        ]);
        let facts = ModelFacts::from_descriptor(&doc.to_string()).unwrap();
        assert_eq!(facts.q35_full_attn_interval, 4);
        assert!(facts.has_linear_attn);
    }

    #[test]
    fn an_irregular_schedule_is_refused_rather_than_rounded() {
        let mut doc = descriptor("qwen3_next");
        doc["layer_types"] = json!([
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
            "full_attention"
        ]);
        let facts = ModelFacts::from_descriptor(&doc.to_string()).unwrap();
        assert_eq!(facts.q35_full_attn_interval, -1);
    }

    #[test]
    fn a_stack_with_no_full_attention_layer_claims_no_interval() {
        let mut doc = descriptor("qwen3_next");
        doc["layer_types"] = json!(["linear_attention", "linear_attention"]);
        let facts = ModelFacts::from_descriptor(&doc.to_string()).unwrap();
        assert_eq!(facts.q35_full_attn_interval, 0);
    }

    #[test]
    fn linear_attention_is_seen_by_head_count_as_well_as_by_schedule() {
        let mut doc = descriptor("llama");
        doc["linear_num_value_heads"] = json!(32);
        let facts = ModelFacts::from_descriptor(&doc.to_string()).unwrap();
        assert!(facts.has_linear_attn);

        let mut doc = descriptor("llama");
        doc["linear_num_value_heads"] = json!(0);
        let facts = ModelFacts::from_descriptor(&doc.to_string()).unwrap();
        assert!(!facts.has_linear_attn);
    }

    #[test]
    fn gemma4_takes_each_ropes_value_from_the_first_layer_of_its_type() {
        let mut doc = descriptor("gemma4");
        doc["layer_types"] = json!([
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention"
        ]);
        doc["gemma_per_layer_rope_theta"] = json!([
            10_000.0,
            20_000.0,
            1_000_000.0,
            30_000.0,
            40_000.0,
            2_000_000.0
        ]);
        doc["gemma_per_layer_partial_rotary_factor"] = json!([0.5, 0.5, 0.75, 0.5, 0.5, 0.9]);
        let facts = ModelFacts::from_descriptor(&doc.to_string()).unwrap();
        assert!((facts.g4_rope_theta_sliding - 10_000.0).abs() < f32::EPSILON);
        assert!((facts.g4_rope_theta_full - 1_000_000.0).abs() < f32::EPSILON);
        assert!((facts.g4_full_partial_rotary - 0.75).abs() < f32::EPSILON);
        assert_eq!(facts.g4_full_attn_interval, 3);
    }

    #[test]
    fn a_gemma4_stack_that_opens_with_full_attention_still_reads_the_sliding_rope() {
        let mut doc = descriptor("gemma4_text");
        doc["layer_types"] = json!(["full_attention", "sliding_attention", "full_attention"]);
        doc["gemma_per_layer_rope_theta"] = json!([3_000_000.0, 40_000.0, 5_000_000.0]);
        let facts = ModelFacts::from_descriptor(&doc.to_string()).unwrap();
        assert!((facts.g4_rope_theta_full - 3_000_000.0).abs() < f32::EPSILON);
        assert!((facts.g4_rope_theta_sliding - 40_000.0).abs() < f32::EPSILON);
        // Full layers at 0 and 2 is not a period-1 schedule.
        assert_eq!(facts.g4_full_attn_interval, -1);
    }

    #[test]
    fn a_negative_gemma4_sliding_window_means_absent() {
        let mut doc = descriptor("gemma4");
        doc["sliding_window"] = json!(-1);
        let facts = ModelFacts::from_descriptor(&doc.to_string()).unwrap();
        assert_eq!(facts.g4_sliding_window, 0);

        doc["sliding_window"] = json!(1024);
        let facts = ModelFacts::from_descriptor(&doc.to_string()).unwrap();
        assert_eq!(facts.g4_sliding_window, 1024);
    }

    #[test]
    fn the_quantization_pair_is_read_once_and_survives() {
        let mut doc = descriptor("llama");
        doc["quant_bits"] = json!(8);
        doc["quant_group_size"] = json!(64);
        let facts = ModelFacts::from_descriptor(&doc.to_string()).unwrap();
        assert_eq!(facts.quant_bits, 8);
        assert_eq!(facts.quant_group_size, 64);
    }

    #[test]
    fn the_architecture_stem_replaces_the_default_only_when_it_is_non_empty() {
        let mut doc = descriptor("gemma4");
        doc["arch_name"] = json!("Gemma4ForConditionalGeneration");
        let facts = ModelFacts::from_descriptor(&doc.to_string()).unwrap();
        assert_eq!(facts.arch_name, "gemma4");

        doc["arch_name"] = json!("");
        let facts = ModelFacts::from_descriptor(&doc.to_string()).unwrap();
        assert_eq!(facts.arch_name, "llama");
    }

    #[test]
    fn the_qwen35_block_reads_the_linear_attention_shape() {
        let mut doc = descriptor("qwen3_5_moe");
        doc["num_hidden_layers"] = json!(48);
        doc["linear_num_key_heads"] = json!(16);
        doc["linear_num_value_heads"] = json!(32);
        doc["linear_key_head_dim"] = json!(128);
        doc["linear_value_head_dim"] = json!(128);
        doc["linear_conv_kernel_dim"] = json!(4);
        doc["shared_expert_intermediate_size"] = json!(512);
        doc["norm_topk_prob"] = json!(false);
        doc["tie_word_embeddings"] = json!(false);
        let facts = ModelFacts::from_descriptor(&doc.to_string()).unwrap();
        assert_eq!(facts.q35_num_hidden_layers, 48);
        assert_eq!(facts.q35_linear_key_heads, 16);
        assert_eq!(facts.q35_linear_value_heads, 32);
        assert_eq!(facts.q35_linear_key_head_dim, 128);
        assert_eq!(facts.q35_linear_value_head_dim, 128);
        assert_eq!(facts.q35_linear_conv_kernel, 4);
        assert_eq!(facts.q35_shared_expert_intermediate, 512);
        assert!(!facts.q35_norm_topk_prob);
        assert!(!facts.q35_tied_embeddings);
        assert!(facts.has_linear_attn);
        // The llama block must not have run for a GDN hybrid.
        assert_eq!(facts.ll_num_hidden_layers, 0);
    }
}

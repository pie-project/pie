//! The config facts a family needs beyond the checkpoint itself.
//!
//! Ported from the CUDA driver's `model/contract.hpp::ModelFacts`, and held to
//! the same rule: the list grows only when a *partition* of a shape has to be
//! known that the checkpoint does not record. Everything else — which tensors
//! exist, what shape they are, how they are encoded, whether the experts ship
//! stacked or per-expert — is in the checkpoint, and reading it there rather
//! than predicting it from `config.json` is what lets a contract author state
//! shapes as assertions instead of guesses.
//!
//! These are read out of the `pie.model/1` descriptor, by
//! [`ModelFacts::from_descriptor`] — one mapping, on this side of the ABI.
//! The drivers used to parse `config.json` themselves and send the result
//! across in the compile request; both parsers are deleted and the request
//! carries the descriptor instead, so a fact cannot be spelled one way by a
//! driver and another way here.

use pie_loader::error::Error;

/// What one authoring call knows about the model, beyond its files.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelFacts {
    /// `model_type` from `config.json` — the key every aspect registry
    /// dispatches on.
    pub model_type: String,
    /// `quantization_config.quant_method`, empty for an unquantized
    /// checkpoint.
    pub quant_method: String,
    pub num_hidden_layers: u32,
    pub num_experts: u32,
    /// `head_dim`. TP splits an attention projection by rows, and whether a
    /// row split lands on a head boundary is not a question the row count can
    /// answer.
    pub head_dim: u32,
    /// `mamba_n_groups`, zero for a family without a Mamba mixer.
    ///
    /// Here for the same reason `head_dim` is, and it is the only way to
    /// know: a Mamba mixer's B and C bands are `groups * state` rows of a
    /// fused tensor, TP splits them by group, and no tensor in the checkpoint
    /// has either factor as an extent — the product is all that is ever
    /// stored.
    pub mamba_groups: u32,
    /// `tie_word_embeddings`, defaulting to true when the config is silent.
    ///
    /// What the config SAYS; a shipped `lm_head` is what the checkpoint DOES,
    /// and when they disagree the tensors win — the MLX authors check both.
    pub tied_embeddings: bool,
    /// The declared quantization width in bits, 0 when the config declares
    /// none.
    ///
    /// Named for what it is rather than for who reads it. It was
    /// `mlx_quant_bits` while Metal was the only side that filled it, from
    /// `mlx_lm`'s bare `quantization` block; the descriptor carries that
    /// spelling *and* `quantization_config`'s, so the value can now come from
    /// an AWQ or GPTQ checkpoint too. Only the MLX authors read it today
    /// (`common/mlx.rs`), which is a fact about the readers, not about the
    /// field.
    ///
    /// Zero is expensive rather than merely absent: `push_mlx_affine_declared`
    /// answers an undeclared width with 4 bits, so an 8-bit checkpoint that
    /// arrives here as 0 is authored with twice the logical columns and no
    /// error anywhere.
    pub quant_bits: u32,
    /// The declared quantization group size beside it, 0 when undeclared.
    pub quant_group_size: u32,
    /// Gemma-4's `num_kv_shared_layers`, 0 for every family without KV
    /// sharing: the tail of the stack attends KV an earlier layer wrote, so
    /// its own k/v projections are dead weight a contract must not declare.
    pub num_kv_shared_layers: u32,
}

impl Default for ModelFacts {
    fn default() -> Self {
        Self {
            model_type: String::new(),
            quant_method: String::new(),
            num_hidden_layers: 0,
            num_experts: 0,
            head_dim: 0,
            mamba_groups: 0,
            // The one non-zero default: HF's own default for
            // `tie_word_embeddings` is true, and the MLX authors read this
            // field the way the config is read.
            tied_embeddings: true,
            quant_bits: 0,
            quant_group_size: 0,
            num_kv_shared_layers: 0,
        }
    }
}

impl ModelFacts {
    /// Read the facts out of a `pie.model/1` descriptor.
    ///
    /// The one place a descriptor becomes facts. Every field above is a field
    /// of the descriptor — no probing of alternate spellings, no
    /// per-`model_type` rule, no defaulting beyond what the schema already
    /// resolved: `pie-model-config` did all of that once, at import, and this
    /// is a projection of the result.
    ///
    /// A field the descriptor does not carry keeps [`Default`], which is what
    /// makes this forward-compatible with a descriptor written by an older
    /// pie. `version` is checked because a *newer* one may have changed what a
    /// name means, and a silently mis-read fact is the failure this whole
    /// arrangement exists to prevent.
    pub fn from_descriptor(json: &[u8]) -> Result<Self, Error> {
        let doc: serde_json::Value = serde_json::from_slice(json)
            .map_err(|err| Error::Contract(format!("model descriptor is not JSON: {err}")))?;
        let version = doc.get("version").and_then(serde_json::Value::as_str);
        if version != Some(pie_model_config::VERSION) {
            return Err(Error::Contract(format!(
                "model descriptor declares version {:?}, this build reads {:?}",
                version.unwrap_or("<absent>"),
                pie_model_config::VERSION,
            )));
        }

        let u32_of = |key: &str| -> u32 {
            doc.get(key)
                .and_then(serde_json::Value::as_i64)
                // The schema stores these as `i32` because the C++ struct it
                // is generated from does. A negative is that struct's "unset",
                // not a count.
                .and_then(|v| u32::try_from(v).ok())
                .unwrap_or(0)
        };
        let default = Self::default();
        Ok(Self {
            model_type: doc
                .get("model_type")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_string(),
            quant_method: doc
                .get("quant_method")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_string(),
            num_hidden_layers: u32_of("num_hidden_layers"),
            num_experts: u32_of("num_experts"),
            head_dim: u32_of("head_dim"),
            // The descriptor spells it the way `config.json` does.
            mamba_groups: u32_of("mamba_n_groups"),
            tied_embeddings: doc
                .get("tie_word_embeddings")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(default.tied_embeddings),
            quant_bits: u32_of("quant_bits"),
            quant_group_size: u32_of("quant_group_size"),
            num_kv_shared_layers: u32_of("num_kv_shared_layers"),
        })
    }
}

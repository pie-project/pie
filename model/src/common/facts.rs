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
//! Today the driver fills this from its own `config.json` parse and sends it
//! across in the compile request. Once serving is artifact-only, the compiled
//! descriptor (`pie.model/1`) carries the same facts and the loader side can
//! read them itself — at which point the request keeps only the policy.

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
    /// `quantization.bits` from an `mlx_lm`-converted checkpoint, 0 when the
    /// config declares none.
    pub mlx_quant_bits: u32,
    /// `quantization.group_size` beside it, 0 when undeclared.
    pub mlx_quant_group_size: u32,
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
            mlx_quant_bits: 0,
            mlx_quant_group_size: 0,
            num_kv_shared_layers: 0,
        }
    }
}

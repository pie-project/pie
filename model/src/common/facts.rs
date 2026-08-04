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
#[derive(Clone, Debug, Default, PartialEq, Eq)]
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
}

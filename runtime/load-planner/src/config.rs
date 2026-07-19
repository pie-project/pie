#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ModelConfig {
    pub model_type: String,
    pub quant_method: String,
    pub runtime_quant: String,
    pub num_hidden_layers: u32,
    pub num_experts: u32,
    pub num_experts_per_tok: u32,
}

use crate::error::CompileError;
use crate::ffi_types::PieLoaderModelConfigView;
use crate::source::ffi_string;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ModelConfig {
    pub model_type: String,
    pub quant_method: String,
    pub quant_bits: u32,
    pub quant_group_size: u32,
    pub quant_desc_act: bool,
    pub quant_symmetric: bool,
    pub quant_zero_point: bool,
    pub runtime_quant: String,
    pub num_hidden_layers: u32,
    pub num_experts: u32,
    pub num_experts_per_tok: u32,
}

impl ModelConfig {
    pub fn from_ffi(view: &PieLoaderModelConfigView) -> Result<Self, CompileError> {
        Ok(Self {
            model_type: ffi_string(view.model_type, "model.model_type")?,
            quant_method: ffi_string(view.quant_method, "model.quant_method")?,
            quant_bits: view.quant_bits,
            quant_group_size: view.quant_group_size,
            quant_desc_act: view.quant_desc_act,
            quant_symmetric: view.quant_symmetric,
            quant_zero_point: view.quant_zero_point,
            runtime_quant: ffi_string(view.runtime_quant, "model.runtime_quant")?,
            num_hidden_layers: view.num_hidden_layers,
            num_experts: view.num_experts,
            num_experts_per_tok: view.num_experts_per_tok,
        })
    }
}

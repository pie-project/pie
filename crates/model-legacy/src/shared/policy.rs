#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum Component {
    #[default]
    Full = 0,
    Text = 1,
    Encode = 2,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum Mxfp4MoeRequest {
    #[default]
    Auto = 0,
    RoutedDecode = 1,
    NativeGemm = 2,
    EagerBf16 = 3,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum Mxfp4MoePolicy {
    #[default]
    RoutedDecode = 0,
    NativeGemm = 1,
    EagerBf16 = 2,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RuntimeQuant {
    #[default]
    None,
    Fp8,
    Int8,
    Mxfp4,

    Int4,
}

impl RuntimeQuant {
    pub fn resolve(request: &str, fp8_native: bool) -> Result<Self, String> {
        match request {
            "" => Ok(Self::None),
            "fp8" if !fp8_native => Ok(Self::None),
            "fp8" => Ok(Self::Fp8),
            "int8" => Ok(Self::Int8),
            "mxfp4" => Ok(Self::Mxfp4),
            "int4" => Ok(Self::Int4),
            other => Err(format!("unknown runtime quantization {other:?}")),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Projections {
    #[default]
    Fused,
    InPlace,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Naming {
    #[default]
    Hf,
    Mlx,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FamilyKnobs {
    pub qwen35_mtp_int8_lm_head: bool,

    pub nemotron_tp_mamba_sharding: bool,
}

impl Default for FamilyKnobs {
    fn default() -> Self {
        Self {
            qwen35_mtp_int8_lm_head: false,
            nemotron_tp_mamba_sharding: true,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Policy {
    pub projections: Projections,
    pub naming: Naming,
    pub runtime_quant: RuntimeQuant,

    pub moe_request: Mxfp4MoeRequest,
    pub component: Component,

    pub stream_routed_experts: bool,

    pub knobs: FamilyKnobs,
}

pub use super::super::spec::GptOssFacts;

#[derive(Debug, Clone, PartialEq)]
pub struct GptOssCudaFacts {
    pub mxfp4_decode_gemv: bool,

    pub mxfp4_decode_max_routes: u32,

    pub streamed_experts: bool,
}

impl GptOssCudaFacts {
    pub fn gpt_oss_20b_synthetic() -> Self {
        Self {
            mxfp4_decode_gemv: true,
            mxfp4_decode_max_routes: 32 * 32,
            streamed_experts: false,
        }
    }
}

mod combine;
mod experts;
mod route;
mod select;
#[cfg(feature = "cuda")]
mod select_bf16;
mod select_fp4;
mod select_fp8;

pub use combine::*;
pub use experts::*;
pub use route::*;
pub use select::*;
#[cfg(feature = "cuda")]
pub use select_bf16::*;
pub use select_fp4::*;
pub use select_fp8::*;

#[cfg(feature = "cuda")]
const ACT_QUANT: &str = "linear/act_quant.cu";
#[cfg(feature = "cuda")]
const DEQUANT: &str = "linear/dequant.cu";
#[cfg(feature = "cuda")]
const GEMV: &str = "linear/gemv.cu";
const COMBINE: &str = "linear/moe/combine.cu";
const ROUTE: &str = "linear/moe/route.cu";
#[cfg(feature = "cuda")]
const ROUTE_SM100: &str = "linear/moe/route_sm100.cu";
const SELECT: &str = "linear/moe/select.cu";
#[cfg(feature = "cuda")]
const SELECT_FP4_SM100: &str = "linear/moe/select_fp4_sm100.cu";
const SELECT_FP4_SM80: &str = "linear/moe/select_fp4_sm80.cu";
const SELECT_SM80: &str = "linear/moe/select_sm80.cu";
#[cfg(feature = "cuda")]
const SELECT_SM90: &str = "linear/moe/select_sm90.cu";

const BLOCK: u32 = 256;

const WARP: u32 = 32;

const MAX_EXPERTS: u32 = 512;

//! One forward pass: scratch, tables, recordings — pooled since a captured
//! graph bakes buffer addresses.
pub mod all_reduce;
pub mod attention_workspace;
pub mod attn_score;
/// Re-exported from `kernels_cuda::gemm` so `gemm::*`/`gemv::*` keep resolving.
pub use kernels_cuda::gemm::dense as gemm;
pub use kernels_cuda::gemm::gemv;
pub mod hand;
#[cfg(feature = "abi")]
pub(crate) mod envelope;
#[cfg(feature = "abi")]
pub mod launch;
/// Host side of `attn/kv_paged.cu`: `serve::transfer`'s cell move and the page-view builders.
pub mod kv_paged;
pub mod lora;
/// `moe::build_moe_ptrs_aligned_bf16`: the aligned MoE leg's pointer build.
pub mod moe_ptrs;

/// `moe::moe_grouped_gemm_bf16`: the aligned MoE leg's two grouped GEMMs.
#[cfg(feature = "_cuda")]
pub mod moe_grouped;

pub mod page_mask;
pub mod predicate;
#[cfg(feature = "abi")]
pub mod recordings;
#[cfg(feature = "abi")]
pub mod scratch;
pub mod sideband_arena;
pub mod stage_hooks;
/// The two supergraph launchers, in Rust.
pub mod supergraph;

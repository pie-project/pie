//! One forward pass: the walk, its pooled scratch and the planes it stages.
//!
//! Six modules STOOD HERE and are deleted with the legacy fire path:
//!
//! * `recordings` — the instantiated-graph cache the legacy walk replayed;
//! * `lora` — the adapter staging whose only firer was `bind::dispatch`'s
//!   `gemm::lora_qkv_correction` arm;
//! * `predicate` — the host-side guard-word evaluator a union capture
//!   uploaded;
//! * `stage_hooks` — the PTIR hook set (`wants_page_mask`, the sinks) the
//!   legacy fire read per layer;
//! * `moe_grouped` — `moe::moe_grouped_gemm_bf16`, which picked the WMMA
//!   kernel over batched cuBLAS on `x::moe::supported`. That choice belongs
//!   in a `#[claims]` body beside the point it serves (the baker backlog's
//!   `moe.matmul_select_bias`), not in the driver; it had no caller left
//!   once `dispatch` went;
//! * `moe_ptrs` — the pointer-array arena `moe_grouped` carved its two
//!   batched GEMMs out of. It outlived its only consumer by one commit.
pub mod all_reduce;
pub mod attention_workspace;
pub mod attn_score;
/// Re-exported from `kernels_cuda::gemm` so `gemm::*`/`gemv::*` keep resolving.
pub use kernels_cuda::gemm::dense as gemm;
pub use kernels_cuda::gemm::gemv;
#[cfg(feature = "abi")]
pub(crate) mod envelope;
/// Host side of `attn/kv_paged.cu`: `serve::transfer`'s cell move and the page-view builders.
pub mod hand;
pub mod kv_paged;
#[cfg(feature = "abi")]
pub mod launch;
pub mod page_mask;
#[cfg(feature = "abi")]
pub mod scratch;
pub mod sideband_arena;

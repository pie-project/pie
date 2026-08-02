//! METAL's kernel signature table — one row per launcher symbol in `csrc/`.
//!
//! Sparse next to CUDA's, and that is where Metal is rather than a gap in the
//! table: Metal has no lowered text yet. It consumes the SEMANTIC trace and
//! re-derives its dispatch selection in C++
//! (`crates/driver-metal/csrc/src/model/llama_like/declared_dag.hpp`) — the
//! same "the driver decides" shape the CUDA side is being cured of, from the
//! other end. What is declared here is what a first `llama_like.metal.*` text
//! would state.
//!
//! The rows are still binding. `model-compiler`'s `kernels::check_plan`
//! refuses any launched symbol no row declares, so such a text CANNOT be
//! written without declaring the kernels it states — the same discipline the
//! CUDA table enforces, arriving before the text that needs it.
//!
//! The words a row is written in are `kernels`'; `default-features = false`
//! gives the table without the shader build, for the same reason it does on
//! the CUDA side.

pub use kernels::{Cap, KernelSig, Prepare};
use kernels::kernel;

/// Every kernel a lowered `*.metal.*` declaration may state.
pub static KERNELS: &[KernelSig] = &[
    // ── io ─────────────────────────────────────────────────────────
    kernel!(embed_gather "embed_gather_4bit"),
    kernel!(embed_gather_mb "embed_gather_mb_4bit"),

    // ── norms / activation / residual ──────────────────────────────
    // One entrypoint serves attn_norm, mlp_norm, q_norm, k_norm and
    // final_norm — the driver fans five `Kernel` kinds onto it.
    kernel!(rms_norm "rms_single_row_bfloat16"),
    kernel!(silu_mul "silu_mul_bfloat16"),
    kernel!(residual_add "residual_add_bfloat16"),

    // ── projections ────────────────────────────────────────────────
    // The `_residual` forms fold the block residual in the GEMV/GEMM
    // epilogue, which is what a `beta_one` matmul is on this backend.
    // The readout takes this one too — `lm_head` is a projection, and
    // the driver has no separate entrypoint for it.
    kernel!(qmv "affine_qmv_fast"),
    kernel!(qmv_residual "affine_qmv_fast_residual"),
    kernel!(qmm "affine_qmm_t"),
    kernel!(qmm_residual "affine_qmm_t_residual"),

    // ── rope / kv ──────────────────────────────────────────────────
    kernel!(rope_decode "rope_neox_decode_bfloat16"),
    kernel!(rope_mb "rope_neox_mb_bfloat16"),
    kernel!(kv_append "kv_append_bfloat16"),
    kernel!(kv_append_paged "kv_append_paged_bfloat16"),

    // ── attention ──────────────────────────────────────────────────
    // No `sink` on either: Metal has no page-mask substitution path, so
    // an `attn.q` tap with PageMaskSink is unservable here — the
    // declaration says so instead of a C++ throw discovering it. No
    // capture variant exists either, so neither can publish scores.
    kernel!(sdpa_vector "sdpa_vector_decode_bfloat16_d_256",
        lacks = &[Cap::Scores, Cap::PageMaskSink]),
    kernel!(sdpa_paged "sdpa_paged_decode_bfloat16_d_256",
        lacks = &[Cap::Scores, Cap::PageMaskSink]),
];

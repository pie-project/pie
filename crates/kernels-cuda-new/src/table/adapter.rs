//! LoRA and friends.
//!
//! One row per launcher symbol. The words a row is written in —
//! [`KernelSig`], `whole`, `needs`, `lacks`, `sink` — are `kernels`'.

use kernels::kernel;
use kernels::KernelSig;

#[rustfmt::skip]
pub static KERNELS: &[KernelSig] = &[
    // PSEUDO-SYMBOL, deliberately unstated: like layout's stash pair, this
    // names an OPERATION of the declared executor, not a C++ launcher. The
    // executor performs the whole per-site LoRA apply -- the staged-table
    // walk and its GEMM sequence -- as one dispatch case ("one operation,
    // many calls"), so there is no single function whose signature a row
    // could state.
    // `launch_abi::the_pseudo_symbol_rows_are_exactly_the_known_three` is
    // where the exception is enforced.
    kernel!(lora_qkv_correction "pie_lora_qkv_correction"),
];

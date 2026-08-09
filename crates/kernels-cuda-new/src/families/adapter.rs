//! `adapter`'s JIT units — none, because `adapter` has no device kernel.
//!
//! The family is one table row, `pie_lora_qkv_correction`, and that row names
//! no symbol the linker resolves: it is a PSEUDO-SYMBOL, an operation of the
//! declared executor rather than a `__global__`. The driver performs the
//! whole per-site LoRA apply — the down projection, the up projection and the
//! scaled add onto q, k and v — as one dispatch case built out of GEMM calls
//! it already had. `kernels-cuda-new/src/table/adapter.rs` says so in as many
//! words, and `launch_abi::the_pseudo_symbol_rows_are_exactly_the_known_three`
//! enforces that the set of such rows does not grow by accident.
//!
//! So there is nothing to split: `csrc/src` has no `adapter` directory, and a
//! search of the tree for a `__global__` the row could name finds none. A
//! unit here would be an empty compile of a header that does not exist.
//!
//! This is not "not yet migrated". It is migrated as far as the concept goes,
//! and the module records that so the next reader does not go looking for the
//! file. If LoRA ever grows a fused kernel of its own — a per-site apply that
//! is one launch rather than three GEMMs — it lands in `csrc/src/adapter/`
//! and gets a unit here, and the pseudo-symbol row is what it replaces.

use crate::unit::Unit;

/// The units `adapter` compiles. Empty, and permanently so while the family
/// is one pseudo-symbol: see this module's header.
pub static UNITS: &[Unit] = &[];

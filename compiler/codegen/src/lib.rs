//! # `pie-codegen` — backend source emission for PTIR
//!
//! Everything Pie generates *from* the IR tables rather than maintaining by
//! hand. Each emitter is a pure function of the tables, so its output is
//! byte-stable and a checked-in artifact can be diffed against it in CI —
//! that drift test is the whole point: host and device cannot disagree about
//! the op vocabulary or the RNG formula if both are printed from one source.
//!
//! * [`header`] — the deterministic C ABI header (`include/ptir_abi.h`): op
//!   tags, dtype/stage/port enums, and the arity table the drivers switch on.
//! * [`rng`] — the CUDA/C++ (`include/rng_contract.generated.h`) and MSL
//!   (`include/ptir_rng.generated.metal`) projections of the canonical RNG
//!   contract in [`pie_ir::rng`].
//!
//! The CUDA and Metal *region* emitters — today's `fused_codegen.hpp`,
//! `singleton_codegen.hpp`, and `m1_codegen.cpp` in the drivers — land here
//! next, taking a [`pie_plan`]-produced region plan and a target profile and
//! returning source. They are already pure `Plan -> String` functions with no
//! device-architecture inputs, which is what makes the move possible.
//!
//! [`pie_plan`]: https://github.com/pie-project/pie/tree/dev/compiler/plan

// The emitters were authored against `alloc` paths in the `no_std` IR crate and
// still use them; `alloc` is available here through `std`.
extern crate alloc;

pub mod cuda;
pub mod header;
pub mod metal;
pub mod op_view;
pub mod program;
pub mod rng;

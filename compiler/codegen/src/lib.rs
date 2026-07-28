//! # `pie-codegen` — backend source emission for PTIR
//!
//! Everything Pie generates *from* the IR tables rather than maintaining by
//! hand. Each emitter is a pure function of the tables, so its output is
//! byte-stable and a checked-in artifact can be diffed against it in CI —
//! that drift test is the whole point: host and device cannot disagree about
//! the op vocabulary or the RNG formula if both are printed from one source.
//!
//! ## The ABI projections — one declaration, many languages
//!
//! * [`header`] — the deterministic C ABI header (`include/ptir_abi.h`): op
//!   tags, dtype/stage/port enums, and the arity table the drivers switch on.
//! * [`rng`] — the CUDA/C++ (`include/rng_contract.generated.h`) and MSL
//!   (`include/ptir_rng.generated.metal`) projections of the canonical RNG
//!   contract in [`pie_ir::rng`].
//! * [`layout`] — the lane-table field list, printed as C and as MSL and
//!   pinned to the [`pie_plan`] `#[repr(C)]` structs with `offset_of!`. Adding
//!   a field to one side without the other is a compile error rather than a
//!   silent offset shift.
//! * [`slots`] — the M1 operand-to-slot rule, shared by both region emitters.
//!
//! ## The region emitters
//!
//! [`cuda`] and [`metal`] take a [`pie_plan`]-produced [`CompiledStage`] and
//! return source (or a refusal — see [`EmittedKernel`]). They are pure
//! `Plan -> String` with no device-architecture inputs, which is what let them
//! move off the drivers' `fused_codegen.hpp` / `singleton_codegen.hpp` /
//! `m1_codegen.cpp`. Supporting them:
//!
//! * [`op_view`] — a decoded, borrow-free view of a normalized op.
//! * [`region_analysis`] — the per-region facts both backends need.
//! * [`launch`] — the launch descriptors the drivers execute.
//! * [`program`] — the whole-program bundle handed across the C ABI.
//!
//! Those last two are built out of [`pie_driver_abi`], which is why this crate
//! is the one that reaches outside `compiler/`. That crate is the contract, not
//! a driver: the compiler writes a `LaunchPackage` and the driver reads one out
//! of the same declarations, so there is no second copy to keep in step.
//!
//! [`pie_plan`]: https://github.com/pie-project/pie/tree/dev/compiler/plan
//! [`CompiledStage`]: https://github.com/pie-project/pie/tree/dev/compiler/plan
//! [`EmittedKernel`]: program::EmittedKernel

// The emitters were authored against `alloc` paths in the `no_std` IR crate and
// still use them; `alloc` is available here through `std`.
extern crate alloc;

pub mod cuda;
pub mod fault;
pub mod header;
pub mod launch;
pub mod layout;
pub mod metal;
pub mod op_view;
pub mod program;
pub mod region_analysis;
pub mod rng;
#[cfg(test)]
mod runtime_scan;
pub mod slots;

//! # `tensor-compiler` — backend source emission for PTIR
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
//!   contract in [`tensor_ir::rng`].
//! * [`layout`] — the lane-table field list, printed as C and as MSL and
//!   pinned to the [`crate::codegen::plan`] `#[repr(C)]` structs with `offset_of!`. Adding
//!   a field to one side without the other is a compile error rather than a
//!   silent offset shift.
//! * [`slots`] — the M1 operand-to-slot rule, shared by both region emitters.
//!
//! ## The region emitters
//!
//! [`cuda`] and [`metal`] take a [`crate::codegen::plan`]-produced [`CompiledStage`] and
//! return source (or a refusal — see [`EmittedKernel`]). They are pure
//! `Plan -> String` with no device-architecture inputs, which is what lets a
//! kernel be emitted, diffed and reviewed on the host without a device in the
//! loop. Supporting them:
//!
//! * [`op_view`] — a decoded, borrow-free view of a normalized op.
//! * [`wellformed`] — what a plan must satisfy before *either* backend emits
//!   from it, so that "well formed" cannot mean two things.
//! * [`alias`] — when a reshape may be elided and its consumers pointed at its
//!   source, and the table that carries that decision.
//! * [`launch`] — the launch descriptors the drivers execute.
//! * [`program`] — the whole-program bundle handed across the C ABI.
//!
//! Anything only one backend's driver reads lives under that backend, not here
//! — see [`cuda::region_analysis`].
//!
//! Those last two are built out of [`driver`], which is why this crate
//! is the one that reaches outside `compiler/`. That crate is the contract, not
//! a driver: the compiler writes a `LaunchPackage` and the driver reads one out
//! of the same declarations, so there is no second copy to keep in step.
//!
//! [`crate::codegen::plan`]: https://github.com/pie-project/pie/tree/dev/tensor-compiler's plan
//! [`CompiledStage`]: https://github.com/pie-project/pie/tree/dev/tensor-compiler's plan
//! [`EmittedKernel`]: program::EmittedKernel

// The emitters were authored against `alloc` paths in the `no_std` IR crate and
// still use them; `alloc` is available here through `std`.

pub mod alias;
pub mod cuda;
pub mod error;
pub mod fault;
pub mod header;
pub mod launch;
pub mod layout;
pub mod metal;
pub mod op_view;
pub mod program;
pub mod rng;
#[cfg(test)]
mod runtime_scan;
pub mod slots;
pub mod wellformed;

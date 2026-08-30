//! # `eta-compiler` — the host toolchain for guest tensor programs
//!
//! Decides how a guest's ETA program executes ([`plan`]), what it means ([`eval`]), and emits
//! the backend source ([`codegen`]). `plan` ⊥ `eval`, enforced by `tests/module_layering.rs`.
//!
//! The output artifact is [`codegen::launch::LaunchPackage`] — a program in the
//! shape an engine executes it, with [`codegen::program::EmittedKernel`] and
//! [`codegen::cuda::region_analysis::RegionAnalysis`] beside it. It is declared
//! HERE, not in the runtime↔engine contract, because the producer owns its
//! output type; `engine` names these and this crate names nothing of
//! `engine`'s. That module's header carries the argument.

extern crate alloc;

pub mod codegen;
pub mod eval;
pub mod plan;

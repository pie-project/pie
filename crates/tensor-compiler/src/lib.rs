//! # `tensor-compiler` — the host toolchain for guest tensor programs
//!
//! Decides how a guest's PTIR program executes ([`plan`]), what it means ([`eval`]), and emits
//! the backend source ([`codegen`]). `plan` ⊥ `eval`, enforced by `tests/module_layering.rs`.

extern crate alloc;

pub mod codegen;
pub mod eval;
pub mod plan;

//! # `tensor-compiler` — the host toolchain for guest tensor programs
//!
//! A guest authors a tensor program with [`tensor-dsl`], encodes it as PTIR,
//! and hands it over. This crate is everything the host then does with it:
//! decide **how** it executes ([`plan`]), say what it **means**
//! ([`eval`]), and write the backend source that runs it ([`codegen`]).
//! `tensor-ir` stays outside — it is the dependency floor both the guest and the
//! host sit on, and the only part of the toolchain a wasm build reaches.
//!
//! ## Why one crate, and what the module boundaries still owe
//!
//! These were three crates (`tensor-compiler`, `tensor-compiler`, `tensor-compiler`) plus a
//! test-only fourth. The split bought nothing a module cannot: the three ship
//! together, version together, and are consumed together by the engine. What
//! it cost was the conformance battery — the goldens, the malformed-wire
//! corpus, the drift checks — having to live in a crate of its own because no
//! one of the three could own tests that span all of them. Folded, that
//! battery is just `tests/`.
//!
//! One boundary the crate system WAS holding is kept by hand:
//!
//! **`plan` ⊥ `eval`.** Planning and semantics are two independent answers
//! about the same trace, and the value of the interpreter as an oracle comes
//! entirely from its not knowing what the planner decided. They were siblings
//! over `ir` with no edge between them, and dev-only in the direction parity
//! tests needed. As modules that discipline is no longer type-checked, so it
//! is stated here and tested in `tests/module_layering.rs`: **`eval` may not
//! name `plan`, and `plan` may not name `eval`.** `codegen` sits above `plan`
//! and may name it; nothing may name `codegen` but the battery.
//!
//! [`tensor-dsl`]: https://docs.rs/tensor-dsl

// `alloc` paths run through the whole crate: the planner and the emitters were
// authored against the `no_std` IR crate's `alloc::{vec, string}` vocabulary
// and keep it, even though this crate itself is host-only.
extern crate alloc;

pub mod codegen;
pub mod eval;
pub mod plan;

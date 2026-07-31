//! Grammar front end for engrain.
//!
//! The EBNF, regex and JSON Schema lowering, the grammar IR and the
//! finite-automaton construction in this crate are copied from `pie-grammar`
//! (<https://github.com/pie-project/pie>, Apache-2.0), which is itself a Rust
//! rewrite derived in part from XGrammar. They are vendored rather than taken
//! as a dependency because the parser above them is being replaced: engrain
//! needs a deterministic LR automaton so that a single terminal's
//! admissibility follows from the stack top, which is what lets the vocabulary
//! collapse into a few hundred terminal-sequence groups and lets the whole
//! decode step run without touching the host.
//!
//! Deliberately not copied: the matcher, the compiled-grammar cache and the
//! compiler driver. Those implement a nondeterministic stack PDA whose current
//! state is a *set*, and removing that nondeterminism is the point.

pub mod bitmask;
pub mod brle;
mod frontend;
pub mod fsm;
pub mod grammar;
pub mod json_schema;
pub mod regex;

//! LALR(1) construction over a grammar's context-free skeleton.
//!
//! The lexicon crate splits a grammar into terminals and structure; this crate
//! turns that structure into the tables the GPU runs: an ACTION row per state,
//! a GOTO table, and production arities. Because canonical LR(1) states
//! explode on the thousand-rule grammars these front ends emit, lookaheads are
//! merged per LR(0) state, which is what makes the tables small enough to keep
//! resident.

pub mod cfg;
pub mod parser;
pub mod tables;

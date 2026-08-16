//! # `tensor-compiler` — backend source emission for PTIR
//!
//! Everything Pie generates *from* the IR tables rather than maintaining by
//! hand. Each emitter is a pure function of the tables, so its output is
//! byte-stable and CI can diff a checked-in artifact against it: host and
//! device cannot disagree about the op vocabulary or the RNG formula if both
//! are printed from one source.
//!
//! [`rng`], [`layout`] and [`slots`] project one declaration into several
//! languages. [`cuda`] and [`metal`] turn a compiled stage into source and are
//! pure `Plan -> String` with no device-architecture inputs, so a kernel can be
//! emitted, diffed and reviewed without a device; [`op_view`], [`wellformed`],
//! [`alias`], [`launch`] and [`program`] support them. Anything only one
//! backend's driver reads lives under that backend — [`cuda::region_analysis`].

pub mod alias;
pub mod cuda;
pub mod error;
pub mod fault;
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

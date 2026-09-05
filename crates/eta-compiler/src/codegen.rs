//! Backend source emission for ETA: everything generated *from* the IR
//! tables rather than maintained by hand. Each emitter is a pure function of
//! the tables, so host and device cannot disagree if both are printed from
//! one source. [`rng`], [`layout`] and [`slots`] project one declaration
//! into several languages; [`cuda`] and [`metal`] turn a compiled stage into
//! source as pure `Plan -> String`, with [`op_view`], [`wellformed`],
//! [`alias`], [`launch`] and [`program`] supporting them.

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
pub mod wgsl;
pub mod wgsl_analysis;

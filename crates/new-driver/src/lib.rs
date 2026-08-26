//! The execution walk, written once, generic over any
//! [`Dispatch`](new_kernels::Dispatch) (design §8): split a plan into its
//! prepare and capture phases ([`phases`]), then run node indices in program
//! order under their guards ([`walk`], or [`fire`] for both phases eagerly).
//!
//! Deliberately policy-free: rows-bucketing, graph capture itself, and
//! memoization of plan ops are the concrete drivers' business (`driver-cuda`,
//! `driver-metal`) — this crate only states *what* runs and in *what order*,
//! never how a backend amortizes it. No async, no threads: a fire is one
//! ordered pass over one queue.

pub mod phase;
pub mod walk;

pub use phase::{Phases, phases};
pub use walk::{fire, walk};

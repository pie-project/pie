//! Executors: run a finished plan.
//!
//! Promoted out of `testkit` because one executor is a production path:
//! `pie model convert` materializes artifacts through [`host`], so the module
//! that does it can no longer live under a name that says "exists only to
//! check the loader". The *oracle* the executor is diffed against
//! ([`crate::testkit::reference`]) keeps that name and its feature gate; the
//! line between them is the line between running a plan and second-guessing
//! one.
//!
//! [`sink`] is where finalized tensors go. An executor that returns a map of
//! every tensor forces its caller's peak memory to the whole output; a sink
//! receives each tensor once, in schedule order, and the streaming entry
//! point frees buffers the moment the schedule is done with them.

pub mod host;
pub mod sink;

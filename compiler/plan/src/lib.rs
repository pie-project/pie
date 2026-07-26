//! # `pie-plan` — PTIR execution planning
//!
//! The backend-neutral middle end. Given a bound trace ([`pie_ir::validate`]),
//! this crate decides **how** a program executes: it normalizes each stage,
//! derives its canonical signature, classifies value domains, partitions the op
//! DAG into generated / library / second-party regions, and lays out the
//! lane-table ABI. Backends consume the serialized result and supply only code
//! generation and library implementations.
//!
//! "Plan" here is the cuDNN/FFTW sense — a reusable, shape-parameterized
//! execution strategy keyed by [`ExecutableCacheKey`] — **not** an LLVM-style
//! optimization pass pipeline. Runtime-varying extents stay symbolic
//! ([`SymbolicExtent`]) so one plan serves many batch shapes.
//!
//! Two wire formats leave this crate:
//!
//! * the **region plan** (`PTRP`, [`encode_stage_plan`]) — per-stage regions,
//!   schedules, and lane records;
//! * the **bound-trace sidecar** (`PTIB`, [`sidecar`]) — the typed lowering that
//!   carries those plans, so a driver never re-infers shapes.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

mod compile;
pub mod sidecar;

pub use compile::*;

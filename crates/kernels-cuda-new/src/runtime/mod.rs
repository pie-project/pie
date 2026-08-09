//! Layer 3: compile, cache, load, fire.
//!
//! Everything below this module is DATA — rows in [`crate::table`], device
//! rows in [`crate::device`], text in [`crate::source`], and units in
//! [`crate::unit`] saying which text a set of rows instantiates. None of it
//! needs a GPU, a driver, or `cudarc`, which is what lets `model-compiler`
//! read a row on a machine that has never had CUDA installed.
//!
//! This module is the other half: the one that turns a row into a launch. It
//! is the only part of the crate that links `cudarc`, and it is behind
//! `cuda-12`/`cuda-13` for that reason — the boundary is a feature, not a
//! naming convention.
//!
//! # The path a fire takes
//!
//! | step | module | what it produces |
//! |---|---|---|
//! | 1 | [`nvrtc`] | a cubin, and the mangled name of every row's instantiation |
//! | 2 | [`module`] | a loaded image with a `CUfunction` per row |
//! | 3 | [`cache`] | that image, once per unit, for the process |
//! | 4 | [`launch`] | a grid and a block, from the row's rule and the fire's rectangle |
//! | 5 | [`args`] | a `void**`, checked against the row's operands |
//! | 6 | [`fire`](fn@fire) | the `cuLaunchKernel` |
//!
//! Steps 1 and 2 happen once per unit; 4, 5 and 6 happen on every fire. Step 3
//! is what separates them, and it is why the split is worth stating: the
//! expensive half is per unit and the cheap half is per token, and a change
//! that moves work across that line is the only kind of change here that has a
//! performance consequence.
//!
//! # Step 4½: the row that names two kernels
//!
//! A specialised row — [`crate::device::Specialisation`] — puts one more step
//! between 5 and 6, and it is the one step in the list that could not exist
//! ahead of time: a predicate over the values just bound chooses which
//! INSTANTIATION to launch. `norm::rmsnorm_strided_bf16` is the first, and it
//! reproduces `rmsnorm.cu`'s `rmsnorm_vec8_ok` — three pointer alignments and
//! three strides — on the addresses the fire is about to hand the driver.
//!
//! It sits deliberately on the cheap side of the line. The choice reads the
//! `ArgValue`s and nothing else: no device memory, no `cudaMemcpy`, no
//! synchronisation, which [`crate::device::Fact`] enforces by having no
//! variant that could carry one. It cost **21 ns per fire** against a 2.2 us
//! launch when `tests/specialise.rs` timed 100 000 of them, and it adds no
//! compile — the arm's row is a row of the SAME unit, so step 1 already
//! produced its symbol and step 3 already cached it. [`cache`]'s header
//! records what the alternative would have cost.
//!
//! # The cache is global, and that is the design choice
//!
//! A compiled unit lives in a `OnceLock` inside [`cache`], keyed by unit and
//! architecture, for the process — not in a `Kernels` handle that every caller
//! threads through. That alternative was considered and rejected for the
//! launch path's sake: a fire happens once per kernel per layer per token, and
//! the handle would be one more argument on every one of them to express a
//! fact that is already global — a process serves one device, and a cubin is
//! per-architecture.
//!
//! What it costs is that the architecture is discovered rather than passed,
//! and that cost is paid in one place, [`cache::arch`], which asks the current
//! device once. What it buys is that [`fire`](fn@fire) takes a symbol, a
//! rectangle, a value list and a stream, which is exactly what a dispatcher
//! has.

pub mod args;
pub mod cache;
pub mod error;
pub mod fire;
pub mod launch;
pub mod module;
pub mod nvrtc;
pub mod stream;

pub use args::{ArgError, ArgValue, Args};
pub use error::Error;
pub use fire::{fire, hosts, row, selects};
pub use launch::{Dims, Launch, Ungeometric, eval};
pub use module::KernelModule;
pub use stream::Stream;

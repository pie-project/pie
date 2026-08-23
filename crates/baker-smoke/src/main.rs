//! One decode step of a catalog SKU, end to end, out of a `Program`.
//!
//! ```text
//! baker-smoke [--sku <sku>] [--cache <hf-dir>] [--base <flavor>]
//!             [--token <id>] [--top <k>] [--stop <steps>]
//!             [--probe <op>]... [--trace]
//! ```
//!
//! THE WHOLE CHAIN, once: `model::catalog()` traces the plan;
//! `model_compiler::program::bound` binds its lanes; the lane whose fact
//! word says `qo_one` is the decode lane; `model::produce` builds the
//! weights out of a real HF checkpoint and this uploads them; the steps run
//! in order against a one-row arena and one page of KV per attention layer;
//! and the value the plan's `out` seam names comes back as logits.
//!
//! WHAT THIS IS NOT. It is not a driver. There is no scheduler, no
//! allocator, no capture, no batching, no request lifecycle -- one fire of
//! one row, allocated once and freed at exit. Everything it does that a
//! driver would also do is done the way the driver does it, and every place
//! it had to decide something the driver decides elsewhere is written down
//! at the decision.
//!
//! THE TWO HALVES OF THE FIRE, and why they read so differently:
//!
//! * a `Call::Point` is answered by the plane's own claim -- `Ctx` wearing
//!   `kernels::points::Norm` and its siblings. The executor builds the marks
//!   from the slots and calls the trait method. That half is mechanical and
//!   is what a generated dispatch will do.
//! * a `Call::Symbol` is a routine that keeps its own `canon` because no
//!   honest delegation exists (`kernels-cuda/src/ssm.rs:103-121` states
//!   three of them by name). Those need STAGING: operands the statement does
//!   not carry, results it does not state, resident objects it only names.
//!   That half is hand-written here, once, and every decision in it cites
//!   the driver-side line it mirrors.

#[cfg(feature = "_cuda")]
mod dev;
#[cfg(feature = "_cuda")]
mod marks;
#[cfg(feature = "_cuda")]
mod smoke;

/// NO `compile_error!` FOR A FEATURELESS BUILD, and the reason is
/// `driver-cuda/src/lib.rs:19-21`'s: nothing links `cudarc` without one, so
/// no segfault is reachable, and refusing to compile would take
/// `cargo check --workspace` down with it. A binary that cannot fire says so
/// when it is run.
#[cfg(not(feature = "_cuda"))]
fn main() {
    eprintln!(
        "baker-smoke selected no CUDA runtime version: rebuild with \
         `--features cuda-12` or `--features cuda-13`, matching the libcudart \
         this binary will load"
    );
    std::process::exit(2);
}

#[cfg(feature = "_cuda")]
fn main() {
    smoke::main();
}

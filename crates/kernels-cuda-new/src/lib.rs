//! CUDA's kernels as a JIT: the table, the sources, and the runtime that
//! fires them.
//!
//! # Three layers, because there are three consumers
//!
//! This crate is `kernels-cuda` with the C++ build removed and the *host*
//! half moved in. That sounds like a bigger crate; it is a smaller one,
//! because the thing being deleted — a CMake project, an archive, a generated
//! `extern "C"` shim, a `links` key, and the toolkit needed to build all four
//! — was only ever there to get a `<<<>>>` from a `.cu` file into a Rust
//! call. NVRTC does that from a string.
//!
//! What is left has to serve three consumers that want three different
//! amounts of it, so it is arranged in three layers and each is a feature
//! boundary rather than a naming convention:
//!
//! | layer | module | needs | consumer |
//! |---|---|---|---|
//! | 1. the table | [`table`] | nothing | `model-compiler` — reads a row on every trace, has no GPU |
//! | 2. the sources | [`source`], [`unit`], [`device`] | nothing | anyone who wants to compile them, including an offline cache builder |
//! | 2½. the generators | [`emit`], [`abi`] | nothing | build scripts — this crate's own, `kernels-cuda`'s shim, `driver-cuda`'s dispatch |
//! | 3. the runtime | [`runtime`] | `cudarc` | `model-loader` (a few symbols, directly), `driver-cuda` (every row, dynamically) |
//!
//! Layers 1 and 2 are DATA — rows, and text. They compile on a machine with
//! no CUDA, no toolkit and no driver, which is what lets `model-compiler`
//! depend on this crate unconditionally. Layer 3 is behind `cuda-12`/
//! `cuda-13`, so turning it on is a decision a binary makes, not one it
//! inherits.
//!
//! The generators are a half-layer rather than a fourth because they add no
//! dependency of their own: they are functions from rows to strings, and a
//! build script calling one must be able to do so on a machine with no
//! toolkit. That is exactly why [`abi`] is here — it emits the AHEAD-OF-TIME
//! artefacts, so it looked like the archive's, but its INPUTS are layers 1
//! and 2 and it needs nothing else. See its own header.
//!
//! # Layer 3 owns the modules, and that is the design choice
//!
//! A compiled unit lives in a `OnceLock` inside [`runtime`], keyed by unit and
//! architecture, for the process. The alternative — a `Kernels` handle every
//! caller threads through — was considered and rejected for the launch path's
//! sake: a fire happens once per kernel per layer per token, and the handle
//! would be one more argument on every one of them to express a fact that is
//! already global (a process serves one device, and a cubin is
//! per-architecture).
//!
//! The cost is that `arch` is discovered rather than passed. That is a real
//! cost and it is paid in one place, [`runtime::cache::arch`], which asks the
//! current device once.
//!
//! # Two surfaces, one row
//!
//! The reason the layers stay honest is that both ways of calling a kernel are
//! generated from the same row:
//!
//! * **Dynamically** — [`runtime::fire`], given a symbol. This is what
//!   `driver-cuda` does for all of its rows, because `model-compiler` writes
//!   a symbol into a trace and the dispatcher looks it up. A symbol is a
//!   string at that boundary and no amount of type machinery makes it
//!   anything else.
//! * **Directly** — a typed `fn` per kernel, in [`x`]. This is what
//!   `model-loader` wants: it calls four kernels by name, from Rust, with no
//!   trace in sight.
//!
//! The second was a *generated* module until north star §6 half A. `build.rs`
//! ran `emit::emit_rust_api` over every row [`unit::UNITS`] compiles and wrote
//! `api.rs`, which `pub mod api` included — the replacement for
//! `kernels-cuda`'s `emit_c_shim`, the same generator reading the same rows
//! and emitting a Rust function instead of an `extern "C" pie_k_*`.
//!
//! It is a hand-written `fn` now, and the reason is what the retirement
//! measured rather than what it planned. **`emit.rs` never read
//! `table::ROW_TABLES`.** It read [`unit::rows`] — the *device* row list,
//! which every fn-world family contributes to because a unit's rows are how
//! the JIT learns which template instantiations to lower. So `api.rs` would
//! have been regenerated for as long as the JIT existed, and "step 6 unblocks
//! when `ROW_TABLES` empties" was true of the dispatcher and false of this.
//!
//! What it was actually blocked on was callers, and there was **one**:
//! `tests/fire.rs`. `model-loader` had already crossed to [`x::quant`]'s four
//! host programs and said so in its own module doc. A 1,070-line generator
//! and a public module survived a whole sweep on one line in one test.
//!
//! # What is deliberately not here
//!
//! No `csrc/`. The device sources are `include_str!`-ed out of
//! `kernels-cuda/csrc/src` (see [`source`]) rather than copied, for the reason
//! the rows themselves are not copied: while both paths must run, one file is
//! one contract, and two copies of a thing agree until the day they do not.
//!
//! [`table`] and [`device`] used to be that too — `pub use`s of
//! `kernels-cuda`'s rows, with the seam written out at length in both files.
//! The seam closed by moving the rows HERE and inverting the dependency:
//! `kernels-cuda` takes the table from this crate and re-exports it under the
//! paths it used to own, so every consumer of `kernels_cuda::*` kept
//! compiling unedited. What did NOT happen is the archive retiring — 109 of
//! the 198 table rows still have no JIT twin — and the tree the sources are
//! read from is the same tree for the same reason. The direction was decided
//! by who authors a row rather than by who is finished: a row here states a
//! `LaunchRule` and an instantiation, which is the whole kernel, and the
//! archive's crate reads rows to emit C.
//!
//! No archive either, and no toolkit: the build script writes a text file
//! (see [`emit`]), so this crate builds on a machine that has never seen a
//! GPU. That claim is now `model-compiler`'s too, since its edge to
//! `kernels-cuda` reaches this crate.

#![cfg_attr(docsrs, feature(doc_cfg))]

// One error naming the two features, rather than a hundred naming `cudarc`.
// `driver-cuda` learned this the hard way: without it, the first thing to
// fail is cudarc's own build script, which panics about a missing CUDA
// version and says nothing about how to fix it.
#[cfg(all(feature = "_cuda", not(any(feature = "cuda-12", feature = "cuda-13"))))]
compile_error!(
    "kernels-cuda-new's runtime needs exactly one CUDA runtime version: \
     enable `cuda-12` or `cuda-13`, matching the libcudart this binary will load"
);

#[cfg(all(feature = "cuda-12", feature = "cuda-13"))]
compile_error!(
    "kernels-cuda-new: `cuda-12` and `cuda-13` are mutually exclusive -- a binary \
     loads one libcudart, and the two disagree on `cudaGraphAddNode`'s arity"
);

/// The words a row is written in.
///
/// Re-exported so a consumer needs one dependency rather than two to read the
/// table, exactly as `kernels-cuda` re-exports them today.
pub use kernels::{Cap, KernelSig, LaunchRule, Lit, Operand, Prepare, Source, Ty};

/// The AHEAD-OF-TIME generator, over the same rows [`emit`] reads.
///
/// Layer 1½, and the layer is the reason it is here rather than beside the
/// archive it serves. Every emitter in this module is a pure function from
/// rows to text — the `extern "C"` shim that gives `csrc/`'s launchers
/// callable symbols, the matching Rust declarations, the device typecheck
/// translation unit, and the dispatch arms `driver-cuda`'s build script
/// writes. Not one of them opens a `.cu`, calls nvcc or links anything, so
/// none of them needs `cuda-12`/`cuda-13` and none of them belongs in a crate
/// that does.
///
/// It was `kernels_cuda::abi` until this round. What that cost was one
/// dependency edge with nothing behind it: a build script wanting a generated
/// dispatch had to depend on the crate that also runs CMake, which is
/// `new-horizon.md` §21.5's first of three and the only one that was not
/// about an archive. `kernels-cuda` re-exports the module so its own shim
/// generator and `driver-cuda`'s tests are unedited.
///
/// A SIBLING of [`emit`] and deliberately not a submodule: they read one
/// table to opposite ends — `emit` writes typed Rust over `runtime::fire`
/// for rows NVRTC compiles, `abi` writes C++ and `extern "C"` for rows an
/// archive already holds — and a generator that served both would have to be
/// told which build it was serving on every call.
pub mod abi;
pub mod device;
/// Which of three things a stated symbol is, and who executes it.
///
/// A sibling of [`device`] rather than part of it: [`device`] says what a row
/// must state to name a `__global__`, and [`execution`] says whether the
/// symbol names one at all — some are host programs over several of our
/// kernels, and some are served by a library the driver links and never were
/// kernels. Layer 2, and DATA: a name, not a call. See its header for why
/// [`KernelSig`] does not change.
pub mod execution;
/// FA2's launch arithmetic: `decode.cuh` and `prefill.cuh`'s host prologues.
///
/// [`plan`]'s sibling and its opposite half. [`plan`] decides which CTA does
/// which work; [`fa2`] decides what shape a CTA is, and therefore which
/// template instantiation has to exist for it. Both are the parts of
/// FlashInfer that nvcc used to run at build time, once per head_dim, in the
/// four `attention_flashinfer_hd<N>.cu` translation units.
pub mod fa2;
/// The kernel families, each owning the units it compiles.
///
/// One module per family, because a single list of units would be a file
/// every migration touches — see [`families`] for the rest of the reasoning.
pub mod families;
/// The attention scheduler: `flashinfer/attention/scheduler.cuh` as host Rust.
pub mod plan;
pub mod source;
pub mod table;
pub mod unit;
/// **kernel-x** — the floor a kernel stands on when it is written as a
/// program rather than as a row.
///
/// `.wiki/kernel-x/northstar.md` §5 steps 1-3. A family that has crossed
/// lives here whole: its device declarations, its host programs, its
/// contracts and its binds, in one file beside no table at all. [`table`]
/// and [`families`] keep the families that have not crossed yet, and the two
/// worlds meet in exactly two lines — `table::TABLES` takes `x::rope::SIGS`
/// and `families::ALL` takes `x::rope::UNITS`.
pub mod x;

#[cfg(feature = "_cuda")]
#[cfg_attr(docsrs, doc(cfg(any(feature = "cuda-12", feature = "cuda-13"))))]
pub mod runtime;

// `pub mod api` was here: `include!(concat!(env!("OUT_DIR"), "/api.rs"))`,
// one typed function per row [`unit::UNITS`] compiles, written by
// `emit::emit_rust_api` at build time. North star §6 half A retired it and
// `src/emit.rs` with it.
//
// The module is gone rather than emptied because an emitter with no caller is
// worse than no emitter: it kept generating, it kept a `pub` surface alive,
// and every row that gained an operand `runtime::args::Args::bind` refuses
// silently became a comment in a file nothing read. The direct-call surface
// is [`x`] now, where the function is written rather than derived, and the
// operands a `fn` takes are checked by rustc at the call site instead of
// being dropped with a note at the generator.

/// The launch path, at the top level because it is the one thing every
/// consumer of layer 3 uses.
#[cfg(feature = "_cuda")]
pub use runtime::{ArgValue, Dims, Error, Stream, fire, hosts};

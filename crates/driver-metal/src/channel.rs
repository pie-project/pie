//! The PTIR channel plane — re-exported, not owned.
//!
//! This was `src/pipeline/`, twenty files and ten thousand lines, until the
//! CUDA shell reached the same layer. Not one of those lines named a Metal
//! type: the directory built and tested on Linux while the crate around it did
//! not, which is the mechanical form of the claim `.wiki/driver/progress-metal.md` opens with —
//! *"the portable half — everything that is a function of the plan and the
//! fire's numbers rather than of the device — goes first, because it is the
//! half that can be tested without a GPU."*
//!
//! `.wiki/driver/progress-metal.md` named the destination before the move happened:
//! *"`driver-cuda` shares `driver` and `tensor-ir` with this crate, so
//! `src/pipeline/` is the natural single home for both device copies when that
//! port reaches this file."* It has, so this file is the re-export and
//! the `driver` crate is the home.
//!
//! # Why the path moved after all
//!
//! It stayed `driver_metal::pipeline::*` for one revision on the argument
//! that a rename makes a behaviour-preserving move look like one that is
//! not. What that cost was the collision
//! `.wiki/driver/real-metal-north-star.md` §5 names: `src/pipeline.rs` was
//! this, and `src/metal/pipeline.rs` was the shader compiler — one word at
//! two altitudes, in one crate, and `super::pipeline` meant a different
//! module from `crate::pipeline` in the same file.
//!
//! They are [`channel`](self) and `gpu::program::compile` now. Every caller
//! is under `gpu/program/` or `gpu/device/`, so the move was a `use` line
//! each and the ambiguity is gone.
//!
//! # Why this sits at the crate root
//!
//! It landed under `gpu/program/`, next to the compiler it had been confused
//! with. That was wrong twice over. This file is `pub use driver::*` and
//! nothing else — it is not a layer, it is this crate's single naming of the
//! ABI it serves, and there is no Metal in it.
//!
//! Worse, `gpu/device/ring.rs` needs the wire format, so the placement made
//! `gpu/device/` point UP at `gpu/program/` while `gpu/program/` pointed
//! down at `gpu/device/` in ten places — a cycle, and one that read like a
//! real dependency on the compiler when it was only a re-export of an
//! external crate. `.wiki/driver/real-metal-north-star.md` §9 asks for
//! layers that point down; at the root this one is below everything, so
//! every caller points down at it and none of them is wrong to.

// `driver`, not `driver_api`. The move landed as `crates/driver-pipeline` and
// then consolidated into `driver`; this file kept pointing at the crate the
// first draft named, so every one of the twenty-odd `pipeline::` imports
// under `gpu/` failed to resolve and the LIBRARY did not build.
pub use driver::*;

// The re-export points at the RIGHT crate, checked at compile time.
//
// `driver` and `driver-api` are one typo apart, and the tree has paid for it
// once: when `driver-pipeline` became `driver`, this file was pointed at
// `driver_api` instead, every `pipeline::` import under `gpu/` stopped
// resolving, and the Apple-only half of the build was broken while Linux CI
// could not see it (`31938a2b6`, fixed in `a87693dff`). That last clause is
// no longer true: the gate is `feature = "metal-4"` and one macOS runner
// builds both halves.
//
// A glob re-export cannot fail loudly on its own — it succeeds and exports
// the wrong names — so the check has to name something. `PassInputs` and
// `Registry` are the channel plane's, they exist ONLY in `driver`, and the
// engine seam reaches for both by this path.
//
// Re-pointing this at `driver_api` produces sixteen errors, and without this
// assertion every one of them is an unresolved import somewhere else — the
// symptom, twenty-odd files from the cause. This one fails ON the re-export,
// which is the line to change.
const _: fn() = || {
    fn from_the_pipeline_crate<T>(_: Option<&T>) {}
    from_the_pipeline_crate::<self::PassInputs<'static>>(None);
    from_the_pipeline_crate::<self::Registry>(None);
};

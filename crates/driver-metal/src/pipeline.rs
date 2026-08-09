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
//! [`driver_pipeline`] is the home.
//!
//! The path stays `driver_metal::pipeline::*` on purpose. Every call site
//! in `src/metal/` and `src/batch/` names it, the names did not change, and a
//! rename would have made a move that changes no behaviour look like one that
//! does.

// `driver`, not `driver_api`. The move landed as `crates/driver-pipeline` and
// then consolidated into `driver`; this file kept pointing at the crate the
// first draft named, so every one of the twenty-odd `pipeline::` imports in
// `src/metal/` failed to resolve and the LIBRARY did not build.
pub use driver::*;

// The re-export points at the RIGHT crate, checked at compile time.
//
// `driver` and `driver-api` are one typo apart, and the tree has paid for it
// once: when `driver-pipeline` became `driver`, this file was pointed at
// `driver_api` instead, every `pipeline::` import under `src/metal/` stopped
// resolving, and the Apple-only half of the build was broken while Linux CI
// could not see it (`31938a2b6`, fixed in `a87693dff`).
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

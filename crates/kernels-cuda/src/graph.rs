//! The supergraph's two arming kernels, and NOT a family.
//!
//! `cudaGraphSetConditional` writes a CONDITIONAL HANDLE, not a tensor, so
//! there is no `FAMILY` and no trace statement names it; the driver calls
//! these two by path from inside its graph capture.
//!
//! Arming from INSIDE the graph lets a replay take a fire's arms with no
//! host round-trip. The handle crosses as `usize`: `cudaGraphConditionalHandle`
//! is `unsigned long long`, a different type from `size_t` on LP64, so the
//! one conversion happens at the one call site, cast explicitly.

use kernels::{Bind, Fire};
use core::ffi::c_void;

use crate::jit::{Ctx, Launch, Root};
use kernels::Refusal;

/// `graph/supergraph.cuh` — the root these two compile a symbol out of.
pub static ROOT: Root = Root::new("graph/supergraph.cuh");

/// `csrc/supergraph.cu:61` and `:74` — `<<<1, 1, 0, stream>>>`, both
/// launchers; one thread, since a second thread writing the handle races undefined behaviour.
const ARM: Launch = Launch::grid([1, 1, 1], [1, 1, 1]);

/// Arms `handle` from `preds[slot]` as a BOOLEAN. `ctx`'s stream must be
/// CAPTURING: the launch becomes the conditional node's upstream dependency
/// and an IF node downstream reads the byte.
///
/// # Errors
/// [`Refusal::Device`] if the compile, load or launch refused (usually "the
/// stream is not capturing"); the caller abandons the capture and runs eagerly.
///
/// # Safety
/// `preds` must address a live device predicate word with at least `slot + 1`
/// bytes, live across the launch, and `ctx`'s stream must outlive it.
pub fn supergraph_set_cond(
    ctx: &Ctx<'_>,
    handle: usize,
    preds: *const c_void,
    slot: i32) -> Result<(), Refusal> {
    arm(ctx, "::pie::graph::supergraph_set_cond", handle, preds, slot)
}

/// Arms `handle` from `preds[slot]` as a body INDEX rather than a boolean.
///
/// Shares [`supergraph_set_cond`]'s contract; `cudaGraphSetConditional` takes
/// the same unsigned value, only an IF reads it as 0/1 and a SWITCH as an
/// arm index, so the byte is written through unchanged.
///
/// **Out-of-range selects no body, deliberately not clamped**: a fire whose
/// predicate says "arm 4" of a three-arm switch has a lowering/driver
/// disagreement, and clamping to arm 0 would run the wrong program instead of nothing.
///
/// # Errors
/// [`supergraph_set_cond`]'s.
///
/// # Safety
/// [`supergraph_set_cond`]'s.
pub fn supergraph_set_switch(
    ctx: &Ctx<'_>,
    handle: usize,
    preds: *const c_void,
    slot: i32) -> Result<(), Refusal> {
    arm(ctx, "::pie::graph::supergraph_set_switch", handle, preds, slot)
}

/// Compile and load both arming kernels NOW, before any capture is open.
///
/// These two are the only device text launched exclusively inside a graph
/// capture, so their lazy JIT can never already have happened by the time a
/// capture wants them. A first resolve inside an open capture fails: its
/// `bind_context` call does a `cudaFree(null)`, which
/// `cudaStreamCaptureModeGlobal` prohibits, leaving the capture
/// `cudaErrorStreamCaptureInvalidated`. So the fix is timing, not the
/// launch: compile before the capture opens, once per process per
/// architecture; after that this is a `OnceLock` read and costs nothing.
///
/// Both forms are warmed — a capture may open a switch without ever opening
/// a conditional, and which one it needs isn't known before the trace is
/// walked.
///
/// # Errors
/// [`Refusal::Device`] if either will not compile or load, before the
/// caller has committed to a capture.
#[cfg(feature = "_cuda")]
pub fn warm() -> Result<(), Refusal> {
    for instantiation in [
        "::pie::graph::supergraph_set_cond",
        "::pie::graph::supergraph_set_switch",
    ] {
        if crate::jit::cache::resolve(&ROOT, instantiation).is_err() {
            return Err(Refusal::Device {
                why: "an arming kernel would not compile or load; see the log",
            });
        }
    }
    Ok(())
}

/// The same, for a build that selected no CUDA runtime: nothing to warm.
/// Not an error — a capture cannot open here either, so refusing would make
/// every caller carry a branch for an unreachable case.
#[cfg(not(feature = "_cuda"))]
#[allow(clippy::unnecessary_wraps)]
pub fn warm() -> Result<(), Refusal> {
    Ok(())
}

/// The body both share: bind three operands, launch one thread. The operand
/// list is `(handle, preds, slot)`; the stream is the launch's, not one of
/// them, just as it was the `<<<>>>`'s.
fn arm(
    ctx: &Ctx<'_>,
    instantiation: &'static str,
    handle: usize,
    preds: *const c_void,
    slot: i32) -> Result<(), Refusal> {
    // A negative slot is `preds[-1]`, a device read off the front of the
    // predicate word — refused rather than clamped, per [`supergraph_set_switch`].
    if slot < 0 {
        return Err(Refusal::Narrow { what: "the predicate slot", at: i64::from(slot) });
    }
    // SAFETY: the caller's assertion, forwarded — `preds` addresses the live
    // predicate word and the stream outlives the launch.
    ctx.fire(Fire::at("graph/supergraph.cuh", instantiation).apply(ARM), &[handle.arg(), preds.cast::<u8>().arg(), slot.arg()])
}

#[cfg(test)]
mod tests {
    use super::{ARM, ROOT};

    /// Every `#include` this root reaches is CARRIED, so NVRTC never goes
    /// looking for a header on disk. The `#else` arm's `<cuda_runtime.h>` is
    /// not reached from here (guarded by `__CUDACC_RTC__`), so NVRTC takes
    /// the hand-declared `cudaGraphSetConditional` instead.
    #[test]
    fn every_include_the_root_reaches_is_carried() {
        let reached = crate::source::reachable(ROOT.name, ROOT.text, ROOT.header_set())
            .unwrap_or_else(|why| panic!("{}: {why}", ROOT.name));
        assert!(reached.contains(&"prelude/device.cuh"), "the prelude, at least: {reached:?}");
    }

    /// One block of one thread, and nothing shared.
    #[test]
    fn the_arming_launch_is_one_thread() {
        assert_eq!(ARM.grid, [1, 1, 1]);
        assert_eq!(ARM.block, [1, 1, 1]);
        assert_eq!(ARM.smem, 0);
        assert!(!ARM.empty());
    }
}

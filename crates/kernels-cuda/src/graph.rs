//! The supergraph's two arming kernels, and NOT a family.
//!
//! The only device text in this tree whose argument is not a tensor:
//! `cudaGraphSetConditional` writes a CONDITIONAL HANDLE, a shell object. That
//! is why `graph/` exists as a directory rather than these landing in
//! `layout/`, and it is also why there is no `FAMILY` here — a trace statement
//! names tensors, and no statement can name a handle. The driver calls these
//! two by path from inside its graph capture; nothing resolves a symbol to
//! them. [`crate::driver_internal`] is the same shape for the same reason.
//!
//! # Why the arming is a kernel at all
//!
//! Arming the handle from INSIDE the graph is the whole mechanism that lets a
//! replay take a fire's arms with no host round-trip: the kernel reads a slot
//! out of the device-resident predicate word and writes the handle, and the
//! conditional node downstream of it reads what was written. A host that armed
//! the handle would have to be told the predicate first, which is the
//! round-trip the supergraph exists to remove.
//!
//! # The handle crosses as `usize`, and that is deliberate
//!
//! `cudaGraphConditionalHandle` is `unsigned long long` (`driver_types.h`),
//! which on LP64 is a DIFFERENT type from `size_t` at the same width. The
//! device parameter is the prelude's `usize` (`decltype(sizeof(0))`), so this
//! takes a `usize` and the one conversion happens at the one call site, cast
//! explicitly.

use core::ffi::c_void;

use crate::jit::{Ctx, Launch, Root};
use crate::jit::Abi;
use kernels::Refusal;

/// `graph/supergraph.cuh` — the root these two compile a symbol out of.
pub static ROOT: Root = Root::new("graph/supergraph.cuh");

/// `csrc/supergraph.cu:61` and `:74` — `<<<1, 1, 0, stream>>>`, both
/// launchers, cited rather than derived.
///
/// One thread, because the kernel is one store to one handle, and a second
/// thread writing the same handle is the racing call CUDA calls undefined.
/// There is no extent here for any rule to have read.
const ARM: Launch = Launch::grid([1, 1, 1], [1, 1, 1]);

/// Arms `handle` from `preds[slot]` as a BOOLEAN.
///
/// `ctx`'s stream must be CAPTURING: the launch becomes the conditional node's
/// upstream dependency, and an IF node downstream reads the byte as 0/1.
///
/// # Errors
///
/// [`Refusal::Device`] if the compile, the load or the launch refused — which
/// for this kernel is usually "the stream is not capturing". The caller
/// abandons the capture and the fire runs eagerly.
///
/// # Safety
///
/// `preds` must address a live device predicate word with at least `slot + 1`
/// bytes, live across the launch, and `ctx`'s stream must outlive it.
pub fn supergraph_set_cond(
    ctx: &Ctx,
    handle: usize,
    preds: *const c_void,
    slot: i32,
) -> Result<(), Refusal> {
    arm(ctx, "::pie::graph::supergraph_set_cond", handle, preds, slot)
}

/// Arms `handle` from `preds[slot]` as a body INDEX rather than a boolean.
///
/// [`supergraph_set_cond`]'s contract. The device-side difference is the whole
/// of the switch: `cudaGraphSetConditional` takes an unsigned value, an IF
/// reads it as 0/1 and a SWITCH reads it as an arm index, so writing the
/// slot's byte through unchanged is the entire change. The predicate word
/// needs no new storage — it is already a byte per slot, and a slot holding a
/// kernel index rather than a boolean is the same byte read differently.
///
/// **An out-of-range index selects no body**, which is CUDA's rule and the one
/// thing this deliberately does NOT clamp: a fire whose predicate says "arm 4"
/// of a three-arm switch has a lowering/driver disagreement, and running arm 0
/// instead would answer with the wrong program rather than with nothing.
///
/// # Errors
///
/// [`supergraph_set_cond`]'s.
///
/// # Safety
///
/// [`supergraph_set_cond`]'s.
pub fn supergraph_set_switch(
    ctx: &Ctx,
    handle: usize,
    preds: *const c_void,
    slot: i32,
) -> Result<(), Refusal> {
    arm(ctx, "::pie::graph::supergraph_set_switch", handle, preds, slot)
}

/// Compile and load both arming kernels NOW, before any capture is open.
///
/// **These two are the only device text in this tree that is launched
/// exclusively from inside a graph capture**, and that makes them the only
/// two whose lazy JIT can never already have happened. Every other kernel
/// reaches its first launch on an ordinary stream — a warm-up fire, an
/// eager step, a preceding layer — so by the time a capture wants it, the
/// compile is a `OnceLock` read. These have no such day.
///
/// What that costs is the whole of the supergraph. The first
/// [`supergraph_set_cond`] of the process runs `jit::cache::resolve` with a
/// capture open, and resolve's first act is `bind_context`'s
/// `cudaFree(null)` — which `cudaStreamCaptureModeGlobal` **prohibits**, by
/// name, as a potentially-unsafe call. It returns an error, `resolve`
/// reports no device, the arming launch refuses, and the capture is left
/// `cudaErrorStreamCaptureInvalidated`. NVRTC and `cuModuleLoadData` sit
/// behind the same door. Measured, not reasoned: a resolve outside a
/// capture succeeds and the kernel launches; the identical resolve with a
/// capture open answers *"no CUDA device is current"* and `EndCapture` then
/// answers `cudaErrorStreamCaptureInvalidated`.
///
/// So the fix is not in the launch, it is in the CLOCK: do the compile
/// before the capture opens. Once per process per architecture — after
/// that this is two `OnceLock` reads and costs nothing, which is why the
/// caller can afford to call it on every capture rather than reasoning
/// about whether this one will branch.
///
/// Both are warmed, not just the IF form. A capture that opens a switch
/// without ever opening a conditional is a supported shape, and finding out
/// which one this capture needs would mean knowing the trace before it is
/// walked.
///
/// # Errors
///
/// [`Refusal::Device`] if either will not compile or load — the same
/// refusal the launch would have returned, only now it arrives while the
/// caller can still do something about it.
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
///
/// Not an error. A capture cannot open here either, so the condition this
/// exists to prevent cannot arise, and refusing would make every caller
/// carry a branch for a build in which the caller is unreachable.
#[cfg(not(feature = "_cuda"))]
#[allow(clippy::unnecessary_wraps)]
pub fn warm() -> Result<(), Refusal> {
    Ok(())
}

/// The body both share: bind three operands, launch one thread.
///
/// The operand list is `(handle, preds, slot)` and the stream is not one of
/// them; it is the launch's, as it was the `<<<>>>`'s.
fn arm(
    ctx: &Ctx,
    instantiation: &'static str,
    handle: usize,
    preds: *const c_void,
    slot: i32,
) -> Result<(), Refusal> {
    // A negative slot is `preds[-1]`, a device read off the front of the
    // predicate word. Refused rather than clamped to 0, for the reason
    // [`supergraph_set_switch`] gives about arm 4 of a three-arm switch.
    if slot < 0 {
        return Err(Refusal::Narrow { what: "the predicate slot", at: i64::from(slot) });
    }
    // SAFETY: the caller's assertion, forwarded — `preds` addresses the live
    // predicate word and the stream outlives the launch.
    unsafe {
        ctx.launch("graph/supergraph.cuh", instantiation, ARM, &[handle.arg(), preds.cast::<u8>().arg(), slot.arg()])
    }
}

#[cfg(test)]
mod tests {
    use super::{ARM, ROOT};

    /// The two entry points are what the deleted `graph/supergraph` unit
    /// handed NVRTC, character for character.
    ///
    /// The unit built each from a `template_path` and `DeviceKernel::PLAIN`,
    /// absolutised; a routine writes it out. Neither carries a `<...>`,
    /// because neither `__global__` is a template.
        /// Every `#include` this root reaches is CARRIED, so NVRTC never goes
    /// looking for a header on disk.
    ///
    /// `source::every_include_reachable_from_a_unit_resolves` walked the
    /// `graph/supergraph` unit for this; there is no unit to walk now. The
    /// `#else` arm's `<cuda_runtime.h>` is not reached from here and must not
    /// be: it is angle-bracketed and guarded by `__CUDACC_RTC__`, so NVRTC
    /// takes the hand-declared `cudaGraphSetConditional` instead.
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

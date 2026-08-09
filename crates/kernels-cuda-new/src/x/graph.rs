//! The supergraph's two arming kernels, and NOT a family.
//!
//! The only device text in this tree whose argument is not a tensor:
//! `cudaGraphSetConditional` writes a CONDITIONAL HANDLE, a shell object. That
//! is why `graph/` exists as a directory rather than these landing in
//! `layout/`, and it is also why there is no `FAMILY` here — a trace statement
//! names tensors, and no statement can name a handle. The driver calls these
//! two by path from inside its graph capture; nothing resolves a symbol to
//! them. [`crate::x::driver_internal`] is the same shape for the same reason.
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
use crate::x::Abi;
use kernels::Refusal;

/// `graph/supergraph.cuh` — the root these two compile a symbol out of.
pub static ROOT: Root = Root::new(
    "graph/supergraph",
    include_str!("../../csrc/src/graph/supergraph.cuh"),
    "graph/supergraph.cuh",
);

/// The entry points NVRTC is handed, spelled as it is handed them.
///
/// Neither is a template, so neither wears a `<...>`: these are the only two
/// `__global__`s in the tree a routine names by their own path.
pub mod inst {
    /// `supergraph.cuh` — arms the handle as a BOOLEAN, for an IF node.
    pub const SET_COND: &str = "::pie_cuda_driver::kernels::graph::device::supergraph_set_cond";
    /// The same, read as a body INDEX by a SWITCH node.
    pub const SET_SWITCH: &str = "::pie_cuda_driver::kernels::graph::device::supergraph_set_switch";
}

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
    arm(ctx, inst::SET_COND, handle, preds, slot)
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
    arm(ctx, inst::SET_SWITCH, handle, preds, slot)
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
        ctx.launch(&ROOT, instantiation, ARM, &[handle.arg(), preds.cast::<u8>().arg(), slot.arg()])
    }
}

#[cfg(test)]
mod tests {
    use super::{ARM, ROOT, inst};

    /// The two entry points are what the deleted `graph/supergraph` unit
    /// handed NVRTC, character for character.
    ///
    /// The unit built each from a `template_path` and `DeviceKernel::PLAIN`,
    /// absolutised; a routine writes it out. Neither carries a `<...>`,
    /// because neither `__global__` is a template.
    #[test]
    fn the_instantiations_are_what_the_unit_asked_for() {
        assert_eq!(
            inst::SET_COND,
            "::pie_cuda_driver::kernels::graph::device::supergraph_set_cond"
        );
        assert_eq!(
            inst::SET_SWITCH,
            "::pie_cuda_driver::kernels::graph::device::supergraph_set_switch"
        );
        assert_ne!(
            ROOT.key(inst::SET_COND, "sm_90"),
            ROOT.key(inst::SET_SWITCH, "sm_90"),
            "one root, two symbols, two cubins"
        );
    }

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
        assert!(reached.contains(&"pie_device.cuh"), "the prelude, at least: {reached:?}");
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

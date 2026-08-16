//! `csrc/supergraph.cu`'s two launchers, in Rust; device text lives in
//! `graph/supergraph.cuh`.

use std::ffi::c_void;

use kernels_cuda::jit::Ctx;
use kernels_cuda::{Refusal, graph};

use crate::error::{Error, Result};

const SET_COND: &str = "graph::supergraph_set_cond";

const SET_SWITCH: &str = "graph::supergraph_set_switch";

/// Arms `handle` from `preds[slot]` on `stream` (must be capturing). `slot`
/// indexes the predicate word, read downstream as 0/1; errors if refused.
pub fn set_cond(handle: u64, preds: *const u8, slot: u32, stream: *mut c_void) -> Result<()> {
    arm(SET_COND, graph::supergraph_set_cond, handle, preds, slot, stream)
}

/// Arms `handle` from `preds[slot]` as a body index, not a boolean. An
/// out-of-range index selects no body — deliberately not clamped to 0.
pub fn set_switch(handle: u64, preds: *const u8, slot: u32, stream: *mut c_void) -> Result<()> {
    arm(SET_SWITCH, graph::supergraph_set_switch, handle, preds, slot, stream)
}

/// Compiles and loads both arming kernels before any capture is opened. Not
/// fatal alone — a straight-line capture needs no conditional to be armed.
pub fn warm() -> Result<()> {
    graph::warm().map_err(|why| Error::invalid("graph::warm", format!("{why:?}")))
}

/// The body both launchers share. `name` survives only so a refusal says which.
#[allow(clippy::not_unsafe_ptr_arg_deref)] // both pointers are the caller's, never read here
fn arm(
    name: &'static str,
    routine: fn(&Ctx, usize, *const c_void, i32) -> std::result::Result<(), Refusal>,
    handle: u64,
    preds: *const u8,
    slot: u32,
    stream: *mut c_void,
) -> Result<()> {
    // A slot that doesn't fit an `int` is refused, not folded to 0 (see `set_switch`).
    let Ok(slot) = i32::try_from(slot) else {
        return Err(Error::invalid(name, "pred slot does not fit an int"));
    };
    let handle = handle as usize;
    // SAFETY: caller keeps the capturing stream live; `preds` is device-resident.
    let ctx = unsafe { Ctx::on(stream) };
    routine(&ctx, handle, preds.cast::<c_void>(), slot)
        .map_err(|why| Error::invalid(name, format!("the arming launch failed: {why:?}")))
}

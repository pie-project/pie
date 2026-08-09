//! `csrc/supergraph.cu`'s two launchers, in Rust — and with them the whole
//! file and the second of this crate's three nvcc builds.
//!
//! The device text is `kernels-cuda-new/csrc/src/graph/supergraph.cuh`,
//! compiled by NVRTC like every other unit. It stayed OUT of `kernels-cuda`
//! for the reason `device/graph.rs`'s header gives and this port keeps: its
//! argument is a conditional handle, a SHELL object, not a tensor. It is its
//! own family, `graph`, for the same reason — `layout` and `attn` are named
//! after kinds of value, and a handle is not one.
//!
//! # The claim this deletes
//!
//! `supergraph.cu` opened with *"the one device function the supergraph
//! cannot express in Rust"*, and `build.rs` gave it *"its own archive … this
//! needs nvcc"*. Both were wrong, and the second was the load-bearing one:
//! `cudaGraphSetConditional` is a `__device__` function, which is a fact about
//! where it RUNS, not about which frontend may emit a call to it.
//!
//! Measured on this box (L40S, sm_89, CUDA 13.0), against `libnvrtc.so.13`
//! and `libcuda.so.1`:
//!
//! | step | result |
//! |---|---|
//! | NVRTC compiles a kernel calling `cudaGraphSetConditional` | rc=0 |
//! | the emitted PTX | `.extern .func cudaGraphSetConditional` + `call.uni` |
//! | `nvrtcGetCUBIN` at `--gpu-architecture=sm_89` | 3,624 bytes; `nm` shows `U cudaGraphSetConditional`; SASS `CALL.ABS.NOINC` |
//! | `cuModuleLoadData` on that PTX | OK — the driver resolves the symbol at module load |
//! | `cuModuleGetFunction` | OK |
//! | capture via `cuStreamBeginCaptureToGraph` + `cuLaunchKernel` | OK, one node |
//! | `cuGraphAddNode` of a `CU_GRAPH_NODE_TYPE_CONDITIONAL` IF on that handle | OK |
//!
//! **What that does not show**, stated here so nobody reads more into it: no
//! conditional graph was executed end to end. The probe stalled populating the
//! IF *body* — `cuGraphAddKernelNode_v2` and `cuStreamBeginCaptureToGraph`
//! both refused with `invalid argument` while the parent was mid-capture —
//! and that is the probe's plumbing, not NVRTC's: the sequence that works is
//! the one [`crate::device::SupergraphBuilder`] has been running all along.
//!
//! What closes the gap is the symbol itself. `cudaGraphSetConditional` is
//! declared `extern __device__ __cudart_builtin__` in
//! `cuda_device_runtime_api.h` with **no definition in any toolkit header**,
//! and it is **not in `libcudadevrt.a`**. A call to an extern device function
//! that no linkable library defines can only be resolved by the driver at
//! module load, whichever frontend emitted it — and NVRTC and nvcc share that
//! frontend (`cicc`). nvcc's PTX for this call was the same `.extern .func`.
//! There was no nvcc-only lowering to lose.
//!
//! # Why the arming kernel exists at all
//!
//! Arming a conditional handle from INSIDE the graph is the whole mechanism
//! that lets a replay take a fire's arms with no host round-trip: the kernel
//! reads a slot out of the device-resident predicate word and writes the
//! handle, and the conditional node downstream of it reads what was written.
//! A host that decided the branch would have to read the predicate back,
//! which is the synchronisation a captured graph exists to avoid.
//!
//! # Why these return a `Result` and not a `bool`
//!
//! `hand::fire` panics on every failure, and it is right to: a caller that
//! reached it has already decided it will launch, and no ahead-of-time
//! launcher is left to fall back to. Here the two halves of that argument
//! come apart, so this module splits them:
//!
//! * **Resolution and compilation panic**, with the symbol named —
//!   `fire/gemv.rs`'s rule. A missing unit, a missing row, a unit that will
//!   not compile, an operand list that disagrees with its signature: each is
//!   drift between this driver and its kernel table.
//! * **The launch itself is refused, not fatal.** That is the one failure the
//!   C++ reported by returning `cudaGetLastError()` as an int, and
//!   [`open_cond`] has always turned a non-zero into an [`Error`] and let the
//!   capture be abandoned. `fire/launch.rs` says it outright — *"a refused
//!   capture is not a refused fire"* — and the supergraph is an optimisation,
//!   so a launch this stream cannot take must stay recoverable.
//!
//! The split matters in one direction only. If a broken table were reported
//! as a refused launch, every capture would abandon and every fire would run
//! eagerly, silently and forever, with no diagnostic anywhere — the exact
//! failure `hand`'s header calls *"the one diagnosis that sends a reader to
//! the wrong file"*.
//!
//! The C++'s comment explaining the `int` return — *"the C++ original throws
//! out of `CUDA_CHECK`, which is not a shape that crosses `extern \"C\"`"* —
//! is not ported. There is no `extern "C"` left for it to be about.
//!
//! [`open_cond`]: crate::device::SupergraphBuilder::open_cond
//! [`Error`]: crate::error::Error

use std::ffi::c_void;

use kernels_cuda_new::runtime::{ArgValue, Args, Launch, Stream, cache};

use crate::error::{Error, Result};

/// `graph/supergraph.cuh`'s IF-arming kernel.
const SET_COND: &str = "graph::supergraph_set_cond";

/// `graph/supergraph.cuh`'s SWITCH-arming kernel.
const SET_SWITCH: &str = "graph::supergraph_set_switch";

/// `csrc/supergraph.cu:61` and `:74` — `<<<1, 1, 0, stream>>>`, both
/// launchers, cited rather than derived.
///
/// One thread, because the kernel is one store to one handle. No
/// [`kernels::LaunchRule`] states this and none should: a rule reads a
/// [`kernels_cuda_new::Dims`], and there is no extent here to read. The row
/// says [`kernels::LaunchRule::Unstated`] and the geometry is the driver's,
/// which is what `fire/attn_score.rs` and `fire/split_packed.rs` each did with
/// a geometry no rule states.
const ARM: Launch = Launch {
    grid: [1, 1, 1],
    block: [1, 1, 1],
    smem: 0,
};

/// Arms `handle` from `preds[slot]` on `stream`, which must be capturing —
/// the launch becomes the conditional node's upstream dependency.
///
/// `slot` is a predicate-word slot index; the kernel reads one byte from it
/// and an IF node downstream reads that as 0/1.
///
/// # Errors
///
/// If the launch is refused — the stream is not capturing, or CUDA declines
/// it. The caller abandons the capture and the fire runs eagerly.
///
/// # Panics
///
/// If this driver and its kernel table disagree, or the `graph/supergraph`
/// unit will not compile or load. See the module header for the split.
pub fn set_cond(handle: u64, preds: *const u8, slot: u32, stream: *mut c_void) -> Result<()> {
    arm(SET_COND, handle, preds, slot, stream)
}

/// Arms `handle` from `preds[slot]` as a body INDEX rather than a boolean.
///
/// Same contract as [`set_cond`]: `stream` must be capturing. The device-side
/// difference is the whole of the switch (`.wiki/driver/graph.md` §6.1) —
/// `cudaGraphSetConditional` takes an unsigned value, an IF reads it as 0/1,
/// and a SWITCH reads it as an arm index, so writing the slot's byte through
/// unchanged is the entire change. The predicate word needs no new storage:
/// it is already a byte per slot, and a slot holding a kernel index rather
/// than a boolean is the same byte read differently.
///
/// **An out-of-range index selects no body**, which is CUDA's rule and the one
/// thing this deliberately does NOT clamp: a fire whose predicate says "arm 4"
/// of a three-arm switch has a lowering/driver disagreement, and running arm 0
/// instead would answer with the wrong program rather than with nothing.
///
/// # Errors
///
/// [`set_cond`]'s.
///
/// # Panics
///
/// [`set_cond`]'s.
pub fn set_switch(handle: u64, preds: *const u8, slot: u32, stream: *mut c_void) -> Result<()> {
    arm(SET_SWITCH, handle, preds, slot, stream)
}

/// The body both launchers share: resolve the row, bind three operands,
/// launch one thread.
///
/// Not [`crate::fire::hand::fire`], and the difference is the last line — that
/// function panics on the launch, this one refuses. The four steps above it
/// are the same and in the same order, which is the contract `hand`'s header
/// states.
///
/// The operand list is `(handle, preds, slot)` and the stream is not one of
/// them; it is the launch's, as it was the `<<<>>>`'s.
#[allow(clippy::not_unsafe_ptr_arg_deref)] // both pointers are the caller's, never read here
fn arm(
    symbol: &'static str,
    handle: u64,
    preds: *const u8,
    slot: u32,
    stream: *mut c_void,
) -> Result<()> {
    let Some((index, unit)) = kernels_cuda_new::unit::unit_of(symbol) else {
        panic!("{symbol} is in no JIT unit — this driver and its kernel table disagree");
    };
    let Some(sig) = unit.row(symbol).map(|row| row.sig) else {
        panic!("{symbol} named unit `{}` and is not one of its rows", unit.name);
    };
    let module = match cache::module(index, unit) {
        Ok(module) => module,
        Err(why) => panic!("{symbol}: unit `{}` would not compile or load: {why}", unit.name),
    };
    // A slot index that does not fit an `int` is a lowering that named a slot
    // no predicate word has — refused rather than folded to 0, for the reason
    // `set_switch` gives about arm 4 of a three-arm switch. The `extern "C"`
    // seam this replaces wrote `i32::try_from(pred_slot).unwrap_or(0)`.
    let Ok(slot) = i32::try_from(slot) else {
        return Err(Error::invalid(symbol, "pred slot does not fit an int"));
    };
    let values = [
        ArgValue::Usize(handle as usize),
        ArgValue::Ptr(preds.cast::<c_void>().cast_mut()),
        ArgValue::I32(slot),
    ];
    let mut args = match Args::bind(sig, &values) {
        Ok(args) => args,
        Err(why) => panic!("{symbol}: {why}"),
    };
    // SAFETY: the caller holds the capturing stream live across the launch —
    // the same assertion `SupergraphBuilder` made when it handed the raw
    // stream to a C++ launcher that put it in a `<<<>>>`.
    let stream = unsafe { Stream::from_runtime(stream) };
    module
        .fire(sig, ARM, &mut args, stream)
        .map_err(|why| Error::invalid(symbol, format!("the arming launch failed: {why}")))
}

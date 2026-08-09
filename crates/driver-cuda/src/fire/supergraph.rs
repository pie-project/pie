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
//! launcher is left to fall back to. Here that is wrong in one direction —
//! **a launch this stream cannot take must stay recoverable.** That is the one
//! failure the C++ reported by returning `cudaGetLastError()` as an int, and
//! [`open_cond`] has always turned a non-zero into an [`Error`] and let the
//! capture be abandoned. `fire/launch.rs` says it outright: *"a refused
//! capture is not a refused fire"*, and the supergraph is an optimisation.
//!
//! This module used to SPLIT that from table drift, panicking when a unit was
//! missing or would not compile and refusing only the launch. **The split is
//! gone with the unit**: [`kernels_cuda_new::x::graph`] is two `fn`s the
//! compiler resolves, so there is no table left to drift from, and what
//! remains — the compile, the load and the launch — comes back as one
//! `Refusal`. The worry the split answered was that a broken table would be
//! reported as a refused launch and every capture would abandon *"with no
//! diagnostic anywhere"*; `jit::Ctx::launch` logs the compiler's or the
//! driver's own words at `error!` once per instantiation, which is the
//! diagnostic that argument was asking for.
//!
//! The C++'s comment explaining the `int` return — *"the C++ original throws
//! out of `CUDA_CHECK`, which is not a shape that crosses `extern \"C\"`"* —
//! is not ported. There is no `extern "C"` left for it to be about.
//!
//! [`open_cond`]: crate::device::SupergraphBuilder::open_cond
//! [`Error`]: crate::error::Error

use std::ffi::c_void;

use kernels_cuda_new::jit::Ctx;
use kernels_cuda_new::x::{Refusal, graph};

use crate::error::{Error, Result};

/// `graph/supergraph.cuh`'s IF-arming kernel.
const SET_COND: &str = "graph::supergraph_set_cond";

/// `graph/supergraph.cuh`'s SWITCH-arming kernel.
const SET_SWITCH: &str = "graph::supergraph_set_switch";

/// Arms `handle` from `preds[slot]` on `stream`, which must be capturing —
/// the launch becomes the conditional node's upstream dependency.
///
/// `slot` is a predicate-word slot index; the kernel reads one byte from it
/// and an IF node downstream reads that as 0/1.
///
/// # Errors
///
/// If the launch is refused — the stream is not capturing, CUDA declines it,
/// or the root will not compile or load. The caller abandons the capture and
/// the fire runs eagerly.
pub fn set_cond(handle: u64, preds: *const u8, slot: u32, stream: *mut c_void) -> Result<()> {
    arm(SET_COND, graph::supergraph_set_cond, handle, preds, slot, stream)
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
pub fn set_switch(handle: u64, preds: *const u8, slot: u32, stream: *mut c_void) -> Result<()> {
    arm(SET_SWITCH, graph::supergraph_set_switch, handle, preds, slot, stream)
}

/// The body both launchers share: the two conversions the driver's vocabulary
/// needs, then the routine.
///
/// `routine` is the `fn` itself and not a symbol, which is what the descent
/// bought: the resolution `symbol` used to drive is the compiler's now, and
/// `name` survives only so a refusal says which of the two was refused.
///
/// The stream is not one of the routine's arguments; it is the launch's, as it
/// was the `<<<>>>`'s.
#[allow(clippy::not_unsafe_ptr_arg_deref)] // both pointers are the caller's, never read here
fn arm(
    name: &'static str,
    routine: fn(&Ctx, usize, *const c_void, i32) -> std::result::Result<(), Refusal>,
    handle: u64,
    preds: *const u8,
    slot: u32,
    stream: *mut c_void,
) -> Result<()> {
    // A slot index that does not fit an `int` is a lowering that named a slot
    // no predicate word has — refused rather than folded to 0, for the reason
    // `set_switch` gives about arm 4 of a three-arm switch. The `extern "C"`
    // seam this replaces wrote `i32::try_from(pred_slot).unwrap_or(0)`.
    let Ok(slot) = i32::try_from(slot) else {
        return Err(Error::invalid(name, "pred slot does not fit an int"));
    };
    // `cudaGraphConditionalHandle` is `unsigned long long` and the device
    // parameter is the prelude's `usize`; the two are the same width and a
    // different type, so the cast is written once, here.
    let handle = handle as usize;
    // SAFETY: the caller holds the capturing stream live across the launch,
    // and `preds` addresses the device-resident predicate word — the same
    // assertion `SupergraphBuilder` made when it handed the raw stream to a
    // C++ launcher that put it in a `<<<>>>`.
    let ctx = unsafe { Ctx::on(stream) };
    routine(&ctx, handle, preds.cast::<c_void>(), slot)
        .map_err(|why| Error::invalid(name, format!("the arming launch failed: {why:?}")))
}

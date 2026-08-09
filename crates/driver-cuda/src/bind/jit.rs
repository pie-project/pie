//! Firing a row through the kernel `kernels-cuda-new` compiled it into.
//!
//! **This is the switch, and it now points at the other crate.** Every kernel
//! launch in this driver went the same way since the C++ was ported: a
//! generated `match` arm calls a generated `extern "C" pie_k_*`, which calls a
//! host launcher in a `.cu`, which holds the `<<<>>>`. Tier A built the other
//! path here — NVRTC compiles the template, `nvrtcGetLoweredName` gives the
//! mangled symbol, `cuLaunchKernel` takes a `void**` — and
//! `new-horizon.md` §12 rebuilt it as a crate: 38 units, 135 rows, a global
//! module cache and a typed façade, proved on an L40S and consumed by nobody.
//! This file is what stopped that from being true.
//!
//! A row named by `kernels_cuda_new::device::JIT_DISPATCHED` comes here, and
//! here is now a translation onto [`kernels_cuda_new::fire`].
//!
//! # What this file stopped being, and why that matters more than its size
//!
//! It held a module cache: one `OnceLock` per unit of `bind::nvrtc::UNITS`, an
//! `arch()` discovered once, a `load` that compiled on first fire. All of it
//! is deleted, because `kernels_cuda_new::runtime::cache` holds exactly the
//! same thing for the same units — and two caches of one unit is not merely
//! twice the compile and twice the `CUmodule`. The cache key spans the NVRTC
//! options (§12.3), so two caches are two option sets, and the same row could
//! then be launched from cubins that disagree about the float flags depending
//! on which half of the tree called it. That is a numerics difference with no
//! symptom and no diff.
//!
//! What is left is the seam, and a seam is worth a file:
//!
//! * **the vocabulary.** The generated arm speaks this driver's [`ArgValue`]
//!   and hands a `cudaStream_t` as a `*mut c_void`; the JIT crate speaks its
//!   own `ArgValue` (which has two variants this driver's does not) and takes
//!   a borrowed [`Stream`]. Translating in the emitter would mean
//!   `kernels-cuda`'s generator naming a feature-gated module of a crate it
//!   does not depend on, once per arm.
//! * **the lifetime assertion.** `Stream::from_runtime` is `unsafe` because it
//!   claims the handle outlives the launch. That claim is made once, here,
//!   against the same `ctx.stream` the shim path passes to a `pie_k_*` entry.
//! * **the answer shape.** `fire` returns a `Result`, and the dispatcher needs
//!   "handled". See below for why the translation has no `false` in it.
//!
//! # Why a failure is a refusal and not a fallback
//!
//! A row that reaches [`fire`] has no shim entry any more. If its unit will
//! not compile or its symbol will not resolve, there is nothing to fall back
//! *to*, and a silent `false` would send the fire to a hand-written arm that
//! does not exist and then to `UnknownKernel` — a diagnosis of the wrong
//! thing, at the wrong layer, about a kernel that is right there in the table.
//!
//! So this function returns `()`. Not a `bool` that is always `true`: the
//! generated arm ends in `true` because there is no other answer it could
//! give, and the value that would have to exist for a fire to be misrouted is
//! not in the type. `kernels_cuda_new::fire` reports every refusal once per
//! symbol at `error` with the unit named, which is the half an operator reads;
//! the half a caller reads is that the launch did not happen and nothing else
//! was tried.
//!
//! # Where the compile happens
//!
//! On first fire, still, and it is still a stall in the wrong place — but it
//! is now `runtime::cache`'s stall rather than this file's, and
//! `runtime::cache::warm` exists to move it. §6.4 of `new-horizon.md` wants
//! one compile per unit at load and §6.6 wants a cubin cache on disk; neither
//! changes this file, which is the argument for the seam being here.

use std::sync::OnceLock;

use kernels_cuda_new::{Dims, Error, Stream};

use super::device::ArgValue;

/// Fire `symbol` through the unit `kernels-cuda-new` compiles it out of.
///
/// `dims` is the fire's rectangle and its geometry — the nine axes every
/// [`kernels::LaunchRule`] is written over. The three the statement states
/// come from the launch's own operands and the six that describe the model
/// come from the fire's context; `bind::dispatch_generated`'s `jit_dims` is
/// where that split is spelled out, because the driver is what knows where it
/// keeps its own geometry.
///
/// `values` are the row's operands in the row's order, in this driver's
/// vocabulary, checked against the row by the JIT crate's `Args::bind` rather
/// than trusted here — the translation below changes the spelling of a value
/// and never its meaning.
///
/// Returns nothing, deliberately. See the module header: a routed row has no
/// other path, so "not mine" is not an answer this can give.
///
/// # Safety
///
/// `stream` must be a live `cudaStream_t` for the duration of the launch, and
/// every [`ArgValue::Ptr`] must address device memory live and large enough
/// for the operand the row states. The same assertion the shim path makes when
/// it passes `ctx.stream` and a run of arena offsets to a `pie_k_*` entry —
/// the launch is asynchronous, so "for the duration" ends when the stream is
/// synchronised and not when this returns.
pub unsafe fn fire(symbol: &str, dims: Dims, values: &[ArgValue], stream: *mut std::ffi::c_void) {
    // One allocation per fire, sized exactly. The alternative — a fixed stack
    // array — buys back a malloc against a launch that costs microseconds, and
    // pays for it with an arity ceiling: the widest rows in the JIT table take
    // twenty-odd operands, and a row that outgrew the array would be refused
    // for being wide rather than for being wrong.
    let mut translated = Vec::with_capacity(values.len());
    translated.extend(values.iter().copied().map(translate));

    // SAFETY: the caller asserts `stream` is live for this launch, exactly as
    // the shim path does when it hands `ctx.stream` to a `pie_k_*` entry.
    // `from_runtime` is the cast between the two APIs' typedefs for one
    // object, which is what lets a process that creates streams with
    // `cudaStreamCreateWithPriority` order work on them with `cuLaunchKernel`.
    let stream = unsafe { Stream::from_runtime(stream) };
    // SAFETY: the pointer obligation above is the caller's, unchanged by the
    // translation — an `ArgValue::Ptr` crosses as the same address it arrived
    // as.
    let outcome = unsafe { kernels_cuda_new::fire(symbol, dims, &translated, stream) };
    if let Err(Error::Unknown { .. }) = outcome {
        // THE ONE REFUSAL THE JIT CRATE DOES NOT REPORT, because for it "no
        // unit hosts this" is the cheap answer a dispatcher with somewhere
        // else to look wants. This dispatcher has nowhere else to look: the
        // arm exists because `driver-cuda`'s build script found the symbol in
        // that crate's table, so hearing it now is drift between a build and
        // its own binary, and it is the one thing worth saying twice.
        report(symbol);
    }
}

/// This driver's [`ArgValue`] in the JIT crate's spelling.
///
/// Total, and it stays total by being a `match` with no wildcard: this
/// driver's enum is the older and narrower of the two — the JIT crate's adds
/// `I64` and `Bool` for the batched SSM rows — so every variant here has a
/// twin there, and a new variant on either side is a compile error rather than
/// a value that arrives as the wrong width.
///
/// `U8` was added to BOTH, in the same change, for the `attention_naive_paged`
/// rows: a `device::KvScheme` is one byte and there was no kind that crossed
/// as one. This `match` is why the addition could not be half-done — leaving
/// it out here is a build error and not a launch that binds eight bytes where
/// the cubin declares one.
const fn translate(value: ArgValue) -> kernels_cuda_new::ArgValue {
    match value {
        ArgValue::Ptr(p) => kernels_cuda_new::ArgValue::Ptr(p),
        ArgValue::I32(v) => kernels_cuda_new::ArgValue::I32(v),
        ArgValue::U32(v) => kernels_cuda_new::ArgValue::U32(v),
        ArgValue::F32(v) => kernels_cuda_new::ArgValue::F32(v),
        ArgValue::Usize(v) => kernels_cuda_new::ArgValue::Usize(v),
        ArgValue::U8(v) => kernels_cuda_new::ArgValue::U8(v),
    }
}

/// Say that a routed symbol is not in the table it was routed by, once.
///
/// Once, because a fire is per layer per token and a broken row would
/// otherwise produce a line per launch — which is how a real diagnosis becomes
/// unreadable. Every other refusal is reported by the JIT crate itself, with
/// the unit named and the compiler's own words, which this side cannot improve
/// on.
fn report(symbol: &str) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SAID: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    let said = SAID.get_or_init(|| Mutex::new(HashSet::new()));
    if let Ok(mut said) = said.lock()
        && said.insert(symbol.to_string())
    {
        tracing::error!(
            symbol,
            "a routed row reached the JIT and no unit hosts it -- the dispatcher was \
             generated from a table this binary does not have"
        );
    }
}

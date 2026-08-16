//! The NCCL collectives — declared, and refused, because no NCCL bindings
//! are generated and nothing in this tree calls a communicator.
//!
//! # Why a module of refusals is the honest shape
//!
//! Three symbols a tensor-parallel model text states, and nothing in this
//! tree implements any of them. That combination is not a gap to be papered
//! over; it is a fact with two halves, and both need saying:
//!
//! * **A lowered model text may name them.** `dsl::cuda` records all three,
//!   and `mistral_7b_v03.cuda.tp2.decode` is a real sharded trace that fires
//!   two of them 32 times each. `model-compiler`'s `check_plan` refuses a
//!   model at LOAD whose launched symbol is undeclared — so if these are not
//!   declared, a TP model text is rejected with *"no such symbol"*, which
//!   names the wrong problem. The symbol is fine. There are no bindings.
//! * **This build cannot run them.** `cudarc` is depended on with
//!   `default-features = false` and an explicit feature list — `std`,
//!   `driver`, `runtime`, `nvrtc`, `cublas`, `cublaslt` — which does **not**
//!   include `nccl`, so no `ncclCommInitRank` or `ncclAllReduce` binding is
//!   generated for any body here to call. Nothing else in the workspace names
//!   an NCCL symbol either. These bodies are the backstop under
//!   `driver-cuda`'s `serve::load` refusal, not the place it is normally
//!   made.
//!
//! # These three are the ABOVE-THRESHOLD path
//!
//! A sharded model text picks between two spellings by message size.
//! [`crate::comm`]'s P2P kernel wins on LATENCY below a crossover — 1 MiB at
//! world size 4, 256 KiB above it, and the plane's whole `max_bytes` at world
//! size 2 — and NCCL wins on BANDWIDTH above it. That is what
//! `CustomAllReduce::can_handle` reports with `Decline::AboveCrossover`, and
//! its doc says in as many words that such a decline is *"the caller's cue to
//! fall back to `ncclAllReduce`"*. **These are that fallback.**
//!
//! So a refusal here is not a duplicate of `comm`'s. `comm` declines because
//! the repository does not vendor the header its launcher lives in;
//! `dist` refuses because the build generates no bindings to a communicator
//! at all. The two absences are independent, and closing either one alone
//! leaves tensor parallelism working over a different half of the message
//! sizes.
//!
//! They were three rows in `not_yet_crossed.rs`, hand-stating their columns.
//! A `fn` that refuses states the same thing and derives them, and it puts
//! the reason at the point of fire rather than in a table a fire never reads.
//!
//! # A declined implementation is still an implementation
//!
//! This is not a stub pretending to be a kernel. `Refusal::Absent` is the
//! vocabulary's own word for *"this build has no answer for that"*, and
//! `attn::kv_paged::write_kv_to_pages` already ends one of its arms with
//! `Err(Refusal::Absent { what: "a quantised writer for Native storage" })`
//! — a real host program declining a case it does not cover. The difference
//! here is one of degree: every case is uncovered, because the dependency is
//! absent rather than the arm.
//!
//! The day NCCL is linked, these bodies fill in and nothing else moves: the
//! symbols keep their names, the rows keep deriving, and the callers keep
//! calling. That is the property a hand-written row could not have.
//!
//! # The columns, and why every one of them is `whole`
//!
//! Recovered from `9e3936fb9^` with the rest, and pinned by
//! `tests/stated_columns.rs`. The reason is stronger than "a reduction is
//! over the whole value": **every rank must enter the same collective the
//! same number of times.** A row window that split one rank's launch and not
//! another's would DEADLOCK rather than compute a wrong answer, so the
//! refusal is not an optimisation and `Uncovered::WholeKernelSplit` is the
//! diagnosis that has to fire. They are also synchronisation points, which
//! the graph-capture rules have to know.

use crate::jit::Ctx;
use kernels::Refusal;

/// What every body here says, in one place so three refusals cannot drift
/// into three different accounts of one absence.
///
/// # The sentence this replaced was wrong twice
///
/// It read *"NCCL: this build links no communicator"*. `driver-cuda`'s
/// `build.rs` emitted `cargo:rustc-link-lib=nccl` under the `abi` feature, so
/// the build DID name the library — and named it uselessly, since no symbol
/// in the workspace resolved out of it (`--as-needed` dropped it from the
/// artifact's `NEEDED` list, leaving only a link-time requirement for a
/// `libnccl.so` nothing called). That flag is gone. What was never true is
/// the implication a reader takes from "links no communicator": that linking
/// one would be enough. It would not — `cudarc`'s `nccl` feature is off, so
/// there is no `ncclAllReduce` to call in the first place.
///
/// It also said `serve::load` refuses `tp_size > 1` *"for the same reason"*.
/// It does not, and never did: the driver's gate is
/// `kernels_cuda::comm::CAN_LAUNCH`, which has nothing to do with NCCL. That
/// constant was `false` while `flashinfer/comm/` was unvendored; both headers
/// are internalised now and it is `true`, so `tp_size > 1` is no longer
/// refused on it at all. These three are the above-crossover arm a sharded
/// model text picks when the message is large, and they still refuse — which
/// makes them the one remaining hole in tensor parallelism rather than the
/// symptom of a larger one.
fn no_nccl(what: &'static str) -> Refusal {
    let _ = what;
    Refusal::Absent {
        what: "NCCL: `cudarc` is built without its `nccl` feature, so no \
               communicator binding is generated and nothing in this \
               workspace calls one. This is the ABOVE-CROSSOVER arm; below \
               the crossover `comm::all_reduce_bf16` is the one that runs",
    }
}

/// `dist::all_reduce_bf16` — the in-place sum across the group.
///
/// `in_place = &[(0, 0)]`: the buffer is read and written, which is what
/// distinguishes it from [`all_reduce_bf16_out`] below and is the whole of
/// the difference between the two rows.
///
/// # Errors
///
/// Always. See the module header.
pub fn all_reduce_bf16(_ctx: &Ctx, _buf: *mut core::ffi::c_void, _elems: i64) -> Result<(), Refusal> {
    Err(no_nccl("all_reduce"))
}

/// `dist::all_reduce_bf16_out` — the same collective, a separate
/// destination, and no alias pair. That absence is the whole difference from
/// the row above.
///
/// # Errors
///
/// Always. See the module header.
pub fn all_reduce_bf16_out(
    _ctx: &Ctx,
    _src: *const core::ffi::c_void,
    _dst: *mut core::ffi::c_void,
    _elems: i64,
) -> Result<(), Refusal> {
    Err(no_nccl("all_reduce_out"))
}

/// `dist::all_gather_bf16` — each rank's shard concatenated on every rank.
///
/// # Errors
///
/// Always. See the module header.
pub fn all_gather_bf16(
    _ctx: &Ctx,
    _src: *const core::ffi::c_void,
    _dst: *mut core::ffi::c_void,
    _elems_per_rank: i64,
) -> Result<(), Refusal> {
    Err(no_nccl("all_gather"))
}

/// The three symbols, declared so a TP model text resolves rather than being
/// refused for the wrong reason.
///
/// `driver_bound!` and not `routine!`: a collective takes a COMMUNICATOR,
/// which is a property of the deployment's process group and something no
/// trace statement carries or could. That is true of the implementation this
/// module does not have, so it stays true when it arrives.
pub static ROUTINES: &[crate::jit::Routine] = &[
    crate::driver_bound!(all_reduce_bf16, whole, in_place = &[(0, 0)]),
    crate::driver_bound!(all_reduce_bf16_out, whole),
    crate::driver_bound!(all_gather_bf16, whole),
];

/// `dist`, as a trace names it.
pub static FAMILY: crate::jit::Family = crate::family!(ROUTINES);

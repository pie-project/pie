//! The thirteen `pie_cuda_*` exports — the cutover's door (retirement
//! plan phase D).
//!
//! The engine consumes a driver through `pie_driver_abi.h`, whose Rust
//! source of truth is `driver_api::local`. This module DEFINES the symbols
//! that crate declares; a test resolving the declaration against these
//! definitions makes the linker prove the contract, the same way the
//! launch bridge's shim makes the C++ compiler prove the rows.
//!
//! **One provider per binary.** The C++ shell this replaced exported the
//! same thirteen names, so the `abi` feature could never be enabled in a
//! build that also linked it — same symbols, duplicate-definition link
//! error, by design rather than by accident. There is one provider now,
//! and the rule survives it: nothing else may define these.
//!
//! # What is in here, and what is next door
//!
//! This file is the DOOR: the thirteen entry points, the panic boundary
//! every one of them crosses (`guard`), and the pointer validation they
//! share (`checked`). The bodies live beside it, one module per thing the
//! driver does:
//!
//! * [`state`] — the shell's nouns. A leaf; everything else calls in.
//! * [`load`] — create, destroy, and the once-per-model work.
//! * [`launch`] — the forward path. `.wiki/driver/graph.md` is about it.
//! * [`encode`] — the multimodal towers, run outside a fire.
//! * [`transfer`] — KV/state copies and pool resizes.
//!
//! It was one 6,364-line file, of which the doors were 18%. The split is
//! by what a caller is asking for, which is also how the work divides:
//! `launch` is `graph.md`'s subject and `state` is nobody's, so they can
//! be edited without meeting.

// Error paths print to stderr with the C++ shell's own prefix — that IS
// the behaviour being replaced (`abi.cpp` writes `[pie-driver-cuda]` to
// cerr), and an ABI boundary has no tracing subscriber to rely on.
#![allow(clippy::print_stderr)]

// Every export takes raw pointers by C-ABI necessity and null-checks them
// before the deref — the same defensive shape the C++ shell has. The
// caller-side contract is `driver_api::local`'s `unsafe extern` block;
// marking the DEFINITIONS `unsafe fn` would change their ABI type for a
// fact the boundary already states.
#![allow(clippy::not_unsafe_ptr_arg_deref)]
use driver_api::local::{
    PIE_DRIVER_ABI_VERSION,
    PIE_STATUS_DRIVER_ERROR,
    PIE_STATUS_EXHAUSTED,
    PIE_STATUS_INVALID_ARGUMENT,
    PIE_STATUS_OK,
    PieChannelDesc,
    PieChannelEndpointBinding,
    PieCompletion,
    PieDriver,
    PieDriverCaps,
    PieDriverCreateDesc,
    PieFrameDesc,
    PieInstanceBinding,
    PieInstanceDesc,
    PieModelLoadDesc,
    PieProgramDesc,
};

/// THE ONE PLACE an `Error` becomes an ABI status, and the only place
/// that logs one.
///
/// Every layer below this returns [`crate::Error`], which carries what
/// happened and how much of it. The status the engine sees is a
/// three-bit summary of that, so the conversion is also the last
/// moment the detail exists — which is why the diagnostic is emitted
/// HERE rather than at the site that failed.
///
/// That reversal is the point of §3.4. Before it, diagnosis and cause
/// travelled on separate channels: the reason went to stderr from
/// wherever noticed it, and the caller got `-1`. An engine could not
/// learn which layer, which kernel or which fire refused — only that
/// something did, and only by reading a log it does not own.
pub(crate) fn status_of(error: &crate::Error) -> i32 {
    eprintln!("[driver-cuda] {error}");
    match error {
        crate::Error::Exhausted { .. } => PIE_STATUS_EXHAUSTED,
        crate::Error::Unsupported { .. } => driver_api::PIE_STATUS_UNSUPPORTED,
        crate::Error::Invalid { .. } => PIE_STATUS_INVALID_ARGUMENT,
        #[cfg(feature = "_cuda")]
        crate::Error::Runtime { .. } | crate::Error::Driver { .. } => PIE_STATUS_DRIVER_ERROR,
    }
}

pub(crate) mod encode;
pub(crate) mod load;
pub(crate) mod state;
pub(crate) mod transfer;

pub use crate::gpu::fire::launch::fire_class_of;
pub use encode::pie_cuda_encode;
pub use transfer::{pie_cuda_copy_kv, pie_cuda_copy_state, pie_cuda_resize_pool};

use crate::gpu::fire::launch::launch_impl;
use load::{adopt_and_compile, create_impl, destroy_impl, load_impl};
use state::{
    ChannelState,
    InstanceEntry,
    ProgramEntry,
    channel_dtype,
    instance_ring_shapes,
    shell,
    slice_of,
};

/// A descriptor pointer, dereferenced ONLY through its validator.
///
/// North star rule 4 says a shared capability must not be optional: if it
/// can be skipped, it will be. `driver-api` ships seventeen `validate_*`
/// functions; `driver-dummy` — the reference implementation of this
/// contract — calls them, and this shell called NONE, re-deriving similar
/// checks by hand at 51 sites. The capability was built, shipped, and
/// routed around.
///
/// Calling them would not have fixed that, only postponed it: a helper
/// that must be remembered is the same shape as a validator that must be
/// remembered. So the dereference and the validation are ONE operation.
/// There is no way to obtain a `&PieKvCopyDesc` in this file without
/// having run `validate_kv_copy_desc` over it, because the only thing
/// that turns the pointer into a reference is this function and it takes
/// the validator as an argument.
///
/// `tests/entry_validation.rs` holds the other half — that no entry point
/// reaches a descriptor any other way — because a rule the compiler
/// cannot state is one a reviewer has to, and this one can at least be
/// stated once rather than at every call.
///
/// # Why the validators are `unsafe` and this is not
///
/// Four of them (`frame`, `encode`, `instance`, `channel`) walk slices the
/// descriptor points at, so they carry the same obligation this function
/// does: the pointer is the caller's to vouch for. Wrapping the null test
/// and the deref together is what discharges it — a non-null descriptor
/// from the engine is a well-formed one, and a null is the error this
/// returns rather than the fault it would otherwise be.
#[cfg(feature = "abi")]
pub(crate) fn checked<'a, T>(
    p: *const T,
    validate: impl FnOnce(&T) -> driver_api::local::PieAbiValidationResult,
    what: &str,
) -> Result<&'a T, i32> {
    let Some(desc) = (unsafe { p.as_ref() }) else {
        eprintln!("[driver-cuda] {what}: the descriptor pointer is null");
        return Err(PIE_STATUS_INVALID_ARGUMENT);
    };
    match validate(desc) {
        Ok(()) => Ok(desc),
        Err(e) => {
            // The MESSAGE is the point. A hand-rolled check returns
            // `INVALID_ARGUMENT` and the caller learns which CALL failed;
            // the validators say which FIELD, and that difference is most
            // of what they are worth.
            eprintln!("[driver-cuda] {what}: {}", e.message());
            Err(e.status())
        }
    }
}

/// Run a driver entry point, turning a panic into a status rather than into
/// the caller's problem.
///
/// # Why this is here now and was not before
///
/// The thirteen entry points used to be `extern "C"`, where a Rust panic is
/// not a recoverable event: unwinding out of one was undefined behaviour and
/// then, from Rust 1.81, an abort. So the C++ shell's `try/catch` on every
/// export had no counterpart on this side and could not have one -- catching
/// would have to happen INSIDE the frame, which is what this does.
///
/// They are plain Rust functions now, so a panic unwinds normally into the
/// engine. That makes catching possible; this makes it the contract. Every one
/// of these already answers a status code, and "the driver hit a bug" is a
/// status, not a reason to take the process down with the other requests it
/// was serving.
///
/// **This does not make a panic safe to ignore.** A caught panic means the
/// shell's invariants are in an unknown state -- a half-built plan, a stream
/// with work queued, an arena mid-resize -- so it is reported loudly and the
/// driver should be considered untrustworthy afterwards. What it buys is that
/// the OTHER requests in flight get to finish, and that the operator gets a
/// message naming the entry point instead of a bare SIGABRT.
pub(crate) fn guard<T>(what: &str, on_panic: T, body: impl FnOnce() -> T) -> T {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)) {
        Ok(value) => value,
        Err(payload) => {
            let why = payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_owned())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "a panic with no message".to_owned());
            eprintln!(
                "[driver-cuda] {what}: PANICKED: {why}. The request is failed \
                 rather than the process; this driver's state is no longer \
                 trustworthy and it should be recreated."
            );
            on_panic
        }
    }
}

/// Create the driver. Refuses a null descriptor or a mismatched ABI
/// version by returning null, as the C++ shell does.
pub fn pie_cuda_create(
    desc: *const PieDriverCreateDesc,
    caps: *mut PieDriverCaps,
) -> *mut PieDriver {
    guard("create", std::ptr::null_mut(), || create_impl(desc, caps))
}

/// Device bytes this driver holds — a DIAGNOSTIC, not one of the thirteen.
///
/// What a leak test should read. `cudaMemGetInfo` answers a question about
/// the DEVICE, and a process sharing a GPU cannot answer that for itself:
/// a second consumer allocating during the measurement is
/// indistinguishable from a leak, and the fifty-step soak failed exactly
/// that way against a concurrent agent on the same card.
///
/// This is a claim about the SHELL. Zero for a null driver or one whose
/// fire allocator has not been made, which reads as "holds nothing" and
/// is true of both.
///
/// Not `extern "C"` and not in `driver_api::local`: the engine has no
/// reason to ask, and a fourteenth door would have to be a contract.
#[must_use]
pub fn live_device_bytes(driver: *mut PieDriver) -> usize {
    let Some(shell) = (unsafe { driver.cast::<state::Shell>().as_ref() }) else {
        return 0;
    };
    shell.fire_alloc.as_ref().map_or(0, crate::gpu::device::Allocator::live_bytes)
}

/// Tear the driver down. Null is a no-op, as everywhere in the ABI.
pub fn pie_cuda_destroy(driver: *mut PieDriver) {
    guard("destroy", (), || destroy_impl(driver));
}

/// Load the model: one parse of the snapshot through the Rust loader,
/// the `pie.model/1` descriptor (embedded meta, else the boot TOML's
/// path), and every bf16 weight resident on the device — with the
/// llama-like fused trace names built beside the checkpoint names, so the
/// executor's resolver asks and receives.
///
/// Still awaited here: quantized encodings (refused, not mis-loaded),
/// the memory plan, and KV materialization — those land with `launch`.
pub fn pie_cuda_load_model(
    driver: *mut PieDriver,
    load: *const PieModelLoadDesc,
    caps: *mut PieDriverCaps,
) -> i32 {
    guard("pie_cuda_load_model", PIE_STATUS_DRIVER_ERROR, move || {
        let Some(state) = shell(driver) else {
            return PIE_STATUS_INVALID_ARGUMENT;
        };
        let load = match checked(load, driver_api::local::validate_model_load_desc, "load_model") {
            Ok(d) => d,
            Err(status) => return status,
        };
        let snapshot = (!load.snapshot_dir.ptr.is_null())
            .then(|| unsafe {
                std::slice::from_raw_parts(load.snapshot_dir.ptr, load.snapshot_dir.len)
            })
            .and_then(|b| std::str::from_utf8(b).ok())
            .map(std::path::PathBuf::from);
        let Some(snapshot) = snapshot else {
            return PIE_STATUS_INVALID_ARGUMENT;
        };
        match load_impl(state, &snapshot) {
            Ok(()) => {
                let m = state.model.as_ref().expect("load_impl stored the model");
                if let Some(out) = unsafe { caps.as_mut() } {
                    out.json_bytes = m.load_caps.as_ptr();
                    out.json_len = m.load_caps.len();
                }
                PIE_STATUS_OK
            }
            Err(code) => code,
        }
    })
}

/// Register a program: adopt its launch package, compile its generated
/// regions, and answer an id.
///
/// The C3 hash is the dedup key — re-registering answers the existing id
/// without recompiling — which is what makes a program that is bound a
/// thousand times compiled once.
///
/// # What a failure here means, and why it is not always one
///
/// Four outcomes, and only two of them are errors:
///
/// * The descriptor carries NO launch package — an empty stage list. That
///   is a forward-only deployment: the model runs and the logits come
///   back through the instance's reader channel, with no user program
///   around the fire. An id is issued and nothing is adopted. `OK`.
/// * The package adopts and every generated region compiles. `OK`.
/// * The package adopts and the plan is UNEXECUTABLE — a per-layer tap
///   stage, an op this driver does not implement. That is not a driver
///   failure and not a registration failure either: the plan is recorded
///   with its reason, and the refusal surfaces at the launch that needs
///   it, where the caller can see which fire it lost.
/// * A region NVRTC rejects, or an emitted table with a hole in it.
///   `UNSUPPORTED`, and remembered: this driver carries no emitter, so a
///   generated region with no host source has no slower path to fall
///   back to.
///
/// A compile needs a device — the architecture comes from the GPU that
/// will run the code, never a guess — so a shell with no model loaded has
/// not bound one yet and defers the compile to the first launch rather
/// than compiling for an architecture it made up.
pub fn pie_cuda_register_program(
    driver: *mut PieDriver,
    program: *const PieProgramDesc,
    program_id: *mut u64,
) -> i32 {
    guard("pie_cuda_register_program", PIE_STATUS_DRIVER_ERROR, move || {
        let Some(state) = shell(driver) else {
            return PIE_STATUS_INVALID_ARGUMENT;
        };
        let desc =
            match checked(program, driver_api::local::validate_program_desc, "register_program") {
                Ok(d) => d,
                Err(status) => return status,
            };
        if desc.abi_version != PIE_DRIVER_ABI_VERSION {
            return PIE_STATUS_INVALID_ARGUMENT;
        }
        if let Some(id) = state
            .programs
            .iter()
            .find(|(_, p)| p.program_hash == desc.program_hash)
            .map(|(&id, _)| id)
        {
            if let Some(out) = unsafe { program_id.as_mut() } {
                *out = id;
            }
            return PIE_STATUS_OK;
        }

        // SAFETY: the engine's contract for `register_program` is that every
        // array reachable from the descriptor is live for the duration of the
        // call. Adoption COPIES, so nothing here outlives that window --
        // which is the reason it is done now rather than by holding the
        // descriptor: `PieProgramDesc` is the caller's transient memory.
        let package = unsafe { driver_api::adopt_package(&desc.launch) };
        let kernels = unsafe { driver_api::adopt_emitted_kernels(desc.emitted_kernels) };

        let id = state.next_id;
        state.next_id += 1;

        // A package with NO STAGES is not a malformed program; it is the
        // absence of one. The engine registers such a descriptor for a
        // forward-only deployment — the model runs, the logits come back
        // through the instance's reader channel, and no user program sits
        // around the fire. `adopt_launch_package` refuses an empty stage list
        // because an ExecPlan with nothing to execute is not a plan, and it is
        // right to; the judgement that this is not an ERROR belongs here,
        // where the difference between "the host sent a broken program" and
        // "the host sent no program" is visible.
        if !package.stages.is_empty() {
            if let Err(code) = adopt_and_compile(state, id, desc, package, &kernels) {
                return code;
            }
        }

        state.programs.insert(
            id,
            ProgramEntry {
                program_hash: desc.program_hash,
                emitter_version: desc.emitter_version,
            },
        );
        if let Some(out) = unsafe { program_id.as_mut() } {
            *out = id;
        }
        PIE_STATUS_OK
    })
}

/// Register a channel endpoint: the C++ registry's binding contract —
/// a pinned host MIRROR of `(capacity + 1)` wire cells and four pinned
/// control words (head 0, tail 1, poison 2, closed 3), both zeroed, with
/// the wire-cell math reproduced exactly (bool bit-packs, everything
/// else is four bytes per element; `capacity + 1 ≤ 64`). Device-side
/// rings and fire delivery ride with the launch integration.
pub fn pie_cuda_register_channel(
    driver: *mut PieDriver,
    channel: *const PieChannelDesc,
    binding: *mut PieChannelEndpointBinding,
) -> i32 {
    guard("pie_cuda_register_channel", PIE_STATUS_DRIVER_ERROR, move || {
        use crate::gpu::fire::attention_workspace::{LiveStagingOps, StagingOps};

        const MAX_RING: u64 = 64;
        let Some(state) = shell(driver) else {
            return PIE_STATUS_INVALID_ARGUMENT;
        };
        let desc = match checked(
            channel,
            |d| unsafe { driver_api::local::validate_channel_desc(d) },
            "register_channel",
        ) {
            Ok(d) => d,
            Err(status) => return status,
        };
        if desc.abi_version != PIE_DRIVER_ABI_VERSION
            || state.channels.contains_key(&desc.channel_id)
            || desc.dtype > driver_api::local::PIE_CHANNEL_DTYPE_ACT
        {
            return PIE_STATUS_INVALID_ARGUMENT;
        }
        let shape = slice_of(desc.shape.ptr, desc.shape.len);
        let mut numel: u64 = 1;
        for &d in shape {
            let Some(next) = numel.checked_mul(u64::from(d)) else {
                return PIE_STATUS_INVALID_ARGUMENT;
            };
            numel = next;
        }
        let wire_bytes: u64 = if desc.dtype == driver_api::local::PIE_CHANNEL_DTYPE_BOOL {
            numel.div_ceil(8)
        } else {
            match numel.checked_mul(4) {
                Some(b) => b,
                None => return PIE_STATUS_INVALID_ARGUMENT,
            }
        };
        let ring = u64::from(desc.capacity) + 1;
        if wire_bytes == 0 || ring > MAX_RING {
            return PIE_STATUS_INVALID_ARGUMENT;
        }
        let Some(mirror_bytes) = wire_bytes.checked_mul(ring) else {
            return PIE_STATUS_INVALID_ARGUMENT;
        };
        let Ok(mirror_bytes) = usize::try_from(mirror_bytes) else {
            return PIE_STATUS_INVALID_ARGUMENT;
        };

        let mut ops = LiveStagingOps;
        let Some(mirror) = ops.malloc_host(mirror_bytes) else {
            return PIE_STATUS_EXHAUSTED;
        };
        let word_bytes = 4 * std::mem::size_of::<u64>();
        let Some(words) = ops.malloc_host(word_bytes) else {
            ops.free_host(mirror);
            return PIE_STATUS_EXHAUSTED;
        };
        unsafe {
            std::ptr::write_bytes(mirror.cast::<u8>(), 0, mirror_bytes);
            std::ptr::write_bytes(words.cast::<u8>(), 0, word_bytes);
        }
        state.channels.insert(
            desc.channel_id,
            ChannelState {
                mirror,
                words,
                mirror_bytes,
                cell_bytes: usize::try_from(wire_bytes).unwrap_or(usize::MAX),
                ring: u32::try_from(ring).expect("ring fits u32"),
                host_role: desc.host_role,
                numel: usize::try_from(numel).unwrap_or(usize::MAX),
                dtype: channel_dtype(desc.dtype),
                // RECORDED AT REGISTRATION, refused at bind. Registering
                // an extern channel is harmless — the endpoint binding
                // below is the same either way, and the host mirror is
                // real. What cannot be served is ATTACHING two programs
                // to it, which is what `bind_instance` turns away.
                extern_dir: desc.extern_dir,
            },
        );
        if let Some(out) = unsafe { binding.as_mut() } {
            *out = PieChannelEndpointBinding {
                channel_id: desc.channel_id,
                mirror_base: mirror as u64,
                word_base: words as u64,
                mirror_bytes: mirror_bytes as u64,
                word_bytes: word_bytes as u64,
                cell_bytes: u32::try_from(wire_bytes).unwrap_or(u32::MAX),
                capacity: desc.capacity,
                head_word_index: 0,
                tail_word_index: 1,
                poison_word_index: 2,
                closed_word_index: 3,
            };
        }
        PIE_STATUS_OK
    })
}

/// Bind an instance to a registered program: the id lifecycle, honoring
/// a nonzero `requested_instance_id` and echoing the geometry class.
/// KV-slot and adapter state ride in with the `launch` arm.
pub fn pie_cuda_bind_instance(
    driver: *mut PieDriver,
    instance: *const PieInstanceDesc,
    binding: *mut PieInstanceBinding,
) -> i32 {
    guard("pie_cuda_bind_instance", PIE_STATUS_DRIVER_ERROR, move || {
        let Some(state) = shell(driver) else {
            return PIE_STATUS_INVALID_ARGUMENT;
        };
        let desc = match checked(
            instance,
            |d| unsafe { driver_api::local::validate_instance_desc(d) },
            "bind_instance",
        ) {
            Ok(d) => d,
            Err(status) => return status,
        };
        if desc.abi_version != PIE_DRIVER_ABI_VERSION
            || !state.programs.contains_key(&desc.program_id)
        {
            return PIE_STATUS_INVALID_ARGUMENT;
        }
        // AN EXTERN CHANNEL IS NOT SERVABLE HERE, and the reason is on
        // `ChannelState::is_extern`: it needs ONE ring shared between the
        // exporting and the importing program, and this driver builds a
        // ring per session. Binding anyway gives the importer a private
        // ring that no one ever fills — a program that blocks forever, or
        // reads a zeroed cell and treats it as a value. Refusing at the
        // attach is the only point where the driver can still say so.
        let attached = slice_of(desc.channel_ids.ptr, desc.channel_ids.len);
        if let Some(&cid) = attached
            .iter()
            .find(|c| state.channels.get(c).is_some_and(ChannelState::is_extern))
        {
            eprintln!(
                "[driver-cuda] bind_instance: channel {cid} is declared extern \
                 and this driver allocates one ring per session, so the two \
                 programs sharing it would not share cells or cursors. \
                 Refusing rather than binding a ring nobody fills."
            );
            return driver_api::PIE_STATUS_UNSUPPORTED;
        }
        let id = if desc.requested_instance_id != 0 {
            desc.requested_instance_id
        } else {
            let id = state.next_id;
            state.next_id += 1;
            id
        };
        if state.instances.contains_key(&id) {
            return PIE_STATUS_INVALID_ARGUMENT;
        }
        state.instances.insert(
            id,
            InstanceEntry {
                program_id: desc.program_id,
                geometry_class: desc.geometry_class,
                channel_ids: slice_of(desc.channel_ids.ptr, desc.channel_ids.len).to_vec(),
            },
        );
        if let Some(out) = unsafe { binding.as_mut() } {
            out.instance_id = id;
            out.geometry_class = desc.geometry_class;
            out.reserved0 = 0;
        }
        PIE_STATUS_OK
    })
}

/// Launch a frame: the executor's fire assembly, promoted from the
/// smokes into the shell.
///
/// What runs today: SINGLE-step, single-sub-batch frames over the loaded
/// llama-like model — the frame's own CSRs become the fire, the KV pools
/// are driver-owned and persist across launches, write targets derive
/// from the CSR tails, and every batch member's terminal cell is
/// published (release) before the runtime is notified. Multi-step
/// frames, device-geometry sub-batches and channel-delivered outputs
/// refuse with UNSUPPORTED until their machinery lands — logits stay in
/// the shell until channels exist to carry them out.
pub fn pie_cuda_launch(
    driver: *mut PieDriver,
    frame: *const PieFrameDesc,
    completion: PieCompletion,
) -> i32 {
    guard("pie_cuda_launch", PIE_STATUS_DRIVER_ERROR, move || {
        let Some(state) = shell(driver) else {
            return PIE_STATUS_INVALID_ARGUMENT;
        };
        // NOT `validate_frame_desc` YET, and the reason is written down
        // rather than left as an omission. That validator requires
        // `terminal_cells.len == roster_rows.len` and a
        // `channel_ticket_indptr` whenever ticket values are present;
        // this shell serves frames today that state neither, so adopting
        // it here would refuse traffic that works. Which of the two is
        // wrong — the frames or the rule — is a question for whoever owns
        // the ticket path, and guessing is how a validator gets weakened
        // instead of a caller fixed. See `validators-unskippable`.
        let frame = match checked(
            frame,
            |d| unsafe { driver_api::local::validate_frame_desc(d) },
            "launch",
        ) {
            Ok(d) => d,
            Err(status) => return status,
        };
        // THE CALL RETURNS WITH THE WORK STILL ON THE STREAM.
        //
        // Publishing the terminal cells and notifying used to happen here,
        // after `launch_impl` had synchronized — which made the driver
        // serialize the engine's `frame_dispatch_depth`, because the call
        // that would enqueue fire n+1 could not start until fire n had
        // retired on the GPU.
        //
        // Both debts moved into a stream-ordered host callback
        // (`retire_fire`), so an `Ok` here means ENQUEUED rather than DONE.
        // An error still returns synchronously: a fire that could not be
        // built owes nothing and the runtime must hear so on this thread.
        match launch_impl(state, frame, completion) {
            Ok(()) => PIE_STATUS_OK,
            Err(code) => code,
        }
    })
}

/// Close an instance — idempotently, the C++'s reading: closing what is
/// not open is not an error.
pub fn pie_cuda_close_instance(driver: *mut PieDriver, instance_id: u64) -> i32 {
    guard("pie_cuda_close_instance", PIE_STATUS_DRIVER_ERROR, move || {
        let Some(state) = shell(driver) else {
            return PIE_STATUS_INVALID_ARGUMENT;
        };
        state.instances.remove(&instance_id);
        // The rings go with it. They are the instance's, and holding them
        // past its close is a device allocation nothing can reach — the
        // channel ids that named it are gone with the entry above.
        state.ptir_sessions.remove(&instance_id);
        PIE_STATUS_OK
    })
}

/// Close a channel — idempotently, freeing its pinned endpoint.
pub fn pie_cuda_close_channel(driver: *mut PieDriver, channel_id: u64) -> i32 {
    guard("pie_cuda_close_channel", PIE_STATUS_DRIVER_ERROR, move || {
        let Some(state) = shell(driver) else {
            return PIE_STATUS_INVALID_ARGUMENT;
        };
        if let Some(ch) = state.channels.remove(&channel_id) {
            // UNREGISTERED NOW, FREED LATER. See `InFlight::closed_channels`:
            // a queued fire's debt holds a copy of this state and will publish
            // into it from a stream callback, so the mirror cannot go back to
            // the allocator until every fire that could name it has retired.
            //
            // No fire in flight means nothing can be holding it, and the free
            // happens here as it always did.
            match state.in_flight.back_mut() {
                Some(last) => last.closed_channels.push(ch),
                None => ch.free(),
            }
        }
        PIE_STATUS_OK
    })
}

#[cfg(test)]
mod tests {
    use super::{ChannelState, InstanceEntry, instance_ring_shapes};

    /// A registered channel becomes the ring shape the device wants, and
    /// the bool case is the one that could not be derived any other way.
    ///
    /// `cell_bytes` is WIRE bytes, and for bools that is
    /// `numel.div_ceil(8)` — so one lane and eight lanes are both one
    /// byte, and a device ring sized from the width would be eight times
    /// too small for the eight-lane case. That is why `numel` and `dtype`
    /// are stored rather than recomputed, and this is the test that says
    /// so out loud.
    #[test]
    fn a_bool_channels_ring_shape_cannot_be_derived_from_its_wire_width() {
        use driver::tensor_ir::DType;

        let chan = |numel: usize, dtype: DType, capacity: u32| ChannelState {
            mirror: std::ptr::null_mut(),
            words: std::ptr::null_mut(),
            mirror_bytes: 0,
            cell_bytes: crate::gpu::program::channel::wire_cell_bytes(dtype, numel),
            extern_dir: driver_api::local::PIE_CHANNEL_EXTERN_NONE,
            ring: capacity + 1,
            host_role: 0,
            numel,
            dtype,
        };

        let one = chan(1, DType::Bool, 1);
        let eight = chan(8, DType::Bool, 1);
        assert_eq!(one.cell_bytes, eight.cell_bytes, "one wire byte either way");
        assert_eq!(one.shape().numel, 1);
        assert_eq!(eight.shape().numel, 8, "the shape knows what the width cannot");
        assert_eq!(eight.shape().cell_bytes(), 8, "native is a byte per lane");

        // And the capacity survives the `ring = capacity + 1` round trip,
        // which is the one place the two vocabularies disagree.
        assert_eq!(chan(4, DType::F32, 7).shape().capacity, 7);
    }

    /// An instance's ring shapes come back in the order the program
    /// indexes them, and a channel it names but this shell lacks is a
    /// refusal rather than a shorter list.
    ///
    /// Skipping would RENUMBER every channel after the gap, so a program
    /// that meant `chan 1` would read channel 2 — a wrong answer that
    /// looks like a working fire, which is the class this whole driver
    /// keeps choosing to refuse instead.
    #[test]
    fn a_channel_an_instance_names_and_this_shell_lacks_refuses_the_whole_list() {
        use driver::tensor_ir::DType;

        let mut channels = std::collections::BTreeMap::new();
        for (id, numel) in [(10u64, 4usize), (20, 9)] {
            channels.insert(id, ChannelState {
                mirror: std::ptr::null_mut(),
                words: std::ptr::null_mut(),
                mirror_bytes: 0,
                cell_bytes: numel * 4,
                extern_dir: driver_api::local::PIE_CHANNEL_EXTERN_NONE,
                ring: 2,
                host_role: 0,
                numel,
                dtype: DType::F32,
            });
        }
        let inst = |ids: Vec<u64>| InstanceEntry {
            program_id: 1,
            geometry_class: 0,
            channel_ids: ids,
        };

        // The list is the INSTANCE's order, not the map's.
        let shapes = instance_ring_shapes(&inst(vec![20, 10]), &channels).expect("all present");
        assert_eq!(shapes.len(), 2);
        assert_eq!(shapes[0].numel, 9, "channel 20 is index 0 because it is listed first");
        assert_eq!(shapes[1].numel, 4);

        assert!(
            instance_ring_shapes(&inst(vec![10, 99]), &channels).is_none(),
            "an unknown channel refuses rather than shortening the list"
        );
    }
    /// The boot-TOML extraction, isolated: the exact chain `create` runs.
    #[test]
    fn the_boot_descriptor_extracts() {
        let boot = "[model]\ndescriptor = \"/tmp/x.json\"\n";
        let v = boot.parse::<toml::Table>().expect("parses");
        let path = v
            .get("model")
            .and_then(|m| m.get("descriptor"))
            .and_then(|d| d.as_str())
            .expect("extracts");
        assert_eq!(path, "/tmp/x.json");
    }
}

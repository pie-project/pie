//! The loader's C entry points.
//!
//! The driver calls in here; nothing calls out. Three rules hold for every
//! function in this module:
//!
//! * **Never unwind.** A panic crossing into C++ is not something the driver can
//!   act on, and Rust already handles it: an unwind out of an `extern "C"`
//!   function prints the panic and aborts. That is the honest outcome — a panic
//!   here is a loader bug, and the shipping profile is `panic = "abort"`
//!   (`Cargo.toml` `[profile.release-min]`) so no unwinding happens at all.
//!   Nothing in this module tries to convert one into a status code.
//! * **Never hold global state.** Ranks compile in parallel
//!   (`runtime/engine/src/driver/backend/cuda.rs:331-345` runs one thread per
//!   rank), so two `pie_loader_compile` calls can be in flight at once. Every
//!   allocation is reachable only from the handle it is returned through.
//! * **Status answers *did it work*; diagnostics answer *what went wrong*.** The
//!   status code is a single value and verification produces a list, so
//!   collapsing the two would throw away the thing that makes verification worth
//!   running.

use std::path::PathBuf;

use pie_loader::checkpoint::read::parse_checkpoint_metadata;
use pie_loader::error::Error;

use super::arena;
use super::checkpoint::PieLoaderCheckpoint;
use super::types::*;

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PieLoaderStatus {
    Ok = 0,
    /// The request was malformed: a null pointer, a non-UTF-8 path, an
    /// out-of-range enum. The caller built the request wrong.
    InvalidRequest = 1,
    /// The checkpoint could not be read or does not say what the request claims.
    InvalidCheckpoint = 2,
    /// The plan was rejected against its contract. Only `pie_loader_verify`
    /// returns this.
    ContractViolation = 3,
    /// The compiler failed on input it should have handled. A bug in the loader.
    Internal = 4,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PieLoaderSeverity {
    /// The plan is still usable. Emitted so a compile can report a suboptimal
    /// choice without failing.
    Warning = 0,
    Error = 1,
}

/// One diagnostic record.
///
/// `message` is borrowed from the owning [`PieLoaderDiagnostics`] and is valid
/// until it is released.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PieLoaderDiagnostic {
    pub severity: PieLoaderSeverity,
    pub message: PieLoaderBytes,
}

/// A flat array of diagnostics plus the storage backing their messages.
///
/// Returned through an out-param by both entry points, including on success when
/// the compile produced warnings. A null out-pointer means the caller does not
/// want them; a non-null pointer left null means there were none.
#[repr(C)]
#[derive(Debug)]
pub struct PieLoaderDiagnostics {
    pub items: *const PieLoaderDiagnostic,
    pub len: usize,
    owner: *mut std::ffi::c_void,
}

struct DiagnosticsArena {
    messages: Vec<Box<[u8]>>,
    items: Vec<PieLoaderDiagnostic>,
}

unsafe impl Send for DiagnosticsArena {}
unsafe impl Sync for DiagnosticsArena {}
unsafe impl Send for PieLoaderDiagnostics {}
unsafe impl Sync for PieLoaderDiagnostics {}

/// Collects diagnostics during a call, then publishes them as one allocation.
#[derive(Default)]
pub(super) struct DiagnosticSink {
    entries: Vec<(PieLoaderSeverity, String)>,
}

impl DiagnosticSink {
    pub(super) fn error(&mut self, message: impl Into<String>) {
        self.entries
            .push((PieLoaderSeverity::Error, message.into()));
    }

    pub(super) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub(super) fn publish(self) -> *mut PieLoaderDiagnostics {
        if self.entries.is_empty() {
            return std::ptr::null_mut();
        }
        let mut arena = DiagnosticsArena {
            messages: Vec::with_capacity(self.entries.len()),
            items: Vec::with_capacity(self.entries.len()),
        };
        for (severity, message) in self.entries {
            let boxed: Box<[u8]> = message.into_bytes().into_boxed_slice();
            arena.items.push(PieLoaderDiagnostic {
                severity,
                message: PieLoaderBytes {
                    ptr: boxed.as_ptr(),
                    len: boxed.len(),
                },
            });
            arena.messages.push(boxed);
        }
        let items = arena.items.as_ptr();
        let len = arena.items.len();
        let owner = Box::into_raw(Box::new(arena)).cast::<std::ffi::c_void>();
        Box::into_raw(Box::new(PieLoaderDiagnostics { items, len, owner }))
    }
}

/// Write `diags` through `out` if the caller supplied a slot, otherwise drop it.
///
/// # Safety
///
/// `out` is either null or a writable `*mut PieLoaderDiagnostics`.
pub(super) unsafe fn emit(out: *mut *mut PieLoaderDiagnostics, diags: *mut PieLoaderDiagnostics) {
    if out.is_null() {
        unsafe { release_diagnostics(diags) };
        return;
    }
    unsafe { *out = diags };
}

/// Publish one message through `out`, or clear it when there is none.
///
/// The whole of [`DiagnosticSink`] for an entry point whose failures are a
/// single sentence — which is every entry point that opens a file rather than
/// compiling something.
///
/// # Safety
///
/// `out` is either null or a writable `*mut PieLoaderDiagnostics`.
pub(super) unsafe fn emit_error(out: *mut *mut PieLoaderDiagnostics, message: Option<String>) {
    let mut sink = DiagnosticSink::default();
    if let Some(message) = message {
        sink.error(message);
    }
    unsafe { emit(out, sink.publish()) };
}

/// # Safety
///
/// `diags` is null or a pointer from [`DiagnosticSink::publish`].
unsafe fn release_diagnostics(diags: *mut PieLoaderDiagnostics) {
    if diags.is_null() {
        return;
    }
    let boxed = unsafe { Box::from_raw(diags) };
    if !boxed.owner.is_null() {
        drop(unsafe { Box::from_raw(boxed.owner.cast::<DiagnosticsArena>()) });
    }
}

/// The device-measured half of a request.
///
/// Every field is a fact only the driver can state: what the device is, how
/// wide the TP group is, and which transforms this backend's kernels implement.
/// The loader never guesses any of it, which is what makes a plan reproducible
/// from a recorded spec on a machine with no GPU (`architecture.md` §2 P2).
///
/// `backend` and `encode_scratch_dtype` are plain `uint32_t` rather than their
/// enum types, and the same goes for every enum-valued *input* field. C++ lets a
/// caller store any integer in an enum-typed field, and reading such a field as
/// a Rust enum is undefined behaviour that happens *before* any check could
/// reject it. Assign `static_cast<uint32_t>(PieLoaderBackendKind::Cuda)`; the
/// loader validates the value and answers `PieLoaderStatus::InvalidRequest` if
/// it is not one of them.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PieLoaderTargetSpec {
    pub backend: u32,
    pub tp_rank: u32,
    pub tp_size: u32,
    pub max_tile_bytes: u64,
    pub preferred_alignment: u32,
    pub tile_map_mask: u32,
    pub native_mxfp4_moe: bool,
    /// Which fused transform chains this build has kernels for
    /// (`PIE_LOADER_FUSION_*`).
    ///
    /// The opt-out that used to be `PIE_CUDA_DISABLE_FUSED_TRANSCODE`, read
    /// inside the executor. As a request field it changes the *plan*, so two
    /// settings produce two artifacts instead of one plan that runs two ways
    /// (`architecture.md` §8.1). Backends with no fused kernels pass `0`.
    pub fusion_mask: u32,
    /// The dtype this target's encode kernels dequantize through. Decides how
    /// many rows of scratch fit in `max_tile_bytes`.
    ///
    /// A `u32` for the reason [`PieLoaderTargetSpec::backend`] is: this is an
    /// *input*, so the value is whatever C++ wrote, and a Rust enum with a
    /// value outside its variants is undefined behaviour before any check can
    /// run. The output side, `PieLoaderTargetView`, keeps the enum — the loader
    /// wrote that one.
    pub encode_scratch_dtype: u32,
    /// Row granularity of this target's block scales, or `0` for none. A
    /// block-scaled source is not tiled, because a tile boundary would cut a
    /// scale block.
    pub block_scale_rows: u32,
}

/// Borrow a `PieLoaderBytes` as `&str`, for the lifetime of the borrow it came
/// from.
///
/// # Safety
///
/// `value.ptr` must either be null or point at `value.len` initialized bytes
/// that stay live and unwritten for `'a`.
pub(super) unsafe fn as_str<'a>(value: &'a PieLoaderBytes, field: &str) -> Result<&'a str, String> {
    if value.ptr.is_null() {
        if value.len == 0 {
            return Ok("");
        }
        return Err(format!("{field}: null pointer with non-zero length"));
    }
    let bytes = unsafe { std::slice::from_raw_parts(value.ptr, value.len) };
    std::str::from_utf8(bytes).map_err(|err| format!("{field}: not valid UTF-8: {err}"))
}

/// Borrow a `PieLoaderBytes` as a byte slice, for the lifetime of the borrow it
/// came from.
///
/// The same borrow as [`as_str`] without the UTF-8 check, for a payload that is
/// parsed rather than read as text — a JSON document reports its own encoding
/// errors, and with better context than "not valid UTF-8" can give.
///
/// An empty payload is refused rather than borrowed as an empty slice: every
/// caller of this sends a document, and a zero-length one is a caller that
/// forgot to, not a document that happens to say nothing.
///
/// # Safety
///
/// `value.ptr` must either be null or point at `value.len` initialized bytes
/// that stay live and unwritten for `'a`.
pub(super) unsafe fn as_bytes<'a>(
    value: &'a PieLoaderBytes,
    field: &str,
) -> Result<&'a [u8], String> {
    if value.ptr.is_null() || value.len == 0 {
        return Err(format!("{field}: empty; the caller must send the document"));
    }
    Ok(unsafe { std::slice::from_raw_parts(value.ptr, value.len) })
}

/// How a compile failure crosses the ABI.
///
/// Every variant that used to be `InvalidInput` still answers
/// `InvalidCheckpoint`, so this is behaviour-preserving: the status enum is
/// coarser than [`Error`] on purpose, because a C caller acts on "retry with a
/// different checkpoint" versus "file a bug" and nothing finer. The full
/// distinction travels in the diagnostic message and in [`Error::code`].
///
/// Written exhaustively rather than with a wildcard so that adding a variant to
/// [`Error`] forces this decision to be made again.
pub(super) fn compile_error_status(err: &Error) -> PieLoaderStatus {
    match err {
        Error::Contract(_)
        | Error::Shard(_)
        | Error::Checkpoint(_)
        | Error::Unsupported(_)
        | Error::Overflow(_) => PieLoaderStatus::InvalidCheckpoint,
        Error::Internal(_) => PieLoaderStatus::Internal,
    }
}

/// Open a checkpoint and read its tensor table.
///
/// The only entry point that touches the filesystem before a plan executes.
/// On success `*out` receives a handle the caller must free with
/// [`pie_loader_close_checkpoint`].
///
/// # Safety
///
/// `snapshot_dir` is a valid [`PieLoaderBytes`] live for the call. `out` is a
/// writable slot. `out_diags` is null or a writable slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_loader_open_checkpoint(
    snapshot_dir: PieLoaderBytes,
    out: *mut *mut PieLoaderCheckpoint,
    out_diags: *mut *mut PieLoaderDiagnostics,
) -> PieLoaderStatus {
    if !out_diags.is_null() {
        unsafe { *out_diags = std::ptr::null_mut() };
    }
    if out.is_null() {
        return PieLoaderStatus::InvalidRequest;
    }
    unsafe { *out = std::ptr::null_mut() };

    let mut sink = DiagnosticSink::default();
    let result = (|| {
        let dir = unsafe { as_str(&snapshot_dir, "snapshot_dir") }
            .map_err(|err| (PieLoaderStatus::InvalidRequest, err))?;
        if dir.is_empty() {
            return Err((
                PieLoaderStatus::InvalidRequest,
                "snapshot_dir is empty".to_string(),
            ));
        }
        let dir = PathBuf::from(dir);
        parse_checkpoint_metadata(&dir)
            .map(|metadata| (metadata, dir))
            .map_err(|err| (compile_error_status(&err), err.to_string()))
    })();
    let status = match result {
        Ok((metadata, dir)) => {
            unsafe { *out = super::checkpoint::build(metadata, dir) };
            PieLoaderStatus::Ok
        }
        Err((status, message)) => {
            sink.error(message);
            status
        }
    };
    unsafe { emit(out_diags, sink.publish()) };
    status
}

/// Free a checkpoint handle.
///
/// # Safety
///
/// `checkpoint` is null, or a handle from [`pie_loader_open_checkpoint`] that
/// has not already been closed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_loader_close_checkpoint(checkpoint: *mut PieLoaderCheckpoint) {
    unsafe { super::checkpoint::release(checkpoint) }
}

/// Free a plan returned by [`pie_loader_compile_model`](crate::model::pie_loader_compile_model).
///
/// # Safety
///
/// `plan` is null or a pointer from
/// [`pie_loader_compile_model`](crate::model::pie_loader_compile_model) that
/// has not already been released.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_loader_release(plan: *mut PieLoaderPlan) {
    unsafe { arena::release(plan) }
}

/// Free diagnostics returned through an out-param.
///
/// # Safety
///
/// `diags` is null or a pointer produced by this module that has not already been
/// released.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_loader_release_diagnostics(diags: *mut PieLoaderDiagnostics) {
    unsafe { release_diagnostics(diags) }
}

/// A function pointer that is `Sync` by fiat, so the table below can be a
/// `static`. Function addresses are immutable; the wrapper exists only because
/// raw pointers are not `Sync` by default.
#[repr(transparent)]
struct EntryAddr(*const ());
unsafe impl Sync for EntryAddr {}

/// Anchors the entry points against dead-code elimination.
///
/// `pie-loader` is an rlib, and nothing in Rust calls these functions — the
/// only caller is the C++ driver, linked afterwards. Without a reference from a
/// reachable item, `rustc` and the linker are free to drop `#[no_mangle]`
/// functions from an rlib, and the failure surfaces as an undefined symbol at
/// final link rather than anywhere near this file (`architecture.md` §3.4).
/// `#[used]` keeps the table, and the table keeps the functions.
#[used]
static KEEP_ALIVE: [EntryAddr; 6] = [
    EntryAddr(pie_loader_open_checkpoint as *const ()),
    EntryAddr(pie_loader_close_checkpoint as *const ()),
    EntryAddr(crate::model::pie_loader_compile_model as *const ()),
    EntryAddr(crate::model::pie_loader_verify_model as *const ()),
    EntryAddr(pie_loader_release as *const ()),
    EntryAddr(pie_loader_release_diagnostics as *const ()),
];

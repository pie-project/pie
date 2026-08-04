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

use crate::cache_key::{ArtifactInputs, artifact_cache_key};
use crate::checkpoint::read::parse_checkpoint_metadata;
use crate::error::Error;
use crate::plan::LoadPlan;
use crate::plan::compile as compile_load_plan;

use super::arena;
use super::checkpoint::PieLoaderCheckpoint;
use super::contract::PieLoaderModelContractView;
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
struct DiagnosticSink {
    entries: Vec<(PieLoaderSeverity, String)>,
}

impl DiagnosticSink {
    fn error(&mut self, message: impl Into<String>) {
        self.entries
            .push((PieLoaderSeverity::Error, message.into()));
    }

    fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn publish(self) -> *mut PieLoaderDiagnostics {
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
unsafe fn emit(out: *mut *mut PieLoaderDiagnostics, diags: *mut PieLoaderDiagnostics) {
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
fn compile_error_status(err: &Error) -> PieLoaderStatus {
    match err {
        Error::Contract(_)
        | Error::Shard(_)
        | Error::Checkpoint(_)
        | Error::Unsupported(_)
        | Error::Overflow(_) => PieLoaderStatus::InvalidCheckpoint,
        Error::Internal(_) => PieLoaderStatus::Internal,
    }
}

/// A compile request that carries its own contract.
///
/// Exactly the north-star signature: the facts read off the checkpoint, the
/// program stated over them, and the device to build for. None of the three is
/// a model's name.
///
/// There is no `model`, because the loader infers nothing from `model_type`; no
/// `runtime_quant`, because every tensor states the encoding it wants; no
/// `mxfp4_moe`, because a contract either declares an MXFP4 expert weight or it
/// does not; no `component`, because a contract that should not load the vision
/// tower simply does not declare it; and no `demands`, because the contract
/// already names every tensor it will produce and may declare the shape of
/// each.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PieLoaderContractRequest {
    /// The checkpoint, already open. Not a directory: the driver had to read
    /// the tensor table to write the contract, and compiling against the same
    /// handle is what makes the contract and the plan provably about one parse.
    pub checkpoint: *const PieLoaderCheckpoint,
    pub target: PieLoaderTargetSpec,
    /// What to build. Borrowed for the call; the loader copies what it keeps.
    pub contract: PieLoaderModelContractView,
}

/// Compile a contract into a plan.
///
/// # Safety
///
/// `req` and everything its pointers reach must be live for the call.
unsafe fn compile_contract_request(
    req: &PieLoaderContractRequest,
) -> Result<(LoadPlan, String), (PieLoaderStatus, String)> {
    let checked = unsafe { read_contract_request(req) }?;
    let source = unsafe { super::checkpoint::arena_of(req.checkpoint) };

    compile_load_plan(&source.metadata, &checked.model, checked.target)
        .map_err(|err| (compile_error_status(&err), err.to_string()))
        .map(|plan| {
            // `runtime_quant` and `component` are empty because neither exists
            // on this path, and neither is missing information: the contract
            // decides both, and the contract is entirely observable in the plan
            // that is hashed alongside them.
            let cache_key = artifact_cache_key(
                &plan,
                &ArtifactInputs {
                    snapshot_dir: &source.snapshot_dir,
                    runtime_quant: "",
                    component: 0,
                },
            );
            (plan, cache_key)
        })
}

/// A contract request with its target resolved and its contract materialized.
struct CheckedContractRequest {
    model: crate::contract::ModelContract,
    target: crate::plan::StorageTarget,
}

/// Validate a contract request without touching the filesystem.
///
/// Split out so that compiling and verifying resolve the target the same way;
/// a second copy of these rules is a second thing to keep in step.
///
/// # Safety
///
/// `req` and everything its pointers reach must be live for the call.
unsafe fn read_contract_request(
    req: &PieLoaderContractRequest,
) -> Result<CheckedContractRequest, (PieLoaderStatus, String)> {
    let bad = |err: String| (PieLoaderStatus::InvalidRequest, err);
    if req.checkpoint.is_null() {
        return Err(bad(
            "request.checkpoint is null; open one with pie_loader_open_checkpoint".to_string(),
        ));
    }
    let backend = PieLoaderBackendKind::try_from(req.target.backend).map_err(|v| {
        bad(format!(
            "request.target.backend: {v} is not a PieLoaderBackendKind"
        ))
    })?;
    if req.target.tp_size == 0 || req.target.tp_rank >= req.target.tp_size {
        return Err(bad(format!(
            "request.target: tp_rank {} is not a rank of a {}-way group",
            req.target.tp_rank, req.target.tp_size
        )));
    }
    // The MoE lowering is a property of the contract now, so the target is
    // resolved with the request that asks for nothing: an expert weight is
    // MXFP4 because a node says so, not because a flag on the side said the
    // model was one of the ones that should be.
    let target = super::storage_target(&req.target, backend).map_err(bad)?;
    let model = unsafe { super::contract::read_contract(&req.contract) }.map_err(bad)?;
    Ok(CheckedContractRequest { model, target })
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

/// Compile a driver-authored contract into a plan.
///
/// This is the only way to obtain a plan: what gets built is decided by the
/// contract the driver authors, never by a model name the loader recognizes.
/// The plan is owned by the caller and freed with [`pie_loader_release`];
/// diagnostics are a separate allocation freed with
/// [`pie_loader_release_diagnostics`].
///
/// # Safety
///
/// `req` must point at a valid [`PieLoaderContractRequest`] whose borrowed
/// memory is live for the call. `out_plan` must be a writable slot. `out_diags`
/// is either null or a writable slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_loader_compile_contract(
    req: *const PieLoaderContractRequest,
    out_plan: *mut *mut PieLoaderPlan,
    out_diags: *mut *mut PieLoaderDiagnostics,
) -> PieLoaderStatus {
    if !out_diags.is_null() {
        unsafe { *out_diags = std::ptr::null_mut() };
    }
    if out_plan.is_null() {
        return PieLoaderStatus::InvalidRequest;
    }
    unsafe { *out_plan = std::ptr::null_mut() };

    let mut sink = DiagnosticSink::default();
    if req.is_null() {
        sink.error("pie_loader_compile_contract: request is null");
        unsafe { emit(out_diags, sink.publish()) };
        return PieLoaderStatus::InvalidRequest;
    }

    let status = match unsafe { compile_contract_request(&*req) } {
        Ok((plan, cache_key)) => {
            unsafe { *out_plan = arena::build(&plan, &cache_key) };
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

/// Check a plan against the contract it was compiled from.
///
/// The contract is read a second time here rather than carried over from the
/// compile, and the plan is read back out of the C arena rather than out of the
/// Rust value it was built from. Both are deliberate: it makes this a check of
/// two independently-derived answers, with the marshalling in scope, instead of
/// a comparison of the compiler with itself.
///
/// # Safety
///
/// `plan` must point at a plan from [`pie_loader_compile_contract`] that has
/// not been released. `req` must point at a valid [`PieLoaderContractRequest`]
/// whose borrowed memory is live for the call. `out_diags` is either null or a
/// writable slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_loader_verify_contract(
    plan: *const PieLoaderPlan,
    req: *const PieLoaderContractRequest,
    out_diags: *mut *mut PieLoaderDiagnostics,
) -> PieLoaderStatus {
    if !out_diags.is_null() {
        unsafe { *out_diags = std::ptr::null_mut() };
    }
    let mut sink = DiagnosticSink::default();
    if plan.is_null() || req.is_null() {
        sink.error("pie_loader_verify_contract: plan or request is null");
        unsafe { emit(out_diags, sink.publish()) };
        return PieLoaderStatus::InvalidRequest;
    }

    let plan = unsafe { &*plan };
    let req = unsafe { &*req };
    match unsafe { read_contract_request(req) } {
        Ok(checked) => {
            verify_plan_contract(
                plan,
                &crate::verify::ContractView::of(&checked.model),
                &mut sink,
            );
        }
        Err((_, message)) => sink.error(message),
    }
    // A contract request states no MoE policy, so `Auto` is what the compile
    // resolved and `Auto` is what verification must resolve.
    verify_target_compat(plan, &req.target, &mut sink);

    let status = if sink.is_empty() {
        PieLoaderStatus::Ok
    } else {
        PieLoaderStatus::ContractViolation
    };
    unsafe { emit(out_diags, sink.publish()) };
    status
}

/// Free a plan returned by [`pie_loader_compile_contract`].
///
/// # Safety
///
/// `plan` is null or a pointer from [`pie_loader_compile_contract`] that has not already
/// been released.
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

/// The plan-level invariants a driver can check without re-running the compiler.
///
/// Two different questions are answered here.
///
/// The first — is this plan *self-consistent and still true of the files it
/// names*? — is [`crate::verify`]'s, and is asked against a [`PlanView`] built
/// from the marshalled plan the driver is actually holding. Verifying the C
/// view rather than the Rust one is deliberate: it puts a marshalling bug in
/// scope, which it would not be if this re-read the plan the driver never sees.
///
/// The second is specific to this boundary and cannot be asked without the
/// request: was this plan compiled *for the caller*? Rank divergence is the
/// motivating case (`architecture.md` §6.2), and no amount of internal
/// consistency detects it.
fn verify_plan_contract(
    plan: &PieLoaderPlan,
    contract: &crate::verify::ContractView<'_>,
    sink: &mut DiagnosticSink,
) {
    let view = match unsafe { super::view::plan_view(plan) } {
        Ok(view) => view,
        Err(err) => {
            sink.error(err);
            return;
        }
    };
    if let Err(violations) = crate::verify::verify(&view, Some(contract)) {
        for violation in violations {
            sink.error(violation.to_string());
        }
    }
}

/// Check that the plan was compiled for *this* caller's device.
///
/// Shared by both entry points because the question does not depend on who
/// authored the contract: a plan built for another rank, another alignment or
/// another fusion set is wrong for this driver either way.
fn verify_target_compat(
    plan: &PieLoaderPlan,
    target: &PieLoaderTargetSpec,
    sink: &mut DiagnosticSink,
) {
    match PieLoaderBackendKind::try_from(target.backend) {
        Ok(backend) if plan.target.backend != backend => sink.error(format!(
            "plan backend {:?} does not match requested {:?}",
            plan.target.backend, backend
        )),
        Ok(_) => {}
        Err(value) => sink.error(format!(
            "request.target.backend: {value} is not a PieLoaderBackendKind"
        )),
    }
    if plan.target.tp_rank != target.tp_rank || plan.target.tp_size != target.tp_size {
        sink.error(format!(
            "plan is rank {}/{} but the caller is rank {}/{}",
            plan.target.tp_rank, plan.target.tp_size, target.tp_rank, target.tp_size
        ));
    }
    if plan.target.tile_map_mask & !target.tile_map_mask != 0 {
        sink.error(format!(
            "plan requires tile maps {:#x} the target does not implement (target mask {:#x})",
            plan.target.tile_map_mask & !target.tile_map_mask,
            target.tile_map_mask
        ));
    }
    if plan.target.preferred_alignment != target.preferred_alignment {
        sink.error(format!(
            "plan alignment {} does not match target {}",
            plan.target.preferred_alignment, target.preferred_alignment
        ));
    }
    if plan.target.max_tile_bytes != target.max_tile_bytes {
        sink.error(format!(
            "plan max_tile_bytes {} does not match target {}",
            plan.target.max_tile_bytes, target.max_tile_bytes
        ));
    }
    if plan.target.native_mxfp4_moe != target.native_mxfp4_moe {
        sink.error(format!(
            "plan native_mxfp4_moe {} does not match target {}",
            plan.target.native_mxfp4_moe, target.native_mxfp4_moe
        ));
    }
    // Not a formality: a plan compiled with fusion on names a different kernel
    // sequence, so running it on a driver that has fusion off would execute
    // something the plan does not describe.
    if plan.target.fusion_mask != target.fusion_mask {
        sink.error(format!(
            "plan fusion_mask {:#x} does not match target {:#x}",
            plan.target.fusion_mask, target.fusion_mask
        ));
    }
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
    EntryAddr(pie_loader_compile_contract as *const ()),
    EntryAddr(pie_loader_verify_contract as *const ()),
    EntryAddr(pie_loader_release as *const ()),
    EntryAddr(pie_loader_release_diagnostics as *const ()),
];

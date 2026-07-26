//! The loader's C entry points.
//!
//! The driver calls in here; nothing calls out. Three rules hold for every
//! function in this module:
//!
//! * **Never unwind.** A panic crossing into C++ is undefined behaviour, so every
//!   body is wrapped in [`catch_unwind`](std::panic::catch_unwind) and a panic is
//!   reported as [`PieLoaderStatus::Panic`] with the payload as a diagnostic.
//! * **Never hold global state.** Ranks compile in parallel
//!   (`runtime/engine/src/driver/backend/cuda.rs:331-345` runs one thread per
//!   rank), so two `pie_loader_compile` calls can be in flight at once. Every
//!   allocation is reachable only from the handle it is returned through.
//! * **Status answers *did it work*; diagnostics answer *what went wrong*.** The
//!   status code is a single value and verification produces a list, so
//!   collapsing the two would throw away the thing that makes verification worth
//!   running.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;

use crate::artifact::{ArtifactInputs, artifact_cache_key};
use crate::error::CompileError;
use crate::ffi::inproc::parse_checkpoint_metadata;
use crate::load_plan::LoadPlan;
use crate::planner::compile_load_plan;
use crate::verify::ContractCoverage;

use super::arena;
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
    /// A panic was caught at the boundary. Also a bug in the loader, but
    /// distinguished because the diagnostic is a panic payload rather than a
    /// structured error.
    Panic = 5,
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

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PieLoaderMxfp4MoeRequest {
    /// Let the loader pick from the target's capabilities.
    Auto = 0,
    RoutedDecode = 1,
    NativeGemm = 2,
    EagerBf16 = 3,
}

impl TryFrom<u32> for PieLoaderMxfp4MoeRequest {
    type Error = u32;

    fn try_from(value: u32) -> Result<Self, u32> {
        match value {
            0 => Ok(Self::Auto),
            1 => Ok(Self::RoutedDecode),
            2 => Ok(Self::NativeGemm),
            3 => Ok(Self::EagerBf16),
            other => Err(other),
        }
    }
}

/// Which part of a multimodal checkpoint to load.
///
/// `Full` and `Text` load every tensor the ABI declares; `Encode` narrows the
/// plan to the vision/audio towers so an encoder-only rank does not materialize
/// the language model.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PieLoaderComponent {
    Full = 0,
    Text = 1,
    Encode = 2,
}

impl TryFrom<u32> for PieLoaderComponent {
    type Error = u32;

    fn try_from(value: u32) -> Result<Self, u32> {
        match value {
            0 => Ok(Self::Full),
            1 => Ok(Self::Text),
            2 => Ok(Self::Encode),
            other => Err(other),
        }
    }
}

/// The device-measured half of a request.
///
/// Every field is a fact only the driver can state: what the device is, how wide
/// the TP group is, and which transforms this backend's kernels implement. The
/// loader never guesses any of it, which is what makes a plan reproducible from
/// a recorded spec on a machine with no GPU (§2 P2).
///
/// `backend` is a plain `uint32_t` rather than `PieLoaderBackendKind`, and the
/// same goes for the enum-valued fields of [`PieLoaderRequest`]. C++ lets a
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
    pub fp8_native: bool,
    pub native_mxfp4_moe: bool,
    /// Whether this build has the fused transform kernels and wants them used.
    ///
    /// The opt-out that used to be `PIE_CUDA_DISABLE_FUSED_TRANSCODE`, read
    /// inside the executor. As a request field it changes the *plan*, so two
    /// settings produce two artifacts instead of one plan that runs two ways
    /// (§8.1). Backends with no fused kernels pass `false`.
    pub fused_transcode: bool,
}

/// The model facts the storage compile keys off.
///
/// The driver has already parsed `config.json` to build its own model, so the
/// loader reading it a second time was duplicated work that could disagree.
/// The driver states these; the loader never opens the file.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PieLoaderModelSpec {
    /// HF `model_type`, from the text tower when the checkpoint nests one.
    pub model_type: PieLoaderBytes,
    /// `quantization_config.quant_method`, or empty for an unquantized
    /// checkpoint.
    pub quant_method: PieLoaderBytes,
    pub num_hidden_layers: u32,
    /// HF `num_local_experts` / `num_experts` / `n_routed_experts`. Zero on a
    /// dense model.
    pub num_experts: u32,
    pub num_experts_per_tok: u32,
}

/// A compile request.
///
/// Strings are pointer + length and are borrowed for the duration of the call;
/// the loader copies anything it keeps.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PieLoaderRequest {
    pub snapshot_dir: PieLoaderBytes,
    pub target: PieLoaderTargetSpec,
    pub model: PieLoaderModelSpec,
    /// The runtime's own quantization request (e.g. `"fp8"`), not a checkpoint
    /// fact. Empty means "whatever the checkpoint is". Ignored when the target
    /// reports no native support for the requested format.
    pub runtime_quant: PieLoaderBytes,
    /// A `PieLoaderMxfp4MoeRequest` value, as `uint32_t`. See
    /// [`PieLoaderTargetSpec::backend`] for why this is not the enum type.
    pub mxfp4_moe: u32,
    /// A `PieLoaderComponent` value, as `uint32_t`.
    pub component: u32,
    /// What the driver promises to bind. Borrowed for the call.
    ///
    /// Empty means the driver declares nothing, and `pie_loader_verify` checks
    /// only the plan's internal consistency — which is where every caller
    /// started. A driver that fills this in gets its shapes checked against the
    /// plan by code that did not compute them.
    pub demands: PieLoaderTensorDemandSlice,
}

/// A request with every enum field checked, so the rest of the module can match
/// on real Rust enums without reasoning about where the bytes came from.
#[derive(Clone, Copy, Debug)]
struct CheckedRequest {
    backend: PieLoaderBackendKind,
    mxfp4_moe: PieLoaderMxfp4MoeRequest,
    component: PieLoaderComponent,
}

impl CheckedRequest {
    fn new(req: &PieLoaderRequest) -> Result<Self, String> {
        Ok(Self {
            backend: PieLoaderBackendKind::try_from(req.target.backend).map_err(|v| {
                format!("request.target.backend: {v} is not a PieLoaderBackendKind")
            })?,
            mxfp4_moe: PieLoaderMxfp4MoeRequest::try_from(req.mxfp4_moe)
                .map_err(|v| format!("request.mxfp4_moe: {v} is not a PieLoaderMxfp4MoeRequest"))?,
            component: PieLoaderComponent::try_from(req.component)
                .map_err(|v| format!("request.component: {v} is not a PieLoaderComponent"))?,
        })
    }
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

fn compile_error_status(err: &CompileError) -> PieLoaderStatus {
    match err {
        CompileError::InvalidInput(_) => PieLoaderStatus::InvalidCheckpoint,
        CompileError::Internal(_) => PieLoaderStatus::Internal,
    }
}

/// The compile itself, in safe Rust. Kept separate from the `extern "C"` shell so
/// the boundary concerns (unwinding, null checks, out-params) stay legible and so
/// tests can drive the real path without going through raw pointers.
fn compile_request(
    req: &PieLoaderRequest,
) -> Result<(LoadPlan, arena::PlanExtras), (PieLoaderStatus, String)> {
    let checked = CheckedRequest::new(req).map_err(|err| (PieLoaderStatus::InvalidRequest, err))?;
    let snapshot_dir = unsafe { as_str(&req.snapshot_dir, "request.snapshot_dir") }
        .map_err(|err| (PieLoaderStatus::InvalidRequest, err))?;
    if snapshot_dir.is_empty() {
        return Err((
            PieLoaderStatus::InvalidRequest,
            "request.snapshot_dir is empty".to_string(),
        ));
    }
    let runtime_quant = unsafe { as_str(&req.runtime_quant, "request.runtime_quant") }
        .map_err(|err| (PieLoaderStatus::InvalidRequest, err))?;
    if req.target.tp_size == 0 || req.target.tp_rank >= req.target.tp_size {
        return Err((
            PieLoaderStatus::InvalidRequest,
            format!(
                "request.target: tp_rank {} is not a rank of a {}-way group",
                req.target.tp_rank, req.target.tp_size
            ),
        ));
    }
    if req.target.preferred_alignment == 0 {
        return Err((
            PieLoaderStatus::InvalidRequest,
            "request.target.preferred_alignment is 0; the driver must state its \
             alignment (1 means unaligned)"
                .to_string(),
        ));
    }

    // A runtime quantization request the device cannot execute is dropped rather
    // than refused: it is a preference, and the checkpoint's own format is always
    // a valid answer.
    let runtime_quant = if runtime_quant == "fp8" && !req.target.fp8_native {
        ""
    } else {
        runtime_quant
    };

    let snapshot_dir = PathBuf::from(snapshot_dir);
    // Resolve the target before touching the filesystem: a target the driver
    // stated inconsistently is a bad *request*, and reporting it as a bad
    // checkpoint (which is what happens if the config read fails first) sends
    // the reader to the wrong place.
    let target = super::storage_target(&req.target, checked.backend, checked.mxfp4_moe)
        .map_err(|err| (PieLoaderStatus::InvalidRequest, err))?;
    let model = super::model_config(&req.model, runtime_quant)
        .map_err(|err| (PieLoaderStatus::InvalidRequest, err.to_string()))?;

    let metadata = parse_checkpoint_metadata(&snapshot_dir)
        .map_err(|err| (compile_error_status(&err), err.to_string()))?;
    let contract = crate::arch::default_contract(&metadata, &model, &target)
        .map_err(|err| (compile_error_status(&err), err.to_string()))?;
    let contract = super::scope_to_component(contract, checked.component, &model, &target)
        .map_err(|err| (compile_error_status(&err), err.to_string()))?;
    compile_load_plan(&metadata, &contract, target)
        .map_err(|err| (compile_error_status(&err), err.to_string()))
        .map(|plan| {
            // Both derived here, while the request is still in hand, so the plan
            // the driver receives already answers "did you build what I asked
            // for?" and "what should I call the result?" (§9).
            let coverage = match unsafe { contract_view(req) } {
                Ok(Some(contract)) => ContractCoverage::measure(
                    plan.tensors.iter().map(|tensor| tensor.name.as_str()),
                    contract
                        .tensors
                        .iter()
                        .map(|demand| (demand.name, demand.optional)),
                ),
                // A driver that declared nothing gets `0 / 0`. A malformed
                // declaration gets the same, and `pie_loader_verify` is what
                // reports it — refusing to compile would hide the real message
                // behind a coverage shortfall.
                _ => ContractCoverage::default(),
            };
            let cache_key = artifact_cache_key(
                &plan,
                &ArtifactInputs {
                    snapshot_dir: &snapshot_dir,
                    runtime_quant,
                    component: req.component,
                },
            );
            (
                plan,
                arena::PlanExtras {
                    coverage,
                    cache_key,
                },
            )
        })
}

/// Compile a checkpoint into a plan.
///
/// On success `*out_plan` receives a plan the caller must free with
/// [`pie_loader_release`]. On failure `*out_plan` is left null.
///
/// # Safety
///
/// `req` must point at a valid [`PieLoaderRequest`] whose borrowed strings are
/// live for the call. `out_plan` must be a writable slot. `out_diags` is either
/// null or a writable slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_loader_compile(
    req: *const PieLoaderRequest,
    out_plan: *mut *mut PieLoaderPlan,
    out_diags: *mut *mut PieLoaderDiagnostics,
) -> PieLoaderStatus {
    // Everything that can allocate, and therefore everything that can panic,
    // has to sit inside the guard: building the arena and publishing
    // diagnostics allocate just as readily as compiling does, and an unwind
    // escaping an `extern "C"` function is an abort, not an error (§5.2).
    never_unwind(|| {
        if !out_diags.is_null() {
            unsafe { *out_diags = std::ptr::null_mut() };
        }
        if out_plan.is_null() {
            return PieLoaderStatus::InvalidRequest;
        }
        unsafe { *out_plan = std::ptr::null_mut() };
        if req.is_null() {
            let mut sink = DiagnosticSink::default();
            sink.error("pie_loader_compile: request is null");
            unsafe { emit(out_diags, sink.publish()) };
            return PieLoaderStatus::InvalidRequest;
        }

        let mut sink = DiagnosticSink::default();
        // Inner guard: catching here lets the panic text reach the caller as a
        // diagnostic. Formatting it can itself allocate, but that happens inside
        // `never_unwind`, so the boundary stays safe either way.
        let status = match catch_unwind(AssertUnwindSafe(|| compile_request(unsafe { &*req }))) {
            Ok(Ok((plan, extras))) => {
                unsafe { *out_plan = arena::build(&plan, &extras) };
                PieLoaderStatus::Ok
            }
            Ok(Err((status, message))) => {
                sink.error(message);
                status
            }
            Err(payload) => {
                sink.error(format!(
                    "pie_loader_compile panicked: {}",
                    panic_text(&payload)
                ));
                PieLoaderStatus::Panic
            }
        };
        unsafe { emit(out_diags, sink.publish()) };
        status
    })
}

/// Check a plan against the contract it claims to satisfy.
///
/// Verification reports *every* violation it finds rather than the first, so the
/// diagnostics array is the result and the status only says whether the array is
/// empty.
///
/// # Safety
///
/// `plan` must point at a live plan from [`pie_loader_compile`]. `req` must point
/// at the request the plan was compiled from. `out_diags` is either null or a
/// writable slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_loader_verify(
    plan: *const PieLoaderPlan,
    req: *const PieLoaderRequest,
    out_diags: *mut *mut PieLoaderDiagnostics,
) -> PieLoaderStatus {
    never_unwind(|| {
        if !out_diags.is_null() {
            unsafe { *out_diags = std::ptr::null_mut() };
        }
        if plan.is_null() || req.is_null() {
            let mut sink = DiagnosticSink::default();
            sink.error("pie_loader_verify: plan or request is null");
            unsafe { emit(out_diags, sink.publish()) };
            return PieLoaderStatus::InvalidRequest;
        }

        let mut sink = DiagnosticSink::default();
        let panicked = catch_unwind(AssertUnwindSafe(|| {
            verify_plan(unsafe { &*plan }, unsafe { &*req }, &mut sink)
        }));
        let status = match panicked {
            Ok(()) if sink.is_empty() => PieLoaderStatus::Ok,
            Ok(()) => PieLoaderStatus::ContractViolation,
            Err(payload) => {
                sink.error(format!(
                    "pie_loader_verify panicked: {}",
                    panic_text(&payload)
                ));
                PieLoaderStatus::Panic
            }
        };
        unsafe { emit(out_diags, sink.publish()) };
        status
    })
}

/// Run `body`, converting any unwind into [`PieLoaderStatus::Panic`].
///
/// The payload is deliberately leaked rather than dropped: running an unknown
/// `Drop` here could itself unwind, and this frame is the last one that can
/// still turn an unwind into a return value. A panicking compile is already an
/// aborted cold start, so the leak is bounded and never repeats in a loop.
fn never_unwind(body: impl FnOnce() -> PieLoaderStatus) -> PieLoaderStatus {
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(status) => status,
        Err(payload) => {
            std::mem::forget(payload);
            PieLoaderStatus::Panic
        }
    }
}

/// Free a plan returned by [`pie_loader_compile`].
///
/// # Safety
///
/// `plan` is null or a pointer from [`pie_loader_compile`] that has not already
/// been released.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_loader_release(plan: *mut PieLoaderPlan) {
    let _ = catch_unwind(AssertUnwindSafe(|| unsafe { arena::release(plan) }));
}

/// Free diagnostics returned through an out-param.
///
/// # Safety
///
/// `diags` is null or a pointer produced by this module that has not already been
/// released.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_loader_release_diagnostics(diags: *mut PieLoaderDiagnostics) {
    let _ = catch_unwind(AssertUnwindSafe(|| unsafe { release_diagnostics(diags) }));
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
/// motivating case (§6.2), and no amount of internal consistency detects it.
fn verify_plan(plan: &PieLoaderPlan, req: &PieLoaderRequest, sink: &mut DiagnosticSink) {
    match unsafe { plan_view(plan) } {
        Ok(view) => {
            // The contract is whatever the driver declared. An empty `demands`
            // is not an error — it means this driver has not adopted the
            // declaration yet, and only the plan's internal consistency can be
            // checked. See `PieLoaderTensorDemand`.
            match unsafe { contract_view(req) } {
                Ok(contract) => {
                    let result = match &contract {
                        Some(contract) => crate::verify::verify(&view, Some(contract)),
                        None => crate::verify::verify(&view, None),
                    };
                    if let Err(violations) = result {
                        for violation in violations {
                            sink.error(violation.to_string());
                        }
                    }
                }
                Err(err) => sink.error(err),
            }
        }
        Err(err) => sink.error(err),
    }
    match CheckedRequest::new(req) {
        Ok(checked) => {
            if plan.target.backend != checked.backend {
                sink.error(format!(
                    "plan backend {:?} does not match requested {:?}",
                    plan.target.backend, checked.backend
                ));
            }
            // The plan records the *resolved* policy, so comparing it against
            // the raw request would report a false mismatch for `Auto`. Resolve
            // the request the same way the compile did and compare the answers;
            // otherwise a plan built under one policy verifies happily against a
            // request that asked for another.
            match super::resolve_mxfp4_moe(checked.mxfp4_moe, req.target.native_mxfp4_moe) {
                Ok(policy) => {
                    let policy = PieLoaderMxfp4MoePolicy::from(policy);
                    if plan.target.mxfp4_moe != policy {
                        sink.error(format!(
                            "plan mxfp4_moe {:?} does not match the request's resolved {policy:?}",
                            plan.target.mxfp4_moe
                        ));
                    }
                }
                Err(err) => sink.error(err),
            }
        }
        Err(err) => sink.error(err),
    }
    if plan.target.tp_rank != req.target.tp_rank || plan.target.tp_size != req.target.tp_size {
        sink.error(format!(
            "plan is rank {}/{} but the caller is rank {}/{}",
            plan.target.tp_rank, plan.target.tp_size, req.target.tp_rank, req.target.tp_size
        ));
    }
    if plan.target.tile_map_mask & !req.target.tile_map_mask != 0 {
        sink.error(format!(
            "plan requires tile maps {:#x} the target does not implement (target mask {:#x})",
            plan.target.tile_map_mask & !req.target.tile_map_mask,
            req.target.tile_map_mask
        ));
    }
    if plan.target.preferred_alignment != req.target.preferred_alignment {
        sink.error(format!(
            "plan alignment {} does not match target {}",
            plan.target.preferred_alignment, req.target.preferred_alignment
        ));
    }
    if plan.target.max_tile_bytes != req.target.max_tile_bytes {
        sink.error(format!(
            "plan max_tile_bytes {} does not match target {}",
            plan.target.max_tile_bytes, req.target.max_tile_bytes
        ));
    }
    if plan.target.native_mxfp4_moe != req.target.native_mxfp4_moe {
        sink.error(format!(
            "plan native_mxfp4_moe {} does not match target {}",
            plan.target.native_mxfp4_moe, req.target.native_mxfp4_moe
        ));
    }
    // Not a formality: a plan compiled with fusion on names a different kernel
    // sequence, so running it on a driver that has fusion off would execute
    // something the plan does not describe.
    if plan.target.fused_transcode != req.target.fused_transcode {
        sink.error(format!(
            "plan fused_transcode {} does not match target {}",
            plan.target.fused_transcode, req.target.fused_transcode
        ));
    }
}

/// Borrow a POD slice, treating a null pointer as empty.
///
/// # Safety
///
/// `ptr` is null, or valid for `len` elements for the lifetime `'a`.
unsafe fn slice_of<'a, T>(ptr: *const T, len: usize) -> &'a [T] {
    if ptr.is_null() {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }
}

/// Read the driver's declared contract out of the request.
///
/// `Ok(None)` means the driver declared nothing, which is the state every caller
/// starts in and is not an error. `Err` means it declared something malformed.
///
/// # Safety
///
/// `req.demands` is null or valid for its stated length, and every `name` /
/// `shape` inside it is null or valid for its own, which is true of any request
/// built by the `pie_loader` C++ contract builder.
unsafe fn contract_view(
    req: &PieLoaderRequest,
) -> Result<Option<crate::verify::ContractView<'_>>, String> {
    let demands = unsafe { slice_of(req.demands.ptr, req.demands.len) };
    if demands.is_empty() {
        return Ok(None);
    }
    let mut tensors = Vec::with_capacity(demands.len());
    for demand in demands {
        let name = unsafe { as_str(&demand.name, "declared tensor demand name") }?;
        if name.is_empty() {
            return Err("a declared tensor demand has an empty name".to_string());
        }
        let shape = unsafe { slice_of(demand.shape.ptr, demand.shape.len) };
        // An empty shape means "unstated", not "scalar". A zero-rank runtime
        // tensor does not exist here, so the two cannot be confused.
        let shape = if shape.is_empty() {
            None
        } else {
            if shape.iter().any(|dim| *dim <= 0) {
                return Err(format!(
                    "declared tensor demand '{name}' has a non-positive dimension in {shape:?}"
                ));
            }
            Some(shape.to_vec())
        };
        tensors.push(crate::verify::TensorDemand::declared(
            name,
            shape,
            demand.optional,
        ));
    }
    Ok(Some(crate::verify::ContractView { tensors }))
}

/// Reconstruct the safe view [`crate::verify`] works on from the POD plan.
///
/// This is the only place that dereferences the plan's slice pointers, so it is
/// where a malformed handle is caught rather than propagated. Names are decoded
/// here too: a non-UTF-8 name is reported once, as itself, instead of silently
/// failing to match later.
///
/// # Safety
///
/// `plan`'s slices are null or valid for their stated lengths, which is true of
/// any plan produced by [`pie_loader_compile`].
unsafe fn plan_view(plan: &PieLoaderPlan) -> Result<crate::verify::PlanView<'_>, String> {
    let instrs = unsafe { slice_of(plan.instrs.ptr, plan.instrs.len) };
    let mut files = Vec::with_capacity(plan.files.len);
    for file in unsafe { slice_of(plan.files.ptr, plan.files.len) } {
        files.push(crate::verify::FileView {
            id: file.id,
            path: unsafe { as_str(&file.path, "file.path") }
                .map_err(|err| format!("file {}: {err}", file.id))?,
            size_bytes: file.size_bytes,
        });
    }
    let mut sources = Vec::with_capacity(plan.sources.len);
    for source in unsafe { slice_of(plan.sources.ptr, plan.sources.len) } {
        sources.push(crate::verify::SourceView {
            name: unsafe { as_str(&source.name, "source.name") }
                .map_err(|err| format!("source tensor {}: {err}", source.id))?,
            file_id: source.file_id,
            offset_bytes: source.file_offset,
            span_bytes: source.span_bytes,
        });
    }
    let mut tensors = Vec::with_capacity(plan.tensors.len);
    for tensor in unsafe { slice_of(plan.tensors.ptr, plan.tensors.len) } {
        tensors.push(crate::verify::TensorView::new(
            unsafe { as_str(&tensor.name, "tensor.name") }
                .map_err(|err| format!("tensor {}: {err}", tensor.id))?,
            unsafe { slice_of(tensor.shape.ptr, tensor.shape.len) },
            &join_encoding(tensor),
        ));
    }

    let mut finalized = Vec::new();
    let mut reads = Vec::new();
    for instr in instrs {
        match instr.kind {
            PieLoaderStorageInstrKind::Finalize => finalized.push(
                unsafe { as_str(&instr.name, "instr.name") }
                    .map_err(|err| format!("instruction {}: {err}", instr.id))?,
            ),
            // Only the reading instructions carry a meaningful `source`; the
            // rest leave it at its zero default, which would look like a valid
            // reference to file 0.
            PieLoaderStorageInstrKind::ExtentWrite
            | PieLoaderStorageInstrKind::BulkExtentWrite
            | PieLoaderStorageInstrKind::TileMap => reads.push(crate::verify::ReadView {
                instr: instr.id,
                file_id: instr.source.file_id,
            }),
            _ => {}
        }
        if instr.slab_file_id != PIE_LOADER_NO_BUFFER {
            reads.push(crate::verify::ReadView {
                instr: instr.id,
                file_id: instr.slab_file_id,
            });
        }
    }

    Ok(crate::verify::PlanView {
        version: plan.version,
        compiler_version: plan.compiler_version,
        files,
        sources,
        tensors,
        instr_count: instrs.len(),
        schedule: unsafe { slice_of(plan.schedule.ptr, plan.schedule.len) }.to_vec(),
        finalized,
        reads,
    })
}

/// The inverse of `arena::split_encoding`: rebuild an [`Encoding`] from the flat
/// fields a POD view carries, so a marshalled tensor can be compared against
/// one the loader still holds in typed form.
fn join_encoding(tensor: &PieLoaderTensorDeclView) -> crate::types::Encoding {
    match tensor.encoding_kind {
        PieLoaderEncodingKind::Quant => crate::types::Encoding::Quant(crate::types::QuantSpec {
            scheme: tensor.quant_scheme.into(),
            logical_dtype: tensor.dtype.into(),
            bits_per_element: tensor.quant_bits_per_element,
            group_size: tensor.quant_group_size,
            channel_axis: None,
            scale_dtype: None,
            zero_point_dtype: None,
            block_shape: Vec::new(),
        }),
        _ => crate::types::Encoding::Raw(tensor.dtype.into()),
    }
}

fn panic_text(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(text) = payload.downcast_ref::<&str>() {
        (*text).to_string()
    } else if let Some(text) = payload.downcast_ref::<String>() {
        text.clone()
    } else {
        "non-string panic payload".to_string()
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
/// `pie-loader` is an rlib, and nothing in Rust calls these functions — the only
/// caller is the C++ driver, linked afterwards. Without a reference from a
/// reachable item, `rustc` and the linker are free to drop `#[no_mangle]`
/// functions from an rlib, and the failure surfaces as an undefined symbol at
/// final link rather than anywhere near this file (§3.4). `#[used]` keeps the
/// table, and the table keeps the functions.
#[used]
static KEEP_ALIVE: [EntryAddr; 4] = [
    EntryAddr(pie_loader_compile as *const ()),
    EntryAddr(pie_loader_verify as *const ()),
    EntryAddr(pie_loader_release as *const ()),
    EntryAddr(pie_loader_release_diagnostics as *const ()),
];

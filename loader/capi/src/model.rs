//! The request entry: facts and policy in, plan out.
//!
//! [`pie_loader_compile_model`] is the boundary the migration arrived at
//! (`plan/model-in-rust.md` §6): the caller sends the facts it parsed and
//! the policy it decided — a handful of scalars — and authoring happens on
//! this side, in `pie_model::contract`. The contract never crosses the ABI
//! at all; it is authored, compiled and dropped in one call, and the same
//! resolved [`StorageTarget`] feeds both the author and the compiler, so the
//! two cannot be told different worlds.
//!
//! This is the only way to obtain a plan. The contract entry that used to
//! stand beside it as the C++ authors' door was retired with the last of
//! them (§8-5): a family this side does not know is a diagnostic asking for
//! an author, not a fallback.
//!
//! Every enum-valued field crosses as a `u32`, for the reason
//! `request.hpp` gives: these are *inputs*, and a Rust enum holding a value
//! outside its variants is undefined behaviour before any check can run.

use pie_loader::cache_key::{ArtifactInputs, artifact_cache_key};
use pie_loader::plan::compile as compile_load_plan;

use pie_model_common::facts::ModelFacts;
use pie_model_common::policy::Mxfp4MoePolicy;
use pie_model_common::policy::{
    Component, FamilyKnobs, Mxfp4MoeRequest, Naming, Policy, Projections, RuntimeQuant,
};

use super::arena;
use super::checkpoint::PieLoaderCheckpoint;
use super::entry::{
    DiagnosticSink, PieLoaderDiagnostics, PieLoaderStatus, PieLoaderTargetSpec, as_bytes,
    compile_error_status, emit,
};
use super::types::{PieLoaderBackendKind, PieLoaderBytes, PieLoaderPlan};

/// The per-family switches, wire form. Mirrors [`FamilyKnobs`] field for
/// field; the caller reads its environment and fills these, so an author
/// never consults the environment and equal requests author equal contracts.
///
/// These two family names are the only ones in the ABI, and each is here
/// because the *forward* path reads the same environment variable. A switch
/// only the loader read would move the weights while the matmuls stayed put
/// — see [`FamilyKnobs`] for the six that went that way.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PieLoaderFamilyKnobs {
    pub qwen35_mtp_int8_lm_head: bool,
    pub nemotron_tp_mamba_sharding: bool,
}

/// Everything one authoring-and-compiling call is decided by.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PieLoaderModelRequest {
    /// The checkpoint, already open: authoring reads the same tensor table
    /// the compile does, which is what makes the contract and the plan
    /// provably about one parse.
    pub checkpoint: *const PieLoaderCheckpoint,
    pub target: PieLoaderTargetSpec,
    /// The `pie.model/1` descriptor, as the JSON bytes the caller read.
    ///
    /// This used to be ten scalars in a `PieLoaderModelFactsView` — the
    /// caller parsed `config.json`, picked out what it thought the loader
    /// needed, and sent that. The document is the request now: the facts are
    /// projected from it by `pie_model::ModelFacts::from_descriptor`, so
    /// which fields matter is a question answered where the authors live
    /// rather than in each driver, and a new fact needs no ABI change.
    ///
    /// Borrowed for the call, like every other pointer here.
    pub descriptor: PieLoaderBytes,
    /// `Projections` wire value: 0 fused, 1 in-place.
    pub projections: u32,
    /// `Naming` wire value: 0 HF, 1 MLX.
    pub naming: u32,
    /// `RuntimeQuant` wire value, already resolved against the device:
    /// 0 none, 1 fp8, 2 int8, 3 mxfp4, 4 int4 (MLX affine, Metal's).
    pub runtime_quant: u32,
    /// `Mxfp4MoeRequest` wire value: 0 auto, 1 routed, 2 native, 3 bf16.
    pub moe_request: u32,
    /// `Component` wire value: 0 full, 1 text, 2 encode.
    pub component: u32,
    pub stream_routed_experts: bool,
    pub knobs: PieLoaderFamilyKnobs,
}

fn enum_field<T>(value: u32, variants: &[(u32, T)], field: &str) -> Result<T, String>
where
    T: Copy,
{
    variants
        .iter()
        .find(|(wire, _)| *wire == value)
        .map(|(_, variant)| *variant)
        .ok_or_else(|| format!("request.{field}: {value} is not a valid value"))
}

/// # Safety
///
/// `req` and everything its pointers reach must be live for the call.
unsafe fn read_model_request(
    req: &PieLoaderModelRequest,
) -> Result<(ModelFacts, Policy), (PieLoaderStatus, String)> {
    let bad = |err: String| (PieLoaderStatus::InvalidRequest, err);
    // The facts are read from the descriptor here rather than sent field by
    // field: they are a projection of a document the caller already holds, and
    // a wire struct of ten scalars was ten chances for the two sides to
    // disagree about what a field means. `pie_model::ModelFacts` owns the
    // mapping now, on this side alone.
    let descriptor = unsafe { as_bytes(&req.descriptor, "descriptor") }.map_err(bad)?;
    let facts = ModelFacts::from_descriptor(descriptor)
        .map_err(|err| (PieLoaderStatus::InvalidRequest, err.to_string()))?;
    let policy = Policy {
        projections: enum_field(
            req.projections,
            &[(0, Projections::Fused), (1, Projections::InPlace)],
            "projections",
        )
        .map_err(bad)?,
        naming: enum_field(req.naming, &[(0, Naming::Hf), (1, Naming::Mlx)], "naming")
            .map_err(bad)?,
        runtime_quant: enum_field(
            req.runtime_quant,
            &[
                (0, RuntimeQuant::None),
                (1, RuntimeQuant::Fp8),
                (2, RuntimeQuant::Int8),
                (3, RuntimeQuant::Mxfp4),
                (4, RuntimeQuant::Int4),
            ],
            "runtime_quant",
        )
        .map_err(bad)?,
        moe_request: enum_field(
            req.moe_request,
            &[
                (0, Mxfp4MoeRequest::Auto),
                (1, Mxfp4MoeRequest::RoutedDecode),
                (2, Mxfp4MoeRequest::NativeGemm),
                (3, Mxfp4MoeRequest::EagerBf16),
            ],
            "moe_request",
        )
        .map_err(bad)?,
        component: enum_field(
            req.component,
            &[
                (0, Component::Full),
                (1, Component::Text),
                (2, Component::Encode),
            ],
            "component",
        )
        .map_err(bad)?,
        stream_routed_experts: req.stream_routed_experts,
        knobs: FamilyKnobs {
            qwen35_mtp_int8_lm_head: req.knobs.qwen35_mtp_int8_lm_head,
            nemotron_tp_mamba_sharding: req.knobs.nemotron_tp_mamba_sharding,
        },
    };
    Ok((facts, policy))
}

/// The authoring front half both entries share: resolve the target once,
/// read the request, author. Compile and verify calling this is what keeps
/// them about the same contract.
///
/// # Safety
///
/// `req` and everything its pointers reach must be live for the call.
unsafe fn author_from_request(
    req: &PieLoaderModelRequest,
) -> Result<
    (
        pie_loader::contract::ModelContract,
        pie_loader::plan::StorageTarget,
        Mxfp4MoePolicy,
    ),
    (PieLoaderStatus, String),
> {
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
    // One resolution, feeding the author and the compiler alike.
    let target = super::storage_target(&req.target, backend).map_err(bad)?;
    let (facts, policy) = unsafe { read_model_request(req) }?;
    let source = unsafe { super::checkpoint::arena_of(req.checkpoint) };

    let (contract, resolved_moe) =
        pie_model::contract::author_with_policy(&facts, &source.metadata, &target, &policy)
            .map_err(|err| (compile_error_status(&err), err.to_string()))?
            .ok_or_else(|| {
                bad(format!(
                    "no author for model_type '{}'; every family loads through \
                     this entry now, so an unknown one needs an author in \
                     pie_model::contract (plan/model-in-rust.md §7)",
                    facts.model_type
                ))
            })?;
    Ok((contract, target, resolved_moe))
}

/// # Safety
///
/// `req` and everything its pointers reach must be live for the call.
unsafe fn compile_model_request(
    req: &PieLoaderModelRequest,
) -> Result<(pie_loader::plan::LoadPlan, String, Mxfp4MoePolicy), (PieLoaderStatus, String)> {
    let (contract, target, resolved_moe) = unsafe { author_from_request(req) }?;
    let source = unsafe { super::checkpoint::arena_of(req.checkpoint) };

    compile_load_plan(&source.metadata, &contract, target)
        .map_err(|err| (compile_error_status(&err), err.to_string()))
        .map(|plan| {
            // Empty `runtime_quant` and `component`, for the reason the
            // contract path gives: the request fields decided the *contract*,
            // and the contract is entirely observable in the plan that is
            // hashed alongside them.
            let cache_key = artifact_cache_key(
                &plan,
                &ArtifactInputs {
                    snapshot_dir: &source.snapshot_dir,
                    runtime_quant: "",
                    component: 0,
                },
            );
            (plan, cache_key, resolved_moe)
        })
}

/// Author this model's contract and compile it into a plan, in one call.
///
/// `out_mxfp4_moe`, when non-null, receives the author's resolved
/// [`Mxfp4MoePolicy`] as its wire value — the answer the bind path branches
/// on, handed back rather than recomputed, because a family may override the
/// device rule.
///
/// # Safety
///
/// `req` and everything its pointers reach must be live for the call.
/// `out_plan` is a writable slot; `out_diags` and `out_mxfp4_moe` are null
/// or writable slots.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_loader_compile_model(
    req: *const PieLoaderModelRequest,
    out_plan: *mut *mut PieLoaderPlan,
    out_mxfp4_moe: *mut u32,
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
        sink.error("pie_loader_compile_model: request is null");
        unsafe { emit(out_diags, sink.publish()) };
        return PieLoaderStatus::InvalidRequest;
    }

    let status = match unsafe { compile_model_request(&*req) } {
        Ok((plan, cache_key, resolved_moe)) => {
            unsafe { *out_plan = arena::build(&plan, &cache_key) };
            if !out_mxfp4_moe.is_null() {
                unsafe { *out_mxfp4_moe = resolved_moe as u32 };
            }
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

/// Check a plan against the contract this request authors.
///
/// The caller has no contract to hand over — a contract is loader-internal
/// IR now — so the verifier re-authors one from the same request and holds
/// the plan to it. Re-authoring is the
/// point, not a cost — the check covers the author's determinism along with
/// everything the contract verifier checks.
///
/// # Safety
///
/// `plan` must point at an unreleased plan; `req` and everything its
/// pointers reach must be live for the call. `out_diags` is null or a
/// writable slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_loader_verify_model(
    plan: *const PieLoaderPlan,
    req: *const PieLoaderModelRequest,
    out_diags: *mut *mut PieLoaderDiagnostics,
) -> PieLoaderStatus {
    if !out_diags.is_null() {
        unsafe { *out_diags = std::ptr::null_mut() };
    }
    let mut sink = DiagnosticSink::default();
    if plan.is_null() || req.is_null() {
        sink.error("pie_loader_verify_model: plan or request is null");
        unsafe { emit(out_diags, sink.publish()) };
        return PieLoaderStatus::InvalidRequest;
    }

    let plan = unsafe { &*plan };
    let req = unsafe { &*req };
    match unsafe { author_from_request(req) } {
        Ok((contract, _, _)) => {
            verify_plan_contract(
                plan,
                &pie_loader::verify::ContractView::of(&contract),
                &mut sink,
            );
        }
        Err((_, message)) => sink.error(message),
    }
    verify_target_compat(plan, &req.target, &mut sink);

    let status = if sink.is_empty() {
        PieLoaderStatus::Ok
    } else {
        PieLoaderStatus::ContractViolation
    };
    unsafe { emit(out_diags, sink.publish()) };
    status
}

/// The plan-level invariants a driver can check without re-running the compiler.
///
/// Two different questions are answered here.
///
/// The first — is this plan *self-consistent and still true of the files it
/// names*? — is [`pie_loader::verify`]'s, and is asked against a [`PlanView`] built
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
    contract: &pie_loader::verify::ContractView<'_>,
    sink: &mut DiagnosticSink,
) {
    let view = match unsafe { super::view::plan_view(plan) } {
        Ok(view) => view,
        Err(err) => {
            sink.error(err);
            return;
        }
    };
    if let Err(violations) = pie_loader::verify::verify(&view, Some(contract)) {
        for violation in violations {
            sink.error(violation.to_string());
        }
    }
}

/// Check that the plan was compiled for *this* caller's device.
///
/// A plan built for another rank, another alignment or another fusion set is
/// wrong for this driver either way. (This and `verify_plan_contract` lived in
/// `entry.rs` while the contract entry shared them; `pie_loader_verify_model`
/// is their only caller now, so they live with it.)
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

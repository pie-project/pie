//! The request entry: facts and policy in, plan out.
//!
//! [`pie_loader_compile_model`] is the boundary the migration is heading for
//! (`plan/model-in-rust.md` §6): the caller sends the facts it parsed and
//! the policy it decided — a handful of scalars — and authoring happens on
//! this side, in `pie_model::contract`. The contract never crosses the ABI
//! at all; it is authored, compiled and dropped in one call, and the same
//! resolved [`StorageTarget`] feeds both the author and the compiler, so the
//! two cannot be told different worlds.
//!
//! [`pie_loader_compile_contract`](super::entry::pie_loader_compile_contract)
//! stays beside this while any C++ author remains: a family this side does
//! not know answers with a diagnostic naming the fallback rather than a
//! guess.
//!
//! Every enum-valued field crosses as a `u32`, for the reason
//! `request.hpp` gives: these are *inputs*, and a Rust enum holding a value
//! outside its variants is undefined behaviour before any check can run.

use pie_loader::cache_key::{ArtifactInputs, artifact_cache_key};
use pie_loader::plan::compile as compile_load_plan;

use pie_model::common::facts::ModelFacts;
use pie_model::common::policy::{
    Component, FamilyKnobs, Mxfp4MoeRequest, Naming, Policy, Projections, RuntimeQuant,
};

use super::arena;
use super::checkpoint::PieLoaderCheckpoint;
use super::entry::{
    DiagnosticSink, PieLoaderDiagnostics, PieLoaderStatus, PieLoaderTargetSpec, as_str,
    compile_error_status, emit,
};
use super::types::{PieLoaderBackendKind, PieLoaderBytes, PieLoaderPlan};

/// The config facts a family needs beyond the checkpoint itself. Mirrors
/// [`ModelFacts`]; strings are borrowed for the call.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PieLoaderModelFactsView {
    /// `model_type` from `config.json` — the key the author registry
    /// dispatches on.
    pub model_type: PieLoaderBytes,
    /// `quantization_config.quant_method`, empty for an unquantized
    /// checkpoint.
    pub quant_method: PieLoaderBytes,
    pub num_hidden_layers: u32,
    pub num_experts: u32,
    pub head_dim: u32,
    pub mamba_groups: u32,
}

/// The per-family switches, wire form. Mirrors [`FamilyKnobs`] field for
/// field; the caller reads its environment and fills these, so an author
/// never consults the environment and equal requests author equal contracts.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PieLoaderFamilyKnobs {
    pub glm5_moe_gate_up_swapped: bool,
    pub qwen35_fused_gdn_projection: bool,
    pub qwen35_mtp_int8_lm_head: bool,
    pub qwen35_moe_gate_up_swapped: bool,
    pub qwen35_fused_shared_scalar_gate: bool,
    pub kimi_k3_moe_gate_up_swapped: bool,
    pub kimi_moe_gate_up_swapped: bool,
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
    pub facts: PieLoaderModelFactsView,
    /// `Projections` wire value: 0 fused, 1 in-place.
    pub projections: u32,
    /// `Naming` wire value: 0 HF, 1 MLX.
    pub naming: u32,
    /// `RuntimeQuant` wire value, already resolved against the device:
    /// 0 none, 1 fp8, 2 int8, 3 mxfp4.
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
    let facts = ModelFacts {
        model_type: unsafe { as_str(&req.facts.model_type, "facts.model_type") }
            .map_err(bad)?
            .to_string(),
        quant_method: unsafe { as_str(&req.facts.quant_method, "facts.quant_method") }
            .map_err(bad)?
            .to_string(),
        num_hidden_layers: req.facts.num_hidden_layers,
        num_experts: req.facts.num_experts,
        head_dim: req.facts.head_dim,
        mamba_groups: req.facts.mamba_groups,
    };
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
            glm5_moe_gate_up_swapped: req.knobs.glm5_moe_gate_up_swapped,
            qwen35_fused_gdn_projection: req.knobs.qwen35_fused_gdn_projection,
            qwen35_mtp_int8_lm_head: req.knobs.qwen35_mtp_int8_lm_head,
            qwen35_moe_gate_up_swapped: req.knobs.qwen35_moe_gate_up_swapped,
            qwen35_fused_shared_scalar_gate: req.knobs.qwen35_fused_shared_scalar_gate,
            kimi_k3_moe_gate_up_swapped: req.knobs.kimi_k3_moe_gate_up_swapped,
            kimi_moe_gate_up_swapped: req.knobs.kimi_moe_gate_up_swapped,
            nemotron_tp_mamba_sharding: req.knobs.nemotron_tp_mamba_sharding,
        },
    };
    Ok((facts, policy))
}

/// # Safety
///
/// `req` and everything its pointers reach must be live for the call.
unsafe fn compile_model_request(
    req: &PieLoaderModelRequest,
) -> Result<(pie_loader::plan::LoadPlan, String), (PieLoaderStatus, String)> {
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

    let contract = pie_model::contract::author(&facts, &source.metadata, &target, &policy)
        .map_err(|err| (compile_error_status(&err), err.to_string()))?
        .ok_or_else(|| {
            bad(format!(
                "no author for model_type '{}'; author a contract and call \
                 pie_loader_compile_contract instead",
                facts.model_type
            ))
        })?;

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
            (plan, cache_key)
        })
}

/// Author this model's contract and compile it into a plan, in one call.
///
/// # Safety
///
/// `req` and everything its pointers reach must be live for the call.
/// `out_plan` is a writable slot; `out_diags` is null or a writable slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_loader_compile_model(
    req: *const PieLoaderModelRequest,
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
        sink.error("pie_loader_compile_model: request is null");
        unsafe { emit(out_diags, sink.publish()) };
        return PieLoaderStatus::InvalidRequest;
    }

    let status = match unsafe { compile_model_request(&*req) } {
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

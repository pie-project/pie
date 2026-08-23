use std::path::Path;

use model_loader::checkpoint::CheckpointMetadata;
use model_loader::plan::{self, LoadPlan, StorageTarget};

use crate::catalog::{self, Override, Unmatched, Variant};
use crate::encoding::Encoding;
use crate::shared::policy::{
    Component, FamilyKnobs, Mxfp4MoePolicy, Mxfp4MoeRequest, Naming, Policy, Projections,
    RuntimeQuant,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Binding {
    pub projections: Projections,

    pub naming: Naming,
}

impl Binding {
    pub const HF_FUSED: Self = Self {
        projections: Projections::Fused,
        naming: Naming::Hf,
    };

    pub const MLX_IN_PLACE: Self = Self {
        projections: Projections::InPlace,
        naming: Naming::Mlx,
    };
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadPlanError {
    Unidentified(Unmatched),

    Compile(String),
}

impl std::fmt::Display for LoadPlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unidentified(u) => write!(f, "{u}"),
            Self::Compile(m) => write!(f, "{m}"),
        }
    }
}

impl From<Unmatched> for LoadPlanError {
    fn from(u: Unmatched) -> Self {
        Self::Unidentified(u)
    }
}

impl std::error::Error for LoadPlanError {}

pub fn compile_load_plan(
    snapshot_dir: &Path,
    metadata: &CheckpointMetadata,
    target: &StorageTarget,
    chosen: &Override,
    encoding: &Encoding,
    binding: Binding,
) -> Result<(LoadPlan, Mxfp4MoePolicy), LoadPlanError> {
    let row = catalog::identify(metadata, chosen)?;
    compile_load_plan_for(snapshot_dir, metadata, target, row, encoding, binding)
}

pub fn compile_load_plan_for(
    snapshot_dir: &Path,
    metadata: &CheckpointMetadata,
    target: &StorageTarget,
    row: &dyn Variant,
    encoding: &Encoding,
    binding: Binding,
) -> Result<(LoadPlan, Mxfp4MoePolicy), LoadPlanError> {
    let policy = Policy {
        projections: binding.projections,
        naming: binding.naming,
        runtime_quant: RuntimeQuant::None,
        moe_request: Mxfp4MoeRequest::Auto,
        component: Component::Full,
        stream_routed_experts: false,
        knobs: FamilyKnobs::default(),
    };
    let (contract, resolved_moe) =
        crate::contract::author_with_policy(row, encoding, metadata, target, &policy)
            .map_err(|e| LoadPlanError::Compile(e.to_string()))?;
    let plan = plan::compile(metadata, &contract, target.clone())
        .map_err(|e| LoadPlanError::Compile(e.to_string()))?;
    model_loader::checkpoint::read::verify_declared_files(&plan, snapshot_dir)
        .map_err(|e| LoadPlanError::Compile(e.to_string()))?;
    Ok((plan, resolved_moe))
}

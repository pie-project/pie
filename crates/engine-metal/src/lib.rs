#![cfg_attr(
    not(target_vendor = "apple"),
    allow(
        clippy::clone_on_copy,
        clippy::unit_arg,
        clippy::let_unit_value,
        clippy::useless_conversion
    )
)]

pub mod abi;
pub mod adapter;
pub mod ane;
pub mod api;
pub mod arena;
pub mod blob;
pub mod boot;
pub mod chip;
pub mod device;
pub mod diag;
mod dispatch;
pub mod encode;
mod error;
pub mod experts;
pub mod weight_store;

pub mod decoded;
pub(crate) mod feeds;
pub mod gather;
pub mod host_source;
#[cfg(target_vendor = "apple")]
pub mod icb;
pub mod inputs;
mod keepalive;
mod legs;
pub mod mapping;
pub mod mask;
pub mod program;
pub mod rebind;
pub mod record;
pub mod rs;
pub mod run;
mod scratch;
pub mod serve;
pub mod settle;
pub mod store;
pub mod tune;
pub mod weights;
pub mod window;

pub use abi::{Armed, At as AbiAt, Axis, DescriptorAbi, Law, SlotAbi, Survey};
pub use adapter::{
    Binding as AdapterBinding, Key as AdapterKey, Role as AdapterRole, Site,
    Source as AdapterSource,
};
pub use api::{ContractFor, DeviceBoot, Metal};
pub use arena::Arena;
pub use blob::{Layout as AdapterLayout, Manifest as AdapterManifest, Stamp as AdapterStamp};
pub use boot::open;
pub use device::{Buffer, Context, Handles, Pipelines};
pub use diag::Diagnostics;
pub use encode::{Sink, kernel_profile, reset_kernel_profile};
pub use error::{Fault, Result};
pub use experts::{GroupResidency, PREDICTION_PREFIXES, Plan as ResidencyPlan, Prediction};
#[cfg(target_vendor = "apple")]
pub use icb::{Icb, Rebound};
pub use inputs::Inputs;
pub use program::{Fired, Launched, Plane as ProgramPlane, Session as ProgramSession};
pub use record::{Arg, Point, Recording, Slot, Tape};
pub use run::{
    CacheGeometry, CachePool, CacheTable, FireBindings, FireTables, PoolSlabs, Run, SlotTable,
    StructSlot, WeightRow, WeightTable,
};
pub use scratch::Scratch;
pub use serve::{
    Attached, Boot, Enqueued, FireCost, Landed, Lane, Prepared, Seated, Shell, StepView,
};
pub use settle::{Airborne, Arms, Done};
pub use store::Pools;
pub use weights::{AdapterPlane, BankSeat, Weights};
pub use window::{Copies, Gathered, GatheredSpace, Window, Windows};

/// The fusions this backend implements: the compiler runs these rules alone.
pub static FUSIONS: std::sync::LazyLock<Vec<poem_compiler::fuse::Rule>> =
    std::sync::LazyLock::new(|| {
        use poem_compiler::fuse::rules;
        vec![rules::residual_add_rmsnorm(), rules::matmul_swiglu_matmul()]
    });

#[cfg(test)]
mod tests {
    use poem_ir::Operation;

    /// Every fused op this backend's rules form on a catalog deployment is
    /// one its dispatch fires: a rule it holds and a form its dispatch turns
    /// away cannot both stand.
    #[test]
    fn every_fused_op_its_rules_form_is_one_dispatch_fires() {
        let mut refused = std::collections::BTreeSet::new();
        for row in poem_compiler::catalog::deployments() {
            let Ok(trace) = row.try_trace(poem_ir::Platform::Metal) else {
                continue;
            };
            let fused = poem_compiler::fuse::fuse(trace, super::FUSIONS.iter());
            for node in &fused.nodes {
                if let Operation::Fused(op) = &node.op
                    && crate::dispatch::refuses(op)
                {
                    refused.insert(format!("{}: {}", row.name, poem_ir::Operands::name(op)));
                }
            }
        }
        assert!(refused.is_empty(), "formed, then refused: {refused:?}");
    }
}

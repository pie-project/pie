//! The Neural Engine's share of this load: which dense MLPs split, how
//! their intermediate channels divide, and the surfaces and plan each
//! split runs over. The kernels and the program are kernels-metal's
//! (`linear::ane`, `ane::ffn`); this is the state between them.

#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
mod banks;
#[cfg(target_vendor = "apple")]
mod private;

pub use banks::Split;
#[cfg(target_vendor = "apple")]
pub use private::Private as Ane;

use kernels_metal::ane::Allotment;
use kernels_metal::{Bank, Tensor};
#[cfg(not(target_vendor = "apple"))]
use kernels_metal::{Ctx, Error};

use crate::device::{Context, Handles};
use crate::error::Result;

/// One split MLP about to run: the GPU's banks, its staging rows, and the
/// two hand-off values the Neural Engine waits on and signals.
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
pub struct Plan {
    pub split: Split,
    pub up_rows: Tensor,
    pub rows: u32,
    layer: u32,
    at: Allotment,
    evaluation: usize,
}

/// One projection about to run split by output column: the GPU's view of
/// the leading `keep` rows, and the hand-off values.
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
pub struct Columns {
    pub view: Bank,
    pub keep: u32,
    pub rows: u32,
    site: usize,
    layer: u32,
    at: Allotment,
    evaluation: usize,
}

#[cfg(not(target_vendor = "apple"))]
pub enum Ane {}

#[cfg(not(target_vendor = "apple"))]
impl Ane {
    pub fn plan(&self, _: poem_ir::ValueId, _: u32) -> Option<Plan> {
        match *self {}
    }
    pub fn before(&self, _: &Ctx<'_>, _: &Plan, _: Tensor) -> std::result::Result<(), Error> {
        match *self {}
    }
    pub fn after(&self, _: &Ctx<'_>, _: &Plan, _: Tensor) -> std::result::Result<(), Error> {
        match *self {}
    }
    pub fn plan_columns(&self, _: poem_ir::ValueId, _: u32) -> Option<Columns> {
        match *self {}
    }
    pub fn before_columns(
        &self,
        _: &Ctx<'_>,
        _: &Columns,
        _: Tensor,
    ) -> std::result::Result<(), Error> {
        match *self {}
    }
    pub fn after_columns(
        &self,
        _: &Ctx<'_>,
        _: &Columns,
        _: Tensor,
    ) -> std::result::Result<(), Error> {
        match *self {}
    }
    pub fn verdict(&self) -> Option<Box<dyn Fn() -> Option<String> + Send>> {
        match *self {}
    }
    pub fn compiled(&self) -> Option<Result<(), String>> {
        match *self {}
    }
    pub fn splits(&self) -> u64 {
        match *self {}
    }
}

/// Sets the Neural Engine up for the dense MLPs and the wide projections
/// in `trace`, or `None` when it is off, unavailable, or the model has
/// nothing it can take.
pub fn load(
    device: &Context,
    handles: &Handles,
    trace: &poem_ir::Trace,
    weights: &crate::weights::Weights,
) -> Result<Option<Ane>> {
    #[cfg(target_vendor = "apple")]
    {
        let mlps = banks::mlps(trace, weights);
        let projections = banks::projections(trace, weights);
        if !kernels_metal::ane::enabled()
            || (mlps.is_empty() && projections.is_empty())
            || kernels_metal::ane::available().is_err()
        {
            return Ok(None);
        }
        private::load(device, handles, &mlps, &projections)
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        let _ = (device, handles, trace, weights);
        Ok(None)
    }
}

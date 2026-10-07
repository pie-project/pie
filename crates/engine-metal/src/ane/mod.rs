#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
mod banks;
#[cfg(target_vendor = "apple")]
mod private;

pub use banks::Split;
#[cfg(target_vendor = "apple")]
pub use private::Private as Ane;

use kernels_metal::Tensor;
#[cfg(not(target_vendor = "apple"))]
use kernels_metal::{Ctx, Error};

use crate::device::{Context, Handles};
use crate::error::Result;

#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
pub struct Plan {
    pub split: Split,
    pub up_rows: Tensor,
    pub rows: u32,
    layer: u32,
    ready: u64,
    done: u64,
    evaluation: usize,
}

#[cfg(not(target_vendor = "apple"))]
pub enum Ane {}

#[cfg(not(target_vendor = "apple"))]
impl Ane {
    pub fn plan(&self, _: u32, _: u32) -> Option<Plan> {
        match *self {}
    }
    pub fn before(&self, _: &Ctx<'_>, _: &Plan, _: Tensor) -> std::result::Result<(), Error> {
        match *self {}
    }
    pub fn after(&self, _: &Ctx<'_>, _: &Plan, _: Tensor) -> std::result::Result<(), Error> {
        match *self {}
    }
    pub fn verdict(&self) -> Option<Box<dyn Fn() -> Option<String> + Send>> {
        match *self {}
    }
}

pub fn load(
    device: &Context,
    handles: &Handles,
    trace: &poem_ir::Trace,
    weights: &crate::weights::Weights,
) -> Result<Option<Ane>> {
    #[cfg(target_vendor = "apple")]
    {
        let mlps = banks::mlps(trace, weights);
        if !ane::enabled() || mlps.is_empty() || ane::private::available().is_err() {
            return Ok(None);
        }
        private::load(device, handles, &mlps)
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        let _ = (device, handles, trace, weights);
        Ok(None)
    }
}

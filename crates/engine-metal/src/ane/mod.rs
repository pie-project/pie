#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
mod banks;
#[cfg(target_vendor = "apple")]
mod coreml;
#[cfg(target_vendor = "apple")]
mod private;

pub use banks::Split;

use kernels_metal::{Ctx, Error, Tensor};

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

#[cfg(target_vendor = "apple")]
pub enum Ane {
    Private(Box<private::Private>),
    CoreMl(coreml::CoreMl),
}

#[cfg(not(target_vendor = "apple"))]
pub enum Ane {}

#[cfg(target_vendor = "apple")]
impl Ane {
    #[must_use]
    pub fn plan(&self, layer: u32, rows: u32) -> Option<Plan> {
        if crate::diag::on().kernel_profile.on() {
            return None;
        }
        match self {
            Ane::Private(path) => path.plan(layer, rows),
            Ane::CoreMl(path) => path.plan(layer, rows),
        }
    }

    pub fn before(&self, ctx: &Ctx<'_>, plan: &Plan, x: Tensor) -> std::result::Result<(), Error> {
        match self {
            Ane::Private(path) => path.before(ctx, plan, x),
            Ane::CoreMl(path) => path.before(ctx, plan, x),
        }
    }

    pub fn after(&self, ctx: &Ctx<'_>, plan: &Plan, out: Tensor) -> std::result::Result<(), Error> {
        match self {
            Ane::Private(path) => path.after(ctx, plan, out),
            Ane::CoreMl(path) => path.after(ctx, plan, out),
        }
    }

    #[must_use]
    pub fn event(&self) -> &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLEvent> {
        let event = match self {
            Ane::Private(path) => path.event(),
            Ane::CoreMl(path) => path.event(),
        };
        objc2::runtime::ProtocolObject::from_ref(event)
    }

    #[must_use]
    pub fn may_split(&self, rows: u32) -> bool {
        match self {
            Ane::Private(path) => path.may_split(rows),
            Ane::CoreMl(path) => path.may_split(rows),
        }
    }

    #[must_use]
    pub fn planned(&self) -> u64 {
        match self {
            Ane::Private(path) => path.planned(),
            Ane::CoreMl(path) => path.planned(),
        }
    }

    #[must_use]
    pub fn verdict(&self) -> Option<Box<dyn Fn() -> Option<String> + Send>> {
        match self {
            Ane::Private(path) => Some(path.verdict()),
            Ane::CoreMl(path) => Some(path.verdict()),
        }
    }
}

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
    pub fn may_split(&self, _: u32) -> bool {
        match *self {}
    }
    pub fn planned(&self) -> u64 {
        match *self {}
    }
}

pub fn load(
    device: &Context,
    handles: &Handles,
    trace: &model_ir::Trace,
    weights: &crate::weights::Weights,
) -> Result<Option<Ane>> {
    #[cfg(target_vendor = "apple")]
    {
        let Some(asked) = ane::requested() else {
            return Ok(None);
        };
        let mlps = banks::mlps(trace, weights);
        if mlps.is_empty() {
            return Ok(None);
        }
        match ane::private::available() {
            Ok(()) if asked.is_empty() => match private::load(device, handles, &mlps) {
                Ok(Some(path)) => return Ok(Some(Ane::Private(Box::new(path)))),
                Ok(None) => {}
                Err(why) => eprintln!("PIE_ANE: the private path failed ({why}); trying CoreML"),
            },
            Ok(()) => {}
            Err(why) => eprintln!(
                "PIE_ANE: the private Neural Engine interface is unavailable ({why}); trying CoreML"
            ),
        }
        Ok(coreml::load(device, handles, &mlps, &asked)?.map(Ane::CoreMl))
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        let _ = (device, handles, trace, weights);
        Ok(None)
    }
}

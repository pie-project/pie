//! Pure Metal kernel definitions — shader entry names, argument marshalling,
//! and dispatch geometry; no IR types, no execution state. A driver `Run`
//! resolves plan ids to handles and calls these entry functions (design §8).

pub mod attn;
pub mod collective;
pub mod elemwise;
pub mod encode;
pub mod error;
pub mod icb;
pub mod layout;
pub mod linear;

/// A recording encode sink, so an entry's point selection and grid can be
/// asserted on a box with no GPU. Test-only, and never compiled into the
/// rlib a driver links.
#[cfg(test)]
pub(crate) mod probe;

pub mod sources;
pub mod tensor;
pub mod tuning;

pub use attn::{DecodePlan, PrefillPlan};
pub use encode::{
    Arg, ArgValue, Ctx, Encode, Fire, Geometry, Grid, elementwise, elementwise_rows, head_grid,
    head_group,
};
pub use error::Error;
pub use sources::{SOURCES, resolve, source};
pub use tensor::{Bank, KvPool, RaggedTensor, RecurrentPool, Tensor};
pub use tuning::{DeviceInfo, DeviceTuning};

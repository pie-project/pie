#[cfg(target_vendor = "apple")]
pub mod ane;
pub mod attn;
pub mod collective;
pub mod conv;
pub mod elemwise;
pub mod encode;
pub mod error;
pub mod gather;
pub mod ledger;
pub mod linear;
pub mod runtime;
pub mod sampling;
pub mod ssm;

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
pub use tuning::Tuning;

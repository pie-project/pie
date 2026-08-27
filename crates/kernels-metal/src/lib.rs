//! Pure Metal kernel definitions — shader entry names, argument marshalling,
//! and dispatch geometry; no IR types, no execution state. A driver `Run`
//! resolves plan ids to handles and calls these entry functions (design §8).

pub mod attn;
pub mod collective;
pub mod elemwise;
pub mod encode;
pub mod layout;
pub mod linear;
pub mod sources;
pub mod tensor;

pub use attn::{DecodePlan, PrefillPlan};
pub use encode::{
    Arg, ArgValue, Ctx, Encode, Fire, Geometry, Grid, elementwise, elementwise_rows, head_grid,
    head_group,
};
pub use kernels::KernelError;
pub use sources::{SOURCES, resolve, source};
pub use tensor::{KvPool, RaggedTensor, RecurrentPool, Tensor};

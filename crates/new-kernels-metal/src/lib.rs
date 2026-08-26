//! Pure Metal kernel definitions — shader entry names, argument marshalling,
//! and dispatch geometry; no IR types, no execution state. A driver `Run`
//! resolves plan ids to handles and calls these entry functions (design §8).

pub mod attn;
pub mod dist;
pub mod encode;
pub mod gemm;
pub mod layout;
pub mod mlp;
pub mod moe;
pub mod norm;
pub mod quant;
pub mod rope;
pub mod ssm;
pub mod tensor;

pub use attn::{DecodePlan, PrefillPlan, gate, index, mla, pool};
pub use encode::{
    Arg, ArgValue, Ctx, Encode, Fire, Geometry, Grid, elementwise, elementwise_rows, head_grid,
    head_group,
};
pub use new_kernels::KernelError;
pub use norm::hc;
pub use tensor::{KvPool, RaggedTensor, RecurrentPool, Tensor};

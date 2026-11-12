pub mod adapter;
pub mod api;
pub mod bench;
pub mod blob;
pub mod boot;
pub mod chip;
pub mod device;
mod dispatch;
pub mod dit;
pub mod emit;
mod error;
pub mod exec;
pub mod guest;
pub mod handle;
pub mod inputs;
pub mod lower;
pub mod mask;
pub mod pack;
pub mod pjrt;
pub mod program;
pub mod readout;
pub mod rows;
pub mod rs;
pub mod run;
pub mod serve;
pub mod settle;
pub mod store;
pub mod trace;
pub mod weights;
pub mod window;

pub use api::{ContractFor, DeviceBoot, Xla};
pub use boot::open;
pub use error::{Fault, Result, kernel};
pub use serve::{Boot, FireCost, Fired, Lane, Media, Seated, Shell};

/// The fusions this backend implements: the routed MLP on one sort
/// (`kernels_xla::linear::moe::experts`) and a dense GeGLU's gate-up with its
/// GLU (`kernels_xla::linear::glu_gemm`), their parts run one by one where
/// the kernel declines them.
pub static FUSIONS: std::sync::LazyLock<Vec<poem_compiler::fuse::Rule>> =
    std::sync::LazyLock::new(|| {
        vec![
            poem_compiler::fuse::rules::moe_experts(),
            poem_compiler::fuse::rules::matmul_glu(&[poem_ir::GluAct::GegluTanh]),
        ]
    });

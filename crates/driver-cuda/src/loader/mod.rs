//! Loading: the checkpoint's bytes onto the device, through the loader's plan.
//!
//! The division the C++ tree did not have. `model-loader` compiles a contract
//! into a `LoadPlan` and `model_loader::executor::host` runs it; this module is
//! only the two things a driver alone can state — what its device is
//! ([`plan`]), and where the arena lives ([`arena`]).

pub mod arena;
pub mod stage;
pub mod plan;

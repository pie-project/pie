//! Loading: the checkpoint's bytes onto the device, through the loader's plan.
//!
//! The division the C++ tree did not have. `model-loader` compiles a contract
//! into a `LoadPlan`, `model_loader::executor::host` runs it, and
//! `model_loader::executor::cuda` is the arena it runs into; this module is
//! only what a driver alone can state — what its device is ([`plan`]), and
//! where the memory comes from ([`stage`]).
//!
//! [`arena`](model_loader::executor::cuda) used to be here too, with the
//! pinned staging and the three transform kernels. It was never
//! device-SPECIFIC in the way it looked: the executor decided every operand
//! and every offset, and what lived here was a table of which dtype pair each
//! kernel covered — a load-time decision on the wrong side of the boundary,
//! and one no second consumer could reach.

pub mod weight_view;
pub mod stage;
pub mod plan;

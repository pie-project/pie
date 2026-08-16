//! Loading the checkpoint's bytes onto the device, through the loader's plan.

#[cfg(feature = "abi")]
pub mod plan;
#[cfg(feature = "abi")]
pub mod stage;
pub mod weight_view;

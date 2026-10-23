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

/// The fused kernels this backend ships; the compiler forms only these.
pub const FUSED: [&str; 0] = [];

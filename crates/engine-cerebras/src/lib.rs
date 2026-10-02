//! The Cerebras engine.
//!
//! The host machinery (serving, windows, weights, pools, programs) follows
//! engine-xla's, which itself follows engine-wgpu's. The device half is
//! [`device`]: roots live on the host, a fire is traced into one CSL program
//! (`kernels-cerebras`), compiled with `cslc` and run through the SDK
//! binding in [`sdk`] by a `fabric-run` process per fire.

pub mod adapter;
pub mod api;
pub mod bench;
pub mod blob;
pub mod boot;
pub mod device;
mod dispatch;
pub mod dit;
mod error;
pub mod exec;
pub mod guest;
pub mod inputs;
pub mod mask;
pub mod ports;
pub mod program;
pub mod readout;
pub mod rows;
pub mod rs;
pub mod run;
pub mod sdk;
pub mod serve;
pub mod settle;
pub mod store;
pub mod trace;
pub mod weights;
pub mod window;

pub use api::{Cerebras, ContractFor, DeviceBoot};
pub use boot::open;
pub use error::{Fault, Result, kernel};
pub use serve::{Boot, FireCost, Fired, Lane, Media, Seated, Shell};

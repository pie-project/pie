//! PTIR on CUDA: the device half of the channel plane.
//!
//! A PTIR program is compiled decoding logic delivered as
//! [`PieProgramDesc`](driver_api::local::PieProgramDesc). Backend-agnostic work
//! lives in [`driver_pipeline`]; this holds the CUDA-symbol parts: NVRTC compilation,
//! module/cubin caching, control kernels, launch, and runtime.

pub mod cache;
pub mod channel;
pub mod compile;
pub mod params;
pub mod run;
pub mod runtime;
pub mod session;

pub use cache::{Disk, disk_key};
pub use channel::{ChannelShape, Cursors, Rings, native_cell_bytes};
pub use compile::{CompileError, FailureKind, Module};
pub use params::{CudaOpParams, params_bytes};
pub use run::{Args, Control, INTRINSIC_SLOTS, MAX_RING, Prepared, launch_control};
pub use runtime::{Compiled, Programs, Region, Runtime, Stage, Target};

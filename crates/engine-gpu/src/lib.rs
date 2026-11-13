//! What engine-vulkan and engine-wgpu share. Everything here is generic
//! over a [`Backend`]: the API half (buffers, frames, kernel encoding) stays
//! in each backend crate, which instantiates these types with its own.

pub mod adapter;
pub mod api;
pub mod arena;
pub mod backend;
pub mod blob;
pub mod chunks;
pub mod decoded;
mod dispatch;
pub mod error;
pub mod experts;
pub mod guest;
pub mod handles;
pub mod inputs;
pub mod kernel_profile;
pub mod mask;
#[cfg(not(target_arch = "wasm32"))]
pub mod pread;
pub mod probe;
pub mod program;
pub mod rs;
pub mod run;
pub mod scratch;
pub mod serve;
pub mod settle;
pub mod store;
pub mod tuning;
pub mod weight_store;
pub mod weights;
pub mod window;

pub use backend::{Backend, Buffer, Memory};
pub use error::{Fault, Result, kernel};
pub use handles::{Binding, Handles, NIL};
pub use settle::{Airborne, Arms, Done};

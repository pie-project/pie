//! What engine-vulkan and engine-wgpu share. Everything here is generic
//! over a [`Backend`]: the API half (buffers, frames, kernel encoding) stays
//! in each backend crate, which instantiates these types with its own.

pub mod decoded;
pub mod error;
pub mod mask;
pub mod settle;
pub mod store;

pub use error::{Fault, Result, kernel};
pub use settle::{Airborne, Arms, Done};

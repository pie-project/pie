//! What engine-vulkan and engine-wgpu share. Everything here is generic
//! over a [`Backend`]: the API half (buffers, frames, kernel encoding) stays
//! in each backend crate, which instantiates these types with its own.

pub mod adapter;
pub mod arena;
pub mod backend;
pub mod blob;
pub mod chunks;
pub mod decoded;
pub mod error;
pub mod handles;
pub mod mask;
pub mod rs;
pub mod run;
pub mod settle;
pub mod store;
pub mod weight_store;
pub mod weights;
pub mod window;

pub use backend::{Backend, Buffer, Memory};
pub use error::{Fault, Result, kernel};
pub use handles::{Binding, Handles, NIL};
pub use settle::{Airborne, Arms, Done};

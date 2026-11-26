//! What engine-vulkan and engine-wgpu share. Everything here is generic
//! over a [`Backend`]: the API half (buffers, frames, kernel encoding) stays
//! in each backend crate, which instantiates these types with its own.

pub mod adapter;
pub mod host_memory;
pub mod api;
pub mod arena;
pub mod backend;
pub mod beam;
pub mod books;
pub mod chunks;
pub mod clock;
pub mod decoded;
pub mod dense;
mod dispatch;
pub mod error;
pub mod experts;
pub mod guest;
pub mod handles;
pub mod inputs;
pub mod kernel_profile;
pub mod mask;
pub mod named;
pub mod naming;
#[cfg(not(target_arch = "wasm32"))]
pub mod pread;
pub mod probe;
pub mod program;
pub mod rates;
pub mod ring;
pub mod region;
pub mod rs;
pub mod run;
pub mod scratch;
pub mod serve;
pub mod settle;
pub mod store;
pub mod stored;
pub mod submit;
pub mod check;
pub mod top;
pub mod tuning;
pub mod weight_store;
pub mod weights;
pub mod window;

pub use backend::{Backend, Buffer, Memory};
pub use books::{Books, Owner};
pub use error::{Fault, Result, kernel, storage_buffers, workgroups};
pub use handles::{Binding, Handles, NIL};
pub use settle::{Airborne, Arms, Done};

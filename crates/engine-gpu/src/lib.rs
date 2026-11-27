//! What engine-vulkan and engine-wgpu share. Everything here is generic
//! over a [`Backend`]: the API half (buffers, frames, kernel encoding) stays
//! in each backend crate, which instantiates these types with its own.

/// The clock this crate reads: std's, or on wasm32, where std's panics, the
/// page's (`web_std`), as engine-wgpu's and the runtime's are.
#[cfg(not(target_arch = "wasm32"))]
pub type Instant = std::time::Instant;
#[cfg(target_arch = "wasm32")]
pub type Instant = web_std::time::Instant;

pub mod adapter;
pub mod api;
pub mod arena;
pub mod backend;
pub mod beam;
pub mod books;
pub mod check;
pub mod chunks;
pub mod clock;
pub mod decoded;
pub mod dense;
mod dispatch;
pub mod error;
pub mod experts;
pub mod guest;
pub mod handles;
pub mod host_memory;
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
pub mod rs;
pub mod run;
pub mod scratch;
pub mod serve;
pub mod settle;
pub mod store;
pub mod stored;
pub mod submit;
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

/// In the tab, a step of a fire's landing said by name (`what`, at `at`),
/// the first 600: where a landing that never tells its step stops. The
/// browser's engine counts its own waits beside these. Elsewhere nothing.
#[allow(unused_variables)]
pub fn tab_step(what: &'static str, at: &'static str) {
    #[cfg(target_arch = "wasm32")]
    {
        static SAID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = SAID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if n < 600 {
            tracing::info!(target: "pie_tab_wait", n, what, at, "a landing's step");
        }
    }
}

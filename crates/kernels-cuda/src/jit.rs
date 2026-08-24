pub mod abi;
mod arg;

#[cfg(feature = "_cuda")]
pub mod cache;
mod ctx;
#[cfg(feature = "_cuda")]
mod error;
#[cfg(feature = "_cuda")]
mod launch;

#[cfg(feature = "_cuda")]
pub(crate) mod device;
#[cfg(feature = "_cuda")]
pub mod nvrtc;
mod root;

pub mod pinned;

pub mod value;

pub mod warm;

pub use abi::{Abi, ByValue, Layout, fp8_kind};
pub use ctx::{Ctx, Cuda, Launch};
#[cfg(feature = "_cuda")]
pub use error::Error;
pub use pinned::PinnedBytes;
pub use root::{Headers, Root, Toolchain};
pub use value::ArgValue;

#[must_use]
pub fn aligned16(p: *const core::ffi::c_void) -> bool {
    p.addr() & 15 == 0
}

pub use kernels::jit::symbol;

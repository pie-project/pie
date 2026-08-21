
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
pub mod nvrtc;
mod root;
#[cfg(feature = "_cuda")]
pub(crate) mod device;

pub mod pinned;

pub mod value;

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

#[must_use]
pub fn symbol(name: &str) -> &'static str {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    static NAMES: OnceLock<Mutex<HashMap<String, &'static str>>> = OnceLock::new();
    let mut map = NAMES
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(found) = map.get(name) {
        return found;
    }

    let fresh: &'static str = Box::leak(name.to_owned().into_boxed_str());
    map.insert(name.to_owned(), fresh);
    fresh
}

pub type Routine = kernels::routine::Routine<Cuda>;

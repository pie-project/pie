pub mod abi;

#[cfg(feature = "cuda")]
pub mod cache;
mod ctx;
#[cfg(feature = "cuda")]
mod launch;

#[cfg(feature = "cuda")]
pub(crate) mod device;
#[cfg(feature = "cuda")]
pub mod nvrtc;
mod root;

pub use abi::{Arg, ArgValue};
pub use ctx::{Ctx, Fire, Launch, Pad, Slabs};
pub use root::{Headers, Root, Toolchain};

use crate::error::Error;

#[must_use]
pub const fn aligned16(addr: u64) -> bool {
    addr & 15 == 0
}

#[must_use]
pub fn symbol(name: &str) -> &'static str {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    static INTERNED: OnceLock<Mutex<HashMap<String, &'static str>>> = OnceLock::new();
    let mut map = INTERNED
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(found) = map.get(name) {
        return found;
    }
    let leaked: &'static str = Box::leak(name.to_owned().into_boxed_str());
    map.insert(name.to_owned(), leaked);
    leaked
}

pub(crate) fn refuse(op: &'static str, detail: impl Into<String>) -> Error {
    Error::Backend {
        op,
        detail: detail.into(),
    }
}

pub(crate) fn nonzero(op: &'static str, axis: &'static str, v: u32) -> Result<u32, Error> {
    if v == 0 {
        return Err(refuse(op, format!("`{axis}` is zero")));
    }
    Ok(v)
}

pub(crate) fn stated(op: &'static str, v: u32) -> Result<i32, Error> {
    i32::try_from(v).map_err(|_| refuse(op, format!("{v} does not fit the kernel's int")))
}

pub(crate) fn count(op: &'static str, axis: &'static str, v: u32) -> Result<i32, Error> {
    stated(op, nonzero(op, axis, v)?)
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn runtimeless(op: &'static str) -> Error {
    Error::Backend {
        op,
        detail: "this build carries no CUDA runtime: enable `cuda`".into(),
    }
}

#[cfg(feature = "cuda")]
#[derive(Clone, Debug)]
pub(crate) enum Fault {
    Device {
        call: &'static str,
        code: i32,
    },
    Compile {
        unit: &'static str,
        log: String,
    },
    Unwarmed {
        name: &'static str,
        have: usize,
        need: usize,
    },
    Exhausted {
        name: &'static str,
        have: usize,
        need: usize,
        free: usize,
    },
}

#[cfg(feature = "cuda")]
impl core::fmt::Display for Fault {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Device { call, code } => write!(f, "`{call}` answered {code}"),
            Self::Compile { unit, log } => write!(f, "`{unit}` would not compile: {log}"),
            Self::Unwarmed { name, have, need } => write!(
                f,
                "the `{name}` scratch holds {have} bytes and this capture needs {need}; \
                 growing it mid-capture would poison the graph — warm it with an eager \
                 fire before capturing"
            ),
            Self::Exhausted {
                name,
                have,
                need,
                free,
            } => write!(
                f,
                "`cudaMalloc` answered cudaErrorMemoryAllocation growing the `{name}` scratch \
                 from {have} to {need} bytes, with {free} bytes free on the device: this \
                 fire's workspace lies outside the fitted pool, so the load left too little \
                 beside the weights, the cache rows and the graph bodies for it"
            ),
        }
    }
}

#[cfg(feature = "cuda")]
impl Fault {
    #[must_use]
    pub(crate) fn at(self, op: &'static str) -> Error {
        Error::Backend {
            op,
            detail: self.to_string(),
        }
    }
}

macro_rules! dtype_dispatch {
    ($op:expr, $dtype:expr, { $($stamped:ident => $arm:expr),+ $(,)? }) => {
        match $dtype {
            $(::dtype::Dtype::$stamped => $arm,)+
            other => {
                return Err(crate::error::Error::DtypeUnsupported {
                    op: $op,
                    dtype: other,
                });
            }
        }
    };
}

pub(crate) use dtype_dispatch;

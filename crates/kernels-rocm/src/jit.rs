pub mod abi;

#[cfg(feature = "rocm")]
pub mod cache;
mod ctx;
mod kernel;
#[cfg(feature = "rocm")]
mod launch;

#[cfg(feature = "rocm")]
pub(crate) mod device;
/// Without the HIP half there is no slab to take: a walk sizes nothing.
#[cfg(not(feature = "rocm"))]
pub(crate) mod device {
    pub(crate) fn sizing_reset() {}

    pub(crate) fn sizing_take() -> crate::sizing::Footprint {
        crate::sizing::Footprint::default()
    }
}
#[cfg(feature = "rocm")]
pub mod hiprtc;
mod root;

pub use abi::{Arg, ArgValue};
pub use ctx::{Ctx, Fire, Launch, PLACED_ALIGN, Pad, SlabKey, Slabs, TICKETS, kernels_strict, strict_kernels};
pub(crate) use kernel::generated;
pub use kernel::{Kernel, kernel};
pub use root::Root;

use crate::error::Error;

#[must_use]
pub const fn aligned16(addr: u64) -> bool {
    addr & 15 == 0
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

#[cfg(not(feature = "rocm"))]
pub(crate) fn runtimeless(op: &'static str) -> Error {
    Error::Backend {
        op,
        detail: "this build carries no CUDA runtime: enable `cuda`".into(),
    }
}

#[cfg(feature = "rocm")]
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
    /// A take past the slabs laid for its arena: a fire no walk sized.
    Unplaced {
        name: &'static str,
        have: usize,
        need: usize,
    },
}

#[cfg(feature = "rocm")]
impl core::fmt::Display for Fault {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Device { call, code } => write!(f, "`{call}` answered {code}"),
            Self::Compile { unit, log } => write!(f, "`{unit}` would not compile: {log}"),
            Self::Unplaced { name, have, need } => write!(
                f,
                "the `{name}` scratch is laid at {have} bytes and this launch needs {need}: \
                 slabs are laid ahead from a size-only walk of the fires they serve, and \
                 none grows, so this fire is one no walk sized"
            ),
        }
    }
}

#[cfg(feature = "rocm")]
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

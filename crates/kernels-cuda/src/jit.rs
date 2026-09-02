//! The launch plane: an entry names a jit unit and instantiation ([`Fire`]),
//! marshals arguments ([`ArgValue`]), and hands both to [`Ctx`], which
//! compiles the unit on first use, caches it, and enqueues the launch.
//! Enqueue only, never sync: `Ok` means the launch is queued, not that it ran.

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

/// A device address on a 16-byte boundary — the vectorised paths' gate.
#[must_use]
pub const fn aligned16(addr: u64) -> bool {
    addr & 15 == 0
}

/// Interns a composed instantiation so it can live in [`Fire`]'s
/// `&'static str`. Names are few and re-composed every fire, so the leak
/// is bounded.
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

/// A launch this backend cannot enqueue: degenerate/overflowing geometry, or
/// an extent no instantiation is stamped for. Shape agreement is checked at
/// trace time, not here.
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

/// An extent stated to a kernel that reads it as `int`.
pub(crate) fn stated(op: &'static str, v: u32) -> Result<i32, Error> {
    i32::try_from(v).map_err(|_| refuse(op, format!("{v} does not fit the kernel's int")))
}

/// [`nonzero`] then [`stated`]: a count that must exist and fit the
/// kernel's `int`.
pub(crate) fn count(op: &'static str, axis: &'static str, v: u32) -> Result<i32, Error> {
    stated(op, nonzero(op, axis, v)?)
}

/// The answer of a build with no CUDA runtime compiled in.
#[cfg(not(feature = "cuda"))]
pub(crate) fn runtimeless(op: &'static str) -> Error {
    Error::Backend {
        op,
        detail: "this build carries no CUDA runtime: enable `cuda`".into(),
    }
}

/// A raw driver/runtime refusal or a failed compile. [`Ctx::fire`] folds
/// this into [`Error::Backend`] with the op's name.
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
    /// A scratch slab asked to grow mid-capture: allocation is host work the
    /// capture refuses (see [`Ctx::scratch`]). The old block retires rather
    /// than frees, so baked addresses stay valid.
    Unwarmed {
        name: &'static str,
        have: usize,
        need: usize,
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

/// Matches on the handle's dtype; any dtype not listed returns
/// [`Error::DtypeUnsupported`] from the enclosing function.
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

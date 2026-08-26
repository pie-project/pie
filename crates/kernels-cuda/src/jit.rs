//! The launch plane: how a kernel entry talks to the device.
//!
//! An entry never touches cudarc. It names a jit unit and an instantiation
//! ([`Fire`]), marshals arguments ([`ArgValue`]), and hands both to the
//! [`Ctx`] — the stream a driver `Run` wraps — which compiles the unit on
//! first use, caches the module, and enqueues the launch. Enqueue only,
//! never sync (decision #15): a returned `Ok` means the launch is on the
//! stream, not that it ran.
//!
//! Where the old plane monomorphized `<T: Scalar>`, entries here match on
//! the handle's runtime dtype with [`dtype_dispatch!`] — the arm's value is
//! usually the `::pie::` element spelling an instantiation is stamped with.

pub mod abi;

#[cfg(feature = "_cuda")]
pub mod cache;
mod ctx;
#[cfg(feature = "_cuda")]
mod launch;

#[cfg(feature = "_cuda")]
pub(crate) mod device;
#[cfg(feature = "_cuda")]
pub mod nvrtc;
mod root;

pub use abi::{Arg, ArgValue};
pub use ctx::{Ctx, Fire, Launch};
pub use root::{Headers, Root, Toolchain};

use kernels::KernelError;

/// A device address on a 16-byte boundary — the vectorised paths' gate.
#[must_use]
pub const fn aligned16(addr: u64) -> bool {
    addr & 15 == 0
}

/// Interns a composed instantiation, so a runtime-built symbol can live in
/// [`Fire`]'s `&'static str` currency. Names are few (one per element per
/// point) and re-composed every fire, so the leak is bounded and the map
/// earns its keep.
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

/// A launch this backend cannot enqueue: degenerate or overflowing geometry,
/// an extent no instantiation is stamped for. Cross-operand *shape
/// agreement* is never reported this way — that is the trace-time
/// validator's guarantee, restated as `debug_assert!` at the entries.
pub(crate) fn refuse(op: &'static str, detail: impl Into<String>) -> KernelError {
    KernelError::Backend {
        op,
        detail: detail.into(),
    }
}

pub(crate) fn nonzero(op: &'static str, axis: &'static str, v: u32) -> Result<u32, KernelError> {
    if v == 0 {
        return Err(refuse(op, format!("`{axis}` is zero")));
    }
    Ok(v)
}

/// An extent stated to a kernel that reads it as `int`.
pub(crate) fn stated(op: &'static str, v: u32) -> Result<i32, KernelError> {
    i32::try_from(v).map_err(|_| refuse(op, format!("{v} does not fit the kernel's int")))
}

/// [`nonzero`] then [`stated`]: the one-word spelling of the entries'
/// commonest prologue — a count that must exist and must fit the kernel's
/// `int`.
pub(crate) fn count(op: &'static str, axis: &'static str, v: u32) -> Result<i32, KernelError> {
    stated(op, nonzero(op, axis, v)?)
}

/// The answer of a build with no CUDA runtime compiled in.
#[cfg(not(feature = "_cuda"))]
pub(crate) fn runtimeless(op: &'static str) -> KernelError {
    KernelError::Backend {
        op,
        detail: "this build carries no CUDA runtime: enable `cuda-12` or `cuda-13`".into(),
    }
}

/// What the runtime layers report upward: a raw driver/runtime refusal or a
/// compile with a log. Entries never see this type — [`Ctx::fire`] folds it
/// into [`KernelError::Backend`] with the op's name, which is the one piece
/// of attribution the deep layers do not have.
#[cfg(feature = "_cuda")]
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
    /// A scratch slab asked to grow mid-capture. Growth frees and
    /// reallocates, which would poison the graph, so the fire is refused —
    /// the driver's warm-before-capture pass is the fix (see
    /// [`Ctx::scratch`]).
    Unwarmed {
        name: &'static str,
        have: usize,
        need: usize,
    },
}

#[cfg(feature = "_cuda")]
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

#[cfg(feature = "_cuda")]
impl Fault {
    #[must_use]
    pub(crate) fn at(self, op: &'static str) -> KernelError {
        KernelError::Backend {
            op,
            detail: self.to_string(),
        }
    }
}

/// The runtime successor of `<T: Scalar>` monomorphization: name the dtypes
/// this entry is stamped for and get the named arm's value; any other dtype
/// **returns** [`KernelError::DtypeUnsupported`] from the enclosing function.
macro_rules! dtype_dispatch {
    ($op:expr, $dtype:expr, { $($stamped:ident => $arm:expr),+ $(,)? }) => {
        match $dtype {
            $(::model_ir::Dtype::$stamped => $arm,)+
            other => {
                return Err(::kernels::KernelError::DtypeUnsupported {
                    op: $op,
                    dtype: other,
                });
            }
        }
    };
}

pub(crate) use dtype_dispatch;

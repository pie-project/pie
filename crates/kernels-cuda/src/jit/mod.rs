//! The per-symbol JIT: a root, an instantiation, and one entry point.
//!
//! Compilation happens per instantiation: [`Root`] holds the device text,
//! [`Ctx`] is what a routine body launches through, [`ArgValue`] is one bound
//! argument, [`Abi`] says how a crossing type spells itself in C++, and
//! [`nvrtc`]/[`Error`] are the compiler and its refusal type.
//!
//! `nvrtc`, `error` and `abi` live here, not in `runtime`/`x`, because each
//! previously pointed the wrong way across a module boundary.

pub mod abi;
mod arg;
/// One instantiation, compiled at most once per process, and its entry point.
#[cfg(feature = "_cuda")]
pub mod cache;
mod ctx;
#[cfg(feature = "_cuda")]
mod error;
#[cfg(feature = "_cuda")]
mod launch;
/// NVRTC, as this crate asks it: one root, one instantiation, one cubin.
#[cfg(feature = "_cuda")]
pub mod nvrtc;
mod root;
#[cfg(feature = "_cuda")]
pub(crate) mod device;
/// The page-locked host buffer a capturable H2D must read from.
pub mod pinned;
/// One bound argument's value, as the launch ABI carries it.
pub mod value;

pub use abi::{Abi, ByValue, Layout, fp8_kind};
pub use ctx::{Ctx, Cuda, Launch};
#[cfg(feature = "_cuda")]
pub use error::Error;
pub use pinned::PinnedBytes;
pub use root::{Headers, Root, Toolchain};
pub use value::ArgValue;

/// Is `p` 16-byte aligned — the test a body makes before it picks a vector
/// width. A host-side check, made before any launch (see
/// `kernels/gemm/gemv.cuh:91`); shared here, feature-free, so any routine
/// body may call it from either build.
#[must_use]
pub fn aligned16(p: *const core::ffi::c_void) -> bool {
    p.addr() & 15 == 0
}

/// One routine's row, from its `fn` and nothing else.
///
/// The backend's three-line wrapper over [`kernels::routine!`], with [`Cuda`]
/// filled in so a declaration names only the `fn`:
///
/// ```ignore
/// pub static ROUTINES: &[Routine] = &[
///     routine!(rope_bf16, in_place = &[(0, 0), (1, 1)]),
///     routine!(rope_write_kv_bf16, whole),
/// ];
/// ```
#[macro_export]
macro_rules! routine {
    // The generic form, split so the BASE reaches `Derivation`. `$name` is
    // the trace symbol and carries no column; the column belongs to the `fn`
    // the turbofish instantiates.
    ($name:ident = $base:ident ::<$($g:ty),* $(,)?> $(, $($fact:tt)*)?) => {
        ::kernels::routine!(
            $crate::jit::Cuda,
            $name = $base::<$($g),*>,
            derived = <$base as ::kernels::Derivation>::DERIVED
            $(, $($fact)*)?
        )
    };
    // The one row whose launcher carries no `#[routine]` and should not:
    // `attn::qkv_decode_fused_dispatch` is an inner leg whose caller has
    // already bound every operand, so a column here would resolve end to end
    // and claim a binding no statement made. Spelled at the call site because
    // a silently empty column is how that would go unnoticed.
    ($body:ident, uncolumned $(, $($fact:tt)*)?) => {
        ::kernels::routine!($crate::jit::Cuda, $body $(, $($fact)*)?)
    };
    ($body:ident $(, $($fact:tt)*)?) => {
        ::kernels::routine!(
            $crate::jit::Cuda,
            $body,
            derived = <$body as ::kernels::Derivation>::DERIVED
            $(, $($fact)*)?
        )
    };
}

/// The same, for a symbol the DRIVER fires by path — see
/// [`kernels::driver_bound!`] for the distinction.
///
/// It sits beside `routine!` in the same `ROUTINES` list on purpose: which
/// symbols a statement can bind vs. the driver binds is a per-symbol fact,
/// stated where the symbol is, not a property of a separate list.
#[macro_export]
macro_rules! driver_bound {
    ($body:ident $(, $($fact:tt)*)?) => {
        ::kernels::driver_bound!($crate::jit::Cuda, $body $(, $($fact)*)?)
    };
}

/// One routine, in this backend's instantiation of the machinery.
pub type Routine = kernels::routine::Routine<Cuda>;

/// One family's routines, and the namespace its trace symbols sit in.
///
/// A `Routine`'s name is its `fn`'s name; the namespace is what turns that
/// into a full trace symbol like `rope::rope_bf16`.
///
/// `namespace` is DERIVED, not stated: [`Family::new`] takes the first path
/// segment after the crate root out of `module_path!()`, so the Rust path
/// and the trace symbol are always the same string:
///
/// ```text
/// kernels_cuda::rope::rope_bf16   <->   "rope::rope_bf16"
/// ```
///
/// "First segment" rather than "last" so a family may have submodules
/// without minting namespaces: `attn::fa2` and `attn::xqa` are separate
/// `Family` values under one `attn` namespace.
pub struct Family {
    /// What a trace prefixes this family's symbols with.
    ///
    /// Private, because it is [`Family::new`]'s answer and not a caller's
    /// choice; [`Family::namespace`] reads it.
    namespace: &'static str,
    /// The routines, in declaration order.
    pub routines: &'static [Routine],
}

impl Family {
    /// The family declared in `module_path`, holding `routines`.
    ///
    /// Call it as [`family!`](crate::family), which supplies `module_path!()`
    /// so that no site can pass a path other than its own.
    ///
    /// # Panics
    ///
    /// If `module_path` names the crate root itself, which has no family
    /// segment to take. A `Family` at the root would be a family with no
    /// namespace, and every symbol it offered would be a bare routine name
    /// that no trace can state — so this refuses at compile time (the call is
    /// a `const`) rather than producing 25 unstatable symbols.
    #[must_use]
    pub const fn new(module_path: &'static str, routines: &'static [Routine]) -> Self {
        Self { namespace: segment_after_crate(module_path), routines }
    }

    /// What a trace prefixes this family's symbols with.
    #[must_use]
    pub const fn namespace(&self) -> &'static str {
        self.namespace
    }

    /// The routine a trace symbol names, if this family declares it.
    #[must_use]
    pub fn routine(&self, symbol: &str) -> Option<&'static Routine> {
        let tail = symbol.strip_prefix(self.namespace)?.strip_prefix("::")?;
        self.routines.iter().find(|r| r.name == tail)
    }

    /// One routine's trace symbol.
    #[must_use]
    pub fn symbol(&self, routine: &Routine) -> String {
        format!("{}::{}", self.namespace, routine.name)
    }
}

/// The first path segment after the crate root, out of a `module_path!()`.
///
/// `kernels_cuda::attn::fa2` -> `attn`. Written as a byte scan over
/// `as_bytes()` because this runs in a `const` initialiser, where
/// `str::split` and `Iterator::nth` are not available; the two `unsafe`
/// blocks reconstruct a `str` from a range of a range this function itself
/// computed, and the bytes at both ends are ASCII `:` or the string's own
/// bounds, so no multi-byte sequence can be cut.
const fn segment_after_crate(module_path: &'static str) -> &'static str {
    let bytes = module_path.as_bytes();
    let mut start = 0;
    while start + 1 < bytes.len() {
        if bytes[start] == b':' && bytes[start + 1] == b':' {
            start += 2;
            break;
        }
        start += 1;
    }
    assert!(start > 0 && start < bytes.len(), "a Family at the crate root has no namespace");
    let mut end = start;
    while end + 1 < bytes.len() {
        if bytes[end] == b':' && bytes[end + 1] == b':' {
            break;
        }
        end += 1;
    }
    if end + 1 == bytes.len() {
        end = bytes.len();
    }
    // SAFETY: `start..end` lies on `::` boundaries or on the string's ends,
    // and `:` is ASCII, so the range begins and ends on a char boundary.
    unsafe { core::str::from_utf8_unchecked(core::slice::from_raw_parts(bytes.as_ptr().add(start), end - start)) }
}

/// One family, named by the module it is declared in.
///
/// ```ignore
/// pub static FAMILY: Family = family!(ROUTINES);
/// ```
///
/// The whole of the macro is `module_path!()`, and it exists so that no
/// declaration site can pass a path other than its own — a `Family::new`
/// called with a string literal would be the stated field back again, one
/// indirection further away.
#[macro_export]
macro_rules! family {
    ($routines:expr) => {
        $crate::jit::Family::new(::core::module_path!(), $routines)
    };
}

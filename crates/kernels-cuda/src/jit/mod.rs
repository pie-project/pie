//! The per-symbol JIT: a root, an instantiation, and one entry point.
//!
//! What changed from `runtime::cache` is the granularity. A UNIT carried a row
//! list because the whole set of instantiations had to be enumerated before
//! anything could be compiled, and one fire of one FA2 symbol compiled its
//! unit's ten rows. Compilation is per instantiation now: the enumeration has
//! no reader left, so it is not data any more.
//!
//! The pieces:
//!
//! * [`Root`] — device text and what a compile of it needs.
//! * [`Ctx`] — what a routine body launches through.
//! * [`ArgValue`] — one bound argument, feature-free, because a routine body
//!   is feature-free.
//! * [`Cuda`] — the marker carrying those last two to the `kernels`
//!   machinery.
//! * [`Abi`] — how a crossing type spells itself in C++ and which
//!   [`ArgValue`] it marshals into.
//! * [`nvrtc`] and [`Error`] — the compiler and what it refuses with.
//!
//! # Why `nvrtc`, `error` and `abi` are HERE
//!
//! All three arrived from somewhere else, and each move deleted a
//! cross-directory reference that pointed the wrong way.
//!
//! `nvrtc.rs` and `error.rs` were `runtime/`'s, and the split was a CYCLE:
//! `cache.rs` and `launch.rs` reached up for `runtime::{Error, nvrtc}` while
//! `nvrtc.rs` reached back down for `jit::{Toolchain, cache::bind_context}`.
//! Nothing else was left in that directory — `runtime::{Launch, Ungeometric}`
//! and `runtime::Stream` had zero readers, and `Launch` had no `cooperative`
//! field, so it SHADOWED this module's [`Launch`] for anyone who reached by
//! name. The whole of `runtime` was two live files and one hazard.
//!
//! `abi.rs` was `x/`'s, and the reference ran the same way for a different
//! reason: [`arg_via_abi!`](crate::arg_via_abi) is machinery, it expands to
//! the `Abi` path, and machinery reaching up into the FAMILIES for the CUDA
//! type vocabulary is an inversion. `Abi` is not a family — it is how a
//! crossing type spells itself in C++ and marshals into an [`ArgValue`], both
//! of which are this directory's words. The file names no family path at all,
//! so the move introduced nothing and turned an inversion into a
//! same-directory reference.

pub mod abi;
mod arg;
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
pub mod value;

pub use abi::{Abi, ByValue, Layout, fp8_kind};
pub use ctx::{Ctx, Cuda, Launch};
#[cfg(feature = "_cuda")]
pub use error::Error;
pub use root::{Headers, Root, Toolchain};
pub use value::ArgValue;

/// Is `p` 16-byte aligned — the test a body makes before it picks a vector
/// width.
///
/// # Why this is one function and was four
///
/// It is a HOST test, made before any launch, and the device text says so
/// itself: `kernels/gemm/gemv.cuh:91` records `<cstdint>` being dropped
/// because *"its one use was `std::uintptr_t` in `aligned16`, which is a HOST
/// alignment test made before any launch and is Rust now."*
///
/// It was shared once, as `fire::hand::aligned16` and then `x::fire::
/// aligned16`. Both modules were deleted, and four families — `norm`, `moe`,
/// `layout` and `gemm::gemv` — each grew a private copy. Three spellings of
/// one predicate resulted (`p.addr() & 15 == 0` twice, `(p as usize) % 16 ==
/// 0` once), each with a doc comment citing a different address that no
/// longer resolves.
///
/// The reason the copies gave for not importing was *"`x::fire` is a
/// `_cuda`-only module and a routine body is feature-free"*. That reason was
/// true and is answered rather than repeated: this module's ungated half is
/// feature-free, so a body may call this from either build.
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
    ($name:ident = $body:expr $(, $($fact:tt)*)?) => {
        ::kernels::routine!($crate::jit::Cuda, $name = $body $(, $($fact)*)?)
    };
    ($body:ident $(, $($fact:tt)*)?) => {
        ::kernels::routine!($crate::jit::Cuda, $body $(, $($fact)*)?)
    };
}

/// The same, for a symbol the DRIVER fires by path — see
/// [`kernels::driver_bound!`] for what the distinction is and why it is not a
/// weaker `routine!`.
///
/// It sits beside `routine!` in the same `ROUTINES` list on purpose. Which
/// symbols a statement can bind and which the driver binds is a per-symbol
/// fact, stated where the symbol is; a second LIST would make it a property
/// of where a line was written, which is how `not_yet_crossed.rs` came to
/// carry columns nothing derived.
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
/// A `Routine`'s name is its `fn`'s name, which is what makes the table
/// underivable-from-anything-else; a trace names `rope::rope_bf16`. The
/// namespace is the difference.
///
/// # It is DERIVED, and that is what dissolving `x` bought
///
/// `namespace` was a stated field: `Family { namespace: "rope", routines }`,
/// written by hand once per family. It was the last column of a row in this
/// crate that a human typed, and typing it is what made it able to drift —
/// nothing compared the string to anything, so a family whose module was
/// renamed, or copied to seed a new one, kept the old prefix and every symbol
/// it offered was silently the wrong string.
///
/// It could not be derived while the families lived under `x`, because
/// `module_path!()` would have answered `kernels_cuda::rope` and the
/// useful segment was neither the first nor the last. With them at the crate
/// root the derivation is exactly [`Family::new`]'s one line — **the first
/// path segment after the crate root** — and the Rust path and the trace
/// symbol become the same string:
///
/// ```text
/// kernels_cuda::rope::rope_bf16   <->   "rope::rope_bf16"
/// ```
///
/// The rule is "first segment" rather than "last" so that a family may have
/// submodules without minting namespaces: `attn::fa2` and `attn::xqa` are
/// separate `Family` values under one `attn` namespace, which is what their
/// symbols have always said and what the Rust used to contradict.
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
